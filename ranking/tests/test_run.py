import hashlib
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pytest

from uplift_ranking import data, forest, learners, run, tuning, versions
from uplift_ranking.learners import Split
from uplift_ranking.roles import TEST, TRAINING, VALIDATION, groups, roles

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def hillstrom_like(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.c_[rng.integers(1, 13, n), rng.integers(0, 7, n), np.round(rng.uniform(30, 900, n), 0),
              rng.integers(0, 2, (n, 2)), rng.integers(0, 3, n), rng.integers(0, 2, n), rng.integers(0, 3, n)].astype(np.float32)
    arm = rng.integers(0, 3, n)
    p = 0.12 + 0.05 * (arm == 1) * (x[:, 3] > 0) + 0.04 * (arm == 2) * (x[:, 4] > 0)
    y = (rng.random(n) < p).astype(np.int64)
    role = roles(x, "hillstrom")
    part = lambda r: {"visit": Split(x[role == r], arm[role == r], y[role == r])}
    test = role == TEST
    return data.Data("hillstrom", part(TRAINING), part(VALIDATION), data.Sealed(part(TEST), groups(x[test])))


class Constant:
    """A model that predicts one value for every row."""

    def __init__(self, value):
        self.value = value

    def predict(self, x):
        return np.full(len(x), self.value)


class FakeRepository:
    """What has been committed and pushed, held in memory: no git commits in tests."""

    def __init__(self):
        self.files, self.is_clean, self.is_pushed, self.history = {}, True, True, {"c-registration", "c-head"}
        self.registration = {commit: run.REGISTRATION.read_bytes() for commit in ("c-registration", "HEAD")}

    def head(self):
        return "c-head"

    def added(self, path):
        return "c-registration"

    def blob(self, commit, path):
        return self.registration.get(commit) if Path(path) == run.REGISTRATION else None

    def is_ancestor(self, a, b):
        return a in self.history

    def contains(self, commit):
        return commit in self.history

    def clean(self):
        return self.is_clean

    def committed(self, path):
        return self.files.get(str(Path(path).resolve()), "")

    def pushed(self):
        return self.is_pushed

    def commit(self, *paths):
        for p in paths:
            self.files[str(Path(p).resolve())] = Path(p).read_text()


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("run")
    patch = pytest.MonkeyPatch()
    patch.setattr(tuning, "BUDGET", 2)
    patch.setattr(forest, "SEARCH_TREES", 8)
    patch.setattr(forest, "FINAL_TREES", 8)
    d = hillstrom_like()
    patch.setattr(data, "load", lambda name: d)
    records = tmp / "records" / "trials.jsonl"
    run.search("hillstrom", records, commit="c-head")
    configs, disagreements = run.select("hillstrom", records)
    chosen = {"configs": configs, "records commit": "c-head"}
    manifest = run.block("hillstrom", chosen, records, tmp / "frozen", tmp / "results", replicates=20)
    first = (records.parent / "hillstrom-manifest.json").read_text()
    run.block("hillstrom", chosen, records, tmp / "frozen-again", tmp / "results", replicates=20)
    yield d, tmp, records, manifest, first, disagreements
    patch.undo()


def test_search_writes_one_record_per_trial(staged):
    d, tmp, records, *_ = staged
    rows = tuning.read(records)
    assert len(rows) == 5 * 2 + 9 + 3 * 2 and staged[5] == []
    assert all(tuple(r) == tuning.FIELDS for r in rows)
    run.search("hillstrom", records, commit="c-head")  # resumes: nothing pending
    assert len(tuning.read(records)) == len(rows)


def test_refits_reproduce_the_manifest(staged):
    _, tmp, records, manifest, first, _ = staged
    assert (records.parent / "hillstrom-manifest.json").read_text() == first
    assert len(manifest["files"]) == 2 + 9 * 5 and manifest["R*"]["4"] in run.NON_DEEP
    assert all(c["fingerprint matches"] for c in manifest["comparisons"].values())
    dry = json.loads((tmp / "results" / "hillstrom-dry-run.json").read_text())
    assert dry["versions"] == versions.check() and dry["resources"]["cache GB"] > 0


def test_selection_reads_committed_records_only(staged):
    _, _, records, *_ = staged
    repo = FakeRepository()
    with pytest.raises(run.Stop, match="differ from HEAD"):
        run.choose("hillstrom", repo, records)
    repo.commit(records)
    repo.is_pushed = False
    with pytest.raises(run.Stop, match="remote"):
        run.choose("hillstrom", repo, records)
    repo.is_pushed = True
    chosen, _ = run.choose("hillstrom", repo, records)
    assert chosen == {"configs": run.select("hillstrom", records)[0], "records commit": "c-head"}


@pytest.mark.parametrize("problem", ["S seed 0", "m-hat", "mu1", "versions", "disk"])
def test_the_block_stops(staged, problem, monkeypatch, tmp_path):
    d, _, staged_records, *_ = staged
    records = tmp_path / "records" / "trials.jsonl"  # a block that does not stop writes its manifest here
    records.parent.mkdir()
    records.write_text(staged_records.read_text())
    configs, _ = run.select("hillstrom", records)
    if problem == "S seed 0":
        monkeypatch.setattr(run, "effect", lambda model, fitted, x: np.full(len(x), np.nan))
    elif problem == "m-hat":
        fitted = learners.nuisances
        monkeypatch.setattr(learners, "nuisances", lambda *a, **k: (lambda f, m, e: (f, np.r_[m[:-1], np.inf], e))(*fitted(*a, **k)))
    elif problem == "mu1":
        fitted = learners.outcome_models
        monkeypatch.setattr(learners, "outcome_models", lambda *a: fitted(*a) | {"mu1": Constant(np.nan)})
    elif problem == "versions":
        monkeypatch.setattr(versions, "installed", lambda: dict(versions.PINNED, **{"scikit-learn": "1.9.2"}))
    else:
        monkeypatch.setattr(run, "DISK_LIMIT", 0.0)
    error = versions.VersionMismatch if problem == "versions" else run.Stop
    match = {"S seed 0": "S seed 0 on validation", "m-hat": "m-hat", "mu1": "outcome model mu1 on validation",
             "versions": "scikit-learn", "disk": "cache directory"}[problem]
    with pytest.raises(error, match=match):
        run.block("hillstrom", {"configs": configs, "records commit": "c-head"}, records, tmp_path / "frozen", tmp_path / "results", 20)
    assert not (tmp_path / "results").exists()


def read_setup(staged):
    d, tmp, records, *_ = staged
    repo = FakeRepository()
    repo.commit(records.parent / "hillstrom-manifest.json")
    (records.parent / "reads.jsonl").unlink(missing_ok=True)
    (tmp / "frozen" / run.TOKEN).unlink(missing_ok=True)
    (tmp / "results" / "hillstrom.json").unlink(missing_ok=True)
    return d, tmp, records.parent, repo


def test_arguments():
    assert run.arguments(["read", "lazada", "go"]).step == "go"
    assert run.arguments(["search", "criteo", "--model", "causal forest", "--model", "S"]).model == ["causal forest", "S"]
    for argv in (["read", "hillstrom"], ["block", "hillstrom", "go"], ["block", "hillstrom", "--model", "S"], ["search", "criteo", "--model", "forest"]):
        with pytest.raises(SystemExit):
            run.arguments(argv)


def test_the_entry_point_reads_once_and_writes_aggregates_only(staged):
    d, tmp, records_dir, repo = read_setup(staged)
    with pytest.raises(PermissionError):
        d.test.open()
    with pytest.raises(run.EntryRefused, match="no committed read entry"):
        run.go("hillstrom", repo, tmp / "frozen", records_dir, tmp / "results", replicates=20)
    entry = run.prepare("hillstrom", repo, tmp / "frozen", records_dir, tmp / "results")
    assert entry["rerun"] is False and entry["commit"] == "c-head"
    repo.commit(records_dir / "reads.jsonl")
    out = run.go("hillstrom", repo, tmp / "frozen", records_dir, tmp / "results", replicates=20)
    with pytest.raises(PermissionError):
        d.test.open()
    assert sorted(p.name for p in (tmp / "results").iterdir()) == ["hillstrom-dry-run.json", "hillstrom.json"]
    unit = out[4]
    assert unit["H2"]["label"] in ("mostly indistinguishable", "mixed", "mostly distinguishable")
    assert len(unit["H6 H7"]) == 9 and set(unit["H3"]["results"]) >= {"TARNet", "DESCN", "EFIN"}

    def longest(o):
        return max([len(o)] + [longest(v) for v in o]) if isinstance(o, list) else max([0] + [longest(v) for v in o.values()]) if isinstance(o, dict) else 0

    assert longest(json.loads((tmp / "results" / "hillstrom.json").read_text())) <= 72
    with pytest.raises(run.EntryRefused, match="token"):
        run.go("hillstrom", repo, tmp / "frozen", records_dir, tmp / "results", replicates=20)
    with pytest.raises(run.EntryRefused, match="has a read entry"):
        run.prepare("hillstrom", repo, tmp / "frozen", records_dir, tmp / "results")
    repo.files[str((run.ROOT / "AMENDMENTS.md").resolve())] = f"Interrupted read of hillstrom at {entry['time']}."
    assert run.prepare("hillstrom", repo, tmp / "frozen", records_dir, tmp / "results")["rerun"] is True
    repo.commit(tmp / "results" / "hillstrom.json")
    with pytest.raises(run.EntryRefused, match="has a read entry"):
        run.prepare("hillstrom", repo, tmp / "frozen", records_dir, tmp / "results")


REFUSALS = {
    "registration": "not an ancestor",
    "registration missing at HEAD": "registration at HEAD",
    "registration changed at HEAD": "registration at HEAD",
    "registration changed when added": "registration at c-registration",
    "dirty": "not clean",
    "frozen": "differs from the manifest",
    "no manifest": "no committed manifest",
    "not pushed": "remote",
}


@pytest.mark.parametrize("problem", REFUSALS)
def test_the_entry_point_refuses(staged, problem):
    d, tmp, records_dir, repo = read_setup(staged)
    frozen = tmp / "frozen"
    if problem == "registration":
        repo.history.discard("c-registration")
    elif problem == "registration missing at HEAD":
        del repo.registration["HEAD"]
    elif problem == "registration changed at HEAD":
        repo.registration["HEAD"] += b"\n"
    elif problem == "registration changed when added":
        repo.registration["c-registration"] = repo.registration["c-registration"].replace(b"20261115", b"20261116", 1)
    elif problem == "dirty":
        repo.is_clean = False
    elif problem == "frozen":
        frozen = tmp / "frozen-copy"
        frozen.mkdir(exist_ok=True)
        for f in (tmp / "frozen").glob("*.pkl"):
            (frozen / f.name).write_bytes(f.read_bytes())
        (frozen / "4 S seed 0.pkl").write_bytes(b"changed")
    elif problem == "no manifest":
        repo.files.clear()
    if problem == "not pushed":
        run.prepare("hillstrom", repo, frozen, records_dir, tmp / "results")
        repo.commit(records_dir / "reads.jsonl")
        repo.is_pushed = False
        with pytest.raises(run.EntryRefused, match=REFUSALS[problem]):
            run.go("hillstrom", repo, frozen, records_dir, tmp / "results", replicates=20)
        return
    with pytest.raises(run.EntryRefused, match=REFUSALS[problem]):
        run.prepare("hillstrom", repo, frozen, records_dir, tmp / "results")


def test_the_repository_holds_the_registered_text():
    repo = run.Repository()
    added = repo.added(run.REGISTRATION)
    assert added and repo.is_ancestor(added, repo.head())
    for text in (repo.blob(added, run.REGISTRATION), repo.blob("HEAD", run.REGISTRATION), run.REGISTRATION.read_bytes()):
        assert hashlib.sha256(text).hexdigest() == run.REGISTRATION_SHA256


def test_a_lost_container_needs_a_new_entry(staged):
    d, tmp, records_dir, repo = read_setup(staged)
    entry = run.prepare("hillstrom", repo, tmp / "frozen", records_dir, tmp / "results")
    repo.commit(records_dir / "reads.jsonl")
    run.go("hillstrom", repo, tmp / "frozen", records_dir, tmp / "results", replicates=20)
    # The container is lost before the results are committed; a new one refits the same frozen files.
    with pytest.raises(run.EntryRefused, match="token"):
        run.go("hillstrom", repo, tmp / "frozen", records_dir, tmp / "results", replicates=20)
    with pytest.raises(run.EntryRefused, match="AMENDMENTS"):
        run.prepare("hillstrom", repo, tmp / "frozen", records_dir, tmp / "results")
    repo.files[str((run.ROOT / "AMENDMENTS.md").resolve())] = f"Interrupted read of hillstrom at {entry['time']}."
    assert run.prepare("hillstrom", repo, tmp / "frozen", records_dir, tmp / "results")["rerun"] is True
    repo.commit(records_dir / "reads.jsonl")
    assert run.go("hillstrom", repo, tmp / "frozen", records_dir, tmp / "results", replicates=20)["versions"] == versions.check()


def amend(repo, *entries):
    repo.files[str((run.ROOT / "AMENDMENTS.md").resolve())] = "\n".join(f"Interrupted read of hillstrom at {e['time']}." for e in entries)


def second_read(staged):
    """A read entry, then refits that do not reproduce one frozen file: a new committed manifest (Section 10)."""
    d, tmp, records_dir, repo = read_setup(staged)
    first = run.prepare("hillstrom", repo, tmp / "frozen", records_dir, tmp / "results")
    repo.commit(records_dir / "reads.jsonl")
    frozen = tmp / "frozen-refit"
    shutil.copytree(tmp / "frozen", frozen, dirs_exist_ok=True)
    (frozen / "4 S seed 0.pkl").write_bytes((frozen / "4 S seed 1.pkl").read_bytes())
    path = records_dir / "hillstrom-manifest.json"
    manifest = json.loads(repo.committed(path))
    manifest["files"]["4 S seed 0"] = hashlib.sha256((frozen / "4 S seed 0.pkl").read_bytes()).hexdigest()
    repo.files[str(path.resolve())] = json.dumps(manifest, indent=1, sort_keys=True) + "\n"
    return tmp, records_dir, repo, frozen, first


def test_a_changed_manifest_is_a_second_read(staged):
    tmp, records_dir, repo, frozen, first = second_read(staged)
    assert first["second read"] is False
    with pytest.raises(run.EntryRefused, match="AMENDMENTS"):
        run.prepare("hillstrom", repo, frozen, records_dir, tmp / "results")
    amend(repo, first)
    second = run.prepare("hillstrom", repo, frozen, records_dir, tmp / "results")
    assert second["rerun"] and second["second read"]
    repo.commit(records_dir / "reads.jsonl")
    amend(repo, first, second)  # the second read is interrupted in turn: its rerun is still a second read
    assert run.prepare("hillstrom", repo, frozen, records_dir, tmp / "results")["second read"] is True


def test_the_results_say_whether_they_come_from_a_second_read(staged):
    tmp, records_dir, repo, frozen, first = second_read(staged)
    amend(repo, first)
    run.prepare("hillstrom", repo, frozen, records_dir, tmp / "results")
    repo.commit(records_dir / "reads.jsonl")
    out = run.go("hillstrom", repo, frozen, records_dir, tmp / "results", replicates=20)
    assert out["second read"] is True and json.loads((tmp / "results" / "hillstrom.json").read_text())["second read"] is True


def test_a_block_after_the_read_entry_reruns_the_committed_manifest(staged, tmp_path):
    d, tmp, records_dir, repo = read_setup(staged)
    path = records_dir / "hillstrom-manifest.json"
    manifest = json.loads(repo.committed(path))
    manifest["R*"]["4"] = next(m for m in run.NON_DEEP if m != manifest["R*"]["4"])  # an R* the refits would not pick
    repo.files[str(path.resolve())] = json.dumps(manifest, indent=1, sort_keys=True) + "\n"
    entry = run.prepare("hillstrom", repo, tmp / "frozen", records_dir, tmp / "results")
    repo.commit(records_dir / "reads.jsonl")
    repo.head = lambda: "c-later"
    repo.history.add("c-later")
    # The container is lost after the read entry; a new one reruns the block at a later HEAD.
    chosen, _ = run.choose("hillstrom", repo, records_dir / "trials.jsonl")
    records = tmp_path / "records" / "trials.jsonl"
    records.parent.mkdir()
    shutil.copy(records_dir / "trials.jsonl", records)
    run.block("hillstrom", chosen, records, tmp_path / "frozen", tmp_path / "results", 20)
    assert (records.parent / "hillstrom-manifest.json").read_text() == repo.committed(path)
    assert not (tmp_path / "results").exists()  # the committed dry run stands
    amend(repo, entry)
    again = run.prepare("hillstrom", repo, tmp_path / "frozen", records_dir, tmp / "results")
    assert again["rerun"] and not again["second read"]
    repo.commit(records_dir / "reads.jsonl")
    assert run.go("hillstrom", repo, tmp_path / "frozen", records_dir, tmp / "results", replicates=20)[4]["R*"] == manifest["R*"]["4"]


@pytest.mark.parametrize("rows", ["validation", "test"])
def test_the_read_stops_on_a_non_finite_prediction(staged, rows, monkeypatch):
    d, tmp, records_dir, repo = read_setup(staged)
    run.prepare("hillstrom", repo, tmp / "frozen", records_dir, tmp / "results")
    repo.commit(records_dir / "reads.jsonl")
    valid_x = d.valid["visit"].x
    real = run.effect
    nan_on = lambda x: (x is valid_x) == (rows == "validation")
    monkeypatch.setattr(run, "effect", lambda model, f, x: np.full(len(x), np.nan) if model == "EFIN" and nan_on(x) else real(model, f, x))
    with pytest.raises(run.Stop, match=f"EFIN seed 0 on {rows} rows"):
        run.go("hillstrom", repo, tmp / "frozen", records_dir, tmp / "results", replicates=20)
    assert (tmp / "frozen" / run.TOKEN).exists() == (rows == "validation")  # the token goes only when the test rows open
    assert not (tmp / "results" / "hillstrom.json").exists()


def test_the_read_stops_on_a_non_finite_nuisance(staged, monkeypatch):
    d, tmp, records_dir, repo = read_setup(staged)
    run.prepare("hillstrom", repo, tmp / "frozen", records_dir, tmp / "results")
    repo.commit(records_dir / "reads.jsonl")
    real = learners.Fitted.propensity
    monkeypatch.setattr(learners.Fitted, "propensity",
                        lambda self, x: np.full(len(x), np.nan) if self.kind == "nuisances" else real(self, x))
    with pytest.raises(run.Stop, match="unit 4 e on validation rows"):
        run.go("hillstrom", repo, tmp / "frozen", records_dir, tmp / "results", replicates=20)
    assert (tmp / "frozen" / run.TOKEN).exists()


def test_a_stale_token_is_refused(staged):
    d, tmp, records_dir, repo = read_setup(staged)
    run.prepare("hillstrom", repo, tmp / "frozen", records_dir, tmp / "results")
    repo.commit(records_dir / "reads.jsonl")
    (tmp / "frozen" / run.TOKEN).write_text("0" * 64)  # a token that is not the last entry's
    with pytest.raises(run.EntryRefused, match="no unused token"):
        run.go("hillstrom", repo, tmp / "frozen", records_dir, tmp / "results", replicates=20)


def test_prepare_and_go_check_the_versions(staged, monkeypatch):
    d, tmp, records_dir, repo = read_setup(staged)
    other = lambda: dict(versions.PINNED, torch="2.15.0")
    with monkeypatch.context() as patch:
        patch.setattr(versions, "installed", other)
        with pytest.raises(versions.VersionMismatch, match="torch"):
            run.prepare("hillstrom", repo, tmp / "frozen", records_dir, tmp / "results")
    run.prepare("hillstrom", repo, tmp / "frozen", records_dir, tmp / "results")
    repo.commit(records_dir / "reads.jsonl")
    monkeypatch.setattr(versions, "installed", other)
    with pytest.raises(versions.VersionMismatch, match="torch"):
        run.go("hillstrom", repo, tmp / "frozen", records_dir, tmp / "results", replicates=20)


@pytest.mark.parametrize("stage", ["search", "block", "read"])
def test_each_stage_checks_the_versions_before_it_loads_the_data(staged, stage, monkeypatch, tmp_path):
    _, tmp, records_dir, repo = read_setup(staged)
    run.prepare("hillstrom", repo, tmp / "frozen", records_dir, tmp / "results")  # the read passes its entry checks
    repo.commit(records_dir / "reads.jsonl")
    chosen = {"configs": run.select("hillstrom", records_dir / "trials.jsonl")[0], "records commit": "c-head"}

    def load(name):
        raise AssertionError("the data were loaded before the versions were checked")

    monkeypatch.setattr(data, "load", load)
    monkeypatch.setattr(versions, "installed", lambda: dict(versions.PINNED, torch="2.15.0"))
    stages = {"search": lambda: run.search("hillstrom", tmp_path / "trials.jsonl"),
              "block": lambda: run.block("hillstrom", chosen, tmp_path / "trials.jsonl", tmp_path / "frozen", tmp_path / "results", 20),
              "read": lambda: run.go("hillstrom", repo, tmp / "frozen", records_dir, tmp / "results", replicates=20)}
    with pytest.raises(versions.VersionMismatch, match="torch"):
        stages[stage]()


def test_the_results_name_their_commit_and_command(staged):
    _, tmp, records_dir, repo = read_setup(staged)
    run.prepare("hillstrom", repo, tmp / "frozen", records_dir, tmp / "results")
    repo.commit(records_dir / "reads.jsonl")
    out = run.go("hillstrom", repo, tmp / "frozen", records_dir, tmp / "results", replicates=20)
    for results in (out, json.loads((tmp / "results" / "hillstrom.json").read_text())):
        assert results["commit"] == "c-head" and results["command"] == "python -m uplift_ranking.run read hillstrom go"
        assert results["second read"] is False


def test_r_star_and_thawed_models_reproduce_the_dry_run(staged):
    d, tmp, records, manifest, *_ = staged
    losses = {m: manifest["comparisons"][f"4 {m}"]["R-loss"] for m in run.NON_DEEP}
    assert manifest["R*"]["4"] == min(run.NON_DEEP, key=lambda m: (losses[m], run.NON_DEEP.index(m)))
    fits = {m: [run.thaw((tmp / "frozen" / f"4 {m} seed {s}.pkl").read_bytes()) for s in range(5)] for m in tuning.MODELS}
    outcome_models = pickle.loads((tmp / "frozen" / "4 outcome models.pkl").read_bytes())
    scores, mu = run.predictions(4, fits, outcome_models, d.valid["visit"].x, "validation")
    again = run.evaluate(4, scores, mu, d.valid["visit"], manifest["R*"]["4"], 20, labels=True)
    dry = json.loads((tmp / "results" / "hillstrom-dry-run.json").read_text())["4"]
    assert json.loads(json.dumps(again["values"])) == dry["values"]
    assert json.loads(json.dumps(again["H2"])) == dry["H2"]
