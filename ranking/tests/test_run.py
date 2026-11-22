import hashlib
import json
import multiprocessing
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from uplift_ranking import data, deep, forest, learners, run, tuning, versions
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
    """What has been committed and pushed, held in memory; the tests of run.Repository use git itself."""

    def __init__(self):
        self.files, self.is_clean, self.is_pushed = {}, True, True
        self.history = {run.REGISTRATION_COMMIT, "c-head"}
        self.registration = {commit: run.REGISTRATION.read_bytes() for commit in (run.REGISTRATION_COMMIT, "HEAD")}

    def head(self):
        return "c-head"

    def blob(self, commit, path):
        return self.registration.get(commit) if Path(path) == run.REGISTRATION else None

    def contains(self, commit):
        return commit in self.history

    def clean(self, *written):
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
    patch.setattr(forest, "FINAL_TREES", 12)
    d = hillstrom_like()
    patch.setattr(data, "load", lambda name: d)
    records = tmp / "records" / "trials.jsonl"
    sizes = {TEST: len(d.test), VALIDATION: len(d.valid["visit"].y), TRAINING: len(d.train["visit"].y)}
    data.check_role_counts("hillstrom", np.repeat(list(sizes), list(sizes.values())), None, records.parent)  # as the first search does
    run.search("hillstrom", FakeRepository(), records)
    configs, disagreements = run.select("hillstrom", records)
    chosen = {"configs": configs, "records commit": "c-head", "disagreements": disagreements}
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
    run.search("hillstrom", FakeRepository(), records)  # resumes: nothing pending
    assert len(tuning.read(records)) == len(rows)


def test_refits_reproduce_the_manifest(staged):
    _, tmp, records, manifest, first, _ = staged
    assert (records.parent / "hillstrom-manifest.json").read_text() == first
    assert len(manifest["files"]) == 2 + 9 * 5 and manifest["R*"]["4"] in run.NON_DEEP
    assert all(c["fingerprint matches"] for c in manifest["comparisons"].values())
    dry = json.loads((tmp / "results" / "hillstrom-dry-run.json").read_text())
    assert dry["versions"] == versions.check() and dry["resources"]["cache GB"] > 0


def test_selection_reads_committed_files_only(staged):
    _, _, records, *_ = staged
    roles = records.parent / "hillstrom-roles.json"
    repo = FakeRepository()
    repo.commit(roles)
    with pytest.raises(run.Stop, match="trials.jsonl differs from HEAD"):
        run.choose("hillstrom", repo, records)
    repo.commit(records)
    del repo.files[str(roles.resolve())]  # Section 2.1: the block does not take role counts that no commit holds
    with pytest.raises(run.Stop, match="hillstrom-roles.json differs from HEAD"):
        run.choose("hillstrom", repo, records)
    repo.commit(roles)
    repo.is_pushed = False
    with pytest.raises(run.Stop, match="remote"):
        run.choose("hillstrom", repo, records)
    repo.is_pushed = True
    configs, disagreements = run.select("hillstrom", records)
    assert run.choose("hillstrom", repo, records) == {"configs": configs, "records commit": "c-head", "disagreements": disagreements}


def test_the_search_and_the_block_refuse_uncommitted_files(staged):
    _, _, records, *_ = staged
    repo = FakeRepository()
    repo.commit(records, records.parent / "hillstrom-roles.json")
    repo.is_clean = False
    for stage in (run.search, run.choose):
        with pytest.raises(run.Stop, match="uncommitted files"):
            stage("hillstrom", repo, records)


def rewritten(staged, tmp_path, change):
    """A copy of the staged trial records after `change` has edited the list of records."""
    rows = tuning.read(staged[2])
    change(rows)
    (tmp_path / "trials.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return tmp_path / "trials.jsonl", rows


def test_the_search_stops_on_records_of_other_nuisances(staged, tmp_path):
    def change(rows):
        for r in rows:
            r["fingerprint"] = "of another machine"

    records, rows = rewritten(staged, tmp_path, change)
    with pytest.raises(tuning.FingerprintMismatch, match="unit 4"):
        run.search("hillstrom", FakeRepository(), records)
    assert tuning.read(records) == rows


def test_selection_stops_on_records_under_two_fingerprints(staged, tmp_path):
    records, _ = rewritten(staged, tmp_path, lambda rows: rows[-1].update(fingerprint="of another machine"))
    with pytest.raises(tuning.FingerprintMismatch, match="unit 4"):
        run.select("hillstrom", records)


def test_selection_lists_disagreeing_duplicates(staged, tmp_path):
    records, rows = rewritten(staged, tmp_path, lambda rows: rows.append(rows[0] | {"r_loss": rows[0]["r_loss"] + 1}))
    assert run.select("hillstrom", records)[1] == [{"unit": 4, "model": rows[0]["model"], "config": rows[0]["config"],
                                                     "R-loss": rows[0]["r_loss"] + 1, "first R-loss": rows[0]["r_loss"]}]


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
        run.block("hillstrom", {"configs": configs, "records commit": "c-head", "disagreements": []}, records, tmp_path / "frozen",
                  tmp_path / "results", 20)
    assert not (tmp_path / "results").exists()


def read_setup(staged):
    d, tmp, records, *_ = staged
    repo = FakeRepository()
    repo.commit(records.parent / "hillstrom-manifest.json", records.parent / "hillstrom-roles.json", tmp / "results" / "hillstrom-dry-run.json")
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
    "registration changed when added": "registration at 9e8678d",
    "dirty": "not clean",
    "no dry run": "hillstrom-dry-run.json is not committed",
    "no role counts": "hillstrom-roles.json is not committed",
    "frozen": "differs from the manifest",
    "no manifest": "no committed manifest",
    "not pushed": "remote",
}


@pytest.mark.parametrize("problem", REFUSALS)
def test_the_entry_point_refuses(staged, problem):
    d, tmp, records_dir, repo = read_setup(staged)
    frozen = tmp / "frozen"
    if problem == "registration":
        repo.history.discard(run.REGISTRATION_COMMIT)
    elif problem == "registration missing at HEAD":
        del repo.registration["HEAD"]
    elif problem == "registration changed at HEAD":
        repo.registration["HEAD"] += b"\n"
    elif problem == "registration changed when added":
        repo.registration[run.REGISTRATION_COMMIT] = repo.registration["HEAD"].replace(b"20261115", b"20261116", 1)
    elif problem == "dirty":
        repo.is_clean = False
    elif problem == "frozen":
        frozen = tmp / "frozen-copy"
        frozen.mkdir(exist_ok=True)
        for f in (tmp / "frozen").glob("*.pkl"):
            (frozen / f.name).write_bytes(f.read_bytes())
        (frozen / "4 S seed 0.pkl").write_bytes(b"changed")
    elif problem.startswith("no "):
        name = {"no dry run": "hillstrom-dry-run.json", "no role counts": "hillstrom-roles.json", "no manifest": "hillstrom-manifest.json"}
        del repo.files[next(path for path in repo.files if path.endswith(name[problem]))]
    if problem == "not pushed":
        run.prepare("hillstrom", repo, frozen, records_dir, tmp / "results")
        repo.commit(records_dir / "reads.jsonl")
        repo.is_pushed = False
        with pytest.raises(run.EntryRefused, match=REFUSALS[problem]):
            run.go("hillstrom", repo, frozen, records_dir, tmp / "results", replicates=20)
        return
    with pytest.raises(run.EntryRefused, match=REFUSALS[problem]):
        run.prepare("hillstrom", repo, frozen, records_dir, tmp / "results")


def test_the_repository_holds_the_registration_commit():
    """On the repository itself, with its whole history (the CI checkout fetches it)."""
    repo = run.Repository()
    assert repo.contains(run.REGISTRATION_COMMIT)
    added = repo.git("diff-tree", "--root", "--no-commit-id", "--name-status", "-r", run.REGISTRATION_COMMIT).stdout
    assert added == "A\tranking/PREREGISTRATION.md\n"
    for text in (repo.blob(run.REGISTRATION_COMMIT, run.REGISTRATION), repo.blob("HEAD", run.REGISTRATION), run.REGISTRATION.read_bytes()):
        assert hashlib.sha256(text).hexdigest() == run.REGISTRATION_SHA256


def git(cwd, *args):
    identity = {f"GIT_{who}_{what}": value for who in ("AUTHOR", "COMMITTER") for what, value in (("NAME", "test"), ("EMAIL", "test@example.invalid"))}
    subprocess.run(["git", *args], cwd=cwd, env=os.environ | identity, check=True, capture_output=True)


def test_a_history_without_the_registration_commit_is_refused(tmp_path):
    """A shallow clone holds the registered text at HEAD, added there for all it can show."""
    git(tmp_path, "clone", "--depth", "1", f"file://{run.ROOT.parent}", "clone")
    root = tmp_path / "clone" / "ranking"
    assert hashlib.sha256((root / "PREREGISTRATION.md").read_bytes()).hexdigest() == run.REGISTRATION_SHA256
    with pytest.raises(run.EntryRefused, match="not an ancestor"):
        run.entry_checks("hillstrom", run.Repository(root), tmp_path, root / "records", root / "results")


@pytest.fixture
def clone(tmp_path):
    """The ranking directory of a clone whose one commit is on the main branch of its remote."""
    git(tmp_path, "init", "--bare", "--initial-branch=main", "remote.git")
    git(tmp_path, "clone", "remote.git", "clone")
    root = tmp_path / "clone" / "ranking"
    (root / "records").mkdir(parents=True)
    (root / "code.py").write_text("")
    git(root, "add", ".")
    git(root, "commit", "-m", "first")
    git(root, "push", "origin", "HEAD:main")
    return root


def test_pushed_asks_the_remote(clone):
    repo = run.Repository(clone)
    assert repo.pushed()
    (clone / "code.py").write_text("x = 1\n")
    git(clone, "commit", "-am", "second")
    git(clone, "update-ref", "refs/remotes/origin/main", "HEAD")  # the local copy of the remote's branch, moved without a push
    assert not repo.pushed()
    git(clone, "push", "origin", "HEAD:main")
    assert repo.pushed()


def test_clean_leaves_aside_the_named_directories_only(clone):
    repo = run.Repository(clone)
    assert repo.clean()
    (clone / "records" / "trials.jsonl").write_text("{}\n")
    assert repo.clean(clone / "records") and not repo.clean() and not repo.clean(clone / "results")
    (clone / "code.py").write_text("x = 1\n")
    assert not repo.clean(clone / "records")


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
    manifest["disagreements"] = [{"unit": 4, "model": "S", "config": 0, "R-loss": 0.2, "first R-loss": 0.1}]
    repo.files[str(path.resolve())] = json.dumps(manifest, indent=1, sort_keys=True) + "\n"
    entry = run.prepare("hillstrom", repo, tmp / "frozen", records_dir, tmp / "results")
    repo.commit(records_dir / "reads.jsonl")
    repo.head = lambda: "c-later"
    repo.history.add("c-later")
    # The container is lost after the read entry; a new one reruns the block at a later HEAD.
    chosen = run.choose("hillstrom", repo, records_dir / "trials.jsonl")
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
    chosen = {"configs": run.select("hillstrom", records_dir / "trials.jsonl")[0], "records commit": "c-head", "disagreements": []}

    def load(name):
        raise AssertionError("the data were loaded before the versions were checked")

    monkeypatch.setattr(data, "load", load)
    monkeypatch.setattr(versions, "installed", lambda: dict(versions.PINNED, torch="2.15.0"))
    stages = {"search": lambda: run.search("hillstrom", repo, tmp_path / "trials.jsonl"),
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
    scores, mu = run.frozen_predictions(4, tmp / "frozen", d.valid["visit"].x, "validation")
    again = run.evaluate(4, scores, mu, d.valid["visit"], manifest["R*"]["4"], 20, labels=True)
    dry = json.loads((tmp / "results" / "hillstrom-dry-run.json").read_text())["4"]
    assert json.loads(json.dumps(again["values"])) == dry["values"]
    assert json.loads(json.dumps(again["H2"])) == dry["H2"]


def test_thawed_deep_models_predict_under_the_registered_torch_settings(staged):
    _, tmp, *_ = staged
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(False)
    run.thaw((tmp / "frozen" / "4 EFIN seed 0.pkl").read_bytes())
    assert torch.get_num_threads() == deep.THREADS and torch.are_deterministic_algorithms_enabled()


def test_final_forests_take_the_final_size_and_on_criteo_the_drawn_rows(staged, monkeypatch):
    _, tmp, *_ = staged
    assert run.thaw((tmp / "frozen" / "4 causal forest seed 0.pkl").read_bytes()).n_estimators == forest.FINAL_TREES != forest.SEARCH_TREES
    train, seen = Split(np.arange(20.0).reshape(10, 2), np.arange(10), np.arange(10)), {}
    monkeypatch.setattr(forest, "criteo_rows", lambda: np.array([1, 4, 7]))
    monkeypatch.setattr(forest, "fit", lambda t, v, config, seed, trees: seen.update(rows=t.y.tolist(), trees=trees))
    run.fit_final("causal forest", {}, train, None, 0, "criteo", 500)
    assert seen == {"rows": [1, 4, 7], "trees": 500}
    run.fit_final("causal forest", {}, train, None, 0, "lazada", 500)
    assert seen["rows"] == list(range(10))


def test_stand_in_rows_take_their_values_from_the_validation_rows(staged):
    valid = staged[0].valid["visit"]
    rows = run.stand_in(valid, 3000, 4)
    assert rows.x.shape == (3000, valid.x.shape[1]) and rows.x.dtype == valid.x.dtype
    assert all(set(rows.x[:, j]) <= set(valid.x[:, j]) for j in range(valid.x.shape[1]))
    assert set(zip(rows.w.tolist(), rows.y.tolist())) <= set(zip(valid.w.tolist(), valid.y.tolist()))
    assert len(np.unique(rows.x, axis=0)) > len(valid.y)  # more distinct feature vectors than rows drawn whole would give


def hold(megabytes):
    global held
    held = np.ones(megabytes * 2**20 // 8)


def test_tree_memory_counts_the_children():
    before = run.tree_memory()
    with multiprocessing.get_context("fork").Pool(1) as pool:
        pool.apply(hold, (200,))
        assert run.tree_memory() - before > 150 * 2**20
    assert before > 2**20


def test_prepare_rehearses_the_read_on_synthetic_rows_of_the_test_size(staged, monkeypatch):
    d, tmp, records_dir, repo = read_setup(staged)
    seen, real = [], run.evaluate

    def evaluate(unit, scores, mu, rows, *args, **kwargs):
        seen.append((len(rows.y), data.OPEN.get(), (tmp / "frozen" / run.TOKEN).exists()))
        return real(unit, scores, mu, rows, *args, **kwargs)

    monkeypatch.setattr(run, "evaluate", evaluate)
    entry = run.prepare("hillstrom", repo, tmp / "frozen", records_dir, tmp / "results")
    assert seen == [(len(d.test), False, False)]  # the test rows stay sealed and no token exists yet
    assert 0 < entry["rehearsal GiB"] < run.MEMORY_LIMIT / 2**30


def test_a_rehearsal_above_the_memory_limit_stops_before_the_entry(staged, monkeypatch):
    _, tmp, records_dir, repo = read_setup(staged)
    monkeypatch.setattr(run, "MEMORY_LIMIT", 2**20)
    with pytest.raises(run.Stop, match="rehearsed on .* synthetic rows took .* GiB"):
        run.prepare("hillstrom", repo, tmp / "frozen", records_dir, tmp / "results")
    assert not (tmp / "frozen" / run.TOKEN).exists() and not (records_dir / "reads.jsonl").exists()
