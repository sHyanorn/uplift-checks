"""Run stages (Sections 10 and 11): search, select, the final block and the single test read.

    python -m uplift_ranking.run search DATASET [--model NAME]
    python -m uplift_ranking.run block DATASET
    python -m uplift_ranking.run read DATASET prepare|go

Records, manifests and results are written under ranking/ and committed by the operator; model
files stay in the cache directory. Test rows are opened only inside `read ... go`.

Every stage checks the package versions first: nothing is downloaded, written or fitted, and no test
row is read, under other versions. Stop conditions of Section 10 raise Stop: a package version other
than the registered ones, a missing or non-finite prediction of a frozen model, a cache directory
above 25 GB, a rehearsed read above MEMORY_LIMIT, uncommitted files in the search or the block, and in
the block trial records or role counts that are not HEAD's or not on the remote's main branch.
"""

import argparse
import datetime
import hashlib
import json
import os
import pickle
import resource
import secrets
import subprocess
import threading
from pathlib import Path

import numpy as np

from . import data, deep, forest, learners, results, tuning, versions
from .learners import Split
from .roles import groups

ROOT = Path(__file__).resolve().parents[2]
REGISTRATION = ROOT / "PREREGISTRATION.md"
REGISTRATION_SHA256 = "e223e6ca8688b3caec522d1911c3f0d8b2a9e50c47cde675be0f27e89838b6b1"
REGISTRATION_COMMIT = "9e8678d0efc03ba51aaa8e2cc7718e993db5e412"  # adds the registration and nothing else
UNITS = {"criteo": {1: "visit", 2: "conversion"}, "lazada": {3: "label"}, "hillstrom": {4: "visit"}}
NON_DEEP = tuning.MODELS[:6]
DISK_LIMIT = 25e9
MEMORY_LIMIT = 12 * 2**30  # for the rehearsed read: 3 GiB under the 15 GiB of Section 11's container
REHEARSAL_REPLICATES = 16  # one for each job of the four replicate workers
TOKEN = "read token"


class Stop(RuntimeError):
    """A stop condition of Section 10, reported and resolved by an amendment before the read."""


class Repository:
    """The git checks of the entry point."""

    def __init__(self, root=ROOT):
        self.root = root

    def git(self, *args, check=True):
        return subprocess.run(["git", *args], cwd=self.root, capture_output=True, text=True, check=check)

    def head(self):
        return self.git("rev-parse", "HEAD").stdout.strip()

    def blob(self, commit, path):
        """The file's bytes at the commit, or None if it is not there."""
        out = subprocess.run(["git", "show", f"{commit}:{self.name(path)}"], cwd=self.root, capture_output=True)
        return out.stdout if out.returncode == 0 else None

    def name(self, path):
        """The path as git names it, from the top of the repository."""
        return Path(path).resolve().relative_to(self.root.resolve().parent).as_posix()

    def is_ancestor(self, a, b):
        return self.git("merge-base", "--is-ancestor", a, b, check=False).returncode == 0

    def clean(self, *written):
        """No uncommitted change, leaving aside the directories `written`, where the steps write their files."""
        aside = [f":(top,exclude){self.name(directory)}" for directory in written]
        return self.git("status", "--porcelain", "--", ":(top)", *aside).stdout == ""

    def committed(self, path):
        """The file's content at HEAD, or '' if it is not there."""
        out = self.git("show", f"HEAD:{self.name(path)}", check=False)
        return out.stdout if out.returncode == 0 else ""

    def contains(self, commit):
        return self.is_ancestor(commit, "HEAD")

    def pushed(self):
        """HEAD is on the remote's main branch as the remote has it now (Section 11), not as last fetched."""
        self.git("fetch", "origin", "main")
        return self.is_ancestor("HEAD", "FETCH_HEAD")


def two_arms(split, unit):
    """U4 takes either e-mail as treated; other units are already two-armed."""
    return Split(split.x, (split.w > 0).astype(np.int64), split.y) if unit == 4 else split


def role_counts(dataset, records_dir):
    """The file of role counts that the dataset's first search writes (Section 2.1), in a list; empty where the
    registration fixes the counts."""
    return [Path(records_dir) / f"{dataset}-roles.json"] if data.EXPECTED[dataset]["roles"] is None else []


def committed_code(repo, records):
    """Stop unless every file outside records/ and results/ is committed: a record's commit then holds the code
    that wrote it, and the next step starts from committed files."""
    if not repo.clean(Path(records).parent, ROOT / "results"):
        raise Stop("uncommitted files outside records/ and results/; commit them first")


def search(dataset, repo, records=tuning.RECORDS, models=tuning.MODELS):
    """Seed-0 trials without a record; stops if m-hat and e differ from the unit's first record."""
    versions.check()
    committed_code(repo, records)
    commit = repo.head()
    d = data.load(dataset)
    for unit, outcome in UNITS[dataset].items():
        train, valid = two_arms(d.train[outcome], unit), two_arms(d.valid[outcome], unit)
        _, m, e = learners.nuisances(train, valid, cross_fitted_e=dataset == "lazada")
        tuning.check_fingerprint(tuning.read(records), unit, tuning.fingerprint(m, e))
        for model in models:
            for index in tuning.pending(tuning.read(records), unit, model):
                tuning.append(tuning.trial(dataset, unit, model, index, train, valid, m, e, commit), records)


def select(dataset, records=tuning.RECORDS):
    """Each unit's configurations, and the duplicate records that disagree with their trial's first (Section 11).
    Stops if a unit's records were made with different m-hat and e (Section 10): their R-losses do not compare."""
    rows = tuning.read(records)
    for unit in UNITS[dataset]:
        tuning.check_fingerprint(rows, unit)
    chosen = {unit: {model: tuning.select(rows, unit, model) for model in tuning.MODELS} for unit in UNITS[dataset]}
    disagreements = [{"unit": unit, "model": model, "config": r["config"], "R-loss": r["r_loss"],
                      "first R-loss": tuning.usable(rows, unit, model)[r["config"]]["r_loss"]}
                     for unit in UNITS[dataset] for model in tuning.MODELS for r in tuning.disagreements(rows, unit, model)]
    return chosen, disagreements


def choose(dataset, repo, records=tuning.RECORDS):
    """Selection from committed files only (Sections 4 and 11): the trial records and, on Hillstrom, the role counts
    (Section 2.1) must be HEAD's, and HEAD on the remote's main branch; HEAD is then the records commit of the
    frozen set.

    Once the dataset has a committed read entry, the block reruns with the configurations, R*, records commit and
    disagreements of the committed manifest (Section 11), so that refits that reproduce the frozen files reproduce
    the manifest hash."""
    records = Path(records)
    committed_code(repo, records)
    rerun = bool(entries(repo.committed(records.parent / "reads.jsonl"), dataset))
    for path in role_counts(dataset, records.parent) + ([] if rerun else [records]):
        if not path.exists() or path.read_text() != repo.committed(path):
            raise Stop(f"{path.name} differs from HEAD's; commit and push it before the block")
    if rerun:
        manifest = json.loads(repo.committed(records.parent / f"{dataset}-manifest.json"))
        return {"configs": {int(u): c for u, c in manifest["configs"].items()}, "records commit": manifest["records commit"],
                "disagreements": manifest["disagreements"], "R*": {int(u): r for u, r in manifest["R*"].items()}}
    if not repo.pushed():
        raise Stop("HEAD is not on the main branch of the remote")
    configs, disagreements = select(dataset, records)
    return {"configs": configs, "records commit": repo.head(), "disagreements": disagreements}


def frozen_bytes(model, config, fitted):
    """A model file's bytes: pickles of LightGBM and forest objects, and of a deep model's weights as arrays."""
    if model in tuning.DEEP:
        state = {k: v.numpy() for k, v in fitted.module.state_dict().items()}
        fitted = {"model": model, "config": config, "state": state, "mean": fitted.mean, "sd": fitted.sd,
                  "epochs": fitted.epochs, "losses": fitted.losses}
    return pickle.dumps(fitted, protocol=5)


def thaw(raw):
    obj = pickle.loads(raw)
    if isinstance(obj, dict) and obj.get("model") in tuning.DEEP:
        deep.configure()  # Section 11: a thawed model predicts under the torch settings of its fit
        module = deep.MODELS[obj["model"]](len(obj["mean"]), obj["config"]["rank"])
        module.load_state_dict({k: deep.torch.from_numpy(v) for k, v in obj["state"].items()})
        return deep.Fitted(module, obj["mean"], obj["sd"], obj["epochs"], obj["losses"])
    return obj


def effect(model, fitted, x):
    return forest.effect(fitted, x) if model == "causal forest" else fitted.effect(x)


def finite(name, values, rows):
    """`values` if it holds one finite prediction per row; Stop otherwise."""
    values = np.asarray(values, dtype=float)
    if values.shape != (rows,) or not np.all(np.isfinite(values)):
        raise Stop(f"{name}: missing or non-finite predictions")
    return values


def outcome_predictions(unit, outcome_models, x, where):
    return {name: finite(f"unit {unit} outcome model {name} on {where} rows", b.predict(x), len(x))
            for name, b in outcome_models.items()}


def frozen_predictions(unit, frozen, x, where):
    """Scores of the unit's frozen models at seeds 0 to 4 and its outcome models' predictions on the rows `x`,
    with one model file in memory at a time."""
    load = lambda name: (Path(frozen) / f"{unit} {name}.pkl").read_bytes()
    scores = {model: [finite(f"unit {unit} {model} seed {s} on {where} rows",
                             effect(model, thaw(load(f"{model} seed {s}")), x), len(x)) for s in range(5)]
              for model in tuning.MODELS}
    return scores, outcome_predictions(unit, pickle.loads(load("outcome models")), x, where)


def check_nuisances(unit, m, e):
    for name, values in (("m-hat", m), ("e", e)):
        finite(f"unit {unit} {name} on validation rows", values, len(m))


def resources(frozen):
    """Peak resident memory of this process and of its largest child, and the size of the dataset's cache directory."""
    peak = {who: resource.getrusage(r).ru_maxrss * 1024 / 1e9
            for who, r in (("process", resource.RUSAGE_SELF), ("largest child", resource.RUSAGE_CHILDREN))}
    disk = sum(f.stat().st_size for f in Path(frozen).parent.rglob("*") if f.is_file())
    return {"peak memory GB": {k: round(v, 3) for k, v in peak.items()}, "cache GB": round(disk / 1e9, 3)}


def block(dataset, chosen, records=tuning.RECORDS, frozen=None, out=ROOT / "results", replicates=None):
    """Outcome models, m-hat and e, final fits at seeds 0 to 4, freeze, and the dry run on validation rows.

    Every final fit's validation predictions are checked as soon as it is fitted. When `chosen` carries R*
    (a rerun after the read entry), R* is not recomputed and the committed dry run is kept: only the fits rerun."""
    installed = versions.check()
    d = data.load(dataset)
    frozen = frozen or data.CACHE / dataset / "frozen"
    frozen.mkdir(parents=True, exist_ok=True)
    manifest = {"dataset": dataset, "records commit": chosen["records commit"], "configs": {}, "R*": {}, "files": {},
                "comparisons": {}, "disagreements": chosen["disagreements"]}
    if dataset == "criteo":
        manifest["forest rows"] = hashlib.sha256(forest.criteo_rows().tobytes()).hexdigest()
    rows = tuning.read(records)
    dry = {}
    for unit, outcome in UNITS[dataset].items():
        train, valid = two_arms(d.train[outcome], unit), two_arms(d.valid[outcome], unit)
        # 5.3: on U4 mu1 pools both e-mail arms; H6 adds arm1 and arm2 and shares mu0.
        arms = {"mu0": [0], "mu1": [1, 2]} | ({"arm1": [1], "arm2": [2]} if unit == 4 else {})
        outcome_models = learners.outcome_models(d.train[outcome], d.valid[outcome], arms)
        mu = outcome_predictions(unit, outcome_models, valid.x, "validation")
        nuisance, m, e = learners.nuisances(train, valid, cross_fitted_e=dataset == "lazada")
        check_nuisances(unit, m, e)
        save(frozen, f"{unit} outcome models", outcome_models, manifest)
        save(frozen, f"{unit} nuisances", nuisance, manifest)
        fits, scores, losses = {}, {}, {}
        for model in tuning.MODELS:
            index = chosen["configs"][unit][model]
            config = tuning.grid(model)[index]
            fits[model], scores[model] = [], []
            for seed in range(5):
                trees = forest.FINAL_TREES if model == "causal forest" else forest.SEARCH_TREES
                fitted = fit_final(model, config, train, valid, seed, dataset, trees)
                scores[model].append(finite(f"unit {unit} {model} seed {seed} on validation rows",
                                            effect(model, fitted, valid.x), len(valid.y)))
                fits[model].append(fitted)
                save(frozen, f"{unit} {model} seed {seed}", fitted, manifest, model, config)
            losses[model] = learners.r_loss(scores[model][0], valid.y, valid.w, m, e)
            record = tuning.usable(rows, unit, model).get(index)
            manifest["comparisons"][f"{unit} {model}"] = {
                "R-loss": losses[model], "record R-loss": record and record["r_loss"],
                "fingerprint matches": bool(record and record["fingerprint"] == tuning.fingerprint(m, e))}
        if "R*" in chosen:
            r_star = chosen["R*"][unit]
        else:
            r_star = min(NON_DEEP, key=lambda model: (losses[model], NON_DEEP.index(model)))
            dry[unit] = evaluate(unit, scores, mu, d.valid[outcome], r_star, replicates, labels=unit != 2)
        manifest["configs"][str(unit)], manifest["R*"][str(unit)] = chosen["configs"][unit], r_star
    path = Path(records).parent / f"{dataset}-manifest.json"
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    if "R*" in chosen:
        return manifest
    out.mkdir(parents=True, exist_ok=True)
    dry |= {"versions": installed, "resources": resources(frozen)}
    (out / f"{dataset}-dry-run.json").write_text(json.dumps(dry, indent=1) + "\n")
    return manifest


def fit_final(model, config, train, valid, seed, dataset, trees):
    if model in tuning.META:
        return learners.fit(model, train, valid, config, seed, cross_fitted_e=dataset == "lazada")
    if model == "causal forest":
        return forest.fit(train.rows(forest.criteo_rows()) if dataset == "criteo" else train, valid, config, seed, trees)
    return deep.fit(model, (train.x, train.w, train.y), (valid.x, valid.w, valid.y), seed=seed, **{"alpha": None, **config})


def save(frozen, name, obj, manifest, model=None, config=None):
    raw = frozen_bytes(model, config, obj) if model else pickle.dumps(obj, protocol=5)
    (frozen / f"{name}.pkl").write_bytes(raw)
    manifest["files"][name] = hashlib.sha256(raw).hexdigest()
    if (cache := resources(frozen)["cache GB"]) > DISK_LIMIT / 1e9:
        raise Stop(f"the cache directory holds {cache} GB, more than {DISK_LIMIT / 1e9:g} GB")


def evaluate(unit, scores, mu, rows, r_star, replicates, labels):
    """Every registered output of the unit on the given rows (validation rows in the dry run, test rows in the
    read), from the models' scores and the outcome models' predictions on those rows. The dry run (Sections 4
    and 11) computes all of them, H6 and H7 on Hillstrom included."""
    split = two_arms(rows, unit)
    resampling = groups(split.x)
    kwargs = {} if replicates is None else {"replicates": replicates}
    out = results.unit_outputs(unit, scores, split.w, split.y, mu["mu0"], mu["mu1"], resampling, r_star, labels, **kwargs)
    if unit == 4:
        v = np.c_[mu["arm1"] - mu["mu0"], mu["arm2"] - mu["mu0"]]
        three = np.c_[mu["mu0"], mu["arm1"], mu["arm2"]]
        out["H6 H7"] = results.allocation_outputs(v, rows.w, rows.y, three, scores[r_star][0], resampling, labels, **kwargs)
    return out


def outputs(dataset, frozen, manifest, rows, where, replicates):
    """Every registered output of the dataset from its frozen models on `rows` (outcome -> Split), one unit
    at a time: a unit's scores and rankings are released before the next unit's are made."""
    out = {}
    for unit, outcome in UNITS[dataset].items():
        scores, mu = frozen_predictions(unit, frozen, rows[outcome].x, where)
        out[unit] = evaluate(unit, scores, mu, rows[outcome], manifest["R*"][str(unit)], replicates, labels=unit != 2)
    return out


def tree_memory():
    """Bytes held by this process and its children: the sum of their proportional set sizes, since the forked
    replicate workers share most of their pages with this process."""
    me, kilobytes = os.getpid(), 0
    for process in Path("/proc").iterdir():
        try:
            if process.name == str(me) or (process.name.isdigit() and
                                           int((process / "stat").read_text().rsplit(")", 1)[1].split()[1]) == me):
                kilobytes += sum(int(line.split()[1]) for line in (process / "smaps_rollup").read_text().splitlines()
                                 if line.startswith("Pss:"))
        except OSError:  # a process that ended after it was listed
            pass
    return kilobytes * 1024


class PeakMemory(threading.Thread):
    """The largest `tree_memory` seen, sampled twice a second inside a `with` block."""

    def __init__(self):
        super().__init__(daemon=True)
        self.peak, self.done = 0, threading.Event()

    def run(self):
        while not self.done.wait(0.5):
            self.peak = max(self.peak, tree_memory())

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *error):
        self.done.set()
        self.join()


def stand_in(valid, rows, unit):
    """`rows` synthetic rows in place of a unit's test rows. Each feature is drawn on its own from the validation
    rows' values, so feature vectors hardly repeat and tie blocks are no larger than among test rows; (w, y)
    are drawn as pairs."""
    rng = np.random.default_rng([20261115, 7, unit])
    x = np.column_stack([rng.choice(valid.x[:, j], rows) for j in range(valid.x.shape[1])])
    pairs = rng.choice(len(valid.y), rows)
    return Split(x, valid.w[pairs], valid.y[pairs])


def rehearse(dataset, frozen, manifest, valid, rows):
    """Run the read's evaluation on synthetic rows of the test size and return its peak memory in bytes, this
    process and its workers together; Stop (Section 10) above MEMORY_LIMIT. No token exists yet."""
    synthetic = {outcome: stand_in(valid[outcome], rows, unit) for unit, outcome in UNITS[dataset].items()}
    with PeakMemory() as memory:
        outputs(dataset, frozen, manifest, synthetic, "synthetic", REHEARSAL_REPLICATES)
    peak = max(memory.peak, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    if peak > MEMORY_LIMIT:
        raise Stop(f"the read rehearsed on {rows} synthetic rows took {peak / 2**30:.2f} GiB of memory, "
                   f"more than {MEMORY_LIMIT / 2**30:g} GiB")
    return peak


class EntryRefused(RuntimeError):
    pass


def entry_checks(dataset, repo, frozen, records_dir, out):
    """Section 10: the registration commit an ancestor of HEAD, and the registered text in it and at HEAD; a clean
    tree; frozen files as in the committed manifest. Also committed: the dry run (Section 11) and, on Hillstrom,
    the role counts (Section 2.1)."""
    if not repo.contains(REGISTRATION_COMMIT):
        raise EntryRefused("the registration commit is not an ancestor of HEAD")
    for commit in (REGISTRATION_COMMIT, "HEAD"):
        text = repo.blob(commit, REGISTRATION)
        if text is None or hashlib.sha256(text).hexdigest() != REGISTRATION_SHA256:
            raise EntryRefused(f"the registration at {commit[:7]} is missing or not the registered text")
    if not repo.clean():
        raise EntryRefused("the working tree is not clean")
    for path in [Path(out) / f"{dataset}-dry-run.json", *role_counts(dataset, records_dir)]:
        if not repo.committed(path):
            raise EntryRefused(f"{path.name} is not committed")
    manifest = repo.committed(Path(records_dir) / f"{dataset}-manifest.json")
    if not manifest:
        raise EntryRefused("no committed manifest")
    for name, digest in json.loads(manifest)["files"].items():
        path = Path(frozen) / f"{name}.pkl"
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise EntryRefused(f"frozen file {name} differs from the manifest")
    return hashlib.sha256(manifest.encode()).hexdigest()


def entries(text, dataset):
    return [e for e in map(json.loads, text.splitlines()) if e["dataset"] == dataset]


def prepare(dataset, repo, frozen, records_dir, out=ROOT / "results"):
    """Rehearse the read on synthetic rows of the test size, then write the read entry (UTC time, commit, manifest
    hash, token hash, the rehearsal's peak memory) for the operator to commit and push.

    The token itself goes to the frozen directory, outside the repository; `go` uses it once. A second
    `go`, a lost container or a restart after `go` began finds no token, so a rerun needs a line in
    AMENDMENTS.md and a new entry (Sections 10 and 11). A manifest other than the first entry's (refits that
    do not reproduce it) makes the entry a second read."""
    versions.check()
    manifest_hash = entry_checks(dataset, repo, frozen, records_dir, out)
    reads = Path(records_dir) / "reads.jsonl"
    earlier = entries(repo.committed(reads), dataset)
    if earlier:
        results_committed = repo.committed(Path(out) / f"{dataset}.json") != ""
        amended = earlier[-1]["time"] in repo.committed(ROOT / "AMENDMENTS.md")
        if results_committed or not amended:
            raise EntryRefused("the dataset has a read entry; only an interrupted read recorded in AMENDMENTS.md may rerun")
    d = data.load(dataset)
    valid, rows = d.valid, len(d.test)
    del d  # the training rows are not used here, and the read does not hold them either
    manifest = json.loads(repo.committed(Path(records_dir) / f"{dataset}-manifest.json"))
    peak = rehearse(dataset, frozen, manifest, valid, rows)
    token = secrets.token_hex(32)
    entry = {"dataset": dataset, "time": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
             "commit": repo.head(), "manifest": manifest_hash, "token": hashlib.sha256(token.encode()).hexdigest(),
             "rerun": bool(earlier), "second read": bool(earlier) and earlier[0]["manifest"] != manifest_hash,
             "rehearsal GiB": round(peak / 2**30, 2)}
    (Path(frozen) / TOKEN).write_text(token)
    with reads.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def go(dataset, repo, frozen, records_dir, out=ROOT / "results", replicates=None):
    """The single path to the test rows: the read entry must be committed and pushed, its token unused, nothing
    else changed, and every frozen model's validation predictions finite.

    The results name HEAD, the commit that holds the read entry, and the command that computed them there
    (Section 15), and say whether they come from a second read (Section 10)."""
    installed = versions.check()
    manifest_hash = entry_checks(dataset, repo, frozen, records_dir, out)
    committed = entries(repo.committed(Path(records_dir) / "reads.jsonl"), dataset)
    if not committed or committed[-1]["manifest"] != manifest_hash or not repo.contains(committed[-1]["commit"]):
        raise EntryRefused("no committed read entry for this manifest")
    if repo.committed(out / f"{dataset}.json"):
        raise EntryRefused("the dataset's results are committed; its test rows are not read again")
    if not repo.pushed():
        raise EntryRefused("HEAD is not on the main branch of the remote")
    token = Path(frozen) / TOKEN
    if not token.exists() or hashlib.sha256(token.read_bytes()).hexdigest() != committed[-1]["token"]:
        raise EntryRefused("this machine holds no unused token of the last read entry; a rerun needs its own entry")
    manifest = json.loads(repo.committed(Path(records_dir) / f"{dataset}-manifest.json"))
    d = data.load(dataset)
    valid, test = d.valid, d.test
    del d  # the training rows are not used here
    for unit, outcome in UNITS[dataset].items():
        x = valid[outcome].x
        frozen_predictions(unit, frozen, x, "validation")
        nuisance = pickle.loads((Path(frozen) / f"{unit} nuisances.pkl").read_bytes())
        check_nuisances(unit, nuisance.predict("m", x), nuisance.propensity(x))
    del valid, x
    token.unlink()
    opened = data.OPEN.set(True)
    try:
        out_all = outputs(dataset, frozen, manifest, test.open()[0], "test", replicates)
    finally:
        data.OPEN.reset(opened)
    out_all |= {"versions": installed, "resources": resources(frozen), "commit": repo.head(),
                "command": f"python -m uplift_ranking.run read {dataset} go", "second read": committed[-1]["second read"]}
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{dataset}.json").write_text(json.dumps(out_all, indent=1) + "\n")
    return out_all


def arguments(argv=None):
    parser = argparse.ArgumentParser(prog="python -m uplift_ranking.run")
    parser.add_argument("stage", choices=("search", "block", "read"))
    parser.add_argument("dataset", choices=tuple(UNITS))
    parser.add_argument("step", nargs="?", choices=("prepare", "go"))
    parser.add_argument("--model", action="append", choices=tuning.MODELS)
    a = parser.parse_args(argv)
    if (a.stage == "read") != (a.step is not None):
        parser.error("read takes a step, prepare or go; search and block take none")
    if a.model and a.stage != "search":
        parser.error("--model applies to search only")
    return a


def main():
    a = arguments()
    repo, frozen, records_dir = Repository(), data.CACHE / a.dataset / "frozen", tuning.RECORDS.parent
    if a.stage == "search":
        search(a.dataset, repo, models=a.model or tuning.MODELS)
    elif a.stage == "block":
        block(a.dataset, choose(a.dataset, repo), frozen=frozen)
    elif a.step == "prepare":
        print(prepare(a.dataset, repo, frozen, records_dir))
    else:
        go(a.dataset, repo, frozen, records_dir)


if __name__ == "__main__":
    main()
