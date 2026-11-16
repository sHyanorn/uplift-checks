"""Grids, configuration draws, search trials, their records and the selection (Sections 6 and 11).

A record is one JSON line per finished trial in ranking/records/trials.jsonl. It holds summary
values only: no row-level value is written. `code` is the SHA-256 of the source files that fit the
model: its family's modules in SOURCES and this module, which holds the grids and the choices of
`fit`. Records written before a change to those files are not used (Section 11).
"""

import hashlib
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np

from . import deep, forest, learners, versions

MODELS = ("S", "T", "X", "R", "DR", "causal forest", "TARNet", "DESCN", "EFIN")
META = MODELS[:5]
DEEP = MODELS[6:]
BUDGET = 20
RECORDS = Path(__file__).resolve().parents[2] / "records" / "trials.jsonl"
FIELDS = ("dataset", "unit", "model", "config", "seed", "r_loss", "stopping", "seconds", "versions", "cpu", "commit",
          "code", "fingerprint")

GRIDS = {
    "lightgbm": {
        "num_leaves": (15, 31, 63, 127, 255),
        "learning_rate": (0.05, 0.1),
        "min_child_samples": (20, 100, 500, 2000),
        "feature_fraction": (0.6, 0.8, 1.0),
        "lambda_l2": (0, 1, 10),
    },
    "forest": {"min_samples_leaf": (500, 2000, 5000), "max_features": (0.5, 0.8, 1.0)},
    "deep": {
        "rank": (32, 64, 128),
        "batch": (256, 512, 1024, 2048),
        "lr": (1e-4, 5e-4, 1e-3, 5e-3, 1e-2),
        "weight_decay": (1e-5, 1e-4, 1e-3, 1e-2),
        "alpha": (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        "standardise": (False, True),
    },
}
SOURCE_DIR = Path(__file__).parent
SOURCES = {"lightgbm": ("learners.py", "roles.py", "tuning.py"), "forest": ("forest.py", "learners.py", "tuning.py"),
           "deep": ("deep.py", "tuning.py")}


def family(model):
    return "lightgbm" if model in META else "forest" if model == "causal forest" else "deep"


def grid(model):
    """Every configuration of the model's grid: parameters in the listed order, the last varying fastest."""
    params = {k: v for k, v in GRIDS[family(model)].items() if not (model == "TARNet" and k == "alpha")}
    return [dict(zip(params, values)) for values in itertools.product(*params.values())]


def drawn(unit, model):
    """Configuration indices of the search: 20 drawn from the grid, or all 9 of the causal forest."""
    size = len(grid(model))
    if model == "causal forest":
        return list(range(size))
    index = MODELS.index(model)
    return [int(i) for i in np.random.default_rng([20261115, 2, unit, index]).choice(size, BUDGET, replace=False)]


def code_hash(model):
    digest = hashlib.sha256()
    for name in SOURCES[family(model)]:
        digest.update((SOURCE_DIR / name).read_bytes())
    return digest.hexdigest()


def fingerprint(m, e):
    """The fingerprint of Section 11: SHA-256 of the validation predictions of m-hat and then e (5.4), as float64."""
    return hashlib.sha256(np.asarray(m, np.float64).tobytes() + np.asarray(e, np.float64).tobytes()).hexdigest()


def cpu_model():
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return ""


def fit(model, config, train, valid, seed, dataset, trees=None):
    """Fit one model; returns its effect function and the trees or epochs chosen by early stopping."""
    if model in META:
        fitted = learners.fit(model, train, valid, config, seed, cross_fitted_e=dataset == "lazada")
        return fitted.effect, fitted.trees
    if model == "causal forest":
        if dataset == "criteo":
            train = train.rows(forest.criteo_rows())
        trees = trees or forest.SEARCH_TREES
        fitted = forest.fit(train, valid, config, seed, trees)
        return (lambda x: forest.effect(fitted, x)), {"trees": trees}
    fitted = deep.fit(model, (train.x, train.w, train.y), (valid.x, valid.w, valid.y), seed=seed, **{"alpha": None, **config})
    return fitted.effect, {"epochs": fitted.epochs}


def trial(dataset, unit, model, index, train, valid, m, e, commit, seed=0):
    """Fit configuration `index` of the model and return its record."""
    start = time.perf_counter()
    effect, stopping = fit(model, grid(model)[index], train, valid, seed, dataset)
    loss = learners.r_loss(effect(valid.x), valid.y, valid.w, m, e)
    return {
        "dataset": dataset, "unit": unit, "model": model, "config": index, "seed": seed, "r_loss": loss,
        "stopping": stopping, "seconds": round(time.perf_counter() - start, 1), "versions": versions.check(),
        "cpu": cpu_model(), "commit": commit, "code": code_hash(model), "fingerprint": fingerprint(m, e),
    }


def read(path=RECORDS):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def append(record, path=RECORDS):
    assert tuple(record) == FIELDS
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


class FingerprintMismatch(RuntimeError):
    pass


def check_fingerprint(records, unit, value):
    """Stop (search) if m-hat and e differ from those of the unit's first record."""
    first = next((r for r in records if r["unit"] == unit), None)
    if first is not None and first["fingerprint"] != value:
        raise FingerprintMismatch(f"unit {unit}: fingerprint {value} differs from the first record's {first['fingerprint']}")


def usable(records, unit, model):
    """First record of each trial of the unit and model written with the current fitting code."""
    code, first = code_hash(model), {}
    for r in records:
        if r["unit"] == unit and r["model"] == model and r["seed"] == 0 and r["code"] == code:
            first.setdefault(r["config"], r)
    return first


def pending(records, unit, model):
    done = usable(records, unit, model)
    return [i for i in drawn(unit, model) if i not in done]


def disagreements(records, unit, model):
    """Later records of a trial whose R-loss differs from its first record."""
    first = usable(records, unit, model)
    code = code_hash(model)
    return [r for r in records if r["unit"] == unit and r["model"] == model and r["seed"] == 0 and r["code"] == code
            and r is not first[r["config"]] and r["r_loss"] != first[r["config"]]["r_loss"]]


def select(records, unit, model):
    """The configuration with the smallest validation R-loss, ties to the smaller index; inf is never selected."""
    if pending(records, unit, model):
        raise ValueError(f"unit {unit} {model}: trials without a record: {pending(records, unit, model)}")
    trials = usable(records, unit, model)
    finite = [(r["r_loss"], i) for i, r in trials.items() if i in drawn(unit, model) and math.isfinite(r["r_loss"])]
    if not finite:
        raise ValueError(f"unit {unit} {model}: no trial with a finite R-loss")
    return min(finite)[1]
