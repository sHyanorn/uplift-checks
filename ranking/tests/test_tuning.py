import json
import math

import numpy as np
import pytest

from uplift_ranking import learners, tuning
from uplift_ranking.learners import Split
from uplift_ranking.synthetic import rct


def test_grids_follow_the_registered_order():
    sizes = {m: len(tuning.grid(m)) for m in tuning.MODELS}
    assert sizes == {"S": 360, "T": 360, "X": 360, "R": 360, "DR": 360, "causal forest": 9, "TARNet": 480, "DESCN": 4320, "EFIN": 4320}
    g = tuning.grid("S")
    assert g[0] == {"num_leaves": 15, "learning_rate": 0.05, "min_child_samples": 20, "feature_fraction": 0.6, "lambda_l2": 0}
    assert g[1]["lambda_l2"] == 1 and g[3]["feature_fraction"] == 0.8 and g[72]["num_leaves"] == 31
    assert g[-1] == {"num_leaves": 255, "learning_rate": 0.1, "min_child_samples": 2000, "feature_fraction": 1.0, "lambda_l2": 10}
    assert tuning.grid("causal forest")[1] == {"min_samples_leaf": 500, "max_features": 0.8}
    d = tuning.grid("EFIN")
    assert d[0] == {"rank": 32, "batch": 256, "lr": 1e-4, "weight_decay": 1e-5, "alpha": 0.1, "standardise": False}
    assert d[1]["standardise"] is True and d[2]["alpha"] == 0.2 and d[-1]["rank"] == 128
    assert "alpha" not in tuning.grid("TARNet")[0] and tuning.grid("TARNet")[2]["weight_decay"] == 1e-4


def test_draws_follow_the_registered_generator():
    for unit in (1, 2, 3, 4):
        for index, model in enumerate(tuning.MODELS):
            if model == "causal forest":
                assert tuning.drawn(unit, model) == list(range(9))
                continue
            size = len(tuning.grid(model))
            expected = np.random.default_rng([20261115, 2, unit, index]).choice(size, 20, replace=False)
            assert tuning.drawn(unit, model) == list(expected) and len(set(expected)) == 20
    assert tuning.drawn(1, "S") != tuning.drawn(2, "S") and tuning.drawn(1, "S") != tuning.drawn(1, "T")


@pytest.fixture(scope="module")
def splits():
    t, v = rct(3000, 1, treated=0.3), rct(1500, 2, treated=0.3)
    train, valid = Split(t.x, t.w, t.y), Split(v.x, v.w, v.y)
    _, m, e = learners.nuisances(train, valid, cross_fitted_e=False)
    return train, valid, m, e


def deep_index(model, **want):
    return next(i for i, c in enumerate(tuning.grid(model)) if all(c[k] == v for k, v in want.items()))


@pytest.mark.parametrize("model, index", [("T", 120), ("TARNet", None)])
def test_a_trial_record_holds_summaries_only(splits, model, index, tmp_path):
    train, valid, m, e = splits
    index = index if index is not None else deep_index("TARNet", rank=32, batch=2048, lr=5e-3)
    record = tuning.trial("hillstrom", 4, model, index, train, valid, m, e, commit="abc123")
    assert tuple(record) == tuning.FIELDS
    effect, _ = tuning.fit(model, tuning.grid(model)[index], train, valid, 0, "hillstrom")
    assert record["r_loss"] == learners.r_loss(effect(valid.x), valid.y, valid.w, m, e)
    assert record["fingerprint"] == tuning.fingerprint(m, e) and record["code"] == tuning.code_hash(model)
    assert set(record["stopping"]) == ({"mu0", "mu1"} if model == "T" else {"epochs"})
    tuning.append(record, tmp_path / "trials.jsonl")
    line = (tmp_path / "trials.jsonl").read_text()
    assert len(line) < 1500 and line.count("\n") == 1
    assert all(not isinstance(v, list) for v in json.loads(line).values())
    assert tuning.read(tmp_path / "trials.jsonl") == [record]


def test_infinite_r_loss_survives_the_record(tmp_path):
    record = dict.fromkeys(tuning.FIELDS, 0) | {"r_loss": float("inf")}
    tuning.append(record, tmp_path / "t.jsonl")
    assert math.isinf(tuning.read(tmp_path / "t.jsonl")[0]["r_loss"])


def record(config, loss, unit=1, model="S", code=None, fingerprint="f"):
    return dict.fromkeys(tuning.FIELDS) | {"unit": unit, "model": model, "config": config, "seed": 0, "r_loss": loss,
                                           "code": code or tuning.code_hash(model), "fingerprint": fingerprint}


def test_resume_and_selection():
    drawn = tuning.drawn(1, "S")
    records = [record(i, 1.0 + k) for k, i in enumerate(drawn[:15])]
    assert tuning.pending(records, 1, "S") == drawn[15:]
    with pytest.raises(ValueError, match="without a record"):
        tuning.select(records, 1, "S")
    records += [record(i, 5.0) for i in drawn[15:]]
    assert tuning.pending(records, 1, "S") == []
    assert tuning.select(records, 1, "S") == drawn[0]
    tie = sorted(drawn[:2])
    records = [record(i, 0.5 if i in tie else 1.0) for i in drawn]
    assert tuning.select(records, 1, "S") == tie[0]
    records[[r["config"] for r in records].index(tie[0])]["r_loss"] = float("inf")
    assert tuning.select(records, 1, "S") == tie[1]
    with pytest.raises(ValueError, match="no trial with a finite R-loss"):
        tuning.select([record(i, float("inf")) for i in drawn], 1, "S")


def test_first_record_counts_and_disagreements_are_listed():
    drawn = tuning.drawn(1, "T")
    records = [record(i, 1.0, model="T") for i in drawn] + [record(drawn[3], 0.1, model="T")]
    assert tuning.select(records, 1, "T") == min(drawn)
    assert tuning.disagreements(records, 1, "T") == [records[-1]]


def test_records_from_older_fitting_code_are_not_used(monkeypatch, tmp_path):
    drawn = tuning.drawn(1, "R")
    old = [record(i, 1.0, model="R") for i in drawn] + [record(i, 1.0, model="causal forest") for i in range(9)]
    assert tuning.pending(old, 1, "R") == []
    source = tmp_path / "learners.py"
    source.write_text("changed")
    monkeypatch.setattr(tuning, "SOURCES", {**tuning.SOURCES, "lightgbm": (str(source),)})
    assert tuning.pending(old, 1, "R") == drawn
    assert tuning.pending(old, 1, "causal forest") == []


def test_a_change_to_the_fitting_choices_changes_every_code_hash(monkeypatch, tmp_path):
    for source in tuning.SOURCE_DIR.glob("*.py"):
        (tmp_path / source.name).write_bytes(source.read_bytes())
    monkeypatch.setattr(tuning, "SOURCE_DIR", tmp_path)
    before = {model: tuning.code_hash(model) for model in tuning.MODELS}
    fitting = (tmp_path / "tuning.py").read_text().replace('cross_fitted_e=dataset == "lazada"', "cross_fitted_e=True")
    (tmp_path / "tuning.py").write_text(fitting)
    assert all(tuning.code_hash(model) != before[model] for model in tuning.MODELS)


def test_fingerprint_stops_the_search():
    records = [record(0, 1.0, unit=1, fingerprint="a"), record(1, 1.0, unit=3, fingerprint="b")]
    tuning.check_fingerprint(records, 1, "a")
    tuning.check_fingerprint(records, 2, "anything")
    with pytest.raises(tuning.FingerprintMismatch):
        tuning.check_fingerprint(records, 3, "a")
    with pytest.raises(tuning.FingerprintMismatch):
        tuning.check_fingerprint(records + [record(2, 1.0, unit=1, fingerprint="b")], 1)
    unused = [record(0, 1.0, code="of older fitting code", fingerprint="b"), record(1, 1.0, fingerprint="a")]
    tuning.check_fingerprint(unused, 1, "a")  # Section 11: a record that is not used does not stop the search
    m, e = np.arange(4.0), np.full(4, 0.3)
    assert tuning.fingerprint(m, e) == tuning.fingerprint(m.astype(np.float32), e) != tuning.fingerprint(m + 1e-12, e)


def test_criteo_forest_is_fitted_on_the_drawn_rows(monkeypatch, splits):
    train, valid, _, _ = splits
    seen = {}
    monkeypatch.setattr(tuning.forest, "criteo_rows", lambda: np.arange(0, len(train.y), 3))
    monkeypatch.setattr(tuning.forest, "fit", lambda t, v, c, s, trees: seen.update(rows=len(t.y), trees=trees))
    tuning.fit("causal forest", tuning.grid("causal forest")[0], train, valid, 0, "criteo")
    assert seen == {"rows": len(range(0, len(train.y), 3)), "trees": 200}
    tuning.fit("causal forest", tuning.grid("causal forest")[0], train, valid, 0, "lazada")
    assert seen["rows"] == len(train.y)
