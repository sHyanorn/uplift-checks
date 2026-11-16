"""Section 2 checks on small synthetic files; every planted error must stop the dataset. No network."""

import gzip
import hashlib
import http.server
import json
import threading
import zipfile
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from uplift_ranking import data
from uplift_ranking.roles import TEST, TRAINING, VALIDATION, roles


def sha(b):
    return hashlib.sha256(b).hexdigest()


def role_counts(x, dataset):
    r = roles(x, dataset)
    return {k: int((r == k).sum()) for k in (TEST, VALIDATION, TRAINING)}


@pytest.fixture
def criteo(tmp_path):
    rng = np.random.default_rng(0)
    n = 3000
    frame = pd.DataFrame(np.round(rng.normal(size=(n, 12)) * 10, 6), columns=data.CRITEO_FEATURES)
    frame["treatment"] = (rng.random(n) < 0.85).astype(int)
    frame["conversion"] = (rng.random(n) < 0.01).astype(int)
    frame["visit"] = (rng.random(n) < 0.05).astype(int)
    frame["exposure"] = 0
    path = tmp_path / "criteo.csv.gz"
    with gzip.open(path, "wt") as f:
        frame.to_csv(f, index=False, float_format="%.6f")
    x = frame[data.CRITEO_FEATURES].to_numpy().astype(np.float32)
    expected = {"rows": n, "treated": int(frame["treatment"].sum()), "roles": role_counts(x, "criteo"),
                "means": {c: round(float(frame[c].mean()), 6) for c in ("visit", "conversion")}}
    return path, frame, expected


def test_criteo_reads_when_everything_matches(criteo, tmp_path):
    path, frame, expected = criteo
    d = data.read_criteo(path, expected, tmp_path)
    assert len(d.train["visit"].y) == expected["roles"][TRAINING] and len(d.valid["conversion"].y) == expected["roles"][VALIDATION]
    assert d.train["visit"].x.dtype == np.float32
    with pytest.raises(PermissionError):
        d.test.open()
    token = data.OPEN.set(True)
    try:
        test, groups = d.test.open()
    finally:
        data.OPEN.reset(token)
    assert len(test["visit"].y) == expected["roles"][TEST] and groups.max() + 1 == len(test["visit"].y)


@pytest.mark.parametrize("change", ["rows", "treated", "mean", "roles"])
def test_criteo_stops_on_each_mismatch(criteo, tmp_path, change):
    path, _, expected = criteo
    wrong = json.loads(json.dumps(expected))
    wrong["roles"] = {int(k): v for k, v in wrong["roles"].items()}
    if change == "rows":
        wrong["rows"] += 1
    elif change == "treated":
        wrong["treated"] -= 1
    elif change == "mean":
        wrong["means"]["visit"] = round(wrong["means"]["visit"] + 1e-6, 6)
    else:
        wrong["roles"][TEST] += 1
    with pytest.raises(data.StopCheck, match={"rows": "rows", "treated": "treated", "mean": "mean of visit", "roles": "role counts"}[change]):
        data.read_criteo(path, wrong, tmp_path)


@pytest.mark.parametrize("edit, message", [("extra row", "rows"), ("missing value", "missing value")])
def test_criteo_stops_on_a_changed_file(criteo, tmp_path, edit, message):
    path, frame, expected = criteo
    frame = pd.concat([frame, frame.iloc[:1]]) if edit == "extra row" else frame.assign(visit=frame["visit"].where(frame.index != 5))
    with gzip.open(path, "wt") as f:
        frame.to_csv(f, index=False, float_format="%.6f")
    with pytest.raises(data.StopCheck, match=message):
        data.read_criteo(path, expected, tmp_path)


def lazada_zip(tmp_path, duplicate_id=False):
    rng = np.random.default_rng(1)
    parts = {}
    for part, n, share in (("train", 1200, 0.22), ("test", 400, 0.52)):
        frame = pd.DataFrame(np.round(rng.normal(size=(n, 83)), 4), columns=data.LAZADA_FEATURES)
        frame["is_treat"] = (rng.random(n) < share).astype(int)
        frame["label"] = (rng.random(n) < 0.035).astype(int)
        frame["data_id"] = np.arange(n) + (0 if part == "train" else 10_000)
        parts[part] = frame
    if duplicate_id:
        parts["test"].loc[0, "data_id"] = 0
    members = {f"full_{p}set.csv": f.to_csv(index=False).encode() for p, f in parts.items()} | {"LICENSE": b"BSD 2-clause"}
    path = tmp_path / "lzd.zip"
    with zipfile.ZipFile(path, "w") as z:
        for name, content in members.items():
            z.writestr(name, content)
    raw = path.read_bytes()
    source = data.Source("", "lzd.zip", len(raw), sha(raw), hashlib.md5(raw).hexdigest(),
                         tuple((name, sha(c)) for name, c in members.items()),
                         frozenset(hashlib.md5(members[f"full_{p}set.csv"]).hexdigest() for p in parts))
    x = parts["train"][data.LAZADA_FEATURES].to_numpy().astype(np.float32)
    counts = role_counts(x, "lazada")
    expected = {p: (len(f), int(f["is_treat"].sum())) for p, f in parts.items()}
    expected["roles"] = {VALIDATION: counts[TEST] + counts[VALIDATION], TRAINING: counts[TRAINING]}
    return path, source, expected


def test_lazada_checks(tmp_path):
    path, source, expected = lazada_zip(tmp_path)
    data.verify(source, path)
    d = data.read_lazada(path, expected, tmp_path)
    assert len(d.valid["label"].y) == expected["roles"][VALIDATION]
    for broken in (replace(source, md5="0" * 32), replace(source, member_md5s=frozenset({"0" * 32})),
                   replace(source, members=source.members[:2] + (("LICENSE", "0" * 64),)), replace(source, size=source.size + 1)):
        with pytest.raises(data.StopCheck):
            data.verify(broken, path)
    with pytest.raises(data.StopCheck):
        data.read_lazada(path, dict(expected, test=(expected["test"][0], expected["test"][1] + 1)), tmp_path)
    path, _, expected = lazada_zip(tmp_path, duplicate_id=True)
    with pytest.raises(data.StopCheck, match="data_id"):
        data.read_lazada(path, expected, tmp_path)


def hillstrom_file(tmp_path, n=600):
    rng = np.random.default_rng(2)
    frame = pd.DataFrame({
        "recency": rng.integers(1, 13, n),
        "history_segment": rng.choice(["2) $100 - $200", "1) $0 - $100", "3) $200 - $350"], n),
        "history": np.round(rng.uniform(29, 3000, n), 2),
        "mens": rng.integers(0, 2, n), "womens": rng.integers(0, 2, n),
        "zip_code": rng.choice(["Urban", "Surburban", "Rural"], n),
        "newbie": rng.integers(0, 2, n),
        "channel": rng.choice(["Web", "Phone", "Multichannel"], n),
        "segment": rng.choice(["Mens E-Mail", "Womens E-Mail", "No E-Mail"], n),
        "visit": rng.integers(0, 2, n), "conversion": 0, "spend": 0.0,
    })
    path = tmp_path / "hillstrom.csv"
    frame.to_csv(path, index=False)
    return path, frame, {"rows": n, "segments": frame["segment"].value_counts().to_dict(), "roles": None}


def test_hillstrom_encoding_arms_and_committed_role_counts(tmp_path):
    path, frame, expected = hillstrom_file(tmp_path)
    d = data.read_hillstrom(path, expected, tmp_path)
    encoded = data.encode(frame[data.HILLSTROM_FEATURES])
    assert list(encoded["zip_code"][:6]) == [["Rural", "Surburban", "Urban"].index(v) for v in frame["zip_code"][:6]]
    assert sorted(encoded["history_segment"].unique()) == [0, 1, 2]
    assert list(encoded["history"]) == list(frame["history"])
    x = encoded.to_numpy().astype(np.float32)
    rows = roles(x, "hillstrom") == TRAINING
    assert np.array_equal(d.train["visit"].x, x[rows])
    assert np.array_equal(d.train["visit"].w, frame["segment"].map({"No E-Mail": 0, "Mens E-Mail": 1, "Womens E-Mail": 2})[rows])
    committed = json.loads((tmp_path / "hillstrom-roles.json").read_text())
    assert sum(committed.values()) == 600
    data.read_hillstrom(path, expected, tmp_path)  # same counts: passes
    (tmp_path / "hillstrom-roles.json").write_text(json.dumps({k: v + (k == "0") for k, v in committed.items()}))
    with pytest.raises(data.StopCheck, match="committed"):
        data.read_hillstrom(path, expected, tmp_path)
    with pytest.raises(data.StopCheck, match="segments"):
        data.read_hillstrom(path, dict(expected, segments={**expected["segments"], "No E-Mail": 1}), tmp_path)


class RangeHandler(http.server.BaseHTTPRequestHandler):
    body = b""
    honour_range = True
    asked = []

    def do_GET(self):
        self.asked.append(self.headers.get("Range"))
        start = int(self.headers["Range"][6:-1]) if self.headers.get("Range") and self.honour_range else 0
        self.send_response(206 if start else 200)
        self.end_headers()
        self.wfile.write(self.body[start:])

    def log_message(self, *args):
        pass


@pytest.mark.parametrize("honour_range", [True, False])
def test_download_resumes_and_verifies(tmp_path, monkeypatch, honour_range):
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    body = bytes(range(256)) * 400
    handler = type("H", (RangeHandler,), {"body": body, "honour_range": honour_range, "asked": []})
    server = http.server.HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        source = data.Source(f"http://127.0.0.1:{server.server_port}/f", "f.bin", len(body), sha(body))
        (tmp_path / "f.bin.part").write_bytes(body[:1000])
        assert data.download(source, tmp_path).read_bytes() == body
        assert handler.asked == ["bytes=1000-"]
        (tmp_path / "f.bin").unlink()
        with pytest.raises(data.StopCheck):
            data.download(replace(source, sha256="0" * 64), tmp_path)
    finally:
        server.shutdown()
