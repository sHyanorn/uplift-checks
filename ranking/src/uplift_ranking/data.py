"""Downloads, checksums, reading and the stop checks of Section 2.

Files live in the cache directory (default ~/.cache/uplift-checks), never in the repository. Test
rows are returned sealed: `Sealed.open` works only inside the entry point of Section 10.
"""

import contextvars
import hashlib
import json
import os
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .learners import Split
from .roles import TEST, TRAINING, VALIDATION, groups, roles

CACHE = Path(os.environ.get("UPLIFT_CHECKS_CACHE", "~/.cache/uplift-checks")).expanduser()
RECORDS = Path(__file__).resolve().parents[2] / "records"


@dataclass(frozen=True)
class Source:
    url: str
    name: str
    size: int
    sha256: str
    md5: str = ""
    members: tuple = ()  # (name, sha256) of archive members
    member_md5s: frozenset = frozenset()  # MD5 sums printed for members (Section 2), in no stated order: a set


SOURCES = {
    # Section 2's Criteo file at its registered revision, through Hugging Face's resolve address.
    "criteo": Source(
        "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/2424920019e49d52d72c13ac1143ec5d53af276b/"
        "criteo-research-uplift-v2.1.csv.gz",
        "criteo-research-uplift-v2.1.csv.gz", 311_422_618,
        "2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc"),
    "lazada": Source(
        "https://www.dropbox.com/s/07r7592h9mfijsb/lzd_data_public.zip?dl=1",
        "lzd_data_public.zip", 101_629_017,
        "a657288e9a54539008004b321fcf986737e94aeea8468b9d70b96f6a974e73db",
        md5="545bb0eae05cca58f0d813a990ed9cba",
        members=(("full_trainset.csv", "79cb6e5c2fb7ad0c35710a84ed82275baa167647b17bf3423f402602b02b5bf0"),
                 ("full_testset.csv", "1eec5f10b838ca3a43f6ccaf1b7af9260e300be36df452be58ee6678d7e1a326"),
                 ("LICENSE", "3ae95b8e3fa8f3cae1af3c44274586179e7dda9f3cddba6bc7565d249c7cc7a3")),
        member_md5s=frozenset({"efffc70375e700bcb91a9d41eddd7f66", "9058fa57b984f0112a1ecf71dc52fb1e"})),
    "hillstrom": Source(
        "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv",
        "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv", 3_964_977,
        "0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece"),
}
EXPECTED = {
    "criteo": {"rows": 13_979_592, "treated": 11_882_655, "means": {"visit": 0.046992, "conversion": 0.002917},
               "roles": {TEST: 4_195_534, VALIDATION: 1_398_844, TRAINING: 8_385_214}},
    "lazada": {"train": (926_669, 205_480), "test": (181_669, 94_682),
               "roles": {VALIDATION: 93_670, TRAINING: 832_999}},  # rows of the training file
    "hillstrom": {"rows": 64_000, "segments": {"Mens E-Mail": 21_307, "Womens E-Mail": 21_387, "No E-Mail": 21_306},
                  "roles": None},  # committed by the first run
}
CRITEO_FEATURES = [f"f{j}" for j in range(12)]
LAZADA_FEATURES = [f"f{j}" for j in range(83)]
HILLSTROM_FEATURES = ["recency", "history_segment", "history", "mens", "womens", "zip_code", "newbie", "channel"]
HILLSTROM_ARMS = {"No E-Mail": 0, "Mens E-Mail": 1, "Womens E-Mail": 2}


class StopCheck(RuntimeError):
    """A failed check of Section 2: the dataset stops and the failure is reported."""


def require(ok, message):
    if not ok:
        raise StopCheck(message)


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(source, directory=CACHE):
    """Fetch the file, resuming a partial download, and verify its size and SHA-256."""
    directory.mkdir(parents=True, exist_ok=True)
    path, part = directory / source.name, directory / (source.name + ".part")
    if not path.exists():
        done = part.stat().st_size if part.exists() else 0
        request = urllib.request.Request(source.url, headers={"Range": f"bytes={done}-"} if done else {})
        with urllib.request.urlopen(request) as response, part.open("ab" if response.status == 206 else "wb") as f:
            for block in iter(lambda: response.read(1 << 20), b""):
                f.write(block)
        part.rename(path)
    verify(source, path)
    return path


def verify(source, path):
    require(path.stat().st_size == source.size, f"{path.name}: {path.stat().st_size} bytes, registered {source.size}")
    require(digest(path) == source.sha256, f"{path.name}: SHA-256 differs from the registered one")
    if source.md5:
        require(digest(path, "md5") == source.md5, f"{path.name}: MD5 differs from the README")
    if source.members:
        with zipfile.ZipFile(path) as z:
            md5s = set()
            for name, sha in source.members:
                data = z.read(name)
                require(hashlib.sha256(data).hexdigest() == sha, f"{path.name}/{name}: SHA-256 differs")
                md5s.add(hashlib.md5(data).hexdigest())
            require(source.member_md5s <= md5s, f"{path.name}: member MD5 sums differ from the README")


class Sealed:
    """Test rows, readable only while the entry point of Section 10 holds them open."""

    def __init__(self, split, groups):
        self._split, self._groups = split, groups

    def open(self):
        if not OPEN.get():
            raise PermissionError("test rows are read only through the entry point (Section 10)")
        return self._split, self._groups


OPEN = contextvars.ContextVar("test rows open", default=False)


@dataclass
class Data:
    """A dataset after its checks: training and validation rows, and the sealed test rows.

    `y` holds every outcome of the dataset; Hillstrom's `w` is the three-arm code (0 none, 1 men's, 2 women's)."""

    name: str
    train: dict  # outcome -> Split
    valid: dict
    test: Sealed


def check_common(frame, columns, label):
    require(not frame[columns].isna().any().any(), f"{label}: missing value in a used column")


def check_role_counts(name, role, expected, records=RECORDS):
    """Role counts against Section 2.1's; Hillstrom's are written by the first run to records/hillstrom-roles.json,
    committed by the operator, and later runs whose counts differ stop."""
    counts = {int(r): int(c) for r, c in zip(*np.unique(role, return_counts=True))}
    if expected is not None:
        require({r: counts.get(r, 0) for r in expected} == expected, f"{name}: role counts {counts}, registered {expected}")
        return counts
    path = Path(records) / f"{name}-roles.json"
    if path.exists():
        require(json.loads(path.read_text()) == {str(r): c for r, c in counts.items()}, f"{name}: role counts differ from the committed ones")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({str(r): c for r, c in counts.items()}) + "\n")
    return counts


def encode(frame):
    """Each string column replaced by the 0-based index of its value among the file's sorted distinct values."""
    out = frame.copy()
    for column in frame.columns:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            out[column] = np.searchsorted(np.sort(frame[column].unique()), frame[column].to_numpy())
    return out


def split_rows(x, w, outcomes, role, keep):
    rows = role == keep
    return {name: Split(x[rows], w[rows], y[rows]) for name, y in outcomes.items()}


def sealed(x, w, outcomes, rows):
    return Sealed({name: Split(x[rows], w[rows], y[rows]) for name, y in outcomes.items()}, groups(x[rows]))


def read_criteo(path, expected=EXPECTED["criteo"], records=RECORDS):
    frame = pd.read_csv(path, engine="c", usecols=CRITEO_FEATURES + ["treatment", "visit", "conversion"],
                        dtype={c: np.float32 for c in CRITEO_FEATURES})
    check_common(frame, frame.columns, "criteo")
    require(len(frame) == expected["rows"], f"criteo: {len(frame)} rows")
    require(int(frame["treatment"].sum()) == expected["treated"], f"criteo: {int(frame['treatment'].sum())} treated")
    for column, mean in expected["means"].items():
        require(round(float(frame[column].mean()), 6) == mean, f"criteo: mean of {column} {frame[column].mean():.8f}")
    x = frame[CRITEO_FEATURES].to_numpy(np.float32)
    w = frame["treatment"].to_numpy(np.int64)
    outcomes = {"visit": frame["visit"].to_numpy(np.int64), "conversion": frame["conversion"].to_numpy(np.int64)}
    role = roles(x, "criteo")
    check_role_counts("criteo", role, expected["roles"], records)
    return Data("criteo", split_rows(x, w, outcomes, role, TRAINING), split_rows(x, w, outcomes, role, VALIDATION),
                sealed(x, w, outcomes, role == TEST))


def read_lazada(path, expected=EXPECTED["lazada"], records=RECORDS):
    with zipfile.ZipFile(path) as z:
        frames = {part: pd.read_csv(z.open(f"full_{part}set.csv")) for part in ("train", "test")}
    for part, frame in frames.items():
        check_common(frame, LAZADA_FEATURES + ["is_treat", "label", "data_id"], f"lazada {part}")
        rows, treated = expected[part]
        require(len(frame) == rows and int(frame["is_treat"].sum()) == treated, f"lazada {part}: {len(frame)} rows")
    ids = pd.concat([frames["train"]["data_id"], frames["test"]["data_id"]])
    require(ids.is_unique, "lazada: data_id not unique")
    data = {part: (f[LAZADA_FEATURES].to_numpy().astype(np.float32), f["is_treat"].to_numpy(np.int64),
                   {"label": f["label"].to_numpy(np.int64)}) for part, f in frames.items()}
    x, w, outcomes = data["train"]
    role = roles(x, "lazada")
    check_role_counts("lazada", role, expected["roles"], records)
    test_x, test_w, test_outcomes = data["test"]
    return Data("lazada", split_rows(x, w, outcomes, role, TRAINING), split_rows(x, w, outcomes, role, VALIDATION),
                sealed(test_x, test_w, test_outcomes, np.ones(len(test_w), bool)))


def read_hillstrom(path, expected=EXPECTED["hillstrom"], records=RECORDS):
    frame = pd.read_csv(path)
    check_common(frame, HILLSTROM_FEATURES + ["segment", "visit"], "hillstrom")
    require(len(frame) == expected["rows"], f"hillstrom: {len(frame)} rows")
    counts = frame["segment"].value_counts().to_dict()
    require(counts == expected["segments"], f"hillstrom: segments {counts}")
    x = encode(frame[HILLSTROM_FEATURES]).to_numpy().astype(np.float32)
    arm = frame["segment"].map(HILLSTROM_ARMS).to_numpy(np.int64)
    outcomes = {"visit": frame["visit"].to_numpy(np.int64)}
    role = roles(x, "hillstrom")
    check_role_counts("hillstrom", role, expected["roles"], records)
    return Data("hillstrom", split_rows(x, arm, outcomes, role, TRAINING), split_rows(x, arm, outcomes, role, VALIDATION),
                sealed(x, arm, outcomes, role == TEST))


READERS = {"criteo": read_criteo, "lazada": read_lazada, "hillstrom": read_hillstrom}


def load(name, directory=CACHE):
    """Download (or find) and verify the file, then read it with the stop checks of Section 2."""
    return READERS[name](download(SOURCES[name], directory))
