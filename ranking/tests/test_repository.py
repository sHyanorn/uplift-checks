"""No data, predictions or model files in the repository (registration, Section 2)."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAX_BYTES = 1_000_000
DATA_SUFFIXES = {
    ".csv", ".tsv", ".gz", ".zip", ".bz2", ".xz", ".parquet", ".feather", ".arrow",
    ".npy", ".npz", ".pkl", ".pickle", ".joblib", ".pt", ".pth", ".h5", ".hdf5", ".lgb",
}


def offending(root, paths):
    bad = []
    for path in paths:
        file = root / path
        if DATA_SUFFIXES & {s.lower() for s in Path(path).suffixes} or file.stat().st_size > MAX_BYTES:
            bad.append(path)
    return bad


def repository_files():
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT, capture_output=True, check=True,
    ).stdout
    return [p for p in out.decode().split("\0") if p and (ROOT / p).is_file()]


def test_scan_flags_data_and_large_files(tmp_path):
    for name, size in [("a.csv", 10), ("b.csv.gz", 10), ("models/m.pt", 10), ("big.json", MAX_BYTES + 1), ("ok.py", 10)]:
        (tmp_path / name).parent.mkdir(exist_ok=True)
        (tmp_path / name).write_bytes(b"x" * size)
    names = ["a.csv", "b.csv.gz", "models/m.pt", "big.json", "ok.py"]
    assert offending(tmp_path, names) == ["a.csv", "b.csv.gz", "models/m.pt", "big.json"]


def test_repository_holds_no_data_file():
    files = repository_files()
    assert "ranking/PREREGISTRATION.md" in files
    assert offending(ROOT, files) == []
