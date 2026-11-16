"""CPU time and peak memory of the deep models on synthetic data of the datasets' shapes.

    python scripts/time_deep.py OUT.jsonl [--rows 100000]

Each (model, features, rank, batch) cell trains one epoch in its own process; each (model,
features, rank) cell predicts in chunks of 50,000 rows in its own process. Cells already in OUT
are skipped, so an interrupted run resumes.
"""

import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SHAPES = {"Criteo": 12, "Lazada": 83, "Hillstrom": 8}
RANKS, BATCHES = (32, 64, 128), (256, 512, 1024, 2048)


def train_cell(model, d, rank, batch, rows):
    from uplift_ranking import deep
    from uplift_ranking.synthetic import rct

    train, valid = rct(rows, 0, d=d), rct(1000, 1, d=d)
    start = time.perf_counter()
    deep.fit(model, (train.x, train.w, train.y), (valid.x, valid.w, valid.y), rank=rank, batch=batch, lr=1e-3,
             weight_decay=1e-4, alpha=0.5, standardise=True, seed=0, max_epochs=1)
    return {"train_s_per_million": (time.perf_counter() - start) / rows * 1e6}


def predict_cell(model, d, rank, rows):
    import torch

    from uplift_ranking import deep
    from uplift_ranking.synthetic import rct

    deep.configure()
    x = rct(rows, 0, d=d).x
    torch.manual_seed(0)
    fitted = deep.Fitted(deep.MODELS[model](d, rank), np.zeros(d, np.float32), np.ones(d, np.float32), 0, [])
    start = time.perf_counter()
    fitted.effect(x)
    return {"predict_s_per_million": (time.perf_counter() - start) / rows * 1e6}


def run(out, rows):
    done = {line["cell"] for line in map(json.loads, out.read_text().splitlines())} if out.exists() else set()
    cells = [(m, s, r, b) for m in ("TARNet", "DESCN", "EFIN") for s in SHAPES for r in RANKS for b in (*BATCHES, None)]
    for model, shape, rank, batch in cells:
        cell = f"{model} {shape} rank {rank} " + (f"batch {batch}" if batch else "predict")
        if cell in done:
            continue
        args = [sys.executable, __file__, "--cell", model, str(SHAPES[shape]), str(rank), str(batch or 0), "--rows", str(rows)]
        result = json.loads(subprocess.run(args, capture_output=True, text=True, check=True).stdout)
        with out.open("a") as f:
            f.write(json.dumps({"cell": cell, "model": model, "shape": shape, "rank": rank, "batch": batch, "rows": rows, **result}) + "\n")
        print(cell, result, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", nargs="?", type=Path)
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--cell", nargs=4)
    a = parser.parse_args()
    if a.cell is None:
        return run(a.out, a.rows)
    model, d, rank, batch = a.cell[0], *map(int, a.cell[1:])
    result = train_cell(model, d, rank, batch, a.rows) if batch else predict_cell(model, d, rank, 2 * a.rows)
    result["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(json.dumps(result))


if __name__ == "__main__":
    main()
