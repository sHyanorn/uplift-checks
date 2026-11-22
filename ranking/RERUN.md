# How to rerun

Each dataset (`criteo`, `lazada` or `hillstrom`) then runs in four steps, each a command `uv run --no-sync python -m uplift_ranking.run ...` in `ranking/`. Commit and push each step's files to the main branch before the next step: the next step refuses uncommitted files, and `block` and `go` also refuse a HEAD that is not on the remote's main branch (Section 11).

1. `search DATASET` downloads the file, checks it against Section 2 (checksums, counts, means) and runs every search trial without a record in `records/trials.jsonl`; `--model NAME` (repeatable, quoted for `"causal forest"`) limits it to the named models. On Hillstrom it also writes `records/hillstrom-roles.json`, as Section 2.1 leaves those counts to the first run; `block` and the read stop unless that file is committed.
2. `block DATASET` selects configurations from the committed records, fits seeds 0 to 4 of every model and the outcome and nuisance models, freezes them, and writes `records/DATASET-manifest.json` and the dry run.
3. `read DATASET prepare` checks the entry conditions (the registration commit `9e8678d` in HEAD's history, the registration's SHA-256 there and at HEAD, a clean tree, the committed dry run, the frozen files against the committed manifest). It then rehearses the read: the frozen models predict synthetic rows of the test size, drawn from the validation rows' values, and the evaluation runs on them with 16 replicates. If the rehearsal's peak memory, this process and its workers together, is above 12 GiB, `prepare` stops (Section 10). Otherwise it leaves a one-time token in the cache and appends a read entry with the token's hash and that peak to `records/reads.jsonl`.
4. `read DATASET go` repeats the entry conditions and requires the entry committed and pushed, the results not yet committed, the entry's token on this machine and finite validation predictions of the frozen models; it then deletes the token, reads the test rows once and writes `results/DATASET.json` (Section 10).

The search may span machines of one CPU type and package build: each machine refits the nuisance models m-hat and e of Section 5.4, stops if their fingerprint differs from the first record, and skips trials that have a record; `block` stops if a unit's records differ in their fingerprint. A record counts only while the source files of its model's family are unchanged (Section 11). Steps 2 to 4 run on one machine, whose cache (`~/.cache/uplift-checks/` or `UPLIFT_CHECKS_CACHE`) holds the downloads and the frozen models. Download addresses and checksums are in `ranking/src/uplift_ranking/data.py`.

The code fixes the thread counts: 4 for LightGBM and torch, one job per forest. Section 11 gives timings on a 4-core container with 15 GiB of memory: about 52 hours of search for Criteo, 22 for Lazada and 1 for Hillstrom, and about 15, 10 and under 1 hour for the final block, from `block` through the test read.

At the records commit, `block` should reproduce the committed manifest and dry run on the same CPU type, package builds and thread counts, solver times and resource figures aside (a test checks the manifest on synthetic data). A completed read is not repeated: `go` deletes the entry's token as it opens the test rows and refuses once the dataset's results are committed. `prepare` accepts a further entry only while the dataset's results are uncommitted and the previous entry's `time` is in the committed `ranking/AMENDMENTS.md`. After a read entry, `block` refits with the configurations, R*, records commit and disagreements stored in the committed manifest and keeps the committed dry run. If the trial records and fitting code are unchanged and the refits reproduce the frozen files, the manifest hash is unchanged and the new entry reruns the interrupted read; if the manifest differs from the dataset's first entry, the entry and its results are marked as a second read (Sections 10 and 11).

The registration file never changes once committed; an amendment is a commit stating what changed and why (Section 15), with an entry in `ranking/AMENDMENTS.md`.

| Path in `ranking/` | Contents |
|---|---|
| `scripts/time_deep.py` | Timing and peak memory of the deep models on synthetic data |
| `records/trials.jsonl` | One line per search trial, summary values only |
| `records/DATASET-manifest.json` | SHA-256 of each frozen file; selected configurations, R*, records commit, R-loss checks and disagreements between duplicate trial records |
| `records/reads.jsonl` | Read entries: dataset, time, commit, manifest hash, token hash, whether the entry reruns a read, whether it is a second read, and the rehearsal's peak memory |
| `results/` | Aggregate outputs of each dry run and test read |
| `AMENDMENTS.md` | Amendments and interrupted test reads |
