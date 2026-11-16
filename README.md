# Uplift checks

A pre-registered study of how uplift models are compared, on three public randomised trials: Criteo Uplift v2.1, Lazada and Hillstrom. The registration, [`ranking/PREREGISTRATION.md`](ranking/PREREGISTRATION.md), fixes the questions, data, models, statistics and decision rules before any model is evaluated. "Section" below means a section of that file.

## What it is

The study asks (Section 1):

- Q1. Given the same predictions, do different implementations and definitions of the Qini and uplift curves and of the rank-weighted average treatment effect (RATE) rank the models alike?
- Q2. Of the differences between nine tuned models, how many can paired intervals tell from zero, and do the deep models beat the best non-deep model?
- Q3. What is the smallest difference each test set detects, and how many differences reported in four papers (Appendix A) fall below it at their test sizes?

A smaller part (Section 8) estimates the value of treating the top-scored users under a budget and, on Hillstrom, with the two e-mails as separate arms, solves a three-arm choice under assumed costs exactly and by its Lagrangian dual.

The repository holds the registration, the code, tests on synthetic data and, from the runs, trial records and aggregate results; never data, predictions or model files (Section 2). Where the registration leaves a detail open, the code committed before the search fixes it, and in most cases a docstring or comment at the choice names the section.

## Results

None yet. Each dataset's test rows are read once (an interrupted read aside, see How to rerun), after its models are frozen and a dry run on validation rows has computed every output; the dry run, in `ranking/results/DATASET-dry-run.json`, computes the labels (the outcomes of the decision rules of Sections 7 and 8), which enter no verdict (Section 4). The read writes `ranking/results/DATASET.json`: every label of Sections 7 and 8 with the statistics behind it, including the intervals and seed values that its rule reads, the package versions, the resource figures, `commit` (the commit that holds the read entry), `command` (the stage that computed the labels at that commit; Section 15) and `second read` (Section 10).

## Method

- Units, the outcomes analysed (U1 to U4 in Section 2.2): Criteo with `visit` and with `conversion` (one search covers both), Lazada, and Hillstrom with both e-mails pooled against none (Section 2.2).
- Roles: a row's role (training, validation or test) depends only on a hash of its features, so identical feature vectors share a role; Lazada's test set is its released randomised test file (Section 2.1).
- Models: S, T, X, R and DR learners on LightGBM, an EconML causal forest, and TARNet, DESCN and EFIN written in torch from the papers' descriptions. Each is tuned by validation R-loss (Section 5.4), never by an uplift metric, on 20 configurations drawn from its grid with a fixed seed (all 9 for the forest), and the chosen one is fitted with seeds 0 to 4. R*, Q2's best non-deep model, has the smallest validation R-loss of the six (Section 6).
- Statistics: fourteen versions of the Qini, uplift and RATE statistics, V1 to V13 and V2* (a re-derivation of V2). Six call causalml or scikit-uplift; the other eight are written from the formulas that Section 5.2 cites and are the only ones with intervals.
- Intervals: test rows with identical features form a group, and each of 2,000 half-sample replicates draws half of the groups and recomputes every statistic that gets an interval, with the models fixed. A difference is separated when its simultaneous 95% interval (max-|t| over the family of comparisons that Section 7 sets) excludes zero and seeds 0 to 4 agree on its sign. The minimum detectable difference is 2.80 standard errors (two-sided 5% level, 80% power; Section 5.5).
- Labels: Sections 7 and 8 state the propositions H1 to H7 and the rules that turn statistics into labels.

## How to rerun

This needs Linux on x86-64, Python 3.13, uv 0.11.32, a clone whose remote `origin` is a repository you can push to, with a `main` branch, disk for the downloads and the environment, and up to 25 GB per dataset for the frozen models, above which `block` stops. Each stage first checks Python and the package versions of Section 12, and stops on any other version before it downloads data, fits a model or reads a test row. From `ranking/`:

```
uv sync --locked --group test
uv run --no-sync pytest -q
```

The steps of each dataset's run, how an interrupted read is rerun, how the registration is amended and the files the runs write are in [`ranking/RERUN.md`](ranking/RERUN.md).

## Limits

Section 13 lists the limits of the design. In addition:

- The test rows are sealed by convention only: in the code only `go` opens them, but nothing stops another program from reading the downloaded files. The stop checks of Section 2 use whole-file counts and means, test rows included, and the mean printed when a mean check fails is the whole-file value.
- The lock file covers only Linux x86-64 and pins the CPU build of torch, and the half-sample replicates of H1 to H5 run in forked worker processes.
- Tie order: V4 and the causalml versions order tied scores by a fixed random permutation of the test rows. causalml's own sort is unstable and orders ties differently on different CPU instruction sets, so causalml receives a tie-free score in V4's order; the other versions treat a tie block as one (Sections 5.1, 5.2).
- No tolerance is set for floating-point differences between machines (Section 11), and HiGHS, the solver of the three-arm choice, uses its default thread count, so its solve times vary.

Licence of the code: MIT ([`LICENSE`](LICENSE)). The data keep their own terms (Section 2).
