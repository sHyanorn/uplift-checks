# Uplift checks

A pre-registered study of how uplift models are compared, on three public datasets with randomised test sets: Criteo Uplift v2.1, Lazada and Hillstrom. The registration, [`ranking/PREREGISTRATION.md`](ranking/PREREGISTRATION.md), fixes the questions, data, models, statistics and decision rules before any model is evaluated. "Section" below means a section of that file.

Lazada and Hillstrom have been read; Criteo has not been run. On both, H2 separates none of the 36 pairs, and H3 finds each deep model, as implemented here, "undecided" against the best non-deep one. On Lazada both literature groups are "most published gaps below the detectable difference" (H4).

## What it is

The study asks (Section 1):

- Q1. Given the same predictions, do different implementations and definitions of the Qini and uplift curves and of the rank-weighted average treatment effect (RATE) rank the models alike?
- Q2. Of the differences between nine tuned models, how many can paired intervals tell from zero, and do the deep models beat the best non-deep model?
- Q3. What is the smallest difference each test set detects, and how many differences reported in four papers (Appendix A) fall below it at their test sizes?

A smaller part (Section 8) estimates the value of treating the top-scored users under a budget and, on Hillstrom, with the two e-mails as separate arms, solves a three-arm choice under assumed costs exactly and by its Lagrangian dual.

The repository holds the registration, the code, tests on synthetic data and, from the runs, trial records and aggregate results; never data, predictions or model files (Section 2). Where the registration leaves a detail open, the code committed before the search fixes it, and in most cases a docstring or comment at the choice names the section.

## Results

Lazada and Hillstrom were each read once, on 2026-11-24, after read entries of 06:48 and 07:13 UTC (`ranking/records/reads.jsonl`); neither entry reruns an interrupted read or is a second read. Criteo has not been run: U1 and U2, the groups L1 Criteo, L2 and L4 and the published gains on U1 have no results.

[`ranking/RESULTS.md`](ranking/RESULTS.md) describes what a read writes and tables every label of the two reads with its statistic and, for H3, H5 and H7, its interval. Figures are rounded to three significant figures here; R* is DR on both units.

- Q1, H1. H1a is "no separated pair; 8 reversals of 36" on Lazada and "no separated pair; 1 reversals of 36" on Hillstrom, and the three H1b readings are "no separated pair; r reversals of 36" on both. Section 7 then makes no statement of agreement (H1a), and no change in the preferred model is shown (H1b).
- Q2, H2. "mostly indistinguishable" on both units: N = 0 of 36 pairs, with a median `MDD_sim` over the family of 72 of 0.00152 on Lazada's 181,669 test rows and 0.00849 on Hillstrom's 25,775; the marginal (unadjusted) count is 0 on Lazada and 5 on Hillstrom.
- Q2, H3. The result for TARNet, DESCN and EFIN, as implemented here, is "undecided" on both units: the V2* label of `M(D) - M(R*)`, read from the simultaneous 95% interval over the unit's family of 12 contrasts. That difference is 0.00557 [-0.0107, 0.0219], 0.00330 [-0.0139, 0.0205] and 0.000442 [-0.0156, 0.0164] on Lazada and 0.0145 [-0.0127, 0.0418], -0.00117 [-0.0296, 0.0273] and 0.0208 [-0.00736, 0.0489] on Hillstrom. Of the 24 contrasts, 23 are "undecided"; one, EFIN minus our T-learner under V2* on Hillstrom, is "gain shown": 0.0274 [0.000913, 0.0539]. It changes no result: a model's result reads only its contrasts with R*, and the label of a contrast with T is read only by the published-gain rule, and U4 has no published gain. On Lazada, DESCN's published gain is "published gain not tested at this size".
- Q3, H4. The median MDD of the 36 pairs under V2* is 0.0268 on Lazada and 0.0262 on Hillstrom. 78 of 78 gaps in Table 4 of Liu et al. (2024) on Lazada, and 6 of 7 between DESCN and the other models in Table 2 of Zhong et al. (2022), are smaller than the difference that a test set of their size detects with 80% power (`MDD_L`, from our nine models' pairs: 0.0266 at 181,669 rows, 0.0119 at 910,000). Both groups are "most published gaps below the detectable difference".
- Section 8, with marginal intervals in H5 and H7. Every H5 contrast with R* is "not distinguishable" (24 per unit, AIPW and IPW). All 9 H6 instances are Optimal with gap 0; 8 are "dual within 1e-4 of exact", and one (`c = 2`, `B` = 0.5 x test rows) is "dual below exact" (`|V_E - V_D| / V_E` = 0.000113 against 1e-4). Under the assumed costs only, the exact policy's estimated value is "higher" than a simple policy's in 5 of 18 comparisons, "lower" in 1 and "not distinguishable" in 12, under AIPW and IPW alike.

## Method

- Units, the outcomes analysed (U1 to U4 in Section 2.2): Criteo with `visit` and with `conversion` (one search covers both), Lazada, and Hillstrom with both e-mails pooled against none (Section 2.2).
- Roles: a row's role (training, validation or test) depends only on a hash of its features, so identical feature vectors share a role; Lazada's test set is its released randomised test file (Section 2.1).
- Models: S, T, X, R and DR learners on LightGBM, an EconML causal forest, and TARNet, DESCN and EFIN written in torch from the papers' descriptions. Each is tuned by validation R-loss (Section 5.4), never by an uplift metric, on 20 configurations drawn from its grid with a fixed seed (all 9 for the forest), and the chosen one is fitted with seeds 0 to 4. R*, Q2's best non-deep model, has the smallest validation R-loss of the six (Section 6).
- Statistics: fourteen versions of the Qini, uplift and RATE statistics, V1 to V13 and V2* (a re-derivation of V2). Six call causalml or scikit-uplift; the other eight are written from the formulas that Section 5.2 cites and are the only ones with intervals.
- Intervals: test rows with identical features form a group, and each of 2,000 half-sample replicates draws half of the groups and recomputes every statistic that gets an interval, with the models fixed. A difference is separated when its simultaneous 95% interval (max-|t| over the family of comparisons that Section 7 sets) excludes zero and seeds 0 to 4 agree on its sign. The minimum detectable difference is 2.80 standard errors (two-sided 5% level, 80% power; Section 5.5).
- Labels: Sections 7 and 8 state the propositions H1 to H7 and the rules that turn statistics into labels.

## How to rerun

This needs Linux on x86-64, Python 3.13, uv 0.11.32, a full clone (the read refuses a history without the registration commit) whose remote `origin` is a repository you can push to, with a `main` branch, 15 GiB of memory, disk for the downloads and the environment, and up to 25 GB per dataset for the frozen models, above which `block` stops. Each stage first checks Python and the package versions of Section 12, and stops on any other version before it downloads data, fits a model or reads a test row. From `ranking/`:

```
uv sync --locked --group test
uv run --no-sync pytest -q
```

The steps of each dataset's run, how an interrupted read is rerun, how the registration is amended and the files the runs write are in [`ranking/RERUN.md`](ranking/RERUN.md). The test rows of Lazada and Hillstrom are not read again: `go` refuses a dataset whose results are committed.

## Limits

Section 13 lists the limits of the design. In addition:

- The test rows are sealed by convention only: in the code only `go` opens them, but nothing stops another program from reading the downloaded files. The stop checks of Section 2 use whole-file counts and means, test rows included, and the mean printed when a mean check fails is the whole-file value.
- The lock file covers only Linux x86-64 and pins the CPU build of torch, and the half-sample replicates of H1 to H5 run in forked worker processes.
- Tie order: V4 and the causalml versions order tied scores by a fixed random permutation of the test rows. causalml's own sort is unstable and orders ties differently on different CPU instruction sets, so causalml receives a tie-free score in V4's order; the other versions treat a tie block as one (Sections 5.1, 5.2).
- No tolerance is set for floating-point differences between machines (Section 11), and HiGHS, the solver of the three-arm choice, uses its default thread count, so its solve times vary.

Licence of the code: MIT ([`LICENSE`](LICENSE)). The data keep their own terms (Section 2).
