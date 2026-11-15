# Pre-registration: rankings, intervals and power of uplift model comparisons on three randomised trials

## 1. Questions

Uplift models are compared by the area under a Qini or uplift curve computed on a randomised test set. Published comparisons differ in the curve, its normalisation, the software that computes it and the test size, and most report one number per model without an interval. On Criteo a T-learner is reported at a QINI of 0.1053 in one benchmark and at a Qini of 0.3503 in another, with different base learners and different definitions of the statistic (Section 3). This study asks three questions on three public randomised trials:

- Q1. Do the same predictions rank models alike under different implementations and definitions of the Qini and uplift curves and of the rank-weighted average treatment effect (RATE)?
- Q2. Of the differences between tuned gradient-boosted meta-learners (S, T, X, R, DR), a causal forest and three deep uplift models, how many can be told from zero with paired intervals, and do the deep models beat the best tuned non-deep learner?
- Q3. What is the smallest difference each test set detects, and how many differences reported in the literature fall below it at the reported test sizes?

A small module asks what model rankings mean for allocation under a budget: the value of treating the top fraction by each model's score, and on Hillstrom a three-arm choice under assumed costs solved exactly and by its Lagrangian dual. Its conclusions concern methods and solvers only.

A result of "cannot be told apart" answers Q2, and Q3 states the size of difference that could have been told apart.

## 2. Data and fixed inputs

| Dataset | File | Bytes | SHA-256 | Terms |
|---|---|---|---|---|
| Criteo Uplift v2.1 | `criteo-research-uplift-v2.1.csv.gz` from `https://huggingface.co/datasets/criteo/criteo-uplift`, revision `2424920019e49d52d72c13ac1143ec5d53af276b` (the official link `http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz` returned 404 on 2026-11-15) | 311,422,618 | `2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc` | CC BY-NC-SA 4.0 ("CRITEO DATA TERMS OF USE", Criteo AI Lab page) |
| Lazada (DESCN release) | `lzd_data_public.zip`, `https://www.dropbox.com/s/07r7592h9mfijsb/lzd_data_public.zip?dl=1`; members `full_trainset.csv` (`79cb6e5c2fb7ad0c35710a84ed82275baa167647b17bf3423f402602b02b5bf0`), `full_testset.csv` (`1eec5f10b838ca3a43f6ccaf1b7af9260e300be36df452be58ee6678d7e1a326`), `LICENSE` (`3ae95b8e3fa8f3cae1af3c44274586179e7dda9f3cddba6bc7565d249c7cc7a3`) | 101,629,017 | `a657288e9a54539008004b321fcf986737e94aeea8468b9d70b96f6a974e73db` | BSD 2-clause text in `LICENSE` |
| Hillstrom (MineThatData, 2008) | `http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv` | 3,964,977 | `0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece` | No licence text; the post invites analysis and sharing of findings |

- The MD5 sums printed in the Lazada README (`545bb0eae05cca58f0d813a990ed9cba` for the archive, `efffc70375e700bcb91a9d41eddd7f66` and `9058fa57b984f0112a1ecf71dc52fb1e` for the two files) match the downloaded files.
- The repository holds download code and checksums only. Data, predictions and any row-level derived file stay outside it.
- Excluded: X5 RetailHero (the site agreement of ODS.ai, clause 3.2, prohibits copying or storing site content without written permission of the rightsholder; if permission is obtained, X5 is added by a separate registration); MegaFon (synthetic data by the organisers' statement); MT-LIFT (no licence; download requires a login); Criteo v1 (the dataset page states that it has a leak).

Columns used:
- Criteo: features `f0` to `f11`, `treatment`, outcomes `visit` and `conversion`. `exposure` is not used.
- Lazada: features `f0` to `f82`, `is_treat`, `label`. `data_id` identifies rows only.
- Hillstrom: features `recency`, `history_segment`, `history`, `mens`, `womens`, `zip_code`, `newbie`, `channel`; `segment`; outcome `visit`. For encoding and hashing, each string column is replaced by the 0-based index of its value in the sorted list of distinct values in the file.

Stop checks, run on each dataset before anything else of that dataset; a failed check stops that dataset and is reported:
- Checksums as above.
- Criteo: 13,979,592 rows; 11,882,655 treated; mean of `visit` 0.046992 and of `conversion` 0.002917 to six decimals.
- Lazada: train 926,669 rows (205,480 treated), test 181,669 rows (94,682 treated), `data_id` unique.
- Hillstrom: 64,000 rows; `Mens E-Mail` 21,307, `Womens E-Mail` 21,387, `No E-Mail` 21,306.
- No missing value in a used column.

### 2.1 Roles of rows

A row's role depends only on its feature vector and a salt, never on an outcome or the treatment:
- Features are cast to float32: Criteo read by pandas 3.1.1 `read_csv` (C engine) with dtype float32 for `f0` to `f11`; Lazada and encoded Hillstrom columns read with default dtypes and cast. The role-count stop check guards the parsing. Let `b_j` be the 32 bits of feature `j` as an unsigned integer.
- `h = 20261115`; for `j = 0, 1, ..., d - 1`: `h = mix(h XOR (b_j + j x 0x9E3779B97F4A7C15))`, all arithmetic modulo 2^64, where `mix` is the splitmix64 finaliser (`h ^= h >> 30; h *= 0xBF58476D1CE4E5B9; h ^= h >> 27; h *= 0x94D049BB133111EB; h ^= h >> 31`).
- `u = (h >> 11) / 2^53`.
- Criteo: test if `u < 0.30`, validation if `0.30 <= u < 0.40`, otherwise training. Counts: 4,195,534 test, 1,398,844 validation, 8,385,214 training.
- Lazada: the randomised file `full_testset.csv` is the test set. In `full_trainset.csv`, validation if `u < 0.10`, otherwise training: 93,670 and 832,999 rows.
- Hillstrom: test if `u < 0.40`, validation if `0.40 <= u < 0.50`, otherwise training. Counts are committed by the first run; a later run whose counts differ stops.
- Rows with identical feature vectors always share a role. On Criteo no feature vector falls in two roles. On Lazada the split is given by the release, and 4,388 test rows have a feature vector that also occurs in the training file.

### 2.2 Units

| Unit | Data | Treated | Control | Outcome | Role |
|---|---|---|---|---|---|
| U1 | Criteo | `treatment = 1` | `treatment = 0` | `visit` | confirmatory |
| U2 | Criteo | as U1 | as U1 | `conversion` | secondary, no verdict |
| U3 | Lazada | `is_treat = 1` | `is_treat = 0` | `label` | confirmatory |
| U4 | Hillstrom | either e-mail | `No E-Mail` | `visit` | confirmatory |

The three-arm Hillstrom data enter the allocation module only (H6, H7). Each model is trained once per dataset per outcome: U1 and U2 have separate models.

## 3. What was seen before this registration

Everything below was known when the rules were written.

Data (counts and whole-file means; no outcome was summarised by treatment arm and no model was evaluated on any split of any dataset):
- The stop-check figures of Section 2 and the role counts of Section 2.1 for Criteo and Lazada, computed on the full files. Hillstrom's role counts have not been computed.
- Criteo: 1,655,747 rows repeat an earlier row in all columns; 2,120,151 rows repeat an earlier feature vector; 11,859,441 distinct feature vectors; the largest group of identical feature vectors has 24 rows and none has more than 100. Rows are not in random order: the first 3,341,564 training rows are all treated.
- Lazada: no row repeats another in all columns (`data_id` included); 22,715 training rows and 2,437 test rows repeat an earlier feature vector within their file; the training file's treated share is 22%, the test file's 52%.
- Hillstrom: overall mean of `visit` 0.14678, of `conversion` 0.00903, of `spend` 1.0509; 6,562 rows repeat an earlier row in all columns.
- Training time and peak memory of every model class on training rows of Criteo and Lazada. No loss or prediction was recorded.

Software (reading of source code, and checks on synthetic data only):
- causalml 0.18.0, `auuc_score`: cumulative gain over the joint ranking row by row, divided by its absolute final value, averaged over rows; ties are ordered by the sort, not grouped. `qini_score`: the Qini curve over the same ranking, divided by its absolute final value, minus the random line, averaged over rows. `rate_score`: `weighting="autoc"` weights the TOC curve by 1/q, where the RATE paper defines the AUTOC with α(u) = 1; `weighting="qini"` weights by q normalised to sum to 1, twice the paper's QINI. On synthetic data with known effects, two rules ranked a > b under causalml's `"autoc"` (0.06922 against 0.01144) and b > a under the paper's AUTOC (0.00501 against 0.00661). Its half-sample bootstrap takes the standard deviation of half-sample estimates without rescaling, as grf does.
- scikit-uplift 0.5.1 (last commit 2022-08-11): `uplift_auc_score` and `qini_auc_score` evaluate the curves at distinct score values (ties grouped) and divide the area above the random line by that of a perfect curve built from the test labels. A re-derivation of `qini_auc_score` matched the library to 0.0 on 20 synthetic sets with heavy ties.
- Normalisations by a positive constant that is the same for every model on one test set cannot change the order of models on that test set.
- EconML 0.17.0: `CausalForestDML` and `econml.grf.CausalForest` with `n_jobs > 1` returned different predictions on two runs with the same `random_state`; with `n_jobs = 1` repeated runs were identical. With `DummyClassifier(strategy="prior")` as treatment model, `CausalForestDML` raised a shape error.
- LightGBM 4.7.0 with `deterministic=True` and `force_row_wise=True` gave identical predictions on repeated runs and with 2 and 4 threads. Without bagging and with `feature_fraction` 1, its seed did not change the fit on 30,000 rows and did on 400,000. torch 2.15.0 (CPU) with deterministic algorithms gave identical results on repeated runs with 4 threads and different results with 2 threads.

Literature (Appendix A lists every figure with page, table and row):
- Liu et al. (2024), benchmark of 13 deep uplift models: Criteo split 8:1:1 at random, Lazada as released; "with instance deduplication" without a definition of the key or the order relative to the split; seeds 0 to 9 "when performing random splits", without saying whether Table 4 holds means; QINI from scikit-uplift; hyperparameters searched "using QINI as the main evaluation metric". On Lazada the validation QINI (0.4192 to 0.5181) is computed on the biased training file, where AUUC is not valid (Renaudin and Martin 2021); test QINI is 0.0078 to 0.0252.
- Liu et al. (2023), EFIN: Criteo split 8/2, no validation split named; QINI from scikit-uplift; significance from a two-sample t-test without the number of runs.
- Zhong et al. (2022), DESCN: Table 2 is computed on a production dataset of 4.17M training and 0.91M test rows (their Table 1), of which the released Lazada file is a sample; each experiment repeated 5 times, mean and standard error reported; the metric in the released code is scikit-uplift's `qini_auc_score`.
- Singh (2026): Criteo v2.1, 80/20 stratified split, LightGBM meta-learners; the causal forest evaluated on a 10% stratified subsample of about 280K rows; Qini defined as the integral of G(φ) - φG(1), G the fraction of incremental conversions captured.
- Diemert et al. (2021), on Criteo v2: "For the test sizes up to 1M, all presented methods are indistinguishable by their AUUC score as their confidence intervals overlap almost entirely. However, starting from 1M points onwards one can separate approaches and perform model selection." Their intervals use a test-set bound; numbers appear only in a figure.
- Yadlowsky et al. (2021, version 2): RATE, its central limit theorem and the half-sample bootstrap. Their Criteo example states the version-1 row count (25,309,483) and reports arm rates (their Table 4: visit 0.0485 and 0.0382) that match version 2.
- Li (2026), UpliftBench: Hillstrom in full with either e-mail against none and `visit`; Criteo as a 10,000-row subsample of a 1M-row tier; no deep model.

## 4. Sample roles and order of work

- Training rows fit models; validation rows select hyperparameters (Section 6) and are read as often as needed. Hyperparameter selection reads only the committed trial records of Section 11. No uplift metric is used for selection.
- The test rows of a dataset are read once (Section 10), after every model that the registered outputs of that dataset use has been frozen. That one read computes every registered output of the dataset: H1 to H7 and U2.
- Order: this registration is committed first; then code, tested on synthetic data only, with passing continuous integration; then, per dataset, the search, which may span containers and whose trial records are committed to the repository, and in one container the final fits, freezing, a full dry run on the validation rows and the test read (Section 11). Before the code stage, nothing beyond the counts and timings of Section 3 is computed on real data.
- The dry run computes every output on the validation rows of each dataset with the frozen models. It enters no verdict and is reported. A change made after it is an amendment committed before the test read; a change to a frozen item then reruns the final block from its start, and a change to evaluation code only reruns the dry run.

## 5. Definitions

### 5.1 Notation and ranking

For a set of `n` rows: treatment `W`, outcome `Y`, a model's score `S` (its predicted uplift). Rows are ranked by `S`, largest first. Rows with equal `S` form a tie block, and a block enters a curve as a whole: every curve is evaluated at the ends of blocks and joined linearly between them, unless a version below says otherwise. `N_T(k)`, `N_C(k)`: treated and control rows among the first `k`; `Y_T(k)`, `Y_C(k)`: their sums of `Y`. A ratio with a zero denominator is taken as 0, as in scikit-uplift. `ê`: share of treated rows in the set evaluated (recomputed in each replicate).

### 5.2 Metric versions

All versions are computed on the same predictions. Versions marked "own" are written from the formula cited and have intervals (Section 5.5); library versions are called as named, at the pinned versions, with default arguments unless stated.

| Id | Version | Definition |
|---|---|---|
| V1 | causalml `qini_score` | causalml 0.18.0, `normalize=True` |
| V2 | scikit-uplift `qini_auc_score` | scikit-uplift 0.5.1, `negative_effect=True` |
| V2* | own, scikit-uplift Qini | exact re-derivation of V2 (equal to V2 to 1e-12 on synthetic data; required by a test) |
| V3 | own Qini | Qini curve `Q(k) = Y_T(k) - Y_C(k) N_T(k)/N_C(k)` (Radcliffe 2007), joint ranking; statistic `∫_0^1 [q(φ) - φ q(1)] dφ` with `q(φ) = Q(φn)/n` |
| V4 | own Qini, rows ordered | as V3, ties not grouped but ordered by a fixed random permutation of the test rows, `numpy.random.default_rng([20261115, 4, unit]).permutation(n)` |
| V5 | own, Singh's Qini | `∫_0^1 [G(φ) - φ G(1)] dφ`, `G(φ) = Q(φn)/Q(n)`; equal to V3 divided by `q(1)`, so it orders models as V3 does when `q(1) > 0`; used for the units of L4 only |
| V6 | own RATE, QINI | `(1/n) Σ_k (k/n) TOC(k)`, `TOC(k) = mean of Γ over the first k - mean of Γ`, Γ the AIPW scores of 5.3, tie blocks replaced by their mean Γ as in grf |
| V7 | causalml `auuc_score` | causalml 0.18.0, `normalize=True` |
| V8 | scikit-uplift `uplift_auc_score` | scikit-uplift 0.5.1 |
| V9 | own uplift, joint | `U(k) = (Y_T(k)/N_T(k) - Y_C(k)/N_C(k)) k`; statistic `∫_0^1 [U(φn)/n - φ U(n)/n] dφ` |
| V10 | own uplift, separate | Devriendt et al. (2020) and Diemert et al. (2021): treated and control rows ranked separately; `V(p) = R_T(p)/\|T\| - R_C(p)/\|C\|` with `R_T(p)` the responders among the top `p\|T\|` treated rows (a partly included tie block counts in proportion); statistic `(1/100) Σ_{p=1..100} [V(p/100) - (p/100) V(1)]` |
| V11 | own RATE, AUTOC | `(1/n) Σ_k TOC(k)` (Yadlowsky et al., α(u) = 1) |
| V12 | causalml `rate_score`, `"autoc"` | causalml 0.18.0, `weighting="autoc"`, outcome and treatment columns (no effect column) |
| V13 | causalml `rate_score`, `"qini"` | causalml 0.18.0, `weighting="qini"`, as V12 |

For V1, V7, V12 and V13 the input frame has one score column that holds no ties: in the ranking of V4 (largest score first, ties in the order of V4's permutation) the first row gets `n`, the second `n - 1` and the last 1. causalml 0.18.0 uses the score only through pandas' default sort, which is not stable and orders tied rows differently on different CPU instruction sets; with this column the four versions order tied scores as V4 does on every machine.

Estimands. In a randomised trial V1 to V10 and V13 estimate the same functional, the QINI of Yadlowsky et al., `∫_0^1 u TOC(u) du`, up to a positive factor (for V5, when `q(1) > 0`) and, for V7, an added constant, both the same for every model on one test set: V9 is that functional with difference-in-means TOC; V6 with AIPW scores; V8 and V13 rescale V9, and V7 rescales V9 and adds a constant; V3 and V4 equal it times the treated share up to the variation of that share along the ranking; V1 rescales V4 and V2 rescales V3; V10 ranks the arms separately, which in a randomised trial targets the same quantiles. They differ in estimator, ranking, ties and normalisation. V11 (AUTOC, α(u) = 1) and V12 (α(q) = 1/q) are different functionals.

### 5.3 Outcome models and scores

- Outcome models: per dataset and outcome, one LightGBM classifier per arm (two arms; three on Hillstrom for Section 8), fitted on training rows of that arm with fixed settings: `num_leaves=63`, `learning_rate=0.05`, `min_child_samples=100`, `feature_fraction=0.8`, `bagging_fraction=0.8`, `bagging_freq=1`, up to 1,000 trees with early stopping after 50 rounds on validation log-loss of the same arm, `seed=0`. `μ̂_w(x)` is the predicted probability for arm `w`. On U4, `μ̂_1` is fitted on the rows of both e-mail arms.
- AIPW scores: `Γ = μ̂_1 - μ̂_0 + W (Y - μ̂_1)/ê - (1 - W)(Y - μ̂_0)/(1 - ê)`. IPW scores drop the `μ̂` terms.
- The outcome models are part of the frozen set (Section 10).

### 5.4 R-loss

On validation rows: `R(τ̂) = mean of [(Y - m̂(X)) - (W - e(X)) τ̂(X)]^2`, with `m̂` a LightGBM classifier of `Y` on `X` fitted on training rows with the settings of 5.3, and `e(X)` the training share of treated rows on Criteo and Hillstrom, and on Lazada a LightGBM classifier of `W` on `X` with the settings of 5.3, fitted in two folds on training rows, clipped to [0.01, 0.99]; on validation rows `e` is the mean of the two fold models.

Folds: wherever two folds are used (5.4 and the X, R and DR learners), fold 1 holds the training rows with `u < 0.5` under the hash of Section 2.1 with salt `20261116 + s`, `s` the fit's seed; identical feature vectors share a fold. Within a learner, `e` is the training share on Criteo and Hillstrom and the cross-fitted classifier on Lazada.

### 5.5 Resampling, intervals and detectable differences

- Groups: test rows with identical feature vectors form a group (most groups are single rows). Groups are numbered by their first row in file order.
- Half-sample replicate `r` (`r = 0, ..., 1,999`) of unit `u` (U1 to U4 numbered 1 to 4): the groups `numpy.random.default_rng([20261115, 1, u, r]).choice(G, G // 2, replace=False)`. Every statistic, including `ê`, the AIPW scores' `ê` and the perfect curve of V2*, is recomputed on the replicate; models, outcome models and scores' `μ̂` stay fixed.
- For a statistic `d` (a version's value or a paired difference of two models' values on the same rows), `SE = standard deviation of the 2,000 replicate values` (ddof 1). This is the half-sample bootstrap of Yadlowsky et al. (Lemma 4) and grf; the replicate spread estimates the full-sample spread without rescaling.
- Marginal 95% interval: `d ± 1.96 SE`. Simultaneous 95% interval over a family `P`: `d_p ± c* SE_p` with `c*` the 95th percentile (linear interpolation) over replicates of `max_{p in P} |d*_{r,p} - d_p| / SE_p`. A difference with `SE_p = 0` (equal in every replicate) leaves the maximum and is reported as "identical" if it is 0.
- Seed rule: seed `s` of one model is paired with seed `s` of the other (`s = 0, ..., 4`, Section 6), giving five point values of a paired difference on the full test set. The rule holds if all five have the sign of the seed-0 value. For each model the run reports whether its five fits gave distinct predictions.
- Separated pair: its simultaneous interval excludes 0 and the seed rule holds.
- Minimum detectable difference of a paired difference: `MDD = 2.80 SE` (two-sided 5% level, 80% power, normal approximation); `MDD_sim = (c* + 0.84) SE` within a family. At another test size `n'`: `MDD x sqrt(G / n')`, with `G` the number of resampling groups of our test set.

## 6. Models and tuning

Nine models per unit:
- LightGBM 4.7.0 meta-learners (every component with `bagging_fraction=0.8`, `bagging_freq=1`, `seed` = the fit's seed, so that seeds change the fit): S (one classifier on `X` and `W`; effect = prediction at `W = 1` minus at `W = 0`), T (one classifier per arm), X (Künzel et al. 2019: T stage, then regressors of imputed effects per arm, combined with `e(X)`), R (Nie and Wager 2021: regressor of the R-loss pseudo-outcome with weights `(W - e)^2`, nuisances `m̂` and `e` cross-fitted in two folds), DR (Kennedy 2023: regressor of the AIPW pseudo-outcome, nuisances cross-fitted in two folds).
- Causal forest: EconML 0.17.0 `CausalForestDML` (outcome model LightGBM as in 5.3, treatment model `LogisticRegression`, two folds as drawn by EconML, `n_jobs=1`, `random_state` = the fit's seed). On Criteo it is fitted on 1,000,000 training rows drawn by `numpy.random.default_rng([20261115, 3]).choice(8385214, 1000000, replace=False)`, indices into the training rows in file order; on Lazada and Hillstrom on all training rows. Reason: memory measured on Criteo with 4 jobs and a default LightGBM outcome model (2.0 GB at 1M rows and 100 trees; 3.45 GB at 500 trees with `min_samples_leaf` 2000), growing with rows and trees.
- Deep models, written from the papers' descriptions in torch (no code taken from the authors' repositories):
  - TARNet (Shalit et al. 2017): shared representation of three ELU layers of width `h`; one head per arm of two ELU layers of width `h/2`; binary cross-entropy on the factual outcome.
  - DESCN (Zhong et al. 2022, Eq. 7): shared layers as TARNet; propensity, treated-response, control-response and pseudo-effect heads; loss = binary cross-entropy of `π` against `W`, of `π μ_1` against `W Y`, of `(1 - π) μ_0` against `(1 - W) Y`, plus `α` times the cross losses (`sigmoid(logit μ_0 + τ')` against `Y` on treated rows, `sigmoid(logit μ_1 - τ')` against `Y` on control rows). The factual response losses are left out, as in Eq. 7. One weight `α` replaces the paper's separate weights of the cross terms, and the entire-space terms have weight 1.
  - EFIN (Liu et al. 2023): each feature embedded as `w_j x_j + b_j` in `h` dimensions (EFIN's "embedded dimension"); a self-attention block and an MLP `[h, h/2]` give the control response; a treatment embedding attends over the feature embeddings and an MLP `[h, h/2]` gives the uplift; the treated response is the control response plus the uplift in logit space; the intervention-constraint head predicts the treatment from the treatment-aware representation and is trained against the inverted treatment label with weight `α`.
  - In all three, `h` is the "rank" of the grid below.
  - Inputs standardised with training means and standard deviations or left as they are, a grid choice (Liu et al. 2024 report both). AdamW; up to 20 epochs; early stopping on validation factual loss with patience 5; batch order from a seeded `torch.Generator`; predictions in chunks of 50,000 rows.

Tuning:
- Selection: every model by validation R-loss of its predicted effect (5.4). In the search every configuration is fitted with seed 0. For the deep models, epochs are chosen by early stopping before the R-loss is read.
- Budget: 20 configurations per model per unit. The grid of a model is enumerated as the Cartesian product of its parameters in the order listed below, values in the order listed (the last parameter varying fastest), and configurations `numpy.random.default_rng([20261115, 2, unit, model index]).choice(size of grid, 20, replace=False)` are run, model index in the order S, T, X, R, DR, causal forest, TARNet, DESCN, EFIN (0 to 8). The causal forest grid has 9 configurations; all are run.
- LightGBM grid (every component of a learner shares one draw; 360 configurations): `num_leaves` {15, 31, 63, 127, 255}; `learning_rate` {0.05, 0.1}; `min_child_samples` {20, 100, 500, 2000}; `feature_fraction` {0.6, 0.8, 1.0}; `lambda_l2` {0, 1, 10}; up to 1,000 trees, early stopping after 50 rounds on the validation loss of each component.
- Causal forest grid: `min_samples_leaf` {500, 2000, 5000} x `max_features` {0.5, 0.8, 1.0}; 200 trees in the search, 500 in the final fits.
- Deep grid (Table 3 of Liu et al. 2024, plus input standardisation): rank `h` {32, 64, 128}; batch {256, 512, 1024, 2048}; learning rate {1e-4, 5e-4, 1e-3, 5e-3, 1e-2}; weight decay {1e-5, 1e-4, 1e-3, 1e-2}; `α` {0.1, 0.2, ..., 0.9}; standardise inputs {no, yes}. 4,320 configurations for DESCN and EFIN; TARNet has no `α`, 480.
- Final fits: the selected configuration of each model is fitted with seeds 0 to 4. A seed sets LightGBM's `seed` and the fold salt, EconML's `random_state`, and torch's initialisation and batch order. Seed 0 gives the values of every proposition; seeds 1 to 4 enter the seed rule only.
- Reference learner `R*` of a unit: the one of S, T, X, R, DR and causal forest with the smallest validation R-loss at seed 0 (ties to the earlier in that order), fixed before the test read.

## 7. Main propositions

Terms used in Sections 7 and 8 (defined in Sections 2.2, 5 and 6): units U1 (Criteo, `visit`), U2 (Criteo, `conversion`), U3 (Lazada), U4 (Hillstrom, either e-mail against none); the nine models S, T, X, R, DR, causal forest, TARNet, DESCN, EFIN, and `R*`, the non-deep model with the smallest validation R-loss; versions V1 to V13 and V2* (Section 5.2), of which the own versions V2*, V3, V4, V5, V6, V9, V10 and V11 have half-sample intervals; `M(D)`: a version's value for model D on the unit's test rows at seed 0; seed rule: the five fits (seeds 0 to 4) give five values of a difference, all with the sign of the seed-0 value; separated in a family: the simultaneous 95% interval over that family (max-|t|, Section 5.5) excludes 0 and the seed rule holds; MDD of a difference: 2.80 times its half-sample standard error (two-sided 5%, power 80%), and `MDD_sim = (c* + 0.84) SE` within a family, rescaled to another test size `n'` by `sqrt(G / n')` with `G` the number of resampling groups of our test set; `Γ`: AIPW scores of Section 5.3. Every label below is reported with the statistic behind it and, where its rule uses one, the interval.

### H1. Implementations and definitions order models alike (Q1)

- Units U1, U3, U4. Inputs: seed-0 test predictions of the nine models.
- Same ordering: two versions order a pair of models the same way if their differences have the same sign (positive, negative or zero). Reversal: both differences nonzero and of opposite sign.
- Families: for each own version, its 36 pairs. A pair is separated under an own version if it is separated in that version's family.
- H1a, one estimand. Group: V1, V2, V3, V4, V6, V7, V8, V9, V10, V13, which estimate the same QINI functional (Section 5.2); own versions in it: V2* (whose values are those of V2), V3, V4, V6, V9, V10. Label per unit, rules applied in order:
  1. "identical orderings" if every version of the group orders each of the 36 pairs the same way;
  2. "contradictory separations" if a pair is separated in one direction under one own version and in the other direction under another;
  3. "a separated pair reversed" if a pair separated under an own version is reversed by the point value of another version (own or library); the versions and models are named;
  4. "no separated pair; r reversals of 36" if no pair is separated under any own version, with `r` the number of model pairs reversed between some two versions;
  5. "agree on every separated pair" otherwise.
- H1b, different estimands. Pairs of versions (V6, V11), (V11, V12), (V6, V12): QINI against AUTOC, AUTOC against causalml's `"autoc"`, QINI against `"autoc"`. Each pair gets a label by the same rules; V12 has no interval, so its pairs use the other version's separation.
- Reported with it, no rule: Kendall's τ_b between every two versions among V1 to V13 except V5; the reversed pairs; per model, the share of test rows in tie blocks of more than one row and the treated share within those blocks.
- Consequences:
  - H1a: "identical orderings" or "agree on every separated pair": the implementations can stand in for each other when ranking these models on that unit. "Contradictory separations" or "a separated pair reversed": the named implementation changes a resolved ordering of the same quantity; rankings computed with it are not comparable with the others on that unit. "No separated pair": no statement of agreement.
  - H1b: "contradictory separations" or "a separated pair reversed": the choice between the two weightings changes which model is preferred where the difference is resolved. Otherwise no such change is shown.
- Known at registration: in a randomised trial the H1a versions differ in estimator, ranking, ties and normalisation only; V5 is V3 divided by `q(1)` and is left out; V12 and V11 weight the TOC curve differently, and in the synthetic example of Section 3 they ordered two scoring rules oppositely.

### H2. How many differences can be told apart (Q2)

- Units U1, U3, U4. Metrics V6 (QINI) and V11 (AUTOC). Family per unit: the 36 pairs under both metrics, 72 differences.
- A pair counts as separated if it is separated in this family of 72 under at least one of the two metrics. `N` is the number of the 36 pairs separated.
- Label per unit: "mostly indistinguishable" if `N <= 9`; "mixed" if `10 <= N <= 26`; "mostly distinguishable" if `N >= 27`.
- Reported with it, no rule: `N` under each metric alone, in the same family of 72; the count among the 10 pairs of meta-learners and among the 18 pairs of a deep and a non-deep model; marginal (unadjusted) counts; every interval.
- Consequence: the label and `N` of 36 are reported per unit with the test size and with the median `MDD_sim` of this family and the median marginal MDD over the 36 pairs under V6 and under V11. Each proposition uses its own family (H1: 36 per version; H2: 72; H3: 12), so a pair can be separated in one and not in another; each count names its family.
- Expected from Section 3: separation is possible on Criteo above 1M test rows (Diemert et al.); the design-stage MDD (Appendix B) makes "mostly indistinguishable" likely on U3 and U4.

### H3. Deep models against the best tuned non-deep learner (Q2)

- Contrasts per unit, for D in {TARNet, DESCN, EFIN}: `Δ_D = M(D) - M(R*)` under V2*, V6 and V11, and `Δ'_D = M(D) - M(T)` under V2*, with T our LightGBM T-learner. Family per unit: these 12 differences (if `R* = T`, the duplicated contrasts stay in the family).
- Published gain `g` (Table 4 of Liu et al. 2024, test QINI of D minus that of their T-learner, where positive): U1 TARNet 0.0010, U1 EFIN 0.0167, U3 DESCN 0.0021. No other model and unit has a positive published gain; on U4 there is none.
- Label of `Δ_D` per metric, rules in order, `[L, U]` the simultaneous interval:
  1. "gain shown" if `L > 0` and the seed rule holds;
  2. "loss shown" if `U < 0` and the seed rule holds;
  3. "undecided", reported with the contrast's MDD in this family, `(c* + 0.84) SE`, and its marginal MDD.
- The result for D on a unit is the V2* label. The V6 and V11 labels are reported beside it; if one of them gives "gain shown" where V2* gives "loss shown", or the reverse, "metric-dependent" is added to the result. Other differences between metrics are reported without a rule.
- Published gain, where `g` exists: with `SE'` the standard error of `Δ'_D` and `SE_g = SE' x sqrt(G / n_L)` the standard error of a difference of the same kind at the published test size (`n_L` of H4), `z = (g - Δ'_D) / sqrt(SE'^2 + SE_g^2)`, and `c*` the critical value of this family. Label: "published gain contradicted" if `z > c*` and all five seed values of `Δ'_D` are below `g`; "published gain reproduced" if `Δ'_D` is labelled "gain shown" by rule 1 and `-c* <= z <= c*`; otherwise "published gain not tested at this size".
- Consequence: each label is reported per model and unit, stating that the main comparison is with the best non-deep learner chosen by validation R-loss, that tuning used R-loss and not Qini, that the deep models are implementations from the papers' descriptions with the deviations of Section 13, and on U1 that the causal forest was fitted on 1M rows. "Published gain contradicted" means our estimate of D over a T-learner and the published one differ by more than both test sets' sampling error allows; it does not say which is right.

### H4. Detectable differences and published gaps (Q3)

- For each unit and own version (including V2* and V5), the MDD of each of the 36 pairs; reported: median, minimum and maximum.
- Literature groups (Appendix A), each with its version and test size `n_L`:

| Group | Gaps | Version | `n_L` |
|---|---|---|---|
| L1 Criteo | 78 pairs of 13 models | V2* on U1 | 1,397,959 |
| L1 Lazada | 78 pairs of 13 models | V2* on U3 | 181,669 |
| L2 | EFIN minus each of 11 models | V2* on U1 | 2,795,918 |
| L3 | DESCN minus each of 7 models | V2* on U3 | 910,000 |
| L4 | S, X, T: 3 pairs | V5 on U1 | 2,795,918 |

  - `n_L` for L1 Criteo is 10% of all rows: the paper does not say how deduplication changed the count, and the largest plausible size gives the smallest MDD. Our side uses `G`, the number of resampling groups of the test set (Section 5.5), not rows. L2 and L4 are 20% of all rows. L3 is the 0.91M test rows of DESCN's Table 1, a larger sample than the released file. L4 leaves out the causal forest, evaluated on a different test subsample.
- `MDD_L` = median over our 36 pairs of `MDD x sqrt(G / n_L)` in the group's version on its unit.
- Statistic: `s_L`, the share of the group's gaps with absolute value below `MDD_L`.
- Label per group: "most published gaps below the detectable difference" if `s_L > 0.5`; otherwise "half or fewer below".
- Reading B for L1 Criteo only (Table 4 values as means over ten random 8:1:1 splits): ten random test sets of 10% overlap by about 10% pairwise, so the mean has about 0.19 times the variance of one test set, and `MDD_L x 0.44` is used; reported beside reading A with its own share; the label uses reading A. On Lazada the test file is the same for every seed, so no reading B applies.
- Reported with it, no rule: `s_L` with the minimum and maximum of our 36 MDDs in place of the median; `s_L` with the MDD of the matching pair of our models where one exists (L4: S, X, T; L1 and L2: S or T against TARNet, DESCN or EFIN, and S against T); for L3, the same with the released test size, `n_L = 181,669`. Every group's gaps are complete in Appendix A (78, 78, 11, 7, 3).
- Consequence: "x of y gaps reported in [source] are smaller than the difference that a test set of that size detects with 80% power", per group, with the caveat that the MDD comes from our nine models' pairs.

## 8. Secondary propositions

### H5. Value under a budget on one treatment

- Units U1, U3, U4; `k` in {0.1, 0.3, 0.5}. For a model, the treated set is the top `k n` rows by score (a tie block at the cut enters in proportion). Value over treating none: `V_k = (1/n) Σ_i a_i Γ_i` with `a_i` the inclusion weight, under AIPW and under IPW scores.
- Contrasts `V_k(D) - V_k(R*)` for each other model D at seed 0; marginal 95% intervals `[L, U]`. Label per contrast and estimator: "higher" if `L > 0`, "lower" if `U < 0`, otherwise "not distinguishable". The AIPW label is the result; the IPW label is reported beside it. The intervals are not adjusted for the number of contrasts (8 models, 3 budgets, 3 units).
- Reported, no rule: the median over models of the ratio of AIPW to IPW interval widths.
- Consequence: per unit and `k`, whether the models separated in H2 and H3 also differ in value at that budget; no statement on profit.

### H6. Exact and dual solutions of the three-arm choice on Hillstrom

- Test rows of Hillstrom (the test rows of U4); arms 0 none, 1 men's e-mail, 2 women's e-mail. Value `v_ij = μ̂_j(x_i) - μ̂_0(x_i)` from the three-arm outcome models of 5.3 (`visit`, test features only). Assumed costs: `c_0 = 0`, `c_1 = 1`, `c_2 = c` in {0.5, 1, 2}. Budget `B` in {0.1, 0.3, 0.5} x number of test rows. Nine instances.
- Problem: maximise `Σ_i Σ_j v_ij x_ij` subject to `Σ_j x_ij = 1` for every user, `Σ_i Σ_j c_j x_ij <= B`, `x_ij` in {0, 1}.
- Exact: HiGHS 1.16.0 MILP, `mip_rel_gap = 1e-6`, time limit 1,800 s; the incumbent value `V_E`, the model status and the gap between incumbent and dual bound are recorded. "Certified" means model status Optimal.
- Dual: bisection on λ over 60 halvings of `[0, max v / min positive cost]`; at each midpoint each user takes `argmax_j (v_ij - λ c_j)`, ties to the lower arm index; if the total cost exceeds `B` the lower end moves to the midpoint, otherwise the upper end does. The assignment at the final upper end is budget-feasible; its value is `V_D`.
- Label per instance, rules in order: "exact not certified" if the status is not Optimal; "dual within 1e-4 of exact" if `|V_E - V_D| <= 1e-4 max(V_E, 1e-12)`; "dual below exact" otherwise. `V_E >= 0` always (assigning arm 0 to everyone is feasible). The bound `max_i max_j v_ij / V_E` is reported with it.
- Consequence: the dual solution is described as matching the exact one only on instances labelled "dual within 1e-4 of exact".

### H7. Estimated value of the knapsack policy

- For each instance of H6, the estimated value of the exact policy (the HiGHS incumbent, certified or not, with its status reported) against two simple policies with the same budget. Let `a` be the e-mail arm with the larger mean of `v_ij` over test rows (ties to arm 1). (b1) `min(n, floor(B / c_a))` users drawn by `numpy.random.default_rng([20261115, 5, instance]).choice(n, size, replace=False)` are sent arm `a`; (b2) users ranked by `R*`'s U4 score (seed 0; ties by row order) are sent arm `a` in that order while the budget allows.
- Estimators: three-arm IPW with the test arm shares and three-arm AIPW with `μ̂_j`. Marginal 95% intervals `[L, U]` of the differences. Label per baseline and estimator: "higher" if `L > 0`, "lower" if `U < 0`, otherwise "not distinguishable"; the AIPW label is the result, the IPW label is reported beside it.
- The policies use test features only; `μ̂_j` is fitted on training rows, so AIPW stays unbiased under the randomisation.
- Consequence: per instance, whether the exact policy's estimated value differs from each simple policy under the assumed costs.
- These statements hold under the assumed costs only.

U2 (Criteo `conversion`) receives the statistics of H1 to H5 without labels.

## 9. What the allocation module can and cannot say

It can say whether the exact and dual solutions agree on these instances and how long each takes, and whether a policy built from estimated three-arm effects has a different estimated value from simple policies under the stated costs. It cannot say anything about real costs, profit, other cost structures or other data. Values are visits because spend is heavy-tailed and its intervals would be wide.

## 10. Test reads and stop conditions

- Frozen set of a dataset: for each unit of the dataset, the nine models at seeds 0 to 4, the outcome models of 5.3 (three arms on Hillstrom), `m̂` and `e` of 5.4, `R*`, the commit of the trial records that the selection read (Section 11), the causal forest's row draw on Criteo, and every configuration and setting. A manifest lists their file hashes, the selected configuration indices and `R*`; it holds no row-level value and is committed before the read.
- The test rows are reachable only through one entry point. It checks that the commit of this registration is an ancestor of HEAD, that the working tree is clean, that the frozen files on disk match the committed manifest, and that the committed read record has no entry for the dataset; it then writes an entry (UTC time, commit, manifest hash), which is committed before any test row is read. A second call for the same dataset stops, except for the rerun of an interrupted read (below), which adds its own entry.
- The read computes every registered output of the dataset (H1 to H7, U2) from the frozen set and writes aggregate results only. No module reads the test rows again.
- If the process stops after the read entry and before its results are committed, it may be rerun with the same manifest hash; the rerun is recorded in AMENDMENTS.md as an interrupted run. If the frozen files are lost and refitting does not reproduce the manifest hash, the dataset's results come from a second read, reported as such.
- Stop conditions before the read: a failed check of Section 2; a role count other than in Section 2.1 (Criteo, Lazada); a package version other than in Section 12; a trial record whose fingerprint of `m̂` and `e` differs from that of the unit's first record (Section 11); a missing or non-finite validation prediction of a final fit or another frozen model; a fit that exceeds the container's memory or 25 GB of disk. A stop is reported and resolved by an amendment before the read.

## 11. Run procedure

- A dataset runs in two parts. The search (the seed-0 trials of Section 6) may span containers. Here and in Section 10, committed means pushed to the main branch of the repository. The final block runs in one container: the outcome models of 5.3, `m̂` and `e` of 5.4, the final fits of the selected configurations at seeds 0 to 4, the freeze, the dry run and the test read.
- Trial records: each finished search trial appends one line to `records/trials.jsonl` in the repository with the dataset, unit, model, configuration index, seed, validation R-loss (a double printed in full), the epochs or trees per component chosen by early stopping, wall time, package versions, CPU model, code commit, and the SHA-256 of the validation predictions of `m̂` and `e` (5.4) as a fingerprint. No row-level value is written. The file is committed at least before work moves to another container.
- A search container runs the checks of Section 2, refits `m̂` and `e`, stops if their fingerprint differs from the unit's first record (Section 10), and runs the trials that have no committed record. Records not committed when a container is lost are lost with it, and their trials run again. Model files and logs of the search stay on the container's disk outside the repository (`~/.cache/uplift-checks/<dataset>/`) and are not used later. Long runs start in the background with a foreground wait.
- A search trial with a non-finite validation prediction is recorded with R-loss `inf` and is not selected. Selection reads only the records: per unit and model, the configuration with the smallest validation R-loss, ties to the smaller configuration index. It runs on a commit that holds a record for every trial of the dataset; that commit enters the frozen set (Section 10). A trial with more than one record is read from its first, and disagreements are reported. Records written before a change to a model's fitting code are not used, and that model's trials run again.
- In the final block every seed-0 final fit except the causal forest's (500 trees against 200 in the search) repeats a search trial; its validation R-loss and the fingerprint of `m̂` and `e` are compared with the record, and any difference is reported without stopping the block. `R*` (Section 6) is taken from the seed-0 final fits. The dry run's outputs are committed before the read entry. If the container is lost before the read entry, the block is rerun from its start and the search is not. If it is lost after the read entry, the block's fits are rerun with the configurations and `R*` of the committed manifest, and Section 10 applies.
- Determinism settings: LightGBM `deterministic=True`, `force_row_wise=True`, `seed` as stated, 4 threads; torch `torch.manual_seed`, `torch.use_deterministic_algorithms(True)`, `torch.set_num_threads(4)`, seeded batch order; EconML `n_jobs=1` with `random_state` (separate configurations run as separate processes); numpy Generators as stated. Bitwise reproduction is expected on the same CPU type with the same package builds and thread counts (Section 3); on other machines small floating-point differences are possible.
- Measured on a 4-core 2.1 GHz container with 15 GiB of memory (training rows of Criteo and Lazada for the gradient-boosted models and the forest; synthetic data of the same shapes for the deep grid): about 52 hours for Criteo, 22 for Lazada and 1 for Hillstrom, about 75 in all; about 125 if every deep configuration ran 20 epochs at the slowest setting. EFIN accounts for about half (on Lazada's 83 features, 338 s per million rows per epoch at rank 128 and batch 256, peak memory 7.9 GB). These figures cover training. The final block, timed on synthetic data of the same shapes with tree and epoch counts scaled so that a whole run matches these totals, takes about 15 hours for Criteo (3 of them for the 2,000-replicate evaluation of the dry run and the read), 10 for Lazada and under 1 for Hillstrom; for Criteo and Lazada at most about 80 and 30 if every selected configuration is its grid's slowest and every fit runs to its cap of 1,000 trees or 20 epochs. If a dataset's final block exceeds the container's memory or disk, that dataset stops and the measured numbers are reported.

## 12. Software

Python 3.13, numpy 2.5.4, pandas 3.1.1, scipy 1.18.1, scikit-learn 1.9.1, lightgbm 4.7.0, torch 2.15.0 (CPU), econml 0.17.0, highspy 1.16.0, pyarrow 26.0.1; for V1, V2, V7, V8, V12, V13: causalml 0.18.0 and scikit-uplift 0.5.1. Versions are recorded with every result.

## 13. Limits

- Three datasets, two of them from advertising and e-commerce platforms with anonymised features; Hillstrom is small (about 25,600 test rows).
- On Criteo the causal forest is fitted on 1M training rows and every other model on 8,385,214. Statements about the causal forest on U1 and U2 say so.
- Every model is tuned by validation R-loss, not by Qini as in the papers compared (L1 to L3). A model that does well under Qini-based tuning may rank differently here.
- The three deep models are implementations from the papers' descriptions, not the authors' code; differences in details can change results. Statements about them say "as implemented here". Known deviations: DESCN uses one weight `α` for both cross terms and weight 1 for the entire-space terms, where Eq. 7 has separate weights; EFIN's layer widths beyond the embedding and its single attention head are our choices; input standardisation is a grid choice for all three; the grid is that of Liu et al. (2024), not each paper's own.
- Criteo rows with identical feature vectors are kept, as released, and kept together in roles and resampling; the 2024 benchmark removed duplicates in a way it does not define, so its population differs from ours.
- Lazada's training file has selective treatment; nuisance models on it may be biased. Evaluation uses its randomised test file only. 4,388 test rows share a feature vector with training rows.
- L3 comes from a larger production sample than the released file. L1's averaging over seeds and its test size after deduplication are unknown; two readings are reported.
- The MDD for published gaps uses our nine models' pairs; other models' pairs may vary more or less.
- The seed rule uses five seeds; intervals cover test-set sampling, not training variation. The comparison with published gains in H3 uses our LightGBM T-learner where Liu et al. used a neural one.
- Tie blocks: versions that group ties (all own versions except V4, and V2, V8) and versions that order them (V1, V4, V7, V12, V13, by a fixed random permutation) can differ for models with large tie blocks; the share of rows in tie blocks is reported per model.
- The half-sample bootstrap is justified for RATE (Yadlowsky et al., Lemma 4). In a randomised trial V1 to V10 and V13 are RATE-type functionals up to positive factors (for V5, when `q(1) > 0`) and, for V7, an added constant (Section 5.2); the scikit-uplift normalisations, the separate ranking of V10 and its 100-point approximation are covered only by that analogy.
- U4 uses either e-mail as one treatment; the two e-mails are not separated outside the allocation module.
- causalml 0.18.0's `"autoc"` weights 1/q where the paper defines α(u) = 1; V12 is kept as an implementation, not as AUTOC. EconML 0.17.0 forests are run with one job each because parallel fits were not reproducible.
- Costs in H6 and H7 are assumptions.

## 14. What is post hoc

Anything not listed in Sections 7 and 8: other budgets, costs or outcomes; analyses by feature or segment; reasons for any reversal; deduplicated or reweighted variants; comparisons with figures not in Appendix A. Post hoc results are labelled as such and do not change a registered label.

## 15. Amendments and reporting

- A change after this registration is an amendment commit stating what changed and why. A change after any dataset's test read is post hoc for every dataset and does not replace a registered label, except a change that only lets a registered step run as registered (memory, disk, an interrupted process), whose amendment says so and gives the reason.
- Every label of a test read is reported as it comes out, with its statistic, its interval where its rule uses one, and the commit and command that produced it.

## 16. Sources

- Criteo AI Lab, Criteo Uplift Prediction Dataset page (key figures, erratum, terms). Hugging Face `criteo/criteo-uplift` at `2424920019e49d52d72c13ac1143ec5d53af276b`.
- Zhong et al. (2022). DESCN. KDD 2022. https://arxiv.org/abs/2207.09920 (version 3: Tables 1 and 2, Section 4). Release: `kailiang-zhong/DESCN` at `80fd11e0`, `data/Lazada_dataset/README.md`, `main.py`.
- Hillstrom (2008). The MineThatData E-Mail Analytics And Data Mining Challenge. Blog post of 2008-03-20.
- Liu et al. (2024). Benchmarking for Deep Uplift Modeling in Online Marketing. https://arxiv.org/abs/2406.00335 (version 1: Sections 3.1 to 3.2.6, Tables 2 to 4).
- Liu et al. (2023). Explicit Feature Interaction-aware Uplift Network for Online Marketing. KDD 2023. https://arxiv.org/abs/2306.00315 (version 1: Section 5.1, Table 3).
- Singh (2026). A Large-Scale Empirical Comparison of Meta-Learners and Causal Forests for Heterogeneous Treatment Effect Estimation in Marketing Uplift Modeling. https://arxiv.org/abs/2604.06123 (version 1: Section 4, Table 2).
- Diemert et al. (2021). A Large Scale Benchmark for Individual Treatment Effect Prediction and Uplift Modeling. https://arxiv.org/abs/2111.10106 (version 1, Section 5).
- Yadlowsky et al. (2021). Evaluating Treatment Prioritization Rules via Rank-Weighted Average Treatment Effects. https://arxiv.org/abs/2111.07966 (version 2: Section 2, Example 3, Lemma 4, the Criteo example with Tables 4 and 5).
- Devriendt, Van Belle, Guns, Verbeke (2020). Learning to rank for uplift modeling. https://arxiv.org/abs/2002.05897 (Table 2, Eq. 16).
- Radcliffe (2007). Using control groups to target on predicted lift.
- Li (2026). UpliftBench. https://arxiv.org/abs/2608.00915 (version 1).
- Renaudin and Martin (2021). About evaluation metrics for contextual uplift modeling. https://arxiv.org/abs/2107.00537.
- Shalit, Johansson, Sontag (2017). https://arxiv.org/abs/1606.03976. Künzel et al. (2019). https://arxiv.org/abs/1706.03461. Nie and Wager (2021). https://arxiv.org/abs/1712.04912. Kennedy (2023). https://arxiv.org/abs/2004.14497. Athey, Tibshirani, Wager (2019). https://arxiv.org/abs/1610.01271.
- causalml 0.18.0 (`uber/causalml` at `d59bb520`), `causalml/metrics/visualize.py`, `causalml/metrics/rate.py`. scikit-uplift 0.5.1 (`maks-sh/scikit-uplift` at `0038e659`), `sklift/metrics/metrics.py`. grf (`grf-labs/grf` at `5bee99b5`), `r-package/grf/R/rank_average_treatment.R`.
- ODS.ai Website agreement (2025), clause 3.2.

## Appendix A. Literature figures

Extracted by script from the PDFs below (page numbers are 1-based PDF pages) and checked by a second reader against the rendered pages and the text layer: 50 of 50 values match their page, table, row and column. The table is fixed with this registration.

PDF SHA-256: 2406.00335v1 `161fafc308c3cfea7d5b8e42cb62445363fd62ceb614cee9fa19e64578d3b6e1`; 2306.00315v1 `a302e62dc25b2ab15ebb32c56d3cde297bd6e3ca0a93394fb9950eeeb25a9c5d`; 2207.09920v3 `b3468010f9ba995f7942a7c9e4064d985949d349f6436a844a6dbf5b58a00afb`; 2604.06123v1 `a3a7fc19eedc7d4b4c77b4abd837b7d88bc312d5e1656287d070b0861bcb4d05`.

| Group | Paper | Page | Table | Dataset | Row | Column | Value | Check |
|---|---|---|---|---|---|---|---|---|
| L1 | Liu et al. 2024 2406.00335v1 | 7 | Table 4 | Criteo | S-Learner | Test QINI | 0.0951 | match |
| L1 | Liu et al. 2024 2406.00335v1 | 7 | Table 4 | Criteo | T-Learner | Test QINI | 0.1053 | match |
| L1 | Liu et al. 2024 2406.00335v1 | 7 | Table 4 | Criteo | BNN | Test QINI | 0.1017 | match |
| L1 | Liu et al. 2024 2406.00335v1 | 7 | Table 4 | Criteo | TARNet | Test QINI | 0.1063 | match |
| L1 | Liu et al. 2024 2406.00335v1 | 7 | Table 4 | Criteo | CFRNet | Test QINI | 0.1085 | match |
| L1 | Liu et al. 2024 2406.00335v1 | 7 | Table 4 | Criteo | CEVAE | Test QINI | 0.0881 | match |
| L1 | Liu et al. 2024 2406.00335v1 | 7 | Table 4 | Criteo | GANITE | Test QINI | 0.1192 | match |
| L1 | Liu et al. 2024 2406.00335v1 | 7 | Table 4 | Criteo | DragonNet | Test QINI | 0.1099 | match |
| L1 | Liu et al. 2024 2406.00335v1 | 7 | Table 4 | Criteo | FlexTENet | Test QINI | 0.1164 | match |
| L1 | Liu et al. 2024 2406.00335v1 | 7 | Table 4 | Criteo | SNet | Test QINI | 0.1044 | match |
| L1 | Liu et al. 2024 2406.00335v1 | 7 | Table 4 | Criteo | EUEN | Test QINI | 0.1143 | match |
| L1 | Liu et al. 2024 2406.00335v1 | 7 | Table 4 | Criteo | DESCN | Test QINI | 0.0795 | match |
| L1 | Liu et al. 2024 2406.00335v1 | 7 | Table 4 | Criteo | EFIN | Test QINI | 0.1220 | match |
| L1 | Liu et al. 2024 2406.00335v1 | 7 | Table 4 | Lazada | S-Learner | Test QINI | 0.0178 | match |
| L1 | Liu et al. 2024 2406.00335v1 | 7 | Table 4 | Lazada | T-Learner | Test QINI | 0.0231 | match |
| L1 | Liu et al. 2024 2406.00335v1 | 7 | Table 4 | Lazada | BNN | Test QINI | 0.0199 | match |
| L1 | Liu et al. 2024 2406.00335v1 | 7 | Table 4 | Lazada | TARNet | Test QINI | 0.0177 | match |
| L1 | Liu et al. 2024 2406.00335v1 | 7 | Table 4 | Lazada | CFRNet | Test QINI | 0.0234 | match |
| L1 | Liu et al. 2024 2406.00335v1 | 7 | Table 4 | Lazada | CEVAE | Test QINI | 0.0078 | match |
| L1 | Liu et al. 2024 2406.00335v1 | 7 | Table 4 | Lazada | GANITE | Test QINI | 0.0187 | match |
| L1 | Liu et al. 2024 2406.00335v1 | 7 | Table 4 | Lazada | DragonNet | Test QINI | 0.0177 | match |
| L1 | Liu et al. 2024 2406.00335v1 | 7 | Table 4 | Lazada | FlexTENet | Test QINI | 0.0215 | match |
| L1 | Liu et al. 2024 2406.00335v1 | 7 | Table 4 | Lazada | SNet | Test QINI | 0.0190 | match |
| L1 | Liu et al. 2024 2406.00335v1 | 7 | Table 4 | Lazada | EUEN | Test QINI | 0.0205 | match |
| L1 | Liu et al. 2024 2406.00335v1 | 7 | Table 4 | Lazada | DESCN | Test QINI | 0.0252 | match |
| L1 | Liu et al. 2024 2406.00335v1 | 7 | Table 4 | Lazada | EFIN | Test QINI | 0.0169 | match |
| L2 | Liu et al. 2023 (EFIN) 2306.00315v1 | 7 | Table 3 | Criteo | S-Learner | CRITEO-UPLIFT QINI | 0.0857 | match |
| L2 | Liu et al. 2023 (EFIN) 2306.00315v1 | 7 | Table 3 | Criteo | T-Learner | CRITEO-UPLIFT QINI | 0.1083 | match |
| L2 | Liu et al. 2023 (EFIN) 2306.00315v1 | 7 | Table 3 | Criteo | TarNet | CRITEO-UPLIFT QINI | 0.1027 | match |
| L2 | Liu et al. 2023 (EFIN) 2306.00315v1 | 7 | Table 3 | Criteo | CFRNet | CRITEO-UPLIFT QINI | 0.1052 | match |
| L2 | Liu et al. 2023 (EFIN) 2306.00315v1 | 7 | Table 3 | Criteo | DragonNet | CRITEO-UPLIFT QINI | 0.1096 | match |
| L2 | Liu et al. 2023 (EFIN) 2306.00315v1 | 7 | Table 3 | Criteo | GANITE | CRITEO-UPLIFT QINI | 0.1170 | match |
| L2 | Liu et al. 2023 (EFIN) 2306.00315v1 | 7 | Table 3 | Criteo | CEVAE | CRITEO-UPLIFT QINI | 0.0951 | match |
| L2 | Liu et al. 2023 (EFIN) 2306.00315v1 | 7 | Table 3 | Criteo | FlexTENet | CRITEO-UPLIFT QINI | 0.1108 | match |
| L2 | Liu et al. 2023 (EFIN) 2306.00315v1 | 7 | Table 3 | Criteo | SNet | CRITEO-UPLIFT QINI | 0.1112 | match |
| L2 | Liu et al. 2023 (EFIN) 2306.00315v1 | 7 | Table 3 | Criteo | EUEN | CRITEO-UPLIFT QINI | 0.1153 | match |
| L2 | Liu et al. 2023 (EFIN) 2306.00315v1 | 7 | Table 3 | Criteo | DESCN | CRITEO-UPLIFT QINI | 0.1129 | match |
| L2 | Liu et al. 2023 (EFIN) 2306.00315v1 | 7 | Table 3 | Criteo | EFIN | CRITEO-UPLIFT QINI | 0.1285 | match |
| L3 | Zhong et al. 2022 (DESCN) 2207.09920v3 | 6 | Table 2 | Lazada (production sample, 0.91M test rows) | X-learner (NN) | Production Dataset AUUC (mean) | 0.0234 | match |
| L3 | Zhong et al. 2022 (DESCN) 2207.09920v3 | 6 | Table 2 | Lazada (production sample, 0.91M test rows) | Causal Forest | Production Dataset AUUC (mean) | 0.0132 | match |
| L3 | Zhong et al. 2022 (DESCN) 2207.09920v3 | 6 | Table 2 | Lazada (production sample, 0.91M test rows) | BART | Production Dataset AUUC (mean) | 0.0222 | match |
| L3 | Zhong et al. 2022 (DESCN) 2207.09920v3 | 6 | Table 2 | Lazada (production sample, 0.91M test rows) | TARNet | Production Dataset AUUC (mean) | 0.0309 | match |
| L3 | Zhong et al. 2022 (DESCN) 2207.09920v3 | 6 | Table 2 | Lazada (production sample, 0.91M test rows) | CFR_wass | Production Dataset AUUC (mean) | 0.0261 | match |
| L3 | Zhong et al. 2022 (DESCN) 2207.09920v3 | 6 | Table 2 | Lazada (production sample, 0.91M test rows) | CFR_mmd | Production Dataset AUUC (mean) | 0.0324 | match |
| L3 | Zhong et al. 2022 (DESCN) 2207.09920v3 | 6 | Table 2 | Lazada (production sample, 0.91M test rows) | X-network | Production Dataset AUUC (mean) | 0.0324 | match |
| L3 | Zhong et al. 2022 (DESCN) 2207.09920v3 | 6 | Table 2 | Lazada (production sample, 0.91M test rows) | DESCN | Production Dataset AUUC (mean) | 0.0340 | match |
| L4 | Singh 2026 2604.06123v1 | 3 | Table 2 | Criteo | S-Learner | Qini | 0.3759 | match |
| L4 | Singh 2026 2604.06123v1 | 3 | Table 2 | Criteo | X-Learner | Qini | 0.3615 | match |
| L4 | Singh 2026 2604.06123v1 | 3 | Table 2 | Criteo | T-Learner | Qini | 0.3503 | match |
| L4 | Singh 2026 2604.06123v1 | 3 | Table 2 | Criteo | Causal Forest | Qini | 0.2524 | match |

Notes from the check: the EFIN value carries a significance mark in the PDF; DESCN's row label is printed "X-leaner (NN)"; DESCN's Table 2 column header is "Production Dataset"; Singh's table also has a row "Random Baseline ≈0.007", not used.

## Appendix B. Design-stage minimum detectable differences

Synthetic data only, calibrated to each unit's test size and treated share, and to published rates: U1 visit rate 0.046992 (dataset page) and average effect 0.0103 (Table 2 of Liu et al. 2024); U3 3.33% and 3.70% (Lazada README); U4 visit rate 0.14678 (Section 3) with an assumed average effect of 0.05. Effects vary as `τ(z) = (average effect) x exp(0.75 z - 0.28)` with `z` standard normal, capped so that rates stay below 1; one score has correlation 0.6 with `z`, a second has correlation `ρ` with the first. IPW scores; 200 half-sample replicates; MDD = 2.80 SE of the paired difference.

| Unit | n | ρ | MDD AUTOC (V11) | MDD QINI (V6) | MDD Qini, scikit-uplift units (V2*) | V2* at literature sizes |
|---|---|---|---|---|---|---|
| U1 | 4,195,534 | 0.5 | 0.000821 | 0.000229 | 0.0053 | L1: 0.0092; L2: 0.0065 |
| U1 | 4,195,534 | 0.8 | 0.000526 | 0.000143 | 0.0034 | L1: 0.0058; L2: 0.0041 |
| U1 | 4,195,534 | 0.95 | 0.000264 | 0.000075 | 0.0017 | L1: 0.0030; L2: 0.0021 |
| U3 | 181,669 | 0.5 | 0.002690 | 0.000726 | 0.0208 | L1: 0.0208; L3: 0.0093 |
| U3 | 181,669 | 0.8 | 0.001643 | 0.000410 | 0.0115 | L1: 0.0115; L3: 0.0051 |
| U3 | 181,669 | 0.95 | 0.000808 | 0.000222 | 0.0064 | L1: 0.0064; L3: 0.0029 |
| U4 | 25,600 | 0.5 | 0.013659 | 0.003857 | 0.0292 |  |
| U4 | 25,600 | 0.8 | 0.009544 | 0.002511 | 0.0190 |  |
| U4 | 25,600 | 0.95 | 0.004878 | 0.001276 | 0.0093 |  |

Median absolute published gap (Appendix A): L1 Criteo 0.0121 (78 gaps), L1 Lazada 0.0034 (78), L2 0.0189 (11), L3 0.0079 (7), L4 0.0144 (3, Singh's units, not simulated here).

These numbers set expectations only. The registered MDDs come from the test sets.
