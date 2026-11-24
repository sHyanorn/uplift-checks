# Results of the Lazada and Hillstrom test reads

Every label of the two test reads, with its statistic and, where its rule uses one, its interval (Section 15), from [`results/lazada.json`](results/lazada.json) and [`results/hillstrom.json`](results/hillstrom.json). "Section" means a section of [`PREREGISTRATION.md`](PREREGISTRATION.md). Statistics are rounded here to three significant figures; counts, registered constants and published figures (`g`, `c`, `k`, `n_L`) are given as they are. The JSON files also hold every version's value per model, Kendall's τ_b between versions, the tie shares, the reversed pairs of H1b, the five seed values of each difference, the intervals of H1 (36 per own version) and H2 (72 per unit), the package versions and the resource figures. On both units each model's five fits gave distinct predictions (`distinct predictions over seeds`, Section 5.5). Criteo (U1, U2) has not been run.

Each dataset's test rows are read once (an interrupted read aside, see [`RERUN.md`](RERUN.md)), after its models are frozen and a dry run on validation rows has computed every output; the dry run, in `results/DATASET-dry-run.json`, computes the labels (the outcomes of the decision rules of Sections 7 and 8), which enter no verdict (Section 4). The read writes `results/DATASET.json`: every label of Sections 7 and 8 with the statistics behind it, including the intervals and seed values that its rule reads, the package versions, the resource figures, `commit` (the commit that holds the read entry), `command` (the stage that computed the labels at that commit; Section 15) and `second read` (Section 10).

| | Lazada | Hillstrom |
|---|---|---|
| Unit | U3 | U4 |
| Test rows | 181,669 | 25,775 |
| Resampling groups `G` | 179,232 | 22,405 |
| R* | DR | DR |
| Read entry in `records/reads.jsonl` | 2026-11-24T06:48:16+00:00 | 2026-11-24T07:13:07+00:00 |
| `commit`, which holds the read entry | `c997e780a7deda31f9290ff437529efc36d32310` | `aa51c01ca3084256d8f7df677aebb2d11930e73f` |
| `command` | `python -m uplift_ranking.run read lazada go` | `python -m uplift_ranking.run read hillstrom go` |
| `second read` | false | false |

Before each read entry the dataset's final block was rerun: its manifest and dry run were rewritten in `e4f9e5e` (Lazada, after amendment 10) and `cb8a9d3` (Hillstrom), the commits that the read entries record. In both manifests only `records commit` changed (Lazada `944628b` to `4fc6589`, Hillstrom `afca85f` to `78ffd3f`), so no frozen file's hash changed; the Lazada dry run changed only in the null of amendment 10 and its two peak-memory figures, the Hillstrom dry run only in its 18 solve times and one peak-memory figure.

## H1. Implementations and definitions (Section 7)

Pairs separated (simultaneous interval excluding 0 and the seed rule holding) in the family of 36 of each own version that H1 uses, with the family's critical value `c*`:

| Version | Lazada `c*` | Lazada separated | Hillstrom `c*` | Hillstrom separated |
|---|---|---|---|---|
| V2* | 3.01 | 0 | 3.08 | 0 |
| V3 | 3.01 | 0 | 3.09 | 0 |
| V4 | 3.02 | 0 | 3.09 | 0 |
| V6 | 3.02 | 0 | 3.07 | 0 |
| V9 | 3.02 | 0 | 3.07 | 0 |
| V10 | 3.03 | 0 | 3.09 | 0 |
| V11 | 3.01 | 0 | 3.08 | 0 |

| Label | Lazada | Hillstrom |
|---|---|---|
| H1a | "no separated pair; 8 reversals of 36" | "no separated pair; 1 reversals of 36" |
| H1b, V6 and V11 | "no separated pair; 20 reversals of 36" | "no separated pair; 8 reversals of 36" |
| H1b, V11 and V12 | "no separated pair; 13 reversals of 36" | "no separated pair; 18 reversals of 36" |
| H1b, V6 and V12 | "no separated pair; 27 reversals of 36" | "no separated pair; 20 reversals of 36" |

Pairs reversed between two H1a versions: on Lazada S and R; X and DR; X and causal forest; X and DESCN; X and EFIN; DR and causal forest; DR and EFIN; DESCN and EFIN; on Hillstrom S and DR. Section 7 draws no statement of agreement from "no separated pair" under H1a, and under H1b no change in the preferred model is shown.

## H2. How many differences can be told apart (Section 7)

Family of 72: the 36 pairs under V6 and under V11. A pair counts if it is separated in this family under at least one of them.

| | Lazada | Hillstrom |
|---|---|---|
| `c*` of the family of 72 | 3.17 | 3.26 |
| N, of 36 pairs | 0 | 0 |
| Label | "mostly indistinguishable" | "mostly indistinguishable" |
| Separated under V6 alone; under V11 alone | 0; 0 | 0; 0 |
| Of N: among the 10 meta-learner pairs; among the 18 deep against non-deep pairs | 0; 0 | 0; 0 |
| Marginal (unadjusted) N; meta-learner pairs; deep against non-deep pairs | 0; 0; 0 | 5; 0; 3 |
| Median `MDD_sim`: over the 72; over V6's 36; over V11's 36 | 0.00152; 0.00130; 0.00524 | 0.00849; 0.00468; 0.0175 |
| Median marginal MDD over the 36 pairs: V6; V11 | 0.000906; 0.00366 | 0.00320; 0.0120 |

Both readings of the median `MDD_sim` are reported (amendment 7).

## H3. Deep models against the best tuned non-deep learner (Section 7)

"Family per unit: these 12 differences": `M(D) - M(R*)` under V2*, V6 and V11, and `M(D) - M(T)` under V2*, for D in TARNet, DESCN and EFIN, with T our LightGBM T-learner. `[L, U]` is the simultaneous 95% interval over the family (Section 5.5); `c*` is 2.62 on Lazada and 2.61 on Hillstrom. `MDD_sim` is `(c* + 0.84) SE` and the marginal MDD `2.80 SE`. A contrast is "gain shown" if `L > 0` and the seed rule holds, "loss shown" if `U < 0` and the seed rule holds, and otherwise "undecided".

| Lazada: contrast | Version | Value [L, U] | `MDD_sim` | Marginal MDD | Label |
|---|---|---|---|---|---|
| TARNet - DR | V2* | 0.00557 [-0.0107, 0.0219] | 0.0215 | 0.0174 | "undecided" |
| DESCN - DR | V2* | 0.00330 [-0.0139, 0.0205] | 0.0227 | 0.0184 | "undecided" |
| EFIN - DR | V2* | 0.000442 [-0.0156, 0.0164] | 0.0211 | 0.0171 | "undecided" |
| TARNet - DR | V6 | 0.000194 [-0.000354, 0.000742] | 0.000724 | 0.000585 | "undecided" |
| DESCN - DR | V6 | 0.0000997 [-0.000482, 0.000681] | 0.000768 | 0.000621 | "undecided" |
| EFIN - DR | V6 | 0.000125 [-0.000410, 0.000661] | 0.000706 | 0.000571 | "undecided" |
| TARNet - DR | V11 | -0.000376 [-0.00241, 0.00166] | 0.00269 | 0.00218 | "undecided" |
| DESCN - DR | V11 | -0.000275 [-0.00249, 0.00195] | 0.00293 | 0.00237 | "undecided" |
| EFIN - DR | V11 | -0.000342 [-0.00232, 0.00163] | 0.00261 | 0.00211 | "undecided" |
| TARNet - T | V2* | 0.0130 [-0.00731, 0.0334] | 0.0269 | 0.0217 | "undecided" |
| DESCN - T | V2* | 0.0108 [-0.00868, 0.0302] | 0.0257 | 0.0208 | "undecided" |
| EFIN - T | V2* | 0.00792 [-0.0127, 0.0285] | 0.0272 | 0.0220 | "undecided" |

| Hillstrom: contrast | Version | Value [L, U] | `MDD_sim` | Marginal MDD | Label |
|---|---|---|---|---|---|
| TARNet - DR | V2* | 0.0145 [-0.0127, 0.0418] | 0.0361 | 0.0293 | "undecided" |
| DESCN - DR | V2* | -0.00117 [-0.0296, 0.0273] | 0.0376 | 0.0305 | "undecided" |
| EFIN - DR | V2* | 0.0208 [-0.00736, 0.0489] | 0.0372 | 0.0302 | "undecided" |
| TARNet - DR | V6 | 0.00162 [-0.00170, 0.00494] | 0.00439 | 0.00356 | "undecided" |
| DESCN - DR | V6 | -0.000249 [-0.00372, 0.00322] | 0.00458 | 0.00372 | "undecided" |
| EFIN - DR | V6 | 0.00242 [-0.000997, 0.00585] | 0.00452 | 0.00367 | "undecided" |
| TARNet - DR | V11 | 0.00219 [-0.00948, 0.0139] | 0.0154 | 0.0125 | "undecided" |
| DESCN - DR | V11 | -0.00402 [-0.0164, 0.00839] | 0.0164 | 0.0133 | "undecided" |
| EFIN - DR | V11 | -0.000846 [-0.0131, 0.0115] | 0.0163 | 0.0132 | "undecided" |
| TARNet - T | V2* | 0.0212 [-0.00380, 0.0462] | 0.0330 | 0.0268 | "undecided" |
| DESCN - T | V2* | 0.00547 [-0.0226, 0.0335] | 0.0370 | 0.0301 | "undecided" |
| EFIN - T | V2* | 0.0274 [0.000913, 0.0539] | 0.0350 | 0.0284 | "gain shown" |

"The result for D on a unit is the V2* label" of `M(D) - M(R*)`: on Lazada TARNet "undecided", DESCN "undecided", EFIN "undecided"; on Hillstrom TARNet "undecided", DESCN "undecided", EFIN "undecided". No result has "metric-dependent" added. The three contrasts with T belong to the family and so enter `c*`; their labels apply rules 1 to 3 to `M(D) - M(T)`, and the registration reads those labels only in the published-gain rule, where a published gain `g` exists. U4 has none, so the "gain shown" of EFIN - T on Hillstrom changes no result. DESCN's published gain on Lazada, `g` = 0.0021 (Table 4 of Liu et al. 2024), gives `z` = -0.830 against `c*` = 2.62: "published gain not tested at this size".

As Section 7 requires: the main comparison is with R*, the non-deep learner with the smallest validation R-loss; tuning used R-loss and not Qini; TARNet, DESCN and EFIN are implementations from the papers' descriptions, with the deviations of Section 13. Two further limits of Section 13 apply to these tables: the comparison with published gains uses our LightGBM T-learner where Liu et al. used a neural one, and the intervals cover test-set sampling, not training variation (the five seed values of each contrast are in the JSON files).

## H4. Detectable differences and published gaps (Section 7)

MDD (`2.80 SE`) of the 36 pairs in each own version:

| Version | Lazada median | min | max | Hillstrom median | min | max |
|---|---|---|---|---|---|---|
| V2* | 0.0268 | 0.00688 | 0.0314 | 0.0262 | 0.0160 | 0.0317 |
| V3 | 0.000482 | 0.000124 | 0.000564 | 0.00218 | 0.00133 | 0.00264 |
| V4 | 0.000482 | 0.000124 | 0.000564 | 0.00219 | 0.00133 | 0.00264 |
| V5 | 0.270 | 0.0829 | 0.325 | 0.0489 | 0.0299 | 0.0596 |
| V6 | 0.000906 | 0.000228 | 0.00107 | 0.00320 | 0.00195 | 0.00389 |
| V9 | 0.000926 | 0.000238 | 0.00109 | 0.00325 | 0.00199 | 0.00394 |
| V10 | 0.000935 | 0.000229 | 0.00110 | 0.00324 | 0.00198 | 0.00395 |
| V11 | 0.00366 | 0.000990 | 0.00430 | 0.0120 | 0.00771 | 0.0142 |

Of the two units read, only U3 has literature groups (Section 7, Appendix A). `MDD_L` is the median over our 36 pairs of `MDD x sqrt(G / n_L)` under V2*; `s_L` is the share of the group's gaps whose absolute value is below it.

| Group | Gaps (Section 7) | `n_L` | `MDD_L` | Gaps below | Label |
|---|---|---|---|---|---|
| L1 Lazada | 78 pairs of 13 models in Table 4 of Liu et al. (2024), Lazada | 181,669 | 0.0266 | 78 of 78 | "most published gaps below the detectable difference" |
| L3 | DESCN minus each of 7 models in Table 2 of Zhong et al. (2022) | 910,000 | 0.0119 | 6 of 7 | "most published gaps below the detectable difference" |

In the words of Section 7: 78 of 78 gaps reported in Table 4 of Liu et al. (2024) on Lazada, and 6 of 7 reported in Table 2 of Zhong et al. (2022), are smaller than the difference that a test set of that size detects with 80% power. The MDD comes from our nine models' pairs.

Reported with it, no rule:

| `s_L` | L1 Lazada | L3 |
|---|---|---|
| With the minimum of our 36 MDDs | 0.795 | 0.286 |
| With the maximum of our 36 MDDs | 1.00 | 0.857 |
| With the MDD of the matching pair of our models | 1.00 | null: no matching pair (amendment 10) |
| L3 at the released test size, `n_L` = 181,669 | | `MDD_L` 0.0266, `s_L` 1.00 |

## H5. Value under a budget on one treatment (Section 8)

`V_k(D) - V_k(R*)` for the top share `k` of rows by each model's score, with marginal 95% intervals, not adjusted for the number of contrasts. The AIPW label is the result; the IPW label is reported beside it.

| Lazada: model | `k` | AIPW value [L, U] | AIPW label | IPW value [L, U] | IPW label |
|---|---|---|---|---|---|
| S | 0.1 | -0.000435 [-0.00133, 0.000458] | "not distinguishable" | -0.000486 [-0.00144, 0.000464] | "not distinguishable" |
| T | 0.1 | -0.000226 [-0.00100, 0.000551] | "not distinguishable" | -0.000419 [-0.00124, 0.000405] | "not distinguishable" |
| X | 0.1 | -0.000221 [-0.00110, 0.000656] | "not distinguishable" | -0.000241 [-0.00117, 0.000688] | "not distinguishable" |
| R | 0.1 | -0.000639 [-0.00147, 0.000189] | "not distinguishable" | -0.000533 [-0.00140, 0.000336] | "not distinguishable" |
| causal forest | 0.1 | -0.000150 [-0.00109, 0.000786] | "not distinguishable" | -0.000373 [-0.00137, 0.000626] | "not distinguishable" |
| TARNet | 0.1 | -0.000153 [-0.000798, 0.000492] | "not distinguishable" | -0.000217 [-0.000903, 0.000470] | "not distinguishable" |
| DESCN | 0.1 | -0.000298 [-0.00100, 0.000407] | "not distinguishable" | -0.000284 [-0.00104, 0.000474] | "not distinguishable" |
| EFIN | 0.1 | -0.000268 [-0.000914, 0.000379] | "not distinguishable" | -0.000162 [-0.000862, 0.000538] | "not distinguishable" |
| S | 0.3 | 0.000359 [-0.000643, 0.00136] | "not distinguishable" | 0.000172 [-0.000873, 0.00122] | "not distinguishable" |
| T | 0.3 | -0.000236 [-0.00114, 0.000668] | "not distinguishable" | -0.000372 [-0.00131, 0.000567] | "not distinguishable" |
| X | 0.3 | 0.000231 [-0.000823, 0.00128] | "not distinguishable" | -0.0000448 [-0.00114, 0.00105] | "not distinguishable" |
| R | 0.3 | -0.000292 [-0.00125, 0.000663] | "not distinguishable" | -0.000115 [-0.00110, 0.000873] | "not distinguishable" |
| causal forest | 0.3 | -0.000439 [-0.00143, 0.000549] | "not distinguishable" | -0.000617 [-0.00166, 0.000423] | "not distinguishable" |
| TARNet | 0.3 | 0.000202 [-0.000563, 0.000968] | "not distinguishable" | 0.000291 [-0.000495, 0.00108] | "not distinguishable" |
| DESCN | 0.3 | 0.000323 [-0.000455, 0.00110] | "not distinguishable" | 0.000343 [-0.000463, 0.00115] | "not distinguishable" |
| EFIN | 0.3 | 0.000177 [-0.000544, 0.000899] | "not distinguishable" | 0.000375 [-0.000372, 0.00112] | "not distinguishable" |
| S | 0.5 | -0.000267 [-0.00124, 0.000704] | "not distinguishable" | -0.000490 [-0.00150, 0.000517] | "not distinguishable" |
| T | 0.5 | -0.000545 [-0.00140, 0.000312] | "not distinguishable" | -0.000665 [-0.00155, 0.000221] | "not distinguishable" |
| X | 0.5 | -0.00000157 [-0.000945, 0.000941] | "not distinguishable" | -0.000357 [-0.00134, 0.000629] | "not distinguishable" |
| R | 0.5 | -0.000574 [-0.00148, 0.000331] | "not distinguishable" | -0.000514 [-0.00144, 0.000416] | "not distinguishable" |
| causal forest | 0.5 | -0.000197 [-0.00100, 0.000606] | "not distinguishable" | -0.000321 [-0.00116, 0.000516] | "not distinguishable" |
| TARNet | 0.5 | 0.000286 [-0.000388, 0.000960] | "not distinguishable" | 0.000278 [-0.000412, 0.000968] | "not distinguishable" |
| DESCN | 0.5 | 0.000234 [-0.000469, 0.000937] | "not distinguishable" | 0.000226 [-0.000495, 0.000947] | "not distinguishable" |
| EFIN | 0.5 | -0.00000398 [-0.000676, 0.000668] | "not distinguishable" | 0.0000650 [-0.000620, 0.000751] | "not distinguishable" |

| Hillstrom: model | `k` | AIPW value [L, U] | AIPW label | IPW value [L, U] | IPW label |
|---|---|---|---|---|---|
| S | 0.1 | -0.000568 [-0.00340, 0.00227] | "not distinguishable" | -0.000218 [-0.00335, 0.00291] | "not distinguishable" |
| T | 0.1 | -0.000167 [-0.00303, 0.00269] | "not distinguishable" | 0.000247 [-0.00291, 0.00340] | "not distinguishable" |
| X | 0.1 | -0.00120 [-0.00389, 0.00149] | "not distinguishable" | -0.000750 [-0.00373, 0.00223] | "not distinguishable" |
| R | 0.1 | -0.000752 [-0.00336, 0.00186] | "not distinguishable" | 0.0000860 [-0.00282, 0.00299] | "not distinguishable" |
| causal forest | 0.1 | -0.000611 [-0.00393, 0.00271] | "not distinguishable" | -0.000522 [-0.00423, 0.00319] | "not distinguishable" |
| TARNet | 0.1 | -0.000489 [-0.00338, 0.00240] | "not distinguishable" | 0.000136 [-0.00312, 0.00339] | "not distinguishable" |
| DESCN | 0.1 | -0.000471 [-0.00359, 0.00264] | "not distinguishable" | 0.000660 [-0.00281, 0.00413] | "not distinguishable" |
| EFIN | 0.1 | -0.00188 [-0.00485, 0.00109] | "not distinguishable" | -0.00103 [-0.00443, 0.00238] | "not distinguishable" |
| S | 0.3 | -0.00161 [-0.00574, 0.00252] | "not distinguishable" | -0.00109 [-0.00551, 0.00333] | "not distinguishable" |
| T | 0.3 | -0.00167 [-0.00575, 0.00240] | "not distinguishable" | -0.000288 [-0.00471, 0.00413] | "not distinguishable" |
| X | 0.3 | -0.00241 [-0.00622, 0.00140] | "not distinguishable" | -0.000636 [-0.00478, 0.00351] | "not distinguishable" |
| R | 0.3 | -0.000909 [-0.00478, 0.00296] | "not distinguishable" | 0.00115 [-0.00297, 0.00527] | "not distinguishable" |
| causal forest | 0.3 | -0.00141 [-0.00568, 0.00287] | "not distinguishable" | 0.000174 [-0.00443, 0.00478] | "not distinguishable" |
| TARNet | 0.3 | 0.000243 [-0.00415, 0.00463] | "not distinguishable" | 0.00262 [-0.00211, 0.00735] | "not distinguishable" |
| DESCN | 0.3 | -0.00132 [-0.00575, 0.00312] | "not distinguishable" | 0.00177 [-0.00300, 0.00654] | "not distinguishable" |
| EFIN | 0.3 | 0.00142 [-0.00313, 0.00597] | "not distinguishable" | 0.00292 [-0.00198, 0.00782] | "not distinguishable" |
| S | 0.5 | -0.00386 [-0.00816, 0.000451] | "not distinguishable" | -0.00356 [-0.00822, 0.00110] | "not distinguishable" |
| T | 0.5 | -0.00391 [-0.00803, 0.000205] | "not distinguishable" | -0.00332 [-0.00773, 0.00108] | "not distinguishable" |
| X | 0.5 | -0.00171 [-0.00544, 0.00202] | "not distinguishable" | -0.00117 [-0.00523, 0.00289] | "not distinguishable" |
| R | 0.5 | 0.00186 [-0.00167, 0.00539] | "not distinguishable" | 0.00316 [-0.000625, 0.00695] | "not distinguishable" |
| causal forest | 0.5 | 0.00112 [-0.00312, 0.00537] | "not distinguishable" | 0.00254 [-0.00205, 0.00714] | "not distinguishable" |
| TARNet | 0.5 | 0.00276 [-0.00128, 0.00681] | "not distinguishable" | 0.00420 [-0.000221, 0.00862] | "not distinguishable" |
| DESCN | 0.5 | -0.00364 [-0.00842, 0.00115] | "not distinguishable" | -0.00235 [-0.00754, 0.00284] | "not distinguishable" |
| EFIN | 0.5 | 0.00363 [-0.000687, 0.00794] | "not distinguishable" | 0.00431 [-0.000411, 0.00904] | "not distinguishable" |

Median ratio of AIPW to IPW interval widths, over all models and budgets and per budget (amendment 7):

| | All | `k` = 0.1 | `k` = 0.3 | `k` = 0.5 |
|---|---|---|---|---|
| Lazada | 0.959 | 0.939 | 0.965 | 0.970 |
| Hillstrom | 0.921 | 0.897 | 0.929 | 0.923 |

H2 separated no pair, and the one H3 contrast labelled "gain shown", EFIN - T on Hillstrom, has no H5 counterpart: H5's contrasts are with R*. Section 8 makes no statement on profit.

## H6. Exact and dual solutions of the three-arm choice on Hillstrom (Section 8)

Instance `i` has cost `c` of the women's e-mail (the men's costs 1) and budget `B` = `k` x the number of test rows. The label reads the HiGHS status and `|V_E - V_D| / V_E`; the bound `max v / V_E` is reported with it.

| `i` | `c` | `k` | Status | Gap | `V_E` | `\|V_E - V_D\| / V_E` | Label | `max v / V_E` | Seconds, exact | Seconds, dual |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.5 | 0.1 | Optimal | 0 | 584 | 0 | "dual within 1e-4 of exact" | 0.000418 | 11.9 | 0.0318 |
| 1 | 0.5 | 0.3 | Optimal | 0 | 1220 | 0 | "dual within 1e-4 of exact" | 0.000200 | 11.8 | 0.0363 |
| 2 | 0.5 | 0.5 | Optimal | 0 | 1650 | 0 | "dual within 1e-4 of exact" | 0.000148 | 12.0 | 0.0416 |
| 3 | 1 | 0.1 | Optimal | 0 | 386 | 0 | "dual within 1e-4 of exact" | 0.000633 | 12.6 | 0.0449 |
| 4 | 1 | 0.3 | Optimal | 0 | 946 | 0 | "dual within 1e-4 of exact" | 0.000258 | 12.2 | 0.0386 |
| 5 | 1 | 0.5 | Optimal | 0 | 1390 | 0 | "dual within 1e-4 of exact" | 0.000175 | 12.3 | 0.0362 |
| 6 | 2 | 0.1 | Optimal | 0 | 351 | 0 | "dual within 1e-4 of exact" | 0.000696 | 7.21 | 0.0294 |
| 7 | 2 | 0.3 | Optimal | 0 | 866 | 0 | "dual within 1e-4 of exact" | 0.000282 | 7.93 | 0.0335 |
| 8 | 2 | 0.5 | Optimal | 0 | 1280 | 0.000113 | "dual below exact" | 0.000191 | 7.46 | 0.0357 |

The dual solution matches the exact one on the instances labelled "dual within 1e-4 of exact".

## H7. Estimated value of the knapsack policy (Section 8)

Estimated value of the exact policy minus that of baseline b1 (users drawn at random) or b2 (users ranked by R*'s score), both sending the e-mail arm `a` with the larger mean of `v_ij`, with marginal 95% intervals. The AIPW label is the result; the IPW label is reported beside it. These statements hold under the assumed costs only.

| `i` | Baseline | AIPW value [L, U] | AIPW label | IPW value [L, U] | IPW label |
|---|---|---|---|---|---|
| 0 | b1 | 0.00841 [0.00267, 0.0141] | "higher" | 0.0101 [0.00394, 0.0163] | "higher" |
| 0 | b2 | 0.00353 [-0.00191, 0.00896] | "not distinguishable" | 0.00410 [-0.00188, 0.0101] | "not distinguishable" |
| 1 | b1 | 0.0110 [0.00226, 0.0196] | "higher" | 0.0138 [0.00430, 0.0232] | "higher" |
| 1 | b2 | 0.00820 [0.000435, 0.0160] | "higher" | 0.0103 [0.00192, 0.0187] | "higher" |
| 2 | b1 | 0.0127 [0.00338, 0.0221] | "higher" | 0.0143 [0.00422, 0.0244] | "higher" |
| 2 | b2 | 0.00944 [0.000625, 0.0182] | "higher" | 0.0111 [0.00161, 0.0206] | "higher" |
| 3 | b1 | -0.000168 [-0.00505, 0.00472] | "not distinguishable" | -0.00110 [-0.00654, 0.00435] | "not distinguishable" |
| 3 | b2 | -0.00531 [-0.00947, -0.00114] | "lower" | -0.00561 [-0.0103, -0.000891] | "lower" |
| 4 | b1 | 0.00453 [-0.00276, 0.0118] | "not distinguishable" | 0.00424 [-0.00370, 0.0122] | "not distinguishable" |
| 4 | b2 | -0.00428 [-0.0107, 0.00211] | "not distinguishable" | -0.00292 [-0.00995, 0.00411] | "not distinguishable" |
| 5 | b1 | -0.000743 [-0.00878, 0.00729] | "not distinguishable" | 0.0000596 [-0.00882, 0.00894] | "not distinguishable" |
| 5 | b2 | -0.00225 [-0.00978, 0.00529] | "not distinguishable" | -0.00207 [-0.0103, 0.00620] | "not distinguishable" |
| 6 | b1 | 0.00390 [-0.000884, 0.00869] | "not distinguishable" | 0.00275 [-0.00259, 0.00808] | "not distinguishable" |
| 6 | b2 | -0.00156 [-0.00537, 0.00225] | "not distinguishable" | -0.00199 [-0.00628, 0.00230] | "not distinguishable" |
| 7 | b1 | 0.00233 [-0.00483, 0.00949] | "not distinguishable" | 0.00159 [-0.00612, 0.00930] | "not distinguishable" |
| 7 | b2 | -0.00265 [-0.00856, 0.00327] | "not distinguishable" | -0.00208 [-0.00856, 0.00439] | "not distinguishable" |
| 8 | b1 | 0.000272 [-0.00694, 0.00749] | "not distinguishable" | -0.00125 [-0.00903, 0.00653] | "not distinguishable" |
| 8 | b2 | -0.00168 [-0.00854, 0.00518] | "not distinguishable" | -0.00234 [-0.00981, 0.00513] | "not distinguishable" |
