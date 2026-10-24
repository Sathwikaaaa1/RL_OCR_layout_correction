# Exploratory three-unseen-page follow-up

User-authorized exploratory follow-up after the preregistered minimum-pool stop, NOT the original preregistered final evaluation. All three qualifying unseen pages were included, without replacement. Seen pages are all previously evaluated magazine/newspaper pages, not necessarily sidebar-style; they were chosen by document type only, never tau. New and seen groups are separate; combined results are descriptive and not an independent new test. Frozen all-lines models, prior 4.0, original gate and normalizers, same three seeds, no training/tuning. Seed taus are averaged within page, not treated as independent pages. Paired bootstrap and two-sided sign flips use 10,000 samples, seed 314159; tiny N=3 uncertainty cannot establish generalization. Scores condition on reliable text alignment, not official OmniDocBench scores. Earlier 150-page results had no method beating native.

New: 3 pages (2 magazine, 1 newspaper), all satisfying the frozen sidebar proxy. Seen: 34 pages (12 magazine, 22 newspaper); 0 satisfy that proxy. Same document type does not mean same sidebar geometry.

## All-method comparison

| Cohort / type | N scored / selected | Method | Mean tau | Delta vs native [95% CI] | p vs native | Better/worse/same | Delta vs gate [95% CI] |
|---|---:|---|---:|---|---:|---|---|
| unseen_three/all | 3/3 | Native | 0.964912 | +0.000000 [+0.000000, +0.000000] | 1.000000 | 0/0/3 | +0.093435 [+0.023032, +0.140351] |
| unseen_three/all | 3/3 | XY-cut | 0.964905 | -0.000008 [-0.000023, +0.000000] | 1.000000 | 0/1/2 | +0.093428 [+0.023010, +0.140351] |
| unseen_three/all | 3/3 | Raw GAT | 0.896730 | -0.068183 [-0.098462, -0.035911] | 0.255674 | 0/3/0 | +0.025253 [-0.012878, +0.070175] |
| unseen_three/all | 3/3 | Gated GAT | 0.871477 | -0.093435 [-0.140351, -0.023032] | 0.255674 | 0/3/0 | +0.000000 [+0.000000, +0.000000] |
| unseen_three/all | 3/3 | Sequential PPO v2 | 0.893430 | -0.071482 [-0.097466, -0.032879] | 0.255674 | 0/3/0 | +0.021953 [-0.009846, +0.042885] |
| unseen_three/all | 3/3 | Page-level PPO | 0.915569 | -0.049343 [-0.140351, +0.000000] | 0.508149 | 0/2/1 | +0.044093 [+0.000000, +0.116923] |
| unseen_three/magazine | 2/2 | Native | 0.947368 | +0.000000 [+0.000000, +0.000000] | 1.000000 | 0/0/2 | +0.128637 [+0.116923, +0.140351] |
| unseen_three/magazine | 2/2 | XY-cut | 0.947368 | +0.000000 [+0.000000, +0.000000] | 1.000000 | 0/0/2 | +0.128637 [+0.116923, +0.140351] |
| unseen_three/magazine | 2/2 | Raw GAT | 0.863050 | -0.084318 [-0.098462, -0.070175] | 0.497550 | 0/2/0 | +0.044318 [+0.018462, +0.070175] |
| unseen_three/magazine | 2/2 | Gated GAT | 0.818731 | -0.128637 [-0.140351, -0.116923] | 0.497550 | 0/2/0 | +0.000000 [+0.000000, +0.000000] |
| unseen_three/magazine | 2/2 | Sequential PPO v2 | 0.856584 | -0.090784 [-0.097466, -0.084103] | 0.497550 | 0/2/0 | +0.037853 [+0.032821, +0.042885] |
| unseen_three/magazine | 2/2 | Page-level PPO | 0.877193 | -0.070175 [-0.140351, +0.000000] | 1.000000 | 0/1/1 | +0.058462 [+0.000000, +0.116923] |
| unseen_three/newspaper | 1/1 | Native | 1.000000 | +0.000000 [+0.000000, +0.000000] | 1.000000 | 0/0/1 | +0.023032 [+0.023032, +0.023032] |
| unseen_three/newspaper | 1/1 | XY-cut | 0.999977 | -0.000023 [-0.000023, -0.000023] | 1.000000 | 0/1/0 | +0.023010 [+0.023010, +0.023010] |
| unseen_three/newspaper | 1/1 | Raw GAT | 0.964089 | -0.035911 [-0.035911, -0.035911] | 1.000000 | 0/1/0 | -0.012878 [-0.012878, -0.012878] |
| unseen_three/newspaper | 1/1 | Gated GAT | 0.976968 | -0.023032 [-0.023032, -0.023032] | 1.000000 | 0/1/0 | +0.000000 [+0.000000, +0.000000] |
| unseen_three/newspaper | 1/1 | Sequential PPO v2 | 0.967121 | -0.032879 [-0.032879, -0.032879] | 1.000000 | 0/1/0 | -0.009846 [-0.009846, -0.009846] |
| unseen_three/newspaper | 1/1 | Page-level PPO | 0.992323 | -0.007677 [-0.007677, -0.007677] | 1.000000 | 0/1/0 | +0.015355 [+0.015355, +0.015355] |
| previously_seen_same_types/all | 34/34 | Native | 0.981516 | +0.000000 [+0.000000, +0.000000] | 1.000000 | 0/0/34 | +0.009146 [+0.005936, +0.012627] |
| previously_seen_same_types/all | 34/34 | XY-cut | 0.958153 | -0.023363 [-0.076197, +0.013133] | 0.377762 | 5/4/25 | -0.014217 [-0.066202, +0.021877] |
| previously_seen_same_types/all | 34/34 | Raw GAT | 0.964551 | -0.016965 [-0.020413, -0.013537] | 0.000100 | 0/31/3 | -0.007819 [-0.009830, -0.005896] |
| previously_seen_same_types/all | 34/34 | Gated GAT | 0.972370 | -0.009146 [-0.012627, -0.005936] | 0.000100 | 0/30/4 | +0.000000 [+0.000000, +0.000000] |
| previously_seen_same_types/all | 34/34 | Sequential PPO v2 | 0.973412 | -0.008104 [-0.011161, -0.005285] | 0.000100 | 0/26/8 | +0.001042 [-0.000792, +0.003114] |
| previously_seen_same_types/all | 34/34 | Page-level PPO | 0.981516 | +0.000000 [+0.000000, +0.000000] | 1.000000 | 0/0/34 | +0.009146 [+0.005936, +0.012627] |
| previously_seen_same_types/magazine | 12/12 | Native | 0.949376 | +0.000000 [+0.000000, +0.000000] | 1.000000 | 0/0/12 | +0.014975 [+0.008432, +0.021433] |
| previously_seen_same_types/magazine | 12/12 | XY-cut | 0.947585 | -0.001791 [-0.057045, +0.056024] | 1.000000 | 1/2/9 | +0.013184 [-0.039246, +0.067369] |
| previously_seen_same_types/magazine | 12/12 | Raw GAT | 0.930193 | -0.019184 [-0.025527, -0.012153] | 0.004800 | 0/9/3 | -0.004208 [-0.008187, -0.000727] |
| previously_seen_same_types/magazine | 12/12 | Gated GAT | 0.934401 | -0.014975 [-0.021433, -0.008432] | 0.004800 | 0/9/3 | +0.000000 [+0.000000, +0.000000] |
| previously_seen_same_types/magazine | 12/12 | Sequential PPO v2 | 0.939095 | -0.010281 [-0.016050, -0.004825] | 0.009199 | 0/8/4 | +0.004694 [+0.001095, +0.008923] |
| previously_seen_same_types/magazine | 12/12 | Page-level PPO | 0.949376 | +0.000000 [+0.000000, +0.000000] | 1.000000 | 0/0/12 | +0.014975 [+0.008432, +0.021433] |
| previously_seen_same_types/newspaper | 22/22 | Native | 0.999047 | +0.000000 [+0.000000, +0.000000] | 1.000000 | 0/0/22 | +0.005966 [+0.003321, +0.009112] |
| previously_seen_same_types/newspaper | 22/22 | XY-cut | 0.963918 | -0.035129 [-0.103740, +0.001408] | 0.504650 | 4/2/16 | -0.029163 [-0.097752, +0.008505] |
| previously_seen_same_types/newspaper | 22/22 | Raw GAT | 0.983292 | -0.015755 [-0.019559, -0.012155] | 0.000100 | 0/22/0 | -0.009789 [-0.011722, -0.008049] |
| previously_seen_same_types/newspaper | 22/22 | Gated GAT | 0.993081 | -0.005966 [-0.009112, -0.003321] | 0.000100 | 0/21/1 | +0.000000 [+0.000000, +0.000000] |
| previously_seen_same_types/newspaper | 22/22 | Sequential PPO v2 | 0.992131 | -0.006916 [-0.010251, -0.003986] | 0.000100 | 0/18/4 | -0.000950 [-0.002589, +0.000612] |
| previously_seen_same_types/newspaper | 22/22 | Page-level PPO | 0.999047 | +0.000000 [+0.000000, +0.000000] | 1.000000 | 0/0/22 | +0.005966 [+0.003321, +0.009112] |
| combined_exploratory/all | 37/37 | Native | 0.980170 | +0.000000 [+0.000000, +0.000000] | 1.000000 | 0/0/37 | +0.015980 [+0.008010, +0.026371] |
| combined_exploratory/all | 37/37 | XY-cut | 0.958700 | -0.021469 [-0.069211, +0.011819] | 0.376062 | 5/5/27 | -0.005489 [-0.053907, +0.029425] |
| combined_exploratory/all | 37/37 | Raw GAT | 0.959052 | -0.021118 [-0.027619, -0.015553] | 0.000100 | 0/34/3 | -0.005138 [-0.008972, +0.000206] |
| combined_exploratory/all | 37/37 | Gated GAT | 0.964190 | -0.015980 [-0.026371, -0.008010] | 0.000100 | 0/33/4 | +0.000000 [+0.000000, +0.000000] |
| combined_exploratory/all | 37/37 | Sequential PPO v2 | 0.966927 | -0.013243 [-0.020555, -0.007271] | 0.000100 | 0/29/8 | +0.002737 [-0.000255, +0.006381] |
| combined_exploratory/all | 37/37 | Page-level PPO | 0.976169 | -0.004001 [-0.011795, +0.000000] | 0.506049 | 0/2/35 | +0.011979 [+0.006695, +0.019501] |
| combined_exploratory/magazine | 14/14 | Native | 0.949089 | +0.000000 [+0.000000, +0.000000] | 1.000000 | 0/0/14 | +0.031213 [+0.012654, +0.054904] |
| combined_exploratory/magazine | 14/14 | XY-cut | 0.947554 | -0.001536 [-0.050431, +0.048021] | 1.000000 | 1/2/11 | +0.029677 [-0.018739, +0.078481] |
| combined_exploratory/magazine | 14/14 | Raw GAT | 0.920601 | -0.028489 [-0.043018, -0.016357] | 0.000600 | 0/11/3 | +0.002724 [-0.005835, +0.015038] |
| combined_exploratory/magazine | 14/14 | Gated GAT | 0.917877 | -0.031213 [-0.054904, -0.012654] | 0.000600 | 0/11/3 | +0.000000 [+0.000000, +0.000000] |
| combined_exploratory/magazine | 14/14 | Sequential PPO v2 | 0.927308 | -0.021781 [-0.038718, -0.008106] | 0.001800 | 0/10/4 | +0.009431 [+0.003055, +0.016886] |
| combined_exploratory/magazine | 14/14 | Page-level PPO | 0.939064 | -0.010025 [-0.030075, +0.000000] | 1.000000 | 0/1/13 | +0.021188 [+0.009201, +0.038192] |
| combined_exploratory/newspaper | 23/23 | Native | 0.999088 | +0.000000 [+0.000000, +0.000000] | 1.000000 | 0/0/23 | +0.006708 [+0.003842, +0.010048] |
| combined_exploratory/newspaper | 23/23 | XY-cut | 0.965485 | -0.033603 [-0.099311, +0.001374] | 0.487351 | 4/3/16 | -0.026895 [-0.092932, +0.009447] |
| combined_exploratory/newspaper | 23/23 | Raw GAT | 0.982457 | -0.016631 [-0.020637, -0.012820] | 0.000100 | 0/23/0 | -0.009923 [-0.011749, -0.008192] |
| combined_exploratory/newspaper | 23/23 | Gated GAT | 0.992380 | -0.006708 [-0.010048, -0.003842] | 0.000100 | 0/22/1 | +0.000000 [+0.000000, +0.000000] |
| combined_exploratory/newspaper | 23/23 | Sequential PPO v2 | 0.991043 | -0.008045 [-0.012057, -0.004547] | 0.000100 | 0/19/4 | -0.001337 [-0.003066, +0.000287] |
| combined_exploratory/newspaper | 23/23 | Page-level PPO | 0.998755 | -0.000334 [-0.001001, +0.000000] | 1.000000 | 0/1/22 | +0.006374 [+0.003711, +0.009500] |

## Individual unseen pages

| Page | Type | Lines / scored | Native | XY-cut | Raw GAT | Gated GAT | Sequential PPO | Page PPO |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| omni_sidebar_0679 | magazine | 34/26 | 1.000000 | 1.000000 | 0.901538 | 0.883077 | 0.915897 | 1.000000 |
| omni_sidebar_1391 | newspaper | 424/422 | 1.000000 | 0.999977 | 0.964089 | 0.976968 | 0.967121 | 0.992323 |
| omni_sidebar_1601 | magazine | 33/19 | 0.894737 | 0.894737 | 0.824561 | 0.754386 | 0.797271 | 0.754386 |

All per-seed scores, switch/fix/break counts, native-correct conditional tau and paired comparisons are in report.json; every page is in per_page.csv. Native-correct is ground-truth-conditioned. CIs are descriptive/unadjusted. Original preregistration and stop reports are unchanged.
