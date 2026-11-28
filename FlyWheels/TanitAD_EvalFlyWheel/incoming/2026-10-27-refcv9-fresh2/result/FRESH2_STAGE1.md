# FRESH2 stage 1 -- T0 label-level decel-decision read on CONFIRM (PREREG_FRESH2 s3, run once)

**F-1 (fusion confirmed): PASS** -- AUC(FUSE) - AUC(LH) = +0.0643 [+0.0423, +0.0847] (bar +0.020 and CI lower bound > 0)  
**G-1 (grown head): FAIL** -- AUC(LH_G) - AUC(LH) = -0.0051 [-0.0160, +0.0056]  
Controls: constant = 0.5000 (must be 0.5000); FUSE_perm - LH = +0.0202 [-0.0026, +0.0417] (must NOT clear F-1's bar: clears = False).  
Committed outcome (s4): F-1 and/or G-1 PASS: the passing decision (FUSE) is the decel decision for refcv9's progress profiles; stage 2 = a T1 roll pre-registration drafted and sealed BEFORE any roll

MEASURED; tier T0 (label level, never driving performance); estimator = Mann-Whitney AUC + paired episode-cluster bootstrap over CONFIRM episodes, B 2000, seed 0. The interval answers another draw of EPISODES only (one refcv8 trunk, one fit per head; heads deterministic).

Population (K-HEAD rule, PREREG_LHA0_D2 s3 code-exact): 384 CONFIRM clips, 13385 windows; S6-valid 10359, compliant 8916, GT-decel 1902, GT-cruise 4306; scored union 6208 windows / 365 episodes.

| arm | AUC | 95 % CI | AUC - LH | paired 95 % CI | n windows | n decel | n episodes | d |
|---|---|---|---|---|---|---|---|---|
| LH | 0.6339 | [0.5864, 0.6817] | -- | -- | 6208 | 1902 | 365 | 513 |
| D2 | 0.7157 | [0.6880, 0.7432] | +0.0818 | [+0.0334, +0.1262] | 6208 | 1902 | 365 | 3 |
| FUSE | 0.6981 | [0.6592, 0.7369] | +0.0643 | [+0.0423, +0.0847] | 6208 | 1902 | 365 | 516 |
| LH_G | 0.6287 | [0.5798, 0.6770] | -0.0051 | [-0.0160, +0.0056] | 6208 | 1902 | 365 | 513 |
| constant | 0.5000 | [0.5000, 0.5000] | -0.1339 | [-0.1817, -0.0864] | 6208 | 1902 | 365 | 0 |
| FUSE_perm | 0.6540 | [0.6138, 0.6933] | +0.0202 | [-0.0026, +0.0417] | 6208 | 1902 | 365 | 516 |
| neg_a0_raw (beside) | 0.7341 | [0.7071, 0.7593] | +0.1002 | [+0.0441, +0.1534] | 6208 | 1902 | 365 | 1 |
| D2_perm (beside) | 0.6239 | [0.5950, 0.6530] | -0.0099 | [-0.0616, +0.0382] | 6208 | 1902 | 365 | 3 |

## Beside (non-gating): the LON6-label GT reading (READING_FRESH2.md)

GT = LON6 label (lead pass): decel {BRAKE_TO, CREEP, HOLD} vs cruise {FOLLOW, CRUISE}; S6-valid & compliant CONFIRM windows on lead-pass clips; F-1 would read FAIL (+0.0027 [-0.0221, +0.0256]), G-1 FAIL (+0.0058 [-0.0079, +0.0191]); n 6053 windows / 1884 decel / 342 episodes.

| arm | AUC | AUC - LH [paired 95 % CI] |
|---|---|---|
| LH | 0.7680 | -- |
| D2 | 0.6866 | -0.0815 [-0.1247, -0.0400] |
| FUSE | 0.7708 | +0.0027 [-0.0221, +0.0256] |
| LH_G | 0.7738 | +0.0058 [-0.0079, +0.0191] |
| constant | 0.5000 | -0.2680 [-0.3046, -0.2291] |
| FUSE_perm | 0.7623 | -0.0057 [-0.0288, +0.0163] |

Four families: TACTICAL = this read; LONGITUDINAL / LATERAL / STRATEGIC = n/a at T0 label level (no plan is produced; stage 2's T1 roll would carry them).
LH_G: logistic hp 100.0, 13407 fit rows, sha256 8769b0b47cbcfde1... (registered before this run).
