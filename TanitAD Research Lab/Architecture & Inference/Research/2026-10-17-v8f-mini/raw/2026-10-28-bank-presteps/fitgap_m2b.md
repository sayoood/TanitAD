# H-DD-FITGAP, M2b only (PARTIAL; deliverable 3)

Frozen cards sha256 `3202a62fbf7699005c60f7a6ae37fb544e4bf0a0f0564b7a9115ee7bf865fd5b`; amendment 1 sha256 `4676bdb8d27c1095c942782889ab9a04932e5aa1cfe1fcda650285ca365b93c7`.
Evidence MEASURED (ours; `fitgap_m2b.json`). TRAIN side = M2b's own rollout states (the DAgger bank) on TRAIN GT-launch windows; EVAL side = the banked T1-kin dump of plain M2b ('B', m = 10, eval-141, reference only, never a selector). Statistic = R2 (executed-chord speed over the last 0.2 s before the tick), ticks 1-2 mean of v_own - v_GT, per launch window; D = -C is the speed deficit (m/s). Estimator: cluster bootstrap over launch-bearing clips / episodes, B = 2000, seed 0 (answers 'another draw of clips / episodes' only; blind to training and inference variance).

Controls PASS: True (my eval-side R2 reproduces the M-LAUNCH instrument's own DAG-OACC R2 mean to < 1e-9 on 327 windows / 49 episodes).

| checkpoint | C_train (m/s) [95 % CI] | C_eval, plain M2b dump (m/s) [95 % CI] | r = D_train / D_eval [95 % CI] | class (frozen: >= 0.5 FIT, <= 0.25 GEN) |
|---|---|---|---|---|
| M2b s0 | -1.2540 [-1.3033, -1.2042] (49618 win / 1891 clips) | -1.0202 [-1.2791, -0.7408] (327 win / 49 ep) | 1.229 [0.976, 1.680] | FIT-LIMITED |
| M2b s1 | -0.9105 [-0.9607, -0.8591] (49618 win / 1891 clips) | -0.5649 [-0.8669, -0.2104] (327 win / 49 ep) | 1.612 [1.023, 4.416] | FIT-LIMITED |

**PARTIAL verdict:** 2 of 6 checkpoints observed (['FIT-LIMITED', 'FIT-LIMITED']); the card's >= 5-of-6 rule is not evaluable. FIT-LIMITED needs 3 of the remaining 4; GEN-GAP needs UNREACHABLE. Remaining: DAG-OACC s0, DAG-OACC s1, REP s0, REP s1.

Config mismatch (disclosed): the plain-M2b dumps ran `--limit logged`, the bank rollouts `--limit causal`. Beside, on the DAG arm that exists in both forms, C_eval(logged) - C_eval(causal) = s0: +0.2992 m/s, s1: +0.3049 m/s (enters no verdict).
Beside (ESTIMATED, no verdict): shifting the plain-M2b eval deficit by the DAG arm's logged-to-causal change gives r = s0: 0.950 (FIT-LIMITED), s1: 1.047 (FIT-LIMITED). R1 (state speed) on the TRAIN side, beside: s0: -1.2562, s1: -0.9067 m/s.

Cost of the other 4 checkpoints (ESTIMATED from a 290 s / 2,399-window dev-box CPU log): TRAIN launch windows on all clips = 49618; launch-windows-only roll-outs ~1.7 h per checkpoint (all clips), ~0.17 h on a 10 % clip subset; all windows of a 10 % clip subset ~1.0 h per checkpoint.
