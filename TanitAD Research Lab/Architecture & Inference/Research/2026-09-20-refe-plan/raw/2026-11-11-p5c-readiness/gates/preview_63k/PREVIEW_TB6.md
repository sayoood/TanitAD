# TB-6 preview on the dev-box-labelled rows, and a full-bank projection

2026-11-12, Master Mind. A diagnostic, **not the gate**.
- PREREG_P5C's table gates run on the full 150,603-row table on the pod, after P7.
- No bar was moved.
- Population: navtrain bank rows (`train_grow_cmd_3c_p7`, sha256 rank0 `a5690f81...`, rank1 `8b6ccd1b...`). No navtest.

## Why
- The gates agent's subset run (30 logs, 4,393 rows, `../subset_run/gates.json`) read **TB-6 rank-0 FAIL**: heading rate **2.018×** (bar ≤ 2×), acc 1.706×.
- Rank 0 FAIL ⇒ STOP and P5-R (PREREG_P5C §4.2). So the MM checked the same gate on every row the dev box could label.

## The reading of TB-6 is fixed, not chosen
"Median per-row max |x| of the labels against the same rows' traj" is a **ratio of medians**. That is P5-S's own rule, which TB-6 takes to bank scale:
```
p5_target_labels.py:388: median(label) > 2 * median(traj)
```
The median of per-row ratios is logged as a diagnostic only.

## MEASURED: 63,495 labelled rows (`gates.json`, `run.log`; `p5c_table_gates.py` on the dev-box label shards)

| rank | n | heading-rate ratio | acc ratio | TB-6 on these rows |
|---|---|---|---|---|
| 0 | 38,578 | **1.9395** | 1.6813 | PASS (thin) |
| 1 | 24,917 | 1.7291 | 1.6513 | PASS |

- TB-2, TB-3, TB-5 and TB-7 PASS.
- TB-4 FAILS by construction: 58 % of rows have no dev-box DB and read as raw fallback.
- TB-0 and TB-1 were not run in this call. TB-1 passed separately, 1018/1018 (`../TB1_control_1018.json`).

## MEASURED: the unlabelled rows are straighter (`projection.json`, `project_tb6.py`)

Raw-`traj` median of max heading rate, °/s:

| rank | labelled | unlabelled | full bank |
|---|---|---|---|
| 0 | 4.937 | **3.563** | **4.052** |
| 1 | 9.058 | 8.901 | 8.966 |

## ESTIMATED: full-bank projection

**Method:**
- The labelled rows are reweighted to the unlabelled rows' raw distribution, in 40 quantile bins of the same metric.
- Assumption (UNVERIFIED): the label given the raw-metric bin is the same across logs.
- The projected label median is divided by the full-bank raw median.

| rank | heading-rate ratio | acc ratio |
|---|---|---|
| 0 | **2.091** | 1.680 |
| 1 | 1.717 | 1.652 |

⇒ **TB-6 rank 0 is projected to FAIL at bank scale.** That would mean STOP P5-C and the committed next lever P5-R.
- **Mechanism:** a denominator effect. On near-straight rows, Tc20s's heading pre-distortion is large relative to a tiny raw heading rate.
- The subset (labels 7.97 °/s on raw 3.95) and P5-S (8.98 on 5.11, ratio 1.757) show the same shape.

## Consequence (rule zero)
P5-R is the card's committed next lever: a regularised inversion, r20 = argmin of the 8-knot tracked error + λ‖Δ²r20‖², with its own pre-registration before code. It is being pre-registered and prototyped now (hold `scorer_gym_20261107/p5r/`), so that the pod gate's verdict finds the next arm ready.
- If the pod's full table **passes** TB-6, P5-C proceeds unchanged and P5-R stays banked.
- If it **fails**, P5-R is the label.

## Driver change for the pod run
`p5c_labels.py` now writes `knot_maxdev` and `knot_checks` on every labelled row, so TB-2 is audited per row on the pod. The 1e-9 bar was already enforced per row at line 134. Label bytes are unchanged. New sha256 is `77724709...`, pinned in `p5c_pod_label.sh` and `POD_SHIP_LIST.txt`.
