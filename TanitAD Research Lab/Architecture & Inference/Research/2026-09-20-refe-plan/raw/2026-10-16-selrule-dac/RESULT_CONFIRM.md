# RESULT: SELRULE-DAC navtest confirm. **ADOPT, as the committed table defines it, and practically inert: +0.02 PDMS, 11 of 12,146 picks changed, 4 of 380 DAC zeros rescued.**

Master Mind, 2026-10-18.

The prereg is `PREREG_SELRULE_DAC.amended.md`: section 4, plus Amendment 1 (A1 to A7), written before the fit. `FIT.json` is frozen, sha256 `44d6d2c1…`. The family leg is completed under `ADDENDUM_A8_FAMILY_LEG.md` (sha256 `b4bf39ee…`), written after the PDMS leg was seen and before any family number.

## Runs
| run | dir | outcome | why |
|---|---|---|---|
| 1, the pre-registered confirm (watcher, 04:40Z) | `confirm_out/`, `confirm_run.log` | UNRESOLVED (family leg not computed) | The watcher staged the table without `proposals.npz`, so the seam fingerprint was unavailable. |
| A8 attempt 1 | `confirm_out_a8_attempt1_relpath/` | UNRESOLVED | My invocation passed a relative `--out-dir`, and `families6.py` runs as a subprocess from another cwd (FileNotFoundError in `families/*.log`). |
| **A8** | `confirm_out_a8/`, `confirm_run_a8.log` | **ADOPT** | Absolute `--out-dir`; `proposals.npz` built from banked bytes, as A8 states. |

A8's control holds. The PDMS leg of all three runs is identical:
- ΔPDMS x100 +0.0235 [+0.0002, +0.0546];
- 11 picks changed;
- zero tokens 509 → 505, DAC=0 380 → 376, NC=0 133 → 133.

## Result (T1 NAVSIM v1.1, P2's full table, paired log-cluster bootstrap, 136 logs, 10,000 resamples, seed 20260927; MEASURED)
- V(0.8) vs B: **ΔPDMS +0.0235 [+0.0002, +0.0546]**. EP +0.0103 [−0.0106, +0.0366], TTC +0.0000.
- No adverse PDMS sub-score.
- Family leg (`families6.py` on both executed-pick seams, `proxy_eval.fam_adverse`): **no adverse component**.
- Oracle controls: O-DAC +3.184 (90 % of the +3.54 SEL ceiling), O-NC +0.989, O-both +4.043.
- Selection quality: random 73.00, actual 88.818 → 88.842, best-of-64 98.231.
- TACTICAL: selection quality above.
- STRATEGIC: PAUSED (PI 2026-09-19, item 25) and unavailable in NAVSIM.

## Reading
1. **The row is ADOPT, and it is reported as written.** Lower bound > 0, no adverse sub-score, no adverse family component.
2. **The magnitude decides what it is worth:**
   - It captures 0.0235 of an O-DAC reach of 3.184, i.e. **0.7 %**.
   - The lower bound sits 0.0002 above zero.
   - The τ was fit on the u400 scorer and transfers to P2 by assumption (A1).
   - The interval answers *another draw of logs* only. P2 is one training seed (H-ESTIM-SEED-1), and with 11 changed picks a training replicate could easily move it.
   - ⇒ Adopting it costs nothing and buys nothing measurable. Deploying it is the PI's call.
3. **This is the expected outcome of the fit's own finding** (`REPORT.md`, 2026-10-15): the DAC failures are confident (median p_DAC 0.93), so re-weighting the scorer's own heads cannot reach them.
   - Rule Zero: the zero-tail lever is not selection re-weighting. It stays P6, the trunk DAC gradient.
   - The full table adds a larger pool: 5.88 of the 9.41-point oracle gap is ranking among NON-zero proposals (`raw/2026-10-16-p2-full-table/RESULT.md`, BACKLOG `SEL-RANK`).

## Process defects (mine, both disclosed in A8 or here)
- The watcher staged one of the two files `confirm.py` reads.
- The first A8 re-run used a relative path that a subprocess resolved against another cwd.
- Neither changed a number. Both made a pre-registered read report UNRESOLVED for a reason outside the experiment.

## Manifest
- Repo: this file, `ADDENDUM_A8_FAMILY_LEG.md` (+ `.sha256`), the three run dirs (CONFIRM.json/.md, families JSON and logs), and both run logs.
- Seam npz (8 files, ~31 MB): dev box only, in the run dirs, sha256 in `CONFIRM_SEAMS.sha256`.
