# R8-7 X-items status at tip `8522e49` (X3 · X4 · X7 · X10) — 2026-10-11

Audit by an Architecture & Inference agent. The agent's Write tool returned no classifier verdict, so the Master Mind
wrote this file from the agent's hand-back. Every count below was re-read from `raw/` before writing.

The audit ran on a **pure-tip** copy of `8522e49`. The unlanded `--grad-share-terms` / `--grad-share-at` patch was
excluded; its overlay-only run is `raw/pytest_X4_OVERLAY_only_not_at_tip.txt`.

**Evidence classes:**
* **MEASURED** — the agent ran it; the raw output is under `raw/`.
* **READ** — code read at file:line.
* **INHERITED** — not re-derived.

| item | fix implemented? | measure instrument? | missing → cheapest next step | in R2 ladder? |
|---|---|---|---|---|
| **X3** speed ceiling / max-speed input | **YES** (READ):<br>• `--r8-speed-input n2\|n3` at `refc_v3_train.py:11975`<br>• trained unknown row p = 0.45 at `refcv8_train.py:75`<br>• v8 future-max sidecar refused by trainer and launch gate<br>`raw/pytest_X3_tip8522e49.txt`: **143 passed** (MEASURED). | **PARTLY.**<br>• The ceiling is an **argmax filter** (`refc.py:3872`), not a clamp: a row with no compliant candidate keeps its whole fan.<br>• Leak (MEASURED, inherited instrument): **0.0193** fed 4-way bins and **0.0348** 8-step, against the 0.05 bar.<br>• SPEC's **CEIL-OBEY** (+0.5 m/s margin) has **no matching instrument**. The nearest is the ladder's `ceil_violation`, which is strict (> 1e-6). | Write CEIL-OBEY with SPEC's margin and **measure the exempt-row rate** (rows where no candidate complies). Refresh SPEC §8.3 / §8.10, which still say "NOT BUILT". | **YES.** Every arm feeds n2 at 0.45, plus V-VSHUF, L4b and the `vmax_off` / `vmax_shuf` eval rows. |
| **X4** loss budget + label-state isolation | **Isolation YES** (READ): the policy travels on the labels (`v7_labels.py:216-471`); trainer scope at `refcv8_train.py:879-893`. `raw/pytest_X4_tip8522e49.txt`: **27 passed** on real blobs (MEASURED).<br>**Budget NO**: `--w-tac-v6` is still 1.0, and the launch gate lists R8-BUDGET as open. | **Gradient-share instrument** exists (`grad_share.py`). Its cost fix is **unlanded** (SPEC_WPB_LADDER_A2).<br>**No LABEL-SCOPE census** exists, only a single-window test. | Land the gs cost fix (in flight), then run **L2** (V-TACk), which sizes the `--w-tac-v6` budget from the measured share. Write a LABEL-SCOPE census over the whole train set. | **PARTLY**: gs readings on every arm, and L2 V-TACk. |
| **X7** distance keeping + agent store range | **Store range NO, by ruling.** The "x ≤ 61 m store" is `vis1.STORE_SCOPE` (`vis1.py:68`). It is unmoved, and the Master Mind ruled it OUT (Q4). | **Metric YES** (PLAN's "no metric anywhere" is STALE): `taniteval/taniteval/lead_metrics.py` (headway / time-gap / TTC), wired into `four_families`. `raw/pytest_X7.txt`: **100 passed, 1 skipped** (MEASURED).<br>The eval139 lead block covers **139 / 139** clips: 27,942 frames at 10 Hz, **7,973 LEAD**, 6,230 NO_LEAD, 10,204 NOT_STRAIGHT, 3,535 NO_LABEL. Its **horizon is only 2.0 s** (K = 10, dt 0.2) (`raw/lead_block_vs_eval139.out.txt`, MEASURED). | **LEAD-VIOL** is not implemented. The lead block needs a **6 s** horizon (`build_lead_block_b1.py --k 30 --dt 0.2`). There is **no refcv7 @ 50,400 distance-keeping baseline row**. | **NO.** |
| **X10** pose-to-image timing, night, left turns | **Pose sync YES** (READ): one shift per window (`pose_sync.py`; `--pose-sync-sidecar` at `refc_v3_train.py:10787`). `raw/pytest_X10_tip8522e49.txt`: **27 passed** (MEASURED). POSE-SYNC 0.067 ms is **INHERITED**. | **NIGHT-LUMA NOT built.**<br>**LEFT-N** reports **window** n, not clip n.<br>The standalone eval loader is **uncorrected** (no pose sync). | Build NIGHT-LUMA (brightness-stratified night). Report left turns as clip n. Apply the sidecar in the eval loader. | Pose-sync flag on every arm **only**. |

## What SPEC_REFCV8 can bar today vs what needs work first

**Barrable now (the instrument exists and is tested):**
* X3 leak (≤ 0.05);
* X3 ceiling compliance, if the SPEC adopts the strict `ceil_violation` or the margin instrument is written first;
* X4 isolation;
* X7 headway / time-gap / TTC at a 2 s horizon;
* X10 pose sync.

**Needs work before its bar can be committed:**
1. **X7.** Three pieces:
   * LEAD-VIOL;
   * the 6 s lead block;
   * the refcv7 @ 50,400 baseline row.

   Without them, the LONGITUDINAL family's distance-keeping half has no baseline to bar against.
2. **X4.** The budget is not a number until L2 reads the tactical gradient share, so the ladder is the critical path.
3. **X3.** The CEIL-OBEY instrument must match SPEC's +0.5 m/s margin, and the exempt-row rate must be reported, because
   an argmax filter silently passes rows with no compliant candidate.
