# RESULT_V8F_FLAT: v8F-mini's hierarchy does NOT beat its own capacity-matched flat control at 2F. Outcome NOT ESTABLISHED

`Project Steering/PREREG_V8F_FLAT.md`, read by the chain's own reader (`code/v8f_flat_read.py`) on box 2. The chain ended `ZZV8FMINI-DONE
10-26 05:56` at tip `ea67b7a0`.

* **Raw:** `raw/2026-10-26-flat/flat_read.json`, plus the eval JSONs and logs.
* **On E: (`E:/Projects/TanitAD-artifacts/v8f-mini-flat/`):**
  * checkpoints `M2b_s{0,1}_flat.pt` (sha256 `8496934a…` / `6fdfc9b0…`, equal to the chain's `ckpt_sha256.json` before and after evals);
  * per-window npz `flat_eval_npz_2026-10-26.tar` (sha256 `fc03322d…`).
* **Tier:** T1-class open loop (eval-141, 2,399 windows, 141 episodes). It is **not** T1-kin.
* **Estimator:** paired episode-cluster bootstrap, B = 2,000. F = 0.1207 m (M2b's own two seeds), 2F = 0.2415 m.

## The arm

FLAT = M2b with `--o-interface cut`:
* O reads a constant in place of T's code and S's checkpoints;
* it keeps the route (nav), the ego state, the limit and the scene;
* S and T are still trained on their own losses, with the same data, the same 19,164 steps and the same eval.

The launch gate proved the cut structural: a zero gradient on `e_code`, and O invariant to S and T, each with its regression arm.

## The read

| row | s0 | s1 |
|---|---|---|
| **PRIMARY** ADE6 M2b − FLAT (m), bar ≤ −2F and separated at both | **−0.036 [−0.113, +0.035]**, n.s. | **−0.003 [−0.081, +0.076]**, n.s. |
| ADE6 M2b / FLAT (m) | 2.175 / 2.211 | 2.055 / 2.058 |
| identity: FLAT's GT-code oracle − own (must be exactly 0) | **+0.0000 [0, 0]** ✅ | **+0.0000 [0, 0]** ✅ |
| M2b's GT-code oracle ADE6 (m), beside | 0.971 | 1.004 |
| LEAD-VIOL share M2b − FLAT (`mdk`), beside | +0.016 [−0.006, +0.039], n.s. | +0.007 [−0.006, +0.021], n.s. |

**Strata, DESCRIPTIVE** (no bar; Δ = M2b − FLAT, m):

| stratum | windows / episodes | s0 | s1 |
|---|---|---|---|
| JUNCTION | 298 / 34 | −0.200 [−0.433, +0.010] | **−0.322 [−0.580, −0.025]**, separated |
| CURVE | 140 / 17 | +0.163 [−0.187, +0.476] | **+0.364 [+0.135, +0.625]**, separated (worse) |
| NO-TURN | 1,961 / 140 | −0.025 | +0.019 |

**Outcome (verbatim): NOT ESTABLISHED** (seeds MID / MID). *The hierarchy, as trained, has not shown 2F of value over flat.* **v8F is not
validated against flat at 2F.**

## What it means

* **O, as trained, gets almost nothing out of S's and T's OWN outputs** beyond what the route already gives it. Yet the same interface carries
  1.2 m of ADE6 when the code is the GT's (oracle 0.97 / 1.00 against own 2.18 / 2.05).
  * ⇒ The channel works; the code O is given at inference does not help it.
  * ⇒ That is the PREREG's committed reading: **the interface is the defect.**
* **The descriptive strata point the same way.** The hierarchy helps at junctions, where a route-level decision matters: −0.20 / −0.32 m. It
  costs on curves: +0.16 / +0.36 m.
* ⚠️ **The T1-kin result is not contradicted:** HACC beats the kinematic floor P10 by 0.73 / 0.64 m (strict). That is a claim against a
  kinematic prior, not against a learned flat planner. The FLAT read says that margin is not yet attributable to the hierarchy.
* ⚠️ **The four families:** each arm's eval JSON carries them, but the read computed no PAIRED M2b − FLAT family rows. That is a work item for
  the MIXO repeat of this read.

## Next (committed by the outcome row)

1. **Run the lever MIXO names,** then repeat this read on that arm. MIXO trains O on a mix of GT and predicted codes, so O learns to use the
   code it will actually receive.
   * Box 2 is running OACC now, the queued arm.
   * MIXO's chain waits on its killing-test mutation proof. That proof is queued on the dev box behind the HACC-MC and T1-heads CPU jobs, which
     are themselves gated on C: free space.
2. The T1-kin claim is reported as "beats the kinematic floor", never as "the hierarchy drives better", until a hierarchy-over-flat read passes.
