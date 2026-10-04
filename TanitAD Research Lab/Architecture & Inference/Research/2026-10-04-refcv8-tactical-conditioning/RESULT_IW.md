# RESULT — SPEC_WPB's identity gate I-W FAILED on Thor; root cause; fix; re-run

*WP-B, 2026-10-04. SPEC_WPB was registered at 12:33:09Z (sha256 `c952d4b4…`). Its I-W criterion is unchanged; this note
reports the failure as written, then the defect behind it.*

## 1. The failure (MEASURED, `raw/I_W.FAILED_unsplit.json`)

The registered I-W ran on 218 grid windows on the Thor GPU (the chain's first job, 16:36–17:22 UTC).

| part | criterion | result |
|---|---|---|
| seams attached, no extension | bit-identical | **PASS**: 0 / 218 windows differ; score Δ 0.0 |
| allocation 32 + prior-free group, emission off | plan + base fan bit-identical, base scores ≤ 1e-5 | **FAIL**: plan differs on 205 / 218 windows, base fan on 218 / 218, base scores by up to 2.1e-5. **sel_idx identical on 218 / 218** |

Under the reading rule ("I-W fails ⇒ stop") the chain stopped, and no arm trained.

## 2. Root cause (MEASURED, `raw/I_W_diag.json` and `raw/r8_fullsize_warmstart.json`)

**GPU measurement** (Thor, unsplit tree, 48 grid windows):

| comparison | max \|Δ plan\| | max \|Δ base fan\| | max \|Δ score\| | sel_idx differs |
|---|---|---|---|---|
| C0: T0 against T0 (determinism control) | **0.0** | 0.0 | 0.0 | 0 |
| T0, cuDNN-TF32 on against off (the instrument's own float sensitivity) | 3.6e-4 m | 2.2e-3 m | 9.1e-4 | 0 |
| allocation 32 only | 1.9e-5 m | 1.5e-4 m | 3.1e-5 | 0 |
| prior-free group only | 4.6e-5 m | 1.1e-4 m | 1.3e-5 | 0 |
| both | 7.6e-5 m | 1.4e-4 m | 1.4e-5 | 0 |

Each extension row reads the same with TF32 off.

**What the GPU measurement shows.**
- **The GPU path is deterministic:** C0 reads exactly 0.
- **The cause is the extension.** Extending the candidate axis (117 → 117 + extras) moves the BASE candidates' floats by
  up to 7.6e-5 m in the plan. That is 5–15× smaller than the effect of the GPU's own TF32 switch.
- **The pick does not change.** sel_idx was identical on every window.

**Mechanism.** The decoder ran ONE pass over all candidates. A different GEMM shape selects different kernels, so the
low bits of every candidate move. The decoder has no candidate-candidate interaction, so the semantics are unchanged;
only the floating-point results move.

**CPU confirmation.** The same effect appears at full size on CPU (dev box, `raw/r8_fullsize_warmstart.json`,
variant B): plan and base fan not bit-equal, base scores ≤ 5.7e-6, sel_idx identical.

**Why the SPEC missed it.** The CPU smoke-size rig on which the SPEC's criterion was written happened to stay
bit-stable at its shapes. The criterion was derived from too small a rig. That is WP-B's error, logged here.

## 3. The fix (code, not criterion)

The decoder now decodes the base slice and the appended extras in two passes (`AnchoredDiffusionDecoder._r8_split`).
The two slices are concatenated back, and per-candidate state follows them: phi is sliced, and the X1h query and the F3
per-layer lists are re-concatenated. Every GEMM over the base candidates therefore has refcv7's exact shape.

* **Rig (CPU):** the extended fan (allocation + prior-free) is now **bit-identical, scores included**
  (`tests/test_refcv8_warm_start.py`; the score bound tightened from ≤ 1e-5 to exact equality). Split and one-pass agree
  to 1e-4 on every candidate, sel_idx included (a new test). The split changes numerics, not the function.
* **Full size (CPU, refcv7-50,400 via the trainer's `--init-from` rule, eval139 via the loader's v9 join):**
  allocation 32 with emission off is **bit-identical** on 4 / 4 windows. Plan, base fan and scores all differ by 0.0
  (`raw/r8_fullsize_warmstart_split.json`).
* **Thor (CPU, the real R1 cache, 4 grid windows):** I-W PASS on both parts, all differences 0.0. The T2 train (3 steps)
  and eval (2 windows) smoke runs end to end on the fixed tree.
* **Cost:** two decoder passes per sampler step instead of one; the FLOPs are the same, only the kernel launch count
  rises. The registered timing phase measures it (B-COST).

## 4. Status

The registered I-W is re-running on the Thor GPU on the fixed tree (`/home/nvidia/refcv8_wpb/tree_c397`) with the SAME
criterion, as the first job of `wpb_chain2.sh`. It is queued under the GPU lock behind R1's H2s eval.
- If it PASSES, the chain proceeds exactly as registered: timing, then the arms.
- If it FAILS, the chain stops again.

The failed artifact is kept unchanged as `arms/I_W.FAILED_unsplit.json`.

**Outcome (MEASURED, 2026-10-04, Thor GPU, `raw/I_W.json`, md5 `1154accc…`): PASS on the SAME registered criterion**
(spec sha256 `c952d4b4…`, 218 grid windows, 19:23–19:57 UTC):
- seams only: 0 windows differ, max |Δ base score| 0.0;
- allocation 32 + prior-free group with emission OFF: 0 windows differ on traj, sel_idx and the base fan, and max |Δ base score| is **0.0**.

The chain then ran its timing phase (`timing_wpb.json`) and its arms, as registered.
