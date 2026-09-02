# COMMS — E-DEPLOY-O2PRE-E2E

`2026-09-02 · Research Lab · Deployment & Optimization`

## ⛔ ESCALATED TO THE MASTER MIND AND THE DeployFlyWheel

Per the operating standard: escalate integration; do not write "please update the
row" into a doc nobody re-reads.

### E1 — `Deployment & Optimization/BACKLOG.md:18` is stale, and so is the stream's STATE.md

MEASURED this pass: the O2-pre **repair is in the working tree**
(`readout.py`'s `_adaptive_avg_matrix` + non-persistent `pool_mh`/`pool_mw`
buffers + the matmul forward branch), and `stack/tests/test_readout_onnx_pool.py`
exists (7,089 B). The BACKLOG row still reads **`🔴 … 🔵 next, 0 deps`**, and
`Deployment & Optimization/Research/STATE.md` has **zero** hits for `O2-pre` and
zero for `176x624` — the stream's state file has no entry for its own next item.

⇒ Move the row to ✅ with **both** links (the shipped repair, and this package for
the end-to-end test), and add an O2-pre line to STATE.md.
**Owner:** DeployFlyWheel.

### E2 — ⭐ The row's actual ask was unmet until now, and that matters more than the stale marker

The row asks for an *"export test **at the deployed geometry**"*. What shipped
exports a **bare `SpatialGridReadout`** against a `[1, 429, 32]` token tensor at a
toy `d_model = 32` — it never constructs an encoder and never exports from an
image. The false claim it exists to falsify is an **encoder** claim.

This package supplies the missing assertion: `image → ViTEncoder →
SpatialGridReadout` at `[1, 9, 176, 624]` **exports**, ORT vs eager `6.87e-07`,
with a **negative control that fails with the original Thor error verbatim** and a
**tiling control at 256×640 that still passes**.

⇒ **Recommend promoting `code/export_encoder_readout_deployed.py` into
`stack/tests/`** as an end-to-end companion to `test_readout_onnx_pool.py`. It is
written as a runnable script with a JSON artifact; converting it to three
parametrised test functions is mechanical. ⚠️ Per the stream's own boundary rule
(`Research/STATE.md`: never write `stack/` directly; fixes go through
`Implementation/incoming/` intake), **the Lab has NOT written into `stack/`** —
this is a request, not a merge.
**Owner:** DeployFlyWheel (owns the intake).

### E3 — ⛔ `…/Implementation/onnx_export/export_encoder_predictor.py` has no geometry flag

The 2026-07-08 "exports clean" script hard-codes `[1, 9, 256, 256]` and its
argparse exposes only `--ckpt/--out-dir/--report/--opset`. **There is no way to
run it at the shipping geometry**, which is precisely how the false claim was
produced. Whatever else is decided, a geometry argument belongs on it — otherwise
the next "exports clean" claim will again be true at exactly one geometry.
**Owner:** DeployFlyWheel.

### E4 — the export guard skips silently where onnx is absent

`test_readout_onnx_pool.py` is guarded by `pytest.importorskip("onnx")` /
`importorskip("onnxruntime")`. That is deliberate — but a **skipped guard protects
nothing**, and the project's assumption that onnx is Thor-only is out of date:
MEASURED this pass, the dev venv carries `onnx 1.22.0` and `onnxruntime 1.27.0`.
⇒ Either run it in CI here, or make the skip **loud** (report the skip in the
suite summary) so an unprotected export path is visible rather than silent.

## Backlog motions taken this pass

* **Proposed L-14** (Deploy) — delete the dead `nn.AdaptiveAvgPool2d` still
  *constructed* at `readout.py:89` on the non-exact path. Inert today (`forward`
  never calls it there, and F3 confirms it does not reach the graph), but it sits
  on exactly the path where it used to be the blocker.

⛔ Not self-ranked — the finder proposes; the Master Mind ranks.

## What was deliberately NOT done

* ⛔ **No TensorRT engine build, no Thor access.** The A40 pod (REF-C v3 H-arm,
  ~35 h remaining) and Thor were untouched, per this pass's compute rules. The
  `#4590` silent-FP32-fallback finding and `quant_gate.py` (backlog row 1) still
  gate every precision claim, and nothing here weakens them.
* ⛔ **No write into `stack/`**, per the stream's intake boundary. The end-to-end
  script lives in this package's `code/`.
* **Commit status of the O2-pre repair is unverified** — `git log` on that path
  timed out repeatedly against the G: mount, and per the environment rule an empty
  or failed git result is not evidence of anything.

## Asks lane

`LAB_ASKS.md` carried **no OPEN row** at this pass (verified twice — a Python read
of the file, and `python stack/scripts/lab_ask.py --list`, which reports
`ASK-1 ANSWERED`). No ask was answered or opened by this package.
