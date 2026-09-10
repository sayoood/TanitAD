# refcv5-v2: THREE heads never received a gradient, not one — and one of them is provably inert

**Date:** 2026-09-10 · **Agent:** FlyWheel (HF upload task) · **Evidence class:** MEASURED (ours)
**Artifact:** `C:/Users/Admin/refcv5v2_final/ckpt.pt` (md5 `9405ec73b2d797c4cebd44f82dbce54b`,
sha256 `a42d68af682521cd8f2f8ca2c4d5949a34e352ce1074351ebb16cc16e9dce57d`), also published as
`Sayood/tanitad-refc-v5:ckpt_40284.pt` (PRIVATE).
**Probe:** `scratchpad/probe_zerograd3.py` (copied here as `probe_zerograd3.py`).

## The claim this corrects

The banked note and the model card both said **one** head — `tac_goal_tok_head`, 11,286 params —
received zero gradient across the 40,284-step run. That is true and **incomplete**.

## What was measured

`torch.optim.Adam` allocates per-parameter state **lazily**, on the first `step()` at which
`p.grad is not None`. A parameter registered in `param_groups` with **no entry in
`opt["state"]`** therefore never received a gradient. This checkpoint registers **357**
parameters against **351** optimizer state entries.

| parameter | shape | numel | weights at step 40,284 |
|---|---|---|---|
| `core.decoder.offset_head.weight` / `.bias` | (16, 384) / (16,) | 6,160 | default init, untrained |
| `tac_goal_tok_head.net.weight` / `.bias` | (22, 512) / (22,) | 11,286 | default init, untrained |
| `scorer.goal_point.weight` / `.bias` | (2, 512) / (2,) | 1,026 | ⛔ **EXACTLY ZERO** |

**18,472 parameters never trained.**

## Why the id→name mapping is trustworthy (it is the part that could have been wrong)

The optimizer's `state_dict` keys are positional ids, not names. The mapping was recovered by
gap-tolerant **shape alignment** against the model's `state_dict` order, and validated by three
checks that are independent of the thing being measured:

1. **351 of 351** present ids match their aligned parameter's shape **exactly**.
2. The mapped total is **108,257,502**, equal to `config.json:param_breakdown.total` — an
   independently authored number, not one derived from this probe.
3. **Same-breath controls** read TRAINED: `str_goal_head` (`exp_avg_abs_sum` 7.59305),
   `scorer.cand_bias` (5.47e-4), `core.decoder.conf_head` (nonzero). A global zero would have
   made the result meaningless; it is excluded.

The three tensors the buffer filter over-counted are named and excluded explicitly
(`core.lat_log_prior`, `core.lon_log_prior`, `core.decoder.anchor_controls` = 240 params).

## ⭐ `scorer.goal_point` is STRUCTURALLY INERT — an analytic target, not an estimate

`stack/tanitad/models/v6.py:2760-2762` (and again `:2979-2983`) **zero-initialises** this head:
`nn.init.zeros_` on both weight and bias. Zero-init **plus** zero gradient makes an analytic
prediction: the tensor must still be **exactly** zero at step 40,284.

**It is:** `absmax = 0`, **0 of 1,024** weights and **0 of 2** biases nonzero.

⇒ The "free regression" half of the goal scorer emits the **constant origin** for every input.
It contributed nothing to any published number. This is an identity, not a noisy measurement —
no seed changes it.

⚠️ **Open, and it is the one that matters:** the panel reports
`goal_setting FDE 0.6607 m / bearing_MAE 1.3730 deg`. Whether that row reads **this** head or the
param-free `goal_point.anchor_goal_prior_at_time` path (`refc.py:1638` states that prior is
param-free) is **NOT established**. Until it is, that row is UNVERIFIED. If it does read this
head, the row is a measurement of a constant.

## `core.decoder.offset_head` — MEASURED untrained, HYPOTHESIS for the cause

Defined `refc.py:1534` as the per-anchor offset head and consumed in the classifier pass at
`refc.py:1987`. This arm runs `--sampler ddim`; the sampler pass `refc.py::_decode_ctrl` reuses
`traj_proj`/`layers`/`conf_head` and refines through **`control_head`**, and its own docstring
says the only new parameters in that seam are `control_head` and `time_mlp` — `offset_head` is
not in that path.
⇒ **HYPOTHESIS:** the ddim configuration orphans the classifier-pass offset head, so this is
expected rather than a defect. ⛔ **Not confirmed.** The discriminating check is cheap: an arm
with `--sampler` unset should show `offset_head` **with** optimizer state.

## Why this outranks a documentation fix

`--sel-refined` is one of the levers this arm was built to test, and refcv5-v2 **failed its
pre-registered bar** (`os − ha0_ext` **+0.0205** [+0.0043, +0.0390], separated the wrong way;
confirmed at a second inference seed, +0.0204 [+0.0047, +0.0395]). A refinement head sitting at
random init inside the arm whose refinement lever failed is a **candidate mechanism for the
failure**, and it is one nobody had looked at. It is not yet an explanation — `offset_head` may
be legitimately bypassed by ddim — but it is the next thing to check, and it is cheap.

## Next levers, cheapest first

1. **Settle the `goal_setting` provenance** (read-only, minutes): does the panel's goal row read
   `scorer.goal_point` or the param-free anchor prior? Decides whether a published row is a
   constant.
2. **Confirm the `offset_head` hypothesis** (read-only): check any non-ddim refc arm's checkpoint
   for `core.decoder.offset_head` optimizer state.
3. **Re-run the same probe across every banked refc checkpoint.** The probe is generic — absence
   of Adam state names any silently-untrained head — and this defect class has now appeared twice
   (`intent_proj`, `tac_goal_tok_head`) before this pass found three more instances at once.

⇒ **Escalation:** item 3 is an instrument, not a note. `probe_zerograd3.py` should move into
`stack/scripts/` and run as a post-training gate, so a head that never trained is caught at the
checkpoint rather than after the arm has been evaluated and published.

## Status of the artifacts

The corrected finding is **already published** in the model card at
`Sayood/tanitad-refc-v5:README.md` (PRIVATE), §4a — verified by sha256 content roundtrip
(`e1c2fada714972f346e5221d50fd72f96a4ff088b1189b7f501c831720340838`), so it is not stranded.
⛔ This directory is **written, NOT staged** — the task brief explicitly forbade git operations.
The Master Mind should stage it.
