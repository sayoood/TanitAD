# The refcv6 perception heads are a tested island — nothing bridges them to the trainer

**Date:** 2026-09-17 · **Evidence class: MEASURED** · **Tier: n/a (a source measurement, not a run)**

## The instruction this is measured against

> *"Let plan to train jointly the resnet-trunk, the bev map (based on the sam2 maps as gt) and a
> head for 3d bounding boxes extracted from the resnet trunk ... I would also recommand to give
> additionally access to the truink for the diffusion planner"* — PI, 2026-09-16

The heads were built, tested and landed. ⛔ **They are not wired into the trainer.** A training run
launched today would optimise none of them.

## What was measured

`code/probe_percep_wiring.py`, against `scripts/refc_v3_train.py` on the current tip. Every count is
a positive assertion carrying a same-breath control that must read non-zero.

| probe | count |
|---|---|
| `map_soft_ce` called in the trainer | **0** |
| `box3d_set_loss` called | **0** |
| `collate_map_targets` / `collate_box_targets` called | **0 / 0** |
| `MapGTStore` / `zh_for_frame` referenced | **0 / 0** |
| trainer imports of `bev_encoder` · `box3d_head` · `bev_lift` · `perception_targets` · `semantic_map_gt` | **0 · 0 · 0 · 0 · 0** |
| `_w_map` **assignments** (it is **read** once, by the conflict detector) | **0** |
| `_w_box3d` **assignments** (read once) | **0** |
| CLI flags `--w-map` · `--w-box3d` · `--trunk-pretrained` · `--no-trunk-pretrained` | **0 · 0 · 0 · 0** |

⭐ **And the fact that makes it pointed rather than merely absent.** `compute_losses_v3` is **757
lines** and already assembles **thirteen** loss terms:

`loss_cls · loss_goal · loss_gp · loss_gstr · loss_lat · loss_lat_tac · loss_law · loss_lon ·
loss_lon_tac · loss_route · loss_sel · loss_traj · loss_u0`

— trajectory, classification, law, route, selection, goal-point, strategic, and **both** tactical
terms. **Not one of them is a perception term.** This is not a trainer that lacks a loss machinery;
it is a rich loss machinery with the PI's two requested heads absent from it.

⚠️ `loss_applied` is bookkeeping and `loss_forward_reduce_cuda_kernel_2d` is a substring of a CUDA
error message, not a term — both are listed in the raw JSON and neither is counted above.

## The second gap, in the same family

`trunk_pretrained` is a **real config field** — `refc.py:348` declares it with the comment
*"timm only; False is the knockout arm"*, `refc.py:1439` consumes it, and the trainer stamps it into
`config.json` (2 sites). ⛔ **There is no CLI flag**, so the ImageNet-versus-random-init knockout
cannot be run — the one `PREREG_REFCV6_V2.md` §4 calls *"not optional"*, because it is the only
control that separates *"pretraining helped"* from *"this architecture is better"*.

## ⚠️ A correction to how I first reported this

My first probe used a 220-line `awk` window after the `def` line and reported that
`compute_losses_v3` *"assembles only `loss_cls`, `loss_lat`, `loss_lon`, `loss_traj`"*. **That is
wrong** — the window truncated a 757-line function at roughly its first third. The correct scope is
the AST source segment, and the real list is the thirteen above. ⛔ The finding itself never
depended on it: the six call-site counts and five import counts are whole-file greps and are
unaffected.

⚠️ **The probe also caught two of its own defects before reporting**, both via the same control:

1. a regex requiring **double quotes** around `loss_*` read **zero** loss keys;
2. its replacement was written through a shell that turned `\b` into a **literal backspace byte
   (0x08)**, so the word-boundary pattern matched nothing.

Both times the control `loss_traj ∈ loss_keys` read 0 and refused to emit. ⭐ **Without it, both
would have produced a confident, tidy, empty answer** — and the second would have strengthened a
conclusion I already believed, which is the direction in which a broken probe is most dangerous.

## Consequence for the pre-registration

⛔ **`E-REFCV6V2-PERCEP` cannot be tested at all** until this is wired — the hypothesis asserts the
map and box heads *"are LEARNED and reach the trunk"*, and today no gradient from either reaches
anything. ⛔ **`E-REFCV6V2-TRUNK`'s ImageNet knockout cannot be run** without the flag.

⇒ both belong on `PREREG_REFCV6_V2.md` §11's OWED list, and **no GPU arm should be spent on the
perception claims before the wiring lands**.

## Status

⭐ **Not left as a diagnosis.** An implementation agent is wiring it: the two CLI flags plus the
pretrained pair, the targets into the batch (honouring `require_map_coverage`'s floor), and both
losses into `compute_losses_v3` — under **bit-identity when the weights are 0.0** and with
**per-head gradient-reach** reported, against the `tac_goal_tok_head` precedent (11,286 parameters,
`grad_abs_sum` exactly 0 for all 40,284 steps: parsed, stamped, reaching nothing).

## Files

| path | what |
|---|---|
| `code/probe_percep_wiring.py` | the probe, controls included |
| `raw/percep_wiring.json` | its output |
