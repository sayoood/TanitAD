# READINESS: can the trainer build HB (a LoRA trunk copy feeding only the head) today?

**Answer: NO. Nothing of HB is built, and two other things the A5 draft needs are not reachable either (Stage 0's loader, HC's 1 m cell and boundary weight).**
Read-only audit, 2026-11-12, of git tip `874ac6aa` (`PK` = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`), files `PK/refe/scorer_finetune.py`
(3,076 lines), `model.py` (846), `aux_drivable_head.py` (509), `vdac_probes.py` (174), and `PK/raw/2026-11-09-scorer1p6-prereg/vdac/s1a_read/s1_read.py` (663).
Evidence class of every line reference: MEASURED by grep/read of those files at that tip. Nothing was built, run or edited. Companion: `PREREG_VDAC_P6_A5_DRAFT.md`.

## 1. The trace today: flag -> model -> forward -> loss -> backward (HA path)

| hop | what happens | file:line |
|---|---|---|
| flag | `--aux-drv`, `--w-aux-drv`, `--aux-drv-tap {visual,scene}`, `--aux-drv-depth`, `--aux-drv-feed`, `--aux-drv-labels`, `--aux-drv-shuffle-seed`; trunk-related options: **none** | `aux_drivable_head.py:110-118`; `grep -ci lora` on `scorer_finetune.py`, `aux_drivable_head.py`, `vdac_probes.py` = **0, 0, 0** |
| model build | `REFe(cfg)`, `load_for_inference`, `model.eval()` | `scorer_finetune.py:1736-1739` |
| freeze | **every** parameter `requires_grad False`; only `score_q_mlp`, `score_dec`, `score_head` (`SCORER`, `:91`) set True | `:1740-1747` |
| scorer arms | `base` frozen copy, `armB` trainable copy of the scorer; both read the captured ctx | `:1749-1759` |
| capture | a forward pre-hook on `model.score_dec[0]` stores its ctx argument as `cap["sctx"]` = `visual_ctx` | `:1760` |
| head build | `AuxDrivable(AuxDrivableConfig(tap, depth, feed, w_aux, seed))` as a sibling; **optimiser `optH` over `aux.head.parameters()` only** | `:1766-1770` |
| forward | `model(img, ego, goal, ...)` runs P6's trunk (no gradient: all frozen); `sctx = cap["sctx"]`; `actx = sctx` (the mutants may replace it); `aux.predict(actx)` | `:1975-1986` |
| model forward | `visual_ctx = scene_proj(backbone(flat) + pos3d)` | `model.py:743-759` |
| loss | `aux.loss(...)` then `aux.combine(0, L_aux)/div`, `tot.backward()` | `scorer_finetune.py:2118-2130`; `aux_drivable_head.py:323-330` |
| backward reach | the graph stops at `sctx` (no grad needed there): the head's 2.44 M params are the only thing `L_aux` moves | by the freeze `:1740-1747` |
| save | `aux_drv.pt` = head (+ feed) state; scorer states in `scorer_states_last.pt` | `:2359-2361`, `:3042-3046` |

**Consequence: with the code as it stands the trunk cannot receive a gradient from `L_aux`, and there is no second trunk.** An HB arm launched from today's flags is HA-s0 under another name (a lever that does not exist: the 2026-09-10/11 class).

## 2. What HB needs (not built; the exact list)

| # | item | where | detail |
|---|---|---|---|
| H1 | flags `--aux-drv-trunk-lora {off,on}` (default off) and `--aux-drv-lora-lr` | `scorer_finetune.py` argparse + `aux_refusals` (`aux_drivable_head.py:121-139`) | refuse the lr without the flag and the flag without `--aux-drv`, and the flag with `--aux-drv-feed on` |
| H2 | build the copy: `copy.deepcopy(model.backbone)`; every param `requires_grad False` except the `.A` / `.B` tensors of `LoRALinear` (`model.py:216-232`); keep it in a sibling container, not in `model` | after `:1747` | 3,145,728 trainable (ARITHMETIC, section 3) |
| H3 | the copy's ctx: re-run the six lines of `model.py:743-759` on the copy (`copy(flat)`, `+ model._pos3d(img, N, calib)`, `model.scene_proj(...)`) **inside `scorer_finetune.py`**, not by editing `model.py` | `fwd()` near `:1982` | keeps `model.py` byte-identical (it is in the pinned closure); `actx` = the copy's ctx **before** the mutant line `:1983-1985` so G19's existing mutant lands on it |
| H4 | optimiser: a second AdamW group (LoRA, own lr, weight decay 0.01), own grad-clip norm; the lr schedule follows the head's shape | `:1770`, `:2959` (`lr_ratio`) | |
| H5 | gradient accumulation of the copy's backward (micro-batch < 8) | `aux_backward(lv, div)` already takes `div` (`:2118`) | UNVERIFIED that batch 8 fits (ESTIMATED: P6's launch peak was <= 44 GB of 48; HB adds a second trunk) |
| H6 | save and load: `aux_drv.pt` format `aux_drv/2` = head + the copy's A/B tensors + both optimiser states; `scorer_states_last.pt` `"AUX"` likewise | `:2359-2361`, `:3042-3046` | **no `aux_drv.pt` loader exists anywhere** (`grep` finds the file name only on the save side), needed by G-EVAL, Stage 0 and S2 |
| H7 | G12's tap row: `outs[1]` of `scene_proj`'s hook is "the visual_ctx" (`:2408-2414`); a second `scene_proj` call makes the copy's ctx `outs[2]` | `:2408-2414` | re-point and assert `scene_proj_calls == 3` |
| H8 | new gate G21 + `vdac_probes.judge_g21` (pure) + `launch_gate.consume_scorer_preflight` re-judge + `gate_checks.SCORER_GATES` entry | `vdac_probes.py`, `gate_checks.py`, `launch_gate.py` | as G19/G20 were added |
| H9 | profile arms HB-s0 / HB-SHUF in `vdac_argv(arm)`, G-DVB rows (A5 section 6), `vdac_postgate.py` Adam-state check extended to the LoRA tensors, chain + pause scripts (timeouts, export of the LoRA files, HF upload) | `launch_gate.py`, `vdac_postgate.py`, `vdac_chain.sh`, `vdac_pause_p7.sh` | the pause timeout of 18,000 s per 10,000-update arm is below HB's upper bound |
| H10 | the in-sample dump at the end of every arm (the Stage 0 frames), reusing the `h48x` forward pattern (`:3050-3068`) | after `:3050` | |
| H11 | `s1_read_v2.py` (manifest-driven arms, paired HB-minus-HA-s0, per-arm controls) | the hold, then `PK/eval/` | A5 section 2.3 |

## 3. Stage 0 and HC: also unreachable today

| # | item | evidence |
|---|---|---|
| S0 | `vdac_stage0.py` (a measuring mode: load `aux_drv.pt` strictly, forward the in-sample draw, the init head on the extra-log frames, write the reader's npz schema, import the frozen reader's `frame_stats` / `city_iou_read` with its sha asserted) | no loader (H6); the trainer's `main()` defines `fwd` as a closure (`:1971`), so a separate script cannot import it: the cheapest form is a `--stage0-*` mode inside `scorer_finetune.py` that trains nothing and returns before the loop (`--pair-calibrate` is the precedent, `:1728`) |
| C1 | `--aux-drv-cell` | `AuxDrivableConfig.cell` and `DrivableAuxHead(cell=...)` exist and `block_stats` supports 1.0 (`aux_drivable_head.py:76-90, 142-150, 194-205`), but the trainer builds the config **without** `cell` (`scorer_finetune.py:1767`) and `add_aux_args` has no cell flag (`:110-118`). A declared 1 m arm would silently build 2 m. The head metrics at `:235` and the reader (`s1_read.py:56, 111-119, 289, 454, 566-567`) assume 40 x 32 |
| C2 | boundary-weighted Huber | `aux_loss` (`:218-232`) has no per-cell weight; add one with default `1.0` everywhere (a multiplier that must reproduce the unweighted loss bit-for-bit when off) |
| C3 | HC reader | A5 section 2.3 |

## 4. Tests to write (none exist)

| test | assertion | mutation that must go RED |
|---|---|---|
| T1 LoRA live | on a tiny REFe with the real LoRA classes and the real 2.4 M head: after one `aux_backward`, **every copy LoRA tensor has a non-zero gradient** | `REFE_AUX_MUTATE_DETACH_HEAD=1` (head cut from loss) and a mutant where the head reads P6's ctx instead of the copy's |
| T2 exact-zero control | `--w-aux-drv 0.0`: copy LoRA gradients are **exactly 0.0, not None** (`combine` always multiplies, `aux_drivable_head.py:326-330`) | a short-circuit at `w_aux == 0` (reads None) |
| T3 P6 untouched | after a backward, every `model` parameter outside `SCORER`: `.grad is None`; sha256 of `model.state_dict()` equal before/after 2 optimiser steps; scorer logits bit-identical to a head-less run | unfreeze P6's `scene_proj`; add `model.backbone` params to `optH` |
| T4 step-0 identity | `ctx_hb == visual_ctx` exactly (same autocast) with the copy's LoRA initialised from P6's | initialise the copy's `B` to zero |
| T5 copy independence | the copy's storage differs from `model.backbone`'s (`data_ptr`); editing a copy LoRA tensor does not change `model(...)` | the copy shares storage (shallow assignment) |
| T6 no scorer path | `autograd.grad(scorer loss, copy LoRA)` is None for every tensor | route `sctx` from the copy into the scorer |
| T7 G19 on HB | G19 passes; RED under `REFE_AUX_MUTATE_EGO=1` and under the new `REFE_AUX_MUTATE_EGO_TRUNK=1` (ego added before `scene_proj`) | both switches |
| T8 G12 tap | `tap_is_visual_ctx` reads the copy's `outs[2]`; RED if it reads `outs[1]` while HB is on | the stale index |
| T9 G21 judge | raw fields: pass, each field refused singly, dead control refused | |
| T10 ckpt | `aux_drv/2` round trip: head, LoRA, both Adam states; 2 resumed updates bit-identical; strict load refuses a head-only file in HB mode | drop the LoRA optimiser state |
| T11 Adam state | post-training gate: every head and LoRA parameter holds state | a LoRA group never stepped |
| T12 cell flag (HC) | declaring `--aux-drv-cell 1.0` builds 80 x 64; the weighted loss with weight 1 equals the unweighted bit-for-bit; a probe cell inside / outside the band gets weight 4 / 1 | build 2.0 under the flag; drop the weight |
| T13 Stage 0 | the tool reproduces a banked head dump on a fixed frame list (C0-ID); the y-flipped read is lower; the in-sample list is a pure function of the store and the sha rule | a list that includes an H48 log |
| T14 reader v2 | regression identity on the S1a inputs (all four values and CIs digit for digit); its mutation arms still RED; per-arm controls; HC's pooled-to-2 m IoU on a synthetic fixture | the S1a `no_reweight` and `frame_resample` mutations |

Run order: tests on CPU (`CUDA_VISIBLE_DEVICES=-1`), one lane at a time, then all ~69 launch_gate-reading tests, then the pod pre-run (`suite-prerun`; a test that passes off-pod can fail on the pod, memory note `pod-gate-test-workspace-dependence`).

## 5. ESTIMATED build size (ESTIMATE, not measured)

| piece | lines (code / tests) |
|---|---|
| trainer: H1-H7, H10, S0 mode, C1, C2 | ~260 / ~200 |
| gates: G21 + judge, profile arms, G-DVB rows, postgate, fingerprint-pinned test updates | ~230 / ~170 |
| chain and pause scripts (timeouts, exports, HF files) | ~50 / ~30 |
| reader v2 and its tests | ~150 / ~150 |
| **total** | **~690 / ~550, about 1,240 lines** |

About **one agent for one working day** (~150-200k tokens), or two Sonnet agents with disjoint files (trainer + gates vs reader + Stage 0), plus the pod suite pre-run (CPU, ~minutes to an hour). Order that unblocks the PI soonest: **S0 first** (Stage 0 tool and its in-sample result decide HB-alone versus the pair), then the reader v2 freeze, then HB, then HC only if Stage 0 reads WEAK. Nothing here needs a GPU to build.

## 6. Risks found while reading (each is a reason to test before launch)

- G12 passes on the wrong tensor if `outs[1]` is left as is (H7): the HB analogue of the "tested but unreachable" class.
- `REFE_AUX_MUTATE_EGO` adds the ego token to `actx` at `:1983-1985`; if H3 assigns the copy's ctx **after** that line, G19's mutant tests the wrong object and the gate cannot see a leak into the copy.
- The `aux_buf` held-out forward and `run_evaluate` call the same `fwd()`; the copy path must be inside `fwd()` so S1's dumps and training see the same ctx (one code path, as the card's G-EVAL requires).
- Memory: a ViT-L forward with trunk backward at 4 x 1,920 tokens beside P6's own no-grad forward and three scorer copies has not been measured; the preflight's speed/memory probe is the instrument and its result goes to the PI before the cap is fixed.
- P6's LoRA-group learning rate is not in the files read (`train.py` shows only the global `--lr` default 2e-4 at line 2143); the HB lr in the draft (5e-5) is a HYPOTHESIS to be confirmed by the MM from P6's argv.
