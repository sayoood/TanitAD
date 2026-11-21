# D-V7-TRUNK-ANCHOR — `--w-trunk-anchor` is wired, and the Observer-Effect monitor rides with it

**Date** 2026-09-03 · **FlyWheel** Architecture & Inference · **GPU** 0 (dev-box CPU only;
Thor and the A40 pod were never contacted) · **Closes** BACKLOG **R23**, the integration request
`D-V7-DINO-SEED` escalated · **Unblocks** `PREREG_V7F.md` rung **R3**'s `anchored` arm.

> ⛔ **ESCALATION, first line:** the anchor is the last of `PREREG_V7F.md` §9's three
> *"flags that do NOT exist"* on the trunk line. With it wired, R3 can run all four
> `trunk_policy` arms — but **R3 still cannot launch**, for reasons outside this change:
> `--exclude-eval-clips` (BACKLOG R8, another owner) and the fact that
> **DINOv3 ViT-B/16 is not on this box** (PI decision, `D-V7-DINO-SEED`).

---

## 1. The gap, as measured

`--w-trunk-anchor` was **DECLARED but NOT WIRED** and refused loudly at any non-zero value
(`train_v6_staged.py:8167-8215` at HEAD `41e0aba`). The refusal named two missing things, and
both were real:

| # | what was missing | why the obvious shortcut does not work |
|---|---|---|
| 1 | **a frozen second forward of the seed's own trunk** | O7's teacher cannot be reused. Three independent reasons, each sufficient and each MEASURED by source read: it is `facebook/dinov3-vitl16-…` (**1024-d ViT-L/16, a different network** — anchoring to a different network is not an anchor); `O7Distill.target()` returns **pooled READOUT CELLS** `[B, n_cells, 1024]`, not per-patch trunk tokens; and it is constructed only under `--w-o7-distill > 0` (`train_v6_staged.py:6561`), which **every v7f arm sets to 0** (`PREREG_V7F.md` §9 do-not-add list). |
| 2 | **the live encoder's patch tokens at the loss site** | `stage_a_losses` (`train_stage_a.py:274`) returns waypoints/latents, and the shared forward's `out` (`train_v6_staged.py:4216`, consumed at `:4226`) carries `z_op_win` / `z_tac` / `z_str` / `plan` — **no token entry**. `V6Stack.encode_window(..., return_tokens=True)` exists (`v6.py:5150-5167`) but the trainer never calls it: the only `return_tokens=True` call site in the repo is `v6.py:5530`, inside `V6Stack.forward`, and it fires **only** under the F-18 arm (`self.agent_slots is not None and cfg.slot_src == "tokens"`), which no v7f arm sets. |

---

## 2. Where the tokens are, and where the second forward sits

### 2.1 Where the tokens already exist

```
tanitad/models/v6.py:5150   def encode_window(self, frames, *, return_tokens=False):
tanitad/models/v6.py:5163       tok = self.encoder(flat)          # <-- the tokens, [B*W, n_tok, d]
tanitad/models/v6.py:5164       z = self.readout(tok).reshape(b, w, -1)
```

`v6.py:5163` is **the only `self.encoder(...)` call in the whole class** (probed twice: `grep`
over `v6.py` and a `Select-String -LiteralPath` pass). The tokens are therefore computed exactly
once per step, and — because `self.readout(tok)`'s backward needs them — they are **already
retained by autograd**. Holding a reference to them costs a pointer, not memory.

### 2.2 The route taken: a forward hook, armed narrowly

`EncoderTokenTap` (`train_v6_staged.py:8290`) registers a forward hook on `stack.encoder` and
captures the **exact `(input, output)` pair** of that one call. It is armed **only around the
`v6_loss_step` call** (`train_v6_staged.py:7094` arm / `:7131-7132` disarm+use, real loop;
`:5652` / `:5692-5693`, dry-run) and the
capture count is **asserted at exactly 1** — a second call REFUSES rather than anchoring an
unknown tensor.

⚠️ **Why the narrow arming is load-bearing:** `train()` calls `stack.encoder` **three more times
per step** under `no_grad`, for the O5/T1/S1 future targets (`train_v6_staged.py:6945`, `:7050`,
`:7074`). All three happen *before* the autocast block, so they fall outside the armed window.

⭐ **Capturing the INPUT, not just the output, is what makes the teacher honest.** The teacher is
fed the identical tensor the live trunk saw. The alternative — rebuilding the teacher's input from
`batch["frames"]` — is what O7 does (`_rgb = batch["frames"][:, -1, -3:]`), and that slice hard-codes
a 9-channel stack; under v7f's `--newest-frame-only --in-channels 3` it would have to be re-derived,
which is the wrong-scope failure class in miniature.

### 2.3 Two alternatives, and why not

| route | change needed | cost at v7f (§3) | verdict |
|---|---|---|---|
| **forward hook (chosen)** | none outside `train_v6_staged.py` | **+4.25 %** of encoder cost | ⭐ taken |
| second **LIVE** forward on the newest frame | none outside `train_v6_staged.py` | **+20.9 %** | correct but ~5× dearer; also recomputes activations the step already has |
| change `V6Stack.forward`'s return contract to emit `tok_win` | **`v6.py`** — a return-contract change | +4.25 % (same) | ⛔ NOT taken: a route exists that does not touch `v6.py`, and the brief makes it a last resort. Its own cost is a public API widening consumed by every caller of `out`. |

### 2.4 Where the second frozen forward sits

`TrunkAnchor.loss` (`train_v6_staged.py:8356`), called from `anchor_and_monitor_step`
(`:8704`) at the **O7-sibling site** — the same place O7/O8/O9/O10 already fold extra terms into
`L["loss"]` after `v6_loss_step` returns (`train_v6_staged.py:7136` is the O7 block, immediately
below). That site was chosen because it is the one place in this trainer where an auxiliary term
composes with the staged objective **without touching `v6_loss_step`**, which is what keeps the
default byte-identical.

The teacher is a **`copy.deepcopy` of the live encoder taken after `apply_encoder_seed` and before
the first step** (built at `:6590` real loop, `:5624` dry-run). It is therefore the **seed's own
weights**, same class, same geometry — the only construction under which the word *anchor* is
literally true.

---

## 3. The cost — MEASURED

⚠️ **Evidence class:** §3.1 is **MEASURED (ours, dev-box CPU, `torch.set_num_threads(4)`,
min of 5 reps after 2 warm-ups)**. §3.2 is **MEASURED per-image on the same CPU**, projected to the
v7f batch by **arithmetic scaling in `B`** (the encoder is linear in batch at fixed token count).
The **ratio** is the quotable quantity; the absolute seconds are CPU numbers and are **not** a
Thor/A40 estimate.

### 3.1 Whole training step, tiny CPU rig (64×128, 64-d ×3 blocks, batch 4, W=3)

| arm | s/step | vs default |
|---|---|---|
| default (`--w-trunk-anchor 0`, monitor off) | 0.16682 | — |
| monitor only (`--obs-monitor-every 50`) | 0.16269 | **−2.5 %** |
| anchor + monitor | 0.17595 | **+5.5 %** |

⚠️ The monitor's −2.5 % is **noise, not a speed-up** — its per-step work is one mean-pool and one
`[d × 32]` matmul on `B` rows, which is below this rig's timing resolution. The honest statement is
*"the monitor's cost is not resolvable at this rig"*, not *"the monitor is free"*.

### 3.2 The encoder alone, at the v7f geometry (768×12, 12 heads, patch 16, 256×640, `in_ch` 3)

| quantity | value |
|---|---|
| tokens per image | **640** (16 × 40) |
| encoder parameters | **86,138,112** (= `MODEL_REGISTRY`'s `ViTEncoder` 768×12 `in_ch=3` figure, reproduced) |
| forward, `no_grad`, 1 image | **0.891 s** |
| forward + backward, 1 image | **3.495 s** (bwd/fwd = **3.92**) |

**Projection at v7f's `--batch 8`, `--window 6` (48 images through the live trunk per step):**

| line | s (CPU, scaled) | % of the live encoder's fwd+bwd |
|---|---|---|
| live trunk, all 6 frames, fwd+bwd | 167.78 | 100 % |
| **anchor via the tap** (teacher fwd, `no_grad`, newest frame, 8 images) | 7.13 | **+4.25 %** |
| anchor via a second LIVE forward (teacher + live fwd+bwd on 8 images) | 35.09 | +20.9 % |
| anchor on **all W frames** instead of the newest | 42.77 | +25.5 % |

**Memory:** the teacher is one extra fp32 copy of the trunk — **344.6 MB**. It carries **no optimizer
state**, which is the load-bearing half: a *trainable* second trunk under AdamW would be
**1,378 MB**. Activation memory is negligible: the teacher runs under `no_grad`, so no graph is kept.

**⇒ The answer the brief asked for: this anchor adds ≈ 4 % to the encoder's cost and ≈ 5 % to a
whole step, not 100 %.** The two rejected routes are 21 % and 25 %. It is the cheap proposition, and
the reason is structural — the live tokens are already computed, so the only new work is one
`no_grad` teacher forward on `1/W` of the frames.

---

## 4. The design decisions, each with what the alternative would change

### 4.1 L2 (MSE), not cosine — as `PREREG_V7F.md` §10 D7 specifies

**Chosen, in two sentences.** The teacher is the *same architecture at the same initialisation*, so
the live and frozen token fields live in **one coordinate frame**, and an MSE is a well-posed pull
back toward the exact starting point — an *anchor*, which is exactly the word D7 uses (*"an MSE
anchor from the live trunk's patch tokens to a frozen copy of the same DINOv3 weights"*).
**What cosine would change:** cosine constrains **direction only** and leaves the token norms free,
so the trunk could rescale its features by any factor — a change the readout absorbs at zero loss
cost — and still read as fully anchored; cosine is the right choice only if a global
re-normalisation of the trunk is something we want to permit, and §6.2b's premise is that the
**frozen features' linear decodability** is what has to survive. Pinned by
`test_the_form_is_L2_and_not_COSINE`, whose fixture is a *pure rescale* (cosine = 1.0 exactly, L2 > 0).

### 4.2 The teacher is the SEED, not a second download

`--trunk-anchor-model` **stops being inert**: it is now **cross-checked against the seed's own
provenance stamp** and REFUSES on disagreement (`build_trunk_anchor`, `:6625`-region). A separately
pulled teacher could silently be a different network — reason (a) of the three that made O7's
teacher unusable — and a checked one cannot. It also means **no HF pull at train time**, which
matters under the standing HF-quota rule.

### 4.3 Newest frame only

The pull is applied to `[:, -1]` of each window, the slice O7/O8/O9 already take. Anchoring all `W`
frames costs **+25.5 %** instead of **+4.25 %** for a sample of the same distribution. Recorded in
`config.json` as `frames: "newest"`, so the choice is auditable and reversible.

### 4.4 ⚠️ At step 0 the term is 0 **up to floating-point batch-shape noise**, not bit-exactly 0

MEASURED on the tiny rig: **MSE 1.5e-14**, relative drift `<1e-5`. The live trunk runs on `B*W`
images and the teacher on `B`; a different batch extent selects a different GEMM blocking, so the
same weights on the same pixels differ in the last bits. Stated rather than papered over, because a
control that *must read a known value* has to say **which** value and **to what precision**. Making
it bit-exact would mean `W` teacher forwards to remove 1e-14.

---

## 5. The monitor (`PREREG_V7F.md` §6.2b) — as important as the term

`ObserverEffectMonitor` (`train_v6_staged.py:8449`), `--obs-monitor-every N` (default **0 = OFF**).

| | |
|---|---|
| **what it fits** | a **LINEAR** ridge on the trunk's mean-pooled patch tokens, projected into a **FIXED seeded** `d = 32` basis (fixed at construction so readings are comparable **across steps and across arms**) |
| **targets** | the **DYNAMIC** ones §6.2b names, taken from the batch: `speed` (`v0`), `steer`, `accel`. ⚠️ The **STATIC contrast** (`n_agents`) is **not** in this batch and belongs to the offline instrument — the record says so rather than leaving it assumed |
| **split** | ridge fitted on the **EVEN** rows of a ring buffer, scored on the **ODD** rows. ⛔ Alternating, **not** contiguous: the buffer spans many steps and the trunk moves across them, so a first-half/second-half split would fit an OLD trunk and score a NEW one, confounding **drift** with **corruption** |
| **λ** | **FIXED at 1.0 and never selected.** This removes the 2026-08-22 λ-selection failure class rather than guarding against it (λ on the point estimate hid a real +0.020; λ on the test set picked 1e6 and made every arm read +0.0000) |
| **controls, all four in-loop** | **constant-only** — reads **EXACTLY 0.0** by construction (Pearson ρ of a zero-variance prediction is *defined* to 0 here, which is what makes it a control that reads a known value); **raw-pixel floor** (`obs_rho_pixel`, an 8×8 pooled frame through its own fixed projection) plus `obs_beats_pixel_floor`; **time-shuffled** no-information control (`obs_rho_shuffled`); **`n` and `d` printed** every read |
| **underpower** | `n < 4·d` emits **`UNDERPOWERED`** and **no number** — n ≪ d correctly chooses maximal shrinkage and then *every* arm reads the floor (2026-08-22 failure #4) |
| **threshold** | `ratio = ρ_dynamic(step) / ρ_dynamic(first read) ≥ 0.70`, `obs_consecutive_fails` counted so §6.2b's *"fails it twice consecutively"* rule is readable from the log |
| **coupling** | ⛔ `--w-trunk-anchor > 0` **REFUSES** without `--obs-monitor-every > 0`. An anchor running with nothing watching the corruption it exists to prevent is a weight, not a defence — and R3 scores the `anchored` arm on the monitor. The monitor is usable **alone**, which is how R3's `full` and `frozen` arms carry it |
| **scope, stated in its own output** | an **IN-TRAINING SENTINEL**. §6.2b's gate read (frozen-at-checkpoint trunk, FIT/val split, static contrast, episode-cluster bootstrap) is the offline instrument. Both sentences ride in `OBS_MONITOR_FLOOR`, printed at construction and attached to every reading |

### 5.1 ⚠️ THE CAVEAT TRAVELS WITH THE READING, IN THE CODE'S OWN OUTPUT

`TRUNK_ANCHOR_CAVEAT` (`train_v6_staged.py:8267`) is **printed at construction by both**
`build_trunk_anchor` and `build_observer_monitor`, is attached to **every** monitor record as
`obs_caveat`, and is written into **`config.json`** and **`dry_run.json`** under `observer_monitor`:

> OBSERVER-EFFECT CAVEAT (BINDING, D-V7-DINO-SEED): this monitor's step-0 control does NOT measure
> the published DINOv3. The seed cannot transfer positional information — DINOv3 is RoPE-only and
> `ViTEncoder` is learned-APE-only, so `pos` is left at its own init — and the CLS / register / mask
> tokens are dropped. **No reading here may be quoted beside the published ρ 0.91 without this
> sentence.**

Pinned by `test_the_CAVEAT_rides_in_the_code_output_not_only_in_a_doc`, which asserts the sentence
is in the **captured stdout**, not merely in a constant.

---

## 6. Tests — `stack/tests/test_trunk_anchor.py`, 20 test functions / **21 test IDs**

(one is parametrized over `S-W` and `S-T`.)

| # | test | what it protects |
|---|---|---|
| 1 | `…default_constructs_NOTHING_and_registers_no_hook` | ⛔ **the load-bearing one, first half.** No anchor object, no monitor, **no forward hook after a default loss step**, one optimizer group. A hook is invisible in a loss value and in a `state_dict`, so it cannot be caught numerically and must be asserted |
| 2 | `…default_loss_is_bit_identical_to_the_PRE_ANCHOR_trainer[S-W, S-T]` | ⛔ **the load-bearing one, second half.** The pre-change trainer (resolved **by content**, the newest revision of the file lacking `TRUNK_ANCHOR_CAVEAT` — never HEAD, C75) against the current one, model held fixed. Loss bits **and** log-key set. ⚠️ **It is allowed to SKIP** — see the note below |
| 2b | `test_v6_loss_step_ITSELF_never_mentions_the_anchor_or_the_monitor` | ⛔ **the GIT-FREE half, and it always runs.** `inspect.getsource(v6_loss_step)` must mention none of `trunk_anchor` / `EncoderTokenTap` / `ObserverEffectMonitor` / `obs_monitor` / `anchor_and_monitor_step`, and `train()`'s source must still compose the term at the O7-sibling site. **That structure is *why* the default is bit-identical**, so it is asserted directly instead of being inferred from a number |
| 3 | `…nonzero_weight_with_NO_SEED_refuses_and_names_the_flag` | names `--init-encoder-from` **and** says why a seedless anchor is meaningless |
| 4 | `…nonzero_weight_WITHOUT_THE_MONITOR_refuses` | §6.2b's coupling |
| 5 | `…negative_weight_refuses` | a negative anchor is a different experiment's regression arm |
| 6 | `…teacher_model_id_is_CROSS_CHECKED_against_the_seed_stamp` | refuses `vitb16` named over a `vitl16` seed, **and passes once they agree** |
| 7 | `…anchor_on_a_FROZEN_trunk_refuses` | a flag that cannot move anything while `config.json` records it |
| 8-9 | `…tap_returns_the_EXACT_live_trunk_tokens` · `…tap_REFUSES_on_a_capture_count_other_than_one` | the tap is the live tensor, or it is a refusal |
| 10 | `…anchor_against_its_OWN_untouched_teacher_is_EXACTLY_zero` | ⭐ **a control that reads a known value** (0, to 1e-9) — any other reading means the teacher is not the seed |
| 11 | `…form_is_L2_and_not_COSINE` | on a pure rescale (cosine ≡ 1.0) the L2 is > 0 |
| 12 | `…anchor_gradient_reaches_the_TRUNK_and_NOT_the_teacher` | trunk grads non-zero; **every** teacher grad `None`; **no non-trunk parameter** touched by this term alone |
| 13 | `…teacher_stays_frozen_and_OUTSIDE_every_optimizer_group` | `requires_grad=False`, absent from `trainable` **and** from every `opt.param_groups` entry, and **byte-equal after a real `opt.step()`** at `lr 0.1` |
| 14 | `…monitor_reads_a_KNOWN_VALUE_and_MOVES_when_the_encoder_is_perturbed` | ⭐ **the deliberate-regression control.** Unchanged trunk + a linearly-decodable target ⇒ ρ > 0.9 and the read is **deterministic**; the **same probe basis** on a **different trunk** ⇒ ρ collapses below half. §6.2b: *"if the monitor does not trip `full`, the monitor is VOID"* |
| 15 | `…monitor_controls_read_their_known_values` | constant **exactly 0.0**, shuffled at the floor, pixel floor present, `n`/`d` printed, λ fixed |
| 16 | `…UNDERPOWERED_buffer_says_so_instead_of_emitting_a_number` | 2026-08-22 failure #4 |
| 17 | `…CAVEAT_rides_in_the_code_output_not_only_in_a_doc` | the sentence is in stdout and in the record |
| 18 | `…monitor_targets_are_the_DYNAMIC_ones_taken_from_the_batch` | speed/steer/accel, read off the batch |
| 20 | `…monitor_runs_ALONE_without_an_anchor` | R3's `full`/`frozen` arms; **the monitor does not touch the loss** |

⚠️ **Two of the 21 IDs SKIP under mount contention, and that is by design.** Test 2 needs a git
reference. MEASURED today: `git log -n 40 -- stack/scripts/train_v6_staged.py` took **6 s** cold and
**> 90 s** an hour later with other agents live; a `git cat-file --batch-check` over 60 commits took
**2 min 28 s**. So the lookup is bounded (`-n 40`, 45 s timeout) and returns `None` rather than
stalling the suite — *"a skipped test is honest, a self-comparison dressed as a real one is not"*.
**It PASSED earlier in this session when git was responsive** (`21 passed` standalone), and test 2b
carries the guarantee unconditionally. Set `TANITAD_REPO=<checkout>` when the suite runs from the
non-git mirror.

⛔ **ONE CROSS-FILE EDIT, ESCALATED RATHER THAN HIDDEN.** `stack/tests/test_dinov3_seed.py`
imported `assert_trunk_anchor_unwired` and asserted the *"DECLARED BUT NOT WIRED"* contract, so
wiring the flag made that module fail to **collect** — which aborts the whole selection, not just
that file. That file is **not mine**. It was edited anyway, minimally, because leaving the suite
uncollectable is worse: the import now names `assert_trunk_anchor_preflight`, and the test is
**superseded in place with the reason** — renamed to
`test_the_trunk_anchor_flag_refuses_when_its_PRECONDITIONS_are_absent`, keeping the half that
survives (a non-zero weight with no seed still refuses) and adding the monitor precondition. Its
sibling `test_o7s_teacher_cannot_be_reused_for_the_trunk_anchor` is **untouched** — all three of its
source facts remain true, and they are exactly why the teacher is a deepcopy of the seed.

---

## 7. The rung-R3 launch line

Four arms, one variable (`trunk_policy`), everything else identical. ⛔ **Illustrative, not
runnable today** — see §8.

```bash
# common to all four arms
COMMON="--stage S-W --v2-cache <b1 cache> --require-parity --v2-lru 64 \
  --s2-labels <v72>/labels/s2_labels_v7.2_train.jsonl.gz --w-s2-goal 1.0 \
  --nav-labels <same> --nav-cond \
  --exclude-eval-clips <eval index>            # BACKLOG R8, another owner \
  --init-encoder-from <seed.pt>                # the DINOv3 ViT-B/16 seed \
  --newest-frame-only --in-channels 3 \
  --enc-dim 768 --enc-depth 12 --enc-heads 12 \
  --patch 16 --frame-h 256 --frame-w 640 --projection cylindrical --frame-hfov 120 \
  --o5-form l1 --w-o5 1.0 --w-o6 0.1 --o5-target ema --ema-decay 0.996 \
  --sigreg-subspaces 32 --sigreg-slices 512 --spectrum-accum 4096 \
  --cond-param omega_accel_v --w-o14 1.0 --o14-mode fut --o14-k 4 \
  --o5-k 60 --bptt-truncate 4 --rollout-grad-checkpoint on \
  --w-o1-ctrl 0 --w-o1-fact 0 --w-o1-scene 0 --w-o2 0 --w-o3 0 \
  --w-o7-distill 0 --w-o8-pixel 0 --w-o9-ema 0 --w-o10-psg 0 \
  --w-o11-cf 0 --w-o13-ego 0 \
  --steps <one epoch> --batch 8 --lr 1e-4 --clip 1.0 --seed 0 \
  --save-every 1000 --log-every 50 --param-budget 300000000 --print-launch \
  --obs-monitor-every 250                      # ⭐ ON IN EVERY ARM"

# ⭐ the anchored arm — the one this change unblocks
$COMMON --out <exp>/v7f-r3-anchored --trunk-lr-scale 0.1 --trunk-lr-warmup-steps 2000 \
        --w-trunk-anchor 1.0 \
        --trunk-anchor-model facebook/dinov3-vitb16-pretrain-lvd1689m

# ⛔ the deliberate-regression arm — the monitor MUST trip it, or the monitor is VOID
$COMMON --out <exp>/v7f-r3-full     --w-trunk-anchor 0

# the second regression arm — must reproduce the dead command channel
$COMMON --out <exp>/v7f-r3-frozen   --w-trunk-anchor 0 --freeze-encoder

# last-k
$COMMON --out <exp>/v7f-r3-lastk    --w-trunk-anchor 0 --trunk-lr-scale 0.1 \
        --trunk-lr-warmup-steps <k-schedule>
```

⚠️ `--obs-monitor-every 250` is on **every** arm on purpose: the monitor's own regression arm is
`full`, and an instrument that is only run on the arm it must clear proves nothing.
⚠️ The `lastk` arm's one variable is **not** expressible with today's flags (`--trunk-lr-scale`
is whole-trunk); a per-block LR policy is a separate, unowned change. **Named, not narrowed.**

**Preflight that costs no GPU:** append `--dry-run --dry-steps 2` to any arm. The dry-run
**exercises the seed, the anchor and the monitor** (`train_v6_staged.py:5624`, `:5693`) — a
pre-launch verifier that skips a flag the launch carries cannot catch that flag's failure class.

---

## 8. ⛔ UNVERIFIED — stated, not buried

1. **No v7 run exists, and none was started.** Every number here is CPU, synthetic, or source-read.
   Nothing in this change has met the B1 corpus.
2. **No real DINOv3 seed was used.** The tests build a *structurally valid* seed from the live
   encoder's own tensors, deliberately, so the anchor is tested rather than the converter.
   **DINOv3 ViT-B/16 is not on this box** (`D-V7-DINO-SEED`: only ViT-L/16 and dinov2-base), and
   the converter never downloads. ⇒ the anchor has **never been run against real DINOv3 weights**.
3. **The monitor's absolute ρ on real driving data is unknown.** The 0.70 threshold is §6.2b's,
   inherited; what this change adds is the instrument and its floors, not a calibration. Its first
   real reading must be taken beside the caveat in §5.1 — **it is not ρ 0.91's trunk**.
4. **The v7f seconds in §3.2 are CPU** and scale-projected in `B`. The Thor/A40 ratio should be
   re-measured with `--dry-run --dry-steps` before the ladder is scheduled (`PREREG_V7F.md` §8
   already requires this for the arm cost).
5. **`--trunk-lr-scale` interaction untested at scale**: the anchor and the discriminative-LR split
   are independent flags and were tested independently, never jointly on a real trunk.
6. **R3 remains blocked** on `--exclude-eval-clips` (R8) and the ViT-B/16 pull (PI).
7. **The git-referenced identity test skipped in the final full-selection run** (git contention on
   the G: mount). It **passed** standalone earlier in the same session. Its git-free counterpart
   (test 2b) ran in every pass. Re-run with `TANITAD_REPO` set and a quiet mount to see it green.
8. **`dry_run()` does not seed the global RNG**, so two identical default dry-runs differ
   (MEASURED: 4.0784 vs 4.1158). Noted because it looks exactly like a regression and is not one —
   it is pre-existing and unrelated to this change. Not fixed here: it is not this change's file
   scope and a seeding change would move every dry-run's numbers.

---

## 9. PROPOSED REGISTER ROW — `D-V7-TRUNK-ANCHOR`

> ✅ **D-V7-TRUNK-ANCHOR — THE DISTILLATION ANCHOR IS WIRED, IT COSTS ≈ 4 % OF THE ENCODER RATHER
> THAN DOUBLING IT, AND THE OBSERVER-EFFECT MONITOR NOW READS DURING TRAINING (2026-09-03, ArchInf
> FlyWheel, 0 GPU; **20 new CPU test functions (21 IDs)**, selection `v6|staged|dinov3_seed|trunk_anchor`
> **3 F / 999 P / 26 S** against a **3 F / 980 P / 24 S** baseline — the same three pre-existing
> failures **by name and by identical assertion text**
> (`test_v6_chain::test_the_WHOLE_LADDER_executes_end_to_end_on_cpu`,
> `test_v6_chain::test_the_dry_ancestor_is_never_named_ckpt_pt`,
> `test_v6_st_launch_fixes::test_E4_the_resolution_is_banked_with_its_reasons`), **0 new**; BACKLOG
> R23 closed).** `--w-trunk-anchor` was DECLARED-BUT-NOT-WIRED and refused at any
> non-zero value, so `PREREG_V7F.md` rung R3's `anchored` arm could not run. It now applies an
> **L2 (MSE) pull of the live trunk's per-patch tokens toward a FROZEN `deepcopy` of the SEED's own
> weights** — `PREREG_V7F.md` §10 D7's form exactly — on the **newest frame** of each window.
> **The two blockers the old refusal named are both discharged:** the second frozen forward is the
> seed's own trunk (O7's teacher is unusable for three independent reasons — ViT-L/16 not ViT-B/16,
> pooled readout CELLS not per-patch tokens, and never constructed at `--w-o7-distill 0`), and the
> live patch tokens are taken by a **forward hook armed only around `v6_loss_step`**, asserted at
> **exactly one capture** because `v6.py:5163` is the class's only `self.encoder(...)` call — so the
> trainer reads the tensor the step already computed instead of paying for it twice, and `v6.py` is
> **not touched**. ⭐ **THE COST, MEASURED and answering the question the brief posed:** at the v7f
> geometry (768×12, 256×640, 640 tokens, batch 8, W 6) the anchor adds **+4.25 %** of the live
> encoder's fwd+bwd (one `no_grad` teacher forward on 1/W of the frames) and **+5.5 %** of a whole
> tiny-rig step; the rejected second-LIVE-forward route costs **+20.9 %** and anchoring all W frames
> **+25.5 %**. Teacher memory is **344.6 MB** fp32 with **no optimizer state** — a *trainable* second
> trunk under AdamW would be **1,378 MB**. ⭐ **THE MONITOR SHIPS WITH THE TERM, and `--w-trunk-anchor
> > 0` REFUSES without `--obs-monitor-every > 0`**: an anchor with nothing watching the corruption it
> exists to prevent is a weight, not a defence. The monitor is a linear ridge on the trunk's tokens
> against speed/steer/accel with a **FIXED, never-selected λ**, a constant-only control that reads
> **EXACTLY 0.0**, a raw-pixel floor, a time-shuffled control, `n`/`d` printed, and an
> **UNDERPOWERED** refusal at `n < 4d`; it reports `ratio_to_step0` against §6.2b's **0.70** and
> counts consecutive failures. It runs **alone**, which is how R3's `full` and `frozen` arms carry
> it. ⚠️ **THE CAVEAT IS CARRIED BY THE CODE, NOT BY A DOC:** every construction prints, and every
> reading and `config.json` embeds, *"this monitor's step-0 control does NOT measure the published
> DINOv3 — DINOv3 is RoPE-only and `ViTEncoder` is learned-APE-only, so `pos` is left at its own
> init, and CLS/register/mask are dropped; no reading may be quoted beside ρ 0.91 without this
> sentence"* — pinned by a test that asserts it in **captured stdout**. ⚠️ **Defaults preserve today
> EXACTLY:** at `--w-trunk-anchor 0 --obs-monitor-every 0` nothing is constructed and **no forward
> hook is registered** (asserted directly — a hook is invisible to a numeric comparison), and the
> loss bits and log-key set are pinned against the newest pre-change revision resolved **by content**,
> not by HEAD. ⚠️ **An honest imprecision, MEASURED and recorded rather than papered over:** at step 0
> the anchor reads **1.5e-14**, not bit-exactly 0, because the live trunk runs on `B*W` images and the
> teacher on `B` and a different batch extent selects a different GEMM blocking. ⭐ `--trunk-anchor-model`
> **stops being inert**: it is cross-checked against the seed's provenance stamp and REFUSES on
> disagreement, so the run row and the anchor cannot name different networks — and no HF pull happens
> at train time. ⛔ **UNVERIFIED: no v7 run exists, no REAL DINOv3 seed was used** (ViT-B/16 is still
> not on this box — PI decision pending), the v7f seconds are CPU per-image measurements scaled in
> batch, and **R3 still cannot launch**: `--exclude-eval-clips` (BACKLOG R8) is another owner's
> deliverable. ⚠️ **NOT NARROWED, NAMED:** R3's `lastk` arm has no flag — `--trunk-lr-scale` is
> whole-trunk, and a per-block LR policy is a separate unowned change. ⛔ **ONE CROSS-FILE EDIT,
> DECLARED:** `stack/tests/test_dinov3_seed.py` imported the deleted `assert_trunk_anchor_unwired`,
> so wiring the flag made that module fail to **collect** and aborted the whole selection. It is not
> this FlyWheel's file; it was edited anyway, minimally — the import renamed and the *"DECLARED BUT
> NOT WIRED"* test **superseded in place with the reason**, keeping the half that survives (a
> non-zero weight with no seed still refuses) and adding the monitor precondition. Its sibling
> `test_o7s_teacher_cannot_be_reused_for_the_trunk_anchor` is untouched: all three of its source
> facts still hold, and they are exactly why the teacher is a deepcopy of the seed. ⚠️ **The
> git-referenced half of the identity test SKIPPED in the final full run** (`git log -n 40 -- <file>`
> went 6 s → >90 s under concurrent agent load on the G: mount); it passed standalone in the same
> session, and a **git-free structural test** — `v6_loss_step`'s own source must mention none of the
> new machinery — carries the guarantee unconditionally in every run.

---

## 10. Deliverable manifest

| artifact | where it lives | only one place? |
|---|---|---|
| the wiring (anchor, tap, monitor, flags, refusals) | `stack/scripts/train_v6_staged.py` (repo, **staged**) | **no** — repo + `C:\Users\Admin\tanitad-wt` run mirror (md5-verified identical) |
| the tests (20 functions / 21 IDs) | `stack/tests/test_trunk_anchor.py` (repo, **staged**) | **no** — repo + run mirror |
| ⛔ the cross-file edit | `stack/tests/test_dinov3_seed.py` (repo, **staged**) — **not my file**, edited because wiring the flag broke its import; see §6 | **no** — repo + run mirror |
| this SPEC | `TanitAD Research Lab/Architecture & Inference/Implementation/incoming/2026-09-03-trunk-anchor/SPEC.md` (repo, **staged**) | **yes** — repo only |
| the cost probe + its raw output | same folder: `cost_probe.py`, `COST.json` (repo, **staged**) | **yes** — repo only |

**Nothing lives only in a worktree, only on a pod, or only in this agent's context.**
**Nothing was committed and nothing was pushed.**
