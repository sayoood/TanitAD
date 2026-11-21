# resnet101 at 416 x 1024 on the 8 GB dev box — what fits, and what it costs

**Date** 2026-09-18 · **FlyWheel** Architecture & Inference · **Box** dev box, RTX 4060, 8188 MiB
(~1.0 GB held by other tenants throughout) · **Rig** `qland/cover139d.sh`'s command line — all heads
live, real v2 caches, real SAM3 map GT, real 3-D agent join, real extrinsics — at `--batch 1`,
`--steps 3` (and `--steps 10` for the timing arms), no eval pass.

---

## The answer, in one line

**YES.** `resnet101.a1_in1k` trains at 416 x 1024 on this 8 GB card at **2.887 GB peak allocated /
3.73 GB reserved, 4.9 s/step** (MEASURED, `raw/r101_b1_fbn_ck1_s10.json`) — a **2.28-day**
40,284-step arm. The lever is **leading-batch gradient checkpointing of the trunk** (chunk 1), and
the single thing it changes about the arm is **BatchNorm's batch composition**, which is why the
fitting arm also pins BN to its ImageNet running statistics.

⭐ It is **faster than today's resnet34 rig**, not slower: resnet34 unpatched is **10.75 s/step**
(5.01 days) because it is silently spilling past the card. See §1 — that finding reframes everything
below.

---

## ⛔⛔ 1. The finding that makes every other number readable: this card silently oversubscribes

On Windows/WDDM the CUDA allocator **spills past the 8 GB of VRAM into system RAM instead of
raising**. So on this box **"the arm ran" is not "the arm fits"** — it runs, at PCIe speed, and the
only thing that says so is the clock.

MEASURED, trunk-only, at the rig's true shape (`raw/micro_wb8.json`):

| trunk arm | peak alloc | peak **reserved** | s/iter | what actually happened |
|---|---|---|---|---|
| resnet34, unpatched | 6.64 GB | **9.18 GB** | 5.83 | ran — **spilling** |
| resnet34 + chunk ckpt 1 | 2.26 GB | 2.29 GB | **0.85** | resident — **6.9x faster** |
| resnet101 + bf16 | 14.74 GB | 16.12 GB | 17.61 | ran — thrashing |
| resnet101 + freeze stem+layer1 | 19.58 GB | 20.18 GB | 25.06 | ran — thrashing |
| resnet101, unpatched | — | — | — | **OOM at 22.37 GB** |

⇒ **The fit criterion used throughout this note is `peak_reserved <= 7.1 GB`** (8188 MiB total minus
the ~1.0 GB other tenants hold), **not "it did not raise"**. Every row below is stamped `fits` on
that rule, and every one is cross-checked against its s/step: a spilling arm is always 4-10x slow,
so the two instruments agree in every case.

⚠️ **This retro-reads a MEASURED number already in circulation, and the retro-read is itself
MEASURED.** `28.30–29.2 s/step` for resnet34 at 416 x 1024 is not a compute cost — it is a
**host-spill** cost. Reproduced here at the same `--batch 2`: **29.1 s/step at 14.63 GB allocated /
18.27 GB reserved** on an 8 GB card (`raw/r34_b2_ref.json`). The same configuration with the lever of
§5 is **2.512 GB / 2.8 s/step** (`raw/r34_b2_fbn_ck1.json`) — a **10.4x** speed-up with no trunk
change at all. **Today's production resnet34 rig has never been resident on this card.**

---

## 2. What the trainer actually offers — read, not assumed

⛔ **`scripts/refc_v3_train.py` has NO grad-checkpoint, NO AMP/autocast, NO channels_last and NO
gradient-accumulation flag.** Greps for `autocast`, `GradScaler`, `channels_last` and `accum` return
0 hits in it. The `--enc-grad-checkpoint auto` named in the brief belongs to
`scripts/train_v6_staged.py` (the **v6 encoder**, a different model); the only `grad_checkpoint`
seams in this repo are `cfg.encoder.grad_checkpoint` on the REF-B / flagship / v6 encoders. **The
refcv6 trunk has no equivalent flag.** Every lever measured here is therefore applied as a
monkeypatch from outside (`raw/rig_arm.py`), so a measurement could not silently become a permanent
behaviour change to a trainer other arms share.

**The shape that decides the whole question.** `--arm hier` calls the trunk **once** on
`frames.reshape(b*w, ...)` (`stack/tanitad/refs/refc.py:3889`), and
`refc_v3_sized_config("tiny", hier=True).core.window == 8`. `TimmResNetTrunk.forward_features`
then folds again into `b*w*K`. So **at batch 1 the trunk processes 24 images of 416 x 1024**, not 3
— confirmed live in every arm (`trunk_input_shape [8, 9, 416, 1024]`). This matches the independent
MEASURED count in `2026-09-17-refcv6-e2e-1024/RESULT.md` §10 (`W x K = 24`). The trunk alone at
WB=1 costs 3.86 GB on resnet101; at the real WB=8 it costs 22.4 GB and dies.

---

## 3. The levers, ranked by MEASURED effect (trunk-only, WB=8, batch 1, 416 x 1024)

`raw/micro_wb8.json`, `raw/micro_wb8_stagegc.json`, `raw/micro_wb8_frozenbn.json`. All
`torch.cuda.max_memory_allocated()`, `reset_peak_memory_stats()` before each arm, one arm per
process-slot, box re-probed between runs.

| # | lever on resnet101 | peak alloc | reserved | s/iter | fits | verdict |
|---|---|---|---|---|---|---|
| — | *(none — the baseline)* | **OOM 22.37** | — | — | ✗ | the inherited claim, **re-verified** |
| **1** | **chunk ckpt 1** (leading-batch checkpointing) | **2.45** | 2.75 | 2.33 | ✓ | **-89 %. The only lever that fits alone.** |
| 2 | chunk ckpt 2 | 3.48 | 3.57 | 2.22 | ✓ | -84 % |
| 3 | chunk ckpt 4 | 5.43 | 5.61 | 2.40 | ✓ | -76 % |
| 4 | `trunk-mode inflate` + bf16 | 6.40 | 6.77 | 0.33 | ✓ | fits, but **K folded into the stem = a different arm** |
| 5 | timm `set_grad_checkpointing` **+ bf16** | 8.08 | 12.07 | 8.64 | ✗ | spills |
| 6 | `trunk-mode inflate` | 10.29 | 11.28 | 4.56 | ✗ | spills |
| 7 | bf16 autocast | 14.74 | 16.12 | 17.61 | ✗ | spills |
| 8 | freeze stem+layer1+layer2 | 14.84 | 15.32 | 20.83 | ✗ | spills |
| 9 | timm `set_grad_checkpointing` | 15.17 | 22.26 | 34.65 | ✗ | spills |
| 10 | freeze stem+layer1 | 19.58 | 20.18 | 25.06 | ✗ | spills |
| 11 | `channels_last` | **OOM 22.40** | — | — | ✗ | **worse than baseline, and 3.2x slower at WB=1** |

⛔ **timm's own grad-checkpointing is BROKEN on this backbone out of the box.**
`net.set_grad_checkpointing(True)` raises *"one of the variables needed for gradient computation has
been modified by an inplace operation ... ReluBackward0 ... is at version 1"* on **resnet34 and
resnet101 alike**. It only runs after 100 inplace activations are forced out-of-place — and then it
still does not fit (row 9/5), because `checkpoint_seq` checkpoints the four `layer*` stages while the
**stem** activation (`[24, 64, 208, 512]` = 653 MB) and every stage boundary are still retained at
the full 24-image batch. **Stage-level checkpointing is the wrong axis; the batch is the right one.**

⚠️ **bf16 and freezing look like levers and are not, at this shape.** Both cut memory by ~33 % and
~20 % — not enough to cross 22.4 GB → 7.1 GB, so both land in the spill region where their *own*
s/iter is 17-25 s. They are only worth anything **on top of** a lever that already fits (§5).

---

## 4. ⛔ The fitting configuration is not a correctness claim — what the parity probe found

**Chunking the leading batch changes BatchNorm.** BN's statistics are computed over the batch, and
the reference arm's "batch" is those 24 images; chunk 1 gives each image its own statistics. The
full-rig parity probe caught it immediately (`raw/r34_b1_ref.json` vs `raw/r34_b1_ck1.json`):

| | resnet34 unchunked | resnet34 chunk 1 | Δ |
|---|---|---|---|
| `ga_trunk` | 1,213,951.3 | 724,635.3 | **-40 %** |
| `loss` | 186.149 | 182.978 | -1.7 % |
| `ga_lift` | 6,790.14 | 7,152.95 | +5.3 % |
| other 5 live heads | — | — | within 1.9 % |

⇒ A run that reported "resnet101 fits" on that configuration would be reporting a **different arm**
under the arm's name. **The fix is to remove the batch coupling rather than hide it:** pin the
trunk's BN to its ImageNet running statistics (`frozen_bn`, 104 BN modules on resnet101, affine
weights still trained). Then the normalisation no longer depends on batch composition and chunking is
**the same function** — which is asserted, not assumed:

**THE EXACTNESS PAIR** (`raw/micro_wb8_frozenbn.json`, identical seeds, identical input):

| | trunk `sum(\|grad\|)` | conv1.weight grad |
|---|---|---|
| resnet34 frozen-BN, **unchunked** | 136.154845336152 | 16.516754150390625 |
| resnet34 frozen-BN, **chunk 1** | 136.155824791775 | 16.517349243164062 |
| **relative difference** | **7.2e-06** | **3.6e-05** |

and chunk-independence on resnet101: chunk 1 / 2 / 3 give `60.594886 / 60.594798 / 60.594970`
(spread **3e-06**). That is float-accumulation noise, not a behaviour change.

**In the FULL rig** (`r34_b1_fbn` vs `r34_b1_fbn_ck1`, step 3 — the 7 heads this window supervises; `ga_tac_decoder` is 0 on BOTH sides here, see §6):

| head | frozen-BN unchunked | frozen-BN chunk 1 | rel. diff |
|---|---|---|---|
| `ga_planner` | 100,391.868 | 100,392.015 | 1.5e-06 |
| `ga_bev_encoder` | 37,424.399 | 37,424.265 | 3.6e-06 |
| `ga_map_head` | 75.820296 | 75.818588 | 2.3e-05 |
| `ga_box_decoder` | 517,003.074 | 517,039.650 | 7.1e-05 |
| `ga_lift` | 8,419.543 | 8,421.657 | 2.5e-04 |
| `ga_box_memory` | 63,947.820 | 63,967.167 | 3.0e-04 |
| `ga_trunk` | 695,257.551 | 694,074.700 | 1.7e-03 |
| `loss` | 188.8302 | 188.7840 | 2.4e-04 |

**Nothing went dark, and the freeze survived contact with the trainer:**
`bn_training_on_first_forward = 0` is read back from the live model *inside the real run* (a plain
`.eval()` would have been silently undone by the trainer's `model.train()`).

⚠️ **Read the liveness claim at the right length.** Over 10 steps, **all 8 heads receive non-zero
gradient** in the fitting arm (`raw/r101_b1_fbn_ck1_s10/metrics.jsonl`). `ga_tac_decoder` and
`ga_map_head` are zero on *individual windows* — exactly the windows whose labels carry no supervised
tactical row (`tacv6_n_supervised_lat 0`) or no map GT (`map_n_labelled 0`) — and they are zero on
those windows in the **untouched reference** too. A 3-step probe can therefore make a live head read
dark; §6 is the finding, not the lever.

⛔ **State it exactly.** Frozen BN **is** a change against today's reference, which trains BN on the
batch. There is **no** configuration that fits *and* reproduces the unpatched arm's BN statistics —
those statistics require all 24 images' activations resident, which is precisely what does not fit.
The honest form of the answer is: *resnet101 fits at 416 x 1024 on 8 GB, and the price is that the
trunk's BatchNorm runs on ImageNet running statistics.* ⚠️ Worth the PI's eye: at batch 1 the
reference's BN "batch" is **24 highly-correlated frames of one clip**, so it was already a degenerate
estimate — frozen BN is the standard choice for a batch-1 detection backbone. That is an observation,
not a result; nothing here measures which trains better.

---

## 5. The full rig — every arm, with s/step in the same breath

`raw/summary.json`, one JSON per arm in `raw/`. `fits` = `peak_reserved <= 7.1 GB`.
`full arm` = median s/step x 40,284 steps.

| arm | levers | peak alloc | reserved | **s/step** | fits | **full arm** |
|---|---|---|---|---|---|---|
| `r101_b1_base` | none | **OOM 22.34** | — | — | ✗ | — |
| **`r101_b1_fbn_ck1_s10`** | **frozen-BN + chunk 1** (10 steps) | **2.887** | **3.73** | **4.9** | ✓ | **2.28 d** |
| `r101_b1_fbn_ck1` | frozen-BN + chunk 1 (3 steps) | 2.887 | 3.61 | 4.6 | ✓ | 2.14 d |
| `r101_b1_fbn_ck1_bf16` | + bf16 on the trunk | 2.724 | 2.89 | 4.35 | ✓ | 2.03 d |
| `r101_b1_fbn_ck3` | frozen-BN + chunk 3 | 4.907 | 5.17 | **3.9** | ✓ | **1.82 d** |
| `r101_b1_ck1` | chunk 1, **BN on the batch** | 2.888 | 3.61 | 4.6 | ✓ | 2.14 d — ⚠️ §4 |
| `r101_b2_fbn_ck1` | frozen-BN + chunk 1, **batch 2** | 3.890 | 4.82 | 8.55 | ✓ | 3.99 d |
| `r101_b4_fbn_ck1` | frozen-BN + chunk 1, **batch 4** | 6.227 | **7.82** | 32.75 | ✗ | — |
| `r34_b1_ref` | none (**today's rig**) | 7.708 | **9.35** | **10.75** | ✗ | 5.01 d |
| `r34_b1_fbn_ck1` | frozen-BN + chunk 1 | 1.660 | 1.80 | **1.65** | ✓ | **0.77 d** |
| `r34_b2_ref` | none (**the 29.2 s/step of record**) | 14.629 | **18.27** | **29.1** | ✗ | 13.57 d |
| `r34_b2_fbn_ck1` | frozen-BN + chunk 1, batch 2 | 2.512 | 2.75 | **2.8** | ✓ | **1.31 d** |

⭐ **The lever is worth more to the arm we are already running than to the one we are pricing:**
resnet34 at batch 2 goes **29.1 → 2.8 s/step** (13.57 d → 1.31 d) with no change to the trunk, only
to how its activations are held. That is a bigger number than anything about resnet101 in this note.

All 22 `ga_*`/`gp_*` reach keys present on every OK arm; peak read from
`torch.cuda.max_memory_allocated()` in-process, `reset_peak_memory_stats()` immediately before
`train()`.

### The recommendation, and its price stated plainly

* **`frozen_bn + chunk_ckpt 1` is the pick.** 2.89 GB / 3.73 GB reserved leaves **~3.4 GB of
  headroom** on a card the other tenants are already using 1.0 GB of — the configuration survives a
  co-tenant, which `ck3` (5.17 GB) would not.
* **`chunk_ckpt 3` is the speed pick** — 3.9 s/step, **1.82 d**, chunk 3 = exactly one window
  position — but only 1.9 GB of headroom.
* **bf16 on the trunk buys 0.163 GB and 0.25 s/step** on top (2.887 → 2.724 GB, 4.6 → 4.35 s/step —
  both the 3-step arms, compared like with like), and this is the *only* place bf16 pays at all (§3).
  It is a second precision change stacked on the BN change; take it only if the schedule needs it.

### Effective batch: accumulation is priced, and it is not the cheapest route

`refc_v3_train.py` has **no accumulation flag**, so this is the one number that is not MEASURED:
accumulating N micro-batches leaves peak at the batch-1 figure (**2.887 GB**) and costs
**N x 4.9 s/step** — ESTIMATED, because nothing implements it. The estimate is checkable against the
arm next to it: **real batch 2 fits** (3.890 GB, **8.55 s/step**, 3.99 d), which is *cheaper* than
the 9.8 s/step that 2-step accumulation would cost. ⇒ **Up to batch 2, raise `--batch`; accumulation
only earns its keep beyond it.** **Batch 4 does not fit** (7.82 GB reserved, 32.75 s/step — the
signature of the spill).

---

## ⚠️ 6. OBSERVATION — tactical and map supervision is SPARSE PER WINDOW, and a short probe misreads it

⛔ **I nearly filed this as a dead head, and it is not one.** At `--steps 3` the tactical decoder
(`2,253,828` parameters) reads `ga_tac_decoder = 0.0 EXACTLY` in **every arm, including the untouched
reference** — the `tac_goal_tok_head` signature (11,286 params at `grad_abs_sum` 0.0 for all 40,284
steps of refcv5-v2). Run it to 10 steps and the head is **alive**:

| | steps with `tacv6_n_supervised_lat > 0` | `ga_tac_decoder` on those steps |
|---|---|---|
| `r34_b1_ref_s10` | **1 of 10** | 549.7 |
| `r101_b1_fbn_ck1_s10` | **4 of 10** (steps 1, 4, 7, 10) | 280.1 / 296.7 / 296.5 / 236.6 |

Same for the map head: `ga_map_head = 0.0` on exactly the one window of ten where
`map_n_labelled = 0` (`r34_b1_ref_s10`, step 10), and non-zero on the other nine.

⇒ Both heads are **reachable and reached**. What the rows show is **label coverage**: on the
`eval139` split-A cache at batch 1, only a minority of windows carry a supervised tactical row, and a
minority carry no SAM3 map GT at all. That is a **data** property, worth an owner's eye (is ~10-30 %
tactical coverage the intent?), and it is neither a defect of this note's levers nor fixed by them.

⭐ **The methodological finding is the useful one: a 3-step liveness probe can report a live head as
dark.** The grad-reach instrument is per-step, and per-step is per-*window*; `tac_goal_tok_head`'s
own evidence was 40,284 steps, not three. Any future "is this head live" check on this rig needs
enough steps to draw a supervised window — **10 was not obviously enough either** (1 of 10 on one
draw). Stated here so the next reader does not inherit my first reading.

---

## 7. What this does NOT claim

* **Not a capability claim.** Every arm ran 3 or 10 steps from an ImageNet-init trunk. No number here
  is a metric-family reading, a tier stamp, or evidence about accuracy. Whether resnet101 is *better*
  than resnet34 is untouched.
* **Not a claim that frozen BN trains as well.** §4 measures that it is the *same function under
  chunking*, not that it is the right arm.
* **Not a pod claim.** Everything is this 8 GB card with ~1.0 GB of co-tenants. A pod has neither the
  WDDM spill nor the ceiling.
* **The microbench arms build with `pretrained=False`** (weight *values* change neither activation
  bytes nor op count); **every full-rig arm uses `--trunk-pretrained`**.
* **`s/step` is measured with `--workers 0`**, as `cover139d.sh` runs it, so it carries the main-
  process PNG decode. A worker-fed run would move the floor; that is a separate measurement.

---

## 8. Deliverable manifest

| path | what |
|---|---|
| `RESULT.md` | this note |
| `raw/summary.json` | all 13 full-rig arms: flags, peak alloc/reserved, s/step, per-head grad norms, fit verdict, 40,284-step projection |
| `raw/trunk_microbench.py` | trunk-only lever sweep (ranking + the frozen-BN exactness pair) |
| `raw/rig_arm.py` | one full-rig arm with levers applied as monkeypatches; reads back `bn_training_on_first_forward` live |
| `raw/run_arms.sh` | driver; probes `qland/boxstat.py` before every arm |
| `raw/micro_wb1.json` | trunk-only at WB=1 — the shape the rig does **not** run, kept because it is what shows the WB=8 discovery mattered |
| `raw/micro_wb8.json` | trunk-only at the rig's true shape — the lever ranking of §3 |
| `raw/micro_wb8_stagegc.json` | timm `set_grad_checkpointing` after the inplace fix |
| `raw/micro_wb8_frozenbn.json` | the exactness pair + chunk-independence on resnet101 |
| `raw/<arm>.json` (15) | one per full-rig arm: flags, peak, per-step deltas, per-head reach, `bn_training_on_first_forward`, `trunk_input_shape` |
| `raw/<arm>/metrics.jsonl` + `config.json` | the trainer's own per-step rows and full run stamp |
| `/c/Users/Admin/tanitad-caches/r101fit-20260918/<arm>.log` | verbatim trainer output (off-repo, not staged) |

⚠️ The trainer writes a **final `ckpt.pt` regardless of `--save-every`** — 1.1 GB per arm, 14 arms, **12 GB** landed inside this research folder. They are 1-to-10-step checkpoints of an ImageNet-init model and carry no evidence, so they were **deleted before staging** (`*.pt` is NOT globally gitignored in this repo — only `stack/experiments/**` and `_pod_backup/**`). Anyone re-running `raw/run_arms.sh` inherits the same 12 GB.

**Integration escalated, not written into a doc:** if the PI takes this, `--trunk-chunk-ckpt N` and
`--trunk-frozen-bn` need to become **real flags on `refc_v3_train.py`, stamped into `config.json`** —
a monkeypatch is admissible for a measurement and inadmissible for an arm of record. That is a
trainer change and is **not** made here.
