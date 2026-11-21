<title>Adversarial verification of the seven REFe fixes — what holds, what does not</title>

# FIX VERIFICATION REVIEW — the seven fixes, attacked

**Reviewer:** independent verification agent · **Date:** 2026-09-20
**Method:** every verdict below is read from SOURCE and, wherever it could be, MEASURED by running
the code — against the released DriveZero/DriveRL source, the released DINOv3 source, the released
NAVSIM source, the real DINOv3 ViT-S checkpoint, and a real banked simulation log. The package's own
prose was never used as evidence that the code does something, and **a fix passing its own test was
not accepted as evidence** — every arm that mattered was re-derived with an independent instrument
and, where possible, a mutation arm that had to go red.

---

## ⛔ READ STAMP — THE PACKAGE WAS EDITED FIVE TIMES DURING THIS REVIEW

`refe/score_proposals.py` changed at **20:07**, **20:20** and **20:33** while I was measuring it;
`refe/train.py` and `refe/build_scorer_targets.py` at **20:05**; `refe/diag_scorer_components.py` at
**20:08**, **20:21** and **20:34**. Every statement below is pinned to these md5s:

| file | md5 (first 32) | mtime |
|---|---|---|
| `refe/model.py` | `9d0e045fba060f771fd23fe5b18b1bd2` | 19:51:38 |
| `refe/train.py` | `35f66a969a9a08b6b93cae7e3cf3275d` | 20:05:02 |
| `refe/planner.py` | `07147080be2d1185b51029493cd47af6` | 19:55:18 |
| `refe/build_scorer_targets.py` | `75bf64bb728de700946d4896fd8eefed` | 20:05:00 |
| `refe/load_dinov3.py` | `44abd8c84e55e5d6612f46f3ecf56df3` | 19:49:26 |
| **`refe/score_proposals.py`** | **`a3a762b0c5edd21fcad4cc1a12a83f85`** | **20:33:28** |
| `refe/validate_model.py` | `bbf0f2f5ad8529f93dfa53a9b8cf3f07` | 14:58:04 |

⛔ **AND IT MOVED ONCE MORE AFTER I FINISHED.** At **20:41:46** `refe/score_proposals.py` read
`2c4fcecc6144596367ecebf85dfd8625` — a **sixth** edit, after every F3/F4 measurement below was taken.
The other six files were byte-identical to the table at that re-check. **Every F3 and F4 row is a
statement about `a3a762b0`, not about whatever is on disk when you read this**, and the two findings
most likely to have been addressed in that edit are the `min`-aggregation of `progress.advance_m` /
`ddc.violation` and the k=1 comfort boundary. Re-run `f3_probe.py` before quoting the comfort spread.

⚠️ Where a number was taken against an earlier md5 it says so. The F3 comfort result was re-measured
against the final 20:33 version and is reported at that stamp.

⚠️ **AND THE SCORER BANK WAS DELETED MID-REVIEW.** At **20:08:06** `D:/Projects/TanitAD/data/
refe_scorer_targets_full/` held 4 shards, **7,860 rows over 874 frames**. At **20:13:23** the
directory was **empty**, and it is still empty at 20:34. Every bank measurement below carries the
20:08:06 stamp and describes the bank that existed then. **A training run started right now would
print `scorer bank: NONE` and train the scorer against the zero placeholder.**

---

## 1 · Headline verdict

**Two fixes are clean, three are partial, one is wrong in effect, and one — the one I was asked to
attack hardest — is WRONG and is measurably a REGRESSION against the defect it replaced.**

| | fixes |
|---|---|
| **CONFIRMED** | **F2** (registers compress — verified by my own hooks, not theirs) · **F6** (WTA metric) |
| **PARTIAL** | **F1** (the join is closed, but the live bank has no `rank` and coverage halves silently) · **F4** (the tuple and the category fix are right; **PROGRESS is capped at ~4 % of real progress and aggregated as a minimum, and DDC is still structurally zero**) · **F5** (structure right; the rule implemented is NAVSIM v1 PDMS × DDC — neither v1 nor v2) |
| **WRONG in effect** | **F3** — the `dt` half is CORRECT and I confirmed it against an independent reference, but **comfort still cannot rank: spread across ten candidates is exactly 0.000000 at the final stamp**, and the guard added to pin `dt` is algebraically blind to `dt` |
| **WRONG** | **F7** — REFe's rotary encoding is **not** DINOv3's (3 of 6 elements differ), and MEASURED on the real checkpoint it moves the frozen trunk **farther** from its pretrained function than having **no** positional encoding at all (rel L2 **0.760** vs **0.585**; cos **0.688** vs **0.834**) |

### ⛔ ESCALATION — three things need a decision, not a merge

1. ⛔ **F7 must be re-done before any REFe number is produced.** It is not "self-consistent but
   different"; on the only metric available it is worse than the defect it replaced. The fix is
   mechanical (the released formulation is 8 lines) and I have written out exactly what differs.
2. ⛔ **The trainable parameter budget nearly DOUBLED and nobody has said so.** `RegisterCompress`
   adds **12,598,272** trainable parameters: **13,986,370 → 26,584,642 (4.384 % → 8.064 %)**, total
   **319,031,874 → 329,664,066**. REFe's trainable count is now **43 % ABOVE the paper's 18.58 M**
   — on ONE camera against their four. Three documents still carry 316.9 M / 11.88 M. This is the
   figure the sub-300M-thesis statement was signed off against.
3. ⚠️ **F5 selects with a rule that is neither navtest PDMS nor navhard EPDMS.** If a REFe number is
   ever placed beside a published one, the selection rule and the scoring rule are different
   functions. That belongs in the arm's stamp, not in a code comment.

---

## 2 · Fix-by-fix table

| fix | claim | what I MEASURED | file:line | verdict |
|---|---|---|---|---|
| **F1** | `--rank` written, `ScorerBank` keys on `(log, token, step, rank)` | argparse `--rank` and `"rank": int(a.rank)` in the row both present; 4-tuple key present; `get()` and the `TargetBank` lookup both carry rank. **The cross-rank join IS closed.** ⛔ But the live bank at 20:08:06 carried **0 of 7,860 rows with a `rank` field**, so every row defaults to rank 0, while the trajectory bank carries rank on all 2,182 rows (1,091 + 1,091). ⇒ every rank-1 tuple now MISSES and coverage halves in silence. Nothing refuses; the "no key served to two ranks" guard is not implemented | `build_scorer_targets.py:165`, `:197-198`, `:281`; `train.py:169`, `:235-236`, `:296-297` | **PARTIAL** |
| **F1b** | the no-`rank` fallback is "honest" | ⛔ The fallback leans on the builder's stats file (`train.py:166-168`, *"asserted in the builder's stats file"*), but **no code reads that file**, and `scorer_targets_stats.json` is **not rank-suffixed** (`:309`) while the data file **is** (`:197`) — so a rank-1 build into the same `--out` silently overwrites the rank-0 provenance. The assertion the fallback rests on is both unread and destructible | `build_scorer_targets.py:197` vs `:309`; `train.py:166-169` | **PARTIAL** |
| **F2** | registers compress; traj decoder gets scene tokens, scorer gets visual tokens | MEASURED with **my own** forward hooks (not `diag_architecture.py`): trajectory ctx `(1, 16, 256)`, scoring ctx `(1, 32, 256)` at a 32-patch geometry ⇒ scene = **16 = n_cameras × n_registers**, visual = P, **different tensors**, and **no decoder ever sees `visual + 16`**. `scene_proj` shared is not a correctness bug — `visual_ctx` is detached *after* the projection, so the score loss cannot shape it, which is the intended asymmetry. Scorer context correctly detached (`:385`), candidates detached (`:383`) | `model.py:304-322`, `:371-386` | **CONFIRMED** |
| **F2b** | — (side effect) | ⛔ trainable **13,986,370 → 26,584,642** (+12,598,272, ×1.90); total **319,031,874 → 329,664,066** (+10,632,192 = +12.6 M `reg_compress` − 1.97 M removed `pos`); trainable share **4.384 % → 8.064 %**. LoRA property HOLDS: 96 trainable backbone tensors, 3,145,728 params, **0** outside `attn.q`/`attn.v`. `reg_compress` is trainable (correct — it is not part of the frozen trunk) | measured by instantiating `model.py` | **NEW DEFECT (doc)** |
| **F3a** | `dt` is 0.2 s, not the engine's 0.1 | **CONFIRMED against an INDEPENDENT reference.** The simulation log's own spacing between banked samples is **0.1999 s**. Derived mean ego speed at `TRAJ_DT=0.2` = **6.197 m/s** against the ego's LOGGED mean of **6.291 m/s** (1.50 % error); at 0.1 = **12.393 m/s** (97.0 % error — reproducing the reported 12.393 exactly). Corroborated a second way: derived `a_long` 1.787 m/s² at 0.2 against a logged 2.816 → 10.004 m/s change over 4 s ≈ 1.8 m/s² | `score_proposals.py:57`; log spacing measured | **CONFIRMED** |
| **F3b** | an identity check now catches a wrong `dt` | ⛔ **THE GUARD IS ALGEBRAICALLY BLIND TO `dt`.** It compares `mean\|Δp\|/TRAJ_DT` with `chord/(T·TRAJ_DT)` — **`TRAJ_DT` is on both sides and cancels**; what remains measures path straightness. **MUTATION ARM MEASURED: the check PASSES at `TRAJ_DT=0.2` AND at `TRAJ_DT=0.1`.** A check that shares the defect it checks for | `diag_scorer_components.py` (velocity-identity arm) | **WRONG** |
| **F3c** | comfort can now rank proposals | ⛔ **NO. MEASURED at the 20:33 stamp over all 10 candidates: `comfort.Comfort.reward` = 0.5000 for every one, spread exactly 0.000000.** Root cause measured: `score_proposal_rollout` aggregates `.reward` with **min** over prefixes k=1,3,…,20, and at **k=1 every candidate reads 0.5** — a boundary artefact of the derivative, not a property of the candidate. Per-prefix values do differ (k=20: teacher **0.5**, stopped **1.0**, jerky **0.5**) — and that ordering is **inverted**: the expert's own realised path reads UNCOMFORTABLE while a car that stops dead reads COMFORTABLE | `score_proposals.py:458-460` (prefix grid), `:365-366`-equivalent (min/max rule) | **WRONG in effect** |
| **F3d** | the `jerky` candidate is a fair comfort violation | ⚠️ It is a violation the calculator **cannot see**. `Comfort.data_preprocessing` sets `a_lat = torch.zeros_like(speed)` — lateral acceleration is **hard zero**, so `a_lat_ok` can never fire; and `_derive_ego_kinematics` computes `a_lat` and **never writes it**. A lateral sawtooth under the teacher's (smooth) yaw lands almost entirely in `a_lat`. It does show up in `comfort.Comfort.info` (**749.68** vs 146.02) — the leaf the trainer does **not** consume | `DriveRL/…/comfort.py:397`; `score_proposals.py` `_derive_ego_kinematics` | **PARTIAL** |
| **F4a** | `COMPONENTS` is the paper's six; `cat == 1` fixes the clipping | Both true and correct. Note the consequence: categories 2/3/4 now read as **no** DAC violation, which is right (2 is a lane-marking event, 3/4 are DDC) but is a behaviour change worth stating | `train.py:127`, `:195` | **CONFIRMED** |
| **F4b** | PROGRESS from `calculate_baseline_progress`, normalised to the teacher, clipped to [0,1] | The call is legal — it **is** a `@staticmethod`, `baselines` must be 4-D `[N,L,P,2]` and `baseline_mask` 2-D `[N,L]`, and `lc[:, b:b+1, :, :2]` / `lm[:, b:b+1]` satisfy both (MEASURED shapes `(1,1024,2,4)` and `(1,1024)`). ⛔ **But each "baseline" is a single 2-POINT SEGMENT.** The chosen line's total arc is **9.75 m**; arc at t0 is **8.682 m**, so advance **saturates at 1.068 m** while the ego travels **24.78 m**. Progress measures ~**4 %** of real progress and every candidate that runs off the 9.75 m segment ties at the cap | `score_proposals.py:237-277`, `:297-300`; `DriveRL/…/center_line.py:662` | **PARTIAL / unsound** |
| **F4c** | — | ⛔ **`progress.advance_m` is aggregated with `min`** (it does not end in `.info`), so the banked value is the **k=1 prefix**: teacher per-prefix **0.58 / 1.07 / 1.07 / 1.07**, banked **0.58**. And because `linspace(0,d,T)[0] = 0`, **every lateral candidate reads exactly the teacher's 0.577** — EP is blind to the entire lateral family | measured per prefix | **NEW DEFECT** |
| **F4d** | a near-zero or negative teacher advance is handled | The builder guards `t_adv > 1e-3` and emits `nan` — but `ScorerBank.components` maps `nan` to **1.0**, the **maximum**. A frame the builder refused to score therefore teaches the scorer that progress was **perfect**. `0.0`, or a mask, would be honest | `build_scorer_targets.py:240-243`; `train.py:196` | **NEW DEFECT** |
| **F4e** | DDC read from `OffRoad.info ∈ {3,4}` | ⛔ **STILL STRUCTURALLY ZERO — do not credit this.** MEASURED over 10 candidates × 5 prefixes on a real frame: `off_road.OffRoad.info` takes only **{0, 1}** (only `over-curb` reaches 1). `ddc.violation` is **identically 0.0**, so the component is a constant 1.0. ⛔ And it is aggregated with **`min`**, i.e. *"did it violate at EVERY step"* — the opposite of an event; `.info` is on the max list and `.violation` is not | `score_proposals.py:318-320`; aggregation rule in `score_proposal_rollout` | **PARTIAL (leaf emitted, supervision still zero)** |
| **F4f** | — (legacy bank) | ⛔ MEASURED at 20:08:06: **0 of 7,860 rows carried `progress.ep`, `ddc.violation` or `progress.advance_m`.** On such a bank both components default to their **MAXIMUM** (`ep` nan → 1.0; `ddc` missing → 1−0 = 1.0) — two of six become constant 1.0. `variance_report` *would* name them "CANNOT RANK" at load, so it is visible; nothing refuses | `train.py:192`, `:196`, `:199` | **NEW DEFECT (mitigated by the guard)** |
| **F5** | `sigmoid`, `NC·DAC·DDC·(5EP+5TTC+2C)/12`, argmax; `logitsum` kept as a control | Implemented exactly as claimed. Against the primary source (`autonomousvision/navsim` `pdm_scorer.py`): the multiplicative group is **NC · DAC · DDC · TLC (FOUR)** and the weighted group is **EP(5), TTC(5), LANE_KEEPING(2), HISTORY_COMFORT(2)** over the active-weight sum (**14**). ⇒ **TTC is correctly a weighted term, not a multiplier**; **DDC is correctly a multiplier** (the programme's "four multipliers" note is confirmed and ours has three of them); ⚠️ **TLC is absent** and the denominator is **12, not 14**, with one "comfort" standing in for LK + HC. What is implemented is exactly **NAVSIM v1 PDMS × DDC** — neither v1 nor v2 | `planner.py:188`, `:190-208`, `:210-217` | **PARTIAL** |
| **F6** | position-only L1 for selection, yaw as a separate wrapped weighted term | Correct, and the counterexample flips the right way (new picks #0, old picked #1) — and that control genuinely re-runs the old metric, so it can go red. **No stale callers**: `train.py:397`, `validate_model.py:99`, `diag_architecture.py:90` all use the new signature | `model.py:389-418` | **CONFIRMED** |
| **F6b** | `yaw_w = 0.1` | ⚠️ **Justified nowhere.** Grep over `refe/*.py` and the package's `.md` files returns exactly **two** hits — the definition and the use. Not a CLI flag, not in any document, no ablation. ⚠️ And the loss **SCALE** moved: MEASURED new/old = **2.018×**, so `--score-w 0.1` now weights the scorer half as heavily as before and no loss number is comparable across the change | `model.py:389`, `:418` | **UNMOTIVATED CONSTANT** |
| **F7** | `build_axial_rope` + `apply_rope` implement DINOv3's axial RoPE; `self.pos` deleted; `EXPECTED_UNLOADED` empty | `self.pos` is gone and the loader reports 0 unconsumed / 0 unexplained — **both true and both irrelevant to the question**, because DINOv3 stores no rope tensor, so no checkpoint-consumption check can ever test the formula. **The formula is wrong** — see §3 | `model.py:126-168`, `:245-251`, `:265-285`; `load_dinov3.py:39-48` | **WRONG** |

---

## 3 · F7 in full — the fix I was asked to attack, and it does not survive

### 3.1 · What DINOv3 actually does (two independent primary sources, in agreement)

`facebookresearch/dinov3` → `dinov3/layers/rope_position_encoding.py` and `dinov3/layers/attention.py`;
`huggingface/transformers` → `src/transformers/models/dinov3_vit/modeling_dinov3_vit.py`.

```
coords_h = arange(0.5, H)/H ;  coords_w = arange(0.5, W)/W      # PATCH CENTRES, normalised
coords   = 2.0 * coords - 1.0                                   # -> [-1, +1]
periods  = base ** (2*arange(D//4) / (D//2))                    # base = 100.0
angles   = 2*pi * coords[:, :, None] / periods                  # [N, 2, D//4]   <- 2*pi
angles   = angles.flatten(1, 2).tile(2)                         # [N, D]  HALF-SPLIT layout
rotate_half(x): x1, x2 = x.chunk(2, -1); cat([-x2, x1], -1)     # pairs i with i + D/2
apply: (x * cos) + (rotate_half(x) * sin), prefix left unrotated
```

### 3.2 · Element-by-element against `refe/model.py:126-168`

| element | DINOv3 | REFe | |
|---|---|---|---|
| base | `100.0` | `100.0` | ✅ |
| exponent ladder | `base ** (2·arange(D/4)/(D/2))` | `arange(0, half, 2)/half`, `half = dh/2` — the same ladder | ✅ |
| prefix excluded | `prefix = N − sin.shape[-2]` | `off = 1 + N_DINOV3_REG = 5` | ✅ |
| coordinates | patch centres, normalised, **mapped to [−1, +1]** | **raw integer patch indices** `arange(gh)`, `arange(gw)` | ❌ |
| scale | **`2π ·` coord / period** | no `2π`, no normalisation | ❌ |
| axis→channel layout | `cat([θy, θx]).tile(2)` — channel *i* pairs with *i + D/2* | `repeat_interleave(…, 2)` — channel *2j* pairs with *2j+1* | ❌ |
| rotation | `chunk(2, -1) → cat([-x2, x1])` | `0::2` / `1::2` interleaved | ❌ |

MEASURED angular spans at an 8×16 grid, head dim 64:

```
reference  y in [-5.498, +5.498]   x in [-5.890, +5.890]   spans 10.996 / 11.781   (equal by design)
REFe       y in [+0.000, +7.000]   x in [+0.000, +15.000]  spans  7.000 / 15.000   (grid-dependent,
                                                                                    and uncentred)
```

Note the second-order consequence: DINOv3's `"separate"` normalisation makes the two axes span the
**same** angle regardless of aspect ratio. REFe's span is proportional to the grid size, so at the
production 32×60 geometry the two axes are stretched by different factors (≈2.5× and ≈4.7×) relative
to the reference — the *relative* geometry the trunk learned is distorted, not merely rescaled.

### 3.3 · The decisive measurement, on the real checkpoint

Loaded the real `dinov3-vits16` weights into the trunk (**0 unconsumed, 0 missing** — refused to
measure on a partial load), ran the same image through the same frozen trunk with four encodings, and
compared the patch tokens to the reference encoding:

| arm | rel L2 vs reference | cosine |
|---|---|---|
| **SELF** — reference vs reference *(control: must be 0)* | **0.00000** | 1.00000 |
| **JITTER** — coords shifted 1e-3 of a patch *(control: must be tiny)* | **0.00005** | 1.00000 |
| **NONE** — no positional encoding at all | **0.58521** | 0.83353 |
| **OURS** — `build_axial_rope` + `apply_rope` | **0.75960** | **0.68802** |

and on block-0 attention logits, where the encoding enters: **OURS 0.27745 / cos 0.97711** against
**NONE 0.15045 / cos 0.99368**.

⛔ **REFe's rotation moves the frozen trunk FARTHER from its pretrained function than omitting the
encoding entirely.** The fix replaced a random additive table with a wrong rotation; on the only
metric available it is a regression, not a repair. The two controls establish that the probe is both
exact and sensitive, so this is not a probe artefact.

⚠️ **Honest limit of that claim.** "Closer in L2 to the pretrained function" is not the same as
"better features for driving", and a ViT with no positional encoding is known to be badly degraded.
What is established is that **the trunk is not computing the function it was pretrained to compute**,
and that the comment at `model.py:127` — *"as the checkpoint expects"* — is false.

### 3.4 · Why no existing check could catch it

* `load_dinov3.py` now reports 0 unconsumed and 0 unexplained tensors. **That is true and cannot
  test this**: DINOv3 stores no rope tensor, so tensor-consumption accounting is structurally silent
  about the *formula*. The empty `EXPECTED_UNLOADED` is a real improvement to a different question.
* `diag_architecture.py`'s D7 arm asserts (a) `backbone.pos` is gone, (b) the rotation changes the
  vectors, (c) it preserves the norm. **All three pass for ANY rotary encoding**, including a random
  one. The header calls it *"is the pretrained position encoding back?"* and it does not test
  "pretrained" at all — a check whose question is narrower than the claim hung on it.

### 3.5 · Minimal fix (described, NOT applied)

Replace the body of `build_axial_rope` with the released formulation above (centred normalised
coordinates, the `2π` factor, `tile(2)`), and replace `apply_rope`'s `0::2/1::2` with
`chunk(2, -1) → cat([-x2, x1], -1)`. Nothing else changes; the ladder and base are already right.

**What would settle it beyond my probe:** `pip install timm --no-deps`, run
`timm.create_model("vit_large_patch16_dinov3", pretrained=...)` on the same tensor and require a
numerical match of the patch tokens. **timm is ABSENT from the venv** (verified), and installing into
that venv is the operation that previously replaced torch with a wheel the driver could not run — so
this needs a deliberate, isolated environment, not an ad-hoc install.

---

## 4 · New defects the fixes introduced, ranked

1. ⛔ **`validate_model.py:115-123` still builds the PRE-F2 architecture by hand.** Its
   deliberate-regression arm does `tok = cat([backbone(img)+pos3d, registers])`, `ctx =
   scene_proj(tok)`, and feeds that one tensor to **both** decoders — the exact defect F2 removed.
   The shapes still work, so it runs green while testing a path that no longer exists, and it can no
   longer detect a re-attached scorer in the **shipped** forward. This is the file whose job is to
   catch exactly this. *Fix: build the arm from `model.forward`'s real tensors — `reg_compress`
   output for `dec`, undetached `visual_ctx` for `score_dec`.*
2. ⛔ **The trainable budget nearly doubled and three documents are stale** (§1, escalation 2).
3. ⛔ **`progress.advance_m` and `ddc.violation` are aggregated with `min`** — progress becomes the
   0.2 s prefix value and DDC becomes *"violated at every step"*. *Fix: add `.violation` to the max
   list, and take progress from the FULL horizon (`@last`), which the builder currently discards.*
4. ⛔ **PROGRESS is capped by a 2-point baseline** (§F4b) — 1.068 m measured against 24.78 m
   travelled. *Fix: accumulate arc across the route's connected lane groups (`lanes_centers_ids` /
   `_next_groups` are already enriched at `score_proposals.py:98-100`) rather than one segment.*
5. ⚠️ **`nan` progress maps to 1.0**, the maximum (§F4d).
6. ⚠️ **`score_proposals.py:58` still holds the OLD `COMPONENTS` tuple** — `("off_road",
   "collision", "ttc", "comfort", "goal_reaching", "center_line")` — and `selftest()` prints it at
   `:406` as *"the order the scorer target must use"*. It is not; `train.py:127` is. The same
   function still asserts at `:408-409` that the scorer is *"NOT DONE YET ... wiring these targets
   into train.py"* and that *"proposal selection at inference stays arbitrary"* — **both now false**
   (`train.py:296-300` consumes the bank; `planner.py:217` uses the PDM rule).
7. ⚠️ **`scorer_targets_stats.json` is not rank-suffixed** while the data file is, so a rank-1 build
   into the same `--out` destroys the rank-0 provenance that `train.py:166-168` cites as its
   justification for defaulting a missing `rank` to 0 — and **no code reads that file anyway**.
8. ⚠️ **`agent_steering_state_all` is the one kinematic tensor `_derive_ego_kinematics` does not
   extend.** Harmless for the verdict (the gates use `yaw_rate`/`yaw_accel`, not steering), but the
   `steering_rate` / `steering_accel` values Comfort returns are stale — worth a line so the next
   reader does not re-derive it.
9. ⚠️ **Comfort re-differentiates our 0.2 s-derived values at 0.1 s.** `Comfort.data_preprocessing`
   passes `delta=self.time_interval` (= `frame_time_interval` = **0.1**, MEASURED) to its Savitzky-
   Golay jerk filter and to `yaw_accel = Δyaw_rate / time_interval`. We write first derivatives on a
   0.2 s grid into a tensor the consumer differentiates on a 0.1 s grid, so the jerk and yaw-accel it
   gates on are **2× too large**. The cadence mismatch was pushed one derivative out, not removed.
   *Fix: resample the candidate to the engine's 0.1 s grid before injection, so one time axis serves
   everything — that is the only version of this that is self-consistent.*
10. ⚠️ **`diag_scorer_components.py` was BROKEN at 20:20** (`NameError: name 'io' is not defined` at
    line 82 — the author's own D3/D4 proof did not run at all). Repaired by 20:34; recorded because
    a package-conformance verdict was resting on it in between.
11. ⚠️ **`n_cameras > 1` cannot be instantiated** — `REFe.forward` raises on the `pos3d` broadcast
    (`32 vs 64`). Pre-existing, and the PI decision is one camera, so the claim *"scene tokens =
    `n_cameras × n_registers`"* is verified **only at `n_cameras = 1`**.

---

## 5 · What I ran, and what each run's control could actually have caught

| instrument | result | could its control go red? |
|---|---|---|
| `refe/diag_architecture.py` | `ARCHITECTURE_CONFORMS`, 9/9 | D2 and D6 arms: **yes** (they re-run the old metric / print real context shapes). D7 arm: **NO** — its three assertions pass for any rotary encoding |
| `refe/diag_schedule.py` | `SCHEDULE_OK`, 5/5; LR 2.000e-04 → **0.000e+00** at step 62, halfway 1.000e-04 | **Yes** — the `--no-cosine` arm holds flat at 2e-4, so the flag is shown to do something. This closes the prior review's one UNVERIFIED caveat: the LR **does** reach exactly zero |
| `refe/diag_planner_holds.py` | `PLANNER_HOLDS_GUARDED`, 4/4; hint resolved from `scenario.log_name`, refusal after 20 holds | **Yes** — arm 1 has no scenario and must raise; arm 3 proves one hold does not |
| `refe/load_dinov3.py` | *(not re-run standalone; exercised inside my probe)* 318/318 checkpoint tensors consumed, 0 unexplained, `EXPECTED_UNLOADED` empty | **No, for the question that matters** — see §3.4 |
| `refe/diag_scorer_components.py` | **did not run at 20:20** (`NameError`); repaired at 20:34 | its D3 arm asserts comfort varies — which, measured independently, it does **not** |
| **my `rope_probe.py`** | §3.3 | SELF = 0.00000 and JITTER = 0.00005 establish exactness and sensitivity |
| **my `f3_probe.py`** | §F3a/F3c | mutation arm at `TRAJ_DT=0.1` shows the package identity stays green while the independent reference goes to 97 % error |
| **my `f3_gates.py`** | gate-by-gate: at 0.2 all six pass; at 0.1 `a_long` 7.07 > 2.4, `max_jerk_long` 8.48 > 4.13, `max_jerk_mag` 9.54 > 8.37 all fail | the un-injected logged state is the third arm and reads all-pass |
| **my `f3_prefix.py`** | comfort per prefix; `off_road` categories per candidate | `stopped` vs `teacher` is the discriminating pair and it reads **backwards** |
| **my `param_probe.py`** | §F2b; scene-token count at `n_cameras` 1/2/4 | the `n_cameras=2,4` arms fail loudly, which is itself the finding |

---

## 6 · Deliverable manifest

| artifact | where it lives | only copy? |
|---|---|---|
| **This review** | `repo:TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/FIX_VERIFICATION_REVIEW.md` — staged | no (staged in the index) |
| DINOv3 RoPE probe (the F7 evidence) | `scratchpad:rope_probe.py` + `rope_probe` stdout | **YES — scratchpad only** |
| F3/F4 probe with the independent speed reference and the `TRAJ_DT` mutation arm | `scratchpad:f3_probe.py`, `f3_probe.out`, `f3_probe2.out` | **YES — scratchpad only** |
| comfort gate-by-gate probe | `scratchpad:f3_gates.py`, `f3_gates.out` | **YES — scratchpad only** |
| comfort-per-prefix + DDC category probe | `scratchpad:f3_prefix.py`, `f3_prefix.out`, `f3_prefix2.out` | **YES — scratchpad only** |
| parameter / LoRA / scene-token / WTA-scale probe | `scratchpad:param_probe.py`, `param_probe.out` | **YES — scratchpad only** |
| broken-diagnostic capture | `scratchpad:diag_sc.txt` | **YES — scratchpad only** |

⚠️ **The five probe scripts are deliberately not staged.** Each measures a package under active edit
(five source changes during this review) and banking them would freeze a snapshot of a moving target.
**Every number they produced is reproduced inline above with its file:line and its read stamp.**
⭐ If the PI wants them kept, `rope_probe.py` is the one that should be banked — it is the only
instrument in the programme that can tell a correct rotary encoding from a wrong one, and its right
home is `refe/diag_rope.py` beside the other diagnostics, with the reference formulation and both
primary-source URLs in its docstring.

**No code was changed by this review.** The edits observed at 20:05/20:07/20:08/20:20/20:21/20:33/20:34
were made by the package author, not by me. Every minimal fix above is described and not applied.
