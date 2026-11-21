# The small validation — the geometry contrast the PI made a precondition of v5

*2026-07-27. The PI: **"we will consider all new trainings with the larger hfov after a small
validation, the v5 should train with the adapted setup."** This is that validation: what is runnable,
what was pre-registered before any number existed, what was run, and the verdict.*

⛔ **Nothing here launches v5.** A v5 launch is the PI's go.

---

## 0. HEADLINE — the feasibility answer, first, because it changed the design twice

⭐⭐ **1. OPTION (a) — old 256×256 / 51.4° vs new 256×640 / 120° cylindrical — IS RUNNABLE, ON POD2,
WITHOUT TOUCHING POD1. IT IS RUNNING.** The brief's binding constraint was falsified at two
independent probes.

⛔⛔ **2. BUT THE COST IS 8× WHAT THE PROGRAM ASSUMED, AND THAT IS THE MORE IMPORTANT FINDING.** One
**optimizer** step at the shipped `--batch 8 --accum 8` is **eight micro-batches**. MEASURED here:
**8.578 s/step at 256×256**. ⇒ the PREP card's *"matched short runs (hours, not a week)"* at 6 000
steps × 3 arms is **54 GPU-h**, not hours. Re-scoped to **1 500 steps** (13.5 GPU-h) — same matched
design, and the limitation is stated rather than hidden (§4).

### 0.1 pod1 is not the only holder — and pod2 has the whole thing

The brief stated: *"The old-geometry corpus is the raw epcache at 256×256 / 51.4°, parity key
`e438721ae894`. **pod1 holds it — and pod1 is TRAINING and must not be touched.** pod2's own clips are
key `0c5d8f7823bd`, **not parity**."*

**MEASURED on pod2** (`raw/pod2_corpus_inventory.txt`):

```
/workspace/data/physicalai_phase0/_epcache/physicalai-train-e438721ae894/  2376 ep_*.pt + 24 skip_* + DONE   260 GB
/workspace/data/physicalai_phase0/_epcache/physicalai-val-0c5f7dac3b11/     600 ep_*.pt            + DONE     66 GB
payload: frames_u8 [199, 9, 256, 256] uint8 · actions [199,2] · poses [199,4]
```

and the **trainer's own parity guard**, from the live run (`raw/smallval_A_old.log`):

```
[parity] --train-cache: physicalai-train-e438721ae894 VERIFIED — 2376 episodes,
         uid sha256 9877bef64da3… matches the committed manifest (skip-hash f09e44db).
[geometry] DEPLOYED (unchanged) - 256x256px, f_ref 266.00, pinhole, HFOV 51.39deg,
           tokens 256 (16x16 @ patch 16), state_dim 2048
```

⇒ pod2 holds the canonical parity corpus at the deployed geometry, in full — and it is not even the
only other holder: **pod3 also carries `physicalai-train-e438721ae894` and
`physicalai-val-0c5f7dac3b11`** (`/workspace/pai_epcache/`, `/workspace/s3parity/views/`).
**Root-cause class: ABSENCE FOUND AT ONE LOCATION IS NOT ABSENCE** (operating-standard rule 2).
⛔ **Had I accepted the constraint, I would have substituted option (b) — the rig question — and
shipped it under the geometry question's name.** The brief forbade exactly that, and no reader
downstream could have caught it.

### 0.2 ⭐ The 2400-vs-2376 mismatch DOES NOT TOUCH THE MEASUREMENT

**MEASURED — the 24 decode failures are entirely inside the TRAIN split. The VAL split is 600 vs 600,
identical membership, zero skips** (`raw/pod2_join_proof.txt`):

```
VAL epcache present: 600   contiguous 0..599: True
VAL v2 clip ids:     600   == parity_val_clips.txt set: True
VAL sha256(sorted) = 0b176d2e5cb4…   reproduces the committed manifest: True
*** VAL MATCHED WITHOUT RESTRICTION: True ***
*** TRAIN asymmetry: 2376 vs 2400 = 1.00 % of episodes ***
```

⇒ **the arms are *scored* on the same 600 episodes in both geometries — no restriction, no subset, no
new corpus key.** The mismatch survives only as a 1.00 % asymmetry in *training* data (§2.4).

### 0.3 The restriction is expressible — verified before promising — but it is not *registrable*

The brief required me to *"verify you can express that restriction before promising it."* Done, and it
holds (`raw/pod2_join_proof.txt`, all MEASURED):

| check | result |
|---|---|
| epcache index space contiguous `0..2399` | ✅ `True` |
| `present ∪ skips` == 2400, `present ∩ skips` == ∅ | ✅ 2376 + 24 = 2400 |
| `parity_train_clips.txt` ordered == sorted ⇒ ordinal *i* ↔ `sorted_clips[i]` | ✅ `True` |
| its digest reproduces the committed manifest | ✅ `e61a04553df5…` |
| ⇒ sha256(sorted **shared 2376**) | `34b233015fe4e10b5cf807b7626848b79340d47983e4b6b5eebbcfbc49a8d660` |
| ⇒ sha256(sorted **excluded 24**) | `077ebf063a27038637f31c23a5bfcb236824b9ca596eaffa7a55e5bc60390e88` |

**The join is proved. I did not apply it, and the reason is the parity invariant, not cost.** A
2376-clip v2 cache is a *subset* of the registered 2400-clip sibling: `register_v2_geometry_sibling`
mints a key **only if** the clip-id digest matches the parent exactly, so it refuses; `corpus_key_of`
returns `None` for the new directory; `--require-parity` then refuses to train. **That machinery is
right — a subset IS a re-selection of episodes, which `CLAUDE.md`'s parity invariant forbids.**
Minting a manifest entry for a 2376-clip corpus to win a 1 % matching argument would put a
non-canonical corpus into the sacred manifest permanently. ⇒ handled as a **named, bounded,
direction-known confound** (§2.4).

### 0.4 What is running, and why three arms

⭐ Two arms answer the PI's question; the third stops it being confounded. v5's *actual* frame is not
256×640 — it is the **176×624 rig-clean sub-frame** (both published v5 launch commands carry
`--v2-subframe 176x624`). "Old vs v5's frame" would move **two** things at once.

| arm | corpus | frame | tokens | isolates |
|---|---|---|---:|---|
| **A_OLD** | raw epcache, 2376 | 256×256, f_ref 266.00, **pinhole**, **51.39°** | 256 | the deployed geometry |
| **B_WIDE** | v2 sibling, 2400 | 256×640, f_ref 305.5775, **cylindrical**, **120°** | 640 | ⭐ **A vs B = the PI's question, and ONLY the field** |
| **C_V5** | v2 sibling, 2400 | **176×624** slice of the same cache | 429 | **B vs C = the rig fix alone**; A vs C = what v5 actually is |

Everything else is byte-identically matched (§2.2). All three printed **`PREFLIGHT: OK`, exit 0** on
the synced pod (§3).

---

## 1. 🔴 CAN THIS VALIDATION DECIDE ANYTHING? — the MDE, from MEASURED program data

**Yes at n = 120 val episodes for an effect ≥ 0.0059; NO at n = 40; and a pre-registered escalation to
n = 600 (MDE 0.0026) if the read is not separated.** This is a design decision the MDE forced.

### 1.1 The primary, named exactly

**`pseudosim_composite_PSS_recovery_progress@twosided_v2`** — the map-free pseudo-simulation
composite. ⛔ **NOT `ade_0_2s`.** ⛔ **And NOT the same metric as the `clamp_v1` composite every PSS
number published before 2026-07-28 was computed under** — the code carries that warning in the value's
own `name`, and I never compare across terms. **Both terms are reported**, because the recompute from
the banked per-window `.npz` is CPU-only and free (MEASURED elsewhere: 20 arms × 6 terms in 76.5 s).

* **Gate: the PANEL gate**, never the per-arm one — a component enters only if `discriminative_range`
  admits it **for every arm in the panel**. ⚠️ This matters: under the per-arm gate the same published
  contrast reads `refc_base − v4_oracle` = **−0.1269** instead of **−0.0217** — a **5.2×** inflation
  with a verdict flip, because the two arms carry *different weight sets* and the paired delta then
  mixes a metric change with a model change.
* **Estimator:** paired episode-cluster bootstrap (`taniteval/ci.py`, B = 2000, unit = **episode**).
  ⛔ `overlapping_holdout_se` refused; no combination in quadrature.
* **Substrate:** the panel's **full 21-point grid** (`pseudosim.default_grid()`: 7 `dyaw` × 3 `dlon`,
  lateral refused in code), stride 8, horizon 20 (2.0 s). ⚠️ **Not** `heldout_gate.probe_grid()`'s
  3-point mid-run subset — every MDE anchor below was measured on the 21-point grid, and a different
  substrate is a different metric.

### 1.2 ⭐ The MDE — anchored on the MEASURED contrast most analogous to mine

⛔ **I first sized this off the gate's injected-degradation stop proof (−0.0647 at n = 8) and that was
the WRONG EVIDENCE CLASS** — two *scorings of one checkpoint* are far more tightly paired than two
*independently trained* ones, so it understates the noise. Retracted before use.

The right anchor is MEASURED and already in the program: the **20-arm pseudo-simulation panel**
(`…/incoming/2026-07-27-pseudosim-arm-panel/artifacts/pseudosim_arm_panel.json`, reproduced in
`…/2026-07-28-tactical-action-input/artifacts/blockA/blockA_full_panel_20arm.json`). Its
cross-checkpoint paired CIs, all at **n = 40 episodes / ~15.5 k windows**, panel gate, B = 2000:

| pairing class | example | ci95 half-width |
|---|---|---:|
| same family, same goal interface, different trained ckpt | `refc_base − refc_small` +0.0002 [−0.0024, +0.0025] | **0.0025–0.0030** |
| ⭐ **same architecture + data, ONE training flag, two separate from-scratch runs** | **`nospeed_tactical_oracle − v1_tactical_oracle` −0.0055 [−0.0130, +0.0011], n.s.** | **0.0068–0.0071** |
| different family and/or goal interface | `v1_tactical_oracle − v4_oracle` −0.0147 [−0.0274, −0.0028], SEP | 0.0123–0.0152 |

⭐ **My design is exactly the middle row** — same architecture, same corpus, one input change, two
separate from-scratch runs. ⇒ **SD_paired ≈ 0.0071 · √40 / 1.96 = 0.0229.**
`MDE(α = 0.05 two-sided, 80 % power) = 2.8016 · SD/√n`:

| n (val episodes) | MDE | cost (v4-class arm, ESTIMATED from the panel's own wall-clock) |
|---:|---:|---|
| 8 (the trainer's own gate probe) | 0.0227 | free, runs during training |
| 40 (`--eval-episodes` default) | **0.0102** | ~13 min/arm |
| ⭐ **120 — the registered primary** | **0.0059** | ~40–50 min/arm |
| 600 (the full matched split) | **0.0026** | ~3.2 h/arm |

**Is that decision-relevant?** Against MEASURED effects on this same metric: an entire architecture
generation (`v1 → v4`) is **0.0147**; a single training-flag change (the speed channel) is **0.0055**.
⇒ **at n = 120 the design resolves an effect finer than a whole architecture generation.**
⚠️ **Stated plainly: an effect the size of *adding the speed input* (0.0055) sits just BELOW the
n = 120 MDE.** If widening the field buys less than that, this design returns **UNPOWERED, not a
null** — which is why the escalation below is registered *now*, not chosen after seeing a number.

**Registered escalation rule:** if the n = 120 primary is **not separated** AND PC-1 passes, extend
the *same* arms to **n = 600** (an order-preserving prefix extension — `12 ⊂ 40 ⊂ 120 ⊂ 600`, so it
**adds episodes and re-selects none**; parity holds) and report at MDE 0.0026. **The rule is fixed
before the data; it is a sequential design, not a fishing expedition.**

### 1.3 ⭐ THE ANTI-C13 MECHANISM: no null is admissible until the instrument is shown to have range

**Registered before any arm finished.** A "not separated" is reported as `NO DIFFERENCE` **only if**:

* **PC-1 — the BLIND control, and it doubles as a training-sufficiency gate.** For **each** arm,
  `sighted − blind` (the identical planner on a zeroed observation, `panel_run.BlindWrapper`
  verbatim) must be **separated-better** on the same rows. ⭐ **This is the exactly-right control for
  a GEOMETRY question: a composite that cannot separate an arm from itself-with-no-image cannot
  possibly resolve a change in what the image SHOWS.** It also catches the other failure this short
  run is exposed to — an arm that has not yet learned to use vision at all. Either way the verdict is
  **`INSTRUMENT-BLIND`**, never a null. *(The panel's own reference value for a converged arm:
  `v4_oracle − v4_blind` = **+0.1882 [+0.1240, +0.2557]**, SEP.)*
* **PC-2 — component range.** The panel gate must admit ≥ 1 weighted component across all arms;
  `composite()` **refuses to emit** (`VacuousMetric`) otherwise, and a refusal is a reported outcome,
  never a silently dropped arm.

⚠️ **Predicted in advance, not discovered afterwards:** `comfort` will be **inadmissible** (published
as saturated ≥ 99.9 %; v4's fan measured a literal constant) and `no_collision`/`ttc` are
**structurally unavailable** (no cuboids on this corpus). So the composite should reduce to
**`ego_progress` + `recovery`** — which is exactly the longitudinal/lateral split §5 decomposes on.

### 1.4 What each outcome licenses — all three committed in advance

| outcome | rule (primary, paired, n = 120 → 600, B = 2000) | what it licenses |
|---|---|---|
| ⭐ **WIDE BETTER** | `B_WIDE − A_OLD` separated **positive** | v5 trains wide; the PI's hypothesis confirmed **on the planning primary**, not only on an encoder probe. |
| **NO DIFFERENCE** | not separated **AND PC-1 passed on both arms** | v5 trains wide **on coverage grounds, not performance grounds** — and the PI is told plainly we are paying **+41.5 GPU-h** (256×640) or **+15.5 GPU-h** (176×624) on a 30 k run (§4.3) for no measured planning gain, and that the resolution study's **+0.04246 AP** encoder-probe advantage **did not convert into planning** at this run length. |
| ⛔ **WIDE WORSE** | `B_WIDE − A_OLD` separated **negative** | **BLOCKS** the wide geometry as configured. Live, not a formality: the resolution study already found the wide frame **separated-WORSE on ego yaw rate (−0.03546 R²)**, and the gate card calls ego-motion perception *"the real item."* |
| **INSTRUMENT-BLIND / UNPOWERED** | PC-1 fails, or not separated at n = 600 | **no geometry verdict is claimed**, and the n needed is stated. |

**Secondary, same estimator:** `C_V5 − B_WIDE` = **the rig fix alone**; `C_V5 − A_OLD` = **what v5
actually is**. ⚠️ `WIDE WORSE` on B with `NOT WORSE` on C would say the penalty is the
**rig-correlated masked region**, not the field — and would license v5 **at 176×624 only**. Committed
here, before the numbers.

⚠️ **Registered scope limit:** these are **1 500-step from-scratch arms** (0.24 epoch) with a
20×-compressed curriculum. They measure the geometry's effect **at that point on the curve**. A result
here does **not** license a 30 k extrapolation — `CLAUDE.md` caps extrapolation at 2×, and 1 500 → 30 k
is 20×.

---

## 2. PRE-REGISTRATION — fixed before any checkpoint existed

### 2.1 Primary, diagnostics, estimator
As §1.1. **Diagnostics only, never the verdict:** `ade_0_2s`, `ade_lat`, `ade_lon`,
`wm_canary_ade_2s`, `miss_at_2m`, and the trainer's own 8-episode gate probes (MDE 0.0227 — reported
**with** that power, never as a verdict).
**Coverage is quoted with every number** (the `selected_frac` discipline, class C19): per arm, the
finite-value fraction per component plus window and episode counts.
⛔ **Nothing is held to v1's 0.4271** — verified in code (`rollout.py:170`,
`actions_source="expert_future"`), that is `wm_fidelity_ade_2s`, the world model handed the **true
actions**, not a planning bar.

### 2.2 What is matched, byte for byte
From `--print-launch` (`raw/printlaunch_*.txt`): `--from-scratch` · same anchors file · `--steps 1500`
· **`--batch 8 --accum 8`** (effective 64; the trainer's own banner checks it, and `--batch 16` OOMs
on a 44 GB A40) · `--lr-head 1e-4 --lr-trunk 1e-4` · `--warmup 100` · `--workers 8` ·
`--phase-a-steps 100 --phase-b-steps 400 --gate-step 400` · `--eval-every 500` · `--save-every 500` ·
`--eval-episodes 40` · `--rollout-k 4` · `--lam-mult-floor 0.25` · `--labels v3` ·
`--lambda-plan sched` · `--strategic full` · `--long-horizon-k 50` ·
`--heldout-gate --heldout-every 500 --heldout-episodes 8 --heldout-patience 2 --heldout-stride 8
--heldout-nboot 2000` · **`--heldout-goal dropped`** (the shipped default) · **panel-wide gate, never
per-arm** · `--seed 0`.

**Only the corpus format and the four geometry flags differ.** `state_dim` stays 2048 at every frame —
the readout is a geometry firewall — so predictor, policies and grounding heads are untouched and the
contrast really is the encoder's input.
⭐ The anchors file is **the same** for all three (`flagship_v4_anchors_dense.pt`, `[256, 20, 2]`,
`method fps`, horizons 1..20). Anchors live in **metric trajectory space, not pixels**, so sharing them
is what makes the arms comparable — not a leak.

### 2.3 The bar and the n
n = **120 episodes** (escalating to 600 by the §1.2 rule), B = 2000. **Bar: separation of the paired
interval from 0 on the primary.** ⚠️ **No absolute threshold is set on the composite's value**,
deliberately: this is a *contrast between arms trained here*, and every historical PSS value is a
**different metric** (`clamp_v1`). Registering an absolute bar against a number computed under another
term is the exact silent-redefinition failure the code's `_progress_term_warning` exists to prevent.

### 2.4 ⚠️ The one confound I could not remove — its size and its DIRECTION
**B_WIDE and C_V5 train on 2400 episodes; A_OLD on 2376** — **+1.00 %** for the wide arms. §0.3 says
why removing it costs more than it buys. Bounded, MEASURED: `[data] train windows=406099`, and
1500 × 64 = **96 000 windows ⇒ 0.24 epoch**. Both arms see far less than one pass, so the extra
episodes enter only as ~1 % of sampled windows.

⭐ **The direction is known and asymmetric, so each verdict is read accordingly:**
* **WIDE BETTER** is **partly confounded** — the effect size is an **upper bound** on pure geometry;
* **WIDE WORSE** is **CONSERVATIVE** — the confound pushes the other way, so a loss is understated;
* **NO DIFFERENCE** is very slightly optimistic for wide.

**`C_V5 − B_WIDE` is entirely free of this confound** (same corpus, same 2400 clips, same cache — only
the loader slice differs).

---

## 3. THE POD2 SYNC — evidence, because a `git log` is not proof

⚠️ pod2's `/workspace/TanitAD/stack` was at **`0f93b98`** with **52 modified tracked files** and had
**none** of the v5 fixes: `stack/tanitad/train/heldout_gate.py` **did not exist at all**, and
`--heldout-goal`, `--v2-train-cache`, `_render_bounds` and `register_v2_geometry_sibling` all grepped
**0**. A launch from there resurrects the crashing held-out gate.

**Sequence (`raw/pod2_sync.txt`):**
1. **Nothing destroyed** — pod2's drift snapshotted first: `pod2:/workspace/_pod2_drift_2026-07-27.patch`
   (28 009 lines) + `/workspace/_pod2_status_2026-07-27.txt` (88 lines).
2. `git -c core.autocrlf=false -c core.eol=lf archive HEAD -- stack taniteval` (662 entries), shipped
   and extracted over `/workspace/TanitAD/`.
   ⚠️ **A trap worth recording: the FIRST archive was CRLF-poisoned.** The dev box has
   `core.autocrlf=true` and no `.gitattributes`, so plain `git archive` rewrote every `.py`/`.sh` to
   CRLF. It was caught only because the parity manifest's sha256 came back **`b3c0ecc0…` instead of
   `51b1792f…`** — a byte check on a file whose digest happened to be published. **A `#!/bin/bash\r`
   is a `bad interpreter` failure hours later; Python swallows it silently.**
   *(Root-cause class: **A SYNC THAT REPORTS SUCCESS AND SHIPS DIFFERENT BYTES.** `git archive` is not
   byte-faithful on a Windows checkout unless the eol filters are explicitly disabled — worth adding
   to the traps list, since "sync the pod" is now a standing runbook step.)*
3. **Verified — not with `git log`:**

| check | result |
|---|---|
| `stack/tanitad/data/parity_manifest.json` sha256 | ✅ **`51b1792ff52aa8ea37cbe62bbf796fa1d23464bc76a0ef92b6a3b48090cd9cea`** — byte-identical to the repo copy |
| files containing CR under `stack/` + `taniteval/` | ✅ **0** |
| ⭐ **a real `import tanitad`** | ✅ `tanitad OK from /workspace/TanitAD/stack/tanitad/__init__.py` |
| `parity.register_v2_geometry_sibling` / `assert_v2_parity_cache` | ✅ both present |
| `heldout_gate.PRIMARY_NAME` | ✅ `pseudosim_composite_PSS_recovery_progress@twosided_v2` |
| `heldout_gate.GOAL_OPTION_DEFAULT` | ✅ `dropped` |
| `taniteval.ci.paired_episode_cluster_bootstrap` / `_render_bounds` | ✅ both present |
| `taniteval.pseudosim.metric_id()` | ✅ `PSS_recovery_progress@twosided_v2` |
| torch / torchvision / CUDA | ✅ `2.4.1+cu124` / `0.19.1+cu124` / `True` |
| ⭐ **a real `--print-launch`, all three arms** | ✅ **`PREFLIGHT: OK`, exit 0, ×3** (`raw/printlaunch_*.txt`) |
| both v2 parity lines | ✅ `2400 clips e61a04553df5…` · `600 clips 0b176d2e5cb4…`, matched to the committed manifest |
| the raw-epcache parity line | ✅ `2376 episodes, uid sha256 9877bef64da3…, skip-hash f09e44db` |

⚠️ **One real gap found and not papered over:** the val epcache guard prints *"count OK (600/600) —
**NO uid digest committed for this split**, so this is a COUNT-ONLY check."* I did not rely on it; the
§0.2 membership proof is stronger than what the trainer can check — it compares the actual clip-id sets
of **both** corpora against `parity_val_clips.txt` and reproduces the committed digest.

---

## 4. THE RUNS — and the cost finding that re-scoped them

### 4.1 ⛔ MEASURED: one optimizer step is 8 micro-batches, and nobody was pricing it that way

`V5_EVALUABLE.md` §8.2's table is headed **"full step s"** and gives 1.6944 s for 256×640. Its own
`images/step` column shows **144 = batch 8 × 18** — i.e. **one MICRO-batch** (`v4_loss_step` encodes
the obs window (8) *and* `plan.needed_fut` (10)). At the shipped `--accum 8`, a real **optimizer** step
is **8×** that.

✅ **To be fair to that document: its §8.3 GPU-hour table is CORRECT and does multiply by accum 8**
(*"30 000 optimizer steps × accum 8"* → 256×640 ≈ 113 GPU-h; 1.6944 × 8 × 30 000/3600 = 113.0 ✓).
**The defect is a column label in §8.2, not the estimate in §8.3** — and I nearly published the
stronger, false version of this. *(Class: A CORRECT NUMBER BEHIND AN AMBIGUOUS LABEL — the label is
what gets quoted.)*

**MEASURED here, and it cross-validates §8.2 independently** (`raw/step_rate.txt`):

| arm | tokens | s / **optimizer** step | basis |
|---|---:|---:|---|
| **A_OLD 256×256** | 256 | **8.578** | ⭐ **MEASURED (ours)** — 50 steps, `elapsed_s` 19.2 → 448.1 |
| B_WIDE 256×640 | 640 | 13.56 | ESTIMATED = 1.6944 × 8 (INHERITED micro-step) |
| C_V5 176×624 | 429 | 10.39 | ESTIMATED = 1.2988 × 8 |

⭐ **The cross-check that says this is real and not an outage** (`CLAUDE.md`: verify before alarming):
8.578 s ÷ 8 = **1.0723 s/micro-step at 256 tokens**, against the independently measured **1.0835 s at
288 tokens** — the 256-token point sits just below the 288-token point, exactly as it must. Two
instruments, two hosts, one curve.

### 4.2 The re-scope, stated rather than hidden

| steps | A | B | C | A+B | A+B+C |
|---:|---:|---:|---:|---:|---:|
| 6 000 (originally launched) | 14.3 h | 22.6 h | 17.3 h | 36.9 h | **54.2 h** ⛔ |
| ⭐ **1 500 (running)** | **3.6 h** | **5.7 h** | **4.3 h** | **9.2 h** | **13.5 h** |

The 6 000-step arm A was **killed by explicit PID** (chain 3693916, trainer 3693920 — never `pkill
-f`) after 50 steps; its log is banked as `raw/A_old_6000step_train_log.jsonl` because **it is the
cost measurement**. The curriculum was re-compressed by the same factor so the arms sit in the same
phase (A[0,100) B[100,400) C[400,1500)).

⚠️ **The honest consequence: 1 500 steps is 0.24 epoch.** That is why PC-1 (§1.3) is also a
*training-sufficiency* gate — if an arm cannot beat its own blind twin, no geometry conclusion is
admissible from it, and the answer is `INSTRUMENT-BLIND`, not a null.

### 4.3 ⭐ A number the PI needs regardless of the verdict: what the wide geometry costs v5

Combining the MEASURED 256×256 point with the INHERITED micro-step ratios, at **30 000 optimizer
steps × accum 8** on one A40:

| frame | GPU-h for a 30 k v5 | vs the deployed frame |
|---|---:|---:|
| deployed 256×256 / 51.4° | **≈ 71.5** (⭐ MEASURED basis) | — |
| **176×624 / ~117° (v5's actual frame)** | ≈ 87 | **+15.5 GPU-h** |
| 256×640 / 120° | ≈ 113 | **+41.5 GPU-h** |

**So the wide geometry as v5 will actually run it (176×624) costs ~+22 % of a 30 k run.** That is the
price the verdict below is buying or refusing.

### 4.4 Ops

| | |
|---|---|
| launched | **2026-07-27T18:49:35Z** on pod2 (A40, **0 MiB used, idle** before launch) |
| chain script | `pod2:/workspace/smallval/chain.sh` (staged as `code/chain.sh`) |
| **chain PID** | **3695397** |
| **arm A trainer PID** | **3695401** |
| chain log | `pod2:/tmp/smallval_chain.log` |
| per-arm logs | `pod2:/tmp/smallval_{A_old,B_wide,C_v5}.log` |
| outputs | `pod2:/workspace/smallval/{A_old-256x256,B_wide-256x640,C_v5-176x624}/` |
| sentinels | `pod2:/workspace/smallval/{A,B,C}_DONE`, `CHAIN_DONE` |

⛔ **pod1 was NEVER CONTACTED.** ⛔ **pod3 was probed read-only** (inventory only); **YouTube never
touched**. ⛔ Kills were by **explicit PID only**. Disk judged by a **real `dd`** (8 GiB at 504 MB/s;
500 MiB at 388 MB/s) — **never `df`**. `PYTHONPATH` and `OMP_NUM_THREADS=6` set on every invocation
*(the panel recorded a 113-thread stall on a 96-core pod without it)*.

---

## 5. 🔴 A SILENT DEFECT FOUND WHILE WIRING THE EVAL — it would have published a wrong number

⛔⛔ **`taniteval` HARDCODES A STALE STACK PATH, AND ON POD2 THAT TREE IS PRE-v5.**

MEASURED: **~15 `taniteval` submodules** (`bench`, `closedloop`, `corridor`, `data`, `driving`,
`blind_baseline`, `cam_overlay`, `corpus_overlay`, `direct_overlay`, `ab`, …) contain a literal
`sys.path.insert(0, "/root/TanitAD/stack")`. On pod2 `/root/TanitAD` is a **12 MB tree with no
`.git`**, **no `tanitad/train/heldout_gate.py`**, and **`resolve_v2_frames` grep count 0**.

⇒ **merely importing `taniteval.pseudosim` re-points `tanitad` at pre-v5 code.** MEASURED, first
attempt (`raw/eval_plumbing_RED.txt`):

```
FAIL DeployableSurfacePlanner -> ModuleNotFoundError("No module named 'tanitad.train.heldout_gate'")
FAIL make_goal_kwargs_fn      -> ModuleNotFoundError("No module named 'tanitad.train.heldout_goal'")
FAIL resolve_v2_frames -> ImportError(... from 'train_flagship_v4' (/root/TanitAD/stack/scripts/train_flagship_v4.py))
```

⚠️ **The dangerous part is not this crash — it is the case that does NOT crash.** A module that
exists in *both* trees resolves silently to the stale one, and the eval publishes a number computed
by pre-v5 code. **My own §3 sync proof would not have caught it**: `import tanitad` on its own
resolved correctly; the shadowing only happens once `taniteval` is imported.

✅ **The cure already ships and nobody was using it.** `taniteval/__init__.py` reads
**`TANITEVAL_STACK_OVERRIDE`**, inserts it at `sys.path[0]` and imports `tanitad` **there, first**, so
the `sys.modules` cache beats every later insert. `code/smallval_pseudosim.py:pin_stack()` sets it and
then **verifies the outcome on six modules rather than trusting the env var** — an env var that is set
but ineffective is the same 'reports success, ships different bytes' shape as §3's CRLF trap.

**RED → GREEN, both MEASURED** (`raw/eval_plumbing_GREEN.txt`):

```
[taniteval] tanitad OVERRIDE -> /workspace/TanitAD/stack (/workspace/TanitAD/stack/tanitad/__init__.py)
✅ PINNED: {"stack": "/workspace/TanitAD/stack", "verified_modules": 6}
✅ DeployableSurfacePlanner / make_goal_kwargs_fn / resolve_v2_frames / load_v4_from_ck /
   build_v2_providers ALL import
PRIMARY_NAME: pseudosim_composite_PSS_recovery_progress@twosided_v2
```

🔴 **ESCALATION (not a README note): every v5-era eval command in the program needs
`TANITEVAL_STACK_OVERRIDE=/workspace/TanitAD/stack`, including the `eval_flagship_v4` and corridor
commands in `V5_EVALUABLE.md` §7.3–7.4. None of them carry it.** *(Root-cause class: A HARDCODED
ABSOLUTE PATH INSIDE A SHARED LIBRARY — the drift is not in the pod's checkout, it is in the code that
every host imports.)*

### 5.1 Two things the frame seam printed that are worth reporting

Verified on pod2 through the trainer's own `resolve_v2_frames` (`raw/eval_plumbing_GREEN.txt`):

* `--v2-subframe none` → TRAIN **256×640**, f_ref 305.5775, **HFOV 120.000°**, cylindrical, no slice.
* `--v2-subframe 176x624` → TRAIN **176×624**, f_ref 305.5775, **HFOV 117.000° / VFOV 32.131°**,
  rows `[40:216]`, cols `[8:632]` — a pure pixel slice. ⭐ **So v5's actual frame is 117°, not 120°:
  the rig-clean fix costs 3° of field.** Worth naming, because "v5 is 120°" is now in several docs.
* ⚠️ **The geometry layer fires an unresolved PI decision, automatically:** *"this frame (120.00deg)
  EXCEEDS comma2k19's entire field (65.203deg). comma2k19 cannot supply it at any resolution — it must
  be letterboxed …, given its own frame, or dropped from the mix. That is a PI decision, not a
  default."* This is `PREP.md` §3 item 7's open comma2k19 question, still open.

---

## 6. THE EVAL — armed and waiting, so it needs no further intervention

⛔ **It cannot run now, and that is deliberate.** pod2's cgroup is at **53.9 GB of a 55.0 GB limit**
while training (MEASURED; trainer RSS 16.6 GB + 8 workers at ~5.5–6 GB). **An eval OOM-killed the
flagship on 2026-07-16** and that must not repeat. The eval chain therefore **blocks on
`CHAIN_DONE`**.

| | |
|---|---|
| **eval chain PID** | **3696611** (`pod2:/workspace/smallval/evalchain.sh`, staged as `code/evalchain.sh`) |
| state | waiting on `/workspace/smallval/CHAIN_DONE` since `2026-07-27T19:00:18Z` |
| logs | `pod2:/tmp/smallval_evalchain.log`, `/tmp/smallval_ps_<arm>.log`, `/tmp/smallval_combine.log` |
| per-window dumps | `pod2:/workspace/smallval/ps/pw_<arm>.npz` + `meta_<arm>.json` |
| final result | `pod2:/workspace/smallval/raw/smallval_result.json` |
| sentinels | `PS_AB_DONE` (the PI's contrast + controls scored), `EVAL_DONE` |

**Six runs, in priority order:** `A_old`, `A_old_blind`, `B_wide`, `B_wide_blind` → `PS_AB_DONE`;
then `C_v5`, `C_v5_blind`; then the combine. n = 120, stride 8, horizon 20, 21-point grid,
`--goal-option dropped`, `ckpt.pt` (the final step-1500 checkpoint — **the same rule for every arm**,
because `ckpt_best.pt` can sit at different steps and would break the matching; each arm's
`ckpt_best.pt` step is reported as a diagnostic).

**The instruments, and what is new versus reused:**

| piece | status |
|---|---|
| planner | ⭐ **reused** — the trainer's own `DeployableSurfacePlanner` (the surface the v5 gate stops on) |
| goal kwargs | **reused** — `heldout_goal.make_goal_kwargs_fn("dropped", …)` |
| frame seam | **reused** — `train_flagship_v4.resolve_v2_frames`, so eval and train cannot disagree |
| scoring / gate / bootstrap | **reused** — `taniteval.pseudosim` + `taniteval.ci` |
| blind control | **reused verbatim** — `panel_run.BlindWrapper` |
| `.npz` schema | ⭐ **identical to `panel_run._save_pw`**, so the published 20-arm `panel_combine.py` consumes it unmodified and the numbers stay comparable to the MDE anchors |
| **new** | only the *wiring*: v2-provider loading + `frame=` pass-through + the shadowing guard — the three things `panel_run.py` cannot do (stated in the driver's docstring) |

<!-- RESULTS-ANCHOR -->

*Results, decomposition and verdict are appended here as each arm lands — this document is banked
incrementally, not held to the end.*

**State at hand-off:** arm A_OLD training, **8.63 s/step replicated** on the relaunch (vs 8.578 on the
first), ETA ≈ 22:25 UTC; B ≈ 04:05 UTC; C ≈ 08:25 UTC; eval ≈ +4–5 h. **No number for the primary
exists yet, and none is asserted.**

---

### 6.1 SUITES — zero new skips

| suite | brief's baseline | measured here | new skips |
|---|---|---|---|
| `stack/` (dev box, project venv) | 1523 passed, 12 skipped | ✅ **1523 passed, 12 skipped** (105.5 s) | **0** |
| `taniteval/` (dev box) | 644 passed | ✅ **644 passed** (68.5 s) | **0** |

**No shipped code was modified by this stream** — the only new files are under
`…/incoming/2026-07-27-small-validation/`, so the suites are expected to be unchanged and are. The
shadowing fix (§5) is implemented **in this stream's own driver**, not by editing `taniteval`, because
changing ~15 shipped modules' `sys.path` behaviour is an integration decision for the owner (§7.1
item 1), not something to slip in under a validation.

---

## 7. DELIVERABLE MANIFEST

| artifact | where it lives | only one place? |
|---|---|---|
| `SMALL_VALIDATION.md` (this) | `repo:…/incoming/2026-07-27-small-validation/` **(staged)** | no |
| ⭐ `code/smallval_pseudosim.py` — the eval driver + `pin_stack()` shadowing guard | `repo:` **(staged)** + `pod2:/workspace/smallval/code/` | no |
| ⭐ `code/smallval_combine.py` — panel gate, both progress terms, paired bootstrap, decomposition | `repo:` **(staged)** + `pod2:/workspace/smallval/code/` | no |
| `code/chain.sh` — the three matched training arms | `repo:` **(staged)** + `pod2:/workspace/smallval/` | no |
| `code/evalchain.sh` — the armed post-training eval | `repo:` **(staged)** + `pod2:/workspace/smallval/` | no |
| `raw/pod2_corpus_inventory.txt` — ⭐ the feasibility falsification | `repo:` **(staged)** + `pod2:/workspace/smallval/raw/` | no |
| `raw/pod2_join_proof.txt` — ⭐ the val-matched proof + the 2376/2400 digests | `repo:` **(staged)** + `pod2:` | no |
| `raw/pod2_sync.txt` — the sync verification incl. the real `import tanitad` | `repo:` **(staged)** + `pod2:` | no |
| `raw/pod3_inventory.txt` — the second-probe evidence | `repo:` **(staged)** | ⚠️ **yes** (a read-only probe transcript; reproducible in one command) |
| `raw/printlaunch_{A_old,B_wide,C_v5}.txt` — `PREFLIGHT: OK` ×3 | `repo:` **(staged)** + `pod2:` | no |
| `raw/step_rate.txt` + `raw/A_old_6000step_train_log.jsonl` — ⭐ the cost finding | `repo:` **(staged)** + `pod2:` | no |
| `raw/eval_plumbing_{RED,GREEN}.txt` — the shadowing defect and its fix | `repo:` **(staged)** | ⚠️ **yes** (probe transcripts) |
| pod2 drift snapshot (28 009-line patch) | `pod2:/workspace/_pod2_drift_2026-07-27.patch` | ⚠️ **yes** — deliberately: it is pod-local uncommitted drift from other streams, not this stream's work, and staging it would sweep 52 foreign files into the index |
| training outputs, `pw_*.npz`, `smallval_result.json` | `pod2:/workspace/smallval/` | ⚠️ **yes until the runs finish** — the pull is the last step |

**I ran no `git commit`, no `git push`, and switched no branch.** I `git add`ed only this stream's
paths, and **verified with `git ls-files --cached` (13/13), not with the exit code** — the trap
`CLAUDE.md` warns about did not fire, but it was checked rather than assumed.

### 7.0 ⚠️ THE INDEX-SWEEP HAZARD FIRED AGAIN, MID-SESSION — fourth recorded occurrence

`HEAD` advanced twice while this stream worked — `f45b100 → a32bdb2` (docs only) and then
`a32bdb2 → 9cbe65e` *("MODEL_REGISTRY §1.5.5 was FALSE …")*. **That second commit swept an EARLY
draft of this file into it**, under a registry message:

```
git show --stat --name-only 9cbe65e -- <this dir>
  → …/2026-07-27-small-validation/SMALL_VALIDATION.md
git show HEAD:…/SMALL_VALIDATION.md | grep -c TANITEVAL_STACK_OVERRIDE  →  0
  (i.e. the committed copy PREDATES §5; the current one is staged on top)
```

✅ **Nothing is stranded** — all 16 files are in the repo and staged, which is what
`AGENT_OPERATING_STANDARD` rule 1 requires. ⚠️ **But the attribution is wrong and `git log --grep`
will not find this work.** Findable by path only. ⛔ I did not attempt a corrective commit:
`git commit -- <pathspec>` **segfaults on this repo** (`CLAUDE.md`), and an `--amend` would re-open the
whole index and repeat the failure. 🔴 **For the orchestrator, not for me to fix.**

✅ **Checked, because it would have invalidated §3:** `git diff --name-only a32bdb2..HEAD -- stack
taniteval` returns **0 files**, so the pod2 sync is still current at HEAD. Had it not been, the pod
would need re-syncing before the eval runs.

### 7.1 🔴 INTEGRATION NEEDED — stated here, not in a README

1. ⭐⭐ **`TANITEVAL_STACK_OVERRIDE` is missing from every v5 eval command in the program**, including
   `V5_EVALUABLE.md` §7.3 (MODE A / MODE B) and §7.4 (`run_gate.py check`). On pod2 those commands
   import `taniteval` and therefore silently prepend a **pre-v5 `/root/TanitAD/stack`**. **This is the
   highest-blast-radius item in this report** — it can produce a plausible wrong number rather than an
   error (§5).
2. **`V5_EVALUABLE.md` §8.2's column header "full step s" is a MICRO-batch step.** §8.3 is correct and
   multiplies by accum 8; only the label misleads, and the label is what gets quoted. A one-word fix
   ("micro-step s") prevents an 8× sizing error (§4.1).
3. **`git archive` is not byte-faithful from a Windows checkout** (`core.autocrlf=true`) — it ships
   CRLF and breaks every `.sh`. "Sync the pod" is now a standing runbook step, so
   `-c core.autocrlf=false -c core.eol=lf` belongs in the traps list (§3).
4. **The val epcache has no committed uid digest**, so the trainer's parity check on
   `physicalai-val-0c5f7dac3b11` is **count-only**. One command fixes it permanently:
   `scripts/make_parity_manifest.py --record --split val --cache-dir <verified cache>`.
5. **`--anchors-dense /workspace/experiments/anchors/anchors_dense_1to20.pt`, the path in BOTH
   published v5 launch commands, DOES NOT EXIST on pod2** — `/workspace/experiments/anchors/` is an
   empty directory. The real file is `/workspace/experiments/flagship_v4_anchors_dense.pt`
   (`[256, 20, 2]`, `method fps`, horizons 1..20). `--print-launch` does **not** check it, so
   `PREFLIGHT: OK` would be followed by a crash after the model build. **Fix the launch card before
   the PI's go.**

---

## 8. Provenance and evidence class of every number

| claim | class · tier | source |
|---|---|---|
| pod2 holds the full 256×256 parity epcache (2376 + 24 skips; 600 val) | **MEASURED (ours)** · DECISION-GRADE | `raw/pod2_corpus_inventory.txt`; the trainer's own parity banner |
| pod3 also holds both parity splits | **MEASURED (ours)** · read-only probe | `raw/pod3_inventory.txt` |
| val 600 vs 600, membership identical, digest reproduces the manifest | **MEASURED (ours)** · DECISION-GRADE | `raw/pod2_join_proof.txt` |
| the ordinal ↔ clip-id join; shared/excluded digests | **MEASURED (ours)** | `raw/pod2_join_proof.txt` |
| all three arms `PREFLIGHT: OK` on the synced pod | **MEASURED (ours)** | `raw/printlaunch_*.txt` |
| the pod2 sync verification table | **MEASURED (ours)** | `raw/pod2_sync.txt` |
| ⭐ **8.578 s/optimizer-step at 256×256** | **MEASURED (ours)** · DECISION-GRADE | `raw/step_rate.txt`, `raw/A_old_6000step_train_log.jsonl` |
| micro-step 1.6944 / 1.2988 / 1.0835 s; `--batch 16` OOMs; §8.3's 113 GPU-h | **INHERITED** (`V5_EVALUABLE.md` §8.1–8.3) — not re-derived | that doc |
| ⭐ **the MDE anchor: `nospeed − v1_tactical_oracle` −0.0055 [−0.0130, +0.0011], ci95 0.0071, n = 40** | **MEASURED (program, not re-derived by me)** — used to *size* the design | `…/2026-07-27-pseudosim-arm-panel/artifacts/pseudosim_arm_panel.json` |
| `v1 − v4` −0.0147 [−0.0274, −0.0028]; `v4_oracle − v4_blind` +0.1882 [+0.1240, +0.2557] | **MEASURED (program)** | same |
| per-arm gate inflates `refc_base − v4_oracle` −0.0217 → −0.1269 | **MEASURED (program)** | `…/2026-07-27-pseudosim-arm-panel/PSEUDOSIM_ARM_PANEL.md` |
| ⛔ the stop proof −0.0647 at n = 8 | **RETRACTED as an MDE anchor** — wrong evidence class (injected degradation, not a training difference) | §1.2 |
| ADE is the wrong axis (4.7× collisions; ρ = −0.36 vs Ego Progress 0.83) | **PUBLISHED / INHERITED** | `flagship-v5-retrain.PREP.md` §0 |
| wide frame separated-WORSE on ego yaw rate (−0.03546 R²); +0.04246 AP | **INHERITED** | `PREP.md` §3 item 7 |
| v1's 0.4271 | ⛔ **NOT USED** — `wm_fidelity_ade_2s`, the world model handed the TRUE actions | — |
| pod1's state | **NOT PROBED** — never contacted | — |

🔒 **Gated-confidential handled, not assumed:** every file in `raw/` and `code/` carries **counts,
digests, paths and step numbers only**. No PhysicalAI-AV clip UUID is written to the repo — the clip
lists and the 24 excluded ids stay on the pods; only their sha256 digests are quoted.
