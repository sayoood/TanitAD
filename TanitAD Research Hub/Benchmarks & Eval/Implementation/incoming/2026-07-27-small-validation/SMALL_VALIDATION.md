# The small validation — the geometry contrast the PI made a precondition of v5

*Written 2026-07-27. The PI: **"we will consider all new trainings with the larger hfov after a small
validation, the v5 should train with the adapted setup."** This document is the validation: what is
runnable, what was pre-registered before any number existed, what was run, and the verdict.*

⛔ **Nothing here launches v5.** A v5 launch is the PI's go.

---

## 0. HEADLINE — the feasibility answer, first, because it changes the design

⭐⭐ **OPTION (a) — old 256×256 / 51.4° vs new 256×640 / 120° cylindrical — IS RUNNABLE, ON POD2,
WITHOUT TOUCHING POD1. IT IS RUNNING.** The brief's binding constraint was falsified at two
independent probes.

The brief stated: *"The **old-geometry** corpus is the raw epcache at 256×256 / 51.4°, parity key
`e438721ae894`. **pod1 holds it — and pod1 is TRAINING and must not be touched.** pod2's own clips are
key `0c5d8f7823bd`, **not parity**."*

**MEASURED, on pod2, 2026-07-27** (`raw/pod2_corpus_inventory.txt`):

```
/workspace/data/physicalai_phase0/_epcache/physicalai-train-e438721ae894/   2376 ep_*.pt + 24 skip_* + DONE   260 GB
/workspace/data/physicalai_phase0/_epcache/physicalai-val-0c5f7dac3b11/      600 ep_*.pt            + DONE    66 GB
payload: frames_u8 [199, 9, 256, 256] uint8 · actions [199,2] · poses [199,4]
```

and the **trainer's own parity guard**, printed by the live run (`raw/smallval_A_old.log`):

```
[parity] --train-cache: physicalai-train-e438721ae894 VERIFIED — 2376 episodes,
         uid sha256 9877bef64da3… matches the committed manifest (skip-hash f09e44db).
[geometry] DEPLOYED (unchanged) - 256x256px, f_ref 266.00, pinhole, HFOV 51.39deg,
           tokens 256 (16x16 @ patch 16), state_dim 2048
```

⇒ **pod2 holds the canonical parity corpus at the deployed geometry, in full.** And it is not even the
only other holder: **pod3 also carries `physicalai-train-e438721ae894`** (`/workspace/pai_epcache/`,
`/workspace/s3parity/views/`) and `physicalai-val-0c5f7dac3b11`.
**Root-cause class: ABSENCE FOUND AT ONE LOCATION IS NOT ABSENCE** (operating-standard rule 2). The
constraint that would have forced me to substitute option (b) for the PI's actual question did not
exist. ⛔ **Had I accepted it, I would have shipped the rig answer under the geometry question's name
— which the brief explicitly forbade, and which no reader downstream could have caught.**

### 0.1 ⭐ And the 2400-vs-2376 mismatch DOES NOT TOUCH THE MEASUREMENT

The brief's second concern: *"they differ in episode count: 2400 vs 2376 … A matched validation must
restrict to the shared 2376."*

**MEASURED — the 24 decode failures are entirely inside the TRAIN split. The VAL split is 600 vs 600,
identical membership, zero skips** (`raw/pod2_join_proof.txt`):

```
VAL epcache present: 600   contiguous 0..599: True
VAL v2 clip ids:     600   == parity_val_clips.txt set: True
VAL sha256(sorted) = 0b176d2e5cb4…  (== the committed manifest entry)
VAL MATCHED WITHOUT RESTRICTION: True
```

⇒ **the comparison is *measured* on 600 episodes that are the same 600 episodes in both geometries,
with no restriction, no subsetting and no new corpus key.** The mismatch survives only as a 1.00 %
asymmetry in *training* data, handled in §2.4.

### 0.2 The restriction is expressible — I verified it before promising anything — but it is not
### *registrable*, and that is the correct outcome

The brief required me to *"verify you can express that restriction before promising it."* I did, and
it holds (`raw/pod2_join_proof.txt`, all MEASURED):

| check | result |
|---|---|
| epcache index space contiguous `0..2399` | ✅ `True` |
| `present ∪ skips` == 2400, `present ∩ skips` == ∅ | ✅ 2376 + 24 = 2400 |
| `parity_train_clips.txt` ordered == sorted | ✅ `True` (so ordinal *i* ↔ `sorted_clips[i]`) |
| its digest reproduces the committed manifest | ✅ `e61a04553df5…` |
| ⇒ sha256(sorted **shared 2376**) | `34b233015fe4e10b5cf807b7626848b79340d47983e4b6b5eebbcfbc49a8d660` |
| ⇒ sha256(sorted **excluded 24**) | `077ebf063a27038637f31c23a5bfcb236824b9ca596eaffa7a55e5bc60390e88` |

**So the join is proved. I did not apply it, and the reason is not cost — it is the parity invariant
itself.** A 2376-clip v2 cache is a *subset* of the registered 2400-clip sibling, so
`register_v2_geometry_sibling` refuses it by construction (it mints a key **only if** the clip-id
digest matches the parent exactly), `corpus_key_of` returns `None` for the new directory, and
`--require-parity` refuses to train. That machinery is right: **a subset IS a re-selection of
episodes, and `CLAUDE.md`'s parity invariant says anything that re-selects episodes must be refused.**
Minting a manifest entry for a 2376-clip corpus to win a 1 % matching argument would put a
non-canonical corpus into the sacred manifest permanently.

⇒ **Handled as a NAMED, BOUNDED, DIRECTION-KNOWN confound (§2.4), not silently and not by breaking
parity.** What it would take to remove it instead: a new registered corpus key plus a change to
`register_v2_geometry_sibling`'s membership rule — i.e. weakening the guard that makes v5's corpus
trustworthy. **Not worth 1.00 %.**

### 0.3 What is running, and why three arms and not two

⭐ **Two arms answer the PI's question; the third separates it from a change that would otherwise
be confounded with it.** v5's *actual* frame is not 256×640 — it is the **176×624 rig-clean
sub-frame** (both published v5 launch commands carry `--v2-subframe 176x624`). So "old vs v5's frame"
would move **two** things at once: the field AND the rig fix.

| arm | corpus | frame | tokens | what it isolates |
|---|---|---|---:|---|
| **A_OLD** | raw epcache, 2376 | 256×256, f_ref 266.00, **pinhole**, **51.39°** | 256 | the deployed geometry |
| **B_WIDE** | v2 sibling, 2400 | 256×640, f_ref 305.5775, **cylindrical**, **120°** | 640 | ⭐ **A vs B = the PI's question, and ONLY the field** |
| **C_V5** | v2 sibling, 2400 | **176×624** slice of the same cache | 429 | **B vs C = the rig fix alone**; A vs C = what v5 actually is |

Everything else is byte-identically matched (§2.2). All three printed **`PREFLIGHT: OK`, exit 0** on
the synced pod (§3).

---

## 1. 🔴 THE ANSWER THE PI NEEDS EVEN IF I AM KILLED: CAN THIS VALIDATION DECIDE ANYTHING?

**Yes at n = 600 val episodes; NO at n = 40.** That is a design decision the MDE forced, not a
preference — and it is stated before any result exists.

### 1.1 The MDE, and the arithmetic behind it

Primary = the **map-free pseudo-simulation composite**, metric id
**`pseudosim_composite_PSS_recovery_progress@twosided_v2`** — ⛔ **NOT `ade_0_2s`**, and ⛔ **not the
`clamp_v1` composite every PSS number published before 2026-07-28 was computed under.** The two terms
are *different metrics* and the code says so in the value's own `name`; I never compare across them.

Estimator: **paired episode-cluster bootstrap** (`taniteval/ci.py`,
`paired_episode_cluster_bootstrap`, B = 2000, unit = **episode**). ⛔ `overlapping_holdout_se` is
refused.

MDE at α = 0.05 two-sided, 80 % power: `MDE = (z₀.₉₇₅ + z₀.₈₀) · SD_paired / √n = 2.8016 · SD/√n`.

**The one usable prior anchor on THIS metric** (INHERITED — `VTBAND_WIRING.md` §3,
`raw/gate_stop_proof_v4fs15k.json`; **not re-derived here**): a direction-verified injected
degradation on `flagship-v4-fromscratch`@15000 measured **−0.0647 [−0.0844, −0.0517]** at **n = 8
episodes** under the same estimator and the same metric. Half-width 0.01635 ⇒ **SD_paired ≈ 0.0236**.

⚠️ **That SD is a LOWER BOUND for my case and I register it as one.** It comes from two *scorings of
one checkpoint*, which are far more tightly paired than two *independently trained checkpoints*. I
therefore carry a 2× band:

| n (val episodes) | MDE at SD = 0.0236 (lower bound) | MDE at SD = 0.0472 (2× band) |
|---:|---:|---:|
| 8 (the trainer's own gate probe) | 0.0234 | 0.0467 |
| 40 (`--eval-episodes` default) | **0.0105** | **0.0209** |
| ⭐ **600 (the full matched val split)** | **0.0027** | **0.0054** |

**Is that decision-relevant?** The largest deliberate perturbation ever measured on this metric moved
it **−0.0647**. So an effect worth a GPU-week sits plausibly in **0.01–0.07**.
⇒ **at n = 600 the MDE (0.0027–0.0054) is 2–25× below the effect it must detect — the design CAN
decide. At n = 40 it is 0.0105–0.0209, i.e. comparable to or larger than the low end — a 40-episode
null here would be UNPOWERED, NOT A REFUTATION.**

⛔ **This is why the primary read is the 600-episode offline evaluation and NOT the trainer's
`--eval-episodes 40` or its 8-episode gate probes.** Both of those run anyway and are reported as
*diagnostics with their power stated*, never as the verdict. A validation whose MDE exceeds the effect
it must detect is class C13 — a guard that cannot fail — and this program has shipped several.

### 1.2 ⭐ THE ANTI-C13 MECHANISM: no null is admissible until the instrument is shown to have range

**Pre-registered, before any arm finished.** A "not separated" is reported as `NO DIFFERENCE` **only
if** the *same estimator on the same rows* separates a known-different pair. Two controls, both free:

* **PC-1 (primary control, on-axis):** for **each** arm, `ckpt@6000 − ckpt@1000` on the same 600
  episodes must be **separated-better**. If 6× more training cannot be seen, nothing can, and the
  verdict for that arm is **`INSTRUMENT-BLIND`**, not a null.
* **PC-2 (component range):** `discriminative_range` must admit ≥ 1 weighted component on every arm's
  rows. `composite()` already **refuses to emit** otherwise (`VacuousMetric`) — a refusal is a
  reported outcome here, never a silently dropped arm.

⚠️ Registered in advance: `comfort` is expected to be **inadmissible** (published as saturated
≥ 99.9 %, and v4's fan measured a literal constant), and `no_collision`/`ttc` are **structurally
unavailable** (no cuboids). So the composite is expected to reduce to **`ego_progress` +
`recovery`** — which is exactly the lateral/longitudinal split §4 decomposes on. **This is stated as
a prediction, not discovered afterwards.**

### 1.3 What each of the three outcomes licenses — all three committed in advance

| outcome | rule (primary, paired, n = 600, B = 2000) | what it licenses |
|---|---|---|
| ⭐ **WIDE BETTER** | `B_WIDE − A_OLD` separated **positive** | v5 trains wide. The PI's hypothesis is confirmed **on the planning primary**, not only on a probe. |
| **NO DIFFERENCE** | not separated **AND PC-1 passed on both arms** | v5 trains wide **on coverage/cost grounds, not performance grounds** — and the PI is told plainly that we are paying **1.694× a training step** (256×640) or **1.299×** (176×624) for **no measured planning gain at 6 k steps**, and that the resolution study's **+0.04246 AP** encoder-probe advantage **did not convert into planning** at this run length. |
| ⛔ **WIDE WORSE** | `B_WIDE − A_OLD` separated **negative** | **BLOCKS** the wide geometry as configured. This is live, not a formality: the resolution study already found the wide frame **separated-WORSE on ego yaw rate (−0.03546 R²)**, and the gate card calls ego-motion perception *"the real item."* |

**Secondary, pre-registered, same estimator:** `C_V5 − B_WIDE` = the **rig fix alone**;
`C_V5 − A_OLD` = **what v5 actually is**. ⚠️ A `WIDE WORSE` on B with a `NOT WORSE` on C would say the
penalty is the **rig-correlated masked region**, not the field — and would license v5 *at 176×624
only*. That branch is committed here, before the numbers.

⚠️ **Registered scope limit, stated now so it cannot be quietly dropped later:** these are **6 000-step
from-scratch arms with a 5×-compressed curriculum**, not 30 k runs. They measure the geometry's effect
**at that point on the training curve**. A result here does **not** entitle anyone to a 30 k
extrapolation — `CLAUDE.md` forbids extrapolating more than 2× beyond a fitted range, and 6 k → 30 k
is 5×.

---

## 2. PRE-REGISTRATION — fixed before any checkpoint existed

### 2.1 Primary, secondaries, estimator

* **Primary:** `pseudosim_composite_PSS_recovery_progress@twosided_v2`, per-window, on the **600-episode
  matched val split**, higher-is-better.
* **Diagnostics (never the verdict):** `ade_0_2s`, `ade_lat`, `ade_lon`, `wm_canary_ade_2s`,
  `miss_at_2m`, and the trainer's own 8-episode gate probes.
* **Estimator:** paired episode-cluster bootstrap, B = 2000, unit = episode, `taniteval/ci.py`.
  ⛔ `overlapping_holdout_se` refused. ⛔ No combination in quadrature.
* **Coverage is quoted with every number** (the `selected_frac` discipline, class C19): per arm, the
  finite-value fraction per component and the window/episode counts. A conditional win quoted without
  its coverage overstates itself.

⛔ **Nothing here is held to v1's 0.4271.** Verified in code (`rollout.py:170`,
`actions_source="expert_future"`): that is `wm_fidelity_ade_2s` — what the world model scores when
**handed the true actions** — not a planning bar.

### 2.2 What is matched, byte for byte

Identical across all three arms (from `--print-launch`, `raw/printlaunch_*.txt`):
`--from-scratch` · same anchors file · `--steps 6000` · **`--batch 8 --accum 8` (effective 64, the
trainer's own banner checks it; `--batch 16` OOMs at 44 GB)** · `--lr-head 1e-4 --lr-trunk 1e-4` ·
`--warmup 400` · `--workers 8` · `--phase-a-steps 400 --phase-b-steps 1600 --gate-step 1600`
(the 30 k curriculum compressed 5×, identically) · `--eval-every 500` · `--eval-episodes 40` ·
`--rollout-k 4` · `--lam-mult-floor 0.25` · `--labels v3` · `--lambda-plan sched` ·
`--strategic full` · `--long-horizon-k 50` · `--heldout-gate --heldout-every 1000
--heldout-episodes 8 --heldout-patience 2 --heldout-stride 8 --heldout-nboot 2000` ·
**`--heldout-goal dropped`** (the shipped default) · **panel-wide gate, never per-arm** ·
`--seed 0`.

**Only the corpus format and the four geometry flags differ.** `state_dim` stays 2048 at every frame —
the readout is a geometry firewall — so the predictor, policies and grounding heads are untouched and
the contrast really is the encoder's input.

⭐ The anchors file is the **same** for all three (`/workspace/experiments/flagship_v4_anchors_dense.pt`,
`[256, 20, 2]`, `method fps`, horizons 1..20). Anchors live in **metric trajectory space, not pixels**,
so sharing them is what makes the arms comparable — not a leak.

### 2.3 The bar and the n

* **n = 600 episodes** (the full matched val split), **B = 2000**.
* **Bar:** separation of the paired interval from 0 on the primary. ⚠️ **No fixed threshold is set on
  the composite's absolute value**, deliberately: this is a *contrast* between two arms trained here,
  not a comparison to a historical number, and every historical PSS value is a **different metric**
  (`clamp_v1`). Registering an absolute bar against a number computed under another term would be the
  exact silent-redefinition failure the code's `_progress_term_warning` exists to prevent.
* **MDE: 0.0027–0.0054** at n = 600 (§1.1). **Stated as a range because the SD anchor is a lower
  bound, not because the estimate is soft.**

### 2.4 ⚠️ The one confound I could not remove, its size, and its DIRECTION

**B_WIDE and C_V5 train on 2400 episodes; A_OLD trains on 2376.** That is **+1.00 % of episodes for
the wide arms** — the 24 clips that failed to decode into the raw epcache but succeeded in the PNG
build. §0.2 says why removing it would cost more than it buys.

Bounded, MEASURED: `[data] train windows=406099` for A_OLD, and 6000 steps × effective 64 =
**384 000 windows ⇒ 0.946 epoch**. Both arms see **less than one pass**, so the extra episodes enter
only as ~1 % of sampled windows, not as extra epochs.

⭐ **The direction is known and it is asymmetric, so I register how each verdict must be read:**

* a **WIDE BETTER** result is **partly confounded** — up to 1.00 % more training data could
  contribute, so the effect size is an **upper bound** on the pure geometry effect;
* a **WIDE WORSE** result is **CONSERVATIVE** — the confound pushes the other way, so a wide-loss is
  if anything *understated*;
* a **NO DIFFERENCE** result is **very slightly optimistic for wide**.

**The `C_V5 − B_WIDE` contrast is entirely free of this confound** (same corpus, same 2400 clips,
same cache — only the loader slice differs).

---

## 3. THE POD2 SYNC — evidence, because a `git log` is not proof

⚠️ pod2's `/workspace/TanitAD/stack` was at **`0f93b98`** with **52 modified tracked files** and had
**none** of the v5 fixes: `stack/tanitad/train/heldout_gate.py` **did not exist at all**, and
`--heldout-goal`, `--v2-train-cache`, `_render_bounds` and `register_v2_geometry_sibling` all
grepped **0**. A launch from there resurrects the crashing held-out gate.

**Sequence (`raw/pod2_sync.txt`):**

1. **Nothing destroyed:** pod2's drift snapshotted first →
   `pod2:/workspace/_pod2_drift_2026-07-27.patch` (28 009 lines) +
   `/workspace/_pod2_status_2026-07-27.txt` (88 lines).
2. `git -c core.autocrlf=false -c core.eol=lf archive HEAD -- stack taniteval` (662 entries) shipped
   and extracted over `/workspace/TanitAD/`.
   ⚠️ **A trap worth recording: the FIRST archive was CRLF-poisoned.** The dev box has
   `core.autocrlf=true` and no `.gitattributes`, so plain `git archive` rewrote every `.py`/`.sh` to
   CRLF. It was caught because the parity manifest's sha256 came back **`b3c0ecc0…` instead of
   `51b1792f…`** — a byte-level check on a file whose digest was published. **A `#!/bin/bash\r` is a
   `bad interpreter` failure hours later; Python would have swallowed it silently.** Redone with
   `-c core.autocrlf=false -c core.eol=lf`. *(Root-cause class: A SYNC THAT REPORTS SUCCESS AND SHIPS
   DIFFERENT BYTES.)*
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
| ⭐ **a real `--print-launch` on all three arms** | ✅ **`PREFLIGHT: OK`, exit 0, ×3** (`raw/printlaunch_*.txt`) |
| both v2 parity lines | ✅ `2400 clips e61a04553df5…` and `600 clips 0b176d2e5cb4…`, **matched to the committed manifest** |
| the raw-epcache parity line | ✅ `2376 episodes, uid sha256 9877bef64da3…, skip-hash f09e44db` |

⚠️ **One real gap found and NOT papered over:** the val epcache's guard prints *"count OK (600/600) —
**NO uid digest committed for this split**, so this is a COUNT-ONLY check."* I did not rely on it. The
membership proof in §0.1 is stronger than what the trainer could check: it compares the actual clip-id
sets of **both** corpora against `parity_val_clips.txt` and reproduces the committed digest
`0b176d2e5cb4…`.

---

## 4. THE RUNS

**Launched 2026-07-27T18:35:16Z on pod2 (A40, 0 MiB used, idle before launch).** Sequential — one
GPU, and B_WIDE alone peaks ~34.5 GiB of 44.4. Order is the **priority order**: the PI's contrast
(A vs B) completes first.

| | |
|---|---|
| chain script | `pod2:/workspace/smallval/chain.sh` (staged as `code/chain.sh`) |
| **chain PID** | **3693916** |
| **arm A trainer PID** | **3693920** |
| chain log | `pod2:/tmp/smallval_chain.log` |
| per-arm logs | `pod2:/tmp/smallval_{A_old,B_wide,C_v5}.log` |
| outputs | `pod2:/workspace/smallval/{A_old-256x256,B_wide-256x640,C_v5-176x624}/` |
| completion sentinel | `pod2:/workspace/smallval/CHAIN_DONE` |

⛔ **pod1 was NEVER CONTACTED.** ⛔ **pod3 was probed read-only** (inventory only) and **YouTube was
never touched**. ⛔ Nothing was killed; no `pkill -f` was ever issued. Disk judged by a **real `dd`**
(8 GiB at 504 MB/s, plus a 500 MiB probe at 388 MB/s) — **never `df`**.

<!-- RESULTS-ANCHOR -->

*Results, decomposition and verdict are filled in below as each arm lands; this document is banked
incrementally rather than held to the end.*

---

## 5. Provenance and evidence class of every number

| claim | class · tier | source |
|---|---|---|
| pod2 holds the full 256×256 parity epcache (2376 + 24 skips; 600 val) | **MEASURED (ours)** · DECISION-GRADE | `raw/pod2_corpus_inventory.txt`; the trainer's own parity banner in `raw/smallval_A_old.log` |
| pod3 also holds `physicalai-train-e438721ae894` / `physicalai-val-0c5f7dac3b11` | **MEASURED (ours)** · read-only probe | `raw/pod3_inventory.txt` |
| val split 600 vs 600, membership identical, digest reproduces the manifest | **MEASURED (ours)** · DECISION-GRADE | `raw/pod2_join_proof.txt` |
| the ordinal ↔ clip-id join; shared/excluded digests | **MEASURED (ours)** | `raw/pod2_join_proof.txt` |
| all three arms `PREFLIGHT: OK` on the synced pod | **MEASURED (ours)** | `raw/printlaunch_{A_old,B_wide,C_v5}.txt` |
| the pod2 sync verification table | **MEASURED (ours)** | `raw/pod2_sync.txt` |
| 256×640 full step 1.6944 s · 176×624 1.2988 s · run-level ratio 0.7665 · `--batch 16` OOMs | **INHERITED** (`V5_EVALUABLE.md` §8.1–8.2) — **not re-derived here** | that doc |
| the MDE's SD anchor: −0.0647 [−0.0844, −0.0517] at n = 8 | **INHERITED** (`VTBAND_WIRING.md` §3) — used only to *size* the design, never quoted as a result | that doc |
| ADE is the wrong axis (4.7× collisions; ρ = −0.36 vs Ego Progress 0.83) | **PUBLISHED / INHERITED** (`flagship-v5-retrain.PREP.md` §0) | that card |
| wide frame separated-WORSE on ego yaw rate (−0.03546 R²) | **INHERITED** (`PREP.md` §3 item 7) — the reason `WIDE WORSE` is a live branch | that card |
| v1's 0.4271 | ⛔ **NOT USED** — it is `wm_fidelity_ade_2s`, the world model handed the TRUE actions | — |
| pod1's state | **NOT PROBED** — never contacted | — |

🔒 **Gated-confidential handled, not assumed:** every file in `raw/` and `code/` carries **counts,
digests, paths and step numbers only**. No PhysicalAI-AV clip UUID is written to the repo — the clip
lists and the 24 excluded ids stay on the pods, and only their sha256 digests are quoted here.
