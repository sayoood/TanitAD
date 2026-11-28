> **⚠️ SUPERSEDED / STOOD DOWN 2026-07-23 — see `PARTIAL.md`.** The coordinator halted this run at
> **step 300 of 10,000** (Sayed went straight to v4.2, joint, no frozen part; pod3 repurposed for
> own-encoder Branch B). No held-out MODE B number was produced; the STARVED-vs-BAD-BY-DESIGN verdict is
> **UNRESOLVED**. The setup below was fully provisioned and O-03-validated before the stand-down.

# planner_on_frozen_wm — the STARVED-vs-BAD-BY-DESIGN discriminator (exp #2)

**Date:** 2026-07-23 (Europe/Berlin, UTC+2). **Host:** pod3 `tanitad-pod3` (NVIDIA A40), GPU lock
`planner-frozen-wm` HELD (pid-bound to the trainer). **Author:** planner-frozen-wm subagent.
**Status at this writing:** provisioned + O-03 validated + **training LAUNCHED and running**; MODE B
result pending at step 10000 (this file is the incremental bank; the verdict lands in
`PLANNER_FROZEN_WM_RESULT.md`).

Evidence classes per CLAUDE.md are inline: **MEASURED** (ours + artifact) · **PUBLISHED/INHERITED**
(cited) · **ESTIMATED** · **HYPOTHESIS**.

---

## 1. The question this settles (pre-registered, both outcomes committed in advance)

Source of truth: `Architecture & Inference/Research/2026-07-23-planner-is-the-bottleneck.md` §3/§4.1
(exp #2 `planner_on_frozen_wm`). v4.1's operative planner underperforms — its **real held-out
4-waypoint `ade_0_2s` = 0.8522** at step 10k (MEASURED, `…/2026-07-23-v4-eval-harness/STATUS.md`),
far above v1's 0.4271. The open question: is the planner **STARVED** (the λ_plan canary controller
clamped `lam_mult` to ~1.5e-5 by step ~2000, so the planner got almost no gradient) or **BAD BY
DESIGN**?

Discriminator: train the v4 anchored-diffusion planner head **fully** on a **FROZEN, known-healthy WM**
(the v1 trunk, `flagship4b-speedjerk-30k`), with the planner seam fully open from step 0 and no
controller in the loop. Read the same gate card at step 10000.

### Pre-registered reading (committed BEFORE the result)
- **STARVED** ⇒ frozen-WM planner **`ade_0_2s` ≤ ~0.50** AND **`oracle_in_fan` ≤ 0.30** AND
  **`miss_at_2m` ≤ 0.10** (approaches v1's 0.4271 / the ~0.30 oracle). ⇒ v4.1's problem is the
  controller starving the planner ⇒ fix = cap-and-hold controller + v4.2, **not** a planner redesign.
- **BAD BY DESIGN** ⇒ `ade_0_2s` **≥ ~0.60**, OR `oracle_in_fan` > 0.30, OR `miss_at_2m` > 0.10, **even
  decoupled on a perfect frozen WM** ⇒ the anchored-diffusion planner isn't the win on our WM latent;
  the design needs rethinking. Stop tuning the controller.

Reference points (all held-out `ade_0_2s`, 4wp, episode-cluster bootstrap): **v1 = 0.4271**
(0.4253 on this run's held-out val, see §3); **v4.1@10k (STARVED) = 0.8522**; **oracle target ~0.30**
(v1.5-ab 0.3073).

---

## 2. Design decision — `--lr-trunk 0` (frozen trunk), the cleanest instance of the discriminator

The mission brief says `--lr-trunk 0` ("FROZEN, known-healthy WM"); the pre-registration §4.1 prose
says `--lambda-plan 0` (trunk keeps training under WM losses, planner→trunk gradient off). **These are
two valid instances of the same discriminator** (both train the planner head at full strength on a
healthy WM); they differ only in whether the WM trunk keeps moving.

**Verified mechanism** (`stack/tanitad/models/flagship_v4.py:30-35,211`): `λ_plan` enters the head via
`grad_scale(states, lambda_plan)` — **identity in forward, scales the planner→trunk gradient in
backward**. The planner loss is added to the total **unscaled** (`train_flagship_v4.py:138`), so the
planner head ALWAYS trains at full strength via `lr_head`; `λ_plan` only gates the trunk gradient.
Therefore with **`lr_trunk 0` the trunk is frozen and `λ_plan` is moot** — the head sees an identical
forward and trains fully. This is the **cleaner** discriminator (zero WM-drift confound; the WM stays
exactly at v1's known-healthy state) and matches the mission title + `V4_FLAGSHIP_DESIGN.md` §11's own
"planner-over-frozen-v1" fallback framing. **Chosen: `--lr-trunk 0 --lambda-plan 1`.** Documented here
so the launching agent can see exactly what ran; the verdict reading is identical under either instance.

---

## 3. Provisioning (pod3 was a REF-A/C pod, NOT "clean & ready" — full build-out was required)

The brief assumed pod3 was ready for a flagship-v4 config. It was not: no v4 code, no v1 trunk, no v4
anchors, and **no disjoint flagship val split**. All prerequisites were obtainable; each step is
MEASURED with its artifact.

| step | what | result |
|---|---|---|
| code | deployed `stack/{tanitad,scripts}` + current `taniteval` → `pod3:/workspace/v4run` | imports OK (incl. `taniteval.driving`, missing from pod3's installed copy) |
| trunk | pulled `Sayood/tanitad-flagship-4b-speedjerk/ckpt.pt` (3.309 GB) from HF | **verified SPEED arm**: `predictor.act_emb.0.weight` = (768,**3**), step 29999, has `grounding` — NOT the no-speed ablation (CLAUDE.md inversion trap avoided) |
| anchors | `build_refc_anchors.py --n-anchors 256 --horizons 1..20 --seed 0` over the parity train | `flagship_v4_anchors_dense.pt` shape **[256,20,2]**, FPS, pool 200k (canonical recipe; MEASURED). Rebuilt (not v4.1's exact file — that lives on pod2/eval, off-limits); `oracle_in_fan` will confirm vocabulary quality |
| **val** | **the existing pod3 val `physicalai-val-f1b378f295ae` LEAKS 62/79 (78%) episodes into the parity train** (MEASURED) — the I-JEPA-class leak CLAUDE.md forbids. **Rebuilt a disjoint held-out val**: 44 clips whose `episode_id ∉` the 2342 parity-train ids, clustered in 4 camera chunks [170,184,919,1864] (minimal HF re-fetch, ~546 MB extracted), built via the same `build_episodes_cached` path as the train cache | `valcache/physicalai-val-heldout-79d4e3d2d4c6`, **44 episodes** |

### O-03 harness validation (MEASURED — `v1-canary-heldout-pod3.json`)
MODE A (v1 canary, true-action rollout) on the rebuilt held-out val:
**`canary_ade_2s` = 0.42535** (879 windows), **Δ vs v1's full-set 0.4271 = −0.0018 m** →
**`HARNESS_VALIDATED: true`**. This single result validates three things at once: the harness plumbing
on pod3, the v1 trunk, AND that the rebuilt held-out val is **calibration-identical** to the canonical
val (so the frozen-WM `ade_0_2s` is directly comparable to v1's 0.4271 and v4.1's 0.8522). 879 windows
≈ the canonical 881.

---

## 4. The run (LAUNCHED, MEASURED-so-far)

Command = v4.1's canonical launch (LOOP_STATE.md:234) with **`--lr-trunk 0 --lambda-plan 1`** and pod3
paths; everything else identical (v3 labels, dense operative anchored diffusion, factorised LAT×LON×DIST,
`--strategic full`, `--seed 0`, batch 16, `lr_head 1e-4`, phases 2000/8000, `--steps 30000
--gate-step 10000`, `--save-every 1000`). Trunk warm-started from v1; anchors loaded. Out dir
`pod3:/workspace/v4run/flagship-v4-frozenwm`.

Startup confirmations (MEASURED, `frozenwm.log`):
- **step-0 canary_baseline = 0.42535** — identical to MODE A → the trunk is frozen at v1's known-healthy
  level; the controller baseline is v1's 0.4253.
- **`lr_trunk` = 0.0** throughout (trunk frozen ✓); **`lam_mult` = 1.0 / `lambda_plan` = 1.0** (planner
  seam fully open, controller inert as expected for a frozen trunk ✓).
- Planner head training from scratch and **descending fast**: in-loop dense-20 `plan_ade`
  69.7 → 62.4 → 30.5 → 17.6 over steps 0→150 (this is the in-loop dense-20 diagnostic, **NOT** the gate
  metric — C1; the gate metric is the held-out 4wp `ade_0_2s` from MODE B at step 10000).
- **Throughput ≈ 1.9 s/step** (MEASURED, steps 50→150) → **ETA to step 10000 ≈ 5.3 h** from launch
  (faster than the ~0.5 A40-day estimate; the frozen trunk skips its optimizer update).

---

## 5. What lands next
1. Monitor to step 10000 (deleting the interim `ckpt_step5000.pt` milestone to hold disk).
2. Run MODE B (`run_modeB.sh`) on `ckpt_step10000.pt` → **held-out 4wp `ade_0_2s`** (episode-cluster
   bootstrap) + `oracle_in_fan` + `miss_at_2m` + `wm_canary_ade_2s`.
3. Apply the §1 reading; write `PLANNER_FROZEN_WM_RESULT.md` with the number + verdict + evidence class.
4. Release the lock; leave pod3 clean (GPU 0 MiB); NO deletions of shared data.

## Deliverable manifest (so far — all STAGED, not committed/pushed)
| artifact | where |
|---|---|
| this status | `repo:…/incoming/2026-07-23-planner-frozen-wm/STATUS.md` |
| O-03 MODE A proof | `repo:…/2026-07-23-planner-frozen-wm/v1-canary-heldout-pod3.json` (+ `pod3:/workspace/v4run/results/`) |
| exact run config | `repo:…/2026-07-23-planner-frozen-wm/frozenwm_run_config.json` (pulled from the live out_dir) |
| held-out val selection | `repo:…/2026-07-23-planner-frozen-wm/val_pick.json` (44 clips, 4 chunks, eids) |
| provisioned env + run | `pod3:/workspace/v4run/` (code, trunk, anchors, valcache, `flagship-v4-frozenwm/`) — isolated from pod3's `/workspace/TanitAD` REF-C checkout |
| checkpoint | `pod3:/workspace/v4run/flagship-v4-frozenwm/ckpt_step10000.pt` (pending; binary, stays on pod per convention) |

## Escalations
- **The brief's "pod3 is clean & ready" was inaccurate** — a full flagship-v4 provisioning (code, trunk,
  anchors, and a *disjoint* val rebuild because the resident val leaks 78% into train) was required.
  Nothing was blocked; flagged so future briefs don't assume a REF pod is flagship-ready.
- **Anchors were rebuilt from the canonical recipe** (v4.1's exact `flagship_v4_anchors_dense.pt` is on
  pod2/eval, off-limits). `oracle_in_fan` at MODE B is the check that the vocabulary is equivalent.
