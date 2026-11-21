# Gate-1 clean closed-loop-aware fine-tune — P0 gate + P1 fixes + honest bound

**Date:** 2026-07-23 (Berlin) · **Host:** `tanitad-eval` (A40, FREE; `gpu_lock=gate1-clean`) ·
**Author:** gate1-clean subagent · **Status:** P0 MEASURED + banked; P1 fixes built + MEASURED;
**the promotable clean run is HELD on a measured bound — training GPU NOT spent on a run P0 proves would memorize.**

**Evidence class on every number** (CLAUDE.md §Operating standard). `MEASURED (ours + path)` unless
tagged `INHERITED`/`PUBLISHED`/`ESTIMATED`.

---

## 0. TL;DR — the pre-registered "inventory too small" branch fired, with two independent measured causes

The mission pre-registered two outcomes: **(promote)** held-out junction off-road drops without the
high-deviation side-effect, or **(bound)** *"memorizes / high-deviation persists / inventory too small →
report the specific bound + what's needed."* **P0 lands squarely in the bound branch, and I did not burn
training GPU pretending otherwise** (the mission's own P0 rule: *"Bank the inventory + protocol before
spending training GPU… if the real-footage junction inventory is ~15 like the prototype, a fine-tune will
ALSO memorize — SAY SO PLAINLY"*).

Two independent, MEASURED reasons a naive clean rerun cannot yield a promotable number:

1. **METRIC MISMATCH (binding).** The prototype's win is defined in **off-road-rate / at-fault-collision /
   pass-rate**, which require a map + reactive agents = the NuRec/AlpaSim renderer (~3.2× OOD — the very
   confound the clean run exists to escape). The low-OOD real-footage source (`lowood_closedloop.py`) is a
   **map-free, agent-free drift/stability loop** that structurally emits **only** closed-loop ADE +
   on-policy deviation + an OOD ratio. **"Low-OOD" and "junction off-road/collision" are mutually exclusive
   with the instruments that exist.** You can measure lane-keeping/longitudinal drift cleanly, or off-road/
   collisions confounded — not both.
2. **INVENTORY IN THE MEMORIZATION REGIME (measured two ways).** The real 40-ep clean val holds only
   **~13–22 distinct junction episodes** (§3). A proper train/held-out-scene split leaves ~11–15 training
   junction episodes — the **same n≈15 scale that memorized in the prototype**. A leave-3-out on the
   prototype's own 15 scenes (§4, MEASURED here) shows the decoder fine-tune fits the training scenes
   (train recovery-L1 5.06 → 0.41) but **does not improve held-out recovery (5.06 → 5.06, Δ ≈ 0)** while
   rewriting the held-out plan by **7.6 m** — memorization + the high-deviation side-effect, on held-out data.

**Bank order honored:** P0 inventory + protocol first; P1 fixes built and measured; P2/P3 GPU **held**.
What unblocks a real promotable clean run is in §6 — more distinct real junction scenes **and** a low-OOD
source that emits a lane-departure metric (a corridor-departure proxy is the cheap path; collisions
require reactive agents and cannot be de-confounded without AlpaSim).

---

## 1. Prototype `a7c1eb9c` — VERIFIED (gating condition (a) is genuinely GREEN, and both flaws are real)

Condition (a) was `PENDING` in the live `LOOP_STATE.md`; my brief asserted it green. I verified against the
**terminal marker + raw JSON**, not the prose (RETRACTION_LOG C1/C4; the 07-22 "launched ≠ completed"
entry). Prototype terminal markers present: `/workspace/gate1_MASTER_DONE` (`NROLL=15 NPRED=675`),
`/workspace/gate1_reeval_CHAIN_DONE`. Raw: `/workspace/gate1_reeval_scores.json`.

| metric (15 NuRec junction scenes, ~3.2× OOD) | base (REF-C-base) | ft800 | Δ |
|---|---|---|---|
| off-road (count / rate) | **11 / 0.733** | **7 / 0.467** | **−4** |
| at-fault collision (count / rate) | **5 / 0.333** | **1 / 0.067** | −4 |
| pass (count / rate) | **3 / 0.20** | **8 / 0.533** | **+5** |
| mean plan_deviation | **0.408** | **2.69** | **+2.28 (6.6×)** |
| mean dist_to_gt (m) | 14.645 | 2.046 | −12.6 |

So the brief's *"11→7, 5→1, 3→8"* reproduces **exactly** (MEASURED). **But** the headline is
**in-sample**: `gate1_ft_result.json` → `holdout_scenes: []`, `n_holdout: 0` — the ft800 re-eval scores the
**same 15 scenes it trained on**. And the two measured flaws are visible in the same JSON:

- **High-deviation side-effect:** mean plan_dev **0.408 → 2.69**; **3 scenes that PASSED in base**
  (`8b04d54e`, `adb72a39`, `fd3a49fa`) went **newly off-road** (`scenes_newly_offroad`). The recovery
  objective buys 7 recoveries but breaks 3 previously-fine scenes — the same "recovery ⇒ over-aggressive
  planner" mechanism retracted for flagship v1 (RETRACTION_LOG 07-23 C7: v1 tactical is a high-deviation
  planner, plan_dev 1.12 vs REF-C 0.34 → **offroad, not collision**).
- **In-sample only:** no held-out split was run for the headline → memorization untested by the prototype.

**Code-level confirmation** (`/workspace/gate1_finetune.py`): the loss is exactly `loss_traj + loss_cls`
(**no deviation regularizer**), trained on **every** on-policy recovery label (**no CAT-K filter**),
decoder-only (8.63 M trainable), frozen-forward cached. That is precisely why plan_dev explodes and why
half the labels are catastrophic (§5).

---

## 2. The binding blocker: the low-OOD source cannot emit the prototype's win metric (3 probes, code-level)

The clean run must use the real-footage low-OOD source (`ae72a9e1`), **not** NuRec. I read what that source
actually is — design doc, hardening report, **and the harness code** (RETRACTION_LOG C2: absence/claims need
≥2 probes + the tool that owns the fact):

- **Design** (`LOWER_OOD_CLOSEDLOOP_DESIGN.md` §4): *"Real-footage log-replay scores route-following /
  lane-keeping / longitudinal control, **not** reactive collision avoidance… the recorded other agents don't
  react to the ego."*
- **Hardening** (`LOWOOD_HARDENING_REPORT.md` §2.3): *"Drift/stability loop, **not safety** — no map/agents,
  so **no collision/PDM**."*
- **Code** (`lowood_closedloop.py`, MEASURED by reading it): the harness emits **only** `closed_ade2s`,
  `peak_lat_m`, `peak_yaw_deg`, `ood_mean/peak_ratio`, stratified into *junction* (`|Δheading|≥10°`
  turn-windows) vs *longitudinal*. There is **no lane polygon, no boundary, no other agent, no PDM**. It
  also runs the **flagship operative rollout** (`world.encode_window → strategic_policy → tactical_policy`),
  **not** REF-C's anchored-diffusion decoder.

**Consequence.** "Held-out junction off-road / collisions / pass-rate" (mission P3) is **not a function the
clean source computes**. The clean source's honest analog of off-road is a **corridor-departure proxy** from
the on-policy signed lateral offset `dlat` it already tracks (threshold |XTE| against a lane half-width) —
lane-keeping, not map-off-road; and **collision-avoidance is unmeasurable** without reactive agents. So a
"clean Gate-1 number" in the prototype's metric does not exist on this source; a clean **lane-keeping/drift**
number does.

*(REF-C wiring into the low-OOD harness — the hardening report's top open gap, blocked on pod1 because the
REF-C ckpt was unreachable there — IS solvable on this pod: `/root/models/refc-base-30k/ckpt.pt` is here,
md5 `8f10d6f934f4199e11ddc7352e074939`. That unblocks a clean REF-C **lane-keeping** read, not off-road.)*

---

## 3. P0 scene inventory — real-footage junctions are in the n≈15 regime (MEASURED)

`gate1_junction_inventory.py` over the full 40-ep clean val `physicalai-val-0c5f7dac3b11`
(`/root/valdata/…`, 881 stride-8 windows), using the harness's own `net_heading_change_deg`. Raw:
`/workspace/gate1_junction_inventory.json`.

| junction cut | junction **episodes** | junction **windows** | eps with ≥5 junc-win |
|---|---|---|---|
| `|Δheading| ≥ 10°` | **22 / 40** | 182 | 17 |
| `≥ 20°` (a real turn) | **16 / 40** | 122 | 13 |
| `≥ 30°` | **13 / 40** | 70 | 8 |

The distinct junction-**episode** count is **13–22**, right on the prototype's n=15. Full per-episode counts
in the JSON (top episodes: `ep_00037` 18, `ep_00011` 17, `ep_00023` 17, `ep_00009` 16 junction windows at
≥10°; 18 of 40 episodes are essentially straight, maxHD < 9°).

**Protocol design (episode-disjoint split, pre-registered in `PRE_REGISTRATION.md`).** At the ≥20° cut:
16 junction episodes → a scene-disjoint **11 train / 5 held-out** split (or 5-fold leave-≈3-out). This is a
*correct* protocol — but it leaves only **~11 distinct training junction episodes**, i.e. the prototype's
memorizing scale. **The protocol is sound; the inventory is the problem.**

---

## 4. Memorization + high-deviation, MEASURED on the prototype's own 15 scenes (leave-3-out, 5 folds)

To convert the brief's **INHERITED** memorization claim (*"leave-3-out: 4.65→4.15 then degrades"* — no
artifact existed on the pod) into a **MEASURED** result, I ran a 5-fold leave-3-out over the prototype's 15
scenes, reusing its exact frozen-forward + decoder objective (`gate1_clean_loo.py`; frozen forward cached
once, decoder re-init to base per fold). This measures the **lever's generalization at n=15**, which
transfers to the real-footage scale (same decoder, same objective, ~same #distinct scenes). Raw:
`/workspace/gate1_clean_loo.json`.

| variant (mean over 5 leave-3-out folds) | held-out recovery-L1  base → final | train-L1 (in-sample) | **held-out plan-shift-from-base** |
|---|---|---|---|
| **A_naive** (prototype recipe: no filter, no regularizer) | **5.06 → 5.06  (Δ ≈ 0)** | 5.06 → **0.41** | **7.58 m** |
| **B_catk** (CAT-K label filter) | 5.06 → 4.67 (marginal) | → 0.08 | 2.88 m (**−62 %**) |
| **C_catk_dev** (CAT-K + λ_dev=1.0 trust region) | 5.06 → 4.86 (marginal) | → 0.94 | **1.49 m (−80 %)** |

*(base held-out L1 measured on the pristine decoder each fold; base = in-sample 5.0596 because the pristine
REF-C base never trained on these NuRec recovery labels, so all 15 scenes are equally novel to it. Per-fold
detail in `gate1_clean_loo.json`; a first pass had a carryover bug in the base baseline, fixed and re-run.)*

**Read — two clean, separable findings:**
1. **Memorization is real and the fixes do NOT cure it.** Under **all three** variants, **train** recovery-L1
   collapses (5.06 → 0.08–0.94, near-perfect in-sample fit) while **held-out** recovery-L1 stays **~5**
   (Δ ≈ 0 naive; ≤ 8 % with CAT-K). The lever does not generalize to held-out junctions at n=15 — this
   **measures** the brief's inherited *"held-out barely moves (4.65→4.15)"* (B_catk lands held-out **4.67**).
   Memorization is **data-quantity-bound**, not a label- or objective-quality problem.
2. **The high-deviation side-effect IS the fixes' domain.** The naive FT leaves the held-out plan **7.58 m**
   from the well-behaved base — the plan is rewritten by metres on scenes it never saw (the retracted v1
   high-deviation mechanism, on held-out data). **CAT-K cuts it to 2.88 m (−62 %); adding λ_dev cuts it to
   1.49 m (−80 %).** The fixes work exactly where designed.

**Conclusion:** the two fixes are **necessary** (they remove the high-deviation side-effect) but **not
sufficient** (memorization needs more distinct scenes). At n≈15 the clean run is pre-committed to the BOUND
outcome — which is why the training GPU is held.

---

## 5. P1 — the two fixes, built and measured (they tame deviation; they do NOT cure memorization)

Both fixes are implemented in `gate1_clean_loo.py` (variants **B_catk**, **C_catk_dev**) and specified as a
drop-in patch to `gate1_finetune.py` in `catk_road_filter_and_dev_regularizer.py`.

### (a) CAT-K / recovery-feasibility target filtering
**Method & citation.** CAT-K = *Closest-Among-Top-K* closed-loop SFT (Zhang, Karkus et al., NVIDIA,
*"Closed-Loop Supervised Fine-Tuning of Tokenized Traffic Models,"* CVPR 2025): keep closed-loop supervision
**anchored to the expert manifold** so recovery targets stay feasible, rather than supervising from
catastrophic states whose "recovery" points sharply backward. *(The mission's "RoAD" pairing is the same
recovery-augmentation-with-filtering principle; the load-bearing, verified citation is CAT-K + the DAgger
covariate-shift frame, Ross et al. 2011. I did not find a canonical "RoAD" paper to attribute exactly —
flagged rather than fabricated, RETRACTION_LOG C4.)*

**Implementation** (`catk_keep_mask`): drop a recovery label if the target leaves the **P1-measured low-OOD
envelope** (|left| > 3.0 m over the 2 s path, or heading-correction > 12°) **or points backward**
(`fwd_end ≤ 0`). **MEASURED:** this drops **328 / 675 (49 %)** of the prototype's labels — **147 backward**,
281 beyond-lateral, 285 beyond-yaw. Half of the recovery signal comes from catastrophic off-manifold states
— exactly what the mission flagged as driving the high-deviation trade.

### (b) Deviation / stability regularizer (base-plan trust region)
**Method.** A trust-region / behaviour-cloning-style penalty `λ_dev · ‖FT_traj − base_traj‖₁`, keeping the
fine-tuned plan close to the well-behaved base planner (plan_dev-0.34 family) so recovery cannot be bought
with an aggressive swerve. Analogous to KL-to-reference regularization in offline RL. Default `λ_dev=1.0`.

**Measured effect (variants B/C vs A over the 5 leave-3-out folds; §4 table):** held-out plan-shift-from-base
**A 7.58 m → B_catk 2.88 m (−62 %) → C_catk_dev 1.49 m (−80 %)** — the deviation side-effect is progressively
removed. Held-out recovery-L1 stays **~5** throughout (5.06 → 5.06 / 4.67 / 4.86) — the fixes **do not
restore generalization**, because memorization at n=15 is a **data-quantity** problem, not a label-quality
one. The fixes are **necessary** (they remove the retracted high-deviation mechanism, RETRACTION_LOG C7) and
**not sufficient** without more distinct scenes.

---

## 6. What a promotable clean Gate-1 actually needs (priority-ordered)

1. **More distinct real junction scenes** — the 40-ep val's ~11–16 trainable junction episodes is the
   memorizing scale. Need many more (order 100+ distinct junctions): harvest junction episodes from the
   **train corpus** held-out-safe (parity firewall: never re-select the canonical train episodes into eval),
   or additional real junction footage (L2D / comma has turns). Below that, any decoder recovery FT
   memorizes — MEASURED, not asserted.
2. **A low-OOD source that emits a lane-departure metric.** Cheapest: add a **corridor-departure rate** to
   `lowood_closedloop.py` (threshold on-policy |XTE| against a lane half-width ~1.75 m — the signal is
   already computed as `dlat`). This gives a clean lane-keeping analog of off-road. **Collisions require
   reactive agents** → AlpaSim (with its ~3.2× OOD caveat) or the design's hybrid (c); they cannot be
   de-confounded on static log-replay.
3. **Wire REF-C into the low-OOD harness** (~1 h; ckpt is on this pod) for a clean REF-C lane-keeping
   baseline + the FT delta on held-out real junctions — the clean **lane-keeping** Gate-1 read, explicitly
   not the off-road/collision one.
4. **Ship the two fixes regardless** — they remove the high-deviation side-effect (the retracted v1
   mechanism) and cut half the label noise; they belong in any future recovery FT.

**Recommendation to Sayed (notify + veto-window, per the standing auth):** **HOLD the Gate-1 training GPU.**
Condition (a) is genuinely green and the mechanism is real on NuRec, but the *clean* promotable run is
blocked by a measured metric-mismatch **and** a measured n≈15 inventory. Spending GPU-hours on a decoder FT
now reproduces the prototype's two flaws on a held-out set (MEASURED §4). The cheap unblocks are §6.1–6.2.

---

## 7. Deliverable manifest

| artifact | where | what | evidence |
|---|---|---|---|
| `GATE1_CLEAN_RUN_P0_FINDINGS.md` | repo (staged) | this report | — |
| `PRE_REGISTRATION.md` | repo (staged) | protocol + split + both fixes + committed outcomes | — |
| `gate1_junction_inventory.py` / `.json` | repo (staged) · pod `/workspace/` | ⭐ P0 real junction inventory | MEASURED |
| `gate1_clean_loo.py` | repo (staged) · pod `/workspace/` | leave-3-out + P1 fixes (A/B/C variants) | MEASURED |
| `gate1_clean_loo.json` | repo (staged) · pod `/workspace/` | ⭐ LOO memorization + fix effects | MEASURED |
| `catk_road_filter_and_dev_regularizer.py` | repo (staged) | the two fixes as a drop-in patch to `gate1_finetune.py` | code |
| `proto_gate1_finetune.py` / `proto_gate1_extract.py` | repo (staged) | prototype scripts read for provenance | INHERITED |
| `gate1_reeval_scores.json` (copy) | repo (staged) · pod `/workspace/` | prototype base-vs-ft800 (verification of (a)) | MEASURED |

**Pod-side inputs (pre-existing):** REF-C base `tanitad-eval:/root/models/refc-base-30k/ckpt.pt`; prototype
data `/workspace/gate1_ft_data/` (15 bundles) + `/workspace/gate1_junc/gate1_summary.json`; clean val
`/root/valdata/physicalai-val-0c5f7dac3b11` (40 ep). **Not modified:** no `stack/` code touched (harnesses
live in `incoming/`), `pytest` unaffected. **Pod state:** `gpu_lock` released, GPU idle, no deletions.
**Staging:** `git add`-ed, **not committed / not pushed**; the index carries other agents' concurrent work →
commit with an explicit pathspec (CLAUDE.md §Git hygiene).
</content>
</invoke>
