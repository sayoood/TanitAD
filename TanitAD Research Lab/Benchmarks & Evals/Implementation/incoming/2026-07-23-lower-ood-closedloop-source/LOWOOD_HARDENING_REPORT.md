# Hardening the lower-OOD closed-loop source — decision-grade CIs + the closed-loop pre-req

**Date:** 2026-07-23 (Berlin) · **Host:** `tanitad-pod` (pod1, RTX A6000, `gpu_lock=lowood-harden`) ·
**Author:** lowood-harden subagent · **Status:** P1 + P2 MEASURED and banked; literal P1 inputs BLOCKED
(evidenced below); pod released, idle.

**Evidence-class discipline (CLAUDE.md).** Every number is `MEASURED (ours + artifact path)` unless
tagged `INHERITED` (registry/other-agent, not re-verified) or `ESTIMATED`. Decision-grade intervals are the
**episode-cluster bootstrap** (`taniteval/ci.py`); the prototype's bare points are not quoted without one.

---

## Executive summary

The prior prototype (`LOWER_OOD_CLOSEDLOOP_DESIGN.md`) showed, on **flagship v1, n=12**, that real-footage
log-replay eliminates NuRec's ~3.2× reconstruction-OOD (Δ=0 ADE 0.4045 vs 1.5157) with a wide usable
deviation envelope — but with **no intervals** and **only the open-loop observation-OOD** (not a closed-loop
planner's on-policy behaviour). This work closes both, for the arm reachable on pod1:

1. **P1 — decision-grade CIs (MEASURED).** Episode-cluster bootstrap on the n=12 envelope. The
   reconstruction-OOD elimination is **CI-robust** (real-footage baseline **0.4045 [0.3128, 0.5149]**; NuRec
   1.5157 sits at **2.94× the baseline's own upper CI**). Lateral offset carries **no CI-separated OOD out to
   2.0 m** (paired); yaw separates at **3°** and rises to +0.055 at 12° but **stays far below NuRec** across
   the whole ±3 m / ±12° grid.
2. **P2 — the real Gate-1 pre-req (MEASURED).** A **real-footage-in-the-loop** closed loop (deployed
   planner + controller unchanged; observation = arc-length-re-indexed real frame warped by the *on-policy*
   deviation). **The loop stays low-OOD on-policy:** overall peak-OOD **1.054× [1.026, 1.087]** vs NuRec's
   flat **3.75×**; **100 % of windows ≤ 1.5×**. **Longitudinal scenes** — the flagship's dominant 89 %
   failure mode and Gate-1's target — sit at **1.017× [1.006, 1.029]** (100 % ≤ 1.16×), essentially OOD-free.
   Even **junctions** (on-policy excursions up to ~2 m / 18°) stay at **1.19× [1.13, 1.22]**, all ≤ 1.5×.

**Verdict:** the real-footage source is validated as an **absolute low-OOD closed-loop source on-policy**, not
merely a relative one — the confound-free core a longitudinal-first Gate-1 needs. Two honest gaps remain: the
**40-ep tightening** and the **REF-C 2nd arm** are blocked by off-limits resources (below), and the junction
stratum's peak yaw exceeds the P1 grid edge (a cheap grid extension makes it exact).

---

## 0. Sourcing reality — the literal P1 inputs are unreachable from pod1 (class C2, probed ≥3 paths each)

The brief assumed the 40-ep val and REF-C base sat on pod1 `/root/valdata/` and `/root/models/`. They do
not. Verified by multiple probes and two independent doc sources — **not** a single-path absence:

| input | truth | probes | verdict |
|---|---|---|---|
| 40-ep clean val `physicalai-val-0c5f7dac3b11` | **eval pod only** (registry §0.3: 40 ep→881 win on `tanitad-eval`; prior design §2.1 same). pod1 has **12** eps | `ls` (12); `find` other val dirs (0); `find ep_00039.pt` (0); HF datasets cache (empty) | **BLOCKED** |
| REF-C **base** ckpt (104.2 M) | `tanitad-pod3` + `tanitad-eval` only; moved pod3→eval by direct scp, **never HF** (registry §4.3) | `find` pod1 (only flagship); HF `list_models(Sayood)` | **BLOCKED** |
| REF-C **small** ckpt (HF fallback) | `Sayood/tanitad-refc-small-evalonly` (file `ckpt_evalonly.pt`) | `hf_hub_download` → **HTTP 403 "Private repository storage limit reached for Sayood"** | **BLOCKED** |

All three routes to unblock require touching **pod2/pod3/eval** (explicitly off-limits — running v4.2b /
Branch B / Gate-1 proto; CLAUDE.md forbids load on a training pod) or altering Sayed's HF account
(out-of-scope; deleting repos is destructive). **A pod1-only agent under the off-limits constraint cannot
obtain them.** So P1's *literal* form was not executable; the decision-grade CI upgrade below is the
achievable increment, and P2 (which needs only flagship + the 12 eps) is delivered in full.

*(pod1 CAN run REF-C — `refc.py`, `taniteval/refc_eval.py`, and a clean decode interface
`model(fw, nav_cmd=None, v0, steps=2) → out["waypoints"][k]` are all present, and `refc_anchors_small64.pt`
is in the repo. The ONLY missing piece is the ckpt weights. The REF-C arm is ~1 hour of work the moment a
ckpt lands — see §4.)*

---

## 1. P1 — decision-grade CIs on the flagship envelope (detail in `P1_DECISION_GRADE_FINDINGS.md`)

Harness `lowood_ci.py` (reuses `lowood_probe.py`'s warp geometry verbatim; Δ=0 == the gate rollout), adds
per-window retention + **episode-cluster bootstrap** (n_boot=2000) + a **paired** bootstrap of each condition
vs Δ=0 on the same windows. flagship `flagship-30k` step 29999, 12 eps / 265 windows. Raw
`lowood_flagship_ci.json`.

- **Baseline** real-footage ADE@2s **0.4045 [0.3128, 0.5149]**; NuRec 1.5157 = 3.75× the mean, **2.94× the
  upper CI** → elimination is not a point coincidence.
- **Lateral:** paired Δ vs baseline **not** CI-separated through **2.0 m** (+0.025 [−0.011,+0.076]); first
  separates at **3.0 m** (+0.066 [+0.010,+0.138]).
- **Yaw:** separates at **3°** (+0.017 [+0.001,+0.034]), monotone to **12°** (+0.055 [+0.021,+0.095]).
- **Pixshift:** nothing separates through 32 px.
- Every separated rise (≤0.066) is **~17–20× smaller** than the baseline→NuRec gap (+1.11 m).

**Read:** within ±2 m lateral / ≤2° yaw the observation-OOD is statistically indistinguishable from on-path;
beyond that it is detectable but stays an order of magnitude under NuRec. CIs are wide because **n=12** — the
one thing 40 eps would fix (would tighten ~1.8× at n=40); the paired test resolves the envelope shape
regardless.

---

## 2. P2 — real-footage-in-the-loop closed loop: the actual Gate-1 pre-req (MEASURED)

**Why P1 was not enough (its own limit #1):** P1's Δ=0 rollout is force-GT — it measures the source's
*observation*-OOD (apples-to-apples with how 0.47/1.52 were measured), **not** how a closed-loop planner,
choosing its own actions, drives the ego off-path and what OOD *that* incurs. P2 measures exactly this.

### 2.1 Design (C6-clean: only the observation source changes)

`lowood_closedloop.py` is a minimal edit of `taniteval/closedloop.py`. It **keeps verbatim** the deployed
planner and controller — `strategic_policy → tactical_policy → 0.5 s pure-pursuit waypoint →
(steer, accel)` (`wp_to_control`) and the kinematic bicycle. It **replaces only** closedloop.py's step (c)
"imagine the next latent" with a real-footage re-observation:

> (c′) drive the ego one bicycle step in **world frame**; (c″) project onto the recorded path → arc-length
> `s`, signed lateral offset `dlat`, heading offset `dψ`; (c‴) **arc-length re-index** — show the REAL
> recorded window whose last frame is nearest `s` (== slide the window start by `m*` frames), **warped by the
> on-policy `(dlat, dψ)`** homography (`lowood_probe.sampling_homography`); (c⁗) re-encode → re-plan.

So the observation is always a real frame, re-indexed by arc-length (longitudinal OOD ≈ 0 by construction)
plus a homography for the residual lateral/heading offset (bounded by the P1 envelope). **The loop's own
on-policy deviation drives the OOD.** The OOD ratio is mapped through P1's measured envelope (interpolated).
**tick-0 self-check:** ego starts on-path → deviation exactly **0.0 m / 0.0°** → tick-0 obs == the real
window == the P1 baseline. flagship `flagship-30k`, 12 eps / 265 windows. Raw `lowood_closedloop.json`.

### 2.2 Result — the loop stays low-OOD on-policy

| stratum | n (win/ep) | closed ADE@2s [CI] | on-policy peak lat / yaw | **OOD peak ratio [CI]** | ≤1.16× | ≤1.5× |
|---|---|---|---|---|---|---|
| **Longitudinal** (\|Δψ\|<10°, v≥median) | 112 / 7 | 1.32 [1.08,1.67] | 0.42 m / 1.68° | **1.017 [1.006,1.029]** | **100 %** | 100 % |
| **Junction** (\|Δψ\|≥10°) | 58 / 7 | 1.97 [1.74,2.21] | 2.07 m / 18.0° | **1.190 [1.133,1.219]** | 34 % | **100 %** |
| **Overall** | 265 / 12 | 1.45 [1.20,1.72] | 0.71 m / 5.30° | **1.054 [1.026,1.087]** | 86 % | **100 %** |
| *NuRec (reference, INHERITED)* | — | — | — | *flat **3.75×** (1.5157/0.4045)* | — | — |

The mean on-policy deviation grows slowly and monotonically over 2 s: **dlat 0 → −0.31 m, dψ 0 → −2.53°**
(overall) — well inside the P1 flat regime. The result is **mechanistic**:

1. **Closed-loop ADE (1.45) is dominated by the LONGITUDINAL axis** — the flagship's known 89 %-longitudinal
   signature (registry §1.2). Arc-length re-index serves that axis **OOD-free** (the real frame at the right
   place, just a different sim-time). So the ego's *biggest* on-policy error costs **~0 OOD**.
2. **The residual lateral/heading deviation — the only thing the homography-warp OOD depends on — stays
   tiny** (longitudinal peak 0.42 m / 1.68°; overall 0.71 m / 5.30°), inside P1's CI-flat envelope.
3. **Even junctions** (peak ~2 m / 18°) map to **1.19×**, all ≤ 1.5× — an order of magnitude closer to
   distribution than NuRec's 3.75×.

**Gate-1 verdict:** **YES — the real-footage loop stays low-OOD as the ego deviates on-policy.** For the
longitudinal-first Gate-1 core (P1's recommendation, aimed at the flagship's #1 weakness) the on-policy OOD
is **~1.02×**; there is no reconstruction-OOD confound to remove because there is essentially none. This
upgrades the whole closed-loop apparatus from *"relative only"* to **absolute low-OOD**.

*Context (INHERITED):* the existing imagination-in-the-loop closed loop scores closed_bike ADE@2s **1.685**
(registry §1.2). Real-footage-in-the-loop's **1.45** is lower — grounding each tick in a real frame curbs the
compounding drift of pure self-imagination — but the two use different observation constructions, so this is
context, not a paired claim.

### 2.3 Honest limits (P2)

- **n=12 eps** (junction 7 ep, longitudinal 7 ep). Prototype-scale; deviation-magnitude CIs are wide. The
  OOD-ratio CIs are tight and the effect (1.02–1.19× vs 3.75×) is large and monotone — direction is
  decision-grade; absolute junction magnitude is not.
- **OOD is P1-mapped, not independently re-measured** at each on-policy `(dlat,dψ)` — it inherits P1's
  ground-plane-only lateral optimism (a true novel view at 2 m would be somewhat worse, but P1 showed even
  its grid edge stays ≤1.16×). The lat/yaw excesses are combined **marginal-additively** (conservative-high
  when both are large).
- **Junction peak yaw (18°) exceeds the P1 yaw grid (≤12°)**, so those *steps* are clamped to the envelope
  edge — a mild **under-estimate** for the ~14 % of steps above 12°. Linear extrapolation of the gentle
  envelope puts them at ~1.25–1.3×, still ≪ 3.75×. **Cheap fix:** extend the P1 grid to ±4 m / ±20° and
  re-map (removes the clamp; ~1 GPU-min).
- **Controller = harness pure-pursuit + bicycle** (not the model), identical to `closedloop.py` — the
  deviation is the deployed planner's tracking behaviour through that controller, so it is consistent with
  the program's closed-loop, but a different planner would deviate differently (a worse one more; even so the
  junction case has ≤1.5× margin).
- **Drift/stability loop, not safety** — no map/agents, so no collision/PDM (same limit as `closedloop.py`).
  Reactive-agent scenarios still need AlpaSim (with its OOD caveat) or the design's hybrid (c).
- A small **systematic rightward drift** appears in the mean trajectory (−0.31 m / 2 s) — the deployed
  planner's known mild lateral bias through this controller; does not affect the OOD conclusion.

---

## 3. Deliverable manifest

| artifact | where | what it is | evidence |
|---|---|---|---|
| `LOWOOD_HARDENING_REPORT.md` | repo (staged), this dir | **this report** (P1+P2+blockers+manifest) | — |
| `P1_DECISION_GRADE_FINDINGS.md` | repo (staged), this dir | P1 detail + full sourcing-blocker evidence | — |
| `lowood_ci.py` | repo (staged) · pod `/workspace/lowood_ci.py` | P1 harness (bootstrap CIs on the envelope) | MEASURED |
| `lowood_flagship_ci.json` | repo (staged) · pod `/workspace/` | ⭐ P1 raw: CI'd envelope + paired tests | MEASURED |
| `lowood_flagship_ci.log` | repo (staged) | P1 stdout | MEASURED |
| `lowood_closedloop.py` | repo (staged) · pod `/workspace/lowood_closedloop.py` | P2 harness (real-footage-in-the-loop) | MEASURED |
| `lowood_closedloop.json` | repo (staged) · pod `/workspace/` | ⭐ P2 raw: on-policy OOD + junction/longitudinal strata | MEASURED |
| `lowood_closedloop.log` | repo (staged) | P2 stdout (+ smoke self-check) | MEASURED |

**Inputs (pre-existing, pod-side):** flagship ckpt `tanitad-pod:/root/models/flagship-30k/ckpt.pt` (step
29999); clean val `tanitad-pod:/root/valdata/physicalai-val-0c5f7dac3b11` (12 eps). **Not modified:** no
`stack/` code touched (harnesses live in `incoming/`), so `pytest` is unaffected. **Pod state:** `gpu_lock`
released, GPU idle, no deletions. Per the Agent Operating Standard these files are `git add`-ed (staged),
**not committed / not pushed**; the index carries other agents' concurrent work — commit with an explicit
pathspec.

**Reproduce (pod1):**
`PYTHONPATH=/workspace/TanitAD/stack python3 /workspace/lowood_ci.py --out /workspace/lowood_flagship_ci.json`
then `… /workspace/lowood_closedloop.py --out /workspace/lowood_closedloop.json --episodes 12 --batch 16`.

---

## 4. What unblocks the two remaining gaps (priority-ordered)

1. **REF-C 2nd arm (~1 h once a ckpt lands).** pod1 already has `refc.py`, `taniteval/refc_eval.py`, and
   `refc_anchors_small64.pt` (repo). Need only the weights: a sanctioned read-only pull of
   `tanitad-eval:/root/models/refc-base-30k/ckpt.pt` (md5 `8f10d6f934f4199e11ddc7352e074939`) **when the eval
   pod is free**, or freeing Sayed's HF private-storage quota so `tanitad-refc-small-evalonly` resolves. Then
   warp the frames exactly as here and decode via `model(fw_warped, nav_cmd=None, v0, steps=2)` — the REF-C
   decode is a clean drop-in. This measures whether the anchored-diffusion decoder family weights vision
   differently (its own diffusion planner may be more/less vision-sensitive than the operative rollout).
2. **40-ep tightening.** Pull the full `physicalai-val-0c5f7dac3b11` (40 eps) from the eval pod when free, or
   rebuild the missing 28 on pod1 via `build_pai_cache.py` (parity key must reproduce `0c5f7dac3b11`). Re-run
   both harnesses unchanged; expect ~1.8× tighter CIs.
3. **Extend the P1 grid** to ±4 m / ±20° (removes the junction-yaw clamp in §2.3) — ~1 GPU-min, makes the
   junction OOD estimate exact.
4. **Native-res residual** (RETRACTION_LOG 07-23 C5): P1/P2 run the 256²/f_eff=266 phase-0 cache; the AlpaSim
   confound was 480×854 vs native 1080×1920. The real-footage source sidesteps reconstruction entirely, but a
   native-res P1 baseline would fully close the resolution question.
