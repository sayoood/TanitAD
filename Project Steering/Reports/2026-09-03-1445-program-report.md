# TanitAD programme report — 2026-09-03 14:45 Europe/Berlin

*Fixed-clock slot (the 12:57 slot, fired late). Previous report: 2026-09-03 10:05. All wall-clock times
Europe/Berlin; **Thor and pod logs are UTC** and are labelled Z. Every number carries its evidence class and
tier. `MODEL_REGISTRY.md` and raw eval JSON are the only quotable sources for model facts.*

⛔ **THIS REPORT WAS COMPILED DURING A STORAGE OUTAGE AND COULD NOT BE COMMITTED WHEN WRITTEN.** The G: mount
(which holds the repository) has been refusing **all content I/O drive-wide** since ~12:05 Berlin — see §6.
The report was written to `C:\Users\Admin\report_20260903_1445.md` and lands with the three pending commits when
the mount returns. Nothing in it depends on the mount: every measurement below came over SSH.

⚠️ **Two of this slot's arguments are stale, as at 10:05.** They cite the refav1 run `refav1-b1-v72-1ep-21109`
(**RETIRED 2026-09-02**, replaced twice; the live run is `refav1-b1-v72-ep3-speed`) and ask for the open
decision **C-NAV-SOURCE-DIVERGENCE** (**CLOSED 2026-09-02** — the PI directed the switch, refcv3 moved to the
v7.2 nav token at step 16,500, verified reaching all three layers). Open decisions are in §5.

---

## 1. Fresh measurements (MEASURED 2026-09-03 12:43Z, tier T0 — train-side instruments, never driving performance)

### refav1 — `/home/nvidia/experiments/refav1-b1-v72-ep3-speed` (Thor)

| quantity | value |
|---|---|
| step | **8,550 / 21,109 (40.5 %)** |
| marginal pace | **3.50 s/step** (two adjacent log rows) |
| ETA | **≈ 12.2 h** ⇒ ends ≈ 2026-09-04 00:55Z |
| log freshness | 152 s (log-every 50 at 3.5 s/step ⇒ healthy) |
| `DRIFT_ALARM` | **NONE** |
| `grad_norm` | 2.16, finite; **1 step SKIPPED** by the overflow guard, unchanged since ~step 1,200 |
| `adapter_std` | 0.480 → **0.836** — RISING, the anti-collapse read |
| `tgt_std_op` | **1.001** — the known-value control, ≈ 1.0 as required |
| participation | **24.0** — a within-run trend, ⛔ NOT a gate (`C-PARTICIPATION-FLOOR-RETIRED`) |
| `loss_feat_op` | 1.382 → **0.400**; best 10-row window 0.430, last-10 0.439 |
| `tgt_std_tac` | max/min **4.80** since row 100 ⚠️ still climbing |

**Reading.** Healthy and past 40 %. The anti-collapse instrument points the right way and the run has needed
exactly one intervention all day — the bf16 overflow guard dropping a single batch, which in the previous arm
would have cost ~150 steps. Two watch items unchanged: the tactical target scale keeps growing (the EMA teacher
slows it, it does not pin it) and the operative loss's last-10 window sits just above its best.

### refcv3 — `/workspace/experiments/refcv3-b1-v72-30k` (A40 pod)

| quantity | value |
|---|---|
| step | **31,400 / 40,284 (78.0 %)** |
| pace | **3.95 s/step** |
| ETA | **≈ 9.8 h** ⇒ ends ≈ 2026-09-03 22:30Z |
| supervisor relaunches | **8** (6 kernel-OOM deaths + 3 planned switches; **0 deaths since 17,500**) |
| `oom_kill` | **24, unchanged** |
| `eval_error` rows | **0** |
| uptime | 52,626 s (14.6 h) uninterrupted |
| last evals (T0) | 30,000: traj 0.9303 / lat_tac 0.8588 · 30,500: 0.9332 / 0.8840 |

**Reading.** The lateral manoeuvre column is at the best values of the run. Era-C flatness stands as reported at
10:05 under the reconciled estimator (§6): nearly flat everywhere except the strategic gate. Recommendation
unchanged: **read the epoch-end checkpoint tonight, do not extend.**

---

## 2. Streams since 10:05 — six completed, all banked or queued

| stream | produced | state |
|---|---|---|
| Trunk anchor (R23) | `--w-trunk-anchor` wired, monitored, refuses without its monitor; **+4.25 %** encoder cost | ✅ committed `899b85b` |
| refcv3 T1 adapter (R20) | the instrument, restarted from `refav1_arm.py` with evidence; found the SELECTION-PROFILE gap | ✅ committed `d64ba75` |
| refcv3 nav-zero | `os_navzero`, null derived from source; caught a defect in the shipped flag | 🔶 **written, blocked by the outage** |
| Cost repair (R41) | the panel; **FAILS** its criterion; the world model prefers straight | 🔶 **RESULT.md blocked by the outage**; 10 artifacts staged pre-outage |
| GS-9 + actdiv units (R12, R22) | GS-9 **admissible** where L3 was not, and it FAILS; found `v` leaking at the model boundary | 🔶 **blocked by the outage** |
| Decodability SOTA | the fourth theme; all four of our failure modes are published failure modes | ✅ committed `4caadc5` |
| Estimator reconciliation (R33) | my page's number was inflated; the package's estimator is primary | ✅ committed `0076755` |

**Struck this period:** R10, R12, R13, R20, R22, R23, R26, R27, R33, R37, R38 (partly), R41 (with a FAIL).
**Added:** R50–R62.

---

## 3. The scientific position — it moved today, and away from the world model

At 10:05 the reading was *"the flat plan is the cost, not the world model."* Two measurements have refined it:

1. **The cost repair FAILS.** On the windows where the car actually turns (27 windows / 10 episodes / 3 tokens),
   the best cell in the whole panel puts the curvature choice on the human's side on **6/27** and **0/27**
   against a bar of 14. Convention-dependent and checkpoint-dependent.
2. **And once the goal term is legible, the model itself prefers straight** — the chord's own κ-argmin sits at
   κ = 0 on **110–130 of 140** windows. No weight repairs that.

⇒ The chain is now: a **degenerate goal** (the goal IS the constant-velocity rollout on 79–92 % of windows; the
decoder asks for curvature on 9/27 turning windows in one checkpoint and **0/27** in the other) → the model,
scored against that goal, prefers straight → no cost weight can rescue it. **The tactical decoder, which MAKES
the goal, is the lever**, and its apparent competence is borrowed from an oracle nav input that will not exist
at deployment. `PREREG_TACTICAL_DECODER.md` holds the arms; its preflight is discharged.

---

## 4. The four edges — evidence grade each

| edge | grade | the evidence |
|---|---|---|
| **Planning** | ⛔ **NEGATIVE, and now localised to the GOAL** | The cost repair fails (6/27, 0/27); the model prefers straight once the goal is legible; the decoder asks for curvature on 0/27 turning windows on one checkpoint; its ranking collapses to chance without oracle nav. |
| **Efficiency** | ✅ **STRONG** | refav1 3.50 s/step (6.1× over fp32) and 40.5 % complete on one skipped batch all day; refcv3 −13.4 % on uint8 and 14.6 h uninterrupted; the trunk anchor costs +4.25 % rather than the +20.9 % a naive second forward would. |
| **Safety / self-knowledge** | ✅ **STRONG — again the edge that carried the day** | **Twelve RETRACTION_LOG entries dated 2026-09-03**, three of them added since 10:05 and all three correcting *my own* published numbers. Every one was caught by the measurement it commissioned; none reached the register as fact. |
| **Data efficiency** | 🔶 **MODERATE, with a new negative** | Parity holds; eval-clip exclusion closed. But 5 of 16 tactical classes have ZERO support, and GS-9 now measures **raw pixels carrying 5–200× more transition-specific structure than the learned Δz** (point estimates, no CI). |

---

## 5. Decisions required from the PI — with defaults

1. ⭐ **Restart the Google Drive client?** It is wedged drive-wide (§6). I have NOT done it: the repository and
   its object store live on that mount, the client is wedged rather than dead, and everything blocked is already
   safe elsewhere. *Default: wait — the watch lands everything on recovery.*
2. **Tier ruling (R30): does the doctrine admit as T1 a model that consumes NO actions?** Blocks register
   decision 9 and gates tonight's refcv3 read. *Default: score refcv3 at its own tier, report each arm's margin
   over the shared `ha0` floor, refuse any head-to-head of levels.*
3. **DINOv3 ViT-B/16 pull** (R24) — HF quota; only ViT-L/16 is on the box. *Default: seed from ViT-L/16.*
4. **`plan_horizon_s` 2.0 → 6.0** (R34/R38) — the planner's own seed cannot reproduce its own goal.
   *Default: do not change it before the tiny-ladder arm.*
5. **D3b (R54): which margin leads the H-vs-F table** — the fed-nav one or the deployment (nav-zero) one.
   *Default: lead with the deployment margin; quoting only the fed-nav one overstates the system.*
6. **`--action-units steer` default** (R21) and **`cost_time_grid` default** (R36) — both need a stated cut-over.
7. **Readiness decision G.1 must NOT be adopted on the drift metric** — a shuffled control reproduces 100–102 %
   of it. *Default: re-measure with the control subtracted.*

---

## 6. Incidents, honestly

- ⛔ **STORAGE OUTAGE, ongoing ~2 h 40 m.** G: refuses ALL content I/O **drive-wide** (verified at five levels of
  the tree; only the root lists, and it holds no files). `ls` succeeds from cached metadata — the trap that
  makes this look like a path problem. The Drive client is alive and "Responding" with **~9.7 h of CPU over 19 h
  of uptime**: wedged, not dead. Three finished deliverables cannot be committed. **Nothing is at risk:** each is
  verified in a durable location off both the scratchpad and the working mirror, hashes checked; the 16 files
  staged before the outage are untouched in the index; two detached landers poll with a real
  write→read→compare round-trip; three idempotent commits are written and refuse cleanly while the mount is down.
- ⚠️ **A near-stranding, correctly escalated by the stream that caused it.** Its work existed in two places —
  both on C:, one of them the working mirror, which a background process re-syncs FROM the repo and would have
  silently reverted. A durable third copy now exists (69 files, hashes verified).
- ⛔ **Three of my own published numbers were corrected since 10:05**, each by the measurement that commissioned
  it: (a) the chord's resolution gain, which I wrote as ~10⁷ and is a realised **2.3–7×**, capped at 21 by the
  candidate grid — I quoted a format's capacity where a system's realised count was meant; (b) the
  lateral/longitudinal ratio **1/300th–1/40th → 1/107th–1/14th** in the convention the model is actually trained
  on, with my "at matched ≈2σ" qualifier shown FALSE (0.1 rad/m is 6.09 σ); (c) the refcv3 era-C scatter
  reading, inflated on both ends of the ratio by an ad-hoc estimator I wrote inline beside a pre-registered one.
- ⚠️ **`v` was leaking into a probe at the MODEL BOUNDARY** — `_lift3` puts it in the third channel while the
  zero-action arm zeroes only steer and accel. `v_t` alone scores the target at **0.9986**, and that was the
  only passing cell. `D-V-EXCLUDED` must be restated to cover inputs, not only decode targets (R58).
- ⚠️ **An unowned file appeared inside another stream's package directory** during the outage (R57); the commit
  that touches that directory asserts its absence.

---

## 7. Ordered next steps

1. **On mount recovery**: land the three commits in order (`commit_tick21/22/23`), then republish Training Watch.
2. **≈ 22:30Z tonight**: refcv3's epoch ends → the first real T1 read with the new instrument, per
   `REFCV3_ARM.md` §3. ⛔ The tier ruling (§5.2) gates what may be CLAIMED, not what may be measured.
3. **≈ 2026-09-04 00:55Z**: refav1's speed epoch ends → the first admissible T1 read of a full-epoch refav1,
   and the first read where the model has had a velocity input.
4. **Queued, 0 GPU**: the tactical-decoder arms (preflight discharged); the selection-profile gate into
   `T1_CHECKLIST.md` (R50); `D-V-EXCLUDED` restated for inputs (R58); the κ² charge conflict reconciled (R56).

---

*Compiled by the Master Mind. ⛔ Not committed when written — the storage outage; it lands with the three pending
commits. Commit route when it does: the scratch-index plumbing pattern, because `scoped_commit.py`'s `read-tree`
has been dying with in-page errors on this mount. ⛔ Never pushed — the standing rule is commit only.*
