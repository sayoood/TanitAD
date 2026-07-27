# TanitAD — program report, filed 2026-07-27 20:17 Berlin (18:17 UTC)

**Serves the D-025 17:57 slot; filed ~20 min late, at real wall-clock.**
**Previous: `2026-07-27-1821`.** ⚠️ Two earlier artifacts are dated **2026-07-28** — that time has not
passed. **Order by commit order, not filename.**

---

## 1. Fleet — MEASURED (live probe, this iteration)

| host | GPU | state |
|---|---|---|
| `tanitad-pod` (pod1) | **100 % / 15,348 MiB** | 🟢 `flagship-v2corpus-30k` **step 21,050 / 30,000 (70.2 %)**, loss **2.326**, **10.71 s/step** ⇒ **~26.6 h** left |
| `tanitad-pod2` | 0 % | 🔴 idle → **refilled** (the PI's small validation) |
| `tanitad-pod3` | 0 % | 🔴 idle → free; **YouTube egress BLOCKED, D-B authorization SPENT** |
| `tanitad-eval` | 0 % | 🔴 idle → available |

**All idle hosts were refilled before this report was written.** ⚠️ `step_s 540` is accumulated over
`--log-every 50` — the documented false-alarm trap.

---

## 2. ⭐ THE HEADLINE: v5 IS NOW LAUNCHABLE AND GATEABLE

Since the last report, **every technical blocker on the v5 run closed**:

| blocker | state |
|---|---|
| corpus built, renamed, membership-proved, **registered** | ✅ train 2400/2400, val 600/600, 0 missing, 0 extra |
| **evaluator reads v2** | ✅ it had **none**; `build_v2_providers` had **zero** eval callers. Verified **bit-identical to the trainer's own seam** |
| **corridor co-primary at an admissible horizon** | ✅ **K=100: 0.3775 [0.2784, 0.4766]**, junction **0.8267 [0.7800, 0.8600]**, n=120/24 |
| `run_gate.py check` | ✅ **`CONTINUE — all pre-registered gates pass`, `horizon_honest: true`**; `INCOMPLETE` without the corridor |
| **`--heldout-gate` reaches its probe** | ✅ on the real config + real caches: `primary_value 0.265049`, 264 windows / 4 episodes |
| **the stop fires on a verified degradation** | ✅ **Δ −0.3969 [−0.4399, −0.3509] separated**, direction checked **four** ways |
| preflight / batch | ✅ refuses a non-parity cache; `--batch 8 --accum 8` preserves effective 64 to **6.51 × 10⁻⁸** |

**Estimator throughout: paired episode-cluster bootstrap, `taniteval/ci.py`, B = 2000, unit = episode.
`overlapping_holdout_se` used nowhere.**

⚠️ **And two things that would have wasted a GPU-week were caught by RUNNING the command, not reasoning
about it:** `--heldout-gate` crashed at its first probe (`cond_vtarget is on but no vt_band supplied`)
**~2,000 steps in**, and **`--batch 16` OOMs at both candidate frames on a 44 GB A40 — v5 would have
died at step 0.**

---

## 3. Results landed since the last report

### 3.1 ⭐ Renderer geometry — fixed, and **my own framing was wrong**
I briefed that a cylindrical frame's horizontal mapping *is not* a homography. Measured by best-fit DLT:
**yaw IS a pure translation, residual 0.000000 px**; lateral is not (43.76 px) — **and yaw is the only
axis pseudo-simulation warps.** ⇒ the shipped code computed **the wrong matrix, not the wrong model
class.** Against an **independent numpy oracle** on 64 real v5 frames: new path **2.44e-05** max
intensity error (41× inside the bar) vs the shipped warp's **0.9937** on a frame of dynamic range 1.0 —
**3.4 × 10⁵× worse.** Deployed 256×256 path **unregressed: `torch.equal`, `max_abs_pixel_diff = 0.0`.**
⭐ **And the renderer is load-bearing:** peak cross-track **Δ −0.9110 m [−1.7359, −0.1309] separated**,
while the corridor *rate* (−0.0208) and ADE@2s (−0.0127) see nothing — **C9's shape reproduced on our
own data.**

### 3.2 ⭐ `vt_band` — priced, then wired
**`vt_band = 0` is `v_stop`**, and `route = 0` is `ROUTE_LEFT` beside `route_graded = 0.0` meaning
straight — **three wrong values, one self-contradictory.** Cost is longitudinal: **24.83 m vs 27.30 m**
travelled (human 27.47), **93 % from the VTARGET channel alone**.
⛔ **The obvious zero-fill is the worst option** — it zeroes `vt_speed` → ranks up the maximally
decelerating candidate, and braking plans NaN out `recovery` **by construction**, so **the composite
goes UP: a gate patched that way reads healthier while probing a braking planner.** It is now **blocked
by preflight with its measured mechanism**, not merely avoided.
**Default `dropped`** — in-distribution by construction (`goal_dropout = 0.5`, ~50 % of every batch, a
learned embedding row). ⚠️ **My choice pending the PI's override; one flag flips it.**

### 3.3 ⚠️ D-B YouTube — the authorization was **already spent**, and the run was **blocked**
The brief I wrote said it had not fired. **It had: 2026-07-26 12:33:31 → 16:33:33 UTC**, with exactly
the config I told the agent to launch. **The agent refused to launch and made zero requests.**
🔴 **And that run ended in a bot-block nobody recorded: 650 of 650 videos refused from 16:11:21 UTC, 0
clips — logged by the driver as `pool exhausted at 343 — proceeding`.** The prior report was finalised
**14:35 UTC**, before the block, and states *"Was it blocked? — NO."* ⇒ it **refutes** that report's
conclusion that rate-limiting was not the binding constraint.
**Delivered at zero traffic:** privacy blur **verified on a real frame** (Laplacian variance
**997.838 → 1.930**); contamination **12.8 %**, **filter caught 0 of 13**; and ⛔ **the resampling unit
was wrong — 343 clips from 55 videos, design effect 9.72×.** Regrouped: only **yaw-rate** survives
(**0.547 [0.419, 0.931] separated**); **`long_accel`'s separation was manufactured by the clip unit**
(1.165 [1.018,1.337] → **[0.976, 1.372] not separated**) — the one channel with no signal.
**LOOP_STATE's D-B paragraph is now marked SPENT + BLOCKED**; it previously read *"fire it ONCE"* and
**a future drumbeat would have fired a second run at a blocked IP.**

### 3.4 Streams NOT re-verified this iteration — INHERITED, do not quote as current
**E1c / CL-SFT · 4-brain Gates A/B/C · H2 · Orin/Thor · AlpaSim · TanitDataSet HF push.** ⚠️ The
drumbeat's *"pod2 finishes its 30k flagship — run the 8-metric gate"* is **stale**: pod2 built the wide
caches; the 30k arm is `flagship-v2corpus-30k` **on pod1 at 70.2 %**.

---

## 4. Retractions logged since the last report

| class | one line |
|---|---|
| **C40** | a **driver that mislabels its own failure**, and a `>` redirect that **erased the proof** |
| *(pending)* | `MODEL_REGISTRY` §1.5.5 records an **intent, not an outcome** — being corrected with artifact evidence |

⚠️ **Instrument defects found and fixed:** `ci.py` could print **`separated: True` beside a rounded
`0.0 [0.0, 0.0]`** — statistics were correct, the **display** lied; **four committed nodes in three
artifacts** are named, two being re-rendered. And **`MODEL_REGISTRY.md` §1.5.5 says
`flagship-v4-fromscratch` was never launched** — `metrics.json` says `final_step 29999`, `rc=0`,
**59.04 h**. **That is the arm every selection experiment this week used.**

---

## 5. Decisions owed by Sayed — four

1. 🔴 **`vt_band`** — `dropped` (shipped default, my choice) vs `band0` vs `produced`. **Defines what
   v5's early-stop stops on.**
2. 🔴 **Frame: 176×624 vs 128×576.** 176×624 **does not tile the readout**; 128×576 does and is the only
   zero-mask option, at **+3.5 pp agent samples and +3.0 m near-field ground**.
3. 🔴 **2400 vs 2376.** ⭐ **Val is 600/600 identical — no evaluation number is affected**; train-side
   only. ⚠️ The extra 24 sit in a tight position band (1798–1941). ⛔ **Cannot be dropped without
   re-registering.**
4. 🟠 **`w`**, the over-travel penalty weight — irrelevant above rank 9, a presentation choice below it.

**Carried:** HF gating on `tanitad-idm-head-v3` (UI-only) · re-issue comma `yaw_rate` (**0.105 → 0.811**)
· re-ship IDM steer (**+0.7993** vs deployed **+0.7419**; staged ckpt is **seed-0 only**) · nuScenes
Terms (**his machine**).

---

## 6. Blocked, and on what

| item | blocked on |
|---|---|
| **the small validation** | ⚠️ the old-geometry corpus lives **only on pod1, which is training** — feasibility is priority 1 of that stream, and *(a)* may not be runnable without touching it |
| **both v5 caches** | **exist on exactly one disk (pod2)**. HF is out (gated-confidential); pod-to-pod is ~1 MB/s for ~115 GB. **Needs a provisioning decision or an explicit relaxation.** |
| pod2's `stack/` | at **`0f93b98`** — **a launch from there resurrects the crashing gate.** Sync is now a runbook step. |
| further YouTube | egress **blocked**; D-B **spent**. A new PI decision. |
| `LakeRecord.image_size` | a scalar that cannot express a non-square frame |

---

## 7. What I would do next if uninterrupted

1. **The small validation** (running) — feasibility and MDE first. ⛔ **If its MDE exceeds the effect it
   must detect, say so rather than produce a confident null.**
2. **Registry repair** (running) — §1.5.5 is the highest-blast-radius falsehood on the board.
3. **Sync pod2's `stack/`** and re-verify `--print-launch PREFLIGHT: OK` from the pod itself.
4. **When pod1 finishes (~26.6 h): gate `flagship-v2corpus-30k`** on the now-complete chain —
   ⚠️ **`nonav_route_beats_majority` is VOID BY CONSTRUCTION: adjudicate INSTRUMENT-FAIL, never
   MODEL-FAIL.** It is already registered `--secondary-void` and printed as such in every verdict.
5. **Second-disk the v5 caches** — the largest single-point loss in the program.
6. Re-ship IDM steer (3-seed); re-issue comma `yaw_rate`; re-render the two flagged CI artifacts.
7. **Then, and only then, v5 — at the PI's chosen frame and `vt_band`.**

**Restart budget: v4 stays 0 / 2 — unspent, not forfeited.**
