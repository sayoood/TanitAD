# TanitAD — program report, filed 2026-07-27 18:21 Berlin (16:21 UTC)

**Serves the D-025 12:57 slot; filed late, at the real wall-clock time.**

## ⚠️ 0. A clock discrepancy that must be fixed before it corrupts ordering

`date -u` on this box returns **2026-07-27 16:21 UTC**. But:

- the previous report is filed as **`2026-07-28-0757`** — **a day into the future**;
- `LOOP_STATE.md` carries **`LAST_UPDATED: 2026-07-28 ~05:57 UTC`** — likewise;
- eleven deliverable directories are named **`…/incoming/2026-07-28-*`**.

**None of that time has passed.** This is the known narrative-clock artifact, and it is now load-bearing:
**report ordering by filename is unreliable**, and a future-dated `LAST_UPDATED` makes a stale state file
look fresh. ⇒ **This report is named for the real time.** The `2026-07-28-*` artifacts are **not renamed**
(they are committed and referenced across a dozen documents; renaming would break more than it fixes) —
they are **flagged here as misdated**, and ordering should follow **commit order**, which is correct.

---

## 1. Fleet — MEASURED (live probe, this iteration)

| host | GPU | state |
|---|---|---|
| `tanitad-pod` (pod1) | **45 % / 15,348 MiB** | 🟢 `flagship-v2corpus-30k` **step 20,400 / 30,000 (68.0 %)**, `step_s` 540 ⇒ **10.8 s/step** ⇒ **~28.8 h** remaining |
| `tanitad-pod2` | **0 % / 0 MiB** | 🔴 idle → **refilled** (renderer geometry). ⚠️ its `/workspace` **hit the MooseFS quota** this session |
| `tanitad-pod3` | **0 % / 0 MiB** | 🔴 idle → **refilled** (D-B YouTube retry) |
| `tanitad-eval` | **0 % / 0 MiB** | 🔴 idle → available to the renderer stream |

⚠️ `step_s 540` is **ACCUMULATED over `--log-every 50`** — the documented false-alarm trap, not a pathology.
**All three idle hosts were refilled before this report was written.**

**What pod2 produced since the last report:** both v5 caches, **complete** — train **2,401 entries**, val
**601 entries / 20 GB** — then renamed, membership-proved and **registered**.

---

## 2. ⚠️ D-B YouTube — the window opened and the retry HAD NOT FIRED

**MEASURED:** the window opened **12:00 UTC**; it is now **16:21 UTC**. pod3 held **zero** processes, and
**no `results_scaleup*.json` exists newer than 12:00 UTC**. `/workspace/tmp/yt_scaleup` exists from the
prior attempt. ⇒ **the standing authorization had lapsed unexecuted for 4 h 21 m.**

**Now launched** — gentle config `W=2 TARGET=400 SEEDS=4 --sleep 4`, GeoCalib geometry, **once**, on pod3.
⛔ **Bot-detection is never bypassed**; a block is a rate-limit signal and the instruction is **STOP and
report**. ⚠️ Priority 1 in that brief is **verifying the privacy blur on a real frame** — GeoCalib's
`opencv-python` dependency once silently clobbered the pinned cv2, dropped `CascadeClassifier` and
**broke the blur**. **Outcome unknown at filing time.**

---

## 3. Results landed since the last report — estimator named on every one

**Estimator throughout: paired episode-cluster bootstrap, `taniteval/ci.py`, B = 2000, unit = episode.
`overlapping_holdout_se` used nowhere.**

### 3.1 ⭐ Resolution: `NO GAIN` — 384×960 refused
Three independent probes returned the pre-registered `NO GAIN`; **none** returned `GAIN`; every other
branch was an explicit **refusal**. Probe sensitivity **demonstrated first** (6× rung: **−0.01544 AP
[−0.02479, −0.00624]**, **−0.04025 R² [−0.05130, −0.03020]**, dynamic ranges 5.71/14.93/4.28 vs a ≥2 bar).
**The mirror of the 384×960 step costs nothing: −0.00150 AP [−0.00402, +0.00139]**; the knee sits
**2.3–3.0× below** the v5 frame. **A real 384×960/1440-token arm was built anyway and won nowhere** —
separated-**worse** on speed, yaw and lane-change AP.
⭐ **And it separated two effects we had conflated: the wide frame is separated-better than today's
(+0.04246 AP, +0.02774 R²) and 98.1 % of that survives at matched px/deg ⇒ what widening buys is NOT
angular resolution.** ⚠️ Counter-case at equal prominence: **ego yaw rate is separated-WORSE (−0.03546 R²)**.

### 3.2 ⭐ Clean rig fix — and it is a SLICE
Pad/mask **0.0000000000 on both rigs (maximum, not mean)**. Verified bit-identical against frames rebuilt
from their own mp4s: **6/6 clips, 1,206 frames, `max_abs_diff 0`.** Cost: **rows-only 0.0165 %** of agent
samples vs **columns-only 1.1770 %** — the vertical band is nearly free. Frame **176×624**, **429 tokens
(−33.0 %)**.
⛔ **Not closed:** the rig stays readable from pixels (all-zero fraction **0.0000834 A vs 0.0079316 B**),
~97 % transient scene black — a **corpus-balance** confound no crop removes. **And no ADE was measured:
the clean frame is FREE TO TRY, NOT PROVEN BETTER.**

### 3.3 ⛔ The closed-loop composite was blind to over-travel — fixed and versioned
`ego_progress = clamp(ratio,0,1)` charged **nothing** for over-travel; **v1 over-travels on 48.80 % of
windows** (p95 **2.430×**). Fixed as `@twosided_v2`, **reproduction gate exact (`max|diff| = 0.000000`)**.
⛔ **The consequential flip: `v1_tactical_follow − cv_holdv0` n.s. → −0.1212 SEP.** *"v1's plan is tied
with doing nothing"* **was a metric artefact — it is separated-worse.** The whole v1 family drops below
every REF-C arm, **including both ego-ablated ones**.
✅ **`cv_holdv0` still ranks first among realisable arms at every *w*.** Rank 1 overall, under **both**
terms, is the **oracle longitudinal schedule**.

### 3.4 ⭐ E-GOAL-4 — joint training confirms the goal and re-scopes it
**+62.09 % / +64.08 %**, beating the fixed rule on both backgrounds. ⛔ **But +46.3 % over-credited it
1.76×**: a trained selector with **no goal** already recovers **+35.62 %**, so the capacity-matched
marginal is **+26.31 points — plan v5 with +26.** ⭐ **The marginal is identical on two backgrounds whose
totals differ by 15 points.** ⭐⭐ **And the goal carries no information at all** — `g_along` = GBM(`v`,
`ax_fd`) at **R² 0.999894**, with the no-goal arm fed both columns ⇒ **an inductive bias, not a channel**
⇒ **funding a strategic supplier is the wrong lever at this feature list.**

### 3.5 ⭐ v5 is now trainable, evaluable, registered — and it would have crashed
Evaluator now reads v2 (it had **no** v2 path; `build_v2_providers` had **zero** eval callers), verified
**bit-identical to the trainer's own seam**. Encoder is **0.662 of a real step** ⇒ the clean frame costs
**0.767× a full step**. Preflight now **fails when it should**, demonstrated three ways. Batch verified on
real steps: accumulated gradient matches a single batch-of-64 to **6.51 × 10⁻⁸**.
🔴 **And running the staged command found that `--heldout-gate` CRASHES at its first probe**
(`cond_vtarget is on but no vt_band supplied`) — **after ~2,000 steps, several GPU-hours.** **Every
existing test of that path uses a stub head.** A **tripwire was added, not a fix**, because the fix is a
design choice.

### 3.6 Streams NOT re-verified this iteration — INHERITED, do not quote as current
**E1c / CL-SFT · 4-brain dominance Gates A/B/C · H2 · Orin/Thor · AlpaSim · TanitDataSet HF push.** No
fresh probe. ⚠️ The drumbeat's *"pod2 finishes its 30k flagship — run the 8-metric gate"* is **stale**:
pod2 built the wide caches; the 30k arm in flight is `flagship-v2corpus-30k` **on pod1 at 68 %**.

---

## 4. Retractions logged since the last report

| class | one line |
|---|---|
| **C34** | a lever measured against the **wrong counterfactual** (+46.3 % → +26.31 capacity-matched) |
| **C35** | a **requirement curve is a property of the consumer** — σ₀ does not transfer between rules |
| **C36** | an input can be **worth points while carrying no information** (R² 0.999894) |
| **C37** | a **gated result relayed as its ungated sibling** — mine; overstated a shipped win **1.154×** |
| **C38** | a **rig signal that survives the geometry fix** |
| **C39** | **a field request the sensor never satisfied**, masked into invisibility (**8.67 %** of clips cannot supply 120°) |

---

## 5. Decisions owed by Sayed — four, all newly priced

1. 🔴 **`vt_band`** — `0` (band 0 is a *real* speed band) vs an explicit `VT_DROPPED`. **This defines what
   v5's early-stop stops on.**
2. 🔴 **2400 vs 2376.** ⭐ **Val is 600/600 identical, so every evaluation number is unaffected** —
   train-side only. ⚠️ The extra 24 are **not a random 1 %**: they sit in a tight position band
   (1798–1941). ⛔ **Dropping them is impossible without re-registering** — it breaks the digest, and the
   position→clip-id mapping **is not on pod2 at all**.
3. 🔴 **Frame: 176×624 vs 128×576.** No longer cosmetic — **176×624 does not tile the readout**; 128×576
   does and is the only zero-mask option, at **3.5 pp more agents and +3.0 m of near-field ground**.
4. 🟠 **`w`, the over-travel penalty weight** — irrelevant above rank 9, a presentation choice below it,
   but it should be **chosen and recorded**.

**Carried:** HF gating on `tanitad-idm-head-v3` (UI-only) · re-issue published comma `yaw_rate`
(**0.105 → 0.811**) · re-ship IDM steer (**+0.7993** vs deployed **+0.7419**, separated on both corpora;
staged checkpoint is **seed-0 only**) · nuScenes Terms (**his machine — indefinite**).

---

## 6. Blocked, and on what

| item | blocked on |
|---|---|
| **v5 launch** | the `vt_band` crash (**would die at step ~2000**) |
| **v5's corridor co-primary** | the renderer is **hard-coded to a pinhole at f=266, c=128** — on v5's cylindrical frame the error is **mean 46.3 px against a true shift of 42.7 px**, 99.08 % of pixels >1 px. **Being fixed now.** |
| gate verdict completeness | the above ⇒ `check` correctly renders **`INCOMPLETE`** |
| both v5 caches | **exist on exactly one disk (pod2)** |
| `LakeRecord.image_size` | a scalar that **cannot express a non-square frame** |

---

## 7. Next steps, priority order

1. **The `vt_band` decision** — both options measured and priced for the PI, not chosen by an agent.
2. **Renderer geometry** (running) — without it v5 cannot be gated on its co-primary, and **C9 says an
   ADE-only gate hid the dominant failure mode by ~168×**.
3. **D-B YouTube** (running) — one gentle run; blur verified first; **STOP if blocked.**
4. **The PI's small validation** — matched short runs, old vs clean frame, **primary = the composite, NOT
   `ade_0_2s`**, with bar, n **and MDE** pre-registered.
5. **Second-disk the v5 caches** — 20 GB + ~95 GB on one host is the program's largest single-point loss.
6. Re-ship IDM steer (3-seed), re-issue comma `yaw_rate`, tag `0.4907` with its deployment scope.

**Restart budget: v4 stays 0 / 2 — unspent, not forfeited.**
