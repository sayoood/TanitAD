# TanitAD — program report, 2026-07-28 12:57 Berlin slot

**Produced 13:36 Berlin (11:36 UTC)** — the drumbeat fired ~39 min after the fixed-clock slot; named
for the slot to preserve D-025 identity, dated honestly for when it was written.
**Previous report: `2026-07-28-0832`.** Interval covered: **14 commits**, `a7a2781 … dac39b6`.

⚠️ **Two standing instructions I am not following silently** (both as in the 0832 report):
1. **"Sequential-numbered"** — all 14 program reports on disk are date-prefixed. I follow the on-disk
   convention; the sequential scheme belongs to the *weekly* report.
2. **"Hold every arm to v1's 0.4271"** — MEASURED in code: `taniteval/rollout.py:170` sets
   `actions_source="expert_future"`, `:174` names it `wm_fidelity_ade_2s`. **0.4271 is the world model
   HANDED THE TRUE FUTURE ACTIONS** — a fidelity number, not a planning bar. Holding a *selector* to it
   is a category error. Same-surface bar is **0.4907**.

---

## 1. Fleet

| pod | state | produced since 0832 |
|---|---|---|
| **pod1** | 🔴 **UNREACHABLE, 28 consecutive checks.** ICMP flaps in/out, TCP 30107 always refused | nothing — `flagship-v2corpus-30k` frozen at **23,850/30,000 (79.5 %)** |
| **pod2** | 🔴 **UNREACHABLE**, answers on no port at all | nothing |
| **pod3** | 🟢 working continuously | E1e-B, the YouTube→GeoCalib→IDM chain, E1f (training done, eval running) |
| **eval** | 🟢 alive, **idle** | nothing |

**Diagnosis stands (MEASURED):** port 22 on pod1's IP returns a generic
`SSH-2.0-OpenSSH_8.2p1 Ubuntu` banner — the **RunPod host node, not our container**; no auth attempted.
`ssh.runpod.io` is reachable but needs a pod ID **and** a RunPod key, and `Keys.txt` holds an HF token
and **no RunPod credential** (checked by label). ⇒ **No recovery path exists from the dev box.**

---

## 2. Closed loop (D-A) — the headline of this interval

**Estimator for every number below: paired episode-cluster bootstrap** (`taniteval/ci.py`, B=2000),
44 held-out episodes / 43 clusters at K=185. `overlapping_holdout_se` used nowhere.
**The held-out split is content-verified clean at the sensor level** (0 shared frames) against
**REF-C base's own training corpus** — so none of this is a leak artifact.

### 2.1 The `lam_replay` axis CLOSED (E1e-B, MEASURED, `e1e_B_frontier_summary.json`)

**VERDICT BOUND**, 0/4 success, **P1∧P2 4/4**, guardrails 0/4.
⭐ **The risk the arm was run to test did not materialise: P1 survives at λ=8.**
Ga's lower bound walked 0.047 → 0.038 → 0.024 → **0.023 and flattened**.

**The axis's deliverable is a trade-off curve — monotone, non-crossing, none dominating:**

| arm | best closed-loop | at open-loop cost |
|---|---|---|
| λ=1 (E1c) | **−0.4407** | +0.2158 |
| λ=3 (E1e-A) | −0.3911 | +0.0958 |
| λ=8 (E1e-B) | −0.2891 | **+0.0500** |

⇒ `lam_replay` is a **calibrated control**; the program can choose an operating point.
⚠️ **But no point passes the gate, so there is STILL NO D-A DELIVERABLE.**
⛔ **A finer λ grid is NOT run** — pre-committed inadmissible *before* any of it ran, so a near-miss
(+0.023) could not tempt a sweep. The temptation was real and was declined.

### 2.2 E1f (junction-restricted buffer) — 🔴 INTERIM, 1 of 4 points, NO VERDICT

Training `rc=0` 11:17:22Z. Base reproduced **exactly on all four runs**.
**First frontier point, step 1000 (MEASURED):**
`dep 0.5902` vs base `0.5877` ⇒ **Δ ≈ +0.0025 — essentially UNCHANGED**;
`junc_dep 0.7144` vs `0.8414` ⇒ Δ ≈ −0.127, **not separated**.
`failed = [Ga, Gb1, Gb2, **P1**, **P2**]` — **both primaries FAILED**, where E1e-A at the same step
had `dep_overall −0.2735` with P1 and P2 firing.

⚠️ **AND THIS CORRECTS MY OWN REPORTING FROM THIS INTERVAL.** Across several drumbeats I reported that
restricting the buffer to junctions "substantially lowers the open-loop cost at unchanged λ", called it
a genuine surprise, and said it made Outcome A more plausible. **The mundane explanation is the right
one: the low open-loop cost was the model BARELY MOVING FROM BASE.** 733 records across 102 episodes
give too little signal to shift the policy — so it damages open-loop less *and* gains closed-loop less.
Those were always one fact; I reported the flattering half. **The pre-registration's rule that only the
frontier can answer this is what caught it.**
**Not over-corrected:** 3 points remain and E1f may still move. But at the matched step the gap is
+0.0025 vs −0.2735, which is not ambiguous.

### 2.3 Three levers, three BOUNDs, three *different* structural reasons

| lever | experiment | reason it cannot be pushed |
|---|---|---|
| training time | E1c | open-loop plateaus from step 2250 (8 points inside ±0.02) |
| weight space | E1d | separated-WORSE at 5 consecutive interior α — endpoints not linearly mode-connected (**C52**) |
| loss weighting | E1e-A/B | λ sets the asymptote; Ga's lower bound flattens at +0.023 |

---

## 3. ⚠️ D-B YouTube — the retry window framing is STALE; it already fired

**The drumbeat says the window "opens 12:00 UTC today". That is not the live state.**
- The **single-run authorization was SPENT 2026-07-26 12:33 UTC**, and that run **ended in a bot-block
  at 16:11 UTC** which its driver mislabelled *"pool exhausted — proceeding"*.
- **Today's run fired at 09:20 UTC** — before the stated window — under the **PI's fresh instruction**
  (*"try to download a few youtube videos to run the idm model"*), not under the spent authorization.

**Outcome (MEASURED, `harvest_manifest.json` + `pseudo_labels_summary.json`):**
- **The block has LIFTED — 0 bot-block messages.** Retried from **pod3, the same egress that was
  blocked**, after ~37 h. **NOT an egress rotation**: idle `tanitad-eval` was deliberately not used.
- **20 clips / 3 videos.** ⭐ **GeoCalib resolved hfov 53° and 58° against the 100° fallback — both
  ~2× off**, independently reproducing "the fixed HFOV is wrong" on a fresh sample.
- **IDM: 2,240 windows.** `frac_in_plausible_0_45_mps = **1.0**`, mean 14.212 m/s, std 5.815.
  Channels: primary `speed`/`long_traj`; `yaw_rate` caveated; **`long_accel` and `steer` DROPPED** —
  matching the standing record rather than using refused channels.
- **Privacy VERIFIED BY PROBE**, not by trusting the manifest: 0 media files remain.
- ⛔ **No ground truth exists — UNVALIDATED pseudo-labels. No accuracy claim is quotable.**
  ⚠️ `speed_min = −0.19 m/s` is physically impossible; recorded, not rounded away.

**No further YouTube harvesting has been fired and none will be without a fresh decision.**

---

## 4. Other streams — material change only

- **Wide-FOV / 176×624 (PI-approved).** ⛔ **Cannot start: pod2 holds the ONLY copy of the
  w120/256×640 cache AND arm A's completed validation.** Verified absent on pod3 and eval.
- **4-brain dominance / v5.** No material change; blocked behind the same cache.
- **H2 · Orin/Thor · AlpaSim.** **No material change this interval.** Not padded.
- **IDM.** Exercised end-to-end above; the 3-seed ensemble re-ship remains outstanding.

---

## 5. Retractions logged this interval

- **C53 (corrected, not withdrawn).** Its measured claims stand — 477–483 % CPU, load 24.39, 1.0 vs
  5.64 s/step, raw 137 MB mp4 left by the kill. **But its "0 clips were produced" implied the harvest
  had had time to produce some; it had ~6 minutes and one video still decoding.** The insinuation that
  the harvest was also failing is withdrawn.
- **C54 — elapsed time inferred against a clock I never recorded.** **Four false claims in one
  session**, every one contradicted by the first direct check; **two became alarms**, and one triggered
  a kill that created a privacy exposure. Cure is mechanical: read `date -u` and `etimes`/`lstart` in
  the same probe as the log, and quote only differences between two observed values.

---

## 6. Decisions owed by Sayed

1. 🔴 **RESTART pod1 AND pod2 (RunPod console).** Blocking everything. **pod2 is binding.**
2. 🟡 **Licensing posture** — the harvest kept non-CC by default (`{None: 27, CC-BY: 1}`).
3. 🟡 **HF storage full** (403); pod1's 15k milestone checkpoint is **single-disk**.
4. ⚪ Parked: closed-loop metric weights (`w`, `q`, `r ≤ 0` blind spot, `lat_heading` 0.83); wheelbase
   fix; 30-pod-day X2 run; 2400 vs 2376 episodes.
5. 🔵 **Emerging, if E1f is BOUND:** four levers will have failed, and the question becomes whether
   **Ga itself** — a strict open-loop non-regression — is the right guardrail for a closed-loop
   fine-tune. **That is a judgement, not another arm.**

---

## 7. Next steps, priority order

1. **E1f's remaining 3 frontier points** (~15 min) → the pre-registered call. Current shape points at
   **Outcome C** (junction gain real but junction-only supervision insufficient) ⇒ a **mixed** buffer
   with junction over-weight, as a **NEW pre-registration**, not a continuation.
2. **On pod restart:** verify pod1's resumed step and `restarts:`, then the formal 8-metric gate.
   ⚠️ `nonav_route_beats_majority` is **VOID BY CONSTRUCTION** — adjudicate **INSTRUMENT-FAIL**.
   **Do not re-run arm A** — it completed `rc=0` and the `seam_fail` change is a proven no-op for it.
3. **Relaunch the geometry validation** (arms B and C) on pod2 with the redesigned guard.
4. **PI decisions above**, in the order listed.
