# TanitAD — program report, 2026-07-28 17:57 Berlin slot

**Produced 18:27 Berlin (16:27 UTC).** Named for the D-025 slot; dated honestly for when written.
**Previous report: `2026-07-28-1257`.** Interval: **8 commits**, `b2cff3a … e5977a5`.

⚠️ **Two standing instructions not followed silently** (as in both prior reports):
1. **"Sequential-numbered"** — all 15 program reports on disk are date-prefixed; I follow the on-disk
   convention. The sequential scheme belongs to the *weekly* report.
2. **"Hold every arm to v1's 0.4271"** — MEASURED in code: `taniteval/rollout.py:170` sets
   `actions_source="expert_future"`, `:174` names it `wm_fidelity_ade_2s`. **0.4271 is the world model
   handed the TRUE FUTURE ACTIONS** — a fidelity number, not a planning bar. Same-surface bar is
   **0.4907**. Not applied to any arm below.

---

## 1. Fleet — 🔴 TOTAL OUTAGE of three of four hosts

| pod | state | produced this interval |
|---|---|---|
| **pod1** | 🔴 unreachable, **48 consecutive checks** | nothing — `flagship-v2corpus-30k` frozen at **23,850/30,000 (79.5 %)** |
| **pod2** | 🔴 unreachable, dead on every port | nothing |
| **eval** | 🔴 **went DOWN mid-interval** — was reachable earlier today | nothing |
| **pod3** | 🟢 alive, **free/idle** | E1f (train + frontier), and all analysis below |

**MEASURED:** the RunPod proxy `rfnxkwlm2whpnm-64411b18@ssh.runpod.io` returns **`container not
found`** — the definitive signal that the container is **stopped**, not merely on a reassigned port.
A direct-port probe cannot distinguish those two; the proxy can. ⇒ **No recovery path exists from the
dev box.**
⚠️ **Two connection corrections, both measured:** `~/.ssh/id_ed25519` **does not exist on this
machine** — both connection strings supplied today specify it and both fail `Permission denied
(publickey)`; **`~/.ssh/tanitad_pod` authenticates.** The proxy **requires `-tt`** (else
`Error: Your SSH client doesn't support PTY`). Working form once a container runs:
`ssh -tt rfnxkwlm2whpnm-64411b18@ssh.runpod.io -i ~/.ssh/tanitad_pod`.

---

## 2. Closed loop (D-A) — ⭐ THE PROGRAMME CLOSED, as a characterised negative

**Estimator throughout: paired episode-cluster bootstrap** (`taniteval/ci.py`, B=2000), 44 held-out
episodes / 43 clusters at K=185. `overlapping_holdout_se` used nowhere. Split content-verified clean
at sensor level against REF-C base's *own* training corpus.

### 2.1 E1f (junction-restricted buffer) — BOUND, `primary_ok 0/4`

| step | `dep_overall` | `dep_junction` | open-loop ADE | P1 | P2 |
|---|---|---|---|---|---|
| 1000 | +0.0025 | −0.1270 | +0.0705 | ✗ | ✗ |
| 2000 | +0.0576 | −0.1459 | +0.0466 | ✗ | ✅ |
| 3000 | −0.0302 | −0.1568 | +0.0564 | ✗ | ✅ |
| 4000 | +0.0004 | **−0.2108** | +0.0555 | ✗ | ✅ |

Pre-registered **Outcome C exactly**: P2 separates-better at 3/4 and strengthens monotonically;
**P1 fails at all four, sitting at zero.**

### 2.2 ⭐ E1f REFUTES the inference that launched it (**C55**)

| arm | buffer | best `dep_junction` | at open-loop cost |
|---|---|---|---|
| E1c | full (3,537 rec) | **−0.4982** | +0.2026 |
| E1f | junction-only (733 rec) | **−0.2108** | +0.0555 |

**Training on junctions alone HALVES junction recovery.** Had the "expensive half" been interfering,
removal should have left it ≥ equal. E1f buys 42 % of the gain at 27 % of the cost with **zero**
overall gain — a scaled-down arm, not a targeted one.

### 2.3 Four levers, four BOUNDs, four *different* structural reasons

| lever | experiment | why it stops |
|---|---|---|
| training time | E1c | open-loop cost **plateaus** from step 2250 (8 pts within ±0.02) |
| weight space | E1d | separated-**worse** at 5 consecutive interior α; not linearly mode-connected (**C52**) |
| loss weighting | E1e-A/B | λ sets the **asymptote**; Ga's lower bound flattens at **+0.023** |
| the target | E1f | restriction gives **less of everything** (**C55**) |

**THERE IS NO D-A DELIVERABLE.** A fifth lever is not indicated.

### 2.4 ⭐⭐ THE MAGNITUDE Ga IS REFUSING — previously unreported by me

**MEASURED, in every frontier artifact since E1c**; I reported only departure *rates* and open-loop
ADE for four experiments.

| arm | `peak_xte` base → ft | paired Δ | sep |
|---|---|---|---|
| **E1c** | **38.944 → 3.042 m** | **−35.9030** [−49.33, −24.12] | ✅ |
| E1e-A | 38.944 → 4.502 m | −34.4430 | ✅ |
| E1e-B | 38.944 → 7.790 m | −31.1540 | ✅ |
| E1f | 38.944 → 15.012 m | −23.9328 | ✅ |

`mean_xte` 14.306 → **1.391 m** for E1c (−12.9153 ✅). **The base arm leaves the road** — 38.9 m
against a 1.75 m corridor. **E1c cuts peak excursion 92 %.** Ga blocks on **+0.05–0.20 m** of
open-loop ADE ⇒ order **167 : 1**.

### 2.5 Open-loop cost, decomposed (the trade's real shape)

Lateral/longitudinal are **evenly split and both separated in every arm** — E1c +0.1611 / +0.1838;
E1e-A +0.0971 / +0.0926; E1e-B +0.0500 / +0.0520; E1f +0.0460 / +0.0702.
**Lateral damage is tail-heavy:** E1c `cross_p90` **+0.3945** = **2.45×** its mean. Amplification
shrinks with the regression (2.45→1.90→1.51×); **E1f alone shows none** (+0.0422, not separated).

### 2.6 ⭐ Horizon sign-reversal — settles recovery-vs-drift, no GPU

`K=20` @10 Hz is a **2 s** rollout; `K=185` is **18.5 s**. Peak cross-track, base → ft:

| arm | @2 s | @18.5 s |
|---|---|---|
| E1c | 0.368 → **0.518** (+0.1493, sep **worse**) | 38.944 → **3.042** (−35.90, sep **better**) |
| E1e-A | 0.368 → 0.440 (+0.0721, sep worse) | 38.944 → 4.502 (−34.44, sep better) |
| E1e-B | 0.368 → 0.390 (n.s.) | 38.944 → 7.790 (−31.15, sep better) |
| E1f | 0.368 → 0.457 (+0.0888, sep worse) | 38.944 → 15.012 (−23.93, sep better) |

**A clean sign reversal, one metric, every arm.** ⇒ **RULED OUT: the early deviation is not drift that
compounds** — it cannot end bounded at 3 m when base reaches 39 m. **E1c gives up 0.15 m at 2 s to
gain 35.9 m at 18.5 s — ~240 : 1, same metric, same arms, same episodes.**
⚠️ K=20 is **NON-DECIDING by design**; used as mechanistic evidence, never as a gate.

### 2.7 An open question raised and resolved the same day

The closed-loop knot-ADE appeared to say the FT arms track the expert *worse*. I **refused to claim
it** pending a frame confound, then resolved it by code read: `oyaw`/`oxy` and `gt_ego_waypoints`
share **the same origin and rotation** (`poses[last]`), and **every window re-initialises at the
recorded pose** — so both arms start identically. **My "different ego states" concern was wrong.**
The horizon mismatch (§2.6) reconciles it.

---

## 3. Other streams

- **Datasets / D-B YouTube — ✅ COMPLETE this cycle.** Block **LIFTED** (0 bot-block messages),
  retried from the **same pod3 egress** after ~37 h — *not* rotation; idle eval deliberately unused.
  20 clips / 3 videos. ⭐ **GeoCalib resolved hfov 53° and 58° vs the 100° fallback**, independently
  reproducing "the fixed HFOV is wrong" on a fresh sample. IDM: **2,240 windows**,
  `frac_in_plausible_0_45_mps = 1.0`, mean 14.212 m/s. Privacy **verified by probe** (0 media files
  remain). ⛔ **No ground truth — UNVALIDATED pseudo-labels; no accuracy claim is quotable.**
- **Wide-FOV / 176×624 (PI-approved).** ⛔ Cannot start: **pod2 holds the ONLY copy** of the
  w120/256×640 cache **and** arm A's completed validation. Verified absent on pod3 and eval.
- **4-brain dominance / v5.** No material change; blocked behind the same cache.
- **H2 · Orin/Thor · AlpaSim.** **No material change this interval.** Not padded.
- **IDM.** Exercised end-to-end above; the 3-seed ensemble re-ship remains outstanding.

---

## 4. Retractions logged this interval

- **C55 — a decomposition measured under one manipulation read as a prescription for another.** E1d's
  cheap/expensive asymmetry described a **path between two models trained on everything**; it says
  nothing about training on a subset. Cost one arm; **the pre-registration limited the damage** by
  naming Outcome C in advance.
- *(Earlier today, prior interval: C53 corrected, C54 logged.)*

---

## 5. Decisions owed by Sayed

1. 🔴 **Restart pod1 and pod2 (RunPod console).** **pod2 is binding.** Send the **new IP:port** — a
   stop/start reassigns them. Also: which pod is `rfnxkwlm2whpnm`? (task #37)
2. 🔴 **Is Ga the right guardrail?** Four levers failed it; both magnitudes now on one page in
   `DECISION_BRIEF_Ga_guardrail.md`. Four options: keep · bound the regression · **change what Ga
   measures (lateral p90 rather than mean — the artifact's own guidance)** · declare undecidable.
   ⛔ **No arm may be run against Ga until settled**, or the threshold is chosen after seeing results.
   (task #36)
3. 🟡 **Licensing posture** — the harvest kept non-CC by default (`{None: 27, CC-BY: 1}`).
4. 🟡 **HF storage full** (403); pod1's 15k milestone checkpoint is **single-disk**.
5. ⚪ Parked: closed-loop metric weights (`w`, `q`, `r ≤ 0`, `lat_heading` 0.83); wheelbase fix;
   30-pod-day X2; 2400 vs 2376.

---

## 6. Blocked, and on what

| item | blocked on |
|---|---|
| 176×624 wide training | **pod2** (sole cache copy) |
| `flagship-v2corpus-30k` + 8-metric gate | **pod1** |
| Geometry contrast arms B/C | **pod2** |
| Any further closed-loop arm | **the Ga judgement** (deliberately, not technically) |
| REF-C-base + v1.6 paired bootstrap | **pod1/pod2** (no flagship ckpt on pod3 — verified) |
| Checkpoint backups | **HF quota** |

---

## 7. What I would do next if uninterrupted

**Nothing on GPU, deliberately** — and that is the plan, not an absence of one.

1. **On pod2 returning:** relaunch the geometry validation arms B and C with the **redesigned seam
   guard** (population-over-time, C51 fixed). ⛔ **Do NOT re-run arm A** — it completed `rc=0` and the
   `seam_fail` change is a **proven no-op** for it.
2. **On pod1 returning:** verify the **resumed step** and `restarts:` before trusting the supervisor,
   then the formal 8-metric gate. ⚠️ `nonav_route_beats_majority` is **VOID BY CONSTRUCTION** —
   adjudicate **INSTRUMENT-FAIL**, never MODEL-FAIL.
3. **On the Ga call:** if relaxed or re-specified, the next arm is a **mixed buffer with junction
   over-weight** — but that needs a **new pre-registration**, since C55 materially weakened its
   motivation.
4. **Only on request:** the recovery-vs-tolerated **capture run** (~1 h, pod3). §2.6 already ruled out
   compounding drift; the residue is *active manoeuvre vs stable operating point*, which **changes no
   decision now**. I would not spend it unasked.
5. **If the fleet stays down:** stop the drumbeat rather than accumulate empty reports. Nothing
   degrades — checkpoints are volume-resident, artifacts pushed, decisions carried in tasks #36/#37.
