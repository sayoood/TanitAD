# TanitAD program report — 2026-08-03 15:00 CEST (13:00 UTC)

*Every number carries its evidence class. Trends read over ≥3 logged steps, never one line.*

## 0. The three sentences that matter

1. **The programme's top defect was mis-diagnosed for weeks, and the correct diagnosis makes a large
   part of the fix free.** REF-C's manoeuvre head does not lose to lateral classes — it loses a
   3-way comparison *inside* `lane_keep`, and prior-corrected decoding recovers `brake_stop`
   0.026 → **0.503 with NO retrain**.
2. **v5f is degrading and needs a decision** (§1).
3. pod2 is terminated and the migration is closed; the fleet is three machines with no dead weight.

---

## 1. Training health — MEASURED, live probes

| run | host | step | `g_op_fwd_ade_m` (last 3) | gnorms (enc / pred) |
|---|---|---|---|---|
| **v5f** | tanitad-new | 2300 | **1.0182 · 0.6358 · 0.6622** | 12.7–37.7 / 2.6–5.2 |
| **v1arch** | pod4 | 13300 | 0.1147 · 0.0818 · 0.1363 | — |

⛔ **v5f IS GOING THE WRONG WAY.** Earlier today it sat around **0.31** (steps 1800–2000); it is now
**0.64–1.02**. gnorm_encoder is also climbing (12.7 → 37.7). This is a **trend across three logged
points**, not the per-batch noise that misled me twice earlier.
⚠️ **HYPOTHESIS, not established:** the run is still inside its LR warm-up and `--batch 4 --accum 16`
is a different optimisation regime from the one v5's schedule was tuned for. It is also only step
2300 of 30 000.
⇒ **Decision needed at the 5 k milestone** (§4.1). Do not restart on this evidence alone.

v1arch remains the programme's best curve (~0.11 here, 0.05–0.07 earlier — noisy but stable).

## 2. Since the last report

### 2.1 ⭐ REF-C's top defect: the mechanism we have been quoting is REFUTED

**"The lateral classes win every argmax" is FALSE.** MEASURED on refc-base@29999, canonical val,
39 episodes / 1364 windows (`…/incoming/2026-08-03-dtac1-tactical-head/`):

- turns are **calibrated** — turn_left 165 predicted vs 174 true; turn_right 114 vs 109
- lateral readout alone: **macro-recall 0.8290, accuracy 0.9348**
- the longitudinal mass lands **entirely in `lane_keep`** (+260 excess vs 256 missing)

⇒ The binding failure is the **within-`lane_keep` 3-way comparison** (steady 0.7104 / accelerate
0.1774 / brake_stop 0.1122), not a lateral-vs-longitudinal fight.

**The agent refuted its own pre-registration.** It registered *INPUT-limited* with a ≥0.65 threshold
fixed in advance; MEASURED `auc_lon_active` = **0.7294** ⇒ **READOUT-limited**. *The longitudinal
information is already inside the head that cannot emit a longitudinal class.*
`RETRACTION_LOG.md` R-2026-08-03-dtac1, root-cause class: **"a mechanism that is real in the source
is not thereby the binding constraint."**

**This reorders the fix and saves a retrain:**
- ⛔ Factorisation ALONE recovers essentially nothing: macro-recall **0.3621** vs a 0.3333 floor.
  The 2026-07-21 spec's factorise-only proposal would have spent a full retrain to land there.
- ✅ **Prior-corrected decoding: `brake_stop` 0.026 → 0.503 at τ=0.5, NO retrain**, for −0.109 accuracy.
- ⚠️ But `accelerate` never exceeds **0.153 at any τ** — brake_stop, the rarest class, takes the
  biggest boost and crowds it out.
- ⭐ **Third defect, previously unrecorded: the manoeuvre head never sees ego speed.**
  `man_logits = maneuver_head(pooled)` — image embedding only — while its label is
  `dv = v(t+2s) − v(t)`. The same file's `refc1` head already concatenates the measurement.
- **9.68 % of windows** carry a live longitudinal manoeuvre *and* are labelled a turn — the 5-way
  target cannot represent them. That is the irreducible part that justifies a retrain.

Capacity, measured not estimated: **+897 params (+0.00086 %)**. The agent's first implementation
cost +272,001 and its own capacity control caught it.

### 2.2 P8 — the PI's vision-only ruling is VINDICATED

`head_img` vs its own permuted-feature null, paired episode-cluster bootstrap, 1,610 clusters:
ΔAP-lift **+1.1749 [+0.7930, +1.6890]** lane_change · +1.4082 [+0.1369, +3.4552] roundabout ·
**+1.5226 [+1.1358, +1.9789]** intersection = **2.17× / 2.58× / 2.60×** base rate.
Discrimination control ran FIRST; the machinery control is ≤0 everywhere.

⚠️ **My "leak" framing was overstated and was corrected by measurement.** Labels *are* pure functions
of the ego pose track (2 probes), **but** the head's window [t−0.7 s, t] and the label's evidence
window [onset, onset+4 s], onset > t, are **disjoint** — no future information. The accurate term is
**same-source privileged access**. One genuine new defect: `omega_pre`/`alon_pre` use `np.gradient`
(a **centred** difference) and so read 0.1 s past t despite the source asserting "STRICTLY CAUSAL".

⛔ `late_fuse_scores` was **NOT wired as the deployed path** — image+ego is closed under the ruling,
and fusing the two *vision* arms is not separated anywhere. **"A fix existing is not a reason to
ship it."** `situations.py:19` retracted; R-2026-08-03-f logged.

### 2.3 P9 — the IDM fix is REFUTED, and the refutation is the result

Deriving `long_accel` from a supervised speed sequence is **separated-WORSE**:
ΔR² **−0.2530 [−0.4831, −0.0997]**. But the *baseline's* `long_accel` is **not separated from a
shuffled-latent control** (−0.0984 [−0.3087, +0.0179]) while speed (+0.7187), yaw-rate (+0.2252) and
ADE (−6.59 m) all are ⇒ **the channel carries no recoverable information from these latents; no
reparameterisation fixes it.**
⛔ **TACTICAL is at or below chance on every arm** — lateral BA 0.3293 vs chance 0.3333, `right`
**never predicted** (0/35). Invisible to ADE 5.18 m.

### 2.4 Migration closed

pod2 **TERMINATED** (`Connection refused`; both trainings verified unaffected in the same probe).
Everything was verified off it by **loading or from the far side**: epcache 349.5 GB / 3053 files
(a 121 MB episode downloaded back and `torch.load`ed), 106 GB caches with **parity sha256 re-matched
after the round trip**, 8 orphan checkpoints, the 319-file diff. SSH entry renamed
`tanitad-pod2-TERMINATED` so a stale reference fails loudly.

## 3. AlpaSim-on-Thor critical path

**Working and delivered.** gsplat renders NuRec natively on aarch64 incl. f-theta at **16–28 ms/frame**;
closed loop **0.09–0.21 s/step (5–11 Hz)**. Four closed-loop videos delivered (flagship v1 + REF-C ×
with-objects/empty-road). Tick **60.3 ms p50 / 63.1 ms p95** end-to-end on real weights (**6.17×**),
under the 100 ms budget. Four dynamic-batch engines at `thor:~/trt_deploy/`.

⚠️ **Scope, stated:** renderer **wire contract**, NOT `alpasim_runtime` ⇒ **no AlpaSim
collision/offroad score**. `TacticalSelector` has **no production caller today** (live driver is
heads-only ~24 ms) — the batch-9 work makes the *designed* path affordable, not a speedup of what runs.

**The headline science:** REF-C beats flagship closed-loop and **the whole separation is LATERAL**
(dist_to_gt +1.171 [0.030, 2.244], heading +0.084, curvature +0.0050, yaw-rate +0.038 — separated);
**ADE NOT separated** (+0.789 [−0.865, +2.728]). *ADE alone would have reported "no difference".*

## 4. Open decisions for the PI

1. **v5f at the 5 k milestone.** It is degrading (0.31 → 0.64–1.02). Continue, restart with the
   original batch schedule, or stop and back v1arch? **Recommend: hold to 5 k**, then decide on the
   gate — restarting on 3 noisy points would repeat the very error this report warns about.
2. **τ for prior-corrected decoding.** The frontier was read off **val** — that is fitting on the
   eval set. A deployed τ must be chosen on train/dev and only then confirmed on val.
3. **Two research agents silently miss their afternoon runs** (Opponent Analyzer, Production &
   Optimization; morning slots fire, afternoon ones do not). Dispatch/host-uptime, not config.
4. **Stranded:** `agent/data-engineering-2026071{0,1}` (~24 days unmerged).

## 5. Next steps

1. τ selected on train/dev, not val (fixes open decision 2 at zero GPU cost).
2. Feed ego speed to the manoeuvre head — the pattern already exists in `refc1`.
3. Cut-in / close-following scene — the discriminating test for the null objects result.
4. Junction scene — STRATEGIC is unscoreable without one.
5. v5f 5 k gate.

**Suite: 1816 passed**, 12 skipped, 2 xfailed.
