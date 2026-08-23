# PRE-REGISTRATION — does in-envelope recovery augmentation reduce REF-C's closed-loop lane departure?

**Stream D2** (`a1f26c92`). **Date:** 2026-07-23 (Berlin). **Binding:** bars committed **before** the run
(GATE_PROTOCOL discipline); **all outcomes written down here in advance.** This is a *cheap discriminating*
experiment, not a promotion gate — REF-C's deployed weights are untouched. Nothing is launched by this
document.

**Host:** eval pod `tanitad-eval` (A6000, 46 GB), **AFTER `abe82f1f` (LaneKeep) frees it** —
`gpu_lock acquire refc-cl-improve`, poll for `abe82f1f`'s `LOWOOD_LANEKEEP_DONE` first. Do **not** collide
with `abe82f1f`; do **not** touch pod2 (v4.2b) / pod3 (Branch B) / pod1 (frozen-WM). If eval stays busy,
this pre-registration + the CPU-smoked scripts are the deliverable and the run happens when eval frees.

---

## 1. Two stages: a ~0-GPU probe that can KILL the FT, then the FT

Mirrors the gradient-coupling stream's Phase-0 discipline: run the near-free discriminator **first**; only
spend FT GPU if it greenlights.

```
P2a recovery-response PROBE (inference only, ~minutes, 0 training)
   ├─ recovery_ratio HIGH (planner already recovers)  → STOP: departures are execution-bound,
   │                                                     FT predicted null. Bank the measured ratio.
   └─ recovery_ratio LOW  (planner covariate-shift BLIND) → GREENLIGHT P2b
         P2b decoder-only recovery-augmentation FT (~1–2 GPU-h) + held-out corridor_departure
            ├─ WIN   → data-efficient covariate-shift lever that GENERALIZES (Gate-1 memorised; this didn't)
            ├─ NULL  → synthetic recovery doesn't transfer through a decoder-only FT → measured bound
            └─ HURT  → recovery bought by over-steering (peak_xte↑) → λ_dev/lat_max fix, re-read
```

---

## 2. P2a — recovery-response probe (zero training)

**Question:** shown a real window warped to an in-envelope off-path pose `(dlat, dpsi)` — the exact
covariate shift the loop drives it into — does REF-C's 0.5 s plan MOVE to recover, or stay on its on-path
plan (blind)?

**Instrument:** `recovery_probe.py` on the 40-ep clean val, over the perturbation grid
`{(±0.5,0),(±1,0),(±1.5,0),(0,±3°),(0,±5°),(1,3°)}`. Warp = the instrument's operator (cross-checked
`_assert_warp_matches_harness`). Per window: `demand` = |recovery-target 0.5 s lateral − base-target|,
`response` = (perturbed plan − clean plan) projected on the demand direction, `recovery_ratio =
response/demand`. Estimator: episode-cluster bootstrap (`taniteval/ci.py`, 2000).

**Pre-committed reads (headline = `(dlat 1.0, dpsi 0)`):**
- **BLIND → GREENLIGHT P2b:** `recovery_ratio` mean **< 0.35** (CI upper < 0.5). REF-C keeps its on-path
  plan from an off-path view → the recovery signal is absent → augmentation should install it.
- **RECOVERS → STOP (FT predicted null):** `recovery_ratio` mean **> 0.7**. The plan already corrects; the
  closed-loop departure is downstream (controller / receding horizon), not planner recovery. Bank the
  measured ratio and the reframed bottleneck; **do not spend FT GPU**.
- **PARTIAL (0.35–0.7):** run P2b (the FT can still add recovery), but pre-note the ceiling is lower.

*Both outcomes are informative:* a low ratio is the mechanism the FT targets; a high ratio **redirects the
whole program's planner-closed-loop effort** away from planner training toward the controller, for ~0 GPU.

---

## 3. P2b — decoder-only recovery-augmentation FT + held-out corridor_departure

**One thing changed vs base REF-C training:** warped input + recovery target (everything else — frozen 90 M
encoder, anchored-diffusion decoder, anchors, diffusion noise, ego-dropout, Adam lr 1e-4 — is base REF-C).

**Episode-disjoint split (parity-safe).** FT on `--ft-slice 0:28` of the sorted 40-ep clean val; evaluate
corridor_departure on the **held-out** `--holdout-slice 28:40` (12 eps, never in the FT). This mirrors
Gate-1's leave-3-out memorization test, swapping ONLY the recovery-data *source* (synthetic-from-all-windows
vs real-from-15-junctions). *(Production variant, if the 2376-ep train corpus is mounted on eval: FT on the
train corpus — REF-C already trained on it, no parity break — and evaluate on all 40 val eps. The pre-reg
below is written for the on-pod val-split mechanism probe; the train-corpus variant only strengthens a WIN.)*

**FT config (frozen for the run):** `recovery_aug_ft.py`, decoder-only (~8.6 M trainable), steps 1500,
batch 16, lr 1e-4, envelope `lat_max 1.75 m / yaw_max 5° / clean_frac 0.30`, `λ_dev 0.5`, seed 0.

**Eval:** `eval_corridor_split.py` — base REF-C vs FT REF-C on the **same** held-out windows in one process;
**paired** episode-cluster bootstrap of Δ(base − FT). Primary metric **corridor_departure_rate @ 1.75 m**
(the abe82f1f instrument's own metric); guard metric **peak_xte**; context **closed_ade2s**. Strata:
overall / junction (|Δheading|≥10°) / longitudinal.

### The three pre-committed verdicts (primary = OVERALL held-out corridor_departure @1.75 m)

- **✅ WIN — data-efficient covariate-shift lever that GENERALIZES.** BOTH:
  1. paired Δ(base − FT) corridor_departure **> 0 and CI excludes 0** (FT departs less on held-out eps),
     **and**
  2. the guard holds: paired Δ(base − FT) **peak_xte ≥ 0** (CI not separated below 0) — recovery was **not**
     bought by over-steering.
  *Reading:* synthetic in-envelope recovery, generated from every held-in window, **generalizes to held-out
  episodes** — exactly where Gate-1's real-junction FT gave held-out Δ≈0. **Decision:** the lever is real
  and data-efficient → candidate for a full-corpus FT + an AlpaSim confirmation (never promoted on the
  low-OOD read alone). Bank the held-out Δ + CI.

- **⚠️ NULL — synthetic recovery does not transfer (measured bound, not failure).** paired Δ corridor_
  departure **CI includes 0**. *Reading:* even data-rich in-envelope recovery does not move held-out lane-
  keeping through a decoder-only FT → the residual is deeper (frozen-encoder features under-separate off-
  path, or decoder capacity). **Decision:** retire the *cheap* (decoder-only) version with a measured
  reason; the next candidate is encoder-in-the-loop recovery FT or the v4 analytic-grad direction — **not**
  another decoder-only attempt. This is the Gate-1-style honest bound and is worth as much as a WIN.

- **❌ HURT — recovery bought by over-steering.** paired Δ corridor_departure **> 0** (fewer departures) but
  Δ peak_xte **CI-separated < 0** (FT peak_xte higher), OR corridor_departure Δ **CI-separated < 0** (more
  departures). *Reading:* the Gate-1 high-deviation side-effect reappeared; the envelope bound + `λ_dev 0.5`
  under-constrained it. **Decision (pre-named, one further thing):** re-run with `λ_dev 1.0` and/or
  `lat_max 1.0` (smaller perturbations, tighter trust region); re-read the same predicate. Do not
  auto-escalate.

### Secondary reads (reported, not decisive)
- Junction vs longitudinal strata (does recovery help more where heading error dominates?).
- corridor_departure at the 1.0 m and 2.5 m thresholds (departure as a curve, not a knife-edge).
- FT-vs-base plan-shift on clean windows (must be ~0 by `λ_dev` — the on-path-preservation check).

---

## 4. Cost, safety, provenance

- **P2a:** ~minutes, inference only, ~0 GPU-day. Can end the FT question for free.
- **P2b:** decoder-only, frozen encoder, ~1500 steps ≈ **1–2 A6000-GPU-h** (`ESTIMATED`; frozen-encoder
  forward + tiny decoder backward). Eval ≈ the LaneKeep runtime (both arms, 12 eps).
- **Safety:** REF-C deployed ckpt is **read-only** (loaded, never written); the FT writes a NEW ckpt in a
  new dir. No `stack/` file is modified (harnesses live in `incoming/`), so `pytest` is unaffected. Encoder
  frozen ⇒ the world model cannot be degraded. `gpu_lock refc-cl-improve`, released at end; kill by explicit
  PID only; poll `abe82f1f`'s terminal marker before acquiring.
- **Estimator:** episode-cluster / paired bootstrap (`taniteval/ci.py`) — the program's decision-grade
  interval, never the deprecated overlapping-holdout.
- **Honest frame (binding):** LANE-KEEPING / on-policy drift at low OOD; **structurally NOT** off-road or
  collision (map/agent-free source). A WIN here is a covariate-shift *mechanism* result, a candidate for
  AlpaSim, not a safety rate.

**Staged launch commands (NOT executed):**
```
# P2a — probe (greenlights/kills the FT for ~0 GPU)
PYTHONPATH=/root/TanitAD/stack:/root/TanitAD/stack/scripts python3 recovery_probe.py \
  --refc-ckpt /root/models/refc-base-30k/ckpt.pt \
  --val-dir /root/valdata/physicalai-val-0c5f7dac3b11 --out recovery_probe.json
# P2b — FT (only if P2a greenlights) + held-out corridor_departure
PYTHONPATH=/root/TanitAD/stack:/root/TanitAD/stack/scripts python3 recovery_aug_ft.py \
  --refc-ckpt /root/models/refc-base-30k/ckpt.pt \
  --val-dir /root/valdata/physicalai-val-0c5f7dac3b11 \
  --ft-slice 0:28 --steps 1500 --out /workspace/refc-recovery-ft
PYTHONPATH=/root/TanitAD/stack:/root/TanitAD/stack/scripts python3 eval_corridor_split.py \
  --base-ckpt /root/models/refc-base-30k/ckpt.pt \
  --ft-ckpt /workspace/refc-recovery-ft/ckpt.pt \
  --val-dir /root/valdata/physicalai-val-0c5f7dac3b11 \
  --holdout-slice 28:40 --out corridor_split_results.json
```

**Nothing in this pre-registration was launched, and no pod was touched.**
