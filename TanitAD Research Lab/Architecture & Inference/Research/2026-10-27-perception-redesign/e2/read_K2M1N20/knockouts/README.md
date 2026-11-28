# E2 NO-GO diagnosis — three zero-training planner knockouts of K2, and the map dose that ran 5× above its band (MM 4a3d05, 2026-11-12)

**Question.** E2 (K2M1N20 vs K0, 6,000 steps each) is NO-GO; the corrected re-read (`../reread_dtfix/`, GOALS H-R9-E2-DT) puts the planner
regression in the LATERAL family: K2 − K0 cross-track +7.4 / +5.0 mm, curvature +0.0003 1/m, yaw rate +0.0011 / +0.0009 rad/s, LON accel
+0.0098 / +0.0108 m/s², all separated at inference seeds 0 / 1. Is it carried by an inference-time path that only K2 has, or is it trained in?

**Estimator / tier / class.** Every K-vs-K row: paired episode-cluster bootstrap (`taniteval.ci`, B 2000, seed 0) over the 70 DEV episodes of
eval139 (even half; HOLD not read), T1 self-action open loop, the same per-frame (`pf`) battery dump as `E2_K2_pf`. DIAGNOSTIC: one training run per
arm (H-ESTIM-SEED-1), so these rows attribute within K2; they are not lever effects between trained arms. Inference seeds 0 and 1 are both read.

## 1. The three knockouts (`results/`)

Each knockout is applied to the trained K2 checkpoint inside the E2 battery (`R9E2_ABLATE`, census in every `roll_s*.json`, `results/census/`):

| knockout | landed | what it zeroes | reached the planner? (`results/moved.txt`) |
|---|---|---|---|
| `r9_map_chan_zero` (W-MAP-2) | 60b3ef23 | the planner's map-channel input gates (Σ\|w\| 1.26 before) | yes: 2,035 / 2,036 of 2,397 windows moved > 1 mm, max 0.50 / 0.85 m |
| `r9_box_tok_zero` (W-BOX) | 60b3ef23 | the box-token gates (trained to \|g\| ≤ 0.0147) | barely: 0 windows > 1 mm, max 0.3 mm (the planner hardly opened W-BOX) |
| `r9_drv_critic_zero` (W-COST critic) | e527e11c | `r8_drv.w` 0.21958 → 0 (no-critic argmax, exact) | yes: the selection flips on 62 windows (> 1 cm, max 5.2 / 4.4 m); 2,335 bit-identical |

Knockout − K2 (seed 0 / seed 1), `results/score_e2abl.json`:

| row | map-zero | box-zero | critic-zero |
|---|---|---|---|
| cross-track m | +0.0000 / within ±0.0007 | < 0.00005 | −0.0000 [−0.0014, 0.0012] / +0.0003 [−0.0014, 0.0019] |
| LON accel m/s² | +0.0002 / n.s. | 0.0000 | +0.0026 [−0.0000, 0.0061] / **+0.0030 [0.0005, 0.0059]** |
| cross, straight | +0.0001 / n.s. | 0.0000 | +0.0013 / **+0.0011 [0.0003, 0.0020]** |

Every knockout − K0 row equals K2 − K0 within its CI. **None of the three removes the regression.** The jitter split
(`results/lat_jitter.json`, signed lateral error at the 2.0 s waypoint): K2 − K0 frame-to-frame |Δe| **+12.3 / +13.0 mm** and within-window lateral
2nd difference +2.4 mm (both separated, both seeds), per-episode bias NOT separated; map-zero and box-zero leave both untouched, and **critic-zero
makes the jitter WORSE** (|Δe| +2.9 [0.1, 5.8] / +4.7 [0.1, 11.2] mm, wiggle +0.7 mm, separated at both seeds): the critic damps it.

### Four families (knockout vs K2; K2 − K0 is in `../reread_dtfix/`)

* **LONGITUDINAL** — accel: null for map-zero / box-zero; critic-zero +0.003 m/s² (separated at seed 1). Distance keeping (headway / TTC) NOT
  COMPUTED here: it needs the lead-block join of the full E2 read, which the knockout rolls (os-only battery) do not run; n = 0 by design.
* **LATERAL** — cross-track, curvature, yaw rate, straight / fast cross: null for all three (table above); the jitter split as stated.
* **TACTICAL** — `results/tactical_knockouts.json`: lateral behaviour decoder acc 0.9273 / κ 0.7292 and v7 heads 0.9263 for K2 and every
  knockout at both seeds; LON decoder 0.6378 (map-zero 0.6373). Identical by construction: the tactical heads sit upstream of all three wires.
* **STRATEGIC** — PAUSED (PI ruling 2026-09-19, `PI_DECISION_QUEUE.md` item 25). Not a pass.

**⇒ The lateral regression is trained into the shared network, not carried by any K2-only inference path.**

## 2. What K2 trained differently: the map term took 77–83 % of the trunk gradient (`diag_dose/`)

K2 − K0 argv = `--r9-map-banded` (+ `--r9-map-banded-near-m 20`), `--r9-hist-supervise-geom`, W-MAP-2, W-BOX, the drivable critic
(`--w-r8-drivable 0.06472`), and **`--w-map-hires 1.0 → 25.0`** (the re-dose, `dagland/e2/PREREG_K2_REDOSE_2026-11-10.md`, PI "Yes, re-dose").
E1's M1N20 — the map-banded arm that earned KEEP — trained at **w 1.0**; no E1 arm ever trained at 25.

The trainer logs the launch gate's own share estimator (`gs_<group>_{norm,proj}_*`, one batch per `--grad-share-at` step) in `metrics.jsonl`.
`code/dose_from_telemetry.py` recovers |a| (map), |b| (everything else) and ⟨a,b⟩ from it (reconstruction exact to 4 digits) and solves the
prereg's own s(k) for the weight that puts the share at 0.05 / 0.10 / 0.15 (design 2.2's dose band 0.05–0.15):

| K2 at w 25 | trunk share | layer2 share | map \|g\| per unit w (trunk) | w for trunk share 0.10 |
|---|---|---|---|---|
| step 50 (the gate's smoke batch, prereg) | 0.104 | 0.130 | 2.09 | ~24 (prereg table: w 20 → 0.073, w 25 → 0.104) |
| step 1,000 | 0.388 | 0.449 | 4.68 | 10.0 |
| step 3,000 | **0.809** | 0.669 | 10.80 | 3.96 |
| step 5,000 | **0.770** | **0.834** | 8.24 | 4.64 |

K0 (w 1, unbanded): trunk 0.0004–0.0036. ⇒ The re-dose was correct **for the batch it was computed on** and then left the band within the first
1,000 steps: the map gradient per unit weight grew ~4–5× while the rest shrank. **No constant weight satisfies both the gate's step-50 floor
(trunk ≥ 0.05 needs w ≥ 15.7) and the design band at the trained state (≤ 0.15 needs w ≤ 5.0–5.8 at steps 3,000–5,000).** Trunk-level
aux/traj gradient-norm ratio (conflict detector, every 50 steps, steps ≥ 3,000): K2 227 vs K0 117; trunk aux·traj cosine −0.017 vs +0.006.

Also measured, not explained here: K2's map is WORSE than K0's (clause (a)) although the map term owned the trunk; a 5×-overdosed banded loss is the
first candidate, not a demonstrated cause.

## 3. Next lever (pre-registration and cost go to the PI)

**HYPOTHESIS:** the overdosed map term (share 0.77–0.83 vs a 0.05–0.15 design) degraded the planner's trunk features (lateral jitter, LON accel) and
did not buy map quality. **Discriminating arm:** K2 with the map dose HELD in the design band during training (a share-target controller that
re-solves s(k) from the trainer's own grad-share probe; the step-50 gate still reads w 25), everything else = K2, 6,000 steps (~17 Thor-h),
read by the unchanged E2 rule vs K0. Both outcomes committed: lateral rows and the map primary return to ≤ K0 ⇒ the dose was the defect and
the arm is the E2 candidate; they do not ⇒ the dose is eliminated and the next suspects are `--r9-hist-supervise-geom` and map-banded itself.

## Manifest

* Here: `results/` (scores, jitter, reachability, tactical, censuses, the 6 `roll_s*.json`, `dumps.sha256`), `diag_dose/` (telemetry solves for
  K2 and K0), `code/` (every script that produced them, incl. the Thor roll `e2abl_roll{,_v2}.sh`). SHA256SUMS covers every file.
* Dumps (63 MB, 864 files, hashed in `results/dumps.sha256`): dev box `C:/Users/Admin/r9e2_read_K2M1N20/abl/E2_K2abl_{mapzero,boxzero,drvzero}_pf/`
  and Thor `/home/nvidia/r8ns/r9e2/runs/E2_ABL_K2M1N20/` (sha256-verified equal after each pull). K2 / K0 reference dumps: `../reread_dtfix/`.
