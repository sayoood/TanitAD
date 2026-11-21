# SPEC — the withheld-bank panel on the v7-tiny rig (`H-EGO-LIT-4`)

Written BEFORE any arm was launched (TanitAD_ValidateAIDesign §1). The hypothesis, arms,
outcomes and controls are those registered in `Project Steering/GOALS_AND_CLAIMS.md`
row `H-EGO-LIT-4` and designed in
`…/2026-09-05-ego-input-literature/RESULT.md` §3.6; this file adds only the reading rules
the registration left implicit, each committed here in advance.

```yaml
hypothesis: H-EGO-LIT-4          # exists in GOALS_AND_CLAIMS.md (row 76); sibling H-EGODROP-PRED
one_variable: "the speed SOURCE a WITHHELD row's anchor bank is rolled at after step N
              (`--withheld-bank`), one arm per source vs A0; A3 moves `ego_dropout` instead"
held_constant:
  - rig: the H-ECHO-8 rig verbatim — `--arm hier --size tiny --data-root <epcache>
    --episodes 48 --steps 2000 --batch 12 --lr 1e-4 --warmup 250 --seed 0` (256x256 epcache,
    kin3 tactical vocabulary, v1 nav derivation — what the epcache path carries)
  - refcv4b's shipped levers: `--anchors refc_anchors_6s_v0cond_alat_117.pt` (md5
    297f6f1db52f6a56094846b0d7f71ed9 = the live pod's anchors.pt) `--n-anchors 117
    --anchor-v0-conditioned --anchor-control-units alat --sel-accel-max 2.0
    --ego-state-inject` (X15 implied), NO `--echo-base` (the live config; H-ECHO-8 §6.4)
  - `--ego-dropout 0.5` on every arm but A3
  - the warm-up N is SHARED by A1 and A2 (so A1 vs A2 is one variable: the source)
  - scoring: the same 640 val windows from the same 40 episode-disjoint val episodes,
    `torch.Generator().manual_seed(20260904)` (H-ECHO-8's draw), for every arm
success: "A1 (pred) beats A0 on the WITHHELD-regime longitudinal family (2 s speed MAE and
          along-track MAE on the selected trajectory, paired episode-cluster bootstrap, CI
          excluding 0) AND A1's kept-regime source-ablation verdict is READS_BOTH (scene
          degradation separated with rel >= 0.05 AND ego separated) AND A2 (random) does
          NOT show the same gain (A1 - A2 paired, separated in A1's favour on the same family)"
failure: "any of: A1 reads ECHOING (scene delta not separated) => refuse (c); A1 ~ A2 on the
          withheld longitudinal family (not separated) => bank geometry is not the binding
          constraint; A4 >= A0 on the echo instrument AND the families => retire the
          per-window roll; A3 better families but lost separation => PlanTF signature,
          keep 0.5; A2 >> A0 => 10 m/s is a biased choice, re-run A0 at the marginal's mean"
controls:
  - constant_only            # must read the no-information value (its own ADE)
  - raw_pixel_floor          # ridge, n_pix=8 => d=192 < n_fit; n and d printed
  - ha, ha0, ha0_ext         # kinematic references, recomputed on the SAME windows
  - source_ablation constant predictor   # must read scene 0.0000 / ego 0.0000 EXACTLY
  - A2 random-marginal bank  # right distribution, zero information (the blindness control)
  - shuffle-the-predicted-speed at eval (A1 only)   # the gain must vanish (H-EGODROP-PRED)
  - A5_regress = A0 + --ablate-frames   # deliberate regression: the gate MUST fail it
splits:
  fit: "first half of the 640 scored windows (ridge lambda for the pixel floor and the D4
        probe is chosen on an inner val carved from FIT)"
  val: "carved from FIT only"
  test: "second half — scored, never tuned on; the paired families use every window (no
         hyper-parameter is fit on them)"
```

## Arms (six launches; the pre-registered five plus the gate control)

| arm | flags beyond COMMON | one variable vs A0 |
|---|---|---|
| A0_fixed | `--ego-dropout 0.5 --withheld-bank fixed` | — (the control; refcv4b's shipped policy) |
| A1_pred | `--ego-dropout 0.5 --withheld-bank pred --withheld-bank-warmup N` | withheld rows rolled at the model's OWN detached vision-only `g_tac` 2 s speed, clamp [0, 35], after step N |
| A2_random | `--ego-dropout 0.5 --withheld-bank random --withheld-bank-warmup N` | withheld rows rolled at a draw from the TRAINING marginal of v0 after step N (the blindness control; vs A1 the ONLY difference is the source) |
| A3_drop25 | `--ego-dropout 0.25 --withheld-bank fixed` | the withholding rate |
| A4_none | `--ego-dropout 0.5 --withheld-bank none` | EVERY row (kept, withheld, eval) at 10 m/s — the field's speed-blind fixed vocabulary, same 117-control family |
| A5_regress | `--ego-dropout 0.5 --withheld-bank fixed --ablate-frames` | GATE CONTROL: an echo by construction; never a model |

**Reading rule for N (committed):** the first logged step at which the **5-row running mean**
of A0's `withheld_speed_mae` (the model's own 2 s speed on withheld rows vs the GT 2 s speed,
logged every 50 steps) is **< 2.5 m/s**; if never, **N = 667** (⅓ of 2,000). Read by
`raw/warmup_from_log.py`, written to `raw/arms/WARMUP_N.txt` before A1 launches.

## What is scored (all mandatory; ADE is one row of the families, never the gate)

1. **The H-ECHO-8 separation instrument** — `tanitad.eval.echo_gate.source_ablation_test` on
   `g_tac` in the KEPT regime (the deployed regime), paired, n_boot 500, `min_degradation`
   0.05: verdict `READS_BOTH` / `ECHOING` / `IGNORES_EGO` / `READS_NEITHER`; plus
   `ego_intervention_test` (structural) and `echo_gate` (gate 1: `ha`, `ha0`, `ha0_ext`,
   `constant_only`, margins 0.10) — the `assert_not_echoing` verdict per arm. The scene half is
   ALSO run in the WITHHELD regime (ego derangement is vacuous there and is reported as such).
2. **The four families on KEPT and WITHHELD rows** (the EGODROP protocol): every val window is
   forwarded twice — ego block measured (keep = 1) and withheld (keep = 0, values zeroed) —
   and the SELECTED trajectory's families come from `taniteval.tools.refav1_arm._components`
   (the banked instrument; 2 s grid = slots 0.5/1.0/1.5/2.0 s, dt 0.5) paired vs A0 per regime
   with the episode-cluster bootstrap; the goal-row families (`echo_gate.trajectory_families`)
   are reported alongside. Longitudinal = speed / along-track / accel MAE; lateral = cross-track
   / heading / yaw-rate MAE; tactical = trajectory-derived lat/lon decision agreement AND the
   factored-head accuracy vs the majority-class control; strategic = ABSENT WITH REASON (no LAN
   label on the epcache rig; the g_str head is unsupervised here).
3. **The withheld-row ceilings by v0 band**: oracle-in-vocabulary of the bank the arm actually
   decoded in the withheld regime, and of the bank rolled at fixed 10 m/s / the model's own
   prediction / the true v0 (the LEAK bound — reported, never a design), by the EGODROP v0 bands
   (0-5, 5-10, 10-15, 15-20, 20-25, 25+ m/s); plus the trainer-`a_star` share of straight-ahead.
4. **D3** — `‖∂(x_T, y_T)/∂ s_0‖`, the 6 s endpoint of the selected trajectory w.r.t. the
   measured ego block (v0 through both its entry points, a, yaw-rate, curvature), one backward
   pass per coordinate per batch, kept regime, mean over windows; and **D4** — the plan-
   predictability probe: a ridge from (v0, a0, yaw-rate) alone to the arm's 8-slot plan vs the
   same probe on the HUMAN plan (GT waypoints), λ chosen on the inner val, scored on TEST;
   a plan MORE self-predictable than the human's is the copycat signature.

Controls that must read known values are printed with the panel: `constant_only` at its own
ADE; the source-ablation constant predictor at exactly 0.0000 / 0.0000; `ha` ≡ `ha0_ext` within
float tolerance where the harness holds (a, κ) — and A5_regress must be FAILED by the gate.

## Outcomes committed in advance (verbatim from the registration)

- A1 stays `READS_BOTH` and beats A0 on the withheld-row families/ceiling ⇒ **adopt (c)** for
  refcv5 with the attribution caveat recorded in the design doc and the registry.
- A1 reads `ECHOING` ⇒ the predicted-speed bank re-opens the echo through the tactical head ⇒
  **refuse (c)**; the bank problem is then solved by (e) or lived with under (a).
- A4 ≥ A0 on the echo instrument AND on the families ⇒ the v0-conditioned bank is not
  load-bearing; **adopt the field's design** and retire the per-window roll.
- A3 beats A0 on the families but loses separation ⇒ the PlanTF signature; **keep 0.5**.
- A2 ≈ A0 ⇒ the harm is "any speed-blind bank", not "10 m/s" ⇒ (b′)-style fixes are dead.
- A2 ≫ A0 ⇒ 10 m/s is a biased choice; report the marginal's mean and re-run A0 there.
- **If A1 gains and A2 gains the same, the gain is not information** (the control reads it).
- **If the gate does not FAIL A5_regress, the panel is VOID** (OUTCOME IV), not the model.

## Attribution risk, named in advance

Under `pred` the withheld-row longitudinal decision becomes downstream of `g_tac`: the
withheld-regime longitudinal family then scores the goal head and the anchor classifier
JOINTLY. The panel separates them with (i) A1's withheld regime re-scored under the FIXED bank
(same weights, bank swapped back — isolates what the weights learned from what the bank gives)
and (ii) the shuffle control (same weights, `pred` bank at a PERMUTED row's speed — the gain
must vanish). Both are reported next to the headline delta.

## Scope stamps

Every number is **RIG scope** (tiny rung, 48 non-parity training episodes, 2,000 steps, one
seed): it validates a DESIGN and a GATE, never a model, and nothing here may enter
`MODEL_REGISTRY.md` (H-SCALE-2). Tier: the families are **T1-style self-action open loop on
the goal/selection heads** (`open-vs-closed-loop` ruling), the ceilings are **T0 model-free**.
Estimator everywhere: `taniteval.ci.paired_episode_cluster_bootstrap` (n_boot 2000 for the
families, 500 for the ablation probes), never `overlapping_holdout_se`.
