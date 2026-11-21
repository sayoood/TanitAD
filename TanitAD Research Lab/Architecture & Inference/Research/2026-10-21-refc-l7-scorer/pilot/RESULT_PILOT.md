# H-L7-1 PILOT (PREREG_L7_SCORER.md sec. 2) -- trained; on held-out TRAIN episodes both scorers pick WORSE than refcv8's own pick

Filed 2026-10-22 19:52 Berlin (clock read). Master Mind. PI GO 2026-10-22 ~18:00 Berlin, in chat: "PILOT now on CPU".
Evidence class: MEASURED (`*/train_log.json`, `*/train.log` in this folder). **The PILOT never decides** (PREREG sec. 2):
this is a training-split validation read, not the registered eval139 read, and no bar of sec. 3 is scored here.

## What ran

`l7pilot_run.sh` (this folder) = `PILOT_COMMANDS.md` sec. 2 verbatim: L7-E and L7-F at training seeds 0 and 1, trained on
train139 s0 + s1 (8,730 train windows after the 10 % episode-disjoint validation slice, seed 20261021), CPU, dev box,
18:04-18:12 Berlin, rc 0 each (73-127 s per run). Code = the landed package at c61f016 (the preflight fixes).

## Validation (980 windows = 14 held-out TRAIN episodes x 2 inference seeds), at each run's best epoch (min val loss)

| run | best epoch | val loss | **pick cost (m)** | own pick (m) | oracle in fan (m) | AUC comf / nc / ttc / dac |
|---|---|---|---|---|---|---|
| L7-E ts0 | 8 | 5.2266 | **1.5892** | 0.99895 | 0.35874 | 0.965 / 0.698 / 0.544 / 0.735 |
| L7-E ts1 | 7 | 5.2469 | **1.5921** | 0.99895 | 0.35874 | 0.958 / 0.704 / 0.591 / 0.733 |
| L7-F ts0 | 6 | 4.9890 | **1.0714** | 0.99895 | 0.35874 | 0.956 / 0.729 / 0.628 / 0.766 |
| L7-F ts1 | 2 | 5.0163 | **1.1627** | 0.99895 | 0.35874 | **0.926** / 0.719 / 0.631 / 0.757 |

* Both arms, both training seeds: the learned pick is WORSE than the shipped E9 pick (L7-F by +0.07 / +0.16 m, L7-E by
  +0.59 m). L7-F already reads the model's own `sel_score` and `refined_logits` (`refc_scorer.model_features`), so the
  sec. 4 lever "(2) score the refined object as an input feature" is already inside it.
* K7 (`comf` AUC >= 0.95): passes for L7-E (both seeds) and L7-F ts0; **fails for L7-F ts1 (0.926)** at its best epoch.
* The training floor is large: L7-F ts0 vs ts1 differ by 0.09 m pick cost, more than L7-F's distance to own pick.
* The eval139 PILOT read (permitted once, never deciding) was NOT spent: the validation result already answers the
  PILOT's question (no learned-pick advantage at this bank size), and an eval139 number would add no decision.

## What it means (ESTIMATED reading, labelled)

On 8,730 detached-token windows, a 155 k-parameter scorer trained to imitate the closest-to-GT candidate does not
recover even the model's own ranking. This is the PILOT-sized version of the sec. 4 "P1 fails" outcome. Two things
it cannot separate: bank size (the registered full bank is ~8x larger: train4572 at 1-in-4, two seeds) and the
detached-token ceiling (the sec. 4 lever (1), a trunk-gradient scorer).

## Next (sec. 4 order, and what blocks each)

1. **The registered full-bank read** (train4572 capture, about 6.2 Thor GPU-h): queued on Thor AFTER NT A1, by the
   PI's choice on 2026-10-22, so A1 is not delayed. Blocked on Thor compute until then.
2. **Trunk-gradient scorer** (sec. 4 lever 1): needs trainer integration and GPU, so it is a PI compute decision.
3. L7-S (scene tokens; the capture banked `l7cap_scene.npz`) is a registered arm, but `train_l7.py` builds only E / F,
   and the PILOT clause admits E / F only. Running it on the pilot bank would need a registered amendment first.

Not in git: the four `best.pt` (627,909 / 631,493 B) at `C:/Users/Admin/refcv8_pf/l7pilot/<run>/best.pt`, md5
L7E_ts0 30c8c12e…, L7E_ts1 143422ee…, L7F_ts0 f64a10dd…, L7F_ts1 2d87f2f3….
