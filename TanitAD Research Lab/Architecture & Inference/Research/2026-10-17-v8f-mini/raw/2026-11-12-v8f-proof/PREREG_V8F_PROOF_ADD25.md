# PREREG_V8F_PROOF_ADD25: HEADWAY. Distance keeping in the rollout on the model's own actions, read from the banked dumps and the lead labels of the evaluation episodes. No training, no new rollout.

Frozen before any number of this card is computed. Addendum to `PREREG_V8F_PROOF.md`. Tier T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance). Evidence class of every result: MEASURED once run. This card closes a gap of every T1-kin read of this programme so far: the LONGITUDINAL family was reported with speed and along-track error only, and headway and time gap were stated as not measured because the dumps carry no lead block. STRATEGIC: PAUSED (PI item 25).

## 1. What is known before it

* The T1-kin dumps (`v8f_t1_eval.py`, 141 evaluation episodes, 2,399 windows per arm) hold the rolled ego motion of every window beside the logged one; the four-family reader `spatial/q3/q3_read.py` reads speed, along-track, cross-track, heading, curvature, yaw-rate and mode errors from them.
* The lead labels of the evaluation episodes exist on the dev box (the label source of card ADD20: validity flag and gap of the lead from the offline obstacle tracks, 0 to 60 m).
* Six paired seeds exist for HIER, FLAT (HIERREP) and G (cards ADD12, ADD22); two seeds for P; the six arms of card ADD24 are training.

## 2. Definitions (fixed here; what cannot be fixed without reading the dump layout is fixed in `HEADWAY_DEFS.md` before any arm number is computed)

* Window: a T1-kin window of an evaluation episode with start frame S0 and rollout steps k = 1 to K at 0.2 s (K = the horizon of the dump, 6.0 s where the dump holds it).
* Lead track of a window: the logged gap gap_l(k) to the lead at frame S0 + k, valid while the validity flag is 1, the gap is inside 0 to 60 m and no step between consecutive frames is above 3.0 m (the continuity rule of card ADD20; a break ends the track). A lead window has a valid track at S0 and for at least the first 10 steps (2.0 s). Scored steps: up to the first break or K.
* The lead does not react: it keeps its logged track. Model gap g_m(k) = gap_l(k) + a_l(k) - a_m(k), with a(k) the progress along the logged path (arc length of the projection on the logged path) of the logged ego (a_l) and of the rolled ego (a_m).
* Time gap tau(k) = g(k) / max(v(k), 0.5 m/s) with the speed of the same trajectory; for the logged driver tau_l from gap_l and the logged speed.
* Unsafe step: g below 2.0 m or tau below 1.0 s. Contact step: g at or below 0 m.
* Cells, per arm, over lead windows:
  * U (primary): share of lead windows with at least one unsafe step of the model among the windows where the logged driver has none (model-induced unsafe rate).
  * C: share of lead windows with a contact step of the model.
  * gap bias at 2.0, 4.0 and 6.0 s: mean of g_m - gap_l (below 0 = the model closes in more than the driver) and its absolute mean (gap error).
  * time-gap error: mean absolute tau_m - tau_l at 2.0 s and over the scored steps.
  * F: share of lead windows where g_m - gap_l is above +10 m at the last scored step (the model falls far behind the driver).
  * For reference, the unrestricted rates of the logged driver by the same rule (unsafe and contact).
* Estimator: paired window differences, two-level bootstrap (seeds, then episodes), B 2,000, rng 0, the engine of `q3_read.py`; the rule of card ADD11: an effect needs the same sign in all seeds read, an interval that excludes 0, and a size outside the six-seed spread of the baseline for the cell.

## 3. Questions and outcomes (both committed)

* Q1, HIER against FLAT, seeds 0 to 5, cell U: `HIER-KEEPS-DISTANCE` if U is lower for HIER under the rule; `FLAT-KEEPS-DISTANCE` if higher; else `DISTANCE-EQUAL`.
* Q2, G against HIER, seeds 0 to 5, cell U: `GRID-DISTANCE-BETTER`, `GRID-DISTANCE-WORSE` or `GRID-DISTANCE-EQUAL`.
* Q3, each arm type of card ADD24 (LS, LSH, LSD) against G on seeds 0 and 1, spread of G over seeds 0 to 5, cell U: `SUP-DISTANCE-BETTER`, `SUP-DISTANCE-WORSE` or `SUP-DISTANCE-EQUAL`. Read when those arms are pulled; Q1 and Q2 do not wait for it.
* Level statement, names nothing: U, C, F, gap bias and time-gap error of every arm with its interval, beside the logged driver.
* Every other cell is reported for every pair with the same rule and names nothing.

## 4. Controls (all must pass before any outcome is named)

* h1 identity: the logged trajectory given to the reader as the model reads g_m equal to gap_l (max abs difference at most 1e-6 m), U 0, gap bias 0.
* h2 planted regression: the logged trajectory advanced along the path by 0.5 m/s times the elapsed time reads a gap bias of -1.0 m at 2.0 s and -3.0 m at 6.0 s within 1e-3 m, and the pair (planted against logged) is named worse on U by the rule machinery.
* h3 swap is minus and self is zero for every pair.
* h4 alignment of windows and labels: the logged ego speed of the dump at S0 equals the ego speed of the label source at the mapped frame within 0.05 m/s for every lead window; with the mapping shifted by 5 frames the check must FAIL.
* h5 counts: the number of lead windows and of episodes with at least one lead window, stated before any arm comparison. Below 200 lead windows or below 20 episodes the read names nothing and is reported only.
* h6 the same windows in every dump read (the control of `q3_read.py`).

## 5. Committed next steps

* `HIER-KEEPS-DISTANCE`: it enters the dominance evidence beside HIERREP, with its limits; the distance cell becomes a standing row of every later T1-kin read.
* `DISTANCE-EQUAL` or `FLAT-KEEPS-DISTANCE`: reported as written. Distance keeping is then not where the hierarchy helps on this rig, and the lever is the lead state (card ADD24 and its stage 1).
* A model-induced unsafe rate U of the baseline above 5 % of lead windows, whatever the names: a distance rule for the safety check (a veto of plans whose gap under the non-reacting lead falls below the bound), first with the label gap as an upper bound of what a perfect lead state gives, own card, pod time; then with the gap of the supervised head of card ADD24.
* In every case the lead block is added to the standing four-family table.

## 6. Cost and limits

* CPU only, banked dumps and labels, under one hour.
* The lead does not react to the rolled ego (log replay); a gap below the bound in the rollout is a statement about the plan against the logged lead, not about a collision in traffic.
* The lead labels come from offline obstacle tracks through the label join; windows without a labelled lead say nothing about distance keeping.
* One corpus, one evaluation set; a deterministic planner, so inference variance is not sampled.
