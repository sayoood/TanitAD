# PREREG_LONHEAD_C -- a CLEAN-SPLIT lon-decision head on refcv8's frozen z_tac, used to sharpen the speed profile only (Master Mind, 2026-10-26 ~08:12 CET)

Sealed before any fit, label build, eval read or roll of this lever exists. The committed next lever of SSGATE (PREREG_SSGATE s5: *"ACCURACY
(train the lon head toward the z_tac probe on the clean L1C split)"*) and of PREREG_CHANNELS_D2 s3 (all channels ABSENT -> train the lon
decision).

## 0. Why (disclosed, MEASURED)
* The fan holds the answer (GT-oracle pick MD 0.075 / FD 0.057) but every post-hoc re-weighting moves refcv8 ALONG one MD/FD frontier
  (PLAN_REFCV9 s12). The explicit channels (lead, curve, crosswalk) add no decel information over the lon head + v0 (H-R9-LEADINFO-1,
  H-R9-CHANNELS-1). The trunk does: a z_tac linear probe reads decel AUC 0.762 vs the head's 0.685 (L1).
* Sharpening the head's OWN argmax onto the speed profile only (FACTORED_DEPLOY8, A5 PASS; SSGUARD REPLICATED) catches decels (MD -0.08) at
  the head's accuracy. This lever keeps that composition and replaces the label with a better head's.

## 1. The head `LH` (fit on FRESH rows only)
* Input: refcv8 EMA-48,852's frozen `z_tac` (FRESH: `F:/l1c/bank/ztac_fresh.npz`; EVAL: the L1 eval banks used by SELCOR's variant Z) + v0/10.
* Target: the per-window GT LON6 class in refcv8's own v7 space (classes 0,1,3,4,5,7 -- the space of `nav_true.tacv6_lon_logits`), built by
  EXACTLY the function refcv8's trainer uses for its lon target (quote file:line), from the fresh kit's v8 label release + egomotion.
  **K-LABEL:** the same function on eval139 reproduces the battery's GT lon label on every eval window (exact), so train and eval targets agree.
* Model: multinomial logistic (lambda in {1e-3 .. 100}) OR a one-hidden-layer MLP (256, ReLU; weight decay in {1e-4, 1e-3, 1e-2}; epochs by
  early stop on the inner fold), the variant and its hyper-parameter chosen by 5-fold episode-grouped CV on FRESH rows, criterion = mean
  cross-entropy. Exactly ONE configuration is frozen (`LH_FROZEN.*`), its sha256 registered by the Master Mind before any eval-side output.
* **DER control head:** the same recipe on targets deranged across FRESH episodes (seed 0), frozen and registered the same way.

## 2. Eval read and rolls
* **K-HEAD (go / no-go, descriptive):** decel AUC of LH on eval139 S6 compliant windows, `P(decel classes)` renormalised exactly as the
  0.685 reference (`H-R9-LEADINFO-1` D4). **If LH's point AUC <= 0.685 the rolls are NOT run**: report HEAD-NOT-BETTER and stop (the lever
  needs a better head; spending GPU on a worse one measures nothing).
* Otherwise three GPU rolls on the dev box, SELFSHARP's machinery with only the label source swapped (`l2_roll.py --arm oracle --labels-npz`,
  one-hot +-20 on the LON6 columns; os-only; per-frame feed; EMA-48,852; 4,754 windows): **LHS s0**, **LHS s1** (labels = LH argmax), **DER s0**
  (labels = DER argmax). Each composed and staged EXACTLY as `FACTORED_DEPLOY8` (A5: `RT.eval_polyline(RT.polyline_geometry(P_os8),
  arc(P_sharp8))` + the registered per-frame RETIME table) -> arms `LHS_FACT8`, `DER_FACT8`.

## 3. Bars (S6 compliant; paired episode-cluster bootstrap B 2000 seed 0 over EPISODES; same inference-seed draw)
* **F1 (beats SELFSHARP on the frontier):** vs `FACTORED_DEPLOY8`: FD separated BELOW and MD NOT separated above.
* **F2 (beats H2cn on the frontier):** vs `H2CN_DEPLOY8`: MD separated BELOW and FD NOT separated above.
* **G:** no S6 row (`PR.METRICS` + `ade_m`) separated WORSE than BASE-os.
* **A (attribution):** on the frontier bar that passes, `LHS_FACT8` minus `DER_FACT8` is separated in the same direction (s0).
* **PASS** = (F1 or F2) and G and A at s0; **REPLICATED** iff the same F-bar and G hold at s1. Beside: compliance row, E1 / E3, the four
  families, LH's per-class confusion on eval (T0, diagnostic).

## 4. Tier / admissibility
T1 self-action OPEN LOOP on logged eval139 frames (not driving performance); one refcv8 training seed; one head fit; inference seeds 0 / 1
never pooled. LH reads the trunk's own features + v0 at t0 (vision-only + admissible ego); it is the tactical DECISION head, not the situation
classifier, and it feeds the speed profile, not a goal input. Deploying it = a 0.1-M-parameter post-hoc head + the factored two-pass decode
(latency unmeasured; a PI decision before any ship).

## 5. Committed next steps
* **PASS (+ REPLICATED)** -> the first lever today that moves BEYOND the frontier: a SHIP-R8 amendment on LHS_FACT8 needs a PI go; refcv9 trains
  the lon head on a clean split as its own component (design input with this number).
* **HEAD-NOT-BETTER** -> the clean split does not reproduce the probe's headroom: the lon decision needs NEW supervision (richer labels or
  more fresh clips), a data decision for the PI.
* **Head better but F1 and F2 FAIL** -> head accuracy is not the binding constraint for the factored composition: the progress the sharpened
  class implies is (next: a class-conditioned progress head on the fresh split, pre-registered first).
* **A FAIL** -> the gain is the head's class RATE, not its per-window decisions: report it; no ship.
