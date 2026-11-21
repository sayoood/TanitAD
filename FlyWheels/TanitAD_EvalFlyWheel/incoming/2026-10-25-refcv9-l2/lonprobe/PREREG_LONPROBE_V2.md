# PREREG_LONPROBE_V2 — the permutation-null control the v1 run lacked (Master Mind, 2026-10-25 ~19:40 CET)

Sealed before v2 runs. ⚠️ Not independent of v1: PROBE / HEAD / PLAN AUCs were SEEN in v1 (PROBE 0.762, HEAD 0.685, PROBE − HEAD
+0.077 [+0.007, +0.147] at both seeds). v1's registered verdict stands as **INCONCLUSIVE**: its single-draw shuffle control read
0.435, outside the pre-registered 0.5 ± 0.03. v2 changes ONLY the control; the probe recipe, data, folds and seeds are v1's unchanged
(`lonprobe.py`, sha256 recorded in LONPROBE_V2.json), so v2 reproduces v1's PROBE number by construction (K-v1: to 1e-12).

* Null distribution: 20 label permutations (numpy seeds 100..119), each permuting labels WITHIN the training folds exactly as v1's
  SHUFFLE arm, cross-fitted with the full inner CV; record each null AUC.
* Controls: (N1) the null mean lies in 0.5 ± 0.02 — else the pipeline is BIASED under the null and the verdict is VOID; (N2) K-v1.
* Verdict (only if N1 and N2 pass): **HEADROOM-CONFIRMED** if PROBE AUC exceeds the null's maximum (permutation p < 1/21) AND
  v1's PROBE − HEAD CI (> 0 at both seeds) holds; **NOT-CONFIRMED** otherwise.
* If N1 fails (null mean outside 0.5 ± 0.02), report the bias b = null mean − 0.5 and VOID; the next step is a fixed-fold design.
