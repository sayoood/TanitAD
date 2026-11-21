# AMENDMENT 2 to PREREG_V8F_STOPPERC: HEAD-UNDERUSES is read on the paired difference; the S0-P readings (Master Mind, 2026-11-11 ~23:50, before any S0-P number on training data)

Card `PREREG_V8F_STOPPERC.md` (sha256 `70c75176…8877`, unchanged) and `AMENDMENT_1_S0.md` (`6c4ad83b…d123`). At this point no S0-P fit on training data existed. The only fit was the pipeline's code-check smoke, which fit on half of eval and scored the other half. It is not a result, and its JSON was not kept.

## 1. HEAD-UNDERUSES uses the paired interval
§4 says: "V40+E probe AUROC exceeds T's own (same windows) by ≥ 0.037, CI lower bound above T". The probe and T are scored on the same eval windows. The binding estimator rule (CLAUDE.md: for two arms on the same windows use the paired version, never a combination) therefore fixes the reading:
* **HEAD-UNDERUSES holds iff, at BOTH BAL-NULL_s0 and BAL-NULL_s1, for the linear OR the MLP probe:** the paired dAUC (V40+E probe − T, the difference taken on the same episode resample, `stat_cluster_bootstrap` B 2,000 seed 0) has point ≥ 0.037 AND CI lower bound > 0.
* The pipeline's first reading compared the probe's own CI lower bound with T's point. It is reported beside (`probe_lo`, `t_point`) and decides nothing.
* Code: `s0p/s0p_fit.py::decide_outcomes`. Test: `s0p/test_s0p.py`. One case has the probe's CI above T's point while the paired CI covers 0; it must read not-HU. The mutant `unpaired`, the reading set aside here, must go red. 13 tests pass.

## 2. The pipeline's other readings are accepted as written (`s0p/README_S0P.md`, "Readings fixed here", items 1-5 and 7-9)
* Sampler: one generator, positives first, clip ids sorted as strings.
* PCA: one label-free fit per grid on the whole training sample, with the dimension chosen by CV.
* z-scoring inside the CV.
* Tie-break order.
* NOW = the window start at age 0.
* Control definitions (label shuffle per window; episode shuffle by cycling clips; refit with the real fit's hyperparameters).
* The dump-input control covers speed, the only stored field; a second route covers the other E blocks.
* Eval nav is dead-reckoned at tick 0, equal to the logged nav there.
