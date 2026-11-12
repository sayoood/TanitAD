**WMCEIL (H-V8F-PROOF-3), MEASURED, tier T0, a world-model diagnostic.** Source `wm/VERDICT_WMCEIL.md`, eval-141, 1,976 windows, 141 episodes, 5 folds by episode, two seeds (s0 / s1). The control reproduced the banked world-model skill to 0.000000 at all four cells.
* Outcome by the frozen card: **WM-BELOW-LINEAR** and **JUMP-DOMINATES**.
* At 0.2 s the learned model reads skill +0.0010 / -0.0010; a closed-form ridge on (latent, actions) reads +0.0222 / +0.0215.
* At 0.4 s both read about +0.027. At 0.8 s the learned model is ahead: +0.132 / +0.135 against +0.075 / +0.075.
* At 2 s the learned model, which composes ten 0.2 s steps, reads -0.150 / -0.172; one direct linear jump reads +0.1225 / +0.1292. A predictor written in closed form, with no gradient step, is 0.27 to 0.30 better.
* Theorem T1, measured in the linear class: the direct 2 s error is 0.416 [0.389, 0.449] / 0.433 of ten times the one-step error, and the composed error is above the direct error on every fold. The multi-level criterion holds with margin.
* The latent changes slowly (lag correlation 0.975 at 0.2 s, 0.883 at 2 s) and the actions add little (ridge without actions: +0.0218 at 0.2 s, +0.089 at 2 s). Even the best predictor here explains about 12 % of the 2 s change: the target space, not only the predictor, limits imagination.
* What this does not show: any effect on driving. The world model is a training auxiliary today.

**Still running:** the stacked predictor and the jump curve (addendum 2), the weight-average and dose tests, the cached-token resolution probe.
