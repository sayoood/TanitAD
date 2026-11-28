# PREREG_NATIVE40, Amendment 1: score the coordinator's windows at cache step g0. Frozen before any head number of this read exists

**Status:** frozen by the Master Mind (MM) on 2026-11-19. At freeze time no model had been loaded for this read: the chain was waiting at its memory gate before the smoke stage (`chain.log`, last line "waiting for free commit"). Everything in PREREG_NATIVE40.md (sha256 55c51fd3) stands except the window map below.

## 1 What the frozen map got wrong, found by its own control
* Under the card, a coordinator window was read at (episode, `t`), with our label at raw frame k = 2t.
* The binding control (our label gap must equal the coordinator's `gap_t` within 1e-3 m) matched **12 of 534 windows (2.2 %)**. Under the card that reads NOT-COMPUTABLE.
* The same labels-only check at raw frame k = 2·g0 matched **533 of 534 (99.8 %)**. MEASURED, labels only, no model loaded; window file sha256 166198fb4cbd9da116689cc47380726af053a7dced5e91ca3c1db8448f41c0be.

## 2 Why g0 is the right frame (from the coordinator's source, not from our data)
The coordinator traced its dump code (`imag_dump.py`, v8proof fork):
* header line 3: window frame 4 = the read tick, and frame f = cache step g0 - 4 + f;
* `:33` `S0 = 4`;
* `:153` the head reads the states of window frames 0 to 4, a history that ends at cache step g0;
* `:272` `gap_t = G15[sel, S0]`, the label at cache step g0, which is raw frame 2·g0.

So `g0` is the cache step of the read tick. The key `t` is carried over from another reader's window list and is not used by the reference. The reference numbers (LS_REF40.json) are therefore aligned: the head and the label are both at g0. The coordinator's sidecar: `LS_REF40_WINDOWS_README.md`.

## 3 The amended map
* Gap windows: a coordinator window is read at (episode, cache step **g0**). Our head is read at cache step g0 (the dump's `x_*` arrays, the same `_encode` / `_lead` call as at t), and our label is taken at raw frame k = 2·g0.
* The match control stays binding: within 1e-3 m, a share of at least 0.95 to score, and only matched windows are scored, with the n.
* Presence windows are unchanged: every labelled evaluation window of the 141 episodes, with the head and the label at that window's own read frame.
* Bars, controls, estimator, labels and outcomes are unchanged.

## 4 Code
* `native40_score.py` gains `--read-at {t,g0}`, default `g0` under this amendment. With `g0` the gap set comes from the dump's `x_*` arrays. Scorer sha256 after the change: 5dd8445d3e7029660ebe83864e9d15cd29b3fcdd0d129c6135b5ffb8889cf13e.
* Checked on synthetic dumps before freeze: match share 1.0, every control reads as stated.
* The dump script is not changed. It already reads the head at both t and g0.
