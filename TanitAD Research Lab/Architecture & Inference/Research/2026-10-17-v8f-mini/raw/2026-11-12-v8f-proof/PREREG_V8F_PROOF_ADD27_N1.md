# PREREG_V8F_PROOF_ADD27_N1: note to card ADD27 (TOKPROBE). Frozen before any number of the card.

Seen before this note: the hand-back of the tool builder (14 tests on synthetic data, CPU only) and the preliminary seed-0 numbers of card ADD24 that the card itself lists. No probe has run on real tokens. No number of card ADD27 exists. Tier T0 (a diagnostic on logged frames, never driving performance). STRATEGIC: PAUSED (PI item 25).

## 1. Normalisation of the tokens

The card does not state one. The adapter of the model starts with a LayerNorm over the token features (LEADSUP tree, `stack/tanitad/models/v8f_mini.py` line 988: `nn.Sequential(nn.LayerNorm(cfg.d_feat), nn.Linear(cfg.d_feat, d))`), so the state that the LSD arms read is made from normalised tokens.

* The four arms of the card (TK16_s0, TK16_s1, TK64_s0, TK64_s1) run with each token normalised over its feature axis, without parameters (`--input-norm ln`). This is the form that names.
* The form of the card as worded (no normalisation, `--input-norm none`) runs as a sensitivity pass after the main pass: the same four arm names, run folder `runs/tokprobe/raw`, states folder `probe/states_raw`. It names nothing and is reported beside the main form.
* If the two forms would give different names for an arm type, the name of the main form is stated with that limit.

## 2. Control p2

The card plants a known gap in synthetic tokens and asks for gap and closing speed at R2 above 0.9 on held-out clips. The built test plants both the gap and the closing speed (p2a; 0.955 and 0.947 on synthetic data, the one number seen).

* p2b, the form of the card: the tokens encode the gap alone, and the closing speed has to come from the change of the gap over the five frames. It is run on the dev box before any probe arm is read, at 500, 2,000 and 5,000 steps; all three are reported. p2b passes if both R2 are above 0.9 at one of the three step counts.
* If p2b fails for the closing speed: reported as written. A flat read then cannot tell "the pooled tokens do not hold it" from "the head does not take a rate from five frames", so `TOKENS-FLAT` is NOT named for any arm type; the result is stated as "no read above the bar; sensitivity of the instrument not shown", and the next step is an instrument step (a head that is given the frame-to-frame change), not the re-encoding blocker.
* `TOKENS-HOLD-CLOSING` can be named in either case: a read above the LSD arm with intervals that exclude 0 does not rest on p2b, and control p3 (permuted closing labels) guards it.

## 3. Reader

* M1, M2, M4 and the slot increment DELTA of note ADD24_N1 come from the functions of `spatial/leadsup/read/leadsup_read_n1.py`, on the same 429 scored windows and the same train probe clips.
* Paired difference: for each draw of episodes (episode-cluster bootstrap, B 2,000, rng 0) M1 is computed for the probe arm and for the LSD arm of the same seed index on the same drawn windows, and the difference is taken; interval = the 2.5 and 97.5 percentiles; point value = the difference on all scored windows. Floor = the absolute difference of M1 between LSD_s0 and LSD_s1.
* Controls of the paired read, added to p6: an arm against itself reads exactly 0; the swapped order reads the negative; a planted shift is read back.
* p3 is read from arm TK16SH_s0 (closing labels permuted, 2,000 steps): closing-speed R2 on the scored windows at most +0.02.

## 4. Unchanged

Arms, widths, seeds, training argv source (LSD_s0), step count 19,164, the head (`LeadSupHead(40 * W, 5)`), outcomes and committed next steps of the card. Other decisions of the builder: `tokprobe/TOKPROBE_DEFS.md`, written before the tool ran.
