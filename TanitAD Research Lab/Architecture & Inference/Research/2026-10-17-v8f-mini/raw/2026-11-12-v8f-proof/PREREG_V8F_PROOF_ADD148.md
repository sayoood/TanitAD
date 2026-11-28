# PREREG_V8F_PROOF_ADD148 — LEADSPEED10K: a vision read of the lead's closing speed on box 2's corpus tokens. Card, build and SMOKE only; the full training run waits for the PI's approval of its cost

Frozen before any tool of this card exists. Open at this time: row 213 (landed at 0179dc61). Tier **T0** (a head on cached tokens of logged frames; it says what the tokens allow, not what a model does with them); never driving performance. PI, 2026-11-25, verbatim: "Yes, card it first" (to the question: write the card, build the head and smoke-test it on box 2, with no full training run until the PI approves its cost).

## 1. Why this card

Row 210: the lead forecast with the lead's measured change of speed lowers U in the label form. Row 212: the token reader cannot carry it (reader acceleration r 0.087 / 0.071); its gap-rate closing sits near r 0.3 (rows 212, card ADD50), and the 2026-11-15 note found the linear token probe cannot learn closing even on a planted signal (held-out R2 at most 0.091). Row 134 / spec: the remaining reader-form contact at age is carried by the missing closing input. Box 2 holds corpus tokens of the eval cache's encode (encoder_module_sha256 361bc287, `D:/tanitad/cache_full/train`, 4,572 clips) and train lead labels (`D:/tanitad/data/v9/lead_train4572.npz`, 4,427 clips, frame k on the 0.1 s clock, token frame j with k = 2 j). This card builds a head with explicit temporal capacity and checks, before any training run of size, that it CAN learn closing and that the data line up.

## 2. The head (inference input: camera tokens only; no ego state, no label)

Per token frame t: the 40 tokens of the 4x10 grid (fp8 to float) -> LayerNorm -> linear 1024 -> 64 -> GELU -> attention pooling over the 40 tokens -> z_t (64). Over the last H = 10 frames (2.0 s, frames t-9 .. t, past only): a causal 1-D convolution over time (kernel 5, two layers, width 64) -> outputs at frame t: presence logit, gap (m, scaled by 50), closing speed (m/s, scaled by 10). Loss: presence BCE on all frames; gap L1 and closing L1 on frames with a lead. Closing label: the least-squares slope of the label gap over the trailing run of up to 5 token frames (steps <= 3.0 m, at least 3 frames), the formula of `LabelSource.read` of the rule wrapper, on the token clock. Clip split: 90 % fit, 10 % held out, by clip, seed 148; the eval-141 clips never enter (they are not in the corpus cache; checked).

## 3. Smoke (this card): controls that must pass, in this order

* s1 clock: label rows on the token clock: the lead file's own speed `v` at k = 2 j against the token cache's `v` at j, on 200 corpus clips (seed 148): the pairing k = 2 j has the smallest error of the shifts -2 .. +2 token frames, and a planted shift of one frame is detected.
* s2 planted closing (synthetic tokens, CPU): tokens carry a planted gap along a fixed direction plus noise, closing = its slope; after training the head the held-out closing R2 is at least 0.80. Beside: the same data through a head without the temporal convolution (mean over frames), which must stay low (the 2026-11-15 failure, reproduced).
* s3 permuted labels: closing labels permuted among clips: held-out closing r at most 0.10 (planted data).
* s4 causal: changing frames after t changes no output at t (exact).
* s5 real-data smoke on box 2 (GPU, safe mode, BOX2_BUSY): 500 steps on the fit clips; loss finite and its last 100-step mean below its first; the measured step time, peak VRAM and data read rate are recorded. Held-out closing r after 500 steps is printed beside and names nothing.

## 4. Names and consequence (committed)

`LEADSPEED10K-SMOKE-PASS` if s1 to s5 pass; then the cost of the full run (steps x measured step time on box 2, held-out read, and the reader-form rollouts of the forecast with this closing) goes to the PI as a decision card, and nothing is trained beyond the smoke without the PI's go. `LEADSPEED10K-SMOKE-FAIL-<control>` otherwise, with the failing control; a successor is frozen (architecture or data), not a widened bar.

## 5. Cost

CPU minutes on the dev box (s2 to s4); about 500 steps on box 2 (s1, s5), expected minutes. No training run of size.
