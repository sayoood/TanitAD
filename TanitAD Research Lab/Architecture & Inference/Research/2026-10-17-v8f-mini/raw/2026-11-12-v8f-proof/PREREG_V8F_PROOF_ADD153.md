# PREREG_V8F_PROOF_ADD153 — ACCHIST10K: does a longer history of image tokens, with an acceleration output, let the vision head read the lead's acceleration?

Frozen before any tool of this card exists and before any number of its arms. Open at this time: row 217 (card ADD150 with amendment 1, `LEADSPEEDALL10K-SPEED-ONLY`: held-out closing r S 0.393 / 0.422 and acceleration r A 0.205 / 0.212 at training seeds 148 / 149, bar 0.30). Tier **T0** (a head on cached tokens of logged frames; it says what the tokens allow, not what a model does with them); never driving performance. PI, 2026-11-28, verbatim: "Card, build and run it. Regarding acceleration readout, think of giving the model enough history in the images and related tokens to extract dynamics."

## 0. Why row 217's head cannot read the acceleration well (design facts, no new data)

* Its two causal Conv1d (kernel 5) see **9 frames** (1.8 s) at the output frame, though H = 10 frames are loaded.
* It never trains on an acceleration. A is derived from two closing outputs 1 s apart, so the noise of two separate outputs adds up in their difference.
* Its frame encoder pools each frame to one 64-wide vector before any temporal step, so the change of single tokens between frames reaches the convolutions only through that pooled vector.

## 1. Data, split, training (all arms)

The clips, labels, clock and split of card ADD150: corpus `D:/tanitad/cache_full/train`, labels `lead_train4572.npz`, split seed 148 (3,984 fit clips, all used; 443 held out); eval-141 clips `D:/tanitad/cache/eval` with `join_eval141.npz`. Tokens are the 4x10 grid (`pool4x10`, 40 tokens x 1024 per frame, fp8) read as read-only memmaps (`rulefc/leadspeed/tokstore.py`, card ADD149 amendment 3). 30,000 steps, batch 64, AdamW 1e-3 with cosine decay to 1e-4, windows drawn uniformly, checkpoints at 10,000, 20,000 and 30,000 steps; the read of record is the 30,000-step checkpoint. The memory guards of `train_full_box2.py` (amendments 1 to 3 of card ADD149) apply unchanged. `rulefc/leadspeed/leadspeed.py` is imported for the labels and is not changed.

**Windows.** A window ends at every frame t >= 9, as in row 217, so every arm emits at the same frames and its reads use the same frames as row 217 (held-out n_S 11,623, n_A 10,325). Frames before the clip start are zero tokens and are masked out (their pooled frame vector is 0, the same value the causal zero padding of the convolutions uses).

**New label (training only and read A_rel):** d(t) = closing(t) − closing(t − 5), the change of the label closing over 1 s, where both exist (m/s).

**New output:** a fourth output, the head's d (scaled by 10 m/s like the closing), trained with L1 where d exists. Loss = presence BCE + gap L1 + closing L1 + d L1, all weight 1.

## 2. Arms

Stage 1, training seed 148, in this order (priority order: an early stop still leaves the first arms):

| arm | history H | temporal stack | frame encoder | levers |
|---|---|---|---|---|
| `K25D` | 25 frames (5 s) | causal Conv1d k 5, dilations 1, 2, 4, residual (sees 29 frames) | row 217's | (a) + (b) |
| `K25DM` | 25 | as `K25D` | row 217's plus the token differences x_t − x_(t−1) (own LayerNorm and Linear, summed before the GELU; 0 at the first frame of a window) | (a) + (b) + (d) |
| `K10D` | 10 (2 s) | row 217's two plain Conv1d k 5 | row 217's | (b) alone (attribution) |
| `K40D` | 40 (8 s) | dilations 1, 2, 4, 8, residual (sees 61 frames) | row 217's | (a) longer + (b) |

Stage 2 (always run, after stage 1): the stage-1 arm with the highest held-out A (record) is run again at **training seed 149** (`<arm>_s149`, the replicate), and once more at seed 148 with a **smoother d label for training** (`<arm>S`: the closing as the least-squares slope over up to 10 frames, 2 s, with the same step rule and at least 3 frames; d from that closing; the reads stay as committed).

Lever (c), the 8x20 grid (160 tokens per frame, in the same clip files), is **not run by this card**, with costs measured at this freeze: a clip file is 20.9 MB, of which the 8x20 tokens are 16.5 MB; 3,984 clips are 66 GB against box 2's 31.9 GB of RAM (18.0 GB available), so training would read the USB disk at random (about 256 MB per step at H = 25, several seconds per step); a cropped store needs about 13 GB and box 2's D: has 1.93 GB free. It needs a disk (or fewer clips, which row 217 shows costs closing r) and is named as the blocker if no arm carries.

## 3. Reads (held-out clips primary; eval-141 for the consequence)

On the frames of row 217:
* **S**: Pearson r of the head's closing against the label closing (unchanged).
* **A (record)** for arms with the d output: a_h(f) = max(0, v(f) + c_h(f)) − max(0, v(f − 5) + c_h(f) − d_h(f)), against the label a_l(f) = max(0, v(f) + c(f)) − max(0, v(f − 5) + c(f − 5)), v the token cache's speed, on the frames where row 217's A is read (f and f − 5 labelled, the head emitting at f − 5). This is row 217's construct with the head's past closing taken as c_h(f) − d_h(f) from one window.
* Beside, each named and naming nothing: **A_derived** (row 217's construct from c_h(f) and c_h(f − 5)); **A_floor**, the ego-only construct with c_h = d_h = 0 (a_h = max(0, v(f)) − max(0, v(f − 5)); what A reads from the shared ego speed term alone); **A_rel** = Pearson of d_h(f) against the label d(f) (the lead's part without the ego term); **A_lab2** = Pearson of the committed a_l against a_l built from the 2 s label closing (how far the committed label agrees with a smoother one); gap MAE, presence AUC, the same per held-out half; and the paired read against row 217's seed-148 head (`D:/tanitad/leadspeed/all/head_30000.pt`, sha256 83faaa6d checked first), clip-cluster bootstrap over the 443 held-out clips, 2,000 resamples, seed 150 (another draw of clips; one training run per arm).
* **dA_floor**: A (record) minus A_floor on the same frames, the same paired clip-cluster bootstrap (2,000 resamples, seed 151).

## 4. Controls (must pass before a name is given)

c1 to c5 of card ADD149 with amendment 3 (run completes 30,000 finite steps under the guard with the last 1,000-step loss below the first; c2 no held-out or eval clip in the fit set; c3 the label closing fed in place of the head, with d_h = the label d, reads S = 1 and A = 1; c4 the 30,000-step checkpoint read is the one saved; c5 the mapped token bytes equal `torch.load`'s on 50 fit, 20 held-out and 20 eval clips and a planted offset differs), plus:
* c6 window alignment: for one held-out clip, the arm's window at frame t holds exactly the token frames max(0, t − H + 1) .. t and zeros before, by byte comparison against the memmap.
* c7 the d output is trained: its parameters' gradient is non-zero at step 1 and the d term is in the logged loss (a finite value).

## 5. Names and consequence (committed)

Per arm and seed: `ACCHIST10K-CARRIED` if held-out A (record) >= 0.30 **and** the 95 % lower bound of dA_floor > 0; `-SPEED-ONLY` if not carried and S >= 0.30; `-NOT-CARRIED` otherwise. The card's name is that of the stage-2 replicate pair (the selected arm at seeds 148 and 149): the shared name when both seeds agree, `ACCHIST10K-SEED-SPLIT` otherwise. The smoother-label arm gets its own name, beside.

Consequence: `ACCHIST10K-CARRIED` with A >= 0.30 on eval-141 at both seeds makes the head's acceleration a_h the acceleration input of the constant-acceleration forecast of card ADD144 (`--lead-forecast ca`) in a reader-form rule arm (a successor card: build, rollouts on box 2 against row 211's every-step reader form, the PI's go for box 2 time). If only the smoother-label arm carries at seed 148, its replicate is the successor. SEED-SPLIT: a third seed of the selected arm is the successor. SPEED-ONLY or NOT-CARRIED in every arm: the blocker is named with the measured values; A_floor, A_rel and A_lab2 say whether it is the tokens (then lever (c), which needs a disk) or the label (then a finer label, the 0.1 s lead rows instead of the 0.2 s token clock).

## 6. Cost and stop rule

Row 217's arm took about 32 to 39 min on box 2 at H = 10 (0.054 s per step). Expected: `K10D` about 0.6 h, `K25D` and `K25DM` about 1 to 1.3 h each, `K40D` about 1.6 h, stage 2 about 2.3 h; about 7 h in all. A smoke job (each stage-1 arm 200 steps on 100 fit, 20 held-out and 10 eval clips; names nothing) runs first and measures the seconds per step; if its projection of the whole card exceeds 11 h, the sequence stops and the PI is asked. One job at a time on box 2, after card ADD152's jobs.
