# PREREG -- the step-5,000 G0 detection residual `eval_box3d_det_ap2_automobile_0_20`  (EvalFlyWheel, 2026-10-16, written BEFORE any arm ran)

**Question.** The step-5,000 G0 (A7) has ONE non-wrapper reason left: in-run 0.33835 vs dev-box 0.31453 (sd 0 over 24 seeds), judge tolerance
max(0.02, 2/n) = 0.020619 at n = 97 GT automobiles (`g0.json` verdict_A7.terms). Hypothesis H-NUM (UNVERIFIED): Thor's in-run eval ran the backbone
under `torch.compile` (Inductor) + bf16 autocast, the dev box runs it eager + bf16 autocast, and an AP at a 2 m match threshold on 97 events flips
on a few near-threshold centres. Rival H-PATH: a real loader / eval-path difference (inputs, flags, shapes) that numerics cannot produce.

## What was already known when this was written (disclosed, so nothing below pretends to be blind to it)
* `g0.json` a6.fp32_s0 (SPEC A6 numerics arm, seed 0, trunk fp32 + NCHW, TF32 off, deterministic) already reads this term at **0.31912** (s0 bf16 = 0.31453),
  i.e. fp32-vs-bf16 moves it +0.00459 = 19 % of the gap; 0.31912 is 0.01923 from the in-run value (inside the 0.0206 judge tolerance).
* The A7 low-support clause passed (N_in 12 <= 2 N_num 9 + 5). This term is the only DETECTION (n >= 30) term out of 100.
* In-run config.json argv carries `--trunk-bf16 --trunk-channels-last --trunk-fold-bn --trunk-dedup-frames --trunk-compile --trunk-chunk-ckpt 8 --trunk-frozen-bn`;
  the loader drops only `--trunk-compile` (refcv7_loader.py:144, 218).

## Fixed quantities
X = 0.33835 (in-run), M = 0.31453 (G0 mean, every seed), D* = |X - M| = 0.02382, T = 0.020619 (judge tolerance), v(arm) = the arm row's
`eval_box3d_det_ap2_automobile_0_20` (5 dp). Same 128 windows (perm_sha256 99adbc7e...), same ckpt (md5 a32e3b70...), DDIM inference seed 0 in every arm
(the term is bit-identical across noise conditions, so the seed is not a lever here).

## Arms (GPU, dev box RTX 4060, one model, one set of collated batches; each arm differs from A0 in exactly the stated lever)
| arm | lever | class |
|---|---|---|
| A0 | the G0 condition (micro 2,2,3,3,3,3; bf16 autocast trunk; NHWC; cuDNN TF32 as G0) | control; MUST read M = 0.31453 to 5 dp, else HARNESS_MISMATCH and nothing is concluded |
| A1 | fp32 eager = `wrapper_probe_r7.CONDITIONS["P3_fp32_det"]` (trunk fp32 + NCHW, TF32 off, deterministic) | numerics-only |
| A2.d | trunk (`trunk.net`) parameters x (1 + u), u ~ U(-2^-8, 2^-8) i.i.d. per element (bf16 half-ulp scale; the autocast re-rounds them), draws d = 1001..1003 (+1004, 1005 if GPU time allows); restored and digest-checked after each | numerics-only |
| A3 | compiled backbone, ONLY if it runs on this box: probe `import triton` / Inductor first; if Inductor is absent, say so and SKIP (an `aot_eager` compile is NOT Inductor and is reported as A3b, not as A3) | numerics-only |
| A4 | EXTRA, not in the brief: micro 8,8 so the backbone sees the in-run chunk shapes (chunk_ckpt 8 over 48 frames = six 8-frame calls); skipped on CUDA OOM | shape (eval-path), not numerics |

CPU-only companion **P** on the banked `g0.packs.npz` (s0 and fp32_s0 box3d packs; descriptive, never gating): P1 recompute v for both arms (must give 0.31453 / 0.31912);
P2 support (GT automobiles in 0-20 m, positives/ignores, detections at the 0.5 gate); P3 near-threshold census: candidate (detection, GT) pairs of the
automobile class whose distance lies within +-0.1 m of 2 m, plus GT / detections within +-0.1 m of the 20 m band edge; P4 a Monte-Carlo dose-response on the
s0 packs: add the EMPIRICAL s0-vs-fp32_s0 per-slot xy / logit displacement (resampled, random sign) scaled by f in {0.5, 1, 2, 4, 8}, 200 draws each, recompute v.

## Decision rule (committed now)
Let Dk = v(k) - v(A0) over the numerics-only arms actually run. An arm row is "bar-clean" when, judged against the in-run row with the G0 judge's own `tol_ok`
(SMOOTH, and DETECTION with support >= 30 at max(0.02, 2/n)), it has no OUT term other than this one (A0 itself: 1 DETECTION, 0 SMOOTH OUT).
* **R1 NUMERICS_SUFFICIENT**: some bar-clean numerics-only arm has |Dk| >= D* = 0.02382. -> write `AMENDMENT_D_R8_1_DRAFT.md` (a measured numerics floor for low-count
  DETECTION terms, with the exact count shown). A big |Dk| in an arm that is NOT bar-clean is a perturbation too violent to be called numerics and does not count.
* **R2 NOT_NUMERICS**: every numerics-only arm has |Dk| < D*/2 = 0.01191. -> H-PATH: localise (L1 GT-side counts vs the in-run row, L2 eval-cache decode path, L3 A4 shape arm,
  L4 flag/argv diff) and say which input or flag differs, or that it is not localisable on the dev box.
* **R3 PARTIAL**: max |Dk| in [0.01191, 0.02382). -> neither verdict; report the fraction of D* reached, the P4 dose-response, and that Thor's Inductor numerics were not replicated; no amendment text proposing a floor.
Reported either way: whether any arm lands inside T of X (|v - X| <= 0.020619) -- the judge's actual pass test -- and why that is NOT the registered success test (an arm landing there
at |Dk| < D* shows the term sits inside the numerics spread, which is R3 evidence at best).
Not modifiable after the data: the thresholds D*, D*/2, T, the arm list, the draw seeds, the u scale. Single training seed, single checkpoint, 128-window subset: stated, not hidden.
G0 is NOT marked PASS by this work; the Master Mind registers any amendment.

## ADDENDUM 1 (written 2026-10-16 16:30, BEFORE any CPU arm ran; the registered GPU arms are still queued)
Why: the shared GPU lock is held since 15:29:23 by a FOREIGN job (`refe-stage2-p2s1_full`, NavSim, pid 33488; never touched), with the G0-10500 chain and this probe both waiting behind it; its end is unknown. RULE ZERO: run the
cheapest arm that does not need the card. A CPU fp32 forward takes ~10 s/window (smoke: 3 windows 30 s) vs ~116 s/window for CPU bf16, so a full 128-window CPU arm is ~20 min.
* **C0** = the same model, same 128 cached windows, fp32 eager on the CPU (trunk fp32 + NCHW; no TF32 exists there): a DIFFERENT-HARDWARE numerics realisation. Read v(C0), D_C0 = v(C0) - M, and |v(C0) - X| <= T?
* **C2.d** (d = 1001, 1002, 1003) = C0 + the A2 perturbation (trunk.net params x (1+u), u ~ U(-2^-8, 2^-8)). Read D_d = v(C2.d) - v(C0).
Rule (descriptive; it can NEVER replace R1-R3, which are judged on the GPU arms only if they run): **CPU_NUMERICS_REACH = YES** iff max(|D_C0|, |D_d|) >= D* = 0.02382 with that arm bar-clean (judge, vs the in-run row);
NO iff all < D*/2; otherwise PARTIAL. Not an A0 reproduction (CPU != the G0 GPU condition by construction) -- C0 is NOT expected to read M. Output: `raw/step5000/detres/cpu_arms_probe.json`.
