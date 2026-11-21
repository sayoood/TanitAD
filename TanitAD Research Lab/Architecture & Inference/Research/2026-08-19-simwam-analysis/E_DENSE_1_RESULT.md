# E-DENSE-1 — the fix is NOT refuted; the PLATFORM cannot test it, and the positive control is how we know

`MEASURED (ours; dev-box RTX 4060, 2026-08-21/22)` · **T0-DIAGNOSTIC** ·
pre-registered in `PREREG_E_DENSE_1.md` **before any arm was trained** ·
**Thor untouched throughout.**

---

## 1. ⛔⛔ The result, and the committed rule it fires

| arm | AP | 95 % CI | AUC | loc err | vs `pixel160` | vs `prior` |
|---|---|---|---|---|---|---|
| ⛔ **`prior`** — closed form, NO features | **0.1242** | [0.1123, 0.1365] | 0.6965 | 5.80 m | ABOVE | — |
| **D `distill`** ⭐ POSITIVE CONTROL | 0.0991 | [0.0872, 0.1106] | 0.6500 | 6.69 m | **same** | BELOW |
| **B `dense`** | 0.0911 | [0.0817, 0.1025] | 0.6315 | 7.46 m | **same** | BELOW |
| ⛔ **`pixel160`** — raw patches, matched resolution | 0.0909 | [0.0810, 0.1018] | 0.6314 | 7.84 m | — | BELOW |
| **A `pooled`** — v6's design | 0.0897 | [0.0793, 0.1007] | 0.6246 | 7.49 m | **same** | BELOW |
| **C `dense_deep`** | 0.0896 | [0.0802, 0.0998] | 0.6283 | 7.35 m | **same** | BELOW |

`PREREG_E_DENSE_1.md` §5, written before training, has exactly one row that
matches this pattern:

> **B ≈ C ≈ A, and D ALSO FAILS** → ⚠️ **nothing about the objective is
> licensed.** 6.4 M / 130 clips cannot carry patch content; the whole line needs
> a bigger platform before any claim.

⇒ **THAT IS THE READ, AND IT IS NOT "THE DENSE LOSS DOES NOT WORK."** The
experiment failed to *test* the fix, and it says so because the arm put there to
detect exactly this failure did its job.

## 2. What each arm actually established

### 2.1 ✅ The harness is faithful — the control reproduces v6's null

**A `pooled` 0.0897 vs `pixel160` 0.0909.** The prereg's validity check required
the control to land at the raw-patch floor, and it does. Had A *beaten* the
floor, the experiment would have been void — the harness would not have been
reproducing the defect it was built to cure.

### 2.2 ⛔ B and C moved nothing

`dense` 0.0911 and `dense_deep` 0.0896 sit on top of the 0.0909 floor. Every
token carrying training signal, and an intermediate-layer target on top of it,
bought **nothing measurable**.

### 2.3 ⛔⛔ AND NEITHER DID THE POSITIVE CONTROL — which is the whole finding

**D `distill` 0.0991 [0.0872, 0.1106]** — trained to regress frozen DINOv3 patch
tokens, i.e. handed the answer directly — is **statistically the same as raw
pixels** and **below the no-feature prior**. For scale: the DINOv3 tokens it was
distilling score **0.1884** on this instrument.

⇒ **A 6.4 M encoder on this corpus cannot absorb DINOv3's patch content even
when it is supervised to copy it.** Whatever is blocking B and C is upstream of
the objective — it is capacity, data, or both.

⚠️ **D's handicap was declared in advance and is part of this reading:** DINOv3
targets are banked only for the 5,617 probe rows, so D saw **21.5 %** of the
frames A/B/C saw. Its failure is therefore confounded with data volume, which
*strengthens* the "platform, not objective" conclusion rather than weakening it —
but it also means **D does not cleanly separate capacity from data**, and a
follow-up must.

## 3. What this does and does not license

✅ **Licensed:** *on a 6.4 M encoder over 130 clips, neither a dense token-level
predictive target nor intermediate-layer supervision produced measurable
decodability — and neither did direct distillation from a trunk that has it. The
platform is the binding constraint.*

⛔ **NOT licensed:**
* ⛔⛔ **any claim that V-JEPA 2.1's Dense Predictive Loss does not transfer.**
  The prereg declared this read asymmetric up front: *a POSITIVE is decisive; a
  NEGATIVE is scale-confounded.* Their gains were at **ViT-G on web-scale
  video**; this is four orders of magnitude away and its own positive control
  failed.
* any **T1 / driving** claim — and see `E_DETECT_1_RESULT.md` §5.1, where the
  programme's only paired evidence runs the *other* way (`refa-dinov2`, a highly
  decodable frozen DINO trunk, drove **+2.6200 m [2.0945, 3.2570] WORSE**).
* any claim about **v6 at 336 M on 2,376 episodes**.

⚠️ **The numbers here are NOT comparable to `E_DETECT_1_RESULT.md`'s table.**
That is 640 tokens × d 1024 at 256×640; this is 160 × d 192 at 128×320. Every
arm here is scored against `pixel160`, built at *this* resolution, for exactly
that reason.

## 4. ⭐ The result that DID land decisively — the encoder is flat

`e_detect_traj.py`, matched rows (**2,809 of 2,809 stride-8 keys present in the
stride-4 set — 100 %**), paired episode-cluster bootstrap:

| comparison | ΔAP | verdict |
|---|---|---|
| **v6@20000 vs v6@11250** | **−0.0032 [−0.0074, +0.0013]** | ⛔ **indistinguishable** |
| v6@20000 vs `pixel` | −0.0046 [−0.0109, +0.0014] | indistinguishable |
| v6@11250 vs `pixel` | −0.0014 [−0.0069, +0.0039] | indistinguishable |
| v6@20000 vs `prior` | −0.0340 [−0.0416, −0.0268] | **BELOW** |

⇒ **8,750 steps of training produced no measurable change in decodability, and
both checkpoints are indistinguishable from raw pixels.** Read beside `E_V6MOVE`
— the run has **1.07 %** of its learning budget left — this settles that **the
remaining steps of `v6F-SW-30k` cannot fix the trunk.** ⚠️ It does not follow
that the run is worthless: it still produces the canonical 30 k artifact the
registry expects, and other modules (`predictor_op`, `readout`) are still moving.

## 5. ⭐ And the ranking survived its robustness check

The **prior-anchored** re-run (head starts *at* the prior, zero-init residual)
lifts every arm, and changes **no** ordering:

| arm | unanchored | anchored |
|---|---|---|
| `oracle` | 0.3673 | 0.3928 |
| `oracle_pooled` | 0.2414 | 0.2577 |
| `dino_tokens` | 0.1884 | **0.2297** |
| `dino_pooled` | 0.1416 | 0.1502 |
| `prior` | 0.1242 | 0.1242 |
| `v6_tokens` | 0.0923 | 0.0990 |
| `pixel` | 0.0912 | 0.0933 |
| `v6_cells` | 0.0888 | 0.0918 |

⚠️ **`v6_tokens` and `pixel` remain BELOW the prior even when the head STARTS at
the prior** — the anchored head actively moves away from it and loses ground.

**Paired deltas on the full grid table** (2,000 replicates) resolve what the
overlapping marginal intervals could not:

| comparison | ΔAP | verdict |
|---|---|---|
| `dino_tokens` vs `prior` | **+0.0641 [+0.0509, +0.0764]** | **ABOVE** |
| `dino_tokens` vs `pixel` | **+0.0972 [+0.0830, +0.1113]** | **ABOVE** |
| `dino_pooled` vs `prior` | **+0.0174 [+0.0084, +0.0259]** | **ABOVE** |
| `oracle` vs `prior` | +0.2431 [+0.2263, +0.2583] | ABOVE |
| **`v6_cells` vs `pixel`** | **−0.0024 [−0.0089, +0.0039]** | ⛔ **indistinguishable** |
| `v6_cells` vs `prior` | −0.0354 [−0.0432, −0.0285] | BELOW |
| `pixel` vs `prior` | −0.0331 [−0.0406, −0.0263] | BELOW |

⭐ `dino_pooled` **does** beat the prior once the comparison is paired
(+0.0174), which the marginal CIs left ambiguous — the estimator mattered, as
the registry's rule says it does.

## 6. What to do next — and what NOT to do

⛔ **Do NOT conclude the dense loss fails and move on.** That is the one reading
the pre-registration forbids on this evidence.

The question E-DENSE-1 *was* meant to answer is still open, and answering it
needs a platform where the positive control passes. In priority order:

1. **Make arm D pass first, and only then re-run B and C.** D is cheap, it is
   supervised, and until a student can absorb DINOv3's patch content on this
   corpus, no unsupervised objective can be evaluated here. Levers: more
   capacity, all 26,108 frames (extract DINOv3 targets beyond the probe rows),
   longer training.
2. ⚠️ **Settle the T0→T1 link before spending on either.** `E_DETECT_1_RESULT.md`
   §5.1: the only paired driving evidence in the programme has the *more*
   decodable trunk driving **2.62 m worse**. If decodability does not buy
   driving, this entire line is the wrong lever, and that is a cheaper question
   than a scaled re-run.

## 7. Manifest

| artifact | where |
|---|---|
| pre-registration | `…/simwam-analysis/PREREG_E_DENSE_1.md` |
| this result | `…/simwam-analysis/E_DENSE_1_RESULT.md` |
| training harness | `…/simwam-analysis/code/e_dense.py` |
| read-out | `…/simwam-analysis/code/e_dense_score.py` |
| trajectory probe | `…/simwam-analysis/code/e_detect_traj.py` |
| paired deltas | `…/simwam-analysis/code/e_detect_paired.py` |
| raw | `…/raw/e_dense.json`, `e_dense_score.json`, `e_detect_traj.json`, `e_detect_paired.json`, `e_detect_anchored.json` |
