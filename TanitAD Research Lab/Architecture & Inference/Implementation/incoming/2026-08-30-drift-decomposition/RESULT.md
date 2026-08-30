# MM-E6 — DRIFT DECOMPOSITION · RESULT

**Run** 2026-08-30 · **Hypothesis** MM-E6 · **Tier** T0-DIAGNOSTIC ·
**Evidence class** MEASURED (ours; dev-box) ·
**Pre-registration** `Project Steering/PREREG_DRIFT_DECOMPOSITION.md` ·
**Estimator** episode-cluster bootstrap over the 80 SCORED clips, paired across
columns, n_boot 4000 · **OMP_NUM_THREADS = 8 for the whole panel** (see §7.5).

⛔⛔ **TWO INSTRUMENT CHANGES WERE MADE TOGETHER IN v2, BOTH APPROVED 2026-08-30 —
THE REPAIRED SCENE CONTROL *AND* THE 16×40 SCENE GRID. A DIFFERENCE BETWEEN v1 AND
v2 CANNOT BE ATTRIBUTED TO EITHER ONE ALONE, AND NOTHING HERE MAY BE READ AS A
SCENE-REPRESENTATION FINDING.** Both were changed because neither could answer, not
to move a number. The grid change is separately measured in §5 and, on its own,
**did not help**.

---

## VERDICT — ⭐ **SELF-DOMINATED**, with a bound that must travel with it

> **prereg:** *"SELF-DOMINATED | `share_self` > 0.6 on the trainable arms AND
> materially lower on the frozen arm | the pathology is confirmed AND localised"*

| half of the criterion | required | measured | |
|---|---|---|---|
| trainable arms | `share_self` > 0.6 | **0.9893 – 0.9975**, every CI95 lower bound ≥ 0.9855 | ✅ MET |
| frozen arm | materially lower | **0.7279 [0.6753, 0.7848]** — CI disjoint from all three trainable arms | ✅ MET *(⚠️ on an ADDED arm, and confounded — §4)* |

**Every control read its known value** (§2). `r_full` reproduces the banked drift on
all five arms (max \|Δ\| 0.0011).

⚠️ **THE BOUND, AND IT IS NOT A FOOTNOTE: `share_self` IS AN UPPER BOUND ON THE
SELF-REFERENTIAL SHARE.** The environment channel is *measurably* weak — the frozen
DINOv3 scene representation is **beaten by a raw 32×80 pixel floor** at predicting
drift, at both grids and both ranks tested (§5). A weak environment channel
understates `r_env`, which inflates `share_self` by construction. So the honest claim
is **"self-reference dominates relative to what a frozen semantic encoder can
attribute to the scene"** — not "the environment contributes ~1 %".

⭐ **What is NOT bounded, because it is rank-controlled:** on every arm the
scene-explained subspace carries **far less** drift than a **random subspace of the
same rank 96** — `postrain30k` +0.0332 vs **+0.6553**, a 20× gap. Drift is
concentrated in directions the frozen scene does not explain, and that statement
survives the tilt that `share_self` alone cannot control for.

---

## 1. The v2 panel

Corpus `physicalai-val130-heldout` (HELD-OUT) · `SCORED = sorted(clips)[:80]`,
`FIT = sorted(clips)[80:]` (49), clip-disjoint · k=4 (0.4 s) · top-8 PCA of Δz ·
rank(P)=96 · scene = **frozen DINOv3, full 16×40 token grid** ·
**n = 7,680 rows / 80 clips, d_input = 2,048 per cell.**

| arm | step | `r_full` | banked | Δ | P oos R² | `r_env` | `r_res` | rand96 | **`share_self`** | CI95 | null | diff, CI95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `postrain30k` | 30000 | +0.6671 | 0.6674 | −0.0003 | +0.0006 | +0.0332 | +0.6660 | +0.6553 | **0.9975** | [0.9961, 0.9986] | 0.5081 | +0.4894 [+0.4864, +0.4922] |
| `o14fut30k` | 30000 | +0.6698 | 0.6709 | −0.0011 | +0.0014 | +0.0698 | +0.6699 | +0.6580 | **0.9893** | [0.9855, 0.9926] | 0.5093 | +0.4799 [+0.4754, +0.4842] |
| `emao14_30k` | 30000 | +0.6952 | 0.6952 | +0.0000 | +0.0004 | +0.0364 | +0.6949 | +0.6888 | **0.9973** | [0.9953, 0.9985] | 0.5037 | +0.4936 [+0.4907, +0.4959] |
| `e4_l4_crop` | 2000 | +0.4950 | 0.4948 | +0.0002 | +0.2611 | +0.1316 | +0.4880 | +0.4659 | **0.9322** | [0.9134, 0.9498] | 0.5320 | +0.4002 [+0.3799, +0.4193] |
| `postrain30k_freeze` ⚠️ **ADDED** | 30000 | +0.3907 | 0.3905 | +0.0002 | +0.5147 | +0.2134 | +0.3489 | +0.3703 | **0.7279** | [0.6753, 0.7848] | 0.5214 | +0.2065 [+0.1520, +0.2666] |
| ⛔ `splitp30k` | — | — | — | — | — | — | — | — | **NOT RUN — absent by checkpoint** | — | — | — |

---

## 2. Controls — every one read its known value

| control | required | measured (all 5 arms) | |
|---|---|---|---|
| **constant** | exactly 0 | `+0.000000` | ✅ |
| **scene control** (repaired, applied-shuffle) | must collapse `r_env` | collapses to **−0.0065 … −0.0001**; paired CI excludes 0 on all 5 | ✅ |
| **rank-matched** (random orthonormal rank-96) | must not match P | distinct on all 5 (e.g. +0.6553 vs +0.0332) | ✅ |
| **reproduction** | `r_full` = banked drift | max \|Δ\| **0.0011**; `emao14_30k` exact | ✅ |
| **P positive control** (carried forward per instruction) | scene→z must transfer | +0.0004 → +0.5147 — see §4 | ⚠️ |
| **vectorisation gate** | rebuilt RFF ≡ banked `rff_fold` | ≤1.3e-13 every fold | ✅ |
| **scene-bank content** | finite, non-zero | 129/129 clips; pooling 16×40→4×8 reproduces the independent 4×8 bank at rel L2 **9.3e-3** | ✅ |

### The repaired control, per arm

| arm | `r_env` | applied-shuffled | paired diff, CI95 |
|---|---|---|---|
| `postrain30k` | +0.0332 | −0.0048 | +0.0381 [+0.0281, +0.0483] |
| `o14fut30k` | +0.0698 | −0.0006 | +0.0705 [+0.0560, +0.0844] |
| `emao14_30k` | +0.0364 | −0.0053 | +0.0417 [+0.0304, +0.0544] |
| `e4_l4_crop` | +0.1316 | −0.0065 | +0.1381 [+0.1116, +0.1632] |
| `postrain30k_freeze` | +0.2134 | −0.0001 | +0.2134 [+0.1727, +0.2523] |

⚠️ **The superseded fit-time-shuffle control still FAILS on all five arms** and is
still computed and banked — it is the evidence for why it was replaced, not a
result. It does not gate.

---

## 3. v1 — the run that produced the repair (⛔ VOID, and it stands)

The first execution followed the prereg exactly and returned the fourth committed
outcome, **VOID**: *"the scene-shuffled control does not collapse `r_env`"*. The
paired CI of `r_env − r_env_shuffled` straddled zero on all five arms
(+0.0004 [−0.0038, +0.0046] … +0.0064 [−0.0068, +0.0198]).

**Root cause, and it is arithmetic.** Shuffling at FIT time and applying to the true
scene gives `z_env_shuf = W_shuf · scene_TRUE` — **still a linear image of this
clip's real scene**. The shuffle changes which directions are projected; it cannot
remove the scene. Measured across the whole range of P's power (oos R² **+0.0002 →
+0.5412**) the control read the same as `r_env` every time. A control that cannot
tell a worthless projection from a strong one is measuring the shuffle's shape.

**The repair** destroys correspondence at APPLICATION time —
`z_env_appshuf = W_true · scene_of_ANOTHER_clip` — same map, same rank, same
marginal, genuinely no information about *this* clip's scene.

⚠️ **Two near-misses worth recording.** (a) My first gate,
`r_shuf < max(0.25·r_env, 0.02)`, **passes when both reads are noise** — it would
have declared v1 valid. Replacing it with a paired-CI reproduction condition is the
only reason v1 reported VOID. (b) A 24-clip smoke produced `share_self` **0.9995**
from a projection with oos R² **−0.0003** — a SELF-DOMINATED verdict manufactured by
an impotent projection, caught only by the P positive control the prereg does not
require. That control is carried forward here for exactly this reason.

---

## 4. ⚠️ The cross-arm half of the criterion is CONFOUNDED — read it with care

`share_self` is **perfectly monotone decreasing in P's out-of-sample R²** across the
five arms (R² 0.0004→0.9973, 0.0006→0.9975, 0.0014→0.9893, 0.2611→0.9322,
0.5147→0.7279). Two accounts fit that exactly:

- **(a) mechanism** — a latent whose content the scene can reconstruct also has more
  of its *change* explained by the scene. This is the hypothesis's own logic.
- **(b) artefact** — `share_self` mechanically tracks how much of `z` the projection
  recovers, so the ordering is about P's reconstruction quality, not about drift.

**This design cannot separate them.** ⇒ The *"materially lower on the frozen arm"*
half is **met in shape but not cleanly attributable**, and it rests on
`postrain30k_freeze`, an **ADDED** arm, because the pre-registered frozen arm
`splitp30k` has no checkpoint. **The within-arm half (>0.6 on the trainable arms) is
NOT affected by this** — it is a statement about one latent at a time, and it is
decisive (all CIs ≥ 0.9855), rank-controlled by the random-projection null.

---

## 5. ⛔ The 16×40 grid did NOT clear the raw-pixel floor — the pooling hypothesis is REFUTED

The grid change was approved on the reasoning that 4×8 pooling (30°/azimuth bin)
destroyed the spatial detail drift lives in. **Measured, thread-matched at OMP=8,
`postrain30k`, identical folds and targets:**

| scene input | rank | drift (top-8 Δz PCA), mean r−shuf | raw-pixel floor | margin |
|---|---|---|---|---|
| DINOv3 **4×8** (d_raw 32,768) | 96 | +0.0357 | +0.0741 | **−0.0384** |
| DINOv3 **16×40** (d_raw 655,360) | 96 | **+0.0304** | +0.0741 | **−0.0437** |
| DINOv3 **16×40** | 256 | +0.0427 | +0.0604 | **−0.0177** |

⇒ **The full token grid is slightly WORSE than the pooled one, not better.** Pooling
was not the bottleneck. Raising the projection rank narrows the gap (−0.0437 →
−0.0177) so rank is *part* of the throttle — but **raw downsampled pixels beat frozen
DINOv3 at predicting drift in every configuration tested.**

⭐ **The likely reason, and it is a design lesson rather than a bug:** DINOv3 is
trained for *semantic invariance*, which deliberately discards the low-level
appearance change that short-horizon Δz is made of. A frozen semantic encoder is the
wrong environment channel for a dynamics question. **Raw pixels are the better
candidate and are already measured as such** — that is the concrete instrument
proposal for MM-E7, and it is also why §0's bound matters.

⚠️ **A label defect was caught and fixed here:** the floor script printed
`"4x8x1024"` over a run that had actually read the 16×40 bank (the computation was
right — `d_raw 655360` on the same line — the LABEL was wrong). Labels are now
derived from the bank, and the bank path is recorded in every JSON.

---

## 6. Instrument diagnostics (not the pre-registered reads)

Same machinery pointed at targets with known answers (clip-disjoint 10-fold,
within-clip r vs time-shuffled null, n = 7,680 rows / 80 clips, d = 96 PCs, OMP=6):

| arm | speed (positive control) | scene→z | scene→Δz | constant |
|---|---|---|---|---|
| `postrain30k` | **+0.1503** (t 2.61) | +0.0329 | +0.0371 | 0.0000 |
| `o14fut30k` | +0.1503 | +0.0485 | +0.0600 | 0.0000 |
| `emao14_30k` | +0.1503 | +0.0379 | +0.0453 | 0.0000 |
| `e4_l4_crop` | +0.1503 | +0.2007 | +0.1526 | 0.0000 |

`z` is **95.5 % within-clip variance**. A 3-frame scene stack (t−2, t−1, t) moved
`r_env` **down** on every arm, so single-frame was never the limit.

---

## 7. Status, escalations, provenance

1. **`splitp30k` — absent by checkpoint.** Two probes: its directory holds only
   `config.json`; no `*split*` `ckpt.pt` exists in any scratchpad. Registry 13.0d
   md5 `4348cad27dbf1895654c40681d92ea97`, pulled from Thor (off-limits). Not
   substituted; `postrain30k_freeze` is labelled ADDITIONAL throughout.
2. **Both instrument changes are approved and both are in force in v2** — and the
   grid half is refuted on its own merits (§5). v1 (4×8, pre-registered control) is
   retained in `raw/mm_e6_main.json` so the pair is auditable.
3. **MM-E7 proposal:** re-run the decomposition with a **raw-pixel or
   motion-sensitive environment channel** and a rank sweep. Until then `share_self`
   is an upper bound.
4. **MM-C7 (GPU breach)** — logged. Mechanism: `CUDA_VISIBLE_DEVICES=""` leaves
   `is_available()` True and `.to("cuda")` landing on cuda:0 while `device_count()`
   reads 0. The probe now **verifies isolation by attempting an allocation that must
   raise**, never by reading a count; CPU runs abort if the allocation succeeds.
5. **Thread-count provenance.** The probe is not stable in the third decimal across
   BLAS thread counts (MEASURED by reproduction: speed +0.1572 at OMP=8 vs +0.1503
   at 6, deterministic within each). **The whole v2 panel is OMP=8**; the §6
   diagnostics are OMP=6; the §5 floor comparisons are OMP=8. Every JSON records
   `omp_num_threads`, so a future cross-arm comparison can check it matched.

---

## 8. Deliverable manifest

Under `TanitAD Research Lab/Architecture & Inference/Implementation/incoming/2026-08-30-drift-decomposition/`
(**staged, not committed, not pushed**).

| file | what |
|---|---|
| `RESULT.md` | this document |
| `mm_e6_drift_decompose.py` | the probe — prereg reads, all controls, repaired control, gate provenance |
| `mm_e6_dino_scene.py` | frozen DINOv3 scene bank (grid via `MME6_GH`/`MME6_GW`) |
| `mm_e6_pdiag.py` · `mm_e6_pctrl.py` · `mm_e6_floor.py` | P power sweep · positive-control panel · raw-pixel floor |
| `run_mm_e6.sh` · `stage_to_repo.py` | runner (RC before pipe, non-zero-count pass) · content-verified stager |
| `raw/mm_e6_v2_16x40.json` | ⭐ **the v2 panel — the result** |
| `raw/mm_e6_main.json` · `raw/mm_e6_repaired.json` | v1 (VOID) and the repair validation |
| `raw/mm_e6_floor_*.json` | floor at 4×8/r96, 16×40/r96, 16×40/r256 |
| `raw/mm_e6_pdiag.json` · `raw/mm_e6_pctrl*.json` | P power sweep · positive controls (incl. the OMP 6/8 reproduction pair) |
| `raw/*.log` | stdout of every run, each with its `RC=` line |
| `raw/dino_*_meta.json` | scene-bank provenance, both grids |

**Not staged (bulk, regenerable):** the DINOv3 banks — 4×8 (~845 MB) and 16×40
(~16.9 GB) — and the raw-pixel bank, all in the session scratchpad. Rebuild with
`mm_e6_dino_scene.py` (~13 min each on the 4060). The 4×8 bank was cross-checked
against the programme's existing GPU bank at per-cell cosine **0.9999**; the 16×40
bank reproduces the 4×8 bank under pooling at rel L2 **9.3e-3**.
