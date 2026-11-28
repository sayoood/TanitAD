# Blast-radius sweep — which register rows rest on the untrained 52 %

**2026-09-06 · ArchInf FlyWheel · 0 GPU.** Companion to `../RESULT.md`.
Source swept: `Project Steering/GOALS_AND_CLAIMS.md`, md5-verified local copy
(`ac`-checked against the branch worktree before reading). **82 rows** — every row naming a
v7-tiny arm (`postrain30k`, `champ30k`, `k60clip05p30k`, `o11p30k`, `o1ctrl30k`, `emao14`,
"v7-tiny"/"v7tiny"), plus the two rows that mention `step_readout` directly. All 82 were found at
their stated lines; no offsets needed.

⛔ **NOTHING IS RETRACTED ON THIS RULE ALONE.** This is an exposure classification, in the shape
`H-ESTIM-SEED-1` used: **a claim about a structural identity is not exposed the way a number
decoded through an untrained module is.**

---

## 0. ⭐⭐ THE SCOPE CORRECTION THAT COMES FIRST — THE DEFECT IS THE *RECIPE*, NOT THE ARCHITECTURE

The sweep surfaced an apparent contradiction and it turned out to be the most useful thing in it.
`E-DEC-20b` reports *"Group movement 2k→10k: `predictor_op` 0.1784, `readout` 0.1644,
**`step_readout_op` 0.1095**"* for `splitfrz10k`. Under my finding `step_readout_op` **cannot**
move: it is reached only by O1, and O1 is 0.

**MEASURED, and the row is right and my finding is right — they are about different recipes.**
`v6F-snapshots/sw_config.json` (the v6F S-W arm, `run` `v6-staged-S-W`, `out`
`/home/nvidia/experiments/v6F-SW-30k`) carries **`o1_ctrl 1.0 · o1_fact 1.0 · o1_scene 0.3 ·
o3_masked 1.0 · o5_rollout 1.0`** — the **full objective set**. Every locally-banked **v7-tiny**
`config.json`, by contrast, reads **`o1_ctrl 0.0 · o1_fact 0.0 · o1_scene 0.0 · o3_masked 0.0 ·
o5_rollout 1.0`** — 4 of 4 (`postrain30k`, `k60clip05p30k`, `k8clip05p30k`, `rdw8p30k`).

⇒ **`E-DEC-20b` STANDS UNCHANGED, and the 52.2 % / 40.3 % starvation is a property of the
v7-tiny/v7f TWO-TERM RECIPE (O5 + O6), not of the v6 architecture or of the v6F family.** Any
row whose arm is a v6F-era arm is outside this sweep's blast radius entirely. *(This is why the
sweep's own list of exposed rows contains no v6F arm.)*

---

## 1. ⭐⭐ THE CHECKPOINT-LEVEL CONTROL — an independent probe, different mechanism

The census is a *forward/backward* argument. This is a *checkpoint* argument, and it needs no
model of the loss at all. If `step_readout_op`, `out_proj` and `heads.2` truly never receive a
gradient, then across arms built from the same seed they must be **bit-identical after 30,000
steps**, while a trained tensor must differ.

md5 of the raw tensor bytes, 9 banked v7-tiny checkpoints:

| arm | `step_readout_op.net.1.w` | `out_proj.w` | `heads.2.w` | **`heads.1.w` (TRAINED CONTROL)** |
|---|---|---|---|---|
| `emao14_30k` | `200db41ae05d` | `7f27621f3aaa` | `735c7ded8da2` | `5cd9aa3b859c` |
| `emao14_30k_tauramp` | `200db41ae05d` | `7f27621f3aaa` | `735c7ded8da2` | `d320943b4374` |
| `o14fut30k` | `200db41ae05d` | `7f27621f3aaa` | `735c7ded8da2` | `b7c773725d62` |
| `postrain30k` | `200db41ae05d` | `7f27621f3aaa` | `735c7ded8da2` | `a573952bdb53` |
| `postrain30k_freeze` | `200db41ae05d` | `7f27621f3aaa` | `735c7ded8da2` | `72aea4a09ada` |
| `splitp30k` | `200db41ae05d` | `7f27621f3aaa` | `735c7ded8da2` | `979721a26ea8` |
| `k60clip05p30k` | `06cb51c0bdc0` | `d3054ad54a76` | `65f0d2d2f612` | `8f9bbeb123dd` |
| `k8clip05p30k` | `06cb51c0bdc0` | `d3054ad54a76` | `65f0d2d2f612` | `7a7f74f43437` |
| `rdw8p30k` | **`0be48bc6e2fa`** | `d2ebc74fd4a8` | **`6e7f380683d7`** | `af3814bfd5c0` |
| **fresh init, seed 0, no training** | **`0be48bc6e2fa`** | `86c7a54cba2c` | **`6e7f380683d7`** | `e35e94a4a238` |

* ⭐ **The trained control differs on all 9 arms** — 9 distinct fingerprints. The test is not vacuous.
* ⭐ **The three "dead" columns take only THREE distinct values across 9 arms, and they co-vary
  perfectly** — three *initialisations* (different flags shift the RNG stream before those modules
  are constructed), not three training outcomes. Six arms that trained to six completely different
  `heads.1` share one bit-identical `step_readout_op`.
* ⭐⭐ **`rdw8p30k`'s `step_readout_op.net.1.weight` and `heads.2.weight` are BIT-IDENTICAL TO A
  FRESH SEED-0 INIT** after 30,000 AdamW steps at `lr 1e-4, wd 0.05`. Its RNG stream at
  construction happened to match my rebuild, and the tensors never moved.

⇒ two independent probes, different mechanisms, same verdict — which is what the
"repeated samples through one broken channel are one sample" rule requires.

---

## 2. THE CLASSIFICATION — 82 rows

| class | meaning | count |
|---|---|---:|
| **E1** | number decoded to **metres / Δpose / waypoints** through the checkpoint's own `step_readout_op` (or `roll_consistency` / `v6_probe_trunk` / `probe_saliency_p9` / `eval_metric_rollout`) | **3** |
| **E2** | quantity read off **`heads.2` / `heads.4`** / a multi-horizon prediction | **4** |
| **E3** | quotes v7-tiny's parameter/trainable budget as an *effectively trained* count | **0** |
| **E4** | structural: an identity, an exact zero, a bit-exactness, an argv/config/file audit, a leak/disjointness fact, an absence claim, a latency measurement | **39** |
| **E5** | rests on the **trained path** — O5/O6/O14 latent-space quantities (rank, participation, effective rank, drift, collapse, loss, probe-fitted ridge R²) | **36** |
| **E6** | cannot tell from the row | **0** |

⭐ **E3 = 0 is a finding, not a gap.** No row anywhere quotes an *effectively trained* count for a
v7-tiny arm. The two rows that quote budgets (`D-P1-ARM-MISLABEL`, `D-P1-GRADREACH-52PCT`) use them
correctly as *declared* values, and "~19 M-param" in `D-V7F-NEVER-TRAINED` / `D-T1-V7-READ` is
accurate as a **total**. What the register has never stated is the effective figure — **4,864,064**
at v7-tiny — which is the number this turn adds.

⭐ **E5 = 36 is the load-bearing half of the scoping.** The starved modules are `step_readout_op`
(O1) and `masked_cells` (O3). **Neither is on the path any latent-space claim uses**: the encoder,
readout, predictor blocks and `heads.1` are reached by O5/O6/O14 and trained normally. So the
whole rank/participation/collapse/drift/decodability body of work — `H-RANK-*`, `H-PROOF-1*`,
`H-PROOF-4/6`, `E-DEC-1/3/4/55/67/69`, `MM-E6`, `D-V7F-L1-MEASURED`, `D-V7F-DRIFT-NULL`,
`D-V7F-L3-RULED`, the four per-arm drift rows — is **NOT exposed**.

---

## 3. THE 7 EXPOSED ROWS, AND WHAT RE-ESTABLISHES EACH

### E1 — three rows, and they are ONE causal chain

⭐ **`D-V7F-READOUT-DEAD` already found the symptom; this turn supplies the mechanism.** That row
had localised every T1 distance metric to a ≈0-motion readout, and
`D-T1-NO-SCALE-ON-EMISSION-PATH` had already ruled out a harness scale error. The replacement
mechanism is **an untrained random projection** — which is also why
`D-ROW3-CONSTANT-EMITTER-REFUTED` found the weight term dominating the bias (so, not a constant
emitter) and yet the module still emits nothing usable: a random projection is neither constant
nor informative.

| row | what is exposed | what re-establishes it |
|---|---|---|
| **`D-T1-V7-READ`** | the programme's **only T1 capability read** — ade 14.069 / 13.879 m, fde 26.297, heading 94.63°, LON_speed 10.703, speed_bias −10.381 | re-analyse the banked `thor:/home/nvidia/t1dumps/*/dumps/` latent rollouts with a readout **fitted at eval time on held-out windows**, and re-state the claim as *latent decodability*, not driving skill; or re-run T1 on an arm trained at O1 > 0 |
| **`H-ARCH-ACTINS`** | the "~1 % closed-loop vs hold-action gap" (`emao14_30k` ade +1.37 %, fde +0.64 %) — arithmetic on those same metres | read the gap in **latent space** — the action-divergence probe this row itself committed to, which `MM-E10` ran at h=1. That route never touches the readout. |
| **`D-V7F-T1-NO-PAIRED-CI`** | only its **correction** of the "every distance metric" headline (cross-track 1.0722 / 1.1704, heading −0.4957, speed −0.1974). Its structural half — `paired_decision_grade` and `paired_legacy` are `{}` on all three arms — **stands untouched** | compute the paired CIs from the banked dumps **in the same pass** as a fitted-readout re-decode, so deltas and intervals come from one admissible decode |

### E2 — four rows, all multi-horizon reads

⛔ **And the finding SHARPENS `MM-E14` rather than repeating it.** `MM-E14` said those heads are
unreached *in this recipe*. The mutation arm shows they are unreached under **every** objective —
`heads.2`, `heads.4` and `out_proj` are the 6 tensors that survive with O1 **and** O3 switched on.
⇒ *"turn the loss on and re-read the heads"* **is not available**. The rolled-h=1 route is the
only one until the heads are wired.

| row | what is exposed | what re-establishes it |
|---|---|---|
| **`H-PROOF-2`** | entirely: "identity map beyond one tick", ratio 0.0002 at h=2 and h=4 | the **rolled h=1** instrument `H-PROOF-7` already built (it found both arms predicting significantly at all six rolled steps) |
| **`H-PROOF-3`** | the h≥2 half (h=2 z −0.07, h=4 z −3.22). The h=1 cos 0.3495 half is sound | rolled h=1 |
| **`H-PROOF-1B2b`** | the horizon-decay half ("0.0002× at h≥2", "~5000× too small at h=2"). The h=1 magnitude 0.264× stands | rolled-h=1 magnitude read |
| **`MM-E10`** (`emao14_30k` row) | h=2 / h=4 = 0.00001 — **already formally withdrawn by `MM-E14`**. The h=1 0.00416 latent action/scene spread stands | rolled-h=1 action divergence per horizon, or wire a loss to the h≥2 heads |

---

## 4. ⛔ ONE COMMITTED CRITERION IS HALF-UNSATISFIABLE — surfaced by the sweep, worth an amendment

**`MM-E11`'s `O1-WORKS` branch requires "h1 ratio rises ≥10× **and h2/h4 leave the floor**".**
`heads.2` / `heads.4` receive no gradient **even with O1 on** (the mutation arm), so the second
conjunct **can never fire**, and `o1ctrl30k` is pre-committed to at best `O1-INSUFFICIENT` on a
criterion that is unsatisfiable by construction. ⛔ **Not a licence to move a goalpost after seeing
data** — the correct action is to record that the criterion was unsatisfiable *when it was
written*, and to amend it before the arm is adjudicated, not after.

---

## 5. NOT EXPOSED — stated explicitly, because a sweep that only lists casualties overstates itself

* every **latency / throughput** row (`H-DEPLOY-2/3/6`, `D-DEPLOY-INTEG`) — a ms figure and a
  bit-identity do not depend on whether the weights are trained;
* every **corpus / leak / split / label** row (`D-V72-SPLIT`, `H-LEAK-1/2/6`,
  `D-V7F-L1-CLIPS-DISJOINT`, `H-DEC-3`) — clip-id arithmetic;
* every **absence** row (`D-V7F-NEVER-TRAINED`, `D-V7F-UNATTENDED`, `D-V7F-TAU-RAMP-UNRUN`) —
  and note that `D-V7F-NEVER-TRAINED` being true is *why* the v7f exposure is prospective, not
  historical: there is no v7f checkpoint to re-read;
* every **source-defect** row (`D-V7F-O6-RULES-ON-FORBIDDEN-STAT`, `D-V7F-L1-GATE-ORDER`,
  `D-V7F-L1-GATE-IS-RELATIVE`) — read from the code that applies them;
* the **P1 rows themselves** (`D-P1-*`, `D-ROW3-*`) — they *are* this finding;
* `H-DEPLOY-4` deserves a special mention: its metres come from the **unicycle integrator**, not
  from `step_readout_op`, so it is E4 despite being a distance claim. That distinction is the
  whole reason this sweep had to read rows rather than grep them.
