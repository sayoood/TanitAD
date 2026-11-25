# PREREG_C1R_READ: the REFe C1 recipe's training-seed floor (C1r vs C1)

MM, 2026-11-25. **Frozen before any C1r navtest number exists**: at freezing, C1r's dev-box seam had not started (its waiter was holding for the MM's test run; `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/c1r/read/CHAIN.log`). C1r's training metrics exist; no navtest output does.

## 1. Question
On the REFe P5-C rig, with the C1 recipe held fixed, how far apart do two training seeds land on navtest tracked PDMS?
This is the sufficient form of `H-ESTIM-SEED-1` for REFe. Every single-seed REFe delta has lacked it: C1 - C0 +1.1702 [0.6730, 1.6450] (`PK/raw/2026-11-20-p5c-read/VERDICT.txt`), SCORER-FT T - C -0.9015, the field read F - C, and the coming P - C.
It is **not a lever test and adopts nothing**.

Why it exists: `PREREG_P5C.md:259-269` registered only C0r, and only on SEPARATED-SINGLE. The P5-C verdict was GUARD-FAIL, so C0r never trained. C1r was approved by the PI on 2026-11-20 ("Follow your recommendations"; `PK/raw/2026-11-21-c1r/BRIEF_C1R.md:8`), but no bar was registered for it alone.

## 2. Arms
| arm | seed | model sha256 | seam |
|---|---|---|---|
| C1 | 0 | `f09949e2a5f2f9661ad47457e77ddfd10f49117d64e095487809c1d04f7a34a4` (MODEL_REGISTRY 14.6) | `stage2_p5c_c1` |
| C1r | 1 | `6e39a57c…54a8`, equal to the pod's `model_final_sha256_C1r.txt` (full value recorded by the read) | `stage2_p5c_c1r` |

Both seams come from the same dev-box chain command (`p5c_read_chain.sh:192`), the same tip (`F:/TanitAD_p5cread_tip`, 751f3ae7), and the same base and extra checkpoints (`--base p6 --also p2`).

## 3. Measures
All measures are tier T1 (NAVSIM v1.1, LQR-tracked), on 12,146 tokens over 136 logs.
Estimator: paired log-cluster bootstrap, 10,000 draws, seed 20260927, 95 % percentile, on `full_set` means.

- **M1** `d_seed = PDMS(C1r) - PDMS(C1)`, with its CI.
- **M2** C1r's PDMS with its CI, and the sub-scores NC, DAC, TTC, EP and C (DDC is reported, not judged).
- **M3** The four families, paired C1r vs C1: longitudinal, lateral, tactical, and strategic in full. Each gets its CI and n, or a per-family reason and n where it cannot be computed.
- **M4** Descriptive only: `C1r - C0`. This is cross-seed and is not a seed-matched pair; C0r does not exist.

## 4. Committed reading (both outcomes written before the numbers)
- **R-SEP: the CI of `d_seed` excludes 0.**
  - Seed alone separates on REFe, so `H-ESTIM-SEED-1` is confirmed on this rig.
  - Any single-seed separated REFe result may then be quoted only as "separated, single seed", never as a lever effect, until it has a replicate.
- **R-NULL: the CI contains 0.**
  - No separated seed effect at this n.
- **In both cases**, `F_REFe := |d_seed|` (point) is this rig's first measured seed floor. It is ONE draw, a 1-degree-of-freedom estimate, and is printed as such. Three consequences are committed now:
  - (a) **C1 - C0 (+1.1702)** is printed against `max(B = 0.7031, 2·F_REFe)`. If 1.1702 <= 2·F_REFe, it is labelled "inside the REFe seed floor". This is descriptive; P5-C's frozen verdict, GUARD-FAIL, does not change.
  - (b) **Frozen cards** (SCORER-FT T and P, field F) get `F_REFe` printed next to their result as a descriptive qualifier; their committed bars do not move. **Cards frozen after this read** use `max(F, F_REFe)` wherever their bar has a 2F term.
  - (c) **C1r is never adopted** on this read, even if `PDMS(C1r) > PDMS(C1)`: keeping the better of two seeds is selection on noise. Goal status is printed as written: BELOW GOAL when C1r's point is < 94.55.

## 5. Controls (any FAIL = INVALID; the numbers are then descriptive only)
- **CTRL-ARGV**: argv(C1r) - argv(C1) = {`--seed`, `--out`} exactly, read from both pod `config.json` files.
- **CTRL-SHA**: both model files equal their sha256 values above.
- **CTRL-SET**: both seam reports read 12,146 rows and 0 misses, and the token sets are identical.
- **CTRL-T**: the 64-proposal table's pick equals the harness CSV to 1e-6, for every token of both arms (`p5c_read_prep.py identity`).
- **CTRL-ID**: C1 read against itself gives every delta exactly 0.0.
- **CTRL-CODE**:
  - Record the sha256 of every `refe/*.py` file in each arm's training package: C1r trained from `code_bce4f9a7`; C1's package is the one its pod config/summary names.
  - If any training-path file differs, each differing hunk must be shown behaviour-neutral by an existing pinned test, else the outcome is stamped **SEED+CODE**: no seed-floor claim, numbers descriptive.

## 6. Tools
- `p5c_read_prep.py` (convert, score, identity), `p5c_families.py`, `instruments/strategic_family.py`, and `taniteval/ci.py` for the paired log-cluster bootstrap.
- A small `c1r_read.py` is written after this freeze. Its only freedom is mechanics: it prints every measure in section 3 and every control in section 5, and nothing chosen after the numbers.
- `GOALS_AND_CLAIMS.md` (`H-ESTIM-SEED-1`, and the registry 14.6 caveat "C1r pending") is updated in the same turn as the read.
