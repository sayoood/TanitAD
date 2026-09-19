# A7 — ImageNet-init vs random-init `resnet34` trunk (the dev-box knockout)

**Owner:** TanitAD_TrainingFlyWheel · **assigned by:** the Master Mind, 2026-09-19 ·
**claim served:** `E-REFCV6V2-TRUNK` at dev-box scale (`GOALS_AND_CLAIMS.md`, A7 block).

> ⛔ **STATUS: RUNNING — NO DATA YET.** The panel has been queued behind A8 at the boxstat gate
> since 11:21 Berlin. The verdict section below stays empty until `a7_analyze.py` writes it. Every
> rule it will apply was pre-registered and **landed before the first arm launched** (`362ce88`).

## What the verdict will be — committed before data

`Project Steering/PREREG_REFCV6_DEVBOX_PREPARATION.md`, **A7 AMENDMENT, A7.1–A7.8**.
* **Headline:** `R = E / F` on `eval_traj` (the held-out planner loss) at step 2,000, over the 1,000
  fixed halfB windows. `E` = mean RND − mean IN, so positive means ImageNet is better. `F` = the
  larger within-condition seed difference, measured **in this panel**.
* **SUPPORTS** iff R > 1. **UNDERPOWERED (not refuted)** iff −1 ≤ R ≤ 1. **RANDOM AHEAD** iff
  R < −1. **VOID** if the A7.3(a) identity control fails, if any arm fails its validity gate, or if
  the argv audit finds a difference beyond seed, pretrained, out or dump.
* **Supporting only:** a paired episode-cluster bootstrap per seed-matched pair. Its statistic is
  the **slot-weighted ratio** (A7.8), with the plain mean as a sensitivity row. It is blind to
  training variance, which is why it is not the headline.
* **Tier T0**, a held-out, open-loop, trainer-side read at 0.38 of one halfA epoch. **Not a
  capability claim.** Family **losses** are reported per family and never pooled. The doctrine's
  family **metrics** are a work item (A7.7 item 3).

## What had to be fixed before a single arm could mean anything (all MEASURED)

| # | defect | evidence | fix and proof |
|---|---|---|---|
| 1 | ⛔ **Frozen BN on a random-init trunk is the IDENTITY.** In all 36 of 36 layers, `running_mean` is exactly 0 and `running_var` exactly 1, so the knockout compared "weights + normalisation" against "weights + none". | `test_bn_recalib.py::test_THE_CONFOUND_…` | `--trunk-bn-recalib N` recalibrates both arms on the same 256 halfA windows, then freezes. It bypasses chunking (per-image BN batches under-estimate variance) and isolates the RNG. 17 tests. |
| 2 | ⛔ **The W-BOOTSTRAP dump ran in TRAIN mode.** Dropout was on, C-REFCV3-EVAL-PRIOR-LEAK re-opened, and the training RNG moved. | Real run: 14.50351 vs 14.59615 on the same 16 windows. Rig: 170 tensors differ after one later step. | `model.eval()` plus an RNG fork. 5 tests through `T.train`. `RETR-2026-09-19-WINDUMP-TRAIN-MODE` |
| 3 | ⚠️ **Even in eval mode, the rows' plain mean is not `eval_traj`.** The loss is valid-slot weighted and the eval pairs windows into batches of 2. | The rows rebuild `eval_traj` exactly by `Σ traj·frac / Σ frac` per pair (rel 1e-5). | The A7 bootstrap uses that ratio. `RETR-2026-09-19-WINDUMP-ESTIMAND` |

**Mutation proof** (`raw/mutation_proof.json`, `stack/scripts/mutate_bn_recalib.py`): 7 of 7
mutations were caught, and the unmutated control is green (22/22). M1 reintroduces the identity
defect. M6/M7 put the dump back in train mode and delete its RNG fork.

## Provenance of the run

* **Run tree** `C:/Users/Admin/tanitad-a7-run` (outside every git index): its `stack/` and
  `taniteval/` are **byte-identical to `git archive 362ce88`**, the landed tip. `diff -rq
  --strip-trailing-cr` against a fresh archive finds **0 differences over 2,151 files**
  (`A7_BASE_HEAD.txt`). Every module imports from inside it; this was checked with
  `module.__file__`.
* **Configuration:** `C:/Users/Admin/qland/a3_heldout_read.sh` with exactly the A7.5 changes
  (`code/a7_run.sh`). The v8 eval labels are pinned as a copy with md5 `eefc38d1453b…`, because
  the mirror they live in deletes files the repo doesn't have when it resyncs.
* **Order:** IN-s0, RND-s0, IN-s1, RND-s1. The conditions are interleaved so that a panel cut
  short still holds a seed-matched pair. The boxstat gate (GPU ≤ 2,500 MiB, host ≥ 8 GB) runs
  before **every** arm, and an unreadable probe counts as INCONCLUSIVE, never as clear.

## How to resume, re-derive or bank

```
bash code/a7_run.sh                       # resumable: VALID arms are skipped, partial ones moved aside
python code/a7_analyze.py <panel> --out <panel>/a7_verdict.json
python code/a7_bank.py <panel> raw/       # per arm: argv, metrics, sha12 dump; refuses any UUID
```
The panel directory is `C:/Users/Admin/tanitad-caches/a7-imagenet-knockout-20260919`. The code
(`a7_check_arm.py`, `a7_analyze.py`, `a7_bank.py`) is tested on synthetic panels with hand-computed
E, F and R in `code/test_a7_verdict_code.py` (12 tests).

## Verdict

*(pending: written from `a7_verdict.json` when the panel ends)*
