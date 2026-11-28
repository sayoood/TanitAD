# REFe final navtest evaluation -- readiness (2026-09-26 ~15:50 Berlin)

`SPEC_NAVTEST.md`: *"Subset first ... then the full 12,146"* -- the final checkpoint is scored on ALL of navtest.
This note makes that run a known quantity before training ends (projected Thu 1 Oct ~01:20 Berlin, inside the
extension the PI accepted on 2026-09-26).

## In place (MEASURED)

| item | state | evidence |
|---|---|---|
| nuPlan DBs for navtest | 136 / 136 logs on the dev box | `SPEC_NAVTEST.md` §0 (`D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/test`) |
| camera frames | **48,584 / 48,584** (12,146 tokens x 4 cameras) | `D:/Projects/TanitAD/data/refe_navtest/build_frames.log`: `ZZNAVTEST_FRAMES_OK 48584 48584` |
| token set | `D:/Projects/TanitAD/data/refe_navtest/navtest_all_tokens.json`: 12,146 tokens / 136 logs from W3's export (`n_missing_vs_yaml 0`), md5 `4764e97f0c60453c96908e02f43012e2` | built 2026-09-26 |
| floors | W3's full-navtest arms (STOP, CV, HUMAN, refcv4b_A1) cover tokens outside the 200-token subset | smoke below |
| loader | `ckpt_io.load_for_inference` takes the FULL `model_final.pt` (written last by the trainer, sha256 in `summary.json`) or a PARTIAL snapshot | `refe/ckpt_io.py:116` |

## Smoke on logs the 200-token subset never touched (MEASURED 2026-09-26 15:45-15:48 Berlin)

24 tokens = 8 sorted tokens from each of 3 navtest logs outside `A1_sub200`'s 93 logs, drawn with
`random.Random(20260926)` from the 43 eligible logs; the snapshot after epoch 12; `eval/eval_checkpoint.py` unchanged.

- **seam** 24 / 24 rows, 0 misses; frame control max 7.1e-15 m (log future vs W3's human future); 61.9 s = 2.6 s/token
  with the dev-box GPU shared with other sessions
- **harness** PASS, 24 / 24 valid rows, C1 max |d| 0.0, 18.3 s
- **floors** present (CV, STOP, HUMAN, refcv4b_A1); **four families** computed (tier T1-family)
- PDMS 39.97 on these 24 tokens -- a SMOKE value (n = 24, 3 logs, one map), **not a result**
- 145 s end to end. Banked: `eval/raw/fullsmoke24/` (token file + point JSON).

## Projected wall-clock for 12,146 tokens (ESTIMATED from three MEASURED seam rates)

Seam per token: **1.35 s** (after-epoch-11 subset seam, light load) · **2.6 s** (this smoke) · **3.5 s** (after-epoch-12
subset seam, while other sessions' refcv6 jobs held the GPU) -> **4.6-11.8 h**. Harness scoring ~0.5 h (from 18 s per 24
tokens and ~45 s per 200). Families: minutes. **The final PDMS lands ~5-12 h after `model_final.pt` reaches the dev box**
(~1.3 GB at the ~6 MB/s measured for snapshot fetches: ~4 min).

## The run (after the pod's `summary.json` says `"done": true`)

1. Fetch `/workspace/data/refe_runs/vitl16_navtrain10_grow_tau0.3/{model_final.pt,summary.json}` to
   `D:/Projects/TanitAD/data/refe_runs_eval/` and check the sha256 against `summary.json["model_final_sha256"]`.
2. `C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe eval/eval_checkpoint.py --ckpt D:/Projects/TanitAD/data/refe_runs_eval/model_final.pt --name navtest_final --tokens D:/Projects/TanitAD/data/refe_navtest/navtest_all_tokens.json`
3. Result: `D:/Projects/TanitAD/data/refe_navtest/points/navtest_final.json` -- PDMS with STOP / CV / HUMAN floors and
   paired log-cluster intervals over 136 logs, the four families, proposal diversity. Bank it into `eval/raw/` and the
   registry the same turn.
4. The learning curve's last point on the final model: `bash eval/eval_snapshot.sh final 2` (DONE 2026-09-26 ~17:15):
   it fetches `summary.json` first and refuses unless `"done": true`, checks the model's sha256 against
   `summary.json["model_final_sha256"]` as well as the pod's md5, runs the `sub200_final` curve point, E-6 and
   Amendment 4. `eval/amendment4_readout.py` places the final model one past the last epoch with a bank line
   (selftest 32/32 incl. the final-name cases); the report names it "final" / "the final model" (rendered on a fixture
   that copied the epoch-12 files as a fake final point: labels on both axes and in every table, 0 chart errors).
   Step 2's full-navtest run is separate and uses `--name navtest_final`, which the report does not read as a
   learning-curve point.

## Risks

- **Dev-box GPU contention** (other sessions' jobs): the seam rate varied 2.6x on 2026-09-26; the run can take the whole
  morning of Oct 1.
- **W3's RAM guard** (`W3_RAM_GUARD_ABORT` below 3 GB free) aborts a scoring run; the full-navtest scoring is ONE run
  (not 64), so a retry costs ~0.5 h.
- **The pod has no persistent volume.** Back up `model_final.pt`, the final `ckpt_last.pt`, `summary.json`,
  `metrics.jsonl` and the on-policy sets BEFORE the pod is stopped -- stopping it deletes them.
