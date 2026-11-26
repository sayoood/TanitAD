# HANDOFF: L1 TRK-SCORE (`H-REFE-TRKSCORE-1`), 2026-11-26 ~20:55Z

## State
- **The card is frozen:** `PREREG_TRKSCORE.md`, sha256 `5cf2c3235b5a6f2d8b38d8b278e105f64a42d7389de5ebc10fc7a1d11cc2714e`, frozen ~20:33Z before any TRK number.
- **`AMENDMENT_1.md`**, sha256 `d5fc1d573afb51efbd58a03f30871403bc635fd4127fcbf80fbb78d372ad92a3`, frozen 20:47Z before any PDMS.
  - It changes only CTRL-PARITY's start-state bar. The harness cache stores velocity and acceleration rounded to float32; the pose is bit-identical.
  - The as-written FAIL stays printed in the read.
- **Unit tests: 8/8 PASS** (`run_tests.sh`, seam venv, CPU). A deliberate mutation (the RED start state not passed to the tracker) turned the RED test RED, as it should.
- **Smoke** (50 tokens, C1, GPU under the lock, 120 s): every control PASSES. `work/smoke/ctrl_smoke_c1.json`; the as-written result is `ctrl_smoke_c1.as_written.json`.
  - CTRL-ID 50/50, max |logit difference| 0.0.
  - CTRL-DET 50/50 picks, ship rows and proposals bit-identical to the banked C1 dump.
  - CTRL-RED: d0 min 1.000 m; the RED pick differs from the TRK pick on 29/50 tokens.
  - CTRL-SUB 50/50.
  - CTRL-PARITY: tracked states vs the harness route, max |Δxy| 4.7e-7 m and |Δh| 1.5e-8 rad. Dropping PDM-Closed from the batch changes nothing (0.0).
  - Mechanics only, not a result: TRK switches the pick on 26/50 tokens. The tracker costs 0.29 s per token. **No PDMS was read.**
- **The full run is LAUNCHED:** WMI, CreateFlags 0x08000200; cmd PID 53924, bash PID 58324.
  - It is waiting on `E:/Projects/TanitAD-artifacts/scorerft_read/markers_readp/read.DONE`, which is absent at hand-off.
  - Order: C1-TRK, then C0-TRK.
  - **ETA:** about 13 h after the gate opens (per arm: seam ~5.5–6 h, harness score ~1 h).

## How to check (one command, no model turns in a loop)
- `cat E:/Projects/TanitAD-artifacts/trkscore/work/CHAIN.log`.
- Markers are in `work/markers/`: `gate`, `C1.seam`, `C1.ctrl`, `C1.score`, `C0.*`, `chain.DONE` or `chain.FAILED`.
- Logs are in `work/logs/`. Each seam also writes its own log at `work/trk_<arm>/trk_<arm>.seam.log`.
- To resume after a failure, relaunch `E:\Projects\TanitAD-artifacts\trkscore\run_trk_chain.cmd` by the same WMI call. Use this launch copy: a quoted path containing `&` breaks `cmd /c`. The chain moves the failed marker aside and resumes.

## Artifacts the run writes
- Seams: `E:/Projects/TanitAD/data/refe_navtest/seams/stage2_trk_c1.{npz,proposals.npz,report.json}`, and the same for `trk_c0`.
- TRK dumps: `work/trk_c1/trk_c1.npz` (+ `.summary.json`, `.seam.log`), holding k_ship, k_id, k_trk, k_red, s_plan, s_trk, trk20, ship8, s0 and meta. The same for `trk_c0`.
- Controls: `work/trk_<arm>/ctrl_<ARM>.json`.
- Harness: `E:/Projects/TanitAD/data/refe_navtest/score/refe_stage2_trk_c1/refe_stage2_trk_c1.csv`.
- ⚠️ `stage2_sft.py` prints a `ZZSTAGE2 … ADOPT/NOT PROVEN` line and writes `result.json` under the tip's `raw/2026-10-05-sft-stage2/trk_c1/`. That line compares against the OLD deployed seam (`refe_navtest_final_routefix`). **It is not this card's reading. Do not quote it.**

## Exact next step (a fresh Sonnet agent, after `chain.DONE`)
1. **Write `trk_read.py`.** Mechanics only, following `PK/raw/2026-11-25-c1r-read/{run_post.sh,c1r_read.py}`. Arms TRK1 vs C1 and TRK0 vs C0.
   - (a) `p5c_read_prep.py convert` with `--arm TRK1=<stage2_trk_c1 npz,proposals> C1=<stage2_p5c_c1 …>` and the same for TRK0/C0.
   - (b) `score` (pass 1, navsim-crun, 2 processes).
   - (c) `identity` for all 4 arms against their CSVs (CTRL-T b).
   - (d) CTRL-T c: the TRK CSV equals `p5c_next/propscore/full_C{1,0}_s*of2.jsonl` `tr` at k_trk to 1e-6.
   - (e) `p5c_families.py --pairs TRK1:C1 TRK0:C0`, then `p5c_tactical.py`, then `strategic_family.py read` (STRATEGIC in full).
   - (f) Bars P1, P2 (0.2316), G and R exactly as in card §5. Controls from `ctrl_C1.json`, `ctrl_C0.json` and the smoke, plus CTRL-SHA (`CTRL-SHA PASS` in `work/logs/seam_<A>.log`) and CTRL-CODE (`CODE.sha256` against the dump meta `code`). M5 and M6 are descriptive.
   - ⛔ **No TRK-MIN number.**
2. Update `GOALS_AND_CLAIMS.md` `H-REFE-TRKSCORE-1` in the same turn as the read, starting from `gac_row_TRKSCORE.md`. Add a registry row on ADOPT.
3. Delete the HF re-pulled checkpoints, if any remain, and `arms/*/.cache` (4 MB). The chain deletes each `model_final.pt` itself after its score, and logs it.

## Deletions logged (this agent)
| path | size |
|---|---|
| scratchpad `probe_import.py` | 1,992 B |
| scratchpad `gac.md` | 5,548,783 B |
| scratchpad `p5c_prop_score.py` | 6,455 B |
| `work/tmp/s0_diag.py` (a copy; the original is staged) | 1,248 B |
| `work/tmp/mut/` (the mutation-test copy) | 31,698 B |

## Integration (for the MM)
Land this staged folder at `PK/raw/2026-11-26-trkscore/`. It holds the card, the amendment, the code, the tests, the chain, the `.sha256` files and this handoff. Then land the GAC row.
