# HANDOFF: L1b ENS-SCORE (`H-REFE-ENSSCORE-1`), 2026-11-26 ~21:25Z

## State
- **The card is frozen:** `PREREG_ENSSCORE.md`, sha256 `7423cf9dabe92915a8a4c4532878a68a813591fc7fd14ab083c8a98c33466d63`, frozen ~21:19Z. No ENS number exists, and no real token has been through the ENS path.
- **Unit tests: 8/8 PASS** (`run_tests.sh`, seam venv driverl-eval, CPU, tiny random REFe models).
  - **Mutation 1** (the member term removed from `ens_probs`): 4 tests went RED (member-reaches-pick, RED, end-to-end, numpy recompute).
  - **Mutation 2** (the member's yaw shim not installed): the yaw test went RED.
  - The self-ensemble identity stays green under mutation 1. That is the documented blind spot of CTRL-ID, which is why CTRL-RED and CTRL-REACH exist.
- **C1r was pulled** from HF `Sayood/tanitad-refe-p5c` `p5c/c1r/model_final.pt` to `E:/Projects/TanitAD-artifacts/ensscore/arms/C1r/model_final.pt`.
  - Its sha256 is `6e39a57c…54a8`, equal to MODEL_REGISTRY 14.6; 1,288,515,000 B.
- **The smoke has NOT run yet.** The GPU lock was held by `refe-stage2-scorerft_p_rep` and commit free was ~2–3 GB, so neither the GPU nor a CPU smoke was admissible. The smoke is stage 2 of the chain.
- **The chain is LAUNCHED:** WMI, CreateFlags 0x08000200; cmd PID 67420, bash PID 22220. It started 21:20:36Z, CODE.sha256 verified, and is waiting on `E:/Projects/TanitAD-artifacts/trkscore/work/markers/chain.DONE`.
  - **Order:**
    1. the gate;
    2. the unit tests;
    3. pull C1 (`p5c/c1`, sha-verified), and re-verify C1r;
    4. the smoke: 50 tokens, ENS1, run twice; it reads controls only, and any FAIL stops the chain;
    5. ENS1, its seam, controls and score;
    6. ENS1r, the same;
    7. delete both checkpoints, logged.
  - **ETA (ESTIMATED):** about 17–20 h after TRK's chain.DONE.
  - GPU memory for two vitl16 models on the 8 GB card is **UNVERIFIED**. The smoke measures it (`cuda_max_mem_mb` in the dump meta). An out-of-memory failure shows up as `chain.FAILED stage=smoke_seam_1`.

## How to check (one command; no model turns in a loop)
- `cat E:/Projects/TanitAD-artifacts/ensscore/work/CHAIN.log`.
- Markers are in `work/markers/`: `gate`, `tests`, `smoke_run{1,2}`, `smoke`, `ENS1.{seam,ctrl,score}`, `ENS1r.*`, `ckpt_deleted`, then `chain.DONE` or `chain.FAILED`.
- Logs are in `work/logs/`: `seam_<name>.log`, `ctrl_<tag>.log`, `score_<ARM>_<n>.log`. Each seam also writes its own log next to its dump (`*.seam.log`).
- Smoke controls: `work/smoke/ctrl_smoke.json`.
- To resume after a failure, relaunch `E:\Projects\TanitAD-artifacts\ensscore\run_ens_chain.cmd` by the same WMI call (`cmd.exe /c <that path>`). The chain sets the failed marker aside and resumes. ⛔ Fix only by an amendment frozen before any PDMS.

## Artifacts the run writes
- Seams: `E:/Projects/TanitAD/data/refe_navtest/seams/stage2_ens_c1.{npz,proposals.npz,report.json}`, and `stage2_ens_c1r.*`.
- ENS dumps: `work/ens_ens1/ens_ens1.npz` and `work/ens_ens1r/ens_ens1r.npz`, each with `.summary.json` and `.seam.log`.
  - They hold k_ship, k_ens, k_id, k_red, k_a_only, k_b_only, k_b_own, k_b_own_id, s_a, s_b, s_b_fwd, a_ens, ship8, ship8_b, props8_b and meta.
  - ⚠️ The arm name is lowercased, so the seams are `ens_ens1` / `ens_ens1r`, **not** `ens_c1`. The seam files are `stage2_ens_ens1*` and `stage2_ens_ens1r*`. Read CHAIN.log for the exact names.
- Controls: `work/ens_ens1/ctrl_ENS1.json` and `work/ens_ens1r/ctrl_ENS1r.json`.
- Harness: `E:/Projects/TanitAD/data/refe_navtest/score/refe_stage2_ens_ens1/refe_stage2_ens_ens1.csv` (and the same for ens1r).
- ⚠️ `stage2_sft.py`'s `ZZSTAGE2 … ADOPT/NOT PROVEN` line compares against the OLD deployed seam. **It is not this card's reading. Do not quote it.**

## Exact next step (a fresh Sonnet agent, after `chain.DONE`)
1. **Write `ens_read.py`.** Mechanics only, following `PK/raw/2026-11-25-c1r-read/{run_post.sh,c1r_read.py}`. Arms ENS1 vs C1 and ENS1r vs C1r.
   - `p5c_read_prep.py convert/score/identity` (CTRL-T b).
   - CTRL-T c: the ENS1 CSV equals `p5c_next/propscore/full_C1_s*of2.jsonl` `tr` at k_ens to 1e-6. It is n/a for ENS1r, by name.
   - `p5c_families.py --pairs ENS1:C1 ENS1r:C1r`, `p5c_tactical.py`, and `strategic_family.py read` (STRATEGIC in full).
   - Bars P1, P2 (0.2316), G and R exactly as in card §5. Controls come from the ctrl JSONs, the smoke, CTRL-SHA (`CTRL-SHA PASS` in `work/logs/seam_*.log`) and CTRL-CODE (`CODE.sha256`).
   - M5 and M6 are descriptive. ⛔ **No ENS-MIN number** under this card.
2. Update GAC `H-REFE-ENSSCORE-1` (from `gac_row_ENSSCORE.md`) in the same turn. Add a registry row on ADOPT.

## Deletions logged (this agent)
| path | size |
|---|---|
| `E:/Projects/TanitAD-artifacts/ensscore/work/tmp/mut1` (mutation copy) | 45,885 B |
| `E:/Projects/TanitAD-artifacts/ensscore/work/tmp/mut2` (mutation copy) | 45,840 B |
| `<stage>/__pycache__` (py_compile) | 74,484 B |
| scratchpad `ref/` (git-show exports of the TRK, C1r-read and plan files; re-creatable) | logged in the hand-back |

## Integration (for the MM)
- Land this staged folder, `E:/Projects/TanitAD-artifacts/ensscore/stage/<PK>/raw/2026-11-27-ensscore/`, at `PK/raw/2026-11-27-ensscore/`. It holds the card, its sha, `CODE.sha256`, the code, the tests, the chain, the cmd, this handoff and the GAC row.
- Then land the GAC row.
- ⚠️ The chain runs FROM the staged path. Do not move or delete the stage folder until `chain.DONE`.
