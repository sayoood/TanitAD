# HANDOFF: read_div.py + post_div.py (PREREG_DIV_P6, card sha256 e5fb2f8e...f6bacf). Neither real step has been run.

## Files (this folder)
`read_div.py` (reader; reuses read_pess.py, read_scorer1p6.py, selcore unchanged), `post_div.py` (dev-box orchestrator), `test_read_div.py` (C3 and more, 24 tests), `test_results.json`, `control_out/` (C1, C2 outputs).

## Controls (all PASS; they read only arms already read; none writes `selectors/ledger_K.json`)
- C1 PESS-8 rebuilt from the eight members (banked order): sha256 `8912f1d04ab3e22571397d53580ddafebc3beb78e1e27d34fe21db25e5bf7ae5` = banked; PDMS x100 91.47901718634304 exactly. `control_out/c1/div_read_control_c1.json`, 21.9 s.
- C2 PESS over S1ZSM alone: PDMS x100 91.29707529608928 exactly = banked; logits bit-equal to the member. `control_out/c2/div_read_control_c2.json`, 5.1 s.
- C3 `test_read_div.py`: 24 passed (every outcome branch DV1/DV2/DV3/Unnamed, overlap rule, strict bar predicates, synthetic Q1/Q2/M end-to-end, card-sha refusal in `div_read` and `main`, wrong or reordered member set refused, post_div helpers).
- Rerun: `CUDA_VISIBLE_DEVICES=-1 C:/Users/Admin/venvs/tanitad/Scripts/python.exe read_div.py read --control c1|c2 --out <dir>`; tests: `... -m pytest -q test_read_div.py --basetemp F:/tanitad_scratch/divread_build/tmp` (create the parent first).

## The real read (counts K + 5; run only after DV0 and DV1 passes exist)
```
set CUDA_VISIBLE_DEVICES=-1
C:/Users/Admin/venvs/tanitad/Scripts/python.exe E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p6base/div/read/read_div.py read --out F:/tanitad_scratch/divread/real
```
Members default to the ten dirs in `read_div.MEMBER_DIRS` (DV0/DV1 = `F:/tanitad_scratch/p6arms/pass_div/DV0|DV1`); `--member NAME=DIR` overrides; `--no-ledger` = dry run (no K key). Exit 0 READ, 2 = card or gate failed (no verdict).
Gates: the read_pess G0 set for P6 and all ten members, plus G0-PESS8 (the rebuilt PESS-8 must equal the banked sha and PDMS exactly, else READ STOPPED).
Outputs: `div_read.json` / `div_read.md` (arms DV0, DV1, PESS-DV, PESS-SEED, PESS-10 full blocks with G1-G4 and families vs P6; PESS-8, C0, C0r as references; bars Q1, Q2a, Q2b, M; DV0-C0 and DV1-C0r; outcome; K block with Bonferroni intervals).
Runtime/RAM: about 30 to 60 s with ten members (C1 with eight: 22 s); peak RAM about 3 GB (waits for `--min-free-ram-gb`, default 3).

## post_div.py
`post_div.py <T0 YYYY-MM-DDTHH:MM:SS> [--chain-log P] [--done-pattern TEXT] [--fail-pattern TEXT] [--no-ledger] [--no-wait]`; launch detached via WMI `cmd /c ... < NUL`. Flow: wait for the pod chain DONE line, pull `div/dv0/model_sft_B.pt` and `div/dv1/model_sft_B.pt` (LFS sha256 checked, token read in place), ONE `chain_arm_pass.py --out F:/tanitad_scratch/p6arms/pass_div --arm DV0=.. --arm DV1=..` (it takes several `--arm`), then `read_div.py read`. Markers `F:/tanitad_scratch/p6arms/post_div.DONE|.FAILED`, log `post_div.log`, step logs `post_div_armpass.log`, `post_div_read.log`.
Lessons built in: 0-byte `_progress/part_N.npz.tmp.npz` removed only if exactly 0 bytes, logged, before the pass and before its one retry (non-empty tmp files are left and warned); the GPU lock file is never touched (chain_arm_pass takes it). The table lookup step of post_dose (`changed_from_table.py`) is not run: read_div scores by table lookup itself.

## To confirm (not verifiable here; I did not read `p6base/div/files/`)
1. Pod chain log path: default `/workspace/data/refe_div/chain_div.log` (CLI `--chain-log`).
2. DONE line: default `ZZDIV_CHAIN_DONE` (CLI `--done-pattern`); failure patterns default `ZZDIV_CHAIN_FAIL`, `REFUSED` (`--fail-pattern`). The wait greps for either, with lines stamped at or after T0 (the log lines must start with an ISO timestamp as in the dose chain).
3. HF paths `div/dv0/model_sft_B.pt`, `div/dv1/model_sft_B.pt` in `Sayood/tanitad-refe-scorer1p6` (as given in the task).
4. Readers' interpretation points (listed D1 to D8 in the output): Q2 needs both parts (point > 0.4971 strict with CI-lo > 0 vs P6, and CI-lo > 0 vs PESS-8); M is not a condition of any label; overlapping labels all listed, primary DV1, DV2, DV3; PESS-10 member order = the banked eight then DV0, DV1 (order is in the sha).
5. Ledger: PESS-SEED is read against truth in the real read (K counted); controls never do it. If any of the five keys is already in the ledger it is not re-added, so K + 5 would read less: the output prints K_before and K.
