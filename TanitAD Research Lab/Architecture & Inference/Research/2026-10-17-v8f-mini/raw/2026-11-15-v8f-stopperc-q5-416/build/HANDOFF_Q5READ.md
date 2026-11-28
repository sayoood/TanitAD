# HANDOFF Q5READ (2026-11-15). READ side of STOPPERC Q5 built; synthetic only, no real data.

## Built and passing (MEASURED, dev-box venv, OMP_NUM_THREADS=2)
`test_q5.py` 19 passed + `test_q5_read.py` 15 passed = 34 (two runs: 33+1 after a test-code fix of the encoding scan, then that test and the C3/plan/HU tests green; the full 34 were not re-run in one call after the last test edit).
* `q5_controls.py`: C5 (20 seeds 100-119, real hyperparameters, window-level shuffle, single-draw CI printed), C3a-e. Regression arms RED: C5 unshuffled, C3a without label, C3b nonzero/absent, C3c substitution skipped (BLIND), C3d overlap, C3e one flipped y byte / one ulp in E / dtype change.
* `q5_read.py` (edited; original `q5_read.py.before_q5read`): computes C3a/b/c/d/e, C5 and C1 (identity) from the dirs; C6a now <= 1 count (Amendment 7a); beside arms gathered, bootstrapped in the SAME call per sample as the outcome inputs, printed under `beside`, never an argument of `decide`.
* `q5_fit_extra.py`: V40@416+(E-limit) arm, s0p_fit.py unedited (sha checked by test).
* `q5_fitplan.py`: `assemble`, `plan`, `run`. Dry-run on synthetic dirs ran assemble, 40 fit-side calls and the read end to end (outcome valid, controls all ok, beside rows present, blocked ones named).

## Beside arms
Computed when inputs exist: route term (needs `work256b`), M@416 vs M@256p, V40@416+(E-limit) vs E-limit, HEAD-UNDERUSES (needs STOPDEC `stopdec_BAL-NULL_s0/_s1.npz` under `s0p_fit.STOPDEC_OUT`).
BLOCKED, with the reason printed in the output:
* P-B vs P-A: `<work416pb>/fit_V40+E_{linear,mlp}.npy`. The pod rows hold only P-A pooled arrays; needs a P-B pool variant plus a 416 NOW-frame re-encode for S1 + eval (~0.6 h pod GPU).
* V416tok+E 13x32: `<work416tok>/fit_V416tok+E_linear.npy`. No 13x32 dump exists; needs the sibling encoder to keep the raw 416-token field ([N,416,1024] fp8, ~6.8 GB / 16,000 windows) plus a token-grid arm wrapper.
* Route term S2: needs C0's S2 fits on box-2 tokens (`banked256b/S2`). S1: S0-P's banked `fit_V40+E_*.npy`, `fit_V160+E_*.npy`.
* C3c: needs `c0/sidecar_limit_eval.npy` ([N,4] logged `limit`, eval windows). Not produced by any extractor (they assert causal). Without it C3c is a MISS and the read REFUSES. Produce it with a read-only pass of `s0p_extract_eval.py`'s loop with `build_item(..., limit='logged')` saving `batch['limit']`.

## Not built (still external inputs; null in `read_cfg.json` = refuse)
C0, C2 (planted-direction fits), C4, C6a/b/c values, C7, C8, C9 counts. Fill them in `read_cfg.json["controls"]` from the pod ledger/hashes, `test_q5.py`, `smoke_c8.json`.

## Commands after the pod build and the C0 run
Layout under B (e.g. `F:/TanitAD-scratch/q5_416/run`): `pod_out/{S1,S2}_{416,256p}`, `pod_out/eval_{416,256p}` (pulled `gather` output), `c0/{S1,S2}`, `c0/s0p_eval.npz`, `c0/sidecar_limit_eval.npy`, optional `banked256b/{S1,S2}`.
```
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe; cd F:/TanitAD-scratch/q5_416/build; export OMP_NUM_THREADS=2
$PY q5_fitplan.py assemble --base B          # asserts pod y/clip/ws/eid == C0's, writes B/assembled
$PY q5_fitplan.py plan --base B              # writes B/fitplan.sh (40 calls) and B/read_cfg.json
# fill B/read_cfg.json controls, then:  bash B/fitplan.sh   (or: $PY q5_fitplan.py run --base B)   ~4-5 h at 2 threads
$PY q5_read.py B/read_cfg.json               # controls first (any miss REFUSES), then outcome + beside; writes B/RESULT_Q5.json
```
Note: the plan fits E / E-limit once per sample (416 work dir); the 256p dirs get M+E, V40+E, V160+E only.

## Deletions (logged)
`F:/TanitAD-scratch/q5_416/build/tmp_pytest_agent` 2.3 GB, `.../pytest_run2.log` 4 KB, `.../.pytest_cache` 8 KB: all deleted. Left from the builder (not mine): `build/c6a_probe`, `q5_416/tmp_pytest`.

## Manifest (all in F:/TanitAD-scratch/q5_416/build/, single copy, not staged)
q5_controls.py, q5_fit_extra.py, q5_fitplan.py, q5_read.py (+ .before_q5read), test_q5.py (unedited; .before_q5read copy), test_q5_read.py, HANDOFF_Q5READ.md.
