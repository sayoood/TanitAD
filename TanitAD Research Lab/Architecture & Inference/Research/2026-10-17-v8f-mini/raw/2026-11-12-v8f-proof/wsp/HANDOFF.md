# HANDOFF: v8F-PROOF build (H-V8F-PROOF-1 SOUP, -2 TASKVEC / P2b)

Card sha256 38d33abd...c3a7 re-checked equal to PREREG_V8F_PROOF.sha256. Everything lives in `F:/TanitAD-scratch/v8proof/wsp/`. Nothing staged, committed or pushed; no GPU, no ssh.

## What exists
- `tree/` : `stack/` and `taniteval/` of 3e264bbf (git archive, autocrlf off; taniteval is needed by v8f_t1_eval.py; its 8 instrument pins equal TREE_PINS_S1.json).
- `wsp_build.py` + `BUILD_REPORT.json` + `ckpt/{SOUP2,SOUPa0,SOUPa1,DOSE050,DOSE100}/ckpt_last.pt` (strict-load round trip OK, WSP_MANIFEST.json each). DONE.
- `wsp_taskvec.py` + `TASKVEC.json` / `TASKVEC.md`. DONE (P2a outcome inside).
- `wsp_dump.py`, `wsp_read.py`, `wsp_common.py`, `job_wsp.cmd`, `launch_wsp.ps1`.
- `selftest/` (SOUP2, 4 windows, strict load, rc 0, 14.3 s), `smoke/` (reader run on STAND-IN banked dumps, never a result).

## Launch (Master Mind)
`powershell -File F:\TanitAD-scratch\v8proof\wsp\launch_wsp.ps1`  (detached, `< NUL`, log `run_wsp_<stamp>.log`).
Markers: `WSP_DONE.txt` (first line `ZZ WSP-DONE`) or `WSP_STOP.txt` (`ZZ WSP-STOP <reason>`).
Order: 9 T1-kin dumps (2 threads, CPU) -> STOPDEC of NULL_s0 NULL_rep G_s0 G2_s0 DOSE050 DOSE100 -> wsp_read.py -> `RESULT_WSP.json`, `VERDICT_WSP.md`.
Expected runtime ESTIMATED 1.0 to 1.3 h (dump 301.5 s MEASURED for BAL-G2_s0 per README_READ_S1.md; STOPDEC and read ~15 to 25 min; reader on stand-ins 149 s H1, 23.5 s H2, MEASURED in smoke/*.log).
Resumable: dumps with a done file are skipped.

## NOT verified
- No real SOUP / DOSE dump or STOPDEC of our tags exists yet; the reader ran only on stand-ins (SOUP2 := BAL-G_s0), H2 stand-in reproduced NULL_s0 0.0598/0.0054 and G2 0.4530/0.1195.
- STOPDEC on our tags: stopdec_arm.py `--selftest` passed (tree and shim), the full run did not.
- Dump thread count 2 for dumps; STOPDEC keeps read_lib's torch.set_num_threads(4) as in the banked reads.
- ESTIMATED runtimes. Free RAM gate 4 GB in wsp_dump.py.
- Reading choices on the card's gaps are listed in the wsp_read.py docstring (kappa read on ADE; mixtures printed as UNSPECIFIED-MIXTURE).
- Temp: `selftest/`, `smoke/`, `tree/` (about 350 MB), `ckpt/` (5 x 29 MB without optimizer state) are deletable by literal path once the read is banked.
