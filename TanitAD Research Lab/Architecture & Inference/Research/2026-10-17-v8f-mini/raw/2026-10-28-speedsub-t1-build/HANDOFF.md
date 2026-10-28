# HANDOFF: SPEEDSUB-T1 instrument build (PREREG_V8F_SPEEDSUB_T1 @ 842ec2ac)

State at hand-off: all five deliverables are built and tested; NOTHING of this experiment has been rolled out (no smoke, no rollout, no GPU, box 2 untouched, nothing staged or committed).
No detached job is running. The final reader mutation pass ended 110/110 killed, no survivor (`evidence/mutation_reader.log`).

Where things are: `E:/Projects/TanitAD-artifacts/dagland/speedsub_t1/build/` (`MANIFEST.txt` = file, sha256, proposed repo landing place; `NOTES.md` = what is proven, what not, and the PREREG amendments needed; `evidence/` = dry-run, self-test, fit, mutation and dry-run JSONs). Scratch: `F:/TanitAD-scratch/speedsub_t1/` (fit/ with `readout_c.pt` sha 799f65ed..., tree/ = a tip checkout the tests run in, out_dry/, selftest/, logs/).

Next step for the Master Mind (in this order):
1. Settle the PREREG amendment items in NOTES section 7 (G-VIOL amendment + the reader pin, the tick-frame reading, the G-NEW reading, the smoke control set).
2. Land the files per MANIFEST (stack/scripts, stack/tests, and the Research Lab raw dir). Run on the landed tip: `pytest -q stack/tests/test_v8f_t1_speedsub.py stack/tests/test_v8f_speedsub_t1_pin.py` (CPU, `CUDA_VISIBLE_DEVICES=-1`), then `bash <code dir>/run_speedsub_t1.sh --dry-run` with `SST1_CODE` pointing at the landed `stack/scripts` and `SST1_READER` at the landed reader: it must end `ARGCHECK 18/18 ok` and `ZZSST1-DRY-DONE`.
3. Launch detached: `powershell -NoProfile -Command "Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='cmd /c <code dir>\run_speedsub_t1.cmd < NUL'}"`. Log `F:\TanitAD-scratch\speedsub_t1\logs\chain_speedsub_t1.out`; wake on `ZZSST1-STOP` or `ZZSST1-CHAIN-DONE` with a script poller, not model turns. The chain stops by itself at the smoke check and at the CTRL s0 preflight if a control fails, and resumes from its done records.
4. The decisive file is `F:/TanitAD-scratch/speedsub_t1/out/SPEEDSUB_T1_READ.json` (OUTCOME, label, guards, controls, Read A on PASS, the floor rule beside); the logged-lane bridge is `SPEEDSUB_T1_BESIDE.json` and decides nothing.

Do not: edit the pinned tool tree, re-fit the read-out (pins.json holds its sha), re-run the chain over an existing `out/` with different code (the pin step stops on any difference), or read the beside lane as a capability row.
