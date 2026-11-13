# HANDOFF: FULLRES probe (PREREG_V8F_PROOF_ADD9)

State: code written and unit-tested on CPU (10 tests pass); the real read and the smoke were NOT run (GPU busy with the encode; the encode was not finished while I worked: `feat/progress.json` and `ENCODE_DONE.json` absent, only `index.npz`, N = 14,035 grid frames).

Files (all in F:/TanitAD-scratch/v8proof/full/):
* `fullres_probe.py` (`run`, `smoke`), `test_fullres_probe.py`, `FULL_DEFS.md` (choices R1..R15), `job_probe.cmd` (CRLF, not launched), this file.
* Output when run: `RESULT_FULLRES.json`, `VERDICT_FULLRES.md`, `OOF_FULLRES.npz`, `kern/` (5 Grams + planted U640 Gram, float32, 11.5k x 11.5k each ~0.5 GB, plus `stoplabels.npz`), `stages/*.npz`, `PROBE_DONE.txt` / `PROBE_STOP.txt`, `probe_job.log`.

Launch order for the lead:
1. After `feat/progress.json` lists >= 10 episodes: smoke (needs the GPU lock to be free; it takes the lock itself): `set PYTHONPATH=F:\TanitAD-scratch\v8proof\res\tree\stack;F:\TanitAD-scratch\v8proof\res\tree\taniteval` then `C:\Users\Admin\venvs\tanitad\Scripts\python.exe fullres_probe.py smoke` (CUDA_VISIBLE_DEVICES must NOT be -1). Expect `SMOKE_OK.txt`. The smoke is the first run of the GPU paths (token sets, Gram, P3 on cuda, label loader via cueprobe.rs()); the CPU paths were run on synthetic data only.
2. After `ENCODE_DONE.json`: launch `job_probe.cmd` via WMI (`cmd /c ... < NUL`). It waits for free commit >= 10 GB.

Expected runtime (ESTIMATED, not measured): stage A ego baselines on the CPU 10 to 25 min (MLP P2 on 6 targets x 5 folds); Grams ~5 to 10 min; P3 ~ 75 main fits + 30 f4 + 30 f5 (+15 TEMPORAL) at roughly 0.5 to 3 min each on the 4060 = 1.5 to 4 h; P1 on 5 sets + planted ~ 30 to 60 min on 4 CPU threads. GPU needed ~3 GB (H1664 int8 set 2.4 GB) beside the lock holder.

Known risks, check before launch:
* The P1 stage loads one 0.5 GB Gram and makes float64 sub-matrices (~0.7 GB each, several copies): peak ~3 to 4 GB RAM; the job gate is 10 GB free commit.
* `get_stop_labels` imports `ruleshield_read` through cueprobe: it needs the res/tree PYTHONPATH and sets CUDA_VISIBLE_DEVICES only by setdefault (the probe sets it to 0 first). If it fails, the smoke fails first (good).
* f5 at T = 640 learned perfectly in a CPU trial of mine on synthetic Gaussian tokens (AUROC 1.0, 29 epochs), but real tokens differ; a miss exits 2 with the numbers.
* STOPDIST has only 131 windows and gets ~3 rows per batch in the multi-task stream; P3 on it may be weak (reported, not tuned).
* Windows of the same clip use the bank fold map; their 2,399 rows are not in the bank's addendum-1 fold definition by construction, only by clip (R2).
* Temp files by me: none left under full/ except those listed above as outputs; my scratch scripts live in the session scratchpad, not here.
