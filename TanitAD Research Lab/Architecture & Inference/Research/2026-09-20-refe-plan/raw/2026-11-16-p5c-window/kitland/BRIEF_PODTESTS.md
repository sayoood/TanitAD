# BRIEF: run the P5-C kit-land test list on the POD (the dev box is memory-blocked). Agent: Sonnet. MM, 2026-11-16

Read `Project Steering/AGENT_OPERATING_STANDARD.md` (repo `E:/Projects/TanitAD`) first.
- **Binding:** stage, never commit or push; end with a manifest.
- **Hold:** `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_kitland/`. Put your results in `pod_run/` there.
- Read `HANDOFF.md` in the hold first. It describes the tree and the test list.

## Why
The kit-land tree has 3 pins filled in `refe/launch_gate.py` and must pass every launch_gate-reading test before the MM lands it.
- The dev-box driver `F:/tanitad_scratch/kitland_tmp/drive.sh` waits on free commit ≥ 5 GB, and the box sits at ~2.6 GB while a P7 eval runs. **Leave that driver alone.** Do not stop it and do not edit it.
- The pod is idle, and its launch gate later runs the same suite on Linux, so a pod run is the better check anyway.

## Inputs
- **Tree:** `F:/tanitad_scratch/kitland_tmp/t/` (193 MB: the tip export + the kit + the 2 A3.1 files + the 3 pins).
- **Script list:** `F:/tanitad_scratch/kitland_tmp/list.txt`, read by `drive.sh`. Lines are `<script> <args>`, run from `t/<PK>/refe` with `PYTHONPATH=<tree>/stack`. `PK` = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`.
- **Pod python:** `/workspace/venv-teacher/bin/python` (torch 2.7.1). There is no `/workspace/venv` on this pod.

## Steps
1. **Ship the tree.**
   - Tar `t/` locally into your scratch `F:/tanitad_scratch/kitland_pod/`; GNU tar on F: paths needs `--force-local`.
   - Ship it with `bash C:/Users/Admin/par18/pd.sh 600 'cat > /workspace/data/kitland/t.tar' < t.tar`.
   - Verify sha256 on both sides, then extract to `/workspace/data/kitland/t/`.
   - Ship `list.txt` too.
2. **Pod driver** (write it locally, ship it, run it detached with `setsid nohup ... < /dev/null &`):
   - For each list line, run `cd <tree>/<PK>/refe` and `env CUDA_VISIBLE_DEVICES= PYTHONPATH=<tree>/stack TMP=TEMP=TMPDIR=<per-script tmp> timeout 1500 nice -n 10 python <script> <args>`.
   - Write the rc, the seconds and the last `ZZ...PASS...` token per script to `summary.txt`, and end with `ZZKLPOD_END`.
   - Up to 4 scripts may run at once, each with its own tmp; the pod has 15 CPUs. Set `OMP_NUM_THREADS=3`.
3. **Wait with a script loop** (a background bash waiter that greps for `ZZKLPOD_END`), never in model turns. Expect under 1 h.
4. **Classify every non-zero rc:**
   - (a) **PIN-CAUSED:** the assertion reads one of the 3 pins or a profile fingerprint that includes them. Prove it by re-running that one script on the same tree with the 3 pin lines reverted to `None`; it must then pass.
   - (b) **ENVIRONMENTAL on the pod:** for example, a Windows path, a missing dev-box file or a missing package. Prove it by the same revert-and-re-run: it still fails without the pins.
   - (c) **anything else.**
   - Do not edit anything to make a test pass. Report each case with its log excerpt.
5. **Pull the results** into `pod_run/`: `summary.txt` and the logs of failing scripts only. Pull them with `pd.sh 'cat ...' > file` with no sed filter, then mask any IPv4-looking string yourself before writing into the hold. Write `pod_run/RESULT.md` with a table (script → rc → token → class).

## Pod rules (binding)
- **Run commands as:** `bash C:/Users/Admin/par18/pd.sh <t> '<cmd>' 2>/dev/null | sed -E 's/[0-9]{1,3}(\.[0-9]{1,3}){3}/<IP>/g; /hf_/d'`.
- ⛔ Never print `C:/Users/Admin/freept/pod_direct.txt`, IPs, ports or tokens.
- **Processes:** no GPU. Kill only your own processes, by PID. Never use `pkill -f` or `pgrep -f`: they self-match your ssh command. Match the executable, e.g. `ps -eo pid,args | awk '$2 ~ /python$/'`.
- **Paths:** write only under `/workspace/data/kitland/`. Never touch `/workspace/data/p5c_*`, `/workspace/data/refe_runs`, `/workspace/data/rj5c_*` or `/workspace/data/p7_zg`.
- **At the end:** delete `/workspace/data/kitland/t.tar` once the tree is extracted and verified, and your local scratch tar. Use literal paths and log sizes. KEEP `/workspace/data/kitland/t/` (the MM may re-use it for the launch gate).

**Report:** at most 300 words, plus a manifest.
