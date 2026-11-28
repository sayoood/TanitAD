# HANDOFF : latency proxy tool (S0 of COORD_D5), built on the dev box, not run with a model

State: built and checked without torch. `run_checks_pod.py` ran on the dev box (stub, pure python): ZZ CHECKS-POD DONE 50 0, 13 of 13 mutations killed. Everything that imports torch, the tree or a checkpoint is NOT RUN.

Files (sha256 prefix): latency_proxy.py d1a5a743de1c, pod_latency_run.sh ac5c9978d7ca, run_checks_pod.py f76b9a2b2b88, LATENCY_DEFS.md (definitions).

Pod commands, in order (copy the latency/ folder next to imagplan/, distrule/, imaghead/ or set LAT_LIBDIRS):
1. `cd <latency>; /workspace/venv/bin/python run_checks_pod.py --work-dir /workspace/v8proof/runs/latency_checks` : last line must be `ZZ CHECKS-POD DONE <n> 0`.
2. `DRY=1 bash pod_latency_run.sh /workspace/v8proof/tree_leadsup /workspace/v8proof/runs/<chain>/<ARM>/ckpt_last.pt` : read the commands.
3. `nohup bash pod_latency_run.sh <tree> <ckpt> [<ckpt>...] > /workspace/v8proof/runs/latency_proxy.nohup 2>&1 &` ; watch `/workspace/v8proof/runs/latency_proxy/latency.log` for `ZZ LATENCY DONE` or `ZZ LATENCY STOP: <reason>`.
Result: `runs/latency_proxy/<chain>_<ARM>/LATENCY_PROXY.json` and `.md`. Expected: checks about 1 min, slot wait 0 to 6 h, about 5 to 15 min per checkpoint.

REVIEW items:
1. Real path (model load, window build, ImagPlanner tick, count wrappers, fp16, compile) never executed: first pod run may need a fix; a failing section is written as NOT RUN with its error.
2. Loader is `imag_dump.load_model` (same construction as `T.cmd_run` lines 505-513, which are inline).
3. Horizon default 10 steps = 2.0 s (planner's `N_ROLL`), not the 6 s of the card; 30 steps timed as an extra.
4. Tick time includes host numpy work; two planner configurations (cand-imag/scale 11 candidates, cand-imagkin/decel 8).
5. s1 tolerates 5 ms: a busy host can fail it; rerun.
6. Compile timing covers regions a and b at the planner horizon only.
7. Slot-wait logic (probe, two probes, 6 h stop) of the shell script was read, `bash -n` and DRY=1 checked, not executed.
8. checks_work/ (dev-box runner output, small) was not deleted.
