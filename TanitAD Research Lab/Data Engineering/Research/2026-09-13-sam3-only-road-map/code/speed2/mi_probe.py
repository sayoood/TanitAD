"""Multi-instance SAM3 throughput on Thor's single GPU. For N in the list: start N mi_worker.py processes (approved spdF4a flags),
wait until all have built and warmed up, release them together, measure over a common window (after a 10 s settle) the frames
completed by all workers together, stop them, record. Nothing else runs on the GPU. Frames go to RAM disk; the npz files are the
same work the production driver does per frame.
Output: /home/nvidia/sam3map/mi_probe.json  Usage: mi_probe.py <N,N,...> <window s>"""
import json, os, shutil, subprocess, sys, time
from pathlib import Path

SM = Path("/home/nvidia/sam3map")
PY = "/home/nvidia/venvs/tanitad-edge/bin/python"
ENV = dict(os.environ, SAM3MAP_ROOT=str(SM / "native7"), PYTHONPATH="/home/nvidia/sam3vendor:/home/nvidia/sam3paint:/home/nvidia/sam3map",
           HF_HUB_OFFLINE="1", OMP_NUM_THREADS="4", MODE="fast", HALF="fp16enc", BATCH="19", ASYNC="2")
Ns = [int(x) for x in sys.argv[1].split(",")]; WIN = float(sys.argv[2]); SETTLE = 10.0
CLIPS = ["73495082f98b", "4fbd97b6a4b7"]
rep = {"flags": {k: ENV[k] for k in ("HALF", "BATCH", "ASYNC")}, "window_s": WIN, "settle_s": SETTLE, "runs": {}}
for n in Ns:
    run = SM / f"mi_run_{n}"
    if run.exists():
        run.rename(SM / f"mi_run_{n}_aside_{int(time.time())}")
    run.mkdir()
    procs = [subprocess.Popen([PY, str(SM / "eval" / "mi_worker.py"), str(run), str(w), CLIPS[w % 2] + "," + CLIPS[(w + 1) % 2]],
                             cwd=str(SM / "eval"), env=ENV, stdout=open(run / f"worker_{w}.log", "w"), stderr=subprocess.STDOUT) for w in range(n)]
    t_wait = time.time()
    while sum((run / f"ready_{w}").exists() for w in range(n)) < n:
        if any(p.poll() is not None for p in procs):
            break
        time.sleep(0.5)
    ready_s = time.time() - t_wait
    (run / "start").write_text("go"); t_start = time.monotonic()
    time.sleep(SETTLE + WIN + 2.0)
    (run / "stop").write_text("stop")
    for p in procs:
        p.wait(timeout=600)
    a, b = t_start + SETTLE, t_start + SETTLE + WIN
    per = {}
    for w in range(n):
        f = run / f"result_{w}.json"
        if not f.exists():
            per[w] = {"error": "no result", "rc": procs[w].returncode}; continue
        d = json.loads(f.read_text())
        inwin = [t for t in d["ts"] if a <= t < b]
        per[w] = {"frames_in_window": len(inwin), "fps": len(inwin) / WIN, "max_mem_gb": round(d["max_mem_gb"], 2), "build_and_warm_s": d["build_and_warm_s"]}
    tot = sum(v.get("fps", 0.0) for v in per.values())
    rep["runs"][n] = {"workers": per, "total_fps": tot, "s_per_frame_equiv": (1.0 / tot) if tot else None, "all_ready_after_s": round(ready_s, 1)}
    print(f"N={n}: total {tot:.3f} frames/s  ({(1.0 / tot) if tot else float('nan'):.3f} s/frame equivalent)  per worker " +
          ", ".join(f"{v.get('fps', 0):.3f}" for v in per.values()) + f"  mem/worker {[v.get('max_mem_gb') for v in per.values()]}", flush=True)
    for w in range(n):
        shm = Path(f"/dev/shm/mi_{run.name}_{w}")
        if shm.exists():
            shm.rename(Path(f"/dev/shm/mi_aside_{run.name}_{w}_{int(time.time())}"))
base = rep["runs"].get(1, {}).get("total_fps")
for n, r in rep["runs"].items():
    r["speedup_vs_1"] = (r["total_fps"] / base) if base else None
(SM / "mi_probe.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
print(json.dumps({n: (round(r["total_fps"], 3), None if r["speedup_vs_1"] is None else round(r["speedup_vs_1"], 2)) for n, r in rep["runs"].items()}))
print("ZZMIPROBE-DONEZZ")
