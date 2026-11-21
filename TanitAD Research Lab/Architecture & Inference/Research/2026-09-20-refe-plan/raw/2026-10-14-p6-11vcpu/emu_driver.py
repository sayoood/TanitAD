#!/usr/bin/env python3
"""Dev-box EMULATION of the pod's labeller pool: the REAL OnlineLabeller (spawn ProcessPoolExecutor, cached parts) on 11 logical CPUs of the i9-12900F
(P-core SMT threads 0-7 and 12-14 = 5.5 physical cores: one vCPU = one hyperthread, as on a cloud host), alone or beside a trainer stand-in
(1 spinning process + 2 loader stand-ins at 0.25 core each; bench_loader.json: ~0.05 CPU-s/sample).
Resumable: an arm whose JSON exists is skipped.    python emu_driver.py <out_dir> <emu_cache_dir> [only-arm-name-prefix ...]
This is a dev-box measurement on a SHARED box: every arm records the background load inside its CPU set (sys_busy_cores_in_affinity minus ours minus the burners).
Its numbers are in DEV-BOX seconds: a pod vCPU is slower by an UNMEASURED ratio r (the probe measures the pod itself)."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent
PK = HERE.parents[1]
OUT, CACHE = Path(sys.argv[1]), Path(sys.argv[2])
ONLY = tuple(sys.argv[3:])
OUT.mkdir(parents=True, exist_ok=True)
PY = "C:/Users/Admin/navsim-crun/venv/Scripts/python.exe"
NAV = "E:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896"
AFF = "0-7,12-14"
TOKS = ",".join(Path(CACHE, "tokens.txt").read_text().split(",")[:128])
BURN = [1.0, 0.25, 0.25]            # trainer main (spins) + 2 loader stand-ins


def env(omp):
    e = dict(os.environ)
    e.update(PYTHONIOENCODING="utf-8", REFE_SIM_HZ="10", NUPLAN_MAPS_ROOT="E:/Archive/devbox-C/navsim/data/maps", NUPLAN_MAP_VERSION="nuplan-maps-v1.0",
             PYTHONPATH=f"{NAV};{PK / 'refe'}", OMP_NUM_THREADS=str(omp))
    if omp == 1:
        e.update(OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    else:
        e.pop("OPENBLAS_NUM_THREADS", None)
        e.pop("MKL_NUM_THREADS", None)
    return e


def tree(procs):
    """On Windows the venv's python.exe is a launcher: the interpreter that spins is its CHILD, so CPU and termination must cover the whole tree."""
    out = []
    for b in procs:
        try:
            pr = psutil.Process(b.pid)
            out += [pr] + pr.children(recursive=True)
        except Exception:
            pass
    return out


def cpu_of(ps):
    tot = 0.0
    for pr in ps:
        try:
            t = pr.cpu_times()
            tot += t.user + t.system
        except Exception:
            pass
    return tot


def arm(name, w, omp, burn=False, mode="thru", n=256, period=None, nmicro=60, chunks=1):
    out = OUT / f"{name}.json"
    if out.exists() or (ONLY and not name.startswith(ONLY)):
        return
    bp = []
    if burn:
        bp = [subprocess.Popen([PY, str(HERE / "emu_burner.py"), str(d), AFF]) for d in BURN]
        time.sleep(3)
    bt = tree(bp)
    cmd = [PY, str(PK / "refe" / "p6_cpu_probe.py"), "pool", "--cache", str(CACHE), "--tokens", TOKS, "--workers", str(w), "--n-sets", str(n), "--mode", mode,
           "--affinity", AFF, "--out", str(out), "--n-micro", str(nmicro), "--chunks", str(chunks)]
    if period:
        cmd += ["--period", str(period)]
    t0 = time.time()
    cpu_b0 = cpu_of(bt)
    r = subprocess.run(cmd, env=env(omp), capture_output=True, text=True, timeout=1200)
    wall = time.time() - t0
    burn_cpu = None
    if bp:
        burn_cpu = cpu_of(bt) - cpu_b0
        for pr in bt:
            try:
                pr.terminate()
            except Exception:
                pass
        psutil.wait_procs(bt, timeout=20)
    if out.exists():
        d = json.loads(out.read_text())
        ours = d["cpu_per_set_tree_s"] * d["n_sets"] / d["wall_s"] + d.get("own_cpu_s", 0.0) / d["wall_s"]
        burn_cores = (burn_cpu or 0.0) / max(wall, 1e-9) if burn else 0.0
        sysb = d.get("sys_busy_cores_in_affinity")
        d["arm"] = dict(name=name, W=w, omp=omp, burner=burn, burner_cpu_s=burn_cpu, burner_cores=round(burn_cores, 3), driver_wall_s=round(wall, 1), affinity=AFF, chunks=chunks,
                        bg_cores_in_affinity=None if sysb is None else round(sysb - ours - burn_cores, 3))
        out.write_text(json.dumps(d, indent=1))
    print(name, r.returncode, (r.stdout.strip().splitlines() or [""])[-1][:220], "bg=", json.loads(out.read_text())["arm"]["bg_cores_in_affinity"] if out.exists() else None, flush=True)
    if r.returncode != 0:
        print(r.stderr[-600:], flush=True)


PODP = 0.890625 / 2.1       # the trainer's micro-batch clock scaled to a pod core 2.1x slower than this box's (the ESTIMATED ratio of P6_ON_11VCPU section 3)
ARMS = (
    [(f"burn_thru_w{w}_omp{o}", dict(w=w, omp=o, burn=True)) for w in (7, 8) for o in (4, 1)]
    + [(f"lat_w7_omp{o}", dict(w=7, omp=o, mode="latency", n=96)) for o in (4, 1)]
    + [(f"burn_paced_pod_w{w}_c{c}", dict(w=w, omp=4, burn=True, mode="paced", period=PODP, nmicro=100, chunks=c)) for w, c in ((7, 1), (7, 2), (8, 1), (8, 2))]
    + [("burn_paced_dev_w7", dict(w=7, omp=4, burn=True, mode="paced", nmicro=40))]
    + [(f"rep2_thru_w{w}_omp{o}", dict(w=w, omp=o)) for w, o in ((7, 4), (7, 1), (8, 4), (8, 1))]
    + [("burn_thru_w7_c2", dict(w=7, omp=4, burn=True, chunks=2)), ("burn_thru_w9_omp4", dict(w=9, omp=4, burn=True)), ("burn_thru_w10_omp4", dict(w=10, omp=4, burn=True))]
    # the break-even sweep: the same paced rig at the micro-batch clock of a pod r times slower than this box (r = 1.3 / 1.6 / 1.9; r = 1.0 and 2.1 are above)
    + [(f"burn_paced_r{r}_w7_c{c}", dict(w=7, omp=4, burn=True, mode="paced", period=0.890625 / r, nmicro=100, chunks=c)) for r in (1.3, 1.6, 1.9) for c in (1, 2)]
)
for nm, kw in ARMS:
    arm(nm, **kw)
print("ZZEMU_DONE", flush=True)
