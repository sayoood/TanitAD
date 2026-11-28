"""One SAM3 instance of the multi-instance throughput probe (PI 2026-09-14: "check if there is the possibility to parallelize the
inference and load different instances of sam3 into the gpu"). Builds its own model with the approved per-frame flags (env, as
sam3map_front_fast.py), warms up on 2 frames, writes <run>/ready_<id>, waits for <run>/start, then runs the complete per-frame work
(image encode, 19 prompts, classify CPU logic, stripe pass, re-lift, npz write to RAM disk) on native frames in a loop until
<run>/stop exists, logging a monotonic timestamp per COMPLETED frame (CPU post included).
Usage: mi_worker.py <run dir> <worker id> <c8>[,<c8>...]"""
import concurrent.futures as cf
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, "/home/nvidia/sam3map"); sys.path.insert(0, "/home/nvidia/sam3map/eval")
import torch
import sam3map_front_fast as FF

run, wid, clips = Path(sys.argv[1]), sys.argv[2], sys.argv[3].split(",")
out = Path(f"/dev/shm/mi_{run.name}_{wid}"); out.mkdir(parents=True, exist_ok=True)
t0 = time.time()
proc, _ = FF.E.S.build(conf=0.25)
fast = FF.FastSAM3(proc, FF.FLAGS)
frames = []
for c8 in clips:
    sd = FF.ROOT / f"seq_{c8}"; toks = sorted(p.name for p in sd.iterdir() if p.is_dir())
    poses = json.loads((sd / "poses.json").read_text())
    frames += [(c8, j, toks, sd, poses) for j in range(len(toks))]
pool = cf.ThreadPoolExecutor(max_workers=max(1, int(FF.FLAGS["ASYNC"])))
done_ts = []


def one(k, timed):
    c8, j, toks, sd, poses = frames[k % len(frames)]
    fi = FF.frame_inputs(c8, j, toks, sd, poses)
    st = fast.encode(fi["img"])
    inst = fast.instances(st, FF.CLS_PT + FF.STRIPE_PT)

    def cpu():
        cls, s_, evid = FF.CLASSIFY_INJECTED(FF._Shim(inst), fi["img"], (fi["camm"], fi["sgrid"]), True)
        FF.post(fi, cls, s_, evid, inst["crosswalk stripe"], inst["white stripe on road"], out, k % 1000, "mi")
        if timed:
            done_ts.append(time.monotonic())
    return pool.submit(cpu)


for k in range(2):                                                        # warm-up, untimed
    one(k, False).result()
(run / f"ready_{wid}").write_text(f"{time.time() - t0:.1f}")
while not (run / "start").exists():
    time.sleep(0.05)
k = 2; futs = []
while not (run / "stop").exists():
    futs.append(one(k, True)); k += 1
    futs = [f for f in futs if not f.done()]
for f in futs:
    f.result()
pool.shutdown()
(run / f"result_{wid}.json").write_text(json.dumps({"worker": wid, "ts": done_ts, "max_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
                                                    "build_and_warm_s": float((run / f"ready_{wid}").read_text())}), encoding="utf-8")
print(f"ZZMIWORKER-{wid}-{len(done_ts)}ZZ", flush=True)
