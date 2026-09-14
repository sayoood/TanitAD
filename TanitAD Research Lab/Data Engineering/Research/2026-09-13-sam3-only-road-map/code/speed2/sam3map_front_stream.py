"""Streaming front-camera map production on Thor with the pipeline-level measures on top of the per-frame ones:
  ONE BUILD      SAM3 is built once per process; refine's ego-mask SAM3 calls (8 frames per clip) run on the same loaded model in
                 fp32 (the fast path's half-precision wrappers are suspended for them), instead of a second model build per clip
  BACKGROUND CPU refine (after its ego masks), consensus, renderer v5m fields and compose run UNMODIFIED as background processes
                 while the GPU extracts the next clip; at most one clip's CPU stages run at a time, in clip order
Per-frame work is sam3map_front_fast.py's (flags from the environment, same meaning). Output names follow the approval chain:
<c8>_<tag>raw, <c8>_<tag>, <c8>_<tag>c, render5_<c8>_<tag>m, render5_<c8>_<tag>r.
Usage: sam3map_front_stream.py <tag> <c8> [<c8> ...]"""
import concurrent.futures as cf
import json, os, subprocess, sys, time
from pathlib import Path
os.environ["SAM3MAP_VIEWS"] = os.environ.get("SAM3MAP_CAM", "CAM_FW")         # refine's module-level camera list: front only
sys.path.insert(0, "/home/nvidia/sam3map"); sys.path.insert(0, "/home/nvidia/sam3map/eval"); sys.path.insert(0, "/home/nvidia/sam3paint")
import numpy as np
import torch
import sam3map_front_fast as FF
import sam3_smoke
import sam3map_refine_v6 as R

SM = Path("/home/nvidia/sam3map"); EV = SM / "eval"
PY = "/home/nvidia/venvs/tanitad-edge/bin/python"
PP = "/home/nvidia/sam3vendor:/home/nvidia/sam3paint:/home/nvidia/sam3map"


def suspended(fast):
    """context: the model's class methods and a plain fp32 processor, for code that must see today's SAM3 path"""
    class _Ctx:
        def __enter__(self):
            self.saved = {n: fast.m.__dict__.pop(n) for n in list(FastNames) if n in fast.m.__dict__}
        def __exit__(self, *a):
            for n, fn in self.saved.items():
                setattr(fast.m, n, fn)
    return _Ctx()


FastNames = tuple(FF.FastSAM3.STAGES.values())


def cpu_chain(tag, c8, ego_path):
    env = dict(os.environ, SAM3MAP_ROOT=str(FF.ROOT), SAM3MAP_VIEWS=FF.CAM, SAM3MAP_TAG=tag, PYTHONPATH=PP, HF_HUB_OFFLINE="1", OMP_NUM_THREADS="4")
    t = {}
    steps = [
        ("refine", [PY, str(EV / "refine_with_ego.py"), c8, str(ego_path)], SM, f"refine_{tag}_{c8}.log"),
        ("consensus", [PY, "sam3map_consensus.py", c8, str(SM / f"{c8}_{tag}"), str(SM / f"{c8}_{tag}c")], EV, f"consensus_{tag}_{c8}.log"),
        ("render", [PY, "sam3map_render_v5m.py", c8, str(SM / f"{c8}_{tag}c"), str(SM / f"render5_{c8}_{tag}m")], EV, f"render5_{c8}_{tag}m.log"),
        ("compose", [PY, "compose.py", str(SM / f"render5_{c8}_{tag}m"), str(SM / f"render5_{c8}_{tag}r")], EV, f"compose_{tag}_{c8}.log"),
    ]
    extra = {"consensus": {"CONSENSUS_NO_PROMOTE": "1"},
             "render": {"SAM3MAP_COMPOSITE": "majority_v65", "SURFACE_MODE": "5", "PAINT_SHARE": "0.2", "PAINT_RULE": "p95", "CROSSWALK_STRIPES": "1", "NO_FRAMES": "1", "FIELDS": "1"},
             "compose": {"LINE_EDGE_GAP": "1", "LINE_MIN_LEN_M": "1.0", "WLK_ISLAND_M2": "3", "EDGE_OBS": "near", "XWALK": "dirclose", "XWALK_LEN": "11"}}
    for name, cmd, cwd, log in steps:
        a = time.time()
        with open(SM / log, "w") as fh:
            rc = subprocess.run(cmd, cwd=cwd, env=dict(env, **extra.get(name, {})), stdout=fh, stderr=subprocess.STDOUT).returncode
        t[name] = {"s": round(time.time() - a, 1), "rc": rc}
        if rc != 0:
            break
    return t


def main():
    tag, clips = sys.argv[1], sys.argv[2:]
    T0 = time.time()
    proc, fix = FF.E.S.build(conf=0.25)
    t_build = time.time() - T0
    fast = FF.FastSAM3(proc, FF.FLAGS)
    sam3_smoke.build = lambda conf=0.25: (proc, fix)                         # refine's ego_masks: the loaded model, no second build
    bg = cf.ThreadPoolExecutor(max_workers=1)
    post_pool = cf.ThreadPoolExecutor(max_workers=int(FF.FLAGS["ASYNC"])) if int(FF.FLAGS["ASYNC"]) > 0 else None
    rep = {"flags": FF.FLAGS, "build_s": t_build, "clips": {}}
    jobs = {}
    for c8 in clips:
        tc = time.time()
        sd = FF.ROOT / f"seq_{c8}"; toks = sorted(p.name for p in sd.iterdir() if p.is_dir())
        poses = json.loads((sd / "poses.json").read_text())
        dst = SM / f"{c8}_{tag}raw"; dst.mkdir(parents=True, exist_ok=True)
        loader = cf.ThreadPoolExecutor(max_workers=1); futs = []
        nxt = loader.submit(FF.frame_inputs, c8, 0, toks, sd, poses)
        for j in range(len(toks)):
            fi = nxt.result()
            if j + 1 < len(toks):
                nxt = loader.submit(FF.frame_inputs, c8, j + 1, toks, sd, poses)
            state = fast.encode(fi["img"])
            inst = fast.instances(state, FF.CLS_PT + FF.STRIPE_PT)

            def cpu(fi=fi, inst=inst, j=j):
                cls, st, evid = FF.CLASSIFY_INJECTED(FF._Shim(inst), fi["img"], (fi["camm"], fi["sgrid"]), True)
                FF.post(fi, cls, st, evid, inst["crosswalk stripe"], inst["white stripe on road"], dst, j, tag)
            if post_pool is None:
                cpu()
            else:
                futs.append(post_pool.submit(cpu))
        for f in futs:
            f.result()
        loader.shutdown()
        t_extract = time.time() - tc
        a = time.time()
        with suspended(fast), torch.inference_mode():
            ego, info = R.ego_masks(sd, toks)
        info = dict(info, _n_toks=len(toks))
        ego_path = SM / f"ego_{tag}_{c8}.npz"
        np.savez_compressed(ego_path, _info=json.dumps(info), **{k: v for k, v in ego.items() if k != "_spread_m"})
        t_ego = time.time() - a
        jobs[c8] = bg.submit(cpu_chain, tag, c8, ego_path)
        rep["clips"][c8] = {"frames": len(toks), "extract_s": round(t_extract, 1), "ego_s": round(t_ego, 1), "gpu_part_done_at_s": round(time.time() - T0, 1)}
        print(f"clip {c8}: extract {t_extract:.1f}s ({t_extract / len(toks):.3f} s/frame) ego {t_ego:.1f}s -> CPU stages queued", flush=True)
    for c8, job in jobs.items():
        rep["clips"][c8]["cpu_stages"] = job.result()
    bg.shutdown()
    if post_pool:
        post_pool.shutdown()
    rep["total_wall_s"] = round(time.time() - T0, 1)
    (SM / f"front_stream_{tag}.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k != "flags"}))
    print(f"ZZFRONTSTREAM-{tag}-DONEZZ")


if __name__ == "__main__":
    main()
