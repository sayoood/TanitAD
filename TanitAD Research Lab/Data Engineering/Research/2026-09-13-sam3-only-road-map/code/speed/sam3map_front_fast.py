"""Front-camera SAM3 map extraction in ONE pass per frame, written as the <c8>_<tag>raw frame files that refine / consensus /
renderer / compose read unchanged: the v6 extractor's classify() on CAM_FW, the v6s stripe pass (sam3map_xwalk_stripes.py) and
the front-only re-lift (make_front_only.py). Keys: tok, T_world_rig, stats, ver, cls_CAM_FW, evid_CAM_FW, xstripe_CAM_FW,
pts_1..7, rng_1..7.

MODE=ref   the pipeline as it runs today: E.classify with the stock processor (set_image + set_text_prompt per prompt), then the
           stripe prompts through E.instances after a second set_image. Every other mode is compared against this one.
MODE=fast  speed measures, each switchable for attribution. classify() runs as the SAME code object (E.classify.__code__) with
           the SAM3 results injected, so its CPU logic cannot drift from the extractor's:
  TEXT_CACHE=1     text features of every prompt computed once per process
  ONE_ENCODE=1     the stripe prompts reuse the classify image features (no second image encode)
  GPU_SELECT=1     the prompt threshold is applied before mask upsampling and the host copy -- only accepted masks are upsampled to
                   full resolution and copied; the score comparison is the same numpy-float32-vs-Python-float test E.instances does
  HALF=none|fp16enc|fp16all|bf16all   autocast of grounding stages; stage outputs cast back to float32 at the stage boundary
  BATCH=<n>        prompts per grounding forward (needs TEXT_CACHE=1)
  BB_HALF=none|fp16|bf16   autocast of the image backbone, features cast back to float32
  ASYNC=<n>        CPU post-processing (classify logic, stripes, re-lift, write) in n worker threads while the GPU runs the next frame
  DROP_PROMPTS=a,b deliberate-regression arm only: these prompts return no instances
Usage: sam3map_front_fast.py <tag> <c8> [<c8> ...]   (SAM3MAP_ROOT native root; writes /home/nvidia/sam3map/<c8>_<tag>raw)"""
import concurrent.futures as cf
import contextlib, dataclasses, json, os, sys, threading, time, types
from pathlib import Path
sys.path.insert(0, "/home/nvidia/sam3map"); sys.path.insert(0, "/home/nvidia/sam3map/eval")
import numpy as np
import cv2
import torch
from PIL import Image
import sam3map_extract_v6 as E
from sam3map_consensus import relift
from sam3.model.data_misc import interpolate

ROOT = Path(os.environ.get("SAM3MAP_ROOT", "/home/nvidia/sam3map/native7"))
SM = Path("/home/nvidia/sam3map")
MODE = os.environ.get("MODE", "fast")
FLAGS = {k: os.environ.get(k, d) for k, d in (("TEXT_CACHE", "1"), ("ONE_ENCODE", "1"), ("GPU_SELECT", "1"), ("HALF", "none"), ("BATCH", "1"),
                                               ("BB_HALF", "none"), ("ASYNC", "0"), ("DROP_PROMPTS", ""))}
STRIPE_THR = 0.4                                                              # sam3map_xwalk_stripes.THR
CLS_PT = [(p, t) for fam, table in E.PROMPTS.items() for p, t in table.items()]
STRIPE_PT = [("crosswalk stripe", STRIPE_THR), ("white stripe on road", STRIPE_THR)]
DROP = set(x for x in FLAGS["DROP_PROMPTS"].split(",") if x)
DT = {"fp16": torch.float16, "bf16": torch.bfloat16}


def sync():
    torch.cuda.synchronize()
    return time.perf_counter()


def to32(x):
    if torch.is_tensor(x):
        return x.float() if x.is_floating_point() else x
    if isinstance(x, dict):
        return {k: to32(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to32(v) for v in x]
    if isinstance(x, tuple):
        return tuple(to32(v) for v in x)
    return x


class FastSAM3:
    STAGES = {"prompt": "_encode_prompt", "enc": "_run_encoder", "dec": "_run_decoder", "seg": "_run_segmentation_heads"}

    def __init__(self, proc, flags):
        self.proc, self.m, self.f = proc, proc.model, flags
        self.batch = int(flags["BATCH"])
        assert self.batch == 1 or flags["TEXT_CACHE"] == "1", "BATCH > 1 needs TEXT_CACHE=1"
        self.txt = None
        if flags["TEXT_CACHE"] == "1":
            with torch.inference_mode():
                self.txt = {p: self.m.backbone.forward_text([p], device="cuda") for p, _ in CLS_PT + STRIPE_PT}
        half = {"none": (None, ()), "fp16enc": ("fp16", ("enc",)), "fp16all": ("fp16", tuple(self.STAGES)), "bf16all": ("bf16", tuple(self.STAGES))}[flags["HALF"]]
        for st in half[1]:                                                    # wrap only the half-precision stages
            name = self.STAGES[st]; fn = getattr(self.m, name); dt = DT[half[0]]

            def w(*a, _fn=fn, _dt=dt, **k):
                with torch.autocast("cuda", dtype=_dt):
                    r = _fn(*a, **k)
                return to32(r)
            setattr(self.m, name, w)

    @torch.inference_mode()
    def encode(self, img):
        bb = self.f["BB_HALF"]
        with (torch.autocast("cuda", dtype=DT[bb]) if bb != "none" else contextlib.nullcontext()):
            state = self.proc.set_image(img)
        if bb != "none":
            state["backbone_out"] = to32(state["backbone_out"])
        return state

    @torch.inference_mode()
    def _forward(self, state, chunk):
        m, n = self.m, len(chunk)
        if self.txt is None:
            text = m.backbone.forward_text([chunk[0][0]], device="cuda")
        elif n == 1:
            text = self.txt[chunk[0][0]]
        else:
            text = {k: torch.cat([self.txt[p][k] for p, _ in chunk], 0 if k == "language_mask" else 1) for k in self.txt[chunk[0][0]]}
        b = dict(state["backbone_out"]); b.update(text)
        fs = self.proc.find_stage if n == 1 else dataclasses.replace(self.proc.find_stage, img_ids=torch.zeros(n, dtype=torch.long, device="cuda"),
                                                                    text_ids=torch.arange(n, dtype=torch.long, device="cuda"))
        return m.forward_grounding(backbone_out=b, find_input=fs, geometric_prompt=m._get_dummy_prompt(n), find_target=None)

    @torch.inference_mode()
    def instances(self, state, pts):
        """{prompt: [(score, bool mask HxW)]} with E.instances' exact semantics"""
        H, W = state["original_height"], state["original_width"]
        res = {p: [] for p, _ in pts}
        todo = [(p, t) for p, t in pts if p not in DROP]
        for s0 in range(0, len(todo), self.batch):
            chunk = todo[s0:s0 + self.batch]
            out = self._forward(state, chunk)
            for i, (p, thr) in enumerate(chunk):
                probs = (out["pred_logits"][i:i + 1].sigmoid() * out["presence_logit_dec"][i:i + 1].sigmoid().unsqueeze(1)).squeeze(-1)
                keep = probs > self.proc.confidence_threshold
                sc = probs[keep].float().cpu().numpy().reshape(-1)
                if not len(sc):
                    continue
                masks = out["pred_masks"][i:i + 1][keep]
                if self.f["GPU_SELECT"] == "1":
                    sel = [k for k, s in enumerate(sc) if s >= thr]
                    if not sel:
                        continue
                    mb = interpolate(masks[sel].unsqueeze(1), (H, W), mode="bilinear", align_corners=False).sigmoid() > 0.5
                    nz = mb.flatten(1).any(1).cpu().numpy()
                    idx = [j for j, z in enumerate(nz) if z]
                    if not idx:
                        continue
                    ms = mb[idx].cpu().numpy().reshape(len(idx), H, W)
                    res[p] = [(float(sc[sel[j]]), ms[q]) for q, j in enumerate(idx)]
                else:
                    ms = (interpolate(masks.unsqueeze(1), (H, W), mode="bilinear", align_corners=False).sigmoid() > 0.5).cpu().numpy()
                    ms = ms.reshape(len(sc), H, W)
                    res[p] = [(float(s), ms[k].astype(bool)) for k, s in enumerate(sc) if s >= thr and ms[k].any()]
        return res


def _lookup(proc, state, prompt, thr):
    ok = dict(CLS_PT)
    assert abs(ok[prompt] - thr) < 1e-12, (prompt, thr)
    return state["inst"][prompt]


class _Shim:
    def __init__(self, inst):
        self.inst = inst

    def set_image(self, img):
        return {"inst": self.inst}


CLASSIFY_INJECTED = types.FunctionType(E.classify.__code__, {**E.__dict__, "instances": _lookup}, "classify_injected")


def frame_inputs(c8, j, toks, sd, poses):
    tok = toks[j]; fd = sd / tok
    c = np.load(fd / "calib.npz"); fr = json.loads((fd / "frame.json").read_text())
    lp = fd / "lidar.npy"
    grid, fb = E.P.ground_grid(np.load(lp).astype(np.float64)) if lp.exists() else (np.full(2500, np.nan), 0.0)
    sgrid = E.GS.smooth_grid(grid, fb)
    img = Image.open(fd / "images" / "CAM_FW.jpg").convert("RGB")
    camm = E.CM.Camera.from_calib(c, fr["cam_order"].index("CAM_FW"), img.width, img.height)
    return {"tok": tok, "c": c, "fr": fr, "sgrid": sgrid, "img": img, "camm": camm, "T": np.array(poses[tok]["T_world_rig"])}


def post(fi, cls, st, evid, xs_list, ws_list, dst, j, tag):
    """extractor downscale -> stripe pass -> front-only re-lift -> write (CPU only)"""
    H, W = fi["img"].height, fi["img"].width
    small_e = cv2.resize(evid, (E.OUT_W, E.OUT_H), interpolation=cv2.INTER_NEAREST)
    small = cv2.resize(cls, (E.OUT_W, E.OUT_H), interpolation=cv2.INTER_NEAREST)
    xs = np.zeros((H, W), bool)
    for s, m in xs_list:
        xs |= m
    ws = np.zeros((H, W), bool)
    for s, m in ws_list:
        ws |= m
    area = cv2.dilate(np.repeat(np.repeat(((small_e & 2) > 0).astype(np.uint8), 2, axis=0), 2, axis=1)[:H, :W], np.ones((31, 31), np.uint8)) > 0
    stripes = (xs | ws) & area
    xsmall = stripes[::2, ::2]
    cl = small.copy(); cl[cl == 3] = 1
    put = xsmall & np.isin(cl, (1, 2, 6)); cl[put] = 3
    p_, r_ = relift(cl, E.CM.Camera.from_calib(fi["c"], fi["fr"]["cam_order"].index("CAM_FW")), fi["sgrid"], fi["T"])
    out = {"tok": fi["tok"], "T_world_rig": fi["T"], "ver": "v6f", "stats": json.dumps({"CAM_FW": st, "_front_only_from": f"front_fast:{tag}"}),
           "cls_CAM_FW": cl, "evid_CAM_FW": small_e, "xstripe_CAM_FW": xsmall}
    for k in range(1, 8):
        out[f"pts_{k}"] = p_.get(k, np.zeros((0, 2), np.float32)); out[f"rng_{k}"] = r_.get(k, np.zeros((0,), np.float16))
    np.savez_compressed(dst / f"{j:03d}.npz", **out)


def main():
    tag, clips = sys.argv[1], sys.argv[2:]
    t0 = time.time()
    proc, _ = E.S.build(conf=0.25)
    t_build = time.time() - t0
    fast = FastSAM3(proc, FLAGS) if MODE == "fast" else None
    t_ready = time.time() - t0
    print(f"mode {MODE} flags {FLAGS} build {t_build:.1f}s ready {t_ready:.1f}s", flush=True)
    pool = cf.ThreadPoolExecutor(max_workers=int(FLAGS["ASYNC"])) if int(FLAGS["ASYNC"]) > 0 else None
    rep = {"mode": MODE, "flags": FLAGS, "build_s": t_build, "ready_s": t_ready, "clips": {}}
    for c8 in clips:
        sd = ROOT / f"seq_{c8}"; toks = sorted(p.name for p in sd.iterdir() if p.is_dir())
        if int(os.environ.get("FRAMES", "0")) > 0:                             # smoke tests only
            toks = toks[: int(os.environ["FRAMES"])]
        poses = json.loads((sd / "poses.json").read_text())
        dst = SM / f"{c8}_{tag}raw"; dst.mkdir(parents=True, exist_ok=True)
        rows, futs = [], []
        tc = time.time()
        loader = cf.ThreadPoolExecutor(max_workers=1)
        nxt = loader.submit(frame_inputs, c8, 0, toks, sd, poses)
        for j in range(len(toks)):
            ta = time.perf_counter()
            fi = nxt.result()
            if j + 1 < len(toks):
                nxt = loader.submit(frame_inputs, c8, j + 1, toks, sd, poses)
            tb = sync()
            if MODE == "ref":
                cls, st, evid = E.classify(proc, fi["img"], (fi["camm"], fi["sgrid"]), True)
                state = proc.set_image(fi["img"])
                xs_list = E.instances(proc, state, "crosswalk stripe", STRIPE_THR)
                ws_list = E.instances(proc, state, "white stripe on road", STRIPE_THR)
                tg = sync()
                post(fi, cls, st, evid, xs_list, ws_list, dst, j, tag)
                tp = time.perf_counter()
            else:
                state = fast.encode(fi["img"])
                if FLAGS["ONE_ENCODE"] == "1":
                    inst = fast.instances(state, CLS_PT + STRIPE_PT)
                else:
                    inst = fast.instances(state, CLS_PT)
                    inst.update(fast.instances(fast.encode(fi["img"]), STRIPE_PT))
                tg = sync()

                def cpu(fi=fi, inst=inst, j=j):
                    cls, st, evid = CLASSIFY_INJECTED(_Shim(inst), fi["img"], (fi["camm"], fi["sgrid"]), True)
                    post(fi, cls, st, evid, inst["crosswalk stripe"], inst["white stripe on road"], dst, j, tag)
                if pool is None:
                    cpu(); tp = time.perf_counter()
                else:
                    futs.append(pool.submit(cpu)); tp = time.perf_counter()
            rows.append({"j": j, "load_wait_s": tb - ta, "sam3_s": tg - tb, "cpu_s": tp - tg, "wall_s": tp - ta})
            if j % 24 == 0:
                print(f"  {c8} j={j:3d} {time.time() - tc:6.1f}s  sam3 {rows[-1]['sam3_s']:.3f}s cpu {rows[-1]['cpu_s']:.3f}s", flush=True)
        for f in futs:
            f.result()
        loader.shutdown()
        wall = time.time() - tc
        n_out = len(list(dst.glob("[0-9][0-9][0-9].npz")))
        rep["clips"][c8] = {"frames": len(toks), "written": n_out, "wall_s": wall, "s_per_frame": wall / len(toks),
                            "sam3_s_mean": float(np.mean([r["sam3_s"] for r in rows[1:]])), "cpu_s_mean": float(np.mean([r["cpu_s"] for r in rows[1:]])),
                            "load_wait_s_mean": float(np.mean([r["load_wait_s"] for r in rows[1:]])), "rows": rows}
        print(f"clip {c8}: {n_out}/{len(toks)} frames, {wall:.1f}s = {wall / len(toks):.3f} s/frame", flush=True)
    if pool:
        pool.shutdown()
    rep["max_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
    (SM / f"front_fast_{tag}.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"ZZFRONTFAST-{tag}-DONEZZ")


if __name__ == "__main__":
    main()
