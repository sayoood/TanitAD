#!/usr/bin/env python3
"""STAGE C of the free-point navtest read (PLAN.md): ONE trunk pass per navtest token with the u400 model, and the scorer logits of THREE scorer states on the SAME
(trajectories, visual context).  Runs on the pod A40 (or on any device: the dev-box CPU smoke uses the same file).

The model call is the planner's own (`REFePlanner.infer`): fp32, batch 1, no autocast, `model(img, ego_vec, goal, calib=calib)` with the image built exactly like
`REFePlanner._image_for` (`cv2.imdecode` of the zip member, `cv2.resize((img_w, img_h))`, BGR->RGB, float32 / 255, cameras in the checkpoint's order).  Two forward PRE-hooks
read what the scoring decoder reads: `score_q_mlp`'s input (the detached, flattened proposals) and `score_dec[0]`'s second argument (the detached visual context `sctx`).
The other scorers are `q_mlp -> blocks(sctx) -> head` with their own weights on those SAME two tensors -- the three lines of `REFe.score_trajectories`.

    python fp_forward.py run   --inputs inputs_full.json --calib calib.npz --ckpt <u400> --state C0A=<c0 u1000>:A --state R0B=<r0 u1000>:B
                               --frames-dir D --logs L1,L2 | --logs-file F  --out out.npz  [--limit-tokens N] [--device cuda|cpu] [--mutate-swap-cams]
    python fp_forward.py serve --work-dir /workspace/ops/freept  ...      (the pod worker: gated on the re-roll's started_epoch; consumes in/chunk_K, writes out/chunk_K.npz)

Outputs per run (npz): token, log, traj[n,64,20,3] f32, logits_BASE / logits_<NAME> [n,64,7] f32, ic2_maxabs[n] (replicated BASE path vs the model's own score: must be 0.0),
meta_json (ckpt md5, state sha256s, update counters, device, torch version, timing).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
import zipfile

import numpy as np

SCORER = ("score_q_mlp", "score_dec", "score_head")
STARTED_EPOCH = "/workspace/ops/bank3c/full/started_epoch"
MIN_FREE_GB = 6.0
EXPECT_UPDATE = 1000


def sha256_tensors(sd: dict) -> str:
    h = hashlib.sha256()
    for m in SCORER:
        for k in sorted(sd[m]):
            t = sd[m][k].detach().cpu().contiguous()
            h.update(k.encode())
            h.update(str(tuple(t.shape)).encode())
            h.update(t.numpy().tobytes())
    return h.hexdigest()


def file_md5(p: str) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


class Taps:
    """what the scoring decoder reads, captured from the model's own forward"""

    def __init__(self, model):
        self.xflat = self.sctx = None
        self.h1 = model.score_q_mlp.register_forward_pre_hook(lambda m, a: setattr(self, "xflat", a[0]))
        self.h2 = model.score_dec[0].register_forward_pre_hook(lambda m, a: setattr(self, "sctx", a[1]))


def score_with(mods: dict, xflat, sctx):
    """REFe.score_trajectories, line for line, on a given set of scorer modules"""
    s = mods["score_q_mlp"](xflat)
    for blk in mods["score_dec"]:
        s = blk(s, sctx)
    return mods["score_head"](s)


def load_model(ckpt: str, device: str, backbone: str = "vitl16"):
    import torch
    import ckpt_io
    from model import REFe, REFeConfig
    cfg = ckpt_io.config_for_checkpoint(REFeConfig.for_backbone(backbone), ckpt)
    model = REFe(cfg).to(device).eval()
    meta: dict = {}
    fmt = ckpt_io.load_for_inference(model, ckpt, map_location=device, backbone=backbone, meta_out=meta)
    if not (cfg.n_score_components == 7 and cfg.nav_mode == "goal" and bool(meta.get("per_sample_calib", False))):
        raise SystemExit(f"REFUSED: checkpoint is n_score_components={cfg.n_score_components} nav_mode={cfg.nav_mode} per_sample_calib={meta.get('per_sample_calib')}")
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg, meta, fmt


def load_states(model, specs: list, device: str):
    """-> {name: {module: nn.Module}}, {name: {sha256, update, key, file_md5}}; strict loads, and the base modules' sha256 for reference"""
    import torch
    out, info = {}, {}
    for spec in specs:
        name, _, rest = spec.partition("=")
        path, _, key = rest.rpartition(":")
        ss = torch.load(path, map_location="cpu", weights_only=False)
        if int(ss.get("update", -1)) != EXPECT_UPDATE:
            raise SystemExit(f"REFUSED: {path} is update {ss.get('update')}, expected {EXPECT_UPDATE}")
        if key not in ("A", "B"):
            raise SystemExit(f"REFUSED: state key {key!r}")
        mods = {}
        for m in SCORER:
            mm = copy.deepcopy(getattr(model, m))
            mm.load_state_dict(ss[key][m], strict=True)
            mods[m] = mm.to(device).eval()
        out[name] = mods
        info[name] = {"file": os.path.basename(path), "key": key, "update": int(ss["update"]), "sha256_tensors": sha256_tensors(ss[key]),
                      "file_md5": file_md5(path)}
    base_sd = {m: getattr(model, m).state_dict() for m in SCORER}
    info["BASE"] = {"sha256_tensors": sha256_tensors(base_sd)}
    hs = [v["sha256_tensors"] for v in info.values()]
    if len(set(hs)) != len(hs):
        raise SystemExit(f"REFUSED: two of the scorer states are tensor-identical: {json.dumps({k: v['sha256_tensors'][:12] for k, v in info.items()})}")
    return out, info


class ZipCache:
    def __init__(self, root: str):
        self.root, self.z = root, {}

    def read(self, ref: str) -> bytes:
        zn, member = ref.split("::", 1)
        if zn not in self.z:
            self.z[zn] = zipfile.ZipFile(os.path.join(self.root, zn))
        return self.z[zn].read(member)

    def close(self):
        for z in self.z.values():
            z.close()
        self.z = {}


def build_img(refs: list, zc: ZipCache, cfg):
    """REFePlanner._image_for, line for line, with the bytes from the shipped zips"""
    import cv2
    import torch
    frames = []
    for r in refs:
        buf = np.frombuffer(zc.read(r), dtype=np.uint8)
        im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if im is None:
            raise RuntimeError(f"undecodable frame {r}")
        im = cv2.resize(im, (cfg.img_w, cfg.img_h))[:, :, ::-1]
        frames.append(np.ascontiguousarray(im.astype(np.float32).transpose(2, 0, 1)) / 255.0)
    if len(frames) != cfg.n_cameras:
        raise RuntimeError(f"{len(frames)} frames, the model wants {cfg.n_cameras}")
    return torch.from_numpy(np.stack(frames, axis=0))[None]


def run_tokens(a, model, cfg, states, tokens: list, inputs: dict, calib, zc: ZipCache, log=print):
    import torch
    dev = a.device
    taps = Taps(model)
    n = len(tokens)
    M, C = cfg.n_proposals, cfg.n_score_components
    names = ["BASE"] + list(states)
    traj_o = np.zeros((n, M, 20, 3), np.float32)
    logits_o = {nm: np.zeros((n, M, C), np.float32) for nm in names}
    ic2 = np.zeros(n, np.float64)
    t0 = time.time()
    for i, t in enumerate(tokens):
        r = inputs[t]
        img = build_img(r["frames"], zc, cfg)
        if a.mutate_swap_cams and i < 8:                                  # D3, a DELIBERATE REGRESSION: exchange camera slots 1 and 2
            img = img.clone()
            img[:, [1, 2]] = img[:, [2, 1]]
        ego = torch.tensor([r["ego"]], dtype=torch.float32, device=dev)
        goal = torch.tensor([r["goal"]], dtype=torch.float32).to(dev)
        cal = torch.tensor(calib[r["log"]], dtype=torch.float64)           # the planner passes the CPU float64 [1, n_cam, 16] rig
        taps.xflat = taps.sctx = None
        with torch.no_grad():
            traj, score = model(img.to(dev), ego, goal, calib=cal)
            if taps.xflat is None or taps.sctx is None:
                raise RuntimeError("the scoring decoder never ran: the taps saw nothing")
            base_mods = {m: getattr(model, m) for m in SCORER}
            rep = score_with(base_mods, taps.xflat, taps.sctx)
            ic2[i] = float((rep - score).abs().max())                      # IC2: must be EXACTLY 0.0
            outs = {"BASE": score}
            for nm, mods in states.items():
                outs[nm] = score_with(mods, taps.xflat, taps.sctx)
        traj_o[i] = traj[0].float().cpu().numpy()
        for nm in names:
            logits_o[nm][i] = outs[nm][0].float().cpu().numpy()
        if (i + 1) % 25 == 0 or i + 1 == n:
            log(f"  {i + 1}/{n} tokens  {(time.time() - t0) / (i + 1):.2f} s/token  ic2 max so far {ic2[:i + 1].max():.3g}", flush=True)
    taps.h1.remove()
    taps.h2.remove()
    return traj_o, logits_o, ic2, time.time() - t0


def repeat_check(a, model, cfg, states, tokens, inputs, calib, zc, traj, logits, n):
    """IC5 (forward arm): the first n tokens again, in the same process -> max |difference| per output; must be EXACTLY 0.0 on every output"""
    n = min(n, len(tokens))
    t2, l2, _ic2, _s = run_tokens(a, model, cfg, states, tokens[:n], inputs, calib, zc, log=lambda *x, **k: None)
    d = {"n": n, "traj_max_abs": float(np.abs(t2 - traj[:n]).max())}
    for nm in l2:
        d[f"logits_{nm}_max_abs"] = float(np.abs(l2[nm] - logits[nm][:n]).max())
    d["bit_identical"] = all(v == 0.0 for k, v in d.items() if k != "n")
    return d


def setup(a):
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    inputs = json.load(open(a.inputs, encoding="utf-8"))
    calib = dict(np.load(a.calib))
    model, cfg, meta, fmt = load_model(a.ckpt, a.device)
    states, sinfo = load_states(model, a.state, a.device)
    ck_md5 = file_md5(a.ckpt)
    if a.ckpt_md5_prefix and not ck_md5.startswith(a.ckpt_md5_prefix):
        raise SystemExit(f"REFUSED: checkpoint md5 {ck_md5} does not start with {a.ckpt_md5_prefix}")
    info = {"ckpt": os.path.basename(a.ckpt), "ckpt_md5": ck_md5, "ckpt_format": fmt, "states": sinfo, "device": a.device,
            "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": (torch.cuda.get_device_name(0) if a.device.startswith("cuda") else None),
            "ckpt_meta_per_sample_calib": bool(meta.get("per_sample_calib", False)), "mutate_swap_cams": bool(a.mutate_swap_cams)}
    return inputs, calib, model, cfg, states, info


def write_out(path, tokens, inputs, traj, logits, ic2, info, secs):
    info = dict(info, seconds=round(secs, 1), n=len(tokens))
    np.savez(path, token=np.array(tokens), log=np.array([inputs[t]["log"] for t in tokens]), traj=traj,
             ic2_maxabs=ic2, meta_json=np.array(json.dumps(info)), **{f"logits_{k}": v for k, v in logits.items()})


def cmd_run(a):
    inputs, calib, model, cfg, states, info = setup(a)
    logs = []
    if a.logs:
        logs = a.logs.split(",")
    elif a.logs_file:
        logs = [l.strip() for l in open(a.logs_file) if l.strip()]
    toks = [t for t, r in inputs.items() if r["log"] in set(logs)] if logs else list(inputs)
    toks = sorted(toks, key=lambda t: (inputs[t]["log"], t))
    if a.limit_tokens:
        toks = toks[: a.limit_tokens]
    zc = ZipCache(a.frames_dir)
    traj, logits, ic2, secs = run_tokens(a, model, model.cfg, states, toks, inputs, calib, zc)
    if a.repeat_first:
        info = dict(info, ic5_forward=repeat_check(a, model, model.cfg, states, toks, inputs, calib, zc, traj, logits, a.repeat_first))
        print("IC5 forward repeat:", json.dumps(info["ic5_forward"]))
    zc.close()
    write_out(a.out, toks, inputs, traj, logits, ic2, info, secs)
    print(f"ZZFWD_OK {len(toks)} ic2_max {ic2.max():.3g} {secs:.0f}s -> {a.out}")
    return 0 if ic2.max() == 0.0 else 2


def free_gb(path: str) -> float:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9


def cmd_serve(a):
    """the pod worker.  GATED on the P6 re-roll having launched (STARTED_EPOCH), refuses a chunk when the volume is short on space, and stops by its pid (no pgrep/pkill)."""
    if not a.no_gate and not os.path.exists(STARTED_EPOCH):
        print(f"GATED: {STARTED_EPOCH} does not exist -- the P6 re-roll has not launched; no GPU job may start (exit 3)")
        return 3
    wd = a.work_dir
    os.makedirs(f"{wd}/out", exist_ok=True)
    open(f"{wd}/worker.pid", "w").write(str(os.getpid()))
    inputs, calib, model, cfg, states, info = setup(a)
    print(f"worker up: pid {os.getpid()} {info['gpu']} states {list(states)}", flush=True)
    idle = 0
    while True:
        ready = sorted(f[:-6] for f in os.listdir(f"{wd}/in") if f.startswith("chunk_") and f.endswith(".ready")) if os.path.isdir(f"{wd}/in") else []
        todo = [c for c in ready if not os.path.exists(f"{wd}/out/{c}.done")]
        if not todo:
            if os.path.exists(f"{wd}/in/ALL_DONE"):
                print("ZZFWD_SERVE_DONE")
                return 0
            time.sleep(10)
            idle += 1
            continue
        c = todo[0]
        logs = [l.strip() for l in open(f"{wd}/in/{c}.logs") if l.strip()]
        toks = sorted([t for t, r in inputs.items() if r["log"] in set(logs)], key=lambda t: (inputs[t]["log"], t))
        zc = ZipCache(f"{wd}/in/{c}")
        traj, logits, ic2, secs = run_tokens(a, model, cfg, states, toks, inputs, calib, zc)
        extra = {}
        if not os.path.exists(f"{wd}/out/ic5_forward.json"):             # IC5: only the first chunk this worker ever processes
            extra["ic5_forward"] = repeat_check(a, model, cfg, states, toks, inputs, calib, zc, traj, logits, a.repeat_first or 128)
            json.dump(extra["ic5_forward"], open(f"{wd}/out/ic5_forward.json", "w"))
            print("IC5 forward repeat:", json.dumps(extra["ic5_forward"]), flush=True)
        zc.close()
        tmp = f"{wd}/out/{c}.npz.tmp.npz"
        write_out(tmp, toks, inputs, traj, logits, ic2, dict(info, chunk=c, logs=logs, **extra), secs)
        os.replace(tmp, f"{wd}/out/{c}.npz")
        h = hashlib.sha256(open(f"{wd}/out/{c}.npz", "rb").read()).hexdigest()
        open(f"{wd}/out/{c}.sha256", "w").write(h)
        import shutil
        shutil.rmtree(f"{wd}/in/{c}", ignore_errors=True)               # the frames of a processed chunk go at once (the volume is shared and 97 % full)
        open(f"{wd}/out/{c}.done", "w").write(h)
        print(f"ZZFWD_CHUNK_DONE {c} {len(toks)} tokens ic2_max {ic2.max():.3g} {secs:.0f}s", flush=True)
        if ic2.max() != 0.0:
            print("ZZFWD_IC2_FAILED -- stopping (the replicated scorer path does not equal the model's own)")
            return 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "serve"))
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ckpt-md5-prefix", default="8e382eeb")
    ap.add_argument("--state", action="append", default=[], help="NAME=path:KEY (KEY A or B)")
    ap.add_argument("--frames-dir", default=None)
    ap.add_argument("--logs", default=None)
    ap.add_argument("--logs-file", default=None)
    ap.add_argument("--limit-tokens", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--work-dir", default="/workspace/ops/freept")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--repeat-first", type=int, default=0, help="IC5: re-run the first N tokens in-process and compare bit for bit")
    ap.add_argument("--mutate-swap-cams", action="store_true")
    ap.add_argument("--no-gate", action="store_true", help="dev-box smoke only: skip the re-roll gate")
    a = ap.parse_args()
    return {"run": cmd_run, "serve": cmd_serve}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
