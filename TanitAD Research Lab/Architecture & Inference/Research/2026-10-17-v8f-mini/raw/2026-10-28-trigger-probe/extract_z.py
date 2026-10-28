#!/usr/bin/env python3
"""extract_z.py -- Z for the sampled TRAIN windows: h_O (the tensor head_o reads, v8f_mini.py run_o `h = self.read_o(...).flatten(1)`, [1024]) of the DAG-OACC s0 checkpoint,
plus the secondary tensors h_T (gelu(trunk_t), [256]), T's code and S's checkpoints.  One CPU forward_infer path per batch, a_t = a_s = 0, fp32.  Resumable and time-boxed.
CUDA_VISIBLE_DEVICES=-1  python -I extract_z.py [--limit N] [--budget-s 200]"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(4)
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
RAW, SCR = f"{HERE}/raw", f"{HERE}/scratch"
TREE = "E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a"
CKPT = "E:/Projects/TanitAD-artifacts/v8f-dagoacc/M2b_s0_dagoacc/ckpt_last.pt"
CKPT_SHA = "67265ec1666fae3a6d51e924ee06dfb68b2cd0630afc62e42f3a46182b8bc45f"
CACHE_ROOT = "F:/TanitAD-scratch/speedread/cache"
SHIP = "C:/Users/Admin/box2_ship/v9ship"
MANIFEST_TRAIN = "C:/Users/Admin/box2_ship/logs/v8f_dscale/manifest_train.jsonl"
TRACKS = "E:/Projects/TanitAD-artifacts/sfresh_inputs/inputs/native_tracks.pt"
V9_MD5 = "83e30bd3791f0767707eade5f4671759"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_model():
    sys.path[:0] = [f"{TREE}/stack", f"{TREE}/stack/scripts", f"{TREE}/taniteval"]
    from tanitad.models import v8f_mini as M
    got = sha256_file(CKPT)
    if got != CKPT_SHA:
        raise SystemExit(f"REFUSED: checkpoint sha256 {got} != {CKPT_SHA}")
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = M.V8FMiniConfig(**ck["cfg"])
    assert cfg.kin_mode == "yaw"
    model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    model.load_state_dict(ck["model"])
    model.eval()
    return M, model, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--budget-s", type=float, default=200.0)
    ap.add_argument("--bs", type=int, default=32)
    a = ap.parse_args()
    t0 = time.time()
    M, model, cfg = load_model()
    from tanitad.data import v8f_mini_data as D
    W = np.load(f"{RAW}/windows_train.npz", allow_pickle=False)
    clips = [str(c) for c in W["clips"]]
    # ---- the join the run trained on, rebuilt from the same inputs (md5-checked)
    jpath = f"{SCR}/join_train_rebuilt.npz"
    if os.path.exists(jpath):
        J = D.load_join(jpath)
    else:
        from tanitad.data.v9_labels import load_v9_release
        man = []
        for ln in open(MANIFEST_TRAIN, encoding="utf-8"):
            if ln.strip():
                r = json.loads(ln)
                man.append((r["clip"], int(r["n_frames"])))
        rel = load_v9_release(f"{SHIP}/v9_labels_train.npz", expect_md5=V9_MD5)
        J = D.build_label_join(man, rel, lead_npz=f"{SHIP}/lead_train4572.npz", speed=D.load_speed_sidecar(f"{SHIP}/refcv6_speed_max_v8_train.jsonl"))
        D.save_join(J, jpath)
    print(f"join ready {time.time() - t0:.0f}s", flush=True)
    tracks = D.load_native_tracks(TRACKS)
    _load = torch.load
    torch.load = lambda *aa, **kk: _load(*aa, **{**kk, "mmap": True, "weights_only": True})      # ClipStore.get reads one tensor of a 20 MB clip file: mmap
    store = D.ClipStore(CACHE_ROOT, "train", lru=6, grid="pool4x10", kin_mode="yaw", tracks=tracks)
    ds = D.V8FMiniWindows(store, J, [], hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s)
    # ---- items: both populations, sorted by (clip, t) so a clip is loaded once
    items = []
    n_pop = {}
    for pop in ("pstop", "pmove"):
        idx = np.load(f"{RAW}/sample_{pop}.npz")["idx"]
        n_pop[pop] = len(idx)
        for row, w in enumerate(idx):
            items.append((int(W["clip_i"][w]), int(W["t"][w]), pop, row))
    items.sort(key=lambda x: (x[0], x[1], x[2]))
    if a.limit:
        items = items[:a.limit]
    prog_p = f"{SCR}/z_progress.json"
    n_done = json.load(open(prog_p))["n_done"] if (os.path.exists(prog_p) and not a.limit) else 0
    out = {}
    dims = {"hO": cfg.n_q * cfg.d, "hT": cfg.d, "code": None, "ck": 9}
    cap = {}
    model.read_o.register_forward_hook(lambda m, i, o: cap.__setitem__("hO", o.flatten(1).detach().float()))
    model.trunk_t.register_forward_hook(lambda m, i, o: cap.__setitem__("hT", F.gelu(o).detach().float()))
    ctrl = {"max_abs_recon_traj60_minus_hook": 0.0, "n_batches": 0, "ade6_train_median_m": []}
    pend = [(i, it) for i, it in enumerate(items) if i >= n_done]
    z_path = {p: f"{SCR}/Z_{p}.npy" for p in n_pop}
    mode = "r+" if os.path.exists(z_path["pstop"]) and n_done > 0 else "w+"
    if a.limit:
        z_path = {p: f"{SCR}/Zsmoke_{p}.npy" for p in n_pop}
        mode = "w+"
    mm = {p: np.lib.format.open_memmap(z_path[p], mode=mode, dtype=np.float32, shape=(n_pop[p], dims["hO"])) for p in n_pop}
    sec_p = f"{SCR}/Zsec_{'smoke' if a.limit else 'full'}.npz"
    sec = dict(np.load(sec_p)) if (os.path.exists(sec_p) and n_done > 0) else None
    nn_ = {p: n_pop[p] for p in n_pop}
    first = True
    pos = n_done
    while pos < len(items):
        if time.time() - t0 > a.budget_s and not a.limit:
            break
        chunk = items[pos:pos + a.bs]
        its = [ds.item(ds.clip_index[clips[ci]], t, 0, 0) for ci, t, _, _ in chunk]
        batch = D.collate(its)
        feed = {k: batch[k] for k in M.INFERENCE_INPUT_KEYS}
        with torch.no_grad():
            o = model.forward_infer(feed)
        hO, hT = cap["hO"], cap["hT"]
        if first:
            print("forward_infer keys:", sorted(k for k in o.keys() if not k.startswith("_"))[:20], "keys read:", sorted(o["_keys_read"]), flush=True)
            code_dim = o["code"].shape[1]
            if sec is None:
                sec = {p: {"hT": np.zeros((nn_[p], dims["hT"]), np.float32), "code": np.zeros((nn_[p], code_dim), np.float32), "ck": np.zeros((nn_[p], 9), np.float32)} for p in n_pop}
                sec = {f"{p}__{k}": v for p, d_ in sec.items() for k, v in d_.items()}
            first = False
        # control: head_o(hO) + the prior reproduces the plan the model emitted (the hooked tensor IS what head_o reads)
        with torch.no_grad():
            coef = model.head_o(hO).reshape(-1, 2, cfg.nb) * cfg.res_scale_m
            res = torch.einsum("tk,bck->btc", model.basis, coef)
            rec = M.upsample_prior(o["p10"]) + res
        ctrl["max_abs_recon_traj60_minus_hook"] = max(ctrl["max_abs_recon_traj60_minus_hook"], float((rec - o["traj60"]).abs().max()))
        ctrl["n_batches"] += 1
        ade = torch.linalg.norm(o["traj30"] - batch["gt"], dim=-1).mean(1)
        ctrl["ade6_train_median_m"].append(float(ade.median()))
        for j, (ci, t, p, row) in enumerate(chunk):
            mm[p][row] = hO[j].numpy()
            sec[f"{p}__hT"][row] = hT[j].numpy()
            sec[f"{p}__code"][row] = o["code"][j].float().numpy()
            sec[f"{p}__ck"][row] = o["ckpt"][j].float().reshape(-1).numpy()
        pos += len(chunk)
        if (pos // a.bs) % 25 == 0:
            print(f"{pos}/{len(items)} {time.time() - t0:.0f}s", flush=True)
            for m_ in mm.values():
                m_.flush()
            np.savez(sec_p, **sec)
            if not a.limit:
                json.dump({"n_done": pos}, open(prog_p, "w"))
    for m_ in mm.values():
        m_.flush()
    if sec is not None:
        np.savez(sec_p, **sec)
    if not a.limit:
        json.dump({"n_done": pos}, open(prog_p, "w"))
    rep = {"n_done": pos, "n_total": len(items), "seconds": time.time() - t0, "ckpt_sha256": CKPT_SHA, "join_npz_sha256": sha256_file(jpath), "v9_md5": V9_MD5,
           "max_abs_recon_traj60_minus_hook": ctrl["max_abs_recon_traj60_minus_hook"], "median_of_batch_median_ade6_m": float(np.median(ctrl["ade6_train_median_m"])) if ctrl["ade6_train_median_m"] else None}
    json.dump(rep, open(f"{RAW}/extract_z_{'smoke' if a.limit else 'status'}.json", "w"), indent=1)
    print(json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
