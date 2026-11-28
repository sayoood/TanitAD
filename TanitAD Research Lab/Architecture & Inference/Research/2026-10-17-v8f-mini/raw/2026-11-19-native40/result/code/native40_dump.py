#!/usr/bin/env python3
"""native40_dump.py -- NATIVE40 study wrapper (PREREG_NATIVE40.md): T0 dump of the native lead head of LLR_s0 / LLR_s1 on EVERY evaluation window.
Zero training, CPU only, no metric computed here. Code of record = the LLR read tree (model 5fd4baea, tool v8f_t1_eval_ri 0417c838), loaded the way llr_readout.py loads it.
Per window it writes: ci, t0 (cache step of the head's frame), k = 2*t0 (raw frame of the label, v8f_mini_data.lead_label_arrays: kj = 2*arange), eid, frame_ok_now,
lead_logit, lead_gap_hat (= sigmoid of the 2nd head output = gap/40, v8f_mini.py _lead), has / lead / gap (from the label file lead_eval139.npz: the file the model was trained against the same way).
The head read is the model's own: adapt = model._encode(b); logit, gap_hat = model._lead(b, adapt, last_now) exactly as V8FMini._forward does (v8f_mini.py:870-875).
--keycheck: on the first batch also runs model.forward_infer and requires out["lead_logit"], out["lead_gap"] to equal the direct read, and (head-off model) the keys to be absent.
usage: native40_dump.py --seed S [--max-episodes N] [--threads 3] [--batch 32] [--out F:/.../dump_s0.npz] [--keycheck]"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np

sys.dont_write_bytecode = True
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
ART = "E:/Projects/TanitAD-artifacts"
TREE = "F:/TanitAD-scratch/motion_c/llr_read/tree"
TOOL = f"{TREE}/stack/scripts/v8f_t1_eval_ri.py"
TOOL_SHA = "0417c838355a51f6f7084c17b5acada96c5b6cbd9eef54cc2602d377bf3c61e7"
MODEL_SHA = "5fd4baea4d08a5400fc73cfd283c546d0bef528e71459f9845e6442244fc4d57"
DATA = f"{ART}/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
TRACKS = f"{ART}/sfresh_inputs/inputs/native_tracks.pt"
EVAL_CACHE = f"{ART}/sfresh_inputs/cache"
LEAD_EVAL = f"{ART}/sfresh_inputs/inputs/v9/lead_eval139.npz"
CKPT = f"{ART}/v8f-recover/M2b_s{{s}}_recft_LLR/ckpt_last.pt"
CKPT_SHA = {0: "5a52b28bcf3e2f153baeb0835f1e9dd44eb918319b51b1e804a5d686276817bc", 1: "71ee9feab2683a66a80e6a1fcbd4bc23e8a961ee7ef702ed160b00835cbf40c6"}
LEAD_RANGE_M = 40.0


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_tool():
    got = sha256(TOOL)
    if got != TOOL_SHA:
        raise SystemExit(f"REFUSED: tool {got} != pinned {TOOL_SHA}")
    got = sha256(f"{TREE}/stack/tanitad/models/v8f_mini.py")
    if got != MODEL_SHA:
        raise SystemExit(f"REFUSED: model {got} != pinned {MODEL_SHA}")
    sys.path.insert(0, f"{TREE}/stack")
    sys.path.insert(0, f"{TREE}/taniteval")
    spec = importlib.util.spec_from_file_location("v8f_t1_eval_ri", TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v8f_t1_eval_ri"] = mod
    spec.loader.exec_module(mod)
    import tanitad
    assert os.path.normcase(os.path.abspath(tanitad.__file__)).startswith(os.path.normcase(os.path.abspath(TREE))), f"wrong tanitad imported: {tanitad.__file__}"
    return mod


def label_lookup(T, J, wins):
    """per window: has / lead / gap at raw frame k = 2 * t0 from the lead file (rows keyed by sha12(clip), k) -- the same rows lead_label_arrays reads. Returns arrays + the near mask by the trainer's rule."""
    z = np.load(LEAD_EVAL, allow_pickle=True)
    sh, kk = np.asarray(z["sha12"]).astype(str), np.asarray(z["k"], np.int64)
    ld, gp = np.asarray(z["lead"], np.int64), np.asarray(z["gap"], np.float64)
    idx = {}
    for i, (s, k) in enumerate(zip(sh, kk)):
        idx[(s, int(k))] = i
    has = np.zeros(len(wins), bool)
    lead = np.zeros(len(wins), np.int64)
    gap = np.full(len(wins), np.nan)
    for w, (ci, t0, _, _) in enumerate(wins):
        i = idx.get((T.D.sha12(str(J["clip"][ci])), 2 * int(t0)))
        if i is not None:
            has[w], lead[w], gap[w] = True, ld[i], gp[i]
    near = has & (lead == 1) & np.isfinite(gap) & (gap <= LEAD_RANGE_M)
    return has, lead, gap, near


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--max-episodes", type=int, default=0)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default="")
    ap.add_argument("--keycheck", action="store_true")
    ap.add_argument("--extra-windows", default="")
    a = ap.parse_args(argv)
    import torch
    torch.set_num_threads(a.threads)
    T = load_tool()
    s = a.seed
    ck_path = CKPT.format(s=s)
    got = sha256(ck_path)
    if got != CKPT_SHA[s]:
        raise SystemExit(f"REFUSED: checkpoint {ck_path} sha256 {got} != bank record {CKPT_SHA[s]}")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    cfg = T.M.V8FMiniConfig(**ck["cfg"])
    assert cfg.lead_head == "on" and cfg.aux_cond == "token" and abs(cfg.w_lead - 3.362) < 1e-9, "checkpoint is not the LLR arm (lead_head on, w_lead 3.362, aux token)"
    model = T.M.V8FMini(cfg, T.M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * T.M.K30)))
    model.load_state_dict(ck["model"])
    model.eval()
    tracks = T.D.load_native_tracks(TRACKS)
    J = T.D.load_join(f"{DATA}/join_eval141.npz")
    store = T.D.ClipStore(EVAL_CACHE, "eval", lru=8, grid=ck.get("grid", "pool4x10"), kin_mode="yaw", tracks=tracks)
    wins = T.D.eval_grid_windows(f"{DATA}/eval_grid_windows.json", J)
    if a.max_episodes:
        keep = list(dict.fromkeys(w[0] for w in wins))[:a.max_episodes]
        wins = [w for w in wins if w[0] in keep]
    has, lead, gap, near = label_lookup(T, J, wins)
    proxy = T._OneClipStore()
    ds = T.D.V8FMiniWindows(proxy, J, [], hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s, window_start=cfg.window_start)
    a_t, a_s = T.ages_for(0, "periodic", 3, 6)
    keyinfo = {}
    t_start = time.time()

    def read(wins_, keycheck):
        L_logit, L_gap, L_ok = [], [], []
        for b0 in range(0, len(wins_), a.batch):
            chunk = wins_[b0:b0 + a.batch]
            states = T.make_states(store, tracks, J, chunk, "deadreckon")
            items = [T.build_item(ds, proxy, st, 0, a_t, a_s, "logged", "causal", None) for st in states]    # the rollout's own tick-0 batch (logged frames)
            full = T.D.collate(items)
            batch = {k: full[k] for k in T.MODEL_KEYS}
            with torch.no_grad():
                adapt = model._encode(batch)
                Fn = batch["feats"].shape[1]
                last_now = torch.full((batch["feats"].shape[0],), Fn - 1, dtype=torch.long)
                logit, gap_hat = model._lead(batch, adapt, last_now)[:2]
                if keycheck and b0 == 0:
                    out = model.forward_infer(batch)
                    keyinfo["keys_present_head_on"] = bool("lead_logit" in out and "lead_gap" in out)
                    if keyinfo["keys_present_head_on"]:
                        keyinfo["forward_infer_vs_direct_max_abs_dev"] = float(max((out["lead_logit"].float() - logit).abs().max(), (out["lead_gap"].float() - gap_hat).abs().max()))
                    import dataclasses
                    cfg_off = dataclasses.replace(cfg, lead_head="off", w_lead=0.0, aux_cond="add")
                    m_off = T.M.V8FMini(cfg_off, T.M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * T.M.K30))).eval()
                    out_off = m_off.forward_infer(batch)
                    keyinfo["keys_absent_head_off"] = bool("lead_logit" not in out_off and "lead_gap" not in out_off)
                    del m_off, out_off
            L_logit.append(logit.double().numpy())
            L_gap.append(gap_hat.double().numpy())
            L_ok.append(batch["frame_ok"][:, -1].numpy().astype(bool))
            print(f"[s{s}] {b0 + len(chunk)}/{len(wins_)} {time.time() - t_start:.0f}s", flush=True)
        return np.concatenate(L_logit), np.concatenate(L_gap), np.concatenate(L_ok)

    logit_a, gap_a, ok_a = read(wins, a.keycheck)
    extra = {}
    if a.extra_windows:        # the coordinator's label frames: cache step g0 of each window (see native40_score.py map control); head read at THAT frame, same builder
        WJ = json.load(open(a.extra_windows, encoding="utf-8"))
        ci_of = {c: i for i, c in enumerate(J["clip"].tolist())}
        pairs = sorted({(ci_of[e], int(g)) for e, g in zip(WJ["eid"], WJ["g0"])})
        if a.max_episodes:
            keep = {w[0] for w in wins}
            pairs = [p for p in pairs if p[0] in keep]
        xw = [(c, g, 0, 0) for c, g in pairs]
        xl, xg, xo = read(xw, False)
        xh, xld, xgp, xnr = label_lookup(T, J, xw)
        extra = dict(x_ci=np.array([w[0] for w in xw]), x_t0=np.array([w[1] for w in xw]), x_eid=np.array([str(J["clip"][w[0]]) for w in xw]), x_frame_ok=xo, x_lead_logit=xl, x_lead_gap_hat=xg,
                     x_has=xh, x_lead=xld, x_gap=xgp, x_near=xnr)
    ci = np.array([w[0] for w in wins])
    t0 = np.array([w[1] for w in wins])
    eid = np.array([str(J["clip"][c]) for c in ci])
    out_path = a.out or f"F:/TanitAD-scratch/native40/out/dump_s{s}.npz"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    meta = {"seed": s, "ckpt": ck_path, "ckpt_sha256": CKPT_SHA[s], "tool_sha256": TOOL_SHA, "model_sha256": MODEL_SHA, "n_windows": int(len(wins)), "n_episodes": int(len(set(eid.tolist()))),
            "max_episodes": a.max_episodes, "keycheck": keyinfo, "wall_s": time.time() - t_start,
            "label_file": LEAD_EVAL, "label_file_md5_pinned": "deb1b2a92c860dc924d6fd3ff0d6dc85", "label_rule": "row (sha12(clip), k = 2*t0) of lead_eval139.npz; near = lead==1 & finite gap & gap<=40 (v8f_mini_data.lead_label_arrays:292)"}
    np.savez(out_path, ci=ci, t0=t0, k=2 * t0, eid=eid, frame_ok_now=ok_a, lead_logit=logit_a, lead_gap_hat=gap_a,
             has=has, lead=lead, gap=gap, near=near, meta=json.dumps(meta), **extra)
    print("wrote", out_path, json.dumps(keyinfo))
    return 0


if __name__ == "__main__":
    sys.exit(main())
