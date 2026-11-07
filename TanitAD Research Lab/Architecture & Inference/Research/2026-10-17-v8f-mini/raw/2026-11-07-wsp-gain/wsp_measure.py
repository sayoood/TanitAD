"""wsp_measure: tick-0 batches (the RI tool's own builder, p2b_check.py's recipe) for TRAIN or eval-141 windows, the replicated forward, and the longitudinal gains.

Plan speed s := (x30[11] - x30[7]) / 0.8 (the 0.8 s secant at +2 s, the P1 / P2 definition) = s_base + s_net + s_res.
Gains := central finite differences of s_tot w.r.t. v0 and a0 with the ego chain rebuilt consistently (f8 = kin_features(kin3), base = kdx_base(kin3)); eps 0.05 / 0.05 (p2b_check.py FD_EPS).
`h2` := the INPUT of head_o's last linear layer (the post-GELU hidden, 2d = 512), the vector the edit's w lives in.
Dev-box CPU only, zero training.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).replace("\\", "/"))
import numpy as np
import torch
import wsp_common as W

KEYS = ["feats", "frame_ok", "p10f", "nav", "limit", "limit_ok", "base", "a_t", "a_s", "pose_now", "pose_t", "pose_s", "kin3"]
FD_EPS = np.array([0.05, 0.05, 5e-4])


class Rig:
    """Tool module, data module, store / join / dataset for one split, and the window-batch builder."""

    def __init__(self, split, cfg, nt=8):
        torch.set_num_threads(nt)
        self.T = W.load_tool()
        self.D, self.M = self.T.D, self.T.M
        D = self.D
        self.cfg, self.split = cfg, split
        self.tracks = D.load_native_tracks(W.TRACKS)
        if split == "eval":
            self.J = D.load_join(W.DATA + "/join_eval141.npz")
            self.store = D.ClipStore(W.ART + "/sfresh_inputs/cache", "eval", lru=32, grid="pool4x10", kin_mode="yaw", tracks=self.tracks)
        else:
            self.J = D.load_join(W.TRAIN_JOIN)
            _load = torch.load
            torch.load = lambda *aa, **kk: _load(*aa, **{**kk, "mmap": True, "weights_only": True})      # ClipStore.get reads one tensor of a 20 MB clip: mmap
            self.store = D.ClipStore(W.TRAIN_CACHE, "train", lru=6, grid="pool4x10", kin_mode="yaw", tracks=self.tracks)
        self.proxy = self.T._OneClipStore()
        self.ds = D.V8FMiniWindows(self.proxy, self.J, [], hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s, window_start=cfg.window_start)
        self.ci_of = {c: i for i, c in enumerate(self.J["clip"].tolist())}
        self.H4 = cfg.hist

    def eval_windows(self):
        return self.D.eval_grid_windows(W.DATA + "/eval_grid_windows.json", self.J)

    def train_windows(self, ids_txt):
        out = []
        for ln in open(ids_txt, encoding="utf-8"):
            ln = ln.strip()
            if ln:
                c, t = ln.rsplit(":", 1)
                out.append((self.ci_of[c], int(t), 0, 0))
        return out

    def v0_of(self, win):
        ci, t0, _, _ = win
        return float(self.store.get(str(self.J["clip"][ci]))["kin3"][t0, 0])

    def build(self, chunk):
        """Tick-0 batch for a window chunk: logged history, dead-reckoned nav, causal speed bucket, a_t = a_s = 0, scene logged (p2b_check.py)."""
        T, D = self.T, self.D
        sts = T.make_states(self.store, self.tracks, self.J, chunk, "deadreckon")
        items = [T.build_item(self.ds, self.proxy, st, 0, 0, 0, "logged", "causal", None) for st in sts]
        full = D.collate(items)
        b = {}
        for k in KEYS:
            v = full[k]
            if k in ("feats", "frame_ok", "p10f", "nav"):
                v = v[:, -self.H4:]
            b[k] = v.clone()
        b["clip"] = [str(self.J["clip"][ci]) for ci, _, _, _ in chunk]
        b["t0"] = np.array([t for _, t, _, _ in chunk])
        return b


def base_inputs(b, model, f8=None, base=None, u0=None):
    B = b["feats"].shape[0]
    f8_ = b["p10f"][:, -1] if f8 is None else f8
    base_ = b["base"] if base is None else base
    if u0 is None:
        u0 = model.adapter[0](b["feats"].float())
    return {"u_st": u0, "u_o": u0, "f8_st": f8_, "f8_tok": f8_, "f8_pri": f8_, "base": base_, "nav_st": b["nav"][:, -1], "nav_o": b["nav"][:, -1],
            "lim_st": b["limit"], "lim_o": b["limit"], "limit_ok": b["limit_ok"], "frame_ok": b["frame_ok"],
            "p10f_hist": b["p10f"], "nav_hist": b["nav"], "a_t": b["a_t"], "a_s": b["a_s"],
            "pose_now": b["pose_now"], "pose_t": b["pose_t"], "pose_s": b["pose_s"]}


def spd(x):
    return (x[:, W.J_HI, 0] - x[:, W.J_LO, 0]) / W.DT_SEC


def forward_h(model, cfg, M, inp, B):
    """p2b_check.py's replicated forward (verified there against forward_infer to 0.0), returning the plan, its speed components and h2 = head_o[2]'s input."""
    F = cfg.hist
    last = torch.full((B,), F - 1, dtype=torch.long)
    pe = model.pos[None, None]
    adapt_st = model.adapter[1](inp["u_st"]) + pe
    adapt_o = model.adapter[1](inp["u_o"]) + pe
    p10f_st = inp["p10f_hist"].clone(); p10f_st[:, -1] = inp["f8_st"]
    nav_st = inp["nav_hist"].clone(); nav_st[:, -1] = inp["nav_st"]
    bb = {"p10f": p10f_st, "nav": nav_st, "limit": inp["lim_st"], "limit_ok": inp["limit_ok"], "frame_ok": inp["frame_ok"]}
    last_s, last_t = last - inp["a_s"], last - inp["a_t"]
    ck_s = model.run_s(bb, adapt_st, last_s)
    ck_ok = torch.ones(B, 3)
    ck_for_t = M.transform_ckpt(ck_s, inp["pose_s"], inp["pose_t"])
    tout = model.run_t(bb, adapt_st, last_t, ck_for_t, ck_ok)
    code = model._code_pred(tout)
    ck_for_o = M.transform_ckpt(ck_s, inp["pose_s"], inp["pose_now"])
    ck_in = torch.cat([ck_for_o[..., :2] / 50.0, ck_for_o[..., 2:3], ck_ok[..., None]], -1).flatten(1)
    ages = torch.stack([inp["a_t"].float() * M.DT, inp["a_s"].float() * M.DT], -1)
    z_ego = model.prior.standardise(inp["f8_tok"])
    lim_in = torch.cat([inp["lim_o"], inp["limit_ok"][:, None].float()], -1)
    cond = torch.stack([model.e_ego(z_ego), model.e_nav(inp["nav_o"]), model.e_lim(lim_in), model.e_ck(ck_in), model.e_code(code), model.e_age(ages)], 1)
    sc, pad = model._scene(adapt_o, inp["frame_ok"], last)
    h = model.read_o(cond, sc, pad).flatten(1)
    h2 = model.head_o[1](model.head_o[0](h.float()))
    out = model.head_o[2](h2)
    coef = out.reshape(-1, 2, cfg.nb) * cfg.res_scale_m
    res = torch.einsum("tk,bck->btc", model.basis, coef)
    p30 = model.prior(inp["f8_pri"], inp["base"])
    traj60 = M.upsample_prior(p30) + res
    traj30 = M.subsample_02(traj60)
    return {"s_tot": spd(traj30), "s_res": spd(res[:, 1::2]), "s_net": spd(p30 - inp["base"]), "s_base": spd(inp["base"]), "traj30": traj30, "h2": h2, "out": out}


def ego_inputs(rig, b, model, k3, u0):
    f8_ = torch.as_tensor(rig.T.fk.kin_features(k3), dtype=torch.float32)
    b_ = torch.as_tensor(rig.D.kdx_base(k3, rig.D.KIN_UNITS["yaw"]), dtype=torch.float32)
    return base_inputs(b, model, f8=f8_, base=b_, u0=u0)


def spd_np(b):
    return (b[:, W.J_HI, 0].astype(np.float64) - b[:, W.J_LO, 0].astype(np.float64)) / W.DT_SEC


def base_gains(rig, kin3):
    """d s_base / d (v0, a0) by central FD of the non-learned kd_x base (p2b_check.py)."""
    out = np.zeros((len(kin3), 2))
    for d in range(2):
        kp, km = kin3.copy(), kin3.copy()
        kp[:, d] += FD_EPS[d]; km[:, d] -= FD_EPS[d]
        out[:, d] = (spd_np(rig.D.kdx_base(kp, rig.D.KIN_UNITS["yaw"])) - spd_np(rig.D.kdx_base(km, rig.D.KIN_UNITS["yaw"]))) / (2 * FD_EPS[d])
    return out


def measure_chunk(rig, models, b):
    """For each model: nominal s components, h2, and the FD gains (d s_tot / d v0, d s_tot / d a0). Returns {name: array} per model plus the shared kin3 / base gains."""
    B = b["feats"].shape[0]
    kin3 = b["kin3"].double().numpy()
    res = {"kin3": kin3, "gb": base_gains(rig, kin3)}
    for name, (model, cfg) in models.items():
        with torch.no_grad():
            u0 = model.adapter[0](b["feats"].float())
            nom = forward_h(model, cfg, rig.M, base_inputs(b, model, u0=u0), B)
            g = np.zeros((B, 2))
            for d in range(2):
                kp, km = kin3.copy(), kin3.copy()
                kp[:, d] += FD_EPS[d]; km[:, d] -= FD_EPS[d]
                ip, im = ego_inputs(rig, b, model, kp, u0), ego_inputs(rig, b, model, km, u0)
                sp = forward_h(model, cfg, rig.M, ip, B)["s_tot"].double().numpy()
                sm = forward_h(model, cfg, rig.M, im, B)["s_tot"].double().numpy()
                g[:, d] = (sp - sm) / (2 * FD_EPS[d])
        res[name] = {"s_tot": nom["s_tot"].double().numpy(), "s_res": nom["s_res"].double().numpy(), "s_net": nom["s_net"].double().numpy(), "s_base": nom["s_base"].double().numpy(),
                     "h2": nom["h2"].float().numpy(), "gv": g[:, 0], "ga": g[:, 1], "traj30": nom["traj30"].float().numpy()}
    return res


def run_measure(rig, models, wins, bs=64, log=None, keep_traj=False):
    """Stream the windows through measure_chunk; concatenate. models: {name: (model, cfg)}."""
    t0 = time.time()
    acc = {}
    for b0 in range(0, len(wins), bs):
        chunk = wins[b0:b0 + bs]
        b = rig.build(chunk)
        r = measure_chunk(rig, models, b)
        r["clip"] = np.array(b["clip"]); r["t0"] = b["t0"]
        for k, v in r.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    if kk == "traj30" and not keep_traj:
                        continue
                    acc.setdefault(f"{k}.{kk}", []).append(vv)
            else:
                acc.setdefault(k, []).append(v)
        if log and (b0 // bs) % 20 == 0:
            log(f"measured {b0 + len(chunk)}/{len(wins)}  ({time.time()-t0:.0f}s)")
    return {k: np.concatenate(v, 0) for k, v in acc.items()}
