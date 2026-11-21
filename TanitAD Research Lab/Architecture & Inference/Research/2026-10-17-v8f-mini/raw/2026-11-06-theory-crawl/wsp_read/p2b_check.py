"""P2b: the check of CLAIMS_P2.md (sha afb9785f..., stamped 22:14:27Z) on held-out windows. CPU only, zero training, no weight edit.
Windows: eval-141 (2,399 windows), tick 0 of the T1-kin rollout (logged history, dead-reckoned nav, causal speed bucket, a_t = a_s = 0, scene logged), raw forward_infer plan.
Plan speed s = (x30[11] - x30[7]) / 0.8 = s_base + s_net + s_res.  Interpretation for W2 fixed BEFORE the first forward: the O-level partial derivative (the reader's direct
inputs) is the primary reading of the capacity claim; the total derivative (through S and T as well) is reported beside it.
Run:  PYTHONPATH="<tree>/stack;<tree>/taniteval" python p2b_check.py   (NT = threads)
"""
import os, sys, json, time, importlib.util
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import torch

NT = int(os.environ.get("NT", "8"))
torch.set_num_threads(NT)
TREE = "E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a"
ART = "E:/Projects/TanitAD-artifacts"
import tanitad
assert os.path.abspath(tanitad.__file__).replace("\\", "/").startswith(TREE), tanitad.__file__
HERE = "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/wsp_read"
RI_TOOL = ART + "/dagland/motion_c/ri/code/v8f_t1_eval_ri.py"
DATA = TREE + "/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
CKPT = ART + "/v8f-recover/M2b_s0_recft_C0/ckpt_last.pt"
SEED_ARG = [a for a in sys.argv[1:] if a.startswith("--ckpt=")]
if SEED_ARG:
    CKPT = SEED_ARG[0].split("=", 1)[1]
OUTTAG = os.environ.get("OUTTAG", "P2B")
MAXW = int(os.environ.get("MAXW", "0"))

spec = importlib.util.spec_from_file_location("ri_tool", RI_TOOL)
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)
D, M = T.D, T.M
t_start = time.time()
def log(*a):
    print(f"[{time.time()-t_start:7.0f}s]", *a, flush=True)

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
# the ckpt's cfg carries launch_intent_w (a LOSS-side lever, 0.0 = inactive) that the dagt1 tree's config does not know; the tree_ri source the T1 runs used differs from the
# dagt1 source ONLY in that lever (diff checked: the cfg field, its validation, launch_mask / launch_along_term, one loss line, one built_levers line) -- the inference path is identical
_cfgd = dict(ck["cfg"])
assert _cfgd.pop("launch_intent_w", 0.0) == 0.0
cfg = M.V8FMiniConfig(**_cfgd)
model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
model.load_state_dict(ck["model"])
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
XS = model.prior.xs.double().numpy()            # (v, a, k, ...) standardiser scale
FD_EPS = np.array([0.05, 0.05, 5e-4])

# ------------------------------------------------------------------ data
tracks = D.load_native_tracks(ART + "/sfresh_inputs/inputs/native_tracks.pt")
J = D.load_join(DATA + "/join_eval141.npz")
store = D.ClipStore(ART + "/sfresh_inputs/cache", "eval", lru=32, grid=ck.get("grid", "pool4x10"), kin_mode="yaw", tracks=tracks)
wins = D.eval_grid_windows(DATA + "/eval_grid_windows.json", J)
if MAXW:
    wins = [wins[i] for i in np.linspace(0, len(wins) - 1, MAXW).round().astype(int)]
proxy = T._OneClipStore()
ds = D.V8FMiniWindows(proxy, J, [], hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s, window_start=cfg.window_start)
H4 = cfg.hist
KEYS = ["feats", "frame_ok", "p10f", "nav", "limit", "limit_ok", "base", "a_t", "a_s", "pose_now", "pose_t", "pose_s", "kin3"]
acc = {k: [] for k in KEYS}
clip_of = []
for b0 in range(0, len(wins), 32):
    chunk = wins[b0:b0 + 32]
    sts = T.make_states(store, tracks, J, chunk, "deadreckon")
    items = [T.build_item(ds, proxy, st, 0, 0, 0, "logged", "causal", None) for st in sts]
    full = D.collate(items)
    for k in KEYS:
        v = full[k]
        if k in ("feats", "frame_ok", "p10f", "nav"):
            v = v[:, -H4:]
        acc[k].append(v.clone())
    clip_of += [str(J["clip"][ci]) for ci, _, _, _ in chunk]
    if (b0 // 32) % 10 == 0:
        log(f"built {b0 + len(chunk)}/{len(wins)}")
BATCH = {k: torch.cat(v, 0) for k, v in acc.items()}
del acc
N = BATCH["feats"].shape[0]
clip_ids = np.array(clip_of)
kin3 = BATCH["kin3"].double().numpy()            # (v0, a0, kappa) at t0
v0 = kin3[:, 0]
log("built", N, "windows; feats", tuple(BATCH["feats"].shape), BATCH["feats"].dtype)
assert np.allclose(BATCH["p10f"][:, -1].double().numpy(), T.fk.kin_features(kin3), rtol=1e-3, atol=1e-3), "p10f != kin_features(kin3)"
base_chk = D.kdx_base(kin3, D.KIN_UNITS["yaw"])
assert np.abs(base_chk - BATCH["base"].numpy()).max() < 1e-3, "base != kdx_base(kin3)"
BANDS = [("v<3", -1e9, 3.0), ("3-8", 3.0, 8.0), ("8-15", 8.0, 15.0), ("v>=15", 15.0, 1e9)]
band_of = np.zeros(N, int)
for i, (_, lo, hi) in enumerate(BANDS):
    band_of[(v0 >= lo) & (v0 < hi)] = i

def spd(x):
    return (x[:, 11, 0] - x[:, 7, 0]) / 0.8

# ------------------------------------------------------------------ the replicated forward (verified against forward_infer below)
def forward_parts(inp, B, code_xf=None, ck_xf=None):
    F = H4
    last = torch.full((B,), F - 1, dtype=torch.long)
    pe = model.pos[None, None]
    adapt_st = model.adapter[1](inp["u_st"]) + pe
    adapt_o = model.adapter[1](inp["u_o"]) + pe
    p10f_st = inp["p10f_hist"].clone(); p10f_st[:, -1] = inp["f8_st"]
    nav_st = inp["nav_hist"].clone(); nav_st[:, -1] = inp["nav_st"]
    b = {"p10f": p10f_st, "nav": nav_st, "limit": inp["lim_st"], "limit_ok": inp["limit_ok"], "frame_ok": inp["frame_ok"]}
    last_s, last_t = last - inp["a_s"], last - inp["a_t"]
    ck_s = model.run_s(b, adapt_st, last_s)
    ck_ok = torch.ones(B, 3)
    ck_for_t = M.transform_ckpt(ck_s, inp["pose_s"], inp["pose_t"])
    tout = model.run_t(b, adapt_st, last_t, ck_for_t, ck_ok)
    code = model._code_pred(tout)
    if code_xf is not None:
        code = code_xf(code)
    ck_for_o = M.transform_ckpt(ck_s, inp["pose_s"], inp["pose_now"])
    ck_in = torch.cat([ck_for_o[..., :2] / 50.0, ck_for_o[..., 2:3], ck_ok[..., None]], -1).flatten(1)
    if ck_xf is not None:
        ck_in = ck_xf(ck_in)
    ages = torch.stack([inp["a_t"].float() * M.DT, inp["a_s"].float() * M.DT], -1)
    z_ego = model.prior.standardise(inp["f8_tok"])
    lim_in = torch.cat([inp["lim_o"], inp["limit_ok"][:, None].float()], -1)
    cond = torch.stack([model.e_ego(z_ego), model.e_nav(inp["nav_o"]), model.e_lim(lim_in), model.e_ck(ck_in), model.e_code(code), model.e_age(ages)], 1)
    sc, pad = model._scene(adapt_o, inp["frame_ok"], last)
    h = model.read_o(cond, sc, pad).flatten(1)
    coef = model.head_o(h).reshape(-1, 2, cfg.nb) * cfg.res_scale_m
    res = torch.einsum("tk,bck->btc", model.basis, coef)
    p30 = model.prior(inp["f8_pri"], inp["base"])
    traj60 = M.upsample_prior(p30) + res
    traj30 = M.subsample_02(traj60)
    return {"s_tot": spd(traj30), "s_res": spd(res[:, 1::2]), "s_net": spd(p30 - inp["base"]), "s_base": spd(inp["base"]), "traj30": traj30,
            "code": code, "ck_in": ck_in, "z_ego": z_ego}

def base_inputs(sl, f8=None, base=None, u0=None):
    """plain (no-grad) inputs for window slice ``sl`` with the logged values; overrides f8 / base / u0 when given."""
    B = sl.stop - sl.start
    f8_ = BATCH["p10f"][sl, -1] if f8 is None else f8
    base_ = BATCH["base"][sl] if base is None else base
    if u0 is None:
        u0 = model.adapter[0](BATCH["feats"][sl].float())
    return {"u_st": u0, "u_o": u0, "f8_st": f8_, "f8_tok": f8_, "f8_pri": f8_, "base": base_, "nav_st": BATCH["nav"][sl, -1], "nav_o": BATCH["nav"][sl, -1],
            "lim_st": BATCH["limit"][sl], "lim_o": BATCH["limit"][sl], "limit_ok": BATCH["limit_ok"][sl], "frame_ok": BATCH["frame_ok"][sl],
            "p10f_hist": BATCH["p10f"][sl], "nav_hist": BATCH["nav"][sl], "a_t": BATCH["a_t"][sl], "a_s": BATCH["a_s"][sl],
            "pose_now": BATCH["pose_now"][sl], "pose_t": BATCH["pose_t"][sl], "pose_s": BATCH["pose_s"][sl]}

# ------------------------------------------------------------------ control 1: the replicated forward equals forward_infer
ctrl = {}
mx = 0.0
with torch.no_grad():
    for b0 in range(0, min(N, 128), 32):
        sl = slice(b0, min(b0 + 32, N))
        inp = base_inputs(sl)
        o = forward_parts(inp, sl.stop - sl.start)
        bt = {k: BATCH[k][sl] for k in KEYS if k != "kin3"}
        ref = model.forward_infer(bt)["traj30"]
        mx = max(mx, float((o["traj30"] - ref).abs().max()))
        # components add up
        assert float((o["s_tot"] - (o["s_base"] + o["s_net"] + o["s_res"])).abs().max()) < 1e-3
ctrl["replica_vs_forward_infer_max_abs_m"] = mx
log("control 1: replica vs forward_infer max |d traj30| =", mx)
assert mx < 1e-5, mx

# ------------------------------------------------------------------ f8 / base Jacobians wrt kin3
def jf8(k3):
    v, a, k = k3[:, 0], k3[:, 1], k3[:, 2]
    z = np.zeros_like(v); o = np.ones_like(v)
    return np.stack([np.stack([o, z, z, 2 * v, a, k, 2 * v * k, z], 1),
                     np.stack([z, o, z, z, v, z, z, k], 1),
                     np.stack([z, z, o, z, z, v, v * v, a], 1)], 1)       # [N, 3(kin), 8(f8)]
JF8 = jf8(kin3)
def spd_np(b):
    return (b[:, 11, 0].astype(np.float64) - b[:, 7, 0].astype(np.float64)) / 0.8
ds_base = np.zeros((N, 3))
for d in range(3):
    kp, km = kin3.copy(), kin3.copy()
    kp[:, d] += FD_EPS[d]; km[:, d] -= FD_EPS[d]
    ds_base[:, d] = (spd_np(D.kdx_base(kp, D.KIN_UNITS["yaw"])) - spd_np(D.kdx_base(km, D.KIN_UNITS["yaw"]))) / (2 * FD_EPS[d])
log("base FD jacobian: median d s_base/d v0, a0, kappa =", np.median(ds_base, 0))

# ------------------------------------------------------------------ Jacobian pass (z-units), per window
SC = ("s_tot", "s_res", "s_net")
NORM = {}
GK = {s: np.zeros((N, 3)) for s in SC}          # total gradient wrt kin3 (v0, a0, kappa), raw units
t_j = time.time()
for b0 in range(0, N, 32):
    sl = slice(b0, min(b0 + 32, N))
    B = sl.stop - sl.start
    inp = base_inputs(sl)
    u0 = inp["u_st"].detach()
    leaves = {}
    def leaf(name, x):
        t = x.detach().clone().requires_grad_(True); leaves[name] = t; return t
    inp2 = dict(inp)
    inp2["u_st"], inp2["u_o"] = leaf("u_st", u0), leaf("u_o", u0)
    f8 = BATCH["p10f"][sl, -1]
    inp2["f8_st"], inp2["f8_tok"], inp2["f8_pri"] = leaf("f8_st", f8), leaf("f8_tok", f8), leaf("f8_pri", f8)
    inp2["base"] = leaf("base", BATCH["base"][sl])
    inp2["nav_st"], inp2["nav_o"] = leaf("nav_st", BATCH["nav"][sl, -1]), leaf("nav_o", BATCH["nav"][sl, -1])
    inp2["lim_st"], inp2["lim_o"] = leaf("lim_st", BATCH["limit"][sl]), leaf("lim_o", BATCH["limit"][sl])
    o = forward_parts(inp2, B)
    wanted = [leaves["u_st"], leaves["u_o"], leaves["f8_st"], leaves["f8_tok"], leaves["f8_pri"], leaves["base"], leaves["nav_st"], leaves["nav_o"],
              leaves["lim_st"], leaves["lim_o"], o["code"], o["ck_in"], o["z_ego"]]
    names = ["u_st", "u_o", "f8_st", "f8_tok", "f8_pri", "base", "nav_st", "nav_o", "lim_st", "lim_o", "code", "ck_in", "z_ego"]
    for s in SC:
        g = torch.autograd.grad(o[s].sum(), wanted, retain_graph=True, allow_unused=True)
        G = {n: (None if x is None else x.double().numpy()) for n, x in zip(names, g)}
        Z = lambda n, shp: np.zeros(shp) if G[n] is None else G[n]
        nrm = lambda x: np.sqrt((x.reshape(x.shape[0], -1) ** 2).sum(1))
        gu_st, gu_o = Z("u_st", (B, 1)), Z("u_o", (B, 1))
        def rec(name, val):
            NORM.setdefault(f"{s}.{name}", np.full(N, np.nan))[sl] = val
        rec("vision_total_z", nrm(gu_st + gu_o)); rec("vision_O_z", nrm(gu_o)); rec("vision_ST_z", nrm(gu_st))
        g_f8_tot = Z("f8_st", (B, 8)) + Z("f8_pri", (B, 8)) + Z("z_ego", (B, 8)) / XS[None, :8]    # z_ego grad -> f8 units via the standardiser scale
        gk = np.einsum("nkf,nf->nk", JF8[sl], g_f8_tot)
        if s in ("s_tot", "s_base"):
            gk = gk + ds_base[sl]
        GK[s][sl] = gk
        rec("ego_total_z", nrm(gk * XS[None, :3])); rec("ego_total_raw", nrm(gk))
        rec("ego_O_f8z", nrm(Z("z_ego", (B, 8)))); rec("ego_ST_f8z", nrm(Z("f8_st", (B, 8)) * XS[None, :8]))
        rec("ego_prior_f8z", nrm(Z("f8_pri", (B, 8)) * XS[None, :8]))
        rec("nav_total", nrm(Z("nav_st", (B, 5)) + Z("nav_o", (B, 5)))); rec("nav_O", nrm(Z("nav_o", (B, 5))))
        rec("lim_total", nrm(Z("lim_st", (B, 4)) + Z("lim_o", (B, 4)))); rec("lim_O", nrm(Z("lim_o", (B, 4))))
        rec("code_O", nrm(Z("code", (B, 60)))); rec("ck_O", nrm(Z("ck_in", (B, 12))))
        rec("code_lon_O", nrm(Z("code", (B, 60))[:, 27:34]))
        if s == "s_res":
            NORM.setdefault("ctrl.s_res.base_grad_is_None", np.zeros(N))[sl] = float(G["base"] is None)
            NORM.setdefault("ctrl.s_res.f8_pri_grad_is_None", np.zeros(N))[sl] = float(G["f8_pri"] is None)
        if s == "s_net":
            NORM.setdefault("ctrl.s_net.u_o_grad_is_None", np.zeros(N))[sl] = float(G["u_o"] is None)
    if (b0 // 32) % 10 == 0:
        log(f"jacobian {sl.stop}/{N}  ({time.time()-t_j:.0f}s)")
# raw vision gradient (total) from the feats leaf, s_tot only
RAWV = np.zeros(N)
for b0 in range(0, N, 32):
    sl = slice(b0, min(b0 + 32, N)); B = sl.stop - sl.start
    fl = BATCH["feats"][sl].float().clone().requires_grad_(True)
    u = model.adapter[0](fl)
    inp = base_inputs(sl, u0=u)
    o = forward_parts(inp, B)
    g, = torch.autograd.grad(o["s_tot"].sum(), [fl])
    RAWV[sl] = np.sqrt((g.double().numpy().reshape(B, -1) ** 2).sum(1))
log("raw vision gradient done")
NORM["s_tot.vision_total_raw"] = RAWV

# ------------------------------------------------------------------ control 2: directional-derivative checks (autograd vs finite difference), first 64 windows
fd = {}
sl = slice(0, min(64, N)); B = sl.stop - sl.start
with torch.no_grad():
    u0 = model.adapter[0](BATCH["feats"][sl].float())
# vision: direction = total gradient of s_tot wrt u
u_st = u0.clone().requires_grad_(True); u_o = u0.clone().requires_grad_(True)
inp = base_inputs(sl); inp["u_st"], inp["u_o"] = u_st, u_o
o = forward_parts(inp, B)
gst, go = torch.autograd.grad(o["s_tot"].sum(), [u_st, u_o])
gtot = (gst + go)
gn = gtot.reshape(B, -1).norm(dim=1)
dirv = gtot / gn.reshape(B, 1, 1, 1)
rows = []
for eps in (1e-3, 1e-2, 1e-1):
    with torch.no_grad():
        sp = forward_parts(base_inputs(sl, u0=u0 + eps * dirv), B)["s_tot"]
        sm = forward_parts(base_inputs(sl, u0=u0 - eps * dirv), B)["s_tot"]
    fdv = ((sp - sm) / (2 * eps)).double().numpy()
    rows.append({"eps": eps, "median_ratio_fd_over_autograd": float(np.median(fdv / gn.double().numpy())), "min": float((fdv / gn.double().numpy()).min()), "max": float((fdv / gn.double().numpy()).max())})
fd["vision_directional"] = rows
# ego chain: perturb v0, a0, kappa consistently (f8 and base rebuilt)
def ego_inputs(sl, k3):
    f8_ = torch.as_tensor(T.fk.kin_features(k3), dtype=torch.float32)
    b_ = torch.as_tensor(D.kdx_base(k3, D.KIN_UNITS["yaw"]), dtype=torch.float32)
    return base_inputs(sl, f8=f8_, base=b_)
out_e = []
for d in range(3):
    kp, km = kin3[sl].copy(), kin3[sl].copy()
    kp[:, d] += FD_EPS[d]; km[:, d] -= FD_EPS[d]
    with torch.no_grad():
        sp = forward_parts(ego_inputs(sl, kp), B)["s_tot"].double().numpy(); sm = forward_parts(ego_inputs(sl, km), B)["s_tot"].double().numpy()
    fdg = (sp - sm) / (2 * FD_EPS[d])
    out_e.append({"coord": ["v0", "a0", "kappa"][d], "median_abs_diff_fd_minus_autograd": float(np.median(np.abs(fdg - GK["s_tot"][sl, d]))),
                  "median_abs_autograd": float(np.median(np.abs(GK["s_tot"][sl, d]))), "p90_abs_diff": float(np.quantile(np.abs(fdg - GK["s_tot"][sl, d]), 0.9))})
fd["ego_chain"] = out_e
ctrl["fd"] = fd
ctrl["s_res_base_grad_exactly_none_all"] = bool(NORM["ctrl.s_res.base_grad_is_None"].min() == 1.0)
ctrl["s_res_f8_pri_grad_exactly_none_all"] = bool(NORM["ctrl.s_res.f8_pri_grad_is_None"].min() == 1.0)
ctrl["s_net_vision_O_grad_exactly_none_all"] = bool(NORM["ctrl.s_net.u_o_grad_is_None"].min() == 1.0)
log("controls", json.dumps(ctrl)[:800])

# ------------------------------------------------------------------ values of the components
VAL = {s: np.zeros(N) for s in ("s_tot", "s_res", "s_net", "s_base")}
with torch.no_grad():
    for b0 in range(0, N, 64):
        sl = slice(b0, min(b0 + 64, N))
        o = forward_parts(base_inputs(sl), sl.stop - sl.start)
        for s in VAL:
            VAL[s][sl] = o[s].double().numpy()
log("component values done")

# ------------------------------------------------------------------ ablations (forward only)
rng = np.random.default_rng(0)
perm_band = np.arange(N)
for bi in range(len(BANDS)):
    idx = np.where(band_of == bi)[0]
    perm_band[idx] = rng.permutation(idx)
perm_glob = rng.permutation(N)
PERMS = {"shuffle_band": perm_band, "shuffle_global": perm_glob}
def run_ablation(group, mode):
    out = np.zeros(N)
    with torch.no_grad():
        for b0 in range(0, N, 64):
            sl = slice(b0, min(b0 + 64, N)); B = sl.stop - sl.start
            idx = torch.as_tensor(np.arange(sl.start, sl.stop)) if mode == "zero" else torch.as_tensor(PERMS[mode][sl])
            inp = base_inputs(sl)
            code_xf = ck_xf = None
            if group == "vision":
                feats = torch.zeros_like(BATCH["feats"][sl]).float() if mode == "zero" else BATCH["feats"][idx].float()
                inp = base_inputs(sl, u0=model.adapter[0](feats))
            elif group == "ego":
                k3 = np.zeros((B, 3)) if mode == "zero" else kin3[PERMS[mode][sl]]
                inp = ego_inputs(sl, k3)
            elif group == "nav":
                nv = torch.zeros(B, 5) if mode == "zero" else BATCH["nav"][idx, -1]
                inp["nav_st"] = nv; inp["nav_o"] = nv
                nh = inp["nav_hist"].clone(); nh[:, -1] = nv; inp["nav_hist"] = nh
            elif group == "limit":
                lm = torch.zeros(B, 4) if mode == "zero" else BATCH["limit"][idx]
                ok = torch.zeros(B, dtype=torch.bool) if mode == "zero" else BATCH["limit_ok"][idx]
                inp["lim_st"] = lm; inp["lim_o"] = lm; inp["limit_ok"] = ok
            o = forward_parts(inp, B, code_xf=code_xf, ck_xf=ck_xf)
            out[sl] = o["s_tot"].double().numpy()
    return out
# code / ck shuffles need the code of OTHER windows: compute the code table once
with torch.no_grad():
    CODE = torch.zeros(N, 60); CKIN = torch.zeros(N, 12)
    for b0 in range(0, N, 64):
        sl = slice(b0, min(b0 + 64, N))
        o = forward_parts(base_inputs(sl), sl.stop - sl.start)
        CODE[sl], CKIN[sl] = o["code"], o["ck_in"]
def run_ablation_oint(group, mode):
    out = np.zeros(N)
    tab = CODE if group == "code" else CKIN
    nominal = torch.zeros(12); nominal[9:] = 0.0
    with torch.no_grad():
        for b0 in range(0, N, 64):
            sl = slice(b0, min(b0 + 64, N)); B = sl.stop - sl.start
            if mode == "zero":
                repl = torch.zeros(B, tab.shape[1])
            else:
                repl = tab[torch.as_tensor(PERMS[mode][sl])]
            inp = base_inputs(sl)
            if group == "code":
                o = forward_parts(inp, B, code_xf=lambda c, r=repl: r)
            else:
                o = forward_parts(inp, B, ck_xf=lambda c, r=repl: r)
            out[sl] = o["s_tot"].double().numpy()
    return out
ABL = {}
for group in ("vision", "ego", "nav", "limit", "code", "ck"):
    for mode in ("zero", "shuffle_band", "shuffle_global"):
        t0 = time.time()
        ABL[f"{group}.{mode}"] = (run_ablation_oint if group in ("code", "ck") else run_ablation)(group, mode)
        log(f"ablation {group}.{mode} done ({time.time()-t0:.0f}s)  rms d s = {np.sqrt(np.mean((ABL[f'{group}.{mode}'] - VAL['s_tot'])**2)):.3f}")

# ------------------------------------------------------------------ save per-window arrays
np.savez_compressed(f"{HERE}/{OUTTAG}_PERWINDOW.npz", clip=clip_ids, v0=v0, band=band_of, kin3=kin3,
                    **{f"val.{k}": v for k, v in VAL.items()}, **{f"norm.{k}": v for k, v in NORM.items()}, **{f"gk.{s}": GK[s] for s in SC},
                    **{f"abl.{k}": v for k, v in ABL.items()})
json.dump({"ckpt": CKPT, "n": int(N), "controls": ctrl, "xs3": XS[:3].tolist(), "bands": BANDS, "wall_s": time.time() - t_start,
           "tree": TREE, "tanitad_file": tanitad.__file__}, open(f"{HERE}/{OUTTAG}_RUN.json", "w"), indent=1)
log("DONE")
