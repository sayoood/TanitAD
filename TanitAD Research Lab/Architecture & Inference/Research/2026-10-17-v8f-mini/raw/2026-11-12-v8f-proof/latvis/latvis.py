"""LATVIS (PREREG_V8F_PROOF_ADD10.md): pictures decoded from the world-model latent Z and from the 4x10 token grid G.
Tier T0: a world-model and perception diagnostic on logged frames, never driving performance. Only small decoders (probes) are fitted.
Subcommands (see LATVIS_DEFS.md for every implementation choice, numbered R1..):
    prep    decode the grid frames once -> latvis/frames64.u8  [N,3,64,160] uint8, resumable per episode, CPU
    run     decoders (GPU behind the shared lock), controls, V1, V2, panels -> RESULT_LATVIS.json, VERDICT_LATVIS.md, panel/
    smoke   mechanics only: 6 episodes, 2 folds, 1 epoch; shapes and finite checks, no statistic (needs the GPU)
Environment: PYTHONPATH must contain F:/TanitAD-scratch/v8proof/wm/tree/stack (wmstack/wmceil need tanitad for the checkpoints).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
ROOT = "F:/TanitAD-scratch/v8proof"
for _p in (f"{ROOT}/res", f"{ROOT}/wm", f"{ROOT}/full"):
    if _p not in sys.path:
        sys.path.append(_p)
import resprobe as RP  # noqa: E402  (make_folds, inner_split, boot_counts: numpy only)
import wmceil as W  # noqa: E402     (numpy only at import; load_model / level_pack / cross_fit / action_block)
import wmstack as WS  # noqa: E402    (stack_features)

ART = "E:/Projects/TanitAD-artifacts"
SRC = f"{ART}/v2ep-eval139-416x1024cyl"
FEAT = f"{ROOT}/full/feat"
INDEX = f"{FEAT}/index.npz"
LO_PATH = f"{FEAT}/LO_r128.i8"
BANK = f"{ROOT}/res/cache/frames.npz"
FR_PATH = f"{HERE}/frames64.u8"
PREP_DONE = f"{HERE}/prep_done.json"
H, WD = 64, 160
NF = 16                      # frames per window (WMACT window: hmax 16, frame j = grid step t - 15 + j)
KS = (4, 10)                 # 0.8 s and 2.0 s in 0.2 s grid steps
SEED = 0
MAX_EPOCHS, BATCH, PATIENCE, LR = 20, 128, 4, 2e-3
WIDTHS = (256, 192, 128, 64, 32)
SOURCES = ("Zs0", "Zs1", "G")
TAGS = {"Zs0": "M2b_s0", "Zs1": "M2b_s1"}
V1_BAR, SKILL_BAR_DIR = 0.02, "train>=heldout"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ================================================================================================ pure helpers (CPU, unit-tested)
def pool4x10(tok: np.ndarray) -> np.ndarray:
    """tok [n, 640, C] (16x40 row-major: token = r * 40 + c) -> [n, C, 4, 10], the mean over 4x4 token blocks."""
    n, T, C = tok.shape
    assert T == 640
    x = tok.astype(np.float32).reshape(n, 4, 4, 10, 4, C)       # (gr, br, gc, bc, ch) with r = gr*4+br, c = gc*4+bc
    return x.mean(axis=(2, 4)).transpose(0, 3, 1, 2).copy()


def skill_pooled(sse: np.ndarray, sse0: np.ndarray) -> float:
    """1 - MSE / MSE of the training-fold mean image, as a ratio of sums over frames."""
    return float(1.0 - np.sum(sse) / np.sum(sse0))


def crossfit(fold: np.ndarray, fit_fn, apply_fn):
    """Cross-fit: for every fold f, fit on the frames of the OTHER folds and apply to the frames of f only.
    fit_fn(train_idx) -> model; apply_fn(model, test_idx) -> anything. Returns {f: (test_idx, output)}."""
    out = {}
    for f in np.unique(fold):
        tr, te = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        out[int(f)] = (te, apply_fn(fit_fn(tr), te))
    return out


def boot_ci(stat_fn, C):
    b = stat_fn(C)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return float(lo), float(hi)


def select_panel(clip, g0, bank_v0, bank_v10, bank_gap, n_each=6):
    """The 12-window picture rule, a function of logged data only. Inputs are per-window arrays (clip id, start grid step, logged speed at the start,
    logged speed 10 steps (2.0 s) later, logged lead gap at the start; NaN = not available). Group A: the n_each windows with the largest
    |speed change over 2.0 s|, at most one per episode. Group B: the n_each windows with the smallest lead gap at the start, at most one per episode
    and never a window already in group A. Ties by (clip, g0). Returns [(window_index, group, value)]."""
    clip, g0 = np.asarray(clip).astype(str), np.asarray(g0)
    chosen, used_w = [], set()

    def pick(order, group, vals):
        seen = set()
        for i in order:
            if len(seen) >= n_each:
                break
            if clip[i] in seen or int(i) in used_w:
                continue
            seen.add(clip[i])
            used_w.add(int(i))
            chosen.append((int(i), group, float(vals[i])))

    dv = np.abs(np.asarray(bank_v10, float) - np.asarray(bank_v0, float))
    ok = np.flatnonzero(np.isfinite(dv))
    pick(sorted(ok, key=lambda i: (-dv[i], clip[i], int(g0[i]))), "speed_change", dv)
    gp = np.asarray(bank_gap, float)
    ok = np.flatnonzero(np.isfinite(gp))
    pick(sorted(ok, key=lambda i: (gp[i], clip[i], int(g0[i]))), "lead_gap", gp)
    return chosen


# ================================================================================================ decoders (torch)
def _torch():
    import torch
    return torch


def make_decoder(kind: str, widths=WIDTHS):
    torch = _torch()
    nn = torch.nn

    class Block(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.up = nn.Upsample(scale_factor=2, mode="nearest")
            self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
            self.gn = nn.GroupNorm(8, cout)
            self.act = nn.GELU()
            self.c2 = nn.Conv2d(cout, cout, 3, padding=1)

        def forward(self, x):
            return self.c2(self.act(self.gn(self.c1(self.up(x)))))

    class Dec(nn.Module):
        def __init__(self):
            super().__init__()
            self.kind, self.c0 = kind, widths[0]
            self.inp = nn.Linear(128, widths[0] * 4 * 10) if kind == "Z" else nn.Conv2d(128, widths[0], 1)
            self.blocks = nn.ModuleList([Block(widths[i], widths[i + 1]) for i in range(4)])
            self.out = nn.Conv2d(widths[4], 3, 3, padding=1)
            shp = (128,) if kind == "Z" else (1, 128, 1, 1)
            self.register_buffer("in_mu", torch.zeros(shp))
            self.register_buffer("in_sd", torch.ones(shp))

        def set_norm(self, X):
            if self.kind == "Z":
                mu, sd = X.mean(0), X.std(0)
            else:
                mu, sd = X.mean(dim=(0, 2, 3), keepdim=True), X.std(dim=(0, 2, 3), keepdim=True)
            self.in_mu.copy_(mu.reshape(self.in_mu.shape))
            self.in_sd.copy_(sd.reshape(self.in_sd.shape).clamp_min(1e-6))

        def forward(self, x):
            x = (x - self.in_mu) / self.in_sd
            h = self.inp(x)
            if self.kind == "Z":
                h = h.view(-1, self.c0, 4, 10)
            for b in self.blocks:
                h = b(h)
            return torch.sigmoid(self.out(h))

    return Dec()


def fit_decoder(kind, X, IMG, tr, ep_tr, seed, device, max_epochs=MAX_EPOCHS, bs=BATCH, patience=PATIENCE, lr=LR, widths=WIDTHS, perm=False):
    """One decoder on the training rows ``tr`` (indices into X / IMG). Early stopping on an inner split of the training EPISODES (20 %, seed 0).
    perm=True (control v1): the inputs are permuted across frames inside the training set (and inside the inner validation set), images stay.
    Returns (model with the best-epoch weights, info). Model is trained on the inner-train rows only; info['itr'] lists them."""
    torch = _torch()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 12345)
    iv = RP.inner_split(ep_tr, seed=SEED, frac=0.2)
    itr, iva = np.asarray(tr)[~iv], np.asarray(tr)[iv]
    src_tr = itr[rng.permutation(len(itr))] if perm else itr
    src_va = iva[rng.permutation(len(iva))] if perm else iva
    model = make_decoder(kind, widths).to(device)
    model.set_norm(X[torch.as_tensor(src_tr, device=device)])
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    itr_t, iva_t = torch.as_tensor(itr, device=device), torch.as_tensor(iva, device=device)
    str_t, sva_t = torch.as_tensor(src_tr, device=device), torch.as_tensor(src_va, device=device)

    def val_l1():
        model.eval()
        tot = 0.0
        with torch.no_grad():
            for a in range(0, len(iva_t), 256):
                tot += float((model(X[sva_t[a:a + 256]]) - IMG[iva_t[a:a + 256]].float() / 255.0).abs().mean(dim=(1, 2, 3)).sum())
        return tot / max(1, len(iva_t))

    best, best_sd, bad, ep_run = 1e30, copy.deepcopy(model.state_dict()), 0, 0
    g = torch.Generator(device="cpu").manual_seed(seed)
    for ep in range(max_epochs):
        model.train()
        order = torch.randperm(len(itr_t), generator=g).to(device)
        for a in range(0, len(order), bs):
            j = order[a:a + bs]
            loss = (model(X[str_t[j]]) - IMG[itr_t[j]].float() / 255.0).abs().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        v = val_l1()
        ep_run = ep + 1
        if v < best - 1e-7:
            best, bad, best_sd = v, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_sd)
    model.eval()
    return model, {"epochs_run": ep_run, "best_val_l1": float(best), "n_train": int(len(itr)), "n_val": int(len(iva)), "itr": itr}


def mse_in(model, Xin, IMG, tgt_idx, bs=256):
    """Per-image MSE (mean over 3x64x160, [0,1] scale) of decode(Xin[i]) against IMG[tgt_idx[i]]. Xin [m, ...] already on the device."""
    torch = _torch()
    out = []
    with torch.no_grad():
        for a in range(0, len(Xin), bs):
            p = model(Xin[a:a + bs])
            out.append(((p - IMG[tgt_idx[a:a + bs]].float() / 255.0) ** 2).mean(dim=(1, 2, 3)))
    return torch.cat(out).double().cpu().numpy() if out else np.zeros(0)


def decode_u8(model, Xin):
    torch = _torch()
    with torch.no_grad():
        return (model(Xin).clamp(0, 1) * 255.0).round().to(torch.uint8).cpu().numpy()


# ================================================================================================ frames (prep)
def _index():
    z = np.load(INDEX)
    return [str(c) for c in z["clips"]], z["T"].astype(np.int64), z["row0"].astype(np.int64), int(z["N"])


def cmd_prep(limit: int = 0, only=None, path: str = FR_PATH, done_path: str = PREP_DONE):
    import fullres_encode as FE
    torch = _torch()
    import torch.nn.functional as F
    clips, T, row0, N = _index()
    done = set(json.load(open(done_path, encoding="utf-8"))) if os.path.exists(done_path) and os.path.exists(path) else set()
    mm = np.memmap(path, dtype=np.uint8, mode="r+" if (done and os.path.exists(path)) else "w+", shape=(N, 3, H, WD))
    todo = [i for i, c in enumerate(clips) if c not in done and (only is None or c in only)]
    if limit:
        todo = todo[:limit]
    t0 = time.time()
    for n_i, i in enumerate(todo):
        o, offs = FE.load_ep(f"{SRC}/{clips[i]}.v2ep.pt")
        raw = list(range(0, len(o["jpeg_len"]), 2))
        assert len(raw) == T[i], (clips[i], len(raw), T[i])
        for a in range(0, len(raw), 24):
            x = FE.decode(o, offs, raw[a:a + 24])
            y = F.interpolate(x.float(), size=(H, WD), mode="area")
            mm[row0[i] + a: row0[i] + a + len(raw[a:a + 24])] = y.round().clamp(0, 255).to(torch.uint8).numpy()
        mm.flush()
        done.add(clips[i])
        tmp = done_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(sorted(done), fh)
        os.replace(tmp, done_path)
        if (n_i + 1) % 10 == 0 or n_i + 1 == len(todo):
            log("prep", n_i + 1, "of", len(todo), "episodes", round(time.time() - t0), "s")
    log("prep done", len(done), "of", len(clips), "episodes marked")
    return mm


# ================================================================================================ context: windows, pool, inputs
def build_ctx(n_eps: int = 0, n_folds: int = 5, light: bool = False, only=None):
    """Windows = the WMACT eval-141 windows with 16 real frames whose clip has frames here (139 clips). Pool = the unique (clip, grid step) frames
    covered by those windows (frame j of window w = grid step t - 15 + j); Z of a pool frame = the banked latent of that frame (identical across
    windows, checked). light=True skips the heavy inputs (finalize only needs indices and folds)."""
    torch = _torch()
    L = torch.load(W.LATENTS, weights_only=False)
    clips, T, row0, N = _index()
    cpos = {c: i for i, c in enumerate(clips)}
    ok = L["ok"].all(1).numpy()
    t = np.asarray(L["t"]).astype(np.int64)
    eid = np.array([str(e) for e in L["eid"]])
    n_ok = int(ok.sum())
    has = np.array([e in cpos for e in eid])
    keep = ok & has
    if only is not None:
        keep &= np.isin(eid, list(only))
    elif n_eps:
        use = sorted(set(eid[keep]))[:n_eps]
        keep &= np.isin(eid, use)
    sel = np.flatnonzero(keep)
    ci = np.array([cpos[e] for e in eid[sel]])
    g0 = t[sel] - (NF - 1)
    assert (g0 >= 0).all() and (g0 + NF - 1 < T[ci]).all()
    rows_w = row0[ci] + g0
    all_rows = (rows_w[:, None] + np.arange(NF)[None]).ravel()
    wi, wj = np.repeat(np.arange(len(sel)), NF), np.tile(np.arange(NF), len(sel))
    prow, first = np.unique(all_rows, return_index=True)
    pool_w, pool_j = wi[first], wj[first]
    pool_ci = np.searchsorted(row0, prow, side="right") - 1
    pool_clip = np.array([clips[i] for i in pool_ci])
    pool_g = prow - row0[pool_ci]
    fold = RP.make_folds(pool_clip, k=n_folds, seed=SEED)
    w_pool0 = np.searchsorted(prow, rows_w)
    w_poolk = {k: np.searchsorted(prow, rows_w + k) for k in KS}
    uep = sorted(set(pool_clip))
    epi = np.array([uep.index(c) for c in pool_clip])
    ctx = {"sel": sel, "eid_w": eid[sel], "g0": g0, "ci": ci, "rows_w": rows_w, "prow": prow, "pool_clip": pool_clip, "pool_g": pool_g, "fold": fold,
           "w_pool0": w_pool0, "w_poolk": w_poolk, "w_fold": fold[w_pool0], "uep": uep, "epi": epi, "w_epi": epi[w_pool0], "n_folds": n_folds,
           "counts": {"windows_eval141": int(len(ok)), "windows_16_real_frames": n_ok, "windows_dropped_no_frames_in_139": int((ok & ~has).sum()),
                      "windows_used": int(len(sel)), "episodes_used": int(len(uep)), "pool_frames": int(len(prow)), "grid_frames_total": N}, "L": L}
    if light:
        return ctx
    # pool inputs
    Zs = {}
    for s, tag in TAGS.items():
        Zp = L["Z1"][tag][sel][torch.as_tensor(pool_w), torch.as_tensor(pool_j)].float()
        Zs[s] = Zp
    # duplicate-frame check (the same frame in different windows gives the same latent): first window vs every other occurrence
    dup = 0.0
    for s, tag in TAGS.items():
        Zall = L["Z1"][tag][sel].float().reshape(-1, 128)
        dup = max(dup, float((Zall - Zs[s][torch.as_tensor(np.searchsorted(prow, all_rows))]).abs().max()))
    ctx["dup_latent_max_abs_diff"] = dup
    LO = np.memmap(LO_PATH, dtype=np.int8, mode="r", shape=(N, 640, 128))
    G = np.zeros((len(prow), 128, 4, 10), np.float32)
    for a in range(0, len(prow), 256):
        G[a:a + 256] = pool4x10(np.asarray(LO[prow[a:a + 256]]).astype(np.float32) * (6.0 / 127.0))
    FR = np.memmap(FR_PATH, dtype=np.uint8, mode="r", shape=(N, 3, H, WD))
    IMG = np.zeros((len(prow), 3, H, WD), np.uint8)
    for a in range(0, len(prow), 512):
        IMG[a:a + 512] = FR[prow[a:a + 512]]
    ctx.update({"Zs": Zs, "G": torch.from_numpy(G), "IMG": torch.from_numpy(IMG)})
    return ctx


def predicted_latents(ctx):
    """Level-1 predicted latents at k in KS (0.8 s, 2.0 s) for both seeds: 'learned' = HJEPAStack.rollout (wmceil.level_pack at level 1, context 1),
    'stack' = z0 + cross-fitted ridge on (z0, actions, learned - z0) as wmstack.py (here with the card-10 folds)."""
    torch = _torch()
    import tanitad
    assert os.path.abspath(tanitad.__file__).replace("\\", "/").lower().startswith(f"{ROOT}/wm/tree".lower()), tanitad.__file__
    L, sel = ctx["L"], ctx["sel"]
    out, chk = {}, {}
    old = W.KS
    for s, tag in TAGS.items():
        model, _ = W.load_model(W.CKPT[tag])
        wm = model.wm
        Z1 = L["Z1"][tag][sel].float()
        A_sc = wm.scale_actions(L["A_raw"][sel].float())[:, :-1]
        W.KS = {1: KS}
        try:
            pk = W.level_pack(wm, Z1, A_sc)
        finally:
            W.KS = old
        zt, at = pk["true"][1], pk["act"][1]
        chk[s] = float(np.abs(zt - Z1.double().numpy()).max())
        o = {"learned": {}, "stack": {}}
        for k in KS:
            z0, lp = zt[:, 0], pk["pred"][(1, k)]
            Xs = WS.stack_features(z0, W.action_block(at, k), lp)
            p_s, _ = W.cross_fit(Xs, zt[:, k] - z0, ctx["eid_w"], ctx["w_fold"])
            o["learned"][k] = lp.astype(np.float32)
            o["stack"][k] = (z0 + p_s).astype(np.float32)
        out[s] = o
        log("predicted latents", s, "done")
    return out, chk


def logged_panel_inputs(ctx):
    """Logged speed (res/cache/frames.npz E[:,0]) at the window start and 2.0 s later, and the logged lead gap at the start, per window (NaN if the
    (clip, step) is not in the bank or the value is not finite)."""
    z = np.load(BANK)
    key = {(str(b), int(g)): i for i, (b, g) in enumerate(zip(z["beid"], z["bg"]))}
    E0, gap = z["E"][:, 0], z["lead_gap"]

    def look(arr, clip, g):
        i = key.get((clip, int(g)))
        return float(arr[i]) if i is not None else np.nan

    clip = ctx["eid_w"]
    v0 = np.array([look(E0, c, g) for c, g in zip(clip, ctx["g0"])])
    v10 = np.array([look(E0, c, g + 10) for c, g in zip(clip, ctx["g0"])])
    gp = np.array([look(gap, c, g) for c, g in zip(clip, ctx["g0"])])
    return v0, v10, gp


def panel_windows(ctx, path=f"{HERE}/PANEL_WINDOWS.json"):
    """Fixed by rule from logged data; written BEFORE any decoder exists (run calls this first). An existing file must agree."""
    v0, v10, gp = logged_panel_inputs(ctx)
    ch = select_panel(ctx["eid_w"], ctx["g0"], v0, v10, gp)
    rec = [{"window": i, "group": grp, "value": val, "clip": str(ctx["eid_w"][i]), "g0": int(ctx["g0"][i]), "speed_start_mps": float(v0[i]),
            "speed_2s_mps": float(v10[i]), "lead_gap_start_m": float(gp[i])} for i, grp, val in ch]
    if os.path.exists(path):
        old = json.load(open(path, encoding="utf-8"))
        assert [(r["clip"], r["g0"], r["group"]) for r in old["windows"]] == [(r["clip"], r["g0"], r["group"]) for r in rec], "PANEL_WINDOWS.json disagrees with the rule"
        return rec
    json.dump({"rule": "6 windows with the largest |speed change over 2.0 s| (<= 1 per episode), then the 6 with the smallest lead gap at the start "
                       "(<= 1 per episode, not already chosen); logged speed and lead gap from res/cache/frames.npz; chosen before any decoder", "windows": rec,
               "n_windows_eligible_speed": int(np.isfinite(np.abs(v10 - v0)).sum()), "n_windows_eligible_gap": int(np.isfinite(gp).sum())},
              open(path, "w", encoding="utf-8"), indent=1)
    return rec


# ================================================================================================ stages
def stage_path(out_dir, src, var, f):
    return f"{out_dir}/stages/{src}_{var}_f{f}.npz"


def run_stage(ctx, pred, panel, src, var, f, device, out_dir, epochs, widths=WIDTHS):
    torch = _torch()
    p = stage_path(out_dir, src, var, f)
    if os.path.exists(p):
        log("stage exists, skip", os.path.basename(p))
        return
    t0 = time.time()
    kind = "G" if src == "G" else "Z"
    X = (ctx["G"] if src == "G" else ctx["Zs"][src]).to(device)
    IMG = ctx["IMG"].to(device)
    fold, epi = ctx["fold"], ctx["epi"]
    tr, te = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
    model, info = fit_decoder(kind, X, IMG, tr, epi[tr], seed=SEED + f, device=device, max_epochs=epochs, perm=(var == "perm"), widths=widths)
    mean_img = IMG[torch.as_tensor(tr, device=device)].float().mean(0) / 255.0
    te_t = torch.as_tensor(te, device=device)
    sse = mse_in(model, X[te_t], IMG, te_t)
    base = torch.cat([((IMG[te_t[a:a + 256]].float() / 255.0 - mean_img) ** 2).mean(dim=(1, 2, 3)) for a in range(0, len(te_t), 256)]).double().cpu().numpy()
    rec = {"te": te, "sse": sse, "sse0": base, "epochs_run": info["epochs_run"], "best_val_l1": info["best_val_l1"], "n_train": info["n_train"]}
    if var == "main":
        itr_t = torch.as_tensor(info["itr"], device=device)
        s_tr = mse_in(model, X[itr_t], IMG, itr_t)
        b_tr = torch.cat([((IMG[itr_t[a:a + 256]].float() / 255.0 - mean_img) ** 2).mean(dim=(1, 2, 3)) for a in range(0, len(itr_t), 256)]).double().cpu().numpy()
        rec["train_sse_sum"], rec["train_sse0_sum"], rec["train_n"] = float(s_tr.sum()), float(b_tr.sum()), int(len(itr_t))
        wf = np.flatnonzero(ctx["w_fold"] == f)
        pan = {r["window"]: r for r in panel}
        pf = [w for w in wf if w in pan]
        w0 = torch.as_tensor(ctx["w_pool0"][wf], device=device)
        rec["wf"] = wf
        if kind == "Z":
            Xz = X
            for k in KS:
                wk = torch.as_tensor(ctx["w_poolk"][k][wf], device=device)
                rec[f"e_b_{k}"] = mse_in(model, Xz[w0], IMG, wk)                       # (b) decoded start latent (copy)
                rec[f"e_c_{k}"] = mse_in(model, Xz[wk], IMG, wk)                       # (c) decoded true future latent (ceiling)
                for pn in ("learned", "stack"):
                    Pk = torch.as_tensor(pred[src][pn][k][wf], device=device)
                    rec[f"e_a_{pn}_{k}"] = mse_in(model, Pk, IMG, wk)                  # (a) decoded predicted latent
            for w in pf:
                j = int(np.flatnonzero(wf == w)[0])
                rec[f"pan{w}_Zt"] = decode_u8(model, Xz[w0[j:j + 1]])[0]
                rec[f"pan{w}_Zp"] = decode_u8(model, torch.as_tensor(pred[src]["learned"][10][wf][j:j + 1], device=device))[0]
                rec[f"pan{w}_Zc"] = decode_u8(model, Xz[torch.as_tensor(ctx["w_poolk"][10][wf][j:j + 1], device=device)])[0]
        else:
            rec["e_start_frame_g"] = mse_in(model, X[w0], IMG, w0)
            for w in pf:
                j = int(np.flatnonzero(wf == w)[0])
                rec[f"pan{w}_G"] = decode_u8(model, X[w0[j:j + 1]])[0]
    np.savez(p, **rec)
    log("stage", src, var, "fold", f, "epochs", info["epochs_run"], "n_train", info["n_train"], "s", round(time.time() - t0, 1))
    del model, X, IMG
    torch.cuda.empty_cache() if device.type == "cuda" else None


def run_core(ctx, pred, panel, device, out_dir, epochs, smoke=False):
    os.makedirs(f"{out_dir}/stages", exist_ok=True)
    for f in range(ctx["n_folds"]):
        for src in SOURCES:
            run_stage(ctx, pred, panel, src, "main", f, device, out_dir, epochs)
            run_stage(ctx, pred, panel, src, "perm", f, device, out_dir, epochs)


# ================================================================================================ assembly, outcomes, outputs
def load_stages(ctx, out_dir):
    n, nf = len(ctx["prow"]), ctx["n_folds"]
    R = {}
    for src in SOURCES:
        for var in ("main", "perm"):
            sse, sse0 = np.full(n, np.nan), np.full(n, np.nan)
            meta = {"epochs": [], "tr_s": 0.0, "tr_b": 0.0, "tr_n": 0, "win": {}}
            for f in range(nf):
                d = np.load(stage_path(out_dir, src, var, f), allow_pickle=False)
                sse[d["te"]], sse0[d["te"]] = d["sse"], d["sse0"]
                meta["epochs"].append(int(d["epochs_run"]))
                if var == "main":
                    meta["tr_s"] += float(d["train_sse_sum"]); meta["tr_b"] += float(d["train_sse0_sum"]); meta["tr_n"] += int(d["train_n"])
                    if src != "G":
                        for k in d.files:
                            if k.startswith("e_") or k == "wf":
                                meta["win"].setdefault(k, []).append(d[k])
            assert np.isfinite(sse).all() and np.isfinite(sse0).all(), (src, var)
            R[(src, var)] = {"sse": sse, "sse0": sse0, "meta": meta}
    return R


def ep_sums(v, epi, n_ep):
    s = np.zeros(n_ep)
    np.add.at(s, epi, np.asarray(v, np.float64))
    return s


def analyse(ctx, R, C):
    n_ep = len(ctx["uep"])
    epi, fold = ctx["epi"], ctx["fold"]
    res = {"V1": {"skill": {}, "controls": {}}, "V2": {}, "controls": {}}
    S, S0 = {}, {}
    for src in SOURCES:
        S[src], S0[src] = ep_sums(R[(src, "main")]["sse"], epi, n_ep), ep_sums(R[(src, "main")]["sse0"], epi, n_ep)
        sk = skill_pooled(R[(src, "main")]["sse"], R[(src, "main")]["sse0"])
        lo, hi = boot_ci(lambda C_: 1.0 - (C_ @ S[src]) / (C_ @ S0[src]), C)
        per_fold = [skill_pooled(R[(src, "main")]["sse"][fold == f], R[(src, "main")]["sse0"][fold == f]) for f in range(ctx["n_folds"])]
        m = R[(src, "main")]["meta"]
        tr_sk = 1.0 - m["tr_s"] / m["tr_b"]
        pm = skill_pooled(R[(src, "perm")]["sse"], R[(src, "perm")]["sse0"])
        res["V1"]["skill"][src] = {"heldout_skill": sk, "ci95": [lo, hi], "per_fold": per_fold, "train_skill": tr_sk, "epochs_per_fold": m["epochs"]}
        res["V1"]["controls"][src] = {"permuted_skill": pm, "bar_max": V1_BAR, "ok": bool(pm <= V1_BAR)}
        res["controls"][f"v3_{src}"] = {"train_skill": tr_sk, "heldout_skill": sk, "ok": bool(tr_sk >= sk)}
    gz = {}
    for s in ("Zs0", "Zs1"):
        pt = (np.sum(S[s]) - np.sum(S["G"])) / np.sum(S0["G"])
        lo, hi = boot_ci(lambda C_: ((C_ @ S[s]) - (C_ @ S["G"])) / (C_ @ S0["G"]), C)
        gz[s] = {"G_minus_Z": float(pt), "ci95": [lo, hi], "separated_pos": bool(lo > 0)}
    res["V1"]["G_minus_Z"] = gz
    res["V1"]["outcome"] = "Z-BELOW-G" if all(v["separated_pos"] for v in gz.values()) else "Z-AT-G"
    # V2
    wepi = ctx["w_epi"]
    nw = len(wepi)
    order = np.concatenate(R[("Zs0", "main")]["meta"]["win"]["wf"])
    inv = np.empty(nw, int)
    inv[order] = np.arange(nw)
    cw = ep_sums(np.ones(nw), wepi, n_ep)
    lab = {}
    for s in ("Zs0", "Zs1"):
        W_ = R[(s, "main")]["meta"]["win"]
        assert (np.concatenate(W_["wf"]) == order).all()
        res["V2"][s] = {}
        for k in KS:
            e = {n: np.concatenate(W_[f"e_{n}_{k}"])[inv] for n in ("b", "c", "a_learned", "a_stack")}
            row = {"n_windows": nw, "mse_copy_b": float(e["b"].mean()), "mse_ceiling_c": float(e["c"].mean())}
            dh = e["b"] - e["c"]
            sh = ep_sums(dh, wepi, n_ep)
            lo, hi = boot_ci(lambda C_: (C_ @ sh) / (C_ @ cw), C)
            row["headroom_b_minus_c"] = {"point": float(dh.mean()), "ci95": [lo, hi], "positive_excl_0": bool(lo > 0)}
            for pn in ("learned", "stack"):
                da = e[f"a_{pn}"] - e["b"]
                sa = ep_sums(da, wepi, n_ep)
                lo2, hi2 = boot_ci(lambda C_: (C_ @ sa) / (C_ @ cw), C)
                row[pn] = {"mse_pred_a": float(e[f"a_{pn}"].mean()), "a_minus_b": {"point": float(da.mean()), "ci95": [lo2, hi2]},
                           "label": "IMAGINE-PIXELS" if hi2 < 0 else "IMAGINE-BELOW-COPY" if lo2 > 0 else "IMAGINE-FLAT"}
            res["V2"][s][f"{k * 0.2:.1f}s"] = row
    key = f"{KS[-1] * 0.2:.1f}s"
    for pn in ("learned", "stack"):
        rows = [res["V2"][s][key] for s in ("Zs0", "Zs1")]
        if not all(r["headroom_b_minus_c"]["positive_excl_0"] for r in rows):
            lab[pn] = "NO-HEADROOM"
        else:
            ls = {r[pn]["label"] for r in rows}
            lab[pn] = ls.pop() if len(ls) == 1 else "IMAGINE-FLAT"
    res["V2"]["outcome_2.0s"] = lab
    res["controls"]["v1"] = {s: res["V1"]["controls"][s]["ok"] for s in SOURCES}
    res["controls"]["v2_headroom_2.0s"] = {s: res["V2"][s][key]["headroom_b_minus_c"]["positive_excl_0"] for s in ("Zs0", "Zs1")}
    res["controls"]["gate_ok"] = bool(all(res["controls"]["v1"].values()) and all(res["controls"][f"v3_{s}"]["ok"] for s in SOURCES))
    return res


def build_panels(ctx, panel, out_dir):
    from PIL import Image, ImageDraw, ImageFont
    clips, T, row0, N = _index()
    FR = np.memmap(FR_PATH, dtype=np.uint8, mode="r", shape=(N, 3, H, WD))
    os.makedirs(f"{out_dir}/panel", exist_ok=True)
    with open(f"{out_dir}/panel/PRIVATE_DO_NOT_LAND.txt", "w", encoding="utf-8") as fh:
        fh.write("These panels hold camera content (frames of the eval episodes). They stay on the dev box and in the private page. "
                 "Never stage, land or publish them to the public repo.\n")
    titles = ["frame t", "frame t + 2.0 s", "decoded G(t)", "decoded Z(t)", "decoded predicted\nZ(t + 2.0 s)", "decoded true\nZ(t + 2.0 s)"]
    try:
        font = ImageFont.load_default(size=11)
    except TypeError:
        font = ImageFont.load_default()
    lab_w, head_h, gap = 96, 30, 3
    wpx, hpx = lab_w + 6 * (WD + gap), head_h + len(panel) * (H + gap)
    for s in ("Zs0", "Zs1"):
        sheet = Image.new("RGB", (wpx, hpx), (24, 24, 24))
        dr = ImageDraw.Draw(sheet)
        for c, t_ in enumerate(titles):
            dr.multiline_text((lab_w + c * (WD + gap) + 2, 2), t_, fill=(235, 235, 235), font=font)
        for r, p in enumerate(panel):
            w = p["window"]
            f = int(ctx["w_fold"][w])
            dz = np.load(stage_path(out_dir, s, "main", f), allow_pickle=False)
            dg = np.load(stage_path(out_dir, "G", "main", f), allow_pickle=False)
            ci = int(np.searchsorted(row0, ctx["rows_w"][w], side="right") - 1)
            rw = int(ctx["rows_w"][w])
            tiles = [FR[rw], FR[rw + 10], dg[f"pan{w}_G"], dz[f"pan{w}_Zt"], dz[f"pan{w}_Zp"], dz[f"pan{w}_Zc"]]
            y = head_h + r * (H + gap)
            lbl = (f"dv {p['value']:.1f} m/s" if p["group"] == "speed_change" else f"gap {p['value']:.1f} m") + f"\n{p['clip'][:8]} g{p['g0']}"
            dr.multiline_text((2, y + 8), lbl, fill=(235, 235, 235), font=font)
            for c, tl in enumerate(tiles):
                sheet.paste(Image.fromarray(np.ascontiguousarray(np.asarray(tl).transpose(1, 2, 0))), (lab_w + c * (WD + gap), y))
        sheet.save(f"{out_dir}/panel/panel_{s.lower().replace('zs', 's')}.png")
        small = sheet.resize((600, int(round(hpx * 600 / wpx))), Image.LANCZOS)
        small.save(f"{out_dir}/panel/panel_{s.lower().replace('zs', 's')}_600.jpg", quality=80)


def fmt(ci):
    return f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"


def write_outputs(ctx, res, out_dir, chk, dup):
    c = ctx["counts"]
    res = dict(res)
    res["meta"] = {"card": "PREREG_V8F_PROOF_ADD10.md", "tier": "T0 (a world-model and perception diagnostic on logged frames, never driving performance)",
                   "evidence_class": "MEASURED (latvis.py)", "counts": c, "folds": ctx["n_folds"], "B": 2000, "bootstrap_seed": SEED,
                   "true_state_equals_banked_z1_max_abs_diff": chk, "duplicate_frame_latent_max_abs_diff": dup,
                   "STRATEGIC": "PAUSED (PI decision queue item 25, 2026-09-19): not a pass, not omitted"}
    json.dump(res, open(f"{out_dir}/RESULT_LATVIS.json", "w", encoding="utf-8"), indent=1)
    v1, v2, ct = res["V1"], res["V2"], res["controls"]
    L = ["# VERDICT_LATVIS: PREREG_V8F_PROOF_ADD10 (pictures decoded from the world-model latent and the token grid)", "",
         "MEASURED (latvis.py; RESULT_LATVIS.json), tier T0: a world-model and perception diagnostic on logged frames, never driving performance.",
         "Metric families: this read measures none of LONGITUDINAL, LATERAL or TACTICAL directly. STRATEGIC: PAUSED (PI decision queue item 25), not a pass, not omitted.",
         f"Data: {c['windows_used']} windows of {c['episodes_used']} episodes ({c['windows_dropped_no_frames_in_139']} windows dropped for lack of frames), {c['pool_frames']} frames in the pool, {res['meta']['folds']} folds by episode, episode-cluster bootstrap B 2,000 seed 0.", "",
         "## Controls first", ""]
    gate = ct["gate_ok"]
    L.append(f"**Gate (v1 permuted-input skill <= {V1_BAR} and v3 train >= held-out): {'ALL OK' if gate else 'MISSED, the reads below are not admissible'}.**")
    for s in SOURCES:
        L.append(f"* {s}: v1 permuted held-out skill {v1['controls'][s]['permuted_skill']:+.4f} (bar <= {V1_BAR}, {'ok' if v1['controls'][s]['ok'] else 'MISSED'}); "
                 f"v3 train-frame skill {ct['v3_' + s]['train_skill']:+.4f} vs held-out {ct['v3_' + s]['heldout_skill']:+.4f} ({'ok' if ct['v3_' + s]['ok'] else 'MISSED'}).")
    key = "2.0s"
    L.append("* v2 (ceiling real, 2.0 s): " + "; ".join(f"{s} headroom {v2[s][key]['headroom_b_minus_c']['point']:+.5f} {fmt(v2[s][key]['headroom_b_minus_c']['ci95'])} "
                                                         f"({'positive' if v2[s][key]['headroom_b_minus_c']['positive_excl_0'] else 'not positive'})" for s in ("Zs0", "Zs1")) +
             ". A missing headroom gives NO-HEADROOM below (not a control miss).")
    L += ["", "## V1 what is retained (skill = 1 - MSE / MSE of the training-fold mean image)", "", "| source | held-out skill [95% CI] | per fold |", "|---|---|---|"]
    for s in SOURCES:
        k = v1["skill"][s]
        L.append(f"| {s} | {k['heldout_skill']:+.4f} {fmt(k['ci95'])} | " + ", ".join(f"{x:+.3f}" for x in k["per_fold"]) + " |")
    L += ["", "G minus Z (paired, episode-cluster bootstrap): " + "; ".join(f"{s} {v1['G_minus_Z'][s]['G_minus_Z']:+.4f} {fmt(v1['G_minus_Z'][s]['ci95'])}" for s in ("Zs0", "Zs1")),
          f"**V1 outcome: {v1['outcome']}**", "", "## V2 what is imagined (pixel MSE against the TRUE future frame; a = decoded predicted latent, b = decoded start latent, c = decoded true future latent)", ""]
    for s in ("Zs0", "Zs1"):
        for kk in ("0.8s", "2.0s"):
            r = v2[s][kk]
            L.append(f"* {s} {kk}: b {r['mse_copy_b']:.5f}, c {r['mse_ceiling_c']:.5f}, headroom b - c {r['headroom_b_minus_c']['point']:+.5f} {fmt(r['headroom_b_minus_c']['ci95'])}; " +
                     "; ".join(f"{pn}: a {r[pn]['mse_pred_a']:.5f}, a - b {r[pn]['a_minus_b']['point']:+.5f} {fmt(r[pn]['a_minus_b']['ci95'])} ({r[pn]['label']})" for pn in ("learned", "stack")))
    L += ["", f"**V2 outcome at 2.0 s (both seeds): learned rollout {v2['outcome_2.0s']['learned']}; stacked predictor {v2['outcome_2.0s']['stack']}**", "",
          "Pictures: panel/panel_s0.png and panel/panel_s1.png (12 windows fixed in PANEL_WINDOWS.json before any decoder; columns frame t, frame t + 2.0 s, decoded G(t), decoded Z(t), decoded predicted Z(t + 2.0 s), decoded true Z(t + 2.0 s)). Camera content: private, never landed.", "",
          "Next lever committed by the card: Z-BELOW-G with IMAGINE-FLAT or NO-HEADROOM means the lever is a spatial world-model state (token grid, not its mean), with this read as its acceptance test (pod, launch gate); IMAGINE-PIXELS means decoded rollouts become the visual debug channel and decoders are fitted for levels 2 and 3.",
          "Limits (card): a decoder shows what is shallowly recoverable from 128 numbers; blur is expected from a mean-pooled state; 139 episodes; one checkpoint per seed; L1 pixel error rewards the mean image, so skill is read against that floor. Implementation choices R1.. are in LATVIS_DEFS.md."]
    open(f"{out_dir}/VERDICT_LATVIS.md", "w", encoding="utf-8").write("\n".join(L) + "\n")


def finalize(ctx, out_dir, panel, chk, dup):
    R = load_stages(ctx, out_dir)
    C = RP.boot_counts(len(ctx["uep"]), seed=SEED)
    res = analyse(ctx, R, C)
    build_panels(ctx, panel, out_dir)
    write_outputs(ctx, res, out_dir, chk, dup)
    return res


# ================================================================================================ CLI
def _device(lock_holder):
    torch = _torch()
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if not os.environ.get("LATVIS_NOLOCK"):
            import fullres_encode as FE
            lock_holder.append(FE.take_lock("v8proof-latvis"))
            log("GPU lock taken")
        return torch.device("cuda")
    return torch.device("cpu")


def cmd_run(smoke: bool = False):
    torch = _torch()
    out_dir = f"{HERE}/smoke" if smoke else HERE
    os.makedirs(out_dir, exist_ok=True)
    n_eps, n_folds, epochs = (6, 2, 1) if smoke else (0, 5, int(os.environ.get("LATVIS_EPOCHS", MAX_EPOCHS)))
    holder = []
    try:
        if smoke:
            ctx0 = build_ctx(n_eps, n_folds, light=True)
            cmd_prep(only=set(ctx0["uep"]))
        ctx = build_ctx(n_eps, n_folds)
        log("ctx", ctx["counts"], "dup latent diff", ctx["dup_latent_max_abs_diff"])
        panel = panel_windows(ctx, f"{out_dir}/PANEL_WINDOWS.json")                   # before any decoder exists
        pred, chk = predicted_latents(ctx)
        device = _device(holder)
        if smoke:
            for s in TAGS:
                for pn in ("learned", "stack"):
                    for k in KS:
                        a = pred[s][pn][k]
                        print("pred", s, pn, k, a.shape, "finite", bool(np.isfinite(a).all()))
            print("Z", {s: tuple(v.shape) for s, v in ctx["Zs"].items()}, "G", tuple(ctx["G"].shape), "IMG", tuple(ctx["IMG"].shape), "panel", len(panel))
            for kind, X in (("Z", ctx["Zs"]["Zs0"]), ("G", ctx["G"])):
                m = make_decoder(kind).to(device)
                m.set_norm(X.to(device)[:64])
                o = m(X.to(device)[:4])
                print("decoder", kind, tuple(o.shape), "range", float(o.min()), float(o.max()), "finite", bool(torch.isfinite(o).all()))
        run_core(ctx, pred, panel, device, out_dir, epochs, smoke)
        if smoke:
            R = load_stages(ctx, out_dir)
            for k, v in R.items():
                print("stage", k, "n", len(v["sse"]), "finite", bool(np.isfinite(v["sse"]).all()))
            build_panels(ctx, panel, out_dir)
            print("panel files", sorted(os.listdir(f"{out_dir}/panel")))
            return 0
        res = finalize(ctx, out_dir, panel, chk, ctx["dup_latent_max_abs_diff"])
        return 0 if res["controls"]["gate_ok"] else 2
    finally:
        if holder:
            import fullres_encode as FE
            FE.release(holder[0])
            log("GPU lock released")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "run", "smoke"])
    ap.add_argument("--limit", type=int, default=0, help="prep: at most this many episodes (test)")
    a = ap.parse_args()
    if a.cmd == "prep":
        cmd_prep(limit=a.limit)
    else:
        sys.exit(cmd_run(smoke=(a.cmd == "smoke")))
