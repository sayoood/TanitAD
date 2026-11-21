"""LATVIS read on the trained states of SPATIAL stage 1 (PREREG_V8F_PROOF_ADD12 section 4; baseline read PREREG_V8F_PROOF_ADD10).
Tier T0: a world-model and perception diagnostic on logged frames, never driving performance. A thin driver over F:/TanitAD-scratch/v8proof/latvis/latvis.py
(imported, not edited). Reading choices and every difference from the baseline read: LATVIS_GW_DEFS.md (numbered RG1..).

Subcommands
  check-tree      sha256 of the two patched model files against the pod pins (LF-normalised), prints and writes tree_hashes.json
  prep            decode the grid frames into latvis_gw/frames64.u8 (CPU; resumable; uses the baseline prep code with other paths)
  collect         level-1 state of every arm on the eval-141 windows -> latvis_gw/latents/<ARM>.pt (CPU)
  run             the read for the arms given with --arm (GPU behind the shared lock), RESULT_LATVIS_GW.json, VERDICT_LATVIS_GW.md
  smoke           mechanics only: the 12 panel episodes, 2 folds, 1 epoch, CPU; shapes and finite checks (writes under smoke/)
--arm "NAME,CKPT_PATH,WIDTH,KIND"   KIND grid | mean; NAME ends in _s0 or _s1 (the seed); the sha256 file is <dir of CKPT>/ckpt_last.sha256
Environment: no PYTHONPATH needed (the driver puts the patched tree first); do not use python -I.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
ROOT = "F:/TanitAD-scratch/v8proof"
TREE = f"{HERE}/tree/stack"
BASE = f"{ROOT}/latvis"
PINS = {"v8f_mini.py": "687a41030e8dc193d9669866d15b37b46bc5d8f90b711db4328d4d479e3c1102",
        "v8f_beta.py": "d30fbdc30d9d20c360c93970b47121a2efaed5c829f18b6fa323410c94716dab"}
sys.path.insert(0, f"{HERE}/tree/taniteval")   # taniteval (floors_kin), copied from res/tree
sys.path.insert(0, TREE)                       # the patched tree FIRST: tanitad must come from here
if BASE not in sys.path:
    sys.path.append(BASE)
import latvis as LV  # noqa: E402  (baseline tool: pure helpers, decoders, fit, bootstrap; adds res/ wm/ full/ to sys.path)

RP, W = LV.RP, LV.W
FR_PATH = f"{HERE}/frames64.u8"
PREP_DONE = f"{HERE}/prep_done_gw.json"
LAT_DIR = f"{HERE}/latents"
RAW = ("E:/Projects/TanitAD-artifacts/sfresh_inputs/code/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw")
SF = "E:/Projects/TanitAD-artifacts/sfresh_inputs"
NEAR_BAR = 0.90          # card section 4 stage 1, Q1
PR_FLOOR = 32.0          # card control c4
KS, KEY = LV.KS, f"{LV.KS[-1] * 0.2:.1f}s"
log = LV.log


# ================================================================================================ tree and checkpoint hygiene
def tree_hashes(tree_dir=TREE):
    out = {}
    for f, pin in PINS.items():
        b = open(f"{tree_dir}/tanitad/models/{f}", "rb").read()
        raw, lf = hashlib.sha256(b).hexdigest(), hashlib.sha256(b.replace(b"\r\n", b"\n")).hexdigest()
        out[f] = {"pin": pin, "sha256_raw": raw, "sha256_lf": lf, "has_crlf": b"\r\n" in b, "matches_pin": bool(pin in (raw, lf)),
                  "matched_form": "raw" if raw == pin else "lf-normalised" if lf == pin else None}
    return out


def assert_tree():
    import tanitad
    tf = os.path.abspath(tanitad.__file__).replace("\\", "/").lower()
    assert tf.startswith(TREE.lower()), f"tanitad imported from {tf}, not from {TREE}"
    h = tree_hashes()
    bad = [f for f, v in h.items() if not v["matches_pin"]]
    assert not bad, f"patched tree files do not match the pod pins: {bad}"
    return h


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def check_sha(ckpt, sha_file=None):
    """The checkpoint's sha256 against its .sha256 file ('<hex>  ckpt_last.pt'). Raises ValueError on any mismatch or a missing file. Returns the hex."""
    sha_file = sha_file or os.path.join(os.path.dirname(ckpt), "ckpt_last.sha256")
    if not os.path.exists(ckpt) or not os.path.exists(sha_file):
        raise FileNotFoundError(f"missing {ckpt if not os.path.exists(ckpt) else sha_file}")
    want = open(sha_file, encoding="utf-8").read().split()[0].lower()
    got = sha256_file(ckpt)
    if got != want:
        raise ValueError(f"sha256 mismatch for {ckpt}: file says {want}, bytes give {got}")
    return got


def parse_arm(s):
    name, ckpt, width, kind = [x.strip() for x in s.split(",")]
    m = re.search(r"_s(\d+)$", name)
    assert m, f"arm name {name!r} must end in _s<seed>"
    assert kind in ("grid", "mean"), kind
    return {"name": name, "ckpt": ckpt.replace("\\", "/"), "width": int(width), "kind": kind, "seed": int(m.group(1)), "type": name.split("_s")[0]}


def load_arm_model(path, width, kind):
    """The arm's model on the CPU with STRICT key matching, and the asserts of state kind and width on the BUILT model."""
    import torch
    from tanitad.models import v8f_mini as M
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = M.V8FMiniConfig(**ck["cfg"])
    model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    pr = model.wm.probe_state(cfg.d)
    assert pr["z1_width"] == width, ("built z1 width", pr["z1_width"], "expected", width)
    assert pr["location_sensitive"] == (kind == "grid"), ("built state kind: location_sensitive", pr["location_sensitive"], "expected", kind)
    assert cfg.wm_state == kind and int(cfg.wm_l1_dim) == width, (cfg.wm_state, cfg.wm_l1_dim)
    assert int(model.wm.stack.cfg[0].latent_dim) == width, model.wm.stack.cfg[0].latent_dim
    assert [int(c.latent_dim) for c in model.wm.stack.cfg[1:]] == [128, 128]
    info = {"step": int(ck.get("step", -1)), "wm_state": cfg.wm_state, "wm_l1_dim": int(cfg.wm_l1_dim), "wm_l1_pred_width": int(cfg.wm_l1_pred_width),
            "wm_cell_dim": int(cfg.wm_cell_dim), "wm_dim": int(cfg.wm_dim), "argv_sha": ck.get("argv_sha"), "n_tok": int(cfg.n_tok), "built_probe": pr}
    return model, info


# ================================================================================================ pure helpers (CPU, unit-tested)
def to_input(arm_kind, X):
    """Arm state [n, D] -> decoder input. grid: cell-major flatten (cell * cell_dim + c, cell = r * 10 + c) -> [n, cell_dim, 4, 10]; mean: unchanged."""
    if arm_kind != "grid":
        return X
    n, D = X.shape
    cd = D // 40
    return X.reshape(n, 4, 10, cd).permute(0, 3, 1, 2).contiguous() if hasattr(X, "permute") else X.reshape(n, 4, 10, cd).transpose(0, 3, 1, 2).copy()


def decoder_kind(arm):
    return f"G{arm['width'] // 40}" if arm["kind"] == "grid" else f"Z{arm['width']}"


def make_decoder_gw(kind: str, widths=LV.WIDTHS):
    """The baseline decoder (latvis.make_decoder, unchanged) with ONLY its input layer and the two standardisation buffers re-shaped:
    'Z<n>' = Linear n -> 256x4x10 (baseline: n 128); 'G<c>' = 1x1 conv c -> 256 on the 4x10 grid (baseline: c 128). Everything after it is the baseline code."""
    import torch
    nn = torch.nn
    base = kind[0]
    n = int(kind[1:])
    dec = LV.make_decoder_orig(base, widths)
    if base == "Z":
        dec.inp = nn.Linear(n, widths[0] * 4 * 10)
        shp = (n,)
    else:
        dec.inp = nn.Conv2d(n, widths[0], 1)
        shp = (1, n, 1, 1)
    dec.register_buffer("in_mu", torch.zeros(shp))
    dec.register_buffer("in_sd", torch.ones(shp))
    return dec


def participation_ratio(X):
    """(sum lambda)^2 / sum lambda^2 of the covariance of X [n, D] (card control c4)."""
    X = np.asarray(X, np.float64)
    X = X - X.mean(0)
    s = np.linalg.svd(X, compute_uv=False) ** 2
    return float(s.sum() ** 2 / (s ** 2).sum())


def q1_name(skill_arm, skill_tg, diff_vs_hier_ci):
    """Card stage 1, Q1, one seed: Z-NEAR-G if skill_arm >= 0.90 x skill_tg (point comparison); else Z-ABOVE-MEAN if arm minus HIER is above 0 with the interval
    excluding 0 (paired); else Z-FLAT. Returns (name, ratio, near, above)."""
    ratio = float(skill_arm / skill_tg) if skill_tg > 0 else float("nan")
    near = bool(skill_tg > 0 and skill_arm >= NEAR_BAR * skill_tg)
    above = bool(diff_vs_hier_ci is not None and diff_vs_hier_ci[0] > 0)
    return ("Z-NEAR-G" if near else "Z-ABOVE-MEAN" if above else "Z-FLAT"), ratio, near, above


def both_seeds(names, mixed):
    """A name needs BOTH seeds: the common per-seed name, else `mixed` (a mixed pair reads the FLAT name, as the baseline R10 reads a mixed pair)."""
    s = set(names)
    return s.pop() if len(s) == 1 else mixed


def pixel_label(rows, pn="learned"):
    """Baseline R10 at 2.0 s over both seeds: any seed without positive headroom -> NO-HEADROOM; both seeds the same label -> it; else IMAGINE-FLAT.
    rows: per seed dict with 'headroom_b_minus_c.positive_excl_0' and [pn]['label']."""
    if not all(r["headroom_b_minus_c"]["positive_excl_0"] for r in rows):
        return "NO-HEADROOM"
    ls = {r[pn]["label"] for r in rows}
    return ls.pop() if len(ls) == 1 else "IMAGINE-FLAT"


def px_label_one(headroom_pos, hi_lo):
    """One seed: NO-HEADROOM, or the label of a - b with interval (lo, hi): IMAGINE-PIXELS if hi < 0, IMAGINE-BELOW-COPY if lo > 0, else IMAGINE-FLAT."""
    if not headroom_pos:
        return "NO-HEADROOM"
    lo, hi = hi_lo
    return "IMAGINE-PIXELS" if hi < 0 else "IMAGINE-BELOW-COPY" if lo > 0 else "IMAGINE-FLAT"


def attribution(effect_exists, cis, sign):
    """Card: an outcome is assigned to LOCATION only if G differs from W in the same direction; if W equals G it is assigned to WIDTH.
    cis: per seed the interval of the paired difference (G better than W is `sign` * value > 0). Returns LOCATION / WIDTH / MIXED / NO-EFFECT-TO-ATTRIBUTE."""
    if not effect_exists:
        return "NO-EFFECT-TO-ATTRIBUTE"
    if all(sign * c[0] > 0 if sign > 0 else c[1] < 0 for c in cis):
        return "LOCATION"
    if all(c[0] <= 0 <= c[1] for c in cis):
        return "WIDTH"
    return "MIXED"


# ================================================================================================ frames (prep, identity check)
def cmd_prep(only=None, path=FR_PATH, done=PREP_DONE, limit=0):
    return LV.cmd_prep(limit=limit, only=only, path=path, done_path=done)


def panel_png_tiles(png, n_rows=12):
    """Columns 0 and 1 (frame t, frame t + 2.0 s) of every row of a baseline panel sheet: PIL layout of LV.build_panels (lab_w 96, head_h 30, gap 3)."""
    from PIL import Image
    a = np.asarray(Image.open(png).convert("RGB"))
    lab_w, head_h, gap = 96, 30, 3
    out = []
    for r in range(n_rows):
        y = head_h + r * (LV.H + gap)
        out.append([a[y:y + LV.H, lab_w + c * (LV.WD + gap): lab_w + c * (LV.WD + gap) + LV.WD].transpose(2, 0, 1) for c in (0, 1)])
    return out


def check_frames(ctx, panel, fr_path=FR_PATH, png=f"{BASE}/panel/panel_s0.png"):
    """Exact equality of the rebuilt frames (window start, +2.0 s) of the 12 panel windows with the tiles of the baseline sheet (lossless PNG)."""
    clips, T, row0, N = LV._index()
    FR = np.memmap(fr_path, dtype=np.uint8, mode="r", shape=(N, 3, LV.H, LV.WD))
    tiles = panel_png_tiles(png, len(panel))
    bad = []
    for r, p in enumerate(panel):
        rw = int(ctx["rows_w"][p["window"]])
        for c, off in enumerate((0, 10)):
            if not np.array_equal(np.asarray(FR[rw + off]), tiles[r][c]):
                bad.append((r, c))
    return {"n_tiles": 2 * len(panel), "n_mismatch": len(bad), "mismatch": bad, "ok": not bad}


# ================================================================================================ collect: level-1 state of every arm on the eval-141 windows
def open_windows():
    """Copy of wmact_lib.open_windows (raw/.../wmact/wmact_lib.py:62-70, the reader that banked latents.pt) with its constants."""
    from tanitad.data import v8f_mini_data as D
    ej = D.load_join(f"{RAW}/join_eval141.npz")
    tracks = D.load_native_tracks(f"{SF}/inputs/native_tracks.pt")
    store = D.ClipStore(f"{SF}/cache", "eval", lru=32, grid="pool4x10", kin_mode="yaw", tracks=tracks)
    wins = D.eval_grid_windows(f"{RAW}/eval_grid_windows.json", ej)
    ds = D.V8FMiniWindows(store, ej, wins, hist=4, hmax=16, max_age_t=2, max_age_s=6)
    eid = np.array([str(ej["clip"][ci]) for ci, _, _, _ in wins])
    return ds, wins, eid


def collect_one_pass(models, idx, batch=64):
    """Copy of wmact_lib.collect_latents (same file, :87-98) restricted to the window indices ``idx``: z1 of the adapter tokens per model."""
    import torch
    ds, wins, eid = open_windows()
    N = len(wins)
    Z = {k: torch.zeros(N, 16, int(m.cfg.wm_l1_dim)) for k, m in models.items()}
    A, OK, T = torch.zeros(N, 16, 2), torch.zeros(N, 16, dtype=torch.bool), np.zeros(N, np.int64)
    t0 = time.time()
    with torch.no_grad():
        for b0 in range(0, len(idx), batch):
            ii = idx[b0:b0 + batch]
            items = [ds.item(ds.windows[i][0], ds.windows[i][1]) for i in ii]
            feats = torch.from_numpy(np.stack([it["feats"] for it in items]))
            A[ii] = torch.from_numpy(np.stack([it["act"] for it in items])).float()
            OK[ii] = torch.from_numpy(np.stack([it["frame_ok"] for it in items])).bool()
            T[ii] = np.array([it["t"] for it in items])
            for k, m in models.items():
                adapt = m.adapter(feats.float()) + m.pos[None, None]
                Z[k][ii] = m.wm.z1(adapt)
            if (b0 // batch) % 8 == 0:
                log("collect", b0 + len(ii), "of", len(idx), round(time.time() - t0), "s")
    return Z, A, OK, T, eid


def lat_path(arm_name, lat_dir=LAT_DIR):
    return f"{lat_dir}/{arm_name}.pt"


def cmd_collect(arms, shas, only=None, lat_dir=LAT_DIR):
    """Level-1 states of the arms for the baseline's ok windows (``only``: restrict to these episodes), saved per arm with the checkpoint sha; the window
    identity (eid, t, raw actions, frame_ok) is asserted equal to the banked latents.pt the baseline read used."""
    import torch
    os.makedirs(lat_dir, exist_ok=True)
    L = torch.load(W.LATENTS, weights_only=False)
    ok = L["ok"].all(1).numpy()
    eid_b = np.array([str(e) for e in L["eid"]])
    keep = ok.copy()
    if only is not None:
        keep &= np.isin(eid_b, list(only))
    idx = np.flatnonzero(keep)
    todo = []
    for a in arms:
        p = lat_path(a["name"], lat_dir)
        if os.path.exists(p):
            z = torch.load(p, weights_only=False)
            if z["ckpt_sha256"] == shas[a["name"]] and bool(z["filled"][idx].all()):
                log("collect: cached", a["name"])
                continue
        todo.append(a)
    if not todo:
        return
    models = {a["name"]: load_arm_model(a["ckpt"], a["width"], a["kind"])[0] for a in todo}
    Z, A, OK, T, eid = collect_one_pass(models, idx)
    assert (eid == eid_b).all(), "window order differs from the banked latents.pt"
    assert (T[idx] == np.asarray(L["t"]).astype(np.int64)[idx]).all(), "window t differs from the banked latents.pt"
    assert torch.equal(A[idx], L["A_raw"][idx].float()), "raw actions differ from the banked latents.pt"
    assert torch.equal(OK[idx], L["ok"][idx].bool()), "frame_ok differs from the banked latents.pt"
    filled = torch.zeros(len(eid), dtype=torch.bool)
    filled[torch.as_tensor(idx)] = True
    for a in todo:
        assert torch.isfinite(Z[a["name"]][idx]).all(), a["name"]
        torch.save({"Z1": Z[a["name"]], "filled": filled, "ckpt_sha256": shas[a["name"]], "arm": a["name"], "width": a["width"], "kind": a["kind"],
                    "tanitad_file": __import__("tanitad").__file__}, lat_path(a["name"], lat_dir))
        log("collect: saved", a["name"], tuple(Z[a["name"]].shape))


# ================================================================================================ context and predicted latents
def build_ctx_gw(arms, n_eps=0, n_folds=5, only=None, lat_dir=LAT_DIR, fr_path=FR_PATH):
    """latvis.build_ctx with TAGS = {} (no banked 128-wide latents: it builds windows, pool, folds, the token grid and the frames) and the arms' own states
    added as ctx['Zs'][arm] (pool frames) from the collected latents; the duplicate-frame check of the baseline (build_ctx) is repeated per arm."""
    import torch
    LV.FR_PATH = fr_path
    old, old_mm, old_lo = LV.TAGS, np.memmap, LV.LO_PATH
    LV.TAGS = {}

    class _ZeroLO:                         # the frozen-token feature file LO_r128.i8 feeds only the token-grid source, which this read takes from the baseline stage files
        def __getitem__(self, idx):
            return np.zeros((len(idx), 640, 128), np.int8)

    sentinel = "ZERO_LO_SENTINEL"
    if not os.path.exists(old_lo):
        LV.LO_PATH = sentinel
        np.memmap = lambda fn, *a_, **k_: _ZeroLO() if fn == sentinel else old_mm(fn, *a_, **k_)
    try:
        ctx = LV.build_ctx(n_eps, n_folds, only=only)
    finally:
        LV.TAGS, np.memmap, LV.LO_PATH = old, old_mm, old_lo
    ctx["G"] = ctx["G"][:0]               # never used here
    sel, prow, rows_w = ctx["sel"], ctx["prow"], ctx["rows_w"]
    NF = LV.NF
    all_rows = (rows_w[:, None] + np.arange(NF)[None]).ravel()
    wi, wj = np.repeat(np.arange(len(sel)), NF), np.tile(np.arange(NF), len(sel))
    _, first = np.unique(all_rows, return_index=True)
    pool_w, pool_j = wi[first], wj[first]
    ctx["Zs"], dup, ckshas = {}, {}, {}
    for a in arms:
        z = torch.load(lat_path(a["name"], lat_dir), weights_only=False)
        assert bool(z["filled"][sel].all()), f"{a['name']}: latents not collected for every used window"
        Z1 = z["Z1"][sel].float()
        ctx["Zs"][a["name"]] = Z1[torch.as_tensor(pool_w), torch.as_tensor(pool_j)]
        Zall = Z1.reshape(-1, Z1.shape[-1])
        dup[a["name"]] = float((Zall - ctx["Zs"][a["name"]][torch.as_tensor(np.searchsorted(prow, all_rows))]).abs().max())
        ckshas[a["name"]] = z["ckpt_sha256"]
    ctx["dup_by_arm"], ctx["lat_ckpt_sha"] = dup, ckshas
    return ctx


def predicted_latents_gw(ctx, arms, models):
    """Copy of latvis.predicted_latents (latvis.py:332-364) with the arms' models and states in place of the banked checkpoints; the baseline's assert that tanitad
    lies under wm/tree is replaced by the assert on the patched tree (assert_tree). Learned rollout = wmceil.level_pack at level 1, context 1, cells k in (4, 10)."""
    import torch
    L, sel = ctx["L"], ctx["sel"]
    out, chk = {}, {}
    old = W.KS
    for a in arms:
        s, wm = a["name"], models[a["name"]].wm
        Z1 = torch.load(lat_path(s, ctx.get("_lat_dir", LAT_DIR)), weights_only=False)["Z1"][sel].float()
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
            Xs = LV.WS.stack_features(z0, W.action_block(at, k), lp)
            p_s, _ = W.cross_fit(Xs, zt[:, k] - z0, ctx["eid_w"], ctx["w_fold"])
            o["learned"][k] = lp.astype(np.float32)
            o["stack"][k] = (z0 + p_s).astype(np.float32)
        out[s] = o
        save_latents_gw(ctx, a, pk, A_sc, ctx.get("_out_dir", HERE))
        log("predicted latents", s, "done")
    return out, chk


def save_latents_gw(ctx, arm, pk, A_sc, out_dir):
    """LATENTS_GW_<ARM>.npz (float16 states, per used window, window order of ctx): z0 start state, pred_2s / pred_0p8s learned rollout (the arrays the pixel read
    decodes), true_2s / true_0p8s true states at +2.0 s / +0.8 s, eid, g0 (grid step of the window start), t (= g0 + 15), w_fold, act_scaled [n, 15, 2]
    (WM-scaled actions), plus a .json with the checkpoint sha256, kind, width, counts and the sha256 of every array's bytes. No lead-gap number is computed here."""
    zt = pk["true"][1]
    arr = {"z0": zt[:, 0].astype(np.float16), "pred_2s": pk["pred"][(1, 10)].astype(np.float16), "true_2s": zt[:, 10].astype(np.float16),
           "pred_0p8s": pk["pred"][(1, 4)].astype(np.float16), "true_0p8s": zt[:, 4].astype(np.float16), "eid": np.asarray(ctx["eid_w"]).astype(str),
           "g0": np.asarray(ctx["g0"]).astype(np.int64), "t": (np.asarray(ctx["g0"]) + LV.NF - 1).astype(np.int64), "w_fold": np.asarray(ctx["w_fold"]).astype(np.int64),
           "act_scaled": A_sc.numpy().astype(np.float32)}
    p = f"{out_dir}/LATENTS_GW_{arm['name']}.npz"
    np.savez(p, **arr)
    meta = {"arm": arm["name"], "ckpt_sha256": ctx["lat_ckpt_sha"][arm["name"]], "state_kind": arm["kind"], "width": arm["width"], "n_windows": int(len(arr["eid"])),
            "n_episodes": int(len(set(arr["eid"]))), "dtype_states": "float16", "keys": sorted(arr), "tier": "T0",
            "npz_sha256": sha256_file(p), "array_sha256": {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k, v in arr.items()}}
    json.dump(meta, open(p + ".json", "w", encoding="utf-8"), indent=1)


def panel_from_baseline(ctx, path=f"{BASE}/PANEL_WINDOWS.json", strict_rule=True, copy_to=None):
    """The 12 windows of the baseline read (PANEL_WINDOWS.json, fixed before any decoder), mapped to this context's window indices. With strict_rule the rule of
    latvis.panel_windows is recomputed on this context and must give the same list (it is the full 139-episode context only)."""
    rec = json.load(open(path, encoding="utf-8"))["windows"]
    key = {(str(c), int(g)): i for i, (c, g) in enumerate(zip(ctx["eid_w"], ctx["g0"]))}
    panel = []
    for r in rec:
        i = key[(r["clip"], int(r["g0"]))]
        panel.append({"window": i, "group": r["group"], "value": r["value"], "clip": r["clip"], "g0": int(r["g0"])})
    if strict_rule:
        chk = LV.panel_windows(ctx, path=copy_to or path)
        assert [(p["clip"], p["g0"], p["group"]) for p in panel] == [(r["clip"], r["g0"], r["group"]) for r in chk], "panel rule disagrees with PANEL_WINDOWS.json"
    return panel


# ================================================================================================ stages (copy of latvis.run_stage with the arm sources; LV.run_stage:395-446)
def stage_file(out_dir, arm, var, f):
    return LV.stage_path(out_dir, arm, var, f)


def run_stage_gw(ctx, pred, panel, arm, var, f, device, out_dir, epochs):
    import torch
    name, kind = arm["name"], decoder_kind(arm)
    p = stage_file(out_dir, name, var, f)
    sha = ctx["lat_ckpt_sha"][name]
    if os.path.exists(p):
        d = np.load(p, allow_pickle=False)
        assert str(d["ckpt_sha256"]) == sha, f"stage {p} was made from another checkpoint"
        log("stage exists, skip", os.path.basename(p))
        return
    t0 = time.time()
    X0 = ctx["Zs"][name]
    X = to_input(arm["kind"], X0).to(device)
    IMG = ctx["IMG"].to(device)
    fold, epi = ctx["fold"], ctx["epi"]
    tr, te = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
    model, info = LV.fit_decoder(kind, X, IMG, tr, epi[tr], seed=LV.SEED + f, device=device, max_epochs=epochs, perm=(var == "perm"))
    mean_img = IMG[torch.as_tensor(tr, device=device)].float().mean(0) / 255.0
    te_t = torch.as_tensor(te, device=device)
    sse = LV.mse_in(model, X[te_t], IMG, te_t)
    base = torch.cat([((IMG[te_t[a:a + 256]].float() / 255.0 - mean_img) ** 2).mean(dim=(1, 2, 3)) for a in range(0, len(te_t), 256)]).double().cpu().numpy()
    rec = {"te": te, "sse": sse, "sse0": base, "epochs_run": info["epochs_run"], "best_val_l1": info["best_val_l1"], "n_train": info["n_train"], "ckpt_sha256": np.array(sha)}
    if var == "main":
        itr_t = torch.as_tensor(info["itr"], device=device)
        s_tr = LV.mse_in(model, X[itr_t], IMG, itr_t)
        b_tr = torch.cat([((IMG[itr_t[a:a + 256]].float() / 255.0 - mean_img) ** 2).mean(dim=(1, 2, 3)) for a in range(0, len(itr_t), 256)]).double().cpu().numpy()
        rec["train_sse_sum"], rec["train_sse0_sum"], rec["train_n"] = float(s_tr.sum()), float(b_tr.sum()), int(len(itr_t))
        wf = np.flatnonzero(ctx["w_fold"] == f)
        pan = {r["window"]: r for r in panel}
        pf = [w for w in wf if w in pan]
        w0 = torch.as_tensor(ctx["w_pool0"][wf], device=device)
        rec["wf"] = wf
        for k in KS:
            wk = torch.as_tensor(ctx["w_poolk"][k][wf], device=device)
            rec[f"e_b_{k}"] = LV.mse_in(model, X[w0], IMG, wk)                          # (b) decoded start state (copy)
            rec[f"e_c_{k}"] = LV.mse_in(model, X[wk], IMG, wk)                          # (c) decoded true future state (ceiling)
            for pn in ("learned", "stack"):
                Pk = to_input(arm["kind"], torch.as_tensor(pred[name][pn][k][wf])).to(device)
                rec[f"e_a_{pn}_{k}"] = LV.mse_in(model, Pk, IMG, wk)                    # (a) decoded predicted state
        for w in pf:
            j = int(np.flatnonzero(wf == w)[0])
            rec[f"pan{w}_Zt"] = LV.decode_u8(model, X[w0[j:j + 1]])[0]
            Pj = to_input(arm["kind"], torch.as_tensor(pred[name]["learned"][10][wf][j:j + 1])).to(device)
            rec[f"pan{w}_Zp"] = LV.decode_u8(model, Pj)[0]
            rec[f"pan{w}_Zc"] = LV.decode_u8(model, X[torch.as_tensor(ctx["w_poolk"][10][wf][j:j + 1], device=device)])[0]
    np.savez(p, **rec)
    log("stage", name, var, "fold", f, "epochs", info["epochs_run"], "n_train", info["n_train"], "s", round(time.time() - t0, 1))
    del model, X, IMG
    if device.type == "cuda":
        torch.cuda.empty_cache()


def run_core_gw(ctx, pred, panel, arms, device, out_dir, epochs):
    os.makedirs(f"{out_dir}/stages", exist_ok=True)
    for f in range(ctx["n_folds"]):
        for a in arms:
            run_stage_gw(ctx, pred, panel, a, "main", f, device, out_dir, epochs)
            run_stage_gw(ctx, pred, panel, a, "perm", f, device, out_dir, epochs)


def _device_gw(holder):
    """latvis._device with its own lock tag; the lock, the wait-for-idle rule and the 3,500 MiB bar are fullres_encode.take_lock (never forced)."""
    import torch
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if not os.environ.get("LATVIS_NOLOCK"):
            import fullres_encode as FE
            holder.append(FE.take_lock("v8proof-latvis-gw"))
            log("GPU lock taken")
        return torch.device("cuda")
    return torch.device("cpu")


# ================================================================================================ assembly and analysis
def load_arm_stages(ctx, arm, out_dir):
    n, nf = len(ctx["prow"]), ctx["n_folds"]
    R = {}
    for var in ("main", "perm"):
        sse, sse0 = np.full(n, np.nan), np.full(n, np.nan)
        meta = {"epochs": [], "tr_s": 0.0, "tr_b": 0.0, "tr_n": 0, "win": {}}
        for f in range(nf):
            d = np.load(stage_file(out_dir, arm["name"], var, f), allow_pickle=False)
            sse[d["te"]], sse0[d["te"]] = d["sse"], d["sse0"]
            meta["epochs"].append(int(d["epochs_run"]))
            if var == "main":
                meta["tr_s"] += float(d["train_sse_sum"]); meta["tr_b"] += float(d["train_sse0_sum"]); meta["tr_n"] += int(d["train_n"])
                for k in d.files:
                    if k.startswith("e_") or k == "wf":
                        meta["win"].setdefault(k, []).append(d[k])
        assert np.isfinite(sse).all() and np.isfinite(sse0).all(), (arm["name"], var)
        R[var] = {"sse": sse, "sse0": sse0, "meta": meta}
    return R


def load_baseline(ctx, base_dir=BASE):
    """The baseline read's stage files (read only): per pool frame SSE of HIER (Zs0, Zs1) and of the token grid (G), and the HIER per-window V2 arrays. Present only
    when every file exists; the frames and folds of this context equal the baseline's (checked through sse0)."""
    out = {}
    try:
        for src in LV.SOURCES:
            sse, sse0 = np.full(len(ctx["prow"]), np.nan), np.full(len(ctx["prow"]), np.nan)
            for f in range(ctx["n_folds"]):
                d = np.load(LV.stage_path(base_dir, src, "main", f), allow_pickle=False)
                sse[d["te"]], sse0[d["te"]] = d["sse"], d["sse0"]
            assert np.isfinite(sse).all(), src
            out[src] = {"sse": sse, "sse0": sse0}
    except (FileNotFoundError, AssertionError, IndexError, ValueError):
        return None
    return out


def analyse_gw(ctx, arms, R, BL, C, base_json):
    n_ep, epi, fold = len(ctx["uep"]), ctx["epi"], ctx["fold"]
    wepi, nw = ctx["w_epi"], len(ctx["w_epi"])
    cw = LV.ep_sums(np.ones(nw), wepi, n_ep)
    res = {"arms": {}, "pairing": {}}
    # ---- pairing check against the baseline stage files: the same frames and folds give the same per-frame training-mean-image error
    first = arms[0]["name"]
    paired = False
    if BL is not None:
        rel = float(np.max(np.abs(BL["G"]["sse0"] - R[first]["main"]["sse0"]) / np.maximum(np.abs(R[first]["main"]["sse0"]), 1e-12)))
        paired = bool(rel < 1e-4)
        res["pairing"] = {"baseline_stage_files": True, "max_rel_diff_sse0": rel, "paired": paired}
    else:
        res["pairing"] = {"baseline_stage_files": False, "paired": False}
    S, S0, Wv = {}, {}, {}
    for a in arms:
        n = a["name"]
        r = R[n]
        S[n], S0[n] = LV.ep_sums(r["main"]["sse"], epi, n_ep), LV.ep_sums(r["main"]["sse0"], epi, n_ep)
        sk = LV.skill_pooled(r["main"]["sse"], r["main"]["sse0"])
        lo, hi = LV.boot_ci(lambda C_: 1.0 - (C_ @ S[n]) / (C_ @ S0[n]), C)
        pf = [LV.skill_pooled(r["main"]["sse"][fold == f], r["main"]["sse0"][fold == f]) for f in range(ctx["n_folds"])]
        m = r["main"]["meta"]
        tr_sk = 1.0 - m["tr_s"] / m["tr_b"]
        pm = LV.skill_pooled(r["perm"]["sse"], r["perm"]["sse0"])
        ent = {"V1": {"heldout_skill": sk, "ci95": [lo, hi], "per_fold": pf, "train_skill": tr_sk, "epochs_per_fold": m["epochs"]},
               "controls": {"v1_permuted_skill": pm, "v1_bar_max": LV.V1_BAR, "v1_ok": bool(pm <= LV.V1_BAR), "v3_train_skill": tr_sk, "v3_heldout_skill": sk, "v3_ok": bool(tr_sk >= sk)},
               "c4_participation_ratio": participation_ratio(ctx["Zs"][n].numpy())}
        ent["c4_collapsed"] = bool(a["kind"] == "grid" and ent["c4_participation_ratio"] < PR_FLOOR)       # the card flags the grid arm; W (mean) is reported only
        # V2
        order = np.concatenate(m["win"]["wf"])
        inv = np.empty(nw, int)
        inv[order] = np.arange(nw)
        e = {}
        ent["V2"] = {}
        for k in KS:
            ek = {nn: np.concatenate(m["win"][f"e_{nn}_{k}"])[inv] for nn in ("b", "c", "a_learned", "a_stack")}
            e[k] = ek
            row = {"n_windows": nw, "mse_copy_b": float(ek["b"].mean()), "mse_ceiling_c": float(ek["c"].mean())}
            dh = ek["b"] - ek["c"]
            sh = LV.ep_sums(dh, wepi, n_ep)
            l2, h2 = LV.boot_ci(lambda C_: (C_ @ sh) / (C_ @ cw), C)
            row["headroom_b_minus_c"] = {"point": float(dh.mean()), "ci95": [l2, h2], "positive_excl_0": bool(l2 > 0)}
            for pn in ("learned", "stack"):
                da = ek[f"a_{pn}"] - ek["b"]
                sa = LV.ep_sums(da, wepi, n_ep)
                l3, h3 = LV.boot_ci(lambda C_: (C_ @ sa) / (C_ @ cw), C)
                row[pn] = {"mse_pred_a": float(ek[f"a_{pn}"].mean()), "a_minus_b": {"point": float(da.mean()), "ci95": [l3, h3]},
                           "label": "IMAGINE-PIXELS" if h3 < 0 else "IMAGINE-BELOW-COPY" if l3 > 0 else "IMAGINE-FLAT"}
            ent["V2"][f"{k * 0.2:.1f}s"] = row
        Wv[n] = e
        res["arms"][n] = ent
    # ---- token grid skill (baseline, same decoder recipe on the same frames) and HIER of the same seed
    tg_sk = None
    if BL is not None:
        tg = LV.skill_pooled(BL["G"]["sse"], BL["G"]["sse0"])
        S_tg = LV.ep_sums(BL["G"]["sse"], epi, n_ep)
        tg_ci = LV.boot_ci(lambda C_: 1.0 - (C_ @ S_tg) / (C_ @ LV.ep_sums(BL["G"]["sse0"], epi, n_ep)), C)
        tg_sk = {"skill": tg, "ci95": list(tg_ci), "source": "baseline stage files latvis/stages/G_main_f*.npz"}
    else:
        g = base_json["V1"]["skill"]["G"]
        tg_sk = {"skill": g["heldout_skill"], "ci95": g["ci95"], "source": "RESULT_LATVIS.json (stage files absent: side by side only)"}
    res["token_grid"] = tg_sk
    for a in arms:
        n, sd = a["name"], a["seed"]
        hier = f"Zs{sd}"
        ent = res["arms"][n]
        ent["token_grid_ratio"] = {"skill_arm": ent["V1"]["heldout_skill"], "skill_token_grid": tg_sk["skill"], "ratio": float(ent["V1"]["heldout_skill"] / tg_sk["skill"]) if tg_sk["skill"] > 0 else None,
                                   "bar": NEAR_BAR}
        if BL is not None and paired:
            S_tg_ = LV.ep_sums(BL["G"]["sse"], epi, n_ep)
            S0_ = S0[n]
            ent["token_grid_ratio"]["ratio_ci95_paired"] = LV.boot_ci(lambda C_: (1.0 - (C_ @ S[n]) / (C_ @ S0_)) / (1.0 - (C_ @ S_tg_) / (C_ @ S0_)), C)
            S_h = LV.ep_sums(BL[hier]["sse"], epi, n_ep)
            pt = float((S_h.sum() - S[n].sum()) / S0[n].sum())
            lo, hi = LV.boot_ci(lambda C_: ((C_ @ S_h) - (C_ @ S[n])) / (C_ @ S0[n]), C)
            ent["minus_baseline"] = {"baseline": f"{hier} (HIER of seed {sd}, 128 wide mean state)", "mode": "paired (same frames, folds, episodes; episode-cluster bootstrap, the same draws)",
                                     "skill_arm_minus_skill_baseline": pt, "ci95": [lo, hi], "separated_pos": bool(lo > 0)}
        else:
            b = base_json["V1"]["skill"][hier]
            ent["minus_baseline"] = {"baseline": f"{hier} (RESULT_LATVIS.json)", "mode": "side by side (baseline stage files absent or not frame-identical: no paired interval)",
                                     "skill_arm_minus_skill_baseline": float(ent["V1"]["heldout_skill"] - b["heldout_skill"]), "baseline_skill": b["heldout_skill"], "baseline_ci95": b["ci95"],
                                     "ci95": None, "separated_pos": False}
        ci_ = ent["minus_baseline"]["ci95"]
        nm, ratio, near, above = q1_name(ent["V1"]["heldout_skill"], tg_sk["skill"], ci_)
        ent["Q1_seed"] = {"name": nm, "ratio": ratio, "near_bar": near, "above_mean": above}
        key2 = KEY
        v2 = ent["V2"][key2]
        ent["Q2_pixels_seed"] = {"headroom_positive": v2["headroom_b_minus_c"]["positive_excl_0"],
                                 "learned": px_label_one(v2["headroom_b_minus_c"]["positive_excl_0"], v2["learned"]["a_minus_b"]["ci95"]),
                                 "stack": px_label_one(v2["headroom_b_minus_c"]["positive_excl_0"], v2["stack"]["a_minus_b"]["ci95"])}
    # ---- per arm type, both seeds
    types = sorted({a["type"] for a in arms})
    res["Q"] = {"Q1": {}, "Q2_pixels": {}, "attribution": {}}
    for t in types:
        ts = [a for a in arms if a["type"] == t]
        seeds = sorted(a["seed"] for a in ts)
        full = seeds == [0, 1]
        coll = [a["name"] for a in ts if res["arms"][a["name"]]["c4_collapsed"]]
        q1 = {"seeds_read": seeds, "per_seed": {a["name"]: res["arms"][a["name"]]["Q1_seed"]["name"] for a in ts}}
        q2 = {"seeds_read": seeds, "per_seed_learned": {a["name"]: res["arms"][a["name"]]["Q2_pixels_seed"]["learned"] for a in ts},
              "per_seed_stack": {a["name"]: res["arms"][a["name"]]["Q2_pixels_seed"]["stack"] for a in ts}}
        if not full:
            q1["name"] = q2["learned"] = q2["stack"] = "NOT-READ (both seeds needed; read: " + ",".join(str(s) for s in seeds) + ")"
        elif coll:
            q1["name"] = q2["learned"] = q2["stack"] = "NOT-QUOTED (c4 collapse flag on " + ",".join(coll) + ")"
        else:
            q1["name"] = both_seeds(q1["per_seed"].values(), "Z-FLAT")
            rows = [res["arms"][a["name"]]["V2"][KEY] for a in sorted(ts, key=lambda x: x["seed"])]
            q2["learned"], q2["stack"] = pixel_label(rows, "learned"), pixel_label(rows, "stack")
        res["Q"]["Q1"][t], res["Q"]["Q2_pixels"][t] = q1, q2
    # ---- attribution: G against W, paired on frames and windows
    if "G" in types and "W" in types:
        gs = {a["seed"]: a for a in arms if a["type"] == "G"}
        ws = {a["seed"]: a for a in arms if a["type"] == "W"}
        if sorted(gs) == [0, 1] and sorted(ws) == [0, 1]:
            cis1, cis2, det = [], [], {}
            for sd in (0, 1):
                g, w_ = gs[sd]["name"], ws[sd]["name"]
                pt = float((S[w_].sum() - S[g].sum()) / S0[g].sum())
                lo, hi = LV.boot_ci(lambda C_: ((C_ @ S[w_]) - (C_ @ S[g])) / (C_ @ S0[g]), C)
                cis1.append((lo, hi))
                dg = Wv[g][KS[-1]]["a_learned"] - Wv[g][KS[-1]]["b"]
                dw = Wv[w_][KS[-1]]["a_learned"] - Wv[w_][KS[-1]]["b"]
                dd = dg - dw
                sd_ = LV.ep_sums(dd, wepi, n_ep)
                l2, h2 = LV.boot_ci(lambda C_: (C_ @ sd_) / (C_ @ cw), C)
                cis2.append((l2, h2))
                det[f"seed{sd}"] = {"G_minus_W_skill": pt, "ci95": [lo, hi], "G_minus_W_a_minus_b_2.0s_learned": float(dd.mean()), "ci95_a_minus_b": [l2, h2]}
            q1g, q2g = res["Q"]["Q1"]["G"]["name"], res["Q"]["Q2_pixels"]["G"]["learned"]
            res["Q"]["attribution"] = {"detail": det,
                                       "Q1": attribution(q1g in ("Z-NEAR-G", "Z-ABOVE-MEAN"), cis1, +1),
                                       "Q2_pixels": attribution(q2g == "IMAGINE-PIXELS", cis2, -1),
                                       "rule": "LOCATION if the paired difference G vs W is in the effect direction with the interval excluding 0 at both seeds; WIDTH if the interval includes 0 at both seeds; else MIXED"}
        else:
            res["Q"]["attribution"] = {"Q1": "NOT-READ (G and W both seeds needed)", "Q2_pixels": "NOT-READ (G and W both seeds needed)"}
    res["controls_gate_ok"] = bool(all(e["controls"]["v1_ok"] and e["controls"]["v3_ok"] for e in res["arms"].values()))
    return res


def fmt(ci):
    return "n/a" if ci is None else f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"


def write_outputs(ctx, arms, res, out_dir, extra):
    c = ctx["counts"]
    res = dict(res)
    res["meta"] = {"card": "PREREG_V8F_PROOF_ADD12.md section 4 (stage 1) on the LATVIS read of PREREG_V8F_PROOF_ADD10.md",
                   "tier": "T0 (a world-model and perception diagnostic on logged frames, never driving performance)", "evidence_class": "MEASURED (latvis_gw.py over latvis.py)",
                   "counts": c, "folds": ctx["n_folds"], "B": 2000, "bootstrap_seed": LV.SEED, "dup_latent_max_abs_diff_by_arm": ctx["dup_by_arm"],
                   "true_state_equals_collected_z1_max_abs_diff": extra["chk"], "tree_hashes": extra["tree"], "checkpoints": extra["arms"],
                   "frames_identity_check": extra.get("frames"), "STRATEGIC": "PAUSED (PI decision queue item 25, 2026-09-19): not a pass, not omitted",
                   "families": "LONGITUDINAL, LATERAL, TACTICAL: not measured by this read"}
    json.dump(res, open(f"{out_dir}/RESULT_LATVIS_GW.json", "w", encoding="utf-8"), indent=1)
    tg = res["token_grid"]
    L = ["# VERDICT_LATVIS_GW: LATVIS on the trained states of SPATIAL stage 1 (PREREG_V8F_PROOF_ADD12 section 4)", "",
         "MEASURED (latvis_gw.py; RESULT_LATVIS_GW.json), tier T0: a world-model and perception diagnostic on logged frames, never driving performance.",
         "Metric families: LONGITUDINAL, LATERAL, TACTICAL not measured by this read; STRATEGIC: PAUSED (PI item 25), not a pass, not omitted.",
         f"Data: {c['windows_used']} windows of {c['episodes_used']} episodes, {c['pool_frames']} pool frames, {ctx['n_folds']} folds by episode, episode-cluster bootstrap B 2,000 seed 0 (the baseline's draws).", "",
         "## Controls first", ""]
    ok = res["controls_gate_ok"]
    L.append(f"**Gate (v1 permuted-input skill <= {LV.V1_BAR}, v3 train >= held-out, per arm): {'ALL OK' if ok else 'MISSED, the reads below are not admissible for the arm that missed'}.**")
    for a in arms:
        e = res["arms"][a["name"]]
        ct = e["controls"]
        L.append(f"* {a['name']}: v1 permuted {ct['v1_permuted_skill']:+.4f} ({'ok' if ct['v1_ok'] else 'MISSED'}); v3 train {ct['v3_train_skill']:+.4f} vs held-out {ct['v3_heldout_skill']:+.4f} ({'ok' if ct['v3_ok'] else 'MISSED'}); "
                 f"v2 headroom {KEY} {e['V2'][KEY]['headroom_b_minus_c']['point']:+.5f} {fmt(e['V2'][KEY]['headroom_b_minus_c']['ci95'])} ({'positive' if e['V2'][KEY]['headroom_b_minus_c']['positive_excl_0'] else 'not positive'}); "
                 f"c4 participation ratio {e['c4_participation_ratio']:.1f} ({'COLLAPSED, below 32: Q1 and Q2 not quoted' if e['c4_collapsed'] else 'ok' if a['kind'] == 'grid' else 'reported only, flag applies to the grid arm'})")
    fi = extra.get("frames")
    L.append(f"* frame identity (rebuilt frames of the 12 panel windows against the baseline sheet): {('%d of %d tiles equal' % (fi['n_tiles'] - fi['n_mismatch'], fi['n_tiles'])) if fi else 'not run'}.")
    L.append(f"* pairing with the baseline stage files: {'paired (max relative diff of the per-frame mean-image error %.2e)' % res['pairing']['max_rel_diff_sse0'] if res['pairing'].get('paired') else 'NOT paired: side by side with RESULT_LATVIS.json'}.")
    L += ["", "## V1 what is retained (skill = 1 - MSE / MSE of the training-fold mean image)", "",
          f"Token grid (baseline decoder on the 4x10 grid of the frozen tokens, 128 channels): skill {tg['skill']:+.4f} {fmt(tg['ci95'])} ({tg['source']}).", "",
          "| arm | held-out skill [95% CI] | ratio to the token grid (bar 0.90) | minus baseline of the same seed | per fold |", "|---|---|---|---|---|"]
    for a in arms:
        e = res["arms"][a["name"]]
        k, tr_, mb = e["V1"], e["token_grid_ratio"], e["minus_baseline"]
        L.append(f"| {a['name']} ({a['kind']}, {a['width']}) | {k['heldout_skill']:+.4f} {fmt(k['ci95'])} | {tr_['ratio']:.3f} | {mb['skill_arm_minus_skill_baseline']:+.4f} {fmt(mb['ci95'])} vs {mb['baseline']}, {mb['mode']} | " + ", ".join(f"{x:+.3f}" for x in k["per_fold"]) + " |")
    L += ["", "## V2 what is imagined (pixel MSE against the TRUE future frame; a = decoded predicted state, b = decoded start state, c = decoded true future state)", ""]
    for a in arms:
        for kk in ("0.8s", KEY):
            r = res["arms"][a["name"]]["V2"][kk]
            L.append(f"* {a['name']} {kk}: b {r['mse_copy_b']:.5f}, c {r['mse_ceiling_c']:.5f}, headroom {r['headroom_b_minus_c']['point']:+.5f} {fmt(r['headroom_b_minus_c']['ci95'])}; " +
                     "; ".join(f"{pn}: a {r[pn]['mse_pred_a']:.5f}, a - b {r[pn]['a_minus_b']['point']:+.5f} {fmt(r[pn]['a_minus_b']['ci95'])} ({r[pn]['label']})" for pn in ("learned", "stack")))
    L += ["", "## Outcome names (card section 4, stage 1; each needs BOTH seeds)", ""]
    for t in sorted(res["Q"]["Q1"]):
        q1, q2 = res["Q"]["Q1"][t], res["Q"]["Q2_pixels"][t]
        L.append(f"* **Arm {t}: Q1 {q1['name']}** (per seed {q1['per_seed']}); **Q2 pixels, learned rollout against the copy at 2.0 s: {q2['learned']}** (per seed {q2['per_seed_learned']}); stacked predictor {q2['stack']} (per seed {q2['per_seed_stack']}).")
    at = res["Q"].get("attribution") or {}
    if at:
        L.append(f"* **Attribution G against W (location against width): Q1 {at.get('Q1')}; Q2 pixels {at.get('Q2_pixels')}.**")
        for sd, d in (at.get("detail") or {}).items():
            L.append(f"  * {sd}: G minus W skill {d['G_minus_W_skill']:+.4f} {fmt(d['ci95'])}; G minus W in (a - b) at 2.0 s {d['G_minus_W_a_minus_b_2.0s_learned']:+.5f} {fmt(d['ci95_a_minus_b'])}")
    L += ["", "Not part of this read: the r-a / r-b lead-gap reads of the stage-1 card (lead gap R2, imagined lead), Q2 IMAGINE-LEAD / IMAGINE-EGO, Q3 and the four families at T1-kin; they have their own tools.",
          "Pictures: panel/panel_<arm>.png (12 windows of PANEL_WINDOWS.json; frame t, frame t + 2.0 s, decoded state(t), decoded predicted state(t + 2.0 s), decoded true state(t + 2.0 s)). Camera content: private, never landed.",
          "Limits (card): a decoder shows what is shallowly recoverable from the state; 139 episodes; one checkpoint per arm and seed; L1 pixel error rewards the mean image. Reading choices RG1.. in LATVIS_GW_DEFS.md."]
    open(f"{out_dir}/VERDICT_LATVIS_GW.md", "w", encoding="utf-8").write("\n".join(L) + "\n")


def build_panels_gw(ctx, arms, panel, out_dir, fr_path=FR_PATH):
    from PIL import Image, ImageDraw, ImageFont
    clips, T, row0, N = LV._index()
    FR = np.memmap(fr_path, dtype=np.uint8, mode="r", shape=(N, 3, LV.H, LV.WD))
    os.makedirs(f"{out_dir}/panel", exist_ok=True)
    open(f"{out_dir}/panel/PRIVATE_DO_NOT_LAND.txt", "w", encoding="utf-8").write("These panels hold camera content (frames of the eval episodes). They stay on the dev box and in the private page. Never stage, land or publish them to the public repo.\n")
    titles = ["frame t", "frame t + 2.0 s", "decoded state(t)", "decoded predicted\nstate(t + 2.0 s)", "decoded true\nstate(t + 2.0 s)"]
    try:
        font = ImageFont.load_default(size=11)
    except TypeError:
        font = ImageFont.load_default()
    lab_w, head_h, gap = 96, 30, 3
    wpx, hpx = lab_w + 5 * (LV.WD + gap), head_h + len(panel) * (LV.H + gap)
    for a in arms:
        sheet = Image.new("RGB", (wpx, hpx), (24, 24, 24))
        dr = ImageDraw.Draw(sheet)
        for c, t_ in enumerate(titles):
            dr.multiline_text((lab_w + c * (LV.WD + gap) + 2, 2), t_, fill=(235, 235, 235), font=font)
        for r, p in enumerate(panel):
            w, rw = p["window"], int(ctx["rows_w"][p["window"]])
            d = np.load(stage_file(out_dir, a["name"], "main", int(ctx["w_fold"][w])), allow_pickle=False)
            tiles = [FR[rw], FR[rw + 10], d[f"pan{w}_Zt"], d[f"pan{w}_Zp"], d[f"pan{w}_Zc"]]
            y = head_h + r * (LV.H + gap)
            lbl = (f"dv {p['value']:.1f} m/s" if p["group"] == "speed_change" else f"gap {p['value']:.1f} m") + f"\n{p['clip'][:8]} g{p['g0']}"
            dr.multiline_text((2, y + 8), lbl, fill=(235, 235, 235), font=font)
            for c, tl in enumerate(tiles):
                sheet.paste(Image.fromarray(np.ascontiguousarray(np.asarray(tl).transpose(1, 2, 0))), (lab_w + c * (LV.WD + gap), y))
        sheet.save(f"{out_dir}/panel/panel_{a['name']}.png")
        sheet.resize((600, int(round(hpx * 600 / wpx))), Image.LANCZOS).save(f"{out_dir}/panel/panel_{a['name']}_600.jpg", quality=80)


# ================================================================================================ CLI
LV.make_decoder_orig = LV.make_decoder
LV.make_decoder = make_decoder_gw           # fit_decoder (baseline, unchanged) calls the module-global make_decoder; kinds 'Z<n>' / 'G<c>' reach this one


def cmd_run(arms, smoke=False):
    import torch
    th = assert_tree()
    out_dir = f"{HERE}/smoke" if smoke else HERE
    os.makedirs(out_dir, exist_ok=True)
    lat_dir = f"{out_dir}/latents" if smoke else LAT_DIR
    fr_path = f"{out_dir}/frames64.u8" if smoke else FR_PATH
    shas = {a["name"]: check_sha(a["ckpt"]) for a in arms}
    log("sha256 ok", {k: v[:12] for k, v in shas.items()})
    n_folds, epochs = (2, 1) if smoke else (5, int(os.environ.get("LATVIS_GW_EPOCHS", LV.MAX_EPOCHS)))
    only = None
    if smoke:
        only = {r["clip"] for r in json.load(open(f"{BASE}/PANEL_WINDOWS.json", encoding="utf-8"))["windows"]}
        pfile = f"{out_dir}/prep_done_gw.json"
        cmd_prep(only=only, path=fr_path, done=pfile)
    cmd_collect(arms, shas, only=only, lat_dir=lat_dir)
    ctx = build_ctx_gw(arms, 0, n_folds, only=only, lat_dir=lat_dir, fr_path=fr_path)
    ctx["_lat_dir"] = lat_dir
    ctx["_out_dir"] = out_dir
    log("ctx", ctx["counts"], "dup latent diff", ctx["dup_by_arm"])
    panel = panel_from_baseline(ctx, strict_rule=not smoke, copy_to=f"{out_dir}/PANEL_WINDOWS.json")
    frames = check_frames(ctx, panel, fr_path)
    log("frame identity", frames)
    if not frames["ok"]:
        raise SystemExit("ZZ frames differ from the baseline sheet")
    models, arm_info = {}, {}
    for a in arms:
        models[a["name"]], built = load_arm_model(a["ckpt"], a["width"], a["kind"])
        arm_info[a["name"]] = dict(a, sha256=shas[a["name"]], built=built)
    pred, chk = predicted_latents_gw(ctx, arms, models)
    holder = []
    try:
        device = _device_gw(holder)
        if smoke:
            for a in arms:
                for pn in ("learned", "stack"):
                    for k in KS:
                        x = pred[a["name"]][pn][k]
                        print("pred", a["name"], pn, k, x.shape, "finite", bool(np.isfinite(x).all()))
                X = to_input(a["kind"], ctx["Zs"][a["name"]])
                m = make_decoder_gw(decoder_kind(a)).to(device)
                m.set_norm(X.to(device)[:64])
                o = m(X.to(device)[:4])
                print("decoder", decoder_kind(a), tuple(X.shape), "->", tuple(o.shape), "finite", bool(torch.isfinite(o).all()))
        run_core_gw(ctx, pred, panel, arms, device, out_dir, epochs)
    finally:
        if holder:
            import fullres_encode as FE
            FE.release(holder[0])
            log("GPU lock released")
    base_json = json.load(open(f"{BASE}/RESULT_LATVIS.json", encoding="utf-8"))
    R = {a["name"]: load_arm_stages(ctx, a, out_dir) for a in arms}
    BL = None if smoke else load_baseline(ctx)
    C = RP.boot_counts(len(ctx["uep"]), seed=LV.SEED)
    if smoke:
        for k, v in R.items():
            print("stage", k, "n", len(v["main"]["sse"]), "finite", bool(np.isfinite(v["main"]["sse"]).all()))
    res = analyse_gw(ctx, arms, R, BL, C, base_json)
    build_panels_gw(ctx, arms, panel, out_dir, fr_path)
    write_outputs(ctx, arms, res, out_dir, {"chk": chk, "tree": th, "arms": arm_info, "frames": frames})
    if smoke:
        print("panel files", sorted(os.listdir(f"{out_dir}/panel")), "outcomes", json.dumps(res["Q"])[:600])
        return 0
    return 0 if res["controls_gate_ok"] else 2


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check-tree", "prep", "collect", "run", "smoke"])
    ap.add_argument("--arm", action="append", default=[], help='"NAME,CKPT,WIDTH,KIND"')
    a = ap.parse_args()
    arms = [parse_arm(s) for s in a.arm]
    if a.cmd == "check-tree":
        h = assert_tree()
        json.dump(h, open(f"{HERE}/tree_hashes.json", "w", encoding="utf-8"), indent=1)
        print(json.dumps(h, indent=1))
    elif a.cmd == "prep":
        cmd_prep()
    elif a.cmd == "collect":
        assert_tree()
        cmd_collect(arms, {x["name"]: check_sha(x["ckpt"]) for x in arms})
    else:
        sys.exit(cmd_run(arms, smoke=(a.cmd == "smoke")))
