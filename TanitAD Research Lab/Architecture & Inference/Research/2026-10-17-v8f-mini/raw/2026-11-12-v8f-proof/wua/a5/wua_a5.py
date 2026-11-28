"""wua_a5.py -- card ADD155 section 5 (A5: use attribution by masking), as frozen in IMPL_A5.md (sha256 beside it). CPU, zero training, tier T0.

  python -I wua_a5.py prep                       # windows, cell masks, cell means      -> C:/Users/Admin/wua_work/a5/SEL.pkl, cellmean.npy
  python -I wua_a5.py run --seed K               # planned-speed masking + c10 probe + c11 for one seed -> RESULT_A5_s<K>.json (+ _arrays.npz)
  python -I wua_a5.py combine                    # label across the three seeds         -> RESULT_A5.json
  python -I wua_a5.py selftest                   # synthetic geometry / mask / bootstrap checks
"""
import argparse, hashlib, json, lzma, math, os, pickle, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.dont_write_bytecode = True
import numpy as np

WUA = "F:/TanitAD-scratch/v8proof/wua"
OUT = "C:/Users/Admin/wua_work/a5"
TREE = "F:/TanitAD-scratch/v8proof/rhacc10k/tree/stack"
PINS = "F:/TanitAD-scratch/v8proof/rhacc10k/pins"
CACHE = "E:/Projects/TanitAD-artifacts/sfresh_inputs/cache"
TRACKS = "E:/Projects/TanitAD-artifacts/sfresh_inputs/inputs/native_tracks.pt"
AGENTS = "E:/Projects/TanitAD-artifacts/b1-agent-join-3d-20260917/b1eval_agents_3d.jsonl.xz"
EXTR = "C:/Users/Admin/tanitad-data/add203/out/refcv8_train4572_eval139_extrinsics.json"
LAB = "C:/Users/Admin/wua_work/labels/WUA_LABELS.npz"

IMG_W, IMG_H, NR, NC, CELL = 640, 256, 4, 10, 64
FREF = (IMG_W / 2) / math.radians(60.0)            # 305.577: column linear in azimuth, 120 deg across 640 px (TRAPS.md)
DT = 0.2
HMAX = 16
B_BOOT = 2000
Z_MIN = 0.1


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ geometry (pure numpy)
def rot_c2v(q):
    """FrontWideExtrinsics.rotation_cam_to_vehicle (pinned tree physicalai.py:258), q = dict qx qy qz qw."""
    n = math.sqrt(q["qx"] ** 2 + q["qy"] ** 2 + q["qz"] ** 2 + q["qw"] ** 2) or 1.0
    qx, qy, qz, qw = q["qx"] / n, q["qy"] / n, q["qz"] / n, q["qw"] / n
    return np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                     [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                     [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])


def corners_rig(a):
    c, s = math.cos(a["yaw"]), math.sin(a["yaw"])
    out = []
    for sx in (-0.5, 0.5):
        for sy in (-0.5, 0.5):
            for sz in (-0.5, 0.5):
                out.append([a["cx"] + c * sx * a["l"] - s * sy * a["w"], a["cy"] + s * sx * a["l"] + c * sy * a["w"], a["cz"] + sz * a["h"]])
    return np.array(out)


def px_of_rig(p_rig, R, t):
    """rig points [N,3] -> (x_px, y_px, z_cam) of the 256 x 640 cylindrical frame: p_cam = R^T (p - t); azimuth = atan2(-x, z) LEFT positive,
    column = 320 - f_ref * az; elevation = atan2(-y, hypot(x, z)), row = 128 - f_ref * tan(el) (cylinder rows are linear in tan(el))."""
    p = (np.asarray(p_rig, np.float64) - np.asarray(t, np.float64)) @ np.asarray(R, np.float64)
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    az = np.arctan2(-x, z)
    tn = -y / np.maximum(np.hypot(x, z), 1e-12)
    return IMG_W / 2 - FREF * az, IMG_H / 2 - FREF * tn, z


def rect_of(a, R, t):
    """bounding rectangle (x0, x1, y0, y1) in px clipped to the image, or None when a corner is behind the camera (z <= Z_MIN)."""
    xp, yp, z = px_of_rig(corners_rig(a), R, t)
    if (z <= Z_MIN).any():
        return None
    return (max(float(xp.min()), 0.0), min(float(xp.max()), float(IMG_W)), max(float(yp.min()), 0.0), min(float(yp.max()), float(IMG_H)))


def cells_of_rect(rc):
    """cells (row, col) whose 64 x 64 px rectangle intersects rc with positive area."""
    x0, x1, y0, y1 = rc
    cols = [c for c in range(NC) if min(x1, CELL * (c + 1)) - max(x0, CELL * c) > 1e-9]
    rows = [r for r in range(NR) if min(y1, CELL * (r + 1)) - max(y0, CELL * r) > 1e-9]
    return {(r, c) for r in rows for c in cols}


def pick_lead(agents, xmax=40.0):
    """x > 0, |y| <= 2.0, x <= xmax, smallest range hypot(x, y); returns the index or None."""
    best, bi = None, None
    for i, a in enumerate(agents):
        if a["cx"] > 0 and abs(a["cy"]) <= 2.0 and a["cx"] <= xmax:
            r = math.hypot(a["cx"], a["cy"])
            if best is None or r < best:
                best, bi = r, i
    return bi


def control_block(lead_cells, occupied):
    """mirror about the centre column (c -> 9 - c, same rows); shift the block outward column by column until it shares no cell with the LEAD cells
    nor with `occupied`; None when it leaves columns 0..9 first (IMPL_A5 section 3)."""
    mc = float(np.mean([c for _, c in lead_cells]))
    d = 1 if mc <= 4.5 else -1
    base = {(r, NC - 1 - c) for r, c in lead_cells}
    for s in range(0, NC + 1):
        blk = {(r, c + d * s) for r, c in base}
        if any(c < 0 or c >= NC for _, c in blk):
            return None
        if not (blk & lead_cells) and not (blk & occupied):
            return blk
    return None


def frame_cells(agents, R, t, require_x40=True):
    """-> (lead_cells, ctrl_cells | None, reason)."""
    li = pick_lead(agents, 40.0 if require_x40 else 40.0)
    if li is None:
        return None, None, "no_corridor_agent"
    rc = rect_of(agents[li], R, t)
    if rc is None:
        return None, None, "corners_behind"
    cells = cells_of_rect(rc)
    if not cells:
        return None, None, "no_cells"
    occ = set()
    for i, a in enumerate(agents):
        if i == li or a["cx"] <= 0 or math.hypot(a["cx"], a["cy"]) > 50.0:
            continue
        r2 = rect_of(a, R, t)
        if r2 is not None:
            occ |= cells_of_rect(r2)
    ctrl = control_block(cells, occ)
    if ctrl is None:
        return cells, None, "no_clean_control"
    return cells, ctrl, "ok"


def mask_of(cells):
    m = np.zeros(NR * NC, bool)
    for r, c in cells:
        m[r * NC + c] = True
    return m


# ------------------------------------------------------------------------------------------------ masking / read / statistics
def apply_mask(feats, mask, fill):
    """feats [B,F,40,1024] fp16 torch, mask [B,F,40] bool torch, fill [40,1024] fp16 torch -> a new tensor; an all-False mask returns the same values bit for bit."""
    import torch
    return torch.where(mask[..., None], fill[None, None], feats)


def speed2s(traj):
    """planned speed at 2.0 s: |0.5 (P[10] - P[8])| / DT (the Catmull-Rom tangent at knot P[9] = 2.0 s, v8f_t1_eval.hermite_frames:97-112)."""
    P = traj.double()
    return (0.5 * (P[:, 10] - P[:, 8])).norm(dim=1).numpy() / DT


def boot_weights(n_ep, B=B_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    W = np.zeros((B, n_ep))
    for b in range(B):
        W[b] = np.bincount(rng.choice(n_ep, size=n_ep, replace=True), minlength=n_ep)
    return W


def boot_mean(v, cid, W, sub=None):
    """mean of v over windows, episode-cluster bootstrap: ratio (W @ sum_v) / (W @ n) over per-clip sums. cid in 0..n_ep-1."""
    v = np.asarray(v, np.float64)
    n_ep = W.shape[1]
    keep = np.ones(len(v), bool) if sub is None else np.asarray(sub, bool)
    s, n = np.zeros(n_ep), np.zeros(n_ep)
    np.add.at(s, cid[keep], v[keep])
    np.add.at(n, cid[keep], 1.0)
    p = float(s.sum() / n.sum())
    b = (W @ s) / np.maximum(W @ n, 1e-12)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return {"point": p, "lo": float(lo), "hi": float(hi), "separated": bool(lo > 0 or hi < 0), "n_windows": int(keep.sum()), "n_clips": int((n > 0).sum())}


# ------------------------------------------------------------------------------------------------ data side (needs the pinned tree)
def open_data():
    sys.path.insert(0, TREE)
    sys.path.insert(0, TREE.replace('/stack', '/taniteval'))      # the pinned tree's taniteval (floors_kin, used by ClipStore)
    import torch
    from tanitad.data import v8f_mini_data as D
    J = D.load_join(f"{PINS}/join_eval141.npz")
    tracks = D.load_native_tracks(TRACKS)
    store = D.ClipStore(CACHE, "eval", lru=16, grid="pool4x10", kin_mode="yaw", tracks=tracks)
    wins = D.eval_grid_windows(f"{PINS}/eval_grid_windows.json", J)
    return D, J, store, wins


def cmd_prep(a):
    import torch
    D, J, store, wins = open_data()
    os.makedirs(OUT, exist_ok=True)
    ext = json.load(open(EXTR))
    AG = {}
    with lzma.open(AGENTS, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            AG[(r["clip_id"], int(r["frame"]))] = r["agents"]
    print("agents loaded", len(AG), flush=True)
    cm_p = f"{OUT}/cellmean.npy"
    if not os.path.exists(cm_p):
        S = np.zeros((NR * NC, 1024), np.float64); n = 0
        for j, c in enumerate(J["clip"].tolist()):
            f = store.get(str(c))["feats"]
            S += np.asarray(f, np.float64).sum(0); n += f.shape[0]
            if (j + 1) % 20 == 0:
                print("cellmean", j + 1, flush=True)
        np.save(cm_p, (S / n).astype(np.float16))
        json.dump({"n_frames": n, "sha256": sha(cm_p)}, open(cm_p + ".json", "w"))
    excl = {}
    kept = []
    cache_fc = {}
    Rt = {}
    for ci, t, _, _ in wins:
        clip = str(J["clip"][ci])
        row = int(J["off"][ci]) + t
        if clip not in ext:
            excl["no_extrinsics"] = excl.get("no_extrinsics", 0) + 1; continue
        if clip not in Rt:
            q = ext[clip]
            Rt[clip] = (rot_c2v(q), np.array([q["x"], q["y"], q["z"]]))
        R, tc = Rt[clip]
        if not (J["lead_valid"][row] == 1 and np.isfinite(J["lead_gap"][row])):
            excl["join_no_lead"] = excl.get("join_no_lead", 0) + 1; continue

        def fc(g):
            k = (clip, g)
            if k not in cache_fc:
                ag = AG.get((clip, 2 * g))
                cache_fc[k] = ("no_agent_line",) if ag is None else frame_cells(ag, R, tc)
            return cache_fc[k]
        cur = fc(t)
        if cur[0] == "no_agent_line":
            excl["no_agent_line"] = excl.get("no_agent_line", 0) + 1; continue
        if cur[2] != "ok":
            excl[cur[2]] = excl.get(cur[2], 0) + 1; continue
        ml = np.zeros((HMAX, NR * NC), bool); mc = np.zeros((HMAX, NR * NC), bool)
        n_own = 0
        for i, g in enumerate(range(t - HMAX + 1, t + 1)):
            r = fc(g) if g >= 0 else ("no_agent_line",)
            if r[0] != "no_agent_line" and r[2] == "ok":
                ml[i], mc[i] = mask_of(r[0]), mask_of(r[1]); n_own += 1
            else:
                ml[i], mc[i] = mask_of(cur[0]), mask_of(cur[1])
        kept.append({"ci": ci, "t": t, "clip": clip, "gap": float(J["lead_gap"][row]), "ml": ml, "mc": mc, "n_own_frames": n_own,
                     "n_lead_cells": int(ml[-1].sum())})
    info = {"n_grid_windows": len(wins), "n_kept": len(kept), "n_clips": len({k["clip"] for k in kept}), "excluded": excl,
            "mean_own_frames_of_10": float(np.mean([k["n_own_frames"] for k in kept])), "mean_lead_cells": float(np.mean([k["n_lead_cells"] for k in kept])),
            "gap_le_20": int(sum(k["gap"] <= 20 for k in kept)), "impl_sha256": sha(f"{WUA}/a5/IMPL_A5.md"), "tool_sha256": sha(os.path.abspath(__file__))}
    pickle.dump({"kept": kept, "info": info}, open(f"{OUT}/SEL.pkl", "wb"))
    json.dump(info, open(f"{OUT}/SEL_info.json", "w"), indent=1)
    print(json.dumps(info, indent=1))


def probe_fit(G, y, ci, folds_clip, f, Gtest_list):
    """P-LIN of card section 1 on TOK-GRID for ONE outer fold: PCA(256) on the training rows, z-score, logistic with C from the inner 3-group log-loss
    (wua_a1.oof_logit's rule), decision value on each array of Gtest_list."""
    sys.path.insert(0, WUA)
    import wua_a1 as A1
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    fo = folds_clip[ci]
    tr = fo != f
    p = PCA(n_components=256, svd_solver="randomized", random_state=0).fit(G[tr])
    Xtr = p.transform(G[tr]).astype(np.float64)
    ytr, cit = y[tr], ci[tr]
    g = A1.inner_groups(cit); gi = np.array([g[c] for c in cit])
    best, bc = np.inf, 1.0
    for al in A1.ALPHAS:
        ll = 0.0
        for k in range(3):
            a_, b_ = gi != k, gi == k
            if len(np.unique(ytr[a_])) < 2:
                continue
            A_, B_ = A1.zs(Xtr[a_], Xtr[b_])
            m = LogisticRegression(C=1.0 / al, max_iter=2000).fit(A_, ytr[a_])
            pr = np.clip(m.predict_proba(B_), 1e-9, 1)
            idx = np.searchsorted(m.classes_, ytr[b_])
            ll -= np.log(pr[np.arange(len(idx)), idx]).sum()
        if ll < best:
            best, bc = ll, 1.0 / al
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    m = LogisticRegression(C=bc, max_iter=2000).fit((Xtr - mu) / sd, ytr)
    return [m.decision_function((p.transform(Z).astype(np.float64) - mu) / sd) for Z in Gtest_list], bc


def cmd_run(a):
    import torch
    torch.set_num_threads(a.threads)
    sys.path.insert(0, WUA)
    import wua_extract as X
    sel = pickle.load(open(f"{OUT}/SEL.pkl", "rb"))
    kept = sel["kept"]
    D, J, store, _ = open_data()
    model, cksha = X.load(a.seed)
    cfg = model.cfg
    ds = D.V8FMiniWindows(store, J, [], hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s, window_start=cfg.window_start)
    assert cfg.hmax == HMAX
    keys = sorted(set(model_keys()))
    fill_mean = torch.from_numpy(np.load(f"{OUT}/cellmean.npy"))
    fill_zero = torch.zeros_like(fill_mean)
    N = len(kept)
    if a.limit:
        kept = kept[:a.limit]; N = len(kept)
    var = ("base", "lead_mean", "ctrl_mean", "lead_zero", "ctrl_zero", "empty_mean")
    V = {k: np.zeros(N) for k in var}
    bit_equal_empty = True
    t0 = time.time()
    for b0 in range(0, N, a.batch):
        sub = kept[b0:b0 + a.batch]
        items = [ds.item(k["ci"], k["t"], 0, 0) for k in sub]
        full = D.collate(items)
        batch = {k: full[k] for k in keys}
        ml = torch.from_numpy(np.stack([k["ml"] for k in sub])); mc = torch.from_numpy(np.stack([k["mc"] for k in sub]))
        me = torch.zeros_like(ml)
        feats = batch["feats"]
        outs = {}
        for name, (m, fill) in {"base": (None, None), "lead_mean": (ml, fill_mean), "ctrl_mean": (mc, fill_mean), "lead_zero": (ml, fill_zero),
                                "ctrl_zero": (mc, fill_zero), "empty_mean": (me, fill_mean)}.items():
            bb = dict(batch)
            if m is not None:
                bb["feats"] = apply_mask(feats, m, fill)
            with torch.no_grad():
                tr = model.forward_infer(bb)["traj30"].float().cpu()
            outs[name] = tr
            V[name][b0:b0 + len(sub)] = speed2s(tr)
        bit_equal_empty &= bool(torch.equal(outs["base"], outs["empty_mean"]))
        print(f"[s{a.seed}] {b0 + len(sub)}/{N} {time.time() - t0:.0f}s", flush=True)
    cid_names = sorted({k["clip"] for k in kept})
    cmap = {c: i for i, c in enumerate(cid_names)}
    cid = np.array([cmap[k["clip"]] for k in kept])
    gap = np.array([k["gap"] for k in kept])
    W = boot_weights(len(cid_names))
    res = {"card": "ADD155", "section": "A5", "tier": "T0", "seed": a.seed, "ckpt_sha256": cksha, "n_windows": N, "n_clips": len(cid_names),
           "selection": sel["info"], "tool_sha256": sha(os.path.abspath(__file__)), "impl_sha256": sha(f"{WUA}/a5/IMPL_A5.md"),
           "read": "planned speed at 2.0 s = |0.5 (P[10]-P[8])| / 0.2 of forward_infer traj30 (m/s)", "bootstrap": {"B": B_BOOT, "rng": 0, "unit": "clip (paired)"}}
    for tag, (l, c) in {"mean": ("lead_mean", "ctrl_mean"), "zero": ("lead_zero", "ctrl_zero")}.items():
        dl, dc = V[l] - V["base"], V[c] - V["base"]
        m = np.abs(dl) - np.abs(dc)
        res[tag] = {"metric_absdv_lead_minus_control": boot_mean(m, cid, W),
                    "absdv_lead": boot_mean(np.abs(dl), cid, W), "absdv_control": boot_mean(np.abs(dc), cid, W),
                    "signed_dv_lead": boot_mean(dl, cid, W), "signed_dv_control": boot_mean(dc, cid, W),
                    "signed_lead_minus_control": boot_mean(dl - dc, cid, W),
                    "share_dv_lead_positive": float((dl > 0).mean()), "share_dv_control_positive": float((dc > 0).mean()),
                    "gap_le_20": {"metric": boot_mean(m, cid, W, gap <= 20), "signed_dv_lead": boot_mean(dl, cid, W, gap <= 20), "signed_dv_control": boot_mean(dc, cid, W, gap <= 20)}}
    de = V["empty_mean"] - V["base"]
    res["c11"] = {"max_abs_dv_empty": float(np.abs(de).max()), "traj30_bit_identical": bit_equal_empty, "PASS": bool(np.abs(de).max() == 0.0 and bit_equal_empty)}
    res["v2_base"] = {"mean": float(V["base"].mean()), "sd": float(V["base"].std())}
    # ---- c10: lead-presence probe (P-LIN on TOK-GRID), primary mean-token masking, current frame t
    Lb = np.load(LAB)
    names = list(Lb["names"]); yl = Lb["Y"][:, names.index("lead_present")]
    lab_clips = list(Lb["clips"]); ci_lab = Lb["clip_idx"]; t_lab = Lb["t"].astype(int)
    sys.path.insert(0, WUA)
    import wua_a1 as A1
    folds_clip = A1.fold_of(lab_clips)
    rows = np.where(np.isfinite(yl))[0]
    G = np.zeros((len(rows), NR * NC * 256), np.float32)
    pos = {(int(ci_lab[r]), int(t_lab[r])): i for i, r in enumerate(rows)}
    with torch.no_grad():
        for j, c in enumerate(lab_clips):
            ii = [(i, int(t_lab[r])) for (i, r) in enumerate(rows) if ci_lab[r] == j]
            if not ii:
                continue
            f = torch.from_numpy(store.get(c)["feats"])[None].float()
            tok = model._encode({"feats": f})[0]
            for i, tt in ii:
                G[i] = tok[tt].reshape(-1).numpy()
    ytr = yl[rows]; cir = ci_lab[rows]
    # masked variants of the kept windows' frame t
    lab_idx = {c: j for j, c in enumerate(lab_clips)}
    GU, GL, GC, wf = [], [], [], []
    with torch.no_grad():
        for k in kept:
            f = torch.from_numpy(store.get(k["clip"])["feats"][k["t"]])[None, None]
            ml = torch.from_numpy(k["ml"][-1])[None, None]; mc = torch.from_numpy(k["mc"][-1])[None, None]
            xs = torch.cat([f, apply_mask(f, ml, fill_mean), apply_mask(f, mc, fill_mean)], 1)         # [1,3,40,1024]
            tk = model._encode({"feats": xs.float()})[0]
            GU.append(tk[0].reshape(-1).numpy()); GL.append(tk[1].reshape(-1).numpy()); GC.append(tk[2].reshape(-1).numpy())
            wf.append(int(folds_clip[lab_idx[k["clip"]]]))
    GU, GL, GC, wf = np.array(GU, np.float32), np.array(GL, np.float32), np.array(GC, np.float32), np.array(wf)
    dU, dL, dC = np.zeros(N), np.zeros(N), np.zeros(N)
    Cs = {}
    for f in range(5):
        w = np.where(wf == f)[0]
        if len(w) == 0:
            continue
        (u, l_, c_), bc = probe_fit(G, ytr, cir, folds_clip, f, [GU[w], GL[w], GC[w]])
        dU[w], dL[w], dC[w] = u, l_, c_
        Cs[f] = bc
    drop_l, drop_c = dU - dL, dU - dC
    res["c10_seed"] = {"drop_lead": boot_mean(drop_l, cid, W), "drop_control": boot_mean(drop_c, cid, W), "lead_minus_control": boot_mean(drop_l - drop_c, cid, W),
                       "logit_unmasked_mean": float(dU.mean()), "n_train_rows": int(len(rows)), "C_per_fold": Cs}
    os.makedirs(OUT, exist_ok=True)
    np.savez(f"{OUT}/RESULT_A5_s{a.seed}_arrays.npz", clip=np.array([k["clip"] for k in kept]), t=np.array([k["t"] for k in kept]), cid=cid, gap=gap,
             drop_l=drop_l, drop_c=drop_c, **{f"v2_{n}": V[n] for n in var})
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(f"{OUT}/RESULT_A5_s{a.seed}.json", "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("mean", "c10_seed", "c11")}, indent=1))


def model_keys():
    sys.path.insert(0, TREE)
    from tanitad.models import v8f_mini as M
    return set(M.INFERENCE_INPUT_KEYS) - {"act", "pose"}             # v8f_t1_eval.py:62


def cmd_combine(a):
    R = [json.load(open(f"{OUT}/RESULT_A5_s{k}.json")) for k in (0, 1, 2)]
    arr = [np.load(f"{OUT}/RESULT_A5_s{k}_arrays.npz") for k in (0, 1, 2)]
    for x in arr[1:]:
        assert (x["clip"] == arr[0]["clip"]).all() and (x["t"] == arr[0]["t"]).all()
    cid = arr[0]["cid"]; W = boot_weights(int(cid.max()) + 1)
    pooled_l = np.mean([x["drop_l"] for x in arr], 0); pooled_c = np.mean([x["drop_c"] for x in arr], 0)
    c10 = boot_mean(pooled_l - pooled_c, cid, W)
    c10["PASS"] = bool(c10["lo"] > 0)
    c11 = all(r["c11"]["PASS"] for r in R)
    per = {}
    for tag in ("mean", "zero"):
        per[tag] = [{"seed": r["seed"], **r[tag]["metric_absdv_lead_minus_control"], "signed_dv_lead": r[tag]["signed_dv_lead"]["point"],
                     "signed_dv_control": r[tag]["signed_dv_control"]["point"]} for r in R]

    def label(rows):
        if all(x["lo"] > 0 for x in rows):
            return "USED"
        if any(x["lo"] <= 0 <= x["hi"] for x in rows):
            return "NOT-USED"
        neg = [x["seed"] for x in rows if x["hi"] < 0]
        return f"UNNAMED-BY-CARD (separated negative at seed {neg})"
    lab = label(per["mean"])
    final = "INVALID (c11 failed)" if not c11 else ("MASK-FAILED (c10 failed)" if not c10["PASS"] else lab)
    out = {"card": "ADD155", "section": "A5", "tier": "T0", "label_primary_mean_token": lab, "FINAL": final, "c10_pooled_seeds": c10, "c11_all_seeds_pass": c11,
           "per_seed_mean_token": per["mean"], "per_seed_zeroing_beside": per["zero"], "label_zeroing_beside": label(per["zero"]),
           "n_windows": R[0]["n_windows"], "n_clips": R[0]["n_clips"], "selection": R[0]["selection"],
           "signed_read": "signed_dv_lead > 0 = masking the lead RAISES the planned speed (expected sign for a close lead); see per_seed rows",
           "families": "STRATEGIC / LONGITUDINAL / LATERAL / TACTICAL: not applicable (card section 0: no rollout scored against the logged future)"}
    json.dump(out, open(f"{OUT}/RESULT_A5.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


# ------------------------------------------------------------------------------------------------ synthetic self-test
def cmd_selftest(a):
    import torch
    # camera axes aligned with the rig: cam +z = rig +x, cam +x (right) = rig -y, cam +y (down) = rig -z; camera at the origin
    R = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)     # columns = cam axes in the rig frame
    t = np.zeros(3)
    xp, yp, z = px_of_rig(np.array([[10.0, 0, 0]]), R, t)
    assert abs(xp[0] - 320) < 1e-9 and abs(yp[0] - 128) < 1e-9 and z[0] == 10.0
    az30 = 10.0 * math.tan(math.radians(30))                        # 30 deg to the LEFT -> column x = 320 - f * 0.5236 = 160.0
    xp, _, _ = px_of_rig(np.array([[10.0, az30, 0]]), R, t)
    assert abs(xp[0] - (320 - FREF * math.atan2(az30, 10.0))) < 1e-9 and 150 < xp[0] < 170
    _, yp, _ = px_of_rig(np.array([[10.0, 0, 10.0 * math.tan(math.radians(5))]]), R, t)   # 5 deg UP -> row 128 - f tan 5
    assert abs(yp[0] - (128 - FREF * math.tan(math.radians(5)))) < 1e-9
    # f_ref reproduces 120 deg: 60 deg at the image edge
    xp, _, _ = px_of_rig(np.array([[10.0, 10.0 * math.tan(math.radians(60)), 0]]), R, t)
    assert abs(xp[0]) < 1e-6
    box = lambda cx, cy, cz=0.0, l=0.2, w=0.2, h=0.2: {"cx": cx, "cy": cy, "cz": cz, "l": l, "w": w, "h": h, "yaw": 0.0, "cls": "automobile"}
    # small box straight ahead -> cells straddling the centre: columns 4/5 only if it crosses the boundary; a box at (10, 0, 0) of 0.2 m crosses both
    cells = cells_of_rect(rect_of(box(10.0, 0.0), R, t))
    assert {c for _, c in cells} == {4, 5} and {r for r, _ in cells} == {1, 2}, cells
    # LEAD at azimuth ~ +30 deg (left): column 2; control = mirrored column 7
    ag = [box(3.0, 1.9)]
    lc, cc, why = frame_cells(ag, R, t)
    assert why == "ok" and {c for _, c in lc} == {2} and {c for _, c in cc} == {7} and {r for r, _ in lc} == {r for r, _ in cc}, (lc, cc, why)
    # another agent in column 7 pushes the control outward to column 8
    ag2 = [box(3.0, 1.9), box(10.0, -az30)]
    lc, cc, why = frame_cells(ag2, R, t)
    assert why == "ok" and {c for _, c in cc} == {8}, (cc, why)
    # a lead at azimuth -30 deg (right, column 7): mirrored column 2, outward = -1
    lc, cc, why = frame_cells([box(3.0, -1.9)], R, t)
    assert {c for _, c in lc} == {7} and {c for _, c in cc} == {2}
    # agents filling columns 7, 8, 9 leave no clean control
    az50 = 10.0 * math.tan(math.radians(50))
    ag3 = [box(3.0, 1.9), box(10.0, -az30), box(10.0, -10.0 * math.tan(math.radians(40))), box(10.0, -az50)]
    lc, cc, why = frame_cells(ag3, R, t)
    assert why == "no_clean_control" and cc is None, (cc, why)
    # lead selection: x > 0, |y| <= 2, x <= 40, nearest
    assert pick_lead([box(30, 1), box(20, 3), box(15, -1.9), box(45, 0), box(-5, 0)]) == 2
    # corners behind the camera: no cells
    assert rect_of(box(0.05, 0.0, l=4.0), R, t) is None
    # mask: empty mask is bit-identical, non-empty changes exactly the masked cells
    f = torch.randn(2, 3, 40, 16).half(); fill = torch.randn(40, 16).half()
    m0 = torch.zeros(2, 3, 40, dtype=torch.bool)
    assert torch.equal(apply_mask(f, m0, fill), f)
    m1 = m0.clone(); m1[0, 1, 5] = True
    o = apply_mask(f, m1, fill)
    assert torch.equal(o[0, 1, 5], fill[5]) and (o != f).any(-1).sum() == 1
    # speed read: constant 5 m/s along +x
    P = torch.stack([torch.arange(1, 31).float() * 1.0, torch.zeros(30)], 1)[None]
    assert abs(speed2s(P)[0] - 5.0) < 1e-6
    # bootstrap: a known positive effect separates, a zero effect does not
    rng = np.random.default_rng(3); cid = np.repeat(np.arange(40), 5); W = boot_weights(40)
    assert boot_mean(rng.normal(1.0, 0.5, 200), cid, W)["lo"] > 0
    z0 = boot_mean(rng.normal(0.0, 0.5, 200), cid, W)
    assert z0["lo"] <= 0 <= z0["hi"], z0
    # extrinsics quaternion of the real mount reproduces a forward-looking camera: boresight points to +x, pitch small
    q = json.load(open(EXTR)); k = next(iter(q)); Rm = rot_c2v(q[k])
    assert Rm[0, 2] > 0.99 and abs(Rm[2, 2]) < 0.1
    print("selftest PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("prep", "run", "combine", "selftest"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    {"prep": cmd_prep, "run": cmd_run, "combine": cmd_combine, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
