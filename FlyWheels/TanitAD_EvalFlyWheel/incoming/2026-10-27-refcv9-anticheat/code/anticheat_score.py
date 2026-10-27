"""anticheat_score.py -- PREREG_ANTICHEAT_REFCV8.md: build REAL / BLIND / EGO-ONLY FLOOR / CONST arms on the SAME eval139 windows, score them with the
battery's own four-family code (pair_read_r8.components -> paired episode-cluster bootstrap), and apply the pre-registered verdict.  CPU only.

NEW file (EvalFlyWheel 2026-10-27).  Nothing in the battery is edited: pf_lane.configure re-points it at the REGISTERED per-frame table exactly as
ship_r8_sub_forward.py does; `RT.surface_inputs` / `CR.prepare_seed` give the same rows (plan, prior, GT, compliance mask, ceiling) the registered
H2CN_DEPLOY8 was built on.  The H2cn stage is the REGISTERED frozen file (sha256 re-checked).  No clip id is printed or written (episodes are
integer cluster codes; clips appear as sha12 only).

    CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=<tree_fb47a58>/stack;<tree>/taniteval python -X utf8 anticheat_score.py --seed 0 \
        [--blind-bank <ztac_blind_s0.npz>] [--identity-bank <ztac_identity_s0.npz>] --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import anticheat_common as AC  # noqa: E402

SUB = Path("C:/Users/Admin/refcv8_pf/ship_r8_sub")
sys.path.insert(0, str(SUB))
import ship_r8_sub_common as C  # noqa: E402

L1_BANK = Path("C:/Users/Admin/refcv8_pf/refcv9_l1/bank")
REG_PF = Path("C:/Users/Admin/qland/hold_r8pf/REGISTERED_PF_TABLE.sha256")
REG_STAGE = SUB / "REGISTERED_H2CN_STAGE.sha256"
LANDED_H2CN = SUB / "out/arms/h2cn/H2CN_DEPLOY8"
TREE = "step48852ema_pf"
HZ8 = [5, 10, 15, 20, 30, 40, 50, 60]
SLOTS = C.SLOTS
SURFS = ("S2", "S6")
N_BOOT = 2000

# ---- the PRE-REGISTERED primary rows (PREREG_ANTICHEAT_REFCV8.md section 4) ----
PRIMARY = {"ADE": [("S2", "ade_m"), ("S6", "ade_m")],
           "LONGITUDINAL": [("S2", "LON_speed_mae_mps")],
           "LATERAL": [("S2", "LAT_heading_mae_deg")],
           "TACTICAL": [("S2", "TAC_traj_lat_correct"), ("S2", "TAC_traj_lon_correct"), ("S6", "TAC_traj_lat_correct"), ("S6", "TAC_traj_lon_correct")],
           "STRATEGIC": []}


def guard(min_avail_gb: float = float(os.environ.get("ANTICHEAT_MIN_AVAIL_GB", "0.4"))):
    """Self-protection: the GPU roll shares this box's RAM.  If available memory falls under the floor, record it and exit THIS process."""
    import psutil

    def run():
        while True:
            a = psutil.virtual_memory().available / 2**30
            if a < min_avail_gb:
                Path(os.environ.get("ANTICHEAT_GUARD_FILE", "guard_exit.txt")).write_text(f"exit: available RAM {a:.2f} GB < {min_avail_gb}\n")
                os._exit(9)
            time.sleep(2)
    threading.Thread(target=run, daemon=True).start()


def sha256_file(p) -> str:
    return C.sha256_file(p)


def sha_arrays(*arrs) -> str:
    h = hashlib.sha256()
    for a in arrs:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode() + str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------------------------------------------------- data (cached)
def build_data(seeds, cache_dir: Path) -> dict:
    """Per (seed, surface): everything the arms need (arrays only).  Per surface TRAIN rows for the floor.  Cached to npz (episodes as int codes)."""
    C.setup_paths()
    sys.path.insert(0, str(C.PF_HOLD / "code"))
    import pf_lane as PL
    reg_pf = PL.read_registered(REG_PF)
    PL.verify_installed(C.PF_HOLD / "pfpkg", reg_pf)
    M = PL.mods()
    PL.configure(M, C.PF_HOLD / "pfpkg", reg_pf)
    RT, CR, TFM = M["RT"], M["CR"], M["TFM"]
    import l1_fit_score as F
    PS = F.PS
    import torch
    from tanitad.models import kinematic_prior as KP
    from tanitad.models.ego_history import ego_channels_from_poses
    tools = RT.tools_load()
    data = {"eval": {}, "train": {}, "meta": {"reg_pf": reg_pf}}
    first = {}
    for s in seeds:
        ctx = CR.prepare_seed(TREE, s, tools, tools["rc"])
        for surf in SURFS:
            S = RT.surface_inputs(tools, ctx, surf, first, TREE)
            rows = ctx["panels"][surf]["rows"]
            win = np.ascontiguousarray(S["dec"]["win"])
            pw = torch.from_numpy(np.ascontiguousarray(win, dtype=np.float32))
            a0, kp = KP.prior_controls("ha0_ext_pose", pw)
            v0 = ego_channels_from_poses(pw, RT.W_OBS, dt=RT.DT)[:, -1, 0]
            prior8 = KP.prior_path(a0, kp, v0, HZ8).numpy().astype(np.float32)
            eid_raw = np.asarray(S["eid"])             # the cluster labels EXACTLY as RT.forward passes them (the bootstrap's draw order depends on them); never printed or written
            lim = np.asarray(ctx["lim"][rows], np.float64)
            hz = [int(h) for h in S["hz"]]
            if [HZ8[i] for i in SLOTS[surf]] != hz:
                raise SystemExit(f"[anticheat] {surf}: panel horizons {hz} are not the model slots {SLOTS[surf]}")
            d = {"clip": np.asarray(S["clip"], np.int64), "ws": np.asarray(S["ws"], np.int64), "eid": eid_raw, "mask": np.asarray(S["mask"], bool),
                 "dt": float(S["dt"]), "G": np.asarray(S["X"]["G"]), "HP": np.asarray(S["X"]["HP"]), "O": np.asarray(S["X"]["O"]), "plan8": np.ascontiguousarray(ctx["plan"][rows]),
                 "prior8": prior8, "lim": lim, "win": win, "v0": np.asarray(S["v0"], np.float64), "X": PS.feats(win, lim), "hz": np.asarray(hz, np.int64)}
            data["eval"][(s, surf)] = d
            del S
        del ctx
    # TRAIN rows (A6 train139, ceiling-compliant) for the floor: what l1_fit_score.load_train returns (model-free: identical across inference seeds)
    _cp, CR2 = TFM._patch_archive(TFM.TRAIN_ROOT)
    ftr = {}
    for surf in SURFS:
        tr = F.load_train(tools, CR2, TFM, surf, ftr)
        u, code = np.unique(tr["eid"], return_inverse=True)
        data["train"][surf] = {"HP": tr["HP"], "G": tr["G"], "X": tr["X"], "eid": code.astype(np.int64), "clip": tr["clip"], "ws": tr["ws"], "dt": float(tr["dt"]),
                               "v0": tr["v0"]}
    data["tools"] = tools
    data["F"] = F
    data["RT"] = RT
    data["PS"] = PS
    return data


# --------------------------------------------------------------------------------------------------------------------- the floor
def ridge_fit(X, Y, lam):
    mu, sd = X[:, 1:].mean(0), X[:, 1:].std(0) + 1e-9
    Z = np.concatenate([X[:, :1], (X[:, 1:] - mu) / sd], 1)
    R = lam * np.eye(Z.shape[1])
    R[0, 0] = 0.0                                                    # intercept unpenalised
    W = np.linalg.solve(Z.T @ Z + R, Z.T @ Y)
    return {"W": W, "mu": mu, "sd": sd}


def ridge_pred(m, X):
    Z = np.concatenate([X[:, :1], (X[:, 1:] - m["mu"]) / m["sd"]], 1)
    return Z @ m["W"]


LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)


def folds(eid, k=5, seed=0):
    u = np.unique(eid)
    rng = np.random.default_rng(seed)
    f_of = dict(zip(u.tolist(), (rng.permutation(len(u)) % k).tolist()))
    return np.asarray([f_of[int(e)] for e in eid])


def fit_floor(tr: dict) -> dict:
    """Joint choice of parametrisation {DIRECT, RESIDUAL over ha0_ext_pose} and lambda by 5-fold EPISODE-grouped CV SSE of the POSITION vector (m^2), train rows only."""
    X, G, HP, eid = tr["X"], tr["G"].astype(np.float64), tr["HP"].astype(np.float64), tr["eid"]
    n, K = G.shape[0], G.shape[1]
    Yd, Yr = G.reshape(n, -1), (G - HP).reshape(n, -1)
    f = folds(eid)
    cv = {}
    for name, Y in (("DIRECT", Yd), ("RESIDUAL", Yr)):
        for lam in LAMBDAS:
            sse = 0.0
            for j in range(5):
                a, b = f != j, f == j
                m = ridge_fit(X[a], Y[a], lam)
                sse += float(((ridge_pred(m, X[b]) - Y[b]) ** 2).sum())        # position error: the residual form's HP cancels
            cv[f"{name}|{lam:g}"] = sse
    best = min(cv, key=cv.get)
    name, lam = best.split("|")[0], float(best.split("|")[1])
    Y = Yd if name == "DIRECT" else Yr
    m = ridge_fit(X, Y, lam)
    P = ridge_pred(m, X)
    resid = P - Y
    r2 = float(1 - (resid ** 2).sum() / ((Yd - Yd.mean(0)) ** 2).sum()) if name == "DIRECT" else float(1 - ((P + HP.reshape(n, -1) - Yd) ** 2).sum() / ((Yd - Yd.mean(0)) ** 2).sum())
    return {"param": name, "lambda": lam, "model": m, "cv_sse": cv, "n": int(n), "d": int(X.shape[1]), "n_episodes": int(len(np.unique(eid))), "K": int(K),
            "train_R2_positions": r2, "cv_best_sse": cv[best], "cv_sse_per_window_m2": cv[best] / n}


def floor_predict(fl: dict, X, HP):
    P = ridge_pred(fl["model"], X)
    n = len(X)
    if fl["param"] == "RESIDUAL":
        return (P.reshape(n, -1, 2) + HP.astype(np.float64)).astype(np.float32)
    return P.reshape(n, -1, 2).astype(np.float32)


def floor_known_value_controls(tr: dict) -> dict:
    """K-FL1: a ridge fitted on a target that IS a feature (v0) recovers it (R^2 = 1 +- 1e-9).  K-FL2: a ridge on a ROW-SHUFFLED target has held-out R^2 <= 0.005."""
    X, eid = tr["X"], tr["eid"]
    y = X[:, 1:2].copy()                                          # column 1 = v0
    m = ridge_fit(X, y, 1e-3)
    p = ridge_pred(m, X)
    r2a = float(1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    rng = np.random.default_rng(0)
    Y = tr["G"].reshape(len(X), -1).astype(np.float64)
    Ys = Y[rng.permutation(len(Y))]
    f = folds(eid)
    sse = sst = 0.0
    for j in range(5):
        a, b = f != j, f == j
        mm = ridge_fit(X[a], Ys[a], 10.0)
        sse += float(((ridge_pred(mm, X[b]) - Ys[b]) ** 2).sum())
        sst += float(((Ys[b] - Ys[a].mean(0)) ** 2).sum())
    r2b = 1 - sse / sst
    return {"K_FL1_recovers_v0_R2": r2a, "K_FL1_PASS": bool(abs(r2a - 1.0) <= 1e-9), "K_FL2_shuffled_heldout_R2": float(r2b), "K_FL2_PASS": bool(r2b <= 0.005)}


def fit_nl(tr: dict):
    from sklearn.ensemble import HistGradientBoostingRegressor
    X = tr["X"][:, 1:]
    Y = tr["G"].reshape(len(X), -1).astype(np.float64)
    ms = [HistGradientBoostingRegressor(random_state=0).fit(X, Y[:, j]) for j in range(Y.shape[1])]
    return ms


def nl_predict(ms, X):
    P = np.stack([m.predict(X[:, 1:]) for m in ms], 1)
    return P.reshape(len(X), -1, 2).astype(np.float32)


# --------------------------------------------------------------------------------------------------------------------- arms
def h2cn_arm(RT, AL, preds, d, plan8, Z):
    """H2CN_DEPLOY8's construction (ship_r8_sub_forward.h2cn_progress + eval_polyline), on an arbitrary 8-slot plan and its z_tac: returns float32 positions at ALL 8 slots."""
    f2, f6 = preds["S2"](d["X"], Z), preds["S6"](d["X"], Z)
    a4 = AL.arc(d["prior8"][:, :4])
    a6 = AL.arc(d["prior8"][:, SLOTS["S6"]])
    s = np.concatenate([a4 + f2, a6[:, 2:] + f6[:, 2:]], axis=1)
    s0 = np.maximum(np.asarray(s, np.float64), 0.0)
    s8 = np.maximum.accumulate(s0, axis=1)
    geo = RT.polyline_geometry(plan8)
    p64 = RT.eval_polyline(geo, s8)
    return p64.astype(plan8.dtype)


def components_all(tools, P, G, dt):
    return tools["PR"].components(tools["ra"], P, G, dt)


def cell(tools, a, b, eid):
    c = tools["PR"].paired_cell(a, b, list(eid), N_BOOT, 0)
    return {k: c.get(k) for k in ("delta", "lo", "hi", "separated", "n_windows", "n_episodes", "mean_new", "mean_ref", "status")}


def row_status(c: dict, lower_is_better: bool) -> str:
    if c is None or c.get("lo") is None or c.get("hi") is None:
        return "NO_INTERVAL"
    if lower_is_better:
        return "BETTER_SEP" if c["hi"] < 0 else ("WORSE_SEP" if c["lo"] > 0 else "NOT_SEP")
    return "BETTER_SEP" if c["lo"] > 0 else ("WORSE_SEP" if c["hi"] < 0 else "NOT_SEP")


def family_verdicts(table: dict, lib: dict) -> dict:
    """table[(surf)][metric] = cell (REAL - X) on the compliant set.  A family PASSES iff every primary row is BETTER_SEP; STRATEGIC n = 0."""
    out = {}
    for fam, rows in PRIMARY.items():
        if not rows:
            out[fam] = {"status": "N/A (n = 0)", "rows": {}}
            continue
        st = {f"{s}|{k}": row_status(table[s][k], lib[k]) for s, k in rows}
        out[fam] = {"status": "PASS" if all(v == "BETTER_SEP" for v in st.values()) else ("FAIL (separated worse)" if any(v == "WORSE_SEP" for v in st.values()) else "FAIL (not separated)"),
                    "rows": st}
    live = [v["status"] for v in out.values() if not v["status"].startswith("N/A")]
    out["OVERALL"] = "PASS" if live and all(x == "PASS" for x in live) else "FAIL: " + ", ".join(f for f, v in out.items() if f != "OVERALL" and not v["status"].startswith(("PASS", "N/A")))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--blind-bank", default=None)
    ap.add_argument("--blind-bank1", default=None, help="the seed-1 blind bank (beside); the seed-0 blind bank is --blind-bank")
    ap.add_argument("--identity-bank", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--partial", action="store_true", help="UNREGISTERED PREVIEW: score only the windows already in a (non-final) blind bank; never a verdict")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ["ANTICHEAT_GUARD_FILE"] = str(out / "guard_exit.txt")
    guard()
    t_start = time.time()
    seeds = tuple(int(x) for x in a.seeds.split(","))
    prereg = HERE.parent / "PREREG_ANTICHEAT_REFCV8.md"
    seal = (HERE.parent / "PREREG_ANTICHEAT_REFCV8.sha256").read_text(encoding="utf-8").split()[0]
    prereg_sha = sha256_file(prereg)
    if prereg_sha != seal:
        raise SystemExit("[anticheat] PREREG does not match its seal: refusing")
    D = build_data(seeds, out)
    tools, F, RT, PS = D["tools"], D["F"], D["RT"], D["PS"]
    AL = F.AL
    keys = [m[1] for m in tools["PR"].METRICS]
    lib = {m[1]: bool(m[4]) for m in tools["PR"].METRICS}
    label = {m[1]: m[2] for m in tools["PR"].METRICS}
    unit = {m[1]: m[3] for m in tools["PR"].METRICS}
    fam_of = {m[1]: m[0] for m in tools["PR"].METRICS}
    reg_sha = C.read_seal(REG_STAGE)
    preds, blob, stage_sha = C.load_stage(C.STAGE_PT, reg_sha)          # refuses unless sha256(file) == the REGISTERED one
    res = {"tool": "anticheat_score.py", "generated": time.strftime("%FT%T%z"), "prereg_sha256": prereg_sha, "prereg_seal_matches": True,
           "script_sha256": sha256_file(__file__), "common_sha256": sha256_file(HERE / "anticheat_common.py"), "stage_sha256": stage_sha,
           "tier": "T1 self-action OPEN LOOP on logged eval139 frames (per-frame nav feed) -- not driving performance",
           "estimator": "pair_read_r8.paired_cell -> taniteval.ci.paired_episode_cluster_bootstrap, B 2000, bootstrap seed 0, over EPISODES, models fixed; "
                        "answers 'another draw of EPISODES' only (not another training run: one seed; not another inference run: seed 0 here)",
           "disclosures": ["refcv8's speed band (fed ceiling `lim`) is FUTURE-DERIVED (ego max speed 2-6 s ahead; leak 0.224 vs 0.05 bar)",
                           "refcv8's per-frame nav token is ego-future-derived (optimistic by construction)", "one training seed"],
           "controls": {}, "floor": {}, "seeds": {}}
    # ------------------------------------------------------------------ the floor (fit once on TRAIN rows; model-free => identical across inference seeds)
    floors = {}
    for surf in SURFS:
        tr = D["train"][surf]
        fl = fit_floor(tr)
        kv = floor_known_value_controls(tr)
        nl = fit_nl(tr)
        floors[surf] = {"fit": fl, "nl": nl, "const": tr["G"].astype(np.float64).mean(0)}
        res["floor"][surf] = {"n_train_windows": fl["n"], "d_features": fl["d"], "n_train_episodes": fl["n_episodes"], "param": fl["param"], "lambda": fl["lambda"],
                              "cv_best_sse_m2_per_window": fl["cv_sse_per_window_m2"], "train_R2_positions": fl["train_R2_positions"],
                              "cv_sse_all": fl["cv_sse"], "known_value_controls": kv, "features": "progress_scalar.feats: [1, v0, a0, omega0, v0*omega0, v0^2, a0*v0, min(lim,40), min(lim,40)-v0, lim_valid]",
                              "train_set": "A6 train139 ceiling-compliant windows (l1_fit_score.load_train)", "targets": f"ego-frame positions at {len(SLOTS[surf])} instants x 2"}
    res["controls"]["floor_known_values"] = bool(all(res["floor"][s]["known_value_controls"][k] for s in SURFS for k in ("K_FL1_PASS", "K_FL2_PASS")))
    # ------------------------------------------------------------------ per seed
    base_ok = True
    for s in seeds:
        bank_real = F.load_bank(L1_BANK / f"ztac_eval_s{s}.npz")
        bb_path = a.blind_bank if s == 0 else a.blind_bank1
        bank_blind = F.load_bank(Path(bb_path)) if bb_path else None
        sres = {"surfaces": {}, "bank_real": {"file": str(L1_BANK / f"ztac_eval_s{s}.npz"), "sha256": sha256_file(L1_BANK / f"ztac_eval_s{s}.npz")}}
        if bank_blind is not None:
            sres["bank_blind"] = {"file": bb_path, "sha256": sha256_file(bb_path), "meta": {k: bank_blind["meta"].get(k) for k in ("camera", "n_windows", "final", "derangement", "counters", "prereg_sha256")}}
        landed = np.load(LANDED_H2CN / f"DEPLOY8_POS_{TREE}_s{s}.npz") if (LANDED_H2CN / f"DEPLOY8_POS_{TREE}_s{s}.npz").exists() else None
        for surf in SURFS:
            d = D["eval"][(s, surf)]
            if a.partial:                                   # UNREGISTERED PREVIEW: the rows the (autosaved) blind bank already holds, nothing else
                bb_ = F.load_bank(Path(a.blind_bank))
                kp = np.array([(int(c), int(w)) in bb_["idx"] for c, w in zip(d["clip"], d["ws"])])
                n0 = len(kp)
                d = {k: (v[kp] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n0 else v) for k, v in d.items()}
            slots = SLOTS[surf]
            m = d["mask"]
            n_all, n_c = int(len(m)), int(m.sum())
            Zr, jr = F.join(bank_real, d["clip"], d["ws"])
            plan_os = np.ascontiguousarray(bank_real["traj"][jr]).astype(np.float32)
            arms = {}
            arms["REAL_REG"] = h2cn_arm(RT, AL, preds, d, d["plan8"], Zr)[:, slots]
            arms["REAL_OS"] = h2cn_arm(RT, AL, preds, d, plan_os, Zr)[:, slots]
            arms["RAW_REAL"] = plan_os[:, slots]
            arms["HP"] = d["HP"]
            fl = floors[surf]
            arms["FLOOR"] = floor_predict(fl["fit"], d["X"], d["HP"])
            arms["FLOOR_NL"] = nl_predict(fl["nl"], d["X"])
            arms["CONST"] = np.broadcast_to(fl["const"].astype(np.float32), (n_all,) + fl["const"].shape).copy()
            if bank_blind is not None:
                Zb, jb = F.join(bank_blind, d["clip"], d["ws"])
                plan_b = np.ascontiguousarray(bank_blind["traj"][jb]).astype(np.float32)
                arms["BLIND"] = h2cn_arm(RT, AL, preds, d, plan_b, Zb)[:, slots]
                arms["RAW_BLIND"] = plan_b[:, slots]
            # ---- K-BASE: REAL_REG == the landed H2CN_DEPLOY8 package, bit for bit
            if landed is not None:
                key = {(int(c), int(w)): j for j, (c, w) in enumerate(zip(landed[f"{surf}_clip"], landed[f"{surf}_ws"]))}
                jj = np.array([key.get((int(c), int(w)), -1) for c, w in zip(d["clip"], d["ws"])])
                kb = {"aligned": bool((jj >= 0).all() and (a.partial or len(jj) == len(landed[f"{surf}_ws"])))}
                if kb["aligned"]:
                    lp = landed[f"{surf}_pos"][jj]
                    kb["bit_equal"] = bool(lp.tobytes() == arms["REAL_REG"].tobytes())
                    kb["max_abs_m"] = float(np.abs(lp.astype(np.float64) - arms["REAL_REG"].astype(np.float64)).max())
                else:
                    kb["bit_equal"] = False
                res["controls"].setdefault("K_BASE_positions", {})[f"{surf}|s{s}"] = kb
                base_ok = base_ok and kb["bit_equal"]
            # ---- components of every arm (all rows once; compliant = mask slice)
            comp = {nm: components_all(tools, P, d["G"], d["dt"]) for nm, P in arms.items()}
            comp["GT"] = components_all(tools, d["G"], d["G"], d["dt"])
            eid = d["eid"]
            sres["surfaces"][surf] = {"n_all": n_all, "n_compliant": n_c, "n_episodes_all": int(len(np.unique(eid))), "n_episodes_compliant": int(len(np.unique(eid[m]))),
                                      "slots": slots, "contrasts": {}, "means": {}}
            # known values: GT vs GT
            gt = comp["GT"]
            sres["surfaces"][surf]["K_ID_GT"] = {"TAC_lat_mean": float(np.nanmean(gt["TAC_traj_lat_correct"])), "TAC_lon_mean": float(np.nanmean(gt["TAC_traj_lon_correct"])),
                                                 "ade_max": float(np.nanmax(gt["ade_m"]))}
            sres["surfaces"][surf]["K_ID_REALREAL"] = cell(tools, comp["REAL_REG"]["ade_m"][m], comp["REAL_REG"]["ade_m"][m], eid[m])
            for nm in arms:
                sres["surfaces"][surf]["means"][nm] = {k: float(np.nanmean(comp[nm][k][m])) for k in keys}
            pairs = [("REAL_REG", "HP"), ("REAL_REG", "FLOOR"), ("REAL_OS", "FLOOR"), ("REAL_REG", "CONST"), ("REAL_REG", "FLOOR_NL"), ("FLOOR", "CONST"), ("FLOOR", "HP"),
                     ("REAL_REG", "REAL_OS")]
            if bank_blind is not None:
                pairs += [("REAL_OS", "BLIND"), ("REAL_REG", "BLIND"), ("RAW_REAL", "RAW_BLIND"), ("BLIND", "FLOOR"), ("BLIND", "HP")]
            for A_, B_ in pairs:
                for setname, sel in (("compliant", m), ("all", np.ones_like(m))):
                    sres["surfaces"][surf]["contrasts"][f"{A_}-{B_}|{setname}"] = {k: cell(tools, comp[A_][k][sel], comp[B_][k][sel], eid[sel]) for k in keys}
            print(f"[anticheat] seed {s} {surf}: n_all {n_all} n_compliant {n_c} done {time.time() - t_start:.0f}s", flush=True)
        # ---- K-BASE cells: E1 / E3 reproduce the landed H2CN_DEPLOY8 cells (compliant, ADE minus ha0_ext_pose)
        c2 = sres["surfaces"]["S2"]["contrasts"]["REAL_REG-HP|compliant"]["ade_m"]
        c6 = sres["surfaces"]["S6"]["contrasts"]["REAL_REG-HP|compliant"]["ade_m"]
        landed_fw = SUB / "out/forward/FORWARD_H2CN_step48852ema_pf.json"
        kb = {"E1_S2": c2, "E3_S6": c6}
        try:
            fj = json.loads(landed_fw.read_text(encoding="utf-8"))
            for surf, c in (("S2", c2), ("S6", c6)):
                lc = fj["seeds"][str(s)]["surfaces"][surf]["sets"]["compliant"]["arms"]["H2CN_DEPLOY8"]["minus_ha0_ext_pose"]
                kb[f"landed_{surf}"] = {k: lc.get(k) for k in ("delta", "lo", "hi", "n_windows", "n_episodes")}
                kb[f"cell_equal_{surf}"] = bool(all(abs(float(lc[k]) - float(c[k])) <= 1e-9 for k in ("delta", "lo", "hi")) and lc["n_windows"] == c["n_windows"])
        except Exception as e:                                 # noqa: BLE001
            kb["landed_read_error"] = type(e).__name__
        res["controls"].setdefault("K_BASE_cells", {})[f"s{s}"] = kb
        res["seeds"][str(s)] = sres
    res["controls"]["K_BASE_all_positions_bit_equal"] = bool(base_ok)
    res["partial_preview"] = bool(a.partial)
    res["controls"]["K_BASE_cells_equal"] = (None if a.partial else bool(all(v.get(f"cell_equal_{x}") is True for v in res["controls"]["K_BASE_cells"].values() for x in ("S2", "S6"))))
    res["controls"]["K_ID"] = bool(all(abs(sres["surfaces"][x]["K_ID_REALREAL"]["delta"]) == 0.0 and sres["surfaces"][x]["K_ID_REALREAL"]["lo"] == 0.0 and sres["surfaces"][x]["K_ID_REALREAL"]["hi"] == 0.0
                                       and sres["surfaces"][x]["K_ID_GT"]["TAC_lat_mean"] == 1.0 and sres["surfaces"][x]["K_ID_GT"]["TAC_lon_mean"] == 1.0 and sres["surfaces"][x]["K_ID_GT"]["ade_max"] == 0.0
                                       for sres in res["seeds"].values() for x in SURFS))
    # ------------------------------------------------------------------ W4 roll controls (a)(b)(c)
    if a.blind_bank:
        zb = np.load(a.blind_bank)
        mb = json.loads(str(zb["meta_json"]))
        res["controls"]["blind_roll_final_4754"] = bool(mb.get("final") is True and mb.get("n_windows") == 4754)
        res["controls"]["derangement"] = mb.get("derangement")
        res["controls"]["derangement_ok"] = bool((mb.get("derangement") or {}).get("fixed_points") == 0 and (mb.get("derangement") or {}).get("each_episode_a_donor_once") is True
                                                  and (mb.get("derangement") or {}).get("mapping_sha256") == "48ff98cb395ef572ccbbadfcad59fd8be883666b4d485631817f3f1b0d33ba43")
        own = zb["own_h"]
        sm = own != 0
        res["controls"]["b_own_sample"] = {"n": int(sm.sum()), "n_own_differs_fed": int((own[sm] != zb["fed_h"][sm]).sum()), "PASS": bool(sm.any() and (own[sm] != zb["fed_h"][sm]).all())}
        if a.identity_bank:
            zi = np.load(a.identity_bank)
            mi = json.loads(str(zi["meta_json"]))
            base = np.load(L1_BANK / "ztac_eval_s0.npz")
            ctl_a = AC.check_bank_equal({k: base[k] for k in ("e_i", "t", "traj_own", "z_tac", "sel_own")}, {k: zi[k] for k in ("e_i", "t", "traj_own", "z_tac", "sel_own")})
            res["controls"]["a_identity_vs_BASE_bank"] = {**ctl_a, "identity_final_4754": bool(mi.get("final") is True and mi.get("n_windows") == 4754)}
            ordb = {(int(e), int(t)): j for j, (e, t) in enumerate(zip(zb["e_i"], zb["t"]))}
            ordi = {(int(e), int(t)): j for j, (e, t) in enumerate(zip(zi["e_i"], zi["t"]))}
            ks = sorted(ordb)
            same_keys = set(ordb) == set(ordi)
            jb = np.array([ordb[k] for k in ks]); ji = np.array([ordi[k] for k in ks]) if same_keys else None
            res["controls"]["b_changed_fraction_vs_identity"] = AC.check_changed_fraction(zb["fed_h"][jb], zi["fed_h"][ji]) if same_keys else {"PASS": False, "reason": "key sets differ"}
            res["controls"]["c_ego_inputs_equal_vs_identity"] = AC.check_ego_equal(zb["other_h"][jb], zi["other_h"][ji]) if same_keys else {"PASS": False, "reason": "key sets differ"}
        else:
            res["controls"]["a_identity_vs_BASE_bank"] = {"PASS": None, "status": "PENDING (identity roll not provided)"}
    # ------------------------------------------------------------------ verdicts (pre-registered rule)
    void_reasons = []
    for k in ("floor_known_values", "K_BASE_all_positions_bit_equal", "K_BASE_cells_equal", "K_ID"):
        if res["controls"].get(k) is not True and not (a.partial and k == "K_BASE_cells_equal"):
            void_reasons.append(k)
    res["verdicts"] = {}
    for s, sres in res["seeds"].items():
        v = {}
        w5 = {surf: sres["surfaces"][surf]["contrasts"]["REAL_REG-FLOOR|compliant"] for surf in SURFS}
        v["W5_REAL_REG_vs_FLOOR"] = family_verdicts(w5, lib)
        if "REAL_OS-BLIND|compliant" in sres["surfaces"]["S2"]["contrasts"]:
            w4 = {surf: sres["surfaces"][surf]["contrasts"]["REAL_OS-BLIND|compliant"] for surf in SURFS}
            v["W4_REAL_OS_vs_BLIND"] = family_verdicts(w4, lib)
            w4b = {surf: sres["surfaces"][surf]["contrasts"]["REAL_REG-BLIND|compliant"] for surf in SURFS}
            v["beside_REAL_REG_vs_BLIND"] = family_verdicts(w4b, lib)
        res["verdicts"][s] = v
    ctl_w4 = {}
    if a.blind_bank:
        for k in ("blind_roll_final_4754", "derangement_ok"):
            ctl_w4[k] = res["controls"].get(k)
        ctl_w4["b_own_sample"] = res["controls"]["b_own_sample"]["PASS"]
        ca = res["controls"]["a_identity_vs_BASE_bank"]
        ctl_w4["a_identity_bit_equal_all"] = ca.get("PASS")
        ctl_w4["b_changed_fraction_100"] = (res["controls"].get("b_changed_fraction_vs_identity") or {}).get("PASS")
        ctl_w4["c_ego_inputs_equal"] = (res["controls"].get("c_ego_inputs_equal_vs_identity") or {}).get("PASS")
    res["controls"]["w4_roll_controls"] = ctl_w4
    res["void_reasons_common"] = void_reasons
    res["w4_status"] = ("PREVIEW ONLY (unregistered, partial windows; no verdict)" if a.partial else "NOT RUN (no blind bank)" if not a.blind_bank else
                        ("VOID: " + ", ".join(k for k, v in ctl_w4.items() if v is False) if any(v is False for v in ctl_w4.values()) else
                         ("PROVISIONAL (control (a)/(b)/(c) vs identity roll pending)" if any(v is None for v in ctl_w4.values()) else "CONTROLS PASS")))
    res["w5_status"] = ("VOID: " + ", ".join(void_reasons)) if void_reasons else "CONTROLS PASS"
    res["wall_s"] = round(time.time() - t_start, 1)
    (out / "RESULT_RAW.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    meta = {"keys": keys, "label": label, "unit": unit, "family": fam_of, "lower_is_better": lib}
    (out / "RESULT_META.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"[anticheat] WROTE {out / 'RESULT_RAW.json'} wall {res['wall_s']}s w4={res['w4_status']} w5={res['w5_status']} ZZACSCOREDONEZZ", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
