"""lha0_diag.py -- PREREG_LHA0_DIAG (sha256 in PREREG_LHA0_DIAG.sha256): do a0 / psi_dot0 at t0 add decel-decision information to LH's head?  T0 label-level diagnostic.

stages:  prep    build the row universe U, the score population P, the instrument gates K-POP / K-ID  (scratch universe.npz)
         fit     run every OOF job (arm x fold-seed x lambda) in a process pool; one scratch npz per job
         assemble  AUCs, paired episode-cluster bootstrap (B 2000, seed 0), controls, verdict -> result/LHA0_DIAG.{json,md}
         all     prep, fit, assemble

Imports UNEDITED from refcv9_lonhead/code (lh_fit: load_fresh, folds_of, std_of, fit_logistic, logits_logistic, log_softmax, ce_rows, load_frozen, predict_logits6;
k_head: softmax, auc_np, WAuc, ci, renorm_decel) and refcv9_l2/code/l2_common (load_train_dump, G_DECEL, G_CRUISE).  CPU only.  NEW file (LHA0 agent, 2026-10-26).
No clip id is printed or written (episodes are integer indices / sha12 only).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.dont_write_bytecode = True
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCRATCH = Path("F:/c_overflow_2026-10-26/tmp/lha0")
TREE = Path("C:/Users/Admin/refcv8_pf/gtmap_oracle/tree_fb47a58")
LONHEAD = Path("C:/Users/Admin/refcv8_pf/refcv9_lonhead")
for _p in (str(TREE / "stack"), str(LONHEAD / "code"), "C:/Users/Admin/refcv8_pf/refcv9_l2/code", "C:/Users/Admin/refcv8_pf/refcv9_l1/code"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

FRESH_DUMP = Path("F:/l1c/tree/step48852ema_fresh/dump_s0")
EVAL_DUMP = Path("C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_pf/dump_s0")
EVAL_BANK = Path("C:/Users/Admin/refcv8_pf/refcv9_l1/bank/ztac_eval_s0.npz")
BOOT_B, BOOT_SEED = 2000, 0
BAR_D = 0.020
LAM_FROZEN = 100.0
LH_OOF_CE_LAM100 = 1.0917272119292791          # LH_FIT_INFO.json LH.table["logistic:100"].mean_ce_oof
LON6 = ("FOLLOW", "CRUISE", "BRAKE_TO", "CREEP", "HOLD", "ACCELERATE")

# arm -> (include_z, include_v0, extra feature names, default lambda)
ARMS = {
    "A": (True, True, (), LAM_FROZEN),
    "B": (True, True, ("a0", "psi"), LAM_FROZEN),
    "C4": (True, True, ("a0_perm", "psi_perm"), LAM_FROZEN),
    "C5": (True, True, ("a0_fwd", "psi"), LAM_FROZEN),
    "B05": (True, True, ("a0_05", "psi"), LAM_FROZEN),
    "Bnom": (True, True, ("a0_nom", "psi_nom"), LAM_FROZEN),
    "C2": (False, True, (), 1.0),
    "C3": (False, False, ("a0",), 1.0),
    "X1": (False, True, ("a0", "psi"), 1.0),
}


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def pr(x, d=4):
    return None if x is None else round(float(x), d)


# ------------------------------------------------------------------------------------------------------------------ prep
def prep() -> dict:
    import lh_fit as LF
    import k_head as KH
    import l2_common as C
    from l1_fit_score import load_bank
    from pathlib import Path as P_
    t0 = time.time()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    cal = json.load(open(ROOT / "result" / "LHA0_CALIBRATION.json", encoding="utf-8"))
    assert cal["verdict"] == "CALIBRATED" and cal["mutation_forward_vs_backward_a0"]["pass"], cal["verdict"]
    ft = np.load(ROOT / "result" / "lha0_features.npz")
    bk = np.load(LF.BANK, allow_pickle=False)
    n_bank = len(bk["e_i"])
    e_i, ws_b = bk["e_i"].astype(np.int64), bk["t"].astype(np.int64) + 7
    assert np.array_equal(ft["e_i"], e_i) and np.array_equal(ft["ws"], ws_b), "feature table != bank order"
    D = LF.load_fresh(LF.FRESH_LEAD_RELEASE, False, True)
    # ---- the fresh dump's population (K-HEAD's plan-class rule, l2_common.load_train_dump) ----
    E = C.load_train_dump(FRESH_DUMP, "fresh")
    assert np.array_equal(E["clip"], e_i) and np.array_equal(E["ws"], ws_b), "dump order != bank order"
    gt = np.asarray(E["gt"])
    # ---- universe U ----
    fit_eps = np.unique(D["eid"])
    in_u = np.isin(e_i, fit_eps)
    feat_keys = ("a0", "psi_dot0", "a0_nominal_dt", "psi_dot0_nominal_dt", "a0_05", "a0_fwd")
    finite = np.all([np.isfinite(ft[k]) for k in feat_keys], axis=0)
    drops_u = int((in_u & ~finite).sum())
    uidx = np.flatnonzero(in_u & finite)                                  # bank rows of U
    row_of_bank = {int(b): i for i, b in enumerate(uidx)}
    key2bank = {(int(c), int(w)): j for j, (c, w) in enumerate(zip(e_i, ws_b))}
    d_bank = np.array([key2bank[(int(c), int(w))] for c, w in zip(D["eid"], D["ws"])])
    assert all(int(b) in row_of_bank for b in d_bank), "a fit row fell out of U"
    d_u = np.array([row_of_bank[int(b)] for b in d_bank])
    Zu = bk["z_tac"].astype(np.float64)[uidx]
    v0u = bk["v0_in"].astype(np.float64)[uidx]
    k_id = {"fit_rows_Z_bitwise_equal_bank": bool(np.array_equal(Zu[d_u], D["Z"])), "fit_rows_v0_bitwise_equal_bank": bool(np.array_equal(v0u[d_u], D["v0"])),
            "bank_v0_in_vs_extras_v0_maxabs": float(D["checks"]["bank_v0_in_vs_extras_v0_maxabs"]),
            "dump_v0_vs_bank_v0_maxabs": float(np.abs(E["v0"].astype(np.float64) - bk["v0_in"].astype(np.float64)).max()),
            "n_U": int(len(uidx)), "n_U_dropped_nonfinite_features": drops_u, "n_fit_rows": int(len(d_u)), "n_fit_episodes": int(len(fit_eps))}
    k_id["pass"] = bool(k_id["fit_rows_Z_bitwise_equal_bank"] and k_id["fit_rows_v0_bitwise_equal_bank"] and k_id["bank_v0_in_vs_extras_v0_maxabs"] == 0.0
                        and k_id["dump_v0_vs_bank_v0_maxabs"] == 0.0)
    eid_u, ws_u = e_i[uidx], ws_b[uidx]
    feats = {"a0": ft["a0"][uidx], "psi": ft["psi_dot0"][uidx], "a0_nom": ft["a0_nominal_dt"][uidx], "psi_nom": ft["psi_dot0_nominal_dt"][uidx],
             "a0_05": ft["a0_05"][uidx], "a0_fwd": ft["a0_fwd"][uidx]}
    # ---- C4: ONE shared permutation of each episode's U rows (default_rng(0), episodes in sorted order) ----
    rng = np.random.default_rng(0)
    a0p, psip = np.empty_like(feats["a0"]), np.empty_like(feats["psi"])
    for e in np.unique(eid_u):
        ix = np.flatnonzero(eid_u == e)
        pm = rng.permutation(len(ix))
        a0p[ix], psip[ix] = feats["a0"][ix][pm], feats["psi"][ix][pm]
        assert np.array_equal(np.sort(a0p[ix]), np.sort(feats["a0"][ix])) and np.array_equal(np.sort(psip[ix]), np.sort(feats["psi"][ix]))
    feats["a0_perm"], feats["psi_perm"] = a0p, psip
    c4_moved = float((a0p != feats["a0"]).mean())
    # ---- population ----
    valid, comp = np.asarray(E["valid_s6"], bool)[uidx], np.asarray(E["compliant"], bool)[uidx]
    g_u = gt[uidx]
    is_dec, is_cru = g_u == "decel", g_u == "cruise"
    score_mask = valid & (is_dec | is_cru)                                # P-nocomp
    Pmask = score_mask & comp                                             # PRIMARY population
    is_fit = np.zeros(len(uidx), bool)
    is_fit[d_u] = True
    y_u = np.full(len(uidx), -1, np.int64)
    y_u[d_u] = D["y"]
    pop = {"U_rows": int(len(uidx)), "U_episodes": int(len(fit_eps)),
           "P_rows": int(Pmask.sum()), "P_decel": int((Pmask & is_dec).sum()), "P_cruise": int((Pmask & is_cru).sum()),
           "P_episodes": int(len(np.unique(eid_u[Pmask]))), "P_decel_base_rate": pr((Pmask & is_dec).sum() / max(Pmask.sum(), 1)),
           "Pfit_rows": int((Pmask & is_fit).sum()), "Pfit_decel": int((Pmask & is_fit & is_dec).sum()),
           "Pnocomp_rows": int(score_mask.sum()), "Pnocomp_decel": int((score_mask & is_dec).sum()),
           "P_rows_in_bank_outside_U": int((np.asarray(E['valid_s6'], bool) & np.asarray(E['compliant'], bool) & ((gt == 'decel') | (gt == 'cruise')) & ~in_u).sum()),
           "gt_accel_rows_in_U_excluded": int((valid & comp & (g_u == "accel")).sum()),
           "fit_rows": int(len(d_u)), "fit_episodes": int(len(fit_eps)), "fit_class_counts": D["class_counts"], "C4_frac_values_moved": pr(c4_moved)}
    # ---- K-POP: the same population/score code on EVAL139, frozen LH, reproduces K-HEAD (union 2495, 639 / 1856, AUC 0.7305) ----
    Ee = C.load_train_dump(EVAL_DUMP, "eval")
    m = np.asarray(Ee["valid_s6"], bool) & np.asarray(Ee["compliant"], bool)
    ge = np.asarray(Ee["gt"])
    ui = np.flatnonzero(m & ((ge == "decel") | (ge == "cruise")))
    zb = load_bank(EVAL_BANK)
    jz = np.array([zb["idx"].get((int(c), int(w)), -1) for c, w in zip(Ee["clip"][ui], Ee["ws"][ui])])
    assert (jz >= 0).all()
    LH = LF.load_frozen(LONHEAD / "LH_FROZEN.pt", "6fd3ea1e3700a15b6506176bc855355d0cc1eefb3d87d390584e791d98e1d8f6")
    Pe = KH.softmax(LF.predict_logits6(LH, zb["z"][jz], np.asarray(zb["v0_in"], np.float64)[jz]))
    se, _ = KH.renorm_decel(Pe)
    auc_e = KH.auc_np((ge[ui] == "decel"), se)
    k_pop = {"s6_valid": int(np.asarray(Ee["valid_s6"], bool).sum()), "compliant": int(m.sum()), "gt_decel": int((m & (ge == "decel")).sum()), "gt_cruise": int((m & (ge == "cruise")).sum()),
             "union_n": int(len(ui)), "n_episodes": int(len(np.unique(Ee["eid"][ui]))), "LH_auc": float(auc_e),
             "want": {"s6_valid": 3668, "compliant": 3251, "gt_decel": 639, "gt_cruise": 1856, "union_n": 2495, "n_episodes": 136, "LH_auc": 0.7305}}
    k_pop["pass"] = bool(k_pop["s6_valid"] == 3668 and k_pop["compliant"] == 3251 and k_pop["gt_decel"] == 639 and k_pop["gt_cruise"] == 1856 and k_pop["union_n"] == 2495
                         and k_pop["n_episodes"] == 136 and abs(auc_e - 0.7305) < 5e-5)
    np.savez(SCRATCH / "universe.npz", Zu=Zu, v0u=v0u, eid_u=eid_u, ws_u=ws_u, is_fit=is_fit, y_u=y_u, d_u=d_u, d_eid=D["eid"], d_y=D["y"], Pmask=Pmask, score_mask=score_mask,
             is_dec=is_dec, **{f"f_{k}": v for k, v in feats.items()})
    out = {"calibration_verdict": cal["verdict"], "K_ID": k_id, "K_POP": k_pop, "population": pop, "wall_s": round(time.time() - t0, 1),
           "inputs_sha256": {"ztac_fresh.npz": sha256_file(LF.BANK), "lha0_features.npz": sha256_file(ROOT / "result" / "lha0_features.npz"),
                             "v9_labels_eval.npz(fresh lead release)": sha256_file(LF.FRESH_LEAD_RELEASE), "LH_FROZEN.pt": sha256_file(LONHEAD / "LH_FROZEN.pt"),
                             "LH_FIT_INFO.json": sha256_file(LONHEAD / "LH_FIT_INFO.json"), "lh_fit.py": sha256_file(LONHEAD / "code" / "lh_fit.py"),
                             "k_head.py": sha256_file(LONHEAD / "code" / "k_head.py"), "l2_common.py": sha256_file("C:/Users/Admin/refcv8_pf/refcv9_l2/code/l2_common.py"),
                             "fresh_extras": sha256_file(FRESH_DUMP / "refcv6_extras.npz"), "eval_bank": sha256_file(EVAL_BANK)}}
    (SCRATCH / "prep.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"ZZPREPZZ K_ID {k_id['pass']} K_POP {k_pop['pass']} (auc {auc_e:.5f}) pop {json.dumps(pop)[:300]}", flush=True)
    return out


# ------------------------------------------------------------------------------------------------------------------ fit
def _load_u():
    z = np.load(SCRATCH / "universe.npz")
    return {k: z[k] for k in z.files}


def fit_job(job: dict) -> str:
    """One OOF job: arm x fold-seed x lambda x model.  Writes SCRATCH/jobs/<name>.npz (logits6 for every scored row, NaN elsewhere)."""
    import lh_fit as LF
    t0 = time.time()
    name = job["name"]
    out = SCRATCH / "jobs" / f"{name}.npz"
    if out.exists():
        return f"{name} (cached)"
    U = _load_u()
    inc_z, inc_v0, extras, lam0 = ARMS[job["arm"]]
    lam = float(job.get("lam", lam0))
    seed = int(job.get("seed", 0))
    fold_d = LF.folds_of(U["d_eid"], 5, seed)
    e2f = {int(e): int(f) for e, f in zip(U["d_eid"], fold_d)}
    fold_u = np.array([e2f[int(e)] for e in U["eid_u"]])
    nU = len(fold_u)
    scored = U["Pmask"] | U["score_mask"] | U["is_fit"]
    logits = np.full((nU, 6), np.nan)
    coef = []
    n_iter = []
    ex_cols = [U[f"f_{e}"] for e in extras]
    for k in range(5):
        trD = fold_d != k
        tr_u = U["d_u"][trD]
        ytr = U["d_y"][trD]
        te_u = np.flatnonzero((fold_u == k) & scored)

        def build(ix, st, mus):
            cols = []
            if inc_z:
                cols.append((U["Zu"][ix] - st["zmu"]) / st["zsd"])
            if inc_v0:
                cols.append((U["v0u"][ix] / 10.0)[:, None])
            for c, (mu, sd) in zip(ex_cols, mus):
                cols.append(((c[ix] - mu) / sd)[:, None])
            return np.column_stack(cols)

        st = LF.std_of(U["Zu"][tr_u]) if inc_z else None
        mus = [(float(c[tr_u].mean()), float(c[tr_u].std() + 1e-9)) for c in ex_cols]
        Xtr, Xte = build(tr_u, st, mus), build(te_u, st, mus)
        if job.get("model", "logreg") == "logreg":
            m = LF.fit_logistic(Xtr, ytr, lam)
            logits[te_u] = LF.logits_logistic(m, Xte)
            n_iter.append(int(m["n_iter"]))
            if ex_cols:
                coef.append(m["W"][:, -len(ex_cols):].copy())
        else:                                                               # LH's MLP recipe, descriptive beside
            etr = np.unique(U["d_eid"][trD])
            rin = np.random.default_rng(100 + k).permutation(len(etr))
            inner_val = set(etr[rin[: max(1, int(round(0.2 * len(etr))))]].tolist())
            iv = np.array([int(e) in inner_val for e in U["d_eid"][trD]])
            net, ep = LF._train_mlp(Xtr[~iv], ytr[~iv], float(job["wd"]), None, Xtr[iv], ytr[iv])
            logits[te_u] = LF.logits_mlp(net, Xte)
            n_iter.append(int(ep))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, logits=logits, d=np.array([Xtr.shape[1]]), n_iter=np.array(n_iter), coef=np.array(coef) if coef else np.zeros((0,)))
    os.replace(tmp, out)
    return f"{name} {round(time.time() - t0, 1)}s d={Xtr.shape[1]} iters={n_iter}"


def jobs_list() -> list:
    J = []
    for a in ("A", "B", "C4", "C5", "B05", "Bnom"):
        J.append({"name": f"{a}_s0_l100", "arm": a, "seed": 0})
    for a in ("C2", "C3", "X1"):
        J.append({"name": f"{a}_s0_l1", "arm": a, "seed": 0})
    for a in ("A", "B"):
        J.append({"name": f"{a}_s0_l10", "arm": a, "seed": 0, "lam": 10.0})
        J.append({"name": f"{a}_s0_l1000", "arm": a, "seed": 0, "lam": 1000.0})
        J.append({"name": f"{a}_s0_mlp", "arm": a, "seed": 0, "model": "mlp", "wd": 1e-2})
    for s in (1, 2, 3, 4):
        for a in ("A", "B", "C4"):
            J.append({"name": f"{a}_s{s}_l100", "arm": a, "seed": s})
    return J


def fit_all(workers: int) -> None:
    J = jobs_list()
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for msg in ex.map(fit_job, J):
            print(f"[fit] {msg}  (+{time.time() - t0:.0f}s)", flush=True)
    print("ZZFITDONEZZ", flush=True)


# ------------------------------------------------------------------------------------------------------------------ assemble
def assemble() -> dict:
    import lh_fit as LF
    import k_head as KH
    t0 = time.time()
    U = _load_u()
    prep_j = json.load(open(SCRATCH / "prep.json", encoding="utf-8"))
    cal = json.load(open(ROOT / "result" / "LHA0_CALIBRATION.json", encoding="utf-8"))
    nU = len(U["eid_u"])
    d_u, d_y = U["d_u"], U["d_y"]
    pops = {"P": U["Pmask"], "P_fit": U["Pmask"] & U["is_fit"], "P_nocomp": U["score_mask"]}
    # cluster structure + the paired bootstrap draws (identical for every arm / comparison): K-HEAD's loop, B 2000, seed 0
    clus = {}
    for pn, pm in pops.items():
        ix = np.flatnonzero(pm)
        ue, inv = np.unique(U["eid_u"][ix], return_inverse=True)
        rng = np.random.default_rng(BOOT_SEED)
        cnt = np.stack([np.bincount(rng.integers(0, len(ue), len(ue)), minlength=len(ue)).astype(np.float64) for _ in range(BOOT_B)])
        clus[pn] = {"ix": ix, "y": U["is_dec"][ix], "inv": inv, "n_ep": len(ue), "cnt": cnt}

    def load_scores(name):
        z = np.load(SCRATCH / "jobs" / f"{name}.npz")
        lg = z["logits"]
        P6 = KH.softmax(np.nan_to_num(lg, nan=0.0))
        s, _ = KH.renorm_decel(P6)
        s = np.where(np.isfinite(lg).all(1), s, np.nan)
        ce = np.nan
        if np.isfinite(lg[d_u]).all():
            ce = float(LF.ce_rows(LF.log_softmax(lg[d_u]), d_y).mean())
        return {"score": s, "ce_fit_oof": ce, "d": int(z["d"][0]), "n_iter": z["n_iter"].tolist(), "coef": z["coef"]}

    cache = {}

    def S(name):
        if name not in cache:
            cache[name] = load_scores(name)
        return cache[name]

    def auc_boot(names, pn):
        c = clus[pn]
        fns, pt = {}, {}
        for nm in names:
            s = S(nm)["score"][c["ix"]]
            assert np.isfinite(s).all(), f"{nm}: non-finite scores on {pn}"
            fns[nm] = KH.WAuc(c["y"], s)
            pt[nm] = KH.auc_np(c["y"], s)
        bt = {nm: np.empty(BOOT_B) for nm in names}
        for b in range(BOOT_B):
            w = c["cnt"][b][c["inv"]]
            for nm in names:
                bt[nm][b] = fns[nm](w)
        return pt, bt, fns

    def diff(pt, bt, a, b):
        return {"point": pr(pt[a] - pt[b], 4), "ci95": [pr(x, 4) for x in KH.ci(bt[a] - bt[b])], "boot_se": pr(np.std(bt[a] - bt[b], ddof=1), 4)}

    names0 = ["A", "B", "C2", "C3", "C4", "C5", "B05", "Bnom", "X1"]
    jn0 = {n: f"{n}_s0_l{ '1' if n in ('C2','C3','X1') else '100'}" for n in names0}
    pt, bt, fns = auc_boot([jn0[n] for n in names0], "P")
    pt = {n: pt[jn0[n]] for n in names0}
    bt = {n: bt[jn0[n]] for n in names0}
    c = clus["P"]
    # ---- known-value controls ----
    auc_const = KH.auc_np(c["y"], np.zeros(len(c["y"])))
    wauc_ok = {n: bool(abs(fns[jn0[n]](np.ones(len(c["y"]))) - pt[n]) < 1e-9) for n in names0}
    repro = S("A_s0_l100")["ce_fit_oof"]
    gates = {"K_REPRO_A": {"oof_mean_ce_A_lam100": repro, "LH_FIT_INFO": LH_OOF_CE_LAM100, "abs_diff": abs(repro - LH_OOF_CE_LAM100), "pass": bool(abs(repro - LH_OOF_CE_LAM100) <= 1e-6)},
             "K_POP": prep_j["K_POP"], "K_ID": prep_j["K_ID"], "K_CONST_C1_auc": float(auc_const), "K_CONST_pass": bool(auc_const == 0.5), "K_WAUC_unit_weights": wauc_ok,
             "calibration": {"verdict": cal["verdict"], "anchor_rule": cal.get("anchor_rule"), "max_abs_diff": cal.get("anchor_rule_max_abs_diff"),
                             "mutation_fwd_neq_bwd_frac": cal["mutation_forward_vs_backward_a0"]["frac_rows_differ_gt_1e-6"]}}
    gates["ALL_PASS"] = bool(gates["K_REPRO_A"]["pass"] and prep_j["K_POP"]["pass"] and prep_j["K_ID"]["pass"] and gates["K_CONST_pass"] and all(wauc_ok.values())
                             and cal["verdict"] == "CALIBRATED")
    # ---- arms table ----
    arms = {}
    for n in names0:
        s = S(jn0[n])
        d_ = ARMS[n]
        arms[n] = {"auc": pr(pt[n]), "ci95": [pr(x) for x in KH.ci(bt[n])], "n_score_rows": int(len(c["ix"])), "n_score_episodes": int(c["n_ep"]), "n_fit_rows": int(len(d_u)),
                   "n_fit_episodes": int(len(np.unique(U["d_eid"]))), "d": s["d"], "lambda": d_[3], "oof_mean_ce_fit_rows": pr(s["ce_fit_oof"], 6), "lbfgs_iters_per_fold": s["n_iter"]}
    arms["C1"] = {"auc": float(auc_const), "n_score_rows": int(len(c["ix"])), "n_score_episodes": int(c["n_ep"]), "n_fit_rows": 0, "d": 0, "note": "one global constant score"}
    # fold-prior constant beside (the lh_fit 'constant' control as an AUC): per-fold class prior of the outer-train rows
    fold_d0 = LF.folds_of(U["d_eid"], 5, 0)
    e2f = {int(e): int(f) for e, f in zip(U["d_eid"], fold_d0)}
    fold_u = np.array([e2f[int(e)] for e in U["eid_u"]])
    sc_prior = np.zeros(nU)
    for k in range(5):
        pri = np.bincount(d_y[fold_d0 != k], minlength=6) / (fold_d0 != k).sum()
        pdc, pcr = pri[[2, 3, 4]].sum(), pri[[0, 1]].sum()
        sc_prior[fold_u == k] = pdc / (pdc + pcr)
    arms["C1_fold_prior_beside"] = {"auc": pr(KH.auc_np(c["y"], sc_prior[c["ix"]])), "note": "per-fold outer-train class prior renormalised; differs by fold, so not exactly 0.5 (between-fold level shift only)"}
    primary = diff(pt, bt, "B", "A")
    c4 = diff(pt, bt, "C4", "A")
    primary_clear = bool(primary["point"] >= BAR_D and primary["ci95"][0] > 0)
    c4_clear = bool(c4["point"] >= BAR_D and c4["ci95"][0] > 0)
    # ---- beside: fold-seed replicates (training-variance) ----
    reps = {}
    for s in (0, 1, 2, 3, 4):
        nm = {"A": f"A_s{s}_l100", "B": f"B_s{s}_l100", "C4": f"C4_s{s}_l100"}
        ptt, btt, _ = auc_boot(list(nm.values()), "P")
        reps[f"fold_seed_{s}"] = {"A_auc": pr(ptt[nm["A"]]), "B_auc": pr(ptt[nm["B"]]), "C4_auc": pr(ptt[nm["C4"]]), "B_minus_A": diff(ptt, btt, nm["B"], nm["A"]), "C4_minus_A": diff(ptt, btt, nm["C4"], nm["A"]),
                                  "B_minus_C4": diff(ptt, btt, nm["B"], nm["C4"]), "oof_ce_A": pr(S(nm["A"])["ce_fit_oof"], 5), "oof_ce_B": pr(S(nm["B"])["ce_fit_oof"], 5)}
    rep_d = [reps[f"fold_seed_{s}"]["B_minus_A"]["point"] for s in range(5)]
    rep_summary = {"B_minus_A_per_seed": rep_d, "mean": pr(np.mean(rep_d)), "min": pr(min(rep_d)), "max": pr(max(rep_d)), "sd_across_fold_seeds": pr(np.std(rep_d, ddof=1)),
                   "all_clear_bar": bool(min(rep_d) >= BAR_D), "C4_minus_A_per_seed": [reps[f"fold_seed_{s}"]["C4_minus_A"]["point"] for s in range(5)]}
    # ---- beside: lambda sensitivity, MLP, other contrasts, sensitivity populations ----
    beside = {}
    for tag in ("l10", "l100", "l1000", "mlp"):
        na, nb = f"A_s0_{tag}", f"B_s0_{tag}"
        ptt, btt, _ = auc_boot([na, nb], "P")
        beside[f"lambda_or_model_{tag}"] = {"A_auc": pr(ptt[na]), "B_auc": pr(ptt[nb]), "B_minus_A": diff(ptt, btt, nb, na), "oof_ce_A": pr(S(na)["ce_fit_oof"], 5), "oof_ce_B": pr(S(nb)["ce_fit_oof"], 5)}
    beside["contrasts_vs_A_seed0_lam100"] = {k: diff(pt, bt, k, "A") for k in ("C5", "B05", "Bnom", "C2", "C3", "X1")}
    beside["C5_minus_B"] = diff(pt, bt, "C5", "B")
    beside["B_minus_C4"] = diff(pt, bt, "B", "C4")
    beside["B_minus_X1"] = diff(pt, bt, "B", "X1")
    for pn in ("P_fit", "P_nocomp"):
        ptt, btt, _ = auc_boot([jn0["A"], jn0["B"], jn0["C4"]], pn)
        cc = clus[pn]
        beside[f"population_{pn}"] = {"n_rows": int(len(cc["ix"])), "n_decel": int(cc["y"].sum()), "n_episodes": int(cc["n_ep"]), "A_auc": pr(ptt[jn0["A"]]), "B_auc": pr(ptt[jn0["B"]]),
                                      "B_minus_A": diff(ptt, btt, jn0["B"], jn0["A"]), "C4_minus_A": diff(ptt, btt, jn0["C4"], jn0["A"])}
    coefB = S("B_s0_l100")["coef"]                                         # [5 folds, 6 classes, 2]
    beside["B_extra_feature_coefficients_std_units"] = {"classes": list(LON6), "a0_mean_over_folds": [pr(x, 4) for x in coefB[:, :, 0].mean(0)], "psi_dot0_mean_over_folds": [pr(x, 4) for x in coefB[:, :, 1].mean(0)],
                                                         "a0_sd_over_folds": [pr(x, 4) for x in coefB[:, :, 0].std(0)]}
    # ---- verdict ----
    if not gates["ALL_PASS"]:
        word = "INVALID (an instrument gate failed)"
    elif c4_clear:
        word = "VOID (C4, the within-episode permuted control, clears the bar)"
    elif primary_clear:
        word = "GO" if rep_summary["all_clear_bar"] else "GO-NOT-REPLICATED (primary clears; at least one fold-seed replicate does not clear +0.020)"
    else:
        word = "NO-GO"
    out = {"tool": "lha0_diag.py", "prereg": "PREREG_LHA0_DIAG.md", "prereg_sha256": sha256_file(ROOT / "PREREG_LHA0_DIAG.md"), "script_sha256": sha256_file(__file__),
           "tier": "T0 label-level diagnostic on FRESH-split out-of-fold heads over frozen z_tac (no planner, no plan metric)",
           "verdict": word,
           "bar": {"GO_iff": f"dAUC(B-A) >= +{BAR_D} AND paired episode-cluster 95% CI lower bound > 0", "VOID_iff": "C4 clears the same bar", "bootstrap": f"B {BOOT_B}, seed {BOOT_SEED}, paired, OOF scores held fixed"},
           "estimator_answers": "another draw of fresh EPISODES with the OOF scores fixed; blind to training (fold-fit) variance (see beside.fold_seed_replicates) and to inference variance (head deterministic)",
           "function_class": "ridge multinomial logistic (LH's frozen family), lambda 100 fixed, over standardised features: a NO-GO is 'no >= +0.020 gain for a LINEAR head', not 'a0 / psi_dot0 carry no decel information'",
           "primary": {"B_minus_A": primary, "clears_bar": primary_clear, "A_auc": arms["A"]["auc"], "B_auc": arms["B"]["auc"]},
           "controls": {"C1_constant_auc": float(auc_const), "C2_v0_only": arms["C2"], "C3_a0_only": arms["C3"], "C4_B_permuted_within_episode": {**arms["C4"], "C4_minus_A": c4, "clears_bar": c4_clear},
                        "C5_B_forward_a0_beside_only": {**arms["C5"], "C5_minus_A": beside["contrasts_vs_A_seed0_lam100"]["C5"], "C5_minus_B": beside["C5_minus_B"]}},
           "gates": gates, "arms": arms, "population": prep_j["population"], "calibration": cal,
           "beside": {"fold_seed_replicates": {"summary": rep_summary, "per_seed": reps}, **beside}, "inputs_sha256": prep_j["inputs_sha256"], "wall_s": round(time.time() - t0, 1)}
    (ROOT / "result").mkdir(exist_ok=True)
    tmp = ROOT / "result" / "LHA0_DIAG.json.tmp"
    tmp.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    os.replace(tmp, ROOT / "result" / "LHA0_DIAG.json")
    write_md(out)
    print(f"ZZASSEMBLEZZ {word} | A {arms['A']['auc']} B {arms['B']['auc']} dAUC {primary} | C4-A {c4} | gates {gates['ALL_PASS']}", flush=True)
    return out


def write_md(o: dict) -> None:
    f = lambda x: "n/a" if x is None else f"{x:+.4f}"
    ci_s = lambda c: f"[{c[0]:+.4f}, {c[1]:+.4f}]"
    L = []
    pa = o["primary"]["B_minus_A"]
    L += [f"# LHA0 diagnostic -- {o['verdict']}", "",
          f"**Tier:** {o['tier']}.  **Estimator:** {o['estimator_answers']}.", "",
          f"**Function class:** {o['function_class']}.", "",
          f"**Primary:** dAUC(B - A) = {f(pa['point'])}  95% CI {ci_s(pa['ci95'])}  (bar: >= +0.020 and CI lower bound > 0)  -> clears bar: {o['primary']['clears_bar']}.  A {o['primary']['A_auc']}  B {o['primary']['B_auc']}.", "",
          f"**C4 (permuted within episode):** dAUC(C4 - A) = {f(o['controls']['C4_B_permuted_within_episode']['C4_minus_A']['point'])}  {ci_s(o['controls']['C4_B_permuted_within_episode']['C4_minus_A']['ci95'])}  clears bar: {o['controls']['C4_B_permuted_within_episode']['clears_bar']}.", "",
          "## Arms (seed-0 LH folds, same fit rows, same scored rows)", "", "| arm | AUC | 95% CI (episode bootstrap) | d | lambda | n fit rows | n scored rows | scored episodes | OOF CE (fit rows) |", "|---|---|---|---|---|---|---|---|---|"]
    for k, a in o["arms"].items():
        if "ci95" in a:
            L.append(f"| {k} | {a['auc']} | {ci_s(a['ci95'])} | {a['d']} | {a['lambda']} | {a['n_fit_rows']} | {a['n_score_rows']} | {a['n_score_episodes']} | {a['oof_mean_ce_fit_rows']} |")
        else:
            L.append(f"| {k} | {a['auc']} | - | {a.get('d', '-')} | - | {a.get('n_fit_rows', '-')} | {a.get('n_score_rows', '-')} | {a.get('n_score_episodes', '-')} | - |")
    L += ["", "## Gates", "", "```", json.dumps(o["gates"], indent=1, default=float), "```", "", "## Population", "", "```", json.dumps(o["population"], indent=1, default=float), "```", "",
          "## Beside (never gating)", "", "```", json.dumps(o["beside"], indent=1, default=float), "```", ""]
    (ROOT / "result" / "LHA0_DIAG.md").write_text(chr(10).join(L), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "fit", "assemble", "all"])
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.stage in ("prep", "all"):
        prep()
    if a.stage in ("fit", "all"):
        fit_all(a.workers)
    if a.stage in ("assemble", "all"):
        assemble()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
