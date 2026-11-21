"""l1c_fit_score.py -- PREREG_L1C (+ AMENDMENTS 1, 2): fit the progress heads on the FRESH calibration split, score them on eval139 along refcv8's own path.
CPU only.  Reuses refcv9_l1/code/l1_fit_score.py (loaders, ridge/PCA head, CV, cells) and l1_followup.py (cross_fit, domain AUC) UNCHANGED.

Arms (PREREG_L1C section 2):  H2c  = PREREG_L1 H2 recipe (scalars + PCA_k(standardised z_tac), k and lambda by 5-fold EPISODE CV) fitted on the FRESH rows only;
H2cs = the scalar ridge (PREREG_L1 H2s recipe) fitted on the SAME fresh rows;  H0 = refcv8 + shipped stage (frozen per-frame RETIME table);
CF2 / CF2s = Amendment-3 cross-fit on EVAL episodes (recomputed with the same code => K2);  H2cm = H2c with eval z_tac deranged across episodes (V-m).
Nothing here is tuned on eval rows: every hyper-parameter is chosen by CV on the fresh rows.

    CUDA_VISIBLE_DEVICES=-1 R8_LAUNCH_TREE=<tree> PYTHONPATH='<tree>/stack;<tree>/taniteval;<battery>/code' python -X utf8 l1c_fit_score.py --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
L1CODE = "C:/Users/Admin/refcv8_pf/refcv9_l1/code"
sys.path.insert(0, L1CODE)
import l1_fit_score as F  # noqa: E402
import l1_followup as FU  # noqa: E402

AL, PH, PS, RT = F.AL, F.PH, F.PS, F.RT
F.DRY = False
SEEDS = F.SEEDS
SLOTS = F.SLOTS
L1_BANK = Path("C:/Users/Admin/refcv8_pf/refcv9_l1/bank")
L1_FOLLOWUP = Path("C:/Users/Admin/refcv8_pf/refcv9_l1/result/followup/FOLLOWUP.json")
LITERAL_K1 = {"0": -0.0343, "1": -0.0330}          # PREREG_L1C section 3
LITERAL_K2 = {"0": -0.0044, "1": -0.0045}
TOL = 1e-4


def sha12(c) -> str:
    return hashlib.sha256(str(c).encode()).hexdigest()[:12]


def sep_below(c):
    return c is not None and c.get("hi") is not None and c["hi"] < 0


# ------------------------------------------------------------------------------------------------------------- loading
def load_fresh(tools, CR2, TF, surf, first):
    """F.load_train + the clip sha12 of every kept row (needed for the r0 / rig / prefix subsets) + the decisions / validation PASS flags."""
    ctx = CR2.prepare_seed(TF.TRAIN_TREE, 0, tools, tools["rc"])
    S = TF.train_surface_inputs(tools, ctx, surf, first, TF.TRAIN_TREE)
    rows = ctx["panels"][surf]["rows"]
    lim = ctx["lim"][rows]
    m = np.asarray(S["mask"], bool)
    d = {"HP": np.asarray(S["X"]["HP"])[m], "G": np.asarray(S["X"]["G"])[m], "O": np.asarray(S["X"]["O"])[m], "X": PS.feats(S["dec"]["win"][m], lim[m]),
         "eid": np.asarray([str(e) for e in S["eid"]])[m], "clip": np.asarray(S["clip"], np.int64)[m],
         "ws": np.asarray(S["ws"], np.int64)[m], "v0": np.asarray(S["v0"], np.float64)[m], "dt": S["dt"],
         "sha12": np.asarray(S["clip_sha12"])[m], "lim": np.asarray(lim)[m], "n_rows_all": int(len(m)), "n_compliant": int(m.sum()),
         "sha12_all": np.asarray(S["clip_sha12"]),
         "checks": {"decisions_PASS": bool(S["dec"]["chk"]["PASS"]), "validation_PASS": bool(S["V"]["PASS"])}}
    del ctx
    return d


def sub(tr, Z, sel):
    out = {k: (v[sel] if isinstance(v, np.ndarray) and len(v) == len(sel) else v) for k, v in tr.items() if k not in ("sha12_all",)}
    return out, Z[sel]


def bank_sha_index(bank_path_dump):
    j = json.load(open(str(bank_path_dump) + "/refcv6_extras.npz.json", encoding="utf-8"))["clip_sha12_by_index"]
    return {int(k): v for k, v in j.items()}


# ------------------------------------------------------------------------------------------------------------- fitting
def fit_heads(tr, Ztr):
    Htr = AL.arc(tr["G"]) - AL.arc(tr["HP"])
    lam_s, _cvs = PS.cv_lambda(tr["X"], Htr, tr["eid"])
    m_s = PS.ridge_fit(tr["X"], Htr, lam_s)
    (k2, l2), cv2, oof2 = F.cv_head(tr["X"], Ztr, Htr, tr["eid"], 0, F.KS)
    m2 = F.head_fit(tr["X"], Ztr, Htr, k2, l2)
    return {"Htr": Htr, "lam_s": lam_s, "m_s": m_s, "k": int(k2), "lam": float(l2), "m2": m2, "oof2": oof2, "cv_sse_best": min(cv2.values())}


def r2cols(P, H):
    return F.r2(P, H)


def score_beside(tools, ev, banks, heads, gains, surf):
    """H2c-H2cs and H2c-H0 ADE cells at both seeds for an alternative fit (no bar; sensitivity)."""
    out = {}
    for s in SEEDS:
        E = ev[s]
        Z, _ = F.join(banks[f"eval{s}"], E["clip"], E["ws"])
        sp = AL.arc(E["HP"])
        P = {"H2c": PH.at_progress(E["O"], sp + F.head_pred(heads["m2"], E["X"], Z)),
             "H2cs": PH.at_progress(E["O"], sp + PS.ridge_pred(heads["m_s"], E["X"])),
             "H0": RT.retime_apply(E["O"], E["HP"], np.asarray(gains, np.float64))}
        C = {a: F.comp(tools, p, E["G"], E["dt"])["ade_m"] for a, p in P.items()}
        out[str(s)] = {"H2c-H2cs": F.cell(tools, C["H2c"], C["H2cs"], E["eid"]), "H2c-H0": F.cell(tools, C["H2c"], C["H0"], E["eid"]),
                       "H2cs-H0": F.cell(tools, C["H2cs"], C["H0"], E["eid"])}
    return out


def paired_domain_auc(Za, Zb, groups):
    """AUC of a PCA-32 logistic separating Za (class 0) from Zb (class 1) where row i of Za and row i of Zb are the SAME window: folds are by EPISODE and
    shared by both classes (so no paired window leaks across a fold)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    Z = np.concatenate([Za, Zb])
    y = np.r_[np.zeros(len(Za)), np.ones(len(Zb))]
    g = np.r_[groups, groups]
    f = F.folds_of(g, 0)
    s = np.zeros(len(y))
    for j in range(F.NFOLD):
        tr, te = f != j, f == j
        mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-9
        zs = (Z - mu) / sd
        _u, _s, vt = np.linalg.svd(zs[tr] - zs[tr].mean(0), full_matrices=False)
        sc = zs @ vt[:32].T
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc[tr], y[tr])
        s[te] = clf.predict_proba(sc[te])[:, 1]
    return float(roc_auc_score(y, s))


def d_platform(plat_bank, plat_dump, ship_bank, ship_dump):
    """D-platform (AMENDMENT 1 A1.5): rebuilt-on-this-box vs shipped z_tac / plan on the SAME windows, joined on (clip sha12, ws)."""
    pm, sm = bank_sha_index(plat_dump), bank_sha_index(ship_dump)
    pk = {(pm[int(e)], int(w)): j for j, (e, w) in enumerate(zip(plat_bank["e_i"], plat_bank["ws"]))}
    sk = {(sm[int(e)], int(w)): j for j, (e, w) in enumerate(zip(ship_bank["e_i"], ship_bank["ws"]))}
    common = sorted(set(pk) & set(sk))
    ip, is_ = np.array([pk[k] for k in common]), np.array([sk[k] for k in common])
    Zp, Zs = plat_bank["z"][ip], ship_bank["z"][is_]
    grp = np.array([k[0] for k in common])
    rel = np.linalg.norm(Zp - Zs, axis=1) / (np.linalg.norm(Zs, axis=1) + 1e-12)
    # reference scale: the typical distance between z_tac of two DIFFERENT shipped windows of the same clip count
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(Zs))
    rel_other = np.linalg.norm(Zs - Zs[perm], axis=1) / (np.linalg.norm(Zs, axis=1) + 1e-12)
    tp, ts = plat_bank["traj"][ip].astype(np.float64), ship_bank["traj"][is_].astype(np.float64)
    pd_ = np.linalg.norm(tp - ts, axis=-1).mean(-1)
    plat_clips = set(pm.values())
    res = {"n_common": int(len(common)), "n_plat": int(len(pk)), "n_ship_in_plat_clips": int(sum(1 for k in sk if k[0] in plat_clips)),
           "n_clips": int(len(set(grp))),
           "domain_auc_rebuilt_vs_shipped": paired_domain_auc(Zs, Zp, grp),
           "z_rel_diff": {"median": float(np.median(rel)), "p90": float(np.quantile(rel, 0.9)), "max": float(rel.max())},
           "z_rel_diff_between_random_shipped_windows_median": float(np.median(rel_other)),
           "plan_mean_slot_dist_m": {"median": float(np.median(pd_)), "p90": float(np.quantile(pd_, 0.9)), "max": float(pd_.max())},
           "plan_bit_equal_frac": float((pd_ == 0.0).mean())}
    res["platform_confounded_flag_auc_gt_0p70"] = bool(res["domain_auc_rebuilt_vs_shipped"] > 0.70)
    return res


def quant(x):
    x = np.asarray(x, np.float64)
    return {"n": int(len(x)), "q05": float(np.quantile(x, 0.05)), "q25": float(np.quantile(x, 0.25)), "median": float(np.median(x)),
            "q75": float(np.quantile(x, 0.75)), "q95": float(np.quantile(x, 0.95)), "mean": float(x.mean())}


# ------------------------------------------------------------------------------------------------------------- one surface
def run_surface(tools, surf, ev, tr, banks, gains, keys, outdir, meta, order_sha12, do_sens):
    res = {"surface": surf, "fit": {}, "seeds": {}, "controls": {}, "sensitivity": {}}
    Ztr, jtr = F.join(banks["fresh"], tr["clip"], tr["ws"])
    v0_gap = float(np.nanmax(np.abs(banks["fresh"]["v0_in"][jtr] - tr["v0"])))
    res["controls"]["K5_fresh_join"] = {"n_rows": int(len(Ztr)), "v0_bank_vs_panel_maxabs": v0_gap, "PASS": bool(v0_gap < 1e-3),
                                        "decisions_PASS": tr["checks"]["decisions_PASS"], "validation_PASS": tr["checks"]["validation_PASS"]}
    Ob = banks["fresh"]["traj"][jtr][:, SLOTS[surf], :]
    res["controls"]["K4b_fresh_bank_slots_vs_panel_os_maxabs_m"] = float(np.abs(Ob - tr["O"].astype(np.float64)).max())
    H = fit_heads(tr, Ztr)
    Htr, m_s, m2 = H["Htr"], H["m_s"], H["m2"]
    hc = Htr.mean(0)
    cgain = (H["oof2"] * Htr).sum(0) / (H["oof2"] * H["oof2"]).sum(0)
    res["fit"]["H2cs"] = {"lambda": H["lam_s"], "train_R2": r2cols(PS.ridge_pred(m_s, tr["X"]), Htr), "n": int(len(Htr)),
                          "n_episodes": int(len(np.unique(tr["eid"]))), "d_in": int(tr["X"].shape[1] - 1)}
    res["fit"]["H2c"] = {"k": H["k"], "lambda": H["lam"], "train_R2": r2cols(F.head_pred(m2, tr["X"], Ztr), Htr), "oof_R2": r2cols(H["oof2"], Htr),
                         "d_tac": int(Ztr.shape[1]), "n": int(len(Htr)), "n_episodes": int(len(np.unique(tr["eid"]))), "n_params": int(m2["W"].size),
                         "cv_sse_best": H["cv_sse_best"], "oof_gain_scalar": cgain.tolist()}
    res["fit"]["H0"] = {"gains": list(gains), "source": str(F.FROZEN_PF)}
    cf = None
    for s in SEEDS:
        E = ev[s]
        Z, _je = F.join(banks[f"eval{s}"], E["clip"], E["ws"])
        sp = AL.arc(E["HP"])
        Hev = AL.arc(E["G"]) - sp
        hh = {"H2cs": PS.ridge_pred(m_s, E["X"]), "CONST": np.repeat(hc[None], len(sp), 0), "H2c": F.head_pred(m2, E["X"], Z),
              "H2cm": F.head_pred(m2, E["X"], Z[F.derange(E["eid"])]),
              "H2c_nozero": F.head_pred(m2, E["X"], np.repeat(m2["zmu"][None], len(sp), 0))}
        if cf is None:                                  # z_tac and the target are seed-independent (L1 K6): cross-fit once, as l1_followup does
            cf = FU.cross_fit(E, Z, Hev, F.KS)
        arms = {"OS": E["O"], "H0": RT.retime_apply(E["O"], E["HP"], np.asarray(gains, np.float64)),
                "H0_G1": RT.retime_apply(E["O"], E["HP"], np.ones(len(gains))), "ZERO_PRIOR": PH.at_progress(E["HP"], sp)}
        for a, h in hh.items():
            arms[a] = PH.at_progress(E["O"], sp + h)
        arms["CF2"], arms["CF2s"] = PH.at_progress(E["O"], sp + cf[0]), PH.at_progress(E["O"], sp + cf[1])
        outdir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(outdir / f"arms_{surf}_s{s}.npz", clip=E["clip"], ws=E["ws"], eid=E["eid"], v0=E["v0"], G=E["G"],
                            **{f"arm_{k}": np.asarray(v, np.float32) for k, v in arms.items()})
        C = {a: F.comp(tools, P, E["G"], E["dt"]) for a, P in arms.items()}
        c_hp = F.comp(tools, E["HP"], E["G"], E["dt"])
        e1 = {a: F.cell(tools, C[a]["ade_m"], c_hp["ade_m"], E["eid"]) for a in arms}
        pairs = {"H2c-H2cs": ("H2c", "H2cs"), "H2c-H0": ("H2c", "H0"), "H2c-OS": ("H2c", "OS"), "H0-OS": ("H0", "OS"), "H2cm-H2cs": ("H2cm", "H2cs"),
                 "H2cm-H2c": ("H2cm", "H2c"), "H2cs-H0": ("H2cs", "H0"), "CONST-H2cs": ("CONST", "H2cs"), "CF2-CF2s": ("CF2", "CF2s"),
                 "CF2s-H2cs": ("CF2s", "H2cs"), "H2c-CF2": ("H2c", "CF2"), "CF2-H0": ("CF2", "H0")}
        pc = {p: F.cell(tools, C[a]["ade_m"], C[b]["ade_m"], E["eid"]) for p, (a, b) in pairs.items()}
        fam = {p: {k: F.cell(tools, C[a][k], C[b][k], E["eid"]) for k in keys if k in C[a]}
               for p, (a, b) in (("H2c-H0", ("H2c", "H0")), ("H2c-H2cs", ("H2c", "H2cs")), ("H2c-OS", ("H2c", "OS")), ("H0-OS", ("H0", "OS")))}
        res["seeds"][str(s)] = {"n_windows": int(len(sp)), "n_episodes": int(len(np.unique(E["eid"]))), "E1_minus_ha0_ext_pose": e1, "paired": pc,
                                "four_families": fam, "eval_R2": {a: F.r2(h, Hev) for a, h in hh.items()} | {"CF2": F.r2(cf[0], Hev), "CF2s": F.r2(cf[1], Hev)},
                                "mean_ade": {a: float(np.mean(C[a]["ade_m"])) for a in arms}}
        k3 = F.r2(hh["CONST"], Hev)
        res["controls"][f"s{s}"] = {
            "K2_arm_identity_g1_is_os_maxabs_m": float(np.abs(arms["H0_G1"].astype(np.float64) - E["O"].astype(np.float64)).max()),
            "K2_arm_identity_zero_prior_maxabs_m": float(np.abs(arms["ZERO_PRIOR"].astype(np.float64) - E["HP"].astype(np.float64)).max()),
            "K3_const_eval_R2": k3, "K3_PASS": bool(max(k3) <= 1e-12)}
        res["controls"][f"s{s}"]["K2_arm_identity_PASS"] = bool(res["controls"][f"s{s}"]["K2_arm_identity_g1_is_os_maxabs_m"] == 0.0
                                                                 and res["controls"][f"s{s}"]["K2_arm_identity_zero_prior_maxabs_m"] <= 1e-5)
        got = pc["CF2-CF2s"]["delta"]
        bk = None
        try:
            bk = json.loads(L1_FOLLOWUP.read_text(encoding="utf-8"))["surfaces"][surf]["seeds"][str(s)]["paired"]["CF2-CF2s"]["delta"]
        except Exception:      # noqa: BLE001
            bk = None
        res["controls"][f"s{s}"]["K2_CF2_minus_CF2s"] = {"got": got, "banked_L1_followup": bk, "literal_prereg": LITERAL_K2.get(str(s)) if surf == "S2" else None,
                                                         "PASS": bool(bk is not None and abs(got - bk) <= TOL and (surf != "S2" or abs(got - LITERAL_K2[str(s)]) <= TOL))}
    res["inference_seed_floor_os"] = F.cell(tools, F.comp(tools, ev[0]["O"], ev[0]["G"], ev[0]["dt"])["ade_m"],
                                            F.comp(tools, ev[1]["O"], ev[1]["G"], ev[1]["dt"])["ade_m"], ev[0]["eid"])
    # ---------------------------------------------------------------------------- sensitivities (BESIDE: no bar, never relabel the verdict)
    if do_sens:
        r0 = {k for k, v in meta.items() if v.get("r0_pass")}
        sets = {"S-r0 (r0-passing fresh clips)": r0, "S-rig A": {k for k, v in meta.items() if v.get("rig") == "A"},
                "S-rig B": {k for k, v in meta.items() if v.get("rig") == "B"}}
        for n in (75, 150, 225):
            sets[f"prefix {n} clips (sha order)"] = set(order_sha12[:n])
        for name, keep in sets.items():
            sel = np.isin(tr["sha12"], list(keep))
            if sel.sum() < 300 or len(np.unique(tr["eid"][sel])) < 15:
                res["sensitivity"][name] = {"skipped": f"only {int(sel.sum())} rows / {len(np.unique(tr['eid'][sel]))} episodes"}
                continue
            t2, Z2 = sub(tr, Ztr, sel)
            h2 = fit_heads(t2, Z2)
            res["sensitivity"][name] = {"n_rows": int(sel.sum()), "n_clips": int(len(np.unique(tr["sha12"][sel]))), "k": h2["k"], "lambda": h2["lam"],
                                        "lambda_s": h2["lam_s"], "seeds": score_beside(tools, ev, banks, h2, gains, surf)}
            print(f"[l1c] {surf} sensitivity {name} done", flush=True)
    return res, Ztr


# ------------------------------------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--bank-fresh", default="F:/l1c/bank")
    ap.add_argument("--tree-root", default="F:/l1c/tree")
    ap.add_argument("--tree", default="step48852ema_fresh")
    ap.add_argument("--plat-tree", default="step48852ema_plat")
    ap.add_argument("--ids", default="F:/l1c/select/final_ids_fresh.json")
    ap.add_argument("--fetch-manifest", default="F:/l1c/select/fetch_manifest.jsonl")
    ap.add_argument("--surfaces", default="S2,S6")
    ap.add_argument("--no-sens", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    import compliant_reread_r8 as CR
    import retime_trainfit_r8 as TF
    tools = RT.tools_load()
    keys = [m[1] for m in tools["PR"].METRICS]
    surfs = [x for x in a.surfaces.split(",") if x]
    out = {"tool": "l1c_fit_score.py", "generated": time.strftime("%FT%T%z"), "script_sha256": F.sha256_file(__file__),
           "prereg_sha256": F.sha256_file(ROOT / "PREREG_L1C.md"), "l1_fit_score_sha256": F.sha256_file(L1CODE + "/l1_fit_score.py"),
           "l1_followup_sha256": F.sha256_file(L1CODE + "/l1_followup.py"), "tree": F.TREE,
           "tier": "T1 self-action open loop on logged eval139 frames; arms are post-hoc emulations of the in-forward re-timing of refcv8's post-cap emitted path "
                   "(cap-binding windows' in-forward difference UNMEASURED)",
           "estimator": "pair_read_r8.paired_cell -> taniteval.ci.paired_episode_cluster_bootstrap B 2000 seed 0 (episodes only, models fixed; blind to training "
                        "variance H-ESTIM-SEED-1 and to inference variance)",
           "frozen_pf_table_sha256": F.sha256_file(F.FROZEN_PF), "branch": "PRIMARY (v_lim in fit and scoring)", "surfaces": {}, "bank_controls": {}}
    ids = json.load(open(a.ids, encoding="utf-8"))
    order = [sha12(r["clip_id"]) for r in ids]
    meta = {}
    for l in open(a.fetch_manifest, encoding="utf-8"):
        r = json.loads(l)
        meta[r["sha12"]] = {"rig": r["rig"], "r0_pass": bool(r["r0_pass"]), "r0_mean_speed": r["r0_mean_speed"], "chunk": r["chunk"]}
    meta = {k: v for k, v in meta.items() if k in set(order)}
    out["population"] = {"n_ids_split": len(order), "rig_counts": {g: sum(1 for v in meta.values() if v["rig"] == g) for g in ("A", "B")},
                         "r0_pass": int(sum(v["r0_pass"] for v in meta.values()))}
    ev = {surf: {} for surf in surfs}
    first = {}
    for s in SEEDS:
        for surf in surfs:
            ev[surf][s] = F.load_eval(tools, CR, s, first, surf)
    TF.TRAIN_ROOT, TF.TRAIN_TREE = a.tree_root, a.tree
    _cp, CR2 = TF._patch_archive(TF.TRAIN_ROOT)
    ftr = {}
    trs = {surf: load_fresh(tools, CR2, TF, surf, ftr) for surf in surfs}
    bank = Path(a.bank_fresh)
    banks = {"eval0": F.load_bank(L1_BANK / "ztac_eval_s0.npz"), "eval1": F.load_bank(L1_BANK / "ztac_eval_s1.npz"),
             "fresh": F.load_bank(bank / "ztac_fresh.npz")}
    out["population"].update({"fresh_clips_in_tree": int(len(set(trs["S2"]["sha12_all"]))), "fresh_rows_all_S2": trs["S2"]["n_rows_all"],
                              "fresh_rows_compliant_S2": trs["S2"]["n_compliant"], "fresh_episodes_compliant_S2": int(len(np.unique(trs["S2"]["eid"]))),
                              "fresh_rows_all_S6": trs.get("S6", {}).get("n_rows_all"), "fresh_rows_compliant_S6": trs.get("S6", {}).get("n_compliant"),
                              "eval_S2": {str(s): {"n_windows": int(len(ev["S2"][s]["eid"])), "n_episodes": int(len(np.unique(ev["S2"][s]["eid"])))} for s in SEEDS},
                              "fresh_bank_meta": banks["fresh"]["meta"]})
    # ---- bank controls: K4 (eval banks, as L1) and the FRESH roll's identity: bank plan == the roll's own os plan (0.0 m)
    for s in SEEDS:
        b = banks[f"eval{s}"]
        ref = (F.OSONLY_S0 if s == 0 else F.ARCH / F.TREE / f"dump_s{s}" / "refcv6_extras.npz")
        ex = np.load(ref)
        kx = list(zip(ex["clip_index"].astype(int), ex["ws"].astype(int)))
        jb = np.array([b["idx"].get(k, -1) for k in kx])
        ok_align = bool((jb >= 0).all() and len(b["idx"]) == len(kx))
        dmax = float(np.abs(b["traj"][jb] - ex["nav_true.traj"].astype(np.float64)).max()) if ok_align else float("nan")
        out["bank_controls"][f"eval_s{s}"] = {"alignment_equal": ok_align, "n_bank": len(b["idx"]), "n_dump": len(kx), "own_vs_dump_maxabs_m": dmax,
                                              "PASS": bool(ok_align and dmax <= 1e-5)}
    fx = np.load(f"{a.tree_root}/{a.tree}/dump_s0/refcv6_extras.npz")
    fb = banks["fresh"]
    kx = list(zip(fx["clip_index"].astype(int), fx["ws"].astype(int)))
    jb = np.array([fb["idx"].get(k, -1) for k in kx])
    ok_align = bool((jb >= 0).all() and len(fb["idx"]) == len(kx))
    d4 = float(np.abs(fb["traj"][jb] - fx["nav_true.traj"].astype(np.float64)).max()) if ok_align else float("nan")
    out["bank_controls"]["K4_fresh"] = {"alignment_equal": ok_align, "n_bank": len(fb["idx"]), "n_dump": len(kx), "bank_plan_vs_roll_os_plan_maxabs_m": d4,
                                        "PASS": bool(ok_align and d4 <= 1e-5), "meta_final": bool(fb["meta"].get("final")),
                                        "bank_clean_finite": bool(np.isfinite(fb["z"]).all() and (np.abs(fb["z"]).sum(1) > 0).all())}
    z0, z1 = banks["eval0"], banks["eval1"]
    common = sorted(set(z0["idx"]) & set(z1["idx"]))
    dz = np.abs(z0["z"][[z0["idx"][k] for k in common]] - z1["z"][[z1["idx"][k] for k in common]])
    out["bank_controls"]["K6_eval_ztac_seed_dependence"] = {"n": len(common), "max_abs_over_rms": float(dz.max() / np.sqrt((z0["z"] ** 2).mean()))}
    gains_all = json.loads(F.FROZEN_PF.read_text(encoding="utf-8"))["surfaces"]
    Zfresh = {}
    outdir = Path(a.out)
    for surf in surfs:
        print(f"[l1c] fitting + scoring {surf} ...", flush=True)
        out["surfaces"][surf], Zfresh[surf] = run_surface(tools, surf, ev[surf], trs[surf], banks, gains_all[surf]["all"]["RETIME"], keys, outdir, meta, order,
                                                          do_sens=(surf == "S2" and not a.no_sens))
    # ---- K1: H0 reproduces the banked per-surface RETIME_TRAINFIT cell on this tree AND the prereg literals
    bt = json.loads(F.BANKED_TF.read_text(encoding="utf-8"))
    k1 = {}
    for surf in surfs:
        for s in SEEDS:
            try:
                want = bt["seeds"][str(s)]["surfaces"][surf]["sets"]["compliant"]["arms"]["RETIME_TRAINFIT"]["minus_ha0_ext_pose"]["delta"]
            except Exception as e:      # noqa: BLE001
                want = f"unreadable: {e!r}"
            got = out["surfaces"][surf]["seeds"][str(s)]["E1_minus_ha0_ext_pose"]["H0"]["delta"]
            lit = LITERAL_K1[str(s)] if surf == "S2" else None
            k1[f"{surf}|s{s}"] = {"got": got, "banked": want, "literal_prereg": lit,
                                  "PASS": bool(isinstance(want, float) and abs(got - want) <= TOL and (lit is None or abs(got - lit) <= TOL))}
    out["controls_K1"] = k1
    # ---- D-fresh (diagnostic, no bar): domain AUC fresh vs eval139 z_tac, PCA-32 logistic, episode CV (the L1 D-shift recipe)
    E0 = ev["S2"][0]
    Zev, _ = F.join(banks["eval0"], E0["clip"], E0["ws"])
    tr2 = trs["S2"]
    out["D_fresh"] = {"domain_auc_compliant_rows": FU.domain_auc(Zfresh["S2"], tr2["eid"], Zev, E0["eid"]), "flag_threshold": 0.70,
                      "n_fresh": int(len(Zfresh["S2"])), "n_eval": int(len(Zev)),
                      "speed_v0_ms": {"fresh": quant(tr2["v0"]), "eval139": quant(E0["v0"])},
                      "fed_ceiling_ms_finite_fresh": quant(tr2["lim"][np.isfinite(tr2["lim"])]),
                      "frac_v0_gt_14ms": {"fresh": float((tr2["v0"] > 14.0).mean()), "eval139": float((E0["v0"] > 14.0).mean())}}
    out["D_fresh"]["flagged"] = bool(out["D_fresh"]["domain_auc_compliant_rows"] > 0.70)
    # ---- D-platform (AMENDMENT 1 A1.5)
    pb = bank / "ztac_plat.npz"
    if pb.exists():
        plat_bank = F.load_bank(pb)
        out["D_platform"] = d_platform(plat_bank, f"{a.tree_root}/{a.plat_tree}/dump_s0", banks["eval0"], str(L1_BANK / "eval_dump_s0"))
    else:
        out["D_platform"] = {"missing": "ztac_plat.npz not present"}
    out["D_fresh"]["platform_confounded_reading"] = bool(out["D_platform"].get("platform_confounded_flag_auc_gt_0p70", False)) and out["D_fresh"]["flagged"]
    out["wall_s"] = round(time.time() - t0, 1)
    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    (o / "L1C_RESULT.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(f"[l1c] WROTE {o / 'L1C_RESULT.json'} wall {out['wall_s']}s ZZL1CFITDONEZZ", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
