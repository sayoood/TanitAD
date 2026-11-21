"""l1_followup.py -- PREREG_L1_AMENDMENT3_FOLLOWUP.md (sealed f38506f0...): H3 (stack), CF2 / CF2s (cross-fitted on eval episodes),
D-shift.  CPU only; reuses l1_fit_score.py's loaders and helpers unchanged.

    python -X utf8 l1_followup.py --bank <dir> --out <dir>/followup
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l1_fit_score as F  # noqa: E402

AL, PH, PS, RT = F.AL, F.PH, F.PS, F.RT


def oof_scalar(X, H, eid, lam):
    f = F.folds_of(eid, 0)
    P = np.zeros_like(H)
    for j in range(F.NFOLD):
        tr, te = f != j, f == j
        P[te] = PS.ridge_pred(PS.ridge_fit(X[tr], H[tr], lam), X[te])
    return P


def cross_fit(E, Z, H, ks):
    """5 folds by EVAL episode (seed 0); each fold predicted by heads fit on the other folds (inner CV on those folds only)."""
    f = F.folds_of(E["eid"], 0)
    P2, P2s, info = np.zeros_like(H), np.zeros_like(H), []
    for j in range(F.NFOLD):
        tr, te = f != j, f == j
        (k, lam), _cv, _ = F.cv_head(E["X"][tr], Z[tr], H[tr], E["eid"][tr], 0, ks)
        m = F.head_fit(E["X"][tr], Z[tr], H[tr], k, lam)
        P2[te] = F.head_pred(m, E["X"][te], Z[te])
        ls, _ = PS.cv_lambda(E["X"][tr], H[tr], E["eid"][tr])
        P2s[te] = PS.ridge_pred(PS.ridge_fit(E["X"][tr], H[tr], ls), E["X"][te])
        info.append({"fold": j, "k": int(k), "lambda": float(lam), "lambda_s": float(ls), "n_fit": int(tr.sum()), "n_scored": int(te.sum())})
    return P2, P2s, info


def domain_auc(Ztr, eid_tr, Zev, eid_ev):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    Z = np.concatenate([Ztr, Zev])
    y = np.r_[np.zeros(len(Ztr)), np.ones(len(Zev))]
    eid = np.r_[np.array(["tr_" + e for e in eid_tr]), np.array(["ev_" + e for e in eid_ev])]
    f = F.folds_of(eid, 0)
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    import compliant_reread_r8 as CR
    import retime_trainfit_r8 as TF
    tools = RT.tools_load()
    keys = [m[1] for m in tools["PR"].METRICS]
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    bank = Path(a.bank)
    banks = {"eval0": F.load_bank(bank / "ztac_eval_s0.npz"), "eval1": F.load_bank(bank / "ztac_eval_s1.npz"),
             "train": F.load_bank(bank / "ztac_train_s0.npz")}
    surfs = ("S2", "S6")
    ev = {surf: {} for surf in surfs}
    first = {}
    for s in F.SEEDS:
        for surf in surfs:
            ev[surf][s] = F.load_eval(tools, CR, s, first, surf)
    _cp, CR2 = TF._patch_archive(TF.TRAIN_ROOT)
    ftr = {}
    trs = {surf: F.load_train(tools, CR2, TF, surf, ftr) for surf in surfs}
    gains_all = json.loads(F.FROZEN_PF.read_text(encoding="utf-8"))["surfaces"]
    res = {"tool": "l1_followup.py", "prereg": "PREREG_L1_AMENDMENT3_FOLLOWUP.md",
           "prereg_sha256": hashlib.sha256((F.ROOT / "PREREG_L1_AMENDMENT3_FOLLOWUP.md").read_bytes()).hexdigest(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "surfaces": {}}
    for surf in surfs:
        tr = trs[surf]
        Htr = AL.arc(tr["G"]) - AL.arc(tr["HP"])
        lam_s, _ = PS.cv_lambda(tr["X"], Htr, tr["eid"])
        m_s = PS.ridge_fit(tr["X"], Htr, lam_s)
        oof = oof_scalar(tr["X"], Htr, tr["eid"], lam_s)
        Otr, _ = F.join(banks["train"], tr["clip"], tr["ws"], "traj")
        Dtr = AL.arc(Otr[:, F.SLOTS[surf], :]) - AL.arc(tr["HP"])
        coef = []
        for k in range(Htr.shape[1]):
            A = np.stack([np.ones(len(Htr)), oof[:, k], Dtr[:, k]], 1)
            coef.append(np.linalg.lstsq(A, Htr[:, k], rcond=None)[0])
        coef = np.asarray(coef)
        Ztr, _ = F.join(banks["train"], tr["clip"], tr["ws"])
        R = {"H3_coef_a_alpha_beta": coef.tolist(), "lambda_s": lam_s, "seeds": {}}
        cf_cache = None
        for s in F.SEEDS:
            E = ev[surf][s]
            Z, _ = F.join(banks[f"eval{s}"], E["clip"], E["ws"])
            sp = AL.arc(E["HP"])
            Hev = AL.arc(E["G"]) - sp
            d_os = AL.arc(E["O"]) - sp
            hs = PS.ridge_pred(m_s, E["X"])
            h3 = coef[None, :, 0] + coef[None, :, 1] * hs + coef[None, :, 2] * d_os
            if cf_cache is None:                       # z_tac and the target are seed-independent (K6 = 0): fit once
                cf_cache = cross_fit(E, Z, Hev, F.KS)
            P2, P2s, info = cf_cache
            arms = {"H0": RT.retime_apply(E["O"], E["HP"], np.asarray(gains_all[surf]["all"]["RETIME"], np.float64)),
                    "H2s": PH.at_progress(E["O"], sp + hs), "H3": PH.at_progress(E["O"], sp + h3),
                    "CF2": PH.at_progress(E["O"], sp + P2), "CF2s": PH.at_progress(E["O"], sp + P2s)}
            np.savez_compressed(out_dir / f"arms_{surf}_s{s}.npz", clip=E["clip"], ws=E["ws"], eid=E["eid"],
                                **{f"arm_{k}": np.asarray(v, np.float32) for k, v in arms.items()})
            C = {k: F.comp(tools, P, E["G"], E["dt"]) for k, P in arms.items()}
            c_hp = F.comp(tools, E["HP"], E["G"], E["dt"])
            e1 = {k: F.cell(tools, C[k]["ade_m"], c_hp["ade_m"], E["eid"]) for k in arms}
            pc = {p: F.cell(tools, C[x]["ade_m"], C[y]["ade_m"], E["eid"]) for p, (x, y) in
                  {"H3-H0": ("H3", "H0"), "H3-H2s": ("H3", "H2s"), "CF2-CF2s": ("CF2", "CF2s"), "CF2s-H2s": ("CF2s", "H2s")}.items()}
            fam = {"H3-H0": {k: F.cell(tools, C["H3"][k], C["H0"][k], E["eid"]) for k in keys if k in C["H3"]}}
            R["seeds"][str(s)] = {"n_windows": int(len(sp)), "E1_minus_ha0_ext_pose": e1, "paired": pc, "four_families": fam,
                                  "eval_R2": {"H2s": F.r2(hs, Hev), "H3": F.r2(h3, Hev), "CF2": F.r2(P2, Hev), "CF2s": F.r2(P2s, Hev)}}
        R["cross_fit_folds"] = cf_cache[2]
        if surf == "S2":
            R["D_shift_domain_auc"] = domain_auc(Ztr, tr["eid"], *[F.join(banks["eval0"], ev[surf][0]["clip"], ev[surf][0]["ws"])[0]],
                                                 ev[surf][0]["eid"])
        res["surfaces"][surf] = R
        print(f"[l1-followup] {surf} done", flush=True)

    def sep_below(c):
        return c.get("hi") is not None and c["hi"] < 0

    S2 = res["surfaces"]["S2"]["seeds"]
    worse = []
    for k in S2["0"]["four_families"]["H3-H0"]:
        hb = k.startswith("TAC_") or k == "LON_speed_within_0p5"
        cs = [S2[s]["four_families"]["H3-H0"][k] for s in ("0", "1")]
        if all((c["hi"] < 0) if hb else (c["lo"] > 0) for c in cs):
            worse.append(k)
    stack = all(sep_below(S2[s]["paired"]["H3-H0"]) and sep_below(S2[s]["paired"]["H3-H2s"]) for s in ("0", "1")) and not worse
    mem = all(sep_below(S2[s]["paired"]["CF2-CF2s"]) for s in ("0", "1"))
    res["outcomes"] = {"STACK": "STACK-WINS" if stack else "STACK-NO", "H3_guard_worse_both": worse, "MEM": "MEM-YES" if mem else "MEM-NO"}
    res["wall_s"] = round(time.time() - t0, 1)
    (out_dir / "FOLLOWUP.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(f"[l1-followup] {res['outcomes']} wall {res['wall_s']}s ZZL1FOLLOWDONEZZ", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
