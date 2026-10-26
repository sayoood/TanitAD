"""d2_posthoc.py -- POST-HOC, NOT PRE-REGISTERED, NON-GATING.  Written AFTER the sealed K-HEAD-D2 verdict (NO-GO) was read.  It answers one question for the Master Mind's next-lever choice:
"why did a0 not carry over, and is the beside-row FUSION (z(D2) + z(LH), no fitted weight) a stable pattern or an eval139 accident?"
  * rebuilds the sealed run's scores through d2_khead's own functions and ASSERTS they reproduce the sealed AUCs (so nothing here is a different instrument);
  * eval139 AUC by v0 band for LH / D2 / -a0 / fusion, next to the fresh numbers of LHA0_POSTHOC;
  * correlation of the D2 and LH logits on eval and on fresh OOF; fusion on the FRESH OOF (fresh-only new compute) vs LH_OOF and vs D2 -- a consistency read, never a bar.
Any use of these numbers as a bar needs its own pre-registration, on windows not already seen.  CPU only.  No clip id is printed or written.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d2_common as D  # noqa: E402
import d2_khead as K  # noqa: E402
import numpy as np  # noqa: E402


def main() -> int:
    import k_head as KH
    reg = K.check_registration()
    sealed = json.loads((D.ROOT / "result" / "LHA0_D2_KHEAD.json").read_text(encoding="utf-8"))
    P = K.load_population()
    zb, jz, y, eid = P["zb"], P["jz"], P["y"], P["eid"]
    a0, psi, fb = K.eval_features(zb)
    A0, PSI, V0 = a0[jz], psi[jz], P["v0"][jz]
    H = D.load_frozen(D.ROOT / "D2_FROZEN.json", reg["pins"]["D2_FROZEN.json"])
    s_d2, lh_lo = D.head_logit(H, A0, V0, PSI), D.lh_logit(P["lg"])
    fuse = D.zscore(s_d2) + D.zscore(lh_lo)
    sc = {"LH": P["s_lh"], "D2": s_d2, "neg_a0": -A0, "fuse": fuse}
    assert abs(KH.auc_np(y, sc["D2"]) - sealed["arms_full_population"]["D2"]["auc"]) < 6e-5 and abs(KH.auc_np(y, sc["fuse"]) - sealed["arms_full_population"]["fuse_D2_LH"]["auc"]) < 6e-5
    strata = {}
    for lo, hi in ((0, 3), (3, 8), (8, 15), (15, 99)):
        m = (V0 >= lo) & (V0 < hi)
        if m.sum() > 50 and y[m].sum() > 10 and (~y[m]).sum() > 10:
            strata[f"v0 in [{lo},{hi})"] = {"n": int(m.sum()), "n_decel": int(y[m].sum()), **{f"auc_{k}": D.pr(KH.auc_np(y[m], v[m])) for k, v in sc.items()}}
    from scipy.stats import spearmanr
    rho_eval = float(spearmanr(s_d2, lh_lo)[0])
    # ---- fresh OOF consistency (fresh-only compute) ----
    z = np.load(D.ROOT / "result" / "d2_fresh_oof.npz")
    Pf, yf, eu = z["P"], z["y"].astype(bool), z["eid_u"]
    lg_lh = np.load(D.LHA0 / "result" / "oof_logits" / "A_s0_l100.npz")["logits"]
    d2f, lhf = z["d2_logit_oof"][Pf], D.lh_logit(lg_lh)[Pf]
    ff = D.zscore(d2f) + D.zscore(lhf)
    Rf = D.paired_boot(yf, eu[Pf], {"LH_OOF": lhf, "D2": d2f, "fuse": ff, "neg_a0": -np.load(D.UNIVERSE)["f_a0"][Pf]})
    rho_fresh = float(spearmanr(d2f, lhf)[0])
    out = {"status": "POST-HOC, NOT PRE-REGISTERED, NON-GATING (written after the sealed NO-GO was read); scores rebuilt through d2_khead's own functions and asserted equal to the sealed AUCs",
           "eval139_strata_by_v0_ms": strata, "spearman_D2_vs_LH_logit": {"eval139": D.pr(rho_eval), "fresh_OOF": D.pr(rho_fresh)},
           "fresh_OOF_fusion": {"n": Rf["n"], "episodes": Rf["n_episodes"], "auc": {k: D.pr(v) for k, v in Rf["pt"].items()}, "fuse_minus_LH_OOF": D.diff(Rf, "fuse", "LH_OOF"),
                                "fuse_minus_D2": D.diff(Rf, "fuse", "D2"), "fuse_minus_neg_a0": D.diff(Rf, "fuse", "neg_a0")},
           "eval139_reference_from_sealed_run": {"fuse_minus_LH": sealed["paired_vs_LH_full"]["fuse_D2_LH"], "fuse_minus_D2": sealed["contrasts_full"]["fuse_minus_D2"]},
           "fresh_reference_from_LHA0_POSTHOC": {"neg_a0_AUC": 0.7699, "A_AUC": 0.6699, "fuse_A+C3_minus_C3": {"point": -0.0006, "ci95": [-0.038, 0.0341]}}}
    (D.ROOT / "result" / "LHA0_D2_POSTHOC.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("ZZPOSTHOCZZ", json.dumps(out)[:1800], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
