"""lha0_posthoc.py -- POST-HOC, NOT PRE-REGISTERED, NON-GATING.  Written AFTER the LHA0 verdict (GO) was read; it exists because the verdict's own beside-row C3
(a0 ALONE, one feature) scored a far higher decel AUC than B (a0 added to the 513-d ridge), i.e. the B recipe leaves most of a0's information on the table.
Everything here is a function of the OOF scores / features the pre-registered run already produced; NO new head is fitted; NO parameter is fitted on the scored rows
(fusion is an equal-weight sum of label-free z-scored logits).  Same population P, same paired episode-cluster bootstrap (B 2000, seed 0, same draws).
Any use of these numbers as a bar needs its own pre-registration.  CPU only.  NEW file (LHA0 agent, 2026-10-26).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.dont_write_bytecode = True
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lha0_diag as L  # noqa: E402
import k_head as KH  # noqa: E402

ROOT = L.ROOT


def logit(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p) - np.log1p(-p)


def main() -> int:
    U = L._load_u()
    ix = np.flatnonzero(U["Pmask"])
    y = U["is_dec"][ix]
    ue, inv = np.unique(U["eid_u"][ix], return_inverse=True)
    rng = np.random.default_rng(L.BOOT_SEED)
    cnt = np.stack([np.bincount(rng.integers(0, len(ue), len(ue)), minlength=len(ue)).astype(np.float64) for _ in range(L.BOOT_B)])

    def oof(name):
        z = np.load(L.SCRATCH / "jobs" / f"{name}.npz")["logits"]
        s, _ = KH.renorm_decel(KH.softmax(np.nan_to_num(z, nan=0.0)))
        return s[ix]

    sc = {"A": oof("A_s0_l100"), "B": oof("B_s0_l100"), "C3": oof("C3_s0_l1"), "X1": oof("X1_s0_l1"), "C4": oof("C4_s0_l100")}
    sc["neg_a0"] = -U["f_a0"][ix]
    sc["neg_a0_05"] = -U["f_a0_05"][ix]
    sc["neg_a0_fwd_LEAK_CONTRAST"] = -U["f_a0_fwd"][ix]
    sc["neg_psi_abs"] = -np.abs(U["f_psi"][ix])
    sc["v0"] = U["v0u"][ix]

    def zl(s):
        l = logit(s)
        return (l - l.mean()) / l.std()

    for a, b in (("A", "C3"), ("B", "C3"), ("A", "X1")):
        sc[f"fuse_{a}+{b}"] = zl(sc[a]) + zl(sc[b])
    fns = {k: KH.WAuc(y, v) for k, v in sc.items()}
    pt = {k: KH.auc_np(y, v) for k, v in sc.items()}
    bt = {k: np.empty(L.BOOT_B) for k in sc}
    for b in range(L.BOOT_B):
        w = cnt[b][inv]
        for k, f in fns.items():
            bt[k][b] = f(w)

    def d(a, b):
        return {"point": round(float(pt[a] - pt[b]), 4), "ci95": [round(float(x), 4) for x in KH.ci(bt[a] - bt[b])]}

    # strata: AUC of -a0 within speed bands (is the a0 signal just 'moving vs stopped'?)
    v0 = U["v0u"][ix]
    strata = {}
    for lo, hi in ((0, 3), (3, 8), (8, 15), (15, 99)):
        m = (v0 >= lo) & (v0 < hi)
        if m.sum() > 50 and y[m].sum() > 10 and (~y[m]).sum() > 10:
            strata[f"v0 in [{lo},{hi})"] = {"n": int(m.sum()), "n_decel": int(y[m].sum()), "auc_neg_a0": round(KH.auc_np(y[m], sc["neg_a0"][m]), 4), "auc_A": round(KH.auc_np(y[m], sc["A"][m]), 4),
                                            "auc_B": round(KH.auc_np(y[m], sc["B"][m]), 4)}
    out = {"status": "POST-HOC, NOT PRE-REGISTERED, NON-GATING; no head fitted here; fusion = equal-weight sum of label-free z-scored logits (no parameter fitted on scored rows)",
           "population": {"n_rows": int(len(ix)), "n_decel": int(y.sum()), "n_episodes": int(len(ue))},
           "auc": {k: {"auc": round(pt[k], 4), "ci95": [round(float(x), 4) for x in KH.ci(bt[k])]} for k in sc},
           "contrasts": {"neg_a0_minus_A": d("neg_a0", "A"), "neg_a0_minus_B": d("neg_a0", "B"), "C3_minus_B": d("C3", "B"), "fuse_A+C3_minus_A": d("fuse_A+C3", "A"), "fuse_A+C3_minus_C3": d("fuse_A+C3", "C3"),
                         "fuse_A+C3_minus_B": d("fuse_A+C3", "B"), "fuse_B+C3_minus_C3": d("fuse_B+C3", "C3"), "fuse_A+X1_minus_A": d("fuse_A+X1", "A"),
                         "neg_a0_fwd_minus_neg_a0 (leak contrast)": d("neg_a0_fwd_LEAK_CONTRAST", "neg_a0")},
           "strata_by_v0_ms": strata}
    (ROOT / "result" / "LHA0_POSTHOC.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("ZZPOSTHOCZZ", json.dumps({k: v["auc"] for k, v in out["auc"].items()}), json.dumps(out["contrasts"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
