"""PRIVGEOM-SAFE, POST-HOC DESCRIPTIVE (no authority over any branch; not pinned): why is the registered C-NULL's Delta_F negative (about -0.017 NC, -0.062 DAC) instead of about 0?

HYPOTHESIS (stated before this run): the within-set permutation keeps each set's G distribution, so the G block still learns the BETWEEN-set association (sets with many boundary-hugging proposals have more unsafe proposals) and then applies it
per proposal, which is pure within-set noise.  Test: permute exactly as C-NULL rep 0 does, then subtract each set's mean over its 64 proposals from every column (this removes the between-set information and nothing else), refit L+G, seed 0.
Expected under the hypothesis: Delta_F about 0 (inside +-0.005).  Writes run/descr/null_centering_<row>.json.   usage: pg_descr_null_centering.py <nc|dac>

SECOND MODE (also post-hoc, no authority): `pg_descr_null_centering.py real <ncG|ncGV|dacG|dacGV|dacGVnomin>` refits the REAL arm with each set's mean removed from every column (within-set-centred G), seed 0, and reports Delta_F vs L
beside the registered (uncentred) arm.  `dacGVnomin` = the DAC-nosign companion without |dac_clear_min| (the 9 key-step |clearances|).  Writes run/descr/centred_real_<variant>.json.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                                                                       # noqa: E402

import pg_lib as P                                                                       # noqa: E402
import pg_run as R                                                                       # noqa: E402

FL = P.FL


def real_mode(variant: str) -> int:
    row = "nc" if variant.startswith("nc") else "dac"
    kind = "G" if variant.endswith("G") else "GV"
    R.require_prereg()
    R.require_token()
    A = R.load_join(labels=True)
    an = P.Analyzer(A["lab"], A["code"], A["log_idx"], A["n_logs"])
    Pth = R.paths()
    col = R.LABEL_COL[row]
    g = R.g_for(A, row, kind).astype(np.float64)
    if variant.endswith("nomin"):
        g = g[..., 1:]
    g = np.ascontiguousarray((g - g.mean(axis=1, keepdims=True)).astype(np.float32))
    z_st, y = np.ascontiguousarray(A["z6"][..., col]), np.ascontiguousarray(A["lab"][..., col])
    oof = P.oof_logits("LG", A["feat"], z_st, y, g, A["fold"], 0, A["p2_W6"][col], float(A["p2_b6"][col]), ram_guard=lambda: FL.wait_for_ram(3.0, log=R.log))
    reg = f"LG_{row}_s0" if kind == "G" else f"LGV_{row}_s0"
    sL = an.per_log_stats(np.load(f"{Pth['fits']}/L_{row}_s0.npy"), row)
    sR = an.per_log_stats(np.load(f"{Pth['fits']}/{reg}.npy"), row)
    sC = an.per_log_stats(oof, row)
    out = {"variant": variant, "row": row, "seed": 0, "authority": "DESCRIPTIVE, post-hoc", "k": int(g.shape[-1]), "registered_arm": reg, "registered_delta_F": an.diff(sR, sL, "Q4") if not variant.endswith("nomin") else None,
           "centred_delta_F": an.diff(sC, sL, "Q4"), "centred_F_auc": an.auc(sC, "Q4"), "centred_delta_Q123": an.diff(sC, sL, "Q123")}
    os.makedirs(f"{R.ROOT}/run/descr", exist_ok=True)
    R.jdump(out, f"{R.ROOT}/run/descr/centred_real_{variant}.json")
    f4 = lambda c: "n/a" if c is None else f"{c['point']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]"   # noqa: E731
    print(variant, "registered:", f4(out["registered_delta_F"]), " | centred:", f4(out["centred_delta_F"]), " | centred Q1-3:", f4(out["centred_delta_Q123"]))
    return 0


def main() -> int:
    if sys.argv[1] == "real":
        return real_mode(sys.argv[2])
    row = sys.argv[1]
    assert row in ("nc", "dac")
    R.require_prereg()
    R.require_token()
    A = R.load_join(labels=True)
    an = P.Analyzer(A["lab"], A["code"], A["log_idx"], A["n_logs"])
    Pth = R.paths()
    col = R.LABEL_COL[row]
    g = R.g_for(A, row, "null", rep=0).astype(np.float64)                                   # exactly C-NULL rep 0
    g = np.ascontiguousarray((g - g.mean(axis=1, keepdims=True)).astype(np.float32))        # remove the between-set information
    z_st, y = np.ascontiguousarray(A["z6"][..., col]), np.ascontiguousarray(A["lab"][..., col])
    oof = P.oof_logits("LG", A["feat"], z_st, y, g, A["fold"], 0, A["p2_W6"][col], float(A["p2_b6"][col]), ram_guard=lambda: FL.wait_for_ram(3.0, log=R.log))
    sL = an.per_log_stats(np.load(f"{Pth['fits']}/L_{row}_s0.npy"), row)
    sN = an.per_log_stats(np.load(f"{Pth['fits']}/null0_{row}_s0.npy"), row)
    sC = an.per_log_stats(oof, row)
    out = {"row": row, "seed": 0, "authority": "DESCRIPTIVE, post-hoc", "registered_null0_delta_F": an.diff(sN, sL, "Q4"), "centred_null0_delta_F": an.diff(sC, sL, "Q4"), "centred_null0_F_auc": an.auc(sC, "Q4")}
    os.makedirs(f"{R.ROOT}/run/descr", exist_ok=True)
    R.jdump(out, f"{R.ROOT}/run/descr/null_centering_{row}.json")
    f4 = lambda c: f"{c['point']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]"                  # noqa: E731
    print(row, "registered null0:", f4(out["registered_null0_delta_F"]), " | centred null0:", f4(out["centred_null0_delta_F"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
