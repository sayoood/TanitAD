#!/usr/bin/env python3
"""agreement_core.py -- load the C3 table + the GT rasters and compute geoDAC for every proposal of every token.

Computes NO pick, NO PDMS and NO veto outcome: the unit of analysis is the PROPOSAL, pooled over all 64 proposals of all 1,058 tokens.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PK = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PK / "refe"))
import aux_label_store as A  # noqa: E402
import geodac as G  # noqa: E402

C3 = PK / "raw/2026-10-11-c3-selection-headroom"
STORE = Path("D:/Projects/TanitAD-artifacts/aux_drivable/label_store_navtest")


def load_c3():
    z = np.load(C3 / "table.npz", allow_pickle=False)
    tj = json.load(open(C3 / "tokens.json"))
    tok = [str(t) for t in z["token"]]
    logs = [tj["token_log"][t] for t in tok]
    return {"token": tok, "log": logs, "proposals": z["proposals"].astype(np.float64), "dac": z["sub"][:, :, 1].astype(np.int8),
            "stratum": [tj["stratum"][t] if isinstance(tj["stratum"], dict) else None for t in tok] if isinstance(tj.get("stratum"), dict) else None}


def gt_field(log: str, token: str) -> np.ndarray:
    return G.gt_field_m(A.read_label(STORE, log, token))


def run(mode: str = "bilinear", unknown: str = "drivable", fields: dict | None = None):
    """-> dict of arrays over [N tokens, 64 proposals]."""
    d = load_c3()
    N = len(d["token"])
    out = {k: [] for k in ("passed", "frac_outside", "min_margin_m", "first_bad_step", "n_unknown")}
    for i in range(N):
        f = fields[d["token"][i]] if fields is not None else gt_field(d["log"][i], d["token"][i])
        r = G.geodac(f, G.GT_GEOM, d["proposals"][i], mode=mode, unknown=unknown)
        out["passed"].append(r.passed)
        out["frac_outside"].append(r.frac_outside)
        out["min_margin_m"].append(r.min_margin_m)
        out["first_bad_step"].append(r.first_bad_step)
        out["n_unknown"].append(r.n_unknown)
    return d, {k: np.stack(v) for k, v in out.items()}


def confusion(geo_fail: np.ndarray, navsim_fail: np.ndarray) -> dict:
    """Positive class = VIOLATION (DAC = 0).  Both [..] bool."""
    g, n = geo_fail.astype(bool).ravel(), navsim_fail.astype(bool).ravel()
    tp, fp, fn, tn = int((g & n).sum()), int((g & ~n).sum()), int((~g & n).sum()), int((~g & ~n).sum())
    N = tp + fp + fn + tn
    po = (tp + tn) / N
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / N ** 2
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {"N": N, "tp_both_fail": tp, "fp_geo_fails_navsim_ok": fp, "fn_geo_ok_navsim_fails": fn, "tn_both_ok": tn,
            "agreement": po, "kappa": (po - pe) / (1 - pe) if pe < 1 else float("nan"), "mcc": (tp * tn - fp * fn) / den if den else float("nan"),
            "sensitivity_navsim_fail_found": tp / (tp + fn) if tp + fn else float("nan"), "specificity": tn / (tn + fp) if tn + fp else float("nan"),
            "ppv_geo_fail_is_navsim_fail": tp / (tp + fp) if tp + fp else float("nan"), "npv": tn / (tn + fn) if tn + fn else float("nan"),
            "base_rate_navsim_fail": (tp + fn) / N, "base_rate_geo_fail": (tp + fp) / N}
