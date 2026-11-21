"""Read the REAL bank2 s0 layout (shapes, dtypes, header keys, value ranges) on the dev box. Read-only."""
import json
import sys

sys.path[:0] = ["F:/TanitAD-scratch/recover_ft/tree/stack"]
import numpy as np
from tanitad.data import v8f_dagger as DG

B = "E:/Projects/TanitAD-artifacts/v8f-dagger2/bank_s0"
bank = DG.DaggerBank.load(B)
print("sha", bank.sha256, "n", bank.n, "hmax", bank.hmax, "partial", bank.header.get("partial"))
print("header keys", sorted(bank.header))
print("n clips in header", len(bank.header["clips"]), "first", list(bank.header["clips"].items())[:2])
for k, v in bank.a.items():
    v = np.asarray(v)
    fin = np.isfinite(v).mean() if v.dtype.kind == "f" else 1.0
    print(f"{k:8s} {str(v.dtype):8s} {str(v.shape):16s} finite {fin:.4f} min {np.nanmin(v) if v.size else None} max {np.nanmax(v) if v.size else None}")
ok = np.nonzero(bank.reason == 0)[0]
print("ok", len(ok), "reason counts", bank.reason_counts())
# structure of one ok record
i = ok[0]
print("rec j", bank.a["j"][i], "t0", bank.a["t0"][i], "tick", bank.a["tick"][i], "r", bank.a["r"][i])
print("pose last 3 rows", bank.a["pose"][i][-3:].tolist())
print("kin3", bank.a["kin3"][i].tolist())
print("NaN rows in pose history (first col) for ok recs: share of recs with any NaN in last 2 rows:", float(np.isnan(np.asarray(bank.a["pose"])[ok][:, -2:, :]).any((1, 2)).mean()))
print("j range", int(bank.a["j"][ok].min()), int(bank.a["j"][ok].max()), " j>=2 share", float((bank.a["j"][ok] >= 2).mean()))
# OWN-side only statistics of the table (the logged side needs the TRAIN store, which lives on box 2)
v = np.asarray(bank.a["kin3"])[ok, 0]
print("own v quantiles", np.quantile(v, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).round(3).tolist(), "own v in [3,8] share", float(((v >= 3) & (v <= 8)).mean()))
print("bank field e_lat (unsigned distance to the logged polyline, m; a proxy, NOT the card's signed same-clock-row dy) quantiles", np.quantile(np.asarray(bank.a["e_lat"])[ok], [0.05, 0.5, 0.95, 0.99]).round(3).tolist())
print("bank field dpsi (|heading error| at the MATCHED route row, DEGREES, unsigned; a proxy, NOT the card's signed same-clock-row dpsi in rad) quantiles", np.quantile(np.asarray(bank.a["dpsi"])[ok], [0.01, 0.05, 0.5, 0.95, 0.99]).round(4).tolist())
