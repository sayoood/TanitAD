"""Slow-regime q4c: the UNCHANGED q4c fit_clip() with the module constant V_MIN lowered, applied to clips that
have < 20 rows above 2 m/s (the 33 stationary ADD clips + the 12 old ones). Controls first, on clips whose
truth is the standard q4c entry (ALLTRAIN).
  S1  relaxed V_MIN on the UNMODIFIED old moving clips -> must reproduce ALLTRAIN
  S2  slow-only: speed channel zeroed where v >= 2.0 so ONLY rows with thr < v < 2.0 enter the fit -> truth = ALLTRAIN
  T   the targets (relaxed fit, unmodified poses)
PRE-REGISTERED acceptance (written before the run): threshold thr is admissible iff on S2 every clip has
|d grid_start| <= 1 ms and |d dt|*200 <= 1 ms; the smallest admissible thr of {0.1,0.2,0.3,0.5} is used for T."""
import hashlib, json, os, sys, time
from pathlib import Path
HOLD = Path("/home/nvidia/clockref203")
os.environ["AUDIT_TIP"] = "/home/nvidia/refcv6_run/fa43d9c5ce"; os.environ["AUDIT_MIN_FREE_GB"] = "25"; os.environ["OMP_NUM_THREADS"] = "4"
sys.path.insert(0, str(HOLD / "code"))
import numpy as np, pandas as pd, torch
import q4c_grid_vs_egolog as Q
import _common as C
Q.EGO_DIR = Path("/home/nvidia/data/obstacle/egomotion_alpamayo")
m = torch.load("/home/nvidia/data/refcv8-b1-416x1024-train-4572/_v2manifest.pt", map_location="cpu", weights_only=False)
ADD = [l.strip() for l in open(HOLD / "code/ADD_203_clip_ids.txt") if l.strip()]; addset = set(ADD)
ref = {r["clip"]: r for r in json.load(open(HOLD / "work/q4c_grid_vs_egolog_ALLTRAIN.json"))["splits"]["train"]["per_clip"]}
std_add = {r["clip"] for r in json.load(open(HOLD / "work/q4c_grid_vs_egolog_ADD203.json"))["splits"]["train"]["per_clip"]}
THRS = [0.1, 0.2, 0.3, 0.5]
t0 = time.time()
old_ok = [i for i, c in enumerate(m["clip_id"]) if c not in addset and C.sha12(c) in ref and ref[C.sha12(c)]["fit_resid_max_ms"] < 5.0]
egos = {}
def ego_of(c):
    return pd.read_parquet(Q.EGO_DIR / f"{c}.parquet")
ctl = {}
for thr in THRS:
    Q.V_MIN = thr
    d1, s2 = [], []
    for i in old_ok:
        c = m["clip_id"][i]; k = C.sha12(c); tr = ref[k]
        p = m["poses"][i].double().numpy(); ns = int(m["n_stack"][i]); eg = ego_of(c)
        f = Q.fit_clip(p, ns, eg)
        if f is not None:
            d1.append((abs(f["grid_start_on_label_timeline_s"] - tr["grid_start_on_label_timeline_s"]), abs(f["dt_s"] - tr["dt_s"]), f["fit_resid_max_ms"]))
        p2 = p.copy(); p2[p[:, 3] >= 2.0, 3] = 0.0
        f2 = Q.fit_clip(p2, ns, eg)
        if f2 is not None:
            s2.append((abs(f2["grid_start_on_label_timeline_s"] - tr["grid_start_on_label_timeline_s"]), abs(f2["dt_s"] - tr["dt_s"]), f2["fit_resid_max_ms"], f2["n_moving_rows"], k))
    def summ(x, ncols=3):
        a = np.array([r[:ncols] for r in x], float)
        return {"n": len(x), "max_d_g0_s": float(a[:, 0].max()), "p99_d_g0_s": float(np.quantile(a[:, 0], .99)), "max_d_dt_s": float(a[:, 1].max()),
                "max_dt_x200_s": float(a[:, 1].max() * 200), "max_resid_ms": float(a[:, 2].max())}
    ctl[str(thr)] = {"S1_full_poses_relaxed_vmin": summ(d1), "S2_slow_only": summ(s2),
                     "S2_worst_clip_sha12": max(s2, key=lambda r: r[0])[4],
                     "S2_admissible": bool(max(r[0] for r in s2) <= 1e-3 and max(r[1] for r in s2) * 200 <= 1e-3)}
    print("THR", thr, json.dumps(ctl[str(thr)]), round(time.time() - t0, 1), "s", flush=True)
adm = [t for t in THRS if ctl[str(t)]["S2_admissible"]]
print("ADMISSIBLE thresholds:", adm, flush=True)
thr_use = adm[0] if adm else None
res = {"what": "Q4c slow-regime: each cache row's true timestamp recovered from the 100 Hz egomotion log, V_MIN lowered (fit_clip UNCHANGED)",
       "evidence_class": "MEASURED (ours)", "ego_dir_n_files": len(list(Q.EGO_DIR.glob("*.parquet"))),
       "controls": {"S_regime_by_threshold": ctl, "admissible_thresholds": adm, "threshold_used": thr_use,
                    "rule": "admissible iff on S2 (old moving clips, fast rows masked out) every clip |d grid_start|<=1 ms and |d dt|*200<=1 ms vs the standard q4c entry"}}
rows, notfit = [], []
if thr_use is not None:
    Q.V_MIN = thr_use
    tg = [i for i, c in enumerate(m["clip_id"]) if (c in addset and C.sha12(c) not in std_add) or (c not in addset and C.sha12(c) not in ref)]
    for i in tg:
        c = m["clip_id"][i]; k = C.sha12(c)
        f = Q.fit_clip(m["poses"][i].double().numpy(), int(m["n_stack"][i]), ego_of(c))
        grp = "add" if c in addset else "old"
        if f is None:
            notfit.append({"clip": k, "grp": grp, "why": "fewer than 20 rows above %.2f m/s" % thr_use}); continue
        f["clip"] = k; f["T_out"] = int(m["poses"][i].shape[0]); f["grp"] = grp
        rows.append(f)
    res["splits"] = {"train": {"n_clips_requested": len(tg), "n_clips_fit": len(rows), "per_clip": rows}}
    res["not_fit"] = notfit
    print("TARGETS requested", len(tg), "fit", len(rows), "notfit", len(notfit), flush=True)
    for r in rows:
        print(r["grp"], r["clip"], "n_mov", r["n_moving_rows"], "resid_max_ms %.4f" % r["fit_resid_max_ms"], "g0 %.6f dt %.9f" % (r["grid_start_on_label_timeline_s"], r["dt_s"]), flush=True)
C.write_json("q4c_slowregime_raw.json", res)
print("DONE", round(time.time() - t0, 1), "s")
