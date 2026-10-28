"""PREREG_V8F_SPEEDSUB families: LONGITUDINAL and LATERAL paired contrasts on the SPEEDSUB T1-class dumps. A PARAMETRISED COPY of SPEEDREAD's raw/2026-10-28-speedread/families/families_longlat.py
(same row sources, same bootstrap, same controls LL-1..LL-3); the only changes are the environment switches below. CPU, read-only on the dump directory.
  SUB_DIR (default F:/TanitAD-scratch/speedsub/), SUB_PREFIX (default speedsub), SUB_ARM (default ego), SUB_REPORT (default <SUB_DIR>/subread/SUBREAD.json), SUB_CHECK (default ego,egoshuf,gt),
  SPEEDREAD_TREE (default F:/TanitAD-scratch/speedread/tree_f3971165, read-only). SMOKE on SPEEDREAD's own dumps: SUB_DIR=F:/TanitAD-scratch/speedread/ SUB_PREFIX=speedread SUB_ARM=read
  SUB_REPORT=<SPEEDREAD>/read/READ.json SUB_CHECK=read,gt,alt,const,perm must reproduce the banked longlat_s<s>.json read-own rows.
Original docstring follows.
LONGITUDINAL and LATERAL paired contrasts on the SPEEDREAD T1-class dumps (CPU, read-only on F:/TanitAD-scratch/speedread).
v8f_paired_families.py covers TACTICAL and STRATEGIC only (its docstring: the longitudinal and lateral paired blocks live in taniteval.four_families and
the readers). So this script REUSES the committed per-window row definitions and does not write new ones:
  comps(X, G)     from raw/2026-10-26-t1kin-hacc/t1k_read.py   (ADE2, ADE6, speed / along-track / accel MAE, heading / yaw-rate / curvature / cross-track MAE)
  lat_rows(X, G)  from raw/2026-10-26-t1-latcond/t1klc_read.py (L1 lateral-accel MAE, L2 curvature MAE with GT >= 5.2 m/s, L3 yaw-rate MAE with GT >= 5.2 m/s)
both read from tip e7ea20d1 by `git show` and executed by AST extraction (the way rr_lib.py does), paired with taniteval.ci.paired_episode_cluster_bootstrap
(B 2000, seed 0) on episodes. The GT and the episode ids are the paired tool's own context (v8f_paired_families.build_context), so the windows are its windows.
Controls (any failure: rc 3, the JSON is still written):
  LL-1  identity: comps(A, A) reads exactly 0.0 [0.0, 0.0] on every row;
  LL-2  the ADE6 row reproduces READ.json rows.<s>.<mode> (delta, lo, hi) to 4 decimals, for read, gt, alt, const and perm;
  LL-3  each arm's per-window rows reproduce the dump's own family point values (raw.families.longitudinal / lateral) to 1e-4.
python -B -I families_longlat.py <seed> <out json>"""
import ast
import hashlib
import json
import math
import os
import subprocess
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
W = os.environ.get("SUB_DIR", "F:/TanitAD-scratch/speedsub/").rstrip("/") + "/"
TREE = os.environ.get("SPEEDREAD_TREE", "F:/TanitAD-scratch/speedread/tree_f3971165").rstrip("/") + "/"
PREFIX, ARM = os.environ.get("SUB_PREFIX", "speedsub"), os.environ.get("SUB_ARM", "ego")
CHECK = os.environ.get("SUB_CHECK", "ego,egoshuf,gt").split(",")
REPORT = os.environ.get("SUB_REPORT", W + "subread/SUBREAD.json")
for p in (TREE + "taniteval", TREE + "stack", TREE + "stack/scripts"):
    sys.path.insert(0, p)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from taniteval import four_families as FF  # noqa: E402
from taniteval.ci import paired_episode_cluster_bootstrap as PB  # noqa: E402
import v8f_paired_families as PF  # noqa: E402

SEED = int(sys.argv[1])
OUT = sys.argv[2]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*", "-c", "core.autocrlf=false"]
REV = "e7ea20d1"
V8 = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
DT, VMIN = 0.2, 5.2
SRC = {"t1k_read.py": V8 + "raw/2026-10-26-t1kin-hacc/t1k_read.py", "t1klc_read.py": V8 + "raw/2026-10-26-t1-latcond/t1klc_read.py"}
BLOB = {}
for k, p in SRC.items():
    r = subprocess.run(GIT + ["show", f"{REV}:{p}"], capture_output=True)
    assert r.returncode == 0, p
    BLOB[k] = r.stdout
ns = {"np": np, "torch": torch, "FF": FF, "PB": PB, "math": math, "DT": DT, "VMIN": VMIN}


def exec_from(name, funcs):
    tree = ast.parse(BLOB[name].decode("utf-8").replace("\r\n", "\n"))
    got = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in funcs:
            exec(compile(ast.Module([node], []), name, "exec"), ns)
            got.append(node.name)
    assert sorted(got) == sorted(funcs), (name, got)


exec_from("t1k_read.py", ("comps",))
exec_from("t1klc_read.py", ("lat_rows",))
comps, lat_rows = ns["comps"], ns["lat_rows"]

ROWS = [("ADE2", "ade2_m", "ADE"), ("ADE6", "ade6_m", "ADE"),
        ("speed MAE", "speed_mae_mps", "LONGITUDINAL"), ("along-track MAE", "along_mae_m", "LONGITUDINAL"), ("accel MAE", "accel_mae_mps2", "LONGITUDINAL"),
        ("heading MAE", "heading_mae_deg", "LATERAL"), ("yaw-rate MAE", "yaw_rate_mae_degps", "LATERAL"), ("curvature MAE", "curvature_mae_1pm", "LATERAL"),
        ("cross-track MAE", "cross_mae_m", "LATERAL"), ("L1 lateral-accel MAE", "L1_lat_accel_mae_mps2", "LATERAL"),
        ("L2 curvature MAE (GT >= 5.2 m/s)", "L2_curvature_mae_fast_1pm", "LATERAL"), ("L3 yaw-rate MAE (GT >= 5.2 m/s)", "L3_yaw_rate_mae_fast_degps", "LATERAL")]


def rows_of(X, G):
    C = comps(X, G)
    pooled = C.pop("_pooled")
    C.update({k: v for k, v in lat_rows(X, G).items() if not k.startswith("_")})
    return C, pooled


def paired(a, b, eid):
    m = np.isfinite(a) & np.isfinite(b)
    r = PB(a[m], b[m], eid[m], n_boot=2000, seed=0)
    return {"delta": float(r["delta"]), "lo": float(r["lo"]), "hi": float(r["hi"]), "separated": bool(r["separated"]),
            "mean_a": float(np.mean(a[m])), "mean_b": float(np.mean(b[m])), "n_windows": int(m.sum()), "n_episodes": int(len(np.unique(eid[m])))}


E = W + f"eval_s{SEED}/eval_M2b-mixo_s{SEED}-{PREFIX}-"
ctx = PF.build_context(TREE + V8 + "raw/eval_grid_windows.json", TREE + V8 + "raw/eval141_kin.npz", TREE + V8 + "raw/join_eval141.npz", "yaw", "cycle")
gt = np.asarray(ctx["gt"], np.float64)
eid = np.asarray(ctx["eid"])
MODES_LOADED = tuple(dict.fromkeys(("own", ARM, "gt") + tuple(CHECK)))
dumps = {m: PF.load_dump(E + m + ".npz") for m in MODES_LOADED if m != "own" or True}
jsons = {m: json.load(open(E + m + ".json", encoding="utf-8")) for m in dumps}
R = json.load(open(REPORT, encoding="utf-8"))
rows, pooled = {}, {}
for m, d in dumps.items():
    assert np.array_equal(d["eid"], eid), m
    assert float(np.max(np.abs(np.linalg.norm(d["pred"].astype(np.float64) - gt, axis=-1).mean(1) - d["ade6"].astype(np.float64)))) <= 1e-6, m
    rows[m], pooled[m] = rows_of(d["pred"].astype(np.float64), gt)

res = {"schema": "speedsub-longlat-v1", "prefix": PREFIX, "arm": ARM, "seed": SEED, "contrast": "A - B, lower is better on every row",
       "tier": "T1-class open loop (own held outputs; no recorded future action); gt arm = T0 ceiling only",
       "estimator": "taniteval.ci.paired_episode_cluster_bootstrap, B=2000, seed 0, on episodes (141); answers 'another draw of EPISODES' only (H-ESTIM-SEED-1)",
       "row_sources": {k: {"path": SRC[k], "rev": REV, "sha256": hashlib.sha256(BLOB[k]).hexdigest()} for k in SRC},
       "taniteval_four_families_sha256": hashlib.sha256(open(TREE + "taniteval/taniteval/four_families.py", "rb").read()).hexdigest(),
       "context_sha256": ctx["_sha256"], "n_windows": int(len(eid)), "contrasts": {}, "controls": {}}
for a in dict.fromkeys((ARM, "gt")):
    res["contrasts"][f"{a}-own"] = {key: {"family": fam, "label": lab, **paired(rows[a][key], rows["own"][key], eid)} for lab, key, fam in ROWS}
# unavailable leaves, as the dumps themselves state them (n = 0)
dk = jsons[ARM]["raw"]["families"]["longitudinal"]
res["unavailable"] = {"distance_keeping": {"status": dk["distance_keeping"]["status"], "n": dk["distance_keeping"]["n"], "reason": dk["distance_keeping"]["reason"]},
                      "anti_echo": {"status": dk["anti_echo"]["status"], "n": dk["anti_echo"]["n"], "n_windows_available": dk["anti_echo"]["n_windows_available"],
                                    "reason": dk["anti_echo"]["reason"]},
                      "same_in_all_dumps_of_this_seed": all(
                          jsons[m]["raw"]["families"]["longitudinal"]["distance_keeping"]["status"] == "UNAVAILABLE"
                          and jsons[m]["raw"]["families"]["longitudinal"]["anti_echo"]["status"] == "UNAVAILABLE" for m in jsons)}
# LL-1 identity
ident = {key: paired(rows["own"][key], rows["own"][key], eid) for _, key, _ in ROWS}
bad = [k for k, r in ident.items() if not (r["delta"] == 0.0 and r["lo"] == 0.0 and r["hi"] == 0.0 and not r["separated"])]
res["controls"]["LL-1_identity"] = {"ok": not bad, "n_rows": len(ident), "not_zero": bad}
# LL-2 the ADE6 row against READ.json
ll2 = {}
for m in CHECK:
    p = paired(rows[m]["ade6_m"], rows["own"]["ade6_m"], eid)
    j = R["rows"][str(SEED)][m]
    ll2[m] = {"mine": [round(p["delta"], 4), round(p["lo"], 4), round(p["hi"], 4)], "READ.json": [j["delta"], j["lo"], j["hi"]],
              "ok": [round(p["delta"], 4), round(p["lo"], 4), round(p["hi"], 4)] == [j["delta"], j["lo"], j["hi"]]}
res["controls"]["LL-2_ade6_equals_READ_json"] = {"ok": all(v["ok"] for v in ll2.values()), "modes": ll2}
# LL-3 per-arm reproduction against the dump's own family point values
ll3, ok3 = [], True
for m in dumps:
    fam = jsons[m]["raw"]["families"]
    pairs = [("speed_mae_mps", float(np.mean(rows[m]["speed_mae_mps"])), fam["longitudinal"]["speed_mae_mps"]),
             ("along_mae_m", float(np.mean(rows[m]["along_mae_m"])), fam["longitudinal"]["along_mae_m"]),
             ("accel_mae_mps2", float(np.mean(rows[m]["accel_mae_mps2"])), fam["longitudinal"]["accel_mae_mps2"]),
             ("cross_mae_m", float(np.mean(rows[m]["cross_mae_m"])), fam["lateral"]["cross_mae_m"]),
             ("heading_mae_deg (pooled)", pooled[m]["heading_mae_deg"], fam["lateral"]["heading_mae_deg"]),
             ("yaw_rate_mae_degps (pooled)", pooled[m]["yaw_rate_mae_degps"], fam["lateral"]["yaw_rate_mae_degps"]),
             ("curvature_mae_1pm (pooled)", pooled[m]["curvature_mae_1pm"], fam["lateral"]["curvature_mae_1pm"])]
    for nm, mine, theirs in pairs:
        dev = abs(mine - theirs)
        ok = dev <= 1e-4
        ok3 &= ok
        ll3.append({"mode": m, "row": nm, "mine": mine, "dump": theirs, "abs_dev": dev, "ok": ok})
res["controls"]["LL-3_reproduces_dump_family_values"] = {"ok": ok3, "n_checked": len(ll3), "max_abs_dev": max(r["abs_dev"] for r in ll3), "checks": ll3}
res["controls_ok"] = all(c["ok"] for c in res["controls"].values())
json.dump(res, open(OUT, "w", encoding="utf-8", newline="\n"), indent=1)
print("controls_ok", res["controls_ok"], {k: v["ok"] for k, v in res["controls"].items()})
for k, r in res["contrasts"][f"{ARM}-own"].items():
    print(f"{k:28s} {r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] {'SEP' if r['separated'] else ''} n={r['n_windows']}")
sys.exit(0 if res["controls_ok"] else 3)
