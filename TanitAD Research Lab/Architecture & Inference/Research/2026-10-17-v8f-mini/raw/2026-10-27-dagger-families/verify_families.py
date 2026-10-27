#!/usr/bin/env python3
"""Independent checks on FAMILIES_DAG.md: (1) every table cell in the document is the formatted value of a row in DAG_FAMILIES_ROWS.json, (2) a sample of rows is
recomputed DIRECTLY from the raw npz with the estimator (not through the driver's wrapper), (3) semantic re-derivations (separated == interval excludes 0, the
worse-than list, vocabulary), (4) file hygiene (LF, non-empty). Exit code 0 iff every check passes."""
import glob
import json
import os
import re
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np

D = "E:/Projects/TanitAD-artifacts/dagfam"
ART = "E:/Projects/TanitAD-artifacts"
TREE = ART + "/dagt1_tree_b43c7a9a"
os.environ["TANITEVAL_STACK_OVERRIDE"] = TREE + "/stack"
for p in (TREE + "/stack", TREE + "/taniteval", TREE + "/stack/scripts"):
    sys.path.insert(0, p)
from taniteval import ci as CI  # noqa: E402
from taniteval import v8f_beta_eval as BE  # noqa: E402

md = open(D + "/FAMILIES_DAG.md", encoding="utf-8", newline="").read()
ROWS = json.load(open(D + "/DAG_FAMILIES_ROWS.json", encoding="utf-8"))
RUNS = json.load(open(ART + "/v8f-dagger-t1/runs.json", encoding="utf-8"))
n_ok = n_bad = 0


def ok(name, cond, detail=""):
    global n_ok, n_bad
    if cond:
        n_ok += 1
    else:
        n_bad += 1
        print("FAIL", name, detail)


sg = lambda x, dp=4: f"{x:+.{dp}f}"  # noqa: E731

# ---- (1) every ROW's formatted interval string appears in the document at least once (the generator is the only author of the numbers)
missing = []
for r in ROWS:
    dp = 6 if abs(r["delta"]) < 1e-3 and r["unit"] in ("1/m",) else 4
    s = f"{sg(r['delta'], dp)} [{sg(r['lo'], dp)}, {sg(r['hi'], dp)}]"
    if s not in md:
        missing.append((r["contrast"], r["metric"], r["tier"], r["seed"], s))
ok("V1 every row's interval string is in the document", not missing, missing[:3])
ok("V1b row count", len(ROWS) == 214, len(ROWS))

# ---- (3) semantic re-derivations
ok("V3a separated == interval excludes 0 (all rows)", all(bool(r["separated"]) == bool(r["lo"] > 0 or r["hi"] < 0) for r in ROWS))
ok("V3b point inside its interval or equal at the rendered precision (all rows)", all(r["lo"] - 5e-5 <= r["delta"] <= r["hi"] + 5e-5 for r in ROWS))
ok("V3c verdict strings consistent with sign and direction", all(
    (r["verdict"] == "not separated") if not r["separated"] else
    (r["verdict"] == "separated (no direction label)" if r["direction_better"] is None else
     r["verdict"] == ("separated, A better" if ((r["delta"] < 0) == (r["direction_better"] == "lower")) else "separated, A WORSE")) for r in ROWS))
for c in ("DAG-DAG0", "DAG-M2b", "DAG-ORC"):
    want = [r for r in ROWS if r["contrast"] == c and r["separated"] and r["direction_better"] is not None and not ((r["delta"] < 0) == (r["direction_better"] == "lower"))]
    m = re.search(r"\* \*\*" + re.escape(c) + r":\*\* (.*)", md)
    ok(f"V3d worse-than list for {c} has {len(want)} entries", m is not None and (m.group(1).count(" [") >= len(want)) and all(w_["metric"] in m.group(1) for w_ in want),
       m.group(1)[:200] if m else None)
# vocabulary: 'closed loop' only in negation; 'drives' never as a claim
for ln in md.splitlines():
    if re.search(r"closed[- ]loop", ln, re.I):
        ok("V3e 'closed loop' only in a negation", bool(re.search(r"never|not|no ", ln, re.I)), ln[:160])
ok("V3f no 'drives well' style claim", not re.search(r"\bdrives (well|autonomously)\b", md, re.I))
ok("V3g STRATEGIC reported (not paused)", "STRATEGIC (NOT paused)" in md and "PAUSED" not in md.replace("NOT paused", ""))

# ---- (2) direct recomputation of a sample of cells from the raw npz (independent of the driver's wrapper)
import importlib.util  # noqa: E402
from types import SimpleNamespace  # noqa: E402
PKG = TREE + "/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
_spec = importlib.util.spec_from_file_location("v8f_dk_read", PKG + "/code/v8f_dk_read.py")
DK = importlib.util.module_from_spec(_spec)
sys.modules["v8f_dk_read"] = DK
_spec.loader.exec_module(DK)
G = DK.ground_truth(SimpleNamespace(gt_npz=None, eval_grid=PKG + "/raw/eval_grid_windows.json", eval_kin=PKG + "/raw/eval141_kin.npz", eval_join=PKG + "/raw/join_eval141.npz"))
GT_GRID = np.asarray(G["gt"], np.float64)


def dump_cat(path):
    fs = sorted(glob.glob(path + "/ep*.npz"))
    parts = {k: [] for k in ("g", "cl", "eid")}
    for f in fs:
        with np.load(f) as z:
            for k in parts:
                parts[k].append(z[k])
    return {k: np.concatenate(v) for k, v in parts.items()}


def row_of(contrast, metric, tier, seed):
    rr = [r for r in ROWS if r["contrast"] == contrast and r["metric"] == metric and r["tier"] == tier and r["seed"] == seed]
    return rr[0] if rr else None


samples = []
for s in (0, 1):
    z0a = np.load(RUNS["arms"]["DAG"][str(s)]["t0"]["npz"])
    z0b = np.load(RUNS["arms"]["DAG0"][str(s)]["t0"]["npz"])
    zm = np.load(f"{D}/ref/m2b/eval_M2b_s{s}.npz")
    da = dump_cat(RUNS["arms"]["DAG"][str(s)]["t1"]["dump_dir"])
    db = dump_cat(RUNS["arms"]["DAG0"][str(s)]["t1"]["dump_dir"])
    gt = GT_GRID
    eid = z0a["eid"]
    ok(f"V2 s{s}: dump eid equals the T0 eid", np.array_equal(da["eid"], eid) and np.array_equal(db["eid"], eid))
    for (contrast, tier, pa, pb) in (("DAG-DAG0", "T0", z0a["pred"], z0b["pred"]), ("DAG-DAG0", "T1", da["cl"], db["cl"]), ("DAG-M2b", "T0", z0a["pred"], zm["pred"])):
        fa = BE.family_windows(np.asarray(pa, np.float64), gt, 0.2)
        fb = BE.family_windows(np.asarray(pb, np.float64), gt, 0.2)
        for key, metric in (("lon_speed_mae", "speed MAE m/s"), ("lon_accel_mae", "accel MAE m/s^2"), ("lat_heading_mae_deg", "heading MAE deg"), ("lat_yaw_rate_mae_degps", "yaw-rate MAE deg/s"),
                            ("tac_lat_acc", "lateral decision agreement (plan-derived)")):
            name = metric.rsplit(" ", 1)[0] if key != "tac_lat_acc" else metric
            a, b = np.asarray(fa[key], np.float64), np.asarray(fb[key], np.float64)
            ok_ = np.isfinite(a) & np.isfinite(b)
            r = CI.paired_episode_cluster_bootstrap(np.where(ok_, a, np.nan), np.where(ok_, b, np.nan), eid, n_boot=2000, seed=0)
            row = row_of(contrast, name, tier, s)
            ok(f"V2 {contrast} {tier} s{s} {name} exists", row is not None, (name,))
            if row is not None:
                samples.append(1)
                ok(f"V2 {contrast} {tier} s{s} {name} equals the direct recompute",
                   (round(r["delta"], 4), round(r["lo"], 4), round(r["hi"], 4), bool(r["separated"])) == (round(row["delta"], 4), round(row["lo"], 4), round(row["hi"], 4), row["separated"]),
                   (r["delta"], r["lo"], r["hi"], row["delta"], row["lo"], row["hi"]))
    # ADE6 levels and delta straight from the npz
    ade = lambda p: np.linalg.norm(np.asarray(p, np.float64) - gt, axis=-1).mean(1)  # noqa: E731
    row = row_of("DAG-DAG0", "ADE6, 6 s", "T0", s)
    ok(f"V2 s{s} T0 ADE6 level DAG", abs(row["mean_a"] - ade(z0a["pred"]).mean()) < 1e-9 and abs(row["mean_b"] - ade(z0b["pred"]).mean()) < 1e-9)
    row = row_of("DAG-M2b", "ADE6, 6 s", "T0", s)
    ok(f"V2 s{s} T0 ADE6 level M2b", abs(row["mean_b"] - float(zm["ade6"].mean())) < 1e-6 and round(float(zm["ade6"].mean()), 4) == {0: 2.1753, 1: 2.0546}[s])

ok("V2 sample size", len(samples) == 30, len(samples))

# ---- (4) hygiene
for p in sorted(glob.glob(D + "/*.md") + glob.glob(D + "/*.json") + glob.glob(D + "/*.py") + glob.glob(D + "/*.sh") + glob.glob(D + "/*.txt")):
    raw = open(p, "rb").read()
    ok(f"V4 non-empty {os.path.basename(p)}", len(raw) > 0)
    ok(f"V4 LF only {os.path.basename(p)}", b"\r" not in raw)
ok("V4 no stale placeholder in the document", "TODO" not in md and "XXX" not in md)
print(f"verify_families: PASSED {n_ok}  FAILED {n_bad}")
sys.exit(0 if n_bad == 0 else 1)
