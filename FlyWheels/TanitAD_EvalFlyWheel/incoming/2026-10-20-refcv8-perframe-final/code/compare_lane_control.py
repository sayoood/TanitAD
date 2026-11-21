"""compare_lane_control.py -- the known-value control's comparator (EvalFlyWheel 2026-10-20).

pf_lane.py only RE-POINTS the battery package's instruments.  Run on a banked per-clip tree with the clip-fed table installed, it must reproduce the banked
per-clip artifacts EXACTLY: every numeric / boolean leaf equal (no tolerance), every array of the DEPLOY8 position npz bit-equal.  String leaves (paths, timestamps,
file hashes of the re-pointed files) are listed but do not decide; a missing numeric leaf on either side is a FAIL.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

IGNORE_KEYS = {"generated", "wall_s", "elapsed_s", "script_sha256", "t_wall_s"}


def walk(a, b, path, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k in IGNORE_KEYS:
                continue
            if k not in a or k not in b:
                out["missing"].append(f"{path}/{k} ({'banked' if k not in a else 'lane'} lacks it)")
                continue
            walk(a[k], b[k], f"{path}/{k}", out)
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out["structural"].append(f"{path} len {len(a)} != {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", out)
        return
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        out["n_leaf"] += 1
        if a != b:
            out["numeric"].append(f"{path}: {a!r} != {b!r}")
        return
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        out["n_leaf"] += 1
        same = (a == b) or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b))
        if not same:
            out["numeric"].append(f"{path}: {a!r} != {b!r}")
        return
    if isinstance(a, str) and isinstance(b, str):
        out["n_str"] += 1
        if a != b:
            out["strings"].append(f"{path}: {a[:90]!r} != {b[:90]!r}")
        return
    out["structural"].append(f"{path}: type {type(a).__name__} != {type(b).__name__}")


def cmp_json(banked: Path, lane: Path, sub=None) -> dict:
    if not banked.exists() or not lane.exists():
        return {"status": "MISSING", "banked_exists": banked.exists(), "lane_exists": lane.exists()}
    a, b = json.load(open(banked, encoding="utf-8")), json.load(open(lane, encoding="utf-8"))
    if sub:
        a, b = sub(a), sub(b)
    return cmp_json_obj(a, b)


def cmp_json_obj(a, b) -> dict:
    out = {"n_leaf": 0, "n_str": 0, "numeric": [], "missing": [], "structural": [], "strings": []}
    walk(a, b, "", out)
    bad = len(out["numeric"]) + len(out["missing"]) + len(out["structural"])
    return {"status": "PASS" if (bad == 0 and out["n_leaf"] > 0) else "FAIL", "n_numeric_bool_leaves": out["n_leaf"], "n_string_leaves": out["n_str"],
            "n_numeric_diffs": len(out["numeric"]), "n_missing": len(out["missing"]), "n_structural": len(out["structural"]),
            "first_numeric_diffs": out["numeric"][:12], "first_missing": out["missing"][:12], "first_structural": out["structural"][:12],
            "n_string_diffs": len(out["strings"]), "first_string_diffs": out["strings"][:12]}


def cmp_npz(banked: Path, lane: Path) -> dict:
    if not banked.exists() or not lane.exists():
        return {"status": "MISSING", "banked_exists": banked.exists(), "lane_exists": lane.exists()}
    za, zb = np.load(banked, allow_pickle=False), np.load(lane, allow_pickle=False)
    ka, kb = set(za.files), set(zb.files)
    diffs, meta = [], {}
    for k in sorted(ka & kb):
        a, b = za[k], zb[k]
        if a.dtype.kind in "US" and b.dtype.kind in "US" and a.ndim == 0 and k.endswith("json"):
            # a JSON metadata string (paths, the table file's own hash): the SAME rule as the json files -- numeric / bool leaves decide, strings are listed
            m = cmp_json_obj(json.loads(str(a)), json.loads(str(b)))
            meta[k] = m
            if m["status"] != "PASS":
                diffs.append(k)
            continue
        if not (a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b, equal_nan=a.dtype.kind == "f")):
            diffs.append(k)
    return {"status": "PASS" if (ka == kb and not diffs and ka) else "FAIL", "n_arrays": len(ka & kb), "only_banked": sorted(ka - kb), "only_lane": sorted(kb - ka),
            "arrays_differing": diffs[:20], "json_string_arrays": meta}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--banked", required=True, help="the battery package's raw/ dir")
    ap.add_argument("--tree", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    T, B, L = a.tree, Path(a.banked), Path(a.pkg) / "raw"
    rows = {
        "clamp_tree": cmp_json(B / "clamp_posthoc/trees" / f"{T}.json", L / "clamp_posthoc/trees" / f"{T}.json"),
        "compliant_tree": cmp_json(B / "compliant_reread/trees" / f"{T}.json", L / "compliant_reread/trees" / f"{T}.json"),
        "FORWARD_TRAINFIT": cmp_json(B / "retime_trainfit" / f"FORWARD_TRAINFIT_{T}.json", L / "retime_trainfit" / f"FORWARD_TRAINFIT_{T}.json"),
        "FORWARD_DEPLOY8": cmp_json(B / "retime_deploy" / f"FORWARD_DEPLOY8_{T}.json", L / "retime_deploy" / f"FORWARD_DEPLOY8_{T}.json"),
        "FORWARD_DEPLOY": cmp_json(B / "retime_deploy" / f"FORWARD_DEPLOY_{T}.json", L / "retime_deploy" / f"FORWARD_DEPLOY_{T}.json"),
        "TAC": cmp_json(B / "retime_deploy" / f"TAC_{T}.json", L / "retime_deploy" / f"TAC_{T}.json"),
        "FROZEN_GAINS_RETIME_deploy": cmp_json(B / "retime_deploy" / "FROZEN_GAINS_RETIME_deploy.json", L / "retime_deploy" / "FROZEN_GAINS_RETIME_deploy.json"),
    }
    for s in (0, 1):
        rows[f"DEPLOY8_POS_s{s}"] = cmp_npz(B / "retime_deploy" / f"DEPLOY8_POS_{T}_s{s}.npz", L / "retime_deploy" / f"DEPLOY8_POS_{T}_s{s}.npz")
    pp_b, pp_l = B / "retime_deploy" / "PRIOR_PARITY.json", L / "retime_deploy" / "PRIOR_PARITY.json"
    rows["PRIOR_PARITY_beside"] = {"note": "BESIDE: the banked PRIOR_PARITY.json may cover other trees; lane file exists = " + str(pp_l.exists()),
                                   "lane_status": (json.load(open(pp_l, encoding="utf-8")).get("status") if pp_l.exists() else None),
                                   "banked_status": (json.load(open(pp_b, encoding="utf-8")).get("status") if pp_b.exists() else None)}
    deciding = [k for k in rows if k != "PRIOR_PARITY_beside"]
    status = "PASS" if all(rows[k]["status"] == "PASS" for k in deciding) else "FAIL"
    rec = {"tool": "compare_lane_control.py", "generated": time.strftime("%FT%T%z"), "tree": T, "lane_pkg": str(Path(a.pkg)), "banked_raw": str(B),
           "status": status, "rule": "every numeric/bool leaf and every npz array bit-equal (no tolerance); strings listed, not deciding",
           "evidence_class": "MEASURED (ours): pf_lane.py on the banked per-clip tree vs the banked per-clip artifacts", "rows": rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: rows[k].get("status") for k in rows}), f"ZZLANECONTROL_{status}ZZ", flush=True)
    return 0 if status == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
