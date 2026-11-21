#!/usr/bin/env python3
"""Regression control for families_run.py --per-window (fam_pertoken).

For each banked run/fam/fam_<ARM>_<C|R>.json it checks, bar <= 1e-9:
  (1) DEFAULT run of the new families_run.py == the banked JSON byte for byte (after CR removal: the Windows text-mode write adds CRLF);
  (2) the --per-window run, with {tokens, logs, per_window_schema, _per_window} removed, == the banked JSON (NaN-aware, every number);
  (3) for EVERY exported component, the aggregate recomputed from its per-token arrays (taniteval.per_window.point / the confusion counts)
      rounded to the block's own decimals equals the block's printed aggregate, |diff| <= 1e-9.

  python regress_pertoken.py --py <interp with numpy/torch> --fam-run families_run.py --rows-dir <dir with scores/> --banked-fam-dir <run/fam>
         --tanitad-repo <repo> --scratch <dir> --out regress.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys

import numpy as np

ARMS = ("R0", "RJ5", "RG8", "R5HZ")
FORMS = ("C", "R")
FRAMES = ("executed_tracked", "executed_transparent", "reference")
BAR = 1e-9


def same(x, y, path, bad):
    if isinstance(x, dict) and isinstance(y, dict):
        for k in sorted(set(x) | set(y)):
            if k not in x or k not in y:
                bad.append(f"{path}/{k}: key on one side only")
            else:
                same(x[k], y[k], f"{path}/{k}", bad)
    elif isinstance(x, list) and isinstance(y, list):
        if len(x) != len(y):
            bad.append(f"{path}: len {len(x)} != {len(y)}")
        else:
            for i, (u, v) in enumerate(zip(x, y)):
                same(u, v, f"{path}[{i}]", bad)
    elif isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y):
        pass
    elif isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
        if abs(x - y) > BAR:
            bad.append(f"{path}: {x} vs {y}")
    elif x != y:
        bad.append(f"{path}: {str(x)[:60]} vs {str(y)[:60]}")


def point(c):
    k = c["kind"]
    v = lambda n: np.asarray(c[n], float)          # noqa: E731
    if k == "mean":
        return float(np.mean(v("value")))
    if k == "rms":
        return float(math.sqrt(float(np.mean(v("value")))))
    if k == "ratio":
        d = float(np.sum(v("den")))
        return float(c["scale"] * float(np.sum(v("num"))) / d) if d > 0 else float("nan")
    return float(np.median(v("value")))


def kappa(g, p, k):
    po = float((g == p).mean())
    pe = float(sum((g == c).mean() * (p == c).mean() for c in range(k)))
    return (po - pe) / (1 - pe) if 1 - pe > 1e-9 else None


def components(blk):
    """(name, recomputed, printed, dp) for every exported component that the block prints."""
    pw = blk["_per_window"]
    rows = []
    for fam in ("longitudinal", "lateral"):
        for name, c in (pw.get(fam) or {}).items():
            node = blk[fam]
            if fam == "longitudinal" and name.startswith("target_speed_acc."):
                node, key = node["target_speed_acc"], name.split(".", 1)[1]
            elif "." in name:
                a, key = name.split(".", 1)
                node = node[a]
            else:
                key = name
            if key not in node:     # e.g. lateral.last_pose_heading_err_deg: "extra_to_the_four_family_block", the block prints no value
                rows.append((f"{fam}.{name}", None, None, c["dp"]))
                continue
            rows.append((f"{fam}.{name}", point(c), node[key], c["dp"]))
    tac = pw.get("tactical") or {}
    if tac.get("status") == "OK":
        for sub in ("lateral_decision", "longitudinal_decision", "maneuver_5way_collapsed"):
            d = tac[sub]
            g, p, k = np.asarray(d["gt"], int), np.asarray(d["pred"], int), len(d["classes"])
            pr = blk["tactical"][sub]
            rows.append((f"tactical.{sub}.accuracy", float((g == p).mean()), pr["accuracy"], 4))
            kp = kappa(g, p, k)
            rows.append((f"tactical.{sub}.kappa", kp, pr["kappa"], 4))
            for ci, cn in enumerate(d["classes"]):
                ntrue, npred, hit = int((g == ci).sum()), int((p == ci).sum()), int(((g == ci) & (p == ci)).sum())
                rows.append((f"tactical.{sub}.per_class.{cn}.recall", hit / ntrue if ntrue else None, pr["per_class"][cn]["recall"], 4))
                rows.append((f"tactical.{sub}.per_class.{cn}.precision", hit / npred if npred else None, pr["per_class"][cn]["precision"], 4))
        for k, c in (tac.get("goal_setting") or {}).items():
            rows.append((f"tactical.goal_setting.{k}", point(c), blk["tactical"]["goal_setting"].get(k), c["dp"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--py", required=True)
    ap.add_argument("--fam-run", required=True)
    ap.add_argument("--rows-dir", required=True)
    ap.add_argument("--banked-fam-dir", required=True)
    ap.add_argument("--tanitad-repo", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(a.scratch, exist_ok=True)
    report = {"bar": BAR, "files": {}, "n_components_checked": 0, "n_components_unprinted": 0, "failures": []}
    for arm in ARMS:
        for form in FORMS:
            nm = f"fam_{arm}_{form}"
            banked_fp = os.path.join(a.banked_fam_dir, nm + ".json")
            res = {}
            outs = {}
            for mode, extra in (("default", []), ("per_window", ["--per-window"])):
                out = os.path.join(a.scratch, f"{nm}_{mode}.json")
                cmd = [a.py, a.fam_run, "--rows", os.path.join(a.rows_dir, "scores", f"{arm}_s*.jsonl"), "--arm", form, "--label", f"{arm}_{form}",
                       "--out", out, "--n-boot", str(a.n_boot), "--tanitad-repo", a.tanitad_repo] + extra
                r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
                if r.returncode:
                    res[mode] = f"RC {r.returncode}: {r.stderr[-300:]}"
                    report["failures"].append(f"{nm} {mode}: rc {r.returncode}")
                    continue
                outs[mode] = out
            if "default" in outs:
                raw_new = open(outs["default"], "rb").read().replace(b"\r", b"")
                raw_old = open(banked_fp, "rb").read().replace(b"\r", b"")
                res["default_byte_identical"] = raw_new == raw_old
                if raw_new != raw_old:
                    report["failures"].append(f"{nm}: default output differs from banked")
            banked = json.load(open(banked_fp, encoding="utf-8"))
            if "per_window" in outs:
                new = json.load(open(outs["per_window"], encoding="utf-8"))
                assert new["tokens"] and len(new["tokens"]) == new["n"] == len(new["logs"])
                for k in ("tokens", "logs", "per_window_schema"):
                    new.pop(k)
                worst, n_chk, unprinted = 0.0, 0, 0
                for fr in FRAMES:
                    blk = new["frames"][fr]["block"]
                    if "_per_window" not in blk:
                        res.setdefault("no_per_window", []).append(fr)
                        continue
                    for cname, got, printed, dp in components(blk):
                        if printed is None and got is None:
                            unprinted += 1
                            continue
                        if got is None or printed is None:
                            if not (got is None and printed is None):
                                report["failures"].append(f"{nm}/{fr}/{cname}: recomputed {got} printed {printed}")
                            continue
                        n_chk += 1
                        diff = abs(round(got, dp) - printed)
                        worst = max(worst, diff)
                        if diff > BAR:
                            report["failures"].append(f"{nm}/{fr}/{cname}: round({got!r},{dp}) != printed {printed!r}")
                    blk.pop("_per_window")
                bad = []
                same(new, banked, "", bad)
                res["aggregates_equal_banked"] = not bad
                res["aggregate_diffs"] = bad[:5]
                res["components_checked"] = n_chk
                res["components_without_printed_value"] = unprinted
                res["worst_recompute_diff"] = worst
                report["n_components_checked"] += n_chk
                report["n_components_unprinted"] += unprinted
                if bad:
                    report["failures"].append(f"{nm}: --per-window aggregates differ from banked ({len(bad)} diffs)")
            report["files"][nm] = res
    report["PASS"] = not report["failures"]
    json.dump(report, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    print(json.dumps({"PASS": report["PASS"], "files": len(report["files"]), "components_checked": report["n_components_checked"],
                      "failures": report["failures"][:10]}, indent=1))
    return 0 if report["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
