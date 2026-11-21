"""D-YAWMASK-1 / M26 -- re-derive D-FEASDEC-T1-1 (Decisions 2026-09-05 M26) with the fixed cell.

The claim: refcv3 @40,284 `os`, base vs the feasibility projection (`proj07`), T1,
4,823 windows / 141 episodes, `paired_openloop.py`, floor `ha0`, n_boot 2000, seed 0:
"yaw-rate error 0.2176 -> 0.0427 rad/s, -80.4 % separated".

Procedure, one variable at a time, in a CLEAN tree (git archive of the tip):
  1. REPRODUCTION CONTROL: the tip's refav1_arm.py (the cell as it was) + the tip's
     paired_openloop.py on the SAME dev-box dumps the chain used
     (feasible-decode/raw/chain_rerun_all.sh). Every banked cell must reproduce
     exactly; if one does not, the re-derivation is NOT admissible and says so.
  2. FIXED: the same command with only refav1_arm.py swapped for the fixed file.
     Every NON-yaw cell must be bit-identical to step 1 (the fix moves one cell).
Arms: proj07 (the headline), proj07e (the +entry variant), projoff (the disabled
lever: a structural zero that must read exactly 0 before AND after).

usage: python rederive_m26.py --tree C:/Users/Admin/ym26/tree_m26 \
          --work-dir C:/Users/Admin/ym26/m26_out --bank-dir <pkg>/raw/m26
The raw tool outputs (which carry full clip ids) stay in --work-dir; --bank-dir receives
sha12-sanitized copies of the FIXED outputs and the summary only.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_regression_arms as R  # noqa: E402
import sanitize as S  # noqa: E402

BASE = "C:/Users/Admin/_wp56/dump/refcv3_40284_dump"
RUN = "C:/Users/Admin/feasdec/run"
BANK = ("TanitAD Research Lab/Deployment & Optimization/Research/2026-09-05-feasible-decode/"
        "raw/paired_{arm}_vs_base.json")
YAW = "LAT_yaw_rate_mae_radps"
FIELDS = ("mean", "delta", "lo", "hi", "separated", "n_windows", "n_episodes",
          "n_dropped_nonfinite")
SECTIONS = ("absolute_pooled_full_set", "margins_over_floor",
            "cross_model_difference_of_margins")


def cells(rec: dict) -> dict:
    """(section, arm, metric) -> the comparable fields."""
    out = {}
    for sec in SECTIONS:
        blk = rec.get(sec) or {}
        if sec == "cross_model_difference_of_margins":
            for mk, c in blk.items():
                if isinstance(c, dict):
                    out[(sec, "cross", mk)] = tuple(c.get(f) for f in FIELDS)
            continue
        for arm, ms in blk.items():
            for mk, c in (ms or {}).items():
                if isinstance(c, dict):
                    out[(sec, arm, mk)] = tuple(c.get(f) for f in FIELDS)
    return out


def compare(a: dict, b: dict, skip_metric: str | None = None) -> dict:
    ca, cb = cells(a), cells(b)
    keys = sorted(set(ca) | set(cb), key=str)
    diffs = [f"{k}: {ca.get(k)} -> {cb.get(k)}" for k in keys
             if (skip_metric is None or k[2] != skip_metric) and ca.get(k) != cb.get(k)]
    n = sum(1 for k in keys if skip_metric is None or k[2] != skip_metric)
    return {"n_cells_compared": n, "n_differences": len(diffs), "differences": diffs[:40]}


def yaw_rows(rec: dict) -> dict:
    out = {}
    for sec in SECTIONS:
        blk = rec.get(sec) or {}
        if sec == "cross_model_difference_of_margins":
            c = blk.get(YAW) or {}
            out["cross"] = {f: c.get(f) for f in FIELDS}
            continue
        for arm, ms in blk.items():
            c = (ms or {}).get(YAW) or {}
            out[f"{sec.split('_')[0]}:{arm}"] = {f: c.get(f) for f in FIELDS}
    return out


def run_po(tree: Path, arm: str, out_json: Path, out_md: Path) -> dict:
    cmd = [R.PY, "-u", str(tree / "taniteval/tools/paired_openloop.py"),
           "--a-dump", BASE, "--a-name", "base", "--a-arm", "os",
           "--b-dump", f"{RUN}/{arm}_dump", "--b-name", arm, "--b-arm", "os",
           "--floor", "ha0", "--n-boot", "2000", "--seed", "0",
           "--out", str(out_json), "--md", str(out_md)]
    if out_json.exists():
        out_json.unlink()
    t0 = time.time()
    env = R._env(tree)
    env["OMP_NUM_THREADS"] = "2"
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(tree),
                       encoding="utf-8", errors="replace", timeout=3600)
    log = out_json.with_suffix(".log")
    log.write_text(r.stdout + "\n--- stderr ---\n" + r.stderr, encoding="utf-8")
    if not out_json.exists():
        raise SystemExit(f"paired_openloop wrote NO JSON for {arm} (rc {r.returncode}); "
                         f"see {log}")
    return {"rc": r.returncode, "wall_s": round(time.time() - t0, 1),
            "json": out_json.name, "md": out_md.name, "log": log.name}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--bank-dir", required=True)
    ap.add_argument("--arms", default="proj07,proj07e,projoff")
    ap.add_argument("--reuse-tree", action="store_true")
    a = ap.parse_args(argv)
    tree, od, bank = Path(a.tree), Path(a.work_dir), Path(a.bank_dir)
    od.mkdir(parents=True, exist_ok=True)
    bank.mkdir(parents=True, exist_ok=True)
    rec = {"tool": "2026-09-26-yaw-rate-mask/code/rederive_m26.py",
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "claim": "D-FEASDEC-T1-1 / Decisions 2026-09-05 M26",
           "dumps": {"A_base": BASE, "B": {x: f"{RUN}/{x}_dump" for x in a.arms.split(",")}},
           "tier": "T1 (self-action OPEN loop)", "estimator":
               "paired_openloop.py: paired episode-cluster bootstrap, n_boot 2000, seed 0, "
               "floor ha0, cluster = clip"}
    if a.reuse_tree and (tree / R.REL_TOOL).exists():
        rec["tree"] = {"reused": True}
    else:
        rec["tree"] = R.build_tree(tree)
    rec["import_probe"] = R.import_probe(tree)
    tip_raw = R._git("show", f"{R.BRANCH}:{R.REL_TOOL}", binary=True)
    fixed_raw = R.FIX_TOOL.read_bytes()
    res = {}
    for arm in a.arms.split(","):
        banked = json.loads(R._git("show", f"{R.BRANCH}:{BANK.format(arm=arm)}"))
        r_arm = {"banked_artifact": BANK.format(arm=arm),
                 "banked_blob": R._git("rev-parse", f"{R.BRANCH}:{BANK.format(arm=arm)}")}
        # (1) the reproduction control on the tip's cell
        R.write_tool(tree, None, tip_raw)
        assert R._blob(tree / R.REL_TOOL) == R._git("rev-parse", f"{R.BRANCH}:{R.REL_TOOL}")
        r_arm["tip_run"] = run_po(tree, arm, od / f"paired_{arm}_vs_base.TIPCELL.json",
                                  od / f"paired_{arm}_vs_base.TIPCELL.md")
        tip_rec = json.loads((od / f"paired_{arm}_vs_base.TIPCELL.json").read_text("utf-8"))
        r_arm["reproduction_vs_banked"] = compare(banked, tip_rec)
        # (2) the fixed cell, one variable moved
        R.write_tool(tree, None, fixed_raw)
        assert R._blob(tree / R.REL_TOOL) == R._blob(R.FIX_TOOL)
        r_arm["fixed_run"] = run_po(tree, arm, od / f"paired_{arm}_vs_base.FIXED.json",
                                    od / f"paired_{arm}_vs_base.FIXED.md")
        fix_rec = json.loads((od / f"paired_{arm}_vs_base.FIXED.json").read_text("utf-8"))
        r_arm["fixed_vs_tip_non_yaw"] = compare(tip_rec, fix_rec, skip_metric=YAW)
        r_arm["yaw_banked"] = yaw_rows(banked)
        r_arm["yaw_tipcell_rerun"] = yaw_rows(tip_rec)
        r_arm["yaw_fixed"] = yaw_rows(fix_rec)
        r_arm["void"] = {"tip": tip_rec.get("void"), "fixed": fix_rec.get("void")}
        res[arm] = r_arm
        print(f"[m26] {arm}: reproduction diffs "
              f"{r_arm['reproduction_vs_banked']['n_differences']}/"
              f"{r_arm['reproduction_vs_banked']['n_cells_compared']}, non-yaw diffs after fix "
              f"{r_arm['fixed_vs_tip_non_yaw']['n_differences']}/"
              f"{r_arm['fixed_vs_tip_non_yaw']['n_cells_compared']}")
        print(f"       yaw banked {r_arm['yaw_banked']['cross']}")
        print(f"       yaw fixed  {r_arm['yaw_fixed']['cross']}")
    R.write_tool(tree, None, fixed_raw)
    rec["arms"] = res
    with open(od / "m26_rederivation.json", "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    # bank: the summary + the FIXED outputs, sha12-sanitized
    S.sanitize_json(od / "m26_rederivation.json", bank / "m26_rederivation.json")
    for arm in a.arms.split(","):
        for tag in ("FIXED", "TIPCELL"):
            S.sanitize_json(od / f"paired_{arm}_vs_base.{tag}.json",
                            bank / f"paired_{arm}_vs_base.{tag}.json")
            S.sanitize_text(od / f"paired_{arm}_vs_base.{tag}.md",
                            bank / f"paired_{arm}_vs_base.{tag}.md")
    sc = S.scan(bank)
    print(f"[m26] bank scan: {sc['n_files']} files, {sc['n_unread']} unread, "
          f"{sc['n_uuid_hits']} UUID hits")
    if sc["n_uuid_hits"] or sc["n_unread"]:
        raise SystemExit(f"bank scan FAILED: {sc}")


if __name__ == "__main__":
    main()
