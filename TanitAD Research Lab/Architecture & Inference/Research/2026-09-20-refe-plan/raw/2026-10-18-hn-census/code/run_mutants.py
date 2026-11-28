#!/usr/bin/env python3
"""PREREG K4: apply each single-line mutation to a COPY of the code, run the tests there, require a RED on the named test(s).
    C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -I run_mutants.py --work <scratch dir> --out ../raw/mutants.json
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
MUTANTS = [
    ("M1_strict_lt", "hn_census.py", "    prox = dist <= d\n", "    prox = dist < d\n", "test_counts_inclusive"),
    ("M2_ade_pose0_only", "hn_census.py", "    return d.mean(-1), d[..., -1]", "    return d[..., 0], d[..., -1]", "test_ade_fde"),
    ("M3_hn_dac_only", "hn_census.py", "    return pdms == 0\n", "    return dac == 0\n", "test_hard_negative_definition"),
    ("M4_rank_dropped_from_key", "hn_load.py", "int(k[3]))", "0)", "test_join_key_includes_rank"),
    ("M5_band_open_low_edge", "hn_census.py", "(dist >= lo) & (dist <= hi) if lo == 0.0", "(dist > lo) & (dist <= hi) if lo == 0.0", "test_band_mask"),
    ("M6_twin_requires_same_label", "hn_census.py", "pair_dac = near & d0[:, None] & d1[None, :]", "pair_dac = near & d0[:, None] & d0[None, :]", "test_pair_stats"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {}
    base = subprocess.run([PY, "-I", os.path.join(HERE, "test_hn_census.py")], capture_output=True, text=True, cwd=HERE)
    res["baseline"] = {"rc": base.returncode, "tail": base.stdout.strip().splitlines()[-1]}
    ok = base.returncode == 0
    for name, fn, old, new, expect in MUTANTS:
        d = os.path.join(a.work, name)
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
        for f in ("hn_census.py", "hn_load.py", "test_hn_census.py"):
            shutil.copy(os.path.join(HERE, f), d)
        s = open(os.path.join(d, fn), encoding="utf-8").read()
        assert s.count(old) == 1, (name, s.count(old))
        open(os.path.join(d, fn), "w", encoding="utf-8").write(s.replace(old, new))
        p = subprocess.run([PY, "-I", os.path.join(d, "test_hn_census.py")], capture_output=True, text=True, cwd=d)
        red = [ln.split()[1] for ln in p.stdout.splitlines() if ln.startswith("FAIL")]
        res[name] = {"rc": p.returncode, "red_tests": red, "expected_red": expect, "ok": (p.returncode != 0 and expect in red)}
        ok &= res[name]["ok"]
    res["all_ok"] = bool(ok)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
