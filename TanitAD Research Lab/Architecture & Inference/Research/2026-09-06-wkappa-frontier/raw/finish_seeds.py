#!/usr/bin/env python3
"""Score the seed replicates and WRITE RESULT_SEEDS.md in one shot.

Runs, in order: the seed table (across-INFERENCE-SEED spread vs the measured
floors), the feasibility audit on the new dumps (with its human-`g` control), and
the per-episode smoke check (whose `ha0`/`g` bit-identity control proves the corpora
match). Then it composes the verdict against the PRE-REGISTERED clauses -- it does
NOT invent a bar. ASCII only.
"""
import io
import os
import subprocess
import sys

HERE = r"C:\Users\Admin\wkfront"
PY = r"C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
RAW = os.path.join(HERE, "raw")
CURV_FLOOR = 0.00200
HA0 = 0.040083


def run(args, env=None):
    e = dict(os.environ)
    e.setdefault("PYTHONIOENCODING", "utf-8")
    e.setdefault("OMP_NUM_THREADS", "2")
    e.setdefault("PYTHONPATH", r"C:\Users\Admin\tanitad-ctg\stack")
    if env:
        e.update(env)
    p = subprocess.run([PY] + args, cwd=HERE, env=e, capture_output=True, text=True)
    return (p.stdout or "") + (p.stderr or "")


def main():
    tags = sys.argv[1:] or ["wk7_s1", "wk7_s2"]
    out = {}
    out["seed_table"] = run(["seed_table.py",
                             "W_KAPPA 7|wk7," + ",".join(tags),
                             "W_KAPPA 3|wk3,wk3_s1,wk3_s2",
                             "W_KAPPA 1|wk1",
                             "W_KAPPA 15.11|wk15",
                             "w_turn 0 (gkappa)|gkappa",
                             "W_KAPPA 7 disjoint eps|wk7_dA,wk7_dB"])
    fa = []
    for t in ["wk7"] + list(tags) + ["wk7_dA", "wk7_dB"]:
        for base in (r"C:\Users\Admin\refav1_margin\p4out",
                     os.path.join(HERE, "out")):
            d = os.path.join(base, "dump_%s" % t)
            if os.path.isdir(d):
                fa += [d, t]
                break
    out["feas"] = run(["feas_audit.py"] + fa, env={"FEAS_VMIN": "2"})
    out["smoke"] = run(["partial_check.py"] + list(tags))
    for k, v in out.items():
        with io.open(os.path.join(RAW, "seeds_%s.txt" % k), "w",
                     encoding="utf-8") as f:
            f.write(v)
        print("=" * 78)
        print("### %s" % k)
        print(v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
