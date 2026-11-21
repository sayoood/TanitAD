#!/usr/bin/env python3
"""r8nt_score.py -- score navtest seams on the Thor CPU backend for the refcv8 EMA-48,852 read (system python3; stdlib only, plus the
backend's finalize/merge which use pandas from the navsim-cpu venv).

The scorer is the Thor backend's W3 path, UNCHANGED in everything that decides a score: score_navtest8_thor.py (= score_navtest_thor.py
with roots on disk + label prefix r8nt, every diff marked # R8NT) -> W3 run_v1.cmd_score.  Sharding = the backend's own: 4 token-disjoint
shards (sorted navtest tokens, index % 4), merged by finalize_navtest_arm.py / merge_shards.py navtest (v1.1 average row recomputed in
the order of a dev-box reference CSV).

    python3 r8nt_score.py jobs     --prep <prep dir> --scores <dir> --tag <t> --arm NAME=<seam.npz>|STOP|CV [...] --out jobs.jsonl
    python3 r8nt_score.py finalize --prep <prep dir> --scores <dir> --tag <t> --arm NAME=<seam.npz>|STOP|CV [...] --order-ref <csv>

Labels: ``r8nt_<tag>_<NAME>``; shard runs in ``<scores>/_shards/<label>_k{k}of4/``; merged arm in ``<scores>/<label>/`` -- the layout
``navsim_bars8.discover_arms`` reads (``scores_navtest/<label>/<label>.counts.json|.csv|_rows.jsonl``).  ``finalize`` adds to the merged
counts the field bars8 keys on, ``"arm": "SEAM:<abs seam path>"`` (finalize_navtest_arm.py does not write it), and concatenates the 4
shards' per-token ``_rows.jsonl`` into ``<label>_rows.jsonl`` after asserting exactly 12,146 unique tokens.  Markers: QQFIN <json>.
"""
import argparse
import json
import os
import subprocess
import sys

DRIVER = "/home/nvidia/r8ns/code/navtest_read/score_navtest8_thor.py"
FINAL = "/home/nvidia/navsim-thor/code/thor/finalize_navtest_arm.py"
NV_PY = "/home/nvidia/venvs/navsim-cpu/bin/python"
K = 4


def arms_of(specs):
    out = []
    for s in specs:
        name, src = s.split("=", 1)
        out.append((name, src))
    return out


def label_of(tag, name):
    return f"r8nt_{tag}_{name}"


def cmd_jobs(a):
    n = 0
    with open(a.out, "w", encoding="utf-8") as fo:
        for name, src in arms_of(a.arm):
            lab = label_of(a.tag, name)
            for k in range(K):
                sl = f"{lab}_k{k}of{K}"
                argv = ["/usr/bin/python3", DRIVER, "--label", sl, "--out", os.path.join(a.scores, "_shards"),
                        "--tokens", os.path.join(a.prep, f"shard_k{k}of{K}.json")]
                argv += (["--official", src] if src in ("STOP", "CV") else ["--seam", os.path.abspath(src)])
                fo.write(json.dumps({"name": sl, "argv": argv, "cwd": a.scores,
                                     "log": os.path.join(a.scores, "_shards", f"{sl}.driver.txt")}) + "\n")
                n += 1
    os.makedirs(os.path.join(a.scores, "_shards"), exist_ok=True)
    print(f"QQJOBS n={n}")
    return 0


def cmd_finalize(a):
    res = {}
    for name, src in arms_of(a.arm):
        lab = label_of(a.tag, name)
        r = subprocess.run([NV_PY, FINAL, "--label", lab, "--shard-root", os.path.join(a.scores, "_shards"), "--k", str(K),
                            "--out", a.scores, "--order-ref", a.order_ref, "--expected", "12146"], capture_output=True, text=True)
        d = os.path.join(a.scores, lab)
        cj = os.path.join(d, f"{lab}.counts.json")
        rec = {"finalize_rc": r.returncode, "stdout": r.stdout.strip()[-400:], "stderr": r.stderr.strip()[-400:]}
        try:
            c = json.load(open(cj, encoding="utf-8"))
        except (OSError, ValueError) as e:
            rec["error"] = repr(e)
            res[name] = rec
            continue
        c["arm"] = (src if src in ("STOP", "CV") else "SEAM:" + os.path.abspath(src))
        rows, seen = [], set()
        for k in range(K):
            sl = f"{lab}_k{k}of{K}"
            rp = os.path.join(a.scores, "_shards", sl, f"{sl}_rows.jsonl")
            if os.path.exists(rp):
                for ln in open(rp, encoding="utf-8"):
                    if ln.strip():
                        t = json.loads(ln)["token"]
                        if t not in seen:
                            seen.add(t)
                            rows.append(ln if ln.endswith("\n") else ln + "\n")
        c["rows_jsonl_tokens"] = len(seen)
        if len(seen) == 12146:
            with open(os.path.join(d, f"{lab}_rows.jsonl"), "w", encoding="utf-8") as fo:
                fo.writelines(rows)
        else:
            c.setdefault("failures", []).append(f"rows_jsonl tokens {len(seen)} != 12146 (not written)")
        c["r8nt"] = {"how": "r8nt_score.py finalize: finalize_navtest_arm.py + 'arm' field + shard rows concatenated",
                     "order_ref": os.path.abspath(a.order_ref)}
        json.dump(c, open(cj, "w", encoding="utf-8"), indent=1)
        rec.update({"status": c.get("status"), "failures": c.get("failures"), "summary_x100_4dp": c.get("summary_x100_4dp"),
                    "csv_sha256": c.get("csv_sha256"), "rows_jsonl_tokens": len(seen)})
        res[name] = rec
    print("QQFIN " + json.dumps(res))
    return 0


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("jobs", "finalize"):
        p = sp.add_parser(nm)
        p.add_argument("--prep", required=True)
        p.add_argument("--scores", required=True)
        p.add_argument("--tag", required=True)
        p.add_argument("--arm", action="append", required=True)
        if nm == "jobs":
            p.add_argument("--out", required=True)
        else:
            p.add_argument("--order-ref", required=True)
    a = ap.parse_args()
    return cmd_jobs(a) if a.cmd == "jobs" else cmd_finalize(a)


if __name__ == "__main__":
    sys.exit(main())
