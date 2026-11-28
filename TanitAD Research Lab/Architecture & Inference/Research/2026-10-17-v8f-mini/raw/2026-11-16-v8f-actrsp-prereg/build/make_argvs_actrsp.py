"""PREREG_V8F_ACTRSP section 3: the run argvs (never launched here). Each is the chassis' ACTSENS NULL argv (= R35_s{k}'s recipe with only the seed changed; actsens_NULL_s2 for chassis 0, actsens_NULL_s3 for chassis 1) with ONLY the card's change:
the seed, the run directory (--out), and `--wm-rsp-w 1.0` appended on the RSP arms (NULL arms leave the flag absent). The pilot = RSP_s0's argv with its --out moved and `--steps 200` appended (built by actrsp_check.py, listed here as PILOT_RSP_s0 for review).
usage: python make_argvs_actrsp.py <outdir>   -> argvs_actrsp.txt, argvs_actrsp.json, and one argv_<tag>.txt per arm"""
import json
import os
import sys

SRC = {0: "C:/Users/Admin/t2v8f_pulls/actsens_NULL_s2/argv.txt", 1: "C:/Users/Admin/t2v8f_pulls/actsens_NULL_s3/argv.txt"}
OUT = "D:\\tanitad\\runs\\v8f_actrsp\\"
PILOT_OUT = "D:\\tanitad\\actrsp\\pilot_out\\RSP_s0"
#: (tag, chassis k, training seed, lever tokens)
ARMS = [("RSP_s0", 0, 0, ["--wm-rsp-w", "1.0"]), ("RSP_s2", 0, 2, ["--wm-rsp-w", "1.0"]), ("NULL_s4", 0, 4, []),
        ("RSP_s1", 1, 1, ["--wm-rsp-w", "1.0"]), ("RSP_s3", 1, 3, ["--wm-rsp-w", "1.0"]), ("NULL_s5", 1, 5, [])]
WHAT = {"RSP": "lever: level-1 action-response term, lambda 1.0 (card section 2)", "NULL": "training-run floor: the flag-off build with another seed"}


def build():
    base = {k: open(SRC[k], encoding="utf-8").read().split() for k in (0, 1)}
    rows = []
    for tag, k, seed, extra in ARMS:
        t = list(base[k])
        assert "--wm-rsp-w" not in t and "--wm-idm-gamma" not in t and "--wm-sampler" not in t
        t[t.index("--seed") + 1] = str(seed)
        t[t.index("--out") + 1] = OUT + tag
        rows.append((tag, t + extra))
    pt = list(rows[0][1])
    pt[pt.index("--out") + 1] = PILOT_OUT
    rows.insert(0, ("PILOT_RSP_s0", pt + ["--steps", "200"]))
    return rows


if __name__ == "__main__":
    d = sys.argv[1]
    rows = build()
    with open(os.path.join(d, "argvs_actrsp.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("# PREREG_V8F_ACTRSP argvs: <arm>\t<argv>. Order: pilot, RSP_s0, RSP_s2, NULL_s4, RSP_s1, RSP_s3, NULL_s5. Pairs: RSP_s0-R35_s0, RSP_s2-NULL_s2, RSP_s1-R35_s1, RSP_s3-NULL_s3. NOT LAUNCHED.\n")
        for n, t in rows:
            f.write(f"{n}\t{' '.join(t)}\n")
    json.dump({n: {"what": WHAT[n.split("_")[-2] if n.startswith("PILOT") else n.split("_")[0]], "argv": t} for n, t in rows}, open(os.path.join(d, "argvs_actrsp.json"), "w", encoding="utf-8", newline="\n"), indent=1)
    for n, t in rows:
        with open(os.path.join(d, f"argv_{n}.txt"), "w", encoding="utf-8", newline="\n") as f:
            f.write(" ".join(t) + "\n")
    for n, t in rows:
        print(n, len(t), "--wm-rsp-w" in t)
