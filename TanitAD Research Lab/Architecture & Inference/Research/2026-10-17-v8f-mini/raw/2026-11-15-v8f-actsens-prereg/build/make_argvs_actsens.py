"""PREREG_V8F_ACTSENS section 2: the six frozen run argvs. Each is R35's argv (ladder_R35_s{k}/argv.txt, the control) with ONLY the card's change:
IDM: `--wm-idm-gamma 1000 100 100` appended; DIV: `--wm-sampler divrs` appended; NULL: `--seed` changed only (the init checkpoint and zcache stay at the chassis' M2b seed);
and `--out` (the run directory). The pilot is DIV_s0's argv with `--steps 200` (built by actsens_check.py, not listed here).
usage: python make_argvs_actsens.py <out.txt> <out.json>"""
import json
import sys

R = "C:/Users/Admin/t2v8f_pulls/ladder_R35_s{}/argv.txt"
OUT = "D:\\tanitad\\runs\\v8f_actsens\\"
#: (tag, chassis k = M2b seed of the init checkpoint / zcache and of R35's argv, training seed, extra tokens)
ARMS = [("IDM_s0", 0, 0, ["--wm-idm-gamma", "1000", "100", "100"]),
        ("NULL_s2", 0, 2, []),
        ("DIV_s0", 0, 0, ["--wm-sampler", "divrs"]),
        ("IDM_s1", 1, 1, ["--wm-idm-gamma", "1000", "100", "100"]),
        ("DIV_s1", 1, 1, ["--wm-sampler", "divrs"]),
        ("NULL_s3", 1, 3, [])]


WHAT = {"IDM": "lever 1: world-model IDM weight 1000 / 100 / 100 (levels 1 / 2 / 3)", "DIV": "lever 2: action-diverse window sampler (DIVRS, frozen 29f388e1)", "NULL": "training-run floor: R35 with another WM seed only"}


def build():
    base = {s: open(R.format(s), encoding="utf-8").read().split() for s in (0, 1)}
    rows = []
    for tag, k, seed, extra in ARMS:
        t = list(base[k])
        i = t.index("--seed"); t[i + 1] = str(seed)
        i = t.index("--out"); t[i + 1] = OUT + tag
        rows.append((tag, t + extra))
    return rows, base


if __name__ == "__main__":
    rows, base = build()
    with open(sys.argv[1], "w", encoding="utf-8", newline="\n") as f:
        f.write("# PREREG_V8F_ACTSENS argvs: <arm>\t<argv>. Control R35 s0/s1 is already trained (ladder_R35_s{k}/argv.txt). The pilot is DIV_s0 + --steps 200.\n")
        for n, t in rows:
            f.write(f"{n}\t{' '.join(t)}\n")
    json.dump({n: {"what": WHAT[n.split("_")[0]], "argv": t} for n, t in rows}, open(sys.argv[2], "w", encoding="utf-8", newline="\n"), indent=1)
    for (tag, k, seed, extra), (n, t) in zip(ARMS, rows):
        print(n, "added/changed vs R35_s%d:" % k, [x for x in t if x not in base[k]], "removed:", [x for x in base[k] if x not in t])
