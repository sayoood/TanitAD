"""PREREG_V8F_WMPOOL: the three argvs. Each is R35H's argv (ladder_R35H_s{k}/argv.txt, the control) with ONLY the card's levers moved:
--wm-pool att25 added, --wm-zcache pointed at the CELL cache of the same adapter (zcells.py output, M2b_s{k}/train), --seed for the replicate, and --out.
The replicate (seed 2) keeps the init checkpoint and the cell cache of M2b_s0."""
import sys
BS = chr(92)
R = "C:/Users/Admin/t2v8f_pulls/ladder_R35H_s{}/argv.txt"
OUT = "D:" + BS + "tanitad" + BS + "runs" + BS + "v8f_wmpool"
CELLS = "D:" + BS + "tanitad" + BS + "zcells" + BS + "M2b_s{}" + BS + "train"
ARMS = [("P-ATT", 0, 0), ("P-ATT", 1, 1), ("P-ATT_rep", 2, 0)]          # (name, seed, M2b seed of the readers / adapter / cell cache)


def build():
    rows, base = [], {s: open(R.format(s)).read().split() for s in (0, 1)}
    for name, seed, src in ARMS:
        t = list(base[src])
        i = t.index("--seed"); t[i + 1] = str(seed)
        i = t.index("--out"); t[i + 1] = OUT + BS + f"{name}_s{seed}"
        i = t.index("--wm-zcache"); assert t[i + 1].endswith(f"M2b_s{src}" + BS + "train"), t[i + 1]; t[i + 1] = CELLS.format(src)
        t += ["--wm-pool", "att25"]
        rows.append((f"{name}_s{seed}", " ".join(t)))
    return rows, base


if __name__ == "__main__":
    rows, base = build()
    with open(sys.argv[1], "w", encoding="utf-8", newline="\n") as f:
        f.write("# PREREG_V8F_WMPOOL argvs: <arm>\t<argv>. Control R35H s0/s1 is already trained (ladder_R35H_s{k}/argv.txt).\n")
        for n, a in rows:
            f.write(f"{n}\t{a}\n")
    for (name, seed, src), (n, a) in zip(ARMS, rows):
        t, b = a.split(), base[src]
        print(n, "added/changed:", [x for x in t if x not in b], "removed/changed:", [x for x in b if x not in t])
