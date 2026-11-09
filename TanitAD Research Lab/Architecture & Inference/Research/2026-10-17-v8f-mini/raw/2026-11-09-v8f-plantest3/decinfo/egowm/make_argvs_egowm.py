"""PREREG_V8F_EGOWM: the five argvs. Each is R35H's argv (ladder_R35H_s{k}/argv.txt, the control) with ONLY the card's levers moved:
--wm-speed {kin,now} added, --seed for the replicate, and --out (the run directory). The init checkpoint and the zcache of the seed-2 replicate stay at M2b_s0 (no M2b_s2 exists; the replicate then differs from E-KIN s0 in the WM seed alone)."""
import sys
R = "C:/Users/Admin/t2v8f_pulls/ladder_R35H_s{}/argv.txt"
OUT = r"D:\tanitad\runs\v8f_egowm"
ARMS = [("E-KIN", "kin", 0, 0), ("E-KIN", "kin", 1, 1), ("E-NOW", "now", 0, 0), ("E-NOW", "now", 1, 1), ("E-KIN_rep", "kin", 2, 0)]   # (name, mode, seed, readers' M2b seed)


def build():
    rows, base = [], {s: open(R.format(s)).read().split() for s in (0, 1)}
    for name, mode, seed, src in ARMS:
        t = list(base[src])
        i = t.index("--seed"); t[i + 1] = str(seed)
        i = t.index("--out"); t[i + 1] = OUT + "\\" + f"{name}_s{seed}"
        t += ["--wm-speed", mode]
        rows.append((f"{name}_s{seed}", " ".join(t)))
    return rows, base


def diff_vs(base_toks, toks):
    a, b = base_toks, toks
    return [t for t in b if t not in a] , [t for t in a if t not in b]


if __name__ == "__main__":
    rows, base = build()
    outp = sys.argv[1]
    with open(outp, "w", encoding="utf-8", newline="\n") as f:
        f.write("# PREREG_V8F_EGOWM argvs: <arm>\t<argv>. Control R35H s0/s1 is already trained (ladder_R35H_s{k}/argv.txt).\n")
        for n, a in rows:
            f.write(f"{n}\t{a}\n")
    for (name, mode, seed, src), (n, a) in zip(ARMS, rows):
        add, rem = diff_vs(base[src], a.split())
        print(n, "added/changed:", add, "removed/changed:", rem)
