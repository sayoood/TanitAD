"""Builds frozen/argvs_slowwm.json from R35H's banked argv (dev box). Each SLOWWM argv is R35H's token for token except:
  --seed, --wm-zcache (the basis cache of the kind), --out (D:\\tanitad\\runs\\v8f_slowwm\\<TAG>), and ONE appended pair --wm-basis <slow|pca>.
    python -I make_argvs_slowwm.py --out-json frozen/argvs_slowwm.json [--r35h-dir C:/Users/Admin/t2v8f_pulls]"""
import argparse
import json
import os
import sys

sys.dont_write_bytecode = True
RUNS = r"D:\tanitad\runs\v8f_slowwm"
ZB = r"D:\tanitad\zbasis"
#: tag -> (R35H source index = the M2b start state, training seed, basis kind)
ARMS = {"SLOW_s0": (0, 0, "slow"), "PCA-B_s0": (0, 0, "pca"), "SLOW_s1": (1, 1, "slow"), "PCA-B_s1": (1, 1, "pca"), "SLOW_rep": (0, 2, "slow")}


def derive(base, tag):
    src, seed, kind = ARMS[tag]
    t = list(base)
    t[t.index("--seed") + 1] = str(seed)
    t[t.index("--wm-zcache") + 1] = ZB + "\\" + kind + "\\train"
    t[t.index("--out") + 1] = RUNS + "\\" + tag
    return t + ["--wm-basis", kind]


def changed_tokens_only(base, derived, tag):
    """[] iff derived differs from base ONLY at --seed / --wm-zcache / --out values and the appended --wm-basis pair"""
    p = []
    if derived[:len(base)] == base:
        p.append("nothing changed")
    if derived[-2:] != ["--wm-basis", ARMS[tag][2]] or len(derived) != len(base) + 2:
        return ["the argv must end with exactly one appended --wm-basis pair"]
    allowed = {base.index(f) + 1 for f in ("--seed", "--wm-zcache", "--out")}
    return p + [f"token {i}: {b!r} -> {d!r}" for i, (b, d) in enumerate(zip(base, derived)) if b != d and i not in allowed]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--r35h-dir", default="C:/Users/Admin/t2v8f_pulls")
    a = ap.parse_args(argv)
    bases = {i: open(os.path.join(a.r35h_dir, f"ladder_R35H_s{i}", "argv.txt"), encoding="utf-8").read().strip().split(" ") for i in (0, 1)}
    out = {}
    for tag, (src, seed, kind) in ARMS.items():
        d = derive(bases[src], tag)
        bad = changed_tokens_only(bases[src], d, tag)
        if bad:
            raise SystemExit(f"REFUSED {tag}: {bad}")
        out[tag] = {"what": f"{kind} basis, seed {seed} on M2b_s{src} (R35H_s{src}'s argv)", "argv": d}
    with open(a.out_json, "w", encoding="utf-8", newline="\n") as f:
        f.write("{\n" + ",\n".join(f" {json.dumps(k)}: {json.dumps(v)}" for k, v in out.items()) + "\n}\n")      # one line per tag
    print("wrote", a.out_json, list(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
