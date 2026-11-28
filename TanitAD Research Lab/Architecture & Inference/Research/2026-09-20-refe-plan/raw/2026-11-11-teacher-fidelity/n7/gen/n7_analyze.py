"""n7_analyze.py - per-token / per-lever tables from n7_run.py output (plain python, no navsim imports).
usage: n7_analyze.py file1.jsonl [file2.jsonl ...] [--levers a,b]  -> prints; --json out writes the compact per-token table (no plans)"""
import argparse
import json
import sys

sys.dont_write_bytecode = True
EPS = 1e-12


def tol(m, ref, full=False):
    ok = m["dac"] == 1.0 and m["nc"] >= ref["nc"] and m["ddc"] >= ref["ddc"] - EPS and m["pdms"] > ref["pdms"]
    if full:
        ok = ok and m["ttc"] >= ref["ttc"] - EPS and m["c"] >= ref["c"] - EPS
    return ok


def load(files):
    rows = {}
    for f in files:
        for ln in open(f, encoding="utf-8"):
            if ln.strip():
                r = json.loads(ln)
                if r["token"] in rows:
                    rows[r["token"]]["levers"].update(r["levers"])
                else:
                    rows[r["token"]] = r
    return rows


def derive_caps(rows):
    """cap 5 / 10 candidates are a PREFIX (k <= cap) of the cap 20 run (pinned by test_real_cap_prefix_property): synthesise those lever entries"""
    for r in rows.values():
        for n in list(r["levers"]):
            if "_c20_" in n:
                for c in (5, 10):
                    r["levers"][n.replace("_c20_", f"_c{c}_")] = [x for x in r["levers"][n] if x["k"] <= c]
    return rows


def lever_table(rows, names=None):
    names = names or sorted({n for r in rows.values() for n in r["levers"]})
    out = {}
    for n in names:
        dac1, tolc, fullc = [], [], []
        for t, r in sorted(rows.items()):
            c = r["levers"].get(n)
            if c is None:
                continue
            if any(x["m"]["dac"] == 1.0 for x in c):
                dac1.append(t)
            if any(tol(x["m"], r["ref"]) for x in c):
                tolc.append(t)
            if any(tol(x["m"], r["ref"], True) for x in c):
                fullc.append(t)
        out[n] = {"n": sum(1 for r in rows.values() if n in r["levers"]), "dac1": dac1, "tol": tolc, "full": fullc}
    return out


def union(rows, names, key="tol"):
    s = set()
    for n in names:
        s |= set(lever_table(rows, [n])[n][key])
    return s


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--levers")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    rows = derive_caps(load(a.files))
    tb = lever_table(rows, a.levers.split(",") if a.levers else None)
    print(f"tokens {len(rows)}")
    print(f"{'lever':14s} {'n':>3s} {'dac1':>5s} {'tol':>4s} {'full':>5s}")
    for n, v in tb.items():
        print(f"{n:14s} {v['n']:3d} {len(v['dac1']):5d} {len(v['tol']):4d} {len(v['full']):5d}")
    if a.json:
        json.dump(tb, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
