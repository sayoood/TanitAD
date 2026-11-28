"""n7_run.py - run named levers on a navtrain token set; one JSON line per token with every candidate (plan + builder metrics).
usage (navsim-crun venv python): n7_run.py --set r17|rep73 --levers A,B --out file.jsonl [--shard i/n] [--sign -1]"""
import argparse
import json
import os
import sys
import time

import n7_env  # noqa: F401
import numpy as np
import n4_geom as G
import n7_lib as L

TF = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/teacher_fidelity/rj5_navtrain/repair_nt"
PL = "F:/tanitad_scratch/rj5nt/repair_nt/nt_plans.npz"
RL = "F:/tanitad_scratch/rj5nt/repair_nt/nt_rolls.jsonl"
ROWS = os.environ.get("N7_ROWS", "C:/Users/Admin/l2dev/x/derive_s{}of2.jsonl")   # derive_rows.tar.xz extracted there (REPAIR_NT.md); override with N7_ROWS


def token_set(which):
    if which == "r17":
        return json.load(open(TF + "/navtrain_residual17.json"))["tokens"]
    rows = [json.loads(ln) for s in (0, 1) for ln in open(ROWS.format(s))]
    return sorted(r["token"] for r in rows if r.get("dac0") and r["arms"]["N4D"]["applied"])


def load_rolls(tokens):
    want, out = set(tokens), {}
    for ln in open(RL, encoding="utf-8"):
        if ln.strip():
            r = json.loads(ln)
            if r.get("token") in want and "traj20" in r and "error" not in r:
                out[r["token"]] = np.asarray(r["traj20"], float)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--levers", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--sign", type=float, default=1.0)
    a = ap.parse_args(argv)
    toks = token_set(a.set)
    si, sn = (int(x) for x in a.shard.split("/"))
    toks = [t for i, t in enumerate(toks) if i % sn == si]
    z = np.load(PL, allow_pickle=False)
    ix = {str(t): i for i, t in enumerate(z["tokens"])}
    rolls = load_rolls(toks)
    by = {}
    for t in toks:
        by.setdefault(str(z["logs"][ix[t]]), []).append(t)
    names = a.levers.split(",")
    try:
        done = {json.loads(ln)["token"] for ln in open(a.out) if ln.strip()}
    except FileNotFoundError:
        done = set()
    with open(a.out, "a", encoding="utf-8") as fo:
        for lg, tok, cx in G.contexts(by):
            if tok in done:
                continue
            t0 = time.time()
            label = np.asarray(z["plans20"][ix[tok]], float)
            ref = cx.score([label])[0][0]
            rec = {"log": lg, "token": tok, "ref": ref, "levers": {}}
            for n in names:
                c = L.run_lever(n, L.LEVERS[n], cx, rolls[tok], label, ref, sign=a.sign)
                rec["levers"][n] = [{"k": x["k"], "plan": x["plan"].tolist(), "m": x["m"]} for x in c]
            rec["t_s"] = round(time.time() - t0, 2)
            fo.write(json.dumps(rec) + "\n")
            fo.flush()
            print(tok, rec["t_s"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
