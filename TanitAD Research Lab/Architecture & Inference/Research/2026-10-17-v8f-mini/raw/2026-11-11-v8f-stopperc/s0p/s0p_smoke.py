"""s0p_smoke.py -- END-TO-END CODE CHECK of s0p_fit.py on the EVAL arrays only (clip-grouped half / half).  NOT A RESULT: the 'train' half is eval data, the sample is not the card's, the numbers mean nothing.
Usage: python s0p_smoke.py --eval F:/.../s0p_eval.npz --work F:/.../smoke_work [--stage prep|fit|read|all]
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import s0p_lib as L  # noqa: E402
import s0p_fit as F  # noqa: E402


def make_halves(eval_npz, work):
    d = L.load_npz(eval_npz)
    clips = np.array(sorted(set(d["clip"].tolist())))
    train_clips = set(clips[::2].tolist())
    mtr = np.array([c in train_clips for c in d["clip"]])
    trd = os.path.join(work, "train_half")
    os.makedirs(trd, exist_ok=True)
    for k in L.FIELDS:
        np.save(os.path.join(trd, k + ".npy"), d[k][mtr])
    json.dump({"smoke": True, "NOT_A_RESULT": True, "c2_second_route_max_abs": None}, open(os.path.join(trd, "meta.json"), "w"))
    ev = {k: d[k][~mtr] for k in L.FIELDS}
    ev["ci"] = d["ci"][~mtr]
    meta = dict(d["meta"], smoke_half=True)
    ep = os.path.join(work, "eval_half.npz")
    L.save_npz(ep, ev["E"], ev["v40"], ev["v160"], ev["mean"], ev["y"], ev["clip"], ev["ws"], ev["eid"], meta, ci=ev["ci"])
    return trd, ep, int(mtr.sum()), int((~mtr).sum()), int(d["y"][mtr].sum()), int(d["y"][~mtr].sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--stage", default="all")
    ap.add_argument("--only", default=None, help="comma list of arm|model to fit (timing runs)")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    trd, ep, ntr, nev, ptr, pev = make_halves(a.eval, a.work) if not os.path.exists(os.path.join(a.work, "eval_half.npz")) else (os.path.join(a.work, "train_half"), os.path.join(a.work, "eval_half.npz"), 0, 0, 0, 0)
    print(f"SMOKE (NOT A RESULT): train half {ntr} windows / {ptr} pos, eval half {nev} / {pev} pos", flush=True)
    work = os.path.join(a.work, "w")
    args = ["--train", trd, "--eval", ep, "--work", work]
    if a.stage == "all" and not a.only:
        F.main(["all"] + args + ["--out", os.path.join(a.work, "RESULT_S0P_SMOKE_NOT_A_RESULT.json")])
    elif a.stage in ("prep", "read"):
        F.main([a.stage] + args + ["--out", os.path.join(a.work, "RESULT_S0P_SMOKE_NOT_A_RESULT.json")])
    else:
        F.main(["prep"] + args)
        for ao in a.only.split(","):
            arm, model = ao.split("|")
            F.main(["fit"] + args + ["--arm", arm, "--model", model])


if __name__ == "__main__":
    main()
