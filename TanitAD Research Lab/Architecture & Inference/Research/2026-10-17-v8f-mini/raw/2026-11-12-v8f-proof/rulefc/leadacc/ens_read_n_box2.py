"""ens_read_n_box2.py -- card PREREG_V8F_PROOF_ADD153 amendment 2 (stage 4): the read of ens_read_box2.py (amendment 1) with N heads and named head sets; no training.
Tier T0; never driving performance. Same split, clips, frames, labels and reads as train_acc_box2.py. Prints QUEUE-COMPLETE at the end in every case unless --no-marker.

  python ens_read_n_box2.py --out D:/tanitad/leadacc/ens4 --arm K25D --expect-ens D:/tanitad/leadacc/ens/ENS.json --record ENS4 --compare ENS2
      --sets "ENS2=0,1;ENS4=0,1,2,3;ENS5=0,1,2,3,4" --heads <path>=<sha256> ...   (heads 0 .. N-1)
"""
import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, "..", "leadspeed"), HERE):
    if os.path.isdir(_p):
        sys.path.insert(0, os.path.abspath(_p))
import ens_read_box2 as E1  # noqa: E402
import leadacc as A  # noqa: E402
import leadspeed as L  # noqa: E402
import tokstore  # noqa: E402
import train_acc_box2 as TA  # noqa: E402
import train_full_box2 as F  # noqa: E402


def parse_sets(txt):
    return {k.strip(): tuple(int(i) for i in v.split(",")) for k, v in (x.split("=") for x in txt.split(";") if x.strip())}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", default="K25D", choices=sorted(A.ARMS))
    ap.add_argument("--heads", nargs="+", required=True, help="path=sha256, heads 0 .. N-1")
    ap.add_argument("--sets", required=True, help="name=i,j;... the head sets read")
    ap.add_argument("--record", required=True, help="the named set")
    ap.add_argument("--compare", required=True, help="the set re-read against amendment 1 (control e1) and paired against the record")
    ap.add_argument("--expect-ens", required=True, help="amendment 1's ENS.json (control e1)")
    ap.add_argument("--n-ho", type=int, default=None)
    ap.add_argument("--n-eval", type=int, default=None)
    ap.add_argument("--corpus", default="D:/tanitad/cache_full/train")
    ap.add_argument("--leadf", default="D:/tanitad/data/v9/lead_train4572.npz")
    ap.add_argument("--evalc", default="D:/tanitad/cache/eval")
    ap.add_argument("--join", default="D:/tanitad/rhacc10k/pins/join_eval141.npz")
    ap.add_argument("--no-marker", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    SETS = parse_sets(a.sets)
    cfg = A.ARMS[a.arm]
    H = cfg["H"]
    res = {"card": "PREREG_V8F_PROOF_ADD153_AM2", "tier": "T0 (a head on cached tokens of logged frames); never driving performance", "arm": a.arm,
           "sets": {k: list(v) for k, v in SETS.items()}, "record": a.record, "compare": a.compare, "controls": {},
           "sha256": {"ens_read_n_box2.py": F.sha_file(os.path.abspath(__file__)), "leadacc.py": F.sha_file(os.path.join(os.path.dirname(A.__file__), "leadacc.py"))}}
    try:
        import torch
        if a.record not in SETS or a.compare not in SETS or max(max(v) for v in SETS.values()) >= len(a.heads):
            raise ValueError("sets do not fit the heads")
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        heads, e2 = [], []
        for spec in a.heads:
            p, sha = spec.rsplit("=", 1)
            got = F.sha_file(p)
            e2.append({"ckpt": p, "sha256_expected": sha, "sha256_read": got, "equal": got == sha})
            h = A.make_acc_head(H, cfg["dil"], cfg["motion"]).to(dev)
            h.load_state_dict(torch.load(p, map_location=dev))
            h.eval()
            heads.append(h)
        res["controls"]["e2_checkpoints"] = {"heads": e2, "PASS": all(x["equal"] for x in e2)}
        if not res["controls"]["e2_checkpoints"]["PASS"]:
            raise RuntimeError("checkpoint sha256 differs")
        man = F.jl_lines(os.path.join(a.corpus, "manifest.jsonl"))
        evm = F.jl_lines(os.path.join(a.evalc, "manifest.jsonl"))
        z = np.load(a.leadf, allow_pickle=True)
        sha, k, has, lead, gap = z["sha12"], z["k"], z["has"], z["lead"], z["gap"]
        order = np.argsort(sha, kind="stable")
        sha_s = sha[order]
        uniq, start = np.unique(sha_s, return_index=True)
        rows = {u: order[s:e] for u, s, e in zip(uniq, start, list(start[1:]) + [len(sha_s)])}
        cand = [r for r in man if r["sha12"] in rows]
        pick = np.random.default_rng(F.SEED).permutation(len(cand))
        n_fit_all = int(round(0.9 * len(cand)))
        ho = [cand[i] for i in pick[n_fit_all:]]
        if a.n_ho is not None:
            ho = ho[:a.n_ho]
        if a.n_eval is not None:
            evm = evm[:a.n_eval]
        res["split"] = {"held_out": len(ho), "eval": len(evm)}
        J = np.load(a.join, allow_pickle=True)
        jclip = [str(c) for c in J["clip"]]

        def lab_corpus(r, T):
            ix = rows[r["sha12"]]
            return L.token_labels(k[ix], has[ix], lead[ix], gap[ix], T)

        def lab_eval(r, T):
            ci = jclip.index(r["clip"])
            o, n = int(J["off"][ci]), int(J["n"][ci])
            lv = (np.asarray(J["lead_valid"][o:o + n]) == 1).astype(int)
            hv = np.asarray(J["has"][o:o + n]).astype(int)
            return L.token_labels(2 * np.arange(n), hv, lv, np.asarray(J["lead_gap"][o:o + n], np.float64), T)

        def stream(recs, root, labeller):
            out = {s: [] for s in SETS}
            for r in recs:
                pth = os.path.join(root, r["file"])
                d = torch.load(pth, map_location="cpu", weights_only=False, mmap=True)
                v = d["v"].numpy().astype(np.float64)
                del d
                tok = tokstore.open_tokens(pth, r.get("shapes", {}).get("pool4x10", F.SHAPE))
                T = tok.shape[0]
                pres, g, clos = labeller(r, T)
                outs = [TA.head_outputs(torch, h, tok, dev, H) for h in heads]
                base = {"v": v, "pres_lab": pres, "gap_lab": g, "clos_lab": clos, "clos_lab2": A.closing_labels(pres, g, run_max=10), "d_lab": A.d_labels(clos)}
                for s, ix in SETS.items():
                    P, G, C, D = E1.mean_outputs(outs, ix)
                    out[s].append(dict(base, pres_head=P, gap_head=G, clos_head=C, d_head=D))
            return out

        def summarise(pcs):
            cs = [A.clip_sums(A.clip_vectors(d)) for d in pcs]
            q = A.reads(cs)
            rq = F.read_quantities(pcs)
            q["S"], q["gap_mae_m"], q["presence_auc"] = rq["S"], rq["gap_mae_m"], rq["presence_auc"]
            return q, cs

        t0 = time.time()
        pho = stream(ho, a.corpus, lab_corpus)
        pev = stream(evm, a.evalc, lab_eval)
        res["read_seconds"] = round(time.time() - t0, 1)
        sums = {}
        for s in SETS:
            qh, ch = summarise(pho[s])
            qe, ce = summarise(pev[s])
            sums[s] = (ch, ce)
            res[s] = {"held_out": qh, "eval141": qe, "dA_floor_held_out": A.paired(ch, "A", ch, "A_floor", seed=151), "dA_floor_eval141": A.paired(ce, "A", ce, "A_floor", seed=151),
                      "held_out_halves": [summarise(pho[s][0::2])[0], summarise(pho[s][1::2])[0]]}
        res["record_vs_compare"] = {w: {x: A.paired(sums[a.record][i], x, sums[a.compare][i], x, seed=152) for x in ("A", "S")} for i, w in enumerate(("held_out", "eval141"))}
        ex = json.load(open(a.expect_ens, encoding="utf-8"))[a.compare]["held_out"]
        sh = res[a.compare]["held_out"]
        res["controls"]["e1_compare_reproduces"] = {"S": [sh["S"], ex["S"]], "A": [sh["A"], ex["A"]],
                                                    "PASS": bool(np.isclose(sh["S"], ex["S"], rtol=0, atol=1e-6, equal_nan=True) and np.isclose(sh["A"], ex["A"], rtol=0, atol=1e-6)
                                                                 and np.isfinite(sh["A"]))}
        planted = [dict(d, clos_head=d["clos_lab"], d_head=d["d_lab"]) for d in pho[a.record]]
        pq = A.reads([A.clip_sums(A.clip_vectors(d)) for d in planted])
        res["controls"]["e3_planted_read"] = {"S": pq["S"], "A": pq["A"], "PASS": bool(abs(pq["S"] - 1.0) < 1e-9 and abs(pq["A"] - 1.0) < 1e-9)}
        ok = all(v["PASS"] for v in res["controls"].values())
        res["controls_all_pass"] = ok
        if ok:
            for s in SETS:
                r_ = res[s]
                r_["name"] = A.name_of(r_["held_out"]["A"], r_["dA_floor_held_out"]["ci95"][0], r_["held_out"]["S"], prefix="ACCHIST10K-" + s)
                r_["eval141_name"] = A.name_of(r_["eval141"]["A"], r_["dA_floor_eval141"]["ci95"][0], r_["eval141"]["S"], prefix="ACCHIST10K-" + s)
            e = res[a.record]
            res["consequence"] = (("WITH-FORECAST: %s's acceleration is the input of card ADD144's constant-acceleration forecast (successor card)" % a.record)
                                  if e["name"].endswith("-CARRIED") and e["eval141_name"].endswith("-CARRIED") else
                                  "SHORTFALL-NAMED: eval-141 A %.4f (dA_floor lower bound %.4f); remaining levers blocked as named" % (e["eval141"]["A"], e["dA_floor_eval141"]["ci95"][0]))
    except Exception as ex_:                                                                    # noqa: BLE001
        res["error"] = "%s: %s" % (type(ex_).__name__, str(ex_)[:500])
        res["traceback"] = traceback.format_exc()[-2000:]
    json.dump(res, open(os.path.join(a.out, "ENSN.json"), "w", encoding="utf-8"), indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k != "traceback"}, indent=1, default=float)[:20000], flush=True)
    if not a.no_marker:
        print("QUEUE-COMPLETE", time.strftime("%Y-%m-%dT%H:%M:%S"), flush=True)
    return res


if __name__ == "__main__":
    main()
    sys.exit(0)
