"""n5_run.py - run A1 REPAIR-D (control), L1 and the deliberate-regression L1x on a navtrain set. CPU only; builder scorer (W-TRANSP rebuild, navtrain route).
Input: a JSONL of instances {"id","token","log","t20": [[x,y,h]*20]}  (teacher TARGET in the ego frame, as the re-roll rows carry it).
Output: one JSONL row per instance: base score, per arm {applied, status, metrics, recheck, cert-violation, n_labels, iterations}.
usage (NAVSIM venv python): python n5_run.py INST.jsonl OUT.jsonl [--arms A1,L1,L1x] [--shard i/n] [--param k=v ...]"""
import argparse
import json
import os
import sys
import time

import n5_env  # noqa: F401
import numpy as np

import n4_geom as G
import n4_repair as RP
import n4_repair_l1 as L1
import n4_nt_build as B
sys.path.insert(0, "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/teacher_fidelity/n4_dac/a1_residual")
import n4_repair_trace as TR  # noqa: E402


class Counter:
    """wraps a Ctx: counts labels/score calls (cost) and delegates the rest"""

    def __init__(self, cx):
        self.cx, self.n_labels, self.n_score = cx, 0, 0

    def __getattr__(self, k):
        return getattr(self.cx, k)

    def labels(self, t):
        self.n_labels += 1
        return self.cx.labels(t)

    def score(self, p):
        self.n_score += 1
        return self.cx.score(p)


def cert_clauses(c, b):
    return TR.failed_clauses(c, b)


def lat_flips(its):
    sg = [np.sign(e["rv"]["u_ego"][1]) for e in its if e.get("rv")]
    sg = [s for s in sg if s != 0]
    return int(sum(1 for a, b in zip(sg, sg[1:]) if a != b))


def run_arm(arm, cx0, t20, base, kw):
    cx = Counter(cx0)
    ts = time.time()
    rec = {}
    if arm == "A1":
        fn = B.make_repair("N4D")
        r, log, _ = TR.traced_call(fn, cx, t20, base)
        rec["extent_seq_m"] = [round(e["ext_max_m"], 4) for e in log]
        rec["lat_sign_flips"] = lat_flips(log)
        rec["status"] = "CERTIFIED" if r is not None else "FAILED"
    else:
        info = {}
        sign = -1.0 if arm == "L1x" else 1.0
        fn = L1.make_l1(sign=sign, **kw)
        r = fn(cx, t20, base, info=info)
        rec["status"] = info["status"]
        rec["n_flips"] = info["n_flips"]
        rec["iters"] = [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in e.items() if k != "infeasible_steps"} | {"n_infeasible_steps": len(e["infeasible_steps"])} for e in info["iters"]]
        rec["extent_seq_m"] = [e["ext_max_m"] for e in rec["iters"]]
        rec["readback"] = L1.readback(fn)
    rec["applied"] = r is not None
    if r is not None:
        rc = cx0.score([r["plan"]])[0][0]
        rec["m"] = {k: float(r["m"][k]) for k in ("dac", "nc", "ddc", "ttc", "c", "pdms")}
        rec["m_recheck"] = {k: float(rc[k]) for k in ("dac", "nc", "ddc", "ttc", "c", "pdms")}
        rec["cert_violation"] = bool(not RP.certified(rc, base))
        rec["violated_clauses"] = cert_clauses(rc, base)
        rec["plan"] = np.asarray(r["plan"]).round(6).tolist()
        rec["param"] = r["param"]
    rec["n_labels"], rec["n_score"], rec["sec"] = cx.n_labels, cx.n_score, round(time.time() - ts, 2)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inst")
    ap.add_argument("out")
    ap.add_argument("--arms", default="A1,L1,L1x")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--param", nargs="*", default=[])
    a = ap.parse_args()
    kw = {}
    for p in a.param:
        k, v = p.split("=")
        kw[k] = (v == "True") if v in ("True", "False") else float(v) if k != "iters" else int(v)
    inst = [json.loads(x) for x in open(a.inst, encoding="utf-8") if x.strip()]
    si, sn = (int(x) for x in a.shard.split("/"))
    inst = [r for i, r in enumerate(inst) if i % sn == si]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(x)["id"] for x in open(a.out, encoding="utf-8") if x.strip().endswith("}")}
    by = {}
    for r in inst:
        if r["id"] not in done:
            by.setdefault(r["log"], set()).add(r["token"])
    by = {k: sorted(v) for k, v in by.items()}
    want = {(r["token"]): [x for x in inst if x["token"] == r["token"] and x["id"] not in done] for r in inst}
    with open(a.out, "a", encoding="utf-8") as fo:
        for lg, tk, cx in G.contexts(by):
            for r in want.get(tk, []):
                t20 = np.asarray(r["t20"], float)
                rec = {"id": r["id"], "token": tk, "log": lg, "tag": r.get("tag")}
                try:
                    lab = cx.labels(t20)["R"]
                    base = cx.score([lab])[0][0]
                    rec["base"] = {k: float(base[k]) for k in ("dac", "nc", "ddc", "ttc", "c", "pdms")}
                    rec["base_dac0"] = bool(base["dac"] == 0.0)
                    rec["arms"] = {}
                    if rec["base_dac0"]:
                        for arm in a.arms.split(","):
                            rec["arms"][arm] = run_arm(arm, cx, t20, base, kw)
                except Exception as e:                                   # noqa: BLE001
                    rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
                fo.write(json.dumps(rec) + "\n")
                fo.flush()
                print(r["id"], {k: v.get("status") for k, v in rec.get("arms", {}).items()}, rec.get("error", ""), flush=True)


if __name__ == "__main__":
    main()
