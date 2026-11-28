#!/usr/bin/env python3
"""n4_repair_trace.py - instrumented trace of the TF-N4 A1 REPAIR-D loop (RJ5 teacher + P5-R + REPAIR-D). CPU only. Descriptive instrument; decides nothing.

It IMPORTS the pinned run tree (F:/tanitad_scratch/n4nt/run_a1/n4, never a copy) and calls the REAL `n4_repair.repair_target` through the builder's own
`n4_nt_build.make_repair` partial. Nothing of the loop is re-implemented: a thin proxy around the token context records every `cx.labels` / `cx.score`
call and a patch of `n4_repair.return_vector` records every worst-outside-corner reading. Hops relied on (file:line in the pinned tree, A1 run):
  n4_nt_build.py:78    make_repair  -> functools.partial(RP.repair_target, margin, iters, sign, cert)   (card CARD near :55)
  n4_nt_build.py:214   builder context: Ctx.from_parts(TS.load_cache_parts(root, log, tok), label_ego_fn=db_ego_fn)
  n4_nt_build.py:259   base = cx.score([base_plan])[0][0] ; :268 r = fns[arm](cx, t20, base) ; :270 None -> {"applied": False}
  n4_repair.py:113     repair_target (loop: labels(tgt)["R"] -> score -> cert? -> return_vector -> shift (ext+margin)*u ramped by t_v -> tgt shifted)
  n4_repair.py:47      return_vector (worst corner over 41 steps -> ext, t_v, unit return vector in the ego frame) ; :25 certified ; :29 certified_dac
  n4_geom.py:84        Ctx.labels ; :92 Ctx.score ; :101 Ctx.ext_dist
Usage (navsim venv python):  python n4_repair_trace.py primary|whatif|control --out <jsonl> [--tokens-file f] [--shard i/N] ..."""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time

RUN = os.environ.get("N4_RUN", "F:/tanitad_scratch/n4nt/run_a1")
DEVKIT = "F:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896"
NAVSIM_ROOT = "F:/Archive/devbox-C/navsim"
CACHE_ROOT = "F:/Archive/devbox-C/navsim/exp/w3_navtest_v1/metric_cache_navtest"
MAPS = "E:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0"
OUT_BUILD = RUN + "/n4run/build/out"
sys.dont_write_bytecode = True                       # never write a __pycache__ into the pinned run tree
os.environ.update({"CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "REFE_SIM_HZ": "10",
                   "NUPLAN_MAPS_ROOT": MAPS, "PYTHONIOENCODING": "utf-8", "TANITAD_NAVSIM_ROOT": NAVSIM_ROOT, "PYTHONDONTWRITEBYTECODE": "1",
                   "N4_W": RUN + "/w/pk", "N4_P5R": RUN + "/w/p5r"})
for _p in (RUN + "/w/pk/refe", DEVKIT):          # same order as the chain's PYTHONPATH (DEVKIT;refe), ahead of any site-packages navsim
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
sys.path.insert(0, RUN + "/n4")                       # the pinned chain's own n4_geom / n4_repair / n4_nt_build win

import numpy as np  # noqa: E402

import n4_nt_build as B  # noqa: E402
import n4_repair as RP  # noqa: E402

CLAUSES = ("dac", "nc", "ddc", "ttc", "c")


def failed_clauses(c: dict, b: dict) -> list:
    """which clause(s) of the FULL certificate (n4_repair.certified, :30-31) a candidate fails against the base label's own tracked scores"""
    f = []
    if c["dac"] != 1.0:
        f.append("dac")
    if not c["nc"] >= b["nc"]:
        f.append("nc")
    if not c["ddc"] >= b["ddc"] - 1e-12:
        f.append("ddc")
    if not c["ttc"] >= b["ttc"] - 1e-12:
        f.append("ttc")
    if not c["c"] >= b["c"] - 1e-12:
        f.append("c")
    return f


class Proxy:
    """records every labels / score call of one repair run; delegates everything else to the real Ctx (ext_dist, drivable, vp, ego used by return_vector)"""

    def __init__(self, cx, base):
        self.cx, self.base, self.log, self.rv_log, self.n_labels = cx, base, [], [], 0

    def __getattr__(self, k):
        return getattr(self.cx, k)

    def labels(self, t):
        self.n_labels += 1
        return self.cx.labels(t)

    def score(self, plans):
        res, sim = self.cx.score(plans)
        for r, s in zip(res, sim):
            d = self.cx.ext_dist(s)
            self.log.append({"metrics": {k: float(r[k]) for k in ("dac", "nc", "ddc", "ttc", "c", "ep", "pdms") if k in r},
                             "failed": failed_clauses(r, self.base), "ext_max_m": float(d.max()), "n_steps_outside": int((d > 0).sum()),
                             "t_worst_s": 0.1 * int(np.argmax(d)) if d.max() > 0 else None, "rv": None})
        return res, sim


def traced_call(fn, cx, t20, base):
    """run `fn` (a builder repair partial) against the proxy; patch return_vector to record. -> (result, per-iteration records, proxy)"""
    px = Proxy(cx, base)
    orig = RP.return_vector

    def rv_rec(c, state41):
        out = orig(c, state41)
        px.log[-1]["rv"] = None if out is None else {"ext_m": out[0], "t_v_s": out[1], "u_ego": [float(out[2][0]), float(out[2][1])]}
        return out
    RP.return_vector = rv_rec
    try:
        r = fn(px, t20, base)
    finally:
        RP.return_vector = orig
    return r, px.log, px


def summarise(fn_spec: dict, r, log: list, margin: float) -> dict:
    """per-iteration records + cumulative shift (same arithmetic as repair_target: total += (ext + margin) * u)"""
    tot = np.zeros(2)
    its = []
    for i, e in enumerate(log):
        rec = dict(e, it=i, cum_shift_ego_m=[round(float(tot[0]), 5), round(float(tot[1]), 5)])
        its.append(rec)
        if e["rv"] is not None:
            tot = tot + (e["rv"]["ext_m"] + margin) * np.asarray(e["rv"]["u_ego"])
    return {"spec": fn_spec, "applied": r is not None, "applied_param": None if r is None else r["param"], "iterations": its}


def contexts(todo: dict):
    """{token: log} -> yields (token, log, Ctx) exactly as the builder's cache route does"""
    import n4_geom as G
    import transp_score as TS
    dbs = G.NS.index_dbs(db_root=G.DB_ROOT)
    for tok, lg in sorted(todo.items()):
        yield tok, lg, G.Ctx.from_parts(TS.load_cache_parts(CACHE_ROOT, lg, tok), label_ego_fn=B.db_ego_fn(dbs, lg, tok))


def load_inputs():
    z = np.load(RUN + "/inputs/n2r/n2_plans_N2R.npz", allow_pickle=False)
    toks, logs, plans = [str(t) for t in z["tokens"]], [str(x) for x in z["logs"]], np.asarray(z["plans20"], float)
    ix = {t: i for i, t in enumerate(toks)}
    rolls = B.load_rolls(RUN + "/inputs/n2r/rolls_RJ5/rows_s*of2.jsonl")
    return toks, logs, plans, ix, rolls


def run_tokens(todo: dict, mode: str, out: str, mutation: str = "none", margins=(0.05, 0.10, 0.20), iters_wi=20):
    toks, logs, plans, ix, rolls = load_inputs()
    done = B.done_tokens(out)
    todo = {t: l for t, l in todo.items() if t not in done}
    with open(out, "a", encoding="utf-8") as fo:
        for tok, lg, cx in contexts(todo):
            t0 = time.time()
            rec = {"token": tok, "log": lg}
            try:
                base_plan, t20 = plans[ix[tok]], rolls[tok]
                base = cx.score([base_plan])[0][0]
                rec["base"] = {k: float(base[k]) for k in ("dac", "nc", "ddc", "ttc", "c", "ep", "pdms")}
                runs = {}
                if mode in ("primary", "control"):
                    for arm in (("N4D", "N4DM") if mode == "primary" else ("N4D",)):
                        fn = B.make_repair(arm, mutation)
                        r, log, px = traced_call(fn, cx, t20, base)
                        runs[arm] = summarise(B.readback(fn), r, log, B.CARD[arm]["margin"] if mutation != "margin_neg" else -0.05)
                        runs[arm]["n_label_builds"] = px.n_labels
                        if r is not None:
                            runs[arm]["plan"] = np.asarray(r["plan"]).tolist()
                else:                                                    # whatif: one iters=20 trace per margin carries caps 5/10/20 and both certificates
                    for m in margins:
                        fn = functools.partial(RP.repair_target, margin=m, iters=iters_wi, sign=1.0, cert=RP.certified)
                        r, log, px = traced_call(fn, cx, t20, base)
                        runs[f"m{m:.2f}"] = summarise({"margin": m, "iters": iters_wi, "cert": "full"}, r, log, m)
                rec["runs"] = runs
            except Exception as e:                                       # noqa: BLE001
                rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            rec["t_s"] = round(time.time() - t0, 2)
            fo.write(json.dumps(rec) + "\n")
            fo.flush()
            print("done", tok, rec["t_s"], rec.get("error", ""), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("primary", "whatif", "control"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", nargs="*", default=[])
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--mutate", default="none")
    a = ap.parse_args()
    toks, logs, _, ix, _ = load_inputs()
    if a.tokens:
        want = a.tokens
    else:
        rep = json.load(open(OUT_BUILD + "/n4_build_report.json", encoding="utf-8"))
        want = rep["unrepaired_dac0"]["N4D"]
    si, sn = (int(x) for x in a.shard.split("/"))
    todo = {t: logs[ix[t]] for i, t in enumerate(sorted(want)) if i % sn == si}
    run_tokens(todo, a.mode, a.out, a.mutate)
    return 0


if __name__ == "__main__":
    sys.exit(main())
