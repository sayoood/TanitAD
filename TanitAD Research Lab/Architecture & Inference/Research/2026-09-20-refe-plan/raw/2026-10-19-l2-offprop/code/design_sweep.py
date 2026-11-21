#!/usr/bin/env python3
"""RECIPE SELECTION (MM rulings 2026-10-19, addendum 14:10): choose the optimiser recipe (lam_bce, lr, epochs, prox) of arms H / H0 / H-null.  No real unplanted label is read or trained on.

THE RULE (committed HERE before the sweep ran; `choose()` / `rank()` implement it literally):
  HARD CONSTRAINT   C-REACH: the planted perfect feature (REAL tokens + the planted arm's label-derived feature; reads the PLANTED arm only) gives A1-OOF >= 0.98 on BOTH training seeds.
                    Seed 0 is read for every setting; seed 1 is read walking DOWN the ranked list (the pick is the first ranked setting that also clears seed 1, which is equivalent to filtering on
                    both seeds first, at a fraction of the cost).  If NO setting clears it: INSTRUMENT DEFECT (no outcome, no recipe).
  SUBJECT TO        in-fan(H) >= in-fan(H0) - 0.01 in EVERY semi-synthetic SIGNAL world (sigA, sigB), where H0 = the SAME setting with the pairwise weight 0 (same init, folds, seed, minibatch stream)
  MAXIMISE          the mean semi-synthetic A1-OOF (pooled lon + lat) of H over the signal worlds
  TIES              within 0.005 of the best -> the smaller lambda = pairwise / in-fan weight = 1 / lam_bce (so the LARGER lam_bce), then the larger prox, then fewer optimiser steps, then the smaller lr
  FALLBACK          if no C-REACH-passing setting satisfies the in-fan clause: take the C-REACH-passing setting with the best in-fan margin (min over signal worlds of in-fan(H) - in-fan(H0)) and REPORT
                    that the clause may bind in the real run
  REPORTED, NOT GATING   the null world (beta 0: no learnable within-slot ordering): |A1(H) - A1(P2)| and A1(H-null, global shuffle) - A1(P2), for the pick (stage 3).
Semi-synthetic worlds = REAL tokens + SYNTHETIC labels (`synthetic.semi_synthetic`): sigA (beta 0.5, alpha 0.25), sigB (beta 0.5, alpha 0.5), null (beta 0, alpha 0.4).  One training seed per cell in the sweep.

    python design_sweep.py sweep  --shard I/N --out part_I.json [--threads 1]     one shard of the grid (resumable: finished settings are kept)
    python design_sweep.py choose --parts part_*.json --out sweep_semi.json         merge the shards, apply the rule, write the ranked list (no recipe file yet)
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import l2lib  # noqa: E402
import synthetic  # noqa: E402

GRID = {"lam_bce": (1.0, 10.0, 100.0), "lr": (1e-3, 3e-3), "epochs": (5, 15, 50), "prox": (0.0, 0.3, 3.0, 30.0)}
EXTRA = [{"lam_bce": 10.0, "lr": 3e-4, "epochs": e, "prox": 0.0} for e in (15, 50)]           # the low-lr corner, so the table shows where C-REACH starts to fail
REACH_BAR = 0.98
R1_MARGIN = 0.01
TIE = 0.005
SIGNAL = ("sigA", "sigB")
FEATS = "D:/Projects/TanitAD-artifacts/l2_offprop/feats.npz"


def settings() -> list[dict]:
    s = [dict(lam_bce=a, lr=b, epochs=c, prox=d) for a, b, c, d in itertools.product(GRID["lam_bce"], GRID["lr"], GRID["epochs"], GRID["prox"])] + [dict(x) for x in EXTRA]
    return sorted(s, key=lambda r: (-r["epochs"], -r["lr"], r["lam_bce"], r["prox"]))        # cost-descending, so a round-robin shard split balances


def key_of(r: dict) -> str:
    return f"lam{r['lam_bce']:g}_lr{r['lr']:g}_ep{r['epochs']}_prox{r['prox']:g}"


def build_worlds(real):
    return {"sigA": synthetic.semi_synthetic(real, beta=0.5, alpha=0.25, seed=21), "sigB": synthetic.semi_synthetic(real, beta=0.5, alpha=0.5, seed=22),
            "null": synthetic.semi_synthetic(real, beta=0.0, alpha=0.4, seed=23)}


def _read(D, o, ref=None):
    ev = l2lib.evaluate_arm(D, o["ep_in"], o["ep_p"], ref)
    return ev["all"]["A1"]["point"], ev["infan_top8_ep_concordance"]["point"], ev


def reach_read(real, rec, seed: int) -> dict:
    """C-REACH on the REAL tokens (planted arm only): the gate's own computation, for one seed"""
    o = l2lib.oof_scores(real, l2lib.ArmSpec(f"planted_s{seed}", "planted", seed=seed), rec)
    ev = l2lib.evaluate_arm(real, o["ep_in"], o["ep_p"])
    return {"a1": ev["all"]["A1"]["point"], "lo": ev["all"]["A1"]["lo"], "steps_per_fold": [f["steps"] for f in o["diag"]["folds"]]}


def world_read(D, base, rec, seed: int = 0) -> dict:
    h = l2lib.oof_scores(D, l2lib.ArmSpec("H", "H", seed=seed), rec)
    h0 = l2lib.oof_scores(D, l2lib.ArmSpec("H0", "H0", seed=seed), rec)
    a1, inf, _ = _read(D, h)
    a1_0, inf_0, _ = _read(D, h0)
    return {"a1": a1, "infan": inf, "a1_h0": a1_0, "infan_h0": inf_0, "infan_minus_h0": inf - inf_0, "a1_gain_over_p2": a1 - base["a1_p2"], "steps_fold0": h["diag"]["folds"][0]["steps"]}


def cmd_sweep(a):
    import torch
    torch.set_num_threads(a.threads)
    i, n = (int(x) for x in a.shard.split("/"))
    t0 = time.time()
    real = l2lib.load_real(a.feats)
    wd = build_worlds(real)
    base = {}
    for w, D in wd.items():
        p2 = l2lib.oof_scores(D, l2lib.ArmSpec("P2", "P2"), l2lib.Recipe())
        x, y, _ = _read(D, p2)
        base[w] = {"a1_p2": x, "infan_p2": y}
    rows = json.load(open(a.out))["rows"] if os.path.exists(a.out) else {}
    mine = [s for k, s in enumerate(settings()) if k % n == i]
    for s in mine:
        key = key_of(s)
        if key in rows:
            continue
        t1 = time.time()
        rec = l2lib.Recipe(**s)
        row = {"recipe": rec.as_dict(), "reach_s0": reach_read(real, rec, 0)}
        row["reach_pass_s0"] = bool(row["reach_s0"]["a1"] >= REACH_BAR)
        if row["reach_pass_s0"]:
            row["worlds"] = {w: world_read(wd[w], base[w], rec) for w in SIGNAL}
        row["seconds"] = round(time.time() - t1, 1)
        rows[key] = row
        json.dump(l2lib.jsonable({"base": base, "rows": rows, "shard": a.shard}), open(a.out + ".tmp", "w"), indent=1)
        os.replace(a.out + ".tmp", a.out)
        print(key, "reach_s0", round(row["reach_s0"]["a1"], 4), "PASS" if row["reach_pass_s0"] else "fail",
              {w: (round(r["a1"], 3), round(r["infan_minus_h0"], 3)) for w, r in row.get("worlds", {}).items()}, f"{row['seconds']}s", flush=True)
    print("ZZSWEEP_SHARD_DONE", a.shard, round(time.time() - t0, 1), "s", flush=True)
    return 0


def summarise_row(k: str, r: dict) -> dict:
    out = {"key": k, "lam": r["recipe"]["lam_bce"], "lr": r["recipe"]["lr"], "epochs": r["recipe"]["epochs"], "prox": r["recipe"]["prox"], "reach_s0": r["reach_s0"]["a1"], "reach_pass_s0": r["reach_pass_s0"]}
    if r["reach_pass_s0"]:
        ws = r["worlds"]
        out["mean_signal_a1"] = float(np.mean([ws[w]["a1"] for w in SIGNAL]))
        out["margin"] = float(min(ws[w]["infan_minus_h0"] for w in SIGNAL))
        out["r1"] = bool(out["margin"] >= -R1_MARGIN)
        out["steps"] = int(r["worlds"][SIGNAL[0]]["steps_fold0"])
    return out


def rank(rows: list[dict]) -> list[dict]:
    """the rule's ordering, by repeated selection: best objective; everything within TIE of it is a tie -> larger lam_bce (smaller lambda), larger prox, fewer steps, smaller lr."""
    rem = list(rows)
    out = []
    while rem:
        best = max(r["mean_signal_a1"] for r in rem)
        tied = [r for r in rem if best - r["mean_signal_a1"] <= TIE]
        pick = sorted(tied, key=lambda r: (-r["lam"], -r["prox"], r["steps"], r["lr"], r["key"]))[0]
        out.append(pick)
        rem.remove(pick)
    return out


def choose(res: dict) -> dict:
    sm = [summarise_row(k, r) for k, r in res["rows"].items()]
    reach = [r for r in sm if r["reach_pass_s0"]]
    if not reach:
        return {"rule": "INSTRUMENT DEFECT: no setting of the grid clears C-REACH >= 0.98 on seed 0", "ranked": [], "n_settings": len(sm), "n_reach_pass_s0": 0, "n_r1": 0}
    ok = [r for r in reach if r["r1"]]
    if ok:
        return {"rule": "C-REACH hard; max semi-synthetic A1-OOF subject to in-fan(H) >= in-fan(H0) - 0.01 in every signal world; ties -> smaller lambda", "ranked": rank(ok), "n_settings": len(sm),
                "n_reach_pass_s0": len(reach), "n_r1": len(ok), "clause_may_bind": False}
    ranked = sorted(reach, key=lambda r: (-r["margin"], -r["mean_signal_a1"], r["key"]))
    return {"rule": "FALLBACK (no C-REACH-passing setting satisfies the in-fan clause): the best in-fan margin; the clause MAY BIND in the real run", "ranked": ranked, "n_settings": len(sm),
            "n_reach_pass_s0": len(reach), "n_r1": 0, "clause_may_bind": True}


def cmd_choose(a):
    parts = sorted(p for pat in a.parts for p in glob.glob(pat))
    res = {"base": None, "rows": {}, "parts": parts}
    for p in parts:
        r = json.load(open(p))
        res["base"] = res["base"] or r["base"]
        res["rows"].update(r["rows"])
    res["choice"] = choose(res)
    res["n_expected_settings"] = len(settings())
    json.dump(l2lib.jsonable(res), open(a.out, "w"), indent=1)
    ch = res["choice"]
    print("ZZSWEEP_CHOSEN", len(res["rows"]), "of", len(settings()), "settings;", ch["rule"], "| reach-pass(s0):", ch["n_reach_pass_s0"], "| R1-pass:", ch["n_r1"],
          "| top:", ch["ranked"][0]["key"] if ch["ranked"] else None, flush=True)
    return 0 if ch["ranked"] and len(res["rows"]) == len(settings()) else 3


def main(argv=None):
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("sweep")
    p.add_argument("--shard", default="0/1")
    p.add_argument("--out", required=True)
    p.add_argument("--feats", default=FEATS)
    p.add_argument("--threads", type=int, default=1)
    p.set_defaults(fn=cmd_sweep)
    p = sp.add_parser("choose")
    p.add_argument("--parts", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_choose)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
