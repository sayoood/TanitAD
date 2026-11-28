#!/usr/bin/env python3
"""YAXN / P2 AXNOISE trainer: a THIN WRAPPER around YAX's unchanged ax_ft.py (policy real = AXON), PREREG_AXD.md section 6.

The ONLY change to the training: the ax column (ego[:, 2]) of TRAINING batches gets  + sigma * N(0,1)  drawn from a DEDICATED torch.Generator(seed + 7919) at the make_batch call (ax_ft.py:204),
so the global RNG and the row-permutation generator (torch.Generator(seed)) are untouched and the row order equals AXON's. Gate and VAL batches are fed honestly (the wrapper switches the noise OFF
inside ax_ft.gates / ax_ft.evaluate). All of ax_ft's own gates (GA_FEED, GB_STEP0, G_REACH with zero + positive controls, G_DECL) run unchanged on the honest gate batch.

Added gates (all in gates.json BEFORE step 1; red -> rc 3 ZZAXFT_GATE_RED), each with a deliberate-regression arm (--axn-regress):
  G_NOISE_STD   16 x 512 TRAINING-mode draws through the real make_batch path on a temporary generator: |std(fed-real)/sigma - 1| <= 0.10 (sigma 0: every difference exactly 0.0)   RED: noise_scaled, noise_not_applied, sigma0_offset
  G_HONEST_GATE the gate batch is fed exactly real ax (max |fed - expected| == 0.0)                                                                                              RED: val_noised
After training (axnoise.json, final.pt relabelled arm AXNOISE; failure -> rc 5 ZZAXN_POST_RED and final.pt is renamed final.RED.pt):
  G_NOISE_N     number of noised draws == steps * accum * batch (every training sample was perturbed, none extra)
  G_NOISE_STD_TRAIN  std over the ACTUAL training draws within 10 % of sigma (widened to 3 standard errors when n < 450: mini mechanics runs only)
  G_HONEST_VAL  every VAL batch of the run (incl. the ax=0 diagnostic eval) was fed exactly its expected ax (max |diff| == 0.0)
Chain-level (axn_chain.py gates_axn): sigma 0 reproduces AXON's 5-step weights bit for bit; sigma 1 does NOT.

    python axd_ft.py --ax-noise-sigma 1.0 --cache <dir> --manifest <json> --ax-policy real --steps 200 --lr 5e-5 --seed 0 --out <dir>      (every other flag is ax_ft.py's)
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
YAX = HERE.parent / "YAX"
sys.path.insert(0, str(YAX))
import torch  # noqa: E402
import ax_ft  # noqa: E402  (YAX, unchanged, read-only)

AX = ax_ft.AX
REGRESS = ("none", "noise_not_applied", "noise_scaled", "val_noised", "sigma0_offset")
STD_TOL, PRE_DRAWS_REPEAT = 0.10, 16


class Stats:
    def __init__(self):
        self.n, self.s, self.ss, self.maxabs, self.calls = 0, 0.0, 0.0, 0.0, 0

    def add(self, diff):
        d = diff.detach().double()
        self.n += d.numel()
        self.calls += 1
        self.s += float(d.sum())
        self.ss += float((d * d).sum())
        self.maxabs = max(self.maxabs, float(d.abs().max())) if d.numel() else self.maxabs

    def mean(self):
        return self.s / self.n if self.n else float("nan")

    def std(self):
        return math.sqrt(max(self.ss / self.n - self.mean() ** 2, 0.0)) if self.n else float("nan")


class State:
    sigma, regress, mode = 0.0, "none", "train"
    gen, gen_leak = None, None
    train, honest = Stats(), Stats()


S = State()
_orig_make_batch, _orig_gates, _orig_evaluate = ax_ft.make_batch, ax_ft.gates, ax_ft.evaluate


def std_gate(stats: Stats, sigma: float, tol=STD_TOL, widen_below_n=0) -> dict:
    """The G_NOISE_STD decision as a pure function (unit-tested with RED inputs). sigma 0: every difference exactly 0.0."""
    if stats.n == 0:
        return {"n": 0, "pass": False, "why": "no draws"}
    if sigma == 0.0:
        return {"n": stats.n, "maxabs": stats.maxabs, "pass": stats.maxabs == 0.0}
    t = max(tol, 3.0 / math.sqrt(2.0 * stats.n)) if stats.n < widen_below_n else tol
    ratio = stats.std() / sigma
    mt = max(0.1 * sigma, 3.0 * sigma / math.sqrt(stats.n))                  # the mean of n draws has standard error sigma/sqrt(n): never demand less than 3 of them
    return {"n": stats.n, "std": stats.std(), "mean": stats.mean(), "ratio": ratio, "tol": t, "mean_tol": mt, "pass": bool(abs(ratio - 1.0) <= t and abs(stats.mean()) <= mt)}


def honest_gate(stats: Stats) -> dict:
    return {"n": stats.n, "calls": stats.calls, "max_abs_diff": stats.maxabs, "pass": bool(stats.n > 0 and stats.maxabs == 0.0)}


@contextlib.contextmanager
def mode(m):
    old = S.mode
    S.mode = m
    try:
        yield
    finally:
        S.mode = old


def make_batch_noised(rows, ids, cache, policy, regress="none"):
    sc, eg, go, tg = _orig_make_batch(rows, ids, cache, policy, regress)
    real = ax_ft.real_ego(rows, ids)[:, AX]                                   # independent recompute from the rows: the measurement does not trust the thing it measures
    expected = torch.zeros_like(real) if policy == "zero" else real
    if S.mode == "train":
        if policy != "real":
            raise RuntimeError("axd_ft: noise is defined for ax-policy real only")
        if S.regress != "noise_not_applied":
            eff = S.sigma * torch.randn(len(ids), generator=S.gen)           # dedicated generator: global RNG and the permutation generator are never touched
            if S.regress == "noise_scaled":
                eff = 0.5 * eff
            if S.regress == "sigma0_offset":
                eff = eff + 0.01
            eg = eg.clone()
            eg[:, AX] = eg[:, AX] + eff
    elif S.regress == "val_noised":                                           # RED arm: noise leaks into gate/VAL batches
        eg = eg.clone()
        eg[:, AX] = eg[:, AX] + 1.0 * torch.randn(len(ids), generator=S.gen_leak)
    (S.train if S.mode == "train" else S.honest).add(eg[:, AX] - expected)
    return sc, eg, go, tg


def evaluate_honest(*a, **k):
    with mode("honest"):
        return _orig_evaluate(*a, **k)


def gates_patched(model, cfg, base_model, rows, cache, policy, regress, n):
    with mode("honest"):
        out = _orig_gates(model, cfg, base_model, rows, cache, policy, regress, n)        # ax_ft's own gates, unchanged, on the honest gate batch
    ids = list(range(min(512, len(rows))))
    keep = (S.gen, S.train)
    S.gen, S.train = torch.Generator().manual_seed(S.seed + S.off), Stats()                  # temporary generator: the real stream starts at draw 0
    try:
        with mode("train"):
            for _ in range(PRE_DRAWS_REPEAT):
                make_batch_noised(rows, ids, cache, policy, regress)
        pre = S.train
    finally:
        S.gen, S.train = keep
    out["AXN_G_NOISE_STD"] = std_gate(pre, S.sigma)
    out["AXN_G_HONEST_GATE"] = honest_gate(S.honest)
    out["AXN_sigma"], out["AXN_regress"] = S.sigma, S.regress
    out["pass"] = bool(out["pass"] and out["AXN_G_NOISE_STD"]["pass"] and out["AXN_G_HONEST_GATE"]["pass"])
    return out


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--ax-noise-sigma", type=float, required=True)                    # no default: a default of 0 would silently disable the lever
    ap.add_argument("--noise-seed-offset", type=int, default=7919)
    ap.add_argument("--axn-regress", choices=REGRESS, default="none")
    mine, rest = ap.parse_known_args(argv)
    p2 = argparse.ArgumentParser(add_help=False)
    for f, t in (("--seed", int), ("--steps", int), ("--batch", int), ("--accum", int), ("--out", str), ("--ax-policy", str)):
        p2.add_argument(f, type=t, default=None)
    p2.add_argument("--gates-only", action="store_true")
    q = p2.parse_known_args(rest)[0]
    if q.ax_policy != "real":
        print("REFUSING: axd_ft is AXNOISE = policy real + noise; --ax-policy real is required")
        return 4
    if mine.ax_noise_sigma < 0:
        print("REFUSING: sigma < 0")
        return 4
    S.sigma, S.regress, S.seed, S.off = float(mine.ax_noise_sigma), mine.axn_regress, int(q.seed or 0), mine.noise_seed_offset
    S.gen = torch.Generator().manual_seed(S.seed + S.off)
    S.gen_leak = torch.Generator().manual_seed(S.seed + S.off + 1)
    S.train, S.honest = Stats(), Stats()
    ax_ft.make_batch, ax_ft.gates, ax_ft.evaluate = make_batch_noised, gates_patched, evaluate_honest
    rc = ax_ft.main(rest)
    if rc != 0 or q.gates_only:
        return rc
    out = Path(q.out)
    post = {"sigma": S.sigma, "regress": S.regress, "seed": S.seed, "noise_seed": S.seed + S.off, "steps": q.steps, "batch": q.batch or 32, "accum": q.accum or 8}
    exp_n = post["steps"] * post["batch"] * post["accum"]
    post["G_NOISE_N"] = {"n": S.train.n, "expected": exp_n, "pass": S.train.n == exp_n}
    post["G_NOISE_STD_TRAIN"] = std_gate(S.train, S.sigma, widen_below_n=450)
    post["G_HONEST_VAL"] = honest_gate(S.honest)
    post["stats_train"] = {"n": S.train.n, "mean": S.train.mean(), "std": S.train.std(), "maxabs": S.train.maxabs}
    post["code_sha256"] = ax_ft.YC.sha256_file(__file__)
    post["pass"] = bool(post["G_NOISE_N"]["pass"] and post["G_NOISE_STD_TRAIN"]["pass"] and post["G_HONEST_VAL"]["pass"])
    fp = out / "final.pt"
    fin = torch.load(fp, map_location="cpu", weights_only=False)
    fin["arm"], fin["axnoise"] = "AXNOISE", post                                      # relabel: the composed checkpoint's meta must not claim to be plain AXON
    tmp = out / "final.pt.tmp"
    torch.save(fin, tmp)
    os.replace(tmp, fp)
    ax_ft.YC.jdump(post, out / "axnoise.json")
    print("axnoise", json.dumps({k: v for k, v in post.items() if k != "stats_train"}), flush=True)
    if not post["pass"]:
        os.replace(fp, out / "final.RED.pt")
        print("ZZAXN_POST_RED")
        return 5
    print("ZZAXN_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
