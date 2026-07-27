#!/usr/bin/env python3
"""Compact the pod's latent-ablation sweep into the repo-sized per-window dump.

The full ``latab_sweep_K185.pt`` (~180 MB, 41 arms x pred/psi/pred_speed/latent)
lives on pod2. What every bar, horizon and interval in ``LATENT_ABLATION.md``
needs is:

* dense per-window ``de`` ``[599,185]`` for every arm — the only thing
  ``T_blind`` / ``de@N`` / ``ade_0_2s`` / ``T_useful`` / beats-CV read — plus the
  two comparator-free floors;
* the FIXED-POINT probe per window for the IMAGINATION arms (the probe's
  question is about the imagined latent), and its per-step MEAN for every arm;
* the window bookkeeping that makes the identity gate checkable.

⛔ THE PLUMBING SELF-TEST IS RESOLVED HERE, at the full K = 185, and its result is
written into the compaction as NUMBERS rather than the tensors being carried — a
bit-identical copy of an arm is not data.

Usage (pod2):
    python3 la_compact.py --dump /workspace/latab/perwindow/latab_sweep_K185.pt \
        --out /workspace/latab/perwindow/latab_perwindow_compact.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

#: self-test arm -> the arm it must be BIT-IDENTICAL to (algebraic reduction of
#: the identity permutation). Direction 1 of the plumbing self-test.
SELFTEST = {
    "selftest__shuffled_id": "imagination__a0.25",
    "selftest__shufobs_id": "full_obs__a0.25",
    "selftest__frozother_id": "frozen_last__a0.25",
}
#: alpha=1 must reduce to the zero-order hold. Rung 1 proved this on the
#: imagination arm only; here it is proved on the FROZEN arm as well.
HOLD_EQUIV = {"imagination__a1": "anchor_a_hold",
              "frozen_last__a1": "anchor_b_hold"}
#: arms whose per-window FIXED-POINT probe is carried (the rest keep step means)
KEEP_LATENT_PERWINDOW = ("imagination__a0", "imagination__a0.25",
                         "imagination__a0.75", "imagination__a1",
                         "frozen_last__a1", "zero_latent__a1",
                         "shuffled__a1", "full_obs__a1")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = torch.load(a.dump, map_location="cpu", weights_only=False)
    gt = d["gt"]
    de = {k: torch.linalg.norm(v - gt, dim=-1).float()
          for k, v in d["pred"].items()}

    # ---- ⛔ PLUMBING SELF-TEST, direction 1: the reductions ---------------- #
    st = {}
    for arm, ref in list(SELFTEST.items()) + list(HOLD_EQUIV.items()):
        m = float((d["pred"][arm] - d["pred"][ref]).abs().max())
        st[arm] = {"must_equal": ref, "max_abs_diff_m": m,
                   "bit_identical": bool(m == 0.0)}
    # ---- ⛔ direction 2: ANTI-NO-OP. No ablation may equal the intact arm -- #
    noop = {}
    for al in ("0", "0.25", "0.75", "1"):
        base = d["pred"][f"imagination__a{al}"]
        for k, v in d["pred"].items():
            if k.endswith(f"__a{al}") and not k.startswith("imagination"):
                noop[k] = float((v - base).abs().max())
    st["_anti_noop_min_abs_diff_vs_intact_m"] = round(min(noop.values()), 6)
    st["_anti_noop_arms_identical_to_intact"] = sorted(
        k for k, v in noop.items() if v == 0.0)
    st["_anti_noop_per_arm_m"] = {k: round(v, 6) for k, v in sorted(noop.items())}
    st["SELFTEST_PASS"] = bool(
        all(st[k]["bit_identical"] for k in
            list(SELFTEST) + list(HOLD_EQUIV))
        and not st["_anti_noop_arms_identical_to_intact"])

    de["d_constant_velocity"] = torch.linalg.norm(d["cv"] - gt, dim=-1).float()
    de["d2_hold_v0"] = torch.linalg.norm(d["hold_v0"] - gt, dim=-1).float()
    for k in SELFTEST:                    # a bit-identical copy is not data
        de.pop(k, None)

    lat = d.get("latent", {})
    lat_mean = {n: {q: v.mean(dim=0) for q, v in blk.items()}
                for n, blk in lat.items()}
    lat_pw = {n: lat[n] for n in KEEP_LATENT_PERWINDOW if n in lat}

    out = {"dense_de": de,
           "latent_perwindow": lat_pw, "latent_stepmean": lat_mean,
           "psi": {k: d["psi"][k] for k in KEEP_LATENT_PERWINDOW
                   if k in d["psi"]},
           "pred_speed": {k: d["pred_speed"][k] for k in d["pred_speed"]},
           "v_last": d["speed"], "head_deg": d["head_deg"],
           "eid": d["eid"], "t0": d["t0"],
           "selftest": st, "meta": d["meta"],
           "note": ("dense per-window de [N,K] for every latent-ablation arm; "
                    "the plumbing self-test arms are bit-identical copies and "
                    "are recorded in `selftest` instead of being carried.")}
    torch.save(out, a.out)
    print(json.dumps(st, indent=1)[:2000])
    print(f"[compact] {a.out} ({Path(a.out).stat().st_size / 1e6:.1f} MB, "
          f"{len(de)} arms, {len(lat_pw)} latent-perwindow)")
    return 0 if st["SELFTEST_PASS"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
