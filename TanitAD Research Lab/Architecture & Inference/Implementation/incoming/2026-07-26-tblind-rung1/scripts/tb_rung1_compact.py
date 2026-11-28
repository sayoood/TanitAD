#!/usr/bin/env python3
"""RUNG 1 — compact the pod's full sweep dump into the repo-sized per-window dump.

The full ``perwindow_sweep_K185.pt`` (~110 MB, 61 arms x pred/psi/pred_speed)
lives on pod2. What every bar, horizon, stratification and interval in
``TBLIND_RUNG1.md`` actually needs is:

* dense per-window ``de`` ``[599, 185]`` for every arm (that is the only thing
  ``T_blind`` / ``de@N`` / ``ade_0_2s`` / ``T_useful`` / beats-CV read), plus the
  two comparator-free floors ``d_constant_velocity`` and ``d2_hold_v0``;
* ``psi`` + ``pred_speed`` for the handful of arms whose ACTION sequence is
  reconstructed (:func:`blindimag.reconstruct_kinematic_actions`);
* the window bookkeeping that makes the identity gate checkable.

⛔ The three PLUMBING SELF-TEST arms are checked HERE, against the arms they must
reduce to, at the full K = 185 — and their max abs difference is written into the
compaction rather than the tensors being carried, because a bit-identical copy of
an arm is not data.

Usage (pod2):
    python3 tb_rung1_compact.py --dump /root/tbr1/perwindow/perwindow_sweep_K185.pt \
        --out /root/tbr1/perwindow/rung1_perwindow_compact.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

SELFTEST = {                      # arm -> the arm it must be BIT-IDENTICAL to
    "a_selftest__blend0": "a_imagination__own__roSTR",
    "a_selftest__every1": "a_imagination__own__roSTR",
    "a_selftest__blend1": "a_imagination__hold__roSTR",
}
#: arms whose raw own-kinematic action sequence is reconstructed over all windows
KEEP_ACTION = ("a_imagination__own__roSTR", "a_imagination__hold__roSTR",
               "b_frozenlast__own__roSTR", "a_gtkin", "a_blend0.5", "a_ema0.8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = torch.load(a.dump, map_location="cpu", weights_only=False)
    gt = d["gt"]
    de = {k: torch.linalg.norm(v - gt, dim=-1).float()
          for k, v in d["pred"].items()}

    # ---- ⛔ PLUMBING SELF-TEST at the FULL horizon ------------------------ #
    selftest = {}
    for arm, ref in SELFTEST.items():
        m = float((d["pred"][arm] - d["pred"][ref]).abs().max())
        selftest[arm] = {"must_equal": ref, "max_abs_diff_m": m,
                         "bit_identical": bool(m == 0.0)}
    # ---- ANTI-NO-OP: every filter arm must move the path ------------------ #
    own = d["pred"]["a_imagination__own__roSTR"]
    noop = {k: float((v - own).abs().max()) for k, v in d["pred"].items()
            if k.startswith("a_") and k not in SELFTEST
            and k != "a_imagination__own__roSTR"}
    selftest["_anti_noop_min_abs_diff_vs_own_m"] = round(min(noop.values()), 6)
    selftest["_anti_noop_arms_identical_to_own"] = sorted(
        k for k, v in noop.items() if v == 0.0)
    selftest["SELFTEST_PASS"] = bool(
        all(v["bit_identical"] for v in
            (selftest[k] for k in SELFTEST))
        and not selftest["_anti_noop_arms_identical_to_own"])

    de["d_constant_velocity"] = torch.linalg.norm(d["cv"] - gt, dim=-1).float()
    de["d2_hold_v0"] = torch.linalg.norm(d["hold_v0"] - gt, dim=-1).float()
    for k in SELFTEST:                       # a bit-identical copy is not data
        de.pop(k, None)

    out = {"dense_de": de,
           "psi": {k: d["psi"][k] for k in KEEP_ACTION if k in d["psi"]},
           "pred_speed": {k: d["pred_speed"][k] for k in KEEP_ACTION
                          if k in d["pred_speed"]},
           "v_last": d["speed"], "head_deg": d["head_deg"],
           "eid": d["eid"], "t0": d["t0"],
           "selftest": selftest, "meta": d["meta"],
           "note": ("dense per-window de [N,K] for every Rung-1 arm; the three "
                    "plumbing self-test arms are bit-identical copies and are "
                    "recorded in `selftest` instead of being carried.")}
    torch.save(out, a.out)
    print(json.dumps(selftest, indent=1))
    print(f"[compact] {a.out} "
          f"({Path(a.out).stat().st_size / 1e6:.1f} MB, {len(de)} arms)")
    return 0 if selftest["SELFTEST_PASS"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
