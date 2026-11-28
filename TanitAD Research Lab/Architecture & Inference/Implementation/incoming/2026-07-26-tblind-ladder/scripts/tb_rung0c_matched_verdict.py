#!/usr/bin/env python3
"""RUNG 0c — the DEPLOYABLE ``T_blind`` with a readout-MATCHED comparator.

Rung 0's pre-registered primary compared ``a_imagination__own__roSTR`` against an
``op``-decoded frozen-last control, because the matched one had never been rolled
out. Rung 0's own sensitivity (P1-S) then measured that this substitution drives
``T_blind`` from **185 steps to 1** in the privileged regime, where the matched
answer is 185 — i.e. the substitution is **destructive, not conservative**, and
the unmatched contrast cannot answer the question either way.

``tb_rung0b_matched_arms.py`` built the four missing arms on pod2. This script
adjudicates on them.

⛔ WINDOW-SET IDENTITY GATE FIRST. Two arms that already exist in the committed
dense dump were re-rolled as anchors. Their dense ``de`` must reproduce and the
``eid``/``t0`` ordering must match exactly, or the new arms are not poolable with
the committed ones and the run is BLOCKED rather than silently mixed.

Estimator: paired episode-cluster bootstrap (``taniteval/ci.py``, B = 2000,
seed 0, resampling unit = val episode).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tb_rung0 import (DT, GRID, B_BOOT, SEED, draws_for, paired_at,  # noqa: E402
                      separated_better_interval, t_blind)

ANCHORS = ("a_imagination__own", "b_frozenlast__own")
ANCHOR_TOL = 1e-4          # m — float-kernel noise between two encode passes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--new", required=True, help="perwindow_sweep_K185.pt from rung0b")
    ap.add_argument("--committed", required=True, help="bi_perwindow_compact.pt")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    new = torch.load(a.new, map_location="cpu", weights_only=False)
    old = torch.load(a.committed, map_location="cpu", weights_only=False)
    gt = new["gt"]
    de_new = {k: torch.linalg.norm(v - gt, dim=-1).double().numpy()
              for k, v in new["pred"].items()}
    de_old = {k: v.double().numpy() for k, v in old["dense_de_headline"].items()}
    eid = [str(x) for x in new["eid"]]
    draws, n_ep = draws_for(eid)

    # ---- ⛔ WINDOW-SET IDENTITY GATE -------------------------------------- #
    gate = {"eid_identical": bool([str(x) for x in old["eid"]] == eid),
            "t0_identical": bool(list(map(int, old["t0"])) == list(map(int, new["t0"]))),
            "n_windows_new": int(gt.shape[0]),
            "n_windows_committed": int(de_old["a_imagination__own"].shape[0]),
            "anchors": {}}
    for arm in ANCHORS:
        if arm not in de_new or arm not in de_old:
            gate["anchors"][arm] = {"present": False}
            continue
        d = float(np.abs(de_new[arm] - de_old[arm]).max())
        gate["anchors"][arm] = {"present": True, "max_abs_diff_m": d,
                                "within_tol": bool(d < ANCHOR_TOL),
                                "tol_m": ANCHOR_TOL}
    gate["GATE_PASS"] = bool(
        gate["eid_identical"] and gate["t0_identical"]
        and all(v.get("within_tol") for v in gate["anchors"].values()))
    print("WINDOW-SET IDENTITY GATE:", "PASS" if gate["GATE_PASS"] else "FAIL")
    print(json.dumps(gate, indent=1))
    if not gate["GATE_PASS"]:
        (out / "rung0c_BLOCKED.json").write_text(
            json.dumps({"blocked": True, "gate": gate}, indent=2), encoding="utf-8")
        return 2

    # pool: committed dense arms + the newly built matched controls
    de = dict(de_old)
    de.update(de_new)

    PAIRS = {
        # regime                        arm (a)                       matched (b)
        "deployable_own__op (committed)": ("a_imagination__own", "b_frozenlast__own"),
        "deployable_own__str NEW":    ("a_imagination__own__roSTR",
                                           "b_frozenlast__own__roSTR"),
        "deployable_own__tac NEW":    ("a_imagination__own__roTAC",
                                           "b_frozenlast__own__roTAC"),
        "deployable_hold__op (committed)": ("a_imagination__hold", "b_frozenlast__hold"),
        "deployable_hold__str NEW":    ("a_imagination__hold__roSTR",
                                           "b_frozenlast__hold__roSTR"),
        "privileged_true__op (committed)": ("a_imagination__true", "b_frozenlast__true"),
        "privileged_true__str (committed)": ("a_imagination__true__roSTR",
                                             "b_frozenlast__true__roSTR"),
    }
    res = {"meta": {"n_windows": int(gt.shape[0]), "n_episode_clusters": n_ep,
                    "K_max": int(gt.shape[1]), "n_boot": B_BOOT, "seed": SEED,
                    "estimator": "paired_episode_cluster_bootstrap",
                    "arm_ckpt": new["meta"].get("arm_ckpt"),
                    "ckpt_step": new["meta"].get("ckpt_step"),
                    "new_arms": sorted(de_new)},
           "window_set_identity_gate": gate, "T_blind_matched": {},
           "paired_de_at_grid": {}}
    for label, (aa, bb) in PAIRS.items():
        if aa not in de or bb not in de:
            res["T_blind_matched"][label] = {"MISSING": [x for x in (aa, bb)
                                                         if x not in de]}
            continue
        res["T_blind_matched"][label] = t_blind(de[aa], de[bb], draws,
                                                label_a=aa, label_b=bb)
        res["paired_de_at_grid"][label] = {
            f"{n * DT:g}s": paired_at(de[aa], de[bb], draws, n) for n in GRID}

    # the pre-registered buckets, applied to the MATCHED deployable numbers
    base = res["T_blind_matched"]["deployable_own__op (committed)"]["T_blind_steps"]
    verdict = {}
    for lv, key in (("str", "deployable_own__str NEW"),
                    ("tac", "deployable_own__tac NEW")):
        blk = res["T_blind_matched"].get(key, {})
        t = blk.get("T_blind_steps")
        if t is None:
            verdict[lv] = {"MISSING": True}
            continue
        verdict[lv] = {
            "T_blind_steps": t, "T_blind_s": round(t * DT, 2),
            "T_blind_ci95_s": blk["T_blind_ci95_s"],
            "baseline_steps": base,
            "bucket": ("CONFIRM" if t >= 16 else "PARTIAL" if t >= 10 else "REFUTE"),
            "beats_cv": separated_better_interval(
                de[PAIRS[key][0]], de["d_constant_velocity"], draws)}
    res["VERDICT_MATCHED"] = {
        "rule": "PRE_REGISTRATION.md §2 — CONFIRM >=16 steps, PARTIAL 10..15, "
                "REFUTE <=9; baseline 8 steps",
        "per_readout": verdict}
    (out / "rung0c_matched_tblind.json").write_text(
        json.dumps(res, indent=2, default=float), encoding="utf-8")

    print("\nMATCHED T_blind (both arms decoded with the SAME readout):")
    for k, v in res["T_blind_matched"].items():
        if "T_blind_steps" in v:
            print("   %-34s %3d steps (%4.1f s)  CI %s  floor-frac %.3f"
                  % (k, v["T_blind_steps"], v["T_blind_s"], v["T_blind_ci95_s"],
                     v["frac_draws_T_blind_at_floor_1step"]))
        else:
            print("   %-34s %s" % (k, v))
    print("\nVERDICT (matched):", json.dumps(verdict, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
