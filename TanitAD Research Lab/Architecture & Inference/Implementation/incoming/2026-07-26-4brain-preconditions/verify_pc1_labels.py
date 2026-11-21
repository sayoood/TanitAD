#!/usr/bin/env python3
"""PC1 verification — does `--labels-v2` actually break the route circularity?

THE CLAIM UNDER TEST (from the HPP-0 audit and the 4-Brain brief)
-----------------------------------------------------------------
    "`--labels-v2` => `route_target_v21` / `route_from_future_v3`,
     coverage 27 % -> 80.4 %."

This script measures, on REAL PhysicalAI poses, three properties of each
labeler version as the TRAINER ACTUALLY WIRES IT:

  1. ECHO RATE  -- P(route_target == _NAV_TO_ROUTE[nav_cmd]).
     This is the defect. A route target that is a deterministic function of the
     fed route command lets the head reach CE 0.0 by copying its own
     conditioning embedding to its logits. `route_skill` is then 0 BY
     CONSTRUCTION.
  2. COVERAGE   -- mean(nav_valid), the fraction of windows carrying a
     judgeable route label at all.
  3. TARGET DISTRIBUTION on the valid subset (is "straight" the silent default?)

The wirings compared are the ones in the repo, not the ones in the docs:

  v1     refb_train.FailLoudWindowDataset, labels_v2=False
             nav  = refb_labels.nav_command(poses, t_last)
             tgt  = refb_labels.route_target(nav)                <- pure lookup
  v2     refb_train.FailLoudWindowDataset, labels_v2=True   (== `--labels-v2`)
             nav  = refb_labels.nav_command_v2(poses, t_last)
             tgt  = refb_labels.route_target_v2(poses, t_last)
  v21    the labeler the audit's PC1 item #3 actually asks for
             nav  = refb_labels.nav_command_v21(poses, t_last)
             tgt  = refb_labels.route_target_v21(poses, t_last)

Read-only: loads pose arrays from an episode cache, touches no training, no
GPU, and re-selects no episodes (parity is untouched -- this measures a
property of the LABELER, not of any corpus split).

    python verify_pc1_labels.py --cache <dir with ep_*.pt> --episodes 100 \
        --out pc1_label_verification.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[5]      # .../TanitAD
sys.path.insert(0, str(_REPO / "stack"))
sys.path.insert(0, str(_REPO / "stack" / "scripts"))

import torch  # noqa: E402

import refb_labels as rl  # noqa: E402
from tanitad.data.mixing import load_episode  # noqa: E402

# The trainer's own windowing (refb_train.build_window_index): t ranges over
# [0, T - window - max_horizon) and the label is taken at t_last = t+window-1.
WINDOW = 8
MAX_HORIZON = 20

ROUTE_NAME = {rl.ROUTE_LEFT: "left", rl.ROUTE_STRAIGHT: "straight",
              rl.ROUTE_RIGHT: "right", rl.ROUTE_UNKNOWN: "unknown"}
NAV_NAME = {rl.NAV_FOLLOW: "follow", rl.NAV_LEFT: "left",
            rl.NAV_RIGHT: "right", rl.NAV_STRAIGHT: "straight"}


def _v1(poses, t_last):
    cmd, valid = rl.nav_command(poses, t_last)
    return cmd, valid, rl.route_target(cmd)


def _v2(poses, t_last):
    cmd, valid = rl.nav_command_v2(poses, t_last)
    return cmd, valid, rl.route_target_v2(poses, t_last)


def _v21(poses, t_last):
    cmd, valid = rl.nav_command_v21(poses, t_last)
    tgt, tvalid = rl.route_target_v21(poses, t_last)
    assert bool(valid) == bool(tvalid), (valid, tvalid)
    return cmd, valid, tgt


LABELERS = {"v1_default": _v1, "v2_labels_v2_flag": _v2, "v21_proposed": _v21}


def measure(cache: Path, n_eps: int, stride: int) -> dict:
    files = sorted(cache.glob("ep_*.pt"))[:n_eps]
    assert files, f"no ep_*.pt under {cache}"
    acc = {k: {"nav": [], "valid": [], "tgt": [], "eid": []} for k in LABELERS}
    lens = []
    t0 = time.time()
    for f in files:
        ep = load_episode(str(f), mmap=True)
        poses = torch.as_tensor(ep.poses)
        T = poses.shape[0]
        lens.append(int(T))
        ts = range(0, max(0, T - WINDOW - MAX_HORIZON), stride)
        for t in ts:
            t_last = t + WINDOW - 1
            for name, fn in LABELERS.items():
                cmd, valid, tgt = fn(poses, t_last)
                a = acc[name]
                a["nav"].append(int(cmd))
                a["valid"].append(bool(valid))
                a["tgt"].append(int(tgt))
                a["eid"].append(str(ep.episode_id))
    elapsed = time.time() - t0

    out = {}
    for name, a in acc.items():
        nav = torch.tensor(a["nav"])
        val = torch.tensor(a["valid"])
        tgt = torch.tensor(a["tgt"])
        lookup = torch.tensor([rl._NAV_TO_ROUTE[int(c)] for c in nav])
        echo_all = float((tgt == lookup).float().mean())
        echo_valid = (float((tgt[val] == lookup[val]).float().mean())
                      if bool(val.any()) else None)
        tv = tgt[val]
        hist = {ROUTE_NAME[k]: int(v) for k, v in
                sorted(Counter(int(x) for x in tv.tolist()).items())}
        navhist = {NAV_NAME[k]: int(v) for k, v in
                   sorted(Counter(int(x) for x in nav.tolist()).items())}
        n_valid = int(val.sum())
        maj = (max(hist.values()) / n_valid) if n_valid else None
        out[name] = {
            "n_windows": int(nav.numel()),
            "n_valid": n_valid,
            "coverage_nav_valid": round(float(val.float().mean()), 4),
            "echo_rate_all_windows": round(echo_all, 6),
            "echo_rate_valid_windows": (round(echo_valid, 6)
                                        if echo_valid is not None else None),
            "route_target_is_a_pure_function_of_nav_cmd": bool(echo_all == 1.0),
            "target_hist_on_valid": hist,
            "nav_cmd_hist_all": navhist,
            "majority_route_base_rate_on_valid": (round(maj, 4) if maj else None),
        }
    # ---- the audit's §1.2 claim, measured: how often is the model FED
    # "follow" while the road it will actually take turns? v2.1 (adaptive-arc,
    # 3x the coverage) is the reference judgement; the comparison is restricted
    # to the windows where v2.1 is willing to judge at all.
    ref_valid = torch.tensor(acc["v21_proposed"]["valid"])
    ref_tgt = torch.tensor(acc["v21_proposed"]["tgt"])
    ref_turn = ref_valid & (ref_tgt != rl.ROUTE_STRAIGHT)
    mis = {}
    for name, a in acc.items():
        nav = torch.tensor(a["nav"])
        fed_follow = nav == rl.NAV_FOLLOW
        wrong = fed_follow & ref_turn
        agree = (torch.tensor([rl._NAV_TO_ROUTE[int(c)] for c in nav])[ref_valid]
                 == ref_tgt[ref_valid]).float().mean()
        mis[name] = {
            "fed_follow_rate": round(float(fed_follow.float().mean()), 4),
            "fed_follow_while_route_turns_rate":
                round(float(wrong.float().mean()), 4),
            "n_fed_follow_while_route_turns": int(wrong.sum()),
            "share_of_true_turns_mislabelled_follow":
                round(float(wrong.sum() / max(int(ref_turn.sum()), 1)), 4),
            "fed_command_agrees_with_v21_route": round(float(agree), 4),
        }
    out["_fed_command_vs_true_route"] = {
        "reference": "route_target_v21 on its own valid subset",
        "n_reference_turn_windows": int(ref_turn.sum()),
        "by_labeler": mis,
        "_read": ("`fed_follow_while_route_turns_rate` is the audit §1.2 defect "
                  "quantified: the share of ALL windows on which the strategic "
                  "level is handed the command `follow` while the vehicle is "
                  "about to take a turn. It is not masked away -- `nav_valid` "
                  "gates the CE, never the INPUT."),
    }
    out["_meta"] = {
        "cache": str(cache),
        "n_episodes": len(files),
        "episode_len_min_max_mean": [min(lens), max(lens),
                                     round(sum(lens) / len(lens), 1)],
        "window": WINDOW, "max_horizon": MAX_HORIZON, "stride": stride,
        "seconds": round(elapsed, 1),
        "NAV_HORIZON_STEPS": rl.NAV_HORIZON_STEPS,
        "NAV_MIN_STEPS": rl.NAV_MIN_STEPS,
        "MIN_ARC_ROUTE_M": rl.MIN_ARC_ROUTE_M,
        "_NAV_TO_ROUTE": {NAV_NAME[k]: ROUTE_NAME[v]
                          for k, v in rl._NAV_TO_ROUTE.items()},
        "_ROUTE_TO_NAV": {ROUTE_NAME[k]: NAV_NAME[v]
                          for k, v in rl._ROUTE_TO_NAV.items()},
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    res = measure(Path(a.cache), a.episodes, a.stride)
    for k, v in res.items():
        if k.startswith("_"):
            continue
        print(f"{k:20s} coverage {v['coverage_nav_valid']:.4f}  "
              f"echo(all) {v['echo_rate_all_windows']:.6f}  "
              f"echo(valid) {v['echo_rate_valid_windows']}  "
              f"n_valid {v['n_valid']}/{v['n_windows']}  "
              f"targets {v['target_hist_on_valid']}")
    print("\nfed command vs the v2.1 route judgement:")
    for k, v in res["_fed_command_vs_true_route"]["by_labeler"].items():
        print(f"  {k:20s} fed-follow {v['fed_follow_rate']:.4f}  "
              f"fed-follow-while-turning {v['fed_follow_while_route_turns_rate']:.4f} "
              f"({v['n_fed_follow_while_route_turns']} windows, "
              f"{v['share_of_true_turns_mislabelled_follow']:.1%} of all turns)  "
              f"agree-with-v21 {v['fed_command_agrees_with_v21_route']:.4f}")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2))
        print(f"[pc1] wrote {a.out}")


if __name__ == "__main__":
    main()
