"""E-ARCH-NAVSRC-1 — refb-derived nav vs the v7.2 nav_command token, per window, all 4,719 B1 clips.

Run on the dev box: labels + the labeler's egomotion tar are local under
C:/Users/Admin/tanitad-wt/_s2build/release. Reproduces raw/nav_agreement_refb_vs_v72.json
(MEASURED 2026-09-02: 65.5 % fed-window agreement; refb feeds `follow` on 94.6 % of windows).
See RESULT.md for the two rules, the method and the caveats.
"""
import gzip
import io
import json
import math
import tarfile

import numpy as np
import pandas as pd

ROOT = "C:/Users/Admin/tanitad-wt/_s2build/release"
T = 199          # MEASURED B1 episode length (D-EPISODE-LENGTH)
HZ = 10.0
W = 8            # the live refcv3 batch shape (20 x 8 frames)
H_STEPS = 250    # refb_labels.py:72
MIN_STEPS = 150  # refb_labels.py:73
TURN = math.pi / 4  # refb_labels.py:71
TOK = {"NAV_FOLLOW_ROAD": "follow", "NAV_TURN_L": "left", "NAV_TURN_R": "right"}


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def yaw_from_q(qx, qy, qz, qw):
    return np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def refb_cmd(yaw, t):
    """refb_labels.nav_command, same semantics: (cmd, valid)."""
    h = min(H_STEPS, T - 1 - t)
    if h < MIN_STEPS:
        return "follow", False
    d = wrap(yaw[t + h] - yaw[t])
    if d > TURN:
        return "left", True
    if d < -TURN:
        return "right", True
    return "follow", True


def main():
    labels = {}
    for split in ("train", "eval"):
        with gzip.open(f"{ROOT}/v72/s2_labels_v7.2_{split}.jsonl.gz", "rt", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    labels[r["clip_id"]] = (split, r)
    tar = tarfile.open(f"{ROOT}/tanitad-v7-training-corpus/egomotion/egomotion_alpamayo.tar")
    members = {m.name.split(".")[0]: m for m in tar.getmembers() if m.isfile()}
    S = dict(n=0, bad=0, tok={}, anchor_agree=0, win_total=0, win_valid=0, win_valid_agree=0,
             fed_agree=0, conf_valid={}, fed_refb={}, fed_v7={}, clip_any_turn_refb=0,
             clip_turn_v7=0, both_turn=0, clip_refb_turn_v7_follow=0, per_split={})
    grid = np.arange(T) / HZ
    for cid, (split, r) in labels.items():
        m = members.get(cid)
        if m is None:
            S["bad"] += 1
            continue
        df = pd.read_parquet(io.BytesIO(tar.extractfile(m).read()),
                             columns=["timestamp", "qx", "qy", "qz", "qw"])
        ts = df["timestamp"].to_numpy() / 1e6
        yr = np.unwrap(yaw_from_q(df["qx"].to_numpy(), df["qy"].to_numpy(),
                                  df["qz"].to_numpy(), df["qw"].to_numpy()))
        if ts[0] > 0.05 or ts[-1] < grid[-1] - 0.05:
            S["bad"] += 1
            continue
        yaw = np.interp(grid, ts, yr)
        tok = TOK[r["nav_command"]["token"]]
        S["n"] += 1
        S["tok"][tok] = S["tok"].get(tok, 0) + 1
        ps = S["per_split"].setdefault(split, dict(n=0, fed_agree=0, win=0))
        ps["n"] += 1
        S["anchor_agree"] += refb_cmd(yaw, 80)[0] == tok      # s2 anchor t0 = 8.0 s
        any_turn = False
        for t in range(W - 1, T):
            c, v = refb_cmd(yaw, t)
            S["win_total"] += 1
            ps["win"] += 1
            S["fed_refb"][c] = S["fed_refb"].get(c, 0) + 1
            S["fed_v7"][tok] = S["fed_v7"].get(tok, 0) + 1
            S["fed_agree"] += c == tok
            ps["fed_agree"] += c == tok
            if v:
                S["win_valid"] += 1
                S["win_valid_agree"] += c == tok
                k = f"refb={c} | v7={tok}"
                S["conf_valid"][k] = S["conf_valid"].get(k, 0) + 1
                if c != "follow":
                    any_turn = True
        S["clip_any_turn_refb"] += any_turn
        if any_turn and tok == "follow":
            S["clip_refb_turn_v7_follow"] += 1
        if tok != "follow":
            S["clip_turn_v7"] += 1
            S["both_turn"] += any_turn
    with open("raw/nav_agreement_refb_vs_v72.json", "w") as fh:
        json.dump(S, fh, indent=1)
    print(f"fed-window agreement {100 * S['fed_agree'] / S['win_total']:.1f}% "
          f"over {S['win_total']} windows, {S['n']} clips")


if __name__ == "__main__":
    main()
