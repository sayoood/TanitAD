"""IDM v3 — assemble and score the SHIPPED artifact, then write the HF payload.

The v3 measurement says rotation and translation want DIFFERENT RECIPES (not
different loss weights — arms `Hrot`/`Htra` tested loss-weight decoupling and it
was *worse*):

    yaw_rate / steer  <- R0   : k=4 (9 frames), d_model 256, no clip-context
    speed / trajectory<- V2R  : k=8 (17 frames), d_model 128, winsorise + ctx

so the artifact is a TWO-EXPERT composite. This script scores the composite
against (a) every single arm and (b) the deployed A0, on the identical 4,195 val
windows with a paired episode-cluster bootstrap, and emits a weights-only
checkpoint.

🔒 The emitted checkpoint carries state_dicts + scalar config ONLY. No frames,
no poses, no clip identifiers, no per-clip geometry table.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/root/idm2")
sys.path.insert(0, "/root/taniteval")
sys.path.insert(0, "/root/v4eval/stack")
sys.path.insert(0, "/root/v4eval/stack/scripts")

import idm2_lib as L                 # noqa: E402
from taniteval import ci as tci      # noqa: E402

OUT = Path("/workspace/idm3/out")
CH = {"speed": 0, "yaw_rate": 1, "steer": 2, "long_accel": 3}
ROT_SRC, TRA_SRC = "R0", "V2R"       # which arm supplies which channel


def paired(pa, pb, g, eid):
    d = tci.paired_episode_cluster_bootstrap(np.abs(pa - g), np.abs(pb - g), eid,
                                             n_boot=2000, seed=0, reduce="mean")
    return {"delta_mae": float(d["delta"]), "lo": float(d["lo"]),
            "hi": float(d["hi"]),
            "separated": bool(d["lo"] > 0 or d["hi"] < 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT / "ship_v3.json"))
    ap.add_argument("--ckpt", default=str(OUT / "idm_head_v3.pt"))
    a = ap.parse_args()

    P = {}
    for f in ("arms_v3_preds.npy", "arms_v3b_preds.npy", "ship_tra_preds.npy",
              "ship_rot_preds.npy"):
        p = OUT / f
        if p.exists():
            P.update(np.load(p, allow_pickle=True).item())
    g = np.load(OUT / "val_gt_v3.npy", allow_pickle=True).item()
    G, Akin, eid, dom = g["S"], g["Akin"], g["eid"], g["dom"]
    a0 = np.load(OUT / "a0_preds.npy", allow_pickle=True).item()["S"]

    # ---- assemble the composite -------------------------------------------
    comp = P[TRA_SRC]["S"].copy()
    comp[:, CH["yaw_rate"]] = P[ROT_SRC]["S"][:, CH["yaw_rate"]]
    comp[:, CH["steer"]] = P[ROT_SRC]["S"][:, CH["steer"]]
    P["V3F"] = {"S": comp, "Traj": P[TRA_SRC]["Traj"]}

    res = {"composite": {"rotation_from": ROT_SRC, "translation_from": TRA_SRC},
           "n_windows": int(G.shape[0]), "n_episodes": int(len(set(eid)))}

    def block(p, gt, mask=None):
        m = np.ones(len(gt), bool) if mask is None else mask
        o = {"pooled": L.chan_metrics(p[m], gt[m])}
        for d in ("pai", "cm"):
            mm = m & (dom == d)
            o[d] = L.chan_metrics(p[mm], gt[mm])
        return o

    res["V3F"] = {nm: block(comp[:, j], G[:, j]) for nm, j in CH.items()}
    res["V3F"]["long_accel_vs_kinematic"] = block(comp[:, 3], Akin)
    res["A0_on_repaired_labels"] = {nm: block(a0[:, j], G[:, j])
                                    for nm, j in CH.items()}

    res["V3F_vs"] = {}
    for other in (TRA_SRC, ROT_SRC, "A0"):
        po = a0 if other == "A0" else P[other]["S"]
        res["V3F_vs"][other] = {nm: paired(comp[:, j], po[:, j], G[:, j], eid)
                                for nm, j in CH.items()}

    # ---- headline table ----------------------------------------------------
    print("\n" + "=" * 76)
    print("SHIPPED COMPOSITE V3F  (yaw/steer <- %s ; speed/traj <- %s)"
          % (ROT_SRC, TRA_SRC))
    print("  channel      pooled R2    PhysicalAI      comma2k19      MAE")
    for nm in CH:
        b = res["V3F"][nm]
        print("  %-11s %+.4f      %+.4f        %+.4f       %.4f"
              % (nm, b["pooled"]["r2"], b["pai"]["r2"], b["cm"]["r2"],
                 b["pooled"]["mae"]))
    print("\nvs the DEPLOYED head A0 (both scored on REPAIRED labels), paired dMAE")
    for nm in CH:
        d = res["V3F_vs"]["A0"][nm]
        print("  %-11s %+.5f [%+.5f,%+.5f] %s"
              % (nm, d["delta_mae"], d["lo"], d["hi"],
                 "SEPARATED" if d["separated"] else "not sep"))

    # ---- emit the weights-only checkpoint ----------------------------------
    rot_p = OUT / f"idm_head_v3_{ROT_SRC}.pt"
    tra_p = OUT / f"idm_head_v3_{TRA_SRC}.pt"
    if rot_p.exists() and tra_p.exists():
        r, t = torch.load(rot_p, weights_only=False), torch.load(tra_p, weights_only=False)
        ck = {
            "state_dict": {**{"rotation." + k: v for k, v in r["state_dict"].items()},
                           **{"translation." + k: v for k, v in t["state_dict"].items()}},
            "config": {
                "rotation": {k: v for k, v in r["cfg"].items()},
                "translation": {k: v for k, v in t["cfg"].items()},
                "state_dim": 2048, "dt": 0.1,
                "horizons": [5, 10, 15, 20],
                "rotation_channels": ["yaw_rate", "steer"],
                "translation_channels": ["speed"],
                "encoder": "tanitad-flagship-v1 (frozen), state_dim 2048",
                "seeds_shipped": 0,
            },
            "scalar_names": ["speed", "yaw_rate", "steer"],
            "metrics": {nm: {"pooled_r2": res["V3F"][nm]["pooled"]["r2"],
                             "pai_r2": res["V3F"][nm]["pai"]["r2"],
                             "cm_r2": res["V3F"][nm]["cm"]["r2"],
                             "pooled_mae": res["V3F"][nm]["pooled"]["mae"]}
                        for nm in ("speed", "yaw_rate", "steer")},
            "provenance": {
                "n_val_windows": int(G.shape[0]),
                "n_val_episodes": int(len(set(eid))),
                "estimator": "paired episode-cluster bootstrap, B=2000",
                "long_accel": "NOT SHIPPED - measured R2 is negative on both corpora",
            },
        }
        torch.save(ck, a.ckpt)
        print(f"\nWROTE {a.ckpt}  "
              f"({sum(v.numel() for v in ck['state_dict'].values()):,} params)")
    else:
        print(f"\n[ship] checkpoints not found ({rot_p}, {tra_p}) — metrics only")

    L.jdump(res, a.out)


if __name__ == "__main__":
    main()
