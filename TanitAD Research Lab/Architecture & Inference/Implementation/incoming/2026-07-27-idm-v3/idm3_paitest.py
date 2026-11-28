"""IDM v3 — the DECISIVE within-corpus geometry test, scored with paired CIs.

Trained AND evaluated on PhysicalAI ONLY: 26 train / 14 val episodes, 1,203 val
windows. Corpus identity is CONSTANT here, while camera height still varies
1.245–1.607 m (29 %) and the principal point still splits into two rig clusters.
**Any gain in this setting cannot be corpus memorisation, because there is only
one corpus.** That is the cleanest possible test of the PI's hypothesis, and it
is the one the pooled arms cannot give.

Safety: the PhysicalAI-only run overwrote `val_gt_v3.npy` with a pai-only ground
truth and a later full-corpus run overwrote it back, so this script does NOT
trust the file's provenance — it ASSERTS that the pai rows line up before using
them.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/root/idm2")
sys.path.insert(0, "/root/taniteval")
sys.path.insert(0, "/root/v4eval/stack")
sys.path.insert(0, "/root/v4eval/stack/scripts")

import idm2_lib as L                 # noqa: E402
import idm_head as ih                # noqa: E402
from idm3_arms import repair_labels  # noqa: E402
from taniteval import ci as tci      # noqa: E402

OUT = Path("/workspace/idm3/out")
CH = {"speed": 0, "yaw_rate": 1, "steer": 2, "long_accel": 3}
KBUILD = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT / "paitest_v3.json"))
    a = ap.parse_args()

    # Rebuild the pai-only val set from scratch — never trust an overwritten npy.
    _, va_tags = L.split_tags()
    va_tags = [t for t in va_tags if t.startswith("pai_")]
    va = L.build_set(va_tags, k=KBUILD, stride=2, want_seq=True)
    va["Akin"] = (va["Vseq"][:, KBUILD + 1] - va["Vseq"][:, KBUILD - 1]) / (2 * ih.DT)
    S, nch = repair_labels(va)
    assert nch == 0, f"PhysicalAI labels must be untouched by the repair, got {nch}"
    G, eid = S.numpy().astype(np.float64), va["eid"]
    print(f"pai-only val: {G.shape[0]} windows over {len(va_tags)} episodes; "
          f"repair changed {nch} labels (must be 0)")

    P = np.load(OUT / "arms_v3pai_preds.npy", allow_pickle=True).item()
    for arm in P:
        assert P[arm]["S"].shape[0] == G.shape[0], f"{arm} row mismatch"

    res = {"n_windows": int(G.shape[0]), "n_episodes": int(len(va_tags)),
           "repair_changed_pai_labels": int(nch), "arms": {}, "contrasts": {}}
    for arm, p in P.items():
        res["arms"][arm] = {nm: L.chan_metrics(p["S"][:, j], G[:, j])
                            for nm, j in CH.items()}

    def cmp(x, y):
        if x not in P or y not in P:
            return None
        o = {}
        for nm, j in CH.items():
            d = tci.paired_episode_cluster_bootstrap(
                np.abs(P[x]["S"][:, j] - G[:, j]),
                np.abs(P[y]["S"][:, j] - G[:, j]), eid, n_boot=2000, seed=0,
                reduce="mean")
            o[nm] = {"delta_mae": float(d["delta"]), "lo": float(d["lo"]),
                     "hi": float(d["hi"]),
                     "separated": bool(d["lo"] > 0 or d["hi"] < 0)}
        return o

    pairs = [("G1n", "R0", "GEOMETRY vs nothing, WITHIN PhysicalAI"),
             ("G1h", "R0", "camera-height ONLY vs nothing, WITHIN PhysicalAI"),
             ("Cshufn", "R0", "SHUFFLED geometry vs nothing"),
             ("G1n", "Cshufn", "real vs shuffled geometry"),
             ("G1h", "Cshufn", "camera height vs shuffled"),
             ("Sctxn", "R0", "clip-context vs nothing"),
             ("Ccorpn", "R0", "constant-token control vs nothing")]
    print("\nWITHIN-PhysicalAI paired dMAE on SPEED (negative = first is better)")
    for x, y, why in pairs:
        c = cmp(x, y)
        if not c:
            continue
        res["contrasts"][f"{x}_vs_{y}"] = {"why": why, **c}
        s = c["speed"]
        print("  %-14s %+.4f [%+.4f,%+.4f] %s   (%s)"
              % (f"{x} vs {y}", s["delta_mae"], s["lo"], s["hi"],
                 "SEPARATED" if s["separated"] else "not sep ", why))

    print("\nspeed MAE by arm (mean over 3 seeds, pai only)")
    for arm in sorted(res["arms"]):
        print("  %-8s MAE %.3f   R2 %+.4f   yaw R2 %+.4f"
              % (arm, res["arms"][arm]["speed"]["mae"],
                 res["arms"][arm]["speed"]["r2"],
                 res["arms"][arm]["yaw_rate"]["r2"]))
    L.jdump(res, a.out)


if __name__ == "__main__":
    main()
