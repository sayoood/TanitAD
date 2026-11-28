"""P0 - prove the LOCAL banked dump IS the arm refcv4b_t1.json reports.

The discriminator is a POSITIVE content assertion, not a filename: recompute the
kin3 tactical confusion from the dump's own `g` and `os` paths with the SAME
labeller the published block names, and require an EXACT integer match against
the published counts.  Integer confusion cells cannot agree by luck.
"""
import json
import sys

import numpy as np
import torch

import _env  # noqa: F401
import load

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

T1 = (r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"
      r"\TanitAD Research Lab\Architecture & Inference\Research"
      r"\2026-09-06-refcv4b-landing\raw\refcv4b_t1.json")


def main():
    from taniteval import four_families as ff

    D = load.load_all()
    n = D["g"].shape[0]
    print(f"dump: n_windows={n}  n_episodes={len(D['_n_per_ep'])}")
    man = load.manifest()
    print("dump manifest ckpt :", man["model"]["ckpt"])
    print("dump manifest step :", man["model"]["step"])

    with open(T1, encoding="utf-8") as fh:
        pub = json.load(fh)
    print("published ckpt     :", pub["ckpt"])
    print("published n_windows:", pub["n_windows"],
          " n_episodes:", pub["n_episodes"])

    g = torch.as_tensor(D["g"])
    os_ = torch.as_tensor(D["os"])
    blk = ff.tactical_from_trajectory(os_, g, dt=float(pub["dt_s"]),
                                      eid=D["epi"].tolist(), n_boot=200,
                                      seed=0, tier="T1")
    pblk = pub["arms"]["os"]["four_families"]["tactical"]

    ok = True
    for axis in ("lateral_decision", "longitudinal_decision"):
        mine = blk[axis]["confusion_gt_rows_pred_cols"]
        theirs = pblk[axis]["confusion_gt_rows_pred_cols"]
        same = (np.asarray(mine) == np.asarray(theirs)).all()
        ok &= bool(same)
        print(f"\n{axis}: EXACT confusion match = {bool(same)}")
        print("  recomputed:", mine)
        print("  published :", theirs)
        print(f"  acc recomputed {blk[axis]['accuracy']:.4f} "
              f"published {pblk[axis]['accuracy']:.4f}")

    print("\nIDENTITY:", "CONFIRMED - same windows, same arm" if ok
          else "MISMATCH - the local dump is NOT the published arm")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
