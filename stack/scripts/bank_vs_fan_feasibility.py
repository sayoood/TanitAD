#!/usr/bin/env python3
"""WHERE DOES refcv3's FAN INFEASIBILITY COME FROM: the VOCABULARY, or the OFFSET HEAD?

0 GPU. Scores the frozen ANCHOR BANK (`core.decoder.anchors`, a FIXED [128, 8, 2] path set
read straight out of the checkpoint) with the same `fan_safety.score_paths` used on the
EMITTED fan, on the same windows and the same v0 / lead track, and differences them.

WHY IT MATTERS FOR WHAT WE BUILD NEXT. `D-RL-FANSAFE-1` established that refcv3's selected
path is 12.7x more envelope-violating than the human. Two very different programmes follow
depending on the answer here:

  * if the BANK is infeasible  -> the lever is the vocabulary (H-DDA-3), and no post-training
    of the decoder can fix a fan whose candidates were never drivable;
  * if the OFFSET is           -> the lever is the decoder head that RL already trains, and
    the question becomes how much of the blow-up a constraint channel can undo.

The probe is deliberately blunt: one checkpoint tensor, one scorer, one differencing. It
reuses the banked fan (`fan_bank_base_240w.npz`) so no forward pass is needed at all.
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch

R = os.environ.get("TANITAD_REPO", r"C:\Users\Admin\refcv4b_repo")
for p in (os.path.join(R, "stack"), os.path.join(R, "taniteval"),
          os.path.join(R, "taniteval", "tools"), R):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)


def _fs():
    sp = importlib.util.spec_from_file_location(
        "fan_safety_for_bankprobe", os.path.join(R, "taniteval", "tools", "fan_safety.py"))
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


FS = _fs()
FLAGS = ("envelope", "kamm_over", "off_reach", "infeasible", "contact", "ttc_below")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--fan-npz", required=True, help="the banked emitted fan (240x128)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)["model"]
    if "core.decoder.anchors" not in sd:
        raise SystemExit("[bankprobe] checkpoint carries no core.decoder.anchors")
    bank = sd["core.decoder.anchors"].float()                        # [N, 8, 2] FIXED
    z = np.load(a.fan_npz)
    v0 = torch.tensor(z["v0"]).float()                               # [W]
    lead5 = torch.tensor(z["lead5"]).float()                         # [W, 5, 2]
    fan2 = torch.tensor(z["fan2"]).float()                           # [W, N, 5, 2]
    W, N = v0.shape[0], bank.shape[0]
    if fan2.shape[1] != N:
        raise SystemExit(f"[bankprobe] fan has {fan2.shape[1]} candidates, bank has {N}")

    # the 2 s prefix on the same grid, origin prepended -- the SAME construction the emitted
    # fan gets, so the two are scored as like for like.
    bank2 = torch.cat([torch.zeros(N, 1, 2), bank[:, :4, :]], dim=1)[None].expand(W, N, 5, 2)
    sc_b = FS.score_paths(bank2, v0, lead5.reshape(W, 1, 5, 2), lead_len_m=4.5)

    out = {"_tool": "stack/scripts/bank_vs_fan_feasibility.py",
           "_tier": "T0 instrument probe, 0 GPU", "_evidence_class": "MEASURED (ours)",
           "ckpt": a.ckpt, "fan_npz": a.fan_npz, "n_windows": int(W), "n_candidates": int(N),
           "note": ("the anchor bank is a FIXED path set (not v0-conditioned), so its "
                    "envelope/kamm rates are window-independent; off_reach and the "
                    "lead-dependent flags are not."),
           "bank": {}, "emitted": {}, "delta": {}}
    for f in FLAGS:
        b = float(sc_b[f].float().mean())
        e = float(z["f_" + f].mean())
        out["bank"][f], out["emitted"][f], out["delta"][f] = b, e, e - b
    out["bank"]["peak_g"] = float(sc_b["peak_g"].mean())
    out["emitted"]["peak_g"] = float(z["peak_g"].mean())
    out["delta"]["peak_g"] = out["emitted"]["peak_g"] - out["bank"]["peak_g"]
    out["mean_abs_offset_m_2s_prefix"] = float((fan2 - bank2).norm(dim=-1).mean())
    out["peak_g_ratio_emitted_over_bank"] = (out["emitted"]["peak_g"]
                                             / max(out["bank"]["peak_g"], 1e-9))

    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"{'flag':12s} {'ANCHOR BANK':>12s} {'EMITTED FAN':>12s}   delta")
    for f in list(FLAGS) + ["peak_g"]:
        print(f"{f:12s} {out['bank'][f]:12.4f} {out['emitted'][f]:12.4f}   {out['delta'][f]:+.4f}")
    print(f"mean |emitted - bank| per waypoint (2 s prefix): "
          f"{out['mean_abs_offset_m_2s_prefix']:.4f} m")
    print(f"peak_g ratio emitted/bank: {out['peak_g_ratio_emitted_over_bank']:.2f}x")
    print(f"[bankprobe] -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
