#!/usr/bin/env python3
"""P5 SCOPING (not the build): is the decode's 9.29 m displacement NECESSARY at that size?

⛔ This is a MEASUREMENT that scopes the successor. It does not build it and does not launch
it. Zero GPU, zero training: it reads the banked fan and the checkpoint's frozen anchor
vocabulary, and sweeps the ONE quantity that separates them.

THE QUESTION. `.../veto-only-fan-safety/` §6 measured that the frozen vocabulary is drivable
(peak_g 0.4808 g, envelope 1.6 %) and the decoded fan is not (4.1131 g, 88.8 %), with a mean
per-waypoint displacement of 9.29 m. It also showed the displacement is not gratuitous: the
bank is 77.8 % off-reach and only 10.8 % of the fan is, so adapting the fixed vocabulary to
the window's own speed is exactly what the offset buys.

⇒ The design question for a FEASIBILITY-AWARE DECODE is therefore not "is displacement bad"
but "is the decode displacing MORE than the ADE it earns requires?":

    path(lambda) = bank + lambda * (fan - bank),   lambda in [0, 1]

  * if ADE degrades only slowly as lambda falls while envelope/peak_g collapse, the decode is
    OVER-SHOOTING and a control-space reparameterisation (design A) recovers feasibility at
    little ADE cost -- the successor is worth building;
  * if ADE degrades in lockstep, the friction cost is INTRINSIC to matching the human on this
    vocabulary, and no reparameterisation of the same decode helps -- the successor must
    change the VOCABULARY instead.

⛔ TWO CONTROLS, and the second is the one RETRACTION #30 demands.
  C1 (arithmetic): lambda = 1 must reproduce the banked fan's metrics EXACTLY.
  C2 (object)    : lambda = 0 must reproduce `bank_vs_fan_feasibility.json`'s BANK rates,
                   which were measured by a DIFFERENT route (a separate script, a separate
                   window draw). C1 alone is blind by construction -- lambda = 1 is the
                   emitted fan whatever the left operand is, which is exactly how #30's
                   control passed while interpolating the wrong tensor.

The bank is read from the checkpoint's `core.decoder.anchors` directly. refcv3 carries no
`--anchor-v0-cond`, so `roll_bank` returns `anchors[None].expand(...)` and that tensor IS the
bank this decode started from (§6, from source).
ASCII-only output.
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch

_REPO = os.environ.get("TANITAD_REPO") or "C:/Users/Admin/refcv4b_repo"


def _load_driver():
    path = os.path.join(_REPO, "stack", "scripts", "rl_refcv3_min.py")
    spec = importlib.util.spec_from_file_location("_rl_refcv3_min_disp", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


D = _load_driver()
FS = D.FS
BANK_VS_FAN = "C:/Users/Admin/veto_run/raw/bank_vs_fan_feasibility.json"
LAMBDAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def find_anchors(sd):
    """The frozen [128, 8, 2] vocabulary, named so the artifact says WHICH object."""
    hits = {k: tuple(v.shape) for k, v in sd.items()
            if k.endswith("anchors") and hasattr(v, "shape") and v.dim() == 3
            and v.shape[-1] == 2}
    if not hits:
        raise SystemExit("no [N,S,2] '*anchors' tensor in the checkpoint")
    key = sorted(hits, key=lambda k: (0 if "decoder.anchors" in k else 1, k))[0]
    return key, sd[key].detach().float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    z = np.load(a.npz)
    bk = {k: torch.from_numpy(z[k]) for k in z.files}
    fan = bk["fan8"]                                            # [W, N, 8, 2]
    W, N = fan.shape[0], fan.shape[1]

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck.get("state_dict", ck))
    key, anchors = find_anchors(sd)
    if tuple(anchors.shape) != (N, fan.shape[2], 2):
        raise SystemExit("anchors %s do not match the fan %s"
                         % (tuple(anchors.shape), tuple(fan.shape[1:])))
    bank = anchors[None].expand(W, -1, -1, -1).contiguous()     # [W, N, 8, 2]

    lead5 = bk["lead5"].reshape(-1, 1, len(FS.GRID_S), 2)
    gt4 = bk["gt4"]
    NS = D.N_REWARD_SLOTS

    def score(p8):
        p5 = D.with_origin(p8[..., :NS, :])
        sc = FS.score_paths(p5, bk["v0"], lead5, lead_len_m=D.LEAD_LEN_DEFAULT_M)
        ade = (p8[..., :NS, :] - gt4[:, None]).norm(dim=-1).mean(dim=-1)   # [W, N]
        return sc, ade

    offs = fan - bank
    rows = []
    for lam in LAMBDAS:
        p8 = bank + lam * offs
        sc, ade = score(p8)
        rows.append({
            "lambda": lam,
            "mean_abs_disp_m": float((lam * offs)[..., :NS, :].abs().mean()),
            "fan_envelope": float(sc["envelope"].float().mean()),
            "fan_kamm_over": float(sc["kamm_over"].float().mean()),
            "fan_off_reach": float(sc["off_reach"].float().mean()),
            "fan_infeasible": float(sc["infeasible"].float().mean()),
            "fan_peak_g": float(sc["peak_g"].float().mean()),
            "oracle_ade_m": float(ade.min(dim=1).values.mean()),
            "sel_ade_m": float(ade[torch.arange(W), bk["sel_idx"].long()].mean()),
            "sel_peak_g": float(sc["peak_g"][torch.arange(W),
                                             bk["sel_idx"].long()].float().mean()),
            "sel_envelope": float(sc["envelope"][torch.arange(W),
                                                 bk["sel_idx"].long()].float().mean()),
        })

    # ---- CONTROLS ----------------------------------------------------------------
    sc1, ade1 = score(fan)
    r1 = rows[-1]
    c1 = {"name": "C1 arithmetic: lambda=1 reproduces the banked fan",
          "max_abs_diff_peak_g": abs(r1["fan_peak_g"] - float(sc1["peak_g"].float().mean())),
          "PASS": abs(r1["fan_peak_g"] - float(sc1["peak_g"].float().mean())) < 1e-9,
          "blind_by_construction": ("lambda=1 is the emitted fan whatever the LEFT operand "
                                    "is -- this control cannot see a wrong bank (RET #30)")}
    with open(BANK_VS_FAN, "r", encoding="utf-8") as f:
        bvf = json.load(f)
    r0 = rows[0]
    c2 = {"name": "C2 object: lambda=0 reproduces the BANK measured by a different route",
          "object_under_test": key,
          "reference_artifact": BANK_VS_FAN,
          "reference_n_windows": bvf["n_windows"],
          "this_n_windows": W,
          "peak_g": {"ours": r0["fan_peak_g"], "ref": bvf["bank"]["peak_g"]},
          "envelope": {"ours": r0["fan_envelope"], "ref": bvf["bank"]["envelope"]},
          "kamm_over": {"ours": r0["fan_kamm_over"], "ref": bvf["bank"]["kamm_over"]}}
    # the bank is a FIXED path set, so envelope/kamm/peak_g are window-INDEPENDENT and must
    # match to float precision even across different window draws; off_reach is NOT.
    c2["PASS"] = bool(abs(c2["peak_g"]["ours"] - c2["peak_g"]["ref"]) < 1e-4
                      and abs(c2["envelope"]["ours"] - c2["envelope"]["ref"]) < 1e-6
                      and abs(c2["kamm_over"]["ours"] - c2["kamm_over"]["ref"]) < 1e-6)
    c2["why_window_independent"] = (
        "the vocabulary is a FIXED [N,S,2] path set (anchor_v0_cond False), so envelope, "
        "kamm_over and peak_g do not depend on the window draw; off_reach and the "
        "lead-dependent flags DO, and are not asserted here")

    out = {"tool": "2026-09-05-kinematic-gate/raw/displacement_frontier.py",
           "tier": "T0", "purpose": "P5 SCOPING ONLY -- the successor is not built here",
           "n_windows": W, "n_candidates": N, "anchors_key": key,
           "ckpt": a.ckpt, "npz": a.npz,
           "controls": {"C1": c1, "C2": c2}, "sweep": rows}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("=== P5 SCOPING: the DISPLACEMENT frontier  path(l) = bank + l*(fan - bank) ===")
    print("  object under test: %s   n=%dw x %dcand" % (key, W, N))
    print("  C1 arithmetic PASS=%s   C2 OBJECT PASS=%s" % (c1["PASS"], c2["PASS"]))
    print("     C2: peak_g ours %.6f vs ref %.6f | envelope ours %.6f vs ref %.6f"
          % (c2["peak_g"]["ours"], c2["peak_g"]["ref"],
             c2["envelope"]["ours"], c2["envelope"]["ref"]))
    print("")
    print("  %6s %9s %10s %10s %10s %10s %10s %10s"
          % ("lambda", "disp_m", "fan_env", "fan_peak_g", "fan_offrch", "oracle_ade",
             "sel_ade", "sel_peak_g"))
    for r in rows:
        print("  %6.2f %9.3f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f"
              % (r["lambda"], r["mean_abs_disp_m"], r["fan_envelope"], r["fan_peak_g"],
                 r["fan_off_reach"], r["oracle_ade_m"], r["sel_ade_m"], r["sel_peak_g"]))
    print("")
    print("[disp] wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
