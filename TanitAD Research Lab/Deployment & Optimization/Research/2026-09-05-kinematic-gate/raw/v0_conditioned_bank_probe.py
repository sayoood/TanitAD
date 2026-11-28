#!/usr/bin/env python3
"""RULE ZERO's next arm: does CONDITIONING THE VOCABULARY ON SPEED create the knee?

`displacement_frontier.py` measured that shrinking refcv3's decode toward its own frozen
vocabulary has NO KNEE -- `sel_ade` and `fan_peak_g` trade monotonically, and `fan_off_reach`
rises 0.109 -> 0.782 as `fan_envelope` falls 0.888 -> 0.016. The diagnosis was that the bank
is SPEED-BLIND: it is rolled at a single `anchor_ref_speed` (refc.py:361, 10.0 m/s) for every
window, so shrinking toward it is shrinking toward paths the ego cannot reach.

⛔ A refutation is a waypoint. THIS is the cheapest experiment that could still make the
feasibility-aware decode work, and it costs ZERO GPU:

    bank_v0[w] = bank * (v0[w] / v_ref)          # the crudest v0 conditioning there is

then sweep `path(l) = bank_v0 + l * (fan - bank_v0)` exactly as before, and ask whether the
off-reach wall comes down and a knee appears.

⚠️ WHAT THIS IS AND IS NOT. `roll_bank` with `anchor_v0_cond` rolls the anchor CONTROL sequence
at the window's own speed (refc.py:1336). This script applies a SIMILARITY SCALING to the
waypoint bank instead, because refcv3's anchors are stored as waypoints and the control sequence
for them is not in the checkpoint. Under a spatial scale s on a FIXED time grid, speed and
acceleration scale by s and curvature by 1/s, so `lat_acc = v^2 * kappa` scales by s -- i.e. the
scaling is kinematically consistent, but it is an APPROXIMATION of the real roll and is labelled
as one. It is decisive for the question asked (does the off-reach wall come down?) and is NOT a
substitute for the real v0-conditioned bank.

⛔ CONTROLS.
  C1 (arithmetic): l = 1 must reproduce the banked fan exactly, for BOTH banks.
  C2 (object)    : the UNCONDITIONED l = 0 must reproduce `bank_vs_fan_feasibility.json`'s bank
                   rates, measured by a different script on a different draw. This is the same
                   object assertion RETRACTION #30 demands, and it is what proves the left
                   operand is the vocabulary and not something that merely behaves like it.
  C3 (identity)  : at v0 == v_ref the scaled bank must equal the unscaled bank exactly.
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
    p = os.path.join(_REPO, "stack", "scripts", "rl_refcv3_min.py")
    spec = importlib.util.spec_from_file_location("_rl_refcv3_min_v0b", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


D = _load_driver()
FS = D.FS
BANK_VS_FAN = "C:/Users/Admin/veto_run/raw/bank_vs_fan_feasibility.json"
LAMBDAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def find_anchors(sd):
    hits = [k for k, v in sd.items()
            if k.endswith("anchors") and hasattr(v, "shape") and v.dim() == 3
            and v.shape[-1] == 2]
    if not hits:
        raise SystemExit("no [N,S,2] '*anchors' tensor in the checkpoint")
    key = sorted(hits, key=lambda k: (0 if "decoder.anchors" in k else 1, k))[0]
    return key, sd[key].detach().float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--v-ref", type=float, default=10.0,
                    help="refc.py:361 ref_speed_ms -- the single speed the bank is rolled at")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    z = np.load(a.npz)
    bk = {k: torch.from_numpy(z[k]) for k in z.files}
    fan = bk["fan8"]
    W, N, S = fan.shape[0], fan.shape[1], fan.shape[2]
    NS = D.N_REWARD_SLOTS
    v0 = bk["v0"].float()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck.get("state_dict", ck))
    key, anchors = find_anchors(sd)
    bank = anchors[None].expand(W, -1, -1, -1).contiguous()

    s = (v0 / float(a.v_ref)).clamp_min(1e-6).reshape(W, 1, 1, 1)
    bank_v0 = bank * s

    lead5 = bk["lead5"].reshape(-1, 1, len(FS.GRID_S), 2)
    gt4 = bk["gt4"]
    ar = torch.arange(W)
    si = bk["sel_idx"].long()

    def sweep(base):
        offs = fan - base
        rows = []
        for lam in LAMBDAS:
            p8 = base + lam * offs
            p5 = D.with_origin(p8[..., :NS, :])
            sc = FS.score_paths(p5, bk["v0"], lead5, lead_len_m=D.LEAD_LEN_DEFAULT_M)
            ade = (p8[..., :NS, :] - gt4[:, None]).norm(dim=-1).mean(dim=-1)
            rows.append({
                "lambda": lam,
                "mean_abs_disp_m": float((lam * offs)[..., :NS, :].abs().mean()),
                "fan_envelope": float(sc["envelope"].float().mean()),
                "fan_kamm_over": float(sc["kamm_over"].float().mean()),
                "fan_off_reach": float(sc["off_reach"].float().mean()),
                "fan_infeasible": float(sc["infeasible"].float().mean()),
                "fan_peak_g": float(sc["peak_g"].float().mean()),
                "oracle_ade_m": float(ade.min(dim=1).values.mean()),
                "sel_ade_m": float(ade[ar, si].mean()),
                "sel_peak_g": float(sc["peak_g"][ar, si].float().mean()),
                "sel_envelope": float(sc["envelope"][ar, si].float().mean()),
            })
        return rows

    fixed = sweep(bank)
    cond = sweep(bank_v0)

    with open(BANK_VS_FAN, "r", encoding="utf-8") as f:
        bvf = json.load(f)
    near = (v0 - float(a.v_ref)).abs() < 1e-6
    c3 = {"name": "C3 identity: at v0 == v_ref the scaled bank IS the unscaled bank",
          "n_windows_at_v_ref": int(near.sum()),
          "max_abs_diff_on_those": (float((bank_v0[near] - bank[near]).abs().max())
                                    if bool(near.any()) else None),
          "note": ("no window sits exactly at v_ref on this corpus, so the identity is "
                   "checked analytically instead: s = v0/v_ref, and s == 1 gives bank_v0 == "
                   "bank by construction of the scalar multiply")}
    controls = {
        "C1_lambda1_fixed": {"max_abs_diff_peak_g":
                             abs(fixed[-1]["fan_peak_g"] - cond[-1]["fan_peak_g"]),
                             "PASS": abs(fixed[-1]["fan_peak_g"]
                                         - cond[-1]["fan_peak_g"]) < 1e-9,
                             "why": "lambda=1 is the emitted fan for BOTH banks"},
        "C2_object_unconditioned_bank": {
            "peak_g": {"ours": fixed[0]["fan_peak_g"], "ref": bvf["bank"]["peak_g"]},
            "envelope": {"ours": fixed[0]["fan_envelope"], "ref": bvf["bank"]["envelope"]},
            "PASS": bool(abs(fixed[0]["fan_peak_g"] - bvf["bank"]["peak_g"]) < 1e-4
                         and abs(fixed[0]["fan_envelope"] - bvf["bank"]["envelope"]) < 1e-6),
            "reference_artifact": BANK_VS_FAN},
        "C3_scale_identity": c3,
    }

    out = {"tool": "2026-09-05-kinematic-gate/raw/v0_conditioned_bank_probe.py",
           "tier": "T0", "purpose": "RULE ZERO next arm for D-DECODE-DISP-FRONTIER-1",
           "approximation": ("similarity scaling of the WAYPOINT bank by v0/v_ref; NOT "
                             "roll_bank's control-space roll (refc.py:1336)"),
           "v_ref_ms": float(a.v_ref), "anchors_key": key,
           "n_windows": W, "n_candidates": N,
           "v0_mean_ms": float(v0.mean()), "v0_median_ms": float(v0.median()),
           "scale_mean": float(s.mean()), "scale_min": float(s.min()),
           "scale_max": float(s.max()),
           "controls": controls,
           "sweep_fixed_bank": fixed, "sweep_v0_conditioned_bank": cond}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("=== RULE ZERO NEXT ARM: does a v0-CONDITIONED vocabulary create the knee? ===")
    print("  object: %s   v_ref %.2f m/s   v0 mean %.2f median %.2f   scale [%.3f, %.3f]"
          % (key, a.v_ref, out["v0_mean_ms"], out["v0_median_ms"],
             out["scale_min"], out["scale_max"]))
    print("  C1 PASS=%s   C2 OBJECT PASS=%s (peak_g %.6f vs ref %.6f)   C3 n_at_vref=%s"
          % (controls["C1_lambda1_fixed"]["PASS"],
             controls["C2_object_unconditioned_bank"]["PASS"],
             controls["C2_object_unconditioned_bank"]["peak_g"]["ours"],
             controls["C2_object_unconditioned_bank"]["peak_g"]["ref"],
             c3["n_windows_at_v_ref"]))
    for name, rows in (("FIXED bank (speed-blind, as shipped)", fixed),
                       ("v0-CONDITIONED bank (scaled to the window's own speed)", cond)):
        print("")
        print("  -- %s --" % name)
        print("  %6s %9s %10s %11s %11s %11s %10s"
              % ("lambda", "disp_m", "fan_env", "fan_peak_g", "fan_offrch", "oracle_ade",
                 "sel_ade"))
        for r in rows:
            print("  %6.2f %9.3f %10.4f %11.4f %11.4f %11.4f %10.4f"
                  % (r["lambda"], r["mean_abs_disp_m"], r["fan_envelope"], r["fan_peak_g"],
                     r["fan_off_reach"], r["oracle_ade_m"], r["sel_ade_m"]))
    f0, c0 = fixed[0], cond[0]
    print("")
    print("  AT lambda = 0 (the vocabulary itself):")
    print("    fixed        : off_reach %.4f  envelope %.4f  peak_g %.4f  oracle_ade %.4f"
          % (f0["fan_off_reach"], f0["fan_envelope"], f0["fan_peak_g"], f0["oracle_ade_m"]))
    print("    v0-conditioned: off_reach %.4f  envelope %.4f  peak_g %.4f  oracle_ade %.4f"
          % (c0["fan_off_reach"], c0["fan_envelope"], c0["fan_peak_g"], c0["oracle_ade_m"]))
    print("    => off_reach %+.4f   oracle_ade %+.4f   (negative = the conditioned bank is better)"
          % (c0["fan_off_reach"] - f0["fan_off_reach"],
             c0["oracle_ade_m"] - f0["oracle_ade_m"]))
    print("")
    print("[v0bank] wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
