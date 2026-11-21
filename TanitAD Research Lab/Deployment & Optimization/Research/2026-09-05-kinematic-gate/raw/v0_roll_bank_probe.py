#!/usr/bin/env python3
"""RULE ZERO, second attempt: the CONTROL-SPACE roll, not a similarity scale.

`v0_conditioned_bank_probe.py` scaled the waypoint bank by v0/v_ref and made things WORSE
(envelope 0.0156 -> 0.2421, oracle_ade 1.0985 -> 1.1571) while barely moving off_reach
(0.7822 -> 0.6770). The arithmetic says why, and it is not a bug: under a spatial scale `s` on
a FIXED time grid, speed scales by `s` and curvature by `1/s`, so
`lat_acc = v^2 * kappa` scales by **s** -- and this corpus needs scales up to 3.63x. A
similarity scale multiplies the friction load; it is the wrong operation.

⛔ `roll_bank`'s conditioning (refc.py:1336) is different in exactly the way that matters: it
re-integrates the anchor's own CONTROL sequence at the window's speed, which changes ARC LENGTH
without multiplying the friction load. This script does that on refcv3's waypoint anchors:

  1. read each anchor's HEADING profile as a function of ARC LENGTH (its control content);
  2. re-integrate that profile at the window's own constant speed v0 over the SAME time grid;
  3. sweep `path(l) = bank_rolled + l * (fan - bank_rolled)` as before.

⚠️ SCOPE. Constant-speed re-integration keeps the anchor's SHAPE (heading vs arc) and replaces
its SPEED PROFILE. It is a faithful stand-in for the control-space roll on a bank that stores
waypoints, and it is NOT the real `roll_controls` (which also carries a longitudinal control).
Labelled as a stand-in, not as the real thing.

⛔ CONTROLS.
  C1 (arithmetic) l = 1 reproduces the emitted fan for BOTH banks.
  C2 (object)     the UNROLLED l = 0 reproduces `bank_vs_fan_feasibility.json`'s bank rates,
                  from a different script on a different draw (RETRACTION #30).
  C3 (identity)   re-integrating at each anchor's OWN mean speed must reproduce that anchor
                  to within float error -- the roll is a no-op when the speed is unchanged.
                  ⛔ This is the control that proves the re-integration is faithful; without it
                  a broken integrator would look like a finding.
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
    spec = importlib.util.spec_from_file_location("_rl_refcv3_min_roll", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


D = _load_driver()
FS = D.FS
BANK_VS_FAN = "C:/Users/Admin/veto_run/raw/bank_vs_fan_feasibility.json"
LAMBDAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
DT = 0.5                       # the reward/eval grid spacing, seconds


def find_anchors(sd):
    hits = [k for k, v in sd.items()
            if k.endswith("anchors") and hasattr(v, "shape") and v.dim() == 3
            and v.shape[-1] == 2]
    if not hits:
        raise SystemExit("no [N,S,2] '*anchors' tensor in the checkpoint")
    return sorted(hits, key=lambda k: (0 if "decoder.anchors" in k else 1, k))[0]


def roll_at_speed(anchors, speed):
    """[N,S,2] anchors -> [W,N,S,2] re-integrated at per-window constant `speed` [W].

    Keeps each anchor's HEADING-vs-ARC profile and replaces its speed profile. Step k of the
    rolled path advances `speed*DT` metres along the anchor's own heading at the ARC POSITION
    the rolled path has reached, so the shape is preserved and the length is re-set.
    """
    N, S, _ = anchors.shape
    d = anchors[:, 1:, :] - anchors[:, :-1, :]                 # [N, S-1, 2]
    seg = d.norm(dim=-1)                                       # [N, S-1]
    head = torch.atan2(d[..., 1], d[..., 0])                   # [N, S-1]
    cum = torch.cat([torch.zeros(N, 1), seg.cumsum(dim=-1)], dim=-1)   # [N, S] arc at node
    W = speed.shape[0]
    out = torch.zeros(W, N, S, 2)
    pos = torch.zeros(W, N, 2)
    arc = torch.zeros(W, N)
    step = (speed.reshape(W, 1) * DT).expand(W, N)             # [W, N] metres per tick
    for k in range(1, S):
        # heading of the anchor segment containing the current arc position
        idx = torch.searchsorted(cum[:, 1:].contiguous(),
                                 arc.transpose(0, 1).contiguous())     # [N, W]
        idx = idx.clamp(max=S - 2).transpose(0, 1)                     # [W, N]
        h = head.expand(W, N, S - 1).gather(2, idx.unsqueeze(-1)).squeeze(-1)   # [W, N]
        pos = pos + torch.stack([step * torch.cos(h), step * torch.sin(h)], dim=-1)
        arc = arc + step
        out[:, :, k, :] = pos
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    z = np.load(a.npz)
    bk = {k: torch.from_numpy(z[k]) for k in z.files}
    fan = bk["fan8"]
    W, N, S = fan.shape[0], fan.shape[1], fan.shape[2]
    NS = D.N_REWARD_SLOTS
    v0 = bk["v0"].float()
    N = fan.shape[1]

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck.get("state_dict", ck))
    key = find_anchors(sd)
    anchors8 = sd[key].detach().float()                         # [N, 8, 2]
    # ⛔ THE GRID IS UNIFORM ONLY OVER THE FIRST FOUR SLOTS (rl_refcv3_min.py:610 asserts
    # ARM_HORIZONS[:4] == [5, 10, 15, 20] frames at 10 Hz = 0.5 s spacing); the tail is the
    # 6 s horizon set on a different spacing. And slot 0 is at t = 0.5 s, not the origin.
    # Both are handled by working on the 5-point 2 s prefix, which is the object every
    # fan-safety metric is computed on anyway.
    anchors = D.with_origin(anchors8[:, :NS, :])                # [N, 5, 2], origin first
    bank8 = anchors8[None].expand(W, -1, -1, -1).contiguous()

    # ---- C3: what a CONSTANT-speed roll can actually be an identity on ---------------
    seg = (anchors[:, 1:, :] - anchors[:, :-1, :]).norm(dim=-1)  # [N, 4]
    own_speed = seg.mean(dim=-1) / DT                            # [N] per-anchor mean speed
    cv = (seg.std(dim=-1) / seg.mean(dim=-1).clamp_min(1e-6))    # [N] speed variation
    self_roll = roll_at_speed(anchors, own_speed)                # [N, N, 5, 2]
    self_diag = self_roll[torch.arange(N), torch.arange(N)]      # [N, 5, 2]
    node_err = (self_diag - anchors).norm(dim=-1).mean(dim=-1)   # [N]
    c3_err = float(node_err.mean())
    c3_max = float((self_diag - anchors).norm(dim=-1).max())
    const_set = cv < 0.05
    c3a_err = float(node_err[const_set].mean()) if bool(const_set.any()) else float("nan")
    c3a_max = float(node_err[const_set].max()) if bool(const_set.any()) else float("nan")
    # C3b: a SYNTHETIC constant-speed arc with a known answer
    _t = torch.arange(5, dtype=torch.float32) * DT
    _v, _k = 12.0, 0.03
    _th = _k * _v * _t
    _syn = torch.stack([(torch.sin(_th) / _k), (1 - torch.cos(_th)) / _k], dim=-1)[None]
    _sr = roll_at_speed(_syn, torch.tensor([_v]))[0, 0]
    c3b = float((_sr - _syn[0]).norm(dim=-1).max())

    rolled5 = roll_at_speed(anchors, v0)                         # [W, N, 5, 2]
    # back to the 8-slot layout the sweep indexes: the 2 s prefix is rolled, the tail is
    # carried over unchanged and is NEVER scored (every metric below reads [..., :NS, :]).
    rolled = bank8.clone()
    rolled[:, :, :NS, :] = rolled5[:, :, 1:, :]
    bank = bank8

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
            rows.append({"lambda": lam,
                         "mean_abs_disp_m": float((lam * offs)[..., :NS, :].abs().mean()),
                         "fan_envelope": float(sc["envelope"].float().mean()),
                         "fan_off_reach": float(sc["off_reach"].float().mean()),
                         "fan_peak_g": float(sc["peak_g"].float().mean()),
                         "fan_infeasible": float(sc["infeasible"].float().mean()),
                         "oracle_ade_m": float(ade.min(dim=1).values.mean()),
                         "sel_ade_m": float(ade[ar, si].mean())})
        return rows

    fixed, roll = sweep(bank), sweep(rolled)

    with open(BANK_VS_FAN, "r", encoding="utf-8") as f:
        bvf = json.load(f)
    controls = {
        "C1_lambda1": {"fixed": fixed[-1]["fan_peak_g"], "rolled": roll[-1]["fan_peak_g"],
                       "PASS": abs(fixed[-1]["fan_peak_g"]
                                   - roll[-1]["fan_peak_g"]) < 1e-9},
        "C2_object": {"peak_g_ours": fixed[0]["fan_peak_g"],
                      "peak_g_ref": bvf["bank"]["peak_g"],
                      "envelope_ours": fixed[0]["fan_envelope"],
                      "envelope_ref": bvf["bank"]["envelope"],
                      "reference_artifact": BANK_VS_FAN,
                      "PASS": bool(abs(fixed[0]["fan_peak_g"]
                                       - bvf["bank"]["peak_g"]) < 1e-4)},
        "C3_roll_is_noop_at_own_speed": {
            "all_anchors_mean_node_error_m": c3_err,
            "all_anchors_max_node_error_m": c3_max,
            "C3a_near_constant_speed_subset": {
                "n_anchors": int(const_set.sum()), "of": int(cv.shape[0]),
                "criterion": "per-anchor step-length coefficient of variation < 0.05",
                "mean_node_error_m": c3a_err, "max_node_error_m": c3a_max,
                "PASS": bool(c3a_err == c3a_err and c3a_err < 0.05)},
            "C3b_synthetic_constant_speed_arc": {
                "max_node_error_m": c3b, "PASS": bool(c3b < 1e-4),
                "why": ("a known-answer case: C3a on a subset could pass by luck, this "
                        "cannot")},
            "C3c_diagnostic": {
                "speed_cv_mean": float(cv.mean()), "speed_cv_max": float(cv.max()),
                "note": ("a CONSTANT-speed roll cannot reproduce a VARIABLE-speed anchor; "
                         "the all-anchor error is a function of speed variation, not of the "
                         "integrator, which is why C3a/C3b are the admissible controls")},
            "PASS": bool(c3b < 1e-4 and c3a_err == c3a_err and c3a_err < 0.05),
            "gates_the_rolled_sweep": True,
            "why": ("re-integrating a CONSTANT-speed path at its own speed must return that "
                    "path; without it a broken integrator would look like a finding")},
    }

    out = {"tool": "2026-09-05-kinematic-gate/raw/v0_roll_bank_probe.py", "tier": "T0",
           "purpose": "RULE ZERO next arm #2 for D-DECODE-DISP-FRONTIER-1",
           "operation": ("constant-speed re-integration of each anchor's heading-vs-arc "
                         "profile at the window's own v0 -- a stand-in for roll_bank's "
                         "control-space roll on a bank that stores WAYPOINTS"),
           "anchors_key": key, "n_windows": W, "n_candidates": N, "dt_s": DT,
           "v0_mean_ms": float(v0.mean()),
           "anchor_own_speed_mean_ms": float(own_speed.mean()),
           "anchor_own_speed_min_ms": float(own_speed.min()),
           "anchor_own_speed_max_ms": float(own_speed.max()),
           "controls": controls,
           "rolled_sweep_admissible": bool(controls["C3_roll_is_noop_at_own_speed"]["PASS"]),
           "sweep_fixed_bank": fixed, "sweep_v0_rolled_bank": roll}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("=== RULE ZERO NEXT ARM #2: the CONTROL-SPACE roll (not a similarity scale) ===")
    print("  object %s  n=%dw x %dcand  v0 mean %.2f m/s ; anchor own speed %.2f "
          "[%.2f, %.2f] m/s"
          % (key, W, N, out["v0_mean_ms"], out["anchor_own_speed_mean_ms"],
             out["anchor_own_speed_min_ms"], out["anchor_own_speed_max_ms"]))
    _c3 = controls["C3_roll_is_noop_at_own_speed"]
    print("  C1 PASS=%s   C2 OBJECT PASS=%s (%.6f vs %.6f)"
          % (controls["C1_lambda1"]["PASS"], controls["C2_object"]["PASS"],
             controls["C2_object"]["peak_g_ours"], controls["C2_object"]["peak_g_ref"]))
    print("  C3 PASS=%s | C3a const-speed subset %d/%d anchors, mean err %.5f m (PASS=%s)"
          " | C3b synthetic max err %.2e (PASS=%s)"
          % (_c3["PASS"], _c3["C3a_near_constant_speed_subset"]["n_anchors"],
             _c3["C3a_near_constant_speed_subset"]["of"],
             _c3["C3a_near_constant_speed_subset"]["mean_node_error_m"],
             _c3["C3a_near_constant_speed_subset"]["PASS"],
             _c3["C3b_synthetic_constant_speed_arc"]["max_node_error_m"],
             _c3["C3b_synthetic_constant_speed_arc"]["PASS"]))
    print("  C3c diagnostic: all-anchor mean err %.4f m at speed CV mean %.3f (max %.3f)"
          " -- a constant-speed roll cannot be an identity on a variable-speed anchor"
          % (c3_err, _c3["C3c_diagnostic"]["speed_cv_mean"],
             _c3["C3c_diagnostic"]["speed_cv_max"]))
    _c3ok = controls["C3_roll_is_noop_at_own_speed"]["PASS"]
    if not _c3ok:
        print("")
        print("  !! C3 FAILED -> THE v0-ROLLED SWEEP IS INADMISSIBLE AND IS NOT QUOTED.")
        print("     A broken integrator would read as a spectacular finding here; that is")
        print("     what this control exists to stop. Printed below for the record only.")
    for name, rows in (("FIXED bank (speed-blind, as shipped)", fixed),
                       ("v0-ROLLED bank%s" % ("" if _c3ok else "  [INADMISSIBLE, C3 FAILED]"),
                        roll)):
        print("")
        print("  -- %s --" % name)
        print("  %6s %9s %10s %11s %11s %11s %10s"
              % ("lambda", "disp_m", "fan_env", "fan_peak_g", "fan_offrch", "oracle_ade",
                 "sel_ade"))
        for r in rows:
            print("  %6.2f %9.3f %10.4f %11.4f %11.4f %11.4f %10.4f"
                  % (r["lambda"], r["mean_abs_disp_m"], r["fan_envelope"], r["fan_peak_g"],
                     r["fan_off_reach"], r["oracle_ade_m"], r["sel_ade_m"]))
    f0, r0 = fixed[0], roll[0]
    print("")
    print("  AT lambda = 0 (the vocabulary itself)   fixed -> v0-rolled:")
    print("    off_reach  %.4f -> %.4f  (%+.4f)" % (f0["fan_off_reach"],
                                                    r0["fan_off_reach"],
                                                    r0["fan_off_reach"] - f0["fan_off_reach"]))
    print("    envelope   %.4f -> %.4f  (%+.4f)" % (f0["fan_envelope"], r0["fan_envelope"],
                                                    r0["fan_envelope"] - f0["fan_envelope"]))
    print("    peak_g     %.4f -> %.4f  (%+.4f)" % (f0["fan_peak_g"], r0["fan_peak_g"],
                                                    r0["fan_peak_g"] - f0["fan_peak_g"]))
    print("    oracle_ade %.4f -> %.4f  (%+.4f)" % (f0["oracle_ade_m"], r0["oracle_ade_m"],
                                                    r0["oracle_ade_m"] - f0["oracle_ade_m"]))
    print("")
    print("[v0roll] wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
