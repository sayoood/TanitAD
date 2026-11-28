"""S2 -- a PENETRATION-GRADED collision reward component, and its controls. 0 GPU.

WHY. `rewards._collision` returns -1 or 0. Inside a GRPO group every colliding candidate
therefore receives the SAME value, so the advantage can say "these are bad" but never "this one
is worse" -- it supplies no direction WITHIN the colliding set. MEASURED (p6): the stock term's
within-colliders std is 0.000e+00 (exactly flat), while severity varies 8.09x and the
within-window spread is non-degenerate in 19/19 mixed windows.

WHAT. Keep "every collision is bad" as a hard floor, and grade the remaining half by SWEPT
penetration depth:

    pen  = (r - d_min_swept).clamp_min(0)              # metres into the disc, 0 if clear
    out  = 0                        where pen == 0
    out  = -(FLOOR + (1-FLOOR) * min(1, pen / r))      where pen  > 0

with FLOOR = 0.5 and r = ego_radius + obs_radius = 2.0 m. Range [-1, 0], neutral 0 -- identical
to the stock component's contract, so it is a drop-in.

⭐ `r` IS THE NORMALISER AND IT IS A GEOMETRIC CONSTANT, NOT A FITTED ONE. The deepest possible
incursion into a disc of radius r is r, so pen/r lands in [0, 1] by construction. Nothing here is
tuned on the data being scored -- the probe-panel rule (CLAUDE.md 2026-08-22) forbids it, and a
severity scale fitted to the observed depths is exactly that error.

⛔ NOT graded by time-to-contact, which is what the brief proposed. MEASURED (p6): `min_ttc_s` is
saturated at the 0.5 s grid floor for >=75 % of colliders (p0 = p25 = p50 = 0.5000 s), so a
TTC-weighted term would be nearly as flat as the binary it replaces.

CONTROLS (all must pass, and each names the object it ran on):
  S1 SUPPORT-IDENTITY : sign(graded) != 0 must equal `rewards._collision` < 0 EXACTLY, on the
                        same tensor. The graded term must punish precisely the same candidates --
                        if it changes WHICH candidates are punished it is a different experiment.
  S2 NOT-FLAT         : within-colliders std must be > 0 (the whole point).
  S3 CLEAN-ZERO       : every non-collider must read EXACTLY 0.0 (the neutral value).
  S4 IN-RANGE         : all values within [-1, 0]; colliders <= -FLOOR.
  S5 MONOTONE         : deeper penetration must never score higher (rank correlation = -1).
"""
import os, sys, json, argparse, importlib.util

_REPO = os.environ.get("TANITAD_REPO") or r"C:\Users\Admin\tanitad-wt"
for _p in (os.path.join(_REPO, "stack"), os.path.join(_REPO, "taniteval"),
           os.path.join(_REPO, "taniteval", "tools")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch

FLOOR = 0.5


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


from tanitad.rl import rewards as RW


def swept_penetration(traj, ctx, sweep_first_segment=False):
    """Depth into the contact disc, in metres, using the SAME geometry as `_swept_hit`:
    the min over (a) every sampled point and (b) every segment's closest approach, in the
    RELATIVE frame (so the lead's own motion over the step is accounted for)."""
    r = float(ctx.get("ego_radius_m", 1.0)) + float(ctx.get("obs_radius_m", 1.0))
    lead = ctx["lead_path"]
    rel = lead[..., 1:, :] - traj[..., 1:, :]                     # [..., S-1, 2]
    origin = torch.zeros_like(rel[..., :1, :])                    # [..., 1, 2]
    d_pt = (rel.unsqueeze(-2) - origin.unsqueeze(-3)).norm(dim=-1).amin(dim=-1).amin(dim=-1)
    d = d_pt
    if rel.shape[-2] >= 2:
        d_seg = RW.segment_point_distance(rel[..., :-1, :], rel[..., 1:, :],
                                          origin).amin(dim=-1).amin(dim=-1)
        d = torch.minimum(d, d_seg)
    # ⭐ THE t0->t1 SEGMENT, which `rewards._collision` NEVER SWEEPS (M39/c9ab82c names the
    # defect; p7 measures it: +2 of 1,053 colliders = +0.19 % on THIS corpus, 0 new windows,
    # because no lead here is within r at t0 -- 0/65. It is NOT immaterial on a corpus with
    # close cut-ins or a parked obstacle, which is the case M39 describes, so the graded term
    # closes it rather than inheriting it).
    # ⛔ The t0 POINT is deliberately NOT tested: the ego is at its own origin at t0, so a lead
    # already inside the disc there is a SCENE property no plan can un-cause. Including it would
    # charge the planner for the scenario it was handed.
    # ⛔ DEFAULT OFF, AND THAT IS AN EXPERIMENTAL-HYGIENE DECISION, NOT AN OVERSIGHT.
    # Turning it on changes the SUPPORT of the term (which candidates are punished) as well as
    # the GRADING (how much). Shipping both at once breaks `one_variable`: if the arm then
    # moves, the movement is not attributable. S2's arm therefore runs support-identical to the
    # stock component, and the blind-spot fix is a SEPARATE lever with its own arm.
    if sweep_first_segment:
        rel0 = lead[..., 0:1, :] - traj[..., 0:1, :]
        d_seg0 = RW.segment_point_distance(rel0, rel[..., 0:1, :],
                                           origin).amin(dim=-1).amin(dim=-1)
        d = torch.minimum(d, d_seg0)
    return (r - d).clamp_min(0.0), r


def collision_graded(traj, ctx, floor=FLOOR, sweep_first_segment=False):
    pen, r = swept_penetration(traj, ctx, sweep_first_segment)
    frac = (pen / r).clamp(0.0, 1.0)
    out = -(floor + (1.0 - floor) * frac)
    return torch.where(pen > 0, out, torch.zeros_like(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="raw/fan_bank_base_240w.npz")
    ap.add_argument("--out", default="raw/s2_graded_collision.json")
    ap.add_argument("--sweep-first-segment", action="store_true",
                    help="close the t0->t1 blind spot too. DEFAULT OFF: it changes the SUPPORT "
                         "as well as the grading, which breaks one_variable for S2's arm.")
    a = ap.parse_args()

    FS = _load_by_path("fan_safety_for_s2",
                       os.path.join(_REPO, "taniteval", "tools", "fan_safety.py"))
    z = np.load(a.bank, allow_pickle=True)
    fan2 = torch.from_numpy(z["fan2"]).double()
    lead5 = torch.from_numpy(z["lead5"]).double()
    has_lead = torch.from_numpy(z["has_lead"]).bool()
    W, K = fan2.shape[0], fan2.shape[1]
    ctx = {"dt": FS.DT_S, "lead_len_m": FS.LEAD_LEN_DEFAULT_M,
           "lead_path": lead5[:, None, :, :].expand(W, K, 5, 2)}

    binary = RW.COMPONENTS["collision"](fan2, ctx).double()
    graded = collision_graded(fan2, ctx, sweep_first_segment=a.sweep_first_segment).double()
    pen, r = swept_penetration(fan2, ctx, a.sweep_first_segment)
    lead_mask = has_lead[:, None].expand(W, K)
    binary = binary * lead_mask
    graded = graded * lead_mask
    pen = pen * lead_mask

    b_hit = binary < 0
    g_hit = graded < 0
    print("OBJECT %s  fan2%s  r=%.1f m  FLOOR=%.2f" % (a.bank, tuple(fan2.shape), r, FLOOR))
    print("   binary colliders %d   graded colliders %d" % (int(b_hit.sum()), int(g_hit.sum())))

    lost = int((b_hit & ~g_hit).sum())
    gained = int((g_hit & ~b_hit).sum())
    if a.sweep_first_segment:
        # the fix is ON: the support must be a strict SUPERSET, gaining exactly p7's 2.
        s1 = lost
        print("S1 SUPPORT-SUPERSET : lost=%d (must be 0)  gained=%d (p7 measured 2) -> %s"
              % (lost, gained, "PASS" if lost == 0 else "FAIL"))
    else:
        s1 = lost + gained
        print("S1 SUPPORT-IDENTITY : disagreements = %d -> %s"
              % (s1, "PASS" if s1 == 0 else "FAIL"))

    gv = graded[b_hit]
    s2 = float(gv.std()) if gv.numel() > 1 else 0.0
    bv = binary[b_hit]
    print("S2 NOT-FLAT         : graded std within colliders = %.6f (binary = %.3e) -> %s"
          % (s2, float(bv.std()), "PASS" if s2 > 0 else "FAIL"))

    off = ~g_hit if a.sweep_first_segment else ~b_hit
    s3 = float(graded[off].abs().max()) if off.any() else 0.0
    print("S3 CLEAN-ZERO       : max|graded| off its OWN support = %.3e -> %s"
          % (s3, "PASS" if s3 == 0.0 else "FAIL"))

    s4 = bool(graded.min() >= -1.0 and graded.max() <= 0.0 and gv.max() <= -FLOOR + 1e-12)
    print("S4 IN-RANGE         : min %.4f max %.4f, colliders <= -%.2f -> %s"
          % (float(graded.min()), float(graded.max()), FLOOR, "PASS" if s4 else "FAIL"))

    pv, gvv = pen[b_hit].numpy(), gv.numpy()
    if pv.size > 1:
        rho = float(np.corrcoef(np.argsort(np.argsort(pv)), np.argsort(np.argsort(gvv)))[0, 1])
    else:
        rho = -1.0
    s5 = abs(rho + 1.0) < 1e-9
    print("S5 MONOTONE         : rank corr(penetration, graded) = %+.9f (must be -1) -> %s"
          % (rho, "PASS" if s5 else "FAIL"))

    print()
    print("graded value spread among colliders: min %.4f  p25 %.4f  p50 %.4f  p75 %.4f  max %.4f"
          % (gvv.min(), np.percentile(gvv, 25), np.percentile(gvv, 50),
             np.percentile(gvv, 75), gvv.max()))
    print("swept penetration (m):              min %.4f  p50 %.4f  max %.4f"
          % (pv.min(), np.median(pv), pv.max()))
    n_sampled = int((b_hit & ((2.0 - (lead5[:, None, 1:, :].expand(W, K, 4, 2)
                                       - fan2[:, :, 1:, :]).norm(dim=-1).amin(-1)) > 0)).sum())
    print("   swept penetration > 0 for ALL %d colliders; the per-SAMPLE depth is > 0 for only "
          "%d, so grading on sampled points alone would leave %d colliders flat at the floor"
          % (int(b_hit.sum()), n_sampled, int(b_hit.sum()) - n_sampled))

    ok = (s1 == 0) and (s2 > 0) and (s3 == 0.0) and s4 and s5
    print("\nS2_COMPONENT_CONTROLS=%s" % ("PASS" if ok else "FAIL"))

    out = {"_tool": "s2_graded_collision.py",
           "_tier": "T0 component readout on the EMITTED fan (never a driving claim)",
           "_evidence_class": "MEASURED (ours)",
           "bank": a.bank, "contact_radius_m": r, "floor": FLOOR,
           "sweep_first_segment": bool(a.sweep_first_segment),
           "normaliser": "the contact radius r itself -- a GEOMETRIC constant, not fitted on "
                         "the scored data (probe-panel rule)",
           "not_graded_by": "min_ttc_s -- saturated at the 0.5 s grid floor for >=75% of "
                            "colliders (p6), so a TTC-weighted term would be nearly as flat "
                            "as the binary it replaces",
           "controls": {"S1_support_disagreements": s1,
                        "S2_graded_std_within_colliders": s2,
                        "S2_binary_std_within_colliders": float(bv.std()),
                        "S3_max_abs_off_support": s3,
                        "S4_in_range": s4,
                        "S5_rank_corr_pen_vs_graded": rho,
                        "ALL_PASS": bool(ok)},
           "n_colliders": int(b_hit.sum()),
           "graded_spread": {"min": float(gvv.min()), "p50": float(np.percentile(gvv, 50)),
                             "max": float(gvv.max())},
           "swept_penetration_m": {"min": float(pv.min()), "p50": float(np.median(pv)),
                                   "max": float(pv.max())}}
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print("WROTE %s" % a.out)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
