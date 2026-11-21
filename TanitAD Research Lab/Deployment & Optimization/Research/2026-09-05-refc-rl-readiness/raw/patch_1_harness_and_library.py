"""Patch 1 (REF-C RL-readiness WP, 2026-09-05): harness inert-buffer tolerance +
time-aligned moving-lead contact in the RL reward library + its pinning test.
Run from C:\\Users\\Admin\\refcv4b_repo. Every anchor must match EXACTLY once."""
import os
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()


def patch(rel, old, new, count=1):
    path = os.path.join(ROOT, rel)
    s = open(path, encoding="utf-8").read()
    n = s.count(old)
    assert n == count, f"{rel}: expected {count} match(es), found {n}: {old[:70]!r}"
    s = s.replace(old, new)
    open(path, "w", encoding="utf-8", newline="\n").write(s)
    print("patched", rel, "x", count)


# ---- A. harness: tolerate the ONE inert buffer, recorded --------------------------
OLD_A = '''    strict_rep = {"missing_keys": list(res.missing_keys),
                  "unexpected_keys": list(res.unexpected_keys)}
    if (res.missing_keys or res.unexpected_keys) and not allow_nonstrict:
        raise SystemExit(f"[refcv3_arm] \u26d4 NON-STRICT LOAD: missing "
                         f"{list(res.missing_keys)[:8]} unexpected "
                         f"{list(res.unexpected_keys)[:8]} \u2014 the checkpoint and "
                         f"the rebuilt model disagree. Pass --allow-nonstrict "
                         f"only for a deliberate diagnostic, and say so.")
'''
NEW_A = '''    strict_rep = {"missing_keys": list(res.missing_keys),
                  "unexpected_keys": list(res.unexpected_keys)}
    # \u2b50 INERT-BUFFER TOLERANCE (2026-09-05, REF-C RL-readiness WP). refcv4-b
    # added the PERSISTENT buffer `core.decoder.anchor_controls` (refc.py:1168),
    # which `roll_bank` reads ONLY when the decoder is `v0_conditioned`. A
    # checkpoint trained before it existed (refcv3 @ 40,284, the published HF
    # weights) is therefore MISSING a key the rebuilt model never reads, and the
    # strict refusal was a FALSE refusal \u2014 MEASURED: STAGE 0 of the RL chain died
    # on it with the md5-verified base. Tolerated only when (a) the decoder is NOT
    # v0-conditioned, (b) nothing else is missing and nothing is unexpected \u2014 and
    # RECORDED in the provenance, never silent.
    _dec = getattr(model.core, "decoder", None)
    inert = sorted(k for k in res.missing_keys
                   if k.endswith(".anchor_controls") and _dec is not None
                   and not bool(getattr(_dec, "anchor_v0_cond", False)))
    strict_rep["tolerated_inert_buffers"] = inert
    real_missing = [k for k in res.missing_keys if k not in inert]
    if (real_missing or res.unexpected_keys) and not allow_nonstrict:
        raise SystemExit(f"[refcv3_arm] \u26d4 NON-STRICT LOAD: missing "
                         f"{real_missing[:8]} unexpected "
                         f"{list(res.unexpected_keys)[:8]} \u2014 the checkpoint and "
                         f"the rebuilt model disagree. Pass --allow-nonstrict "
                         f"only for a deliberate diagnostic, and say so.")
'''
patch("taniteval/tools/refcv3_arm.py", OLD_A, NEW_A)

# ---- B. library: time-aligned contact against a MOVING lead ---------------------
OLD_B = '''    obs = ctx.get("obstacles")
    if obs is None or obs.numel() == 0:
        return torch.zeros(traj.shape[:-2], device=traj.device, dtype=traj.dtype)
    r = float(ctx.get("ego_radius_m", 1.0)) + float(ctx.get("obs_radius_m", 1.0))
    # traj [..., S, 2] vs obs [..., K, 2] -> pairwise [..., S, K]
    d = (traj.unsqueeze(-2) - obs.unsqueeze(-3)).norm(dim=-1)
    hit = (d < r).any(dim=-1).any(dim=-1)
    return torch.where(hit, -torch.ones_like(hit, dtype=traj.dtype),
                       torch.zeros_like(hit, dtype=traj.dtype))
'''
NEW_B = '''    obs = ctx.get("obstacles")
    r = float(ctx.get("ego_radius_m", 1.0)) + float(ctx.get("obs_radius_m", 1.0))
    if obs is None or obs.numel() == 0:
        # \u2b50 MOVING-LEAD CONTACT (2026-09-05, REF-C RL-readiness WP). When no
        # static obstacle set is supplied but a `lead_path [..., S, 2]` is, contact
        # is TIME-ALIGNED: step s of the candidate against step s of the lead \u2014
        # the per-step convention `_headway` and `ttc_violation` already use.
        # WHY: holding the lead STATIC at its first sample turns every competent
        # follower into a "collision" \u2014 the ego reaches the lead's t0 position
        # after one time-gap, so any human path with a time gap shorter than the
        # horizon is flagged (the H-RL-THRESH-1 class: a safety term that fires
        # on the demonstration). Measured on the refcv3 RL-fit clips by
        # `rl_refcv3_min.py --mode humanflag`. Step 0 is skipped (the ego is at
        # its own origin; the lead is ahead by construction).
        lead = ctx.get("lead_path")
        if lead is None:
            return torch.zeros(traj.shape[:-2], device=traj.device, dtype=traj.dtype)
        d = (lead[..., 1:, :] - traj[..., 1:, :]).norm(dim=-1)      # [..., S-1]
        hit = (d < r).any(dim=-1)
        return torch.where(hit, -torch.ones_like(hit, dtype=traj.dtype),
                           torch.zeros_like(hit, dtype=traj.dtype))
    # traj [..., S, 2] vs obs [..., K, 2] -> pairwise [..., S, K]
    d = (traj.unsqueeze(-2) - obs.unsqueeze(-3)).norm(dim=-1)
    hit = (d < r).any(dim=-1).any(dim=-1)
    return torch.where(hit, -torch.ones_like(hit, dtype=traj.dtype),
                       torch.zeros_like(hit, dtype=traj.dtype))
'''
patch("stack/tanitad/rl/rewards.py", OLD_B, NEW_B)

OLD_B2 = '''    """-1 if the path comes within (ego_r + obs_r) of any obstacle, else 0.
'''
NEW_B2 = '''    """-1 if the path comes within (ego_r + obs_r) of any obstacle, else 0.

    Two obstacle models: a STATIC set ``ctx["obstacles"]`` (every step against
    every obstacle), or \u2014 when no static set is given \u2014 a MOVING
    ``ctx["lead_path"]`` checked TIME-ALIGNED per step (see the branch below).
'''
patch("stack/tanitad/rl/rewards.py", OLD_B2, NEW_B2)

# ---- B'. the pinning test ---------------------------------------------------------
TEST = '''

def test_collision_time_aligned_against_moving_lead():
    """A STATIC lead flags a competent follower; a MOVING (time-aligned) lead does
    not \u2014 and a genuinely too-close moving lead still fires. (2026-09-05: the
    static-lead defect measured by `rl_refcv3_min.py --mode humanflag`.)"""
    import torch
    from tanitad.rl import rewards as R
    dt, v = 0.5, 10.0
    t = torch.arange(5, dtype=torch.float32) * dt
    ego = torch.stack([v * t, torch.zeros_like(t)], dim=-1)              # [5, 2]
    # lead 8 m ahead (centre-to-centre), same speed: the gap stays 8 m > r = 2 m
    lead_far = torch.stack([8.0 + v * t, torch.zeros_like(t)], dim=-1)
    static_ctx = {"dt": dt, "obstacles": lead_far[:1].clone()}   # held at t0
    moving_ctx = {"dt": dt, "lead_path": lead_far}
    assert float(R.COMPONENTS["collision"](ego, static_ctx)) == -1.0
    assert float(R.COMPONENTS["collision"](ego, moving_ctx)) == 0.0
    # a moving lead only 1.5 m ahead (inside r = 2 m) DOES fire time-aligned
    lead_close = torch.stack([1.5 + v * t, torch.zeros_like(t)], dim=-1)
    assert float(R.COMPONENTS["collision"](ego, {"dt": dt, "lead_path": lead_close})) == -1.0
    # the legacy path is untouched: obstacles present -> static semantics
    both = {"dt": dt, "obstacles": lead_far[:1].clone(), "lead_path": lead_far}
    assert float(R.COMPONENTS["collision"](ego, both)) == -1.0
    # fan-shaped broadcasting: [B, N, G, S, 2] against a lead [B, 1, 1, S, 2]
    fan = ego.reshape(1, 1, 1, 5, 2).expand(2, 3, 4, 5, 2).clone()
    out = R.COMPONENTS["collision"](fan, {"dt": dt,
                                          "lead_path": lead_far.reshape(1, 1, 1, 5, 2)})
    assert tuple(out.shape) == (2, 3, 4) and float(out.abs().sum()) == 0.0
'''
tp = os.path.join(ROOT, "stack/tests/test_rl_rewards.py")
s = open(tp, encoding="utf-8").read()
assert "test_collision_time_aligned_against_moving_lead" not in s
open(tp, "w", encoding="utf-8", newline="\n").write(s + TEST)
print("appended test")
