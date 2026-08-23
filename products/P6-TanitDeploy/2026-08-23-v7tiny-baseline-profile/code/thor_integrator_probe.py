"""E-DEPLOY-1d - is the fp16 plan deviation born INSIDE the kinematic integrator?

THE ARGUMENT THIS TESTS. On the champ30k (stage S-W) checkpoint the emission
head's controls are EXACTLY zero (`a` 480/480 zeros, `kappa` 480/480 zeros,
verified by content), and `waypoints = unicycle_rollout(a, kappa, v0, dt)`.
If a and kappa are bit-identical between fp32 and fp16, the ONLY place a
waypoint difference can be born is inside the rollout itself - finite-precision
accumulation of the travelled distance.

DECISIVE TEST: call `unicycle_rollout` STANDALONE with exactly-zero controls and
the model's own v0, in fp32 and under autocast-fp16/bf16. If the standalone
deviation reproduces the end-to-end deviation, the mechanism is confirmed and
the fix - keep the integrator out of autocast - is proven to remove all of it.

CONTROLS
  * zero-control sanity: with a=kappa=0 the fp32 rollout must be a straight line
    at constant v0 -> x[T-1] == v0*T*dt and y == 0 everywhere. If that does not
    hold, `unicycle_rollout` is not doing what this argument assumes.
  * fp16 ULP reference: the deviation is compared against the fp16 spacing at
    the travelled distance. A deviation of a few ULP IS representation error;
    orders more would mean something else is going on.

Touches no model file. Writes one JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

STACK = os.path.expanduser("~/TanitAD/stack")
sys.path.insert(0, STACK)
sys.path.insert(0, os.path.join(STACK, "scripts"))

import torch  # noqa: E402

R = {"spec": "E-DEPLOY-1d", "controls": {}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dv = torch.device("cuda")

    from train_v58f_unicycle_head import unicycle_rollout

    B, N, T, dt = 1, 8, 60, 0.1
    # the SAME v0 the profile used: synthetic_train_batch's rand(b)*20+1, seed 0
    g = torch.Generator().manual_seed(0)
    # reproduce the draw order of synthetic_train_batch exactly enough to get a
    # v0 in the same range; the exact value is reported, not assumed.
    v0 = (torch.rand(B, generator=g) * 20.0 + 1.0).to(dv)
    ctl_a = torch.zeros(B, N, T, device=dv)
    ctl_k = torch.zeros(B, N, T, device=dv)

    wp32, _ = unicycle_rollout(ctl_a, ctl_k, v0, dt=dt)
    wp32 = wp32.float()

    # ---- CONTROL: zero control => straight line at constant v0 ------------- #
    x_end = float(wp32[0, 0, -1, 0].item())
    y_absmax = float(wp32[..., 1].abs().max().item())
    x_expect = float(v0.item()) * T * dt
    straight = abs(x_end - x_expect) < 0.5 and y_absmax == 0.0
    R["controls"]["zero_control_is_straight_line"] = {
        "expect": "x[T-1] ~= v0*T*dt and y == 0 everywhere",
        "v0_m_s": float(v0.item()), "x_end_m": x_end,
        "x_expect_m": x_expect, "y_absmax": y_absmax, "pass": bool(straight)}
    print("CONTROL straight-line: v0=%.4f x_end=%.4f expect=%.4f y_absmax=%.3g"
          " -> %s" % (v0.item(), x_end, x_expect, y_absmax, straight),
          flush=True)

    R["standalone_deviation"] = {}
    for name, dt_t in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
        with torch.autocast("cuda", dtype=dt_t):
            wp, _ = unicycle_rollout(ctl_a, ctl_k, v0, dt=dt)
        wp = wp.float()
        d = (wp - wp32).abs()
        per_step = d.amax(dim=(0, 1, 3)).tolist()
        # what is one ULP of this dtype at the travelled distance?
        ulp = float((torch.tensor([x_end], dtype=dt_t).to(dv).float()
                     - torch.nextafter(torch.tensor([x_end], dtype=dt_t),
                                       torch.tensor(float("inf"),
                                                    dtype=dt_t)).to(dv).float()
                     ).abs().item())
        R["standalone_deviation"][name] = {
            "max_abs_m": float(d.max().item()),
            "mean_abs_m": float(d.mean().item()),
            "out_dtype": str(wp.dtype),
            "ulp_at_x_end_m": ulp,
            "deviation_in_ulp": (float(d.max().item()) / ulp) if ulp else None,
            "per_step_t0": per_step[0], "per_step_t9": per_step[9],
            "per_step_t29": per_step[29], "per_step_t59": per_step[-1],
            "monotone_nondecreasing": all(
                per_step[i] <= per_step[i + 1] + 1e-9
                for i in range(len(per_step) - 1))}
        print("%s standalone: max_abs %.4e m  = %.1f ULP (ulp %.4e)  "
              "t0 %.3e -> t59 %.3e  mono=%s"
              % (name, d.max().item(),
                 (d.max().item() / ulp) if ulp else float("nan"), ulp,
                 per_step[0], per_step[-1],
                 R["standalone_deviation"][name]["monotone_nondecreasing"]),
              flush=True)

    R["end_to_end_reference"] = {
        "note": "from raw/thor_v7tiny_mechanism.json, b=1",
        "fp16_max_abs_m": 8.6628e-02, "bf16_max_abs_m": 2.6179e+00}
    for nm in ("fp16", "bf16"):
        s = R["standalone_deviation"][nm]["max_abs_m"]
        e = R["end_to_end_reference"][nm + "_max_abs_m"]
        R["standalone_deviation"][nm]["reproduces_end_to_end"] = {
            "standalone_m": s, "end_to_end_m": e,
            "ratio_standalone_over_e2e": (s / e) if e else None,
            "same_order_of_magnitude": bool(e > 0 and 0.1 <= (s / e) <= 10.0)}
        print("%s reproduction: standalone %.4e vs end-to-end %.4e -> ratio "
              "%.3f" % (nm, s, e, (s / e) if e else float("nan")), flush=True)

    R["_evidence_class"] = "MEASURED (ours; Thor, in-process probes only)"
    with open(a.out, "w") as fh:
        json.dump(R, fh, indent=1)
    print("WROTE", a.out, flush=True)


if __name__ == "__main__":
    main()
