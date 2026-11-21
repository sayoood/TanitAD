"""E-DEPLOY-1d (v2) - the integrator probe, with the dtype variable controlled.

⚠️ WHY THERE IS A v2. v1 passed fp32 zeros for `a`/`kappa` and measured EXACTLY
zero deviation, apparently refuting the integrator hypothesis. That was an
instrument defect, not a result: inside the model under autocast, `a_ctl` and
`kappa` are produced by the denoiser and therefore arrive as **fp16**, so
`unicycle_rollout` runs in fp16 and emits fp16 waypoints. v1 held the very
variable it meant to vary at fp32. The controls are numerically zero in BOTH
precisions - what differs is their DTYPE, and the dtype is what propagates.

v2 varies the control dtype, and adds the discriminating measurement v1 lacked:
the DTYPE of the emitted waypoints, and the fp16/bf16 spacing (ULP) at the
travelled distance. A deviation of ~1-2 ULP of the OUTPUT dtype is output
representation error and nothing deeper.

Touches no model file.
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

R = {"spec": "E-DEPLOY-1d-v2", "controls": {},
     "v1_defect": "v1 passed fp32 controls under autocast and measured 0.0 "
                  "deviation; autocast does not cast cumsum/sin/cos inputs, so "
                  "the rollout stayed fp32 and the test held its own "
                  "independent variable fixed."}


def ulp_at(x, dtype, dev):
    t = torch.tensor([x], dtype=dtype)
    nxt = torch.nextafter(t, torch.tensor(float("inf"), dtype=dtype))
    return float((nxt.float() - t.float()).abs().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dv = torch.device("cuda")
    from train_v58f_unicycle_head import unicycle_rollout

    B, N, T, dt = 1, 8, 60, 0.1
    g = torch.Generator().manual_seed(0)
    v0 = (torch.rand(B, generator=g) * 20.0 + 1.0).to(dv)

    z32a = torch.zeros(B, N, T, device=dv, dtype=torch.float32)
    wp32, _ = unicycle_rollout(z32a, z32a.clone(), v0, dt=dt)
    wp32f = wp32.float()
    x_end = float(wp32f[0, 0, -1, 0].item())

    R["reference"] = {"v0_m_s": float(v0.item()), "x_end_m": x_end,
                      "wp32_dtype": str(wp32.dtype),
                      "y_absmax": float(wp32f[..., 1].abs().max().item())}
    print("fp32 reference: v0 %.4f  x_end %.4f m  dtype %s"
          % (v0.item(), x_end, wp32.dtype), flush=True)

    R["by_control_dtype"] = {}
    for name, dtp in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
        za = torch.zeros(B, N, T, device=dv, dtype=dtp)
        wp, _ = unicycle_rollout(za, za.clone(), v0, dt=dt)
        d = (wp.float() - wp32f).abs()
        u_out = ulp_at(x_end, wp.dtype if wp.dtype != torch.float32 else dtp,
                       dv)
        per = d.amax(dim=(0, 1, 3)).tolist()
        R["by_control_dtype"][name] = {
            "control_dtype": str(dtp), "waypoints_out_dtype": str(wp.dtype),
            "max_abs_m": float(d.max().item()),
            "mean_abs_m": float(d.mean().item()),
            "ulp_of_out_dtype_at_x_end_m": u_out,
            "deviation_in_ulp": (float(d.max().item()) / u_out) if u_out else None,
            "per_step_t0": per[0], "per_step_t9": per[9],
            "per_step_t29": per[29], "per_step_t59": per[-1],
            "monotone_nondecreasing": all(per[i] <= per[i + 1] + 1e-9
                                          for i in range(len(per) - 1))}
        print("%s controls -> wp dtype %s  max_abs %.4e m = %.2f ULP  "
              "t0 %.3e t29 %.3e t59 %.3e"
              % (name, wp.dtype, d.max().item(),
                 (d.max().item() / u_out) if u_out else float("nan"),
                 per[0], per[29], per[-1]), flush=True)

    # ---- does it reproduce the end-to-end number? ------------------------- #
    R["end_to_end_reference"] = {"source":
                                 "raw/thor_v7tiny_mechanism.json b=1",
                                 "fp16_max_abs_m": 8.6628e-02,
                                 "bf16_max_abs_m": 2.6179e+00}
    for nm in ("fp16", "bf16"):
        s = R["by_control_dtype"][nm]["max_abs_m"]
        e = R["end_to_end_reference"][nm + "_max_abs_m"]
        ratio = (s / e) if e else None
        R["by_control_dtype"][nm]["reproduces_end_to_end"] = {
            "standalone_m": s, "end_to_end_m": e, "ratio": ratio,
            "same_order_of_magnitude": bool(
                e > 0 and 0.1 <= (s / e) <= 10.0)}
        print("%s reproduction: standalone %.4e vs end-to-end %.4e  ratio %.3f"
              % (nm, s, e, ratio if ratio else float("nan")), flush=True)

    # ---- CONTROL: the claimed FIX must remove it -------------------------- #
    # "Keep the integrator in fp32" = cast the controls up before the rollout.
    # If that does NOT return a bit-identical fp32 result, the fix is not the
    # fix and must not be recommended.
    for name, dtp in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
        za = torch.zeros(B, N, T, device=dv, dtype=dtp)
        wp_fix, _ = unicycle_rollout(za.float(), za.float().clone(), v0, dt=dt)
        same = bool(torch.equal(wp_fix.float(), wp32f))
        R["by_control_dtype"][name]["fix_upcast_before_rollout"] = {
            "expect": "bit-identical to the fp32 reference",
            "bit_identical": same,
            "max_abs_m": float((wp_fix.float() - wp32f).abs().max().item())}
        print("%s FIX (upcast before rollout): bit-identical=%s"
              % (name, same), flush=True)

    R["_evidence_class"] = "MEASURED (ours; Thor, in-process probes only)"
    with open(a.out, "w") as fh:
        json.dump(R, fh, indent=1)
    print("WROTE", a.out, flush=True)


if __name__ == "__main__":
    main()
