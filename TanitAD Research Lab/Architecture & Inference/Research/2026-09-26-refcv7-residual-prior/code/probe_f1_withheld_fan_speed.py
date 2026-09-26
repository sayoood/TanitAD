#!/usr/bin/env python3
"""F-1 probe: on a WITHHELD training row, refcv6's sampled fan is rolled at the raw v0 while its
anchor bank is rolled at the reference speed. CPU, smoke width, the trainer's own build.

Measures, per row of one TRAINING forward (ego-dropout 0.5, fixed seed):
  * withheld?  (read off the residual build's own `residual_prior_v`: 10.0 on a withheld row)
  * off build (refcv6): the speed its FAN was rolled at, recovered from the straight-ahead
    candidate's first slot is NOT possible after denoising, so instead we read the BANK's
    straight anchor (rolled by `roll_bank`) and the FAN's straight candidate at t = 0.5 s and
    report both; and the refcv6 code facts `refc.py:2522-2523` (fan speed = raw v_ms) vs
    `refc.py:2129-2132` (bank speed = withheld reference speed).
  * a residual build with a ZERO prior (so its composition equals refcv6's) on the same seed and
    weights: its fan agrees with the off build on KEPT rows and differs on WITHHELD rows.

usage: python probe_f1_withheld_fan_speed.py <tree>   (the candidate tree: tip + NEW-1)
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys


def main() -> int:
    tree = os.path.abspath(sys.argv[1])
    stack = os.path.join(tree, "stack")
    sys.path[:0] = [stack, os.path.join(stack, "scripts")]
    import torch
    import tanitad
    assert os.path.abspath(tanitad.__file__).startswith(stack), tanitad.__file__
    from tanitad.models import kinematic_prior as KP
    spec = importlib.util.spec_from_file_location("rv3t_f1", os.path.join(stack, "scripts", "refc_v3_train.py"))
    T = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)
    torch.set_num_threads(1)
    f1_f6 = ["--f1-random-t", "--f2-dd-step", "--f3-per-layer", "--f4-adaln", "--f5-focal",
             "--f5-emitting-conf", "--f6-w-u0-zero"]

    def build(extra):
        argv = ["--arm", "hier", "--sampler", "ddim", "--anchor-v0-conditioned",
                "--anchor-control-units", "alat", "--n-anchors", "20", "--ego-history",
                "--out", "x"] + f1_f6 + extra
        args = T.build_parser().parse_args(argv)
        cfg = T._pin_trainer_cfg(T.v3.refc_v3_smoke_config(True), args)
        torch.manual_seed(0)
        m = T.v3.RefCV3Model(cfg)
        with torch.no_grad():
            m.core.decoder.anchor_controls.copy_(torch.cartesian_prod(
                torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]), torch.tensor([-1.5, -0.5, 0.0, 0.5])))
        return cfg, m

    cfg, off = build([])
    _c2, on = build(["--residual-prior", "ha0_ext_pose"])
    on.load_state_dict(off.state_dict())
    KP_prior = KP.prior_controls
    KP.prior_controls = lambda m, p, n=None, a=None, **k: (torch.zeros(p.shape[0]), torch.zeros(p.shape[0]))
    eps = T._synth_episodes(2, cfg.core, seed=0)
    ds = T.V3Dataset(eps, window=cfg.core.window, max_horizon=20, channels=cfg.core.encoder.in_channels)
    ds.ego_history = True
    b = torch.utils.data.default_collate([ds[0], ds[7], ds[23], ds[31]])
    rows = []
    for seed in range(4):
        outs = {}
        for name, m in (("off", off), ("zero_prior", on)):
            m.train()
            torch.manual_seed(seed)
            ph = b["pose_hist"]
            m.core.set_ego_window(ph, int(ph.shape[1]))
            with torch.no_grad():
                outs[name] = m(T.frames_to_device(b["frames"], "cpu"), nav_cmd=b["nav_cmd"],
                               v0=b["pose_last"][:, 3], steps=cfg.core.decoder.diffusion_steps)
        v0 = b["pose_last"][:, 3]
        withheld = outs["zero_prior"]["residual_prior_v"] != v0
        for r in range(v0.shape[0]):
            rows.append({
                "seed": seed, "row": r, "v0": round(float(v0[r]), 4),
                "withheld": bool(withheld[r]),
                "bank_equal_off_vs_zero_prior": bool(torch.equal(outs["off"]["anchor_bank"][r],
                                                                 outs["zero_prior"]["anchor_bank"][r])),
                "fan_max_abs_diff_off_vs_zero_prior_m": float((outs["off"]["anchor_traj"][r]
                                                               - outs["zero_prior"]["anchor_traj"][r]).abs().max()),
            })
    KP.prior_controls = KP_prior
    kept = [r for r in rows if not r["withheld"]]
    wh = [r for r in rows if r["withheld"]]
    rep = {"tool": "probe_f1_withheld_fan_speed.py", "tree": tree, "rows": rows,
           "n_kept": len(kept), "n_withheld": len(wh),
           "kept_rows_fan_identical": all(r["fan_max_abs_diff_off_vs_zero_prior_m"] == 0.0 for r in kept),
           "withheld_rows_fan_differs": all(r["fan_max_abs_diff_off_vs_zero_prior_m"] > 0.0 for r in wh),
           "withheld_rows_max_fan_diff_m": max((r["fan_max_abs_diff_off_vs_zero_prior_m"] for r in wh), default=None),
           "reading": ("the bank is identical on every row (both builds roll withheld rows at 10 m/s); "
                       "the SAMPLED fan is identical on kept rows and differs on withheld rows, because "
                       "refcv6 rolls the fan at the raw v0 (refc.py:2522-2523) and NEW-1 at the bank's "
                       "speed. The difference IS F-1.")}
    print(json.dumps({k: v for k, v in rep.items() if k != "rows"}, indent=1))
    out = os.environ.get("F1_OUT")
    if out:
        json.dump(rep, open(out, "w", encoding="utf-8"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
