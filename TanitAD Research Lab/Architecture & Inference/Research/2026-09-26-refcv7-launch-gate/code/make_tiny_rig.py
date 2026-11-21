"""Write the dev-box REHEARSAL rig for the launch gate: a tiny anchor vocabulary and a launch argv.

    python make_tiny_rig.py --tree <code tree> --out-dir <dir>

The argv is the one `stack/tests/test_refcv6_f3_cascade_reaches_loss.py` already trains on the
REAL `train()` (smoke config, synthetic episodes, CPU, refcv6's F1-F6 diffusion flags), with the
LAUNCH cadences (--log-every 50, --save-every 500) instead of the test's --log-every 1: a smoke
must run as launched (the 2026-09-23 log-cadence trap). It exercises the gate's machinery --
F3's cascade term, the optimizer, the checkpoint, the resume -- in seconds. ⛔ It is a rehearsal,
never launch evidence: its argv sha is not refcv7's, so no token it produces can bind a launch.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--synth-episodes", type=int, default=4)
    a = ap.parse_args(argv)
    tree = Path(a.tree).resolve()
    sys.path.insert(0, str(tree / "stack"))
    import torch
    from tanitad.refs import anchor_meta as am
    from tanitad.refs import refc_sampler as rs
    from tanitad.refs.refc_v3 import V3_HORIZONS
    out = Path(a.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    # the F3 test's 20-anchor v0-conditioned alat vocabulary, INCLUDING straight-ahead
    a_lon = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    a_lat = torch.tensor([-1.5, -0.5, 0.0, 0.5])
    ctrl = torch.cartesian_prod(a_lon, a_lat)
    n, s = ctrl.shape[0], len(V3_HORIZONS)
    u = ctrl[None, :, None, :].expand(1, n, s, 2).contiguous()
    anchors = rs.roll_controls(u, torch.tensor([10.0]), tuple(V3_HORIZONS),
                               control_units="alat")[0]
    art = am.build_anchor_artifact(anchors, ctrl, control_units="alat", horizons=V3_HORIZONS,
                                   dt=0.1, ref_speed_ms=10.0, kappa_cap=0.12, alat_v_floor=4.0,
                                   builder=None)
    anc = out / "anchors_v0cond_alat_20.pt"
    torch.save(art, anc)
    argv_launch = ["--arm", "hier", "--smoke", "--synth-episodes", str(a.synth_episodes),
                   "--device", "cpu", "--sampler", "ddim", "--anchors", anc.as_posix(),
                   "--anchor-v0-conditioned", "--n-anchors", "20",
                   "--f1-random-t", "--f2-dd-step", "--f3-per-layer", "--f4-adaln", "--f5-focal",
                   "--f5-emitting-conf", "--f6-w-u0-zero",
                   "--seed", "0", "--save-every", "500", "--log-every", "50", "--batch", "2",
                   "--steps", "50400", "--out", (out / "launch_out").as_posix()]
    (out / "argv_tiny.json").write_text(json.dumps(argv_launch, indent=1), encoding="utf-8")
    print(json.dumps({"anchors": anc.as_posix(), "argv": (out / "argv_tiny.json").as_posix()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
