"""X8 / D-REFCV7-FANCEIL: dump the refcv7 candidate FAN, one row per window, on the SAME windows,
SAME ckpt, SAME code path as the banked battery's `os` arm (inference seed 0, os-only roll).

WHY: the banked battery dumps hold only the SELECTED 4-waypoint plan per arm, so a fan-vs-selected
comparison (oracle-in-fan, random-pick control) cannot be computed from them. The only prior fan
measurement (taniteval/results/scaleab_refc-base-30k_vs_refc-xl-30k.json) is refc-base-30k /
refc-xl-30k, NOT refcv7.

HOW: a WRAPPER. It does not edit run_battery_r7.py / r7_roll.py. It replaces
`Refcv6Proxy.__call__` with a function that calls the original UNCHANGED (same RNG stream, same
model call, same banked sidecar rows) and only then copies the `nav_true` row of
`out["anchor_traj"]` (and siblings) out of the returned dict. After the roll it asserts the fan rows
align 1:1 with the sidecar the roll itself wrote and that the captured `traj` equals the sidecar's
`nav_true.traj` exactly.

The fan is kept for the 2 s grid only (model slots 0..3 = horizons 5/10/15/20 steps): the GT `g`
in the dump has 4 waypoints. fp32 on disk.
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8")
from tanitad.eval import refcv7_loader as L  # noqa: E402

L.bootstrap()

import numpy as np  # noqa: E402
import torch  # noqa: E402

FAN_KEYS = ("anchor_traj", "anchor_logits", "refined_logits", "sel_score", "sel_idx", "traj")
FAN = {}
STATS = {"n_calls_with_nav": 0, "conds_seen": set()}


def _install_capture(RR):
    orig = RR.Refcv6Proxy.__call__

    def wrapped(self, frames, nav_cmd=None, v0=None, steps=0, ego_state=None, **kw):
        out = orig(self, frames, nav_cmd=nav_cmd, v0=v0, steps=steps, ego_state=ego_state, **kw)
        if nav_cmd is not None:
            conds = list(self.conds_fed[:int(frames.shape[0])])
            STATS["conds_seen"].add(tuple(conds))
            if "nav_true" not in conds:
                raise RuntimeError("[fan_roll] nav_true not among conds %s" % (conds,))
            r = conds.index("nav_true")
            row = {}
            for k in FAN_KEYS:
                v = out.get(k) if hasattr(out, "get") else None
                if v is None or not torch.is_tensor(v):
                    continue
                a = v[r].detach().float().cpu().numpy()
                if k == "anchor_traj":
                    a = a[:, :4, :]            # N x 4 x 2 : the 2 s grid (slots 0..3)
                row[k] = a
            if "anchor_traj" not in row:
                raise RuntimeError("[fan_roll] model output carries no anchor_traj")
            if self.ctx.index in FAN:
                raise RuntimeError("[fan_roll] duplicate nav call for window %s" % (self.ctx.index,))
            FAN[self.ctx.index] = row
            STATS["n_calls_with_nav"] += 1
        return out

    RR.Refcv6Proxy.__call__ = wrapped
    return orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--windows-json", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0, help="inference seed (banked s0 = 0)")
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    import run_battery_r7 as RB
    import r7_roll as RR
    assert RB.RR is RR, "run_battery_r7.RR is not the r7_roll module this script patched"
    _install_capture(RR)
    windows = json.load(open(a.windows_json, encoding="utf-8"))
    t0 = time.time()
    rec = RB.roll_one(a.ckpt, a.config, a.seed, a.dump_dir, windows, None, device=a.device,
                      full=False)
    side = np.load(str(Path(a.dump_dir) / "refcv6_extras.npz"), allow_pickle=True)
    keys = sorted(FAN)
    if len(keys) != len(side["ws"]):
        raise SystemExit("[fan_roll] FAN rows %d != sidecar windows %d" % (len(keys), len(side["ws"])))
    ci = np.asarray([k[0] for k in keys], np.int64)
    if not np.array_equal(ci, side["clip_index"]):
        raise SystemExit("[fan_roll] fan clip_index order != sidecar clip_index order")
    out = {"clip_index": ci, "ws": np.asarray(side["ws"], np.int64),
           "t": np.asarray([k[1] for k in keys], np.int64)}
    for k in FAN_KEYS:
        if all(k in FAN[kk] for kk in keys):
            out[k] = np.stack([FAN[kk][k] for kk in keys]).astype(np.float32)
    tr_ok = bool(np.array_equal(out["traj"], side["nav_true.traj"].astype(np.float32)))
    np.savez_compressed(str(Path(a.dump_dir) / "fan_extras.npz"), **out)
    meta = {"n_windows": int(len(keys)), "fan_shape": list(out["anchor_traj"].shape),
            "keys_saved": sorted(out), "conds_seen": sorted(map(list, STATS["conds_seen"])),
            "traj_equals_sidecar_nav_true": tr_ok, "inference_seed": a.seed,
            "wall_s": round(time.time() - t0, 1),
            "roll_record": {k: rec.get(k) for k in ("n_windows", "n_episodes", "skipped", "wall_s",
                                                    "cuda_max_memory_allocated_gib", "arm_set")},
            "ckpt": a.ckpt, "config": a.config, "windows_json": a.windows_json,
            "producer": "battery/code/fan_roll_r7.py (wrapper over run_battery_r7.roll_one, full=False)"}
    json.dump(meta, open(a.out_json, "w", encoding="utf-8"), indent=1, default=str)
    print("[fan_roll] %d windows, fan %s, traj==sidecar %s -> %s" % (
        len(keys), out["anchor_traj"].shape, tr_ok, a.out_json))
    if not tr_ok:
        raise SystemExit("[fan_roll] traj != sidecar nav_true.traj (capture is NOT the banked call)")


if __name__ == "__main__":
    main()
