"""fan_roll_r8_pf.py -- D-REFCV8-FANCEIL: dump refcv8's full candidate FAN (149 = 117 base + 32 extra), one row per
window, on the SAME windows, SAME checkpoint, SAME code path and SAME per-frame nav feed (perframe_full) as the banked
per-frame battery's `os` arm (step48852ema_pf).

WHY: the battery's decisions sidecar keeps sel_idx, the terminal heading and the reach mask of every candidate, but NOT the
candidate POSITIONS, so oracle-in-fan / random-pick cannot be computed from the bank.

HOW: a WRAPPER, exactly the fan_roll_r7.py pattern (refcv7 X8). It edits no file. It
  1. re-points `r8_repoint.REMAP_OVERRIDES` at a Thor remap JSON (the dev-box JSON's seven flags mapped to the md5-equal
     Thor files) BEFORE anything reads it -- the only departure from the dev-box child, and a PATH, not a number;
  2. imports `roll_seed_r8_pf` as a module (its module-level code = the registered nav-feed overlay check, the r8
     re-pointing, the overlay install, the --nav-feed perframe_full argv append -- all UNCHANGED);
  3. replaces `r7_roll.Refcv6Proxy.__call__` with a function that calls the original UNCHANGED (same RNG stream, same
     model call, same sidecar rows) and only then copies the `nav_true` row of the fan keys out of the returned dict;
  4. calls `roll_seed_r8_pf.main()` with sys.argv set to EXACTLY the battery's child argv for that seed.
After the roll it asserts the fan rows align 1:1 with the sidecar the roll itself wrote and that the captured `traj`
equals the sidecar's `nav_true.traj` exactly. All 8 model slots are kept (0.5,1,1.5,2,3,4,5,6 s), fp32 on disk.

    python fan_roll_r8_pf.py --pf-code <hold_r8pf/code> --remap-json <thor remap> --ckpt CK --config CFG --seed 1 \
        --dump-dir D --windows-json W --out-json RJ --fan-meta-json M [--os-only]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:      # noqa: BLE001
    pass

FAN_KEYS = ("anchor_traj", "anchor_logits", "refined_logits", "sel_score", "sel_idx", "traj", "reach_keep")
FAN: dict = {}
STATS = {"n_calls_with_nav": 0, "conds_seen": set(), "n_calls_navzero": 0}


def _install_capture(RR, torch):
    orig = RR.Refcv6Proxy.__call__

    def wrapped(self, frames, nav_cmd=None, v0=None, steps=0, ego_state=None, **kw):
        out = orig(self, frames, nav_cmd=nav_cmd, v0=v0, steps=steps, ego_state=ego_state, **kw)
        if nav_cmd is None:
            STATS["n_calls_navzero"] += 1
            return out
        conds = list(self.conds_fed[:int(frames.shape[0])])
        STATS["conds_seen"].add(tuple(conds))
        if "nav_true" not in conds:
            raise RuntimeError("[fan_roll_r8] nav_true not among conds %s" % (conds,))
        r = conds.index("nav_true")
        row = {}
        for k in FAN_KEYS:
            v = out.get(k) if hasattr(out, "get") else None
            if v is None or not torch.is_tensor(v):
                continue
            row[k] = v[r].detach().float().cpu().numpy()
        if "anchor_traj" not in row:
            raise RuntimeError("[fan_roll_r8] model output carries no anchor_traj")
        if self.ctx.index in FAN:
            raise RuntimeError("[fan_roll_r8] duplicate nav call for window %s" % (self.ctx.index,))
        FAN[self.ctx.index] = row
        STATS["n_calls_with_nav"] += 1
        return out

    RR.Refcv6Proxy.__call__ = wrapped
    return orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pf-code", required=True, help="hold_r8pf .../2026-10-20-refcv8-perframe-final/code")
    ap.add_argument("--remap-json", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--windows-json", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--fan-meta-json", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--os-only", action="store_true")
    a = ap.parse_args()
    pf_code = Path(a.pf_code)
    r8_code = Path(os.environ["PF_R8_CODE"])
    # 1. the Thor remap (a PATH re-point; recorded)
    sys.path.insert(0, str(r8_code))
    import r8_repoint  # noqa: E402
    dev_remap = r8_repoint.REMAP_OVERRIDES
    r8_repoint.REMAP_OVERRIDES = Path(a.remap_json)
    # 2. the battery child argv, EXACTLY (run_battery_r7._run: --ckpt --config --seed --dump-dir --windows-json --device
    #    --out-json [+ --os-only for every seed after the first])
    child = [str(pf_code / "roll_seed_r8_pf.py"), "--ckpt", a.ckpt, "--config", a.config, "--seed", str(a.seed),
             "--dump-dir", a.dump_dir, "--windows-json", a.windows_json, "--device", a.device, "--out-json", a.out_json]
    if a.os_only:
        child.append("--os-only")
    sys.argv = child
    sys.path.insert(0, str(pf_code))
    import roll_seed_r8_pf as RP  # noqa: E402  (module-level: overlay check + apply_child + navfeed install)
    import r7_roll as RR  # noqa: E402
    import torch  # noqa: E402
    import numpy as np  # noqa: E402
    assert sys.modules["r7_roll"] is RR
    _install_capture(RR, torch)
    t0 = time.time()
    if a.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    rc = RP.main()
    wall = time.time() - t0
    side = np.load(str(Path(a.dump_dir) / "refcv6_extras.npz"), allow_pickle=True)
    keys = sorted(FAN)
    if len(keys) != len(side["ws"]):
        raise SystemExit("[fan_roll_r8] FAN rows %d != sidecar windows %d" % (len(keys), len(side["ws"])))
    ci = np.asarray([k[0] for k in keys], np.int64)
    if not np.array_equal(ci, side["clip_index"]):
        raise SystemExit("[fan_roll_r8] fan clip_index order != sidecar clip_index order")
    out = {"clip_index": ci, "ws": np.asarray(side["ws"], np.int64),
           "t": np.asarray([k[1] for k in keys], np.int64)}
    for k in FAN_KEYS:
        if all(k in FAN[kk] for kk in keys):
            out[k] = np.stack([FAN[kk][k] for kk in keys]).astype(np.float32)
    tr_ok = bool(np.array_equal(out["traj"], side["nav_true.traj"].astype(np.float32)))
    np.savez_compressed(str(Path(a.dump_dir) / "fan_extras.npz"), **out)
    meta = {"n_windows": int(len(keys)), "fan_shape": list(out["anchor_traj"].shape),
            "keys_saved": sorted(out), "conds_seen": sorted(map(list, STATS["conds_seen"])),
            "n_calls_navzero": STATS["n_calls_navzero"],
            "traj_equals_sidecar_nav_true": tr_ok, "inference_seed": a.seed, "os_only": bool(a.os_only),
            "child_rc": rc, "child_argv": child, "wall_s": round(wall, 1),
            "cuda_max_memory_allocated_gib": (round(torch.cuda.max_memory_allocated() / 2**30, 3)
                                              if torch.cuda.is_initialized() else None),
            "remap_overrides": {"thor_file": a.remap_json, "devbox_file_replaced": str(dev_remap),
                                "thor_map": json.load(open(a.remap_json, encoding="utf-8"))},
            "env": {k: os.environ.get(k) for k in ("R8_LAUNCH_TREE", "R8_EVAL_KIT", "PF_R8_CODE", "PF_NF_HOLD",
                                                    "R8_NO_POSE_SYNC", "OMP_NUM_THREADS", "PYTHONPATH")},
            "torch": torch.__version__, "host": os.uname().nodename if hasattr(os, "uname") else None,
            "ckpt": a.ckpt, "config": a.config, "windows_json": a.windows_json,
            "producer": "fan_roll_r8_pf.py (wrapper over roll_seed_r8_pf.main, capture on r7_roll.Refcv6Proxy.__call__)"}
    json.dump(meta, open(a.fan_meta_json, "w", encoding="utf-8"), indent=1, default=str)
    print("[fan_roll_r8] %d windows, fan %s, traj==sidecar %s rc=%s -> %s ZZFANROLLDONEZZ" % (
        len(keys), out["anchor_traj"].shape, tr_ok, rc, a.fan_meta_json), flush=True)
    if not tr_ok:
        raise SystemExit("[fan_roll_r8] traj != sidecar nav_true.traj (capture is NOT the banked call)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
