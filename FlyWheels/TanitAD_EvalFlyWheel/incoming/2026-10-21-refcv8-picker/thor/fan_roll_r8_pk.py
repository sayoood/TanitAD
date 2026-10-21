"""fan_roll_r8_pk.py -- PREREG R8-PICKER (sha256 e6ec30e7..., sealed 2026-10-21 21:01:51 +0200) roll-time capture.

EXTENDS fan_roll_r8_pf.py (D-REFCV8-FANCEIL, sha256 f1c80d5e...), the same WRAPPER pattern. It edits no file. It
  1. (optional, Thor) re-points `r8_repoint.REMAP_OVERRIDES` at a Thor remap JSON BEFORE anything reads it (a PATH);
  2. imports `roll_seed_r8_pf` as a module (overlay check, apply_child, the perframe_full argv append -- all UNCHANGED);
  3. replaces `r7_roll.Refcv6Proxy.__call__` with a function that calls the original UNCHANGED (same RNG stream, same model
     call, same sidecar rows) and only then COPIES the `nav_true` row of the keys below out of the returned dict, plus the
     model's own E9 admissible set `e9_keep = isfinite(refc_v3.e9_rank(zeros_like(sel_score_v3), out))` -- the model's own
     function on the model's own masks (reach_keep AND ceil_keep AND r8_emit_keep, with its dead-row fallbacks). e9_rank only
     READS `out`; the forward is not changed;
  4. calls `roll_seed_r8_pf.main()` with sys.argv = EXACTLY the battery's child argv for that seed (+ --os-only for the
     seeds after the first, as run_battery_r7._run does; R0 = seed 0 full arm, R1 = seed 1 --os-only).
A call whose conds carry no `nav_true` is passed through uncaptured and COUNTED (none is expected: the outer calls are the
nav batch [nav_true(, nav_shuffled, nav_flipped)] and the nav_zero call, which has nav_cmd None).

After the roll it asserts on the ARTIFACT: the captured rows align 1:1 with the sidecar the roll wrote (clip_index order), the
captured `traj` equals the sidecar's `nav_true.traj` exactly (K3b), and `anchor_traj[sel_idx] == traj` exactly (K3a).
Output: <dump-dir>/fan_extras.npz (floats fp32, masks bool, indices int64) + --fan-meta-json.

    python fan_roll_r8_pk.py --pf-code <hold_r8pf/code> [--remap-json <thor remap>] --ckpt CK --config CFG --seed S \
        --dump-dir D --windows-json W --out-json RJ --fan-meta-json M [--os-only] [--device cuda|cpu] [--smoke-cpu-fp32-trunk]
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:      # noqa: BLE001
    pass

FLOAT_KEYS = ("anchor_traj", "anchor_logits", "refined_logits", "sel_score", "sel_score_v3", "traj", "traj_base", "goal_dist")
BOOL_KEYS = ("reach_keep", "ceil_keep", "r8_emit_keep")
INT_KEYS = ("sel_idx", "sel_idx_base", "r8_cand_src")
FAN: dict = {}
STATS = {"n_calls_with_nav": 0, "conds_seen": set(), "n_calls_navzero": 0, "n_calls_without_nav_true": 0, "keys_missing": {}}


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _install_capture(RR, torch, RV3):
    orig = RR.Refcv6Proxy.__call__

    def wrapped(self, frames, nav_cmd=None, v0=None, steps=0, ego_state=None, **kw):
        out = orig(self, frames, nav_cmd=nav_cmd, v0=v0, steps=steps, ego_state=ego_state, **kw)
        if nav_cmd is None:
            STATS["n_calls_navzero"] += 1
            return out
        conds = list(self.conds_fed[:int(frames.shape[0])])
        STATS["conds_seen"].add(tuple(conds))
        if "nav_true" not in conds:
            STATS["n_calls_without_nav_true"] += 1
            return out
        r = conds.index("nav_true")
        row = {}
        with torch.no_grad():
            for k in FLOAT_KEYS + BOOL_KEYS + INT_KEYS:
                v = out.get(k) if hasattr(out, "get") else None
                if v is None or not torch.is_tensor(v):
                    STATS["keys_missing"][k] = STATS["keys_missing"].get(k, 0) + 1
                    continue
                x = v[r].detach()
                if k in FLOAT_KEYS:
                    row[k] = x.float().cpu().numpy()
                elif k in BOOL_KEYS:
                    row[k] = x.bool().cpu().numpy()
                else:
                    row[k] = x.long().cpu().numpy()
            if "sel_score_v3" in out and torch.is_tensor(out["sel_score_v3"]):
                rk, _tele = RV3.e9_rank(torch.zeros_like(out["sel_score_v3"]), out)      # the model's OWN mask function, READ-ONLY
                row["e9_keep"] = torch.isfinite(rk[r]).cpu().numpy()
                rk2, _ = RV3.e9_rank(out["sel_score_v3"], out)                             # K4 at roll time (same device)
                row["e9_argmax"] = rk2[r].argmax().long().cpu().numpy()
        if "anchor_traj" not in row or "e9_keep" not in row:
            raise RuntimeError("[fan_roll_r8_pk] model output carries no anchor_traj / sel_score_v3")
        if self.ctx.index in FAN:
            raise RuntimeError("[fan_roll_r8_pk] duplicate nav call for window %s" % (self.ctx.index,))
        FAN[self.ctx.index] = row
        STATS["n_calls_with_nav"] += 1
        return out

    RR.Refcv6Proxy.__call__ = wrapped
    return orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pf-code", required=True, help="hold_r8pf .../2026-10-20-refcv8-perframe-final/code")
    ap.add_argument("--remap-json", default=None, help="Thor only: the md5-equal Thor remap (dev box: leave unset)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--windows-json", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--fan-meta-json", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--os-only", action="store_true")
    ap.add_argument("--smoke-cpu-fp32-trunk", action="store_true", help="SMOKE ONLY (passed to the child unchanged)")
    a = ap.parse_args()
    pf_code = Path(a.pf_code)
    r8_code = Path(os.environ.get("PF_R8_CODE", "C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/"
                                                "2026-10-15-refcv8-milestone-battery/code"))
    sys.path.insert(0, str(r8_code))
    import r8_repoint  # noqa: E402
    dev_remap = r8_repoint.REMAP_OVERRIDES
    if a.remap_json:
        r8_repoint.REMAP_OVERRIDES = Path(a.remap_json)
    child = [str(pf_code / "roll_seed_r8_pf.py"), "--ckpt", a.ckpt, "--config", a.config, "--seed", str(a.seed),
             "--dump-dir", a.dump_dir, "--windows-json", a.windows_json, "--device", a.device, "--out-json", a.out_json]
    if a.smoke_cpu_fp32_trunk:
        child.append("--smoke-cpu-fp32-trunk")
    if a.os_only:
        child.append("--os-only")
    sys.argv = child
    sys.path.insert(0, str(pf_code))
    import roll_seed_r8_pf as RP  # noqa: E402  (module-level: overlay check + apply_child + navfeed install)
    import r7_roll as RR  # noqa: E402
    import torch  # noqa: E402
    import numpy as np  # noqa: E402
    from tanitad.refs import refc_v3 as RV3  # noqa: E402  (the launch tree's model module, the class the roll builds)
    assert sys.modules["r7_roll"] is RR
    _install_capture(RR, torch, RV3)
    t0 = time.time()
    if a.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    rc = RP.main()
    wall = time.time() - t0
    side = np.load(str(Path(a.dump_dir) / "refcv6_extras.npz"), allow_pickle=True)
    keys = sorted(FAN)
    if len(keys) != len(side["ws"]):
        raise SystemExit("[fan_roll_r8_pk] FAN rows %d != sidecar windows %d" % (len(keys), len(side["ws"])))
    ci = np.asarray([k[0] for k in keys], np.int64)
    if not np.array_equal(ci, side["clip_index"]):
        raise SystemExit("[fan_roll_r8_pk] fan clip_index order != sidecar clip_index order")
    out = {"clip_index": ci, "ws": np.asarray(side["ws"], np.int64), "t": np.asarray([k[1] for k in keys], np.int64)}
    for k in FLOAT_KEYS + BOOL_KEYS + INT_KEYS + ("e9_keep", "e9_argmax"):
        if all(k in FAN[kk] for kk in keys):
            v = np.stack([FAN[kk][k] for kk in keys])
            out[k] = v.astype(np.float32) if k in FLOAT_KEYS else (v.astype(bool) if k in BOOL_KEYS + ("e9_keep",) else v.astype(np.int64))
    k3b = bool(np.array_equal(out["traj"], side["nav_true.traj"].astype(np.float32)))
    sel = out["sel_idx"].reshape(-1)
    k3a = float(np.abs(out["anchor_traj"][np.arange(len(sel)), sel] - out["traj"]).max())
    k4 = float((out["e9_argmax"].reshape(-1) == sel).mean()) if "e9_argmax" in out else None
    np.savez_compressed(str(Path(a.dump_dir) / "fan_extras.npz"), **out)
    meta = {"n_windows": int(len(keys)), "fan_shape": list(out["anchor_traj"].shape),
            "keys_saved": sorted(out), "keys_missing_counts": STATS["keys_missing"],
            "conds_seen": sorted(map(list, STATS["conds_seen"])), "n_calls_navzero": STATS["n_calls_navzero"],
            "n_calls_without_nav_true": STATS["n_calls_without_nav_true"],
            "K3a_max_abs_fan_at_sel_minus_traj_m": k3a, "K3b_traj_equals_sidecar_nav_true": k3b,
            "K4_rolltime_e9_argmax_equals_sel_idx_frac": k4,
            "e9_keep_frac_mean": float(out["e9_keep"].mean()), "e9_keep_min_per_row": int(out["e9_keep"].sum(1).min()),
            "ceil_keep_frac_mean": (float(out["ceil_keep"].mean()) if "ceil_keep" in out else None),
            "inference_seed": a.seed, "os_only": bool(a.os_only), "smoke_cpu_fp32_trunk": bool(a.smoke_cpu_fp32_trunk),
            "child_rc": rc, "child_argv": child, "wall_s": round(wall, 1),
            "cuda_max_memory_allocated_gib": (round(torch.cuda.max_memory_allocated() / 2**30, 3)
                                              if (a.device.startswith("cuda") and torch.cuda.is_initialized()) else None),
            "remap_overrides": {"file_used": str(r8_repoint.REMAP_OVERRIDES), "default_file": str(dev_remap),
                                "repointed": bool(a.remap_json)},
            "env": {k: os.environ.get(k) for k in ("R8_LAUNCH_TREE", "R8_EVAL_KIT", "PF_R8_CODE", "PF_NF_HOLD",
                                                    "R8_NO_POSE_SYNC", "OMP_NUM_THREADS", "PYTHONPATH")},
            "torch": torch.__version__, "host": (os.uname().nodename if hasattr(os, "uname") else os.environ.get("COMPUTERNAME")),
            "ckpt": a.ckpt, "config": a.config, "windows_json": a.windows_json,
            "this_script_sha256": sha256_file(__file__), "refc_v3_file": RV3.__file__, "refc_v3_sha256": sha256_file(RV3.__file__),
            "prereg": "PREREG_R8_PICKER.md sha256 e6ec30e7bc03b3dc4811674352f94c9f128203cbf814f284876f239ae867b258",
            "producer": "fan_roll_r8_pk.py (wrapper over roll_seed_r8_pf.main, capture on r7_roll.Refcv6Proxy.__call__)"}
    json.dump(meta, open(a.fan_meta_json, "w", encoding="utf-8"), indent=1, default=str)
    ok = bool(k3b and k3a == 0.0 and rc in (0, None))
    print("[fan_roll_r8_pk] %d windows fan %s K3a %.3g K3b %s K4 %s rc=%s -> %s %s" % (
        len(keys), out["anchor_traj"].shape, k3a, k3b, k4, rc, a.fan_meta_json,
        "ZZPKROLLOKZZ" if ok else "ZZPKROLLBADZZ"), flush=True)
    if not ok:
        raise SystemExit("[fan_roll_r8_pk] K3 failed or the child returned %s (capture is NOT the banked call)" % (rc,))
    return 0


if __name__ == "__main__":
    sys.exit(main())
