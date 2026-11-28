"""mapsel_roll.py -- refcv9 MAP-SELECT (PREREG_MAPSEL.md): ONE inference-seed os-only NATIVE roll of the shipped refcv8 EMA-48,852 at PER-FRAME nav (no hook, no force) that BANKS, per window,
(1) everything L2-B banked (the 149-candidate fan, sel_score(_v3), the E9 eligibility masks, plan8, sel_idx, the unrefined bank, the residual prior) and
(2) the map RASTERS the selection rules read, at 10 cm in the ego frame (1000 x 600): the model's OWN planes -- P^(drivable set) uint8 (liveness only), P^(edge) uint8, P^(sidewalk) uint8 (the AMENDED T1 rule), the decided class code uint8, under the DECLARED
    prior-corrected rule with the model's own frozen class weights -- and the SAM3 GT code plane uint8 (255 = not seen; absent for the clips with no GT file).
Rasters go to `--rast-dir/e{clip_index:03d}_w{ws:04d}.npz` (compressed); the small per-window arrays to `--trace-out`.  NO per-candidate map feature is computed here: that is the CPU scorer's job.
Adapted from `l2b_locus_roll.py` (same battery child / remap / Refcv6Proxy subclass) and `p4_roll.py` (the GT attach `_wrap_build_eval_dataset`, the trainer's own `enable_map_hires`).

    R8_LAUNCH_TREE=<tree_fb47a58> PYTHONPATH='<tree>/stack;<tree>/taniteval' R8_NO_POSE_SYNC=1 OMP_NUM_THREADS=4 \
      python -X utf8 mapsel_roll.py --trace-out F:/refcv9_mapsel/trace_s0.npz --rast-dir F:/refcv9_mapsel/rast --map-gt-root E:/refcv6_eval_kit/data/sam3_gt_v3_eval --require-seal -- \
         --ckpt E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt --config E:/refcv8_eval_kit/ckpt/config.json --seed 0 --dump-dir <dir> --windows-json <windows_s0.json> --device cuda --out-json <roll.json> --os-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PREREG, SEAL = ROOT / "PREREG_MAPSEL.md", ROOT / "PREREG_MAPSEL.sha256"
PREREG2, SEAL2 = ROOT / "PREREG_MAPSEL_AMENDMENT1.md", ROOT / "PREREG_MAPSEL_AMENDMENT1.sha256"
PF_CODE = Path(os.environ.get("L2_PF_CODE", "C:/Users/Admin/qland/hold_r8pf/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/"
                                            "2026-10-20-refcv8-perframe-final/code"))
R8_CODE = Path(os.environ.get("L2_R8_CODE", "C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/"
                                            "2026-10-15-refcv8-milestone-battery/code"))
EVAL_REMAP = HERE / "refcv8_eval_remap_devbox_mapsel.json"
DRV_CODES = (1, 2, 3, 4, 6)
EDGE_CODE = 5
SIDEWALK_CODE = 7

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:      # noqa: BLE001
    pass


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def parse():
    if "--" not in sys.argv:
        raise SystemExit("[mapsel] usage: mapsel_roll.py [opts] -- <battery child argv>")
    i = sys.argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-out", required=True)
    ap.add_argument("--rast-dir", required=True)
    ap.add_argument("--map-gt-root", required=True)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--require-seal", action="store_true")
    ap.add_argument("--budget-probe-windows", type=int, default=0, help="after N windows project the total wall (marker)")
    ap.add_argument("--budget-max-s", type=float, default=0.0, help="stop (exit 7) if the projected total wall exceeds this")
    ap.add_argument("--projected-windows", type=int, default=4754)
    a = ap.parse_args(sys.argv[1:i])
    child = sys.argv[i + 1:]
    if "--os-only" not in child:
        raise SystemExit("[mapsel] the os-only battery child is required; --os-only missing")
    return a, child


A, CHILD = parse()
SEAL_SHA = None
if SEAL.exists() and PREREG.exists() and SEAL2.exists() and PREREG2.exists():
    ok1 = SEAL.read_text(encoding="utf-8").split()[0].strip() == sha256_file(PREREG)
    ok2 = SEAL2.read_text(encoding="utf-8").split()[0].strip() == sha256_file(PREREG2)
    SEAL_SHA = (sha256_file(PREREG)[:16] + "+" + sha256_file(PREREG2)[:16]) if (ok1 and ok2) else None
if A.require_seal and SEAL_SHA is None:
    raise SystemExit("[mapsel] PREREG_MAPSEL.md / PREREG_MAPSEL_AMENDMENT1.md is missing / does not match its seal: refusing a scored roll")

sys.path.insert(0, str(R8_CODE))
import r8_repoint  # noqa: E402
REMAP_REC = {"devbox_file_replaced": str(r8_repoint.REMAP_OVERRIDES), "launch_tree": str(r8_repoint.LAUNCH_TREE), "file": str(EVAL_REMAP), "sha256": sha256_file(EVAL_REMAP)}
r8_repoint.REMAP_OVERRIDES = EVAL_REMAP
sys.argv = [str(PF_CODE / "roll_seed_r8_pf.py")] + CHILD
sys.path.insert(0, str(PF_CODE))
import roll_seed_r8_pf as PF  # noqa: E402  (READ-ONLY, UNCHANGED)
RR = PF.RR
RUNNER = PF.main

import torch  # noqa: E402

_Base = RR.Refcv6Proxy
RAST = Path(A.rast_dir)
ST = {"W": None, "t0": time.time(), "records": [], "present": {}, "n_stack": None,
      "c": {"calls": 0, "batch_gt1": 0, "navnull_calls": 0, "fan_n_seen": set(), "no_perception": 0, "decide_checked": 0, "decide_mismatch_cells": 0,
            "gt_windows": 0, "nogt_windows": 0, "rast_bytes": 0, "logit_shapes": set(), "rule_seen": set()}}
FIELDS = (("fan", "anchor_traj", np.float32), ("sel_score", "sel_score", np.float32), ("sel_score_v3", "sel_score_v3", np.float32),
          ("reach_keep", "reach_keep", np.int8), ("ceil_keep", "ceil_keep", np.int8), ("emit_keep", "r8_emit_keep", np.int8),
          ("kind", "r8_kind", np.int8), ("lon6", "r8_lon6", np.int8), ("lat3", "r8_lat3", np.int8), ("cons_raw", "r8_cons_raw", np.float32),
          ("hyp_alloc", "r8_hyp_alloc", np.int8), ("prior_path", "residual_prior_path", np.float32), ("bank", "anchor_bank", np.float16))


def _wrap_build_eval_dataset(L, map_root):
    """The p4_roll / l7cap pattern: the roll's OWN build (perception targets OFF), then ONLY the 10 cm GT through the trainer's `enable_map_hires` -> every item carries `map_fine`
    (= _map_fine_item(ep, t+w-1)) and `map_fine_label`.  The window index must stay unchanged."""
    prev = L.build_eval_dataset

    def bed(model, cfg, args, config, *, with_perception_targets=True, dataset_cls=None):
        e_ds, e_eps, rec = prev(model, cfg, args, config, with_perception_targets=with_perception_targets, dataset_cls=dataset_cls)
        n0 = len(e_ds.index)
        tr = L.trainer()
        _clip, _ns = tr._clip_table_for_caches([args.eval_cache])
        ext = tr._mhr.declared_extent(args)
        store = tr._sem_fine.FineMapGTStore(Path(map_root), max_open=4, extent=ext)
        rec["mapsel_map_fine"] = e_ds.enable_map_hires(store, _clip, _ns, min_coverage=None)
        if len(e_ds.index) != n0:
            raise SystemExit("[mapsel] attaching the map GT changed the window index (%d -> %d): refusing" % (n0, len(e_ds.index)))
        ST["n_stack"] = int(_ns)
        rec["mapsel_label_side"] = {"map_gt_root": str(map_root), "index_unchanged": True, "n_stack": int(_ns), "extent": [float(ext.x_max_m), float(ext.y_half_m)]}
        return e_ds, e_eps, rec

    L.build_eval_dataset = bed


def _window_len(model):
    core = getattr(getattr(model, "cfg", None), "core", None)
    if core is None or not hasattr(core, "window"):
        for _n, m in model.named_modules():
            core = getattr(getattr(m, "cfg", None), "core", None)
            if core is not None and hasattr(core, "window"):
                break
    if core is None or not hasattr(core, "window"):
        raise SystemExit("[mapsel] cannot read the window length W from the model config")
    return int(core.window)


def _row(o, key, r, B):
    t = o[key]
    if not torch.is_tensor(t):
        return None
    t = t.detach()
    if t.dim() >= 1 and t.shape[0] == B:
        t = t[r]
    return t.float().cpu().numpy() if t.is_floating_point() else t.cpu().numpy()


def own_planes(model, logits_r):
    """logits_r [1, 8, H, W] -> (pdrv_u8, pedge_u8, psw_u8, code_u8) numpy [H, W], under the model's DECLARED rule: p^ = softmax(z - log w) (w = the loss's frozen class weights),
    code = argmax(z - log w) (== map_head_hires.decide('prior_corrected', w), asserted on a sample), pdrv = sum of p^ over the DRIVABLE codes."""
    from tanitad.models import map_head_hires as MH
    cw = model._map_hires_class_weight
    rule = str(model._map_hires.cfg.decision_rule)
    ST["c"]["rule_seen"].add(rule)
    if rule != "prior_corrected" or cw is None:
        raise RuntimeError(f"[mapsel] the declared decision rule is {rule!r} / class weights {'absent' if cw is None else 'present'}: this arm is defined for prior_corrected + weights")
    z = logits_r.detach().float()
    s = z - MH._decision_logw(cw, z).view(1, -1, 1, 1)
    p = torch.softmax(s, 1)[0]
    code = s.argmax(1)[0]
    if ST["c"]["decide_checked"] < 25 or ST["c"]["calls"] % 100 == 0:
        ref = MH.decide(z, rule, cw)[0]
        ST["c"]["decide_checked"] += 1
        ST["c"]["decide_mismatch_cells"] += int((ref != code).sum())
    pdrv = p[list(DRV_CODES)].sum(0)
    q = lambda t: t.mul(255.0).round().clamp_(0, 255).to(torch.uint8).cpu().numpy()   # noqa: E731
    return q(pdrv), q(p[EDGE_CODE]), q(p[SIDEWALK_CODE]), code.to(torch.uint8).cpu().numpy()


class MapSelProxy(_Base):
    def __call__(self, frames, nav_cmd=None, v0=None, steps=0, ego_state=None, **kw):
        d = self.__dict__
        model = d["model"]
        if ST["W"] is None:
            ST["W"] = _window_len(model)
        if nav_cmd is None:
            ST["c"]["navnull_calls"] += 1
            return _Base.__call__(self, frames, nav_cmd=nav_cmd, v0=v0, steps=steps, ego_state=ego_state, **dict(kw))
        n = int(frames.shape[0])
        e_i, t = (int(x) for x in d["ctx"].index)
        ws = t + int(ST["W"]) - 1
        out = _Base.__call__(self, frames, nav_cmd=nav_cmd, v0=v0, steps=steps, ego_state=ego_state, **dict(kw))
        ST["c"]["calls"] += 1
        if n != 1:
            ST["c"]["batch_gt1"] += 1
        conds = list(d["conds_fed"][:n])
        if "nav_true" not in conds:
            raise RuntimeError(f"[mapsel] no nav_true row among conds {conds}")
        r = conds.index("nav_true")
        rec = {"e_i": e_i, "t": t, "ws": ws, "plan": out["traj"][r].detach().float().cpu().numpy(), "sel": int(out["sel_idx"][r]),
               "sel_base": (int(out["sel_idx_base"][r]) if "sel_idx_base" in out else -1),
               "v0_in": (float(v0.reshape(-1)[r]) if v0 is not None else float("nan"))}
        for name, key, dt in FIELDS:
            if key in out and torch.is_tensor(out[key]):
                rec[name] = _row(out, key, r, n).astype(dt)
                ST["present"][name] = ST["present"].get(name, 0) + 1
            else:
                rec[name] = None
        if rec["fan"] is not None:
            ST["c"]["fan_n_seen"].add(int(rec["fan"].shape[0]))
        # ---- the rasters
        lg = (out.get("perception") or {}).get("map_hires_logits")
        if lg is None:
            ST["c"]["no_perception"] += 1
            raise RuntimeError("[mapsel] out['perception']['map_hires_logits'] is absent: the 10 cm map head did not run (--map-hires on is required)")
        ST["c"]["logit_shapes"].add(tuple(int(x) for x in lg.shape))
        pdrv, pedge, psw, code = own_planes(model, lg[r:r + 1])
        it = d["ctx"].item
        gt_label = bool(it.get("map_fine_label", torch.tensor(False))) if "map_fine" in it else False
        arrs = {"pdrv": pdrv, "pedge": pedge, "psw": psw, "code": code}
        acc = float("nan")
        if gt_label:
            gt = np.asarray(it["map_fine"].numpy() if torch.is_tensor(it["map_fine"]) else it["map_fine"], np.uint8)
            arrs["gt"] = gt
            seen = gt != 255
            if seen.any():                                   # liveness: the own drivable decision vs the GT drivable on SEEN cells (descriptive; the smoke gate reads it)
                gdrv = np.isin(gt, DRV_CODES)
                acc = float(((pdrv >= 128) == gdrv)[seen].mean())
            ST["c"]["gt_windows"] += 1
        else:
            ST["c"]["nogt_windows"] += 1
        RAST.mkdir(parents=True, exist_ok=True)
        fp = RAST / f"e{e_i:03d}_w{ws:04d}.npz"
        tmp = fp.with_name(fp.stem + ".tmp.npz")
        np.savez_compressed(tmp, **arrs)
        ST["c"]["rast_bytes"] += os.path.getsize(tmp)
        os.replace(tmp, fp)
        rec["gt_label"], rec["acc_own"] = gt_label, acc
        ST["records"].append(rec)
        nw = len(ST["records"])
        if nw % int(A.save_every) == 0:
            save(False)
        if nw % 250 == 0 or nw <= 2:
            print(f"ZZMSPROGZZ {nw} el={time.time() - ST['t0']:.0f}s rast_mb={ST['c']['rast_bytes'] / 2**20:.0f}", flush=True)
        if A.budget_probe_windows and nw == int(A.budget_probe_windows):
            el = time.time() - ST["t0"]
            proj = el / nw * int(A.projected_windows)
            print(f"ZZMSBUDGETZZ windows={nw} elapsed_s={el:.0f} projected_total_s={proj:.0f} budget_s={A.budget_max_s:.0f}", flush=True)
            if A.budget_max_s and proj > float(A.budget_max_s):
                save(False, {"over_budget": {"projected_total_s": proj, "budget_s": A.budget_max_s}})
                print("ZZMSOVERBUDGETZZ stopping before the budget is spent", flush=True)
                raise SystemExit(7)
        return out


def _stack(recs, name, fill, dtype):
    got = [r[name] for r in recs]
    ref = next((g for g in got if g is not None), None)
    if ref is None:
        return None
    return np.stack([g if g is not None else np.full(ref.shape, fill, dtype) for g in got]).astype(dtype)


def save(final: bool, extra: dict | None = None):
    recs = ST["records"]
    if not recs:
        return
    out = {"e_i": np.array([r["e_i"] for r in recs], np.int64), "t": np.array([r["t"] for r in recs], np.int64), "ws": np.array([r["ws"] for r in recs], np.int64),
           "plan8": np.stack([r["plan"] for r in recs]).astype(np.float32), "sel": np.array([r["sel"] for r in recs], np.int64),
           "sel_base": np.array([r["sel_base"] for r in recs], np.int64), "v0_in": np.array([r["v0_in"] for r in recs], np.float64),
           "gt_label": np.array([bool(r["gt_label"]) for r in recs]), "acc_own": np.array([r["acc_own"] for r in recs], np.float64)}
    for name, _key, dt in FIELDS:
        fill = -1 if name in ("reach_keep", "ceil_keep", "emit_keep") else 0
        arr = _stack(recs, name, fill, dt)
        if arr is not None:
            out[name] = arr
    meta = {"tool": "mapsel_roll.py", "prereg_sha256": SEAL_SHA, "script_sha256": sha256_file(__file__), "child_argv": CHILD, "remap": REMAP_REC, "rast_dir": str(RAST),
            "map_gt_root": A.map_gt_root, "window_W": ST["W"], "n_stack": ST["n_stack"], "final": bool(final), "n_windows": len(recs), "present_counts": ST["present"],
            "counters": {k: (sorted(v) if isinstance(v, set) else v) for k, v in ST["c"].items()}, "wall_s": round(time.time() - ST["t0"], 1),
            "saved": time.strftime("%FT%T%z"), **(extra or {})}
    out["meta_json"] = np.array(json.dumps(meta, default=str))
    p = Path(A.trace_out)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.stem + ".tmp.npz")
    np.savez_compressed(tmp, **out)
    os.replace(tmp, p)


def main() -> int:
    _wrap_build_eval_dataset(RR.L, A.map_gt_root)
    RR.Refcv6Proxy = MapSelProxy                                # install_patches' load_model resolves the name at call time
    print(f"[mapsel] prereg {'sealed ' + SEAL_SHA[:16] if SEAL_SHA else 'UNSEALED'} child={' '.join(CHILD)} ZZMSROLLSTARTZZ", flush=True)
    t0 = time.time()
    rc = RUNNER()
    if not ST["records"]:
        print("ZZMSNOWINDOWSZZ the MapSelProxy never ran", flush=True)
        return 4
    save(True, {"runner_rc": rc, "roll_wall_s": round(time.time() - t0, 1),
                "cuda_max_memory_allocated_gib": (round(torch.cuda.max_memory_allocated() / 2**30, 3) if torch.cuda.is_initialized() else None)})
    print(f"[mapsel] banked {len(ST['records'])} windows present={json.dumps(ST['present'])} counters={json.dumps({k: (sorted(v) if isinstance(v, set) else v) for k, v in ST['c'].items()})} ZZMSBANKEDZZ", flush=True)
    return 0 if rc in (0, None) else int(rc)


if __name__ == "__main__":
    sys.exit(main())
