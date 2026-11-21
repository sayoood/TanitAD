"""fan_roll_r8_l7cap.py -- the L7 (H-L7-1) TRAIN-SPLIT FAN CAPTURE. Extends fan_roll_r8_pk.py (R8-PICKER) / fan_roll_r8_pf.py (FANCEIL),
the same WRAPPER pattern: it edits no file under stack/ or the battery packages.

  --mode train139   rolls the REGISTERED TRAIN139 windows (RULE.md): it calls `roll_seed_r8_train_pf.main()` (the RETIME train-fit roll,
                    per-frame nav feed, os-only, one nav_true row per window) with the capture installed.
  --mode eval139    rolls eval139 exactly as fan_roll_r8_pk.py does (`roll_seed_r8_pf.main()` + --windows-json); this is the K3 capture of
                    PREREG_L7_SCORER sec. 7 (the new eval capture at seed 1 must reproduce fanceil s1's `anchor_traj`), now with the scene sources.

Per window it dumps (INPUT bank = `fan_extras.npz` + `l7cap_scene.npz`; LABEL bank = `l7cap_labels.npz`, kept apart so a scorer input can never be a label):
  fan_extras.npz   clip_index, ws, t, v0, inference_seed, anchor_traj [W,149,8,2], anchor_logits [W,117], refined_logits, sel_score, sel_score_v3,
                   traj, traj_base, goal_dist, r8_sub_logits [W,149,4], reach_keep, ceil_keep, r8_emit_keep, e9_keep, sel_idx, sel_idx_base,
                   e9_argmax, r8_cand_src, is_alloc [149]   (same key names as the picker's fan_extras.npz, plus the new ones)
  l7cap_scene.npz  clip_index, ws, r7_bev_tokens, r7_bev_pad, r7_agent_tokens, r7_agent_pad  (DETACHED, fp16 when it is lossless-range)
  l7cap_labels.npz clip_index, ws, r7_agent_fut [W,8,pad,5] / _valid / _label   (with --l7-agent-join: ds.r7_agent_future = True)
                   map_drivable_u8 / map_seen [W,120,64]                          (with --l7-map-gt-root: the SAM3 10 cm codes at t0, cropped to the
                                                                                  oracle's 60 m x +-16 m geometry and block-pooled to 0.5 m)
NEVER dumped: `cond`, `r7_cond`, `z_tac`, the fed ceiling, nav (hooks.FORBIDDEN_BANK_KEYS; asserted before every write).

    python fan_roll_r8_l7cap.py --mode train139 --pf-code <hold_r8pf/code> --ckpt CK --config CFG --seed S --dump-dir D --out-json RJ \
        --fan-meta-json M [--registered-windows TRAIN139_WINDOWS.json] [--l7-agent-join F] [--l7-map-gt-root R] [--device cuda|cpu]
        [--smoke-cpu-fp32-trunk --max-clips N --max-windows-per-clip M]  [Thor: --train-remap-json J]
"""
from __future__ import annotations

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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l7cap_hooks as H  # noqa: E402

FLOAT_KEYS = ("anchor_traj", "anchor_logits", "refined_logits", "sel_score", "sel_score_v3", "traj", "traj_base", "goal_dist", "r8_sub_logits")
BOOL_KEYS = ("reach_keep", "ceil_keep", "r8_emit_keep")
INT_KEYS = ("sel_idx", "sel_idx_base", "r8_cand_src")
ROWS: dict = {}       # (e_i, t) -> input row
LABS: dict = {}       # (e_i, t) -> label row
STATS = {"n_calls_with_nav": 0, "n_calls_navzero": 0, "n_calls_without_nav_true": 0, "keys_missing": {}, "conds_seen": set(),
         "n_label_agent": 0, "n_label_map": 0, "scene_missing": 0, "call_s": []}
ARGS: dict = {}


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha12(s: str) -> str:
    return hashlib.sha256(str(s).encode()).hexdigest()[:12]


# ------------------------------------------------------------------------------------------------ capture on the model call
def _install_capture(RR, torch, RV3):
    orig = RR.Refcv6Proxy.__call__
    mut = H.MUTATION

    def wrapped(self, frames, nav_cmd=None, v0=None, steps=0, ego_state=None, **kw):
        _t = time.perf_counter()
        out = orig(self, frames, nav_cmd=nav_cmd, v0=v0, steps=steps, ego_state=ego_state, **kw)
        STATS["call_s"].append(time.perf_counter() - _t)           # the model call only (dataset decode / build excluded)
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
            for k in FLOAT_KEYS + BOOL_KEYS + INT_KEYS + H.SCENE_KEYS:
                v = out.get(k) if hasattr(out, "get") else None
                if v is None or not torch.is_tensor(v):
                    STATS["keys_missing"][k] = STATS["keys_missing"].get(k, 0) + 1
                    continue
                x = v[r].detach()
                if k in FLOAT_KEYS:
                    row[k] = x.float().cpu().numpy()
                elif k in BOOL_KEYS or k.endswith("_pad"):
                    row[k] = x.bool().cpu().numpy()
                elif k in INT_KEYS:
                    row[k] = x.long().cpu().numpy()
                else:                                                    # scene tokens: fp16 per row (lossless-range check), so the roll's host RAM
                    row[k] = H.to_fp16_checked(x.float().cpu().numpy())[0]       # holds 2 B/element, not 4 (300x384 + 480x96 tokens/window)
            if "sel_score_v3" in out and torch.is_tensor(out["sel_score_v3"]):
                rk, _ = RV3.e9_rank(torch.zeros_like(out["sel_score_v3"]), out)          # the model's OWN mask function, READ-ONLY
                row["e9_keep"] = torch.isfinite(rk[r]).cpu().numpy()
                rk2, _ = RV3.e9_rank(out["sel_score_v3"], out)
                row["e9_argmax"] = rk2[r].argmax().long().cpu().numpy()
            if "r8_n_base" in out:
                row["r8_n_base"] = np_int(out["r8_n_base"])
        if v0 is not None:
            row["v0"] = float(v0.reshape(-1)[r if v0.numel() > 1 else 0])
        if "anchor_traj" not in row or "e9_keep" not in row:
            raise RuntimeError("[l7cap] model output carries no anchor_traj / sel_score_v3")
        if ARGS.get("scene") and not all(k in row for k in ("r7_bev_tokens", "r7_agent_tokens")):
            STATS["scene_missing"] += 1
        if self.ctx.index in ROWS:
            raise RuntimeError("[l7cap] duplicate nav call for window %s" % (self.ctx.index,))
        ROWS[self.ctx.index] = row
        it = self.ctx.item
        lab = {}
        if it is not None and "r7_agent_fut" in it:
            for k in ("r7_agent_fut", "r7_agent_fut_valid", "r7_agent_fut_label"):
                lab[k] = it[k].numpy()
            STATS["n_label_agent"] += 1
        if it is not None and "map_fine" in it and ARGS.get("map_extent") is not None:
            drv, seen = H.crop_drivable(it["map_fine"].numpy()[None], ARGS["map_extent"])
            lab["map_drivable_u8"], lab["map_seen"] = drv[0], seen[0]
            lab["map_label"] = bool(it["map_fine_label"]) if "map_fine_label" in it else True
            STATS["n_label_map"] += 1
        if lab:
            LABS[self.ctx.index] = lab
        STATS["n_calls_with_nav"] += 1
        return out

    RR.Refcv6Proxy.__call__ = wrapped
    return orig


def np_int(t) -> int:
    return int(t.reshape(-1)[0]) if hasattr(t, "reshape") else int(t)


# ------------------------------------------------------------------------------------------------ label-side dataset patch
def _wrap_build_eval_dataset(L, agent_join, map_root):
    """After the roll's own `build_eval_dataset` (perception targets OFF), attach ONLY the label-side joins the L7 targets need. The agent
    join goes through the loader's OWN code path (it is called with the map/3-D/VIS-1 flags temporarily None), so `enable_agent_join` is
    the trainer's, unmodified. `ds.r7_agent_future = True` is the trainer's flag (refc_v3_train.py:9818). Neither touches `ds.index` or the
    model input: the capture reads them from `ctx.item` AFTER the model call."""
    prev = L.build_eval_dataset

    def bed(model, cfg, args, config, *, with_perception_targets=True, dataset_cls=None):
        if not (agent_join or map_root):
            return prev(model, cfg, args, config, with_perception_targets=with_perception_targets, dataset_cls=dataset_cls)
        saved = {k: getattr(args, k, None) for k in ("agent_join", "map_gt_root", "join3d", "slot_vis1")}
        try:
            args.agent_join, args.map_gt_root, args.join3d, args.slot_vis1 = (agent_join or None), None, None, False
            e_ds, e_eps, rec = prev(model, cfg, args, config, with_perception_targets=bool(agent_join), dataset_cls=dataset_cls)
        finally:
            for k, v in saved.items():
                setattr(args, k, v)
        n0 = len(e_ds.index)
        if agent_join:
            e_ds.r7_agent_future = True
        if map_root:
            tr = L.trainer()
            _clip, _ns = tr._clip_table_for_caches([args.eval_cache])
            store = tr._sem_fine.FineMapGTStore(Path(map_root), max_open=int(getattr(args, "map_lru", 4) or 4),
                                                extent=tr._mhr.declared_extent(args))
            rec["map_fine"] = e_ds.enable_map_hires(store, _clip, _ns, min_coverage=getattr(args, "map_min_coverage", None))
            ARGS["map_extent"] = store.extent
        if len(e_ds.index) != n0:
            raise SystemExit("[l7cap] the label-side attach changed the window index (%d -> %d): refusing" % (n0, len(e_ds.index)))
        rec["l7cap_label_side"] = {"agent_join": bool(agent_join), "map_gt_root": bool(map_root), "index_unchanged": True}
        return e_ds, e_eps, rec

    bed._l7cap = True
    L.build_eval_dataset = bed


# ------------------------------------------------------------------------------------------------ writing
def assert_no_forbidden(keys) -> None:
    bad = sorted(set(keys) & H.FORBIDDEN_BANK_KEYS)
    if bad:
        raise SystemExit("[l7cap] REFUSING to write forbidden keys into an INPUT bank: %s" % (bad,))


def write_banks(dump_dir: Path, side, seed: int, n_base_expected: int) -> dict:
    import numpy as np
    keys = sorted(ROWS)
    if len(keys) != len(side["ws"]):
        raise SystemExit("[l7cap] captured rows %d != sidecar windows %d" % (len(keys), len(side["ws"])))
    ci = np.asarray([k[0] for k in keys], np.int64)
    if not np.array_equal(ci, side["clip_index"]):
        raise SystemExit("[l7cap] captured clip_index order != sidecar order")
    ws = np.asarray(side["ws"], np.int64)
    out = {"clip_index": ci, "ws": ws, "t": np.asarray([k[1] for k in keys], np.int64),
           "inference_seed": np.asarray([seed], np.int64)}
    for k in FLOAT_KEYS + BOOL_KEYS + INT_KEYS + ("e9_keep", "e9_argmax"):
        if all(k in ROWS[kk] for kk in keys):
            v = np.stack([ROWS[kk][k] for kk in keys])
            out[k] = (v.astype(np.float32) if k in FLOAT_KEYS else v.astype(bool) if k in BOOL_KEYS + ("e9_keep",) else v.astype(np.int64))
    out["v0"] = np.asarray([ROWS[kk].get("v0", np.nan) for kk in keys], np.float32)
    n_base = {ROWS[kk].get("r8_n_base", n_base_expected) for kk in keys}
    if n_base != {n_base_expected}:
        raise SystemExit("[l7cap] r8_n_base %s != %d" % (n_base, n_base_expected))
    N = out["anchor_traj"].shape[1]
    out["is_alloc"] = (np.arange(N) >= n_base_expected)
    if H.MUTATION == "wrong_sel_idx":                                  # deliberate regression (control c): dump the WRONG candidate index
        out["sel_idx"] = (out["sel_idx"] + 1) % N
    assert_no_forbidden(out.keys())
    np.savez_compressed(str(dump_dir / "fan_extras.npz"), **out)
    info = {"fan_extras_keys": sorted(out)}
    # scene bank: filled row by row and the per-row arrays freed as we go (peak = rows + ONE output array, not rows + a full np.stack copy)
    sc = {"clip_index": ci, "ws": ws}
    sinfo = {}
    for k in H.SCENE_KEYS:
        if not all(k in ROWS[kk] for kk in keys):
            continue
        if k.endswith("_pad"):
            sc[k] = np.stack([ROWS[kk].pop(k) for kk in keys]).astype(bool)
            continue
        dts = {ROWS[kk][k].dtype for kk in keys}
        dt = np.dtype("float16") if dts == {np.dtype("float16")} else np.dtype("float32")
        arr = np.empty((len(keys),) + ROWS[keys[0]][k].shape, dt)
        mx = 0.0
        for i, kk in enumerate(keys):
            r_ = ROWS[kk].pop(k)
            arr[i] = r_
            mx = max(mx, float(np.abs(r_.astype(np.float32)).max()))
        sc[k], sinfo[k] = arr, {"stored": str(dt), "max_abs": mx, "n_rows_fp32_fallback": int(sum(d != np.dtype("float16") for d in dts))}
    assert_no_forbidden(sc.keys())
    if len(sc) > 2:
        np.savez_compressed(str(dump_dir / "l7cap_scene.npz"), **sc)
    info["scene_keys"] = sorted(set(sc) - {"clip_index", "ws"})
    info["scene_shapes"] = {k: list(sc[k].shape) for k in info["scene_keys"]}
    info["scene_storage"] = sinfo
    # label bank
    if LABS:
        lb = {"clip_index": ci, "ws": ws}
        have = [kk for kk in keys if kk in LABS]
        info["n_label_windows"] = len(have)
        for k in sorted({k for kk in have for k in LABS[kk]}):
            if all(k in LABS[kk] for kk in keys):
                v = np.stack([np.asarray(LABS[kk][k]) for kk in keys])
                lb[k] = v
        np.savez_compressed(str(dump_dir / "l7cap_labels.npz"), **lb)
        info["label_keys"] = sorted(set(lb) - {"clip_index", "ws"})
        info["label_shapes"] = {k: list(lb[k].shape) for k in info["label_keys"]}
    return info


def check_registered(reg_path: str, roll_rec: dict, ci, ws, smoke: bool) -> dict:
    reg = json.load(open(reg_path, encoding="utf-8"))
    by_idx = roll_rec["extras"]["clip_sha12_by_index"]
    got = {}
    for c, w in zip(ci.tolist(), ws.tolist()):
        got.setdefault("clip-sha12-" + by_idx[str(c)], []).append(int(w))
    want = {k: v["ws"] for k, v in reg["clips"].items()}
    extra = sorted(k for k in got if k not in want)
    bad = sorted(k for k in got if k in want and (not set(got[k]) <= set(want[k]) if smoke else sorted(got[k]) != sorted(want[k])))
    miss = sorted(set(want) - set(got))
    ok = (not extra and not bad) and (smoke or not miss)
    return {"registered_windows_sha256": sha256_file(reg_path), "ok": bool(ok), "smoke_subset_rule": bool(smoke),
            "n_clips_got": len(got), "n_windows_got": int(len(ws)), "n_clips_registered": len(want),
            "n_clips_not_registered": len(extra), "n_clips_window_mismatch": len(bad), "n_registered_clips_missing": len(miss)}


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("train139", "eval139"), required=True)
    ap.add_argument("--pf-code", required=True)
    ap.add_argument("--remap-json", default=None, help="Thor eval139 mode only: the md5-equal Thor eval remap")
    ap.add_argument("--train-remap-json", default=None, help="Thor train139 mode only: the TRAIN remap re-pointed at Thor paths")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--windows-json", default=None, help="eval139 mode")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--fan-meta-json", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--os-only", action="store_true", help="eval139 mode: seeds after the first")
    ap.add_argument("--smoke-cpu-fp32-trunk", action="store_true")
    ap.add_argument("--max-clips", type=int, default=0)
    ap.add_argument("--max-windows-per-clip", type=int, default=0)
    ap.add_argument("--registered-windows", default=None)
    ap.add_argument("--l7-agent-join", default=None)
    ap.add_argument("--l7-map-gt-root", default=None)
    ap.add_argument("--no-scene", action="store_true")
    a = ap.parse_args()
    ARGS.update(scene=not a.no_scene)
    pf_code = Path(a.pf_code)
    r8_code = Path(os.environ.get("PF_R8_CODE", "C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/"
                                                "2026-10-15-refcv8-milestone-battery/code"))
    sys.path.insert(0, str(r8_code))
    import r8_repoint  # noqa: E402
    dev_remap = r8_repoint.REMAP_OVERRIDES
    if a.mode == "eval139":
        if a.remap_json:
            r8_repoint.REMAP_OVERRIDES = Path(a.remap_json)
        if not a.windows_json:
            raise SystemExit("[l7cap] eval139 mode needs --windows-json")
        child = [str(pf_code / "roll_seed_r8_pf.py"), "--ckpt", a.ckpt, "--config", a.config, "--seed", str(a.seed), "--dump-dir", a.dump_dir,
                 "--windows-json", a.windows_json, "--device", a.device, "--out-json", a.out_json]
        if a.smoke_cpu_fp32_trunk:
            child.append("--smoke-cpu-fp32-trunk")
        if a.os_only:
            child.append("--os-only")
        sys.argv = child
        sys.path.insert(0, str(pf_code))
        import roll_seed_r8_pf as RP  # noqa: E402
        loader = None
    else:
        if a.train_remap_json:                       # Thor: re-point the TRAIN remap BEFORE roll_seed_r8_train applies it
            _apply = r8_repoint.apply
            tr_remap = Path(a.train_remap_json)

            def apply(*aa, **kk):
                r8_repoint.REMAP_OVERRIDES = tr_remap
                return _apply(*aa, **kk)
            r8_repoint.apply = apply
        import roll_seed_r8_train as T  # noqa: E402  (module-level: TRAIN remap + loader patch, UNCHANGED)
        if a.train_remap_json:
            T.TRAIN_REMAP = Path(a.train_remap_json)
        sys.path.insert(0, str(pf_code))
        import roll_seed_r8_train_pf as RP  # noqa: E402
        child = [str(pf_code / "roll_seed_r8_train_pf.py"), "--ckpt", a.ckpt, "--config", a.config, "--seed", str(a.seed), "--dump-dir", a.dump_dir,
                 "--out-json", a.out_json, "--device", a.device]
        if a.smoke_cpu_fp32_trunk:
            child.append("--smoke-cpu-fp32-trunk")
        if a.max_clips:
            child += ["--max-clips", str(a.max_clips)]
        if a.max_windows_per_clip:
            child += ["--max-windows-per-clip", str(a.max_windows_per_clip)]
        sys.argv = child
        loader = T.L
    import r7_roll as RR  # noqa: E402
    import torch  # noqa: E402
    import numpy as np  # noqa: E402
    from tanitad.refs import refc_v3 as RV3  # noqa: E402
    if a.mode == "eval139":
        loader = RR.L
    H.install_scene_capture(RV3, torch)
    _wrap_build_eval_dataset(loader, a.l7_agent_join, a.l7_map_gt_root)
    _install_capture(RR, torch, RV3)
    t0 = time.time()
    rc = RP.main()
    wall = time.time() - t0
    dump = Path(a.dump_dir)
    side = np.load(str(dump / "refcv6_extras.npz"), allow_pickle=True)
    roll_rec = json.load(open(a.out_json, encoding="utf-8"))
    info = write_banks(dump, side, a.seed, 117)
    ci = np.asarray(side["clip_index"], np.int64)
    ws = np.asarray(side["ws"], np.int64)
    fan = np.load(str(dump / "fan_extras.npz"))
    k3b = bool(np.array_equal(fan["traj"], side["nav_true.traj"].astype(np.float32)))
    sel_true = fan["sel_idx"]                    # the index AS WRITTEN: a mutated dump must fail these roll-time checks too
    k3a = float(np.abs(fan["anchor_traj"][np.arange(len(sel_true)), sel_true] - fan["traj"]).max())
    k4 = float((fan["e9_argmax"].reshape(-1) == sel_true.reshape(-1)).mean())
    reg = (check_registered(a.registered_windows, roll_rec, ci, ws, bool(a.max_clips or a.max_windows_per_clip))
           if (a.registered_windows and a.mode == "train139") else None)
    pk = H.peak_ws_gb()
    meta = {"mode": a.mode, "n_windows": int(len(ci)), "fan_shape": list(fan["anchor_traj"].shape), **info,
            "keys_missing_counts": STATS["keys_missing"], "conds_seen": sorted(map(list, STATS["conds_seen"])),
            "n_calls_navzero": STATS["n_calls_navzero"], "n_calls_without_nav_true": STATS["n_calls_without_nav_true"],
            "n_label_agent_windows": STATS["n_label_agent"], "n_label_map_windows": STATS["n_label_map"],
            "scene_missing_windows": STATS["scene_missing"],
            "K3a_max_abs_fan_at_sel_minus_traj_m": k3a, "K3b_traj_equals_sidecar_nav_true": k3b, "K4_e9_argmax_equals_sel_idx_frac": k4,
            "e9_keep_min_per_row": int(fan["e9_keep"].sum(1).min()), "registered_windows_check": reg,
            "inference_seed": a.seed, "mutation": H.MUTATION or None, "device": a.device, "child_rc": rc, "child_argv": child,
            "model_call_s": ({"n": len(STATS["call_s"]), "first": round(STATS["call_s"][0], 2), "median": round(float(np.median(STATS["call_s"])), 2),
                              "mean": round(float(np.mean(STATS["call_s"])), 2), "max": round(max(STATS["call_s"]), 2)} if STATS["call_s"] else None),
            "wall_s": round(wall, 1), "wall_s_per_window_incl_build": round(wall / max(len(ci), 1), 2),
            "peak_working_set_gb": (round(pk[0], 3) if pk else None), "peak_commit_gb": (round(pk[1], 3) if pk else None),
            "cuda_max_memory_allocated_gib": (round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
                                              if (a.device.startswith("cuda") and torch.cuda.is_initialized()) else None),
            "remap": {"dev_default": str(dev_remap), "re_pointed_eval": bool(a.remap_json), "re_pointed_train": bool(a.train_remap_json)},
            "env": {k: os.environ.get(k) for k in ("R8_LAUNCH_TREE", "R8_EVAL_KIT", "PF_R8_CODE", "PF_NF_HOLD", "R8_NO_POSE_SYNC",
                                                    "OMP_NUM_THREADS", "PYTHONPATH", "CUDA_VISIBLE_DEVICES")},
            "torch": torch.__version__, "ckpt": a.ckpt, "config": a.config,
            "this_script_sha256": sha256_file(__file__), "hooks_sha256": sha256_file(HERE / "l7cap_hooks.py"),
            "refc_v3_file": RV3.__file__, "refc_v3_sha256": sha256_file(RV3.__file__),
            "prereg": "PREREG_L7_SCORER.md sha256 f9c1431c... (sealed 2026-10-21 23:33 +0200)",
            "producer": "fan_roll_r8_l7cap.py (wrapper over roll_seed_r8_train_pf.main / roll_seed_r8_pf.main; capture on r7_roll.Refcv6Proxy.__call__)"}
    json.dump(meta, open(a.fan_meta_json, "w", encoding="utf-8"), indent=1, default=str)
    ok = bool(k3b and k3a == 0.0 and rc in (0, None) and (reg is None or reg["ok"]) and STATS["scene_missing"] == 0 and
              (not a.l7_agent_join or STATS["n_label_agent"] == len(ci)))
    print("[l7cap] %d windows fan %s K3a %.3g K3b %s K4 %.3f scene %s labels(agent %d map %d) reg %s rc=%s -> %s %s" % (
        len(ci), fan["anchor_traj"].shape, k3a, k3b, k4, info.get("scene_shapes"), STATS["n_label_agent"], STATS["n_label_map"],
        (reg or {}).get("ok"), rc, a.fan_meta_json, "ZZL7CROLLOKZZ" if ok else "ZZL7CROLLBADZZ"), flush=True)
    if not ok:
        raise SystemExit("[l7cap] a capture check failed (see %s)" % a.fan_meta_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
