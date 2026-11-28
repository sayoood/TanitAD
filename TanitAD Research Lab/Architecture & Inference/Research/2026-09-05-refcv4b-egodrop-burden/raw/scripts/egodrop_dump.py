"""refcv4b ego-dropout GEOMETRIC-BURDEN dump -- PAIRED kept / withheld forwards.

For every window of the banked refcv3 open-loop grid (stride 5 -> 4,823 windows
/ 141 B1-v7.2 EVAL clips) the SAME frames are pushed through the model twice in
ONE batch: row 0 with the measured ego block (keep = 1), row 1 with the ego
block WITHHELD (keep = 0), which is exactly the training-time dropout regime
(refc_v3.py: `ego_state[:, 4] *= k`; refc.py: `keep = keep * ego_keep; v *= keep`;
roll_bank: withheld rows roll the anchor bank at `anchor_ref_speed` = 10 m/s).
Nothing else differs between the rows, so every contrast below is paired.

EARLY-TRAINING DIAGNOSTIC (step ~9.5k of 40,284). Not a capability claim.
Reuses the refcv3 arm's instruments by import (model rebuild through the
trainer's own parser, the trainer's own window contract, the programme's one
unicycle integrator for the ha / ha0 controls). Writes:

  <out>/dump8/ep%03d.npz            g [n,8,2], six arms [n,8,2], v0, ws, eid, clip_index
  <out>/dump8/decisions/ep%03d.npz  per-regime selection / offset / bank / heads / labels
  <out>/dump8/manifest.json

Arms (all on the model's own 8 slots = 0.5,1,1.5,2,3,4,5,6 s):
  os_k   the DEPLOYED selection, ego KEPT          T1 (open loop, self-action)
  os_w   the DEPLOYED selection, ego WITHHELD      T1
  orc_k  oracle-selected refined anchor, KEPT      T0 (GT-nearest a_star, on the KEPT bank)
  orc_w  oracle-selected refined anchor, WITHHELD  T0 (a_star on the 10 m/s bank)
  ha     hold the action that closes at t0         T1 (model-free control)
  ha0    constant velocity at v0, straight         T1 (model-free control)
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np

REPO = r"C:\Users\Admin\refcv4b_repo"
for p in (os.path.join(REPO, "stack"), os.path.join(REPO, "taniteval"),
          os.path.join(REPO, "stack", "scripts"), REPO):
    if p not in sys.path:
        sys.path.insert(0, p)


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


arm = _load_by_path("refcv3_arm", os.path.join(REPO, "taniteval", "tools", "refcv3_arm.py"))
import torch                                                     # noqa: E402
import tanitad                                                   # noqa: E402
from tanitad.refs import refc_v3 as v3                           # noqa: E402
import refb_labels                                               # noqa: E402

assert os.path.normcase(os.path.abspath(tanitad.__file__)).startswith(
    os.path.normcase(os.path.abspath(REPO))), tanitad.__file__

ra, tr = arm.ra, arm.trainer()
DT = arm.DT_FRAME
GRID2S = ("2s", arm.GRIDS["2s"])


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def p(*a):
    print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--anchors-file", required=True, help="the pod's anchors.pt (sha e86cf507...)")
    ap.add_argument("--episodes", default=r"C:\Users\Admin\run_refcv3_ol\data\eval")
    ap.add_argument("--labels", default=r"C:\Users\Admin\run_refcv3_ol\data\s2_labels_v7.2_eval.jsonl.gz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--window-stride", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lru", type=int, default=8)
    ap.add_argument("--episodes-n", type=int, default=0)
    ap.add_argument("--max-windows", type=int, default=0, help="smoke: stop after n windows")
    a = ap.parse_args()
    a.nav_source = "v72"
    a.allow_nonstrict = False
    t_start = time.time()
    dev = a.device
    torch.backends.cudnn.benchmark = False

    # ---- model: rebuilt through the trainer's own parser, STRICT load ------ #
    model, cfg, targs, prov = arm.load_model(a.ckpt, a.config, dev, False)
    steps = int(prov["decoder_steps"])
    dec = model.core.decoder
    checks = {
        "ego_state_inject": bool(cfg.ego_state_inject),
        "core.ego_valid_channel": bool(cfg.core.ego_valid_channel),
        "core.ego_dropout": float(cfg.core.ego_dropout),
        "echo_base": bool(cfg.echo_base),
        "anchor_v0_cond": bool(getattr(dec, "anchor_v0_cond", False)),
        "anchor_ref_speed": float(getattr(dec, "anchor_ref_speed", float("nan"))),
        "anchor_control_units": str(getattr(dec, "anchor_control_units", "?")),
        "anchor_kappa_cap": float(getattr(dec, "anchor_kappa_cap", float("nan"))),
        "anchor_alat_v_floor": float(getattr(dec, "anchor_alat_v_floor", float("nan"))),
        "sel_reach_clamp": bool(cfg.core.sel_reach_clamp),
        "sel_accel_max": float(cfg.core.sel_accel_max),
        "horizon_s": float(getattr(dec.sel, "horizon_s", float("nan"))),
        "decoder_steps": steps, "window": int(cfg.core.window),
        "n_anchors": int(cfg.core.anchors.n_anchors),
        "refc1_speed_head": bool(cfg.core.refc1),
        "hier": bool(cfg.hier), "tac_vocab_version": cfg.tac_vocab_version,
    }
    assert checks["ego_state_inject"] and checks["core.ego_valid_channel"], checks
    assert checks["anchor_v0_cond"] and abs(checks["anchor_ref_speed"] - 10.0) < 1e-9, checks
    assert checks["core.ego_dropout"] == 0.5, checks
    # the bank the checkpoint carries IS the live pod file
    anc = torch.load(a.anchors_file, map_location="cpu", weights_only=False)
    checks["anchors_file_sha256"] = _sha256(a.anchors_file)
    checks["anchors_buffer_equals_file"] = bool(torch.equal(
        dec.anchors.detach().cpu().float(), anc["anchors"].float()))
    checks["controls_buffer_equals_file"] = bool(torch.equal(
        dec.anchor_controls.detach().cpu().float(), anc["controls"].float()))
    ctrl = anc["controls"].float()
    zero_rows = (ctrl.abs().sum(1) == 0).nonzero().flatten().tolist()
    checks["straight_ahead_index"] = zero_rows
    assert checks["anchors_buffer_equals_file"] and checks["controls_buffer_equals_file"], checks
    assert zero_rows == [67], zero_rows
    p(f"[model] step={prov['step']} steps={steps} W={checks['window']} "
      f"N={checks['n_anchors']} refc1_speed_head={checks['refc1_speed_head']} "
      f"anchors==file {checks['anchors_buffer_equals_file']}")

    # ---- corpus: the trainer's window contract, the v7.2 join, nav v72 ----- #
    eps, files, clip_ids, ds, lman, join, nav_src, raw_off = arm.build_corpus(a, cfg, prov)
    W = int(cfg.core.window)
    horizons = list(cfg.core.trajectory.horizons)
    S = len(horizons)
    slot_idx = torch.tensor([h - 1 for h in horizons])
    grid2 = arm.grid_slots(horizons, "2s")
    stride = max(1, int(a.window_stride))
    sel = [(wi, e_i, t) for wi, (e_i, t) in enumerate(ds.index) if wi % stride == 0]
    nav_true = np.zeros(len(sel), dtype=np.int64)
    nav_valid = np.zeros(len(sel), dtype=bool)
    for i, (wi, e_i, t) in enumerate(sel):
        nid = ds._nav_by_sid.get(int(ds.episodes[e_i].episode_id))
        nav_true[i] = 0 if nid is None else int(nid)
        nav_valid[i] = nid is not None
    p(f"[grid] {len(sel)} candidate windows / {len(eps)} episodes, stride {stride}, "
      f"slots {horizons} (0.1 s ticks); nav={nav_src} raw_off={raw_off}")

    out_dir = os.path.join(a.out, "dump8")
    os.makedirs(os.path.join(out_dir, "decisions"), exist_ok=True)
    by_ep: dict[int, list] = {}
    for i, (wi, e_i, t) in enumerate(sel):
        by_ep.setdefault(e_i, []).append((i, wi, t))

    n_done = n_skip = 0
    skip_reasons: dict[str, int] = {}
    episodes_manifest = []
    t_first = None
    bit_control = None
    out_keys_seen = None
    for fi, e_i in enumerate(sorted(by_ep)):
        ep = ds.episodes[e_i]
        v_ep = ep.poses[:, 3].float()
        kap_ep = ep.actions[:, 0].float()
        acc: dict[str, list] = {k: [] for k in
                                ("g", "v0", "os_k", "os_w", "orc_k", "orc_w", "ha", "ha0")}
        dcs: dict[str, list] = {}
        ws = []
        for (i, wi, t) in by_ep[e_i]:
            if a.max_windows and n_done >= a.max_windows:
                break
            t0 = t + W - 1
            item = ds[wi]
            fv = item["future_valid_ext"]
            if not bool(fv[[h - 1 for h in grid2["horizons_steps"]]].all()):
                n_skip += 1
                skip_reasons["horizon_beyond_episode_2s"] = skip_reasons.get("horizon_beyond_episode_2s", 0) + 1
                continue
            if t0 < 1:
                n_skip += 1
                skip_reasons["no_closed_action_before_t0"] = skip_reasons.get("no_closed_action_before_t0", 0) + 1
                continue
            pose_last = item["pose_last"].float()
            v0 = float(pose_last[3])
            traj_tgt = refb_labels.waypoint_targets(
                pose_last[None], item["future_poses_ext"].float()[None], horizons)  # [1,S,2]
            sv = torch.stack([fv[h - 1] for h in horizons]).float()                  # [S]
            # -- model-free controls on the SAME 8 slots (programme's integrator) --
            hold = arm.hold_controls(v_ep, kap_ep, t0)
            n_f = int(horizons[-1])
            ha = ra.paths_from_controls(hold[None].expand(n_f, 2), v0, DT, n_f,
                                        action_units="steer")[:, slot_idx]          # [1,S,2]
            ha0 = ra.paths_from_controls(arm.hold_v0_controls(n_f), v0, DT, n_f,
                                         action_units="steer")[:, slot_idx]
            # -- the ego block: MEASURED (keep=1) and WITHHELD (keep=0) ----------
            ego1 = v3.ego_state_from_batch(
                {"pose_last": pose_last[None], "actions": item["actions"][None]})  # [1,5]
            ego2 = torch.cat([ego1, ego1], 0)
            ego2[1, 4] = 0.0
            fr = tr.frames_to_device(item["frames"][None], dev)                     # [1,W,C,H,W']
            fr2 = fr.expand(2, *fr.shape[1:]).contiguous()
            nav_t = torch.tensor([int(nav_true[i])] * 2, dtype=torch.long, device=dev)
            v0_t = torch.full((2,), v0, dtype=torch.float32, device=dev)
            tf = time.time()
            with torch.no_grad():
                out = model(fr2, nav_cmd=nav_t, v0=v0_t, steps=steps,
                            ego_state=ego2.to(dev))
            if t_first is None:
                t_first = time.time() - tf
                out_keys_seen = sorted(out.keys())
                p(f"[cost] first paired forward {t_first:.2f} s -> ~"
                  f"{t_first * len(sel) / 3600:.2f} h for {len(sel)} windows")
                p(f"[out] keys: {out_keys_seen}")
                # ⭐ CONTROLS THAT MUST READ KNOWN VALUES, on the first window:
                with torch.no_grad():
                    o1 = model(fr, nav_cmd=nav_t[:1], v0=v0_t[:1], steps=steps,
                               ego_state=ego2[:1].to(dev))
                    # SAME batch shape, withheld row with its VALUES zeroed too
                    ego_z = ego2.clone(); ego_z[1, :4] = 0.0
                    oz = model(fr2, nav_cmd=nav_t, v0=v0_t, steps=steps,
                               ego_state=ego_z.to(dev))
                    bank_ref = dec.roll_bank(None, None, 1, out["anchor_bank"].dtype)
                bit_control = {
                    "kept_row_vs_single_row_forward_max_abs_traj_m": float(
                        (out["traj"][0] - o1["traj"][0]).abs().max()),
                    "kept_row_vs_single_row_sel_idx_equal": bool(int(out["sel_idx"][0]) == int(o1["sel_idx"][0])),
                    "withheld_row_vs_zeroed_values_row_max_abs_traj_m": float(
                        (out["traj"][1] - oz["traj"][1]).abs().max()),
                    "withheld_row_vs_zeroed_values_row_max_abs_fan_m": float(
                        (out["anchor_traj"][1] - oz["anchor_traj"][1]).abs().max()),
                    "withheld_bank_equals_ref_speed_roll_max_abs_m": float(
                        (out["anchor_bank"][1] - bank_ref[0]).abs().max()),
                    "kept_bank_vs_withheld_bank_max_abs_m": float(
                        (out["anchor_bank"][0] - out["anchor_bank"][1]).abs().max()),
                    "v0_first_window": v0,
                    "_reads": ("row 0 of the paired batch vs a SINGLE-row forward is a cross-call "
                               "comparison and carries the float32 batching floor (~1e-6..1e-5 m, "
                               "informational, tolerance 1e-4); the withheld row must be EXACTLY "
                               "invariant to the VALUES of the ego block at the same batch shape "
                               "(the keep bit zeroes them downstream); the withheld bank must equal "
                               "the decoder's own reference-speed roll (v_ms=None) EXACTLY"),
                }
                p(f"[bit-control] {json.dumps(bit_control)}")
                assert bit_control["withheld_bank_equals_ref_speed_roll_max_abs_m"] == 0.0, bit_control
                assert bit_control["withheld_row_vs_zeroed_values_row_max_abs_traj_m"] == 0.0, bit_control
                assert bit_control["withheld_row_vs_zeroed_values_row_max_abs_fan_m"] == 0.0, bit_control
                assert bit_control["kept_row_vs_single_row_forward_max_abs_traj_m"] < 1e-4, bit_control
            bank = out["anchor_bank"].float()                     # [2,N,S,2]
            fan = out["anchor_traj"].float()                      # [2,N,S,2]  (bank + ALL offsets)
            off0 = out["offset"].float()                          # [2,N,S,2]  (first-pass offset only)
            traj = out["traj"].float()                            # [2,S,2]
            tgt = traj_tgt.to(dev)
            dist = (((tgt[:, None] - bank) ** 2).sum(-1) * sv.to(dev)[None, None]).sum(-1)  # [2,N]
            a_star = dist.argmin(dim=1)                           # [2]
            ar2 = torch.arange(2, device=dev)
            sidx = out["sel_idx"]
            acc["g"].append(traj_tgt.numpy())
            acc["v0"].append(np.array([v0], dtype=np.float32))
            acc["os_k"].append(traj[0:1].cpu().numpy())
            acc["os_w"].append(traj[1:2].cpu().numpy())
            acc["orc_k"].append(fan[0:1, a_star[0]].cpu().numpy())
            acc["orc_w"].append(fan[1:2, a_star[1]].cpu().numpy())
            acc["ha"].append(ha.float().cpu().numpy())
            acc["ha0"].append(ha0.float().cpu().numpy())

            def put(k, v):
                dcs.setdefault(k, []).append(v)
            for r, tag in ((0, "k"), (1, "w")):
                si, ai = int(sidx[r]), int(a_star[r])
                put(f"sel_idx_{tag}", si)
                put(f"sel_idx_base_{tag}", int(out["sel_idx_base"][r]) if "sel_idx_base" in out else -1)
                put(f"a_star_{tag}", ai)
                put(f"cls_top1_{tag}", int(out["anchor_logits"][r].argmax()))
                put(f"sel_score_{tag}", out["sel_score_v3"][r].float().cpu().numpy()[None]
                    if "sel_score_v3" in out else out["sel_score"][r].float().cpu().numpy()[None])
                put(f"anchor_logits_{tag}", out["anchor_logits"][r].float().cpu().numpy()[None])
                put(f"reach_n_{tag}", int(out["reach_keep"][r].sum()) if "reach_keep" in out else -1)
                put(f"bank_sel_{tag}", bank[r, si].cpu().numpy()[None])
                put(f"bank_astar_{tag}", bank[r, ai].cpu().numpy()[None])
                put(f"off0_sel_{tag}", off0[r, si].cpu().numpy()[None])
                put(f"off0_astar_{tag}", off0[r, ai].cpu().numpy()[None])
                put(f"fan_minus_bank_mean_{tag}", float((fan[r] - bank[r]).norm(dim=-1).mean()))
                put(f"off0_fan_mean_{tag}", float(off0[r].norm(dim=-1).mean()))
                put(f"lat_tac_{tag}", int(out["lat_logits_tac"][r].argmax()) if "lat_logits_tac" in out else -1)
                put(f"lon_tac_{tag}", int(out["lon_logits_tac"][r].argmax()) if "lon_logits_tac" in out else -1)
                put(f"lat_core_{tag}", int(out["lat_decision"][r]) if "lat_decision" in out else -1)
                put(f"lon_core_{tag}", int(out["lon_decision"][r]) if "lon_decision" in out else -1)
                put(f"route_pred_{tag}", int(out["route_logits"][r].argmax()))
                put(f"g_str_{tag}", out["g_str"][r].float().cpu().numpy()[None] if "g_str" in out else np.full((1, 3), np.nan, np.float32))
                put(f"g_tac_{tag}", out["g_tac"][r].float().cpu().numpy()[None] if "g_tac" in out else np.full((1, 3, 4), np.nan, np.float32))
                put(f"goal_point_tac_{tag}", out["goal_point_tac"][r].float().cpu().numpy()[None] if "goal_point_tac" in out else np.full((1, 2), np.nan, np.float32))
                put(f"goal_dist_sel_{tag}", float(out["goal_dist"][r, si]) if "goal_dist" in out else float("nan"))
                put(f"sel_score_max_{tag}", float((out["sel_score_v3"] if "sel_score_v3" in out else out["sel_score"])[r].max()))
                if "target_speed" in out:
                    put(f"target_speed_{tag}", float(out["target_speed"][r]))
            put("goal_gate", float(out["goal_gate_value"]) if "goal_gate_value" in out else float("nan"))
            put("nav_injected", float(bool(out.get("nav_injected", False))))
            put("sv", sv.numpy()[None])
            put("ego_state", ego1.numpy())
            put("ha_controls", hold.float().numpy()[None])
            lat_v7 = int(item["lat_v7"]) if "lat_v7" in item else -100
            lon_v7 = int(item["lon_v7"]) if "lon_v7" in item else -100
            from tanitad.data import v7_labels as _v7l
            put("lat_label", -100 if lat_v7 == _v7l.IGNORE_ID else lat_v7)
            put("lon_label", -100 if lon_v7 == _v7l.IGNORE_ID else lon_v7)
            rv = bool(item["route_valid"])
            put("route_label", int(item["route_target"]) if rv else -100)
            put("nav_cmd", int(nav_true[i]))
            put("nav_valid", bool(nav_valid[i]))
            put("goal_tac_lab", item["goal_tac"].float().numpy()[None])
            put("goal_tac_valid", item["goal_tac_valid"].float().numpy()[None])
            ws.append(int(t0))
            n_done += 1
        if not ws:
            p(f"  [{fi + 1}/{len(by_ep)}] {clip_ids[e_i][:12]} - 0 scoreable windows, SKIPPED")
            if a.max_windows and n_done >= a.max_windows:
                break
            continue
        k = len(episodes_manifest)
        np.savez_compressed(os.path.join(out_dir, f"ep{k:03d}.npz"),
                            **{kk: np.concatenate(v).astype(np.float32) for kk, v in acc.items()},
                            ws=np.array(ws), eid=np.array([k]), clip_index=np.array([e_i]))
        dnp = {}
        for kk, v in dcs.items():
            if isinstance(v[0], np.ndarray):
                dnp[kk] = np.concatenate(v).astype(np.float32)
            elif isinstance(v[0], bool):
                dnp[kk] = np.array(v, dtype=bool)
            elif isinstance(v[0], float):
                dnp[kk] = np.array(v, dtype=np.float32)
            else:
                dnp[kk] = np.array(v, dtype=np.int64)
        np.savez_compressed(os.path.join(out_dir, "decisions", f"ep{k:03d}.npz"),
                            ws=np.array(ws), **dnp)
        episodes_manifest.append({"file_index": k, "episode_index": e_i,
                                  "clip_id": clip_ids[e_i],
                                  "episode_id": int(ds.episodes[e_i].episode_id),
                                  "n_windows": len(ws)})
        p(f"  [{fi + 1}/{len(by_ep)}] {clip_ids[e_i][:12]} {len(ws)} windows "
          f"{time.time() - t_start:.0f}s")
        if a.max_windows and n_done >= a.max_windows:
            break

    manifest = {
        "tool": "refcv4b_egodrop/egodrop_dump.py (paired kept/withheld; instruments IMPORTED from "
                "taniteval/tools/refcv3_arm.py + refav1_arm.py)",
        "model": prov, "model_checks": checks, "bit_control": bit_control,
        "out_keys": out_keys_seen,
        "ckpt_md5": _md5(a.ckpt), "config_md5": _md5(a.config),
        "regimes": {"k": "ego block MEASURED at t0, keep=1 (the eval regime)",
                    "w": "ego block WITHHELD, keep=0 (the training-time dropout regime; "
                         "the anchor bank is rolled at anchor_ref_speed=10 m/s)"},
        "arms": ["os_k", "os_w", "orc_k", "orc_w", "ha", "ha0"],
        "tiers": {"os_k": "T1", "os_w": "T1", "orc_k": "T0", "orc_w": "T0",
                  "ha": "T1", "ha0": "T1"},
        "arm_meaning": {
            "os_k": "deployed selection out['traj'], ego kept (T1 open loop, self-action)",
            "os_w": "deployed selection out['traj'], ego withheld (T1 open loop)",
            "orc_k": "anchor_traj[a_star] with a_star = GT-nearest anchor on the KEPT bank (T0 oracle selection)",
            "orc_w": "anchor_traj[a_star] with a_star = GT-nearest anchor on the WITHHELD 10 m/s bank (T0)",
            "ha": "hold the action that closes at t0 (steer units, programme integrator)",
            "ha0": "constant velocity at measured v0, straight (a=0, kappa=0)"},
        "grid": {"slots_steps": horizons, "instants_s": [h * DT for h in horizons],
                 "dt_frame_s": DT, "n_windows": n_done, "n_skipped": n_skip,
                 "skip_reasons": skip_reasons, "n_episodes": len(episodes_manifest),
                 "window_stride": stride, "obs_window": W,
                 "ws_is": "PROVIDER frame index of the window origin t0 = t + window - 1; "
                          f"RAW frame = ws + {raw_off}",
                 "window_selection_rule": "identical to refcv3_arm.run_dump --grid 2s "
                                          "--window-stride 5 (2 s slots valid, t0 >= 1)"},
        "corpus": {"episodes": a.episodes, "labels": a.labels, **join},
        "episodes": episodes_manifest,
        "stamp": {"evidence_class": "MEASURED (ours)", "tier": "T1 for os_*/ha/ha0, T0 for orc_*",
                  "scope": "EARLY-TRAINING DIAGNOSTIC of refcv4b-b1-v72-40k at the pulled checkpoint; "
                           "NOT a capability claim; NOT quotable as driving performance",
                  "loop": "OPEN LOOP (PI ruling 2026-09-02)"},
        "wallclock_s": round(time.time() - t_start, 1), "first_forward_s": t_first,
        "torch": torch.__version__, "device": dev, "dtype": "float32, no autocast",
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, default=str)
    p(f"[dump] {n_done} windows / {len(episodes_manifest)} episodes ({n_skip} skipped: "
      f"{skip_reasons or 'none'}) in {time.time() - t_start:.0f}s")
    p("EGODROP_DUMP_DONE")


if __name__ == "__main__":
    main()
