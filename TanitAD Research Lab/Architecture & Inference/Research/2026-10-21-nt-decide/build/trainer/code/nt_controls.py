#!/usr/bin/env python3
"""nt_controls -- the CPU-checkable halves of H-NT-DECIDE-1's controls K0-K7 (PREREG_NT_DECIDE sec. 6).

Each subcommand writes ONE JSON verdict (``--out``) and exits 0 on PASS, 3 on FAIL. A mutation of the defect each
control exists for must turn it RED -- ``--mutate`` runs that arm and the expected verdict is FAIL.

* ``k0-weights``  : the trainer's step-0 RAW and EMA saves are tensor-for-tensor, BIT-equal to theta_DEC
                    (``ckpt_wiseft_mod2_DEC.pt``, md5 dad4cb5b...). The bridge half (R8_LEGAL rows bit-identical to the
                    banked MOD2 rows) is ``nt_k0_bridge.sh`` on Thor's GPU.
* ``k1-feed``     : the TRAIN item of a token-episode vs the NavSim INFERENCE feed of the same token: frames sha16 =
                    E1's recorded t0 stitch sha16 for every window slot; ``pose_hist`` = the bridge's
                    ``ego_history_poses`` (E1 wrote it by calling the bridge -- re-checked here against the log
                    through an independent SE(2) derivation, tolerance 2e-3 m at the 2 Hz knots); ``nav_cmd`` = the
                    bridge's NAVSIM_LEGAL_NAV_CMD of the log's ``driving_command`` (a LITERAL pinned against the bridge
                    SOURCE text); the speed row = (0, 0); the lift camera from the train extrinsics table = the
                    bridge's ``rig6.rig_camera`` (R, t max |d| <= 1e-12) and the lift grid through the model's own
                    ``build_lift_geometry`` (max |d| <= 1e-5). Mutation ``wrong_token``: the neighbour token's
                    stitch / pose / command -> RED.
* ``k2-targets``  : the 2 Hz trajectory targets (rows 9 + 5k of the episode poses, k = 1..8) vs the log's own SE(2)
                    future (the devkit's ``convert_absolute_to_relative_se2_array`` arithmetic, re-implemented
                    independently from ego2global), <= 1e-3 m; analytic label controls: a straight constant-speed log
                    -> LANE_KEEP (+ lon CRUISE allowed), a left circle R = 20 m at 8 m/s -> TURN_L; and the FOLLOW
                    evidence (2026-10-21 ~23:55): straight with EMPTY anns -> hard CRUISE (allowed = {CRUISE}), a
                    vehicle 15 m ahead -> FOLLOW, on a left circle R = 40 m too. Mutation adds the lead box's cy
                    negated on the circle -> not FOLLOW -> RED.
* ``k3-heldout``  : the training token table holds ZERO held-out logs; mutation: one selection-scan log added ->
                    the trainer's own guard REFUSES.
* ``k4-map-zero`` : from a smoke run's metrics: the 10 cm map loss read 0 labelled windows and its classifier
                    group took EXACTLY 0.0 gradient (``ga_*`` rows).
* ``k6-self-pair``: the paired episode-cluster bootstrap of an arm against ITSELF reads delta 0.0, CI [0, 0].
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nt_common as C  # noqa: E402

NT_TREE = Path(os.environ.get("NT_TREE") or (HERE.parent / "nt_tree"))  # Thor: the launch tree (thor_build_a1.sh exports it)
BRIDGE8 = NT_TREE / "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-19-refcv8-navsim/code/refcv8_bridge.py"
RIG6_DIR = NT_TREE / "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/navsim/code"


def _stack():
    st = str(NT_TREE / "stack")
    for p in (st, st + "/scripts"):
        if p not in sys.path:
            sys.path.insert(0, p)


def _verdict(out, name: str, ok: bool, det: dict) -> int:
    rec = {"control": name, "PASS": bool(ok), **det, "tool": {"nt_controls.py": C.md5_file(__file__)}}
    if out:
        C.json_dump(rec, out)
    print(json.dumps({"control": name, "PASS": bool(ok), **{k: v for k, v in det.items()
                                                            if not isinstance(v, (list, dict))}}))
    return 0 if ok else 3


# --------------------------------------------------------------------------------------------------- K0
def _sd(path):
    import torch
    d = torch.load(str(path), map_location="cpu", weights_only=False)
    return d.get("model", d) if isinstance(d, dict) else d


def cmd_k0(a) -> int:
    import torch
    ref = _sd(a.theta_dec)
    det = {"theta_dec": a.theta_dec, "theta_dec_md5": C.md5_file(a.theta_dec), "files": {}}
    ok = det["theta_dec_md5"] == "dad4cb5b5fb316922b4e9a7e85cee8f8"
    for tag, p in (("raw", a.ckpt0), ("ema", a.ema0)):
        if not p:
            continue
        sd = _sd(p)
        if a.mutate and tag == "raw":
            k0 = sorted(k for k, v in sd.items() if torch.is_floating_point(v) and v.numel())[0]
            sd[k0] = sd[k0].clone()
            sd[k0].view(-1)[0] += 1e-6
            det["mutation"] = f"+1e-6 on {k0}[0]"
        only_ref = sorted(set(ref) - set(sd))
        only_new = sorted(set(sd) - set(ref))
        neq = [k for k in sorted(set(ref) & set(sd))
               if ref[k].dtype != sd[k].dtype or ref[k].shape != sd[k].shape or not torch.equal(ref[k], sd[k])]
        det["files"][tag] = {"path": p, "md5": C.md5_file(p), "n_keys": len(sd), "only_in_theta_dec": only_ref[:10],
                             "only_in_save": only_new[:10], "n_unequal": len(neq), "unequal": neq[:10]}
        ok = ok and not only_ref and not only_new and not neq
    return _verdict(a.out, "K0-weights", ok, det)


# --------------------------------------------------------------------------------------------------- K1
def _bridge_literal() -> dict:
    """NAVSIM_LEGAL_NAV_CMD as WRITTEN in refcv8_bridge.py (parsed from the source text: an independent read)."""
    import ast
    src = BRIDGE8.read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "NAVSIM_LEGAL_NAV_CMD" for t in node.targets):
            return ast.literal_eval(node.value)
    raise SystemExit("[k1] NAVSIM_LEGAL_NAV_CMD not found in refcv8_bridge.py")


def _se2_rel(f_ref: dict, f: dict):
    """Pose of log frame ``f`` in the ego frame of log frame ``f_ref`` (x fwd, y left, yaw), the devkit's arithmetic
    (``convert_absolute_to_relative_se2_array`` on ``StateSE2(ego2global_translation[:2], devkit yaw)``),
    re-implemented here from the log fields."""
    X0, Y0 = (float(v) for v in np.asarray(f_ref["ego2global_translation"], np.float64)[:2])
    X, Y = (float(v) for v in np.asarray(f["ego2global_translation"], np.float64)[:2])
    h0, h = C.devkit_yaw(f_ref["ego2global_rotation"]), C.devkit_yaw(f["ego2global_rotation"])
    c, s = math.cos(h0), math.sin(h0)
    dx, dy = X - X0, Y - Y0
    return c * dx + s * dy, -s * dx + c * dy, math.atan2(math.sin(h - h0), math.cos(h - h0))


def cmd_k1(a) -> int:
    """The TRAIN item of each token-episode vs the INFERENCE feed of the same token.

    Train side = what the trainer READS: the v2 provider (``build_v2_providers``, E1's patched loader) for frames
    and ``pose_hist`` at window t = 0 (the only window ``--window-starts first`` keeps), the v9 release's
    ``row__nav_token`` (``--r8-nav-from-v9``), the road-type speed sidecar row, and the per-clip camera the trainer
    builds (``refc_v3_train._read_rig_extrinsics`` -> ``RigCamera.from_extrinsics``).
    Inference side = the bridge's OWN functions / literals: ``NAVSIM_LEGAL_NAV_CMD`` parsed from
    ``refcv8_bridge.py``'s source text, ``rig6.rig_camera`` on E1's ``frames416.rig_record`` and the bridge road
    plane, the speed UNKNOWN row literal ``(0, 0)``; poses re-derived from the LOG's ego2global (independent SE(2)
    arithmetic) at the 2 Hz knots; frames = E1's recorded t0 stitch sha16 (E1-K1a proved it = the r8ns bank row)."""
    _stack()
    from tanitad.data.v2_dataset import build_v2_providers, stable_episode_id
    det: dict = {"cache": a.cache, "scope": "provider-level train read at window t = 0 (V3Dataset slices "
                 "frames[t:t+8] / poses[t:t+8]); the lift is compared at CAMERA level and through "
                 "build_lift_geometry with one fixed parameter set (rig6's heights, stride 16, observed mask)"}
    lit = _bridge_literal()
    det["bridge_literal"] = lit
    lit_ok = lit == {"left": 1, "forward": 0, "right": 2, "unknown": 0}       # the brief's literal
    man = {json.loads(l)["clip_id"]: json.loads(l) for l in open(a.e1_manifest, encoding="utf-8") if l.strip()}
    tt = {r["clip_id"]: r for r in json.load(open(a.token_table, encoding="utf-8"))["rows"]}
    z = np.load(a.v9, allow_pickle=False)
    sid2row = {int(s): int(r0) for s, r0 in zip(z["clip__sid"], z["clip__row0"])}
    v9_nav = z["row__nav_token"]
    spd = {}
    for l in open(a.speed, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            spd[int(r["sid"])] = r
    if str(RIG6_DIR) not in sys.path:
        sys.path.insert(0, str(RIG6_DIR))
    import refcv6_bridge as B6
    eps = build_v2_providers(a.cache, lru_size=2, verbose=False)
    sid2ep = {int(e.episode_id): e for e in eps}
    cids = sorted(c for c in man if stable_episode_id(c) in sid2ep)
    if a.limit:
        cids = cids[: a.limit]
    logs: dict = {}
    n = n_bad = 0
    worst_pose = 0.0
    bad = []
    nav_hist: dict = {}
    for i, cid in enumerate(cids):
        src = cids[(i + 1) % len(cids)] if a.mutate else cid       # wrong_token: the neighbour token's feed
        m, r = man[src], tt[src]
        sid = stable_episode_id(cid)
        ep = sid2ep[sid]
        fr = ep.frames[0:8]                                          # [8, 3 * n_stack, H, W] u8
        if fr.shape[1] != 9 or tuple(fr.shape[2:]) != (416, 1024):
            raise SystemExit(f"[k1] {cid}: provider frames {tuple(fr.shape)}, expected [8, 9, 416, 1024]")
        shas = set()
        for g in range(3):
            x = fr[:, 3 * g:3 * g + 3].permute(0, 2, 3, 1).contiguous().numpy()
            shas |= {hashlib.sha256(np.ascontiguousarray(xx).tobytes()).hexdigest()[:16] for xx in x}
        frame_ok = shas == {m["t0_frame_sha16"]}
        ln = r["log_name"]
        if ln not in logs:
            logs[ln] = C.load_log(Path(a.logs_dir) / f"{ln}.pkl")
        L = logs[ln]
        j = int(r["frame_idx"])
        p = ep.poses.double().numpy()                                # provider row k = raw row k + 2
        e_now = float(np.abs(p[7, :3]).max())                        # raw row 9 = NOW = the t0 ego origin
        # pose_hist (provider rows 0..7 = t0-0.7 .. t0) vs the BRIDGE's ego_history_poses fed a declaration built
        # from the LOG here (SE(2) relative poses, ego_dynamic_state velocity, actual 2 Hz times) -- E1 fed it the
        # devkit export instead, so agreement is two routes into one function, not one route read twice
        decl, times = {}, []
        for ii, jj in enumerate(range(j - 3, j + 1)):
            jj_ = max(jj, 0)
            x_, y_, h_ = _se2_rel(L[j], L[jj_])
            decl[f"ego_pose[{ii}]"] = [float(v) for v in np.asarray([x_, y_, h_], np.float32)]   # EgoStatus is f32
            decl[f"ego_velocity[{ii}]"] = [float(v) for v in np.asarray(L[jj_]["ego_dynamic_state"], np.float32)[:2]]
            times.append((int(L[jj_]["timestamp"]) - int(L[j]["timestamp"])) / 1e6)
        try:
            hist = np.asarray(B6.ego_history_poses(decl, times), np.float64)
            e_m1 = float(np.abs(hist - p[0:8, :4]).max())
        except Exception as ex:  # noqa: BLE001 -- a refusal is a FAIL with its reason
            e_m1 = float("inf")
            det.setdefault("hist_refused", []).append(f"{cid}: {type(ex).__name__}: {ex}"[:200])
        pose_ok = e_now <= 1e-6 and e_m1 <= a.pose_tol
        worst_pose = max(worst_pose, e_now, e_m1)
        dc = np.asarray(L[j]["driving_command"]).astype(int).tolist()
        if sorted(dc) != [0, 0, 0, 1]:
            raise SystemExit(f"[k1] {cid}: the log's driving_command {dc} is not one-hot")
        key = {0: "left", 1: "forward", 2: "right", 3: "unknown"}[int(np.argmax(dc))]
        want_nav = int(lit[key])
        got_nav = float(v9_nav[sid2row[sid]])
        nav_ok = got_nav == float(want_nav)
        nav_hist[key] = nav_hist.get(key, 0) + 1
        sr = spd.get(sid)
        speed_ok = (sr is not None and int(sr["valid"]) == 0 and list(map(float, sr["one_hot"])) == [0.0] * 4)
        n += 1
        if not (frame_ok and pose_ok and nav_ok and speed_ok):
            n_bad += 1
            bad.append({"clip": cid, "frame_ok": frame_ok, "pose_ok": pose_ok, "nav_ok": nav_ok,
                        "speed_ok": speed_ok, "e_now": e_now, "e_m1": e_m1, "nav_v9": got_nav,
                        "nav_bridge": want_nav})
    # the camera: what the TRAINER builds from the table vs what the BRIDGE builds from the rig record
    for pth in (str(RIG6_DIR), str(NT_TREE / "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-refcv4b-bridge/code")):
        if pth not in sys.path:
            sys.path.insert(0, pth)
    import rig6
    from tanitad.data.rig_projection import RigCamera
    from tanitad.models.trunk_shapes import FRAME_416x1024
    from tanitad.models.bev_lift import build_lift_geometry
    from tanitad.data.bev_raster import GRID_DEFAULT
    import refc_v3_train as TR
    _e, table = TR._read_rig_extrinsics(a.extrinsics)
    rigs = json.load(open(a.rigs, encoding="utf-8"))
    cam_worst, lift_worst, lift_vm, n_cam = 0.0, 0.0, 0, 0
    obs = rig6.observed_mask(FRAME_416x1024)
    for i, cid in enumerate(cids[: max(1, a.n_lift)]):
        src = cids[(i + 1) % len(cids)] if a.mutate else cid
        rr = rigs[man[src]["rig_key"]]
        rr = rr.get("rig_record", rr)
        ct = RigCamera.from_extrinsics(table[cid], FRAME_416x1024)
        cb = rig6.rig_camera(rr, C.ROAD_PLANE_MEDIAN_M, FRAME_416x1024)
        cam_worst = max(cam_worst, float((ct.R_cam_to_rig - cb.R_cam_to_rig).abs().max()),
                        float((ct.t_cam_in_rig - cb.t_cam_in_rig).abs().max()))
        ga = build_lift_geometry(ct, frame=FRAME_416x1024, stride=16, heights_m=tuple(rig6.HEIGHTS_M),
                                 grid=GRID_DEFAULT, observed=obs)
        gb = build_lift_geometry(cb, frame=FRAME_416x1024, stride=16, heights_m=tuple(rig6.HEIGHTS_M),
                                 grid=GRID_DEFAULT, observed=obs)
        both = (ga.valid & gb.valid)[..., None]
        lift_worst = max(lift_worst, float(((ga.grid - gb.grid).abs() * both).max()))
        lift_vm += int((ga.valid != gb.valid).sum())
        n_cam += 1
    det.update({"n_items": n, "n_bad": n_bad, "bad": bad[:8], "worst_pose_abs": worst_pose,
                "nav_keys": nav_hist, "nav_literal_ok": lit_ok, "n_cameras": n_cam,
                "camera_worst_abs": cam_worst, "lift_grid_worst_abs": lift_worst,
                "lift_valid_mismatch": lift_vm, "road_plane_m": C.ROAD_PLANE_MEDIAN_M,
                "tolerances": {"frames": "sha16 equal", "pose_now": 1e-6, "pose_hist_vs_bridge": a.pose_tol,
                               "camera": 1e-12, "lift_grid": 1e-5},
                "mutation": "wrong_token (the neighbour token's feed)" if a.mutate else None})
    ok = (lit_ok and n > 0 and n_bad == 0 and cam_worst <= 1e-12 and lift_worst <= 1e-5 and lift_vm == 0)
    return _verdict(a.out, "K1-feed", ok, det)


# --------------------------------------------------------------------------------------------------- K2
def _synth_pose(kind: str, s: float, R: float) -> tuple:
    if kind == "straight":
        return s, 0.0, 0.0
    th = s / R
    return R * math.sin(th), R * (1 - math.cos(th)), th


def _synth_log(kind: str, *, v: float = 8.0, R: float = 20.0, n: int = 60, mirror: bool = False,
               anns: bool = False, lead_dx: float | None = None, flip_cy: bool = False) -> list:
    """A synthetic 2 Hz log in the OpenScene frame-dict layout (translation + (w,x,y,z) quaternion, zero pitch/roll):
    ``straight`` = constant ``v`` along +x; ``left_circle`` = a circle of radius ``R`` turning LEFT (y > 0) at ``v``.
    ``mirror`` reflects y and yaw -- the y-left / y-right convention defect, for the mutation arm.
    ``anns`` adds OpenScene ``anns`` per frame (EMPTY unless ``lead_dx``: one ``vehicle`` that many metres AHEAD along
    the ego's own path, moving with it, in each frame's ego frame x fwd / y LEFT); ``flip_cy`` negates that box's y
    (the agent-side y convention defect, for the mutation arm)."""
    frs = []
    for i in range(n):
        t = 0.5 * i
        x, y, yaw = _synth_pose(kind, v * t, R)
        if mirror:
            y, yaw = -y, -yaw
        fr = {"timestamp": int(1_000_000 + round(t * 1e6)), "token": f"syn{i:04d}",
              "ego2global_translation": [x, y, 0.0],
              "ego2global_rotation": [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)],
              "ego_dynamic_state": [v, 0.0, 0.0, 0.0], "driving_command": [0, 1, 0, 0]}
        if anns:
            boxes, names = [], []
            if lead_dx is not None:
                lx, ly, lyaw = _synth_pose(kind, v * t + lead_dx, R)
                dx, dy = lx - x, ly - y
                cx = math.cos(yaw) * dx + math.sin(yaw) * dy
                cy = -math.sin(yaw) * dx + math.cos(yaw) * dy
                boxes.append([cx, -cy if flip_cy else cy, 0.8, 4.5, 1.9, 1.6, lyaw - yaw])
                names.append("vehicle")
            fr["anns"] = {"gt_boxes": np.asarray(boxes, dtype=np.float64).reshape(-1, 7), "gt_names": names,
                          "track_tokens": ["0123456789abcdef"] * len(names)}
        frs.append(fr)
    return frs


def cmd_k2(a) -> int:
    """(a) the 2 Hz trajectory targets the trainer reads (E1 poses at raw rows 9 + 5k, k = 1..8 = 0.5 .. 4.0 s) vs
    the DEVKIT's own future (``human_future_poses`` of the r8ns export = ``Scene.get_future_trajectory``, when
    ``--devkit-inputs`` is given) AND vs the log's SE(2) future re-derived here; (b) analytic label literals through
    the real builder path (``nt_labels.labels_for_log``): straight constant speed -> LANE_KEEP with CRUISE allowed;
    left circle R = 20 m at 8 m/s -> TURN_L. Mutation: targets +1 cm AND the circle mirrored (y-right) -> RED."""
    _stack()
    import torch
    det: dict = {}
    tt = {r["clip_id"]: r for r in json.load(open(a.token_table, encoding="utf-8"))["rows"]}
    dk = {}
    if a.devkit_inputs:
        dk = json.load(open(a.devkit_inputs, encoding="utf-8"))["records"]
    logs: dict = {}
    w_log = w_dk = 0.0
    n = nk_log = nk_dk = n_dk_missing = 0
    for f in sorted(Path(a.cache).glob("*.v2ep.pt")):
        cid = f.name[: -len(".v2ep.pt")]
        d = torch.load(str(f), map_location="cpu", weights_only=False)
        poses = d["poses"].double().numpy()                       # raw rows, t0-ego frame (x fwd, y left)
        r = tt[cid]
        ln = r["log_name"]
        if ln not in logs:
            logs[ln] = C.load_log(Path(a.logs_dir) / f"{ln}.pkl")
        L = logs[ln]
        j = int(r["frame_idx"])
        ts = [int(x["timestamp"]) for x in L]
        rec = dk.get(r["token"])
        if a.devkit_inputs and rec is None:
            n_dk_missing += 1
        for k in range(1, 9):
            row = C.NOW_RAW_ROW + 5 * k
            if row >= poses.shape[0]:
                break
            px, py = poses[row, 0] + (0.01 if a.mutate else 0.0), poses[row, 1]
            if rec is not None and k - 1 < len(rec["human_future_poses"]):
                hx, hy = rec["human_future_poses"][k - 1][:2]
                w_dk = max(w_dk, abs(px - hx), abs(py - hy))
                nk_dk += 1
            if j + k < len(L) and abs((ts[j + k] - ts[j]) / 1e6 - 0.5 * k) <= 0.05 * k:
                x, y, _ = _se2_rel(L[j], L[j + k])
                w_log = max(w_log, abs(px - x), abs(py - y))
                nk_log += 1
        n += 1
    det["targets"] = {"n_tokens": n, "vs_devkit": {"n_knots": nk_dk, "worst_abs_m": w_dk,
                                                     "n_tokens_without_record": n_dk_missing},
                      "vs_log_se2": {"n_knots": nk_log, "worst_abs_m": w_log}, "tol_m": 1e-3,
                      "horizons_s": [0.5 * k for k in range(1, 9)]}
    ok_t = n > 0 and nk_log > 0 and w_log <= 1e-3 and (not a.devkit_inputs or (nk_dk > 0 and w_dk <= 1e-3
                                                                                and n_dk_missing == 0))
    import nt_labels as NL
    BV = NL._bv()
    res = {}
    for kind, synth_kind in (("straight", "straight"), ("left_circle_R20_v8", "circle")):
        frs = _synth_log(synth_kind, mirror=bool(a.mutate) and synth_kind == "circle")
        idx = 10
        out = NL.labels_for_log(BV, frs, [{"token": frs[idx]["token"], "frame_idx": idx}])["by_token"][frs[idx]["token"]]
        lat, lon = int(out["lat_v7id_a"]), int(out["lon_v7id"])
        lon_al = int(out["lon_allowed_v7"])
        res[kind] = {"lat": NL.LAT7[lat] if 0 <= lat < len(NL.LAT7) else f"IGNORE({lat})",
                     "lon": NL.LON7[lon] if 0 <= lon < len(NL.LON7) else f"IGNORE({lon})",
                     "lon_allowed": [NL.LON7[i] for i in range(8) if (lon_al >> i) & 1]}
    want = {"straight": "LANE_KEEP", "left_circle_R20_v8": "TURN_L"}                # literals (PREREG K2)
    ok_l = all(res[k]["lat"] == want[k] for k in want) and \
        ("CRUISE" in res["straight"]["lon_allowed"] or res["straight"]["lon"] == "CRUISE")
    # (c) the FOLLOW evidence (lead pass, 2026-10-21 ~23:55) through the same builder path: literals
    lres = {}
    for kind, synth_kind, R, ld in (("straight_empty_anns", "straight", 20.0, None),
                                    ("straight_lead15", "straight", 20.0, 15.0),
                                    ("left_circle_R40_lead15", "circle", 40.0, 15.0)):
        frs = _synth_log(synth_kind, R=R, anns=True, lead_dx=ld,
                         flip_cy=bool(a.mutate) and synth_kind == "circle")
        out = NL.labels_for_log(BV, frs, [{"token": frs[10]["token"], "frame_idx": 10}])["by_token"][frs[10]["token"]]
        lon, lon_al = int(out["lon_v7id"]), int(out["lon_allowed_v7"])
        lres[kind] = {"lon": NL.LON7[lon] if 0 <= lon < len(NL.LON7) else f"IGNORE({lon})",
                      "lon_allowed": [NL.LON7[i] for i in range(8) if (lon_al >> i) & 1],
                      "lead_gap_m": float(out.get("lead_gap_m", np.nan))}
    want_lon = {"straight_empty_anns": "CRUISE", "straight_lead15": "FOLLOW", "left_circle_R40_lead15": "FOLLOW"}
    ok_lon = all(lres[k]["lon"] == want_lon[k] for k in want_lon) and lres["straight_empty_anns"]["lon_allowed"] == ["CRUISE"]
    det["analytic"] = {"got": res, "want_lat": want, "want_straight_lon": "CRUISE (allowed)",
                       "lead_got": lres, "want_lead_lon": want_lon}
    det["mutation"] = ("targets +1 cm in x; left circle mirrored to y-right (the y-axis convention defect); the lead "
                       "box's cy negated on the circle (the agent-side y defect)" if a.mutate else None)
    return _verdict(a.out, "K2-targets", ok_t and ok_l and ok_lon, det)


# --------------------------------------------------------------------------------------------------- K3
def cmd_k3(a) -> int:
    _stack()
    from tanitad.train import nt_guard as G
    guard = G.load_guard(a.heldout, expect_sha256=a.heldout_sha256)
    rows = json.load(open(a.token_table, encoding="utf-8"))["rows"]
    logs = [r["log_name"] for r in rows]
    det = {"n_rows": len(rows), "n_logs": len(set(logs)), "n_heldout_logs": len(guard["heldout_logs"])}
    if a.mutate:
        logs = logs + [guard["selection_scan_logs"][0]]
        det["mutation"] = f"+ selection log {guard['selection_scan_logs'][0]}"
    try:
        cen = G.assert_disjoint(logs, guard, where="K3")
        det["census"] = cen
        ok = cen["n_heldout_hits"] == 0
    except SystemExit as e:
        det["refused"] = str(e)[:400]
        ok = False
    return _verdict(a.out, "K3-heldout", ok, det)


# --------------------------------------------------------------------------------------------------- K4
#: the 10 cm map's MAP-ONLY parts (map_head_hires.grad_reach_report_hires) and its parts SHARED with the planner BEV
K4_ZERO = ("ga_mh_refine", "ga_mh_near", "ga_mh_near_refine")
K4_LIVE = ("ga_mh_lift", "ga_mh_encoder", "ga_mh_trunk_s8_stage")
K4_MAP_ONLY_PREFIXES = ("_map_hires.refine.", "_map_hires.near.", "_map_hires.near_refine.")


def cmd_k4(a) -> int:
    """K4 (PREREG sec. 6): the 10 cm map head is OFF by data, not by architecture. On every logged TRAIN row of a run:
    the map-only parts' gradient reach (``ga_mh_refine`` / ``ga_mh_near`` / ``ga_mh_near_refine``, the trainer's own
    ``grad_abs_sum`` after backward) is EXACTLY 0.0 with their gradient tensors present (``_n`` = all parameters), the
    shared parts (lift / encoder / the trunk's s8 stage) are > 0 (the planner still trains them), and the map loss read
    ZERO labelled windows. With ``--ckpt0`` / ``--ckpt-final``: the map-only parameters' relative L2 change (AdamW's
    decoupled weight decay still moves a zero-gradient parameter) is REPORTED. Mutation: one row's ``ga_mh_refine``
    set to 1e-9 (map supervision leaking in) -> RED."""
    rows = [json.loads(l) for l in open(Path(a.run) / "metrics.jsonl", encoding="utf-8") if l.strip()]
    tr = [r for r in rows if isinstance(r.get("step"), int) and "ga_mh_lift" in r]
    if a.mutate and tr:
        tr[-1] = dict(tr[-1], ga_mh_refine=1e-9)
    det: dict = {"run": a.run, "n_train_rows": len(tr), "steps": [r["step"] for r in tr]}
    bad = []
    for r in tr:
        for k in K4_ZERO:
            v = r.get(k)
            n_with = r.get(f"{k}_n")
            if v is None or not isinstance(v, (int, float)) or float(v) != 0.0:
                bad.append((r["step"], k, v))
            elif not n_with:
                bad.append((r["step"], f"{k}_n", n_with))      # 0 tensors with grad = graph=no: not the identity
        for k in K4_LIVE:
            v = r.get(k)
            if v is None or not (isinstance(v, (int, float)) and math.isfinite(float(v)) and float(v) > 0):
                bad.append((r["step"], k, v))
        if float(r.get("map_hires_n_labelled", -1)) != 0.0:
            bad.append((r["step"], "map_hires_n_labelled", r.get("map_hires_n_labelled")))
    det["zero_keys"] = {k: [r.get(k) for r in tr] for k in K4_ZERO}
    det["zero_keys_n_params_with_grad"] = {k: [r.get(f"{k}_n") for r in tr] for k in K4_ZERO}
    det["live_keys"] = {k: [r.get(k) for r in tr] for k in K4_LIVE}
    det["map_hires_n_labelled"] = [r.get("map_hires_n_labelled") for r in tr]
    det["violations"] = bad[:20]
    if a.ckpt0 and a.ckpt_final:
        import torch
        s0, s1 = _sd(a.ckpt0), _sd(a.ckpt_final)
        num = den = 0.0
        n_t = 0
        for k, v in s0.items():
            if k.startswith(K4_MAP_ONLY_PREFIXES) and torch.is_floating_point(v) and k in s1:
                num += float(((s1[k].double() - v.double()) ** 2).sum())
                den += float((v.double() ** 2).sum())
                n_t += 1
        det["map_only_rel_l2_change"] = {"value": math.sqrt(num / den) if den > 0 else None, "n_tensors": n_t,
                                         "from": a.ckpt0, "to": a.ckpt_final,
                                         "note": "reported, not gated: AdamW decoupled weight decay (wd 1e-4) moves "
                                                 "a zero-gradient parameter by lr * wd per step"}
    det["mutation"] = "ga_mh_refine = 1e-9 on the last row" if a.mutate else None
    ok = bool(tr) and not bad
    return _verdict(a.out, "K4-map-zero", ok, det)


K7_TRUNK = "core.encoder."


def cmd_k7(a) -> int:
    """K7 (PREREG sec. 6): the trunk actually trained. Relative L2 drift of the trunk (``core.encoder.*``, the 316
    tensors the conflict detector calls theta_trunk) from theta_DEC to each given snapshot, overall and per top-level
    stage: must be finite and > 0 (reported, no bar). With ``--metrics``: the P-GRAD rows (``gs_*``) must exist at
    every ``--grad-share-at`` step that the run reached. Mutation: theta_DEC against ITSELF -> drift 0.0 -> RED."""
    import torch
    ref = _sd(a.theta_dec)
    det: dict = {"theta_dec": a.theta_dec, "theta_dec_md5": C.md5_file(a.theta_dec), "snapshots": {}}
    ok = True
    snaps = [a.theta_dec] if a.mutate else list(a.snapshots)
    for sp in snaps:
        sd = _sd(sp)
        num = den = 0.0
        per: dict = {}
        n_t = 0
        for k, v in ref.items():
            if not k.startswith(K7_TRUNK) or not torch.is_floating_point(v) or k not in sd:
                continue
            d2 = float(((sd[k].double() - v.double()) ** 2).sum())
            r2 = float((v.double() ** 2).sum())
            num, den, n_t = num + d2, den + r2, n_t + 1
            stage = k[len(K7_TRUNK):].split(".")[1] if k[len(K7_TRUNK):].startswith("net.") else k[len(K7_TRUNK):].split(".")[0]
            e = per.setdefault(stage, [0.0, 0.0])
            e[0] += d2
            e[1] += r2
        rel = math.sqrt(num / den) if den > 0 else float("nan")
        det["snapshots"][sp] = {"trunk_rel_l2": rel, "n_tensors": n_t,
                                "per_stage": {s: (math.sqrt(x / y) if y > 0 else None) for s, (x, y) in sorted(per.items())}}
        ok = ok and n_t > 0 and math.isfinite(rel) and rel > 0.0
    if a.metrics:
        rows = [json.loads(l) for l in open(a.metrics, encoding="utf-8") if l.strip()]
        last = max((r.get("step", -1) for r in rows if isinstance(r.get("step"), int)), default=-1)
        want = [int(x) for x in str(a.grad_share_at or "").split(",") if x.strip()]
        have = {}
        for s in want:
            if s > last:
                have[s] = "not reached"
                continue
            row = next((r for r in rows if r.get("step") == s and any(k.startswith("gs_") for k in r)), None)
            have[s] = ({k: v for k, v in row.items() if k.startswith("gs_trunk_")} if row else None)
            ok = ok and row is not None
        det["p_grad_rows"] = have
    det["mutation"] = "theta_DEC vs itself" if a.mutate else None
    return _verdict(a.out, "K7-trunk-trained", ok, det)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("k0-weights")
    p.add_argument("--theta-dec", required=True)
    p.add_argument("--ckpt0", required=True)
    p.add_argument("--ema0", default=None)
    p.add_argument("--mutate", action="store_true")
    p.add_argument("--out", default=None)
    p = sub.add_parser("k1-feed")
    for k in ("--cache", "--e1-manifest", "--token-table", "--logs-dir", "--extrinsics", "--rigs", "--v9", "--speed"):
        p.add_argument(k, required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--n-lift", type=int, default=3)
    p.add_argument("--pose-tol", type=float, default=1e-6, help="PREREG K1 literal: pose_hist <= 1e-6")
    p.add_argument("--mutate", action="store_true")
    p.add_argument("--out", default=None)
    p = sub.add_parser("k2-targets")
    for k in ("--cache", "--token-table", "--logs-dir"):
        p.add_argument(k, required=True)
    p.add_argument("--e1-manifest", default=None)
    p.add_argument("--devkit-inputs", default=None, help="the r8ns export records (thor_pull_devkit_inputs.py)")
    p.add_argument("--mutate", action="store_true")
    p.add_argument("--out", default=None)
    p = sub.add_parser("k3-heldout")
    p.add_argument("--heldout", required=True)
    p.add_argument("--heldout-sha256", required=True)
    p.add_argument("--token-table", required=True)
    p.add_argument("--mutate", action="store_true")
    p.add_argument("--out", default=None)
    p = sub.add_parser("k4-map-zero")
    p.add_argument("--run", required=True)
    p.add_argument("--ckpt0", default=None)
    p.add_argument("--ckpt-final", default=None)
    p.add_argument("--mutate", action="store_true")
    p.add_argument("--out", default=None)
    p = sub.add_parser("k7-trunk")
    p.add_argument("--theta-dec", required=True)
    p.add_argument("--snapshots", nargs="+", required=True)
    p.add_argument("--metrics", default=None)
    p.add_argument("--grad-share-at", default="1000,6250,12450")
    p.add_argument("--mutate", action="store_true")
    p.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    return {"k0-weights": cmd_k0, "k1-feed": cmd_k1, "k2-targets": cmd_k2, "k3-heldout": cmd_k3,
            "k4-map-zero": cmd_k4, "k7-trunk": cmd_k7}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
