#!/usr/bin/env python3
"""render_refcv7_replay.py -- an INTERACTIVE REPLAY TOOL and a TACTICAL video for refcv7-r101-s0 (step 50,400).

PI (Sayed), verbatim, after the first reel: *"I need a replay tool where the different labels can be
selected and switched on/off. It is very hard to recognize what is GT and what is from the model. I need
a video where we can see the outputted tactical behavior and goals. The fan is sometimes completely
messy."*

⛔ READ ``RENDER_REFCV7_REPLAY.md`` (beside this file) FIRST. This is OPEN-LOOP perception + planning on the
held-out eval139 clips -- never closed-loop driving, never a four-family eval result -- and the speed
ceiling does NOT reach the emitted plan on this launch tree (SPEC_REFCV7 26.1).

STAGES (one file; ``taniteval/tools/render_refcv7_video.py`` is imported for its pure helpers)
---------------------------------------------------------------------------------------------
``capture``  (Thor GPU, under ``flock /home/nvidia/refcv7_post/thor_gpu.lock``) -- the reel's forward
    pass, EXTENDED with read-only reads of what the reel did not bank: the full tactical lat/lon
    probability vectors, ALL 22 goal-token validities + confidences, the spatial goal the planner uses
    (``goal_point_tac``), ALL candidate trajectories with the E9 score, the decoder's own score and the
    reach / ceiling / nav-compliance flags, and both picks. Same model build, same clip rule, same
    ``compute_losses_v3`` + non-raising forward hook, same seed-0 DDIM draw as the reel.
``assets``   (Thor CPU) -- the static tool's data: camera JPEGs, 10 cm map PNGs, per-clip ``.js`` files.
``video``    (Thor CPU) -- the 1920x1080 TACTICAL video (preset of the same drawing code).

Clip ids never leave the process: every artifact carries ``sha12 = sha256(clip_id)[:12]`` only.

The module imports numpy only at top level so every pure function below is unit-testable on any box
(``taniteval/tests/test_render_refcv7_replay.py``).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent


def _load_rv7():
    """The reel renderer, loaded by path (it imports numpy only at module level)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("render_refcv7_video_for_replay",
                                                  str(_HERE / "render_refcv7_video.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


rv7 = _load_rv7()

# ============================================================================================= #
# identity                                                                                       #
# ============================================================================================= #
RUN_LABEL = rv7.RUN_LABEL
TIER_NOTE = rv7.TIER_NOTE
#: ⭐ the stamp every planner overlay carries (brief, rule 2)
CEIL_STAMP = "speed ceiling not applied to the emitted plan (SPEC_REFCV7 26.1)"
DEFAULTS = {
    "tree": rv7.DEFAULTS["tree"],
    "run": rv7.DEFAULTS["run"],
    "kit": rv7.DEFAULTS["kit"],
    "work": "/home/nvidia/refcv7_post/replay",
}
OLD_BANK = "/home/nvidia/refcv7_post/video/bank/final"
EXPECT = dict(rv7.EXPECT)
HORIZONS = rv7.HORIZONS
MAP_CLASSES = rv7.MAP_CLASSES
CLASS_KEYS = rv7.CLASS_KEYS
NOT_SEEN = rv7.NOT_SEEN
SAVE_SLOT_P_MIN = rv7.SAVE_SLOT_P_MIN          # 0.10: the lowest box-score slider position
FPS = 10

# ============================================================================================= #
# colour semantics -- ONE RULE EVERYWHERE (tool and video)                                       #
#   GT    = GREEN, dashed or outline-only, labelled "GT"                                         #
#   MODEL = ORANGE (the emitted plan) / BLUE (candidates, decoder pick, detections), solid/filled #
# ============================================================================================= #
C_GT = rv7.C_GT                        # (110, 231, 138)
C_PLAN = rv7.C_SEL                     # (249, 115, 22)   orange: the emitted plan (E9 pick)
C_DEC = (59, 130, 246)                 # blue: the DECODER's own pick (when it differs)
C_BOX = (56, 132, 255)                 # blue: model detections (filled)
C_GIVEN = rv7.C_GIVEN                  # amber: a GIVEN input (text only)
C_GOAL = (232, 121, 249)               # fuchsia: the model's tactical goal POINT (2 s) -- MODEL
# fan colour ramp, rank 1 -> rank k: light blue -> indigo (no green, no orange)
FAN_HI, FAN_LO = (165, 225, 255), (88, 70, 214)

# ============================================================================================= #
# the plan -> (lat3, lon3) reading, by the PROGRAMME's own factorisation                          #
#   refc_tactical.factor_from_kinematics v2 rule at the label's own 2 s horizon                  #
# ============================================================================================= #
LAT3 = ("lane_keep", "turn_left", "turn_right")
LON3 = ("brake_stop", "steady", "accelerate")      # refc_tactical.LON_CLASSES order (ids 0, 1, 2)
# literals of refc_tactical.py:148-155 (asserted equal to the module's at capture time)
CURV_TURN_MAN_PER_M = 1.0 / 60.0
DV_ACCEL_MS, DV_BRAKE_MS = 1.0, -1.0
STOP_V_MS, MOVING_V_MS = 0.3, 1.0
MIN_ARC_M = 0.10
SLOT_2S = 3                            # horizons (5, 10, 15, 20, ...) -> index 3 = 2.0 s
STALL_M = 0.05                         # refc_selector_targets.compliance_target's stall_m

#: v7 tactical LAT action -> the lat3 class it can be compared with (None = no counterpart in lat3)
LAT_TO_LAT3 = {"LANE_KEEP": 0, "TURN_L": 1, "TURN_R": 2, "LANE_CHANGE_L": None, "LANE_CHANGE_R": None,
               "ABORT_LC": None, "NUDGE_L": None, "NUDGE_R": None}
#: v7 tactical LON action -> the SET of lon3 classes that count as agreeing
LON_TO_LON3 = {"FOLLOW": (1,), "CRUISE": (1,), "YIELD_MERGE": (0,), "BRAKE_TO": (0,), "CREEP": (1, 2),
               "HOLD": (0, 1), "ADAPT_SPEED_FOR_CURVE": (0, 1), "ACCELERATE": (2,)}


def plan_factor(path_xy, v0: float, valid=None) -> dict:
    """A planned (or GT) path -> ``(lat3, lon3)`` by ``refc_tactical.factor_from_kinematics`` v2.

    ``path_xy`` ``[>=4, 2]`` metres at 0.5 / 1 / 1.5 / 2 s (the first four of the plan's eight slots), ego
    frame (x forward, y LEFT), the ego at (0, 0) heading +x. The label horizon is 2 s, so is this.
    ``dyaw`` is the heading of the last 0.5 s segment (the programme's ``compliance_target`` terminal-
    heading definition; a stalled segment < 0.05 m reads 0), ``kappa = dyaw / arc`` over 0 -> 2 s,
    ``v1`` the last segment's speed, ``dv = v1 - v0``. Returns ``None`` classes when a slot is invalid."""
    p = np.asarray(path_xy, np.float64)[:4]
    if p.shape != (4, 2) or (valid is not None and not bool(np.all(np.asarray(valid)[:4]))):
        return {"lat3": None, "lon3": None, "dyaw": None, "kappa": None, "dv": None, "v1": None}
    pts = np.concatenate([np.zeros((1, 2)), p], axis=0)
    d = pts[-1] - pts[-2]
    seg = float(np.hypot(d[0], d[1]))
    dyaw = 0.0 if seg < STALL_M else float(math.atan2(d[1], d[0]))
    arc = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    kappa = dyaw / max(arc, MIN_ARC_M)
    v1 = seg / 0.5
    dv = v1 - float(v0)
    lat3 = 0
    if abs(kappa) >= CURV_TURN_MAN_PER_M:
        lat3 = 1 if dyaw > 0 else 2
    lon3 = 1                                                   # steady
    if dv > DV_ACCEL_MS:
        lon3 = 2                                               # accelerate
    if (dv < DV_BRAKE_MS) or (v1 < STOP_V_MS and float(v0) >= MOVING_V_MS):
        lon3 = 0                                               # brake_stop (overwrites accelerate, as the labeler)
    return {"lat3": lat3, "lon3": lon3, "dyaw": dyaw, "kappa": kappa, "dv": dv, "v1": v1}


def agree_tactical(lat3, lon3, lat_name: str, lon_name: str) -> dict:
    """Does the path's ``(lat3, lon3)`` agree with the tactical decoder's argmax actions?

    ``lat_ok`` / ``lon_ok`` are ``True`` / ``False``, or ``None`` when there is nothing to compare (the
    path could not be classified, or the decoder's action has no lat3 counterpart -- a nudge or a lane
    change is not a turn). The mapping tables are DATA above, not logic."""
    lt = LAT_TO_LAT3.get(lat_name)
    lat_ok = None if (lat3 is None or lt is None) else bool(lat3 == lt)
    ln = LON_TO_LON3.get(lon_name)
    lon_ok = None if (lon3 is None or ln is None) else bool(lon3 in ln)
    return {"lat_ok": lat_ok, "lon_ok": lon_ok}


def agree_nav(nav: str, lat3, navc_ok_sel) -> bool | None:
    """The emitted plan against the GIVEN nav token. ``left`` / ``right``: the programme's own
    ``compliance_target`` predicate evaluated on the emitted path (terminal heading of the commanded
    sign, |.| >= tau). ``follow`` / ``straight`` carry no commanded side: agree iff the plan does not
    turn (lat3 == lane_keep)."""
    if nav in ("left", "right"):
        return None if navc_ok_sel is None else bool(navc_ok_sel)
    return None if lat3 is None else bool(lat3 == 0)


# ============================================================================================= #
# data encoders (pure)                                                                           #
# ============================================================================================= #
def encode_map(codes: np.ndarray, lv: np.ndarray, notseen: np.ndarray | None = None) -> bytes:
    """A ``[1000, 600]`` uint8 class map + the lift-valid mask -> an 8-bit GRAYSCALE PNG whose pixel value
    carries everything the tool needs: ``class (bits 0-2) | not-seen (bit 3, SAM3 code 255) |
    lift-invalid (bit 4)``. Grayscale, not palette: the browser decodes R = G = B = value, so the client
    reads the byte straight from ``getImageData`` (a palette PNG would force a colour -> index lookup).
    ``notseen`` (optional) is OR-ed into bit 3 so a prediction file carries the GT's coverage as well."""
    from PIL import Image
    c = np.asarray(codes, np.uint8)
    ns = c == NOT_SEEN
    v = np.where(ns, 0, c).astype(np.uint8)
    if notseen is not None:                       # the prediction file also carries the GT's "never seen" flag
        ns = ns | np.asarray(notseen, bool)
    if int(v.max()) > 7:
        raise ValueError(f"map code {int(v.max())} > 7")
    v = v | (ns.astype(np.uint8) << 3) | ((~np.asarray(lv, bool)).astype(np.uint8) << 4)
    b = io.BytesIO()
    Image.fromarray(v, "L").save(b, "PNG", optimize=True, compress_level=9)
    return b.getvalue()


def decode_map(png: bytes) -> tuple:
    """Inverse of :func:`encode_map` -> ``(codes uint8 with 255 = not seen, lift_valid bool)``."""
    from PIL import Image
    v = np.asarray(Image.open(io.BytesIO(png)))
    cls = (v & 7).astype(np.uint8)
    ns = (v & 8) != 0
    lvb = (v & 16) == 0
    return np.where(ns, np.uint8(NOT_SEEN), cls), lvb


def encode_fan_cm(fan: np.ndarray) -> tuple:
    """``[N, S, 2]`` metres -> (base64 of little-endian int16 CENTIMETRES, n_clipped)."""
    cm = np.rint(np.asarray(fan, np.float64) * 100.0)
    n_clip = int((np.abs(cm) > 32767).sum())
    cm = np.clip(cm, -32767, 32767).astype("<i2")
    return base64.b64encode(cm.tobytes()).decode("ascii"), n_clip


def decode_fan_cm(b64: str, n: int, s: int = 8) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64), dtype="<i2").reshape(n, s, 2).astype(np.float64) / 100.0


def bits(mask) -> str:
    return "".join("1" if bool(x) else "0" for x in np.asarray(mask).reshape(-1))


def greedy_match(det_xy, det_p, gt_xy, gt_pos, gt_ign, dist_m: float = 2.0, min_conf: float = 0.0) -> list:
    """The programme's greedy rule (``box3d_head.box3d_match_rows`` via ``detection_metrics.greedy_rows``),
    in plain numpy so the JS port can be checked against it: detections by score descending; each takes the
    NEAREST still-untaken GT of (positives U IGNORE); a hit if within ``dist_m`` (centre distance). Returns
    ``[(slot, gt_index, kind)]`` in matching order, kind 1 = TP (a positive), -1 = DontCare (an IGNORE),
    0 = FP (gt_index -1)."""
    det_xy = np.asarray(det_xy, np.float64).reshape(-1, 2)
    det_p = np.asarray(det_p, np.float64)
    gt_xy = np.asarray(gt_xy, np.float64).reshape(-1, 2)
    gi = np.nonzero(np.asarray(gt_pos, bool) | np.asarray(gt_ign, bool))[0]
    taken = np.zeros(len(gi), bool)
    order = [i for i in np.argsort(-det_p, kind="stable") if det_p[i] >= min_conf]
    rows = []
    for i in order:
        if len(gi) == 0:
            rows.append((int(i), -1, 0))
            continue
        d = np.linalg.norm(gt_xy[gi] - det_xy[i][None, :], axis=1)
        d = np.where(taken, np.inf, d)
        j = int(np.argmin(d))
        if d[j] <= dist_m:
            taken[j] = True
            g = int(gi[j])
            rows.append((int(i), g, 1 if bool(np.asarray(gt_pos)[g]) else -1))
        else:
            rows.append((int(i), -1, 0))
    return rows


def bev_nms(det_xy, det_p, radius_m: float) -> list:
    """Class-agnostic BEV centre-distance NMS: keep the highest score, suppress every other detection whose
    centre lies within ``radius_m`` of a kept one. Returns the kept indices in score order."""
    det_xy = np.asarray(det_xy, np.float64).reshape(-1, 2)
    order = list(np.argsort(-np.asarray(det_p, np.float64), kind="stable"))
    kept = []
    for i in order:
        if all(np.hypot(*(det_xy[i] - det_xy[k])) > radius_m for k in kept):
            kept.append(int(i))
    return kept


def fan_spread_m(fan_end: np.ndarray, keep: np.ndarray | None = None) -> float:
    """A scalar for 'how messy is the fan': the mean pairwise endpoint distance of the (kept) candidates'
    6 s points. Reported, never used for selection."""
    e = np.asarray(fan_end, np.float64).reshape(-1, 2)
    if keep is not None:
        e = e[np.asarray(keep, bool)]
    if len(e) < 2:
        return 0.0
    d = np.linalg.norm(e[:, None, :] - e[None, :, :], axis=-1)
    return float(d.sum() / (len(e) * (len(e) - 1)))


def top_k_by_score(score, keep, k: int, exclude=()) -> list:
    """Indices of the ``k`` best candidates by ``score`` among those ``keep``-ed, minus ``exclude``."""
    s = np.asarray(score, np.float64).copy()
    s[~np.asarray(keep, bool)] = -np.inf
    for e in exclude:
        if e is not None:
            s[int(e)] = -np.inf
    order = [int(i) for i in np.argsort(-s, kind="stable") if np.isfinite(s[i])]
    return order[:k]


def test_vector_probe(ci: int) -> np.ndarray:
    """The 68 rig-frame 3-D probe points (x forward, y left, z up; metres) the capture projects through the model's
    own ``RigCamera`` and the tool re-projects in JavaScript. Deterministic in the clip index, so the assets stage can
    regenerate them at FULL precision (the capture stores them rounded to 1e-6 m, which alone moves a pixel by up to
    ~2e-4 at a 2 m range)."""
    rng_ = np.random.RandomState(1234 + int(ci))
    return np.concatenate([
        np.c_[rng_.uniform(2.0, 90.0, 60), rng_.uniform(-25.0, 25.0, 60), rng_.uniform(0.0, 3.0, 60)],
        np.c_[rng_.uniform(-3.0, 3.0, 8), rng_.uniform(-8.0, 8.0, 8), rng_.uniform(0.0, 2.0, 8)]])


def lerp_colour(c0, c1, t: float) -> tuple:
    t = min(max(float(t), 0.0), 1.0)
    return tuple(int(round(c0[i] + (c1[i] - c0[i]) * t)) for i in range(3))


def _p(*a):
    print(*a, flush=True)


def _r(x, n=4):
    return None if x is None else round(float(x), n)


# ============================================================================================= #
# STAGE 1 -- capture (Thor GPU)                                                                  #
# ============================================================================================= #
def stage_capture(a) -> int:  # noqa: C901 -- one linear pipeline, sectioned
    t_all = time.time()
    tree, run, work = Path(a.tree), Path(a.run), Path(a.work)
    tag = "smoke" if a.smoke else "final"
    bank = work / "bank" / tag
    if bank.exists():
        shutil.rmtree(bank)
    (bank / "win").mkdir(parents=True)
    os.environ["REFCV6_REPO"] = str(tree)
    os.environ["REFCV6_KIT"] = str(a.kit)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    rec: dict = {"tool": "taniteval/tools/render_refcv7_replay.py", "stage": "capture", "tag": tag,
                 "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "argv": sys.argv[1:], "tool_md5": rv7.md5_file(__file__),
                 "reel_tool_md5": rv7.md5_file(_HERE / "render_refcv7_video.py"), "departures": []}
    rec["palette"] = rv7.assert_palette_disjoint()
    L = rv7._by_path("refcv7_loader", tree / "stack/tanitad/eval/refcv7_loader.py")
    L.bootstrap()
    import tanitad
    if not os.path.abspath(tanitad.__file__).startswith(str(tree / "stack")):
        raise SystemExit(f"[rv7r] tanitad imported from {tanitad.__file__}, not {tree}/stack")
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "6")))
    tr = L.trainer()
    from tanitad.data import perception_targets as ptg
    from tanitad.data import semantic_map_gt_fine as smf
    from tanitad.data import v7_labels as v7l
    from tanitad.eval import detection_metrics as det
    from tanitad.models import agent_slots as _as
    from tanitad.models import map_head_hires as mhr
    from tanitad.models import slot_presence as _sp
    from tanitad.refs import refb
    from tanitad.refs import refc_tactical as rtac
    from tanitad.refs import refcv6_max_speed as v6ms
    from tanitad.refs import refcv6_selection as v6sel
    import refb_labels
    from taniteval import four_families as ff
    # ---- the same asserts as the reel's forward ----------------------------------------------- #
    if tuple(_as.AGENT_CLASSES) != rv7.AGENT_CLASSES:
        raise SystemExit("[rv7r] AGENT_CLASSES drifted")
    if (len(smf.FINE_CLASSES) != len(MAP_CLASSES) or tuple(mhr.CLASS_KEYS) != CLASS_KEYS):
        raise SystemExit("[rv7r] map classes drifted")
    ext = smf.EXTENT_REFCV7
    if (float(ext.x_max_m), float(ext.y_half_m), tuple(ext.fine_shape)) != (rv7.X_MAX_M, rv7.Y_HALF_M,
                                                                            (rv7.GRID_X, rv7.GRID_Y)):
        raise SystemExit("[rv7r] EXTENT_REFCV7 drifted")
    if int(smf.NOT_SEEN_CODE) != NOT_SEEN:
        raise SystemExit("[rv7r] NOT_SEEN_CODE drifted")
    if not os.path.abspath(ff.__file__).startswith(str(tree)):
        raise SystemExit("[rv7r] four_families not from the launch tree")
    # the literals of this module == the programme's own constants (the plan->lat3/lon3 reading)
    for nm, lit in (("CURV_TURN_MAN_PER_M", CURV_TURN_MAN_PER_M), ("DV_ACCEL_MS", DV_ACCEL_MS),
                    ("DV_BRAKE_MS", DV_BRAKE_MS), ("STOP_V_MS", STOP_V_MS), ("MOVING_V_MS", MOVING_V_MS),
                    ("MIN_ARC_M", MIN_ARC_M)):
        if abs(float(getattr(rtac, nm)) - lit) > 1e-12:
            raise SystemExit(f"[rv7r] refc_tactical.{nm} = {getattr(rtac, nm)} != the literal {lit}")
    rcv3 = rv7._by_path("render_refcv3_video_rv7", tree / "taniteval/tools/render_refcv3_video.py")
    rec["code"] = {"launch_tree": str(tree), "tanitad": tanitad.__file__,
                   "refcv7_loader_md5": rv7.md5_file(tree / "stack/tanitad/eval/refcv7_loader.py"),
                   "trainer_md5": rv7.md5_file(tree / "stack/scripts/refc_v3_train.py"),
                   "refc_v3_md5": rv7.md5_file(tree / "stack/tanitad/refs/refc_v3.py"),
                   "refc_md5": rv7.md5_file(tree / "stack/tanitad/refs/refc.py"),
                   "refcv6_tactical_md5": rv7.md5_file(tree / "stack/tanitad/refs/refcv6_tactical.py"),
                   "refcv6_selection_md5": rv7.md5_file(tree / "stack/tanitad/refs/refcv6_selection.py"),
                   "render_refcv3_video_md5": rv7.md5_file(tree / "taniteval/tools/render_refcv3_video.py")}
    # ---- checkpoint + config by CONTENT --------------------------------------------------------- #
    ckpt, cfgp = run / "ckpt.pt", run / "config.json"
    t0 = time.time()
    ck_md5 = rv7.md5_file(ckpt)
    rec["ckpt"] = {"path": str(ckpt), "md5": ck_md5, "bytes": ckpt.stat().st_size,
                   "md5_s": round(time.time() - t0, 1)}
    if ck_md5 != a.expect_ckpt_md5:
        raise SystemExit(f"[rv7r] ckpt md5 {ck_md5} != expected {a.expect_ckpt_md5}")
    rec["config"] = {"path": str(cfgp), "md5": rv7.md5_file(cfgp)}
    summ = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    if not summ.get("done") or int(summ.get("final_step", -1)) != a.expect_step:
        raise SystemExit(f"[rv7r] summary.json {summ} does not say done at {a.expect_step}")
    rec["run_summary"] = {k: summ.get(k) for k in ("done", "final_step", "written_utc", "gate_token")}
    config = L.load_config(str(cfgp))
    ev = None
    with open(run / "metrics.jsonl", encoding="utf-8") as fh:
        for ln in fh:
            if '"eval_' in ln and f'"step": {a.expect_step}' in ln:
                r_ = json.loads(ln)
                if r_.get("step") == a.expect_step and any(k.startswith("eval_") for k in r_):
                    ev = r_
    if ev is None:
        raise SystemExit(f"[rv7r] no in-run eval row at step {a.expect_step}")
    thr_box = float(ev["eval_box3d_calib_pr_gate"])
    rec["detection_display"] = {"box3d_threshold": thr_box, "declared_gate": float(_sp.DETECTION_GATE),
                                "match_dist_m": float(det.PR_DIST_M),
                                "slider_min": SAVE_SLOT_P_MIN,
                                "source": "metrics.jsonl step 50400 eval_box3d_calib_pr_gate (P=R on 256 TRAIN "
                                          "calibration windows) -- not tuned on the clips"}
    rec["inrun_eval_row"] = {"row": {k: v for k, v in ev.items() if k == "step" or (
        k.startswith("eval_") and ("calib" in k or k.endswith(("_conf_ratio", "_rec@gate", "_prec@gate"))
                                   or k.startswith("eval_map_hires_iou_")))}}
    # ---- the MODEL ------------------------------------------------------------------------------ #
    model, cfg, targs, mrec = L.build_model(config, str(ckpt), device="cuda")
    sd = mrec["state_dict"]
    if int(sd.get("step") or -1) != a.expect_step:
        raise SystemExit(f"[rv7r] STEP MISMATCH: {sd.get('step')!r}")
    if sd["missing"] or sd["unexpected"] or not mrec["param_breakdown"]["equal"]:
        raise SystemExit("[rv7r] strict load not clean / param breakdown differs")
    if [k for k, v in mrec["anchor_file_vs_ckpt_buffers"].items() if v["max_abs_diff"] != 0.0]:
        raise SystemExit("[rv7r] anchor file != checkpoint anchor buffers")
    br = getattr(model, "_map_hires", None)
    if br is None or str(br.cfg.decision_rule) != "prior_corrected":
        raise SystemExit("[rv7r] no 10 cm branch / decision rule is not prior_corrected")
    cw = model._map_hires_class_weight
    if tuple(int(h) for h in cfg.core.trajectory.horizons) != HORIZONS:
        raise SystemExit("[rv7r] horizons drifted")
    if bool(getattr(getattr(cfg.core, "agents", None), "oracle", False)):
        raise SystemExit("[rv7r] --agents oracle build: refusing")
    W = int(cfg.core.window)
    mode = getattr(targs, "mode", "diffusion")
    ablate = bool(getattr(targs, "ablate_frames", False))
    rec["model"] = {k: mrec[k] for k in ("departures", "mode", "sampler", "decoder_steps", "build_s",
                                        "stamp_checks") if k in mrec}
    rec["model"]["state_dict"] = {k: v for k, v in sd.items() if k != "ckpt_keys"}
    rec["departures"] += list(mrec.get("departures", []))
    # the selection internals this capture reads (recorded, not assumed)
    core = getattr(model, "core", None)
    dec_mod = getattr(core, "decoder", None)
    navc_tau = float(getattr(dec_mod, "navc_tau_rad", None) or cfg.core.nav_compliance_tau_rad)
    navc_gate = getattr(dec_mod, "navc_gate", None)
    rec["selection"] = {
        "speed_ceiling_filter": bool(getattr(dec_mod, "speed_ceiling_filter", False)),
        "anchor_horizons": [int(h) for h in getattr(dec_mod, "anchor_horizons", HORIZONS)],
        "anchor_dt": float(getattr(dec_mod, "anchor_dt", 0.1)),
        "navc_tau_rad": navc_tau,
        "navc_gate_value": None if navc_gate is None else float(navc_gate.detach()),
        "goal_gate_value": float(model.goal_gate.detach()),
        "refcv7_select": bool(getattr(cfg, "refcv7_select", False)),
        "note": ("E9 rank = argmax(sel_score_v3 masked by reach_keep); decoder rank = argmax(sel_score masked by "
                 "reach_keep & the speed-ceiling filter). nav compliance is a GATED SCORE TERM inside sel_score, "
                 "not a hard mask on this tree.")}
    _p(f"[rv7r] model built {mrec['build_s']} s: step {sd['step']} strict 0/0; selection {rec['selection']}")
    # ---- the held-out eval split + the FIXED clip rule (identical to the reel's) ------------------ #
    t0 = time.time()
    e_ds, e_eps, drec = L.build_eval_dataset(model, cfg, targs, config, with_perception_targets=True)
    store = e_ds.map_fine_store
    n_stack = int(e_ds.map_n_stack)
    clip_of_ep = e_ds.map_clip_of_ep
    nav_names = list(refb.NAV_COMMANDS)
    tok_of_legacy = {v: k for k, v in tr.NAV_TOKEN_TO_LEGACY.items()}
    by_ep: dict = {}
    for wi, (e_i, t) in enumerate(e_ds.index):
        by_ep.setdefault(e_i, []).append((int(t), wi))
    cands = []
    for e_i, ep in enumerate(e_eps):
        cid = clip_of_ep[int(ep.episode_id)]
        wl = sorted(by_ep.get(e_i, []))
        if not wl:
            continue
        cov = ptg.require_map_coverage([(cid, t + W - 1) for t, _ in wl], store, n_stack=n_stack,
                                       raise_on_low=False)
        nav_idx = (e_ds._nav_by_sid or {}).get(int(ep.episode_id))
        sp = ep.poses[:, 3].double().numpy()
        n_vru = n_lead = n_lab = 0
        for t, _ in wl:
            it = e_ds._agent_item(ep, t + W - 1)
            if not bool(it["agent_label"]):
                continue
            n_lab += 1
            b = it["agent_box"].double().numpy()
            v = it["agent_valid"].numpy().astype(bool)
            c = it["agent_cls"].numpy()
            rng = np.hypot(b[:, 0], b[:, 1])
            fov = (np.degrees(np.arctan2(np.abs(b[:, 1]), b[:, 0])) <= rv7.FOV_HALF_DEG) & (b[:, 0] >= 0)
            vru = v & np.isin(c, rv7.VRU_CLASSES) & (rng <= rv7.VRU_RANGE_M) & fov
            lead = (v & np.isin(c, rv7.VEHICLE_CLASSES) & (b[:, 0] > 0) & (b[:, 0] <= rv7.LEAD_MAX_X_M)
                    & (np.abs(b[:, 1]) <= rv7.LEAD_LANE_HALF_M))
            n_vru += int(vru.any())
            n_lead += int(lead.any())
        cands.append({"sha12": rv7.sha12(cid), "_cid": cid, "_e_i": e_i,
                      "nav": None if nav_idx is None else nav_names[int(nav_idx)],
                      "n_windows": len(wl), "n_map_ok": int(cov["n_ok"]),
                      "full_cov": int(cov["n_ok"]) == len(wl),
                      "v_mean": float(np.mean([sp[t + W - 1] for t, _ in wl])),
                      "v_min": float(min(sp[t + W - 1] for t, _ in wl)),
                      "v_max": float(max(sp[t + W - 1] for t, _ in wl)),
                      "n_agent_labelled_win": n_lab, "n_vru_win": n_vru, "n_lead_win": n_lead})
    chosen, sel_rep = rv7.select_clips(cands)
    # the rule must reproduce the reel's 12 clips exactly (a different set would be a different reel)
    prev_sel = json.loads((Path(OLD_BANK) / "clip_selection.json").read_text(encoding="utf-8"))
    if [c["sha12"] for c in prev_sel["chosen"]] != [c["sha12"] for c in chosen]:
        raise SystemExit(f"[rv7r] the rule chose {[c['sha12'] for c in chosen]}, the reel's bank records "
                         f"{[c['sha12'] for c in prev_sel['chosen']]}")
    rec["clip_selection"] = {"agrees_with_reel": True, "n_eval_clips": len(e_eps),
                             "chosen": [c["sha12"] for c in chosen], "selection_s": round(time.time() - t0, 1)}
    _p("[rv7r] clip rule reproduces the reel's 12 clips: " + ", ".join(c["sha12"] for c in chosen))
    run_list = chosen[:1] if a.smoke else chosen
    # ---- the forward, with the reel's hook ------------------------------------------------------ #
    cap: dict = {}

    def _hook(_m, _i, out):
        cap["out"] = out
    hook = model.register_forward_hook(_hook)

    def forward(batch, seed):
        cap.clear()
        torch.manual_seed(int(seed))
        with torch.no_grad():
            extra = tr.compute_losses_v3(model, batch, "cuda", mode=mode, ablate_frames=ablate)
        if "out" not in cap:
            raise SystemExit("[rv7r] the forward hook never fired")
        return cap["out"], extra
    torch.cuda.reset_peak_memory_stats()
    lat_names = list(v7l.HEADS["tac_lat"])
    lon_names = list(v7l.HEADS["tac_lon"])
    goal_names = list(v7l.TAC_GOAL_TOKENS)
    if len(lat_names) != 8 or len(lon_names) != 8 or len(goal_names) != 22:
        raise SystemExit(f"[rv7r] tactical vocab sizes {len(lat_names)}/{len(lon_names)}/{len(goal_names)}")
    if set(lat_names) != set(LAT_TO_LAT3) or set(lon_names) != set(LON_TO_LON3):
        raise SystemExit(f"[rv7r] the v7 action names {lat_names} / {lon_names} are not the mapping tables' keys")
    _p(f"[rv7r] vocab: lat {lat_names} | lon {lon_names} | goal {goal_names}")
    extr_tab_path = getattr(targs, "agent_rig_extrinsics")
    extr_tab = rcv3.load_extrinsics(extr_tab_path, [c["_cid"] for c in run_list])
    rec["extrinsics"] = {"path": extr_tab_path, "md5": rv7.md5_file(extr_tab_path)}
    old_rows = {}
    if not a.no_compare and (Path(OLD_BANK) / "rows.jsonl").exists():
        for ln in open(Path(OLD_BANK) / "rows.jsonl", encoding="utf-8"):
            if ln.strip():
                r_ = json.loads(ln)
                old_rows[(r_["clip_rank"], r_["win"])] = (r_["traj"], r_["sel_idx"], r_["core_sel_idx"])
    rows_f = open(bank / "rows.jsonl", "w", encoding="utf-8")
    clips_meta = []
    gpu_s = 0.0
    n_done = 0
    ident = {"map_inter_union_max_abs_diff": 0.0, "det_logit_max_abs_diff": 0.0, "det_xy_max_abs_diff": 0.0,
             "det_gt_xy_max_abs_diff": 0.0, "traj_vs_fan_max_abs_diff": 0.0, "n_det_packs": 0,
             "n_windows_no_agent_label": 0, "fan_clipped_values": 0,
             "e9_pick_reproduced": 0, "dec_pick_reproduced": 0, "ceil_mask_equals_planned_speed": 0,
             "plan_factor_equals_programme": 0, "greedy_numpy_checked": 0, "greedy_numpy_mismatch": 0,
             "n_checked_pick": 0, "n_dead_reach": 0,
             "vs_reel": {"n_compared": 0, "traj_max_abs_diff_m": 0.0, "n_traj_differs": 0,
                         "n_sel_idx_differs": 0, "n_core_idx_differs": 0, "n_pred_map_differs": 0}}
    for ci, c in enumerate(run_list):
        cid, s12, e_i = c["_cid"], c["sha12"], c["_e_i"]
        ep = e_eps[e_i]
        wl = sorted(by_ep[e_i])
        if a.smoke:
            wl = wl[:a.smoke_windows]
        elif a.max_windows:
            wl = wl[:a.max_windows]
        pay_path = Path(ep.frames._cache.cache_dir) / ep.frames._cache.files[ep.frames._clip]
        pay = torch.load(str(pay_path), map_location="cpu", weights_only=False, mmap=True)
        fr = dict(pay.get("frame") or {})
        codec = pay.get("codec")
        del pay
        fr = {"height": int(fr["height"]), "width": int(fr["width"]), "f_ref": float(fr["f_ref"]),
              "projection": str(fr["projection"])}
        if (fr["height"], fr["width"]) != (rv7.CAM_H, rv7.CAM_W):
            raise SystemExit(f"[rv7r] payload frame {fr}")
        extr = extr_tab.get(cid)
        if extr is None:
            raise SystemExit(f"[rv7r] no per-clip extrinsics for {s12}")
        extr_n = {k: float(extr[k]) for k in ("x", "y", "z", "qx", "qy", "qz", "qw")}
        proj = rcv3.CylProjector(fr, extr_n)
        rig = model._rig_camera.get(int(ep.episode_id))
        if rig is None:
            raise SystemExit(f"[rv7r] no RigCamera for {s12}")
        gp = np.array([[x, y] for x in (4.0, 8.0, 15.0, 30.0, 55.0) for y in (-6.0, 0.0, 6.0)])
        cpx = proj(gp, up=1)
        _c, _rw, _ok = rig.project(torch.as_tensor(np.c_[gp, np.zeros(len(gp))], dtype=torch.float64))
        dd = [math.hypot(q[0] - float(_c[i]), q[1] - float(_rw[i])) for i, q in enumerate(cpx)
              if q is not None and bool(_ok[i])]
        if not dd or max(dd) > 0.01:
            raise SystemExit(f"[rv7r] CylProjector vs RigCamera disagree on {s12}: {dd}")
        lr = proj(np.array([[20.0, 5.0], [20.0, -5.0]]), up=1)
        if not (lr[0] and lr[1] and lr[0][0] < (rv7.CAM_W - 1) / 2 < lr[1][0]):
            raise SystemExit(f"[rv7r] LEFT/RIGHT control failed on {s12}")
        # ⭐ the JS port's test vector: 3-D points -> (col, row, ok) by the MODEL'S OWN RigCamera. The
        # tool re-projects the same points in JavaScript and must reproduce them to <= 1e-6 px.
        probe = test_vector_probe(ci)
        pc, pr_, pok = rig.project(torch.as_tensor(probe, dtype=torch.float64))
        meta = {"clip_rank": ci + 1, "sha12": s12, "nav": c["nav"], "nav_token": tok_of_legacy.get(c["nav"]),
                "slot": c["slot"], "why": c["why"],
                "selection": {k: c[k] for k in ("n_windows", "v_mean", "v_min", "v_max",
                                                "n_agent_labelled_win", "n_vru_win", "n_lead_win")},
                "frame": fr, "codec": codec, "extrinsics": extr_n,
                "rig_camera": {"R_cam_to_rig": rig.R_cam_to_rig.double().tolist(),
                               "t_cam_in_rig": rig.t_cam_in_rig.double().tolist()},
                "js_test_vector": {"xyz": probe.tolist(),
                                   "col_row_ok": [[float(pc[i]), float(pr_[i]), bool(pok[i])]
                                                  for i in range(len(probe))]},
                "projector_cross_check_px_max": float(max(dd)), "camera_label": proj.label(),
                "n_windows_rendered": len(wl)}
        _p(f"[clip {ci + 1}/{len(run_list)}] {s12} {c['slot']} windows {len(wl)}; {proj.label()}")
        for k_i, (t, wi) in enumerate(wl):
            t_w = time.time()
            item = e_ds[wi]
            batch = torch.utils.data.default_collate([item])
            torch.cuda.synchronize()
            t_g = time.time()
            out, extra = forward(batch, a.seed)
            torch.cuda.synchronize()
            gpu_s += time.time() - t_g
            pout = out["perception"]
            lg = pout["map_hires_logits"].detach()
            lv025 = pout["map_hires_lift_valid"].detach()
            gt_codes = batch["map_fine"].to(lg.device)
            if not bool(batch["map_fine_label"][0]):
                raise SystemExit(f"[rv7r] {s12} window {k_i}: no 10 cm label on a fully covered clip")
            pred = mhr.decide(lg, str(br.cfg.decision_rule), cw)[0].to(torch.uint8)
            lvf = mhr.lift_valid_to_fine(lv025, (rv7.GRID_X, rv7.GRID_Y))[0]
            sig = mhr.per_class_signal(lg, gt_codes, class_weight=cw, lift_valid=lvf[None],
                                       decision_rule=str(br.cfg.decision_rule))
            inter = sig["inter"].cpu().numpy()
            union = sig["union"].cpu().numpy()
            bks = list(sig["band_keys"])
            dmax = 0.0
            for ci_ in range(len(CLASS_KEYS)):
                for bi, bk in enumerate(bks):
                    for st, arr in (("inter", inter), ("union", union)):
                        k_ = mhr.per_class_key(st, ci_, bk)
                        if k_ not in extra:
                            raise SystemExit(f"[rv7r] trainer extras carry no {k_}")
                        dmax = max(dmax, abs(float(extra[k_]) - float(arr[ci_, bi])))
            if dmax > 1e-6:
                raise SystemExit(f"[rv7r] map per-class counts differ from the trainer's extras by {dmax}")
            ident["map_inter_union_max_abs_diff"] = max(ident["map_inter_union_max_abs_diff"], dmax)
            n_sup = int(sig["n_supervised"])
            if abs(float(extra["n_map_hires_cells"]) - n_sup) > 0.5:
                raise SystemExit("[rv7r] supervised cell count differs from the trainer's")
            # ---- the plan, the whole fan, the scores and masks -------------------------------- #
            traj = out["traj"][0].detach().float().cpu().numpy()
            sel_idx = int(out["sel_idx"][0])
            fan_t = out["anchor_traj"].detach().float()
            fan = fan_t[0].cpu().numpy()                                     # [N, 8, 2]
            n_cand = int(fan.shape[0])
            dtf = float(np.abs(traj - fan[sel_idx]).max())
            ident["traj_vs_fan_max_abs_diff"] = max(ident["traj_vs_fan_max_abs_diff"], dtf)
            e9 = out["sel_score_v3"][0].detach().float().cpu().numpy()       # [N]  E9 blended score
            dscore = out["sel_score"][0].detach().float().cpu().numpy()      # [N]  decoder's own score
            if "reach_keep" in out:
                reach = out["reach_keep"][0].detach().cpu().numpy().astype(bool)
            else:
                reach = np.ones(n_cand, bool)
                ident["n_dead_reach"] += 1
            core_idx = int(out["sel_idx_base"][0]) if "sel_idx_base" in out else None
            v_max_raw = float(batch["v_max_ms"][0])
            v_max_ok = float(batch["v_max_valid"][0])
            oh, _over = model.max_speed_1h_v6(batch["v_max_ms"].to(lg.device).reshape(-1),
                                              batch["v_max_valid"].to(lg.device))
            if float(oh.sum()) > 0.5:
                ceil_bin = int(oh.argmax(dim=-1)[0])
                v_lim = float(v6ms.limit_ms_of_bin(oh.argmax(dim=-1))[0])
            else:
                ceil_bin, v_lim = None, float("inf")
            plan_vmax = float(v6sel.planned_max_speed(out["traj"].detach().float()[:, None],
                                                      horizons=HORIZONS, tick_s=0.1)[0, 0])
            fan_vmax = v6sel.planned_max_speed(fan_t, horizons=HORIZONS, tick_s=0.1)[0].cpu().numpy()
            ceil_keep = fan_vmax <= v_lim
            if bool(getattr(dec_mod, "speed_ceiling_filter", False)) and math.isfinite(v_lim):
                flt, _st = v6sel.SpeedCeilingFilter(horizons=dec_mod.anchor_horizons,
                                                    tick_s=dec_mod.anchor_dt)(
                    fan_t, torch.tensor([v_lim], device=fan_t.device, dtype=fan_t.dtype))
                flt = flt[0].cpu().numpy().astype(bool)
                same = bool(np.array_equal(flt, ceil_keep) or (not flt.any()))
                ident["ceil_mask_equals_planned_speed"] += int(same)
                ceil_dec = flt
            else:
                ident["ceil_mask_equals_planned_speed"] += 1
                ceil_dec = np.ones(n_cand, bool)
            # the picks, reproduced from the exported scores (a control, recorded not enforced)
            ident["n_checked_pick"] += 1
            rk_e9 = np.where(reach, e9, -np.inf)
            ident["e9_pick_reproduced"] += int(int(np.argmax(rk_e9)) == sel_idx)
            if core_idx is not None:
                keep_dec = reach & ceil_dec
                rk_dec = np.where(keep_dec if keep_dec.any() else np.ones(n_cand, bool), dscore, -np.inf)
                ident["dec_pick_reproduced"] += int(int(np.argmax(rk_dec)) == core_idx)
            # nav compliance of every candidate (the programme's predicate), the goal point and its distances
            navc_ok, navc_inf = v6sel.nav_compliance_prior(fan_t, batch["nav_cmd"].to(fan_t.device),
                                                           tau_rad=navc_tau)
            navc_ok = navc_ok[0].cpu().numpy().astype(bool)
            navc_inf = bool(navc_inf[0, 0])
            gpt = out["goal_point_tac"][0].detach().float().cpu().numpy()
            gdist = out["goal_dist"][0].detach().float().cpu().numpy()
            # ⭐ the tactical layer's SPATIAL goals: g_tac [3, 4] = (x, y, heading, speed) at 2 / 4 / 6 s
            # (refc_v3.GOAL_TAU_STEPS = 20, 40, 60 ticks), vs the label the trainer supervises it with
            # (batch["goal_tac"] / ["goal_tac_valid"], refb_labels.goal_tac_targets)
            g_tac = out["g_tac"][0].detach().float().cpu().numpy()
            goal_lab = batch["goal_tac"][0].float().numpy()
            goal_lab_ok = batch["goal_tac_valid"][0].numpy().astype(bool)
            if g_tac.shape != (3, 4) or goal_lab.shape != (3, 4):
                raise SystemExit(f"[rv7r] g_tac {g_tac.shape} / goal_tac {goal_lab.shape} are not [3, 4]")
            if "r7_candidates" in out:
                ident["r7_candidates_present"] = ident.get("r7_candidates_present", 0) + 1
            # ---- GT paths ---------------------------------------------------------------------- #
            pl = item["pose_last"].float()
            fut = item["future_poses_ext"].float()
            fv = item["future_valid_ext"].bool().numpy()
            gt_slots = refb_labels.waypoint_targets(pl[None], fut[None], HORIZONS)[0].numpy()
            slot_valid = [bool(fv[h - 1]) for h in HORIZONS]
            gt_dense = refb_labels.waypoint_targets(pl[None], fut[None], tuple(range(1, 61)))[0].numpy()
            pe = rv7.plan_errors(traj, gt_slots, slot_valid)
            gp_ = ff._seq_geometry(torch.as_tensor(traj[None, :4]), dt=0.5)
            gg_ = ff._seq_geometry(torch.as_tensor(gt_slots[None, :4]), dt=0.5)
            v4 = np.asarray(slot_valid[:4], bool)
            spd_err = (np.abs(gp_["speed"][0].numpy() - gg_["speed"][0].numpy())[v4].mean() if v4.any() else None)
            hv = gp_["valid"][0].numpy() & gg_["valid"][0].numpy() & v4
            dh = np.abs((gp_["heading"][0].numpy() - gg_["heading"][0].numpy() + math.pi) % (2 * math.pi) - math.pi)
            head_err = float(np.degrees(dh[hv]).mean()) if hv.any() else None
            pv = gp_["pair_valid"][0].numpy() & gg_["pair_valid"][0].numpy() & v4[1:] & v4[:-1]
            dk = np.abs(gp_["curvature"][0].numpy() - gg_["curvature"][0].numpy())
            curv_err = float(dk[pv].mean()) if pv.any() else None
            # ---- tactical: EVERYTHING the decoder emits ---------------------------------------- #
            plat = out["tacv6_lat_logits"][0].float().softmax(-1).cpu().numpy()
            plon = out["tacv6_lon_logits"][0].float().softmax(-1).cpu().numpy()
            pgoal = out["tacv6_goal_logits"][0].float().sigmoid().cpu().numpy()
            pconf = out["tacv6_goal_conf"][0].float().sigmoid().cpu().numpy()
            gl, gn = int(item["lat_v7"]), int(item["lon_v7"])
            gy, gw = item["tac_goal_y"].numpy(), item["tac_goal_w"].numpy()
            v0 = float(pl[3])
            # the plan / GT path read as (lat3, lon3) by the programme's factorisation
            pf = plan_factor(traj[:4], v0)
            gf = plan_factor(gt_slots[:4], v0, valid=slot_valid[:4])
            ref_l, ref_o = rtac.factor_from_kinematics(
                torch.tensor([pf["dyaw"]]), torch.tensor([pf["dv"]]), torch.tensor([v0]),
                torch.tensor([pf["v1"]]), kappa=torch.tensor([pf["kappa"]]))
            ident["plan_factor_equals_programme"] += int(int(ref_l[0]) == pf["lat3"] and int(ref_o[0]) == pf["lon3"])
            lat_pred, lon_pred = int(plat.argmax()), int(plon.argmax())
            ag = agree_tactical(pf["lat3"], pf["lon3"], lat_names[lat_pred], lon_names[lon_pred])
            ag_gt = (agree_tactical(gf["lat3"], gf["lon3"], lat_names[gl], lon_names[gn]) if
                     (gl != v7l.IGNORE_ID and gn != v7l.IGNORE_ID) else {"lat_ok": None, "lon_ok": None})
            nav_ok = agree_nav(c["nav"], pf["lat3"], bool(navc_ok[sel_idx]))
            # ---- detections: the trainer's own packs --------------------------------------------- #
            bs = pout["box_slots"]
            p_all = bs["presence_logit"][0].float().sigmoid().cpu().numpy()
            bx = bs["box"][0].float().cpu().numpy()
            byaw = bs["yaw"][0].float().cpu().numpy()
            bcz = bs["cz"][0].float().cpu().numpy()
            bh = bs["h"][0].float().cpu().numpy()
            bcls = bs["cls_logits"][0].float().argmax(-1).cpu().numpy()
            keep_s = np.nonzero(p_all >= SAVE_SLOT_P_MIN)[0]
            ag_label = bool(batch["agent_label"][0])
            gvalid = batch["agent_valid"][0].numpy().astype(bool)
            gbox = batch["agent_box"][0].numpy()[gvalid]
            gyaw = batch["agent_yaw"][0].numpy()[gvalid]
            gcls = batch["agent_cls"][0].numpy()[gvalid]
            gcz = batch["agent_cz"][0].numpy()[gvalid]
            gh = batch["agent_h"][0].numpy()[gvalid]
            gzh = batch["agent_zh_mask"][0].numpy().astype(bool)[gvalid]
            detrow = {"label": ag_label}
            if ag_label:
                if "_det_pack_box3d" not in extra:
                    raise SystemExit("[rv7r] agent-labelled window but no box3d detection pack")
                pk = extra["_det_pack_box3d"][0]
                ident["n_det_packs"] += 1
                ident["det_logit_max_abs_diff"] = max(ident["det_logit_max_abs_diff"], float(np.abs(
                    pk["logit"] - bs["presence_logit"][0].float().cpu().numpy()).max()))
                ident["det_xy_max_abs_diff"] = max(ident["det_xy_max_abs_diff"],
                                                   float(np.abs(pk["xy"] - bx[:, :2]).max()))
                if pk["gt_xy"].shape[0] != gbox.shape[0]:
                    raise SystemExit("[rv7r] pack GT rows != the batch's valid GT rows")
                if gbox.shape[0]:
                    ident["det_gt_xy_max_abs_diff"] = max(ident["det_gt_xy_max_abs_diff"], float(
                        np.abs(pk["gt_xy"] - gbox[:, :2]).max()))
                pos, ign = pk["pos"].astype(bool), pk["ign"].astype(bool)
                hid = pk["hidden"].astype(bool)
                rows_ = det.greedy_rows(pk, det.PR_DIST_M, None, min_conf=thr_box)
                detrow["thr"] = {"thr": thr_box, "n_det": sum(1 for r_ in rows_ if r_[1] >= 0),
                                 "tp": sum(1 for r_ in rows_ if r_[1] == 1),
                                 "fp": sum(1 for r_ in rows_ if r_[1] == 0),
                                 "dontcare": sum(1 for r_ in rows_ if r_[1] == -1),
                                 "pairs": [[int(r_[3]), int(r_[4]), int(r_[1])] for r_ in rows_]}
                detrow["n_pos"] = int(pos.sum())
                detrow["n_ign"] = int(ign.sum())
                # the numpy port of the matcher must reproduce the programme's rows exactly
                mine = greedy_match(bx[:, :2], p_all, gbox[:, :2] if gbox.shape[0] else np.zeros((0, 2)),
                                    pos, ign, dist_m=float(det.PR_DIST_M), min_conf=thr_box)
                ident["greedy_numpy_checked"] += 1
                if [tuple(x) for x in mine] != [(int(r_[3]), int(r_[4]), int(r_[1])) for r_ in rows_]:
                    ident["greedy_numpy_mismatch"] += 1          # float32 vs float64 at the 2 m border
            else:
                ident["n_windows_no_agent_label"] += 1
                pos = ign = hid = np.zeros(gbox.shape[0], bool)
            # ---- first window of each clip: the draw-independence control ---------------------- #
            if k_i == 0:
                out2, _ = forward(batch, a.seed + 1)
                meta["draw_independence"] = {
                    "seeds": [a.seed, a.seed + 1],
                    "traj_max_abs_diff_m": float((out2["traj"][0].float().cpu() - out["traj"][0].float().cpu())
                                                 .abs().max()),
                    "sel_idx": [sel_idx, int(out2["sel_idx"][0])],
                    "map_logits_max_abs_diff": float((out2["perception"]["map_hires_logits"].float()
                                                      - lg.float()).abs().max()),
                    "box_presence_max_abs_diff": float((out2["perception"]["box_slots"]["presence_logit"].float()
                                                        - bs["presence_logit"].float()).abs().max())}
                del out2
            # ---- bank ------------------------------------------------------------------------ #
            cam = item["frames"][-1, -3:].permute(1, 2, 0).contiguous().numpy()
            if cam.shape != (rv7.CAM_H, rv7.CAM_W, 3) or cam.dtype != np.uint8:
                raise SystemExit(f"[rv7r] camera bytes {cam.shape} {cam.dtype}")
            pred_np, gt_np, lv_np = pred.cpu().numpy(), gt_codes[0].cpu().numpy(), lvf.cpu().numpy()
            np.savez_compressed(bank / "win" / f"c{ci:02d}_w{k_i:04d}.npz", cam=cam, pred=pred_np, gt=gt_np, lv=lv_np)
            # ---- the control against the reel's own bank ---------------------------------------- #
            o = old_rows.get((ci + 1, k_i))
            if o is not None:
                vr = ident["vs_reel"]
                vr["n_compared"] += 1
                dtr = float(np.abs(np.asarray(o[0]) - np.round(traj, 3)).max())
                vr["traj_max_abs_diff_m"] = max(vr["traj_max_abs_diff_m"], dtr)
                vr["n_traj_differs"] += int(dtr > 1.5e-3)
                vr["n_sel_idx_differs"] += int(o[1] != sel_idx)
                vr["n_core_idx_differs"] += int(o[2] != core_idx)
                zf = Path(OLD_BANK) / "win" / f"c{ci:02d}_w{k_i:04d}.npz"
                if zf.exists():
                    vr["n_pred_map_differs"] += int(not np.array_equal(np.load(zf)["pred"], pred_np))
            fan_b64, n_clip = encode_fan_cm(fan)
            ident["fan_clipped_values"] += n_clip
            t_lab = float(e_ds._now_s(ep, t))
            e9c = np.where(np.isfinite(e9), e9, -99.0)
            row = {
                "clip_rank": ci + 1, "clip_sha12": s12, "win": k_i, "n_win": len(wl), "t_start_row": int(t),
                "now_row": int(t + W - 1), "t_label_s": round(t_lab, 4), "v0_ms": round(v0, 4), "nav": c["nav"],
                "nav_cmd": nav_names[int(item["nav_cmd"])], "nav_valid": bool(item["nav_valid"]),
                "v_max_raw_ms": round(v_max_raw, 4), "v_max_valid": v_max_ok, "ceil_bin": ceil_bin,
                "ceil_kmh": None if ceil_bin is None else int(v6ms.SPEED_MAX_STEPS_KMH_V6[ceil_bin]),
                "v_lim_ms": None if not math.isfinite(v_lim) else round(v_lim, 4),
                "plan_vmax_ms": round(plan_vmax, 4), "plan_exceeds_ceiling": bool(plan_vmax > v_lim),
                "n_fan_over_ceiling": int((~ceil_keep).sum()),
                "sel_idx": sel_idx, "core_sel_idx": core_idx,
                "core_sel_in_ceil_keep": None if core_idx is None else bool(ceil_keep[core_idx]),
                "sel_in_ceil_keep": bool(ceil_keep[sel_idx]),
                "traj": np.round(traj, 3).tolist(), "n_cand": n_cand, "fan_b64": fan_b64,
                "e9": np.round(e9c, 4).tolist(), "dec": np.round(dscore, 4).tolist(),
                "reach": bits(reach), "ceilk": bits(ceil_keep), "ceilk_dec": bits(ceil_dec),
                "vmx": np.round(fan_vmax, 2).tolist(), "navc": bits(navc_ok), "navc_inf": navc_inf,
                "gdist": np.round(gdist, 2).tolist(), "gpt": np.round(gpt, 3).tolist(),
                "g_tac": np.round(g_tac, 3).tolist(), "g_tac_gt": np.round(goal_lab, 3).tolist(),
                "g_tac_gt_valid": [bool(x) for x in goal_lab_ok],
                "n_reach_keep": int(reach.sum()),
                "gt_slots": np.round(gt_slots, 3).tolist(), "slot_valid": slot_valid,
                "gt_dense": np.round(gt_dense, 3).tolist(), "gt_dense_valid": [bool(x) for x in fv],
                **{k: _r(v) for k, v in pe.items() if k != "slot_err_m"},
                "slot_err_m": [round(x, 3) for x in pe["slot_err_m"]],
                "speed_mae_0_2s": _r(spd_err), "heading_mae_0_2s_deg": _r(head_err), "curv_mae_0_2s": _r(curv_err, 5),
                "tac_lat_p": np.round(plat, 4).tolist(), "tac_lon_p": np.round(plon, 4).tolist(),
                "tac_lat_pred": lat_pred, "tac_lon_pred": lon_pred,
                "tac_lat_gt": None if gl == v7l.IGNORE_ID else gl, "tac_lon_gt": None if gn == v7l.IGNORE_ID else gn,
                "goal_p": np.round(pgoal, 4).tolist(), "goal_conf": np.round(pconf, 4).tolist(),
                "goal_gt_pos": [int(i) for i in np.nonzero((gy > 0.5) & (gw > 0))[0]],
                "goal_scored": bits(gw > 0),
                "plan3": {"lat3": pf["lat3"], "lon3": pf["lon3"], "dyaw": _r(pf["dyaw"], 4), "kappa": _r(pf["kappa"], 5),
                          "dv": _r(pf["dv"], 3), "v1": _r(pf["v1"], 3), **ag, "nav_ok": nav_ok},
                "gt3": {"lat3": gf["lat3"], "lon3": gf["lon3"], **ag_gt},
                "map_inter": inter.tolist(), "map_union": union.tolist(), "n_map_scored": n_sup,
                "n_map_seen": int((gt_codes[0] != NOT_SEEN).sum()), "n_lift_valid": int(lvf.sum()),
                "pred": {"idx": keep_s.tolist(), "p": np.round(p_all[keep_s].astype(np.float64), 6).tolist(),
                         "box": np.round(bx[keep_s].astype(np.float64), 4).tolist(), "yaw": np.round(byaw[keep_s], 4).tolist(),
                         "cz": np.round(bcz[keep_s], 3).tolist(), "h": np.round(bh[keep_s], 3).tolist(),
                         "cls": bcls[keep_s].tolist()},
                "p_max": round(float(p_all.max()), 4), "n_slots": int(p_all.shape[0]),
                "gt": {"box": np.round(gbox.astype(np.float64), 4).tolist(), "yaw": np.round(gyaw, 4).tolist(), "cls": gcls.tolist(),
                       "cz": np.round(gcz, 3).tolist(), "h": np.round(gh, 3).tolist(), "zh": gzh.tolist(),
                       "pos": pos.tolist(), "ign": ign.tolist(), "hidden": hid.tolist()},
                "det": detrow, "seed": a.seed, "wall_s": None}
            row["wall_s"] = round(time.time() - t_w, 3)
            rows_f.write(rv7.dumps(row) + "\n")
            n_done += 1
            if k_i % 25 == 0 or k_i == len(wl) - 1:
                _p(f"  w{k_i + 1:03d}/{len(wl)} t={t_lab:.1f}s ADE {row['ade_m']} sel {sel_idx} core {core_idx} "
                   f"navc_inf {navc_inf} plan3 {pf['lat3']}/{pf['lon3']} | {row['wall_s']} s")
            del out, extra, lg
        clips_meta.append(meta)
        rows_f.flush()
    rows_f.close()
    hook.remove()
    for k_, v_ in (("map_inter_union_max_abs_diff", 1e-6), ("det_logit_max_abs_diff", 0.0),
                   ("det_xy_max_abs_diff", 0.0), ("traj_vs_fan_max_abs_diff", 0.0)):
        if ident[k_] > v_:
            raise SystemExit(f"[rv7r] identity {k_} = {ident[k_]} > {v_}")
    n_pf_bad = n_done - ident["plan_factor_equals_programme"]
    if n_pf_bad > max(1, int(0.005 * n_done)) or ident["greedy_numpy_mismatch"] > max(1, int(0.005 * n_done)):
        raise SystemExit(f"[rv7r] numpy ports differ from the programme: plan_factor on {n_pf_bad} windows, "
                         f"greedy_match on {ident['greedy_numpy_mismatch']} windows")
    rec["identity_checks"] = ident
    rec["dataset"] = {"label_names": {"tac_lat": lat_names, "tac_lon": lon_names, "goal": goal_names},
                      "agent_classes": list(rv7.AGENT_CLASSES), "n_cand": n_cand}
    rec["clips"] = clips_meta
    rec["n_windows"] = n_done
    rec["compute"] = {"gpu_forward_s": round(gpu_s, 1), "wall_s": round(time.time() - t_all, 1),
                      "cuda_max_memory_allocated_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 3),
                      "device": torch.cuda.get_device_name(0)}
    rec["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rv7.write_json(bank / "capture_record.json", rec)
    (bank / "CAPTURE_DONE").write_text(rv7.dumps({"n_windows": n_done, "wall_s": rec["compute"]["wall_s"]}),
                                       encoding="utf-8")
    _p(f"[rv7r] CAPTURE DONE: {n_done} windows, GPU {gpu_s:.0f} s, wall {time.time() - t_all:.0f} s; "
       f"identity {json.dumps(ident)}")
    return 0


# ============================================================================================= #
# STAGE 2 -- assets (Thor CPU): the static tool's data                                           #
# ============================================================================================= #
def compact_window(r: dict) -> dict:
    """One captured row -> the compact record ``replay.js`` reads (keys documented in RENDER_REFCV7_REPLAY.md)."""
    dv = list(r["gt_dense_valid"])
    n_ok = 0
    while n_ok < len(dv) and dv[n_ok]:                       # the valid part of the 6 s GT path is a PREFIX
        n_ok += 1
    if any(dv[n_ok:]):
        raise SystemExit(f"[rv7r] clip {r['clip_rank']} w{r['win']}: GT dense validity is not a prefix")
    det = r["det"]
    thr = det.get("thr") or {}
    pr, gt, p3 = r["pred"], r["gt"], r["plan3"]
    it = np.sum(np.asarray(r["map_inter"], np.float64), axis=1)
    un = np.sum(np.asarray(r["map_union"], np.float64), axis=1)
    return {
        "i": r["win"], "t": r["t_label_s"], "v0": r["v0_ms"], "nav": r["nav_cmd"],
        "vmr": r["v_max_raw_ms"], "vmv": r["v_max_valid"], "ck": r["ceil_kmh"], "vl": r["v_lim_ms"],
        "tr": r["traj"], "sel": r["sel_idx"], "core": r["core_sel_idx"],
        "fan": r["fan_b64"], "e9": r["e9"], "dc": r["dec"], "rch": r["reach"], "ceil": r["ceilk"],
        "navc": r["navc"], "nci": r["navc_inf"], "vmx": r["vmx"], "gd": r["gdist"],
        "pvm": r["plan_vmax_ms"], "pex": r["plan_exceeds_ceiling"],
        "gtd": r["gt_dense"][:n_ok],
        "latp": r["tac_lat_p"], "lonp": r["tac_lon_p"], "latg": r["tac_lat_gt"], "long_": r["tac_lon_gt"],
        "gp": r["goal_p"], "gcf": r["goal_conf"], "ggt": r["goal_gt_pos"], "gsc": r["goal_scored"],
        "gtac": r["g_tac"], "gtacg": r["g_tac_gt"], "gtacv": r["g_tac_gt_valid"],
        "p3": {"lat3": p3["lat3"], "lon3": p3["lon3"], "latok": p3["lat_ok"], "lonok": p3["lon_ok"],
               "navok": p3["nav_ok"]},
        "m": {"ade": r["ade_m"], "fde": r["fde_m"], "spd": r["speed_mae_0_2s"], "hd": r["heading_mae_0_2s_deg"],
              "cv": r["curv_mae_0_2s"], "it": [int(x) for x in it], "un": [int(x) for x in un],
              "nsc": r["n_map_scored"]},
        "bx": {"i": pr["idx"], "p": pr["p"], "b": pr["box"], "y": pr["yaw"], "z": pr["cz"], "h": pr["h"],
               "c": pr["cls"]},
        "gb": {"b": gt["box"], "y": gt["yaw"], "c": gt["cls"], "z": gt["cz"], "h": gt["h"], "zh": gt["zh"],
               "pos": gt["pos"], "ign": gt["ign"], "hid": gt["hidden"]},
        "dt": {"lab": bool(det["label"]), "np": int(det.get("n_pos", 0)), "pairs": thr.get("pairs", []),
               "tp0": int(thr.get("tp", 0)), "n0": int(thr.get("n_det", 0))},
    }


def _asset_job(job):
    """One window: camera JPEG file + the two map PNGs (base64). Runs in a pool worker."""
    from PIL import Image
    ci, k, bank, out, quality = job
    z = np.load(Path(bank) / "win" / f"c{ci:02d}_w{k:04d}.npz")
    d = Path(out) / "data" / "cam" / f"c{ci + 1:02d}"
    path = d / f"w{k:04d}.jpg"
    Image.fromarray(z["cam"]).save(path, "JPEG", quality=int(quality), optimize=True)
    gt, pred, lv = z["gt"], z["pred"], z["lv"]
    # the prediction PNG carries the GT's "never seen" flag too, so BOTH files are self-contained
    p_png = encode_map(pred, lv, notseen=(gt == NOT_SEEN))
    g_png = encode_map(gt, lv)
    n_scored = int(((gt != NOT_SEEN) & np.asarray(lv, bool)).sum())
    return (ci, k, base64.b64encode(p_png).decode("ascii"), base64.b64encode(g_png).decode("ascii"),
            path.stat().st_size, len(p_png) + len(g_png), n_scored)


def stage_assets(a) -> int:  # noqa: C901
    t_all = time.time()
    work = Path(a.work)
    tag = "smoke" if a.smoke else "final"
    bank = work / "bank" / tag
    if not (bank / "CAPTURE_DONE").exists():
        raise SystemExit(f"[rv7r] {bank}/CAPTURE_DONE is missing -- run --stage capture first")
    rec = json.loads((bank / "capture_record.json").read_text(encoding="utf-8"))
    rows_all = [json.loads(ln) for ln in open(bank / "rows.jsonl", encoding="utf-8") if ln.strip()]
    clips = rec["clips"]
    out = work / "out" / tag / "tool"
    if out.exists():
        shutil.rmtree(out)
    (out / "data").mkdir(parents=True)
    for ci in range(len(clips)):
        (out / "data" / "cam" / f"c{ci + 1:02d}").mkdir(parents=True)
    for f in ("index.html", "replay.css", "replay.js"):
        shutil.copyfile(_HERE / "refcv7_replay" / f, out / f)
    by_clip = [[r for r in rows_all if r["clip_rank"] == ci + 1] for ci in range(len(clips))]
    for ci, rr in enumerate(by_clip):
        if [r["win"] for r in rr] != list(range(len(rr))) or len(rr) != clips[ci]["n_windows_rendered"]:
            raise SystemExit(f"[rv7r] clip {ci + 1} rows are incomplete / out of order")
    jobs = [(ci, k, str(bank), str(out), a.jpeg_quality) for ci in range(len(clips)) for k in range(len(by_clip[ci]))]
    import multiprocessing as mp
    t0 = time.time()
    if a.workers <= 1 or os.name == "nt":                       # Windows has no fork: run in-process
        res = [_asset_job(j) for j in jobs]
    else:
        with mp.get_context("fork").Pool(processes=a.workers) as pool:
            res = pool.map(_asset_job, jobs, chunksize=8)
    _p(f"[rv7r] {len(res)} windows encoded in {time.time() - t0:.0f} s")
    maps = [[None] * len(by_clip[ci]) for ci in range(len(clips))]
    jpeg_bytes = png_bytes = 0
    for ci, k, pb, gb_, jb, nb, n_scored in res:
        if n_scored != by_clip[ci][k]["n_map_scored"]:
            raise SystemExit(f"[rv7r] clip {ci + 1} w{k}: scored cells {n_scored} != the trainer's "
                             f"{by_clip[ci][k]['n_map_scored']}")
        maps[ci][k] = [pb, gb_]
        jpeg_bytes += jb
        png_bytes += nb
    thr_box = float(rec["detection_display"]["box3d_threshold"])
    names = rec["dataset"]["label_names"]
    man_clips = []
    js_bytes = 0
    for ci, m in enumerate(clips):
        ws = [compact_window(r) for r in by_clip[ci]]
        fr, rc = m["frame"], m["rig_camera"]
        xyz_full = test_vector_probe(ci)
        if np.abs(np.round(xyz_full, 6) - np.asarray(m["js_test_vector"]["xyz"])).max() > 1.01e-6:
            raise SystemExit(f"[rv7r] clip {ci + 1}: the regenerated test-vector probe differs from the captured one")
        cdat = {"rank": ci + 1, "sha12": m["sha12"], "nav": m["nav"], "slot": m["slot"], "why": m["why"],
                "n": len(ws),
                "cam": {"W": fr["width"], "H": fr["height"], "f": fr["f_ref"], "R": rc["R_cam_to_rig"],
                        "t": rc["t_cam_in_rig"]},
                "tv": {"xyz": xyz_full.tolist(), "cro": m["js_test_vector"]["col_row_ok"]},
                "w": ws, "mp": maps[ci]}
        txt = ("window.RV7=window.RV7||{};window.RV7.clips=window.RV7.clips||{};"
               f"window.RV7.clips[{ci + 1}]=" + rv7.dumps(cdat, separators=(",", ":")) + ";\n")
        (out / "data" / f"clip_{ci + 1:02d}.js").write_text(txt, encoding="utf-8")
        js_bytes += len(txt.encode("utf-8"))
        man_clips.append({"rank": ci + 1, "sha12": m["sha12"], "nav": m["nav"], "slot": m["slot"], "why": m["why"],
                          "n": len(ws), "file": f"data/clip_{ci + 1:02d}.js", "cam_dir": f"data/cam/c{ci + 1:02d}/"})
    manifest = {"run": RUN_LABEL, "step": int(a.expect_step), "ckpt_md5": rec["ckpt"]["md5"],
                "tier_note": TIER_NOTE, "ceil_stamp": CEIL_STAMP, "thr_box": thr_box, "thr_min": SAVE_SLOT_P_MIN,
                "match_dist": float(rec["detection_display"]["match_dist_m"]),
                "lat": names["tac_lat"], "lon": names["tac_lon"], "goal": names["goal"],
                "mapcls": list(MAP_CLASSES), "mapcol": [list(c) for c in rv7.PALETTE],
                "agent": list(rv7.AGENT_CLASSES), "short_cls": list(rv7.SHORT_CLS), "clips": man_clips,
                "control": gt_control(rows_all)}
    (out / "data" / "manifest.js").write_text(
        "window.RV7=window.RV7||{};window.RV7.manifest=" + rv7.dumps(manifest, separators=(",", ":")) + ";\n",
        encoding="utf-8")
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    arec = {"tool": "taniteval/tools/render_refcv7_replay.py", "stage": "assets", "tag": tag,
            "tool_md5": rv7.md5_file(__file__), "n_windows": len(res), "jpeg_quality": a.jpeg_quality,
            "bytes": {"camera_jpeg": jpeg_bytes, "map_png_raw": png_bytes, "clip_js": js_bytes, "total_tool_dir": total},
            "files": {f: rv7.md5_file(out / f) for f in ("index.html", "replay.css", "replay.js", "data/manifest.js")},
            "wall_s": round(time.time() - t_all, 1), "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    rv7.write_json(out / "REPLAY_ASSETS_RECORD.json", arec)
    _p(f"[rv7r] ASSETS DONE: {total / 1e6:.1f} MB ({jpeg_bytes / 1e6:.1f} MB JPEG, {js_bytes / 1e6:.1f} MB clip js) "
       f"in {time.time() - t_all:.0f} s -> {out}")
    return 0


# ============================================================================================= #
# STAGE 3 -- the TACTICAL video (Thor CPU)                                                       #
# ============================================================================================= #
W_TOT, H_TOT = 1920, 1080
CAM_X, CAM_Y, CAM_W, CAM_H = 16, 46, 1024, 416
BEV_X, BEV_Y, BEV_W, BEV_H = 1056, 74, 480, 800       # 8 px / m: 100 m ahead x +-30 m, forward is up, LEFT is left
BSX = BSY = 8.0
TOPK_VIDEO = 3                                         # the fan is shown as its top-3 by the E9 score
NMS_RADIUS_M = 2.0
BG, PANEL, BAN = (9, 12, 17), (16, 21, 29), (15, 20, 28)
FG, DIM, DIM2, WARN = (233, 238, 245), (140, 152, 168), (104, 116, 132), (245, 180, 90)
LAT_RGB = [(100, 116, 139), (56, 189, 248), (129, 140, 248), (167, 139, 250), (34, 211, 238), (192, 132, 252),
           (244, 114, 182), (251, 113, 133)]
LON_RGB = [(148, 163, 184), (100, 116, 139), (167, 139, 250), (244, 63, 94), (56, 189, 248), (71, 85, 105),
           (232, 121, 249), (250, 204, 21)]
_V: dict = {}


def m2v(x, y):
    """Ego-frame metres (x forward, y LEFT) -> BEV-panel pixels (8 px/m); the ego is at the bottom centre."""
    return ((30.0 - float(y)) * BSX, (100.0 - float(x)) * BSY)


def _vfont(size: int, bold: bool = False):
    key = (size, bold)
    cache = _V.setdefault("fonts", {})
    if key not in cache:
        cache[key] = _V["rcv3"].font(size, bold)
    return cache[key]


def _fitv(d, text, f, w):
    return _V["rcv3"].fit(d, text, f, w)


def _dash_pts(d, pts, fill, width, dash=14.0, gap=9.0):
    """Dashed polyline through ``pts`` (a ``None`` breaks the line)."""
    run = []
    runs = []
    for p in pts:
        if p is None:
            if len(run) >= 2:
                runs.append(run)
            run = []
        else:
            run.append(p)
    if len(run) >= 2:
        runs.append(run)
    for r in runs:
        rv7._dashed(d, r, fill, width=width, dash=dash, gap=gap, closed=False)


def _chip_v(d, text, xy, fg, bg, font, pad=3):
    tw = d.textlength(text, font=font)
    x, y = xy
    d.rectangle([x - pad, y - 1, x + tw + pad, y + font.size + 1], fill=bg)
    d.text((x, y), text, fill=fg, font=font)
    return tw + 2 * pad


def fan_top(row: dict, k: int = TOPK_VIDEO) -> list:
    """The top-``k`` candidates by the E9 score among those the reachability mask keeps (the E9 pick is #1)."""
    return top_k_by_score(row["e9"], [c == "1" for c in row["reach"]], k)


def shown_boxes(row: dict, thr: float, nms_r: float = NMS_RADIUS_M) -> dict:
    """The model boxes the video draws: score >= ``thr``, BEV centre-distance NMS, then the programme's greedy
    2 m match against GT (positives U IGNORE). ``kind``: 1 TP / 0 FP / -1 DontCare, keyed by saved-slot position."""
    pr, gt = row["pred"], row["gt"]
    idx = [j for j, p in enumerate(pr["p"]) if p >= thr - 1e-12]
    if idx:
        kept = bev_nms([pr["box"][j][:2] for j in idx], [pr["p"][j] for j in idx], nms_r)
        idx = [idx[q] for q in kept]
    res = {"idx": idx, "tp": 0, "fp": 0, "dc": 0, "kind": {}, "link": {}, "label": bool(row["det"]["label"]),
           "npos": int(row["det"].get("n_pos", 0))}
    if res["label"] and idx:
        rows = greedy_match([pr["box"][j][:2] for j in idx], [pr["p"][j] for j in idx],
                            [b[:2] for b in gt["box"]], gt["pos"], gt["ign"], dist_m=2.0)
        for i, g, kd in rows:
            j = idx[i]
            res["kind"][j] = kd
            if kd == 1:
                res["tp"] += 1
                res["link"][j] = g
            elif kd == 0:
                res["fp"] += 1
            else:
                res["dc"] += 1
    return res


def _cam_overlay_v(row, cam, meta, boxes):
    import torch
    from PIL import Image, ImageDraw
    rcv3, rv6 = _V["rcv3"], _V["rv6"]
    proj, rig = _V["proj"][meta["sha12"]], _V["rig"][meta["sha12"]]
    F = _vfont
    im = Image.fromarray(cam)
    d = ImageDraw.Draw(im, "RGBA")
    fan = decode_fan_cm(row["fan_b64"], row["n_cand"])
    top = fan_top(row)
    for rank in range(len(top) - 1, -1, -1):
        col = lerp_colour(FAN_HI, FAN_LO, rank / max(TOPK_VIDEO - 1, 1))
        pp = proj(rcv3.densify(fan[top[rank]]), up=1)
        rcv3.polyline(d, pp, col + (235,), 4)
        last = [p for p in pp if p is not None]
        if last and rank > 0:
            _chip_v(d, f"MODEL #{rank + 1}", (min(last[-1][0] + 6, CAM_W - 80), min(max(last[-1][1] - 6, 16), CAM_H - 20)),
                    (255, 255, 255), col + (230,), F(12, True))
    placed = []
    labels = []
    g = row["gt"]
    for i, b in enumerate(g["box"]):
        main, ign = g["pos"][i], g["ign"][i]
        if not (main or ign):
            continue
        hz = bool(g["zh"][i])
        cor = rv6.cuboid_corners(b[0], b[1], g["cz"][i] if hz else 0.0, b[2], b[3], g["h"][i] if hz else 0.0,
                                 g["yaw"][i])
        for _e, pts in rv6.cuboid_edge_runs(rig, cor):
            if main:
                d.line(pts, fill=C_GT + (255,), width=3, joint="curve")
            else:
                rv7._dashed(d, pts, C_GT + (230,), width=2, dash=6, gap=4, closed=False)
        tp = torch.as_tensor([[b[0], b[1], (g["cz"][i] + 0.5 * g["h"][i]) if hz else 0.0]], dtype=torch.float64)
        cc, rr, ok = rig.project(tp)
        if bool(ok[0]):
            labels.append((float(cc[0]), float(rr[0]), f"GT {rv7.SHORT_CLS[int(g['cls'][i])]}", C_GT, (6, 22, 12, 150), 1.0))
    pr = row["pred"]
    for j in boxes["idx"]:
        b = pr["box"][j]
        cor = rv6.cuboid_corners(b[0], b[1], pr["cz"][j], b[2], b[3], pr["h"][j], pr["yaw"][j])
        pts8 = [rig.project(torch.as_tensor([[c[0], c[1], c[2]]], dtype=torch.float64)) for c in cor]
        for face in ((0, 1, 2, 3), (4, 5, 6, 7)):                # filled footprint + top face when fully in frame
            if all(bool(pts8[q][2][0]) for q in face):
                d.polygon([(float(pts8[q][0][0]), float(pts8[q][1][0])) for q in face], fill=C_BOX + (70,))
        for _e, pts in rv6.cuboid_edge_runs(rig, cor):
            d.line(pts, fill=C_BOX + (255,), width=3, joint="curve")
        tp = torch.as_tensor([[b[0], b[1], pr["cz"][j] + 0.5 * pr["h"][j]]], dtype=torch.float64)
        cc, rr, ok = rig.project(tp)
        if bool(ok[0]):
            kd = boxes["kind"].get(j)
            labels.append((float(cc[0]), float(rr[0]),
                           f"{rv7.SHORT_CLS[int(pr['cls'][j])]} {pr['p'][j]:.2f}" + (" TP" if kd == 1 else ""),
                           (255, 255, 255), C_BOX + (235,), 2.0 + pr["p"][j]))
    labels.sort(key=lambda t: -t[5])
    for x, y, text, fg, bg, _pri in labels:
        tw = d.textlength(text, font=F(12, True))
        tx = min(max(x - tw / 2, 2.0), CAM_W - tw - 4)
        for dy in (-16, -30, 2, -44, 16):
            ty = y + dy
            rc = (tx - 2, ty, tx + tw + 2, ty + 15)
            if ty < 20 or ty > CAM_H - 24:
                continue
            if all(rc[2] < q[0] or rc[0] > q[2] or rc[3] < q[1] or rc[1] > q[3] for q in placed):
                placed.append(rc)
                d.rectangle(rc, fill=bg)
                d.text((tx, ty), text, fill=fg, font=F(12, True))
                break
    # GT path (dashed, wide) under the decoder pick and the plan
    gv = np.asarray(row["gt_dense_valid"], bool)
    gd = np.asarray(row["gt_dense"])[gv]
    if len(gd):
        pp = proj(np.concatenate([np.zeros((1, 2)), gd]), up=1)
        _dash_pts(d, pp, C_GT + (255,), 9, dash=16.0, gap=9.0)
        for k in range(9, len(gv), 10):
            if gv[k]:
                q = proj([row["gt_dense"][k]], up=1)[0]
                if q is not None:
                    d.ellipse([q[0] - 5, q[1] - 5, q[0] + 5, q[1] + 5], fill=C_GT + (255,))
        last = [p for p in pp if p is not None]
        if last:
            _chip_v(d, "GT path", (min(last[-1][0] + 8, CAM_W - 70), min(last[-1][1] + 6, CAM_H - 40)), (6, 22, 12), C_GT + (240,), F(13, True))
    if row["core_sel_idx"] is not None and row["core_sel_idx"] != row["sel_idx"]:
        pp = proj(rcv3.densify(fan[row["core_sel_idx"]]), up=1)
        rcv3.polyline(d, pp, C_DEC + (255,), 5)
        last = [p for p in pp if p is not None]
        if last:
            _chip_v(d, f"MODEL decoder pick #{row['core_sel_idx']}", (min(last[-1][0] + 40, CAM_W - 200), min(last[-1][1] + 34, CAM_H - 40)),
                    (255, 255, 255), C_DEC + (240,), F(13, True))
    pp = proj(rcv3.densify(np.asarray(row["traj"])), up=1)
    rcv3.polyline(d, pp, C_PLAN + (255,), 4)
    for h, q in zip(HORIZONS, proj(row["traj"], up=1)):
        if q is None:
            continue
        if h % 10 == 0:
            d.ellipse([q[0] - 5, q[1] - 5, q[0] + 5, q[1] + 5], fill=C_PLAN + (255,), outline=(15, 15, 15, 255))
        else:
            d.ellipse([q[0] - 4, q[1] - 4, q[0] + 4, q[1] + 4], outline=C_PLAN + (255,), width=2)
    last = [p for p in pp if p is not None]
    if last:
        _chip_v(d, f"MODEL plan (E9 #{row['sel_idx']})", (min(last[-1][0] + 8, CAM_W - 190), max(last[-1][1] - 22, 16)),
                (255, 255, 255), C_PLAN + (245,), F(13, True))
    # the tactical geometric goals on the ground
    for q, g4 in enumerate(row["g_tac"]):
        p = proj([[g4[0], g4[1]]], up=1)[0]
        if p is not None:
            d.polygon([(p[0], p[1] - 10), (p[0] + 10, p[1]), (p[0], p[1] + 10), (p[0] - 10, p[1])], fill=C_GOAL + (240,),
                      outline=(40, 10, 45, 255))
            _chip_v(d, f"g{2 * (q + 1)}", (p[0] + 13, p[1] - 8), (40, 10, 45), C_GOAL + (235,), F(12, True))
        if row["g_tac_gt_valid"][q]:
            gg = row["g_tac_gt"][q]
            p = proj([[gg[0], gg[1]]], up=1)[0]
            if p is not None:
                d.ellipse([p[0] - 11, p[1] - 11, p[0] + 11, p[1] + 11], outline=C_GT + (255,), width=3)
                _chip_v(d, f"GT{2 * (q + 1)}", (p[0] + 14, p[1] + 6), (6, 22, 12), C_GT + (235,), F(12, True))
    # the stamp every planner overlay carries
    txt = "MODEL plan: " + CEIL_STAMP
    tw = d.textlength(txt, font=F(13, True))
    d.rectangle([0, CAM_H - 22, tw + 14, CAM_H], fill=(9, 12, 17, 215))
    d.text((7, CAM_H - 19), txt, fill=(252, 165, 165) if row["plan_exceeds_ceiling"] else WARN, font=F(13, True))
    # the legend tags
    x0 = 6
    x0 += _chip_v(d, "MODEL", (x0, 4), (255, 255, 255), C_PLAN + (245,), F(13, True)) + 10
    _chip_v(d, "GT", (x0, 4), (6, 22, 12), C_GT + (245,), F(13, True))
    return im


def _bev_v(row, z, boxes):
    from PIL import Image, ImageDraw
    F = _vfont
    pred, gt, lv = z["pred"], z["gt"], z["lv"]
    seen = gt != NOT_SEEN
    rgbm = rv7.class_rgb(pred).astype(np.float32)
    rgbm[lv & ~seen] *= 0.42
    rgbm *= 0.62
    img = rv7.downsample_rgb(rv7.grid_to_image(rgbm.astype(np.uint8)), (BEV_W, BEV_H)).copy()
    hole = rv7.downsample_mask(rv7.grid_to_image(~lv), (BEV_W, BEV_H))
    img[hole] = (14, 16, 21)
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im, "RGBA")
    ex, ey = m2v(0, 0)
    for r in range(10, 101, 10):
        d.ellipse([ex - r * BSX, ey - r * BSY, ex + r * BSX, ey + r * BSY], outline=(255, 255, 255, 60 if r % 50 else 110), width=1)
    for yl in (-20.0, -10.0, 0.0, 10.0, 20.0):
        px, _ = m2v(0, yl)
        d.line([(px, 0), (px, BEV_H)], fill=(255, 255, 255, 22 if yl else 40), width=1)
    # the top-3 fan, then GT boxes (outline), model boxes (filled)
    fan = decode_fan_cm(row["fan_b64"], row["n_cand"])
    top = fan_top(row)
    for rank in range(len(top) - 1, -1, -1):
        col = lerp_colour(FAN_HI, FAN_LO, rank / max(TOPK_VIDEO - 1, 1))
        d.line([m2v(0, 0)] + [m2v(x, y) for x, y in fan[top[rank]]], fill=col + (235,), width=3, joint="curve")
    g = row["gt"]
    for i, b in enumerate(g["box"]):
        if not (0.0 <= b[0] <= 100.0 and abs(b[1]) <= 30.0):
            continue
        pts = [m2v(x, y) for x, y in rv7.box_corners_xy(b[0], b[1], b[2], b[3], g["yaw"][i])]
        if g["pos"][i]:
            d.polygon(pts, outline=C_GT + (255,), width=2)
        elif g["ign"][i]:
            rv7._dashed(d, pts, C_GT + (225,), width=2, dash=4, gap=3)
    pr = row["pred"]
    for j in boxes["idx"]:
        b = pr["box"][j]
        pts = [m2v(x, y) for x, y in rv7.box_corners_xy(b[0], b[1], b[2], b[3], pr["yaw"][j])]
        d.polygon(pts, fill=C_BOX + (120,), outline=(255, 255, 255, 255) if boxes["kind"].get(j) == 1 else C_BOX + (255,), width=2)
        tx, ty = m2v(b[0], b[1])
        d.text((tx + 7, ty - 8), f"{pr['p'][j]:.2f}", fill=(219, 234, 254, 255), font=F(12, True))
        if boxes["kind"].get(j) == 1:
            gb = g["box"][boxes["link"][j]]
            d.line([m2v(b[0], b[1]), m2v(gb[0], gb[1])], fill=(255, 255, 255, 220), width=1)
    gv = np.asarray(row["gt_dense_valid"], bool)
    gd = np.asarray(row["gt_dense"])[gv]
    if len(gd):
        rv7._dashed(d, [m2v(0, 0)] + [m2v(x, y) for x, y in gd], C_GT + (255,), width=6, dash=11, gap=7, closed=False)
        for k in range(9, len(gv), 10):
            if gv[k]:
                px, py = m2v(*row["gt_dense"][k])
                d.ellipse([px - 4, py - 4, px + 4, py + 4], fill=C_GT + (255,))
        e = m2v(*gd[-1])
        _chip_v(d, "GT path", (min(max(e[0] + 6, 2), BEV_W - 60), min(max(e[1] + 4, 14), BEV_H - 22)), (6, 22, 12), C_GT + (240,), F(12, True))
    if row["core_sel_idx"] is not None and row["core_sel_idx"] != row["sel_idx"]:
        pts = [m2v(0, 0)] + [m2v(x, y) for x, y in fan[row["core_sel_idx"]]]
        d.line(pts, fill=C_DEC + (255,), width=4, joint="curve")
        e = pts[-1]
        _chip_v(d, f"MODEL decoder pick #{row['core_sel_idx']}", (min(max(e[0] + 6, 2), BEV_W - 170), min(max(e[1] + 18, 14), BEV_H - 22)), (255, 255, 255), C_DEC + (240,), F(12, True))
    pts = [m2v(0, 0)] + [m2v(x, y) for x, y in row["traj"]]
    d.line(pts, fill=C_PLAN + (255,), width=3, joint="curve")
    for h, (x, y) in zip(HORIZONS, row["traj"]):
        px, py = m2v(x, y)
        if h % 10 == 0:
            d.ellipse([px - 5, py - 5, px + 5, py + 5], fill=C_PLAN + (255,), outline=(20, 20, 20, 255))
        else:
            d.ellipse([px - 4, py - 4, px + 4, py + 4], outline=C_PLAN + (255,), width=2)
    e = pts[-1]
    _chip_v(d, f"MODEL plan (E9 #{row['sel_idx']})", (min(max(e[0] + 8, 2), BEV_W - 150), min(max(e[1] - 20, 14), BEV_H - 22)), (255, 255, 255), C_PLAN + (245,), F(12, True))
    for q, g4 in enumerate(row["g_tac"]):
        px, py = m2v(g4[0], g4[1])
        d.polygon([(px, py - 8), (px + 8, py), (px, py + 8), (px - 8, py)], fill=C_GOAL + (240,), outline=(40, 10, 45, 255))
        hx, hy = px - math.sin(g4[2]) * 16, py - math.cos(g4[2]) * 16
        d.line([(px, py), (hx, hy)], fill=C_GOAL + (255,), width=2)
        d.text((min(px + 10, BEV_W - 110), py - 6), f"MODEL g{2 * (q + 1)}s {g4[3]:.1f}m/s", fill=C_GOAL + (255,), font=F(11, True))
        if row["g_tac_gt_valid"][q]:
            gg = row["g_tac_gt"][q]
            gx, gy = m2v(gg[0], gg[1])
            d.ellipse([gx - 9, gy - 9, gx + 9, gy + 9], outline=C_GT + (255,), width=3)
            d.text((min(gx + 12, BEV_W - 50), gy + 4), f"GT {2 * (q + 1)}s", fill=C_GT + (255,), font=F(11, True))
    d.polygon([(ex - 7, ey), (ex + 7, ey), (ex, ey - 15)], fill=(240, 244, 250, 255))
    for r in (20, 40, 60, 80, 100):
        _, py = m2v(r, 0.0)
        lab = f"{r} m"
        tw = d.textlength(lab, font=F(11))
        d.rectangle([BEV_W - tw - 8, max(py, 8) - 7, BEV_W - 1, max(py, 8) + 6], fill=(9, 12, 17, 170))
        d.text((BEV_W - tw - 4, max(py, 8) - 7), lab, fill=(200, 208, 220, 255), font=F(11))
    txt = CEIL_STAMP
    d.rectangle([0, BEV_H - 20, BEV_W, BEV_H], fill=(9, 12, 17, 215))
    d.text((5, BEV_H - 17), _fitv(d, txt, F(11, True), BEV_W - 8), fill=(252, 165, 165) if row["plan_exceeds_ceiling"] else WARN, font=F(11, True))
    return im


def _bars(d, x, y, w, names, probs, pred_i, gt_i, row_h=20, name_w=196, label_scored=None, sort=False, thr_line=False,
          font_size=14):
    """Probability bars. MODEL: blue, the arg-max orange + a MODEL pill. GT: a green outline around the row + a GT pill."""
    F = _vfont
    order = sorted(range(len(probs)), key=lambda i: -probs[i]) if sort else list(range(len(probs)))
    bw = w - name_w - 150
    for r, i in enumerate(order):
        yy = y + r * row_h
        isgt = gt_i is not None and (i in gt_i if isinstance(gt_i, (list, tuple, set)) else i == gt_i)
        dimmed = label_scored is not None and label_scored[i] != "1"
        top = (i == pred_i)
        nm_col = DIM2 if dimmed else (C_GT if isgt else FG)
        d.text((x, yy + 1), _fitv(d, names[i], F(font_size, top), name_w - 6), fill=nm_col, font=F(font_size, top))
        bx0 = x + name_w
        d.rectangle([bx0, yy + 4, bx0 + bw, yy + row_h - 5], fill=(28, 36, 48))
        if probs[i] > 0:
            d.rectangle([bx0, yy + 4, bx0 + max(bw * probs[i], 1), yy + row_h - 5],
                        fill=(C_PLAN if top else (C_BOX if not dimmed else (60, 80, 120))))
        if thr_line:
            d.line([(bx0 + bw * 0.5, yy + 2), (bx0 + bw * 0.5, yy + row_h - 3)], fill=(229, 231, 235), width=1)
        d.text((bx0 + bw + 6, yy + 1), f"{100 * probs[i]:.1f}%", fill=(DIM2 if dimmed else FG), font=F(font_size - 1, top))
        px = bx0 + bw + 58
        if isgt:
            d.rectangle([x - 3, yy + 1, x + w - 8, yy + row_h - 1], outline=C_GT + (255,), width=1)
            _chip_v(d, "GT", (px, yy + 2), (6, 22, 12), C_GT + (255,), F(12, True), pad=3)
            px += 28
        if top:
            _chip_v(d, "MODEL", (px, yy + 2), (255, 255, 255), C_PLAN + (255,), F(12, True), pad=3)


def _strip_image(rows, width, height):
    """The whole-clip decision strip (cached per clip): MODEL / GT lat and lon, plan-vs-tactical, plan-vs-nav."""
    from PIL import Image, ImageDraw
    F = _vfont
    n = len(rows)
    im = Image.new("RGB", (width, height), (10, 14, 20))
    d = ImageDraw.Draw(im)
    left = 150
    cw = (width - left - 6) / n
    rh, gap = 11, 2
    spec = [("MODEL", "lat", "m"), ("GT", "lat", "g"), ("MODEL", "lon", "m"), ("GT", "lon", "g"),
            ("MODEL", "plan vs tactical", "a"), ("MODEL", "plan vs nav", "n")]
    for ri, (fam, nm, kind) in enumerate(spec):
        y = 2 + ri * (rh + gap)
        d.text((4, y - 2), fam, fill=(C_GT if fam == "GT" else C_PLAN), font=F(11, True))
        d.text((4 + d.textlength(fam, font=F(11, True)) + 6, y - 2), nm, fill=DIM, font=F(11))
        for i, r in enumerate(rows):
            x0 = left + i * cw
            x1 = left + (i + 1) * cw
            col = None
            if kind == "m":
                col = (LAT_RGB if nm == "lat" else LON_RGB)[int(np.argmax(r["tac_lat_p" if nm == "lat" else "tac_lon_p"]))]
            elif kind == "g":
                gi = r["tac_lat_gt" if nm == "lat" else "tac_lon_gt"]
                col = None if gi is None else (LAT_RGB if nm == "lat" else LON_RGB)[gi]
            elif kind == "a":
                a, b = r["plan3"]["lat_ok"], r["plan3"]["lon_ok"]
                col = (42, 51, 66) if (a is None and b is None) else ((239, 68, 68) if (a is False or b is False) else (59, 130, 246))
            else:
                a = r["plan3"]["nav_ok"]
                col = (42, 51, 66) if a is None else ((59, 130, 246) if a else (239, 68, 68))
            if col is None:
                d.rectangle([x0, y, x1 - 0.01, y + rh], fill=(20, 27, 38))
            else:
                d.rectangle([x0, y, x1 - 0.01, y + rh], fill=col)
            if kind == "g" and col is not None:
                pi = int(np.argmax(r["tac_lat_p" if nm == "lat" else "tac_lon_p"]))
                if pi != r["tac_lat_gt" if nm == "lat" else "tac_lon_gt"]:
                    d.rectangle([x0, y - 2, x1 - 0.01, y], fill=(255, 45, 45))
        d.rectangle([left - 1, y - 1, left + cw * n, y + rh + 1], outline=(C_GT if fam == "GT" else (200, 110, 40)))
    ya = 2 + len(spec) * (rh + gap) + 1
    t0, t1 = rows[0]["t_label_s"], rows[-1]["t_label_s"]
    for t in range(int(math.ceil(t0 / 5.0)) * 5, int(t1) + 1, 5):
        i = (t - t0) / max((t1 - t0) / max(n - 1, 1), 1e-9)
        x = left + (i + 0.5) * cw
        d.line([(x, ya), (x, ya + 4)], fill=DIM2)
        d.text((x - 8, ya + 4), f"{t} s", fill=DIM2, font=F(10))
    return im, left, cw


def _tactical_panel(d, cv, row, meta, k):
    """The LARGE tactical panel (left column below the camera). Returns the bottom y it used."""
    F = _vfont
    names = _V["names"]
    x0, w = 16, 1024
    y = 470
    d.rectangle([x0, y, x0 + w, 1076], fill=PANEL)
    x = x0 + 10
    y += 6
    # agreement chips
    p3 = row["plan3"]
    lat = names["tac_lat"][int(np.argmax(row["tac_lat_p"]))]
    lon = names["tac_lon"][int(np.argmax(row["tac_lon_p"]))]

    def chip3(val, yes, no, na):
        if val is None:
            return na, (30, 38, 52), (51, 66, 90), DIM
        if val:
            return "[ok] " + yes, (16, 35, 63), (59, 130, 246), (191, 219, 254)
        return "[X] " + no, (58, 20, 22), (239, 68, 68), (254, 202, 202)
    chips = [chip3(p3["nav_ok"], f"plan follows NAV {row['nav_cmd'].upper()}", f"plan ignores NAV {row['nav_cmd'].upper()}", "nav: n/a"),
             chip3(p3["lat_ok"], f"plan = tactical lat ({lat})", f"plan != tactical lat (decoder {lat}, plan {LAT3[p3['lat3']]})", f"tactical lat n/a ({lat})"),
             chip3(p3["lon_ok"], f"plan = tactical lon ({lon})", f"plan != tactical lon (decoder {lon}, plan {LON3[p3['lon3']]})", f"tactical lon n/a ({lon})")]
    cx, cy = x, y
    for txt, bgc, brc, fgc in chips:
        tw = d.textlength(txt, font=F(14, True)) + 16
        if cx + tw > x0 + w - 8:
            cx, cy = x, cy + 26
        d.rectangle([cx, cy, cx + tw, cy + 22], fill=bgc, outline=brc)
        d.text((cx + 8, cy + 3), txt, fill=fgc, font=F(14, True))
        cx += tw + 8
    y = cy + 28
    # lat | lon bars
    half = (w - 30) // 2
    for col, (title, nm, probs, gti, pi) in enumerate((
            ("LATERAL action - MODEL posterior vs GT label", names["tac_lat"], row["tac_lat_p"], row["tac_lat_gt"], int(np.argmax(row["tac_lat_p"]))),
            ("LONGITUDINAL action - MODEL posterior vs GT label", names["tac_lon"], row["tac_lon_p"], row["tac_lon_gt"], int(np.argmax(row["tac_lon_p"]))))):
        bx = x + col * (half + 10)
        d.text((bx, y), title, fill=FG, font=F(15, True))
        _bars(d, bx, y + 22, half, nm, probs, pi, gti, row_h=19)
        if gti is None:
            d.text((bx, y + 22 + 8 * 19 + 1), "GT: not labelled at this instant (outside the v7 label band)", fill=DIM, font=F(12))
    y += 22 + 8 * 19 + 20
    # goal tokens
    d.text((x, y), "GOAL TOKENS - MODEL validity (22 independent sigmoids) vs GT; white tick = 0.5", fill=FG, font=F(15, True))
    y += 22
    gp = row["goal_p"]
    order = sorted(range(22), key=lambda i: -gp[i])
    labelled = "1" in row["goal_scored"]
    gt_pos = row["goal_gt_pos"]
    for col in range(2):
        sub = order[col * 11:(col + 1) * 11]
        bx = x + col * (half + 10)
        for r, i in enumerate(sub):
            yy = y + r * 17
            isgt = labelled and i in gt_pos
            unsc = labelled and row["goal_scored"][i] != "1"
            d.text((bx, yy), _fitv(d, names["goal"][i], F(13, False), 222), fill=(DIM2 if unsc else (C_GT if isgt else FG)), font=F(13))
            bx0 = bx + 228
            bw = half - 228 - 112
            d.rectangle([bx0, yy + 3, bx0 + bw, yy + 13], fill=(28, 36, 48))
            d.rectangle([bx0, yy + 3, bx0 + max(bw * gp[i], 1), yy + 13], fill=((60, 80, 120) if unsc else C_BOX))
            d.line([(bx0 + bw * 0.5, yy + 1), (bx0 + bw * 0.5, yy + 15)], fill=(229, 231, 235), width=1)
            d.text((bx0 + bw + 6, yy), f"{100 * gp[i]:.0f}%", fill=(DIM2 if unsc else FG), font=F(12))
            if isgt:
                d.rectangle([bx - 3, yy, bx + half - 4, yy + 16], outline=C_GT + (255,))
                _chip_v(d, "GT", (bx0 + bw + 44, yy + 1), (6, 22, 12), C_GT + (255,), F(11, True), pad=2)
    y += 11 * 17 + 4
    note = ("GT goals: not labelled at this instant" if not labelled else
            "GT: green outline = a positive goal token; dimmed = unscored token (no supervised negatives / n < floor)")
    d.text((x, y), note, fill=DIM, font=F(12))
    y += 16
    # the whole-clip decision strip + cursor
    strip, left, cw = _V["strip"][meta["clip_rank"] - 1]
    d.text((x, y), _fitv(d, "DECISIONS OVER THE CLIP - cursor = now; red cap = MODEL decision differs from GT; plan rows: blue agrees, red disagrees",
                         F(13, True), w - 20), fill=FG, font=F(13, True))
    y += 17
    cv.paste(strip, (x, y))
    cxp = x + left + (k + 0.5) * cw
    d.line([(cxp, y - 2), (cxp, y + strip.size[1])], fill=(255, 255, 255, 255), width=2)
    return y + strip.size[1]


def _info_column(d, row, boxes):
    """Right text column next to the BEV: this frame's numbers, the tactical goals table, boxes, the fan."""
    F = _vfont
    x, y, w = 1552, 74, 352
    d.rectangle([x - 6, y - 2, x + w + 4, 874], fill=PANEL)
    d.text((x, y), "INPUTS (GIVEN) and the ceiling", fill=C_GIVEN, font=F(15, True))
    y += 22
    d.text((x, y), f"nav = {row['nav_cmd'].upper()} (GIVEN input, not a prediction)", fill=C_GIVEN, font=F(14, True))
    d.text((x, y + 19), f"ego speed v0 {row['v0_ms']:.1f} m/s ({3.6 * row['v0_ms']:.0f} km/h) at t0", fill=C_GIVEN, font=F(14))
    ceil_txt = ("no valid fed max speed (ceiling inert)" if row["ceil_kmh"] is None else
                f"fed max {row['v_max_raw_ms']:.1f} m/s -> ceiling {row['ceil_kmh']} km/h ({row['v_lim_ms']:.1f} m/s)")
    d.text((x, y + 38), _fitv(d, ceil_txt, F(14), w), fill=C_GIVEN, font=F(14))
    ptxt = (f"plan peak {row['plan_vmax_ms']:.1f} m/s" + ("" if row["v_lim_ms"] is None else
            (f" > {row['v_lim_ms']:.1f}: EXCEEDS" if row["plan_exceeds_ceiling"] else f" <= {row['v_lim_ms']:.1f}: ok")))
    _chip_v(d, "MODEL", (x, y + 59), (255, 255, 255), C_PLAN + (255,), F(12, True), pad=3)
    d.text((x + 58, y + 59), _fitv(d, ptxt, F(14, True), w - 58), fill=((252, 165, 165) if row["plan_exceeds_ceiling"] else FG), font=F(14, True))
    for q, sub in enumerate(_wrap_v(d, CEIL_STAMP, F(12), w)):
        d.text((x, y + 80 + 15 * q), sub, fill=WARN, font=F(12))
    y += 118
    d.line([(x, y), (x + w, y)], fill=(40, 48, 60))
    y += 8
    d.text((x, y), "THIS FRAME", fill=FG, font=F(15, True))
    _chip_v(d, "MODEL", (x + 118, y + 1), (255, 255, 255), C_PLAN + (255,), F(12, True))
    d.text((x + 178, y), "vs", fill=DIM, font=F(13))
    _chip_v(d, "GT", (x + 202, y + 1), (6, 22, 12), C_GT + (255,), F(12, True))
    y += 26
    fm = lambda v, f=".2f": "-" if v is None else format(v, f)
    d.text((x, y), "ADE 8 slots", fill=DIM, font=F(12))
    d.text((x + 170, y), "FDE @ 6 s", fill=DIM, font=F(12))
    d.text((x, y + 14), fm(row["ade_m"]) + " m", fill=C_PLAN, font=F(28, True))
    d.text((x + 170, y + 14), fm(row["fde_m"]) + " m", fill=C_PLAN, font=F(28, True))
    y += 54
    d.text((x, y), f"speed err 0-2 s {fm(row['speed_mae_0_2s'])} m/s", fill=FG, font=F(14))
    d.text((x, y + 19), f"heading err 0-2 s {fm(row['heading_mae_0_2s_deg'], '.1f')} deg", fill=FG, font=F(14))
    d.text((x, y + 38), f"curvature err 0-2 s {fm(row['curv_mae_0_2s'], '.4f')} 1/m", fill=FG, font=F(14))
    y += 66
    d.line([(x, y), (x + w, y)], fill=(40, 48, 60))
    y += 8
    d.text((x, y), _fitv(d, "TACTICAL GOALS g_tac (MODEL / GT)", F(14, True), w), fill=FG, font=F(14, True))
    y += 22
    for q in range(3):
        g4 = row["g_tac"][q]
        d.text((x, y), f"MODEL {2 * (q + 1)}s", fill=C_GOAL, font=F(13, True))
        d.text((x + 74, y), f"{g4[0]:6.1f} {g4[1]:6.1f} {math.degrees(g4[2]):6.1f} {g4[3]:5.1f}", fill=FG, font=F(13))
        y += 17
        if row["g_tac_gt_valid"][q]:
            gg = row["g_tac_gt"][q]
            d.text((x, y), f"GT    {2 * (q + 1)}s", fill=C_GT, font=F(13, True))
            d.text((x + 74, y), f"{gg[0]:6.1f} {gg[1]:6.1f} {math.degrees(gg[2]):6.1f} {gg[3]:5.1f}", fill=FG, font=F(13))
        else:
            d.text((x, y), f"GT    {2 * (q + 1)}s", fill=C_GT, font=F(13, True))
            d.text((x + 74, y), "not valid (clip ends)", fill=DIM, font=F(13))
        y += 20
    d.text((x, y), "columns: x m, y m, heading deg, speed m/s (ego frame at now)", fill=DIM2, font=F(11))
    y += 20
    d.line([(x, y), (x + w, y)], fill=(40, 48, 60))
    y += 8
    d.text((x, y), _fitv(d, f"BOXES - MODEL p>={_V['thr']:.3f}, NMS {NMS_RADIUS_M:.0f} m, vs GT", F(14, True), w), fill=FG, font=F(14, True))
    y += 22
    if boxes["label"]:
        npos = boxes["npos"]
        n_show = len(boxes["idx"])
        d.text((x, y), f"shown {n_show}  TP {boxes['tp']}  FP {boxes['fp']}  DontCare {boxes['dc']}", fill=FG, font=F(14))
        d.text((x, y + 19), f"GT positives (VIS-1) {npos}  recall {fm(boxes['tp'] / npos if npos else None)}  precision {fm(boxes['tp'] / max(boxes['tp'] + boxes['fp'], 1) if n_show else None)}", fill=FG, font=F(13))
    else:
        d.text((x, y), "no agent label on this frame - matches undefined", fill=WARN, font=F(13))
    y += 46
    d.line([(x, y), (x + w, y)], fill=(40, 48, 60))
    y += 8
    d.text((x, y), "FAN - top-3 by the E9 score", fill=FG, font=F(14, True))
    y += 22
    top = fan_top(row)
    for rk, kk in enumerate(top):
        col = lerp_colour(FAN_HI, FAN_LO, rk / max(TOPK_VIDEO - 1, 1))
        d.line([(x, y + 8), (x + 26, y + 8)], fill=col, width=4)
        tag = " = emitted plan" if kk == row["sel_idx"] else ""
        d.text((x + 34, y), f"#{rk + 1}: cand {kk}  E9 {row['e9'][kk]:.2f}  dec {row['dec'][kk]:.2f}{tag}", fill=FG, font=F(13))
        y += 19
    if row["core_sel_idx"] is not None and row["core_sel_idx"] != row["sel_idx"]:
        d.line([(x, y + 8), (x + 26, y + 8)], fill=C_DEC, width=4)
        d.text((x + 34, y), f"decoder pick: cand {row['core_sel_idx']}  (E9 re-ranked it)", fill=FG, font=F(13))
        y += 19
    d.text((x, y), f"117 candidates: reach {row['reach'].count('1')}  ceiling {row['ceilk'].count('1')}", fill=DIM, font=F(12))
    y += 16
    d.text((x, y), "(the full fan: use the replay tool)", fill=DIM2, font=F(12))
    return y


def render_video_frame(job):
    from PIL import Image, ImageDraw
    from tanitad.viz_standard import VizElement, check_frame
    ci, k = job
    F = _vfont
    meta = _V["clips"][ci]
    row = _V["rows"][ci][k]
    z = np.load(_V["bank"] / "win" / f"c{ci:02d}_w{k:04d}.npz")
    cam = z["cam"]
    boxes = shown_boxes(row, _V["thr"])
    cv = Image.new("RGB", (W_TOT, H_TOT), BG)
    d = ImageDraw.Draw(cv, "RGBA")
    d.rectangle([0, 0, W_TOT, 40], fill=BAN)
    head = f"{RUN_LABEL} - FINAL step {_V['step']:,} - TACTICAL view"
    d.text((16, 9), head, fill=FG, font=F(19, True))
    d.text((16 + d.textlength(head, font=F(19, True)) + 16, 12),
           "OPEN-LOOP replay of held-out eval139 windows - NOT closed-loop driving", fill=WARN, font=F(15, True))
    rt = f"clip {ci + 1}/{len(_V['clips'])} - sha12 {meta['sha12']} - window {k + 1}/{len(_V['rows'][ci])} - t = {row['t_label_s']:.1f} s"
    d.text((W_TOT - 16 - d.textlength(rt, font=F(17, True)), 11), rt, fill=FG, font=F(17, True))
    cv.paste(_cam_overlay_v(row, cam, meta, boxes), (CAM_X, CAM_Y))
    cv.paste(_bev_v(row, z, boxes), (BEV_X, BEV_Y))
    d.text((BEV_X, 48), "BEV - MODEL predicted map (context) + overlays", fill=FG, font=F(15, True))
    d.rectangle([BEV_X - 1, BEV_Y - 1, BEV_X + BEV_W, BEV_Y + BEV_H], outline=(60, 70, 84))
    for yl, lab in ((30.0, "30 m L"), (15.0, "15"), (0.0, "0"), (-15.0, "15"), (-30.0, "30 m R")):
        px, _ = m2v(0, yl)
        tw = d.textlength(lab, font=F(11))
        d.text((min(max(BEV_X + px - tw / 2, BEV_X), BEV_X + BEV_W - tw), BEV_Y + BEV_H + 2), lab, fill=DIM2, font=F(11))
    bot = _tactical_panel(d, cv, row, meta, k)
    if bot > 1078:
        raise SystemExit(f"[rv7r] the tactical panel overflows the frame (bottom {bot})")
    _info_column(d, row, boxes)
    # legend + notes under the BEV
    ly = BEV_Y + BEV_H + 22
    d.text((BEV_X, ly), "ONE RULE:", fill=FG, font=F(14, True))
    xx = BEV_X + 92
    _chip_v(d, "GT", (xx, ly), (6, 22, 12), C_GT + (255,), F(13, True))
    d.text((xx + 30, ly), "green, dashed / outline", fill=DIM, font=F(13))
    xx += 200
    _chip_v(d, "MODEL", (xx, ly), (255, 255, 255), C_PLAN + (255,), F(13, True))
    d.text((xx + 62, ly), "orange / blue, solid / filled", fill=DIM, font=F(13))
    ly += 24
    for txt in ("orange = emitted plan (E9 pick) - blue ramp = top-3 fan (rank 1 = the plan) - blue fill = detections (score)",
                "fuchsia diamond = MODEL tactical goal g_tac (2/4/6 s) - green ring = GT goal label",
                "dashed green = GT path (dots 1 s) - map: MODEL-predicted 10 cm classes, dimmed"):
        d.text((BEV_X, ly), _fitv(d, txt, F(12), W_TOT - BEV_X - 16), fill=DIM, font=F(12))
        ly += 17
    ly += 6
    d.text((BEV_X, ly), "STRIP KEY (decision colours; NOT the GT/MODEL colours)", fill=FG, font=F(12, True))
    ly += 17
    for nm_key, cols, nms in (("lat", LAT_RGB, _V["names"]["tac_lat"]), ("lon", LON_RGB, _V["names"]["tac_lon"])):
        xx = BEV_X
        d.text((xx, ly), nm_key, fill=DIM, font=F(11, True))
        xx += 30
        for c_, n_ in zip(cols, nms):
            d.rectangle([xx, ly + 2, xx + 10, ly + 12], fill=c_)
            d.text((xx + 14, ly), n_, fill=DIM, font=F(11))
            xx += 20 + d.textlength(n_, font=F(11)) + 6
        ly += 16
    els = [
        VizElement.present("camera", "GT path + emitted plan + top-3 fan + GT/model boxes + goals projected",
                           source="render_refcv3_video.CylProjector (paths) + model RigCamera (boxes)", kind="derived"),
        VizElement.present("bev", "MODEL map + plan + top-3 fan + boxes + goals, metric",
                           source="map_head_hires.decide(prior_corrected) | out['traj'] | box slots", kind="model_output",
                           conditioned_on=("nav_cmd", "v_max_ms")),
        VizElement.present("tactical", f"lat {_V['names']['tac_lat'][int(np.argmax(row['tac_lat_p']))]} / lon "
                                       f"{_V['names']['tac_lon'][int(np.argmax(row['tac_lon_p']))]}",
                           source="out['tacv6_lat_logits'/'tacv6_lon_logits'].argmax(-1); goal tokens = sigmoid(out['tacv6_goal_logits'])",
                           kind="model_output", conditioned_on=("nav_cmd", "v_max_ms")),
        (VizElement.present("tactical_gt", "v7 lat/lon label", source="item['lat_v7'] / item['lon_v7']", kind="gt_label")
         if (row["tac_lat_gt"] is not None and row["tac_lon_gt"] is not None) else
         VizElement.unavailable("tactical_gt", "outside the label record's band at this instant (IGNORE)")),
        VizElement.unavailable("strategic", "--no-strategic: the strategic level is OFF in this arm; PhysicalAI-AV has no strategic label"),
        VizElement.present("ade", f"ADE {row['ade_m']} m", source="||out['traj'] - waypoint_targets||", kind="derived"),
        VizElement.present("strategic_input", row["nav_cmd"], source="V3Dataset.enable_nav_from_v7 (oracle nav)", kind="given_input"),
    ]
    check_frame(els, where=f"refcv7-replay @ {meta['sha12']}/w{k}")
    arr = np.asarray(cv)
    cam_reg = arr[CAM_Y + 20:CAM_Y + CAM_H, CAM_X:CAM_X + CAM_W]
    bev_reg = arr[BEV_Y:BEV_Y + BEV_H, BEV_X:BEV_X + BEV_W]
    if float(cam_reg.mean()) < 8.0 or float(cam_reg.std()) < 4.0 or float(bev_reg.std()) < 3.0:
        raise SystemExit(f"[rv7r] frame {meta['sha12']} w{k}: camera / BEV region is blank")
    return arr


def video_summary(rows: list) -> dict:
    """Per-clip numbers for the title card (plain means / counts over the clip's windows; one episode)."""
    def acc(pk, gk):
        lw = [r for r in rows if r[gk] is not None]
        if not lw:
            return None
        return sum(int(np.argmax(r[pk])) == r[gk] for r in lw), len(lw)

    def cnt(sub):
        v = [r["plan3"][sub] for r in rows if r["plan3"][sub] is not None]
        return (sum(bool(x) for x in v), len(v))
    return {"n": len(rows), "t0": rows[0]["t_label_s"], "t1": rows[-1]["t_label_s"],
            "ade_mean": float(np.mean([r["ade_m"] for r in rows])), "lat": acc("tac_lat_p", "tac_lat_gt"),
            "lon": acc("tac_lon_p", "tac_lon_gt"), "nav_ok": cnt("nav_ok"), "lat_ok": cnt("lat_ok"),
            "lon_ok": cnt("lon_ok"), "n_exceed": sum(bool(r["plan_exceeds_ceiling"]) for r in rows),
            "n_core_ne_sel": sum(1 for r in rows if r["core_sel_idx"] is not None and r["core_sel_idx"] != r["sel_idx"])}


def gt_control(rows_all: list) -> dict:
    """KNOWN-VALUE CONTROL for the plan -> (lat3, lon3) reading: the SAME rule applied to the GT path, scored against
    the GT v7 labels through the SAME mapping tables, on the labelled windows."""
    lat = [r["gt3"]["lat_ok"] for r in rows_all if r["gt3"]["lat_ok"] is not None]
    lon = [r["gt3"]["lon_ok"] for r in rows_all if r["gt3"]["lon_ok"] is not None]
    return {"lat": [int(sum(lat)), len(lat)], "lon": [int(sum(lon)), len(lon)]}


def _wrap_v(d, text, f, w):
    return _V["rcv3"].wrap(d, text, f, w)


def _opening_card_v(rec, control):
    from PIL import Image, ImageDraw
    F = _vfont
    im = Image.new("RGB", (W_TOT, H_TOT), BG)
    d = ImageDraw.Draw(im)
    x = 110
    d.text((x, 54), f"{RUN_LABEL} - tactical behaviour and goals, FINAL checkpoint", fill=FG, font=F(40, True))
    ck = rec["ckpt"]
    lines = [
        (f"step {_V['step']:,} (ck['step'] asserted, strict load 0 missing / 0 unexpected) - ckpt.pt md5 {ck['md5']} - launch tree fec3a0dccf", DIM, 15, False),
        ("", FG, 10, False),
        (TIER_NOTE.split(" (")[0].replace(" — ", " - "), WARN, 24, True),
        ("Each frame is one recorded held-out eval139 window. The model sees the camera window and the measured speed at t0 and emits a 6 s plan; "
         "the plan never drives the car, and the next frame comes from the recording.", FG, 17, False),
        ("(!) " + CEIL_STAMP + ": the decoder masks over-ceiling candidates in its own ranking, the E9 goal selection re-ranks the fan without that mask.", WARN, 17, True),
        ("", FG, 10, False),
        ("WHAT YOU SEE", FG, 22, True),
        ("- TACTICAL BEHAVIOUR (large panel, left): the refcv6 behaviour decoder's lateral and longitudinal posteriors (8 + 8 actions) and its 22 goal-token validities, "
         "with the v7 GT label marked. Three chips say whether the EMITTED PLAN agrees with the given nav token and with the decoder's own lat / lon decision.", DIM, 16, False),
        ("- DECISIONS OVER THE CLIP (strip, bottom-left): MODEL and GT lat / lon decision per window, with a red cap wherever they differ; the cursor is now. "
         "A late or wrong decision is visible at a glance.", DIM, 16, False),
        ("- CAMERA + BEV: the emitted plan (orange), the GT future path (green, dashed), the top-3 fan only (blue ramp), detections at the run's own P=R gate "
         f"p >= {_V['thr']:.4f} with BEV NMS 2 m (MODEL: blue, filled, with score; GT: green outline), and the tactical geometric goals g_tac at 2 / 4 / 6 s (fuchsia diamond = MODEL, green ring = GT).", DIM, 16, False),
        ("", FG, 10, False),
        ("KNOWN LIMITS (not hidden)", FG, 22, True),
        ("- The plan -> (lat, lon) agreement chips are DERIVED by this tool (the programme's factor_from_kinematics v2 rule at the label's own 2 s horizon, plus the nav-compliance predicate) "
         f"and are coarse. Known-value control: the SAME rule on the GT path agrees with the GT v7 labels on lat {control['lat'][0]}/{control['lat'][1]} and lon {control['lon'][0]}/{control['lon'][1]} labelled windows.", DIM, 16, False),
        ("- Only 7 of the 22 goal tokens are both trainable and scoreable (FOLLOW_LANE, TURN_L, TURN_R, STOP_POINT, EVADE_IN_CORRIDOR, TRAFFIC_LIGHT_REACT_RED / _GREEN); "
         "the others are dimmed when a GT label exists. lat / lon GT labels exist only within +-2 s of each clip's anchor.", DIM, 16, False),
        ("- One DDIM draw per window (seed 0); strategic level OFF in this arm (UNAVAILABLE). 12 clips chosen by a rule fixed before rendering; no interval is quoted: one clip = one episode.", DIM, 16, False),
    ]
    y = 118
    for txt, col, size, bold in lines:
        if not txt:
            y += size
            continue
        for sub in _wrap_v(d, txt, F(size, bold), W_TOT - 2 * x):
            d.text((x, y), sub, fill=col, font=F(size, bold))
            y += int(size * 1.4)
    y += 14
    d.text((x, y), "COLOUR RULE", fill=FG, font=F(22, True))
    y += 36
    _chip_v(d, "GT", (x, y), (6, 22, 12), C_GT + (255,), F(18, True), pad=6)
    d.text((x + 56, y), "ground truth: green, dashed or outline only", fill=FG, font=F(18))
    y += 34
    _chip_v(d, "MODEL", (x, y), (255, 255, 255), C_PLAN + (255,), F(18, True), pad=6)
    d.text((x + 100, y), "model output: orange (emitted plan) / blue (candidates, detections), solid or filled", fill=FG, font=F(18))
    y += 34
    d.text((x, y), "amber text = a GIVEN input (nav token, measured speed, fed ceiling)", fill=C_GIVEN, font=F(18))
    return im


def _title_card_v(ci, summ, meta):
    from PIL import Image, ImageDraw
    F = _vfont
    im = Image.new("RGB", (W_TOT, H_TOT), BG)
    d = ImageDraw.Draw(im)
    x = 120
    d.text((x, 150), f"Clip {ci + 1} of {len(_V['clips'])}  -  nav {meta['nav'].upper()} (GIVEN input)", fill=FG, font=F(40, True))
    s = meta["selection"]
    d.text((x, 212), f"sha12 {meta['sha12']}  -  {summ['n']} eval windows in time order  -  t {summ['t0']:.1f} ... {summ['t1']:.1f} s  -  v0 {s['v_min']:.1f} ... {s['v_max']:.1f} m/s",
           fill=DIM, font=F(20, True))
    d.text((x, 258), "why chosen: " + meta["why"], fill=C_GIVEN, font=F(17))
    d.text((x, 300), f"{RUN_LABEL} - step {_V['step']:,} - OPEN-LOOP replay, not closed-loop driving", fill=WARN, font=F(17))
    ff = lambda t: "no labelled window" if t is None else f"{t[0]}/{t[1]}"
    L_ = [
        f"tactical decoder vs v7 GT (labelled windows only, argmax): lat {ff(summ['lat'])}  -  lon {ff(summ['lon'])}",
        f"emitted plan agrees with the given nav: {ff(summ['nav_ok'])}  -  with the decoder's lat: {ff(summ['lat_ok'])}  -  with its lon: {ff(summ['lon_ok'])}   (windows where defined)",
        f"E9 pick differs from the decoder's own pick on {summ['n_core_ne_sel']}/{summ['n']} windows  -  emitted plan exceeds the fed ceiling on {summ['n_exceed']}/{summ['n']} windows ({CEIL_STAMP})",
        f"plan error: ADE mean {summ['ade_mean']:.2f} m over the clip (a viewing aid on one episode, no interval)",
    ]
    y = 380
    for ln in L_:
        for sub in _wrap_v(d, ln, F(19), W_TOT - 2 * x - 20):
            d.text((x + 20, y), sub, fill=FG, font=F(19))
            y += 28
        y += 18
    return im


def stage_video(a) -> int:  # noqa: C901
    t_all = time.time()
    tree, work = Path(a.tree), Path(a.work)
    tag = "smoke" if a.smoke else "final"
    bank = work / "bank" / tag
    if not (bank / "CAPTURE_DONE").exists():
        raise SystemExit(f"[rv7r] {bank}/CAPTURE_DONE is missing -- run --stage capture first")
    out_dir = work / "out" / tag / "video"
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(tree / "stack"))
    import av
    import torch
    from PIL import Image
    from tanitad.data.calib import CanonicalFrame
    from tanitad.data.rig_projection import RigCamera
    torch.set_num_threads(1)
    rcv3 = rv7._by_path("render_refcv3_video_rv7", tree / "taniteval/tools/render_refcv3_video.py")
    rv6 = rv7._by_path("render_refcv6_map_video_rv7", tree / "taniteval/tools/render_refcv6_map_video.py")
    rec = json.loads((bank / "capture_record.json").read_text(encoding="utf-8"))
    rows_all = [json.loads(ln) for ln in open(bank / "rows.jsonl", encoding="utf-8") if ln.strip()]
    clips = rec["clips"]
    rows = [[r for r in rows_all if r["clip_rank"] == ci + 1] for ci in range(len(clips))]
    for ci, rr in enumerate(rows):
        if [r["win"] for r in rr] != list(range(len(rr))) or len(rr) != clips[ci]["n_windows_rendered"]:
            raise SystemExit(f"[rv7r] clip {ci} rows are incomplete / out of order")
    projs, rigs = {}, {}
    for m in clips:
        fr = m["frame"]
        rig = RigCamera(R_cam_to_rig=torch.tensor(m["rig_camera"]["R_cam_to_rig"], dtype=torch.float64),
                        t_cam_in_rig=torch.tensor(m["rig_camera"]["t_cam_in_rig"], dtype=torch.float64),
                        frame=CanonicalFrame(height=fr["height"], width=fr["width"], f_ref=fr["f_ref"],
                                             projection=fr["projection"]))
        tv = m["js_test_vector"]
        pc, pr_, pok = rig.project(torch.tensor(test_vector_probe(m["clip_rank"] - 1), dtype=torch.float64))
        dmax = max(abs(float(pc[i]) - q[0]) + abs(float(pr_[i]) - q[1]) for i, q in enumerate(tv["col_row_ok"]) if q[2])
        if dmax > 1e-6:
            raise SystemExit(f"[rv7r] rebuilt RigCamera for {m['sha12']} differs from the model's by {dmax}")
        rigs[m["sha12"]] = rig
        projs[m["sha12"]] = rcv3.CylProjector(fr, m["extrinsics"])
    names = rec["dataset"]["label_names"]
    thr = float(rec["detection_display"]["box3d_threshold"])
    _V.update({"rcv3": rcv3, "rv6": rv6, "bank": bank, "clips": clips, "rows": rows, "thr": thr, "names": names,
               "proj": projs, "rig": rigs, "step": int(a.expect_step)})
    _V["strip"] = [_strip_image(rows[ci], 1000, 104) for ci in range(len(clips))]
    summaries = [video_summary(rows[ci]) for ci in range(len(clips))]
    control = gt_control(rows_all)
    _p(f"[rv7r] GT-path control (same rule, GT labels): {control}")
    n_open, n_title = int(round(a.open_s * FPS)), int(round(a.title_s * FPS))
    seq = [("open", None)] * n_open
    for ci in range(len(clips)):
        seq += [("title", ci)] * n_title
        seq += [("win", (ci, k)) for k in range(len(rows[ci]))]
    if a.smoke:
        n_open, n_title = 1, 1
        seq = [("open", None), ("title", 0)] + [("win", (0, k)) for k in range(len(rows[0]))]
    reel = out_dir / ("refcv7_final_50400_tactical.mp4" if not a.smoke else "refcv7_smoke_tactical.mp4")
    cont = av.open(str(reel), mode="w", options={"movflags": "+faststart"})
    st = cont.add_stream("libx264", rate=FPS)
    st.width, st.height, st.pix_fmt = W_TOT, H_TOT, "yuv420p"
    st.options = {"crf": str(a.crf), "preset": a.preset}
    kf_dir = out_dir / "keyframes"
    kf_dir.mkdir(exist_ok=True)
    cards = {"open": np.asarray(_opening_card_v(rec, control))}
    for ci in range(len(clips)):
        cards[("title", ci)] = np.asarray(_title_card_v(ci, summaries[ci], clips[ci]))
    Image.fromarray(cards["open"]).save(kf_dir / "card_opening.png")
    Image.fromarray(cards[("title", 0)]).save(kf_dir / "card_title_clip01.png")
    jobs = [j for kind, j in seq if kind == "win"]
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    kf_set = set()
    for ci in range(len(clips)):
        nw = len(rows[ci])
        kf_set |= {(ci, int(round(q * (nw - 1)))) for q in (0.10, 0.35, 0.60, 0.85)}
    n_frames = 0

    def emit(arr):
        nonlocal n_frames
        fr_ = av.VideoFrame.from_ndarray(arr, format="rgb24")
        for pkt in st.encode(fr_):
            cont.mux(pkt)
        n_frames += 1
    from collections import deque
    t_r = time.time()
    with ctx.Pool(processes=a.workers) as pool:
        pending, nxt = deque(), 0
        max_inflight = 3 * a.workers
        for kind, j in seq:
            if kind == "open":
                emit(cards["open"])
            elif kind == "title":
                emit(cards[("title", j)])
            else:
                while nxt < len(jobs) and len(pending) < max_inflight:
                    pending.append(pool.apply_async(render_video_frame, (jobs[nxt],)))
                    nxt += 1
                arr = pending.popleft().get()
                emit(arr)
                if j in kf_set:
                    Image.fromarray(arr).save(kf_dir / f"c{j[0] + 1:02d}_w{j[1] + 1:04d}.png")
            if n_frames % 200 == 0:
                _p(f"[rv7r] {n_frames}/{len(seq)} frames ({time.time() - t_r:.0f} s)")
    for pkt in st.encode():
        cont.mux(pkt)
    cont.close()
    if n_frames != len(seq):
        raise SystemExit(f"[rv7r] encoded {n_frames} frames, expected {len(seq)}")
    dec = av.open(str(reel))
    n_dec = 0
    for _f in dec.decode(dec.streams.video[0]):
        if n_dec == 0 and (_f.width, _f.height) != (W_TOT, H_TOT):
            raise SystemExit(f"[rv7r] decoded {_f.width}x{_f.height}")
        n_dec += 1
    dec.close()
    if n_dec != n_frames:
        raise SystemExit(f"[rv7r] decoded {n_dec} frames, wrote {n_frames}")
    vrec = {"tool": "taniteval/tools/render_refcv7_replay.py", "stage": "video", "tag": tag,
            "tool_md5": rv7.md5_file(__file__), "video": {
                "path": str(reel), "bytes": reel.stat().st_size, "md5": rv7.md5_file(reel), "frames": n_frames,
                "decoded_frames": n_dec, "fps": FPS, "duration_s": round(n_frames / FPS, 1),
                "resolution": f"{W_TOT}x{H_TOT}", "codec": "libx264 yuv420p (PyAV " + av.__version__ + ")",
                "crf": a.crf, "preset": a.preset, "open_card_frames": n_open, "title_frames_per_clip": n_title},
            "gt_path_control": control,
            "per_clip": [{"clip_rank": ci + 1, "sha12": clips[ci]["sha12"], "nav": clips[ci]["nav"], **summaries[ci]}
                         for ci in range(len(clips))],
            "wall_s": round(time.time() - t_all, 1), "workers": a.workers,
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    rv7.write_json(out_dir / "video_record.json", vrec)
    _p(f"[rv7r] VIDEO DONE: {reel} {vrec['video']['bytes']:,} B, {n_frames} frames, {time.time() - t_all:.0f} s")
    return 0


# ============================================================================================= #
# main                                                                                           #
# ============================================================================================= #
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--stage", choices=("capture", "assets", "video"), required=True)
    for k, v in DEFAULTS.items():
        ap.add_argument("--" + k, default=v)
    ap.add_argument("--expect-step", type=int, default=EXPECT["step"])
    ap.add_argument("--expect-ckpt-md5", default=EXPECT["ckpt_md5"])
    ap.add_argument("--seed", type=int, default=0, help="inference seed, set before EVERY window")
    ap.add_argument("--smoke", action="store_true", help="1 clip x --smoke-windows; outputs under */smoke")
    ap.add_argument("--smoke-windows", type=int, default=6)
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--no-compare", action="store_true", help="skip the comparison against the reel's bank")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--jpeg-quality", type=int, default=82)
    ap.add_argument("--crf", type=int, default=19)
    ap.add_argument("--preset", default="slow")
    ap.add_argument("--open-s", type=float, default=10.0)
    ap.add_argument("--title-s", type=float, default=3.0)
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:                                           # noqa: BLE001
        pass
    if a.stage == "capture":
        return stage_capture(a)
    if a.stage == "assets":
        return stage_assets(a)
    return stage_video(a)


if __name__ == "__main__":
    sys.exit(main())
