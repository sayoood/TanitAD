#!/usr/bin/env python3
"""render_refcv7_video.py -- refcv7-r101-s0 at its FINAL checkpoint (step 50,400), frame by frame.

PI (Sayed), verbatim: *"Generate a long video showing the performance frame by frame including high
quality visualizations, overlays and bev representation (no static images)"*.

⛔ READ ``RENDER_REFCV7_VIDEO.md`` (beside this file) FIRST. It fixes the clip rule BEFORE any render,
names every source by file, and states what this reel is NOT: it is OPEN-LOOP perception + planning on
the held-out eval139 clips -- never closed-loop driving, never a four-family eval result.

TWO STAGES, ONE FILE
--------------------
``--stage forward``  (Thor, under ``flock /home/nvidia/refcv7_post/thor_gpu.lock``) builds the model with
   the LAUNCH TREE's own ``stack/tanitad/eval/refcv7_loader.py`` (strict load, step asserted), builds the
   held-out eval split exactly as ``train()`` does, applies the FIXED clip rule (GT and labels only, before
   any model output exists), then runs EVERY eval window of every chosen clip through the trainer's own
   ``compute_losses_v3`` in eval mode. A NON-raising forward hook captures ``out``; the loss path's own
   eval extras supply the map per-class counts and the detection packs, so every per-frame metric is the
   trainer's arithmetic. Per window it banks the camera bytes, the 10 cm map under the DECLARED decision
   rule, the SAM3 codes, the lift mask (npz) and one JSON row.
``--stage render``   (Thor CPU, no lock) draws every frame from the bank (parallel workers), encodes the
   reel with libx264 through PyAV (Thor has no ffmpeg binary), then the <= 15 MB preview by decoding the
   reel and re-encoding at a size-targeted bitrate.

Clip ids never leave the process: every artifact carries ``sha12 = sha256(clip_id)[:12]`` only, and every
JSON written is scrubbed and REFUSED if a raw UUID survives.

The module imports numpy only at top level, so the pure functions below are testable on any box
(``taniteval/tests/test_render_refcv7_video.py``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

# ============================================================================================= #
# identity                                                                                       #
# ============================================================================================= #
RUN_LABEL = "refcv7-r101-s0"
TIER_NOTE = ("OPEN-LOOP perception + planning on the held-out eval139 clips — NOT closed-loop "
             "driving (T0/T1 are both open loop; EVAL_DOCTRINE.md)")
CEIL_NOTE = ("the speed ceiling does NOT reach the emitted plan on this launch tree "
             "(SPEC_REFCV7 §26.1)")
DEFAULTS = {
    "tree": "/home/nvidia/refcv7_run/fec3a0dccf",
    "run": "/home/nvidia/refcv7_run/runs/refcv7-r101-s0",
    "kit": "/home/nvidia",
    "work": "/home/nvidia/refcv7_post/video",
}
EXPECT = {"step": 50400, "ckpt_md5": "d5f104ee54ba6b2861e38030b4f6fcf1"}

# ============================================================================================= #
# the 10 cm map (semantic_map_gt_fine.EXTENT_REFCV7 / FINE_CLASSES; asserted at run time)         #
# ============================================================================================= #
X_MAX_M, Y_HALF_M, CELL_M = 100.0, 30.0, 0.1
GRID_X, GRID_Y = 1000, 600          # row 0 = x in [0, 0.1) m (NEAREST); col 0 = y in [-30, -29.9) (RIGHT)
NOT_SEEN = 255
MAP_CLASSES = ("seen, no class", "drivable", "lane / road line", "crosswalk", "arrow / text",
               "non-drivable edge", "hatched area", "sidewalk / verge")
CLASS_KEYS = ("nocls", "drivable", "lane", "crosswalk", "arrow", "edge", "hatched", "sidewalk")
ROAD_SURFACE = (1, 2, 3, 4)          # orientation control: cells a driven path may legitimately cross
HORIZONS = (5, 10, 15, 20, 30, 40, 50, 60)
NAV_ORDER = ("left", "right", "follow")
AGENT_CLASSES = ("automobile", "heavy_truck", "bus", "other_vehicle", "trailer", "person",
                 "rider", "stroller", "animal", "protruding_object")
VRU_CLASSES = (5, 6, 7)              # person, rider, stroller
VEHICLE_CLASSES = (0, 1, 2, 3, 4)
VRU_RANGE_M = 30.0
FOV_HALF_DEG = 60.0
LEAD_LANE_HALF_M = 1.75
LEAD_MAX_X_M = 60.0
TOPK_FAN = 8
SAVE_SLOT_P_MIN = 0.10               # slots banked per window (the draw threshold is higher, stated)

# ============================================================================================= #
# colour semantics -- one meaning per colour                                                     #
# ============================================================================================= #
C_GT = (110, 231, 138)        # GROUND TRUTH: path, boxes -- everywhere, always green
C_SEL = (249, 115, 22)        # refcv7's OWN emitted plan out['traj'] -- orange (L1 > 90 from amber)
C_FAN = (0, 229, 255)         # the top-k candidates of the SAME fan, faded cyan
C_GIVEN = (240, 190, 90)      # a GIVEN INPUT (nav token, fed max speed) -- amber, text only
PALETTE = ((78, 86, 100), (46, 98, 178), (240, 240, 240), (226, 88, 200),
           (150, 110, 230), (214, 64, 64), (64, 196, 206), (150, 116, 84))
AGENT_PALETTE = ((80, 170, 255), (255, 90, 90), (250, 235, 70), (200, 120, 255),
                 (255, 120, 200), (255, 255, 255), (120, 255, 230), (255, 200, 150),
                 (170, 130, 90), (160, 160, 175))
HATCH_BG, HATCH_A, HATCH_B = (14, 16, 21), (70, 76, 90), (150, 140, 96)
C_BG, C_PANEL, C_BAN = (9, 12, 17), (16, 21, 29), (15, 20, 28)
C_FG, C_DIM, C_DIM2, C_WARN = (233, 238, 245), (140, 152, 168), (104, 116, 132), (245, 180, 90)
C_OK, C_BAD, C_GRID = (110, 231, 138), (248, 113, 113), (40, 48, 60)
DIM_FACTOR = 0.42             # a cell drawn but NOT scored

# ============================================================================================= #
# layout, 1920 x 1080                                                                            #
# ============================================================================================= #
W_TOT, H_TOT, PAD = 1920, 1080, 16
H_BAN = 40
X_CAM, Y_CAM, CAM_W, CAM_H = 16, 46, 1024, 416
PW = 404                                    # BEV panel: 60 m lateral
PH = 673                                    # BEV panel: 100 m ahead
SX, SY = PW / (2 * Y_HALF_M), PH / X_MAX_M  # px per metre (6.733 / 6.730)
X_P1 = 1060
X_P2 = X_P1 + PW + 14
Y_PT = 46                                   # panel title
Y_P = 70                                    # panel image top
Y_INFO = Y_CAM + CAM_H + 8                  # 470
Y_LEG = Y_P + PH + 18                       # 761 (legend under the panels)
Y_TS = 834                                  # time-series strip
H_TS = H_TOT - Y_TS - 6
#: short class names for the camera labels (the full name is in the legend)
SHORT_CLS = ("car", "truck", "bus", "veh", "trailer", "ped", "rider", "stroller", "animal", "protr")
FPS = 10


def _cdist(a, b) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def assert_palette_disjoint(min_dist: float = 60.0) -> dict:
    """⛔ No class / agent / fan colour may read as the GT green or the plan orange, and the plan orange
    must sit > 90 (L1) from the given-input amber (RENDER_REFAV1_ARMS_VIDEO.md §3)."""
    rep = {}
    named = ([(f"class_{CLASS_KEYS[i]}", c) for i, c in enumerate(PALETTE)]
             + [(f"agent_{n}", c) for n, c in zip(AGENT_CLASSES, AGENT_PALETTE)] + [("fan", C_FAN)])
    for name, c in named:
        d = min(_cdist(c, C_GT), _cdist(c, C_SEL))
        rep[name] = round(d, 1)
        if d < min_dist:
            raise SystemExit(f"[rv7] colour {name} {c} is {d:.1f} from the GT green / plan orange")
    l1 = sum(abs(a - b) for a, b in zip(C_SEL, C_GIVEN))
    if l1 <= 90:
        raise SystemExit(f"[rv7] plan orange is L1 {l1} from the given-input amber (needs > 90)")
    rep["orange_vs_amber_L1"] = l1
    return rep


# ============================================================================================= #
# PURE geometry (unit-tested with known values)                                                  #
# ============================================================================================= #
def m2px(x_m, y_m) -> tuple:
    """Ego-frame metres (x forward, y LEFT) -> panel-local pixels. The ego (0, 0) sits at the BOTTOM
    CENTRE; +x goes UP; +y (left) goes LEFT -- the camera's own left."""
    return ((Y_HALF_M - float(y_m)) * SX, (X_MAX_M - float(x_m)) * SY)


def grid_to_image(a: np.ndarray) -> np.ndarray:
    """GRID order ``[X=1000, Y=600, ...]`` (row 0 nearest, col 0 = RIGHT) -> IMAGE order (row 0 = the
    FAR edge at the top, col 0 = the LEFT edge). A pure flip of both axes; no resampling."""
    return np.asarray(a)[::-1, ::-1]


def cell_of(x_m: float, y_m: float) -> tuple:
    """Metric point -> (row, col) of the 10 cm grid, or None outside the extent."""
    r = int(math.floor(float(x_m) / CELL_M))
    c = int(math.floor((float(y_m) + Y_HALF_M) / CELL_M))
    if 0 <= r < GRID_X and 0 <= c < GRID_Y:
        return r, c
    return None


def class_rgb(codes: np.ndarray) -> np.ndarray:
    """uint8 codes [X, Y] -> RGB [X, Y, 3]; NOT_SEEN -> the hatch background."""
    lut = np.zeros((256, 3), np.uint8)
    lut[:] = HATCH_BG
    for i, c in enumerate(PALETTE):
        lut[i] = c
    return lut[np.asarray(codes, dtype=np.uint8)]


def downsample_rgb(img: np.ndarray, size: tuple) -> np.ndarray:
    """Area (box) downsample of an IMAGE-order array to (w, h). Thin classes survive as blended
    lines instead of vanishing under nearest-neighbour."""
    from PIL import Image
    return np.asarray(Image.fromarray(np.ascontiguousarray(img)).resize(size, Image.BOX))


def downsample_mask(mask: np.ndarray, size: tuple) -> np.ndarray:
    from PIL import Image
    m = Image.fromarray((np.ascontiguousarray(mask).astype(np.uint8) * 255))
    return np.asarray(m.resize(size, Image.BOX)) >= 128


_HATCH = {}


def hatch(kind: str) -> np.ndarray:
    """'A' diagonal lines = SAM3 never saw the cell; 'B' dots = the lift's camera cannot reach it now."""
    if kind not in _HATCH:
        yy, xx = np.mgrid[0:PH, 0:PW]
        a = np.empty((PH, PW, 3), np.uint8)
        a[:] = HATCH_BG
        if kind == "A":
            a[((xx + yy) % 9) < 2] = HATCH_A
        else:
            a[((xx % 6) < 2) & ((yy % 6) < 2)] = HATCH_B
        _HATCH[kind] = a
    return _HATCH[kind]


def map_panel(codes: np.ndarray, gt: np.ndarray, lv: np.ndarray, *, kind: str) -> np.ndarray:
    """ONE function for BOTH map panels -> ``[PH, PW, 3]`` uint8.

    Full brightness = a SCORED cell (SAM3 saw it AND the lift reaches it now) -- the SAME set on both
    panels. ``kind='pred'``: the model's code everywhere the lift reaches; DIMMED where SAM3 has no label;
    hatch B where the lift cannot reach. ``kind='gt'``: the SAM3 code everywhere SAM3 saw; DIMMED where
    the lift cannot reach; hatch A where SAM3 never saw."""
    codes = np.asarray(codes)
    seen = np.asarray(gt) != NOT_SEEN
    lv = np.asarray(lv, dtype=bool)
    rgb = class_rgb(codes).astype(np.float32)
    if kind == "pred":
        dim = lv & ~seen
        hole = ~lv
        hk = "B"
    elif kind == "gt":
        dim = seen & ~lv
        hole = ~seen
        hk = "A"
    else:
        raise ValueError(kind)
    rgb[dim] *= DIM_FACTOR
    img = downsample_rgb(grid_to_image(rgb.astype(np.uint8)), (PW, PH)).copy()
    hm = downsample_mask(grid_to_image(hole), (PW, PH))
    img[hm] = hatch(hk)[hm]
    return img


def box_corners_xy(cx, cy, l, w, yaw) -> np.ndarray:
    """``[4, 2]`` footprint corners (+x fwd, +y LEFT): FL, FR, RR, RL. ``l`` runs along ``yaw``."""
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    hl, hw = 0.5 * float(l), 0.5 * float(w)
    loc = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]], dtype=np.float64)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return loc @ rot.T + np.array([float(cx), float(cy)], dtype=np.float64)


def plan_errors(sel, gt_slots, slot_valid) -> dict:
    """ADE over the VALID emitted slots and FDE at the 6 s slot (None when that slot is not valid).
    ``sel`` / ``gt_slots`` ``[8, 2]`` metres; L2 per slot."""
    sel = np.asarray(sel, np.float64)
    gt = np.asarray(gt_slots, np.float64)
    v = np.asarray(slot_valid, bool)
    d = np.linalg.norm(sel - gt, axis=1)
    return {"ade_m": float(d[v].mean()) if v.any() else None,
            "fde_m": float(d[-1]) if bool(v[-1]) else None,
            "n_valid_slots": int(v.sum()), "slot_err_m": [float(x) for x in d]}


def select_clips(cands: list, per_token_halves=("low", "high"),
                 criteria=("vru", "lead")) -> tuple:
    """THE FIXED CLIP RULE (RENDER_REFCV7_VIDEO.md §2) -- pure, GT and labels only.

    ``cands``: dicts with ``sha12``, ``nav``, ``full_cov`` (bool), ``v_mean`` (m/s), ``n_vru_win``,
    ``n_lead_win``. For each nav token in ``NAV_ORDER``: the token's fully covered clips are split at
    THEIR median ``v_mean`` into LOW (< median) and HIGH (>= median); in each half, pick (1) the clip with
    the most VRU windows, then (2) of the rest the clip with the most lead-vehicle windows. Ties and an
    all-zero criterion fall back to the smallest sha12 (lowercase hex). 4 clips per token, 12 in all.
    Returns ``(chosen, report)``; every chosen entry carries ``why``."""
    pool = [c for c in cands if c["full_cov"] and c["nav"] in NAV_ORDER]
    chosen, report = [], {"by_token": {}}
    key_of = {"vru": "n_vru_win", "lead": "n_lead_win"}
    for nav in NAV_ORDER:
        P = sorted((c for c in pool if c["nav"] == nav), key=lambda c: c["sha12"])
        if len(P) < 2 * len(criteria):
            raise SystemExit(f"[rv7] nav {nav!r}: only {len(P)} fully covered clips")
        med = float(np.median([c["v_mean"] for c in P]))
        report["by_token"][nav] = {"n_pool": len(P), "v_mean_median_ms": round(med, 4)}
        for half in per_token_halves:
            H = [c for c in P if (c["v_mean"] < med) == (half == "low")]
            taken = set()
            for crit in criteria:
                k = key_of[crit]
                rest = [c for c in H if c["sha12"] not in taken]
                if not rest:
                    raise SystemExit(f"[rv7] nav {nav} {half}: nothing left for {crit}")
                best = max(c[k] for c in rest)
                if best > 0:
                    pick = min((c for c in rest if c[k] == best), key=lambda c: c["sha12"])
                    why = (f"nav {nav.upper()}, {half} speed (v_mean {'<' if half == 'low' else '>='} "
                           f"token median {med:.1f} m/s): most {crit.upper()} windows ({best})")
                else:
                    pick = min(rest, key=lambda c: c["sha12"])
                    why = (f"nav {nav.upper()}, {half} speed: no clip in this half has a {crit.upper()} "
                           f"window -> smallest sha12")
                taken.add(pick["sha12"])
                chosen.append({**pick, "slot": f"{nav}/{half}/{crit}", "why": why})
    if not any(c["n_vru_win"] > 0 for c in chosen):
        raise SystemExit("[rv7] the rule chose no clip with a VRU window -- the brief requires one")
    if not any(c["n_lead_win"] > 0 for c in chosen):
        raise SystemExit("[rv7] the rule chose no clip with a lead-vehicle window")
    return chosen, report


# ============================================================================================= #
# small utilities                                                                                #
# ============================================================================================= #
_UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")


def sha12(s: str) -> str:
    return hashlib.sha256(str(s).encode("utf-8")).hexdigest()[:12]


def scrub(obj):
    if isinstance(obj, str):
        return _UUID.sub(lambda m: "sha12:" + sha12(m.group(0)), obj)
    if isinstance(obj, dict):
        return {scrub(k): scrub(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [scrub(v) for v in obj]
    return obj


def _jsonable(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def dumps(obj, **kw) -> str:
    txt = json.dumps(scrub(obj), ensure_ascii=True, default=_jsonable, **kw)
    if _UUID.search(txt):
        raise SystemExit("[rv7] refusing to write: a raw clip UUID survived the scrub")
    return txt


def write_json(path, obj) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(dumps(obj, indent=1), encoding="utf-8")
    return str(path)


def md5_file(p, chunk=1 << 22) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(chunk), b""):
            h.update(c)
    return h.hexdigest()


def _p(*a):
    print(*a, flush=True)


def _by_path(name, p):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, str(p))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _r(x, n=4):
    return None if x is None else round(float(x), n)


# ============================================================================================= #
# STAGE 1 -- the forward (Thor GPU)                                                              #
# ============================================================================================= #
def stage_forward(a) -> int:  # noqa: C901 -- one linear pipeline, sectioned
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
    rec: dict = {"tool": "taniteval/tools/render_refcv7_video.py", "stage": "forward", "tag": tag,
                 "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "argv": sys.argv[1:], "tool_md5": md5_file(__file__), "departures": []}
    rec["palette"] = assert_palette_disjoint()
    # ---- the code tree, ASSERTED before any model code ------------------------------------------ #
    L = _by_path("refcv7_loader", tree / "stack/tanitad/eval/refcv7_loader.py")
    L.bootstrap()
    import tanitad
    if not os.path.abspath(tanitad.__file__).startswith(str(tree / "stack")):
        raise SystemExit(f"[rv7] tanitad imported from {tanitad.__file__}, not {tree}/stack")
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "6")))
    tr = L.trainer()
    from tanitad.data import perception_targets as ptg
    from tanitad.data import semantic_map_gt_fine as smf
    from tanitad.data import v7_labels as v7l
    from tanitad.eval import detection_metrics as det
    from tanitad.models import agent_slots as _as
    from tanitad.models import map_head_hires as mhr
    from tanitad.refs import refb
    from tanitad.refs import refcv6_max_speed as v6ms
    from tanitad.refs import refcv6_selection as v6sel
    import refb_labels
    from taniteval import four_families as ff
    if tuple(_as.AGENT_CLASSES) != AGENT_CLASSES:
        raise SystemExit(f"[rv7] AGENT_CLASSES {_as.AGENT_CLASSES} != the literal")
    if (len(smf.FINE_CLASSES) != len(MAP_CLASSES) or tuple(mhr.CLASS_KEYS) != CLASS_KEYS
            or smf.FINE_CLASSES[1] != "drivable" or smf.FINE_CLASSES[7] != "sidewalk / verge"):
        raise SystemExit(f"[rv7] map classes {smf.FINE_CLASSES} / {mhr.CLASS_KEYS} drifted")
    ext = smf.EXTENT_REFCV7
    if (float(ext.x_max_m), float(ext.y_half_m), tuple(ext.fine_shape)) != (X_MAX_M, Y_HALF_M,
                                                                            (GRID_X, GRID_Y)):
        raise SystemExit(f"[rv7] EXTENT_REFCV7 {ext.as_dict()} != this renderer's grid")
    if int(smf.NOT_SEEN_CODE) != NOT_SEEN:
        raise SystemExit("[rv7] NOT_SEEN_CODE drifted")
    if not os.path.abspath(ff.__file__).startswith(str(tree)):
        raise SystemExit(f"[rv7] four_families imported from {ff.__file__}, not the launch tree")
    rcv3 = _by_path("render_refcv3_video_rv7", tree / "taniteval/tools/render_refcv3_video.py")
    rec["code"] = {
        "launch_tree": str(tree), "tanitad": tanitad.__file__,
        "refcv7_loader_md5": md5_file(tree / "stack/tanitad/eval/refcv7_loader.py"),
        "trainer_md5": md5_file(tree / "stack/scripts/refc_v3_train.py"),
        "render_refcv3_video_md5": md5_file(tree / "taniteval/tools/render_refcv3_video.py"),
        "four_families": ff.__file__}
    # ---- checkpoint + config, VERIFIED BY CONTENT ------------------------------------------------ #
    ckpt, cfgp = run / "ckpt.pt", run / "config.json"
    t0 = time.time()
    ck_md5 = md5_file(ckpt)
    rec["ckpt"] = {"path": str(ckpt), "md5": ck_md5, "bytes": ckpt.stat().st_size,
                   "md5_s": round(time.time() - t0, 1)}
    if ck_md5 != a.expect_ckpt_md5:
        raise SystemExit(f"[rv7] ckpt md5 {ck_md5} != expected {a.expect_ckpt_md5}")
    rec["config"] = {"path": str(cfgp), "md5": md5_file(cfgp)}
    summ = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    if not summ.get("done") or int(summ.get("final_step", -1)) != a.expect_step:
        raise SystemExit(f"[rv7] summary.json {summ} does not say done at {a.expect_step}")
    rec["run_summary"] = {k: summ.get(k) for k in ("done", "final_step", "written_utc", "gate_token")}
    config = L.load_config(str(cfgp))
    # the run's OWN in-run eval row at this step: the detection display threshold comes from HERE
    ev = None
    with open(run / "metrics.jsonl", encoding="utf-8") as fh:
        for ln in fh:
            if '"eval_' in ln and f'"step": {a.expect_step}' in ln:
                r_ = json.loads(ln)
                if r_.get("step") == a.expect_step and any(k.startswith("eval_") for k in r_):
                    ev = r_
    if ev is None:
        raise SystemExit(f"[rv7] no in-run eval row at step {a.expect_step} in metrics.jsonl")
    thr_box = float(ev["eval_box3d_calib_pr_gate"])
    thr_agent = float(ev["eval_agent_calib_pr_gate"])
    keep_ev = {k: v for k, v in ev.items() if k == "step" or (k.startswith("eval_") and (
        "calib" in k or k.endswith(("_prec@gate", "_rec@gate", "_conf_ratio", "_ap2m", "_n_pos",
                                    "_n_conf")) or k.startswith("eval_map_hires_iou_")))}
    rec["inrun_eval_row"] = {"path": str(run / "metrics.jsonl"), "row": keep_ev,
                             "note": "128 OTHER eval windows (8 batches x 16); calib keys = 256 TRAIN "
                                     "calibration windows (detection_metrics.CALIB_WINDOWS_FILE)"}
    rec["detection_display"] = {
        "box3d_threshold": thr_box, "agent_threshold": thr_agent,
        "source": "metrics.jsonl step 50400 eval_box3d_calib_pr_gate / eval_agent_calib_pr_gate: the P=R "
                  "operating point (detection_metrics.pr_equal_gate) on 256 TRAIN calibration windows -- "
                  "not tuned on the rendered clips",
        "declared_gate": None}
    # ---- the MODEL: the launch tree's loader, strict ---------------------------------------------- #
    t0 = time.time()
    model, cfg, targs, mrec = L.build_model(config, str(ckpt), device="cuda")
    sd = mrec["state_dict"]
    if int(sd.get("step") or -1) != a.expect_step:
        raise SystemExit(f"[rv7] STEP MISMATCH: ck['step'] = {sd.get('step')!r}")
    if sd["missing"] or sd["unexpected"]:
        raise SystemExit(f"[rv7] strict load not clean: {sd['missing'][:5]} {sd['unexpected'][:5]}")
    if not mrec["param_breakdown"]["equal"]:
        raise SystemExit("[rv7] param_breakdown differs from config.json")
    bad_anc = [k for k, v in mrec["anchor_file_vs_ckpt_buffers"].items() if v["max_abs_diff"] != 0.0]
    if bad_anc:
        raise SystemExit(f"[rv7] anchor file != checkpoint anchor buffers: {bad_anc}")
    br = getattr(model, "_map_hires", None)
    if br is None or str(br.cfg.decision_rule) != "prior_corrected":
        raise SystemExit("[rv7] no 10 cm branch, or its decision rule is not prior_corrected")
    if tuple(br.cfg.out_hw) != (GRID_X, GRID_Y):
        raise SystemExit(f"[rv7] map out_hw {br.cfg.out_hw} != ({GRID_X}, {GRID_Y})")
    cw = model._map_hires_class_weight
    from tanitad.models import slot_presence as _sp
    rec["detection_display"]["declared_gate"] = float(_sp.DETECTION_GATE)
    if tuple(int(h) for h in cfg.core.trajectory.horizons) != HORIZONS:
        raise SystemExit(f"[rv7] horizons {cfg.core.trajectory.horizons} != {HORIZONS}")
    if bool(getattr(getattr(cfg.core, "agents", None), "oracle", False)):
        raise SystemExit("[rv7] --agents oracle build: refusing")
    W = int(cfg.core.window)
    mode = getattr(targs, "mode", "diffusion")
    ablate = bool(getattr(targs, "ablate_frames", False))
    eq_trunk = int(cfg.core.encoder.trunk_equalize_bottom_rows)
    eq_lift = int(getattr(model._lift_bank_hires, "equalize_bottom_rows", 0) or 0)
    rec["model"] = {k: mrec[k] for k in ("argv_remap", "departures", "param_breakdown", "mode", "sampler",
                                        "decoder_steps", "build_s", "map_hires", "perception",
                                        "rig_camera", "trunk_equalize_as_trained", "stamp_checks",
                                        "declared_vs_built") if k in mrec}
    rec["model"]["state_dict"] = {k: v for k, v in sd.items() if k != "ckpt_keys"}
    rec["model"]["class_weights"] = [float(x) for x in cw.tolist()]
    rec["equalize_bottom_rows"] = {"trunk_as_trained": eq_trunk, "lift_bank_hires": eq_lift}
    rec["departures"] += list(mrec.get("departures", []))
    _p(f"[rv7] model built {mrec['build_s']} s: step {sd['step']} strict 0/0, window {W}, mode {mode}, "
       f"sampler {mrec.get('sampler')} x{mrec.get('decoder_steps')}, eq rows trunk {eq_trunk} lift {eq_lift}")
    # ---- the held-out eval split, exactly as train() builds it ------------------------------------ #
    t0 = time.time()
    e_ds, e_eps, drec = L.build_eval_dataset(model, cfg, targs, config, with_perception_targets=True)
    rec["dataset"] = {k: drec.get(k) for k in ("n_episodes", "n_windows", "map_fine", "agent_join",
                                              "join3d", "max_speed_v6", "nav", "labels", "label_clock")}
    rec["dataset"]["build_s"] = round(time.time() - t0, 1)
    store = e_ds.map_fine_store
    n_stack = int(e_ds.map_n_stack)
    clip_of_ep = e_ds.map_clip_of_ep
    nav_names = list(refb.NAV_COMMANDS)
    tok_of_legacy = {v: k for k, v in tr.NAV_TOKEN_TO_LEGACY.items()}
    by_ep: dict = {}
    for wi, (e_i, t) in enumerate(e_ds.index):
        by_ep.setdefault(e_i, []).append((int(t), wi))
    # ---- THE CLIP RULE: GT + labels only, BEFORE any forward --------------------------------------- #
    t0 = time.time()
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
            fov = (np.degrees(np.arctan2(np.abs(b[:, 1]), b[:, 0])) <= FOV_HALF_DEG) & (b[:, 0] >= 0)
            vru = v & np.isin(c, VRU_CLASSES) & (rng <= VRU_RANGE_M) & fov
            lead = (v & np.isin(c, VEHICLE_CLASSES) & (b[:, 0] > 0) & (b[:, 0] <= LEAD_MAX_X_M)
                    & (np.abs(b[:, 1]) <= LEAD_LANE_HALF_M))
            n_vru += int(vru.any())
            n_lead += int(lead.any())
        cands.append({"sha12": sha12(cid), "_cid": cid, "_e_i": e_i,
                      "nav": None if nav_idx is None else nav_names[int(nav_idx)],
                      "n_windows": len(wl), "n_map_ok": int(cov["n_ok"]),
                      "full_cov": int(cov["n_ok"]) == len(wl),
                      "v_mean": float(np.mean([sp[t + W - 1] for t, _ in wl])),
                      "v_min": float(min(sp[t + W - 1] for t, _ in wl)),
                      "v_max": float(max(sp[t + W - 1] for t, _ in wl)),
                      "n_agent_labelled_win": n_lab, "n_vru_win": n_vru, "n_lead_win": n_lead})
    chosen, sel_rep = select_clips(cands)
    sel_rec = {
        "rule": ("candidates = eval139 clips whose EVERY eval window has a 10 cm SAM3 GT frame "
                 "(perception_targets.require_map_coverage, n_ok == n_windows) and a v7 nav token; per nav "
                 "token (left, right, follow) the pool is split at its median clip-mean measured speed "
                 "(v0 at each window's NOW) into LOW (<) and HIGH (>=); in each half pick (1) the clip with "
                 "the most VRU windows (a GT person/rider/stroller within 30 m inside the 120 deg field at "
                 "NOW), then (2) of the rest the clip with the most lead-vehicle windows (a GT vehicle-class "
                 "box 0 < x <= 60 m, |y| <= 1.75 m at NOW); ties and an all-zero criterion -> smallest "
                 "sha12. 12 clips; every eval window of each, in time order. Uses GT/labels only, fixed "
                 "before any model output is read."),
        "n_eval_clips": len(e_eps), "n_full_coverage": sum(c["full_cov"] for c in cands),
        "n_by_nav_full": {n: sum(1 for c in cands if c["full_cov"] and c["nav"] == n) for n in NAV_ORDER},
        "not_full": [{"sha12": c["sha12"], "n_windows": c["n_windows"], "n_map_ok": c["n_map_ok"]}
                     for c in cands if not c["full_cov"]],
        "token_split": sel_rep["by_token"],
        "chosen": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()
                    if not k.startswith("_") and k != "full_cov"} for c in chosen],
        "selection_s": round(time.time() - t0, 1), "window": W}
    sel_path = work / "clip_selection.json"
    if sel_path.exists() and not a.smoke:
        prev = json.loads(sel_path.read_text(encoding="utf-8"))
        if [c["sha12"] for c in prev["chosen"]] != [c["sha12"] for c in chosen]:
            raise SystemExit(f"[rv7] the rule chose {[c['sha12'] for c in chosen]} but {sel_path} "
                             f"records {[c['sha12'] for c in prev['chosen']]}")
        sel_rec["agrees_with_previous_run"] = True
    write_json(sel_path if not a.smoke else bank / "clip_selection.json", sel_rec)
    write_json(bank / "clip_selection.json", sel_rec)
    _p(f"[rv7] selection ({sel_rec['selection_s']} s): " + ", ".join(
        f"{c['slot']}={c['sha12']}(v{c['v_mean']:.1f},vru{c['n_vru_win']},lead{c['n_lead_win']})"
        for c in chosen))
    run_list = chosen[:1] if a.smoke else chosen
    # ---- the forward: the trainer's compute_losses_v3 in eval mode, out captured by a hook --------- #
    cap = {}

    def _hook(_m, _i, out):
        cap["out"] = out
    hook = model.register_forward_hook(_hook)

    def forward(batch, seed):
        cap.clear()
        torch.manual_seed(int(seed))
        with torch.no_grad():
            extra = tr.compute_losses_v3(model, batch, "cuda", mode=mode, ablate_frames=ablate)
        if "out" not in cap:
            raise SystemExit("[rv7] the forward hook never fired")
        return cap["out"], extra
    torch.cuda.reset_peak_memory_stats()
    lat_names = list(v7l.HEADS["tac_lat"])
    lon_names = list(v7l.HEADS["tac_lon"])
    goal_names = list(v7l.TAC_GOAL_TOKENS)
    extr_tab_path = getattr(targs, "agent_rig_extrinsics")
    extr_tab = rcv3.load_extrinsics(extr_tab_path, [c["_cid"] for c in run_list])
    rec["extrinsics"] = {"path": extr_tab_path, "md5": md5_file(extr_tab_path)}
    rows_f = open(bank / "rows.jsonl", "w", encoding="utf-8")
    clips_meta = []
    gpu_s = 0.0
    n_done = 0
    ident = {"map_inter_union_max_abs_diff": 0.0, "det_logit_max_abs_diff": 0.0,
             "det_xy_max_abs_diff": 0.0, "det_gt_xy_max_abs_diff": 0.0, "traj_vs_fan_max_abs_diff": 0.0,
             "n_det_packs": 0, "n_windows_no_agent_label": 0}
    for ci, c in enumerate(run_list):
        cid, s12, e_i = c["_cid"], c["sha12"], c["_e_i"]
        ep = e_eps[e_i]
        wl = sorted(by_ep[e_i])
        if a.smoke:
            wl = wl[:a.smoke_windows]
        elif a.max_windows:
            wl = wl[:a.max_windows]
        # ---- this clip's cameras: the payload frame, the CylProjector, the model's RigCamera ------ #
        pay_path = Path(ep.frames._cache.cache_dir) / ep.frames._cache.files[ep.frames._clip]
        pay = torch.load(str(pay_path), map_location="cpu", weights_only=False, mmap=True)
        fr = dict(pay.get("frame") or {})
        codec = pay.get("codec")
        del pay
        fr = {"height": int(fr["height"]), "width": int(fr["width"]), "f_ref": float(fr["f_ref"]),
              "projection": str(fr["projection"])}
        if (fr["height"], fr["width"]) != (CAM_H, CAM_W):
            raise SystemExit(f"[rv7] payload frame {fr} is not {CAM_W}x{CAM_H}")
        extr = extr_tab.get(cid)
        if extr is None:
            raise SystemExit(f"[rv7] no per-clip extrinsics for {s12}")
        extr_n = {k: float(extr[k]) for k in ("x", "y", "z", "qx", "qy", "qz", "qw")}
        proj = rcv3.CylProjector(fr, extr_n)
        rig = model._rig_camera.get(int(ep.episode_id))
        if rig is None:
            raise SystemExit(f"[rv7] no RigCamera for {s12}")
        rf = rig.frame
        if (int(rf.height), int(rf.width), str(rf.projection)) != (fr["height"], fr["width"],
                                                                   fr["projection"]) \
                or abs(float(rf.f_ref) - fr["f_ref"]) > 1e-6:
            raise SystemExit(f"[rv7] RigCamera frame {rf} != payload frame {fr}")
        gp = np.array([[x, y] for x in (4.0, 8.0, 15.0, 30.0, 55.0) for y in (-6.0, 0.0, 6.0)])
        cpx = proj(gp, up=1)
        _c, _rw, _ok = rig.project(torch.as_tensor(np.c_[gp, np.zeros(len(gp))], dtype=torch.float64))
        dd = [math.hypot(q[0] - float(_c[i]), q[1] - float(_rw[i])) for i, q in enumerate(cpx)
              if q is not None and bool(_ok[i])]
        if not dd or max(dd) > 0.01:
            raise SystemExit(f"[rv7] CylProjector vs RigCamera disagree on {s12}: {dd}")
        lr = proj(np.array([[20.0, 5.0], [20.0, -5.0]]), up=1)
        if not (lr[0] and lr[1] and lr[0][0] < (CAM_W - 1) / 2 < lr[1][0]):
            raise SystemExit(f"[rv7] LEFT/RIGHT control failed on {s12}: y=+5 -> {lr[0]}, y=-5 -> {lr[1]}")
        probe_xyz = np.array([[12.0, 3.0, 0.8], [25.0, -4.0, 1.6], [40.0, 0.5, 0.0], [8.0, -2.5, 2.2]])
        pc, pr_, pok = rig.project(torch.as_tensor(probe_xyz, dtype=torch.float64))
        meta = {"clip_rank": ci + 1, "sha12": s12, "nav": c["nav"], "nav_token": tok_of_legacy.get(c["nav"]),
                "slot": c["slot"], "why": c["why"], "selection": {k: c[k] for k in (
                    "n_windows", "v_mean", "v_min", "v_max", "n_agent_labelled_win", "n_vru_win",
                    "n_lead_win")},
                "frame": fr, "codec": codec, "extrinsics": extr_n,
                "rig_camera": {"R_cam_to_rig": rig.R_cam_to_rig.double().tolist(),
                               "t_cam_in_rig": rig.t_cam_in_rig.double().tolist(),
                               "probe_xyz": probe_xyz.tolist(),
                               "probe_col_row_ok": [[float(pc[i]), float(pr_[i]), bool(pok[i])]
                                                    for i in range(len(probe_xyz))]},
                "projector_cross_check_px_max": float(max(dd)), "left_right_control": [lr[0], lr[1]],
                "camera_label": proj.label(), "n_windows_rendered": len(wl),
                "draw_independence": None}
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
                raise SystemExit(f"[rv7] {s12} window {k_i}: no 10 cm label on a fully covered clip")
            pred = mhr.decide(lg, str(br.cfg.decision_rule), cw)[0].to(torch.uint8)
            lvf = mhr.lift_valid_to_fine(lv025, (GRID_X, GRID_Y))[0]
            # ⭐ IDENTITY: the per-class counts recomputed through the module's own function must equal
            # the trainer's eval extras for this window (same cells, same rule)
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
                            raise SystemExit(f"[rv7] trainer extras carry no {k_}")
                        dmax = max(dmax, abs(float(extra[k_]) - float(arr[ci_, bi])))
            if dmax > 1e-6:
                raise SystemExit(f"[rv7] map per-class counts differ from the trainer's extras by {dmax}")
            ident["map_inter_union_max_abs_diff"] = max(ident["map_inter_union_max_abs_diff"], dmax)
            n_sup = int(sig["n_supervised"])
            if abs(float(extra["n_map_hires_cells"]) - n_sup) > 0.5:
                raise SystemExit("[rv7] supervised cell count differs from the trainer's")
            # ---- the plan, the fan, the ceiling ------------------------------------------------- #
            traj = out["traj"][0].detach().float().cpu().numpy()
            sel_idx = int(out["sel_idx"][0])
            fan = out["anchor_traj"][0].detach().float().cpu().numpy()
            dtf = float(np.abs(traj - fan[sel_idx]).max())
            ident["traj_vs_fan_max_abs_diff"] = max(ident["traj_vs_fan_max_abs_diff"], dtf)
            score = out["sel_score_v3"][0].detach().float().cpu().numpy()
            reach = out["reach_keep"][0].detach().cpu().numpy().astype(bool)
            order = [int(i) for i in np.argsort(-score) if int(i) != sel_idx][:TOPK_FAN]
            v_max_raw = float(batch["v_max_ms"][0])
            v_max_ok = float(batch["v_max_valid"][0])
            oh, over = model.max_speed_1h_v6(batch["v_max_ms"].to(lg.device).reshape(-1),
                                             batch["v_max_valid"].to(lg.device))
            if float(oh.sum()) > 0.5:
                ceil_bin = int(oh.argmax(dim=-1)[0])
                v_lim = float(v6ms.limit_ms_of_bin(oh.argmax(dim=-1))[0])
            else:
                ceil_bin, v_lim = None, float("inf")
            plan_vmax = float(v6sel.planned_max_speed(out["traj"].detach().float()[:, None],
                                                      horizons=HORIZONS, tick_s=0.1)[0, 0])
            fan_t = out["anchor_traj"].detach().float()
            fan_vmax = v6sel.planned_max_speed(fan_t, horizons=HORIZONS, tick_s=0.1)[0].cpu().numpy()
            ceil_keep = fan_vmax <= v_lim
            # ---- GT paths: the trainer's own label function ----------------------------------- #
            pl = item["pose_last"].float()
            fut = item["future_poses_ext"].float()
            fv = item["future_valid_ext"].bool().numpy()
            gt_slots = refb_labels.waypoint_targets(pl[None], fut[None], HORIZONS)[0].numpy()
            slot_valid = [bool(fv[h - 1]) for h in HORIZONS]
            gt_dense = refb_labels.waypoint_targets(pl[None], fut[None], tuple(range(1, 61)))[0].numpy()
            pe = plan_errors(traj, gt_slots, slot_valid)
            # longitudinal / lateral on the 0.5 s grid (first 4 slots), four_families' own geometry
            gp_ = ff._seq_geometry(torch.as_tensor(traj[None, :4]), dt=0.5)
            gg_ = ff._seq_geometry(torch.as_tensor(gt_slots[None, :4]), dt=0.5)
            v4 = np.asarray(slot_valid[:4], bool)
            spd_err = (np.abs(gp_["speed"][0].numpy() - gg_["speed"][0].numpy())[v4].mean()
                       if v4.any() else None)
            hv = gp_["valid"][0].numpy() & gg_["valid"][0].numpy() & v4
            dh = np.abs((gp_["heading"][0].numpy() - gg_["heading"][0].numpy() + math.pi)
                        % (2 * math.pi) - math.pi)
            head_err = float(np.degrees(dh[hv]).mean()) if hv.any() else None
            pv = gp_["pair_valid"][0].numpy() & gg_["pair_valid"][0].numpy() & v4[1:] & v4[:-1]
            dk = np.abs(gp_["curvature"][0].numpy() - gg_["curvature"][0].numpy())
            curv_err = float(dk[pv].mean()) if pv.any() else None
            # ---- tactical (the refcv6 tactical decoder the battery scores) -------------------- #
            plat = out["tacv6_lat_logits"][0].float().softmax(-1).cpu().numpy()
            plon = out["tacv6_lon_logits"][0].float().softmax(-1).cpu().numpy()
            pgoal = out["tacv6_goal_logits"][0].float().sigmoid().cpu().numpy()
            gl = int(item["lat_v7"])
            gn = int(item["lon_v7"])
            gy = item["tac_goal_y"].numpy()
            gw = item["tac_goal_w"].numpy()
            # ---- detections: the trainer's own eval packs (VIS-1 positives, IGNORE = DontCare) -- #
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
                    raise SystemExit("[rv7] agent-labelled window but no box3d detection pack")
                pk = extra["_det_pack_box3d"][0]
                ident["n_det_packs"] += 1
                ident["det_logit_max_abs_diff"] = max(ident["det_logit_max_abs_diff"], float(np.abs(
                    pk["logit"] - bs["presence_logit"][0].float().cpu().numpy()).max()))
                ident["det_xy_max_abs_diff"] = max(ident["det_xy_max_abs_diff"],
                                                   float(np.abs(pk["xy"] - bx[:, :2]).max()))
                if pk["gt_xy"].shape[0] != gbox.shape[0]:
                    raise SystemExit("[rv7] pack GT rows != the batch's valid GT rows")
                if gbox.shape[0]:
                    ident["det_gt_xy_max_abs_diff"] = max(ident["det_gt_xy_max_abs_diff"], float(
                        np.abs(pk["gt_xy"] - gbox[:, :2]).max()))
                pos, ign = pk["pos"].astype(bool), pk["ign"].astype(bool)
                hid = pk["hidden"].astype(bool)
                for tag_, thr_ in (("thr", thr_box), ("gate", float(_sp.DETECTION_GATE))):
                    rows_ = det.greedy_rows(pk, det.PR_DIST_M, None, min_conf=thr_)
                    detrow[tag_] = {"thr": thr_, "n_det": sum(1 for r_ in rows_ if r_[1] >= 0),
                                    "tp": sum(1 for r_ in rows_ if r_[1] == 1),
                                    "fp": sum(1 for r_ in rows_ if r_[1] == 0),
                                    "dontcare": sum(1 for r_ in rows_ if r_[1] == -1),
                                    "pairs": [[int(r_[3]), int(r_[4]), int(r_[1])] for r_ in rows_]}
                detrow["n_pos"] = int(pos.sum())
                detrow["n_ign"] = int(ign.sum())
                pka = extra["_det_pack_agent"][0]
                rows_a = det.greedy_rows(pka, det.PR_DIST_M, None, min_conf=thr_agent)
                detrow["agent_head"] = {"thr": thr_agent, "n_det": sum(1 for r_ in rows_a if r_[1] >= 0),
                                        "tp": sum(1 for r_ in rows_a if r_[1] == 1),
                                        "p_max": float(1 / (1 + np.exp(-pka["logit"].astype(np.float64))).max())}
            else:
                ident["n_windows_no_agent_label"] += 1
                pos = ign = hid = np.zeros(gbox.shape[0], bool)
            # ---- the draw-independence control (first window of each clip) --------------------- #
            if k_i == 0:
                out2, _ = forward(batch, a.seed + 1)
                meta["draw_independence"] = {
                    "seeds": [a.seed, a.seed + 1],
                    "traj_max_abs_diff_m": float((out2["traj"][0].float().cpu() - out["traj"][0].float().cpu())
                                                 .abs().max()),
                    "sel_idx": [sel_idx, int(out2["sel_idx"][0])],
                    "map_logits_max_abs_diff": float((out2["perception"]["map_hires_logits"].float()
                                                      - lg.float()).abs().max()),
                    "box_presence_max_abs_diff": float((out2["perception"]["box_slots"]["presence_logit"]
                                                        .float() - bs["presence_logit"].float()).abs().max())}
                del out2
                _p(f"  [draw-independence] {json.dumps(meta['draw_independence'])}")
            # ---- bank ------------------------------------------------------------------------- #
            cam = item["frames"][-1, -3:].permute(1, 2, 0).contiguous().numpy()
            if cam.shape != (CAM_H, CAM_W, 3) or cam.dtype != np.uint8:
                raise SystemExit(f"[rv7] camera bytes {cam.shape} {cam.dtype}")
            np.savez_compressed(bank / "win" / f"c{ci:02d}_w{k_i:04d}.npz", cam=cam,
                                pred=pred.cpu().numpy(), gt=gt_codes[0].cpu().numpy(),
                                lv=lvf.cpu().numpy())
            t_lab = float(e_ds._now_s(ep, t))
            row = {
                "clip_rank": ci + 1, "clip_sha12": s12, "win": k_i, "n_win": len(wl), "t_start_row": int(t),
                "now_row": int(t + W - 1), "t_label_s": round(t_lab, 4),
                "v0_ms": round(float(pl[3]), 4), "nav": c["nav"],
                "nav_cmd": nav_names[int(item["nav_cmd"])], "nav_valid": bool(item["nav_valid"]),
                "v_max_raw_ms": round(v_max_raw, 4), "v_max_valid": v_max_ok, "ceil_bin": ceil_bin,
                "ceil_kmh": None if ceil_bin is None else int(v6ms.SPEED_MAX_STEPS_KMH_V6[ceil_bin]),
                "v_lim_ms": None if not math.isfinite(v_lim) else round(v_lim, 4),
                "plan_vmax_ms": round(plan_vmax, 4),
                "plan_exceeds_ceiling": bool(plan_vmax > v_lim),
                "n_fan_over_ceiling": int((~ceil_keep).sum()), "sel_in_ceil_keep": bool(ceil_keep[sel_idx]),
                "core_sel_idx": int(out["sel_idx_base"][0]) if "sel_idx_base" in out else None,
                "core_sel_in_ceil_keep": (bool(ceil_keep[int(out["sel_idx_base"][0])])
                                          if "sel_idx_base" in out else None),
                "sel_idx": sel_idx, "traj": np.round(traj, 3).tolist(),
                "fan_idx": order, "fan": np.round(fan[order], 3).tolist(),
                "fan_score": [round(float(score[i]), 4) for i in order],
                "sel_score": round(float(score[sel_idx]), 4), "n_reach_keep": int(reach.sum()),
                "gt_slots": np.round(gt_slots, 3).tolist(), "slot_valid": slot_valid,
                "gt_dense": np.round(gt_dense, 3).tolist(), "gt_dense_valid": [bool(x) for x in fv],
                **{k: _r(v) for k, v in pe.items() if k != "slot_err_m"},
                "slot_err_m": [round(x, 3) for x in pe["slot_err_m"]],
                "speed_mae_0_2s": _r(spd_err), "heading_mae_0_2s_deg": _r(head_err),
                "curv_mae_0_2s": _r(curv_err, 5),
                "tac_lat_pred": int(plat.argmax()), "tac_lat_p": _r(plat.max()),
                "tac_lon_pred": int(plon.argmax()), "tac_lon_p": _r(plon.max()),
                "tac_lat_gt": None if gl == v7l.IGNORE_ID else gl,
                "tac_lon_gt": None if gn == v7l.IGNORE_ID else gn,
                "goal_top": [[int(i), round(float(pgoal[i]), 3)] for i in np.argsort(-pgoal)[:3]],
                "goal_gt_pos": [int(i) for i in np.nonzero((gy > 0.5) & (gw > 0))[0]],
                "goal_gt_scored": int((gw > 0).sum()),
                "map_inter": inter.tolist(), "map_union": union.tolist(), "map_band_keys": bks,
                "n_map_scored": n_sup, "n_map_seen": int((gt_codes[0] != NOT_SEEN).sum()),
                "n_lift_valid": int(lvf.sum()),
                "pred": {"idx": keep_s.tolist(), "p": np.round(p_all[keep_s], 4).tolist(),
                         "box": np.round(bx[keep_s], 3).tolist(), "yaw": np.round(byaw[keep_s], 4).tolist(),
                         "cz": np.round(bcz[keep_s], 3).tolist(), "h": np.round(bh[keep_s], 3).tolist(),
                         "cls": bcls[keep_s].tolist()},
                "p_max": round(float(p_all.max()), 4), "n_slots": int(p_all.shape[0]),
                "gt": {"box": np.round(gbox, 3).tolist(), "yaw": np.round(gyaw, 4).tolist(),
                       "cls": gcls.tolist(), "cz": np.round(gcz, 3).tolist(), "h": np.round(gh, 3).tolist(),
                       "zh": gzh.tolist(), "pos": pos.tolist(), "ign": ign.tolist(), "hidden": hid.tolist()},
                "det": detrow, "seed": a.seed, "wall_s": None}
            row["wall_s"] = round(time.time() - t_w, 3)
            rows_f.write(dumps(row) + "\n")
            n_done += 1
            if k_i % 25 == 0 or k_i == len(wl) - 1:
                _p(f"  w{k_i + 1:03d}/{len(wl)} t={t_lab:.1f}s ADE {row['ade_m']} FDE {row['fde_m']} "
                   f"vmax {plan_vmax:.1f}/{v_lim:.1f} det {detrow.get('thr', {}).get('tp')}/"
                   f"{detrow.get('n_pos')} | {row['wall_s']} s")
            del out, extra, lg
        clips_meta.append(meta)
        rows_f.flush()
    rows_f.close()
    hook.remove()
    for k_, v_ in (("map_inter_union_max_abs_diff", 1e-6), ("det_logit_max_abs_diff", 0.0),
                   ("det_xy_max_abs_diff", 0.0), ("traj_vs_fan_max_abs_diff", 0.0)):
        if ident[k_] > v_:
            raise SystemExit(f"[rv7] identity {k_} = {ident[k_]} > {v_}")
    rec["identity_checks"] = ident
    rec["dataset"]["label_names"] = {"tac_lat": lat_names, "tac_lon": lon_names, "goal": goal_names}
    rec["clips"] = clips_meta
    rec["n_windows"] = n_done
    rec["compute"] = {"gpu_forward_s": round(gpu_s, 1), "wall_s": round(time.time() - t_all, 1),
                      "cuda_max_memory_allocated_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 3),
                      "device": torch.cuda.get_device_name(0)}
    rec["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json(bank / "forward_record.json", rec)
    (bank / "FORWARD_DONE").write_text(dumps({"n_windows": n_done, "wall_s": rec["compute"]["wall_s"]}),
                                       encoding="utf-8")
    _p(f"[rv7] FORWARD DONE: {n_done} windows, GPU {gpu_s:.0f} s, wall {time.time() - t_all:.0f} s")
    return 0


# ============================================================================================= #
# STAGE 2 -- drawing (Thor CPU)                                                                  #
# ============================================================================================= #
_G: dict = {}            # per-worker globals (fork-inherited)


def _fonts():
    if "F" not in _G:
        rcv3 = _G["rcv3"]
        _G["F"] = {"ban": rcv3.font(21, True), "hud": rcv3.font(16), "hudb": rcv3.font(16, True),
                   "sub": rcv3.font(14), "subb": rcv3.font(14, True), "tiny": rcv3.font(12),
                   "micro": rcv3.font(11), "big": rcv3.font(30, True), "med": rcv3.font(19, True),
                   "title": rcv3.font(40, True), "h2": rcv3.font(26, True)}
    return _G["F"]


def _fit(d, text, f, w):
    return _G["rcv3"].fit(d, text, f, w)


def _dashed(d, pts, fill, width=1, dash=5.0, gap=4.0, closed=True):
    pts = list(pts) + ([pts[0]] if closed else [])
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        seg = math.hypot(x1 - x0, y1 - y0)
        if seg <= 0:
            continue
        ux, uy = (x1 - x0) / seg, (y1 - y0) / seg
        s = 0.0
        while s < seg:
            e = min(s + dash, seg)
            d.line([(x0 + ux * s, y0 + uy * s), (x0 + ux * e, y0 + uy * e)], fill=fill, width=width)
            s = e + gap


def _bev_overlays(img: np.ndarray, row: dict, thr: float, F, boxes: bool = True):
    """Range rings, lateral guides, the fan, GT path (1 s dots), the plan (slot circles, 1 s filled),
    GT boxes and the 3-D head's boxes >= thr with their scores. Everything metric through ``m2px``."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im, "RGBA")
    ex, ey = m2px(0.0, 0.0)
    for r in range(10, 101, 10):
        rx, ry = r * SX, r * SY
        d.ellipse([ex - rx, ey - ry, ex + rx, ey + ry], outline=(255, 255, 255, 70 if r % 50 else 120),
                  width=1)
    for yl in (-20.0, -10.0, 10.0, 20.0):
        px, _ = m2px(0.0, yl)
        d.line([(px, 0), (px, PH)], fill=(255, 255, 255, 22), width=1)
    px0, _ = m2px(0.0, 0.0)
    d.line([(px0, 0), (px0, PH)], fill=(255, 255, 255, 40), width=1)
    # the fan (faded, rank-graded), under everything
    n_f = len(row["fan"])
    for j in range(n_f - 1, -1, -1):
        al = int(60 + 100 * (1.0 - j / max(n_f, 1)))
        pts = [m2px(0, 0)] + [m2px(x, y) for x, y in row["fan"][j]]
        d.line(pts, fill=C_FAN + (al,), width=1)
    if boxes:
        g = row["gt"]
        for i, b in enumerate(g["box"]):
            if not (0.0 <= b[0] <= X_MAX_M and abs(b[1]) <= Y_HALF_M):
                continue
            pts = [m2px(x, y) for x, y in box_corners_xy(b[0], b[1], b[2], b[3], g["yaw"][i])]
            if g["pos"][i]:
                d.polygon(pts, outline=C_GT + (255,), width=2)
            elif g["ign"][i]:
                _dashed(d, pts, C_GT + (220,), width=2, dash=3, gap=2)
            else:
                _dashed(d, pts, C_GT + (110,), width=1, dash=2, gap=3)
        pr = row["pred"]
        for j, p in enumerate(pr["p"]):
            if p < thr:
                continue
            b = pr["box"][j]
            col = AGENT_PALETTE[int(pr["cls"][j]) % len(AGENT_PALETTE)]
            pts = [m2px(x, y) for x, y in box_corners_xy(b[0], b[1], b[2], b[3], pr["yaw"][j])]
            d.polygon(pts, fill=col + (70,), outline=col + (255,), width=2)
            tx, ty = m2px(b[0], b[1])
            d.text((tx + 6, ty - 7), f"{p:.2f}", fill=col + (255,), font=F["micro"])
        # TP links (the programme's greedy 2 m rule)
        slot_pos = {int(s): j for j, s in enumerate(pr["idx"])}
        for s, gi, kind in (row["det"].get("thr") or {}).get("pairs", []):
            if kind == 1 and s in slot_pos:
                b = pr["box"][slot_pos[s]]
                gb = g["box"][gi]
                d.line([m2px(b[0], b[1]), m2px(gb[0], gb[1])], fill=(255, 255, 255, 220), width=1)
    # GT path + 1 s dots
    gv = np.asarray(row["gt_dense_valid"], bool)
    gd = np.asarray(row["gt_dense"])[gv]
    if len(gd):
        d.line([m2px(0, 0)] + [m2px(x, y) for x, y in gd], fill=C_GT + (255,), width=3, joint="curve")
        for k in range(9, len(gv), 10):
            if gv[k]:
                x, y = row["gt_dense"][k]
                px, py = m2px(x, y)
                d.ellipse([px - 3, py - 3, px + 3, py + 3], fill=C_GT + (255,))
    # the emitted plan: 8 slots; the six 1 s slots filled, 0.5 / 1.5 s hollow
    tr_ = row["traj"]
    d.line([m2px(0, 0)] + [m2px(x, y) for x, y in tr_], fill=C_SEL + (255,), width=3, joint="curve")
    for h, (x, y) in zip(HORIZONS, tr_):
        px, py = m2px(x, y)
        if h % 10 == 0:
            d.ellipse([px - 4, py - 4, px + 4, py + 4], fill=C_SEL + (255,), outline=(20, 20, 20, 255))
        else:
            d.ellipse([px - 3, py - 3, px + 3, py + 3], outline=C_SEL + (255,), width=2)
    px, py = m2px(*tr_[-1])
    d.text((px + 7, py - 8), "6 s", fill=C_SEL + (255,), font=F["micro"])
    d.polygon([(ex - 6, ey), (ex + 6, ey), (ex, ey - 12)], fill=(240, 244, 250, 255))
    for r in (20, 40, 60, 80, 100):
        _, py = m2px(r, 0.0)
        py = max(py, 8.0)                     # the 100 m label sits just inside the top edge
        lab = f"{r} m"
        tw = d.textlength(lab, font=F["micro"])
        d.rectangle([PW - tw - 8, py - 7, PW - 1, py + 6], fill=(9, 12, 17, 170))
        d.text((PW - tw - 4, py - 7), lab, fill=(200, 208, 220, 255), font=F["micro"])
    return im


def _cam_overlay(row: dict, cam: np.ndarray, meta: dict, thr: float, F):
    import torch
    from PIL import Image, ImageDraw
    rcv3, rv6 = _G["rcv3"], _G["rv6"]
    proj, rig = _G["proj"][meta["sha12"]], _G["rig"][meta["sha12"]]
    im = Image.fromarray(cam)
    d = ImageDraw.Draw(im, "RGBA")
    hz = proj([[1e5, 0.0]], up=1)
    if hz and hz[0] is not None:
        yh = hz[0][1]
        d.line([(0, yh), (CAM_W, yh)], fill=(150, 132, 62, 170), width=1)
        d.text((8, yh - 15), "horizon predicted from the clip's own extrinsics", fill=(190, 170, 90, 255),
               font=F["micro"])
    eq = _G["eq_rows"]
    if eq > 0:
        ye = CAM_H - eq
        for xx in range(0, CAM_W, 14):
            d.line([(xx, ye), (xx + 7, ye)], fill=(245, 180, 90, 200), width=1)
        txt = _G["eq_caption"]
        d.text((CAM_W - 8 - d.textlength(txt, font=F["micro"]), ye + 3), txt, fill=(245, 180, 90, 255),
               font=F["micro"])
    for j in range(len(row["fan"]) - 1, -1, -1):
        al = int(70 + 110 * (1.0 - j / max(len(row["fan"]), 1)))
        rcv3.polyline(d, proj(rcv3.densify(np.asarray(row["fan"][j])), up=1), C_FAN + (al,), 2)
    # boxes: GT targets (VIS-1 positive solid, IGNORE thin), the 3-D head >= thr in class colour
    g = row["gt"]
    for i, b in enumerate(g["box"]):
        if not (g["pos"][i] or g["ign"][i]):
            continue
        hz_ = bool(g["zh"][i])
        cor = rv6.cuboid_corners(b[0], b[1], g["cz"][i] if hz_ else 0.0, b[2], b[3],
                                 g["h"][i] if hz_ else 0.0, g["yaw"][i])
        rv6.draw_cuboid_cam(d, rig, cor, C_GT + (255 if g["pos"][i] else 150,), width=2 if g["pos"][i] else 1)
    pr = row["pred"]
    placed = []                                   # label rectangles already drawn (no overprinting)
    order = sorted(range(len(pr["p"])), key=lambda j: -pr["p"][j])
    for j in order:
        p = pr["p"][j]
        if p < thr:
            continue
        b = pr["box"][j]
        col = AGENT_PALETTE[int(pr["cls"][j]) % len(AGENT_PALETTE)]
        cor = rv6.cuboid_corners(b[0], b[1], pr["cz"][j], b[2], b[3], pr["h"][j], pr["yaw"][j])
        rv6.draw_cuboid_cam(d, rig, cor, col + (255,), width=2)
    for j in order:                               # labels AFTER all boxes, highest score first
        p = pr["p"][j]
        if p < thr:
            continue
        b = pr["box"][j]
        col = AGENT_PALETTE[int(pr["cls"][j]) % len(AGENT_PALETTE)]
        top = torch.as_tensor([[b[0], b[1], pr["cz"][j] + 0.5 * pr["h"][j]]], dtype=torch.float64)
        cc, rr, ok = rig.project(top)
        if not bool(ok[0]):
            continue
        lab = f"{SHORT_CLS[int(pr['cls'][j])]} {p:.2f}"
        tw = d.textlength(lab, font=F["micro"])
        tx = min(max(float(cc[0]) - tw / 2, 2.0), CAM_W - tw - 2)
        for dy in (-15, -29, 1, -43, 15):
            ty = float(rr[0]) + dy
            rect = (tx - 2, ty, tx + tw + 2, ty + 13)
            if ty < 20 or ty > CAM_H - 14:
                continue
            if all(rect[2] < q[0] or rect[0] > q[2] or rect[3] < q[1] or rect[1] > q[3] for q in placed):
                placed.append(rect)
                d.rectangle(rect, fill=(0, 0, 0, 160))
                d.text((tx, ty), lab, fill=col + (255,), font=F["micro"])
                break
    # GT path (wide, under) then the plan on top; 1 s marks on both
    gv = np.asarray(row["gt_dense_valid"], bool)
    gd = np.asarray(row["gt_dense"])[gv]
    if len(gd):
        rcv3.polyline(d, proj(np.concatenate([np.zeros((1, 2)), gd]), up=1), C_GT + (255,), 6)
        for k in range(9, len(gv), 10):
            if gv[k]:
                q = proj([row["gt_dense"][k]], up=1)[0]
                if q is not None:
                    d.ellipse([q[0] - 4, q[1] - 4, q[0] + 4, q[1] + 4], fill=C_GT + (255,))
    rcv3.polyline(d, proj(rcv3.densify(np.asarray(row["traj"])), up=1), C_SEL + (255,), 4)
    for h, q in zip(HORIZONS, proj(row["traj"], up=1)):
        if q is None:
            continue
        if h % 10 == 0:
            d.ellipse([q[0] - 5, q[1] - 5, q[0] + 5, q[1] + 5], fill=C_SEL + (255,), outline=(15, 15, 15, 255))
            if h in (20, 40, 60):
                d.text((q[0] + 7, q[1] - 7), f"{h // 10}s", fill=C_SEL + (255,), font=F["micro"])
        else:
            d.ellipse([q[0] - 4, q[1] - 4, q[0] + 4, q[1] + 4], outline=C_SEL + (255,), width=2)
    return im


def _ts_chart(d, x0, y0, w, h, title, series, n_total, cur, F, ymin=0.0, ymax=None, unit="",
              legend=None):
    """A running time-series for the CLIP: x = window index (0.1 s apart), drawn up to the current
    window; the cursor marks NOW. ``series``: list of (values, colour, width, dashed)."""
    d.rectangle([x0, y0, x0 + w, y0 + h], fill=(12, 15, 20), outline=(40, 48, 60))
    d.text((x0 + 6, y0 + 3), title, fill=C_FG, font=F["subb"])
    top, bot = y0 + 24, y0 + h - 18
    left, right = x0 + 34, x0 + w - 8
    # the axis is fixed for the WHOLE clip (stable while the trace grows); only the trace is running
    vals = [v for s in series for v in s[0] if v is not None and math.isfinite(v)]
    if ymax is None:
        hi = max(vals) if vals else 1.0
        ymax = max(1e-6, hi * 1.15)
        mag = 10 ** math.floor(math.log10(ymax)) if ymax > 0 else 1
        ymax = math.ceil(ymax / mag * 2) / 2 * mag

    def ypx(v):
        return bot - (min(max(v, ymin), ymax) - ymin) / max(ymax - ymin, 1e-9) * (bot - top)

    def xpx(i):
        return left + i / max(n_total - 1, 1) * (right - left)
    for frac in (0.0, 0.5, 1.0):
        v = ymin + frac * (ymax - ymin)
        d.line([(left, ypx(v)), (right, ypx(v))], fill=(34, 42, 53))
        lab = f"{v:g}"
        d.text((x0 + 4, ypx(v) - 7), lab, fill=C_DIM2, font=F["micro"])
    for vals_, col, wid, dashed in series:
        run_ = []
        for i, v in enumerate(vals_[:cur + 1]):
            if v is None or not math.isfinite(v):
                if len(run_) >= 2:
                    d.line(run_, fill=col, width=wid)
                run_ = []
                continue
            run_.append((xpx(i), ypx(v)))
        if len(run_) >= 2:
            if dashed:
                _dashed(d, run_, col, width=wid, dash=4, gap=3, closed=False)
            else:
                d.line(run_, fill=col, width=wid)
        elif len(run_) == 1:
            d.ellipse([run_[0][0] - 1, run_[0][1] - 1, run_[0][0] + 1, run_[0][1] + 1], fill=col)
    xc = xpx(cur)
    d.line([(xc, top - 2), (xc, bot)], fill=(255, 255, 255, 160), width=1)
    d.text((left, bot + 2), "clip start", fill=C_DIM2, font=F["micro"])
    tl = f"clip end ({n_total} windows, 0.1 s apart){unit}"
    d.text((right - d.textlength(tl, font=F["micro"]), bot + 2), tl, fill=C_DIM2, font=F["micro"])
    if legend:
        lx = x0 + w - 8
        for name, col in reversed(legend):
            tw = d.textlength(name, font=F["micro"])
            lx -= tw + 22
            d.line([(lx, y0 + 11), (lx + 14, y0 + 11)], fill=col, width=3)
            d.text((lx + 17, y0 + 4), name, fill=C_DIM, font=F["micro"])


def _iou(inter_cb, union_cb, c):
    u = float(np.sum(union_cb[c]))
    return (float(np.sum(inter_cb[c])) / u) if u > 0 else None


def _fmt(v, f=".2f", none="—"):
    return none if v is None else format(v, f)


def render_frame(job):  # noqa: C901 -- one frame, top to bottom
    """One window -> one 1920x1080 RGB frame. Every element is drawn on every frame."""
    from PIL import Image, ImageDraw
    from tanitad.viz_standard import VizElement, check_frame
    ci, k = job
    F = _fonts()
    meta = _G["clips"][ci]
    rows = _G["rows"][ci]
    row = rows[k]
    thr = _G["thr"]
    names = _G["names"]
    z = np.load(_G["bank"] / "win" / f"c{ci:02d}_w{k:04d}.npz")
    cam, pred, gt, lv = z["cam"], z["pred"], z["gt"], z["lv"]
    cv = Image.new("RGB", (W_TOT, H_TOT), C_BG)
    d = ImageDraw.Draw(cv, "RGBA")
    # ---- banner ---------------------------------------------------------------------------- #
    d.rectangle([0, 0, W_TOT, H_BAN], fill=C_BAN)
    d.text((PAD, 8), f"{RUN_LABEL} · FINAL step {_G['step']:,}", fill=C_FG, font=F["ban"])
    x_ = PAD + d.textlength(f"{RUN_LABEL} · FINAL step {_G['step']:,}", font=F["ban"]) + 18
    d.text((x_, 12), "OPEN-LOOP perception + planning, held-out eval139 — NOT closed-loop driving",
           fill=C_WARN, font=F["hudb"])
    rt = (f"clip {ci + 1}/{len(_G['clips'])} · sha12 {meta['sha12']} · window {k + 1}/"
          f"{len(rows)} · t = {row['t_label_s']:.1f} s")
    d.text((W_TOT - PAD - d.textlength(rt, font=F["med"]), 9), rt, fill=C_FG, font=F["med"])
    # ---- camera ---------------------------------------------------------------------------- #
    cim = _cam_overlay(row, cam, meta, thr, F)
    cv.paste(cim, (X_CAM, Y_CAM))
    d.rectangle([X_CAM, Y_CAM, X_CAM + CAM_W, Y_CAM + 18], fill=(9, 12, 17, 200))
    d.text((X_CAM + 6, Y_CAM + 2), _fit(d, "FRONT CAMERA — the window's last input frame (the trunk's "
                                            f"bytes) · {meta['camera_label']}", F["micro"], CAM_W - 12),
           fill=C_DIM, font=F["micro"])
    # ---- BEV panels ------------------------------------------------------------------------- #
    pim = _bev_overlays(map_panel(pred, gt, lv, kind="pred"), row, thr, F)
    gim = _bev_overlays(map_panel(gt, gt, lv, kind="gt"), row, thr, F)
    for x0, im, title, sub in ((X_P1, pim, "BEV — PREDICTED 10 cm MAP (refcv7)",
                                "decision rule prior_corrected = argmax(z − log w), as trained"),
                               (X_P2, gim, "BEV — SAM3 GT MAP (10 cm)",
                                "SAM3 codes; same overlays, same scored set")):
        d.text((x0, Y_PT + 2), title, fill=C_FG, font=F["subb"])
        cv.paste(im, (x0, Y_P))
        sub_ = _fit(d, sub, F["micro"], PW - 70)
        d.rectangle([x0 + 1, Y_P + 1, x0 + 6 + d.textlength(sub_, font=F["micro"]), Y_P + 15],
                    fill=(9, 12, 17, 190))
        d.text((x0 + 4, Y_P + 2), sub_, fill=C_DIM, font=F["micro"])
        d.rectangle([x0 - 1, Y_P - 1, x0 + PW, Y_P + PH], outline=(60, 70, 84))
        for yl, lab in ((30.0, "30 m L"), (15.0, "15"), (0.0, "0"), (-15.0, "15"), (-30.0, "30 m R")):
            px, _ = m2px(0.0, yl)
            tw = d.textlength(lab, font=F["micro"])
            xx = min(max(x0 + px - tw / 2, x0), x0 + PW - tw)
            d.text((xx, Y_P + PH + 2), lab, fill=C_DIM2, font=F["micro"])
    # ---- legend under the panels ----------------------------------------------------------- #
    yl0 = Y_LEG
    xl = X_P1
    for i, nm in enumerate(MAP_CLASSES):
        cx_ = xl + (i % 4) * 206
        cy_ = yl0 + (i // 4) * 16
        d.rectangle([cx_, cy_ + 2, cx_ + 14, cy_ + 12], fill=PALETTE[i], outline=(90, 100, 115))
        d.text((cx_ + 19, cy_), nm, fill=C_DIM, font=F["micro"])
    yh_ = yl0 + 33
    for kind, lab, xo in (("A", "SAM3 never saw", 0), ("B", "lift cannot reach now", 160)):
        arr = hatch(kind)[:12, :18]
        cv.paste(Image.fromarray(np.ascontiguousarray(arr)), (xl + xo, yh_ + 1))
        d.rectangle([xl + xo, yh_ + 1, xl + xo + 18, yh_ + 13], outline=(90, 100, 115))
        d.text((xl + xo + 23, yh_), lab, fill=C_DIM, font=F["micro"])
    d.text((xl + 330, yh_), "dimmed = drawn, not scored · full brightness = scored (same set on both)",
           fill=C_DIM, font=F["micro"])
    ya_ = yl0 + 50
    d.text((xl, ya_), "boxes:", fill=C_DIM, font=F["micro"])
    xx_ = xl + 44
    for i, nm in enumerate(SHORT_CLS):
        d.rectangle([xx_, ya_ + 2, xx_ + 12, ya_ + 11], outline=AGENT_PALETTE[i], width=2)
        d.text((xx_ + 16, ya_), nm, fill=C_DIM, font=F["micro"])
        xx_ += 22 + d.textlength(nm, font=F["micro"]) + 10
    d.rectangle([xx_, ya_ + 2, xx_ + 12, ya_ + 11], outline=C_GT, width=2)
    d.text((xx_ + 16, ya_), "GT", fill=C_DIM, font=F["micro"])
    # ---- left info: INPUTS & DECISIONS | THIS FRAME ------------------------------------------ #
    xa, ya, wa = X_CAM, Y_INFO, 500
    d.rectangle([xa, ya, xa + CAM_W, Y_TS - 10], fill=C_PANEL)
    xa += 10
    d.text((xa, ya + 6), "INPUTS (given) · DECISIONS (model)", fill=C_FG, font=F["subb"])
    v0 = row["v0_ms"]
    d.text((xa, ya + 28), f"ego speed v0 {v0:5.2f} m/s ({v0 * 3.6:5.1f} km/h) — measured at t0",
           fill=C_FG, font=F["hud"])
    if row["ceil_kmh"] is not None:
        ctxt = (f"max speed fed {row['v_max_raw_ms']:.1f} m/s → ceiling step {row['ceil_kmh']} km/h "
                f"= {row['v_lim_ms']:.1f} m/s")
    else:
        ctxt = "max speed fed: none valid (ceiling inert)"
    d.text((xa, ya + 50), _fit(d, ctxt, F["hud"], wa), fill=C_GIVEN, font=F["hud"])
    nav_tok = meta.get("nav_token") or "?"
    nav_txt = f"nav = {row['nav_cmd'].upper()} ({nav_tok})"
    d.text((xa, ya + 72), nav_txt, fill=C_GIVEN, font=F["hudb"])
    x_nv = xa + d.textlength(nav_txt, font=F["hudb"]) + 10
    d.text((x_nv, ya + 74), _fit(d, "GIVEN INPUT, not a prediction", F["tiny"], xa + wa - x_nv),
           fill=C_DIM, font=F["tiny"])
    # plan vs ceiling
    exc = row["plan_exceeds_ceiling"]
    ptxt = (f"plan #{row['sel_idx']} (E9): peak {row['plan_vmax_ms']:.1f} m/s"
            + ("" if row["v_lim_ms"] is None else
               f" vs ceiling {row['v_lim_ms']:.1f} → " + ("EXCEEDS the ceiling" if exc else "within")))
    d.text((xa, ya + 96), _fit(d, ptxt, F["subb"], wa), fill=(C_BAD if exc else C_FG), font=F["subb"])
    d.text((xa, ya + 114), _fit(d, f"fan over ceiling {row['n_fan_over_ceiling']}/117 · decoder's pick "
                                   f"#{row['core_sel_idx']} "
                                   + ("obeys" if row["core_sel_in_ceil_keep"] else "does NOT obey")
                                   + " · E9 ignores the mask (§26.1)", F["micro"], wa),
           fill=C_DIM, font=F["micro"])
    # tactical
    yt = ya + 136
    d.text((xa, yt), "TACTICAL — refcv6 tactical decoder vs v7 GT", fill=C_FG, font=F["subb"])
    for li, (hd, pk, pp, gk) in enumerate((("lat", "tac_lat_pred", "tac_lat_p", "tac_lat_gt"),
                                           ("lon", "tac_lon_pred", "tac_lon_p", "tac_lon_gt"))):
        nm = names["tac_lat" if hd == "lat" else "tac_lon"]
        pv, gv_ = row[pk], row[gk]
        yy = yt + 20 + li * 20
        nm_txt = f"{hd}: {nm[pv]}"
        d.text((xa, yy), nm_txt, fill=C_SEL, font=F["hudb"])
        x_p = max(xa + 190, xa + d.textlength(nm_txt, font=F["hudb"]) + 10)
        d.text((x_p, yy + 1), f"p {row[pp]:.2f}", fill=C_DIM, font=F["sub"])
        x_g = x_p + 60
        if gv_ is None:
            d.text((x_g, yy + 1), _fit(d, "GT: not labelled at this instant", F["sub"], xa + wa - x_g),
                   fill=C_DIM2, font=F["sub"])
        else:
            ok = gv_ == pv
            d.text((x_g, yy + 1), _fit(d, f"GT {nm[gv_]} " + ("✓ match" if ok else "✗ mismatch"),
                                        F["sub"], xa + wa - x_g), fill=(C_OK if ok else C_BAD), font=F["sub"])
    gtop = ", ".join(f"{names['goal'][i]} {p:.2f}" for i, p in row["goal_top"])
    d.text((xa, yt + 62), _fit(d, f"goal tokens (top-3 sigmoid): {gtop}", F["micro"], wa), fill=C_DIM,
           font=F["micro"])
    gpos = ", ".join(names["goal"][i] for i in row["goal_gt_pos"]) or (
        "none positive" if row["goal_gt_scored"] else "not labelled at this instant")
    d.text((xa, yt + 76), _fit(d, f"goal GT: {gpos}", F["micro"], wa), fill=C_DIM, font=F["micro"])
    d.text((xa, yt + 96), _fit(d, "strategic: UNAVAILABLE (--no-strategic; no strategic label in "
                                  "PhysicalAI-AV)", F["micro"], wa), fill=C_WARN, font=F["micro"])
    # paths legend
    yp = yt + 116
    d.line([(xa, yp + 7), (xa + 26, yp + 7)], fill=C_GT, width=4)
    d.text((xa + 32, yp), "GT future (dots 1 s)", fill=C_DIM, font=F["tiny"])
    d.line([(xa + 170, yp + 7), (xa + 196, yp + 7)], fill=C_SEL, width=4)
    d.text((xa + 202, yp), f"emitted plan (● 1 s, ○ 0.5/1.5 s; DDIM seed {row['seed']})",
           fill=C_DIM, font=F["tiny"])
    d.line([(xa, yp + 25), (xa + 26, yp + 25)], fill=C_FAN, width=2)
    d.text((xa + 32, yp + 18), f"top-{TOPK_FAN} other candidates by E9 score, faded by rank", fill=C_DIM,
           font=F["tiny"])
    if not all(row["slot_valid"]):
        last = max([h for h, v in zip(HORIZONS, row["slot_valid"]) if v] or [0])
        d.text((xa + 330, yp + 18), f"(!) GT only to {last / 10:g} s (clip ends)", fill=C_WARN,
               font=F["tiny"])
    # ---- this frame's metrics ---------------------------------------------------------------- #
    xb, wb = X_CAM + 528, 486
    d.line([(xb - 12, ya + 6), (xb - 12, Y_TS - 18)], fill=(40, 48, 60))
    d.text((xb, ya + 6), "THIS FRAME (vs GT)", fill=C_FG, font=F["subb"])
    d.text((xb, ya + 26), "ADE 8 slots", fill=C_DIM, font=F["tiny"])
    d.text((xb, ya + 40), _fmt(row["ade_m"]) + " m", fill=C_SEL, font=F["big"])
    d.text((xb + 150, ya + 26), "FDE @ 6 s", fill=C_DIM, font=F["tiny"])
    d.text((xb + 150, ya + 40), _fmt(row["fde_m"]) + " m", fill=C_SEL, font=F["big"])
    d.text((xb + 300, ya + 26), "speed err 0–2 s", fill=C_DIM, font=F["tiny"])
    d.text((xb + 300, ya + 40), _fmt(row["speed_mae_0_2s"]) + " m/s", fill=C_FG, font=F["big"])
    d.text((xb, ya + 80), _fit(d, f"heading err 0–2 s {_fmt(row['heading_mae_0_2s_deg'], '.1f')}° "
                                  f"· curvature err 0–2 s {_fmt(row['curv_mae_0_2s'], '.4f')} 1/m "
                                  "(four_families, 0.5 s grid)", F["micro"], wb),
           fill=C_FG, font=F["micro"])
    # map IoU bars
    ym = ya + 100
    inter, union = np.asarray(row["map_inter"]), np.asarray(row["map_union"])
    d.text((xb, ym), _fit(d, f"MAP IoU this frame — prior_corrected, {row['n_map_scored']:,} scored cells",
                          F["subb"], wb), fill=C_FG, font=F["subb"])
    for c_ in range(len(MAP_CLASSES)):
        yy = ym + 20 + c_ * 15
        v = _iou(inter, union, c_)
        d.rectangle([xb, yy + 3, xb + 10, yy + 12], fill=PALETTE[c_])
        d.text((xb + 15, yy), MAP_CLASSES[c_], fill=C_DIM, font=F["micro"])
        bx0, bw_ = xb + 160, 230
        d.rectangle([bx0, yy + 3, bx0 + bw_, yy + 12], fill=(28, 34, 44))
        if v is not None:
            d.rectangle([bx0, yy + 3, bx0 + bw_ * v, yy + 12], fill=PALETTE[c_])
        nu = float(np.sum(union[c_]))
        d.text((bx0 + bw_ + 8, yy), ("— (absent)" if v is None else f"{v:.2f}") +
               ("" if v is None else f"  (∪ {int(nu):,})"), fill=C_FG, font=F["micro"])
    # boxes
    yb = ym + 20 + 8 * 15 + 6
    dt_ = row["det"]
    d.text((xb, yb), f"BOXES — 3-D box head, drawn at p ≥ {thr:.3f} (greedy 2 m)", fill=C_FG,
           font=F["subb"])
    if not dt_.get("label"):
        d.text((xb, yb + 19), "no agent label on this frame — matches undefined", fill=C_WARN,
               font=F["tiny"])
    else:
        a_, g_ = dt_["thr"], dt_["gate"]
        npos = dt_["n_pos"]
        rec_ = f"R {a_['tp'] / npos:.2f}" if npos else "R —"
        pre_ = f"P {a_['tp'] / a_['n_det']:.2f}" if a_["n_det"] else "P —"
        d.text((xb, yb + 19), _fit(d, f"det {a_['n_det']} · TP {a_['tp']} · FP {a_['fp']} · "
                                      f"DontCare {a_['dontcare']} · VIS-1 positives {npos} → {rec_} "
                                      f"{pre_}", F["tiny"], wb), fill=C_FG, font=F["tiny"])
        d.text((xb, yb + 35), _fit(d, f"declared gate 0.5: det {g_['n_det']}, TP {g_['tp']} · max p "
                                      f"{row['p_max']:.3f} · agent head p≥{dt_['agent_head']['thr']:.2f}: "
                                      f"det {dt_['agent_head']['n_det']}, TP {dt_['agent_head']['tp']}",
                                   F["micro"], wb), fill=C_DIM, font=F["micro"])
    d.text((xb, yb + 51), _fit(d, "GT green: solid VIS-1 positive · dashed IGNORE (DontCare) · faint "
                                  "outside the field / hidden", F["micro"], wb),
           fill=C_DIM2, font=F["micro"])
    # ---- time series ------------------------------------------------------------------------- #
    n = len(rows)
    S_ = _G["series"][ci]
    wts = (W_TOT - 2 * PAD - 3 * 12) // 4
    xs = [PAD + i * (wts + 12) for i in range(4)]
    _ts_chart(d, xs[0], Y_TS, wts, H_TS, "plan error (m)",
              [(S_["ade"], C_SEL, 2, False), (S_["fde"], (253, 186, 116), 1, True)],
              n, k, F, legend=[("ADE", C_SEL), ("FDE 6 s", (253, 186, 116))])
    _ts_chart(d, xs[1], Y_TS, wts, H_TS, "speed (m/s)",
              [(S_["vl"], C_GIVEN, 1, True), (S_["v0"], C_GT, 2, False), (S_["pvm"], C_SEL, 2, False)],
              n, k, F, ymax=S_["ymax_speed"],
              legend=[("v0 measured", C_GT), ("plan peak", C_SEL), ("ceiling", C_GIVEN)])
    _ts_chart(d, xs[2], Y_TS, wts, H_TS, "map IoU per class",
              [(S_["iou"][c_], PALETTE[c_], w_, False) for c_, w_ in ((1, 2), (7, 2), (2, 1), (3, 1),
                                                                      (5, 1), (6, 1))],
              n, k, F, ymax=1.0,
              legend=[("drv", PALETTE[1]), ("side", PALETTE[7]), ("lane", PALETTE[2]),
                      ("xwalk", PALETTE[3]), ("edge", PALETTE[5]), ("hatch", PALETTE[6])])
    _ts_chart(d, xs[3], Y_TS, wts, H_TS, f"boxes (3-D head, p ≥ {thr:.2f})",
              [(S_["npos"], C_GT, 2, False), (S_["ndet"], (200, 210, 225), 1, False),
               (S_["tp"], (80, 170, 255), 2, False)],
              n, k, F, ymax=S_["ymax_box"],
              legend=[("VIS-1 pos", C_GT), ("det", (200, 210, 225)), ("TP", (80, 170, 255))])
    # ---- the viz standard: declare, or do not render ----------------------------------------- #
    els = [
        VizElement.present("camera", "GT path + emitted plan + top-k fan + GT/3-D boxes projected",
                           source="render_refcv3_video.CylProjector (paths) + model RigCamera via "
                                  "render_refcv6_map_video.draw_cuboid_cam (boxes) on item['frames'][-1][-3:]",
                           kind="derived"),
        VizElement.present("bev", "predicted 10 cm map | SAM3 GT map + plan + boxes",
                           source="map_head_hires.decide(out['perception']['map_hires_logits'], "
                                  "'prior_corrected', class_weight) | batch['map_fine']",
                           kind="model_output", conditioned_on=("nav_cmd", "v_max_ms")),
        VizElement.present("tactical", f"lat {names['tac_lat'][row['tac_lat_pred']]} / lon "
                                       f"{names['tac_lon'][row['tac_lon_pred']]}",
                           source="out['tacv6_lat_logits'/'tacv6_lon_logits'].argmax(-1)",
                           kind="model_output", conditioned_on=("nav_cmd", "v_max_ms")),
        (VizElement.present("tactical_gt", f"lat {names['tac_lat'][row['tac_lat_gt']]} / lon "
                                           f"{names['tac_lon'][row['tac_lon_gt']]}",
                            source="item['lat_v7'] / item['lon_v7'] (v8 eval labels)", kind="gt_label")
         if (row["tac_lat_gt"] is not None and row["tac_lon_gt"] is not None) else
         VizElement.unavailable("tactical_gt", "outside the label record's band at this instant "
                                               "(lat_v7/lon_v7 = IGNORE)")),
        VizElement.unavailable("strategic", "--no-strategic: the strategic level is OFF in this arm, and "
                                            "PhysicalAI-AV carries no strategic label"),
        VizElement.present("ade", f"ADE {_fmt(row['ade_m'])} m / FDE {_fmt(row['fde_m'])} m",
                           source="||out['traj'] - refb_labels.waypoint_targets(...)|| over valid slots",
                           kind="derived"),
        VizElement.present("strategic_input", row["nav_cmd"],
                           source="V3Dataset.enable_nav_from_v7 (v8 eval labels, oracle nav)",
                           kind="given_input"),
    ]
    check_frame(els, where=f"refcv7 @ {meta['sha12']}/w{k}")
    arr = np.asarray(cv)
    _assert_content(arr, f"{meta['sha12']} w{k}")
    return arr


def _assert_content(arr: np.ndarray, where: str):
    """⛔ A renderer that decoded nothing writes a valid BLACK video and exits 0 (refav1 arms reel §4e):
    every frame must carry a camera region and BEV panels that are not blank."""
    if arr.shape != (H_TOT, W_TOT, 3):
        raise SystemExit(f"[rv7] frame {where} is {arr.shape}")
    cam = arr[Y_CAM + 20:Y_CAM + CAM_H, X_CAM:X_CAM + CAM_W]
    bev = arr[Y_P:Y_P + PH, X_P1:X_P1 + PW]
    if float(cam.mean()) < 8.0 or float(cam.std()) < 4.0:
        raise SystemExit(f"[rv7] frame {where}: camera region is blank (mean {cam.mean():.1f})")
    if float(bev.std()) < 3.0:
        raise SystemExit(f"[rv7] frame {where}: BEV panel is blank")


def _wrap(d, text, f, w):
    return _G["rcv3"].wrap(d, text, f, w)


def opening_card(rec_f: dict, summ: dict):
    from PIL import Image, ImageDraw
    F = _fonts()
    im = Image.new("RGB", (W_TOT, H_TOT), C_BG)
    d = ImageDraw.Draw(im)
    x = 110
    d.text((x, 60), f"{RUN_LABEL} — the FINAL checkpoint, frame by frame", fill=C_FG, font=F["title"])
    ck = rec_f["ckpt"]
    lines = [
        (f"step {_G['step']:,} (ck['step'] asserted; strict load 0 missing / 0 unexpected) · ckpt.pt md5 "
         f"{ck['md5']} ({ck['bytes']:,} B)", C_FG, "hud"),
        (f"launch tree fec3a0dccf (refcv7_loader.py replays train(); trunk ResNet-101 @ 416×1024, DDIM "
         f"x{rec_f['model'].get('decoder_steps')}, 117 v0-conditioned anchors) · config.json md5 "
         f"{rec_f['config']['md5']}", C_DIM, "sub"),
        ("", C_FG, "sub"),
        ("⚠ " + TIER_NOTE, C_WARN, "h2"),
        ("Each frame is one held-out eval window: the model sees the recorded camera window and the measured "
         "speed at t0, and emits a 6 s plan, a 10 cm semantic map and 3-D boxes. The plan never drives the "
         "car; the next frame comes from the recording.", C_FG, "hud"),
        ("⚠ " + CEIL_NOTE + ": the decoder masks over-ceiling candidates in its own ranking, but the E9 "
         "goal selection re-ranks the fan without that mask — each frame says whether the emitted plan "
         "exceeds the fed ceiling.", C_WARN, "hud"),
        ("", C_FG, "sub"),
        ("WHAT EACH FRAME SHOWS", C_FG, "med"),
        ("• FRONT CAMERA (top-left): GT future path (green, dots every 1 s), the EMITTED plan out['traj'] "
         f"(orange; filled = 1 s slots), the top-{TOPK_FAN} other candidates (faded cyan), GT boxes (green "
         f"cuboids) and the 3-D box head's boxes at p ≥ {_G['thr']:.3f} with class + score.", C_DIM, "hud"),
        ("• BEV (top-right): the predicted 10 cm map (decision rule prior_corrected, as trained) beside the "
         "SAM3 GT map, 100 m ahead × ±30 m, range rings every 10 m, the same paths and boxes on both; "
         "left is left, forward is up.", C_DIM, "hud"),
        ("• Middle: given inputs (speed, fed max speed, nav), the tactical decoder vs v7 GT, and this "
         "frame's ADE/FDE, speed/heading/curvature error, per-class map IoU and box matches.", C_DIM, "hud"),
        ("• Bottom: running per-clip time series (plan error, speed vs ceiling, map IoU, boxes).", C_DIM,
         "hud"),
        ("", C_FG, "sub"),
        ("KNOWN WEAKNESSES YOU SHOULD SEE (not hidden)", C_FG, "med"),
        (f"• thin map classes (edge, hatched) are almost never predicted: in-run eval IoU edge 0–20 m "
         f"{summ['edge_0_20']:.4f}, hatched {summ['hatched_0_20']:.4f} (metrics.jsonl, step 50,400).", C_DIM,
         "hud"),
        (f"• box confidence is very low: at the declared gate 0.5 conf_ratio = {summ['conf_ratio']:.4f} "
         f"(recall {summ['rec_gate']:.4f}); boxes are therefore drawn at the run's own P=R operating point "
         f"p ≥ {_G['thr']:.4f} (eval_box3d_calib_pr_gate on 256 TRAIN calibration windows) — not "
         "tuned on these clips; the score is printed on every box.", C_DIM, "hud"),
        ("• one DDIM draw per window (torch.manual_seed(0) before every forward): the plan is one sample "
         "of a stochastic planner; the map and boxes do not depend on the draw (checked per clip).", C_DIM,
         "hud"),
        ("", C_FG, "sub"),
        (f"{len(_G['clips'])} clips, every eval window in time order at 10 fps = real time (0.1 s per window); "
         "clip rule fixed before rendering (GT + labels only): 4 per nav token (left/right/follow) × low/"
         "high speed × most-VRU / most-lead-vehicle, ties by smallest sha12.", C_FG, "hud"),
        ("UNAVAILABLE: strategic decision (--no-strategic; no strategic label in PhysicalAI-AV). No interval is "
         "quoted anywhere: one clip = one episode, its windows are not independent.", C_DIM, "hud"),
    ]
    y = 130
    for txt, col, fk in lines:
        if not txt:
            y += 12
            continue
        for sub in _wrap(d, txt, F[fk], W_TOT - 2 * x):
            d.text((x, y), sub, fill=col, font=F[fk])
            y += {"title": 46, "h2": 33, "med": 27, "hud": 22, "sub": 19}[fk]
    # ---- the colour key, drawn with the SAME constants every frame uses ------------------------ #
    y += 18
    d.text((x, y), "COLOUR KEY (one meaning per colour)", fill=C_FG, font=F["med"])
    y += 34
    xx = x
    for col, lab, w_ in ((C_GT, "ground truth (path, boxes)", 6), (C_SEL, "the model's emitted plan", 5),
                         (C_FAN, "other candidates of the same fan", 2),
                         (C_GIVEN, "a GIVEN input (text only)", 5)):
        d.line([(xx, y + 10), (xx + 40, y + 10)], fill=col, width=w_)
        d.text((xx + 50, y), lab, fill=C_DIM, font=F["hud"])
        xx += 60 + d.textlength(lab, font=F["hud"]) + 30
    y += 34
    xx = x
    d.text((xx, y), "map:", fill=C_DIM, font=F["hud"])
    xx += 54
    for i, nm in enumerate(MAP_CLASSES):
        d.rectangle([xx, y + 3, xx + 18, y + 17], fill=PALETTE[i], outline=(90, 100, 115))
        d.text((xx + 24, y + 1), nm, fill=C_DIM, font=F["sub"])
        xx += 34 + d.textlength(nm, font=F["sub"]) + 8
    y += 32
    xx = x
    d.text((xx, y), "boxes:", fill=C_DIM, font=F["hud"])
    xx += 66
    for i, nm in enumerate(AGENT_CLASSES):
        d.rectangle([xx, y + 3, xx + 18, y + 17], outline=AGENT_PALETTE[i], width=3)
        d.text((xx + 24, y), f"{nm} ({SHORT_CLS[i]})", fill=C_DIM, font=F["sub"])
        xx += 34 + d.textlength(f"{nm} ({SHORT_CLS[i]})", font=F["sub"]) + 8
    return im


def title_card(ci: int, cs: dict):
    from PIL import Image, ImageDraw
    F = _fonts()
    meta = _G["clips"][ci]
    im = Image.new("RGB", (W_TOT, H_TOT), C_BG)
    d = ImageDraw.Draw(im)
    x = 120
    d.text((x, 120), f"Clip {ci + 1} of {len(_G['clips'])}  —  nav {meta['nav'].upper()} "
                     f"({meta.get('nav_token')}, a GIVEN INPUT)", fill=C_FG, font=F["title"])
    s = meta["selection"]
    d.text((x, 182), f"sha12 {meta['sha12']}  ·  {cs['n']} eval windows, every one, in time order  "
                     f"·  t {cs['t0']:.1f} … {cs['t1']:.1f} s  ·  v0 {s['v_min']:.1f} … "
                     f"{s['v_max']:.1f} m/s (mean {s['v_mean']:.1f})", fill=C_DIM, font=F["med"])
    d.text((x, 236), "why chosen: " + meta["why"], fill=C_GIVEN, font=F["hud"])
    d.text((x, 262), f"VRU windows {s['n_vru_win']} · lead-vehicle windows {s['n_lead_win']} · "
                     f"agent-labelled windows {s['n_agent_labelled_win']} (slot {meta['slot']})", fill=C_DIM,
           font=F["hud"])
    d.text((x, 310), f"{RUN_LABEL} · step {_G['step']:,} · " + TIER_NOTE, fill=C_WARN, font=F["hud"])
    y = 360
    d.text((x, y), "this clip, all windows (one episode — correlated windows, no interval quoted):",
           fill=C_DIM, font=F["hud"])
    L_ = [
        f"plan: ADE mean {_fmt(cs['ade_mean'])} m (median {_fmt(cs['ade_median'])}) · FDE@6 s mean "
        f"{_fmt(cs['fde_mean'])} m (n {cs['n_fde']}) · speed err 0–2 s {_fmt(cs['spd_mean'])} m/s · "
        f"heading err {_fmt(cs['head_mean'], '.1f')}° · curvature err {_fmt(cs['curv_mean'], '.4f')} 1/m",
        f"ceiling: emitted plan exceeds the fed ceiling on {cs['n_exceed']}/{cs['n']} windows (SPEC 26.1)",
        "map IoU pooled over the clip (sum of intersections / sum of unions, trainer rule; — = class "
        "absent): " + " · ".join(
            f"{CLASS_KEYS[c]} {_fmt(v)}" for c, v in enumerate(cs["iou_pooled"])),
        f"boxes (3-D head, p ≥ {_G['thr']:.3f}, greedy 2 m, VIS-1): TP {cs['tp']} / det {cs['ndet']} / "
        f"positives {cs['npos']} → recall {_fmt(cs['recall'])} · precision {_fmt(cs['precision'])} "
        f"· at gate 0.5: det {cs['ndet_gate']}",
        f"tactical (refcv6 decoder vs v7 GT, labelled windows only): lat {cs['lat_acc']} · lon {cs['lon_acc']}",
        "draw independence (seed 0 vs 1, first window): plan moved "
        f"{cs['di']['traj_max_abs_diff_m']:.2f} m; map logits max |Δ| {cs['di']['map_logits_max_abs_diff']:.1e}; "
        f"box presence max |Δ| {cs['di']['box_presence_max_abs_diff']:.1e}",
        f"orientation control: GT path on SAM3 road surface {cs['orient']['file']:.3f} vs "
        f"{cs['orient']['mirror']:.3f} under a lateral mirror (n {cs['orient']['n']} path points)",
    ]
    yy = y + 34
    for ln in L_:
        for sub in _wrap(d, ln, F["hud"], W_TOT - 2 * x - 20):
            d.text((x + 20, yy), sub, fill=C_FG, font=F["hud"])
            yy += 22
        yy += 24
    return im


def clip_summary(rows: list, meta: dict) -> dict:
    def mean(vs):
        vs = [v for v in vs if v is not None]
        return float(np.mean(vs)) if vs else None
    inter = np.sum([np.asarray(r["map_inter"]) for r in rows], axis=0)
    union = np.sum([np.asarray(r["map_union"]) for r in rows], axis=0)
    lab = [r for r in rows if r["det"].get("label")]
    tp = sum(r["det"]["thr"]["tp"] for r in lab)
    ndet = sum(r["det"]["thr"]["n_det"] for r in lab)
    npos = sum(r["det"]["n_pos"] for r in lab)

    def acc(pk, gk):
        lw = [r for r in rows if r[gk] is not None]
        if not lw:
            return "no labelled window"
        return f"{sum(r[pk] == r[gk] for r in lw)}/{len(lw)} = {sum(r[pk] == r[gk] for r in lw) / len(lw):.2f}"
    # orientation control: GT path points on SAM3 road surface, file contract vs lateral mirror
    on = onm = n = 0
    for r in rows[::5]:
        z = np.load(_G["bank"] / "win" / f"c{r['clip_rank'] - 1:02d}_w{r['win']:04d}.npz")
        g = z["gt"]
        for (x, y), v in zip(r["gt_dense"], r["gt_dense_valid"]):
            if not v:
                continue
            a_, b_ = cell_of(x, y), cell_of(x, -y)
            if a_ is None or b_ is None or g[a_] == NOT_SEEN or g[b_] == NOT_SEEN:
                continue
            n += 1
            on += int(g[a_] in ROAD_SURFACE)
            onm += int(g[b_] in ROAD_SURFACE)
    return {"n": len(rows), "t0": rows[0]["t_label_s"], "t1": rows[-1]["t_label_s"],
            "ade_mean": mean([r["ade_m"] for r in rows]),
            "ade_median": float(np.median([r["ade_m"] for r in rows if r["ade_m"] is not None])),
            "fde_mean": mean([r["fde_m"] for r in rows]), "n_fde": sum(r["fde_m"] is not None for r in rows),
            "spd_mean": mean([r["speed_mae_0_2s"] for r in rows]),
            "head_mean": mean([r["heading_mae_0_2s_deg"] for r in rows]),
            "curv_mean": mean([r["curv_mae_0_2s"] for r in rows]),
            "n_exceed": sum(bool(r["plan_exceeds_ceiling"]) for r in rows),
            "iou_pooled": [(float(inter[c].sum() / union[c].sum()) if union[c].sum() > 0 else None)
                           for c in range(len(CLASS_KEYS))],
            "map_inter_by_class_band": inter.tolist(), "map_union_by_class_band": union.tolist(),
            "tp": tp, "ndet": ndet, "npos": npos, "n_windows_agent_labelled": len(lab),
            "recall": (tp / npos) if npos else None, "precision": (tp / ndet) if ndet else None,
            "ndet_gate": sum(r["det"]["gate"]["n_det"] for r in lab),
            "tp_gate": sum(r["det"]["gate"]["tp"] for r in lab),
            "lat_acc": acc("tac_lat_pred", "tac_lat_gt"), "lon_acc": acc("tac_lon_pred", "tac_lon_gt"),
            "di": meta["draw_independence"],
            "orient": {"file": on / n if n else float("nan"), "mirror": onm / n if n else float("nan"), "n": n},
            "estimator": "plain means over the clip's windows; IoU pooled sum/sum; no interval (one episode)"}


def _init_worker(state):
    _G.update(state)


def clip_series(rows: list) -> dict:
    """The per-clip time series, computed ONCE (every frame draws a prefix of them)."""
    lab = [r["det"].get("label") for r in rows]
    s = {"ade": [r["ade_m"] for r in rows], "fde": [r["fde_m"] for r in rows],
         "v0": [r["v0_ms"] for r in rows], "pvm": [r["plan_vmax_ms"] for r in rows],
         "vl": [r["v_lim_ms"] for r in rows],
         "iou": {c: [_iou(np.asarray(r["map_inter"]), np.asarray(r["map_union"]), c) for r in rows]
                 for c in range(len(CLASS_KEYS))},
         "npos": [(r["det"]["n_pos"] if lb else None) for r, lb in zip(rows, lab)],
         "ndet": [(r["det"]["thr"]["n_det"] if lb else None) for r, lb in zip(rows, lab)],
         "tp": [(r["det"]["thr"]["tp"] if lb else None) for r, lb in zip(rows, lab)]}
    sp = [x for x in s["v0"] + s["pvm"] + s["vl"] if x is not None]
    s["ymax_speed"] = float(math.ceil(max(sp + [5.0]) * 1.1 / 5) * 5)
    s["ymax_box"] = float(max([x for x in s["npos"] + s["ndet"] if x is not None] + [3]) + 1)
    return s


def stage_render(a) -> int:  # noqa: C901
    t_all = time.time()
    tree, work = Path(a.tree), Path(a.work)
    tag = "smoke" if a.smoke else "final"
    bank = work / "bank" / tag
    if not (bank / "FORWARD_DONE").exists():
        raise SystemExit(f"[rv7] {bank}/FORWARD_DONE is missing -- run --stage forward first")
    out_dir = work / "out" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(tree / "stack"))
    import av
    import torch
    from PIL import Image
    from tanitad.data.calib import CanonicalFrame
    from tanitad.data.rig_projection import RigCamera
    torch.set_num_threads(1)
    rcv3 = _by_path("render_refcv3_video_rv7", tree / "taniteval/tools/render_refcv3_video.py")
    rv6 = _by_path("render_refcv6_map_video_rv7", tree / "taniteval/tools/render_refcv6_map_video.py")
    rec_f = json.loads((bank / "forward_record.json").read_text(encoding="utf-8"))
    rows_all = [json.loads(ln) for ln in open(bank / "rows.jsonl", encoding="utf-8") if ln.strip()]
    clips = rec_f["clips"]
    rows = [[r for r in rows_all if r["clip_rank"] == ci + 1] for ci in range(len(clips))]
    for ci, rr in enumerate(rows):
        if [r["win"] for r in rr] != list(range(len(rr))) or len(rr) != clips[ci]["n_windows_rendered"]:
            raise SystemExit(f"[rv7] clip {ci} rows are not complete and in order")
    # cameras rebuilt from the banked R, t, frame; CHECKED against the model's own projections
    projs, rigs, cam_chk = {}, {}, {}
    for m in clips:
        fr = m["frame"]
        rig = RigCamera(R_cam_to_rig=torch.tensor(m["rig_camera"]["R_cam_to_rig"], dtype=torch.float64),
                        t_cam_in_rig=torch.tensor(m["rig_camera"]["t_cam_in_rig"], dtype=torch.float64),
                        frame=CanonicalFrame(height=fr["height"], width=fr["width"], f_ref=fr["f_ref"],
                                             projection=fr["projection"]))
        pc, pr_, pok = rig.project(torch.tensor(m["rig_camera"]["probe_xyz"], dtype=torch.float64))
        dmax = max(abs(float(pc[i]) - q[0]) + abs(float(pr_[i]) - q[1])
                   for i, q in enumerate(m["rig_camera"]["probe_col_row_ok"]))
        if dmax > 1e-9 or [bool(x) for x in pok] != [q[2] for q in m["rig_camera"]["probe_col_row_ok"]]:
            raise SystemExit(f"[rv7] rebuilt RigCamera for {m['sha12']} differs from the model's by {dmax}")
        cam_chk[m["sha12"]] = dmax
        rigs[m["sha12"]] = rig
        projs[m["sha12"]] = rcv3.CylProjector(fr, m["extrinsics"])
    eq = rec_f["equalize_bottom_rows"]
    n_eq = max(int(eq["trunk_as_trained"]), int(eq["lift_bank_hires"]))
    eq_caption = (f"↓ bottom {n_eq} rows: zeroed in the trunk as trained ({eq['trunk_as_trained']}) "
                  f"and unobserved for the BEV lift ({eq['lift_bank_hires']})")
    names = rec_f["dataset"]["label_names"]
    thr = float(rec_f["detection_display"]["box3d_threshold"])
    ev = rec_f["inrun_eval_row"]["row"]
    summ_ev = {"edge_0_20": float(ev["eval_map_hires_iou_edge_0_20"]),
               "hatched_0_20": float(ev["eval_map_hires_iou_hatched_0_20"]),
               "conf_ratio": float(ev["eval_box3d_conf_ratio"]), "rec_gate": float(ev["eval_box3d_rec@gate"])}
    state = {"rcv3": rcv3, "rv6": rv6, "bank": bank, "clips": clips, "rows": rows, "thr": thr,
             "names": names, "proj": projs, "rig": rigs, "eq_rows": n_eq, "eq_caption": eq_caption,
             "step": int(a.expect_step)}
    _G.update(state)
    _G["series"] = [clip_series(rows[ci]) for ci in range(len(clips))]
    summaries = [clip_summary(rows[ci], clips[ci]) for ci in range(len(clips))]
    for ci, s in enumerate(summaries):
        if not (s["orient"]["n"] > 0 and s["orient"]["file"] >= s["orient"]["mirror"]):
            _p(f"[rv7] ⚠ orientation control on clip {ci + 1}: file {s['orient']['file']:.3f} < mirror "
               f"{s['orient']['mirror']:.3f} (n {s['orient']['n']}) -- recorded, inspected by eye")
    # ---- the sequence ------------------------------------------------------------------------ #
    n_open = int(round(a.open_s * FPS))
    n_title = int(round(a.title_s * FPS))
    seq = [("open", None)] * n_open
    for ci in range(len(clips)):
        seq += [("title", ci)] * n_title
        seq += [("win", (ci, k)) for k in range(len(rows[ci]))]
    if a.smoke:
        n_open, n_title = 1, 1
        seq = [("open", None), ("title", 0)] + [("win", (0, k)) for k in range(len(rows[0]))]
    reel = out_dir / ("refcv7_final_50400_reel.mp4" if not a.smoke else "refcv7_smoke_reel.mp4")
    cont = av.open(str(reel), mode="w", options={"movflags": "+faststart"})
    st = cont.add_stream("libx264", rate=FPS)
    st.width, st.height, st.pix_fmt = W_TOT, H_TOT, "yuv420p"
    st.options = {"crf": str(a.crf), "preset": a.preset}
    kf_dir = out_dir / "keyframes"
    kf_dir.mkdir(exist_ok=True)
    cards = {"open": np.asarray(opening_card(rec_f, summ_ev))}
    for ci in range(len(clips)):
        cards[("title", ci)] = np.asarray(title_card(ci, summaries[ci]))
    Image.fromarray(cards["open"]).save(kf_dir / "card_opening.png")
    Image.fromarray(cards[("title", 0)]).save(kf_dir / "card_title_clip01.png")
    for c_ in cards.values():
        if c_.shape != (H_TOT, W_TOT, 3):
            raise SystemExit("[rv7] a card is not 1920x1080")
    jobs = [j for kind, j in seq if kind == "win"]
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    t_r = time.time()
    n_frames = 0
    kf_set = set()
    for ci in range(len(clips)):
        nw = len(rows[ci])
        kf_set |= {(ci, int(round(q * (nw - 1)))) for q in (0.15, 0.6)}

    def emit(arr):
        nonlocal n_frames
        fr_ = av.VideoFrame.from_ndarray(arr, format="rgb24")
        for pkt in st.encode(fr_):
            cont.mux(pkt)
        n_frames += 1
    from collections import deque
    with ctx.Pool(processes=a.workers, initializer=_init_worker, initargs=({},)) as pool:
        # a BOUNDED window of in-flight frames (imap has no back-pressure: 2,000 x 6.2 MB would queue)
        pending, nxt = deque(), 0
        max_inflight = 3 * a.workers
        for kind, j in seq:
            if kind == "open":
                emit(cards["open"])
            elif kind == "title":
                emit(cards[("title", j)])
            else:
                while nxt < len(jobs) and len(pending) < max_inflight:
                    pending.append(pool.apply_async(render_frame, (jobs[nxt],)))
                    nxt += 1
                arr = pending.popleft().get()
                emit(arr)
                if j in kf_set:
                    Image.fromarray(arr).save(kf_dir / f"c{j[0] + 1:02d}_w{j[1] + 1:04d}.png")
            if n_frames % 200 == 0:
                _p(f"[rv7] {n_frames}/{len(seq)} frames ({time.time() - t_r:.0f} s)")
    for pkt in st.encode():
        cont.mux(pkt)
    cont.close()
    if n_frames != len(seq):
        raise SystemExit(f"[rv7] encoded {n_frames} frames, expected {len(seq)}")
    render_s = time.time() - t_r
    # ---- decode-back: count frames, check size (the reel is read, not trusted) ---------------- #
    dec = av.open(str(reel))
    vs = dec.streams.video[0]
    n_dec = 0
    for _f in dec.decode(vs):
        if n_dec == 0 and (_f.width, _f.height) != (W_TOT, H_TOT):
            raise SystemExit(f"[rv7] decoded {_f.width}x{_f.height}")
        n_dec += 1
    dec.close()
    if n_dec != n_frames:
        raise SystemExit(f"[rv7] decoded {n_dec} frames, wrote {n_frames}")
    vid = {"path": str(reel), "bytes": reel.stat().st_size, "md5": md5_file(reel), "frames": n_frames,
           "decoded_frames": n_dec, "fps": FPS, "duration_s": round(n_frames / FPS, 1),
           "resolution": f"{W_TOT}x{H_TOT}", "codec": "libx264 yuv420p (PyAV " + av.__version__ + ")",
           "crf": a.crf, "preset": a.preset, "open_card_frames": n_open, "title_frames_per_clip": n_title,
           "render_encode_s": round(render_s, 1)}
    _p(f"[rv7] reel {reel} {vid['bytes']:,} B, {n_frames} frames ({vid['duration_s']} s), "
       f"{render_s:.0f} s")
    prev = make_preview(reel, out_dir / ("refcv7_final_50400_preview.mp4" if not a.smoke
                                         else "refcv7_smoke_preview.mp4"), n_frames, a.preview_mb)
    # ---- the record -------------------------------------------------------------------------- #
    per_clip = []
    for ci, m in enumerate(clips):
        s = summaries[ci]
        per_clip.append({"clip_rank": ci + 1, "sha12": m["sha12"], "nav": m["nav"], "nav_token": m["nav_token"],
                         "slot": m["slot"], "why": m["why"], "selection": m["selection"],
                         "n_windows": s["n"], "t_label_s": [s["t0"], s["t1"]],
                         "plan": {"ade_mean_m": s["ade_mean"], "ade_median_m": s["ade_median"],
                                  "fde6s_mean_m": s["fde_mean"], "n_fde": s["n_fde"],
                                  "speed_mae_0_2s_mean": s["spd_mean"], "heading_mae_0_2s_deg_mean": s["head_mean"],
                                  "curv_mae_0_2s_mean": s["curv_mean"],
                                  "n_windows_plan_exceeds_ceiling": s["n_exceed"]},
                         "map_iou_pooled": dict(zip(CLASS_KEYS, s["iou_pooled"])),
                         "map_inter_by_class_band": s["map_inter_by_class_band"],
                         "map_union_by_class_band": s["map_union_by_class_band"],
                         "boxes_box3d_at_display_thr": {"thr": thr, "tp": s["tp"], "n_det": s["ndet"],
                                                        "n_pos_vis1": s["npos"], "recall": s["recall"],
                                                        "precision": s["precision"],
                                                        "n_windows_labelled": s["n_windows_agent_labelled"]},
                         "boxes_box3d_at_gate_0p5": {"n_det": s["ndet_gate"], "tp": s["tp_gate"]},
                         "tactical_lat_acc": s["lat_acc"], "tactical_lon_acc": s["lon_acc"],
                         "draw_independence": s["di"], "orientation_control": s["orient"],
                         "projector_cross_check_px_max": m["projector_cross_check_px_max"],
                         "rebuilt_rigcamera_vs_model_max_px": cam_chk[m["sha12"]],
                         "camera": {"label": m["camera_label"], "codec": m["codec"]},
                         "estimator": s["estimator"]})
    rec = {"tool": "taniteval/tools/render_refcv7_video.py", "tool_md5": md5_file(__file__),
           "what": f"{RUN_LABEL} final checkpoint, every eval window of {len(clips)} eval139 clips, frame by frame",
           "tier": TIER_NOTE, "ceiling_stamp": CEIL_NOTE,
           "forward": {k: rec_f[k] for k in ("started_utc", "finished_utc", "code", "ckpt", "config",
                                             "run_summary", "inrun_eval_row", "detection_display", "model",
                                             "equalize_bottom_rows", "dataset", "identity_checks", "compute",
                                             "extrinsics", "departures", "palette", "n_windows", "tool_md5")},
           "clip_selection": json.loads((bank / "clip_selection.json").read_text(encoding="utf-8")),
           "per_clip": per_clip,
           "metric_definitions": {
               "ade_m": "mean L2 over the VALID of the 8 emitted slots (0.5,1,1.5,2,3,4,5,6 s) between out['traj'] "
                        "and refb_labels.waypoint_targets(pose_last, future_poses_ext, horizons)",
               "fde_m": "L2 at the 6 s slot (only when valid)",
               "speed/heading/curvature": "taniteval.four_families._seq_geometry on the first 4 slots (0.5 s grid), "
                                          "plan vs GT, masked as that function masks",
               "map_iou": "map_head_hires.per_class_signal inter/union under the declared rule prior_corrected on "
                          "seen & lift-valid cells; per window == the trainer's eval extras (identity-checked)",
               "boxes": "tanitad.eval.detection_metrics.greedy_rows (greedy 2 m, VIS-1 positives, IGNORE = DontCare) "
                        "on the trainer's own eval pack _det_pack_box3d",
               "tactical": "argmax of out['tacv6_lat_logits'/'tacv6_lon_logits'] vs item['lat_v7'/'lon_v7']",
               "plan_exceeds_ceiling": "refcv6_selection.planned_max_speed(out['traj']) > limit_ms_of_bin(the model's "
                                       "own max_speed_1h_v6 bin)"},
           "unavailable": {"strategic": "--no-strategic: the strategic level is OFF in this arm; PhysicalAI-AV "
                                        "carries no strategic label (four_families.STRATEGIC_UNAVAILABLE_REASON)",
                           "interval": "no interval: the windows of one clip are one episode"},
           "video": vid, "preview": prev,
           "keyframes": sorted(str(p.name) for p in kf_dir.glob("*.png")),
           "compute": {"render_wall_s": round(time.time() - t_all, 1), "workers": a.workers,
                       "forward_gpu_s": rec_f["compute"]["gpu_forward_s"],
                       "forward_wall_s": rec_f["compute"]["wall_s"],
                       "cuda_max_memory_allocated_gib": rec_f["compute"]["cuda_max_memory_allocated_gib"]},
           "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_json(out_dir / "render_record.json", rec)
    shutil.copyfile(bank / "clip_selection.json", out_dir / "clip_selection.json")
    _p(f"[rv7] RENDER DONE in {time.time() - t_all:.0f} s -> {out_dir}")
    return 0


def make_preview(src: Path, dst: Path, n_frames: int, max_mb: float) -> dict:
    """720p preview: decode the reel, downscale (Lanczos), re-encode at a bitrate sized to ``max_mb``;
    retried at 85 % if the first pass lands over the limit."""
    import av
    from PIL import Image
    dur = n_frames / FPS
    out = {}
    for frac in (0.88, 0.75, 0.62):
        br = int(max_mb * 1e6 * 8 * frac / dur)
        cin = av.open(str(src))
        cout = av.open(str(dst), mode="w", options={"movflags": "+faststart"})
        st = cout.add_stream("libx264", rate=FPS)
        st.width, st.height, st.pix_fmt = 1280, 720, "yuv420p"
        st.bit_rate = br
        st.options = {"preset": "slow", "maxrate": str(int(br * 1.5)), "bufsize": str(int(br * 3))}
        n = 0
        for f in cin.decode(cin.streams.video[0]):
            im = f.to_image().resize((1280, 720), Image.LANCZOS)
            for pkt in st.encode(av.VideoFrame.from_image(im)):
                cout.mux(pkt)
            n += 1
        for pkt in st.encode():
            cout.mux(pkt)
        cout.close()
        cin.close()
        size = dst.stat().st_size
        out = {"path": str(dst), "bytes": size, "md5": md5_file(dst), "frames": n, "resolution": "1280x720",
               "target_bitrate_bps": br, "limit_bytes": int(max_mb * 1e6)}
        _p(f"[rv7] preview {size:,} B at {br} bps ({n} frames)")
        if size <= max_mb * 1e6:
            return out
    raise SystemExit(f"[rv7] preview still {out.get('bytes')} B > {max_mb} MB after three passes")


# ============================================================================================= #
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", choices=("forward", "render"), required=True)
    for k, v in DEFAULTS.items():
        ap.add_argument("--" + k, default=v)
    ap.add_argument("--expect-step", type=int, default=EXPECT["step"])
    ap.add_argument("--expect-ckpt-md5", default=EXPECT["ckpt_md5"])
    ap.add_argument("--seed", type=int, default=0, help="inference seed, set before EVERY window")
    ap.add_argument("--smoke", action="store_true", help="1 clip x --smoke-windows; outputs under */smoke")
    ap.add_argument("--smoke-windows", type=int, default=6)
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--open-s", type=float, default=10.0)
    ap.add_argument("--title-s", type=float, default=3.5)
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--preset", default="slow")
    ap.add_argument("--preview-mb", type=float, default=15.0)
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:                                           # noqa: BLE001
        pass
    return stage_forward(a) if a.stage == "forward" else stage_render(a)


if __name__ == "__main__":
    sys.exit(main())
