"""PH0 pilot runner — per-clip VLM/ego-algorithmic/SAM/Alpamayo labeling pipeline.

Implements the PH0 mini-pilot per:
  * `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/
    2026-08-07-hierarchical-wm-redesign/VLM_STRATEGIC_LABELING.md` (two-pass
    protocol, fusion gate, output schema v0),
  * same dir `HIERARCHY_VOCABULARY.md` (vocabulary, constraint slots, and the
    IMAGE-COORDINATE RULE: every semantic claim carries bbox [x0,y0,x1,y1] +
    frame index, mask ref where available; ungrounded claims are `disputed`),
  * same dir `PREREG_PH0_VLM.md` (measured quantities; gates are NOT changed
    here — this file only produces the rows the gates grade).

Four engines per clip:
  ENGINE A (algorithmic, deterministic): hindsight geometric summary from ego
    poses ([T, 4] = x, y, yaw, v @ 10 Hz, the repo episode contract). Reuses
    `scripts/refb_labels.py` (route_from_future_v3 / latmaneuver / lonmode /
    path_curvature) — imported, never reimplemented. Pluggable loader; when no
    ego data is found the record carries `engine_a: null` and the run continues.
  ENGINE B (VLM): two-pass protocol — pass 1 STRICT-JSON extract with REQUIRED
    image bboxes + frame indices, pass 2 self-verify CONFIRM/RETRACT against
    Engine A. transformers AutoProcessor/AutoModelForCausalLM, video input via
    the chat template (fps 2, 448 px). Robust JSON extraction, one retry on
    parse failure (counted: pass-1-valid-before-retry is the prereg G2 input).
  ENGINE D (Alpamayo): joins the clip's rows from records.parquet (all 5 task
    outputs incl. meta-actions and captions) into the record.
  ENGINE C (SAM, optional): if --sam is given, SAM2 segmentation on ~8
    keyframes prompted by the VLM's agent/sign boxes; masks stored as
    uncompressed RLE + bbox. Skips gracefully (with a reason) if absent.
  FUSION: deterministic gate — VLM strategic claims must be consistent with
    Engine A geometry when Engine A exists; every action is tagged
    pass|disputed; ungrounded claims (missing bbox/frame) are auto-disputed.

Outputs: one JSON per clip (schema-versioned, `_provenance` with model id +
prompt hash) + `pilot_summary.json` (per-clip status, wall s/clip, VRAM peak).

Pod-side notes (GPU): HF auth is read by huggingface_hub from
`/root/.cache/huggingface/token` automatically — this script never touches,
prints, or forwards tokens. `PYTHONPATH=/workspace/TanitAD/stack` required.
All torch/transformers/sam2/cv2 imports are lazy so CPU tests import this
module without GPU deps (only numpy + torch-for-refb_labels at call time).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

SCHEMA_VERSION = "ph0-v0.1"
VIDEO_FPS = 2.0                    # prompt video sampling rate (spec: ~2 fps)
VIDEO_PX = 448                     # frame long-side pixels (spec: 448 px)
PAST_S = 8.0                       # PAST clip: t0-8 s -> t0
FUTURE_S = 12.0                    # FUTURE clip: t0 -> t0+12 s
POSE_HZ = 10.0                     # episode pose contract
SAM_KEYFRAMES = 8

# ---- vocabulary (mirrors HIERARCHY_VOCABULARY.md; lowercase in the schema) ---
GOAL_KINDS = ("keep_corridor", "lane_target", "exit_left", "exit_right",
              "turn_left", "turn_right", "straight_through", "route_to",
              "stop_at", "follow_main_road", "none_abstain")
ACTION_VERBS = ("prepare_lane_change", "hold_corridor", "reduce_to",
                "prepare_exit", "prepare_stop", "resume_cruise")
CONSTRAINT_SLOTS = ("within_m", "by_time_s", "at_arc_m", "hold_for_s")
SIGN_KINDS = ("light", "speed", "nav", "stop", "other")
ACTION_SOURCES = ("path", "signage", "vlm-fused")

# =============================================================================
# small pure utils
# =============================================================================

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_json(text: str):
    """Robustly pull the first JSON object out of model output.

    Handles ```json fences, leading prose, and trailing text by scanning for
    the first balanced {...} block (string/escape aware). Returns the parsed
    object or raises ValueError."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty model output")
    m = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            text = m.group(1)          # fall through to balanced-scan on body
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    raise ValueError("no parseable JSON object in model output")


def rle_encode(mask: np.ndarray) -> dict:
    """Binary mask [H, W] -> uncompressed COCO-style RLE (column-major counts,
    starting with the run of zeros)."""
    m = np.asarray(mask, dtype=bool)
    flat = m.flatten(order="F").astype(np.int8)
    diff = np.flatnonzero(np.diff(flat))
    idx = np.concatenate([[0], diff + 1, [flat.size]])
    counts = np.diff(idx).tolist()
    if flat.size and flat[0] == 1:            # RLE starts with a zero-run
        counts = [0] + counts
    return {"size": [int(m.shape[0]), int(m.shape[1])], "counts": counts}


def rle_decode(rle: dict) -> np.ndarray:
    h, w = rle["size"]
    flat = np.zeros(h * w, dtype=bool)
    pos, val = 0, False
    for c in rle["counts"]:
        if val:
            flat[pos:pos + c] = True
        pos += c
        val = not val
    return flat.reshape((h, w), order="F")


def mask_bbox(mask: np.ndarray):
    ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    if ys.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _valid_bbox(b) -> bool:
    return (isinstance(b, (list, tuple)) and len(b) == 4
            and all(isinstance(v, (int, float)) and np.isfinite(v) for v in b)
            and b[2] > b[0] and b[3] > b[1])


def is_grounded(item: dict) -> bool:
    """Image-coordinate rule (HIERARCHY_VOCABULARY.md): a semantic claim is
    grounded iff it carries a valid bbox [x0,y0,x1,y1] AND a frame index."""
    return _valid_bbox(item.get("bbox")) and isinstance(
        item.get("frame_idx"), int) and item["frame_idx"] >= 0

# =============================================================================
# schema validation (pure; the prereg G2 instrument)
# =============================================================================

def validate_record(rec: dict) -> list[str]:
    """Validate a per-clip record against schema ph0-v0.1. Returns a list of
    error strings (empty = valid). Grounding is NOT a schema error — the
    fusion gate disputes ungrounded claims instead of dropping the row."""
    errs: list[str] = []

    def need(obj, key, typ, where):
        v = obj.get(key) if isinstance(obj, dict) else None
        if v is None or not isinstance(v, typ):
            name = (typ.__name__ if isinstance(typ, type)
                    else "/".join(t.__name__ for t in typ))
            errs.append(f"{where}.{key}: missing or not {name}")
            return None
        return v

    if not isinstance(rec, dict):
        return ["record is not an object"]
    if rec.get("schema_version") != SCHEMA_VERSION:
        errs.append(f"schema_version != {SCHEMA_VERSION}")
    need(rec, "clip_id", str, "record")
    scenario = need(rec, "scenario", dict, "record")
    if scenario is not None:
        for k in ("illumination", "weather", "daynight", "ego_behaviour"):
            need(scenario, k, str, "scenario")
        road = need(scenario, "road", dict, "scenario")
        if road is not None:
            need(road, "type", str, "scenario.road")
        agents = need(scenario, "agents", list, "scenario")
        for i, ag in enumerate(agents or []):
            if not isinstance(ag, dict):
                errs.append(f"scenario.agents[{i}]: not an object")
                continue
            need(ag, "class", str, f"scenario.agents[{i}]")
    domain = need(rec, "domain", dict, "record")
    if domain is not None:
        need(domain, "class", str, "domain")
        need(domain, "confidence", (int, float), "domain")
    signs = need(rec, "signs", list, "record")
    for i, s in enumerate(signs or []):
        if not isinstance(s, dict):
            errs.append(f"signs[{i}]: not an object")
            continue
        if s.get("kind") not in SIGN_KINDS:
            errs.append(f"signs[{i}].kind: {s.get('kind')!r} not in "
                        f"{SIGN_KINDS}")
        if not isinstance(s.get("applies_to_ego"), bool):
            errs.append(f"signs[{i}].applies_to_ego: missing bool")
    strat = need(rec, "strategic", dict, "record")
    if strat is not None:
        goal = need(strat, "goal", dict, "strategic")
        if goal is not None:
            if goal.get("kind") not in GOAL_KINDS:
                errs.append(f"strategic.goal.kind: {goal.get('kind')!r} not "
                            f"in vocabulary")
            if goal.get("source") not in ACTION_SOURCES:
                errs.append("strategic.goal.source: missing or not in "
                            f"{ACTION_SOURCES}")
        actions = need(strat, "actions", list, "strategic")
        for i, a in enumerate(actions or []):
            if not isinstance(a, dict):
                errs.append(f"strategic.actions[{i}]: not an object")
                continue
            need(a, "verb", str, f"strategic.actions[{i}]")
            cons = a.get("constraints")
            if cons is not None:
                if not isinstance(cons, dict):
                    errs.append(f"strategic.actions[{i}].constraints: "
                                "not an object")
                else:
                    for k in cons:
                        if k not in CONSTRAINT_SLOTS:
                            errs.append(f"strategic.actions[{i}].constraints."
                                        f"{k}: unknown slot")
            gc = a.get("geometric_consistency")
            if gc is not None and gc not in ("pass", "disputed"):
                errs.append(f"strategic.actions[{i}].geometric_consistency: "
                            f"{gc!r} not pass|disputed")
    prov = need(rec, "_provenance", dict, "record")
    if prov is not None:
        need(prov, "model_id", str, "_provenance")
        need(prov, "prompt_hash", str, "_provenance")
    return errs

# =============================================================================
# ENGINE A — algorithmic hindsight geometry (imports refb_labels machinery)
# =============================================================================

def load_ego_poses(clip_id: str, ego_root: str | None):
    """Pluggable ego loader. Tries, in order:
      {ego_root}/{clip_id}.npz  (key 'poses' or first array, [T, 4])
      {ego_root}/{clip_id}.npy  ([T, 4])
      {ego_root}/{clip_id}.parquet (columns x, y, yaw, v)
    Returns a torch.FloatTensor [T, 4] (x, y, yaw, v @ 10 Hz) or None.
    Absence is NOT an error: the caller emits engine_a: null and continues."""
    import torch
    if not ego_root:
        return None
    root = Path(ego_root)
    npz = root / f"{clip_id}.npz"
    if npz.exists():
        d = np.load(npz)
        arr = d["poses"] if "poses" in d.files else d[d.files[0]]
        return torch.as_tensor(np.asarray(arr), dtype=torch.float32)
    npy = root / f"{clip_id}.npy"
    if npy.exists():
        return torch.as_tensor(np.load(npy), dtype=torch.float32)
    pq = root / f"{clip_id}.parquet"
    if pq.exists():
        import pandas as pd
        df = pd.read_parquet(pq)
        cols = ["x", "y", "yaw", "v"]
        if all(c in df.columns for c in cols):
            return torch.as_tensor(df[cols].to_numpy(), dtype=torch.float32)
    return None


def _dedup_events(events: list[dict]) -> list[dict]:
    out = []
    for ev in events:
        if out and out[-1]["token"] == ev["token"]:
            out[-1]["t_end_s"] = ev["t_end_s"]
        else:
            out.append(dict(ev))
    return out


def engine_a_summary(poses, t0_idx: int, dt: float = 1.0 / POSE_HZ,
                     stride: int = 10) -> dict:
    """Hindsight geometric summary from ego poses [T, 4] (torch tensor).

    Reuses refb_labels (E4.1/E7.1 corridor machinery): integrated ego-frame
    path polyline (frame of pose t0), curvature-relative route/turn events
    (route_from_future_v3), lane-change displacement events (latmaneuver),
    speed-profile events (lonmode). All times are seconds relative to t0;
    arc distances are metres along the realized path from t0."""
    import torch
    import refb_labels as rl

    T = int(poses.shape[0])
    t0_idx = max(0, min(t0_idx, T - 1))
    yaw0 = poses[t0_idx, 2]
    pts = rl.ego_frame(poses[:, :2] - poses[t0_idx, :2],
                       yaw0.expand(T))                       # [T, 2] ego@t0
    seg = (poses[1:, :2] - poses[:-1, :2]).norm(dim=-1)
    arc_from_t0 = torch.zeros(T)
    arc_from_t0[t0_idx + 1:] = torch.cumsum(seg[t0_idx:], 0)
    kappa = rl.path_curvature(poses)

    route = rl.route_from_future_v3(poses, t0_idx)
    lat_events, lon_events = [], []
    for t in range(t0_idx, T - 1, stride):
        rel = (t - t0_idx) * dt
        lat = rl.latmaneuver_from_future(poses, t)
        if lat["valid"] and lat["active"]:
            # direction-consistency filter: a window that STARTS inside a
            # lateral maneuver has a tilted start yaw and latmaneuver then
            # mints a spurious opposite-direction event as the path
            # straightens. Require the t0-frame lateral displacement over the
            # window to agree with the token's direction (MEASURED on the
            # synthetic left-LC track in tests/test_ph0_pilot.py: without
            # this, the tail windows mint lc_right on a pure-left shift).
            t_end = min(t + rl.LAT_HORIZON_STEPS, T - 1)
            d_lat = float(pts[t_end, 1] - pts[t, 1])
            tok = lat["token"]
            if tok in ("lc_left", "nudge_left") and \
                    d_lat < rl.LANE_HALF_M / 2.0:
                lat = dict(lat, active=False)
            elif tok in ("lc_right", "nudge_right") and \
                    d_lat > -rl.LANE_HALF_M / 2.0:
                lat = dict(lat, active=False)
        if lat["valid"] and lat["active"]:
            lat_events.append({"token": lat["token"], "t_start_s": rel,
                               "t_end_s": rel + stride * dt,
                               "lat_m": round(lat["lat_m"], 2),
                               "arc_from_t0_m":
                                   round(float(arc_from_t0[t]), 1)})
        lon = rl.lonmode_from_future(poses, t)
        if lon["valid"] and lon["active"]:
            lon_events.append({"token": lon["token"], "t_start_s": rel,
                               "t_end_s": rel + stride * dt,
                               "stop_dist_m": lon["stop_dist_m"],
                               "dv": round(lon["dv"], 2),
                               "arc_from_t0_m":
                                   round(float(arc_from_t0[t]), 1)})
    v = poses[:, 3]
    fut_v = v[t0_idx:]
    summary = {
        "t0_idx": t0_idx,
        "duration_s": round((T - 1) * dt, 2),
        "polyline_xy": [[round(float(x), 2), round(float(y), 2)]
                        for x, y in pts.tolist()],
        "route": {
            "token": route["token"], "token_valid": bool(route["token_valid"]),
            "reason": route["reason"], "dist_m": route["dist_m"],
            "dist_band": route["dist_band"],
            "maneuver_dyaw_rad": round(float(route["maneuver_dyaw"]), 3),
            "graded_route": round(float(route["graded_route"]), 3),
            "arc_m": round(float(route["arc_m"]), 1),
        },
        "lane_change_events": _dedup_events(lat_events),
        "speed_events": _dedup_events(lon_events),
        "speed_profile": {
            "v_t0_ms": round(float(v[t0_idx]), 2),
            "v_min_future_ms": round(float(fut_v.min()), 2),
            "v_max_future_ms": round(float(fut_v.max()), 2),
            "net_dv_ms": round(float(fut_v[-1] - v[t0_idx]), 2),
            "stops": bool((fut_v < rl.STOP_V_MS).any()),
        },
        "peak_kappa_per_m": round(float(kappa.abs().max()), 4)
        if kappa.numel() else 0.0,
    }
    return summary


def engine_a_for_prompt(engine_a: dict) -> dict:
    """The prompt-embeddable view: everything except the polyline."""
    return {k: v for k, v in engine_a.items() if k != "polyline_xy"}

# =============================================================================
# FUSION — the deterministic gate (pure; unit-tested on CPU)
# =============================================================================

_LC_TOKENS = {"left": ("lc_left",), "right": ("lc_right",)}
_EXIT_TOKENS = {"left": ("exit_left", "turn_left"),
                "right": ("exit_right", "turn_right")}
_JUNCTION_TOKENS = ("turn_left", "turn_right", "exit_left", "exit_right",
                    "roundabout", "u_turn")
_GOAL_EXPECT = {  # goal kind -> route tokens that confirm it geometrically
    "turn_left": ("turn_left", "roundabout", "u_turn"),
    "turn_right": ("turn_right",),
    "exit_left": ("exit_left", "turn_left"),
    "exit_right": ("exit_right", "turn_right"),
}


def _within(ev_arc_m, within_m) -> bool:
    if within_m is None:
        return True
    return ev_arc_m is not None and 0.0 <= ev_arc_m <= float(within_m)


def _sign_ok(rec: dict, idx) -> bool:
    signs = rec.get("signs") or []
    if not isinstance(idx, int) or not 0 <= idx < len(signs):
        return False
    s = signs[idx]
    return (is_grounded(s) and bool(s.get("applies_to_ego"))
            and not s.get("retracted", False))


def _check_action_geometry(action: dict, ea: dict, rec: dict) -> list[str]:
    """Return the list of dispute reasons for one strategic action against
    Engine A geometry (empty = geometrically consistent)."""
    verb = str(action.get("verb", "")).lower()
    direction = str(action.get("direction", "")).lower() or None
    cons = action.get("constraints") or {}
    within_m = cons.get("within_m", cons.get("at_arc_m"))
    reasons: list[str] = []
    lc = ea.get("lane_change_events", [])
    lon = ea.get("speed_events", [])
    route = ea.get("route", {})
    sp = ea.get("speed_profile", {})

    if verb == "prepare_lane_change":
        want = _LC_TOKENS.get(direction, ("lc_left", "lc_right"))
        hits = [e for e in lc if e["token"] in want
                and _within(e.get("arc_from_t0_m"), within_m)]
        if not hits:
            reasons.append(f"no {direction or 'any'}-lane-change event in "
                           "hindsight path within envelope")
    elif verb == "prepare_exit":
        want = _EXIT_TOKENS.get(direction,
                                ("exit_left", "exit_right",
                                 "turn_left", "turn_right"))
        ok = route.get("token") in want and route.get("token_valid")
        if ok and within_m is not None and route.get("dist_m") is not None:
            ok = _within(route["dist_m"], within_m)
        if not ok:
            reasons.append(f"route token {route.get('token')!r} does not "
                           f"confirm exit {direction or ''}".strip())
    elif verb == "prepare_stop":
        stop_ev = [e for e in lon if e["token"] in
                   ("stop_at_point", "hold_stop")
                   and _within(e.get("arc_from_t0_m"), within_m)]
        signage = _sign_evidence_ok(action, rec, kinds=("light", "stop"))
        if not stop_ev and not sp.get("stops") and not signage:
            reasons.append("no stop event in hindsight speed profile and no "
                           "grounded ego-applying stop/light signage")
    elif verb == "reduce_to":
        decel = ([e for e in lon if e["token"] in
                  ("stop_at_point", "coast", "creep")]
                 or sp.get("net_dv_ms", 0.0) <= -1.0)
        signage = _sign_evidence_ok(action, rec, kinds=("speed",))
        if not decel and not signage:
            reasons.append("no deceleration in hindsight speed profile and "
                           "no grounded ego-applying speed signage")
    elif verb in ("hold_corridor", "resume_cruise"):
        tok = route.get("token")
        if (route.get("token_valid") and tok in _JUNCTION_TOKENS
                and _within(route.get("dist_m"), within_m)):
            reasons.append(f"junction-scale route event {tok!r} inside the "
                           "hold/cruise envelope")
        if verb == "resume_cruise" and any(
                e["token"] in ("stop_at_point", "hold_stop") for e in lon):
            reasons.append("stop event ahead contradicts resume_cruise")
    else:
        reasons.append(f"verb {verb!r} not in the strategic action vocabulary")
    return reasons


def _sign_evidence_ok(action: dict, rec: dict, kinds) -> bool:
    ev = action.get("evidence") or {}
    idx = ev.get("sign_idx")
    if not _sign_ok(rec, idx):
        return False
    return (rec["signs"][idx].get("kind") in kinds)


def fusion_gate(rec: dict, engine_a: dict | None) -> dict:
    """The deterministic fusion gate (VLM_STRATEGIC_LABELING.md §3).

    Mutates and returns `rec`:
      * every sign/agent claim gets `grounded` (image-coordinate rule); an
        ungrounded claim is auto-`disputed`;
      * every strategic action gets `geometric_consistency: pass|disputed` +
        `fusion_reasons`; with Engine A absent, path-sourced claims cannot be
        verified and are disputed (reason `engine_a_absent`), signage-sourced
        claims stand on their grounded+confirmed sign evidence alone;
      * `strategic.goal` gets `fusion: pass|disputed` (route_to requires a
        grounded, ego-applying, non-retracted nav sign with OCR text);
      * a record-level `fusion` block summarises verdicts (the prereg G3
        numerator/denominator).
    """
    disputed_items: list[dict] = []
    for kind, items in (("sign", rec.get("signs") or []),
                        ("agent", (rec.get("scenario") or {})
                         .get("agents") or [])):
        for i, it in enumerate(items):
            if not isinstance(it, dict):
                continue
            g = is_grounded(it)
            it["grounded"] = g
            if not g:
                it["disputed"] = True
                disputed_items.append(
                    {"kind": kind, "index": i,
                     "reason": "ungrounded (missing bbox/frame_idx)"})

    strat = rec.get("strategic") or {}
    goal = strat.get("goal") or {}
    goal_reasons: list[str] = []
    gkind = str(goal.get("kind", "")).lower()
    if gkind == "route_to":
        ev = goal.get("evidence") or {}
        idx = ev.get("sign_idx")
        ok = (_sign_ok(rec, idx)
              and rec["signs"][idx].get("kind") == "nav"
              and bool(str(rec["signs"][idx].get("text_ocr") or "").strip()))
        if not ok:
            goal_reasons.append("route_to without a grounded ego-applying nav "
                                "sign with OCR text")
    elif gkind in _GOAL_EXPECT and engine_a is not None:
        route = engine_a.get("route", {})
        if route.get("token_valid") and \
                route.get("token") not in _GOAL_EXPECT[gkind] and \
                route.get("token") in _JUNCTION_TOKENS + ("follow",):
            goal_reasons.append(
                f"goal {gkind!r} contradicts hindsight route token "
                f"{route.get('token')!r}")
    goal["fusion"] = "disputed" if goal_reasons else "pass"
    goal["fusion_reasons"] = goal_reasons

    n_pass = n_disputed = 0
    for a in strat.get("actions") or []:
        if not isinstance(a, dict):
            continue
        reasons: list[str] = []
        src = a.get("source")
        if src not in ACTION_SOURCES:
            reasons.append(f"source {src!r} not in {ACTION_SOURCES}")
        if src == "signage" and not _sign_evidence_ok(
                a, rec, kinds=SIGN_KINDS):
            reasons.append("signage-sourced action without grounded "
                           "ego-applying sign evidence")
        if engine_a is not None:
            reasons += _check_action_geometry(a, engine_a, rec)
        elif src != "signage":
            reasons.append("engine_a_absent: path-derived claim cannot be "
                           "geometrically verified")
        a["geometric_consistency"] = "disputed" if reasons else "pass"
        a["fusion_reasons"] = reasons
        if reasons:
            n_disputed += 1
        else:
            n_pass += 1

    rec["fusion"] = {
        "engine_a_available": engine_a is not None,
        "actions_pass": n_pass,
        "actions_disputed": n_disputed,
        "goal_verdict": goal.get("fusion", "pass"),
        "ungrounded_disputed": disputed_items,
    }
    return rec

# =============================================================================
# ENGINE D — Alpamayo records join
# =============================================================================

def join_alpamayo(records_df, clip_id: str) -> dict | None:
    """Join all of a clip's rows from the augmentation records.parquet
    (schema: DESIGN.md 2026-08-06 — one row per clip x task; raw_output +
    parsed for the 5 tasks incl. meta_action). Returns None when the clip has
    no rows (reported, never silently dropped)."""
    if records_df is None:
        return None
    rows = records_df[records_df["clip_id"] == clip_id]
    if len(rows) == 0:
        return None
    out: dict = {"n_rows": int(len(rows)), "tasks": {}}
    for _, r in rows.iterrows():
        task = str(r.get("task", "unknown"))
        entry = {}
        for col in ("parsed", "raw_output", "vqa_qid", "model_rev", "quant",
                    "t0_frame_idx", "latency_s", "sample_idx"):
            if col in rows.columns and r.get(col) is not None:
                v = r[col]
                if isinstance(v, np.generic):
                    v = v.item()
                if isinstance(v, np.ndarray):
                    v = v.tolist()
                entry[col] = v
        out["tasks"].setdefault(task, []).append(entry)
    ma = out["tasks"].get("meta_action")
    if ma:
        out["meta_actions"] = [e.get("parsed") or e.get("raw_output")
                               for e in ma]
    return out

# =============================================================================
# video frame sampling (lazy cv2 / PyAV)
# =============================================================================

def sample_clip_frames(video_path: str, t0_s: float, fps: float = VIDEO_FPS,
                       past_s: float = PAST_S, future_s: float = FUTURE_S,
                       px: int = VIDEO_PX):
    """Sample PAST (t0-past_s -> t0) + FUTURE (t0 -> t0+future_s) frames at
    `fps`, long side resized to `px`. Returns (frames [list of HxWx3 uint8],
    times_rel_s [list], n_past). Uses cv2 if available, else PyAV."""
    times = []
    t = -past_s
    while t < future_s - 1e-9:
        times.append(t)
        t += 1.0 / fps
    n_past = sum(1 for t in times if t < 0)
    want = [t0_s + t for t in times]
    frames = _read_frames_at(video_path, want, px)
    return frames, times, n_past


def _resize_long(img: np.ndarray, px: int) -> np.ndarray:
    from PIL import Image
    h, w = img.shape[:2]
    sc = px / max(h, w)
    if abs(sc - 1.0) < 1e-6:
        return img
    return np.asarray(Image.fromarray(img).resize(
        (max(1, int(round(w * sc))), max(1, int(round(h * sc)))),
        Image.BILINEAR))


def _decoder_threads() -> int:
    """Threads to allow the video decoder — deliberately TINY.

    ⛔ MEASURED on pod4 2026-08-12: every VLM arm failed with
    ``BlockingIOError: [Errno 11] ... [swscaler] Failed initializing scaling
    graph (Resource temporarily unavailable)``. EAGAIN there is a THREAD
    CREATION failure inside libswscale, not a corrupt video — the clips decode
    fine on their own. The cause is the same "a probe reporting the wrong
    scope" trap CLAUDE.md records for ``df``/``free``/cgroup counters:
    ``nproc`` reports the HOST's 96 CPUs, so ffmpeg (and torch, at ~113 threads
    per process) size their pools to 96 and collide with the container's real
    allowance. This looked exactly like "the model is text-only / broken",
    which is how it reached a PI decision request as a model-availability
    problem when it was an environment one.

    ⇒ Pin it to 1. Video decode here is a handful of frames at 2 fps; threads
    buy nothing and cost the whole run."""
    return int(os.environ.get("PH0_DECODER_THREADS", "1"))


def _read_frames_at(video_path: str, times_s: list[float],
                    px: int) -> list[np.ndarray]:
    try:
        import cv2
        # same reason as _decoder_threads(): cv2 auto-sizes to the host CPU
        # count and its resize/convert pools hit the same ceiling.
        cv2.setNumThreads(_decoder_threads())
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"cv2 cannot open {video_path}")
        out = []
        for ts in times_s:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, ts * 1000.0))
            ok, bgr = cap.read()
            if not ok:
                bgr = np.zeros((VIDEO_PX, VIDEO_PX, 3), np.uint8)
            out.append(_resize_long(bgr[:, :, ::-1].copy(), px))
        cap.release()
        return out
    except ImportError:
        pass
    import av                                        # stack [real] extra
    container = av.open(video_path)
    stream = container.streams.video[0]
    # see _decoder_threads(): this is THE fix for the swscaler EAGAIN that made
    # every VLM arm look unavailable. Set on the stream before any decode.
    stream.thread_count = _decoder_threads()
    stream.thread_type = "NONE"
    decoded, want = [], sorted(set(max(0.0, t) for t in times_s))
    frames_by_t: dict[float, np.ndarray] = {}
    wi = 0
    for frame in container.decode(stream):
        ft = float(frame.pts * stream.time_base) if frame.pts is not None \
            else len(decoded) / (stream.average_rate or 30)
        while wi < len(want) and ft >= want[wi] - 1e-3:
            frames_by_t[want[wi]] = _resize_long(
                frame.to_ndarray(format="rgb24"), px)
            wi += 1
        if wi >= len(want):
            break
    container.close()
    fallback = next(iter(frames_by_t.values()),
                    np.zeros((VIDEO_PX, VIDEO_PX, 3), np.uint8))
    out = []
    for ts in times_s:
        key = min(want, key=lambda w_: abs(w_ - max(0.0, ts)))
        out.append(frames_by_t.get(key, fallback))
    return out

# =============================================================================
# ENGINE B — VLM two-pass protocol (lazy transformers; GPU pod-side)
# =============================================================================

_SCHEMA_HINT = json.dumps({
    "scenario": {"illumination": "<day|dusk|night|...>", "weather": "<str>",
                 "daynight": "<day|night>",
                 "road": {"type": "<highway|urban|rural|...>",
                          "lanes_visible": "<int>", "lane_ego": "<int>"},
                 "agents": [{"class": "<car|pedestrian|...>",
                             "position_rel": "<str>", "behaviour": "<str>",
                             "bbox": [0, 0, 0, 0], "frame_idx": 0}],
                 "ego_behaviour": "<str>"},
    "domain": {"class": "<highway|urban|roundabout|intersection|rural>",
               "confidence": 0.0},
    "signs": [{"kind": "<light|speed|nav|stop|other>", "state": "<str|null>",
               "text_ocr": "<verbatim text or empty>",
               "applies_to_ego": True, "bbox": [0, 0, 0, 0], "frame_idx": 0}],
    "strategic": {
        "goal": {"kind": "<" + "|".join(GOAL_KINDS) + ">",
                 "target_text": "<str|null>",
                 "source": "<path|signage|vlm-fused>", "confidence": 0.0,
                 "evidence": {"sign_idx": None, "frame_idx": None}},
        "actions": [{"verb": "<" + "|".join(ACTION_VERBS) + ">",
                     "direction": "<left|right|null>",
                     "args": {"v_target_ms": None},
                     "constraints": {"within_m": None, "by_time_s": None,
                                     "at_arc_m": None, "hold_for_s": None},
                     "reason": "<str>",
                     "source": "<path|signage|vlm-fused>",
                     "confidence": 0.0,
                     "evidence": {"sign_idx": None, "frame_idx": None}}]},
}, indent=1)

PROMPT_PASS1 = """You are a precise autonomous-driving scene labeler.
You are shown ONE video: frames 0..{n_past_minus1} are the PAST 8 s (2 fps) \
up to the decision time t0; frames {n_past}..{n_last} are the FUTURE 12 s \
(2 fps) after t0. The future is HINDSIGHT evidence: use it to determine what \
the ego driver actually did.

A deterministic geometric analysis of the ego trajectory (Engine A) is \
provided below as ground context; your strategic claims will be checked \
against it, so do not contradict the geometry:
ENGINE_A = {engine_a_json}

Extract, as ONE strict JSON object and NOTHING else (no prose, no markdown):
{schema_hint}

BINDING RULES:
1. EVERY sign and EVERY agent claim MUST carry "bbox": [x0,y0,x1,y1] in pixel
   coordinates of the sampled frame it appears in, and "frame_idx" (the index
   in THIS video, 0..{n_last}). A claim without both will be marked disputed.
2. signs[].text_ocr is the VERBATIM legible text (city names, numbers). If no
   text is legible, use "". NEVER invent text.
3. strategic.goal.kind = "route_to" is allowed ONLY with a nav sign you
   actually read (evidence.sign_idx). With no navigation route evidence, use
   "follow_main_road" (the default) or a corridor/lane-level goal. Abstaining
   ("none_abstain") is better than guessing.
4. strategic actions use ONLY the listed verbs, with optional constraint
   slots within_m / by_time_s / at_arc_m / hold_for_s (units m, s).
5. Every strategic goal/action carries "source": "path" (derived from the
   trajectory geometry), "signage" (from a sign you read), or "vlm-fused".
6. Confidence values are in [0, 1]. Output the JSON object only."""

PROMPT_PASS2 = """You previously made the claims below about this driving \
clip. The deterministic geometric analysis of the actual ego trajectory is \
ENGINE_A. Re-examine the video and verify each claim.

CLAIMS = {claims_json}
ENGINE_A = {engine_a_json}

For every claim id, answer CONFIRM (evidence is visible / consistent) or \
RETRACT (not clearly visible, invented, or contradicted by the geometry). \
Retract any sign whose text you cannot actually read in the frames.
Reply with ONE strict JSON object and NOTHING else:
{{"verdicts": [{{"claim_id": "<id>", "verdict": "CONFIRM|RETRACT", \
"reason": "<short>"}}]}}"""


def prompt_hash() -> str:
    return sha256_text(PROMPT_PASS1 + "\n---\n" + PROMPT_PASS2 + "\n---\n"
                       + _SCHEMA_HINT + "\n---\n" + SCHEMA_VERSION)


def enumerate_claims(rec: dict) -> list[dict]:
    """Flatten a pass-1 record into id'd claims for the pass-2 verify."""
    claims = []
    for i, s in enumerate(rec.get("signs") or []):
        claims.append({"claim_id": f"sign_{i}", "claim": s})
    for i, ag in enumerate((rec.get("scenario") or {}).get("agents") or []):
        claims.append({"claim_id": f"agent_{i}", "claim": ag})
    strat = rec.get("strategic") or {}
    if strat.get("goal"):
        claims.append({"claim_id": "goal", "claim": strat["goal"]})
    for i, a in enumerate(strat.get("actions") or []):
        claims.append({"claim_id": f"action_{i}", "claim": a})
    return claims


def apply_verdicts(rec: dict, verdicts: list[dict]) -> dict:
    """Apply pass-2 CONFIRM/RETRACT: retracted signs/agents are flagged
    (kept in place, `retracted: true` — banked, not dropped); retracted
    actions are removed to `strategic.retracted_actions`; a retracted goal
    degrades to follow_main_road/path (the vocabulary's no-route default)."""
    vmap = {str(v.get("claim_id")): str(v.get("verdict", "")).upper()
            for v in verdicts if isinstance(v, dict)}

    def _is_retracted(cid):
        return vmap.get(cid) == "RETRACT"

    for i, s in enumerate(rec.get("signs") or []):
        if _is_retracted(f"sign_{i}"):
            s["retracted"] = True
    for i, ag in enumerate((rec.get("scenario") or {}).get("agents") or []):
        if _is_retracted(f"agent_{i}"):
            ag["retracted"] = True
    strat = rec.get("strategic") or {}
    if _is_retracted("goal") and strat.get("goal"):
        strat["retracted_goal"] = strat["goal"]
        strat["goal"] = {"kind": "follow_main_road", "target_text": None,
                         "source": "path", "confidence": 0.0,
                         "evidence": {}}
    kept, retracted = [], []
    for i, a in enumerate(strat.get("actions") or []):
        (retracted if _is_retracted(f"action_{i}") else kept).append(a)
    strat["actions"] = kept
    if retracted:
        strat["retracted_actions"] = retracted
    return rec


class NullVLM:
    """Engine B DISABLED — the other three engines still run.

    Exists because engine B's availability is an entirely separate question from
    whether engines A (ego geometry), C (SAM) and D (Alpamayo) and the fusion
    gate work, and on 2026-08-12 the VLM arms blocked the whole pilot for reasons
    that had nothing to do with the pipeline: `Qwen3.5-9B` is TEXT-only and
    rejected the video kwargs, `Qwen3.5-27B-FP8` needs 43.23 GiB of a 44.43 GiB
    card, and `gemma-4-31B-it-qat-w4a16-ct` loaded UNQUANTISED (38.83 GiB, then
    `KeyError: 'weight_packed'`) under transformers 5.15 + compressed-tensors
    0.18. Running the pilot without engine B isolates that failure instead of
    letting it masquerade as a pipeline failure.

    ⚠️ A record produced this way has `engine_b_disabled: true` in its provenance
    and EMPTY scenario/domain/signs/strategic blocks. It is a pipeline validation
    artifact and must NEVER be read as a vocabulary-extraction result."""

    model_id = "none (--no-vlm)"

    def chat_json(self, frames, prompt):                    # noqa: ARG002
        return {}, False


# Auto-classes tried IN ORDER for engine B. The first two are the
# vision-language classes; AutoModelForCausalLM is LAST and is text-only.
#
# ⛔ MEASURED on pod4 2026-08-12, and it is the real reason this arm looked
# "text-only" for a day: loading a VLM through AutoModelForCausalLM SUCCEEDS —
# the weights load, VRAM fills (16.68 GB for the 9B), nothing raises — and then
# `generate()` dies with
#   ValueError: The following `model_kwargs` are not used by the model:
#   ['mm_token_type_ids', 'pixel_values_videos', 'video_grid_thw']
# because the text-only class has no vision tower to consume them. 8/8 clips
# failed in 0.6 s each while the run exited 0 with no traceback. A checkpoint
# that is perfectly capable of video therefore presented as a model-availability
# problem. The auto-class is OURS to choose, so choose it correctly and RECORD
# which one was used.
VLM_AUTO_CLASSES = ("AutoModelForImageTextToText", "AutoModelForVision2Seq",
                    "AutoModelForCausalLM")


class VLMEngine:
    """Engine B wrapper: AutoProcessor + the first VLM auto-class that both
    imports and accepts the processor's video kwargs. Pod-side only (lazy
    heavy imports)."""

    def __init__(self, model_id: str, max_new_tokens: int = 2048):
        import torch
        import transformers
        from transformers import AutoProcessor
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True)
        self.model, self.auto_class, errs = None, None, []
        for name in VLM_AUTO_CLASSES:
            cls = getattr(transformers, name, None)
            if cls is None:
                errs.append(f"{name}: not in transformers "
                            f"{transformers.__version__}")
                continue
            try:
                self.model = cls.from_pretrained(
                    model_id, torch_dtype="auto", device_map="cuda:0",
                    trust_remote_code=True)
                self.auto_class = name
                break
            except Exception as e:                       # try the next class
                errs.append(f"{name}: {type(e).__name__}: {str(e)[:120]}")
        if self.model is None:
            raise RuntimeError(
                f"no usable auto-class for {model_id}; tried "
                f"{list(VLM_AUTO_CLASSES)} -> {errs}")
        if self.auto_class == "AutoModelForCausalLM":
            # not fatal (a text-only arm is a legitimate ablation) but it must
            # never be mistaken for a working video path again
            print(f"[ph0][WARN] {model_id} loaded through the TEXT-ONLY "
                  f"AutoModelForCausalLM — video kwargs will be rejected at "
                  f"generate(). Earlier classes failed: {errs}", flush=True)
        self.model.eval()
        self._torch = torch

    def chat(self, frames: list[np.ndarray], prompt: str) -> str:
        """One video+text turn through the arm's OFFICIAL chat template —
        the prereg's mandatory video-template check runs through this exact
        path."""
        torch = self._torch
        from PIL import Image
        pil = [Image.fromarray(f) for f in frames]
        messages = [{"role": "user", "content": [
            {"type": "video", "video": pil, "fps": VIDEO_FPS},
            {"type": "text", "text": prompt}]}]
        try:
            inputs = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt")
        except Exception:
            text = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False)
            inputs = self.processor(text=[text], videos=[pil],
                                    return_tensors="pt")
        inputs = {k: (v.to(self.model.device)
                      if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.inference_mode():
            out = self.model.generate(**inputs,
                                      max_new_tokens=self.max_new_tokens,
                                      do_sample=False)
        n_in = inputs["input_ids"].shape[1]
        return self.processor.batch_decode(out[:, n_in:],
                                           skip_special_tokens=True)[0]

    def chat_json(self, frames, prompt: str):
        """STRICT JSON with retry-once-on-parse-failure. Returns
        (obj, pass1_valid_before_retry)."""
        text = self.chat(frames, prompt)
        try:
            return extract_json(text), True
        except ValueError:
            retry = (prompt + "\n\nYour previous output was not valid JSON. "
                     "Output ONLY the JSON object, with no other text.")
            return extract_json(self.chat(frames, retry)), False

# =============================================================================
# ENGINE C — SAM2 segmentation on keyframes (optional, lazy)
# =============================================================================

def run_sam(sam_ckpt: str, frames: list[np.ndarray], rec: dict,
            n_keyframes: int = SAM_KEYFRAMES) -> dict | None:
    """SAM2 on ~n_keyframes evenly-spaced frames, prompted with the VLM's
    grounded agent/sign boxes (Engine C, HIERARCHY_VOCABULARY.md §0b). Masks
    stored as uncompressed RLE + bbox + a mask_ref written back onto the
    prompting claim. Returns None (with no side effects) when sam2 or the
    checkpoint is unavailable — Engine C is optional by design."""
    try:
        import torch
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        predictor = SAM2ImagePredictor.from_pretrained(sam_ckpt)
    except Exception as e:                                  # noqa: BLE001
        return {"skipped": True, "reason": f"sam2 unavailable: {e}"}
    idxs = np.unique(np.linspace(0, len(frames) - 1,
                                 min(n_keyframes, len(frames)),
                                 dtype=int)).tolist()
    prompts = []
    for kind, items in (("agent",
                         (rec.get("scenario") or {}).get("agents") or []),
                        ("sign", rec.get("signs") or [])):
        for i, it in enumerate(items):
            if isinstance(it, dict) and is_grounded(it) \
                    and not it.get("retracted"):
                prompts.append((kind, i, it))
    instances = []
    for fi in idxs:
        boxes = [(k, i, it) for (k, i, it) in prompts
                 if abs(int(it["frame_idx"]) - fi) <= 1]
        if not boxes:
            continue
        with torch.inference_mode():
            predictor.set_image(frames[fi])
            for kind, i, it in boxes:
                masks, scores, _ = predictor.predict(
                    box=np.asarray(it["bbox"], dtype=np.float32),
                    multimask_output=False)
                m = np.asarray(masks[0], dtype=bool)
                ref = f"sam_{kind}_{i}_f{fi}"
                instances.append({
                    "mask_ref": ref, "class_prompt": kind,
                    "source_index": i, "frame_idx": int(fi),
                    "score": float(scores[0]),
                    "bbox": mask_bbox(m) or list(it["bbox"]),
                    "rle": rle_encode(m)})
                it.setdefault("mask_ref", ref)
    return {"skipped": False, "checkpoint": sam_ckpt,
            "keyframes": [int(i) for i in idxs], "instances": instances}

# =============================================================================
# main loop
# =============================================================================

def _resolve_clip(entry, video_root: str | None):
    """A --clips entry is a clip id, a video path, or {clip_id, video}."""
    if isinstance(entry, dict):
        cid = str(entry.get("clip_id"))
        video = entry.get("video") or (
            str(Path(video_root) / f"{cid}.mp4") if video_root else None)
        return cid, video
    e = str(entry)
    if e.endswith((".mp4", ".mkv", ".avi")):
        return Path(e).stem, e
    return e, (str(Path(video_root) / f"{e}.mp4") if video_root else None)


def process_clip(clip_id: str, video_path: str | None, vlm, records_df,
                 args) -> dict:
    """Run all engines + fusion for one clip; returns the schema-v0 record."""
    import torch
    rec_meta: dict = {}

    # ---- ENGINE A -----------------------------------------------------------
    poses = load_ego_poses(clip_id, args.ego_root)
    t0_idx = int(round(args.t0_s * POSE_HZ))
    engine_a = engine_a_summary(poses, t0_idx) if poses is not None else None

    # ---- ENGINE B -----------------------------------------------------------
    frames, times, n_past = sample_clip_frames(
        video_path, t0_s=args.t0_s, fps=args.fps, px=args.px)
    ea_json = json.dumps(engine_a_for_prompt(engine_a)) if engine_a \
        else "null (no ego data for this clip)"
    p1 = PROMPT_PASS1.format(n_past_minus1=n_past - 1, n_past=n_past,
                             n_last=len(frames) - 1,
                             engine_a_json=ea_json,
                             schema_hint=_SCHEMA_HINT)
    extraction, pass1_valid = vlm.chat_json(frames, p1)
    rec: dict = {
        "schema_version": SCHEMA_VERSION,
        "clip_id": clip_id,
        "t0_us": int(args.t0_s * 1e6),
        "scenario": extraction.get("scenario") or {},
        "domain": extraction.get("domain") or {},
        "signs": extraction.get("signs") or [],
        "strategic": extraction.get("strategic") or {"goal": {}, "actions": []},
    }
    claims = enumerate_claims(rec)
    verdicts: list[dict] = []
    if claims:
        p2 = PROMPT_PASS2.format(claims_json=json.dumps(claims),
                                 engine_a_json=ea_json)
        try:
            v_obj, _ = vlm.chat_json(frames, p2)
            verdicts = v_obj.get("verdicts") or []
            rec = apply_verdicts(rec, verdicts)
        except ValueError as e:
            rec_meta["pass2_error"] = str(e)

    # ---- ENGINE C (optional) ------------------------------------------------
    rec["sam"] = run_sam(args.sam, frames, rec) if args.sam else None

    # ---- ENGINE D -----------------------------------------------------------
    rec["alpamayo"] = join_alpamayo(records_df, clip_id)

    # ---- FUSION -------------------------------------------------------------
    rec["engine_a"] = engine_a
    rec = fusion_gate(rec, engine_a)

    rec["_provenance"] = {
        "model_id": vlm.model_id,
        "prompt_hash": prompt_hash(),
        "schema_version": SCHEMA_VERSION,
        "pass1_valid_before_retry": pass1_valid,
        "pass2_verdicts": verdicts,
        "video_sampling": {"fps": args.fps, "px": args.px,
                           "t0_s": args.t0_s, "past_s": PAST_S,
                           "future_s": FUTURE_S, "n_past": n_past,
                           "n_frames": len(frames),
                           "video_path": video_path},
        "engine_a_available": engine_a is not None,
        **rec_meta,
    }
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clips", required=True,
                    help="json file: list of clip ids / video paths / "
                         "{clip_id, video} objects (ph0_clips.json works)")
    ap.add_argument("--records", default=None,
                    help="Alpamayo augmentation records.parquet (Engine D)")
    ap.add_argument("--no-vlm", action="store_true",
                    help="DISABLE engine B and run engines A/C/D + fusion only. "
                         "Records get engine_b_disabled:true and empty "
                         "scenario/domain/signs/strategic blocks — a pipeline "
                         "validation artifact, NEVER a vocabulary result.")
    ap.add_argument("--arm", default="Qwen/Qwen3.5-9B",
                    help="VLM model id (PH0 arm)")
    ap.add_argument("--sam", default=None,
                    help="optional SAM2 checkpoint id (Engine C); skipped "
                         "gracefully if sam2/checkpoint is unavailable")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--video-root", default=None,
                    help="root dir resolving clip_id -> {root}/{clip_id}.mp4")
    ap.add_argument("--ego-root", default=None,
                    help="root dir with per-clip ego poses "
                         "({clip_id}.npz|npy|parquet, [T,4] x,y,yaw,v @10Hz); "
                         "absent -> engine_a: null, run continues")
    ap.add_argument("--t0-s", type=float, default=PAST_S, dest="t0_s",
                    help="decision time t0 seconds into the clip")
    ap.add_argument("--fps", type=float, default=VIDEO_FPS)
    ap.add_argument("--px", type=int, default=VIDEO_PX)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    clip_spec = json.loads(Path(args.clips).read_text())
    if isinstance(clip_spec, dict):
        clip_spec = clip_spec.get("clips", [])

    records_df = None
    if args.records:
        import pandas as pd
        records_df = pd.read_parquet(args.records)

    import torch
    vlm = NullVLM() if args.no_vlm \
        else VLMEngine(args.arm, max_new_tokens=args.max_new_tokens)
    arm_tag = ("no-vlm" if args.no_vlm else args.arm).replace("/", "_")

    summary: dict = {"arm": vlm.model_id, "engine_b_disabled": bool(args.no_vlm),
                     "prompt_hash": prompt_hash(),
                     "schema_version": SCHEMA_VERSION, "clips": []}
    walls = []
    for entry in clip_spec:
        clip_id, video_path = _resolve_clip(entry, args.video_root)
        row = {"clip_id": clip_id}
        t_start = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        try:
            if video_path is None or not Path(video_path).exists():
                raise FileNotFoundError(f"no video for {clip_id!r} "
                                        f"(resolved {video_path!r})")
            rec = process_clip(clip_id, video_path, vlm, records_df, args)
            errs = validate_record(rec)
            rec["_provenance"]["schema_errors"] = errs
            out_path = out_dir / f"{clip_id}.{arm_tag}.json"
            out_path.write_text(json.dumps(rec, indent=1))
            row.update(
                status="ok", out=str(out_path), schema_valid=not errs,
                pass1_valid_before_retry=
                rec["_provenance"]["pass1_valid_before_retry"],
                engine_a=rec["engine_a"] is not None,
                alpamayo_rows=(rec["alpamayo"] or {}).get("n_rows", 0),
                actions_pass=rec["fusion"]["actions_pass"],
                actions_disputed=rec["fusion"]["actions_disputed"])
        except Exception as e:                              # noqa: BLE001
            row.update(status="error", error=f"{type(e).__name__}: {e}")
        row["wall_s"] = round(time.time() - t_start, 1)
        if torch.cuda.is_available():
            row["vram_peak_gb"] = round(
                torch.cuda.max_memory_allocated() / 2**30, 2)
        walls.append(row["wall_s"])
        summary["clips"].append(row)
        print(f"[ph0] {clip_id}: {row['status']} {row['wall_s']}s", flush=True)
        (out_dir / "pilot_summary.json").write_text(
            json.dumps(summary, indent=1))       # bank incrementally

    ok = [c for c in summary["clips"] if c["status"] == "ok"]
    summary["aggregate"] = {
        "n": len(summary["clips"]), "n_ok": len(ok),
        "n_schema_valid_pass1": sum(
            1 for c in ok if c.get("pass1_valid_before_retry")),
        "wall_s_median": float(np.median(walls)) if walls else None,
        "wall_s_p90": float(np.percentile(walls, 90)) if walls else None,
        "vram_peak_gb_max": max((c.get("vram_peak_gb", 0.0)
                                 for c in summary["clips"]), default=None),
    }
    (out_dir / "pilot_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"[ph0] done: {len(ok)}/{len(summary['clips'])} ok -> {out_dir}",
          flush=True)
    print("PH0_PILOT_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
