#!/usr/bin/env python3
"""PH1 fusion — one aligned record per clip from ego + VLM + SAM3 (+ Alpamayo).

Strategy doc: `…/2026-08-07-hierarchical-wm-redesign/PH1_FUSION_STRATEGY.md`.
The one-line version: **jurisdiction, not averaging** — SAM3 owns pixels, the
VLM owns symbols, ego owns the metric spine, Alpamayo is an external second
opinion; every cross-source relation is a recorded corroboration or a recorded
conflict, never a silent merge.

Binding rules enforced structurally here (not by convention):
  * pixels only from SAM3 — there is no code path that promotes a VLM box to
    geometry (B3 measured 2/23 same-frame agreement → diagnostic-only);
  * every field carries provenance; `inference_admissible` whitelists the
    vision-only fields (labels may use ego; INFERENCE IS VISION-ONLY);
  * the goal/vocab fields never contain situation-classifier output — the
    goal/situation information-disjointness rule survives fusion (asserted);
  * sign OCR text carried but `pending_g1_gate` until the PI grades it;
  * vocabulary tokens are IMPORTED from `tanitad.models.v6` — the emitter and
    the consumer cannot drift.

Usage:
  PYTHONPATH=<stack> python3 scripts/ph1_fuse.py \
      --v2-json <v2/ph0_v2.json> --sam3 <sam3 dir-or-json> \
      --ego-root <ego/> [--records <records.parquet>] \
      [--missing-sam3-ok REASON] --out <fused/>

A clip present in --v2-json but absent from the SAM3 output is a NAMED
PARTIAL, never a silent zero: without --missing-sam3-ok the run refuses
loudly; with it, the clip is fused from its other layers and its perception
layer carries {"absent": REASON}, and every SAM3-dependent corroboration
degrades to not_computable instead of fabricating a verdict from an empty
detector. (The 600-clip val run predates this and silently fused 4 such
clips with an empty perception layer — MEASURED n_v2=600 vs n_sam3=596.)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import s2_derive  # noqa: E402  — the ONE home of the S2 strategic mapping
from s2_derive import GOAL_TO_GSTR  # noqa: E402  (moved there; re-exported)
from tanitad.models.v6 import (STRATEGIC_GOAL_TOKENS,  # noqa: E402
                               TACTICAL_LAT_ACTIONS, TACTICAL_LON_ACTIONS)

# the pins in s2_derive must equal the real v6 lists — loudly, at import
assert s2_derive.check_vocab_drift() == "checked"

SCHEMA = "ph1-fused-v1"
IOU_TRACK = 0.3          # greedy same-concept association across frames
SPEED_TOL = 0.15         # speed-sign corroboration margin
STOP_V = 0.5             # m/s — "stopped" threshold
# substring rules mapping free-ish action text onto the FACTORED axes
LAT_RULES = (("lane_change_l", "LANE_CHANGE_L"), ("left_lane", "LANE_CHANGE_L"),
             ("lane_change_r", "LANE_CHANGE_R"), ("right_lane", "LANE_CHANGE_R"),
             ("change_left", "LANE_CHANGE_L"), ("change_right", "LANE_CHANGE_R"),
             ("nudge_l", "NUDGE_L"), ("nudge_r", "NUDGE_R"),
             ("keep", "LANE_KEEP"), ("hold_corridor", "LANE_KEEP"),
             ("straight", "LANE_KEEP"))
LON_RULES = (("brake", "BRAKE_TO"), ("stop", "BRAKE_TO"), ("decel", "BRAKE_TO"),
             ("yield", "YIELD_MERGE"), ("merge", "YIELD_MERGE"),
             ("creep", "CREEP"), ("hold", "HOLD"), ("wait", "HOLD"),
             ("follow", "FOLLOW"), ("cruise", "CRUISE"), ("accel", "CRUISE"))


def _map_rules(text: str, rules) -> str | None:
    t = (text or "").lower()
    for sub, tok in rules:
        if sub in t:
            return tok
    return None


def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


def build_tracks(frames: dict) -> list[dict]:
    """Greedy same-concept IoU association across SAM3's processed frames.

    Dynamics are ORDINAL on purpose: box-area trend + image-x drift classify
    approaching/receding/crossing. No metric depth is invented — the strategy
    doc's §7 limit, kept honest here.
    """
    idxs = sorted(int(k) for k in frames)
    open_tracks: list[dict] = []
    done: list[dict] = []
    for fi in idxs:
        dets = list((frames[str(fi)] or {}).get("det") or [])
        used = set()
        for tr in list(open_tracks):
            best, bj = 0.0, None
            for j, d in enumerate(dets):
                if j in used or d.get("concept") != tr["concept"]:
                    continue
                v = _iou(tr["boxes"][-1][1], d["box_xyxy"])
                if v > best:
                    best, bj = v, j
            if bj is not None and best >= IOU_TRACK:
                d = dets[bj]
                used.add(bj)
                tr["boxes"].append((fi, d["box_xyxy"]))
                tr["scores"].append(float(d.get("score", 0)))
                tr["areas"].append(float(d.get("mask_area_px") or 0))
            else:
                done.append(open_tracks.pop(open_tracks.index(tr)))
        for j, d in enumerate(dets):
            if j not in used:
                open_tracks.append({
                    "concept": d.get("concept"),
                    "boxes": [(fi, d["box_xyxy"])],
                    "scores": [float(d.get("score", 0))],
                    "areas": [float(d.get("mask_area_px") or 0)],
                })
    done.extend(open_tracks)
    out = []
    for k, tr in enumerate(done):
        a0, a1 = tr["areas"][0], tr["areas"][-1]
        x0 = (tr["boxes"][0][1][0] + tr["boxes"][0][1][2]) / 2
        x1 = (tr["boxes"][-1][1][0] + tr["boxes"][-1][1][2]) / 2
        if len(tr["boxes"]) < 2 or a0 <= 0:
            dyn = "single_frame"
        elif a1 > 1.3 * a0:
            dyn = "approaching"
        elif a1 < 0.7 * a0:
            dyn = "receding"
        elif abs(x1 - x0) > 40:
            dyn = "crossing"
        else:
            dyn = "steady"
        out.append({"track_id": k, "concept": tr["concept"],
                    "n_frames": len(tr["boxes"]),
                    "frame_span": [tr["boxes"][0][0], tr["boxes"][-1][0]],
                    "mean_score": round(sum(tr["scores"]) / len(tr["scores"]), 3),
                    "boxes": tr["boxes"], "dynamics": dyn, "src": "sam3"})
    return out


def ego_from_npz(path: str) -> dict:
    """Engine-A style metric spine computed from the bridged ego npz.

    The v2 records carry only ego_state (the prompt view); route/speed_profile/
    situations are NOT in them (MEASURED on the 600-clip production output).
    speed_profile is recomputed here deterministically from poses[:, 3]; the
    frozen situation detectors are NOT re-implemented — situations stay absent
    unless the record carries them, and checks degrade to not_computable.
    """
    import numpy as np
    try:
        d = np.load(path)
        poses = d["poses"]
    except Exception:                                       # noqa: BLE001
        return {}
    v = poses[:, 3].astype(float)
    yaw = poses[:, 2].astype(float)
    stops = int(((v[:-1] >= STOP_V) & (v[1:] < STOP_V)).sum()
                + (1 if v[0] < STOP_V else 0))
    net_dyaw = float(yaw[-1] - yaw[0])
    return {"speed_profile": {
                "v_t0": float(v[0]), "v_min_future": float(v.min()),
                "v_max_future": float(v.max()),
                "net_dv": float(v[-1] - v[0]), "stops": stops,
                "src": "ego_npz"},
            "net_dyaw_rad": net_dyaw}


def corroborate(v2: dict, sam3: dict, tracks: list[dict],
                sam3_absent: bool = False) -> tuple[dict, list]:
    cor, conflicts = {}, []
    sp = v2.get("speed_profile") or {}
    v_now = (v2.get("ego_state") or {}).get("v_now_ms")
    # --- speed sign vs ego speed profile ---------------------------------- #
    for s in ((v2.get("signs") or {}).get("signs") or []):
        if s.get("kind") == "speed" and str(s.get("text", "")).isdigit():
            lim = float(s["text"])
            vmin = sp.get("v_min_future")
            row = {"sign_text": s["text"], "v_now_ms": v_now,
                   "v_min_future": vmin, "src": ["vlm", "ego"]}
            if vmin is None:
                row["verdict"] = "not_computable"
            else:
                # unit-honest: corroborated under EITHER km/h or mph reading
                ok = any(min(vmin, v_now or vmin) <= (1 + SPEED_TOL) * lim / f
                         for f in (3.6, 2.237))
                row["verdict"] = "corroborated" if ok else "conflict"
                if not ok:
                    conflicts.append({"check": "speed_sign_vs_ego", **row})
            cor["speed_sign_vs_ego"] = row
    # --- red light vs ego stop -------------------------------------------- #
    reds = [s for s in ((v2.get("signs") or {}).get("signs") or [])
            if s.get("state") == "red"]
    if reds:
        stopped = (sp.get("stops") or 0) > 0 or (
            sp.get("v_min_future") is not None
            and sp["v_min_future"] < STOP_V)
        cor["red_light_vs_stop"] = {
            "n_red": len(reds), "ego_stopped": bool(stopped),
            "verdict": "corroborated" if stopped else "conflict",
            "src": ["vlm", "ego"]}
        if not stopped:
            conflicts.append({"check": "red_light_vs_stop", "n_red": len(reds)})
    # --- scene vs situations ---------------------------------------------- #
    scene = v2.get("scene") or {}
    sit = v2.get("situations")
    claims_int = "intersection" in str(scene.get("domain", "")) or \
                 scene.get("road_type") == "junction"
    if claims_int:
        if sit is None:
            # ⚠️ absent data must not manufacture a conflict — the frozen
            # situation detectors did not run on this batch, and saying
            # "conflict" here would be a fabricated disagreement.
            cor["scene_vs_situations"] = {"verdict": "not_computable",
                                          "reason": "no situations source",
                                          "src": ["vlm"]}
        else:
            ok = bool(sit.get("intersection"))
            cor["scene_vs_situations"] = {
                "scene_claims_intersection": True,
                "ego_intersection_window": ok,
                "verdict": "corroborated" if ok else "conflict",
                "src": ["vlm", "ego"]}
            if not ok:
                conflicts.append({"check": "scene_vs_situations"})
    # --- goal evidence grounded by SAM3 ----------------------------------- #
    sym = v2.get("symbols") or {}
    if sym.get("goal_kind") == "route_to":
        ev = sym.get("goal_evidence_sign")
        if sam3_absent:
            # ⚠️ same rule as the situations fix: an absent detector must not
            # manufacture a "provisional/ungrounded" verdict — it saw nothing.
            cor["goal_evidence"] = {"evidence_sign_idx": ev,
                                    "verdict": "not_computable",
                                    "reason": "sam3 absent for this clip",
                                    "src": ["vlm"]}
        else:
            n_sign_frames = sum(1 for t in tracks
                                if t["concept"] == "traffic sign")
            cor["goal_evidence"] = {
                "evidence_sign_idx": ev, "sam3_sign_tracks": n_sign_frames,
                "verdict": ("grounded" if ev is not None and n_sign_frames > 0
                            else "provisional"), "src": ["vlm", "sam3"]}
    # --- census vs scene --------------------------------------------------- #
    if scene.get("road_type") == "urban":
        n_agents = sum(1 for t in tracks
                       if t["concept"] in ("car", "truck", "bus", "pedestrian",
                                           "cyclist"))
        if sam3_absent:
            cor["census_vs_scene"] = {"verdict": "not_computable",
                                      "reason": "sam3 absent for this clip",
                                      "src": ["vlm"]}
        elif n_agents == 0:
            cor["census_vs_scene"] = {"verdict": "flagged_empty_urban",
                                      "src": ["sam3", "vlm"]}
    return cor, conflicts


# ⚠️ ego_past_state emits turning ∈ {turning_left, turning_right, straight}
# (ph0_v2.py:400-401). The first vote map keyed on {left, right} and the ego
# lateral vote was DEAD on every turning clip — MEASURED: null on 36/36
# turning clips, LANE_KEEP on 165/165 straight (S2_STRATEGIC_GAP §6 item 6).
_EGO_TURN_VOTE = {"turning_left": "NUDGE_L", "turning_right": "NUDGE_R",
                  "left": "NUDGE_L", "right": "NUDGE_R",
                  "straight": "LANE_KEEP"}


def emit_vocab(v2: dict, alp: dict | None,
               engine_a: dict | None = None) -> tuple[dict, list]:
    """g_str/a_str GEOMETRY-PRIMARY via s2_derive (VLM demoted to recorded
    corroboration; ROUTE_TO gated); factored g_tac by 2-of-3 votes.

    With ``engine_a`` absent (legacy inputs) the strategic tokens fall back
    to VLM-primary, tagged ``vlm-fused`` — stated, never silent. The old
    ``corroborated_by_route`` field (structurally dead: it read a key the
    production records never carry, False on 801/801) is DELETED — the g_str
    block's ``sources``/``corroboration`` supersede it.
    """
    conflicts = []
    sym = v2.get("symbols") or {}
    if str(sym.get("goal_kind", "none")).lower() not in GOAL_TO_GSTR:
        conflicts.append({"check": "goal_kind_unmapped",
                          "value": sym.get("goal_kind")})

    # ---- strategic layer: geometry decides, the VLM corroborates ----------
    g_str = s2_derive.derive_g_str(engine_a, sym)
    a_str = s2_derive.derive_a_str(engine_a, sym)
    for blk in (g_str, a_str):
        blk["src"] = "engine_a" if blk["provenance"] == "path" else "vlm"
    assert g_str["token"] in STRATEGIC_GOAL_TOKENS

    # ---- tactical votes (unchanged jurisdiction: 2-of-3) ------------------
    ego = v2.get("ego_state") or {}
    turning = str(ego.get("turning", ""))
    lat_votes = [("ego", _EGO_TURN_VOTE.get(turning))]
    lon_ego = None
    sp = v2.get("speed_profile") or {}
    if (sp.get("stops") or 0) > 0:
        lon_ego = "BRAKE_TO"
    elif str(ego.get("motion")) == "steady":
        lon_ego = "CRUISE"
    lon_votes = [("ego", lon_ego)]
    for a in (sym.get("actions") or []):
        txt = f"{a.get('verb', '')}_{a.get('direction', '')}"
        lat_votes.append(("vlm", _map_rules(txt, LAT_RULES)))
        lon_votes.append(("vlm", _map_rules(txt, LON_RULES)))
    if alp and alp.get("meta_action"):
        t = json.dumps(alp["meta_action"])[:400]
        lat_votes.append(("alpamayo", _map_rules(t, LAT_RULES)))
        lon_votes.append(("alpamayo", _map_rules(t, LON_RULES)))

    def majority(votes, valid):
        counts: dict[str, list] = {}
        for src, tok in votes:
            if tok in valid:
                counts.setdefault(tok, []).append(src)
        if not counts:
            return None, [], votes
        tok = max(counts, key=lambda k: len(counts[k]))
        return tok, counts[tok], votes

    lat, lat_src, lat_all = majority(lat_votes, TACTICAL_LAT_ACTIONS)
    lon, lon_src, lon_all = majority(lon_votes, TACTICAL_LON_ACTIONS)
    vocab = {"g_str": g_str,
             "a_str": a_str,
             "g_tac_lat": {"token": lat, "voters": lat_src,
                           "votes": [[s, t] for s, t in lat_all]},
             "g_tac_lon": {"token": lon, "voters": lon_src,
                           "votes": [[s, t] for s, t in lon_all]}}
    # ⛔ the disjointness rule: no situation-classifier output inside vocab
    assert "situation" not in json.dumps(vocab).lower()
    return vocab, conflicts


def scenario_line(v2: dict, tracks: list[dict]) -> str:
    sc = v2.get("scene") or {}
    ego = v2.get("ego_state") or {}
    sit = v2.get("situations") or {}
    census: dict[str, int] = {}
    for t in tracks:
        census[t["concept"]] = census.get(t["concept"], 0) + 1
    parts = [
        f"{sc.get('illumination', '?')}, {sc.get('weather', '?')}, "
        f"{sc.get('road_type', '?')} {sc.get('lanes_visible', '?')}-lane",
        f"ego {ego.get('v_now_ms', float('nan')):.1f} m/s {ego.get('motion', '?')}"
        f"/{ego.get('turning', '?')}",
        ", ".join(f"{v} {k}" for k, v in sorted(census.items())) or "no agents",
    ]
    flags = [k for k in ("lane_change", "intersection", "roundabout")
             if sit.get(k)]
    if flags:
        parts.append("situations: " + "+".join(flags))
    return "; ".join(parts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2-json", required=True)
    ap.add_argument("--sam3", required=True,
                    help="sam3 output: a JSON file or a directory of them")
    ap.add_argument("--ego-root", required=True)
    ap.add_argument("--records", default=None,
                    help="Alpamayo records.parquet (optional layer)")
    ap.add_argument("--engine-a", default=None,
                    help="optional Engine A sidecar: JSONL rows "
                         '{"clip_id", "engine_a"} (e.g. the S2 v1 recompute '
                         "from the bridged npz). A v2 record's own persisted "
                         "engine_a field wins over the sidecar.")
    ap.add_argument("--missing-sam3-ok", default=None, metavar="REASON",
                    help="permit clips present in --v2-json but absent from "
                         "the SAM3 output: fuse their other layers, stamp "
                         "perception {'absent': REASON}, and degrade the "
                         "SAM3-dependent checks to not_computable. Without "
                         "this flag a partial SAM3 leg refuses loudly.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    v2 = json.load(open(a.v2_json))
    v2 = v2 if isinstance(v2, list) else v2.get("clips", v2)
    v2_by = {r["clip_id"]: r for r in v2 if isinstance(r, dict)}

    sam3_by: dict[str, dict] = {}
    paths = ([a.sam3] if os.path.isfile(a.sam3)
             else sorted(glob.glob(os.path.join(a.sam3, "*.json"))))
    for p in paths:
        d = json.load(open(p))
        # MEASURED formats: the production sam3.json is a metadata wrapper with
        # the records under "clips" (596 rows); older outputs were a bare list
        # or one record. The first fuse run matched 0 of 600 because this
        # wrapper wasn't handled — n_sam3 is asserted below so an empty
        # perception layer can never again look like a successful fuse.
        if isinstance(d, dict) and isinstance(d.get("clips"), list):
            rows = d["clips"]
        elif isinstance(d, list):
            rows = d
        elif isinstance(d, dict) and "clip_id" in d:
            rows = [d]
        else:
            rows = list(d.values()) if isinstance(d, dict) else []
        for r in rows:
            if isinstance(r, dict) and r.get("clip_id"):
                sam3_by[r["clip_id"]] = r
    if not sam3_by:
        raise SystemExit("[fuse] loaded 0 SAM3 records — refusing to emit "
                         "fused records with an empty perception layer "
                         f"(looked in {a.sam3})")
    missing = sorted(c for c in v2_by if c not in sam3_by)
    if missing and not a.missing_sam3_ok:
        raise SystemExit(
            f"[fuse] {len(missing)} of {len(v2_by)} v2 clips have NO SAM3 "
            f"record (e.g. {missing[:3]}) — pass --missing-sam3-ok REASON "
            "to fuse them with the perception layer explicitly marked "
            "absent, or complete the SAM3 leg first")

    alp_by: dict[str, dict] = {}
    if a.records:
        import pandas as pd
        df = pd.read_parquet(a.records)
        for cid, g in df.groupby("clip_id"):
            alp_by[cid] = {row["task"]: row.get("raw_json")
                           for _, row in g.iterrows()}

    ea_by: dict[str, dict] = {}
    if a.engine_a:
        with open(a.engine_a, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    if row.get("clip_id") and row.get("engine_a"):
                        ea_by[row["clip_id"]] = row["engine_a"]
        print(f"[fuse] engine A sidecar: {len(ea_by)} clips", flush=True)

    n, summ = 0, {"corroborated": 0, "conflicts": 0, "with_alpamayo": 0,
                  "sam3_missing": 0}
    for cid, r in sorted(v2_by.items()):
        dst = os.path.join(a.out, f"{cid}.json")
        if os.path.exists(dst):
            continue
        absent_reason = a.missing_sam3_ok if cid not in sam3_by else None
        s3 = sam3_by.get(cid) or {}
        tracks = build_tracks(s3.get("frames") or {})
        # engine-A spine: recompute from the bridged npz when the v2 record
        # does not carry it (the production records do not — MEASURED)
        if "speed_profile" not in r:
            spine = ego_from_npz(os.path.join(a.ego_root, f"{cid}.npz"))
            if spine:
                r = {**r, **spine}
        cor, conf = corroborate(r, s3, tracks,
                                sam3_absent=absent_reason is not None)
        alp = alp_by.get(cid)
        engine_a = r.get("engine_a") or ea_by.get(cid)
        vocab, vconf = emit_vocab(r, alp, engine_a=engine_a)
        conf += vconf
        # ⛔ the provenance lie fix (S2_STRATEGIC_GAP §5): the v2.2 records
        # were produced with ego-past kinematics AND the Engine A hindsight
        # block in the prompt — the VLM layer is NOT pure vision, and the
        # tag must say it. `semantics` is then also removed from the
        # inference whitelist: labels may use ego, INFERENCE IS VISION-ONLY.
        ego_mode = r.get("_ego_prompt_mode") or "none"
        vlm_prov = ("vision" if ego_mode == "none" else
                    f"vision+ego-{ego_mode}-prompt+engineA-prompt")
        admissible = (["perception", "semantics"] if ego_mode == "none"
                      else ["perception"])
        fused = {
            "schema_version": SCHEMA, "clip_id": cid,
            "geometry": {"frame_wh": r.get("_frame_wh"),
                         "note": "w120 cylindrical vs 256px pinhole batches "
                                 "must not be pooled"},
            "ego": {k: v for k, v in
                    [(k, r.get(k)) for k in
                     ("ego_state", "route", "speed_profile", "speed_events",
                      "lane_change_events", "situations")] +
                    [("engine_a", engine_a)] if v is not None or k in r},
            "perception": {"tracks": tracks,
                           "per_concept_hits": s3.get("per_concept_hits"),
                           "src": "sam3",
                           **({"absent": absent_reason}
                              if absent_reason else {})},
            "semantics": {"scene": r.get("scene"), "signs": r.get("signs"),
                          "symbols": r.get("symbols"), "src": "vlm",
                          "sign_text_status": "pending_g1_gate"},
            "alpamayo": alp,
            "corroboration": cor, "vocab": vocab,
            "scenario_description": scenario_line(r, tracks),
            "_conflicts": conf,
            "inference_admissible": admissible,
            **({"_inference_admissible_note":
                f"semantics EXCLUDED: VLM ran with ego in the prompt "
                f"(mode={ego_mode!r}) — its outputs are ego-touched"}
               if ego_mode != "none" else {}),
            "_provenance": {"ego": "privileged-labels-only",
                            "sam3": "vision", "vlm": vlm_prov,
                            "alpamayo": "external-labels-only",
                            **({"engine_a": "privileged-labels-only "
                                            "(hindsight ego path)"}
                               if engine_a else {})},
        }
        json.dump(fused, open(dst, "w"), indent=1)
        n += 1
        summ["conflicts"] += len(conf)
        summ["corroborated"] += sum(1 for c in cor.values()
                                    if c.get("verdict") == "corroborated")
        summ["with_alpamayo"] += int(alp is not None)
        summ["sam3_missing"] += int(absent_reason is not None)
        if n % 100 == 0:
            print(f"[fuse] {n} fused", flush=True)
    summ["n_fused"] = n
    summ["n_v2"] = len(v2_by)
    summ["n_sam3"] = len(sam3_by)
    if a.missing_sam3_ok:
        summ["missing_sam3_reason"] = a.missing_sam3_ok
    json.dump(summ, open(os.path.join(a.out, "_summary.json"), "w"), indent=1)
    print(f"FUSE_DONE {json.dumps(summ)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
