#!/usr/bin/env python3
"""nt_labels -- E2 of H-NT-DECIDE-1: the label release for the navtrain fine-tune of refc (TANITAD VENV, CPU).

Outputs (one call, one --out directory):

* ``v9_labels_nt_train.npz`` + ``.manifest.json`` -- the v9 per-frame release (schema ``tanitad.v9_labels/1``) with
  ONE row per token at ``k = 9`` (the window NOW), ``now_s`` on the EPISODE clock. Its md5 is NEW and is what the
  launch argv passes as ``--r8-v9-md5``.
  - tactical lat / lon (+ allowed masks, + the v7-id encodings the trainer reads), the 22 goal bits, the
    continuous lat / lon / speed constraint vectors: ``build_v9_labels.clip_geo`` + ``clip_labels`` UNCHANGED, run
    on the LOG's 2 Hz ego track (``g_max_s`` 1.0 admits the 0.5 s steps; 10 Hz is the builder's own linear
    resampling). ``rec = None`` (no Alpamayo CoT on NavSim: every VLM goal is IGNORE, ``vlm_has_cot = 0``),
    ``lead`` = the FOLLOW evidence from each log frame's OpenScene ``anns`` (``--lead anns``, default since
    2026-10-21 ~23:55: v1 shipped ``lead = None``, which made every DECEL/ACCEL/KEEP row a partial {cls, FOLLOW}
    label -- MEASURED on the provisional release: hard ``lon_cls`` IGNORE on 74.15 % of rows vs 30.86 % on refcv8's
    PhysicalAI release; ``--lead none`` reproduces v1), ``lc = None`` with ``lc_labels off`` (as the PhysicalAI
    release).
  - ``nav_token`` = the NavSim ``driving_command`` at t0 through ``NAVSIM_LEGAL_NAV_CMD`` (the INFERENCE function,
    ``refcv8_bridge.py:154-180``). The builder's own future-derived token is kept as ``nav_token_geom``
    (diagnostic, never read by the trainer). The nav ARGS stay future-derived (the recipe's 0.5 dropout; the
    LEGAL inference row is always unknown).
* ``s2_labels_nt_train.jsonl.gz`` -- the minimal ``s2-geom-v7`` record per token the trainer's v7 join needs
  (coverage, the G3 clock, ``--nav-from-v7``'s token, the v7 goal census). The v9 release REPLACES its tactical
  targets in ``V3Dataset.__getitem__`` (``r8_v9_targets``), so a placeholder ``a_tac`` is stamped as one.
* ``clip_clock_nt.jsonl`` (+ ``.meta.json``) -- ``grid_start_s 0.0``, ``dt_s 0.1`` per token, optionally MERGED
  with the PhysicalAI sidecar (the trainer reads ONE ``--clip-clock-sidecar`` for train AND the eval139 monitor).
* ``speed_label_nt_train.jsonl`` (+ ``.meta.json``) -- the road-type sidecar with EVERY row UNKNOWN (``valid 0``):
  the LEGAL inference feed. ⚠️ Both sidecar readers refuse an all-unknown file unless the trainer runs with
  ``--r8-speed-unknown-always`` (the E4 trainer patch).
* ``token_table_nt.json`` + ``LABEL_REPORT.json`` (census, md5s, the agreement between ``driving_command`` and
  the geometry token, the provenance of every clip id).

Clip ids: E1's manifest (``--e1-manifest``) when given, else PROVISIONAL ``navtrain-<token>`` (stamped).
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nt_common as C  # noqa: E402


def _bv(tree=None):
    C.tree_stack(tree)
    import build_v9_labels as BV        # noqa: E402  (stack/scripts, by sys.path)
    C.assert_not_g(BV)
    return BV


V7_SCHEMA = "s2-geom-v7"
V7_VOCAB = "v7"
LAT7 = ("LANE_KEEP", "LANE_CHANGE_L", "LANE_CHANGE_R", "ABORT_LC", "NUDGE_L", "NUDGE_R", "TURN_L", "TURN_R")
LON7 = ("FOLLOW", "CRUISE", "YIELD_MERGE", "BRAKE_TO", "CREEP", "HOLD", "ADAPT_SPEED_FOR_CURVE", "ACCELERATE")


LEAD_MODES = ("anns", "none")
LEAD_TICK_S = 0.1          # the builder's band grid for a one-row call (``clip_labels``: dtk = 0.1 when n == 1)
LEAD_SNAP_S = 0.25         # a 10 Hz tick takes the evaluation of the 2 Hz log frame nearest it (half the spacing)
LEAD_MAX_TICKS = 90        # keys 0 .. 90 ticks after NOW: a superset of the builder's band (2 .. 8 s)


def lead_agents_of_frame(frame: dict, is_ego, census: Counter) -> list | None:
    """The FOLLOW-evidence reading of one frame's OpenScene ``anns``: E3's class map (``C.CLASS_MAP``: vehicle ->
    automobile, bicycle -> rider, pedestrian -> person; every other class dropped), E3's finite filter and E3's
    ego-footprint drop (``join_label_hygiene.is_ego_footprint``), boxes in THAT frame's ego frame (``gt_boxes`` are
    lidar = ego, ``lidar2ego`` identity -- measured by E3). Identical to ``nt_joins.agents_of_frame`` on
    (cx, cy, l, cls) wherever that one succeeds (pinned by the test); it differs ONLY in never reading track ids, so
    a 60-bit track collision cannot kill an unattended label build. Returns ``None`` (= NO DATA for this frame,
    counted) when the frame has no ``anns`` or its arrays disagree in length."""
    an = frame.get("anns")
    if an is None:
        census["lead_frame_no_anns"] += 1
        return None
    boxes = np.asarray(an.get("gt_boxes", np.zeros((0, 7))), dtype=np.float64).reshape(-1, 7)
    names = [str(x) for x in an.get("gt_names", [])]
    if len(boxes) != len(names):
        census["lead_frame_len_mismatch"] += 1
        return None
    out = []
    for b, nm in zip(boxes, names):
        cls = C.CLASS_MAP.get(nm)
        if cls is None:
            continue
        cx, cy, cz, l, w, h, yaw = (float(v) for v in b)
        if not all(math.isfinite(v) for v in (cx, cy, cz, l, w, h, yaw)) or is_ego(cx, cy):
            continue
        out.append({"cx": round(cx, 4), "cy": round(cy, 4), "l": round(l, 3), "cls": cls})
    return out


def lead_stream(BV, log, G, frames: list, t_s: np.ndarray, census: Counter) -> list:
    """Per 2 Hz log frame: ``(has_data, lead?, gap_m, time_gap_s, s_along_m, v_ego)`` = ``BV.lead_for_frame``
    (UNCHANGED, the PhysicalAI lead pass's function) on that frame's OWN agents at that frame's OWN time, on the
    builder's own track (``G.tr``) and the same unsmoothed path the PhysicalAI lead pass uses (``smooth_path(tr,
    0.0)``, ``build_v9_labels.cmd_lead_pass``)."""
    from tanitad.data.join_label_hygiene import is_ego_footprint
    p0 = BV.smooth_path(G.tr, 0.0)
    out = []
    for f, fr in enumerate(frames):
        t = float(t_s[f])
        ag = lead_agents_of_frame(fr, is_ego_footprint, census)
        if ag is None or t < 0 or t > float(log.t_end):
            out.append((False, False, np.nan, np.nan, np.nan, np.nan))
            continue
        L, gap, tg, s, v = BV.lead_for_frame(log, G.tr, p0, t, ag)
        out.append((True, bool(L), float(gap), float(tg), float(s), float(v)))
        census["lead_frames_with_data"] += 1
        census["lead_frames_lead"] += int(bool(L))
    return out


def lead_for_token(stream: list, t_s: np.ndarray, now: float, kq: int) -> dict:
    """The builder's ``lead`` dict for ONE token, keyed exactly as ``clip_labels`` looks it up for a one-row call
    (``band_k = k + j``, j on the 0.1 s grid): tick ``kq + j`` (time ``now + 0.1 j``) carries the stream entry of the
    log frame nearest it when that frame is within ``LEAD_SNAP_S`` (sample-and-hold of the 2 Hz perception onto the
    builder's 10 Hz band, so ``cov_s = ticks * 0.1`` measures seconds as on PhysicalAI); a tick with no frame that
    close (a log gap) carries nothing = not covered."""
    d = {}
    for j in range(0, LEAD_MAX_TICKS + 1):
        tq = now + LEAD_TICK_S * j
        f = int(np.searchsorted(t_s, tq))
        cand = [c for c in (f - 1, f) if 0 <= c < len(t_s)]
        if not cand:
            continue
        b = min(cand, key=lambda c: (abs(float(t_s[c]) - tq), c))
        if abs(float(t_s[b]) - tq) <= LEAD_SNAP_S + 1e-9:
            d[kq + j] = stream[b]
    return d


def token_fields(BV, G, now: float, kq: int, ld: dict | None) -> dict:
    """ONE ``clip_labels`` call for ONE token (n = 1, so the builder's band grid is its own 0.1 s; ``k = kq``, the
    token's tick on that grid, so ``ld``'s keys are found). Only the columns this row WRITES are returned."""
    F = BV.clip_labels(G, np.array([now], dtype=np.float64), np.array([kq], dtype=np.int64), None, ld, None,
                       lc_labels=False)
    return {nm: np.asarray(col)[0] for nm, col in F.items()}


def labels_for_log(BV, frames: list, rows: list, lead: str = "anns", census: Counter | None = None) -> dict:
    """Run the v9 builder for every token of ONE log. ``rows`` = this log's token-table rows (in any order).
    Returns ``{token: {field: scalar}}``.

    Base = v1's ONE call per log (``lead=None``) -- every column, with the BUILDER'S OWN fill where a row does not
    write it (the builder creates conditional columns lazily; ``write_v9_release``'s ``_fill_like`` would fill an
    int8 column with 0 where the builder writes -1, MEASURED on ``lc_measurable``). With ``lead="anns"`` each token's
    OWN call (:func:`token_fields`, with its ``lead`` dict from the log's OpenScene ``anns``) is laid over it.
    ``"none"`` = v1 exactly (the MM's off-switch). Pinned: per-token fields == the v1 call wherever both write
    (``rec = None`` / ``lc = None``: no field reads across rows), and ``"anns"`` changes only lon / lead / goal fields."""
    if lead not in LEAD_MODES:
        raise C.NTRefused(f"[nt] lead mode {lead!r} not in {LEAD_MODES}")
    census = Counter() if census is None else census
    ea = C.ego_arrays(frames)
    log = BV.log_from_arrays(ea["t_s"], ea["x"], ea["y"], ea["yaw"], ea["v"], None, ea["rev"])
    G = BV.clip_geo(log, None)
    rows = sorted(rows, key=lambda r: r["frame_idx"])
    now_k = np.array([float(ea["t_s"][r["frame_idx"]]) for r in rows], dtype=np.float64)
    F1 = BV.clip_labels(G, now_k, np.full(len(rows), C.NOW_RAW_ROW, dtype=np.int64), None, None, None,
                        lc_labels=False)
    stream = lead_stream(BV, log, G, frames, ea["t_s"], census) if lead == "anns" else None
    out, has_lead = {}, set()
    for i, r in enumerate(rows):
        now = float(now_k[i])
        d = {nm: np.asarray(col)[i] for nm, col in F1.items()}
        if stream is not None:
            kq = int(round(now / LEAD_TICK_S))
            ld = lead_for_token(stream, ea["t_s"], now, kq)
            d.update(token_fields(BV, G, now, kq, ld))
            if any(v[0] for v in ld.values()):
                has_lead.add(r["token"])
        nav = C.nav_token_of_driving_command(ea["driving_command"][r["frame_idx"]])
        d["nav_token_geom"] = np.float64(d["nav_token"])
        d["nav_token"] = np.float64(nav["nav_token"])
        d["navsim_cmd_argmax"] = np.int8(nav["navsim_argmax"])
        d["log_now_s"] = np.float64(now)
        out[r["token"]] = d
    meta = {"n_turns": len(G.turns), "n_announced": int(sum(u.announced for u in G.turns)),
            "suppressed": 0, "log_t_end_s": float(log.t_end), "has_lead_tokens": has_lead}
    return {"by_token": out, "clip_meta": meta}


def _fill_like(ref: np.ndarray, n: int) -> np.ndarray:
    ref = np.asarray(ref)
    if np.issubdtype(ref.dtype, np.floating):
        return np.full(n, np.nan, dtype=ref.dtype)
    if ref.dtype == np.int16:
        return np.full(n, -100, dtype=ref.dtype)
    return np.zeros(n, dtype=ref.dtype)


def write_v9_release(BV, table: list, per_token: dict, clip_meta: dict, out_dir: Path, split: str,
                     base_commit: str, sources: dict) -> dict:
    """The release in ``build_v9_labels.cmd_build``'s exact npz layout (``row__*`` / ``clip__*``), one row per
    token, rows in TABLE order; ``k`` int16 = 9; ``now_s`` = the trainer's own NOW on the episode clock."""
    order = [r for r in table if r["token"] in per_token]
    now_ep = C.episode_now_s()
    names = sorted(set().union(*[set(per_token[r["token"]]) for r in order]))
    cols = {}
    for nm in names:
        ref = next(per_token[r["token"]][nm] for r in order if nm in per_token[r["token"]])
        arr = _fill_like(np.asarray([ref]), len(order))
        for i, r in enumerate(order):
            if nm in per_token[r["token"]]:
                arr[i] = per_token[r["token"]][nm]
        cols[nm] = arr
    cols["k"] = np.full(len(order), C.NOW_RAW_ROW, dtype=np.int16)
    cols["now_s"] = np.full(len(order), now_ep, dtype=np.float64)
    clip = {"sid": [], "sha12": [], "row0": [], "n_rows": [], "k0": [], "grid_start_s": [], "dt_s": [],
            "clock_src": [], "has_record": [], "has_lead": [], "has_lc": [], "n_turns": [], "n_announced": [],
            "suppressed": []}
    for i, r in enumerate(order):
        m = clip_meta[r["log_name"]]
        for key, val in (("sid", r["sid"]), ("sha12", C.sha12(r["clip_id"])), ("row0", i), ("n_rows", 1),
                         ("k0", C.NOW_RAW_ROW), ("grid_start_s", C.GRID_START_S), ("dt_s", C.DT_S),
                         ("clock_src", 0), ("has_record", 1),
                         ("has_lead", int(r["token"] in m.get("has_lead_tokens", ()))), ("has_lc", 0),
                         ("n_turns", m["n_turns"]), ("n_announced", m["n_announced"]), ("suppressed", 0)):
            clip[key].append(val)
    out = {"row__" + k: v for k, v in cols.items()}
    out.update({"clip__" + k: np.asarray(v) for k, v in clip.items()})
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / f"v9_labels_{split}.npz"
    np.savez_compressed(npz, **out)
    S2 = BV._builder()
    man = {"schema": "tanitad.v9_labels/1", "split": split, "n_clips": len(order), "n_rows": len(order),
           "row_key": "(sid, k): ONE row per NavSim token at k = 9 = the token-episode's window NOW "
                      "(t = 0, w 8, n_stack 3)",
           "clock_rule": "now_s = grid_start_s + k * dt_s on the EPISODE clock (grid_start 0.0, dt 0.1, clip-local; "
                         "the clip-clock sidecar of this release); geometry computed on the LOG timeline "
                         "(row__log_now_s)",
           "vocab": {"LAT9": BV.LAT9, "LON9": BV.LON9, "LAT7": BV.LAT7, "LON7": BV.LON7, "GOAL22": BV.GOAL22,
                     "GOAL_PROV": BV.GOAL_PROV, "LAT9_TO_7": BV.LAT9_TO_7, "LON9_TO_7": BV.LON9_TO_7},
           "literals": {k_: (list(v_) if isinstance(v_, tuple) else v_) for k_, v_ in BV.LITERALS.items()},
           "lc_labels": "off",
           "policies": {"nav_token": "NavSim driving_command at t0 -> NAVSIM_LEGAL_NAV_CMD (left->TURN_L 1, "
                                     "forward->FOLLOW 0, right->TURN_R 2, unknown->FOLLOW 0) = the inference "
                                     "function (refcv8_bridge.py:154-180); the builder's future-derived token is "
                                     "row__nav_token_geom (diagnostic)",
                        "nav_args": "future-derived (build_v9_labels.nav_fields), dropout 0.5 in training, "
                                    "unknown at inference",
                        "goal_negatives_vlm": "NONE on NavSim (no CoT record: every VLM goal IGNORE)",
                        "lead": sources.get("lead_policy", "NONE (no FOLLOW evidence pass: lon DECEL/ACCEL/KEEP "
                                                           "are partial {cls, FOLLOW})"),
                        "rc": "computed (spatial ego-future) but NOT FED (--r8-no-rc)",
                        "geometry_source": "OpenScene log ego2global (2 Hz) + ego_dynamic_state |v|; 10 Hz = the "
                                           "builder's own linear resampling"},
           "sources": sources,
           "builder": {"build_v9_labels.py": C.md5_file(BV.__file__), "s2_geom_emit_v7.py": C.md5_file(S2.__file__),
                       "nt_labels.py": C.md5_file(__file__), "nt_common.py": C.md5_file(C.__file__),
                       "base_commit": base_commit},
           "fields": sorted(out), "npz_md5": C.md5_file(npz)}
    C.json_dump(man, out_dir / f"v9_labels_{split}.manifest.json")
    return {"npz": str(npz), "md5": man["npz_md5"], "n_rows": len(order)}


def v7_record(r: dict, lab: dict, BV) -> dict:
    """The minimal s2-geom-v7 record of one token (``load_v7_labels`` reads exactly these keys)."""
    lat_id, lon_id = int(lab["lat_v7id_a"]), int(lab["lon_v7id"])
    a_tac = {"lat": LAT7[lat_id] if lat_id >= 0 else "LANE_KEEP",
             "lon": LON7[lon_id] if lon_id >= 0 else "CRUISE"}
    placeholder = {"lat": lat_id < 0, "lon": lon_id < 0}
    gy, gw = int(lab["goal_y"]), int(lab["goal_w"])
    goals = {t: {"provenance": "geometry", "grounded": False, "source": "v9 builder goal bits (NavSim nt)"}
             for i, t in enumerate(BV.GOAL22) if (gw >> i) & 1 and (gy >> i) & 1 and t != "SPEED_BAND"}
    tok = int(lab["nav_token"])
    return {"schema_version": V7_SCHEMA, "vocab": V7_VOCAB, "release": "nt-v1", "clip_id": r["clip_id"],
            "t0_s": C.episode_now_s(),
            "bands": {"operative_s": [0.0, 2.0], "tactical_s": [2.0, 6.0], "strategic_s": [8.0, 30.0],
                      "unassigned_manoeuvres": []},
            "horizon": {"available_s": round(r["future_ticks"] * C.DT_S, 3)},
            "a_tac": a_tac, "a_str": {"token": "HOLD_MAIN_ROAD", "provenance": "placeholder (no strategic label)"},
            "g_str": {"token": "FOLLOW_ROUTE", "provenance": "placeholder (no strategic label)"},
            "g_tac": {"anchor": None, "goals": goals, "violations": []},
            "nav_command": {"token": C.V7_NAV_OF_V9[tok], "provenance": "nav-system", "oracle": False, "args": {},
                            "source": "NavSim driving_command[t0] (navsim-legal mapping)"},
            "_nt": {"token": r["token"], "log_name": r["log_name"], "a_tac_placeholder": placeholder,
                    "note": "the v9 release (r8_v9_targets) supervises lat/lon/goals; a_tac is read only for its "
                            "vocabulary"}}


def write_v7(table: list, per_token: dict, out_path: Path, BV) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(filename="", mode="wb", fileobj=open(out_path, "wb"), mtime=0) as gz:
        for r in table:
            if r["token"] in per_token:
                gz.write((json.dumps(v7_record(r, per_token[r["token"]], BV), sort_keys=True) + "\n").encode())
    return {"path": str(out_path), "md5": C.md5_file(out_path)}


def write_clock(table: list, out_path: Path, merge_from: str | None) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_nt = n_m = 0
    sids = set()
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        for r in table:
            fh.write(json.dumps({"sid": r["sid"], "grid_start_s": C.GRID_START_S, "dt_s": C.DT_S,
                                 "source": "navsim-token-episode (constructed: raw rows at exactly 0.1 s)"}) + "\n")
            sids.add(r["sid"])
            n_nt += 1
        if merge_from:
            with open(merge_from, encoding="utf-8") as src:
                for line in src:
                    if line.strip():
                        if int(json.loads(line)["sid"]) in sids:
                            raise C.NTRefused(f"[nt] clock merge: sid collision with {merge_from}")
                        fh.write(line if line.endswith("\n") else line + "\n")
                        n_m += 1
    meta = {"schema": "tanitad.clip_clock/1", "built_by": "nt_labels.py", "builder_md5": C.md5_file(__file__),
            "n_rows": n_nt + n_m, "n_navsim_rows": n_nt, "n_merged_rows": n_m,
            "merged_from": (None if not merge_from else {"path": merge_from, "md5": C.md5_file(merge_from)}),
            "rule": "navsim token-episode: grid_start_s 0.0, dt_s 0.1 (raw row k at k * 0.1 s; NOW = row 9)"}
    C.json_dump(meta, str(out_path) + ".meta.json")
    return {"path": str(out_path), "md5": C.md5_file(out_path), **{k: meta[k] for k in ("n_navsim_rows", "n_merged_rows")}}


def write_speed(table: list, out_path: Path, v7_md5: str) -> dict:
    """Every row UNKNOWN: ``valid 0``, no bin, no limit, an all-zero one-hot (``refcv8_train._label_row_refusal``
    admits exactly this as the explicit UNKNOWN row)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        for r in table:
            fh.write(json.dumps({"sid": r["sid"], "v_hi_ms": None, "bin": None, "limit_kmh": None,
                                 "one_hot": [0.0, 0.0, 0.0, 0.0], "over_ceiling": False, "valid": 0}) + "\n")
    meta = {"produced_by": "nt_labels.py (H-NT-DECIDE-1 E2)", "builder_md5": C.md5_file(__file__),
            "rule": "EVERY row UNKNOWN (valid 0): the NavSim LEGAL inference feed (refcv8_bridge R8_LEGAL = the "
                    "trained unknown row). Declared departure from p_unknown 0.45 (DESIGN sec. 3). Readable only "
                    "under the trainer's --r8-speed-unknown-always.",
            "source_labels": "s2_labels_nt_train.jsonl.gz", "source_md5": v7_md5, "split": "train",
            "n_clips": len(table), "n_with_speed_band": 0, "n_without_speed_band": len(table),
            "join_key": "sid = stable_episode_id(clip_id)"}
    C.json_dump(meta, str(out_path) + ".meta.json")
    return {"path": str(out_path), "md5": C.md5_file(out_path), "n_rows": len(table)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--logs-dir", required=True, help="OpenScene navsim_logs/trainval (log pickles)")
    ap.add_argument("--logs", nargs="*", default=None, help="restrict to these log names")
    ap.add_argument("--navtrain-filter", required=True, help="navtrain_filter_v11.json")
    ap.add_argument("--heldout", required=True, help="the held-out guard file (nt_heldout.py build)")
    ap.add_argument("--e1-manifest", default=None, help="E1's token table (INTERFACE.md); else PROVISIONAL ids")
    ap.add_argument("--merge-clock", default=None, help="PhysicalAI clip-clock sidecar to append (eval139 monitor)")
    ap.add_argument("--split", default="nt_train")
    ap.add_argument("--base-commit", default="fb47a58")
    ap.add_argument("--tree", default=None)
    ap.add_argument("--lead", default="anns", choices=LEAD_MODES,
                    help="FOLLOW evidence: 'anns' = BV.lead_for_frame on each log frame's OpenScene anns (default, "
                         "the PhysicalAI release's lead pass); 'none' = the v1 release (lon DECEL/ACCEL/KEEP partial)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    BV = _bv(a.tree)
    BV.assert_no_g_imports()
    out = Path(a.out)
    nav = C.read_navtrain_filter(a.navtrain_filter)
    import nt_heldout as H
    guard = H.load_guard(a.heldout)
    logs_dir = Path(a.logs_dir)
    names = a.logs if a.logs else sorted(p.stem for p in logs_dir.glob("*.pkl"))
    names = [n for n in names if n in nav["logs"]]
    paths = [logs_dir / f"{n}.pkl" for n in names]
    cids = C.read_e1_manifest(a.e1_manifest) if a.e1_manifest else None
    tt = C.build_token_table(paths, nav, set(guard["heldout_logs"]), cids)
    table = tt["rows"]
    H.assert_disjoint([r["log_name"] for r in table], guard, where="nt_labels token table")
    by_log: dict = {}
    for r in table:
        by_log.setdefault(r["log_name"], []).append(r)
    per_token, clip_meta, lead_census = {}, {}, Counter()
    for ln, rows in by_log.items():
        res = labels_for_log(BV, C.load_log(logs_dir / f"{ln}.pkl"), rows, lead=a.lead, census=lead_census)
        per_token.update(res["by_token"])
        clip_meta[ln] = res["clip_meta"]
    lead_policy = ("NONE (no FOLLOW evidence pass: lon DECEL/ACCEL/KEEP are partial {cls, FOLLOW})" if a.lead == "none"
                   else "OpenScene anns per 2 Hz log frame -> build_v9_labels.lead_for_frame (unchanged; E3 class map "
                        f"{C.CLASS_MAP}, ego footprint dropped) at the frame's own time, sample-and-hold onto the "
                        f"builder's 0.1 s band within +-{LEAD_SNAP_S} s; one clip_labels call per token")
    sources = {"lead_mode": a.lead, "lead_policy": lead_policy, "lead_census": dict(sorted(lead_census.items())),
               "logs_dir": str(logs_dir), "n_logs": len(by_log),
               "navtrain_filter": [a.navtrain_filter, C.sha256_file(a.navtrain_filter)],
               "heldout_guard": [a.heldout, C.sha256_file(a.heldout)],
               "e1_manifest": ([a.e1_manifest, C.sha256_file(a.e1_manifest)] if a.e1_manifest else None),
               "clip_id_source": sorted({r["clip_id_source"] for r in table})}
    v9 = write_v9_release(BV, table, per_token, clip_meta, out, a.split, a.base_commit, sources)
    v7 = write_v7(table, per_token, out / f"s2_labels_{a.split}.jsonl.gz", BV)
    clk = write_clock(table, out / "clip_clock_nt.jsonl", a.merge_clock)
    spd = write_speed(table, out / f"speed_label_{a.split}.jsonl", v7["md5"])
    C.json_dump({"rows": table, "census": tt["census"]}, out / "token_table_nt.json")
    # ---- census (the report reads these, never a recomputation) ----
    lat = Counter(int(per_token[r["token"]]["lat_v7id_a"]) for r in table)
    lon = Counter(int(per_token[r["token"]]["lon_v7id"]) for r in table)
    lon9 = Counter(int(per_token[r["token"]]["lon_cls"]) for r in table)
    n_has_lead = sum(int(r["token"] in clip_meta[r["log_name"]]["has_lead_tokens"]) for r in table)
    agree = Counter((int(per_token[r["token"]]["nav_token"]), int(per_token[r["token"]]["nav_token_geom"]))
                    for r in table)
    rep = {"what": "H-NT-DECIDE-1 E2 label release", "wall_s": round(time.time() - t0, 1),
           "n_tokens": len(table), "n_logs": len(by_log), "token_table_census": tt["census"],
           "v9": v9, "v7": v7, "clock": clk, "speed": spd,
           "lat_v7id_a_hist": {(LAT7[k] if k >= 0 else f"IGNORE({k})"): v for k, v in sorted(lat.items())},
           "lon_v7id_hist": {(LON7[k] if k >= 0 else f"IGNORE({k})"): v for k, v in sorted(lon.items())},
           "lon_cls_v9_hist": {(BV.LON9[k] if k >= 0 else f"IGNORE({k})"): v for k, v in sorted(lon9.items())},
           "lead": {"mode": a.lead, "n_tokens_with_lead_data": n_has_lead, "census": sources["lead_census"]},
           "nav_cmd_vs_geom_token": {f"{C.NAV_NAMES8[a_]}|{C.NAV_NAMES8[b_]}": v for (a_, b_), v in sorted(agree.items())},
           "clip_id_source": sources["clip_id_source"]}
    C.json_dump(rep, out / "LABEL_REPORT.json")
    print(json.dumps({k: rep[k] for k in ("n_tokens", "n_logs", "wall_s", "lat_v7id_a_hist", "lon_v7id_hist", "lon_cls_v9_hist", "lead",
                                          "nav_cmd_vs_geom_token")}, indent=1))
    print(f"v9 md5 {v9['md5']}  v7 md5 {v7['md5']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
