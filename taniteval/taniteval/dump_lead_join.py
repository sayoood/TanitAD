#!/usr/bin/env python3
"""Attach ``win["lead"]`` to a BANKED window dump from an `obstacle.offline` agents JOIN.

⭐ WHAT THIS CLOSES. ``driving.py`` retired its stale refusal on 2026-08-18: the lead state
EXISTS program-wide (`obstacle.offline` join → `lead_source` registration → `lead_metrics`,
admitted by the pre-registered D-LEAD-1 control) and ``four_families`` scores the
distance-keeping half of the LONGITUDINAL family wherever ``win["lead"]`` is supplied. What was
missing is the piece that puts ``win["lead"]`` onto an **already-banked** ``windows_<arm>.pt``
without re-inference, from label material that lives OFF the pods. This module is that piece.

RELATIONSHIP TO THE SIBLINGS — three ingest shapes, one pure join, one pure metric:

* ``taniteval/tools/build_lead_block.py`` — POD-side: raw `egomotion`/`obstacle.offline`
  parquet zips + the episode corpus → a standalone positional lead block. Needs the multi-GB
  label zips next to it.
* ``stack/scripts/build_obstacle_join.py`` — POD-side: the same zips → a portable **agents
  JSONL join** (one line per labelled episode frame; agents in the ego frame of that frame's
  own pose; ``frame_idx`` in EPISODE index space; ``t_s`` the registered clip time).
* **THIS MODULE** — DEV-BOX side: a banked dump + that JSONL + the episode ``poses`` →
  ``win["lead"]`` keyed by the dump's own ``eid``, with per-episode coverage. No egomotion
  parquet needed: the join already banked the registration (its ``t_s``), and the composition
  frame chain closes over the episode's own poses.

⛔ WHY NO ``register_poses_to_time`` HERE, AND WHAT REPLACES IT. The jsonl join was built by
``join_clip``, which ran the registration and wrote its result INTO the rows (``frame_idx`` ↔
``t_s``). Re-deriving it would need the egomotion parquet this box does not have. What this
module verifies instead — because an unverified alignment is exactly how a plausible wrong
number gets published — is:

1. **The affine grid.** ``t_s`` must be affine in ``frame_idx`` (the episode grid IS an affine
   reparametrisation of the clip clock; jsonl rounds to 1e-4 s, so a healthy fit residual is
   ≤ 5e-5 s). Fit residual above :data:`MAX_GRID_RESID_S`, or a spacing outside
   lead_source's plausible 0.05–0.5 s band, refuses the episode loudly (``JOIN_GRID_BAD``).
2. **The speed cross-check** — the load-bearing one, and it needs NO labels.
   ``rollout.collect`` persists ``speed`` as ``ep.poses[origin, 3]`` (rollout.py:203), so for a
   correctly-mapped episode ``win["speed"][row] == poses[window_last_indices(T), 3]`` to float32
   exactness. One inequality above :data:`SPEED_TOL_MPS` means the eid→episode mapping, the
   window grid, or the row order is wrong — the episode is refused (``SPEED_MISMATCH``) rather
   than scored against another clip's traffic.

⛔ THE SPAN SENTINELS, because the join's row semantics allow a sharper NO_LABEL/NO_LEAD split
than raw obstacle arrays do. ``join_clip`` emits **no line at all** for a frame outside the
clip's labelled span ("NO_LABEL: no line at all", build_obstacle_join.py:469) — so a row that
EXISTS with ``agents: []`` is a genuinely clear road, not missing labels. ``lead_source
.lead_block`` derives the labelled span from the obs timestamps it is handed; with an
agents-empty stretch that span would collapse and manufacture NO_LABEL out of labelled empty
road. Two non-vehicle SENTINEL entries pinned to the row span's endpoints carry the span
without ever being selectable (``select_lead_causal`` gates on ``is_vehicle``). The bias
direction is preserved: a frame with **no row** stays NO_LABEL, never free flow.

⛔ THREE WINDOW STATES, NEVER TWO — inherited from `lead_source` unchanged: ``LEAD`` /
``NO_LEAD`` / ``NO_LABEL``, and a refused EPISODE (no record, no join, grid or speed mismatch)
leaves its windows ``NO_LABEL`` **with the reason and n in** ``coverage["episodes"]`` — the
binding four-families rule: absence is reported per episode, never silently dropped.

MEASURED coverage fact this module was born under (2026-08-18, probe artifact in
`…/incoming/2026-08-18-dump-lead-wiring/`): the 27 banked dumps are all **val40**
(`physicalai-val-0c5f7dac3b11`), while every local/HF join is over the TRAIN corpus
(`physicalai-train-e438721ae894`): val40 ∩ lead130 = ∅ and val40 ∩ train2308 = ∅ (4-char-prefix
intersections, id convention verified on a v2ep record). So today this module attaches
NO_LABEL-everywhere blocks to val dumps — with the speed cross-check green, which is the
wiring proof — and becomes fully live the moment a **val-corpus jsonl join** is built by the
same `build_obstacle_join.py`.

Run (module CLI):
    python -m taniteval.dump_lead_join --windows results/windows_<key>.pt \
        --agents <agents.jsonl[.xz]> --epdir <episode dir of ep_*.pt> [--out lead_<key>.pt]
"""
from __future__ import annotations

import json
import lzma
from pathlib import Path

import numpy as np

from taniteval import lead_source as ls

#: rollout.WP_STEPS — the tier-0 sparse surface (4 waypoints at 0.5 s nominal spacing).
WP_STEPS = (5, 10, 15, 20)

#: episode statuses. Exhaustive and mutually exclusive; every non-OK carries a ``reason``.
EP_OK = "OK"
EP_NO_RECORD = "NO_EPISODE_RECORD"
EP_BAD_POSES = "BAD_POSES"
EP_GRID_MISMATCH = "GRID_MISMATCH"
EP_SPEED_MISMATCH = "SPEED_MISMATCH"
EP_NO_JOIN = "NO_JOIN"
EP_AMBIGUOUS = "AMBIGUOUS_PREFIX"
EP_JOIN_GRID_BAD = "JOIN_GRID_BAD"

#: ``win["speed"]`` is a float32 COPY of ``poses[origin, 3]`` (rollout.py:203); any real
#: mismatch is a mapping error, not noise. 1e-3 m/s absorbs float32 round-tripping only.
SPEED_TOL_MPS = 1e-3
#: jsonl ``t_s`` is rounded to 1e-4 s; an affine fit residual above this is not rounding.
MAX_GRID_RESID_S = 2e-3
#: ⚠️ added to every query time. ``lead_track_in_window``/``select_lead_causal`` take the LAST
#: sample ≤ t; a query rebuilt from the affine fit can land ~5e-5 s BELOW the intended row's
#: ROUNDED ``t_s`` and silently pick the frame before — a one-frame-stale lead (~v·0.1 s of
#: displacement) on about half the queries, purely from rounding. MEASURED in this module's own
#: hand-computable fixture before the epsilon existed. 2e-3 s clears the rounding band (≤2e-4)
#: by 10x and stays 50x below the ~0.1007 s grid step, so it can never reach the NEXT row.
QUERY_EPS_S = 2e-3
#: never selectable (``is_vehicle`` False); exists only to carry the labelled span.
SENTINEL_TRACK = "__labelled_span__"

__all__ = [
    "WP_STEPS", "EP_OK", "EP_NO_RECORD", "EP_BAD_POSES", "EP_GRID_MISMATCH",
    "EP_SPEED_MISMATCH", "EP_NO_JOIN", "EP_AMBIGUOUS", "EP_JOIN_GRID_BAD",
    "SPEED_TOL_MPS", "MAX_GRID_RESID_S", "QUERY_EPS_S", "SENTINEL_TRACK",
    "read_agents_jsonl", "episodes_from_epdir", "episodes_from_v2ep_dir",
    "coverage_probe", "attach_lead",
]


# --------------------------------------------------------------------------- #
# 1. the agents JSONL join (build_obstacle_join.py's output format)             #
# --------------------------------------------------------------------------- #
def read_agents_jsonl(path, clips=None) -> dict:
    """-> ``{clip_id: {"frame_idx", "t_s", "obs", "n_rows"}}`` from an agents jsonl(.xz).

    One pass, line-streamed (the full train join is ~GB-scale). ``clips`` (a set of clip ids)
    restricts what is kept in memory; every line is still read so a truncated file cannot pass
    for a filtered one. ``obs`` is the equal-length-array dict `lead_source.select_lead_causal`
    takes; per-cuboid ``t`` is the ROW's ``t_s`` (join_clip snapped each cuboid to the frame,
    within its match tolerance — the row time IS the sample's frame time).
    """
    path = Path(path)
    keep = None if clips is None else {str(c) for c in clips}
    acc: dict[str, dict] = {}
    op = lzma.open if path.suffix == ".xz" else open
    with op(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cid = str(rec["clip_id"])
            if keep is not None and cid not in keep:
                continue
            c = acc.setdefault(cid, {"frame_idx": [], "t_s": [], "_ag": []})
            c["frame_idx"].append(int(rec["frame_idx"]))
            c["t_s"].append(float(rec["t_s"]))
            for a in rec.get("agents", ()):
                c["_ag"].append((float(rec["t_s"]), str(a["track_id"]),
                                 float(a["cx"]), float(a["cy"]),
                                 float(a["l"]),
                                 str(a.get("cls", "")) in ls.VEHICLE_CLASSES))
    out = {}
    for cid, c in acc.items():
        ag = c.pop("_ag")
        obs = {
            "t": np.array([a[0] for a in ag], dtype=np.float64),
            "track": np.array([a[1] for a in ag], dtype=object),
            "center_x": np.array([a[2] for a in ag], dtype=np.float64),
            "center_y": np.array([a[3] for a in ag], dtype=np.float64),
            "size_x": np.array([a[4] for a in ag], dtype=np.float64),
            "is_vehicle": np.array([a[5] for a in ag], dtype=bool),
        }
        out[cid] = {"frame_idx": np.asarray(c["frame_idx"], dtype=np.int64),
                    "t_s": np.asarray(c["t_s"], dtype=np.float64),
                    "obs": obs, "n_rows": len(c["frame_idx"])}
    return out


def _affine_fit(idx, t):
    """Least-squares ``t = a + b*idx`` + max abs residual. The episode grid is affine by
    construction (linspace over the clip span), so a large residual means the rows are not an
    episode grid at all — wrong file, wrong clip, or a format change."""
    idx = np.asarray(idx, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    if idx.size < 2:
        return float("nan"), float("nan"), float("inf")
    A = np.column_stack([np.ones(idx.size), idx])
    coef, *_ = np.linalg.lstsq(A, t, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    return a, b, float(np.max(np.abs(a + b * idx - t)))


def _with_span_sentinels(join_clip_rec: dict) -> dict:
    """obs + two non-vehicle sentinels at the ROW span's endpoints.

    join_clip's contract: a row EXISTS iff the frame is labelled — so the labelled span is the
    row span, not the agent-occurrence span. Without the sentinels a labelled-but-empty road
    would read NO_LABEL (conservative but wrong); with them it reads NO_LEAD. A frame with no
    row stays outside the span → NO_LABEL, so the forbidden direction (missing labels counted
    as free flow) remains impossible."""
    t_rows = np.asarray(join_clip_rec["t_s"], dtype=np.float64)
    o = join_clip_rec["obs"]
    lo, hi = float(t_rows.min()), float(t_rows.max())
    return {
        "t": np.concatenate([o["t"], [lo, hi]]),
        "track": np.concatenate([o["track"],
                                 np.array([SENTINEL_TRACK] * 2, dtype=object)]),
        "center_x": np.concatenate([o["center_x"], [1e9, 1e9]]),
        "center_y": np.concatenate([o["center_y"], [1e9, 1e9]]),
        "size_x": np.concatenate([o["size_x"], [0.0, 0.0]]),
        "is_vehicle": np.concatenate([o["is_vehicle"], [False, False]]),
    }


# --------------------------------------------------------------------------- #
# 2. episode records — the two cache shapes that exist                          #
# --------------------------------------------------------------------------- #
def _decode_packed(v) -> str | None:
    """Big-endian ASCII unpacking of an ``episode_id`` int (the id4 defect's encoding —
    same guard as ``rollout._unpack_ascii``: small ints are genuine indices, left alone)."""
    try:
        iv = int(v)
    except (TypeError, ValueError):
        return None
    if iv < (1 << 24):
        return None
    b = iv.to_bytes(8, "big").lstrip(b"\x00")
    try:
        s = b.decode("ascii")
    except UnicodeDecodeError:
        return None
    return s if s.isprintable() else None


def episodes_from_epdir(epdir) -> dict:
    """``ep_*.pt`` (episode contract: poses [T,4] + episode_id) → ``{index: rec}``.

    Keys are the FILE-SORT index, which is ``collect``'s enumeration order and therefore the
    canonical val-list eid (normalise_eid's first-appearance rank reproduces it — measured by
    ``test_eid_normalisation``). ``id4`` is the decoded 4-char episode id when the record's
    ``episode_id`` is a packed-ASCII int (MEASURED on a v2ep record 2026-08-18: the packed id
    is the clip UUID's first 4 chars — 0045da77-… ↔ 808465461 == b"0045")."""
    import torch
    out = {}
    for i, p in enumerate(sorted(Path(epdir).glob("ep_*.pt"))):
        d = torch.load(str(p), map_location="cpu", weights_only=False)
        poses = np.asarray(d["poses"], dtype=np.float64)
        out[i] = {"poses": poses, "clip_id": None,
                  "id4": _decode_packed(d.get("episode_id")),
                  "episode_id": d.get("episode_id"), "file": p.name}
    return out


def episodes_from_v2ep_dir(v2ep_dir, clips=None) -> dict:
    """``<clip_uuid>.v2ep.pt`` records → ``{clip_id: rec}``, poses FRONT-TRIMMED by
    ``n_stack - 1`` (the raw v2ep carries the stack warm-up frames; the episode contract —
    and the jsonl join's ``frame_idx`` space — is the post-trim grid,
    build_obstacle_join.py:443-444 / physicalai.py ``poses[k:n]``)."""
    import torch
    keep = None if clips is None else {str(c) for c in clips}
    out = {}
    for p in sorted(Path(v2ep_dir).glob("*.v2ep.pt")):
        cid = p.name[:-len(".v2ep.pt")]
        if keep is not None and cid not in keep:
            continue
        d = torch.load(str(p), map_location="cpu", weights_only=False)
        trim = max(int(d.get("n_stack", 1)) - 1, 0)
        poses = np.asarray(d["poses"], dtype=np.float64)[trim:]
        out[cid] = {"poses": poses, "clip_id": str(d.get("clip_id", cid)),
                    "id4": cid[:4], "episode_id": d.get("episode_id"),
                    "file": p.name, "n_stack_trim": trim}
    return out


def _resolve_clip(rec: dict, join_clips) -> tuple[str | None, str, list]:
    """-> ``(clip_id or None, method, candidates)``. An explicit ``clip_id`` wins; else the
    unique join clip whose UUID starts with ``id4``. ≥2 candidates is REFUSED (ambiguous), not
    guessed — a 4-char prefix over thousands of clips can collide."""
    cid = rec.get("clip_id")
    if cid:
        return (str(cid), "explicit", [str(cid)])
    id4 = rec.get("id4")
    if not id4:
        return (None, "no_identity", [])
    cand = sorted(c for c in join_clips if str(c).startswith(str(id4)))
    if len(cand) == 1:
        return (cand[0], "id4_prefix", cand)
    if len(cand) > 1:
        return (None, "ambiguous", cand)
    return (None, "id4_prefix", [])


def coverage_probe(episodes: dict, joins: dict) -> dict:
    """Cheap identity intersection — which dump episodes COULD the join cover, before any
    label work. This is the two-location absence probe in callable form."""
    rows = {}
    n_hit = 0
    for k, rec in episodes.items():
        cid, method, cand = _resolve_clip(rec, joins)
        hit = bool(cid is not None and cid in joins)
        n_hit += int(hit)
        rows[str(k)] = {"id4": rec.get("id4"), "clip_id": rec.get("clip_id"),
                        "matched": hit, "method": method,
                        "n_candidates": len(cand)}
    return {"n_episodes": len(episodes), "n_join_clips": len(joins),
            "n_matched": n_hit, "episodes": rows}


# --------------------------------------------------------------------------- #
# 3. the attach                                                                #
# --------------------------------------------------------------------------- #
def _key(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return str(v)


def attach_lead(win: dict, episodes: dict, joins: dict, *,
                wp_steps=WP_STEPS, speed_tol: float = SPEED_TOL_MPS,
                n_boot: int = 2000, seed: int = 0) -> dict:
    """Build the ``win["lead"]`` block for a banked dump, in DUMP ROW ORDER.

    Args:
        win:      the loaded ``windows_<key>.pt`` dict. ``eid`` must be the canonical ids
                  (``rollout.load_windows`` normalises the three packed-ASCII dumps; a raw
                  ``torch.load`` of those three feeds packed ints and every episode will
                  correctly refuse as ``NO_EPISODE_RECORD`` — loud, not wrong).
        episodes: ``{eid_key: {"poses": [T,4], "clip_id": str|None, "id4": str|None}}`` —
                  from :func:`episodes_from_epdir` / :func:`episodes_from_v2ep_dir` or built
                  by the caller. Keys must be the values ``win["eid"]`` takes.
        joins:    :func:`read_agents_jsonl`'s output.

    Returns the ``lead`` dict `four_families.longitudinal` consumes (``leads``/``lead_lens``/
    ``speeds``/``state``/``eid`` …) plus ``coverage`` with a per-episode status/reason — the
    binding rule's per-family absence reporting, at the episode grain. Attach it as
    ``win["lead"] = attach_lead(...)`` and every downstream ``four_families.four_families(win)``
    call scores distance-keeping on exactly the windows it is defensible on.
    """
    eid_list = list(win["eid"])
    w = len(eid_list)
    k = len(tuple(wp_steps))
    steps = np.asarray(tuple(wp_steps), dtype=np.float64)
    if steps.size < 2 or len(set(np.diff(steps))) != 1:
        raise ValueError(f"wp_steps must be >=2 uniformly spaced ticks, got {tuple(wp_steps)}")

    leads = np.full((w, k, 2), np.nan)
    lead_lens = np.full(w, np.nan)
    speeds = np.full(w, np.nan)
    state = np.array([ls.NO_LABEL] * w, dtype=object)
    gap0 = np.full(w, np.nan)
    dump_speed = (np.asarray(win["speed"], dtype=np.float64).reshape(-1)
                  if win.get("speed") is not None else None)
    if dump_speed is not None:
        if dump_speed.size != w:
            raise ValueError(f"win['speed'] has {dump_speed.size} entries for {w} windows")
        speeds[:] = dump_speed

    ep_keys = {_key(kk): kk for kk in episodes}
    eid_norm = np.array([_key(e) for e in eid_list], dtype=object)
    cov_eps: dict[str, dict] = {}
    b_list: list[float] = []

    def _refuse(rows, cov, status, reason):
        cov["status"] = status
        cov["reason"] = reason
        cov["counts"] = {ls.LEAD: 0, ls.NO_LEAD: 0, ls.NO_LABEL: int(len(rows))}

    for e in dict.fromkeys(eid_norm.tolist()):
        rows = np.flatnonzero(eid_norm == e)
        cov: dict = {"n_windows": int(rows.size)}
        cov_eps[str(e)] = cov
        rec = episodes.get(ep_keys.get(e)) if e in ep_keys else None
        if rec is None:
            _refuse(rows, cov, EP_NO_RECORD,
                    f"no episode record for eid {e!r} — cannot place its "
                    f"{rows.size} windows on any clip")
            continue
        poses = np.asarray(rec["poses"], dtype=np.float64)
        if poses.ndim != 2 or poses.shape[1] < 4:
            _refuse(rows, cov, EP_BAD_POSES,
                    f"poses must be [T,>=4] (x,y,yaw,v — physicalai.py:144), "
                    f"got {poses.shape}")
            continue
        t_len = int(poses.shape[0])
        origins = ls.window_last_indices(t_len)
        cov["T"] = t_len
        if origins.size != rows.size:
            _refuse(rows, cov, EP_GRID_MISMATCH,
                    f"dump has {rows.size} windows for this eid but "
                    f"window_last_indices(T={t_len}) gives {origins.size} — "
                    f"wrong episode record or a non-collect window grid")
            continue
        # --- the label-free alignment proof: dump speed IS poses[origin, 3] --- #
        if dump_speed is not None:
            d = float(np.max(np.abs(dump_speed[rows] - poses[origins, 3])))
            cov["speed_check_max_mps"] = round(d, 6)
            if d > float(speed_tol):
                _refuse(rows, cov, EP_SPEED_MISMATCH,
                        f"max |win['speed'] - poses[origin,3]| = {d:.4f} m/s > "
                        f"{speed_tol} — eid→episode mapping, grid or row order "
                        f"is wrong; refusing to place another clip's traffic "
                        f"on these windows")
                continue
        else:
            cov["speed_check_max_mps"] = None
            cov["speed_check_note"] = ("dump carries no 'speed'; the label-free "
                                       "alignment proof could not run")
        clip, method, cand = _resolve_clip(rec, joins)
        cov["clip_id"] = clip
        cov["matched_by"] = method
        if method == "ambiguous":
            _refuse(rows, cov, EP_AMBIGUOUS,
                    f"id4 {rec.get('id4')!r} matches {len(cand)} join clips "
                    f"({cand[:4]}…) — refusing to guess; supply an explicit "
                    f"clip_id")
            continue
        if clip is None or clip not in joins:
            _refuse(rows, cov, EP_NO_JOIN,
                    f"clip for eid {e!r} (id4 {rec.get('id4')!r}) is not in the "
                    f"agents join ({len(joins)} clips) — no obstacle labels "
                    f"available here; NEVER counted as free flow")
            continue
        jn = joins[clip]
        if int(np.max(jn["frame_idx"])) >= t_len:
            _refuse(rows, cov, EP_GRID_MISMATCH,
                    f"join frame_idx max {int(np.max(jn['frame_idx']))} >= "
                    f"T {t_len} — the join was built on a different trim of "
                    f"this episode")
            continue
        a, b, resid = _affine_fit(jn["frame_idx"], jn["t_s"])
        cov["grid"] = {"a": round(a, 6), "b": round(b, 6),
                       "max_resid_s": round(resid, 6)}
        if not np.isfinite(resid) or resid > MAX_GRID_RESID_S \
                or not (0.05 <= b <= 0.5):
            _refuse(rows, cov, EP_JOIN_GRID_BAD,
                    f"join rows are not a plausible episode grid (b={b!r}, "
                    f"max residual {resid!r} s) — wrong clip or format change")
            continue
        t_grid = a + b * np.arange(t_len, dtype=np.float64)
        ego = {"t": t_grid, "x": poses[:, 0], "y": poses[:, 1],
               "yaw": poses[:, 2], "v": poses[:, 3]}
        # QUERY_EPS_S: keep every query strictly ABOVE its row's rounded t_s so
        # the causal last-sample rule picks the intended frame, never the one
        # before (see the constant's docstring for the measured failure).
        blk = ls.lead_block(t_grid[origins] + QUERY_EPS_S, b * steps,
                            _with_span_sentinels(jn), ego)
        leads[rows] = blk["leads"]
        lead_lens[rows] = blk["lead_lens"]
        state[rows] = blk["state"]
        gap0[rows] = blk["gap0_m"]
        if dump_speed is None:
            speeds[rows] = blk["speeds"]
        b_list.append(b)
        cov["status"] = EP_OK
        cov["counts"] = dict(blk["counts"])
        cov["label_span_s"] = blk["label_span_s"]
        cov["n_join_rows"] = int(jn["n_rows"])

    counts = {s: int((state == s).sum()) for s in (ls.LEAD, ls.NO_LEAD, ls.NO_LABEL)}
    b_med = float(np.median(b_list)) if b_list else None
    step_gap = float(steps[1] - steps[0])
    out = {
        "leads": leads, "lead_lens": lead_lens, "speeds": speeds,
        "has_lead": state == ls.LEAD, "state": state, "gap0_m": gap0,
        "eid": eid_list,
        "wp_steps": [int(s) for s in steps],
        "ts_rel_s": (None if b_med is None else
                     np.round(b_med * steps, 6)),
        "dt_s": None if b_med is None else round(b_med * step_gap, 6),
        "n_boot": int(n_boot), "seed": int(seed),
        "counts": counts,
        "coverage": {
            "n_windows": w,
            "n_episodes": len(cov_eps),
            "n_episodes_ok": sum(1 for c in cov_eps.values()
                                 if c.get("status") == EP_OK),
            "n_windows_labelled": counts[ls.LEAD] + counts[ls.NO_LEAD],
            "n_windows_lead": counts[ls.LEAD],
            "n_windows_no_label": counts[ls.NO_LABEL],
            "grid_dt_s_median": b_med,
            "episodes": cov_eps,
            "note": ("a refused episode (any non-OK status) leaves its windows "
                     "NO_LABEL with the reason above — reported, never scored, "
                     "never counted as free flow"),
        },
        "conventions": {
            "frame": "window-origin ego frame at t0; x forward, y left, metres "
                     "(lead_source, unchanged)",
            "gap": "along - size_x/2 (rig origin to lead REAR face); NOT "
                   "bumper-to-bumper",
            "selection": f"strictly causal via lead_source.select_lead_causal; "
                         f"classes {ls.VEHICLE_CLASSES}",
            "span": "labelled span = the join's ROW span (join_clip emits no "
                    "line outside the obstacle span), carried by two "
                    "non-vehicle sentinels; a frame with no row is NO_LABEL",
            "time_base": "t_s banked by build_obstacle_join's registration; "
                         "re-verified here as affine in frame_idx and by the "
                         "dump-speed cross-check",
            "occlusion": "no occ filter — same convention as the D-LEAD-1 "
                         "admitted instrument (lead_state_gate has none)",
        },
    }
    return out


# --------------------------------------------------------------------------- #
# 4. CLI                                                                       #
# --------------------------------------------------------------------------- #
def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser("taniteval.dump_lead_join")
    ap.add_argument("--windows", required=True,
                    help="banked windows_<key>.pt (loaded via rollout."
                         "load_windows, so packed eids are normalised)")
    ap.add_argument("--agents", required=True, help="agents jsonl or jsonl.xz")
    ap.add_argument("--epdir", required=True,
                    help="episode dir of ep_*.pt (poses+episode_id contract)")
    ap.add_argument("--clip-map", default=None,
                    help="optional json {eid: clip_uuid} overriding id4 matching")
    ap.add_argument("--out", default=None,
                    help="torch.save the lead block here (also writes "
                         "<out>.coverage.json)")
    ap.add_argument("--probe", action="store_true",
                    help="identity intersection only — no label work")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    import torch
    from taniteval.rollout import load_windows
    win = load_windows(a.windows)
    episodes = episodes_from_epdir(a.epdir)
    if a.clip_map:
        cm = json.loads(Path(a.clip_map).read_text())
        for kk, cid in cm.items():
            key = _key(kk)
            if key in episodes:
                episodes[key]["clip_id"] = str(cid)
    joins = read_agents_jsonl(a.agents)
    if a.probe:
        print(json.dumps(coverage_probe(episodes, joins), indent=2))
        return
    lead = attach_lead(win, episodes, joins, n_boot=a.n_boot, seed=a.seed)
    cov = lead["coverage"]
    print(f"[dump_lead_join] {Path(a.windows).name}: {cov['n_windows']} windows/"
          f"{cov['n_episodes']} eps · OK eps {cov['n_episodes_ok']} · "
          f"LEAD {lead['counts'][ls.LEAD]} · NO_LEAD {lead['counts'][ls.NO_LEAD]}"
          f" · NO_LABEL {lead['counts'][ls.NO_LABEL]}", flush=True)
    for e, c in cov["episodes"].items():
        if c.get("status") != EP_OK:
            print(f"  ep {e}: {c.get('status')} — {c.get('reason', '')[:110]}",
                  flush=True)
    if a.out:
        torch.save(lead, a.out)
        Path(a.out + ".coverage.json").write_text(
            json.dumps({"coverage": cov, "counts": lead["counts"],
                        "conventions": lead["conventions"]},
                       indent=2, default=str))
        print(f"[dump_lead_join] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
