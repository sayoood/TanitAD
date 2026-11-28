"""H-NT-DECIDE-1 E2: the FOLLOW-evidence (lead) pass of ``nt_labels.py`` (added 2026-10-21 ~23:55).

Why it exists (MEASURED on the provisional A1 release, 99,824 tokens): v1 called the v9 builder with ``lead=None``,
so every DECEL / ACCEL / KEEP row shipped as a partial {cls, FOLLOW} label and the hard ``lon_cls`` was IGNORE on
74.15 % of rows, against 30.86 % on refcv8's PhysicalAI release (909,651 rows) -- the singleton allowed sets {FOLLOW},
{ACCELERATE}, {KEEP} (12.6 / 16.8 / 21.2 % on PhysicalAI) never occurred.

Pins (expectations are LITERALS; the reference for the identity test is a FROZEN copy of the v1 code path):
  * the lead parser == E3's ``nt_joins.agents_of_frame`` on (cx, cy, l, cls), on real OpenScene smoke logs;
  * ``lead="none"`` per-token calls are bit-identical, field by field, to v1's single per-log call (real logs);
  * analytic: straight 8 m/s with EMPTY anns -> KEEP (hard, allowed = {KEEP}, v7 CRUISE); a vehicle 15 m ahead at
    the same speed -> FOLLOW (v7 FOLLOW), gap ~ 15 - 4.5/2 - 3.6 m; the same on a LEFT circle R = 40 m;
    NO anns at all -> partial (no data is not "no lead");
  * mutations that must go RED: the v1 defect (``lead="none"``) on the FOLLOW case, and the lead's ``cy`` sign
    flipped on the circle (the y-left / y-right convention defect).
"""
from __future__ import annotations

import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
import nt_common as C  # noqa: E402
import nt_labels as NL  # noqa: E402

TREE = os.environ.get("NT_TREE") or str(CODE.parent / "nt_tree")
SMOKE = CODE.parent / "smoke_data" / "navsim_logs"
KEEP, FOLLOW = 6, 3                    # build_v9_labels.LON9 indices (literals)
V7_FOLLOW, V7_CRUISE = 0, 1            # build_v9_labels.LON7 indices (literals)


@pytest.fixture(scope="module")
def BV():
    bv = NL._bv(TREE)
    assert bv.LON9[KEEP] == "KEEP" and bv.LON9[FOLLOW] == "FOLLOW"
    assert bv.LON7[V7_FOLLOW] == "FOLLOW" and bv.LON7[V7_CRUISE] == "CRUISE"
    return bv


def _smoke_logs(n):
    ps = sorted(SMOKE.glob("*.pkl"))
    if len(ps) < n:
        pytest.skip(f"needs {n} OpenScene smoke logs under {SMOKE}")
    return ps[:n]


# ---------------------------------------------------------------------------------------------- the v1 path, FROZEN
def _labels_v1(BV, frames, rows):
    """``nt_labels.labels_for_log`` as shipped in the v1 provisional release (one call per log, ``lead=None``)."""
    ea = C.ego_arrays(frames)
    log = BV.log_from_arrays(ea["t_s"], ea["x"], ea["y"], ea["yaw"], ea["v"], None, ea["rev"])
    G = BV.clip_geo(log, None)
    rows = sorted(rows, key=lambda r: r["frame_idx"])
    now_k = np.array([float(ea["t_s"][r["frame_idx"]]) for r in rows], dtype=np.float64)
    k = np.full(len(rows), C.NOW_RAW_ROW, dtype=np.int64)
    F = BV.clip_labels(G, now_k, k, None, None, None, lc_labels=False)
    out = {}
    for i, r in enumerate(rows):
        d = {nm: np.asarray(col)[i] for nm, col in F.items()}
        nav = C.nav_token_of_driving_command(ea["driving_command"][r["frame_idx"]])
        d["nav_token_geom"] = np.float64(d["nav_token"])
        d["nav_token"] = np.float64(nav["nav_token"])
        d["navsim_cmd_argmax"] = np.int8(nav["navsim_argmax"])
        d["log_now_s"] = np.float64(now_k[i])
        out[r["token"]] = d
    return out


def _rows_of(frames, step=3):
    return [{"token": str(f["token"]), "frame_idx": i} for i, f in enumerate(frames) if i >= 4 and i % step == 0]


LEAD_CHANGES_ONLY = ("lon_", "lead_", "goal_", "v_lo_ms", "v_hi_ms")      # literal: what FOLLOW evidence may move


def test_per_token_fields_equal_the_v1_call_wherever_both_write(BV):
    """The per-token call (n = 1, k = the token's 0.1 s tick) is the SAME computation as v1's per-log call for every
    column it writes, when no lead is given: rows are independent (rec = None, lc = None)."""
    n_tok = n_val = 0
    for p in _smoke_logs(3):
        frames = C.load_log(p)
        rows = sorted(_rows_of(frames), key=lambda r: r["frame_idx"])
        ea = C.ego_arrays(frames)
        log = BV.log_from_arrays(ea["t_s"], ea["x"], ea["y"], ea["yaw"], ea["v"], None, ea["rev"])
        G = BV.clip_geo(log, None)
        now_k = np.array([float(ea["t_s"][r["frame_idx"]]) for r in rows])
        F1 = BV.clip_labels(G, now_k, np.full(len(rows), C.NOW_RAW_ROW, dtype=np.int64), None, None, None,
                            lc_labels=False)
        for i in range(len(rows)):
            d = NL.token_fields(BV, G, float(now_k[i]), int(round(now_k[i] / 0.1)), None)
            assert set(d) <= set(F1), sorted(set(d) - set(F1))
            for nm, v in d.items():
                a, b = np.asarray(F1[nm])[i], np.asarray(v)
                assert a.dtype == b.dtype and np.array_equal(a, b, equal_nan=a.dtype.kind == "f"), (p.name, i, nm, a, b)
                n_val += 1
            n_tok += 1
    assert n_tok >= 30 and n_val >= 30 * 40, (n_tok, n_val)


def test_lead_anns_changes_only_lon_lead_goal_fields_on_real_logs(BV):
    """On real OpenScene logs: the FOLLOW evidence moves ONLY the longitudinal / lead / goal columns (a literal prefix
    set), and it moves SOME of them (the pass is live, not a no-op)."""
    changed, n_tok, n_lon_changed = set(), 0, 0
    for p in _smoke_logs(4):
        frames = C.load_log(p)
        rows = _rows_of(frames, step=2)
        a = NL.labels_for_log(BV, frames, rows, lead="none")["by_token"]
        b = NL.labels_for_log(BV, frames, rows, lead="anns")["by_token"]
        for tok in a:
            assert set(a[tok]) <= set(b[tok])
            for nm in b[tok]:
                if nm not in a[tok]:
                    changed.add(nm)
                    continue
                x, y = np.asarray(a[tok][nm]), np.asarray(b[tok][nm])
                if not np.array_equal(x, y, equal_nan=x.dtype.kind == "f"):
                    changed.add(nm)
            n_lon_changed += int(int(a[tok]["lon_cls"]) != int(b[tok]["lon_cls"]))
            n_tok += 1
    bad = sorted(nm for nm in changed if not nm.startswith(LEAD_CHANGES_ONLY))
    assert not bad, bad
    assert n_tok >= 60 and n_lon_changed >= 1, (n_tok, n_lon_changed, sorted(changed))


def test_per_token_call_is_identical_to_v1_without_lead(BV, tmp_path):
    """The CONTRACT is the release npz the trainer reads, not the per-token dicts: the builder creates conditional
    columns lazily (``stop_x_m`` exists only where some row stopped), so v1's one-call-per-log dicts carry them for
    every row (builder fill) and the per-token dicts only where hit (``write_v9_release``'s ``_fill_like``). Both
    paths are written through the SAME writer and compared array by array, bit for bit."""
    table, ref, got, meta = [], {}, {}, {}
    for p in _smoke_logs(3):
        frames = C.load_log(p)
        ln = str(frames[0]["log_name"])
        rows = _rows_of(frames)
        ref.update(_labels_v1(BV, frames, rows))
        res = NL.labels_for_log(BV, frames, rows, lead="none")
        got.update(res["by_token"])
        assert not res["clip_meta"]["has_lead_tokens"]
        meta[ln] = res["clip_meta"]
        table += [{"token": r["token"], "log_name": ln, "sid": len(table) + i, "clip_id": f"navtrain-{r['token']}"}
                  for i, r in enumerate(rows)]
    assert set(ref) == set(got) and len(table) >= 30, len(table)
    za, zb = {}, {}
    for tag, per, z in (("v1", ref, za), ("v2", got, zb)):
        rel = NL.write_v9_release(BV, table, per, meta, tmp_path / tag, "t", "test", {})
        with np.load(rel["npz"], allow_pickle=False) as f:
            z.update({k: f[k] for k in f.files})
    assert sorted(za) == sorted(zb)
    n_cols = 0
    for k in za:
        a, b = za[k], zb[k]
        assert a.dtype == b.dtype and a.shape == b.shape, (k, a.dtype, b.dtype, a.shape, b.shape)
        assert np.array_equal(a, b, equal_nan=a.dtype.kind == "f"), (k, np.flatnonzero(a != b)[:5])
        n_cols += 1
    assert n_cols >= 80, n_cols


def test_lead_parser_equals_e3_agents_of_frame(BV):
    import nt_joins as J
    from tanitad.data.join_label_hygiene import is_ego_footprint
    n_frames = n_agents = 0
    for p in _smoke_logs(2):
        for fr in C.load_log(p):
            try:
                e3 = J.agents_of_frame(fr, 0.0, Counter(), is_ego_footprint)
            except C.NTRefused:
                continue
            mine = NL.lead_agents_of_frame(fr, is_ego_footprint, Counter())
            assert mine is not None
            assert [(a["cx"], a["cy"], a["l"], a["cls"]) for a in e3] == \
                   [(a["cx"], a["cy"], a["l"], a["cls"]) for a in mine]
            n_frames += 1
            n_agents += len(mine)
    assert n_frames >= 100 and n_agents >= 100, (n_frames, n_agents)


# ---------------------------------------------------------------------------------------------- analytic logs
def _pose(kind, s, R):
    if kind == "straight":
        return s, 0.0, 0.0
    th = s / R
    return R * math.sin(th), R * (1.0 - math.cos(th)), th


def _synth(kind, *, v=8.0, R=40.0, n=60, lead_dx=None, anns=True, flip_cy=False):
    """A 2 Hz OpenScene-layout log at constant ``v``; ``lead_dx`` puts ONE vehicle that many metres AHEAD along the
    ego's own path, moving with it, expressed in each frame's ego frame (x fwd, y LEFT)."""
    frs = []
    for i in range(n):
        t = 0.5 * i
        x, y, yaw = _pose(kind, v * t, R)
        fr = {"timestamp": int(1_000_000 + round(t * 1e6)), "token": f"syn{i:04d}", "log_name": "syn",
              "ego2global_translation": [x, y, 0.0],
              "ego2global_rotation": [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)],
              "ego_dynamic_state": [v, 0.0, 0.0, 0.0], "driving_command": [0, 1, 0, 0]}
        if anns:
            boxes, names = [], []
            if lead_dx is not None:
                lx, ly, lyaw = _pose(kind, v * t + lead_dx, R)
                dx, dy = lx - x, ly - y
                cx = math.cos(yaw) * dx + math.sin(yaw) * dy
                cy = -math.sin(yaw) * dx + math.cos(yaw) * dy
                boxes.append([cx, -cy if flip_cy else cy, 0.8, 4.5, 1.9, 1.6, lyaw - yaw])
                names.append("vehicle")
            fr["anns"] = {"gt_boxes": np.asarray(boxes, dtype=np.float64).reshape(-1, 7), "gt_names": names,
                          "track_tokens": ["0123456789abcdef"] * len(names)}
        frs.append(fr)
    return frs


def _label(BV, frs, lead="anns", idx=10):
    out = NL.labels_for_log(BV, frs, [{"token": frs[idx]["token"], "frame_idx": idx}], lead=lead)
    return out["by_token"][frs[idx]["token"]]


def test_straight_no_lead_is_hard_KEEP(BV):
    d = _label(BV, _synth("straight"))
    assert int(d["lon_cls"]) == KEEP
    assert int(d["lon_allowed"]) == 1 << KEEP
    assert int(d["lon_v7id"]) == V7_CRUISE


def test_straight_lead_15m_ahead_is_FOLLOW(BV):
    d = _label(BV, _synth("straight", lead_dx=15.0))
    assert int(d["lon_cls"]) == FOLLOW
    assert int(d["lon_v7id"]) == V7_FOLLOW
    assert int(d["lead_valid"]) == 1
    assert abs(float(d["lead_gap_m"]) - (15.0 - 4.5 / 2 - 3.6)) <= 0.45       # the 10 Hz path's 0.8 m spacing / 2


def test_left_circle_lead_on_the_arc_is_FOLLOW(BV):
    d = _label(BV, _synth("circle", lead_dx=15.0))
    assert int(d["lon_cls"]) == FOLLOW
    assert int(d["lat_v7id_a"]) == 6                                            # TURN_L: the circle is LEFT


def test_no_anns_is_no_data_not_no_lead(BV):
    d = _label(BV, _synth("straight", anns=False))
    assert int(d["lon_cls"]) == -100
    assert int(d["lon_allowed"]) == (1 << KEEP) | (1 << FOLLOW)


def test_MUT_v1_defect_lead_none_loses_the_FOLLOW_label(BV):
    d = _label(BV, _synth("straight", lead_dx=15.0), lead="none")
    assert int(d["lon_cls"]) != FOLLOW                       # the deliberate regression must NOT read FOLLOW
    assert int(d["lon_cls"]) == -100


def test_MUT_lead_cy_sign_flipped_on_the_circle_is_not_FOLLOW(BV):
    d = _label(BV, _synth("circle", lead_dx=15.0, flip_cy=True))
    assert int(d["lon_cls"]) == KEEP                         # off the path by ~2 x 2.8 m > the 1.75 m corridor


def test_unknown_lead_mode_refuses(BV):
    with pytest.raises(C.NTRefused):
        NL.labels_for_log(BV, _synth("straight"), [{"token": "syn0010", "frame_idx": 10}], lead="maybe")


def test_k2_control_synth_reads_the_lead_literals_and_its_mutation_goes_red(BV):
    """K2's own synthetic logs (``nt_controls._synth_log``) through the production path: the three lead literals,
    and the mutation arm's flipped-cy circle must NOT read FOLLOW."""
    import nt_controls as K
    got = {}
    for kind, sk, R, ld, flip in (("straight_empty_anns", "straight", 20.0, None, False),
                                  ("straight_lead15", "straight", 20.0, 15.0, False),
                                  ("circle_lead15", "circle", 40.0, 15.0, False),
                                  ("circle_lead15_MUT", "circle", 40.0, 15.0, True)):
        frs = K._synth_log(sk, R=R, anns=True, lead_dx=ld, flip_cy=flip)
        got[kind] = int(_label(BV, frs)["lon_v7id"])
    assert got == {"straight_empty_anns": V7_CRUISE, "straight_lead15": V7_FOLLOW, "circle_lead15": V7_FOLLOW,
                   "circle_lead15_MUT": V7_CRUISE}, got
