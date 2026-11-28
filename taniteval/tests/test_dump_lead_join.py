"""Tests for `taniteval.dump_lead_join` — banked-dump ``win["lead"]`` wiring.

Every synthetic case is hand-computable (straight road, constant speed, constant-gap lead), so
an assertion failure localises the defect rather than reporting "numbers differ". The
load-bearing cases are the refusals: a silently mis-attached lead block scores an arm against
another clip's traffic and the metric still returns a plausible number — the module must refuse
loudly instead, and these tests pin that it does.

The final test runs the wiring end-to-end on REAL local material (a lead130 v2ep record + the
md5-manifested lead130 agents join) and is skipped wherever that cache is absent.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from taniteval import lead_source as ls
from taniteval.dump_lead_join import (EP_AMBIGUOUS, EP_GRID_MISMATCH, EP_NO_JOIN,
                                      EP_NO_RECORD, EP_OK, EP_SPEED_MISMATCH,
                                      attach_lead, coverage_probe,
                                      episodes_from_v2ep_dir, read_agents_jsonl)
from taniteval.four_families import _distance_keeping

A_GRID = 0.35        # episode grid offset (s) — arbitrary, non-zero on purpose
B_GRID = 0.1007      # realised episode spacing (s) — the MEASURED ~0.1007, not 0.1
V_EGO = 10.0         # m/s
LEAD_CX = 20.0       # constant rig-frame gap to the lead's CENTRE
LEAD_LEN = 4.0
CLIP_A = "aaaa1111-0000-0000-0000-000000000001"
CLIP_B = "bbbb2222-0000-0000-0000-000000000002"

# real local material for the integration test (md5-manifested 2026-08-18)
LEAD130 = Path("C:/Users/Admin/tanitad-caches/sp2-lead130-20260818")
LEAD130_JSONL = LEAD130 / "join" / "lead130_agents.jsonl"
LEAD130_V2EP = LEAD130 / "cache" / "slotprobe-lead130-w120-256x640cyl"
REAL_CLIP = "0045da77-dcdd-46ad-8f97-e94f32c3711c"


# --------------------------------------------------------------------------- #
# fixtures                                                                     #
# --------------------------------------------------------------------------- #
def _episode(t_len=60, v=V_EGO):
    """Straight road, constant speed: poses (x, y, yaw, v) on the affine grid."""
    t = A_GRID + B_GRID * np.arange(t_len)
    return np.column_stack([v * t, np.zeros(t_len), np.zeros(t_len),
                            np.full(t_len, v)]), t


def _join_rows(clip, t, frames=None, agents_fn=None):
    """One jsonl row per labelled frame. Default: a lead at constant rig gap."""
    if agents_fn is None:
        def agents_fn(i):
            return [{"cx": LEAD_CX, "cy": 0.0, "yaw": 0.0, "l": LEAD_LEN,
                     "w": 1.9, "occ": 0, "track_id": "L1", "cls": "automobile"}]
    idxs = range(len(t)) if frames is None else frames
    return [{"clip_id": clip, "frame_idx": int(i), "t_s": round(float(t[i]), 4),
             "agents": agents_fn(i)} for i in idxs]


def _write_jsonl(tmp_path, rows, name="agents.jsonl"):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


def _win_for(poses, eid=0):
    """A dump-shaped dict whose gt/pred are the episode's own future (origin frame)."""
    t_len = poses.shape[0]
    origins = ls.window_last_indices(t_len)
    steps = np.array([5, 10, 15, 20])
    gt = np.zeros((origins.size, 4, 2))
    for j, o in enumerate(origins):
        dx = poses[o + steps, 0] - poses[o, 0]
        dy = poses[o + steps, 1] - poses[o, 1]
        c, s = np.cos(poses[o, 2]), np.sin(poses[o, 2])
        gt[j, :, 0] = dx * c + dy * s
        gt[j, :, 1] = -dx * s + dy * c
    return {"pred": torch.tensor(gt, dtype=torch.float32),
            "gt": torch.tensor(gt, dtype=torch.float32),
            "eid": [eid] * origins.size,
            "speed": torch.tensor(poses[origins, 3], dtype=torch.float32),
            "wp_steps": [5, 10, 15, 20]}, origins


def _rec(poses, clip_id=None, id4=None):
    return {"poses": poses, "clip_id": clip_id, "id4": id4}


# --------------------------------------------------------------------------- #
# happy path                                                                   #
# --------------------------------------------------------------------------- #
def test_happy_path_all_windows_lead(tmp_path):
    poses, t = _episode()
    win, origins = _win_for(poses)
    joins = read_agents_jsonl(_write_jsonl(tmp_path, _join_rows(CLIP_A, t)))
    lead = attach_lead(win, {0: _rec(poses, clip_id=CLIP_A)}, joins)

    assert list(lead["state"]) == [ls.LEAD] * origins.size
    assert lead["counts"] == {ls.LEAD: origins.size, ls.NO_LEAD: 0, ls.NO_LABEL: 0}
    # lead moves WITH the ego (constant rig gap): origin-frame x = CX + v*b*step.
    # atol: jsonl rounds t_s to 1e-4 and queries carry QUERY_EPS_S (2e-3 s), so
    # positions can shift by up to ~v*(eps+2e-4) ≈ 0.023 m at 10 m/s.
    steps = np.array([5, 10, 15, 20], dtype=float)
    want_x = LEAD_CX + V_EGO * B_GRID * steps
    assert np.allclose(lead["leads"][:, :, 0], want_x[None, :], atol=0.05)
    assert np.allclose(lead["leads"][:, :, 1], 0.0, atol=0.05)
    assert np.allclose(lead["lead_lens"], LEAD_LEN)
    assert np.allclose(lead["speeds"], V_EGO)
    assert np.allclose(lead["gap0_m"], LEAD_CX - LEAD_LEN / 2, atol=0.05)

    cov = lead["coverage"]["episodes"]["0"]
    assert cov["status"] == EP_OK
    assert cov["speed_check_max_mps"] <= 1e-3       # the label-free alignment proof
    assert abs(cov["grid"]["b"] - B_GRID) < 1e-3
    assert cov["grid"]["max_resid_s"] < 2e-3        # jsonl rounds t_s to 1e-4
    assert lead["dt_s"] == pytest.approx(5 * B_GRID, abs=1e-3)


def test_end_to_end_four_families_distance_keeping(tmp_path):
    """The consumer contract: the block this module emits IS what
    `four_families._distance_keeping` takes, and the numbers are the hand values."""
    poses, t = _episode()
    win, origins = _win_for(poses)
    joins = read_agents_jsonl(_write_jsonl(tmp_path, _join_rows(CLIP_A, t)))
    lead = attach_lead(win, {0: _rec(poses, clip_id=CLIP_A)}, joins, n_boot=25)

    dk = _distance_keeping(win["pred"], 0.5, lead)
    assert dk["status"] == "OK"
    assert dk["n"] == origins.size
    # constant-gap follow: headway = CX - len/2, never closing -> TTC censored at cap
    assert dk["mean_headway_min_m"] == pytest.approx(LEAD_CX - LEAD_LEN / 2, abs=0.05)
    assert dk["mean_time_gap_min_s"] == pytest.approx(
        (LEAD_CX - LEAD_LEN / 2) / V_EGO, abs=0.01)
    assert dk["mean_min_ttc_s"] == pytest.approx(30.0)
    assert dk["n_closing"] == 0
    assert "_per_window" in dk and "headway_min_m" in dk["_per_window"]
    # the eid travelled with the block -> the stratified read is emitted, with CIs
    assert dk["by_speed"]["n_windows_total"] == origins.size
    band = dk["by_speed"]["strata"]["10-15"]
    assert band["n_with_lead"] == origins.size
    assert band["status"] == "UNPOWERED"            # 4 windows < MIN_STRATUM_N, by design


# --------------------------------------------------------------------------- #
# absence is reported per episode, never silently dropped                      #
# --------------------------------------------------------------------------- #
def test_no_join_episode_is_no_label_with_reason(tmp_path):
    poses, t = _episode()
    win_a, origins = _win_for(poses, eid=0)
    win_b, _ = _win_for(poses, eid=1)
    win = {"pred": torch.cat([win_a["pred"], win_b["pred"]]),
           "gt": torch.cat([win_a["gt"], win_b["gt"]]),
           "eid": win_a["eid"] + win_b["eid"],
           "speed": torch.cat([win_a["speed"], win_b["speed"]])}
    joins = read_agents_jsonl(_write_jsonl(tmp_path, _join_rows(CLIP_A, t)))
    lead = attach_lead(win, {0: _rec(poses, clip_id=CLIP_A),
                             1: _rec(poses, clip_id=CLIP_B)}, joins)

    n = origins.size
    assert list(lead["state"][:n]) == [ls.LEAD] * n
    assert list(lead["state"][n:]) == [ls.NO_LABEL] * n
    cov = lead["coverage"]["episodes"]["1"]
    assert cov["status"] == EP_NO_JOIN
    assert "free flow" in cov["reason"]
    assert cov["counts"][ls.NO_LABEL] == n
    # speeds still populated from the dump for the uncovered episode (stratifier input)
    assert np.allclose(lead["speeds"][n:], V_EGO)
    assert lead["coverage"]["n_windows_labelled"] == n
    assert lead["coverage"]["n_episodes_ok"] == 1


def test_missing_episode_record_is_refused_not_crashed(tmp_path):
    poses, t = _episode()
    win, origins = _win_for(poses, eid=7)
    joins = read_agents_jsonl(_write_jsonl(tmp_path, _join_rows(CLIP_A, t)))
    lead = attach_lead(win, {}, joins)              # no record for eid 7 at all
    assert list(lead["state"]) == [ls.NO_LABEL] * origins.size
    assert lead["coverage"]["episodes"]["7"]["status"] == EP_NO_RECORD


def test_speed_mismatch_refuses_the_episode(tmp_path):
    """The misalignment detector: a corrupted mapping must never be scored."""
    poses, t = _episode()
    win, origins = _win_for(poses)
    win["speed"] = win["speed"] + 0.5               # not this episode's v0 any more
    joins = read_agents_jsonl(_write_jsonl(tmp_path, _join_rows(CLIP_A, t)))
    lead = attach_lead(win, {0: _rec(poses, clip_id=CLIP_A)}, joins)
    cov = lead["coverage"]["episodes"]["0"]
    assert cov["status"] == EP_SPEED_MISMATCH
    assert list(lead["state"]) == [ls.NO_LABEL] * origins.size


def test_grid_mismatch_refuses_the_episode(tmp_path):
    poses, t = _episode(t_len=60)
    win, _ = _win_for(poses)                        # 4 windows for T=60
    joins = read_agents_jsonl(_write_jsonl(tmp_path, _join_rows(CLIP_A, t)))
    lead = attach_lead(win, {0: _rec(poses[:52], clip_id=CLIP_A)}, joins)
    assert lead["coverage"]["episodes"]["0"]["status"] == EP_GRID_MISMATCH
    assert all(s == ls.NO_LABEL for s in lead["state"])


def test_ambiguous_prefix_is_refused_not_guessed(tmp_path):
    poses, t = _episode()
    win, _ = _win_for(poses)
    rows = _join_rows("aaaa1111-x", t) + _join_rows("aaaa2222-y", t)
    joins = read_agents_jsonl(_write_jsonl(tmp_path, rows))
    lead = attach_lead(win, {0: _rec(poses, id4="aaaa")}, joins)
    assert lead["coverage"]["episodes"]["0"]["status"] == EP_AMBIGUOUS
    assert all(s == ls.NO_LABEL for s in lead["state"])


# --------------------------------------------------------------------------- #
# the span semantics — labels-ended vs road-clear, never conflated              #
# --------------------------------------------------------------------------- #
def test_labels_ended_is_no_label_not_no_lead(tmp_path):
    """Rows stop halfway through the episode: windows past the row span are NO_LABEL
    (missing labels), never NO_LEAD (which would manufacture free flow)."""
    poses, t = _episode(t_len=60)
    win, origins = _win_for(poses)                  # origins [7, 15, 23, 31]
    rows = _join_rows(CLIP_A, t, frames=range(30))  # labelled span ends at idx 29
    joins = read_agents_jsonl(_write_jsonl(tmp_path, rows))
    lead = attach_lead(win, {0: _rec(poses, clip_id=CLIP_A)}, joins)
    # origin 7: horizon ends at idx 27, inside the span -> LEAD.
    # origins 15/23/31: horizon (or origin) leaves the span -> NO_LABEL.
    assert list(lead["state"]) == [ls.LEAD, ls.NO_LABEL, ls.NO_LABEL, ls.NO_LABEL]
    assert lead["counts"][ls.NO_LEAD] == 0


def test_clear_road_inside_span_is_no_lead(tmp_path):
    """Rows exist with agents=[] — join_clip's contract says the frame IS labelled and the
    road is clear. The span sentinels keep that from collapsing into NO_LABEL."""
    poses, t = _episode()
    win, origins = _win_for(poses)
    rows = _join_rows(CLIP_A, t, agents_fn=lambda i: [])
    joins = read_agents_jsonl(_write_jsonl(tmp_path, rows))
    lead = attach_lead(win, {0: _rec(poses, clip_id=CLIP_A)}, joins)
    assert list(lead["state"]) == [ls.NO_LEAD] * origins.size
    assert lead["coverage"]["episodes"]["0"]["status"] == EP_OK


def test_non_vehicle_agents_are_never_leads(tmp_path):
    poses, t = _episode()
    win, origins = _win_for(poses)
    rows = _join_rows(CLIP_A, t, agents_fn=lambda i: [
        {"cx": 10.0, "cy": 0.0, "yaw": 0.0, "l": 0.6, "w": 0.6, "occ": 0,
         "track_id": "P1", "cls": "pedestrian"}])
    joins = read_agents_jsonl(_write_jsonl(tmp_path, rows))
    assert not joins[CLIP_A]["obs"]["is_vehicle"].any()
    lead = attach_lead(win, {0: _rec(poses, clip_id=CLIP_A)}, joins)
    assert list(lead["state"]) == [ls.NO_LEAD] * origins.size


# --------------------------------------------------------------------------- #
# reader + probe units                                                         #
# --------------------------------------------------------------------------- #
def test_read_agents_jsonl_filters_and_shapes(tmp_path):
    poses, t = _episode(t_len=40)
    rows = _join_rows(CLIP_A, t) + _join_rows(CLIP_B, t, frames=range(10))
    p = _write_jsonl(tmp_path, rows)
    joins = read_agents_jsonl(p)
    assert set(joins) == {CLIP_A, CLIP_B}
    assert joins[CLIP_B]["n_rows"] == 10
    only_a = read_agents_jsonl(p, clips={CLIP_A})
    assert set(only_a) == {CLIP_A}
    o = joins[CLIP_A]["obs"]
    assert o["t"].shape == o["center_x"].shape == o["size_x"].shape
    assert o["is_vehicle"].all()


def test_coverage_probe_counts_matches(tmp_path):
    poses, t = _episode(t_len=40)
    joins = read_agents_jsonl(_write_jsonl(tmp_path, _join_rows(CLIP_A, t)))
    eps = {0: _rec(poses, id4=CLIP_A[:4]), 1: _rec(poses, id4="ffff")}
    pr = coverage_probe(eps, joins)
    assert pr["n_matched"] == 1
    assert pr["episodes"]["0"]["matched"] is True
    assert pr["episodes"]["1"]["matched"] is False


# --------------------------------------------------------------------------- #
# REAL DATA — the lead130 material (skipped where the local cache is absent)    #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not (LEAD130_JSONL.exists()
                         and (LEAD130_V2EP / f"{REAL_CLIP}.v2ep.pt").exists()),
                    reason="lead130 local cache not present on this host")
def test_lead130_real_clip_end_to_end():
    """The whole path on real labels: v2ep poses + the md5-manifested agents join.

    The identity-roundtrip assertion is the sharp one: composing a cuboid rig->world->same-frame
    through the TRIMMED poses must return the cuboid's own (cx, cy). An n_stack trim error of
    even one frame shifts the ego pose by ~v*0.1 s (metres at speed) and fails it loudly."""
    eps = episodes_from_v2ep_dir(LEAD130_V2EP, clips={REAL_CLIP})
    rec = eps[REAL_CLIP]
    joins = read_agents_jsonl(LEAD130_JSONL, clips={REAL_CLIP})
    jn = joins[REAL_CLIP]
    poses = rec["poses"]
    assert int(jn["frame_idx"].max()) < poses.shape[0]

    # --- identity roundtrip at a moving frame with a vehicle present --------- #
    from taniteval.dump_lead_join import _affine_fit
    a, b, resid = _affine_fit(jn["frame_idx"], jn["t_s"])
    assert resid < 2e-3 and 0.095 < b < 0.11
    t_grid = a + b * np.arange(poses.shape[0])
    o = jn["obs"]
    moving = np.interp(o["t"], t_grid, poses[:, 3]) > 2.0
    cand = np.flatnonzero(o["is_vehicle"] & moving)
    assert cand.size, "no moving-frame vehicle sample to roundtrip on this clip"
    i = int(cand[0])
    got = ls.lead_track_in_window(
        o["t"], o["track"], o["center_x"], o["center_y"], o["track"][i],
        float(o["t"][i]), np.array([float(o["t"][i])]),
        t_grid, poses[:, 0], poses[:, 1], poses[:, 2])
    # ts is the absolute sample time (K=1) with t0 at the same instant: identity must hold
    assert np.allclose(got[0], [o["center_x"][i], o["center_y"][i]], atol=0.06), (
        f"rig->world->frame roundtrip broke: {got[0]} vs "
        f"({o['center_x'][i]}, {o['center_y'][i]}) — n_stack trim or pose "
        f"convention is off")

    # --- full attach + the four_families consumer ---------------------------- #
    win, origins = _win_for(poses, eid=0)
    lead = attach_lead(win, {0: rec}, joins, n_boot=25)
    cov = lead["coverage"]["episodes"]["0"]
    assert cov["status"] == EP_OK, cov
    assert cov["speed_check_max_mps"] <= 1e-3
    assert lead["counts"][ls.LEAD] >= 1, lead["counts"]
    g = lead["gap0_m"][np.isfinite(lead["gap0_m"])]
    assert (g >= 0).all() and (g <= 80.0).all()

    dk = _distance_keeping(win["pred"], 0.5, lead)
    assert dk["status"] == "OK" and dk["n"] >= 1
    assert "by_speed" in dk
