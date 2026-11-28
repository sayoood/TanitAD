"""Backlog R1 — LONGITUDINAL distance-keeping on refav1's (clip, RAW frame) grid.

``taniteval/tools/build_lead_block_b1.py`` (the per-frame block) and the
``--lead-block`` join in ``taniteval/tools/refav1_arm.py``. CPU only; the real
checkpoint, Thor and the pods are never touched. The real-data tests run only
where the local eval slice / the banked block exist (they skip elsewhere).

WHAT IS PINNED
  (1) the container round-trips through ``eval_four_families.load_lead_block``
      (the programme's one two-container reader) and carries the contract:
      required keys, (clip_id, frame) rows, ts_rel_s = 0.2*(1..K), the four
      states in a <U12 array, GT reference columns, coverage/meta JSON;
  (2) the time base: ``episode_grid`` IS the builder's formula, and on the real
      20-clip slice the reconstructed poses equal the banked v2ep poses
      (float32-identical — MEASURED 2026-09-02, asserted here at 1 cm / 1 deg);
  (3) the adapter: distance-keeping is PRESENT (status OK, n, intervals, by_speed,
      GT reference, kinematic-contract check, paired deltas) when the block covers
      the windows; REFUSED WITH THE COUNT when it has no rows for the clips;
      NO_LABEL rows are counted and never scored as free flow; a block on another
      horizon grid, a POSITIONAL block, and a speed mismatch are refused by name;
      without a block the family stays UNAVAILABLE with its reason.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_REPO = Path(__file__).resolve().parents[2]
TOOLS = _REPO / "taniteval" / "tools"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bb = _load("build_lead_block_b1_under_test", TOOLS / "build_lead_block_b1.py")
ra = _load("refav1_arm_under_test_lead", TOOLS / "refav1_arm.py")
eff = _load("eval_four_families_under_test_lead", TOOLS / "eval_four_families.py")
tra = _load("test_refav1_arm_helpers", Path(__file__).resolve().parent / "test_refav1_arm.py")

from taniteval import lead_source as ls  # noqa: E402

K, DT = 10, 0.2
GAP_M, LEAD_LEN_M = 20.0, 4.4
SLICE_EPS = Path(r"C:\Users\Admin\refav1_eval_slice\eps")
CORPUS = Path(r"C:\Users\Admin\tanitad-wt\_s2build\release\tanitad-v7-training-corpus")
BANKED_CANDIDATES = [
    Path(ra.LEAD_BLOCK_DEFAULT),
    Path(r"C:\Users\Admin\tanitad-wt\_lead_b1\b1_eval_lead_block.npz"),
    Path(os.environ.get("TANITAD_B1_LEAD_BLOCK", "")),
]


# --------------------------------------------------------------------------- #
# synthetic world: straight ego, a lead 20 m ahead moving with it               #
# --------------------------------------------------------------------------- #
def _straight_poses(i: int, T: int = tra.T_EP):
    """Straight-line unicycle at 10 Hz (kappa 0) so the corridor gate is
    trivially straight; speed varies sinusoidally like the arm fixture."""
    import math
    t = torch.arange(T, dtype=torch.float32)
    v = 6.0 + 1.5 * i + 0.8 * torch.sin(2 * math.pi * t / 60.0)
    poses = torch.zeros(T, 4)
    x = 0.0
    for f in range(T):
        poses[f] = torch.tensor([x, 0.0, 0.0, float(v[f])])
        x += float(v[f]) * 0.1
    acts = torch.zeros(T, 2)
    acts[1:, 1] = (v[1:] - v[:-1]) / 0.1
    return poses, acts


def _fixture_straight(root: Path):
    """``test_refav1_arm._fixture`` with straight motion (same shapes/dtypes)."""
    cache, eps = root / "cache", root / "eps"
    cache.mkdir(parents=True, exist_ok=True)
    eps.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(0)
    poses_by = {}
    for i, nm in enumerate(tra.NAMES):
        f = torch.randn(tra.T_C, tra.N_TOK, tra.D, generator=g)
        torch.save(f.to(torch.float8_e4m3fn), cache / f"{nm}.pt")
        poses, acts = _straight_poses(i)
        torch.save({"poses": poses, "actions": acts, "episode_id": nm,
                    "clip_id": tra.CLIP[nm]}, eps / f"{nm}.v2ep.pt")
        poses_by[tra.CLIP[nm]] = poses.numpy().astype(np.float64)
    return cache, eps, poses_by


def _ego_from_poses(poses: np.ndarray) -> dict:
    n = poses.shape[0]
    return {"t": 0.1 * np.arange(n), "x": poses[:, 0], "y": poses[:, 1],
            "yaw": np.unwrap(poses[:, 2]), "v": poses[:, 3]}


def _lead_obs(poses: np.ndarray, from_s: float | None = None,
              gap: float = GAP_M) -> dict:
    """A vehicle track that sits ``gap`` m ahead of the ego's REAR-face-to-rig
    convention (centre at gap + L/2) at every 10 Hz sample, plus a far-away
    non-vehicle track spanning the whole episode so the labelled span never
    collapses when the vehicle track is absent. ``from_s``: the vehicle appears
    only from that time on — windows before it are NO_LEAD (causal selection
    sees no sample <= t0), windows after it carry the FULL track, so the metric
    sits at its known value (a track that ENDS inside a horizon is held for the
    0.5 s staleness and shrinks the gap — the instrument's convention, which a
    known-value test must not straddle)."""
    n = poses.shape[0]
    # ⚠️ sample times sit 1 ns BEFORE the 10 Hz grid on purpose. `lead_source`
    # takes the LAST cuboid <= t, and a query `0.1*i + 0.2*k` lands a few ulp
    # below `0.1*(i+2k)` about half the time — which would silently pick the
    # previous sample (one frame stale, ~v*0.1 s of gap) purely from float
    # rounding. Same phenomenon `dump_lead_join.QUERY_EPS_S` documents; here the
    # world is made exact so the metric can be asserted at a KNOWN value.
    t = 0.1 * np.arange(n) - 1e-9
    keep = np.ones(n, dtype=bool) if from_s is None else t >= from_s
    tv, tp = t[keep], t
    return {"t": np.concatenate([tv, tp]),
            "track": np.array(["veh-1"] * tv.size + ["ped-9"] * tp.size, dtype=object),
            "center_x": np.concatenate([np.full(tv.size, gap + LEAD_LEN_M / 2.0),
                                        np.full(tp.size, 5.0)]),
            "center_y": np.concatenate([np.zeros(tv.size), np.full(tp.size, 50.0)]),
            "size_x": np.concatenate([np.full(tv.size, LEAD_LEN_M), np.full(tp.size, 0.6)]),
            "is_vehicle": np.concatenate([np.ones(tv.size, bool), np.zeros(tp.size, bool)])}


def _build_block(path: Path, clips: dict, *, k: int = K, dt: float = DT,
                 speed_bump: dict | None = None) -> Path:
    """``clips``: {clip_id: (poses, obs or None)} -> a block file via the builder."""
    rows, cov = [], {}
    for cid, (poses, obs) in clips.items():
        ego = _ego_from_poses(poses)
        r, c = bb.build_clip_rows(cid, ego, ego["t"], obs, k=k, dt=dt, poses=poses,
                                  obs_reason=None if obs is not None else "synthetic: none")
        if speed_bump and cid in speed_bump:
            r["speeds"] = r["speeds"] + float(speed_bump[cid])
        rows.append(r)
        cov[cid] = c
    bb.write_block(str(path), rows, cov, {"version": "test", "tool": "test"}, k=k, dt=dt)
    return path


# =========================================================================== #
# (1) the container                                                            #
# =========================================================================== #
def test_container_roundtrips_through_the_banked_loader_and_carries_the_contract(tmp_path):
    p1, _ = _straight_poses(0)
    p2, _ = _straight_poses(1)
    p1, p2 = p1.numpy().astype(np.float64), p2.numpy().astype(np.float64)
    path = _build_block(tmp_path / "blk.npz", {"clip-a": (p1, _lead_obs(p1)),
                                               "clip-b": (p2, None)})
    blk = eff.load_lead_block(str(path))            # the ONE container reader
    assert isinstance(blk, dict)
    for key in bb.REQUIRED_KEYS + ("gap0_m", "has_lead", "gt_headway_min_m",
                                   "gt_time_gap_min_s", "gt_min_ttc_s",
                                   "coverage_json", "meta_json"):
        assert key in blk, key
    n = p1.shape[0] + p2.shape[0]
    assert blk["leads"].shape == (n, K, 2) and blk["lead_lens"].shape == (n,)
    assert blk["state"].dtype == np.dtype(bb.STATE_DTYPE)     # NOT_STRAIGHT fits
    assert np.allclose(blk["ts_rel_s"], DT * np.arange(1, K + 1))
    assert float(blk["dt_s"][0]) == DT
    assert (blk["eid"] == blk["clip_id"]).all()
    a = blk["clip_id"] == "clip-a"
    b = blk["clip_id"] == "clip-b"
    assert (blk["frame"][a] == np.arange(p1.shape[0])).all()
    assert set(blk["state"][b].tolist()) == {ls.NO_LABEL}           # no obs -> NO_LABEL
    st_a = blk["state"][a]
    assert (st_a == ls.LEAD).sum() > 0 and (st_a == ls.NO_LABEL).sum() > 0  # span tail
    assert ((blk["has_lead"]) == (blk["state"] == ls.LEAD)).all()
    lead_rows = blk["state"] == ls.LEAD
    assert np.allclose(blk["gap0_m"][lead_rows], GAP_M, atol=1e-6)
    assert np.allclose(blk["gt_headway_min_m"][lead_rows], GAP_M, atol=1e-6)
    assert np.isnan(blk["gt_headway_min_m"][~lead_rows]).all()
    # the lead moves with the ego: its along-speed over the last 0.6 s minus the
    # ego speed AT t0 differs only by the ego's own acceleration inside that
    # window (|dv/dt| <= 0.84 m/s^2 in this fixture -> < 0.3 m/s). The first
    # frames have no 0.15 s of lead history yet -> NaN, and that is reported.
    rv = blk["rel_speed_mps"][lead_rows]
    assert np.isnan(rv).sum() <= 2 and np.nanmax(np.abs(rv)) < 0.3, rv[:6]
    cov = bb.read_block_json(blk, "coverage_json")
    meta = bb.read_block_json(blk, "meta_json")
    assert cov["clip-b"]["obstacle_offline"] is False and cov["clip-b"]["free_flow_share"] is None
    assert cov["clip-a"]["counts"][ls.LEAD] == int((st_a == ls.LEAD).sum())
    assert meta["counts"][ls.NO_LABEL] == int((blk["state"] == ls.NO_LABEL).sum())
    assert meta["k"] == K and meta["dt_s"] == DT


def test_episode_grid_is_the_builders_formula():
    from tanitad.data.physicalai import TARGET_HZ
    t_us = (-14158.0 + np.arange(605) * (1e6 / 30.0))           # a 30 Hz camera clock
    tq, unit, n = bb.episode_grid(t_us)
    span_s = (t_us[-1] - t_us[0]) / 1e6
    assert unit == 1e6 and n == int(span_s * TARGET_HZ) == 201
    assert tq[0] == t_us[0] and tq[-1] == t_us[-1] and tq.size == n
    assert abs((tq[1] - tq[0]) / 1e6 - span_s / (n - 1)) < 1e-12   # ~0.1007 s, not 0.1


# =========================================================================== #
# (3) the adapter, on a rolled dump                                            #
# =========================================================================== #
@pytest.fixture(scope="module")
def rolled(tmp_path_factory):
    root = tmp_path_factory.mktemp("refav1_lead")
    cache, eps, poses_by = _fixture_straight(root)
    ck = tra._ckpt(root)
    a = tra._args(root, ck, cache, eps, None, no_navshuf=True)
    manifest = ra.run_dump(a)
    return root, a, manifest, poses_by


def _blocks(root: Path, poses_by: dict) -> dict:
    c0, c1, c2 = (tra.CLIP[nm] for nm in tra.NAMES)
    p0, p1, p2 = poses_by[c0], poses_by[c1], poses_by[c2]
    return {
        # c0: lead everywhere; c1: lead only from 2.0 s on (NO_LEAD before it,
        # labels present through the non-vehicle track); c2: no obstacle.offline
        "cover": _build_block(root / "cover.npz", {c0: (p0, _lead_obs(p0)),
                                                   c1: (p1, _lead_obs(p1, from_s=2.0)),
                                                   c2: (p2, None)}),
        "other": _build_block(root / "other.npz", {"zzz-1": (p0, _lead_obs(p0)),
                                                   "zzz-2": (p1, _lead_obs(p1))}),
        "nolabel": _build_block(root / "nolabel.npz", {c0: (p0, None), c1: (p1, None),
                                                       c2: (p2, None)}),
        "grid05": _build_block(root / "grid05.npz", {c0: (p0, _lead_obs(p0))},
                               k=4, dt=0.5),
        "bump": _build_block(root / "bump.npz", {c0: (p0, _lead_obs(p0)),
                                                 c1: (p1, _lead_obs(p1))},
                             speed_bump={c1: 1.0}),
    }


def test_distance_keeping_is_PRESENT_when_the_block_covers_the_windows(rolled):
    root, a, manifest, poses_by = rolled
    blocks = _blocks(root, poses_by)
    rec = ra.analyze_refav1(a.dump_dir, n_boot=40, seed=0, lead_block=str(blocks["cover"]))
    N = rec["n_windows"]
    dk = rec["refav1"]["distance_keeping"]
    assert dk["status"] == "PRESENT"
    cov = dk["coverage"]
    assert cov["n_windows"] == N and cov["n_episodes"] == 3
    assert cov["counts"][ls.LEAD] + cov["counts"][ls.NO_LEAD] + cov["counts"][ls.NO_LABEL] == N
    assert cov["counts"][ls.LEAD] > 0 and cov["counts"][ls.NO_LEAD] > 0
    assert cov["speed_check"]["max_mps"] < ra.LEAD_SPEED_TOL_MPS       # label-free proof
    eps = cov["episodes"]
    assert eps["ep000"]["status"] == ra.LEAD_EP_OK and eps["ep001"]["status"] == ra.LEAD_EP_OK
    assert eps["ep002"]["status"] == ra.LEAD_EP_OK                      # rows exist, all NO_LABEL
    assert eps["ep002"]["counts"][ls.NO_LABEL] == eps["ep002"]["n_windows"]
    assert cov["n_windows_no_row"] == 0
    # -- every arm's four_families LONGITUDINAL row now carries the family ------
    for arm, blk in rec["arms"].items():
        lon = blk["four_families"]["longitudinal"]
        d = lon["distance_keeping"]
        assert d["status"] == "OK", (arm, d)
        assert 0 < d["n"] <= cov["counts"][ls.LEAD]
        assert d["by_speed"]["window_states_total"][ls.NO_LABEL] == cov["counts"][ls.NO_LABEL]
        assert "distance_keeping" not in blk["four_families"]["_families_unavailable"]
        comps = lon["ci"]["components"]
        assert "distance_keeping.mean_headway_min_m" in comps
        assert comps["distance_keeping.mean_headway_min_m"]["n_windows_with_lead"] == d["n"]
        pa = dk["per_arm"][arm]
        assert pa["status"] == "OK" and pa["n"] == d["n"] and pa["tier"] == blk["tier"]
        assert pa["ci"]["mean_headway_min_m"] is not None
        assert isinstance(d["_per_window"], str)                        # stripped for JSON
    # -- the known value: ol (recorded actions from v0) reproduces the GT arm ----
    kc = dk["kinematic_contract_check"]
    assert kc["n_both"] > 0 and kc["max_abs_headway_diff_m"] < 0.5, kc
    assert abs(dk["per_arm"]["ol"]["mean_headway_min_m"] - GAP_M) < 0.5
    assert abs(dk["gt_reference"]["headway_min_m"]["mean"] - GAP_M) < 1e-3
    assert dk["gt_reference"]["headway_min_m"]["n"] == cov["counts"][ls.LEAD]
    # -- paired deltas on the same windows ---------------------------------------
    for nm in ("paired_closed_minus_open", "paired_cl_minus_ha"):
        m = dk["paired"][nm]["metrics"]["headway_min_m"]
        assert m["status"] == "OK" and m["n_used"] > 0
        assert dk["paired"][nm]["estimator"].startswith("paired_episode_cluster_bootstrap")


def test_no_rows_for_the_clips_is_REFUSED_with_the_count(rolled):
    root, a, manifest, poses_by = rolled
    blocks = _blocks(root, poses_by)
    rec = ra.analyze_refav1(a.dump_dir, n_boot=40, seed=0, lead_block=str(blocks["other"]))
    N = rec["n_windows"]
    dk = rec["refav1"]["distance_keeping"]
    assert dk["status"] == "REFUSED" and dk["n"] == 0
    assert "0 labelled windows" in dk["reason"]
    cov = dk["coverage"]
    assert cov["n_windows_no_row"] == N and cov["n_episodes_ok"] == 0
    assert all(c["status"] == ra.LEAD_EP_NO_ROWS for c in cov["episodes"].values())
    for arm, blk in rec["arms"].items():                    # the family row: unchanged
        d = blk["four_families"]["longitudinal"]["distance_keeping"]
        assert d["status"] == "UNAVAILABLE" and d["n"] == 0 and d["reason"]


def test_NO_LABEL_rows_are_counted_and_never_scored_as_free_flow(rolled):
    root, a, manifest, poses_by = rolled
    blocks = _blocks(root, poses_by)
    rec = ra.analyze_refav1(a.dump_dir, n_boot=40, seed=0, lead_block=str(blocks["nolabel"]))
    N = rec["n_windows"]
    dk = rec["refav1"]["distance_keeping"]
    assert dk["status"] == "REFUSED"
    assert dk["coverage"]["counts"][ls.NO_LABEL] == N and dk["coverage"]["n_windows_no_row"] == 0
    assert dk["coverage"]["n_episodes_ok"] == 3                   # rows joined, none labelled
    assert "NOT read as free flow" in dk["reason"]
    for arm, blk in rec["arms"].items():
        d = blk["four_families"]["longitudinal"]["distance_keeping"]
        assert d["status"] == "UNAVAILABLE"                        # never NOT-APPLICABLE/free-flow


def test_a_block_on_another_horizon_grid_is_refused_not_truncated(rolled):
    root, a, manifest, poses_by = rolled
    blocks = _blocks(root, poses_by)
    with pytest.raises(SystemExit, match="horizon grid"):
        ra.analyze_refav1(a.dump_dir, n_boot=40, seed=0, lead_block=str(blocks["grid05"]))


def test_a_positional_block_without_clip_and_frame_is_refused(rolled, tmp_path):
    root, a, manifest, poses_by = rolled
    n = rec_n = 6
    p = tmp_path / "positional.npz"
    np.savez(p, leads=np.zeros((n, K, 2)), lead_lens=np.full(n, 4.4), speeds=np.full(n, 10.0),
             state=np.array(["LEAD"] * n, dtype="<U12"),
             eid=np.array([f"e{i}" for i in range(n)], dtype="<U8"),
             ts_rel_s=DT * np.arange(1, K + 1))
    with pytest.raises(SystemExit, match="positional"):
        ra.analyze_refav1(a.dump_dir, n_boot=40, seed=0, lead_block=str(p))
    assert rec_n == 6


def test_a_speed_mismatch_refuses_that_episode_and_keeps_the_others(rolled):
    root, a, manifest, poses_by = rolled
    blocks = _blocks(root, poses_by)
    rec = ra.analyze_refav1(a.dump_dir, n_boot=40, seed=0, lead_block=str(blocks["bump"]))
    cov = rec["refav1"]["distance_keeping"]["coverage"]
    eps = cov["episodes"]
    assert eps["ep000"]["status"] == ra.LEAD_EP_OK
    assert eps["ep001"]["status"] == ra.LEAD_EP_SPEED
    assert eps["ep001"]["speed_check_max_mps"] > 0.9
    assert eps["ep001"]["counts"][ls.NO_LABEL] == eps["ep001"]["n_windows"]   # refused -> NO_LABEL
    assert eps["ep002"]["status"] == ra.LEAD_EP_NO_ROWS
    assert rec["refav1"]["distance_keeping"]["status"] == "PRESENT"          # ep000 still scores


def test_without_a_block_the_family_stays_UNAVAILABLE_with_its_reason(rolled):
    root, a, manifest, poses_by = rolled
    rec = ra.analyze_refav1(a.dump_dir, n_boot=40, seed=0)
    dk = rec["refav1"]["distance_keeping"]
    assert dk["status"] == "REFUSED" and "no lead block" in dk["reason"]
    for arm, blk in rec["arms"].items():
        d = blk["four_families"]["longitudinal"]["distance_keeping"]
        assert d["status"] == "UNAVAILABLE" and "WORK ITEM" in d["reason"]


# =========================================================================== #
# (2) real data — the time base and the banked block (skip where absent)       #
# =========================================================================== #
def _real_inputs_present() -> bool:
    return (SLICE_EPS.is_dir() and any(SLICE_EPS.glob("*.v2ep.pt"))
            and (CORPUS / "egomotion" / "egomotion_alpamayo.tar").exists()
            and (CORPUS / "timestamps" / "timestamps.tar").exists())


@pytest.mark.skipif(not _real_inputs_present(), reason="local eval slice / corpus tars absent")
def test_real_slice_poses_are_reconstructed_exactly_from_egomotion_and_timestamps():
    import io
    import tarfile

    import pandas as pd
    ego_tf = tarfile.open(CORPUS / "egomotion" / "egomotion_alpamayo.tar")
    ts_tf = tarfile.open(CORPUS / "timestamps" / "timestamps.tar")
    ego_names = {m.name.rsplit("/", 1)[-1]: m for m in ego_tf.getmembers()}
    ts_names = {m.name.rsplit("/", 1)[-1]: m for m in ts_tf.getmembers()}
    worst = {"max_dxy_m": 0.0, "max_dyaw_deg": 0.0, "max_dv_mps": 0.0, "n": 0}
    for vp in sorted(SLICE_EPS.glob("*.v2ep.pt")):
        cid = vp.name[:-len(".v2ep.pt")]
        ego_df = pd.read_parquet(io.BytesIO(ego_tf.extractfile(ego_names[f"{cid}.parquet"]).read()))
        ts_df = pd.read_parquet(io.BytesIO(
            ts_tf.extractfile(ts_names[f"{cid}.timestamps.parquet"]).read()))
        tcol = next(c for c in ts_df.columns if "time" in c.lower())
        tq, unit, n = bb.episode_grid(ts_df[tcol].to_numpy(np.float64))
        xc = bb.crosscheck_v2ep(str(vp), bb.episode_poses(ego_df, tq))
        assert xc["pass"], xc
        assert xc["T_v2ep"] == n == xc["T_reconstructed"]
        for k in ("max_dxy_m", "max_dyaw_deg", "max_dv_mps"):
            worst[k] = max(worst[k], xc[k])
        worst["n"] += 1
    assert worst["n"] >= 1
    assert worst["max_dxy_m"] <= bb.XCHECK_TOL_M and worst["max_dyaw_deg"] <= bb.XCHECK_TOL_DEG
    print("REAL-SLICE CROSS-CHECK", worst)


def _banked() -> Path | None:
    for p in BANKED_CANDIDATES:
        try:
            if p and p.is_file():
                return p
        except OSError:
            continue
    return None


@pytest.mark.skipif(_banked() is None or not SLICE_EPS.is_dir(),
                    reason="banked B1 lead block / local eval slice absent")
def test_banked_block_carries_the_real_slice_clips_and_their_speeds():
    """The label-free alignment proof on REAL data: the block's `speeds` at frame
    2t equal the v2ep poses[2t, 3] the adapter dumps as v0 (same interpolation)."""
    blk, idx, meta = ra.load_lead_block_rows(str(_banked()))
    assert meta.get("k") == K and meta.get("dt_s") == DT
    n_clips, worst = 0, 0.0
    for vp in sorted(SLICE_EPS.glob("*.v2ep.pt")):
        cid = vp.name[:-len(".v2ep.pt")]
        d = torch.load(str(vp), map_location="cpu", weights_only=False, mmap=True)
        poses = d["poses"].float().numpy()
        rows = np.array([idx.get((cid, 2 * t), -1) for t in range(poses.shape[0] // 2)])
        assert (rows >= 0).all(), f"{cid}: frames missing from the banked block"
        dv = float(np.max(np.abs(np.asarray(blk["speeds"])[rows] - poses[::2][:rows.size, 3])))
        worst = max(worst, dv)
        n_clips += 1
    assert n_clips >= 1 and worst < ra.LEAD_SPEED_TOL_MPS, worst
    cov = bb.read_block_json(blk, "coverage_json")
    assert len(cov) == meta["n_clips"] == 147
    no_obs = [c for c, v in cov.items() if not v.get("obstacle_offline")]
    assert len(no_obs) == meta["refusals"]["clips_without_obstacle_offline"]
    for c in no_obs:                                   # NO_LABEL, never free flow
        assert cov[c]["counts"][ls.NO_LABEL] == cov[c]["n_frames"]
        assert cov[c]["free_flow_share"] is None
