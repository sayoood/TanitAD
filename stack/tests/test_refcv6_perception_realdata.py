"""refcv6 perception — the REAL-DATA checks, on the artifacts that are on the box.

Three fixtures, each independently skippable:

* ``$TANITAD_SAM3_MAP_DIR``  — the 135 SAM3 eval maps copied from Thor
  (``*.sam3mapgt.npz``, named by sha12);
* ``$TANITAD_V2EP_DIR``      — the matching ``*.v2ep.pt`` pixel payloads;
* ``$TANITAD_CALIB_DIR``     — ``sensor_extrinsics.parquet`` (the per-clip mount
  pose the lift geometry needs);
* ``$TANITAD_OBSTACLE_DIR``  — ``obstacle_offline_b1eval/*.parquet`` (3-D cuboids).

⛔ Clip ids never enter a report: every identity here is a sha12.

⚠️ :func:`test_orientation_guard_mirror_goes_red` is the expensive one (it
decodes frames and fits a probe). By default it scores **every** scorable clip,
because that is what makes it powered -- see its docstring. Set
``$TANITAD_ORIENT_CLIPS`` to cap it for a fast local run; it then SKIPS instead
of asserting.
"""
from __future__ import annotations

import glob
import hashlib
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import tanitad.models.bev_lift as L
from tanitad.data.agent_cuboid_gt import (
    _equivalence_probe, check_cuboid_schema, cuboids_at_time,
    ground_bottom_stats, read_clip_cuboids, zh_by_track,
)
from tanitad.data.bev_raster import GRID_DEFAULT, agents_at_time
from tanitad.data.lift_orientation import (
    DRIVABLE_CHANNEL, MIN_CLIPS_FOR_POWER, cells_for_probe,
    lift_image_features, mirror_grid, orientation_probe, sign_test_power,
)
from tanitad.data.perception_targets import (
    MapGTStore, assert_frame_alignment, collate_map_targets,
    require_map_coverage,
)
from tanitad.data.semantic_map_gt import TimeMisalignment, sha12
from tanitad.models.bev_encoder import BEVMapBranch, map_metrics, map_soft_ce
from tanitad.models.bev_lift import BEVLift

MAP_DIR = Path(os.environ.get("TANITAD_SAM3_MAP_DIR",
                              "D:/Projects/TanitAD-artifacts/sam3-maps-eval"))
V2_DIR = Path(os.environ.get("TANITAD_V2EP_DIR",
                             "C:/Users/Admin/refcv5cmp/data/eval"))
CALIB = Path(os.environ.get(
    "TANITAD_CALIB_DIR",
    "D:/Projects/TanitAD-artifacts/hf-corpus-aug-20260915/stage/calibration"))
OBST_DIR = Path(os.environ.get(
    "TANITAD_OBSTACLE_DIR",
    "C:/Users/Admin/tanitad-data/physicalai/labels/obstacle_offline_b1eval"))
#: 0 == score EVERY scorable clip, which is the default: the guard is
#: powered by construction (see `test_orientation_guard_mirror_goes_red`).
#: A positive value is a deliberate cap for a fast local run, and the test
#: then SKIPS rather than asserting a p-value it has no power to earn.
N_ORIENT = int(os.environ.get("TANITAD_ORIENT_CLIPS", "0"))
#: clips spent fitting the probe. Fixed, so the held-out n is
#: `scorable - 20` and the power statement does not move with the store.
N_FIT_CLIPS = 20
GT_SUFFIX = ".sam3mapgt.npz"


# --------------------------------------------------------------------------- #
# fixtures                                                                     #
# --------------------------------------------------------------------------- #
def _extrinsics():
    p = CALIB / "sensor_extrinsics.parquet"
    if not p.is_file():
        pytest.skip(f"needs $TANITAD_CALIB_DIR/sensor_extrinsics.parquet")
    from tanitad.data.physicalai import _load_chunk_extrinsics
    return _load_chunk_extrinsics(str(p))


def _joined_clips(need_pixels: bool = False):
    """Clip ids that have a SAM3 map (and pixels, when asked). sha12-keyed on
    disk, so the join goes through sha12 and the CONTENT check re-derives it."""
    if not MAP_DIR.is_dir():
        pytest.skip(f"needs $TANITAD_SAM3_MAP_DIR ({MAP_DIR})")
    ex = _extrinsics()
    have = {p.name[:-len(GT_SUFFIX)] for p in MAP_DIR.glob(f"*{GT_SUFFIX}")}
    out = sorted(c for c in ex if sha12(c) in have)
    if need_pixels:
        if not V2_DIR.is_dir():
            pytest.skip(f"needs $TANITAD_V2EP_DIR ({V2_DIR})")
        v2 = {p.name[:-len('.v2ep.pt')] for p in V2_DIR.glob("*.v2ep.pt")}
        out = [c for c in out if c in v2]
    if len(out) < 5:
        pytest.skip(f"only {len(out)} clips join map+calibration"
                    f"{'+pixels' if need_pixels else ''}")
    return out, ex


def _frames(v2_path: Path):
    d = torch.load(v2_path, map_location="cpu", weights_only=False)
    offs = torch.cat([torch.zeros(1, dtype=torch.int64),
                      torch.cumsum(d["jpeg_len"].to(torch.int64), 0)])
    return d, offs


def _decode(d, offs, i: int) -> torch.Tensor:
    import torchvision.io as tvio
    return tvio.decode_image(d["jpeg_buf"][int(offs[i]):int(offs[i + 1])],
                             mode=tvio.ImageReadMode.RGB).float() / 255.0


# =========================================================================== #
# 1. the maps join, and the loader hooks, on real files                       #
# =========================================================================== #
def test_every_map_on_the_box_opens_as_its_own_clip():
    clips, _ = _joined_clips()
    store = MapGTStore(MAP_DIR, max_open=2)
    n_frames, layouts = [], set()
    for cid in clips:
        gt = store.open(cid)
        assert gt.clip_sha12 == sha12(cid)          # identity from CONTENT
        assert gt.meta["cartesian"]["shape"] == [120, 64]
        n_frames.append(gt.n_frames)
        layouts.add(store.layout_of[sha12(cid)])
    assert len(clips) >= 100, f"only {len(clips)} clips joined"
    assert layouts == {"flat_sha12"}, f"unexpected layouts {layouts}"
    assert min(n_frames) > 0


def test_MUT_a_map_renamed_to_another_clip_is_refused():
    """MUTATION: the classic mis-copy — a GT file under another clip's name.
    The reader must catch it from the CONTENT, not the filename."""
    from tanitad.data.semantic_map_gt import ClipIdentityMismatch, open_path
    clips, _ = _joined_clips()
    a, b = clips[0], clips[1]
    path_a = MAP_DIR / f"{sha12(a)}{GT_SUFFIX}"
    open_path(path_a, a)                             # green
    with pytest.raises(ClipIdentityMismatch):
        open_path(path_a, b)                         # RED


def test_coverage_on_the_real_windows_and_the_floor_bites():
    from tanitad.data.perception_targets import MapCoverageTooLow
    clips, _ = _joined_clips()
    store = MapGTStore(MAP_DIR, max_open=2)
    n = min(int(store.open(clips[0]).n_frames), 201)
    windows = [(c, w) for c in clips[:40] for w in range(0, n - 3, 20)]
    rep = require_map_coverage(windows, store)
    assert rep["verdict"] == "PASS" and rep["frac_ok"] == 1.0
    assert rep["n_inconclusive"] == 0
    # MUTATION: add windows for clips that have no map -> the floor must bite
    ghosts = [(f"no-such-clip-{i}", w) for i in range(20) for w in range(0, n - 3, 20)]
    with pytest.raises(MapCoverageTooLow):
        require_map_coverage(windows[:len(ghosts)] + ghosts, store)


def test_alignment_assertion_holds_and_an_off_by_n_stack_goes_red():
    """⛔ The label's ``t_img_us`` equals the window's own frame timestamp — and
    MUTATION: passing RAW frames as windows (forgetting ``+ n_stack - 1``) must
    be caught, not silently mislabel every window by 2 frames."""
    clips, _ = _joined_clips()
    store = MapGTStore(MAP_DIR, max_open=2)
    cid = clips[0]
    gt = store.open(cid)
    wins = np.arange(0, gt.n_frames - 3, 17)
    t = gt.t_img_us[wins + 2].astype(np.float64)     # the CORRECT frame times
    rep = assert_frame_alignment(store, cid, wins, t, n_stack=3)
    assert rep["worst_abs_us"] == 0.0 and rep["n"] == len(wins)
    with pytest.raises(TimeMisalignment):            # the off-by-n_stack
        assert_frame_alignment(store, cid, wins, t, n_stack=1)


def test_collate_returns_the_label_grid_with_a_plausible_seen_share():
    clips, _ = _joined_clips()
    store = MapGTStore(MAP_DIR, max_open=2)
    windows = [(c, 20) for c in clips[:8]]
    b = collate_map_targets(store, windows)
    assert b.frac.shape == (8, 9, 120, 64) and b.seen.shape == (8, 120, 64)
    s = float(b.seen.float().mean())
    assert 0.05 < s < 1.0, f"seen share {s:.3f} is not a camera's footprint"
    # the fractions really are a distribution on seen cells
    out = map_soft_ce(torch.zeros(8, 9, 120, 64), b.frac, b.seen)
    assert out["n_cells"] == int(b.seen.sum()) and torch.isfinite(out["loss"])


# =========================================================================== #
# 2. the 3-D cuboid recovery                                                   #
# =========================================================================== #
def _obstacle_files(n=8):
    if not OBST_DIR.is_dir():
        pytest.skip(f"needs $TANITAD_OBSTACLE_DIR ({OBST_DIR})")
    fs = sorted(OBST_DIR.glob("*.parquet"))[:n]
    # the floor is what the CALLER asked for, capped at 3 -- a test that wants
    # one file must not be skipped for "only 1 obstacle parquets"
    if len(fs) < min(int(n), 3):
        pytest.skip(f"only {len(fs)} obstacle parquets (wanted {n})")
    return fs


def test_first_six_columns_are_agents_at_time():
    """⭐ The equivalence the duplication is admissible under: columns 0..5 of
    :func:`cuboids_at_time` are BIT-IDENTICAL to ``agents_at_time``'s."""
    n_rows = 0
    for p in _obstacle_files():
        obs = read_clip_cuboids(p)
        check_cuboid_schema(obs)
        ts = obs["timestamp_us"] / 1e6
        for t in np.linspace(ts.min() + 0.5, ts.max() - 0.5, 7):
            r = _equivalence_probe(obs, float(t))
            assert r["same_rows"], r
            assert r["bit_identical"], f"max |delta| {r['max_abs_delta']}"
            n_rows += r["n_a"]
    assert n_rows > 200, f"only {n_rows} rows compared — no power"


def test_the_join_carries_no_z_and_the_parquet_does():
    """The gap this module exists to close, MEASURED rather than assumed."""
    p = _obstacle_files(1)[0]
    obs = read_clip_cuboids(p)
    assert "center_z" in obs and "size_z" in obs
    ts = obs["timestamp_us"] / 1e6
    t = float(np.median(ts))
    a6 = agents_at_time(obs, t)
    c8 = cuboids_at_time(obs, t)
    assert a6.shape[1] == 6 and c8.shape[1] == 8
    if c8.shape[0]:
        assert np.all(c8[:, 7] > 0.0), "a height must be positive"
        zh = zh_by_track(obs, t)
        assert len(zh) == c8.shape[0]


def test_cuboids_are_ground_standing_as_the_spec_says():
    """``SPEC_REFCV6_V2.md`` §6: *ground-standing bottom faces*.

    ⚠️ **This test was RED at 8 clips and is the reason it now runs on 40.**
    MEASURED 2026-09-16: on 8 parquets ``person``'s bottom median read 0.608 m
    and this assertion failed; over the FULL 145-file eval corpus (945,091
    cuboids) it is 0.010 m, and the pooled median is -0.004 m. The 8-clip number
    was not a defect in the data, it was a sample too small to carry the claim —
    so the sample size is now part of the test, not an accident of the default.

    Reported PER CLASS: ``protruding_object`` is airborne by definition
    (median +0.861 m over the corpus) and pooling it would move the answer.
    """
    bots, hs = {}, {}
    for p in _obstacle_files(40):
        obs = read_clip_cuboids(p)
        b = obs["center_z"] - obs["size_z"] / 2.0
        c = obs["label_class"].astype(str)
        for k in set(c.tolist()):
            m = c == k
            bots.setdefault(k, []).append(b[m])
            hs.setdefault(k, []).append(obs["size_z"][m])
    # the ROAD USERS are ground-standing; protruding_object is NOT and is named
    for cls in ("automobile", "person"):
        if cls not in bots:
            continue
        b = np.concatenate(bots[cls])
        assert b.size > 20_000, f"{cls}: only {b.size} cuboids — too few to claim"
        assert abs(float(np.median(b))) < 0.25, (
            f"{cls} bottom median {float(np.median(b)):.3f} m is not ground")
        h = np.concatenate(hs[cls])
        assert 0.8 < float(np.median(h)) < 2.5, f"{cls} h {float(np.median(h)):.3f}"
    if "protruding_object" in bots:
        po = float(np.median(np.concatenate(bots["protruding_object"])))
        assert po > 0.4, (f"protruding_object bottom median {po:.3f} m — the "
                          f"class that is NOT ground-standing now looks like it "
                          f"is, so the per-class split has stopped separating")
    pooled = float(np.median(np.concatenate(
        [np.concatenate(v) for v in bots.values()])))
    assert abs(pooled) < 0.1, f"pooled bottom median {pooled:.3f} m"
    # the helper agrees with the arithmetic done here
    st = ground_bottom_stats(read_clip_cuboids(_obstacle_files(1)[0]))
    assert set(st["per_class"]) <= set(bots) and st["n"] > 0


def test_MUT_a_2d_obstacle_table_is_refused_not_imputed():
    from tanitad.data.agent_cuboid_gt import CuboidSchemaMismatch, CuboidFrameMismatch
    obs = read_clip_cuboids(_obstacle_files(1)[0])
    flat = {k: v for k, v in obs.items() if k not in ("center_z", "size_z")}
    with pytest.raises(CuboidSchemaMismatch, match="must be REPORTED"):
        check_cuboid_schema(flat)
    bad = dict(obs)
    bad["reference_frame"] = np.full(len(obs["timestamp_us"]), "world")
    with pytest.raises(CuboidFrameMismatch):
        check_cuboid_schema(bad)
    neg = dict(obs)
    neg["size_z"] = np.zeros_like(obs["size_z"])
    with pytest.raises(CuboidSchemaMismatch, match="non-positive"):
        check_cuboid_schema(neg)


# =========================================================================== #
# 3. the orientation guard — the MIRROR must go RED                            #
# =========================================================================== #
def _probe_samples(cid, ex, store, n_frames_per_clip=3, x_hi=40.0):
    geo = L.build_lift_geometry(ex[cid], stride=16)
    grid_t = geo.grid
    grid_m = mirror_grid(grid_t)
    d, offs = _frames(V2_DIR / f"{cid}.v2ep.pt")
    gt = store.open(cid)
    idx = np.linspace(10, gt.n_frames - 10, n_frames_per_clip).astype(int)
    mf = gt.read(idx)
    X_all, _ = L.cell_centers_xy(GRID_DEFAULT)
    band = torch.as_tensor(X_all <= x_hi) & geo.valid.any(0)
    Xt, Xm, ys = [], [], []
    for k, fi in enumerate(idx):
        img = _decode(d, offs, int(fi))
        ft = lift_image_features(img, grid_t)
        fm = lift_image_features(img, grid_m)
        m, y = cells_for_probe(mf.seen[k], mf.cart[k][DRIVABLE_CHANNEL], band)
        if int(m.sum()) < 100 or y.mean() in (0.0, 1.0):
            continue
        Xt.append(ft[m].numpy())
        Xm.append(fm[m].numpy())
        ys.append(y)
    if not ys:
        return None
    return (sha12(cid), np.concatenate(Xt), np.concatenate(Xm),
            np.concatenate(ys))


def _all_probe_samples(store, clips, ex):
    out = []
    for cid in clips:
        smp = _probe_samples(cid, ex, store)
        if smp is not None:
            out.append(smp)
    return out


@pytest.mark.slow
def test_orientation_guard_mirror_goes_red():
    """⛔ THE GUARD. A fitted image->drivable probe through the lift's own grid
    must beat the same probe through a LEFT-RIGHT MIRRORED grid, on held-out
    clips, by an exact sign test.

    ## The n, and how it was chosen

    ⛔ **This test is POWERED BY CONSTRUCTION: it scores EVERY scorable clip in
    the map store.** It does not sample, and there is no tuned subset.

    ``lift_orientation.MIN_CLIPS_FOR_POWER`` = **96** is *computed*
    (``lift_orientation.min_n_for_power``), not picked: the smallest held-out n
    at which the one-sided sign test at alpha = 0.01 rejects with probability
    >= 0.95 when the true per-clip win rate is 0.70 -- the LOWEST of the three
    rates this instrument actually measured (0.7304 / 0.7200 / 1.0000). With the
    135 maps on this box and 20 clips spent on the fit, n = 115 and the power is
    **0.977**.

    ⚠️ **Why this is spelled out.** The first version of this test sliced
    ``clips[:max(N_ORIENT, 12)]`` and so scored ~20 held-out clips on a default
    run. At n = 20 the power is **0.24**: a CORRECT lift was expected to be
    reported RED about three runs in four, and it duly was --
    ``AUC 0.6770 true vs 0.6056 mirror, wins 14/20, p = 5.77e-02 -> FAIL``, with
    the direction right and the margin (+0.0714) LARGER than the headline. A
    guard that goes red on its own default run cannot be told apart from the
    defect it exists to catch. Below ``MIN_CLIPS_FOR_POWER`` this test now SKIPS
    and names the n it needs; it never asserts a p-value it has no power to earn.

    ``$TANITAD_ORIENT_CLIPS`` caps the clip count for a fast local run. That is a
    deliberate under-powering, so it skips -- honestly -- rather than failing.
    """
    clips, ex = _joined_clips(need_pixels=True)
    if N_ORIENT:                       # an explicit, deliberate cap
        clips = clips[:N_ORIENT]
    store = MapGTStore(MAP_DIR, max_open=2)
    samples = _all_probe_samples(store, clips, ex)
    n_fit = N_FIT_CLIPS
    n_eval = len(samples) - n_fit
    if n_eval < MIN_CLIPS_FOR_POWER:
        pytest.skip(
            f"UNDERPOWERED, not failed: {n_eval} held-out clips "
            f"({len(samples)} scorable - {n_fit} spent on the fit) is below "
            f"MIN_CLIPS_FOR_POWER = {MIN_CLIPS_FOR_POWER}, where the sign test "
            f"reaches power {sign_test_power(MIN_CLIPS_FOR_POWER):.2f} at the "
            f"measured win rate. At n = {n_eval} the power is only "
            f"{sign_test_power(max(n_eval, 1)):.2f}, so neither a PASS nor a "
            f"FAIL here would mean anything. Unset $TANITAD_ORIENT_CLIPS to "
            f"score all {len(clips)} clips.")
    r = orientation_probe(samples[:n_fit], samples[n_fit:])
    assert r.n_clips_scored >= MIN_CLIPS_FOR_POWER, r.n_clips_scored
    assert r.verdict == "PASS", (
        f"the orientation guard did not separate TRUE from MIRROR: "
        f"AUC {r.mean_auc_true:.4f} vs {r.mean_auc_mirror:.4f}, wins "
        f"{r.n_wins}/{r.n_clips_scored} (p = {r.p_sign:.2e}, power "
        f"{sign_test_power(r.n_clips_scored):.2f}), margin {r.mean_margin:+.4f}")
    assert r.mean_auc_true > r.mean_auc_mirror
    assert r.mean_auc_true > 0.6, (
        f"the probe itself is weak (AUC {r.mean_auc_true:.4f}) -- a guard that "
        f"cannot see the road cannot see a mirror either")


@pytest.mark.slow
def test_MUT_the_guard_itself_fails_when_the_lift_is_mirrored():
    """MUTATION: hand the guard a MIRRORED lift as the 'true' geometry. It must
    then FAIL -- a guard that passes a mirrored build is decoration.

    ⭐ This direction needs no power argument and is asserted at ANY n: under a
    mirrored lift the win rate is ~0.27, so the one-sided test cannot reject and
    the verdict is FAIL however few clips there are. The asymmetry is the point
    -- the guard is conservative in exactly the direction that matters.
    """
    clips, ex = _joined_clips(need_pixels=True)
    if N_ORIENT:
        clips = clips[:N_ORIENT]
    store = MapGTStore(MAP_DIR, max_open=2)
    samples = _all_probe_samples(store, clips, ex)
    n_fit = min(N_FIT_CLIPS, max(len(samples) // 3, 4))
    if len(samples) - n_fit < 5:
        pytest.skip(f"only {len(samples)} scorable clips")
    #        the SWAP: column 1 (true) and column 2 (mirror) exchange roles
    swapped = [(s[0], s[2], s[1], s[3]) for s in samples]
    r = orientation_probe(swapped[:n_fit], swapped[n_fit:])
    assert r.verdict == "FAIL", (
        f"the guard PASSED a mirrored lift: AUC {r.mean_auc_true:.4f} vs "
        f"{r.mean_auc_mirror:.4f}, wins {r.n_wins}/{r.n_clips_scored}")
    assert r.mean_auc_true < r.mean_auc_mirror


def test_the_power_constant_is_computed_not_chosen():
    """⛔ ``MIN_CLIPS_FOR_POWER`` must equal what the power calculation returns,
    so the two can never drift -- and the n the first version ran at must still
    be visibly underpowered, or this whole story has quietly stopped being true.
    """
    from tanitad.data.lift_orientation import MEASURED_WIN_RATE, min_n_for_power
    assert MIN_CLIPS_FOR_POWER == min_n_for_power(0.95, MEASURED_WIN_RATE, 0.01)
    assert sign_test_power(MIN_CLIPS_FOR_POWER) >= 0.95
    assert sign_test_power(MIN_CLIPS_FOR_POWER - 1) < 0.95
    assert sign_test_power(20) < 0.30, "n = 20 is no longer the cautionary case"
    # the measured rate is the LOWEST of the three observed, not the best
    assert MEASURED_WIN_RATE <= 0.7200


# =========================================================================== #
# 4. end to end: real frames -> lift -> BEV encoder -> MAP loss                #
# =========================================================================== #
def test_the_whole_map_branch_runs_on_real_frames_and_the_loss_reaches_the_trunk():
    clips, ex = _joined_clips(need_pixels=True)
    store = MapGTStore(MAP_DIR, max_open=2)
    cid = clips[0]
    geo = L.build_lift_geometry(ex[cid], stride=16)
    assert geo.feat_hw == (16, 40), f"stride-16 map is {geo.feat_hw}"
    d_img = 24
    fmap = torch.randn(2, d_img, 16, 40, requires_grad=True)   # a trunk stand-in
    lift = BEVLift(d_in=d_img, d_out=32, n_heights=len(geo.heights_m),
                   feat_hw=(16, 40))
    grid = geo.grid.unsqueeze(0).expand(2, -1, -1, -1, -1).contiguous()
    valid = geo.valid.unsqueeze(0).expand(2, -1, -1, -1).contiguous()
    bev = lift(fmap, grid, valid)
    assert tuple(bev.shape) == (2, 32, 120, 64)
    from tanitad.models.bev_encoder import BEVEncoderConfig
    br = BEVMapBranch(BEVEncoderConfig(d_in=32, d_model=32, d_out=32,
                                       dilations=(1, 2), norm_groups=8))
    out = br(bev)
    assert tuple(out["map_logits"].shape) == (2, 9, 120, 64)
    gt = store.open(cid)
    mf = gt.read([20, 40])
    frac = torch.from_numpy(mf.cart)
    seen = torch.from_numpy(mf.seen)
    loss = map_soft_ce(out["map_logits"], frac, seen)
    assert loss["n_cells"] == int(seen.sum()) > 1000
    loss["loss"].backward()
    assert fmap.grad is not None and float(fmap.grad.abs().sum()) > 0.0, (
        "the map loss does not reach the trunk")
    m = map_metrics(out["map_logits"].detach(), frac, seen)
    assert m["n_cells"] == loss["n_cells"] and len(m["iou"]) == 9


def test_the_lift_puts_left_on_the_left_for_every_clip():
    """The parameter-free half of the orientation guard: a rig point at +y
    (LEFT) must project LEFT of the frame centre, for every clip's own mount."""
    clips, ex = _joined_clips()
    fr = L.PHYSICALAI_WIDE120_256x640
    mid = (fr.width - 1) / 2.0
    left = torch.tensor([[20.0, 6.0, 0.0]], dtype=torch.float64)
    right = torch.tensor([[20.0, -6.0, 0.0]], dtype=torch.float64)
    n = 0
    for cid in clips:
        cam = L.camera_from_extrinsics(ex[cid], fr)
        cl = float(L.project_rig_points(left, cam)["col"][0])
        cr = float(L.project_rig_points(right, cam)["col"][0])
        assert cl < mid < cr, (
            f"[{sha12(cid)}] +y (LEFT) projected to column {cl:.1f} and -y to "
            f"{cr:.1f}, centre {mid:.1f} — the lift is MIRRORED")
        n += 1
    assert n >= 100, f"only {n} clips checked"
    # the mutation: a mirrored column index breaks it on every clip
    cam = L.camera_from_extrinsics(ex[clips[0]], fr)
    cl = (fr.width - 1) - float(L.project_rig_points(left, cam)["col"][0])
    cr = (fr.width - 1) - float(L.project_rig_points(right, cam)["col"][0])
    assert not (cl < mid < cr), "the mirror cost nothing"


_ = (F, glob, hashlib)
