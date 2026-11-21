"""Local tests of the CROPREAD data-side kit (synthetic data, no pod, no GPU, no network).

Run the whole file:  PYCAP_MB=3000 bash /f/TanitAD-scratch/v8proof/ops/pycap.sh -m pytest test_enc.py -q -p no:cacheprovider --basetemp <new dir>
Needs the stack tree (TANITAD_STACK, default E:/Projects/TanitAD/stack); the geometry tests skip when it is absent.
"""
import json
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import enc_common as E   # noqa: E402
import make_lists as ML   # noqa: E402
import make_final as MF   # noqa: E402

torch = pytest.importorskip("torch")
HAVE_STACK = os.path.isdir(os.path.join(E.STACK, "tanitad"))
needs_stack = pytest.mark.skipif(not HAVE_STACK, reason="stack tree not found")


# ------------------------------------------------------------------------------------------------ synthetic camera
def synth_intr():
    from tanitad.data.calib import FThetaIntrinsics
    return FThetaIntrinsics(poly=(0.0, 927.5032, 23.1353, -58.5012, 16.5067), cx=958.0, cy=700.0, width=1920, height=1080, per_clip=True)


def ramp_video(T=3, h=270, w=480, seed=0):
    """uint8 [T,3,h,w]: channel 0 = horizontal ramp (value ~ column), channel 1 = vertical ramp, channel 2 = noise texture."""
    g = torch.Generator().manual_seed(seed)
    xs = torch.linspace(0, 255, w).expand(h, w)
    ys = torch.linspace(0, 255, h).unsqueeze(1).expand(h, w)
    vid = []
    for t in range(T):
        tex = (torch.rand(h, w, generator=g) * 255)
        vid.append(torch.stack([xs, ys, tex]).clamp(0, 255).to(torch.uint8))
    return torch.stack(vid)


# ------------------------------------------------------------------------------------------------ windows
@needs_stack
def test_window_with_full_frame_parameters_reproduces_the_builders_frame():
    from tanitad.data.calib import CanonicalFrame, cylindrical_rectify, cylindrical_grid
    intr = synth_intr()
    vid = ramp_video()
    full = CanonicalFrame.from_hfov(120.0, 256, 640, "cylindrical")
    ref = cylindrical_rectify(vid, intr, full)
    grid, mask = E.window_grid(intr, vid.shape[-2], vid.shape[-1], full, 0.0, 0.0)
    got = E.window_sample(vid, grid.expand(vid.shape[0], -1, -1, -1), mask.expand(vid.shape[0], -1, -1))
    assert torch.equal(ref, got)
    g0, m0 = cylindrical_grid(intr, vid.shape[-2], vid.shape[-1], full)
    assert torch.equal(g0, grid) and torch.equal(m0, mask[0])


@needs_stack
def test_window_angular_size_is_exact_in_azimuth():
    for size in (32.0, 16.0):
        f = E.window_frame(size)
        assert f.projection == "cylindrical" and f.height == f.width == 256
        assert abs(math.degrees(2.0 * f.half_angle_x_rad()) - size) < 1e-9


@needs_stack
def test_window_direction_round_trip_left_and_up_positive():
    import lead_box as LB
    for az, el in ((0.0, 0.0), (10.0, 0.0), (-7.0, 3.0), (25.0, -6.0), (-30.0, 8.0)):
        d = (E.rotation_az_el(az, el) @ torch.tensor([0.0, 0.0, 1.0])).double().numpy()
        a, e, _, _ = LB.cam_angles(d[None] * 10.0, np.eye(3), np.zeros(3))
        assert abs(a[0] - az) < 1e-4 and abs(e[0] - el) < 1e-4, (az, el, a, e)


@needs_stack
def test_window_looks_where_it_is_told():
    """Left (az>0) sees smaller columns of a horizontal ramp, up (el>0) sees smaller rows of a vertical ramp."""
    intr = synth_intr()
    vid = ramp_video(T=1)
    h, w = vid.shape[-2:]
    f = E.window_frame(32.0)

    def mean_ch(az, el, ch):
        g, m = E.window_grid(intr, h, w, f, az, el)
        out = E.window_sample(vid, g, m)
        return float(out[0, ch].float().mean())
    c0 = mean_ch(0.0, 0.0, 0)
    assert mean_ch(12.0, 0.0, 0) < c0 - 3.0           # left: smaller column values
    assert mean_ch(-12.0, 0.0, 0) > c0 + 3.0
    r0 = mean_ch(0.0, 0.0, 1)
    assert mean_ch(0.0, 8.0, 1) < r0 - 3.0            # up: smaller row values
    assert mean_ch(0.0, -8.0, 1) > r0 + 3.0


@needs_stack
def test_window_matches_the_builders_frame_where_both_see_the_same_scene():
    """A W32 window cut from the raw frame equals the same region cut from the builder's rectified frame (analytic cylinder relation)."""
    import torch.nn.functional as F
    from tanitad.data.calib import CanonicalFrame, cylindrical_rectify, cylindrical_rays
    intr = synth_intr()
    vid = ramp_video(T=1, h=540, w=960)
    full = CanonicalFrame.from_hfov(120.0, 256, 640, "cylindrical")
    rect = cylindrical_rectify(vid, intr, full).float()
    f = E.window_frame(32.0)
    az, el = 6.0, -4.0
    g, m = E.window_grid(intr, 540, 960, f, az, el)
    win = E.window_sample(vid, g, m).float()
    x, y, z = cylindrical_rays(f)
    P = torch.stack([x, y, z], -1).reshape(-1, 3) @ E.rotation_az_el(az, el).T
    u = float(full.f_ref) * torch.atan2(P[:, 0], P[:, 2]) + 319.5
    v = float(full.f_ref) * P[:, 1] / torch.hypot(P[:, 0], P[:, 2]) + 127.5
    gg = torch.stack([u / 639 * 2 - 1, v / 255 * 2 - 1], -1).reshape(1, 256, 256, 2)
    ex = F.grid_sample(rect, gg, mode="bilinear", padding_mode="zeros", align_corners=False)
    a, b = win[0, :2].flatten(), ex[0, :2].flatten()                     # the two ramp channels (smooth: resolution differences vanish)
    assert float((a - b).abs().mean()) < 2.5


# ------------------------------------------------------------------------------------------------ planted fault 1: frame shift -> c1
@needs_stack
def test_c1_passes_on_matching_frames_and_fails_on_a_frame_shift_in_the_window_builder():
    import ctl_c1
    from tanitad.data.calib import CanonicalFrame
    intr = synth_intr()
    T = 14
    g = torch.Generator().manual_seed(3)
    base = torch.rand(3, 135, 240, generator=g) * 255
    frames = []
    for t in range(T + 2):                                              # a moving scene: shift the texture by 9 px per frame
        frames.append(torch.roll(base, shifts=(0, 9 * t), dims=(1, 2)).clamp(0, 255).to(torch.uint8))
    vid = torch.stack(frames)
    full = CanonicalFrame.from_hfov(120.0, 256, 640, "cylindrical")
    grid, mask = E.window_grid(intr, vid.shape[-2], vid.shape[-1], full, 0.0, 0.0)

    def tokens(idx):
        out = E.window_sample(vid[idx], grid.expand(len(idx), -1, -1, -1), mask.expand(len(idx), -1, -1)).float()
        return out.reshape(len(idx), 3, 8, 32, 20, 32).mean(dim=(3, 5)).reshape(len(idx), -1)[:, None, :]     # [n,1,3*8*20]
    cidx = np.arange(0, T, 1)
    cached = tokens(cidx)
    v = np.full(len(cidx), 5.0)
    cos, flags, _, _, _ = ctl_c1.compare_tokens(tokens(cidx), cached, v)
    ok = ctl_c1.clause_values(cos, flags)
    assert ok["clause1_pass"] and ok["clause2_pass"]
    cos_s, flags_s, _, _, _ = ctl_c1.compare_tokens(tokens(cidx + 1), cached, v)                         # THE FAULT: one frame ahead
    bad = ctl_c1.clause_values(cos_s, flags_s)
    assert not bad["clause2_pass"], bad


# ------------------------------------------------------------------------------------------------ planted fault 2: mirrored azimuth -> p1 (ii)
@needs_stack
def test_box_azimuth_is_left_positive_and_p1_clause_ii_fails_when_mirrored():
    import lead_box as LB
    R = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])        # columns: cam x -> veh -y, cam y -> veh -z, cam z -> veh +x
    t = np.array([2.0, 0.0, 1.4])
    agent = {"cx": 22.0, "cy": 3.0, "cz": 0.5, "l": 4.5, "w": 1.9, "h": 1.6, "yaw": 0.0}     # ahead and 3 m to the LEFT (+y is left)
    g = LB.box_geometry(agent, R, t)
    assert g["ctr_az"] > 0.0 and 8.0 < g["ctr_az"] < 9.5, g
    assert g["bot_el"] < 0.0
    n = 400
    kap = np.full(n, 1 / 150.0)                                              # path curving LEFT (positive)

    def item(az_sign):
        return {"clip": "c", "role": "train", "gap_join": np.full(n, 18.0), "valid": np.ones(n, bool),
                "box": {"has_lead": np.ones(n, bool), "gap_re": np.full(n, 18.0), "ctr_az": np.full(n, az_sign * g["ctr_az"]),
                        "kappa_ahead": kap, "bot_el": np.full(n, g["bot_el"])}}
    good = LB.p1_from_arrays([item(+1.0)])
    assert good["clause_i"]["pass"] and good["clause_ii"]["pass"] and good["clause_iii"]["pass"] and good["pass_all"]
    mirrored = LB.p1_from_arrays([item(-1.0)])
    assert not mirrored["clause_ii"]["pass"] and not mirrored["pass_all"]
    few = LB.p1_from_arrays([{**item(+1.0), "valid": np.arange(n) < 10, "gap_join": np.full(n, 18.0)}])
    assert not few["clause_ii"]["pass"]                                       # fewer than 200 frames counts as failed


def test_p1_clause_i_and_iii_catch_wrong_gap_and_a_box_above_the_horizon():
    import lead_box as LB
    n = 300
    base = {"has_lead": np.ones(n, bool), "gap_re": np.full(n, 10.0), "ctr_az": np.zeros(n), "kappa_ahead": np.zeros(n), "bot_el": np.full(n, -3.0)}
    it = {"clip": "c", "role": "eval", "gap_join": np.full(n, 10.0), "valid": np.ones(n, bool), "box": base}
    assert LB.p1_from_arrays([it])["clause_i"]["pass"]
    wrong = {**base, "gap_re": np.full(n, 11.0)}
    assert not LB.p1_from_arrays([{**it, "box": wrong}])["clause_i"]["pass"]
    up = {**base, "bot_el": np.full(n, 2.0)}
    assert not LB.p1_from_arrays([{**it, "box": up}])["clause_iii"]["pass"]
    assert not LB.p1_from_arrays([{**it, "box": None}])["clause_i"]["pass"]


@needs_stack
def test_kappa_ahead_is_left_positive():
    import lead_box as LB
    s = np.arange(0, 200, 1.0)
    R = 100.0
    th = s / R
    left = np.stack([R * np.sin(th), R * (1 - np.cos(th)), th, np.full_like(s, 10.0)], 1)       # turning left: yaw grows
    k = LB.kappa_ahead(left)
    assert abs(k[0] - 1 / R) < 2e-4 and np.isnan(k[-1])
    right = left.copy()
    right[:, 1], right[:, 2] = -left[:, 1], -left[:, 2]
    assert LB.kappa_ahead(right)[0] < 0


@needs_stack
def test_lead_index_agrees_with_the_join_rule_on_a_straight_road():
    import build_v9_labels as V
    import lead_box as LB
    ts = np.arange(0.0, 20.0, 0.1)
    log = V.log_from_arrays(ts, 10.0 * ts, np.zeros_like(ts), np.zeros_like(ts), np.full_like(ts, 10.0))
    tr = V.track10(log)
    p0 = V.smooth_path(tr, 0.0)
    agents = [{"cx": 12.0, "cy": 6.0, "yaw": 0.0, "l": 4.5, "w": 1.9, "cls": "automobile", "cz": 0.5, "h": 1.6},          # off the path
              {"cx": 40.0, "cy": 0.4, "yaw": 0.0, "l": 5.0, "w": 2.0, "cls": "automobile", "cz": 0.5, "h": 1.6},          # lead
              {"cx": 70.0, "cy": 0.0, "yaw": 0.0, "l": 5.0, "w": 2.0, "cls": "automobile", "cz": 0.5, "h": 1.6},
              {"cx": 20.0, "cy": 0.0, "yaw": 0.0, "l": 1.0, "w": 0.6, "cls": "pedestrian", "cz": 0.9, "h": 1.7}]          # not a lead class
    lead, gap, *_ = V.lead_for_frame(log, tr, p0, 5.0, agents)
    idx, gap2 = LB.lead_index(V, log, tr, p0, 5.0, agents)
    assert lead and idx == 1 and abs(gap - gap2) < 1e-9
    assert abs(gap - (40.0 - 0.5 * 5.0 - 3.6)) < 0.3


# ------------------------------------------------------------------------------------------------ lists
def synth_join(clips, n_per=30, lead_frames=None):
    off = np.arange(len(clips)) * n_per
    N = len(clips) * n_per
    lv = np.zeros(N, np.int8)
    gap = np.full(N, np.nan, np.float32)
    for i, c in enumerate(clips):
        k = (lead_frames or {}).get(c, 20)
        lv[off[i]:off[i] + k] = 1
        gap[off[i]:off[i] + k] = 25.0
    return {"clip": np.array(clips), "off": off, "n": np.full(len(clips), n_per), "lead_valid": lv, "lead_gap": gap}


def test_list_building_rules_order_and_record(tmp_path):
    clips = [f"clip-{i:04d}" for i in range(400)]
    few = {c: 5 for c in clips[:60]}                                          # fewer than 10 valid lead frames
    J = synth_join(clips, lead_frames=few)
    elig, cnt = ML.eligible_train(J)
    assert cnt["n_join_clips"] == 400 and cnt["n_eligible"] == len(elig)
    assert all(not ML.held_out(c) for c in elig)
    assert all(c not in elig for c in clips[:60])
    assert elig == sorted(elig, key=ML.sha_hex)
    ev = [f"eval-{i:03d}" for i in range(20)]
    rec = ML.build(synth_join(ev), J, str(tmp_path / "lists"), expect_eligible=len(elig))
    assert rec["overlap_eval_vs_train_join"] == 0 and rec["eligible_matches_expectation"]
    order = (tmp_path / "lists" / "train_order.txt").read_text().split("\n")
    assert order[0] == f"1 {elig[0]}" and len([x for x in order if x]) == min(520, len(elig))
    assert json.loads((tmp_path / "lists" / "LISTS_RECORD.json").read_text())["sha256"]["eval141.txt"] == ML.file_sha(str(tmp_path / "lists" / "eval141.txt"))


def test_list_step_refuses_an_evaluation_clip_in_the_train_list(tmp_path):
    clips = [f"clip-{i:04d}" for i in range(100)]
    J = synth_join(clips)
    ev = synth_join([clips[7], "eval-x"])                                       # PLANTED: an evaluation clip is in the train join
    with pytest.raises(SystemExit) as e:
        ML.build(ev, J, str(tmp_path / "lists"))
    assert "REFUSED" in str(e.value)
    assert not (tmp_path / "lists" / "LISTS_RECORD.json").exists()


def test_replacement_rule_picks_next_of_the_order_and_assigns_roles():
    order = [(i + 1, f"c{i + 1}") for i in range(520)]
    ledger = {c: {"ok": True} for _, c in order}
    for bad in ("c3", "c10", "c11"):
        ledger[bad] = {"ok": False, "reason": "no obstacle source for this clip"}
    roles, skipped = MF.pick(order, ledger)
    assert len(roles) == 500 and [s["clip"] for s in skipped] == ["c3", "c10", "c11"]
    assert "c3" not in [c for c, _ in roles] and roles[0] == ("c1", "fit")
    assert [r for _, r in roles].count("fit") == 450 and [r for _, r in roles].count("sel") == 50 and roles[-1][1] == "sel"
    assert roles[-1][0] == "c503"
    ledger2 = {c: {"ok": i % 2 == 0} for i, (_, c) in enumerate(order)}          # half unstageable: the walk stops at position 520
    roles2, _ = MF.pick(order, ledger2)
    assert len(roles2) == 260


def test_frame_plan_follows_the_builders_arithmetic(tmp_path):
    pd = pytest.importorskip("pandas")
    t0 = 1_000_000
    ts = t0 + np.arange(0, 6_000_000, 33_333, dtype=np.int64)                  # 30 Hz, 6 s, microseconds
    p = tmp_path / "x.timestamps.parquet"
    pd.DataFrame({"timestamp": ts}).to_parquet(p)
    plan = E.frame_plan(str(p))
    assert plan["n_ep"] == int(((ts[-1] - ts[0]) / 1e6) * 10.0) and plan["n_cache"] == math.ceil(plan["n_ep"] / 2)
    assert np.all(np.diff(plan["frame_idx"]) >= 0) and plan["cache_idx"].shape[0] == plan["n_cache"]
    assert plan["cache_idx"][0] == 0 and plan["frame_idx"][-1] == len(ts) - 1
