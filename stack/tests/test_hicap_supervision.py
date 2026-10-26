"""Imagination supervision producer (`tanitad.hicap.supervision`): synthetic scenes with known answers + the FullLoss contract."""
import math
import numpy as np, pytest
from tanitad.hicap import supervision as S, vocab as V

DT, GS, K0 = 0.1, 0.1, 2                    # synthetic v9 clock: t_ego(k) = GS + k DT + t_ego0 with t_ego0 = -0.2 -> row j at 0.1 + 0.1 j s


def row_at(t): return int(round((t - 0.1) / DT))


def make_ego(v=10.0, T=40.0):
    t = np.arange(-0.2, T, 0.01)
    return S.EgoTrack(t, v * (t + 0.2), np.zeros_like(t), np.zeros_like(t), np.full_like(t, v))


def make_obs(ego, agents, t_end=20.0):
    """agents: list of (track_id, cls, f(t) -> (x, y, yaw) WORLD, length, width). Boxes are written in the RIG frame at their own
    timestamps (10 Hz, small per-track phase), exactly as obstacle.offline stores them."""
    cols = {k: [] for k in ("timestamp_us", "track_id", "center_x", "center_y", "size_x", "size_y", "orientation_x", "orientation_y",
                             "orientation_z", "orientation_w", "label_class")}
    for n, (tid, cls, f, L, W) in enumerate(agents):
        ts = np.arange(0.02 + 0.013 * n, t_end, 0.1)
        x, y, yaw = f(ts); ex, ey, eyaw, _ = ego.at(ts)
        cx, cy = S.to_frame(x, y, ex, ey, eyaw); ry = yaw - eyaw
        cols["timestamp_us"] += list((ts * 1e6).astype(np.uint64)); cols["track_id"] += [str(tid)] * len(ts)
        cols["center_x"] += list(cx); cols["center_y"] += list(cy); cols["size_x"] += [L] * len(ts); cols["size_y"] += [W] * len(ts)
        cols["orientation_x"] += [0.0] * len(ts); cols["orientation_y"] += [0.0] * len(ts)
        cols["orientation_z"] += list(np.sin(ry / 2)); cols["orientation_w"] += list(np.cos(ry / 2)); cols["label_class"] += [cls] * len(ts)
    return {k: np.asarray(v) for k, v in cols.items()}


def build(ego, obs, n_rows=150, **kw):
    return S.build_rows(ego, obs, GS, DT, K0, n_rows, probe=kw.pop("probe", False), **kw)


def const(x0, y0, vx, vy=0.0):
    return lambda t: (x0 + vx * t, y0 + vy * t + 0 * t, np.zeros_like(t) if vy == 0 else np.full_like(t, math.atan2(vy, vx)))


# ------------------------------------------------------------------ consistency with the consumers
def test_constants_match_imagination_and_config():
    torch = pytest.importorskip("torch")
    from tanitad.hicap import imagination as I
    from tanitad.hicap.model import HiCAPConfig
    assert S.H_T == I.H_T and S.MODE_ACC == I.MODE_ACC and S.FAMILIES == I.FAMILIES and len(S.COST_CHANNELS) == HiCAPConfig().n_cost
    J = np.array([[0.0, 1.0, 5.0], [2.0, 0.5, 0.1]]); star = np.array([2, 0])
    assert np.allclose(S.tier0_soft_target(J, star), I.tier0_target(torch.tensor(J), torch.tensor(star)).numpy(), atol=1e-6)


def test_clock_uses_first_egomotion_sample():
    ego = make_ego(); t = S.clip_clock(ego, GS, DT, 5)
    assert np.allclose(t, GS + np.arange(5) * DT - 0.2)


# ------------------------------------------------------------------ geometry
def test_obb_distance_exact_and_overlap():
    a = S.box_corners(np.array(0.0), np.array(0.0), np.array(0.0), 4.0, 2.0)
    b = S.box_corners(np.array(7.0), np.array(0.0), np.array(0.0), 2.0, 2.0)
    d = S.obb_distance(a, np.zeros(2), np.array(0.0), 4.0, 2.0, b, np.array([7.0, 0.0]), np.array(0.0), 2.0, 2.0)
    assert abs(float(d) - 4.0) < 1e-9
    r = np.pi / 4; c = S.box_corners(np.array(5.0), np.array(3.0), np.array(r), 2.0, 2.0)          # rotated: compare to dense sampling
    d2 = float(S.obb_distance(a, np.zeros(2), np.array(0.0), 4.0, 2.0, c, np.array([5.0, 3.0]), np.array(r), 2.0, 2.0))
    g = np.linspace(-0.5, 0.5, 201); u, w = np.meshgrid(g, g)
    pa = np.stack([u.ravel() * 4.0, w.ravel() * 2.0], 1)
    pc = np.stack([u.ravel() * 2.0, w.ravel() * 2.0], 1) @ np.array([[np.cos(r), np.sin(r)], [-np.sin(r), np.cos(r)]]) + [5.0, 3.0]
    brute = min(np.linalg.norm(pa[::50, None] - pc[None, ::7], axis=-1).min(), np.inf)
    assert d2 <= brute + 1e-6 and d2 > brute - 0.1
    assert S.obb_overlap(a, np.array(0.0), S.box_corners(np.array(1.0), np.array(0.5), np.array(0.3), 4.0, 2.0), np.array(0.3))


# ------------------------------------------------------------------ Tier 1 scenes
def test_stationary_lead():
    ego = make_ego(10.0); L = 4.8
    R = build(ego, make_obs(ego, [(1, "automobile", const(120.0, 0.0, 0.0), L, 1.9)]))
    j = row_at(6.0); t_now = 0.1 + 0.1 * j; x_ego = 10 * (t_now + 0.2)
    assert R["exo_w"][j] == 1 and R["exo_has_lead"][j] == 1
    gap0 = 120.0 - L / 2 - x_ego
    assert abs(R["lead_gap0"][j] - gap0) < 0.3
    assert np.allclose(R["exo_gap"][j], gap0 - 10.0 * np.array(S.H_T), atol=0.3)          # L(h) constant: gap if the ego held v0
    assert np.allclose(R["exo_close"][j], 10.0, atol=0.2) and np.allclose(R["exo_acc"][j], 0.0, atol=0.2)
    assert R["exo_mode"][j] == S.MODE_NAMES.index("hold") and R["exo_mode4"][j] == 0 and R["exo_onset"][j] == pytest.approx(S.MODE_WIN_S)
    assert (R["exo_hmask"][j] == 1).all() and R["exo_cutout"][j] == 0


def test_braking_lead_known_onset():
    ego = make_ego(15.0); tb, a = 8.0, -4.0
    def f(t):
        dt_ = np.maximum(t - tb, 0.0); tstop = 15.0 / -a; dd = np.minimum(dt_, tstop)
        return 40.0 + 15.0 * (t + 0.2) + 15.0 * dd + 0.5 * a * dd ** 2 - 15.0 * dt_, np.zeros_like(t), np.zeros_like(t)
    R = build(ego, make_obs(ego, [(7, "automobile", f, 4.5, 1.9)]))
    j = row_at(6.0)
    assert R["exo_has_lead"][j] == 1 and R["exo_mode"][j] == S.MODE_NAMES.index("brake") and R["exo_mode4"][j] == 1
    assert abs(R["exo_onset"][j] - (tb - (0.1 + 0.1 * j))) < 0.35                               # onset ~ 2.0 s after NOW
    assert R["exo_acc"][j, 3] < -3.0 and abs(R["exo_acc"][j, 0]) < 0.3                         # still cruising at +0.5 s, braking at +3 s
    q = R["exo_onset_q"][j]; assert q[0] <= q[1] <= q[2] and q[0] > 1.5
    jh = row_at(1.0); assert R["exo_mode"][jh] == S.MODE_NAMES.index("hold")                     # 7 s before the brake: holds


def test_hard_brake_maps_to_brake_hard():
    ego = make_ego(15.0); tb, a = 8.0, -7.0
    def f(t):
        dt_ = np.maximum(t - tb, 0.0); dd = np.minimum(dt_, 15.0 / -a)
        return 40.0 + 15.0 * (t + 0.2) + 15.0 * dd + 0.5 * a * dd ** 2 - 15.0 * dt_, np.zeros_like(t), np.zeros_like(t)
    R = build(ego, make_obs(ego, [(7, "automobile", f, 4.5, 1.9)]))
    assert R["exo_mode"][row_at(7.0)] == S.MODE_NAMES.index("brake_hard")


def test_cut_out_is_ignore_in_mode_acc_space():
    ego = make_ego(12.0)
    def f(t):
        lat = 3.6 * np.clip((t - 7.0) / 2.0, 0, 1)
        return 30.0 + 12.0 * (t + 0.2), lat, np.zeros_like(t)
    R = build(ego, make_obs(ego, [(3, "automobile", f, 4.5, 1.9)]))
    j = row_at(6.0)
    assert R["exo_has_lead"][j] == 1 and R["exo_cutout"][j] == 1 and R["exo_mode4"][j] == 3 and R["exo_mode"][j] == S.IGNORE
    assert 0.5 < R["exo_onset"][j] < 3.0


def test_no_lead_and_cut_in():
    ego = make_ego(10.0)
    side = (5, "automobile", const(30.0, 3.6, 10.0), 4.5, 1.9)                      # adjacent lane, same speed: never in path
    R = build(ego, make_obs(ego, [side, (6, "person", const(5.0, -6.0, 0.0), 0.6, 0.6)]))
    assert R["exo_w"][row_at(6.0)] == 1 and R["exo_has_lead"].sum() == 0 and R["exo_cutin"].sum() == 0
    def merge(t):
        lat = 3.6 * (1 - np.clip((t - 7.0) / 1.5, 0, 1))
        return 25.0 + 10.0 * (t + 0.2), lat, np.zeros_like(t)
    R2 = build(ego, make_obs(ego, [(9, "automobile", merge, 4.5, 1.9)]))
    j = row_at(6.0)
    assert R2["exo_has_lead"][j] == 0 and R2["exo_cutin"][j] == 1 and 0.5 < R2["exo_cutin_t"][j] < 3.0


def test_follower_is_not_a_cut_in_or_lead():
    ego = make_ego(8.0)
    R = build(ego, make_obs(ego, [(2, "automobile", const(-10.0, 0.0, 12.0), 4.5, 1.9)]))       # faster car from behind in our lane
    assert R["exo_cutin"].sum() == 0 and R["exo_has_lead"][: row_at(3.0)].sum() == 0


def test_no_label_is_not_no_lead():
    ego = make_ego(10.0)
    R = build(ego, None)
    assert R["exo_w"].sum() == 0 and R["J_w"].sum() == 0
    R2 = build(ego, make_obs(ego, [(1, "automobile", const(300.0, 0.0, 0.0), 4.5, 1.9)], t_end=10.0))
    assert R2["exo_w"][row_at(5.0)] == 1 and R2["exo_w"][row_at(8.0)] == 0                    # labels must cover NOW .. NOW + 3 s
    assert R2["J_w"][row_at(6.0)] == 1 and R2["J_w"][row_at(8.0)] == 0


# ------------------------------------------------------------------ teacher cost
def _agents(x, y, yaw, L=4.8, W=1.9, T=60):
    x, y, yaw = (np.broadcast_to(np.asarray(z, float), (T,))[None].copy() for z in (x, y, yaw))
    return dict(x=x, y=y, yaw=yaw, length=np.array([L]), width=np.array([W]), valid=np.ones((1, T), bool))


def test_ttc_rear_end_and_head_on():
    t = (np.arange(60) + 1) * DT
    go = np.stack([10.0 * t, 0 * t], 1); stop = np.stack([np.minimum(10.0 * t - 5 * t ** 2, 5.0) * (t < 1) + 5.0 * (t >= 1), 0 * t], 1)
    c = S.candidate_costs(np.stack([go, stop]), _agents(15.0, 0.0, 0.0), 10.0, go, DT)
    assert c["collide"][0] and abs(c["ttc_s"][0] - 0.9) < 0.11 and c["cost3"][0, 0] > 0.6 and c["clear_m"][0] == 0.0
    assert not c["collide"][1] and c["clear_m"][1] > 1.0 and c["cost3"][1, 2] > 0.5              # the stop avoids it but loses progress
    onc = _agents(40.0 - 10.0 * t, np.zeros(60), np.full(60, np.pi))                             # head-on, closing 20 m/s
    h = S.candidate_costs(go[None], onc, 10.0, go, DT)
    assert abs(h["ttc_s"][0] - 1.7) < 0.11 and h["J"][0] > S.J_WEIGHTS[1]
    assert S.candidate_costs(go[None], None, 10.0, go, DT)["J"][0] == 0.0                      # empty road, logged speed: zero cost


def test_one_sided_ignores_logged_follower():
    t = (np.arange(60) + 1) * DT
    brake = np.stack([np.cumsum(np.maximum(10 - 3 * t, 0) * DT), 0 * t], 1)
    follower = _agents(-8.0 + 10.0 * t, np.zeros(60), np.zeros(60))                              # keeps 10 m/s: would rear-end a braking ego
    assert not S.candidate_costs(brake[None], follower, 10.0, brake, DT)["collide"][0]
    S.ONE_SIDED = False
    try:
        assert S.candidate_costs(brake[None], follower, 10.0, brake, DT)["collide"][0]
    finally:
        S.ONE_SIDED = True


def test_probe_logged_trajectory_is_collision_free_in_scene():
    ego = make_ego(10.0)
    R = build(ego, make_obs(ego, [(1, "automobile", const(30.0, 0.0, 10.0), 4.5, 1.9), (2, "automobile", const(-15.0, 0.0, 10.0), 4.5, 1.9)]),
              n_rows=40, probe=True)
    ok = R["J_w"] > 0
    assert ok.any() and not np.isfinite(R["probe_ttc"][ok, 0]).any()                            # the logged ego never hits anything
    assert np.isfinite(R["probe_ttc"][ok, 4]).all() and (R["probe_J"][ok, 4] > R["probe_J"][ok, 0]).all()     # +1.5 m/s^2 closes the 22 m gap


# ------------------------------------------------------------------ contract: the produced dict feeds FullLoss
def test_rows_feed_full_loss_without_keyerror():
    torch = pytest.importorskip("torch")
    from tanitad.hicap.full import HiCAPFull, FullLoss
    from tanitad.hicap.model import HiCAPConfig
    from test_hicap_architecture import SMALL, synth_world, build_vt, label_batch
    p, lat, dv, lon, v0 = synth_world(120)
    vocab = V.build_factored_vocab(p, lat, dv, lon, v0, K_P=24, K_V=12, floor_P=2, floor_V=2)
    ego = make_ego(10.0)
    def brake(t):
        dt_ = np.maximum(t - 6.0, 0.0); dd = np.minimum(dt_, 3.0)
        return 30.0 + 10.0 * (t + 0.2) + 10.0 * dd - 0.5 * 3.33 * dd ** 2 - 10.0 * dt_, np.zeros_like(t), np.zeros_like(t)
    R = build(ego, make_obs(ego, [(1, "automobile", brake, 4.5, 1.9)]), n_rows=60, vocab=vocab, n_cand=5)
    rows = np.flatnonzero((R["exo_w"] > 0) & (R["J_w"] > 0))[:4]
    assert len(rows) == 4 and R["exo_has_lead"][rows].all()
    c = HiCAPConfig(**SMALL); m = HiCAPFull(c); Vt = build_vt(c, vocab)
    b = label_batch(c, vocab, B=4)
    for k, v in S.trainer_batch_keys(R, rows).items():
        b[k] = torch.from_numpy(np.ascontiguousarray(v))
    assert b["cand_p"].shape == (4, 5) and b["cost_t"].shape == (4, 5, 3) and b["cf_cost"].shape == (4, 5, 4) and b["J_teacher"].dtype == torch.float32
    tot, parts = FullLoss()(m, Vt, b, stage=3)
    assert {"exo", "tier2", "tier0", "cost"} <= set(parts) and torch.isfinite(tot)
    tot.backward()
    assert m.ehead.m.net[0].weight.grad is not None
    y = R["y_tier0"][rows]; assert np.allclose(y.sum(-1), 1.0, atol=1e-5) and (y[:, 0] >= 0.3).all()
