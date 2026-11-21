"""Synthetic checks of t2lib.analyse: a known launch, known non-launches, a known guard, the three conditions, the lead stratum."""
import sys
sys.path.insert(0, "E:/Projects/TanitAD-artifacts/dagland/motion_c/t2_scenes/code")
import numpy as np
import t2lib

EMPTY = ([], [], [], [])


def scene(speeds_by_tick):
    n = len(speeds_by_tick) + 1
    xs = np.concatenate([[0.0], np.cumsum(np.asarray(speeds_by_tick) * 0.1)])
    T = np.tile(np.eye(4), (n, 1, 1))
    T[:, 0, 3] = xs
    return T, (np.arange(n) * 100000 + 1_000_000).astype(np.int64)


def one_track(label, T_x_at_ts, tus, dx, dy):
    ts_all = tus.astype(float)
    poses = np.tile(np.array([[0.0, dy, 0, 0, 0, 0, 1.0]]), (len(ts_all), 1))
    poses[:, 0] = dx
    return (["a"], [label], [poses], [ts_all])


# 1: stand still 8 s then accelerate 1 m/s2 to 12 m/s: exactly one launch event
v = np.concatenate([np.zeros(80), np.minimum(np.arange(1, 221) * 0.1, 12.0)])
T, tus = scene(v)
r = t2lib.analyse(T, tus, EMPTY)
assert r["is_launch"] and len(r["launch_events"]) == 1 and not r["is_guard"], r
e = r["launch_events"][0]
print("launch", e)
# the lead track is a static world point; the ego frame x of it at t_s is (world_x - ego_x(t_s))
m = int(np.searchsorted(tus, tus[0] + int(round(e["t_s_s"] * 1e6))))
x_ego = T[m, 0, 3]


def lead_for(label, dx_rel, dy):
    tr = one_track(label, None, tus, x_ego + dx_rel, dy)
    return t2lib.analyse(T, tus, tr)["launch_events"][0]


assert lead_for("automobile", 10.0, 0.5)["lead"] is True
assert abs(lead_for("automobile", 10.0, 0.5)["lead_detail"]["x_m"] - 10.0) < 1e-6
assert lead_for("automobile", 10.0, 2.5)["lead"] is False      # |y| > 2
assert lead_for("person", 10.0, 0.0)["lead"] is False          # not a vehicle class
assert lead_for("automobile", 31.0, 0.0)["lead"] is False      # x > 30
assert lead_for("automobile", 30.0, 2.0)["lead"] is True       # both bounds inclusive
assert lead_for("automobile", -5.0, 0.0)["lead"] is False      # behind
# the track only annotated near tick 40 (long before t_s): no annotation within 0.15 s of t_s, so no lead
tr = (["a"], ["automobile"], [np.array([[x_ego + 10.0, 0.0, 0, 0, 0, 0, 1.0]])], [np.array([float(tus[40])])])
assert t2lib.analyse(T, tus, tr)["launch_events"][0]["lead"] is False
# 2: the crossing happens before first tick + 2 s: not an event
v = np.concatenate([np.full(5, 2.0), np.linspace(2.0, 8.0, 12), np.full(200, 8.0)])
T, tus = scene(v)
r = t2lib.analyse(T, tus, EMPTY)
print("early", r["launch_events"], r["guard_share"])
assert not r["is_launch"]
# 3: the clip ends < 6 s after the crossing: not an event
v = np.concatenate([np.zeros(80), np.linspace(0, 10, 25), np.full(30, 10.0)])
T, tus = scene(v)
assert not t2lib.analyse(T, tus, EMPTY)["is_launch"]
# 4: never at or below 3 m/s in the 5 s before the crossing: not an event
v = np.concatenate([np.full(100, 4.0), np.linspace(4.0, 9.0, 40), np.full(120, 9.0)])
T, tus = scene(v)
assert not t2lib.analyse(T, tus, EMPTY)["is_launch"]
# 5: guard
T, tus = scene(np.full(250, 10.0))
r = t2lib.analyse(T, tus, EMPTY)
print("guard", r["guard_share"], r["is_guard"])
assert r["is_guard"] and not r["is_launch"]
# 6: slow scene is neither
T, tus = scene(np.full(250, 2.0))
r = t2lib.analyse(T, tus, EMPTY)
assert not r["is_guard"] and not r["is_launch"]
# 7: a launch scene is never a guard, even with a high share
v = np.concatenate([np.zeros(60), np.full(300, 12.0)])
T, tus = scene(v)
r = t2lib.analyse(T, tus, EMPTY)
assert r["is_launch"] and not r["is_guard"] and r["guard_share"] > 0.5
# 8: controls: a constant 7 m/s line reads exactly 7.0 where defined, NaN on the five edge ticks each side
T, tus = scene(np.full(60, 7.0))
vv = t2lib.speed_series(T[:, :2, 3])
assert np.allclose(vv[5:-5], 7.0) and np.isnan(vv[:5]).all() and np.isnan(vv[-5:]).all()
# 9: mutation: moving the crossing threshold to 6.0 must change the event time of case 1 (the check is live)
v = np.concatenate([np.zeros(80), np.minimum(np.arange(1, 221) * 0.1, 12.0)])
T, tus = scene(v)
t_a = t2lib.analyse(T, tus, EMPTY)["launch_events"][0]["t_c_s"]
t2lib.V_UP = 6.0
t_b = t2lib.analyse(T, tus, EMPTY)["launch_events"][0]["t_c_s"]
t2lib.V_UP = 5.2
assert t_b > t_a, (t_a, t_b)
print("SELFTEST PASS")
