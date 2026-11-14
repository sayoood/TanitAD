"""Synthetic tests for S0-D.  Run in the driverl-eval venv (nuPlan fork + navsim 3e8291b via s0d_rsim's sys.path), CUDA off.

  python test_s0d.py                 the suite (no real data)
  python test_s0d.py --mutations     copies the scripts to a temp dir, applies each mutation, REQUIRES the suite to go RED
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402

import s0d_rsim as RS  # noqa: E402  (also sets sys.path for navsim / refe)
import s0d_hook as HK  # noqa: E402
from nuplan.common.actor_state.agent import Agent  # noqa: E402
from nuplan.common.actor_state.oriented_box import OrientedBox  # noqa: E402
from nuplan.common.actor_state.scene_object import SceneObjectMetadata  # noqa: E402
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D  # noqa: E402
from nuplan.common.actor_state.static_object import StaticObject  # noqa: E402
from nuplan.common.actor_state.tracked_objects import TrackedObjects  # noqa: E402
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType as TT  # noqa: E402
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks  # noqa: E402

DB_INT = 0.05
ITERS = list(range(0, 101, 10))


def meta(tok, i):
    return SceneObjectMetadata(timestamp_us=int(i * DB_INT * 1e6), token=f"{tok}-{i}", track_id=None, track_token=tok)


def agent(tok, typ, x, y, h, vx, vy, i, l=4.0, w=2.0):
    return Agent(typ, OrientedBox(StateSE2(x, y, h), l, w, 1.5), StateVector2D(vx, vy), meta(tok, i))


def static(tok, x, y, i):
    return StaticObject(TT.TRAFFIC_CONE, OrientedBox(StateSE2(x, y, 0.0), 0.5, 0.5, 1.0), meta(tok, i))


class FakeScenario:
    """11 detection frames (iterations 0..100 step 10).  car A accelerates (true path is NOT constant velocity), ped P appears
    only from step 5 (absent at t0), car D vanishes after step 5, cone C is static, cone N appears late."""
    database_interval = DB_INT

    def __init__(self):
        self.calls = []
        self.frames = {}
        for k, i in enumerate(ITERS):
            t = i * DB_INT
            objs = [agent("A", TT.VEHICLE, 10.0 + 3.0 * t + 0.8 * t * t, 1.0, 0.1, 3.0, 0.2, i),
                    static("C", 5.0, -3.0, i)]
            if k <= 5:
                objs.append(agent("D", TT.BICYCLE, -8.0 + 1.0 * t, 4.0, 0.0, 1.0, 0.0, i, l=1.8, w=0.6))
            if k >= 5:
                objs.append(agent("P", TT.PEDESTRIAN, 20.0, 6.0 - 1.2 * t, 1.5, 0.0, -1.2, i, l=0.6, w=0.6))
            if k >= 7:
                objs.append(static("N", 30.0, 2.0, i))
            self.frames[i] = DetectionsTracks(TrackedObjects(objs))

    def get_tracked_objects_at_iteration(self, iteration, future_trajectory_sampling=None):
        self.calls.append(iteration)
        return self.frames[iteration]


_PROC = {}


def proc():
    if "p" not in _PROC:
        import navsim_pdm_targets as PT
        _PROC["p"] = PT._setup()["proc"]
    return _PROC["p"]


def obs_for(mode, sc=None):
    sc = sc or FakeScenario()
    return proc()._interpolate_gt_observation(HK.HookedScenario(sc, mode)), sc


def geoms(obs, k):
    occ = obs[k]
    return {t: g.wkb for t, g in zip(occ.tokens, occ._geometries)}


def centre(obs, k, tok):
    occ = obs[k]
    i = occ.tokens.index(tok)
    c = occ._geometries[i].centroid
    return np.array([c.x, c.y])


class TestHook(unittest.TestCase):
    def test_identity_hook_is_bit_identical_to_unhooked_navsim(self):
        sc = FakeScenario()
        base = proc()._interpolate_gt_observation(sc)                    # NAVSIM's own, no hook
        hooked = proc()._interpolate_gt_observation(HK.HookedScenario(FakeScenario(), "true"))
        for k in range(51):
            self.assertEqual(geoms(base, k), geoms(hooked, k), f"step {k}")
        self.assertEqual(sorted(base.unique_objects), sorted(hooked.unique_objects))
        for tok, o in base.unique_objects.items():
            h = hooked.unique_objects[tok]
            self.assertEqual(o.tracked_object_type, h.tracked_object_type)
            if hasattr(o, "velocity"):
                self.assertEqual((o.velocity.x, o.velocity.y), (h.velocity.x, h.velocity.y))

    def test_identity_returns_the_same_objects(self):
        sc = FakeScenario()
        tr = [sc.frames[i] for i in ITERS]
        self.assertIs(HK.transform_tracks(tr, [i * DB_INT for i in ITERS], "true"), tr)

    def test_hook_reads_only_what_navsim_reads(self):
        sc = FakeScenario()
        proc()._interpolate_gt_observation(HK.HookedScenario(sc, "t0"))
        self.assertEqual(sorted(set(sc.calls)), ITERS)                   # exactly gt_indices = 0,10,..,100

    def test_t0_agents_frozen_at_t0_box_velocity_zero(self):
        obs, sc = obs_for("t0")
        a0 = geoms(obs, 0)
        for k in (1, 10, 25, 50):
            gk = geoms(obs, k)
            self.assertEqual(gk["A"], a0["A"], f"A moved at step {k}")
            self.assertEqual(gk["D"], a0["D"])
        self.assertEqual(centre(obs, 0, "A").tolist(), centre(obs_for("true")[0], 0, "A").tolist())   # t0 box = true t0 box
        for tok in ("A", "D"):
            u = obs.unique_objects[tok]
            self.assertEqual((u.velocity.x, u.velocity.y), (0.0, 0.0))
            self.assertAlmostEqual(u.box.length, 4.0 if tok == "A" else 1.8)
        self.assertNotIn("P", obs[0].tokens)                              # absent at t0 -> does not exist
        self.assertNotIn("P", obs[50].tokens)

    def test_t0v_freezes_pose_keeps_true_t0_velocity(self):
        obs, _ = obs_for("t0v")
        t0o, _ = obs_for("t0")
        for k in (0, 10, 50):
            self.assertEqual(geoms(obs, k), geoms(t0o, k), f"pose differs from t0 at step {k}")
        for tok, v in (("A", (3.0, 0.2)), ("D", (1.0, 0.0))):
            u = obs.unique_objects[tok]
            self.assertEqual((u.velocity.x, u.velocity.y), v)
        self.assertNotIn("P", obs[50].tokens)

    def test_cv_propagates_t0_velocity_heading_fixed(self):
        obs, sc = obs_for("cv")
        c0 = centre(obs, 0, "A")
        for k in (1, 10, 50):
            t = k * 0.1
            np.testing.assert_allclose(centre(obs, k, "A"), c0 + t * np.array([3.0, 0.2]), atol=1e-9)
        np.testing.assert_allclose(centre(obs, 50, "D") - centre(obs, 0, "D"), 5.0 * np.array([1.0, 0.0]), atol=1e-9)   # D persists past its true end
        u = obs.unique_objects["A"]
        self.assertEqual((u.velocity.x, u.velocity.y), (3.0, 0.2))
        self.assertAlmostEqual(u.box.center.heading, 0.1)
        # and the truth is NOT cv for A (the arm must differ from the true track)
        tru, _ = obs_for("true")
        self.assertGreater(np.linalg.norm(centre(tru, 50, "A") - centre(obs, 50, "A")), 5.0)
        self.assertNotIn("P", obs[50].tokens)

    def test_none_removes_all_dynamic_keeps_statics(self):
        obs, _ = obs_for("none")
        for k in (0, 25, 50):
            self.assertEqual(sorted(obs[k].tokens), sorted(t for t in obs[k].tokens if t in ("C", "N")))
            self.assertIn("C", obs[k].tokens)
        tru, _ = obs_for("true")
        self.assertEqual(geoms(obs, 50)["C"], geoms(tru, 50)["C"])        # statics byte-identical to NAVSIM's treatment
        self.assertEqual(geoms(obs, 50)["N"], geoms(tru, 50)["N"])

    def test_statics_unmodified_in_every_arm(self):
        tru, _ = obs_for("true")
        for mode in ("t0", "cv", "none"):
            obs, _ = obs_for(mode)
            for k in (0, 20, 50):
                for tok in ("C", "N"):
                    self.assertEqual(geoms(obs, k).get(tok), geoms(tru, k).get(tok), f"{mode} step {k} {tok}")

    def test_true_arm_keeps_late_and_vanishing_agents(self):
        obs, _ = obs_for("true")
        self.assertIn("P", obs[50].tokens)                                # P exists in the true list
        self.assertIn("D", obs[0].tokens)


class TestMechanics(unittest.TestCase):
    def test_outcome_rows(self):
        f = RS.outcome_row
        self.assertEqual(f(True, True, 0.25, 0.0)[0], "S0D-GO")
        self.assertEqual(f(True, True, 0.9, 0.9)[0], "S0D-GO")
        self.assertEqual(f(True, True, 0.2499, 0.50)[0], "S0D-MOTION")
        self.assertEqual(f(True, True, -1.0, 0.7)[0], "S0D-MOTION")
        self.assertEqual(f(True, True, 0.2499, 0.4999)[0], "S0D-NULL")
        self.assertEqual(f(True, True, -2.0, -0.5)[0], "S0D-NULL")
        self.assertEqual(f(False, True, 0.9, 0.9)[0], "INVALID")          # identity broken beats GO
        self.assertEqual(f(True, False, 0.9, 0.9)[0], "INVALID")          # monotone broken beats GO
        self.assertEqual(f(True, True, None, 0.9)[0], "NOT RUN")
        self.assertEqual(f(True, True, 0.2499, 0.9, 0.25)[0], "S0D-STOPFLAG")      # STOPFLAG before MOTION
        self.assertEqual(f(True, True, 0.2499, 0.1, 0.9)[0], "S0D-STOPFLAG")
        self.assertEqual(f(True, True, 0.25, 0.1, 0.9)[0], "S0D-GO")               # GO before STOPFLAG
        self.assertEqual(f(True, True, 0.2, 0.9, 0.2499)[0], "S0D-MOTION")
        self.assertEqual(f(True, True, 0.2, 0.4, 0.2499)[0], "S0D-NULL")
        self.assertEqual(f(False, True, 0.2, 0.4, 0.9)[0], "INVALID")

    def test_identity_check(self):
        nc_t = np.array([[1.0, 0.0, 0.5, 1.0]]); ttc_t = np.array([[1.0, 1.0, 0.0, 1.0]])
        ok = RS.identity_check(nc_t == 0, ttc_t == 0, nc_t, ttc_t)
        self.assertEqual(ok, 0)
        bad = RS.identity_check(np.array([[False, False, False, False]]), ttc_t == 0, nc_t, ttc_t)
        self.assertEqual(bad, 1)

    def test_monotone_none(self):
        t = np.array([[1, 1, 0, 0], [0, 0, 0, 0]], bool)
        n = np.array([[1, 0, 0, 0], [0, 0, 0, 0]], bool)
        self.assertEqual(RS.monotone_none(n, t), (0, 0))
        n2 = np.array([[1, 0, 0, 0], [1, 0, 0, 0]], bool)                 # created a collision where truth had none
        self.assertEqual(RS.monotone_none(n2, t), (1, 1))
        n3 = np.array([[0, 0, 1, 0], [0, 0, 0, 0]], bool)                 # same count, different proposal: subset violation only
        self.assertEqual(RS.monotone_none(n3, t), (0, 1))

    def test_gain_equals_oracle_for_truth_flags(self):
        import s0b_readout as RO
        rng = np.random.default_rng(0)
        S = 30
        rule = rng.normal(size=(S, 64)); pd = rng.random((S, 64)); nc = rng.choice([0.0, 0.5, 1.0], (S, 64)); ttc = rng.choice([0.0, 1.0], (S, 64))
        p6 = rule.argmax(1)
        O = RO.oracle_gains(rule, pd, nc, ttc, p6)
        g = RO.set_gains(rule, pd, p6, nc == 0, ttc == 0)
        self.assertTrue(np.array_equal(g["both"], O["both"]))


MUTATIONS = [
    ("cv ignores velocity", "s0d_hook.py", "vx * t_s", "0.0 * t_s"),
    ("t0 hook not applied", "s0d_hook.py", 'if self._mode == "true":\n            return self._true(iteration)', 'if self._mode in ("true", "t0"):\n            return self._true(iteration)'),
    ("none keeps the agents", "s0d_hook.py", 'if mode == "none":\n        return statics', 'if mode == "none":\n        return list(objs)'),
    ("t0 keeps the velocity", "s0d_hook.py", "a.center.heading, 0.0, 0.0)", "a.center.heading, a.velocity.x, a.velocity.y)"),
    ("identity arm time-shifted", "s0d_hook.py", 'if self._mode == "true":\n            return self._true(iteration)', 'if self._mode == "true":\n            return self._true(max(iteration - 10, 0))'),
    ("statics dropped", "s0d_hook.py", "statics = [o for o in objs if not is_dynamic(o)]", "statics = []"),
    ("t0 admits agents absent at t0", "s0d_hook.py", "t0_dyn = {o.track_token: o for o in tracks[0].tracked_objects.tracked_objects if is_dynamic(o)}",
     "t0_dyn = {o.track_token: o for o in tracks[-1].tracked_objects.tracked_objects if is_dynamic(o)}"),
    ("t0v drops velocity", "s0d_hook.py", "a.center.heading, a.velocity.x, a.velocity.y))", "a.center.heading, 0.0, 0.0))"),
    ("STOPFLAG disabled", "s0d_rsim.py", "if c_t0v is not None and c_t0v >= GO_T0:", "if False:"),
    ("GO threshold broken", "s0d_rsim.py", "if c_t0 >= GO_T0:", "if c_t0 > GO_T0 + 1.0:"),
    ("MOTION threshold broken", "s0d_rsim.py", "if c_cv >= MOTION_CV:", "if c_cv >= 0.0:"),
    ("monotone check inverted", "s0d_rsim.py", "nc_flag_none.sum(axis=1) > nc_flag_true.sum(axis=1)", "nc_flag_none.sum(axis=1) < nc_flag_true.sum(axis=1)"),
]


def run_mutations():
    tmp_root = os.environ.get("TMP") or tempfile.gettempdir()
    bad = 0
    for name, fname, old, new in MUTATIONS:
        d = tempfile.mkdtemp(prefix="s0dmut_", dir=tmp_root)
        try:
            for f in ("s0d_hook.py", "s0d_rsim.py", "test_s0d.py"):
                shutil.copy(HERE / f, Path(d) / f)
            src = (Path(d) / fname).read_text(encoding="utf-8")
            if old not in src:
                print(f"MUTATION NOT APPLICABLE: {name}")
                bad += 1
                continue
            (Path(d) / fname).write_text(src.replace(old, new, 1), encoding="utf-8", newline="\n")
            env = dict(os.environ, S0D_S0B_DIR=str(HERE.parent))
            r = subprocess.run([sys.executable, str(Path(d) / "test_s0d.py")], capture_output=True, text=True, encoding="utf-8", env=env, cwd=d)
            red = r.returncode != 0
            print(f"{'RED ' if red else 'GREEN (BAD)'} {name}")
            bad += 0 if red else 1
        finally:
            shutil.rmtree(d, ignore_errors=True)
    print("MUTATIONS", "ALL RED" if bad == 0 else f"{bad} NOT RED", f"({len(MUTATIONS)})")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    if "--mutations" in sys.argv:
        sys.exit(run_mutations())
    unittest.main(argv=[sys.argv[0]], verbosity=1)
