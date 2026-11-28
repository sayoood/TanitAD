"""Unit tests of dataeff_prep.py on SYNTHETIC joins and tracks (no real data)."""
import os
import sys
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import dataeff_prep as P  # noqa: E402

N = 4572
TMP = os.path.dirname(os.path.abspath(__file__)) + "/tmp"
os.makedirs(TMP, exist_ok=True)


def ids(n=N):
    return [f"clip-{i:05d}" for i in range(n)]


def synth_join(clips, T=40, seed=0):
    rng = np.random.default_rng(seed)
    C = len(clips)
    n = np.full(C, T, np.int64)
    off = np.concatenate([[0], np.cumsum(n)[:-1]])
    R = int(n.sum())
    J = {"clip": np.array(clips), "off": off, "n": n, "limit": rng.random((C, 4)).astype(np.float32), "limit_kmh": rng.integers(0, 100, C).astype(np.int16),
         "limit_ok": rng.random(C) > 0.5, "has": rng.random(R) > 0.2, "now_s": rng.random(R), "nav": rng.random((R, 5)).astype(np.float32), "nav_d": rng.random(R).astype(np.float32),
         "rc": rng.random((R, 3, 3)).astype(np.float32), "rc_ok": rng.random((R, 3)) > 0.5, "goal_y": rng.integers(0, 9, R).astype(np.uint32),
         "goal_w": rng.integers(0, 9, R).astype(np.uint32), "lat_cls": rng.integers(0, 4, R).astype(np.int16), "lat_allowed": rng.integers(0, 4, R).astype(np.uint8),
         "lon_cls": rng.integers(0, 4, R).astype(np.int16), "lon_allowed": rng.integers(0, 4, R).astype(np.uint8), "cons": rng.random((R, 26)).astype(np.float32),
         "lead_valid": rng.integers(0, 2, R).astype(np.int8), "lead_gap": rng.random(R).astype(np.float32), "n_clips_missing_labels": 0}
    return J


def track(kind, T=60):
    """native track [2T,4]: raw rows, the first two NaN as in the real files.  kind: straight (const speed), stop, turn."""
    t = np.arange(2 * T) * 0.1
    if kind == "straight":
        x, y, yaw = 8.0 * t, 0 * t, 0 * t
    elif kind == "stop":
        v = np.clip(8.0 - 2.0 * np.clip(t - 3.0, 0, None), 0, None)           # brakes to rest by t = 7 s, then at rest
        x, y, yaw = np.cumsum(v) * 0.1, 0 * t, 0 * t
    elif kind == "turn":
        yaw = np.clip(0.25 * (t - 2.0), 0, None)                              # 14 deg/s turn from t = 2 s
        x, y = np.cumsum(8.0 * np.cos(yaw)) * 0.1, np.cumsum(8.0 * np.sin(yaw)) * 0.1
    tr = np.stack([x, y, yaw, np.zeros_like(x)], 1)
    tr[:2] = np.nan
    return tr


class TestRanges(unittest.TestCase):
    def test_range_ends_of_the_card(self):
        self.assertEqual({k: P.range_end(N, v) for k, v in P.KEPT.items()}, {"R_half": 2856, "R_quarter": 3984, "R_eighth": 4548})

    def test_exclude_arithmetic_at_the_ends(self):
        order = P.hash_order(ids())
        for kept, b in ((2256, 2856), (1128, 3984), (564, 4548)):
            k = P.apply_exclude(order, 540, b)
            self.assertEqual(len(k), kept)
            self.assertEqual(k[:540], order[:540])                              # start inclusive: positions 0..539 stay
            self.assertEqual(k[540:], order[b:])                                 # end exclusive: position b stays
            self.assertNotIn(order[540], k)
            self.assertNotIn(order[b - 1], k)
            self.assertIn(order[b], k)
        k600 = P.apply_exclude(order, 540, 600)
        self.assertEqual(len(k600), N - 60)                                      # the full arm: 4,512
        self.assertEqual(P.apply_exclude(order, 540, N), order[:540])           # B = N keeps only the first 540

    def test_range_end_refuses_out_of_range(self):
        with self.assertRaises(ValueError):
            P.range_end(N, 4570)                                                 # more than the 4,512 eligible
        with self.assertRaises(ValueError):
            P.range_end(N, 10)

    def test_nesting_and_manifest_n_other_than_4572(self):
        order = P.hash_order(ids())
        a, b, c = (set(P.apply_exclude(order, 540, P.range_end(N, k))) for k in (564, 1128, 2256))
        self.assertTrue(a <= b <= c)
        self.assertEqual(P.range_end(4600, 1128), 540 + 4600 - 1128)             # note N1 bullet 3: the end follows from the kept count

    def test_sha12_order_not_manifest_order(self):
        cl = ids(50)
        order = P.hash_order(cl)
        self.assertEqual(order, sorted(cl, key=P.sha12))
        self.assertNotEqual(order, cl)


class TestSelect(unittest.TestCase):
    def setUp(self):
        self.order = P.hash_order(ids())
        self.ex = set(self.order[540:600])

    def test_floor_of_282_and_size(self):
        rng = np.random.default_rng(1)
        sc = {c: float(rng.random()) for c in self.order}
        S = P.select_S(self.order, sc, self.ex)
        self.assertEqual(len(S), 1128)
        self.assertEqual(len(set(S)), 1128)
        self.assertFalse(set(S) & self.ex)
        elig = [c for c in self.order if c not in self.ex]
        top = sorted(elig, key=lambda c: (-sc[c], self.order.index(c)))[:846]
        rest = [c for c in elig if c not in set(top)][:282]
        self.assertEqual(set(S), set(top) | set(rest))
        self.assertEqual(len(set(S) - set(top)), 282)
        pos = {c: i for i, c in enumerate(self.order)}
        self.assertEqual(S, sorted(S, key=lambda c: pos[c]))                    # output in hash order

    def test_ties_by_hash_order(self):
        sc = {c: 0.5 for c in self.order}                                       # every score tied: top 846 = the first 846 eligible in hash order
        S = P.select_S(self.order, sc, self.ex)
        elig = [c for c in self.order if c not in self.ex]
        self.assertEqual(S, elig[:846 + 282])
        sc2 = {c: 0.5 for c in self.order}
        top_c = elig[-1]
        sc2[top_c] = 0.9                                                         # one clearly higher clip at the end of the hash order enters the top set
        S2 = P.select_S(self.order, sc2, self.ex)
        self.assertIn(top_c, S2)
        self.assertEqual(S2, sorted(set(elig[:845 + 282]) | {top_c}, key=self.order.index))

    def test_excluded_clips_never_eligible_even_with_top_score(self):
        sc = {c: 0.1 for c in self.order}
        for c in list(self.ex)[:10]:
            sc[c] = 1.0
        self.assertFalse(set(P.select_S(self.order, sc, self.ex)) & self.ex)

    def test_unlabelled_clip_scores_zero(self):
        st = {"a": (30, 0, 0), "b": (30, 30, 15), "c": (30, 20, 20)}
        self.assertEqual(P.scores_of(st), {"a": 0.0, "b": 0.5, "c": 1.0})


class TestRule(unittest.TestCase):
    def test_modes_of_synthetic_tracks(self):
        T = 60
        res = {}
        for kind in ("straight", "stop", "turn"):
            t, g, ok = P.window_futures(track(kind, T), T)
            self.assertTrue(ok.all())
            self.assertEqual(len(t), T - 33)
            res[kind] = P.selection_critical(g)
        self.assertFalse(res["straight"].any())                                  # KEEP, STRAIGHT: bulk
        self.assertTrue(res["stop"][:5].all())                                   # braking to rest within the window: STOP
        self.assertTrue(res["turn"][:5].all())                                   # 14 deg/s over 6 s: LEFT

    def test_window_count_matches_train_windows_rule(self):
        T = 101
        t, g, ok = P.window_futures(track("straight", T), T)
        self.assertEqual(len(t), T - 33)
        self.assertEqual((int(t[0]), int(t[-1])), (3, T - 31))

    def test_track_length_mismatch_refused(self):
        with self.assertRaises(ValueError):
            P.window_futures(track("straight", 60), 61)

    def test_nan_future_is_unlabelled(self):
        tr = track("straight", 60)
        tr[2 * 50:2 * 52] = np.nan
        t, g, ok = P.window_futures(tr, 60)
        self.assertFalse(ok.all())
        self.assertTrue(ok[0])


class TestRestriction(unittest.TestCase):
    def test_pure_filter_and_window_count(self):
        cl = ids(30)
        J = synth_join(cl, T=40)
        keep = [cl[i] for i in (3, 4, 9, 20)]
        R = P.restrict_join(J, keep)
        self.assertEqual(R["clip"].tolist(), keep)
        self.assertEqual(P.windows_of(R), P.windows_of(J, keep))
        self.assertEqual(P.windows_of(R), 4 * (40 - 33))
        self.assertEqual(R["off"].tolist(), [0, 40, 80, 120])
        for k, i in enumerate((3, 4, 9, 20)):
            a = int(J["off"][i])
            for key in P.PER_ROW:
                self.assertTrue(np.array_equal(J[key][a:a + 40], R[key][40 * k:40 * k + 40]))
            self.assertTrue(np.array_equal(J["limit"][i], R["limit"][k]))
        self.assertEqual(set(R), set(J))

    def test_missing_clip_refused(self):
        J = synth_join(ids(5))
        with self.assertRaises(ValueError):
            P.restrict_join(J, ["nope"])

    def test_d5_on_synthetic(self):
        cl = ids(40)
        J = synth_join(cl, T=45)
        r = P.control_d5(J, cl[5:25])
        self.assertTrue(r["PASS"])
        self.assertEqual(r["windows_in_restricted_join"], 20 * 12)

    def test_d5_mutation_corrupted_row_goes_red(self):
        cl = ids(40)
        J = synth_join(cl, T=45)
        orig = P.restrict_join

        def bad(J_, clips):
            R = orig(J_, clips)
            R["cons"] = R["cons"].copy()
            R["cons"][3, 0] += 1.0
            return R
        P.restrict_join = bad
        try:
            self.assertFalse(P.control_d5(J, cl[5:25])["PASS"])
        finally:
            P.restrict_join = orig

    def test_roundtrip_file(self):
        import tempfile
        J = synth_join(ids(8), T=40)
        R = P.restrict_join(J, ids(8)[:3])
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            P.save_join(R, d + "/j.npz")
            R2 = P.load_join(d + "/j.npz")
        self.assertEqual(R2["clip"].tolist(), R["clip"].tolist())
        self.assertEqual(R2["n_clips_missing_labels"], R["n_clips_missing_labels"])


class TestControls(unittest.TestCase):
    def lists(self):
        order = P.hash_order(ids())
        ex = set(order[540:600])
        L = {k: P.apply_exclude(order, 540, P.range_end(N, v)) for k, v in P.KEPT.items()}
        rng = np.random.default_rng(3)
        L["S_quarter"] = P.select_S(order, {c: float(rng.random()) for c in order}, ex)
        return order, L

    def test_d1(self):
        order, L = self.lists()
        r = P.control_d1(L, order, None)
        self.assertTrue(r["PASS"])
        L2 = dict(L)
        L2["R_eighth"] = L["R_eighth"][:-1] + [order[560]]                       # a clip outside R_quarter: nesting broken
        self.assertFalse(P.control_d1(L2, order, None)["PASS"])

    def test_d2_clean_and_mutation(self):
        order, L = self.lists()
        outside = [c for c in order if all(c not in set(v) for v in L.values())]
        ev = [f"eval-{i:03d}" for i in range(140)] + [outside[0]]                # 141 eval ids, none in a list (an unused train clip stands in)
        self.assertTrue(P.control_d2(L, ev)["PASS"])
        L_bad = {k: list(v) for k, v in L.items()}
        L_bad["S_quarter"][5] = ev[7]                                            # MUTATION: one eval clip let into a list
        r = P.control_d2(L_bad, ev)
        self.assertFalse(r["PASS"])
        self.assertEqual(r["overlap_by_id"]["S_quarter"], 1)
        self.assertEqual(r["overlap_by_id"]["R_quarter"], 0)

    def test_d2_needs_141(self):
        order, L = self.lists()
        self.assertFalse(P.control_d2(L, ["e1", "e2"])["PASS"])


class TestStatic(unittest.TestCase):
    def test_list_sha_and_train_sha(self):
        import tempfile
        cl = P.hash_order(ids(20))
        with tempfile.TemporaryDirectory(dir=TMP) as d:
            h = P.write_list(d + "/l.txt", cl)
            self.assertEqual(h, P.sha256_file(d + "/l.txt"))
        import hashlib
        self.assertEqual(P.train_sha(cl[::-1]), hashlib.sha256("\n".join(cl).encode()).hexdigest())


if __name__ == "__main__":
    unittest.main(verbosity=1)
