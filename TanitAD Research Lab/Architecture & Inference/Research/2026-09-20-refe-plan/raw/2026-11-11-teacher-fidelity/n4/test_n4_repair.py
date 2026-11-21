"""Tests for the N4-DAC repair (real navtrain token 634f20124fa3558c, a failing RJ5 form-R label; token 4... passing control chosen at run time).
Run:  ./run.sh -m unittest test_n4_repair -v      (CPU, ~1 min)
 T1 zero shift is bit-identical            T2 repair D is certified and deterministic (two runs, same plan bit for bit)
 T3 REGRESSION ARM: the sign-flipped informed shift (pushes outward) never certifies
 T4 a passing label is returned unchanged by D (iteration 0) and the certificate is not weaker than DAC = 1
 T5 the certificate refuses a candidate that is inside the area but has a worse NC / TTC / C than the base"""
import sys
import unittest
import numpy as np
sys.path.insert(0, ".")
import n4_geom as G
import n4_common as N
import n4_repair as RP

TOK = "634f20124fa3558c"
Rl, S = N.load_rolls(), N.load_scores()
_cx = {}


def ctx(tok):
    if tok not in _cx:
        for _, _, c in G.contexts({S["RJ5"][tok]["log"]: [tok]}):
            _cx[tok] = c
    return _cx[tok]


class T(unittest.TestCase):
    def setUp(self):
        self.cx = ctx(TOK)
        self.t20 = np.asarray(Rl["RJ5"][TOK]["traj20"], float)
        self.lab = self.cx.labels(self.t20)["R"]
        self.base = self.cx.score([self.lab])[0][0]

    def test_t1_zero_shift_identity(self):
        p = RP.apply_shift(self.lab, [0.0, 0.0], 1.0)
        self.assertTrue(np.array_equal(p, self.lab))
        self.assertEqual(self.cx.score([p])[0][0], self.base)

    def test_t2_repair_d_certified_and_deterministic(self):
        self.assertEqual(self.base["dac"], 0.0)
        r1 = RP.repair_target(self.cx, self.t20, self.base)
        r2 = RP.repair_target(self.cx, self.t20, self.base)
        self.assertIsNotNone(r1)
        self.assertEqual(r1["m"]["dac"], 1.0)
        self.assertTrue(RP.certified(r1["m"], self.base))
        self.assertTrue(np.array_equal(r1["plan"], r2["plan"]))

    def test_t3_sign_flipped_shift_never_certifies(self):
        self.assertIsNone(RP.repair_informed(self.cx, self.lab, self.base, sign=-1.0))
        self.assertIsNotNone(RP.repair_informed(self.cx, self.lab, self.base, sign=+1.0))      # the correct sign does (control)

    def test_t4_passing_label_unchanged(self):
        ok = next(t for t in sorted(S["RJ5"]) if S["RJ5"][t]["arms"]["R"]["tracked"]["dac"] == 1.0 and S["RJ5"][t]["log"] == S["RJ5"][TOK]["log"])
        cx = ctx(ok)
        t20 = np.asarray(Rl["RJ5"][ok]["traj20"], float)
        lab = cx.labels(t20)["R"]
        base = cx.score([lab])[0][0]
        r = RP.repair_target(cx, t20, base)
        self.assertEqual(r["param"]["iters"], 0)
        self.assertTrue(np.array_equal(r["plan"], lab))

    def test_t5_certificate_refuses_worse_neighbours(self):
        b = dict(self.base, dac=1.0)
        c = dict(b)
        for k in ("nc", "ttc", "c", "ddc"):
            w = dict(c, **{k: b[k] - 0.5})
            self.assertFalse(RP.certified(w, b), k)
        self.assertFalse(RP.certified(dict(b, dac=0.0), b))
        self.assertTrue(RP.certified(dict(b), b))


if __name__ == "__main__":
    unittest.main()
