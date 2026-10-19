"""Anchored patch of test_ep_admission_sim.py: appends the --rep-offset tests.  python patch_test.py <in> <out>"""
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src, encoding="utf-8", newline="").read()
assert "\r\n" not in s
anchor = '\n\nif __name__ == "__main__":\n    unittest.main()'
assert s.count(anchor) == 1
new = '''

OFF = 1_000_000


class TestRepOffset(unittest.TestCase):
    """EP-ADMISSION-R2-CONFIRM: `--rep-offset` gives a FRESH set of replicates; offset 0 is the banked draws exactly."""

    # golden values taken from the PRE-EDIT module (blob da9e9547...), so offset 0 is pinned to the banked EP-ADMISSION-R1 draws and not to this file's own code
    GOLD_OUTER0 = [85, 115, 25, 112, 70, 3, 1, 51, 108, 44, 49, 16]
    GOLD_OUTER5 = [11, 31, 9, 29, 75, 84, 57, 12, 53, 53, 18, 40]
    GOLD_WILD0 = [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0]
    GOLD_OUTER0_OFF = [7, 47, 67, 75, 92, 65, 28, 29, 87, 107, 14, 84]

    @classmethod
    def setUpClass(cls):
        comp, _ = synth(over=0.05)
        cls.condA = cond(comp["vec"], "cal", name="A")
        cls.condB = cond(comp["vec"], "cal", name="B", gen="B", a2=False)

    @staticmethod
    def same(a, b):
        return all(np.array_equal(a[k], b[k], equal_nan=True) for k in ("dm", "a2", "mp", "me"))

    def test_offset_zero_is_the_banked_draw_and_a_million_is_a_different_one(self):
        self.assertEqual(S.outer_idx(0)[:12].tolist(), self.GOLD_OUTER0)
        self.assertEqual(S.outer_idx(5)[:12].tolist(), self.GOLD_OUTER5)
        self.assertEqual(S.wild_signs(0)[:12].tolist(), self.GOLD_WILD0)
        for r in (0, 1, 5, 7, 999):
            self.assertTrue(np.array_equal(S.outer_idx(r), np.random.default_rng([S.OUTER_SALT, r]).integers(0, S.N_LOGS, S.N_LOGS)))               # the pre-edit formula
            self.assertTrue(np.array_equal(S.wild_signs(r), np.random.default_rng([S.WILD_SALT, r]).choice(np.array([-1.0, 1.0]), S.N_LOGS)))
            self.assertTrue(np.array_equal(S.outer_idx(r, 0), S.outer_idx(r)))
            self.assertTrue(np.array_equal(S.wild_signs(r, 0), S.wild_signs(r)))
            self.assertFalse(np.array_equal(S.outer_idx(r, OFF), S.outer_idx(r)))                                                                    # a fresh draw
            self.assertFalse(np.array_equal(S.wild_signs(r, OFF), S.wild_signs(r)))
            self.assertTrue(np.array_equal(S.outer_idx(r, OFF), S.outer_idx(r + OFF)))                                                              # and exactly the draw of index rep + offset
            self.assertTrue(np.array_equal(S.wild_signs(r, OFF), S.wild_signs(r + OFF)))
        self.assertEqual(S.outer_idx(0, OFF)[:12].tolist(), self.GOLD_OUTER0_OFF)

    def test_no_replicate_of_the_fresh_set_reuses_a_banked_draw(self):
        banked = {S.outer_idx(r).tobytes() for r in range(1000)}
        fresh = {S.outer_idx(r, OFF).tobytes() for r in range(1000)}
        self.assertEqual(len(banked), 1000)
        self.assertEqual(len(fresh), 1000)
        self.assertEqual(len(banked & fresh), 0)
        bw, fw = {S.wild_signs(r).tobytes() for r in range(1000)}, {S.wild_signs(r, OFF).tobytes() for r in range(1000)}
        self.assertEqual(len(bw & fw), 0)

    def test_replicate_default_is_offset_zero_and_offset_changes_it_in_both_generators(self):
        for c in (self.condA, self.condB):
            base = S.replicate(c, 3, DRAWS)
            self.assertTrue(self.same(base, S.replicate(c, 3, DRAWS, 0)))
            fresh = S.replicate(c, 3, DRAWS, OFF)
            self.assertFalse(np.array_equal(base["dm"], fresh["dm"]))
            self.assertTrue(self.same(fresh, S.replicate(c, 3 + OFF, DRAWS)))                          # offset is a pure index shift

    def test_the_pool_worker_applies_the_offset_it_was_initialised_with(self):
        old = (S._CONDS, S._DRAWS, S._REP_OFFSET)
        try:
            for off in (0, OFF):
                S._init({"A": self.condA}, off)
                _, a0, rec = S._chunk(("A", 2, 4))
                self.assertEqual(a0, 2)
                for i, r in enumerate((2, 3)):
                    exp = S.replicate(self.condA, r, S._DRAWS, off)
                    self.assertTrue(all(np.array_equal(rec[k][i], exp[k], equal_nan=True) for k in ("dm", "a2", "mp", "me")))
            S._init({"A": self.condA}, OFF)
            _, _, fresh = S._chunk(("A", 2, 4))
            S._init({"A": self.condA}, 0)
            _, _, banked = S._chunk(("A", 2, 4))
            self.assertFalse(np.array_equal(fresh["dm"], banked["dm"]))
        finally:
            S._CONDS, S._DRAWS, S._REP_OFFSET = old

    def test_cli_default_is_zero_and_the_flag_is_an_int(self):
        ap = S.build_parser()
        self.assertEqual(ap.parse_args(["run", "--workdir", "D:/x"]).rep_offset, 0)
        self.assertEqual(ap.parse_args(["run", "--workdir", "D:/x", "--rep-offset", "1000000"]).rep_offset, OFF)
        self.assertEqual(ap.parse_args(["repro", "--workdir", "D:/x"]).rep_offset, 0)
        self.assertEqual(ap.parse_args(["repro", "--workdir", "D:/x", "--rep-offset", "1000000", "--against", "f.npz"]).rep_offset, OFF)
        self.assertIsNone(ap.parse_args(["repro", "--workdir", "D:/x"]).against)
        with self.assertRaises(SystemExit):
            ap.parse_args(["run", "--workdir", "D:/x", "--rep-offset", "abc"])

    # ---- RED arms: the mutants below must be distinguishable from the correct code by the assertions above ----
    def test_red_a_generator_that_ignores_the_offset_is_caught(self):
        orig = S.outer_idx
        base, fresh = S.replicate(self.condA, 3, DRAWS), S.replicate(self.condA, 3, DRAWS, OFF)
        with mock.patch.object(S, "outer_idx", lambda rep, offset=0: orig(rep)):                      # mutant: offset dropped
            dead = S.replicate(self.condA, 3, DRAWS, OFF)
        self.assertTrue(self.same(dead, base))                                                         # the mutant reproduces offset 0 ...
        self.assertFalse(self.same(dead, fresh))                                                       # ... and so FAILS `fresh == replicate(rep + OFF)`

    def test_red_a_worker_that_drops_the_offset_is_caught(self):
        orig = S.replicate
        old = (S._CONDS, S._DRAWS, S._REP_OFFSET)
        try:
            S._init({"A": self.condA}, OFF)
            _, _, good = S._chunk(("A", 0, 2))
            with mock.patch.object(S, "replicate", lambda c, r, d, offset=0: orig(c, r, d, 0)):       # mutant: _chunk loses the offset
                _, _, dead = S._chunk(("A", 0, 2))
            self.assertFalse(np.array_equal(good["dm"], dead["dm"]))
        finally:
            S._CONDS, S._DRAWS, S._REP_OFFSET = old

    def test_red_wild_signs_that_ignore_the_offset_are_caught(self):
        orig = S.wild_signs
        base, fresh = S.replicate(self.condB, 3, DRAWS), S.replicate(self.condB, 3, DRAWS, OFF)
        with mock.patch.object(S, "wild_signs", lambda rep, offset=0: orig(rep)):
            dead = S.replicate(self.condB, 3, DRAWS, OFF)
        self.assertTrue(self.same(dead, base))
        self.assertFalse(self.same(dead, fresh))
'''
s = s.replace(anchor, new + anchor)
open(dst, "w", encoding="utf-8", newline="").write(s)
print("patched ->", dst)
