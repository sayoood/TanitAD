"""TF-N6 chain tests: CPU, synthetic data (plus the real navtrain fixtures in one test), no navtest.  Run (navsim venv, N4_RUN = the run tree, CUDA off):
    N4_RUN=C:/Users/Admin/n6nt/run  python -m pytest -q test_n6_chain.py            (the `real` test needs the nuPlan DB and takes about a minute)

 SELECT  the three selectors read back as the card and have the card's behavioural signature; EVERY clause moves the signature AND is refused: tolerance off (strict TTC/C),
         NC guard dropped, DDC guard dropped, PDMS-improvement dropped, human-first ordering, DAC requirement inverted, tie rule (latest); the card semantics on hand-built candidate sets
 CAPTURE the recording proxy leaves the banked repairs' results bit-identical, records every iterate, the tolerance rescues a candidate the full certificate refuses (TTC/C worse) through
         the REAL repair loops on a fake scorer, an NC-worse / DDC-worse / PDMS-not-better candidate is not taken; the human plan has the harness human arm's format; the real fixtures
 REPAIR  REPAIR-D (A1 settings) and L1 (TF-N5 frozen) objects read back; a different cap / margin / sign / certificate is refused
 UNCH    file-level C-UNCH: 2e-6 edit of an unchanged plan, applied-to-all, flagged-but-unchanged, seam changed without plan, wrong residual count, N6-X changed each go RED
 MERGE   the build merge on synthetic rows: green; each of {strict, NC guard dropped, DDC guard dropped, PDMS clause dropped, human-first, DAC inverted (H and X), tie-latest} recorded by the
         derive goes RED (C-SELECT / C-CERT-B6 / C-ORDER / C-X0); m_recheck differs, wrong DAC map, cache-ego, non-deterministic capture, missing row, changed residual set, replay
         violation, bad reach json, seam function mismatch each go RED; every npz is read ONCE (the TF-N5 OOM lesson, mutation = the per-token access goes RED)
 READ    n6_nt_analyze: bar 1 (paper level point estimates, the boundary), bar 2 (NC/DDC mean and per-token), the declared TTC/C tolerance, bar 3, naming with the teacher / human split, K = 49
         in the Bonferroni key, the control exit BEFORE any N6-H aggregate (certificate -> C-CERT; 2e-6 -> C-ID-C; dropped row -> C-COV; N6-X changed -> C-REG; pinned file edited and an
         old families_run.py -> C-PIN; a wrong draw scheme -> C-EST)
 CHAIN   guards refuse: card sha, ledger entry / n / higher n / missing TF-N5-A1 entry, pins sha, an edited pinned file, an unpinned file, procs, a failing reach, an OLD families_run.py,
         low disk; --dry-run prints every stage's command and no mutation flag; the families JSON check refuses a JSON without tokens"""
import contextlib
import csv
import gzip
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("N4_RUN", os.path.dirname(HERE))
sys.path.insert(0, HERE)
import n6_nt_analyze as AN      # noqa: E402
import n6_nt_build as BD        # noqa: E402
import n6_nt_chain as CH        # noqa: E402

RUN = os.environ["N4_RUN"]
A1 = os.environ.get("N6_A1_ANALYZE", os.path.join(RUN, "n2", "tf_n1_analyze_a1.py"))
TMP = os.environ.get("N6_TMP") or None
COLS = ["", "token", "valid", "no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound", "comfort",
        "driving_direction_compliance", "score"]
OK5 = {"dac": 1.0, "nc": 1.0, "ddc": 1.0, "ttc": 1.0, "c": 1.0, "ep": 0.9, "pdms": 0.9}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


# ------------------------------------------------------------------------------------------------------------- SELECT
class TestSelect(unittest.TestCase):
    def test_card_selectors_pass_and_read_back(self):
        for arm in BD.ARMS:
            r = BD.check_selector(arm, BD.make_selector(arm))
            self.assertTrue(r["ok"], r["bad"])
        self.assertEqual(BD.CARD6["N6X"]["dac"], 0.0)
        self.assertEqual(BD.CARD6["N6H"]["dac"], 1.0)
        self.assertEqual(BD.CARD6["N6H"]["order"], ("teacher", "human"))
        self.assertEqual(BD.CARD6["N6T"]["order"], ("teacher",))
        for arm in BD.ARMS:                                                   # the card: TTC and C free; NC, DDC guards on; builder PDMS strictly better
            s = BD.CARD6[arm]
            self.assertFalse(s["ttc"] or s["c"])
            self.assertTrue(s["nc"] and s["ddc"] and s["pdms_gt"])

    def test_every_clause_and_the_ordering_reaches_the_selection(self):
        """MUTATE, do not inspect: each mutation (tolerance off, NC guard dropped, DDC guard dropped, PDMS clause dropped, human first, DAC inverted, tie latest) changes the behavioural
        signature of the H selector AND is refused by check_selector; every mutation moves at least one arm's signature"""
        for mut in BD.SEL_MUTATIONS:
            moved = False
            for arm in BD.ARMS:
                fn = BD.make_selector(arm, mut)
                if BD.selector_signature(fn) != BD.selector_signature(BD.make_selector(arm)):
                    moved = True
                    r = BD.check_selector(arm, fn)
                    self.assertFalse(r["ok"], (mut, arm))
            self.assertTrue(moved, f"{mut}: no selector's behaviour moved")
            r = BD.check_selector("N6H", BD.make_selector("N6H", mut))
            self.assertFalse(r["ok"], mut)
            self.assertNotEqual(BD.selector_signature(BD.make_selector("N6H", mut)), BD.selector_signature(BD.make_selector("N6H")), mut)

    def test_wrong_function_is_refused(self):
        import functools
        fn = functools.partial(lambda *a, **k: None, spec=dict(BD.CARD6["N6H"]))
        self.assertFalse(BD.check_selector("N6H", fn)["ok"])

    def test_card_semantics(self):
        H, T, X = (BD.make_selector(a) for a in ("N6H", "N6T", "N6X"))
        S = BD.SCENARIOS
        pick = lambda fn, nm: (lambda r: None if r is None else (r["src"], r["k"]))(fn(S[nm][1], S[nm][0]))      # noqa: E731
        self.assertEqual(pick(H, "max_pdms"), ("L1", 2))                      # max builder PDMS among passing teacher candidates
        self.assertEqual(pick(H, "tie_earliest"), ("D", 2))                   # tie -> earliest in generation order (REPAIR-D before L1)
        self.assertEqual(pick(H, "tie_d_before_l1"), ("L1", 1))               # generation order is the tie-break: the list order given
        self.assertEqual(pick(H, "ttc_c_free"), ("L1", 1))                    # TTC and C worse: still accepted
        self.assertIsNone(pick(H, "nc_worse"))                                # NC guard
        self.assertEqual(pick(H, "nc_equal_below_one"), ("L1", 1))            # NC equal to N5-A's passes
        self.assertIsNone(pick(H, "ddc_worse"))                               # DDC guard
        self.assertIsNone(pick(H, "pdms_not_improved"))                       # builder PDMS must improve
        self.assertIsNone(pick(H, "dac0_candidate"))                          # DAC = 1 required
        self.assertEqual(pick(H, "human_only"), ("H", 0))                     # human only where no teacher passes
        self.assertEqual(pick(H, "teacher_and_human"), ("L1", 1))             # teacher first, even though the human PDMS is higher
        self.assertIsNone(pick(H, "human_fails_nc"))                          # the human candidate must pass the SAME certificate
        self.assertIsNone(pick(H, "nothing"))
        self.assertIsNone(pick(T, "human_only"))                              # N6-T: teacher only
        self.assertEqual(pick(T, "teacher_and_human"), ("L1", 1))
        self.assertIsNone(pick(X, "max_pdms"))                                # N6-X: DAC = 0 required: a DAC-1 candidate is not taken
        self.assertEqual(pick(X, "dac0_candidate"), ("L1", 1))                # (the synthetic DAC-0 candidate with PDMS > ref; real DAC-0 plans score PDMS 0, test below)
        # in the builder's real scoring a DAC = 0 plan has PDMS 0, which cannot beat the DAC-0 reference's 0: N6-X changes nothing
        zero = BD._cand("L1", 1, "teacher", dac=0.0, pdms=0.0)
        self.assertIsNone(X([zero], BD.REF0))

    def test_ties_are_resolved_by_order_not_by_hash_or_dict_order(self):
        c = [BD._cand("L1", 1, "teacher", pdms=0.5), BD._cand("D", 4, "teacher", pdms=0.5), BD._cand("L1", 2, "teacher", pdms=0.5)]
        r = BD.make_selector("N6H")(c, BD.REF0)
        self.assertEqual((r["src"], r["k"]), ("L1", 1))
        r = BD.make_selector("N6H", "tie_latest")(c, BD.REF0)
        self.assertEqual((r["src"], r["k"]), ("L1", 2))


# ------------------------------------------------------------------------------------------------------------ REPAIR
class TestRepair(unittest.TestCase):
    def test_repair_objects_read_back(self):
        fd, fl, _ = BD.make_objects()
        self.assertTrue(BD.B4.check_arm(BD.D_ARM, fd)["ok"])
        self.assertTrue(BD.B5.check_arm(BD.L1_ARM, fl)["ok"])
        rd = BD.B4.readback(fd)
        self.assertEqual((rd["margin"], rd["iters"], rd["sign"], rd["cert"]), (0.05, 5, 1.0, "full"))

    def test_different_cap_margin_sign_cert_are_refused(self):
        import functools
        for kw in ({"iters": 4}, {"margin": 0.06}, {"sign": -1.0}, {"cert": BD.RP.certified_dac}):
            fn = functools.partial(BD.RP.repair_target, **dict({"margin": 0.05, "iters": 5, "sign": 1.0, "cert": BD.RP.certified}, **kw))
            self.assertFalse(BD.B4.check_arm(BD.D_ARM, fn)["ok"], kw)
        self.assertFalse(BD.B5.check_arm(BD.L1_ARM, BD.B5.make_repair(dict(BD.B5.CARD["N5A"], iters=4)))["ok"])
        self.assertFalse(BD.B5.check_arm(BD.L1_ARM, BD.B5.make_repair(dict(BD.B5.CARD["N5A"], sign=-1.0)))["ok"])
        self.assertFalse(BD.B5.check_arm(BD.L1_ARM, BD.B5.make_repair(BD.B5.CARD["N5A"], "cert_off"))["ok"])

    def test_build_cli_refuses_mutation_without_smoke_flag(self):
        with self.assertRaises(SystemExit) as c:
            BD.cmd_derive(SimpleNamespace(mutate="strict", smoke_allow_mutation=False))
        self.assertIn("REFUSING", str(c.exception))

    def test_chain_argv_names_no_mutation(self):
        a = SimpleNamespace(run_dir="R")
        argv = CH.build_argv(a, "R", 0, 2)
        self.assertFalse(any("mutate" in x or "smoke" in x for x in argv))
        self.assertIn("--n5a-plans", argv)
        self.assertEqual(argv[argv.index("--route") + 1], "cache")


# ----------------------------------------------------------------------------------------------------------- CAPTURE
def _fx():
    return BD.B5.fakes().T


def var_cx(T, corridor_args, m1=None):
    """FakeCx whose DAC-1 plans carry the metrics m1 (TTC / C / NC / DDC / PDMS variants) and whose DAC-0 plans score PDMS 0 (as the real builder does)"""
    m1 = dict(m1 or {})

    class Cx(T.FakeCx):
        def score(self, plans):
            res, sims = super().score(plans)
            for r in res:
                if r["dac"] == 1.0:
                    r.update(m1)
                else:
                    r.update(pdms=0.0, ep=0.0)
            return res, sims
    return Cx(T.corridor(*corridor_args))


BASE_FAIL = {"dac": 0.0, "nc": 1.0, "ddc": 1.0, "ttc": 1.0, "c": 1.0, "ep": 0.0, "pdms": 0.0}


class TestCapture(unittest.TestCase):
    def setUp(self):
        self.T = _fx()
        self.t20 = self.T.teacher(lambda t: np.where(t < 2.0, 0.30, -0.16))
        self.fd, self.fl, self.sels = BD.make_objects()

    def cap(self, m1):
        cx = var_cx(self.T, (0.50, 0.34), m1)
        return cx, BD.capture(cx, self.t20, dict(BASE_FAIL), self.fd, self.fl)

    def test_proxy_leaves_the_banked_repairs_bit_identical_and_records_every_iterate(self):
        cx = var_cx(self.T, (0.50, 0.34), {"ttc": 1.0, "c": 1.0, "ep": 0.9, "pdms": 0.9})
        direct = BD.B5.make_repair(BD.B5.CARD["N5A"])(cx, self.t20, dict(BASE_FAIL), info={})
        cands, repl = BD.capture(var_cx(self.T, (0.50, 0.34), {"ep": 0.9, "pdms": 0.9}), self.t20, dict(BASE_FAIL), self.fd, self.fl)
        self.assertIsNotNone(direct)
        self.assertTrue(repl["L1"]["returns_plan"])
        self.assertTrue(any(np.array_equal(c["plan"], direct["plan"]) for c in cands))                # the L1 plan the unproxied run returns is among the candidates
        self.assertEqual(cands[0]["src"], "D")
        self.assertEqual(cands[0]["k"], 0)
        self.assertTrue(np.array_equal(cands[0]["plan"], np.asarray(self.t20)))                         # iterate 0 = the base label (identity labeller)
        plans = [c["plan"].tobytes() for c in cands]
        self.assertEqual(len(plans), len(set(plans)))                                                   # deduplicated by plan bytes, the first kept
        self.assertGreaterEqual(len(cands), 3)

    def test_capture_is_deterministic(self):
        _, (c1, _) = self.cap({"ep": 0.9, "pdms": 0.9})
        _, (c2, _) = self.cap({"ep": 0.9, "pdms": 0.9})
        self.assertTrue(BD.same_cands(c1, c2))
        c2[-1]["plan"] = c2[-1]["plan"] + 1e-9
        self.assertFalse(BD.same_cands(c1, c2))

    def test_tolerance_rescues_a_candidate_the_full_certificate_refuses(self):
        """DAC-1 candidates whose TTC and C are worse than the base: the frozen full certificate refuses them in BOTH repairs (so TF-N5's N5-A left the token at DAC 0); the card selector takes one;
        the strict mutation (TTC/C required) takes none"""
        base = dict(BASE_FAIL)
        m1 = {"ttc": 0.0, "c": 0.0, "ep": 0.9, "pdms": 0.5}
        cx, (cands, repl) = self.cap(m1)
        self.assertFalse(repl["D"]["returns_plan"] or repl["L1"]["returns_plan"])                       # the banked certificate: nothing
        self.assertTrue(any(c["m"]["dac"] == 1.0 for c in cands))
        got = self.sels["N6H"](cands, base)
        self.assertIsNotNone(got)
        self.assertEqual(got["m"]["dac"], 1.0)
        self.assertEqual(got["cls"], "teacher")
        self.assertIsNone(BD.make_selector("N6H", "strict")(cands, base))
        self.assertIsNone(self.sels["N6X"](cands, base))

    def test_nc_ddc_and_pdms_guards_in_the_real_loops(self):
        base = dict(BASE_FAIL)
        for m1, mut in (({"nc": 0.0, "pdms": 0.5}, "nc_drop"), ({"ddc": 0.5, "pdms": 0.5}, "ddc_drop"), ({"pdms": 0.0}, "pdms_drop")):
            cx, (cands, _) = self.cap(dict({"ep": 0.9, "pdms": 0.5}, **m1))
            self.assertIsNone(self.sels["N6H"](cands, base), m1)
            self.assertIsNotNone(BD.make_selector("N6H", mut)(cands, base), mut)

    def test_human_after_teacher_and_only_then(self):
        base = dict(BASE_FAIL)
        cx, (cands, _) = self.cap({"ttc": 0.0, "ep": 0.9, "pdms": 0.5})
        hum = BD._cand("H", 0, "human", pdms=0.99)
        hum["plan"] = np.zeros((20, 3))
        got = self.sels["N6H"](cands + [hum], base)
        self.assertEqual(got["cls"], "teacher")                                                        # a teacher candidate passes: the human one is not taken
        self.assertEqual(BD.make_selector("N6H", "human_first")(cands + [hum], base)["cls"], "human")
        got2 = self.sels["N6H"]([c for c in cands if c["m"]["dac"] == 0.0] + [hum], base)
        self.assertEqual(got2["cls"], "human")                                                         # no teacher passes: the human candidate
        self.assertIsNone(self.sels["N6T"]([c for c in cands if c["m"]["dac"] == 0.0] + [hum], base))

    def test_human_plan_has_the_harness_human_arm_format(self):
        import transp_score as TS

        class P:
            def __init__(self, x, y, h):
                self.x, self.y, self.heading = x, y, h

        class St:
            def __init__(self, i):
                self.rear_axle = P(1.0 * i, 0.1 * i, 0.01 * i)

        class Sc:
            def get_ego_future_trajectory(self, a, b, n):
                assert (a, b, n) == (0, 4.0, 40)
                return [St(i + 1) for i in range(n)]

        class Ego:
            rear_axle = P(0.0, 0.0, 0.0)

        cx = SimpleNamespace(sc=Sc(), ego=Ego())
        p, why = BD.human_plan20(cx, None, "l", "t")
        self.assertEqual(why, "ok")
        hum = TS.human_rel(Sc(), Ego())
        self.assertEqual(p.shape, (20, 3))
        self.assertTrue(np.array_equal(p, hum[1::2][:20]))                                               # exactly the harness human arm's slice
        self.assertAlmostEqual(float(p[0, 0]), 2.0)                                                      # 0.2 s
        self.assertAlmostEqual(float(p[-1, 0]), 40.0)                                                    # 4.0 s

        class Bad(Sc):
            def get_ego_future_trajectory(self, a, b, n):
                return [St(i + 1) for i in range(10)]
        p2, why2 = BD.human_plan20(SimpleNamespace(sc=Bad(), ego=Ego()), None, "l", "t")
        self.assertIsNone(p2)
        self.assertIn("shape", why2)

        class Raise(Sc):
            def get_ego_future_trajectory(self, a, b, n):
                raise RuntimeError("no future")
        p3, why3 = BD.human_plan20(SimpleNamespace(sc=Raise(), ego=Ego()), None, "l", "t")
        self.assertIsNone(p3)
        self.assertIn("unavailable", why3)

    def test_real_two_failing_navtrain_instances(self):
        if os.environ.get("N6_SKIP_REAL"):
            self.skipTest("N6_SKIP_REAL")
        r = BD.reach_real(os.path.join(RUN, "n5", "fixture_real_instances.json"))
        self.assertTrue(r["ok"], r)
        for k, v in r["instances"].items():
            self.assertTrue(v["human_built"] and v["n6h_dac"] == 1.0 and v["n6x"] is None and v["deterministic"], (k, v))


# -------------------------------------------------------------------------------------------------------------- UNCH
class TestUnch(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(dir=TMP)
        n = 8
        rng = np.random.default_rng(1)
        self.tok = np.array([f"t{i:02d}" for i in range(n)])
        self.n5a = rng.normal(size=(n, 20, 3))
        self.p8 = rng.normal(size=(n, 8, 3)).astype(np.float32)
        np.savez(f"{self.d}/n5a_plans.npz", tokens=self.tok, logs=self.tok, plans20=self.n5a)
        np.savez(f"{self.d}/n5a_seam.npz", token=self.tok, poses=self.p8, fingerprint=np.array(["f"] * n), sampling=np.array([8.0, 0.5]), arm=np.array("x_refe_tfn5_n5a"))
        self.app = {"N6H": ["t02", "t03"], "N6T": ["t02"], "N6X": []}
        self.write()

    def write(self, mut=None):
        for arm in BD.ARMS:
            P, S = self.n5a.copy(), self.p8.copy()
            for t in self.app[arm]:
                i = list(self.tok).index(t)
                P[i, 5, 0] += 0.1
                S[i, 2, 0] += 0.1
            if mut and mut[0] == arm:
                mut[1](P, S)
            np.savez(f"{self.d}/n6_plans_{arm}.npz", tokens=self.tok, logs=self.tok, plans20=P, arm=np.array(arm))
            np.savez(f"{self.d}/{BD.SEAM_LABEL[arm]}.npz", token=self.tok, poses=S, fingerprint=np.array(["f"] * 8), sampling=np.array([8.0, 0.5]), arm=np.array("y"))

    def chk(self, n_res=3, expect=3):
        return BD.check_unch(self.d, f"{self.d}/n5a_plans.npz", f"{self.d}/n5a_seam.npz", self.app, n_res, expect)

    def test_green(self):
        r = self.chk()
        self.assertTrue(r["pass"], r)
        self.assertTrue(r["arms"]["N6X"]["bit_identical_to_n5a"])

    def test_2e6_perturbation_of_one_unchanged_plan_is_red(self):
        def m(P, S):
            P[0, 3, 1] += 2e-6
        self.write(("N6H", m))
        r = self.chk()
        self.assertFalse(r["pass"])
        self.assertEqual(r["arms"]["N6H"]["unflagged_changed"], ["t00"])

    def test_applied_to_all_tokens_is_red(self):
        def m(P, S):
            P += 1e-12
        self.write(("N6T", m))
        self.assertFalse(self.chk()["pass"])

    def test_flagged_but_unchanged_is_red(self):
        self.app["N6T"] = ["t02", "t06"]
        self.assertFalse(self.chk()["pass"])

    def test_seam_changed_without_plan_is_red(self):
        def m(P, S):
            S[5, 1, 1] += 1e-3
        self.write(("N6H", m))
        self.assertFalse(self.chk()["pass"])

    def test_n6x_must_be_bit_identical_to_n5a(self):
        def m(P, S):
            S[4, 0, 0] += np.float32(1e-6)
        self.write(("N6X", m))
        r = self.chk()
        self.assertFalse(r["pass"])
        self.assertFalse(r["arms"]["N6X"]["bit_identical_to_n5a"])

    def test_residual_count_must_equal_the_card(self):
        self.assertFalse(self.chk(n_res=21, expect=20)["pass"])
        self.assertFalse(self.chk(n_res=19, expect=20)["pass"])


# -------------------------------------------------------------------------------------------------------------- MERGE
class TestMerge(unittest.TestCase):
    """the build merge on 8 synthetic tokens (to_navsim is the real function; nothing touches the harness). t01: teacher candidates (and a human one); t02: only the human passes; t03: nothing"""

    def setUp(self):
        import n4_geom as G
        self.ND = G.ND
        self.d = tempfile.mkdtemp(dir=TMP)
        n = 8
        self.tok = [f"t{i:02d}" for i in range(n)]
        rng = np.random.default_rng(3)
        t = 0.2 * np.arange(1, 21)
        self.n5a = np.stack([np.stack([8 * t, 0.05 * rng.normal(size=20).cumsum() * 0.1, np.zeros(20)], -1) for _ in range(n)])
        np.savez(f"{self.d}/n5a_plans.npz", tokens=np.array(self.tok), logs=np.array([f"l{i % 2}" for i in range(n)]), plans20=self.n5a)
        p8 = np.stack([self.ND.to_navsim(self.n5a[i]).astype(np.float32) for i in range(n)])
        np.savez(f"{self.d}/n5a_seam.npz", token=np.array(self.tok), fingerprint=np.array(["f"] * n), poses=p8, sampling=np.array([8.0, 0.5]),
                 arm=np.array("TEACHER_x_refe_tfn5_n5a"))
        self.resid = ["t01", "t02", "t03"]
        self.write_csv()
        json.dump({"unrepaired_l1": {"N5A": self.resid}}, open(f"{self.d}/n5a_report.json", "w"))
        json.dump({"d_returns_plan": [], "l1_returns_plan": []}, open(f"{self.d}/replay.json", "w"))
        json.dump({"ok": True, "arms": [{"arm": a, "ok": True, "what": a} for a in ("D", "L1")], "real": {"ok": True, "instances": {"easy": {"ok": True}}}}, open(f"{self.d}/reach.json", "w"))
        self.rows = self.make_rows()

    def write_csv(self):
        with open(f"{self.d}/n5a.csv", "w", encoding="utf-8", newline="") as fo:
            w = csv.writer(fo)
            w.writerow(COLS)
            for i, tk in enumerate(self.tok):
                r = dict(OK5, dac=0.0 if tk in self.resid else 1.0, pdms=0.0 if tk in self.resid else 0.9)
                w.writerow([i, tk, True, r["nc"], r["dac"], r["ep"], r["ttc"], r["c"], r["ddc"], r["pdms"]])

    @staticmethod
    def cand(plan, src, k, cls, **m):
        base = dict(BD.REF0, dac=1.0, ep=0.5, pdms=0.5)
        base.update(m)
        return {"src": src, "k": k, "cls": cls, "plan": plan.tolist(), "m": base}

    def cands_of(self, tk):
        p = self.n5a[self.tok.index(tk)]
        sh = lambda d: p + np.array([0, d, 0.0])[None, :]                                                  # noqa: E731
        c = [self.cand(p, "D", 0, "teacher", dac=0.0, pdms=0.0, ep=0.0)]
        if tk == "t01":
            c += [self.cand(sh(0.1), "D", 1, "teacher", ttc=0.5, c=0.5, pdms=0.9),                          # the card's pick (TTC/C free)
                  self.cand(sh(0.2), "L1", 1, "teacher", ttc=0.5, c=0.5, pdms=0.9),                         # a tie: tie_latest takes it
                  self.cand(sh(0.3), "L1", 2, "teacher", nc=0.0, pdms=0.95),                                # NC worse: nc_drop takes it
                  self.cand(sh(0.4), "L1", 3, "teacher", ddc=0.5, pdms=0.97),                               # DDC worse: ddc_drop takes it
                  self.cand(sh(0.5), "H", 0, "human", pdms=0.99)]                                           # human_first takes it
        if tk == "t02":
            c += [self.cand(sh(0.1), "L1", 1, "teacher", dac=0.0, pdms=0.0, ep=0.0), self.cand(sh(0.6), "H", 0, "human", pdms=0.7)]
        if tk == "t03":
            c += [self.cand(sh(0.1), "D", 1, "teacher", pdms=0.0, ep=0.0),                                  # DAC 1 but builder PDMS not improved: pdms_drop takes it
                  self.cand(sh(0.7), "H", 0, "human", dac=0.0, pdms=0.0, ep=0.0)]
        return c

    def make_rows(self, mut=None, mut_arm=None):
        rows = []
        for i, tk in enumerate(self.tok):
            if tk not in self.resid:
                rows.append({"log": f"l{i % 2}", "token": tk, "ref": dict(OK5), "residual": False, "cderiv_maxdev_m": 0.0, "label_ego": "db", "t_s": 0.1})
                continue
            ref = dict(BD.REF0)
            cands = self.cands_of(tk)
            ch = {}
            for arm in BD.ARMS:
                fn = BD.make_selector(arm, mut if (mut and (mut_arm in (None, arm))) else "none")
                c = fn([dict(x) for x in cands], ref)
                ch[arm] = None if c is None else {"src": c["src"], "k": c["k"], "cls": c["cls"], "plan": c["plan"], "m": c["m"], "m_recheck": dict(c["m"])}
            hum = [x for x in cands if x["cls"] == "human"]
            rows.append({"log": f"l{i % 2}", "token": tk, "ref": ref, "residual": True, "cderiv_maxdev_m": 0.0, "n5a_equals_n2r_plan": True, "cands": cands, "chosen": ch,
                         "repl": {"D": {"returns_plan": False, "n_scored": 5, "info": {}}, "L1": {"returns_plan": False, "n_scored": 5, "info": {}}}, "det_equal": True,
                         "human": {"plan": hum[0]["plan"], "m": hum[0]["m"], "ego_src": "db"} if hum else None, "human_why": "ok", "label_ego": "db", "t_s": 0.2})
        return rows

    def merge(self, **over):
        with open(f"{self.d}/rows.jsonl", "w", encoding="utf-8") as fo:
            for r in self.rows:
                fo.write(json.dumps(r) + "\n")
        out = f"{self.d}/out"
        kw = dict(rows=[f"{self.d}/rows.jsonl"], n5a_plans=f"{self.d}/n5a_plans.npz", n5a_seam=f"{self.d}/n5a_seam.npz", n5a_report=f"{self.d}/n5a_report.json",
                  expect_residual_json="", replay_expect_json=f"{self.d}/replay.json", harness_csv_n5a=f"{self.d}/n5a.csv", reach_json=f"{self.d}/reach.json", out_dir=out,
                  expect_residual=3)
        kw.update(over)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = BD.cmd_merge(SimpleNamespace(**kw))
        return rc, json.load(open(f"{out}/n6_build_report.json")), out

    def red(self, ctrl, mut=None, mut_arm=None, also=()):
        self.rows = self.make_rows(mut, mut_arm)
        rc, rep, _ = self.merge()
        self.assertNotEqual(rc, 0, mut)
        self.assertFalse(rep["controls"][ctrl]["pass"], (mut, {k: v["pass"] for k, v in rep["controls"].items()}))
        for x in also:
            self.assertFalse(rep["controls"][x]["pass"], (mut, x))
        return rep

    def test_green_and_files(self):
        rc, rep, out = self.merge()
        self.assertEqual(rc, 0, {k: v for k, v in rep["controls"].items() if not v["pass"]})
        self.assertEqual(rep["repaired"]["N6H"], ["t01", "t02"])
        self.assertEqual(rep["repaired"]["N6T"], ["t01"])
        self.assertEqual(rep["repaired"]["N6X"], [])
        self.assertEqual(rep["counts"]["N6H"], {"teacher_tol": 1, "human": 1})
        self.assertEqual(rep["counts"]["N6T"], {"teacher_tol": 1, "human": 0})
        self.assertEqual(rep["source_class"]["N6H"], {"t01": "teacher", "t02": "human"})
        self.assertEqual(rep["residual_table"]["t01"]["resolved_by"], ["D", 1, "teacher"])
        self.assertIn("DAC-1 candidates exist but fail", rep["residual_table"]["t03"]["blocking_class"])
        s, sb = np.load(f"{out}/refe_tfn6_n6h.npz"), np.load(f"{self.d}/n5a_seam.npz")
        diff = [i for i in range(8) if not np.array_equal(s["poses"][i], sb["poses"][i])]
        self.assertEqual(diff, [1, 2])
        self.assertTrue(str(s["arm"]).endswith("refe_tfn6_n6h"))
        sx = np.load(f"{out}/refe_tfn6_n6x.npz")
        self.assertTrue(np.array_equal(sx["poses"], sb["poses"]))                                          # N6-X: bit-identical to N5-A
        # the changed seam row is ND.to_navsim of the chosen plan, the same function that made the banked rows
        want = self.ND.to_navsim(np.asarray(self.rows[1]["chosen"]["N6H"]["plan"])).astype(np.float32)
        self.assertTrue(np.array_equal(s["poses"][1], want))

    def test_npz_read_once_not_per_token(self):
        # TF-N5 merge OOM 2026-11-13: indexing an NpzFile per token re-reads (and, for kept rows, retains) the whole member each time.
        # Every npz member must be read at most once; the pre-fix per-token access (mutation) must make this probe go RED.
        real_load = np.load
        hits = {"n": 0, "members": 0}

        class Counting:
            def __init__(s, z):
                s.z, s.files = z, z.files

            def __getitem__(s, k):
                hits["n"] += 1
                return s.z[k]

            def close(s):
                s.z.close()

        def counting_load(p, *a, **k):
            z = real_load(p, *a, **k)
            if hasattr(z, "files"):
                hits["members"] += len(z.files)
                return Counting(z)
            return z
        BD.np.load = counting_load
        try:
            rc, rep, out = self.merge()
            self.assertEqual(rc, 0, rep["controls"])
            self.assertGreater(hits["members"], 0)
            self.assertLessEqual(hits["n"], hits["members"], hits)
            hits.update(n=0, members=0)
            fixed = BD._npz
            BD._npz = lambda p: counting_load(p, allow_pickle=False)
            try:
                self.merge()
            except Exception:
                pass
            finally:
                BD._npz = fixed
            self.assertGreater(hits["n"], hits["members"], ("mutation not detected", hits))
        finally:
            BD.np.load = real_load

    # --- the required mutation arms: the selector the derive RAN differs from the card; the merge must go RED
    def test_mutation_tolerance_off_strict_is_red(self):
        rep = self.red("C-SELECT", "strict")
        self.assertEqual(rep["source_class"]["N6H"], {"t01": "human", "t02": "human"})   # strict rejects the TTC/C-worse teacher candidates: the human fallback takes t01 instead
        self.assertEqual(rep["repaired"]["N6T"], [])

    def test_mutation_nc_guard_dropped_is_red(self):
        rep = self.red("C-SELECT", "nc_drop", also=("C-CERT-B6",))
        self.assertIn("t01", rep["repaired"]["N6H"])

    def test_mutation_ddc_guard_dropped_is_red(self):
        self.red("C-SELECT", "ddc_drop", also=("C-CERT-B6",))

    def test_mutation_pdms_clause_dropped_is_red(self):
        self.red("C-SELECT", "pdms_drop", also=("C-CERT-B6",))

    def test_mutation_human_first_ordering_is_red(self):
        self.red("C-SELECT", "human_first", also=("C-ORDER",))

    def test_mutation_dac_requirement_inverted_is_red(self):
        self.red("C-SELECT", "dac_inverted", also=("C-X0",))
        self.red("C-X0", "dac_inverted", mut_arm="N6X")                          # only the X arm's certificate flipped to require DAC = 1: it now changes tokens

    def test_mutation_tie_rule_latest_is_red(self):
        self.red("C-SELECT", "tie_latest")

    def test_recheck_differs_is_red(self):
        self.rows[1]["chosen"]["N6H"]["m_recheck"] = dict(self.rows[1]["chosen"]["N6H"]["m"], ttc=0.4)
        rc, rep, _ = self.merge()
        self.assertFalse(rep["controls"]["C-CERT-B6"]["pass"])

    def test_certificate_dac_off_is_red_at_c_cert_b6(self):
        for k in ("m", "m_recheck"):
            self.rows[1]["chosen"]["N6H"][k] = dict(self.rows[1]["chosen"]["N6H"][k], dac=0.0)
        rc, rep, _ = self.merge()
        self.assertFalse(rep["controls"]["C-CERT-B6"]["pass"])

    def test_non_deterministic_capture_is_red(self):
        self.rows[1]["det_equal"] = False
        rc, rep, _ = self.merge()
        self.assertFalse(rep["controls"]["C-DET"]["pass"])

    def test_wrong_dac_map_is_red(self):
        self.rows[6]["ref"]["dac"] = 0.0
        rc, rep, _ = self.merge()
        self.assertFalse(rep["controls"]["C-DACMAP6"]["pass"])

    def test_cache_ego_label_or_human_ego_is_red(self):
        self.rows[2]["label_ego"] = "cache"
        rc, rep, _ = self.merge()
        self.assertFalse(rep["controls"]["C-LABELEGO"]["pass"])
        self.setUp()
        self.rows[1]["human"]["ego_src"] = "cache"
        rc, rep, _ = self.merge()
        self.assertFalse(rep["controls"]["C-LABELEGO"]["pass"])

    def test_missing_row_and_changed_residual_set_are_red(self):
        self.rows.pop(4)
        rc, rep, _ = self.merge()
        self.assertFalse(rep["controls"]["C-COVB"]["pass"])
        self.setUp()
        self.rows[3]["ref"]["dac"] = 1.0                       # t03 no longer builder-DAC-0: the set differs from the banked TF-N5 A1 residual
        rc, rep, _ = self.merge()
        self.assertFalse(rep["controls"]["C-SET"]["pass"])

    def test_extra_residual_token_is_red_at_c_set(self):
        self.rows[5]["ref"]["dac"] = 0.0
        self.rows[5].update(residual=True, cands=self.cands_of("t01"), chosen={a: None for a in BD.ARMS}, repl=self.rows[1]["repl"], det_equal=True, human=None,
                            n5a_equals_n2r_plan=True)
        rc, rep, _ = self.merge()
        self.assertFalse(rep["controls"]["C-SET"]["pass"])

    def test_replay_violation_is_red(self):
        self.rows[1]["repl"]["L1"]["returns_plan"] = True        # the banked L1 returned no plan on this token; the re-run did
        rc, rep, _ = self.merge()
        self.assertFalse(rep["controls"]["C-REPLAY"]["pass"])
        self.setUp()
        self.rows[2]["cands"][0]["plan"][0][0] += 1e-6           # iterate 0 is not the banked plan
        rc, rep, _ = self.merge()
        self.assertFalse(rep["controls"]["C-REPLAY"]["pass"])
        self.setUp()
        rc, rep, _ = self.merge(replay_expect_json="")
        self.assertFalse(rep["controls"]["C-REPLAY"]["pass"])

    def test_reach_json_missing_or_failed_is_red(self):
        json.dump({"ok": False, "arms": [], "real": {"ok": False, "instances": {}}}, open(f"{self.d}/reach.json", "w"))
        rc, rep, _ = self.merge()
        self.assertFalse(rep["controls"]["C-REACH"]["pass"])
        rc, rep, _ = self.merge(reach_json="")
        self.assertFalse(rep["controls"]["C-REACH"]["pass"])

    def test_seamfn_mismatch_is_red(self):
        sb = dict(np.load(f"{self.d}/n5a_seam.npz"))
        sb["poses"] = sb["poses"].copy()
        sb["poses"][3, 0, 0] += np.float32(1e-3)
        np.savez(f"{self.d}/n5a_seam.npz", **sb)
        rc, rep, _ = self.merge()
        self.assertFalse(rep["controls"]["C-SEAMFN"]["pass"])

    def test_residual_count_must_equal_the_card(self):
        rc, rep, _ = self.merge(expect_residual=20)
        self.assertFalse(rep["controls"]["C-UNCH"]["pass"])


# ---------------------------------------------------------------------------------------------------------- analyzer
class Syn:
    """2000 tokens, 20 logs, 2 known36 -> 1998 primary. 20 N5-A residual tokens (zero DAC in N5-A): the first 20 non-known tokens"""
    N, G = 2000, 20

    def __init__(self, tmp):
        self.d = tmp
        self.tok = [f"k{i:04d}" for i in range(self.N)]
        self.log = {t: f"log{i % self.G}" for i, t in enumerate(self.tok)}
        self.known = self.tok[-2:]
        self.resid = self.tok[:20]
        json.dump({"over_bar": {t: 1 for t in self.known}}, open(f"{tmp}/known.json", "w"))
        with gzip.open(f"{tmp}/tl.json.gz", "wt", encoding="utf-8") as fh:
            json.dump({"tokens": {t: {"log_name": self.log[t]} for t in self.tok}}, fh)
        os.makedirs(f"{tmp}/pe", exist_ok=True)
        open(f"{tmp}/pe/proxy_eval.py", "w").write(
            "import importlib.util\nimport sys\nspec = importlib.util.spec_from_file_location('a1', %r)\nA = importlib.util.module_from_spec(spec)\nspec.loader.exec_module(A)\n"
            "def boot(pt, tl):\n    return A.Boot(sorted(pt), tl).stat(pt)['_raw']\n" % A1)
        base = {t: dict(nc=1.0, dac=1.0, ddc=1.0, ep=0.9, ttc=1.0, c=1.0, pdms=0.97) for t in self.tok}
        for t in self.resid:
            base[t].update(dac=0.0, pdms=0.0)
        self.arms = {"N2R": {t: dict(v) for t, v in base.items()}, "N4D": {t: dict(v) for t, v in base.items()}, "N5A": base,
                     "N6H": {t: dict(v) for t, v in base.items()}, "N6T": {t: dict(v) for t, v in base.items()}}
        self.rep = {"N6H": [], "N6T": [], "N6X": []}
        self.cls = {"N6H": {}, "N6T": {}, "N6X": {}}
        self.build = {"verdict": "PASS", "controls": {"C-CERT-B6": {"pass": True}, "C-DACMAP6": {"pass": True}, "C-X0": {"pass": True}}, "repaired": self.rep, "source_class": self.cls,
                      "residual_table": {t: {"blocking_class": "no candidate (teacher or human) reaches DAC = 1"} for t in self.resid}}
        self.unch = {"pass": True, "arms": {"N6H": {"pass": True}, "N6T": {"pass": True}, "N6X": {"pass": True, "bit_identical_to_n5a": True}}, "n_residual": 20}
        self.pins_ok()

    def fix(self, arm, toks, pdms=0.97, cls="teacher", **m):
        for t in toks:
            self.arms[arm][t].update(dac=1.0, pdms=pdms, **m)
            self.cls[arm][t] = cls
        self.rep[arm] = sorted(set(self.rep[arm]) | set(toks))

    def pins_ok(self, fam_sha=None):
        os.makedirs(f"{self.d}/w/pk/eval/instruments", exist_ok=True)
        fp = f"{self.d}/{AN.A5.TIP_FAMILIES}"
        open(fp, "w").write("tip")
        self.fam_sha = sha(fp)
        open(f"{self.d}/pinned.txt", "w").write("x")
        AN.A5.TIP_FAMILIES_SHA256 = fam_sha or self.fam_sha
        json.dump({"files": {"pinned.txt": sha(f"{self.d}/pinned.txt"), AN.A5.TIP_FAMILIES: sha(fp)}}, open(f"{self.d}/pins.json", "w"))

    def write(self):
        for nm, rows in self.arms.items():
            with open(f"{self.d}/{nm}.csv", "w", encoding="utf-8", newline="") as fo:
                w = csv.writer(fo)
                w.writerow(COLS)
                for i, t in enumerate(self.tok):
                    if t not in rows:
                        continue
                    r = rows[t]
                    w.writerow([i, t, True, r["nc"], r["dac"], r["ep"], r["ttc"], r["c"], r["ddc"], r["pdms"]])
        json.dump(self.build, open(f"{self.d}/build.json", "w"))
        json.dump(self.unch, open(f"{self.d}/unch.json", "w"))

    def run(self, k=49):
        self.write()
        a = SimpleNamespace(harness_csv=[f"{n}={self.d}/{n}.csv" for n in self.arms], build_report=f"{self.d}/build.json", unch=f"{self.d}/unch.json",
                            pins=f"{self.d}/pins.json", pin_root=self.d, proxy_eval=f"{self.d}/pe", prereg_sha256="abc", token_log=f"{self.d}/tl.json.gz",
                            known36=f"{self.d}/known.json", k=k, a1_analyze=A1, expect_n=self.N, expect_primary=self.N - 2, id_tol=1e-6, out=f"{self.d}/out.json")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = AN.evaluate(a)
        return rc, buf.getvalue(), json.load(open(a.out, encoding="utf-8"))


class TestRead(unittest.TestCase):
    def setUp(self):
        self._tip = AN.A5.TIP_FAMILIES_SHA256
        self.S = Syn(tempfile.mkdtemp(dir=TMP))

    def tearDown(self):
        AN.A5.TIP_FAMILIES_SHA256 = self._tip

    def fix19(self, **kw):
        self.S.fix("N6H", self.S.resid[:19], **kw)

    def test_confirmed_naming_split_and_k_in_bonferroni_key(self):
        self.S.fix("N6H", self.S.resid[:15], cls="teacher")
        self.S.fix("N6H", self.S.resid[15:19], cls="human")
        self.S.fix("N6T", self.S.resid[:15])
        rc, out, res = self.S.run(k=49)
        self.assertEqual(rc, 0, res["controls"])
        p = res["primary"]
        self.assertEqual(p["outcome"], "CONFIRMED", p)
        self.assertEqual((p["n_teacher_tol_replacements"], p["n_human_replacements"]), (15, 4))
        self.assertIn("human-log fallback on 4 of 12,110 tokens", res["naming"])
        self.assertIn("N_teacher_tol = 15", res["naming"])
        self.assertIn("tolerance (NC/DDC kept, TTC/C free)", res["naming"])
        self.assertIn("NOT teacher output", res["human_log_notice"])
        self.assertIn("ci_bonferroni_K49", p["delta_vs_N5A_paired"]["pdms"])
        self.assertEqual(p["bars"], {"1_paper_level_pdms_ge_95.8_and_dac_ge_99.9": True, "2_no_harm_nc_ddc": True, "3_certificate_integrity": True})
        self.assertEqual(res["paper_level"]["N6H"]["level"], "PAPER LEVEL")
        self.assertEqual(res["paper_level"]["N6T_teacher_only"]["zero_dac_tokens"], 20 - 15)
        self.assertEqual(res["remaining_zero_dac_N6H"]["count"], 1)
        self.assertIn("blocking", res["remaining_zero_dac_N6H"])

    def test_paper_level_boundary_is_the_point_estimate(self):
        self.S.fix("N6H", self.S.resid[:18])                      # 2 zero-DAC of 1998 -> DAC 99.8999 < 99.9
        rc, _, res = self.S.run()
        self.assertEqual(rc, 0)
        self.assertFalse(res["primary"]["bars"]["1_paper_level_pdms_ge_95.8_and_dac_ge_99.9"])
        self.assertEqual(res["primary"]["outcome"], "NOT CONFIRMED")
        self.assertEqual(res["primary"]["not_confirmed_branches_in_card_order"], ["paper_level_not_reached"])
        self.assertEqual(res["paper_level"]["N6H"]["level"], "NEAR")
        self.setUp()
        self.fix19()
        rc, _, res = self.S.run()
        self.assertEqual(res["primary"]["outcome"], "CONFIRMED")

    def test_pdms_below_95_8_is_not_paper_level_even_at_dac_999(self):
        for arm in ("N2R", "N4D", "N5A", "N6H", "N6T"):
            for t in self.S.tok:
                if self.S.arms[arm][t]["dac"] == 1.0:
                    self.S.arms[arm][t]["pdms"] = 0.955
        self.fix19(pdms=0.955)
        rc, _, res = self.S.run()
        self.assertEqual(rc, 0)
        self.assertFalse(res["paper_level"]["N6H"]["paper_pdms_point_ok"])
        self.assertTrue(res["paper_level"]["N6H"]["paper_dac_point_ok"])
        self.assertFalse(res["primary"]["bars"]["1_paper_level_pdms_ge_95.8_and_dac_ge_99.9"])

    def test_a_changed_token_losing_nc_or_ddc_is_the_harm_branch(self):
        self.fix19()
        self.S.arms["N6H"][self.S.resid[0]]["nc"] = 0.5          # N5-A's NC is 1.0 on that token: a per-token violation even though the mean barely moves
        rc, _, res = self.S.run()
        self.assertEqual(rc, 0)
        p = res["primary"]
        self.assertEqual(p["no_harm_changed_token_violations"]["count"], 1)
        self.assertFalse(p["bars"]["2_no_harm_nc_ddc"])
        self.assertIn("harm", p["not_confirmed_branches_in_card_order"])
        self.setUp()
        self.fix19()
        self.S.arms["N6H"][self.S.resid[1]]["ddc"] = 0.5
        rc, _, res = self.S.run()
        self.assertEqual(res["primary"]["no_harm_changed_token_violations"]["tokens"], [self.S.resid[1]])
        self.assertIn("harm", res["primary"]["not_confirmed_branches_in_card_order"])

    def test_nc_ddc_improvement_on_changed_tokens_is_not_harm_and_ttc_c_worse_is_the_declared_tolerance(self):
        for t in self.S.resid[:19]:
            self.S.arms["N5A"][t].update(nc=0.5, ddc=0.5)        # N5-A's own NC / DDC on the residual are worse than the repaired plans'
            for arm in ("N2R", "N4D", "N6T"):
                self.S.arms[arm][t].update(nc=0.5, ddc=0.5)
        self.S.fix("N6H", self.S.resid[:19], ttc=0.0, c=0.0)
        for t in self.S.resid[:19]:
            self.S.arms["N6H"][t].update(nc=1.0, ddc=1.0)
        rc, _, res = self.S.run()
        self.assertEqual(rc, 0, res["controls"])
        p = res["primary"]
        self.assertTrue(p["bars"]["2_no_harm_nc_ddc"], p["no_harm_paired_x100_N6H_minus_N5A"])
        self.assertLess(p["tolerance_declared_ttc_c_reported"]["ttc"]["mean"], 0.0)                     # reported, not a bar
        self.assertEqual(p["outcome"], "CONFIRMED")

    def test_mean_level_nc_harm_alone_is_red(self):
        self.fix19()
        for t in self.S.resid[:19]:
            self.S.arms["N6H"][t]["nc"] = 0.0                    # also trips the per-token clause; assert the mean is read as negative
        rc, _, res = self.S.run()
        self.assertLess(res["primary"]["no_harm_paired_x100_N6H_minus_N5A"]["nc"]["mean"], 0.0)
        self.assertFalse(res["primary"]["bars"]["2_no_harm_nc_ddc"])

    def test_cert_integrity_branch(self):
        self.fix19()
        self.S.build["controls"]["C-CERT-B6"] = {"pass": False}
        self.S.build["verdict"] = "PASS"                         # builder verdict claims PASS but the bar reads the C-CERT-B6 control itself
        rc, _, res = self.S.run()
        self.assertEqual(rc, 0)
        self.assertFalse(res["primary"]["bars"]["3_certificate_integrity"])
        self.assertIn("cert_integrity", res["primary"]["not_confirmed_branches_in_card_order"])

    def test_certificate_disabled_is_red_at_c_cert_and_no_aggregate(self):
        self.S.rep["N6H"] = [self.S.resid[0]]                   # builder says certified, harness DAC 0
        self.S.cls["N6H"][self.S.resid[0]] = "teacher"
        rc, out, res = self.S.run()
        self.assertEqual(rc, 1)
        self.assertFalse(res["controls"]["C-CERT"]["pass"])
        self.assertNotIn("primary", res)
        self.assertNotIn("paper_level", res)
        self.assertIn("NO VERDICT", res["verdict"])

    def test_2e6_perturbation_is_red_at_c_id_c_for_both_arms(self):
        for arm in ("N6H", "N6T"):
            self.setUp()
            self.S.arms[arm]["k0500"]["pdms"] += 2e-6
            rc, _, res = self.S.run()
            self.assertEqual(rc, 1, arm)
            self.assertFalse(res["controls"]["C-ID-C"]["pass"])
            self.assertEqual(res["controls"]["C-ID-C"]["arms"][arm]["tokens_over_tol"], 1)

    def test_an_unflagged_token_that_moved_is_red(self):
        self.fix19()
        self.S.arms["N6H"]["k0100"]["dac"] = 0.0                # a DAC-1 N5-A token reads 0 in N6-H and is NOT in the applied set
        rc, out, res = self.S.run()
        self.assertEqual(rc, 1)
        self.assertFalse(res["controls"]["C-ID-C"]["pass"])
        self.assertNotIn("primary", res)

    def test_dropped_row_is_red_at_c_cov(self):
        self.S.arms["N6H"].pop(self.S.tok[29])
        rc, _, res = self.S.run()
        self.assertEqual(rc, 1)
        self.assertFalse(res["controls"]["C-COV"]["pass"])

    def test_n6x_that_changed_anything_is_red_at_c_reg(self):
        self.fix19()
        self.S.rep["N6X"] = [self.S.resid[0]]
        rc, _, res = self.S.run()
        self.assertEqual(rc, 1)
        self.assertFalse(res["controls"]["C-REG"]["pass"])
        self.assertNotIn("primary", res)
        self.setUp()
        self.fix19()
        self.S.unch["arms"]["N6X"]["bit_identical_to_n5a"] = False
        rc, _, res = self.S.run()
        self.assertEqual(rc, 1)
        self.assertFalse(res["controls"]["C-REG"]["pass"])

    def test_pinned_file_edited_is_red_at_c_pin(self):
        self.fix19()
        open(f"{self.S.d}/pinned.txt", "w").write("edited")
        rc, _, res = self.S.run()
        self.assertEqual(rc, 1)
        self.assertFalse(res["controls"]["C-PIN"]["pass"])
        self.assertIn("pinned.txt", res["controls"]["C-PIN"]["mismatch"])

    def test_an_old_families_run_pinned_is_red_at_c_pin(self):
        self.fix19()
        AN.A5.TIP_FAMILIES_SHA256 = "44c62807f1254e2ca4a41e4094097071b96bccaba9e754703ddf4a689e4d73a6"      # the real TIP; the tree pins and holds a different file
        rc, _, res = self.S.run()
        self.assertEqual(rc, 1)
        self.assertFalse(res["controls"]["C-PIN"]["pass"])
        self.assertFalse(res["controls"]["C-PIN"]["families_run_is_tip"])

    def test_c_est_detects_a_wrong_draw_scheme(self):
        self.fix19()
        open(f"{self.S.d}/pe/proxy_eval.py", "w").write("def boot(pt, tl):\n    return (0.0, 0.0, 0.0)\n")
        rc, _, res = self.S.run()
        self.assertFalse(res["controls"]["C-EST"]["pass"])


class TestFamread(unittest.TestCase):
    def _run(self, doc):
        d = tempfile.mkdtemp(dir=TMP)
        json.dump(doc, open(f"{d}/ci.json", "w"))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            AN.cmd_famread(SimpleNamespace(fam_ci=f"{d}/ci.json", out=f"{d}/o.json", k=49))
        return json.load(open(f"{d}/o.json"))

    def doc(self, sb, sm, cm):
        return {"frames": {"executed_tracked": {"families": {"longitudinal": {"speed_bias_mps": {"diff_B_minus_A": {"point": sb, "lo": 0, "hi": 0}},
                                                                              "speed_mae_mps": {"diff_B_minus_A": {"point": sm, "lo": 0, "hi": 0}}},
                                                            "lateral": {"cross_mae_m": {"diff_B_minus_A": {"point": cm, "lo": 0, "hi": 0}}}}}}}

    def test_no_flag_and_each_flag(self):
        self.assertFalse(self._run(self.doc(0.0, 0.0, 0.0))["guard"]["flag"])
        self.assertTrue(self._run(self.doc(0.30, 0.0, 0.0))["guard"]["flag"])
        self.assertTrue(self._run(self.doc(0.0, 0.2, 0.0))["guard"]["flag"])
        self.assertTrue(self._run(self.doc(0.0, 0.0, 0.06))["guard"]["flag"])
        r = self._run(self.doc(0.0, 0.0, 0.0))
        self.assertEqual(r["strategic"], "PAUSED (PI 2026-09-19)")
        self.assertEqual(r["K"], 49)
        self.assertEqual(AN.GUARD, {"speed_bias_diff_mps": 0.25, "speed_mae_diff_mps": 0.15, "cross_mae_diff_m": 0.05})
        for k in ("lateral.heading_mae_deg", "lateral.curvature_mae_1pm", "lateral.yaw_rate_mae_degps", "tactical.maneuver_5way_collapsed.accuracy"):
            self.assertIn(k, r["reported_not_bars"])

    def test_a_ci_without_paired_diffs_is_refused_not_read_as_a_flag(self):
        with self.assertRaises(SystemExit):
            self._run({"frames": {"executed_tracked": {"families": {}}}})


# ----------------------------------------------------------------------------------------------------------- CHAIN
LEDGER_N5 = {"key": "TF-N5:aaaa:E3", "n": 48}


class TestChain(unittest.TestCase):
    """guards on a synthetic mini run tree (dummy REQUIRED files; the cache listing and the reach subprocess are stubbed)"""

    def setUp(self):
        self._tip, self._reach = CH.TIP_FAMILIES_SHA256, CH.run_reach
        self.run = tempfile.mkdtemp(dir=TMP).replace("\\", "/")
        for rel in CH.REQUIRED:
            if os.path.isabs(rel) or rel[1:2] == ":":
                continue
            p = os.path.join(self.run, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "w").write("dummy " + rel)
        os.makedirs(f"{self.run}/repo", exist_ok=True)
        open(f"{self.run}/repo/x.py", "w").write("x")
        self.card = f"{self.run}/card.md"
        open(self.card, "w").write("card")
        CH.TIP_FAMILIES_SHA256 = sha(f"{self.run}/{CH.TIP_FAMILIES}")
        CH.run_reach = lambda a, real=False, out="": (0, "ok")
        self.mk_pins()
        self.write_ledger()

    def tearDown(self):
        CH.TIP_FAMILIES_SHA256, CH.run_reach = self._tip, self._reach

    def mk_pins(self):
        files = {}
        for top in CH.WATCH_TOP:
            for d, _, fs in os.walk(f"{self.run}/{top}"):
                for f in fs:
                    if f.endswith(CH.WATCH_EXT):
                        rel = os.path.relpath(os.path.join(d, f), self.run).replace(os.sep, "/")
                        if rel != CH.PINS_REL:
                            files[rel] = sha(os.path.join(d, f))
        for rel in CH.REQUIRED:
            files[rel] = sha(rel) if (os.path.isabs(rel) or rel[1:2] == ":") else sha(os.path.join(self.run, rel))
        self.pins = f"{self.run}/{CH.PINS_REL}"
        json.dump({"metric_cache_listing_sha256": "x", "files": files}, open(self.pins, "w"))

    def write_ledger(self, n=49, extra=None, drop_prev=False):
        key = f"TF-N6:{sha(self.card)}:E3"
        reads = ([] if drop_prev else [LEDGER_N5]) + [{"key": key, "n": n}] + (extra or [])
        json.dump({"reads": reads}, open(f"{self.run}/ledger.json", "w"))

    def args(self, **kw):
        d = dict(run_dir=self.run, card=self.card, card_sha256=sha(self.card), pins_sha256=sha(self.pins), ledger=f"{self.run}/ledger.json", k=49, procs=2,
                 min_free_virt_mb=1, min_free_gb=0.0, retries=1, stop_after="", check_only=False, dry_run=False, skip_cache_listing=True)
        d.update(kw)
        return SimpleNamespace(**d)

    def refused(self, a, frag):
        with self.assertRaises(SystemExit) as c:
            CH.guards(a)
        self.assertIn("REFUSING", str(c.exception))
        self.assertIn(frag, str(c.exception))

    def test_green(self):
        self.assertEqual(CH.guards(self.args())["reach"], "PASS")

    def test_card_sha_and_ledger(self):
        self.refused(self.args(card_sha256="0" * 64), "sha256(")
        self.write_ledger(n=48)
        self.refused(self.args(), "n = 48")
        self.write_ledger(extra=[{"key": "other", "n": 50}])
        self.refused(self.args(), "n = 50")
        self.write_ledger(drop_prev=True)
        self.refused(self.args(), "n = 48) is not in the ledger")
        json.dump({"reads": [LEDGER_N5]}, open(f"{self.run}/ledger.json", "w"))
        self.refused(self.args(), "no entry")
        self.write_ledger()
        self.refused(self.args(k=47), "--k is 47")

    def test_pins_sha_edited_and_unpinned_files(self):
        self.refused(self.args(pins_sha256="0" * 64), "pins-sha256")
        open(f"{self.run}/n6/n6_nt_build.py", "a").write("#")
        self.refused(self.args(), "pinned files differ")
        self.mk_pins()
        a = self.args(pins_sha256=sha(self.pins))
        open(f"{self.run}/n6/extra_tool.py", "w").write("x")
        self.refused(a, "unpinned files")

    def test_old_families_run_is_refused(self):
        """the TF-N4 A1 failure: the tree holds (and pins) an older families_run.py; the guard compares with the TIP constant, not with the pins"""
        CH.TIP_FAMILIES_SHA256 = "44c62807f1254e2ca4a41e4094097071b96bccaba9e754703ddf4a689e4d73a6"
        self.refused(self.args(), "TIP version")
        CH.TIP_FAMILIES_SHA256 = sha(f"{self.run}/{CH.TIP_FAMILIES}")
        open(f"{self.run}/{CH.TIP_FAMILIES}", "w").write("older families_run")
        self.refused(self.args(), "pinned files differ")

    def test_procs_and_a_failing_reach(self):
        self.refused(self.args(procs=3), "--procs")
        CH.run_reach = lambda a, real=False, out="": (1, "ZZN6_REACH BAD selector N6H nc: built False != card True")
        self.refused(self.args(), "reachability")

    def test_low_disk_is_refused(self):
        self.refused(self.args(min_free_gb=1e9), "GB free")

    def test_families_json_check_refuses_a_json_without_tokens(self):
        d = tempfile.mkdtemp(dir=TMP)
        json.dump({"arm": "N5A", "frames": {}}, open(f"{d}/old.json", "w"))
        json.dump({"arm": "N5A", "tokens": ["a", "b"], "logs": ["l", "l"], "frames": {}}, open(f"{d}/new.json", "w"))
        self.assertFalse(CH.fam_json_ok(f"{d}/old.json")[0])
        self.assertTrue(CH.fam_json_ok(f"{d}/new.json")[0])
        self.assertFalse(CH.fam_json_ok(f"{d}/missing.json")[0])

    def test_dry_run_prints_every_stage_and_no_mutation_flag_and_writes_nothing(self):
        argv = ["n6_nt_chain.py", "--run-dir", self.run, "--card", self.card, "--card-sha256", sha(self.card), "--pins-sha256", sha(self.pins), "--ledger",
                f"{self.run}/ledger.json", "--k", "49", "--dry-run", "--skip-cache-listing", "--min-free-gb", "0"]
        buf = io.StringIO()
        old = sys.argv
        sys.argv = argv
        try:
            with contextlib.redirect_stdout(buf):
                rc = CH.main()
        finally:
            sys.argv = old
        out = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertFalse(os.path.exists(f"{self.run}/n6run"))
        for frag in ("n6_nt_build.py reach", "n6_nt_build.py derive", "n6_nt_build.py merge", "n6_nt_build.py unch", "score_p4_a1.py --label refe_tfn6_n6h",
                     "score_p4_a1.py --label refe_tfn6_n6t", "n6_nt_analyze.py read", "transp_score.py", "famprep", "families_run.py", "--per-window", "fam_ci.py", "famread"):
            self.assertIn(frag, out, frag)
        self.assertNotIn("--mutate", out)
        self.assertNotIn("--smoke-allow-mutation", out)
        self.assertNotIn("refe_tfn6_n6x", out)                                   # N6-X is proven bit-identical and is not scored
        self.assertIn("N6H=", out)
        self.assertIn("N5A=", out)
        self.assertIn("ZZN6_CHAIN DONE", out)

    def test_make_pins_refuses_a_tree_with_an_old_families_run(self):
        import n6_make_pins as MP
        CH.TIP_FAMILIES_SHA256 = "44c62807f1254e2ca4a41e4094097071b96bccaba9e754703ddf4a689e4d73a6"
        old = sys.argv
        sys.argv = ["x", "--run-dir", self.run, "--tip", "t", "--out", f"{self.run}/p.json", "--skip-cache-listing"]
        try:
            with self.assertRaises(SystemExit):
                MP.main()
        finally:
            sys.argv = old


if __name__ == "__main__":
    unittest.main()
