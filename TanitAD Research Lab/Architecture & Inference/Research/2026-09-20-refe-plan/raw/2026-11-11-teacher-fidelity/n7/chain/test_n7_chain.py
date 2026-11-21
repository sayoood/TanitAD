"""TF-N7 chain tests: CPU, synthetic data and fake scorers, plus REAL NAVTRAIN tokens (no navtest). Run (navsim venv, N4_RUN = the run tree, CUDA off):
    N4_RUN=C:/Users/Admin/n7chain/run  python -m pytest -q n7/test_n7_chain.py
 Env: N7_SKIP_REAL=1 skips the two tests that execute the cascade on real navtrain tokens (reach_real: about 4 min; they need the nuPlan DB);
      N7_DEV_TAR (default: the n7_gen hold's raw/candidates_navtrain.tar.xz) = the development's stored real candidates; N7_SMOKE_DIR = the smoke dir (smoke/in + derive_clean.jsonl).

 CARD    the generator's tables (n7_lib.LEVERS, n7_select.TIERS, the grids n7_select decodes, the certificate) read back as the card; the card TEXT carries the same numbers; every
         selector mutation (tolerance off, NC / DDC / PDMS clause dropped, DAC inverted, tie latest, tier order reversed, floor off, T4 inside T3, a tier dropped) moves the behavioural
         signature AND is refused; a flag with a disabling default is found by the audit (regression arm); a mutated table is refused
 SELECT  the card semantics on hand-built candidate sets: tier order, T0 beats later tiers, the cap-5 prefix, tie rule, the floor, every certificate clause
 REAL    the 17 + 73 REAL navtrain candidate sets of the development: the build's selector == the development's selection (tier, lever, k) on 90 of 90, the dev cascade agrees, every tier
         has tokens, a tier list without a tier no longer chooses at it, N7-T3 == N7-T off T4, N7-X none; every tier executes on a real token through the build's own code path
 CAPTURE the generator through the real repair loops on a fake scorer: deterministic, a 1e-9 edit is seen by C-DET, the tolerance rescues what the full certificate refuses
 UNCH / MERGE / READ / FAMREAD / CHAIN  as the TF-N6 tests, for three arms (N7T3, N7T, N7X), no human candidate, plus C-N6SAME (live, banked report, banked plans, read side) and C-REACH"""
import contextlib
import csv
import functools
import gzip
import hashlib
import io
import json
import os
import re
import sys
import tarfile
import tempfile
import unittest
from types import SimpleNamespace

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):          # BEFORE numpy: a BLAS thread count above 1 moves the real navtrain plans by about 1e-6 (the bit-identity checks)
    os.environ[_v] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("N4_RUN", os.path.dirname(HERE))
sys.path.insert(0, HERE)
import n7_nt_analyze as AN      # noqa: E402
import n7_nt_build as BD        # noqa: E402
import n7_nt_chain as CH        # noqa: E402

RUN = os.environ["N4_RUN"]
A1 = os.environ.get("N7_A1_ANALYZE", os.path.join(RUN, "n2", "tf_n1_analyze_a1.py"))
TMP = os.environ.get("N7_TMP") or None
TF = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/teacher_fidelity"
DEV_TAR = os.environ.get("N7_DEV_TAR", TF + "/n7_gen/raw/candidates_navtrain.tar.xz")
DEV_SEL17 = os.environ.get("N7_DEV_SEL17", TF + "/n7_gen/raw/sel17_floor05.json")
DEV_SEL73 = os.environ.get("N7_DEV_SEL73", TF + "/n7_gen/raw/sel73_floor05.json")
CARD_MD = os.environ.get("N7_CARD", os.path.join(HERE, "PREREG_TF_N7.md") if os.path.exists(os.path.join(HERE, "PREREG_TF_N7.md")) else TF + "/n7_chain/PREREG_TF_N7.md")
COLS = ["", "token", "valid", "no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound", "comfort",
        "driving_direction_compliance", "score"]
OK5 = {"dac": 1.0, "nc": 1.0, "ddc": 1.0, "ttc": 1.0, "c": 1.0, "ep": 0.9, "pdms": 0.9}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def ident(c):
    return None if c is None else [c["tier"], c["lever"], c["k"]]


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------- CARD
class TestCard(unittest.TestCase):
    def test_card_tables_read_back_and_the_certificate_is_the_n6_one(self):
        r = BD.check_card7(BD.build_parser(), {"--route"})
        self.assertTrue(r["ok"], r["bad"])
        self.assertEqual([n for n, _ in BD.S.TIERS], ["T0_N6", "T1_L2", "T2_STR", "T3_OFF", "T4_DEEP"])
        self.assertEqual(BD.FLOOR, 0.5)
        n6t = BD.B6.CARD6["N6T"]
        self.assertEqual({k: n6t[k] for k in BD.CERT}, BD.CERT)                                      # C-CERT-B7: CARD6 N6T == the card certificate
        self.assertEqual(BD.L.CERT_TOL, BD.CERT)
        self.assertFalse(BD.CERT["ttc"] or BD.CERT["c"])
        self.assertTrue(BD.CERT["nc"] and BD.CERT["ddc"] and BD.CERT["pdms_gt"] and BD.CERT["dac"] == 1.0)
        self.assertEqual(BD.CERT_X["dac"], 0.0)
        self.assertEqual(tuple(BD.ARMS), ("N7T3", "N7T", "N7X"))
        self.assertEqual([n for n, _ in BD.CARD7["N7T3"]["tiers"]], ["T0_N6", "T1_L2", "T2_STR", "T3_OFF"])      # N7-T3: no T4
        self.assertIsNone(BD.CARD7["N7T3"]["floor"])
        self.assertEqual(BD.CARD7["N7T"]["floor"], 0.5)

    def test_a_mutated_table_is_refused_by_the_readback(self):
        import copy
        L, S = BD.L, BD.S
        saved = (copy.deepcopy(L.LEVERS), copy.deepcopy(S.TIERS), S.OFF_DY, S.S_GRID, S.S_DEEP, L.CERT_TOL)
        try:
            L.LEVERS["STR_label"] = dict(L.LEVERS["STR_label"], s=(0.95, 0.90, 0.85, 0.75))
            self.assertIn("n7_lib.LEVERS != card table", " ".join(BD.check_card7()["bad"]))
            L.LEVERS.update(copy.deepcopy(saved[0]))
            L.LEVERS["D_c20_m10"] = {"kind": "D", "cap": 20, "margin": 0.11}
            self.assertFalse(BD.check_card7()["ok"])
            L.LEVERS.update(copy.deepcopy(saved[0]))
            S.TIERS[:] = [t for t in saved[1] if t[0] != "T2_STR"]
            self.assertIn("n7_select.TIERS != card tiers", " ".join(BD.check_card7()["bad"]))
            S.TIERS[:] = copy.deepcopy(saved[1])
            S.OFF_DY = tuple(reversed(saved[2]))
            self.assertIn("n7_select.OFF_DY", " ".join(BD.check_card7()["bad"]))              # describe() would label the wrong parameter
            S.OFF_DY = saved[2]
            S.S_DEEP = (0.7, 0.6, 0.5, 0.4, 0.3, 0.25)
            self.assertIn("n7_select.S_DEEP", " ".join(BD.check_card7()["bad"]))
            S.S_DEEP = saved[4]
            L.CERT_TOL = dict(saved[5], ttc=True)                                              # a strict certificate in the generator's own copy
            self.assertIn("n7_lib.CERT_TOL", " ".join(BD.check_card7()["bad"]))
            L.CERT_TOL = saved[5]
            n6 = BD.B6.CARD6["N6T"]
            BD.B6.CARD6["N6T"] = dict(n6, ddc=False)
            self.assertIn("CARD6 N6T", " ".join(BD.check_card7()["bad"]))
            BD.B6.CARD6["N6T"] = n6
            self.assertTrue(BD.check_card7()["ok"])
        finally:
            L.LEVERS.clear()
            L.LEVERS.update(saved[0])
            S.TIERS[:] = saved[1]
            S.OFF_DY, S.S_GRID, S.S_DEEP, L.CERT_TOL = saved[2], saved[3], saved[4], saved[5]

    def test_the_card_text_carries_the_tables(self):
        if not os.path.exists(CARD_MD):
            self.skipTest("no card text at " + CARD_MD)
        txt = open(CARD_MD, encoding="utf-8").read()
        for tier in ("T0", "T1", "T2", "T3", "T4"):
            self.assertRegex(txt, rf"\|\s*{tier}\s*\|")
        for frag in ("cap 5, margin 0.05", "cap 20", "0.05, 0.10, 0.20", "s = 0.95, 0.90, 0.85, 0.80", "0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0", "0.2, 1.0, 2.0", "0.9, 0.8",
                     "s = 0.7, 0.6, 0.5", "floor 0.5", "K = 50", "N7-T3", "N7-X"):
            self.assertIn(frag, txt, frag)
        self.assertIn("PRIMARY", txt)
        self.assertIn("rests on", txt)

    def test_every_selector_mutation_moves_the_behavioural_signature_and_is_refused(self):
        for mut in BD.SEL_MUTATIONS:
            moved = False
            for arm in BD.ARMS:
                fn = BD.make_selector(arm, mut)
                if BD.selector_signature(fn) != BD.selector_signature(BD.make_selector(arm)):
                    moved = True
                    r = BD.check_selector(arm, fn)
                    self.assertFalse(r["ok"], (mut, arm))
                    self.assertTrue(any("behavioural signature" in b for b in r["bad"]), (mut, arm, r["bad"]))
            self.assertTrue(moved, f"{mut}: no selector's behaviour moved")

    def test_wrong_function_is_refused(self):
        fn = functools.partial(lambda *a, **k: None, spec=dict(BD.CERT), tiers=BD.CARD7["N7T"]["tiers"], floor=0.5, tie="earliest")
        self.assertFalse(BD.check_selector("N7T", fn)["ok"])

    def test_flag_audit_finds_a_flag_with_a_disabling_default_and_the_real_parser_has_none(self):
        self.assertEqual([f for f in BD.audit_flags(BD.build_parser()) if f["disabling_default"]], [])
        ap = BD.build_parser()
        ap.add_argument("--floor", type=float, default=None)                       # the regression arm: a floor flag that silently disables the T4 floor
        found = BD.audit_flags(ap)
        self.assertEqual([f["flag"] for f in found if f["disabling_default"] and not f["named_in_argv"]], ["--floor"])
        self.assertFalse(BD.check_card7(ap, {"--route"})["ok"])
        self.assertTrue(BD.check_card7(ap, {"--route", "--floor"})["ok"])          # named in the arm's argv: not a finding
        ap2 = BD.build_parser()
        ap2.add_argument("--tiers", default="")
        self.assertFalse(BD.check_card7(ap2, {"--route"})["ok"])
        # the chain's derive argv names no card lever, and the build has no lever flag at all
        argv = CH.build_argv(SimpleNamespace(run_dir="R"), "R", 0, 2)
        self.assertFalse(any(x in ("--floor", "--tiers", "--levers", "--cap", "--margin", "--human") for x in argv))


# --------------------------------------------------------------------------------------------------------------------------------------------------------------- SELECT
class TestSelect(unittest.TestCase):
    def pick(self, arm, name, mut="none"):
        ref, lev = BD.SCENARIOS[name]
        return ident(BD.make_selector(arm, mut)({"ref": ref, "levers": lev}))

    def test_card_semantics(self):
        P = self.pick
        for arm in ("N7T3", "N7T"):
            self.assertEqual(P(arm, "t0_beats_later_tiers"), ["T0_N6", "L1_c20_m05", 2])        # a certified T0 candidate keeps the TF-N6 plan whatever a later tier offers
            self.assertEqual(P(arm, "t1_when_t0_empty"), ["T1_L2", "L1_c20_m20", 2])            # max builder PDMS inside the first certified tier
            self.assertEqual(P(arm, "cap5_prefix"), ["T1_L2", "D_c20_m05", 6])                  # k = 6 is outside T0 (cap 5) and inside T1 (cap 20)
            self.assertEqual(P(arm, "tie_earliest_lever_order"), ["T1_L2", "D_c20_m05", 9])     # tie: the earlier LEVER of the tier's list (D m05 before D m10), not the smaller k
            self.assertEqual(P(arm, "tie_earliest_k"), ["T1_L2", "L1_c20_m20", 3])
            self.assertEqual(P(arm, "t2_stretch"), ["T2_STR", "STR_target", 1])
            self.assertEqual(P(arm, "t3_offsets"), ["T3_OFF", "STR_OFF", 3])
            self.assertEqual(P(arm, "ttc_c_free"), ["T0_N6", "D_c20_m05", 1])                   # TTC and C worse: still accepted
            self.assertEqual(P(arm, "nc_equal_below_one"), ["T0_N6", "D_c20_m05", 1])
            for none in ("nc_worse", "ddc_worse", "pdms_not_improved", "dac0_candidate", "nothing"):
                self.assertIsNone(P(arm, none), none)
        self.assertIsNone(P("N7T3", "t4_only"))                                                 # no T4 in N7-T3
        self.assertEqual(P("N7T", "t4_only"), ["T4_DEEP", "STR_deep", 1])
        self.assertIsNone(P("N7T", "t4_floor_below"))                                           # s = 0.4 is below the floor
        self.assertEqual(P("N7T", "t4_floor_at"), ["T4_DEEP", "STR_deep", 2])                   # s = 0.5 is at the floor (admissible); s = 0.4 (pdms 0.9) is not
        self.assertEqual(P("N7T", "t4_two_s"), ["T4_DEEP", "STR_deep", 1])
        self.assertIsNone(P("N7X", "t0_beats_later_tiers"))                                    # N7-X: a DAC-1 candidate is not taken
        zero = BD._cd(1, dac=0.0, pdms=0.0)
        self.assertIsNone(BD.make_selector("N7X")({"ref": BD.REF0, "levers": {"D_c20_m05": [zero]}}))      # a real DAC-0 plan scores PDMS 0 and cannot beat the DAC-0 reference's 0

    def test_mutations_change_choices(self):
        P = self.pick
        self.assertIsNone(P("N7T", "ttc_c_free", "strict"))
        self.assertEqual(P("N7T", "nc_worse", "nc_drop"), ["T0_N6", "D_c20_m05", 1])
        self.assertEqual(P("N7T", "ddc_worse", "ddc_drop"), ["T0_N6", "D_c20_m05", 1])
        self.assertEqual(P("N7T", "pdms_not_improved", "pdms_drop"), ["T0_N6", "D_c20_m05", 1])
        self.assertEqual(P("N7T", "t4_floor_below", "floor_off"), ["T4_DEEP", "STR_deep", 3])
        self.assertEqual(P("N7T3", "t4_only", "t4_in_t3"), ["T4_DEEP", "STR_deep", 1])
        self.assertEqual(P("N7T", "t0_beats_later_tiers", "tier_reverse"), ["T3_OFF", "OFF_label", 5])      # reversed tier order takes the late tier
        self.assertEqual(P("N7T", "t0_beats_later_tiers", "drop_T0"), ["T1_L2", "D_c20_m10", 3])
        self.assertEqual(P("N7T", "t1_when_t0_empty", "drop_T1"), ["T2_STR", "STR_label", 1])
        self.assertEqual(P("N7T", "tie_earliest_k", "tie_latest"), ["T1_L2", "L1_c20_m20", 5])
        self.assertIsNotNone(P("N7X", "dac0_candidate"))                                        # (the synthetic DAC-0 candidate with PDMS 0.2 > the reference 0)

    def test_ties_are_resolved_by_order_not_by_hash_or_dict_order(self):
        row = {"ref": BD.REF0, "levers": {"D_c20_m10": [BD._cd(4, pdms=0.5)], "D_c20_m05": [BD._cd(9, pdms=0.5)]}}
        self.assertEqual(ident(BD.make_selector("N7T")(row)), ["T1_L2", "D_c20_m05", 9])
        row2 = {"ref": BD.REF0, "levers": dict(reversed(list(row["levers"].items())))}              # dict insertion order must not matter
        self.assertEqual(ident(BD.make_selector("N7T")(row2)), ["T1_L2", "D_c20_m05", 9])


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------ REAL
def load_dev_rows():
    """the development's stored REAL navtrain candidate sets (n7_gen/raw/candidates_navtrain.tar.xz), read in memory (no extraction) -> (rows17, rows73), or None"""
    if not os.path.exists(DEV_TAR):
        return None
    r17, r73 = {}, {}
    with tarfile.open(DEV_TAR) as tf:
        for m in tf.getmembers():
            if not (m.isfile() and m.name.endswith(".jsonl")):
                continue
            dst = r73 if "chp73" in m.name else r17
            for ln in tf.extractfile(m):
                if ln.strip():
                    r = json.loads(ln)                                     # the same merge as n7_analyze.load: one row per token, levers dicts updated
                    if r["token"] in dst:
                        dst[r["token"]]["levers"].update(r["levers"])
                    else:
                        dst[r["token"]] = r
    return r17, r73


_DEV = {}


def dev_rows():
    if "v" not in _DEV:
        _DEV["v"] = load_dev_rows()
    return _DEV["v"]


class TestReal(unittest.TestCase):
    def setUp(self):
        if dev_rows() is None:
            self.skipTest("no development candidate tarball at " + DEV_TAR)
        self.r17, self.r73 = dev_rows()
        self.sel17 = json.load(open(DEV_SEL17, encoding="utf-8"))["per_token"]
        self.sel73 = json.load(open(DEV_SEL73, encoding="utf-8"))["per_token"]

    def test_the_build_selector_equals_the_development_selection_on_the_90_real_tokens(self):
        t, tt3, x = BD.make_selector("N7T"), BD.make_selector("N7T3"), BD.make_selector("N7X")
        n = 0
        for rows, sel in ((self.r17, self.sel17), (self.r73, self.sel73)):
            self.assertEqual(set(rows), set(sel))
            for tok, r in rows.items():
                row = {"ref": r["ref"], "levers": r["levers"]}
                want = sel[tok]
                got = t(row)
                self.assertEqual(ident(got), None if want is None else [want["tier"], want["lever"], want["k"]], tok)
                self.assertEqual(None if got is None else got["param"], None if want is None else want["param"], tok)
                g3 = tt3(row)
                self.assertEqual(ident(g3), None if (want is None or want["tier"] == "T4_DEEP") else [want["tier"], want["lever"], want["k"]], tok)      # N7-T3: T0..T3 only
                self.assertIsNone(x(row), tok)                                                                                                   # N7-X changes nothing (DAC-0 plans score PDMS 0)
                self.assertEqual(BD.dev_cascade(r, "N7T"), None if got is None else tuple(ident(got)))                                            # the dev cascade agrees
                self.assertEqual(BD.dev_cascade(r, "N7T3"), None if g3 is None else tuple(ident(g3)))
                self.assertEqual(BD.order_violations(r, {"N7T3": None if g3 is None else dict(g3, plan=list(map(list, g3["plan"]))), "N7T": None if got is None else
                                                         dict(got, plan=list(map(list, got["plan"])))}), [], tok)
                n += 1
        self.assertEqual(n, 90)

    def test_tier_counts_and_a_tier_list_without_a_tier_drops_its_tokens(self):
        counts = {}
        by = {}
        for tok, r in self.r17.items():
            c = BD.make_selector("N7T")({"ref": r["ref"], "levers": r["levers"]})
            key = "NONE" if c is None else c["tier"]
            counts[key] = counts.get(key, 0) + 1
            by.setdefault(key, []).append(tok)
        self.assertEqual(counts, {"T0_N6": 2, "T1_L2": 7, "T2_STR": 2, "T3_OFF": 2, "T4_DEEP": 3, "NONE": 1})        # DEV_N7.md section 3
        for tier in ("T0_N6", "T1_L2", "T2_STR", "T3_OFF", "T4_DEEP"):
            full = BD.make_selector("N7T")
            tl = [t for t in full.keywords["tiers"] if t[0] != tier]
            sel = functools.partial(BD.card_cascade, spec=full.keywords["spec"], tiers=tl, floor=0.5, tie="earliest")
            for tok in by[tier]:
                r = self.r17[tok]
                c = sel({"ref": r["ref"], "levers": r["levers"]})
                self.assertTrue(c is None or c["tier"] != tier, (tier, tok))                            # the token is dropped from the tier
        m2 = BD.make_selector("N7T", "drop_T1")                                                        # the mutated tier list changes what the T1 tokens get
        full = BD.make_selector("N7T")
        self.assertTrue(any(ident(m2({"ref": self.r17[t]["ref"], "levers": self.r17[t]["levers"]})) != ident(full({"ref": self.r17[t]["ref"], "levers": self.r17[t]["levers"]})) for t in by["T1_L2"]))
        # without T4 (N7-T3) exactly the T4 tokens and the unrescued one stay unchanged: 13 of 17 certified
        n3 = sum(1 for r in self.r17.values() if BD.make_selector("N7T3")({"ref": r["ref"], "levers": r["levers"]}) is not None)
        self.assertEqual(n3, 13)
        self.assertEqual(sum(1 for r in self.r73.values() if BD.make_selector("N7T")({"ref": r["ref"], "levers": r["levers"]})["tier"] == "T0_N6"), 73)   # 73 of 73 stay on the TF-N6 tier

    def test_the_t4_floor_and_the_cap5_prefix_on_real_candidates(self):
        r = self.r17["ff81f481d2ad5270"]                                                                # chosen at s = 0.5 (the floor)
        self.assertEqual(ident(BD.make_selector("N7T")({"ref": r["ref"], "levers": r["levers"]})), ["T4_DEEP", "STR_deep", 2])
        self.assertIsNone(BD.make_selector("N7T3")({"ref": r["ref"], "levers": r["levers"]}))
        r = self.r17["69b54b1998f05c13"]
        c = BD.make_selector("N7T")({"ref": r["ref"], "levers": r["levers"]})
        self.assertEqual(c["tier"], "T0_N6")
        self.assertLessEqual(c["k"], 5)

    @unittest.skipIf(os.environ.get("N7_SKIP_REAL"), "N7_SKIP_REAL")
    def test_every_tier_executes_on_a_real_navtrain_token_through_the_builds_own_code_path(self):
        fx = os.path.join(HERE, "fixture_navtrain_tiers.json")
        r = BD.cmd_reach_real_instances(fx)
        self.assertTrue(r["ok"], json.dumps(r)[:1500])
        self.assertEqual(r["tiers_executed"], list(BD.TIER_NAMES))
        for tier, v in r["instances"].items():
            self.assertTrue(v["match"] and v["plan_sha256_match"] and v["dropped_tier_removes_it"] and v["t3_ok"] and v["x_none"] and v["base_dac0"], (tier, v))
        self.assertTrue(r["instances"]["T0_N6"]["deterministic"])

    def test_c_reach_goes_red_when_a_tier_is_not_executed_or_the_fixture_tier_is_wrong(self):
        d = tempfile.mkdtemp(dir=TMP)
        for tiers in (list(BD.TIER_NAMES)[:4], list(BD.TIER_NAMES)[1:]):
            json.dump({"ok": True, "arms": [], "real": {"ok": True, "tiers_executed": tiers, "instances": {}}}, open(f"{d}/reach.json", "w"))
            self.assertFalse(MergeBase.reach_pass(f"{d}/reach.json"))
        json.dump({"ok": True, "arms": [], "real": {"ok": True, "tiers_executed": list(BD.TIER_NAMES), "instances": {}}}, open(f"{d}/reach.json", "w"))
        self.assertTrue(MergeBase.reach_pass(f"{d}/reach.json"))


class MergeBase:
    @staticmethod
    def reach_pass(path):
        rj = json.load(open(path))
        real = rj.get("real") or {}
        return bool(rj.get("ok") and real.get("ok") and real.get("tiers_executed") == list(BD.TIER_NAMES))


# --------------------------------------------------------------------------------------------------------------------------------------------------------------- CAPTURE
def _fx():
    return BD.B5.fakes().T


BASE_FAIL = {"dac": 0.0, "nc": 1.0, "ddc": 1.0, "ttc": 1.0, "c": 1.0, "ep": 0.0, "pdms": 0.0}


def var_cx(T, corridor_args, m1=None):
    """FakeCx whose DAC-1 plans carry the metrics m1 and whose DAC-0 plans score PDMS 0 (as the real builder does)"""
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


class TestCapture(unittest.TestCase):
    def setUp(self):
        self.T = _fx()
        self.t20 = self.T.teacher(lambda t: np.where(t < 2.0, 0.30, -0.16))
        self.sels = BD.make_objects()
        self.fd, self.fl, self.sel6 = BD.B4.make_repair(BD.D_ARM), BD.B5.make_repair(BD.B5.CARD[BD.L1_ARM]), BD.B6.make_selector("N6T")

    def gen(self, m1, corridor=(0.50, 0.34)):
        cx = var_cx(self.T, corridor, m1)
        ref = cx.score([self.t20])[0][0]
        return cx, ref, BD.gen_levers(cx, self.t20, np.array(self.t20), ref)

    def test_generator_runs_every_lever_and_is_deterministic(self):
        cx, ref, lev = self.gen({"ep": 0.9, "pdms": 0.9})
        self.assertEqual(set(lev), set(BD.RUN_LEVERS))
        self.assertEqual(len(lev["OFF_label"]), 14 * 3)
        self.assertEqual(len(lev["STR_OFF"]), 2 * 14 * 3)
        self.assertEqual((len(lev["STR_label"]), len(lev["STR_target"]), len(lev["STR_deep"])), (4, 4, 6))
        self.assertTrue(np.array_equal(lev["D_c20_m05"][0]["plan"], np.asarray(self.t20)))                   # iterate 0 = the base label
        lev2 = BD.gen_levers(cx, self.t20, np.array(self.t20), ref)
        self.assertTrue(BD.levers_equal(lev, lev2))
        lev2["OFF_label"][5]["plan"] = lev2["OFF_label"][5]["plan"] + 1e-9                                  # C-DET sees a 1e-9 edit
        self.assertFalse(BD.levers_equal(lev, lev2))
        lev3 = BD.gen_levers(cx, self.t20, np.array(self.t20), ref)
        lev3["D_c20_m10"][2]["m"] = dict(lev3["D_c20_m10"][2]["m"], pdms=0.123)
        self.assertFalse(BD.levers_equal(lev, lev3))

    def test_the_cascade_takes_t0_and_equals_the_live_n6_selection(self):
        cx, ref, lev = self.gen({"ttc": 0.0, "c": 0.0, "ep": 0.9, "pdms": 0.5})                              # TTC / C worse: the full certificate refuses, the tolerance takes it
        row = {"ref": ref, "levers": lev}
        c = self.sels["N7T3"](row)
        live = BD.n6_live(cx, self.t20, ref, self.fd, self.fl, self.sel6)
        self.assertIsNotNone(live["chosen"])
        self.assertEqual(c["tier"], "T0_N6")
        self.assertEqual((("D" if c["lever"].startswith("D_") else "L1"), c["k"]), (live["chosen"]["src"], live["chosen"]["k"]))
        self.assertEqual(np.asarray(c["plan"]).tolist(), live["chosen"]["plan"])
        self.assertEqual(c["m"], live["chosen"]["m"])
        self.assertEqual(live["repl"], {"D": False, "L1": False})                                           # the banked certificate (full) returns nothing; the tolerance does
        self.assertIsNone(BD.make_selector("N7T3", "strict")(row))                                          # strict (TTC/C required) takes none

    def test_a_token_outside_t0_reaches_a_later_tier_through_the_real_loops(self):
        cx, ref, lev = self.gen({"ep": 0.9, "pdms": 0.9}, corridor=(0.05, 0.05))                              # a corridor the repairs cannot satisfy
        row = {"ref": ref, "levers": lev}
        t = self.sels["N7T"](row)
        t3 = self.sels["N7T3"](row)
        live = BD.n6_live(cx, self.t20, ref, self.fd, self.fl, self.sel6)
        self.assertTrue(t is None or t["tier"] != "T0_N6" or live["chosen"] is not None)
        if t is not None and t["tier"] != "T0_N6":
            self.assertIsNone(live["chosen"])                                                               # T0 == N6: no N6 selection where T0 certifies nothing
        if t3 is not None:
            self.assertEqual(ident(t3), ident(t))

    def test_nc_ddc_and_pdms_guards_in_the_generated_candidates(self):
        for m1, mut in (({"nc": 0.0, "pdms": 0.5}, "nc_drop"), ({"ddc": 0.5, "pdms": 0.5}, "ddc_drop"), ({"pdms": 0.0}, "pdms_drop")):
            cx, ref, lev = self.gen(dict({"ep": 0.9, "pdms": 0.5}, **m1))
            row = {"ref": ref, "levers": lev}
            self.assertIsNone(self.sels["N7T"](row), m1)
            self.assertIsNotNone(BD.make_selector("N7T", mut)(row), mut)

    def test_token_row_records_both_runs_the_arms_and_the_live_n6(self):
        cx = var_cx(self.T, (0.50, 0.34), {"ttc": 0.0, "c": 0.0, "ep": 0.9, "pdms": 0.5})
        ref = cx.score([self.t20])[0][0]
        rec = BD.token_row(cx, self.t20, np.array(self.t20), ref, self.sels, self.fd, self.fl, self.sel6)
        self.assertTrue(rec["det_equal"])
        self.assertEqual(set(rec["levers"]), set(BD.RUN_LEVERS))
        self.assertEqual(set(rec["chosen"]), set(BD.ARMS))
        self.assertIsNone(rec["chosen"]["N7X"])
        ch = rec["chosen"]["N7T3"]
        self.assertEqual(ch["m_recheck"], ch["m"])
        json.dumps(rec)                                                                                    # JSON-able
        off = BD.token_row(cx, self.t20, np.array(self.t20), ref, self.sels, self.fd, self.fl, self.sel6, "levers_off")
        self.assertFalse(off["det_equal"])
        self.assertTrue(all(v is None for v in off["chosen"].values()))

    def test_build_cli_refuses_mutation_without_smoke_flag(self):
        with self.assertRaises(SystemExit) as c:
            BD.cmd_derive(SimpleNamespace(mutate="strict", smoke_allow_mutation=False))
        self.assertIn("REFUSING", str(c.exception))

    def test_chain_argv_names_no_mutation(self):
        argv = CH.build_argv(SimpleNamespace(run_dir="R"), "R", 0, 2)
        self.assertFalse(any("mutate" in x or "smoke" in x for x in argv))
        self.assertEqual(argv[argv.index("--route") + 1], "cache")
        self.assertIn("n7_nt_build.py", " ".join(argv))


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------- UNCH
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
        self.app = {"N7T3": ["t02", "t03"], "N7T": ["t02", "t03", "t05"], "N7X": []}
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
            np.savez(f"{self.d}/n7_plans_{arm}.npz", tokens=self.tok, logs=self.tok, plans20=P, arm=np.array(arm))
            np.savez(f"{self.d}/{BD.SEAM_LABEL[arm]}.npz", token=self.tok, poses=S, fingerprint=np.array(["f"] * 8), sampling=np.array([8.0, 0.5]), arm=np.array("y"))

    def chk(self, n_res=4, expect=4):
        return BD.check_unch(self.d, f"{self.d}/n5a_plans.npz", f"{self.d}/n5a_seam.npz", self.app, n_res, expect)

    def test_green(self):
        r = self.chk()
        self.assertTrue(r["pass"], r)
        self.assertTrue(r["arms"]["N7X"]["bit_identical_to_n5a"])

    def test_2e6_perturbation_of_one_unchanged_plan_is_red(self):
        def m(P, S):
            P[0, 3, 1] += 2e-6
        self.write(("N7T3", m))
        r = self.chk()
        self.assertFalse(r["pass"])
        self.assertEqual(r["arms"]["N7T3"]["unflagged_changed"], ["t00"])

    def test_applied_to_all_tokens_is_red(self):
        def m(P, S):
            P += 1e-12
        self.write(("N7T", m))
        self.assertFalse(self.chk()["pass"])

    def test_flagged_but_unchanged_is_red(self):
        self.app["N7T"] = ["t02", "t06"]
        self.assertFalse(self.chk()["pass"])

    def test_seam_changed_without_plan_is_red(self):
        def m(P, S):
            S[5, 1, 1] += 1e-3
        self.write(("N7T3", m))
        self.assertFalse(self.chk()["pass"])

    def test_n7x_must_be_bit_identical_to_n5a(self):
        def m(P, S):
            S[4, 0, 0] += np.float32(1e-6)
        self.write(("N7X", m))
        r = self.chk()
        self.assertFalse(r["pass"])
        self.assertFalse(r["arms"]["N7X"]["bit_identical_to_n5a"])

    def test_residual_count_must_equal_the_card(self):
        self.assertFalse(self.chk(n_res=21, expect=20)["pass"])
        self.assertFalse(self.chk(n_res=19, expect=20)["pass"])


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------- MERGE
class TestMerge(unittest.TestCase):
    """the build merge on 8 synthetic tokens (to_navsim is the real function; nothing touches the harness). t01: a T0 candidate (the N6-T token; later tiers offer higher PDMS);
    t02: T1 (T2 and T3 offer more); t03: T2; t04: T3 (T4 offers more); t05: only T4 certifies (below-floor candidate present); t06: DAC-1 candidates fail a clause; t00, t07 not residual"""

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
        np.savez(f"{self.d}/n5a_seam.npz", token=np.array(self.tok), fingerprint=np.array(["f"] * n), poses=p8, sampling=np.array([8.0, 0.5]), arm=np.array("TEACHER_x_refe_tfn5_n5a"))
        self.resid = ["t01", "t02", "t03", "t04", "t05", "t06"]
        self.write_csv()
        json.dump({"unrepaired_l1": {"N5A": self.resid}}, open(f"{self.d}/n5a_report.json", "w"))
        json.dump({"d_returns_plan": [], "l1_returns_plan": []}, open(f"{self.d}/replay.json", "w"))
        self.write_reach()
        self.levers = {t: self.levers_of(t) for t in self.resid}
        self.write_n6()
        self.rows = self.make_rows()

    def write_reach(self, tiers=None):
        json.dump({"ok": True, "arms": [{"arm": a, "ok": True, "what": a} for a in ("D", "L1")],
                   "real": {"ok": True, "tiers_executed": list(tiers or BD.TIER_NAMES), "instances": {x: {"ok": True} for x in BD.TIER_NAMES}}}, open(f"{self.d}/reach.json", "w"))

    def write_csv(self):
        with open(f"{self.d}/n5a.csv", "w", encoding="utf-8", newline="") as fo:
            w = csv.writer(fo)
            w.writerow(COLS)
            for i, tk in enumerate(self.tok):
                r = dict(OK5, dac=0.0 if tk in self.resid else 1.0, pdms=0.0 if tk in self.resid else 0.9)
                w.writerow([i, tk, True, r["nc"], r["dac"], r["ep"], r["ttc"], r["c"], r["ddc"], r["pdms"]])

    @staticmethod
    def cand(plan, k, **m):
        base = dict(BD.REF0, dac=1.0, ep=0.5, pdms=0.5)
        base.update(m)
        return {"k": k, "plan": plan.tolist(), "m": base}

    def levers_of(self, tk):
        p = self.n5a[self.tok.index(tk)]
        sh = lambda d: p + np.array([0, d, 0.0])[None, :]                                                  # noqa: E731
        base = self.cand(p, 0, dac=0.0, pdms=0.0, ep=0.0)
        lv = {n: [] for n in BD.RUN_LEVERS}
        lv["D_c20_m05"] = [base]
        lv["L1_c20_m05"] = [dict(base)]
        if tk == "t01":
            lv["D_c20_m05"] += [self.cand(sh(0.1), 1, ttc=0.5, c=0.5, pdms=0.9),                          # the N6-T pick (TTC/C free)
                                self.cand(sh(0.11), 3, nc=0.0, pdms=0.95),                                 # NC worse: nc_drop takes it
                                self.cand(sh(0.12), 4, ddc=0.5, pdms=0.97)]                                # DDC worse: ddc_drop takes it
            lv["L1_c20_m05"] += [self.cand(sh(0.2), 1, ttc=0.5, c=0.5, pdms=0.9)]                          # a tie: tie_latest takes it
            lv["D_c20_m10"] = [self.cand(sh(0.3), 3, pdms=0.99)]                                           # T1 offers more: it must not win (T0 first)
            lv["OFF_label"] = [self.cand(sh(0.4), 5, pdms=1.0)]
        if tk == "t02":
            lv["L1_c20_m10"] = [dict(base), self.cand(sh(0.5), 2, pdms=0.6)]
            lv["STR_label"] = [self.cand(sh(0.55), 1, pdms=0.7)]
            lv["OFF_label"] = [self.cand(sh(0.6), 4, pdms=0.9)]
        if tk == "t03":
            lv["STR_label"] = [self.cand(sh(0.5), 1, pdms=0.7)]
            lv["STR_target"] = [self.cand(sh(0.55), 2, pdms=0.75)]
            lv["OFF_label"] = [self.cand(sh(0.6), 4, pdms=0.9)]
        if tk == "t04":
            lv["OFF_label"] = [self.cand(sh(0.6), 8, pdms=0.6)]
            lv["STR_OFF"] = [self.cand(sh(0.65), 3, pdms=0.8)]
            lv["STR_deep"] = [self.cand(sh(0.7), 1, pdms=0.99)]
        if tk == "t05":
            lv["STR_deep"] = [self.cand(sh(0.7), 1, pdms=0.6), self.cand(sh(0.8), 3, pdms=0.9)]            # s = 0.6 admissible; s = 0.4 below the floor
        if tk == "t06":
            lv["D_c20_m05"] += [self.cand(sh(0.9), 1, pdms=0.0, ep=0.0)]                                    # DAC 1 but builder PDMS not improved: pdms_drop takes it
            lv["STR_deep"] = [self.cand(sh(1.0), 3, pdms=0.9)]                                              # below the floor
        return lv

    def chosen_for(self, tk, mut=None, mut_arm=None):
        row = {"ref": dict(BD.REF0), "levers": self.levers[tk]}
        ch = {}
        for arm in BD.ARMS:
            fn = BD.make_selector(arm, mut if (mut and (mut_arm in (None, arm))) else "none")
            c = fn(row)
            ch[arm] = None if c is None else {"tier": c["tier"], "lever": c["lever"], "k": c["k"], "param": c["param"], "plan": c["plan"], "m": c["m"], "m_recheck": dict(c["m"])}
        return ch

    def write_n6(self, same=True):
        """the banked TF-N6 artifacts for this synthetic set: N6-T certified t01 only"""
        c = [x for x in self.levers["t01"]["D_c20_m05"] if x["k"] == 1][0]
        P = self.n5a.copy()
        P[1] = np.asarray(c["plan"], float)
        np.savez(f"{self.d}/n6t_plans.npz", tokens=np.array(self.tok), logs=np.array(self.tok), plans20=P, arm=np.array("N6T"))
        cm = dict(c["m"], src="D", k=1)
        json.dump({"repaired": {"N6T": ["t01"]}, "residual_table": {"t01": {"resolved_by": ["D", 1, "teacher"], "candidates": [cm]}},
                   "files": {"n6_plans_N6T.npz": sha(f"{self.d}/n6t_plans.npz")}}, open(f"{self.d}/n6_report.json", "w"))

    def make_rows(self, mut=None, mut_arm=None):
        rows = []
        for i, tk in enumerate(self.tok):
            if tk not in self.resid:
                rows.append({"log": f"l{i % 2}", "token": tk, "ref": dict(OK5), "residual": False, "cderiv_maxdev_m": 0.0, "label_ego": "db", "t_s": 0.1})
                continue
            ch = self.chosen_for(tk, mut, mut_arm)
            live = None
            if tk == "t01":
                c = [x for x in self.levers[tk]["D_c20_m05"] if x["k"] == 1][0]
                live = {"src": "D", "k": 1, "plan": c["plan"], "m": c["m"]}
            rows.append({"log": f"l{i % 2}", "token": tk, "ref": dict(BD.REF0), "residual": True, "cderiv_maxdev_m": 0.0, "n5a_equals_n2r_plan": True, "levers": self.levers[tk], "chosen": ch,
                         "n6_live": {"repl": {"D": False, "L1": False}, "chosen": live, "n_cands": 3}, "det_equal": True, "label_ego": "db", "t_s": 0.2})
        return rows

    def merge(self, **over):
        with open(f"{self.d}/rows.jsonl", "w", encoding="utf-8") as fo:
            for r in self.rows:
                fo.write(json.dumps(r) + "\n")
        out = f"{self.d}/out"
        kw = dict(rows=[f"{self.d}/rows.jsonl"], n5a_plans=f"{self.d}/n5a_plans.npz", n5a_seam=f"{self.d}/n5a_seam.npz", n5a_report=f"{self.d}/n5a_report.json",
                  expect_residual_json="", replay_expect_json=f"{self.d}/replay.json", harness_csv_n5a=f"{self.d}/n5a.csv", reach_json=f"{self.d}/reach.json", out_dir=out,
                  expect_residual=6, n6_report=f"{self.d}/n6_report.json", n6t_plans=f"{self.d}/n6t_plans.npz")
        kw.update(over)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = BD.cmd_merge(SimpleNamespace(**kw))
        return rc, json.load(open(f"{out}/n7_build_report.json")), out

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
        self.assertEqual(rep["repaired"]["N7T3"], ["t01", "t02", "t03", "t04"])
        self.assertEqual(rep["repaired"]["N7T"], ["t01", "t02", "t03", "t04", "t05"])
        self.assertEqual(rep["repaired"]["N7X"], [])
        self.assertEqual(rep["tier_of"]["N7T"], {"t01": "T0_N6", "t02": "T1_L2", "t03": "T2_STR", "t04": "T3_OFF", "t05": "T4_DEEP"})
        self.assertEqual(rep["counts"]["N7T3"]["by_tier"], {"T0_N6": 1, "T1_L2": 1, "T2_STR": 1, "T3_OFF": 1, "T4_DEEP": 0})
        self.assertEqual(rep["counts"]["N7T"]["n_teacher_tol"], 5)
        self.assertEqual(rep["residual_table"]["t03"]["resolved_by"][:3], ["T2_STR", "STR_target", 2])
        self.assertEqual(rep["residual_table"]["t04"]["resolved_by"][:3], ["T3_OFF", "STR_OFF", 3])
        self.assertEqual(rep["residual_table"]["t01"]["resolved_by"][:3], ["T0_N6", "D_c20_m05", 1])
        self.assertIn("DAC-1 candidates exist but fail", rep["residual_table"]["t06"]["blocking_class"])
        self.assertIn("below_floor", rep["residual_table"]["t06"]["blocking_class"])
        self.assertIn("T0-T3", rep["residual_table"]["t05"]["blocking_class_n7t3"])
        self.assertTrue(rep["controls"]["C-N6SAME"]["banked_plans_checked"])
        for c in ("C-COVB", "C-DACMAP6", "C-DERIV", "C-LABELEGO", "C-SET", "C-REPLAY", "C-DET", "C-SELECT", "C-CERT-B7", "C-ORDER", "C-X0", "C-SEAMFN", "C-UNCH", "C-N6SAME", "C-REACH"):
            self.assertIn(c, rep["controls"])
        s, sb = np.load(f"{out}/refe_tfn7_n7t3.npz"), np.load(f"{self.d}/n5a_seam.npz")
        self.assertEqual([i for i in range(8) if not np.array_equal(s["poses"][i], sb["poses"][i])], [1, 2, 3, 4])
        st = np.load(f"{out}/refe_tfn7_n7t.npz")
        self.assertEqual([i for i in range(8) if not np.array_equal(st["poses"][i], sb["poses"][i])], [1, 2, 3, 4, 5])
        self.assertTrue(str(s["arm"]).endswith("refe_tfn7_n7t3"))
        sx = np.load(f"{out}/refe_tfn7_n7x.npz")
        self.assertTrue(np.array_equal(sx["poses"], sb["poses"]))                                          # N7-X: bit-identical to N5-A
        want = self.ND.to_navsim(np.asarray(self.rows[1]["chosen"]["N7T3"]["plan"])).astype(np.float32)
        self.assertTrue(np.array_equal(s["poses"][1], want))
        zt, z3 = np.load(f"{out}/n7_plans_N7T.npz"), np.load(f"{out}/n7_plans_N7T3.npz")
        self.assertTrue(np.array_equal(zt["plans20"][1], z3["plans20"][1]))                                 # the N6-T token is the same plan in both arms
        self.assertTrue(np.array_equal(zt["plans20"][1], np.load(f"{self.d}/n6t_plans.npz")["plans20"][1]))

    def test_npz_read_once_not_per_token(self):
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

    # --- the selector the derive RAN differs from the card; the merge must go RED
    def test_mutation_tolerance_off_strict_is_red(self):
        self.red("C-SELECT", "strict", also=("C-N6SAME",))                    # strict refuses the TTC/C-worse T0 candidate of t01 and takes a later tier: the TF-N6 plan is lost

    def test_mutation_nc_guard_dropped_is_red(self):
        self.red("C-SELECT", "nc_drop", also=("C-CERT-B7", "C-ORDER"))

    def test_mutation_ddc_guard_dropped_is_red(self):
        self.red("C-SELECT", "ddc_drop", also=("C-CERT-B7", "C-ORDER"))

    def test_mutation_pdms_clause_dropped_is_red(self):
        self.red("C-SELECT", "pdms_drop", also=("C-CERT-B7", "C-ORDER"))

    def test_mutation_dac_requirement_inverted_is_red(self):
        self.red("C-SELECT", "dac_inverted", also=("C-X0",))
        self.red("C-X0", "dac_inverted", mut_arm="N7X")                        # only the X arm's certificate flipped to require DAC = 1: it now changes tokens

    def test_mutation_tie_rule_latest_is_red(self):
        self.red("C-SELECT", "tie_latest", also=("C-ORDER",))

    def test_mutation_tier_order_reversed_is_red(self):
        self.red("C-SELECT", "tier_reverse", also=("C-ORDER",))

    def test_mutation_t4_floor_off_is_red(self):
        self.red("C-SELECT", "floor_off", also=("C-ORDER",))

    def test_mutation_t4_inside_n7t3_is_red(self):
        self.red("C-SELECT", "t4_in_t3", mut_arm="N7T3", also=("C-ORDER",))

    def test_mutation_a_tier_dropped_is_red(self):
        self.red("C-SELECT", "drop_T0", also=("C-N6SAME",))                    # t01 leaves the TF-N6 plan: C-N6SAME sees it
        self.red("C-SELECT", "drop_T1")
        self.red("C-SELECT", "drop_T0", mut_arm="N7T")

    def test_the_n6_pick_is_never_replaced_by_a_later_tier(self):
        row = {"ref": dict(BD.REF0), "levers": self.levers["t01"]}
        c = BD.make_selector("N7T")(row)
        self.assertEqual((c["tier"], c["lever"], c["k"]), ("T0_N6", "D_c20_m05", 1))                          # although D_c20_m10 and OFF_label offer PDMS 0.99 and 1.0
        self.assertEqual(self.rows[1]["chosen"]["N7T3"]["plan"], self.rows[1]["chosen"]["N7T"]["plan"])

    def test_recheck_differs_is_red(self):
        self.rows[1]["chosen"]["N7T3"]["m_recheck"] = dict(self.rows[1]["chosen"]["N7T3"]["m"], ttc=0.4)
        rc, rep, _ = self.merge()
        self.assertFalse(rep["controls"]["C-CERT-B7"]["pass"])

    def test_certificate_dac_off_is_red_at_c_cert_b7(self):
        for k in ("m", "m_recheck"):
            self.rows[1]["chosen"]["N7T3"][k] = dict(self.rows[1]["chosen"]["N7T3"][k], dac=0.0)
        rc, rep, _ = self.merge()
        self.assertFalse(rep["controls"]["C-CERT-B7"]["pass"])

    def test_certificate_changed_from_card6_is_red_at_c_cert_b7(self):
        n6 = BD.B6.CARD6["N6T"]
        BD.B6.CARD6["N6T"] = dict(n6, ttc=True)
        try:
            rc, rep, _ = self.merge()
        finally:
            BD.B6.CARD6["N6T"] = n6
        self.assertFalse(rep["controls"]["C-CERT-B7"]["pass"])
        self.assertFalse(rep["controls"]["C-CERT-B7"]["certificate_equals_card6_n6t"])

    def test_non_deterministic_generator_is_red(self):
        self.rows[1]["det_equal"] = False
        rc, rep, _ = self.merge()
        self.assertFalse(rep["controls"]["C-DET"]["pass"])

    def test_levers_not_run_is_red(self):
        for r in self.rows:
            if r.get("residual"):
                r["levers"] = {}
                r["chosen"] = {a: None for a in BD.ARMS}
        rc, rep, _ = self.merge()
        self.assertFalse(rep["controls"]["C-DET"]["pass"])
        self.assertFalse(rep["controls"]["C-REPLAY"]["pass"])

    def test_wrong_dac_map_is_red(self):
        self.rows[7]["ref"]["dac"] = 0.0
        rc, rep, _ = self.merge()
        self.assertFalse(rep["controls"]["C-DACMAP6"]["pass"])

    def test_cache_ego_label_is_red(self):
        self.rows[2]["label_ego"] = "cache"
        rc, rep, _ = self.merge()
        self.assertFalse(rep["controls"]["C-LABELEGO"]["pass"])

    def test_missing_row_and_changed_residual_set_are_red(self):
        self.rows.pop(4)
        rc, rep, _ = self.merge()
        self.assertFalse(rep["controls"]["C-COVB"]["pass"])
        self.setUp()
        self.rows[3]["ref"]["dac"] = 1.0
        rc, rep, _ = self.merge()
        self.assertFalse(rep["controls"]["C-SET"]["pass"])

    def test_extra_residual_token_is_red_at_c_set(self):
        self.rows[7]["ref"]["dac"] = 0.0
        self.rows[7].update(residual=True, levers=self.levers["t01"], chosen={a: None for a in BD.ARMS}, det_equal=True, n5a_equals_n2r_plan=True,
                            n6_live={"repl": {"D": False, "L1": False}, "chosen": None, "n_cands": 1})
        rc, rep, _ = self.merge()
        self.assertFalse(rep["controls"]["C-SET"]["pass"])

    def test_replay_violation_is_red(self):
        self.rows[1]["n6_live"]["repl"]["L1"] = True              # the banked L1 returned no plan on this token; the re-run did
        rc, rep, _ = self.merge()
        self.assertFalse(rep["controls"]["C-REPLAY"]["pass"])
        self.setUp()
        self.rows[2]["levers"]["D_c20_m05"][0]["plan"][0][0] += 1e-6           # iterate 0 is not the banked plan
        rc, rep, _ = self.merge()
        self.assertFalse(rep["controls"]["C-REPLAY"]["pass"])
        self.setUp()
        rc, rep, _ = self.merge(replay_expect_json="")
        self.assertFalse(rep["controls"]["C-REPLAY"]["pass"])

    def test_reach_json_missing_failed_or_missing_a_tier_is_red(self):
        json.dump({"ok": False, "arms": [], "real": {"ok": False, "instances": {}}}, open(f"{self.d}/reach.json", "w"))
        rc, rep, _ = self.merge()
        self.assertFalse(rep["controls"]["C-REACH"]["pass"])
        rc, rep, _ = self.merge(reach_json="")
        self.assertFalse(rep["controls"]["C-REACH"]["pass"])
        self.write_reach(tiers=BD.TIER_NAMES[:4])                              # T4 never executed on a real token
        rc, rep, _ = self.merge()
        self.assertFalse(rep["controls"]["C-REACH"]["pass"])
        self.write_reach()
        rc, rep, _ = self.merge()
        self.assertTrue(rep["controls"]["C-REACH"]["pass"])

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

    # --- C-N6SAME: live, banked report, banked plans
    def test_n6same_live_selection_differs_is_red(self):
        self.rows[1]["n6_live"]["chosen"] = None                              # the live N6 pipeline certified nothing where T0 certifies
        rc, rep, _ = self.merge()
        self.assertFalse(rep["controls"]["C-N6SAME"]["pass"])
        self.setUp()
        self.rows[1]["n6_live"]["chosen"]["k"] = 3
        rc, rep, _ = self.merge()
        self.assertFalse(rep["controls"]["C-N6SAME"]["pass"])
        self.setUp()
        self.rows[2]["n6_live"]["chosen"] = dict(self.rows[1]["n6_live"]["chosen"])      # N6 certified a token that T0 does not
        rc, rep, _ = self.merge()
        self.assertFalse(rep["controls"]["C-N6SAME"]["pass"])

    def test_n6same_banked_report_and_plans_mismatch_is_red(self):
        n6 = json.load(open(f"{self.d}/n6_report.json"))
        n6["repaired"]["N6T"] = ["t01", "t02"]                                  # the banked N6 report names a token the cascade handles in T2
        n6["residual_table"]["t02"] = {"resolved_by": ["D", 2, "teacher"], "candidates": []}
        json.dump(n6, open(f"{self.d}/n6_report2.json", "w"))
        rc, rep, _ = self.merge(n6_report=f"{self.d}/n6_report2.json")
        self.assertFalse(rep["controls"]["C-N6SAME"]["pass"])
        z = dict(np.load(f"{self.d}/n6t_plans.npz"))
        z["plans20"] = z["plans20"].copy()
        z["plans20"][1, 3, 1] += 1e-9                                             # the banked N6-T plan differs by 1e-9 from the cascade's T0 plan
        np.savez(f"{self.d}/n6t_bad.npz", **z)
        rc, rep, _ = self.merge(n6t_plans=f"{self.d}/n6t_bad.npz")
        self.assertFalse(rep["controls"]["C-N6SAME"]["pass"])
        rc, rep, _ = self.merge()
        self.assertTrue(rep["controls"]["C-N6SAME"]["pass"])
        n6["repaired"]["N6T"] = []
        n6["residual_table"] = {}
        json.dump(n6, open(f"{self.d}/n6_report3.json", "w"))                     # the banked report says nobody was certified but T0 certifies t01
        rc, rep, _ = self.merge(n6_report=f"{self.d}/n6_report3.json")
        self.assertFalse(rep["controls"]["C-N6SAME"]["pass"])

    def test_n6same_a_t0_pick_moved_to_a_later_tier_is_red(self):
        self.rows[1]["chosen"]["N7T3"] = self.chosen_for("t02")["N7T3"]            # a different candidate in the slot
        rc, rep, _ = self.merge()
        self.assertFalse(rep["controls"]["C-N6SAME"]["pass"])
        self.assertFalse(rep["controls"]["C-SELECT"]["pass"])


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------- READ
class Syn:
    """2000 tokens, 20 logs, 2 known36 -> 1998 primary. 20 N5-A residual tokens (zero DAC in N5-A): the first 20 non-known tokens; N6-T certified the first 4"""
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
        self.arms = {n: {t: dict(v) for t, v in base.items()} for n in ("N2R", "N4D", "N5A", "N6T", "N7T3", "N7T")}
        self.rep = {"N7T3": [], "N7T": [], "N7X": []}
        self.tier = {"N7T3": {}, "N7T": {}, "N7X": {}}
        self.n6 = self.resid[:4]
        for t in self.n6:                                                   # N6-T certified these 4 (and the cascade keeps them: same harness row)
            self.arms["N6T"][t].update(dac=1.0, pdms=0.97)
        self.build = {"verdict": "PASS", "controls": {"C-CERT-B7": {"pass": True}, "C-DACMAP6": {"pass": True}, "C-X0": {"pass": True},
                                                      "C-N6SAME": {"pass": True, "n6t_tokens": list(self.n6)}},
                      "repaired": self.rep, "tier_of": self.tier, "counts": {"N7T3": {"by_tier": {}}, "N7T": {"by_tier": {}}},
                      "residual_table": {t: {"blocking_class": "no candidate in any tier reaches DAC = 1", "blocking_class_n7t3": "no candidate in tiers T0-T3 reaches DAC = 1"} for t in self.resid}}
        self.unch = {"pass": True, "arms": {"N7T3": {"pass": True}, "N7T": {"pass": True}, "N7X": {"pass": True, "bit_identical_to_n5a": True}}, "n_residual": 20}
        for arm in ("N7T3", "N7T"):
            self.fix(arm, self.n6, tier="T0_N6")
        self.pins_ok()

    def fix(self, arm, toks, pdms=0.97, tier="T1_L2", **m):
        for t in toks:
            self.arms[arm][t].update(dac=1.0, pdms=pdms, **m)
            self.tier[arm][t] = tier
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

    def run(self, k=50):
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

    def fix_both(self, n=19, **kw):
        for arm in ("N7T3", "N7T"):
            self.S.fix(arm, self.S.resid[4:n], **kw)

    def test_confirmed_naming_tiers_and_k_in_bonferroni_key(self):
        self.S.fix("N7T3", self.S.resid[4:17], tier="T1_L2")
        self.S.fix("N7T", self.S.resid[4:17], tier="T1_L2")
        self.S.fix("N7T", self.S.resid[17:19], tier="T4_DEEP")
        self.S.fix("N7T3", self.S.resid[17:19], tier="T3_OFF")
        rc, out, res = self.S.run(k=50)
        self.assertEqual(rc, 0, res["controls"])
        p = res["primary"]
        self.assertEqual(p["outcome"], "CONFIRMED", p)
        self.assertEqual(p["n_teacher_tol_replacements"], 19)
        self.assertIn("tiers T0 to T3", res["naming"])
        self.assertIn("N_teacher_tol = 19 of 20 residual tokens", res["naming"])
        self.assertIn("tolerance (NC/DDC kept, TTC/C free), teacher only", res["naming"])
        self.assertIn("tiers T0 to T4", res["secondary_N7T"]["naming"])
        self.assertNotIn("human", json.dumps(res["primary"]["naming"]).lower())
        self.assertIn("ci_bonferroni_K50", p["delta_vs_N5A_paired"]["pdms"])
        self.assertEqual(p["bars"], {"1_paper_level_pdms_ge_95.8_and_dac_ge_99.9": True, "2_no_harm_nc_ddc_and_no_regression_vs_N6T": True, "3_certificate_integrity": True})
        self.assertEqual(res["paper_level"]["N7T3"]["level"], "PAPER LEVEL")
        self.assertEqual(res["paper_level"]["N6T_teacher_only"]["zero_dac_tokens"], 20 - 4)
        self.assertEqual(p["remaining_zero_dac"]["count"], 1)
        self.assertIn("blocking", p["remaining_zero_dac"])
        self.assertIn("does not apply", res["rests_on_T4"])
        self.assertIn("N7T3_minus_N6T", res["attribution"])
        self.assertTrue(res["controls"]["C-N6SAME"]["pass"])

    def test_the_result_rests_on_t4_sentence(self):
        self.S.fix("N7T3", self.S.resid[4:14])                                       # 4 + 10 repaired: 6 zero-DAC of 1998 = 99.70 (NEAR)
        self.S.fix("N7T", self.S.resid[4:19])                                        # 4 + 15 repaired: 1 zero-DAC = 99.95 (PAPER LEVEL)
        rc, _, res = self.S.run()
        self.assertEqual(rc, 0)
        self.assertEqual(res["paper_level"]["N7T3"]["level"], "NEAR")
        self.assertEqual(res["paper_level"]["N7T"]["level"], "PAPER LEVEL")
        self.assertIn("rests on the T4 slow-down tier", res["rests_on_T4"])
        self.assertEqual(res["primary"]["outcome"], "NOT CONFIRMED")                 # the primary is N7-T3 alone
        self.assertEqual(res["primary"]["not_confirmed_branches_in_card_order"], ["paper_level_not_reached"])
        self.assertEqual(res["secondary_N7T"]["outcome"], "CONFIRMED")

    def test_paper_level_boundary_is_the_point_estimate(self):
        self.fix_both(18)                                                    # 2 zero-DAC of 1998 -> 99.8999 < 99.9
        rc, _, res = self.S.run()
        self.assertEqual(rc, 0)
        self.assertFalse(res["primary"]["bars"]["1_paper_level_pdms_ge_95.8_and_dac_ge_99.9"])
        self.assertEqual(res["primary"]["outcome"], "NOT CONFIRMED")
        self.assertEqual(res["paper_level"]["N7T3"]["level"], "NEAR")
        self.setUp()
        self.fix_both(19)
        rc, _, res = self.S.run()
        self.assertEqual(res["primary"]["outcome"], "CONFIRMED")

    def test_pdms_below_95_8_is_not_paper_level_even_at_dac_999(self):
        for arm in self.S.arms:
            for t in self.S.tok:
                if self.S.arms[arm][t]["dac"] == 1.0:
                    self.S.arms[arm][t]["pdms"] = 0.955
        self.fix_both(19, pdms=0.955)
        rc, _, res = self.S.run()
        self.assertEqual(rc, 0)
        self.assertFalse(res["paper_level"]["N7T3"]["paper_pdms_point_ok"])
        self.assertTrue(res["paper_level"]["N7T3"]["paper_dac_point_ok"])
        self.assertFalse(res["primary"]["bars"]["1_paper_level_pdms_ge_95.8_and_dac_ge_99.9"])

    def test_a_changed_token_losing_nc_or_ddc_is_the_harm_branch(self):
        self.fix_both(19)
        self.S.arms["N7T3"][self.S.resid[5]]["nc"] = 0.5
        rc, _, res = self.S.run()
        self.assertEqual(rc, 0)
        p = res["primary"]
        self.assertEqual(p["no_harm_changed_token_violations"]["count"], 1)
        self.assertFalse(p["bars"]["2_no_harm_nc_ddc_and_no_regression_vs_N6T"])
        self.assertIn("harm", p["not_confirmed_branches_in_card_order"])
        self.assertTrue(res["secondary_N7T"]["bars"]["2_no_harm_nc_ddc_and_no_regression_vs_N6T"])         # the secondary is read separately
        self.setUp()
        self.fix_both(19)
        self.S.arms["N7T3"][self.S.resid[6]]["ddc"] = 0.5
        rc, _, res = self.S.run()
        self.assertEqual(res["primary"]["no_harm_changed_token_violations"]["tokens"], [self.S.resid[6]])

    def test_ttc_c_worse_on_changed_tokens_is_the_declared_tolerance(self):
        for t in self.S.resid[:19]:
            for arm in ("N2R", "N4D", "N5A", "N6T"):
                self.S.arms[arm][t].update(nc=0.5, ddc=0.5)
        self.fix_both(19, ttc=0.0, c=0.0)
        for arm in ("N7T3", "N7T"):
            for t in self.S.resid[:19]:
                self.S.arms[arm][t].update(nc=1.0, ddc=1.0)
            for t in self.S.n6:
                self.S.arms[arm][t].update(ttc=0.0, c=0.0)
        for t in self.S.n6:
            self.S.arms["N6T"][t].update(dac=1.0, pdms=0.97, ttc=0.0, c=0.0, nc=1.0, ddc=1.0)
        rc, _, res = self.S.run()
        self.assertEqual(rc, 0, res["controls"])
        p = res["primary"]
        self.assertTrue(p["bars"]["2_no_harm_nc_ddc_and_no_regression_vs_N6T"], p["no_harm_paired_x100_minus_N5A"])
        self.assertLess(p["tolerance_declared_ttc_c_reported"]["ttc"]["mean"], 0.0)
        self.assertEqual(p["outcome"], "CONFIRMED")

    def test_cert_integrity_branch(self):
        self.fix_both(19)
        self.S.build["controls"]["C-CERT-B7"] = {"pass": False}
        rc, _, res = self.S.run()
        self.assertEqual(rc, 0)
        self.assertFalse(res["primary"]["bars"]["3_certificate_integrity"])
        self.assertIn("cert_integrity", res["primary"]["not_confirmed_branches_in_card_order"])

    def test_certificate_disabled_is_red_at_c_cert_and_no_aggregate(self):
        self.S.rep["N7T3"] = self.S.n6 + [self.S.resid[7]]                  # builder says certified, harness DAC 0
        self.S.tier["N7T3"][self.S.resid[7]] = "T1_L2"
        rc, out, res = self.S.run()
        self.assertEqual(rc, 1)
        self.assertFalse(res["controls"]["C-CERT"]["pass"])
        self.assertNotIn("primary", res)
        self.assertNotIn("paper_level", res)
        self.assertIn("NO VERDICT", res["verdict"])

    def test_2e6_perturbation_is_red_at_c_id_c_for_both_arms(self):
        for arm in ("N7T3", "N7T"):
            self.setUp()
            self.S.arms[arm]["k0500"]["pdms"] += 2e-6
            rc, _, res = self.S.run()
            self.assertEqual(rc, 1, arm)
            self.assertFalse(res["controls"]["C-ID-C"]["pass"])
            self.assertEqual(res["controls"]["C-ID-C"]["arms"][arm]["tokens_over_tol"], 1)

    def test_an_unflagged_token_that_moved_is_red(self):
        self.fix_both(19)
        self.S.arms["N7T"]["k0100"]["dac"] = 0.0
        rc, out, res = self.S.run()
        self.assertEqual(rc, 1)
        self.assertFalse(res["controls"]["C-ID-C"]["pass"])
        self.assertNotIn("primary", res)

    def test_dropped_row_is_red_at_c_cov(self):
        self.S.arms["N7T3"].pop(self.S.tok[29])
        rc, _, res = self.S.run()
        self.assertEqual(rc, 1)
        self.assertFalse(res["controls"]["C-COV"]["pass"])

    def test_n7x_that_changed_anything_is_red_at_c_reg(self):
        self.fix_both(19)
        self.S.rep["N7X"] = [self.S.resid[0]]
        rc, _, res = self.S.run()
        self.assertEqual(rc, 1)
        self.assertFalse(res["controls"]["C-REG"]["pass"])
        self.assertNotIn("primary", res)
        self.setUp()
        self.fix_both(19)
        self.S.unch["arms"]["N7X"]["bit_identical_to_n5a"] = False
        rc, _, res = self.S.run()
        self.assertEqual(rc, 1)
        self.assertFalse(res["controls"]["C-REG"]["pass"])

    def test_c_n6same_read_side_a_changed_n6t_token_must_read_the_same_row(self):
        self.fix_both(19)
        self.S.arms["N7T3"][self.S.n6[1]]["pdms"] = 0.5                       # the cascade's row for an N6-T token differs from N6-T's (a replaced T0 plan)
        rc, _, res = self.S.run()
        self.assertEqual(rc, 1)
        self.assertFalse(res["controls"]["C-N6SAME"]["pass"])
        self.assertNotIn("primary", res)
        self.setUp()
        self.fix_both(19)
        self.S.rep["N7T"] = [t for t in self.S.rep["N7T"] if t != self.S.n6[0]]       # an N6-T token left out of N7-T's applied set
        rc, _, res = self.S.run()
        self.assertEqual(rc, 1)
        self.assertFalse(res["controls"]["C-N6SAME"]["pass"])
        self.setUp()
        self.fix_both(19)
        self.S.build["controls"]["C-N6SAME"] = {"pass": False, "n6t_tokens": list(self.S.n6)}
        rc, _, res = self.S.run()
        self.assertEqual(rc, 1)

    def test_pinned_file_edited_is_red_at_c_pin(self):
        self.fix_both(19)
        open(f"{self.S.d}/pinned.txt", "w").write("edited")
        rc, _, res = self.S.run()
        self.assertEqual(rc, 1)
        self.assertFalse(res["controls"]["C-PIN"]["pass"])
        self.assertIn("pinned.txt", res["controls"]["C-PIN"]["mismatch"])

    def test_an_old_families_run_pinned_is_red_at_c_pin(self):
        self.fix_both(19)
        AN.A5.TIP_FAMILIES_SHA256 = "44c62807f1254e2ca4a41e4094097071b96bccaba9e754703ddf4a689e4d73a6"
        rc, _, res = self.S.run()
        self.assertEqual(rc, 1)
        self.assertFalse(res["controls"]["C-PIN"]["pass"])
        self.assertFalse(res["controls"]["C-PIN"]["families_run_is_tip"])

    def test_c_est_detects_a_wrong_draw_scheme(self):
        self.fix_both(19)
        open(f"{self.S.d}/pe/proxy_eval.py", "w").write("def boot(pt, tl):\n    return (0.0, 0.0, 0.0)\n")
        rc, _, res = self.S.run()
        self.assertFalse(res["controls"]["C-EST"]["pass"])


class TestFamread(unittest.TestCase):
    def _ci(self, sb, sm, cm):
        return {"frames": {"executed_tracked": {"families": {"longitudinal": {"speed_bias_mps": {"diff_B_minus_A": {"point": sb, "lo": 0, "hi": 0}},
                                                                              "speed_mae_mps": {"diff_B_minus_A": {"point": sm, "lo": 0, "hi": 0}}},
                                                            "lateral": {"cross_mae_m": {"diff_B_minus_A": {"point": cm, "lo": 0, "hi": 0}}}}}}}

    def _fam(self, toks, vals):
        return {"tokens": toks, "logs": ["l"] * len(toks), "frames": {"executed_tracked": {"block": {"_per_window": {"status": "OK", "longitudinal": {"speed_mae_mps": {"kind": "mean", "value": vals}}}}}}}

    def _run(self, t3, t=None, fams=None, build=None):
        d = tempfile.mkdtemp(dir=TMP)
        json.dump(t3, open(f"{d}/ci_t3.json", "w"))
        kw = dict(fam_ci_t3=f"{d}/ci_t3.json", fam_ci_t="", fam_a="", fam_t3="", fam_t="", build_report="", out=f"{d}/o.json", k=50)
        if t is not None:
            json.dump(t, open(f"{d}/ci_t.json", "w"))
            kw["fam_ci_t"] = f"{d}/ci_t.json"
        if fams:
            for nm, doc in fams.items():
                json.dump(doc, open(f"{d}/{nm}.json", "w"))
                kw["fam_" + nm] = f"{d}/{nm}.json"
        if build:
            json.dump(build, open(f"{d}/b.json", "w"))
            kw["build_report"] = f"{d}/b.json"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            AN.cmd_famread(SimpleNamespace(**kw))
        return json.load(open(f"{d}/o.json")), buf.getvalue()

    def test_no_flag_and_each_flag_on_the_primary(self):
        self.assertFalse(self._run(self._ci(0.0, 0.0, 0.0))[0]["guard"]["flag"])
        self.assertTrue(self._run(self._ci(0.30, 0.0, 0.0))[0]["guard"]["flag"])
        self.assertTrue(self._run(self._ci(0.0, 0.2, 0.0))[0]["guard"]["flag"])
        self.assertTrue(self._run(self._ci(0.0, 0.0, 0.06))[0]["guard"]["flag"])
        r, _ = self._run(self._ci(0.0, 0.0, 0.0), self._ci(0.0, 0.3, 0.0))
        self.assertEqual(r["strategic"], "PAUSED (PI 2026-09-19)")
        self.assertEqual(r["K"], 50)
        self.assertFalse(r["guard"]["flag"])
        self.assertTrue(r["secondary_N7T"]["guard"]["flag"])                    # the secondary is reported beside the primary, never mixed into it
        self.assertEqual(AN.GUARD, {"speed_bias_diff_mps": 0.25, "speed_mae_diff_mps": 0.15, "cross_mae_diff_m": 0.05})
        for k in ("lateral.heading_mae_deg", "lateral.curvature_mae_1pm", "lateral.yaw_rate_mae_degps", "tactical.maneuver_5way_collapsed.accuracy"):
            self.assertIn(k, r["reported_not_bars"])

    def test_a_ci_without_paired_diffs_is_refused_not_read_as_a_flag(self):
        with self.assertRaises(SystemExit):
            self._run({"frames": {"executed_tracked": {"families": {}}}})

    def test_the_changed_tokens_own_speed_mae_per_token(self):
        toks = ["a", "b", "c", "d"]
        fams = {"a": self._fam(toks, [0.1, 0.1, 0.1, 0.1]), "t3": self._fam(toks, [0.1, 0.12, 0.1, 0.1]), "t": self._fam(toks, [0.1, 0.12, 0.1, 0.9])}
        build = {"repaired": {"N7T3": ["b"], "N7T": ["b", "d"]}, "tier_of": {"N7T3": {"b": "T0_N6"}, "N7T": {"b": "T0_N6", "d": "T4_DEEP"}}}
        r, out = self._run(self._ci(0, 0, 0), self._ci(0, 0, 0), fams, build)
        t3, t = r["changed_tokens_speed_mae_N7T3"], r["changed_tokens_speed_mae_N7T"]
        self.assertEqual(t3["status"], "OK")
        self.assertAlmostEqual(t3["per_token"]["b"]["diff_mps"], 0.02)
        self.assertAlmostEqual(t["per_token"]["d"]["diff_mps"], 0.8)                                  # the T4 token's own speed change is visible
        self.assertEqual(t["per_token"]["d"]["tier"], "T4_DEEP")
        self.assertAlmostEqual(t["by_tier_mean_diff_mps"]["T4_DEEP"], 0.8)
        self.assertIn("ZZN7_SPEEDTOK", out)

    def test_a_families_json_without_the_per_window_speed_is_reported_unavailable_not_dropped(self):
        fams = {"a": {"tokens": ["a"], "frames": {}}, "t3": {"tokens": ["a"], "frames": {}}}
        r, out = self._run(self._ci(0, 0, 0), None, fams, {"repaired": {"N7T3": ["a"]}, "tier_of": {}})
        self.assertEqual(r["changed_tokens_speed_mae_N7T3"]["status"], "UNAVAILABLE")
        self.assertIn("UNVERIFIED", r["changed_tokens_speed_mae_N7T3"]["note"])
        self.assertIn("UNAVAILABLE", out)


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- CHAIN
LEDGER_N6 = {"key": "TF-N6:aaaa:E3", "n": 49}


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

    def write_ledger(self, n=50, extra=None, drop_prev=False):
        key = f"TF-N7:{sha(self.card)}:E3"
        reads = ([] if drop_prev else [LEDGER_N6]) + [{"key": key, "n": n}] + (extra or [])
        json.dump({"reads": reads}, open(f"{self.run}/ledger.json", "w"))

    def args(self, **kw):
        d = dict(run_dir=self.run, card=self.card, card_sha256=sha(self.card), pins_sha256=sha(self.pins), ledger=f"{self.run}/ledger.json", k=50, procs=2,
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

    def test_constants_and_required_inputs(self):
        self.assertEqual((CH.K_CARD, CH.K_PREV), (50, 49))
        for rel in ("inputs/n6t/n6_plans_N6T.npz", "inputs/n6t/refe_tfn6_n6t.csv", "inputs/n6t/n6_build_report.json", "n7/n7_nt_build.py", "n7/n7_lib.py", "n7/n7_select.py",
                    "n7/n7_analyze.py", "n7/fixture_navtrain_tiers.json", "n6/n6_nt_build.py"):
            self.assertIn(rel, CH.REQUIRED)
        self.assertEqual(CH.PINS_REL, "n7/N7_PINS.json")
        self.assertIn("n7", CH.WATCH_TOP)

    def test_card_sha_and_ledger(self):
        self.refused(self.args(card_sha256="0" * 64), "sha256(")
        self.write_ledger(n=49)
        self.refused(self.args(), "n = 49")
        self.write_ledger(extra=[{"key": "other", "n": 51}])
        self.refused(self.args(), "n = 51")
        self.write_ledger(drop_prev=True)
        self.refused(self.args(), "n = 49) is not in the ledger")
        json.dump({"reads": [LEDGER_N6]}, open(f"{self.run}/ledger.json", "w"))
        self.refused(self.args(), "no entry")
        self.write_ledger()
        self.refused(self.args(k=49), "--k is 49")
        json.dump({"reads": [LEDGER_N6, {"key": f"TF-N6:{sha(self.card)}:E3", "n": 50}]}, open(f"{self.run}/ledger.json", "w"))          # the N6 key is not an N7 entry
        self.refused(self.args(), "no entry")

    def test_pins_sha_edited_and_unpinned_files(self):
        self.refused(self.args(pins_sha256="0" * 64), "pins-sha256")
        open(f"{self.run}/n7/n7_nt_build.py", "a").write("#")
        self.refused(self.args(), "pinned files differ")
        self.mk_pins()
        a = self.args(pins_sha256=sha(self.pins))
        open(f"{self.run}/n7/extra_tool.py", "w").write("x")
        self.refused(a, "unpinned files")
        os.remove(f"{self.run}/n7/extra_tool.py")
        open(f"{self.run}/n6/n6_nt_build.py", "a").write("#")                                  # the frozen N6 build edited
        self.refused(a, "pinned files differ")

    def test_old_families_run_is_refused(self):
        CH.TIP_FAMILIES_SHA256 = "44c62807f1254e2ca4a41e4094097071b96bccaba9e754703ddf4a689e4d73a6"
        self.refused(self.args(), "TIP version")
        CH.TIP_FAMILIES_SHA256 = sha(f"{self.run}/{CH.TIP_FAMILIES}")
        open(f"{self.run}/{CH.TIP_FAMILIES}", "w").write("older families_run")
        self.refused(self.args(), "pinned files differ")

    def test_procs_and_a_failing_reach(self):
        self.refused(self.args(procs=3), "--procs")
        CH.run_reach = lambda a, real=False, out="": (1, "ZZN7_REACH BAD card tables read back: n7_lib.LEVERS != card table on STR_label")
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
        argv = ["n7_nt_chain.py", "--run-dir", self.run, "--card", self.card, "--card-sha256", sha(self.card), "--pins-sha256", sha(self.pins), "--ledger",
                f"{self.run}/ledger.json", "--k", "50", "--dry-run", "--skip-cache-listing", "--min-free-gb", "0"]
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
        self.assertFalse(os.path.exists(f"{self.run}/n7run"))
        for frag in ("n7_nt_build.py reach", "n7_nt_build.py derive", "n7_nt_build.py merge", "n7_nt_build.py unch", "score_p4_a1.py --label refe_tfn7_n7t3", "score_p4_a1.py --label refe_tfn7_n7t",
                     "n7_nt_analyze.py read", "transp_score.py", "famprep", "families_run.py", "--per-window", "fam_ci.py", "famread", "fixture_navtrain_tiers.json",
                     "--n6-report", "--n6t-plans", "fam_ci_N7T3_vs_N5A.json", "fam_ci_N7T_vs_N5A.json"):
            self.assertIn(frag, out, frag)
        self.assertNotIn("--mutate", out)
        self.assertNotIn("--smoke-allow-mutation", out)
        self.assertNotIn("refe_tfn7_n7x", out)                                   # N7-X is proven bit-identical and is not scored
        self.assertNotIn("human_first", out)
        self.assertIn("N7T3=", out)
        self.assertIn("N6T=", out)
        self.assertIn("N5A=", out)
        self.assertIn("ZZN7_CHAIN DONE", out)
        self.assertEqual(out.count("DRY-RUN CMD") >= 12, True)

    def test_make_pins_refuses_a_tree_with_an_old_families_run(self):
        import n7_make_pins as MP
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
