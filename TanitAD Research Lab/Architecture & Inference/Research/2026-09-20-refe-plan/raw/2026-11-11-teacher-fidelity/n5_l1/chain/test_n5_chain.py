"""TF-N5 chain tests: CPU, synthetic data, no navtest.  Run (navsim venv, N4_RUN = the run tree, CUDA off):
    N4_RUN=F:/tanitad_scratch/n5nt/run  python -m pytest -q test_n5_chain.py            (the `real` test needs the nuPlan DB and takes about a minute)

 REACH   every card L1 parameter reaches the executed repair (a perturbed object changes the behavioural signature AND is refused); margin -0.05, sign -1, a cert-off
         object, a different cap are refused; a zero-correction control reads EXACTLY the base label while the card object moves it; the build CLI and the chain argv
         carry every card flag; a different flag value is refused at the chain guard and at derive
 UNCH    the file-level C-UNCH: a 2e-6 edit of one unchanged plan, L1 applied to ALL tokens, a flagged-but-unchanged token, a wrong L1-run count each go RED
 MERGE   the build merge on synthetic rows: green; certificate off / L1 on all tokens / non-deterministic repair / wrong DAC map / cache-ego label each go RED
 READ    n5_nt_analyze: PRIMARY bars (CONFIRMED and every NOT-CONFIRMED branch, the 5-of-20 boundary), exact binomial interval, paper-level tiers, K in the Bonferroni
         key, the control exit BEFORE any N5-A aggregate (certificate disabled -> C-CERT; 2e-6 -> C-ID-C; dropped row -> C-COV; sign-flipped arm that helps -> C-REG;
         pinned file edited -> C-PIN; an old families_run.py pinned -> C-PIN)
 CHAIN   guards refuse: card sha, ledger entry / n / higher n / missing TF-N4-A1 entry, pins sha, an edited pinned file, an unpinned file, procs, every L1 flag, a failing reach,
         an OLD families_run.py (pinned, on disk), low disk; --dry-run prints every stage's command and no mutation flag; the families JSON check refuses a JSON without tokens"""
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
import n5_nt_analyze as AN      # noqa: E402
import n5_nt_build as BD        # noqa: E402
import n5_nt_chain as CH        # noqa: E402

RUN = os.environ["N4_RUN"]
A1 = os.environ.get("N5_A1_ANALYZE", os.path.join(RUN, "n2", "tf_n1_analyze_a1.py"))
TMP = os.environ.get("N5_TMP") or None
COLS = ["", "token", "valid", "no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound", "comfort",
        "driving_direction_compliance", "score"]
OK5 = {"dac": 1.0, "nc": 1.0, "ddc": 1.0, "ttc": 1.0, "c": 1.0, "ep": 0.9, "pdms": 0.9}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


# ------------------------------------------------------------------------------------------------------------- REACH
class TestReach(unittest.TestCase):
    def test_card_arms_pass_and_read_back(self):
        for arm, spec in BD.CARD.items():
            r = BD.check_arm(arm, BD.make_repair(spec))
            self.assertTrue(r["ok"], r["bad"])
            for k in ("margin", "iters", "sign", "cert", "centre_m", "fallback_margin", "damp", "damp_cum"):
                self.assertEqual(r["readback"][k], spec[k])
            self.assertEqual(r["readback"]["target"], BD.TARGET)
        self.assertEqual(BD.CARD["N5X"]["sign"], -1.0)
        self.assertEqual(BD.CARD["N5A"]["sign"], 1.0)

    def test_every_card_parameter_reaches_the_executed_repair(self):
        """MUTATE, do not inspect: perturb each parameter; the behavioural signature on the synthetic scenarios must change AND the arm must be refused"""
        ref = BD.signature(BD.make_repair(BD.CARD["N5A"]))
        pert = {"margin": 0.06, "iters": 4, "cert": "dac", "centre_m": 0.02, "fallback_margin": 0.01, "damp": 0.25, "damp_cum": True, "sign": -1.0}
        self.assertEqual(set(pert), set(BD.CARD["N5A"]))
        for k, v in pert.items():
            fn = BD.make_repair(dict(BD.CARD["N5A"], **{k: v}))
            self.assertNotEqual(BD.signature(fn), ref, f"{k}: the signature did not move: the parameter does not reach the repair")
            r = BD.check_arm("N5A", fn)
            self.assertFalse(r["ok"], k)
            self.assertTrue(any(k in b or (k == "iters" and "iters" in b) for b in r["bad"]), (k, r["bad"]))

    def test_deterministic_signature_and_a_control_that_reads_exactly_the_base(self):
        self.assertEqual(BD.signature(BD.make_repair(BD.CARD["N5A"])), BD.signature(BD.make_repair(BD.CARD["N5A"])))
        F = BD.fakes()
        T = F.T
        t20 = T.teacher(lambda t: np.where(t < 2.0, 0.30, -0.16))
        cx = T.FakeCx(T.corridor(0.50, 0.34))
        base = cx.labels(t20)["R"]
        r = BD.make_repair(BD.CARD["N5A"])(cx, t20, dict(T.BASE_OK))
        self.assertIsNotNone(r)
        self.assertGreater(float(np.abs(r["plan"] - base).max()), 0.05)                      # non-zero effect: the card object moves the label
        z = BD.L1.repair_l1(T.FakeCx(T.corridor(0.50, 0.34)), t20, dict(T.BASE_OK), cert=lambda c, b: True)
        self.assertTrue(np.array_equal(z["plan"], base))                                      # control: zero correction == the base label, bit for bit

    def test_minus_margin_sign_cert_off_cap_are_refused(self):
        for m in ("margin_neg", "cert_off"):
            self.assertFalse(BD.check_arm("N5A", BD.make_repair(BD.CARD["N5A"], m))["ok"], m)
        self.assertFalse(BD.check_arm("N5A", BD.make_repair(dict(BD.CARD["N5A"], sign=-1.0)))["ok"])
        self.assertFalse(BD.check_arm("N5X", BD.make_repair(dict(BD.CARD["N5X"], sign=1.0)))["ok"])
        self.assertFalse(BD.check_arm("N5A", BD.make_repair(dict(BD.CARD["N5A"], iters=4)))["ok"])
        self.assertFalse(BD.check_arm("N5A", BD.make_repair(dict(BD.CARD["N5A"], margin=-0.05)))["ok"])

    def test_wrong_target_function_is_refused(self):
        import functools
        fn = functools.partial(lambda *a, **k: None, margin=0.05)
        with self.assertRaises(Exception):
            BD.check_arm("N5A", fn)

    def test_build_cli_refuses_mutation_without_smoke_flag_and_a_changed_flag(self):
        base = dict(margin=0.05, iters=5, cert="full", centre_m=0.05, fallback_margin=0.005, damp=0.5, damp_cum=False, sign=1.0, sign_x=-1.0, mutate="none",
                    smoke_allow_mutation=False)
        with self.assertRaises(SystemExit) as c:
            BD.cmd_derive(SimpleNamespace(**dict(base, mutate="margin_neg")))
        self.assertIn("REFUSING", str(c.exception))
        for k, v in (("margin", 0.06), ("iters", 4), ("centre_m", 0.04), ("fallback_margin", 0.01), ("damp", 0.4), ("cert", "dac"), ("sign", -1.0), ("sign_x", 1.0)):
            with self.assertRaises(SystemExit) as c:
                BD.cmd_derive(SimpleNamespace(**dict(base, **{k: v})))
            self.assertIn("differs from the card", str(c.exception), k)

    def test_chain_argv_carries_every_flag_and_no_mutation(self):
        a = SimpleNamespace(run_dir="R", margin=0.05, iters=5, cert="full", centre_m=0.05, fallback_margin=0.005, damp=0.5, damp_cum=False, sign=1.0, sign_n5x=-1.0)
        argv = CH.build_argv(a, "R", 0, 2)
        self.assertFalse(any("mutate" in x for x in argv))
        for f in ("--margin", "--iters", "--cert", "--centre-m", "--fallback-margin", "--damp", "--sign", "--sign-x"):
            self.assertIn(f, argv)
        self.assertEqual(argv[argv.index("--sign-x") + 1], "-1.0")
        self.assertIn("--n4d-report", argv)

    def test_real_two_failing_navtrain_instances(self):
        if os.environ.get("N5_SKIP_REAL"):
            self.skipTest("N5_SKIP_REAL")
        r = BD.reach_real(os.path.join(HERE, "fixture_real_instances.json"))
        self.assertTrue(r["ok"], r)
        for k, v in r["instances"].items():
            self.assertTrue(v["l1_certified"] and not v["sign_flipped_certified"] and v["zero_identity"] and v["deterministic"], (k, v))


# -------------------------------------------------------------------------------------------------------------- UNCH
class TestUnch(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(dir=TMP)
        n = 8
        rng = np.random.default_rng(1)
        self.tok = np.array([f"t{i:02d}" for i in range(n)])
        self.n2r = rng.normal(size=(n, 20, 3))
        self.n4d = self.n2r.copy()
        self.n4d[1, 4, 0] += 0.3                          # a REPAIR-D repair (banked)
        self.p8 = rng.normal(size=(n, 8, 3)).astype(np.float32)
        np.savez(f"{self.d}/n2r_plans.npz", tokens=self.tok, logs=self.tok, plans20=self.n2r)
        np.savez(f"{self.d}/n4d_plans.npz", tokens=self.tok, logs=self.tok, plans20=self.n4d)
        np.savez(f"{self.d}/n4d_seam.npz", token=self.tok, poses=self.p8)
        self.app = {"N5A": ["t02"], "N5X": [], "N5R": ["t01", "t02"]}
        self.write()

    def write(self, mut=None):
        for arm in BD.ARMS:
            P = (self.n4d if arm != "N5R" else self.n2r).copy()
            S = self.p8.copy()
            for t in self.app[arm]:
                i = list(self.tok).index(t)
                P[i, 5, 0] += 0.1 if arm != "N5R" or t == "t02" else 0.2
                S[i, 2, 0] += 0.1
            if mut and mut[0] == arm:
                mut[1](P, S)
            np.savez(f"{self.d}/n5_plans_{arm}.npz", tokens=self.tok, logs=self.tok, plans20=P, arm=np.array(arm))
            if arm in BD.SEAM_ARMS:
                np.savez(f"{self.d}/{BD.SEAM_LABEL[arm]}.npz", token=self.tok, poses=S)

    def chk(self, n_run=2, expect=2):
        return BD.check_unch(self.d, f"{self.d}/n4d_plans.npz", f"{self.d}/n4d_seam.npz", f"{self.d}/n2r_plans.npz", self.app, n_run, expect)

    def test_green(self):
        self.assertTrue(self.chk()["pass"])

    def test_2e6_perturbation_of_one_unchanged_plan_is_red(self):
        def m(P, S):
            P[0, 3, 1] += 2e-6
        self.write(("N5A", m))
        r = self.chk()
        self.assertFalse(r["pass"])
        self.assertEqual(r["arms"]["N5A"]["unflagged_changed"], ["t00"])

    def test_l1_applied_to_all_tokens_is_red(self):
        def m(P, S):
            P += 1e-12
        self.write(("N5X", m))
        self.assertFalse(self.chk()["pass"])

    def test_flagged_but_unchanged_is_red(self):
        self.app["N5X"] = ["t03"]
        self.assertFalse(self.chk()["pass"])

    def test_seam_changed_without_plan_is_red(self):
        def m(P, S):
            S[5, 1, 1] += 1e-3
        self.write(("N5A", m))
        self.assertFalse(self.chk()["pass"])

    def test_l1_run_count_must_equal_the_card(self):
        self.assertFalse(self.chk(n_run=26, expect=25)["pass"])
        self.assertFalse(self.chk(n_run=24, expect=25)["pass"])

    def test_n5r_builder_only_is_checked_against_n2r(self):
        def m(P, S):
            P[7, 0, 0] += 1e-9
        self.write(("N5R", m))
        self.assertFalse(self.chk()["pass"])


# -------------------------------------------------------------------------------------------------------------- MERGE
class TestMerge(unittest.TestCase):
    """the build merge on 8 synthetic tokens (to_navsim is the real function; nothing touches the harness)"""

    def setUp(self):
        import n4_geom as G
        self.ND = G.ND
        self.d = tempfile.mkdtemp(dir=TMP)
        n = 8
        self.tok = [f"t{i:02d}" for i in range(n)]
        rng = np.random.default_rng(3)
        t = 0.2 * np.arange(1, 21)
        base = np.stack([np.stack([8 * t, 0.05 * rng.normal(size=20).cumsum() * 0.1, np.zeros(20)], -1) for _ in range(n)])
        self.n2r = base
        self.n4d = base.copy()
        self.n4d[1, :, 1] += 0.1
        np.savez(f"{self.d}/n2r_plans.npz", tokens=np.array(self.tok), logs=np.array([f"l{i % 2}" for i in range(n)]), plans20=self.n2r)
        np.savez(f"{self.d}/n4d_plans.npz", tokens=np.array(self.tok), logs=np.array([f"l{i % 2}" for i in range(n)]), plans20=self.n4d)
        p8 = np.stack([self.ND.to_navsim(self.n4d[i]).astype(np.float32) for i in range(n)])
        np.savez(f"{self.d}/n4d_seam.npz", token=np.array(self.tok), fingerprint=np.array(["f"] * n), poses=p8, sampling=np.array([8.0, 0.5]),
                 arm=np.array("TEACHER_x_refe_tfn4_n4d"))
        with open(f"{self.d}/n2r.csv", "w", encoding="utf-8", newline="") as fo:
            w = csv.writer(fo)
            w.writerow(["", "token", "valid", "drivable_area_compliance"])
            for i, tk in enumerate(self.tok):
                w.writerow([i, tk, True, 0.0 if tk in ("t01", "t02", "t03") else 1.0])
        json.dump({"repaired": {"N4D": ["t01"]}, "unrepaired_dac0": {"N4D": ["t02", "t03"]}}, open(f"{self.d}/n4d_report.json", "w"))
        json.dump({"all_25": ["t02", "t03"], "guard_blocked": ["t03"], "non_guard": ["t02"]}, open(f"{self.d}/resid.json", "w"))
        json.dump({"ok": True, "arms": [{"arm": a, "ok": True} for a in BD.ARMS], "real": {"ok": True, "instances": {"easy": {"ok": True}}}}, open(f"{self.d}/reach.json", "w"))
        self.rows = self.make_rows()

    def make_rows(self):
        rows = []
        for i, tk in enumerate(self.tok):
            dac0 = tk in ("t01", "t02", "t03")
            r = {"log": f"l{i % 2}", "token": tk, "base": dict(OK5, dac=0.0 if dac0 else 1.0), "cderiv_maxdev_m": 0.0, "dac0": dac0, "in_l1_set": tk in ("t02", "t03"),
                 "arms": {}, "label_ego": "db" if dac0 else "db", "t_s": 0.1}

            def app(base_plan, dy):
                p = base_plan.copy()
                p[:, 1] += dy
                return {"applied": True, "plan": p.tolist(), "m": dict(OK5), "m_recheck": dict(OK5), "param": {}, "info": {"status": "CERTIFIED"}}
            if dac0:
                r["arms"]["N5R"] = app(self.n2r[i], 0.2) if tk != "t03" else {"applied": False, "info": {"status": "CAP_EXHAUSTED"}}
                r["det_equal"] = True
                if tk in ("t02", "t03"):
                    r["arms"]["N5A"] = app(self.n4d[i], 0.2) if tk == "t02" else {"applied": False, "info": {"status": "CAP_EXHAUSTED"}}
                    r["arms"]["N5X"] = {"applied": False, "info": {"status": "INFEASIBLE"}}
                    r["a_equals_r"] = True
                    if tk == "t02":
                        r["arms"]["N5R"]["plan"] = r["arms"]["N5A"]["plan"]
            rows.append(r)
        return rows

    def merge(self):
        with open(f"{self.d}/rows.jsonl", "w", encoding="utf-8") as fo:
            for r in self.rows:
                fo.write(json.dumps(r) + "\n")
        out = f"{self.d}/out"
        a = SimpleNamespace(rows=[f"{self.d}/rows.jsonl"], n2r_plans=f"{self.d}/n2r_plans.npz", n4d_plans=f"{self.d}/n4d_plans.npz", n4d_seam=f"{self.d}/n4d_seam.npz",
                            n4d_report=f"{self.d}/n4d_report.json", harness_csv_n2r=f"{self.d}/n2r.csv", residual=f"{self.d}/resid.json", reach_json=f"{self.d}/reach.json",
                            a1_base_rows="", out_dir=out, expect_l1_run=2)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = BD.cmd_merge(a)
        return rc, json.load(open(f"{out}/n5_build_report.json")), out

    def test_green_and_files(self):
        rc, rep, out = self.merge()
        self.assertEqual(rc, 0, {k: v for k, v in rep["controls"].items() if not v["pass"]})
        self.assertEqual(rep["repaired"]["N5A"], ["t02"])
        self.assertEqual(rep["repaired"]["N5X"], [])
        self.assertEqual(rep["repaired"]["N5R"], ["t01", "t02"])
        self.assertEqual(rep["unrepaired_l1"]["N5A"], ["t03"])
        # the seam of the unchanged tokens is bit-identical to the banked N4-D seam; the changed one is ND.to_navsim of the new plan
        s, sb = np.load(f"{out}/refe_tfn5_n5a.npz"), np.load(f"{self.d}/n4d_seam.npz")
        diff = [i for i in range(8) if not np.array_equal(s["poses"][i], sb["poses"][i])]
        self.assertEqual(diff, [2])
        self.assertTrue(str(s["arm"]).endswith("refe_tfn5_n5a"))

    def test_certificate_off_is_red_at_c_cert_b(self):
        self.rows[2]["arms"]["N5A"]["m_recheck"] = dict(OK5, dac=0.0)
        rc, rep, _ = self.merge()
        self.assertNotEqual(rc, 0)
        self.assertFalse(rep["controls"]["C-CERT-B"]["pass"])

    def test_cert_clause_worse_is_red(self):
        self.rows[2]["arms"]["N5A"]["m_recheck"] = dict(OK5, nc=0.0)
        rc, rep, _ = self.merge()
        self.assertFalse(rep["controls"]["C-CERT-B"]["pass"])

    def test_l1_on_all_tokens_is_red(self):
        for r in self.rows:                                   # N5-A / N5-X "run" on a token outside the L1 set (here a DAC-1 token), applied
            if r["token"] == "t05":
                r["arms"]["N5A"] = {"applied": True, "plan": (self.n4d[5] + 0.01).tolist(), "m": dict(OK5), "m_recheck": dict(OK5), "param": {}, "info": {"status": "CERTIFIED"}}
        rc, rep, _ = self.merge()
        self.assertNotEqual(rc, 0)
        self.assertFalse(rep["controls"]["C-UNCH"]["pass"])
        self.assertFalse(rep["controls"]["C-SET"]["pass"])

    def test_non_deterministic_repair_is_red(self):
        self.rows[2]["det_equal"] = False
        rc, rep, _ = self.merge()
        self.assertFalse(rep["controls"]["C-DET"]["pass"])

    def test_wrong_dac_map_is_red(self):
        self.rows[6]["base"]["dac"] = 0.0
        rc, rep, _ = self.merge()
        self.assertFalse(rep["controls"]["C-DACMAP"]["pass"])

    def test_cache_ego_label_is_red(self):
        self.rows[2]["label_ego"] = "cache"
        rc, rep, _ = self.merge()
        self.assertFalse(rep["controls"]["C-LABELEGO"]["pass"])

    def test_missing_row_and_changed_dac0_set_are_red(self):
        self.rows.pop(4)
        rc, rep, _ = self.merge()
        self.assertFalse(rep["controls"]["C-COVB"]["pass"])
        self.setUp()
        self.rows[1]["base"]["dac"] = 1.0                      # t01 no longer builder-DAC-0: the set differs from the banked N4-D build's
        self.rows[1]["arms"] = {}
        self.rows[1].pop("det_equal")
        rc, rep, _ = self.merge()
        self.assertFalse(rep["controls"]["C-SET"]["pass"])

    def test_reach_json_missing_or_failed_is_red(self):
        json.dump({"ok": False, "arms": [], "real": {"ok": False, "instances": {}}}, open(f"{self.d}/reach.json", "w"))
        rc, rep, _ = self.merge()
        self.assertFalse(rep["controls"]["C-REACH"]["pass"])

    def test_seamfn_mismatch_is_red(self):
        sb = dict(np.load(f"{self.d}/n4d_seam.npz"))
        sb["poses"] = sb["poses"].copy()
        sb["poses"][3, 0, 0] += np.float32(1e-3)
        np.savez(f"{self.d}/n4d_seam.npz", **sb)
        rc, rep, _ = self.merge()
        self.assertFalse(rep["controls"]["C-SEAMFN"]["pass"])


# ---------------------------------------------------------------------------------------------------------- analyzer
class Syn:
    """2000 tokens, 20 logs, 2 known36 -> 1998 primary. 25 REPAIR-D residual tokens (zero DAC in N2-R and N4-D): 5 guard-blocked + 20 non-guard, the first 25 non-known tokens"""
    N, G = 2000, 20

    def __init__(self, tmp):
        self.d = tmp
        self.tok = [f"k{i:04d}" for i in range(self.N)]
        self.log = {t: f"log{i % self.G}" for i, t in enumerate(self.tok)}
        self.known = self.tok[-2:]
        self.l1 = self.tok[:25]
        self.guard, self.nonguard = self.l1[:5], self.l1[5:]
        json.dump({"over_bar": {t: 1 for t in self.known}}, open(f"{tmp}/known.json", "w"))
        json.dump({"all_25": sorted(self.l1), "guard_blocked": sorted(self.guard), "non_guard": sorted(self.nonguard)}, open(f"{tmp}/resid.json", "w"))
        with gzip.open(f"{tmp}/tl.json.gz", "wt", encoding="utf-8") as fh:
            json.dump({"tokens": {t: {"log_name": self.log[t]} for t in self.tok}}, fh)
        os.makedirs(f"{tmp}/pe", exist_ok=True)
        open(f"{tmp}/pe/proxy_eval.py", "w").write(
            "import importlib.util\nimport sys\nspec = importlib.util.spec_from_file_location('a1', %r)\nA = importlib.util.module_from_spec(spec)\nspec.loader.exec_module(A)\n"
            "def boot(pt, tl):\n    return A.Boot(sorted(pt), tl).stat(pt)['_raw']\n" % A1)
        base = {t: dict(nc=1.0, dac=1.0, ddc=1.0, ep=0.9, ttc=1.0, c=1.0, pdms=0.97) for t in self.tok}
        for t in self.l1:
            base[t].update(dac=0.0, pdms=0.0)
        self.arms = {"N2R": base, "N4D": {t: dict(v) for t, v in base.items()}}
        self.rep = {"N5A": [], "N5X": [], "N5R": []}
        self.arms["N5A"] = {t: dict(v) for t, v in base.items()}
        self.arms["N5X"] = {t: dict(v) for t, v in base.items()}
        self.build = {"verdict": "PASS", "controls": {"C-CERT-B": {"pass": True}, "C-DACMAP": {"pass": True}}, "repaired": self.rep}
        self.unch = {"pass": True, "arms": {a: {"pass": True} for a in BD.ARMS}, "n_l1_run": 25}
        self.pins_ok()

    def fix(self, arm, toks, pdms=0.97):
        for t in toks:
            self.arms[arm][t].update(dac=1.0, pdms=pdms)
        self.rep[arm] = sorted(set(self.rep[arm]) | set(toks))

    def pins_ok(self, fam_sha=None):
        os.makedirs(f"{self.d}/w/pk/eval/instruments", exist_ok=True)
        fp = f"{self.d}/{AN.TIP_FAMILIES}"
        open(fp, "w").write("tip")
        self.fam_sha = sha(fp)
        open(f"{self.d}/pinned.txt", "w").write("x")
        AN.TIP_FAMILIES_SHA256 = fam_sha or self.fam_sha
        json.dump({"files": {"pinned.txt": sha(f"{self.d}/pinned.txt"), AN.TIP_FAMILIES: sha(fp)}}, open(f"{self.d}/pins.json", "w"))

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

    def run(self, k=47):
        self.write()
        a = SimpleNamespace(harness_csv=[f"{n}={self.d}/{n}.csv" for n in self.arms], build_report=f"{self.d}/build.json", unch=f"{self.d}/unch.json",
                            pins=f"{self.d}/pins.json", pin_root=self.d, proxy_eval=f"{self.d}/pe", prereg_sha256="abc", token_log=f"{self.d}/tl.json.gz",
                            known36=f"{self.d}/known.json", residual=f"{self.d}/resid.json", k=k, a1_analyze=A1, expect_n=self.N, expect_primary=self.N - 2, id_tol=1e-6,
                            out=f"{self.d}/out.json")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = AN.evaluate(a)
        return rc, buf.getvalue(), json.load(open(a.out, encoding="utf-8"))


class TestRead(unittest.TestCase):
    def setUp(self):
        self._tip = AN.TIP_FAMILIES_SHA256
        self.S = Syn(tempfile.mkdtemp(dir=TMP))

    def tearDown(self):
        AN.TIP_FAMILIES_SHA256 = self._tip

    def test_confirmed_and_k_in_bonferroni_key(self):
        self.S.fix("N5A", self.S.nonguard[:6])
        rc, out, res = self.S.run(k=47)
        self.assertEqual(rc, 0, res["controls"])
        p = res["primary"]
        self.assertEqual(p["outcome"], "CONFIRMED", p)
        self.assertEqual(p["repair_count"]["non_guard_dac1"], 6)
        self.assertIn("ci_bonferroni_K47", p["delta_pdms_not_a_bar"])
        self.assertEqual(p["not_confirmed_branches_in_card_order"], [])
        self.assertEqual(len(p["repair_count"]["exact_ci95"]), 2)

    def test_five_of_twenty_is_the_boundary(self):
        self.S.fix("N5A", self.S.nonguard[:5])
        rc, _, res = self.S.run()
        self.assertEqual(res["primary"]["outcome"], "CONFIRMED")
        self.setUp()
        self.S.fix("N5A", self.S.nonguard[:4])
        rc, _, res = self.S.run()
        self.assertEqual(rc, 0)
        p = res["primary"]
        self.assertEqual(p["outcome"], "NOT CONFIRMED")
        self.assertFalse(p["bars"]["1_repair_count_ge_5_of_20"])
        self.assertEqual(p["not_confirmed_branches_in_card_order"], ["repair_count_below_5"])

    def test_guard_blocked_repairs_do_not_count_toward_the_bar(self):
        self.S.fix("N5A", self.S.guard + self.S.nonguard[:2])           # 7 repaired of the 25, but only 2 of the 20 non-guard
        rc, _, res = self.S.run()
        p = res["primary"]
        self.assertEqual(p["repair_count"]["non_guard_dac1"], 2)
        self.assertEqual(p["repair_count"]["of_all_25"], 7)
        self.assertFalse(p["bars"]["1_repair_count_ge_5_of_20"])

    def test_not_confirmed_harm_branch(self):
        self.S.fix("N5A", self.S.nonguard[:6])
        t = self.S.nonguard[0]
        self.S.arms["N5A"][t]["nc"] = 0.0                      # a repaired token loses NC: paired dNC x100 = -0.05, so use several
        for t in self.S.nonguard[1:6]:
            self.S.arms["N5A"][t]["nc"] = 0.0
        rc, _, res = self.S.run()
        self.assertEqual(rc, 0)
        p = res["primary"]
        self.assertLess(p["no_harm_paired_x100"]["nc"]["mean"], -0.10)
        self.assertFalse(p["bars"]["2_no_harm_each_ge_-0.10_and_identity"])
        self.assertIn("harm", p["not_confirmed_branches_in_card_order"])

    def test_not_confirmed_cert_integrity_branch(self):
        self.S.fix("N5A", self.S.nonguard[:6])
        self.S.build["controls"]["C-CERT-B"] = {"pass": False}
        self.S.build["verdict"] = "PASS"                         # builder verdict claims PASS but the bar reads the C-CERT-B control itself
        rc, _, res = self.S.run()
        self.assertEqual(rc, 0)
        self.assertFalse(res["primary"]["bars"]["3_certificate_integrity"])
        self.assertIn("cert_integrity", res["primary"]["not_confirmed_branches_in_card_order"])

    def test_new_failure_voids_the_read_before_any_aggregate(self):
        self.S.fix("N5A", self.S.nonguard[:6])
        self.S.arms["N5A"]["k0100"]["dac"] = 0.0                # a DAC-1 N4-D token reads 0 in N5-A and is NOT in the L1-applied set
        rc, out, res = self.S.run()
        self.assertEqual(rc, 1)
        self.assertFalse(res["controls"]["C-ID-C"]["pass"])
        self.assertNotIn("primary", res)

    def test_paper_level_tiers(self):
        self.S.fix("N5A", self.S.l1)                          # all 25 repaired -> zero DAC rows 0
        rc, _, res = self.S.run()
        self.assertEqual(res["paper_level"]["N5A"]["level"], "PAPER LEVEL", res["paper_level"]["N5A"]["pdms"])
        self.setUp()
        self.S.fix("N5A", self.S.nonguard)                    # 20 of 25: 5 zero-DAC left -> DAC 99.75 -> NEAR
        rc, _, res = self.S.run()
        self.assertEqual(res["paper_level"]["N5A"]["level"], "NEAR")
        self.assertEqual(res["paper_level"]["N5A"]["zero_dac_tokens"], 5)
        self.setUp()
        rc, _, res = self.S.run()                              # nothing repaired -> DAC 98.75 -> BELOW
        self.assertEqual(res["paper_level"]["N5A"]["level"], "BELOW")
        self.assertEqual(res["paper_level"]["N4D_comparator"]["level"], "BELOW")

    def test_certificate_disabled_is_red_at_c_cert_and_no_aggregate(self):
        self.S.rep["N5A"] = [self.S.nonguard[0]]                # builder says certified, harness DAC 0
        rc, out, res = self.S.run()
        self.assertEqual(rc, 1)
        self.assertFalse(res["controls"]["C-CERT"]["pass"])
        self.assertNotIn("primary", res)
        self.assertNotIn("paper_level", res)
        self.assertIn("NO VERDICT", res["verdict"])

    def test_2e6_perturbation_is_red_at_c_id_c(self):
        self.S.arms["N5A"]["k0500"]["pdms"] += 2e-6
        rc, _, res = self.S.run()
        self.assertEqual(rc, 1)
        self.assertFalse(res["controls"]["C-ID-C"]["pass"])
        self.assertEqual(res["controls"]["C-ID-C"]["arms"]["N5A"]["tokens_over_tol"], 1)

    def test_dropped_row_is_red_at_c_cov(self):
        self.S.arms["N5A"].pop(self.S.tok[29])                  # the N5-A CSV is written one row short
        rc, _, res = self.S.run()
        self.assertEqual(rc, 1)
        self.assertFalse(res["controls"]["C-COV"]["pass"])

    def test_sign_flipped_arm_that_helps_is_red_at_c_reg(self):
        self.S.fix("N5A", self.S.nonguard[:6])
        self.S.fix("N5X", self.S.l1[:6])                        # N5-X repairs 6 of 25 > 10 %
        rc, _, res = self.S.run()
        self.assertEqual(rc, 1)
        self.assertFalse(res["controls"]["C-REG"]["pass"])
        self.assertNotIn("primary", res)

    def test_sign_flipped_arm_that_raises_pdms_is_red_at_c_reg(self):
        self.S.fix("N5A", self.S.nonguard[:6])
        for t in self.S.tok[1000:1050]:          # +0.03 x 50 tokens = +0.075 PDMS x100 > the 0.05 slack
            self.S.arms["N5X"][t]["pdms"] = 1.0
        rc, _, res = self.S.run()
        self.assertEqual(rc, 1)
        self.assertFalse(res["controls"]["C-REG"]["pass"])

    def test_pinned_file_edited_is_red_at_c_pin(self):
        self.S.fix("N5A", self.S.nonguard[:6])
        open(f"{self.S.d}/pinned.txt", "w").write("edited")
        rc, _, res = self.S.run()
        self.assertEqual(rc, 1)
        self.assertFalse(res["controls"]["C-PIN"]["pass"])
        self.assertIn("pinned.txt", res["controls"]["C-PIN"]["mismatch"])

    def test_an_old_families_run_pinned_is_red_at_c_pin(self):
        self.S.fix("N5A", self.S.nonguard[:6])
        AN.TIP_FAMILIES_SHA256 = "44c62807f1254e2ca4a41e4094097071b96bccaba9e754703ddf4a689e4d73a6"      # the real TIP; the tree pins and holds a different file
        rc, _, res = self.S.run()
        self.assertEqual(rc, 1)
        self.assertFalse(res["controls"]["C-PIN"]["pass"])
        self.assertFalse(res["controls"]["C-PIN"]["families_run_is_tip"])

    def test_clopper_pearson_values(self):
        lo, hi = AN.clopper_pearson(5, 20)
        self.assertAlmostEqual(lo, 0.0866, places=3)
        self.assertAlmostEqual(hi, 0.4910, places=3)
        self.assertEqual(AN.clopper_pearson(0, 20)[0], 0.0)
        self.assertEqual(AN.clopper_pearson(20, 20)[1], 1.0)

    def test_c_est_detects_a_wrong_draw_scheme(self):
        self.S.fix("N5A", self.S.nonguard[:6])
        open(f"{self.S.d}/pe/proxy_eval.py", "w").write("def boot(pt, tl):\n    return (0.0, 0.0, 0.0)\n")
        rc, _, res = self.S.run()
        self.assertFalse(res["controls"]["C-EST"]["pass"])


class TestFamread(unittest.TestCase):
    def _run(self, doc):
        d = tempfile.mkdtemp(dir=TMP)
        json.dump(doc, open(f"{d}/ci.json", "w"))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            AN.cmd_famread(SimpleNamespace(fam_ci=f"{d}/ci.json", out=f"{d}/o.json"))
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
        for k in ("lateral.heading_mae_deg", "lateral.curvature_mae_1pm", "lateral.yaw_rate_mae_degps", "tactical.maneuver_5way_collapsed.accuracy"):
            self.assertIn(k, r["reported_not_bars"])

    def test_a_ci_without_paired_diffs_is_refused_not_read_as_a_flag(self):
        with self.assertRaises(SystemExit):
            self._run({"frames": {"executed_tracked": {"families": {}}}})


# ----------------------------------------------------------------------------------------------------------- CHAIN
LEDGER_N4 = {"key": "TF-N4:aaaa:E3", "n": 46}


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

    def write_ledger(self, n=47, extra=None, drop_n4=False):
        key = f"TF-N5:{sha(self.card)}:E3"
        reads = ([] if drop_n4 else [LEDGER_N4]) + [{"key": key, "n": n}] + (extra or [])
        json.dump({"reads": reads}, open(f"{self.run}/ledger.json", "w"))

    def args(self, **kw):
        d = dict(run_dir=self.run, card=self.card, card_sha256=sha(self.card), pins_sha256=sha(self.pins), ledger=f"{self.run}/ledger.json", k=47, procs=2,
                 margin=0.05, iters=5, cert="full", centre_m=0.05, fallback_margin=0.005, damp=0.5, damp_cum=False, sign=1.0, sign_n5x=-1.0, min_free_virt_mb=1,
                 min_free_gb=0.0, retries=1, stop_after="", check_only=False, dry_run=False, skip_cache_listing=True)
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
        self.write_ledger(n=46)
        self.refused(self.args(), "n = 46")
        self.write_ledger(extra=[{"key": "other", "n": 48}])
        self.refused(self.args(), "n = 48")
        self.write_ledger(drop_n4=True)
        self.refused(self.args(), "n = 46) is not in the ledger")
        json.dump({"reads": [LEDGER_N4]}, open(f"{self.run}/ledger.json", "w"))
        self.refused(self.args(), "no entry")

    def test_pins_sha_edited_and_unpinned_files(self):
        self.refused(self.args(pins_sha256="0" * 64), "pins-sha256")
        open(f"{self.run}/n5/n5_nt_build.py", "a").write("#")
        self.refused(self.args(), "pinned files differ")
        self.mk_pins()
        a = self.args(pins_sha256=sha(self.pins))
        open(f"{self.run}/n5/extra_tool.py", "w").write("x")
        self.refused(a, "unpinned files")

    def test_old_families_run_is_refused(self):
        """the TF-N4 A1 failure: the tree holds (and pins) an older families_run.py; the guard compares with the TIP constant, not with the pins"""
        CH.TIP_FAMILIES_SHA256 = "44c62807f1254e2ca4a41e4094097071b96bccaba9e754703ddf4a689e4d73a6"
        self.refused(self.args(), "TIP version")
        # and a tree whose pins say TIP but whose file on disk is older
        CH.TIP_FAMILIES_SHA256 = sha(f"{self.run}/{CH.TIP_FAMILIES}")
        open(f"{self.run}/{CH.TIP_FAMILIES}", "w").write("older families_run")
        self.refused(self.args(), "pinned files differ")

    def test_procs_and_every_l1_flag(self):
        self.refused(self.args(procs=3), "--procs")
        for k, v in (("margin", 0.06), ("iters", 4), ("cert", "dac"), ("centre_m", 0.04), ("fallback_margin", 0.01), ("damp", 0.4), ("sign", -1.0), ("sign_n5x", 1.0)):
            self.refused(self.args(**{k: v}), "--" + k.replace("_", "-"))
        self.refused(self.args(margin=-0.05), "--margin")
        self.refused(self.args(damp_cum=True), "--damp-cum")

    def test_failing_reach_is_refused(self):
        CH.run_reach = lambda a, real=False, out="": (1, "ZZN5_REACH BAD N5A margin")
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
        argv = ["n5_nt_chain.py", "--run-dir", self.run, "--card", self.card, "--card-sha256", sha(self.card), "--pins-sha256", sha(self.pins), "--ledger",
                f"{self.run}/ledger.json", "--k", "47", "--dry-run", "--skip-cache-listing", "--min-free-gb", "0"]
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
        self.assertFalse(os.path.exists(f"{self.run}/n5run"))
        for frag in ("n5_nt_build.py reach", "n5_nt_build.py derive", "n5_nt_build.py merge", "n5_nt_build.py unch", "score_p4_a1.py --label refe_tfn5_n5a",
                     "score_p4_a1.py --label refe_tfn5_n5x", "n5_nt_analyze.py read", "transp_score.py", "famprep", "families_run.py", "--per-window", "fam_ci.py", "famread"):
            self.assertIn(frag, out, frag)
        self.assertNotIn("--mutate", out)
        self.assertNotIn("--smoke-allow-mutation", out)
        self.assertIn("N5A=", out)
        self.assertIn("families_run.py", out)
        self.assertIn("ZZN5_CHAIN DONE", out)

    def test_make_pins_refuses_a_tree_with_an_old_families_run(self):
        import n5_make_pins as MP
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
