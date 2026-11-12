"""TF-N4 chain tests: CPU, synthetic data, no navtest, no scipy.  Run (navsim venv, env as CHAIN_N4.md):  python -m unittest test_n4_chain -v

 REACH   margin / cap / sign / certificate read back from the BUILT repair; an argv-style change is refused (build CLI and chain CLI)
 UNCH    the file-level C-UNCH: a 2e-6 edit of one unchanged plan, a repair-all, a flagged-but-unchanged token each go RED
 READ    n4_nt_analyze: PRIMARY bars (CONFIRMED + every NOT-CONFIRMED branch), paper-level tiers, K in the Bonferroni key, the control exit BEFORE any
         N4-D aggregate (certificate disabled -> C-CERT; 2e-6 -> C-ID-C; dropped row -> C-COV; flipped arm that helps -> C-REG; pinned file edited -> C-PIN)
 CHAIN   guards refuse: card sha, ledger entry / n / higher n / missing N2-A1 entry, pins sha, an edited pinned file, an unpinned file, procs, margin,
         iters, a failing reach probe; --dry-run prints every stage's command and no mutation flag"""
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
sys.path.insert(0, HERE)
import n4_nt_analyze as AN      # noqa: E402
import n4_nt_build as BD        # noqa: E402
import n4_nt_chain as CH        # noqa: E402

A1 = os.environ.get("N4_A1_ANALYZE", os.path.join(HERE, "testdata", "tf_n1_analyze_a1.py"))
TMP = os.environ.get("N4_TMP") or None
COLS = ["", "token", "valid", "no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound", "comfort",
        "driving_direction_compliance", "score"]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


# ------------------------------------------------------------------------------------------------------------- REACH
class TestReach(unittest.TestCase):
    def test_card_arms_pass_and_read_back(self):
        for arm, spec in BD.CARD.items():
            r = BD.check_arm(arm, BD.make_repair(arm))
            self.assertTrue(r["ok"], r["bad"])
            for k in ("margin", "iters", "sign", "cert"):
                self.assertEqual(r["readback"][k], spec[k])
        self.assertAlmostEqual(BD.check_arm("N4X", BD.make_repair("N4X"))["probe"]["last_pose_y_after_cap_m"], -5 * 0.15, places=12)
        self.assertAlmostEqual(BD.check_arm("N4D", BD.make_repair("N4D"))["probe"]["last_pose_y_after_cap_m"], 5 * 0.15, places=12)

    def test_certificate_behaviour_reaches_the_loop(self):
        self.assertFalse(BD.reach_probe(BD.make_repair("N4D"))["dac1_nc_worse_accepted"])
        self.assertTrue(BD.reach_probe(BD.make_repair("N4DM"))["dac1_nc_worse_accepted"])

    def test_a_different_margin_cap_sign_cert_is_refused(self):
        import functools
        RP = BD._rp()
        bad = {"margin": functools.partial(RP.repair_target, margin=-0.05, iters=5, sign=1.0, cert=RP.certified),
               "cap": functools.partial(RP.repair_target, margin=0.05, iters=4, sign=1.0, cert=RP.certified),
               "sign": functools.partial(RP.repair_target, margin=0.05, iters=5, sign=-1.0, cert=RP.certified),
               "cert": functools.partial(RP.repair_target, margin=0.05, iters=5, sign=1.0, cert=RP.certified_dac)}
        for k, fn in bad.items():
            r = BD.check_arm("N4D", fn)
            self.assertFalse(r["ok"], k)
            self.assertTrue(any(k in b or (k == "cap" and "iters" in b) for b in r["bad"]), (k, r["bad"]))
        for m in ("margin_neg", "cert_off"):
            self.assertFalse(BD.check_arm("N4D", BD.make_repair("N4D", m))["ok"], m)

    def test_build_cli_refuses_mutation_without_smoke_flag(self):
        ns = SimpleNamespace(mutate="margin_neg", smoke_allow_mutation=False)
        with self.assertRaises(SystemExit) as c:
            BD.cmd_derive(ns)
        self.assertIn("REFUSING", str(c.exception))

    def test_chain_argv_has_no_mutation_flag(self):
        self.assertFalse(any("mutate" in x for x in CH.build_argv("R", "R", 0, 2)))


# -------------------------------------------------------------------------------------------------------------- UNCH
class TestUnch(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(dir=TMP)
        n = 8
        rng = np.random.default_rng(1)
        self.tok = np.array([f"t{i:02d}" for i in range(n)])
        self.plans = rng.normal(size=(n, 20, 3))
        np.savez(f"{self.d}/b_plans.npz", tokens=self.tok, logs=self.tok, plans20=self.plans)
        self.p8 = rng.normal(size=(n, 8, 3)).astype(np.float32)
        np.savez(f"{self.d}/b_seam.npz", token=self.tok, poses=self.p8)
        self.rep = {"N4D": ["t03"], "N4X": [], "N4DM": ["t03"]}
        self.write()

    def write(self, mut=None):
        for arm in BD.ARMS:
            P, S = self.plans.copy(), self.p8.copy()
            for t in self.rep[arm]:
                i = list(self.tok).index(t)
                P[i, 5, 0] += 0.1
                S[i, 2, 0] += 0.1
            if mut and mut[0] == arm:
                mut[1](P, S)
            np.savez(f"{self.d}/n4_plans_{arm}.npz", tokens=self.tok, logs=self.tok, plans20=P, arm=np.array(arm))
            np.savez(f"{self.d}/{BD.SEAM_LABEL[arm]}.npz", token=self.tok, poses=S)

    def chk(self):
        return BD.check_unch(self.d, f"{self.d}/b_plans.npz", f"{self.d}/b_seam.npz", self.rep)

    def test_green(self):
        self.assertTrue(self.chk()["pass"])

    def test_2e6_perturbation_of_one_unchanged_plan_is_red(self):
        def m(P, S):
            P[0, 3, 1] += 2e-6
        self.write(("N4D", m))
        r = self.chk()
        self.assertFalse(r["pass"])
        self.assertEqual(r["arms"]["N4D"]["unflagged_changed"], ["t00"])

    def test_repair_all_is_red(self):
        def m(P, S):
            P += 1e-12
        self.write(("N4X", m))
        self.assertFalse(self.chk()["pass"])

    def test_flagged_but_unchanged_is_red(self):
        self.rep["N4X"] = ["t01"]                           # flagged repaired, file never changed it
        self.assertFalse(self.chk()["pass"])


# ---------------------------------------------------------------------------------------------------------- analyzer
class Syn:
    """60 tokens, 6 logs, 2 known36 -> 58 primary; one DAC-0 token per log in N2R (the 6 'zero' tokens)"""
    N, G = 60, 6

    def __init__(self, tmp):
        self.d = tmp
        self.tok = [f"k{i:03d}" for i in range(self.N)]
        self.log = {t: f"log{i % self.G}" for i, t in enumerate(self.tok)}
        self.known = self.tok[-2:]
        self.zero = [self.tok[i] for i in range(self.G)]
        json.dump({"over_bar": {t: 1 for t in self.known}}, open(f"{tmp}/known.json", "w"))
        with gzip.open(f"{tmp}/tl.json.gz", "wt", encoding="utf-8") as fh:
            json.dump({"tokens": {t: {"log_name": self.log[t]} for t in self.tok}}, fh)
        os.makedirs(f"{tmp}/pe", exist_ok=True)
        open(f"{tmp}/pe/proxy_eval.py", "w").write(
            "import importlib.util\nimport sys\nspec = importlib.util.spec_from_file_location('a1', %r)\nA = importlib.util.module_from_spec(spec)\nspec.loader.exec_module(A)\n"
            "def boot(pt, tl):\n    return A.Boot(sorted(pt), tl).stat(pt)['_raw']\n" % A1)
        base = {t: dict(nc=1.0, dac=1.0, ddc=1.0, ep=0.9, ttc=1.0, c=1.0, pdms=0.9) for t in self.tok}
        for t in self.zero:
            base[t].update(dac=0.0, pdms=0.0)
        self.arms = {"N2R": base}
        self.rep = {"N4D": list(self.zero[:6]), "N4X": [], "N4DM": list(self.zero[:6])}
        self.arms["N4D"] = self.fixed(self.zero[:6])
        self.arms["N4DM"] = self.fixed(self.zero[:6])
        self.arms["N4X"] = {t: dict(v) for t, v in base.items()}
        self.arms["N2C"] = {t: dict(v) for t, v in base.items()}
        self.arms["N1C"] = {t: dict(v) for t, v in base.items()}
        self.arms["T"] = {t: dict(v) for t, v in base.items()}
        self.arms["human"] = {t: dict(v) for t, v in base.items()}
        self.build = {"verdict": "PASS", "controls": {"C-DACMAP": {"pass": True}}, "repaired": self.rep}
        self.unch = {"pass": True, "arms": {a: {"pass": True} for a in BD.ARMS}}
        self.pins_ok()

    def fixed(self, toks):
        out = {t: dict(v) for t, v in self.arms["N2R"].items()}
        for t in toks:
            out[t].update(dac=1.0, pdms=0.9)
        return out

    def pins_ok(self):
        open(f"{self.d}/pinned.txt", "w").write("x")
        json.dump({"files": {"pinned.txt": sha(f"{self.d}/pinned.txt")}}, open(f"{self.d}/pins.json", "w"))

    def write(self):
        for nm, rows in self.arms.items():
            with open(f"{self.d}/{nm}.csv", "w", encoding="utf-8", newline="") as fo:
                w = csv.writer(fo)
                w.writerow(COLS)
                for i, t in enumerate(self.tok):
                    r = rows[t]
                    w.writerow([i, t, True, r["nc"], r["dac"], r["ep"], r["ttc"], r["c"], r["ddc"], r["pdms"]])
        json.dump(self.build, open(f"{self.d}/build.json", "w"))
        json.dump(self.unch, open(f"{self.d}/unch.json", "w"))

    def run(self, k=45):
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
        self.S = Syn(tempfile.mkdtemp(dir=TMP))

    def test_confirmed_and_k_in_bonferroni_key(self):
        rc, out, res = self.S.run(k=45)
        self.assertEqual(rc, 0, res["controls"])
        self.assertEqual(res["primary"]["outcome"], "CONFIRMED", res["primary"])
        self.assertIn("ci_bonferroni_K45", res["primary"]["delta"])
        self.assertEqual(res["primary"]["repair"]["rate"], 1.0)
        self.assertEqual(res["primary"]["not_confirmed_branches_in_card_order"], [])

    def test_not_confirmed_repair_rate_branch(self):
        keep = self.S.zero[:3]                              # 3 of 6 repaired = 50 % < 75 %
        self.S.arms["N4D"] = self.S.fixed(keep)
        self.S.rep["N4D"] = keep
        rc, out, res = self.S.run()
        self.assertEqual(rc, 0)
        p = res["primary"]
        self.assertEqual(p["outcome"], "NOT CONFIRMED")
        self.assertFalse(p["bars"]["3_repair_rate_ge_0.75"])
        self.assertEqual(p["not_confirmed_branches_in_card_order"][0], "repair_rate_below_75")

    def test_not_confirmed_harm_branch(self):
        for t in self.S.tok[10:14]:                         # NC drops on 4 repaired-nothing tokens: dNC = -4*0.5/58*100 = -3.4
            self.S.arms["N4D"][t]["nc"] = 0.5
            self.S.arms["N4D"][t]["pdms"] = 0.5
            self.S.rep["N4D"].append(t) if False else None
        self.S.unch = {"pass": True, "arms": {a: {"pass": True} for a in BD.ARMS}}
        self.S.rep["N4D"] = self.S.zero[:6] + self.S.tok[10:14]          # declared as repaired so C-ID-C does not void: this is the HARM path
        for t in self.S.tok[10:14]:
            self.S.arms["N4D"][t]["dac"] = 1.0
        rc, out, res = self.S.run()
        self.assertEqual(rc, 0, res["controls"])
        self.assertFalse(res["primary"]["bars"]["4_no_harm_each_ge_-0.10"])
        self.assertIn("harm", res["primary"]["not_confirmed_branches_in_card_order"])

    def test_not_confirmed_effect_branch(self):
        s = self.S
        s.arms["N4D"] = {t: dict(v) for t, v in s.arms["N2R"].items()}
        s.arms["N4D"][s.zero[0]].update(dac=1.0, pdms=0.02)        # only 1 zero token repaired and worth 0.02: effect < 0.5, rate 1/6
        s.rep["N4D"] = [s.zero[0]]
        rc, out, res = s.run()
        self.assertEqual(rc, 0)
        self.assertEqual(res["primary"]["outcome"], "NOT CONFIRMED")
        self.assertIn("effect_below_bar_or_ci", res["primary"]["not_confirmed_branches_in_card_order"])

    def test_not_confirmed_ci_branch_when_one_log_carries_the_gain(self):
        s = self.S
        s.zero = [t for t in s.tok if s.log[t] == "log0"][:6]
        for arm in ("N2R",):
            for t in s.tok:
                s.arms[arm][t].update(dac=1.0, pdms=0.9)
            for t in s.zero:
                s.arms[arm][t].update(dac=0.0, pdms=0.0)
        for arm in ("N4X", "N2C", "N1C", "T", "human"):
            s.arms[arm] = {t: dict(v) for t, v in s.arms["N2R"].items()}
        s.arms["N4D"] = s.fixed(s.zero)
        s.arms["N4DM"] = s.fixed(s.zero)
        s.rep["N4D"], s.rep["N4DM"] = list(s.zero), list(s.zero)
        rc, out, res = s.run()
        self.assertEqual(rc, 0, res["controls"])
        self.assertFalse(res["primary"]["bars"]["2_ci_lo_gt_0"], res["primary"]["delta"])
        self.assertEqual(res["primary"]["outcome"], "NOT CONFIRMED")

    def test_paper_level_tiers(self):
        for pd_, want in ((0.97, "PAPER LEVEL"), (0.954, "NEAR"), (0.90, "BELOW")):
            s = Syn(tempfile.mkdtemp(dir=TMP))
            for arm in s.arms:
                for t in s.tok:
                    s.arms[arm][t].update(dac=1.0, pdms=pd_)
            s.rep = {"N4D": [], "N4X": [], "N4DM": []}
            s.build["repaired"] = s.rep
            rc, out, res = s.run()
            self.assertEqual(res["paper_level"]["N4D"]["level"], want, (pd_, res["paper_level"]["N4D"]["pdms"]))

    def _red(self, why):
        rc, out, res = self.S.run()
        self.assertEqual(rc, 1)
        self.assertFalse(res["controls"][why]["pass"], res["controls"])
        self.assertNotIn("primary", res)
        self.assertNotIn("tables", res)
        self.assertNotIn("ZZN4_VERDICT", out)
        self.assertIn("ZZN4_CONTROLS FAIL", out)

    def test_control_exit_before_aggregate_certificate_disabled_c_cert(self):
        self.S.rep["N4D"] = list(self.S.zero)               # builder claims all 6 certified ...
        self.S.arms["N4D"] = {t: dict(v) for t, v in self.S.arms["N2R"].items()}      # ... harness DAC still 0 (the first iterate was accepted)
        self._red("C-CERT")

    def test_control_c_id_c_2e6_on_an_unchanged_token(self):
        self.S.arms["N4D"][self.S.tok[20]]["pdms"] += 2e-6
        self._red("C-ID-C")

    def test_control_c_cov_dropped_row(self):
        self.S.write()
        self.S.tok = self.S.tok[:-1]
        self._red("C-COV")

    def test_control_c_reg_flipped_arm_that_helps(self):
        self.S.arms["N4X"] = self.S.fixed(self.S.zero[:6])
        self.S.rep["N4X"] = list(self.S.zero[:6])
        rc, out, res = self.S.run()
        self.assertEqual(rc, 1)
        self.assertFalse(res["controls"]["C-REG"]["pass"])
        self.assertNotIn("primary", res)

    def test_control_c_pin_pinned_file_edited(self):
        open(f"{self.S.d}/pinned.txt", "w").write("edited")
        self._red("C-PIN")

    def test_control_c_build_and_c_unch_failures(self):
        self.S.build["verdict"] = "FAIL"
        self._red("C-BUILD")
        self.S = Syn(tempfile.mkdtemp(dir=TMP))
        self.S.unch = {"pass": False, "arms": {"N4D": {"pass": False}}}
        self._red("C-UNCH")

    def test_famread_guard(self):
        def d(p):
            return {"diff_B_minus_A": {"point": p, "lo": None, "hi": None}}
        fc = {"frames": {"executed_tracked": {"families": {"longitudinal": {"speed_bias_mps": d(0.01), "speed_mae_mps": d(0.2)},
                                                           "lateral": {"cross_mae_m": d(0.0), "heading_mae_deg": d(0.0), "curvature_mae_1pm": d(0.0)}}}}}
        p = f"{self.S.d}/fc.json"
        json.dump(fc, open(p, "w"))
        a = SimpleNamespace(fam_ci=p, out=f"{self.S.d}/fam.json")
        with contextlib.redirect_stdout(io.StringIO()):
            AN.cmd_famread(a)
        r = json.load(open(a.out))
        self.assertTrue(r["guard"]["flag"])                  # speed MAE diff 0.2 > 0.15
        self.assertTrue(r["guard"]["checks"]["speed_mae_diff_mps"]["exceeded"])
        self.assertFalse(r["guard"]["checks"]["speed_bias_diff_mps"]["exceeded"])


# -------------------------------------------------------------------------------------------------------------- CHAIN
class TestChain(unittest.TestCase):
    def setUp(self):
        self.run = tempfile.mkdtemp(dir=TMP)
        self._reach = CH.run_reach
        CH.run_reach = lambda run: (0, "stub")
        for rel in CH.REQUIRED:
            if not os.path.isabs(rel) and not rel.startswith("F:"):
                p = os.path.join(self.run, rel)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                open(p, "w").write(rel)
        self.token_log_rel = CH.TOKEN_LOG
        self.card = os.path.join(self.run, "card.md")
        open(self.card, "w").write("card")
        self.csha = sha(self.card)
        self.ledger = os.path.join(self.run, "ledger.json")
        self.write_ledger()
        self.write_pins()
        CH.REQUIRED_SAVED = CH.REQUIRED
        CH.REQUIRED = tuple(r for r in CH.REQUIRED if not r.startswith("F:"))

    def tearDown(self):
        CH.run_reach = self._reach
        CH.REQUIRED = CH.REQUIRED_SAVED

    def write_ledger(self, entries=None):
        reads = entries if entries is not None else [{"key": "TF-N2-A1:xx:E3", "n": 44}, {"key": f"TF-N4:{self.csha}:E3", "n": 45}]
        json.dump({"reads": reads}, open(self.ledger, "w"))

    def write_pins(self):
        files = {}
        for d, _, fs in os.walk(self.run):
            for f in fs:
                rel = os.path.relpath(os.path.join(d, f), self.run).replace(os.sep, "/")
                if rel.startswith(("w/", "n2/", "n4/", "inputs/")):
                    files[rel] = sha(os.path.join(d, f))
        self.pp = os.path.join(self.run, "n4", "N4_PINS.json")
        files.pop("n4/N4_PINS.json", None)
        os.makedirs(os.path.dirname(self.pp), exist_ok=True)
        json.dump({"files": files, "metric_cache_listing_sha256": "x"}, open(self.pp, "w"))
        self.psha = sha(self.pp)

    def args(self, **kw):
        d = dict(run_dir=self.run, card=self.card, card_sha256=self.csha, pins_sha256=self.psha, ledger=self.ledger, k=45, procs=2, margin=0.05, iters=5,
                 sign_n4x=-1.0, skip_cache_listing=True)
        d.update(kw)
        return SimpleNamespace(**d)

    def refused(self, why, **kw):
        with self.assertRaises(SystemExit) as c:
            CH.guards(self.args(**kw))
        self.assertIn("REFUSING", str(c.exception))
        self.assertIn(why, str(c.exception))

    def test_guards_pass(self):
        self.assertEqual(CH.guards(self.args())["k"], 45)

    def test_refusals(self):
        self.refused("--card-sha256", card_sha256="0" * 64)
        self.refused("--procs", procs=3)
        self.refused("--margin", margin=0.06)
        self.refused("--iters", iters=6)
        self.refused("--sign-n4x", sign_n4x=1.0)
        self.refused("--pins-sha256", pins_sha256="1" * 64)

    def test_ledger_refusals(self):
        self.write_ledger([{"key": "TF-N2-A1:xx:E3", "n": 44}])
        self.refused("no entry")
        self.write_ledger([{"key": "TF-N2-A1:xx:E3", "n": 44}, {"key": f"TF-N4:{self.csha}:E3", "n": 46}])
        self.refused("n = 46")
        self.write_ledger([{"key": f"TF-N4:{self.csha}:E3", "n": 45}])
        self.refused("TF-N2-A1 entry")
        self.write_ledger([{"key": "TF-N2-A1:xx:E3", "n": 44}, {"key": f"TF-N4:{self.csha}:E3", "n": 45}, {"key": "Z:1:E3", "n": 47}])
        self.refused("another read landed")

    def test_a_pinned_file_edited_is_refused(self):
        open(os.path.join(self.run, "n4", "n4_repair.py"), "w").write("edited")
        self.refused("pinned files differ")

    def test_an_unpinned_file_is_refused(self):
        open(os.path.join(self.run, "n4", "extra.py"), "w").write("x")
        self.refused("unpinned files")

    def test_reach_failure_is_refused(self):
        CH.run_reach = lambda run: (1, "margin: built -0.05 != card 0.05")
        self.refused("reachability")

    def test_dry_run_prints_every_stage_command(self):
        argv = ["n4_nt_chain.py", "--run-dir", self.run, "--card", self.card, "--card-sha256", self.csha, "--pins-sha256", self.psha, "--ledger", self.ledger,
                "--skip-cache-listing", "--dry-run"]
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
        for needle in ("n4_nt_build.py derive", "n4_nt_build.py merge", "n4_nt_build.py unch", "score_p4_a1.py --label refe_tfn4_n4d", "--label refe_tfn4_n4dm",
                       "--label refe_tfn4_n4x", "n4_nt_analyze.py read", "transp_score.py", "--metric-cache-root", "famprep", "families_run.py", "--per-window",
                       "fam_ci.py", "n4_nt_analyze.py famread", "--shard 0/2", "--shard 1/2"):
            self.assertIn(needle, out, needle)
        self.assertNotIn("--mutate", out)
        self.assertFalse(os.path.exists(os.path.join(self.run, "n4run")))


if __name__ == "__main__":
    unittest.main()
