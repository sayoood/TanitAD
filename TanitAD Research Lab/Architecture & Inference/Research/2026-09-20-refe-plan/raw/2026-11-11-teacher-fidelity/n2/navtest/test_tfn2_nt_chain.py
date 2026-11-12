"""Tests for the TF-N2 navtest chain and readout (run with CUDA_VISIBLE_DEVICES=-1; `python -m unittest` with a python that has numpy).
Every refusal / control test builds a tiny fixture, flips ONE thing and requires the guard or the control to go RED, and the unmodified fixture
must PASS (a control that must read GREEN), so a refusal is never an artefact of a broken fixture. Mutation arms: each control of the card has one."""
import csv
import glob
import gzip
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
CHAIN = os.path.join(HERE, "tfn2_nt_chain.py")
ANALYZE = os.path.join(HERE, "tfn2_nt_analyze.py")
sys.path.insert(0, HERE)
import tfn2_nt_chain as C  # noqa: E402

K = 43
EVAL_DIR = os.environ.get("TFN2_NT_EVAL", "F:/tanitad_scratch/tfn2_nt/smoke_run/w/pk/eval")      # holds proxy_eval.py (C-EST)
N1A1 = os.path.join(os.environ.get("TFN2_NT_RUN", "F:/tanitad_scratch/tfn2_nt/smoke_run"), "n2", "tf_n1_analyze_a1.py")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def w(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)


class Fx:
    """a fake run tree: every REQUIRED file present and pinned (the absolute token export is pinned by its real bytes)"""

    def __init__(self, root):
        self.run = os.path.join(root, "run")
        for f in C.REQUIRED:
            if os.path.isabs(f):
                continue
            w(os.path.join(self.run, f), f"content of {f}\n")
        self.write_pins()
        self.card = os.path.join(root, "PREREG_TF_N2.md")
        w(self.card, "the card\n")
        self.sha = sha(self.card)
        self.ledger = os.path.join(root, "ledger_K.json")
        self.write_ledger()

    def write_pins(self, drop=None):
        files = {}
        for f in C.REQUIRED:
            p = f if os.path.isabs(f) else os.path.join(self.run, f)
            files[f] = sha(p)
        if drop:
            del files[drop]
        w(os.path.join(self.run, "n2", "N2_PINS.json"), json.dumps({"files": files}))
        self.pins_sha = sha(os.path.join(self.run, "n2", "N2_PINS.json"))

    def write_ledger(self, mutate=None):
        reads = [{"key": "X:1:E3", "n": 40}, {"key": "TF-N1-A1:aaaa:E3", "n": 42}, {"key": f"TF-N2:{self.sha}:E3", "n": K}]
        if mutate:
            reads = mutate(reads)
        w(self.ledger, json.dumps({"reads": reads}))

    def run_chain(self, **over):
        a = dict(run_dir=self.run, card=self.card, card_sha256=self.sha, pins_sha256=self.pins_sha, ledger=self.ledger, procs="2")
        a.update(over)
        argv = [sys.executable, CHAIN, "--check-only"] + [x for k, v in a.items() for x in (f"--{k.replace('_', '-')}", str(v))]
        return subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", stdin=subprocess.DEVNULL, env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1"))


@unittest.skipUnless(os.path.exists(C.TOKEN_LOG), "the navtest token export is not on this machine")
class ChainRefusals(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="tfn2nt_", dir=os.environ.get("TFN2_NT_TMP"))
        self.fx = Fx(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def refused(self, r, needle):
        self.assertNotEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("REFUSING", r.stderr + r.stdout)
        self.assertIn(needle, r.stderr + r.stdout)

    def test_control_good_fixture_passes(self):
        r = self.fx.run_chain()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("ZZN2 GUARDS PASS", r.stdout)

    def test_wrong_card_sha(self):
        self.refused(self.fx.run_chain(card_sha256="0" * 64), "card-sha256")

    def test_edited_card(self):
        with open(self.fx.card, "a", encoding="utf-8") as f:
            f.write("one more line after the hash\n")
        self.refused(self.fx.run_chain(), "card-sha256")

    def test_ledger_entry_missing(self):
        self.fx.write_ledger(lambda r: [x for x in r if not x["key"].startswith("TF-N2")])
        self.refused(self.fx.run_chain(), "no entry")

    def test_ledger_entry_for_a_different_sha(self):
        self.fx.write_ledger(lambda r: [dict(x, key=x["key"].replace(self.fx.sha, "f" * 64)) for x in r])
        self.refused(self.fx.run_chain(), "no entry")

    def test_ledger_n_differs_from_k(self):
        self.fx.write_ledger(lambda r: [dict(x, n=44) if x["key"].startswith("TF-N2") else x for x in r])
        self.refused(self.fx.run_chain(), "n = 44")

    def test_a_later_read_landed(self):
        self.fx.write_ledger(lambda r: r + [{"key": "OTHER:1:E3", "n": 44}])
        self.refused(self.fx.run_chain(), "another read landed")

    def test_n1_a1_entry_missing(self):
        self.fx.write_ledger(lambda r: [x for x in r if not x["key"].startswith("TF-N1-A1")])
        self.refused(self.fx.run_chain(), "TF-N1-A1 entry")

    def test_procs_over_two(self):
        self.refused(self.fx.run_chain(procs=3), "at most 2 processes")

    def test_pins_sha_mismatch(self):
        self.refused(self.fx.run_chain(pins_sha256="1" * 64), "N2_PINS.json missing or its sha256")

    def test_roll_script_edited_after_pinning(self):
        w(os.path.join(self.fx.run, "w/pk/eval/tfn2_nt_roll.py"), "edited after the pins\n")
        self.refused(self.fx.run_chain(), "pinned files differ")

    def test_patched_engine_hook_edited(self):
        w(os.path.join(self.fx.run, "w/pk/refe/build_teacher_rollouts.py"), "unpatched\n")
        self.refused(self.fx.run_chain(), "pinned files differ")

    def test_build_script_edited(self):
        w(os.path.join(self.fx.run, "n2/tfn2_nt_build.py"), "edited\n")
        self.refused(self.fx.run_chain(), "pinned files differ")

    def test_harness_file_edited(self):
        w(os.path.join(self.fx.run, "w/pk/eval/harness/score_e.py"), "edited harness\n")
        self.refused(self.fx.run_chain(), "pinned files differ")

    def test_banked_rollout_swapped(self):
        w(os.path.join(self.fx.run, "inputs/rollouts/rows_s2of4.jsonl.xz"), "another teacher\n")
        self.refused(self.fx.run_chain(), "pinned files differ")

    def test_comparator_csv_swapped(self):
        w(os.path.join(self.fx.run, "inputs/n1_csv/refe_a1_tfn1_c.csv"), "another comparator\n")
        self.refused(self.fx.run_chain(), "pinned files differ")

    def test_unpinned_py_file_under_w(self):
        w(os.path.join(self.fx.run, "w/pk/eval/harness/w3e/code/sneaky.py"), "x = 1\n")
        self.refused(self.fx.run_chain(), "unpinned files")

    def test_unpinned_json_under_n2(self):
        w(os.path.join(self.fx.run, "n2/override.json"), "{}")
        self.refused(self.fx.run_chain(), "unpinned files")

    def test_pins_file_missing_a_required_entry(self):
        self.fx.write_pins(drop="w/p5r/p5r_labels.py")
        self.refused(self.fx.run_chain(), "does not pin w/p5r/p5r_labels.py")

    def test_pins_file_missing_the_roll_hook(self):
        self.fx.write_pins(drop="w/pk/refe/build_teacher_rollouts.py")
        self.refused(self.fx.run_chain(), "does not pin w/pk/refe/build_teacher_rollouts.py")


class CardConstants(unittest.TestCase):
    """the candidate's levers are CONSTANTS of the code the card pins; nothing on the command line can change them"""

    def test_chain_jerk_constants(self):
        self.assertEqual(C.JERK, {"R0": 1.0, "RJ5": 5.0})

    def test_roll_arm_table_is_the_two_arms_only(self):
        src = open(os.path.join(HERE, "tfn2_nt_roll.py"), encoding="utf-8").read()
        self.assertIn('ARM_JERK = {"R0": None, "RJ5": 5.0}', src)
        self.assertEqual(len(re.findall(r'ap\.add_argument\("--(?:pos|route|policy)', src)), 0)     # no knob flag exists: only the two arms

    def test_build_lambda_is_p5r_lambda_star(self):
        src = open(os.path.join(HERE, "tfn2_nt_build.py"), encoding="utf-8").read()
        self.assertIn("LAM = 3.0", src)

    def test_chain_k_default(self):
        src = open(CHAIN, encoding="utf-8").read()
        self.assertIn('ap.add_argument("--k", type=int, default=43)', src)

    def test_card_pins_the_same_bars_as_the_analysis(self):
        card = os.path.join(HERE, "PREREG_TF_N2.md")
        if not os.path.exists(card):
            self.skipTest("card not written")
        A = import_analyze()
        txt = open(card, encoding="utf-8").read()
        for needle in (f"+{A.BAR['confirm_delta']}", str(A.BAR["paper_pdms"]), str(A.BAR["paper_dac"]), str(A.BAR["near_pdms"]), "K = 43"):
            self.assertIn(needle, txt)


class Compare(unittest.TestCase):
    def _csvs(self, tmp, n=12146, bump=None, drop=False, sub=0):
        head = ",token,valid,no_at_fault_collisions,drivable_area_compliance,driving_direction_compliance,ego_progress,time_to_collision_within_bound,comfort,score\n"
        a, b = [head], [head]
        for i in range(n):
            row = f"{i},tok{i},True,1.0,1.0,1.0,0.5,1.0,1.0,0.8\n"
            b.append(row)
            if drop and i == 0:
                continue
            if sub and i >= sub:
                continue
            a.append(row.replace(",0.5,", f",{0.5 + bump},") if (bump and i == 7) else row)
        pa, pb = os.path.join(tmp, "a.csv"), os.path.join(tmp, "b.csv")
        w(pa, "".join(a))
        w(pb, "".join(b))
        return pa, pb

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="tfn2cmp_", dir=os.environ.get("TFN2_NT_TMP"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_identical_passes(self):
        self.assertTrue(C.compare_csv(*self._csvs(self.tmp), 12146)["pass"])

    def test_breach_over_tolerance_fails(self):
        r = C.compare_csv(*self._csvs(self.tmp, bump=2e-6), 12146)
        self.assertFalse(r["pass"])
        self.assertEqual(r["tokens_over_tol"], 1)

    def test_breach_under_tolerance_passes(self):
        self.assertTrue(C.compare_csv(*self._csvs(self.tmp, bump=5e-7), 12146)["pass"])

    def test_missing_token_fails(self):
        self.assertFalse(C.compare_csv(*self._csvs(self.tmp, drop=True), 12146)["pass"])

    def test_short_set_fails_in_full_mode(self):
        self.assertFalse(C.compare_csv(*self._csvs(self.tmp, sub=200), 12146)["pass"])

    def test_subset_mode_passes_for_exactly_200_and_fails_for_199(self):
        self.assertTrue(C.compare_csv(*self._csvs(self.tmp, sub=200), 200, subset=True)["pass"])
        self.assertFalse(C.compare_csv(*self._csvs(self.tmp, sub=199), 200, subset=True)["pass"])

    def test_subset_mode_breach_fails(self):
        self.assertFalse(C.compare_csv(*self._csvs(self.tmp, bump=2e-6, sub=200), 200, subset=True)["pass"])


class R0Identity(unittest.TestCase):
    def _rows(self, n=240, bump=0.0, knob=1.0):
        bank = {f"t{i}": {"traj20": [[0.1 * j + i, 0.01 * j, 0.001 * j] for j in range(20)]} for i in range(n)}
        rolled = {t: {"traj20": [list(p) for p in r["traj20"]], "knobs": {"positive_jerk_limit_when_nonnegative_acc": knob}} for t, r in bank.items()}
        if bump:
            rolled["t3"]["traj20"][5][1] += bump
        return rolled, bank, sorted(bank)

    def test_identical_passes(self):
        self.assertTrue(C.r0_identity(*self._rows())["PASS"])

    def test_breach_in_xy_fails(self):
        self.assertFalse(C.r0_identity(*self._rows(bump=2e-6))["PASS"])

    def test_breach_in_heading_fails(self):
        r, b, t = self._rows()
        r["t9"]["traj20"][19][2] += 1e-5
        self.assertFalse(C.r0_identity(r, b, t)["PASS"])

    def test_sub_tolerance_passes(self):
        self.assertTrue(C.r0_identity(*self._rows(bump=5e-7))["PASS"])

    def test_missing_token_fails(self):
        r, b, t = self._rows()
        del r["t0"]
        self.assertFalse(C.r0_identity(r, b, t)["PASS"])

    def test_token_not_in_bank_fails(self):
        r, b, t = self._rows()
        del b["t1"]
        self.assertFalse(C.r0_identity(r, b, t)["PASS"])

    def test_row_rolled_with_the_jerk_override_fails(self):
        """the identity arm must be the REGISTERED build: a row carrying jerk 5.0 is not an R0 row"""
        self.assertFalse(C.r0_identity(*self._rows(knob=5.0))["PASS"])

    def test_empty_subset_fails(self):
        self.assertFalse(C.r0_identity({}, {}, [])["PASS"])

    def test_subset_rule_is_stable(self):
        toks = [hashlib.sha1(str(i).encode()).hexdigest()[:16] for i in range(12146)]
        s = C.r0_subset(toks)
        self.assertEqual(s, C.r0_subset(list(reversed(toks))))
        self.assertTrue(200 <= len(s) <= 300, len(s))


class StripErrors(unittest.TestCase):
    def test_error_rows_are_dropped_so_a_resume_retries_them(self):
        d = tempfile.mkdtemp(prefix="tfn2se_", dir=os.environ.get("TFN2_NT_TMP"))
        try:
            p = os.path.join(d, "rows_s0of2.jsonl")
            w(p, '{"token": "a", "traj20": [1]}\n{"token": "b", "error": "KNOB-NOT-REACHED"}\n{"token": "c", "traj2')
            self.assertEqual(C.strip_error_rows(os.path.join(d, "rows_s*of*.jsonl")), 1)
            self.assertEqual(open(p, encoding="utf-8").read(), '{"token": "a", "traj20": [1]}\n')
        finally:
            shutil.rmtree(d, ignore_errors=True)


@unittest.skipUnless(os.path.exists(N1A1) and os.path.exists(os.path.join(EVAL_DIR, "proxy_eval.py")), "needs the prepped run tree (F: smoke_run)")
class ReadControls(unittest.TestCase):
    """the readout on a synthetic 300-token navtest stand-in: GREEN on the good fixture, RED for each mutated control, and the verdict rules"""
    N, LOGS = 300, 12

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="tfn2rd_", dir=os.environ.get("TFN2_NT_TMP"))
        shutil.copyfile(N1A1, os.path.join(self.tmp, "tf_n1_analyze_a1.py"))
        shutil.copyfile(ANALYZE, os.path.join(self.tmp, "tfn2_nt_analyze.py"))
        rnd = random.Random(7)
        self.toks = [f"tok{i:04d}" for i in range(self.N)]
        self.known = set(self.toks[:5])
        w(os.path.join(self.tmp, "tlog.json"), "")
        with gzip.open(os.path.join(self.tmp, "tlog.json.gz"), "wt", encoding="utf-8") as f:
            json.dump({"tokens": {t: {"log_name": f"log{i % self.LOGS}"} for i, t in enumerate(self.toks)}}, f)
        w(os.path.join(self.tmp, "known36.json"), json.dumps({"over_bar": {t: 1.0 for t in self.known}}))
        base = {t: (0.6 + 0.35 * rnd.random(), 1.0 if rnd.random() > 0.02 else 0.0) for t in self.toks}      # (pdms, dac)
        self.arm = {"N1C": {t: base[t] for t in self.toks}, "T": {t: (max(0.0, base[t][0] - 0.04), base[t][1]) for t in self.toks},
                    "N2R": {t: (min(1.0, base[t][0] + 0.015), base[t][1]) for t in self.toks},
                    "N2C": {t: (min(1.0, base[t][0] + 0.005), base[t][1]) for t in self.toks},
                    "N2W": {t: (base[t][0] * 0.5, 0.0) for t in self.toks}, "human": {t: (base[t][0], 1.0) for t in self.toks}}
        self.paths = {}
        for nm in self.arm:
            self.write_csv(nm)
        self.pins = os.path.join(self.tmp, "pins.json")
        w(self.pins, json.dumps({"files": {"tf_n1_analyze_a1.py": sha(os.path.join(self.tmp, "tf_n1_analyze_a1.py"))}}))
        self.good = {"build": {"verdict": "PASS", "controls": {"C-BUILD-COV": {"pass": True}}}, "r0": {"PASS": True, "n": 240, "max_abs_m": 0.0, "bar": "x"},
                     "cidc": {"pass": True, "tokens_rescored": 200, "tokens_over_tol": 0, "token_set_mismatch": 0}}
        self.write_ctl()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write_csv(self, nm, mutate=None):
        head = ",token,valid,no_at_fault_collisions,drivable_area_compliance,driving_direction_compliance,ego_progress,time_to_collision_within_bound,comfort,score\n"
        rows = [head]
        for i, t in enumerate(self.toks):
            p, d = self.arm[nm][t]
            if mutate:
                p, d = mutate(t, p, d)
            rows.append(f"{i},{t},True,1.0,{d},1.0,0.9,1.0,1.0,{p!r}\n")
        self.paths[nm] = os.path.join(self.tmp, f"{nm}.csv")
        w(self.paths[nm], "".join(rows))

    def write_ctl(self, **over):
        d = dict(self.good)
        d.update(over)
        w(os.path.join(self.tmp, "build.json"), json.dumps(d["build"]))
        w(os.path.join(self.tmp, "r0.json"), json.dumps(d["r0"]))
        w(os.path.join(self.tmp, "cidc.json"), json.dumps(d["cidc"]))

    def read(self, identity=None):
        a = [sys.executable, os.path.join(self.tmp, "tfn2_nt_analyze.py"), "read", "--harness-csv"] + [f"{k}={self.paths[k]}" for k in ("N2R", "N2C", "N2W", "N1C", "T")] + [
            "--identity-csv", identity or self.paths["T"], "--human-csv", self.paths["human"], "--known36", os.path.join(self.tmp, "known36.json"),
            "--build-report", os.path.join(self.tmp, "build.json"), "--ctl-r0", os.path.join(self.tmp, "r0.json"), "--cidc", os.path.join(self.tmp, "cidc.json"),
            "--pins", self.pins, "--pin-root", self.tmp, "--proxy-eval", EVAL_DIR, "--prereg-sha256", "x", "--token-log", os.path.join(self.tmp, "tlog.json.gz"),
            "--expect-n", str(self.N), "--expect-primary", str(self.N - len(self.known)), "--out", os.path.join(self.tmp, "out.json")]
        r = subprocess.run(a, capture_output=True, text=True, encoding="utf-8", stdin=subprocess.DEVNULL, env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1"))
        out = json.load(open(os.path.join(self.tmp, "out.json"), encoding="utf-8")) if os.path.exists(os.path.join(self.tmp, "out.json")) else {}
        return r, out

    def red(self, key):
        r, out = self.read()
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertFalse(out["controls"][key]["pass"], out["controls"])
        self.assertNotIn("primary", out)                                 # no aggregate exists when a control fails
        self.assertIn("NO VERDICT", out["verdict"])

    def test_control_good_fixture_is_green_and_confirmed(self):
        r, out = self.read()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertTrue(all(c["pass"] for c in out["controls"].values()), out["controls"])
        self.assertEqual(out["primary"]["outcome"], "CONFIRMED")
        self.assertEqual(out["K"], 43)
        self.assertIn("ci_bonferroni_K43", out["primary"]["delta"])

    def test_C_ID_H_red_on_a_1e6_breach_of_T(self):
        ident = os.path.join(self.tmp, "T_banked.csv")
        w(ident, open(self.paths["T"], encoding="utf-8").read().replace(f"{self.arm['T']['tok0100'][0]!r}", f"{self.arm['T']['tok0100'][0] + 2e-6!r}", 1))
        r, out = self.read(identity=ident)
        self.assertEqual(r.returncode, 1)
        self.assertFalse(out["controls"]["C-ID-H"]["pass"])
        self.assertNotIn("primary", out)

    def test_C_ID_C_red(self):
        self.write_ctl(cidc={"pass": False})
        self.red("C-ID-C")

    def test_C_R0_red(self):
        self.write_ctl(r0={"PASS": False})
        self.red("C-R0")

    def test_C_BUILD_red(self):
        self.write_ctl(build={"verdict": "FAIL", "controls": {}})
        self.red("C-BUILD")

    def test_C_REG_red_when_the_regression_arm_does_not_regress(self):
        self.write_csv("N2W", mutate=lambda t, p, d: (self.arm["N1C"][t][0] - 0.001, d))
        self.red("C-REG")

    def test_C_COV_red_when_a_token_is_missing_from_an_arm(self):
        self.write_csv("N2R")
        lines = open(self.paths["N2R"], encoding="utf-8").read().splitlines(True)
        w(self.paths["N2R"], "".join(lines[:-1]))
        self.red("C-COV")

    def test_C_PIN_red_when_a_pinned_file_changes(self):
        with open(os.path.join(self.tmp, "tf_n1_analyze_a1.py"), "a", encoding="utf-8") as f:
            f.write("# edited after pinning\n")
        r, out = self.read()
        self.assertEqual(r.returncode, 1)
        self.assertFalse(out["controls"]["C-PIN"]["pass"])

    def test_not_confirmed_when_the_candidate_equals_the_comparator(self):
        self.arm["N2R"] = dict(self.arm["N1C"])
        self.write_csv("N2R")
        r, out = self.read()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(out["primary"]["outcome"], "NOT CONFIRMED")
        self.assertFalse(out["primary"]["parts"]["ci_lo_gt_0"])

    def test_not_confirmed_when_progress_is_bought_with_dac(self):
        """a +1.5 PDMS gain that costs drivable-area compliance is NOT the lever: the DAC part of the rule must bite"""
        self.write_csv("N2R", mutate=lambda t, p, d: (p, 0.0 if (int(t[3:]) % 20 == 7) else d))
        r, out = self.read()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertLess(out["primary"]["dac_delta"]["mean"], -0.25)
        self.assertFalse(out["primary"]["parts"]["dac_delta_ge_-0.25"])
        self.assertEqual(out["primary"]["outcome"], "NOT CONFIRMED")

    def test_cert2_order_and_none_pass_fallback(self):
        A = import_analyze()
        arms = {a: {"t": {"dac": 0.0}} for a in ("N2R", "N2C", "N1C", "T")}
        self.assertEqual(A.cert2(arms, "t"), ("N2R", True))
        arms["N1C"]["t"]["dac"] = 1.0
        self.assertEqual(A.cert2(arms, "t"), ("N1C", False))
        arms["N2C"]["t"]["dac"] = 1.0
        self.assertEqual(A.cert2(arms, "t"), ("N2C", False))
        arms["N2R"]["t"]["dac"] = 1.0
        self.assertEqual(A.cert2(arms, "t"), ("N2R", False))


def import_analyze():
    """the readout is imported from a temp copy next to the pinned estimator module, as it sits in the run tree's n2/"""
    d = tempfile.mkdtemp(prefix="tfn2imp_", dir=os.environ.get("TFN2_NT_TMP"))
    shutil.copyfile(N1A1, os.path.join(d, "tf_n1_analyze_a1.py"))
    shutil.copyfile(ANALYZE, os.path.join(d, "tfn2_nt_analyze.py"))
    sys.path.insert(0, d)
    import importlib
    sys.modules.pop("tfn2_nt_analyze", None)
    return importlib.import_module("tfn2_nt_analyze")


class Guard(unittest.TestCase):
    """the PROPOSED longitudinal guard: GREEN at the navtrain reference, RED for each of its four checks when mutated"""

    def setUp(self):
        self.A = import_analyze()
        self.n1c = {"speed_bias_mps": 0.357, "speed_mae_mps": 0.699, "accel_mae_mps2": 0.456}
        self.n2r = {"speed_bias_mps": 0.641, "speed_mae_mps": 0.876, "accel_mae_mps2": 0.520}      # navtrain RJ5.R (MEASURED, TFN2_REROLL_RESULT.json)

    def test_navtrain_reference_does_not_flag(self):
        self.assertFalse(self.A.guard_eval(self.n2r, self.n1c)["flag"])

    def test_each_check_flags_when_mutated(self):
        for key, bump in (("speed_bias_mps", 0.40), ("speed_mae_mps", 0.20), ("accel_mae_mps2", 0.12)):
            n2 = dict(self.n2r)
            n2[key] += bump
            self.assertTrue(self.A.guard_eval(n2, self.n1c)["flag"], key)
        n2 = dict(self.n2r, speed_bias_mps=1.05)
        n1 = dict(self.n1c, speed_bias_mps=0.9)                      # the difference is small, the absolute bias is over 1.0
        g = self.A.guard_eval(n2, n1)
        self.assertTrue(g["checks"]["speed_bias_abs_mps"]["exceeded"])
        self.assertTrue(g["flag"])


if __name__ == "__main__":
    unittest.main()
