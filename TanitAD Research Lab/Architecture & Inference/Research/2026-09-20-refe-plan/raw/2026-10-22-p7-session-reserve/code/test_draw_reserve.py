#!/usr/bin/env python3
"""Tests for the P7 session-reserve draw (`draw_reserve.py`). Stdlib only (unittest); `pytest code/test_draw_reserve.py` also runs it.

    python -I code/test_draw_reserve.py        ->  prints  ZZRESERVE_PASS <n>  when every test passes

What each test proves, and which MUTATION makes it fail:
  reproducible      two fresh draws, and the recorded file, are byte-identical          (mutation: SEED + 1 changes the draw -> test_seed_is_load_bearing)
  order_independent shuffling the input rows / lines leaves the draw unchanged           (mutation: an unsorted dict walk would change it)
  no_navtest        0 shared sessions, 0 shared logs, 0 shared TOKENS with navtest       (mutation: plant a navtest log in navtrain -> the draw REFUSES)
  rule              every acceptance bound of the rule holds on the recorded draw
  pinned            constants and the recorded digests are pinned: editing the rule or the draw fails here
"""
import collections
import csv
import gzip
import importlib.util
import io
import json
import os
import random
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "raw", "src")
RECORDED = os.path.join(ROOT, "raw", "reserve_sessions.json")
RECORDED_B = os.path.join(ROOT, "raw", "reserve_vehicleday.json")
_spec = importlib.util.spec_from_file_location("draw_reserve", os.path.join(HERE, "draw_reserve.py"))
DR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(DR)

PIN_SESSIONS_SHA256 = "b16dc83386ad02ce"       # first 16 hex of reserve.sessions_sha256 (the full digest is in raw/reserve_sessions.json)
PROBE = os.path.join(SRC, "logless251_probe.json")


def _recorded():
    return json.load(open(RECORDED, encoding="utf-8"))


def _copy_src(dst, shuffle=False, plant_navtest_log=None):
    """Copy the inputs; optionally shuffle row/line order, or plant one navtest log into navtrain's log list and token table."""
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(SRC):
        shutil.copy(os.path.join(SRC, f), os.path.join(dst, f))
    rng = random.Random(7)
    if shuffle:
        with gzip.open(os.path.join(SRC, "navtrain_token_table.tsv.gz"), "rt", newline="") as f:
            lines = f.read().splitlines()
        head, body = lines[0], lines[1:]
        rng.shuffle(body)
        with gzip.open(os.path.join(dst, "navtrain_token_table.tsv.gz"), "wt", newline="", encoding="utf-8") as g:
            g.write("\n".join([head] + body) + "\n")
        logs = [ln for ln in open(os.path.join(SRC, "navtrain_logs.txt"), encoding="utf-8").read().split("\n") if ln]
        rng.shuffle(logs)
        open(os.path.join(dst, "navtrain_logs.txt"), "w", newline="\n").write("\n".join(logs) + "\n")
    if plant_navtest_log:
        with open(os.path.join(dst, "navtrain_logs.txt"), "a", newline="\n") as f:
            f.write(plant_navtest_log + "\n")


class DrawTests(unittest.TestCase):
    n = 0

    # ------------------------------------------------------------------------------------------------ reproducible
    def test_reproducible_byte_identical(self):
        a = DR.dumps(DR.draw(SRC, PROBE))
        b = DR.dumps(DR.draw(SRC, PROBE))
        self.assertEqual(a, b)
        self.assertEqual(a, open(RECORDED, encoding="utf-8", newline="").read(), "the recorded draw is not what the script produces")

    def test_check_cli_passes_and_detects_tampering(self):
        self.assertEqual(DR.main(["--src", SRC, "--out", RECORDED, "--check"]), 0)
        with tempfile.TemporaryDirectory() as d:
            bad = os.path.join(d, "r.json")
            txt = open(RECORDED, encoding="utf-8", newline="").read().replace('"attempt_accepted": 2', '"attempt_accepted": 3')
            self.assertNotEqual(txt, open(RECORDED, encoding="utf-8", newline="").read())
            open(bad, "w", encoding="utf-8", newline="").write(txt)
            self.assertEqual(DR.main(["--src", SRC, "--out", bad, "--check"]), 1)

    def test_seed_is_load_bearing(self):
        inp = DR.load_inputs(SRC)
        pool = DR.eligible_pool(inp)
        base, _ = DR.attempt_draw(inp, pool, 2, seed=DR.SEED)
        other = [DR.attempt_draw(inp, pool, 2, seed=DR.SEED + k)[0] for k in (1, 2, 3)]
        self.assertEqual(base, _recorded_sessions())
        self.assertTrue(all(o != base for o in other), "another seed gave the same draw: the seed does nothing")

    def test_order_independent(self):
        with tempfile.TemporaryDirectory() as d:
            _copy_src(d, shuffle=True)
            shuffled = DR.draw(d, os.path.join(d, "logless251_probe.json"))
        rec = _recorded()
        self.assertEqual(shuffled["sessions"], rec["sessions"])
        self.assertEqual(shuffled["reserve"], rec["reserve"])
        self.assertEqual(shuffled["attempt_accepted"], rec["attempt_accepted"])

    def test_cross_platform_prng_is_pure_sha256(self):
        # the PRNG must not depend on `random`, numpy or the Python version: pin two values
        # pinned on Python 3.14.5 AND 3.12 (identical): a change here means the PRNG, and therefore the draw, moved
        self.assertEqual(DR.u01(20261022, 0, "vehicle-seats"), 0.6537298816764163)
        self.assertEqual(DR.u01(5, 6, "x"), 0.6897688583545766)
        self.assertTrue(0.0 <= DR.u01(1, 2, "x") < 1.0)
        self.assertEqual(DR.u01(5, 6, "x"), DR.u01(5, 6, "x"))
        self.assertNotEqual(DR.u01(5, 6, "x"), DR.u01(5, 7, "x"))

    # ------------------------------------------------------------------------------------------------ no navtest
    def test_no_navtest_session_log_or_token(self):
        rec = _recorded()
        nt = json.load(open(os.path.join(SRC, "navtest_logs.json"), encoding="utf-8"))
        self.assertEqual(nt["tokens"], 12146)
        self.assertEqual(len(nt["logs"]), 136)
        nt_logs = set(nt["logs"])
        nt_sess = {DR.session_of(x) for x in nt_logs}
        self.assertEqual(len(nt_sess), 44)
        res_sess = {s["session"] for s in rec["sessions"]}
        res_logs = {lg["log"] for s in rec["sessions"] for lg in s["logs"]}
        self.assertFalse(res_sess & nt_sess)
        self.assertFalse(res_logs & nt_logs)
        # token level: every token of every reserved log, and in fact every token in navtrain's table, is absent from navtest
        nt_tok = set(gzip.open(os.path.join(SRC, "navtest_tokens.txt.gz"), "rt").read().split())
        self.assertEqual(len(nt_tok), 12146)
        tr_tok, res_tok = set(), set()
        flagged = 0
        with gzip.open(os.path.join(SRC, "navtrain_token_table.tsv.gz"), "rt", newline="") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                tr_tok.add(r["token"])
                flagged += int(r["in_navtest"])
                if r["log"] in res_logs:
                    res_tok.add(r["token"])
        self.assertEqual(flagged, 0)
        self.assertFalse(tr_tok & nt_tok, "navtrain and navtest share a token")
        self.assertEqual(len(res_tok), rec["reserve"]["tokens"])
        self.assertFalse(res_tok & nt_tok)

    def test_guard_refuses_a_navtest_contaminated_pool(self):
        """MUTATION arm: plant one navtest log in navtrain's list; the draw must REFUSE. The control (no plant) must not."""
        nt = json.load(open(os.path.join(SRC, "navtest_logs.json"), encoding="utf-8"))["logs"]
        with tempfile.TemporaryDirectory() as d:
            _copy_src(d)
            DR.eligible_pool(DR.load_inputs(d))                      # control: clean inputs pass
        with tempfile.TemporaryDirectory() as d:
            _copy_src(d, plant_navtest_log=nt[0])
            with self.assertRaises(SystemExit) as cm:
                DR.eligible_pool(DR.load_inputs(d))
            self.assertIn("REFUSED", str(cm.exception))

    # ------------------------------------------------------------------------------------------------ the rule
    def test_rule_bounds_hold(self):
        rec = _recorded()
        R = rec["reserve"]
        self.assertEqual(R["n_sessions"], 16)
        lo, hi = DR.FRAC_BAND
        self.assertTrue(lo <= R["fraction_of_bank_scenes"] <= hi)
        self.assertEqual(R["tokens"], sum(s["tokens"] for s in rec["sessions"]))
        self.assertEqual(R["tokens"], sum(lg["tokens"] for s in rec["sessions"] for lg in s["logs"]))
        self.assertEqual(R["n_logs"], sum(len(s["logs"]) for s in rec["sessions"]))
        for s in rec["sessions"]:
            self.assertTrue(DR.MIN_TOK <= s["tokens"] <= DR.MAX_TOK, s["session"])
            self.assertEqual(DR.session_of(s["logs"][0]["log"]), s["session"])
            self.assertTrue(all(DR.session_of(lg["log"]) == s["session"] for lg in s["logs"]))
        ho_sess = {DR.session_of(x) for x in open(os.path.join(SRC, "sft_heldout_logs.txt")).read().split()}
        self.assertEqual(len(ho_sess), 21)
        self.assertFalse({s["session"] for s in rec["sessions"]} & ho_sess, "a session that holds an SFT-held-out log was reserved")
        self.assertEqual(len({s["session"] for s in rec["sessions"]}), 16)
        veh = collections.Counter()
        for s in rec["sessions"]:
            veh[s["vehicle"]] += s["tokens"]
        for v, n in veh.items():
            self.assertLessEqual(n, DR.MAX_VEH_SHARE * rec["by_vehicle"][v]["tokens_total"], v)
            self.assertGreaterEqual(rec["by_vehicle"][v]["sessions_total"], DR.MIN_VEH_SESS, v)
        self.assertGreaterEqual(len(veh), DR.MIN_VEHICLES)
        self.assertGreaterEqual(len({s["city"] for s in rec["sessions"]}), DR.MIN_CITIES)
        self.assertGreaterEqual(R["size_strata"]["small"], DR.MIN_PER_TERCILE)
        self.assertGreaterEqual(R["size_strata"]["large"], DR.MIN_PER_TERCILE)

    def test_every_log_of_a_reserved_session_is_listed(self):
        rec = _recorded()
        yaml_logs = [x for x in open(os.path.join(SRC, "navtrain_logs.txt"), encoding="utf-8").read().split("\n") if x]
        self.assertEqual(len(yaml_logs), 1192)
        want = collections.defaultdict(set)
        for lg in yaml_logs:
            want[DR.session_of(lg)].add(lg)
        for s in rec["sessions"]:
            self.assertEqual({lg["log"] for lg in s["logs"]}, want[s["session"]], s["session"])

    def test_totals_reconcile_with_the_bank(self):
        rec = _recorded()
        I = rec["inputs"]
        self.assertEqual(I["table_rows"], 103288)
        self.assertEqual(I["attributed_tokens"], 103037)           # = the 3-C bank's rank-0 scenes (INHERITED, launch_gate.P6_BANK_TUPLES comment)
        self.assertEqual(I["logless_tokens"], 251)
        self.assertEqual(I["table_rows"] - I["logless_tokens"], I["attributed_tokens"])
        self.assertEqual(I["sessions"], 162)
        self.assertEqual(I["yaml_logs"], 1192)
        R = rec["reserve"]
        self.assertEqual(R["bank_scenes_left_for_training"], 103037 - R["tokens"])
        steps = 25 * -(-R["bank_scenes_left_for_training"] // 256)
        self.assertEqual(steps, 9075)                              # SPEC_P7 section 2: 25 x ceil(92,896 / 256)
        self.assertEqual(25 * -(-103037 // 256), 10075)            # control: the same formula reproduces P6's registered 10,075

    def test_allocation_property(self):
        rng = random.Random(3)
        for _ in range(200):
            mass = {"v%02d" % i: rng.randint(1, 5000) for i in range(rng.randint(1, 14))}
            n = rng.randint(1, 20)
            seats = DR.allocate(mass, n, rng.random())
            self.assertEqual(sum(seats.values()), n)
            tot = sum(mass.values())
            for v, k in seats.items():
                q = n * mass[v] / tot
                self.assertTrue(int(q) <= k <= int(q) + 1 + (q != int(q)), (v, q, k))

    def test_pick_within_is_distinct_and_spans(self):
        items = [(i * 10, "s%03d" % i) for i in range(1, 30)]
        for u in (0.0, 0.3, 0.999):
            pk = DR.pick_within(items, 5, u)
            self.assertEqual(len(set(pk)), 5)
            self.assertLess(pk[0][0], pk[-1][0])
        # the spread: the picks come from every fifth of the size-sorted list
        pk = DR.pick_within(items, 5, 0.5)
        idx = [items.index(p) for p in pk]
        self.assertEqual([i * 5 // len(items) for i in idx], [0, 1, 2, 3, 4])

    # ------------------------------------------------------------------------------------------------ closure (session vs vehicle-day)
    def test_session_closure_leaves_same_day_siblings_in_training(self):
        """MEASURED property of the registered draw A: all 16 reserved sessions keep a same-vehicle same-date sibling in training. Navtest has 0 of 20."""
        self.assertEqual(_recorded()["reserve"]["sessions_with_same_vehicle_same_date_sibling_in_training"], 16)
        nt = json.load(open(os.path.join(SRC, "navtest_logs.json"), encoding="utf-8"))["logs"]
        nt_vd = {DR.unit_of(DR.session_of(x), "vehicle-day") for x in nt}
        tr_vd = {DR.unit_of(DR.session_of(x), "vehicle-day") for x in open(os.path.join(SRC, "navtrain_logs.txt")).read().split()}
        self.assertEqual(len(nt_vd), 20)
        self.assertEqual(len(nt_vd & tr_vd), 0, "navtest shares a vehicle-day with navtrain")
        self.assertEqual(len({x.split("_")[0][:10] for x in map(DR.session_of, nt)} & {x.split("_")[0][:10] for x in map(DR.session_of, open(os.path.join(SRC, "navtrain_logs.txt")).read().split())}), 1)

    def test_vehicle_day_alternative_is_reproducible_closed_and_valid(self):
        txt = DR.dumps(DR.draw(SRC, PROBE, "vehicle-day"))
        self.assertEqual(txt, open(RECORDED_B, encoding="utf-8", newline="").read())
        rec = json.loads(txt)
        self.assertEqual(rec["closure"], "vehicle-day")
        self.assertEqual(rec["reserve"]["sessions_with_same_vehicle_same_date_sibling_in_training"], 0)
        res = {s["session"] for s in rec["sessions"]}
        every = {DR.session_of(x) for x in open(os.path.join(SRC, "navtrain_logs.txt")).read().split()}
        for s in res:                                              # closure: every session of a reserved vehicle-day is reserved
            self.assertTrue(all(m in res for m in every if DR.unit_of(m, "vehicle-day") == DR.unit_of(s, "vehicle-day")), s)
        K = DR.MODES["vehicle-day"]
        lo, hi = DR.FRAC_BAND
        self.assertTrue(lo <= rec["reserve"]["fraction_of_bank_scenes"] <= hi)
        self.assertTrue(K["n_sessions"][0] <= rec["reserve"]["n_sessions"] <= K["n_sessions"][1])
        self.assertGreaterEqual(len({s["vehicle"] for s in rec["sessions"]}), K["min_veh"])
        self.assertGreaterEqual(len({s["city"] for s in rec["sessions"]}), K["min_cities"])
        nt = json.load(open(os.path.join(SRC, "navtest_logs.json"), encoding="utf-8"))["logs"]
        self.assertFalse(res & {DR.session_of(x) for x in nt})
        ho_sess = {DR.session_of(x) for x in open(os.path.join(SRC, "sft_heldout_logs.txt")).read().split()}
        self.assertFalse(res & ho_sess)
        self.assertEqual((rec["reserve"]["tokens"], rec["reserve"]["n_sessions"], rec["attempt_accepted"]), (9974, 17, 521))
        self.assertTrue(rec["reserve"]["sessions_sha256"].startswith("8cd2c0478b48cd91"))

    def test_vehicle_day_mutation_unit_collapsed_to_session_breaks_closure(self):
        """MUTATION arm: if unit_of ignored the closure the draw would leave siblings behind; the closure test above must see it."""
        orig = DR.unit_of
        try:
            DR.unit_of = lambda session, closure: session if closure == "session" else session
            DR._UC.clear()
            try:
                rec = DR.draw(SRC, PROBE, "vehicle-day")
            except SystemExit:                                     # the mutated rule cannot meet its own bounds: the draw refuses (also a catch)
                return
            self.assertGreater(rec["reserve"]["sessions_with_same_vehicle_same_date_sibling_in_training"], 0)
        finally:
            DR.unit_of = orig
            DR._UC.clear()


    # ------------------------------------------------------------------------------------------------ pinned
    def test_pinned_constants_and_digest(self):
        self.assertEqual((DR.SEED, DR.N_SESSIONS, DR.FRAC_BAND, DR.MIN_TOK, DR.MAX_TOK), (20261022, 16, (0.09, 0.11), 150, 1500))
        self.assertEqual((DR.MIN_VEH_SESS, DR.MAX_VEH_SHARE, DR.MIN_PER_TERCILE, DR.MIN_VEHICLES, DR.MIN_CITIES), (3, 0.35, 3, 6, 4))
        self.assertEqual(DR.LABEL, "p7-session-reserve-v1")
        rec = _recorded()
        self.assertTrue(rec["reserve"]["sessions_sha256"].startswith(PIN_SESSIONS_SHA256))
        self.assertEqual(rec["reserve"]["tokens"], 10141)
        self.assertEqual(rec["attempt_accepted"], 2)
        # the digest is recomputable from the sessions alone
        self.assertEqual(DR.digest([s["session"] for s in rec["sessions"]]), rec["reserve"]["sessions_sha256"])
        self.assertEqual(DR.digest([lg["log"] for s in rec["sessions"] for lg in s["logs"]]), rec["reserve"]["logs_sha256"])

    def test_input_hashes_match_the_files(self):
        rec = _recorded()
        for f, want in rec["inputs"]["sha256"].items():
            self.assertEqual(DR.sha256_file(os.path.join(SRC, f)), want, f)


def _recorded_sessions():
    return sorted(s["session"] for s in _recorded()["sessions"])


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(DrawTests)
    res = unittest.TextTestRunner(verbosity=1, stream=sys.stdout).run(suite)
    if res.wasSuccessful():
        print("ZZRESERVE_PASS %d" % res.testsRun)
        sys.exit(0)
    sys.exit(1)
