"""OPS1 tests (operational Amendment 1; UNPINNED evidence, CPU only, no label, no real data, no real extraction).

A. THE CHAIN.  The real `pgc_read_chain.main()` is run on a throw-away directory against PLANTED launcher+child processes (`fake_pg_extract_ops1.py`: the process the chain Popens has a different pid
   from the interpreter that writes `WORKER_DONE_<pid>.txt`, exactly as on Windows).  `stage()` is stubbed (preflight rc 0, join rc 1) so the run ends right after the extraction phase.
   * RED on the CURRENT bytes (ops1/orig/pgc_read_chain.py): a launcher-pid mismatch with clean completion counts as a crash and the chain stops.
   * GREEN on the fix.   * RED arms on the fix: rc 0 with logs missing and no done record, and a nonzero rc with a done record, are still crashes.
   * MUTATION arms: each guard of the fix is disabled in a temp copy of the source and the matching RED scenario must then be passed wrongly.
B. THE SEAL.  An operational amendment may re-pin an operational file and nothing else (RED arms + mutation arms), on throw-away mini packages built by test_pgc_seal's own helpers.
C. THE REAL PACKAGE (mirror).  A temp mirror of the real package + the DRAFT amendment with its STATUS flipped to SEALED in the mirror only: the seal check accepts it, and refuses without it.
"""
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import pgc_seal as S                                                                     # noqa: E402
import test_pgc_seal as T                                                                # noqa: E402

ORIG = f"{ROOT}/ops1/orig/pgc_read_chain.py"
NEW = f"{HERE}/pgc_read_chain.py"
FAKE = f"{HERE}/fake_pg_extract_ops1.py"
N_LOGS = 4
LOGS = [f"log{i}" for i in range(N_LOGS)]


# ======================================================================================================================================================================
# A. the chain
# ======================================================================================================================================================================
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def run_chain(path, plan, *, prefill=0, stale_done=0, tag="m"):
    """Run main() of the chain at `path` on a temp dir.  -> dict(reason, log, claims, td).  `plan` = {"modes": [...], "default": ...}."""
    td = tempfile.mkdtemp(prefix="ops1_chain_")
    m = load(path, f"chain_{tag}_{os.path.basename(td)}")
    m.RC, m.SCAL, m.SUBSET = f"{td}/rc", f"{td}/rc/scalars", f"{td}/rc/subset.json"
    m.N_LOGS, m.MIN_GB, m.PY, m.EXTRACT, m.STALE_MIN = N_LOGS, 0.0, sys.executable, FAKE, 5
    os.makedirs(m.SCAL)
    json.dump({"logs_in_order": LOGS}, open(m.SUBSET, "w"))
    for name in LOGS[:prefill]:
        open(f"{m.SCAL}/{name}.npz", "wb").write(b"x")
    for i in range(stale_done):                                                          # a previous attempt's done records: older than any worker this run starts
        p = f"{m.SCAL}/WORKER_DONE_{900000 + i}.txt"
        open(p, "w").write("old logs_done 0\n")
        os.utime(p, (time.time() - 3600, time.time() - 3600))
    m.time = types.SimpleNamespace(time=time.time, sleep=lambda s: time.sleep(0.4 if s >= 90 else 1.0))
    lines = []
    m.say = lambda msg: lines.append(msg)
    m.stage = lambda name, args, **kw: 0 if args == ["preflight"] else 1                 # preflight ok; join 'fails' => the run ends right after the extraction phase
    os.environ["OPS1_PLAN"] = json.dumps(plan)
    for k in [k for k in os.environ if k.startswith("PGC_")]:
        del os.environ[k]
    try:
        m.main()
    except SystemExit:
        pass
    stop = open(f"{m.RC}/CHAIN_STOP.txt", encoding="utf-8").read() if os.path.exists(f"{m.RC}/CHAIN_STOP.txt") else ""
    claims = len(os.listdir(f"{m.RC}/claims")) if os.path.isdir(f"{m.RC}/claims") else 0
    return {"stop": stop, "lines": lines, "claims": claims, "td": td, "extract_done": any("ZZPGC_EXTRACT_DONE" in x for x in lines),
            "crash_lines": [x for x in lines if "without finishing" in x], "log": "\n".join(lines)}


def cleanup(r):
    shutil.rmtree(r["td"], ignore_errors=True)


CLEAN2 = {"modes": ["clean", "clean"], "default": "clean"}


class TestChainDiagnosis(unittest.TestCase):
    def test_the_planted_launcher_has_a_different_pid_from_its_interpreter(self):
        """the premise: Popen.pid is not the pid that writes WORKER_DONE_<pid>.txt."""
        import subprocess
        td = tempfile.mkdtemp(prefix="ops1_pid_")
        try:
            json.dump({"logs_in_order": LOGS}, open(f"{td}/s.json", "w"))
            os.environ["OPS1_PLAN"] = json.dumps({"modes": ["clean"], "default": "clean"})
            p = subprocess.Popen([sys.executable, FAKE, "work", "--subset", f"{td}/s.json", "--out", f"{td}/o"], stdin=subprocess.DEVNULL)
            p.wait()
            recs = [f for f in os.listdir(f"{td}/o") if f.startswith("WORKER_DONE_")]
            self.assertEqual(len(recs), 1)
            self.assertNotEqual(recs[0], f"WORKER_DONE_{p.pid}.txt")
        finally:
            shutil.rmtree(td, ignore_errors=True)


class TestChainRedOnCurrentBytesGreenOnFix(unittest.TestCase):
    def test_red_current_bytes_launcher_pid_mismatch_with_clean_completion_is_counted_as_a_crash(self):
        r = run_chain(ORIG, CLEAN2, tag="orig")
        try:
            self.assertFalse(r["extract_done"], r["log"])
            self.assertIn("extraction worker crashed", r["stop"], r["log"])               # the exact failure of the 2026-10-20 01:09 stop
            self.assertTrue(any("exited rc=0 without finishing" in x for x in r["crash_lines"]), r["log"])
        finally:
            cleanup(r)

    def test_green_fix_clean_completion_under_a_different_pid_is_finished(self):
        r = run_chain(NEW, CLEAN2, tag="new")
        try:
            self.assertTrue(r["extract_done"], r["log"])
            self.assertEqual(r["crash_lines"], [], r["log"])
            self.assertIn("join failed", r["stop"])                                       # the stub join 'failed': the chain got past extraction
        finally:
            cleanup(r)

    def test_green_fix_all_logs_extracted_no_lock_held_counts_even_without_a_done_record(self):
        r = run_chain(NEW, {"modes": ["allnorecord", "silent0"], "default": "clean"}, tag="new5")
        try:
            self.assertTrue(r["extract_done"], r["log"])
            self.assertEqual(r["crash_lines"], [], r["log"])
        finally:
            cleanup(r)

    def test_green_fix_relaunch_with_everything_extracted_spawns_no_worker(self):
        r = run_chain(NEW, CLEAN2, prefill=N_LOGS, stale_done=5, tag="new2")
        try:
            self.assertEqual(r["claims"], 0, r["log"])                                    # no worker process was started
            self.assertTrue(r["extract_done"], r["log"])
            self.assertTrue(any("no worker spawned" in x for x in r["lines"]), r["log"])
            self.assertIn("join failed", r["stop"])
        finally:
            cleanup(r)

    def test_red_current_bytes_relaunch_spawns_workers_and_stops_the_same_way(self):
        r = run_chain(ORIG, CLEAN2, prefill=N_LOGS, stale_done=5, tag="orig2")
        try:
            self.assertGreaterEqual(r["claims"], 2, r["log"])                             # the old chain starts 2 workers for nothing ...
            self.assertIn("extraction worker crashed", r["stop"], r["log"])               # ... and then stops on their clean exits
        finally:
            cleanup(r)


class TestChainRedArmsOfTheFix(unittest.TestCase):
    def test_red_rc0_with_logs_missing_and_no_done_record_is_still_a_crash(self):
        r = run_chain(NEW, {"modes": [], "default": "silent0"}, tag="new3")
        try:
            self.assertFalse(r["extract_done"], r["log"])
            self.assertIn("extraction worker crashed", r["stop"], r["log"])
            self.assertTrue(any("exited rc=0 without finishing" in x for x in r["crash_lines"]), r["log"])
        finally:
            cleanup(r)

    def test_red_nonzero_rc_is_a_crash_even_with_a_done_record_and_its_stale_lock_is_freed(self):
        r = run_chain(NEW, {"modes": ["crash_with_record", "clean"], "default": "clean"}, tag="new4")
        try:
            self.assertTrue(any("exited rc=5 without finishing (respawn 1/" in x for x in r["lines"]), r["log"])
            self.assertTrue(r["extract_done"], r["log"])                                  # the replacement could take the crashed worker's log: its lock (owner = the dead INTERPRETER pid) was freed
            self.assertFalse(any(f.endswith(".lock") for f in os.listdir(f"{r['td']}/rc/scalars")))
        finally:
            cleanup(r)

    def test_one_done_record_finishes_ONE_worker(self):
        """worker1 exits rc 0 with a record but extracted nothing; worker2 exits rc 0 with NO record and logs missing: worker2 must not borrow worker1's record."""
        r = run_chain(NEW, {"modes": ["record_only", "silent0"], "default": "clean"}, tag="new6")
        try:
            self.assertTrue(any("worker2" in x and "exited rc=0 without finishing" in x for x in r["lines"]), r["log"])
        finally:
            cleanup(r)


def mutate(src_path, old, new, tag):
    s = open(src_path, encoding="utf-8", newline="").read()
    assert s.count(old) == 1, f"mutation target not found exactly once: {old!r}"
    td = tempfile.mkdtemp(prefix="ops1_mut_")
    p = f"{td}/pgc_read_chain_{tag}.py"
    open(p, "w", encoding="utf-8", newline="\n").write(s.replace(old, new))
    return p, td


class TestChainMutationArms(unittest.TestCase):
    """disable ONE guard of the fix => the matching RED scenario is passed WRONGLY (so the RED tests above are the guards, not accidents)."""

    def run_mut(self, old, new, plan, tag, **kw):
        p, td = mutate(NEW, old, new, tag)
        try:
            return run_chain(p, plan, tag=tag, **kw)
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_mutation_nonzero_rc_guard_removed(self):
        r = self.run_mut('    if rc != 0:\n        return False, f"rc={rc}", None\n', "", {"modes": ["crash_with_record", "clean"], "default": "clean"}, "m1")
        try:
            self.assertFalse(any("exited rc=5 without finishing" in x for x in r["lines"]), r["log"])      # the crash is now counted as a clean finish
        finally:
            cleanup(r)

    def test_mutation_logs_extracted_guard_always_true(self):
        r = self.run_mut("    if n >= N_LOGS and not locks and not failed:\n", "    if True:\n", {"modes": [], "default": "silent0"}, "m2")
        try:
            self.assertNotIn("extraction worker crashed", r["stop"], r["log"])           # rc 0 with logs missing is now passed as finished
            self.assertIn("both workers finished but extraction is incomplete", r["stop"], r["log"])
        finally:
            cleanup(r)

    def test_mutation_claim_guard_removed(self):
        r = self.run_mut("if name not in claimed and mt >= t_start - DONE_SLACK_S:", "if mt >= t_start - DONE_SLACK_S:", {"modes": ["record_only", "silent0"], "default": "clean"}, "m3")
        try:
            self.assertFalse(any("worker2" in x and "exited rc=0 without finishing" in x for x in r["lines"]), r["log"])      # worker2 borrowed worker1's record
        finally:
            cleanup(r)

    def test_mutation_start_time_guard_removed(self):
        """a done record from a PREVIOUS attempt must not finish a worker started now."""
        r = self.run_mut("if name not in claimed and mt >= t_start - DONE_SLACK_S:", "if name not in claimed:", {"modes": [], "default": "silent0"}, "m4", stale_done=5)
        try:
            self.assertNotIn("extraction worker crashed", r["stop"], r["log"])           # the old records finished the silent workers
        finally:
            cleanup(r)
        r2 = run_chain(NEW, {"modes": [], "default": "silent0"}, stale_done=5, tag="new7")   # control: the real fix with the same stale records still stops as a crash
        try:
            self.assertIn("extraction worker crashed", r2["stop"], r2["log"])
        finally:
            cleanup(r2)

    def test_mutation_stale_lock_freeing_disabled(self):
        r = self.run_mut("    if pid is None:\n        return False  ", "    return False\n    if pid is None:\n        return False  ", {"modes": ["crash_with_record", "clean"], "default": "clean"}, "m5")
        try:
            self.assertFalse(r["extract_done"], r["log"])                                # the dead worker's lock is never freed: its log is never taken
        finally:
            cleanup(r)

    def test_mutation_skip_when_complete_removed(self):
        r = self.run_mut("    if n0 >= N_LOGS:  ", "    if False:  ", CLEAN2, "m6", prefill=N_LOGS, stale_done=5)
        try:
            self.assertGreaterEqual(r["claims"], 2, r["log"])                            # workers are spawned for nothing again
        finally:
            cleanup(r)


class TestWorkerFinishedTable(unittest.TestCase):
    def setUp(self):
        self.m = load(NEW, "chain_table")
        self.m.N_LOGS = 76

    def test_table(self):
        f = self.m.worker_finished
        t = 1000.0
        rec = [("WORKER_DONE_5.txt", 1500.0)]
        self.assertEqual(f(0, t, rec, set(), 10, [], [])[:2], (True, "done record WORKER_DONE_5.txt"))
        self.assertFalse(f(0, t, rec, {"WORKER_DONE_5.txt"}, 10, [], [])[0])             # already claimed
        self.assertFalse(f(0, t, [("WORKER_DONE_5.txt", 900.0)], set(), 10, [], [])[0])  # older than the worker
        self.assertTrue(f(0, t, [("WORKER_DONE_5.txt", 999.0)], set(), 10, [], [])[0])   # inside the 2 s mtime slack
        self.assertTrue(f(0, t, [], set(), 76, [], [])[0])                               # all extracted, no lock
        self.assertFalse(f(0, t, [], set(), 76, ["x.lock"], [])[0])                      # a lock still held
        self.assertFalse(f(0, t, [], set(), 75, [], [])[0])                              # one log missing
        self.assertFalse(f(3, t, rec, set(), 76, [], [])[0])                             # nonzero rc: a crash whatever else is true
        self.assertFalse(f(1, t, [], set(), 76, [], [])[0])                              # nonzero rc with every log extracted: still a crash


# ======================================================================================================================================================================
# B. the seal: operational amendments
# ======================================================================================================================================================================
def ops_text(repin, status="SEALED"):
    block = json.dumps({"repin": repin}, indent=1)
    return f"# OPS amendment (test)\n\n**STATUS: {status}.**\n\n{S.REPIN_BEGIN}\n```json\n{block}\n```\n{S.REPIN_END}\n"


def put_amend(root, text, k=1, series="_AMENDMENT_OPS", hash_ok=True):
    base = f"{root}/PREREG_PRIVGEOM_CLEAN{series}{k}"
    T.write(base + ".md", text)
    T.write(base + ".sha256", f"{S.sha_file(base + '.md') if hash_ok else '0' * 64}  x\n")


class TestSealOperationalAmendment(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp(prefix="ops1_seal_")
        self.root = T.build(self.td)
        self.old = S.sha_file(f"{self.root}/code/PINS.json")
        T.write(f"{self.root}/code/PINS.json", json.dumps({"frozen_ok": True, "sha256": {}, "edited": 1}))        # an OPERATIONAL file edited after the seal
        self.new = S.sha_file(f"{self.root}/code/PINS.json")

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def check(self, **kw):
        return S.check_seal(self.root, required=T.REQUIRED, **kw)

    def refused(self, needle):
        with self.assertRaises(S.SealError) as cm:
            self.check()
        self.assertIn(needle, str(cm.exception))

    def good(self, **kw):
        return {"code/PINS.json": {"from": self.old, "to": self.new}}

    def test_red_edited_operational_file_without_an_amendment_is_refused(self):
        self.refused("pinned file(s) differ")

    def test_green_sealed_operational_amendment_re_pins_it(self):
        put_amend(self.root, ops_text(self.good()))
        info = self.check()
        self.assertEqual(info["repinned"], {"code/PINS.json": self.new})
        self.assertIn("amend_ops1", info["pins"])
        self.assertEqual(info["status"], "SEALED")

    def test_red_draft_status_amendment_re_pins_nothing(self):
        put_amend(self.root, ops_text(self.good(), status="DRAFT"))
        self.refused("not SEALED")

    def test_red_amendment_edited_after_its_hash(self):
        put_amend(self.root, ops_text(self.good()))
        with open(f"{self.root}/PREREG_PRIVGEOM_CLEAN_AMENDMENT_OPS1.md", "a", newline="\n") as f:
            f.write("x")
        self.refused("Operational amendment 1 does not hash")

    def test_red_wrong_sha_file(self):
        put_amend(self.root, ops_text(self.good()), hash_ok=False)
        self.refused("does not hash")

    def test_red_to_is_not_what_is_on_disk(self):
        put_amend(self.root, ops_text({"code/PINS.json": {"from": self.old, "to": "ab" * 32}}))
        self.refused("pinned file(s) differ")

    def test_red_from_is_not_the_pin_in_force(self):
        put_amend(self.root, ops_text({"code/PINS.json": {"from": "cd" * 32, "to": self.new}}))
        self.refused("pin in force")

    def test_red_an_analysis_file_can_never_be_re_pinned(self):
        old_b = S.sha_file(f"{self.root}/code/b.py")
        T.write(f"{self.root}/code/b.py", "B = 99\n")
        put_amend(self.root, ops_text({"code/b.py": {"from": old_b, "to": S.sha_file(f"{self.root}/code/b.py")}, **self.good()}))
        self.refused("not an operational file")

    def test_red_the_plain_amendment_series_cannot_re_pin(self):
        put_amend(self.root, ops_text(self.good()), series="_AMENDMENT")
        self.refused("pinned file(s) differ")

    def test_red_malformed_block(self):
        put_amend(self.root, f"**STATUS: SEALED.**\n{S.REPIN_BEGIN}\nnot json\n{S.REPIN_END}\n")
        self.refused("unreadable OPS_REPIN")

    def test_green_a_chain_of_two_operational_amendments_and_a_broken_chain(self):
        put_amend(self.root, ops_text(self.good()), k=1)
        T.write(f"{self.root}/code/PINS.json", json.dumps({"frozen_ok": True, "sha256": {}, "edited": 2}))
        third = S.sha_file(f"{self.root}/code/PINS.json")
        put_amend(self.root, ops_text({"code/PINS.json": {"from": self.new, "to": third}}), k=2)
        self.assertEqual(self.check()["repinned"], {"code/PINS.json": third})
        put_amend(self.root, ops_text({"code/PINS.json": {"from": self.old, "to": third}}), k=2)       # starts from the SEALED pin, not the pin in force after amendment 1
        self.refused("pin in force")

    def test_the_repinnable_set_is_operational_only(self):
        for n in ("pg_geom.py", "pg_extract.py", "pg_lib.py", "pgc_lib.py", "pgc_admit.py", "pgc_run.py", "pgc_coverage.py", "pgc_design.py", "pgc_calibrate_plant.py", "pgc_glive.py",
                  "pgc_admit_seen.py", "pgc_cost.py", "pgc_rehearsal_identity.py", "test_pg_geom.py", "test_pg_lib.py", "test_pgc_lib.py", "test_pgc_admit.py", "test_pgc_seal.py"):
            self.assertNotIn(f"code/{n}", S.REPINNABLE, n)
        self.assertNotIn("ref/fastneg_lib.py", S.REPINNABLE)
        for n in ("code/pgc_read_chain.py", "code/pgc_seal.py", "code/PINS.json"):
            self.assertIn(n, S.REPINNABLE)

    # ---- MUTATION arms ---------------------------------------------------------------------------------------------------------------------------------------
    def test_mutation_status_guard_disabled_lets_a_draft_amendment_through(self):
        put_amend(self.root, ops_text(self.good(), status="DRAFT"))
        self.refused("not SEALED")
        real = S.status_of
        S.status_of = lambda text: "SEALED"
        try:
            self.assertEqual(self.check()["repinned"], {"code/PINS.json": self.new})
        finally:
            S.status_of = real

    def test_mutation_repinnable_guard_disabled_lets_an_analysis_file_through(self):
        old_b = S.sha_file(f"{self.root}/code/b.py")
        T.write(f"{self.root}/code/b.py", "B = 99\n")
        put_amend(self.root, ops_text({"code/b.py": {"from": old_b, "to": S.sha_file(f"{self.root}/code/b.py")}, **self.good()}))
        self.refused("not an operational file")
        real = S.REPINNABLE
        S.REPINNABLE = real + ("code/b.py",)
        try:
            self.assertIn("code/b.py", self.check()["repinned"])
        finally:
            S.REPINNABLE = real

    def test_mutation_chain_guard_disabled_lets_a_wrong_from_through(self):
        put_amend(self.root, ops_text({"code/PINS.json": {"from": "cd" * 32, "to": self.new}}))
        self.refused("pin in force")
        src = open(f"{HERE}/pgc_seal.py", encoding="utf-8").read()
        self.assertEqual(src.count('if e["from"] != eff[name]:'), 1)
        m = types.ModuleType("pgc_seal_mut")
        m.__file__ = f"{HERE}/pgc_seal.py"
        exec(compile(src.replace('if e["from"] != eff[name]:', "if False:"), "pgc_seal_mut", "exec"), m.__dict__)
        self.assertEqual(m.check_seal(self.root, required=T.REQUIRED)["repinned"], {"code/PINS.json": self.new})        # the mutated check ACCEPTS it


# ======================================================================================================================================================================
# C. the real package, on a temp mirror
# ======================================================================================================================================================================
AMEND = f"{ROOT}/PREREG_PRIVGEOM_CLEAN_AMENDMENT_OPS1.md"


@unittest.skipUnless(os.path.exists(AMEND), "the draft amendment is not written yet")
class TestRealPackageMirror(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp(prefix="ops1_mirror_")
        self.m = f"{self.td}/pkg"
        os.makedirs(f"{self.m}/code")
        os.makedirs(f"{self.m}/ref")
        for f in os.listdir(f"{ROOT}/code"):
            if os.path.isfile(f"{ROOT}/code/{f}"):
                shutil.copy(f"{ROOT}/code/{f}", f"{self.m}/code/{f}")
        shutil.copy(f"{ROOT}/ref/fastneg_lib.py", f"{self.m}/ref/fastneg_lib.py")
        for f in ("PREREG_PRIVGEOM_CLEAN.md", "PREREG_PRIVGEOM_CLEAN.sha256", "SEALED_AT.txt", "MM_RULING_PGCLEAN.md"):
            shutil.copy(f"{ROOT}/{f}", f"{self.m}/{f}")
        import pgc_run
        self.required = pgc_run.SEAL_REQUIRED

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def check(self):
        return S.check_seal(self.m, required=self.required)

    def test_the_real_package_with_the_new_bytes_is_refused_without_the_sealed_amendment_and_accepted_with_it(self):
        with self.assertRaises(S.SealError) as cm:
            self.check()
        self.assertIn("pgc_read_chain.py", str(cm.exception))
        text = open(AMEND, encoding="utf-8", newline="").read().replace("\r\n", "\n")
        # the real amendment was a DRAFT when this test was written and is SEALED since 2026-10-20 (MM); both states are exercised in the MIRROR only
        self.assertIn(S.status_of(text), ("DRAFT", "SEALED"))
        draft = text.replace("**STATUS: SEALED", "**STATUS: DRAFT", 1)
        T.write(f"{self.m}/PREREG_PRIVGEOM_CLEAN_AMENDMENT_OPS1.md", draft)
        T.write(f"{self.m}/PREREG_PRIVGEOM_CLEAN_AMENDMENT_OPS1.sha256", f"{S.sha_file(self.m + '/PREREG_PRIVGEOM_CLEAN_AMENDMENT_OPS1.md')}  x\n")
        with self.assertRaises(S.SealError):                                              # a DRAFT re-pins nothing, even with a matching .sha256
            self.check()
        sealed = text.replace("**STATUS: DRAFT", "**STATUS: SEALED", 1)                  # in the MIRROR only; the real amendment is never edited or hashed by this test
        T.write(f"{self.m}/PREREG_PRIVGEOM_CLEAN_AMENDMENT_OPS1.md", sealed)
        T.write(f"{self.m}/PREREG_PRIVGEOM_CLEAN_AMENDMENT_OPS1.sha256", f"{S.sha_file(self.m + '/PREREG_PRIVGEOM_CLEAN_AMENDMENT_OPS1.md')}  x\n")
        info = self.check()
        self.assertEqual(sorted(info["repinned"]), ["code/PINS.json", "code/pgc_read_chain.py", "code/pgc_seal.py"])
        self.assertIn("amend_ops1", info["pins"])
        if S.status_of(text) == "SEALED":                                                 # the real seal record matches the real amendment's bytes
            real = open(f"{ROOT}/PREREG_PRIVGEOM_CLEAN_AMENDMENT_OPS1.sha256", encoding="utf-8").read().split()[0]
            self.assertEqual(real, S.sha_file(AMEND))
        else:                                                                             # a DRAFT re-pins nothing and the real root is untouched
            self.assertFalse(os.path.exists(f"{ROOT}/PREREG_PRIVGEOM_CLEAN_AMENDMENT_OPS1.sha256"))

    def test_the_sealed_prereg_and_seal_record_are_byte_identical_to_the_real_ones(self):
        for f in ("PREREG_PRIVGEOM_CLEAN.md", "SEALED_AT.txt", "PREREG_PRIVGEOM_CLEAN.sha256"):
            self.assertEqual(S.sha_file(f"{self.m}/{f}"), S.sha_file(f"{ROOT}/{f}"))
        self.assertEqual(S.sha_file(f"{ROOT}/PREREG_PRIVGEOM_CLEAN.md"), "148ebf0f2162b43ba7a1c29243fceac5900fe6b88ace3cae04fc27bdf29c6cf9")

    def test_an_analysis_file_edit_is_still_refused_with_the_amendment_sealed(self):
        text = open(AMEND, encoding="utf-8", newline="").read().replace("\r\n", "\n").replace("**STATUS: DRAFT", "**STATUS: SEALED", 1)
        T.write(f"{self.m}/PREREG_PRIVGEOM_CLEAN_AMENDMENT_OPS1.md", text)
        T.write(f"{self.m}/PREREG_PRIVGEOM_CLEAN_AMENDMENT_OPS1.sha256", f"{S.sha_file(self.m + '/PREREG_PRIVGEOM_CLEAN_AMENDMENT_OPS1.md')}  x\n")
        self.check()
        with open(f"{self.m}/code/pgc_lib.py", "a", newline="\n") as f:
            f.write("# one byte of analysis code\n")
        with self.assertRaises(S.SealError) as cm:
            self.check()
        self.assertIn("pgc_lib.py", str(cm.exception))


if __name__ == "__main__":
    unittest.main(verbosity=1)
