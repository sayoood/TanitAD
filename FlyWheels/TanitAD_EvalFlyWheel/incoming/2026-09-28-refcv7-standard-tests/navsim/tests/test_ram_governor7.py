"""The RAM governor + slot lock of the NavSim scorer drivers (any python, no GPU, no NavSim).

Literal expectations only (CLAUDE.md "a check that shares the defect it checks is green forever"):
every threshold below is typed as a number, never computed from the module under test, and the
two safety properties that matter most -- "it never suspends the driver or an ancestor" and "stop()
never strands a suspended scorer" -- each carry a DELIBERATE-REGRESSION arm that must go RED.

MEASURED 2026-10-04: 17 scorer aborts at step 50,400 came from the scorers' OWN RAM guard (hard floor =
ONE 2 s sample below 2,000 MB) while the scorers held 0.84 / 2.22 GB; the governor must pause 1.5 GB
above that floor and must resume only with room for the pages it gave back.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import psutil
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "code"))
import ram_governor7 as RG  # noqa: E402

HEARTBEAT = ("import sys,time\n"
             "p=sys.argv[1]; n=0\n"
             "while True:\n"
             "    n+=1\n"
             "    open(p,'w').write(str(n))\n"
             "    time.sleep(0.03)\n")


def _count(path) -> int:
    try:
        return int(open(path, encoding="utf-8").read() or 0)
    except (OSError, ValueError):
        return 0


@pytest.fixture()
def child(tmp_path):
    hb = tmp_path / "hb.txt"
    proc = subprocess.Popen([sys.executable, "-c", HEARTBEAT, str(hb)])
    t0 = time.time()
    while _count(hb) < 3 and time.time() - t0 < 20:
        time.sleep(0.05)
    assert _count(hb) >= 3, "the heartbeat child never started"
    yield proc, hb
    for q in tree(proc):
        try:
            q.resume()
        except psutil.Error:
            pass
    for q in tree(proc)[::-1]:
        try:
            q.kill()
        except psutil.Error:
            pass
    proc.kill()
    proc.wait(timeout=20)


def tree(proc) -> list:
    """The Popen AND its descendants. A venv ``python.exe`` is a launcher stub whose CHILD is the real
    interpreter -- exactly the shape of the real scorers (stub -> navsim python), so suspending only
    ``proc.pid`` would leave the heartbeat running."""
    try:
        root = psutil.Process(proc.pid)
        return [root] + root.children(recursive=True)
    except psutil.Error:
        return []


def _advances(hb, secs=0.6) -> bool:
    a = _count(hb)
    time.sleep(secs)
    return _count(hb) > a


class Probe:
    def __init__(self, v):
        self.v = v

    def __call__(self):
        return self.v


def _gov(proc, probe, **kw):
    kw.setdefault("resume_ok_s", 0.0)
    kw.setdefault("jitter_s", 0.0)
    kw.setdefault("trim", False)
    return RG.Governor("t", None, probe=probe, procs=lambda: tree(proc), **kw)


# --------------------------------------------------------------------------- #
# thresholds, literally                                                        #
# --------------------------------------------------------------------------- #
def test_policy_constants_are_the_documented_ones():
    # 1.5 GB above the guard's hard floor (2,000 MB), 0.5 GB above its soft floor (3,000 MB)
    assert RG.PAUSE_MB == 3500.0 and RG.HEADROOM_MB == 500.0
    assert RG.N_SLOTS == 2 and RG.EXTRA_SLOT_MIN_MB == 8000.0


def test_pause_boundary_3500_exactly_does_not_pause_3499_does(child):
    proc, hb = child
    p = Probe(3500.0)
    g = _gov(proc, p)
    g.step()
    assert g.paused is False and _advances(hb)
    p.v = 3499.0
    g.step()
    assert g.paused is True and g.n_pauses == 1
    assert not _advances(hb), "a paused scorer must make NO progress"
    g.stop()


def test_resume_needs_pause_plus_headroom_literally_4000_when_nothing_is_owed(child):
    proc, hb = child
    p = Probe(3000.0)
    g = _gov(proc, p)                       # trim=False: nothing was given back, so nothing is owed
    g.step()
    assert g.paused
    p.v = 3999.0
    g.step()
    assert g.paused and not _advances(hb)
    p.v = 4000.0
    g.step()
    assert g.paused is False and _advances(hb), "a scorer resumed at 4,000 MB must run again"
    g.stop()


def test_resume_requires_the_condition_to_HOLD_resume_ok_s(child):
    proc, hb = child
    p = Probe(3000.0)
    g = _gov(proc, p, resume_ok_s=0.5)
    g.step()
    p.v = 5000.0
    g.step()                                # condition first seen: not yet held for 0.5 s
    assert g.paused
    p.v = 3100.0
    g.step()                                # a dip resets the timer
    p.v = 5000.0
    g.step()
    time.sleep(0.3)
    g.step()
    assert g.paused, "0.3 s after the dip is not 0.5 s of sustained room"
    time.sleep(0.4)
    g.step()
    assert g.paused is False
    g.stop()


def test_owed_pages_raise_the_resume_bar():
    """A scorer that gave 2,200 MB back (trimmed) needs 3,500 + 500 + 2,200 = 6,200 MB to resume."""
    class Fake:
        pid = 424242
        suspended = False

        def create_time(self):
            return 1.0

        def memory_info(self):
            class M:
                rss = (2200 if not Fake.suspended else 0) * 2 ** 20
            return M()

        def suspend(self):
            Fake.suspended = True

        def resume(self):
            Fake.suspended = False
    Fake.suspended = False
    f = Fake()
    p = Probe(3000.0)
    g = RG.Governor("t", None, probe=p, procs=lambda: [f], resume_ok_s=0.0, jitter_s=0.0, trim=False,
                    lookup=lambda pid: f)
    g._targets = lambda: [f]                # the fake pid is not a real process; bypass the real filter
    g.step()
    assert g.paused and Fake.suspended and g._rss_before_mb == 2200.0
    p.v = 6199.0
    g.step()
    assert g.paused
    p.v = 6200.0
    g.step()
    assert g.paused is False and Fake.suspended is False


# --------------------------------------------------------------------------- #
# safety                                                                       #
# --------------------------------------------------------------------------- #
def test_never_targets_the_driver_or_an_ancestor_and_a_regression_would():
    me = psutil.Process()
    parent = me.parent()

    class P:
        def __init__(self, pid):
            self.pid = pid

    procs = [P(me.pid), P(parent.pid), P(10 ** 7)]
    g = RG.Governor("t", None, probe=Probe(9999.0), procs=lambda: procs)
    assert [t.pid for t in g._targets()] == [10 ** 7], "driver / ancestor must be excluded"

    # DELIBERATE REGRESSION: the unfiltered target list contains the driver itself -> a governor
    # built that way would suspend its own thread's process (and hang the run). Must differ.
    unfiltered = [t.pid for t in procs]
    assert me.pid in unfiltered and me.pid not in [t.pid for t in g._targets()]


def test_stop_resumes_a_paused_child_and_a_stop_that_forgot_would_strand_it(child):
    proc, hb = child
    g = _gov(proc, Probe(100.0))
    g.step()
    assert g.paused and not _advances(hb)
    g.stop()
    assert _advances(hb), "stop() must never leave the scorer suspended"
    assert g.n_forced_resumes == 0          # a normal stop is not a timeout

    # DELIBERATE REGRESSION: the same pause WITHOUT stop() leaves the child frozen
    g2 = _gov(proc, Probe(100.0))
    g2.step()
    assert g2.paused and not _advances(hb)
    # (fixture teardown resumes it)


def test_max_pause_forces_a_resume_so_the_scorers_own_guard_decides(child):
    proc, hb = child
    g = _gov(proc, Probe(100.0), max_pause_s=0.3)
    g.step()
    assert g.paused
    time.sleep(0.4)
    g.step()
    assert g.paused is False and g.n_forced_resumes == 1 and _advances(hb)
    g.stop()


def test_a_dead_scorer_does_not_break_the_governor(child):
    proc, hb = child
    g = _gov(proc, Probe(100.0))
    g.step()
    assert g.paused
    proc.kill()
    proc.wait(timeout=20)
    g.probe = Probe(9000.0)
    g.step()                                # resume() on a dead pid: recorded, not raised
    assert g.paused is False


def test_log_records_pause_and_resume(child, tmp_path):
    proc, hb = child
    log = tmp_path / "gov.jsonl"
    g = RG.Governor("lbl", str(log), probe=Probe(100.0), procs=lambda: tree(proc),
                    resume_ok_s=0.0, jitter_s=0.0, trim=False)
    g.step()
    g.probe = Probe(9000.0)
    g.step()
    ev = [json.loads(x)["event"] for x in open(log, encoding="utf-8")]
    assert ev == ["PAUSE", "RESUME"]
    s = g.summary()
    assert s["n_pauses"] == 1 and s["min_avail_mb_seen"] == 100 and s["paused_s_total"] >= 0.0


# --------------------------------------------------------------------------- #
# orphans                                                                      #
# --------------------------------------------------------------------------- #
def test_orphan_of_a_dead_driver_is_resumed_but_a_live_owners_is_not(child, tmp_path):
    proc, hb = child
    held = [[q.pid, q.create_time()] for q in tree(proc)]
    for q in tree(proc):
        q.suspend()
    live = tmp_path / "live.1.suspended.json"
    live.write_text(json.dumps({"owner_pid": os.getpid(),
                                "owner_create_time": psutil.Process().create_time(),
                                "held": held}))
    assert RG.resume_orphans(str(tmp_path)) == [] and not _advances(hb)
    dead = 999999
    while psutil.pid_exists(dead):
        dead += 1
    live.unlink()
    (tmp_path / "dead.2.suspended.json").write_text(
        json.dumps({"owner_pid": dead, "owner_create_time": 1.0, "held": held}))
    assert sorted(RG.resume_orphans(str(tmp_path))) == sorted(h[0] for h in held)
    assert _advances(hb) and not (tmp_path / "dead.2.suspended.json").exists()


def test_orphan_with_a_recycled_pid_is_not_resumed(child, tmp_path):
    proc, hb = child
    held = [[q.pid, q.create_time() + 500.0] for q in tree(proc)]    # a recycled pid: other create_time
    for q in tree(proc):
        q.suspend()
    dead = 999999
    while psutil.pid_exists(dead):
        dead += 1
    (tmp_path / "x.3.suspended.json").write_text(
        json.dumps({"owner_pid": dead, "owner_create_time": 1.0, "held": held}))
    assert RG.resume_orphans(str(tmp_path)) == [] and not _advances(hb)


# --------------------------------------------------------------------------- #
# slots + the start gate                                                       #
# --------------------------------------------------------------------------- #
def test_slots_two_at_most_and_the_second_needs_8000(tmp_path):
    a = RG.SlotLock(str(tmp_path), probe=lambda: 7999.0)
    b = RG.SlotLock(str(tmp_path), probe=lambda: 7999.0)
    assert a.try_take("a") is True                 # slot 0 never needs more than the start gate
    assert b.try_take("b") is False                # slot 1 closed below 8,000 MB
    b2 = RG.SlotLock(str(tmp_path), probe=lambda: 8000.0)
    assert b2.try_take("b2") is True               # exactly 8,000 MB opens it
    c = RG.SlotLock(str(tmp_path), probe=lambda: 99999.0)
    assert c.try_take("c") is False                # never a third
    a.release()
    assert c.try_take("c") is True
    assert sorted(os.listdir(tmp_path)) == ["slot0.lock", "slot1.lock"]


def test_a_dead_holders_slot_is_reclaimed_and_a_live_holders_is_not(tmp_path):
    dead = 999999
    while psutil.pid_exists(dead):
        dead += 1
    (tmp_path / "slot0.lock").write_text(json.dumps({"pid": dead, "create_time": 1.0}))
    s = RG.SlotLock(str(tmp_path), probe=lambda: 1.0)
    assert s.try_take("x") is True
    s.release()
    me = psutil.Process()
    (tmp_path / "slot0.lock").write_text(json.dumps({"pid": me.pid, "create_time": me.create_time()}))
    assert RG.SlotLock(str(tmp_path), probe=lambda: 1.0).try_take("y") is False


def test_release_never_removes_another_drivers_slot(tmp_path):
    s = RG.SlotLock(str(tmp_path), probe=lambda: 1.0)
    (tmp_path / "slot0.lock").write_text(json.dumps({"pid": 1, "create_time": 1.0}))
    s.held = str(tmp_path / "slot0.lock")
    s.release()
    assert (tmp_path / "slot0.lock").exists()


def test_start_gate_needs_3500_plus_rss_plus_500_three_times_and_3000_commit(tmp_path):
    seq = iter([6299.0, 6299.0, 6300.0, 6300.0, 6300.0, 6300.0, 6300.0])
    probe = lambda: next(seq)                                           # noqa: E731
    slots = RG.SlotLock(str(tmp_path), probe=lambda: 99999.0)
    got = RG.wait_for_start("lbl", 2300.0, slots=slots, probe=probe,
                            commit_probe=lambda: 9999.0, sleep=lambda s: None, log=lambda m: None)
    assert got is slots and slots.held
    slots.release()
    # commit gate: 2,999 MB free commit blocks, 3,000 passes
    cm = iter([2999.0, 2999.0, 2999.0, 3000.0, 3000.0, 3000.0, 3000.0])
    got = RG.wait_for_start("lbl2", 0.0, slots=slots, probe=lambda: 99999.0,
                            commit_probe=lambda: next(cm), sleep=lambda s: None, log=lambda m: None)
    assert got.held
    slots.release()


def test_start_gate_times_out_instead_of_blocking_forever(tmp_path):
    with pytest.raises(TimeoutError):
        RG.wait_for_start("lbl", 2300.0, slots=RG.SlotLock(str(tmp_path)), probe=lambda: 10.0,
                          commit_probe=lambda: 99999.0, sleep=lambda s: time.sleep(0.01),
                          max_wait_s=0.1, log=lambda m: None)


# --------------------------------------------------------------------------- #
# the context manager, end to end                                              #
# --------------------------------------------------------------------------- #
def test_governed_end_to_end_pauses_and_resumes_a_real_child_and_frees_the_slot(tmp_path, monkeypatch):
    monkeypatch.setattr(RG, "SLOT_DIR", str(tmp_path / "slots"))
    hb = tmp_path / "hb.txt"
    probe = Probe(99999.0)
    with RG.governed("e2e", str(tmp_path / "out"), est_rss_mb=0.0, probe=probe, sample_s=0.05,
                     paused_sample_s=0.05, resume_ok_s=0.0, jitter_s=0.0, trim=False,
                     start_kw={"probe": lambda: 99999.0, "commit_probe": lambda: 99999.0}) as gov:
        proc = subprocess.Popen([sys.executable, "-c", HEARTBEAT, str(hb)])
        try:
            t0 = time.time()
            while _count(hb) < 3 and time.time() - t0 < 20:
                time.sleep(0.05)
            probe.v = 100.0                               # the box collapses
            t0 = time.time()
            while not gov.paused and time.time() - t0 < 5:
                time.sleep(0.05)
            assert gov.paused and not _advances(hb)
            probe.v = 99999.0                             # and recovers
            t0 = time.time()
            while gov.paused and time.time() - t0 < 5:
                time.sleep(0.05)
            assert gov.paused is False and _advances(hb)
        finally:
            proc.kill()
            proc.wait(timeout=20)
    assert sorted(os.listdir(tmp_path / "slots")) == [], "slot and state file must be released"
    events = [json.loads(x)["event"] for x in open(next((tmp_path / "out").glob("e2e*.jsonl")),
                                                    encoding="utf-8")]
    assert events[0] == "START" and "PAUSE" in events and "RESUME" in events and events[-1] == "END"


def test_governor_can_be_switched_off(tmp_path, monkeypatch):
    monkeypatch.setenv("R7_GOVERNOR", "0")
    monkeypatch.setattr(RG, "SLOT_DIR", str(tmp_path / "slots"))
    with RG.governed("off", str(tmp_path / "out"), est_rss_mb=99999.0) as gov:
        assert gov.summary()["governor"] == "DISABLED"
    assert not (tmp_path / "slots").exists()


# --------------------------------------------------------------------------- #
# idempotence: a duplicate launch never rescores a banked PASS                 #
# --------------------------------------------------------------------------- #
def _bank(tmp_path, status="PASS", seam_sha="abc", arm=None, csv=True, csv_sha=True):
    import hashlib
    seam = tmp_path / "seam_X.npz"
    seam.write_bytes(b"seam")
    time.sleep(0.05)
    csvp = tmp_path / "x.csv"
    if csv:
        csvp.write_bytes(b"token,score\n")
    rec = {"status": status}
    if seam_sha is not None:
        rec["seam_sha256"] = seam_sha
    if arm:
        rec["arm"] = arm.format(seam=str(seam))
    if csv_sha:
        rec["csv_sha256"] = hashlib.sha256(b"token,score\n").hexdigest()
    counts = tmp_path / "x.counts.json"
    counts.write_text(json.dumps(rec))
    return str(counts), str(seam), str(csvp)


def test_already_scored_by_seam_sha(tmp_path):
    c, s, v = _bank(tmp_path)
    assert RG.already_scored(c, s, v, seam_sha="abc") is not None
    assert RG.already_scored(c, s, v, seam_sha="DIFFERENT") is None        # a re-bridged seam rescoring


def test_already_scored_refuses_a_fail_a_missing_csv_and_a_changed_csv(tmp_path):
    c, s, v = _bank(tmp_path, status="FAIL")
    assert RG.already_scored(c, s, v, seam_sha="abc") is None
    c, s, v = _bank(tmp_path)
    os.remove(v)
    assert RG.already_scored(c, s, v, seam_sha="abc") is None
    c, s, v = _bank(tmp_path)
    open(v, "ab").write(b"tampered")
    assert RG.already_scored(c, s, v, seam_sha="abc") is None


def test_already_scored_navtest_form_names_the_seam_and_needs_counts_newer_than_it(tmp_path):
    c, s, v = _bank(tmp_path, seam_sha=None, arm="SEAM:{seam}")
    assert RG.already_scored(c, s, v) is not None
    # backslash / case differences in the recorded path are the same seam
    rec = json.load(open(c))
    rec["arm"] = ("SEAM:" + os.path.abspath(s)).replace("/", "\\").upper()
    open(c, "w").write(json.dumps(rec))
    assert RG.already_scored(c, s, v) is not None
    # another seam path -> not the same run
    rec["arm"] = "SEAM:" + os.path.abspath(s) + ".other"
    open(c, "w").write(json.dumps(rec))
    assert RG.already_scored(c, s, v) is None
    # a seam REWRITTEN after the PASS (re-bridged) is newer than the counts -> rescore
    rec["arm"] = "SEAM:" + os.path.abspath(s)
    open(c, "w").write(json.dumps(rec))
    time.sleep(0.05)
    os.utime(s, None)
    assert RG.already_scored(c, s, v) is None


def test_already_scored_force_env_disables_the_guard(tmp_path, monkeypatch):
    c, s, v = _bank(tmp_path)
    assert RG.already_scored(c, s, v, seam_sha="abc") is not None
    monkeypatch.setenv("R7_FORCE_RESCORE", "1")
    assert RG.already_scored(c, s, v, seam_sha="abc") is None


# --------------------------------------------------------------------------- #
# one live driver per scorer job                                               #
# --------------------------------------------------------------------------- #
def test_joblock_second_driver_waits_and_a_dead_holder_is_reclaimed(tmp_path):
    a = RG.JobLock("navtest_r7s50400_R7_A1", str(tmp_path))
    b = RG.JobLock("navtest_r7s50400_R7_A1", str(tmp_path))
    assert a.try_acquire() is True
    assert b.try_acquire() is False                 # same live pid holds it -> a duplicate must wait
    with pytest.raises(TimeoutError):
        b.acquire(poll_s=0.01, max_wait_s=0.05, log=lambda m: None, sleep=lambda s: time.sleep(0.01))
    a.release()
    assert b.try_acquire() is True                  # free again
    b.release()
    dead = 999999
    while psutil.pid_exists(dead):
        dead += 1
    open(a.path, "w").write(json.dumps({"pid": dead, "create_time": 1.0}))
    assert a.try_acquire() is True                  # a dead driver's lock is debris
    a.release()
    assert not os.path.exists(a.path)


def test_joblock_keys_are_per_job(tmp_path):
    a = RG.JobLock("navtest_r7s50400_R7_A1", str(tmp_path))
    c = RG.JobLock("navtest_r7s50400_R7_A1_s1", str(tmp_path))
    assert a.try_acquire() and c.try_acquire()
    a.release(); c.release()


def test_joblock_release_never_removes_another_drivers_lock(tmp_path):
    a = RG.JobLock("k", str(tmp_path))
    open(a.path, "w").write(json.dumps({"pid": 1, "create_time": 1.0}))
    a.held = True
    a.release()
    assert os.path.exists(a.path)
