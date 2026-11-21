"""Control-flow tests of run_navfeed.sh + wait_navfeed.sh in a SANDBOX with stub python tools (no GPU, no model, no foreign lock touched).

The shell code under test is the REAL one (copied byte-for-byte into the sandbox); only the python tools it calls are stubs that honour the same
artifact contracts (gpu_lock.py acquire/release JSON, wait_commit.py JSON, navfeed_roll.py dump + roll record, smoke_check.py JSON, ...).  The sandbox
has its own lock file, kit, out dir and prereg, so nothing outside pytest's tmp dir is read or written (the REAL GPU lock path is never named).

    cd <hold>/code/tests && C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q -p no:cacheprovider test_wrapper_flow.py

Cases: bad seal -> 21 | happy path (marker ORDER, roll ORDER, env, lock released, results copied) | YIELD to the final chain after the first roll, then
resume WITHOUT redoing it | smoke failure stops the slot | a foreign lock is never broken | the waiter waits on a held lock, goes when it frees, and is single-instance.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

HOLD = Path(__file__).resolve().parents[2]
SH = "C:/Program Files/Git/usr/bin/sh.exe"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"


def msys(p) -> str:
    p = str(p).replace("\\", "/")
    return "/" + p[0].lower() + p[2:] if len(p) > 1 and p[1] == ":" else p


def win(p) -> str:
    return str(p).replace("\\", "/")


STUB_ROLL = r'''
import argparse, json, os, sys, time
from pathlib import Path
ap = argparse.ArgumentParser()
for k in ("--ckpt", "--config", "--nav-feed", "--seed", "--dump-dir", "--windows-json", "--out-json", "--device"):
    ap.add_argument(k)
ap.add_argument("--expect-windows")
a = ap.parse_args()
n = sum(len(v) for v in json.load(open(a.windows_json)).values())
with open(os.environ["STUB_CALLS"], "a") as f:
    f.write(json.dumps({"arm": a.nav_feed, "seed": int(a.seed), "n": n, "omp": os.environ.get("OMP_NUM_THREADS"),
                        "nopose": os.environ.get("R8_NO_POSE_SYNC"), "lock_present": os.path.exists(os.environ["STUB_LOCKF"])}) + "\n")
d = Path(a.dump_dir); d.mkdir(parents=True, exist_ok=True)
json.dump({"grid": {"n_windows": n}, "nav_feed": {"mode": a.nav_feed}}, open(d / "manifest.json", "w"))
json.dump({"done": os.environ.get("STUB_ROLL_FAIL") != a.nav_feed + a.seed, "nav_feed": a.nav_feed, "seed": int(a.seed), "n_windows": n},
          open(a.out_json, "w"))
if n > 100 and os.environ.get("STUB_BANK_AFTER_FIRST_ROLL") and not Path(os.environ["Y48_BANK"]).exists():
    Path(os.environ["Y48_BANK"]).write_text("x")      # the final chain's weights get banked while we roll
'''
STUB_LOCK = r'''
import argparse, json, os, sys
LOCK = os.environ["STUB_LOCKF"]
ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--job"); ap.add_argument("--pid"); ap.add_argument("--max-wait-s"); ap.add_argument("--out")
a = ap.parse_args()
if a.cmd == "acquire":
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, json.dumps({"job": a.job, "pid": a.pid}).encode()); os.close(fd); r = {"ok": True}
    except FileExistsError:
        r = {"ok": False, "holder": open(LOCK).read()}
else:
    cur = json.load(open(LOCK)) if os.path.exists(LOCK) else None
    if cur and cur.get("job") == a.job and str(cur.get("pid")) == str(a.pid):
        os.remove(LOCK); r = {"released": True}
    else:
        r = {"released": False, "why": "not ours"}
print(json.dumps(r))
if a.out: json.dump(r, open(a.out, "w"))
'''
STUB_COMMIT = r'''
import argparse, json
ap = argparse.ArgumentParser(); ap.add_argument("--min-gb"); ap.add_argument("--max-wait-s"); ap.add_argument("--out")
a = ap.parse_args(); json.dump({"met": True}, open(a.out, "w"))
'''
STUB_WINDOWS = r'''
import argparse, json
ap = argparse.ArgumentParser(); ap.add_argument("--windows"); ap.add_argument("--archive-dump"); ap.add_argument("--out"); ap.add_argument("--smoke-out")
a = ap.parse_args()
json.dump({"c%d" % i: list(range(1457 // 95 + (1 if i < 1457 % 95 else 0))) for i in range(95)}, open(a.out, "w"))
json.dump({"c0": [1, 2, 3, 4, 5, 6]}, open(a.smoke_out, "w"))
print(json.dumps({"n_windows": 1457, "n_episodes": 95, "smoke_episode_index": 0}))
'''
STUB_SMOKE = r'''
import argparse, json, os
ap = argparse.ArgumentParser()
for k in ("--clip-dump", "--full-dump", "--windows", "--smoke-episode-index", "--out"): ap.add_argument(k)
a = ap.parse_args(); json.dump({"ok": os.environ.get("STUB_SMOKE_FAIL") != "1", "checks": {}}, open(a.out, "w"))
'''
STUB_ANALYZE = r'''
import argparse, json
ap = argparse.ArgumentParser()
for k in ("--out-dir", "--windows", "--prereg"): ap.add_argument(k)
a = ap.parse_args()
json.dump({"committed_label": {"label": "FEED-FIX-NULL"}}, open(a.out_dir + "/NAV_FEED.json", "w"))
open(a.out_dir + "/NAV_FEED.md", "w").write("stub")
'''


APPENDED = "\nappended after sealing (every sealed code hash is still present, so ONLY the sha256 seal can catch this)\n"


class Sandbox:
    def __init__(self, tmp: Path, mutate=None):
        self.root = tmp
        self.hold = tmp / "hold"
        self.out = tmp / "out"
        self.bat = tmp / "bat"
        self.kit = tmp / "kit"
        for d in (self.hold / "code" / "overlay", self.hold / "raw", self.bat, self.kit):
            d.mkdir(parents=True, exist_ok=True)
        for f in ("run_navfeed.sh", "navfeed_common.sh", "wait_navfeed.sh"):
            shutil.copyfile(HOLD / "code" / f, self.hold / "code" / f)          # the REAL shell code
        if mutate:                      # deliberate-regression arm: break ONE guard in the sandbox copy (bytes in/out: no CRLF translation of a shell script)
            f, old, new = mutate
            raw = (self.hold / "code" / f).read_bytes()
            assert raw.count(old.encode()) == 1, (f, old, raw.count(old.encode()))        # the anchor must exist exactly once or the mutation proves nothing
            (self.hold / "code" / f).write_bytes(raw.replace(old.encode(), new.encode()))
        (self.hold / "code" / "overlay" / "navfeed_overlay.py").write_text("# stub overlay\n")
        (self.hold / "code" / "overlay" / "navfeed_roll.py").write_text(STUB_ROLL)
        (self.hold / "code" / "make_windows_navfeed.py").write_text("# stub\n")
        (self.hold / "code" / "build_roll_windows.py").write_text(STUB_WINDOWS)
        (self.hold / "code" / "smoke_check.py").write_text(STUB_SMOKE)
        (self.hold / "code" / "nav_feed_analyze.py").write_text(STUB_ANALYZE)
        (self.hold / "raw" / "windows_navfeed.json").write_text("{}")
        (self.bat / "gpu_lock.py").write_text(STUB_LOCK)
        (self.bat / "wait_commit.py").write_text(STUB_COMMIT)
        (self.kit / "ckpt.pt").write_bytes(b"sandbox checkpoint bytes")
        self.lock = tmp / "devbox_gpu.lock"
        self.calls = tmp / "calls.log"
        self.y48_bank = tmp / "ckpt_full_48852.pt"
        self.y48_log = tmp / "chain_final.log"
        self.y48_sum = tmp / "battery_summary.json"
        self.seal()

    def seal(self):
        names = ["code/overlay/navfeed_overlay.py", "code/overlay/navfeed_roll.py", "code/make_windows_navfeed.py", "code/build_roll_windows.py",
                 "code/smoke_check.py", "code/nav_feed_analyze.py", "raw/windows_navfeed.json"]
        body = "sandbox prereg\n" + "\n".join(f"| {n} | {hashlib.sha256((self.hold / n).read_bytes()).hexdigest()} |" for n in names) + "\n"
        pre = self.hold / "PREREG_NAV_FEED.md"
        pre.write_text(body)
        (self.hold / "PREREG_NAV_FEED.md.sha256").write_text(hashlib.sha256(pre.read_bytes()).hexdigest() + "  PREREG_NAV_FEED.md\n")

    def env(self, **extra):
        e = dict(os.environ)
        e.update({"NF_HOLDW": win(self.hold), "NF_HOLDL": msys(self.hold), "NF_OUTW": win(self.out), "NF_OUTL": msys(self.out),
                  "NF_BAT": win(self.bat), "NF_CKPT": win(self.kit / "ckpt.pt"), "NF_CKPT_L": msys(self.kit / "ckpt.pt"),
                  "NF_CKPT_MD5": hashlib.md5(b"sandbox checkpoint bytes").hexdigest(), "NF_MIN_FREE_MB": "50", "NF_COMMIT_WAIT_S": "5",
                  "NF_LOCK_WAIT_S": "5", "NF_POLL_S": "1", "NF_LOCKF": msys(self.lock), "STUB_LOCKF": win(self.lock), "STUB_CALLS": win(self.calls),
                  "Y48_BANK": msys(self.y48_bank), "Y48_LOG": msys(self.y48_log), "Y48_SUM": msys(self.y48_sum), "NF_PY": msys(PY)})
        e.update(extra)
        return e

    def run(self, script, timeout=120, **extra):
        return subprocess.run([SH, msys(self.hold / "code" / script)], env=self.env(**extra), capture_output=True, text=True, timeout=timeout,
                              stdin=subprocess.DEVNULL)

    def markers(self):
        p = self.out / "navfeed.log"
        if not p.exists():
            return []
        out = []
        for ln in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if ln.startswith("ZZNF"):
                out.append(ln.split("ZZ")[1][2:])
        return out

    def rolls(self):
        if not self.calls.exists():
            return []
        return [json.loads(x) for x in self.calls.read_text().splitlines()]


@pytest.fixture()
def sb(tmp_path):
    return Sandbox(tmp_path)


def test_bad_seal_stops_before_any_gpu_work(sb):
    (sb.hold / "PREREG_NAV_FEED.md").write_text("edited after sealing\n")
    r = sb.run("run_navfeed.sh")
    assert r.returncode == 21 and "SEALBAD" in sb.markers() and not sb.rolls() and not sb.lock.exists()


def test_a_sealed_file_that_changed_stops_the_run(sb):
    (sb.hold / "code" / "overlay" / "navfeed_roll.py").write_text(STUB_ROLL + "\n# edited after sealing\n")
    r = sb.run("run_navfeed.sh")
    assert r.returncode == 21 and "SEALBAD" in sb.markers() and not sb.rolls()


def test_happy_path_order_env_lock_and_results(sb):
    r = sb.run("run_navfeed.sh")
    assert r.returncode == 0, r.stderr[-800:]
    m = sb.markers()
    expect = ["START", "SEALOK", "LOCKED", "CKPTOK", "SMOKESTART", "SMOKEOK"] + ["ROLLSTART", "ROLLDONE"] * 6 + ["RELEASED", "ANALYSIS", "DONE"]
    assert m == expect, m
    rolls = sb.rolls()
    assert [(x["arm"], x["seed"]) for x in rolls] == [("clip", 0), ("perframe_full", 0),                                   # the smoke pair
                                                      ("clip", 0), ("perframe_full", 0), ("clip", 1), ("perframe_full", 1),
                                                      ("perframe_token", 0), ("perframe_token", 1)]
    assert [x["n"] for x in rolls] == [6, 6] + [1457] * 6
    assert all(x["omp"] == "6" and x["nopose"] == "1" and x["lock_present"] for x in rolls)                              # env + the lock held DURING every roll
    assert not sb.lock.exists()                                                                                           # released
    assert (sb.hold / "raw" / "run" / "NAV_FEED.json").exists() and (sb.hold / "raw" / "run" / "navfeed.log").exists()
    assert sorted(p.name for p in (sb.hold / "raw" / "run").glob("roll_*.json")) == sorted(
        f"roll_{a}_s{s}.json" for a, s in [("clip", 0), ("perframe_full", 0), ("clip", 1), ("perframe_full", 1), ("perframe_token", 0), ("perframe_token", 1)])


def test_yield_to_the_final_chain_then_resume_without_redoing_work(sb):
    # a fake live chain_final_r8 process + a log that says it is open + (banked weights appear after our first roll)
    sb.y48_log.write_text("ZZCFNSTARTZZ 2026 pid 1\n")
    chain = sb.root / "chain_final_r8.sh"
    chain.write_text("sleep 90\n")
    proc = subprocess.Popen([SH, msys(chain)], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(1.5)
        r = sb.run("run_navfeed.sh", STUB_BANK_AFTER_FIRST_ROLL="1")
        assert r.returncode == 10, (r.returncode, r.stderr[-500:], sb.markers())
        m = sb.markers()
        assert m.count("ROLLDONE") == 1 and "YIELD" in m and m[-2:] == ["YIELD", "RELEASED"], m
        assert not sb.lock.exists()                                                                                       # a job never holds the lock while yielding
        done_first = [(x["arm"], x["seed"]) for x in sb.rolls() if x["n"] == 1457]
        assert done_first == [("clip", 0)]
        # the chain ends -> we resume; the finished roll is NOT redone and the smoke is NOT rerun
        sb.y48_log.write_text("ZZCFNSTARTZZ 2026 pid 1\nZZCFNENDZZ 2026 done\n")
        r2 = sb.run("run_navfeed.sh")
        assert r2.returncode == 0, (r2.stderr[-500:], sb.markers())
        big = [(x["arm"], x["seed"]) for x in sb.rolls() if x["n"] == 1457]
        assert big == [("clip", 0), ("perframe_full", 0), ("clip", 1), ("perframe_full", 1), ("perframe_token", 0), ("perframe_token", 1)]
        assert sum(1 for x in sb.rolls() if x["n"] == 6) == 2 and "ROLLHAVE" in sb.markers()
    finally:
        proc.kill()


def test_smoke_failure_spends_no_more_gpu_and_releases(sb):
    r = sb.run("run_navfeed.sh", STUB_SMOKE_FAIL="1")
    assert r.returncode == 20 and "SMOKEFAIL" in sb.markers() and not any(x["n"] == 1457 for x in sb.rolls()) and not sb.lock.exists()


def test_a_roll_that_does_not_complete_is_a_failure_not_a_skip(sb):
    r = sb.run("run_navfeed.sh", STUB_ROLL_FAIL="perframe_full0")
    assert r.returncode == 20 and "ROLLFAIL" in sb.markers() and not sb.lock.exists()
    assert sb.markers().count("ROLLDONE") == 1                                                                           # clip s0 done, full s0 failed


def test_a_foreign_lock_is_never_broken(sb):
    sb.lock.write_text(json.dumps({"job": "someone-else", "pid": 4242}))
    before = sb.lock.read_text()
    r = sb.run("run_navfeed.sh")
    assert r.returncode == 11 and "NOLOCK" in sb.markers() and sb.lock.read_text() == before and not sb.rolls()


def test_disk_gate_refuses_when_free_space_is_below_the_bar(sb):
    r = sb.run("run_navfeed.sh", NF_MIN_FREE_MB="999999999")
    assert r.returncode == 11 and "NODISK" in sb.markers() and not sb.rolls() and not sb.lock.exists()


def test_waiter_waits_on_a_held_lock_goes_when_free_and_is_single_instance(sb):
    sb.lock.write_text(json.dumps({"job": "someone-else", "pid": 4242}))
    w = subprocess.Popen([SH, msys(sb.hold / "code" / "wait_navfeed.sh")], env=sb.env(), stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(4)
        assert w.poll() is None and "WAITERGO" not in sb.markers() and not sb.rolls()                                    # waiting on the foreign lock
        dup = sb.run("wait_navfeed.sh", timeout=30)
        assert dup.returncode == 5 and "DUP" in sb.markers()                                                             # single instance
        sb.lock.unlink()
        for _ in range(90):
            if w.poll() is not None:
                break
            time.sleep(1)
        assert w.poll() == 0, (w.poll(), sb.markers())
        m = sb.markers()
        assert "WAITERGO" in m and m[-1] == "WAITERDONE" and "DONE" in m
        assert not (sb.out / "state" / "waiter.pid").exists()
    finally:
        if w.poll() is None:
            w.kill()


def test_waiter_halts_instead_of_retrying_a_failed_gpu_run(sb):
    w = subprocess.run([SH, msys(sb.hold / "code" / "wait_navfeed.sh")], env=sb.env(STUB_SMOKE_FAIL="1"), stdin=subprocess.DEVNULL,
                       capture_output=True, text=True, timeout=90)
    assert w.returncode == 20 and "WAITERHALT" in sb.markers() and sum(1 for x in sb.rolls() if x["n"] == 6) == 2        # ONE smoke attempt, no retry loop


# ---- deliberate-regression arms: break ONE guard in the sandbox copy and the matching expectation above must FAIL ---------------------------------- #
def test_MUTATION_seal_check_disabled_lets_a_tampered_prereg_run(tmp_path):
    sb = Sandbox(tmp_path, mutate=("run_navfeed.sh", 'never a match)"; exit 21; fi', 'never a match)"; fi'))
    pre = sb.hold / "PREREG_NAV_FEED.md"
    pre.write_text(pre.read_text() + APPENDED)
    r = sb.run("run_navfeed.sh")
    # the intact guard gave 21 + zero rolls in test_bad_seal_stops_before_any_gpu_work; with the guard broken the tampered prereg RUNS
    assert r.returncode == 0 and sb.rolls()


def test_the_seal_guard_catches_the_same_append_tamper_when_intact(tmp_path):
    sb = Sandbox(tmp_path)
    pre = sb.hold / "PREREG_NAV_FEED.md"
    pre.write_text(pre.read_text() + APPENDED)
    r = sb.run("run_navfeed.sh")
    assert r.returncode == 21 and not sb.rolls()                              # the control arm of the mutation above


def test_MUTATION_yield_disabled_never_yields(tmp_path):
    sb = Sandbox(tmp_path, mutate=("run_navfeed.sh", '[ "$FIRST" = 0 ] && bfinal_pending', '[ "$FIRST" = 0 ] && false'))
    sb.y48_log.write_text("ZZCFNSTARTZZ 2026 pid 1\n")
    chain = sb.root / "chain_final_r8.sh"
    chain.write_text("sleep 60\n")
    proc = subprocess.Popen([SH, msys(chain)], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(1.5)
        r = sb.run("run_navfeed.sh", STUB_BANK_AFTER_FIRST_ROLL="1")
        assert r.returncode == 0 and "YIELD" not in sb.markers()          # i.e. test_yield_... would be RED
    finally:
        proc.kill()


def test_MUTATION_no_exit_trap_leaks_the_lock_on_failure(tmp_path):
    sb = Sandbox(tmp_path, mutate=("run_navfeed.sh", "trap on_exit EXIT", "true"))
    r = sb.run("run_navfeed.sh", STUB_SMOKE_FAIL="1")
    assert r.returncode == 20 and sb.lock.exists()                         # a leaked lock would starve every other job: the real wrapper must not do this
