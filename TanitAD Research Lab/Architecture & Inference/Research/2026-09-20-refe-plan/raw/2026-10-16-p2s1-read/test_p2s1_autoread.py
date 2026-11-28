#!/usr/bin/env python3
"""Tests for p2s1_autoread.py.  Prints `ZZTEST_PASS <n>` (exit 0) or `ZZTEST_FAIL ...` (exit 1).  ~20 s, no GPU, no network, no real job.

The REAL (non-dry) code path is exercised end to end with a scripted HF backend and STUB commands in place of stage 2 / stage 3 / the reader:
the marker logic, the sha256 refusal, the stage ordering (attempt 1 may fail only if the seam exists), the markers, the token hygiene, the GPU gate,
the single-instance lock, resume; then the SOURCE is edited (one line each) and the check that guards it must go RED.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p2s1_autoread as A  # noqa: E402

SRC = (HERE / "p2s1_autoread.py").read_text(encoding="utf-8")
N_OK, FAILS = 0, []
FAKE_TOKEN = "hf_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6"


def check(name, cond, detail=""):
    global N_OK
    if cond:
        N_OK += 1
    else:
        FAILS.append(f"{name} {detail}".strip())
        print("  FAIL", name, detail, flush=True)


def mutated(old, new):
    assert SRC.count(old) == 1, f"mutation anchor not unique: {old!r} x{SRC.count(old)}"
    m = types.ModuleType("p2s1_autoread_mut")
    m.__file__ = str(HERE / "p2s1_autoread.py")
    exec(compile(SRC.replace(old, new), str(HERE / "p2s1_autoread.py"), "exec"), m.__dict__)
    return m


# ---------------------------------------------------------------- scenario builders
BLOB = b"stand-in for model_final.pt " * 97 + b"\x00\xff"
SHA = hashlib.sha256(BLOB).hexdigest()
SUMM_DONE = json.dumps({"done": True, "steps": 1209, "model_final_sha256": SHA})


def tree_final(**over):
    t = {"run/summary.json": {"size": 100, "sha256": None}, "run/model_final.pt": {"size": len(BLOB), "sha256": SHA},
         "logs/chain.log": {"size": 10, "sha256": None}}
    t.update(over)
    return t


def chain(sha=SHA, failed_after=False):
    s = f"x ZZP2REP_FINAL_OK model_final sha256 {sha} remote=sha256-verified\n"
    return s + ("y ZZP2REP_FINAL_FAILED (final.log)\n" if failed_after else "")


def tx(tree, text=None):
    text = {"run/summary.json": SUMM_DONE, "logs/chain.log": chain(), **(text or {})}
    return lambda p: text[p]


def marker_cases(mod):
    """{case: bool}: the truth table of final_marker"""
    fm = mod.final_marker
    out = {}
    out["no_repo_waits"] = fm(None, tx(None))["status"] == "waiting"
    out["no_summary_waits"] = fm({}, tx({}))["status"] == "waiting"
    out["summary_not_done_waits"] = fm({"run/summary.json": {}}, tx(None, {"run/summary.json": json.dumps({"done": False})}))["status"] == "waiting"
    out["done_without_model_final_WAITS"] = fm({"run/summary.json": {}}, tx(None))["status"] == "waiting"            # the light-tick race
    out["done_without_sha_inconsistent"] = fm({"run/summary.json": {}, "run/model_final.pt": {"size": 1}}, tx(None, {"run/summary.json": json.dumps({"done": True})}))["status"] == "inconsistent"
    out["lfs_mismatch_inconsistent"] = fm(tree_final(**{"run/model_final.pt": {"size": 1, "sha256": "1" * 64}}), tx(None))["status"] == "inconsistent"
    out["chain_sha_mismatch_inconsistent"] = fm(tree_final(), tx(None, {"logs/chain.log": chain("2" * 64)}))["status"] == "inconsistent"
    f = fm(tree_final(), tx(None))
    out["consistent_final"] = f["status"] == "final" and f["sha256"] == SHA and f["size"] == len(BLOB) and f["server_hash_verified"] and f["chain_final_ok"]
    out["final_without_chain_log"] = fm({k: v for k, v in tree_final().items() if k != "logs/chain.log"}, tx(None))["status"] == "final"
    out["final_without_server_hash_still_final"] = fm(tree_final(**{"run/model_final.pt": {"size": len(BLOB), "sha256": None}}), tx(None, {"logs/chain.log": "nothing yet\n"}))["status"] == "final"
    out["chain_final_failed_and_no_server_hash_waits"] = fm(tree_final(**{"run/model_final.pt": {"size": len(BLOB), "sha256": None}}), tx(None, {"logs/chain.log": chain(failed_after=True)}))["status"] == "waiting"
    out["unreadable_summary_waits"] = fm({"run/summary.json": {}}, lambda p: "{not json")["status"] == "waiting"
    return out


# ---------------------------------------------------------------- real-path harness with stub commands
STUB = r'''
import json, os, sys
from pathlib import Path
role, root = sys.argv[1], Path(sys.argv[2])
(root / f"ran_{role}").write_text(json.dumps({"argv": sys.argv[1:], "has_hf_token_env": any(k in os.environ for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"))}))
mode = (root / "mode.json").read_text() if (root / "mode.json").exists() else "{}"
mode = json.loads(mode)
if role == "att1":
    if not mode.get("no_seam"):
        (root / "seam.npz").write_bytes(b"npz")
    sys.exit(1)                                  # attempt 1 ends INCOMPLETE (seam failed) BY DESIGN
if role == "att2":
    sys.exit(mode.get("att2_rc", 0))
if role == "stage3":
    if not mode.get("no_r2"):
        (root / "r2_result.json").write_text("{}")
    sys.exit(1 if mode.get("stage3_invalid") else 0)
if role == "read":
    out = Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
    inv = mode.get("read_outcome") == "INVALID"
    res = {"outcome": mode.get("read_outcome", "REPLICATED"), "failed_gates": ["stage3:harness_status"] if inv else [],
           "RR1": {"primary": {"mean": 0.55, "ci95": [0.1, 1.0]}, "adverse_subscores": []},
           "RR2": {"delta_seed_mean": 0.05, "delta_seed_ci95": [-0.2, 0.3], "F": 0.05}, "D_s0": 0.57, "D_s1": 0.55}
    (out / "p2rep_result.json").write_text(json.dumps(res))
    sys.exit(2 if inv else 0)
'''


class Rig:
    def __init__(self, mod, td: Path, steps, mode=None, **cfg_over):
        self.mod, self.td = mod, td
        mod.LOCK, mod.MARK = str(td / "gpu.lock"), str(td / "chain_active")        # NEVER the real shared dev-box lock
        (td / "stub.py").write_text(STUB)
        (td / "mode.json").write_text(json.dumps(mode or {}))
        py = sys.executable
        self.out = td / "out"

        def plan_hook(plan):
            r = str(td)
            return {**plan,
                    "idle_gate": [py, "-c", "import sys; sys.exit(0)"],
                    "stage2_attempt1": [py, str(td / "stub.py"), "att1", r], "stage2_attempt2": [py, str(td / "stub.py"), "att2", r],
                    "stage3": [py, str(td / "stub.py"), "stage3", r], "read": [py, str(td / "stub.py"), "read", r, str(self.out / "read")],
                    "seam_npz": str(td / "seam.npz"), "r2_result": str(td / "r2_result.json")}

        self.cfg = {"repo": "x/y", "name": "p2s1_full", "pkg_root": str(td), "data": str(td), "py": py, "wait_idle": "stub", "out": str(self.out),
                    "ck_path": str(td / "ck" / "model_final.pt"), "stage_dir": str(td / "dl"), "poll_s": 0, "max_wait_h": 1, "max_errors": 3, "max_inconsistent": 3,
                    "gpu_wait_max_h": 1, "step_timeout_h": 1, "min_free_gib": 0, "force": False, "extra_negative": None,
                    "plan_hook": plan_hook, "preflight_hook": lambda plan: None}
        self.cfg.update(cfg_over)
        self.backend = mod.ScriptedBackend(steps)
        self.job = mod.Job(self.cfg, self.backend, dry=False, sleep=lambda s: None, secrets=[FAKE_TOKEN])

    def go(self):
        os.environ["HF_TOKEN"] = FAKE_TOKEN            # the parent HAS a token in its environment: a child must NOT inherit it
        try:
            return self.job.run()
        finally:
            os.environ.pop("HF_TOKEN", None)

    def marker(self):
        for k in ("P2S1READ_DONE", "P2S1READ_FAILED"):
            if (self.out / k).is_file():
                return k, json.loads((self.out / k).read_text(encoding="utf-8"))
        return None, None

    def ran(self, role):
        return (self.td / f"ran_{role}").is_file()


FINAL_STEP = {"tree": tree_final(), "text": {"run/summary.json": SUMM_DONE, "logs/chain.log": chain()}, "blobs": {"run/model_final.pt": BLOB}}


def scenarios(mod):
    """{name: bool}: the real path under a scripted HF and stub commands (works on any module built from the source, so mutations can be run through it)"""
    res = {}
    with tempfile.TemporaryDirectory() as t:
        td = Path(t) / "happy"; td.mkdir()
        r = Rig(mod, td, [{"tree": None, "text": {}, "blobs": {}}, FINAL_STEP])
        rc = r.go(); k, m = r.marker()
        res["happy_done"] = rc == 0 and k == "P2S1READ_DONE" and m["outcome"] == "REPLICATED" and m["sha256"] == SHA
        res["happy_ran_all_four_in_order"] = all(r.ran(x) for x in ("att1", "att2", "stage3", "read")) and (
            (td / "ran_att1").stat().st_mtime_ns <= (td / "ran_att2").stat().st_mtime_ns <= (td / "ran_stage3").stat().st_mtime_ns <= (td / "ran_read").stat().st_mtime_ns)
        res["happy_ckpt_landed_and_verified"] = Path(r.cfg["ck_path"]).is_file() and hashlib.sha256(Path(r.cfg["ck_path"]).read_bytes()).hexdigest() == SHA
        res["happy_no_token_in_child_env"] = all(not json.loads((td / f"ran_{x}").read_text())["has_hf_token_env"] for x in ("att1", "att2", "stage3", "read"))
        res["happy_no_token_anywhere_in_out"] = all(FAKE_TOKEN not in p.read_text(encoding="utf-8", errors="ignore") for p in r.out.rglob("*") if p.is_file())
        res["happy_log_is_scrubbed"] = (r.job.scrub(f"x {FAKE_TOKEN} y") == "x hf_*** y") and (r.job.scrub("hf_abcdefghijkl") == "hf_***")
        res["happy_state_phase_done"] = json.loads((r.out / "autoread_state.json").read_text())["phase"] == "done"
    with tempfile.TemporaryDirectory() as t:                                         # corrupt download -> refuse, run nothing
        td = Path(t) / "corrupt"; td.mkdir()
        bad = dict(FINAL_STEP, blobs={"run/model_final.pt": BLOB[:-1] + b"\x00"})
        r = Rig(mod, td, [bad]); rc = r.go(); k, m = r.marker()
        res["corrupt_refused"] = rc == 1 and k == "P2S1READ_FAILED" and m["phase"] == "download" and "REFUSED" in m["reason"] and SHA in m["reason"]
        res["corrupt_leaves_no_ckpt"] = not Path(r.cfg["ck_path"]).exists()
        res["corrupt_runs_nothing"] = not any(r.ran(x) for x in ("att1", "att2", "stage3", "read"))
    with tempfile.TemporaryDirectory() as t:                                         # truncated (right hash impossible, wrong size)
        td = Path(t) / "short"; td.mkdir()
        short = dict(FINAL_STEP, blobs={"run/model_final.pt": BLOB[:100]})
        r = Rig(mod, td, [short]); rc = r.go()
        res["truncated_refused"] = rc == 1 and r.marker()[0] == "P2S1READ_FAILED" and not Path(r.cfg["ck_path"]).exists()
    with tempfile.TemporaryDirectory() as t:                                         # attempt 1 left no seam
        td = Path(t) / "noseam"; td.mkdir()
        r = Rig(mod, td, [FINAL_STEP], mode={"no_seam": True}); rc = r.go(); k, m = r.marker()
        res["no_seam_fails_before_attempt2"] = rc == 1 and k == "P2S1READ_FAILED" and m["phase"] == "stage2_attempt1" and not r.ran("att2") and not r.ran("stage3")
    with tempfile.TemporaryDirectory() as t:                                         # attempt 2 (harness) fails
        td = Path(t) / "att2"; td.mkdir()
        r = Rig(mod, td, [FINAL_STEP], mode={"att2_rc": 1}); rc = r.go(); k, m = r.marker()
        res["attempt2_failure_stops"] = rc == 1 and k == "P2S1READ_FAILED" and not r.ran("stage3")
    with tempfile.TemporaryDirectory() as t:                                         # stage 3 writes no r2_result
        td = Path(t) / "nor2"; td.mkdir()
        r = Rig(mod, td, [FINAL_STEP], mode={"no_r2": True}); rc = r.go(); k, m = r.marker()
        res["no_r2_result_fails_before_read"] = rc == 1 and k == "P2S1READ_FAILED" and not r.ran("read")
    with tempfile.TemporaryDirectory() as t:                                         # an INVALID read is FAILED, never DONE
        td = Path(t) / "invalid"; td.mkdir()
        r = Rig(mod, td, [FINAL_STEP], mode={"read_outcome": "INVALID", "stage3_invalid": True}); rc = r.go(); k, m = r.marker()
        res["invalid_read_is_FAILED_not_DONE"] = rc == 1 and k == "P2S1READ_FAILED" and "INVALID" in m["reason"] and "stage3:harness_status" in m["reason"] and m["outcome"] == "INVALID"
    for oc in ("REPLICATED", "REPLICATED, FLOOR-BOUND", "NOT REPLICATED"):
        with tempfile.TemporaryDirectory() as t:
            td = Path(t) / "oc"; td.mkdir()
            r = Rig(mod, td, [FINAL_STEP], mode={"read_outcome": oc}); rc = r.go(); k, m = r.marker()
            res[f"outcome_{oc}_is_DONE"] = rc == 0 and k == "P2S1READ_DONE" and m["outcome"] == oc
    with tempfile.TemporaryDirectory() as t:                                         # no final marker before the deadline
        td = Path(t) / "deadline"; td.mkdir()
        waiting = {"tree": {"run/summary.json": {}}, "text": {"run/summary.json": json.dumps({"done": False})}, "blobs": {}}
        r = Rig(mod, td, [waiting], max_wait_h=-1); rc = r.go(); k, m = r.marker()
        res["deadline_fails_loud"] = rc == 1 and k == "P2S1READ_FAILED" and m["phase"] == "poll" and "no final marker" in m["reason"]
    with tempfile.TemporaryDirectory() as t:                                         # consecutive poll errors
        td = Path(t) / "errs"; td.mkdir()

        class Boom(mod.ScriptedBackend):
            def tree(self):
                raise ConnectionError("network down " + FAKE_TOKEN)
        r = Rig(mod, td, [FINAL_STEP]); r.job.b = Boom([FINAL_STEP]); rc = r.go(); k, m = r.marker()
        res["poll_errors_fail_after_max"] = rc == 1 and k == "P2S1READ_FAILED" and "consecutive poll errors" in m["reason"] and FAKE_TOKEN not in json.dumps(m)
    with tempfile.TemporaryDirectory() as t:                                         # disagreeing published hashes, persistently
        td = Path(t) / "incons"; td.mkdir()
        inc = {"tree": tree_final(**{"run/model_final.pt": {"size": 1, "sha256": "3" * 64}}), "text": {"run/summary.json": SUMM_DONE, "logs/chain.log": chain()}, "blobs": {}}
        r = Rig(mod, td, [inc]); rc = r.go(); k, m = r.marker()
        res["inconsistent_hashes_fail_after_max"] = rc == 1 and k == "P2S1READ_FAILED" and "disagree" in m["reason"] and not r.ran("att1")
    with tempfile.TemporaryDirectory() as t:                                         # resume semantics
        td = Path(t) / "resume"; td.mkdir()
        r = Rig(mod, td, [FINAL_STEP]); r.go()
        r2 = Rig(mod, td, [FINAL_STEP]); (td / "ran_att1").unlink()
        rc = r2.go()
        res["done_marker_makes_relaunch_a_noop"] = rc == 0 and not r2.ran("att1") and r2.marker()[0] == "P2S1READ_DONE"
        r2.cfg["force"] = True
        rc = mod.Job(r2.cfg, mod.ScriptedBackend([FINAL_STEP]), dry=False, sleep=lambda s: None).run()
        res["force_reruns_and_reuses_a_verified_ckpt"] = rc == 0 and r2.ran("att1")
    with tempfile.TemporaryDirectory() as t:                                         # a FAILED marker is cleared by the relaunch
        td = Path(t) / "retry"; td.mkdir()
        r = Rig(mod, td, [FINAL_STEP], mode={"no_seam": True}); r.go()
        (td / "mode.json").write_text("{}")
        r3 = Rig(mod, td, [FINAL_STEP]); (td / "mode.json").write_text("{}")
        rc = r3.go()
        res["relaunch_after_FAILED_retries_and_clears_it"] = rc == 0 and r3.marker()[0] == "P2S1READ_DONE" and not (r3.out / "P2S1READ_FAILED").exists()
    return res


def main() -> int:
    # ---- pure pieces
    for k, v in marker_cases(A).items():
        check(f"marker_{k}", v)
    # ---- build_plan == the runbook section 3 command, argument for argument
    cfg = {"py": "PY", "pkg_root": "P:/pk", "data": "D:/data", "ck_path": "C:/ck/model_final.pt", "name": "p2s1_full", "wait_idle": "W", "out": "O"}
    pl = A.build_plan(cfg, SHA)
    want2 = ["PY", str(Path("P:/pk/eval/stage2_sft.py")), "--ckpt", "C:/ck/model_final.pt", "--name", "p2s1_full", "--rule", "navsim_v1", "--tokens", "D:/data/navtest_full_tokens.json"]
    check("plan_stage2_is_runbook_s3", pl["stage2_attempt1"] == want2 and pl["stage2_attempt2"] == want2)
    check("plan_stage2_attempts_are_identical_not_aliased", pl["stage2_attempt1"] is not pl["stage2_attempt2"])
    check("plan_stage3_is_runbook_s3", pl["stage3"] == ["PY", str(Path("P:/pk/eval/stage3_full.py")), "--name", "p2s1_full", "--ckpt", "C:/ck/model_final.pt", "--expect-ckpt-sha256", SHA])
    check("plan_same_ckpt_string_in_stage2_and_stage3", pl["stage2_attempt1"][3] == pl["stage3"][5])
    check("plan_name_is_p2s1_full", pl["stage2_attempt1"][5] == "p2s1_full" and pl["stage3"][3] == "p2s1_full")
    check("plan_no_exploratory_flag", "--exploratory" not in (pl["stage2_attempt1"] + pl["stage3"]))
    check("plan_no_token_shaped_argv", not any(A.TOKEN_RE.search(a) for k in ("idle_gate", "stage2_attempt1", "stage2_attempt2", "stage3", "read") for a in pl[k]))
    check("plan_seam_and_r2_paths", pl["seam_npz"] == "D:/data/seams/stage2_p2s1_full.npz" and pl["r2_result"].replace("\\", "/").endswith("raw/p2_readout/p2s1_full/r2_result.json"))
    check("plan_read_reads_the_stage3_record", "--r2-result" in pl["read"] and pl["read"][pl["read"].index("--r2-result") + 1] == pl["r2_result"])
    # ---- the real path
    for k, v in scenarios(A).items():
        check(f"real_{k}", v)
    # ---- token hygiene
    with tempfile.TemporaryDirectory() as t:
        kf = Path(t) / "Keys.txt"
        kf.write_text(f"Openrouter key\nsk-or-v1-0123456789abcdef\n\nhuggingface key:\n{FAKE_TOKEN}\nanother: nothing\n", encoding="utf-8")
        check("read_token_in_place_first_hf_match", A.read_token([str(Path(t) / "absent.txt"), str(kf)]) == FAKE_TOKEN)
        check("read_token_none_when_absent", A.read_token([str(Path(t) / "absent.txt")]) is None)
        kf2 = Path(t) / "k2.txt"; kf2.write_text("no token here\n"); check("read_token_none_without_hf", A.read_token([str(kf2)]) is None)
    check("child_env_drops_all_hf_tokens", (lambda e: not any(k in e for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_API_TOKEN")))(
        (lambda: (os.environ.update(HF_TOKEN="x", HUGGING_FACE_HUB_TOKEN="y", HF_API_TOKEN="z"), A.Job.child_env(types.SimpleNamespace()))[1])()))
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_API_TOKEN"):
        os.environ.pop(k, None)
    check("xet_disabled_before_hf_import", os.environ.get("HF_HUB_DISABLE_XET") == "1" and SRC.index("HF_HUB_DISABLE_XET") < SRC.index("from huggingface_hub"))
    # ---- GPU gate and the lock
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        A.LOCK, A.MARK = str(t / "gpu.lock"), str(t / "chain_active")
        job = A.Job({"out": str(t / "o"), "repo": "r", "name": "n"}, None, sleep=lambda s: None)
        check("lock_free", job.lock_state() is None)
        Path(A.LOCK).write_text(json.dumps({"pid": os.getpid(), "owner": "x"}))
        check("lock_held_by_live_pid_is_not_taken", job.lock_state() is not None and Path(A.LOCK).exists())
        os.remove(A.LOCK)
        Path(A.MARK).write_text("x")
        check("chain_active_marker_blocks", job.lock_state() is not None)
        os.remove(A.MARK)
        real_alive = A.pid_alive
        A.pid_alive = lambda pid: False
        # LOCK-STEAL-2 (2026-10-18): only a lock carrying OUR marker (stage2_sft.py's job = "refe-stage2-<name>") may be removed.  The two checks below used to pin the old
        # stealing with a marker-less {"pid", "owner": "dead"} body; the fixture now carries the marker, and the inverse is pinned right after (and in test_p2s1_autoread_foreign_lock.py).
        Path(A.LOCK).write_text(json.dumps({"job": "refe-stage2-n", "pid": 99999999, "host": "H", "acquired": "now"}))
        check("dead_holder_but_fresh_lock_is_kept", job.lock_state() is not None and Path(A.LOCK).exists())
        old = os.path.getmtime(A.LOCK) - 3600
        os.utime(A.LOCK, (old, old))
        check("dead_holder_and_old_lock_is_removed", job.lock_state() is None and not Path(A.LOCK).exists())
        Path(A.LOCK).write_text(json.dumps({"pid": 99999999, "owner": "dead"}))
        os.utime(A.LOCK, (old, old))
        check("dead_holder_old_FOREIGN_lock_is_NOT_removed", job.lock_state() is not None and Path(A.LOCK).exists())
        os.remove(A.LOCK)
        A.pid_alive = real_alive
        # gate loop: held for two polls, then free -> returns
        calls = {"n": 0}

        def sleeper(s):
            calls["n"] += 1
            if calls["n"] == 2 and Path(A.LOCK).exists():
                os.remove(A.LOCK)
        Path(A.LOCK).write_text(json.dumps({"pid": os.getpid(), "owner": "x"}))
        job2 = A.Job({"out": str(t / "o2"), "repo": "r", "name": "n", "gpu_wait_max_h": 1}, None, sleep=sleeper)
        job2.gpu_gate({"idle_gate": [sys.executable, "-c", "import sys; sys.exit(0)"]})
        check("gpu_gate_waits_for_the_lock_then_hands_over", calls["n"] == 2 and not Path(A.LOCK).exists())
    # ---- single instance
    with tempfile.TemporaryDirectory() as t:
        lk = Path(t) / "autoread.lock"
        with A.SingleInstance(lk):
            check("single_instance_lock_created", lk.is_file())
            # our own pid holds it: a second entrant with another pid must be refused. Fake the holder as the parent shell (alive).
            lk.write_text(json.dumps({"pid": os.getppid(), "started": "x"}))
            try:
                with A.SingleInstance(lk):
                    refused = False
            except SystemExit as e:
                refused = "another instance" in str(e)
            check("second_instance_refused", refused)
            lk.write_text(json.dumps({"pid": os.getpid()}))
        check("lock_removed_on_exit", not lk.exists())
        lk.write_text(json.dumps({"pid": 99999999, "started": "dead"}))
        real_alive = A.pid_alive; A.pid_alive = lambda pid: False
        with A.SingleInstance(lk):
            check("stale_instance_lock_is_taken_over", json.loads(lk.read_text())["pid"] == os.getpid())
        A.pid_alive = real_alive
    # ---- the dry run end to end, as a subprocess (the exact command the handoff names)
    import subprocess
    r = subprocess.run([sys.executable, str(HERE / "p2s1_autoread.py"), "--dry-run", "--no-live"], capture_output=True, text=True, encoding="utf-8")
    check("dry_run_cli_passes", r.returncode == 0 and "ZZP2S1AUTOREAD_DRYRUN_PASS" in r.stdout, r.stdout[-400:] + r.stderr[-400:])
    check("dry_run_walked_poll_download_plan", all(s in r.stdout for s in ("poll 1: waiting", "poll 3: waiting", "poll 4: inconsistent", "poll 5: FINAL", "download: downloaded", "negative control OK", "WOULD RUN [stage3]", "WOULD RUN [read]")))
    check("dry_run_never_writes_real_markers", not (HERE / "out" / "P2S1READ_DONE").exists() and not (HERE / "out" / "P2S1READ_FAILED").exists() and (HERE / "out" / "dryrun" / "P2S1READ_DRYRUN_OK").is_file())
    check("dry_run_prints_no_token", not A.TOKEN_RE.search(r.stdout + r.stderr))

    # ---- mutations: edit the SOURCE, the guarding scenario/check must go RED
    muts = [
        ("N1 done=true alone is the marker (model_final.pt not required)", '    if M not in tree:\n        return w(', '    if False:\n        return w(', "marker"),
        ("N2 the published-hash comparison dropped from the download", "if got != sha or (size and got_size != size):", "if False:", "scen"),
        ("N3 continue past a dead seam (attempt 1 gate removed)", "if not self.dry and not Path(plan[\"seam_npz\"]).is_file():", "if False:", "scen"),
        ("N4 children inherit HF_TOKEN", 'if k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_API_TOKEN")}', 'if True}', "scen"),
        ("N5 an INVALID read is recorded as DONE", 'if res["outcome"] == "INVALID":', 'if False:', "scen"),
        ("N6 LFS-vs-summary hash disagreement ignored", "if lfs is not None and lfs != sha:", "if False:", "marker"),
        ("N7 attempt 2 failure ignored", "        if rc2 != 0:\n", "        if False:\n", "scen"),
        ("N8 the token is not scrubbed from logs", '        s = TOKEN_RE.sub("hf_***", str(s))\n        for sec in self.secrets:\n            s = s.replace(sec, "hf_***")\n', '        s = str(s)\n', "scen"),
    ]
    for label, old, new, which in muts:
        mod = mutated(old, new)
        try:
            red = (not all(marker_cases(mod).values())) if which == "marker" else (not all(scenarios(mod).values()))
        except Exception:                                # the mutation broke the code under test: a failing check, by any name
            red = True
        check(f"mutation_RED: {label}", red)
    check("unmutated_source_is_GREEN", all(marker_cases(A).values()) and all(scenarios(A).values()))

    if FAILS:
        print(f"ZZTEST_FAIL {len(FAILS)} failed, {N_OK} passed: " + " | ".join(FAILS))
        return 1
    print(f"ZZTEST_PASS {N_OK}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
