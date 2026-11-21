"""Tests for the chunked P2 full-table pipeline: a known-answer WORLD (hidden per-(token, proposal) PDMS) and a FAKE runner that writes synthetic dumps / harness csvs,
so the real orchestration (chunking, per-chunk gate BEFORE scoring, abort, resume, circuit breaker, lock protocol, guards, table assembly, same-pick proof) runs end
to end in seconds on the CPU, with DELIBERATE-REGRESSION arms that must go RED.  Run:
    C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -I test_p2full.py        (no GPU, no model, no harness)
"""
import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import c3_run_p2full as C  # noqa: E402
import p2full_driver as D  # noqa: E402

K = 64
N_LOGS, PER_LOG = 6, 3               # 18 tokens, chunk size 7 -> 3 chunks of whole logs (6 + 6 + 6)
CHUNK = 7


def toks_doc():
    toks, tl = [], {}
    for li in range(N_LOGS):
        for j in range(PER_LOG):
            t = f"t{li}{j}{(li * 7 + j * 3) % 10}ab"
            toks.append(t)
            tl[t] = f"log_{li:02d}"
    return {"rule": "unit world", "tokens": toks, "token_log": tl}


class World:
    def __init__(self, tmp):
        rng = np.random.default_rng(11)
        self.doc = toks_doc()
        self.toks = self.doc["tokens"]
        n = len(self.toks)
        self.props = rng.normal(size=(n, K, 8, 3)).astype(np.float32)
        self.logits = rng.normal(size=(n, K, 6)).astype(np.float32)
        self.pick = (np.arange(n) * 5) % K
        self.pdms = np.round(rng.uniform(0, 1, size=(n, K)), 6)
        self.pdms[rng.uniform(size=(n, K)) < 0.3] = 0.0
        self.sub = np.round(rng.uniform(0, 1, size=(n, K, 6)), 6)
        self.pdms[0, self.pick[0]] = 0.0                       # token 0 is a "zero token" at the pick
        self.idx = {t: i for i, t in enumerate(self.toks)}
        self.tokfile = os.path.join(tmp, "tokens.json")
        json.dump(self.doc, open(self.tokfile, "w"))
        np.savez(os.path.join(tmp, "banked_seam.npz"), token=np.array(self.toks), fingerprint=np.array([f"fp_{t}" for t in self.toks]),
                 poses=self.props[np.arange(n), self.pick])
        with open(os.path.join(tmp, "banked.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([""] + ["token", "valid"] + list(C.SUB) + ["score"])
            for i, t in enumerate(self.toks):
                w.writerow([i, t, True] + [float(x) for x in self.sub[i, self.pick[i]]] + [float(self.pdms[i, self.pick[i]])])
            w.writerow([n, "average", True] + [0.0] * 6 + [float(self.pdms[np.arange(n), self.pick].mean())])


class Fake:
    """Replaces RealRunner.  knobs: pose_shift_chunk {cid: metres}, bad_ckpt_chunk, bad_fp_chunk, fail_score (callable (cid,k)->bool), crash_dump_on (cid)."""

    def __init__(self, ctx, world, **kw):
        self.ctx, self.w = ctx, world
        self.kw = kw
        self.dump_calls, self.score_calls = [], []
        self.lock = threading.Lock()

    def dump(self, ch, out_part, log):
        cid = ch["id"]
        self.dump_calls.append(cid)
        if self.kw.get("crash_dump_on") == cid:
            raise RuntimeError("simulated crash in the dump of chunk %d" % cid)
        toks = sorted(self.ctx.tokens_of(ch)["tokens"], key=lambda t: self.ctx.tokens_of(ch)["token_log"][t])
        ii = np.array([self.w.idx[t] for t in toks])
        props = self.w.props[ii].copy()
        props[np.arange(len(ii)), self.w.pick[ii], 0, 0] += self.kw.get("pose_shift_chunk", {}).get(cid, 0.0)
        np.savez(out_part, token=np.array(toks), fingerprint=np.array([("bad_" + t) if self.kw.get("bad_fp_chunk") == cid else f"fp_{t}" for t in toks]),
                 proposals=props, logits=self.w.logits[ii], pick=self.w.pick[ii].astype(np.int64), select=np.array("best"), rule=np.array(C.RULE),
                 repair_last_heading=np.array(True), ckpt=np.array("WRONG" if self.kw.get("bad_ckpt_chunk") == cid else C.CKPT), sampling=np.array([8, 0.5]))
        rep = {"tokens_asked": len(toks), "rows": len(toks), "misses": 0, "frame_control": {"max_m": self.kw.get("fc_max", 0.0)}, "seconds": 1.0}
        with open(os.path.splitext(self.ctx.rerun_seam(cid))[0] + ".report.json", "w") as f:
            json.dump(rep, f)
        return 0

    def score(self, ch, k, lp):
        cid = ch["id"]
        with self.lock:
            self.score_calls.append((cid, k))
        if self.kw.get("ram_abort") and self.kw["ram_abort"](cid, k, self.score_calls.count((cid, k))):
            with open(lp, "w") as f:
                f.write("W3_RAM_GUARD_ABORT: available memory below the floor\n")
            return None
        if self.kw.get("fail_score") and self.kw["fail_score"](cid, k):
            open(lp, "w").write(json.dumps({"status": "FAIL", "csv_valid_rows": 0}) + "\n")
            return {"status": "FAIL", "csv_valid_rows": 0}
        toks = self.ctx.tokens_of(ch)["tokens"]
        p = self.ctx.csv_k(cid, k)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow([""] + ["token", "valid"] + list(C.SUB) + ["score"])
            for i, t in enumerate(toks):
                j = self.w.idx[t]
                pd = self.w.pdms[j, k] + self.kw.get("pdms_tamper", {}).get((cid, k, t), 0.0)
                wr.writerow([i, t, True] + [float(x) for x in self.w.sub[j, k]] + [float(pd)])
            wr.writerow([len(toks), "average", True] + [0.0] * 6 + [0.5])
        st = {"label": ctx_label(self.ctx, cid, k), "status": "PASS", "csv_valid_rows": len(toks)}
        open(lp, "w").write(json.dumps(st) + "\n")
        return st


def ctx_label(ctx, cid, k):
    return ctx.label(cid, k)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="p2full_test_")
        self.w = World(self.tmp)
        self._save = {k: getattr(C, k) for k in ("HERE", "WORK", "BANKED_SEAM", "BANKED_CSV", "CKPT", "CKPT_SHA256", "LOCK", "MARK", "FRAME_CONTROL_CENSUS")}
        C.HERE = os.path.join(self.tmp, "pkg")
        os.makedirs(C.HERE)
        C.WORK = os.path.join(self.tmp, "work")
        C.BANKED_SEAM, C.BANKED_CSV = os.path.join(self.tmp, "banked_seam.npz"), os.path.join(self.tmp, "banked.csv")
        C.CKPT = os.path.join(self.tmp, "model_final.pt")
        open(C.CKPT, "wb").write(b"unit-test checkpoint")
        C.CKPT_SHA256 = C.sha256_file(C.CKPT)
        C.LOCK, C.MARK = os.path.join(self.tmp, "gpu.lock"), os.path.join(self.tmp, "chain_active")
        C.FRAME_CONTROL_CENSUS = os.path.join(self.tmp, "fc.json")
        json.dump({"over_bar": {"tkn": {"log": "log_01"}}}, open(C.FRAME_CONTROL_CENSUS, "w"))

    def tearDown(self):
        for k, v in self._save.items():
            setattr(C, k, v)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def ctx(self, **kw):
        a = argparse.Namespace(tag="unit", tokens=self.w.tokfile, chunk_tokens=CHUNK, device="cpu", workers=3, m=K, pose_tol_m=None, stage="all")
        ctx = C.Ctx(a)
        ctx.runner = Fake(ctx, self.w, **kw)
        return ctx


class Chunking(unittest.TestCase):
    def test_whole_logs_cover_and_deterministic(self):
        doc = toks_doc()
        ch = C.build_chunks(doc, CHUNK)
        self.assertEqual(len(ch), 3)
        self.assertEqual(sorted(t for c in ch for t in c["tokens"]), sorted(doc["tokens"]))
        for c in ch:
            self.assertLessEqual(len(c["tokens"]), CHUNK)
            self.assertEqual({doc["token_log"][t] for t in c["tokens"]}, set(c["logs"]))        # whole logs
        logs_seen = [lg for c in ch for lg in c["logs"]]
        self.assertEqual(len(logs_seen), len(set(logs_seen)))                                    # no log in two chunks
        self.assertEqual([c["tokens"] for c in ch], [c["tokens"] for c in C.build_chunks(doc, CHUNK)])

    def test_oversize_log_gets_its_own_chunk_never_split(self):
        doc = toks_doc()
        ch = C.build_chunks(doc, 2)                      # PER_LOG = 3 > 2
        self.assertTrue(all(len(c["logs"]) == 1 for c in ch))
        self.assertEqual(len(ch), N_LOGS)

    def test_the_real_navtest_plan_if_present(self):
        if not os.path.exists(C.TOKENS_FULL):
            self.skipTest("navtest_full_tokens.json not on this machine")
        doc = json.load(open(C.TOKENS_FULL, encoding="utf-8"))
        ch = C.build_chunks(doc, C.MAX_CHUNK_TOKENS)
        self.assertEqual(sum(len(c["tokens"]) for c in ch), C.N_FULL)
        self.assertEqual(len({lg for c in ch for lg in c["logs"]}), C.N_LOGS_FULL)
        self.assertLessEqual(max(len(c["tokens"]) for c in ch), C.MAX_CHUNK_TOKENS)

    def test_written_plan_refuses_a_different_overwrite(self):
        with tempfile.TemporaryDirectory() as t:
            C.write_chunks(toks_doc(), t, CHUNK)
            C.write_chunks(toks_doc(), t, CHUNK)                                    # same plan: fine
            with self.assertRaises(AssertionError):
                C.write_chunks(toks_doc(), t, 2)                                     # different plan: refused


class Pipeline(Base):
    def run_all(self, ctx):
        rc = C.run_pipeline(ctx)
        return rc

    def test_end_to_end_table_equals_the_hidden_world_and_the_proof_passes(self):
        ctx = self.ctx()
        self.assertEqual(self.run_all(ctx), 0, ctx.status)
        self.assertEqual(ctx.counts["score_pass"], 3 * K)
        out = os.path.join(self.tmp, "table.npz")
        self.assertTrue(C.stage_table(ctx, out))
        T = np.load(out)
        order = np.argsort(np.array(self.w.toks))
        self.assertEqual([str(t) for t in T["token"]], [self.w.toks[i] for i in order])
        self.assertTrue(np.array_equal(T["pdms"], self.w.pdms[order]))
        self.assertTrue(np.array_equal(T["sub"], self.w.sub[order]))
        self.assertTrue(np.array_equal(T["proposals"], self.w.props[order]))
        R = D.same_pick_proof(out, os.path.join(self.tmp, "proof.json"), expected_tokens=self.w.toks)
        self.assertTrue(R["PASS"], R)
        self.assertEqual(R["pose"]["n_bit_identical"], len(self.w.toks))
        self.assertEqual(R["pdms_at_pick_vs_banked_csv"]["n_exact"], len(self.w.toks))
        A = D.anatomy(out, os.path.join(self.tmp, "anatomy.json"), boot_b=200)
        self.assertEqual(A["n_tokens"], len(self.w.toks))
        self.assertGreaterEqual(A["mean_pdms_x100"]["best_of_64_oracle"]["point"], A["mean_pdms_x100"]["P2_pick"]["point"])

    def test_REGRESSION_gate_failure_stops_before_that_chunk_is_scored(self):
        ctx = self.ctx(pose_shift_chunk={1: 2e-3})
        rc = self.run_all(ctx)
        self.assertEqual(rc, 2)
        self.assertIn("chunk 1", ctx.abort_reason)
        self.assertEqual([c for c, k in ctx.runner.score_calls if c == 1], [], "a chunk whose dump does not reproduce P2 was scored")
        self.assertEqual([c for c, k in ctx.runner.score_calls if c == 2], [], "a LATER chunk was scored after the abort")
        self.assertFalse(any(os.path.exists(ctx.seam_k(1, k)) for k in range(K)))
        self.assertNotIn(2, ctx.runner.dump_calls, "the dump loop went on after the failed gate")

    def test_REGRESSION_wrong_checkpoint_recorded_in_the_dump_fails_the_gate(self):
        self.assertEqual(self.run_all(self.ctx(bad_ckpt_chunk=0)), 2)

    def test_REGRESSION_fingerprint_mismatch_fails_the_gate(self):
        self.assertEqual(self.run_all(self.ctx(bad_fp_chunk=2)), 2)

    def test_pose_off_by_half_mm_is_inside_the_registered_bar(self):
        ctx = self.ctx(pose_shift_chunk={0: 5e-4})
        self.assertEqual(self.run_all(ctx), 0)

    def test_frame_control_over_the_bar_is_accepted_only_on_the_known_logs(self):
        ctx = self.ctx(fc_max=6.18)           # the known over-bar log is log_01 (in chunk 0); chunk 1 = log_02 + log_03 has none
        self.assertEqual(self.run_all(ctx), 2)
        self.assertEqual(ctx.runner.dump_calls, [0, 1])
        self.assertIn("chunk 1", ctx.abort_reason)
        self.assertTrue(json.load(open(os.path.join(ctx.wd, "state", "gate_fidelity_c00.json")))["seam_report"]["frame_control_over_bar_only_on_the_known_4_logs"])

    def test_REGRESSION_frame_control_beyond_the_known_maximum_fails_even_on_a_known_log(self):
        self.assertEqual(self.run_all(self.ctx(fc_max=9.0)), 2)

    def test_resume_redoes_nothing_that_is_done(self):
        ctx = self.ctx(crash_dump_on=1)
        with self.assertRaises(RuntimeError):
            self.run_all(ctx)
        first = list(ctx.runner.score_calls)
        self.assertTrue(all(c == 0 for c, k in first))
        ctx2 = self.ctx()                                 # a fresh process: new Ctx, state read from disk
        self.assertEqual(self.run_all(ctx2), 0, ctx2.status)
        self.assertNotIn(0, ctx2.runner.dump_calls, "chunk 0 was dumped again")
        self.assertEqual(sorted(ctx2.runner.dump_calls), [1, 2])
        again = set(first) & set(ctx2.runner.score_calls)
        self.assertEqual(again, set(), f"columns scored twice: {sorted(again)[:5]}")
        self.assertEqual(len(set(first) | set(ctx2.runner.score_calls)), 3 * K)
        # a third launch with everything done calls nothing
        ctx3 = self.ctx()
        self.assertEqual(self.run_all(ctx3), 0)
        self.assertEqual((ctx3.runner.dump_calls, ctx3.runner.score_calls), ([], []))

    def test_REGRESSION_a_tampered_dump_is_not_trusted_on_resume(self):
        ctx = self.ctx()
        self.assertEqual(self.run_all(ctx), 0)
        dp = ctx.dump_path(0)
        with open(dp, "ab") as f:
            f.write(b"x")
        ctx2 = self.ctx()
        self.assertFalse(C.chunk_dump_valid(ctx2, ctx2.chunks[0]))
        self.assertTrue(C.chunk_dump_valid(ctx2, ctx2.chunks[1]))

    def test_REGRESSION_circuit_breaker_aborts_on_repeated_harness_failures(self):
        ctx = self.ctx(fail_score=lambda c, k: True)
        rc = self.run_all(ctx)
        self.assertEqual(rc, 3)
        self.assertIn("circuit breaker", ctx.abort_reason or "")
        self.assertLess(len(ctx.runner.score_calls), 3 * K * 3)       # it stopped early, it did not grind through every column x every retry

    def test_a_single_failed_column_fails_the_job_but_is_retried_and_named(self):
        calls = {"n": 0}

        def flaky(c, k):
            if (c, k) == (1, 5) and calls["n"] < 1:
                calls["n"] += 1
                return True
            return False
        ctx = self.ctx(fail_score=flaky)
        self.assertEqual(self.run_all(ctx), 0)                       # the second attempt passes
        self.assertEqual(ctx.runner.score_calls.count((1, 5)), 2)

    def test_a_ram_guard_abort_is_retried_without_costing_an_attempt_or_tripping_the_breaker(self):
        ctx = self.ctx(ram_abort=lambda c, k, n: (c, k) in {(0, 0), (0, 1), (1, 7)} and n < 5)       # 4 aborts each > the 3 real attempts
        self.assertEqual(self.run_all(ctx), 0, ctx.status)
        self.assertEqual(ctx.counts["score_fail"], 0)
        self.assertEqual(ctx.counts["score_ram_guard_retries"], 12)
        self.assertEqual(ctx.runner.score_calls.count((0, 0)), 5)

    def test_harness_starts_are_serialised_through_one_gate(self):
        ctx = self.ctx()
        starts = []
        orig = ctx.runner.score

        def spy(ch, k, lp):
            starts.append(time.time())
            return orig(ch, k, lp)
        ctx.runner.score = spy
        self.assertEqual(self.run_all(ctx), 0)
        self.assertEqual(len(starts), 3 * K)
        self.assertTrue(hasattr(ctx, "start_lk"))

    def test_REGRESSION_table_G2_catches_a_wrong_pdms_at_the_pick(self):
        t = self.w.toks[4]
        k = int(self.w.pick[self.w.idx[t]])
        cid = [c["id"] for c in self.ctx().chunks if t in self.ctx().tokens_of(c)["tokens"]][0]
        ctx = self.ctx(pdms_tamper={(cid, k, t): 1e-3})
        self.assertEqual(self.run_all(ctx), 0)
        self.assertFalse(C.stage_table(ctx, os.path.join(self.tmp, "table.npz")))
        self.assertGreater(json.load(open(os.path.join(ctx.wd, "gate_table.json")))["G2_n_score_diff_gt_1e_9"], 0)


class Proof(Base):
    def table(self, **kw):
        ctx = self.ctx(**kw)
        self.assertEqual(C.run_pipeline(ctx), 0)
        out = os.path.join(self.tmp, "table.npz")
        C.stage_table(ctx, out)
        return out

    def mutate(self, path, fn):
        T = dict(np.load(path))
        fn(T)
        np.savez(path, **T)

    def test_REGRESSION_each_equality_alone_can_fail_the_proof(self):
        base = self.table()
        ok = D.same_pick_proof(base, os.path.join(self.tmp, "p.json"), expected_tokens=self.w.toks)
        self.assertTrue(ok["PASS"])
        arms = {
            "pose_2mm": lambda T: T["proposals"].__setitem__((0, int(T["pick"][0]), 0, 0), T["proposals"][0, int(T["pick"][0]), 0, 0] + 2e-3),
            "pdms_1e-6_at_pick": lambda T: T["pdms"].__setitem__((1, int(T["pick"][1])), T["pdms"][1, int(T["pick"][1])] + 1e-6),
            "subscore_at_pick": lambda T: T["sub"].__setitem__((2, int(T["pick"][2]), 3), T["sub"][2, int(T["pick"][2]), 3] + 1e-6),
            "different_pick": lambda T: T["pick"].__setitem__(3, (int(T["pick"][3]) + 1) % K),
            "fingerprint": lambda T: T["fingerprint"].__setitem__(4, "other"),
            "invalid_cell": lambda T: T["valid"].__setitem__((5, 5), False),
        }
        for name, fn in arms.items():
            p = os.path.join(self.tmp, f"t_{name}.npz")
            shutil.copy(base, p)
            self.mutate(p, fn)
            R = D.same_pick_proof(p, os.path.join(self.tmp, f"p_{name}.json"), expected_tokens=self.w.toks)
            self.assertFalse(R["PASS"], name)

    def test_REGRESSION_a_missing_token_fails_the_proof(self):
        base = self.table()
        R = D.same_pick_proof(base, os.path.join(self.tmp, "p.json"), expected_tokens=self.w.toks + ["extra_token"])
        self.assertFalse(R["PASS"])


class LockProtocol(Base):
    def setUp(self):
        super().setUp()
        self._wi, self._gpu, self._alive, self._blk = C.wait_idle_once, C.gpu_used_mib, C.pid_alive, C.blockers
        C.wait_idle_once = lambda ctx, timeout_s=0: True
        C.blockers = lambda: []
        C.gpu_used_mib = lambda: 100
        self.c = self.ctx()

    def tearDown(self):
        C.wait_idle_once, C.gpu_used_mib, C.pid_alive, C.blockers = self._wi, self._gpu, self._alive, self._blk
        super().tearDown()

    def attempt(self, secs=2.0):
        res = {}

        def go():
            try:
                res["body"] = C.take_lock(self.c, "unit")
            except RuntimeError as e:
                res["err"] = str(e)
        th = threading.Thread(target=go, daemon=True)
        th.start()
        th.join(secs)
        self.c.abort.set()
        th.join(40)                                  # take_lock sleeps up to 20 s between polls
        assert not th.is_alive(), "the take_lock thread did not stop after abort"
        self.c.abort = threading.Event()             # a FRESH event: a straggler must never see the abort cleared
        return res

    def test_takes_a_free_lock_writes_pid_owner_and_releases_only_its_own(self):
        res = self.attempt(3)
        self.assertIn("body", res)
        held = json.load(open(C.LOCK))
        self.assertEqual((held["pid"], held["owner"]), (os.getpid(), C.OWNER))
        C.release(json.dumps({"pid": os.getpid() + 1}))                 # someone else's body does not release it
        self.assertTrue(os.path.exists(C.LOCK))
        C.release(res["body"])
        self.assertFalse(os.path.exists(C.LOCK))

    def test_REGRESSION_a_live_holder_is_never_preempted(self):
        json.dump({"pid": 4242, "owner": "probe", "t": time.time()}, open(C.LOCK, "w"))
        C.pid_alive = lambda pid: True
        res = self.attempt(2)
        self.assertNotIn("body", res)
        self.assertEqual(json.load(open(C.LOCK))["owner"], "probe")

    def test_REGRESSION_another_jobs_lock_is_NEVER_removed_however_old_and_whatever_its_pid(self):
        # gpu_lock.py writes the JOB's MSYS pid, which tasklist cannot see: "pid not alive" + "older than 10 min" must NOT license removing it
        json.dump({"job": "refcv8-g0-step5000", "pid": 2699, "win_pid_of_locker": 4100}, open(C.LOCK, "w"))
        C.pid_alive = lambda pid: False
        old = time.time() - 7200
        os.utime(C.LOCK, (old, old))
        res = self.attempt(3)
        self.assertNotIn("body", res)
        self.assertTrue(os.path.exists(C.LOCK), "an old lock of another job was removed")
        self.assertEqual(json.load(open(C.LOCK))["job"], "refcv8-g0-step5000")
        self.assertIn("looks_stale", self.c.status.get("state", ""), "a stale-looking lock must be reported, not silently waited on")

    def test_REGRESSION_a_running_probe_process_blocks_the_lock_even_when_the_lock_file_is_free(self):
        C.blockers = lambda: ["30716 python scene_tap_probe.py run"]
        res = self.attempt(2)
        self.assertNotIn("body", res)
        self.assertFalse(os.path.exists(C.LOCK))
        self.assertIn("scene_tap_probe", self.c.status.get("state", ""))
        C.blockers = lambda: []
        self.assertIn("body", self.attempt(3))

    def test_the_real_blockers_probe_runs_and_does_not_flag_this_process(self):
        C.blockers = self._blk
        self.assertIsInstance(C.blockers(), list)

    def test_REGRESSION_chain_active_marker_blocks_the_lock(self):
        open(C.MARK, "w").write("1")
        res = self.attempt(2)
        self.assertNotIn("body", res)
        self.assertFalse(os.path.exists(C.LOCK))

    def test_REGRESSION_memory_over_the_bar_after_taking_gives_the_lock_back(self):
        C.gpu_used_mib = lambda: 5000
        res = self.attempt(6)
        self.assertNotIn("body", res)
        self.assertFalse(os.path.exists(C.LOCK), "the lock was left behind after the memory re-check failed")


class LockHold(Base):
    """device = cuda path with the lock/idle/waiter machinery mocked: the lock is KEPT across chunks unless a neighbour is queued, and never left behind."""

    def setUp(self):
        super().setUp()
        self._save2 = {k: getattr(C, k) for k in ("wait_idle_once", "gpu_used_mib", "blockers", "waiters", "WAIT_IDLE", "take_lock")}
        with open(os.path.join(self.tmp, "wait_idle.py"), "w"):
            pass
        C.WAIT_IDLE = os.path.join(self.tmp, "wait_idle.py")
        C.wait_idle_once = lambda ctx, timeout_s=0: True
        C.gpu_used_mib = lambda: 100
        C.blockers = lambda: []
        C.waiters = lambda: []
        self.takes = []
        real = C.take_lock

        def counting(ctx, job):
            self.takes.append(job)
            return real(ctx, job)
        C.take_lock = counting

    def tearDown(self):
        for k, v in self._save2.items():
            setattr(C, k, v)
        super().tearDown()

    def cuda_ctx(self, **kw):
        a = argparse.Namespace(tag="unit", tokens=self.w.tokfile, chunk_tokens=CHUNK, device="cuda", workers=3, m=K, pose_tol_m=None, stage="all")
        ctx = C.Ctx(a)
        ctx.runner = Fake(ctx, self.w, **kw)
        return ctx

    def test_no_neighbour_queued_keeps_the_lock_across_all_chunks_and_releases_at_the_end(self):
        ctx = self.cuda_ctx()
        self.assertEqual(C.run_pipeline(ctx), 0, ctx.status)
        self.assertEqual(len(self.takes), 1, self.takes)
        self.assertFalse(os.path.exists(C.LOCK), "the lock was left behind")

    def test_a_queued_neighbour_gets_the_lock_at_every_chunk_boundary(self):
        C.waiters = lambda: ["4100 gpu_lock.py acquire --job refcv8"]
        ctx = self.cuda_ctx()
        self.assertEqual(C.run_pipeline(ctx), 0, ctx.status)
        self.assertEqual(len(self.takes), 3, self.takes)
        self.assertFalse(os.path.exists(C.LOCK))

    def test_REGRESSION_the_lock_is_released_when_the_gate_fails_or_the_dump_crashes(self):
        self.assertEqual(C.run_pipeline(self.cuda_ctx(pose_shift_chunk={1: 2e-3})), 2)
        self.assertFalse(os.path.exists(C.LOCK), "a failed gate left the GPU lock behind")
        with self.assertRaises(RuntimeError):
            C.run_pipeline(self.cuda_ctx(crash_dump_on=1))      # chunk 0 is already done (resume); chunk 1 crashes
        self.assertFalse(os.path.exists(C.LOCK), "a crashed dump left the GPU lock behind")

    def test_a_lock_that_is_no_longer_ours_is_retaken_not_trusted(self):
        ctx = self.cuda_ctx()
        C.hold_gpu(ctx, "a")
        os.remove(C.LOCK)                                           # someone cleared it
        C.hold_gpu(ctx, "b")
        self.assertEqual(self.takes, ["a", "b"])
        C.drop_gpu(ctx)
        self.assertFalse(os.path.exists(C.LOCK))


class Guards(Base):
    def test_disk_guard_pauses_then_resumes(self):
        c = self.ctx()
        old_free, old_ram = C.free_gb, C.free_mb
        state = {"gb": 1.0}
        C.free_gb = lambda p: state["gb"]
        C.free_mb = lambda: 99999
        done = threading.Event()
        th = threading.Thread(target=lambda: (C.wait_resources(c, "unit", [self.tmp]), done.set()), daemon=True)
        try:
            th.start()
            self.assertFalse(done.wait(1.5), "the guard let a write through on a nearly full drive")
            state["gb"] = 50.0
            self.assertTrue(done.wait(35), "the guard did not release when space came back")
            self.assertIn("PAUSED", c.status.get("guard", ""))
        finally:
            c.abort.set()
            C.free_gb, C.free_mb = old_free, old_ram

    def test_ram_probe_returns_a_plausible_number(self):
        self.assertGreater(C.free_mb(), 100)


class Preflight(Base):
    def test_preflight_passes_and_REGRESSION_wrong_checkpoint_fails(self):
        ctx = self.ctx()
        self.assertTrue(C.preflight(ctx))
        C.CKPT_SHA256 = "0" * 64
        self.assertFalse(C.preflight(ctx))

    def test_REGRESSION_checkpoint_changed_during_the_job_stops_the_next_dump(self):
        ctx = self.ctx()
        self.assertTrue(C.preflight(ctx))
        os.utime(C.CKPT, (1, 1))
        self.assertEqual(C.ensure_chunk(ctx, ctx.chunks[0]), "dump_failed")
        self.assertEqual(ctx.runner.dump_calls, [])


class Aggregate(unittest.TestCase):
    def test_navsim_v1_aggregate_known_answer(self):
        lg = np.full((2, 3, 6), -20.0)
        lg[0, 1, :5] = 20.0
        lg[1, 2, :5] = 20.0
        lg[1, 2, 1] = -20.0
        lg[1, 0, :5] = 20.0
        lg[1, 0, 2] = 0.0
        self.assertEqual(D.pick_from_logits(lg).tolist(), [1, 0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
