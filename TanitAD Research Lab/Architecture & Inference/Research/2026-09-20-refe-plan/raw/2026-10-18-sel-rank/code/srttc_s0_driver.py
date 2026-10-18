#!/usr/bin/env python3
"""SR-TTC STAGE S0 -- the driver.  Registered: PREREG_SELRANK_ARM.md (+ amendment1, amendment2 = the nested log subsample, option (b), f0 = 10 %).

    for every batch of the 117 registered fit logs (8,922 tokens / 15,808 sets):
        STAGE   nuPlan DBs + the needed camera frames from the (idle) pod, sha256 equal on both sides             srttc_s0_stage.py  (podcopy.py unchanged)
        DUMP    P2 (model_final.pt, sha256 ea31db22...) on every fit set's OWN 64 proposals: score-head INPUT [64,256], six logits, frozen top-8,
                the PDM labels P2 trained on                                                                      srttc_s0_dump.py  (GPU behind the shared dev-box lock)
        GATE    integrity only: counts, shapes, finiteness, label range, top-8 validity, identity controls (see gate_unit)
        BANK    work/bank/<batch>.sets.npz (+ .fan.npz, .report.json); run/state/unit_<batch>.json carries the sha256s
        CLEAN   delete exactly the staged copies of the batch (verified pod originals)
    -> S0_DONE (+ S0_SUMMARY.json) / S0_FAILED.   Nothing is fitted, cross-validated or scored here, and no S0 value is read beyond the integrity gates.

Gates before the first batch:  G1 master plan == the registered selection (list sha256, counts) and the set stores reproduce 15,808 sets;  G2 checkpoint sha256;
G2b DG-DUMP (this dump script, fan path, on the GPU, must reproduce the landed full-navtest table rows bit for bit; the negative control must FAIL);
G-ID (per batch, in gate_unit) the identity controls of the set-scoring path: POSITIVE (P2's own fan through score_trajectories reproduces forward's logits, <= 1e-4) and NEGATIVE (a stored set's
logits differ from the fan's for >= 90 % of sets).  NOT a control: "a stored ckpt-10075 set equals P2's fan" -- it does not (MEASURED in the smoke: P2 was trained with an un-detached trunk).
Resumable: re-run the SAME command; a batch whose state file carries gate_PASS and matching sha256s is skipped (and never re-staged).
Exit/marker codes (first line of S0_FAILED): rc1 preflight / dump, rc2 a gate (REPORT, do not re-run), rc3 staging, rc4 final assembly, rc5 coverage / replacement rule, rc6 C: < 5 GiB.
"""
from __future__ import annotations

import argparse
import atexit
import ctypes
import glob
import hashlib
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = "C:/Users/Admin/srttc_s0"
TREE = os.environ.get("P2HO_TREE", "C:/Users/Admin/p2heldout/tree/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan")
sys.path.insert(0, os.path.join(TREE, "eval"))
sys.path.insert(0, os.path.join(HERE, "landed"))
sys.path.insert(0, HERE)
import eval_checkpoint as EC  # noqa: E402  (the tree's: one source for the venvs + env)
import c3_run_p2full as C     # noqa: E402  (landed, unchanged: GPU-lock protocol, RAM/disk helpers, atomic_json, sha256)
import srttc_s0_stage as ST   # noqa: E402

C.OWNER = "srttc_s0"
CKPT = "D:/Projects/TanitAD/data/refe_runs_eval/p2/model_final.pt"
CKPT_SHA256 = "ea31db2296210b6a35dbf32ad95f1813edfa259ab330002b3ca7564aac727e51"
SELECTED_SHA256 = "9d4d606ad26f871bda37eed5bb3b29b1f91f74526d27433da0105f842265db25"
MIN_FREE_RAM_MB = 8000            # a dump start waits for this much free RAM; --min-free-ram-mb overrides (set from the MEASURED peak RSS of the dump, see HANDOFF)
MIN_FREE_DISK_GB = 5.0            # the brief: stop cleanly below 5 GB free on C:
BATCH_MAX_TOKENS, BATCH_MAX_LOGS = 500, 12
CTRL_FANPATH_TOL = 1e-4           # positive identity control: P2's own fan through score_trajectories vs the logits of the same forward (fp32, same modules): expected exactly 0
now, sha256_file, jdump, jload = C.now, C.sha256_file, C.atomic_json, C.jload

# ---- the neighbours' idle waiter, with ONE change.  `wait_idle.py` (shared, not edited) treats any python command line containing "train.py" as a GPU job; a pytest run whose FILE NAME
# ends in `_train.py` (tests/test_smoke_train.py, ...) matched it (MEASURED 2026-10-18 11:18: 8 pytest shards of another session blocked the held-out-style lock for > 15 min with the GPU at
# 1.7 GB / 9 %).  Same criteria otherwise: used memory < 3500 MiB, no GPU-job python process, 4 consecutive polls 15 s apart.  The lock file itself (O_EXCL), the post-lock memory check and
# every other lock rule are the landed ones.
IDLE_PAT = ("ladder_eval_diag", "scorer_finetune", "rl_fan_dump", "onpolicy_dump", "train.py", "eval_checkpoint", "lgdiag")


def _idle_others():
    ps = "Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python' } | ForEach-Object { $_.CommandLine }"
    try:
        out = subprocess.run(["powershell.exe", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=120).stdout
    except (OSError, subprocess.SubprocessError):
        return ["(process list unreadable)"]
    return [ln for ln in out.splitlines() if any(p in ln for p in IDLE_PAT) and "wait_idle.py" not in ln and "pytest" not in ln and "srttc_s0" not in ln]


def my_wait_idle_once(ctx, timeout_s=3600.0):
    t0, idle = time.time(), 0
    while True:
        if ctx.abort.is_set() or time.time() - t0 > timeout_s:
            return False
        u, o = C.gpu_used_mib(), _idle_others()
        idle = idle + 1 if (0 <= u < C.GPU_MAX_USED_MIB and not o) else 0
        if idle >= 4:
            return True
        time.sleep(15)


C.wait_idle_once = my_wait_idle_once


class Stop(Exception):
    pass


class Ctx:
    def __init__(self, a):
        self.a = a
        self.tag = a.tag
        self.main = a.tag == "main"
        self.wd = f"{SRC}/run" if self.main else f"{SRC}/run/_{a.tag}"
        self.work = f"{SRC}/work" if self.main else f"{SRC}/work/_{a.tag}"
        for d in ("logs", "state", "plan", "batches"):
            os.makedirs(os.path.join(self.wd, d), exist_ok=True)
        for d in ("bank", "dumps", "stage", "tmp"):
            os.makedirs(os.path.join(self.work, d), exist_ok=True)
        self.status_p = os.path.join(self.wd, "STATUS.json")
        self.status = jload(self.status_p) if os.path.exists(self.status_p) else {}
        self._lk = threading.RLock()
        self.abort = threading.Event()
        self.abort_reason = None
        self.lock_body = None
        self.counts = {"batches_done": 0, "sets_banked": 0, "tokens_banked": 0}

    def set(self, **kw):
        with self._lk:
            self.status.update(kw, updated=now())
            jdump(self.status, self.status_p)
        print(f"[{now()}] STATUS {json.dumps(kw, default=str)[:300]}", flush=True)


def disk_guard(ctx, what):
    free = C.free_gb(ctx.work)
    if free < MIN_FREE_DISK_GB:
        raise Stop(f"lowdisk: {ctx.work} has {free:.2f} GiB free < {MIN_FREE_DISK_GB} before {what}")
    t0, said = time.time(), False
    while C.free_mb() < MIN_FREE_RAM_MB and not ctx.abort.is_set():
        if not said or time.time() - t0 > 600:
            ctx.set(guard=f"PAUSED before {what}: free RAM {C.free_mb()} MiB < {MIN_FREE_RAM_MB}")
            said, t0 = True, time.time()
        time.sleep(30)


def run_logged(cmd, env, log, cwd=HERE):
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n--- {now()} {' '.join(str(c) for c in cmd[:3])} ...\n")
        f.flush()
        return subprocess.call(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)


# ------------------------------------------------------------------------------------------ preflight
def stage_preflight(ctx):
    ctx.set(stage="preflight")
    mp = json.load(open(f"{SRC}/run/plan/master_plan.json", encoding="utf-8"))
    if not (mp["selected_logs_sha256"] == SELECTED_SHA256 and mp["n_logs"] == 117 and mp["n_tokens"] == 8922 and mp["n_sets"] == 15808):
        raise Stop("rc2 preflight: the master plan is not the registered 10 % selection (selected-list sha256 / counts)")
    prov = jload(f"{SRC}/run/plan/podsets_provenance.json")
    chk = prov.get("checks", {})
    if not (prov.get("per_log_stores", {}).get("sets") == 15808 and chk.get("served_missing_from_bank") == 0 and chk.get("ckpt_step_differs_from_served_keys") == 0):
        raise Stop("rc2 preflight: the fit-set stores do not reproduce the 15,808 served sets (podsets_provenance.json)")
    for r in mp["logs"]:
        if not os.path.exists(f"{SRC}/work/sets/{r['log']}.npz"):
            raise Stop(f"rc2 preflight: the fit-set store of {r['log']} is missing")
    pf = {"ckpt_sha256": sha256_file(CKPT), "expected": CKPT_SHA256}
    pf["PASS"] = pf["ckpt_sha256"] == CKPT_SHA256
    st = os.stat(CKPT)
    ctx.set(preflight={"selected_logs_sha256": mp["selected_logs_sha256"], "n_logs": mp["n_logs"], "n_tokens": mp["n_tokens"], "n_sets": mp["n_sets"], "ckpt_sha256": pf["ckpt_sha256"][:16]},
            ckpt_stat=[st.st_size, st.st_mtime_ns])
    if not pf["PASS"]:
        raise Stop("rc1 preflight: checkpoint sha256 differs from P2's model_final.pt")
    sr = ST.STAGE_ROOT
    os.makedirs(sr, exist_ok=True)
    if C.free_gb(sr) < 22.0 or C.free_gb(ctx.work) < 5.5:
        raise Stop(f"rc6 preflight: staging drive {sr} has {C.free_gb(sr):.1f} GiB free (need >= 22) / C: {C.free_gb(ctx.work):.1f} GiB (need >= 5.5)")
    if ctx.a.device == "cuda" and not os.path.exists(C.WAIT_IDLE):
        raise Stop(f"rc1 preflight: {C.WAIT_IDLE} (the idle waiter) is missing")
    jdump({"master_plan_sha256": sha256_file(f"{SRC}/run/plan/master_plan.json"), **pf}, os.path.join(ctx.wd, "preflight.json"))
    return mp


def stage_dgdump(ctx):
    """G2b: this dump script's FAN path (no --sets-dir) reproduces the landed full-navtest P2 table rows on the GPU; the negative control must read FAIL.  Cached on a PASS file."""
    fp = os.path.join(ctx.wd, "dgdump.json")
    if os.path.exists(fp) and jload(fp).get("PASS"):
        return jload(fp)
    ctx.set(stage="dgdump")
    disk_guard(ctx, "DG-DUMP")
    locked = ctx.a.device == "cuda"
    try:
        if locked:
            C.hold_gpu(ctx, "srttc-s0-dgdump")
        rc = run_logged([EC.DRIVERL_PY, os.path.join(HERE, "srttc_s0_dgdump.py"), "--out", fp, "--workdir", os.path.join(ctx.work, "dgdump"), "--k", "4", "--device", ctx.a.device],
                        EC.env_driverl(), os.path.join(ctx.wd, "logs", "dgdump.log"))
    finally:
        if locked and (ctx.abort.is_set() or C.waiters()):
            C.drop_gpu(ctx)
    R = jload(fp) if os.path.exists(fp) else {"PASS": False}
    ctx.set(dgdump={k: R.get(k) for k in ("PASS", "n", "proposals_max_abs_m", "logits_max_abs", "pick_equal", "negative_control_reads_FAIL")})
    if rc or not R.get("PASS"):
        C.drop_gpu(ctx)
        raise Stop("rc2 DG-DUMP: this dump script does not reproduce the landed full-navtest table rows (dgdump.json) -- report it, do not re-run")
    return R


# ------------------------------------------------------------------------------------------ batches / units
def unit_id(prefix, tokens):
    """CONTENT-addressed: the same token set is the same unit, whatever phase or restart formed it."""
    return prefix + hashlib.sha256(",".join(sorted(tokens)).encode()).hexdigest()[:10]


def make_batches(mp, tokens, sets_per_tok, prefix):
    """consecutive logs in the registered KEY order, over the GIVEN tokens only ({token: log}); a batch holds <= BATCH_MAX_TOKENS tokens and <= BATCH_MAX_LOGS logs (a bigger log is a batch of its own)."""
    by_log: dict = {}
    for t, lg in tokens.items():
        by_log.setdefault(lg, []).append(t)
    groups, cur, cur_t = [], [], 0
    for r in mp["logs"]:
        toks = by_log.get(r["log"])
        if not toks:
            continue
        if cur and (cur_t + len(toks) > BATCH_MAX_TOKENS or len(cur) >= BATCH_MAX_LOGS):
            groups.append(cur)
            cur, cur_t = [], 0
        cur.append((r, toks))
        cur_t += len(toks)
    if cur:
        groups.append(cur)
    out = []
    for grp in groups:
        toks = {t: r["log"] for r, ts in grp for t in ts}
        out.append({"id": unit_id(prefix, toks), "logs": [r["log"] for r, _ in grp], "ranks": [r["rank"] for r, _ in grp], "n_tokens": len(toks),
                    "n_sets": sum(sets_per_tok.get(t, 0) for t in toks), "tokens": toks})
    return out


def bank_p(ctx, b, kind):
    return f"{ctx.work}/bank/{b['id']}.{kind}"


def state_p(ctx, b):
    return os.path.join(ctx.wd, "state", f"unit_{b['id']}.json")


def valid_units(ctx):
    """{unit id: state} for every unit whose state says gate_PASS and whose bank files hash to the recorded sha256.  A unit that fails this is IGNORED (its tokens are dumped again)."""
    out = {}
    for sp in sorted(glob.glob(os.path.join(ctx.wd, "state", "unit_*.json"))):
        try:
            st = jload(sp)
        except (OSError, ValueError):
            continue
        uid = st.get("unit")
        if not (uid and st.get("gate_PASS")):
            continue
        if st.get("empty"):
            out[uid] = st
            continue
        try:
            if st["sets_sha256"] == sha256_file(f"{ctx.work}/bank/{uid}.sets.npz") and st["fan_sha256"] == sha256_file(f"{ctx.work}/bank/{uid}.fan.npz"):
                out[uid] = st
        except (OSError, KeyError):
            pass
    return out


def settled_tokens(units):
    """tokens that need no further work: banked, or missed by the scenario builder (margin guard, disclosed), or blocked FINALLY (staged from the pod and still absent there)."""
    done = set()
    for st in units.values():
        for k in ("tokens_banked", "tokens_missed", "tokens_blocked_final"):
            done |= set(st.get(k) or [])
    return done


def gate_unit(ctx, b, plan, rep, sets_npz, fan_npz):
    """G-DUMP (per batch, BEFORE banking).  INTEGRITY ONLY: counts, shapes, finiteness, label range, top-8 validity, identity controls.  No feature / label / logit value is read for any other purpose."""
    Z = np.load(sets_npz)
    F = np.load(fan_npz)
    tok_fan = [str(t) for t in F["token"]]
    g = {"unit": b["id"], "n_tokens_planned": len(plan["tokens"]), "n_tokens_rows": len(tok_fan), "n_sets_rows": int(Z["feat"].shape[0]), "feat_shape": list(Z["feat"].shape),
         "logits_shape": list(Z["logits"].shape), "labels_shape": list(Z["labels6"].shape), "rule": str(Z["rule"]), "ckpt": str(Z["ckpt"])}
    g["tokens_unique"] = bool(len(set(tok_fan)) == len(tok_fan))
    g["misses"] = rep.get("misses")
    g["margin_only"] = bool(rep.get("margin_only"))
    g["miss_tokens"] = list(rep.get("miss_tokens") or [])
    g["dumped_tokens"] = tok_fan
    miss_t = set(rep.get("miss_tokens") or [])
    g["all_planned_tokens_accounted"] = bool(set(plan["tokens"]) == set(tok_fan) | miss_t and not (set(tok_fan) & miss_t))
    g["sets_expected_by_dump"] = rep.get("sets_asked")
    g["sets_match_dump_expectation"] = bool(rep.get("sets_asked") == g["n_sets_rows"] == rep.get("sets_rows"))
    f32 = Z["feat"].astype(np.float32)
    g["finite"] = bool(np.isfinite(f32).all() and np.isfinite(Z["logits"]).all() and np.isfinite(Z["labels6"]).all() and np.isfinite(Z["ctrl_traj_maxabs"]).all() and np.isfinite(Z["ctrl_logit_maxabs"]).all())
    g["labels_in_unit_interval"] = bool(Z["labels6"].min() >= 0.0 and Z["labels6"].max() <= 1.0)
    g["feat_dim_ok"] = bool(Z["feat"].shape[1:] == (64, 256) and Z["logits"].shape[1:] == (64, 6) and Z["labels6"].shape[1:] == (64, 6) and Z["top8"].shape[1:] == (8,))
    t8 = Z["top8"].astype(np.int64)
    g["top8_valid"] = bool(((t8 >= 0) & (t8 < 64)).all() and all(len(set(r.tolist())) == 8 for r in t8))
    # the sets of this unit are the plan's sets of the dumped tokens (each (log, token, step, rank) once)
    keys = list(zip([str(x) for x in Z["log"]], [str(x) for x in Z["token"]], Z["step"].tolist(), Z["rank"].tolist()))
    g["set_keys_unique"] = bool(len(set(keys)) == len(keys))
    mp_sets = ctx.master_sets
    g["set_keys_in_plan"] = bool(all(k in mp_sets for k in keys))
    g["ckpt_step_equals_plan"] = bool(all(mp_sets[k] == int(c) for k, c in zip(keys, Z["ckpt_step"].tolist())))
    # identity controls.  POSITIVE: P2's own fan through score_trajectories reproduces forward's logits (the set-scoring path IS the model's scoring path).  NEGATIVE: a stored set is a different fan,
    # so its logits must differ from the fan's (the control can read non-zero).  Diagnostic: stored-trajectory distance to the fan, by checkpoint.
    fp_, cl, ct = Z["ctrl_fanpath_logit_maxabs"], Z["ctrl_logit_maxabs"], Z["ctrl_traj_maxabs"]
    g["ctrl"] = {"fanpath_logit_max": float(fp_.max()) if fp_.size else 0.0, "fanpath_tol": CTRL_FANPATH_TOL, "n_sets": int(fp_.size),
                 "setpath_logit_differs_from_fan_gt_1e-3": int((cl > 1e-3).sum()), "traj_dist_median_m": float(np.median(ct)) if ct.size else None,
                 "traj_dist_min_m": float(ct.min()) if ct.size else None}
    g["ctrl_positive_fanpath"] = bool(rep.get("ctrl_fanpath_max") is not None and rep["ctrl_fanpath_max"] <= CTRL_FANPATH_TOL and (not fp_.size or float(fp_.max()) <= CTRL_FANPATH_TOL))
    g["ctrl_negative_sets_differ"] = bool((not cl.size) or (cl > 1e-3).mean() >= 0.9)
    g["PASS"] = bool(g["tokens_unique"] and g["all_planned_tokens_accounted"] and g["sets_match_dump_expectation"] and g["finite"] and g["labels_in_unit_interval"] and g["feat_dim_ok"]
                     and g["top8_valid"] and g["set_keys_unique"] and g["set_keys_in_plan"] and g["ckpt_step_equals_plan"] and g["ctrl_positive_fanpath"] and g["ctrl_negative_sets_differ"]
                     and (not g["misses"] or g["margin_only"]) and "navsim_v1" in g["rule"] and g["ckpt"] == CKPT)
    jdump(g, os.path.join(ctx.wd, "state", f"gate_{b['id']}.json"))
    return g


def dump_batch(ctx, b, info, more_fn):
    """GPU dump of one staged batch -> (gate dict).  Raises Stop on a gate failure."""
    plan = jload(info["plan"])
    if not plan["tokens"]:
        return {"PASS": True, "empty": True, "n_tokens_planned": 0}
    locked = ctx.a.device == "cuda"
    part_sets, part_fan = bank_p(ctx, b, "sets.part.npz"), bank_p(ctx, b, "fan.part.npz")
    for attempt in range(1, 4):
        if ctx.abort.is_set():
            raise Stop("aborted")
        disk_guard(ctx, f"dump of {b['id']}")
        try:
            if locked:
                C.hold_gpu(ctx, f"srttc-s0-{b['id']}")
            for x in (part_sets, part_fan):
                if os.path.exists(x):
                    os.remove(x)
            st = os.stat(CKPT)
            if [st.st_size, st.st_mtime_ns] != ctx.status.get("ckpt_stat"):
                C.drop_gpu(ctx)
                raise Stop("rc1 the checkpoint file changed since the preflight")
            ctx.set(stage="dump", state=f"{b['id']} ({len(b['logs'])} logs, {len(plan['tokens'])} tokens, attempt {attempt}): dumping")
            t0 = time.time()
            lp = os.path.join(ctx.wd, "logs", f"dump_{b['id']}.log")
            rc = run_logged([EC.DRIVERL_PY, os.path.join(HERE, "srttc_s0_dump.py"), "--ckpt", CKPT, "--plan", info["plan"], "--out", part_fan, "--device", ctx.a.device, "--lean",
                             "--sets-dir", f"{SRC}/work/sets", "--sets-out", part_sets, "--arm", f"SRTTC_S0_{b['id']}"], EC.env_driverl(), lp)
            ctx.set(stage="dump", state=f"{b['id']}: dump rc={rc}", last_dump_seconds=round(time.time() - t0))
        except BaseException:
            C.drop_gpu(ctx)
            raise
        if locked and (not os.path.exists(part_sets) or not more_fn() or C.waiters()):       # keep the lock only while the next unit is already waiting
            C.drop_gpu(ctx)
        rep_p = os.path.splitext(part_fan)[0] + ".report.json"
        if os.path.exists(part_sets) and os.path.exists(part_fan) and os.path.exists(rep_p):
            rep = jload(rep_p)
            g = gate_unit(ctx, b, plan, rep, part_sets, part_fan)
            ctx.set(stage="gate", **{f"gate_{b['id']}": {k: g.get(k) for k in ("PASS", "n_tokens_rows", "n_sets_rows", "misses", "ctrl")}})
            if not g["PASS"]:
                C.drop_gpu(ctx)
                raise Stop(f"rc2 gate of {b['id']} failed (state/gate_{b['id']}.json): REPORT, do not re-run")
            os.replace(part_sets, bank_p(ctx, b, "sets.npz"))
            os.replace(part_fan, bank_p(ctx, b, "fan.npz"))
            shutil.copyfile(rep_p, bank_p(ctx, b, "report.json"))
            os.remove(rep_p)
            return g
        time.sleep(30 if locked else 1)
    raise Stop(f"rc1 the dump of {b['id']} failed 3 times (logs/dump_{b['id']}.log)")


class Heartbeat(threading.Thread):
    def __init__(self, ctx, every=60):
        super().__init__(daemon=True)
        self.ctx, self.every, self.stop_ev = ctx, every, threading.Event()

    def run(self):
        while not self.stop_ev.wait(self.every):
            with self.ctx._lk:
                self.ctx.status["heartbeat"] = now()
                jdump(self.ctx.status, self.ctx.status_p)


def make_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main", help="main = the job; any other tag is a smoke (own dirs, restricted to --smoke-logs)")
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--smoke-logs", default="", help="comma list of logs (smoke only)")
    ap.add_argument("--min-free-ram-mb", type=int, default=MIN_FREE_RAM_MB)
    ap.add_argument("--no-stage", action="store_true", help="smoke: never touch the pod (only locally available tokens)")
    return ap


def bank_unit(ctx, b, info, g, phase):
    """write the unit's state (sha256s, tokens banked / missed) AFTER the bank files are in place."""
    si = {k: info.get(k) for k in ("n_tokens_in", "n_tokens_ok", "blocked_reasons", "missing_dbs_on_pod", "missing_frames_on_pod", "bytes_staged")}
    blocked_final = []
    if phase == "S":                                   # a token that is still blocked AFTER staging from the pod is final (absent there too)
        bt = os.path.join(ctx.wd, "batches", b["id"], "blocked_tokens.json")
        if os.path.exists(bt):
            blocked_final = sorted(jload(bt))
    if g.get("empty"):
        jdump({"unit": b["id"], "phase": phase, "logs": b["logs"], "gate_PASS": True, "empty": True, "tokens_banked": [], "tokens_missed": [], "tokens_blocked_final": blocked_final,
               "sets_sha256": None, "fan_sha256": None, "when": now(), "stage_info": si}, state_p(ctx, b))
        return
    jdump({"unit": b["id"], "phase": phase, "logs": b["logs"], "ranks": b["ranks"], "n_tokens_planned": g["n_tokens_planned"], "gate_PASS": True,
           "tokens_banked": sorted(g["dumped_tokens"]), "tokens_missed": sorted(g["miss_tokens"]), "tokens_blocked_final": blocked_final,
           "sets_sha256": sha256_file(bank_p(ctx, b, "sets.npz")), "fan_sha256": sha256_file(bank_p(ctx, b, "fan.npz")), "when": now(), "gate": {k: v for k, v in g.items() if k != "dumped_tokens"},
           "stage_info": si}, state_p(ctx, b))


def run_phase_local(ctx, mp, stager, sets_per_tok):
    """PHASE L: the tokens whose nuPlan DB and four camera frames are ALREADY on this box (no pod access, no copy).  Runs first, so a busy pod (P6 has priority) never idles the GPU."""
    units = valid_units(ctx)
    remaining = {t: l for t, l in mp["tokens"].items() if t not in settled_tokens(units)}
    if not remaining:
        return
    ctx.set(stage="local-audit", state=f"phase L: auditing {len(remaining)} unsettled tokens for DB + frames already on this box")
    pd = os.path.join(ctx.wd, "batches", "_local_audit")
    os.makedirs(pd, exist_ok=True)
    tj = os.path.join(pd, "tokens_in.json")
    json.dump({"tokens": remaining}, open(tj, "w", encoding="utf-8"))
    stager.audit(pd, tj, [])
    avail = jload(os.path.join(pd, "batch_tokens.json"))
    loc = {t: remaining[t] for t in avail["tokens"]}
    batches = make_batches(mp, loc, sets_per_tok, "L")
    ctx.set(phase_L={"tokens_unsettled": len(remaining), "tokens_local": len(loc), "batches": len(batches)}, stage="local-dump")
    for i, b in enumerate(batches):
        info = stager.audit_only(b)
        g = dump_batch(ctx, b, info, more_fn=lambda i=i: i + 1 < len(batches))
        bank_unit(ctx, b, info, g, "L")
        with ctx._lk:
            ctx.counts["batches_done"] += 1
            ctx.counts["sets_banked"] += int(g.get("n_sets_rows", 0))
            ctx.counts["tokens_banked"] += int(g.get("n_tokens_rows", 0))
        ctx.set(progress=dict(ctx.counts), state=f"phase L batch {i + 1}/{len(batches)} banked")


def run_phase_stage(ctx, mp, stager, sets_per_tok):
    """PHASE S: everything else, staged from the pod in log batches (waits while the pod is busy), dumped, banked, staged copies deleted."""
    units = valid_units(ctx)
    remaining = {t: l for t, l in mp["tokens"].items() if t not in settled_tokens(units)}
    if not remaining:
        return
    batches = make_batches(mp, remaining, sets_per_tok, "S")
    ctx.set(phase_S={"tokens_remaining": len(remaining), "batches": len(batches)}, stage="staging")
    q: queue.Queue = queue.Queue()
    slots = threading.Semaphore(2)                       # at most TWO batches on disk at once: the one being dumped and the one staged ahead (released after the cleanup)
    err: dict = {}
    tokens_by_log = {r["log"]: r["n_tokens"] for r in mp["logs"]}

    def producer():
        try:
            for b in batches:
                while not slots.acquire(timeout=5):
                    if ctx.abort.is_set():
                        return
                if ctx.abort.is_set():
                    break
                need = stager.estimate_bytes(b["logs"], tokens_by_log)
                t0, said = time.time(), False
                while (C.free_gb(stager.stage_root) - need / 2 ** 30 < 22.0 or C.free_gb(ctx.work) < 5.5) and not ctx.abort.is_set():      # D: keeps >= 20 GiB free AFTER the batch (22 with margin), C: >= 5 GiB (the coordinator)
                    if not said or time.time() - t0 > 600:
                        ctx.set(guard=f"staging of {b['id']} WAITS for disk: D: free {C.free_gb(stager.stage_root):.1f} GiB (batch needs ~{need / 2 ** 30:.1f} GiB, keep 22 spare), C: free {C.free_gb(ctx.work):.1f} GiB")
                        said, t0 = True, time.time()
                    time.sleep(30)
                ctx.set(stage="staging", state=f"staging {b['id']} ({len(b['logs'])} logs, {b['n_tokens']} tokens)")
                info = stager.stage(b["id"], b["logs"], b["tokens"])
                ctx.set(stage="staging", state=f"staged {b['id']}: {info['n_tokens_ok']}/{info['n_tokens_in']} tokens ready, {info['bytes_staged'] / 2 ** 20:.0f} MiB copied",
                        **{f"staged_{b['id']}": {k: info[k] for k in ("n_tokens_ok", "n_tokens_in", "blocked_reasons", "missing_dbs_on_pod", "missing_frames_on_pod", "bytes_staged")}})
                q.put((b, info))
        except BaseException as e:  # noqa: BLE001
            err["e"] = e
            err["tb"] = traceback.format_exc()
            ctx.abort_reason = f"rc3 staging failed: {e}"
            ctx.abort.set()
        finally:
            q.put(None)

    th = threading.Thread(target=producer, daemon=True)
    th.start()
    n_seen = 0
    while True:
        try:
            item = q.get(timeout=20)
        except queue.Empty:
            if ctx.abort.is_set() and not th.is_alive():
                break
            continue
        if item is None:
            break
        b, info = item
        n_seen += 1
        g = dump_batch(ctx, b, info, more_fn=lambda: not q.empty())
        bank_unit(ctx, b, info, g, "S")
        n, freed = stager.cleanup(info)
        ctx.set(stage="cleanup", state=f"{b['id']} banked; {n} staged files deleted ({freed / 2 ** 20:.0f} MiB freed); C: free {C.free_gb(ctx.work):.1f} GiB")
        slots.release()
        with ctx._lk:
            ctx.counts["batches_done"] += 1
            ctx.counts["sets_banked"] += int(g.get("n_sets_rows", 0))
            ctx.counts["tokens_banked"] += int(g.get("n_tokens_rows", 0))
        ctx.set(progress=dict(ctx.counts), batches_remaining=len(batches) - n_seen, guard=None)
    th.join(timeout=30)
    if err:
        raise Stop(ctx.abort_reason or "rc3 staging failed")


def main(argv=None):
    a = make_parser().parse_args(argv)
    global MIN_FREE_RAM_MB
    MIN_FREE_RAM_MB = int(a.min_free_ram_mb)
    ctx = Ctx(a)
    wd = ctx.wd
    done, failed = os.path.join(wd, "S0_DONE"), os.path.join(wd, "S0_FAILED")
    for p in (done, failed):
        if os.path.exists(p):
            os.remove(p)
    print(f"[{now()}] srttc_s0_driver start tag={a.tag} pid={os.getpid()} device={a.device}", flush=True)
    with open(os.path.join(wd, "driver.pid"), "w") as f:
        f.write(str(os.getpid()))
    ctx.set(tag=a.tag, pid=os.getpid(), device=a.device, started=ctx.status.get("started", now()), resumed_at=now(), result=None, guard=None)
    hb = Heartbeat(ctx)
    hb.start()
    atexit.register(C.drop_gpu, ctx)
    if os.name == "nt":
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        except Exception:  # noqa: BLE001
            pass

    def fail(msg, rc=1):
        with open(failed, "w", encoding="utf-8") as f:
            f.write(f"{now()} rc={rc} {msg}\n--- STATUS.json ---\n")
            f.write(C._read_text(ctx.status_p) if os.path.exists(ctx.status_p) else "(none)")
        ctx.set(result=f"FAILED rc={rc}: {msg[:300]}")
        print(f"[{now()}] FAILED rc={rc}: {msg}", flush=True)
        return rc

    try:
        mp = stage_preflight(ctx)
        ctx.master_sets = {(l, t, s_, r): c for (l, t, s_, r, c) in mp["sets"]}
        if a.tag != "main":
            keep = set(x for x in a.smoke_logs.split(",") if x)
            mp = dict(mp, logs=[r for r in mp["logs"] if r["log"] in keep])
            mp["tokens"] = {t: l for t, l in mp["tokens"].items() if l in keep}
        sets_per_tok: dict = {}
        for (l, t, s_, r) in ctx.master_sets:
            if t in mp["tokens"]:
                sets_per_tok[t] = sets_per_tok.get(t, 0) + 1
        ctx.set(n_tokens_planned=len(mp["tokens"]), n_sets_planned=sum(sets_per_tok.values()))
        units = valid_units(ctx)
        ctx.counts.update(batches_done=len(units), sets_banked=sum(int(u.get("gate", {}).get("n_sets_rows", 0)) for u in units.values()),
                          tokens_banked=sum(len(u.get("tokens_banked") or []) for u in units.values()))
        ctx.set(progress=dict(ctx.counts), resumed_units=len(units))
        stager = ST.Stager(wd, ctx.work, EC.DRIVERL_PY, log=lambda m: print(m, flush=True), stop_event=ctx.abort, status=lambda m: ctx.set(guard=m))
        os.makedirs(stager.stage_db_dir, exist_ok=True)
        stage_dgdump(ctx)
        C.drop_gpu(ctx)
        run_phase_local(ctx, mp, stager, sets_per_tok)
        C.drop_gpu(ctx)
        if not a.no_stage:
            run_phase_stage(ctx, mp, stager, sets_per_tok)
        C.drop_gpu(ctx)
        # ---- final assembly
        ctx.set(stage="final")
        summ = assemble(ctx, mp, sets_per_tok)
        jdump(summ, os.path.join(wd, "S0_SUMMARY.json"))
        if summ["replacement_rule_exceeded"]:
            return fail(f"rc5 coverage: {summ['n_logs_without_any_token']} of {summ['n_logs']} selected logs have no usable token (> 5 %): the sample would be drifting towards availability (A2.1 item 7)", 5)
        with open(done, "w", encoding="utf-8") as f:
            json.dump({"when": now(), "PARTIAL": summ["PARTIAL"], "n_logs": summ["n_logs"], "n_tokens_planned": summ["n_tokens_planned"], "n_tokens_banked": summ["n_tokens_banked"],
                       "n_sets_planned": summ["n_sets_planned"], "n_sets_banked": summ["n_sets_banked"], "n_units": summ["n_units"], "summary": "S0_SUMMARY.json",
                       "bank_dir": f"{ctx.work}/bank", "manifest": "MANIFEST.sha256.json (written by the post-step)"}, f, indent=1)
        ctx.set(result="S0_DONE")
        print(f"[{now()}] S0_DONE", flush=True)
        return 0
    except Stop as e:
        ctx.abort.set()
        m = str(e)
        return fail(m, int(m.split()[0][2:]) if m.startswith("rc") and m.split()[0][2:].isdigit() else (6 if m.startswith("lowdisk") else 1))
    except BaseException:  # noqa: BLE001
        ctx.abort.set()
        return fail("EXCEPTION\n" + traceback.format_exc(), 1)
    finally:
        hb.stop_ev.set()


def assemble(ctx, mp, sets_per_tok):
    """coverage per log + totals from the valid unit states; integrity aggregates of the identity controls.  Counts only: no bank file is opened."""
    units = valid_units(ctx)
    tok_log = mp["tokens"]
    per_log = {r["log"]: {"tokens_planned": r["n_tokens"], "sets_planned": r["n_sets"], "tokens_banked": 0, "sets_banked": 0, "tokens_missed": 0, "tokens_blocked_final": 0} for r in mp["logs"]}
    ctrl = {"fanpath_logit_max": 0.0, "n_sets": 0, "setpath_logit_differs_from_fan_gt_1e-3": 0}
    phases: dict = {}
    for uid, st in sorted(units.items()):
        phases[st.get("phase", "?")] = phases.get(st.get("phase", "?"), 0) + 1
        for t in st.get("tokens_banked") or []:
            per_log[tok_log[t]]["tokens_banked"] += 1
            per_log[tok_log[t]]["sets_banked"] += sets_per_tok.get(t, 0)
        for t in st.get("tokens_missed") or []:
            per_log[tok_log[t]]["tokens_missed"] += 1
        for t in st.get("tokens_blocked_final") or []:
            per_log[tok_log[t]]["tokens_blocked_final"] += 1
        c_ = (st.get("gate") or {}).get("ctrl")
        if c_:
            ctrl["fanpath_logit_max"] = max(ctrl["fanpath_logit_max"], float(c_["fanpath_logit_max"]))
            ctrl["n_sets"] += int(c_["n_sets"])
            ctrl["setpath_logit_differs_from_fan_gt_1e-3"] += int(c_["setpath_logit_differs_from_fan_gt_1e-3"])
    n_tok = sum(v["tokens_banked"] for v in per_log.values())
    n_set = sum(v["sets_banked"] for v in per_log.values())
    planned_t = sum(r["n_tokens"] for r in mp["logs"])
    planned_s = sum(r["n_sets"] for r in mp["logs"])
    empty_logs = [lg for lg, v in per_log.items() if v["tokens_banked"] == 0]
    partial_logs = {lg: {"tokens_banked_of_planned": [v["tokens_banked"], v["tokens_planned"]], "missed": v["tokens_missed"], "blocked_final": v["tokens_blocked_final"]}
                    for lg, v in per_log.items() if 0 < v["tokens_banked"] < v["tokens_planned"]}
    unsettled = [t for t in tok_log if t not in settled_tokens(units)]
    return {"what": "SR-TTC S0 summary (counts and integrity aggregates only)", "written": now(), "n_logs": len(mp["logs"]), "n_units": len(units), "units_by_phase": phases,
            "n_tokens_planned": planned_t, "n_tokens_banked": n_tok, "n_sets_planned": planned_s, "n_sets_banked": n_set, "PARTIAL": bool(n_set < planned_s),
            "n_tokens_unsettled": len(unsettled), "n_tokens_missed_margin": sum(v["tokens_missed"] for v in per_log.values()), "n_tokens_blocked_final": sum(v["tokens_blocked_final"] for v in per_log.values()),
            "n_logs_without_any_token": len(empty_logs), "logs_without_any_token": empty_logs, "logs_partial": partial_logs,
            "replacement_rule_exceeded": bool(len(empty_logs) > 0.05 * len(mp["logs"])),
            "note_replacement": "A2.1 item 7 replaces a selected log that exists on NO host by the next log in key order; this driver does NOT replace automatically: logs_without_any_token is the list for the Master Mind.",
            "identity_controls": ctrl, "per_log": per_log}


if __name__ == "__main__":
    sys.exit(main())
