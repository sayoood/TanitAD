#!/usr/bin/env python3
"""C3 selection-headroom test (eval/P6_GO_SIGNAL.md section 3 "C3") -- the measurement pipeline.

    dump   u400 re-run on tokens.json under the shared dev-box GPU lock, ALL 64 proposals + raw scorer logits
           (refe_navtest_seam.py --dump-proposals, the exact stage-2 invocation: --rule navsim_v1_lane)
    gate   FIDELITY GATE: the dump's pick must reproduce the banked u400 seam, max |d| <= 1e-3 m over all tokens
    score  one single-proposal seam per proposal index k, each scored by score_navtest_refe.py UNCHANGED
           (W3's harness) -- the tooling eval/proposal_table.py used for refe_a5_ep015_m5r2_p00..p63
    table  PDMS + the six sub-scores per (token, proposal) -> table.npz; G2 table-at-pick == the banked csv
    all    dump -> gate -> score -> table -> c3_analyze.py ; stops at the first failed gate

Resumable: a stage whose output exists and passed is not redone. Progress is in STATUS.json (this directory).
Nothing is fitted on these tokens. `--tag smoke` runs a plumbing check on a tiny token file (CPU, no lock).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, "..", ".."))
EVAL = os.path.join(PKG, "eval")
sys.path.insert(0, EVAL)
import eval_checkpoint as EC  # noqa: E402  (one source for the venv + env, as proposal_table.py)

CKPT = "D:/Projects/TanitAD/data/refe_runs_eval/sft4_peek/model_sft4_B_u400.pt"
CKPT_MD5_PREFIX = "8e382eeb"
RULE = "navsim_v1_lane"
FRAMES = f"{EC.DATA}/frames"
DB_DIR = "D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/test"
BANKED_SEAM = f"{EC.DATA}/seams/stage2_sft4_B_u400_navtest_full.npz"
BANKED_CSV = (f"{EC.DATA}/score/refe_stage2_sft4_B_u400_navtest_full/"
              "refe_stage2_sft4_B_u400_navtest_full.csv")
LOCK = "C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock"
MARK = "C:/Users/Admin/qland/work/refcv7/devbox_gpu.chain_active"
OWN_JOB_PREFIX = "c3-selection-headroom-"        # the `job` prefix of the lock body stage_dump writes: the ONLY lock this helper may clear
FIDELITY_TOL_M = 1e-3
GPU_MAX_USED_MIB = 3500          # wait for nvidia-smi memory.used at or below this before loading (8188 MiB card; desktop baseline ~1-3 GB)
SUB = ("no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
       "time_to_collision_within_bound", "comfort", "driving_direction_compliance")
M = 64            # proposals per token (the dump asserts it); --m < 64 is for the plumbing smoke only


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


class Ctx:
    def __init__(self, a):
        self.a = a
        self.tag = a.tag
        self.wd = HERE if a.tag == "main" else os.path.join(HERE, f"_{a.tag}")
        os.makedirs(os.path.join(self.wd, "logs"), exist_ok=True)
        self.tokens = a.tokens or os.path.join(HERE, "tokens.json")
        self.doc = json.load(open(self.tokens, encoding="utf-8"))
        self.toks = list(self.doc["tokens"])
        self.dump = os.path.join(self.wd, "proposals.npz")
        self.rerun_seam = f"{EC.DATA}/seams/c3/{self.tag}/refe_c3_u400_dumprun.npz"
        self.M = a.m
        self.status_p = os.path.join(self.wd, "STATUS.json")
        self.status = json.load(open(self.status_p)) if os.path.exists(self.status_p) else {}

    def set(self, **kw):
        self.status.update(kw, updated=now())
        json.dump(self.status, open(self.status_p, "w"), indent=1)
        print(f"[{now()}] STATUS {kw}", flush=True)

    def label(self, k):
        return f"refe_c3_u400_{self.tag}_p{k:02d}"

    def seam_k(self, k):
        return f"{EC.DATA}/seams/c3/{self.tag}/{self.label(k)}.npz"

    def csv_k(self, k):
        return f"{EC.DATA}/score/{self.label(k)}/{self.label(k)}.csv"


# --------------------------------------------------------------------------------------- GPU lock
def pid_alive(pid):
    # bytes, not text=True: tasklist's localised "no tasks" line carries 0x81 (cp850 u-umlaut), which cp1252 cannot decode -> r.stdout was None and a DEAD pid raised
    # TypeError (MEASURED 2026-10-18), so the helper crashed instead of ever judging a lock stale.
    r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"], capture_output=True)
    return f'"{pid}"' in r.stdout.decode("utf-8", "replace")


def take_lock(me, ctx):
    """Poll FAST (3 s): a neighbour job that re-takes the lock the moment it frees it would starve a 60 s poller.
    The lock alone is not enough -- the neighbour's process may stay CUDA-resident between its lock holds -- so a lock is
    taken only while nvidia-smi memory.used <= GPU_MAX_USED_MIB, and re-checked after taking it."""
    n = 0
    t_start = time.time()
    while True:
        used = gpu_used_mib()
        if not os.path.exists(MARK) and 0 <= used <= GPU_MAX_USED_MIB:
            try:
                fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, me.encode())
                os.close(fd)
                time.sleep(4)                                   # the neighbour's allocation may land just after it released
                if 0 <= gpu_used_mib() <= GPU_MAX_USED_MIB:
                    return
                release(me)                                     # memory is still held by a resident process: give the lock back
            except FileExistsError:
                try:                       # a lock whose holder process is gone (and which is not fresh) is stale
                    holder = json.loads(open(LOCK, encoding="utf-8").read())
                    age = time.time() - os.path.getmtime(LOCK)
                    # THE LOCK IS SHARED with other job families that write a DIFFERENT schema, e.g. the refcv8 battery's {job, pid = an MSYS pid, win_pid_of_locker, host, acquired}.
                    # An MSYS pid is invisible to tasklist, so "pid not alive" cannot tell such a live job from a dead one: we clear ONLY a lock carrying OUR job marker.
                    if isinstance(holder, dict) and str(holder.get("job", "")).startswith(OWN_JOB_PREFIX) and not pid_alive(holder["pid"]) and age > 600:
                        print(f"[{now()}] STALE lock (pid {holder['pid']} gone, age {age:.0f} s) -> removing: {holder}", flush=True)
                        os.remove(LOCK)
                        continue
                except (OSError, ValueError, KeyError):
                    pass
        n += 1
        if n % 100 == 1:                                        # every ~5 min
            try:
                h = open(LOCK, encoding="utf-8").read()[:160]
            except OSError:
                h = "(no lock file)"
            ctx.set(stage="dump", state=f"waiting {round((time.time() - t_start) / 60)} min; lock: {h}; gpu memory.used {used} MiB (need <= {GPU_MAX_USED_MIB})")
        time.sleep(3)


def release(me):
    try:
        if open(LOCK, encoding="utf-8").read() == me:
            os.remove(LOCK)
            print(f"[{now()}] GPU lock released", flush=True)
    except OSError:
        pass


def gpu_used_mib():
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    try:
        return int(r.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return -1


# --------------------------------------------------------------------------------------------- dump
def stage_dump(ctx):
    a = ctx.a
    log = os.path.join(ctx.wd, "logs", "1_dump.log")
    if os.path.exists(ctx.dump) and os.path.exists(log) and "ZZSEAM_" in open(log, encoding="utf-8", errors="replace").read():
        print("dump: already done", flush=True)
        return True
    os.makedirs(os.path.dirname(ctx.rerun_seam), exist_ok=True)
    cmd = [EC.DRIVERL_PY, "refe_navtest_seam.py", "--ckpt", CKPT, "--frames", FRAMES, "--db-dir", DB_DIR,
           "--tokens", ctx.tokens, "--out", ctx.rerun_seam, "--arm", "REFe_c3_u400_dumprun", "--rule", RULE,
           "--dump-proposals", ctx.dump, "--device", a.device]
    me = json.dumps({"job": f"c3-selection-headroom-{ctx.tag}", "pid": os.getpid(), "host": os.environ.get("COMPUTERNAME", "?"),
                     "acquired": now()})
    locked = a.device == "cuda"
    for attempt in range(1, 6):
        if locked:
            take_lock(me, ctx)
            print(f"[{now()}] GPU lock taken (attempt {attempt}); memory.used {gpu_used_mib()} MiB before load", flush=True)
        try:
            ctx.set(stage="dump", state=f"running u400 seam+dump (attempt {attempt})", started=now(), gpu_used_mib_before=gpu_used_mib())
            t0 = time.time()
            with open(log, "w", encoding="utf-8") as f:
                rc = subprocess.call(cmd, cwd=EVAL, env=EC.env_driverl(), stdout=f, stderr=subprocess.STDOUT)
            ctx.set(stage="dump", state=f"seam rc={rc} (attempt {attempt})", dump_seconds=round(time.time() - t0))
        finally:
            if locked:
                release(me)
        if os.path.exists(ctx.dump):
            break
        time.sleep(30)                                          # e.g. a CUDA OOM against a neighbour: wait for a clean window and retry
    return os.path.exists(ctx.dump)


# --------------------------------------------------------------------------------------------- gate
def stage_gate(ctx):
    """FIDELITY GATE (pre-registered): max |d pose| <= 1e-3 m between the dump's pick and the banked u400 plan."""
    D = np.load(ctx.dump)
    tok = [str(t) for t in D["token"]]
    P, pick = D["proposals"], D["pick"]
    N = len(tok)
    B = np.load(BANKED_SEAM)
    bidx = {str(t): i for i, t in enumerate(B["token"])}
    g = {"n_tokens_asked": len(ctx.toks), "n_rows": N, "proposals_shape": list(P.shape), "logits_shape": list(D["logits"].shape),
         "rule_in_dump": str(D["rule"]), "ckpt_in_dump": str(D["ckpt"]), "tolerance_m": FIDELITY_TOL_M}
    g["all_tokens_present"] = bool(set(tok) == set(ctx.toks) and N == len(ctx.toks))
    g["tokens_in_banked_seam"] = bool(all(t in bidx for t in tok))
    ok = g["all_tokens_present"] and g["tokens_in_banked_seam"] and P.shape[1] == M and str(D["rule"]) == RULE
    if ok:
        ban = B["poses"][[bidx[t] for t in tok]].astype(np.float64)
        sel = P[np.arange(N), pick].astype(np.float64)
        d = np.abs(sel - ban).reshape(N, -1).max(1)
        fp_ok = all(str(D["fingerprint"][i]) == str(B["fingerprint"][bidx[t]]) for i, t in enumerate(tok))
        g.update(max_abs_pose_diff_m=float(d.max()), n_identical=int((d == 0).sum()), n_diff_gt_1e_4=int((d > 1e-4).sum()),
                 n_diff_gt_tol=int((d > FIDELITY_TOL_M).sum()), fingerprints_equal=bool(fp_ok),
                 worst_tokens=[(tok[i], float(d[i])) for i in np.argsort(-d)[:5]],
                 pick_index_distinct=int(len(set(int(x) for x in pick))))
        ok = bool(d.max() <= FIDELITY_TOL_M and fp_ok)
    g["PASS"] = ok
    json.dump(g, open(os.path.join(ctx.wd, "gate_fidelity.json"), "w"), indent=1)
    ctx.set(stage="gate", fidelity_pass=ok, fidelity=g)
    return ok


# -------------------------------------------------------------------------------------------- score
def status_of(txt):
    for line in reversed(txt.splitlines()):
        if line.startswith("{") and '"status"' in line:
            return json.loads(line)
    return None


def free_mb():
    r = subprocess.run(["powershell", "-NoProfile", "-Command",
                        "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1024)"], capture_output=True, text=True)
    try:
        return int(r.stdout.strip())
    except ValueError:
        return 0


def stage_score(ctx):
    D = np.load(ctx.dump)
    P = D["proposals"]
    N = P.shape[0]
    os.makedirs(os.path.dirname(ctx.seam_k(0)), exist_ok=True)
    for k in range(ctx.M):
        if not os.path.exists(ctx.seam_k(k)):
            np.savez(ctx.seam_k(k), token=D["token"], fingerprint=D["fingerprint"], poses=P[:, k].astype(np.float32),
                     sampling=D["sampling"], arm=np.array(f"REFe_c3_u400_{ctx.tag}_p{k:02d}"))
    ks = list(range(ctx.M)) if ctx.a.only is None else list(ctx.a.only)

    def score_one(k):
        lp = os.path.join(ctx.wd, "logs", f"p{k:02d}.log")
        st = None
        for att in range(1, 6):
            if os.path.exists(ctx.csv_k(k)) and os.path.exists(lp):
                st = status_of(open(lp, encoding="utf-8", errors="replace").read())
                if st and st.get("status") == "PASS" and st.get("csv_valid_rows") == N:
                    return k, st
            while free_mb() < 4000:          # the harness has a RAM guard; do not start into it
                time.sleep(30)
            EC.run([EC.DRIVERL_PY, "score_navtest_refe.py", "--label", ctx.label(k), "--seam", ctx.seam_k(k),
                    "--tokens", ctx.tokens, "--out", f"{EC.DATA}/score"],
                   EVAL, dict(os.environ, PYTHONIOENCODING="utf-8"), lp)
            st = status_of(open(lp, encoding="utf-8", errors="replace").read())
            if st and st.get("status") == "PASS" and st.get("csv_valid_rows") == N:
                return k, st
            time.sleep(90)
        return k, st

    t1 = time.time()
    done = {}
    with ThreadPoolExecutor(max_workers=max(1, ctx.a.workers)) as ex:
        for i, (k, st) in enumerate(ex.map(score_one, ks)):
            done[k] = st
            npass = sum(1 for s in done.values() if s and s.get("status") == "PASS")
            ctx.set(stage="score", state=f"{i + 1}/{len(ks)} scored, {npass} PASS", score_seconds=round(time.time() - t1))
    bad = [k for k in ks if not (done.get(k) and done[k].get("status") == "PASS" and done[k].get("csv_valid_rows") == N)]
    ctx.set(stage="score", g3_all_pass=not bad, g3_failed=bad[:20])
    return not bad


# -------------------------------------------------------------------------------------------- table
def read_csv(p):
    rows = {}
    for r in csv.DictReader(open(p, encoding="utf-8")):
        if r["token"] == "average":
            continue
        rows[r["token"]] = (r["valid"] == "True", float(r["score"]), tuple(float(r[c]) for c in SUB))
    return rows


def stage_table(ctx):
    if ctx.a.only is not None:
        print("table: skipped (--only smoke)")
        return True
    D = np.load(ctx.dump)
    tok = [str(t) for t in D["token"]]
    N = len(tok)
    Mx = ctx.M
    pdms = np.full((N, Mx), np.nan)
    sub = np.full((N, Mx, 6), np.nan)
    valid = np.zeros((N, Mx), dtype=bool)
    for k in range(Mx):
        rows = read_csv(ctx.csv_k(k))
        for i, t in enumerate(tok):
            v, s, c = rows[t]
            valid[i, k], pdms[i, k], sub[i, k] = v, s, c
    pick = D["pick"]
    g = {"G3_all_valid": bool(valid.all()), "G3_finite": bool(np.isfinite(pdms).all())}
    # G2 -- the table at the pick against the BANKED full-navtest per-token score (needs all 64 columns)
    B = np.load(BANKED_SEAM)
    bidx = {str(t): i for i, t in enumerate(B["token"])}
    lr = read_csv(BANKED_CSV)
    ban = B["poses"][[bidx[t] for t in tok]].astype(np.float64)
    dpos = np.abs(D["proposals"][np.arange(N), pick].astype(np.float64) - ban).reshape(N, -1).max(1)
    if Mx == M:
        at_pick = pdms[np.arange(N), pick]
        dd = np.array([abs(at_pick[i] - lr[t][1]) for i, t in enumerate(tok)])
        same = dpos == 0
        g["G2_banked_csv"] = BANKED_CSV
        g["G2_tokens_identical_poses"] = int(same.sum())
        g["G2_max_abs_score_diff_identical_poses"] = float(dd[same].max()) if same.any() else None
        g["G2_max_abs_score_diff_all"] = float(dd.max())
        g["G2_n_score_diff_gt_1e-9"] = int((dd > 1e-9).sum())
        g["G2_PASS"] = bool(same.any() and dd[same].max() <= 1e-9)
    else:
        g["G2_PASS"] = True
        g["G2_note"] = f"SMOKE with {Mx} columns: G2 not evaluated"
    np.savez_compressed(os.path.join(ctx.wd, "table.npz"), token=np.array(tok), pdms=pdms,
                        sub=sub, valid=valid, logits=D["logits"], proposals=D["proposals"], pick=pick,
                        sub_names=np.array(SUB), pdms_at_pick_banked=np.array([lr[t][1] for t in tok]))
    json.dump(g, open(os.path.join(ctx.wd, "gate_table.json"), "w"), indent=1)
    ctx.set(stage="table", table_gates=g)
    return bool(g["G3_all_valid"] and g["G3_finite"] and g["G2_PASS"])


# --------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("dump", "gate", "score", "table", "analyze", "all"))
    ap.add_argument("--tag", default="main")
    ap.add_argument("--tokens", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--m", type=int, default=M, help="number of proposal columns (64; smaller ONLY for the plumbing smoke)")
    ap.add_argument("--only", type=int, nargs="*", default=None)
    a = ap.parse_args()
    ctx = Ctx(a)
    ctx.set(tag=a.tag, n_tokens=len(ctx.toks), pid=os.getpid(), started=ctx.status.get("started", now()))
    st = a.stage
    if st in ("dump", "all") and not stage_dump(ctx):
        ctx.set(stage="dump", result="FAILED: no proposals.npz (see logs/1_dump.log)")
        return 1
    if st in ("gate", "all") and not stage_gate(ctx):
        ctx.set(stage="gate", result="INSTRUMENT FAILURE: the fidelity gate failed -- no verdict (gate_fidelity.json)")
        return 2
    if st in ("score", "all") and not stage_score(ctx):
        ctx.set(stage="score", result="FAILED: some proposal columns did not PASS the harness")
        return 3
    if st in ("table", "all") and not stage_table(ctx):
        ctx.set(stage="table", result="INSTRUMENT FAILURE: table gates failed (gate_table.json)")
        return 4
    if st in ("analyze", "all") and a.only is None:
        import c3_analyze
        rc = c3_analyze.main(["--dir", ctx.wd, "--tokens", ctx.tokens])
        ctx.set(stage="analyze", rc=rc, result="DONE -- see readout.json / RESULT draft")
        return rc
    ctx.set(result=f"stage {st} done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
