#!/usr/bin/env python3
"""P2 NAVTRAIN HELD-OUT 64-PROPOSAL TABLE -- the driver.  Parametrised from the landed full-navtest driver (landed/p2full_driver.py + landed/c3_run_p2full.py, blob f249a02a, IMPORTED UNCHANGED
for the lock / guard / state helpers; `p2ho_driver_vs_p2full_driver.diff` is the edit list of this file against the landed driver).

    P2 (model_final.pt, sha256 ea31db22..., rule navsim_v1) -> per held-out LOG: 64 proposals + 6 scorer logits + the scorer head's INPUT (one forward, GPU behind the shared dev-box lock)
    -> per-chunk dump gate -> PDM scoring of all 64 proposals of every token (navsim_pdm_targets = the instrument that LABELLED the held-out sets and P2's training sets), W workers
    -> table_p2_heldout.npz (same schema as table_p2_full.npz) + proposals.npz + prehead features -> G3 + the same-pick proofs -> DONE / FAILED marker.

Gates, in order (the first failure stops the job; each has a deliberate-regression arm that must read FAIL, see p2ho_fidelity.py / p2ho_dgdump.py / test_p2ho.py):
  G1  preflight       ckpt sha256; DISJOINTNESS PROOF vs navtest AND vs P2's own training bank (p2ho_preflight.py); plan audit (DB + four camera frames per token); disk / RAM
  G2  fidelity A      the scoring instrument reproduces the banked held-out labels to 1e-5 (rounded) on EVERY proposal of K sets per log, negative control reads FAIL
  G2b DG-DUMP         this job's dump script, on the GPU, reproduces the landed full-navtest table rows (pose <= 1e-3 m, logits <= 1e-4, pick equal), negative control reads FAIL
  G3  every (token, proposal) cell valid and finite
  G4  same-pick     per row: the planner rule recomputed from the table's own logits reproduces the dumped pick; PDMS == NC*DAC*(5EP+5TTC+2C)/12 on every cell
COVERAGE IS PART OF THE RESULT.  The plan holds only the tokens that have a DB AND four camera frames on this box; the DONE marker says PARTIAL and names every blocked token class.

Resumable: re-run the SAME command (done chunks and done score units are skipped).  Nothing is fitted on this table (it is SR-TTC's DEV GATE, PREREG_SELRANK_ARM.draft.md sec 3).
"""
from __future__ import annotations

import argparse
import atexit
import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                                                   # C:/Users/Admin/p2heldout  (the package)
TREE = os.environ.get("P2HO_TREE", f"{ROOT}/tree/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan")
LANDED = os.path.join(ROOT, "landed")
WORK_ROOT = "C:/Users/Admin/p2heldout_work"                                    # bulk (NTFS).  D: is FULL and is never written.
sys.path.insert(0, os.path.join(TREE, "eval"))
sys.path.insert(0, LANDED)
sys.path.insert(0, HERE)
import eval_checkpoint as EC  # noqa: E402  (the tree's: one source for the venvs + env, as the landed pipeline)
import c3_run_p2full as C     # noqa: E402  (landed, unchanged: lock protocol, guards, atomic_json, sha256)
import p2ho_preflight as PF   # noqa: E402

C.OWNER = "p2heldout_table"
CKPT, CKPT_SHA256 = PF.CKPT, PF.CKPT_SHA256
NAVSIM_PY = "C:/Users/Admin/navsim-crun/venv/Scripts/python.exe"
V11 = "D:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896"
MAPS = "D:/Archive/devbox-C/navsim/data/maps"
MIN_FREE_RAM_MB = 8000            # the brief: >= 8 GB free before a heavy start
MIN_FREE_DISK_GB = 5.0            # the brief: stop cleanly below 5 GB free (on the drive the job writes: C:)
SUB = ("no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound", "comfort", "driving_direction_compliance")
now, sha256_file, jdump, jload = C.now, C.sha256_file, C.atomic_json, C.jload


class Stop(Exception):
    pass


class Ctx:
    """The attributes the landed lock / guard helpers read: abort, set(), status, lock_body, a.device."""

    def __init__(self, a):
        self.a = a
        self.tag = a.tag
        self.main = a.tag == "main"
        self.wd = ROOT if self.main else os.path.join(ROOT, f"_{a.tag}")
        self.work = WORK_ROOT if self.main else os.path.join(WORK_ROOT, f"_{a.tag}")
        for d in ("logs", "state", "plan"):
            os.makedirs(os.path.join(self.wd, d), exist_ok=True)
        for d in ("dumps", "feats", "scores", "final"):
            os.makedirs(os.path.join(self.work, d), exist_ok=True)
        self.status_p = os.path.join(self.wd, "STATUS.json")
        self.status = jload(self.status_p) if os.path.exists(self.status_p) else {}
        self._lk = threading.RLock()
        self.abort = threading.Event()
        self.abort_reason = None
        self.lock_body = None
        self.start_lk = threading.Lock()
        self.last_start = 0.0
        self.counts = {"chunks_dumped": 0, "chunks_scored": 0, "score_fail": 0}

    def set(self, **kw):
        with self._lk:
            self.status.update(kw, updated=now())
            jdump(self.status, self.status_p)
        print(f"[{now()}] STATUS {json.dumps(kw, default=str)[:300]}", flush=True)


# ------------------------------------------------------------------------------------------ guards
def disk_guard(ctx, what):
    """STOP CLEANLY (never write into a nearly full disk) below MIN_FREE_DISK_GB on C:; wait (not fail) for RAM."""
    free = C.free_gb(WORK_ROOT)
    if free < MIN_FREE_DISK_GB:
        raise Stop(f"lowdisk: {WORK_ROOT} has {free:.2f} GiB free < {MIN_FREE_DISK_GB} before {what}")
    t0, said = time.time(), False
    while C.free_mb() < MIN_FREE_RAM_MB and not ctx.abort.is_set():
        if not said or time.time() - t0 > 600:
            ctx.set(guard=f"PAUSED before {what}: free RAM {C.free_mb()} MiB < {MIN_FREE_RAM_MB}")
            said, t0 = True, time.time()
        time.sleep(30)


def env_navsim():
    e = dict(os.environ)
    e.update({"PYTHONPATH": f"{V11};{TREE}/refe", "NUPLAN_MAPS_ROOT": MAPS, "NUPLAN_MAP_VERSION": "nuplan-maps-v1.0", "REFE_SIM_HZ": "10", "OMP_NUM_THREADS": "1",
              "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"})
    return e


def run_logged(cmd, env, log, cwd=HERE):
    with open(log, "w", encoding="utf-8") as f:
        return subprocess.call(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)


# ------------------------------------------------------------------------------------------ stages
def stage_preflight(ctx):
    a = ctx.a
    ctx.set(stage="preflight")
    pf = PF.proof(check_ckpt_sha=True)
    jdump(pf, os.path.join(ctx.wd, "disjointness_proof.json"))
    st = os.stat(CKPT)
    ctx.set(preflight_disjointness=pf["PASS_parts"], preflight_ckpt_sha256=pf["checkpoint"].get("sha256"), ckpt_stat=[st.st_size, st.st_mtime_ns])
    if not pf["PASS"]:
        raise Stop("rc1 preflight: disjointness proof / checkpoint FAILED (disjointness_proof.json)")
    if a.device == "cuda" and not os.path.exists(C.WAIT_IDLE):
        raise Stop(f"rc1 preflight: {C.WAIT_IDLE} (the idle waiter) is missing")
    if a.plan:
        plan_p = a.plan
    else:
        plan_p = os.path.join(ctx.wd, "plan", "heldout_tokens.json")
        rc = run_logged([EC.DRIVERL_PY, os.path.join(HERE, "p2ho_audit.py"), "--out", os.path.join(ctx.wd, "plan")], EC.env_driverl(), os.path.join(ctx.wd, "logs", "audit.log"))
        if rc:
            raise Stop("rc1 preflight: the plan audit failed (logs/audit.log)")
    plan = jload(plan_p)
    aud_p = os.path.join(os.path.dirname(plan_p), "audit.json")
    aud = jload(aud_p) if os.path.exists(aud_p) else {}
    cov = {"n_sets_banked": aud.get("n_sets"), "n_tokens_banked": aud.get("n_tokens"), "n_logs_banked": aud.get("n_logs"), "n_tokens_in_plan": len(plan["tokens"]),
           "n_logs_in_plan": len(set(plan["token_log"].values())), "n_tokens_blocked": aud.get("n_tokens_blocked"), "blocked_reasons": aud.get("blocked_reasons"),
           "n_logs_with_db": aud.get("n_logs_with_db"), "n_tokens_with_db": aud.get("n_tokens_with_db"), "plan": plan_p, "plan_sha256": sha256_file(plan_p),
           "PARTIAL": bool(aud) and len(plan["tokens"]) < aud.get("n_tokens", 0)}
    jdump(cov, os.path.join(ctx.wd, "COVERAGE.json"))
    ctx.set(coverage=cov)
    if not plan["tokens"]:
        raise Stop("rc1 preflight: the plan holds 0 tokens (no token has a DB AND four camera frames on this box): see COVERAGE.json / plan/audit.json")
    return plan, plan_p


def make_units(plan, max_tokens):
    """one unit per LOG (a log is never split: the map load is per log); a unit is keyed by the sha of its token list, so a plan that later gains frames makes a NEW unit."""
    by = {}
    for t in plan["tokens"]:
        by.setdefault(plan["token_log"][t], []).append(t)
    units = []
    for lg in sorted(by):
        toks = sorted(by[lg])
        for i in range(0, len(toks), max_tokens):
            part = toks[i:i + max_tokens]
            key = hashlib.sha256((lg + "|" + ",".join(part)).encode()).hexdigest()[:10]
            units.append({"key": key, "log": lg, "tokens": part, "n": len(part)})
    return units


def unit_plan_path(ctx, u):
    return os.path.join(ctx.wd, "plan", f"unit_{u['key']}.json")


def write_unit_plan(ctx, plan, u):
    p = unit_plan_path(ctx, u)
    if not os.path.exists(p):
        jdump({"tokens": u["tokens"], "token_log": {t: u["log"] for t in u["tokens"]}, "token_root": {t: plan["token_root"][t] for t in u["tokens"]},
               "db": {u["log"]: plan["db"][u["log"]]}}, p)
    return p


def dump_p(ctx, u):
    return f"{ctx.work}/dumps/dump_{u['key']}.npz"


def feat_p(ctx, u):
    return f"{ctx.work}/feats/feat_{u['key']}.npz"


def score_p(ctx, u):
    return f"{ctx.work}/scores/score_{u['key']}.npz"


def state_p(ctx, u):
    return os.path.join(ctx.wd, "state", f"unit_{u['key']}.json")


def gate_unit(ctx, u):
    """G-DUMP (per chunk, BEFORE it is scored): every asked token has a finite row, 64 proposals on NAVSIM's 8-pose grid, rule navsim_v1, A7 repair on, the P2 checkpoint, features finite."""
    D = np.load(dump_p(ctx, u))
    F = np.load(feat_p(ctx, u))
    tok = [str(t) for t in D["token"]]
    g = {"unit": u["key"], "log": u["log"], "n_asked": u["n"], "n_rows": len(tok), "proposals_shape": list(D["proposals"].shape), "logits_shape": list(D["logits"].shape),
         "rule": str(D["rule"]), "repair_last_heading": bool(D["repair_last_heading"]), "ckpt": str(D["ckpt"]),
         "all_tokens_present": bool(set(tok) == set(u["tokens"]) and len(tok) == len(set(tok)) == u["n"])}
    P, L = D["proposals"], D["logits"]
    pick = D["pick"]
    g["finite"] = bool(np.isfinite(P).all() and np.isfinite(L).all())
    g["pick_in_range"] = bool(((pick >= 0) & (pick < 64)).all())
    g["features_shape"] = list(F["prehead_features"].shape)
    g["features_finite"] = bool(np.isfinite(F["prehead_features"].astype(np.float32)).all())
    g["features_token_order_equal"] = bool([str(t) for t in F["token"]] == tok)
    rep_p = os.path.splitext(dump_p(ctx, u))[0] + ".report.json"
    g["report_misses"] = jload(rep_p).get("misses") if os.path.exists(rep_p) else None
    g["PASS"] = bool(g["all_tokens_present"] and P.shape[1:] == (64, 8, 3) and L.shape[1:] == (64, 6) and g["rule"] == "navsim_v1" and g["repair_last_heading"]
                     and g["ckpt"] == CKPT and g["finite"] and g["pick_in_range"] and g["features_shape"][1:] == [64, 256] and g["features_finite"]
                     and g["features_token_order_equal"] and g["report_misses"] == 0)
    jdump(g, os.path.join(ctx.wd, "state", f"gate_dump_{u['key']}.json"))
    return g


def unit_dump_done(ctx, u):
    sp = state_p(ctx, u)
    if not (os.path.exists(sp) and os.path.exists(dump_p(ctx, u)) and os.path.exists(feat_p(ctx, u))):
        return False
    try:
        st = jload(sp)
        return bool(st.get("gate_PASS") and st.get("dump_sha256") == sha256_file(dump_p(ctx, u)))
    except (OSError, ValueError):
        return False


def stage_gate2(ctx):
    """G2: fidelity control A (+ B reported).  Cached on a PASS file."""
    fp = os.path.join(ctx.wd, "fidelity.json")
    if os.path.exists(fp) and jload(fp).get("PASS"):
        return jload(fp)
    ctx.set(stage="fidelity")
    disk_guard(ctx, "the fidelity control")
    rc = run_logged([NAVSIM_PY, os.path.join(HERE, "p2ho_fidelity.py"), "--out", fp, "--tree", TREE, "--n-logs", str(ctx.a.fid_logs), "--sets-per-log", str(ctx.a.fid_sets),
                     "--nav-tokens", str(ctx.a.fid_nav)], env_navsim(), os.path.join(ctx.wd, "logs", "fidelity.log"))
    R = jload(fp) if os.path.exists(fp) else {"PASS": False}
    ctx.set(fidelity_A=R.get("A", {}).get("PASS"), fidelity_B=R.get("B"))
    if rc or not R.get("PASS"):
        raise Stop("rc2 fidelity: the scoring instrument does NOT reproduce the banked held-out labels (fidelity.json) -- report it, do not re-run")
    return R


def stage_dgdump(ctx):
    fp = os.path.join(ctx.wd, "dgdump.json")
    if os.path.exists(fp) and jload(fp).get("PASS"):
        return jload(fp)
    ctx.set(stage="dgdump")
    disk_guard(ctx, "DG-DUMP")
    locked = ctx.a.device == "cuda"
    try:
        if locked:
            C.hold_gpu(ctx, "p2heldout-dgdump")
        rc = run_logged([EC.DRIVERL_PY, os.path.join(HERE, "p2ho_dgdump.py"), "--out", fp, "--workdir", os.path.join(ctx.work, "dgdump"), "--k", str(ctx.a.dg_k), "--device", ctx.a.device],
                        EC.env_driverl(), os.path.join(ctx.wd, "logs", "dgdump.log"))
    finally:
        if locked and (ctx.abort.is_set() or C.waiters()):
            C.drop_gpu(ctx)
    R = jload(fp) if os.path.exists(fp) else {"PASS": False}
    ctx.set(dgdump={k: R.get(k) for k in ("PASS", "n", "proposals_max_abs_m", "logits_max_abs", "pick_equal")})
    if rc or not R.get("PASS"):
        C.drop_gpu(ctx)
        raise Stop("rc2 DG-DUMP: this dump script does not reproduce the landed full-navtest table rows (dgdump.json) -- report it, do not re-run")
    return R


def ensure_dump(ctx, plan, u, more_to_dump):
    if unit_dump_done(ctx, u):
        print(f"[{now()}] unit {u['key']} ({u['log']}): dump already done + gated", flush=True)
        return "ok"
    locked = ctx.a.device == "cuda"
    pp = write_unit_plan(ctx, plan, u)
    for attempt in range(1, 4):
        if ctx.abort.is_set():
            return "dump_failed"
        disk_guard(ctx, f"dump of {u['log']}")
        part, fpart = dump_p(ctx, u)[:-4] + ".part.npz", feat_p(ctx, u)[:-4] + ".part.npz"
        try:
            if locked:
                C.hold_gpu(ctx, f"p2heldout-{u['key']}")
            for x in (part, fpart):
                if os.path.exists(x):
                    os.remove(x)
            st = os.stat(CKPT)
            if [st.st_size, st.st_mtime_ns] != ctx.status.get("ckpt_stat"):
                C.drop_gpu(ctx)
                raise Stop("rc1 the checkpoint file changed since the preflight")
            ctx.set(stage="dump", state=f"unit {u['key']} {u['log']}: dumping {u['n']} tokens (attempt {attempt})")
            t0 = time.time()
            lp = os.path.join(ctx.wd, "logs", f"dump_{u['key']}.log")
            rc = run_logged([EC.DRIVERL_PY, os.path.join(HERE, "p2ho_dump.py"), "--ckpt", CKPT, "--plan", pp, "--out", part, "--feat-out", fpart, "--device", ctx.a.device,
                             "--arm", f"REFe_p2heldout_{u['key']}"], EC.env_driverl(), lp)
            ctx.set(stage="dump", state=f"unit {u['key']}: dump rc={rc}", last_dump_seconds=round(time.time() - t0))
        except BaseException:
            C.drop_gpu(ctx)
            raise
        if locked and (not os.path.exists(part) or not more_to_dump or C.waiters()):
            C.drop_gpu(ctx)                    # failed attempt / last dump / a neighbour is queued: give the GPU back
        if os.path.exists(part) and os.path.exists(fpart):
            os.replace(part, dump_p(ctx, u))
            os.replace(fpart, feat_p(ctx, u))
            rsrc = os.path.splitext(part)[0] + ".report.json"          # p2ho_dump writes <out minus .npz>.report.json  ('.part.report.json')
            if os.path.exists(rsrc):
                os.replace(rsrc, os.path.splitext(dump_p(ctx, u))[0] + ".report.json")
            g = gate_unit(ctx, u)
            ctx.set(stage="gate", **{f"gate_{u['key']}": {k: g.get(k) for k in ("PASS", "n_rows", "all_tokens_present", "finite", "features_finite")}})
            if not g["PASS"]:
                C.drop_gpu(ctx)
                return "gate_failed"
            jdump({"unit": u["key"], "log": u["log"], "n": u["n"], "gate_PASS": True, "dump_sha256": sha256_file(dump_p(ctx, u)), "feat_sha256": sha256_file(feat_p(ctx, u)),
                   "when": now(), "gate": g}, state_p(ctx, u))
            with ctx._lk:
                ctx.counts["chunks_dumped"] += 1
            return "ok"
        time.sleep(30 if locked else 1)
    return "dump_failed"


def score_unit(ctx, plan, u):
    sp = score_p(ctx, u)
    if os.path.exists(sp):
        try:
            z = np.load(sp)
            if bool(z["valid_token"].all()) and len(z["token"]) == u["n"]:
                return True
        except Exception:  # noqa: BLE001
            pass
    for attempt in range(1, 3):
        if ctx.abort.is_set():
            return False
        with ctx.start_lk:                       # one start at a time, >= 15 s apart, each behind the RAM / disk guard
            disk_guard(ctx, f"scoring {u['log']}")
            dt = time.time() - ctx.last_start
            if dt < (C.STAGGER_S if ctx.main else 0.05):
                time.sleep((C.STAGGER_S if ctx.main else 0.05) - dt)
            ctx.last_start = time.time()
        pp = write_unit_plan(ctx, plan, u)
        rc = run_logged([NAVSIM_PY, os.path.join(HERE, "p2ho_score.py"), "--dump", dump_p(ctx, u), "--plan", pp, "--out", sp, "--tree", TREE], env_navsim(),
                        os.path.join(ctx.wd, "logs", f"score_{u['key']}.log"))
        if rc == 0 and os.path.exists(sp):
            with ctx._lk:
                ctx.counts["chunks_scored"] += 1
            ctx.set(progress=dict(ctx.counts))
            return True
        time.sleep(30 if ctx.main else 0.2)
    with ctx._lk:
        ctx.counts["score_fail"] += 1
    return False


def pick_from_logits(logits, dtype=np.float64):
    x = np.asarray(logits, dtype)
    p = 1.0 / (1.0 + np.exp(-x))
    agg = p[..., 0] * p[..., 1] * (5 * p[..., 2] + 5 * p[..., 3] + 2 * p[..., 4]) / 12.0
    return agg.argmax(-1)


def stage_table(ctx, plan, units):
    """merge dumps + scores -> table_p2_heldout.npz (schema of table_p2_full.npz) + proposals.npz + prehead features.  G3 + G4 (same-pick proofs)."""
    Ds, Fs, Ss = [], [], []
    for u in units:
        Ds.append(np.load(dump_p(ctx, u)))
        Fs.append(np.load(feat_p(ctx, u)))
        Ss.append(np.load(score_p(ctx, u)))
    cat = lambda objs, key: np.concatenate([o[key] for o in objs], axis=0)  # noqa: E731
    tok = np.array([str(t) for t in cat(Ds, "token")])
    N = len(tok)
    assert len(set(tok.tolist())) == N, "a token is in two units"
    for D, S, u in zip(Ds, Ss, units):
        assert [str(t) for t in D["token"]] == [str(t) for t in S["token"]], f"unit {u['key']}: dump/score token order differ"
    order = np.argsort(tok)
    g = lambda x: x[order]  # noqa: E731
    pdms, sub = g(cat(Ss, "pdms")), g(cat(Ss, "sub"))
    P, LG, PK, FP, LOG = g(cat(Ds, "proposals")), g(cat(Ds, "logits")), g(cat(Ds, "pick")), g(cat(Ds, "fingerprint")), g(cat(Ds, "log"))
    P20, FEAT = g(cat(Ds, "proposals20")), g(cat(Fs, "prehead_features"))
    CID = g(np.concatenate([np.full(len(D["token"]), i) for i, D in enumerate(Ds)]))
    tok = tok[order]
    valid = np.isfinite(sub).all(-1) & np.isfinite(pdms)                       # [N, 64] per cell
    ar = np.arange(N)
    G = {"n_tokens": int(N), "n_columns": 64, "G3_all_valid": bool(valid.all()), "G3_finite": bool(np.isfinite(pdms).all() and np.isfinite(sub).all())}
    # G4a: the planner's own rule recomputed from the table's logits reproduces the dumped pick
    p64, p32 = pick_from_logits(LG, np.float64), pick_from_logits(LG, np.float32)
    G["G4a_pick_from_logits_f64_equal"] = int((p64 == PK).sum())
    G["G4a_pick_from_logits_f32_equal"] = int((p32 == PK).sum())
    # G4b: PDMS == NC*DAC*(5EP+5TTC+2C)/12 on every cell (DDC weight 0 in v1)
    form = sub[..., 0] * sub[..., 1] * (5 * sub[..., 2] + 5 * sub[..., 3] + 2 * sub[..., 4]) / 12.0
    G["G4b_pdms_formula_max_abs_diff"] = float(np.abs(form - pdms).max())
    G["G4b_ddc_in_product_control_max_abs_diff"] = float(np.abs(form * sub[..., 5] - pdms).max())
    G["G4c_pick_valid_cells"] = bool(valid[ar, PK].all())
    G["PASS"] = bool(G["G3_all_valid"] and G["G3_finite"] and G["G4a_pick_from_logits_f64_equal"] == N and G["G4b_pdms_formula_max_abs_diff"] <= 1e-6 and G["G4c_pick_valid_cells"])
    jdump(G, os.path.join(ctx.wd, "gate_table.json"))
    out = f"{ctx.work}/final/table_p2_heldout.npz"
    tmp = out[:-4] + f".tmp{os.getpid()}.npz"
    np.savez_compressed(tmp, token=tok, log=LOG, chunk=CID, fingerprint=FP, pdms=pdms, sub=sub, valid=valid, logits=LG, proposals=P, pick=PK, sub_names=np.array(SUB),
                        pdms_at_pick=pdms[ar, PK])
    os.replace(tmp, out)
    pr = f"{ctx.work}/final/proposals.npz"
    tmp = pr[:-4] + f".tmp{os.getpid()}.npz"
    np.savez_compressed(tmp, token=tok, fingerprint=FP, proposals=P, proposals20=P20, sampling=Ds[0]["sampling"],
                        log=LOG, pick=PK, rule=np.array("navsim_v1"), ckpt=np.array(CKPT))
    os.replace(tmp, pr)
    ft = f"{ctx.work}/final/prehead_features_p2_heldout.npz"
    tmp = ft[:-4] + f".tmp{os.getpid()}.npz"
    np.savez_compressed(tmp, token=tok, log=LOG, prehead_features=FEAT, what=np.array("input of model.score_head [N, 64, 256] fp16 from the same forward as logits"))
    os.replace(tmp, ft)
    ctx.set(stage="table", table_gates=G, table_npz=out)
    return G, out, pr, ft


def sets_index(ctx, tok_to_row, out_path):
    """the 3,137 banked held-out sets -> the P2 row of their token (P2's fan depends on the token only; rank-0 and the goal-augmented rank-1 twin share a row)."""
    keys = PF.heldout_keys()[0]
    rows = [(lg, t, st, rk, tok_to_row.get(t, -1)) for (lg, t, st, rk) in keys]
    np.savez_compressed(out_path, log=np.array([r[0] for r in rows]), token=np.array([r[1] for r in rows]), step=np.array([r[2] for r in rows]), rank=np.array([r[3] for r in rows]),
                        row=np.array([r[4] for r in rows]))
    return {"n_sets": len(rows), "n_sets_with_row": int(sum(r[4] >= 0 for r in rows))}


def anatomy(table_npz):
    T = np.load(table_npz, allow_pickle=False)
    pdms, pick = T["pdms"].astype(float), T["pick"].astype(int)
    ar = np.arange(len(pick))
    at, best = pdms[ar, pick], pdms.max(1)
    return {"n_tokens": int(len(pick)), "mean_pdms_x100": {"P2_pick": float(at.mean() * 100), "best_of_64_oracle": float(best.mean() * 100), "random_selector": float(pdms.mean() * 100)},
            "n_pick_zero": int((at == 0).sum()), "n_best_of_64_zero": int((best == 0).sum()), "n_oracle_strictly_better": int((best > at).sum()),
            "what": "DESCRIPTIVE only, no interval: the plan covers few logs, so a log-cluster interval would be wide and is not claimed here. Nothing is fitted."}


def manifest(files, out_paths):
    man = {}
    for src in files:
        if os.path.exists(src):
            man[os.path.basename(src)] = {"path": src.replace("\\", "/"), "sha256": sha256_file(src), "bytes": os.path.getsize(src)}
    doc = {"written": now(), "what": "P2 held-out 64-proposal table: bulk artifacts + small results; sha256 of each as written", "files": man}
    for p in out_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        jdump(doc, p)
    return man


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
    ap.add_argument("--tag", default="main", help="main = the job; any other tag is a plumbing smoke (own dirs)")
    ap.add_argument("--workers", type=int, default=2, help="concurrent scoring processes (each ~1-2 GB; the RAM guard waits for >= 8 GB free)")
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--plan", default=None, help="a ready plan.json (smoke); default = run the audit")
    ap.add_argument("--chunk-tokens", type=int, default=250)
    ap.add_argument("--fid-logs", type=int, default=3)
    ap.add_argument("--fid-sets", type=int, default=4)
    ap.add_argument("--fid-nav", type=int, default=8)
    ap.add_argument("--dg-k", type=int, default=4)
    ap.add_argument("--limit-units", type=int, default=0, help="smoke: process only the first N units")
    return ap


def main(argv=None):
    a = make_parser().parse_args(argv)
    ctx = Ctx(a)
    wd = ctx.wd
    done, failed = os.path.join(wd, "P2HO_DONE"), os.path.join(wd, "P2HO_FAILED")
    for p in (done, failed):
        if os.path.exists(p):
            os.remove(p)
    print(f"[{now()}] p2ho_driver start tag={a.tag} pid={os.getpid()} workers={a.workers} device={a.device}", flush=True)
    with open(os.path.join(wd, "driver.pid"), "w") as f:
        f.write(str(os.getpid()))
    ctx.set(tag=a.tag, pid=os.getpid(), workers=a.workers, device=a.device, started=ctx.status.get("started", now()), resumed_at=now(), result=None, guard=None)
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
        plan, plan_p = stage_preflight(ctx)
        fid = stage_gate2(ctx)
        units = make_units(plan, a.chunk_tokens)
        if a.limit_units:
            units = units[:a.limit_units]
        ctx.set(n_units=len(units), n_tokens_planned=sum(u["n"] for u in units))
        dg = stage_dgdump(ctx)
        ex = ThreadPoolExecutor(max_workers=max(1, a.workers))
        futs = []
        rc = 0
        try:
            todo = list(units)
            for i, u in enumerate(todo):
                if ctx.abort.is_set():
                    break
                r = ensure_dump(ctx, plan, u, more_to_dump=(i < len(todo) - 1))
                if r != "ok":
                    ctx.abort_reason = f"dump/gate of unit {u['key']} ({u['log']}) failed: {r}"
                    ctx.abort.set()
                    rc = 2 if r == "gate_failed" else 1
                    break
                futs.append((u, ex.submit(score_unit, ctx, plan, u)))
            for u, f in futs:
                if not f.result():
                    ctx.abort_reason = ctx.abort_reason or f"scoring of unit {u['key']} ({u['log']}) failed (logs/score_{u['key']}.log)"
                    rc = rc or 3
        finally:
            C.drop_gpu(ctx)
            ex.shutdown(wait=True, cancel_futures=True)
        if rc:
            return fail(ctx.abort_reason or "pipeline failed", rc)
        G, table_npz, prop_npz, feat_npz = stage_table(ctx, plan, units)
        if not G["PASS"]:
            return fail("table gates failed (gate_table.json): G3 / G4", 4)
        T = np.load(table_npz, allow_pickle=False)
        idx = sets_index(ctx, {str(t): i for i, t in enumerate(T["token"])}, f"{ctx.work}/final/heldout_sets_index.npz")
        an = anatomy(table_npz)
        jdump(an, os.path.join(wd, "anatomy.json"))
        cov = jload(os.path.join(wd, "COVERAGE.json"))
        small = [os.path.join(wd, n) for n in ("disjointness_proof.json", "fidelity.json", "dgdump.json", "gate_table.json", "COVERAGE.json", "anatomy.json", "STATUS.json")]
        bulk = [table_npz, prop_npz, feat_npz, f"{ctx.work}/final/heldout_sets_index.npz"] + [dump_p(ctx, u) for u in units] + [score_p(ctx, u) for u in units]
        man = manifest(bulk + small, [os.path.join(wd, "MANIFEST.sha256.json"), f"{ctx.work}/final/MANIFEST.sha256.json"])
        with open(done, "w", encoding="utf-8") as f:
            json.dump({"when": now(), "n_tokens": G["n_tokens"], "PARTIAL": cov["PARTIAL"], "coverage": {k: cov[k] for k in ("n_tokens_in_plan", "n_logs_in_plan", "n_tokens_banked", "n_logs_banked", "n_tokens_blocked", "blocked_reasons")},
                       "proofs": {"disjointness": "disjointness_proof.json PASS", "fidelity_A": fid["A"]["PASS"], "dgdump": dg.get("PASS"), "G3": G["G3_all_valid"] and G["G3_finite"],
                                  "G4a_pick_from_logits": f"{G['G4a_pick_from_logits_f64_equal']}/{G['n_tokens']}", "G4b_pdms_formula_max_abs_diff": G["G4b_pdms_formula_max_abs_diff"]},
                       "table": table_npz, "proposals": prop_npz, "prehead_features": feat_npz, "sets_index": idx, "manifest": "MANIFEST.sha256.json", "n_files": len(man)}, f, indent=1)
        ctx.set(result="P2HO_DONE")
        print(f"[{now()}] P2HO_DONE", flush=True)
        return 0
    except Stop as e:
        m = str(e)
        return fail(m, int(m.split()[0][2:]) if m.startswith("rc") and m.split()[0][2:].isdigit() else (6 if m.startswith("lowdisk") else 1))
    except BaseException:  # noqa: BLE001
        return fail("EXCEPTION\n" + traceback.format_exc(), 1)
    finally:
        hb.stop_ev.set()


if __name__ == "__main__":
    sys.exit(main())
