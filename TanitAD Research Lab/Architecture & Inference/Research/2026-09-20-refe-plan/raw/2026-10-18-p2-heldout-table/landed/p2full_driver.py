#!/usr/bin/env python3
"""P2 FULL-NAVTEST 64-PROPOSAL TABLE -- the driver (p2tab_driver.py, landed 80775e9 and NOT modified, copied and parametrised: `p2full_driver_vs_p2tab.diff`).

    --tag main    the job: 13 whole-log chunks of the 12,146 navtest tokens.  Per chunk: dump (GPU, behind the shared dev-box lock, released between chunks)
                  -> fidelity gate (STOPS before that chunk is scored if P2's pick is not reproduced; the first failing chunk aborts the job)
                  -> 64 harness columns on a 5-worker pool, overlapped with the next chunk's dump.
                  Then: table (G2/G3) -> SAME-PICK PROOF on all 12,146 -> descriptive anatomy -> cross-check of the 509 zero tokens against the landed
                  509-token table -> the landed SEL/PROP analysis on the zero subset -> sha256 manifest + second copy.  Markers: P2FULL_DONE / P2FULL_FAILED.
    --tag smoke*  the same code on `--tokens` (8 P2 zero tokens, 2 chunks) on the CPU, no lock, `--m` columns, `--pose-tol-m` loosened for CPU-vs-GPU noise.

Resumable: re-run the SAME command.  Done chunks (state/chunk_NN.json, gate PASS + dump sha256) and done harness columns (csv + PASS log) are skipped.
`--from-stage final` re-runs only table + proof + analysis (all columns must exist).

Evidence class of every number written: MEASURED on P2 (own 64-proposal table, full navtest), tier T1 (NAVSIM v1.1 pseudo-simulation of an open-loop plan).
Estimator for every interval: log-cluster bootstrap over the 136 navtest logs, B 10,000, seed 20260927 (zero_tail.Boot) -- it answers "another draw of LOGS"
only; NOT another training run (H-ESTIM-SEED-1: P2 is one seed) and NOT another inference run (P2's inference is deterministic).
NOTHING IS FITTED on these tokens.  This job builds the instrument; a selection rule fit on it must be pre-registered and confirmed on a disjoint set.
"""
from __future__ import annotations

import argparse
import atexit
import ctypes
import glob
import hashlib
import importlib
import json
import os
import shutil
import sys
import threading
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ZT_DIR = os.path.abspath(os.path.join(HERE, "..", "2026-10-15-p2-zero-tail"))           # zero_tail.py, RESULT.json  (landed, unmodified)
LANDED_DIR = os.path.join(ZT_DIR, "p2_own_table")                                        # p2tab_driver.py, table.npz, P2_MEASURED.json (landed, unmodified)
sys.path.insert(0, HERE)
sys.path.insert(0, ZT_DIR)
import c3_run_p2full as C   # noqa: E402
import zero_tail as ZT      # noqa: E402

PDMS_TOL = 1e-9            # "PDMS exact": the banked csv is full precision (G2 compares at 1e-9)
now = C.now
sha256_file = C.sha256_file
jdump = C.atomic_json


# ================================================================================ the same-pick proof (all tokens)
def sig(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def pick_from_logits(logits):
    """planner.REFePlanner.aggregate, rule navsim_v1: sigmoid, NC x DAC x (5 EP + 5 TTC + 2 C) / 12, argmax.  INFORMATIONAL cross-check only."""
    p = sig(logits)
    agg = p[..., 0] * p[..., 1] * (5 * p[..., 2] + 5 * p[..., 3] + 2 * p[..., 4]) / 12.0
    return agg.argmax(-1)


def same_pick_proof(table_npz, out_json, pose_tol_m=C.FIDELITY_TOL_M, expected_tokens=None):
    """Re-derive P2's own pick from the table and require it to EQUAL P2's banked pick, on EVERY token.  Independent equalities per token:
       (1) POSE   the dumped pick's pose == the banked P2 seam row (max |d| <= 1e-3 m; n bit-identical reported -- expected all)
       (2) FPRINT the dump's fingerprint == the banked seam's
       (3) SCORE  the table's PDMS at the pick == the banked P2 csv score (<= 1e-9 = exact; n exact reported)
       (4) SUB    the six sub-scores at the pick == the banked csv's (<= 1e-9)
       plus (informational) argmax of the navsim_v1 aggregate of the dumped logits == the planner's pick.
    The banked seam / csv are read from C.BANKED_SEAM / C.BANKED_CSV at call time."""
    T = np.load(table_npz, allow_pickle=False)
    tok = [str(t) for t in T["token"]]
    N = len(tok)
    P, pick = T["proposals"], T["pick"].astype(int)
    B = np.load(C.BANKED_SEAM)
    bidx = {str(t): i for i, t in enumerate(B["token"])}
    lr = C.read_csv(C.BANKED_CSV)
    in_bank = all(t in bidx and t in lr for t in tok)
    want = set(expected_tokens) if expected_tokens is not None else None
    R = {"evidence": "MEASURED on P2 (dump + harness through the chunked C3 pipeline); tier T1", "n_tokens": N,
         "token_set_equals_expected": bool(want is not None and set(tok) == want and len(set(tok)) == N), "all_tokens_in_banked_seam_and_csv": bool(in_bank),
         "table_all_valid": bool(T["valid"].all())}
    if not in_bank:
        R.update(PASS=False, reading="NOT the same pick -- tokens missing from the banked seam/csv")
        jdump(R, out_json)
        return R
    ban = B["poses"][[bidx[t] for t in tok]].astype(np.float64)
    sel = P[np.arange(N), pick].astype(np.float64)
    dpose = np.abs(sel - ban).reshape(N, -1).max(1)
    fp_ok = np.array([str(T["fingerprint"][i]) == str(B["fingerprint"][bidx[t]]) for i, t in enumerate(tok)])
    pd = T["pdms"][np.arange(N), pick]
    csv_score = np.array([lr[t][1] for t in tok])
    dsc = np.abs(pd - csv_score)
    sub_at = T["sub"][np.arange(N), pick]                                      # [N, 6]
    sub_csv = np.array([lr[t][2] for t in tok])
    dsub = np.abs(sub_at - sub_csv).max(1)
    pl = pick_from_logits(T["logits"])
    worst = np.argsort(-np.maximum(dpose, dsc))[:5]
    R.update(
        pose={"max_abs_m": float(dpose.max()), "n_bit_identical": int((dpose == 0).sum()), "n_gt_1e-4": int((dpose > 1e-4).sum()),
              "n_gt_tol": int((dpose > pose_tol_m).sum()), "tol_m": pose_tol_m},
        fingerprints_equal=bool(fp_ok.all()),
        pdms_at_pick_vs_banked_csv={"max_abs": float(dsc.max()), "n_exact": int((dsc == 0).sum()), "n_gt_tol": int((dsc > PDMS_TOL).sum()), "tol": PDMS_TOL,
                                    "n_banked_csv_exactly_0": int((csv_score == 0.0).sum()), "n_table_pdms_at_pick_exactly_0": int((pd == 0.0).sum()),
                                    "mean_x100_table_at_pick": float(pd.mean() * 100), "mean_x100_banked_csv": float(csv_score.mean() * 100)},
        subscores_at_pick_vs_banked_csv={"max_abs": float(dsub.max()), "n_exact": int((dsub == 0).sum()), "n_gt_tol": int((dsub > PDMS_TOL).sum())},
        pick_from_logits_informational={"n_equal_to_planner_pick": int((pl == pick).sum()), "n": N,
                                        "what": "argmax of sigmoid(logits) navsim_v1 aggregate (float64, numpy) vs the planner's torch float32 pick; not gating"},
        distinct_pick_indices=int(len(set(pick.tolist()))),
        worst_tokens=[(tok[i], float(dpose[i]), float(dsc[i])) for i in worst])
    R["PASS"] = bool(R["token_set_equals_expected"] and R["table_all_valid"] and fp_ok.all() and dpose.max() <= pose_tol_m
                     and dsc.max() <= PDMS_TOL and dsub.max() <= PDMS_TOL)
    R["reading"] = ("P2's own pick is REPRODUCED on every token: the table belongs to P2" if R["PASS"] else
                    "NOT the same pick -- the checkpoint, rule or seam is wrong; no number from this table is admissible")
    jdump(R, out_json)
    return R


# ================================================================================ descriptive anatomy (no fitting)
def anatomy(table_npz, out_json, boot_b=None):
    """Best-of-64 vs P2's pick on ALL tokens, by cause at the pick.  DESCRIPTIVE ONLY: no selection rule is fitted or scored here (a rule study must be
    pre-registered and confirmed on a disjoint set -- this table IS the confirmation set).  CIs: log-cluster bootstrap, B 10,000, seed 20260927."""
    T = np.load(table_npz, allow_pickle=False)
    tok = [str(t) for t in T["token"]]
    N = len(tok)
    pdms, sub, pick = T["pdms"].astype(float), T["sub"].astype(float), T["pick"].astype(int)
    log = np.array([str(x) for x in T["log"]])
    log_names, log_idx = np.unique(log, return_inverse=True)
    boot = ZT.Boot(log_idx, len(log_names), **({"B": boot_b} if boot_b else {}))
    ar = np.arange(N)
    at, best, mean64 = pdms[ar, pick], pdms.max(1), pdms.mean(1)
    nc, dac = sub[ar, pick, 0], sub[ar, pick, 1]
    ones = np.ones(N)
    zero = at == 0.0
    cause = np.where(~zero, "pick_nonzero", np.where((nc == 0) & (dac == 0), "zero_both", np.where(dac == 0, "zero_DAC", np.where(nc == 0, "zero_NC", "zero_other"))))
    ci = lambda num: boot.ratio(num, ones, scale=100.0)  # noqa: E731
    gap = best - at
    R = {"evidence": "MEASURED on P2 (own table, full navtest); tier T1; DESCRIPTIVE, nothing fitted",
         "estimator": {"name": "log-cluster bootstrap (zero_tail.Boot)", "B": boot.B, "seed": ZT.SEED, "n_logs": int(len(log_names)),
                       "answers": "another draw of LOGS only; NOT another training run (H-ESTIM-SEED-1) and NOT another inference run"},
         "n_tokens": N,
         "mean_pdms_x100": {"P2_pick": ci(at), "best_of_64_oracle": ci(best), "random_selector_mean_over_64": ci(mean64), "oracle_minus_pick": ci(gap)},
         "n_best_of_64_zero": int((best == 0).sum()), "n_pick_zero": int(zero.sum()),
         "n_SEL_among_zero_pick_tokens": int((zero & (best > 0)).sum()), "n_PROP_among_zero_pick_tokens": int((zero & (best == 0)).sum()),
         "n_oracle_strictly_better_than_pick": int((gap > 0).sum()),
         "oracle_gap_by_cause_at_pick_pdms_pts": {c: ci(np.where(cause == c, gap, 0.0)) for c in ("zero_both", "zero_DAC", "zero_NC", "zero_other", "pick_nonzero")},
         "n_by_cause_at_pick": {c: int((cause == c).sum()) for c in ("zero_both", "zero_DAC", "zero_NC", "zero_other", "pick_nonzero")},
         "n_proposals_with_pdms_gt_0_per_token": {"mean": float((pdms > 0).sum(1).mean()), "median": float(np.median((pdms > 0).sum(1)))}}
    jdump(R, out_json)
    return R


# ================================================================================ cross-check vs the landed 509-token table
def crosscheck_509(table_npz, out_json, strict=True):
    """The 509 P2 zero tokens were dumped + scored once already (landed 80775e9, table.npz).  The same rows of THIS table must be identical, array by array.
    strict=False (the CPU smoke): only the tokens present in both are compared and the result is informational (a CPU dump is not the GPU dump)."""
    lp = os.path.join(LANDED_DIR, "table.npz")
    if not os.path.exists(lp):
        R = {"PASS": None, "reason": f"{lp} not present"}
        jdump(R, out_json)
        return R
    T, L = np.load(table_npz, allow_pickle=False), np.load(lp, allow_pickle=False)
    pos = {str(t): i for i, t in enumerate(T["token"])}
    lpos = [(pos[str(t)], j) for j, t in enumerate(L["token"]) if str(t) in pos]
    idx, jdx = np.array([a for a, _ in lpos], dtype=int), np.array([b for _, b in lpos], dtype=int)
    R = {"n_landed_tokens": int(len(L["token"])), "n_compared": int(len(idx)), "all_in_full_table": bool(len(idx) == len(L["token"])), "strict": strict}
    ok = R["all_in_full_table"] if strict else len(idx) > 0
    for key in ("pdms", "sub", "valid", "logits", "proposals", "pick"):
        a, b = T[key][idx], L[key][jdx]
        eq = bool(np.array_equal(a, b))
        R[key] = {"identical": eq, "max_abs_diff": float(np.abs(a.astype(float) - b.astype(float)).max()) if a.shape == b.shape else None}
        ok = ok and (eq or not strict)
    R["PASS"] = bool(ok) if strict else None
    R["reading"] = ("the 509 zero-token rows of the full table are bit-identical to the landed 509-token table" if ok and strict
                    else "INFORMATIONAL (smoke: CPU dump vs the landed GPU table)" if not strict
                    else "DIFFERENT from the landed 509-token table -- read per-key max_abs_diff; the dump or the harness is not deterministic across runs")
    jdump(R, out_json)
    return R


def landed_driver():
    sys.path.insert(0, LANDED_DIR)
    return importlib.import_module("p2tab_driver")


def zero_subset_measured(table_npz, proof, wd, tmp_dir):
    """The landed SEL/PROP analysis (p2tab_driver.measured -> zero_tail), UNCHANGED, on the zero-PDMS rows of the full table (the tokens whose BANKED csv score is
    exactly 0.0 -- the proof has shown the table's PDMS at the pick equals it); compared key by key with the landed 509-token reading."""
    LD = landed_driver()
    T = np.load(table_npz, allow_pickle=False)
    sel = T["pdms_at_pick_banked"] == 0.0
    os.makedirs(tmp_dir, exist_ok=True)
    np.savez_compressed(os.path.join(tmp_dir, "table.npz"), **{k: (T[k][sel] if T[k].shape[:1] == sel.shape else T[k]) for k in T.files})
    pr = dict(proof, n_tokens=int(sel.sum()), note="the proof of the FULL table, handed to the landed measured() for its meta block only")
    R = LD.measured(tmp_dir, os.path.join(wd, "P2_MEASURED_from_full.json"), os.path.join(wd, "p2_zero_tokens_sel_prop_from_full.csv"), pr)
    landed = C.jload(os.path.join(LANDED_DIR, "P2_MEASURED.json"))["reading"]
    same = {k: (R["reading"]["P2_MEASURED"][k] == landed["P2_MEASURED"][k]) for k in ("n_zero", "n_SEL", "n_PROP", "SEL_share", "oracle_on_SEL_ceiling_pdms_pts")}
    cmp_ = {"landed_509_reading_P2_MEASURED": landed["P2_MEASURED"], "from_full_table_P2_MEASURED": R["reading"]["P2_MEASURED"], "equal_per_key": same,
            "PASS": bool(all(same.values()))}
    jdump(cmp_, os.path.join(wd, "zero_subset_vs_landed.json"))
    return cmp_


def json_tokens(p):
    return C.jload(p)["tokens"]


# ================================================================================ markers + manifest
def manifest(ctx, files, out_paths):
    man = {}
    for src in files:
        if os.path.exists(src):
            man[os.path.basename(src)] = {"path": src.replace("\\", "/"), "sha256": sha256_file(src), "bytes": os.path.getsize(src)}
    doc = {"written": now(), "what": "P2 full-navtest 64-proposal table: bulk artifacts + small results; sha256 of each as written", "files": man}
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


def main(argv=None):
    ap = C.make_parser()
    ap.add_argument("--from-stage", default="all", choices=("all", "final"), help="final = skip the dump/score pipeline; redo table + proof + analysis")
    ap.set_defaults(stage="all")
    a = ap.parse_args(argv)
    ctx = C.Ctx(a)
    wd = ctx.wd
    done, failed = os.path.join(wd, "P2FULL_DONE"), os.path.join(wd, "P2FULL_FAILED")
    for p in (done, failed):
        if os.path.exists(p):
            os.remove(p)
    print(f"[{now()}] p2full_driver start tag={a.tag} pid={os.getpid()} workers={a.workers} device={a.device}", flush=True)
    with open(os.path.join(wd, "driver.pid"), "w") as f:
        f.write(str(os.getpid()))
    ctx.set(tag=a.tag, n_tokens=sum(c["n"] for c in ctx.chunks), n_chunks=len(ctx.chunks), pid=os.getpid(), workers=a.workers,
            started=ctx.status.get("started", now()), resumed_at=now(), result=None)
    hb = Heartbeat(ctx)
    hb.start()
    atexit.register(C.drop_gpu, ctx)          # never leave the shared lock behind on a normal interpreter exit
    if os.name == "nt":                       # an overnight job must not be stalled by idle sleep: ES_CONTINUOUS | ES_SYSTEM_REQUIRED, released when this process exits
        try:                                  # (a per-thread execution-state request, not a power SETTING; the display may still turn off)
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
            ctx.set(keep_awake="requested (ES_SYSTEM_REQUIRED) by the driver main thread")
        except Exception:  # noqa: BLE001
            ctx.set(keep_awake="request FAILED")

    def fail(msg):
        with open(failed, "w", encoding="utf-8") as f:
            f.write(f"{now()} {msg}\n--- STATUS.json ---\n")
            f.write(C._read_text(ctx.status_p) if os.path.exists(ctx.status_p) else "(none)")
        ctx.set(result=f"FAILED: {msg[:300]}")
        print(f"[{now()}] FAILED: {msg}", flush=True)
        return 1

    try:
        if a.from_stage == "all":
            rc = C.run_pipeline(ctx)
            if rc == 2:                       # the brief: same_pick_proof.json must exist -- here it records the failing per-chunk gates
                gates = [C.jload(g) for g in sorted(glob.glob(os.path.join(wd, "state", "gate_fidelity_c*.json")))]
                jdump({"PASS": False, "stopped_before_scoring_the_failing_chunk": True, "per_chunk_gates": gates,
                       "reading": "NOT the same pick -- the dump does not reproduce P2's banked plan; nothing from the failing chunk was scored"},
                      os.path.join(wd, "same_pick_proof.json"))
            if rc:
                return fail(f"pipeline rc={rc} (1=dump/preflight, 2=FIDELITY gate = P2's pick not reproduced -- report it, do NOT re-run, 3=score, "
                            f"4=table G2/G3, 5=same-pick proof): {ctx.abort_reason}")
        table_npz = f"{ctx.bulk}/table_p2_full.npz"
        if not C.stage_table(ctx, table_npz):
            return fail("table gates failed (gate_table.json): rc=4")
        if ctx.main:
            expected = json_tokens(C.TOKENS_FULL)
        else:
            expected = [t for ch in ctx.chunks for t in ctx.tokens_of(ch)["tokens"]]
        proof = same_pick_proof(table_npz, os.path.join(wd, "same_pick_proof.json"), pose_tol_m=ctx.pose_tol_m, expected_tokens=expected)
        ctx.set(stage="proof", same_pick_proof={k: proof.get(k) for k in ("PASS", "n_tokens", "reading")})
        print(json.dumps({k: proof.get(k) for k in ("PASS", "pose", "pdms_at_pick_vs_banked_csv", "subscores_at_pick_vs_banked_csv")}, indent=1), flush=True)
        if not proof["PASS"]:
            return fail("SAME-PICK PROOF FAILED (rc=5): " + proof["reading"])
        secondary, sec_err = {}, {}
        for name, fn in (("anatomy", lambda: anatomy(table_npz, os.path.join(wd, "anatomy.json"))),
                         ("crosscheck_509", lambda: crosscheck_509(table_npz, os.path.join(wd, "crosscheck_509.json"), strict=ctx.main)),
                         ("zero_subset_vs_landed", lambda: zero_subset_measured(table_npz, proof, wd, os.path.join(ctx.work, "zero_subset")) if ctx.main else None)):
            try:
                r = fn()
                secondary[name] = (r.get("PASS") if isinstance(r, dict) and "PASS" in r else "written") if r is not None else "skipped (smoke)"
            except BaseException:  # noqa: BLE001
                sec_err[name] = traceback.format_exc()
                secondary[name] = "ERROR"
        ctx.set(stage="analysis", secondary=secondary, secondary_errors=sec_err or None)
        if sec_err:
            jdump(sec_err, os.path.join(wd, "secondary_errors.json"))
        # durable copies: the table on C: as well (D: is an external exFAT drive), the sha256 manifest in the package and on both drives
        second = f"{ctx.work}/final/table_p2_full.npz"
        os.makedirs(os.path.dirname(second), exist_ok=True)
        shutil.copy2(table_npz, second)
        assert sha256_file(second) == sha256_file(table_npz), "the C: copy of the table differs from the D: copy"
        small = [os.path.join(wd, n) for n in ("same_pick_proof.json", "anatomy.json", "crosscheck_509.json", "P2_MEASURED_from_full.json", "zero_subset_vs_landed.json",
                                                "p2_zero_tokens_sel_prop_from_full.csv", "gate_table.json", "STATUS.json")]
        dumps = [ctx.dump_path(c["id"]) for c in ctx.chunks]
        man = manifest(ctx, [table_npz, second] + dumps + small + [os.path.join(wd, "chunks", "chunks.json")],
                       [os.path.join(wd, "MANIFEST.sha256.json"), f"{ctx.bulk}/MANIFEST.sha256.json"])
        with open(done, "w", encoding="utf-8") as f:
            json.dump({"when": now(), "n_tokens": proof["n_tokens"], "same_pick_proof": proof["reading"], "secondary": secondary,
                       "table_primary": table_npz, "table_second_copy": second, "manifest": "MANIFEST.sha256.json", "n_files": len(man)}, f, indent=1)
        ctx.set(result="P2FULL_DONE")
        print(f"[{now()}] P2FULL_DONE", flush=True)
        return 0
    except BaseException:  # noqa: BLE001
        return fail("EXCEPTION\n" + traceback.format_exc())
    finally:
        hb.stop_ev.set()


if __name__ == "__main__":
    sys.exit(main())
