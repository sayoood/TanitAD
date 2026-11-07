#!/usr/bin/env python3
"""PDM scoring for the H-SEL-1 confirmation: W3's banked NAVSIM v1.1 harness, UNCHANGED (eval/harness/score_e.py = score_navtest_refe.py with path roots
re-pointed at E:/F:; the P6 reads' runner: cwd eval/harness, RUN_V1_FIX1_ALLOW_DRIFT=1, PYTHONHASHSEED=1 inside W3). CPU only. One harness call scores
one single-proposal seam on one token chunk; chunks are whole logs, <= 1,000 tokens (the token list rides on the Hydra command line).

    smoke    64 columns + 1 RED column on the 6 smoke tokens (from dump/smoke_dump.npz); the 64 must equal C3's table rows BIT FOR BIT
             (PDMS and all six sub-scores), and the RED column (the banked pick's poses shifted +2.0 m in y) must DIFFER on >= 1 token.
    verdict  [refuses unless PREREG_SEL_CONFIRM.md matches its frozen sha256 and CODE_PIN.json matches the code]
             the banked-pick column and the frozen-rule-pick column on every chunk. Gates: banked-pick poses == the banked full-navtest u400 seam
             (<= 1e-3 m, as C3's fidelity gate), the banked column == the banked full-navtest csv BIT FOR BIT (G2), every row valid, and on tokens whose
             rule pick == banked pick the two columns are identical. -> score/verdict_columns.npz
    table    all 64 proposal columns on every chunk (448 calls; resumable) -> score/table_confirm.npz
Retries: a call that does not PASS is retried up to 6 times (waits 120 s x attempt), each start waits for >= 3.5 GiB available RAM and >= 3 GiB free on
C:/E:/F: (bounded 3 h, then the stage fails). Every retry is logged in score/retries.jsonl.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import selc_rule as R  # noqa: E402

PK = "E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
HARNESS = PK + "/eval/harness"
PY_D = "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe"
WORK = "F:/TanitAD_selc_20261107"                     # NTFS: seams + harness run dirs (7 files per call) live here, not on the 1 MiB-cluster E:
C3_TABLE = PK + "/raw/2026-10-11-c3-selection-headroom/table.npz"
BANKED_SEAM = "E:/Projects/TanitAD/data/refe_navtest/seams/stage2_sft4_B_u400_navtest_full.npz"
BANKED_CSV = "E:/Projects/TanitAD/data/refe_navtest/score/refe_stage2_sft4_B_u400_navtest_full/refe_stage2_sft4_B_u400_navtest_full.csv"
SUB = ("no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound", "comfort", "driving_direction_compliance")
M = 64
# harness labels must start with "refe" (score_navtest_refe.py). "refe_selcf" for the real run: W3 also keeps a run dir per LABEL under the shared
# navsim exp/runs/, and the CPU plumbing run of 2026-11-07 already used "refe_selc_*" there -- distinct labels, so no stale file can ever be re-read.
PREFIX = "refe_selcf"
MIN_RAM_GIB, MIN_DISK_GIB = 3.5, 3.0
FIDELITY_TOL_M = 1e-3


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log_to(p, msg):
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def free_gib(drive):
    return shutil.disk_usage(drive + "/").free / 2 ** 30


def ram_gib():
    import psutil
    return psutil.virtual_memory().available / 2 ** 30


def wait_resources(logp, max_s=3 * 3600):
    t0 = time.time()
    n = 0
    while True:
        ok_ram = ram_gib() >= MIN_RAM_GIB
        ok_disk = all(free_gib(d) >= MIN_DISK_GIB for d in ("C:", "E:", "F:"))
        if ok_ram and ok_disk:
            return
        if time.time() - t0 > max_s:
            raise RuntimeError(f"resources not available for {max_s} s: RAM {ram_gib():.1f} GiB, disk C {free_gib('C:'):.1f} E {free_gib('E:'):.1f} F {free_gib('F:'):.1f}")
        if n % 10 == 0:
            log_to(logp, f"waiting for resources: RAM {ram_gib():.1f} GiB (need {MIN_RAM_GIB}), disk C {free_gib('C:'):.1f} E {free_gib('E:'):.1f} F {free_gib('F:'):.1f} (need {MIN_DISK_GIB})")
        n += 1
        time.sleep(30)


def write_seam(p, token, fingerprint, poses, arm):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    if os.path.exists(p):
        with np.load(p, allow_pickle=False) as z:
            if (np.array_equal(z["poses"], poses.astype(np.float32)) and [str(t) for t in z["token"]] == [str(t) for t in token]):
                return
    np.savez(p, token=np.asarray(token), fingerprint=np.asarray(fingerprint), poses=poses.astype(np.float32), sampling=np.array([8, 0.5]), arm=np.array(arm))


def status_of(txt):
    for line in reversed(txt.splitlines()):
        if line.startswith("{") and '"status"' in line:
            try:
                return json.loads(line)
            except ValueError:
                return None
    return None


def read_csv(p):
    rows = {}
    for r in csv.DictReader(open(p, encoding="utf-8")):
        if r["token"] == "average":
            continue
        rows[r["token"]] = (r["valid"] == "True", float(r["score"]), tuple(float(r[c]) for c in SUB))
    return rows


def csv_of(label):
    return f"{WORK}/score/{label}/{label}.csv"


def score_one(job, logdir, retries_p):
    """job = (label, seam, tokens_json, n). Resumable: a PASS log with n valid rows is not redone."""
    label, seam, tj, n = job
    lp = os.path.join(logdir, f"{label}.log")
    st = None
    for att in range(1, 7):
        if os.path.exists(csv_of(label)) and os.path.exists(lp):
            st = status_of(open(lp, encoding="utf-8", errors="replace").read())
            if st and st.get("status") == "PASS" and st.get("csv_valid_rows") == n:
                return label, st
        wait_resources(os.path.join(logdir, "_resources.log"))
        cmd = [PY_D, "score_e.py", "--label", label, "--seam", seam, "--tokens", tj, "--out", f"{WORK}/score"]
        env = dict(os.environ, PYTHONIOENCODING="utf-8", RUN_V1_FIX1_ALLOW_DRIFT="1", CUDA_VISIBLE_DEVICES="-1")
        with open(lp, "w", encoding="utf-8") as f:
            subprocess.call(cmd, cwd=HARNESS, env=env, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        st = status_of(open(lp, encoding="utf-8", errors="replace").read())
        if st and st.get("status") == "PASS" and st.get("csv_valid_rows") == n:
            return label, st
        tail = open(lp, encoding="utf-8", errors="replace").read()[-600:]
        with open(retries_p, "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": now(), "label": label, "attempt": att, "status": st, "tail": tail}) + "\n")
        time.sleep(120 * att)
    return label, st


def run_jobs(jobs, workers, logdir, stage_log):
    os.makedirs(logdir, exist_ok=True)
    retries_p = os.path.join(os.path.dirname(logdir), "retries.jsonl")
    done = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for i, (label, st) in enumerate(ex.map(lambda j: score_one(j, logdir, retries_p), jobs)):
            done[label] = st
            if (i + 1) % 8 == 0 or i + 1 == len(jobs):
                npass = sum(1 for s in done.values() if s and s.get("status") == "PASS")
                log_to(stage_log, f"  {i + 1}/{len(jobs)} calls, {npass} PASS, {time.time() - t0:.0f} s")
    bad = [j[0] for j in jobs if not (done.get(j[0]) and done[j[0]].get("status") == "PASS" and done[j[0]].get("csv_valid_rows") == j[3])]
    return bad


def chunk_files(doc):
    """W3-format token file per chunk (whole logs), on F:."""
    os.makedirs(f"{WORK}/tok", exist_ok=True)
    out = []
    for c in doc["chunks"]:
        toks = sorted(t for t in doc["tokens"] if doc["token_log"][t] in set(c["logs"]))
        assert len(toks) == c["n_tokens"]
        p = f"{WORK}/tok/chunk{c['chunk']}.json"
        body = json.dumps({"tokens": toks, "token_log": {t: doc["token_log"][t] for t in toks}}).encode("utf-8")
        if not (os.path.exists(p) and open(p, "rb").read() == body):
            open(p, "wb").write(body)
        out.append((c["chunk"], toks, p))
    return out


# ----------------------------------------------------------------------------------------------------------------- smoke
def stage_smoke(a):
    out = a.out_dir
    logp = os.path.join(out, "score", "score.log")
    os.makedirs(os.path.join(out, "score"), exist_ok=True)
    S = np.load(os.path.join(out, "dump", "smoke_dump.npz"), allow_pickle=False)
    tok = [str(t) for t in S["token"]]
    sdoc = json.load(open(a.smoke, encoding="utf-8"))
    assert sorted(tok) == sorted(sdoc["tokens"])
    tj = f"{WORK}/tok/smoke.json"
    os.makedirs(os.path.dirname(tj), exist_ok=True)
    open(tj, "wb").write(json.dumps({"tokens": sorted(tok), "token_log": sdoc["token_log"]}).encode("utf-8"))
    P = S["props8"]
    n = len(tok)
    jobs = []
    MM = a.smoke_m
    for k in range(MM):
        lab = f"{PREFIX}_smoke_p{k:02d}"
        sp = f"{WORK}/seams/smoke/{lab}.npz"
        write_seam(sp, S["token"], S["fingerprint"], P[:, k], lab)
        jobs.append((lab, sp, tj, n))
    red = P[np.arange(n), S["pick"]].astype(np.float64).copy()
    red[..., 1] += 2.0
    lab = f"{PREFIX}_smoke_RED"
    sp = f"{WORK}/seams/smoke/{lab}.npz"
    write_seam(sp, S["token"], S["fingerprint"], red, lab)
    jobs.append((lab, sp, tj, n))
    log_to(logp, f"smoke: {len(jobs)} harness calls on {n} tokens")
    bad = run_jobs(jobs, a.workers, os.path.join(out, "score", "logs_smoke"), logp)
    T = np.load(C3_TABLE, allow_pickle=False)
    ti = {str(t): i for i, t in enumerate(T["token"])}
    g = {"n_tokens": n, "calls_failed": bad, "cells": n * MM, "columns": MM}
    if not bad:
        same_p = same_s = 0
        worst = 0.0
        for k in range(MM):
            rows = read_csv(csv_of(f"{PREFIX}_smoke_p{k:02d}"))
            for t in tok:
                v, s, c = rows[t]
                i = ti[t]
                same_p += int(v and s == float(T["pdms"][i, k]))
                same_s += int(all(c[j] == float(T["sub"][i, k, j]) for j in range(6)))
                worst = max(worst, abs(s - float(T["pdms"][i, k])))
        rows = read_csv(csv_of(f"{PREFIX}_smoke_RED"))
        pk = S["pick"]
        red_diff = sum(int(rows[t][1] != float(T["pdms"][ti[t], pk[ii]]) or any(rows[t][2][j] != float(T["sub"][ti[t], pk[ii], j]) for j in range(6)))
                       for ii, t in enumerate(tok))
        g.update(pdms_bit_identical=same_p, sub_bit_identical=same_s, max_abs_pdms_diff=worst, RED_tokens_differing=red_diff)
        g["PASS"] = bool(MM == M and same_p == n * M and same_s == n * M)
        g["RED_FAILS_AS_REQUIRED"] = bool(red_diff >= 1)
    else:
        g["PASS"] = False
        g["RED_FAILS_AS_REQUIRED"] = False
    g["GATE"] = bool(g["PASS"] and g["RED_FAILS_AS_REQUIRED"])
    json.dump(g, open(os.path.join(out, "score", "smoke_score_gate.json"), "w"), indent=1)
    log_to(logp, f"smoke gate: {g}")
    return 0 if g["GATE"] else 2


# ----------------------------------------------------------------------------------------------------------------- pins
def check_pins():
    import selc_analyze as SA
    SA.check_prereg()
    pin_sha = hashlib.sha256(open(os.path.join(HERE, "CODE_PIN.json"), "rb").read()).hexdigest()
    if pin_sha not in open(SA.PREREG, encoding="utf-8").read():
        sys.exit(f"REFUSED: CODE_PIN.json sha256 {pin_sha} is not the one PREREG_SEL_CONFIRM.md freezes")
    pin = json.load(open(os.path.join(HERE, "CODE_PIN.json")))
    bad = []
    for f, h in pin["files"].items():
        got = hashlib.sha256(open(os.path.join(HERE, f), "rb").read()).hexdigest()
        if got != h:
            bad.append(f)
    if bad:
        sys.exit(f"REFUSED: code drifted from CODE_PIN.json: {bad}")


def plumbing_guard(a):
    """--plumbing skips the prereg/code pins ONLY for a CPU plumbing run that can never write into the real package or the real F: work dir."""
    if not a.plumbing:
        check_pins()
        return
    if os.path.abspath(a.out_dir) == os.path.abspath(HERE) or os.path.abspath(WORK) == os.path.abspath("F:/TanitAD_selc_20261107"):
        sys.exit("REFUSED: --plumbing needs --out-dir outside the package and --work outside F:/TanitAD_selc_20261107")


# ----------------------------------------------------------------------------------------------------------------- verdict columns
def stage_verdict(a):
    plumbing_guard(a)
    out = a.out_dir
    logp = os.path.join(out, "score", "score.log")
    D = np.load(os.path.join(out, "dump", "dump_confirm.npz"), allow_pickle=False)
    doc = json.load(open(a.tokens, encoding="utf-8"))
    tok = [str(t) for t in D["token"]]
    assert tok == sorted(doc["tokens"])
    N = len(tok)
    ego = D["ego"].astype(np.float64)
    v4 = R.step_speed4(D["traj"])
    rule = R.rule_vec(D["logits"], v4, ego[:, 6], ego[:, 2])
    base = D["pick"].astype(np.int64)
    # the dump's own pick must be the rule's banked pick (planner._pick == the offline torch argmax): G0 on fresh tokens
    g = {"n": N, "G0_pick_repro_fresh": int((R.banked_pick(D["logits"]) == base).sum())}
    P = D["props8"]
    hr = np.arange(N)
    # fidelity: banked-pick poses vs the banked full-navtest u400 seam
    Bs = np.load(BANKED_SEAM, allow_pickle=False)
    bi = {str(t): i for i, t in enumerate(Bs["token"])}
    ban = Bs["poses"][[bi[t] for t in tok]].astype(np.float64)
    dpos = np.abs(P[hr, base].astype(np.float64) - ban).reshape(N, -1).max(1)
    fp_ok = all(str(D["fingerprint"][i]) == str(Bs["fingerprint"][bi[t]]) for i, t in enumerate(tok))
    g.update(fidelity_max_abs_m=float(dpos.max()), fidelity_n_identical=int((dpos == 0).sum()), fidelity_n_gt_tol=int((dpos > FIDELITY_TOL_M).sum()),
             fingerprints_equal=bool(fp_ok))
    idx = {t: i for i, t in enumerate(tok)}
    jobs = []
    for c, ctoks, tj in chunk_files(doc):
        I = [idx[t] for t in ctoks]
        for arm, pk in (("banked", base), ("rule", rule)):
            lab = f"{PREFIX}_c{c}_{arm}"
            sp = f"{WORK}/seams/verdict/{lab}.npz"
            write_seam(sp, D["token"][I], D["fingerprint"][I], P[I, pk[I]], lab)
            jobs.append((lab, sp, tj, len(I)))
    log_to(logp, f"verdict: {len(jobs)} harness calls on {N} tokens; rule changes {int((rule != base).sum())} picks")
    bad = run_jobs(jobs, a.workers, os.path.join(out, "score", "logs_verdict"), logp)
    g["calls_failed"] = bad
    if bad:
        g["PASS"] = False
        json.dump(g, open(os.path.join(out, "score", "verdict_gates.json"), "w"), indent=1)
        return 3
    pdms = {"banked": np.full(N, np.nan), "rule": np.full(N, np.nan)}
    sub = {"banked": np.full((N, 6), np.nan), "rule": np.full((N, 6), np.nan)}
    valid = {"banked": np.zeros(N, bool), "rule": np.zeros(N, bool)}
    for c, ctoks, tj in chunk_files(doc):
        for arm in ("banked", "rule"):
            rows = read_csv(csv_of(f"{PREFIX}_c{c}_{arm}"))
            for t in ctoks:
                v, s, cc = rows[t]
                pdms[arm][idx[t]], sub[arm][idx[t]], valid[arm][idx[t]] = s, cc, v
    lr = read_csv(BANKED_CSV)
    same = dpos == 0
    dd = np.array([abs(pdms["banked"][i] - lr[t][1]) for i, t in enumerate(tok)])
    unchanged = rule == base
    g.update(all_valid=bool(valid["banked"].all() and valid["rule"].all()), finite=bool(np.isfinite(pdms["banked"]).all() and np.isfinite(pdms["rule"]).all()),
             G2_tokens_identical_poses=int(same.sum()), G2_max_abs_diff_identical_poses=float(dd[same].max()) if same.any() else None,
             G2_n_diff_gt_1e9=int((dd > 1e-9).sum()), G2_max_abs_diff_all=float(dd.max()),
             unchanged_tokens_columns_identical=int((pdms["banked"][unchanged] == pdms["rule"][unchanged]).sum()), n_unchanged=int(unchanged.sum()))
    g["PASS"] = bool(g["G0_pick_repro_fresh"] == N and g["fidelity_n_gt_tol"] == 0 and fp_ok and g["all_valid"] and g["finite"] and same.any()
                     and g["G2_max_abs_diff_identical_poses"] <= 1e-9 and g["unchanged_tokens_columns_identical"] == g["n_unchanged"])
    np.savez(os.path.join(out, "score", "verdict_columns.npz"), token=np.array(tok), pick_banked=base, pick_rule=rule, pdms_banked=pdms["banked"],
             pdms_rule=pdms["rule"], sub_banked=sub["banked"], sub_rule=sub["rule"], valid_banked=valid["banked"], valid_rule=valid["rule"], sub_names=np.array(SUB))
    json.dump(g, open(os.path.join(out, "score", "verdict_gates.json"), "w"), indent=1)
    log_to(logp, f"verdict gates: {g}")
    return 0 if g["PASS"] else 2


# ----------------------------------------------------------------------------------------------------------------- 64-column table
def stage_table(a):
    plumbing_guard(a)
    out = a.out_dir
    logp = os.path.join(out, "score", "score.log")
    D = np.load(os.path.join(out, "dump", "dump_confirm.npz"), allow_pickle=False)
    doc = json.load(open(a.tokens, encoding="utf-8"))
    tok = [str(t) for t in D["token"]]
    N = len(tok)
    idx = {t: i for i, t in enumerate(tok)}
    P = D["props8"]
    jobs = []
    cf = chunk_files(doc)
    for c, ctoks, tj in cf:
        I = [idx[t] for t in ctoks]
        for k in range(M):
            lab = f"{PREFIX}_c{c}_p{k:02d}"
            sp = f"{WORK}/seams/table/{lab}.npz"
            write_seam(sp, D["token"][I], D["fingerprint"][I], P[I, k], lab)
            jobs.append((lab, sp, tj, len(I)))
    log_to(logp, f"table: {len(jobs)} harness calls")
    bad = run_jobs(jobs, a.workers, os.path.join(out, "score", "logs_table"), logp)
    if bad:
        log_to(logp, f"table: {len(bad)} calls failed: {bad[:10]}")
        return 3
    pdms = np.full((N, M), np.nan)
    sub = np.full((N, M, 6), np.nan)
    valid = np.zeros((N, M), bool)
    for c, ctoks, tj in cf:
        for k in range(M):
            rows = read_csv(csv_of(f"{PREFIX}_c{c}_p{k:02d}"))
            for t in ctoks:
                v, s, cc = rows[t]
                pdms[idx[t], k], sub[idx[t], k], valid[idx[t], k] = s, cc, v
    np.savez_compressed(os.path.join(out, "score", "table_confirm.npz"), token=np.array(tok), log=D["log"], pdms=pdms, sub=sub, valid=valid,
                        logits=D["logits"], proposals=P, pick=D["pick"], sub_names=np.array(SUB))
    ok = bool(valid.all() and np.isfinite(pdms).all())
    log_to(logp, f"table assembled: all valid {ok}")
    return 0 if ok else 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("smoke", "verdict", "table"))
    ap.add_argument("--out-dir", default=HERE)
    ap.add_argument("--tokens", default=os.path.join(HERE, "tokens_confirm.json"))
    ap.add_argument("--smoke", default=os.path.join(HERE, "tokens_smoke.json"))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--smoke-m", type=int, default=M, help="smoke columns: 64 (the gate); fewer ONLY for a CPU plumbing run, which can never PASS")
    ap.add_argument("--work", default=None, help="override the F: work dir (plumbing only)")
    ap.add_argument("--plumbing", action="store_true", help="CPU plumbing run: skips the pins, refuses the real package / work dir")
    a = ap.parse_args(argv)
    if a.work:
        global WORK
        WORK = a.work
    return {"smoke": stage_smoke, "verdict": stage_verdict, "table": stage_table}[a.stage](a)


if __name__ == "__main__":
    sys.exit(main())
