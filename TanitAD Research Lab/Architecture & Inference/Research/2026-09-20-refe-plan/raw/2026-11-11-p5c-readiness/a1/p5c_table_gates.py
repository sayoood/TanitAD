#!/usr/bin/env python3
"""P5-C label-table gates TB-0 .. TB-7 (PREREG_P5C.md section 4.2, frozen) and the section 4.3 rank-1 policy.  CPU only; stdlib + numpy (no scipy).

Inputs
  --bank PATH...      bank rows: files (.jsonl / .jsonl.gz), directories (their targets_rank*.jsonl), or globs.  Row schema = the P7 bank (log_name, token, rank, traj [20][3], sim_hz).
  --labels GLOB...    labeller shards (p5c_labels.py `label` output): {"log","token","rank","traj_sha256","tc20s":[20][3]} or {...,"error"}.
  --control GLOB...   TB-0 control shards (the dev-box build, same schema).        omitted => TB-0 NOT_RUN
  --tb1 FILE          TB-1 output of `p5c_labels.py control --n 1018`.              omitted => TB-1 NOT_RUN
  --repair0/--repair1 repair_rank{0,1}.json (per-row sim8 raw vs Tc20s) for the 4.3 policy.
  --identity-table F  optional: a `traj_label_table.py identity` table, checked bit-for-bit against the bank (TB-3).
  --expect-rows N     the bank size the table must cover (150603 on the pod); default = the bank given.
Outputs (--out-dir): gates.json, merged_labels_all.jsonl, merged_labels_rank0only.jsonl, table_meta_fields.json, and a last line ZZTB_ALL_PASS | ZZTB_FAIL <ids>.
A gate that could not run is NOT_RUN and counts as not passed (it is listed in the ZZTB_FAIL line).
"""
import argparse
import glob
import gzip
import hashlib
import json
import math
import os
import platform
import sys
from collections import Counter

import numpy as np

DT = 0.2                      # P5 section 3 learnability() grid
B_BOOT, SEED_BOOT = 10_000, 20260927
BAR_KNOT = 1e-9               # TB-2
BAR_LEARN = 2.0               # TB-6
BAR_FALLBACK = 0.01           # TB-4
BAR_N_TB1 = 1018              # TB-1
MIN_RK1_ROWS, MIN_RK1_LOGS = 200, 5
KNOT_FAIL_MSG = "control/finite/shape check failed"     # p5c_labels.py: raised when ctl["knot_exact"] is False (knot dev > 1e-9), or not finite, or not [20,3]
GATE_IDS = ["TB-0", "TB-1", "TB-2", "TB-3", "TB-4", "TB-5", "TB-6", "TB-7"]
TC20S_TOL = 1e-9              # MM ruling R-A (2026-11-12, A1 section U): Tc20s TB-0 / TB-1 tolerance via --tc20s-tol (CLI default). 0.0 reproduces the frozen PREREG_P5C behaviour exactly (function default)


# ------------------------------------------------------------------ keys / io
def traj_sha(traj):
    """sha256 of the [20,3] traj as float64 little-endian bytes (== p5c_labels.traj_sha == traj_label_table.traj_sha256)."""
    return hashlib.sha256(np.ascontiguousarray(np.asarray(traj, np.float64)).tobytes()).hexdigest()


def git_blob_sha1(path):
    b = open(path, "rb").read()
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _open(p):
    return gzip.open(p, "rt", encoding="utf-8") if p.endswith(".gz") else open(p, encoding="utf-8")


def expand(paths, bank_dirs=False):
    out = []
    for p in paths:
        if os.path.isdir(p):
            out += sorted(glob.glob(os.path.join(p, "targets_rank*.jsonl"))) if bank_dirs else sorted(glob.glob(os.path.join(p, "*.jsonl")))
        else:
            g = sorted(glob.glob(p))
            out += g if g else ([p] if os.path.exists(p) else [])
    return out


def load_bank(paths):
    """-> dict of parallel lists/arrays over the bank rows, in file order.  Keeps only log, token, rank, traj, sim_hz."""
    files = expand(paths, bank_dirs=True)
    if not files:
        raise SystemExit(f"no bank rows under {paths}")
    logs, toks, ranks, trajs, hz = [], [], [], [], []
    for p in files:
        with _open(p) as f:
            for ln in f:
                if not ln.strip():
                    continue
                r = json.loads(ln)
                logs.append(r["log_name"]); toks.append(r["token"]); ranks.append(int(r["rank"]))
                trajs.append(np.asarray(r["traj"], np.float64)); hz.append(r.get("sim_hz"))
    return {"log": logs, "token": toks, "rank": np.asarray(ranks, np.int64), "traj": np.stack(trajs), "sim_hz": hz,
            "sha": [traj_sha(t) for t in trajs], "files": files}


def triple(lg, tok, rk):
    return f"{lg}|{tok}|{int(rk)}"


def load_label_recs(globs):
    """-> (recs {triple: rec}, n_records, n_duplicate_keys).  Duplicate triples keep the FIRST and are counted (TB-4 refuses on any)."""
    recs, n, dup = {}, 0, 0
    for p in expand(globs):
        with _open(p) as f:
            for ln in f:
                if not ln.strip():
                    continue
                d = json.loads(ln)
                n += 1
                t = triple(d["log"], d["token"], d["rank"])
                if t in recs:
                    dup += 1
                    continue
                recs[t] = d
    return recs, n, dup


# ------------------------------------------------------------------ merge (section 4.1 raw-fallback rule)
def merge(bank, recs, n_dup=0):
    """Join label records onto bank rows.  source per row: tc20s | raw_fallback (record errored / malformed: label := traj) | absent (no record at all: NOT in the table)."""
    N = len(bank["log"])
    label = np.array(bank["traj"], copy=True)
    source = np.empty(N, dtype=object)
    sha_mismatch = np.zeros(N, bool)
    knot_fail = np.zeros(N, bool)
    knot_dev = np.full(N, np.nan)
    errs = Counter()
    keys = set()
    for i in range(N):
        t = triple(bank["log"][i], bank["token"][i], bank["rank"][i])
        keys.add(t)
        d = recs.get(t)
        if d is None:
            source[i] = "absent"
            continue
        if d.get("traj_sha256") is not None and d["traj_sha256"] != bank["sha"][i]:
            sha_mismatch[i] = True
            source[i] = "absent"            # a label computed from another traj never enters the table
            continue
        if "knot_maxdev" in d:
            knot_dev[i] = float(d["knot_maxdev"])
        tc = d.get("tc20s")
        ok = tc is not None and "error" not in d
        if ok:
            a = np.asarray(tc, np.float64)
            if a.shape != (20, 3):
                ok = False
                errs["malformed_shape"] += 1
            else:
                label[i] = a
                source[i] = "tc20s"
                continue
        source[i] = "raw_fallback"
        e = str(d.get("error", "malformed"))
        errs[e.split(":")[0][:60] if e != KNOT_FAIL_MSG else KNOT_FAIL_MSG] += 1
        if e == KNOT_FAIL_MSG:
            knot_fail[i] = True
    extra = [t for t in recs if t not in keys]
    return {"label": label, "source": source, "sha_mismatch": sha_mismatch, "knot_fail": knot_fail, "knot_dev": knot_dev, "errs": errs,
            "extra_keys": extra, "n_dup": n_dup, "n_records": len(recs)}


def _res(gid, status, name, bar, value=None, n=None, detail=None, evidence="MEASURED"):
    return {"id": gid, "name": name, "status": status, "bar": bar, "value": value, "n": n, "evidence": evidence, "detail": detail or {}}


def learn(q):
    """P5 section 3 learnability(), vectorised over rows: per-row (max |acc| m/s^2, max |heading rate| deg/s), origin prepended at t=0."""
    q = np.asarray(q, np.float64)
    Q = np.concatenate([np.zeros((q.shape[0], 1, 3)), q], 1)
    acc = np.linalg.norm(np.diff(Q[:, :, :2], n=2, axis=1), axis=2) / DT ** 2
    hr = np.abs(np.diff(np.unwrap(Q[:, :, 2], axis=1), axis=1)) / DT
    return acc.max(1), np.degrees(hr.max(1))


# ------------------------------------------------------------------ gates
def gate_tb0(bank, M, control_recs, tol=0.0, recs=None):
    """tol == 0.0 is the frozen PREREG_P5C gate (max abs diff == 0.0).  tol > 0 is A1 section U / MM ruling R-A (`--tc20s-tol`, CLI default 1e-9): max abs diff <= tol (m, rad) on every
    common row AND sim8 verdict identical on every common row where BOTH sides carry one (a `tc20s_sim8` field on the control record and on the table record `recs`)."""
    bar = ("max abs diff == 0.0 on every common row; every control-ok row present and ok in the table build" if tol == 0.0 else
           f"R-A: max abs diff <= {tol:g} (m, rad) on every common row AND sim8 verdict identical on every common row where sim8 is available; every control-ok row present and ok in the table build")
    if control_recs is None:
        return _res("TB-0", "NOT_RUN", "cross-environment bit-identity", bar, detail={"reason": "no --control shards given"})
    idx = {triple(bank["log"][i], bank["token"][i], bank["rank"][i]): i for i in range(len(bank["log"]))}
    n_common = n_missing = n_sha = n_sim8_cmp = n_sim8_diff = 0
    maxd = 0.0
    logs = set()
    for t, d in control_recs.items():
        i = idx.get(t)
        if i is None or "tc20s" not in d or "error" in d:
            continue                                  # not a control-ok row of THIS bank
        logs.add(bank["log"][i])
        if d.get("traj_sha256") not in (None, bank["sha"][i]):
            n_sha += 1
            continue
        if M["source"][i] != "tc20s":
            n_missing += 1
            continue
        a = np.asarray(d["tc20s"], np.float64)
        diff = float(np.abs(M["label"][i] - a).max()) if a.shape == (20, 3) else math.inf
        maxd = max(maxd, diff if not math.isnan(diff) else math.inf)
        n_common += 1
        s_c, s_t = d.get("tc20s_sim8"), (recs or {}).get(t, {}).get("tc20s_sim8")
        if s_c is not None and s_t is not None:
            n_sim8_cmp += 1
            n_sim8_diff += int(bool(s_c) != bool(s_t))
    ok = n_common > 0 and n_missing == 0 and n_sha == 0 and maxd <= tol and n_sim8_diff == 0
    return _res("TB-0", "PASS" if ok else "FAIL", "cross-environment bit-identity" if tol == 0.0 else "cross-environment identity (Tc20s, R-A)", bar, value=maxd, n=n_common,
                detail={"n_common_rows": n_common, "n_control_logs": len(logs), "n_control_ok_missing_in_table": n_missing, "n_traj_sha_mismatch": n_sha, "tol": tol,
                        "n_sim8_compared": n_sim8_cmp, "n_sim8_verdict_differs": n_sim8_diff})


def gate_tb1(tb1, expect_n=BAR_N_TB1, tol=0.0):
    """tol == 0.0: the frozen gate.  tol > 0 (R-A): n == expect_n, n_errors == 0, max_abs_diff <= tol (bit_identical is then recorded, not required)."""
    bar = (f"bit_identical == n == {expect_n}, max_abs_diff == 0.0, n_errors == 0" if tol == 0.0 else f"R-A: n == {expect_n}, max_abs_diff <= {tol:g} (m, rad), n_errors == 0 (bit_identical recorded)")
    if tb1 is None:
        return _res("TB-1", "NOT_RUN", "reproduction of p5s_cubic.npz", bar, detail={"reason": "no --tb1 file given"})
    mad = tb1.get("max_abs_diff")
    if tol == 0.0:
        ok = tb1.get("n") == expect_n and tb1.get("bit_identical") == expect_n and mad == 0.0 and tb1.get("n_errors", 1) == 0
    else:
        ok = tb1.get("n") == expect_n and isinstance(mad, (int, float)) and math.isfinite(mad) and mad <= tol and tb1.get("n_errors", 1) == 0
    return _res("TB-1", "PASS" if ok else "FAIL", "reproduction of p5s_cubic.npz", bar, value=f"{tb1.get('bit_identical')}/{tb1.get('n')}", n=tb1.get("n"),
                detail={**{k: tb1.get(k) for k in ("max_abs_diff", "n_errors", "seconds", "errors", "bit_identical")}, "tol": tol})


def gate_tb2(bank, M):
    bar = f"knot deviation <= {BAR_KNOT:g} on every row (no row failed the driver's knot control)"
    ranks = bank["rank"]
    nk = int(M["knot_fail"].sum())
    dev = M["knot_dev"]
    have = ~np.isnan(dev)
    dev_bad = int((dev[have] > BAR_KNOT).sum())
    n_ok = int((M["source"] == "tc20s").sum())
    per_rank = {int(k): {"n_tc20s": int(((M["source"] == "tc20s") & (ranks == k)).sum()), "knot_fail": int((M["knot_fail"] & (ranks == k)).sum()),
                         "knot_dev_bad": int(((dev > BAR_KNOT) & have & (ranks == k)).sum())} for k in (0, 1)}
    ok = nk == 0 and dev_bad == 0 and n_ok > 0
    return _res("TB-2", "PASS" if ok else "FAIL", "knot exactness C2'", bar, value=(float(np.nanmax(dev)) if have.any() else None), n=n_ok,
                evidence="MEASURED-INDIRECT" if not have.any() else "MEASURED",
                detail={"n_knot_control_failures": nk, "n_rows_with_knot_maxdev_field": int(have.sum()), "n_knot_maxdev_over_bar": dev_bad, "per_rank": per_rank,
                        "note": ("the driver does not persist the per-row knot deviation; its control raises (=> error record => raw_fallback) when dev > 1e-9, so "
                                 "'no knot-control failure among the ok rows' is the available evidence" if not have.any() else "per-row knot_maxdev read")})


def ident_check(bank, ident_labels=None):
    """TB-3. ident_labels: [N,20,3] (default: the identity table's label, i.e. the traj through a JSON round trip, which is what `traj_label_table.py identity` stores)."""
    if ident_labels is None:
        ident_labels = np.stack([np.asarray(json.loads(json.dumps([[float(v) for v in p] for p in t])), np.float64) for t in bank["traj"]])
    same = np.array([a.tobytes() == b.tobytes() for a, b in zip(np.ascontiguousarray(ident_labels), np.ascontiguousarray(bank["traj"]))])
    return int(same.sum()), len(same)


def gate_tb3(bank, ident_labels=None):
    bar = "ident == traj bit-for-bit on 100 % of rows"
    n_same, n = ident_check(bank, ident_labels)
    return _res("TB-3", "PASS" if n_same == n and n > 0 else "FAIL", "identity control", bar, value=f"{n_same}/{n}", n=n,
                detail={"note": "the driver's own `ident` row (sign 0.0) is not persisted; this checks the identity TABLE (label := traj through the table's JSON encoding) against the bank bytes"})


def table_keys(bank, M):
    return {f"{triple(bank['log'][i], bank['token'][i], bank['rank'][i])}|{bank['sha'][i]}" for i in range(len(bank["log"])) if M["source"][i] != "absent"}


def gate_tb4(bank, M, expect_rows=None):
    bar = "table keys == bank keys (all of them, none extra, no duplicate); n_raw_fallback <= 1.0 % of rows"
    N = len(bank["log"])
    want = expect_rows if expect_rows is not None else N
    bank_keys = {f"{triple(bank['log'][i], bank['token'][i], bank['rank'][i])}|{bank['sha'][i]}" for i in range(N)}
    tk = table_keys(bank, M)
    n_absent = int((M["source"] == "absent").sum())
    n_fb = int((M["source"] == "raw_fallback").sum())
    frac = n_fb / N if N else 1.0
    ok = (N == want and len(bank_keys) == N and tk == bank_keys and n_absent == 0 and not M["extra_keys"] and M["n_dup"] == 0
          and not M["sha_mismatch"].any() and frac <= BAR_FALLBACK)
    return _res("TB-4", "PASS" if ok else "FAIL", "coverage", bar, value=f"{len(tk & bank_keys)}/{want}", n=N,
                detail={"bank_rows": N, "expected_rows": want, "unique_bank_keys": len(bank_keys), "table_keys": len(tk), "absent": n_absent, "extra_label_records": len(M["extra_keys"]),
                        "duplicate_label_records": M["n_dup"], "traj_sha_mismatch": int(M["sha_mismatch"].sum()), "n_raw_fallback": n_fb, "raw_fallback_frac": frac,
                        "fallback_causes": dict(M["errs"]), "REFUSE_if_fallback_over_1pct": frac > BAR_FALLBACK})


def gate_tb5(bank, M):
    bar = "every label finite; |dyaw vs traj| < pi; sim_hz == 10 on every row (100 %)"
    rows = np.where(M["source"] != "absent")[0]
    L, T = M["label"][rows], bank["traj"][rows]
    finite = np.isfinite(L).all(axis=(1, 2))
    with np.errstate(invalid="ignore"):
        dyaw = np.abs(L[:, :, 2] - T[:, :, 2]).max(1)
        yaw_ok = np.where(finite, dyaw < math.pi, False)
    hz_ok = np.array([bank["sim_hz"][i] == 10 for i in rows])
    allok = finite & yaw_ok & hz_ok
    n = len(rows)
    return _res("TB-5", "PASS" if n > 0 and allok.all() else "FAIL", "sanity", bar, value=f"{int(allok.sum())}/{n}", n=n,
                detail={"non_finite": int((~finite).sum()), "yaw_branch_ge_pi": int((finite & ~yaw_ok).sum()), "sim_hz_not_10": int((~hz_ok).sum()),
                        "max_abs_dyaw": float(np.nanmax(np.where(finite, dyaw, np.nan))) if finite.any() else None})


def tb6_rank(bank, M, k, src="tc20s"):
    sel = np.where((M["source"] == src) & (bank["rank"] == k))[0]
    if len(sel) == 0:
        return {"rank": k, "n": 0, "pass": False, "reason": f"no {src} rows at this rank"}
    la, lh = learn(M["label"][sel])
    ta, th = learn(bank["traj"][sel])
    ma, mh, mta, mth = (float(np.median(x)) for x in (la, lh, ta, th))
    ra = ma / mta if mta > 0 else math.inf
    rh = mh / mth if mth > 0 else math.inf
    with np.errstate(divide="ignore", invalid="ignore"):
        mra, mrh = float(np.nanmedian(la / ta)), float(np.nanmedian(lh / th))      # DIAGNOSTIC only (median of per-row ratios); the bar reads the ratio of medians
    return {"rank": k, "n": int(len(sel)), "diag_median_of_row_ratios_acc": mra, "diag_median_of_row_ratios_hr": mrh, "median_max_acc_label": ma, "median_max_acc_traj": mta, "ratio_acc": ra,
            "median_max_hr_label_deg_s": mh, "median_max_hr_traj_deg_s": mth, "ratio_hr": rh, "pass": bool(ra <= BAR_LEARN and rh <= BAR_LEARN)}


def gate_tb6(bank, M, src="tc20s"):
    bar = f"per rank: median per-row max|acc| and max|heading rate| of labels / same for traj <= {BAR_LEARN:g}x on both. Rank-0 FAIL => STOP (P5-R); rank-1 FAIL => rank-0 only"
    r0, r1 = tb6_rank(bank, M, 0, src), tb6_rank(bank, M, 1, src)
    stop = not r0["pass"]
    return _res("TB-6", "PASS" if (r0["pass"] and r1["pass"]) else "FAIL", "learnability per rank", bar,
                value={"rank0": [r0.get("ratio_acc"), r0.get("ratio_hr")], "rank1": [r1.get("ratio_acc"), r1.get("ratio_hr")]}, n=r0["n"] + r1["n"],
                detail={"rank0": r0, "rank1": r1, "rank0_fail_STOP_next_lever_P5R": stop, "rank1_fail_means_rank0_only": not r1["pass"],
                        "population": f"rows whose label source is {src} (raw_fallback rows are identity and excluded)"})


def gate_tb7(bank, M, src="tc20s"):
    bar = "n_differs per rank recorded"
    per = {}
    for k in (0, 1):
        sel = (M["source"] == src) & (bank["rank"] == k)
        idx = np.where(sel)[0]
        nd = int(sum(M["label"][i].tobytes() != bank["traj"][i].tobytes() for i in idx))
        per[k] = {"n_rows": int((bank["rank"] == k).sum()), "n_tc20s": int(len(idx)), "n_raw_fallback": int(((M["source"] == "raw_fallback") & (bank["rank"] == k)).sum()), "n_differs": nd}
    ok = all(per[k]["n_tc20s"] > 0 for k in (0, 1))
    return _res("TB-7", "PASS" if ok else "FAIL", "n_differs recorded", bar, value={str(k): per[k]["n_differs"] for k in per}, n=sum(p["n_tc20s"] for p in per.values()),
                detail={"per_rank": per, "feeds": "G-LIVE (a): n_differs must be > 0 for the substitution to be non-trivial",
                        "warn_zero_differs": [k for k in per if per[k]["n_tc20s"] > 0 and per[k]["n_differs"] == 0]})


# ------------------------------------------------------------------ section 4.3 rank-1 policy
def paired_boot(rows, B=B_BOOT, seed=SEED_BOOT):
    """P5 section 3 paired log-cluster bootstrap (boot_weights + paired_ci, ratio estimator): raw - tc20s sim8-fail rate, in pp.  -> point, lo, hi."""
    logs = [r["log"] for r in rows]
    raw = np.array([r["raw_sim8"] for r in rows], np.float64)
    tc = np.array([r["tc20s_sim8"] for r in rows], np.float64)
    u = sorted(set(logs))
    ui = {g: i for i, g in enumerate(u)}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(u), size=(B, len(u)))
    W = np.stack([np.bincount(i, minlength=len(u)) for i in idx]).astype(np.float64)
    w = W[:, np.array([ui[g] for g in logs])]
    bs = (w @ raw - w @ tc) / w.sum(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"n": len(rows), "n_logs": len(u), "point_pp": 100 * float(raw.mean() - tc.mean()), "lo_pp": 100 * float(lo), "hi_pp": 100 * float(hi),
            "raw_fail": int(raw.sum()), "tc_fail": int(tc.sum()), "estimator": f"paired log-cluster bootstrap, ratio, {B} draws, seed {seed}, 95% percentile"}


def rank_policy(rep0, rep1, g6, g2, g3_rank1_ok):
    """-> decision dict.  rep0/rep1: parsed repair_rank{0,1}.json (with per-row records) or None.  g6: gate_tb6 result.  g2: gate_tb2 result."""
    need = []
    if rep0 is None or "rows" not in rep0:
        need.append("repair_rank0.json (per-row records) for r_0")
    if rep1 is None or "rows" not in rep1:
        need.append("repair_rank1.json (per-row records) for r_1")
    if need:
        return {"decision": "UNDECIDED", "missing_inputs": need}
    r0, r1 = paired_boot(rep0["rows"]), paired_boot(rep1["rows"])
    d6 = g6["detail"]
    rk = {
        "RK0_stop_rule_r0_hi_lt_0": bool(r0["hi_pp"] < 0),
        "RK1_n_ge_200_and_logs_ge_5": bool(r1["n"] >= MIN_RK1_ROWS and r1["n_logs"] >= MIN_RK1_LOGS),
        "RK2_r1_lo_gt_0": bool(r1["lo_pp"] > 0),
        "RK3_r0_point_gt_0": bool(r0["point_pp"] > 0),
        "RK4_TB6_passes_on_rank1": bool(d6["rank1"].get("pass")),
        "RK5_TB2_TB3_pass_on_every_rank1_row": bool(g2["detail"]["per_rank"][1]["knot_fail"] == 0 and g2["detail"]["per_rank"][1]["knot_dev_bad"] == 0
                                                     and g2["detail"]["per_rank"][1]["n_tc20s"] > 0 and g3_rank1_ok),
        "RK6_guards": "not required: the repair json carries sim8 only (no DDC / comfort / lk10 / executed-vs-intended)",
    }
    if rk["RK0_stop_rule_r0_hi_lt_0"]:
        dec = "STOP"
    elif all(rk[k] for k in ("RK1_n_ge_200_and_logs_ge_5", "RK2_r1_lo_gt_0", "RK3_r0_point_gt_0", "RK4_TB6_passes_on_rank1", "RK5_TB2_TB3_pass_on_every_rank1_row")):
        dec = "all"
    else:
        dec = "rank0"
    stated = {k: {kk: v.get(kk) for kk in ("raw_minus_tc20s", "ci95_log_cluster_bootstrap", "n", "n_logs")} for k, v in (("rank0_report", rep0), ("rank1_report", rep1))}
    return {"decision": dec, "r0": r0, "r1": r1, "rk": rk, "repair_reports_own_numbers_(seed 20261111, NOT the 4.3 estimator)": stated,
            "also_STOP_if_TB6_rank0_fails": bool(d6["rank0_fail_STOP_next_lever_P5R"]),
            "disclosure": "section 17 item 9: the rule was written after the MM's one-line rank-1 summary; the estimator here is 4.3's (seed 20260927), recomputed from the per-row records"}


# ------------------------------------------------------------------ outputs
def write_merged(bank, M, out_dir):
    """merged labeller jsonl for `traj_label_table.py build`.  _all: every row's label (tc20s, or raw_fallback). _rank0only: rank-1 rows carry label := traj (source raw_rank1)."""
    if not np.isfinite(M["label"][M["source"] != "absent"]).all():
        return {"written": False, "reason": "non-finite label present (TB-5 FAIL)"}
    paths = {"all": os.path.join(out_dir, "merged_labels_all.jsonl"), "rank0only": os.path.join(out_dir, "merged_labels_rank0only.jsonl")}
    fa, fb = (open(paths[k], "w", encoding="utf-8", newline="\n") for k in ("all", "rank0only"))
    try:
        for i in range(len(bank["log"])):
            if M["source"][i] == "absent":
                continue
            base = {"log_name": bank["log"][i], "token": bank["token"][i], "rank": int(bank["rank"][i]), "traj_sha256": bank["sha"][i]}
            lab = M["label"][i].tolist()
            fa.write(json.dumps(dict(base, label=lab, source=str(M["source"][i]))) + "\n")
            if bank["rank"][i] == 1:
                fb.write(json.dumps(dict(base, label=bank["traj"][i].tolist(), source="raw_rank1")) + "\n")
            else:
                fb.write(json.dumps(dict(base, label=lab, source=str(M["source"][i]))) + "\n")
    finally:
        fa.close(); fb.close()
    return {"written": True, **paths}


def meta_fields(bank, M, g7, policy, code_files, bank_files, label_name="Tc20s"):
    per = g7["detail"]["per_rank"]
    return {"label_name": label_name, "schema": "traj-label-table-v1", "interp": "cubic", "iters": 3, "n_rows": len(bank["log"]),
            "n_raw_fallback": int((M["source"] == "raw_fallback").sum()), "n_raw_fallback_per_rank": {str(k): per[k]["n_raw_fallback"] for k in per},
            "n_differs_per_rank": {str(k): per[k]["n_differs"] for k in per}, "rank_policy": policy.get("decision"),
            "code_blob_sha1": {os.path.basename(p): git_blob_sha1(p) for p in code_files},
            "bank_files_sha256": {os.path.basename(p): sha256_file(p) for p in bank_files}, "build_host": platform.node(),
            "table_sha256": "UNKNOWN until traj_label_table.py build; the pod runner appends it (P5C_LABEL_TABLE_SHA256)",
            "note": "traj_label_table.py build records only label_name/require_coverage/bank_rows/covered/partial; these extra fields live in this file until that module takes a --meta-json"}


def run_all(bank, recs, n_dup=0, control_recs=None, tb1=None, rep0=None, rep1=None, expect_rows=None, ident_labels=None, tb1_n=BAR_N_TB1, tc20s_tol=0.0):
    M = merge(bank, recs, n_dup)
    g = {"TB-0": gate_tb0(bank, M, control_recs, tc20s_tol, recs), "TB-1": gate_tb1(tb1, tb1_n, tc20s_tol), "TB-2": gate_tb2(bank, M), "TB-3": gate_tb3(bank, ident_labels),
         "TB-4": gate_tb4(bank, M, expect_rows), "TB-5": gate_tb5(bank, M), "TB-6": gate_tb6(bank, M), "TB-7": gate_tb7(bank, M)}
    r1 = bank["rank"] == 1
    n1, _ = ident_check({"traj": bank["traj"][r1]}, None if ident_labels is None else ident_labels[r1]) if r1.any() else (0, 0)
    pol = rank_policy(rep0, rep1, g["TB-6"], g["TB-2"], bool(r1.any() and n1 == int(r1.sum())))
    if pol.get("decision") in ("all", "rank0") and g["TB-6"]["detail"]["rank0_fail_STOP_next_lever_P5R"]:
        pol["decision_from_4_3_rules_alone"] = pol["decision"]
        pol["decision"] = "STOP"                      # TB-6 rank-0 FAIL => STOP (P5-R): no table, whatever RK1..RK6 say
    failing = [k for k in GATE_IDS if g[k]["status"] != "PASS"]
    if pol.get("decision") == "STOP":
        failing.append("RK0-STOP")
    if g["TB-6"]["detail"]["rank0_fail_STOP_next_lever_P5R"]:
        failing.append("TB-6-rank0-STOP")
        if "RK0-STOP" in failing and pol.get("rk", {}).get("RK0_stop_rule_r0_hi_lt_0") is not True:
            failing.remove("RK0-STOP")
    return {"gates": g, "rank_policy": pol, "failing": failing, "M": M}


# ------------------------------------------------------------------ P5-R mode (--label p5r): PREREG_P5C amendment A1 (CONDITIONAL; applies only if the R4 trigger fires)
# Tc20s mode above is untouched except for the R-A tolerance (function default 0.0 == the frozen gate; CLI --tc20s-tol default 1e-9).  Here the label is the dual labeller's `p5r` field
# (PREREG_P5R, lam*), and (MM rulings R-A, R-B of 2026-11-12, A1 section 2):
#   TB-0  = R-B: on the EVAL control rows (pod relabelling of the 2,400 EVAL rows vs raw/eval_r{0,1}.jsonl), PER RANK, ALL THREE of
#             R-TOL-1  sim8 verdict identical on >= 99.5 % of common rows
#             R-TOL-2  per-row max abs label difference (all components, m / rad): median <= 1e-3 and p99 <= 5e-3 (the max is recorded, not gated)
#             R-TOL-3  the decision statistics reproduce: TB-6 acc and heading-rate ratios each within +-0.01 of the dev-box EVAL ratios computed on the SAME common rows (row-matched; verdict.json = context only, not gated),
#                      sim8-fail rate within +-0.25 pp of the dev-box EVAL value
#           AND the Tc20s warm start equals the dev-box Tc20s build within --tc20s-tol (R-A; it feeds P5-R)
#   TB-1  = the Tc20s TB-1 file (R-A tolerance) AND the same three criteria with coverage n == --eval-n
#   TB-2  = S3 (re-simulation identity) <= 1e-6 on every p5r row (replaces the spline knot exactness)
#   TB-3  = unchanged;  TB-4 = unchanged + S4 per rank;  TB-5 = unchanged + S1/S2/S5 flags;  TB-6 = unchanged and binding;  TB-7 = unchanged, on p5r
TOL1_FRAC = 0.995         # R-TOL-1
TOL2_MEDIAN, TOL2_P99 = 1e-3, 5e-3     # R-TOL-2 (the per-row max is recorded only)
TOL3_RATIO, TOL3_SIM8_PP = 0.01, 0.25  # R-TOL-3
BAR_S3 = 1e-6
EVAL_N = 2400


def merge_p5r(bank, recs, n_dup=0):
    """Same join as merge(), on the dual record's `p5r` field.  source: p5r | raw_fallback (p5r_error / malformed: label := traj) | absent."""
    N = len(bank["log"])
    label = np.array(bank["traj"], copy=True)
    source = np.empty(N, dtype=object)
    sha_mismatch = np.zeros(N, bool)
    s3 = np.full(N, np.nan)
    flags = {k: np.full(N, None, dtype=object) for k in ("S1", "S2", "S5")}
    sim8 = np.full(N, None, dtype=object)
    errs = Counter()
    keys = set()
    for i in range(N):
        t = triple(bank["log"][i], bank["token"][i], bank["rank"][i])
        keys.add(t)
        d = recs.get(t)
        if d is None:
            source[i] = "absent"
            continue
        if d.get("traj_sha256") is not None and d["traj_sha256"] != bank["sha"][i]:
            sha_mismatch[i] = True
            source[i] = "absent"
            continue
        pr = d.get("p5r")
        if pr is not None and "p5r_error" not in d:
            a = np.asarray(pr, np.float64)
            if a.shape == (20, 3):
                label[i] = a
                source[i] = "p5r"
                s3[i] = float(d["p5r_S3_dev"]) if d.get("p5r_S3_dev") is not None else np.nan
                flags["S1"][i], flags["S2"][i], flags["S5"][i] = d.get("p5r_S1_finite"), d.get("p5r_S2_yaw_branch"), d.get("p5r_S5_lambda_reached")
                sim8[i] = d.get("p5r_sim8")
                continue
            errs["malformed_shape"] += 1
        source[i] = "raw_fallback"
        errs[str(d.get("p5r_error", "malformed")).split(":")[0][:60]] += 1
    extra = [t for t in recs if t not in keys]
    return {"label": label, "source": source, "sha_mismatch": sha_mismatch, "knot_fail": np.zeros(N, bool), "knot_dev": np.full(N, np.nan), "errs": errs,
            "extra_keys": extra, "n_dup": n_dup, "n_records": len(recs), "s3": s3, "flags": flags, "sim8": sim8}


def load_eval_control(paths, lam_key="3.0"):
    """P5-R EVAL rows (p5r/raw/eval_r{0,1}.jsonl) -> {triple: {"label", "sim8", "learn" (of the label, or None), "raw_learn" (of the raw traj, or None)}}.  Rows with an `error` are not control rows."""
    out = {}
    for p in expand(paths):
        with _open(p) as f:
            for ln in f:
                if not ln.strip():
                    continue
                d = json.loads(ln)
                z = (d.get("lam") or {}).get(lam_key)
                if "error" in d or z is None or "label" not in z:
                    continue
                out[triple(d["log"], d["token"], d["rank"])] = {"label": z["label"], "sim8": bool(z["sim8"]), "learn": z.get("learn"), "raw_learn": (d.get("raw") or {}).get("learn")}
    return out


def load_verdict_ref(path):
    """P5-R's own dev-box EVAL values (p5r/raw/verdict.json, per_rank.<k>.TB6_p5r) -> {rank: {"R_acc", "R_hr"}}.  Medians of per-row label learnability over medians of the bank trajectories'
    on the same 1,200 EVAL rows per rank: the same ratio-of-medians as tb6_rank()."""
    v = json.load(open(path, encoding="utf-8")) if isinstance(path, str) else path
    return {int(k): {"R_acc": float(p["TB6_p5r"]["R_acc"]), "R_hr": float(p["TB6_p5r"]["R_hr"])} for k, p in v["per_rank"].items()}


def _p5r_vs_control(bank, M, ctl, ref):
    """R-B: R-TOL-1/2/3 per rank on the EVAL control rows.  R-TOL-3's reference is ROW-MATCHED (MM ruling on caveat 1): the dev-box TB-6 ratios (ratio of medians, as tb6_rank / p5_target_labels)
    of exactly the common rows, from the control records' per-row `learn` (label) and `raw.learn` when every common row has them, else recomputed with learn() from the stored label and the bank traj
    (MEASURED identical, max diff 0, on the 33 fixture rows).  `ref` = {rank: {"R_acc", "R_hr"}} (verdict.json) is a RECORDED CONTEXT VALUE only, never gated."""
    idx = {triple(bank["log"][i], bank["token"][i], bank["rank"][i]): i for i in range(len(bank["log"]))}
    sel = {0: [], 1: []}
    diffs = {0: [], 1: []}
    sim8_same = {0: 0, 1: 0}
    dev_fail = {0: 0, 1: 0}
    pod_fail = {0: 0, 1: 0}
    n_ctl = {0: 0, 1: 0}
    n_missing = n_not_in_bank = 0
    for t, d in ctl.items():
        i = idx.get(t)
        if i is None:
            n_not_in_bank += 1
            continue
        k = int(bank["rank"][i])
        n_ctl[k] += 1
        if M["source"][i] != "p5r":
            n_missing += 1
            continue
        a = np.asarray(d["label"], np.float64)
        diff = float(np.abs(M["label"][i] - a).max()) if a.shape == (20, 3) else math.inf
        diffs[k].append(diff if not math.isnan(diff) else math.inf)
        sel[k].append(i)
        pod = None if M["sim8"][i] is None else bool(M["sim8"][i])
        sim8_same[k] += int(pod is not None and pod == bool(d["sim8"]))
        dev_fail[k] += int(bool(d["sim8"]))
        pod_fail[k] += int(bool(pod))
    per, failed = {}, []
    for k in (0, 1):
        n = len(sel[k])
        if n == 0:
            per[k] = {"n_common_rows": 0, "n_control_rows": n_ctl[k], "note": "no EVAL control row of this rank in the table"}
            continue
        dd = np.asarray(diffs[k])
        r = {"n_common_rows": n, "n_control_rows": n_ctl[k], "n_sim8_identical": sim8_same[k], "frac_sim8_identical": sim8_same[k] / n,
             "rowmax_median": float(np.median(dd)), "rowmax_p99": float(np.percentile(dd, 99)), "rowmax_max": float(dd.max()),
             "p99_estimator": "numpy.percentile linear, over per-row max abs label difference (x, y, yaw)"}
        r["R-TOL-1"] = bool(r["frac_sim8_identical"] >= TOL1_FRAC)
        r["R-TOL-2"] = bool(r["rowmax_median"] <= TOL2_MEDIAN and r["rowmax_p99"] <= TOL2_P99)
        ix = np.asarray(sel[k])
        la, lh = learn(M["label"][ix])
        ta, th = learn(bank["traj"][ix])
        ra, rh = float(np.median(la) / np.median(ta)), float(np.median(lh) / np.median(th))
        rate_pod, rate_dev = pod_fail[k] / n, dev_fail[k] / n
        r["tb6_ratio_acc"], r["tb6_ratio_hr"] = ra, rh
        cs = [ctl[triple(bank["log"][i], bank["token"][i], k)] for i in sel[k]]
        if all(c.get("learn") is not None and c.get("raw_learn") is not None for c in cs):
            cl = np.asarray([c["learn"] for c in cs], np.float64); cr = np.asarray([c["raw_learn"] for c in cs], np.float64)
            src = "control records' learn / raw.learn"
            r["reference_learn_vs_recomputed_max_abs_diff"] = float(max(np.abs(cl[:, 0] - learn(np.stack([np.asarray(c["label"], np.float64) for c in cs]))[0]).max(), np.abs(cr[:, 0] - ta).max()))
        else:
            cl = np.stack(learn(np.stack([np.asarray(c["label"], np.float64) for c in cs])), 1); cr = np.stack([ta, th], 1)
            src = "recomputed with learn() from the stored control labels and the bank trajectories"
        r["tb6_ref_source"] = src
        r["tb6_ref_acc"], r["tb6_ref_hr"] = float(np.median(cl[:, 0]) / np.median(cr[:, 0])), float(np.median(cl[:, 1]) / np.median(cr[:, 1]))
        r["tb6_diff_acc"], r["tb6_diff_hr"] = ra - r["tb6_ref_acc"], rh - r["tb6_ref_hr"]
        if ref is not None and k in ref:
            r["verdict_json_context"] = {"R_acc": ref[k]["R_acc"], "R_hr": ref[k]["R_hr"], "ref_minus_verdict_acc": r["tb6_ref_acc"] - ref[k]["R_acc"], "ref_minus_verdict_hr": r["tb6_ref_hr"] - ref[k]["R_hr"],
                                         "note": "recorded only, not gated; with full 2,400-row coverage the row-matched reference must agree with verdict.json to 1e-9"}
        r["sim8_fail_rate_pod"], r["sim8_fail_rate_dev_eval"], r["sim8_fail_rate_diff_pp"] = rate_pod, rate_dev, 100 * (rate_pod - rate_dev)
        r["R-TOL-3"] = bool(abs(r["tb6_diff_acc"]) <= TOL3_RATIO and abs(r["tb6_diff_hr"]) <= TOL3_RATIO and abs(r["sim8_fail_rate_diff_pp"]) <= TOL3_SIM8_PP)
        for c in ("R-TOL-1", "R-TOL-2", "R-TOL-3"):
            if r[c] is not True:
                failed.append(f"rank{k}:{c}")
        per[k] = r
    ran = [k for k in (0, 1) if per[k].get("n_common_rows")]
    ok = bool(ran) and n_missing == 0 and not failed
    return ok, {"n_common_rows": sum(per[k].get("n_common_rows", 0) for k in (0, 1)), "n_control_rows": len(ctl), "n_control_rows_not_in_bank": n_not_in_bank,
                "n_control_ok_missing_in_table": n_missing, "max_abs_diff": max([per[k]["rowmax_max"] for k in ran], default=math.inf), "failed_criteria": failed, "per_rank": per,
                "criteria": {"R-TOL-1": f"sim8 identical >= {TOL1_FRAC:.1%} of common rows", "R-TOL-2": f"per-row max abs diff: median <= {TOL2_MEDIAN:g} and p99 <= {TOL2_P99:g}",
                             "R-TOL-3": f"TB-6 acc/hr ratio within +-{TOL3_RATIO:g} of the dev-box ratio on the SAME common rows (row-matched; verdict.json is context only) and sim8-fail rate within +-{TOL3_SIM8_PP:g} pp of the dev-box EVAL value"}}


def gate_tb0_p5r(bank, M_p, M_t, eval_ctl, tc_control_recs, ref=None, tc20s_tol=TC20S_TOL, recs=None):
    bar = (f"R-B: per rank, ALL of R-TOL-1 (sim8 identical >= {TOL1_FRAC:.1%}), R-TOL-2 (per-row max abs diff median <= {TOL2_MEDIAN:g}, p99 <= {TOL2_P99:g}), R-TOL-3 (TB-6 ratios within "
           f"+-{TOL3_RATIO:g} of the dev-box ratios on the same rows, sim8-fail rate within +-{TOL3_SIM8_PP:g} pp) on the P5-R EVAL control rows; AND Tc20s warm start == dev-box Tc20s build within {tc20s_tol:g} (R-A)")
    if eval_ctl is None or tc_control_recs is None:
        return _res("TB-0", "NOT_RUN", "cross-environment identity (P5-R, R-B)", bar, detail={"reason": "needs --eval-control (P5-R EVAL jsonl) and --control (dev-box Tc20s shards)"})
    ok_p, det_p = _p5r_vs_control(bank, M_p, eval_ctl, ref)
    g_t = gate_tb0(bank, M_t, tc_control_recs, tc20s_tol, recs)
    return _res("TB-0", "PASS" if (ok_p and g_t["status"] == "PASS") else "FAIL", "cross-environment identity (P5-R, R-B)", bar, value=det_p["max_abs_diff"], n=det_p["n_common_rows"],
                detail={"p5r_vs_eval_labels": det_p, "tc20s_warm_start_vs_dev_build": {"status": g_t["status"], "value": g_t["value"], "n": g_t["n"], **g_t["detail"]}})


def gate_tb1_p5r(bank, M_p, eval_ctl, tb1, ref=None, eval_n=EVAL_N, tb1_n=BAR_N_TB1, tc20s_tol=TC20S_TOL):
    bar = f"Tc20s TB-1 (1018/1018, diff <= {tc20s_tol:g}) AND reproduction of all {eval_n} P5-R EVAL labels under R-TOL-1/2/3 (per rank, as TB-0)"
    g_t = gate_tb1(tb1, tb1_n, tc20s_tol)
    if eval_ctl is None:
        return _res("TB-1", "NOT_RUN", "reproduction (Tc20s npz + P5-R EVAL)", bar, detail={"reason": "no --eval-control given", "tc20s_tb1": g_t["status"]})
    ok_p, det_p = _p5r_vs_control(bank, M_p, eval_ctl, ref)
    ok = ok_p and det_p["n_common_rows"] == eval_n and det_p["n_control_rows"] == eval_n and g_t["status"] == "PASS"
    return _res("TB-1", "PASS" if ok else "FAIL", "reproduction (Tc20s npz + P5-R EVAL)", bar, value=f"{det_p['n_common_rows']}/{eval_n}", n=det_p["n_common_rows"],
                detail={"p5r_eval_reproduction": det_p, "tc20s_tb1": {"status": g_t["status"], "value": g_t["value"]}})


def gate_tb2_p5r(bank, M, bar_s3=BAR_S3):
    bar = f"S3: tracked8 of the stored label, simulated alone, equals the solver's final s: <= {bar_s3:g} (m, rad) on every p5r row (replaces TB-2's spline knot exactness)"
    ranks = bank["rank"]
    is_p = M["source"] == "p5r"
    s3 = M["s3"]
    missing = int((is_p & np.isnan(s3)).sum())
    bad = int((is_p & ~np.isnan(s3) & (s3 > bar_s3)).sum())
    n = int(is_p.sum())
    per_rank = {int(k): {"n_tc20s": int((is_p & (ranks == k)).sum()), "n_p5r": int((is_p & (ranks == k)).sum()), "knot_fail": int((is_p & (ranks == k) & ((s3 > bar_s3) | np.isnan(s3))).sum()),
                         "knot_dev_bad": 0, "note": "keys named for rank_policy(): n_tc20s == n_p5r, knot_fail == S3 failures"} for k in (0, 1)}
    ok = n > 0 and bad == 0 and missing == 0
    return _res("TB-2", "PASS" if ok else "FAIL", "S3 re-simulation identity", bar, value=(float(np.nanmax(s3[is_p])) if (is_p & ~np.isnan(s3)).any() else None), n=n,
                detail={"n_over_bar": bad, "n_rows_missing_S3_field": missing, "per_rank": per_rank})


def gate_tb4_p5r(bank, M, expect_rows=None):
    g = gate_tb4(bank, M, expect_rows)
    per = {}
    ok_s4 = True
    for k in (0, 1):
        m = bank["rank"] == k
        nf = int(((M["source"] == "raw_fallback") & m).sum())
        per[k] = {"n_rows": int(m.sum()), "n_raw_fallback": nf, "frac": (nf / int(m.sum())) if m.any() else None}
        ok_s4 = ok_s4 and (not m.any() or nf <= BAR_FALLBACK * int(m.sum()))
    g["detail"]["S4_per_rank"] = per
    g["bar"] += "; S4: fallbacks <= 1.0 % of rows PER RANK"
    if not ok_s4:
        g["status"] = "FAIL"
    return g


def gate_tb5_p5r(bank, M):
    g = gate_tb5(bank, M)
    is_p = M["source"] == "p5r"
    bad = {}
    for k in ("S1", "S2", "S5"):
        bad[k] = int(sum(1 for i in np.where(is_p)[0] if M["flags"][k][i] is not True))
    g["detail"]["S1_S2_S5_not_true"] = bad
    g["bar"] += "; S1/S2/S5 flags recorded true on every p5r row"
    if any(bad.values()):
        g["status"] = "FAIL"
    return g


def prep_repair_p5r(rep):
    """The 4.3 estimator reads `tc20s_sim8`; for P5-R it reads `p5r_sim8` (r_k recomputed with P5-R labels, A1 point 4).  -> rep copy, or None if p5r_sim8 is absent."""
    if rep is None or "rows" not in rep or not all("p5r_sim8" in r for r in rep["rows"]):
        return None
    d = dict(rep)
    d["rows"] = [dict(r, tc20s_sim8=r["p5r_sim8"]) for r in rep["rows"]]
    return d


def run_all_p5r(bank, recs, n_dup=0, control_tc=None, eval_ctl=None, tb1=None, rep0=None, rep1=None, expect_rows=None, ident_labels=None, tb1_n=BAR_N_TB1,
                ref=None, eval_n=EVAL_N, tc20s_tol=TC20S_TOL):
    M = merge_p5r(bank, recs, n_dup)
    M_t = merge(bank, recs, n_dup)                   # the Tc20s warm start, for the TB-0 half that demands 0.0
    g = {"TB-0": gate_tb0_p5r(bank, M, M_t, eval_ctl, control_tc, ref, tc20s_tol, recs), "TB-1": gate_tb1_p5r(bank, M, eval_ctl, tb1, ref, eval_n, tb1_n, tc20s_tol), "TB-2": gate_tb2_p5r(bank, M),
         "TB-3": gate_tb3(bank, ident_labels), "TB-4": gate_tb4_p5r(bank, M, expect_rows), "TB-5": gate_tb5_p5r(bank, M), "TB-6": gate_tb6(bank, M, "p5r"), "TB-7": gate_tb7(bank, M, "p5r")}
    g["TB-6"]["bar"] = g["TB-6"]["bar"].replace("STOP (P5-R)", "STOP (PREREG_P5R 10: next lever P5-T)")
    g["TB-6"]["detail"]["rank0_fail_next_lever"] = "P5-T (selective label), PREREG_P5R 10"
    r1 = bank["rank"] == 1
    n1, _ = ident_check({"traj": bank["traj"][r1]}, None if ident_labels is None else ident_labels[r1]) if r1.any() else (0, 0)
    pol = rank_policy(prep_repair_p5r(rep0), prep_repair_p5r(rep1), g["TB-6"], g["TB-2"], bool(r1.any() and n1 == int(r1.sum())))
    if pol.get("decision") == "UNDECIDED":
        pol["missing_inputs"] = [m + " WITH p5r_sim8 (p5c_dual_labels.py repair)" for m in pol.get("missing_inputs", [])]
    if pol.get("decision") in ("all", "rank0") and g["TB-6"]["detail"]["rank0_fail_STOP_next_lever_P5R"]:
        pol["decision_from_4_3_rules_alone"] = pol["decision"]
        pol["decision"] = "STOP"
    failing = [k for k in GATE_IDS if g[k]["status"] != "PASS"]
    if pol.get("decision") == "STOP":
        failing.append("RK0-STOP")
    if g["TB-6"]["detail"]["rank0_fail_STOP_next_lever_P5R"]:
        failing.append("TB-6-rank0-STOP")
        if "RK0-STOP" in failing and pol.get("rk", {}).get("RK0_stop_rule_r0_hi_lt_0") is not True:
            failing.remove("RK0-STOP")
    return {"gates": g, "rank_policy": pol, "failing": failing, "M": M}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bank", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--control", nargs="+")
    ap.add_argument("--tb1")
    ap.add_argument("--tb1-n", type=int, default=BAR_N_TB1)
    ap.add_argument("--repair0")
    ap.add_argument("--repair1")
    ap.add_argument("--identity-table")
    ap.add_argument("--expect-rows", type=int)
    ap.add_argument("--code-file", nargs="*", default=[])
    ap.add_argument("--label", choices=("tc20s", "p5r"), default="tc20s", help="p5r = PREREG_P5C amendment A1 mode (conditional): gates on the dual records' p5r field")
    ap.add_argument("--eval-control", nargs="+", help="p5r mode: the P5-R EVAL jsonl (p5r/raw/eval_r{0,1}.jsonl) for TB-0 / TB-1")
    ap.add_argument("--lambda-star", help="p5r mode: lambda_star.json (selects the EVAL label key and is recorded in the meta)")
    ap.add_argument("--eval-n", type=int, default=EVAL_N)
    ap.add_argument("--p5r-verdict", help="p5r mode: P5-R's verdict.json (p5r/raw/verdict.json): recorded as a context value next to R-TOL-3's row-matched reference, never gated")
    ap.add_argument("--tc20s-tol", type=float, default=TC20S_TOL, help="R-A: Tc20s TB-0 / TB-1 tolerance (m, rad), default 1e-9; 0 reproduces the frozen PREREG_P5C gates (bit identity)")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)

    bank = load_bank(a.bank)
    recs, n_rec, n_dup = load_label_recs(a.labels)
    control = load_label_recs(a.control)[0] if a.control else None
    ident = None
    if a.identity_table:
        with gzip.open(a.identity_table, "rt", encoding="utf-8") as f:
            tab = json.load(f)["table"]
        ident = np.stack([np.asarray(tab.get(f"{triple(bank['log'][i], bank['token'][i], bank['rank'][i])}|{bank['sha'][i]}", [[math.nan] * 3] * 20), np.float64) for i in range(len(bank["log"]))])
    load = lambda p: json.load(open(p, encoding="utf-8")) if p else None
    if a.label == "p5r":
        lam = float(load(a.lambda_star)["lambda_star"]) if a.lambda_star else 3.0
        ev = load_eval_control(a.eval_control, repr(lam)) if a.eval_control else None
        ref = load_verdict_ref(a.p5r_verdict) if a.p5r_verdict else None
        R = run_all_p5r(bank, recs, n_dup, control, ev, load(a.tb1), load(a.repair0), load(a.repair1), a.expect_rows, ident, a.tb1_n, ref, a.eval_n, a.tc20s_tol)
    else:
        R = run_all(bank, recs, n_dup, control, load(a.tb1), load(a.repair0), load(a.repair1), a.expect_rows, ident, a.tb1_n, a.tc20s_tol)
    wr = write_merged(bank, R["M"], a.out_dir)
    meta = meta_fields(bank, R["M"], R["gates"]["TB-7"], R["rank_policy"], a.code_file, bank["files"], label_name="P5R" if a.label == "p5r" else "Tc20s")
    if a.label == "p5r":
        meta["p5r"] = {"lambda_star": lam, "k_max": 10, "warm_start": "Tc20s (iterate20, cubic, ITERS 3)", "R_B": {"R-TOL-1_min_frac_sim8_identical": TOL1_FRAC, "R-TOL-2_median_max": TOL2_MEDIAN, "R-TOL-2_p99_max": TOL2_P99, "R-TOL-3_ratio_abs": TOL3_RATIO, "R-TOL-3_sim8_pp": TOL3_SIM8_PP}, "tc20s_tol_R_A": a.tc20s_tol, "prereg": "PREREG_P5R.md sha256 cfd23e0651d66e497238a0708c717ba00acbcb2189295efcfb15dad657922a42"}
    json.dump(meta, open(os.path.join(a.out_dir, "table_meta_fields.json"), "w", newline="\n"), indent=1)
    out = {"gates": R["gates"], "rank_policy": R["rank_policy"], "failing": R["failing"], "merged_jsonl": wr, "label_records": n_rec,
           "bank_files": bank["files"], "inputs": {"labels": a.labels, "control": a.control, "tb1": a.tb1}}
    json.dump(out, open(os.path.join(a.out_dir, "gates.json"), "w", newline="\n"), indent=1, default=str)
    if a.label == "p5r":
        print("LABEL p5r (PREREG_P5C amendment A1, conditional)")
    for k in GATE_IDS:
        g = R["gates"][k]
        print(f"{k} {g['status']:8s} n={g['n']} value={g['value']} bar=[{g['bar']}]")
    print("RANK_POLICY", R["rank_policy"]["decision"], json.dumps(R["rank_policy"].get("rk", R["rank_policy"].get("missing_inputs"))))
    if R["failing"]:
        print("ZZTB_FAIL " + " ".join(R["failing"]))
        return 1
    print("ZZTB_ALL_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
