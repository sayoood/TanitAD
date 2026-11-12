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
def gate_tb0(bank, M, control_recs):
    bar = "max abs diff == 0.0 on every common row; every control-ok row present and ok in the table build"
    if control_recs is None:
        return _res("TB-0", "NOT_RUN", "cross-environment bit-identity", bar, detail={"reason": "no --control shards given"})
    idx = {triple(bank["log"][i], bank["token"][i], bank["rank"][i]): i for i in range(len(bank["log"]))}
    n_common = n_missing = n_sha = 0
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
    ok = n_common > 0 and n_missing == 0 and n_sha == 0 and maxd == 0.0
    return _res("TB-0", "PASS" if ok else "FAIL", "cross-environment bit-identity", bar, value=maxd, n=n_common,
                detail={"n_common_rows": n_common, "n_control_logs": len(logs), "n_control_ok_missing_in_table": n_missing, "n_traj_sha_mismatch": n_sha})


def gate_tb1(tb1, expect_n=BAR_N_TB1):
    bar = f"bit_identical == n == {expect_n}, max_abs_diff == 0.0, n_errors == 0"
    if tb1 is None:
        return _res("TB-1", "NOT_RUN", "reproduction of p5s_cubic.npz", bar, detail={"reason": "no --tb1 file given"})
    ok = tb1.get("n") == expect_n and tb1.get("bit_identical") == expect_n and tb1.get("max_abs_diff") == 0.0 and tb1.get("n_errors", 1) == 0
    return _res("TB-1", "PASS" if ok else "FAIL", "reproduction of p5s_cubic.npz", bar, value=f"{tb1.get('bit_identical')}/{tb1.get('n')}", n=tb1.get("n"),
                detail={k: tb1.get(k) for k in ("max_abs_diff", "n_errors", "seconds", "errors")})


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


def tb6_rank(bank, M, k):
    sel = np.where((M["source"] == "tc20s") & (bank["rank"] == k))[0]
    if len(sel) == 0:
        return {"rank": k, "n": 0, "pass": False, "reason": "no tc20s rows at this rank"}
    la, lh = learn(M["label"][sel])
    ta, th = learn(bank["traj"][sel])
    ma, mh, mta, mth = (float(np.median(x)) for x in (la, lh, ta, th))
    ra = ma / mta if mta > 0 else math.inf
    rh = mh / mth if mth > 0 else math.inf
    with np.errstate(divide="ignore", invalid="ignore"):
        mra, mrh = float(np.nanmedian(la / ta)), float(np.nanmedian(lh / th))      # DIAGNOSTIC only (median of per-row ratios); the bar reads the ratio of medians
    return {"rank": k, "n": int(len(sel)), "diag_median_of_row_ratios_acc": mra, "diag_median_of_row_ratios_hr": mrh, "median_max_acc_label": ma, "median_max_acc_traj": mta, "ratio_acc": ra,
            "median_max_hr_label_deg_s": mh, "median_max_hr_traj_deg_s": mth, "ratio_hr": rh, "pass": bool(ra <= BAR_LEARN and rh <= BAR_LEARN)}


def gate_tb6(bank, M):
    bar = f"per rank: median per-row max|acc| and max|heading rate| of labels / same for traj <= {BAR_LEARN:g}x on both. Rank-0 FAIL => STOP (P5-R); rank-1 FAIL => rank-0 only"
    r0, r1 = tb6_rank(bank, M, 0), tb6_rank(bank, M, 1)
    stop = not r0["pass"]
    return _res("TB-6", "PASS" if (r0["pass"] and r1["pass"]) else "FAIL", "learnability per rank", bar,
                value={"rank0": [r0.get("ratio_acc"), r0.get("ratio_hr")], "rank1": [r1.get("ratio_acc"), r1.get("ratio_hr")]}, n=r0["n"] + r1["n"],
                detail={"rank0": r0, "rank1": r1, "rank0_fail_STOP_next_lever_P5R": stop, "rank1_fail_means_rank0_only": not r1["pass"],
                        "population": "rows whose label source is tc20s (raw_fallback rows are identity and excluded)"})


def gate_tb7(bank, M):
    bar = "n_differs per rank recorded"
    per = {}
    for k in (0, 1):
        sel = (M["source"] == "tc20s") & (bank["rank"] == k)
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


def run_all(bank, recs, n_dup=0, control_recs=None, tb1=None, rep0=None, rep1=None, expect_rows=None, ident_labels=None, tb1_n=BAR_N_TB1):
    M = merge(bank, recs, n_dup)
    g = {"TB-0": gate_tb0(bank, M, control_recs), "TB-1": gate_tb1(tb1, tb1_n), "TB-2": gate_tb2(bank, M), "TB-3": gate_tb3(bank, ident_labels),
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
    R = run_all(bank, recs, n_dup, control, load(a.tb1), load(a.repair0), load(a.repair1), a.expect_rows, ident, a.tb1_n)
    wr = write_merged(bank, R["M"], a.out_dir)
    meta = meta_fields(bank, R["M"], R["gates"]["TB-7"], R["rank_policy"], a.code_file, bank["files"])
    json.dump(meta, open(os.path.join(a.out_dir, "table_meta_fields.json"), "w", newline="\n"), indent=1)
    out = {"gates": R["gates"], "rank_policy": R["rank_policy"], "failing": R["failing"], "merged_jsonl": wr, "label_records": n_rec,
           "bank_files": bank["files"], "inputs": {"labels": a.labels, "control": a.control, "tb1": a.tb1}}
    json.dump(out, open(os.path.join(a.out_dir, "gates.json"), "w", newline="\n"), indent=1, default=str)
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
