#!/usr/bin/env python3
"""Paired log-cluster bootstrap CI for the four metric families (families_run.py JSONs).

Estimator (matches teacher_fidelity/n2/tfn2_analyze.py::boot exactly): clusters = logs, resample logs with replacement,
statistic = (sum of the drawn logs' per-token sums) / (sum of the drawn logs' token counts) of the paired difference
B - A, B = 10,000, seed 20260927, 95 % percentile. It answers "would another draw of LOGS say this?" and nothing else
(not another training run, not another inference run).

Per-token data: families_run.py's JSON stores per-window arrays ONLY for longitudinal.distance_keeping (headway_min_m,
time_gap_min_s, min_ttc_s, lead_viol, lead_at_risk, lead_progress_ratio). Every other metric is reported as
NO-PER-TOKEN-DATA (work item: see FAM_CI.md). The JSON stores neither tokens nor log names: pass --tokens (lines
`log:token`); array order = tokens sorted by the token part (families_run.read_rows sorts by token; verified on the
banked data against the lead state of every token). A JSON that carries top-level `tokens` / `logs` lists is read directly.

Subset metrics (NaN = undefined for that token, e.g. no lead agent) pair on the INTERSECTION of tokens finite in both arms.
Logs with no valid token in the intersection are not resampled (the resample is over logs that have support).
STRATEGIC is PAUSED (PI ruling 2026-09-19), reported as PAUSED, never as a pass.

NEW (fam_pertoken): a JSON made by `families_run.py --per-window` carries top-level `tokens`/`logs` and, per frame, `block._per_window`
(taniteval.per_window/1: one record per LONGITUDINAL / LATERAL / TACTICAL component). Those metrics now get a paired CI too, by token ID
(refused on any mismatch). Per record kind the bootstrap statistic over the drawn logs is
  mean    sum(v)/count                       (tokens finite in BOTH arms)
  rms     sqrt(sum(v)/count)
  ratio   scale * sum(num)/sum(den)          RATIO OF SUMS (micro mean: LATERAL errors over valid steps; subset means), NOT a mean of per-token
                                             means; each arm uses its OWN num/den; diff = stat(B) - stat(A)
  median  median of the drawn tokens' values
  class   decision gt/pred ids: accuracy = mean(gt==pred) (per-token 0/1 correctness), per-class recall/precision (ratio of sums), kappa
          from the drawn confusion counts
All new metrics share one draw matrix (seed 20260927). Not exported by taniteval.per_window (so still NO-PER-TOKEN-DATA):
tactical goal_setting.goal_bearing_bias_deg.

  python fam_ci.py --a fam_R0_R.json --b fam_RJ5_R.json --tokens tokens.txt --out out.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np

B_DRAWS, SEED = 10000, 20260927
FRAMES = ("executed_tracked", "executed_transparent", "reference")
PER_TOKEN_LONG = ("headway_min_m", "time_gap_min_s", "min_ttc_s", "lead_viol", "lead_at_risk", "lead_progress_ratio")
STRATEGIC_PAUSED = {"status": "PAUSED", "reason": "PI ruling 2026-09-19 (PI_DECISION_QUEUE item 25): strategic layer "
                    "switched off temporarily. NOT a pass, not omitted."}
NO_PT_REASON = ("families_run.py emits no per-token array for this metric (only longitudinal.distance_keeping._per_window "
                "is exported); a paired CI needs it")


class TokenMismatch(Exception):
    pass


def cluster_boot(v, logs, n_boot=B_DRAWS, seed=SEED):
    """v: finite per-token values, logs: cluster label per token -> (point, lo, hi, n_logs). tfn2_analyze.boot convention."""
    v = np.asarray(v, float)
    lg = np.asarray(logs)
    ul = sorted(set(lg.tolist()))
    s = np.array([v[lg == l].sum() for l in ul])
    n = np.array([(lg == l).sum() for l in ul])
    rng = np.random.default_rng(seed)
    d = rng.integers(0, len(ul), (n_boot, len(ul)))
    m = s[d].sum(1) / n[d].sum(1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), len(ul)


def pertoken_boot(v, n_boot=B_DRAWS, seed=SEED):
    """MUTATION / contrast arm ONLY: ignores the log structure (every token its own cluster). compare() never uses it."""
    v = np.asarray(v, float)
    rng = np.random.default_rng(seed)
    d = rng.integers(0, len(v), (n_boot, len(v)))
    m = v[d].mean(1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), len(v)



# ----------------------------------------------------------------------------- per-window (families_run --per-window) path
HUMAN_ONLY = ("ego_progress.gt_progress_mean_m", "ego_progress.t0_axis_gt_self_ratio")
SKIP_NEW = ("distance_keeping.",)     # superseded by the legacy distance_keeping._per_window rows (intersection pairing)


def _sums_by_log(M, logs, ul):
    """M [n,k] -> [L,k] per-log column sums, in the order of `ul`. (Same per-log summation as cluster_boot.)"""
    return np.stack([M[logs == l].sum(0) for l in ul]) if len(ul) else np.zeros((0, M.shape[1]))


def _draws(L, n_boot, seed):
    return np.random.default_rng(seed).integers(0, L, (n_boot, L))


def _stat_ratio(S, scale=1.0, root=False):
    num, den = S[..., 0], S[..., 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        v = num / den
    if root:
        v = np.sqrt(v)
    return scale * v


def _stat_kappa(S, k):
    C = S.reshape(S.shape[:-1] + (k, k)).astype(float)       # [.., gt, pred]
    N = C.sum((-1, -2))
    po = np.trace(C, axis1=-2, axis2=-1) / N
    pe = (C.sum(-1) * C.sum(-2)).sum(-1) / N ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(1 - pe > 1e-9, (po - pe) / (1 - pe), np.nan)


def _sufficient(rec_a, rec_b):
    """per-token sufficient statistics of ONE component for both arms -> (SA, SB, stat_fn, defined_A, defined_B, kind)."""
    k = rec_a["kind"]
    if k in ("mean", "rms"):
        va, vb = np.asarray(rec_a["value"], float), np.asarray(rec_b["value"], float)
        ok = np.isfinite(va) & np.isfinite(vb)
        mk = lambda v: np.stack([np.where(ok, v, 0.0), ok.astype(float)], 1)            # noqa: E731
        return mk(va), mk(vb), (lambda S: _stat_ratio(S, 1.0, k == "rms")), ok, ok, k
    if k == "ratio":
        mk = lambda r: np.stack([np.asarray(r["num"], float), np.asarray(r["den"], float)], 1)   # noqa: E731
        sc = float(rec_a["scale"])
        return mk(rec_a), mk(rec_b), (lambda S: _stat_ratio(S, sc)), np.asarray(rec_a["den"]) > 0, np.asarray(rec_b["den"]) > 0, k
    raise ValueError(f"unknown component kind {k!r}")


def _boot_diff(SA, SB, fn, logs, support, n_boot, seed):
    """-> ({A,B,diff_B_minus_A: {point,lo,hi}}, n_logs). Only logs with support are resampled."""
    ul = sorted(set(logs[support].tolist()))
    la, lb = _sums_by_log(SA, logs, ul), _sums_by_log(SB, logs, ul)
    d = _draws(len(ul), n_boot, seed)
    out = {}
    pa, pb = fn(la.sum(0)), fn(lb.sum(0))
    ba, bb = fn(la[d].sum(1)), fn(lb[d].sum(1))
    for nm, p, b in (("A", pa, ba), ("B", pb, bb), ("diff_B_minus_A", pb - pa, bb - ba)):
        fin = np.isfinite(b)        # a draw on which the statistic is undefined (e.g. kappa, 1-pe = 0) is dropped and counted
        out[nm] = {"point": float(p), "lo": float(np.percentile(b[fin], 2.5)), "hi": float(np.percentile(b[fin], 97.5))}
        if not fin.all():
            out[nm]["n_draws_undefined"] = int((~fin).sum())
    return out, len(ul)


def _boot_median(va, vb, rows, logs, n_boot, seed):
    ul = sorted(set(logs[rows].tolist()))
    idx = [rows[logs[rows] == l] for l in ul]
    d = _draws(len(ul), n_boot, seed)
    ma, mb = np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        j = np.concatenate([idx[c] for c in d[i]])
        ma[i], mb[i] = np.median(va[j]), np.median(vb[j])
    out = {}
    pa, pb = float(np.median(va[rows])), float(np.median(vb[rows]))
    for nm, p, b in (("A", pa, ma), ("B", pb, mb), ("diff_B_minus_A", pb - pa, mb - ma)):
        out[nm] = {"point": float(p), "lo": float(np.percentile(b, 2.5)), "hi": float(np.percentile(b, 97.5))}
    return out, len(ul)


def _entry(res, nlogs, support, dA, dB, kind, extra=None):
    e = {"status": "OK", "kind": kind, "n_tokens": int(support.sum()), "n_tokens_defined_A": int(dA.sum()), "n_tokens_defined_B": int(dB.sum()),
         "n_logs": nlogs}
    e.update(res)
    d = e["diff_B_minus_A"]
    e["separated_from_0"] = bool(d["lo"] > 0 or d["hi"] < 0)
    if extra:
        e.update(extra)
    return e


def _too_few(support, logs, dA, dB, kind):
    return {"status": "TOO-FEW", "kind": kind, "note": "fewer than 2 tokens or 2 logs in the resample support", "n_tokens": int(support.sum()),
            "n_tokens_defined_A": int(dA.sum()), "n_tokens_defined_B": int(dB.sum()), "n_logs": len(set(logs[support].tolist()))}


def _perm(rec, perm):
    """B's per-token arrays re-ordered to A's token order."""
    if perm is None:
        return rec
    out = dict(rec)
    for k in ("value", "num", "den"):
        if k in rec:
            out[k] = np.asarray(rec[k])[perm]
    return out


def _paired(SA, SB, fn, dA, dB, kind, logs, n_boot, seed, ratio_totals=False):
    support = dA | dB
    if support.sum() < 2 or len(set(logs[support].tolist())) < 2:
        return _too_few(support, logs, dA, dB, kind)
    res, nl = _boot_diff(SA, SB, fn, logs, support, n_boot, seed)
    extra = {"den_total_A": float(SA[:, 1].sum()), "den_total_B": float(SB[:, 1].sum())} if ratio_totals else None
    return _entry(res, nl, support, dA, dB, kind, extra)


def paired_components(a, b, fr, tokens_logs, perm, n_boot, seed):
    """-> {family: {name: entry}} for every component both JSONs export under `frames[fr].block._per_window`; {} when either lacks it."""
    try:
        pa = a["frames"][fr]["block"]["_per_window"]
        pb = b["frames"][fr]["block"]["_per_window"]
    except (KeyError, TypeError):
        return {}
    if pa.get("status") != "OK" or pb.get("status") != "OK":
        return {}
    tk, lg = tokens_logs
    logs = np.asarray(lg)
    for nm, j, pw in (("A", a, pa), ("B", b, pb)):
        if pw.get("tokens") is not None and list(pw["tokens"]) != list(j["tokens"]):
            raise TokenMismatch(f"{fr} {nm}: _per_window.tokens differ from the JSON's top-level tokens")
        if int(pw.get("n", -1)) != len(tk):
            raise TokenMismatch(f"{fr} {nm}: _per_window.n={pw.get('n')} != {len(tk)} tokens")
    out = {"longitudinal": {}, "lateral": {}, "tactical": {}}
    n = len(tk)
    for fam in ("longitudinal", "lateral"):
        for k, ra in (pa.get(fam) or {}).items():
            rb = (pb.get(fam) or {}).get(k)
            if rb is None or any(k.startswith(p) for p in SKIP_NEW):
                continue
            if k in HUMAN_ONLY:
                out[fam][k] = {"status": "HUMAN-ONLY", "note": "a property of the logged human future, identical for every arm; never compared"}
                continue
            if ra["kind"] == "median":
                va, vb = np.full(n, np.nan), np.full(n, np.nan)
                va[np.asarray(ra["rows"], int)] = ra["value"]
                vb[np.asarray(rb["rows"], int)] = rb["value"]
                vb = vb[perm] if perm is not None else vb
                dA, dB = np.isfinite(va), np.isfinite(vb)
                rows = dA & dB
                if rows.sum() < 2 or len(set(logs[rows].tolist())) < 2:
                    out[fam][k] = _too_few(rows, logs, dA, dB, "median")
                else:
                    res, nl = _boot_median(va, vb, np.flatnonzero(rows), logs, n_boot, seed)
                    out[fam][k] = _entry(res, nl, rows, dA, dB, "median")
                continue
            SA, SB, fn, dA, dB, kind = _sufficient(ra, _perm(rb, perm))
            out[fam][k] = _paired(SA, SB, fn, dA, dB, kind, logs, n_boot, seed, ratio_totals=(kind == "ratio"))
    tac_a, tac_b = pa.get("tactical") or {}, pb.get("tactical") or {}
    if tac_a.get("status") == "OK" and tac_b.get("status") == "OK":
        for sub in ("lateral_decision", "longitudinal_decision", "maneuver_5way_collapsed"):
            da, db = tac_a[sub], tac_b[sub]
            gb = np.asarray(db["gt"], int)
            gb = gb[perm] if perm is not None else gb
            if da["classes"] != db["classes"] or not np.array_equal(np.asarray(da["gt"], int), gb):
                raise TokenMismatch(f"{fr} tactical.{sub}: the human's executed-manoeuvre ids differ between arms (same windows must give the same gt)")
            g = np.asarray(da["gt"], int)
            pr_a = np.asarray(da["pred"], int)
            pr_b = np.asarray(db["pred"], int)
            pr_b = pr_b[perm] if perm is not None else pr_b
            kk = len(da["classes"])
            allv = np.ones(n, bool)
            ones = np.ones(n)
            sets = {"accuracy": (np.stack([(g == pr_a).astype(float), ones], 1), np.stack([(g == pr_b).astype(float), ones], 1), _stat_ratio, allv, allv)}
            conf = lambda p: np.eye(kk * kk)[g * kk + p]                                         # noqa: E731
            sets["kappa"] = (conf(pr_a), conf(pr_b), (lambda S, kk=kk: _stat_kappa(S, kk)), allv, allv)
            for ci_, cn in enumerate(da["classes"]):
                for nm, dn in (("recall", lambda p, ci_=ci_: g == ci_), ("precision", lambda p, ci_=ci_: p == ci_)):
                    mk = lambda p, dn=dn, ci_=ci_: np.stack([((g == ci_) & (p == ci_)).astype(float), dn(p).astype(float)], 1)   # noqa: E731
                    sets[f"per_class.{cn}.{nm}"] = (mk(pr_a), mk(pr_b), _stat_ratio, dn(pr_a), dn(pr_b))
            for nm, (SA, SB, fn, dA, dB) in sets.items():
                out["tactical"][f"{sub}.{nm}"] = _paired(SA, SB, fn, dA, dB, "class", logs, n_boot, seed)
        ga, gb_ = tac_a.get("goal_setting") or {}, tac_b.get("goal_setting") or {}
        for k, ra in ga.items():
            rb = gb_.get(k)
            if rb is None:
                continue
            SA, SB, fn, dA, dB, kind = _sufficient(ra, _perm(rb, perm))
            out["tactical"][f"goal_setting.{k}"] = _paired(SA, SB, fn, dA, dB, kind, logs, n_boot, seed, ratio_totals=(kind == "ratio"))
    return out


def read_tokens(path):
    """lines `log:token` -> (tokens, logs) in array order = sorted by the token part."""
    pairs = []
    for ln in open(path, encoding="utf-8"):
        ln = ln.strip()
        if ln:
            lg, tk = ln.rsplit(":", 1)
            pairs.append((tk, lg))
    pairs.sort()
    return [p[0] for p in pairs], [p[1] for p in pairs]


def _tokens_of(j, given):
    if isinstance(j.get("tokens"), list) and isinstance(j.get("logs"), list):
        return list(j["tokens"]), list(j["logs"])
    if given is None:
        raise TokenMismatch("JSON carries no tokens/logs and no --tokens file was given")
    return given


def _fl(arr):
    return np.array([np.nan if x is None else x for x in arr], float)


def _pw(j, frame):
    try:
        return j["frames"][frame]["block"]["longitudinal"]["distance_keeping"]["_per_window"]
    except (KeyError, TypeError):
        return None


def _scalars(d):
    return {k: float(v) for k, v in (d or {}).items()
            if not k.startswith("_") and k not in ("n", "n_windows", "n_bearing") and not k.startswith("n_excluded")
            and isinstance(v, (int, float)) and not isinstance(v, bool)}


def _point_metrics(j, frame):
    """every metric the JSON reports as a point value: (family, name) -> (value, own_ci)."""
    try:
        blk = j["frames"][frame]["block"]
    except (KeyError, TypeError):
        return {}
    out = {}
    for fam in ("longitudinal", "lateral"):
        comp = ((blk.get(fam) or {}).get("ci") or {}).get("components") or {}
        for k, c in comp.items():
            out[(fam, k)] = (c.get("mean"), [c.get("lo"), c.get("hi")])
    tac = blk.get("tactical") or {}
    for sub in ("lateral_decision", "longitudinal_decision", "maneuver_5way_collapsed", "goal_setting"):
        for k, v in _scalars(tac.get(sub)).items():
            out[("tactical", f"{sub}.{k}")] = (v, None)
    return out


def _check_pair(a, b, ta, tb):
    (tka, _), (tkb, _) = ta, tb
    for nm, j, tk in (("A", a, tka), ("B", b, tkb)):
        if j.get("n") is not None and j["n"] != len(tk):
            raise TokenMismatch(f"{nm}: JSON n={j['n']} but {len(tk)} tokens supplied")
    if len(tka) != len(tkb) or set(tka) != set(tkb):
        raise TokenMismatch(f"token sets differ: A {len(tka)} tokens / {len(set(tka))} unique, B {len(tkb)} / {len(set(tkb))}, "
                            f"|A&B|={len(set(tka) & set(tkb))}")
    if tka and (ta[1] is not None) and (tb[1] is not None) and dict(zip(tka, ta[1])) != dict(zip(tkb, tb[1])):
        raise TokenMismatch("token -> log mapping differs between A and B (same token under different logs)")
    if a.get("n_logs") is not None and b.get("n_logs") is not None and a["n_logs"] != b["n_logs"]:
        raise TokenMismatch(f"n_logs differ: A {a['n_logs']} vs B {b['n_logs']}")
    if a.get("lead_states") != b.get("lead_states"):
        raise TokenMismatch(f"lead_states differ (shared across arms by construction): {a.get('lead_states')} vs {b.get('lead_states')}")
    if tka != tkb:  # same set, different order: re-align B to A
        pos = {t: i for i, t in enumerate(tkb)}
        return [pos[t] for t in tka]
    return None


def compare(a, b, tokens=None, tokens_b=None, n_boot=B_DRAWS, seed=SEED):
    ta = _tokens_of(a, tokens)
    tb = _tokens_of(b, tokens_b if tokens_b is not None else tokens)
    perm = _check_pair(a, b, ta, tb)
    logs = np.asarray(ta[1])
    res = {"estimator": f"paired log-cluster bootstrap: clusters = logs, resample logs with replacement, statistic = mean over the "
                        f"drawn tokens of B-A (ratio of sums for micro-mean/subset metrics: see module doc); B={n_boot}, seed {seed}, 95% percentile (tfn2_analyze.boot convention)",
           "answers": "another draw of LOGS (not another training run, not another inference run)",
           "direction": "difference = B - A", "n_tokens": len(ta[0]), "n_logs_all": len(set(ta[1])),
           "label_a": a.get("_label"), "label_b": b.get("_label"), "frames": {}}
    for fr in FRAMES:
        fres = {"role": (a.get("frames", {}).get(fr) or {}).get("role"), "families": {}}
        pa, pb = _pw(a, fr), _pw(b, fr)
        ptA, ptB = _point_metrics(a, fr), _point_metrics(b, fr)
        have_pt = pa is not None and pb is not None
        newc = (paired_components(a, b, fr, (ta[0], ta[1]), perm, n_boot, seed)
                if isinstance(a.get("tokens"), list) and isinstance(b.get("tokens"), list) else {})
        for fam in ("longitudinal", "lateral", "tactical"):
            fam_out = {}
            if fam == "longitudinal" and have_pt:
                for k in PER_TOKEN_LONG:
                    if k not in pa or k not in pb:
                        continue
                    va, vb = _fl(pa[k]), _fl(pb[k])
                    if len(va) != len(logs) or len(vb) != len(logs):
                        raise TokenMismatch(f"{fr}/{k}: per-window length {len(va)}/{len(vb)} != {len(logs)} tokens")
                    if perm is not None:
                        vb = vb[perm]
                    ok = np.isfinite(va) & np.isfinite(vb)
                    entry = {"status": "OK", "n_tokens": int(ok.sum()), "n_tokens_defined_A": int(np.isfinite(va).sum()),
                             "n_tokens_defined_B": int(np.isfinite(vb).sum()), "n_logs": len(set(logs[ok].tolist()))}
                    if ok.sum() < 2 or entry["n_logs"] < 2:
                        entry.update(status="TOO-FEW", note="fewer than 2 tokens or 2 logs on the intersection")
                    else:
                        for nm, vals in (("A", va[ok]), ("B", vb[ok]), ("diff_B_minus_A", vb[ok] - va[ok])):
                            p, lo, hi, _ = cluster_boot(vals, logs[ok].tolist(), n_boot, seed)
                            entry[nm] = {"point": p, "lo": lo, "hi": hi}
                        d = entry["diff_B_minus_A"]
                        entry["separated_from_0"] = bool(d["lo"] > 0 or d["hi"] < 0)
                    fam_out[k] = entry
            fam_out.update(newc.get(fam, {}))
            for (f2, k), (va, ci) in ptA.items():
                if f2 != fam:
                    continue
                if k in fam_out:
                    continue          # a paired entry from the per-window export supersedes the point-only row
                if fam == "longitudinal" and have_pt and k.startswith("distance_keeping.mean_"):
                    continue  # the per-token rows above supersede these aggregates
                fam_out[k] = {"status": "NO-PER-TOKEN-DATA", "reason": NO_PT_REASON, "A_point_from_json": va,
                              "B_point_from_json": (ptB.get((f2, k)) or (None,))[0], "A_own_ci_from_json": ci,
                              "note": "descriptive only: unpaired JSON-level values; no paired difference claimed"}
            fres["families"][fam] = fam_out
        fres["families"]["strategic"] = dict(STRATEGIC_PAUSED)
        res["frames"][fr] = fres
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--tokens", help="lines log:token (A, and B unless --tokens-b)")
    ap.add_argument("--tokens-b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=B_DRAWS)
    a = ap.parse_args()
    ja = json.load(open(a.a, encoding="utf-8"))
    jb = json.load(open(a.b, encoding="utf-8"))
    tok = read_tokens(a.tokens) if a.tokens else None
    tokb = read_tokens(a.tokens_b) if a.tokens_b else None
    try:
        res = compare(ja, jb, tok, tokb, a.n_boot)
    except TokenMismatch as e:
        print("REFUSED:", e, file=sys.stderr)
        return 2
    res["inputs"] = {"a": a.a, "b": a.b, "tokens": a.tokens}
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, indent=1, default=str)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
