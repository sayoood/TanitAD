#!/usr/bin/env python3
"""r9_e2_verdict.py -- the refcv9 E2 GO / NO-GO JUDGE (REFCV9_DESIGN 4.4): K2 vs K0 on DEV at inference seeds 0 AND 1.  Pure JSON in, JSON + md out; no GPU.

    python r9_e2_verdict.py --inputs INPUTS.json --out-dir <dir>      # -> E2_VERDICT.json + E2_VERDICT.md   (rc 0 GO, rc 1 NO-GO, rc 2 REFUSED)

GO iff, at EVERY inference seed, ALL FIVE clauses PASS (REFCV9_DESIGN 4.4, verbatim in ``RULE``):
  (a) MAP      K2 beats K0 in POINT estimate on the map primary ``nf_swedge_prec_lv_5_20`` (higher better; ``r9_e1_eval.py`` single-arm eval JSON,
               ``nearfield.ci.<key>.point``; T0 label level).
  (b) BOX      K2 NOT SEPARATED WORSE than K0 on box AP@2 m, head output, NO post-hoc NMS (PI 2026-11-17, "Not separated worse"; T0): PASS unless the PAIRED
               episode-cluster bootstrap interval of K2 - K0 on ``boxes.ap2m_all`` lies ENTIRELY below 0 (higher is better).  Read from the per-seed ``box_paired`` record
               (``r9e2_box_paired.py`` -> BOX_PAIRED.json, ``{delta, lo, hi, separated, better:"higher"}``); the point delta is reported as information.  An absent / non-finite
               record is NOT-COMPUTABLE -- never a silent fall-back to the old point rule.  Banked verdicts produced under the old point rule are NOT re-scored.
  (c) S6c      K2 beats K0 (LOWER) on the planner's sidewalk rate on S6c-DEV (``rate``, or MAPSEL's ``map_compliance.S6c.arms.BASE.off_drivable.rate``; T1).
  (d) FAMILIES no four-family row is SEPARATED WORSE (K2 - K0, paired episode-cluster bootstrap over EPISODES; T1).  LONGITUDINAL / LATERAL / TACTICAL are
               read row by row.  STRATEGIC is REPORT-ONLY (PI 2026-11-16, strategic switched back on): supplied rows are read and shown BESIDE
               the rule and never enter (a)-(e); NOT-SUPPLIED when the input has none; never a pass, never omitted (PREREG_K2DC02CL section 1: the rule is unchanged).
               A required family with no rows is NOT-COMPUTABLE.
  (e) W4       a partial W4 preview: K2 vs its CAMERA-BLIND is separated BETTER on ``ade_m`` and each of the three families' primary row (T1).
A clause that is NOT-COMPUTABLE (an input absent / non-finite / unaligned) is NO-GO -- never silently dropped.  NO-GO NAMES every failing clause and
seed (Rule Zero: the verdict names the next lever, it does not end the programme item) -- ``next_lever`` carries the per-clause pointer.

NO NEW ESTIMATOR.  A row is either a precomputed paired record ``{delta, lo, hi, separated, better}`` (``taniteval.ci.paired_*`` output) or the per-window
form ``{a, b, eid, better}`` which is scored HERE by ``taniteval.ci.paired_episode_cluster_bootstrap`` (B 2000, seed 0).  ``delta`` is always ARM - REFERENCE;
``better`` is "higher" or "lower" (the direction in which delta is an improvement).

H-ESTIM-SEED-1.  Perception clauses (a), (b) are deterministic at inference: seeds 0 and 1 are read separately and, if the two inputs are byte-identical, the
record says so (``identical_inputs``) instead of presenting two draws.  The paired interval answers *another draw of EPISODES* only.

INPUTS.json::
    {"k2_label": "K2", "k0_label": "K0",
     "seeds": {"0": {"eval_k2": p, "eval_k0": p, "s6c_k2": p, "s6c_k0": p, "families": p, "camblind": p, "box_paired": p},
               "1": {...}}}
``families`` = ``{"rows": {"LONGITUDINAL": {row_id: row}, "LATERAL": {...}, "TACTICAL": {...}, "STRATEGIC": {...optional, REPORT-ONLY, outside clause (d)}}}``;
``camblind`` = ``{"rows": {"ade_m": row, "LONGITUDINAL": row, "LATERAL": row, "TACTICAL": row}}`` (delta = K2 - CAMERA-BLIND).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

SCHEMA = "r9e2-verdict-v1"
MAP_KEY = "nf_swedge_prec_lv_5_20"
FAMILIES = ("LONGITUDINAL", "LATERAL", "TACTICAL")
STRATEGIC_REPORT_ONLY = ("REPORT-ONLY: the strategic layer was switched back on by the PI on 2026-11-16 and is reported in full; clause (d) was pre-registered over "
                         "LONGITUDINAL / LATERAL / TACTICAL and stays exactly that (PREREG_K2DC02CL_2026-11-22 section 1, 'unchanged'), so STRATEGIC rows are shown "
                         "BESIDE the rule and never enter clauses (a)-(e); never a pass, never omitted.")
STRATEGIC_PRODUCER = ("taniteval/tools/openloop_suite.py:943 (_strategic_block: the route head vs the v2.1 route label under nav true / shuffled / zero; headline at :1140) "
                      "and taniteval/tools/refcv3_arm.py:2646 (_navcomp_row)")
STRATEGIC_NOT_SUPPLIED = ("NOT-SUPPLIED: the families input carried no STRATEGIC rows (never a pass, never omitted); the strategic numbers are computed by " + STRATEGIC_PRODUCER
                          + " and are not yet paired {delta, lo, hi, separated, better} rows, so none reach this reader; reported BESIDE clause (d), outside it.")
STRATEGIC_WORSE = "separated worse (report-only, outside clause (d))"
CLAUSES = (("a", "MAP primary nf_swedge_prec_lv_5_20: K2 > K0 (point)"), ("b", "box AP@2 m head output no NMS: K2 not separated worse than K0 (paired, episodes)"),
           ("c", "planner sidewalk rate S6c-DEV: K2 < K0 (point)"), ("d", "no four-family row separated worse (paired, episodes)"),
           ("e", "W4 preview: K2 separated BETTER than camera-blind on ade_m + each family primary"))
TIER = {"a": "T0 (label level)", "b": "T0 (label level)", "c": "T1 self-action open loop (S6c-DEV)", "d": "T1 self-action open loop", "e": "T1 self-action open loop"}
EST = {"a": "single-arm point (episode-cluster bootstrap in the eval JSON)", "b": "taniteval.ci paired episode-cluster bootstrap, B 2000 seed 0, over EPISODES (not separated worse; point delta informational)", "c": "point (S6c rate)",
       "d": "taniteval.ci paired episode-cluster bootstrap, B 2000 seed 0, over EPISODES", "e": "taniteval.ci paired episode-cluster bootstrap, B 2000 seed 0, over EPISODES"}
NEXT = {"a": "map primary lever: M1/M3 representation + w_map share (REFCV9_DESIGN 4.3), re-read nf_swedge_prec_lv_5_20",
        "b": "box lever: B2 perspective aux / B3 quality-aware presence + DN / B4 queries (REFCV9_DESIGN 4.3), re-read AP@2 m",
        "c": "planner wiring: W-MAP-2 projection / W-BOX tokens + gates (C8), re-read S6c sidewalk rate",
        "d": "the separated-worse row(s) named in the clause: diagnose that family before the 214 h are spent",
        "e": "the planner does not read the camera beyond what the camera-blind already gets: W4 fails => wiring / grounding lever (C8) first"}
RULE = ("GO iff K2 beats K0 in point estimate on the map primary and the planner's S6c sidewalk rate, K2 is not separated worse than K0 on box AP@2 m, "
        "no four-family row is separated worse, and a partial W4 preview (K2 vs camera-blind) is separated -- at inference seeds 0 and 1 (REFCV9_DESIGN 4.4) "
        "(clause (b) amended by the PI 2026-11-17: not separated worse).")


def _ci():
    try:
        from taniteval import ci
    except ModuleNotFoundError:
        here = Path(__file__).resolve()
        for cand in (here.parents[2] / "taniteval", here.parents[1] / "taniteval"):
            if (cand / "taniteval" / "ci.py").is_file():
                sys.path.insert(0, str(cand))
                break
        from taniteval import ci
    return ci


def _fin(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _load(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")), hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except (OSError, ValueError, TypeError):
        return None, None


def _dig(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _clause(cid, status, detail, **kw) -> dict:
    return {"id": cid, "name": dict(CLAUSES)[cid], "status": status, "detail": detail, "tier": TIER[cid], "estimator": EST[cid], **kw}


def point_clause(cid, k2, k0, higher_better, where) -> dict:
    """K2 beats K0 in POINT estimate (strict).  Absent / non-finite on either arm = NOT-COMPUTABLE."""
    if not (_fin(k2) and _fin(k0)):
        return _clause(cid, "NOT-COMPUTABLE", f"{where}: K2={k2!r} K0={k0!r} (absent / non-finite)", k2=k2, k0=k0)
    win = (k2 > k0) if higher_better else (k2 < k0)
    return _clause(cid, "PASS" if win else "FAIL", f"{where}: K2={k2:.6g} K0={k0:.6g} delta={k2 - k0:+.6g} ({'higher' if higher_better else 'lower'} is better)",
                   k2=k2, k0=k0, delta=k2 - k0)


def box_paired_clause(rec) -> dict:
    """Clause (b), PI 2026-11-17: PASS unless the paired interval of K2 - K0 on box AP@2 m is ENTIRELY below 0 (higher is better).  Absent / non-finite / wrong-direction = NOT-COMPUTABLE."""
    if (not isinstance(rec, dict) or rec.get("better") != "higher" or rec.get("undefined") or not all(_fin(rec.get(k)) for k in ("delta", "lo", "hi"))
            or rec["lo"] > rec["hi"]):
        return _clause("b", "NOT-COMPUTABLE", "box_paired record absent / non-finite / undefined / not `better: higher` (no fall-back to the point rule)",
                       record=rec if isinstance(rec, dict) else None)
    worse = rec["hi"] < 0
    return _clause("b", "FAIL" if worse else "PASS",
                   f"box AP@2 m K2 - K0: delta={rec['delta']:+.6g} paired CI [{rec['lo']:.6g}, {rec['hi']:.6g}] -> "
                   + ("SEPARATED WORSE (interval entirely below 0)" if worse else "not separated worse (interval reaches 0 or above)")
                   + f"; K2 higher by point={rec['delta'] > 0} (information only){'; degenerate interval' if rec.get('degenerate') else ''}",
                   delta=rec["delta"], lo=rec["lo"], hi=rec["hi"], separated_worse=worse, separated_better=bool(rec["lo"] > 0))


def row_read(row, boot=2000, seed=0) -> dict:
    """Normalise a row to {delta, lo, hi, separated, better, n} -- precomputed, or scored by taniteval.ci's paired bootstrap from per-window arrays."""
    if not isinstance(row, dict) or row.get("better") not in ("higher", "lower"):
        return {"ok": False, "why": "row absent or has no `better` direction"}
    if "a" in row and "b" in row and "eid" in row and isinstance(row["a"], list):
        try:
            r = _ci().paired_episode_cluster_bootstrap(row["a"], row["b"], row["eid"], n_boot=boot, seed=seed)
        except Exception as e:                       # noqa: BLE001  (an unaligned / empty row is NOT-COMPUTABLE, reported with its reason)
            return {"ok": False, "why": f"paired bootstrap refused: {e}"}
        d = {"delta": r["delta"], "lo": r["lo"], "hi": r["hi"], "separated": r["separated"], "n": r["n_windows"], "n_episodes": r["n_episodes"], "src": "ci.paired_episode_cluster_bootstrap"}
    else:
        d = {k: row.get(k) for k in ("delta", "lo", "hi", "separated", "n", "n_episodes")}
        d["src"] = "precomputed paired record"
        if not (_fin(d["delta"]) and isinstance(d["separated"], bool)):
            return {"ok": False, "why": "row has no finite delta / boolean `separated`"}
    if row.get("undefined") or row.get("degenerate"):
        return {"ok": False, "why": "row undefined / degenerate (float-resolution interval): not evidence"}
    good = (d["delta"] > 0) if row["better"] == "higher" else (d["delta"] < 0)
    d.update(ok=True, better=row["better"], separated_worse=bool(d["separated"] and not good), separated_better=bool(d["separated"] and good))
    return d


def strategic_report(sr) -> dict:
    """STRATEGIC family: REPORT-ONLY (rows read with the SAME `row_read`, shown beside the rule, never in clause (d)) or NOT-SUPPLIED.  Only dict-valued entries are rows;
    a stamp such as {"status": ..., "reason": ...} carries no rows and reads NOT-SUPPLIED."""
    got = {k: v for k, v in sr.items() if isinstance(v, dict)} if isinstance(sr, dict) else {}
    if not got:
        return {"status": "NOT-SUPPLIED", "reason": STRATEGIC_NOT_SUPPLIED, "rows": {}}
    out = {rid: row_read(r) for rid, r in got.items()}
    worse = [rid for rid, v in out.items() if v.get("separated_worse")]
    for rid in worse:
        out[rid]["report_only_note"] = STRATEGIC_WORSE
    return {"status": "REPORT-ONLY", "reason": STRATEGIC_REPORT_ONLY, "rows": out, "separated_worse_rows": worse,
            "unreadable_rows": [rid for rid, v in out.items() if not v["ok"]]}


def families_clause(doc) -> tuple[dict, dict]:
    """Clause (d) + the per-family report.  A required family with no rows is NOT-COMPUTABLE; any separated-worse row is FAIL."""
    rows = _dig(doc, "rows") or {}
    rep, worse, missing, bad = {}, [], [], []
    for fam in FAMILIES:
        fr = rows.get(fam) if isinstance(rows, dict) else None
        if not isinstance(fr, dict) or not fr:
            rep[fam] = {"status": "NOT-COMPUTABLE", "reason": "no rows for this family in the input", "rows": {}}
            missing.append(fam)
            continue
        out = {}
        for rid, r in fr.items():
            rr = row_read(r)
            out[rid] = rr
            if not rr["ok"]:
                bad.append(f"{fam}/{rid}: {rr['why']}")
            elif rr["separated_worse"]:
                worse.append(f"{fam}/{rid}")
        rep[fam] = {"status": "FAIL" if any(v.get("separated_worse") for v in out.values()) else
                    ("NOT-COMPUTABLE" if any(not v["ok"] for v in out.values()) else "PASS"), "rows": out}
    rep["STRATEGIC"] = strategic_report(rows.get("STRATEGIC") if isinstance(rows, dict) else None)
    if worse:
        c = _clause("d", "FAIL", "separated WORSE: " + ", ".join(worse), failing_rows=worse)
    elif missing or bad:
        c = _clause("d", "NOT-COMPUTABLE", "; ".join([f"family {m} has no rows" for m in missing] + bad), missing_families=missing, unreadable=bad)
    else:
        c = _clause("d", "PASS", "no LONGITUDINAL / LATERAL / TACTICAL row separated worse (STRATEGIC is reported beside, outside this clause; never a pass)")
    return c, rep


def w4_clause(doc) -> tuple[dict, dict]:
    rows = _dig(doc, "rows") or {}
    need = ("ade_m",) + FAMILIES
    rep, fail, miss = {}, [], []
    for k in need:
        rr = row_read(rows.get(k)) if isinstance(rows, dict) else {"ok": False, "why": "no rows"}
        rep[k] = rr
        if not rr["ok"]:
            miss.append(f"{k}: {rr['why']}")
        elif not rr["separated_better"]:
            fail.append(k)
    if fail:
        return _clause("e", "FAIL", "K2 is NOT separated better than camera-blind on: " + ", ".join(fail), failing_rows=fail), rep
    if miss:
        return _clause("e", "NOT-COMPUTABLE", "; ".join(miss), unreadable=miss), rep
    return _clause("e", "PASS", "K2 separated better than camera-blind on ade_m + LONGITUDINAL + LATERAL + TACTICAL primary rows"), rep


def judge_seed(paths: dict) -> dict:
    ev2, sh2 = _load(paths.get("eval_k2"))
    ev0, sh0 = _load(paths.get("eval_k0"))
    s2, _ = _load(paths.get("s6c_k2"))
    s0, _ = _load(paths.get("s6c_k0"))
    fam, _ = _load(paths.get("families"))
    cb, _ = _load(paths.get("camblind"))
    bp, _ = _load(paths.get("box_paired"))
    rate = lambda s: (s.get("rate") if isinstance(s, dict) and "rate" in s else                                   # noqa: E731
                      _dig(s, "map_compliance", "S6c", "arms", "BASE", "off_drivable", "rate"))
    cl = [point_clause("a", _dig(ev2, "nearfield", "ci", MAP_KEY, "point"), _dig(ev0, "nearfield", "ci", MAP_KEY, "point"), True, "nearfield.ci." + MAP_KEY + ".point"),
          box_paired_clause(bp),
          point_clause("c", rate(s2), rate(s0), False, "S6c-DEV sidewalk (off_drivable) rate")]
    cd, fam_rep = families_clause(fam)
    ce, w4_rep = w4_clause(cb)
    n_info = {"n_windows_boxes": _dig(ev2, "n_windows"), "n_clips": _dig(ev2, "n_clips"), "n_files_map": _dig(ev2, "nearfield", "n_files"), "n_s6c": (s2 or {}).get("n")}
    return {"clauses": cl + [cd, ce], "families": fam_rep, "w4_rows": w4_rep, "n": n_info, "identical_inputs": bool(sh2 and sh2 == sh0)}


def build(inputs: dict) -> dict:
    seeds = inputs.get("seeds") or {}
    if set(seeds) != {"0", "1"}:
        raise SystemExit(f"[r9-e2-verdict] REFUSED: inference seeds 0 AND 1 are both required, got {sorted(seeds)}")
    per = {s: judge_seed(seeds[s]) for s in ("0", "1")}
    clauses = []
    for cid, name in CLAUSES:
        ps = {s: next(c for c in per[s]["clauses"] if c["id"] == cid) for s in ("0", "1")}
        st = [ps[s]["status"] for s in ("0", "1")]
        status = "FAIL" if "FAIL" in st else ("NOT-COMPUTABLE" if "NOT-COMPUTABLE" in st else "PASS")
        clauses.append({"id": cid, "name": name, "status": status, "tier": TIER[cid], "estimator": EST[cid], "per_seed": ps,
                        "seeds_failing": [s for s in ("0", "1") if ps[s]["status"] != "PASS"]})
    bad = [c for c in clauses if c["status"] != "PASS"]
    verdict = "GO" if not bad else "NO-GO"
    return {"schema": SCHEMA, "rule": RULE, "verdict": verdict,
            "failing_clauses": [{"id": c["id"], "name": c["name"], "status": c["status"], "seeds": c["seeds_failing"], "next_lever": NEXT[c["id"]]} for c in bad],
            "clauses": clauses, "four_families": {s: per[s]["families"] for s in ("0", "1")}, "w4_rows": {s: per[s]["w4_rows"] for s in ("0", "1")},
            "n": {s: per[s]["n"] for s in ("0", "1")}, "identical_inputs_across_seeds_note": {s: per[s]["identical_inputs"] for s in ("0", "1")},
            "windows": "DEV (eval139 even half, DEV_HOLD_SPLIT.json); HOLD is never read here", "labels": inputs.get("k2_label", "K2") + " vs " + inputs.get("k0_label", "K0"),
            "strategic": STRATEGIC_REPORT_ONLY,
            "caveat": "H-ESTIM-SEED-1: the paired interval answers another draw of EPISODES, not another TRAINING run; K2 and K0 are ONE training run each."}


def render_md(v: dict) -> str:
    L = [f"# E2 verdict: {v['verdict']}  ({v['labels']})", "", v["rule"], "", "| clause | status | tier | seeds not PASS |", "|---|---|---|---|"]
    for c in v["clauses"]:
        L.append(f"| ({c['id']}) {c['name']} | **{c['status']}** | {c['tier']} | {','.join(c['seeds_failing']) or '-'} |")
    for f in v["failing_clauses"]:
        L += ["", f"NO-GO names clause ({f['id']}) {f['name']} [{f['status']}, seeds {f['seeds']}] -- next lever: {f['next_lever']}"]
    for s in ("0", "1"):
        L += ["", f"## inference seed {s}"]
        for c in v["clauses"]:
            L.append(f"* ({c['id']}) {c['per_seed'][s]['status']}: {c['per_seed'][s]['detail']}")
        for fam, r in v["four_families"][s].items():
            L.append(f"  * family {fam}: {r['status']}" + (f" -- {r['reason']}" if r.get("reason") else ""))
            if fam == "STRATEGIC":
                for rid, rr in r["rows"].items():
                    if not rr.get("ok"):
                        L.append(f"    * STRATEGIC/{rid}: unreadable ({rr.get('why')})")
                    else:
                        tag = STRATEGIC_WORSE if rr["separated_worse"] else ("separated better (report-only)" if rr["separated_better"] else "not separated (report-only)")
                        L.append(f"    * STRATEGIC/{rid}: delta={rr['delta']:+.6g} CI [{rr['lo']:.6g}, {rr['hi']:.6g}] separated={rr['separated']} -- {tag}")
    L += ["", f"STRATEGIC: {v['strategic']}", f"Windows: {v['windows']}", f"Caveat: {v['caveat']}"]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    inputs, _ = _load(a.inputs)
    if not isinstance(inputs, dict):
        print(f"[r9-e2-verdict] REFUSED: cannot read {a.inputs}", file=sys.stderr)
        return 2
    try:
        v = build(inputs)
    except SystemExit as e:
        print(e, file=sys.stderr)
        return 2
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "E2_VERDICT.json").write_text(json.dumps(v, indent=1, default=float) + "\n", encoding="utf-8")
    (out / "E2_VERDICT.md").write_text(render_md(v), encoding="utf-8")
    print(f"E2 verdict: {v['verdict']}" + ("" if v["verdict"] == "GO" else "  failing: " + ", ".join(f"({f['id']}) {f['status']} seeds {f['seeds']}" for f in v["failing_clauses"])))
    return 0 if v["verdict"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
