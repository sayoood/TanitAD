#!/usr/bin/env python3
"""C1r read (PREREG_C1R_READ.md, sha256 af358db92d965fa86edb7297d32275ad2e0bfc092c18fb405adf10555378774c): C1r (seed 1) vs C1 (seed 0).
Mechanics only: prints M1-M4 and every card control and the committed reading. Reuses p5c_read.{load_rows,per_token,Boot} (the P5-C read's own estimator:
paired log-cluster bootstrap, 10,000 draws, seed 20260927, 95 % percentile, full_set means). Tier T1 (NAVSIM v1.1 LQR-tracked). Evidence: MEASURED.
Names: --new (C1r), --ref (C1), --c0 (C0). Rehearsal: --rehearsal TEXT runs on non-C1r data and labels everything REHEARSAL.
"""
from __future__ import annotations

import argparse, glob, hashlib, json, math, os, sys
import numpy as np

B_BAR, D_C1_C0_COMMITTED, GOAL = 0.7031, 1.1702, 94.55
SCORER_FT_TC = -0.9015
EXPECT = {"C1": "f09949e2a5f2f9661ad47457e77ddfd10f49117d64e095487809c1d04f7a34a4",
          "C1r": "6e39a57c9a2e3ead0fef2b0d00d1684a708c901b6602ffe0717444f6180654a8"}
SUBS = ("nc", "dac", "ttc", "ep", "c", "ddc")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def flags(argv):
    d, cur = {}, None
    for t in argv[1:]:
        if t.startswith("--"):
            cur = t
            d[cur] = []
        elif cur:
            d[cur].append(t)
    return {k: " ".join(v) for k, v in d.items()}


def sep(c):
    return bool(c["lo"] > 0 or c["hi"] < 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p5c-dir", required=True)
    ap.add_argument("--rows-new", required=True, help="glob: rows with arms NEW and REF")
    ap.add_argument("--rows-c0", required=True, help="glob: rows with arm C0 (and REF, for the repro cross-check)")
    ap.add_argument("--new", default="C1r"); ap.add_argument("--ref", default="C1"); ap.add_argument("--c0", default="C0")
    ap.add_argument("--rehearsal", default=None)
    ap.add_argument("--fam-ci", default=None); ap.add_argument("--tactical", default=None)
    ap.add_argument("--strategic", default=None, help="strategic_family.py read json, NEW vs REF")
    ap.add_argument("--strategic-c0", default=None, help="NEW vs C0 (descriptive M4)")
    ap.add_argument("--identity", nargs="*", default=[])
    ap.add_argument("--seam-reports", nargs="*", default=[], help="NEW=report.json REF=report.json")
    ap.add_argument("--seams", nargs="*", default=[], help="NEW=seam.npz REF=seam.npz")
    ap.add_argument("--models", nargs="*", default=[], help="NEW=model.pt REF=model.pt")
    ap.add_argument("--cfg-new"); ap.add_argument("--cfg-ref")
    ap.add_argument("--code-ctrl", default=None, help="json with 'decision' NEUTRAL|SEED+CODE and evidence")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.p5c_dir)
    import p5c_read as R
    reh = a.rehearsal
    N, F, C = a.new, a.ref, a.c0

    # ---- data
    rn, bad1 = R.load_rows(sorted(glob.glob(a.rows_new)))
    ro, bad2 = R.load_rows(sorted(glob.glob(a.rows_c0)))
    if bad1 or bad2:
        raise SystemExit(f"REFUSED: error rows {bad1[:3]} {bad2[:3]}")
    toks = sorted(set(rn) & set(ro)) if set(rn) != set(ro) else sorted(rn)
    set_ok = set(rn) == set(ro) and len(toks) == 12146
    merged = {}
    for t in toks:
        arms = {}
        for nm, src in ((N, rn), (F, rn), (C, ro)):
            arms[nm] = src[t]["arms"][nm]
        arms["F_old"] = ro[t]["arms"][F]
        merged[t] = {"log": rn[t]["log"], "arms": arms}
        assert rn[t]["log"] == ro[t]["log"]
    P = R.per_token(merged, [N, F, C, "F_old"])
    bt = R.Boot(P["logs"])
    n_tok, n_log = len(P["tokens"]), bt.L
    T = lambda arm, m: P[arm]["tracked"][m]  # noqa: E731

    # ---- M1, M2, M4
    d_seed = bt.ci(T(N, "pdms") - T(F, "pdms"))
    lvl = {m: bt.ci(T(N, m)) for m in ("pdms",) + SUBS}
    d_sub = {m: bt.ci(T(N, m) - T(F, m)) for m in SUBS}
    m4 = bt.ci(T(N, "pdms") - T(C, "pdms"))
    m4_sub = {m: bt.ci(T(N, m) - T(C, m)) for m in SUBS}
    d_c1c0 = bt.ci(T(F, "pdms") - T(C, "pdms"))
    lvl_ref = bt.ci(T(F, "pdms"))
    transp = {"newminusref_transparent": bt.ci(P[N]["transparent"]["pdms"] - P[F]["transparent"]["pdms"]),
              "label": "no tracker in this number; descriptive"}

    # ---- controls
    ctrl = {}
    # CTRL-ID: REF read against itself through the same code path
    idc = {m: bt.ci(T(F, m) - T(F, m)) for m in R.METRICS}
    ctrl["CTRL-ID"] = {"pass": all(v["point"] == 0.0 and v["lo"] == 0.0 and v["hi"] == 0.0 for v in idc.values()), "n_metrics": len(idc)}
    # CTRL-REPRO (extra): REF - C0 reproduces the P5-C read; new REF rows == old REF rows
    mx = 0.0
    for s in R.SCORERS:
        for m in R.METRICS:
            mx = max(mx, float(np.max(np.abs(P[F][s][m] - P["F_old"][s][m]))))
    rep_ok = abs(d_c1c0["point"] - D_C1_C0_COMMITTED) < 5e-5 and mx == 0.0
    ctrl["CTRL-REPRO"] = {"pass": bool(rep_ok), "extra": True, "d_ref_minus_c0": d_c1c0, "committed_P5C": D_C1_C0_COMMITTED,
                          "max_abs_diff_new_REF_rows_vs_P5C_REF_rows": mx}
    # CTRL-SET
    seamset = {}
    rows_ok = True
    for s in a.seam_reports:
        k, p = s.split("=", 1)
        j = json.load(open(p, encoding="utf-8"))
        seamset[k] = {"rows": j["rows"], "misses": j["misses"], "tokens_asked": j["tokens_asked"]}
        rows_ok &= j["rows"] == 12146 and j["misses"] == 0
    tsets = {}
    for s in a.seams:
        k, p = s.split("=", 1)
        tsets[k] = sorted(str(t) for t in np.load(p, allow_pickle=False)["token"])
    same = len({tuple(v) for v in tsets.values()}) == 1 if tsets else False
    ctrl["CTRL-SET"] = {"pass": bool(set_ok and rows_ok and same and n_tok == 12146 and n_log == 136 and tsets and tsets[N] == P["tokens"] if tsets.get(N) else False),
                        "reports": seamset, "token_sets_identical": same, "n_tokens": n_tok, "n_logs": n_log}
    # CTRL-T
    ids = []
    for p in a.identity:
        j = json.load(open(p, encoding="utf-8"))
        ids.append({"arm": j["arm"], "verdict": j["verdict"], "n": j["n_scored"], "n_over_bar": j["n_over_bar"], "bar": j["bar"], "max_abs_diff": max(j["max_abs_diff"].values())})
    ctrl["CTRL-T"] = {"pass": bool(len(ids) >= 2 and all(i["verdict"] == "PASS" and i["n"] == 12146 and i["bar"] <= 1e-6 for i in ids)), "items": ids}
    # CTRL-SHA
    shas = {}
    for s in a.models:
        k, p = s.split("=", 1)
        if p.startswith("chainlog:"):      # the model file is gone; the chain that ran its seam passed --expect-ckpt-sha256 and rc 0
            txt = open(p[9:], encoding="utf-8", errors="replace").read()
            i = txt.find("RUN s2_c1:")
            seg = txt[i:i + 4000]
            ok_ = ("--expect-ckpt-sha256 " + EXPECT[k]) in seg and "rc 0 after" in seg
            shas[k] = EXPECT[k] if ok_ else "NOT-VERIFIED"
            shas[k + "_source"] = "p7_read_chain --expect-ckpt-sha256 + rc 0 in " + p[9:] + " (model file no longer on disk)"
        else:
            shas[k] = sha256_file(p)
    ctrl["CTRL-SHA"] = {"pass": bool(shas and all(shas.get(k) == EXPECT[k] for k in EXPECT)), "sha256": shas}
    # CTRL-ARGV
    if a.cfg_new and a.cfg_ref:
        cn, cf = json.load(open(a.cfg_new)), json.load(open(a.cfg_ref))
        fn, ff = flags(cn["argv"]), flags(cf["argv"])
        dif = sorted(k for k in set(fn) | set(ff) if fn.get(k) != ff.get(k))
        s0n, s0f = cn["argv"][0], cf["argv"][0]
        tail_eq = s0n.split("/", 5)[-1].split("/", 1)[-1] == s0f.split("/", 5)[-1].split("/", 1)[-1] if False else (s0n.rsplit("/refe/", 1)[1] == s0f.rsplit("/refe/", 1)[1])
        ckeys = sorted(k for k in set(cn) | set(cf) if k not in ("argv", "source_sha256", "started_at") and cn.get(k) != cf.get(k))
        ctrl["CTRL-ARGV"] = {"pass": bool(dif == ["--out", "--seed"] and fn["--seed"] == "1" and ff["--seed"] == "0" and tail_eq),
                             "differing_flags": dif, "values": {k: [ff.get(k), fn.get(k)] for k in dif},
                             "argv0": [s0f, s0n], "argv0_same_script_name": tail_eq, "config_keys_differing_other_than_argv": ckeys,
                             "config_rank_keys_new_only_in_C1r": {k: cn[k] for k in cn if k.startswith("rank_") and k not in cf},
                             "source_sha256": {"C1": cf.get("source_sha256"), "C1r": cn.get("source_sha256")}}
    else:
        ctrl["CTRL-ARGV"] = {"pass": False, "reason": "no configs supplied"}
    # CTRL-CODE
    if a.code_ctrl:
        cc = json.load(open(a.code_ctrl, encoding="utf-8"))
        ctrl["CTRL-CODE"] = {"pass": cc.get("decision") == "NEUTRAL", **cc}
    else:
        ctrl["CTRL-CODE"] = {"pass": False, "reason": "not supplied"}
    ctrl_fail = [k for k, v in ctrl.items() if not v["pass"] and not v.get("extra")]
    code_stamp = (ctrl["CTRL-CODE"].get("decision") == "SEED+CODE")
    valid = not ctrl_fail
    stamp = "REHEARSAL: " + reh if reh else ("VALID" if valid else ("SEED+CODE (no seed-floor claim; numbers descriptive)" if code_stamp and ctrl_fail == ["CTRL-CODE"] else "INVALID (numbers descriptive): " + ",".join(ctrl_fail)))

    # ---- committed reading
    d, F_ = d_seed["point"], abs(d_seed["point"])
    reading = "R-SEP" if sep(d_seed) else "R-NULL"
    thr = max(B_BAR, 2 * F_)
    a_lbl = "inside the REFe seed floor" if D_C1_C0_COMMITTED <= 2 * F_ else "exceeds 2*F_REFe"
    a_line = (f"(a) C1 - C0 {D_C1_C0_COMMITTED:+.4f} vs max(B={B_BAR}, 2F={2*F_:.4f}) = {thr:.4f}: "
              f"{'1.1702 <= 2F: ' + a_lbl if D_C1_C0_COMMITTED <= 2*F_ else '1.1702 > 2F (' + a_lbl + ')'}; vs B: {'above' if D_C1_C0_COMMITTED > B_BAR else 'not above'} B. Descriptive; P5-C GUARD-FAIL unchanged.")
    goal = "BELOW GOAL" if lvl["pdms"]["point"] < GOAL else "AT/ABOVE GOAL (level half only)"
    reading_blk = {"reading": reading, "d_seed": d_seed, "F_REFe": F_, "F_note": "ONE draw, 1 degree of freedom", "consequence_a": a_line,
                   "consequence_b": f"F_REFe={F_:.4f} printed beside frozen cards: SCORER-FT T - C {SCORER_FT_TC:+.4f} ({'|T-C| > F' if abs(SCORER_FT_TC) > F_ else '|T-C| <= F'}); field F-C and P-C: F_REFe qualifier only, bars unchanged; cards frozen after this read use max(F, F_REFe) at 2F terms",
                   "consequence_c": "C1r is never adopted on this read (selection on noise)", "goal": {"goal": GOAL, "C1r_level": lvl["pdms"], "status": goal},
                   "H_ESTIM_SEED_1_REFE": ("CONFIRMED on this rig (seed alone separates)" if reading == "R-SEP" else "no separated seed effect at this n") + ("" if valid else " -- but run stamp: " + stamp)}

    # ---- families (read, not re-derived)
    fam = {}
    if a.fam_ci:
        fj = json.load(open(a.fam_ci, encoding="utf-8"))
        for frame in ("executed_tracked", "executed_transparent", "reference"):
            fr = (fj.get("frames") or {}).get(frame)
            if not fr:
                continue
            fam[frame] = {}
            for fname, comps in fr["families"].items():
                if not isinstance(comps, dict):
                    continue
                rows = {}
                for k, e in comps.items():
                    if isinstance(e, dict) and e.get("status") == "OK":
                        dd = e["diff_B_minus_A"]
                        rows[k] = {"diff_new_minus_ref": dd, "n_tokens": e.get("n_tokens"), "n_logs": e.get("n_logs"), "separated": sep(dd)}
                    elif isinstance(e, dict):
                        rows[k] = {"status": e.get("status"), "reason": e.get("reason")}
                fam[frame][fname] = rows
    tact = json.load(open(a.tactical, encoding="utf-8")) if a.tactical else None
    strat = json.load(open(a.strategic, encoding="utf-8")) if a.strategic else None
    strat0 = json.load(open(a.strategic_c0, encoding="utf-8")) if a.strategic_c0 else None

    out = {"instrument": "c1r_read.py", "card": "PREREG_C1R_READ.md af358db92d965fa86edb7297d32275ad2e0bfc092c18fb405adf10555378774c", "run_stamp": stamp,
           "tier": "T1 (NAVSIM v1.1, LQR-tracked)", "evidence": "MEASURED (this run); one seed per arm",
           "estimator": "paired log-cluster bootstrap, 10,000 draws, seed 20260927, 95 % percentile, full_set means; answers 'another draw of LOGS' only",
           "n_tokens": n_tok, "n_logs": n_log, "units": "PDMS and sub-scores x 100",
           "M1_d_seed_tracked_pdms": d_seed, "M1_subscore_deltas": d_sub, "M2_C1r_level": lvl, "M2_C1_level_pdms": lvl_ref,
           "M3_families": {"longitudinal_lateral_tactical_components_fam_ci": fam, "tactical_command_adherence": (tact or {}).get("command_adherence"),
                           "strategic": strat, "note": "tactical roles: 'C1'=C1r, 'C0'=C1, 'base'=C0 in tactical.json; strategic base=C1"},
           "M4_descriptive_C1r_minus_C0": m4, "M4_subscores": m4_sub, "C1_minus_C0_repro": d_c1c0, "transparent_descriptive": transp,
           "reading": reading_blk, "controls": ctrl, "strategic_vs_C0": strat0}
    json.dump(out, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1, default=float)

    # ---- text report
    f = lambda c: f"{c['point']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]"  # noqa: E731
    g = lambda c: f"{c['point']:.4f} [{c['lo']:.4f}, {c['hi']:.4f}]"  # noqa: E731
    L = [f"C1R-READ  stamp: {stamp}", f"tier T1; n={n_tok} tokens, {n_log} logs; paired log-cluster bootstrap 10,000 draws seed 20260927 95% percentile; MEASURED; one seed per arm",
         f"M1  d_seed = PDMS({N}) - PDMS({F}) = {f(d_seed)}   CI {'EXCLUDES' if sep(d_seed) else 'contains'} 0  -> {reading}",
         f"M2  {N} PDMS {g(lvl['pdms'])}  (" + "  ".join(f"{m.upper()} {g(lvl[m])}" for m in SUBS) + ")",
         f"    {N}-{F} sub-scores: " + "  ".join(f"{m.upper()} {f(d_sub[m])}" for m in SUBS),
         f"M4  descriptive, cross-seed, NOT seed-matched: {N} - {C} = {f(m4)}   ({F} - {C} = {f(d_c1c0)})",
         f"F_REFe = |d_seed| = {F_:.4f} (ONE draw, 1 df)", a_line, reading_blk["consequence_b"], reading_blk["consequence_c"],
         f"goal {GOAL}: {N} {g(lvl['pdms'])} -> {goal}", "M3 families (executed_tracked frame, new - ref; separated = CI excludes 0; each with n):"]
    for fname, comps in (fam.get("executed_tracked") or {}).items():
        sepl = [k for k, e in comps.items() if e.get("separated")]
        L.append(f"  {fname.upper()}: {len(comps)} components, {len(sepl)} separated")
        for k, e in comps.items():
            if "diff_new_minus_ref" in e:
                dd = e["diff_new_minus_ref"]
                L.append(f"    {k}: {f(dd)} n={e['n_tokens']} logs={e['n_logs']}{' SEPARATED' if e['separated'] else ''}")
            else:
                L.append(f"    {k}: {e.get('status')} {e.get('reason')}")
    if tact:
        ca = tact["command_adherence"]
        L.append(f"  TACTICAL command adherence status {ca['status']}: turn-active accuracy {N} {ca['arms']['C1']['turn_active']['accuracy']['point']:.4f} ci95 {ca['arms']['C1']['turn_active']['accuracy']['ci95']}; paired vs {F}: "
                 + json.dumps(ca["paired_C1_minus_x"]["C1-C0"]["turn_active"]["accuracy_delta"]) + "; selection quality / goal-setting: " + tact["goal_setting"]["status"] + f" ({tact['goal_setting']['reason'][:80]}... n={tact['goal_setting']['n']})")
    if strat:
        pk = list(strat["paired_vs_base"])[0]
        sd = strat["paired_vs_base"][pk]
        L.append(f"  STRATEGIC (unpaused 2026-11-16) paired {pk}: S_DEC turn-active accuracy delta {json.dumps(sd['S_DEC']['turn_active']['accuracy_delta'])} n={sd['S_DEC']['turn_active']['n']}")
        for sub, blk in sd["S_GOAL"].items():
            if isinstance(blk, dict) and "endpoint_error_m" in blk:
                L.append(f"    S_GOAL {sub} n={blk['n']}: endpoint_error_m {json.dumps(blk['endpoint_error_m'])}  route_progress_ratio {json.dumps(blk['route_progress_ratio'])}")
    L.append("CONTROLS: " + "  ".join(f"{k}={'PASS' if v['pass'] else 'FAIL'}" for k, v in ctrl.items()))
    for k, v in ctrl.items():
        L.append(f"  {k}: {json.dumps(v, default=float)[:900]}")
    open(a.out.replace(".json", ".txt").replace("read_", "report_"), "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    v1 = (f"C1R-READ {stamp} n={n_tok} logs={n_log} d_seed={f(d_seed)} {reading} F_REFe={F_:.4f} C1r_PDMS={g(lvl['pdms'])} goal={goal} "
          f"C1-C0 {D_C1_C0_COMMITTED:+.4f} {'inside' if D_C1_C0_COMMITTED <= 2*F_ else 'outside'} 2F; C1r NOT ADOPTED; controls " + ",".join(f"{k}={'PASS' if v['pass'] else 'FAIL'}" for k, v in ctrl.items()))
    open(os.path.join(os.path.dirname(os.path.abspath(a.out)), "VERDICT_c1r.txt"), "w", encoding="utf-8", newline="\n").write(v1 + "\n")
    print("\n".join(L[:14])); print(v1)


if __name__ == "__main__":
    main()
