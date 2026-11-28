#!/usr/bin/env python3
"""TF-N1 readout (PREREG_TF_N1.md). Reads ONLY the rows `transp_score.py` wrote; no DB, no scorer.

Two steps, in this order:
  cert   python tf_n1_analyze.py cert --rows "<score>/rows_s*.jsonl" --known36 <frame_control_navtest_full.json>
             --out <score>/rows_cert.jsonl --out-primary <score>/rows_cert_primary.jsonl
         Adds the virtual arm CERT to every row: the first of [N1C, N1T, N1M] whose TRACKED `dac` == 1.0 (the scorer's own
         DAC, read from the same row), else N1C (section 4: the none-pass tie-break). `rows_cert_primary.jsonl` drops
         the 36 C1-known tokens, so families_run.py reads the 12,110.
  read   python tf_n1_analyze.py read --rows <score>/rows_cert.jsonl --build-report <build>/n1_build_report.json
             --identity-csv <banked teacher CSV> [--human-csv <banked human CSV>] --known36 <json>
             --pins N1_PINS.json --pin-root <dir> --proxy-eval <eval dir> [--families ARM=fam.json ...] --out n1_result.json
         Controls FIRST (C-COV, C-BUILD, C-PIN, C-EST, C-ID-H, C-REG). If any fails it prints ZZN1_CONTROLS FAIL and exits 1
         BEFORE any N1C / N1M / CERT aggregate is computed. Then the registered outcomes. Prints ZZN1_VERDICT lines.

Estimator (section 7): paired log-cluster bootstrap over the logs, 10,000 draws, seed 20260927, the draw scheme of
`eval/proxy_eval.boot` reproduced exactly (sorted tokens, groups by sorted log, one `rng.integers(0, G, G)` per draw),
95 % percentile interval, plus the Bonferroni interval at K = 41 (percentiles 2.5/41 and 100 - 2.5/41). C-EST checks
it against proxy_eval.boot itself. It answers "another draw of LOGS?" only: labels, tracker and scorer are deterministic.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np

ARMS_PLAN = ("N1T", "N1C", "N1W", "N1M")
ARMS_ALL = ARMS_PLAN + ("human", "CERT")
CERT_ORDER = ("N1C", "N1T", "N1M")                  # = tf2_analyze.cert order [Tc20s, T, M]
MET = ("nc", "dac", "ddc", "ep", "ttc", "c", "pdms")
CSV_COL = {"nc": "no_at_fault_collisions", "dac": "drivable_area_compliance", "ep": "ego_progress",
           "ttc": "time_to_collision_within_bound", "c": "comfort", "ddc": "driving_direction_compliance", "pdms": "score"}
PAPER = {"pdms": 0.958, "nc": 0.998, "dac": 0.999, "ttc": 0.991, "c": 0.990, "ep": 0.915}   # DriveRL Table 4 p. 11 (PUBLISHED)
PAPER_HUMAN_PDMS = 94.8
B, SEED, K = 10000, 20260927, 41
BAR = {"confirm_delta": 3.0, "confirm_dac": 98.5, "paper_pdms": 95.8, "paper_dac": 99.9, "near_pdms": 95.3, "near_dac": 99.5,
       "reg_margin": 10.0, "id_tol": 1e-6}


def read_rows(patterns) -> tuple[list, int]:
    rows, seen, n_err = [], set(), 0
    for pat in patterns:
        for fp in sorted(glob.glob(pat)):
            for ln in open(fp, encoding="utf-8"):
                if not ln.strip():
                    continue
                try:
                    r = json.loads(ln)
                except json.JSONDecodeError:
                    n_err += 1
                    continue
                if "error" in r:
                    n_err += 1
                    continue
                if r["token"] in seen:
                    continue
                seen.add(r["token"])
                rows.append(r)
    return sorted(rows, key=lambda r: r["token"]), n_err


def load_known36(p):
    return set(json.load(open(p, encoding="utf-8")).get("over_bar", {})) if p else set()


def cert_choice(r) -> tuple[str, bool]:
    for c in CERT_ORDER:
        if r["arms"][c]["tracked"]["dac"] == 1.0:
            return c, False
    return "N1C", True


def cmd_cert(a) -> int:
    rows, n_err = read_rows(a.rows)
    known = load_known36(a.known36)
    cnt: dict = {}
    n_none = 0
    with open(a.out, "w", encoding="utf-8") as fo, open(a.out_primary, "w", encoding="utf-8") as fp:
        for r in rows:
            ch, none = cert_choice(r)
            n_none += none
            cnt[ch] = cnt.get(ch, 0) + 1
            r["arms"]["CERT"] = dict(r["arms"][ch])
            r["cert_choice"], r["cert_none_pass"] = ch, none
            s = json.dumps(r) + "\n"
            fo.write(s)
            if r["token"] not in known:
                fp.write(s)
    print("ZZN1_CERT", json.dumps({"rows": len(rows), "error_lines": n_err, "choice": cnt, "none_pass": n_none}))
    return 0


class Boot:
    """proxy_eval.boot's draws, exactly: tokens sorted, groups = logs sorted, rng re-seeded, one integers(0, G, G) per draw."""

    def __init__(self, toks: list, tlog: dict, n: int = B, seed: int = SEED):
        self.toks = sorted(toks)
        by: dict = {}
        for i, t in enumerate(self.toks):
            by.setdefault(tlog[t], []).append(i)
        self.groups = [np.asarray(v) for _k, v in sorted(by.items())]
        G = len(self.groups)
        rng = np.random.default_rng(seed)
        self.pick = np.stack([rng.integers(0, G, G) for _ in range(n)])                 # [n, G]
        self.gid = np.empty(len(self.toks), np.int64)
        for g, ix in enumerate(self.groups):
            self.gid[ix] = g
        self.cnt = np.bincount(self.gid, minlength=G).astype(np.float64)
        self.den = self.cnt[self.pick].sum(1)

    def stat(self, per_token: dict, scale: float = 100.0) -> dict:
        v = np.asarray([per_token[t] for t in self.toks], np.float64)
        s = np.bincount(self.gid, weights=v, minlength=len(self.groups))
        means = s[self.pick].sum(1) / self.den
        qb = 100.0 * 0.025 / K
        lo, hi = np.percentile(means, [2.5, 97.5])
        blo, bhi = np.percentile(means, [qb, 100.0 - qb])
        return {"mean": round(scale * float(v.mean()), 4), "ci95": [round(scale * float(lo), 4), round(scale * float(hi), 4)],
                f"ci_bonferroni_K{K}": [round(scale * float(blo), 4), round(scale * float(bhi), 4)],
                "n": len(v), "n_logs": len(self.groups), "_raw": (float(v.mean()), float(lo), float(hi))}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_identity(a) -> dict:
    """-> {token: {met: value}} from a W3 harness CSV, or (smoke on navtrain) from TF rows: 'tfrows:<glob>:<field>'."""
    ref = {}
    if a.identity_csv:
        for row in csv.DictReader(open(a.identity_csv, encoding="utf-8")):
            if row["token"] == "average" or row.get("valid", "True") != "True":
                continue
            ref[row["token"]] = {m: float(row[CSV_COL[m]]) for m in MET}
    elif a.identity_ref:
        _, pat, field = a.identity_ref.split("|")
        for fp in sorted(glob.glob(pat)):
            for ln in open(fp, encoding="utf-8"):
                if ln.strip():
                    r = json.loads(ln)
                    if "error" not in r and field in r:
                        ref.setdefault(r["token"], {m: float(r[field][m]) for m in MET})
    return ref


def rows_from_harness(a) -> tuple[list, int]:
    """FALLBACK ROUTE ONLY (section 6, C-ID-H): rows rebuilt from pinned-W3-harness CSVs, tracked sub-scores only.
    --harness-csv N1T=<re-scored T seam csv> N1C=<csv> N1W=<csv> N1M=<csv>; human from --human-csv; logs from --token-log."""
    import gzip
    d = json.load(gzip.open(a.token_log, "rt", encoding="utf-8"))
    d = d.get("tokens", d)
    tlog = {t: (v["log_name"] if isinstance(v, dict) else v) for t, v in d.items()}
    srcs = dict(s.split("=", 1) for s in a.harness_csv)
    if a.human_csv:
        srcs["human"] = a.human_csv
    arms: dict = {}
    for nm, fp in srcs.items():
        arms[nm] = {}
        for row in csv.DictReader(open(fp, encoding="utf-8")):
            if row["token"] != "average" and row.get("valid", "True") == "True":
                arms[nm][row["token"]] = {m: float(row[CSV_COL[m]]) for m in MET}
    toks = sorted(set.intersection(*[set(v) for v in arms.values()]))
    n_err = max(len(v) for v in arms.values()) - len(toks)
    rows = []
    for t in toks:
        r = {"token": t, "log": tlog[t], "arms": {nm: {"tracked": arms[nm][t]} for nm in arms}}
        ch, none = cert_choice(r)
        r["arms"]["CERT"], r["cert_choice"], r["cert_none_pass"] = dict(r["arms"][ch]), ch, none
        rows.append(r)
    return rows, n_err


def cmd_read(a) -> int:
    rows, n_err = rows_from_harness(a) if a.harness_csv else read_rows(a.rows)
    FR = ("tracked",) if a.harness_csv else ("tracked", "transparent")
    known = load_known36(a.known36)
    tlog = {r["token"]: r["log"] for r in rows}
    allt = sorted(tlog)
    prim = [t for t in allt if t not in known]
    R = {r["token"]: r for r in rows}
    res: dict = {"prereg": a.prereg_sha256 or "UNSET (the MM writes the frozen sha256 here)", "rows": a.rows,
                 "n_rows": len(rows), "n_error_lines": n_err, "n_primary": len(prim), "controls": {}}
    C = res["controls"]

    # ---- C-COV (iii): coverage, every arm on every token
    arms_ok = all(all(x in R[t]["arms"] for x in ARMS_ALL) for t in allt)
    C["C-COV"] = {"pass": len(allt) == a.expect_n and n_err == 0 and arms_ok and len(prim) == a.expect_primary,
                  "tokens": len(allt), "expect": a.expect_n, "primary": len(prim), "expect_primary": a.expect_primary,
                  "error_lines": n_err, "every_arm_on_every_token": arms_ok}
    # ---- C-BUILD: the plan build passed its own controls (C-BUILD-COV, C-SEAM / C-DERIV)
    if a.build_report:
        br = json.load(open(a.build_report, encoding="utf-8"))
        C["C-BUILD"] = {"pass": br.get("verdict") == "PASS", "controls": {k: v.get("pass") for k, v in br["controls"].items()}}
    else:
        C["C-BUILD"] = {"pass": False, "why": "no --build-report"}
    # ---- C-PIN: the instruments are the frozen bytes
    if a.pins:
        pins = json.load(open(a.pins, encoding="utf-8"))["files"]
        got = {k: (sha256(os.path.join(a.pin_root, k)) if os.path.exists(os.path.join(a.pin_root, k)) else None) for k in pins}
        C["C-PIN"] = {"pass": all(got[k] == pins[k] for k in pins), "mismatch": [k for k in pins if got[k] != pins[k]]}
    else:
        C["C-PIN"] = {"pass": False, "why": "no --pins"}
    bt = Boot(prim, tlog)
    # ---- C-EST: our draws reproduce eval/proxy_eval.boot (on the identity arm only)
    if a.proxy_eval:
        spec = importlib.util.spec_from_file_location("proxy_eval_n1", os.path.join(a.proxy_eval, "proxy_eval.py"))
        PE = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(PE)
        pt = {t: R[t]["arms"]["N1T"]["tracked"]["pdms"] for t in prim}
        ref = PE.boot(pt, tlog)
        ours = bt.stat(pt)["_raw"]
        dev = max(abs(x - y) for x, y in zip(ref, ours))
        C["C-EST"] = {"pass": dev <= 1e-9, "proxy_eval_boot": ref, "ours": ours, "max_abs_dev": dev}
    else:
        C["C-EST"] = {"pass": False, "why": "no --proxy-eval"}
    # ---- C-ID-H (i): N1T tracked == the banked per-token harness run, every sub-score, every token
    ref = load_identity(a)
    worst, n_diff, miss = 0.0, 0, 0
    wm = {m: 0.0 for m in MET}
    for t in allt:
        if t not in ref:
            miss += 1
            continue
        tr = R[t]["arms"]["N1T"]["tracked"]
        d = max(abs(tr[m] - ref[t][m]) for m in MET)
        for m in MET:
            wm[m] = max(wm[m], abs(tr[m] - ref[t][m]))
        worst = max(worst, d)
        n_diff += d > a.id_tol
    agg_all = 100 * float(np.mean([R[t]["arms"]["N1T"]["tracked"]["pdms"] for t in allt]))
    agg_ref = 100 * float(np.mean([ref[t]["pdms"] for t in allt if t in ref])) if ref else float("nan")
    C["C-ID-H"] = {"pass": bool(ref) and miss == 0 and n_diff == 0, "tol": a.id_tol, "tokens_compared": len(allt) - miss,
                   "missing_in_reference": miss, "tokens_over_tol": n_diff, "max_abs_diff": worst, "max_abs_diff_by_metric": wm,
                   "pdms_all_tokens_x100": round(agg_all, 4), "reference_pdms_all_tokens_x100": round(agg_ref, 4),
                   "reference": a.identity_csv or a.identity_ref}
    # ---- C-REG (ii): the deliberate regression reads worse
    pT = {t: R[t]["arms"]["N1T"]["tracked"]["pdms"] for t in prim}
    pW = {t: R[t]["arms"]["N1W"]["tracked"]["pdms"] for t in prim}
    mT, mW = 100 * np.mean(list(pT.values())), 100 * np.mean(list(pW.values()))
    C["C-REG"] = {"pass": bool(mW <= mT - a.reg_margin), "N1W_pdms": round(float(mW), 4), "N1T_pdms": round(float(mT), 4),
                  "bar": f"PDMS(N1W) <= PDMS(N1T) - {a.reg_margin}"}
    # ---- diagnostic only (never voids): the human arm against the banked human harness CSV
    if a.human_csv:
        href = {}
        for row in csv.DictReader(open(a.human_csv, encoding="utf-8")):
            if row["token"] != "average" and row.get("valid", "True") == "True":
                href[row["token"]] = {m: float(row[CSV_COL[m]]) for m in MET}
        dd = [max(abs(R[t]["arms"]["human"]["tracked"][m] - href[t][m]) for m in MET) for t in allt if t in href]
        res["diag_human_vs_banked_harness"] = {"n": len(dd), "tokens_over_tol": int(sum(x > a.id_tol for x in dd)),
                                               "max_abs_diff": float(max(dd)) if dd else None,
                                               "note": "diagnostic, decides nothing: the banked human seam is NAVSIM's human, the "
                                                       "W-TRANSP human arm is the DB's logged future (they differ on the 36 C1 tokens by construction)"}
    allpass = all(c["pass"] for c in C.values())
    print("ZZN1_CONTROLS", "PASS" if allpass else "FAIL", json.dumps({k: v["pass"] for k, v in C.items()}), flush=True)
    if not allpass:
        res["verdict"] = "NO VERDICT: a control failed; no N1C / N1M / CERT aggregate was computed"
        if not C["C-ID-H"]["pass"] and all(C[k]["pass"] for k in C if k != "C-ID-H"):
            res["next"] = ("C-ID-H alone failed: run the PRE-REGISTERED fallback route (PREREG_TF_N1 section 6, C-ID-H): score "
                           "refe_tfn1_tc20s.npz, refe_tfn1_tw.npz and the registered T seam through the pinned W3 harness "
                           "(eval/score_p4.py), then p4-style read from the harness CSVs. W-TRANSP rows stay for transparent "
                           "scores and families, flagged 'not harness-identical'.")
        json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
        return 1

    def arm_table(arm, frame, toks, bb):
        out = {}
        for m in MET:
            out[m] = {k: v for k, v in bb.stat({t: R[t]["arms"][arm][frame][m] for t in toks}).items() if k != "_raw"}
        out["zero_rows"] = {"pdms": int(sum(R[t]["arms"][arm][frame]["pdms"] == 0 for t in toks)),
                            "dac": int(sum(R[t]["arms"][arm][frame]["dac"] == 0 for t in toks)),
                            "nc": int(sum(R[t]["arms"][arm][frame]["nc"] == 0 for t in toks))}
        return out

    def paired(x, y, frame, toks, bb, metric="pdms"):
        st = bb.stat({t: R[t]["arms"][x][frame][metric] - R[t]["arms"][y][frame][metric] for t in toks})
        return {k: v for k, v in st.items() if k != "_raw"}

    bt_all = Boot(allt, tlog)
    res["scorer_of_record"] = ("pinned W3 harness CSVs (FALLBACK ROUTE: C-ID-H failed on W-TRANSP)" if a.harness_csv else
                               "W-TRANSP transp_score.py tracked (harness-identical by C-ID-H)")
    res["tables"] = {"primary_12110": {f"{arm}_{fr}": arm_table(arm, fr, prim, bt) for arm in ARMS_ALL for fr in FR},
                     "all_12146_sensitivity": {f"{arm}_tracked": arm_table(arm, "tracked", allt, bt_all) for arm in ("N1T", "N1C", "CERT", "human")}}
    TP = res["tables"]["primary_12110"]
    # ---- PRIMARY: N1C - N1T, paired PDMS, tracked
    d = paired("N1C", "N1T", "tracked", prim, bt)
    dacC = TP["N1C_tracked"]["dac"]["mean"]
    conf = d["mean"] >= BAR["confirm_delta"] and d["ci95"][0] > 0 and dacC >= BAR["confirm_dac"]
    res["primary"] = {"contrast": "N1C - N1T, paired PDMS x100, tracked (W-TRANSP), primary 12,110", "delta": d,
                      "N1C_dac": dacC, "N1C_pdms": TP["N1C_tracked"]["pdms"], "N1T_pdms": TP["N1T_tracked"]["pdms"],
                      "rule": "CONFIRMED iff delta >= +3.0 AND ci95 lo > 0 AND N1C DAC >= 98.5",
                      "outcome": "CONFIRMED" if conf else "NOT CONFIRMED"}
    res["primary"]["dac_delta"] = paired("N1C", "N1T", "tracked", prim, bt, "dac")
    res["primary"]["sensitivity_all_12146"] = paired("N1C", "N1T", "tracked", allt, bt_all)

    def paper_read(arm):
        p, dc = TP[f"{arm}_tracked"]["pdms"]["mean"], TP[f"{arm}_tracked"]["dac"]["mean"]
        lvl = ("PAPER LEVEL" if p >= BAR["paper_pdms"] and dc >= BAR["paper_dac"] else
               "NEAR" if p >= BAR["near_pdms"] and dc >= BAR["near_dac"] else "BELOW")
        mm = {m: TP[f"{arm}_tracked"][m]["mean"] / 100.0 for m in ("nc", "dac", "ep", "ttc", "c")}
        f = lambda v: v["nc"] * v["dac"] * (5 * v["ep"] + 5 * v["ttc"] + 2 * v["c"]) / 12.0     # noqa: E731
        base = f(mm)
        swaps = {m: round(100 * (f(dict(mm, **{m: PAPER[m]})) - base), 4) for m in mm}
        return {"pdms": TP[f"{arm}_tracked"]["pdms"], "dac": TP[f"{arm}_tracked"]["dac"], "level": lvl,
                "gap_to_paper_pdms": round(BAR["paper_pdms"] - p, 4),
                "decomposition_mean_field": {"what": "PDMS formula on the arm's sub-score MEANS; each sub-score swapped one at a "
                                             "time to the paper's Table 4 value (descriptive, never a bar)",
                                             "formula_on_means_x100": round(100 * base, 4),
                                             "residual_actual_minus_formula": round(p - 100 * base, 4), "swap_gain_x100": swaps}}
    res["paper_level"] = {"N1C": paper_read("N1C"),
                          "rule": "PAPER LEVEL iff PDMS >= 95.8 AND DAC >= 99.9; NEAR iff >= 95.3 AND >= 99.5; else BELOW (point values)"}
    hum = TP["human_tracked"]["pdms"]["mean"]
    res["paper_level"]["human_anchored_descriptive"] = {"our_human": hum, "paper_human": PAPER_HUMAN_PDMS,
                                                        "human_anchored_paper_level": round(BAR["paper_pdms"] - (PAPER_HUMAN_PDMS - hum), 4),
                                                        "N1C_minus_human": paired("N1C", "human", "tracked", prim, bt)}
    # ---- SECONDARY (decides nothing): CERT = teacher + map-certified selection
    ch = {}
    for t in prim:
        ch[R[t]["cert_choice"]] = ch.get(R[t]["cert_choice"], 0) + 1
    res["secondary_CERT"] = {"label": "teacher + map-certified selection (NEVER the teacher alone, NEVER checkpoint identity)",
                             "decides": "nothing", "choice_primary": ch,
                             "none_pass_primary": int(sum(R[t]["cert_none_pass"] for t in prim)),
                             "CERT_minus_N1C": paired("CERT", "N1C", "tracked", prim, bt),
                             "CERT_minus_N1C_dac": paired("CERT", "N1C", "tracked", prim, bt, "dac"),
                             "CERT_minus_N1T": paired("CERT", "N1T", "tracked", prim, bt),
                             "paper_read_descriptive": paper_read("CERT")}
    # ---- tracker-transparent (W-TRANSP), descriptive; transparent comfort is never a driving result
    res["transparent_descriptive"] = "from the W-TRANSP rows (not in the harness CSVs)" if a.harness_csv else {
                                      "N1T_transparent_minus_tracked": paired_tt("N1T", R, prim, bt),
                                      "N1C_tracked_minus_N1T_transparent": {k: v for k, v in bt.stat(
                                          {t: R[t]["arms"]["N1C"]["tracked"]["pdms"] - R[t]["arms"]["N1T"]["transparent"]["pdms"]
                                           for t in prim}).items() if k != "_raw"},
                                      "caveat": "transparent comfort depends on how the states are derived (E2); never quote it as driving"}
    # ---- four families (W-FAM): executed_tracked = PRIMARY, reference = diagnostic; STRATEGIC PAUSED
    fam = {}
    for spec in a.families or []:
        nm, fp = spec.split("=", 1)
        F = json.load(open(fp, encoding="utf-8"))
        fam[nm] = {"file": fp, "sha256": sha256(fp), "n": F.get("n"), "n_logs": F.get("n_logs"), "lead_states": F.get("lead_states"),
                   "strategic": F.get("strategic"),
                   "frames": {k: {"role": v["role"], "block": v["block"]} for k, v in F.get("frames", {}).items()}}
    res["families"] = fam or "NOT RUN (a read requirement: run families_run.py per arm on rows_cert_primary.jsonl)"
    res["k_ledger"] = {"entry": "ONE K entry (TF-N1). N1T (identity), N1W (regression), N1M (CERT input) and CERT (secondary) are "
                                "logged under it as non-selection reads", "K": K}
    print("ZZN1_VERDICT PRIMARY", res["primary"]["outcome"], json.dumps(res["primary"]["delta"]), "N1C DAC", dacC, flush=True)
    print("ZZN1_VERDICT PAPER", res["paper_level"]["N1C"]["level"], json.dumps(res["paper_level"]["N1C"]["pdms"]), flush=True)
    print("ZZN1_SECONDARY CERT", json.dumps(res["secondary_CERT"]["CERT_minus_N1C"]), json.dumps(ch), flush=True)
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
    return 0


def paired_tt(arm, R, toks, bt):
    st = bt.stat({t: R[t]["arms"][arm]["transparent"]["pdms"] - R[t]["arms"][arm]["tracked"]["pdms"] for t in toks})
    return {k: v for k, v in st.items() if k != "_raw"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("cert", "read"))
    ap.add_argument("--rows", nargs="*", default=[])
    ap.add_argument("--known36", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-primary", default=None)
    ap.add_argument("--build-report", default=None)
    ap.add_argument("--identity-csv", default=None)
    ap.add_argument("--identity-ref", default=None, help="smoke only: 'tfrows|<glob>|<field>' (navtrain TF rows as the reference)")
    ap.add_argument("--human-csv", default=None)
    ap.add_argument("--pins", default=None)
    ap.add_argument("--pin-root", default=".")
    ap.add_argument("--proxy-eval", default=None, help="dir holding eval/proxy_eval.py (C-EST)")
    ap.add_argument("--families", nargs="*", default=[])
    ap.add_argument("--expect-n", type=int, default=12146)
    ap.add_argument("--expect-primary", type=int, default=12110)
    ap.add_argument("--id-tol", type=float, default=BAR["id_tol"])
    ap.add_argument("--reg-margin", type=float, default=BAR["reg_margin"])
    ap.add_argument("--prereg-sha256", default=None)
    ap.add_argument("--harness-csv", nargs="*", default=[],
                    help="FALLBACK ROUTE ONLY: ARM=<pinned-harness csv> for N1T (the re-scored T seam), N1C, N1W, N1M")
    ap.add_argument("--token-log", default="F:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if a.mode == "cert":
        if not a.out_primary:
            raise SystemExit("cert needs --out-primary")
        return cmd_cert(a)
    return cmd_read(a)


if __name__ == "__main__":
    sys.exit(main())
