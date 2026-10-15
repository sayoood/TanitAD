#!/usr/bin/env python3
"""Tests for p2rep_read.py (the P2-s1 read, PREREG_P2_REPLICATE sections 2-3).  Prints `ZZTEST_PASS <n>` (exit 0) or `ZZTEST_FAIL ...` (exit 1).

    C:/Users/Admin/venvs/tanitad/Scripts/python.exe test_p2rep_read.py          # ~2 min (the identity reads are the real 10,000-resample bootstrap)

    (a) IDENTITY       P2-s0 vs u400 from the banked CSVs through the production `rr_read` = +0.57 [+0.02, +1.18], and BIT-equal to P2's banked R2
    (b) ZERO CONTROL   P2-s0 vs P2-s0: Delta_seed EXACTLY 0.0, CI [0.0, 0.0], six sub-scores exactly [0, 0] -- and the read refuses it (INVALID)
    (c) EVERY ROW      a synthetic table reaches each outcome row of PREREG_P2_REPLICATE section 3, exactly one
    (d) MUTATIONS      the SOURCE of p2rep_read.py is edited (one line each) and the check that guards it must go RED
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p2rep_read as R  # noqa: E402

S3 = R.S3
SRC = (HERE / "p2rep_read.py").read_text(encoding="utf-8")
N_OK, FAILS = 0, []


def check(name, cond, detail=""):
    global N_OK
    if cond:
        N_OK += 1
    else:
        FAILS.append(f"{name} {detail}".strip())
        print("  FAIL", name, detail, flush=True)


# ---------------------------------------------------------------- real banked inputs (loaded once)
_CACHE = {}


def banked():
    if not _CACHE:
        s0p, u4p, tkp = (R.resolve("p2_s0_csv"), R.resolve("u400_csv"), None)
        for c in R.CANDS["tokens"]:
            if Path(c).is_file() and R.pin_match(c, R.PINS["tokens"])[0]:
                tkp = c
                break
        assert s0p["ok"] and u4p["ok"] and tkp, (s0p, u4p, tkp)
        tok = R.load_token_file(tkp, R.PINS["tokens"])
        _CACHE.update(s0_path=Path(s0p["path"]), u4_path=Path(u4p["path"]), tok=tok, s0=S3.load_table(s0p["path"]), u4=S3.load_table(u4p["path"]),
                      banked_r2=json.load(open(R.first_existing("p2_s0_r2_result"), encoding="utf-8")))
    return _CACHE


OK_GATES = {"stage3:gates_ok": {"ok": True}, "stage3:harness_status": {"ok": True}}


def mutated(old, new):
    """a fresh module built from p2rep_read.py's SOURCE with exactly one occurrence of `old` replaced by `new`"""
    assert SRC.count(old) == 1, f"mutation anchor not unique: {old!r} x{SRC.count(old)}"
    m = types.ModuleType("p2rep_read_mut")
    m.__file__ = str(HERE / "p2rep_read.py")
    exec(compile(SRC.replace(old, new), str(HERE / "p2rep_read.py"), "exec"), m.__dict__)
    return m


# ---------------------------------------------------------------- (a) + (b) on the banked data
def identity_and_zero(mod):
    """returns {check name: bool}: the production rr_read(s1 := P2-s0, s0 := P2-s0, u400) -- RR1 is then P2-s0 vs u400, RR2 the zero control"""
    b = banked()
    res = mod.rr_read(b["s0"], b["s0"], b["u4"], b["tok"]["tokens"], b["tok"]["token_log"], gates=OK_GATES)
    p, bk = res["RR1"]["primary"], b["banked_r2"]["R2"]
    z = res["RR2"]
    out = {
        "identity_printed_2dp": f"{p['mean']:+.2f} [{p['ci95'][0]:+.2f}, {p['ci95'][1]:+.2f}]" == "+0.57 [+0.02, +1.18]",
        "identity_bit_equal_banked": p["mean"] == bk["primary"]["mean"] and p["ci95"] == bk["primary"]["ci95"],
        "identity_subscores_bit_equal": all(res["RR1"]["subscores"][k]["mean"] == bk["subscores"][k]["mean"] and res["RR1"]["subscores"][k]["ci95"] == bk["subscores"][k]["ci95"] for k in R.SUBS),
        "identity_counts": (p["better"], p["worse"], p["tied"]) == (5449, 3178, 3519) and p["n"] == 12146 and p["n_logs"] == 136,
        "zero_primary_exact": z["primary"]["mean"] == 0.0 and z["primary"]["ci95"] == [0.0, 0.0] and z["primary"]["tied"] == 12146,
        "zero_subscores_exact": all(z["subscores"][k]["mean"] == 0.0 and z["subscores"][k]["ci95"] == [0.0, 0.0] for k in R.SUBS),
        "zero_F_is_0": z["F"] == 0.0 and z["delta_seed_mean"] == 0.0 and z["delta_seed_ci95"] == [0.0, 0.0],
        "zero_read_refused_INVALID": res["outcome"] == "INVALID" and "s1_not_s0" in res["failed_gates"] and not res["result_recorded"],
    }
    return out, res


# ---------------------------------------------------------------- (c) synthetic tables: each outcome row
N_LOGS, PER = 40, 10


def synth(delta0=1.0, delta1=1.0, nc_drop=False, seed=11):
    rng = np.random.default_rng(seed)
    toks = [f"t{i:04d}" for i in range(N_LOGS * PER)]
    tl = {t: f"log{i // PER:03d}" for i, t in enumerate(toks)}
    base = rng.uniform(60.0, 95.0, len(toks))

    def noise(mean):
        d = rng.normal(0.0, 0.8, len(toks))
        return d - d.mean() + mean                       # the arm's mean Delta vs u400 is EXACTLY `mean`

    def table(pdms, nc=None):
        return {t: {"PDMS": float(pdms[i]), "NC": float(100.0 if nc is None else nc[i]), "DAC": 100.0, "EP": 80.0, "TTC": 100.0, "C": 100.0, "DDC": 100.0}
                for i, t in enumerate(toks)}

    u4 = table(base)
    s0 = table(base + noise(delta0))
    nc = np.where(rng.uniform(size=len(toks)) < 0.3, 90.0, 100.0) if nc_drop else None
    s1 = table(base + noise(delta1), nc)
    return toks, tl, u4, s0, s1


def run_rows(mod):
    """{case: (expected outcome, got outcome)} on small synthetic tables (n_boot 300: the logic, not the interval)"""
    rows = {}

    def go(case, expect, delta0, delta1, gates=OK_GATES, **kw):
        toks, tl, u4, s0, s1 = synth(delta0, delta1, **kw)
        r = mod.rr_read(s1, s0, u4, toks, tl, gates=gates, n_boot=300)
        rows[case] = (expect, r["outcome"], r)

    go("replicated", "REPLICATED", 1.0, 1.0)
    go("floor_bound_negative_dseed", "REPLICATED, FLOOR-BOUND", 1.0, 0.3)            # Delta_seed = -0.7: the SIGN must not matter
    go("floor_bound_positive_dseed", "REPLICATED, FLOOR-BOUND", 0.3, 1.0)
    go("not_replicated_not_proven", "NOT REPLICATED", 1.0, 0.0)                      # P2-s1 == u400 on average: interval holds 0
    go("not_replicated_adverse_subscore", "NOT REPLICATED", 1.0, 1.0, nc_drop=True)  # lo > 0 but NC adverse
    go("not_replicated_below_u400", "NOT REPLICATED", 1.0, -1.0)                     # hi < 0
    go("invalid_failed_stage3_gate", "INVALID", 1.0, 1.0, gates={**OK_GATES, "stage3:harness_status": {"ok": False}})
    go("invalid_no_gate_record", "INVALID", 1.0, 1.0, gates=None)
    toks, tl, u4, s0, _ = synth(1.0, 1.0)
    r = mod.rr_read(s0, s0, u4, toks, tl, gates=OK_GATES, n_boot=300)                # the replicate's table IS P2-s0's
    rows["invalid_s1_is_s0"] = ("INVALID", r["outcome"], r)
    return rows


def rows_ok(mod):
    return {k: v[0] == v[1] for k, v in run_rows(mod).items()}


def main() -> int:
    b = banked()
    print(f"banked inputs: s0 {b['s0_path']}  u400 {b['u4_path']}  tokens {b['tok']['gate']['path']}", flush=True)

    # ---- inputs and pins
    pin_s0 = R.pin_match(b["s0_path"], R.PINS["p2_s0_csv"])
    check("pin_p2_s0_table", pin_s0[0], pin_s0[1])
    check("pin_u400_table", R.pin_match(b["u4_path"], R.PINS["u400_csv"])[0])
    check("pin_tokens_file", b["tok"]["gate"]["ok"] and b["tok"]["gate"]["n"] == 12146 and b["tok"]["gate"]["n_logs"] == 136, str(b["tok"]["gate"]))
    check("estimator_constants_pinned", (R.N_BOOT, R.SEED) == (10000, 20260927) and (S3.N_BOOT, S3.SEED) == (10000, 20260927))
    check("imports_the_programs_boot", R.PE.boot.__module__ == "proxy_eval" and R.S3.read_pair.__module__ == "stage3_full")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        raw = b["s0_path"].read_bytes()
        lf = td / "lf.csv"; lf.write_bytes(raw.replace(b"\r\n", b"\n"))
        crlf = td / "crlf.csv"; crlf.write_bytes(raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
        bad = td / "bad.csv"; bad.write_bytes(raw.replace(b"0.9361607541859801", b"0.9361607541859802", 1))
        check("pin_accepts_lf_and_crlf_forms", R.pin_match(lf, R.PINS["p2_s0_csv"])[0] and R.pin_match(crlf, R.PINS["p2_s0_csv"])[0])
        check("pin_refuses_one_changed_digit", raw.count(b"0.9361607541859801") >= 1 and not R.pin_match(bad, R.PINS["p2_s0_csv"])[0])
        check("pin_refuses_the_other_arm", not R.pin_match(b["u4_path"], R.PINS["p2_s0_csv"])[0] and not R.pin_match(b["s0_path"], R.PINS["u400_csv"])[0])
        tk = json.load(open(b["tok"]["gate"]["path"], encoding="utf-8"))
        tk["tokens"] = tk["tokens"][:-1]
        short = td / "short_tokens.json"; short.write_text(json.dumps(tk), encoding="utf-8")
        check("token_file_one_short_refused", not R.load_token_file(short, R.PINS["tokens"])["gate"]["ok"])

    # ---- (a) + (b): identity and the zero control, on the real banked tables, 10,000 resamples
    ok, res = identity_and_zero(R)
    for k, v in ok.items():
        check(k, v)
    print(f"  identity: RR1 {res['RR1']['primary']['mean']:+.6f} {res['RR1']['primary']['ci95']}  zero control dseed {res['RR2']['delta_seed_mean']} {res['RR2']['delta_seed_ci95']}", flush=True)
    check("report_has_all_six_subscores_with_ci", set(res["RR1"]["subscores"]) == set(R.SUBS) and all(len(v["ci95"]) == 2 for v in res["RR1"]["subscores"].values())
          and set(res["RR2"]["subscores"]) == set(R.SUBS))

    # ---- (c) each outcome row
    rows = run_rows(R)
    for case, (want, got, _r) in rows.items():
        check(f"row_{case}", want == got, f"want {want} got {got}")
    check("rows_cover_all_four_outcomes", {v[1] for v in rows.values()} == set(R.OUTCOMES))
    check("replicated_records_mean_of_two_seeds", abs(rows["replicated"][2]["mean_of_two_seeds_delta"] - 1.0) < 1e-9)
    check("adverse_row_names_NC", rows["not_replicated_adverse_subscore"][2]["RR1"]["adverse_subscores"] == ["NC"]
          and rows["not_replicated_adverse_subscore"][2]["RR1"]["amendment_4_row"] == "NO-GO (adverse sub-score)")
    check("not_proven_row_is_amendment_4_not_proven", rows["not_replicated_not_proven"][2]["RR1"]["amendment_4_row"] == "NO-GO (NOT PROVEN)")
    check("below_row_is_amendment_4_below", rows["not_replicated_below_u400"][2]["RR1"]["amendment_4_row"] == "NO-GO (P2 below u400)")
    check("invalid_records_no_result", all(not rows[k][2]["result_recorded"] for k in rows if rows[k][0] == "INVALID")
          and all(rows[k][2]["result_recorded"] for k in rows if rows[k][0] != "INVALID"))
    check("invalid_names_the_failed_gate", "stage3:harness_status" in rows["invalid_failed_stage3_gate"][2]["failed_gates"]
          and "stage3_gate_record" in rows["invalid_no_gate_record"][2]["failed_gates"])
    fb = rows["floor_bound_negative_dseed"][2]
    check("floor_bound_F_is_abs", fb["RR2"]["F"] > 0 and abs(fb["RR2"]["delta_seed_mean"] + 0.7) < 1e-9 and abs(fb["RR2"]["F"] - 0.7) < 1e-9)
    # map_outcome as a pure table
    mo = R.map_outcome
    check("map_boundary_F_equals_min_is_floor_bound", mo(True, 0.5, 0.5, 0.5, True)["outcome"] == "REPLICATED, FLOOR-BOUND")
    check("map_boundary_just_below", mo(True, 0.5, 0.6, 0.4999, True)["outcome"] == "REPLICATED")
    check("map_uses_the_smaller_delta", mo(True, 0.9, 0.3, 0.4, True)["outcome"] == "REPLICATED, FLOOR-BOUND" and mo(True, 0.3, 0.9, 0.2, True)["outcome"] == "REPLICATED")
    check("map_invalid_beats_everything", mo(True, 1, 1, 0, False)["outcome"] == "INVALID" and mo(False, 1, 1, 0, False)["outcome"] == "INVALID")
    check("map_rr1_failure_is_not_replicated", mo(False, 1, 1, 0, True)["outcome"] == "NOT REPLICATED")

    # ---- stage-3 gate record
    r2 = b["banked_r2"]
    s0md5 = R.md5_pair(b["s0_path"])[0]
    g, prim = R.stage3_gates(r2, "p2_full", s0md5)
    check("stage3_gates_banked_p2_all_ok", all(v["ok"] for v in g.values()) and prim == r2["R2"]["primary"], str({k: v for k, v in g.items() if not v["ok"]}))
    import copy
    for what, mut, key in (("gates_ok_false", lambda d: d.update(gates_ok=False), "stage3:gates_ok"),
                           ("one_gate_failed", lambda d: d["gates"]["seam_report"].update(ok=False), "stage3:seam_report"),
                           ("dry_run", lambda d: d.update(dry_run=True), "stage3:not_dry_run"),
                           ("ckpt_sha_unchecked", lambda d: d["gates"]["checkpoint"].update(sha256_matches_expected=None), "stage3:checkpoint_sha256_was_checked"),
                           ("ckpt_sha_mismatch", lambda d: d["gates"]["checkpoint"].update(sha256_matches_expected=False), "stage3:checkpoint_sha256_was_checked"),
                           ("other_name", lambda d: d.update(name="p2_other"), "stage3:name"),
                           ("other_table", lambda d: d["arm"].update(csv_md5="0" * 32), "stage3:same_table")):
        d = copy.deepcopy(r2); mut(d)
        gg, _ = R.stage3_gates(d, "p2_full", s0md5)
        check(f"stage3_gate_{what}_fails", not gg[key]["ok"], key)
    check("stage3_gates_absent_record_fails", not R.stage3_gates(None, "x", "y")[0]["stage3_record"]["ok"])

    # ---- the cross-instrument controls inside rr_read
    toks, tl, u4, s0, s1 = synth(1.0, 1.0)
    r_good = R.rr_read(s1, s0, u4, toks, tl, gates=OK_GATES, n_boot=300)
    r_x = R.rr_read(s1, s0, u4, toks, tl, gates=OK_GATES, n_boot=300, stage3_rr1=r_good["RR1"]["primary"], banked_s0=r_good["P2_s0_vs_u400"])
    check("controls_equal_pass", r_x["gates"]["rr1_equals_stage3_R2"]["ok"] and r_x["gates"]["p2s0_vs_u400_reproduces_banked_R2"]["ok"] and r_x["outcome"] == "REPLICATED")
    off = dict(r_good["RR1"]["primary"]); off["mean"] = off["mean"] + 1e-12
    r_y = R.rr_read(s1, s0, u4, toks, tl, gates=OK_GATES, n_boot=300, stage3_rr1=off, banked_s0=r_good["P2_s0_vs_u400"])
    check("stage3_rr1_off_by_1e-12_is_INVALID", r_y["outcome"] == "INVALID" and "rr1_equals_stage3_R2" in r_y["failed_gates"])
    r_z = R.rr_read(s1, s0, u4, toks, tl, gates=OK_GATES, n_boot=300, require_controls=True)
    check("missing_controls_are_INVALID_in_production", r_z["outcome"] == "INVALID" and {"rr1_equals_stage3_R2", "p2s0_vs_u400_reproduces_banked_R2"} <= set(r_z["failed_gates"]))
    s1m = {t: dict(v) for t, v in s1.items()}; del s1m[toks[0]]
    check("missing_token_row_is_INVALID", R.rr_read(s1m, s0, u4, toks, tl, gates=OK_GATES, n_boot=300)["outcome"] == "INVALID")
    s1n = {t: dict(v) for t, v in s1.items()}; s1n[toks[3]]["EP"] = float("nan")
    check("nan_score_is_INVALID", R.rr_read(s1n, s0, u4, toks, tl, gates=OK_GATES, n_boot=300)["outcome"] == "INVALID")
    r_u = R.rr_read(u4, s0, u4, toks, tl, gates=OK_GATES, n_boot=300)
    check("s1_equal_u400_is_INVALID", r_u["outcome"] == "INVALID" and "s1_not_u400" in r_u["failed_gates"])

    # ---- families (RR3) on the banked JSONs: P2-s0 stands in for the replicate
    f0, fu = R.load_json(R.first_existing("p2_s0_families")), R.load_json(R.first_existing("u400_families"))
    fam = R.families_block(f0, f0, fu)
    bank_adv = [(a["family"], a["metric"]) for a in r2["R2"]["families_vs_base"]["adverse"]]
    check("families_vs_u400_reproduce_banked_R2", [(a["family"], a["metric"]) for a in fam["s1_vs_u400"]["adverse"]] == bank_adv and fam["s1_vs_u400"]["components_compared"] == 21,
          str(bank_adv))
    check("families_self_pair_has_no_adverse", fam["s1_vs_p2_s0"]["adverse"] == [] and fam["s1_vs_p2_s0"]["components_compared"] == 21)
    check("families_tactical_rows_carry_no_interval_note", set(fam["s1_vs_u400"]["tactical_no_interval"]) == {"lateral_decision", "longitudinal_decision", "maneuver_5way_collapsed"})
    check("strategic_is_paused_never_a_pass", "PAUSED" in fam["strategic"] and "never as a pass" in fam["strategic"] and fam["gating"] is False)
    check("absent_families_are_NOT_COMPUTED_not_a_pass", R.families_block(None, f0, fu)["s1_vs_u400"].get("missing") is True)

    # ---- the CLI end to end on FULL-SIZE data (10,000 resamples)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # 1. the replicate's table = P2-s0's own file: the CLI must refuse (INVALID, exit 2)
        out1 = td / "o1"
        shutil.copy(b["s0_path"], td / "s1_same.csv")
        rc = R.main(["--name", "p2_full", "--s1-csv", str(td / "s1_same.csv"), "--r2-result", str(R.first_existing("p2_s0_r2_result")), "--out-dir", str(out1)])
        j1 = json.load(open(out1 / "p2rep_result.json", encoding="utf-8"))
        check("cli_s0_as_s1_exit_2_INVALID", rc == 2 and j1["outcome"] == "INVALID" and "s1_not_s0" in j1["failed_gates"], str((rc, j1["outcome"], j1["failed_gates"])))
        check("cli_writes_json_and_md", (out1 / "P2REP_READ.md").is_file() and "INVALID" in (out1 / "P2REP_READ.md").read_text(encoding="utf-8"))
        # 2. a perturbed copy of P2-s0 with a fabricated (test-only) stage-3 record that is consistent with it: a full-size valid read
        import csv
        rows = list(csv.reader(open(b["s0_path"], encoding="utf-8", newline="")))
        hdr, body = rows[0], rows[1:]
        iscore = hdr.index("score"); iep = hdr.index("ego_progress")
        k = 0
        for r in body:
            if r[1] == "average":
                continue
            k += 1
            if k % 20 == 0 and float(r[iscore]) < 0.99:
                r[iscore] = repr(float(r[iscore]) + 0.3 * (1.0 - float(r[iscore])))
                r[iep] = repr(min(1.0, float(r[iep]) + 0.2))
        s1csv = td / "s1_perturbed.csv"
        with open(s1csv, "w", encoding="utf-8", newline="") as fh:
            csv.writer(fh).writerows([hdr] + body)
        s1t = S3.load_table(s1csv)
        rr1_here = S3.read_pair(s1t, b["u4"], b["tok"]["tokens"], b["tok"]["token_log"])
        fake = copy.deepcopy(r2)
        fake.update(name="p2s1_testfixture", dry_run=False)
        fake["arm"]["csv_md5"] = R.md5_pair(s1csv)[0]
        fake["R2"]["primary"] = {"mean": rr1_here["primary"]["mean"], "ci95": rr1_here["primary"]["ci95"]}
        r2f = td / "r2_fixture.json"; r2f.write_text(json.dumps(fake), encoding="utf-8")
        out2 = td / "o2"
        rc2 = R.main(["--name", "p2s1_testfixture", "--s1-csv", str(s1csv), "--r2-result", str(r2f), "--out-dir", str(out2),
                      "--s1-families", str(R.first_existing("p2_s0_families"))])
        j2 = json.load(open(out2 / "p2rep_result.json", encoding="utf-8"))
        want = "REPLICATED" if (j2["RR1"]["passes"] and j2["RR2"]["F"] < min(j2["D_s0"], j2["D_s1"])) else \
            ("REPLICATED, FLOOR-BOUND" if j2["RR1"]["passes"] else "NOT REPLICATED")
        check("cli_full_size_valid_read_exit_0", rc2 == 0 and j2["gates_ok"] and j2["result_recorded"], str((rc2, j2["failed_gates"])))
        check("cli_full_size_outcome_matches_the_rule", j2["outcome"] == want and j2["RR1"]["passes"], str((j2["outcome"], want, j2["RR1"]["primary"])))
        check("cli_full_size_dseed_positive_small", 0 < j2["RR2"]["delta_seed_mean"] < 0.4, str(j2["RR2"]["delta_seed_mean"]))
        check("cli_identity_control_in_production", j2["gates"]["p2s0_vs_u400_reproduces_banked_R2"]["ok"] and j2["gates"]["rr1_equals_stage3_R2"]["ok"])
        check("cli_json_has_families_and_strategic", "PAUSED" in j2["families"]["strategic"] and j2["families"]["s1_vs_u400"]["components_compared"] == 21)
        md = (out2 / "P2REP_READ.md").read_text(encoding="utf-8")
        check("cli_md_has_six_subscores_and_strategic", all(f"| {s} |" in md for s in R.SUBS) and "STRATEGIC: PAUSED" in md)
        # 3. the same fixture with ONE stage-3 gate failed: INVALID, no outcome recorded
        bad = copy.deepcopy(fake); bad["gates"]["harness_status"]["ok"] = False
        r2b = td / "r2_bad.json"; r2b.write_text(json.dumps(bad), encoding="utf-8")
        out3 = td / "o3"
        rc3 = R.main(["--name", "p2s1_testfixture", "--s1-csv", str(s1csv), "--r2-result", str(r2b), "--out-dir", str(out3)])
        j3 = json.load(open(out3 / "p2rep_result.json", encoding="utf-8"))
        check("cli_failed_stage3_gate_exit_2_INVALID", rc3 == 2 and j3["outcome"] == "INVALID" and "stage3:harness_status" in j3["failed_gates"])
        # 4. a missing stage-3 record
        out4 = td / "o4"
        rc4 = R.main(["--name", "p2s1_testfixture", "--s1-csv", str(s1csv), "--r2-result", str(td / "nope.json"), "--out-dir", str(out4)])
        check("cli_missing_stage3_record_INVALID", rc4 == 2 and json.load(open(out4 / "p2rep_result.json", encoding="utf-8"))["outcome"] == "INVALID")
        check("cli_missing_s1_table_exit_3", R.main(["--name", "p2s1_testfixture", "--s1-csv", str(td / "absent.csv"), "--r2-result", str(r2f), "--out-dir", str(td / "o5")]) == 3)
        check("cli_check_inputs_ok", R.main(["--check-inputs"]) == 0)

    # ---- (d) deliberate regressions: edit the SOURCE, the guarding check must go RED
    muts = [
        ("M1 floor compared SIGNED (abs(d_seed) -> d_seed)", "F = abs(d_seed)", "F = d_seed", "rows"),
        ("M2 wrong seed (20260927 -> 7)", "N_BOOT, SEED = 10000, 20260927", "N_BOOT, SEED = 10000, 7", "identity"),
        ("M3 unpaired tokens in RR2 (s0 re-paired one token over)", "rr2 = S3.read_pair(s1, s0, sel, tl, n_boot, seed)",
         "rr2 = S3.read_pair(s1, {t: s0[sel[(i + 1) % len(sel)]] for i, t in enumerate(sel)}, sel, tl, n_boot, seed)", "identity"),
        ("M4 RR1 ignores the adverse sub-score rule", 'rr1_pass = dec["outcome"] == "GO"', 'rr1_pass = rr1["primary"]["ci95"][0] > 0', "rows"),
        ("M5 failed gates do not make the read INVALID", "gates_ok = all(isinstance(g, dict) and g.get(\"ok\") is True for g in allg.values())", "gates_ok = True", "rows"),
        ("M6 the LARGER of the two Deltas is the bar (min -> max)", "floor = min(d_s0, d_s1)", "floor = max(d_s0, d_s1)", "rows"),
        ("M7 RR1 paired against P2-s0 instead of u400", "rr1 = S3.read_pair(s1, u400, sel, tl, n_boot, seed)", "rr1 = S3.read_pair(s1, s0, sel, tl, n_boot, seed)", "identity"),
        ("M8 the identical-table guard removed", 'own["s1_not_s0"] = {"ok": not (rr2["primary"]["tied"] == rr2["primary"]["n"]),', 'own["s1_not_s0"] = {"ok": True,', "rows"),
        ("M9 outcome INVALID dropped from the mapping (gates ignored in map_outcome)", "    if not gates_ok:\n        name = \"INVALID\"\n    elif not rr1_pass:", "    if False:\n        name = \"INVALID\"\n    elif not rr1_pass:", "rows"),
        ("M10 sub-score adverse test dropped from the amendment rule (decide() sees no adverse)", "rr1[\"adverse_subscores\"], True)", "[], True)", "rows"),
    ]
    for label, old, new, which in muts:
        mod = mutated(old, new)
        if which == "rows":
            red = not all(rows_ok(mod).values())
        else:
            iz, _ = identity_and_zero(mod)
            red = not all(iz.values())
        check(f"mutation_RED: {label}", red)
    check("unmutated_source_is_GREEN", all(rows_ok(R).values()))

    if FAILS:
        print(f"ZZTEST_FAIL {len(FAILS)} failed, {N_OK} passed: " + " | ".join(FAILS))
        return 1
    print(f"ZZTEST_PASS {N_OK}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
