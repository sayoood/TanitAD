#!/usr/bin/env python3
"""Self-test of eval/r0e_diagnostics.py: the D1 / D2 / D3 diagnostics of PREREG_R0E_PICK.draft.md section 8 (launch blocker B6).
Run: python eval/test_r0e_diagnostics.py -> ZZTEST_PASS <n>.   No pod, no GPU, no network.

WHICH FIXTURE COMES FROM WHICH PRODUCER (the "built, tested, unreachable" rule: a reader tested on a schema its author invented passes its test and reads nothing)
  REAL PRODUCERS
    * the trainer's log: raw/2026-10-18-r0d-outcome/raw/sft.jsonl, R0d's REAL scorer_finetune log (40 train rows, the data / done events).  Every synthetic log below is THIS file with the numbers of some rows rewritten
      (the schema and the event order are the trainer's).
    * the banked pre-B3 outcome record raw/2026-10-18-r0d-outcome/raw/safeobj_outcome.json (another tool read the same bytes on the pod): the first breached check, the post-ramp lambda counts.
    * the A2R2 harness's banked numbers (raw/2026-10-18-so-pickcon/a2r2_tieaware.json) and, when the dev box holds them, the banked R0 states: the validation side must REPRODUCE them.
  INDEPENDENT RECOMPUTATION: the banked-log numbers are recomputed here with plain loops and `statistics` (a different code path from the tool) and ALSO pinned to the values the draft's text quotes
    (g -0.085 +- 0.026, p1gap -0.232 +- 0.069, wedge -0.317 over 31 rows u226-u1000; epo_J 0.8171 -> 0.8766; lambda > 0 on 941 of 1000 updates).
  SYNTHETIC (no producer): the planted rows (a wedge of an exact value, an outlier row to show which rows are excluded), the planted arrays of the validation side, the corrupted logs.
Then: the CLI end to end, and MUTATIONS of the tool's source, each of which must turn the checks it exists for RED.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent                    # eval/
PKROOT = HERE.parent
TOOL_PATH = HERE / "r0e_diagnostics.py"
BANKED = PKROOT / "raw" / "2026-10-18-r0d-outcome" / "raw" / "sft.jsonl"
BANKED_REC = PKROOT / "raw" / "2026-10-18-r0d-outcome" / "raw" / "safeobj_outcome.json"
A2R2_BANKED = PKROOT / "raw" / "2026-10-18-so-pickcon" / "a2r2_tieaware.json"
DRAFT = [PKROOT / "raw" / "2026-10-18-r0e-pick" / n for n in ("PREREG_R0E_PICK.md", "PREREG_R0E_PICK.draft.md")]
R0_STATES = Path("C:/Users/Admin/rls1b_dl/r0/final")             # the banked R0 states on the dev box
R0_LABELS = Path("C:/Users/Admin/rls1b_dl/data/x")


def rows_of(p):
    return [json.loads(ln) for ln in Path(p).read_text(encoding="utf-8").splitlines() if ln.strip()]


def write_log(d: Path, rows, name="sft.jsonl"):
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return d / name


BASE_ROWS = rows_of(BANKED)
TRAIN = [r for r in BASE_ROWS if r.get("event") == "train"]


def planted(*, g=None, p=None, J=None, lo=250, hi=1000, fill=(0.0007, -0.0005), ramp=None, drop=None, last=None, mutate=None, extra=None):
    """the REAL R0d log with the con_g_ttc / con_p1_gap_ttc of the rows lo..hi rewritten to g / p (callables of the update or constants), every other train row to `fill`; epo_J rewritten by J(update) when
    given; the data event's built ramp to `ramp`; the rows after `last` dropped.  SYNTHETIC numbers on the trainer's real row schema."""
    out = []
    for r in BASE_ROWS:
        r = json.loads(json.dumps(r))
        if r.get("event") == "data" and ramp is not None:
            r["epo"]["ramp"] = ramp
        if r.get("event") == "train":
            u = r["update"]
            if last is not None and u > last:
                continue
            inside = lo <= u <= hi
            gv = (g(u) if callable(g) else g) if (g is not None and inside) else fill[0]
            pv = (p(u) if callable(p) else p) if (p is not None and inside) else fill[1]
            r["con_g_ttc"], r["con_p1_gap_ttc"] = gv, pv
            if J is not None:
                r["epo_J"] = J(u)
            if mutate:
                r = mutate(r) or r
        if r.get("event") == "done" and last is not None:
            continue
        if drop and r.get("event") == "train" and r["update"] in drop:
            continue
        out.append(r)
    return out + (extra or [])


def load_tool(src: str, where: Path):
    where.parent.mkdir(parents=True, exist_ok=True)
    where.write_text(src, encoding="utf-8", newline="\n")
    m = types.ModuleType("r0e_diagnostics_under_test")
    m.__file__ = str(where)
    exec(compile(src, str(where), "exec"), m.__dict__)
    return m


CHECKS = []
HEAVY = []


def check(name, heavy=False):
    def deco(fn):
        (HEAVY if heavy else CHECKS).append((name, fn))
        return fn
    return deco


def reads(T, rows, root, tag, name="a"):
    return T.read_log(str(write_log(Path(root) / f"{tag}_{name}", rows)))


# ============================================================ the banked R0d log, recomputed with plain loops
def indep_banked():
    """-> the D1 / D2 numbers of the banked log by a different code path (plain loops and `statistics`)."""
    g = [100 * r["con_g_ttc"] for r in TRAIN if r["update"] >= 226]
    p = [100 * r["con_p1_gap_ttc"] for r in TRAIN if r["update"] >= 226]
    w = [a + b for a, b in zip(g, p)]
    se = lambda xs: statistics.stdev(xs) / math.sqrt(len(xs))                       # noqa: E731
    late = [r["epo_J"] for r in TRAIN if 751 <= r["update"] <= 1000]
    j25 = next(r["epo_J"] for r in TRAIN if r["update"] == 25)
    return {"n": len(w), "g": (statistics.mean(g), se(g)), "p": (statistics.mean(p), se(p)), "w": (statistics.mean(w), se(w)), "rise": statistics.mean(late) - j25, "j25": j25, "jl": statistics.mean(late)}


@check("banked:D1_of_the_REAL_R0d_log_matches_an_independent_recomputation")
def _(T, root, tag):
    ref = indep_banked()
    d1 = T.d1_wedge(T.read_log(str(BANKED)))
    assert d1["status"] == "READ" and d1["rows"] == {"first": 250, "last": 1000, "n": 31} == {"first": 250, "last": 1000, "n": ref["n"]}, d1.get("rows")
    for k, key in (("g_x100", "g"), ("p1gap_x100", "p"), ("wedge_x100", "w")):
        assert abs(d1["ttc"][k]["mean"] - ref[key][0]) < 1e-12 and abs(d1["ttc"][k]["se"] - ref[key][1]) < 1e-12, (k, d1["ttc"][k], ref[key])
    return abs(d1["D1_x100"] - ref["w"][0]) < 1e-12 and d1["se"] == d1["ttc"]["wedge_x100"]["se"]


@check("banked:D1_of_the_REAL_R0d_log_reproduces_the_numbers_the_draft_quotes_and_is_NOT_ALIGNED")
def _(T, root, tag):
    d1 = T.d1_wedge(T.read_log(str(BANKED)))
    t = d1["ttc"]
    tol = 1e-3                                                                                                              # the draft quotes three decimals: one unit of its last place (g's se is 0.02549, the draft's 0.026 is a double rounding)
    assert abs(t["g_x100"]["mean"] + 0.085) < tol and abs(t["g_x100"]["se"] - 0.026) < tol, t["g_x100"]               # draft 1.1: g -0.085 +- 0.026
    assert abs(t["p1gap_x100"]["mean"] + 0.232) < tol and abs(t["p1gap_x100"]["se"] - 0.069) < tol, t["p1gap_x100"]    # p1gap -0.232 +- 0.069
    assert abs(d1["D1_x100"] + 0.317) < tol, d1["D1_x100"]                                                                # in-run wedge -0.317
    return d1["aligned"] is False and d1["borderline"] is False and d1["bar"] == 0.10


@check("banked:the_draft_does_not_say_u225_in_or_out_so_the_sensitivity_reads_beside_it_and_the_verdict_is_the_same_on_R0d")
def _(T, root, tag):
    d1 = T.d1_wedge(T.read_log(str(BANKED)))
    s = d1["sensitivity_rows_gt_ramp"]
    w = [100 * (r["con_g_ttc"] + r["con_p1_gap_ttc"]) for r in TRAIN if r["update"] > 200]
    assert s["rows"] == {"first": 225, "last": 1000, "n": 32} and abs(s["D1_x100"] - statistics.mean(w)) < 1e-12, s["rows"]
    return abs(s["D1_x100"] + 0.294) < 5e-4 and s["aligned"] is False and d1["sensitivity_changes_the_verdict"] is False


@check("banked:D2_of_the_REAL_R0d_log_matches_the_draft_0.8171_to_0.8766_rise_0.0595")
def _(T, root, tag):
    lg = T.read_log(str(BANKED))
    ref, pb = T.reference_rise(str(BANKED))
    d2 = T.d2_crowding(lg, ref)
    i = indep_banked()
    assert pb == [] and d2["status"] == "READ" and d2["n_late_rows"] == 10
    assert abs(d2["J_u25"] - i["j25"]) < 1e-15 and abs(d2["J_mean_u751_1000"] - i["jl"]) < 1e-15 and abs(d2["rise"] - i["rise"]) < 1e-15
    assert abs(d2["J_u25"] - 0.8171) < 5e-5 and abs(d2["J_mean_u751_1000"] - 0.8766) < 5e-5 and abs(d2["rise"] - 0.0595) < 5e-5, d2          # the draft's section 6 / NOTES
    return d2["ratio_to_reference"] == 1.0 and d2["crowded"] is False


@check("banked:the_post_ramp_lambda_share_equals_the_counts_another_tool_banked_for_the_same_bytes")
def _(T, root, tag):
    lg = T.read_log(str(BANKED))
    rec = json.loads(BANKED_REC.read_text(encoding="utf-8"))["arms"]["R0d"]["log"]["numbers"]
    c = int(json.loads(BANKED_REC.read_text(encoding="utf-8"))["arms"]["R0d"]["first_breached_check"][len("update"):])
    share, k, m = T.post_ramp_lambda_share(lg, c)
    assert (k, m) == (rec["post_ramp_updates_with_lambda_gt_0"]["value"], rec["post_ramp_updates"]["value"]) == (528, 550) and share == rec["post_ramp_share"]["value"] == 0.96
    full = T.post_ramp_lambda_share(lg)
    last = TRAIN[-1]
    assert full[1] == last["con_lam_pos_updates"] - next(r for r in TRAIN if r["update"] == 200)["con_lam_pos_updates"] and full[2] == 800 and last["con_lam_pos_updates"] == 941      # lambda > 0 on 941 of 1000
    return True


@check("banked:the_per_check_table_is_the_window_mean_of_the_trainers_own_columns")
def _(T, root, tag):
    t = T.per_check(T.read_log(str(BANKED)))
    assert sorted(t) == ["1000", "250", "500", "750"]
    for c in (250, 500, 750, 1000):
        rows = [r for r in TRAIN if c - 99 <= r["update"] <= c]
        row = next(r for r in TRAIN if r["update"] == c)
        assert len(rows) == 4
        x = t[str(c)]
        assert x["status"] == "READ" and x["window"] == [c - 99, c] and x["lam_ttc"] == row["con_lam_ttc"] and x["lam_nc"] == row["con_lam_nc"]
        assert x["lam_pos_updates"] == row["con_lam_pos_updates"] and x["cap_hits"] == row["con_cap_hits"]
        ins = x["in_sample_x100"]
        mean = lambda k: sum(r[k] for r in rows) / 4                                           # noqa: E731
        assert abs(ins["L_T_ttc"] - 100 * mean("con_g_ttc")) < 1e-12 and abs(ins["L_P1_ttc"] + 100 * mean("con_p1_gap_ttc")) < 1e-12           # loss form: L_P1 = -100 p1gap
        assert abs(ins["wedge_ttc"] - (ins["L_T_ttc"] - ins["L_P1_ttc"])) < 1e-12 and abs(ins["L_T_nc"] - 100 * mean("con_g_nc")) < 1e-12 and abs(ins["L_P1_nc"] + 100 * mean("con_p1_gap_nc")) < 1e-12
    return [t[str(c)]["window_overlaps_ramp"] for c in (250, 500, 750, 1000)] == [True, False, False, False]          # u250's window 151..250 overlaps the lambda_pg ramp: said, not moved


# ============================================================ D1: planted values on the trainer's schema
@check("D1:a_planted_constant_wedge_reads_exactly_and_the_rows_before_u226_do_not_count")
def _(T, root, tag):
    d1 = T.d1_wedge(reads(T, planted(g=0.002, p=-0.004), root, tag))                          # wedge = 100 (0.002 - 0.004) = -0.2 on rows u250+; the filler rows read +0.012
    return d1["status"] == "READ" and abs(d1["D1_x100"] + 0.2) < 1e-12 and d1["se"] < 1e-12 and d1["rows"] == {"first": 250, "last": 1000, "n": 31} and d1["first_row_rule"] == "update >= ramp + 26"


@check("D1:the_u225_row_is_excluded_by_the_draft_rule_and_read_in_the_sensitivity")
def _(T, root, tag):
    rows = planted(g=lambda u: 0.5 if u == 225 else 0.002, p=-0.0025, lo=225)                      # wedge -0.05 on u250+ (ALIGNED); the u225 row alone reads +49.55
    d1 = T.d1_wedge(reads(T, rows, root, tag))
    s = d1["sensitivity_rows_gt_ramp"]
    return abs(d1["D1_x100"] + 0.05) < 1e-12 and d1["aligned"] is True and d1["rows"]["first"] == 250 and s["rows"]["first"] == 225 and s["D1_x100"] > 1.0 and s["aligned"] is False and d1["sensitivity_changes_the_verdict"] is True


@check("D1:the_first_row_follows_the_BUILT_ramp_of_the_log_not_a_constant")
def _(T, root, tag):
    rows = planted(g=lambda u: 0.5 if u == 125 else (0.002 if u >= 150 else 0.5), p=-0.004, lo=25, ramp=100)
    d1 = T.d1_wedge(reads(T, rows, root, tag))
    return d1["ramp"] == 100 and d1["rows"]["first"] == 150 and abs(d1["D1_x100"] + 0.2) < 1e-12 and d1["rows"]["n"] == 35


@check("D1:through_the_first_breached_check_stops_there_and_a_check_past_the_log_is_refused")
def _(T, root, tag):
    rows = planted(g=lambda u: 0.002 if u <= 750 else 0.2, p=-0.004)
    lg = reads(T, rows, root, tag)
    d1, full = T.d1_wedge(lg, 750), T.d1_wedge(lg)
    assert abs(d1["D1_x100"] + 0.2) < 1e-12 and d1["rows"] == {"first": 250, "last": 750, "n": 21} and d1["through_update"] == 750, d1
    assert full["D1_x100"] > 1.0
    bad = T.d1_wedge(lg, 1025)
    return bad["status"] == "NOT_READ" and any("not a logged update" in p for p in bad["problems"])


@check("D1:the_bar_is_inclusive_and_the_wedge_is_signed_by_its_magnitude")
def _(T, root, tag):
    a = 2.0 ** -10                                                                                 # 100 * 2^-10 = 0.09765625 exactly, dyadic: no rounding
    for sign in (1, -1):
        lg = reads(T, planted(g=sign * a, p=0.0), root, tag, name=f"s{sign}")
        d1 = T.d1_wedge(lg, bar=100 * a)
        assert d1["D1_x100"] == sign * 100 * a and d1["aligned"] is True, d1["D1_x100"]
        assert T.d1_wedge(lg, bar=math.nextafter(100 * a, 0.0))["aligned"] is False
    return T.D1_BAR == 0.10


ALT = lambda u: 0.0030 if (u // 25) % 2 else -0.0006                                              # noqa: E731  rows alternate +0.30 / -0.06 points: mean +0.114, se 0.032 over u250..u1000


@check("D1:borderline_means_within_one_standard_error_of_the_bar")
def _(T, root, tag):
    wide = T.d1_wedge(reads(T, planted(g=ALT, p=0.0), root, tag, name="wide"))
    xs = [100 * ALT(u) for u in range(250, 1001, 25)]
    m, se = statistics.mean(xs), statistics.stdev(xs) / math.sqrt(len(xs))
    assert 0.10 < m and abs(m - 0.10) <= se and abs(wide["D1_x100"] - m) < 1e-12, (m, se)                  # the fixture: outside the bar by less than one se
    assert wide["aligned"] is False and wide["borderline"] is True, (wide["D1_x100"], wide["se"])
    far2 = lambda u: 0.0030 if (u // 25) % 2 else -0.0002                                           # noqa: E731  mean +0.135, se 0.029: outside the bar by 1.2 se (not borderline; a 3-se rule would flag it)
    xs2 = [100 * far2(u) for u in range(250, 1001, 25)]
    m2, se2 = statistics.mean(xs2), statistics.stdev(xs2) / math.sqrt(len(xs2))
    assert se2 < abs(m2 - 0.10) < 3 * se2, (m2, se2)
    mid = T.d1_wedge(reads(T, planted(g=far2, p=0.0), root, tag, name="mid"))
    assert mid["aligned"] is False and mid["borderline"] is False, (mid["D1_x100"], mid["se"])
    const = T.d1_wedge(reads(T, planted(g=0.0012, p=0.0), root, tag, name="const"))
    far = T.d1_wedge(reads(T, planted(g=0.003, p=0.0), root, tag, name="far"))
    return const["borderline"] is False and far["borderline"] is False and abs(far["D1_x100"] - 0.3) < 1e-12


@check("D1:the_standard_error_is_the_sample_one_ddof_1_over_sqrt_n")
def _(T, root, tag):
    d1 = T.d1_wedge(reads(T, planted(g=ALT, p=0.0), root, tag))
    xs = [100 * ALT(u) for u in range(250, 1001, 25)]
    return abs(d1["se"] - statistics.stdev(xs) / math.sqrt(len(xs))) < 1e-12 and abs(d1["D1_x100"] - statistics.mean(xs)) < 1e-12


@check("D1:the_NC_wedge_is_read_the_same_way_and_does_not_decide")
def _(T, root, tag):
    rows = planted(g=0.002, p=-0.004, mutate=lambda r: {**r, "con_g_nc": 0.01, "con_p1_gap_nc": -0.0125} if r["update"] >= 250 else r)
    d1 = T.d1_wedge(reads(T, rows, root, tag))
    return abs(d1["nc_descriptive"]["wedge_x100"]["mean"] + 0.25) < 1e-12 and abs(d1["D1_x100"] + 0.2) < 1e-12


# ============================================================ D2
def jplan(j25, late, other=0.7, row750=None):
    return lambda u: j25 if u == 25 else (late if u >= 751 else (row750 if (u == 750 and row750 is not None) else other))


@check("D2:the_rise_is_the_mean_over_u751_1000_minus_the_u25_row_and_u750_is_not_in_the_window")
def _(T, root, tag):
    lg = reads(T, planted(J=jplan(0.5, 0.625, row750=100.0)), root, tag)
    d2 = T.d2_crowding(lg, 0.5)
    return d2["status"] == "READ" and d2["rise"] == 0.125 and d2["n_late_rows"] == 10 and d2["J_u25"] == 0.5 and d2["J_mean_u751_1000"] == 0.625


@check("D2:crowded_is_strictly_less_than_half_the_reference_rise")
def _(T, root, tag):
    lg = reads(T, planted(J=jplan(0.5, 0.625)), root, tag)                                       # rise 0.125 exactly
    eq, below, above = T.d2_crowding(lg, 0.25), T.d2_crowding(lg, math.nextafter(0.25, 1.0)), T.d2_crowding(lg, math.nextafter(0.25, 0.0))
    return eq["crowded"] is False and below["crowded"] is True and above["crowded"] is False and eq["ratio_to_reference"] == 0.5 and eq["fraction"] == 0.5


@check("D2:a_log_that_ends_before_u1000_or_a_missing_reference_is_NOT_READ_never_crowded_by_default")
def _(T, root, tag):
    partial = T.d2_crowding(reads(T, planted(J=jplan(0.5, 0.625), last=900), root, tag, name="p"), 0.5)         # u751..900 exist: a shorter window mean must not be read as D2
    assert partial["status"] == "NOT_READ" and "rise" not in partial and any("ends at u900" in p for p in partial["problems"]), partial
    killed = T.d2_crowding(reads(T, planted(J=jplan(0.5, 0.625), last=525), root, tag, name="k"), 0.5)
    lg = reads(T, planted(J=jplan(0.5, 0.625)), root, tag, name="f")
    none, zero, neg = T.d2_crowding(lg, None), T.d2_crowding(lg, 0.0), T.d2_crowding(lg, -0.1)
    return killed["status"] == "NOT_READ" and all(x["status"] == "NOT_READ" and "crowded" not in x for x in (killed, none, zero, neg)) and any("ends at u525" in p for p in killed["problems"])


@check("D2:a_run_that_gained_nothing_is_crowded_and_a_run_that_gained_more_than_the_reference_is_not")
def _(T, root, tag):
    flat = T.d2_crowding(reads(T, planted(J=jplan(0.5, 0.5)), root, tag, name="a"), 0.25)
    big = T.d2_crowding(reads(T, planted(J=jplan(0.5, 1.0)), root, tag, name="b"), 0.25)
    down = T.d2_crowding(reads(T, planted(J=jplan(0.5, 0.25)), root, tag, name="c"), 0.25)
    return flat["crowded"] is True and big["crowded"] is False and big["ratio_to_reference"] == 2.0 and down["crowded"] is True and down["rise"] < 0


# ============================================================ refusals: fail closed
@check("refuse:a_missing_non_finite_or_non_contiguous_log_reads_NOT_READ_and_no_number")
def _(T, root, tag):
    cases = {
        "col": planted(mutate=lambda r: {k: v for k, v in r.items() if k != "con_p1_gap_ttc"} if r["update"] == 400 else r),
        "nan": planted(mutate=lambda r: {**r, "con_g_ttc": float("nan")} if r["update"] == 400 else r),
        "str": planted(mutate=lambda r: {**r, "epo_J": "0.8"} if r["update"] == 400 else r),
        "gap": planted(drop={400}),
        "noramp": [({**r, "epo": {k: v for k, v in r["epo"].items() if k != "ramp"}} if r.get("event") == "data" else r) for r in BASE_ROWS],
        "notrain": [r for r in BASE_ROWS if r.get("event") != "train"],
    }
    for nm, rows in cases.items():
        lg = reads(T, rows, root, tag, name=nm)
        d1, d2 = T.d1_wedge(lg), T.d2_crowding(lg, 0.0595)
        assert lg["readable"] is False and d1["status"] == "NOT_READ" and d2["status"] == "NOT_READ" and "D1_x100" not in d1 and "rise" not in d2, nm
        assert T.per_check(lg) == {} and T.post_ramp_lambda_share(lg) is None, nm
    miss = T.read_log(str(Path(root) / f"{tag}_no_such.jsonl"))
    return miss["readable"] is False and any("missing" in p for p in miss["problems"])


@check("refuse:a_half_written_final_line_is_a_note_two_bad_lines_are_a_problem_and_a_missing_window_row_is_never_a_shorter_window")
def _(T, root, tag):
    p = write_log(Path(root) / f"{tag}_half", BASE_ROWS)
    p.write_text(p.read_text(encoding="utf-8") + '{"event": "train", "update": 1025, "con_l', encoding="utf-8")
    lg = T.read_log(str(p))
    assert lg["readable"] is True and any("half-written" in n for n in lg["notes"])
    bad = T.read_log(str(write_log(Path(root) / f"{tag}_two_bad", BASE_ROWS)))
    q = Path(bad["path"])
    q.write_text('{"half\n' + q.read_text(encoding="utf-8") + '{"half2\n', encoding="utf-8")
    assert T.read_log(str(q))["readable"] is False
    # a window with a row missing: the log is non-contiguous, so it is refused as a whole (never a quietly shorter window)
    return T.per_check(T.read_log(str(write_log(Path(root) / f"{tag}_gap", planted(drop={475}))))) == {}


# ============================================================ the reading aid for section 6
@check("row:O4_splits_on_D1_and_O3_on_the_lambda_share_and_D2_and_nothing_else_is_refined")
def _(T, root, tag):
    ok_d1 = {"status": "READ", "aligned": True, "borderline": False, "D1_x100": 0.05, "se": 0.01}
    bad_d1 = {**ok_d1, "aligned": False, "D1_x100": -0.3}
    bord = {**bad_d1, "borderline": True}
    d2_c = {"status": "READ", "crowded": True, "rise": 0.01, "ratio_to_reference": 0.2}
    d2_n = {**d2_c, "crowded": False, "ratio_to_reference": 1.0}
    unread = {"status": "NOT_READ", "problems": ["x"]}
    r = T.refine_row
    assert r("O4", [], ok_d1, d2_n, None)[0] == "P-O4-PICK" and r("O4", [], bad_d1, d2_n, None)[0] == "P-O4-WEDGE" and r("O4", [], bord, d2_n, None)[0] == "P-O4-WEDGE (BORDERLINE)"
    assert r("O4", ["SLACK"], bad_d1, d2_n, None)[0] == "O4" and r("O4", [], unread, d2_n, None)[0] == "UNREAD"
    assert r("O3", [], bad_d1, d2_c, (0.9, 90, 100))[0] == "O3-a" and r("O3", [], bad_d1, d2_n, (0.9, 90, 100))[0] == "O3-b"
    assert r("O3", [], bad_d1, d2_c, (0.5, 50, 100))[0] == "O3-a" and r("O3", [], bad_d1, d2_c, (0.49, 49, 100))[0] == "O3 SLACK"
    assert r("O3", [], bad_d1, unread, (0.9, 90, 100))[0] == "UNREAD" and r("O3", [], bad_d1, d2_c, None)[0] == "UNREAD"
    return all(r(o, [], bad_d1, d2_c, (0.9, 90, 100))[0] == o for o in ("O1", "O2", "O5", "O6", "O7", "O4+O5"))


# ============================================================ D3: the validation side
def np_():
    import numpy as np
    return np


@check("D3:the_validation_side_on_planted_arrays_is_the_hand_computed_one")
def _(T, root, tag):
    np = np_()
    # 4 sets in 2 logs, 3 proposals.  Labels y; arm A's best-scored proposal is index 0 everywhere, arm B's is index 1 on sets 0 and 2 (clear margins, no near-tie).
    y = np.array([[1.0, 0.0, 0.5], [1.0, 0.5, 0.0], [0.5, 1.0, 0.0], [1.0, 1.0, 0.0]])
    lA = np.array([[0.0, -5.0, -9.0]] * 4)
    lB = np.array([[-0.05, 0.0, -9.0], [0.0, -5.0, -9.0], [-5.0, 0.0, -9.0], [0.0, -5.0, -9.0]])        # set 0: B's margin is 0.05 (> tau0 = 0.002: still a clear pick, but at T = 0.1 the softmax is NOT one-hot)
    logs = ["l0", "l0", "l1", "l1"]
    v = T.val_side(lA, lB, y, logs, 1e-9, 0.002, n_boot=200)
    # pick labels: A -> [1, 1, 0.5, 1]; B -> [0, 1, 1, 1];  d = 100 (B - A) = [-100, 0, +50, 0];  L = -mean(d) = +12.5 (B worse on average)
    assert v["L_P1_argmax"]["value"] == 12.5 and v["L_P1_tie_tau0"]["value"] == 12.5 and abs(v["L_T"]["value"] - 12.5) < 1e-9, v      # T -> 0 is the pick
    assert v["n_sets"] == 4 and v["n_logs"] == 2
    z = T.val_side(lA, lA, y, logs, 0.1, 0.002, n_boot=200)
    assert z["L_T"]["value"] == 0.0 and z["L_P1_argmax"]["value"] == 0.0 and z["L_P1_tie_tau0"]["value"] == 0.0               # the zero control: B := A reads EXACTLY 0
    hot = T.val_side(lA, lB, y, logs, 0.1, 0.002, n_boot=200)
    assert hot["L_T"]["value"] != v["L_T"]["value"], "at T = 0.1 the expectation is not the pick"                                 # the twin of the T -> 0 identity: it is not vacuous
    lo, hi = v["L_P1_argmax"]["ci"]
    return lo <= 12.5 <= hi and T.d3_gap({"L_T_ttc": 10.0, "L_P1_ttc": 20.0, "wedge_ttc": -10.0}, v)["gap_expectation"] == v["L_T"]["value"] - 10.0 and T.d3_gap({"L_T_ttc": 10.0, "L_P1_ttc": 20.0, "wedge_ttc": -10.0}, v)["gap_pick_argmax"] == 12.5 - 20.0


@check("D3:the_tie_aware_pick_averages_the_labels_within_tau_of_the_top_and_clear_sets_are_unchanged")
def _(T, root, tag):
    np = np_()
    y = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    lA = np.array([[0.0, -3.0, -9.0], [0.0, -3.0, -9.0]])
    lB = np.array([[0.0, -1e-3, -9.0], [0.0, -3.0, -9.0]])                                           # set 0: a near-tie in B (gap 1e-3), argmax = proposal 0 (unsafe); set 1: clear
    tie = T.val_side(lA, lB, y, ["a", "b"], 0.1, 2e-3, n_boot=50)                                    # tau >= the gap: B's set-0 label is the tie mean 0.5, A's is 0: d = +50 on set 0
    arg = T.val_side(lA, lB, y, ["a", "b"], 0.1, 5e-4, n_boot=50)                                    # tau < the gap: the argmax label 0: nothing moves
    lBx = np.array([[0.0, -(2.0 ** -9), -9.0], [0.0, -3.0, -9.0]])                                   # set 0: the runner-up is EXACTLY tau below the top (dyadic): it is in Q(tau) (>=)
    edge = T.val_side(lA, lBx, y, ["a", "b"], 0.1, 2.0 ** -9, n_boot=50)
    return (tie["L_P1_tie_tau0"]["value"] == -25.0 and arg["L_P1_tie_tau0"]["value"] == 0.0 and tie["L_P1_argmax"]["value"] == 0.0 and abs(T.tie_pick_mean(lB, y, 2e-3)[0] - 0.5) < 1e-15
            and edge["L_P1_tie_tau0"]["value"] == -25.0)


@check("D3:the_softmax_is_stable_and_normalised_and_T_changes_it")
def _(T, root, tag):
    np = np_()
    l = np.array([[0.0, -1.0, -2.0, -50.0], [-1e4, -1e4 - 1.0, -1e4 - 2.0, -1e4 - 50.0]])
    p = T.softmax_T(l, 0.01)
    return np.allclose(p.sum(-1), 1.0, atol=1e-12) and np.all(np.isfinite(p)) and np.allclose(p[0], p[1], atol=1e-12) and not np.allclose(T.softmax_T(l, 0.1), p)


@check("D3:the_validation_functions_equal_the_A2R2_harness_on_random_arrays_when_the_harness_is_importable")
def _(T, root, tag):
    np = np_()
    try:
        sys.path.insert(0, str(PKROOT / "raw" / "2026-10-18-so-pickcon"))
        os.environ.setdefault("SOCARVE_ARCH", str(PKROOT))
        if (PKROOT / "raw" / "2026-10-18-so-carve").is_dir():
            os.environ.setdefault("SOCARVE_DIR", str(PKROOT / "raw" / "2026-10-18-so-carve"))
        import so_a2r2_tieaware as T2
    except Exception as e:                                                                          # noqa: BLE001
        print(f"      SKIPPED (the A2R2 harness is not importable here: {type(e).__name__})", flush=True)
        return True
    NT = T2.NT
    rng = np.random.default_rng(7)
    l = rng.normal(size=(40, 64)) * 0.3
    l2 = l + rng.normal(size=l.shape) * 0.05
    y = rng.integers(0, 3, size=l.shape) / 2.0
    ok = all(np.abs(T.softmax_T(l, t) - NT.pol_l(l, t)).max() < 1e-12 for t in (0.1, 0.02, 0.01))
    for tau in (0.0, 1e-3, T2.TAU0, 1e-2):
        ok &= bool(np.abs(100.0 * (T.tie_pick_mean(l2, y, tau) - T.tie_pick_mean(l, y, tau)) - T2.d_tie(l, l2, y, tau)).max() < 1e-12)
        ok &= bool(np.abs(T.tie_pick_mean(l, y, tau) - T2.qmean(l, y, tau)).max() < 1e-15)
    ok &= bool(abs(T.eps_eq() - T2.TAU0) == 0.0)
    return ok


@check("real_states:the_validation_side_reproduces_the_banked_A2R2_numbers_on_the_banked_R0_states", heavy=True)
def _(T, root, tag):
    if not (R0_STATES.is_dir() and R0_LABELS.is_dir() and A2R2_BANKED.exists()):
        print("      SKIPPED (the banked R0 states are not on this machine)", flush=True)
        return True
    np = np_()
    AT, keys, cmp_, ck = T.load_val(str(PKROOT), str(R0_LABELS / "pdm_val" / "pdm_val"), str(R0_LABELS / "terms_val" / "terms_val"))
    banked = json.loads(A2R2_BANKED.read_text(encoding="utf-8"))
    tau0 = float(banked["tau0"])                                                                    # the BANKED tau0, never the tool's own: a tool whose tau drifted to another grid point (1e-3) would otherwise be checked against that point's banked row and pass
    assert T.eps_eq() == tau0, (T.eps_eq(), tau0)
    logs = [k[0] for k in keys]
    for u in (250, 1000):
        lA = np.log(AT.sel_v1_lane(AT.RV.sigmoid(AT.load_scores_aligned(str(R0_STATES / f"scores_update{u}_A.npz"), keys, ck))) + AT.EPS)
        lB = np.log(AT.sel_v1_lane(AT.RV.sigmoid(AT.load_scores_aligned(str(R0_STATES / f"scores_update{u}_B.npz"), keys, ck))) + AT.EPS)
        for t in (0.1, 0.01):
            v = T.val_side(lA, lB, cmp_["ttc"], logs, t, tau0, n_boot=100)
            b = banked["states"][f"r0_u{u}"]
            assert abs(v["L_T"]["value"] - b["T"][str(t)]["L_T_all"]) < 1e-9, (u, t, v["L_T"]["value"], b["T"][str(t)]["L_T_all"])
            assert abs(v["L_P1_tie_tau0"]["value"] - b["tau"][str(tau0)]["L_P1_tau"]) < 1e-9
            assert abs((v["L_T"]["value"] - v["L_P1_tie_tau0"]["value"]) - b["tau"][str(tau0)]["diff"][str(t)]["diff"]) < 1e-9               # the A2R2 `diff`, e.g. -0.103 at T 0.01, u1000
    return True


# ============================================================ pins: the registered numbers against the draft's text
def draft_text():
    for p in DRAFT:
        if p.exists():
            return p.read_text(encoding="utf-8")
    return None


@check("pin:the_constants_against_the_drafts_section_8_and_6_text")
def _(T, root, tag):
    txt = draft_text()
    if txt is None:                                                                                 # a package shipped without raw/
        return True
    assert re.search(r"\|D1\| ≤ 0\.10", txt) and T.D1_BAR == 0.10
    assert "u751–1000" in txt and (T.D2_LATE_FROM, 1000) == (751, 1000) and "u25 row" in txt and T.D2_FROM_ROW == 25
    assert "less than half of R0d's rise" in txt and T.D2_FRACTION == 0.5
    assert "31 post-ramp rows u226–u1000" in txt and T.D1_FIRST_ROW_OFFSET == 26 and T.REGISTERED_RAMP == 200
    assert "λ > 0 on ≥ 50 % post-ramp" in txt.replace("**", "") or "λ > 0 on ≥ 50 %" in txt
    assert T.LAM_POS_SHARE_MIN == 0.5 and T.CHECKS == (250, 500, 750, 1000) and T.WINDOW_UPDATES == 100 and (T.N_BOOT, T.SEED) == (10000, 20260927)
    return T.LOG_EVERY == 25


@check("pin:the_trainers_log_columns_the_tool_reads_exist_in_the_trainers_row_writer")
def _(T, root, tag):
    sf = PKROOT / "refe" / "scorer_finetune.py"
    if not sf.exists():
        return True
    src = sf.read_text(encoding="utf-8")
    for k in T.REQ:
        if k == "update":
            continue
        assert f'"{k}"' in src, k
    return all(k in TRAIN[0] for k in T.REQ)                                                         # and every one is in the REAL banked row


# ============================================================ the CLI
def cli_checks(tpath, root, tag):
    chk = []

    def run(*args):
        return subprocess.run([sys.executable, "-X", "utf8", str(tpath), *args], capture_output=True, text=True, timeout=300)
    out = Path(root) / f"cli_{tag}"
    out.mkdir(exist_ok=True)
    arm = out / "arm"
    arm.mkdir(exist_ok=True)
    (arm / "sft.jsonl").write_bytes(BANKED.read_bytes())
    r = run("--arm-dir", str(arm), "--reference-log", str(BANKED), "--out", str(out / "a.json"))
    rec = json.loads((out / "a.json").read_text(encoding="utf-8")) if (out / "a.json").exists() else {}
    chk.append(("rc0_and_the_D1_D2_lines", r.returncode == 0 and "ZZR0E_DIAG D1 READ -0.3165 +- 0.0524 points over 31 rows u250..u1000: NOT ALIGNED" in r.stdout and "ZZR0E_DIAG D2 READ rise +0.0595" in r.stdout and "= 1.000 x the reference" in r.stdout))
    body = json.dumps({k: v for k, v in rec.items() if k != "record_sha256"}, sort_keys=True, separators=(",", ":"), default=str) if rec else ""
    chk.append(("writes_a_self_consistent_record", bool(rec) and rec["record_sha256"] == hashlib.sha256(body.encode("utf-8")).hexdigest() and rec["D1"]["status"] == "READ" and rec["log"]["sha256"] == hashlib.sha256(BANKED.read_bytes()).hexdigest()))
    chk.append(("the_per_check_table_is_in_the_record", sorted(rec.get("per_check", {})) == ["1000", "250", "500", "750"]))
    r = run("--arm-dir", str(arm), "--reference-log", str(BANKED), "--outcome-json", str(BANKED_REC), "--slot", "R0d", "--out", str(out / "b.json"))
    rb = json.loads((out / "b.json").read_text(encoding="utf-8")) if (out / "b.json").exists() else {}
    chk.append(("outcome_json_sets_through_the_first_breached_check_and_refines_O4", r.returncode == 0 and "over 21 rows u250..u750" in r.stdout and "ZZR0E_DIAG ROW P-O4-WEDGE (O4)" in r.stdout
                and rb.get("D1", {}).get("through_update") == 750 and rb.get("lambda_post_ramp_share", {}).get("m") == 550))
    short = out / "short"
    write_log(short, planted(J=jplan(0.5, 0.625), last=525))
    r = run("--arm-dir", str(short), "--reference-log", str(BANKED), "--out", str(out / "c.json"))
    chk.append(("rc2_for_a_killed_log_whose_D2_is_NOT_READ", r.returncode == 2 and "ZZR0E_DIAG D2 NOT_READ" in r.stdout and "ZZR0E_DIAG D1 READ" in r.stdout))
    r = run("--arm-dir", str(arm), "--reference-log", str(out / "no_such_reference.jsonl"), "--out", str(out / "d.json"))
    chk.append(("rc2_when_the_reference_log_is_missing_D2_NOT_READ", r.returncode == 2 and "ZZR0E_DIAG D2 NOT_READ" in r.stdout))
    chk.append(("rc3_without_an_arm_dir", run("--arm-dir", str(out / "no_such_dir")).returncode == 3))
    chk.append(("rc3_for_outcome_json_without_slot", run("--arm-dir", str(arm), "--outcome-json", str(BANKED_REC)).returncode == 3))
    chk.append(("rc3_for_a_slot_the_record_does_not_have", run("--arm-dir", str(arm), "--outcome-json", str(BANKED_REC), "--slot", "R1d").returncode == 3))
    chk.append(("rc3_for_through_update_beside_outcome_json", run("--arm-dir", str(arm), "--outcome-json", str(BANKED_REC), "--slot", "R0d", "--through-update", "500").returncode == 3))
    r = run("--arm-dir", str(arm), "--reference-log", str(BANKED), "--through-update", "500", "--out", str(out / "e.json"))
    chk.append(("through_update_is_honoured", r.returncode == 0 and "u250..u500" in r.stdout))
    r = run("--arm-dir", str(arm), "--reference-log", str(BANKED), "--scores-dir", str(out), "--out", str(out / "f.json"))
    chk.append(("d3_without_its_registration_and_labels_is_NOT_READ_rc2", r.returncode == 2 and "ZZR0E_DIAG D3 NOT_READ" in r.stdout))
    r = run("--arm-dir", str(arm), "--reference-log", str(BANKED), "--scores-dir", str(out), "--registered-temp", "0.01", "--val-pdm", str(out), "--val-terms", str(out), "--out", str(out / "g.json"))
    chk.append(("d3_refuses_a_T_the_log_did_not_run_at", r.returncode == 2 and "ZZR0E_DIAG D3 NOT_READ" in r.stdout and "is not the T the log says the constraint RAN at" in ((json.loads((out / "g.json").read_text(encoding="utf-8")) if (out / "g.json").exists() else {}).get("D3", {}).get("problems") or [""])[0]))
    return chk


def run_suite(src, root, tag, *, e2e=False, heavy=False):
    """-> the names of the failing checks for the tool whose source is `src`."""
    bad = []
    tpath = Path(root) / f"tool_{tag}" / "r0e_diagnostics.py"
    try:
        T = load_tool(src, tpath)
    except Exception as e:                                                                            # noqa: BLE001
        return [f"IMPORT:{type(e).__name__}:{str(e)[:60]}"]

    def t(name, fn):
        try:
            ok = bool(fn())
        except Exception:                                                                              # noqa: BLE001
            ok = False
        if not ok:
            bad.append(name)
    for nm, fn in CHECKS + (HEAVY if heavy else []):
        t(nm, lambda fn=fn: fn(T, root, tag))
    if e2e:
        try:
            for nm, ok in cli_checks(tpath, root, tag):
                t(f"e2e:{nm}", lambda ok=ok: ok)
        except Exception as e:                                                                         # noqa: BLE001
            bad.append(f"e2e:CRASH:{type(e).__name__}")
    return bad


# ============================================================ mutations of the tool's source; each must turn the checks it exists for RED
def MU(name, old, new, expect):
    return (name, old, new, set(expect))


B_ALL = "banked:D1_of_the_REAL_R0d_log_matches_an_independent_recomputation"
B_DRAFT = "banked:D1_of_the_REAL_R0d_log_reproduces_the_numbers_the_draft_quotes_and_is_NOT_ALIGNED"
B_D2 = "banked:D2_of_the_REAL_R0d_log_matches_the_draft_0.8171_to_0.8766_rise_0.0595"
B_SENS = "banked:the_draft_does_not_say_u225_in_or_out_so_the_sensitivity_reads_beside_it_and_the_verdict_is_the_same_on_R0d"
B_LAM = "banked:the_post_ramp_lambda_share_equals_the_counts_another_tool_banked_for_the_same_bytes"
B_TAB = "banked:the_per_check_table_is_the_window_mean_of_the_trainers_own_columns"
D1_PLANT = "D1:a_planted_constant_wedge_reads_exactly_and_the_rows_before_u226_do_not_count"
D1_225 = "D1:the_u225_row_is_excluded_by_the_draft_rule_and_read_in_the_sensitivity"
D1_RAMP = "D1:the_first_row_follows_the_BUILT_ramp_of_the_log_not_a_constant"
D1_THR = "D1:through_the_first_breached_check_stops_there_and_a_check_past_the_log_is_refused"
D1_BAR = "D1:the_bar_is_inclusive_and_the_wedge_is_signed_by_its_magnitude"
D1_BORD = "D1:borderline_means_within_one_standard_error_of_the_bar"
D1_SE = "D1:the_standard_error_is_the_sample_one_ddof_1_over_sqrt_n"
D1_NC = "D1:the_NC_wedge_is_read_the_same_way_and_does_not_decide"
D2_RISE = "D2:the_rise_is_the_mean_over_u751_1000_minus_the_u25_row_and_u750_is_not_in_the_window"
D2_HALF = "D2:crowded_is_strictly_less_than_half_the_reference_rise"
D2_KILL = "D2:a_log_that_ends_before_u1000_or_a_missing_reference_is_NOT_READ_never_crowded_by_default"
D2_FLAT = "D2:a_run_that_gained_nothing_is_crowded_and_a_run_that_gained_more_than_the_reference_is_not"
REF = "refuse:a_missing_non_finite_or_non_contiguous_log_reads_NOT_READ_and_no_number"
ROW = "row:O4_splits_on_D1_and_O3_on_the_lambda_share_and_D2_and_nothing_else_is_refined"
V_HAND = "D3:the_validation_side_on_planted_arrays_is_the_hand_computed_one"
V_TIE = "D3:the_tie_aware_pick_averages_the_labels_within_tau_of_the_top_and_clear_sets_are_unchanged"
V_SM = "D3:the_softmax_is_stable_and_normalised_and_T_changes_it"
V_HARN = "D3:the_validation_functions_equal_the_A2R2_harness_on_random_arrays_when_the_harness_is_importable"
V_REAL = "real_states:the_validation_side_reproduces_the_banked_A2R2_numbers_on_the_banked_R0_states"
PIN = "pin:the_constants_against_the_drafts_section_8_and_6_text"
MUTATIONS = [
    MU("D1 bar 0.10 -> 0.35", "D1_BAR = 0.10 ", "D1_BAR = 0.35 ", {B_DRAFT, PIN, D1_BAR}),
    MU("D1 aligned is strict (< for <=)", "\"aligned\": bool(abs(d1) <= bar)", "\"aligned\": bool(abs(d1) < bar)", {D1_BAR}),
    MU("D1 reads the wrong sign: g - p1gap instead of g + p1gap", "w = [a + b for a, b in zip(g, p)]", "w = [a - b for a, b in zip(g, p)]", {B_ALL, B_DRAFT, D1_PLANT}),
    MU("D1 rows from the u225 row (update > ramp) instead of the draft's u226", "D1_FIRST_ROW_OFFSET = LOG_EVERY + 1 ", "D1_FIRST_ROW_OFFSET = 1 ", {B_ALL, B_DRAFT, D1_225, D1_PLANT, PIN}),
    MU("D1 first row is a constant 250, not ramp-relative", "rows = _rows_between(trains, ramp + D1_FIRST_ROW_OFFSET, hi)", "rows = _rows_between(trains, 250, hi)", {D1_RAMP}),
    MU("D1 reads the whole run, ignoring the through-update", "hi = last if through is None else through", "hi = last", {D1_THR}),
    MU("D1 accepts a through-update past the log", "if not _num(hi) or hi > last:", "if not _num(hi):", {D1_THR}),
    MU("D1 standard error uses ddof 0", "var = math.fsum((x - m) ** 2 for x in xs) / (n - 1)", "var = math.fsum((x - m) ** 2 for x in xs) / n", {B_ALL, D1_SE}),
    MU("D1 borderline is never flagged", "\"borderline\": bool(abs(abs(d1) - bar) <= se)", "\"borderline\": False", {D1_BORD}),
    MU("D1 borderline is flagged within 3 standard errors", "\"borderline\": bool(abs(abs(d1) - bar) <= se)", "\"borderline\": bool(abs(abs(d1) - bar) <= 3 * se)", {D1_BORD}),
    MU("D1 reads the NC wedge off the TTC columns", "\"nc_descriptive\": _wedge_block(rows, \"nc\"),", "\"nc_descriptive\": _wedge_block(rows, \"ttc\"),", {D1_NC}),
    MU("D1 sensitivity block is dropped from the verdict flag", "out[\"sensitivity_changes_the_verdict\"] = bool(out[\"aligned\"] != out[\"sensitivity_rows_gt_ramp\"][\"aligned\"])", "out[\"sensitivity_changes_the_verdict\"] = False", {D1_225}),
    MU("D2 late window starts at u726", "D2_LATE_FROM, D2_FROM_ROW = 751, 25", "D2_LATE_FROM, D2_FROM_ROW = 726, 25", {D2_RISE, B_D2, PIN}),
    MU("D2 compares against the u50 row, not the u25 row", "D2_LATE_FROM, D2_FROM_ROW = 751, 25", "D2_LATE_FROM, D2_FROM_ROW = 751, 50", {D2_RISE, B_D2, PIN}),
    MU("D2 crowded is <= half", "out[\"crowded\"] = bool(rise < fraction * ref_rise)", "out[\"crowded\"] = bool(rise <= fraction * ref_rise)", {D2_HALF}),
    MU("D2 fraction 0.5 -> 0.25", "D2_FRACTION = 0.5 ", "D2_FRACTION = 0.25 ", {PIN, D2_HALF}),
    MU("D2 accepts a log that ends before u1000", "if D2_FROM_ROW not in by_u or last < 1000 or not late:", "if D2_FROM_ROW not in by_u or not late:", {D2_KILL}),
    MU("D2 treats a missing reference as no crowding", "    if ref_rise is None or not _num(ref_rise):\n        out[\"status\"] = \"NOT_READ\"", "    if False:\n        out[\"status\"] = \"NOT_READ\"", {D2_KILL}),
    MU("D2 accepts a non-positive reference rise", "    if ref_rise <= 0:\n", "    if False:\n", {D2_KILL}),
    MU("a non-finite log value is accepted", "miss = [k for k in REQ if not _num(r.get(k))]", "miss = [k for k in REQ if k not in r]", {REF}),
    MU("the contiguous 25-update series is not required", "if ups != list(range(LOG_EVERY, LOG_EVERY * len(trains) + 1, LOG_EVERY)):", "if False:", {REF}),
    MU("a missing built ramp is accepted", "if not isinstance(ramp, int) or isinstance(ramp, bool) or ramp < 0:", "if False:", {REF}),
    MU("a half-written final line is a problem", "(out[\"notes\"] if (bad == 1 and last_bad) else pb)", "pb", {"refuse:a_half_written_final_line_is_a_note_two_bad_lines_are_a_problem_and_a_missing_window_row_is_never_a_shorter_window"}),
    MU("per-check window is 50 updates", "WINDOW_UPDATES = 100 ", "WINDOW_UPDATES = 50 ", {B_TAB, PIN}),
    MU("per-check pick loss keeps the log's sign", "\"L_P1_ttc\": -100.0 * m(\"con_p1_gap_ttc\")", "\"L_P1_ttc\": 100.0 * m(\"con_p1_gap_ttc\")", {B_TAB}),
    MU("per-check window overlap with the ramp is never disclosed", "\"window_overlaps_ramp\": bool(need[0] - LOG_EVERY + 1 <= ramp)", "\"window_overlaps_ramp\": False", {B_TAB}),
    MU("the lambda share counts from update 0, not from the ramp end", "k, m = by_u[end][\"con_lam_pos_updates\"] - by_u[ramp][\"con_lam_pos_updates\"], end - ramp", "k, m = by_u[end][\"con_lam_pos_updates\"], end", {B_LAM}),
    MU("O4 is not refined on D1", "return (\"P-O4-PICK\" if d1[\"aligned\"] else \"P-O4-WEDGE\")", "return (\"P-O4-WEDGE\" if d1[\"aligned\"] else \"P-O4-PICK\")", {ROW, "e2e:outcome_json_sets_through_the_first_breached_check_and_refines_O4"}),
    MU("O3 SLACK threshold 50 % -> 60 %", "LAM_POS_SHARE_MIN = 0.5 ", "LAM_POS_SHARE_MIN = 0.6 ", {ROW, PIN}),
    MU("O3 crowded and not crowded are swapped", "return (\"O3-a\" if d2[\"crowded\"] else \"O3-b\")", "return (\"O3-b\" if d2[\"crowded\"] else \"O3-a\")", {ROW}),
    MU("validation L_T has the wrong sign", "return {\"value\": pt, \"ci\"", "return {\"value\": -pt, \"ci\"", {V_HAND, V_TIE}),
    MU("validation L_T ignores T (always 0.1)", "dT = 100.0 * ((softmax_T(lB, T) * y).sum(-1) - (softmax_T(lA, T) * y).sum(-1))", "dT = 100.0 * ((softmax_T(lB, 0.1) * y).sum(-1) - (softmax_T(lA, 0.1) * y).sum(-1))", {V_HAND}),
    MU("validation tie-aware pick uses the argmax", "dt = 100.0 * (tie_pick_mean(lB, y, tau0) - tie_pick_mean(lA, y, tau0))", "dt = d1", {V_TIE}),
    MU("the tie set is strict (> not >=)", "w = (l >= l.max(-1, keepdims=True) - tau).astype(np.float64)", "w = (l > l.max(-1, keepdims=True) - tau).astype(np.float64)", {V_TIE, V_HARN}),
    MU("the softmax is not max-shifted", "    z = z - z.max(-1, keepdims=True)\n    e = np.exp(z)", "    e = np.exp(z)", {V_SM}),
    MU("the pick's A and B are swapped on validation", "d1 = 100.0 * (y[ar, lB.argmax(-1)] - y[ar, lA.argmax(-1)])", "d1 = 100.0 * (y[ar, lA.argmax(-1)] - y[ar, lB.argmax(-1)])", {V_HAND}),
    MU("the tie-aware tau is 1e-3", "return float(1e-4 * np.log((64 - 1) / 5e-9))", "return float(1e-3)", {V_HARN, V_REAL}),
    MU("the transfer gap of the pick is the wrong way round", "\"gap_pick_argmax\": val[\"L_P1_argmax\"][\"value\"] - ins[f\"L_P1_{comp}\"]", "\"gap_pick_argmax\": ins[f\"L_P1_{comp}\"] - val[\"L_P1_argmax\"][\"value\"]", {V_HAND}),
    MU("D3 accepts a T the log did not run at", "    if built != T:\n", "    if False:\n", {"e2e:d3_refuses_a_T_the_log_did_not_run_at"}),
    MU("the CLI exit code is 0 for a NOT_READ diagnostic", "return 0 if all(x[\"status\"] == \"READ\" for x in asked) else 2", "return 0", {"e2e:rc2_for_a_killed_log_whose_D2_is_NOT_READ", "e2e:rc2_when_the_reference_log_is_missing_D2_NOT_READ"}),
    MU("the CLI ignores --through-update", "rec = run(a.arm_dir, reference_log=a.reference_log, through=a.through_update,", "rec = run(a.arm_dir, reference_log=a.reference_log, through=None,", {"e2e:through_update_is_honoured"}),
    MU("the first breached check of the outcome record is ignored", "    if outcome_arm is not None and through is None:\n", "    if False:\n", {"e2e:outcome_json_sets_through_the_first_breached_check_and_refines_O4"}),
    MU("the record is not written", "    os.replace(tmp, out)\n    d1, d2 = rec[\"D1\"], rec[\"D2\"]", "    d1, d2 = rec[\"D1\"], rec[\"D2\"]", {"e2e:writes_a_self_consistent_record"}),
]
MUTATIONS = [m for m in MUTATIONS if m[3]]


def main() -> int:
    src = TOOL_PATH.read_text(encoding="utf-8")
    n = 0
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        bad = run_suite(src, td, "base", e2e=True, heavy=True)
        assert bad == [], f"the UNMUTATED tool fails: {bad}"
        n += 1
        names = {nm for nm, _ in CHECKS + HEAVY}
        n_checks = len(CHECKS) + len(HEAVY)
        for i, (name, old, new, expect) in enumerate(MUTATIONS):
            assert src.count(old) == 1, f"mutation anchor x{src.count(old)}: {name}: {old[:80]!r}"
            unknown = {x for x in expect if x not in names and not x.startswith("e2e:")}
            assert not unknown, f"mutation '{name}' expects checks that do not exist: {sorted(unknown)}"
            got = run_suite(src.replace(old, new), td, f"m{i}", e2e=any(x.startswith("e2e:") for x in expect), heavy=any(x.startswith("real_states:") for x in expect))
            assert got, f"mutation '{name}' stayed GREEN"
            assert expect <= set(got), f"mutation '{name}' went RED but NOT where it exists to: expected {sorted(expect)}, got {got}"
            print(f"  RED  {name}  ->  {len(got)} check(s): {', '.join(got[:3])}{' ...' if len(got) > 3 else ''}", flush=True)
            n += 1
    print(f"ZZTEST_PASS {n}  (1 unmutated suite of {n_checks} checks + the CLI end to end, {len(MUTATIONS)} RED mutations)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
