#!/usr/bin/env python3
"""r0c_read.py -- the MECHANICAL reader that maps an R0c arm directory to ONE of the pre-registered outcomes O1-O5 of eval/PREREG_R0C.md section 4, or to AMBIGUOUS.

    python r0c_read.py <arm_dir> [--out R0C_READ.json] [--prereg eval/PREREG_R0C.md] [--policy policy.json]

stdlib only (runs on the pod or the dev box).  Prints one line  ZZR0C_OUTCOME <O1|O2|O3|O4|O5|AMBIGUOUS>.  Exit 0 = a decided outcome, 2 = AMBIGUOUS, 3 = refused (no prereg / no arm dir).

WHAT IT READS (the arm dir, as refe/rls1_pod.sh writes it, plus ONE file the chain does not write -- see GAP 0 in REPORT.md):
  kill_replay.json / kill_checks.jsonl   per-check g3 / g5 / K4 records of the R arm against its CONTROL (R0c: its own arm A)      [refe/rls1_checks.py judge, replay, monitor]
  KILLED.json, monitor.log               the kill that stopped the arm, and the live monitor's own lines
  sft.jsonl                              trainer events: data.epo.ref (the BUILT KL reference), train.epo_kl (every 25 updates), score_dump tags, non-finite rows (K2)
  rlval_BvsA_update<N>.json              B minus A on RL-val at check N: rl_val_eval.py --scores A=.. B=.. --base A (make it with r0c_ba_read.py).  Gives the PDMS LOWER BOUND.
  rlval_vs_A0.json                       the chain's u1000 read (A0 = base, CTRL = arm A, R = arm B, --base A0): B-A as a POINT only (difference of means); no CI for B-A exists in it
  rlval_vs_C0.json                       the SECOND control (B vs C0's arm A): reported, NEVER enters the outcome (present or absent, the outcome is identical)
  arm_meta.json  (optional)              {"monitor_control": "armA" (default) | "C0" ...}: which control the monitor records were taken against

THE CONTRACT.  The five rows of PREREG_R0C.md section 4 are parsed from the prereg file at run time (the next-lever text is COPIED, never retyped); the file's sha256 is pinned below and a change is
reported.  Every clause the table leaves open is a named READING (Q1..Q8 below).  The reader evaluates the arm under EVERY reading that has no committed answer; it names an outcome only when all
readings agree.  A state in which no row applies (a table gap) or two rows apply is AMBIGUOUS too.  It never picks.  Answers to Q1..Q8 are given ONLY by a policy file (a dated amendment that
must predate the read); the policy's sha256 is printed.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import sys

TOOL = "r0c_read/1"
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PREREG = os.path.normpath(os.path.join(HERE, "..", "..", "eval", "PREREG_R0C.md"))
PREREG_SHA256 = "cab2bb1c56f90c3eec74a4e541a15e04d7e15108b5763fa895e7d7c7bcea92cd"      # eval/PREREG_R0C.md at origin/agent/arch-inf-20260803 1b09570
GRID = (250, 500, 750, 1000)                         # the four RL-val checks (PREREG_RL_S1 section 5, K3: "two consecutive 250-update checks")
G5_BOUND = 0.5                                       # rl_val_eval.G5_SUBSCORE_FALL_MAX: NC / TTC point delta (x100) may not fall by more than this
K4_BOUND = 0.5                                       # rl_val_eval.K4_PDMS_FALL_MAX
KL_SPLIT = 0.2                                       # PREREG_R0C section 4: O3 if mean epo_kl <= 0.2, O4 if > 0.2
S4_FLOOR = 0.25                                      # PREREG_RL_S1 section 4 (1): each R - control pair >= +0.25 (x100 PDMS points)
SUBS = ("NC", "DAC", "EP", "TTC", "C", "DDC")
EPO_FINITE_KEYS = ("bceA", "lossB", "listnet", "epo", "epo_J", "epo_kl")        # rls1_checks._finite_row
KILL_CODES = ("K0", "K1", "K2", "K3", "K4", "K5", "K6", "K-MON")
TIER = ("RL-val selection tier: the true NAVSIM v1.1 PDMS of the scorer's pick on 3,000 RL-val sets of a frozen fan (open-loop 4 s plan, table lookup); NOT T1, NOT an endpoint. "
        "One training seed per arm: a separated interval is NECESSARY, not sufficient (H-ESTIM-SEED-1).")
ESTIMATOR = "paired log-cluster bootstrap, 10,000 resamples, 95 %, seed 20260927 (proxy_eval.boot) when a CI is shown; 'point' = a difference of means with NO interval"

# ---------------------------------------------------------------------------------------------------------------- the open clauses
READINGS = {
    "Q1": ("g4 on RL-val", ("vacuous", "blocking"),
           "O1 says g3-g5 hold, but rl_val_eval lists g4 (longitudinal bias rows) under not_computed: nothing on RL-val can show g4 holds. vacuous = treat g4 as holding; blocking = O1 cannot be certified."),
    "Q2": ("which epo_kl is 'the mean at the breach'", ("second_breach_window", "first_breach_window", "both_breach_windows", "row_at_second_breach", "row_at_first_breach"),
           "epo_kl is logged every 25 updates as a window mean. second/first_breach_window = mean of the rows in (previous check, that check]; both = (check before the first breach, second breach]; row_at_* = the single row at that update."),
    "Q3": ("an NC fall inside a g5 breach", ("g5", "O5"),
           "g5 = NC and TTC; O5 lists 'an NC / DAC bound', which section 5 never defines. g5 = an NC-involved breach is still O3/O4; O5 = it is the O5 NC bound."),
    "Q4": ("O2's 'PDMS B - A lower bound <= 0' (which check)", ("both_le0", "final_le0", "either_le0"),
           "O1 needs LB > 0 at u500 AND u1000; O2 names no check. both = LB <= 0 at u500 and u1000; final = at u1000; either = at u500 or u1000."),
    "Q5": ("does O2 ('safe') need g3", ("as_written", "g3_also"),
           "O2's condition names g5 only. as_written = g3 is not required; g3_also = O2 also needs g3 to hold at every check."),
    "Q6": ("the +0.25 floor of section 4 (1) on RL-val", ("applies", "not_applicable"),
           "O1 cites 'the section 4 bar', defined on full navtest with two pairs. applies = B - A PDMS point >= +0.25 at u1000; not_applicable = RL-val needs only LB > 0."),
    "Q7": ("scope of 'no longitudinal / lateral component separates adversely'", ("narrow", "longlat", "all"),
           "adverse = the B - A interval lies entirely below 0 at u1000. narrow = EP, DAC; longlat = EP, TTC, DAC, DDC; all = all six sub-scores."),
    "Q8": ("a K-MON stop (the monitor died)", ("O5", "not_O5"),
           "O5 lists K0-K2 and K4-K6; K-MON is the chain's fail-closed stop of a dead monitor and is not listed. O5 = count it as 'other kill'; not_O5 = it is no outcome."),
}
FAMILY_NOTE = {
    "TACTICAL": "N/A on this scorer-only arm: R0c trains the scorer heads on a FROZEN fan and picks among its proposals; no tactical-decision output exists to score (decision kappa and lane-change precision/recall are full-navtest endpoints, R2 only: PREREG_RL_S1 section 4).",
    "STRATEGIC": "PAUSED (PI ruling 2026-09-19, PI_DECISION_QUEUE ITEM 25): reported as paused, not as a pass, not omitted.",
}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def jlines(p):
    rows = []
    with open(p, encoding="utf-8", errors="replace") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                pass
    return rows


def tag_u(tag):
    m = re.fullmatch(r"update(\d+)", str(tag))
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------------------------------------------- the prereg table
def parse_prereg(path):
    """-> ({O1..O5: {name, condition, meaning, next_lever}}, sha256).  Raises ValueError when the five rows are not all found."""
    rows = {}
    with open(path, encoding="utf-8") as f:
        for ln in f:
            m = re.match(r"^\|\s*\*\*(O[1-5])\b([^*]*)\*\*\s*\|(.*)$", ln.rstrip("\n"))
            if not m:
                continue
            cells = [c.strip() for c in m.group(3).rsplit("|", 1)[0].split("|")]
            if len(cells) != 3:
                raise ValueError(f"{m.group(1)}: expected condition | meaning | next lever, got {len(cells)} cells")
            rows[m.group(1)] = {"name": (m.group(1) + m.group(2)).strip(), "condition": cells[0], "meaning": cells[1], "next_lever": cells[2]}
    if sorted(rows) != ["O1", "O2", "O3", "O4", "O5"]:
        raise ValueError(f"PREREG table rows found: {sorted(rows)}")
    return rows, sha256_file(path)


# ---------------------------------------------------------------------------------------------------------------- input loading
def load_checks(d, ev):
    """Per-check monitor records, merged from kill_replay.json and kill_checks.jsonl.  A disagreement between the two on any bound is a CONFLICT."""
    src = {}
    rp = os.path.join(d, "kill_replay.json")
    kc = os.path.join(d, "kill_checks.jsonl")
    ev["recorded_kill_sources"] = {}
    if os.path.isfile(rp):
        j = jload(rp)
        src["kill_replay.json"] = {tag_u(c["tag"]): c for c in j.get("checks", []) if tag_u(c.get("tag")) is not None}
        ev["recorded_kill_sources"]["kill_replay.json"] = j.get("kill")
        ev["inputs"]["kill_replay.json"] = sha256_file(rp)
    if os.path.isfile(kc):
        rows = [r for r in jlines(kc) if tag_u(r.get("tag")) is not None]
        src["kill_checks.jsonl"] = {tag_u(r["tag"]): r for r in rows}
        ks = [r.get("kill") for r in rows if r.get("kill")]
        ev["recorded_kill_sources"]["kill_checks.jsonl"] = ks[-1] if ks else None
        ev["inputs"]["kill_checks.jsonl"] = sha256_file(kc)
    merged = {}
    for name, recs in src.items():
        for u, r in recs.items():
            if u in merged:
                a, b = merged[u][1], r
                for fld in ("g3_ok", "g5_ok", "k4_breach"):
                    if bool(a.get(fld)) != bool(b.get(fld)):
                        ev["conflicts"].append(f"check update{u}: {merged[u][0]} says {fld}={a.get(fld)}, {name} says {b.get(fld)}")
                for fld in ("NC_fall_ok", "TTC_fall_ok"):
                    if bool((a.get("g5") or {}).get(fld)) != bool((b.get("g5") or {}).get(fld)):
                        ev["conflicts"].append(f"check update{u}: {merged[u][0]} and {name} disagree on g5.{fld}")
            else:
                merged[u] = (name, r)
    out = {}
    for u in sorted(merged):
        r = merged[u][1]
        g5 = r.get("g5") or {}
        out[u] = {"g3_ok": bool(r.get("g3_ok")), "g5_ok": bool(r.get("g5_ok")), "nc_ok": bool(g5.get("NC_fall_ok", True)), "ttc_ok": bool(g5.get("TTC_fall_ok", True)),
                  "k4": bool(r.get("k4_breach")), "pdms_delta_x100": r.get("pdms_delta_x100"), "k3_enabled": bool(r.get("k3_enabled", True)),
                  "g3": r.get("g3"), "source": merged[u][0], "lane_change": r.get("lane_change"), "n_sets": r.get("n_sets")}
        if r.get("g5_ok") is not None and bool(r["g5_ok"]) != (out[u]["nc_ok"] and out[u]["ttc_ok"]):
            ev["conflicts"].append(f"check update{u}: g5_ok={r['g5_ok']} but NC_fall_ok={out[u]['nc_ok']}, TTC_fall_ok={out[u]['ttc_ok']}")
    return out


def load_kills(d, ev):
    p = os.path.join(d, "KILLED.json")
    if os.path.isfile(p):
        j = jload(p)
        ev["recorded_kill_sources"]["KILLED.json"] = j.get("code")
        ev["inputs"]["KILLED.json"] = sha256_file(p)
    p = os.path.join(d, "monitor.log")
    mon = {"present": os.path.isfile(p), "bytes": 0, "ready": False, "k3_state": None, "check_errors": 0, "kill": None}
    if mon["present"]:
        txt = open(p, encoding="utf-8", errors="replace").read()
        mon["bytes"] = len(txt.encode("utf-8"))
        mon["ready"] = "ZZRLS1_MONITOR_READY" in txt
        mon["check_errors"] = txt.count("ZZRLS1_CHECK_ERROR")
        mon["k3_state"] = "LIVE" if "ZZRLS1_K3_LIVE" in txt else ("DISABLED" if "ZZRLS1_K3_DISABLED" in txt else None)
        m = re.search(r"^ZZRLS1_KILL \S+ (K-MON|K\d)\b", txt, re.M)
        mon["kill"] = m.group(1) if m else None
        if mon["kill"]:
            ev["recorded_kill_sources"]["monitor.log"] = mon["kill"]
        ev["inputs"]["monitor.log"] = sha256_file(p)
    ev["monitor"] = mon
    codes = {c for c in ev["recorded_kill_sources"].values() if c}
    bad = codes - set(KILL_CODES)
    if bad:
        ev["conflicts"].append(f"unknown kill code(s) {sorted(bad)}")
    if len(codes) > 1:
        ev["conflicts"].append(f"the recorded kills disagree: {ev['recorded_kill_sources']}")
    ev["recorded_kill"] = sorted(codes)[0] if len(codes) == 1 else None


def load_sft(d, ev):
    p = os.path.join(d, "sft.jsonl")
    ev["sft"] = {"present": os.path.isfile(p)}
    if not os.path.isfile(p):
        return
    rows = jlines(p)
    ev["inputs"]["sft.jsonl"] = sha256_file(p)
    data = [r for r in rows if r.get("event") == "data"]
    epo = (data[0].get("epo") or {}) if data else {}
    ev["sft"].update({"epo_ref": epo.get("ref"), "epo_lam": epo.get("lam"), "epo_beta": epo.get("beta"), "has_data_event": bool(data),
                      "score_dump_tags": [r.get("tag") for r in rows if r.get("event") == "score_dump"], "done": any(r.get("event") == "done" for r in rows)})
    tr = [r for r in rows if r.get("event") == "train"]
    ev["sft"]["train_rows"] = len(tr)
    kl = {}
    for r in tr:
        if "epo_kl" in r and isinstance(r.get("update"), int):
            kl[r["update"]] = r["epo_kl"]
    ev["epo_kl_rows"] = kl
    for r in tr:
        bad = [k for k in EPO_FINITE_KEYS if isinstance(r.get(k), (int, float)) and not math.isfinite(float(r[k]))]
        if bad:
            ev["k2_update"] = r.get("update")
            break


def load_reads(d, ev):
    ev["ba"], ev["ba_errors"] = {}, {}
    for u in GRID:
        p = os.path.join(d, f"rlval_BvsA_update{u}.json")
        if not os.path.isfile(p):
            continue
        ev["inputs"][os.path.basename(p)] = sha256_file(p)
        try:
            j = jload(p)
            if j.get("tool") != "rl_val_eval/1":
                raise ValueError(f"tool is {j.get('tool')!r}, not rl_val_eval/1")
            if j.get("base") != "A" or sorted((j.get("paired_vs_base") or {})) != ["B"]:
                raise ValueError(f"must be rl_val_eval --scores A=.. B=.. --base A (base={j.get('base')!r}, arms={sorted(j.get('paired_vs_base') or {})}): a read against any other control is not B - A")
            q = j["paired_vs_base"]["B"]["quantities"]
            ev["ba"][u] = {"quantities": {k: {"mean": v["mean"], "ci95": list(v["ci95"]), "adverse_recomputed": bool(v["ci95"][1] < 0)} for k, v in q.items()},
                           "n_sets": j["paired_vs_base"]["B"].get("n_sets")}
        except (KeyError, ValueError, TypeError, IndexError) as e:
            ev["ba_errors"][u] = f"{type(e).__name__}: {e}"
    # the chain's u1000 read: B - A as a POINT (exact for means), NO interval
    p = os.path.join(d, "rlval_vs_A0.json")
    ev["a0"] = None
    if os.path.isfile(p):
        ev["inputs"]["rlval_vs_A0.json"] = sha256_file(p)
        try:
            j = jload(p)
            pv = j["paired_vs_base"]
            ev["a0"] = {"B_minus_A_point_x100": {k: pv["R"]["quantities"][k]["mean"] - pv["CTRL"]["quantities"][k]["mean"] for k in pv["R"]["quantities"] if k in pv["CTRL"]["quantities"]},
                        "A_minus_base_x100": {k: pv["CTRL"]["quantities"][k]["mean"] for k in pv["CTRL"]["quantities"]},
                        "B_minus_base_x100": {k: pv["R"]["quantities"][k]["mean"] for k in pv["R"]["quantities"]}}
        except (KeyError, TypeError, ValueError) as e:
            ev["a0"] = {"error": f"{type(e).__name__}: {e}"}
    p = os.path.join(d, "rlval_vs_C0.json")
    ev["second_control"] = {"present": os.path.isfile(p)}
    if os.path.isfile(p):
        ev["inputs"]["rlval_vs_C0.json"] = sha256_file(p)
        try:
            j = jload(p)
            name = sorted(j["paired_vs_base"])[0]
            q = j["paired_vs_base"][name]["quantities"]
            ev["second_control"].update({"base": j.get("base"), "arm": name, "B_minus_C0_arm_A": {k: {"mean": q[k]["mean"], "ci95": q[k]["ci95"]} for k in ("PDMS", "TTC", "NC", "EP", "DAC") if k in q},
                                         "g5": j["paired_vs_base"][name].get("g5"),
                                         "note": "the SECOND control (C0's banked arm A). It is reported and NEVER enters the outcome."})
        except (KeyError, TypeError, ValueError, IndexError) as e:
            ev["second_control"]["error"] = f"{type(e).__name__}: {e}"
    # cross-check: the chain's u1000 point B - A must equal the u1000 B-vs-A read's mean (same sets)
    if ev["a0"] and "B_minus_A_point_x100" in ev["a0"] and 1000 in ev["ba"]:
        for k, v in ev["a0"]["B_minus_A_point_x100"].items():
            q = ev["ba"][1000]["quantities"].get(k)
            if q and abs(q["mean"] - v) > 1e-6:
                ev["conflicts"].append(f"u1000 {k}: rlval_vs_A0.json gives B - A = {v:.6f} (difference of means), rlval_BvsA_update1000.json gives {q['mean']:.6f}: different score files or sets")


def load_meta(d, ev):
    p = os.path.join(d, "arm_meta.json")
    ev["monitor_control"] = "armA"
    if os.path.isfile(p):
        ev["monitor_control"] = str(jload(p).get("monitor_control", "armA"))
        ev["inputs"]["arm_meta.json"] = sha256_file(p)


# ---------------------------------------------------------------------------------------------------------------- derived facts
def judge(checks):
    """rls1_checks.judge replayed over the records in order: K3 (g3 or g5 breached at two consecutive records), K4 (PDMS below the control by > 0.5 at two consecutive).  K5 is R2's."""
    hist, k3_enabled = [], None
    for u in sorted(checks):
        rec = checks[u]
        k3_enabled = rec["k3_enabled"] if k3_enabled is None else k3_enabled
        prev = hist[-1] if hist else None
        breach = lambda r: not (r["g3_ok"] and r["g5_ok"])                      # noqa: E731
        if k3_enabled and prev is not None and breach(rec) and breach(prev):
            return {"code": "K3", "pair": (prev["u"], u)}
        if prev is not None and rec["k4"] and prev["k4"]:
            return {"code": "K4", "pair": (prev["u"], u)}
        hist.append({**rec, "u": u})
    return None


def adjacent(pair):
    """True when the two checks are neighbours on the 250-update grid (the 'two consecutive checks' of K3 and of O3/O4)."""
    a, b = pair
    return a in GRID and GRID.index(a) + 1 < len(GRID) and GRID[GRID.index(a) + 1] == b


def first_g5_pair(checks):
    for a, b in zip(GRID, GRID[1:]):
        if a in checks and b in checks and not checks[a]["g5_ok"] and not checks[b]["g5_ok"]:
            return (a, b)
    return None


def kl_windows(ev, pair):
    """The five candidate values of 'mean epo_kl at the breach' for the first g5 pair (a, b); None when the logged rows do not cover the window."""
    kl = ev.get("epo_kl_rows") or {}
    prev = {GRID[i]: (GRID[i - 1] if i else 0) for i in range(len(GRID))}

    def win(lo, hi):
        v = [kl[u] for u in sorted(kl) if lo < u <= hi]
        return sum(v) / len(v) if v else None
    a, b = pair
    return {"second_breach_window": win(prev[b], b), "first_breach_window": win(prev[a], a), "both_breach_windows": win(prev[a], b),
            "row_at_second_breach": kl.get(b), "row_at_first_breach": kl.get(a)}


def prepare(d):
    ev = {"inputs": {}, "conflicts": [], "notes": [], "input_errors": []}
    if not os.path.isdir(d):
        raise FileNotFoundError(d)
    load_meta(d, ev)
    ev["checks"] = load_checks(d, ev)
    load_kills(d, ev)
    load_sft(d, ev)
    load_reads(d, ev)
    ck = ev["checks"]
    mon = ev["monitor"]
    ev["tags_present"] = sorted(ck)
    ev["judge"] = judge(ck)
    ev["g5_pair"] = first_g5_pair(ck)
    ev["k2_derived"] = ev.get("k2_update") is not None
    rk = ev["recorded_kill"]
    if rk == "K2" or ev["k2_derived"]:
        ev["notes"].append("K2 (non-finite loss) " + ("recorded" if rk == "K2" else "derived from sft.jsonl") + (f" at update {ev.get('k2_update')}" if ev.get("k2_update") else ""))
    j = ev["judge"]
    for code in ("K3", "K4"):
        if rk == code and not (j and j["code"] == code):
            ev["conflicts"].append(f"the recorded kill is {code} but the {len(ck)} check record(s) do not derive it (rls1_checks.judge)")
    if j and not adjacent(j["pair"]):
        ev["conflicts"].append(f"{j['code']} fired on checks {j['pair']}, which are not adjacent on the 250-update grid (a check is missing between them)")
    if j and rk in ("K3", "K4") and j["code"] != rk:
        ev["conflicts"].append(f"recorded kill {rk} but the records derive {j['code']}")
    if mon["present"] and mon["bytes"] == 0:
        ev["notes"].append("monitor.log is 0 bytes: the live monitor wrote nothing (the R0 failure mode); the checks come from the post-run replay")
    if mon["check_errors"]:
        ev["notes"].append(f"monitor.log carries {mon['check_errors']} ZZRLS1_CHECK_ERROR line(s): a live check was skipped")
    if ck and not all(r["k3_enabled"] for r in ck.values()):
        ev["notes"].append("k3_enabled=False in the records: the base already breached g3 on RL-val, so the live K3 never kills; O3/O4 are still read from the data (g5 at two consecutive checks)")
    if ev["monitor_control"] != "armA":
        ev["notes"].append(f"the monitor records were taken against {ev['monitor_control']!r}, NOT against the arm's own A: they are not R0c's control (PREREG_R0C section 2 guards)")
    # g5 re-derived from the B-vs-A reads (point means, x100): an independent check of the monitor's booleans when both are against arm A
    ev["g5_from_ba"] = {}
    for u, r in ev["ba"].items():
        q = r["quantities"]
        if "NC" in q and "TTC" in q:
            ev["g5_from_ba"][u] = {"NC_fall_ok": q["NC"]["mean"] >= -G5_BOUND, "TTC_fall_ok": q["TTC"]["mean"] >= -G5_BOUND, "NC": q["NC"]["mean"], "TTC": q["TTC"]["mean"]}
            if u in ck and ev["monitor_control"] == "armA":
                for fld, key in (("nc_ok", "NC_fall_ok"), ("ttc_ok", "TTC_fall_ok")):
                    if ck[u][fld] != ev["g5_from_ba"][u][key]:
                        ev["conflicts"].append(f"update{u}: the monitor record says {key}={ck[u][fld]} but the B-vs-A read (NC {q['NC']['mean']:+.4f}, TTC {q['TTC']['mean']:+.4f}) says {ev['g5_from_ba'][u][key]}")
    ev["kl_ref"] = ev["sft"].get("epo_ref")
    ev["kl_windows"] = kl_windows(ev, ev["g5_pair"]) if ev["g5_pair"] else None
    return ev


# ---------------------------------------------------------------------------------------------------------------- classification under ONE reading
def classify(ev, R):
    """-> (label, why).  label in O1..O5, NONE (the table is silent), OVERLAP (two rows apply), MISSING (a needed input is absent)."""
    ck, rk, j, pair = ev["checks"], ev["recorded_kill"], ev["judge"], ev["g5_pair"]
    other_rec = rk in ("K0", "K1", "K2", "K5", "K6") or ev["k2_derived"]
    if other_rec and (j or pair):
        return "OVERLAP", f"a {rk or 'K2'} stop AND a derived {(j or {}).get('code', 'g5 pair')} are both on record, and no timestamp orders them"
    if other_rec:
        return "O5", f"kill {rk or 'K2'} (O5: K0-K2, K4-K6)"
    if rk == "K-MON":
        if j or pair:
            return "OVERLAP", "K-MON and a derived kill are both on record"
        return ("O5", "K-MON counted as 'other kill' (Q8=O5)") if R["Q8"] == "O5" else ("NONE", "K-MON (monitor died): not among O5's K0-K2, K4-K6 (Q8=not_O5)")
    if j and j["code"] == "K4":
        if pair and pair[1] <= j["pair"][1]:
            return "OVERLAP", f"K4 at {j['pair']} and a g5 pair {pair} both complete"
        return "O5", f"K4 at checks {j['pair']} (O5)"
    if (j and j["code"] == "K3") or (j is None and pair):
        p = j["pair"] if j else pair
        both_g5 = all(p[i] in ck and not ck[p[i]]["g5_ok"] for i in (0, 1))
        if not both_g5:
            return "NONE", (f"K3 fired on checks {p} but g5 was not breached at both (g3 / mixed): O3/O4 say g5 at two consecutive checks and O5 lists K0-K2, K4-K6 -- K3 is in neither")
        nc = any(not ck[x]["nc_ok"] for x in p)
        if nc and R["Q3"] == "O5":
            return "O5", f"g5 breach at {p} involves NC and Q3=O5 reads it as the O5 NC bound"
        if ev["kl_ref"] != "armA":
            return "MISSING", f"g5 breached at {p} but the logged epo_kl is not A-to-B (data.epo.ref = {ev['kl_ref']!r}); O3/O4 need epo_kl (A to B)"
        v = (ev["kl_windows"] or {}).get(R["Q2"])
        if v is None:
            return "MISSING", f"g5 breached at {p}; epo_kl under reading Q2={R['Q2']} is not covered by the logged rows"
        return ("O3" if v <= KL_SPLIT else "O4"), f"g5 breached at {p}; mean epo_kl (A to B) = {v:.4f} under Q2={R['Q2']} ({'<=' if v <= KL_SPLIT else '>'} {KL_SPLIT})"
    # ---- an arm that ran to the end with no kill
    miss = [u for u in GRID if u not in ck]
    if miss:
        return "MISSING", f"no monitor record for check(s) {miss} and no kill on record"
    g3_all = all(ck[u]["g3_ok"] for u in GRID)
    g5_all = all(ck[u]["g5_ok"] for u in GRID)
    if not g5_all:
        bad = [u for u in GRID if not ck[u]["g5_ok"]]
        return "NONE", f"g5 breached at {bad}, never at two consecutive checks: O1/O2 need g5 to hold everywhere, O3/O4 need two consecutive"
    ba = ev["ba"]
    need = [u for u in (500, 1000) if u not in ba]
    if need:
        return "MISSING", f"no B-vs-A read at {need} (rlval_BvsA_update<N>.json): the PDMS lower bound B - A cannot be read from the chain's own outputs (GAP 0)" + (f"; {ev['ba_errors']}" if ev["ba_errors"] else "")
    q = {u: ba[u]["quantities"] for u in (500, 1000)}
    if "PDMS" not in q[500] or "PDMS" not in q[1000]:
        return "MISSING", "a B-vs-A read carries no PDMS quantity"
    lb5, lb10, mean10 = q[500]["PDMS"]["ci95"][0], q[1000]["PDMS"]["ci95"][0], q[1000]["PDMS"]["mean"]
    why = []
    if not g3_all:
        why.append(f"g3 breached at {[u for u in GRID if not ck[u]['g3_ok']]}")
    if R["Q1"] == "blocking":
        why.append("g4 cannot be evaluated on RL-val (Q1=blocking)")
    if not lb5 > 0:
        why.append(f"LB(u500)={lb5:+.3f} <= 0")
    if not lb10 > 0:
        why.append(f"LB(u1000)={lb10:+.3f} <= 0")
    if R["Q6"] == "applies" and not mean10 >= S4_FLOOR:
        why.append(f"B - A PDMS point {mean10:+.3f} < +{S4_FLOOR} (Q6=applies)")
    o1 = not why
    if o1:                                                                  # only now can the adverse-separation clause decide anything
        scope = {"narrow": ("EP", "DAC"), "longlat": ("EP", "TTC", "DAC", "DDC"), "all": SUBS}[R["Q7"]]
        absent = [s_ for s_ in scope if s_ not in q[1000]]
        if absent:
            return "MISSING", f"the u1000 B-vs-A read lacks {absent}, needed for the adverse-separation clause (Q7={R['Q7']})"
        adv = [s_ for s_ in scope if q[1000][s_]["ci95"][1] < 0]
        if adv:
            o1 = False
            why.append(f"adverse separation at u1000: {adv} (Q7={R['Q7']})")
    le = {"both_le0": lb5 <= 0 and lb10 <= 0, "final_le0": lb10 <= 0, "either_le0": lb5 <= 0 or lb10 <= 0}[R["Q4"]]
    o2 = le and (R["Q5"] == "as_written" or g3_all)
    if o1 and o2:
        return "OVERLAP", "O1 and O2 both apply"
    if o1:
        return "O1", f"g3-g5 hold, LB(u500)={lb5:+.3f}, LB(u1000)={lb10:+.3f} > 0" + (f", point(u1000)={mean10:+.3f}" if R["Q6"] == "applies" else "")
    if o2:
        return "O2", f"g5 holds at every check; PDMS LB(u500)={lb5:+.3f}, LB(u1000)={lb10:+.3f} (Q4={R['Q4']})"
    return "NONE", "g5 holds but neither O1 nor O2 applies: " + "; ".join(why or ["O2's lower-bound clause fails (Q4=%s)" % R["Q4"]])


# ---------------------------------------------------------------------------------------------------------------- aggregation over the unanswered readings
def load_policy(path):
    if not path:
        return {}, None, None
    j = jload(path)
    ans = j.get("answers", {})
    for k, v in ans.items():
        if k not in READINGS or v not in READINGS[k][1]:
            raise ValueError(f"policy answer {k}={v!r} is not one of {READINGS.get(k, ('', ()))[1]}")
    if ans and not str(j.get("amendment", "")).strip():
        raise ValueError("a policy with answers must name the dated amendment that committed them (key 'amendment')")
    return ans, sha256_file(path), j.get("amendment")


def decide(ev, answers):
    keys = sorted(READINGS)
    opts = [(answers[k],) if k in answers else READINGS[k][1] for k in keys]
    table = {}
    for combo in itertools.product(*opts):
        R = dict(zip(keys, combo))
        table[combo] = classify(ev, R)
    labels = {lab for lab, _ in table.values()}
    affecting = set()
    for i, k in enumerate(keys):
        if len(opts[i]) < 2:
            continue
        for combo, (lab, _) in table.items():
            if any(table[combo[:i] + (alt,) + combo[i + 1:]][0] != lab for alt in opts[i] if alt != combo[i]):
                affecting.add(k)
                break
    reasons = {}
    for (lab, why) in table.values():
        reasons.setdefault(lab, [])
        if why not in reasons[lab]:
            reasons[lab].append(why)
    return labels, sorted(affecting), reasons


def family_block(ev):
    """The four metric families RL-val can show, from the latest B-vs-A read with an interval, else the chain's u1000 point."""
    u = max(ev["ba"]) if ev["ba"] else None
    out = {"tier": TIER, "estimator": ESTIMATOR}

    def val(k):
        if u is not None and k in ev["ba"][u]["quantities"]:
            q = ev["ba"][u]["quantities"][k]
            return {"B_minus_A_x100": q["mean"], "ci95": q["ci95"], "check": f"update{u}", "kind": "interval", "adverse": q["ci95"][1] < 0}
        if ev["a0"] and k in (ev["a0"].get("B_minus_A_point_x100") or {}):
            return {"B_minus_A_x100": ev["a0"]["B_minus_A_point_x100"][k], "ci95": None, "check": "update1000", "kind": "point (difference of means; no interval in rlval_vs_A0.json)"}
        return "NOT AVAILABLE in the arm dir"
    g3 = {x: ev["checks"][x].get("g3") for x in sorted(ev["checks"]) if ev["checks"][x].get("g3")}
    out["LONGITUDINAL"] = {"EP_progress": val("EP"), "progress_telemetry_g3": {f"update{x}": {"pick_over_fan_mean_path": g["pick_over_fan_mean_path"], "shortest_bin_share": g["shortest_bin_share"]} for x, g in g3.items()},
                           "g4_longitudinal_bias": "NOT COMPUTED on RL-val (rl_val_eval not_computed)"}
    out["LATERAL"] = {"DAC": val("DAC"), "DDC": val("DDC")}
    out["SUBSCORES"] = {"TTC": val("TTC"), "NC": val("NC"), "comfort_C": val("C"), "PDMS": val("PDMS")}
    out["TACTICAL"] = FAMILY_NOTE["TACTICAL"]
    out["STRATEGIC"] = FAMILY_NOTE["STRATEGIC"]
    return out


def read_arm(arm_dir, prereg=DEFAULT_PREREG, policy=None):
    rows, psha = parse_prereg(prereg)
    answers, pol_sha, amend = load_policy(policy)
    ev = prepare(arm_dir)
    res = {"tool": TOOL, "prereg": {"path": os.path.abspath(prereg), "sha256": psha, "matches_pinned": psha == PREREG_SHA256},
           "arm_dir": os.path.abspath(arm_dir), "tier": TIER, "estimator": ESTIMATOR}
    labels, affecting, reasons = decide(ev, answers)
    if len(labels) == 1 and "O1" in labels and not ev["conflicts"]:
        ev["notes"].append("O1 is a CANDIDATE: section 4 (1) 'mean of two pairs >= +0.5' and (3) 'contrast > 2 x the seed / inference floor' need the replicate arm and the full-navtest endpoint; "
                           "this reader cannot evaluate them (H-ESTIM-SEED-1: a separated interval from one seed is necessary, not sufficient)")
    if not res["prereg"]["matches_pinned"]:
        ev["notes"].append("PREREG_R0C.md differs from the version this reader was built against: re-read the reader against the amendment before trusting a label")
    if ev["conflicts"]:
        outcome, kind = "AMBIGUOUS", "conflict"
        why = "the arm dir contradicts itself: " + " | ".join(ev["conflicts"])
        cand = []
    elif len(labels) == 1 and next(iter(labels)) in rows:
        outcome, kind, cand = next(iter(labels)), "decided", []
        why = " | ".join(reasons[outcome][:3]) + (f"  [unanswered readings that did not matter: {sorted(set(READINGS) - set(answers) - set(affecting))}]" if len(set(READINGS) - set(answers)) else "")
    else:
        outcome, cand = "AMBIGUOUS", sorted(labels)
        if labels <= {"MISSING"}:
            kind = "missing_input"
        elif labels <= {"NONE"}:
            kind = "table_gap"
        elif labels & {"OVERLAP"} and len(labels) == 1:
            kind = "overlap"
        else:
            kind = "reading_split"
        why = (f"labels across the readings: {cand}. " + " || ".join(f"{lab}: {w[0]}" for lab, w in sorted(reasons.items()))
               + (f" || the split is caused by unanswered {affecting}" if affecting else ""))
    res.update({"outcome": outcome, "ambiguity_kind": kind if outcome == "AMBIGUOUS" else None, "reason": why, "candidate_labels": cand,
                "meaning": rows[outcome]["meaning"] if outcome in rows else None, "next_lever": rows[outcome]["next_lever"] if outcome in rows else None,
                "prereg_row": rows[outcome] if outcome in rows else None,
                "readings": {"policy_sha256": pol_sha, "policy_amendment": amend, "answered": answers, "unanswered": sorted(set(READINGS) - set(answers)), "split_by": affecting,
                             "questions": {k: {"what": READINGS[k][0], "options": list(READINGS[k][1]), "text": READINGS[k][2]} for k in sorted(READINGS)}},
                "evidence": {"checks": {f"update{u}": {k: v for k, v in r.items() if k not in ("g3", "lane_change")} for u, r in ev["checks"].items()},
                             "terminal_kill_derived": ({"code": ev["judge"]["code"], "checks": list(ev["judge"]["pair"])} if ev["judge"] else None),
                             "first_g5_pair": list(ev["g5_pair"]) if ev["g5_pair"] else None, "recorded_kill": ev["recorded_kill"],
                             "recorded_kill_reproduced": (None if ev["recorded_kill"] not in ("K3", "K4") else bool(ev["judge"] and ev["judge"]["code"] == ev["recorded_kill"])), "recorded_kill_sources": ev["recorded_kill_sources"],
                             "monitor_log": ev["monitor"], "monitor_control": ev["monitor_control"], "sft": ev["sft"],
                             "epo_kl": {"logged_reference": ev["kl_ref"], "is_A_to_B": ev["kl_ref"] == "armA", "candidate_windows_for_first_g5_pair": ev["kl_windows"],
                                        "split": KL_SPLIT, "note": "values are labelled A-to-B only when sft.jsonl data.epo.ref == 'armA'; otherwise they are base-to-B (R0 / R0b) and decide nothing"},
                             "b_minus_a_reads": {f"update{u}": {k: ev["ba"][u]["quantities"][k] for k in ev["ba"][u]["quantities"]} for u in sorted(ev["ba"])},
                             "b_minus_a_reads_errors": ev["ba_errors"], "g5_from_b_minus_a_reads": {f"update{u}": v for u, v in sorted(ev["g5_from_ba"].items())},
                             "chain_u1000_read_rlval_vs_A0": ev["a0"], "second_control": ev["second_control"]},
                "families": family_block(ev), "notes": ev["notes"], "conflicts": ev["conflicts"], "inputs_sha256": ev["inputs"]})
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("arm_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--prereg", default=DEFAULT_PREREG)
    ap.add_argument("--policy", default=None, help="a JSON file {'amendment': '<dated amendment>', 'answers': {'Q1': 'vacuous', ...}} committed BEFORE the read")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")                                  # a cp1252 console must not turn a prereg character into a crash
    try:
        res = read_arm(a.arm_dir, a.prereg, a.policy)
    except (FileNotFoundError, ValueError) as e:
        print(f"ZZR0C_REFUSED {type(e).__name__}: {e}", flush=True)
        return 3
    out = a.out or os.path.join(a.arm_dir, "R0C_READ.json")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, sort_keys=False)
    print(f"arm {res['arm_dir']}\n  {res['reason'][:600]}")
    if res["next_lever"]:
        print(f"  next lever (copied from the prereg row): {res['next_lever'][:300]}")
    if res["notes"]:
        print("  notes: " + " | ".join(res["notes"])[:600])
    print(f"ZZR0C_OUTCOME {res['outcome']}", flush=True)
    return 0 if res["outcome"] != "AMBIGUOUS" else 2


if __name__ == "__main__":
    sys.exit(main())
