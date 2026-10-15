#!/usr/bin/env python3
"""Tests for r0c_read.py (the mechanical reader of PREREG_R0C.md section 4).   Run:  python test_r0c_read.py   ->   ZZTEST_PASS <n>   (stdlib only; no pod, no GPU, no data).

  S  SCENARIOS  synthetic arm dirs in the chain's own formats, one per outcome O1-O5 and several AMBIGUOUS kinds, each read to its expected label.
  P  POLICY     a committed answer narrows a reading split; an answer without its amendment, or outside the options, is refused.
  C  CONTROL    rlval_vs_C0.json (the SECOND control) present, absent, or contradicting: the outcome is identical.
  F  FAMILIES   the four metric families (STRATEGIC paused, TACTICAL n/a) and the contract (next lever copied from the prereg, sha256 pinned).
  R  RED        every mutation edits the reader's SOURCE (the snippet must occur exactly once) and must turn a named check RED while the unmutated suite is green.
"""
from __future__ import annotations

import io
import contextlib
import json
import os
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "r0c_read.py")
PREREG = os.path.normpath(os.path.join(HERE, "..", "..", "eval", "PREREG_R0C.md"))


def load_reader(mutation=None):
    s = open(SRC, encoding="utf-8").read()
    if mutation:
        old, new = mutation
        assert s.count(old) == 1, f"mutation snippet must occur exactly once, found {s.count(old)}: {old[:80]!r}"
        s = s.replace(old, new)
    m = types.ModuleType("r0c_read_under_test")
    m.__file__ = SRC
    exec(compile(s, SRC, "exec"), m.__dict__)
    return m


# ------------------------------------------------------------------------------------------------------------------ synthetic arm dirs (the chain's own formats)
def rec(u, g3=True, nc=True, ttc=True, k4=False, k3en=True, delta=0.9):
    return {"tag": f"update{u}", "g3_ok": g3, "g3": {"pick_over_fan_mean_path": 0.98, "min_allowed": 0.894, "pick_ok": g3, "shortest_bin_share": 0.31, "max_allowed": 0.55, "share_ok": g3},
            "g5_ok": nc and ttc, "g5": {"NC_fall_ok": nc, "TTC_fall_ok": ttc, "bound": 0.5}, "k4_breach": k4, "pdms_delta_x100": delta, "k5_breach": False, "lane_change": None,
            "n_sets": 3000, "k3_enabled": k3en}


BA_OK = {"PDMS": (1.0, 0.3, 1.7), "NC": (0.0, -0.2, 0.2), "DAC": (0.1, -0.1, 0.3), "EP": (2.0, 1.0, 3.0), "TTC": (0.0, -0.3, 0.3), "C": (0.0, 0.0, 0.0), "DDC": (0.0, -0.2, 0.2), "R": (0.5, 0.1, 0.9)}
A_MINUS_A0 = {"PDMS": 0.2, "NC": 0.0, "DAC": 0.0, "EP": 0.5, "TTC": 0.4, "C": 0.0, "DDC": 0.0, "R": 0.1}


def qjson(q):
    return {k: {"mean": m, "ci95": [lo, hi], "adverse": hi < 0, "better": 1, "worse": 1, "tied": 1, "n": 3000} for k, (m, lo, hi) in q.items()}


def ba_file(q, base="A", arm="B", tool="rl_val_eval/1"):
    return {"tool": tool, "base": base, "eligibility": {"sets_used": 3000}, "paired_vs_base": {arm: {"n_sets": 3000, "quantities": qjson(q), "g5": {}, "K4": {}}}}


def a0_file(ba):
    ctrl = {k: (A_MINUS_A0[k], -1, 1) for k in ba}
    r = {k: (A_MINUS_A0[k] + ba[k][0], -1, 1) for k in ba}
    return {"tool": "rl_val_eval/1", "base": "A0", "paired_vs_base": {"CTRL": {"quantities": qjson(ctrl)}, "R": {"quantities": qjson(r)}}}


def make_arm(root, name, checks=None, kill=None, killed=None, monitor=None, kl=0.05, ref="armA", ba=None, a0=True, c0=None, meta=None, live=False, nonfinite=None, upto=1000, tags=None):
    d = os.path.join(root, name)
    os.makedirs(d)
    checks = checks if checks is not None else [rec(u) for u in (250, 500, 750, 1000)]
    w = lambda f, o: json.dump(o, open(os.path.join(d, f), "w"))                       # noqa: E731
    if live:                                                                            # the live monitor's file, one row per check
        with open(os.path.join(d, "kill_checks.jsonl"), "w") as f:
            for c in checks:
                f.write(json.dumps({**c, "kill": None}) + "\n")
    else:
        w("kill_replay.json", {"arm": "R0c", "kill": kill, "why": "synthetic", "checks": checks})
    if killed:
        w("KILLED.json", {"arm": "R0c", "code": killed, "why": "synthetic", "at": "2026-10-16T00:00:00"})
    if monitor is not None:
        open(os.path.join(d, "monitor.log"), "w").write(monitor)
    rows = [{"event": "data", "epo": {"lam": 1.0, "beta": 0.1, "ref": ref} if ref else {"lam": 1.0, "beta": 0.1}}]
    for u in range(25, upto + 1, 25):
        v = kl(u) if callable(kl) else kl
        rows.append({"event": "train", "update": u, "bceA": 0.5, "lossB": 0.4, "listnet": 4.0, "epo": -0.1, "epo_J": 0.8, "epo_kl": float("nan") if nonfinite == u else v})
        if u in (250, 500, 750, 1000):
            rows.append({"event": "score_dump", "tag": f"update{u}", "files": {}})
    open(os.path.join(d, "sft.jsonl"), "w").write("\n".join(json.dumps(r) for r in rows) + "\n")
    for u, q in (ba or {}).items():
        w(f"rlval_BvsA_update{u}.json", ba_file(q))
    if a0 and ba and 1000 in ba:
        w("rlval_vs_A0.json", a0_file(ba[1000]))
    if c0 is not None:
        w("rlval_vs_C0.json", {"tool": "rl_val_eval/1", "base": "C", "paired_vs_base": {"R": {"quantities": qjson(c0), "g5": {"NC_fall_ok": True, "TTC_fall_ok": True}}}})
    if meta:
        w("arm_meta.json", meta)
    return d


def ba_all(**over):
    q = dict(BA_OK)
    q.update(over)
    return {500: dict(q), 1000: dict(q)}


def policy(root, answers, amendment="2026-10-15 amendment (synthetic)"):
    p = os.path.join(root, f"policy_{len(os.listdir(root))}.json")
    json.dump({"amendment": amendment, "answers": answers}, open(p, "w"))
    return p


# ------------------------------------------------------------------------------------------------------------------ the suite
class Suite:
    def __init__(self, mod):
        self.m, self.n, self.fails = mod, 0, []

    def check(self, name, cond, detail=""):
        self.n += 1
        if not cond:
            self.fails.append(name + (f"  [{detail}]" if detail else ""))

    def read(self, d, pol=None):
        return self.m.read_arm(d, PREREG, pol)

    def expect(self, name, d, outcome, kind=None, pol=None, split=None):
        r = self.read(d, pol)
        self.check(f"{name}: outcome {outcome}", r["outcome"] == outcome, f"got {r['outcome']} ({r['ambiguity_kind']}): {r['reason'][:300]}")
        if kind:
            self.check(f"{name}: ambiguity kind {kind}", r["ambiguity_kind"] == kind, f"got {r['ambiguity_kind']}")
        if split:
            self.check(f"{name}: split by {split}", r["readings"]["split_by"] == split, f"got {r['readings']['split_by']}")
        if outcome != "AMBIGUOUS":
            row = self.m.parse_prereg(PREREG)[0][outcome]
            self.check(f"{name}: next lever copied from the prereg row", r["next_lever"] == row["next_lever"] and bool(r["next_lever"]))
        else:
            self.check(f"{name}: no next lever invented", r["next_lever"] is None)
        return r

    def run(self):
        with tempfile.TemporaryDirectory() as T:
            self.scenarios(T)
            self.policy(T)
            self.control(T)
            self.families(T)
            self.contract(T)
        return self.fails

    # ---- S
    def scenarios(self, T):
        m = self.m
        pol_q1 = policy(T, {"Q1": "vacuous"})
        # O1: everything holds, LB > 0 at u500 and u1000, point above the floor, nothing adverse.  O1 needs Q1 (g4 is not computable on RL-val) answered.
        d = make_arm(T, "o1", ba=ba_all())
        r = self.expect("O1 with Q1=vacuous committed", d, "O1", pol=pol_q1)
        self.check("O1: the deferred section-4 items (replicate, two pairs) are named", any("replicate" in n and "CANDIDATE" in n for n in r["notes"]))
        self.expect("O1 candidate WITHOUT the g4 answer is a reading split, never a pick", d, "AMBIGUOUS", "reading_split", split=["Q1"])
        # O2: g5 holds everywhere, a POSITIVE point but a lower bound <= 0 at both checks.
        d = make_arm(T, "o2", ba=ba_all(PDMS=(0.3, -0.2, 0.8)))
        self.expect("O2 safe but flat (point > 0, LB <= 0)", d, "O2")
        # O3: g5 (TTC) breached at u250 and u500, epo_kl A-to-B small in every candidate window.
        d = make_arm(T, "o3", checks=[rec(250, ttc=False), rec(500, ttc=False)], kill="K3", kl=0.05, upto=500)
        r = self.expect("O3 g5 breach, small KL", d, "O3")
        self.check("O3: the derived kill reproduces the recorded one", r["evidence"]["recorded_kill_reproduced"] is True and r["evidence"]["terminal_kill_derived"]["code"] == "K3")
        # O4: same breach, KL above 0.2 in every window.
        d = make_arm(T, "o4", checks=[rec(250, ttc=False), rec(500, ttc=False)], kill="K3", kl=0.4, upto=500)
        self.expect("O4 g5 breach, large KL", d, "O4")
        # O5: K4, K2, K6.
        d = make_arm(T, "o5_k4", checks=[rec(250, k4=True), rec(500, k4=True)], kill="K4", upto=500)
        self.expect("O5 via K4 (PDMS below the control at two consecutive checks)", d, "O5")
        d = make_arm(T, "o5_k2", checks=[], kill="K2", killed="K2", upto=300)
        self.expect("O5 via K2 (replay with no checks)", d, "O5")
        d = make_arm(T, "o5_k2b", checks=[rec(250)], nonfinite=275, upto=300)
        self.expect("O5 via a non-finite sft.jsonl row (K2 derived)", d, "O5")
        d = make_arm(T, "o5_k6", checks=[rec(250), rec(500)], killed="K6", upto=500, live=True)
        self.expect("O5 via K6", d, "O5")
        d = make_arm(T, "single_k4", ba=ba_all(PDMS=(0.3, -0.2, 0.8)), checks=[rec(250, k4=True), rec(500), rec(750), rec(1000)])
        self.expect("a single K4 flag is not a kill (stays O2)", d, "O2")
        # AMBIGUOUS: the table is silent
        d = make_arm(T, "gap_single_g5", ba={500: {**BA_OK, "TTC": (-1.0, -1.5, -0.5)}, 1000: dict(BA_OK)}, checks=[rec(250), rec(500, ttc=False), rec(750), rec(1000)])
        r = self.expect("table gap: g5 breached at ONE check, then recovered", d, "AMBIGUOUS", "table_gap", pol=pol_q1)
        self.check("single g5 breach: no g5 pair and no derived kill", r["evidence"]["first_g5_pair"] is None and r["evidence"]["terminal_kill_derived"] is None)
        d = make_arm(T, "gap_split_g5", ba=ba_all(), checks=[rec(250, ttc=False), rec(500), rec(750, ttc=False), rec(1000)])
        self.expect("table gap: g5 breached at NON-consecutive checks", d, "AMBIGUOUS", "table_gap", pol=pol_q1)
        d = make_arm(T, "gap_g3_k3", checks=[rec(250, g3=False), rec(500, g3=False)], kill="K3", upto=500)
        self.expect("table gap: K3 fired on g3 only (O3/O4 say g5, O5 omits K3)", d, "AMBIGUOUS", "table_gap")
        d = make_arm(T, "gap_mixed_k3", checks=[rec(250, g3=False), rec(500, ttc=False)], kill="K3", upto=500)
        self.expect("table gap: K3 on a mixed g3 / g5 pair", d, "AMBIGUOUS", "table_gap")
        d = make_arm(T, "gap_g3_o1", ba=ba_all(), checks=[rec(250), rec(500), rec(750, g3=False), rec(1000)])
        self.expect("g3 breached once, LB > 0: neither O1 nor O2 applies", d, "AMBIGUOUS", "table_gap", pol=pol_q1)
        d = make_arm(T, "gap_adverse", ba=ba_all(DAC=(-1.0, -1.5, -0.5)))
        self.expect("LB > 0 but DAC adversely separated: no row", d, "AMBIGUOUS", "table_gap", pol=policy(T, {"Q1": "vacuous", "Q7": "narrow"}))
        # AMBIGUOUS: a reading splits the outcome
        d = make_arm(T, "kl_straddle", checks=[rec(250, ttc=False), rec(500, ttc=False)], kill="K3", kl=lambda u: 0.1 if u <= 250 else 0.3, upto=500)
        self.expect("epo_kl window straddles 0.2", d, "AMBIGUOUS", "reading_split", split=["Q2"])
        d = make_arm(T, "nc_breach", checks=[rec(250, nc=False), rec(500, nc=False)], kill="K3", upto=500)
        self.expect("an NC-involved g5 breach (O5 names an NC bound)", d, "AMBIGUOUS", "reading_split", split=["Q3"])
        d = make_arm(T, "kmon", checks=[rec(250)], killed="K-MON", upto=300)
        self.expect("K-MON is not in O5's list", d, "AMBIGUOUS", "reading_split", split=["Q8"])
        d = make_arm(T, "lb_mixed", ba={500: dict(BA_OK), 1000: {**BA_OK, "PDMS": (0.3, -0.2, 0.8)}})
        self.expect("LB > 0 at u500 only: O2's check is unnamed", d, "AMBIGUOUS", "reading_split", split=["Q4"])
        d = make_arm(T, "under_floor", ba=ba_all(PDMS=(0.1, 0.02, 0.2)))
        self.expect("LB > 0 but the point is under +0.25 (section 4 bar)", d, "AMBIGUOUS", "reading_split", split=["Q6"], pol=pol_q1)
        d = make_arm(T, "ddc_adverse", ba=ba_all(DDC=(-1.0, -1.5, -0.5)))
        self.expect("DDC adverse: the scope of 'longitudinal / lateral' decides", d, "AMBIGUOUS", "reading_split", split=["Q7"], pol=pol_q1)
        # AMBIGUOUS: an input the chain does not write / does not give
        d = make_arm(T, "base_kl", checks=[rec(250, ttc=False), rec(500, ttc=False)], kill="K3", ref="base", kl=0.05, upto=500)
        self.expect("epo_kl is base-to-B (R0 / R0b trainer): O3 / O4 cannot be read", d, "AMBIGUOUS", "missing_input")
        d = make_arm(T, "no_ref", checks=[rec(250, ttc=False), rec(500, ttc=False)], kill="K3", ref=None, kl=0.05, upto=500)
        self.expect("data.epo.ref absent (pre-R0c trainer)", d, "AMBIGUOUS", "missing_input")
        d = make_arm(T, "no_ba", ba=None)
        self.expect("a finished arm with no B-vs-A read: the chain writes no B - A interval (GAP 0)", d, "AMBIGUOUS", "missing_input")
        d = make_arm(T, "bad_base", ba=None)
        json.dump(ba_file(BA_OK, base="C", arm="B"), open(os.path.join(d, "rlval_BvsA_update500.json"), "w"))
        json.dump(ba_file(BA_OK, base="C", arm="B"), open(os.path.join(d, "rlval_BvsA_update1000.json"), "w"))
        r = self.expect("a B-vs-A file against any control but A is refused", d, "AMBIGUOUS", "missing_input")
        self.check("bad_base: the error names the control", "against any other control" in json.dumps(r["evidence"]["b_minus_a_reads_errors"]) or "must be rl_val_eval" in json.dumps(r["evidence"]["b_minus_a_reads_errors"]))
        d = make_arm(T, "missing_check", ba=ba_all(), checks=[rec(250), rec(500), rec(1000)])
        self.expect("a check missing and no kill on record", d, "AMBIGUOUS", "missing_input", pol=pol_q1)
        # AMBIGUOUS: the arm dir contradicts itself
        d = make_arm(T, "conflict_kill", checks=[rec(250), rec(500)], kill="K3", upto=500)
        self.expect("recorded K3 that the check records do not derive", d, "AMBIGUOUS", "conflict")
        d = make_arm(T, "conflict_ba", ba=ba_all(TTC=(-1.0, -1.5, -0.5)), checks=[rec(250), rec(500), rec(750), rec(1000)])
        self.expect("monitor says g5 holds, the B-vs-A read says TTC fell by 1.0", d, "AMBIGUOUS", "conflict", pol=pol_q1)
        d = make_arm(T, "conflict_a0", ba=ba_all())
        j = json.load(open(os.path.join(d, "rlval_vs_A0.json")))
        j["paired_vs_base"]["R"]["quantities"]["PDMS"]["mean"] += 0.5
        json.dump(j, open(os.path.join(d, "rlval_vs_A0.json"), "w"))
        self.expect("rlval_vs_A0.json and the B-vs-A read disagree on B - A", d, "AMBIGUOUS", "conflict", pol=pol_q1)
        d = make_arm(T, "conflict_sources", checks=[rec(250, ttc=False), rec(500, ttc=False)], kill="K3", killed="K4", upto=500)
        self.expect("KILLED.json and kill_replay.json name different kills", d, "AMBIGUOUS", "conflict")
        # boundaries
        d = make_arm(T, "kl_eq", checks=[rec(250, ttc=False), rec(500, ttc=False)], kill="K3", kl=0.2, upto=500)
        self.expect("epo_kl exactly 0.2 is O3 (<=)", d, "O3")
        d = make_arm(T, "kl_gt", checks=[rec(250, ttc=False), rec(500, ttc=False)], kill="K3", kl=0.2000001, upto=500)
        self.expect("epo_kl just above 0.2 is O4", d, "O4")
        d = make_arm(T, "lb_zero", ba=ba_all(PDMS=(0.5, 0.0, 1.0)))
        self.expect("a lower bound of exactly 0 is NOT > 0 (O2)", d, "O2")
        d = make_arm(T, "lb_zero_final", ba={500: dict(BA_OK), 1000: {**BA_OK, "PDMS": (0.5, 0.0, 1.0)}})
        self.expect("a lower bound of exactly 0 at u1000 only, Q1 and Q4 committed: no row", d, "AMBIGUOUS", "table_gap", pol=policy(T, {"Q1": "vacuous", "Q4": "both_le0"}))
        d = make_arm(T, "g5_edge", ba=ba_all(PDMS=(0.3, -0.2, 0.8)), checks=[rec(u) for u in (250, 500, 750, 1000)])
        self.expect("g5 holding everywhere is O2 when the interval touches 0", d, "O2")
        # the live monitor's file and the replay agree / disagree
        d = make_arm(T, "live_ok", checks=[rec(250, ttc=False), rec(500, ttc=False)], killed="K3", live=True, kl=0.05, upto=500, monitor="ZZRLS1_MONITOR_READY rss_mb=57\nZZRLS1_K3_LIVE R0c {}\nZZRLS1_KILL R0c K3 g3/g5 bound breached at update250 and update500\n")
        r = self.expect("live monitor files (kill_checks.jsonl, KILLED.json, monitor.log) read the same as the replay", d, "O3")
        self.check("live: monitor.log kill line recorded", r["evidence"]["recorded_kill_sources"].get("monitor.log") == "K3")
        d = make_arm(T, "k3_disabled", checks=[rec(250, g3=False, ttc=False, k3en=False), rec(500, g3=False, ttc=False, k3en=False), rec(750, g3=False, ttc=False, k3en=False)], kl=0.05, upto=750)
        r = self.expect("K3 disabled (base breaches g3 on RL-val): O3/O4 still read from the data", d, "O3")
        self.check("k3 disabled is noted", any("k3_enabled=False" in n for n in r["notes"]))
        d = make_arm(T, "c0_ctrl", checks=[rec(250, ttc=False), rec(500, ttc=False)], kill="K3", kl=0.05, upto=500, meta={"monitor_control": "C0"})
        r = self.expect("records taken against C0 are labelled, not hidden", d, "O3")
        self.check("monitor control note", any("NOT against the arm's own A" in n for n in r["notes"]))
        # CLI exit codes and the ZZ line
        for name, want in (("o2", 0), ("kl_straddle", 2)):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = self.m.main([os.path.join(T, name), "--prereg", PREREG])
            self.check(f"CLI {name}: exit {want}", rc == want, str(rc))
            self.check(f"CLI {name}: ZZR0C_OUTCOME line", f"ZZR0C_OUTCOME {'O2' if name == 'o2' else 'AMBIGUOUS'}" in buf.getvalue())
        with contextlib.redirect_stdout(io.StringIO()):
            self.check("CLI: a missing arm dir is refused (3)", self.m.main([os.path.join(T, "nope")]) == 3)
            self.check("CLI: a missing prereg is refused (3)", self.m.main([os.path.join(T, "o2"), "--prereg", os.path.join(T, "nope.md")]) == 3)

    # ---- P
    def policy(self, T):
        d = os.path.join(T, "kl_straddle")
        r = self.read(d, policy(T, {"Q2": "second_breach_window"}))
        self.check("policy: Q2 committed narrows the straddle to O4", r["outcome"] == "O4" and r["readings"]["answered"] == {"Q2": "second_breach_window"}, r["outcome"])
        r = self.read(d, policy(T, {"Q2": "first_breach_window"}))
        self.check("policy: the other answer gives the other outcome (the policy decides, the reader does not)", r["outcome"] == "O3", r["outcome"])
        p = policy(T, {"Q2": "row_at_second_breach"})
        r = self.read(d, p)
        self.check("policy: its sha256 and amendment are recorded", r["readings"]["policy_sha256"] == self.m.sha256_file(p) and r["readings"]["policy_amendment"].startswith("2026"))
        for name, ans, am in (("no amendment", {"Q2": "first_breach_window"}, ""), ("unknown option", {"Q2": "median"}, "x"), ("unknown question", {"Q99": "a"}, "x")):
            try:
                self.read(d, policy(T, ans, am))
                self.check(f"policy refused: {name}", False)
            except ValueError:
                self.check(f"policy refused: {name}", True)

    # ---- C
    def control(self, T):
        for name, pol in (("o2", None), ("o3", None), ("o1", policy(T, {"Q1": "vacuous"})), ("o5_k4", None), ("gap_single_g5", policy(T, {"Q1": "vacuous"}))):
            src = os.path.join(T, name)
            base = self.read(src, pol)
            for variant, c0 in (("with a C0 read that agrees", dict(BA_OK)), ("with a C0 read that contradicts (PDMS LB > 0, TTC fell)", {**BA_OK, "PDMS": (2.0, 1.0, 3.0), "TTC": (-3.0, -4.0, -2.0)}),
                                ("with a C0 read that is adverse everywhere", {k: (-1.0, -2.0, -0.5) for k in BA_OK})):
                d2 = os.path.join(T, f"{name}__c0_{abs(hash(variant)) % 10 ** 6}")
                import shutil
                shutil.copytree(src, d2)
                json.dump({"tool": "rl_val_eval/1", "base": "C", "paired_vs_base": {"R": {"quantities": qjson(c0), "g5": {"NC_fall_ok": True, "TTC_fall_ok": False}}}}, open(os.path.join(d2, "rlval_vs_C0.json"), "w"))
                r2 = self.read(d2, pol)
                self.check(f"second control {variant}: {name} keeps its outcome", r2["outcome"] == base["outcome"] and r2["reason"] == base["reason"] and r2["next_lever"] == base["next_lever"], f"{base['outcome']} -> {r2['outcome']}")
                self.check(f"second control {variant}: reported as present", r2["evidence"]["second_control"]["present"] is True and "NEVER enters" in r2["evidence"]["second_control"]["note"])
            self.check(f"{name}: no C0 file -> present False", base["evidence"]["second_control"]["present"] is False)

    # ---- F
    def families(self, T):
        r = self.read(os.path.join(T, "o2"))
        f = r["families"]
        self.check("families: STRATEGIC is PAUSED, not omitted, not a pass", "PAUSED" in f["STRATEGIC"] and "ITEM 25" in f["STRATEGIC"])
        self.check("families: TACTICAL is N/A with its reason", f["TACTICAL"].startswith("N/A") and "scorer-only" in f["TACTICAL"])
        self.check("families: longitudinal EP carries its interval", f["LONGITUDINAL"]["EP_progress"]["kind"] == "interval" and f["LONGITUDINAL"]["EP_progress"]["ci95"] == [1.0, 3.0])
        self.check("families: lateral DAC and the TTC / NC / comfort sub-scores are present", "B_minus_A_x100" in f["LATERAL"]["DAC"] and all("B_minus_A_x100" in f["SUBSCORES"][k] for k in ("TTC", "NC", "comfort_C")))
        self.check("families: g4 is declared NOT COMPUTED", "NOT COMPUTED" in f["LONGITUDINAL"]["g4_longitudinal_bias"])
        self.check("families: the tier is stamped and is not T1", "NOT T1" in r["tier"] and "selection tier" in r["tier"])
        d = make_arm(T, "only_a0", checks=[rec(u) for u in (250, 500, 750, 1000)], ba={1000: dict(BA_OK)})
        os.remove(os.path.join(d, "rlval_BvsA_update1000.json"))
        r = self.read(d)
        self.check("families: with only the chain's rlval_vs_A0.json B - A is a POINT with no interval", r["families"]["LONGITUDINAL"]["EP_progress"]["ci95"] is None and "point" in r["families"]["LONGITUDINAL"]["EP_progress"]["kind"])
        self.check("families: such an arm cannot reach O1 / O2 (no interval)", r["outcome"] == "AMBIGUOUS" and r["ambiguity_kind"] == "missing_input")
        r = self.read(os.path.join(T, "o3"))
        self.check("families: a killed arm reports what it has and says what it lacks", r["families"]["LATERAL"]["DAC"] == "NOT AVAILABLE in the arm dir")

    # ---- contract
    def contract(self, T):
        rows, sha = self.m.parse_prereg(PREREG)
        self.check("contract: five rows parsed", sorted(rows) == ["O1", "O2", "O3", "O4", "O5"])
        self.check("contract: the prereg sha256 is the pinned one", sha == self.m.PREREG_SHA256)
        self.check("contract: the O3 / O4 split is the prereg's 0.2", "0.2" in rows["O3"]["condition"] and "> 0.2" in rows["O4"]["condition"] and self.m.KL_SPLIT == 0.2)
        self.check("contract: the g5 and K4 bounds are rl_val_eval's", self.m.G5_BOUND == 0.5 and self.m.K4_BOUND == 0.5 and self.m.S4_FLOOR == 0.25)
        amended = os.path.join(T, "PREREG_amended.md")
        open(amended, "w", encoding="utf-8").write(open(PREREG, encoding="utf-8").read().replace("<= 0.2", "<= 0.3", 1))
        r = self.m.read_arm(os.path.join(T, "o2"), amended)
        self.check("contract: an amended prereg is flagged and noted", r["prereg"]["matches_pinned"] is False and any("differs" in n for n in r["notes"]))
        broken = os.path.join(T, "PREREG_broken.md")
        open(broken, "w", encoding="utf-8").write("# nothing\n")
        try:
            self.m.read_arm(os.path.join(T, "o2"), broken)
            self.check("contract: a prereg without the table is refused", False)
        except ValueError:
            self.check("contract: a prereg without the table is refused", True)
        r = self.read(os.path.join(T, "o3"))
        self.check("contract: inputs carry sha256", all(len(v) == 64 for v in r["inputs_sha256"].values()) and "sft.jsonl" in r["inputs_sha256"])
        self.check("contract: every question has options and text", all(q["options"] and q["text"] for q in r["readings"]["questions"].values()) and len(r["readings"]["questions"]) == 8)


# ------------------------------------------------------------------------------------------------------------------ RED mutations
MUTATIONS = {
    "swap the O3 / O4 KL threshold": ('("O3" if v <= KL_SPLIT else "O4")', '("O4" if v <= KL_SPLIT else "O3")'),
    "KL boundary: <= becomes <": ('("O3" if v <= KL_SPLIT else "O4")', '("O3" if v < KL_SPLIT else "O4")'),
    "g5 'any' instead of 'two consecutive' (first_g5_pair)": ('a in checks and b in checks and not checks[a]["g5_ok"] and not checks[b]["g5_ok"]', 'a in checks and b in checks and (not checks[a]["g5_ok"] or not checks[b]["g5_ok"])'),
    "K3 'any' instead of 'two consecutive' (judge)": ('prev is not None and breach(rec) and breach(prev)', 'prev is not None and (breach(rec) or breach(prev))'),
    "K4 on ONE check (judge)": ('if prev is not None and rec["k4"] and prev["k4"]:', 'if prev is not None and (rec["k4"] or prev["k4"]):'),
    "lower bound read as the point": ('lb5, lb10, mean10 = q[500]["PDMS"]["ci95"][0], q[1000]["PDMS"]["ci95"][0], q[1000]["PDMS"]["mean"]',
                                      'lb5, lb10, mean10 = q[500]["PDMS"]["mean"], q[1000]["PDMS"]["mean"], q[1000]["PDMS"]["mean"]'),
    "lower bound > 0 becomes >= 0": ('if not lb10 > 0:', 'if not lb10 >= 0:'),
    "a missing C0 read changes the outcome": ('    other_rec = rk in (', '    if not ev["second_control"]["present"]:\n        return "MISSING", "no C0"\n    other_rec = rk in ('),
    "the C0 read enters the outcome": ('    lb5, lb10, mean10 =', '    if ev["second_control"].get("B_minus_C0_arm_A", {}).get("PDMS", {}).get("ci95", [1])[0] > 0:\n        return "O1", "C0 says so"\n    lb5, lb10, mean10 ='),
    "g3 ignored for O1": ('    if not g3_all:\n        why.append', '    if False:\n        why.append'),
    "the KL reference is not checked (base-to-B used as A-to-B)": ('if ev["kl_ref"] != "armA":', 'if False:'),
    "AMBIGUOUS replaced by a pick": ('elif len(labels) == 1 and next(iter(labels)) in rows:\n        outcome, kind, cand = next(iter(labels)), "decided", []',
                                     'elif any(l_ in rows for l_ in labels):\n        outcome, kind, cand = sorted(l_ for l_ in labels if l_ in rows)[0], "decided", []'),
    "conflicts ignored": ('    if ev["conflicts"]:\n        outcome, kind = "AMBIGUOUS", "conflict"', '    if False:\n        outcome, kind = "AMBIGUOUS", "conflict"'),
    "Q1 (g4) ignored": ('    if R["Q1"] == "blocking":', '    if False:'),
    "the next lever is retyped, not copied": ('"next_lever": rows[outcome]["next_lever"] if outcome in rows else None', '"next_lever": "see prereg" if outcome in rows else None'),
    "STRATEGIC family reported as a pass": ('"PAUSED (PI ruling 2026-09-19, PI_DECISION_QUEUE ITEM 25): reported as paused, not as a pass, not omitted."', '"PASS"'),
    "a B-vs-A file against any control is accepted": ('if j.get("base") != "A" or sorted((j.get("paired_vs_base") or {})) != ["B"]:', 'if False:'),
    "a kill the records cannot derive is accepted": ('        if rk == code and not (j and j["code"] == code):', '        if False:'),
}
# which check names must go RED for each mutation (substring of the failing check name)
EXPECT = {
    "swap the O3 / O4 KL threshold": "O3 g5 breach, small KL",
    "KL boundary: <= becomes <": "epo_kl exactly 0.2 is O3",
    "g5 'any' instead of 'two consecutive' (first_g5_pair)": "single g5 breach: no g5 pair",
    "K3 'any' instead of 'two consecutive' (judge)": "single g5 breach: no g5 pair",
    "K4 on ONE check (judge)": "a single K4 flag is not a kill",
    "lower bound read as the point": "O2 safe but flat",
    "lower bound > 0 becomes >= 0": "a lower bound of exactly 0",
    "a missing C0 read changes the outcome": "keeps its outcome",
    "the C0 read enters the outcome": "second control",
    "g3 ignored for O1": "g3 breached once",
    "the KL reference is not checked (base-to-B used as A-to-B)": "epo_kl is base-to-B",
    "AMBIGUOUS replaced by a pick": "reading split",
    "conflicts ignored": "recorded K3 that the check records do not derive",
    "Q1 (g4) ignored": "O1 candidate WITHOUT the g4 answer",
    "the next lever is retyped, not copied": "next lever copied",
    "STRATEGIC family reported as a pass": "STRATEGIC is PAUSED",
    "a B-vs-A file against any control is accepted": "against any control but A",
    "a kill the records cannot derive is accepted": "recorded K3 that the check records do not derive",
}


def main():
    base = Suite(load_reader())
    fails = base.run()
    for f in fails:
        print("RED  ", f)
    assert not fails, f"{len(fails)} check(s) failed on the UNMUTATED reader"
    total = base.n
    print(f"unmutated suite: {base.n} checks, all green")
    assert set(MUTATIONS) == set(EXPECT), "every mutation needs its expected-RED check"
    for name, mut in MUTATIONS.items():
        s = Suite(load_reader(mut))
        try:
            fl = s.run()
        except Exception as e:                                                      # a mutation may break the reader outright: that is RED too
            fl = [f"CRASH {type(e).__name__}: {e}"]
        hit = [f for f in fl if EXPECT[name] in f] or ([f for f in fl if f.startswith("CRASH")] if fl and fl[0].startswith("CRASH") else [])
        assert hit, f"mutation {name!r}: the check containing {EXPECT[name]!r} must go RED; failing checks were {fl[:4]}"
        print(f"RED mutation caught: {name:62s} -> {len(fl)} failing check(s), e.g. {hit[0][:70]}")
        total += 1
    print(f"ZZTEST_PASS {total}  ({base.n} checks on the unmutated reader + {len(MUTATIONS)} RED mutations)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
