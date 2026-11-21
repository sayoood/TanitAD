#!/usr/bin/env python3
"""dag2t1_selftest.py -- the self-test AND the mutation proof of the PREREG_V8F_DAGGER2 dev-box read.  A plain script (NO pytest), CPU only, launches no rollout.

  python dag2t1_selftest.py [--quick] [--real-pulls]   prints one line per check; exit 0 iff everything is as expected.  --quick skips the chain scenarios and the chain mutants.
  BLINDING (default): no check opens, reads or hashes a real `eval_M2b-dag2*` file under v8f-dagger2/ (the DAG2 / DAG2-0 T0 evals): the pull check stream-hashes every file its SHA256SUMS.txt lists.
  The fixtures carry a SYNTHETIC stand-in eval; the REAL pulled-run checks (`P REAL pulled M2b_s0_dag2 / dag20`) and the real-configuration chain scenario (`CH real`) are NOT RUN, and
  `static defaults` pins the chain's default paths instead.  --real-pulls runs them (it hashes the pulled eval files; for a session that is no longer under that blinding).
  AMENDMENT 1 (A1-2026-10-27, 1446333d): the outcome table is complete (FIXED 1 / PARTIAL 11 / NO EFFECT 4 over the 16 guarded cells); E checks tie EXPECT16 to the PREREG's own enumerate_cells.py.

Sections
  T/O/B/C/G/N/L  the logic of dag2_read.py (the PREREG's own composition): the outcome table (one case per outcome, built from synthetic reader outputs, all 16 crawl/move combinations
                 against a hand-written expected table), the fixed-RF bars, the crawl set per seed, the guards, the newly-worse clause with its two exemptions, the two layer-G substitutions.
  P              dag2t1_pullcheck.py on synthetic runs (COMPLETE / WRONG cases) and on the REAL pulled s0 pair.
  CH             dag2_t1_chain.sh --dry-run: run order, arm names, tags, paths, markers, the C: and E: gates, the tool tree, the PREREG pin (scenarios on a fixture ART, one on the real configuration).
  M              MUTATION PROOF: each mutant is ONE textual defect in a COPY of dag2_read.py / dag2t1_pullcheck.py / dag2_t1_chain.sh; the sections that cover that file are re-run on the copy and the
                 mutant is KILLED iff a check goes RED.  A mutant whose anchor is not found exactly once is an ANCHOR ERROR (counted as a failure).  The unmutated control must be fully green.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import time
import types
from collections import Counter

sys.dont_write_bytecode = True
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = os.path.dirname(os.path.abspath(__file__))
H2 = HERE.replace("\\", "/")
ART = "E:/Projects/TanitAD-artifacts"
PY = sys.executable
QUICK = "--quick" in sys.argv
REAL_PULLS = "--real-pulls" in sys.argv         # default OFF: the real DAG2 eval files are never opened (blinding), see the module docstring
BASH = "C:/Program Files/Git/bin/bash.exe"      # NOT the bare name: PATH finds System32\bash.exe (WSL) first
MUT = os.path.join(HERE, "_mut")
ST = os.path.join(HERE, "_st").replace("\\", "/")
os.makedirs(MUT, exist_ok=True)
os.makedirs(ST, exist_ok=True)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


REAL = load(os.path.join(HERE, "dag2_read.py"), "dag2_read_real")
DR = REAL.load_dag_read(REAL.DEFAULTS["dag_read"])
BANKED = json.load(open(REAL.DEFAULTS["reada_banked"], encoding="utf-8"))["READ_A"]["rows"]


def row(delta, sep, **kw):
    r = {"delta": delta, "lo": delta - 0.01, "hi": delta + 0.01, "separated": sep}
    r.update(kw)
    return r


def ok_row(lab):
    return row(0.0, False)


C_LABELS = ["curvature MAE", "L2 curvature MAE (GT>=5.2)", "M1 crawl share", "yaw-rate MAE"]
ALL_LABELS = ["ADE6", "speed MAE", "along-track MAE", "accel MAE", "DK-VIOL", "DK-HW", "DK-TG", "heading MAE", "yaw-rate MAE", "curvature MAE", "cross-track MAE", "L1 lateral-accel MAE",
              "L2 curvature MAE (GT>=5.2)", "TAC-LAT", "TAC-LON", "S-POS@30", "S-POS@50", "S-POS@80", "M1 crawl share"]
DIRS = {l: "lower" for l in ALL_LABELS}
DIRS.update({"TAC-LAT": "higher", "TAC-LON": "higher"})


def table(over=None):
    """a Read A row table, every row not separated (ns); ``over`` = {label: row}"""
    t = {l: ok_row(l) for l in ALL_LABELS}
    t.update(over or {})
    return t


WORSE, BETTER, NS = row(0.5, True), row(-0.5, True), row(0.0, False)


def flags4(c0, c1, m0, m1, M):
    """layer rows that make R-CRAWL / R-MOVE equal (c_s, m_s) at each seed, through compose()"""
    g = {s: {"R-TLAT": row(0.1, True), "R-TLON": row(0.1, True), "R-ADE": row(-0.5, True), "R-T0": {"point": 0.0}} for s in M.SEEDS}
    lay = {"DAG2_minus_P10": {}, "DAG2_minus_DAG20": {}, "DAG_minus_P10": {}}
    for s, (c, m) in zip(M.SEEDS, ((c0, m0), (c1, m1))):
        lay["DAG2_minus_P10"][s] = table({l: (NS if c else WORSE) for l in M.CRAWL_SET[s]})
        lay["DAG2_minus_DAG20"][s] = table({l: (BETTER if m else NS) for l in M.CRAWL_SET[s]})
        lay["DAG_minus_P10"][s] = table({l: WORSE for l in M.CRAWL_SET[s]})              # DAG was separated-worse on the crawl set (so a worse DAG2 there is not NEW)
    return g, lay


EXPECT16 = {   # (c0, c1, m0, m1) -> matching rows in table order; written by hand from the AMENDED table (A1-2026-10-27): the guarded cells are partitioned on R-MOVE alone
    **{(c0, c1, 0, 0): ["NO EFFECT"] for c0 in (0, 1) for c1 in (0, 1)},
    (1, 1, 1, 1): ["FIXED"], (1, 0, 1, 1): ["PARTIAL"], (0, 1, 1, 1): ["PARTIAL"], (0, 0, 1, 1): ["PARTIAL"],
    (1, 0, 1, 0): ["PARTIAL"], (1, 1, 1, 0): ["PARTIAL"], (0, 0, 1, 0): ["PARTIAL"], (0, 1, 1, 0): ["PARTIAL"],
    (0, 1, 0, 1): ["PARTIAL"], (1, 1, 0, 1): ["PARTIAL"], (0, 0, 0, 1): ["PARTIAL"], (1, 0, 0, 1): ["PARTIAL"],
}
NEWLY_PLACED = [(1, 0, 0, 1), (0, 1, 1, 0), (0, 0, 1, 0), (0, 0, 0, 1)]       # the four cells that matched NO row as landed (R-MOVE at one seed, R-CRAWL failing at that seed)
ONE_SEED_PARTIAL = {(1, 1, 1, 0), (1, 1, 0, 1), (1, 0, 1, 0), (0, 1, 0, 1)} | set(NEWLY_PLACED)     # PARTIAL with R-MOVE at exactly one seed: reported "PARTIAL (one seed)", adopts nothing
ADOPTS = {(1, 1, 1, 1), (1, 0, 1, 1), (0, 1, 1, 1), (0, 0, 1, 1)}               # FIXED, and PARTIAL with R-MOVE at BOTH seeds: the only cells that adopt DAG2
PARTIAL_COND = "R-MOVE at both seeds but R-CRAWL fails somewhere; or R-MOVE at one seed only; guards held"       # Amendment 1's PARTIAL row, written out independently of the reader
PARTIAL_NEXT = ("adopt DAG2 as the base if R-MOVE holds at both seeds, else DAG stays the base; report. The residual crawl is then a launch-intent defect (O's v0 at tick 1, `R76`). "
                "Next: the launch-recovery target of `PREREG_V8F_HACC_MC` §7 on the base's states, under its own PREREG")
ENUM_PATH = os.path.join(HERE, "inputs", "enumerate_cells.py")                 # the PREREG's own enumeration (blob of 1446333d), staged and sha-pinned
ENUM_SHA = "3a437ea88e87b56957f73e8613d655fe74e441f222354ab9abcab053a7661255"


def load_enumeration():
    """-> (sha256 of the staged enumerate_cells.py, its namespace after running, its exit code, its stdout).  It is executed in-process with stdout captured; its SystemExit is its exit code."""
    raw = open(ENUM_PATH, "rb").read()
    ns, buf, rc = {"__name__": "enumerate_cells"}, io.StringIO(), 0
    with contextlib.redirect_stdout(buf):
        try:
            exec(compile(raw.decode("utf-8"), ENUM_PATH, "exec"), ns)
        except SystemExit as e:
            rc = e.code
    return hashlib.sha256(raw).hexdigest(), ns, rc, buf.getvalue()


def logic_checks(M):
    """-> [(name, ok, why)] of the pure logic of module M (the real dag2_read, or a mutant of it)."""
    R = []

    def chk(n, ok, why=""):
        R.append((n, bool(ok), why))

    def guarded(fn, n):
        try:
            return fn()
        except Exception as e:      # noqa: BLE001 -- a defect that raises is a defect that is caught
            chk(n, False, f"raised {type(e).__name__}: {e}")
            return None
    # ---- T: the table is the PREREG's
    ok, det = guarded(lambda: M.check_prereg2_text(M.DEFAULTS["prereg2"]), "T1") or (False, "raised")
    chk("T1 the pinned PREREG_V8F_DAGGER2.md (1446333d blob) holds exactly the six amended outcome-table rows the reader codes", ok, det)
    chk("T2 table order", M.OUTCOME_NAMES_2 == ("FIXED", "PARTIAL", "NO EFFECT", "HARMS", "BLOCKED", "REFUSED"))
    chk("T3 pinned PREREG sha256 (the 1446333d blob)", M.PREREG2_SHA256 == "c4239e4590ee26e30ce6f525987aca82e3fb8ff034b7264321e2d0faa724f1c6")
    ptxt = open(M.DEFAULTS["prereg2"], encoding="utf-8").read()
    chk("T4 table_problems on the pinned amended text: none", guarded(lambda: M.table_problems(ptxt), "T4") == [], guarded(lambda: M.table_problems(ptxt), "T4b"))
    asl = ptxt.split("## Amendment 1")[0]
    chk("T5 the AS-LANDED text (everything before Amendment 1) is not the amended table", guarded(lambda: M.table_problems(asl), "T5") not in (None, []))
    chk("T6 a PREREG whose PARTIAL condition is the as-landed one is refused",
        guarded(lambda: M.table_problems(ptxt.replace(PARTIAL_COND, "R-MOVE at both seeds but R-CRAWL fails somewhere; or R-CRAWL and R-MOVE at one seed only; guards held")), "T6") not in (None, []))
    chk("T7 a PREREG whose PARTIAL committed next is the as-landed one is refused", guarded(lambda: M.table_problems(ptxt.replace("on the base's states, under its own PREREG", "on DAG2 states, under its own PREREG")), "T7") not in (None, []))
    head_, tail_ = ptxt.split("The outcome table, as read from now on", 1)
    nofixed = head_ + "The outcome table, as read from now on" + "\n".join(x for x in tail_.split("\n") if not x.startswith("| **FIXED**"))
    chk("T8 an amended table with a row missing is refused", guarded(lambda: M.table_problems(nofixed), "T8") not in (None, []))
    chk("T9 the PARTIAL row is Amendment 1's, written out independently of the reader", M.table_row_2("PARTIAL") == (PARTIAL_COND, PARTIAL_NEXT))
    # ---- O: the outcome table, one case per outcome, from synthetic reader outputs through compose()
    for k, want in EXPECT16.items():
        g, lay = flags4(*k, M)
        dec, _ = guarded(lambda: M.compose(g, lay, DIRS), f"O16:{k}") or ({"matching_rows_in_table_order": ["raised"]}, None)
        chk(f"O16 (c0,c1,m0,m1)={k} -> {want or 'UNPLACED'}", dec["matching_rows_in_table_order"] == want and dec["OUTCOME"] == (want[0] if want else "UNPLACED"), dec["matching_rows_in_table_order"])
    for nm, k in (("FIXED", (1, 1, 1, 1)), ("PARTIAL(a)", (1, 0, 1, 1)), ("PARTIAL(b)", (1, 1, 1, 0)), ("PARTIAL(b, newly placed)", (0, 0, 1, 0)), ("NO EFFECT", (0, 1, 0, 0))):
        g, lay = flags4(*k, M)
        dec, _ = M.compose(g, lay, DIRS)
        w = nm.split("(")[0]
        cond, nxt = next((c, n) for x, c, n in M.OUTCOME_TABLE_2 if x == w)
        chk(f"O1 {nm}: outcome word, condition and committed next carried verbatim", dec["OUTCOME"] == w and dec["condition"] == cond and dec["committed_next"] == nxt)
    for s in (0, 1):
        for gname in ("R-TLAT", "R-TLON", "R-ADE", "R-T0", "NEWLY-WORSE"):
            for base in ((1, 1, 1, 1), (0, 0, 0, 0)):
                g, lay = flags4(*base, M)
                if gname == "R-TLAT":
                    g[s]["R-TLAT"] = row(-0.1, True)
                elif gname == "R-TLON":
                    g[s]["R-TLON"] = row(-0.1, True)
                elif gname == "R-ADE":
                    g[s]["R-ADE"] = row(0.3, True)
                elif gname == "R-T0":
                    g[s]["R-T0"] = {"point": 0.0605}
                else:
                    lay["DAG_minus_P10"][s]["speed MAE"] = NS
                    lay["DAG2_minus_P10"][s]["speed MAE"] = WORSE
                dec, _ = M.compose(g, lay, DIRS)
                chk(f"O2 guard {gname} fails at s{s} (base {base}) -> HARMS and only HARMS", dec["matching_rows_in_table_order"] == ["HARMS"], dec["matching_rows_in_table_order"])
    g, lay = flags4(1, 1, 1, 1, M)
    g[0]["R-T0"] = {"point": 0.0604}
    chk("O3 R-T0 exactly 0.0604 holds (inclusive)", M.compose(g, lay, DIRS)[0]["OUTCOME"] == "FIXED")
    g[0]["R-ADE"] = row(0.3, False)
    g[1]["R-TLAT"] = row(-0.3, False)
    chk("O4 a guard that reads worse but is NOT separated holds", M.compose(g, lay, DIRS)[0]["OUTCOME"] == "FIXED")
    bb = M.blocked_block_2({"s0": 0.4})
    cond, nxt = next((c, n) for x, c, n in M.OUTCOME_TABLE_2 if x == "BLOCKED")
    chk("O5 BLOCKED block", bb["OUTCOME"] == "BLOCKED" and bb["committed_next"] == nxt and bb["condition"] == cond and bb["coverage_min"] == 0.5)
    chk("O6 REFUSED cell", M.table_row_2("REFUSED") == ("any control fails", "fix and re-run"))
    ub = M.outcome_block_2({"OUTCOME": "UNPLACED", "matching_rows_in_table_order": [], "flags": {}})
    chk("O7 UNPLACED carries no outcome word and no committed next, and says it is unreachable since Amendment 1", ub["committed_next"] is None and "PREREG gap" in ub["note"] and "unreachable" in ub["note"] and "A1-2026-10-27" in ub["note"])
    try:
        M.decide2({0: {"R-CRAWL": True, "R-MOVE": True}}, {0: {}})
        chk("O8 decide2 needs both seeds", False)
    except ValueError:
        chk("O8 decide2 needs both seeds", True)
    # ---- O9-O11: the Amendment 1 output fields, clause flags and HARMS
    for k in EXPECT16:
        g, lay = flags4(*k, M)
        dec, _ = M.compose(g, lay, DIRS)
        want = (k in ONE_SEED_PARTIAL, k in ADOPTS, "one seed" if k in ONE_SEED_PARTIAL else None)
        got = (dec.get("PARTIAL_one_seed"), dec.get("adopts_DAG2"), dec.get("outcome_qualifier"))
        chk(f"O9 (c0,c1,m0,m1)={k}: (PARTIAL_one_seed, adopts_DAG2, outcome_qualifier) == {want}", got == want and type(got[0]) is bool and type(got[1]) is bool, got)
        wa, wb = (k[2] == 1 and k[3] == 1 and not (k[0] == 1 and k[1] == 1)), (k[2] + k[3] == 1)
        fl = dec["flags"]
        chk(f"O11 {k}: PARTIAL clause a (R-MOVE at both seeds, R-CRAWL fails somewhere) == {wa}, clause b (R-MOVE at exactly one seed) == {wb}", fl["PARTIAL_clause_a"] == wa and fl["PARTIAL_clause_b"] == wb, fl)
    for base in ((1, 1, 1, 1), (0, 0, 1, 0), (1, 0, 1, 1), (0, 0, 0, 0)):
        g, lay = flags4(*base, M)
        g[0]["R-ADE"] = row(0.3, True)
        dec, _ = M.compose(g, lay, DIRS)
        chk(f"O10 HARMS on base {base}: DAG stays the base (adopts_DAG2 False), no PARTIAL_one_seed, no qualifier",
            dec["OUTCOME"] == "HARMS" and dec["adopts_DAG2"] is False and dec["PARTIAL_one_seed"] is False and dec["outcome_qualifier"] is None, dec)
    g, lay = flags4(0, 0, 1, 0, M)
    dec, _ = M.compose(g, lay, DIRS)
    cond, nxt = M.table_row_2("PARTIAL")
    chk("O12 a one-seed R-MOVE with R-CRAWL failing there is PARTIAL (one seed), carries PARTIAL's condition and committed next, and adopts nothing",
        dec["OUTCOME"] == "PARTIAL" and dec["outcome_qualifier"] == "one seed" and dec["condition"] == cond and dec["committed_next"] == nxt and dec["adopts_DAG2"] is False, dec)
    # ---- U: the defensive UNPLACED path.  No real cell reaches it now, so it is exercised by DROPPING a row (PARTIAL) from the outcome names: a PARTIAL cell must then come back UNPLACED, rc 3, no committed next
    names0 = M.OUTCOME_NAMES_2
    try:
        M.OUTCOME_NAMES_2 = tuple(n for n in names0 if n != "PARTIAL")
        g, lay = flags4(1, 0, 1, 1, M)
        dec, _ = M.compose(g, lay, DIRS)
        chk("U1 a cell whose row is missing from the table comes back UNPLACED: no outcome word, no matching row, no condition, no committed next",
            dec["OUTCOME"] == "UNPLACED" and dec["matching_rows_in_table_order"] == [] and dec["committed_next"] is None and dec["condition"] is None, dec)
        chk("U2 UNPLACED exits with rc 3", M.outcome_rc(dec) == 3, M.outcome_rc(dec))
        chk("U3 UNPLACED adopts nothing and carries no qualifier", dec["adopts_DAG2"] is False and dec["PARTIAL_one_seed"] is False and dec["outcome_qualifier"] is None, dec)
    finally:
        M.OUTCOME_NAMES_2 = names0
    chk("U4 every placed outcome exits 0", all(M.outcome_rc({"OUTCOME": o}) == 0 for o in ("FIXED", "PARTIAL", "NO EFFECT", "HARMS")))
    # ---- W: what write_out prints and writes
    for k, wq, wad in (((0, 0, 1, 0), "one seed", False), ((1, 1, 1, 1), None, True), ((1, 0, 1, 1), None, True)):
        g, lay = flags4(*k, M)
        dec, _ = M.compose(g, lay, DIRS)
        outp = os.path.join(ST, "w1_out.json")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc_w = M.write_out(types.SimpleNamespace(out=outp), {"OUTCOME": dec}, M.outcome_rc(dec))
        head = buf.getvalue().split("\nwrote ")[0]
        try:
            pj, fj = json.loads(head), json.load(open(outp, encoding="utf-8"))["OUTCOME"]
            okw = (pj["outcome_qualifier"] == wq and pj["adopts_DAG2"] is wad and pj["OUTCOME"] == dec["OUTCOME"] and fj["outcome_qualifier"] == wq and fj["adopts_DAG2"] is wad
                   and fj["PARTIAL_one_seed"] is (wq is not None) and rc_w == 0)
        except (ValueError, KeyError) as e:
            okw = False
            head = f"{type(e).__name__}: {e}: {head[:100]}"
        chk(f"W1 write_out {k}: prints and writes outcome_qualifier={wq!r}, adopts_DAG2={wad}, rc 0", okw, head[:160])
    # ---- E: the PREREG's own enumeration (enumerate_cells.py, blob of 1446333d)
    sha_e, ns_e, rc_e, out_e = guarded(load_enumeration, "E0") or (None, {}, None, "")
    new_, old_ = ns_e.get("new", {}), ns_e.get("old", {})
    chk("E1 inputs/enumerate_cells.py is the pinned blob of 1446333d (sha256)", sha_e == ENUM_SHA, sha_e)
    chk("E2 enumerate_cells.py exits 0: no placed cell moves and no cell is unplaced under the amended partition", rc_e == 0, (rc_e, out_e[-200:]))
    chk("E3 the hand-written EXPECT16 equals the enumeration's amended partition on all 16 cells", len(new_) == 16 and set(new_) == set(EXPECT16) and all([new_[k]] == v for k, v in EXPECT16.items()), new_)
    chk("E4 amended counts: FIXED 1 / PARTIAL 11 / NO EFFECT 4, nothing unplaced", dict(Counter(new_.values())) == {"FIXED": 1, "PARTIAL": 11, "NO EFFECT": 4}, dict(Counter(new_.values())))
    chk("E5 as landed: FIXED 1 / PARTIAL 7 / NO EFFECT 4 / unplaced 4, and the unplaced cells are exactly NEWLY_PLACED",
        dict(Counter(old_.values())) == {"FIXED": 1, "PARTIAL": 7, "NO EFFECT": 4, None: 4} and sorted(k for k, v in old_.items() if v is None) == sorted(NEWLY_PLACED), dict(Counter(old_.values())))
    got16 = {}
    for k in EXPECT16:
        g, lay = flags4(*k, M)
        got16[k] = guarded(lambda: M.compose(g, lay, DIRS)[0]["OUTCOME"], f"E6:{k}")
    chk("E6 the reader's outcome equals the enumeration's amended outcome on every one of the 16 cells", len(new_) == 16 and all(got16[k] == new_[k] for k in EXPECT16), got16)
    # ---- B: the fixed-RF bars
    bars = M.rf_bars(DR.bar_for)
    chk("B1 bars == the PREREG's printed bars", all(abs(bars[h][s] - M.BARS_PREREG[h][s]) < 1e-12 for h in bars for s in M.SEEDS), bars)
    chk("B2 RF constants are the PREREG's", M.RF_FIXED == {"lat": {0: 0.0067, 1: 0.0043}, "lon": {0: 0.0251, 1: 0.0139}})
    chk("B3 bars literal 0.0248 / 0.0248 / 0.0502 / 0.0278", (bars["lat"][0], bars["lat"][1], bars["lon"][0], bars["lon"][1]) == (0.0248, 0.0248, 0.0502, 0.0278))
    chk("B4 clears is inclusive at the bar", M.clears_bar(row(0.0502, True), 0.0502) and not M.clears_bar(row(0.0501, True), 0.0502))
    chk("B5 clears needs separation", not M.clears_bar(row(0.5, False), 0.0502))
    chk("B6 a negative delta never clears", not M.clears_bar(row(-0.5, True), 0.0502))
    # ---- C: the crawl set per seed
    chk("C1 crawl set s0", tuple(M.CRAWL_SET[0]) == ("curvature MAE", "L2 curvature MAE (GT>=5.2)", "M1 crawl share"), M.CRAWL_SET[0])
    chk("C2 crawl set s1", tuple(M.CRAWL_SET[1]) == ("curvature MAE", "L2 curvature MAE (GT>=5.2)", "M1 crawl share", "yaw-rate MAE"), M.CRAWL_SET[1])
    for s in (0, 1):
        der = sorted(l for l, r in BANKED[f"s{s}"].items() if r.get("separated_worse") and l not in M.EXEMPT_NEWLY)
        chk(f"C3 s{s}: the crawl set is the banked DAG - P10 separated-worse rows minus accel / DK-VIOL (MEASURED, banked READ_A.json)", sorted(M.CRAWL_SET[s]) == der, der)
    for s in (0, 1):
        p10, mv = table(), table()
        f = M.crawl_flags(s, p10, mv)
        chk(f"C4 s{s}: all rows ns -> R-CRAWL true, R-MOVE false", f["R-CRAWL"] and not f["R-MOVE"])
        for l in M.CRAWL_SET[s]:
            f = M.crawl_flags(s, table({l: WORSE}), table({x: BETTER for x in M.CRAWL_SET[s]}))
            chk(f"C5 s{s}: {l} separated-worse for DAG2 - P10 breaks R-CRAWL only", (not f["R-CRAWL"]) and f["R-MOVE"])
            f = M.crawl_flags(s, table(), table({x: BETTER for x in M.CRAWL_SET[s] if x != l}))
            chk(f"C6 s{s}: {l} not better for DAG2 - DAG2-0 breaks R-MOVE only", f["R-CRAWL"] and not f["R-MOVE"])
            f = M.crawl_flags(s, table({l: BETTER}), table({x: BETTER for x in M.CRAWL_SET[s]}))
            chk(f"C7 s{s}: {l} separated-BETTER vs P10 holds R-CRAWL", f["R-CRAWL"] and f["R-MOVE"])
    f0 = M.crawl_flags(0, table({"yaw-rate MAE": WORSE, "accel MAE": WORSE}), table({x: BETTER for x in M.CRAWL_SET[0]}))
    chk("C8 s0: yaw-rate and accel are NOT in C_0 (a worse yaw-rate / accel does not break s0's R-CRAWL)", f0["R-CRAWL"])
    f1 = M.crawl_flags(1, table({"yaw-rate MAE": WORSE}), table({x: BETTER for x in M.CRAWL_SET[1]}))
    chk("C9 s1: yaw-rate IS in C_1", not f1["R-CRAWL"])
    # ---- G: the guards
    gf = M.guard_flags(row(0.1, True), row(0.1, True), row(-0.1, True), 0.0)
    chk("G1 all guards held", all(gf.values()), gf)
    for nm, args in (("R-TLAT", (row(-0.1, True), row(0.1, True), row(0, False), 0.0)), ("R-TLON", (row(0.1, True), row(-0.1, True), row(0, False), 0.0)),
                     ("R-ADE", (row(0.1, True), row(0.1, True), row(0.1, True), 0.0)), ("R-T0", (row(0.1, True), row(0.1, True), row(0, False), 0.0605))):
        gf = M.guard_flags(*args)
        chk(f"G2 {nm} alone fails", (not gf[nm]) and all(v for k, v in gf.items() if k != nm), gf)
    gf = M.guard_flags(row(-0.1, False), row(-0.1, False), row(0.1, False), 0.0)
    chk("G3 not separated -> held", all(gf.values()))
    # ---- N: the newly-worse clause
    dag = table({"curvature MAE": WORSE, "accel MAE": WORSE})
    nw = M.newly_worse(dag, table({"curvature MAE": WORSE, "accel MAE": WORSE}), DIRS)
    chk("N1 rows worse for DAG and still worse for DAG2 are NOT new", nw["held"] and nw["violations"] == [], nw)
    nw = M.newly_worse(table(), table({"heading MAE": WORSE}), DIRS)
    chk("N2 a row ns for DAG and worse for DAG2 IS new", (not nw["held"]) and nw["violations"] == ["heading MAE"], nw)
    nw = M.newly_worse(table(), table({"accel MAE": WORSE, "DK-VIOL": WORSE}), DIRS)
    chk("N3 accel and DK-VIOL are exempt, held, and reported", nw["held"] and sorted(nw["exempt_but_newly_worse_reported"]) == ["DK-VIOL", "accel MAE"] and nw["violations"] == [], nw)
    nw = M.newly_worse(table(), table({"DK-HW": WORSE}), DIRS)
    chk("N4 DK-HW is NOT exempt", nw["violations"] == ["DK-HW"], nw)
    nw = M.newly_worse(table(), table({"S-POS@50": WORSE}), DIRS)
    chk("N5 a STRATEGIC row newly worse counts", nw["violations"] == ["S-POS@50"], nw)
    nw = M.newly_worse(table(), table({"TAC-LAT": row(-0.2, True)}), DIRS)
    chk("N6 an agreement row (higher is better) newly worse (negative delta) counts", nw["violations"] == ["TAC-LAT"], nw)
    nw = M.newly_worse(table(), table({"TAC-LAT": row(0.2, True)}), DIRS)
    chk("N7 an agreement row with a POSITIVE separated delta is better, not worse", nw["held"], nw)
    nw = M.newly_worse(table(), table({"ADE6": BETTER}), DIRS)
    chk("N8 a better row never violates", nw["held"])
    t2 = table()
    t2["S-POS@30"] = {"family": "STRATEGIC", "status": "NOT COMPUTED"}
    nw = M.newly_worse(table(), t2, DIRS)
    chk("N9 a NOT COMPUTED row cannot be judged and is never silently passing", (not nw["held"]) and nw["not_judged"] == ["S-POS@30"], nw)
    nw = M.newly_worse(table({"curvature MAE": WORSE}), table({"curvature MAE": row(0.6, False)}), DIRS)
    chk("N10 worse for DAG, ns for DAG2: not new", nw["held"])
    # ---- L: the two layer-G substitutions
    seen = []

    def orig(tag, arm, seed, r):
        seen.append(r["flags"].get("--init-ckpt"))
        return True, "orig"
    stub = types.SimpleNamespace(g_argv=orig, one_value=DR.one_value)
    ga = M.make_g_argv(stub)
    fl = lambda p: {"--init-ckpt": [[p]], "--seed": [["0"]]}
    r = {"flags": fl("D:\\tanitad\\runs\\v8f_mini_full\\M2b_s0_dag\\ckpt_last.pt")}
    res = ga("t", "DAG", 0, r)
    chk("L1 init DAG_s0 accepted; the reader's own clause sees the M2b-shaped tail; the record is not mutated", res == (True, "orig") and seen[-1] == [["M2b_s0/ckpt_last.pt"]] and r["flags"]["--init-ckpt"] == [["D:\\tanitad\\runs\\v8f_mini_full\\M2b_s0_dag\\ckpt_last.pt"]], seen)
    n0 = len(seen)
    res = ga("t", "DAG", 0, {"flags": fl("D:\\x\\M2b_s0\\ckpt_last.pt")})
    chk("L2 an M2b init (iteration 1) is refused and the reader is not asked", res[0] is False and len(seen) == n0, res)
    res = ga("t", "DAG", 0, {"flags": fl("D:\\x\\M2b_s1_dag\\ckpt_last.pt")})
    chk("L3 the other seed's DAG is refused", res[0] is False and len(seen) == n0, res)
    res = ga("t", "DAG", 0, {"flags": {"--seed": [["0"]]}})
    chk("L4 no --init-ckpt is refused", res[0] is False and len(seen) == n0, res)
    gr = M.make_g_replicate(stub)
    chk("L5 the ORC0 slot IS DAG2-0's directory -> ok", gr(0, {"dir": "E:/a/M2b_s0_dag20"}, {"dir": "E:\\a\\M2b_s0_dag20"}, "k")[0] is True)
    chk("L6 an ORC0 slot that is another directory is refused", gr(0, {"dir": "E:/a/M2b_s0_dag20"}, {"dir": "E:/a/M2b_s0_orc0"}, "k")[0] is False)
    chk("L7 coverage minimum", M.COVERAGE_MIN == 0.5)
    return R


def real_layer_g_checks(M):
    """The patched run-identity clause on REAL records: DAG2 s0 accepted; the ITERATION-1 DAG run (init M2b) refused."""
    R = []
    e = {"train_dir": f"{ART}/v8f-dagger2/M2b_s0_dag2"}
    tr = DR.train_records(e, "init_ckpt.sha256")
    ga = M.make_g_argv(DR)
    ok, det = ga("DAG_s0", "DAG", 0, tr)
    R.append(("L8 REAL: the pulled DAG2 s0 record passes the patched ARGV control", ok, det))
    e1 = {"train_dir": f"{ART}/v8f-dagger/M2b_s0_dag"}
    tr1 = DR.train_records(e1, "init_ckpt.sha256")
    ok1, det1 = ga("DAG_s0", "DAG", 0, tr1)
    R.append(("L9 REAL: the iteration-1 DAG s0 record (init M2b) is REFUSED by the patched ARGV control", ok1 is False, det1))
    return R


# ---------------------------------------------------------------------------------------------------------------- P: pullcheck
def write_run(art, arm, seed, edit_argv=None, edit_sum=None, extra_sums=None, listed_override=None):
    """A tiny COMPLETE pulled run built from the REAL records of the pulled s0 pair (small json files); big files are stand-ins."""
    import hashlib
    real = f"{ART}/v8f-dagger2/M2b_s0_{arm}"
    d = os.path.join(art, f"M2b_s{seed}_{arm}")
    os.makedirs(d, exist_ok=True)
    tg = f"M2b-{arm}_s{seed}"
    files = {"ckpt_last.pt": b"ckpt-standin-" + arm.encode(), f"eval_{tg}.npz": b"npz-standin", "train_log.jsonl": b'{"step": 1}\n'}
    for nm in ("selfcheck.json", "summary.json", "argv.json", "launch_gate_token.json"):
        files[nm] = open(os.path.join(real, nm), "rb").read()
    files[f"eval_{tg}.json"] = b'{"standin": "synthetic T0 eval; the real eval_M2b-dag2* files are never opened (blinding)"}'      # BLINDING: the real eval JSON is NOT copied
    aj = json.loads(files["argv.json"])
    sj = json.loads(files["summary.json"])
    aj["argv"][aj["argv"].index("--seed") + 1] = str(seed)
    aj["argv"][aj["argv"].index("--out") + 1] = f"D:\\tanitad\\runs\\v8f_mini_full\\M2b_s{seed}_{arm}"
    ic = aj["argv"].index("--init-ckpt")
    aj["argv"][ic + 1] = f"D:\\tanitad\\runs\\v8f_mini_full\\M2b_s{seed}_dag\\ckpt_last.pt"
    if "--dagger-bank" in aj["argv"]:
        aj["argv"][aj["argv"].index("--dagger-bank") + 1] = f"D:\\tanitad\\runs\\v8f_mini_full\\dagger_bank_dag2\\s{seed}"
        aj["argv"][aj["argv"].index("--dagger-seed") + 1] = str(seed)
    sj["seed"] = seed
    if edit_argv:
        edit_argv(aj["argv"])
    if edit_sum:
        edit_sum(sj)
    files["argv.json"] = json.dumps(aj).encode()
    files["summary.json"] = json.dumps(sj).encode()
    lines = []
    for nm, b in files.items():
        open(os.path.join(d, nm), "wb").write(b)
        lines.append(f"{hashlib.sha256(b).hexdigest()} *{nm}")
    for nm, b in (extra_sums or {}).items():
        open(os.path.join(d, nm), "wb").write(b)
        lines.append(f"{hashlib.sha256(b).hexdigest()} *{nm}")
    sp = os.path.join(d, "SHA256SUMS.txt")
    open(sp, "w", newline="\n").write("\n".join(lines) + "\n")
    os.utime(sp, (time.time() - 100, time.time() - 100))
    return d


def pullcheck_checks(P):
    R = []
    art = os.path.join(ST, "art_pc")
    shutil.rmtree(art, ignore_errors=True) if False else None            # never delete: fixtures are overwritten in place
    os.makedirs(art, exist_ok=True)

    def st(arm, seed=0):
        return P.check(art, arm, seed)[0]
    write_run(art, "dag2", 0)
    write_run(art, "dag20", 0)
    R.append(("P1 a COMPLETE dag2 run", st("dag2") == "COMPLETE", P.check(art, "dag2", 0)[1]))
    R.append(("P2 a COMPLETE dag20 run", st("dag20") == "COMPLETE", P.check(art, "dag20", 0)[1]))

    def wrong(label, arm, seed, **kw):
        write_run(art, arm, seed, **kw)
        s, why, _ = P.check(art, arm, seed)
        R.append((f"P {label} -> WRONG", s == "WRONG", f"{s}: {why[:140]}"))

    def drop(argv, flag, n=1):
        i = argv.index(flag)
        del argv[i:i + 1 + n]

    def set_(flag, v):
        return lambda argv: argv.__setitem__(argv.index(flag) + 1, v)
    wrong("dag2 whose init is M2b (an iteration-1 run)", "dag2", 0, edit_argv=set_("--init-ckpt", "D:\\x\\M2b_s0\\ckpt_last.pt"))
    wrong("dag2 whose init is the other seed's DAG", "dag2", 0, edit_argv=set_("--init-ckpt", "D:\\x\\M2b_s1_dag\\ckpt_last.pt"))
    wrong("dag2 fed the ITERATION-1 bank", "dag2", 0, edit_argv=set_("--dagger-bank", "D:\\x\\dagger_bank\\s0"))
    wrong("dag2 with --dagger-seed of the other seed", "dag2", 0, edit_argv=set_("--dagger-seed", "1"))
    wrong("dag2 naming --dagger-p 0 (a DAG2-0 run in the dag2 slot)", "dag2", 0, edit_argv=set_("--dagger-p", "0"))
    wrong("dag2 without --dagger-bank-sha256", "dag2", 0, edit_argv=lambda a: drop(a, "--dagger-bank-sha256"))
    wrong("dag20 naming a bank", "dag20", 0, edit_argv=lambda a: a.extend(["--dagger-bank", "D:\\x\\dagger_bank_dag2\\s0"]))
    wrong("dag20 naming --dagger-p 0.1", "dag20", 0, edit_argv=set_("--dagger-p", "0.1"))
    wrong("dag20 naming --dagger-p 0.1 with a CONSISTENT summary (only the arm clause can catch it)", "dag20", 0, edit_argv=set_("--dagger-p", "0.1"), edit_sum=lambda s: s.setdefault("levers", {}).__setitem__("dagger_p", 0.1))
    wrong("dag2 naming --dagger-p 0 with a CONSISTENT summary (only the arm clause can catch it)", "dag2", 0, edit_argv=set_("--dagger-p", "0"), edit_sum=lambda s: s.setdefault("levers", {}).__setitem__("dagger_p", 0.0))
    wrong("dag2 with the 10-name SUMS (a box-2 fresh eval listed, decision O1)", "dag2", 0, extra_sums={"eval_M2b-dag2_s0-fresh.json": b"{}"})
    wrong("a seed-1 dir whose argv says --seed 0", "dag2", 1, edit_argv=set_("--seed", "0"))
    # a dag20 directory holding DAG2's records
    write_run(art, "dag2", 0)
    d2, d20 = os.path.join(art, "M2b_s0_dag2"), os.path.join(art, "M2b_s0_dag20")
    for nm in os.listdir(d20):
        os.remove(os.path.join(d20, nm)) if False else None
    bad = os.path.join(art, "M2b_s5_dag20")
    shutil.copytree(d2, bad, dirs_exist_ok=True)
    s, why, _ = P.check(art, "dag20", 5)
    R.append(("P a dag20 directory holding DAG2's run -> WRONG (a STOP, never a wait)", s == "WRONG", f"{s}: {why[:140]}"))
    write_run(art, "dag2", 0)
    if REAL_PULLS:       # BLINDING: this stream-hashes the pulled eval_M2b-dag2*_s0.{json,npz}; off by default
        real = P.check(f"{ART}/v8f-dagger2", "dag2", 0)
        R.append(("P REAL pulled M2b_s0_dag2 -> COMPLETE (MEASURED)", real[0] == "COMPLETE", real[1][:100]))
        real = P.check(f"{ART}/v8f-dagger2", "dag20", 0)
        R.append(("P REAL pulled M2b_s0_dag20 -> COMPLETE (MEASURED)", real[0] == "COMPLETE", real[1][:100]))
    rb = P.check_bank_status(f"{ART}/v8f-dagger2", 0, expect_shards=4)
    R.append(("P REAL iteration-2 bank s0 -> COMPLETE and == the gate's declared sha (MEASURED)", rb[0] == "COMPLETE", rb[1][:100]))
    R.append(("P REAL iteration-2 bank s1 -> COMPLETE and == the gate's declared sha (MEASURED)", P.check_bank_status(f"{ART}/v8f-dagger2", 1, expect_shards=4)[0] == "COMPLETE", ""))
    R.append(("P arms are exactly dag2 / dag20", tuple(P.ARMS) == ("dag2", "dag20")))
    R.append(("P the pin read is the DAG2 gate's", P.declared_bank_sha(f"{ART}/v8f-dagger2", 0)[0] == "OK" and P.declared_bank_sha(f"{ART}/v8f-dagger2", 0)[1] == "90c43c66f52f250d41ff23e837d5fb31d1592cf2aad14afacd7ded011734b589"))
    return [(x[0], x[1], x[2] if len(x) > 2 else "") for x in R]


# ---------------------------------------------------------------------------------------------------------------- CH: the chain
FIX_ART = ST + "/art_empty"
os.makedirs(FIX_ART, exist_ok=True)


def dry(script, env=None, timeout=600):
    e = dict(os.environ)
    e.update({"DAG2T1_TEST_CFREE_MB": "3000", "DAG2T1_TEST_EFREE_MB": "3000", "TMP": f"{H2}/_tmp", "TEMP": f"{H2}/_tmp"})
    e.update(env or {})
    p = subprocess.run([BASH, script.replace(chr(92), "/"), "--dry-run"], capture_output=True, text=True, env=e, timeout=timeout, cwd=HERE)
    return p.returncode, p.stdout + p.stderr


def tags_in_order(out, lane):
    import re
    return [m.group(1) for ln in out.splitlines() if f"[lane {lane}]" in ln for m in [re.search(r"--tag (\S+)", ln)] if m]


def chain_scenarios():
    """name -> (env, predicate(rc, out) -> (ok, why)); the first is the REAL configuration, the rest run on an empty fixture ART."""
    S = {}
    fx = {"DAG2T1_ART": FIX_ART}

    def base_ok(rc, out):
        return rc == 0 and "ZZDAG2T1-DRY-OK" in out, f"rc {rc}"
    S["real"] = ({"DAG2T1_B2_OUT": "C:/Users/Admin/box2_ship/logs/chain_v8f_dag2.out"}, lambda rc, out: (
        base_ok(rc, out)[0]
        and tags_in_order(out, "causal") == ["DAG2_s0_m10_causal_heads", "DAG20_s0_m10_causal_heads", "DAG2_s1_m10_causal_heads", "DAG20_s1_m10_causal_heads"]
        and tags_in_order(out, "logged") == ["DAG2_s0_m10_logged_heads", "DAG20_s0_m10_logged_heads", "DAG2_s1_m10_logged_heads", "DAG20_s1_m10_logged_heads"]
        and f"--ckpt {ART}/v8f-dagger2/M2b_s0_dag2/ckpt_last.pt" in out and f"--ckpt {ART}/v8f-dagger2/M2b_s1_dag20/ckpt_last.pt" in out
        and f"--out-dir {ART}/v8f-dagger2-t1" in out and f"--dump-ticks {ART}/v8f-dagger2-t1/ticks" in out
        and "M2b-dag2_s0-fresh" in out and "M2b-dag20_s1-fresh" in out and "ZZDAG2T1-DONE" in out and "DAG2_READ.json" in out
        and "--replan-steps 10 --nav deadreckon --scene logged --schedule periodic" in out and "--ages fresh" in out and f"{ART}/dagt1_tree_b43c7a9a/stack/scripts/v8f_t1_eval.py" in out
        and out.count("--device cpu") >= 8 and "M2b_s0_dag2  COMPLETE" in out, "real configuration facts"))
    S["order"] = (fx, lambda rc, out: (base_ok(rc, out)[0] and tags_in_order(out, "causal") == ["DAG2_s0_m10_causal_heads", "DAG20_s0_m10_causal_heads", "DAG2_s1_m10_causal_heads", "DAG20_s1_m10_causal_heads"]
                                       and tags_in_order(out, "logged")[0] == "DAG2_s0_m10_logged_heads", "run order"))
    import re
    OUT2 = f"{ART}/v8f-dagger2-t1"
    outs = lambda out: set(re.findall(r"--out-dir (\S+)", out))
    S["paths"] = (fx, lambda rc, out: (base_ok(rc, out)[0] and outs(out) == {OUT2, OUT2 + "/t0fresh"} and f"--ckpt {FIX_ART}/M2b_s0_dag2/ckpt_last.pt" in out
                                       and f"--ckpt {FIX_ART}/M2b_s1_dag20/ckpt_last.pt" in out, f"OUT / ART paths: {sorted(outs(out))}"))
    S["out_not_dag"] = (fx, lambda rc, out: (base_ok(rc, out)[0] and bool(outs(out)) and not any(o.startswith(f"{ART}/v8f-dagger-t1") for o in outs(out)) and all(o.startswith(OUT2) for o in outs(out)),
                                             "DAG2 never writes into DAG's output directory"))
    S["c1500"] = ({**fx, "DAG2T1_TEST_CFREE_MB": "1500"}, lambda rc, out: (rc != 0 and "ZZDAG2T1-STOP" in out and "C: has 1500 MB free" in out, f"rc {rc}"))
    S["c2047"] = ({**fx, "DAG2T1_TEST_CFREE_MB": "2047"}, lambda rc, out: (rc != 0 and "ZZDAG2T1-STOP" in out, f"rc {rc}"))
    S["c2048"] = ({**fx, "DAG2T1_TEST_CFREE_MB": "2048"}, lambda rc, out: (base_ok(rc, out)[0] and "C: gate: 2048 MB free >= 2048 MB" in out, f"rc {rc}"))
    S["e1499"] = ({**fx, "DAG2T1_TEST_EFREE_MB": "1499"}, lambda rc, out: (rc != 0 and "ZZDAG2T1-STOP" in out and "E: has 1499 MB free" in out, f"rc {rc}"))
    S["badtree"] = ({**fx, "DAG2T1_TREE": f"{ART}/dag2t1/_st"}, lambda rc, out: (rc != 0 and "ZZDAG2T1-STOP" in out, f"rc {rc}"))
    S["b2fresh"] = ({**fx, "DAG2T1_B2_FRESH": "1"}, lambda rc, out: (rc != 0 and "ZZDAG2T1-STOP" in out and "REMOVED" in out, f"rc {rc}"))
    tamper = os.path.join(ST, "PREREG_V8F_DAGGER2_tampered.md")
    open(tamper, "wb").write(open(REAL.DEFAULTS["prereg2"], "rb").read().replace(b"FIXED", b"FIXEd", 1))
    S["prereg_tamper"] = ({**fx, "DAG2T1_PREREG2": tamper.replace("\\", "/")}, lambda rc, out: (rc != 0 and "ZZDAG2T1-STOP" in out and "is not the pinned text" in out, f"rc {rc}"))   # the CHAIN's own check (dag2_read's static check pins the same file a second time)
    if not REAL_PULLS:
        S.pop("real")        # BLINDING: the real-configuration dry run pull-checks (stream-hashes) the pulled eval_M2b-dag2*_s0.{json,npz}
    return S


def static_chain_ok(path):
    """BLIND-mode stand-in for the parts of `CH real` that only pin the chain's DEFAULT paths (a dry run on the real configuration would hash the pulled DAG2 eval files): the default ART, tool tree
    and PREREG pin are the literal text below."""
    t = open(path, encoding="utf-8", newline="").read()
    want = ["ART=${DAG2T1_ART:-/e/Projects/TanitAD-artifacts/v8f-dagger2};", "TT=${DAG2T1_TREE:-E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a}",
            "PREREG2=${DAG2T1_PREREG2:-$Hm/inputs/PREREG_V8F_DAGGER2.md}; PREREG2_SHA=c4239e4590ee26e30ce6f525987aca82e3fb8ff034b7264321e2d0faa724f1c6"]
    miss = [w for w in want if w not in t]
    return not miss, miss or "default ART / tool tree / PREREG pin literal"


def chain_checks(script):
    R = []
    for nm, (env, pred) in chain_scenarios().items():
        rc, out = dry(script, env)
        r = pred(rc, out)
        R.append((f"CH {nm}", bool(r[0]), r[1] if len(r) > 1 else "") if isinstance(r, tuple) else (f"CH {nm}", bool(r), ""))
    ok_, det_ = static_chain_ok(script)
    R.append(("CH static defaults", ok_, det_))
    return R


# ---------------------------------------------------------------------------------------------------------------- M: mutants
READ_MUTANTS = [   # (id, anchor, replacement, what)
    ("M01", '0: ("curvature MAE", "L2 curvature MAE (GT>=5.2)", "M1 crawl share"),', '0: ("curvature MAE", "L2 curvature MAE (GT>=5.2)", "M1 crawl share", "yaw-rate MAE"),', "s0's C also holds yaw-rate"),
    ("M02", '"M1 crawl share", "yaw-rate MAE")}', '"M1 crawl share")}', "s1's C drops yaw-rate"),
    ("M03", '0: ("curvature MAE", "L2 curvature MAE (GT>=5.2)", "M1 crawl share"),', '0: ("curvature MAE", "M1 crawl share"),', "s0's C drops L2"),
    ("M04", '"lon": {0: 0.0251, 1: 0.0139}}   #', '"lon": {0: 0.0125, 1: 0.0139}}   #', "RF lon s0 halved"),
    ("M05", "HALF_F = 0.0604 ", "HALF_F = 0.0640 ", "R-T0 bar moved"),
    ("M06", 'EXEMPT_NEWLY = ("accel MAE", "DK-VIOL")', 'EXEMPT_NEWLY = ("DK-VIOL",)', "accel no longer exempt"),
    ("M07", 'EXEMPT_NEWLY = ("accel MAE", "DK-VIOL")', 'EXEMPT_NEWLY = ("accel MAE", "DK-VIOL", "DK-HW")', "DK-HW exempt"),
    ("M08", "crawl = {l: not sep_worse(rows_vs_p10[l]) for l in c}", "crawl = {l: sep_better(rows_vs_p10[l]) for l in c}", "R-CRAWL demands separated-better"),
    ("M09", "move = {l: sep_better(rows_vs_dag20[l]) for l in c}", "move = {l: not sep_worse(rows_vs_dag20[l]) for l in c}", "R-MOVE accepts not-worse"),
    ("M10", '"R-TLAT": not sep_worse(tlat, "higher")', '"R-TLAT": not sep_worse(tlat, "lower")', "R-TLAT direction flipped"),
    ("M11", '"R-TLON": not sep_worse(tlon, "higher")', '"R-TLON": True', "R-TLON guard dropped"),
    ("M12", '"R-ADE": not sep_worse(ade, "lower")', '"R-ADE": not sep_better(ade, "lower")', "R-ADE guard reads better"),
    ("M13", '"R-T0": bool(t0_point <= HALF_F)', '"R-T0": bool(t0_point < HALF_F)', "R-T0 bar strict"),
    ("M14", "if now_worse and not was_worse:", "if now_worse:", "newly-worse counts rows that were already worse"),
    ("M15", "if now_worse and not was_worse:", "if now_worse and not was_worse and lab not in SPOS_ROWS:", "newly-worse skips STRATEGIC"),
    ("M16", "(exempt_hit if lab in EXEMPT_NEWLY else viol).append(lab)", "viol.append(lab)", "no exemption"),
    ("M17", "fixed = held and all(both.values())", "fixed = held and any(both.values())", "FIXED needs one seed"),
    ("M18", "partial_a = held and all(mv.values()) and not all(cr.values())", "partial_a = held and any(mv.values()) and not all(cr.values())", "PARTIAL(a) with one seed's move"),
    ("M19", "partial_b = held and sum(mv.values()) == 1", "partial_b = held and sum(mv.values()) >= 1", "PARTIAL(b) R-MOVE at least one seed (== 1 -> >= 1)"),
    ("M20", "no_effect = held and not any(mv.values())", "no_effect = held and not all(mv.values())", "NO EFFECT on one seed's failure"),
    ("M21", '"HARMS": not held}', '"HARMS": False}', "guards ignored"),
    ("M22", 'out = matching[0] if matching else "UNPLACED"', 'out = matching[0] if matching else "PARTIAL"', "a table gap silently becomes PARTIAL"),
    ("M23", "held = all(all(guards[s].values()) for s in SEEDS)", "held = all(all(guards[s].values()) for s in (0,))", "seed 1's guards ignored"),
    ("M24", 'return {hd: {s: bar_for(hd, RF_FIXED[hd][s]) for s in SEEDS} for hd in ("lat", "lon")}', 'return {hd: {s: bar_for(hd, 0.0) for s in SEEDS} for hd in ("lat", "lon")}', "RF ignored in the bars"),
    ("M25", 'return bool(sep_better(row, "higher") and row["delta"] >= bar)', 'return bool(sep_better(row, "higher") and row["delta"] > bar)', "bar strict"),
    ("M26", 'return bool(row["separated"] and ((row["delta"] > 0) if direction == "lower" else (row["delta"] < 0)))', 'return bool(row["separated"] and ((row["delta"] < 0) if direction == "lower" else (row["delta"] > 0)))', "sep_worse flipped"),
    ("M27", 'return {"OUTCOME": "BLOCKED"', 'return {"OUTCOME": "REFUSED"', "BLOCKED reads REFUSED"),
    ("M28", "COVERAGE_MIN = 0.5", "COVERAGE_MIN = 0.4", "coverage minimum lowered"),
    ("M29", 'want_tail = f"M2b_s{seed}_dag/ckpt_last.pt"', 'want_tail = f"M2b_s{seed}/ckpt_last.pt"', "init tail is the iteration-1 one"),
    ("M30", "if not tail_norm(v).endswith(want_tail):", "if False:", "init tail check removed"),
    ("M31", 'if tail_norm(ro["dir"]) != tail_norm(r0["dir"]):', "if False:", "ORC0 slot may be any directory"),
    ("M32", 'lay_rows["DAG2_minus_DAG20"][s])\n        seed_flags[s] = sf', 'lay_rows["DAG2_minus_P10"][s])\n        seed_flags[s] = sf', "R-MOVE reads DAG2 - P10"),
    ("M33", 'newly_worse(lay_rows["DAG_minus_P10"][s], lay_rows["DAG2_minus_P10"][s], directions)', 'newly_worse(lay_rows["DAG2_minus_DAG20"][s], lay_rows["DAG2_minus_P10"][s], directions)', "newly-worse baseline is DAG2-0"),
    ("M34", 'PREREG2_SHA256 = "c4239e45', 'PREREG2_SHA256 = "d4239e45', "PREREG pin changed"),
    ("M35", '"adopt DAG2 as the base. Read A is re-run', '"adopt DAG as the base. Read A is re-run', "FIXED's committed next edited"),
    ("M36", 'gf["NEWLY-WORSE"] = nw[s]["held"]', 'gf["NEWLY-WORSE"] = True', "newly-worse clause not a guard"),
    ("M37", 'crawl_flags(s, lay_rows["DAG2_minus_P10"][s]', 'crawl_flags(s, lay_rows["DAG_minus_P10"][s]', "R-CRAWL reads DAG - P10"),
    # ---- Amendment 1 (A1-2026-10-27)
    ("M38", "partial_b = held and sum(mv.values()) == 1", "partial_b = held and sum(both.values()) == 1", "PARTIAL(b) as landed: R-CRAWL and R-MOVE at one seed only (4 cells unplaced again)"),
    ("M39", 'adopts = out == "FIXED" or (out == "PARTIAL" and all(mv.values()))', 'adopts = out in ("FIXED", "PARTIAL")', "adopts_DAG2 ignores the both-seeds condition (a one-seed PARTIAL adopts)"),
    ("M40", 'adopts = out == "FIXED" or (out == "PARTIAL" and all(mv.values()))', 'adopts = out == "PARTIAL" and all(mv.values())', "FIXED does not adopt"),
    ("M41", 'adopts = out == "FIXED" or (out == "PARTIAL" and all(mv.values()))', 'adopts = out == "FIXED" or (out == "PARTIAL" and any(mv.values()))', "adopts_DAG2 on R-MOVE at ONE seed (all -> any)"),
    ("M42", 'partial_one_seed = out == "PARTIAL" and sum(mv.values()) == 1', "partial_one_seed = sum(mv.values()) == 1", "PARTIAL_one_seed without the PARTIAL condition"),
    ("M43", 'partial_one_seed = out == "PARTIAL" and sum(mv.values()) == 1', 'partial_one_seed = out == "PARTIAL" and sum(mv.values()) >= 1', "PARTIAL_one_seed for a both-seeds PARTIAL (== 1 -> >= 1)"),
    ("M44", 'qualifier = "one seed" if partial_one_seed else None', 'qualifier = "one seed" if out == "PARTIAL" else None', "outcome_qualifier on every PARTIAL"),
    ("M45", 'qualifier = "one seed" if partial_one_seed else None', "qualifier = None", "outcome_qualifier never set"),
    ("M46", 'return {"UNPLACED": 3}.get(dec["OUTCOME"], 0)', 'return {"UNPLACED": 0}.get(dec["OUTCOME"], 0)', "UNPLACED exits 0 instead of 3"),
    ("M47", '"R-MOVE at both seeds but R-CRAWL fails somewhere; or R-MOVE at one seed only; guards held"', '"R-MOVE at both seeds but R-CRAWL fails somewhere; or R-CRAWL and R-MOVE at one seed only; guards held"', "PARTIAL condition as landed"),
    ("M48", "\u00a77 on the base's states", "\u00a77 on DAG2 states", "PARTIAL committed next as landed (on DAG2 states)"),
    ("M49", "for ln in t.split(AMENDED_TABLE_HEADING, 1)[1].splitlines():", "for ln in t.splitlines():", "the table is read from the FIRST (as-landed) table in the file"),
    ("M50", "if AMENDED_TABLE_HEADING not in t:", "if False:", "a PREREG without the amendment heading is not refused"),
]
PC_MUTANTS = [
    ("P01", 'if ic is None or not norm(ic).endswith(f"M2b_s{seed}_dag/ckpt_last.pt"):', 'if ic is None or not norm(ic).endswith(f"M2b_s{seed}/ckpt_last.pt"):', "init tail reverted to M2b"),
    ("P02", 'if bk is not None and not norm(bk).rstrip("/").endswith(f"dagger_bank_dag2/s{seed}"):', "if False:", "iteration-1 bank accepted"),
    ("P03", '    elif arm == "dag20" and pv != 0.0:', '    elif False:', "dag20 p-check removed"),
    ("P04", 'pin = os.path.join(art, run_name("dag2", seed), "gate", "pin.txt")', 'pin = os.path.join(art, run_name("dag20", seed), "gate", "pin.txt")', "pin read from the dag20 gate"),
    ("P05", 'ARMS = ("dag2", "dag20")', 'ARMS = ("dag", "dag0")', "arm names are DAG's"),
    ("P06", 'if one(fl, "--dagger-seed") != str(seed):', "if False:", "--dagger-seed unchecked"),
]
CH_MUTANTS = [
    ("C01", "RUNS=(dag2:0 dag20:0 dag2:1 dag20:1)", "RUNS=(dag2:0 dag2:1 dag20:0 dag20:1)", "run order swapped", ["order"]),
    ("C02", "OUT=${DAG2T1_OUT:-/e/Projects/TanitAD-artifacts/v8f-dagger2-t1}", "OUT=${DAG2T1_OUT:-/e/Projects/TanitAD-artifacts/v8f-dagger-t1}", "OUT is DAG's output directory", ["out_not_dag", "paths"]),
    ("C03", "ART=${DAG2T1_ART:-/e/Projects/TanitAD-artifacts/v8f-dagger2}", "ART=${DAG2T1_ART:-/e/Projects/TanitAD-artifacts/v8f-dagger}", "ART is DAG's pull directory", ["real", "defaults"]),
    ("C04", '[ "$CF0" -ge "$CMIN" ] ||', "true ||", "C: gate removed", ["c1500"]),
    ("C05", '[ "$EF0" -ge "$EMIN" ] ||', "true ||", "E: gate removed", ["e1499"]),
    ("C06", 'armU() { if [ "$1" = dag2 ]; then echo DAG2; else echo DAG20; fi; }', 'armU() { if [ "$1" = dag2 ]; then echo DAG; else echo DAG0; fi; }', "rollout tags are DAG's", ["real", "order"]),
    ("C07", "TT=${DAG2T1_TREE:-E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a}", "TT=${DAG2T1_TREE:-E:/Projects/TanitAD-artifacts/di/tree}", "another tool tree", ["real", "order"]),
    ("C08", '[ "$(sha256sum < "$PREREG2" | cut -c1-64)" = "$PREREG2_SHA" ] ||', "true ||", "PREREG pin check removed", ["prereg_tamper"]),
    ("C09", "[ -z \"${DAG2T1_B2_FRESH:-}\" ] ||", "true ||", "the removed switch accepted", ["b2fresh"]),
    ("C10", "CMIN=${DAG2T1_C_MIN_MB:-2048}", "CMIN=${DAG2T1_C_MIN_MB:-1024}", "C: gate lowered", ["c1500"]),
    ("C11", "PREREG2=${DAG2T1_PREREG2:-$Hm/inputs/PREREG_V8F_DAGGER2.md};", "PREREG2=${DAG2T1_PREREG2:-/e/Projects/TanitAD-artifacts/dagland/PREREG_V8F_DAGGER2.md};", "PREREG default is the as-landed (1d4f4fbb) dagland copy", ["order", "defaults"]),
    ("C12", "PREREG2_SHA=c4239e4590ee26e30ce6f525987aca82e3fb8ff034b7264321e2d0faa724f1c6", "PREREG2_SHA=87d6c77dc3362bfb65af1e8966c913b957143f88d9c04b3867576f4bba423e39", "PREREG pin is the as-landed sha", ["order", "defaults"]),
]


def mutate(src_path, anchor, repl, out_path):
    t = open(src_path, encoding="utf-8", newline="").read()
    n = t.count(anchor)
    if n != 1:
        return n
    open(out_path, "w", encoding="utf-8", newline="").write(t.replace(anchor, repl))
    return 1


def main():
    t0 = time.time()
    allr, summary = [], []

    def run_section(name, fn):
        t1 = time.time()
        try:
            r = fn()
        except Exception as e:      # noqa: BLE001
            r = [(f"{name} section crashed", False, f"{type(e).__name__}: {e}")]
        r = [(x[0], x[1], x[2] if len(x) > 2 else "") for x in r]
        for n, ok, why in r:
            print(("PASS " if ok else "FAIL ") + n + ("" if ok else f"   <<< {str(why)[:200]}"))
        allr.extend((name, n, ok) for n, ok, why in r)
        print(f"--- {name}: {sum(ok for _, ok, _ in r)}/{len(r)} ({time.time() - t1:.0f} s)")
        return r
    print("BLINDING: " + ("OFF (--real-pulls): the pulled eval_M2b-dag2* files are stream-hashed" if REAL_PULLS else
                          "ON: no real eval_M2b-dag2* file is opened, read or hashed; NOT RUN: `P REAL pulled M2b_s0_dag2`, `P REAL pulled M2b_s0_dag20`, `CH real`; instead `CH static defaults`"))
    base_logic = run_section("LOGIC (T/O/U/W/E/B/C/G/N/L)", lambda: logic_checks(REAL) + real_layer_g_checks(REAL))
    PCM = load(os.path.join(HERE, "dag2t1_pullcheck.py"), "dag2t1_pullcheck_real")
    base_pc = run_section("PULLCHECK (P)", lambda: pullcheck_checks(PCM))
    script = os.path.join(HERE, "dag2_t1_chain.sh")
    base_ch = [] if QUICK else run_section("CHAIN (CH)", lambda: chain_checks(script))
    # ---- mutation proof
    print("\n=== MUTATION PROOF: each mutant is one defect in a copy; killed iff a check goes RED ===")
    results = []
    ctrl_ok = all(ok for _, ok, _ in base_logic) and all(ok for _, ok, _ in base_pc) and all(ok for _, ok, _ in base_ch)
    print(f"control (unmutated): {'GREEN' if ctrl_ok else 'RED -- the mutation proof is void'}")
    src = os.path.join(HERE, "dag2_read.py")
    # copy controls: the SAME module text loaded from _mut/ (where every mutant lives) must be fully green, or a mutant could be "killed" by its LOCATION (a path relative to the module) and not by its defect
    shutil.copyfile(src, os.path.join(MUT, "dag2_read_mut.py"))
    cc_logic = [n_ for n_, ok, _ in logic_checks(load(os.path.join(MUT, "dag2_read_mut.py"), "dag2_read_copy_ctrl")) if not ok]
    shutil.copyfile(os.path.join(HERE, "dag2t1_pullcheck.py"), os.path.join(MUT, "dag2t1_pullcheck_mut.py"))
    cc_pc = [n_ for n_, ok, _ in pullcheck_checks(load(os.path.join(MUT, "dag2t1_pullcheck_mut.py"), "dag2t1_pullcheck_copy_ctrl")) if not ok]
    ctrl_ok = ctrl_ok and not cc_logic and not cc_pc
    print(f"control (unmutated COPIES in _mut/, where every mutant lives): {'GREEN' if not cc_logic and not cc_pc else 'RED ' + str(cc_logic[:3] + cc_pc[:3])}")
    for mid, anchor, repl, what in READ_MUTANTS:
        mp = os.path.join(MUT, "dag2_read_mut.py")
        n = mutate(src, anchor, repl, mp)
        if n != 1:
            results.append((mid, what, "ANCHOR ERROR", f"anchor found {n}x"))
            print(f"{mid} ANCHOR ERROR ({n}x): {what}")
            continue
        try:
            M = load(mp, f"dag2_read_mut_{mid}")
            red = [n_ for n_, ok, _ in logic_checks(M) if not ok]
        except Exception as e:      # noqa: BLE001
            red = [f"raised {type(e).__name__}: {e}"]
        results.append((mid, what, "KILLED" if red else "SURVIVED", red[:3]))
        print(f"{mid} {'KILLED  ' if red else 'SURVIVED'} {what}   [{len(red)} red; first: {str(red[0])[:90] if red else '-'}]")
    pcsrc = os.path.join(HERE, "dag2t1_pullcheck.py")
    for mid, anchor, repl, what in PC_MUTANTS:
        mp = os.path.join(MUT, "dag2t1_pullcheck_mut.py")
        n = mutate(pcsrc, anchor, repl, mp)
        if n != 1:
            results.append((mid, what, "ANCHOR ERROR", f"anchor found {n}x"))
            print(f"{mid} ANCHOR ERROR ({n}x): {what}")
            continue
        try:
            P = load(mp, f"dag2t1_pullcheck_mut_{mid}")
            red = [n_ for n_, ok, _ in pullcheck_checks(P) if not ok]
        except Exception as e:      # noqa: BLE001
            red = [f"raised {type(e).__name__}: {e}"]
        results.append((mid, what, "KILLED" if red else "SURVIVED", red[:3]))
        print(f"{mid} {'KILLED  ' if red else 'SURVIVED'} {what}   [{len(red)} red; first: {str(red[0])[:90] if red else '-'}]")
    if not QUICK:
        sc = chain_scenarios()
        for mid, anchor, repl, what, scen in CH_MUTANTS:
            mp = os.path.join(HERE, "_mutant_chain.sh")
            n = mutate(script, anchor, repl, mp)
            if n != 1:
                results.append((mid, what, "ANCHOR ERROR", f"anchor found {n}x"))
                print(f"{mid} ANCHOR ERROR ({n}x): {what}")
                continue
            red = []
            for s in scen:
                if s == "defaults":
                    if not static_chain_ok(mp)[0]:
                        red.append("static defaults red")
                    continue
                if s not in sc:
                    continue         # `real` is not run in blind mode
                env, pred = sc[s]
                rc, out = dry(mp, env)
                r = pred(rc, out)
                if not r[0]:
                    red.append(f"scenario {s} red")
            results.append((mid, what, "KILLED" if red else "SURVIVED", red))
            print(f"{mid} {'KILLED  ' if red else 'SURVIVED'} {what}   [{red}]")
    killed = sum(1 for r in results if r[2] == "KILLED")
    bad = [r for r in results if r[2] != "KILLED"]
    nfail = sum(1 for _, _, ok in allr if not ok)
    print(f"\nSELFTEST: {len(allr) - nfail}/{len(allr)} checks as expected; MUTANTS: {killed}/{len(results)} killed; not killed: {[(r[0], r[2]) for r in bad]}; control {'green' if ctrl_ok else 'RED'}; {time.time() - t0:.0f} s")
    json.dump({"checks": len(allr), "checks_failed": nfail, "mutants": [{"id": r[0], "what": r[1], "result": r[2], "red": r[3]} for r in results], "control_green": ctrl_ok, "real_pulls_run": REAL_PULLS,
               "not_run_when_blind": [] if REAL_PULLS else ["P REAL pulled M2b_s0_dag2", "P REAL pulled M2b_s0_dag20", "CH real"]}, open(os.path.join(HERE, "mutation.json"), "w"), indent=1)
    return 0 if (nfail == 0 and not bad and ctrl_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
