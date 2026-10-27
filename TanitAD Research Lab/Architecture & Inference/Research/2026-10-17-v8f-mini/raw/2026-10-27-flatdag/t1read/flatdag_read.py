#!/usr/bin/env python3
"""flatdag_read.py -- the dev-box T1-kin read of PREREG_V8F_FLATDAG (claim row H-V8F-FLATDAG-1).  BUILT AND SELF-TESTED, NOT RUN ON REAL FLAT-DAG OUTPUTS.

PREREG (binding, nothing here changes it): Project Steering/PREREG_V8F_FLATDAG.md @ 9b91b654, raw-blob sha256 PREREG_FLATDAG_SHA256 below.  Its section 2 rows and section 4 outcome table are checked CELL BY CELL against the
pinned text (static check).  It inherits Read A's rows and C1-C4 (PREREG_V8F_DAG_REREAD section 2 and 4) and DAGGER's R-ADE / R-T0 guards, K-B0..K-B5 and COVERAGE (PREREG_V8F_DAGGER), each pinned by raw sha256 too.

NOTHING IS RE-DERIVED.  Every number comes from a function of an existing reader, run on these arms:
  DAG       the adopted base, BANKED (v8f-dagger-t1/ dumps and ticks; read, never re-run)
  FLAT-DAG  FLAT_s<S> + 3,000 steps, --o-interface cut, --dagger-p 0.1 on FLAT's OWN bank       (reader slot DAG  of dag_read)
  FLAT-DAG-0  the same argv with --dagger-p 0                                                   (reader slot DAG0 of dag_read)
  FLAT      BANKED (dagreread/packed/dumps_FLAT_*): beside, and the C1 reproduction
  layer G  dag_read (pinned): the bank controls K-B0..K-B5 + COVERAGE (C-F1), the run-record controls (C-F2), the T1-HEADS pipeline (Pack: R-TLAT / R-TLON, S-POS vs CTRV, K1 K2 K3 K5 K6 K7 of each run).
           One single-purpose substitution, checked and never skipped: g_argv (the init is FLAT_s<S>, so `--init-ckpt` must end `M2b_s<S>_flat/ckpt_last.pt`; the reader's own check then runs with that tail presented as
           the M2b tail it is written for) plus `--o-interface cut` named exactly once and FLAT's every other flag.
  layer A  Read A's driver code over the pinned reader sources: comps / L1 / L2 / M1 rows (readA.rows_block), TAC + DK rows (t1kf_read_a1.prep / rows), the Read A table (build_report.collect), as dag2_read.ReadA.
The only new logic is the PREREG's composition, and it is the part the self-test mutates: PRIMARY H-T1, the family rows, the build check, and the outcome partition (decide()).

THE DECISION (section 4, applied in order): REFUSED (any control fails) -> BLOCKED (the build check fails, or COVERAGE < 0.5 at >= 1 seed) -> FLAT BETTER (ADE6 DAG - FLAT-DAG separated-worse at >= 1 seed)
  -> HIERARCHY BEATS FLAT (PRIMARY H-T1 at both seeds AND no family row separated-worse) -> NOT ESTABLISHED.
READING CHOICES the PREREG leaves open (reported in the output, flagged where they matter; the Master Mind rules before any row is seen):
  (a) "Read A's rows of DAG_REREAD section 2" = the rows Read A's table holds: ADE6 + the four families' rows + M1 (the crawl share, defined in section 2 though not in its table).  The table-only reading (M1 out) is computed
      too; `ambiguity.differs` says whether the two readings give different outcomes.
  (b) STRATEGIC S-POS rows are against CTRV WITHIN a run (no paired DAG - FLAT-DAG instrument exists; the registered B-2 read did not compute them either).  They are NOT COMPUTED as paired family rows, are listed with
      that reason, and never count as passing; per-arm S-POS vs CTRV is reported (the PREREG's S-POS row).
  (c) The T1-HEADS K4 (tick 0 == the arm's own fresh-age T0) needs a logged-bucket rollout and a fresh T0 eval that PREREG section 5 does not schedule and section 3 does not require: K4 is reported NOT RUN, and the
      H-T1-HEADS rows are marked with the K controls that DID run.

    python flatdag_read.py --manifest runs.json --out FLATDAG_READ.json --tool-root <tree> ... (see build_parser)      rc 0 outcome written | 2 REFUSED (nothing written) | 3 UNPLACED (defensive) | 4 BLOCKED (written)
    python flatdag_read.py --static-check ...        pins, PREREG cells and static inputs only; touches no run output
    python flatdag_read.py --preflight ...           the bank and run-record controls and the build check on whatever FLAT-DAG runs are pulled (no eval_* and no rollout output is opened)
    python flatdag_read.py --repro-only ...          layer A only, on BANKED data: DAG - FLAT recomputed and compared with the banked B-2 table (control C1) and the identity (C2)
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
H2 = HERE.replace("\\", "/")
ART = "E:/Projects/TanitAD-artifacts"
# ---------------------------------------------------------------------------------------------------------------- PREREG_V8F_FLATDAG (committed numbers)
PREREG_FLATDAG_ID = "Project Steering/PREREG_V8F_FLATDAG.md @ 9b91b654"
PREREG_FLATDAG_SHA256 = "8713d0a38d8dfe1283f03e3865421c8f066c1700a8339fd9d59bdd5868810c46"
PREREG_REREAD_SHA256 = "7acaa4dd25373d554d9d2cdc68d0c9b94687167e343cceec0e8cf0939d7657fa"       # PREREG_V8F_DAG_REREAD.md @ 9b91b654 (with Amendment 1)
PREREG_DAGGER_SHA256 = "c99a881e0d010af8604a21dfb949491ff2aaeab8d60cc7077c4777e5197b2945"       # PREREG_V8F_DAGGER.md @ 9b91b654 (with Amendments 1, 2)
D2_READ_SHA256 = "1425e8a3895c442b9841f3bcd6fd0893021d583e06485fde171b04bfa73fbc20"                                                            # dag2_read.py as built for PREREG_V8F_DAGGER2 (imported unchanged)
CLAIM_ROW = "H-V8F-FLATDAG-1"
SEEDS = (0, 1)
F_M, TWO_F, HALF_F = 0.1207, 0.2415, 0.0604       # F (MODEL_REGISTRY section 17), 2F = the printed bar, F/2 = DAGGER's R-T0 guard
N_BOOT = 2000
COVERAGE_MIN = 0.5
SWAP_P, SWAP_TOL = 0.1, 0.02                       # PREREG section 1: the swap reached, share 0.1 +- 0.02
FLAT_CKPT_SHA = {0: "8496934a2ce262c128a3e975da156de2f40e422925893c9979169a4c114493cc", 1: "6fdfc9b02ce87248c5315dd7604d3642a3b39f7ae0f593ec6b6ef5ef8df64c95"}
DAG_CKPT_SHA = {0: "d186fd0bcfa7ee9fbfe5c9de543fe85edda7c3e6b9c56c898958f027ae2b8cb1", 1: "1d12ec2eb825990d57dcc45dda50b5178d1d733961b47ab0e3d50e6155fa39ef"}
SLOT_LABEL = {"DAG": "FLATDAG", "DAG0": "FLATDAG0"}            # reader slot -> rollout tag stem
RECIPE_COMMON = ("--init-ckpt", "--steps", "--lr", "--warmup", "--p-gt-start", "--p-gt-end", "--ego-perturb-p", "--dagger-p")
RECIPE_FLATDAG = ("--dagger-bank", "--dagger-bank-sha256", "--dagger-seed")
ROLLOUT_PATH_FLAGS = ("--ckpt", "--tag", "--out-dir", "--dump-ticks")      # the only flags in which a new rollout's argv may differ from FLAT's banked one
SPOS_KEY = "S-POS vs CTRV (ticks>=1, within this run)"
SPOS_ROWS = ("S-POS@30", "S-POS@50", "S-POS@80")

# The PREREG's section 2 rows and section 4 table AS PRINTED (markdown bold markers are markup, not text); a control checks every cell against the pinned file.
ROWS_2 = (
    ("PRIMARY H-T1", "ADE6, DAG − FLAT-DAG", "≤ −2F = −0.2415 m, separated, at both seeds"),
    ("families", "Read A's rows of `PREREG_V8F_DAG_REREAD` §2, DAG − FLAT-DAG", "none separated-worse, for the claim to read \"beats\""),
    ("H-T1-HEADS", "R-TLAT / R-TLON, DAG − FLAT-DAG (T's heads exist in both; FLAT's O does not read them)", "beside"),
    ("D-FLAT", "ADE6, FLAT-DAG − FLAT-DAG-0: does DAgger help flat too?", "beside, with DAGGER's R-ADE / R-T0 guards reported"),
    ("S-POS", "against CTRV, both arms", "beside"),
)
OUTCOME_TABLE = (
    ("REFUSED", "any control fails", "fix and re-run"),
    ("BLOCKED", "the build check fails, or COVERAGE < 0.5 at ≥ 1 seed", "the failing check is named; for COVERAGE, the rejoin-target form (DAGGER §7)"),
    ("FLAT BETTER", "ADE6 DAG − FLAT-DAG separated-worse at ≥ 1 seed", "report; the hierarchy costs at T1-kin; the PI decides the architecture"),
    ("HIERARCHY BEATS FLAT", "PRIMARY H-T1 at both seeds, and no family row separated-worse", "the first hierarchy-over-flat claim of v8F, at T1-kin: register it and signal the PI"),
    ("NOT ESTABLISHED", "otherwise", "B-2's margin was DAgger's, not the hierarchy's. The interface stays the defect; next is MIXO's committed T-side arm, under its own PREREG"),
)
OUTCOME_NAMES = tuple(r[0] for r in OUTCOME_TABLE)
CONTROL_IDS_3 = ("C-F1", "C-F2", "C-F3", "C-F4")
UNPLACED_TEXT = ("UNPLACED: this reading matches no row of the PREREG's outcome table. NOT ESTABLISHED is 'otherwise', so this path is unreachable through real cells: it means a row of the table is "
                 "missing from this reader, or the PREREG changed: a reader / PREREG gap, not a result. No outcome word is invented.")


# ---------------------------------------------------------------------------------------------------------------- pure logic (no numpy, no torch): this is what the self-test mutates
def sep_worse(row, direction="lower"):
    """separated AND the point on the wrong side.  'lower' = an error row (positive delta is worse); 'higher' = an agreement / F1 row (negative delta is worse)."""
    return bool(row["separated"] and ((row["delta"] > 0) if direction == "lower" else (row["delta"] < 0)))


def sep_better(row, direction="lower"):
    return bool(row["separated"] and ((row["delta"] < 0) if direction == "lower" else (row["delta"] > 0)))


def primary_met(row):
    """PRIMARY H-T1 at one seed: ADE6 DAG - FLAT-DAG <= -2F (inclusive), separated."""
    return bool(row["separated"] and row["delta"] <= -TWO_F)


def primary_state(row):
    """MET / SHORT (separated-better but above -2F) / WORSE (separated-worse) / NS (not separated)."""
    if primary_met(row):
        return "MET"
    if not row["separated"]:
        return "NS"
    return "WORSE" if row["delta"] > 0 else "SHORT"


def family_rows(table, include_crawl=True):
    """The Read A table rows that count as 'family rows' for the claim: every computed row of the table; M1 (family 'crawl') in or out by the reading (a)."""
    return {lab: r for lab, r in table.items() if r.get("status") != "NOT COMPUTED" and (include_crawl or r.get("family") != "crawl")}


def family_flags(table, directions, include_crawl=True):
    """-> (separated-worse rows, NOT COMPUTED rows with their reason) of one seed's Read A table for DAG - FLAT-DAG."""
    worse = sorted(lab for lab, r in family_rows(table, include_crawl).items() if sep_worse(r, directions.get(lab, "lower")))
    notc = sorted(lab for lab, r in table.items() if r.get("status") == "NOT COMPUTED")
    return worse, notc


def build_check(arm, summary, selfcheck, p_declared):
    """PREREG section 1: the trainer's cut and swap, as the run's OWN records show them.  -> [(check name, ok, detail)].  A failing check is BLOCKED, with the check named."""
    out = []
    lv = summary.get("levers") if isinstance(summary.get("levers"), dict) else {}
    slv = selfcheck.get("levers") if isinstance(selfcheck.get("levers"), dict) else {}
    built = lv.get("o_interface", slv.get("o_interface"))
    out.append(("BUILD-CUT-LEVER", built == "cut", f"built o_interface {built!r} (summary.levers / selfcheck.levers), declared cut"))
    live = selfcheck.get("live") if isinstance(selfcheck.get("live"), dict) else {}
    dead, decl = live.get("dead_params"), live.get("declared_dead")
    ok = bool(live.get("pass") is True and dead == ["e_code.weight"] and decl == ["e_code.weight"] and not live.get("declared_dead_but_alive") and not live.get("unexpected_dead"))
    out.append(("BUILD-CUT-GRAD", ok, {"dead_params": dead, "declared_dead": decl, "declared_dead_but_alive": live.get("declared_dead_but_alive"), "unexpected_dead": live.get("unexpected_dead"), "live_pass": live.get("pass"),
                                        "meaning": "zero gradient on e_code.weight (O reads a constant), everything else alive"}))
    pr = selfcheck.get("o_interface_probe") if isinstance(selfcheck.get("o_interface_probe"), dict) else {}
    perts = pr.get("perturbations") if isinstance(pr.get("perturbations"), dict) else {}
    inv = bool(perts) and all(isinstance(v, dict) and v.get("max_abs_dtraj60") == 0.0 and v.get("o_read_code_abs_max") == 0.0 for v in perts.values())
    out.append(("BUILD-CUT-INVARIANT", bool(pr.get("pass") is True and pr.get("o_interface") == "cut" and inv),
                {"probe_pass": pr.get("pass"), "o_interface": pr.get("o_interface"), "perturbations": sorted(perts), "all_traj60_and_code_invariant": inv, "meaning": "O invariant to S and T"}))
    n, e = summary.get("dagger_n"), summary.get("dagger_eligible")
    if p_declared == 0.0:
        ok = isinstance(n, int) and not isinstance(n, bool) and n == 0
        out.append(("BUILD-SWAP", ok, {"dagger_n": n, "dagger_eligible": e, "must": "exactly 0 at p = 0"}))
    else:
        ok = bool(isinstance(n, int) and isinstance(e, int) and e > 0 and 0 <= n <= e and p_declared - SWAP_TOL - 1e-12 <= n / e <= p_declared + SWAP_TOL + 1e-12)
        out.append(("BUILD-SWAP", ok, {"dagger_n": n, "dagger_eligible": e, "share": (n / e) if ok else None, "must": f"{p_declared} +- {SWAP_TOL}"}))
    return out


def decide(controls_failed, blocked, primary, fam_worse):
    """Section 4, applied in order.  controls_failed: list of failed control ids; blocked: list of BLOCKED reasons (build check / COVERAGE); primary[s]: the ADE6 DAG - FLAT-DAG row; fam_worse[s]: separated-worse
    family rows.  Returns the outcome with EVERY matching row in table order (the first wins)."""
    if tuple(sorted(primary)) != SEEDS or tuple(sorted(fam_worse)) != SEEDS:
        raise ValueError(f"decide needs exactly seeds {SEEDS}")
    st = {s: primary_state(primary[s]) for s in SEEDS}
    worse_any = any(st[s] == "WORSE" for s in SEEDS)
    met_both = all(st[s] == "MET" for s in SEEDS)
    fam_none = not any(fam_worse[s] for s in SEEDS)
    match = {"REFUSED": bool(controls_failed), "BLOCKED": bool(blocked), "FLAT BETTER": worse_any, "HIERARCHY BEATS FLAT": met_both and fam_none, "NOT ESTABLISHED": True}
    matching = [n for n in OUTCOME_NAMES if match[n]]
    out = matching[0] if matching else "UNPLACED"
    return {"OUTCOME": out, "matching_rows_in_table_order": matching,
            "flags": {"primary_state": {f"s{s}": st[s] for s in SEEDS}, "ADE6_separated_worse_at_ge1_seed": worse_any, "PRIMARY_met_at_both_seeds": met_both, "no_family_row_separated_worse": fam_none,
                      "family_rows_separated_worse": {f"s{s}": list(fam_worse[s]) for s in SEEDS}, "controls_failed": list(controls_failed), "blocked": list(blocked)}}


def outcome_block(d):
    out = d["OUTCOME"]
    if out == "UNPLACED":
        return {**d, "condition": None, "committed_next": None, "note": UNPLACED_TEXT}
    cond, nxt = next((c, n) for nm, c, n in OUTCOME_TABLE if nm == out)
    return {**d, "condition": cond, "committed_next": nxt}


def outcome_rc(dec):
    return {"UNPLACED": 3, "BLOCKED": 4, "REFUSED": 2}.get(dec["OUTCOME"], 0)


def guards_beside(r_ade, r_t0_point):
    """D-FLAT's beside guards, DAGGER's definitions applied to FLAT-DAG - FLAT-DAG-0: R-ADE not separated-worse; R-T0 point <= +F/2 (inclusive)."""
    return {"R-ADE": not sep_worse(r_ade, "lower"), "R-T0": bool(r_t0_point <= HALF_F)}


def argv_flags(argv):
    out, i = {}, 0
    while i < len(argv):
        t = str(argv[i])
        if t.startswith("--"):
            j, vals = i + 1, []
            while j < len(argv) and not str(argv[j]).startswith("--"):
                vals.append(str(argv[j]))
                j += 1
            out.setdefault(t, []).append(vals)
            i = j
        else:
            i += 1
    return out


def flat_argv_problems(arm_flags, flat_flags, with_bank):
    """C-F2, 'every other flag is FLAT's argv, and FLAT's `--o-interface cut` is kept': the flags FLAT-DAG names beyond FLAT's are EXACTLY the recipe, nothing FLAT names is missing, every shared flag but --out is equal."""
    bad = []
    want = set(RECIPE_COMMON) | (set(RECIPE_FLATDAG) if with_bank else set())
    extra, missing = set(arm_flags) - set(flat_flags), set(flat_flags) - set(arm_flags)
    if extra != want:
        bad.append(f"flags beyond FLAT's argv are {sorted(extra)}, the recipe is {sorted(want)}")
    if missing:
        bad.append(f"FLAT names {sorted(missing)} and this argv does not")
    diff = sorted(k for k in set(arm_flags) & set(flat_flags) if k != "--out" and arm_flags[k] != flat_flags[k])
    if diff:
        bad.append(f"shared flags with other values than FLAT's: {diff}")
    ic = arm_flags.get("--o-interface")
    if ic != [["cut"]]:
        bad.append(f"--o-interface is {ic}, FLAT's cut is kept and NAMED exactly once")
    return bad


def rollout_argv_problems(new_tokens, banked_tokens):
    """Control C-ARGV: the new rollout's argv equals FLAT's banked B-2 rollout argv except in --ckpt, --tag, --out-dir, --dump-ticks (token by token)."""
    if len(new_tokens) != len(banked_tokens):
        return [f"{len(new_tokens)} tokens, FLAT's banked rollout has {len(banked_tokens)}"]
    bad = []
    for i, (x, y) in enumerate(zip(new_tokens, banked_tokens)):
        if x != y:
            flag = new_tokens[i - 1] if i else None
            if flag not in ROLLOUT_PATH_FLAGS:
                bad.append(f"token {i} after {flag!r}: {x!r} vs FLAT's {y!r}")
    return bad


def oracle_identity_problems(np, z_own, z_orc, js_orc, PB):
    """C-F3's identity row: the GT-code oracle minus own ADE6 reads EXACTLY 0 [0, 0] on the eval windows, per window and on the predictions, on an oracle that DID change the code offered to the interface."""
    bad = []
    m = (js_orc or {}).get("tac_manipulation") or {}
    if m.get("mode") != "oracle" or not (isinstance(m.get("frac_code_changed"), (int, float)) and m["frac_code_changed"] > 0.0):
        bad.append(f"the oracle eval json is not an oracle that changed the code offered to the interface (mode {m.get('mode')!r}, frac_code_changed {m.get('frac_code_changed')}): a zero read on it would say nothing")
    if not np.array_equal(np.asarray(z_own["eid"]), np.asarray(z_orc["eid"])):
        bad.append("the oracle eval is not on the own eval's grid (eid differs)")
        return bad, None
    a_o, a_n = np.asarray(z_orc["ade6"], np.float64), np.asarray(z_own["ade6"], np.float64)
    worst = float(np.abs(a_o - a_n).max())
    worst_pred = float(np.abs(np.asarray(z_orc["pred"], np.float64) - np.asarray(z_own["pred"], np.float64)).max())
    r = PB(a_o, a_n, np.asarray(z_own["eid"]), n_boot=N_BOOT, seed=0)
    if not (np.array_equal(a_o, a_n) and worst_pred == 0.0 and r["delta"] == 0.0 and r["lo"] == 0.0 and r["hi"] == 0.0):
        bad.append(f"oracle - own ADE6 reads {r['delta']} [{r['lo']}, {r['hi']}], max per-window |d ade6| {worst:.3g}, max |d pred| {worst_pred:.3g}: it must be EXACTLY 0 (O still reads something above it, or the eval did not build the cut)")
    return bad, {"delta": r["delta"], "lo": r["lo"], "hi": r["hi"], "max_abs_dade6": worst, "max_abs_dpred": worst_pred, "frac_code_changed": m.get("frac_code_changed")}


def rows_diff(a, b, tol=1e-9):
    """Two row dicts {label: row} (the shapes of readA.rows_block / t1kf.rows): same labels, same separated flag, delta / lo / hi (and the means, n) within tol.  Labels that hold no row (metadata, "_M2") are skipped.
    -> (differences, max |diff|)."""
    bad, mx = [], 0.0
    is_row = lambda r: isinstance(r, dict) and "delta" in r      # noqa: E731
    for lab in sorted(set(a) | set(b)):
        ra, rb = a.get(lab), b.get(lab)
        if not is_row(ra) and not is_row(rb):
            continue
        if not (is_row(ra) and is_row(rb)):
            bad.append(f"{lab}: present on one side only")
            continue
        for k in ("delta", "lo", "hi", "mean_a", "mean_b"):
            if k in ra and k in rb and ra[k] is not None and rb[k] is not None:
                d = abs(float(ra[k]) - float(rb[k]))
                mx = max(mx, d)
                if d > tol:
                    bad.append(f"{lab}.{k}: {ra[k]} vs {rb[k]}")
        for k in ("n_windows", "n_episodes"):
            if k in ra and k in rb and ra[k] != rb[k]:
                bad.append(f"{lab}.{k}: {ra[k]} vs {rb[k]}")
        if bool(ra.get("separated")) != bool(rb.get("separated")):
            bad.append(f"{lab}.separated: {ra.get('separated')} vs {rb.get('separated')}")
    return bad, mx


# ---------------------------------------------------------------------------------------------------------------- the PREREG text
def table_after(text, heading):
    """The rows of the first markdown table after ``heading`` (markdown bold stripped), header and separator rows dropped."""
    t = text.replace("**", "")
    if t.count(heading) != 1:
        return None
    rows = []
    for ln in t.split(heading, 1)[1].splitlines():
        ln = ln.strip()
        if ln.startswith("|"):
            rows.append(tuple(c.strip() for c in ln.strip("|").split("|")))
        elif rows:
            break
    return [r for r in rows if r[0] not in ("row", "outcome") and not set("".join(r)) <= set("-: ")]


def prereg_problems(text):
    """-> [] iff section 2's five rows and section 4's five rows equal ROWS_2 / OUTCOME_TABLE cell by cell, the section-3 controls are named, and F / 2F are the printed values."""
    probs = []
    for heading, want, label in (("## 2. Rows", ROWS_2, "section 2"), ("## 4. Outcomes, committed now", OUTCOME_TABLE, "section 4")):
        rows = table_after(text, heading)
        if rows is None:
            probs.append(f"the PREREG has no unique heading {heading!r}")
            continue
        if len(rows) != len(want):
            probs.append(f"{label}: the PREREG has {len(rows)} rows, the reader codes {len(want)}")
        for i, w in enumerate(want):
            r = rows[i] if i < len(rows) else None
            if r != tuple(w):
                probs.append(f"{label} row {w[0]}: the PREREG reads {r!r}, the reader codes {tuple(w)!r}")
    for c in CONTROL_IDS_3:
        if f"**{c}" not in text:
            probs.append(f"section 3 control {c} is not named")
    for frag in ("F = 0.1207 m", "−2F = −0.2415 m", "Applied in this order: REFUSED → BLOCKED → FLAT BETTER → HIERARCHY BEATS FLAT → NOT ESTABLISHED",
                 "Paired episode-cluster bootstrap, B = 2,000, per seed", "STRATEGIC is reported; it is not paused for v8F"):
        if frag not in text:
            probs.append(f"the PREREG no longer contains {frag!r}")
    return probs


def check_text_pin(path, want_sha, what):
    if not os.path.isfile(path):
        return False, f"{what}: {path} is missing"
    raw = open(path, "rb").read()
    got = hashlib.sha256(raw).hexdigest()
    return got == want_sha, (f"{what}: raw sha256 {got[:12]}... == the pin" if got == want_sha else f"{what}: {path} has sha256 {got[:16]}..., pinned {want_sha[:16]}...")


# ---------------------------------------------------------------------------------------------------------------- pins
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def load_d2(path):
    """dag2_read.py, imported unchanged after its pin was checked (Read A's driver class ReadA, extract_from, pin tables)."""
    if not os.path.isfile(path) or sha256_file(path) != D2_READ_SHA256:
        raise SystemExit(f"REFUSED: {path} is not the pinned dag2_read.py (sha256 {D2_READ_SHA256[:12]}...)")
    return load_module("dag2_read", path)


DEFAULTS = {
    "dag_read": "C:/Users/Admin/dagread/dag_read.py",
    "dag2_read": f"{ART}/dag2t1/dag2_read.py",
    "readers": f"{ART}/dagreread/reader_src",
    "reada_code": f"{H2}/readA_code",
    "lead_block": f"{H2}/inputs/b1_eval_lead_block_k30.npz",
    "dag_t1": f"{ART}/v8f-dagger-t1",
    "dagreread": f"{ART}/dagreread",
    "flat_dir": f"{ART}/v8f-mini-flat",
    "prereg_flatdag": f"{H2}/inputs/PREREG_V8F_FLATDAG.md",
    "prereg_reread": f"{H2}/inputs/PREREG_V8F_DAG_REREAD.md",
    "prereg_dagger": f"{H2}/inputs/PREREG_V8F_DAGGER_9b91b654.md",
    "flat_argv_dir": f"{H2}/inputs",
}
FLAT_ARGV_SHA = {0: "630d16a89a626b2e17453d45dcee1253e6b492eac43b6bf2420ffe9271b5829f", 1: "8d894b6bd27cfb93061a8a81eac90e073b78a0cc12853df196b0c81365a1ba4e"}   # FLAT_s<S>'s argv.json (the repo blobs)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for k in ("manifest", "out", "t1hd_read", "t1k_read", "eval_cache", "tracks", "eval_grid", "eval_join", "prereg", "work_dir"):
        ap.add_argument("--" + k.replace("_", "-"))
    ap.add_argument("--tool-root", default=f"{ART}/dagt1_tree_b43c7a9a")
    ap.add_argument("--init-sha", action="append", help="SEED=sha256 of the checkpoint FLAT-DAG / FLAT-DAG-0 started from (the runs' own records; must equal FLAT_s<S>)")
    for k, v in DEFAULTS.items():
        ap.add_argument("--" + k.replace("_", "-"), default=v)
    ap.add_argument("--static-check", action="store_true")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--repro-only", action="store_true")
    return ap


def static_paths(a):
    d2 = a.D2
    return {**d2.static_paths(a)}


def own_pin_problems(a):
    bad = []
    for path, sha, what in ((a.prereg_flatdag, PREREG_FLATDAG_SHA256, "PREREG_V8F_FLATDAG.md"), (a.prereg_reread, PREREG_REREAD_SHA256, "PREREG_V8F_DAG_REREAD.md"),
                            (a.prereg_dagger, PREREG_DAGGER_SHA256, "PREREG_V8F_DAGGER.md (9b91b654)")):
        ok, det = check_text_pin(path, sha, what)
        if not ok:
            bad.append(det)
    if os.path.isfile(a.prereg_flatdag):
        bad += [f"PREREG_V8F_FLATDAG: {p}" for p in prereg_problems(open(a.prereg_flatdag, encoding="utf-8").read())]
    for s in SEEDS:
        p = os.path.join(a.flat_argv_dir, f"flat_argv_s{s}.json")
        if not os.path.isfile(p) or sha256_file(p) != FLAT_ARGV_SHA[s]:
            bad.append(f"FLAT_s{s}'s argv.json at {p} is missing or not the pinned blob {FLAT_ARGV_SHA[s][:12]}...")
    return bad


# ---------------------------------------------------------------------------------------------------------------- layer G: the patched g_argv and the run-record / bank controls
def tail_norm(p):
    return str(p).replace("\\", "/")


def make_g_argv(DR):
    orig = DR.g_argv

    def g_argv(tag, arm, seed, r):
        fl = dict(r["flags"])
        ok, v = DR.one_value(fl, "--init-ckpt")
        if not ok:
            return False, ["--init-ckpt is not named exactly once"]
        want_tail = f"M2b_s{seed}_flat/ckpt_last.pt"
        if not tail_norm(v).endswith(want_tail):
            return False, [f"--init-ckpt {v} is not .../{want_tail} (PREREG_V8F_FLATDAG section 1: the init is FLAT_s{seed})"]
        fl["--init-ckpt"] = [[f"M2b_s{seed}/ckpt_last.pt"]]                  # the reader's own clause, presented the tail it is written for
        r2 = dict(r)
        r2["flags"] = fl
        return orig(tag, arm, seed, r2)
    return g_argv


def stage_banks(DR, ctl, manifest, init_sha, eval_ids, train_ids, state):
    """C-F1: every bank control (K-B0..K-B5, BANK-FILES, BANK-CONFIG, BANK-ACCOUNTING) on each seed's FLAT bank, K-B2 against FLAT_s<S>'s checkpoint; then COVERAGE (>= 0.5 or BLOCKED)."""
    banks, blocked = {}, []
    for s in SEEDS:
        be = (manifest.get("bank") or {}).get(str(s))
        if not be or not os.path.isfile(be["report"]):
            state[f"bank_s{s}"] = "not pulled"
            continue
        B = DR.load_bank(be["report"], be.get("bank_file"), what=f"bank seed {s} report")
        DR.bank_stage(ctl, s, B, init_sha.get(s, ""), eval_ids, train_ids)
        banks[s] = B
        cov = DR.coverage_of(B)
        state[f"coverage_s{s}"] = cov
        if cov["eligible_fraction"] < COVERAGE_MIN:
            blocked.append(f"COVERAGE s{s} = {cov['eligible_fraction']:.4f} < {COVERAGE_MIN}")
    return banks, blocked


def stage_runs(DR, ctl, manifest, init_sha, banks, train_ids, flat_argv, state):
    """C-F2: the run-record controls of each arm (SUMMARY, INIT-CKPT, ARGV, REACHED, GATE-TOKEN, BANK-DECLARED), FLAT's every other flag, the pair rule (PAIR, K-B3.2, TRAIN-LIST); and the BUILD CHECK."""
    ki, kw, kt = "init_ckpt.sha256", "train_window_list_sha256", "train_sha256"
    kn, ke, kp, kb = "dagger_n", "dagger_eligible", "dagger_p", "dagger_bank_sha256"
    saved = (DR.BANKED_M2B_SHA, DR.g_argv)
    DR.BANKED_M2B_SHA, DR.g_argv = dict(FLAT_CKPT_SHA), make_g_argv(DR)
    recs, build = {}, {}
    try:
        for s in SEEDS:
            ctl.add(f"M2B-CKPT:s{s}", init_sha.get(s) == FLAT_CKPT_SHA[s], {"passed": str(init_sha.get(s))[:16], f"FLAT_s{s}": FLAT_CKPT_SHA[s][:16]})
            for arm in ("DAG", "DAG0"):
                e = DR.entry(manifest, arm, s)
                tag = f"{SLOT_LABEL[arm]}_s{s}"
                if not os.path.isfile(os.path.join(e["train_dir"], "summary.json")):
                    state[tag] = "not pulled"
                    continue
                tr = DR.train_records(e, ki)
                recs[(arm, s)] = tr
                ctl.run(f"SUMMARY:{tag}", lambda tr=tr, s=s, tag=tag: DR.g_summary(tag, s, tr))
                ctl.run(f"INIT-CKPT:{tag}", lambda tr=tr, s=s, tag=tag: DR.g_init(tag, s, tr, init_sha.get(s, ""), ki))
                ctl.run(f"ARGV:{tag}", lambda tr=tr, arm=arm, s=s, tag=tag: DR.g_argv(tag, arm, s, tr))
                ctl.run(f"FLAT-ARGV:{tag}", lambda tr=tr, arm=arm, s=s: (lambda bad: (not bad, bad or "every other flag is FLAT's argv; --o-interface cut named once"))(
                    flat_argv_problems(argv_flags(tr["argv"]), flat_argv[s], arm == "DAG")))
                ctl.run(f"DAGGER-REACHED:{tag}", lambda tr=tr, arm=arm, tag=tag: DR.g_reached(tag, arm, tr, kn, ke, kp))
                ctl.run(f"GATE-TOKEN:{tag}", lambda tr=tr, e=e, tag=tag: DR.g_token(tag, tr, DR.read_token(e.get("token"))))
                if arm == "DAG" and s in banks:
                    ctl.run(f"BANK-DECLARED:{tag}", lambda tr=tr, s=s, tag=tag: DR.g_bank_declared(tag, s, banks[s], tr, init_sha.get(s, ""), ki, kb))
                build[(arm, s)] = build_check(arm, tr["summary.json"], tr["selfcheck.json"], 0.1 if arm == "DAG" else 0.0)
            if ("DAG", s) in recs and ("DAG0", s) in recs and s in banks:
                rd, r0 = recs[("DAG", s)], recs[("DAG0", s)]
                ctl.run(f"PAIR:s{s}", lambda rd=rd, r0=r0, s=s: DR.g_pair(s, rd, r0, ki))
                ctl.run(f"K-B3.2:s{s}", lambda rd=rd, r0=r0, s=s: DR.g_kb32(banks[s], rd, r0, kw))
                ctl.run(f"TRAIN-LIST:s{s}", lambda rd=rd, r0=r0, s=s: DR.g_train_list(banks[s], train_ids, rd, r0, kt))
    finally:
        DR.BANKED_M2B_SHA, DR.g_argv = saved
    blocked = [f"{nm} ({SLOT_LABEL[arm]}_s{s}): {det}" for (arm, s), chk in build.items() for nm, ok, det in chk if not ok]
    return recs, build, blocked


def load_flat_argv(a):
    return {s: argv_flags(json.load(open(os.path.join(a.flat_argv_dir, f"flat_argv_s{s}.json"), encoding="utf-8"))["argv"]) for s in SEEDS}


# ---------------------------------------------------------------------------------------------------------------- the read
def family_of(cid):
    if cid.startswith(("BANK-", "K-B", "COVERAGE")):
        return "C-F1"
    if cid.startswith(("SUMMARY", "INIT-CKPT", "ARGV", "FLAT-ARGV", "DAGGER-REACHED", "GATE-TOKEN", "BANK-DECLARED", "PAIR", "TRAIN-LIST", "M2B-CKPT")):
        return "C-F2"
    if cid.startswith(("CFG-CUT", "ORACLE-IDENTITY")):
        return "C-F3"
    return "C-F4"


def main(argv=None):
    a = build_parser().parse_args(argv)
    t_start = time.time()
    D2 = load_d2(a.dag2_read)
    a.D2 = D2
    a.dag_read_json = f"{a.dag_t1}/read/DAG_READ.json"
    a.reada_banked = f"{a.dagreread}/raw/READ_A.json"
    a.t1kf_banked = f"{a.dagreread}/raw/t1kf_DAG/T1KF_READ_A1_DAG.json"
    bad = D2.pin_problems(static_paths(a)) + own_pin_problems(a)
    if a.static_check:
        print(json.dumps({"pins_checked": len(static_paths(a)) + 5, "problems": bad, "prereg_cells_verbatim": [r[0] for r in ROWS_2] + list(OUTCOME_NAMES)}, indent=1, default=str))
        print("FLATDAG-READ-STATIC-OK" if not bad else "FLATDAG-READ-STATIC-REFUSED")
        return 0 if not bad else 2
    if bad:
        print("FLATDAG READ REFUSED (nothing written): pinned inputs:", file=sys.stderr)
        for b in bad:
            print("  -", b, file=sys.stderr)
        return 2
    DR = D2.load_dag_read(a.dag_read)
    try:
        return run(a, DR, D2, t_start)
    except DR.Refusal as e:
        print("FLATDAG READ REFUSED (nothing written):", file=sys.stderr)
        for m_ in e.msgs:
            print("  -", m_, file=sys.stderr)
        return 2
    except (OSError, KeyError, ValueError) as e:
        print(f"FLATDAG READ INPUT ERROR (nothing written): REFUSED: {type(e).__name__}: {e}", file=sys.stderr)
        return 2


def banked_dump_dir(a, s):
    return f"{a.dag_t1}/dump_DAG_s{s}_m10_causal_heads"


def run(a, DR, D2, t_start):
    ctl = DR.Ctl()
    ctl.add("PINS", True, "pinned inputs re-hashed; PREREG_V8F_FLATDAG section 2 and 4 cells verbatim")
    A = D2.ReadA(a)
    np = A.np
    if a.repro_only:
        return repro_only(a, DR, A, ctl, t_start)
    init_sha = {s: v.lower() for s, v in DR.parse_kv(a.init_sha, "--init-sha").items()}
    manifest = DR.load_manifest(a.manifest)
    if set(manifest["arms"]) - {"DAG", "DAG0"}:
        raise DR.Refusal(f"the manifest must hold exactly the arms DAG (:= FLAT-DAG) and DAG0 (:= FLAT-DAG-0), it holds {sorted(manifest['arms'])}")
    eval_ids, train_ids = DR.jload(manifest["eval141_clip_ids"]), DR.jload(manifest["train_clip_ids"])
    flat_argv = load_flat_argv(a)
    state = {}
    banks, blocked = stage_banks(DR, ctl, manifest, init_sha, eval_ids, train_ids, state)
    if a.preflight:
        recs, build, blocked2 = stage_runs(DR, ctl, manifest, init_sha, banks, train_ids, flat_argv, state)
        for k, v in ctl.items.items():
            print(("PASS " if v["PASS"] else "FAIL ") + k + "  " + str(v["detail"])[:170].replace(chr(10), " "))
        for (arm, s), chk in sorted(build.items()):
            for nm, ok, det in chk:
                print(("PASS " if ok else "FAIL ") + f"{nm}:{SLOT_LABEL[arm]}_s{s}  " + str(det)[:170])
        print(json.dumps({"state": state, "controls": len(ctl.items), "failed": sorted(ctl.failed()), "blocked": blocked + blocked2}, indent=1, default=str))
        if a.out:
            write_out(a, {"preflight": True, "state": state, "controls_PASS": ctl.flags(), "controls": ctl.items, "blocked": blocked + blocked2}, 0)
        return 0 if not ctl.failed() else 1
    if len(banks) != 2:
        raise DR.Refusal([f"both FLAT banks are read for COVERAGE and the bank controls; pulled: {sorted(banks)}"])
    recs, build, blocked_b = stage_runs(DR, ctl, manifest, init_sha, banks, train_ids, flat_argv, state)
    if ctl.failed():
        return refused(ctl)
    if any("COVERAGE" in b for b in blocked):                       # BLOCKED by COVERAGE: no arm was trained on a bank below the minimum; no arm is read
        dec = outcome_block(decide([], blocked + blocked_b, {s: {"separated": False, "delta": 0.0} for s in SEEDS}, {s: [] for s in SEEDS}))
        return write_out(a, {"OUTCOME": dec, "controls_PASS": ctl.flags(), "bank": state, "provenance": provenance(a, t_start), "note": "no arm was read (COVERAGE below the minimum)"}, 4)
    # ------------------------------------------------------------------------------------------------ rollouts, windows, C-F3, C-F4
    T = DR.load_t1hd(a.t1hd_read, a.tool_root)
    K = DR.load_t1k(a.t1k_read, a.tool_root)
    work = a.work_dir or os.path.join(os.path.dirname(os.path.abspath(a.out)), "_flatdag_work")
    os.makedirs(work, exist_ok=True)
    ns = T.parse(["--mode", "real", "--tool-root", a.tool_root, "--eval-cache", a.eval_cache, "--tracks", a.tracks, "--eval-grid", a.eval_grid, "--eval-join", a.eval_join,
                  "--banked-dir", ".", "--out", os.path.join(work, "t1hd_unused.json")])
    backend = DR.RealBackend(T, ns, N_BOOT, work)
    runs, dumps, packs = {}, {}, {}
    banked_dag_entry = lambda s: {"ticks": f"{a.dag_t1}/ticks/ticks_DAG_s{s}_m10_causal_heads.npz", "dump_dir": banked_dump_dir(a, s), "tag": f"DAG_s{s}_m10_causal_heads"}   # noqa: E731
    for s in SEEDS:
        for arm in ("DAG", "DAG0"):
            e = DR.entry(manifest, arm, s)
            tag = f"{SLOT_LABEL[arm]}_s{s}"
            ckpt_tail = e["ckpt_tail"]
            t1 = e["t1"]
            rec = {"entry": e, "tag": tag, "ckpt_tail": ckpt_tail}
            rec["t1json"] = DR.jload(DR.need(t1["json"], f"{tag} t1v8f json"))
            rec["t0json"], rec["t0arr"], t0_ok = DR.t0_arrays(e["t0"], ckpt_tail)
            ctl.add(f"T0-CKPT:{tag}", t0_ok, f"T0 eval json ckpt {rec['t0json'].get('ckpt')!r} vs .../{ckpt_tail}")
            rec["pack"] = pk = backend.analyze_t1(t1, t1.get("tag") or tag, s)
            for k, (ok, det) in pk.controls.items():
                rec.setdefault("k_controls", {})[k] = (ok, det)
            ctl.run(f"T1-CONFIG:{tag}", lambda rec=rec, pk=pk: DR.g_t1cfg(rec["tag"], rec["t1json"], rec["ckpt_tail"], "causal", pk.meta))
            rec["dump"] = A.dump(t1["dump_dir"])
            dumps[(arm, s)] = rec["dump"]
            # C-ARGV: the new rollout's argv equals FLAT's banked rollout argv except --ckpt / --tag / --out-dir / --dump-ticks
            nt = [x for x in open(t1["argv"], encoding="utf-8").read().split("\n") if x != ""] if os.path.isfile(t1["argv"]) else None
            bt = [x for x in open(f"{a.dagreread}/argv/FLAT_s{s}_m10_causal_heads.argv", encoding="utf-8").read().split("\n") if x != ""]
            ctl.add(f"C-ARGV:{tag}", nt is not None and not rollout_argv_problems(nt, bt), {"problems": rollout_argv_problems(nt, bt) if nt is not None else f"{t1['argv']} missing",
                                                                                         "compared_with": f"dagreread/argv/FLAT_s{s}_m10_causal_heads.argv"})
            dn = os.path.join(os.path.dirname(t1["json"]), f"done_{t1['tag']}.txt")
            dr = DR.jload(dn)
            ck = os.path.join(e["train_dir"], "ckpt_last.pt")
            ctl.add(f"C-DONE:{tag}", bool(dr.get("ckpt_sha256") == dr.get("ckpt_sha256_before") == dr.get("ckpt_sha256_after") == sha256_file(ck)),
                    {"done_record": {k: str(dr.get(k))[:16] for k in ("ckpt_sha256", "ckpt_sha256_before", "ckpt_sha256_after")}, "pulled_checkpoint": sha256_file(ck)[:16]})
            runs[(arm, s)] = rec
    for s in SEEDS:
        fd, fd0 = runs[("DAG", s)], runs[("DAG0", s)]
        # C-F3: the checkpoint's own config reads cut (FLAT-DAG; FLAT-DAG-0 beside), and the oracle - own identity reads 0 [0, 0]
        ctl.run(f"CFG-CUT:FLATDAG_s{s}", lambda fd=fd: cfg_cut(fd["entry"]))
        ev = fd["entry"]
        z_own, z_orc = np.load(ev["t0"]["npz"]), np.load(ev["t0_oracle"]["npz"])
        js_orc = DR.jload(ev["t0_oracle"]["json"]) if os.path.isfile(ev["t0_oracle"]["json"]) else None
        probs, idr = oracle_identity_problems(np, z_own, z_orc, js_orc, A.R.PB)
        if js_orc is None:
            probs.append(f"{ev['t0_oracle']['json']} is absent: frac_code_changed cannot be shown, so a zero would say nothing")
        ctl.add(f"ORACLE-IDENTITY:FLATDAG_s{s}", not probs, {"problems": probs, "identity": idr})
    banked_flat = {s: A.R.load_npz(f"{a.dagreread}/packed/dumps_FLAT_s{s}_m10_causal.npz") for s in SEEDS}
    banked_dag = {s: A.dump(banked_dump_dir(a, s)) for s in SEEDS}
    lay = layer_a(a, A, DR, ctl, dumps, banked_dag, banked_flat)
    if ctl.failed():
        return refused(ctl)
    # ------------------------------------------------------------------------------------------------ the rows
    dag_read = DR.jload(a.dag_read_json)
    prim, fam_w, fam_w_tab, fam_nc, tables = {}, {}, {}, {}, {}
    for s in SEEDS:
        tables[s] = lay["tables"]["DAG_minus_FD"][s]
        prim[s] = tables[s]["ADE6"]
        fam_w[s], fam_nc[s] = family_flags(tables[s], A.DIRECTIONS, include_crawl=True)
        fam_w_tab[s], _ = family_flags(tables[s], A.DIRECTIONS, include_crawl=False)
    dec = decide([], blocked_b, prim, fam_w)
    dec_tab = decide([], blocked_b, prim, fam_w_tab)
    dec["ambiguity"] = {"reading_a_primary": "family rows = every computed Read A row, M1 (crawl share) included", "table_only_outcome": dec_tab["OUTCOME"], "differs": dec_tab["OUTCOME"] != dec["OUTCOME"],
                        "family_rows_separated_worse_table_only": {f"s{s}": fam_w_tab[s] for s in SEEDS}, "strategic_rows_not_computed": {f"s{s}": fam_nc[s] for s in SEEDS},
                        "note": "STRATEGIC rows have no paired DAG - FLAT-DAG instrument (reading b): they never count as passing; per-arm S-POS vs CTRV is in beside.s_pos"}
    beside = beside_rows(a, A, DR, T, K, backend, ctl, runs, lay, banked_dag, manifest, dag_read)
    out = {"OUTCOME": outcome_block(dec), "controls_PASS": ctl.flags(), "rows": {f"s{s}": {"PRIMARY_H-T1_ADE6_DAG_minus_FLATDAG": {k: prim[s].get(k) for k in ("delta", "lo", "hi", "separated", "n", "mean_arm", "mean_ref")},
                                                                                           "primary_state": primary_state(prim[s]), "bar": f"point <= -2F = -{TWO_F} m and separated",
                                                                                           "family_table_DAG_minus_FLATDAG": tables[s], "family_rows_separated_worse": fam_w[s],
                                                                                           "family_rows_not_computed": fam_nc[s]} for s in SEEDS},
           "beside": beside, "controls": ctl.items, "bank": state,
           "tier": "T1-kin (self-action OPEN loop; m=10, causal bucket, dead-reckoned nav, logged scene, periodic; plain arms); R-T0 at T0 (box 2's standard eval). Never 'closed loop'; the lead follows its logged future.",
           "estimator": "taniteval.ci.paired_episode_cluster_bootstrap, B=2000, seed 0, 141 episodes (2,399 windows), per seed, paired. It answers 'would another draw of EPISODES say this?'; the training-seed question is "
                        "answered by the two seeds (H-ESTIM-SEED-1: no replicate arm was trained, so a separated row is a result about one training run per seed); the planner is deterministic.",
           "four_families": "ADE6 + LONGITUDINAL / LATERAL / TACTICAL / STRATEGIC rows of Read A, per seed (never pooled); STRATEGIC is reported, not paused for v8F",
           "provenance": provenance(a, t_start)}
    return write_out(a, out, outcome_rc(dec))


def refused(ctl):
    fam = {}
    for k in ctl.failed():
        fam.setdefault(family_of(k), []).append(k)
    print("FLATDAG READ REFUSED (nothing written):", file=sys.stderr)
    for f in sorted(fam):
        print(f"  [{f}]", file=sys.stderr)
        for k in fam[f]:
            print("   -", k, str(ctl.items[k]["detail"])[:300], file=sys.stderr)
    return 2


def cfg_cut(e):
    import torch
    ck = torch.load(os.path.join(e["train_dir"], "ckpt_last.pt"), map_location="cpu", weights_only=False)
    cfg = ck.get("cfg", {})
    oi = cfg.get("o_interface") if isinstance(cfg, dict) else getattr(cfg, "o_interface", None)
    return oi == "cut", {"checkpoint_cfg_o_interface": oi, "n_cfg_keys": len(cfg) if hasattr(cfg, "__len__") else None}


def layer_a(a, A, DR, ctl, dumps, banked_dag, banked_flat):
    """Read A's rows on the pairs; the controls C1 (RC-B2), C2, C3, C4."""
    np, T, RA = A.np, A.T, A.RA
    b2 = DR.jload(f"{a.dagreread}/raw/B2_T1kin_DAG_vs_FLAT.json")
    comps, tacdk, ident = {}, {}, {}
    pair_def = {"DAG_minus_FD": ("dag", "fd"), "FD_minus_FD0": ("fd", "fd0"), "DAG_minus_FD0": ("dag", "fd0"), "FD_minus_FLAT": ("fd", "flat"), "FD0_minus_FLAT": ("fd0", "flat")}
    for k in pair_def:
        comps[k], tacdk[k] = {}, {}
    for s in SEEDS:
        D = {"dag": banked_dag[s], "fd": dumps[("DAG", s)], "fd0": dumps[("DAG0", s)], "flat": banked_flat[s]}
        for nm in ("fd", "fd0", "flat"):
            same = all(np.array_equal(D[nm][k], D["dag"][k]) for k in ("clip_index", "ws", "eid", "g", "v0"))
            ctl.add(f"C4-WINDOWS:{nm}-vs-DAG:s{s}", bool(same and len(D[nm]["eid"]) == DR.N_WINDOWS and len(np.unique(D[nm]["eid"])) == DR.N_EPISODES),
                    {"n_windows": int(len(D[nm]["eid"])), "n_episodes": int(len(np.unique(D[nm]["eid"]))), "note": "clip_index, ws, eid, g, v0 bit-equal to the banked DAG's"})
            ctl.add(f"C3-FLOOR:{nm}-vs-DAG:s{s}", bool(np.array_equal(D[nm]["fl"], D["dag"]["fl"])), "P10's tracks read no model: max |delta| must be exactly 0.0")
        cx = A.tacdk_context(D["dag"])
        P = {nm: T.prep(nm, D[nm]["cl"], cx) for nm in D}
        for k, (x, y) in pair_def.items():
            comps[k][s], tacdk[k][s] = A.pair_rows(D[x]["cl"], D[y]["cl"], D["dag"], P[x], P[y])
        idc = RA.ident_block(D["fd"]["cl"], D["fd"]["g"], D["fd"]["eid"])
        idt = T.rows(P["fd"], P["fd"], D["fd"]["eid"])
        ok = bool(all(v[0] == 0.0 and v[1] == 0.0 and v[2] == 0.0 for v in idc.values()) and all(T.is_zero(v) for v in idt.values()))
        ctl.add(f"C2-IDENTITY:FLATDAG-FLATDAG:s{s}", ok, "(FLAT-DAG, FLAT-DAG) reads +0 [0, 0] on every comps and TAC / DK row")
        # C1 (RC-B2): the banked pair (DAG, FLAT) through the SAME code path reproduces the banked B-2 table with 0 differences
        rc_c, rc_t = A.pair_rows(D["dag"]["cl"], D["flat"]["cl"], D["dag"], P["dag"], P["flat"])
        bs = b2["seeds"][f"s{s}"]
        d1, m1 = rows_diff(rc_c, bs["DAG_minus_FLAT"])
        d2, m2 = rows_diff(rc_t, bs["tac_dk_DAG_minus_FLAT"])
        ctl.add(f"RC-B2:s{s}", not d1 and not d2, {"rows_compared": len(bs["DAG_minus_FLAT"]) + len(bs["tac_dk_DAG_minus_FLAT"]), "max_abs_diff": max(m1, m2), "differences": (d1 + d2)[:6],
                                                   "banked": "dagreread/raw/B2_T1kin_DAG_vs_FLAT.json (PRIMARY ADE6 DAG - FLAT -0.5957 / -0.3505)"})
        # the Read A table: DAG - FLAT-DAG family labels with directions
    dagread = DR.jload(a.dag_read_json)
    tabs = {k: A.table(comps[k], tacdk[k], dagread, None) for k in pair_def}
    return {"tables": tabs, "comps": comps, "tacdk": tacdk}


def beside_rows(a, A, DR, T, K, backend, ctl, runs, lay, banked_dag, manifest, dag_read):
    np = A.np
    nb = N_BOOT
    out = {"per_seed": {}}
    for s in SEEDS:
        fd, fd0 = runs[("DAG", s)], runs[("DAG0", s)]
        dagpack = backend.analyze_t1({"ticks": f"{a.dag_t1}/ticks/ticks_DAG_s{s}_m10_causal_heads.npz", "dump_dir": banked_dump_dir(a, s)}, f"DAG_s{s}_banked", s)
        sd = {}
        # H-T1-HEADS (beside): R-TLAT / R-TLON, DAG - FLAT-DAG; and FLAT-DAG - FLAT-DAG-0
        heads = {}
        for hd, nm in (("lat", "R-TLAT"), ("lon", "R-TLON")):
            heads[nm + "_DAG_minus_FLATDAG"] = DR.f1_pair_row(T, dagpack, fd["pack"], hd, nb, matched=True)[0]
            heads[nm + "_FLATDAG_minus_FLATDAG0"] = DR.f1_pair_row(T, fd["pack"], fd0["pack"], hd, nb, matched=True)[0]
        kc = {lab: {k: v[0] for k, v in ctrls.items()} for lab, ctrls in (("DAG_banked", dagpack.controls), ("FLATDAG", fd["k_controls"]), ("FLATDAG0", fd0["k_controls"]))}
        sd["H-T1-HEADS"] = {"rows": heads, "K_controls_that_ran": kc,
                            "K4": "NOT RUN (needs a logged-bucket rollout and a fresh-age T0 eval; PREREG section 5 schedules neither and section 3 does not require K4)",
                            "admissible": bool(all(ok for lab in kc.values() for ok in lab.values()))}
        # D-FLAT (beside): FLAT-DAG - FLAT-DAG-0 ADE6 + DAGGER's R-ADE / R-T0 guards
        rade = lay["tables"]["FD_minus_FD0"][s]["ADE6"]
        t0 = DR.err_pair_row(T, f"R-T0 T0 own-code ADE6 (FLAT-DAG - FLAT-DAG-0) s{s}", fd["t0arr"]["ade6"], fd0["t0arr"]["ade6"], fd["t0arr"]["eid"], nb)
        t0["bar_point_le"] = HALF_F
        sd["D-FLAT"] = {"ADE6_FLATDAG_minus_FLATDAG0": rade, "R-ADE_guard_not_separated_worse": not sep_worse(rade, "lower"), "R-T0": t0, "guards": guards_beside(rade, t0["point"]),
                        "table_FLATDAG_minus_FLATDAG0": lay["tables"]["FD_minus_FD0"][s]}
        # S-POS against CTRV, both arms (within run, ticks >= 1)
        sd["s_pos"] = {"DAG_banked": dagpack.extra.get(SPOS_KEY), "FLATDAG": fd["pack"].extra.get(SPOS_KEY), "FLATDAG0": fd0["pack"].extra.get(SPOS_KEY)}
        sd["DAG_minus_FLATDAG0_table"] = lay["tables"]["DAG_minus_FD0"][s]
        sd["FLATDAG_minus_FLAT_banked_table"] = lay["tables"]["FD_minus_FLAT"][s]
        sd["FLATDAG0_minus_FLAT_banked_table"] = lay["tables"]["FD0_minus_FLAT"][s]
        out["per_seed"][f"s{s}"] = sd
    out["notes"] = {"D-FLAT": "does DAgger help flat too? FLAT-DAG - FLAT-DAG-0, beside, never decisive", "H-T1-HEADS": "beside; FLAT's O does not read T's code, so T's heads are an input-side diagnostic there"}
    return out


def repro_only(a, DR, A, ctl, t_start):
    """Layer A on BANKED data only: DAG - FLAT recomputed and compared with the banked B-2 table (C1), (DAG, DAG) identity (C2), the two banked arms on the same windows (C4), the floor (C3)."""
    np, T, RA = A.np, A.T, A.RA
    b2 = DR.jload(f"{a.dagreread}/raw/B2_T1kin_DAG_vs_FLAT.json")
    res = {}
    for s in SEEDS:
        dag, flat = A.dump(banked_dump_dir(a, s)), A.R.load_npz(f"{a.dagreread}/packed/dumps_FLAT_s{s}_m10_causal.npz")
        same = all(np.array_equal(flat[k], dag[k]) for k in ("clip_index", "ws", "eid", "g", "v0"))
        ctl.add(f"C4-WINDOWS:FLAT-vs-DAG:s{s}", bool(same and len(dag["eid"]) == DR.N_WINDOWS and len(np.unique(dag["eid"])) == DR.N_EPISODES), {"n_windows": int(len(dag["eid"]))})
        ctl.add(f"C3-FLOOR:FLAT-vs-DAG:s{s}", bool(np.array_equal(flat["fl"], dag["fl"])), "P10 floor")
        cx = A.tacdk_context(dag)
        Pd, Pf = T.prep("DAG", dag["cl"], cx), T.prep("FLAT", flat["cl"], cx)
        rc_c, rc_t = A.pair_rows(dag["cl"], flat["cl"], dag, Pd, Pf)
        bs = b2["seeds"][f"s{s}"]
        d1, m1 = rows_diff(rc_c, bs["DAG_minus_FLAT"])
        d2, m2 = rows_diff(rc_t, bs["tac_dk_DAG_minus_FLAT"])
        ctl.add(f"RC-B2:s{s}", not d1 and not d2, {"rows_compared": len(bs["DAG_minus_FLAT"]) + len(bs["tac_dk_DAG_minus_FLAT"]), "max_abs_diff": max(m1, m2), "differences": (d1 + d2)[:6]})
        idc = RA.ident_block(dag["cl"], dag["g"], dag["eid"])
        idt = T.rows(Pd, Pd, dag["eid"])
        ctl.add(f"C2-IDENTITY:DAG-DAG:s{s}", bool(all(v[0] == 0.0 and v[1] == 0.0 and v[2] == 0.0 for v in idc.values()) and all(T.is_zero(v) for v in idt.values())), "(DAG, DAG) reads +0 [0, 0] on every row")
        res[f"s{s}"] = {"ADE6_DAG_minus_FLAT": {k: rc_c["ADE6"][k] for k in ("delta", "lo", "hi", "separated")}, "max_abs_diff_vs_banked": max(m1, m2)}
    for k, v in ctl.items.items():
        print(("PASS " if v["PASS"] else "FAIL ") + k + "  " + str(v["detail"])[:170].replace(chr(10), " "))
    if a.out:
        write_out(a, {"RC-B2": res, "controls_PASS": ctl.flags(), "controls": ctl.items, "provenance": provenance(a, t_start)}, 0)
    return 0 if not ctl.failed() else 2


def provenance(a, t_start):
    return {"prereg": PREREG_FLATDAG_ID, "prereg_sha256": PREREG_FLATDAG_SHA256, "claim_row": CLAIM_ROW, "reader_sha256": sha256_file(os.path.abspath(__file__)),
            "dag2_read_sha256": D2_READ_SHA256, "manifest_sha256": sha256_file(a.manifest) if a.manifest and os.path.isfile(a.manifest) else None, "wall_s": time.time() - t_start,
            "evidence_class": "MEASURED (ours; artifact paths in the manifest)"}


def write_out(a, out, rc):
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    oc = out.get("OUTCOME")
    if isinstance(oc, dict):
        print(json.dumps({"OUTCOME": oc["OUTCOME"], "matching_rows": oc["matching_rows_in_table_order"], "committed_next": oc.get("committed_next"), "ambiguity": oc.get("ambiguity")}, indent=1, ensure_ascii=False))
    print(f"wrote {a.out}  sha256 {sha256_file(a.out)}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
