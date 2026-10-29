#!/usr/bin/env python3
"""soup_read.py -- the dev-box T1-kin read of PREREG_V8F_SOUP (DRAFT; proposed claim row H-V8F-SOUP-1).
BUILT AND TESTED (read/test_soup_read.py: the pure layer and the composition on a fake heavy layer; read/test_soup_read_heavy.py: the heavy layer's wiring on stubs; read/mutation_soup_reader.py: one-change mutants).
NOT RUN ON SOUP OUTPUTS (none exist).  The heavy layer is written against the LAUNCH-INTENT reader's `Heavy` interface and WAS EXERCISED ON REAL DUMPS WITH STAND-INS (launch precondition 4): read/soup_dry_run.py reads the
banked LAUNCH-INTENT dumps as A_s -> SOUP_s and A_rep,s -> SOUPR_s and reproduces the banked A_CTRL / A_P10 / Arep_P10 tables, the main pair's composition, RF, the 19 bars and the slow-draw flag with 0 differences
(evidence/dry_run_soup_read.json).  What a stand-in run cannot test is listed there: K-SOUP-REC, K-MID, K-BASIN, K-XINIT, K-ZERO-W (they read a soup's own files) and the CONTENT of K-T0DEV (no dev-box standard T0 of
DAG-OACC or A exists yet).  The outcome that run prints is the machinery's on stand-ins: NOT A SOUP RESULT.

THE DESIGN IT READS.  At each seed s two same-seed one-lever pairs (the lever: replace one training draw by the average of two):
    main       X = SOUP_s  (DAG-OACC_s + REP_s)    Y = CTRL_s = DAG-OACC_s
    replicate  X = SOUPR_s (A_s + A_rep,s)         Y = A_s
plus P10 (the `fl` arm of every dump) for Read A, the headline.  SOUP4_s (all four) is beside.

NOTHING IS RE-DERIVED.  The Read-A rows, bars, clears / fires / domain, compose_pair and read_a are the landed LAUNCH-INTENT reader's
(`--li-reader`, sha256 pinned), whose Heavy wraps the DAG-ORC Rig unchanged.  This file adds the soup composition:
  PURE LAYER   per-seed AND over both pairs, the outcome table as ONE function, the soup gain SG and its routing (NEXT-SOUP), K-SOUP-REC on
               records, PREREG-TEXT (the coded table equals the registered table), the four-family block.
  HEAVY LAYER  SoupHeavy(LI.Heavy): soup slots (K-SOUP-REC in place of the training-record controls), same-device R-T0 (t0_dev), SG arrays.

    python soup_read.py --manifest soup_runs.json --prereg PREREG_V8F_SOUP.md --prereg-sha <pin> --li-reader <launch_intent_read.py>
                        --li-reader-sha fb3d2d7d... --orc-dir E:/Projects/TanitAD-artifacts/dagorc_rebase/read --banked-li-read <LAUNCH_INTENT_READ.json>
                        --soup-build <soup_build.py> --spec <soups.json> --out SOUP_READ.json
    rc 0 outcome written | rc 2 REFUSED | rc 3 UNPLACED

Manifest: the LAUNCH-INTENT runs.json ("arms": CTRL, REP, A, Arep; ENTRY as there) plus "SOUP", "SOUPR", "SOUP4": {"0": ENTRY, "1": ENTRY} with
ENTRY = {ckpt_tail, soup_dir, t1: {...}, t1_logged: {...}, t0: {json, npz} (dev-box standard T0), t0_fresh: {...}}, and "t0_dev": {json, npz}
added to the CTRL and A entries (their dev-box standard T0).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PREREG_ID = "PREREG_V8F_SOUP.md (DRAFT, 2026-10-28)"
CLAIM_ROW = "H-V8F-SOUP-1"
LI_READER_SHA = "fb3d2d7d945b54209b62d5f906d4dcea30925a24f0b807459b8cbf3d7f0747a9"
TOOL_DIGEST = "579183a362f7f7843d36bd6c2cc339457b41511c3721e476bfc126eee2321224"      # tree_li (PREREG section 13a): the tool tree every soup rollout must have run in (K-TREE)
SEEDS = (0, 1)
PAIRS = {"main": ("SOUP", "CTRL"), "rep": ("SOUPR", "A")}
MEMBERS = {"SOUP": ("CTRL", "REP"), "SOUPR": ("A", "Arep"), "SOUP4": ("CTRL", "REP", "A", "Arep")}
SOUP_ROLES = ("SOUP", "SOUPR", "SOUP4")
MEMBER_ROLES = ("CTRL", "REP", "A", "Arep")
SPEC_NAME = {"SOUP": "SOUP_s{s}", "SOUPR": "SOUPR_s{s}", "SOUP4": "SOUP4_s{s}"}
SPEC_MEMBER = {"CTRL": "DAGOACC_{s}", "REP": "REP_{s}", "A": "A_{s}", "Arep": "AREP_{s}"}
DONOR = {"SOUP": "CTRL", "SOUPR": "A", "SOUP4": "CTRL"}
GUARDS = ("R-TLAT", "R-TLON", "R-ADE", "R-T0", "R-NEW")
# K-RC (PREREG section 9): the LAUNCH-INTENT tables the reader must reproduce from the member runs, as {banked read_a key: (X, Y)}
K_RC_TABLES = {"A_CTRL": ("A", "CTRL"), "Arep_REP": ("Arep", "REP"), "CTRL_P10": ("CTRL", "P10"), "REP_P10": ("REP", "P10"), "REP_CTRL": ("REP", "CTRL"), "A_P10": ("A", "P10"),
               "Arep_P10": ("Arep", "P10")}
TIER_T1 = "T1-kin (self-action OPEN loop; m=10, causal bucket, dead-reckoned nav, logged scene, periodic schedule, executor plan). Never 'closed loop'."
TIER_T0 = "T0 (teacher-forced own-code, eval-141, ages cycle, dev-box CPU on both arms of a pair). Never 'driving performance'."
ESTIMATOR = "taniteval.ci.paired_episode_cluster_bootstrap, B=2000, seed 0, 141 episodes / 2,399 windows, per seed and per pair, paired."

# The outcome table, verbatim from PREREG section 10 (PREREG-TEXT checks each cell against the file).
OUTCOME_TABLE = (
    ("REFUSED", "any control fails", "fix and re-read; no row is reported"),
    ("HARMS", "controls pass; H fails at ≥ 1 seed", "the uniform two-draw soup is not adopted. NEXT-SOUP (§11)"),
    ("DOMINANT / ADOPTED", "H at both seeds; DM at both seeds; AD at both seeds",
     "adopt SOUP_s as the base; register *\"at T1-kin, the two-draw same-init soup of DAG-OACC is no worse than P10 on any row of the four families, "
     "better on each family, and ≥ 2F better on ADE6, at both seeds, replicated on disjoint members\"*; re-read `PREREG_V8F_DAG_REREAD` Read B-2 on "
     "SOUP (banked FLAT dumps, zero training); signal the PI"),
    ("DOMINANT / NOT ADOPTED", "H at both seeds; DM at both seeds; AD fails at ≥ 1 seed",
     "register the dominance claim on SOUP_s; the base question (SOUP_s trades the R-KEEP point against DAG-OACC) goes to the PI; meanwhile SOUP-K (§11)"),
    ("MOVED / ADOPTED", "H at both seeds; DM fails at ≥ 1 seed; MV at both seeds; AD at both seeds",
     "adopt SOUP_s as the base; the rows that still fail D2 go to SOUP-K on the pure recipe"),
    ("MOVED / NOT ADOPTED", "H at both seeds; DM fails at ≥ 1 seed; MV at both seeds; AD fails at ≥ 1 seed",
     "the lever is attributed but SOUP_s cannot replace DAG-OACC. NEXT-SOUP (§11)"),
    ("ONE SEED", "H at both seeds; DM fails at ≥ 1 seed; MV at exactly one seed",
     "not adopted, never called an effect of the lever (`H-ESTIM-SEED-1`). NEXT-SOUP (§11)"),
    ("NO EFFECT", "H at both seeds; DM fails at ≥ 1 seed; MV at neither seed", "averaging two draws does not move the crawl at this N. NEXT-SOUP (§11)"),
)
OUTCOME_NAMES = tuple(r[0] for r in OUTCOME_TABLE)
ADOPTING = ("DOMINANT / ADOPTED", "MOVED / ADOPTED")
ROUTED = ("HARMS", "MOVED / NOT ADOPTED", "ONE SEED", "NO EFFECT")
NEXT_SOUP = {"BARRIER": "ENS2", "FLAT": "SOUP-K"}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_li(path, pin=LI_READER_SHA):
    """The landed LAUNCH-INTENT reader, by file path, only if its bytes are the pinned ones."""
    got = sha256_file(path)
    if pin is not None and got != pin:
        raise SystemExit(f"REFUSED: LAUNCH-INTENT reader {path} sha256 {got} != pinned {pin}")
    spec = importlib.util.spec_from_file_location("launch_intent_read", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["launch_intent_read"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- pure layer
def per_seed_booleans(pairs, dm):
    """pairs[s] = {"main": compose_pair(...), "rep": compose_pair(...)}; dm[s] = {"SOUP": bool, "SOUPR": bool} (Read A D1 and D2 and D3 at s).
    Section 10: every boolean holds at s only when it holds in BOTH pairs (DM: on BOTH decisive soups)."""
    out = {}
    for s in SEEDS:
        m, r = pairs[s]["main"], pairs[s]["rep"]
        out[s] = {"H": all(m["guards"][g] and r["guards"][g] for g in GUARDS),
                  "MV": bool(m["flags"]["R-MOVE"] and r["flags"]["R-MOVE"]),
                  "AD": bool(m["flags"]["R-KEEP"] and m["flags"]["R-NOBASE"] and r["flags"]["R-KEEP"] and r["flags"]["R-NOBASE"]),
                  "DM": bool(dm[s]["SOUP"] and dm[s]["SOUPR"])}
    return out


def outcome_rows(ctrl, b):
    """Section 10, each row as its own literal predicate.  b[s] = {"H", "MV", "AD", "DM"}."""
    h = all(b[s]["H"] for s in SEEDS)
    dm = all(b[s]["DM"] for s in SEEDS)
    ad = all(b[s]["AD"] for s in SEEDS)
    mv_n = sum(bool(b[s]["MV"]) for s in SEEDS)
    return {
        "REFUSED": not ctrl,
        "HARMS": bool(ctrl and not h),
        "DOMINANT / ADOPTED": bool(ctrl and h and dm and ad),
        "DOMINANT / NOT ADOPTED": bool(ctrl and h and dm and not ad),
        "MOVED / ADOPTED": bool(ctrl and h and not dm and mv_n == 2 and ad),
        "MOVED / NOT ADOPTED": bool(ctrl and h and not dm and mv_n == 2 and not ad),
        "ONE SEED": bool(ctrl and h and not dm and mv_n == 1),
        "NO EFFECT": bool(ctrl and h and not dm and mv_n == 0),
    }


def decide(ctrl, b):
    r = outcome_rows(ctrl, b)
    matching = [n for n in OUTCOME_NAMES if r[n]]
    out = matching[0] if len(matching) == 1 else "UNPLACED"
    return {"OUTCOME": out, "matching_rows_in_table_order": matching, "adopts": out in ADOPTING, "controls_pass": bool(ctrl),
            "per_seed": {f"s{s}": dict(b[s]) for s in SEEDS}}


def soup_gain(x, members, eid, pb, n_boot=2000):
    """SG (section 7): per window x_w - mean_i m_i,w, paired episode-cluster bootstrap through ``pb`` (the rig's T.PB: pb(A, B, eid, n_boot, seed)).
    Windows where any arm is non-finite are dropped and counted."""
    x = np.asarray(x, np.float64)
    ms = np.stack([np.asarray(m, np.float64) for m in members])
    if ms.shape[1] != x.shape[0] or len(eid) != x.shape[0]:
        raise ValueError("SG needs every arm on the same windows")
    mk = np.isfinite(x) & np.isfinite(ms).all(axis=0)
    r = pb(x[mk], ms[:, mk].mean(axis=0), np.asarray(eid)[mk], n_boot=n_boot, seed=0)
    d, lo, hi = float(r["delta"]), float(r["lo"]), float(r["hi"])
    return {"delta": d, "lo": lo, "hi": hi, "separated": bool(lo > 0 or hi < 0), "barrier": bool(lo > 0), "gain": bool(hi < 0),
            "n_windows": int(mk.sum()), "n_dropped": int((~mk).sum())}


def sg_route(cells):
    """NEXT-SOUP routing (section 11) over the four decisive soup-seed cells: BARRIER if any is separated-worse, FLAT otherwise; None if a cell is missing."""
    need = [f"{x}:s{s}" for x in ("SOUP", "SOUPR") for s in SEEDS]
    if any(cells.get(k) is None for k in need):
        return None
    return "BARRIER" if any(cells[k]["barrier"] for k in need) else "FLAT"


def committed_next(outcome, route):
    row = next((r for r in OUTCOME_TABLE if r[0] == outcome), None)
    if row is None:
        return {"committed_next": None, "next_soup": None}
    if outcome not in ROUTED:
        return {"committed_next": row[2], "next_soup": None}
    if route is None:
        return {"committed_next": row[2], "next_soup": "UNDECIDED: SG not computed in every decisive cell; route NEXT-SOUP by hand"}
    return {"committed_next": row[2], "next_soup": NEXT_SOUP[route], "route": route}


def k_soup_rec(rec, spec, name, file_sha, builder_sha, donor_cfg_canon=None, ckpt_cfg_canon=None, opt_is_none=True):
    """K-SOUP-REC on a soup record (pure).  -> list of problems (empty = PASS)."""
    s = spec["soups"][name]
    p = []
    if rec.get("name") != name:
        p.append("record name")
    if rec.get("output_sha256") != file_sha:
        p.append("output sha256 != file")
    if rec.get("members") != list(s["members"]) or rec.get("donor") != s["donor"]:
        p.append("members / donor != spec")
    for m in s["members"]:
        if rec.get("member_sha256", {}).get(m) != spec["members"][m]["sha256"]:
            p.append(f"member {m} sha256 != pin")
    if rec.get("builder_sha256") != builder_sha:
        p.append("builder sha256 != pinned builder")
    kb = rec.get("K-BASIN") or {}
    if not (kb.get("PASS") and kb.get("B1") and kb.get("B2")):
        p.append("K-BASIN not PASS in the record")
    if not (rec.get("K-MID") or {}).get("PASS"):
        p.append("K-MID not PASS in the record")
    if rec.get("weights") != [1.0 / len(s["members"])] * len(s["members"]):
        p.append("weights are not uniform")
    if donor_cfg_canon is not None and ckpt_cfg_canon != donor_cfg_canon:
        p.append("cfg != donor cfg (K-INPUT)")
    if not opt_is_none:
        p.append("opt is not None")
    return p


def k_tree(done, ckpt_sha, tool_digest=TOOL_DIGEST):
    """K-TREE on a rollout's done record (pure): it names the checkpoint that was scored (before, after and at the end), the pinned tool tree, and is not a smoke run.  -> list of problems."""
    if not isinstance(done, dict):
        return ["no done record"]
    p = []
    for k in ("ckpt_sha256", "ckpt_sha256_before", "ckpt_sha256_after"):
        if done.get(k) != ckpt_sha:
            p.append(f"{k} {done.get(k)!r} != the checkpoint {ckpt_sha}")
    if done.get("tool_digest") != tool_digest:
        p.append(f"tool_digest {done.get('tool_digest')!r} != the pinned tree {tool_digest}")
    if done.get("smoke") is not False or done.get("max_windows") is not None:
        p.append("a smoke / window-limited run")
    return p


def rollout_files(man, roles=("A", "Arep") + SOUP_ROLES):
    """Every file of the dev-box rollouts the read depends on (dumps, tick files, run jsons, both lanes): K-P0 requires the frozen RF0 to predate all of them."""
    out = []
    for role in roles:
        for s in SEEDS:
            e = ((man.get("arms") or {}).get(role) or {}).get(str(s)) or {}
            for lane in ("t1", "t1_logged"):
                t = e.get(lane) or {}
                out += glob.glob(os.path.join(t.get("dump_dir") or "", "ep*.npz")) + [t.get("json") or "", t.get("ticks") or ""]
    return out


def strip_md(t):
    return t.replace("**", "").strip()


def prereg_text_problems(path, pin):
    """PREREG-TEXT: the file hashes to the pin, and every row of the coded outcome table appears in section 10 cell for cell."""
    if not os.path.exists(path):
        return [f"PREREG {path} missing"]
    probs = []
    got = sha256_file(path)
    if pin is not None and got != pin:
        probs.append(f"PREREG sha256 {got} != pin {pin}")
    with open(path, encoding="utf-8") as f:
        txt = f.read()
    sec = txt.split("## 10.", 1)[-1].split("## 11.", 1)[0]
    rows = {}
    for ln in sec.splitlines():
        if ln.startswith("| **"):
            cells = [strip_md(c) for c in ln.strip().strip("|").split("|")]
            if len(cells) >= 3:
                rows[cells[0]] = (cells[1], cells[2])
    for name, cond, nxt in OUTCOME_TABLE:
        if rows.get(name) != (strip_md(cond), strip_md(nxt)):
            probs.append(f"outcome row {name!r} differs from the PREREG's section 10")
    extra = set(rows) - set(OUTCOME_NAMES)
    if extra:
        probs.append(f"PREREG section 10 has rows the reader does not code: {sorted(extra)}")
    return probs


def families_block(LI, rows_by_contrast):
    fb = LI.families_report(rows_by_contrast)
    fb["LATERAL"]["note"] = "the crawl rows are LATERAL plus M1: read first (PREREG_V8F_SOUP section 8)."
    fb["LONGITUDINAL"]["note"] = "the anti-echo block is NOT BUILT (the DAG-ORC rig carries none)."
    return fb


# ---------------------------------------------------------------- composition (heavy layer injected; tested on a fake)
def read_core(H, LI, ctl, man, rf0, bars=None):
    """The whole analysis on an injected heavy layer H.  -> result dict (nothing written)."""
    trs = {r: {s: LI.load_train_records(man["arms"][r][str(s)]) for s in SEEDS} for r in MEMBER_ROLES}
    LI.run_record_controls(ctl, trs)                                                     # K-MEMBERS
    for r in MEMBER_ROLES:
        for s in SEEDS:
            H.load_slot(ctl, r, s, man["arms"][r][str(s)])
    soups = [x for x in SOUP_ROLES if all((man["arms"].get(x) or {}).get(str(s)) for s in SEEDS)]
    for need in ("SOUP", "SOUPR"):
        if need not in soups:
            raise ValueError(f"manifest has no {need} at both seeds")
    for x in soups:
        for s in SEEDS:
            H.load_soup_slot(ctl, x, s, man["arms"][x][str(s)])
    H.global_controls(ctl)                                                               # K-XINIT, K-ZERO-W (weights only)
    H.prepare(ctl)
    H.k_rc(ctl)                                                                          # K-RC before any soup row
    bars = bars or LI.bar_table(rf0)
    tabs = {"REP_CTRL": H.contrast("REP", "CTRL"), "CTRL_P10": H.contrast("CTRL", "P10"), "A_P10": H.contrast("A", "P10")}
    rf = LI.floors_from({s: tabs["REP_CTRL"][s] for s in SEEDS})
    for role in soups + ["CTRL", "A"]:
        z = H.contrast(role, role)
        for s in SEEDS:
            ctl.add(f"K-ZERO:{role}-{role}:s{s}", LI.is_exact_zero(z[s]), "(X, X) reads +0 [0, 0] on every Read-A row")
    pairs, g_rows = {s: {} for s in SEEDS}, {}
    for pname, (x, y) in PAIRS.items():
        tabs[f"{x}_P10"], tabs[f"{x}_{y}"] = H.contrast(x, "P10"), H.contrast(x, y)
        for s in SEEDS:
            g = H.guard_rows(ctl, x, y, s, tabs[f"{x}_{y}"])
            g_rows[(pname, s)] = g
            t = {"X_P10": tabs[f"{x}_P10"][s], "X_Y": tabs[f"{x}_{y}"][s], "Y_P10": tabs[f"{y}_P10"][s]}
            pairs[s][pname] = LI.compose_pair(t, g, bars)
    if "SOUP4" in soups:
        tabs["SOUP4_P10"], tabs["SOUP4_CTRL"] = H.contrast("SOUP4", "P10"), H.contrast("SOUP4", "CTRL")
    tabs["SOUP_REP"], tabs["SOUPR_Arep"], tabs["SOUPR_CTRL"] = H.contrast("SOUP", "REP"), H.contrast("SOUPR", "Arep"), H.contrast("SOUPR", "CTRL")
    read_a = {x: LI.read_a({s: tabs[f"{x}_P10"][s] for s in SEEDS}) for x in soups}
    dm = {s: {x: bool(read_a[x]["per_seed"][f"s{s}"]["D1"] and read_a[x]["per_seed"][f"s{s}"]["D2"] and read_a[x]["per_seed"][f"s{s}"]["D3"])
              for x in ("SOUP", "SOUPR")} for s in SEEDS}
    sg, sg_errors = {}, {}
    for x in soups:
        for s in SEEDS:
            try:
                sg[f"{x}:s{s}"] = H.sg(x, MEMBERS[x], s)
            except Exception as e:   # noqa: BLE001 -- SG decides no outcome: a missing cell routes NEXT-SOUP UNDECIDED and is reported, never filled in
                sg[f"{x}:s{s}"] = None
                sg_errors[f"{x}:s{s}"] = repr(e)
    slow, slow_errors = {}, {}
    for pname, (x, y) in PAIRS.items():                  # beside, decides nothing (PREREG section 7): the slow-draw flag of each soup against its pair's control, on the dev-box standard T0
        for s in SEEDS:
            k_ = f"{pname}:{x}-{y}:s{s}"
            try:
                slow[k_] = LI.slow_draw(H.speed(x, y, s))
            except Exception as e:   # noqa: BLE001 -- beside only: reported, never filled in
                slow[k_], slow_errors[k_] = None, repr(e)
    H.finish(ctl)
    b = per_seed_booleans(pairs, dm)
    d = decide(not ctl.failed(), b)
    route = sg_route(sg)
    d.update(committed_next(d["OUTCOME"], route))
    return {"decision": d, "pairs": pairs, "read_a": read_a, "sg": sg, "sg_errors": sg_errors, "route": route, "tabs": tabs, "slow": slow, "slow_errors": slow_errors,
            "floors": {"RF0": rf0, "RF": rf, "bar": bars}, "g_rows": g_rows}




# ---------------------------------------------------------------- the record that is written
def refused_record(d, ctl, prov):
    """REFUSED writes the failed controls and NO row (PREREG section 10)."""
    return {"OUTCOME": d, "controls_PASS": ctl.flags(), "failed": {k: v["detail"] for k, v in ctl.failed().items()}, "note": "no row is reported (PREREG section 10)", "provenance": prov}


def result_record(LI, res, ctl, prov):
    d = res["decision"]
    return {"OUTCOME": d, "controls_PASS": ctl.flags(), "read_a": res["read_a"], "sg": res["sg"], "sg_errors": res["sg_errors"], "route": res["route"],
            "beside": {"slow_draw": res["slow"], "slow_draw_errors": res["slow_errors"], "RF_per_row": res["floors"]["RF"]},
            "pairs": {f"s{s}": res["pairs"][s] for s in SEEDS}, "floors": res["floors"],
            "tables": {k: {f"s{s}": LI.trim(v[s]) for s in SEEDS} for k, v in res["tabs"].items()},
            "four_families": {f"{p}:s{s}": families_block(LI, {f"{x}-{y}": res["tabs"][f"{x}_{y}"][s], f"{x}-P10": res["tabs"][f"{x}_P10"][s]})
                              for p, (x, y) in PAIRS.items() for s in SEEDS},
            "tier": {"T1": TIER_T1, "T0": TIER_T0}, "estimator": ESTIMATOR, "strategic_family": "NOT PAUSED for v8F (PI ruling 2026-10-17)", "provenance": prov}


# ---------------------------------------------------------------- heavy layer (real rig; exercised on the banked LAUNCH-INTENT dumps by soup_dry_run.py, never on a soup)
def make_heavy(LI, orc_dir, spec_path, soup_build_path, soup_build_sha, banked_li_read, n_boot=2000, work_dir=None, standin=False):
    """``standin`` (soup_dry_run.py only; main() never sets it): a slot is a banked LAUNCH-INTENT run standing in for a soup, so the controls that read a soup's own files (K-SOUP-REC, K-MID,
    K-BASIN, K-XINIT, K-ZERO-W) cannot apply; they are listed in ``H.na`` with the reason and are never reported as passed."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(soup_build_path)))
    if sha256_file(soup_build_path) != soup_build_sha:
        raise SystemExit("REFUSED: soup_build.py sha256 != pin")
    import soup_build as SB  # noqa: E402
    if os.path.normcase(os.path.abspath(SB.__file__)) != os.path.normcase(os.path.abspath(soup_build_path)):
        raise SystemExit(f"REFUSED: soup_build was imported from {SB.__file__}, not from {soup_build_path}")

    class SoupHeavy(LI.Heavy):
        def __init__(self):
            super().__init__(orc_dir=orc_dir, work_dir=work_dir, n_boot=n_boot)
            with open(spec_path, encoding="utf-8") as f:
                self.spec = json.load(f)
            self.dev, self.na, self.standin, self.banked_path, self._cc = {}, [], standin, banked_li_read, {}

        # ---- member slots (CTRL, REP, A, Arep): the LAUNCH-INTENT reader's own controls; CTRL and A also carry the dev-box standard T0 (K-T0DEV)
        def load_slot(self, ctl, role, s, e, standin=False):
            rec = super().load_slot(ctl, role, s, e, standin)
            td = e.get("t0_dev")
            if role in ("CTRL", "A"):
                if not td:
                    raise self.DR.Refusal(f"{role}_s{s}: manifest has no 't0_dev' (K-T0DEV: R-T0 reads dev-box T0 on both arms)")
                dj, darr, ok = self.DR.t0_arrays(td, rec["ckpt_tail"])
                ctl.add(f"T0DEV-CKPT:{role}_s{s}", ok, f"dev T0 json ckpt {dj.get('ckpt')!r}")
                with np.load(self.DR.need(td["npz"], "dev T0 npz")) as z:
                    self.dev[(role, s)] = {"t0arr": darr, "t0cons": (np.asarray(z["cons_pred"]), np.asarray(z["eid"])), "t0json": dj}
                ctl.add(f"K-T0DEV-BESIDE:{role}_s{s}", True, {"dev_minus_box2_ade6_mean": float(np.nanmean(darr["ade6"]) - np.nanmean(rec["t0arr"]["ade6"]))})
            return rec

        # ---- soup slots
        def load_soup_slot(self, ctl, role, s, e):
            name, tag = SPEC_NAME[role].format(s=s), f"{role}_s{s}"
            if self.standin:
                self.na += [f"K-SOUP-REC:{tag}", f"K-MID:{tag}", f"K-BASIN:{tag}"]
                ck_path = f"{e['train_dir']}/ckpt_last.pt"
            else:
                self.soup_record_controls(ctl, role, s, e, name, tag)
                ck_path = f"{e['soup_dir']}/ckpt_last.pt"
            return self.load_soup_run(ctl, role, s, e, tag, ck_path)

        def soup_record_controls(self, ctl, role, s, e, name, tag):
            """K-SOUP-REC (the record, re-read), K-MID (the average, re-measured from the member files) and K-BASIN (the weight distances, re-measured)."""
            d = e["soup_dir"]
            ckp, recp = f"{d}/ckpt_last.pt", f"{d}/soup.json"
            with open(recp, encoding="utf-8") as f:
                srec = json.load(f)
            import torch
            ck = torch.load(ckp, map_location="cpu", weights_only=False)
            donor = self.spec["members"][self.spec["soups"][name]["donor"]]
            dck = torch.load(donor["path"], map_location="cpu", weights_only=False)
            probs = k_soup_rec(srec, self.spec, name, sha256_file(ckp), sha256_file(soup_build_path), SB._canon(dck["cfg"]), SB._canon(ck["cfg"]), ck.get("opt") is None)
            ctl.add(f"K-SOUP-REC:{tag}", not probs, probs)
            v = SB.verify(self.spec, name, os.path.dirname(d))
            ctl.add(f"K-MID:{tag}", v["PASS"], v["problems"])
            kb = SB.basin_for(self.spec, name)
            ctl.add(f"K-BASIN:{tag}", kb["PASS"], {"spread_ratio": kb["spread_ratio"], "B1": kb["B1"], "B2": kb["B2"]})

        def load_soup_run(self, ctl, role, s, e, tag, ck_path):
            """The per-run pipeline controls of the DAG-ORC rig on a soup's rollouts (LI.Heavy.load_slot's, without the training records a soup does not have) + K-TREE."""
            DR, rig, np_, nb = self.DR, self.rig, np, self.n_boot
            ckpt_tail = e["ckpt_tail"]
            rec = {"role": role, "seed": s, "tag": tag, "entry": e, "ckpt_tail": ckpt_tail}
            t1 = e["t1"]
            rec["t1json"] = DR.jload(DR.need(t1["json"], f"{tag} t1v8f json"))
            rec["dump"] = DR.load_dump(t1["dump_dir"])
            rec["pack"] = pk = rig.backend.analyze_t1(t1, t1.get("tag") or tag, s)
            for k, (ok, det) in pk.controls.items():
                ctl.add(f"{k}:{tag}", ok, det)
            ctl.run(f"T1-CONFIG:{tag}", lambda: DR.g_t1cfg(tag, rec["t1json"], ckpt_tail, "causal", pk.meta))
            lg, t0f = e["t1_logged"], e["t0_fresh"]
            lpk = rig.backend.analyze_t1_light(lg, lg.get("tag") or f"{tag}-logged", s)
            for k, (ok, det) in lpk.controls.items():
                ctl.add(f"{k}-logged:{tag}", ok, det)
            lj = DR.jload(DR.need(lg["json"], f"{tag} logged t1v8f json"))
            ctl.run(f"T1-CONFIG-logged:{tag}", lambda: DR.g_t1cfg(tag, lj, ckpt_tail, "logged", lpk.meta))
            ok, det = rig.backend.k4(lg, pk, t0f, s)
            ctl.add(f"K4:{tag}", ok, det)
            t0fj = DR.jload(DR.need(t0f["json"], "fresh T0 json"))
            ctl.add(f"T0F-CKPT:{tag}", DR.tail2(str(t0fj.get("ckpt", ""))) == ckpt_tail, f"fresh T0 eval json ckpt {t0fj.get('ckpt')!r} vs .../{ckpt_tail}")
            rec["t0json"], rec["t0arr"], t0_ok = DR.t0_arrays(e["t0"], ckpt_tail)
            ctl.add(f"T0-CKPT:{tag}", t0_ok, f"standard T0 eval json ckpt {rec['t0json'].get('ckpt')!r} vs .../{ckpt_tail}")
            with np_.load(DR.need(e["t0"]["npz"], "T0 npz")) as z:
                rec["t0cons"] = (np_.asarray(z["cons_pred"]), np_.asarray(z["eid"]))
            d = rec["dump"]
            ctl.add(f"DUMP-WINDOWS:{tag}", bool(len(d["ws"]) == pk.N and np_.array_equal(d["clip_index"], pk.ci) and np_.array_equal(d["ws"], pk.t0)), {"dump_n": int(len(d["ws"])), "ticks_n": int(pk.N)})
            ctl.add(f"GRID-COUNTS:{tag}", bool(pk.N == LI.N_WINDOWS and len(np_.unique(d["eid"])) == LI.N_EPISODES), {"windows": int(pk.N), "episodes": int(len(np_.unique(d["eid"])))})
            js = (rec["t1json"].get("paired_decision_grade") or {}).get("paired_model_minus_floor_t1", {}).get("ade_m")

            def r2():
                if js is None:
                    return False, "the run json carries no paired_model_minus_floor_t1.ade_m to reproduce"
                A_, B_ = self.K.comps(d["cl"], d["g"])["ade6_m"], self.K.comps(d["fl"], d["g"])["ade6_m"]
                mk = np_.isfinite(A_) & np_.isfinite(B_)
                r = self.T.PB(A_[mk], B_[mk], d["eid"][mk], n_boot=nb, seed=0)
                return abs(r["delta"] - float(js["delta"])) <= 1e-6, {"reader_delta": r["delta"], "analysis_delta": js["delta"]}
            ctl.run(f"R2-ADE6:{tag}", r2)

            def ktree():
                want = sha256_file(ck_path)
                bad = {}
                for lane, ent in (("t1", t1), ("t1_logged", lg)):
                    dn = ent.get("done")
                    if not dn or not os.path.isfile(dn):
                        bad[lane] = f"no done record {dn!r}"
                        continue
                    with open(dn, encoding="utf-8") as f:
                        probs = k_tree(json.loads(f.read()), want)
                    if probs:
                        bad[lane] = probs
                return not bad, bad or {"ckpt_sha256": want, "tool_digest": TOOL_DIGEST}
            ctl.run(f"K-TREE:{tag}", ktree)
            self.recs.setdefault(role, {})[s] = rec
            return rec

        # ---- weights-only controls that belong to no single soup
        def global_controls(self, ctl):
            if self.standin:
                self.na += ["K-XINIT:s0", "K-XINIT:s1", "K-ZERO-W"]
                return
            cache = {}
            for s in SEEDS:
                name = f"XINIT_s{s}"

                def xinit(name=name):
                    full, b2, b1 = SB.basin_for(self.spec, name, cache), SB.basin_for(self.spec, name, cache, clauses=("B2",)), SB.basin_for(self.spec, name, cache, clauses=("B1",))
                    return bool(not full["PASS"] and not b2["PASS"] and not b1["PASS"]), {"full": full["PASS"], "B2_only": b2["PASS"], "B1_only": b1["PASS"], "spread_ratio": full["spread_ratio"]}
                ctl.run(f"K-XINIT:s{s}", xinit)

            def kzero():
                m = self.spec["members"]["DAGOACC_0"]
                ck, _ = SB.load_ckpt(m["path"], m["sha256"])
                kz = SB.k_zero(ck)
                return kz["PASS"], kz["per_n"]
            ctl.run("K-ZERO-W", kzero)

        # ---- K-RC, then the tables
        def contrast(self, x, y):
            if (x, y) not in self._cc:
                self._cc[(x, y)] = super().contrast(x, y)
            return self._cc[(x, y)]

        def k_rc(self, ctl):
            """K-RC (PREREG section 9): the seven banked LAUNCH-INTENT tables reproduced from the member runs with 0 differences (every key of every row, exact equality), and RF equal to the banked floors.RF."""
            with open(self.banked_path, encoding="utf-8") as f:
                banked = json.load(f)
            got = {}
            for name, (x, y) in K_RC_TABLES.items():
                t = got[name] = self.contrast(x, y)
                diffs = [(s, d) for s in SEEDS for d in LI.row_diffs(LI.trim(t[s]), banked["read_a"][name][f"s{s}"])]
                ctl.add(f"K-RC:{name}", not diffs, diffs[:10])
            rf = LI.floors_from({s: got["REP_CTRL"][s] for s in SEEDS})
            ctl.add("K-RC:RF", rf == banked["floors"]["RF"], {l: (rf[l], banked["floors"]["RF"].get(l)) for l in LI.LABELS if rf[l] != banked["floors"]["RF"].get(l)})

        # ---- the guards: R-T0 reads the dev-box standard T0 on BOTH arms of a pair (the control's box-2 T0 is swapped for its dev-box one while the LAUNCH-INTENT guard runs)
        def _with_dev_control(self, y, s):
            keep = self.recs[y][s]
            self.recs[y][s] = {**keep, "t0arr": self.dev[(y, s)]["t0arr"], "t0cons": self.dev[(y, s)]["t0cons"]}
            return keep

        def guard_rows(self, ctl, x, y, s, tab_xy):
            keep = self._with_dev_control(y, s)
            try:
                return super().guard_rows(ctl, x, y, s, tab_xy)
            finally:
                self.recs[y][s] = keep

        def speed(self, x, y, s, rng=None):
            keep = self._with_dev_control(y, s)
            try:
                return super().speed(x, y, s, rng)
            finally:
                self.recs[y][s] = keep

        def sg(self, x, members, s):
            K, T = self.K, self.T
            dx = self.recs[x][s]["dump"]
            a = K.comps(dx["cl"], dx["g"])["ade6_m"]
            ms = [K.comps(self.recs[m][s]["dump"]["cl"], self.recs[m][s]["dump"]["g"])["ade6_m"] for m in members]
            for m in members:
                if not np.array_equal(self.recs[m][s]["dump"]["eid"], dx["eid"]):
                    raise ValueError(f"SG: {m} windows differ from {x} at s{s}")
            return soup_gain(a, ms, dx["eid"], T.PB, self.n_boot)

    return SoupHeavy()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prereg", required=True)
    ap.add_argument("--prereg-sha", required=True)
    ap.add_argument("--li-reader", required=True)
    ap.add_argument("--li-reader-sha", default=LI_READER_SHA)
    ap.add_argument("--orc-dir", required=True)
    ap.add_argument("--banked-li-read", required=True)
    ap.add_argument("--soup-build", required=True)
    ap.add_argument("--soup-build-sha", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--work-dir", default="")
    a = ap.parse_args(argv)
    t0 = time.time()
    probs = prereg_text_problems(a.prereg, a.prereg_sha)
    if probs:
        print("REFUSED: PREREG-TEXT", probs)
        return 2
    LI = load_li(a.li_reader, a.li_reader_sha)
    with open(a.manifest, encoding="utf-8") as f:
        man = json.load(f)
    H = make_heavy(LI, a.orc_dir, a.spec, a.soup_build, a.soup_build_sha, a.banked_li_read, work_dir=a.work_dir or None)
    ctl = LI.Ctl()
    sp = H.static_problems()
    ctl.add("STATIC", not sp, sp or "staged readers, Read A's banked inputs, DAG-ORC's PREREG text, RF0 pin")

    def prov():
        return {"prereg": PREREG_ID, "prereg_sha256": a.prereg_sha, "claim_row": CLAIM_ROW, "reader_sha256": sha256_file(os.path.abspath(__file__)),
                "li_reader_sha256": a.li_reader_sha, "manifest_sha256": sha256_file(a.manifest), "wall_s": round(time.time() - t0, 1),
                "evidence_class": "MEASURED (ours; artifact paths in the manifest)"}

    def write(rec):
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(rec, f, indent=1, sort_keys=True, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    pr, rf0, det = H.rf0(rollout_files(man))
    ctl.add("K-P0", not pr, pr or {"RF0_sha256": LI.RF0_SHA256[:16] + "...", **det})
    refused_d = {"OUTCOME": "REFUSED", "controls_pass": False, "committed_next": OUTCOME_TABLE[0][2]}
    if not rf0:                                              # K-P0 failed: no RF0, no bars; the record carries the controls read so far
        write(refused_record(refused_d, ctl, prov()))
        print("SOUP READ REFUSED: K-P0", file=sys.stderr)
        return 2
    try:
        res = read_core(H, LI, ctl, man, rf0)
    except (LI.RowNotComputed, OSError, KeyError, ValueError, getattr(getattr(H, "DR", None), "Refusal", ValueError)) as e:
        print(f"SOUP READ REFUSED (nothing written): {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    d = res["decision"]
    if d["OUTCOME"] == "REFUSED":
        write(refused_record(d, ctl, prov()))
        print("SOUP READ REFUSED: failed controls:", *[f"{k}: {str(v['detail'])[:200]}" for k, v in ctl.failed().items()], sep="\n  - ", file=sys.stderr)
        return 2
    write(result_record(LI, res, ctl, prov()))
    print("SOUP READ", d["OUTCOME"], "next:", d.get("committed_next"), "|", d.get("next_soup"))
    return 3 if d["OUTCOME"] == "UNPLACED" else 0


if __name__ == "__main__":
    sys.exit(main())
