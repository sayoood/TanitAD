#!/usr/bin/env python3
"""recover_read.py (RECOVER-NEXT extension) -- the RECOVER-FT T1-kin read AND the COMPOSITE read, every rule of both frozen cards in ONE JSON (raw/RECOVER_READ_<ARM>.json).

EXTENSION (dagland/recover_next/PREREG_RECOVER_NEXT_FROZEN.md section 3): ``--r-arm {R,RAPR,RTL,RP01,RAPRP01,RPT}`` (default R) substitutes the run ``M2b_s{s}_recft_<ARM>`` for ``M2b_s{s}_recft_R`` as the R role.
Every rule, bar and row is the source reader's (dagland/recover_read/code/recover_read.py, sha256 bf533d6b...); the C0 / C0REP / FZC0 / C0C roles REUSE the finished rollouts in --reuse-dir.
Only the controls that state a fact about the R run itself are parametrised by the arm's declared levers (ARM_SPEC; each is listed in HANDOFF.md): KR1 band, KR2 argv audit, KR3, KMARK path.
Added for ARM != R ONLY (so ``--r-arm R`` reproduces raw/RECOVER_READ.json key for key): KCARD of the RECOVER-NEXT card, KREUSE (reused rollouts' checkpoint and tool shas), the ``r_arm`` keys.

Cards: dagland/recover_ft/PREREG_RECOVER_FT_FROZEN.md (sha256 57caba3b..., + AMENDMENT_1.md) sections 6-7, and dagland/recover_read/PREREG_COMPOSITE_FROZEN.md (FROZEN_SHA.txt).
Tier T1-kin; evidence class MEASURED (ours) once run on the six box-2 checkpoints.  Estimator of every interval: paired episode-cluster bootstrap, B = 2000, seed 0 (taniteval.ci draw stream via
gapdecomp.decompose) over the 141 eval episodes / 2,399 windows.  It answers "another draw of EPISODES" ONLY; the training-run question is read by the C0rep floor (F), the inference question by KA1.

Rules in one place (pure layer, unit-checked in tests/):  primary_rule (P)  guards_rule (G)  floor_rule (F)  label  read_a_block (Read A of R vs P10)  freeze_reading (ROBUST / ARTEFACT / PARTIAL)
composite_rule (DOMINANT of RC vs P10 at both seeds, counted only if R's freeze reading is ROBUST / PARTIAL; attribution beside it).
Heavy layer: pinned reuse of SMOOTH-T1's / HISTSWAP-LF's / FRAMEFREEZE's readers (loaded BY FILE PATH; no second stack root is ever put on sys.path), per-window arrays by SMOOTH-T1's extract_smooth.py,
and the file controls KP1 KR1 KR2 KR3 KA1 KFZ1 KFZ3(R, C0) KFZ4 (read from framefreeze/raw/FZ_CONTROLS.json) KK1 KK2 KC1 KC3 KF KB1.
Modes: production (the six pulled checkpoints' rollouts) | kat1 | kat2 | kat3 (known-answer runs on EXISTING arrays; file controls are not applicable there and the JSON says so).
A failed KFZ control REFUSES the frame-freeze contrast only (RECOVER-FT section 7): P, G, F and Read A are still read, the label and the composite say UNREAD; any other failed control REFUSES the read (exit 3).
STRATEGIC is PAUSED (PI ITEM 25): reported PAUSED beside every verdict, never scored as a pass of its own; D3 is evaluated over four families as Read A defines it, and the three-family form is reported beside it.
"""
import argparse
import concurrent.futures as cf
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)                          # THIS package (recover_next/read): logs, raw
ART = "E:/Projects/TanitAD-artifacts"
DL = f"{ART}/dagland"
RR_PK = f"{DL}/recover_read"                        # the source package: the composite card and its FROZEN_SHA.txt live there
NX_PK = f"{DL}/recover_next"                        # the RECOVER-NEXT card
SMT, LF_PK, FZ_PK, FT_PK, GD_PK, KAP_PK = (f"{DL}/smooth_t1", f"{DL}/histswap_lf", f"{DL}/framefreeze", f"{DL}/recover_ft", f"{DL}/gapdecomp", f"{DL}/kappa_t1")
RR_SCRATCH = "C:/Users/Admin/rr_scratch"            # F: is full: every write is on C: (READER_FIX_RREAD2 precedent)
REUSE_DEFAULT = f"{RR_SCRATCH}/out"                 # the finished C0 / C0rep / FZ-C0 / C0C (and R, KA1R, FZR, RC) rollouts: READ ONLY
WORK_DEFAULT = f"{RR_SCRATCH}/rnext_work"           # extractor rigs (a new directory)
RECOVER_DIR = f"{ART}/v8f-recover"
DAGOACC_DIR = f"{ART}/v8f-dagoacc"
FD_PATH = f"{ART}/dagt1_tree_b43c7a9a/stack/tanitad/refs/feasible_decode.py"      # loaded by FILE PATH (imports only math / torch)
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
#: module pins (sha256): the reused instruments are the ones the earlier cards froze
PINS = {f"{SMT}/code/smooth_t1_read.py": "3de1db811b1b26652f9d2a80477f2e0955696f04dd0047c823d7162a3468ded5",
        f"{SMT}/code/extract_smooth.py": "7c41c0e464c2babd799cb97dfb9a57ea55f0f073012d2d3b12cb293bf36c2351",
        f"{GD_PK}/gapdecomp.py": "824c177d8fc7b8ce3728c334bece1f2d628ea2ee684fefe8f62c2d3b7e310f7f",
        f"{LF_PK}/code/histswap_lf_read.py": "4e3a75db6023d7eb07465bb826fe56455dac01aa1aba42fda0b197626cbe119f",
        f"{FZ_PK}/code/framefreeze_read.py": "bedd8e0d591230edd16e11c6d43e369bb8cb3802b81d043db80b9b510e756070",
        f"{FZ_PK}/raw/FZ_CONTROLS.json": "92ac5c6c4232600ccee2ea740adb0941328c9bce7837f1305f39aec8c65ba08b"}
BANK_SHA = {0: "90c43c66f52f250d41ff23e837d5fb31d1592cf2aad14afacd7ded011734b589", 1: "a06c8289b20854a6933264e95dca201ea7d9e2723fd0b2248872f56361ea689a"}
INIT_PREFIX = {0: "67265ec1666f", 1: "90b014748190"}

WORK = f"{WORK_DEFAULT}/rig"                         # extractor work root (main() sets it from --work-dir)
TOOL_PIN_LINE = "code/v8f_t1_eval_framefreeze.py sha256 "
SEEDS = (0, 1)
M1 = "M1 crawl share"
SHARE_BAR, ARTEFACT_BAR = 0.5, 0.25                 # RECOVER-FT sections 6 and 7
ADE_GUARD_M, D1_POINT_M = 0.0604, -0.2415
KR1_LO, KR1_HI = 0.25, 0.35
TOL = 1e-4                                          # same-session bit-identity tolerance (the earlier readers' KA1 / KF / KC rule)
FL_GROSS_M = 1.0
EXEC_TOL_M = 5e-3
KFZ3_TOL_M = 1e-3
FAMILIES = ("LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC")
STRATEGIC_NOTE = "PAUSED (PI ITEM 25, 2026-09-19): reported beside the verdict, never scored as a pass of its own; D3 is evaluated over four families as Read A defines it and the three-family D3 is reported beside it"
VARIANCE = {"episodes": "answered by the paired episode-cluster bootstrap (every interval here)",
            "training_run": "answered ONLY by the C0rep floor (F); a separated CI is necessary, not sufficient (H-ESTIM-SEED-1)",
            "inference": "the planner is deterministic; read by the KA1 bit-identity rerun of R_s0"}
FLAGS = ["--replan-steps", "10", "--nav", "deadreckon", "--schedule", "periodic", "--limit", "causal", "--dump-heads", "--device", "cpu"]

#: The R-role arms (RECOVER-NEXT card section 2).  ``declared`` = the recovery flags the arm's shipped argv names (and the values the controls expect); ``diff_vs_R`` = the EXACT difference of its argv from R's
#: (the card: "init, data, steps, every other flag = R_s's shipped argv"); ``kr1_band`` = the card's selected / eligible band scaled with p (R: [0.25, 0.35] at p 0.3, i.e. p x [5/6, 7/6]);
#: ``bank`` = whether the arm draws from the dagger bank (then KR3 reads the table build and K-B3; RAPR's a-priori law reads no bank and builds no table, so KR3 reads that fact instead).
ARM_SPEC = {
    "R": {"p": 0.3, "law": None, "target": "rejoin", "rejoin_steps": "10", "bank": True, "kr1_band": (KR1_LO, KR1_HI), "diff_vs_R": {"added": {}, "removed": [], "changed": {}}},
    "RAPR": {"p": 0.3, "law": "apriori", "target": "rejoin", "rejoin_steps": "10", "bank": False, "kr1_band": (KR1_LO, KR1_HI),
             "diff_vs_R": {"added": {"--recover-dev-law": "apriori"}, "removed": ["--recover-dev-bank", "--recover-dev-bank-sha256"], "changed": {}}},
    "RTL": {"p": 0.3, "law": None, "target": "logged", "rejoin_steps": None, "bank": True, "kr1_band": (KR1_LO, KR1_HI),
            "diff_vs_R": {"added": {}, "removed": ["--recover-rejoin-steps"], "changed": {"--recover-target": ("rejoin", "logged")}}},
    "RP01": {"p": 0.1, "law": None, "target": "rejoin", "rejoin_steps": "10", "bank": True, "kr1_band": (0.0833, 0.1167),
             "diff_vs_R": {"added": {}, "removed": [], "changed": {"--recover-p": ("0.3", "0.1")}}},
    #: RAPR-P01 (PREREG_RECOVER_NEXT_ADDENDUM2_RAPR.md row E): RAPR's a-priori law (no bank, no table) with RP01's dose (p 0.1, KR1 band scaled exactly as RP01's)
    "RAPRP01": {"p": 0.1, "law": "apriori", "target": "rejoin", "rejoin_steps": "10", "bank": False, "kr1_band": (0.0833, 0.1167),
                "diff_vs_R": {"added": {"--recover-dev-law": "apriori"}, "removed": ["--recover-dev-bank", "--recover-dev-bank-sha256"], "changed": {"--recover-p": ("0.3", "0.1")}}},
    #: RPT (PREREG_RECOVER_NEXT_ADDENDUM4_RPT.md): R's frozen argv (empirical TRAIN-bank law, p 0.3, rejoin target) with ONE added flag, --recover-dev-shape pertick; ``shape`` is the built / declared shape the controls read back (arms without a ``shape`` key read none)
    "RPT": {"p": 0.3, "law": None, "target": "rejoin", "rejoin_steps": "10", "bank": True, "kr1_band": (KR1_LO, KR1_HI), "shape": "pertick",
            "diff_vs_R": {"added": {"--recover-dev-shape": "pertick"}, "removed": [], "changed": {}}},
}
ARMS = tuple(ARM_SPEC)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _jd(o):
    """json default: numpy scalars / arrays to python (np.bool_ must stay a bool, never 1.0)."""
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ run table (the single source of the chain's run list)
def run_table(arm="R"):
    """The 15 rollouts in PRIORITY order (a killed chain still yields the primary contrast first).  Each: tag, role, seed, ckpt run dir, scene, feasible, reused.
    ``arm`` names the run in the R role: its rollouts are tagged <ARM>_s, KA1<ARM>_s, FZ<ARM>_s, <ARM>C_s (arm R: R_s, KA1R_s, FZR_s, RC_s, the source's tags); the C0 / C0REP / FZC0 / C0C roles are
    ``reused`` (finished rollouts, not re-rolled) when arm != R."""
    if arm not in ARM_SPEC:
        raise ValueError(f"unknown --r-arm {arm!r}; one of {ARMS}")
    tagx = {"R": lambda s: f"{arm}_s{s}", "KA1R": lambda s: f"KA1{arm}_s{s}", "FZR": lambda s: f"FZ{arm}_s{s}", "RC": lambda s: f"{arm}C_s{s}"}
    xrole = ("R", "KA1R", "FZR", "RC")

    def r(role, s, base, scene="logged", feas=None):
        x = role in xrole
        return {"tag": tagx[role](s) if x else f"{role}_s{s}", "role": role, "seed": s, "ckpt_run": f"M2b_s{s}_recft_{arm if x else base}", "scene": scene, "feasible": feas, "reused": bool(arm != "R" and not x)}
    rows = [r("R", 0, "R"), r("C0", 0, "C0"), r("KA1R", 0, "R"), r("R", 1, "R"), r("C0", 1, "C0"), r("C0REP", 0, "C0rep"), r("C0REP", 1, "C0rep"),
            r("FZR", 0, "R", "hold_now"), r("FZC0", 0, "C0", "hold_now"), r("FZR", 1, "R", "hold_now"), r("FZC0", 1, "C0", "hold_now"),
            r("RC", 0, "R", feas="kbox"), r("C0C", 0, "C0", feas="kbox"), r("RC", 1, "R", feas="kbox"), r("C0C", 1, "C0", feas="kbox")]
    return rows


def tag_of(table, role, s):
    """The tag of the (single) rollout with this role and seed."""
    t = [r["tag"] for r in table if r["role"] == role and r["seed"] == s]
    assert len(t) == 1, (role, s, t)
    return t[0]


def tag_dir(r, out_dir, reuse_dir):
    """Where a rollout lives: reused rows in ``reuse_dir`` (read only), the others in ``out_dir``."""
    return reuse_dir if r.get("reused") else out_dir


def run_flags(row):
    f = ["--smooth", "sg72", "--scene", row["scene"]] + (["--feasible", row["feasible"]] if row["feasible"] else [])
    return f


# ------------------------------------------------------------------------------------------------ pure layer (unit-checkable, no file access)
def share_mean(cells):
    """Mean share closed over cells; an undefined share (no excess to close) counts 0 (the conservative reading, as m1_cell's contract says)."""
    return float(np.mean([0.0 if c["share_closed"] is None else float(c["share_closed"]) for c in cells.values()]))


def m1_cell(xy, yp):
    """One seed's M1 cell from the M1 rows of X - C and C - P10 ({delta, lo, hi, separated}): share closed = -(X - C) / (C - P10), undefined (None) unless C - P10 > 0."""
    better = bool((xy["lo"] > 0 or xy["hi"] < 0) and xy["delta"] < 0)
    ex = float(yp["delta"])
    return {"better": better, "xy_delta": float(xy["delta"]), "yp_delta": ex, "yp_separated_worse": bool(yp["separated"] and ex > 0), "share_closed": (-float(xy["delta"]) / ex) if ex > 0 else None}


def primary_rule(cells):
    """PRIMARY (P): M1 of R - C0 separated-better at BOTH seeds AND mean share closed >= 0.5 over the two seeds."""
    n = sum(bool(c["better"]) for c in cells.values())
    mean = share_mean(cells)
    return {"n_seeds_separated_better": int(n), "mean_share_closed": mean, "share_closed": {k: c["share_closed"] for k, c in cells.items()}, "share_bar": SHARE_BAR, "PASS": bool(len(cells) == 2 and n == 2 and mean >= SHARE_BAR)}


def guards_rule(xy_by_seed):
    """GUARDS (G): ADE6 R - C0 point <= +0.0604 m at each seed AND no separated-worse non-crawl row among the 19 in R - C0.  ``xy_by_seed`` = {seed: read_xy dict (smooth_t1_read.read_xy)}."""
    per = {k: {"ade6_delta": float(v["ade6_delta"]), "ade6_within_guard": bool(v["ade6_delta"] <= ADE_GUARD_M), "non_crawl_separated_worse": list(v["non_crawl_separated_worse"])} for k, v in xy_by_seed.items()}
    return {"per_seed": per, "ade_guard_m": ADE_GUARD_M, "PASS": bool(all(p["ade6_within_guard"] and not p["non_crawl_separated_worse"] for p in per.values()))}


def floor_rule(rc0_m1_delta, rep_m1_delta):
    """FLOOR (F): at each seed |R - C0| on M1 (point) > |C0rep - C0| on M1 (point).  Both args {seed: delta}."""
    per = {k: {"abs_R_minus_C0": abs(float(rc0_m1_delta[k])), "abs_C0rep_minus_C0": abs(float(rep_m1_delta[k]))} for k in rc0_m1_delta}
    for p in per.values():
        p["exceeds_floor"] = bool(p["abs_R_minus_C0"] > p["abs_C0rep_minus_C0"])
    return {"per_seed": per, "PASS": bool(all(p["exceeds_floor"] for p in per.values()))}


def freeze_reading(p_logged, cells_f):
    """Frame-freeze reading (RECOVER-FT section 7).  ``cells_f`` = {seed: m1_cell of FZ-R - FZ-C0 against FZ-C0 - P10}.
    ROBUST: separated-better at both seeds AND mean share >= 0.5.  ARTEFACT: the logged-frame read passes P and (separated-better at fewer than 2 seeds OR mean share < 0.25).  PARTIAL: anything else."""
    n = sum(bool(c["better"]) for c in cells_f.values())
    mean = share_mean(cells_f)
    if len(cells_f) == 2 and n == 2 and mean >= SHARE_BAR:
        rd = "ROBUST"
    elif p_logged and (n < 2 or mean < ARTEFACT_BAR):
        rd = "ARTEFACT"
    else:
        rd = "PARTIAL"
    return {"reading": rd, "n_seeds_separated_better": int(n), "mean_share_closed": mean, "share_closed": {k: c["share_closed"] for k, c in cells_f.items()}, "robust_bar": SHARE_BAR, "artefact_bar": ARTEFACT_BAR,
            "P_logged_holds": bool(p_logged), "applies_to_label": None}


def label(P, G, F, fz_reading):
    """The committed outcome label table (RECOVER-FT section 6): HARMS if G fails; NOT MOVED if P fails; INSIDE-FLOOR if P and G hold and F fails; else the frame-freeze reading names it."""
    if not G:
        return "HARMS"
    if not P:
        return "NOT MOVED"
    if not F:
        return "INSIDE-FLOOR"
    return {"ROBUST": "RECOVERS", "ARTEFACT": "ARTEFACT", "PARTIAL": "PARTIAL"}[fz_reading]


def read_a_block(SR, GD, xp):
    """Read A (D1, D2, D3 over four families) of one seed from {label: row} of X - P10, plus the three-family D3 and the strategic status; ``failing_terms`` names what decides a NOT DOMINANT."""
    ra = SR.read_a(GD, xp)
    fam3 = [f for f in FAMILIES if f != "STRATEGIC"]
    ra["D3_three_families_strategic_paused"] = bool(all(ra["D3_by_family"][f] for f in fam3))
    ra["DOMINANT_three_families_strategic_paused"] = bool(ra["D1"] and ra["D2"] and ra["D3_three_families_strategic_paused"])
    ra["strategic_family"] = STRATEGIC_NOTE
    ra["strategic_rows_separated_better"] = sorted(l for l in ra["separated_better_rows"] if SR.FAMILY[l] == "STRATEGIC")
    ft = []
    if not ra["D1"]:
        ft.append("D1 (ADE6 not separated-better with point <= %.4f m; point %+.4f)" % (D1_POINT_M, xp["ADE6"]["delta"]))
    if not ra["D2"]:
        ft.append("D2 (separated-worse: %s)" % ", ".join(ra["separated_worse_rows"]))
    for f in FAMILIES:
        if not ra["D3_by_family"][f]:
            ft.append(f"D3 ({f}: no separated-better row)")
    ra["failing_terms"] = ft
    return ra


def composite_rule(rc_a, c0c_a, freeze_rd, kfz_ok):
    """COMPOSITE (PREREG_COMPOSITE_FROZEN.md section 3).  ``rc_a`` / ``c0c_a`` = {seed: read_a_block of RC / C0C vs P10}; ``freeze_rd`` = R's freeze reading dict (or None); ``kfz_ok`` = the frame-freeze controls passed.
    DOMINANT iff Read A of RC vs P10 holds at BOTH seeds.  It COUNTS only if R's frame-freeze reading is ROBUST or PARTIAL (ARTEFACT => rig artefact; not computable => UNREAD).
    Attribution: R-SPECIFIC iff C0C is DOMINANT at neither seed; RECIPE-LEVEL iff at both; MIXED otherwise."""
    dom = bool(len(rc_a) == 2 and all(v["DOMINANT"] for v in rc_a.values()))
    dom3 = bool(len(rc_a) == 2 and all(v["DOMINANT_three_families_strategic_paused"] for v in rc_a.values()))
    rd = None if freeze_rd is None else freeze_rd["reading"]
    if not dom:
        status = "NOT DOMINANT"
    elif not kfz_ok or rd is None:
        status = "DOMINANT, UNREAD (frame-freeze controls failed or reading not computable)"
    elif rd == "ARTEFACT":
        status = "DOMINANT, RIG ARTEFACT"
    else:
        status = "DOMINANT, COUNTED"
    n_c0c = sum(bool(v["DOMINANT"]) for v in c0c_a.values()) if c0c_a else None
    attr = None if n_c0c is None else ("R-SPECIFIC" if n_c0c == 0 else ("RECIPE-LEVEL" if n_c0c == len(c0c_a) else "MIXED"))
    by_seed = {k: {"DOMINANT": bool(v["DOMINANT"]), "D1": bool(v["D1"]), "D2": bool(v["D2"]), "D3": bool(v["D3"]), "D3_by_family": v["D3_by_family"], "separated_worse_rows": v["separated_worse_rows"],
                   "DOMINANT_three_families_strategic_paused": bool(v["DOMINANT_three_families_strategic_paused"]), "failing_terms": v["failing_terms"]} for k, v in rc_a.items()}
    return {"DOMINANT": dom, "DOMINANT_three_families_strategic_paused": dom3, "status": status, "counts": status == "DOMINANT, COUNTED", "R_freeze_reading": rd, "attribution": attr,
            "C0C_dominant_seeds": n_c0c, "by_seed": by_seed, "strategic_family": STRATEGIC_NOTE}


def parse_argv(tokens):
    d, i = {}, 0
    while i < len(tokens):
        t = tokens[i]
        if not t.startswith("--"):
            raise ValueError(f"argv token {t!r} is not a flag")
        if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            d[t], i = tokens[i + 1], i + 2
        else:
            d[t], i = True, i + 1
    return d


RECOVER_FLAGS = ("--recover-p", "--recover-dev-bank", "--recover-dev-bank-sha256", "--recover-dev-law", "--recover-target", "--recover-rejoin-steps", "--recover-seed")
SHAPE_FLAG = "--recover-dev-shape"      # a recovery flag of the arms whose ARM_SPEC names a ``shape`` (RPT); for every other arm, an argv that names it fails the audit (it is not one of that arm's declared levers)


def argv_diff(x, ref):
    """The exact difference of the argv ``x`` from the argv ``ref`` (both parse_argv dicts), ``--out`` ignored: {added: {flag: value}, removed: [flags], changed: {flag: (ref value, x value)}}."""
    x2, r2 = {k: v for k, v in x.items() if k != "--out"}, {k: v for k, v in ref.items() if k != "--out"}
    return {"added": {k: x2[k] for k in x2 if k not in r2}, "removed": sorted(k for k in r2 if k not in x2), "changed": {k: (r2[k], x2[k]) for k in x2 if k in r2 and r2[k] != x2[k]}}


def argv_audit(r_tokens, c0_tokens, rep_tokens, s, arm="R", ref_r_tokens=None):
    """KR2 argv audit: C0 = R minus the lever (--recover-p 0 only); C0rep = C0 with --seed 10 + s; the lever flags of R are the card's.
    ``arm`` != R: ``r_tokens`` is the ARM's argv (the R role); its lever flags are ARM_SPEC's; every RECOVER flag is still dropped before the C0 comparison (the arms differ from R in recovery flags only);
    ``ref_r_tokens`` (R's own argv) adds the exact-difference check: the arm's argv differs from R's by the declared change and nothing else (arm != R only)."""
    spec = ARM_SPEC[arm]
    lever_flags = set(RECOVER_FLAGS) | ({SHAPE_FLAG} if spec.get("shape") else set())
    R, C, CR = parse_argv(r_tokens), parse_argv(c0_tokens), parse_argv(rep_tokens)
    drop = lambda d, ks: {k: v for k, v in d.items() if k not in ks}   # noqa: E731
    tkey = f"R_target_{spec['target']}" + (f"_K{spec['rejoin_steps']}" if spec["rejoin_steps"] else "_no_rejoin_steps_flag")
    c = {"R_minus_lever_equals_C0_minus_p": drop(R, lever_flags | {"--out"}) == drop(C, {"--recover-p", "--out"}),
         "C0_recover_p_is_0": float(C.get("--recover-p", "nan")) == 0.0, "C0_names_no_other_recover_flag": not any(k in C for k in lever_flags if k != "--recover-p"),
         f"R_recover_p_{spec['p']}": float(R.get("--recover-p", "nan")) == spec["p"], tkey: bool(R.get("--recover-target") == spec["target"] and R.get("--recover-rejoin-steps") == spec["rejoin_steps"]),
         "R_recover_seed_is_s": R.get("--recover-seed") == str(s)}
    if spec["bank"]:
        c["R_bank_sha_is_the_cards"] = R.get("--recover-dev-bank-sha256") == BANK_SHA[s]
    else:
        c["R_declares_no_bank"] = bool("--recover-dev-bank" not in R and "--recover-dev-bank-sha256" not in R)
    c.update({"R_seed_is_s": R.get("--seed") == str(s),
              "C0rep_equals_C0_except_seed": drop(CR, {"--recover-p", "--out", "--seed"}) == drop(C, {"--recover-p", "--out", "--seed"}), "C0rep_seed_is_10_plus_s": CR.get("--seed") == str(10 + s),
              "C0rep_recover_p_is_0": float(CR.get("--recover-p", "nan")) == 0.0})
    if arm != "R":
        c["R_dev_law_is_as_declared"] = R.get("--recover-dev-law") == spec["law"]
        if spec.get("shape"):
            c["R_dev_shape_is_as_declared"] = R.get(SHAPE_FLAG) == spec["shape"]
        if ref_r_tokens is not None:
            d, want = argv_diff(R, parse_argv(ref_r_tokens)), spec["diff_vs_R"]
            c["R_differs_from_R_argv_only_by_the_declared_change"] = bool(d["added"] == want["added"] and d["removed"] == sorted(want["removed"]) and d["changed"] == want["changed"])
    c["PASS"] = all(c.values())
    return c


def kr1_check(sr, sc0, sc0rep, arm="R"):
    """KR1: R's n_selected / n_eligible within [0.25, 0.35]; C0 and C0rep exactly 0 selected (and recover_p 0).
    ``arm``: the band is the card's scaled with the arm's declared p (ARM_SPEC kr1_band: [0.25, 0.35] at p 0.3, [0.0833, 0.1167] at p 0.1); the selection is Bernoulli(p) over the eligible rows whatever the law
    (RecoverPerturb.apply: ``u < p``), so the band scales with p and not with the law or the target."""
    lo, hi = ARM_SPEC[arm]["kr1_band"]
    n, el = sr.get("recover_n"), sr.get("recover_eligible")
    share = (n / el) if (n is not None and el) else None
    c = {"R_share": share, "R_n": n, "R_eligible": el, "R_share_in_band": bool(share is not None and n > 0 and lo <= share <= hi),
         "C0_n": sc0.get("recover_n"), "C0_p": sc0.get("recover_p"), "C0rep_n": sc0rep.get("recover_n"), "C0rep_p": sc0rep.get("recover_p")}
    if arm != "R":
        c["arm"], c["band"], c["R_p_recorded"] = arm, [lo, hi], sr.get("recover_p")
    c["C0_zero"] = bool(c["C0_n"] == 0 and c["C0_p"] == 0 and c["C0rep_n"] == 0 and c["C0rep_p"] == 0)
    c["PASS"] = bool(c["R_share_in_band"] and c["C0_zero"] and (arm == "R" or sr.get("recover_p") == ARM_SPEC[arm]["p"]))
    return c


RECOVER_BUILT_REJOIN_STEPS = 10      # v8f_recover.RECOVER_DEFAULTS["rejoin_steps"]: what the BUILT perturber reports when the argv drops the flag (RTL) as well as when it names 10


def kr3_apriori_check(sc, s, arm):
    """KR3 for an arm whose deviation law is the a-priori one (RAPR).  The law reads no bank and builds no table (v8f_mini_train.make_recover builds a table only for ``--recover-dev-law empirical`` with a bank;
    ``info['recover']`` is set only there, ``levers['recover_table']`` is ``info['recover']['table']``), so the table / K-B3 record the bank arms carry does not exist BY DESIGN.  The control that keeps its meaning
    (the deviation source is the train data and nothing else) is the affirmative one: no bank, no table, the BUILT law is the a-priori one and the BUILT levers are the arm's declared ones."""
    spec = ARM_SPEC[arm]
    lv = sc.get("levers") or {}
    rec = lv.get("recover") or {}
    c = {"info_recover_absent": bool((sc.get("info") or {}).get("recover") is None), "levers_recover_table_is_None": bool(lv.get("recover_table") is None),
         "built_dev_law": rec.get("recover_dev_law"), "built_bank_path": rec.get("recover_dev_bank"), "built_bank_sha256": rec.get("recover_dev_bank_sha256"),
         "built_p": rec.get("recover_p"), "built_target": rec.get("recover_target"), "built_rejoin_steps": rec.get("recover_rejoin_steps"), "built_seed": rec.get("recover_seed"),
         "declared_vs_built": sc.get("declared_vs_built"), "recover_probe_live_pass_informational": ((sc.get("recover_probe") or {}).get("live_recovered") or {}).get("pass")}
    c["PASS"] = bool(c["info_recover_absent"] and c["levers_recover_table_is_None"] and c["built_dev_law"] == "apriori" and c["built_bank_path"] is None and c["built_bank_sha256"] is None
                     and c["built_p"] == spec["p"] and c["built_target"] == spec["target"] and c["built_rejoin_steps"] == RECOVER_BUILT_REJOIN_STEPS and c["built_seed"] == s)
    return c


def kr3_check(sc, s, arm="R"):
    """KR3: the table's build summary exists and its records are all TRAIN clips (K-B3: zero overlap with eval-141), from the gate's selfcheck.json of R.
    ``arm`` != R: a bank arm (RTL, RP01) additionally has its BUILT levers read back against its declared ones (p, empirical law, target, seed); the a-priori arm (RAPR) reads kr3_apriori_check."""
    if not ARM_SPEC[arm]["bank"]:
        return kr3_apriori_check(sc, s, arm)
    rec = ((sc.get("info") or {}).get("recover") or {})
    tb, kb3 = rec.get("table") or {}, rec.get("kb3") or {}
    c = {"table_n": tb.get("n"), "table_content_sha256": tb.get("content_sha256"), "bank_sha256": rec.get("bank_sha256"), "bank_sha_is_the_cards": rec.get("bank_sha256") == BANK_SHA[s],
         "overlap_with_eval": kb3.get("overlap_with_eval"), "n_bank_clips": kb3.get("n_bank_clips"), "n_train_clips": kb3.get("n_train_clips"), "n_eval_clips_checked": kb3.get("n_eval_clips_checked"),
         "share_logged_v_3_8": tb.get("share_logged_v_3_8")}
    c["PASS"] = bool(tb.get("n") and tb.get("content_sha256") and c["bank_sha_is_the_cards"] and kb3.get("overlap_with_eval") == 0 and c["n_bank_clips"] == c["n_train_clips"] and c["n_eval_clips_checked"] == 141)
    if arm != "R":
        spec, bl = ARM_SPEC[arm], ((sc.get("levers") or {}).get("recover") or {})
        c["built_levers"] = {k: bl.get(k) for k in ("recover_p", "recover_dev_law", "recover_target", "recover_rejoin_steps", "recover_seed", "recover_dev_bank_sha256")}
        c["built_levers_as_declared"] = bool(bl.get("recover_p") == spec["p"] and bl.get("recover_dev_law") == "empirical" and bl.get("recover_target") == spec["target"]
                                             and bl.get("recover_rejoin_steps") == RECOVER_BUILT_REJOIN_STEPS and bl.get("recover_seed") == s and bl.get("recover_dev_bank_sha256") == BANK_SHA[s])
        shape_ok = True
        if spec.get("shape"):      # RPT: the BUILT shape, the build record's shape and the table's per-tick block (ticks, kept / dropped counts); the NOW-column identity is REPORTED, not decided
            pt, ct = tb.get("per_tick") or {}, tb.get("counts") or {}
            now = {k: (pt.get(f"mean_abs_{k}_t") or [None])[-1] == tb.get(f"mean_abs_{k}") for k in ("dv", "da", "dom")}
            c.update({"built_dev_shape": bl.get("recover_dev_shape"), "info_recover_dev_shape": rec.get("dev_shape"), "per_tick_ticks": pt.get("ticks"), "table_counts": ct, "now_column_mean_abs_equals_constant_column_REPORTED": now})
            shape_ok = bool(bl.get("recover_dev_shape") == spec["shape"] and rec.get("dev_shape") == spec["shape"] and isinstance(pt.get("ticks"), int) and pt["ticks"] > 0
                            and isinstance(ct.get("dropped_per_tick_row_unreal"), int) and isinstance(ct.get("kept"), int) and ct["kept"] > 0)
            c["dev_shape_built_and_recorded_as_declared"] = shape_ok
        c["PASS"] = bool(c["PASS"] and c["built_levers_as_declared"] and shape_ok)
    return c


def kp1_check(summ, ref):
    """KP1 parity: a run's train_sha256 and train_window_list_sha256 equal DAG-OACC's at the same seed."""
    c = {k: {"run": summ.get(k), "dagoacc": ref.get(k), "equal": bool(summ.get(k) is not None and summ.get(k) == ref.get(k))} for k in ("train_sha256", "train_window_list_sha256")}
    c["steps"], c["done"] = summ.get("steps"), summ.get("done")
    c["PASS"] = bool(all(v["equal"] for k, v in c.items() if isinstance(v, dict)) and summ.get("done") is True and summ.get("steps") == 3000)
    return c


# ------------------------------------------------------------------------------------------------ heavy layer
LR = SR = FZ = GD = None


def heavy_init():
    """Load the reused readers BY FILE PATH after their pins hold; GD = gapdecomp (pure numpy).  No stack root is put on sys.path."""
    global LR, SR, FZ, GD
    for p, want in PINS.items():
        got = sha256_file(p)
        if got != want:
            raise SystemExit(f"REFUSED: {p} is {got}, not the pinned {want}")
    LR = _load("histswap_lf_read", f"{LF_PK}/code/histswap_lf_read.py")
    SR = LR.SR
    FZ = _load("framefreeze_read", f"{FZ_PK}/code/framefreeze_read.py")
    GD = _load("gapdecomp", f"{GD_PK}/gapdecomp.py")


def rows_of(ua, va, eid, cls, n_boot, full_label=None):
    """The 19 rows of the contrast mean(ua - va) by gapdecomp.decompose.  ``ua`` / ``va`` = lists of 19 per-window arrays.  -> ({label: {delta, lo, hi, separated}}, full decompose of ``full_label``)."""
    rows, full = {}, None
    for i, lab in enumerate(GD.LABELS):
        r = GD.decompose(ua[i], va[i], eid, cls, GD.POOL[lab], n_boot)
        rows[lab] = {q: r[q] for q in ("delta", "lo", "hi", "separated")}
        if lab == full_label:
            full = r
    return rows, full


def Aof(z):
    return [z[f"a_{i}"] for i in range(len(GD.LABELS))]


def Bof(z):
    return [z[f"b_{i}"] for i in range(len(GD.LABELS))]


def read_arrays(path):
    z = np.load(path, allow_pickle=True)
    assert tuple(str(q) for q in z["labels"]) == GD.LABELS, "array labels differ from the instrument's"
    return {k: z[k] for k in z.files}


def evaluate(arrays, n_boot, log=print, freeze_usable=True, freeze_refusal=None):
    """The whole rule set on per-window arrays.  ``arrays`` = {seed: {role: array dict}} with roles R, C0 (required), C0REP, FZR, FZC0, RC, C0C (each optional; a rule whose roles are absent says NOT COMPUTED).
    ``freeze_usable`` False (a KFZ control failed): the frame-freeze contrast is REFUSED (RECOVER-FT section 7), everything else is still read; the label and the composite then say UNREAD.
    Returns (result dict, per-seed M1 cells)."""
    res = {"seeds": {}}
    cells_p, cells_f, xy, rc0_m1, rep_m1, ra_r, ra_rc, ra_c0c = {}, {}, {}, {}, {}, {}, {}, {}
    for s in SEEDS:
        k, ar = f"s{s}", arrays[s]
        eid = ar["R"]["eid"]
        cls, _ = GD.classify(ar["R"]["v0"], ar["R"]["g"])
        blk = {}
        blk["R_minus_C0"], full = rows_of(Aof(ar["R"]), Aof(ar["C0"]), eid, cls, n_boot, M1)
        blk["M1_per_class_split_R_minus_C0"] = full
        blk["C0_minus_P10"], _ = rows_of(Aof(ar["C0"]), Bof(ar["C0"]), eid, cls, n_boot)
        blk["R_minus_P10"], _ = rows_of(Aof(ar["R"]), Bof(ar["R"]), eid, cls, n_boot)
        cells_p[k] = m1_cell(blk["R_minus_C0"][M1], blk["C0_minus_P10"][M1])
        xy[k] = SR.read_xy(GD, blk["R_minus_C0"])
        ra_r[k] = read_a_block(SR, GD, blk["R_minus_P10"])
        rc0_m1[k] = blk["R_minus_C0"][M1]["delta"]
        if "C0REP" in ar:
            blk["C0REP_minus_C0"], _ = rows_of(Aof(ar["C0REP"]), Aof(ar["C0"]), eid, cls, n_boot)
            rep_m1[k] = blk["C0REP_minus_C0"][M1]["delta"]
        if freeze_usable and "FZR" in ar and "FZC0" in ar:
            blk["FZR_minus_FZC0"], _ = rows_of(Aof(ar["FZR"]), Aof(ar["FZC0"]), eid, cls, n_boot)
            blk["FZC0_minus_P10"], _ = rows_of(Aof(ar["FZC0"]), Bof(ar["FZC0"]), eid, cls, n_boot)
            cells_f[k] = m1_cell(blk["FZR_minus_FZC0"][M1], blk["FZC0_minus_P10"][M1])
        if "RC" in ar:
            blk["RC_minus_P10"], _ = rows_of(Aof(ar["RC"]), Bof(ar["RC"]), eid, cls, n_boot)
            blk["RC_minus_R"], _ = rows_of(Aof(ar["RC"]), Aof(ar["R"]), eid, cls, n_boot)
            ra_rc[k] = read_a_block(SR, GD, blk["RC_minus_P10"])
        if "C0C" in ar:
            blk["C0C_minus_P10"], _ = rows_of(Aof(ar["C0C"]), Bof(ar["C0C"]), eid, cls, n_boot)
            blk["C0C_minus_C0"], _ = rows_of(Aof(ar["C0C"]), Aof(ar["C0"]), eid, cls, n_boot)
            ra_c0c[k] = read_a_block(SR, GD, blk["C0C_minus_P10"])
        blk["read_xy_R_minus_C0"], blk["m1_cell_logged"] = xy[k], cells_p[k]
        res["seeds"][k] = blk
        log(k, "M1 R-C0 %+.4f [%+.4f, %+.4f] sep %s | C0-P10 %+.4f | share closed %s | ADE6 R-C0 %+.4f" % (
            blk["R_minus_C0"][M1]["delta"], blk["R_minus_C0"][M1]["lo"], blk["R_minus_C0"][M1]["hi"], blk["R_minus_C0"][M1]["separated"], blk["C0_minus_P10"][M1]["delta"], cells_p[k]["share_closed"], blk["R_minus_C0"]["ADE6"]["delta"]))
    res["primary"] = primary_rule(cells_p)
    res["guards"] = guards_rule(xy)
    res["floor"] = floor_rule(rc0_m1, rep_m1) if len(rep_m1) == len(SEEDS) else {"PASS": None, "note": "NOT COMPUTED: the C0rep role is absent"}
    res["read_a_R_vs_P10"] = {"by_seed": ra_r, "DOMINANT_both_seeds": bool(all(v["DOMINANT"] for v in ra_r.values())),
                              "DOMINANT_three_families_strategic_paused_both_seeds": bool(all(v["DOMINANT_three_families_strategic_paused"] for v in ra_r.values())), "reported_beside_the_label_not_part_of_it": True}
    fz = None
    if len(cells_f) == len(SEEDS):
        fz = freeze_reading(res["primary"]["PASS"], cells_f)
        fz["m1_cells_freeze"] = cells_f
        fz["freeze_cells_note"] = "G_F = M1 share of FZ-R - FZ-C0 against FZ-C0 - P10 (each frame mode against its own excess); G_L = primary.share_closed"
    res["frame_freeze"] = fz or ({"reading": None, "refused": f"frame-freeze controls failed: {freeze_refusal}; the contrast is REFUSED (RECOVER-FT section 7), not interpreted"} if not freeze_usable
                                 else {"reading": None, "note": "NOT COMPUTED: FZR / FZC0 roles are absent"})
    pgf = bool(res["primary"]["PASS"] and res["guards"]["PASS"] and res["floor"]["PASS"])
    if fz is not None:
        fz["applies_to_label"] = pgf
    if res["floor"]["PASS"] is None:
        res["label"] = None
    elif pgf and fz is None:
        res["label"] = "UNREAD (P, G and F hold but the frame-freeze contrast is absent or REFUSED)"
    else:
        res["label"] = label(res["primary"]["PASS"], res["guards"]["PASS"], bool(res["floor"]["PASS"]), (fz or {}).get("reading") or "PARTIAL")
    if len(ra_rc) == len(SEEDS):
        res["composite"] = composite_rule(ra_rc, ra_c0c or None, fz, bool(freeze_usable))
        res["composite"]["RC_minus_R_is_kboxs_own_effect"] = {k: {l: res["seeds"][k]["RC_minus_R"][l] for l in ("ADE6", M1, "heading MAE", "curvature MAE", "yaw-rate MAE", "S-POS@30")} for k in res["seeds"]}
        res["composite"]["read_a_C0C_vs_P10"] = ra_c0c or None
        if fz is not None and not fz["P_logged_holds"] and fz["reading"] == "PARTIAL":
            res["composite"]["freeze_reading_note"] = "PARTIAL by default: R's own PRIMARY did not hold, so ARTEFACT is not defined by the RECOVER-FT card; no frame-freeze arm of RC exists"
    else:
        res["composite"] = {"status": None, "note": "NOT COMPUTED: the RC role is absent"}
    res["strategic_family"] = STRATEGIC_NOTE
    res["variance_questions"] = VARIANCE
    return res, cells_p


# ------------------------------------------------------------------------------------------------ file controls (production)
def run_header(out_dir, tag):
    return json.load(open(f"{out_dir}/{tag}/t1v8f_{tag}.json", encoding="utf-8"))["t1v8f"]


def kk2_check(t, tol=TOL):
    """KK2: the issued plan equals kbox(S(plan_raw)) at every tick: plan_s = P + SG(plan_raw - P) (7, 2, interp) and plan = project_feasible(kbox, plan_s).  feasible_decode loaded by FILE PATH."""
    import torch
    from scipy.signal import savgol_filter
    FD = _load("feasible_decode_pinned_tree", FD_PATH)
    s_rec = (t["P"] + savgol_filter(t["plan_raw"] - t["P"], 7, 2, axis=1, mode="interp")).astype(np.float32).astype(np.float64)
    d_s = float(np.abs(s_rec - t["plan_s"]).max())
    p = np.asarray(t["plan_s"], np.float64)
    full = torch.from_numpy(np.concatenate([np.zeros_like(p[:, :1]), p], 1))
    out = FD.project_feasible(full, dt=0.2, a_max=float("inf"), kappa_max=FD.KAPPA_MAX_1PM, mu=None, clamp_entry=False)[:, 1:].numpy().astype(np.float32).astype(np.float64)
    d_l = float(np.abs(out - t["plan"]).max())
    return {"plan_s_vs_P_plus_SG_max_abs_m": d_s, "plan_vs_kbox_of_plan_s_max_abs_m": d_l, "kbox_changed_plan_max_abs_m": float(np.abs(t["plan"] - t["plan_s"]).max()), "tol_m": tol, "PASS": bool(d_s <= tol and d_l <= tol)}


def extract(tag, seed, out_dir, arr_dir):
    dst = f"{arr_dir}/{tag}.npz"
    if os.path.exists(dst):
        return dst
    os.makedirs(arr_dir, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    cmd = [PY, "-I", f"{SMT}/code/extract_smooth.py", "--arm", "CTRL", "--seed", str(seed), "--out", dst, "--dump-dir", f"{out_dir}/{tag}/dump_{tag}",
           "--ticks", f"{out_dir}/{tag}/ticks/ticks_{tag}.npz", "--work-dir", f"{WORK}/{tag}", "--floor-guard-m", str(FL_GROSS_M)]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    os.makedirs(f"{PK}/logs", exist_ok=True)
    open(f"{PK}/logs/extract_{tag}.log", "w", encoding="utf-8", newline="\n").write(r.stdout + r.stderr)
    if r.returncode:
        raise SystemExit(f"extract {tag} failed rc={r.returncode}: {r.stderr[-400:]}")
    return dst


def file_controls(a, table):
    """Controls computed before any row.  Returns ({name: block with PASS}, context dict).  Every block carries PASS; a failure REFUSES the read (the frozen cards)."""
    C, ctx = {}, {}
    rd = a.recover_dir
    arm = a.r_arm
    by_tag = {r["tag"]: r for r in table}
    first = open(f"{RR_PK}/FROZEN_SHA.txt", encoding="utf-8").readline().split()
    C["KCARD_composite_card_hashes_to_FROZEN_SHA"] = {"sha256": sha256_file(f"{RR_PK}/PREREG_COMPOSITE_FROZEN.md"), "frozen": first[2], "PASS": sha256_file(f"{RR_PK}/PREREG_COMPOSITE_FROZEN.md") == first[2]}
    ft = open(f"{FT_PK}/FROZEN_SHA.txt", encoding="utf-8").readline().split()
    C["KCARD_recover_ft_card_hashes_to_FROZEN_SHA"] = {"sha256": sha256_file(f"{FT_PK}/{ft[0]}"), "frozen": ft[2], "PASS": sha256_file(f"{FT_PK}/{ft[0]}") == ft[2]}
    if arm != "R":
        nx = open(f"{NX_PK}/FROZEN_SHA.txt", encoding="utf-8").readline().split()
        C["KCARD_recover_next_card_hashes_to_FROZEN_SHA"] = {"sha256": sha256_file(f"{NX_PK}/{nx[0]}"), "frozen": nx[2], "PASS": sha256_file(f"{NX_PK}/{nx[0]}") == nx[2]}
    summ, argv, ck = {}, {}, {}
    rdir = lambda s, base: f"{rd}/M2b_s{s}_recft_{arm if base == 'R' else base}"   # noqa: E731  (the R role is the ARM's run; C0 / C0rep are the RECOVER-FT comparators)
    for s in SEEDS:
        for base in ("R", "C0", "C0rep"):
            d = rdir(s, base)
            summ[(s, base)] = json.load(open(f"{d}/summary.json", encoding="utf-8"))
            argv[(s, base)] = json.load(open(f"{d}/argv.json", encoding="utf-8"))["argv"]
            ck[(s, base)] = sha256_file(f"{d}/ckpt_last.pt")
    C["KP1_parity_train_and_window_list_sha_equal_DAGOACC"] = {f"s{s}:{b}": kp1_check(summ[(s, b)], json.load(open(f"{DAGOACC_DIR}/M2b_s{s}_dagoacc/summary.json", encoding="utf-8"))) for s in SEEDS for b in ("R", "C0", "C0rep")}
    C["KP1_parity_train_and_window_list_sha_equal_DAGOACC"]["PASS"] = all(v["PASS"] for v in C["KP1_parity_train_and_window_list_sha_equal_DAGOACC"].values())
    C["KR1_selected_over_eligible_in_band_for_R_zero_for_C0"] = {f"s{s}": kr1_check(summ[(s, "R")], summ[(s, "C0")], summ[(s, "C0rep")], arm) for s in SEEDS}
    C["KR1_selected_over_eligible_in_band_for_R_zero_for_C0"]["PASS"] = all(v["PASS"] for v in C["KR1_selected_over_eligible_in_band_for_R_zero_for_C0"].values())
    kr2 = {}
    for s in SEEDS:
        init = sha256_file(f"{DAGOACC_DIR}/M2b_s{s}_dagoacc/ckpt_last.pt")
        c = {"ckpt_sha256": {b: ck[(s, b)] for b in ("R", "C0", "C0rep")}, "init_sha256": init, "init_prefix_is_the_cards": init.startswith(INIT_PREFIX[s]),
             "R_differs_from_C0": ck[(s, "R")] != ck[(s, "C0")], "R_differs_from_C0rep": ck[(s, "R")] != ck[(s, "C0rep")], "C0_differs_from_C0rep": ck[(s, "C0")] != ck[(s, "C0rep")],
             "all_differ_from_init": all(ck[(s, b)] != init for b in ("R", "C0", "C0rep")),
             "argv_audit": argv_audit(argv[(s, "R")], argv[(s, "C0")], argv[(s, "C0rep")], s, arm, (json.load(open(f"{rd}/M2b_s{s}_recft_R/argv.json", encoding="utf-8"))["argv"] if arm != "R" else None))}
        c["PASS"] = bool(c["init_prefix_is_the_cards"] and c["R_differs_from_C0"] and c["R_differs_from_C0rep"] and c["C0_differs_from_C0rep"] and c["all_differ_from_init"] and c["argv_audit"]["PASS"])
        if arm != "R":    # a mis-pull that filed R's checkpoint under the arm's name would otherwise read as an arm
            c["R_role_run_differs_from_the_RECOVER_FT_R_run"] = ck[(s, "R")] != sha256_file(f"{rd}/M2b_s{s}_recft_R/ckpt_last.pt")
            c["PASS"] = bool(c["PASS"] and c["R_role_run_differs_from_the_RECOVER_FT_R_run"])
        kr2[f"s{s}"] = c
    kr2["PASS"] = all(v["PASS"] for v in kr2.values())
    C["KR2_R_ckpt_differs_from_C0_and_C0_recipe_is_R_minus_the_lever"] = kr2
    kr3 = {f"s{s}": kr3_check(json.load(open(f"{rdir(s, 'R')}/gate_selfcheck.json", encoding="utf-8")), s, arm) for s in SEEDS}
    kr3["PASS"] = all(v["PASS"] for v in kr3.values())
    C["KR3_table_build_summary_exists_all_records_TRAIN" if ARM_SPEC[arm]["bank"] else "KR3_apriori_law_builds_no_table_and_reads_no_bank"] = kr3
    # markers: each box-2 RUN_<tag>.done sha (when it carries one) equals the pulled checkpoint
    mk = {}
    for s in SEEDS:
        for tg, b, mdir in ((f"{arm}_{s}", "R", "recft_markers" if arm == "R" else "rnext_markers"), (f"C0_{s}", "C0", "recft_markers"), (f"C0rep_{s}", "C0rep", "recft_markers")):
            p = f"{rd}/{mdir}/RUN_{tg}.done"
            want = None
            if os.path.exists(p):
                try:
                    want = json.load(open(p, encoding="utf-8")).get("evidence", {}).get("ckpt_sha256")
                except Exception:
                    want = None
            mk[tg] = {"marker_sha256": want, "pulled_sha256": ck[(s, b)], "PASS": bool(want is None or want == ck[(s, b)]), "note": "UNVERIFIED: the marker carries no sha" if want is None else "marker equals pulled"}
    mk["PASS"] = all(v["PASS"] for v in mk.values())
    C["KMARK_box2_marker_sha_equals_pulled_ckpt"] = mk
    # run-level controls on the rollouts
    out = a.out_dir
    hd, tk, flag = {}, {}, {}
    for r in table:
        t = r["tag"]
        base_dir = tag_dir(r, a.out_dir, a.reuse_dir)
        hd[t] = run_header(base_dir, t)
        tk[t] = LR.tickload(f"{base_dir}/{t}/ticks/ticks_{t}.npz")
        c = {"smooth": hd[t].get("smooth", {}).get("mode"), "scene": hd[t].get("scene"), "feasible": (hd[t].get("feasible") or {}).get("mode"), "histswap_absent": "histswap" not in hd[t], "replan_steps": hd[t].get("replan_steps"),
             "nav": hd[t].get("nav"), "limit": hd[t].get("limit"), "schedule": hd[t].get("schedule"), "n_windows": hd[t].get("n_windows"),
             "ckpt_is_the_pulled_one": os.path.normpath(str(hd[t].get("ckpt"))) == os.path.normpath(f"{rd}/{r['ckpt_run']}/ckpt_last.pt")}
        c["PASS"] = bool(c["smooth"] == "sg72" and c["scene"] == r["scene"] and c["feasible"] == r["feasible"] and c["histswap_absent"] and c["replan_steps"] == 10 and c["nav"] == "deadreckon" and c["limit"] == "causal"
                         and c["schedule"] == "periodic" and c["n_windows"] == 2399 and c["ckpt_is_the_pulled_one"])
        flag[t] = c
    flag["PASS"] = all(v["PASS"] for v in flag.values())
    C["KC1_flags_reached_the_runs"] = flag
    ec = {t: LR.exec_chain_dev(x) for t, x in tk.items()}
    C["KC3_plan_committed_to_the_executed_state"] = {t: {"max_dev_m": float(e.max()), "frac_within_tol": float((e <= EXEC_TOL_M).mean()), "PASS": bool(len(e) and (e <= EXEC_TOL_M).mean() >= 0.999)} for t, e in ec.items()}
    C["KC3_plan_committed_to_the_executed_state"]["PASS"] = all(v["PASS"] for v in C["KC3_plan_committed_to_the_executed_state"].values())
    # KA1: bit-identity rerun of R_s0 (the planner is deterministic; this reads the thread-order floor)
    ka, r0 = tag_of(table, "KA1R", 0), tag_of(table, "R", 0)
    _, d1 = SR.load_dump(f"{out}/{ka}/dump_{ka}")
    _, d0 = SR.load_dump(f"{out}/{r0}/dump_{r0}")
    same_d, dev_d, _, bad_d, only_d = SR.cmp_arrays(d1, d0)
    same_t, dev_t, _, bad_t, only_t = SR.cmp_arrays(tk[ka], tk[r0])
    C["KA1_bit_identity_rerun_of_R_s0"] = {"dump_bit_identical": same_d, "dump_max_abs_dev": dev_d, "ticks_bit_identical": same_t, "ticks_max_abs_dev": dev_t, "differing_keys": bad_d + bad_t, "key_set_diff": only_d + only_t,
                                           "PASS": bool((same_d and same_t) or (dev_d <= TOL and dev_t <= TOL and not only_d and not only_t))}
    # frame-freeze controls on FZR / FZC0 (KFZ1 receipt, KFZ3 analogue: the freeze reached the network, against the logged-frame run of the same checkpoint)
    k1, k3 = {}, {}
    for s in SEEDS:
        for fz, lg in ((tag_of(table, "FZR", s), tag_of(table, "R", s)), (tag_of(table, "FZC0", s), tag_of(table, "C0", s))):
            k1[fz] = FZ.freeze_receipt(hd[fz], len(tk[fz]["tick"]))
            k3[fz] = FZ.kfz3(tk[fz], tk[lg])
    k1["PASS"] = all(v["PASS"] for v in k1.values())
    k3["PASS"] = all(v["PASS"] for v in k3.values())
    C["KFZ1_production_receipt_every_item_FZR_FZC0"] = k1
    C["KFZ3_the_freeze_reached_the_network_FZR_FZC0"] = k3
    C["KFZ2_unit_tests_of_the_freeze_tool"] = FZ.pytest_verdict(f"{FZ_PK}/logs/pytest_framefreeze.log")
    fzc = json.load(open(f"{FZ_PK}/raw/FZ_CONTROLS.json", encoding="utf-8"))
    C["KFZ4_FZ_HLVL_keeps_its_closure_read_from_FZ_CONTROLS"] = {"kfz4": fzc["kfz4"], "recover_ft_contrast_usable": fzc["recover_ft_contrast_usable"], "failed_controls": fzc["failed_controls"],
                                                                 "fz_controls_sha256": sha256_file(f"{FZ_PK}/raw/FZ_CONTROLS.json"), "PASS": bool(fzc["kfz4"]["PASS"] and fzc["recover_ft_contrast_usable"] and not fzc["failed_controls"])}
    kk1 = {t: {"feasible": flag[t]["feasible"], "smooth": flag[t]["smooth"], "PASS": bool(flag[t]["feasible"] == "kbox" and flag[t]["smooth"] == "sg72")} for t in tk if by_tag[t]["role"] in ("RC", "C0C")}
    kk1["PASS"] = all(v["PASS"] for v in kk1.values())
    C["KK1_RC_C0C_headers_carry_kbox_and_sg72"] = kk1
    kk2 = {t: kk2_check(tk[t]) for t in tk if by_tag[t]["role"] in ("RC", "C0C")}
    kk2["PASS"] = all(v["PASS"] for v in kk2.values())
    C["KK2_issued_plan_is_kbox_of_S_of_plan_raw"] = kk2
    if arm != "R":      # the reused rollouts were rolled by an earlier chain: their checkpoint and tool must be the ones the pulled files and the pinned tool are today
        tool_pin = [ln.split()[-1] for ln in open(f"{FZ_PK}/PINS.txt", encoding="utf-8") if ln.startswith(TOOL_PIN_LINE)]
        ru = {}
        for r in table:
            m = json.load(open(f"{tag_dir(r, a.out_dir, a.reuse_dir)}/done_{r['tag']}.txt", encoding="utf-8"))
            want_ck = sha256_file(f"{rd}/{r['ckpt_run']}/ckpt_last.pt")
            ru[r["tag"]] = {"reused": r["reused"], "done_ckpt_sha256": m.get("ckpt_sha256"), "pulled_ckpt_sha256": want_ck, "done_tool_sha256": m.get("tool_sha256"), "pinned_tool_sha256": tool_pin[0] if tool_pin else None,
                            "PASS": bool(m.get("ckpt_sha256") == want_ck and tool_pin and m.get("tool_sha256") == tool_pin[0])}
        ru["PASS"] = all(v["PASS"] for v in ru.values())
        C["KREUSE_every_rollout_done_marker_has_the_pulled_ckpt_sha_and_the_pinned_tool_sha"] = ru
    ctx["ticks"], ctx["dumps_seeds"] = tk, None
    return C, ctx


def array_controls(a, table, arr_dir, workers):
    """KB1 / KF on the extracted arrays and dumps (after extraction): windows equal the banked CTRL's, the P10 side equals the banked b, and the floor equals R's within TOL / the gross guard."""
    jobs = [(r["tag"], r["seed"], tag_dir(r, a.out_dir, a.reuse_dir)) for r in table]
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(lambda j: extract(j[0], j[1], j[2], arr_dir), jobs))
    C = {"KB1": {}, "KF": {}}
    arrays = {s: {} for s in SEEDS}
    dref_fl = {s: SR.load_dump(f"{a.out_dir}/{tag_of(table, 'R', s)}/dump_{tag_of(table, 'R', s)}")[1]["fl"] for s in SEEDS}
    for r in table:
        t, s = r["tag"], r["seed"]
        z = read_arrays(f"{arr_dir}/{t}.npz")
        zl = np.load(f"{GD_PK}/raw/arrays/CTRL_s{s}.npz", allow_pickle=True)
        eq = {l: bool(np.array_equal(z[f"b_{i}"], zl[f"b_{i}"], equal_nan=True)) for i, l in enumerate(GD.LABELS) if not l.startswith("S-POS")}
        win = bool(all(np.array_equal(z[q], zl[q]) for q in ("eid", "v0", "g", "ws", "clip_index")))
        C["KB1"][t] = {"windows_equal_landed": win, "n_rows_b_equal": sum(eq.values()), "n_rows": len(eq), "failed": [q for q, v in eq.items() if not v], "PASS": bool(win and all(eq.values()))}
        if r["role"] in ("R", "C0", "C0REP", "FZR", "FZC0", "RC", "C0C"):
            arrays[s][r["role"]] = z
        fl = SR.load_dump(f"{tag_dir(r, a.out_dir, a.reuse_dir)}/{t}/dump_{t}")[1]["fl"]
        C["KF"][t] = {"floor_vs_R_max_abs_m": float(np.abs(fl - dref_fl[s]).max()), "PASS": bool(np.abs(fl - dref_fl[s]).max() <= TOL)}
    C["KB1"]["PASS"] = all(v["PASS"] for v in C["KB1"].values())
    C["KF"]["PASS"] = all(v["PASS"] for v in C["KF"].values())
    return {"KB1_P10_side_equals_landed_banked_b": C["KB1"], "KF_floor_equal_across_runs": C["KF"]}, arrays


# ------------------------------------------------------------------------------------------------ known-answer modes (EXISTING arrays; no file controls)
def kat_arrays(mode):
    """kat1: R := C0 := the DAG-OACC SREP runs (C0rep := the banked REP runs, so the floor is defined).  kat2: R := HISTSWAP's logged-history run, C0 := SREP.  kat3: R := SREP, RC := FEAS2's KBOX runs."""
    srep = lambda s: f"{KAP_PK}/raw/arrays/SREP_DAGOACC_s{s}.npz"   # noqa: E731
    out = {}
    for s in SEEDS:
        if mode == "kat1":
            out[s] = {"R": read_arrays(srep(s)), "C0": read_arrays(srep(s)), "C0REP": read_arrays(f"{GD_PK}/raw/arrays/REP_s{s}.npz")}
        elif mode == "kat2":
            out[s] = {"R": read_arrays(f"{DL}/histswap/raw/arrays/HSW_DAGOACC_s{s}.npz"), "C0": read_arrays(srep(s)), "C0REP": read_arrays(f"{GD_PK}/raw/arrays/REP_s{s}.npz")}
        elif mode == "kat3":
            out[s] = {"R": read_arrays(srep(s)), "C0": read_arrays(srep(s)), "C0REP": read_arrays(f"{GD_PK}/raw/arrays/REP_s{s}.npz"), "RC": read_arrays(f"{DL}/feas2_t1/raw/arrays/KBOX_DAGOACC_s{s}.npz"),
                      "C0C": read_arrays(f"{DL}/feas2_t1/raw/arrays/KBOX_DAGOACC_s{s}.npz")}
        else:
            raise ValueError(mode)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="production", choices=("production", "kat1", "kat2", "kat3"))
    ap.add_argument("--r-arm", default="R", choices=ARMS, help="the run in the R role: M2b_s{s}_recft_<ARM> (default R = the RECOVER-FT read itself); C0 / C0REP / FZC0 / C0C are then REUSED from --reuse-dir")
    ap.add_argument("--out-dir", default=None, help=f"where the R-role rollouts live (default: {REUSE_DEFAULT} for arm R, {RR_SCRATCH}/rnext/out_<ARM> otherwise)")
    ap.add_argument("--reuse-dir", default=REUSE_DEFAULT, help="the finished C0 / C0REP / FZC0 / C0C rollouts (arm != R); read only")
    ap.add_argument("--work-dir", default=WORK_DEFAULT, help="extractor work root (a directory on C:, never F:)")
    ap.add_argument("--raw", default=f"{PK}/raw")
    ap.add_argument("--recover-dir", default=RECOVER_DIR)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--extract-workers", type=int, default=2, help="parallel single-thread extractors (free RAM on the dev box can be ~4 GB: keep small)")
    ap.add_argument("--list-runs", action="store_true", help="print the 15-rollout table (tab separated: tag seed ckpt_run scene feasible extra_flags) and exit; the chain reads this")
    ap.add_argument("--dry-run", action="store_true", help="print the plan (runs, controls, outputs) and exit; touches no data")
    a = ap.parse_args(argv)
    global WORK
    WORK = f"{a.work_dir}/rig"
    if a.out_dir is None:
        a.out_dir = REUSE_DEFAULT if a.r_arm == "R" else f"{RR_SCRATCH}/rnext/out_{a.r_arm}"
    table = run_table(a.r_arm)
    if a.list_runs:      # the rollouts the chain must roll (reused ones are not listed)
        for r in table:
            if not r["reused"]:
                print("\t".join([r["tag"], str(r["seed"]), r["ckpt_run"], r["scene"], r["feasible"] or "-", " ".join(run_flags(r))]))
        return 0
    if a.dry_run:
        print(f"recover_read dry run: mode={a.mode}  r_arm={a.r_arm}  n_runs={len(table)}  out_dir={a.out_dir}  reuse_dir={a.reuse_dir}  raw={a.raw}")
        for r in table:
            print(f"  {r['tag']:<10} seed {r['seed']} ckpt {r['ckpt_run']:<20} {' '.join(run_flags(r))}{'   [REUSED from reuse_dir]' if r['reused'] else ''}")
        print("  then: file controls (KCARD KP1 KR1 KR2 KR3 KMARK KC1 KC3 KA1 KFZ1 KFZ2 KFZ3 KFZ4 KK1 KK2), extraction (KB1 KF), rows, rules -> RECOVER_READ_<ARM>.json (arm != R adds KREUSE and the RECOVER-NEXT KCARD)")
        return 0
    heavy_init()
    os.makedirs(a.raw, exist_ok=True)
    res = {"hypothesis": ("H-RECOVER-FT-1 (primary / guards / floor / Read A / frame-freeze) + H-RECOVER-COMPOSITE-1" if a.r_arm == "R" else
                          f"RECOVER-NEXT card (PREREG_RECOVER_NEXT_FROZEN.md) section 3: the RECOVER-FT section 6 / 7 rules and the COMPOSITE rules with R := {a.r_arm}"), "tier": "T1-kin", "evidence_class": "MEASURED (ours; artifacts in MANIFEST.txt) once mode=production",
           "mode": a.mode, "estimator": f"paired episode-cluster bootstrap, B={a.n_boot}, seed 0 (taniteval.ci draw stream via gapdecomp.decompose); answers another draw of EPISODES only",
           "run_clock": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "cards": {"composite": sha256_file(f"{RR_PK}/PREREG_COMPOSITE_FROZEN.md"), "recover_ft": sha256_file(f"{FT_PK}/PREREG_RECOVER_FT_FROZEN.md"),
           "recover_ft_amendment_1": sha256_file(f"{FT_PK}/AMENDMENT_1.md")}, "reader_sha256": sha256_file(f"{HERE}/recover_read.py"), "pins": PINS}
    if a.r_arm != "R":
        res["cards"]["recover_next"] = sha256_file(f"{NX_PK}/PREREG_RECOVER_NEXT_FROZEN.md")
        res["r_arm"], res["r_arm_spec"] = a.r_arm, {k: v for k, v in ARM_SPEC[a.r_arm].items()}
        res["r_arm_note"] = f"the R role of every key below (R_minus_C0, read_a_R_vs_P10, 'R' in controls) is the run M2b_s{{s}}_recft_{a.r_arm}; C0 / C0REP / FZC0 / C0C are reused rollouts"
    out_json = f"{a.raw}/RECOVER_READ_{a.r_arm}.json" if a.mode == "production" else f"{a.raw}/{a.mode.upper()}.json"

    def dump(refused=None):
        res["refused"] = refused
        json.dump(res, open(out_json, "w", encoding="utf-8", newline="\n"), indent=1, default=_jd)

    if a.mode == "production":
        miss = [r["tag"] for r in table if not os.path.exists(f"{tag_dir(r, a.out_dir, a.reuse_dir)}/done_{r['tag']}.txt")]
        if miss:
            dump(f"missing rollouts (no done_<tag>.txt): {miss}")
            print("REFUSED: missing rollouts", miss, file=sys.stderr)
            return 3
        res["controls"], _ = file_controls(a, table)
        failed = [q for q, v in res["controls"].items() if not v["PASS"]]
        res["controls_failed"] = failed
        if [q for q in failed if not q.startswith("KFZ")]:
            dump(f"a control failed: {failed}")
            print("REFUSED: a control failed:", failed, file=sys.stderr)
            return 3
        ac, arrays = array_controls(a, table, f"{a.raw}/arrays", a.extract_workers)
        res["controls"].update(ac)
        failed = [q for q, v in res["controls"].items() if not v["PASS"]]
        res["controls_failed"] = failed
        if [q for q in failed if not q.startswith("KFZ")]:
            dump(f"a control failed: {failed}")
            print("REFUSED: a control failed:", failed, file=sys.stderr)
            return 3
        fz_failed = [q for q in failed if q.startswith("KFZ")]
        freeze_usable, freeze_refusal = not fz_failed, fz_failed
    else:
        res["controls"] = {"file_controls": "NOT APPLICABLE: known-answer mode on existing arrays (KP1 KR1 KR2 KR3 KMARK KC1 KC3 KA1 KFZ KK are production-only)"}
        arrays = kat_arrays(a.mode)
        freeze_usable, freeze_refusal = True, []
    ev, _ = evaluate(arrays, a.n_boot, freeze_usable=freeze_usable, freeze_refusal=freeze_refusal)
    res.update(ev)
    dump(None)
    print("R-ARM", a.r_arm, "| LABEL", res["label"], "| PRIMARY", res["primary"]["PASS"], "mean share", round(res["primary"]["mean_share_closed"], 4), "| GUARDS", res["guards"]["PASS"], "| FLOOR", res["floor"]["PASS"],
          "| FREEZE", res["frame_freeze"].get("reading"), "| READ A R vs P10 DOMINANT", res["read_a_R_vs_P10"]["DOMINANT_both_seeds"], "| COMPOSITE", res["composite"].get("status"), "| STRATEGIC PAUSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
