"""obd_read.py -- the PREREG_V8F_OBD read (card FROZEN 2026-11-10, sha256 2d42cd51...; code of record cdbff971fa6dad4d20b937fe8abda6cb9f622adb).
Tier T1-kin (self-action open loop, m = 10, causal bucket, dead-reckoned nav, logged scene; deterministic planner), 141 episodes / 2,399 windows, per seed.
CPU, zero training, no GPU.  Estimator: taniteval.ci.paired_episode_cluster_bootstrap, B = 2,000, seed 0 ("another draw of EPISODES" only;
"another TRAINING run" is the two floors below).

Usage:  python obd_read.py --out RESULT_OBD.json [--md RESULT_OBD.md] [--dumps DIR] [--ref-dumps DIR] [--parity-json NULL_PARITY.json]
                           [--map arms.json] [--parity auto|holds|fails] [--smoke] [--n-boot 2000]

Inputs (default layout; --map {tag: dump_dir} overrides every dump path):
  OBD-<X>   <DUMPS>/dump_OBD-<X>_m10_causal_heads/ep*.npz        (run_dumps_obd.py; DUMPS = F:/TanitAD-scratch/otband_read/t1, the OTBAND environment)
  DAGOACC_s{k}  <REF>/dump_PARITY_DAGOACC_s{k}_m10_causal_heads/  (the SAME-environment DAG-OACC dumps made by the OTBAND read; the banked DAG-OACC dump is the window reference)
  NULL parity   C:/Users/Admin/obd_state/NULL_PARITY.json         (written by the chain: `append_obd_null_s1` is True iff parity FAILED on the chain's basis)
  banks         C:/Users/Admin/t2v8f_pulls/obd_<tag>/             (selfcheck.json: bank identity, detach, band weights; BANK_MANIFEST.json)

COMMITTED RULES (the card's, verbatim in force; nothing here is a reading choice that changes them)
  Bar cells: per seed 24 = 11 metric rows per band (ADE, FDE, speed MAE, acceleration MAE, along-track MAE, distance keeping: headway abs error and time-gap abs error,
  heading MAE, curvature MAE, yaw-rate MAE, cross-track MAE) x operative knots 1-10 and tactical knots 11-30, plus TAC-LAT and TAC-LON accuracy.  Lower is better for every
  error row, higher for accuracy.  Delta = arm minus reference, paired 95 % interval.
  COST  on a cell at a seed: Delta separated on the worse side AND |Delta| above that cell's floor.   GAIN: separated on the better side AND |Delta| above the floor.
  FLOORS (read at seed 0, applied to both seeds; for an accuracy cell the floor is the pair's |Delta| on that accuracy):
     F = |OBD-NULL_rep - OBD-NULL_s0|  for the detach question,   G = |OBD-BAND_rep - OBD-BAND_s0|  for the band question.
  DETACH question: OBD-CTL against DAG-OACC (same seed; against OBD-NULL of the same seed if NULL parity FAILED), floor F.
     DETACH-KEEPS = no cost on any of the 24 cells;  DETACH-COSTS = at least one cost, every costing cell named with Delta, CI and floor.
  BAND question: OBD-BAND against OBD-CTL (same seed), floor G.
     BAND-ADDS = operative ADE is a gain AND no cell is a cost;  BAND-NEUTRAL = no cost and operative ADE is not a gain;  BAND-COSTS = at least one cost.
  A seed's label is read per seed; an outcome needs both seeds to agree, else SEEDS-DISAGREE (reported per seed; the read is repeated with a third training seed per init,
  --seed 10 / 11, for the arms concerned before any outcome is quoted).
  NULL parity (card, Controls 1): not INVALID when it fails; the fallback applies (OBD-NULL_s1 is trained, every row "against DAG-OACC" reads against OBD-NULL of the same
  seed, NULL minus DAG-OACC is a code-drift row).  An absent NULL_PARITY.json means the detach question cannot be read at all (NOT-YET); the band question does not use it.
READING CHOICES made here (the card is silent), written before any OBD number exists:
  (o1) The point value of a cell and of a floor is the UNROUNDED difference of means over the windows finite in both arms; the interval and `separated` are the estimator's.
       (The OTBAND read compared the estimator's rounded `delta`; the two differ in the 4th decimal and decide nothing except exactly at a floor.)
  (o2) A cell with an applicable Delta but no floor (a floor arm has no finite window there) is never a gain or a cost and is named in `cells_without_floor`; if EVERY cell
       lacks a floor (a floor arm absent) the question is NOT-YET and decides nothing.
  (o3) The DETACH reference at a seed is chosen by the parity state: holds -> DAG-OACC; fails -> OBD-NULL of that seed (s1 needs OBD-NULL_s1, else that seed is NOT-YET).
  (o4) DK-VIOL counts, the code-drift row (NULL minus DAG-OACC), the DAG-ORC pair floor (seed plus trainer version, labelled) and CTL/BAND against DAG-OACC are BESIDE rows: no bar.
  (o5) Controls that fail make the outcome INVALID (fail closed): window identity per load, floor arms present, bank sha256 (done file vs BANK_MANIFEST), DAgger bank identity
       (selfcheck levers.dagger_bank_sha256 prefix 197a68bb / 36c7aea9), identity control (an arm minus itself reads 0 [0, 0]).  The chain's G-SUITE (control 5 of the card's
       controls list is the per-run token / purity / bank; the suite is the chain's) is NOT checked here and is printed as such.  `--smoke` skips the bank checks.
STRATEGIC: PAUSED (PI item 25, PI_DECISION_QUEUE.md, 2026-09-19): reported as paused with this ruling, never as a pass, never omitted.
"""
import argparse
import glob
import hashlib
import importlib.util
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np  # noqa: E402

CARD_SHA256 = "2d42cd51bb1913f32341f224900fcb3a367befe96114144138388e1db4d02dd4"
CODE_OF_RECORD = "cdbff971fa6dad4d20b937fe8abda6cb9f622adb"
NB, SEED = 2000, 0
OP, TAC = "operative_0_2s", "tactical_2_6s"
BANDS = (OP, TAC)
METRICS = ("ade_m", "fde_m", "speed_mae_mps", "along_mae_m", "accel_mae_mps2", "cross_mae_m", "heading_mae_deg", "yaw_rate_mae_degps", "curvature_mae_1pm",
           "dk_hw_abs_err_m", "dk_tg_abs_err_s")
ACCURACY = ("TAC-LAT", "TAC-LON")
CELLS = tuple(f"{b}:{m}" for b in BANDS for m in METRICS) + ACCURACY
assert len(CELLS) == 24
FAMILY = {"ade_m": "PRIMARY", "fde_m": "PRIMARY", "speed_mae_mps": "LONGITUDINAL", "along_mae_m": "LONGITUDINAL", "accel_mae_mps2": "LONGITUDINAL",
          "dk_hw_abs_err_m": "LONGITUDINAL", "dk_tg_abs_err_s": "LONGITUDINAL", "cross_mae_m": "LATERAL", "heading_mae_deg": "LATERAL", "yaw_rate_mae_degps": "LATERAL",
          "curvature_mae_1pm": "LATERAL", "TAC-LAT": "TACTICAL", "TAC-LON": "TACTICAL"}
OP_ADE = f"{OP}:ade_m"
DAG_BANK_PREFIX = {0: "197a68bb", 1: "36c7aea9"}
W_OP, W_TAC = 0.3333333, 0.6666667
SEP = "SEEDS-DISAGREE"
NEXT = {
    "DETACH-KEEPS+BAND-ADDS": "OBD-BAND's argv becomes the O/T training recipe of record (it meets R-T2O); HWM-PLAN (SPEC_HWM.md section 4) and the top-K card build on it.",
    "DETACH-KEEPS+BAND-NEUTRAL": "OBD-CTL's argv becomes the recipe of record; the band loss leaves the recipe, with its logged-only operative gain recorded as not carried to the DAgger base.",
    "DETACH-KEEPS+BAND-COSTS": "OBD-CTL's argv becomes the recipe of record; the band loss leaves the recipe, with its logged-only operative gain recorded as not carried to the DAgger base.",
    "DETACH-COSTS": "the base cannot meet R-T2O at this recipe without the named cost. Next levers, cheapest first, under this card's rules: (1) T's own loss weight --w-tac x2 (the detach "
                    "removes O's gradient into T, so T's own term carries T alone), (2) T's own DAgger term on the DAgger rows (already present; read its weight), (3) report the costing "
                    "cells to the PI: R-T2O is the PI's ruling, and accepting a measured cost of it is the PI's decision.",
    "SEEDS-DISAGREE": "reported per seed; the read is repeated with a third training seed per init (--seed 10 / 11) for the arms concerned before any outcome is quoted.",
    "NULL-PARITY-FAILS": "the code of record changed the training path: OBD-NULL_s1 is trained, every row 'against DAG-OACC' is read against OBD-NULL of the same seed instead, and "
                         "NULL minus DAG-OACC is reported as a code-drift row; the rules are unchanged.",
}


# ------------------------------------------------------------------------------------------------------------------------------------------ estimator access
def load_ci():
    """taniteval/ci.py is pure numpy: loaded from its file (tree: env OBD_TREE, then the OTBAND export, then the STOPBAL export), so the pure tests need no torch."""
    for t in (os.environ.get("OBD_TREE"), "F:/TanitAD-scratch/otband_read/tree", "F:/TanitAD-scratch/stopbal_read/tree"):
        if t and os.path.exists(t + "/taniteval/taniteval/ci.py"):
            spec = importlib.util.spec_from_file_location("obd_ci", t + "/taniteval/taniteval/ci.py")
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            m.SOURCE = t + "/taniteval/taniteval/ci.py"
            return m
    raise RuntimeError("taniteval/ci.py not found in any read tree (set OBD_TREE)")


# ------------------------------------------------------------------------------------------------------------------------------------------ pure reading logic
def _fin(a, b):
    return np.isfinite(np.asarray(a, np.float64)) & np.isfinite(np.asarray(b, np.float64))


def floor_of(rep, base):
    """|mean(rep) - mean(base)| over the windows finite in both (o1); None when nothing is comparable."""
    rep, base = np.asarray(rep, np.float64), np.asarray(base, np.float64)
    m = _fin(rep, base)
    return float(abs(rep[m].mean() - base[m].mean())) if m.any() else None


def floors_from(rep_cells, base_cells):
    """{cell: floor or None} for every bar cell, from a same-recipe pair (replicate arm minus its seed-0 twin)."""
    return {c: floor_of(rep_cells[c], base_cells[c]) for c in CELLS}


def cell_delta(CI, a, b, eid, higher, floor, nb=NB):
    """One cell of arm a minus reference b, paired episode-cluster bootstrap; gain / cost by the card's rule."""
    a, b, eid = np.asarray(a, np.float64), np.asarray(b, np.float64), np.asarray(eid)
    m = _fin(a, b)
    if not m.any():
        return {"status": "NOT-APPLICABLE", "n_windows": 0, "gain": False, "cost": False, "floor": floor}
    aa, bb, ee = a[m], b[m], eid[m]
    pt = float(aa.mean() - bb.mean())
    r = CI.paired_episode_cluster_bootstrap(aa, bb, ee, n_boot=nb, seed=SEED)
    sep = bool(r["separated"])
    better = (pt > 0) if higher else (pt < 0)
    worse = (pt < 0) if higher else (pt > 0)
    if floor is None:
        above = False
    else:
        above = abs(pt) > floor    # FLOORCMP  (the mutation check drops this comparison)
    return {"status": "OK", "delta": pt, "delta_rounded": float(r["delta"]), "lo": float(r["lo"]), "hi": float(r["hi"]), "separated": sep, "floor": floor,
            "above_floor": bool(above), "gain": bool(sep and better and above), "cost": bool(sep and worse and above), "n_windows": int(m.sum()),
            "n_episodes": int(len(np.unique(ee)))}


def compare_cells(CI, arm, ref, eid, floors, nb=NB):
    """arm / ref: {cell: [N]} for all 24 cells.  -> {cell: cell_delta}"""
    return {c: cell_delta(CI, arm[c], ref[c], eid, c in ACCURACY, None if floors is None else floors.get(c), nb) for c in CELLS}


def named(cells, key):
    """cells that are a cost / gain, each with its delta, CI and floor."""
    return [{"cell": c, "family": FAMILY[c.split(":")[-1]], "delta": v["delta"], "lo": v["lo"], "hi": v["hi"], "floor": v["floor"]} for c, v in cells.items() if v.get(key)]


def label_detach(cells):
    return "DETACH-COSTS" if any(v.get("cost") for v in cells.values()) else "DETACH-KEEPS"


def label_band(cells):
    if any(v.get("cost") for v in cells.values()):
        return "BAND-COSTS"
    return "BAND-ADDS" if cells[OP_ADE].get("gain") else "BAND-NEUTRAL"


def combine(per_seed_labels):
    """{seed: label or None} -> outcome string.  Both seeds must agree; one seed read decides nothing."""
    got = {s: l for s, l in per_seed_labels.items() if l}
    if len(got) < 2:
        return f"NOT-YET: per-seed labels so far {got} decide nothing (the card needs both seeds)"
    return next(iter(got.values())) if len(set(got.values())) == 1 else f"{SEP} (s0 {got[0]}; s1 {got[1]})"


def read_question(CI, question, arms, refs_of, eids, floors, nb=NB):
    """question 'detach' | 'band'.  arms: {seed: cells}; refs_of: {seed: (ref_name, cells) or None}; floors: {cell: floor} or None."""
    lab = label_detach if question == "detach" else label_band
    per = {}
    for s in (0, 1):
        if s not in arms or refs_of.get(s) is None:
            per[s] = {"status": "NOT-YET", "label": None}
            continue
        rn, rc = refs_of[s]
        cells = compare_cells(CI, arms[s], rc, eids[s], floors, nb)
        nofl = [c for c, v in cells.items() if v.get("status") == "OK" and v.get("floor") is None]
        per[s] = {"status": "READ", "reference": rn, "cells": cells, "named_costs": named(cells, "cost"), "named_gains": named(cells, "gain"),
                  "n_cells_read": sum(1 for v in cells.values() if v.get("status") == "OK"), "cells_without_floor": nofl,
                  "label": None if (floors is None or all(v.get("status") != "OK" or v.get("floor") is None for v in cells.values())) else lab(cells)}
    return per


def next_step(detach, band, parity_state):
    """The card's committed next step for the combination of outcomes (Rule Zero: printed with every outcome)."""
    out = []
    if parity_state == "fails":
        out.append("NULL parity FAILED: " + NEXT["NULL-PARITY-FAILS"])
    if detach == "DETACH-KEEPS" and band == "BAND-ADDS":
        out.append(NEXT["DETACH-KEEPS+BAND-ADDS"])
    elif detach == "DETACH-KEEPS" and band in ("BAND-NEUTRAL", "BAND-COSTS"):
        out.append(NEXT["DETACH-KEEPS+" + band])
    elif detach == "DETACH-COSTS":
        out.append("DETACH-COSTS: " + NEXT["DETACH-COSTS"] + (f" (band question read beside: {band}; the card commits no combined action)" if band else ""))
    for q, o in (("detach", detach), ("band", band)):
        if o and o.startswith(SEP):
            out.append(f"{q.upper()} {SEP}: " + NEXT[SEP])
    if not out:
        out.append("no committed next step yet: an outcome is missing (NOT-YET) or INVALID")
    return out


def read_all(CI, arms, eids, parity_state, nb=NB):
    """The whole committed read on in-memory cell arrays (what the unit tests plant).
    arms: {tag: {cell: [N]}} with tags OBD-NULL_s0, OBD-NULL_rep, OBD-CTL_s0/s1, OBD-BAND_s0/s1, OBD-BAND_rep, DAGOACC_s0/s1 (+ OBD-NULL_s1).
    eids: {0: eid, 1: eid}.  parity_state: 'holds' | 'fails' | 'unknown'."""
    have = lambda t: t in arms  # noqa: E731
    F = floors_from(arms["OBD-NULL_rep"], arms["OBD-NULL_s0"]) if have("OBD-NULL_rep") and have("OBD-NULL_s0") else None
    G = floors_from(arms["OBD-BAND_rep"], arms["OBD-BAND_s0"]) if have("OBD-BAND_rep") and have("OBD-BAND_s0") else None
    rep = {"floors": {"F": F, "G": G}, "parity_state": parity_state}
    ctl = {s: arms.get(f"OBD-CTL_s{s}") for s in (0, 1)}
    band = {s: arms.get(f"OBD-BAND_s{s}") for s in (0, 1)}
    det_ref = {}
    for s in (0, 1):
        if parity_state == "holds":
            det_ref[s] = ("DAG-OACC", arms[f"DAGOACC_s{s}"]) if have(f"DAGOACC_s{s}") else None
        elif parity_state == "fails":
            det_ref[s] = (f"OBD-NULL_s{s} (NULL parity failed)", arms[f"OBD-NULL_s{s}"]) if have(f"OBD-NULL_s{s}") else None
        else:
            det_ref[s] = None
    rep["detach"] = {"question": "OBD-CTL against DAG-OACC (OBD-NULL of the same seed if NULL parity failed), floor F = |OBD-NULL_rep - OBD-NULL_s0|",
                     "per_seed": read_question(CI, "detach", {s: c for s, c in ctl.items() if c is not None}, det_ref, eids, F, nb) if parity_state != "unknown" else
                     {s: {"status": "NOT-YET", "label": None, "reason": "NULL_PARITY.json absent or unreadable: the reference is undecided"} for s in (0, 1)}}
    rep["band"] = {"question": "OBD-BAND against OBD-CTL (same seed), floor G = |OBD-BAND_rep - OBD-BAND_s0|",
                   "per_seed": read_question(CI, "band", {s: c for s, c in band.items() if c is not None},
                                             {s: (f"OBD-CTL_s{s}", ctl[s]) if ctl[s] is not None else None for s in (0, 1)}, eids, G, nb)}
    for q in ("detach", "band"):
        rep[q]["outcome"] = combine({s: rep[q]["per_seed"][s].get("label") for s in (0, 1)})
    # beside rows (no bar): the code-drift row NULL minus DAG-OACC, CTL / BAND against DAG-OACC
    beside = {"code_drift_NULL_minus_DAGOACC": {}, "CTL_minus_DAGOACC": {}, "BAND_minus_DAGOACC": {}}
    for s in (0, 1):
        if have(f"OBD-NULL_s{s}") and have(f"DAGOACC_s{s}"):
            beside["code_drift_NULL_minus_DAGOACC"][f"s{s}"] = compare_cells(CI, arms[f"OBD-NULL_s{s}"], arms[f"DAGOACC_s{s}"], eids[s], None, nb)
        for nm, grp in (("CTL_minus_DAGOACC", ctl), ("BAND_minus_DAGOACC", band)):
            if grp[s] is not None and have(f"DAGOACC_s{s}"):
                beside[nm][f"s{s}"] = compare_cells(CI, grp[s], arms[f"DAGOACC_s{s}"], eids[s], F, nb)
    rep["beside"] = beside
    rep["next_step"] = next_step(rep["detach"]["outcome"], rep["band"]["outcome"], parity_state)
    return rep


# ------------------------------------------------------------------------------------------------------------------------------------------ controls (pure)
def parity_state_from(js):
    """NULL_PARITY.json dict (or None) -> 'holds' | 'fails' | 'unknown'.  `append_obd_null_s1` is the chain's own verdict (True iff parity FAILED on its basis)."""
    if not isinstance(js, dict) or not isinstance(js.get("append_obd_null_s1"), bool):
        return "unknown"
    return "fails" if js["append_obd_null_s1"] else "holds"


def check_manifest(tag, man, done_sha, commit=CODE_OF_RECORD):
    """bank manifest + the dump's done file -> problem strings."""
    bad = []
    want = ((man or {}).get("files") or {}).get("ckpt_last.pt", {}).get("sha256")
    if not want:
        bad.append(f"{tag}: BANK_MANIFEST.json holds no ckpt_last.pt sha256")
    elif done_sha != want:
        bad.append(f"{tag}: the dump's checkpoint sha256 {str(done_sha)[:12]} differs from the bank manifest {want[:12]}")
    if (man or {}).get("commit") not in (None, commit):
        bad.append(f"{tag}: the bank was trained at commit {(man or {}).get('commit')}, the code of record is {commit}")
    if (man or {}).get("tag") not in (None, tag):
        bad.append(f"{tag}: the bank manifest names tag {(man or {}).get('tag')}")
    return bad


def _num(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def check_selfcheck(tag, sc, init_seed):
    """Bank identity read back from selfcheck.json (control 4 + the detach / band lever read-back).  A key that is present and WRONG is a problem; a key whose layout this
    reader could not confirm (UNVERIFIED) and is absent is returned in notes, not as a problem; a missing selfcheck.json is the caller's problem."""
    bad, notes = [], []
    lv = (sc or {}).get("levers") or {}
    if (sc or {}).get("declared_vs_built") != "PASS":
        bad.append(f"{tag}: G-DVB declared_vs_built is {(sc or {}).get('declared_vs_built')!r}, not 'PASS'")
    bsha = lv.get("dagger_bank_sha256")
    if bsha is None:
        notes.append(f"{tag}: levers.dagger_bank_sha256 absent (UNVERIFIED layout)")
    elif not str(bsha).startswith(DAG_BANK_PREFIX[init_seed]):
        bad.append(f"{tag}: DAgger bank sha256 {str(bsha)[:8]} is not DAG-OACC s{init_seed}'s {DAG_BANK_PREFIX[init_seed]} (control 4)")
    kind = tag.split("-")[1].split("_")[0]          # NULL | CTL | BAND
    ob = (sc or {}).get("otband_probe")
    if kind == "NULL":
        if isinstance(ob, dict) and (ob.get("detach_upper") is True or ob.get("otband") not in (None, "off")):
            bad.append(f"{tag}: a NULL arm's selfcheck carries an otband_probe that says detach/band is on")
    else:
        if not isinstance(ob, dict) or ob.get("detach_upper") is not True:
            bad.append(f"{tag}: the BUILT config's detach (otband_probe.detach_upper) is not True")
        if kind == "BAND":
            for k, want in (("w_op", W_OP), ("w_tac_traj", W_TAC)):
                v = _num(lv.get(k))
                if v is None:
                    notes.append(f"{tag}: levers.{k} absent (UNVERIFIED layout)")
                elif abs(v - want) > 1e-6:
                    bad.append(f"{tag}: levers.{k} reads {v}, the card wants {want}")
    return bad, notes


def identity_control(CI, eid, nb=50):
    """An arm minus itself must read 0 [0, 0] through the same function the cells use."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=len(eid))
    r = cell_delta(CI, x, x, eid, False, 0.0, nb)
    return {"delta": r["delta"], "lo": r["lo"], "hi": r["hi"], "pass": r["delta"] == 0.0 and r["lo"] == 0.0 and r["hi"] == 0.0 and not r["gain"] and not r["cost"]}


# ------------------------------------------------------------------------------------------------------------------------------------------ loading
TAGS = ("OBD-NULL_s0", "OBD-NULL_s1", "OBD-NULL_rep", "OBD-CTL_s0", "OBD-CTL_s1", "OBD-BAND_s0", "OBD-BAND_s1", "OBD-BAND_rep")
BANKED_REF = "E:/Projects/TanitAD-artifacts/v8f-dagoacc-t1/dump_DAGOACC_s{s}_m10_causal_heads"
SAME = ("eid", "clip_index", "ws", "g", "fl", "v0")


def default_map(dumps, ref_dumps):
    m = {t: f"{dumps}/dump_{t}_m10_causal_heads" for t in TAGS}
    for s in (0, 1):
        m[f"DAGOACC_s{s}"] = f"{ref_dumps}/dump_PARITY_DAGOACC_s{s}_m10_causal_heads"
    m["DAGORC_s0"] = f"{ref_dumps}/dump_PARITY_DAGOACCREP_s0_m10_causal_heads"        # beside only (seed 10 and a later trainer file)
    return m


def init_seed_of(tag):
    return 1 if tag.endswith("_s1") else 0              # reps are initialised from M2b s0


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_dump(d):
    fs = sorted(glob.glob(os.path.join(d, "ep*.npz")))
    assert len(fs) == 141, (d, len(fs))
    zs = [np.load(f) for f in fs]
    return {k: np.concatenate([z[k] for z in zs]) for k in ("g", "cl", "ol", "fl", "v0", "eid", "clip_index", "ws")}


def dump_ready(d):
    return os.path.isdir(d) and len(glob.glob(os.path.join(d, "ep*.npz"))) == 141


def done_sha(d):
    """the checkpoint sha256 the dump driver recorded in its done file (<dumps>/done_<full_tag>.txt beside dump_<full_tag>/)"""
    base = os.path.basename(d)
    p = os.path.join(os.path.dirname(d), "done_" + base[len("dump_"):] + ".txt")
    return json.load(open(p, encoding="utf-8")).get("ckpt_sha256") if os.path.exists(p) else None


def to_cells(A, glat, glon):
    cells = {f"{b}:{m}": np.asarray(A[b][m], np.float64) for b in BANDS for m in METRICS}
    for nm, gl, k in (("TAC-LAT", glat, "_lat"), ("TAC-LON", glon, "_lon")):
        cells[nm] = (A[k] == gl).astype(np.float64)
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", default=None)
    ap.add_argument("--map", default=None, help="JSON {tag: dump_dir}; tags OBD-*, DAGOACC_s0/s1 (+ DAGORC_s0 beside); default layout otherwise")
    ap.add_argument("--dumps", default=os.environ.get("OBD_DUMPS", "F:/TanitAD-scratch/otband_read/t1"))
    ap.add_argument("--ref-dumps", default=os.environ.get("OBD_REF_DUMPS", "F:/TanitAD-scratch/otband_read/t1"))
    ap.add_argument("--pull", default=os.environ.get("OBD_PULLS", "C:/Users/Admin/t2v8f_pulls"))
    ap.add_argument("--parity-json", default=os.environ.get("OBD_STATE", "C:/Users/Admin/obd_state") + "/NULL_PARITY.json")
    ap.add_argument("--parity", default="auto", choices=("auto", "holds", "fails"), help="holds / fails only with --smoke (a test hook, never a result)")
    ap.add_argument("--smoke", action="store_true", help="code check on stand-in dumps: skips bank and parity-file reads; never a result")
    ap.add_argument("--n-boot", type=int, default=NB)
    a = ap.parse_args()
    t0 = time.time()
    if a.parity != "auto" and not a.smoke:
        raise SystemExit("--parity holds|fails is a test hook: only with --smoke")
    CI = load_ci()
    amap = json.load(open(a.map, encoding="utf-8")) if a.map else default_map(a.dumps, a.ref_dumps)
    rep = {"tool": "obd_read.py", "card": "theory/obd/PREREG_V8F_OBD.md (FROZEN 2026-11-10)", "card_sha256": CARD_SHA256, "code_of_record": CODE_OF_RECORD,
           "tier": "T1-kin (self-action open loop; m = 10, causal bucket, dead-reckoned nav, logged scene; deterministic planner)", "evidence_class": "MEASURED (ours; dumps, zero training)",
           "estimator": f"taniteval.ci.paired_episode_cluster_bootstrap, B={a.n_boot}, seed {SEED}; ci.py from {CI.SOURCE}. It answers 'another draw of EPISODES'; 'another TRAINING run' is "
                        "the floors F (OBD-NULL_rep vs OBD-NULL_s0) and G (OBD-BAND_rep vs OBD-BAND_s0)",
           "smoke": bool(a.smoke), "reading_choices": __doc__.split("READING CHOICES made here")[1].split("STRATEGIC: PAUSED")[0].strip(),
           "committed_rules": __doc__.split("COMMITTED RULES")[1].split("READING CHOICES made here")[0].strip(),
           "STRATEGIC": {"status": "PAUSED", "ruling": "PI item 25 (PI_DECISION_QUEUE.md ITEM 25, 2026-09-19): reported as PAUSED, never as a pass, never omitted."},
           "families": {"PRIMARY": "ADE, FDE", "LONGITUDINAL": "speed, acceleration, along-track, distance keeping (headway / time-gap abs error; DK-VIOL beside)",
                        "LATERAL": "heading, curvature, yaw rate, cross-track", "TACTICAL": "TAC-LAT / TAC-LON accuracy (bar); T's own decisions via beside_obd.py heads (no bar)",
                        "STRATEGIC": "PAUSED"},
           "controls": {"problems": [], "notes": {"suite_control": "the chain's G-SUITE / token / purity gate is NOT checked here; only its read-back in selfcheck.json"}}}
    prob, notes = rep["controls"]["problems"], rep["controls"]["notes"]
    # ---- parity
    if a.parity != "auto":
        pstate, pjs = a.parity, None
        notes["parity"] = f"FORCED {a.parity} (smoke test hook)"
    else:
        pjs = json.load(open(a.parity_json, encoding="utf-8")) if os.path.exists(a.parity_json) else None
        pstate = parity_state_from(pjs)
        notes["parity"] = {"state": pstate, "file": a.parity_json, "basis": (pjs or {}).get("basis"), "file_sha_equal": (pjs or {}).get("file_sha_equal"),
                           "tensor_equal": (pjs or {}).get("tensor_equal"), "append_obd_null_s1": (pjs or {}).get("append_obd_null_s1")}
        if pstate == "unknown":
            notes["parity"]["reason"] = "NULL_PARITY.json absent or without a boolean append_obd_null_s1: the detach question cannot be read"
    rep["parity_state"] = pstate
    # ---- which arms exist
    ready = {t: (t in amap and dump_ready(amap[t])) for t in list(amap)}
    need_floor = ("OBD-NULL_s0", "OBD-NULL_rep", "OBD-BAND_s0", "OBD-BAND_rep")
    miss = [t for t in need_floor if not ready.get(t)]
    rep["arms_ready"] = ready
    if not any(ready.get(f"OBD-CTL_s{s}") and ready.get(f"OBD-BAND_s{s}") and ready.get(f"DAGOACC_s{s}") for s in (0, 1)):
        rep["outcome"] = "NOT-YET"
        rep["reason"] = "no seed has OBD-CTL, OBD-BAND and the same-environment DAG-OACC dump"
        json.dump(rep, open(a.out, "w"), indent=1, default=float)
        print("WROTE", a.out, rep["outcome"], rep["reason"])
        return
    # ---- geometry + arms
    import obd_geom as G
    assert tuple(G.METRICS) == METRICS and tuple(G.BANDS) == BANDS, "geometry module metrics / bands differ from the card's cells"
    geom = G.build()
    notes["geometry"] = {"extracted_nodes": geom.found, "source": G.SRC_PATH, "source_sha256": sha_file(G.SRC_PATH)}
    arms, eids, dkv, banked = {}, {}, {}, {}
    for s in (0, 1):
        if ready.get(f"DAGOACC_s{s}"):
            banked[s] = load_dump(BANKED_REF.format(s=s))
    cache = {}
    for s in (0, 1):
        if s not in banked or not ready.get(f"DAGOACC_s{s}"):
            continue
        dg = load_dump(amap[f"DAGOACC_s{s}"])
        for k in SAME:
            assert np.array_equal(dg[k], banked[s][k]), f"window or model-free array differs from the banked DAG-OACC dump: {k} in {amap[f'DAGOACC_s{s}']}"
        cache[s] = (dg, geom.join(dg), *geom_labels(geom, dg))
    # a dump is read against the seed-0 or seed-1 reference of its init
    for tag, d in amap.items():
        if not ready.get(tag):
            continue
        s = 1 if tag.endswith("_s1") else 0
        if s not in cache:
            continue
        dg, J, glat, glon = cache[s]
        x = dg if tag == f"DAGOACC_s{s}" else load_dump(d)
        for k in SAME:
            assert np.array_equal(x[k], banked[s][k]), f"window or model-free array differs from the banked DAG-OACC dump: {k} in {d}"
        A = geom.arm_arrays(x["cl"], dg["g"], J, dg["eid"])
        arms[tag] = to_cells(A, glat, glon)
        dkv[tag] = {b: A[b]["_dk_viol"] for b in BANDS}
        eids[s] = dg["eid"]
        print(f"loaded {tag} {time.time() - t0:.0f}s", flush=True)
    for s in (0, 1):
        eids.setdefault(s, eids.get(0))
    if miss:
        notes["floor_arms_missing"] = f"{miss}: the question that needs them reads NOT-YET (no gain or cost can be read without its floor); not INVALID"
    # ---- controls on banks (not in smoke)
    rep["controls"]["identity_control"] = identity_control(CI, next(iter(eids.values())))
    if not rep["controls"]["identity_control"]["pass"]:
        prob.append("identity control failed: an arm minus itself did not read 0 [0, 0]")
    if not a.smoke:
        for tag in arms:
            if not tag.startswith("OBD-"):
                continue
            bd = f"{a.pull}/obd_{tag}"
            mp, sp_ = f"{bd}/BANK_MANIFEST.json", f"{bd}/selfcheck.json"
            if not os.path.exists(mp):
                prob.append(f"{tag}: BANK_MANIFEST.json missing ({mp})")
            else:
                prob += check_manifest(tag, json.load(open(mp, encoding="utf-8")), done_sha(amap[tag]))
            if not os.path.exists(sp_):
                prob.append(f"{tag}: selfcheck.json missing ({sp_}): bank identity (control 4) cannot be read")
            else:
                b_, n_ = check_selfcheck(tag, json.load(open(sp_, encoding="utf-8")), init_seed_of(tag))
                prob += b_
                notes[f"selfcheck_{tag}"] = n_
    # ---- the read
    r = read_all(CI, arms, eids, pstate, a.n_boot)
    rep.update({k: r[k] for k in ("floors", "detach", "band", "beside", "next_step")})
    rep["beside"]["DK-VIOL"] = dkv
    rep["beside"]["T0_train_terms"] = {t: train_tail(a.pull, t) for t in arms if t.startswith("OBD-")}
    if "DAGORC_s0" in arms and "DAGOACC_s0" in arms:
        cells_r = arms["DAGORC_s0"]
        rep["beside"]["DAG_ORC_pair_floor"] = {"label": "PARITY_DAGOACCREP_s0 minus PARITY_DAGOACC_s0: seed (10) PLUS trainer version (v8f_mini_train.py 1c888280 vs 30819a6f); NOT the card's floor",
                                                "floor": floors_from(cells_r, arms["DAGOACC_s0"])}
    # parity-holds control: the NULL dump should equal the DAG-OACC dump when the weights are bit-equal
    if pstate == "holds" and ready.get("OBD-NULL_s0") and ready.get("DAGOACC_s0"):
        d0 = np.abs(load_dump(amap["OBD-NULL_s0"])["cl"] - cache[0][0]["cl"]).max()
        notes["parity_holds_dump_check"] = {"max_abs_diff_rolled_track_m": float(d0), "warning": None if d0 == 0.0 else
                                            "NULL_PARITY says the weights are bit-equal but the NULL and DAG-OACC dumps differ: the parity file or the dump environment is suspect"}
    rep["outcome_detach"], rep["outcome_band"] = rep["detach"]["outcome"], rep["band"]["outcome"]
    if prob:
        rep["outcome"] = "INVALID"
    elif a.smoke:
        rep["outcome"] = f"SMOKE (code check, never a result): DETACH {rep['detach']['outcome']}; BAND {rep['band']['outcome']}"
    else:
        rep["outcome"] = f"DETACH: {rep['detach']['outcome']} | BAND: {rep['band']['outcome']}"
    rep["wall_s"] = round(time.time() - t0, 1)
    json.dump(rep, open(a.out, "w"), indent=1, default=float)
    if a.md:
        open(a.md, "w", encoding="utf-8").write(render_md(rep))
    print("WROTE", a.out, "|", rep["outcome"], "| problems:", len(prob), "|", rep["wall_s"], "s")
    for p in prob:
        print("  INVALID:", p)
    for n in rep["next_step"]:
        print("  NEXT:", n)


def geom_labels(geom, dg):
    return geom.labels(dg["g"])


def train_tail(pull, tag):
    p = f"{pull}/obd_{tag}/train_log.jsonl"
    if not os.path.exists(p):
        return {"status": "absent", "path": p}
    last = None
    for line in open(p, encoding="utf-8"):
        if line.strip():
            last = json.loads(line)
    keep = {k: v for k, v in (last or {}).items() if any(s in k for s in ("ade", "fde", "op", "tac", "step", "loss", "acc"))}
    return {"tier": "T0 (teacher-forced train terms; a curve, not a result)", "last_row": keep, "path": p}


def _f(v, nd=4):
    return "-" if v is None else f"{v:+.{nd}f}"


def render_md(rep):
    L = [f"# OBD read: {rep.get('outcome')}", "", f"Card `PREREG_V8F_OBD.md` (frozen, sha256 `{rep['card_sha256'][:8]}`), code of record `{rep['code_of_record'][:8]}`. "
         f"Tier {rep['tier']}. Evidence class {rep['evidence_class']}. Estimator: {rep['estimator']}.", "",
         f"NULL parity state: **{rep.get('parity_state')}**. STRATEGIC: **PAUSED** ({rep['STRATEGIC']['ruling']})", ""]
    if rep["controls"]["problems"]:
        L += ["## INVALID: controls that failed", *[f"* {p}" for p in rep["controls"]["problems"]], ""]
    for q, title in (("detach", "DETACH question: OBD-CTL against its reference, floor F"), ("band", "BAND question: OBD-BAND against OBD-CTL, floor G")):
        d = rep[q]
        L += [f"## {title}", f"**Outcome: {d['outcome']}**", ""]
        for s in (0, 1):
            ps = d["per_seed"][s] if s in d["per_seed"] else d["per_seed"].get(str(s), {})
            L.append(f"* seed {s}: {ps.get('label')} (reference {ps.get('reference')}, {ps.get('n_cells_read', 0)} cells read)"
                     + (f", cells without floor {ps['cells_without_floor']}" if ps.get("cells_without_floor") else ""))
            for c in ps.get("named_costs", []):
                L.append(f"    * COST {c['cell']}: {_f(c['delta'])} [{_f(c['lo'])}, {_f(c['hi'])}], floor {_f(c['floor'])}")
            for c in ps.get("named_gains", []):
                L.append(f"    * gain {c['cell']}: {_f(c['delta'])} [{_f(c['lo'])}, {_f(c['hi'])}], floor {_f(c['floor'])}")
        L += ["", "| family | cell | s0 delta [CI] | s0 floor | s0 | s1 delta [CI] | s1 | "]
        L += ["|---|---|---|---|---|---|---|"]
        p0, p1 = (d["per_seed"].get(0) or d["per_seed"].get("0") or {}), (d["per_seed"].get(1) or d["per_seed"].get("1") or {})
        for c in CELLS:
            row = [FAMILY[c.split(":")[-1]], c]
            for p in (p0, p1):
                v = (p.get("cells") or {}).get(c)
                if not v or v.get("status") != "OK":
                    row += ["n/a", ""] if p is p0 else ["n/a"]
                    continue
                tag = "COST" if v["cost"] else ("gain" if v["gain"] else ("sep, sub-floor" if v["separated"] else ""))
                row += [f"{_f(v['delta'])} [{_f(v['lo'])}, {_f(v['hi'])}]"] + ([_f(v["floor"]), tag] if p is p0 else [tag])
            L.append("| " + " | ".join(row) + " |")
        L.append("")
    L += ["## What the card commits to do next", *[f"* {n}" for n in rep.get("next_step", [])], ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
