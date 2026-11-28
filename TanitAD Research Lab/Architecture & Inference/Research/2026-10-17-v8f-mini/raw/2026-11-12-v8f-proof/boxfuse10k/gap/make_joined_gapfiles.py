#!/usr/bin/env python
"""make_joined_gapfiles.py -- card PREREG_V8F_PROOF_ADD131 (BOXFUSE10K), section 2 and controls j0 to j4. CPU only; no fit; no label enters a gap file.

  build  --base VGAPLE.npz --new-det-dir DIR --det-call FILE --add129-result FILE --out-dir DIR --rb-args FILE
         writes VGAPJE.npz (joined gap), VGAPJE90.npz (0.90 times it) and BOXFUSE10K_GAPFILES_RECORD.json (pass_j0 .. pass_j4, clauses, counts, hashes).
         Prints clause lines, counts and pass flags; it prints no gap error. The output folder must not exist.
  t0     --base ... --gap-dir DIR --out-dir DIR --rb-args FILE
         the T0 measure of card section 4 (error of the joined file against the base file on the rows with a label gap); refuses unless the record in
         --gap-dir says pass_j0 .. pass_j4. Run only after every control of the card is logged.

Joined gap: on a row where the base file has an output and the box read has a pick (locator's direction, vision only; detector files of the pass over every
frame; size-to-gap map of row 197), when box gap and base gap differ by at most GATE_FACTOR, their geometric mean; else the base gap. No output where the base has
none. --rb-args is the argument file of the reader of card ADD129 (one token per line): it names the join, the fits of arm D, the lists, the label box folder
(map fitting only), the detector files of card ADD127 (map fitting, j2, j3) and the locator. The readers of cards ADD127 and ADD129 are imported unchanged."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import io
import json
import math
import os
import sys

import numpy as np

CARD = "PREREG_V8F_PROOF_ADD131"
CARD_129 = "PREREG_V8F_PROOF_ADD129"
RF_PREFIX = "6b25176e0bff7442"
KEYS = ("clip_index", "frame", "gap", "has_out", "clips", "seed")
GATE_FACTOR = 1.5
MARGIN = 0.90
TOL = 1e-9
IDENT_TOL = 1e-6
J3_EXIST_MIN, J3_GAP_REL, J3_GAP_MIN = 0.99, 0.05, 0.99
J4_MAX_RATIO, J4_MIN_SHARE = 0.75, 0.30
SEED = 131
N_BOOT = 2000
TIER_T0 = "T0 (recorded frames, no rollout; a diagnostic of a perception read and not driving performance)"


def sha_file(p):
    h = hashlib.sha256()
    with io.open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_fuse():
    m = sys.modules.get("read_boxfuse")
    if m is not None and getattr(m, "__file__", None) and os.path.isfile(m.__file__):
        if not sha_file(m.__file__).startswith(RF_PREFIX):
            raise SystemExit("REFUSED: the loaded read_boxfuse.py is not the tool of row 198 (%s)" % RF_PREFIX)
        return m
    for p in (os.environ.get("BOXFUSE_READER"), "/workspace/v8proof/boxread/kit_f/read_boxfuse.py", "F:/TanitAD-scratch/v8proof/boxread/fuse/read_boxfuse.py"):
        if p and os.path.isfile(p):
            if not sha_file(p).startswith(RF_PREFIX):
                raise SystemExit("REFUSED: %s is not the tool of row 198 (%s)" % (p, RF_PREFIX))
            spec = importlib.util.spec_from_file_location("read_boxfuse", p)
            m = importlib.util.module_from_spec(spec)
            sys.modules["read_boxfuse"] = m
            spec.loader.exec_module(m)
            return m
    raise SystemExit("REFUSED: read_boxfuse.py not found (set BOXFUSE_READER)")


RF = load_fuse()
RB, BS = RF.RB, RF.BS


def load(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def rb_args(path, out_dir):
    ap = RB.build_parser()
    for k in ("--add128-controls", "--add128-result"):
        ap.add_argument(k, default="")
    toks = [x.strip() for x in io.open(path, encoding="utf-8").read().replace("\r", "").split("\n") if x.strip()]
    return ap.parse_args(["controls", "--out-dir", out_dir] + toks)


# ============================================================================================ the joined gap
def gate_of(base, has, gbox, pick, factor):
    ok = np.asarray(has, bool) & np.asarray(pick, bool) & np.isfinite(base) & (base > 0) & np.isfinite(gbox) & (gbox > 0)
    out = np.zeros(len(base), bool)
    out[ok] = np.abs(np.log(gbox[ok]) - np.log(base[ok])) <= math.log(factor)
    return out


def join_gaps(base, has, gbox, pick, factor=None, fault=None):
    """-> (joined gap [rows], gate bool [rows]); not a number where the base has no output"""
    f = GATE_FACTOR if factor is None else factor
    base = np.asarray(base, np.float64)
    has = np.asarray(has, bool)
    g = gate_of(base, has, gbox, pick, f)
    out = base.copy()
    out[g] = np.sqrt(base[g] * gbox[g])
    same = g & (gbox == base)
    out[same] = base[same]
    out[~has] = np.nan
    if fault == "shift":
        out[np.flatnonzero(has & ~g)[0]] += 0.01
    if fault == "filled":
        k = np.flatnonzero(~has)
        if len(k):
            out[k[0]] = 1.0
    return out, g


def join_rows(base, has, gbox, pick, factor):
    """a second routine, row by row"""
    out = np.full(len(base), np.nan)
    lf = math.log(factor)
    for i in range(len(base)):
        if not has[i]:
            continue
        b = float(base[i])
        out[i] = b
        if pick[i] and math.isfinite(b) and b > 0 and math.isfinite(float(gbox[i])) and gbox[i] > 0:
            if abs(math.log(float(gbox[i])) - math.log(b)) <= lf and float(gbox[i]) != b:
                out[i] = math.sqrt(b * float(gbox[i]))
    return out


# ============================================================================================ box gaps on the rows of a gap file
class RowMap:
    """rows of the base file -> rows of the evaluation join (by clip id and frame)"""

    def __init__(self, J, B):
        n, off = np.asarray(J["n"], np.int64), np.asarray(J["off"], np.int64)
        jc = {str(c): i for i, c in enumerate(J["clip"])}
        clips = [str(c) for c in B["clips"]]
        ci, fr = np.asarray(B["clip_index"], np.int64), np.asarray(B["frame"], np.int64)
        self.clip_of_row = [clips[k] for k in ci]
        self.frame = fr
        self.join_row = np.full(len(ci), -1, np.int64)
        self.unknown = 0
        for i in range(len(ci)):
            j = jc.get(clips[ci[i]])
            if j is None or fr[i] < 0 or fr[i] >= n[j]:
                self.unknown += 1
                continue
            self.join_row[i] = off[j] + fr[i]
        self.n, self.off, self.jclips, self.clips = n, off, [str(c) for c in J["clip"]], clips


def box_gaps(J, RM, store, loc_az, loc_el, Gw):
    """box gap and pick flag per join row (locator's direction; no label). -> gbox [join rows], pick [join rows], run [join rows], n clips with no detector file"""
    nrow = int(RM.n.sum())
    gbox, pick, run = np.full(nrow, np.nan), np.zeros(nrow, bool), np.zeros(nrow, bool)
    missing = 0
    for i, c in enumerate(RM.jclips):
        d = store.get("wide", c)
        if d is None:
            missing += 1
            continue
        ni = int(RM.n[i])
        sl = slice(int(RM.off[i]), int(RM.off[i] + ni))
        pk = RB.pick_clip(d, np.asarray(loc_az[sl], np.float64), np.asarray(loc_el[sl], np.float64), ni)
        w, h = RB.sizes_of(d, pk)
        m = pk >= 0
        g = np.full(ni, np.nan)
        if m.any():
            g[m] = Gw.predict(w[m], h[m])
        ok = m & np.isfinite(g) & (g > 0)
        gbox[sl], pick[sl], run[sl] = np.where(ok, g, np.nan), ok, RB.run_of(d, ni)
    return gbox, pick, run, missing


def on_rows(RM, x, fill):
    out = np.full(len(RM.join_row), fill, dtype=np.asarray(x).dtype)
    m = RM.join_row >= 0
    out[m] = np.asarray(x)[RM.join_row[m]]
    return out


# ============================================================================================ controls
def ctl_j0(a, new_dir, det_call, RM):
    bad = []
    rec = BS.try_json(os.path.join(new_dir, "wide", "BOXDET_RECORD.json"))
    wr = BS.try_json(a.weights_record) if a.weights_record else None
    if not isinstance(rec, dict) or not isinstance(wr, dict) or str(rec.get("weights_sha256", "")).lower() != str(wr.get("sha256", "x")).lower():
        bad.append("the record of the pass does not carry the weights hash of card ADD127")
    call = BS.read_text(det_call) if det_call else None
    if not call or "--camera wide" not in call:
        bad.append("no record of the call of the wide pass")
    elif "--only-lead-frames" in call or "--box-dir" in call:
        bad.append("the pass was called with a selection of lead frames or a label folder")
    store = RB.DetStore(new_dir)
    n_files = n_sel = n_notok = 0
    for i, c in enumerate(RM.jclips):
        d = store.get("wide", c)
        if d is None:
            continue
        n_files += 1
        fr, fo = np.asarray(d["frame_run"], bool), np.asarray(d["frame_ok"], bool)
        n_sel += int((fo & ~fr).sum())
        n_notok += int((~fo).sum())
        if d["n_frames"] < int(RM.n[i]):
            bad.append("a detector file has fewer frames than the join")
    if n_files != len(RM.jclips):
        bad.append("%d of %d evaluation clips have a detector file" % (n_files, len(RM.jclips)))
    if n_sel:
        bad.append("%d decodable frames on which the detector was not run" % n_sel)
    if store.wins["wide"] and store.wins["wide"] != {tuple(RB.WIDE_WIN)}:
        bad.append("a detector file carries a window other than the card's")
    if store.no_frame_run["wide"]:
        bad.append("detector files without a record of where the detector ran")
    return {"pass": not bad, "failures": bad, "n_clip_files": n_files, "n_frames_not_decodable": n_notok}


def ctl_j1(B, files, gbox_r, pick_r):
    has = np.asarray(B["has_out"], bool)
    base = np.asarray(B["gap"], np.float64)
    JE, J90 = files["VGAPJE"], files["VGAPJE90"]
    jg, g90 = np.asarray(JE["gap"], np.float64), np.asarray(J90["gap"], np.float64)
    gate = gate_of(base, has, gbox_r, pick_r, GATE_FACTOR)
    c = {"same_keys": sorted(JE) == sorted(B) and sorted(J90) == sorted(B)}
    c["only_gap_differs"] = c["same_keys"] and all(np.array_equal(np.asarray(S[k]), np.asarray(B[k])) and np.asarray(S[k]).dtype == np.asarray(B[k]).dtype
                                                   for S in (JE, J90) for k in B if k != "gap")
    c["output_exactly_where_the_base_has_one"] = bool(np.array_equal(np.isfinite(jg), has & np.isfinite(base)) and np.isnan(jg[~has]).all())
    c["geometric_mean_on_gated_rows"] = bool(np.all(np.abs(jg[gate] - np.sqrt(base[gate] * gbox_r[gate])) <= TOL))
    rest = has & ~gate
    c["base_gap_on_other_rows"] = bool(np.array_equal(jg[rest], base[rest]))
    one, _ = join_gaps(base, has, gbox_r, pick_r, 1.0)
    c["gate_factor_one_gives_the_base"] = bool(np.array_equal(one, np.where(has, base, np.nan), equal_nan=True))
    c["second_routine"] = bool(np.allclose(jg, join_rows(base, has, gbox_r, pick_r, GATE_FACTOR), rtol=0, atol=TOL, equal_nan=True))
    c["margin_file_is_0p90_of_the_joined"] = bool(np.all(np.abs(g90[has] - MARGIN * jg[has]) <= TOL) and np.isnan(g90[~has]).all())
    return {"pass": all(c.values()), "failures": [k for k, v in c.items() if not v], "clauses": c}


def ctl_j2(a, r129_path):
    bad, flags = [], {}
    r9 = BS.try_json(r129_path) if r129_path else None
    if not isinstance(r9, dict) or r9.get("card") != CARD_129 or not isinstance(r9.get("measures"), dict):
        return {"pass": False, "failures": ["no result file of card ADD129 given"], "reproduced": flags}
    W = RF.load_world(a)
    E9 = r9["measures"].get("E") or {}
    ones = np.ones(len(W["labg"]), bool)
    for i, k in enumerate(RF.ARM_D):
        mine, _ = join_gaps(W["gD"][k], ones, W["gw"], W["pw"], GATE_FACTOR)
        e = float(np.abs(mine - W["labg"]).mean())
        nm = "FG_%d" % i
        flags[nm] = bool(nm in E9 and abs(e - float(E9[nm])) <= IDENT_TOL)
        if not flags[nm]:
            bad.append("%s: the composer does not reproduce the banked value of row 198 within %g" % (nm, IDENT_TOL))
    return {"pass": not bad, "failures": bad, "reproduced": flags}


def ctl_j3(old, new):
    """old, new: (gbox, pick, run) on the join rows from the pass of card ADD127 and from the pass over every frame"""
    both = old[2] & new[2]
    n = int(both.sum())
    if not n:
        return {"pass": False, "failures": ["no frame read by both passes"], "n_frames": 0}
    exist = float(((old[1] == new[1]) & both).sum() / n)
    pp = both & old[1] & new[1]
    close = float((np.abs(new[0][pp] / old[0][pp] - 1.0) <= J3_GAP_REL).mean()) if pp.any() else None
    bad = []
    if exist < J3_EXIST_MIN:
        bad.append("the pick exists in both passes or in neither on %.4f of the frames (bar %.2f)" % (exist, J3_EXIST_MIN))
    if close is None or close < J3_GAP_MIN:
        bad.append("the box gaps of the two passes differ by at most 5 per cent on %s of the frames with both picks (bar %.2f)" % ("none" if close is None else "%.4f" % close, J3_GAP_MIN))
    return {"pass": not bad, "failures": bad, "n_frames": n, "share_pick_agrees": exist, "n_both_picks": int(pp.sum()), "share_gap_within_5pct": close}


def ctl_j4(base, has, gbox_r, pick_r):
    hp = has & pick_r
    n = int(hp.sum())
    if not n:
        return {"pass": False, "failures": ["no row with an output and a pick"], "n_rows": 0}
    sp = float(gate_of(base, has, gbox_r, pick_r, GATE_FACTOR).sum() / n)
    rng = np.random.RandomState(SEED)
    idx = np.flatnonzero(has)
    pb = base.copy()
    pb[idx] = base[idx][rng.permutation(len(idx))]
    sm = float(gate_of(pb, has, gbox_r, pick_r, GATE_FACTOR).sum() / n)
    bad = []
    if sm > J4_MAX_RATIO * sp:
        bad.append("with the base gaps permuted the gate passes on %.4f of the picked rows, above %.2f of %.4f" % (sm, J4_MAX_RATIO, sp))
    if sp < J4_MIN_SHARE:
        bad.append("the gate passes on %.4f of the rows with an output and a pick (bar %.2f)" % (sp, J4_MIN_SHARE))
    return {"pass": not bad, "failures": bad, "n_rows_with_output_and_pick": n, "share_gated": sp, "share_gated_permuted": sm}


# ============================================================================================ build
def world(a, B, new_dir):
    J = BS.load_npz(a.eval_join)
    RM = RowMap(J, B)
    tclips = BS.ids_of(BS.parse_kv(a.list_file)["train_list"])
    old_store = RB.DetStore(a.det_dir)
    Gw = RB.build_map(RB.train_reads(a.box_dir, old_store, tclips, "wide"))
    if Gw is None:
        raise SystemExit("REFUSED: no wide map could be fitted")
    Pz = BS.lay_pred(J, RB.load_locator(RB.locator_path(a)))
    new = box_gaps(J, RM, RB.DetStore(new_dir), Pz["az"], Pz["el"], Gw)
    old = box_gaps(J, RM, old_store, Pz["az"], Pz["el"], Gw)
    return J, RM, new, old


def build(base_path, new_dir, det_call, r129, out_dir, a, fault=None):
    if os.path.exists(out_dir):
        raise SystemExit("REFUSED: %s exists" % out_dir)
    B = load(base_path)
    miss = [k for k in KEYS if k not in B]
    if miss:
        raise SystemExit("REFUSED: the base file lacks %s" % miss)
    J, RM, new, old = world(a, B, new_dir)
    has = np.asarray(B["has_out"], bool)
    base = np.asarray(B["gap"], np.float64)
    gbox_r, pick_r = on_rows(RM, new[0], np.nan), on_rows(RM, new[1], False)
    jg, gate = join_gaps(base, has, gbox_r, pick_r, GATE_FACTOR, fault)
    os.makedirs(out_dir)
    files, written = {}, {}
    for nm, g in (("VGAPJE", jg), ("VGAPJE90", np.where(has, MARGIN * jg, np.nan))):
        S = {k: np.array(v, copy=True) for k, v in B.items()}
        S["gap"] = g.astype(np.asarray(B["gap"]).dtype)
        p = os.path.join(out_dir, nm + ".npz")
        np.savez(p, **S)
        files[nm] = load(p)
        written[nm] = {"file": nm + ".npz", "sha256": sha_file(p)}
    ctl = {"j0": ctl_j0(a, new_dir, det_call, RM), "j1": ctl_j1(B, files, gbox_r, pick_r), "j2": ctl_j2(a, r129), "j3": ctl_j3(old, new), "j4": ctl_j4(base, has, gbox_r, pick_r)}
    if RM.unknown:
        ctl["j1"]["pass"] = False
        ctl["j1"]["failures"].append("%d rows of the base file are not rows of the join" % RM.unknown)
    rec = {"card": CARD, "base": {"path": base_path, "sha256": sha_file(base_path)}, "files": written, "controls": ctl, "gate_factor": GATE_FACTOR, "margin": MARGIN,
           "counts": {"n_rows": int(len(base)), "n_rows_with_output": int(has.sum()), "n_rows_with_output_and_pick": int((has & pick_r).sum()), "n_rows_gated": int(gate.sum()),
                      "n_rows_with_output_whose_frame_was_not_run": int((has & ~on_rows(RM, new[2], False)).sum()), "n_clips": len(RM.clips)},
           "inputs": {"new_det_listing": RB.dir_listing_sha(os.path.join(new_dir, "wide"), ".npz"), "old_det_listing": RB.dir_listing_sha(os.path.join(a.det_dir, "wide"), ".npz"),
                      "locator": RB.sha_of(RB.locator_path(a)), "join": RB.sha_of(a.eval_join), "add129_result": sha_file(r129) if r129 and os.path.isfile(r129) else None,
                      "det_call": sha_file(det_call) if det_call and os.path.isfile(det_call) else None, "tool": sha_file(os.path.abspath(__file__)), "read_boxfuse.py": sha_file(RF.__file__)},
           "written_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    for k in ("j0", "j1", "j2", "j3", "j4"):
        rec["pass_" + k] = bool(ctl[k]["pass"])
    io.open(os.path.join(out_dir, "BOXFUSE10K_GAPFILES_RECORD.json"), "w", encoding="utf-8", newline="\n").write(json.dumps(rec, indent=1, default=BS._jdefault))
    return rec


def print_record(rec):
    for k in ("j0", "j1", "j2", "j3", "j4"):
        c = rec["controls"][k]
        print("ctl %s PASS" % k if c["pass"] else "ctl %s FAIL %s" % (k, (c["failures"][0] if c["failures"] else "")[:140]))
    print("counts %s" % json.dumps(rec["counts"], sort_keys=True))
    j3, j4 = rec["controls"]["j3"], rec["controls"]["j4"]
    if j3.get("share_pick_agrees") is not None:
        print("j3 shares %.4f %s on %d frames" % (j3["share_pick_agrees"], "None" if j3.get("share_gap_within_5pct") is None else "%.4f" % j3["share_gap_within_5pct"], j3["n_frames"]))
    if j4.get("share_gated") is not None:
        print("j4 shares %.4f %.4f" % (j4["share_gated"], j4["share_gated_permuted"]))
    ok = all(rec["pass_j%d" % i] for i in range(5))
    print("ZZ JOINED-GAPFILES %s" % ("ALL-PASS" if ok else "FAILED"))
    return ok


# ============================================================================================ T0 measure
def t0_main(base_path, gap_dir, out_dir, a):
    rec = BS.try_json(os.path.join(gap_dir, "BOXFUSE10K_GAPFILES_RECORD.json")) or {}
    if rec.get("card") != CARD or not all(rec.get("pass_j%d" % i) is True for i in range(5)):
        print("REFUSED: the record of the gap files is not all-pass: no value is opened")
        return 3
    jp = os.path.join(gap_dir, "VGAPJE.npz")
    if sha_file(jp) != rec["files"]["VGAPJE"]["sha256"] or sha_file(base_path) != rec["base"]["sha256"]:
        print("REFUSED: a gap file is not the one of the record")
        return 3
    B, S = load(base_path), load(jp)
    J = BS.load_npz(a.eval_join)
    L = {k: BS.lay_dump(J, BS.load_npz(p)) for k, p in RB.dump_paths(a).items()}
    RM = RowMap(J, B)
    t3 = BS.t3_rows(L)
    labg = L["D_0"]["lab_gap"]
    has = np.asarray(B["has_out"], bool)
    m = (RM.join_row >= 0) & has
    m[m] &= t3[RM.join_row[m]]
    rows = np.flatnonzero(m)
    lab = labg[RM.join_row[rows]]
    bg, jg = np.asarray(B["gap"], np.float64)[rows], np.asarray(S["gap"], np.float64)[rows]
    jc = {c: i for i, c in enumerate(RM.jclips)}
    ci = np.array([jc[RM.clip_of_row[r]] for r in rows], np.int64)
    nc = len(RM.jclips)
    eb, ej = np.abs(bg - lab), np.abs(jg - lab)
    sb, sj = np.bincount(ci, weights=eb, minlength=nc), np.bincount(ci, weights=ej, minlength=nc)
    R = float(ej.mean() / eb.mean())
    iv = BS.boot_ratio(sj, sb, N_BOOT, SEED)
    name = "BOXFUSE10K-T0-JOINED-" + ("BETTER" if iv[1] < 1.0 else ("WORSE" if iv[0] > 1.0 else "NOT-SEPARATED"))
    far = BS.gap_bin_mask(lab, 40, 60)
    bins = []
    for lo, hi in RB.GAP_BINS:
        k = BS.gap_bin_mask(lab, lo, hi)
        bins.append({"bin": "%d-%d m" % (lo, hi), "n": int(k.sum()), "E_base": BS.fnum(eb[k].mean()) if k.any() else None, "E_joined": BS.fnum(ej[k].mean()) if k.any() else None})
    res = {"card": CARD, "tier": TIER_T0, "name": name, "n_rows": int(len(rows)), "n_rows_with_label_gap_and_no_base_output": int((t3.sum()) - len(rows)), "n_clips": nc,
           "E_base": BS.fnum(eb.mean()), "E_joined": BS.fnum(ej.mean()), "ratio": BS.fnum(R), "interval": iv, "n_boot": N_BOOT, "seed": SEED,
           "share_rows_changed": BS.fnum((jg != bg).mean()), "far": {"n": int(far.sum()), "signed_base": BS.fnum((bg - lab)[far].mean()) if far.any() else None,
                                                                    "signed_joined": BS.fnum((jg - lab)[far].mean()) if far.any() else None},
           "bins": bins, "record_sha256": sha_file(os.path.join(gap_dir, "BOXFUSE10K_GAPFILES_RECORD.json"))}
    os.makedirs(out_dir, exist_ok=True)
    BS.wt(RB.fresh(os.path.join(out_dir, "BOXFUSE10K_T0_RESULT.json")), json.dumps(res, indent=1, default=BS._jdefault))
    print("ZZ BOXFUSE10K T0 NAME %s" % name)
    print("T0 rows %d of %d clips: base %.3f m, joined %.3f m, ratio %.4f, interval [%.4f, %.4f]" % (len(rows), nc, eb.mean(), ej.mean(), R, iv[0], iv[1]))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("mode", choices=("build", "t0"))
    for k in ("--base", "--new-det-dir", "--det-call", "--add129-result", "--out-dir", "--rb-args", "--gap-dir"):
        ap.add_argument(k, default="")
    x = ap.parse_args(argv)
    if not x.base or not x.out_dir or not x.rb_args:
        print("REFUSED: --base, --out-dir and --rb-args are required")
        return 2
    if x.mode == "t0":
        return t0_main(x.base, x.gap_dir, x.out_dir, rb_args(x.rb_args, x.out_dir))
    a = rb_args(x.rb_args, x.out_dir + "_scratch")
    rec = build(x.base, x.new_det_dir, x.det_call, x.add129_result, x.out_dir, a)
    return 0 if print_record(rec) else 1


if __name__ == "__main__":
    sys.exit(main())
