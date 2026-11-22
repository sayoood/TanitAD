#!/usr/bin/env python
"""loc_dirs.py -- stage A (direction files) of card PREREG_V8F_PROOF_ADD108 (LOOKGAP-T0). Tier T0.

  python loc_dirs.py folds   --root R1 --out-root FO [--expect-train 500] [--no-link]
  python loc_dirs.py predict --root R1 --head H --fit-record FR --clips L --out-dir D --record REC [--kit-fit KF] [--locator-dir LD] [--device cuda] [--expect-seed 0]
  python loc_dirs.py check   --root R1 --dirs D --folds-json FJ --fold-root FO --records-dir RD --add107-fit-record FR7 --out OUT [--expect-total 641]
  python loc_dirs.py score   --dirs D --box-dir B --clips L --check-json OUT [--label NAME]       (reads box files; prints only)
  python loc_dirs.py --selftest

folds: the clips of the pilot's train list (lists/train_final.txt under --root, lines '<clip> fit|sel') are sorted by clip id; clip number i (from 0) is in fold i mod 5.
  Fold k: fit clips = fit-role clips not in fold k, selection clips = sel-role clips not in fold k, held-out = the clips of fold k. Per fold a run root FO/f<k>/run with lists/train_final.txt (fit lines, then sel lines,
  each in clip-id order), lists/eval141.txt (the held-out clips; the name is the one locator_fit.py reads) and tok -> <root>/tok (a link, no token file is copied). locator_fit.py runs on that root unchanged.
predict: a direction file per clip in the box-file format the encode tool reads (has_lead, ctr_az, ctr_el); no box folder and no label is opened. Refuses to overwrite.
check: the controls of m0 over a finished direction folder. score: S4, S8, S8_out of card ADD107 on the frames with a label lead (printed only; refuses unless the check file says all clauses pass).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile

import numpy as np

CARD, TIER = "PREREG_V8F_PROOF_ADD108", "T0"
CARD107 = "PREREG_V8F_PROOF_ADD107"
NFOLD = 5
HIST = 5
NEUTRAL_AZ, NEUTRAL_EL = 0.0, -2.0                 # a frame with no full history of five frames
KEYS = ("has_lead", "ctr_az", "ctr_el")           # the keys the encode tool reads from a box file, and the only keys a direction file holds
DEFAULT_LOC_DIR = "/workspace/v8proof/lookat/kit"
DEFAULT_KIT_FIT = "/workspace/v8proof/cropread/kit/fit"


# ============================================================================================ small helpers
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_ids(ids):
    return hashlib.sha256("".join(i + "\n" for i in ids).encode("utf-8")).hexdigest()


def read_ids(path):
    """first token of every non-empty line."""
    with open(path, encoding="utf-8") as fh:
        return [ln.split()[0] for ln in fh if ln.strip()]


def read_train_rows(path):
    rows, seen = [], set()
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            if not ln.strip():
                continue
            p = ln.split()
            if len(p) != 2 or p[1] not in ("fit", "sel"):
                raise SystemExit("REFUSED: train list line is not '<clip> fit|sel': %r" % ln.strip())
            if p[0] in seen:
                raise SystemExit("REFUSED: clip %s appears twice in the train list" % p[0])
            seen.add(p[0])
            rows.append((p[0], p[1]))
    return rows


def write_new(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "x", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    except FileExistsError:
        raise SystemExit("REFUSED: %s exists" % path)


def write_json_new(path, obj):
    write_new(path, json.dumps(obj, indent=1))


def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def token_frames(root, clip):
    """frame count of <root>/tok/D/<clip>.npy from its header (memory map)."""
    return int(np.load(os.path.join(root, "tok", "D", clip + ".npy"), mmap_mode="r").shape[0])


# ============================================================================================ folds
def make_folds(rows, nfold=NFOLD):
    """rows [(clip, role)] -> list of dicts {fold, heldout, fit, sel}; sorted clip ids, clip number i in fold i mod nfold."""
    ids = sorted(c for c, _ in rows)
    role = dict(rows)
    num = {c: i for i, c in enumerate(ids)}
    out = []
    for k in range(nfold):
        out.append({"fold": k,
                    "heldout": [c for c in ids if num[c] % nfold == k],
                    "fit": [c for c in ids if role[c] == "fit" and num[c] % nfold != k],
                    "sel": [c for c in ids if role[c] == "sel" and num[c] % nfold != k]})
    return out


def cmd_folds(a):
    tl = os.path.join(a.root, "lists", "train_final.txt")
    rows = read_train_rows(tl)
    if a.expect_train is not None and len(rows) != a.expect_train:
        raise SystemExit("REFUSED: the train list has %d clips, expected %d" % (len(rows), a.expect_train))
    folds = make_folds(rows)
    rec = {"card": CARD, "tier": TIER, "rule": "clips sorted by clip id; clip number i from 0 is in fold i mod 5; fold fit k: fit-role clips not in fold k as fit clips, sel-role clips not in fold k as selection clips",
           "root": a.root, "train_list_sha256": sha256_file(tl), "n_train": len(rows), "n_fit_role": sum(1 for _, r in rows if r == "fit"), "n_sel_role": sum(1 for _, r in rows if r == "sel"), "folds": []}
    all_held = []
    for f in folds:
        k = f["fold"]
        if set(f["heldout"]) & (set(f["fit"]) | set(f["sel"])):
            raise SystemExit("REFUSED: fold %d: a held-out clip is in its own fit or selection list" % k)
        all_held += f["heldout"]
        rr = os.path.join(a.out_root, "f%d" % k, "run")
        write_new(os.path.join(rr, "lists", "train_final.txt"), "".join("%s fit\n" % c for c in f["fit"]) + "".join("%s sel\n" % c for c in f["sel"]))
        write_new(os.path.join(rr, "lists", "eval141.txt"), "".join(c + "\n" for c in f["heldout"]))
        write_new(os.path.join(a.out_root, "f%d" % k, "heldout.txt"), "".join(c + "\n" for c in f["heldout"]))
        if not a.no_link:
            try:
                os.symlink(os.path.join(a.root, "tok"), os.path.join(rr, "tok"), target_is_directory=True)
            except OSError as e:
                raise SystemExit("REFUSED: cannot link %s: %s" % (os.path.join(rr, "tok"), e))
        rec["folds"].append({"fold": k, "run_root": rr, "n_heldout": len(f["heldout"]), "n_fit": len(f["fit"]), "n_sel": len(f["sel"]),
                             "heldout_sha256": sha256_ids(f["heldout"]), "fit_sha256": sha256_ids(f["fit"]), "sel_sha256": sha256_ids(f["sel"]),
                             "train_list_sha256": sha256_file(os.path.join(rr, "lists", "train_final.txt")),
                             "heldout": f["heldout"], "fit": f["fit"], "sel": f["sel"]})
    if sorted(all_held) != sorted(c for c, _ in rows):
        raise SystemExit("REFUSED: the held-out lists are not a partition of the train list")
    write_json_new(os.path.join(a.out_root, "FOLDS.json"), rec)
    print("ZZ LOOKGAP-FOLDS %d clips, folds %s" % (len(rows), [(f["n_heldout"], f["n_fit"], f["n_sel"]) for f in rec["folds"]]))
    return 0


# ============================================================================================ direction files
def write_direction_files(clips, n_of, predict_fn, out_dir):
    """predict_fn(clip) -> (az[n], el[n]) in degrees, NaN where there is no output. Writes <out_dir>/<clip>.npz with has_lead (True on every frame), ctr_az, ctr_el; frames with no output get (0, -2).
    Refuses before writing anything when a target exists. -> per-clip stats dict."""
    for c in clips:
        if os.path.exists(os.path.join(out_dir, c + ".npz")):
            raise SystemExit("REFUSED: %s exists" % os.path.join(out_dir, c + ".npz"))
    os.makedirs(out_dir, exist_ok=True)
    stats = {}
    for c in clips:
        n = int(n_of(c))
        az, el = predict_fn(c)
        az, el = np.asarray(az, np.float64), np.asarray(el, np.float64)
        if az.shape != (n,) or el.shape != (n,):
            raise SystemExit("REFUSED: output of %s has shape %s / %s, the token file has %d frames" % (c, az.shape, el.shape, n))
        hist = np.isfinite(az) & np.isfinite(el)
        if hist.sum() != max(0, n - (HIST - 1)) or (n >= HIST and not hist[HIST - 1:].all()) or hist[:HIST - 1].any():
            raise SystemExit("REFUSED: clip %s has an output on %d frames, expected exactly frames %d to %d" % (c, int(hist.sum()), HIST - 1, n - 1))
        p = os.path.join(out_dir, c + ".npz")
        part = p + ".part"
        with open(part, "xb") as fh:
            np.savez_compressed(fh, has_lead=np.ones(n, bool), ctr_az=np.where(hist, az, NEUTRAL_AZ), ctr_el=np.where(hist, el, NEUTRAL_EL))
        os.replace(part, p)
        stats[c] = {"n": n, "n_with_history": int(hist.sum()), "n_without_history": int(n - hist.sum()), "sha256": sha256_file(p)}
    return stats


def load_locator(loc_dir, kit_fit):
    p = os.path.join(loc_dir, "locator_fit.py")
    if not os.path.isfile(p):
        raise SystemExit("REFUSED: no locator_fit.py in %s" % loc_dir)
    spec = importlib.util.spec_from_file_location("locator_fit_add107", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.set_cft(kit_fit)
    mod.__sha256_path__ = p
    return mod


def cmd_predict(a):
    import torch
    LF = load_locator(a.locator_dir, a.kit_fit)
    fr = load_json(a.fit_record)
    head_sha = sha256_file(a.head)
    if fr.get("head_pt_sha256") != head_sha:
        raise SystemExit("REFUSED: the head file hash %s is not the one the fit record names (%s)" % (head_sha[:16], str(fr.get("head_pt_sha256"))[:16]))
    if fr.get("card") != CARD107 or fr.get("permute_targets") or int(fr.get("seed", -1)) != a.expect_seed:
        raise SystemExit("REFUSED: the fit record is not a locator fit of card ADD107 with seed %d and no permuted targets" % a.expect_seed)
    ck = torch.load(a.head, map_location="cpu")
    if ck.get("card") != CARD107 or ck.get("out_dim") != 2:
        raise SystemExit("REFUSED: head file is not a locator head of card ADD107")
    clips = read_ids(a.clips)
    if len(set(clips)) != len(clips):
        raise SystemExit("REFUSED: a clip is repeated in %s" % a.clips)
    if os.path.exists(a.record):
        raise SystemExit("REFUSED: %s exists" % a.record)
    dev = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")
    model = LF.make_loc_model()
    model.load_state_dict(ck["state_dict"])
    model.to(dev)
    store = LF.CFT.TokenStore(a.root, "D")
    stats = write_direction_files(clips, lambda c: store.get(c).shape[0], lambda c: LF.predict_clip(model, store.get(c), dev), a.out_dir)
    rec = {"card": CARD, "tier": TIER, "mode": "predict", "head_file": a.head, "head_pt_sha256": head_sha, "fit_record": a.fit_record, "fit_record_head_pt_sha256": fr.get("head_pt_sha256"),
           "fit_record_seed": fr.get("seed"), "fit_record_card": fr.get("card"), "tool_sha256": sha256_file(os.path.abspath(__file__)), "locator_fit_sha256": sha256_file(LF.__sha256_path__),
           "crop_fit_sha256": sha256_file(LF.CFT.__sha256_path__), "clips_list": a.clips, "clips_list_sha256": sha256_ids(clips), "out_dir": a.out_dir, "n_clips": len(clips), "clips": clips,
           "n_frames": int(sum(s["n"] for s in stats.values())), "n_frames_with_history": int(sum(s["n_with_history"] for s in stats.values())),
           "n_frames_without_history": int(sum(s["n_without_history"] for s in stats.values())), "per_clip": stats, "device": str(dev), "torch": torch.__version__,
           "keys_written": list(KEYS), "neutral_no_history": [NEUTRAL_AZ, NEUTRAL_EL], "box_folder_opened": False}
    write_json_new(a.record, rec)
    print("ZZ LOOKGAP-PREDICT %d clips, %d frames (%d with history, %d without)" % (rec["n_clips"], rec["n_frames"], rec["n_frames_with_history"], rec["n_frames_without_history"]))
    return 0


# ============================================================================================ check (m0)
def _clause(failures, **extra):
    d = {"pass": not failures, "n_failures": len(failures), "failures": failures[:20]}
    d.update(extra)
    return d


def run_check(root, dirs, folds_json, fold_root, records_dir, add107_fit_record, expect_total=None):
    fj = load_json(folds_json)
    train = []
    for f in fj["folds"]:
        train += f["heldout"]
    ev = read_ids(os.path.join(root, "lists", "eval141.txt"))
    want = sorted(train + ev)
    cl = {}
    # c0: the folds file belongs to the current train list and the fold lists are those of the rule
    f0 = []
    rows = read_train_rows(os.path.join(root, "lists", "train_final.txt"))
    if sha256_file(os.path.join(root, "lists", "train_final.txt")) != fj.get("train_list_sha256"):
        f0.append("train list hash differs from the folds file")
    ref = make_folds(rows)
    for f, r in zip(fj["folds"], ref):
        for key in ("heldout", "fit", "sel"):
            if f[key] != r[key]:
                f0.append("fold %d list %s differs from the rule" % (f["fold"], key))
        if sha256_ids(f["heldout"]) != f["heldout_sha256"] or sha256_ids(f["fit"]) != f["fit_sha256"] or sha256_ids(f["sel"]) != f["sel_sha256"]:
            f0.append("fold %d list hash differs from its list" % f["fold"])
    if set(train) & set(ev):
        f0.append("a train clip is also an evaluation clip")
    cl["c0_folds_follow_the_rule"] = _clause(f0)
    # c1: one direction file per clip, nothing else; the count
    present = sorted(os.listdir(dirs)) if os.path.isdir(dirs) else []
    f1 = ["missing: %s" % c for c in want if c + ".npz" not in present] + ["extra file: %s" % p for p in present if p[:-4] not in set(want) or not p.endswith(".npz")]
    if expect_total is not None and len(want) != expect_total:
        f1.append("the lists name %d clips, expected %d" % (len(want), expect_total))
    cl["c1_a_file_for_each_clip"] = _clause(f1, n_expected=len(want), n_present=len(present))
    # c2 / c3 / c3b: frame counts, values, keys, neutral frames
    f2, f3, f4, f6 = [], [], [], []
    for c in want:
        p = os.path.join(dirs, c + ".npz")
        if not os.path.isfile(p):
            continue
        try:
            n = token_frames(root, c)
        except Exception as e:                                   # noqa: BLE001 -- a missing or unreadable token file is a failure of this clause
            f2.append("%s: token file unreadable (%s)" % (c, type(e).__name__))
            continue
        z = np.load(p)
        if sorted(z.files) != sorted(KEYS):
            f4.append("%s: keys %s" % (c, sorted(z.files)))
            continue
        h, az, el = z["has_lead"], z["ctr_az"], z["ctr_el"]
        if not (len(h) == len(az) == len(el) == n):
            f2.append("%s: %d / %d / %d frames, the token file %d" % (c, len(h), len(az), len(el), n))
            continue
        if h.dtype != np.bool_ or not h.all() or not (np.isfinite(az).all() and np.isfinite(el).all()):
            f3.append("%s: has_lead not True everywhere or a non-finite centre" % c)
        k = min(n, HIST - 1)
        if not ((az[:k] == NEUTRAL_AZ).all() and (el[:k] == NEUTRAL_EL).all()):
            f6.append("%s: a frame with no history is not (0, -2)" % c)
    cl["c2_frame_count_equals_the_token_file"] = _clause(f2)
    cl["c3_a_direction_on_every_frame"] = _clause(f3)
    cl["c4_no_field_from_a_label"] = _clause(f4, keys_allowed=list(KEYS))
    cl["c5_no_history_frames_are_neutral"] = _clause(f6)
    # c6: the records of the fold fits list no held-out clip as fit or selection clip, and match the folds file
    f7, f8 = [], []
    heads = {}
    for f in fj["folds"]:
        k = f["fold"]
        fp = os.path.join(fold_root, "f%d" % k, "loc", "fits", "LOC_s0", "FIT_RECORD.json")
        if not os.path.isfile(fp):
            f7.append("fold %d: no fit record" % k)
            continue
        r = load_json(fp)
        cr = r.get("clips_read", {})
        if int(r.get("seed", -1)) != 0 or r.get("permute_targets") or r.get("card") != CARD107:
            f7.append("fold %d: the record is not a seed 0 locator fit without permuted targets" % k)
        bad = sorted(set(f["heldout"]) & (set(cr.get("fit", [])) | set(cr.get("sel", []))))
        if bad:
            f7.append("fold %d: held-out clips in the record: %s" % (k, bad[:3]))
        if cr.get("fit") != f["fit"] or cr.get("sel") != f["sel"]:
            f7.append("fold %d: the record's clip lists are not the fold's lists" % k)
        heads[k] = r.get("head_pt_sha256")
        pp = os.path.join(records_dir, "predict_f%d.json" % k)
        if not os.path.isfile(pp):
            f8.append("fold %d: no predict record" % k)
            continue
        pr = load_json(pp)
        if pr.get("head_pt_sha256") != heads[k] or pr.get("fit_record_head_pt_sha256") != heads[k]:
            f8.append("fold %d: the directions come from a head other than the fold's fit" % k)
        if pr.get("clips") != f["heldout"]:
            f8.append("fold %d: the predict record's clips are not the fold's held-out clips" % k)
    cl["c6_fold_fit_records_do_not_list_their_clips"] = _clause(f7)
    # c7: evaluation clips come from the head the ADD107 fit record names
    f9 = []
    r7 = load_json(add107_fit_record)
    pe = os.path.join(records_dir, "predict_eval.json")
    if not os.path.isfile(pe):
        f9.append("no predict record of the evaluation clips")
    else:
        pr = load_json(pe)
        if pr.get("head_pt_sha256") != r7.get("head_pt_sha256") or pr.get("fit_record_head_pt_sha256") != r7.get("head_pt_sha256"):
            f9.append("the evaluation directions come from a head other than the one the fit record names")
        if pr.get("clips") != ev:
            f9.append("the predict record's clips are not the evaluation clips")
        if int(r7.get("seed", -1)) != 0 or r7.get("card") != CARD107:
            f9.append("the named fit record is not seed 0 of card ADD107")
    cl["c7_train_directions_come_from_their_own_fold_head"] = _clause(f8)
    cl["c8_evaluation_clips_come_from_the_named_head"] = _clause(f9, head_pt_sha256_named=r7.get("head_pt_sha256"))
    return {"card": CARD, "tier": TIER, "n_train": len(train), "n_eval": len(ev), "clauses": cl, "all_pass": all(v["pass"] for v in cl.values())}


def cmd_check(a):
    if os.path.exists(a.out):
        raise SystemExit("REFUSED: %s exists" % a.out)
    res = run_check(a.root, a.dirs, a.folds_json, a.fold_root, a.records_dir, a.add107_fit_record, a.expect_total)
    write_json_new(a.out, res)
    for k, v in res["clauses"].items():
        print("%s %s%s" % (k, "PASS" if v["pass"] else "FAIL", "" if v["pass"] else " (%d failures, first: %s)" % (v["n_failures"], v["failures"][0])))
    print("ZZ LOOKGAP-DIRS-CHECK all_pass %s" % res["all_pass"])
    return 0 if res["all_pass"] else 1


# ============================================================================================ score (S4, S8, S8_out)
def score_clips(dirs, box_dir, clips):
    """frames with a label lead (has_lead, finite centre) and a full history; d_az = |pred - label|, d_el likewise. -> dict of counts and shares."""
    n_all = n_nohist = 0
    d_az, d_el, laz = [], [], []
    for c in clips:
        b = np.load(os.path.join(box_dir, c + ".npz"))
        z = np.load(os.path.join(dirs, c + ".npz"))
        has = b["has_lead"].astype(bool) & np.isfinite(b["ctr_az"]) & np.isfinite(b["ctr_el"])
        if len(has) != len(z["ctr_az"]):
            raise SystemExit("REFUSED: %s has %d box frames and %d direction frames" % (c, len(has), len(z["ctr_az"])))
        f = np.arange(len(has))
        n_all += int(has.sum())
        n_nohist += int((has & (f < HIST - 1)).sum())
        m = has & (f >= HIST - 1)
        d_az.append(np.abs(z["ctr_az"][m] - b["ctr_az"][m]))
        d_el.append(np.abs(z["ctr_el"][m] - b["ctr_el"][m]))
        laz.append(b["ctr_az"][m])
    d_az, d_el, laz = (np.concatenate(x) if x else np.zeros(0) for x in (d_az, d_el, laz))
    out = np.abs(laz) > 8.0
    r = {"n_clips": len(clips), "n_label_lead_frames": n_all, "n_without_history_not_scored": n_nohist, "n_rows": int(len(d_az)), "n_out_rows": int(out.sum())}
    s4 = (d_az <= 4) & (d_el <= 4)
    s8 = (d_az <= 8) & (d_el <= 8)
    r["S4"] = float(s4.mean()) if len(s4) else None
    r["S8"] = float(s8.mean()) if len(s8) else None
    r["S8_out"] = float(s8[out].mean()) if out.any() else None
    return r


def cmd_score(a):
    ck = load_json(a.check_json)
    if not ck.get("all_pass"):
        raise SystemExit("REFUSED: the check file %s does not say all clauses pass" % a.check_json)
    r = score_clips(a.dirs, a.box_dir, read_ids(a.clips))
    print("SCORE %s: clips %d, rows %d (label-lead frames %d, %d of them without history, not scored), outside rows (|az| > 8) %d" % (a.label, r["n_clips"], r["n_rows"], r["n_label_lead_frames"], r["n_without_history_not_scored"], r["n_out_rows"]))
    print("SCORE %s: S4 %s S8 %s S8_out %s" % (a.label, *("%.4f" % r[k] if r[k] is not None else "n/a" for k in ("S4", "S8", "S8_out"))))
    return 0


# ============================================================================================ selftest
def selftest():
    import shutil
    rng = np.random.RandomState(3)
    tmpd = tempfile.mkdtemp(prefix="lgdirs_selftest_")
    try:
        root = os.path.join(tmpd, "run1")
        os.makedirs(os.path.join(root, "lists"))
        os.makedirs(os.path.join(root, "tok", "D"))
        box = os.path.join(tmpd, "box")
        os.makedirs(box)
        names = ["c%02d" % i for i in range(20)]
        rng.shuffle(names)
        train, ev = sorted(names[:15]), sorted(names[15:])
        write_new(os.path.join(root, "lists", "train_final.txt"), "".join("%s %s\n" % (c, "fit" if i % 4 else "sel") for i, c in enumerate(train)))
        write_new(os.path.join(root, "lists", "eval141.txt"), "".join(c + "\n" for c in ev))
        nfr = {}
        for c in names:
            n = int(rng.randint(3, 12))
            nfr[c] = n
            np.save(os.path.join(root, "tok", "D", c + ".npy"), np.zeros((n, 1, 1, 1), np.float16))
            has = rng.rand(n) < 0.8
            np.savez(os.path.join(box, c + ".npz"), has_lead=has, ctr_az=rng.uniform(-20, 20, n), ctr_el=rng.uniform(-8, 4, n))
        fo = os.path.join(tmpd, "folds")

        class A:
            pass
        a = A()
        a.root, a.out_root, a.expect_train, a.no_link = root, fo, 15, True
        assert cmd_folds(a) == 0
        fj = load_json(os.path.join(fo, "FOLDS.json"))
        assert [f["n_heldout"] for f in fj["folds"]] == [3, 3, 3, 3, 3]
        # a fake head per fold: the direction is the fold number; fit records as locator_fit.py writes them
        rd, dd = os.path.join(tmpd, "records"), os.path.join(tmpd, "dirs")
        for f in fj["folds"]:
            k = f["fold"]
            fd = os.path.join(fo, "f%d" % k, "loc", "fits", "LOC_s0")
            os.makedirs(fd)
            json.dump({"card": CARD107, "seed": 0, "permute_targets": False, "head_pt_sha256": "h%d" % k, "clips_read": {"fit": f["fit"], "sel": f["sel"]}}, open(os.path.join(fd, "FIT_RECORD.json"), "w"))
            st = write_direction_files(f["heldout"], lambda c: nfr[c], lambda c, k=k: (np.r_[np.full(min(nfr[c], 4), np.nan), np.full(max(0, nfr[c] - 4), float(k))][:nfr[c]],) * 2, dd)
            write_json_new(os.path.join(rd, "predict_f%d.json" % k), {"head_pt_sha256": "h%d" % k, "fit_record_head_pt_sha256": "h%d" % k, "clips": f["heldout"], "n": len(st)})
        write_direction_files(ev, lambda c: nfr[c], lambda c: (np.r_[np.full(min(nfr[c], 4), np.nan), np.full(max(0, nfr[c] - 4), 1.0)][:nfr[c]],) * 2, dd)
        write_json_new(os.path.join(rd, "predict_eval.json"), {"head_pt_sha256": "e", "fit_record_head_pt_sha256": "e", "clips": ev})
        f7 = os.path.join(tmpd, "fr7.json")
        json.dump({"card": CARD107, "seed": 0, "head_pt_sha256": "e"}, open(f7, "w"))
        res = run_check(root, dd, os.path.join(fo, "FOLDS.json"), fo, rd, f7, 20)
        assert res["all_pass"], res
        try:
            write_direction_files(ev[:1], lambda c: nfr[c], lambda c: (np.full(nfr[c], np.nan),) * 2, dd)
            raise AssertionError("overwrite not refused")
        except SystemExit:
            pass
        # a planted fault: a label field in one file must fail c4
        z = dict(np.load(os.path.join(dd, ev[0] + ".npz")))
        np.savez(os.path.join(dd, ev[0] + ".npz"), gap_re=np.zeros(nfr[ev[0]]), **z)
        assert not run_check(root, dd, os.path.join(fo, "FOLDS.json"), fo, rd, f7, 20)["clauses"]["c4_no_field_from_a_label"]["pass"]
        r = score_clips(dd, box, train)
        assert r["n_rows"] > 0
        print("selftest OK: %d clips, folds %s, score rows %d" % (len(names), [f["n_heldout"] for f in fj["folds"]], r["n_rows"]))
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)                  # own temporary folder, made above
    return 0


# ============================================================================================ main
def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv == ["--selftest"]:
        return selftest()
    ap = argparse.ArgumentParser(allow_abbrev=False)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("folds")
    p.add_argument("--root", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--expect-train", type=int, default=None)
    p.add_argument("--no-link", action="store_true")
    p = sub.add_parser("predict")
    p.add_argument("--root", required=True)
    p.add_argument("--head", required=True)
    p.add_argument("--fit-record", required=True)
    p.add_argument("--clips", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--record", required=True)
    p.add_argument("--kit-fit", default=DEFAULT_KIT_FIT)
    p.add_argument("--locator-dir", default=DEFAULT_LOC_DIR)
    p.add_argument("--device", default="cuda")
    p.add_argument("--expect-seed", type=int, default=0)
    p = sub.add_parser("check")
    p.add_argument("--root", required=True)
    p.add_argument("--dirs", required=True)
    p.add_argument("--folds-json", required=True)
    p.add_argument("--fold-root", required=True)
    p.add_argument("--records-dir", required=True)
    p.add_argument("--add107-fit-record", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--expect-total", type=int, default=None)
    p = sub.add_parser("score")
    p.add_argument("--dirs", required=True)
    p.add_argument("--box-dir", required=True)
    p.add_argument("--clips", required=True)
    p.add_argument("--check-json", required=True)
    p.add_argument("--label", default="")
    a = ap.parse_args(argv)
    return {"folds": cmd_folds, "predict": cmd_predict, "check": cmd_check, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
