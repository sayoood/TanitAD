#!/usr/bin/env python
"""tele_dump.py -- dump of one fit of card PREREG_V8F_PROOF_ADD122 (TELEGAP-T0) in the form of crop_dump.py. Tier T0.

  python tele_dump.py --arm M|F|D2 --head H5 --seed K [--permute-labels] --wide-tok-dir DIR --tele-tok-dir DIR --match-dir DIR
         [--out-root DIR] [--present-dump DIR | --present-file FILE] [--kit-fit DIR] [--device cuda]

crop_dump.py is imported unchanged (head_outputs, dump_from_present). A dump is keyed by the rows of the present reader's dump: (cid, frame) with clips[cid] the clip id; the rows are
copied from the present dump, so the keys are those of crop_dump: cid, clip_index, frame, lab_present, lab_gap, lab_closing, lab_run, clips, dt_s, gap, p, clos, valid, seed.
Added key: tele_dropped (bool per row, true where the frame's run holds a frame with mask false; arms M and F only; always false for D2). On such a row gap is NaN, so valid is false.
Reads <out-root>/fits/<arm>_<head>_s<K>[_perm]/head.pt, writes <out-root>/dump/<arm>_<head>[_perm]/reader_s<K>_score.npz and reader_s<K>_meta.json (neither may exist).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import tele_fit as TF  # noqa: E402

DEFAULT_PRESENT = "/workspace/v8proof/closeread/dump/both"


def run_tele_dump(a, log=print):
    import torch
    kit, CFT, CDP = TF.load_kit(a.kit_fit)
    arm, head, seed = a.arm, a.head, int(a.seed)
    out_root = os.path.abspath(a.out_root)
    name = CFT.dump_name(arm, head, a.permute_labels)
    fdir = os.path.join(out_root, "fits", CFT.fit_name(arm, head, seed, a.permute_labels))
    odir = os.path.join(out_root, "dump", name)
    opath = os.path.join(odir, "reader_s%d_score.npz" % seed)
    mpath = os.path.join(odir, "reader_s%d_meta.json" % seed)
    if os.path.exists(opath) or os.path.exists(opath + ".part") or os.path.exists(mpath):
        TF.refuse("%s or its meta file exists" % opath)
    if not os.path.isfile(os.path.join(fdir, "FIT_RECORD.json")) or not os.path.isfile(os.path.join(fdir, "TELE_FIT_RECORD.json")):
        TF.refuse("no finished fit in " + fdir)
    ppath = a.present_file or os.path.join(a.present_dump, "reader_s%d_score.npz" % seed)
    if not os.path.isfile(ppath):
        TF.refuse("no present dump " + ppath)
    z = np.load(ppath, allow_pickle=False)
    Dp = {k: np.asarray(z[k]) for k in z.files}
    dev = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")
    ck = torch.load(os.path.join(fdir, "head.pt"), map_location="cpu")
    n_win = len(TF.ARM_WINDOWS[arm])
    if ck["head"] != head or ck["arm"] != arm:
        TF.refuse("head.pt is for arm %s head %s" % (ck["arm"], ck["head"]))
    if int(ck["n_tok"]) != n_win * CFT.N_TOK_WIN:
        TF.refuse("head.pt has n_tok %s, arm %s needs %d" % (ck["n_tok"], arm, n_win * CFT.N_TOK_WIN))
    store = TF.TeleStore(CFT, arm, a.wide_tok_dir, a.tele_tok_dir, a.match_dir)
    clips = [str(c) for c in Dp["clips"]]
    per, dropped_by_clip = {}, {}
    with TF.patched(CFT, {"N_WIN": n_win, "N_TOK": n_win * CFT.N_TOK_WIN}):
        model = CFT.make_model(head)
        model.load_state_dict(ck["state_dict"])
        model.to(dev)
        for c in clips:
            g = CDP.head_outputs(model, store.get(c), dev)
            df = store.drop_frames(c, head)
            g[df] = np.nan
            per[c], dropped_by_clip[c] = g, df
        D = CDP.dump_from_present(Dp, per, seed)
    cid, fr = np.asarray(Dp["cid"], np.int64), np.asarray(Dp["frame"], np.int64)
    dropped = np.zeros(len(cid), bool)
    for i, c in enumerate(clips):
        m = cid == i
        if m.any():
            dropped[m] = dropped_by_clip[c][fr[m]]
    D["tele_dropped"] = dropped
    os.makedirs(odir, exist_ok=True)
    part = opath + ".part"
    with open(part, "wb") as fh:
        np.savez(fh, **D)
    os.replace(part, opath)
    meta = {"card": TF.CARD, "tool": "tele_dump.py", "arm": arm, "head": head, "seed": seed, "permute_labels": bool(a.permute_labels),
            "head_pt_sha256": TF.sha256_file(os.path.join(fdir, "head.pt")), "present_dump": ppath, "present_dump_sha256": TF.sha256_file(ppath),
            "dump_sha256": TF.sha256_file(opath), "n_rows": int(len(D["frame"])), "n_valid": int(D["valid"].sum()), "n_tele_dropped_rows": int(dropped.sum()),
            "crop_fit_sha256": TF.sha256_file(CFT.__file__), "crop_dump_sha256": TF.sha256_file(CDP.__file__), "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(mpath, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(meta, fh, indent=1)
    log("ZZ TELE-DUMP %s s%d rows %d valid %d tele_dropped %d" % (name, seed, meta["n_rows"], meta["n_valid"], meta["n_tele_dropped_rows"]))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--arm", required=True, choices=TF.TELE_ARMS)
    ap.add_argument("--head", default="H5")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--permute-labels", action="store_true")
    ap.add_argument("--wide-tok-dir", required=True)
    ap.add_argument("--tele-tok-dir", default=None)
    ap.add_argument("--match-dir", default=None)
    ap.add_argument("--out-root", default=TF.DEFAULT_OUT_ROOT)
    ap.add_argument("--present-dump", default=DEFAULT_PRESENT)
    ap.add_argument("--present-file", default=None, help="a present dump file; default <present-dump>/reader_s<K>_score.npz")
    ap.add_argument("--kit-fit", default=None)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    if a.head not in ("H1", "H5"):
        TF.refuse("head " + a.head)
    return run_tele_dump(a)


if __name__ == "__main__":
    sys.exit(main())
