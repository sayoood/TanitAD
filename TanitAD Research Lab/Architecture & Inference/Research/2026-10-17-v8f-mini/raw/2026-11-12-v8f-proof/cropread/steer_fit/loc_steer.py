#!/usr/bin/env python
"""loc_steer.py -- steering files of card PREREG_V8F_PROOF_ADD100 (STEERREAD), section 2, and controls s1 and s2 (first half). Tier T0.

  python loc_steer.py [--run1 R1] [--s-root S] [--device cuda] [--resume]

For each of the 641 clips (500 train clips of <run1>/lists/train_final.txt, then the clips of eval141.txt) writes <s-root>/steer/<clip>.npy, float32 [n_frames, 2]
(azimuth, elevation in degrees), every frame, from the localiser the cross-fit assigns: a fit clip at list position i by L(i mod 5), the 50 selection clips and the 141
evaluation clips by LA. REFUSES (nothing written) when a localiser's record lists a clip it would steer, lists an evaluation clip, or lists other clips than its name defines.
Writes <s-root>/ctl/S1.json (control s1, top-level `pass`, the sha256 of every steering file) and <s-root>/ctl/S2.json (control s2, first half: for the 6 evaluation clips first by
sha256 hex of the clip id the steering file is reproduced from the arm D tokens and LA, by a second pass with another chunk size, within 1e-3 degrees; top-level `pass`).
The token manifest of arm S, ctl/S3.json and ctl/S8.json are written by the data side's tool. Existing steering files are refused unless --resume (then kept as they are).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import steer_common as SC  # noqa: E402

CFT = SC.CFT


def load_records(s_root):
    recs = {}
    for n in SC.LOC_NAMES:
        p = os.path.join(s_root, "loc", n, "LOC_RECORD.json")
        if not os.path.isfile(p):
            raise SystemExit("REFUSED: no finished localiser %s (%s)" % (n, p))
        r = SC.read_json(p)
        hp = os.path.join(s_root, "loc", n, "head.pt")
        if not os.path.isfile(hp) or SC.sha256_file(hp) != r.get("head_pt_sha256"):
            raise SystemExit("REFUSED: head.pt of %s is missing or its sha256 differs from the record" % n)
        recs[n] = r
    return recs


def save_steer(path, arr):
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        np.save(fh, arr)
    os.replace(tmp, path)


def run_steer(a, log=print):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows, eval_ids = SC.read_lists(a.run1)
    assign = SC.steer_assignment(rows, eval_ids)
    recs = load_records(a.s_root)
    bad = SC.audit_assignment(assign, recs, rows, eval_ids)
    if bad:
        raise SystemExit("REFUSED: cross-fit audit failed: %s" % "; ".join(bad[:5]))
    ctl_dir = os.path.join(a.s_root, "ctl")
    for n in ("S1.json", "S2.json"):
        if os.path.exists(os.path.join(ctl_dir, n)):
            raise SystemExit("REFUSED: %s exists" % os.path.join(ctl_dir, n))
    dev = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")
    models = {n: SC.load_localiser(os.path.join(a.s_root, "loc", n), dev)[0] for n in SC.LOC_NAMES}
    store = CFT.TokenStore(a.run1, "D")
    os.makedirs(os.path.join(a.s_root, "steer"), exist_ok=True)
    shas, counts, t0 = {}, {n: 0 for n in SC.LOC_NAMES}, time.time()
    for i, (clip, role, loc) in enumerate(assign):
        p = os.path.join(a.s_root, "steer", clip + ".npy")
        tok = store.get(clip)
        if os.path.exists(p):
            if not a.resume:
                raise SystemExit("REFUSED: %s exists (use --resume to keep finished files)" % p)
            SC.SteerStore(a.s_root).get(clip, tok.shape[0])               # shape, dtype, finite, frame count
        else:
            st = SC.steer_clip(models[loc], tok, dev)
            if not np.isfinite(st).all():
                raise SystemExit("REFUSED: non-finite steering for %s" % clip)
            save_steer(p, st)
        shas[clip] = SC.sha256_file(p)
        counts[loc] += 1
        store.maps.pop(clip, None)                                           # keep the number of open maps small
        if i % 50 == 0:
            log("steered %d/%d (%.0f s)" % (i + 1, len(assign), time.time() - t0))
    s1 = {"card": SC.CARD, "control": "s1", "pass": True, "detail": {
        "n_steered": len(shas), "n_by_localiser": counts, "violations": [], "eval_clips_read_by_a_localiser": 0,
        "records": {n: {"record_sha256": SC.sha256_file(os.path.join(a.s_root, "loc", n, "LOC_RECORD.json")), "head_pt_sha256": recs[n]["head_pt_sha256"], "n_clips_read": len(recs[n]["clips_read"])} for n in SC.LOC_NAMES},
        "steer_sha256": shas}}
    # control s2, first half: a second pass from the disk files (localiser LA reloaded from its head.pt, chunk size 7, no shared state with the writing pass)
    la, _ = SC.load_localiser(os.path.join(a.s_root, "loc", "LA"), dev)
    chosen = SC.first_eval_by_sha(eval_ids)
    per = {}
    ok2 = True
    for c in chosen:
        disk = np.load(os.path.join(a.s_root, "steer", c + ".npy"))
        again = SC.steer_clip(la, CFT.TokenStore(a.run1, "D").get(c), dev, chunk=7)
        d = float(np.max(np.abs(disk.astype(np.float64) - again.astype(np.float64)))) if disk.shape == again.shape else float("inf")
        per[c] = {"max_abs_diff_deg": d, "n_frames": int(disk.shape[0]), "steer_sha256": shas[c]}
        ok2 = ok2 and d <= SC.S2_TOL_DEG
    s2 = {"card": SC.CARD, "control": "s2_first_half", "pass": bool(ok2), "clips": chosen, "tol_deg": SC.S2_TOL_DEG, "localiser": "LA",
          "localiser_head_pt_sha256": recs["LA"]["head_pt_sha256"], "per_clip": per}
    SC.write_json(os.path.join(ctl_dir, "S1.json"), s1)
    SC.write_json(os.path.join(ctl_dir, "S2.json"), s2)
    log("ZZ LOC-STEER clips %d s1 PASS s2 %s" % (len(shas), "PASS" if ok2 else "FAIL"))
    return 0 if ok2 else 1


def main(argv=None):
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--run1", default=SC.RUN1)
    ap.add_argument("--s-root", default=SC.S_ROOT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true")
    return run_steer(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
