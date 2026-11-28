#!/usr/bin/env python3
"""Consolidate the per-clip bank into MEMMAPPED arrays so the cross-fit can read
each fold without re-decompressing 4.7 GB of npz five times over.

⚠️ WHY THIS EXISTS AND WHY IT CHUNKS. The naive fitter re-read every compressed
clip once per fold; the decompression, not the arithmetic, was the wall. The fix
is a memmap -- but a memmap read whole and cast to float32 would allocate
14237 x 160 x 1024 x 4 B = **9.3 GB** in one line, on a box with ~14 GB free,
which is exactly the dense-windowed-tensor trap that once paged a job to disk for
2.5 h at 41 % GPU while `nvidia-smi` reported an idle accelerator. ⇒ everything
below streams in row CHUNKS and the float32 copy is never larger than one chunk.

⛔ THE `dino` ARM IS DELIBERATELY NOT CONSOLIDATED. It is M84's external
reference, already published on the TRAIN corpus, and it is in NONE of this
spec's pre-registered bars -- the required control is the RAW-PIXEL FLOOR. Adding
its 4.7 GB beside `field`'s would put the working set over the free RAM and
re-create the very thrash this file exists to avoid. Stated as a decision, not
omitted silently.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=512)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    bmeta = json.load(open(os.path.join(a.bank, "_bank_meta.json")))
    files = sorted(glob.glob(os.path.join(a.bank, "*.npz")))
    clips = [os.path.basename(p)[:-4] for p in files]
    n_rows = sum(m["n_rows"] for m in bmeta["clips"])
    pool = int(np.prod(bmeta["pool"]))
    d_state = int(bmeta["d_state"])
    ph, pw = bmeta["pix"]
    print("consolidating %d clips / %d rows  field[%d,%d,%d] pix[%d,%d,%d,3]"
          % (len(files), n_rows, n_rows, pool, d_state, n_rows, ph, pw),
          flush=True)

    F = np.lib.format.open_memmap(os.path.join(a.out, "field.npy"), mode="w+",
                                  dtype=np.float16, shape=(n_rows, pool, d_state))
    P = np.lib.format.open_memmap(os.path.join(a.out, "pix.npy"), mode="w+",
                                  dtype=np.uint8, shape=(n_rows, ph, pw, 3))
    cid, frm, gp, st, sp, owner = [], [], [], [], [], []
    i = 0
    for ci, (p, c) in enumerate(zip(files, clips)):
        z = np.load(p, allow_pickle=True)
        n = len(z["gap"])
        F[i:i + n] = z["field"]
        P[i:i + n] = z["pix"]
        cid.extend([c] * n)
        frm.append(z["frame"])
        gp.append(z["gap"])
        st.append(z["state"].astype(str))
        sp.append(z["speed"])
        owner.extend([(ci, li) for li in range(n)])
        i += n
        if (ci + 1) % 25 == 0:
            print("  %d/%d clips (%d rows)" % (ci + 1, len(files), i), flush=True)
    if i != n_rows:
        raise SystemExit("row count %d != expected %d -- refusing a short bank"
                         % (i, n_rows))
    F.flush()
    P.flush()
    # ⛔ ASSERT ON CONTENT, not on the file's existence: a pre-allocated memmap
    # that a failed write left untouched is a full-size file of ZEROS, and an
    # all-zero floor scores at chance so EVERY arm would appear to beat it.
    fs = np.asarray(F[:: max(1, n_rows // 200)], dtype=np.float32)
    ps = np.asarray(P[:: max(1, n_rows // 200)], dtype=np.float32)
    f_nz = float(np.mean(np.abs(fs) > 0))
    p_nz = float(np.mean(ps > 0))
    print("content check: field |x|>0 on %.4f of sampled cells (mean %.4f); "
          "pix >0 on %.4f (mean %.2f)"
          % (f_nz, float(fs.mean()), p_nz, float(ps.mean())), flush=True)
    if f_nz < 0.5 or p_nz < 0.1:
        raise SystemExit("a consolidated array is (near-)all-zero -- refusing")
    np.savez_compressed(os.path.join(a.out, "meta.npz"),
                        clip_id=np.array(cid), frame=np.concatenate(frm),
                        gap=np.concatenate(gp), state=np.concatenate(st),
                        speed=np.concatenate(sp),
                        owner=np.array(owner, dtype=np.int64),
                        clips=np.array(clips))
    json.dump(bmeta, open(os.path.join(a.out, "_bank_meta.json"), "w"), indent=1)
    print("done -> %s" % a.out, flush=True)


if __name__ == "__main__":
    sys.exit(main())
