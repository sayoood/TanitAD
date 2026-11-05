"""Patch 2 of 3 (lead head): stack/tanitad/data/v8f_mini_data.py of the tip 219f6c7 -> adds ``lead_label_arrays`` (the lead head's label, row-aligned to the label join; build_label_join itself is UNTOUCHED).

usage: python -I pb_data.py <path of the tip file> <path of the file to write>"""
import sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
t = src.read_bytes().decode("utf-8")
assert "\r\n" not in t
old = '''def load_join(path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        J = {k: z[k] for k in z.files}
    J["n_clips_missing_labels"] = int(J["n_clips_missing_labels"])
    return J
'''
assert t.count(old) == 1
new = old + '''

#: PREREG_LEVELHEAD_A2: the lead head's presence / gap range (m). ``models.v8f_mini.LEAD_RANGE_M`` is the same number; a test pins their equality.
LEAD_RANGE_M = 40.0


def lead_label_arrays(J: dict, lead_npz) -> dict:
    """PREREG_LEVELHEAD_A2: the lead head's label, ROW-ALIGNED to the label join ``J`` (``off[c] + j`` is cache step ``j`` of clip ``c``, which reads raw frame ``k = 2 j``), read straight from the lead file the trainer
    already loads with ``--lead-train`` (keys ``sha12``, ``k``, ``lead``, ``gap``; a row exists = the frame HAS a lead label). It is deliberately NOT read from ``J["lead_gap"]``: that array is NaN both for "no lead"
    and for "no label", and a join file written earlier carries no more than that.

      ``has``   [N] bool    the frame has a lead label (a row for ``(sha12(clip), 2 j)`` exists)
      ``near``  [N] bool    ... and a lead within :data:`LEAD_RANGE_M` (``lead == 1``, the gap finite and ``gap <= LEAD_RANGE_M``; a negative gap is a lead at 0 m and counts as near)
      ``y``     [N] float32 ``clip(min(gap, LEAD_RANGE_M) / LEAD_RANGE_M, 0, 1)`` where ``near``, else 0.0
      ``off``   [C] int64   the join's clip offsets (the lookup is ``off[ci] + t``)
    plus the coverage record (``n_rows``, ``n_has``, ``n_near``, ``n_clips``, ``n_clips_with_labels``)."""
    z = lead_npz if (isinstance(lead_npz, dict) or hasattr(lead_npz, "files")) else np.load(lead_npz, allow_pickle=True)
    sh, kk = np.asarray(z["sha12"]).astype(str), np.asarray(z["k"], np.int64)
    ld, gp = np.asarray(z["lead"], np.int64), np.asarray(z["gap"], np.float64)
    order = np.argsort(sh, kind="stable")
    sh, kk, ld, gp = sh[order], kk[order], ld[order], gp[order]
    uniq, start = np.unique(sh, return_index=True)
    stop = np.append(start[1:], len(sh))
    where = {str(u): (int(s0), int(s1)) for u, s0, s1 in zip(uniq, start, stop)}
    off, n = np.asarray(J["off"], np.int64), np.asarray(J["n"], np.int64)
    N = int(n.sum())
    has, near, y = np.zeros(N, bool), np.zeros(N, bool), np.zeros(N, np.float32)
    n_with = 0
    for ci, cid in enumerate(J["clip"]):
        rng = where.get(sha12(str(cid)))
        if rng is None:
            continue
        n_with += 1
        k, lead, gap = kk[rng[0]:rng[1]], ld[rng[0]:rng[1]], gp[rng[0]:rng[1]]
        top = int(k.max()) + 1
        lut_has, lut_near = np.zeros(top, bool), np.zeros(top, bool)
        lut_y = np.zeros(top, np.float32)
        lut_has[k] = True
        isnear = (lead == 1) & np.isfinite(gap) & (gap <= LEAD_RANGE_M)
        lut_near[k] = isnear
        lut_y[k] = np.where(isnear, np.clip(np.minimum(np.nan_to_num(gap, nan=0.0), LEAD_RANGE_M) / LEAD_RANGE_M, 0.0, 1.0), 0.0)
        kj = 2 * np.arange(int(n[ci]))
        inb = kj < top
        sl = slice(int(off[ci]), int(off[ci]) + int(n[ci]))
        h, nr, yy = np.zeros(int(n[ci]), bool), np.zeros(int(n[ci]), bool), np.zeros(int(n[ci]), np.float32)
        h[inb], nr[inb], yy[inb] = lut_has[kj[inb]], lut_near[kj[inb]], lut_y[kj[inb]]
        has[sl], near[sl], y[sl] = h, nr, yy
    return {"has": has, "near": near, "y": y, "off": off, "n_rows": N, "n_has": int(has.sum()), "n_near": int(near.sum()), "n_clips": int(len(off)), "n_clips_with_labels": n_with}
'''
t = t.replace(old, new)
assert "\r\n" not in t
dst.write_bytes(t.encode("utf-8"))
print("wrote", dst)
