#!/usr/bin/env python
"""dose_d2_state_check.py -- control d2 of note PREREG_V8F_PROOF_ADD42_N2 section 3: the states of manifest positions lo .. hi-1 (default 600 .. 1799) in the new state file equal the banked
state file of the same checkpoint within 1e-4 of the root mean square of the state, and the same comparison with the clip order shifted by one clip MUST fail (planted regression).

  python dose_d2_state_check.py --new <ARM>_dose_train.npz --banked <ARM>_train.npz [--lo 600] [--hi 1800] [--layout train_layout_0_4572.npz] [--rel-bar 1e-4] [--out d2.json]

Runs on the pod (or anywhere with numpy). Memory-light: the small arrays (clips, clip_idx, t) are loaded whole, the state array Z is STREAMED row range by row range from the .npz member
(chunks of 8,192 rows), so about 0.6 GB of new states never sit in memory. Checks, all required:
  * records: both <file>.json exist; checkpoint sha256 equal; the new record has lo 0 and n_clips equal to the clips of the file; state width equal;
  * structure: clips[lo:hi] of the new file equal the banked clip list; the rows of those clips are one contiguous run; clip_idx - lo and t equal the banked ones row by row;
  * values: largest absolute difference (float64 of the float16 values) <= rel-bar x rms of the banked state (rms over every element of the banked file);
  * planted: the same values comparison, positionally, with the new rows taken from the clips lo+1 .. hi (one clip later; lo-1 .. hi-2 when clip hi is absent) must exceed the bar;
    if it does not, the comparison is blind and the control fails;
  * optional --layout: clips, clip_idx and t of the new file equal the layout file's (the layout the label pack was built from).
JSON on stdout (and --out). Exit 0 pass, 2 fail or refused. Tier T0 input control.
"""
import argparse
import json
import os
import sys
import zipfile

import numpy as np

CHUNK = 8192


class RowStream:
    """Row range reader of a 2-D array stored as a member of an .npz written by np.savez (stored, not compressed: seeks are cheap; compressed members still work, slower)."""

    def __init__(self, path, key):
        self.zf = zipfile.ZipFile(path)
        self.f = self.zf.open(key + ".npy")
        ver = np.lib.format.read_magic(self.f)
        if ver == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(self.f)
        else:
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(self.f)
        if fortran or len(shape) != 2:
            raise ValueError("%s[%s]: expected a C-ordered 2-D array, got shape %s fortran %s" % (path, key, shape, fortran))
        self.shape, self.dtype, self.start = shape, np.dtype(dtype), self.f.tell()
        self.row_bytes = int(shape[1]) * self.dtype.itemsize

    def rows(self, r0, r1, chunk=CHUNK):
        """Yield (row index of the first row, float64 array) for rows r0 .. r1-1."""
        self.f.seek(self.start + int(r0) * self.row_bytes)
        r = int(r0)
        while r < r1:
            n = min(chunk, r1 - r)
            buf = self.f.read(n * self.row_bytes)
            if len(buf) != n * self.row_bytes:
                raise ValueError("short read at row %d" % r)
            yield r, np.frombuffer(buf, dtype=self.dtype).reshape(n, self.shape[1]).astype(np.float64)
            r += n

    def close(self):
        self.f.close()
        self.zf.close()


def small(path, key):
    with np.load(path, allow_pickle=False) as z:
        return z[key]


def rms_of(rs):
    s, n = 0.0, 0
    for _, x in rs.rows(0, rs.shape[0]):
        s += float(np.sum(x * x))
        n += x.size
    return float(np.sqrt(s / max(n, 1)))


def max_abs_diff(rs_new, r0_new, rs_ban, r0_ban, n_rows):
    """Largest absolute difference of rows new[r0_new + i] and banked[r0_ban + i], i < n_rows (streamed in step)."""
    m = 0.0
    g_new = rs_new.rows(r0_new, r0_new + n_rows)
    g_ban = rs_ban.rows(r0_ban, r0_ban + n_rows)
    for (_, xn), (_, xb) in zip(g_new, g_ban):
        m = max(m, float(np.max(np.abs(xn - xb))) if xn.size else 0.0)
    return m


def run(new, banked, lo, hi, rel_bar, layout=""):
    R = {"new": new.replace("\\", "/"), "banked": banked.replace("\\", "/"), "lo": int(lo), "hi": int(hi), "rel_bar": rel_bar}
    for p in (new, banked, new + ".json", banked + ".json"):
        if not os.path.isfile(p):
            R.update({"pass": False, "why": "refused: no such file " + p})
            return R
    rn, rb = (json.load(open(p + ".json", encoding="utf-8")) for p in (new, banked))
    cn, cb = small(new, "clips").astype(str), small(banked, "clips").astype(str)
    cin, tn = small(new, "clip_idx").astype(np.int64), small(new, "t").astype(np.int64)
    cib, tb = small(banked, "clip_idx").astype(np.int64), small(banked, "t").astype(np.int64)
    sn, sb = RowStream(new, "Z"), RowStream(banked, "Z")
    try:
        rec = {"ckpt_sha256_equal": bool(rn.get("ckpt_sha256") and rn.get("ckpt_sha256") == rb.get("ckpt_sha256")), "new_record_lo": rn.get("lo"), "new_record_n_clips": rn.get("n_clips"),
               "new_record_split": rn.get("split"), "width_equal": bool(sn.shape[1] == sb.shape[1])}
        rec["ok"] = bool(rec["ckpt_sha256_equal"] and rn.get("lo") == 0 and rn.get("n_clips") == len(cn) and rn.get("split") == "train" and rec["width_equal"])
        R["records"] = rec
        sel = np.flatnonzero((cin >= lo) & (cin < hi))
        st = {"clips_equal": bool(len(cn) >= hi and hi - lo == len(cb) and np.array_equal(cn[lo:hi], cb)), "rows_new": int(len(sel)), "rows_banked": int(len(cib))}
        st["contiguous"] = bool(len(sel) > 0 and sel[-1] - sel[0] + 1 == len(sel))
        st["rows_equal"] = bool(len(sel) == len(cib))
        st["clip_idx_and_t_equal"] = bool(st["contiguous"] and st["rows_equal"] and np.array_equal(cin[sel] - lo, cib) and np.array_equal(tn[sel], tb))
        st["ok"] = bool(st["clips_equal"] and st["contiguous"] and st["rows_equal"] and st["clip_idx_and_t_equal"])
        R["structure"] = st
        ok = rec["ok"] and st["ok"]
        if ok:
            rms = rms_of(sb)
            bar = rel_bar * rms
            d = max_abs_diff(sn, int(sel[0]), sb, 0, len(cib))
            R["values"] = {"rms_state": rms, "bar_abs": bar, "max_abs_diff": d, "max_abs_diff_over_rms": (d / rms if rms > 0 else None), "ok": bool(rms > 0 and d <= bar)}
            ok = ok and R["values"]["ok"]
            # planted: one clip later (or one earlier), rows compared by position
            shift = 1 if int(cin.max()) >= hi else -1
            rows_sh = np.flatnonzero(cin == (lo + shift))
            if len(rows_sh) == 0 or lo + shift < 0:
                R["planted_shift"] = {"fails": False, "why": "no clip to shift to"}
            else:
                r0 = int(rows_sh[0])
                n = int(min(len(cib), sn.shape[0] - r0))
                dp = max_abs_diff(sn, r0, sb, 0, n)
                R["planted_shift"] = {"shift_clips": shift, "rows_compared": n, "max_abs_diff": dp, "fails": bool(dp > bar)}
            ok = ok and R["planted_shift"]["fails"]
            if not R["planted_shift"]["fails"]:
                R["why"] = "the planted one-clip shift did not exceed the bar: the comparison is blind"
        else:
            R["why"] = "records or structure differ: values not compared"
        if layout:
            if not os.path.isfile(layout):
                R["layout"] = {"ok": False, "why": "no such file " + layout}
            else:
                lc, lci, lt = small(layout, "clips").astype(str), small(layout, "clip_idx").astype(np.int64), small(layout, "t").astype(np.int64)
                R["layout"] = {"ok": bool(np.array_equal(lc, cn) and np.array_equal(lci, cin) and np.array_equal(lt, tn))}
            ok = ok and R["layout"]["ok"]
        R["pass"] = bool(ok)
    finally:
        sn.close()
        sb.close()
    return R


def main(argv=None):
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--new", required=True)
    ap.add_argument("--banked", required=True)
    ap.add_argument("--lo", type=int, default=600)
    ap.add_argument("--hi", type=int, default=1800)
    ap.add_argument("--rel-bar", type=float, default=1e-4)
    ap.add_argument("--layout", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    try:
        R = run(a.new, a.banked, a.lo, a.hi, a.rel_bar, a.layout)
    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as e:
        R = {"pass": False, "why": "refused: %s: %s" % (type(e).__name__, e)}
    R["control"] = "d2 (note ADD42_N2 section 3): new states against the banked states on manifest positions %d..%d, tier T0" % (a.lo, a.hi - 1)
    s = json.dumps(R, indent=1)
    print(s)
    if a.out:
        with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(s + "\n")
    return 0 if R["pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
