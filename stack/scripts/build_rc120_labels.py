"""build_rc120_labels.py -- PREREG_V8F_CKPT120 card 2.1 / section 4: the RC-A120 label table (``--rc120-npz``).

RC-A at arc L is the v9 builder's OWN ``rc_point`` on the 8 m-smoothed egomotion path: valid where ``s_now + L <= s_hi_valid`` and the chord sagitta over ``[s_now - 24, s_now + L + 24]`` m is at most 0.25 m
(``build_v9_labels.py:400-422`` at a4357e79, git blob ``edf5424d...``, pinned here). The table is the SAME function at L = 120 m, per cache step ``j`` of every train clip (row ``off[c] + j``, aligned to the trainer's join) and every
eval-141 clip. Labels may use the ego future (PI 2026-08-03). K-LABEL: the SAME code at L = 30 / 50 / 80 must reproduce the release's ``rc_A*`` columns (train) and the eval join's ``rc`` / ``rc_ok`` EXACTLY
(validity 0 mismatches, |d xy| / |d psi| at float32 precision); the build REFUSES to write a table if it does not.

Output ``--out`` (npz): ``train_clip / train_off / train_n`` (the join's, copied from ``--train-align``), ``train_rc120`` [N, 3] float32 (x m, y m, psi RAD; 0 where invalid), ``train_rc120_ok`` [N] bool, and the same four for ``eval_*``;
``<out>.report.json`` carries the md5 of every input, the builder blob, the K-LABEL numbers and the counts.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time

import numpy as np

ARC_FAR = 120.0
CHECK_ARCS = (30.0, 50.0, 80.0)
SIG, NSIG, SAG_MAX = 8.0, 3.0, 0.25
BLOB_BUILDER = "edf5424d613e90844f0ed984c3a2677c6b6c02e7"
MD5_V9_TRAIN_OF_RECORD = "83e30bd3791f0767707eade5f4671759"
EGO = "C:/Users/Admin/tanitad-data/physicalai/labels/egomotion_alpamayo/{}.parquet"


def git_blob(path):
    b = open(path, "rb").read()
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def fhash(path, algo):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def load_builder(path):
    got = git_blob(path)
    if got != BLOB_BUILDER:
        raise SystemExit(f"{path}: git blob {got} != pinned {BLOB_BUILDER}")
    spec = importlib.util.spec_from_file_location("bv9_rc120", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["bv9_rc120"] = m
    spec.loader.exec_module(m)
    return m


class SparseMax:
    def __init__(self, a):
        a = np.asarray(a, np.float64)
        self.t = [a]
        k = 1
        while (1 << k) <= len(a):
            p = self.t[-1]
            h = 1 << (k - 1)
            self.t.append(np.maximum(p[:-h], p[h:]))
            k += 1

    def query(self, lo, hi):
        n = hi - lo + 1
        k = np.floor(np.log2(np.maximum(n, 1))).astype(int)
        out = np.empty(len(lo))
        for kk in np.unique(k):
            m = k == kk
            t = self.t[kk]
            out[m] = np.maximum(t[lo[m]], t[hi[m] - (1 << kk) + 1])
        return out


class ClipGeom:
    """One clip's egomotion log, smoothed path and the vectorised chord-sagitta maximum (equal to the builder's ``chord_sagitta_max``; checked on every run by ``--sagitta-control``)."""

    def __init__(self, B, path):
        self.B = B
        self.log = B.load_log(path)
        self.tr = B.track10(self.log)
        self.p8 = B.smooth_path(self.tr, SIG)
        self.s_nat = np.asarray(B.s_at(self.tr, self.log.ts), np.float64)
        lg = self.log
        c = np.hypot(np.diff(lg.x), np.diff(lg.y)) * np.abs(np.diff(lg.yaw)) / 8.0
        self.sm = SparseMax(c) if len(c) else None

    def sag(self, s_lo, s_hi):
        s = self.s_nat
        i0 = np.searchsorted(s, s_lo, "left")
        i1 = np.searchsorted(s, s_hi, "right") - 1
        out = np.full(len(s_lo), np.inf)
        ok = (i1 - i0 + 1) >= 2
        if ok.any() and self.sm is not None:
            lo = np.maximum(i0[ok] - 1, 0)
            hiend = np.minimum(i1[ok] + 2, len(s))
            out[ok] = self.sm.query(lo, hiend - 2)
        return out

    def rc(self, now, arc):
        """RC-A at arc ``arc`` m for step times ``now`` [n]: ``ok [n] bool, x [n], y [n], psi_deg [n]`` in the vehicle frame at ``now`` (NaN where invalid)."""
        B, p8 = self.B, self.p8
        now = np.asarray(now, np.float64)
        x0, y0, psi0, _ = (np.asarray(a, np.float64) for a in B.sample(self.log, now))
        s_now = np.asarray(B.s_at(self.tr, now), np.float64)
        st = s_now + float(arc)
        ins = (st <= p8.s_hi_valid) & (st >= p8.s[0])
        px, py, pp = np.interp(st, p8.s, p8.x), np.interp(st, p8.s, p8.y), np.interp(st, p8.s, p8.psi)
        c, s = np.cos(psi0), np.sin(psi0)
        dx, dy = px - x0, py - y0
        xl, yl = c * dx + s * dy, -s * dx + c * dy
        rel = (pp - psi0 + math.pi) % (2 * math.pi) - math.pi
        sg = self.sag(s_now - NSIG * SIG, s_now + float(arc) + NSIG * SIG)
        good = ins & (sg <= SAG_MAX)
        x, y, p = np.full(len(now), np.nan), np.full(len(now), np.nan), np.full(len(now), np.nan)
        x[good], y[good], p[good] = xl[good], yl[good], np.degrees(rel[good])
        return good, x, y, p


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--builder", required=True, help="build_v9_labels.py (git blob edf5424d..., a4357e79)")
    ap.add_argument("--v9-train", required=True)
    ap.add_argument("--expect-md5", default=MD5_V9_TRAIN_OF_RECORD, help="the train release the table is built on (default: the release of record, 83e30bd3); any other md5 is a refusal unless named here")
    ap.add_argument("--train-align", required=True, help="an npz with train_clip / train_off / train_n (the trainer join's; navtok_table.npz)")
    ap.add_argument("--eval-join", required=True, help="join_eval141.npz (clip / off / n / has / now_s / rc / rc_ok)")
    ap.add_argument("--ego-dir-fmt", default=EGO)
    ap.add_argument("--train-every", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    B = load_builder(a.builder)
    md5 = fhash(a.v9_train, "md5")
    if md5 != a.expect_md5:
        raise SystemExit(f"v9 train release md5 {md5} != expected {a.expect_md5}")
    rep = {"tool": "build_rc120_labels.py", "builder_blob": BLOB_BUILDER, "arc_far_m": ARC_FAR, "v9_train_md5": md5, "v9_train_is_release_of_record": md5 == MD5_V9_TRAIN_OF_RECORD,
           "train_align_sha256": fhash(a.train_align, "sha256"), "eval_join_sha256": fhash(a.eval_join, "sha256")}
    out = {}
    # ------------------------------------------------------------------ eval-141
    J = np.load(a.eval_join, allow_pickle=True)
    ec, eoff, en = np.asarray(J["clip"]).astype(str), np.asarray(J["off"]), np.asarray(J["n"])
    N = int(eoff[-1] + en[-1])
    e_rc, e_ok = np.zeros((N, 3), np.float32), np.zeros(N, bool)
    chk = {L: {"valid_mismatch": 0, "xy": 0.0, "psi": 0.0, "n": 0} for L in CHECK_ARCS}
    for ci, cid in enumerate(ec):
        sl = slice(int(eoff[ci]), int(eoff[ci] + en[ci]))
        has = np.asarray(J["has"])[sl]
        if not has.any():
            continue
        g = ClipGeom(B, a.ego_dir_fmt.format(cid))
        now = np.asarray(J["now_s"], np.float64)[sl][has]
        ok, x, y, p = g.rc(now, ARC_FAR)
        idx = np.nonzero(has)[0] + sl.start
        e_ok[idx] = ok
        e_rc[idx[ok], 0], e_rc[idx[ok], 1], e_rc[idx[ok], 2] = x[ok], y[ok], np.radians(p[ok])
        for m, L in enumerate(CHECK_ARCS):                                        # K-LABEL on the eval join: the SAME function at the existing arcs
            o2, x2, y2, p2 = g.rc(now, L)
            jr, jo = np.asarray(J["rc"])[idx, m], np.asarray(J["rc_ok"])[idx, m]
            chk[L]["valid_mismatch"] += int((o2 != jo).sum())
            both = o2 & jo
            if both.any():
                chk[L]["xy"] = max(chk[L]["xy"], float(np.max(np.abs(np.stack([x2[both] - jr[both, 0], y2[both] - jr[both, 1]])))))
                chk[L]["psi"] = max(chk[L]["psi"], float(np.max(np.abs(np.radians(p2[both]) - jr[both, 2]))))
            chk[L]["n"] += int(has.sum())
    rep["K_LABEL_eval"] = {str(int(L)): v for L, v in chk.items()}
    out.update({"eval_clip": ec, "eval_off": eoff, "eval_n": en, "eval_rc120": e_rc, "eval_rc120_ok": e_ok})
    rep["eval"] = {"n_rows": N, "n_has": int(np.asarray(J["has"]).sum()), "n_rc120_ok": int(e_ok.sum())}
    print(f"[eval] done t={time.time() - t0:.0f}s {rep['eval']}", flush=True)
    # ------------------------------------------------------------------ train
    R = np.load(a.v9_train, allow_pickle=True)
    rsha = [str(s) for s in R["clip__sha12"]]
    by = {s: (int(R["clip__row0"][i]), int(R["clip__n_rows"][i]), int(R["clip__k0"][i])) for i, s in enumerate(rsha)}
    now_r = R["row__now_s"]
    val = {int(L): R[f"row__rc_A{int(L)}_valid"] for L in CHECK_ARCS}
    rx = {int(L): (R[f"row__rc_A{int(L)}_x"], R[f"row__rc_A{int(L)}_y"], R[f"row__rc_A{int(L)}_psi"]) for L in CHECK_ARCS}
    A = np.load(a.train_align, allow_pickle=True)
    tc, toff, tn = np.asarray(A["train_clip"]).astype(str), np.asarray(A["train_off"]), np.asarray(A["train_n"])
    NT = int(toff[-1] + tn[-1])
    t_rc, t_ok = np.zeros((NT, 3), np.float32), np.zeros(NT, bool)
    ck = {int(L): {"valid_mismatch": 0, "xy": 0.0, "psi": 0.0, "n": 0} for L in CHECK_ARCS}
    n_missing_release = n_missing_log = n_has = 0
    for ci in range(0, len(tc), a.train_every):
        cid, n = str(tc[ci]), int(tn[ci])
        meta = by.get(hashlib.sha256(cid.encode()).hexdigest()[:12])
        if meta is None:
            n_missing_release += 1
            continue
        r0, nr, k0 = meta
        j = np.arange(n)
        k = 2 * j
        ins = (k >= k0) & (k < k0 + nr)
        if not ins.any():
            continue
        rr = r0 + (k[ins] - k0)
        try:
            g = ClipGeom(B, a.ego_dir_fmt.format(cid))
        except FileNotFoundError:
            n_missing_log += 1
            continue
        now = now_r[rr]
        ok, x, y, p = g.rc(now, ARC_FAR)
        row = int(toff[ci]) + j[ins]
        t_ok[row] = ok
        t_rc[row[ok], 0], t_rc[row[ok], 1], t_rc[row[ok], 2] = x[ok], y[ok], np.radians(p[ok])
        n_has += int(ins.sum())
        for L in CHECK_ARCS:                                                      # K-LABEL on the release of record
            o2, x2, y2, p2 = g.rc(now, L)
            rv = val[int(L)][rr] == 1
            ck[int(L)]["valid_mismatch"] += int((rv != o2).sum())
            ck[int(L)]["n"] += int(len(rr))
            b2 = rv & o2
            if b2.any():
                ck[int(L)]["xy"] = max(ck[int(L)]["xy"], float(np.max(np.abs(np.stack([x2[b2] - rx[int(L)][0][rr][b2], y2[b2] - rx[int(L)][1][rr][b2]])))))
                ck[int(L)]["psi"] = max(ck[int(L)]["psi"], float(np.max(np.abs(p2[b2] - rx[int(L)][2][rr][b2]))))
        if ci % 500 == 0:
            print(f"[train] clip {ci}/{len(tc)} t={time.time() - t0:.0f}s", flush=True)
    rep["K_LABEL_train"] = {str(L): v for L, v in ck.items()}
    rep["train"] = {"n_rows": NT, "n_clips": int(len(tc)), "n_has": n_has, "n_rc120_ok": int(t_ok.sum()), "missing_release_clips": n_missing_release, "missing_log_clips": n_missing_log}
    out.update({"train_clip": tc, "train_off": toff, "train_n": tn, "train_rc120": t_rc, "train_rc120_ok": t_ok})
    bad = [f"eval {L} m: {v['valid_mismatch']} validity mismatches" for L, v in rep["K_LABEL_eval"].items() if v["valid_mismatch"]]
    bad += [f"train {L} m: {v['valid_mismatch']} validity mismatches" for L, v in rep["K_LABEL_train"].items() if v["valid_mismatch"]]
    bad += [f"{sp} {L} m: max |d xy| {v['xy']} / |d psi| {v['psi']}" for sp, d in (("eval", rep["K_LABEL_eval"]), ("train", rep["K_LABEL_train"])) for L, v in d.items() if v["xy"] > 1e-5 or v["psi"] > 1e-5]
    rep["K_LABEL_pass"], rep["K_LABEL_problems"] = not bad, bad
    if bad:
        print(json.dumps(rep, indent=1))
        raise SystemExit("K-LABEL FAILED, no table written: " + "; ".join(bad))
    np.savez(a.out, **out)
    rep["out_sha256"], rep["t_elapsed_s"] = fhash(a.out, "sha256"), round(time.time() - t0, 1)
    json.dump(rep, open(a.out + ".report.json", "w", encoding="utf-8"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
