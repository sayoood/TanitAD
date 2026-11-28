"""make_surrogate.py -- pod tool of card PREREG_V8F_PROOF_ADD83 (SHUFCLOSE10K): the surrogate tables of the closing-speed estimate (section 2 of the card).

Inputs  --est-dir   folder with est_score_<E>_s<k>.npz (arrays clip_index int64, frame int64, est float64; NaN where the history is not full), E in N10, L25
        --dump-dir  folder with reader_s<k>_score.npz (the score dump of closeread: cid, clip_index, frame, p, gap, clos, valid, lab_present, lab_gap, lab_closing, lab_run, clips);
                    a subfolder both/ is tried when the file is not in the folder itself
        --out-dir   new folder (default /workspace/v8proof/closeread/est_shuf); refused when it exists
        --seeds 0,1 (default)   --ests N10,L25 (default)   --fit-dir  folder of closeread_fit.py (default: CLOSEREAD_DIR, this folder, ../closeread)
The map pi of reader seed k (the same for both estimators): the clips (the distinct clip_index values of the dump) are sorted by the share of their frames with p >= 0.5 in the dump of seed k,
ties by clip index; neighbours in that order are exchanged (first with second, third with fourth, ...); with an odd count the last three form a cycle a -> b -> c -> a (a, b, c in the sorted
order). No clip maps to itself.
Writes in the out folder:
  surr_<E>_s<k>.npz   clip_index, frame, est (= estimate of clip pi(c) at frame j; NaN where clip pi(c) has no frame j or no estimate there), map [n] (map[i] = pi(map_clips[i])), map_clips [n]
  ident_<E>_s<k>.npz  clip_index, frame, est (the estimate itself, row for row)
  zero_table.npz, const2_table.npz   the rows of the first estimate table with est 0.0 and 2.0 everywhere (pod check of the wrapper)
  SURROGATE_RECORD.json   maps, sha256 of inputs and outputs, u1, u4, share of NaN, verdict
Controls: u1 per table (the map is a bijection of the clips without a fixed point; it equals the map recomputed from the dump by an independent routine; every surrogate value equals the estimate
table at the mapped clip and the same frame exactly, NaN where there is none); u4 per estimator, pooled over the seeds (on the scored frames of card ADD78 for H 10 [N10] and H 25 [L25],
rows of closeread_fit.build_rows on the dump: Pearson r of the surrogate with lab_closing of the clip it is fed to inside [-0.15, +0.15]; r of the estimate itself printed beside it).
Last line: ZZ SURROGATE PASS   or   ZZ SURROGATE FAIL <names>.
--selftest runs the whole path on synthetic tables, plants three failures (a fixed point, a value from the wrong clip, a surrogate that equals the estimate) and requires each to fail
the control it belongs to. Exit 0 = PASS (selftest: every planted failure caught and the clean case passes)."""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import time
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HS = {"N10": 10, "L25": 25}
R_BAND = 0.15
STRIDE = 1 << 20


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_fit(fit_dir=""):
    for c in (fit_dir, os.environ.get("CLOSEREAD_DIR", ""), HERE, os.path.join(HERE, "..", "closeread")):
        if c and os.path.isfile(os.path.join(c, "closeread_fit.py")):
            p = os.path.normpath(os.path.join(c, "closeread_fit.py"))
            spec = importlib.util.spec_from_file_location("closeread_fit_sc", p)
            m = importlib.util.module_from_spec(spec)
            sys.modules["closeread_fit_sc"] = m
            spec.loader.exec_module(m)
            return m
    raise SystemExit("REFUSED: closeread_fit.py not found (give --fit-dir)")


# ====================================================================================================== the map
def compute_map(clip_ids, share):
    """clip_ids: ascending distinct clip indices [n]; share [n] float. -> pi as an array of clip indices, pi[i] = image of clip_ids[i]. Exchange of neighbours in the order (share, clip index)."""
    clip_ids = np.asarray(clip_ids, np.int64)
    n = len(clip_ids)
    if n < 2:
        raise SystemExit("REFUSED: fewer than two clips")
    order = np.lexsort((clip_ids, np.asarray(share, np.float64)))
    pos = np.empty(n, np.int64)
    m = n if n % 2 == 0 else n - 3
    for a in range(0, m, 2):
        pos[order[a]], pos[order[a + 1]] = order[a + 1], order[a]
    if n % 2 == 1:
        if n < 3:
            raise SystemExit("REFUSED: one clip cannot be mapped without a fixed point")
        a, b, c = order[-3], order[-2], order[-1]
        pos[a], pos[b], pos[c] = b, c, a
    return clip_ids[pos]


def map_reference(clip_ids, counts, totals):
    """independent routine of control u1: exact fractions, Python sorting, dictionaries. -> {clip: image}."""
    ids = [int(c) for c in clip_ids]
    srt = sorted(ids, key=lambda c: (Fraction(int(counts[c]), int(totals[c])), c))
    out = {}
    n = len(srt)
    if n < 2:
        raise SystemExit("REFUSED: too few clips")
    k = n if n % 2 == 0 else n - 3
    for a in range(0, k, 2):
        out[srt[a]], out[srt[a + 1]] = srt[a + 1], srt[a]
    if n % 2 == 1:
        a, b, c = srt[-3:]
        out[a], out[b], out[c] = b, c, a
    return out


def share_of(dump):
    """-> clip_ids ascending, share [n], counts dict, totals dict (frames with p >= 0.5 over frames of the clip in the dump)."""
    ci, p = np.asarray(dump["clip_index"], np.int64), np.asarray(dump["p"], np.float64)
    ids = np.unique(ci)
    counts = {int(c): int(((ci == c) & (p >= 0.5)).sum()) for c in ids}
    totals = {int(c): int((ci == c).sum()) for c in ids}
    share = np.array([counts[int(c)] / totals[int(c)] for c in ids], np.float64)
    return ids, share, counts, totals


# ====================================================================================================== tables
def surrogate_values(est_ci, est_fr, est_v, ids, pi):
    """-> est value of clip pi(c) at frame j for every row (c, j) of the table; NaN where that clip has no such row."""
    est_ci, est_fr, est_v = np.asarray(est_ci, np.int64), np.asarray(est_fr, np.int64), np.asarray(est_v, np.float64)
    key = est_ci * STRIDE + est_fr
    order = np.argsort(key, kind="stable")
    ks, vs = key[order], est_v[order]
    img = dict(zip(np.asarray(ids, np.int64).tolist(), np.asarray(pi, np.int64).tolist()))
    tgt = np.array([img[int(c)] for c in est_ci], np.int64) * STRIDE + est_fr
    pos = np.minimum(np.searchsorted(ks, tgt), len(ks) - 1)
    hit = ks[pos] == tgt
    return np.where(hit, vs[pos], np.nan)


def load_est(path):
    with np.load(path, allow_pickle=False) as z:
        for k in ("clip_index", "frame", "est"):
            if k not in z.files:
                raise SystemExit("REFUSED: %s lacks %s" % (path, k))
        return np.asarray(z["clip_index"], np.int64), np.asarray(z["frame"], np.int64), np.asarray(z["est"], np.float64)


def load_table(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


# ====================================================================================================== controls
def check_u1(surr, est, dump_ids, counts, totals):
    """surr: dict of the surrogate file (clip_index, frame, est, map, map_clips); est: (ci, fr, v) of the estimate table. -> (pass, detail dict)."""
    d = {}
    mc, mp = np.asarray(surr["map_clips"], np.int64), np.asarray(surr["map"], np.int64)
    d["n_clips"] = int(len(mc))
    d["clips_are_the_dump_clips"] = bool(np.array_equal(np.sort(mc), np.sort(np.asarray(dump_ids, np.int64))))
    d["bijection"] = bool(len(set(mp.tolist())) == len(mp) and set(mp.tolist()) == set(mc.tolist()))
    d["fixed_points"] = int((mc == mp).sum())
    ref = map_reference(dump_ids, counts, totals)
    d["equals_recomputed"] = bool(all(ref.get(int(c)) == int(m) for c, m in zip(mc, mp)) and len(ref) == len(mc))
    ci, fr, v = np.asarray(surr["clip_index"], np.int64), np.asarray(surr["frame"], np.int64), np.asarray(surr["est"], np.float64)
    ec, ef, ev = est
    d["rows_equal_estimate_rows"] = bool(np.array_equal(ci, ec) and np.array_equal(fr, ef))
    look = {(int(a), int(b)): float(c) for a, b, c in zip(ec, ef, ev)}
    img = dict(zip(mc.tolist(), mp.tolist()))
    bad = 0
    for a, b, c in zip(ci.tolist(), fr.tolist(), v.tolist()):
        want = look.get((img.get(a, -1), b), float("nan"))
        if not ((want != want and c != c) or want == c):
            bad += 1
    d["values_not_equal_to_the_mapped_estimate"] = bad
    d["nan_share"] = float(np.isnan(v).mean()) if len(v) else float("nan")
    d["nan_share_of_estimate"] = float(np.isnan(ev).mean()) if len(ev) else float("nan")
    ok = d["clips_are_the_dump_clips"] and d["bijection"] and d["fixed_points"] == 0 and d["equals_recomputed"] and d["rows_equal_estimate_rows"] and bad == 0
    return bool(ok), d


def pearson(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def u4_rows(CF, dump, H, surr, ident):
    """scored frames of card ADD78 (CF.build_rows) -> (surrogate value, estimate value, label closing speed) per row, lookups by (clip_index, frame)."""
    rows = CF.build_rows(dump, H)
    if len(rows["y"]) == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    cl_of = {str(c): i for i, c in enumerate(dump["clips"])}
    cid = np.array([cl_of[str(c)] for c in rows["clip"]], np.int64)
    idx_of_cid = {}
    for a, b in zip(np.asarray(dump["cid"], np.int64).tolist(), np.asarray(dump["clip_index"], np.int64).tolist()):
        idx_of_cid.setdefault(a, b)
    ci = np.array([idx_of_cid[int(c)] for c in cid], np.int64)
    fr = np.asarray(rows["frame"], np.int64)

    def at(tab):
        key = np.asarray(tab["clip_index"], np.int64) * STRIDE + np.asarray(tab["frame"], np.int64)
        o = np.argsort(key, kind="stable")
        ks, vs = key[o], np.asarray(tab["est"], np.float64)[o]
        t = ci * STRIDE + fr
        pos = np.minimum(np.searchsorted(ks, t), len(ks) - 1)
        return np.where(ks[pos] == t, vs[pos], np.nan)
    return at(surr), at(ident), np.asarray(rows["y"], np.float64)


def check_u4(per_seed_rows):
    """per_seed_rows: list of (surrogate, estimate, label) per seed -> (pass, detail). Pooled over the seeds; rows with a finite surrogate value."""
    s = np.concatenate([r[0] for r in per_seed_rows]) if per_seed_rows else np.zeros(0)
    e = np.concatenate([r[1] for r in per_seed_rows]) if per_seed_rows else np.zeros(0)
    y = np.concatenate([r[2] for r in per_seed_rows]) if per_seed_rows else np.zeros(0)
    fin = np.isfinite(s) & np.isfinite(y)
    fe = np.isfinite(e) & np.isfinite(y)
    r_s = pearson(s[fin], y[fin])
    r_e = pearson(e[fe], y[fe])
    d = {"rows_scored": int(len(y)), "rows_with_surrogate": int(fin.sum()), "rows_with_estimate": int(fe.sum()), "r_surrogate": r_s, "r_estimate": r_e, "band": [-R_BAND, R_BAND],
         "rows_per_seed": [int(len(r[2])) for r in per_seed_rows]}
    ok = bool(np.isfinite(r_s) and -R_BAND <= r_s <= R_BAND)
    return ok, d


# ====================================================================================================== the tool
def run(est_dir, dump_dir, out_dir, seeds, ests, fit_dir="", tamper=None):
    """tamper: selftest hook, called as tamper(kind, arrays) before a table is written. -> (record dict, failed names)."""
    if os.path.exists(out_dir):
        raise SystemExit("REFUSED: %s exists" % out_dir)
    CF = load_fit(fit_dir)
    os.makedirs(out_dir)
    rec = {"card": "PREREG_V8F_PROOF_ADD83", "seeds": list(seeds), "estimators": list(ests), "maps": {}, "files": {}, "inputs": {}, "u1": {}, "u4": {}, "r_band": R_BAND}
    failed = []
    dumps, first = {}, None
    for k in seeds:
        p = os.path.join(dump_dir, "reader_s%d_score.npz" % k)
        if not os.path.isfile(p):
            p = os.path.join(dump_dir, "both", "reader_s%d_score.npz" % k)
        if not os.path.isfile(p):
            raise SystemExit("REFUSED: no reader_s%d_score.npz in %s" % (k, dump_dir))
        D = CF.load_dump(p)
        ids, share, counts, totals = share_of(D)
        pi = compute_map(ids, share)
        dumps[k] = (D, ids, counts, totals)
        rec["inputs"]["dump_s%d" % k] = {"path": p, "sha256": sha256_file(p)}
        rec["maps"]["s%d" % k] = {"clips": ids.tolist(), "map": pi.tolist(), "share": share.tolist()}
        for E in ests:
            ep = os.path.join(est_dir, "est_score_%s_s%d.npz" % (E, k))
            if not os.path.isfile(ep):
                raise SystemExit("REFUSED: %s not found" % ep)
            ec, ef, ev = load_est(ep)
            rec["inputs"]["est_%s_s%d" % (E, k)] = {"path": ep, "sha256": sha256_file(ep), "rows": int(len(ec))}
            if not np.array_equal(np.unique(ec), ids):
                raise SystemExit("REFUSED: the clips of %s are not the clips of the dump of seed %d" % (ep, k))
            sv = surrogate_values(ec, ef, ev, ids, pi)
            tab = {"clip_index": ec.copy(), "frame": ef.copy(), "est": sv, "map": pi.copy(), "map_clips": ids.copy()}
            idt = {"clip_index": ec.copy(), "frame": ef.copy(), "est": ev.copy()}
            if tamper is not None:
                tamper(("surr", E, k), tab)
            for nm, arr in (("surr_%s_s%d.npz" % (E, k), tab), ("ident_%s_s%d.npz" % (E, k), idt)):
                np.savez(os.path.join(out_dir, nm), **arr)
            if first is None:
                first = (ec.copy(), ef.copy())
    z0 = np.zeros(len(first[0]))
    np.savez(os.path.join(out_dir, "zero_table.npz"), clip_index=first[0], frame=first[1], est=z0)
    np.savez(os.path.join(out_dir, "const2_table.npz"), clip_index=first[0], frame=first[1], est=z0 + 2.0)
    # ---- the controls read the files back from disk
    for E in ests:
        per_seed = []
        for k in seeds:
            D, ids, counts, totals = dumps[k]
            surr = load_table(os.path.join(out_dir, "surr_%s_s%d.npz" % (E, k)))
            ident = load_table(os.path.join(out_dir, "ident_%s_s%d.npz" % (E, k)))
            est = load_est(os.path.join(est_dir, "est_score_%s_s%d.npz" % (E, k)))
            ok1, d1 = check_u1(surr, est, ids, counts, totals)
            rec["u1"]["%s_s%d" % (E, k)] = dict(d1, passed=ok1)
            print("%s u1 %s s%d %s" % ("PASS" if ok1 else "FAIL", E, k, json.dumps(d1)), flush=True)
            if not ok1:
                failed.append("u1-%s-s%d" % (E, k))
            per_seed.append(u4_rows(CF, D, HS[E], surr, ident))
        ok4, d4 = check_u4(per_seed)
        rec["u4"][E] = dict(d4, passed=ok4, H=HS[E])
        print("%s u4 %s H %d r_surrogate %.4f r_estimate %.4f rows %d/%d (band +-%.2f)" % ("PASS" if ok4 else "FAIL", E, HS[E], d4["r_surrogate"], d4["r_estimate"], d4["rows_with_surrogate"], d4["rows_scored"], R_BAND),
              flush=True)
        if not ok4:
            failed.append("u4-%s" % E)
    for nm in sorted(os.listdir(out_dir)):
        rec["files"][nm] = sha256_file(os.path.join(out_dir, nm))
    rec["pass"] = not failed
    rec["failed"] = failed
    with open(os.path.join(out_dir, "SURROGATE_RECORD.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(rec, fh, indent=1, sort_keys=True)
    return rec, failed


# ====================================================================================================== selftest
def synth(root, n_clips=21, seeds=(0, 1), ests=("N10", "L25"), seed=7):
    """synthetic dumps and estimate tables in new folders root/est and root/dump. The label follows the estimate of the clip itself; the dump has p, so every clip has a share."""
    rg = np.random.default_rng(seed)
    os.makedirs(os.path.join(root, "est"))
    os.makedirs(os.path.join(root, "dump"))
    lens = {c: int(rg.integers(100, 105)) for c in range(n_clips)}
    for k in seeds:
        cid, ci, fr = [], [], []
        for c in range(n_clips):
            cid += [c] * lens[c]
            ci += [c] * lens[c]
            fr += list(range(lens[c]))
        cid, ci, fr = np.array(cid, np.int64), np.array(ci, np.int64), np.array(fr, np.int64)
        n = len(cid)
        share = rg.uniform(0.2, 0.9, n_clips)
        p = np.where(rg.uniform(size=n) < share[cid], 0.9, 0.1)
        ev = {}
        for E in ests:
            v = rg.normal(0, 1.3, n)
            v[fr < HS[E] + 4] = np.nan
            ev[E] = v
            np.savez(os.path.join(root, "est", "est_score_%s_s%d.npz" % (E, k)), clip_index=ci, frame=fr, est=v)
        lab = np.where(np.isfinite(ev["N10"]), ev["N10"], 0.0) * 0.6 + rg.normal(0, 0.5, n)
        lab25 = lab.copy()
        np.savez(os.path.join(root, "dump", "reader_s%d_score.npz" % k), cid=cid, clip_index=ci, frame=fr, p=p, gap=rg.uniform(5, 50, n), clos=rg.normal(0, 0.1, n), valid=np.ones(n, bool),
                 lab_present=np.ones(n, bool), lab_gap=rg.uniform(5, 50, n), lab_closing=lab25, lab_run=np.full(n, 5, np.int64), clips=np.array(["clip%03d" % c for c in range(n_clips)]))
    return root


def _label_from_est(root, E, seeds=(0, 1)):
    """selftest helper: sets lab_closing of the dumps to the estimate of estimator E (plus noise) so that a surrogate equal to the estimate has a high r and an honest one has none."""
    rg = np.random.default_rng(3)
    for k in seeds:
        p = os.path.join(root, "dump", "reader_s%d_score.npz" % k)
        z = {a: b for a, b in load_table(p).items()}
        e = load_est(os.path.join(root, "est", "est_score_%s_s%d.npz" % (E, k)))[2]
        z["lab_closing"] = np.where(np.isfinite(e), e, 0.0) * 0.7 + rg.normal(0, 0.5, len(e))
        np.savez(p, **z)


def selftest():
    res = []

    def need(name, cond):
        print(("PASS " if cond else "FAIL ") + name, flush=True)
        res.append(bool(cond))

    # pure map properties on small cases
    for n in (2, 4, 5, 7, 11, 141):
        ids = np.arange(n) * 3 + 1
        sh = np.random.default_rng(n).uniform(0, 1, n)
        pi = compute_map(ids, sh)
        need("map n=%d is a bijection without a fixed point" % n, sorted(pi.tolist()) == sorted(ids.tolist()) and not (pi == ids).any())
    ids = np.arange(5)
    pi = compute_map(ids, np.array([0.5, 0.1, 0.9, 0.1, 0.3]))   # order by (share, id): 1, 3, 4, 0, 2 -> 1<->3, cycle 4->0->2->4
    need("map of a known case", pi.tolist() == [2, 3, 4, 1, 0])
    need("ties are broken by the clip index (five clips, all shares equal)", compute_map(np.arange(5), np.zeros(5)).tolist() == [1, 0, 3, 4, 2])
    root = tempfile.mkdtemp(prefix="tmp_selfsurr_", dir=HERE)
    synth(root)
    _label_from_est(root, "N10")
    # make L25 label-following too in a second pooled sense is not needed: u4 reads each estimator against the same label
    rec, failed = run(os.path.join(root, "est"), os.path.join(root, "dump"), os.path.join(root, "out_clean"), (0, 1), ("N10",), fit_dir="")
    need("clean case: u1 and u4 pass", not failed and rec["u4"]["N10"]["r_estimate"] > 0.3 and abs(rec["u4"]["N10"]["r_surrogate"]) < R_BAND)
    # planted 1: a fixed point
    def fixed_point(kind, tab):
        tab["map"] = tab["map"].copy()
        tab["map"][0] = tab["map_clips"][0]
    rec, failed = run(os.path.join(root, "est"), os.path.join(root, "dump"), os.path.join(root, "out_fixed"), (0, 1), ("N10",), tamper=fixed_point)
    need("planted fixed point fails u1", any(f.startswith("u1-") for f in failed))
    # planted 2: a value from the wrong clip
    def wrong_clip(kind, tab):
        v = tab["est"].copy()
        j = int(np.nonzero(np.isfinite(v))[0][50])
        v[j] = v[j] + 0.001
        tab["est"] = v
    rec, failed = run(os.path.join(root, "est"), os.path.join(root, "dump"), os.path.join(root, "out_wrong"), (0, 1), ("N10",), tamper=wrong_clip)
    need("planted value from the wrong clip fails u1", any(f.startswith("u1-") for f in failed))
    # planted 3: the surrogate equals the estimate
    def equal_est(kind, tab):
        e = load_est(os.path.join(root, "est", "est_score_%s_s%d.npz" % (kind[1], kind[2])))
        tab["est"] = e[2].copy()
    rec, failed = run(os.path.join(root, "est"), os.path.join(root, "dump"), os.path.join(root, "out_same"), (0, 1), ("N10",), tamper=equal_est)
    need("planted surrogate equal to the estimate fails u4", "u4-N10" in failed and rec["u4"]["N10"]["r_surrogate"] > R_BAND)
    # the outputs refuse an existing folder
    try:
        run(os.path.join(root, "est"), os.path.join(root, "dump"), os.path.join(root, "out_clean"), (0, 1), ("N10",))
        need("an existing out folder is refused", False)
    except SystemExit as e:
        need("an existing out folder is refused", "REFUSED" in str(e))
    # zero and const tables
    z = load_table(os.path.join(root, "out_clean", "zero_table.npz"))
    c2 = load_table(os.path.join(root, "out_clean", "const2_table.npz"))
    need("zero and const2 tables", bool((z["est"] == 0.0).all() and (c2["est"] == 2.0).all() and len(z["est"]) == len(c2["est"]) > 0))
    sh = share_of({"clip_index": np.array([4, 4, 4, 4, 9, 9]), "p": np.array([0.9, 0.1, 0.5, 0.49, 0.2, 0.7])})
    need("share of frames with p >= 0.5 (0.5 counts)", sh[0].tolist() == [4, 9] and sh[1].tolist() == [0.5, 0.5] and sh[2] == {4: 2, 9: 1})
    idt = load_table(os.path.join(root, "out_clean", "ident_N10_s0.npz"))
    e0 = load_est(os.path.join(root, "est", "est_score_N10_s0.npz"))
    need("the identity table is the estimate itself, row for row", bool(np.array_equal(idt["clip_index"], e0[0]) and np.array_equal(idt["frame"], e0[1]) and np.array_equal(idt["est"], e0[2], equal_nan=True)))
    ok = all(res)
    print("ZZ SURROGATE-SELFTEST " + ("PASS" if ok else "FAIL") + " checks %d failed %d tmp %s" % (len(res), len(res) - sum(res), os.path.basename(root)))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--est-dir")
    ap.add_argument("--dump-dir")
    ap.add_argument("--out-dir", default="/workspace/v8proof/closeread/est_shuf")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--ests", default="N10,L25")
    ap.add_argument("--fit-dir", default="")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if not a.est_dir or not a.dump_dir:
        print("ZZ SURROGATE STOP missing --est-dir or --dump-dir")
        sys.exit(3)
    seeds = [int(x) for x in a.seeds.split(",") if x != ""]
    ests = [x for x in a.ests.split(",") if x != ""]
    for e in ests:
        if e not in HS:
            raise SystemExit("REFUSED: estimator %s is not one of %s" % (e, sorted(HS)))
    rec, failed = run(a.est_dir, a.dump_dir, a.out_dir, seeds, ests, a.fit_dir)
    print("ZZ SURROGATE " + ("PASS" if not failed else "FAIL " + "; ".join(failed)))
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
