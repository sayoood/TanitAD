r"""dscale_precheck.py -- CPU-only precheck of card PREREG_V8F_PROOF_ADD45 (DSCALE-LEAD) with note N1. numpy and the standard library only (no torch, no GPU). Tier T0.

  python dscale_precheck.py --union-cache DIR --bset-list FILE --lead-union NPZ --lead-corpus NPZ --eval-manifest FILE [--eval-ids FILE] [--join-union NPZ]
        [--hash NAME=PATH ...] --out JSON [--plant] [--n-clips 200] [--seed 45] [--tol 1e-4] [--expect-union 14512] [--expect-corpus 4512]
Also reachable as `tok_probe_u.py precheck ...` (same arguments).

Checks (every one is written to the JSON with its numbers; exit 0 = all pass, 2 = one failed, 3 = an input is missing):
  counts : the union manifest lists --expect-union clips; the corpus id list holds --expect-corpus ids, every one in the union manifest.
  d1     : the number of clip ids shared between the evaluation ids (eval manifest, and --eval-ids) and the union manifest is exactly 0. --plant adds one union id to the evaluation ids: d1 must then FAIL (exit 2).
  d2     : (note N1) for --n-clips corpus clips drawn with a fixed seed, the lead flag and the gap of the union lead file equal those of the corpus lead file at the same frame index k (tolerance --tol);
           the rows present in only one of the two files for these clips are counted and must be 0; at least 200 clips are compared, else refused.
  d2j    : (with --join-union) the join file's lead_gap at row off + j equals the lead file's gap where lead == 1 (else NaN) at k = 2 j, for the same sampled clips: the probe reads its labels from the JOIN, not from the lead file.
  hashes : sha256 of every input file.
Lead file keys: `sha12`, `k`, `lead`, `gap` (study tree v8f_mini_data.py:176 and :270-272; written by stack/scripts/build_v9_labels.py cmd_lead_pass 1111-1140 at cdbff971, per note N1). A row exists = the frame has a label; k = raw frame = 2 * cache step.
The same-clock check of the tree owner session (shift 0 passes, planted shifts fail) is NOT recomputed here: pass its result file with --clock-check-json and it is recorded with its sha256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

LEAD_KEYS = ("sha12", "k", "lead", "gap")
D2_COLUMNS = ("lead", "gap")                      # the two columns d2 compares; the mutation list changes this line
MIN_CLIPS = 200


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def sha12(clip_id):
    return hashlib.sha256(str(clip_id).encode("utf-8")).hexdigest()[:12]


def read_ids(path):
    return [ln.strip() for ln in open(path, encoding="utf-8") if ln.strip() and not ln.lstrip().startswith("#")]


def read_manifest_ids(path):
    ids = []
    for ln in open(path, encoding="utf-8"):
        if ln.strip():
            ids.append(json.loads(ln)["clip"])
    return list(dict.fromkeys(ids))


def find_manifest(cache):
    for p in (os.path.join(cache, "train", "manifest.jsonl"), os.path.join(cache, "manifest.jsonl")):
        if os.path.exists(p):
            return p
    raise SystemExit("MISSING: no manifest.jsonl under %s (looked in train/ and the root)" % cache)


class LeadFile:
    """A lead npz sorted by sha12: rows of one clip as (k, lead, gap) arrays."""

    def __init__(self, path):
        z = np.load(path, allow_pickle=True)
        missing = [k for k in LEAD_KEYS if k not in z.files]
        if missing:
            raise SystemExit("MISSING: lead file %s has no key %s (has %s)" % (path, missing, z.files))
        sh = np.asarray(z["sha12"]).astype(str)
        order = np.argsort(sh, kind="stable")
        self.sh = sh[order]
        self.cols = {"k": np.asarray(z["k"], np.int64)[order], "lead": np.asarray(z["lead"], np.int64)[order], "gap": np.asarray(z["gap"], np.float64)[order]}
        self.uniq, self.start = np.unique(self.sh, return_index=True)
        self.stop = np.append(self.start[1:], len(self.sh))
        self.where = {str(u): (int(a), int(b)) for u, a, b in zip(self.uniq, self.start, self.stop)}
        self.n_rows = int(len(self.sh))

    def clip(self, cid):
        r = self.where.get(sha12(cid))
        if r is None:
            return None
        o = np.argsort(self.cols["k"][r[0]:r[1]], kind="stable")
        return {c: self.cols[c][r[0]:r[1]][o] for c in ("k", "lead", "gap")}


def compare_clip(a, b, tol, columns=D2_COLUMNS):
    """a, b = clip dicts of the union and the corpus file. Returns (rows compared, rows in one file only, mismatching rows)."""
    ka, kb = a["k"], b["k"]
    common, ia, ib = np.intersect1d(ka, kb, return_indices=True)
    only = int(len(ka) - len(common) + len(kb) - len(common))
    bad = np.zeros(len(common), bool)
    for c in columns:
        x, y = np.asarray(a[c], np.float64)[ia], np.asarray(b[c], np.float64)[ib]
        bad |= ~np.isclose(x, y, rtol=0.0, atol=tol, equal_nan=True)
    return int(len(common)), only, int(bad.sum())


def check_d2(lead_u, lead_c, corpus_ids, n_clips, seed, tol):
    rng = np.random.default_rng(seed)
    pool = [c for c in corpus_ids if sha12(c) in lead_c.where]
    if len(pool) < n_clips or n_clips < MIN_CLIPS:
        return {"pass": False, "reason": "REFUSED: %d corpus clips with rows in the corpus lead file, %d asked, minimum %d" % (len(pool), n_clips, MIN_CLIPS), "sampled": []}
    pick = [pool[i] for i in rng.choice(len(pool), size=n_clips, replace=False)]
    n_rows = n_only = n_bad = n_absent = 0
    bad_clips = []
    for c in pick:
        a, b = lead_u.clip(c), lead_c.clip(c)
        if a is None:
            n_absent += 1
            bad_clips.append(c)
            continue
        r, o, m = compare_clip(a, b, tol)
        n_rows, n_only, n_bad = n_rows + r, n_only + o, n_bad + m
        if o or m:
            bad_clips.append(c)
    ok = bool(n_absent == 0 and n_only == 0 and n_bad == 0 and n_rows > 0)
    return {"pass": ok, "n_clips": len(pick), "seed": seed, "tol": tol, "columns": list(D2_COLUMNS), "rows_compared": n_rows, "rows_in_one_file_only": n_only, "rows_mismatching": n_bad,
            "clips_absent_from_union_file": n_absent, "first_bad_clips": bad_clips[:5], "sampled": pick}


def check_d2j(join_path, manifest_ids, lead_u, sampled, tol):
    """The join file (the probe's label source) against the lead file at k = 2 j."""
    z = np.load(join_path, allow_pickle=False)
    for k in ("clip", "off", "n", "lead_valid", "lead_gap"):
        if k not in z.files:
            return {"pass": False, "reason": "join file has no key %s" % k}
    clips = [str(c) for c in z["clip"]]
    where = {c: i for i, c in enumerate(clips)}
    off, n, lg = np.asarray(z["off"], np.int64), np.asarray(z["n"], np.int64), np.asarray(z["lead_gap"], np.float64)
    rows = bad = absent = 0
    for c in sampled:
        ci = where.get(c)
        a = lead_u.clip(c)
        if ci is None or a is None:
            absent += 1
            continue
        lut = {int(k): (int(ld), float(gp)) for k, ld, gp in zip(a["k"], a["lead"], a["gap"])}
        for j in range(int(n[ci])):
            ld, gp = lut.get(2 * j, (0, np.nan))
            want = gp if ld == 1 else np.nan
            have = lg[int(off[ci]) + j]
            rows += 1
            if not np.isclose(have, want, rtol=0.0, atol=tol, equal_nan=True):
                bad += 1
    return {"pass": bool(absent == 0 and bad == 0 and rows > 0), "rows_compared": rows, "rows_mismatching": bad, "clips_absent": absent, "n_clips_in_join": len(clips)}


GAP_MAX_M = 60.0                    # = LEAD_SUP_GAP_MAX_M of the study tree (v8f_mini_data.py, lead_sup_label_arrays): a gap label exists where lead_valid == 1 and 0 <= gap <= 60 m


def join_group_counts(join_path, groups):
    """What the probe will train on, per group of clips (added by the lead before any run; note N2 of the card). The probe's labels are built from the JOIN file alone
    (study tree, v8f_mini_data.py lead_sup_label_arrays): presence = lead_valid == 1, gap = lead_gap where lead_valid == 1 and 0 <= gap <= 60 m. A group whose clips carry
    no presence label or no gap label would add nothing to the arm that is said to train on it, so each group must show both."""
    z = np.load(join_path, allow_pickle=False)
    where = {str(c): i for i, c in enumerate(z["clip"])}
    off, n = np.asarray(z["off"], np.int64), np.asarray(z["n"], np.int64)
    lv, lg = np.asarray(z["lead_valid"]), np.asarray(z["lead_gap"], np.float64)
    out = {"pass": True}
    for name, ids in groups.items():
        frames = pres = gap_ok = absent = with_gap = 0
        for c in ids:
            ci = where.get(c)
            if ci is None:
                absent += 1
                continue
            sl = slice(int(off[ci]), int(off[ci]) + int(n[ci]))
            v, g = lv[sl] == 1, lg[sl]
            ok = v & np.isfinite(g) & (g >= 0.0) & (g <= GAP_MAX_M)
            frames, pres, gap_ok, with_gap = frames + int(n[ci]), pres + int(v.sum()), gap_ok + int(ok.sum()), with_gap + int(ok.any())
        good = bool(len(ids) > 0 and absent == 0 and pres > 0 and gap_ok > 0)
        out[name] = {"clips": len(ids), "clips_absent_from_join": absent, "frames": frames, "presence_frames": pres, "gap_label_frames": gap_ok, "clips_with_a_gap_label": with_gap,
                     "presence_share": round(pres / max(frames, 1), 4), "gap_label_share": round(gap_ok / max(frames, 1), 4), "pass": good}
        out["pass"] = bool(out["pass"] and good)
    return out


def run(a):
    t0 = time.time()
    res = {"card": "PREREG_V8F_PROOF_ADD45 + N1", "tier": "T0 (a probe on cached tokens of logged frames; never driving performance)", "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "checks": {}}
    for nm, p in (("union_manifest", find_manifest(a.union_cache)), ("bset_list", a.bset_list), ("lead_union", a.lead_union), ("lead_corpus", a.lead_corpus), ("eval_manifest", a.eval_manifest)):
        if not os.path.exists(p):
            print("MISSING: %s = %s" % (nm, p))
            return 3
    man_path = find_manifest(a.union_cache)
    union_ids = read_manifest_ids(man_path)
    corpus_ids = read_ids(a.bset_list)
    um = set(union_ids)
    res["checks"]["counts"] = {"pass": bool(len(union_ids) == a.expect_union and len(corpus_ids) == a.expect_corpus and len(set(corpus_ids)) == len(corpus_ids) and all(c in um for c in corpus_ids)),
                                "union_manifest_clips": len(union_ids), "expect_union": a.expect_union, "corpus_ids": len(corpus_ids), "expect_corpus": a.expect_corpus,
                                "corpus_ids_missing_from_union": sum(1 for c in corpus_ids if c not in um)}
    ev = set(read_manifest_ids(a.eval_manifest))
    if a.eval_ids:
        ev |= set(read_ids(a.eval_ids))
    if a.plant:
        ev.add(union_ids[0])
    shared = sorted(ev & um)
    res["checks"]["d1_no_leak"] = {"pass": bool(len(shared) == 0 and len(ev) > 0), "n_eval_ids": len(ev), "n_shared_with_union": len(shared), "first_shared": shared[:3], "planted": bool(a.plant)}
    lu, lc = LeadFile(a.lead_union), LeadFile(a.lead_corpus)
    d2 = check_d2(lu, lc, corpus_ids, a.n_clips, a.seed, a.tol)
    sampled = d2.pop("sampled", [])
    res["checks"]["d2_labels_one_clock"] = d2
    res["checks"]["d2_rows"] = {"lead_union_rows": lu.n_rows, "lead_corpus_rows": lc.n_rows}
    if a.join_union:
        res["checks"]["d2j_join_matches_lead_file"] = check_d2j(a.join_union, union_ids, lu, sampled, a.tol) if sampled else {"pass": False, "reason": "no sampled clips"}
        cs = set(corpus_ids)                                         # the same two checks on the NEW clips (not in the corpus list): the arm that trains on the union must find labels there
        new_ids = [c for c in union_ids if c not in cs]
        pool_new = [c for c in new_ids if sha12(c) in lu.where]
        rng_n = np.random.default_rng(a.seed + 1)
        sampled_new = [pool_new[i] for i in rng_n.choice(len(pool_new), size=a.n_clips, replace=False)] if len(pool_new) >= max(a.n_clips, MIN_CLIPS) else []
        d2n = check_d2j(a.join_union, union_ids, lu, sampled_new, a.tol) if sampled_new else {"pass": False, "reason": "fewer than %d new clips with rows in the union lead file" % MIN_CLIPS}
        d2n.update({"new_clips": len(new_ids), "new_clips_with_rows_in_the_lead_file": len(pool_new), "sampled": len(sampled_new), "seed": a.seed + 1})
        res["checks"]["d2n_join_matches_lead_file_new_clips"] = d2n
        res["checks"]["d2g_join_labels_per_group"] = join_group_counts(a.join_union, {"corpus": sampled, "new": sampled_new}) if (sampled and sampled_new) else {"pass": False, "reason": "a group has no sampled clips"}
    if a.clock_check_json:
        res["clock_check_of_the_tree_owner_session"] = {"path": a.clock_check_json, "sha256": sha256_file(a.clock_check_json) if os.path.exists(a.clock_check_json) else None}
    hashes = {"union_manifest": man_path, "bset_list": a.bset_list, "lead_union": a.lead_union, "lead_corpus": a.lead_corpus, "eval_manifest": a.eval_manifest}
    for kv in a.hash or []:
        k, _, p = kv.partition("=")
        hashes[k] = p
    res["input_sha256"] = {k: (sha256_file(p) if os.path.exists(p) else None) for k, p in hashes.items()}
    res["pass"] = bool(all(c.get("pass") for k, c in res["checks"].items() if "pass" in c))
    res["elapsed_s"] = round(time.time() - t0, 1)
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, indent=1)
    print("precheck: pass %s -> %s" % (res["pass"], a.out), {k: c.get("pass") for k, c in res["checks"].items() if "pass" in c})
    return 0 if res["pass"] else 2


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--union-cache", required=True)
    ap.add_argument("--bset-list", required=True)
    ap.add_argument("--lead-union", required=True)
    ap.add_argument("--lead-corpus", required=True)
    ap.add_argument("--eval-manifest", required=True)
    ap.add_argument("--eval-ids", default="")
    ap.add_argument("--join-union", default="")
    ap.add_argument("--clock-check-json", default="")
    ap.add_argument("--hash", action="append", default=[], help="NAME=PATH, hashed into the JSON (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--plant", action="store_true", help="add one union clip id to the evaluation ids: d1 must FAIL")
    ap.add_argument("--n-clips", type=int, default=200)
    ap.add_argument("--seed", type=int, default=45)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--expect-union", type=int, default=14512)
    ap.add_argument("--expect-corpus", type=int, default=4512)
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
