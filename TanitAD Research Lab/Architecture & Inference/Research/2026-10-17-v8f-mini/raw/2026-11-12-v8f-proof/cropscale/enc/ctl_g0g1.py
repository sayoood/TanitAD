#!/usr/bin/env python3
"""Controls g0 (lists) and g1 (tokens) of card ADD106 (CROPSCALE10K), section 5. Reads files only; never writes a token or a list.

  ctl_g0g1.py run       both controls -> ENC_CONTROLS.json (exit 0 only when both pass)
  ctl_g0g1.py identity  only the three-clip identity clause of g1 (the queue runs it before any new clip is staged) -> ENC_IDENTITY.json
  ctl_g0g1.py --selftest   tiny synthetic data, one planted fault per clause (every clause must flip, the clean case must pass)

g0 clauses (the card's five, plus two consistency clauses the code adds):
  g0_all_in_eligible        every clip of the large list is in train_eligible.txt
  g0_none_is_evaluation     no clip of it is one of the 141 evaluation clips
  g0_pilot_500_unchanged    its first lines are the pilot's 500 lines (same clips, same order, same roles)
  g0_selection_is_pilot_50  its `sel` clips are exactly the pilot's 50 selection clips (no other `sel` line)
  g0_left_out_listed        every eligible clip not in the list is in the record's left_out with a reason, and the two sets partition the eligible set
  g0_no_duplicates          (added) no clip twice; every new line has role fit and the new clips keep the order of train_eligible.txt
  g0_record_matches_file    (added) the record's sha256 of train_final.txt and its counts equal the file's
g1 clauses:
  g1_new_files_shape_type   every new token file: float16 [n, 2, 65, 1024], n = the join's frame count of that clip; no file of other kind
  g1_new_hash_in_manifest   every new token file is in tok/D/MANIFEST.json with its byte size and the sha256 recomputed now; no token file outside the manifest
  g1_pilot_unchanged        every token file of the pilot's manifest has its recorded size and sha256 now
  g1_identity_three_clips   the first three of the pilot's train_final_ids.txt, encoded again by this card's call (check_tok), equal the pilot's arm D tokens
                            within 1e-3 in the largest absolute difference (float32 differences of the float16 values)
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = "/workspace/v8proof/cropscale"
PILOT_RUN = "/workspace/v8proof/cropread/run1"
ELIGIBLE = "/workspace/v8proof/cropread/enc_out/lists/train_eligible.txt"
JOIN_TRAIN = "/workspace/v8proof/data/inputs/join_train.npz"
TAIL = (2, 65, 1024)
TOL = 1e-3
N_PILOT_SEL = 50


# ------------------------------------------------------------------------------------------------ small readers
def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def read_ids(path):
    out = []
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            t = ln.split()
            if t:
                out.append(t[0])
    return out


def read_rows(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            t = ln.split()
            if t:
                rows.append((t[0], t[1] if len(t) > 1 else ""))
    return rows


def join_n(path):
    J = dict(np.load(path, allow_pickle=True))
    return {str(c): int(n) for c, n in zip(J["clip"], J["n"])}


def hash_many(paths, workers=4):
    def one(p):
        try:
            return p, sha256_file(p)
        except OSError as e:
            return p, "ERR " + str(e)[:60]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return dict(ex.map(one, paths))


def clause(ok, **kw):
    d = {"pass": bool(ok)}
    d.update(kw)
    return d


# ------------------------------------------------------------------------------------------------ g0
def g0(eligible_p, eval_p, pilot_final_p, large_final_p, record_p, n_sel=N_PILOT_SEL):
    elig, ev = read_ids(eligible_p), set(read_ids(eval_p))
    prow, rows = read_rows(pilot_final_p), read_rows(large_final_p)
    rec = json.load(open(record_p, encoding="utf-8"))
    ids = [c for c, _ in rows]
    es = set(elig)
    out = {}
    bad = [c for c in ids if c not in es]
    out["g0_all_in_eligible"] = clause(not bad, n_not_in_eligible=len(bad), examples=bad[:3])
    bad = [c for c in ids if c in ev]
    out["g0_none_is_evaluation"] = clause(not bad, n_evaluation_clips=len(bad), examples=bad[:3])
    out["g0_pilot_500_unchanged"] = clause(len(prow) > 0 and rows[:len(prow)] == prow, n_pilot_lines=len(prow),
                                           first_difference=next((i for i, (a, b) in enumerate(zip(prow, rows)) if a != b), None))
    psel = {c for c, r in prow if r == "sel"}
    lsel = {c for c, r in rows if r == "sel"}
    out["g0_selection_is_pilot_50"] = clause(lsel == psel and len(psel) == n_sel and sum(1 for _, r in rows if r == "sel") == n_sel,
                                             n_pilot_sel=len(psel), n_large_sel=sum(1 for _, r in rows if r == "sel"))
    left = rec.get("left_out", [])
    lc = [x.get("clip") for x in left]
    missing_reason = [x.get("clip") for x in left if not str(x.get("reason", "")).strip()]
    outside = [c for c in elig if c not in set(ids)]
    unlisted = sorted(set(outside) - set(lc))
    extra = sorted((set(lc) & set(ids)) | (set(lc) - es))
    out["g0_left_out_listed"] = clause(not unlisted and not extra and not missing_reason and len(lc) == len(set(lc)),
                                       n_left_out=len(lc), n_unlisted=len(unlisted), n_listed_but_in_list_or_not_eligible=len(extra),
                                       n_without_reason=len(missing_reason), n_eligible=len(elig), n_large=len(ids))
    new = [c for c, _ in rows[len(prow):]]
    order = {c: i for i, c in enumerate(elig)}
    in_order = all(c in order for c in new) and all(order[a] < order[b] for a, b in zip(new, new[1:]))
    out["g0_no_duplicates"] = clause(len(ids) == len(set(ids)) and all(r == "fit" for _, r in rows[len(prow):]) and in_order,
                                     n_lines=len(ids), n_distinct=len(set(ids)), new_roles_all_fit=all(r == "fit" for _, r in rows[len(prow):]),
                                     new_in_eligible_order=bool(in_order))
    body = "".join("%s %s\n" % (c, r) for c, r in rows).encode("utf-8")
    ok = (rec.get("sha256", {}).get("train_final") == hashlib.sha256(body).hexdigest() and rec.get("n_large_list") == len(rows)
          and rec.get("n_fit") == sum(1 for _, r in rows if r == "fit") and rec.get("n_sel") == sum(1 for _, r in rows if r == "sel"))
    out["g0_record_matches_file"] = clause(ok)
    out["pass"] = all(v["pass"] for v in out.values())
    out["n_large_list"], out["n_fit"], out["n_sel"] = len(rows), sum(1 for _, r in rows if r == "fit"), sum(1 for _, r in rows if r == "sel")
    return out


# ------------------------------------------------------------------------------------------------ g1
def check_token_array(path, n_expected):
    """-> why (None when fine)."""
    try:
        a = np.load(path, mmap_mode="r")
    except Exception as e:                                                # noqa: BLE001
        return "not readable as .npy: %s" % str(e)[:60]
    if a.ndim != 4 or tuple(a.shape[1:]) != TAIL:
        return "shape %s, expected [n, 2, 65, 1024]" % (list(a.shape),)
    if a.dtype != np.float16:
        return "dtype %s, expected float16" % a.dtype
    if n_expected is not None and a.shape[0] != n_expected:
        return "%d frames, the join has %d" % (a.shape[0], n_expected)
    return None


def g1_new(new_clips, tok_d, nmap, workers=4):
    man_p = os.path.join(tok_d, "MANIFEST.json")
    man = json.load(open(man_p, encoding="utf-8"))["clips"] if os.path.exists(man_p) else {}
    on_disk = sorted(f[:-4] for f in os.listdir(tok_d) if f.endswith(".npy")) if os.path.isdir(tok_d) else []
    shape_bad, hash_bad = [], []
    for c in new_clips:
        p = os.path.join(tok_d, c + ".npy")
        if not os.path.exists(p):
            shape_bad.append({"clip": c, "why": "no token file"})
            continue
        why = check_token_array(p, nmap.get(c))
        if c not in nmap:
            why = (why + "; " if why else "") + "clip not in the join"
        if why:
            shape_bad.append({"clip": c, "why": why})
    # every token file on disk counts as a new token file: a file outside the list is also checked
    stray = sorted(set(on_disk) - set(new_clips))
    for c in stray:
        why = check_token_array(os.path.join(tok_d, c + ".npy"), nmap.get(c))
        shape_bad.append({"clip": c, "why": "file not in the large list" + ("; " + why if why else "")})
    paths = [os.path.join(tok_d, c + ".npy") for c in sorted(set(new_clips) | set(on_disk)) if os.path.exists(os.path.join(tok_d, c + ".npy"))]
    shas = hash_many(paths, workers)
    for c in sorted(set(new_clips) | set(on_disk)):
        p = os.path.join(tok_d, c + ".npy")
        if not os.path.exists(p):
            continue
        m = man.get(c)
        if m is None:
            hash_bad.append({"clip": c, "why": "not in MANIFEST.json"})
        elif os.path.getsize(p) != m.get("bytes"):
            hash_bad.append({"clip": c, "why": "byte size differs from the manifest"})
        elif shas[p] != m.get("sha256"):
            hash_bad.append({"clip": c, "why": "sha256 differs from the manifest"})
    for c in new_clips:
        if not os.path.exists(os.path.join(tok_d, c + ".npy")) and not any(x["clip"] == c for x in hash_bad):
            hash_bad.append({"clip": c, "why": "no token file"})
    return (clause(not shape_bad, n_checked=len(paths), n_bad=len(shape_bad), examples=shape_bad[:3]),
            clause(not hash_bad, n_checked=len(paths), n_bad=len(hash_bad), examples=hash_bad[:3]))


def g1_pilot(pilot_tok_d, workers=4):
    mp = os.path.join(pilot_tok_d, "MANIFEST.json")
    man = json.load(open(mp, encoding="utf-8"))["clips"]
    bad, paths = [], []
    for c, m in sorted(man.items()):
        p = os.path.join(pilot_tok_d, c + ".npy")
        if not os.path.exists(p):
            bad.append({"clip": c, "why": "file missing"})
        elif os.path.getsize(p) != m["bytes"]:
            bad.append({"clip": c, "why": "byte size differs"})
        else:
            paths.append(p)
    shas = hash_many(paths, workers)
    for c, m in sorted(man.items()):
        p = os.path.join(pilot_tok_d, c + ".npy")
        if p in shas and shas[p] != m["sha256"]:
            bad.append({"clip": c, "why": "sha256 differs"})
    return clause(not bad and len(man) > 0, n_in_manifest=len(man), n_bad=len(bad), examples=bad[:3])


def g1_identity(ids, pilot_tok_d, check_tok_d, tol=TOL):
    rows, ok = [], len(ids) == 3
    for c in ids:
        pp, cp = os.path.join(pilot_tok_d, c + ".npy"), os.path.join(check_tok_d, c + ".npy")
        if not (os.path.exists(pp) and os.path.exists(cp)):
            rows.append({"clip": c, "why": "file missing", "max_abs_diff": None})
            ok = False
            continue
        try:
            a, b = np.load(pp, mmap_mode="r"), np.load(cp, mmap_mode="r")
        except Exception as e:                                            # noqa: BLE001
            rows.append({"clip": c, "why": "not readable: " + str(e)[:60], "max_abs_diff": None})
            ok = False
            continue
        if a.shape != b.shape or a.dtype != b.dtype:
            rows.append({"clip": c, "why": "shape or type differs %s %s / %s %s" % (list(a.shape), a.dtype, list(b.shape), b.dtype), "max_abs_diff": None})
            ok = False
            continue
        mx = 0.0
        for i in range(0, a.shape[0], 8):
            d = np.abs(np.asarray(a[i:i + 8], dtype=np.float32) - np.asarray(b[i:i + 8], dtype=np.float32))
            if not np.isfinite(d).all():
                mx = float("inf")
                break
            mx = max(mx, float(d.max()))
        rows.append({"clip": c, "n_frames": int(a.shape[0]), "max_abs_diff": mx})
        ok = ok and mx <= tol
    return clause(ok, tolerance=tol, clips=rows)


# ------------------------------------------------------------------------------------------------ drivers
def run(a):
    t0 = time.time()
    lists = os.path.join(a.root, "lists")
    res = {"card": "ADD106", "controls": {}}
    res["controls"]["g0"] = g0(a.eligible, a.eval, a.pilot_final, os.path.join(lists, "train_final.txt"), os.path.join(lists, "LISTS_LARGE_RECORD.json"))
    prow = read_rows(a.pilot_final)
    rows = read_rows(os.path.join(lists, "train_final.txt"))
    new = [c for c, _ in rows[len(prow):]]
    tok_d = os.path.join(a.root, "enc_out", "tok", "D")
    s, h = g1_new(new, tok_d, join_n(a.join_train), a.workers)
    cl = {"g1_new_files_shape_type": s, "g1_new_hash_in_manifest": h, "g1_pilot_unchanged": g1_pilot(os.path.join(a.pilot_run, "tok", "D"), a.workers),
          "g1_identity_three_clips": g1_identity(read_ids(os.path.join(a.pilot_run, "lists", "train_final_ids.txt"))[:3],
                                                 os.path.join(a.pilot_run, "tok", "D"), os.path.join(a.root, "check_tok", "tok", "D"))}
    cl["pass"] = all(v["pass"] for v in cl.values())
    res["controls"]["g1"] = cl
    res["pass_g0"], res["pass_g1"] = res["controls"]["g0"]["pass"], cl["pass"]
    res["pass_all"] = bool(res["pass_g0"] and res["pass_g1"])
    res["seconds"] = round(time.time() - t0, 1)
    with open(a.out or os.path.join(a.root, "ENC_CONTROLS.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({"pass_g0": res["pass_g0"], "pass_g1": res["pass_g1"], "pass_all": res["pass_all"]}))
    return 0 if res["pass_all"] else 3


def identity(a):
    ids = read_ids(a.ids)
    r = g1_identity(ids, os.path.join(a.pilot_run, "tok", "D"), os.path.join(a.root, "check_tok", "tok", "D"))
    with open(a.out or os.path.join(a.root, "ENC_IDENTITY.json"), "w", encoding="utf-8") as fh:
        json.dump(r, fh, indent=1)
    print(json.dumps({"identity_pass": r["pass"], "clips": r["clips"]}))
    return 0 if r["pass"] else 3


# ------------------------------------------------------------------------------------------------ synthetic fixture and selftest
def make_fixture(d, seed=0):
    """Tiny world: eligible c0..c7; evaluation e0,e1; pilot = c0..c3 (c3 is `sel`); new = c4..c7 where c6 did not stage and c7 did not encode."""
    rng = np.random.RandomState(seed)
    P = lambda *x: os.path.join(d, *x)                                   # noqa: E731
    for s in ("lists", "enc_out/tok/D", "pilot/tok/D", "pilot/lists", "check_tok/tok/D", "pl"):
        os.makedirs(P(*s.split("/")), exist_ok=True)

    def tok(path, n):
        a = rng.randn(n, *TAIL).astype(np.float16)
        np.save(path, a)
        return a

    elig = ["c%d" % i for i in range(8)]
    open(P("pl", "eligible.txt"), "w").write("".join(c + "\n" for c in elig))
    open(P("pl", "eval.txt"), "w").write("e0\ne1\n")
    prow = [("c0", "fit"), ("c1", "fit"), ("c2", "fit"), ("c3", "sel")]
    open(P("pl", "pilot_final.txt"), "w").write("".join("%s %s\n" % x for x in prow))
    open(P("pilot", "lists", "train_final_ids.txt"), "w").write("".join(c + "\n" for c, _ in prow))
    nmap = {c: 1 for c in elig + ["e0", "e1"]}
    nmap["c1"] = 2
    pman = {}
    for c in [x for x, _ in prow] + ["e0", "e1"]:
        p = P("pilot", "tok", "D", c + ".npy")
        tok(p, nmap[c])
        pman[c] = {"n_frames": nmap[c], "sha256": sha256_file(p), "bytes": os.path.getsize(p)}
    json.dump({"clips": pman}, open(P("pilot", "tok", "D", "MANIFEST.json"), "w"))
    nman = {}
    for c in ("c4", "c5"):
        p = P("enc_out", "tok", "D", c + ".npy")
        tok(p, nmap[c])
        nman[c] = {"n_frames": nmap[c], "sha256": sha256_file(p), "bytes": os.path.getsize(p)}
    json.dump({"clips": nman}, open(P("enc_out", "tok", "D", "MANIFEST.json"), "w"))
    for c, _ in prow[:3]:                                                 # the identity clips: a copy of the pilot's files
        np.save(P("check_tok", "tok", "D", c + ".npy"), np.load(P("pilot", "tok", "D", c + ".npy")))
    rows = prow + [("c4", "fit"), ("c5", "fit")]
    body = "".join("%s %s\n" % x for x in rows)
    open(P("lists", "train_final.txt"), "w").write(body)
    rec = {"n_large_list": len(rows), "n_fit": 5, "n_sel": 1, "sha256": {"train_final": hashlib.sha256(body.encode()).hexdigest()},
           "left_out": [{"clip": "c6", "stage": "did not stage", "reason": "did not stage: no obstacle source for this clip"},
                        {"clip": "c7", "stage": "staged", "reason": "did not encode: boom"}]}
    json.dump(rec, open(P("lists", "LISTS_LARGE_RECORD.json"), "w"))
    return nmap


def eval_all(d, nmap):
    P = lambda *x: os.path.join(d, *x)                                   # noqa: E731
    r0 = g0(P("pl", "eligible.txt"), P("pl", "eval.txt"), P("pl", "pilot_final.txt"), P("lists", "train_final.txt"), P("lists", "LISTS_LARGE_RECORD.json"), n_sel=1)
    rows = read_rows(P("lists", "train_final.txt"))
    new = [c for c, _ in rows[4:]]
    s, h = g1_new(new, P("enc_out", "tok", "D"), nmap, 2)
    r1 = {"g1_new_files_shape_type": s, "g1_new_hash_in_manifest": h, "g1_pilot_unchanged": g1_pilot(P("pilot", "tok", "D"), 2),
          "g1_identity_three_clips": g1_identity(["c0", "c1", "c2"], P("pilot", "tok", "D"), P("check_tok", "tok", "D"))}
    return r0, r1


def _flip_last_byte(path):
    with open(path, "r+b") as fh:
        fh.seek(-1, 2)
        b = fh.read(1)
        fh.seek(-1, 2)
        fh.write(bytes([b[0] ^ 0x40]))


def selftest(base=None):
    """-> list of (name, expected_clause, ok). Clean case: every clause passes; each planted fault flips its clause."""
    results = []
    base = base or tempfile.mkdtemp(prefix="g0g1_")
    k = [0]

    def world():
        k[0] += 1
        d = os.path.join(base, "w%d" % k[0])
        os.makedirs(d)
        return d, make_fixture(d)

    d, nmap = world()
    r0, r1 = eval_all(d, nmap)
    results.append(("clean g0", "pass", bool(r0["pass"])))
    results.append(("clean g1", "pass", all(v["pass"] for v in r1.values())))

    def planted(name, clause_name, which, mutate):
        d, nmap = world()
        mutate(d, nmap)
        r0, r1 = eval_all(d, nmap)
        r = (r0 if which == 0 else r1)[clause_name]["pass"]
        results.append((name, clause_name, not r))

    P = lambda d, *x: os.path.join(d, *x)                                # noqa: E731

    def rewrite(d, fn):
        p = P(d, "lists", "train_final.txt")
        rows = read_rows(p)
        rows = fn(rows)
        body = "".join("%s %s\n" % x for x in rows)
        open(p, "w").write(body)
        rec = json.load(open(P(d, "lists", "LISTS_LARGE_RECORD.json")))
        rec["sha256"]["train_final"] = hashlib.sha256(body.encode()).hexdigest()
        rec["n_large_list"] = len(rows)
        rec["n_fit"] = sum(1 for _, r in rows if r == "fit")
        rec["n_sel"] = sum(1 for _, r in rows if r == "sel")
        json.dump(rec, open(P(d, "lists", "LISTS_LARGE_RECORD.json"), "w"))

    planted("a clip outside train_eligible", "g0_all_in_eligible", 0, lambda d, n: rewrite(d, lambda r: r + [("zz", "fit")]))
    planted("an evaluation clip in the list", "g0_none_is_evaluation", 0, lambda d, n: rewrite(d, lambda r: r + [("e0", "fit")]))
    planted("pilot lines reordered", "g0_pilot_500_unchanged", 0, lambda d, n: rewrite(d, lambda r: [r[1], r[0]] + r[2:]))
    planted("a pilot role changed", "g0_pilot_500_unchanged", 0, lambda d, n: rewrite(d, lambda r: [(r[0][0], "sel")] + r[1:]))
    planted("a new clip with role sel", "g0_selection_is_pilot_50", 0, lambda d, n: rewrite(d, lambda r: r[:-1] + [(r[-1][0], "sel")]))
    planted("a left-out entry removed", "g0_left_out_listed", 0,
            lambda d, n: json.dump({**json.load(open(P(d, "lists", "LISTS_LARGE_RECORD.json"))), "left_out": [json.load(open(P(d, "lists", "LISTS_LARGE_RECORD.json")))["left_out"][0]]},
                                   open(P(d, "lists", "LISTS_LARGE_RECORD.json"), "w")))
    planted("a left-out reason empty", "g0_left_out_listed", 0,
            lambda d, n: json.dump({**json.load(open(P(d, "lists", "LISTS_LARGE_RECORD.json"))), "left_out": [{"clip": "c6", "reason": " "}, {"clip": "c7", "reason": "x"}]},
                                   open(P(d, "lists", "LISTS_LARGE_RECORD.json"), "w")))
    planted("a duplicated line", "g0_no_duplicates", 0, lambda d, n: rewrite(d, lambda r: r + [r[-1]]))
    planted("a new clip out of eligible order", "g0_no_duplicates", 0, lambda d, n: rewrite(d, lambda r: r[:4] + [r[5], r[4]]))
    planted("record hash edited", "g0_record_matches_file", 0,
            lambda d, n: json.dump({**json.load(open(P(d, "lists", "LISTS_LARGE_RECORD.json"))), "sha256": {"train_final": "0" * 64}}, open(P(d, "lists", "LISTS_LARGE_RECORD.json"), "w")))

    def to_float32(d, n):
        np.save(P(d, "enc_out", "tok", "D", "c4.npy"), np.load(P(d, "enc_out", "tok", "D", "c4.npy")).astype(np.float32))
    planted("a float32 new token file", "g1_new_files_shape_type", 1, to_float32)

    def bad_shape(d, n):
        np.save(P(d, "enc_out", "tok", "D", "c5.npy"), np.zeros((1, 2, 64, 1024), np.float16))
    planted("a new token file of the wrong shape", "g1_new_files_shape_type", 1, bad_shape)

    def bad_n(d, n):
        n["c4"] = 5
    planted("frame count differs from the join", "g1_new_files_shape_type", 1, bad_n)
    planted("a new token byte flipped", "g1_new_hash_in_manifest", 1, lambda d, n: _flip_last_byte(P(d, "enc_out", "tok", "D", "c4.npy")))

    def stray(d, n):
        np.save(P(d, "enc_out", "tok", "D", "c9.npy"), np.zeros((1,) + TAIL, np.float16))
    planted("a token file outside the manifest", "g1_new_hash_in_manifest", 1, stray)
    planted("a pilot token byte flipped", "g1_pilot_unchanged", 1, lambda d, n: _flip_last_byte(P(d, "pilot", "tok", "D", "c2.npy")))

    def big_diff(d, n):
        a = np.load(P(d, "check_tok", "tok", "D", "c1.npy")).copy()
        a[0, 0, 0, 0] = np.float16(float(a[0, 0, 0, 0]) + 0.5)
        np.save(P(d, "check_tok", "tok", "D", "c1.npy"), a)
    planted("a check token off by 0.5", "g1_identity_three_clips", 1, big_diff)
    planted("a check token that is not an array", "g1_identity_three_clips", 1, lambda d, n: open(P(d, "check_tok", "tok", "D", "c0.npy"), "wb").write(b"not an array"))
    # a difference below the tolerance must still pass
    d, nmap = world()
    a = np.load(P(d, "check_tok", "tok", "D", "c1.npy")).copy()
    a = (a.astype(np.float32) + 2e-4).astype(np.float16)
    np.save(P(d, "check_tok", "tok", "D", "c1.npy"), a)
    r = g1_identity(["c0", "c1", "c2"], P(d, "pilot", "tok", "D"), P(d, "check_tok", "tok", "D"))
    mx = max(x["max_abs_diff"] for x in r["clips"])
    results.append(("float16 rounding difference of %.1e is inside 1e-3" % mx, "g1_identity_three_clips", bool(r["pass"]) and mx < TOL))
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", nargs="?", choices=["run", "identity"])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--pilot-run", default=PILOT_RUN)
    ap.add_argument("--eligible", default=ELIGIBLE)
    ap.add_argument("--eval", default=PILOT_RUN + "/lists/eval141.txt")
    ap.add_argument("--pilot-final", default=PILOT_RUN + "/lists/train_final.txt")
    ap.add_argument("--join-train", default=JOIN_TRAIN)
    ap.add_argument("--ids", default=None, help="identity: the three clip ids (default: the first three of the pilot's train_final_ids.txt)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.selftest:
        res = selftest()
        bad = [r for r in res if not r[2]]
        for n, c, ok in res:
            print(("ok   " if ok else "FAIL ") + c + " :: " + n)
        print("ZZ SELFTEST %s (%d cases)" % ("PASS" if not bad else "FAIL", len(res)))
        return 0 if not bad else 4
    if not a.mode:
        ap.error("mode or --selftest required")
    if a.mode == "identity" and not a.ids:
        a.ids = os.path.join(a.root, "lists", "identity3.txt")
    return {"run": run, "identity": identity}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
