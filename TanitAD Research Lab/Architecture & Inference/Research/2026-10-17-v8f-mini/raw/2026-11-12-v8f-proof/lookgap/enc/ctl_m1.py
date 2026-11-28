#!/usr/bin/env python3
"""Control m1 (encode) of card ADD108 (LOOKGAP-T0), section 5. CPU. Reads files only; never writes a token file or a list.

  ctl_m1.py identity   clause i only  -> ENC_M1_IDENTITY.json  (pass_m1 is null: not all clauses were run)
  ctl_m1.py tokens     clauses ii and iii -> ENC_M1_TOKENS.json (pass_m1 is null)
  ctl_m1.py all        all three clauses -> ENC_M1.json (pass_m1 true only when all three pass)
  ctl_m1.py --selftest tiny synthetic data, one planted fault per clause

Clauses (the card's wording, section 5, m1):
  i_identity_three_clips      for three named clips the token files written by the variant from the pilot's BOX folder (arm name B_ident) have
                              the sha256 (and byte size) that the pilot's manifest of arm B lists. Byte for byte, no tolerance.
  ii_arm_p_files              every token file of arm P: float16 [n, 2, 65, 1024] with n = the frame count of the pilot's token file of that clip
                              (pilot manifest of arm B), its sha256 and byte size are in the manifest of this card, the count of the files is the
                              expected 641, and no .npy file lies outside the 641 clips.
  iii_pilot_arms_unchanged    every token file of the pilot's manifests of arm D and of arm B has its recorded size and sha256 now (8 reader threads).
"""
import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = "/workspace/v8proof/lookgap"
PILOT_RUN = "/workspace/v8proof/cropread/run1"
TAIL = (2, 65, 1024)
N_EXPECTED = 641
WORKERS = 8


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


def hash_many(paths, workers=WORKERS):
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


def read_manifest(path):
    return json.load(open(path, encoding="utf-8"))["clips"]


# ------------------------------------------------------------------------------------------------ clause i
def clause_identity(ids, pilot_man_b, ident_dir):
    rows, ok = [], len(ids) == 3 and len(set(ids)) == 3
    for c in ids:
        p = os.path.join(ident_dir, c + ".npy")
        m = pilot_man_b.get(c)
        if m is None:
            rows.append({"clip": c, "why": "not in the pilot's manifest of arm B"})
            ok = False
        elif not os.path.exists(p):
            rows.append({"clip": c, "why": "no file written by the variant"})
            ok = False
        else:
            sh, sz = sha256_file(p), os.path.getsize(p)
            same = sh == m.get("sha256") and sz == m.get("bytes")
            rows.append({"clip": c, "same_sha256": sh == m.get("sha256"), "same_bytes": sz == m.get("bytes"), "sha256_16": sh[:16]})
            ok = ok and same
    return clause(ok, n_clips=len(ids), clips=rows)


# ------------------------------------------------------------------------------------------------ clause ii
def check_token_array(path, n_expected):
    try:
        a = np.load(path, mmap_mode="r")
    except Exception as e:                                                # noqa: BLE001
        return "not readable as .npy: %s" % str(e)[:60]
    if a.ndim != 4 or tuple(a.shape[1:]) != TAIL:
        return "shape %s, expected [n, 2, 65, 1024]" % (list(a.shape),)
    if a.dtype != np.float16:
        return "dtype %s, expected float16" % a.dtype
    if n_expected is not None and a.shape[0] != n_expected:
        return "%d frames, the pilot's token file has %d" % (a.shape[0], n_expected)
    return None


def clause_arm_p(ids, p_dir, p_man, pilot_man_b, n_expected=N_EXPECTED, workers=WORKERS):
    on_disk = sorted(f[:-4] for f in os.listdir(p_dir) if f.endswith(".npy")) if os.path.isdir(p_dir) else []
    bad = []
    if len(ids) != n_expected or len(set(ids)) != len(ids):
        bad.append({"clip": None, "why": "the clip list has %d lines, %d distinct, expected %d" % (len(ids), len(set(ids)), n_expected)})
    for c in ids:
        p = os.path.join(p_dir, c + ".npy")
        if not os.path.exists(p):
            bad.append({"clip": c, "why": "no token file"})
            continue
        pm = pilot_man_b.get(c)
        why = check_token_array(p, pm.get("n_frames") if pm else None)
        if pm is None:
            why = (why + "; " if why else "") + "clip not in the pilot's manifest of arm B"
        if why:
            bad.append({"clip": c, "why": why})
    for c in sorted(set(on_disk) - set(ids)):
        bad.append({"clip": c, "why": "token file outside the clip list"})
    present = [os.path.join(p_dir, c + ".npy") for c in sorted(set(ids) | set(on_disk)) if os.path.exists(os.path.join(p_dir, c + ".npy"))]
    shas = hash_many(present, workers)
    for c in sorted(set(ids) | set(on_disk)):
        p = os.path.join(p_dir, c + ".npy")
        if not os.path.exists(p):
            continue
        m = p_man.get(c)
        if m is None:
            bad.append({"clip": c, "why": "not in the manifest of this card"})
        elif os.path.getsize(p) != m.get("bytes"):
            bad.append({"clip": c, "why": "byte size differs from the manifest"})
        elif shas[p] != m.get("sha256"):
            bad.append({"clip": c, "why": "sha256 differs from the manifest"})
    return clause(not bad, n_expected=n_expected, n_files=len(on_disk), n_bad=len(bad), examples=bad[:3])


# ------------------------------------------------------------------------------------------------ clause iii
def pilot_arm_unchanged(tok_dir, workers=WORKERS):
    man = read_manifest(os.path.join(tok_dir, "MANIFEST.json"))
    bad, paths = [], []
    for c, m in sorted(man.items()):
        p = os.path.join(tok_dir, c + ".npy")
        if not os.path.exists(p):
            bad.append({"clip": c, "why": "file missing"})
        elif os.path.getsize(p) != m["bytes"]:
            bad.append({"clip": c, "why": "byte size differs"})
        else:
            paths.append(p)
    shas = hash_many(paths, workers)
    for c, m in sorted(man.items()):
        p = os.path.join(tok_dir, c + ".npy")
        if p in shas and shas[p] != m["sha256"]:
            bad.append({"clip": c, "why": "sha256 differs"})
    return {"pass": bool(not bad and len(man) > 0), "n_in_manifest": len(man), "n_bad": len(bad), "examples": bad[:3]}


def clause_pilot_unchanged(pilot_run, workers=WORKERS):
    arms = {a: pilot_arm_unchanged(os.path.join(pilot_run, "tok", a), workers) for a in ("D", "B")}
    return clause(all(v["pass"] for v in arms.values()), arms=arms)


# ------------------------------------------------------------------------------------------------ drivers
def run(a):
    t0 = time.time()
    lists = os.path.join(a.pilot_run, "lists")
    man_b = read_manifest(os.path.join(a.pilot_run, "tok", "B", "MANIFEST.json"))
    res = {"card": "ADD108", "control": "m1", "clauses": {}}
    ev = read_ids(a.eval) if os.path.exists(a.eval) else read_ids(os.path.join(lists, "eval141.txt"))
    if a.mode in ("identity", "all"):
        ids = read_ids(a.ids) if a.ids else ev[:3]
        res["clauses"]["i_identity_three_clips"] = clause_identity(ids, man_b, os.path.join(a.root, "check_tok", "tok", a.ident_arm))
    if a.mode in ("tokens", "all"):
        ids = read_ids(a.train_ids) + ev
        p_dir = os.path.join(a.root, "tok", "P")
        res["clauses"]["ii_arm_p_files"] = clause_arm_p(ids, p_dir, read_manifest(os.path.join(p_dir, "MANIFEST.json")), man_b, a.n_expected, a.workers)
        res["clauses"]["iii_pilot_arms_unchanged"] = clause_pilot_unchanged(a.pilot_run, a.workers)
    res["pass_m1"] = bool(all(v["pass"] for v in res["clauses"].values())) if a.mode == "all" else None
    res["mode"] = a.mode
    res["seconds"] = round(time.time() - t0, 1)
    out = a.out or os.path.join(a.root, {"identity": "ENC_M1_IDENTITY.json", "tokens": "ENC_M1_TOKENS.json", "all": "ENC_M1.json"}[a.mode])
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({k: (v["pass"]) for k, v in res["clauses"].items()}), "pass_m1 =", res["pass_m1"])
    if a.mode == "all":
        return 0 if res["pass_m1"] else 1
    return 0 if all(v["pass"] for v in res["clauses"].values()) else 1


# ------------------------------------------------------------------------------------------------ selftest
def _tok(seed, n):
    return np.random.RandomState(seed).randn(n, *TAIL).astype(np.float16)


def _put(d, c, arr):
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, c + ".npy")
    np.save(p, arr)
    return {"n_frames": int(arr.shape[0]), "bytes": os.path.getsize(p), "sha256": sha256_file(p)}


def selftest(base=None):
    """-> [(case, clause, ok)]: the clean world passes every clause, and each planted fault flips its own clause."""
    base = base or os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp_pt", "selftest_m1_%d" % int(time.time()))
    res = []
    ids = ["c%d" % i for i in range(5)]
    n = {c: 1 + (i % 2) for i, c in enumerate(ids)}

    def world(name):
        w = os.path.join(base, name)
        man_d, man_b, man_p = {}, {}, {}
        for i, c in enumerate(ids):
            man_d[c] = _put(os.path.join(w, "pilot", "tok", "D"), c, _tok(10 + i, n[c]))
            man_b[c] = _put(os.path.join(w, "pilot", "tok", "B"), c, _tok(20 + i, n[c]))
            man_p[c] = _put(os.path.join(w, "tok", "P"), c, _tok(30 + i, n[c]))
            _put(os.path.join(w, "check_tok", "tok", "B_ident"), c, _tok(20 + i, n[c]))        # the variant reproduces the pilot's arm B
        for arm, man in (("D", man_d), ("B", man_b)):
            json.dump({"clips": man}, open(os.path.join(w, "pilot", "tok", arm, "MANIFEST.json"), "w"))
        json.dump({"clips": man_p}, open(os.path.join(w, "tok", "P", "MANIFEST.json"), "w"))
        return w, man_b, man_p

    def evaluate(w, man_b):
        pm = read_manifest(os.path.join(w, "pilot", "tok", "B", "MANIFEST.json"))
        pp = read_manifest(os.path.join(w, "tok", "P", "MANIFEST.json"))
        return {"i": clause_identity(ids[:3], pm, os.path.join(w, "check_tok", "tok", "B_ident"))["pass"],
                "ii": clause_arm_p(ids, os.path.join(w, "tok", "P"), pp, pm, n_expected=5, workers=2)["pass"],
                "iii": clause_pilot_unchanged(os.path.join(w, "pilot"), 2)["pass"]}

    w, mb, mp = world("clean")
    r = evaluate(w, mb)
    for k in ("i", "ii", "iii"):
        res.append(("clean", k, r[k]))
    # i: one identity file differs by one value
    w, mb, mp = world("i_bad")
    a = _tok(20 + 1, n["c1"]).copy()
    a[0, 0, 0, 0] = np.float16(float(a[0, 0, 0, 0]) + 0.5)
    _put(os.path.join(w, "check_tok", "tok", "B_ident"), "c1", a)
    r = evaluate(w, mb)
    res.append(("i_one_value_differs", "i", not r["i"]))
    res.append(("i_one_value_differs", "ii_unaffected", r["ii"]))
    w, mb, mp = world("i_missing")
    os.remove(os.path.join(w, "check_tok", "tok", "B_ident", "c2.npy"))
    res.append(("i_file_missing", "i", not evaluate(w, mb)["i"]))
    # ii: wrong dtype, wrong shape, wrong frame count, stray file, missing file, hash not in the manifest, count
    w, mb, mp = world("ii_dtype")
    np.save(os.path.join(w, "tok", "P", "c0.npy"), _tok(30, n["c0"]).astype(np.float32))
    res.append(("ii_dtype_float32", "ii", not evaluate(w, mb)["ii"]))
    w, mb, mp = world("ii_shape")
    np.save(os.path.join(w, "tok", "P", "c0.npy"), np.zeros((n["c0"], 2, 64, 1024), np.float16))
    res.append(("ii_shape", "ii", not evaluate(w, mb)["ii"]))
    w, mb, mp = world("ii_frames")
    _put(os.path.join(w, "tok", "P"), "c3", _tok(33, n["c3"] + 1))
    res.append(("ii_frame_count_differs_from_pilot", "ii", not evaluate(w, mb)["ii"]))
    w, mb, mp = world("ii_stray")
    _put(os.path.join(w, "tok", "P"), "zz", _tok(99, 1))
    res.append(("ii_stray_file", "ii", not evaluate(w, mb)["ii"]))
    w, mb, mp = world("ii_missing")
    os.remove(os.path.join(w, "tok", "P", "c4.npy"))
    res.append(("ii_missing_file", "ii", not evaluate(w, mb)["ii"]))
    w, mb, mp = world("ii_hash")
    _put(os.path.join(w, "tok", "P"), "c2", _tok(77, n["c2"]))                       # a valid file the manifest does not know
    res.append(("ii_hash_not_in_manifest", "ii", not evaluate(w, mb)["ii"]))
    w, mb, mp = world("ii_count")
    pm = read_manifest(os.path.join(w, "pilot", "tok", "B", "MANIFEST.json"))
    pp = read_manifest(os.path.join(w, "tok", "P", "MANIFEST.json"))
    res.append(("ii_count_is_not_expected", "ii", not clause_arm_p(ids, os.path.join(w, "tok", "P"), pp, pm, n_expected=641, workers=2)["pass"]))
    # iii: a changed pilot file of arm D, a changed pilot file of arm B, a missing one
    w, mb, mp = world("iii_d")
    _put(os.path.join(w, "pilot", "tok", "D"), "c1", _tok(555, n["c1"]))
    res.append(("iii_arm_d_changed", "iii", not evaluate(w, mb)["iii"]))
    w, mb, mp = world("iii_b")
    _put(os.path.join(w, "pilot", "tok", "B"), "c2", _tok(556, n["c2"]))
    r = evaluate(w, mb)
    res.append(("iii_arm_b_changed", "iii", not r["iii"]))
    w, mb, mp = world("iii_missing")
    os.remove(os.path.join(w, "pilot", "tok", "D", "c0.npy"))
    res.append(("iii_arm_d_missing", "iii", not evaluate(w, mb)["iii"]))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", nargs="?", choices=["identity", "tokens", "all"])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--pilot-run", default=PILOT_RUN)
    ap.add_argument("--eval", default=PILOT_RUN + "/lists/eval141.txt")
    ap.add_argument("--train-ids", default=PILOT_RUN + "/lists/train_final_ids.txt")
    ap.add_argument("--ids", default=None, help="identity: the three clip ids (default: the first three of the evaluation list)")
    ap.add_argument("--ident-arm", default="B_ident", help="arm name the identity run wrote (folder check_tok/tok/<name>)")
    ap.add_argument("--n-expected", type=int, default=N_EXPECTED)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.selftest:
        res = selftest()
        bad = [r for r in res if not r[2]]
        for r in res:
            print(("ok   " if r[2] else "FAIL ") + r[0] + " / " + r[1])
        print("selftest: %d cases, %d failed" % (len(res), len(bad)))
        return 1 if bad else 0
    if not a.mode:
        ap.error("mode or --selftest is required")
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
