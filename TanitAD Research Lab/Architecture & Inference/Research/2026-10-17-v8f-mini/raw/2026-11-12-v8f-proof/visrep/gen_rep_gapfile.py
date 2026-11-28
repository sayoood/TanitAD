"""gen_rep_gapfile.py -- writes make_rep_gapfile.py from ../visrule/make_vis_gapfile.py (card ADD102, sha256 prefix asserted) by counted text edits (card ADD103, VISREP10K).
Every old text must occur exactly the stated number of times. Refuses to overwrite (VR_OVERWRITE_OWN=1 rewrites it). Standard library only."""
import hashlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "visrule", "make_vis_gapfile.py"))
OUT = os.path.join(HERE, "make_rep_gapfile.py")
SRC_SHA_PREFIX = "97a90ec432bcfc5c"
raw = io.open(SRC, "rb").read()
src_sha = hashlib.sha256(raw).hexdigest()[:16]
if src_sha != SRC_SHA_PREFIX:
    sys.exit("REFUSED: the source is not the gap-file tool of card ADD102 (%s)" % src_sha)
if os.path.exists(OUT) and os.environ.get("VR_OVERWRITE_OWN") != "1":
    sys.exit("REFUSED: %s exists (VR_OVERWRITE_OWN=1 rewrites it)" % OUT)
text = raw.decode("utf-8")
assert "\r" not in text
N = 0


def once(a, b, n=1):
    global text, N
    assert text.count(a) == n, (a[:80], text.count(a), n)
    text = text.replace(a, b)
    N += 1


once('make_vis_gapfile.py -- the two gap files of card PREREG_V8F_PROOF_ADD102 (VISRULE10K). numpy and the standard library only (runs on the pod with /workspace/venv/bin/python).',
     'make_rep_gapfile.py -- the two gap files of the head on one frame (D-H1) for card PREREG_V8F_PROOF_ADD103 (VISREP10K), and the check that the two gap files of card ADD102 have the hashes of that card\'s record.\n'
     'Derived from make_vis_gapfile.py (card ADD102): the dump folder, the file names, the record name and the add102 check are the only changes of the build. numpy and the standard library only (runs on the pod with /workspace/venv/bin/python).')
once('Usage: python make_vis_gapfile.py --eval-join NPZ --dump-root DIR --out-dir NEWDIR [--expect-clips 141]',
     'Usage: python make_rep_gapfile.py --eval-join NPZ --dump-root DIR --out-dir NEWDIR --add102-gap-dir DIR [--expect-clips 141]')
once('reads DIR/D_H5/reader_s<k>_score.npz (the fixed-window head H5 of card ADD99, fit seed k) for k in 0 1, writes NEWDIR/VGAP_s<k>.npz and NEWDIR/VISRULE_GAPFILES_RECORD.json (control v0 and the build check)',
     'reads DIR/D_H1/reader_s<k>_score.npz (the head on one frame, fixed windows, of card ADD99, fit seed k) for k in 0 1, writes NEWDIR/VGAP1_s<k>.npz and NEWDIR/VISREP_GAPFILES_RECORD.json (control v0, the build check,\n'
     '         the add102 check of the two gap files of card ADD102 in --add102-gap-dir against the hashes of that card\'s VISRULE_GAPFILES_RECORD.json, and w0 = all three)')
once('       python make_vis_gapfile.py --selftest', '       python make_rep_gapfile.py --selftest')
once('CARD = "PREREG_V8F_PROOF_ADD102"\n', 'CARD = "PREREG_V8F_PROOF_ADD103"\nCARD_102 = "PREREG_V8F_PROOF_ADD102"\n')
once('"VISRULE_GAPFILES_RECORD.json"', '"VISREP_GAPFILES_RECORD.json"')
once('DUMP_DIR = "D_H5"\n', 'DUMP_DIR = "D_H1"                            # the head on one frame (cropread/fit/crop_dump.py:34, :45): the folder is the only change of the build\n'
     'ADD102_RECORD = "VISRULE_GAPFILES_RECORD.json"\nADD102_NAMES = ("VGAP_s0", "VGAP_s1")\n')
once('FIRST_OUTPUT_FRAME = 4                      # the head H5 reads five frames (cropread/fit/crop_dump.py:29-35): no output at frames 0 to 3',
     'FIRST_OUTPUT_FRAME = 4                      # synthetic dumps only: the real head H1 has an output on every frame (crop_dump.py:34); NaN rows are kept as dumped either way')
once('"VGAP_s%d"', '"VGAP1_s%d"', 3)
once('"VGAP_s0.npz"', '"VGAP1_s0.npz"')

# ---- the add102 check
once('def load_npz(p):\n',
     'def add102_check(add102_dir):\n'
     '    """w0, second sentence of the card: the two gap files of card ADD102 (VGAP_s0.npz, VGAP_s1.npz in add102_dir) have the sha256 of that card\'s record (VISRULE_GAPFILES_RECORD.json in the same folder)."""\n'
     '    p = os.path.join(add102_dir or "", ADD102_RECORD)\n'
     '    if not add102_dir or not os.path.isfile(p):\n'
     '        return {"PASS": False, "failures": ["no record %s" % p], "hashes": {}}\n'
     '    bad, hashes = [], {}\n'
     '    try:\n'
     '        with open(p, encoding="utf-8") as fh:\n'
     '            r = json.load(fh)\n'
     '        if r.get("card") != CARD_102:\n'
     '            bad.append("the record is for card %s" % r.get("card"))\n'
     '        for nm in ADD102_NAMES:\n'
     '            want = (r.get("files", {}).get(nm) or {}).get("file_sha256")\n'
     '            fp = os.path.join(add102_dir, nm + ".npz")\n'
     '            got = sha_file(fp) if os.path.isfile(fp) else None\n'
     '            hashes[nm] = {"record": want, "disk": got}\n'
     '            if not (isinstance(want, str) and len(want) == 64):\n'
     '                bad.append("%s: the record has no sha256" % nm)\n'
     '            elif got != want:\n'
     '                bad.append("%s: file %s, the record says %s" % (nm, "missing" if got is None else got[:16], want[:16]))\n'
     '        info = {"record_v0_pass": (r.get("v0") or {}).get("PASS"), "record_build_pass": (r.get("build") or {}).get("PASS"), "record_sha256": sha_file(p)}\n'
     '    except (ValueError, TypeError, AttributeError, OSError) as ex:\n'
     '        bad.append("record unreadable: %s: %s" % (type(ex).__name__, ex))\n'
     '        info = {}\n'
     '    return dict({"PASS": not bad, "failures": bad, "hashes": hashes}, **info)\n\n\n'
     'def load_npz(p):\n')
once('def build(J, dumps, out_dir, faults=(), meta=None, expect_clips=None, eval_join_sha=None):',
     'def build(J, dumps, out_dir, faults=(), meta=None, expect_clips=None, eval_join_sha=None, add102_dir=None):')
once('    rec["v0"] = v0_check(rec)\n    rec["build"] = build_check(rec)\n',
     '    rec["v0"] = v0_check(rec)\n    rec["build"] = build_check(rec)\n    rec["add102"] = add102_check(add102_dir)\n'
     '    rec["w0"] = {"PASS": bool(rec["v0"]["PASS"] and rec["build"]["PASS"] and rec["add102"]["PASS"]), "parts": ["v0", "build", "add102"]}\n')

# ---- synthetic data
once('def synthetic_dump(J, seed, rng_seed=7, out_range=True):', 'def synthetic_dump(J, seed, rng_seed=7, out_range=True, first_out=FIRST_OUTPUT_FRAME):')
once('hg = np.nan if f < FIRST_OUTPUT_FRAME else', 'hg = np.nan if f < first_out else')
once('"""a dump in the format of FIT_DEFS section 2: every frame of every clip once; the head H5 has no output at frames 0 to 3."""',
     '"""a dump in the format of FIT_DEFS section 2: every frame of every clip once; no output (NaN) at frames below first_out (4 by default; 0 is the real head on one frame)."""')
once('def v0_faults(J, dumps):\n',
     'def synthetic_add102(root):\n'
     '    """a folder like the gap-file folder of card ADD102: two small files and a record with their hashes"""\n'
     '    os.makedirs(root)\n'
     '    rec = {"card": CARD_102, "files": {}}\n'
     '    for nm in ADD102_NAMES:\n'
     '        fp = os.path.join(root, nm + ".npz")\n'
     '        np.savez(fp, gap=np.arange(5, dtype=np.float64) + len(nm))\n'
     '        rec["files"][nm] = {"file_sha256": sha_file(fp)}\n'
     '    with open(os.path.join(root, ADD102_RECORD), "w", encoding="utf-8", newline="\\n") as fh:\n'
     '        json.dump(rec, fh)\n'
     '    return root\n\n\n'
     'def add102_faults(base, src):\n'
     '    """each planted fault of the ADD102 folder turns the add102 check red -> dict name -> bool"""\n'
     '    import shutil\n'
     '    out = {}\n\n'
     '    def variant(name, mut):\n'
     '        d = os.path.join(base, "a102_" + name)\n'
     '        shutil.copytree(src, d)\n'
     '        mut(d)\n'
     '        return not add102_check(d)["PASS"]\n\n'
     '    def edit_rec(d, f):\n'
     '        p = os.path.join(d, ADD102_RECORD)\n'
     '        with open(p, encoding="utf-8") as fh:\n'
     '            r = json.load(fh)\n'
     '        f(r)\n'
     '        with open(p, "w", encoding="utf-8", newline="\\n") as fh:\n'
     '            json.dump(r, fh)\n'
     '    out["file_changed"] = variant("changed", lambda d: np.savez(os.path.join(d, "VGAP_s1.npz"), gap=np.zeros(5)))\n'
     '    out["file_missing"] = variant("missing", lambda d: os.rename(os.path.join(d, "VGAP_s0.npz"), os.path.join(d, "other.npz")))\n'
     '    out["record_hash_changed"] = variant("rechash", lambda d: edit_rec(d, lambda r: r["files"]["VGAP_s0"].update(file_sha256="0" * 64)))\n'
     '    out["record_wrong_card"] = variant("card", lambda d: edit_rec(d, lambda r: r.update(card="PREREG_V8F_PROOF_ADD101")))\n'
     '    out["record_missing"] = variant("norec", lambda d: os.rename(os.path.join(d, ADD102_RECORD), os.path.join(d, "other.json")))\n'
     '    return out\n\n\n'
     'def v0_faults(J, dumps):\n')

# ---- selftest
once('"tmp_selftest_gap_%d"', '"tmp_selftest_repgap_%d"')
once('    rec = build(J, dumps, os.path.join(base, "clean"), meta=meta)\n    print("clean build: v0 %s build %s" % (rec["v0"]["PASS"], rec["build"]["PASS"]))\n    ok &= rec["v0"]["PASS"] and rec["build"]["PASS"]\n',
     '    a102 = synthetic_add102(os.path.join(base, "a102"))\n'
     '    rec = build(J, dumps, os.path.join(base, "clean"), meta=meta, add102_dir=a102)\n'
     '    print("clean build: v0 %s build %s add102 %s w0 %s" % (rec["v0"]["PASS"], rec["build"]["PASS"], rec["add102"]["PASS"], rec["w0"]["PASS"]))\n'
     '    ok &= rec["v0"]["PASS"] and rec["build"]["PASS"] and rec["add102"]["PASS"] and rec["w0"]["PASS"]\n'
     '    rec0 = build(J, {s: synthetic_dump(J, s, first_out=0) for s in SEEDS}, os.path.join(base, "clean_h1"), meta=meta, add102_dir=a102)\n'
     '    print("clean build with an output on every frame (as the real head): w0 %s" % rec0["w0"]["PASS"])\n'
     '    ok &= rec0["w0"]["PASS"]\n'
     '    af = add102_faults(base, a102)\n'
     '    print("add102 planted faults turned red:", af)\n'
     '    ok &= all(af.values())\n'
     '    recn = build(J, dumps, os.path.join(base, "no_add102"), meta=meta)\n'
     '    ok &= recn["w0"]["PASS"] is False\n')
once('print("ZZ MAKE-VIS-GAPFILE-SELFTEST " + ("PASS" if ok else "FAIL"))', 'print("ZZ MAKE-REP-GAPFILE-SELFTEST " + ("PASS" if ok else "FAIL"))')

# ---- main
once('    ap.add_argument("--expect-clips", type=int, default=141)\n', '    ap.add_argument("--expect-clips", type=int, default=141)\n    ap.add_argument("--add102-gap-dir")\n')
once('    if not (a.eval_join and a.dump_root and a.out_dir):\n        print("REFUSED: --eval-join, --dump-root and --out-dir are required")',
     '    if not (a.eval_join and a.dump_root and a.out_dir and a.add102_gap_dir):\n        print("REFUSED: --eval-join, --dump-root, --out-dir and --add102-gap-dir are required")')
once('eval_join_sha=sha_file(a.eval_join))\n    print("ZZ VIS-GAPFILES v0 %s build %s" % ("PASS" if rec["v0"]["PASS"] else "FAIL", "PASS" if rec["build"]["PASS"] else "FAIL"))\n    for k in ("v0", "build"):\n        for x in rec[k]["failures"][:6]:\n            print(k, x)\n    return 0 if rec["v0"]["PASS"] and rec["build"]["PASS"] else 1\n',
     'eval_join_sha=sha_file(a.eval_join), add102_dir=a.add102_gap_dir)\n'
     '    P = lambda k: "PASS" if rec[k]["PASS"] else "FAIL"                                                    # noqa: E731\n'
     '    print("ZZ REP-GAPFILES v0 %s build %s add102 %s w0 %s" % (P("v0"), P("build"), P("add102"), P("w0")))\n'
     '    for k in ("v0", "build", "add102"):\n        for x in rec[k]["failures"][:6]:\n            print(k, x)\n    return 0 if rec["w0"]["PASS"] else 1\n')
assert "\r" not in text
for w in ("vo" + "id", "super" + "seded", "retr" + "acted", "for" + "merly", "corr" + "ection", "with" + "drawn", "no " + "longer", "closed " + "loop"):
    assert w not in text, w
io.open(OUT, "w", encoding="utf-8", newline="\n").write(text)
print("wrote %s sha256 %s edits %d (source %s)" % (os.path.basename(OUT), hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], N, src_sha))
