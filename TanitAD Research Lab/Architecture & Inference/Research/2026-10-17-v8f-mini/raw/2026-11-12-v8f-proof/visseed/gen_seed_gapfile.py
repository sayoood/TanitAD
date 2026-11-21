"""gen_seed_gapfile.py -- writes make_seed_gapfile.py from ../visrule/make_vis_gapfile.py (card ADD102, sha256 prefix asserted) by counted text edits and one inserted block (INSERT_GAPFILE.txt) (card ADD104, VISSEED10K).
Every old text must occur exactly the stated number of times. Refuses to overwrite (VS_OVERWRITE_OWN=1 rewrites it). Standard library only."""
import hashlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "visrule", "make_vis_gapfile.py"))
INS = os.path.join(HERE, "INSERT_GAPFILE.txt")
OUT = os.path.join(HERE, "make_seed_gapfile.py")
SRC_SHA_PREFIX = "97a90ec432bcfc5c"
raw = io.open(SRC, "rb").read()
src_sha = hashlib.sha256(raw).hexdigest()[:16]
if src_sha != SRC_SHA_PREFIX:
    sys.exit("REFUSED: the source is not the gap-file tool of card ADD102 (%s)" % src_sha)
if os.path.exists(OUT) and os.environ.get("VS_OVERWRITE_OWN") != "1":
    sys.exit("REFUSED: %s exists (VS_OVERWRITE_OWN=1 rewrites it)" % OUT)
text = raw.decode("utf-8")
insert = io.open(INS, "r", encoding="utf-8", newline="").read()
assert "\r" not in text and "\r" not in insert
N = 0


def once(a, b, n=1):
    global text, N
    assert text.count(a) == n, (a[:80], text.count(a), n)
    text = text.replace(a, b)
    N += 1


once('make_vis_gapfile.py -- the two gap files of card PREREG_V8F_PROOF_ADD102 (VISRULE10K). numpy and the standard library only (runs on the pod with /workspace/venv/bin/python).',
     'make_seed_gapfile.py -- the three gap files of the head on five frames (D-H5) of the fit seeds 2, 3 and 4 for card PREREG_V8F_PROOF_ADD104 (VISSEED10K), with controls x0 (the fits) and x1 (the gap files).\n'
     'Derived from make_vis_gapfile.py (card ADD102) by counted edits (gen_seed_gapfile.py): seeds, file names, record name, the fit-record check x0 and the dump-meta clause of x1 are the changes; the build is the same. '
     'numpy and the standard library only (runs on the pod with /workspace/venv/bin/python).')
once('Usage: python make_vis_gapfile.py --eval-join NPZ --dump-root DIR --out-dir NEWDIR [--expect-clips 141]',
     'Usage: python make_seed_gapfile.py --eval-join NPZ --fit-root DIR --out-dir NEWDIR [--add99-fits DIR] [--eval-list FILE] [--expect-clips 141]')
once('reads DIR/D_H5/reader_s<k>_score.npz (the fixed-window head H5 of card ADD99, fit seed k) for k in 0 1, writes NEWDIR/VGAP_s<k>.npz and NEWDIR/VISRULE_GAPFILES_RECORD.json (control v0 and the build check)',
     'reads FITROOT/dump/D_H5/reader_s<S>_score.npz and reader_s<S>_meta.json (the fixed-window head H5 of card ADD99, fit seed S) and FITROOT/fits/D_H5_s<S>/FIT_RECORD.json for S in 2 3 4, and the records of fit seeds 0 and 1\n'
     '         in --add99-fits (default /workspace/v8proof/cropread/run1/fits); writes NEWDIR/VGAPF_s<S>.npz and NEWDIR/VISSEED_GAPFILES_RECORD.json with x0 (the fits) and x1 = v0 (the dumps are on the join frames) + the build check\n'
     '         (saved gap equals a second routine) + the meta clause (each dump meta names the sha256 of the head file of its fit)')
once('python make_vis_gapfile.py --selftest', 'python make_seed_gapfile.py --selftest')
once('CARD = "PREREG_V8F_PROOF_ADD102"\n', 'CARD = "PREREG_V8F_PROOF_ADD104"\n')
once('SEEDS = (0, 1)\n', 'SEEDS = (2, 3, 4)                           # fit seeds (the dump of fit seed S is reader_s<S>_score.npz, made on the rows of the present reader\'s dump of seed 0)\n')
once('"VGAP_s%d"', '"VGAPF_s%d"', 3)
once('"VGAP_s0.npz"', '"VGAPF_s2.npz"')
once('"VISRULE_GAPFILES_RECORD.json"', '"VISSEED_GAPFILES_RECORD.json"')
once('a["model_seed"]', 'a["fit_seed"]')
once('f["model_seed"] = s', 'f["fit_seed"] = s')
once('def load_npz(p):\n', insert + 'def load_npz(p):\n')

# ---- build: the facts of the fits and the dump metas go into the record; x0 and x1
once('def build(J, dumps, out_dir, faults=(), meta=None, expect_clips=None, eval_join_sha=None):', 'def build(J, dumps, out_dir, faults=(), meta=None, expect_clips=None, eval_join_sha=None, fits=None):')
once('rec = {"card": CARD, "eval_join_sha256": eval_join_sha, "files": {}, "dumps": {}}', 'rec = {"card": CARD, "eval_join_sha256": eval_join_sha, "files": {}, "dumps": {}, "fits": fits}')
once('rec["dumps"][nm] = {"dump_file": dp, "dump_dir_name": dn, "e0":', 'rec["dumps"][nm] = {"dump_file": dp, "dump_dir_name": dn, "dump_meta": m.get("dump_meta"), "e0":')
once('    rec["v0"] = v0_check(rec)\n    rec["build"] = build_check(rec)\n',
     '    rec["v0"] = v0_check(rec)\n    rec["build"] = build_check(rec)\n    rec["meta"] = meta_check(rec)\n'
     '    rec["x0"] = x0_check(rec.get("fits"))\n'
     '    rec["x1"] = {"PASS": bool(rec["v0"]["PASS"] and rec["build"]["PASS"] and rec["meta"]["PASS"]), "parts": ["v0", "build", "meta"]}\n')

# ---- the v0 faults act on the first fit seed (2)
once('d[0]', 'd[2]', 17)
once('dumps[0]', 'dumps[2]')
once('d[2].update(seed=np.int64(1))', 'd[2].update(seed=np.int64(3))')

# ---- selftest
once('"tmp_selftest_gap_%d"', '"tmp_selftest_seedgap_%d"')
once('    meta = synthetic_meta(dumps, os.path.join(base, "dumps"))\n'
     '    rec = build(J, dumps, os.path.join(base, "clean"), meta=meta)\n'
     '    print("clean build: v0 %s build %s" % (rec["v0"]["PASS"], rec["build"]["PASS"]))\n'
     '    ok &= rec["v0"]["PASS"] and rec["build"]["PASS"]\n',
     '    fits, meta, _, _ = synthetic_fit_world(os.path.join(base, "world"), J, dumps)\n'
     '    rec = build(J, dumps, os.path.join(base, "clean"), meta=meta, fits=fits)\n'
     '    print("clean build: v0 %s build %s meta %s x0 %s x1 %s" % (rec["v0"]["PASS"], rec["build"]["PASS"], rec["meta"]["PASS"], rec["x0"]["PASS"], rec["x1"]["PASS"]))\n'
     '    ok &= rec["v0"]["PASS"] and rec["build"]["PASS"] and rec["meta"]["PASS"] and rec["x0"]["PASS"] and rec["x1"]["PASS"]\n'
     '    ok &= sorted(rec["files"]) == ["VGAPF_s2", "VGAPF_s3", "VGAPF_s4"]\n')
once('    print("ZZ MAKE-VIS-GAPFILE-SELFTEST " + ("PASS" if ok else "FAIL"))',
     '    xf = x_faults(base, J, dumps)\n'
     '    print("x0 and meta planted faults:", xf)\n'
     '    ok &= all(xf.values())\n'
     '    print("ZZ MAKE-SEED-GAPFILE-SELFTEST " + ("PASS" if ok else "FAIL"))')

# ---- main
once('    ap.add_argument("--dump-root")\n', '    ap.add_argument("--fit-root")\n    ap.add_argument("--add99-fits", default=ADD99_FITS)\n    ap.add_argument("--eval-list")\n')
once('    if not (a.eval_join and a.dump_root and a.out_dir):\n        print("REFUSED: --eval-join, --dump-root and --out-dir are required")',
     '    if not (a.eval_join and a.fit_root and a.out_dir):\n        print("REFUSED: --eval-join, --fit-root and --out-dir are required")')
once('    dumps, meta = {}, {}\n'
     '    for s in SEEDS:\n'
     '        p = os.path.join(a.dump_root, DUMP_DIR, "reader_s%d_score.npz" % s)\n'
     '        if not os.path.isfile(p):\n'
     '            print("REFUSED: no dump %s" % p)\n'
     '            return 2\n'
     '        dumps[s] = load_npz(p)\n'
     '        meta[s] = {"dump_file": p, "dump_dir_name": DUMP_DIR}\n'
     '    rec = build(J, dumps, a.out_dir, meta=meta, expect_clips=a.expect_clips, eval_join_sha=sha_file(a.eval_join))\n'
     '    print("ZZ VIS-GAPFILES v0 %s build %s" % ("PASS" if rec["v0"]["PASS"] else "FAIL", "PASS" if rec["build"]["PASS"] else "FAIL"))\n'
     '    for k in ("v0", "build"):\n'
     '        for x in rec[k]["failures"][:6]:\n'
     '            print(k, x)\n'
     '    return 0 if rec["v0"]["PASS"] and rec["build"]["PASS"] else 1\n',
     '    dump_root = os.path.join(a.fit_root, "dump")\n'
     '    fits, meta = collect(a.fit_root, a.add99_fits, a.eval_list or os.path.join(a.fit_root, "lists", "eval141.txt"), dump_root)\n'
     '    dumps = {}\n'
     '    for s in SEEDS:\n'
     '        p = meta[s]["dump_file"]\n'
     '        if not os.path.isfile(p):\n'
     '            print("REFUSED: no dump %s" % p)\n'
     '            return 2\n'
     '        dumps[s] = load_npz(p)\n'
     '    rec = build(J, dumps, a.out_dir, meta=meta, expect_clips=a.expect_clips, eval_join_sha=sha_file(a.eval_join), fits=fits)\n'
     '    P = lambda k: "PASS" if rec[k]["PASS"] else "FAIL"                                                    # noqa: E731\n'
     '    print("ZZ SEED-GAPFILES x0 %s x1 %s (v0 %s build %s meta %s)" % (P("x0"), P("x1"), P("v0"), P("build"), P("meta")))\n'
     '    for k in ("x0", "v0", "build", "meta"):\n'
     '        for x in rec[k]["failures"][:6]:\n'
     '            print(k, x)\n'
     '    return 0 if rec["x0"]["PASS"] and rec["x1"]["PASS"] else 1\n')
assert "\r" not in text
for w in ("vo" + "id", "super" + "seded", "retr" + "acted", "for" + "merly", "corr" + "ection", "with" + "drawn", "no " + "longer", "closed " + "loop"):
    assert w not in text, w
io.open(OUT, "w", encoding="utf-8", newline="\n").write(text)
print("wrote %s sha256 %s edits %d (source %s)" % (os.path.basename(OUT), hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], N, src_sha))
