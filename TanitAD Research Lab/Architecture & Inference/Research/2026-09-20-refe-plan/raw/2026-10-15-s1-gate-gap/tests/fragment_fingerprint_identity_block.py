"""FRAGMENT, not runnable on its own: the identity block exactly as it sits in refe/test_launch_gate.py after PATCH_test_launch_gate.py.diff (functions _old_*, fp_fixtures, fp_violations,
fingerprint_identity_static).  Run it where it lives:  python refe/test_launch_gate.py --fp-identity-only   (seconds; prints ZZLG_FP_IDENTITY_PASS <n>)."""
# ====================================================================================================== the FASTER FINGERPRINT is the SAME fingerprint (2026-10-15)
# gate_token.fingerprint / sha256_file / data_manifest were made faster for the GPU-idle gap before every S1 arm (raw/2026-10-15-s1-gate-gap/REPORT.md): the 413,184-file image listing is
# stat'ed on worker threads, the independent inputs are fingerprinted concurrently, and the checkpoint is hashed through a reusable buffer.  A token binds these digests, so the change is only
# admissible if EVERY digest is bit-identical to the old serial one.  The `_old_*` functions below are the pre-change implementations, VERBATIM (with their constants frozen here too, so a
# change to gate_token's constants cannot move the oracle with it), kept as the oracle.  The comparison runs over fixtures that exercise every branch (nesting, empty dirs, skip names, odd
# names, a directory spanning many stat tasks, symlinks where this box has them, a missing path, a flag named twice, path_map, gate inputs) at several thread counts, and every
# digest-altering mutation of the new code must make it RED.
_OLD_DIR_SKIP_NAMES = frozenset({"_v2manifest.pt", "_v2manifest.pt.tmp", "__pycache__"})
_OLD_SKIP_DIRS = frozenset({"__pycache__", ".pytest_cache", ".mypy_cache"})
_OLD_TREE_SUFFIX = (".py", ".sh", ".yaml", ".yml", ".json", ".txt", ".patch")
_OLD_TREE_ROOTS = ("refe", "code", "splits")


def _old_sha256_file(p, chunk: int = 1 << 22) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(chunk), b""):
            h.update(c)
    return h.hexdigest()


def _old_fingerprint(path) -> dict:
    import hashlib
    p = Path(path)
    if p.is_file():
        return {"kind": "file", "bytes": p.stat().st_size, "sha256": _old_sha256_file(p), "method": "sha256-v1"}
    if p.is_dir():
        lines, n, total = [], 0, 0
        for dirpath, dirnames, filenames in os.walk(p):
            dirnames[:] = sorted(d for d in dirnames if d not in _OLD_DIR_SKIP_NAMES)
            for fn in filenames:
                if fn in _OLD_DIR_SKIP_NAMES:
                    continue
                f = Path(dirpath) / fn
                st = f.stat()
                lines.append(f"{f.relative_to(p).as_posix()}\t{st.st_size}\n")
                n += 1
                total += st.st_size
        lines.sort()
        return {"kind": "dir", "n_files": n, "bytes": total, "listing_sha256": hashlib.sha256("".join(lines).encode("utf-8")).hexdigest(),
                "method": "dir-listing-v1 (relpath + size of every file; contents NOT hashed; _v2manifest.pt excluded -- a cache "
                          "rewrites it on first use)"}
    return {"kind": "missing"}


def _old_data_manifest(argv, data_flags, gate_inputs=None, path_map=None) -> dict:
    man: dict = {}
    want = set(data_flags)
    for flag, vals in GT.flag_pairs(argv):
        if flag in want and vals:
            local = vals[0]
            for a, b in (path_map or []):
                if local.startswith(a):
                    local = b + local[len(a):]
            man[flag] = {"path": vals[0], **({"local": local} if local != vals[0] else {}), **_old_fingerprint(local)}
    for key, path in sorted((gate_inputs or {}).items()):
        man[key] = {"path": str(path), **_old_fingerprint(path)}
    return man


def _old_tree_manifest(pkg) -> dict:
    pkg = Path(pkg)
    out: dict = {}
    for root in _OLD_TREE_ROOTS:
        base = pkg / root
        if not base.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d not in _OLD_SKIP_DIRS)
            for fn in sorted(filenames):
                if fn.endswith(_OLD_TREE_SUFFIX):
                    f = Path(dirpath) / fn
                    out[f.relative_to(pkg).as_posix()] = _old_sha256_file(f)
    return out


def fp_fixtures(td: Path) -> dict:
    """build the directories / files the identity oracle runs over under `td`; returns {name: path} plus the flags `symlinks` / `posix_names`"""
    fx: dict = {}

    def put(rel, data=b"x"):
        p = td / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return p
    for i, n in enumerate((0, 1, 7, 4095, 4096, 4097, 100000)):
        put(f"flat/f{i}.bin", bytes(range(256)) * (n // 256) + bytes(range(n % 256)))
    for rel in ("a/1.txt", "a/b/2.txt", "a/b/c/3.txt", "a/b/c/d/4.txt", "a/z.txt", "m/n/o/5.txt", "m/6.txt", "top.txt", "same/x.txt", "other/same/x.txt", "a/same/x.txt"):
        put("nested/" + rel, rel.encode() * 3)
    for i in range(41):                                                  # many stat tasks for ONE directory once the test shrinks the chunk
        put(f"big/f{i:03d}.dat", b"y" * i)
    for rel in ("_v2manifest.pt", "_v2manifest.pt.tmp", "keep.txt", "__pycache__/x.pyc", "sub/keep2.txt", "sub/__pycache__/y.pyc", "sub/_v2manifest.pt", "sub/deeper/__pycache__/z.pyc"):
        put("skips/" + rel, rel.encode())
    for rel in ("with space.txt", "ünï.txt", "a#b.txt", "x'y.txt", "dir with space/é.txt", "100%.txt", "a,b.txt", "UPPER.TXT", "lower.txt"):
        put("names/" + rel, rel.encode("utf-8"))
    fx["posix_names"] = False
    if os.name != "nt":
        for rel in ("tab\tname.txt", "back\\slash.txt", "new\nline.txt", "-dash.txt"):
            try:
                put("names/" + rel, b"p")
                fx["posix_names"] = True
            except OSError:
                pass
    (td / "empty").mkdir()
    (td / "emptynest" / "a" / "b").mkdir(parents=True)
    put("emptynest/a/only.txt", b"o")
    fx["symlinks"] = False
    try:
        (td / "links").mkdir()
        put("links/real.txt", b"real-file")
        put("links/realdir/inner.txt", b"inner")
        os.symlink(str(td / "links" / "real.txt"), str(td / "links" / "to_file"))
        os.symlink(str(td / "links" / "realdir"), str(td / "links" / "to_dir"), target_is_directory=True)
        fx["symlinks"] = True
    except (OSError, NotImplementedError):
        (td / "links").mkdir(exist_ok=True)
    for k in ("flat", "nested", "big", "skips", "names", "empty", "emptynest", "links"):
        fx[k] = td / k
    fx["whole"] = td
    return fx


def fp_violations(fp_fn, sha_fn, dm_fn, tm_fn, fx: dict, pkg: Path, threads=("1", "2", "16"), thorough: bool = True) -> list:
    """every way the candidate functions differ from the frozen pre-change ones, over the fixtures `fx`.  [] = bit-identical.  `thorough` = every path form (str / Path / trailing
    separator / relative) and every thread count; the mutation runs use the cheap form."""
    v: list = []
    td = fx["whole"]
    saved_env, saved_chunk = os.environ.get(GT.FP_THREADS_ENV), GT.FP_STAT_CHUNK
    GT.FP_STAT_CHUNK = 4
    try:
        for nthr in threads:
            os.environ[GT.FP_THREADS_ENV] = nthr
            for name in ("flat", "nested", "big", "skips", "names", "empty", "emptynest", "links", "whole"):
                p = fx[name]
                for form in ([str(p)] if not thorough else [str(p), p, str(p) + os.sep]):
                    if _old_fingerprint(form) != fp_fn(form):
                        v.append(f"fingerprint[{name}] threads={nthr} form={type(form).__name__}")
            if _old_fingerprint(td / "absent") != fp_fn(td / "absent"):
                v.append("fingerprint[missing]")
            for fl in ("flat/f0.bin", "flat/f6.bin", "nested/top.txt"):
                if _old_fingerprint(td / fl) != fp_fn(td / fl):
                    v.append(f"fingerprint[file {fl}]")
            if thorough:
                cwd = os.getcwd()
                os.chdir(str(td))
                try:
                    for rel in ("nested", "./nested", "nested/a"):
                        if _old_fingerprint(rel) != fp_fn(rel):
                            v.append(f"fingerprint[relative {rel}] threads={nthr}")
                finally:
                    os.chdir(cwd)
            argv = ["--targets", str(fx["flat"]), "--images", str(fx["nested"]), "--calib", str(td / "flat" / "f4.bin"), "--targets", str(fx["skips"]), "--gone", str(td / "absent"),
                    "--onpolicy", str(fx["big"]), "--noval", "--names", str(fx["names"]), "--ck", str(td / "flat" / "f6.bin"), "--lane", str(fx["links"])]
            flags = ("--targets", "--images", "--calib", "--gone", "--onpolicy", "--noval", "--names", "--ck", "--lane")
            gi = {"gate:b": str(td / "flat" / "f5.bin"), "gate:a": str(fx["empty"]), "gate:c": str(td / "absent")}
            cases = [(None, argv)]
            if thorough:                                                 # path_map: the argv names a path the box does not have, mapped onto one it does
                cases.append(([(str(td) + os.sep + "nested", str(fx["flat"]))], ["--targets", str(td) + os.sep + "nested"] + argv[2:]))
            for pmap, ag in cases:
                o, n = _old_data_manifest(ag, flags, gi, pmap), dm_fn(ag, flags, gi, pmap)
                if o != n or list(o) != list(n):
                    v.append(f"data_manifest threads={nthr} path_map={bool(pmap)}")
            if _old_data_manifest([], flags) != dm_fn([], flags) or _old_data_manifest(argv, ()) != dm_fn(argv, ()):
                v.append(f"data_manifest[empty] threads={nthr}")
            if tm_fn(pkg) != _old_tree_manifest(pkg):
                v.append(f"tree_manifest threads={nthr}")
        for size in (0, 1, 4095, 4096, 4097, 3 * 4096 + 5, (1 << 22) + 3):
            f = td / f"hash_{size}.bin"
            f.write_bytes((bytes(range(251)) * (size // 251 + 1))[:size])
            want = _old_sha256_file(f)
            for chunk in ((4096, 1 << 22) if size <= 3 * 4096 + 5 else (1 << 22,)):
                if sha_fn(f, chunk) != want or _old_sha256_file(f, chunk) != want:
                    v.append(f"sha256_file size={size} chunk={chunk}")
            if sha_fn(f) != want:
                v.append(f"sha256_file size={size} default chunk")
            f.unlink()
        if fx.get("symlinks"):                                           # a dangling symlink: both raise, the same error class
            dang = td / "links" / "dangling"
            os.symlink(str(td / "links" / "nowhere"), str(dang))
            try:
                errs = []
                for fn in (_old_fingerprint, fp_fn):
                    try:
                        fn(td / "links")
                        errs.append(None)
                    except OSError as e:
                        errs.append(type(e).__name__)
                if errs[0] != errs[1]:
                    v.append(f"dangling symlink: old {errs[0]} vs new {errs[1]}")
            finally:
                dang.unlink()
    finally:
        GT.FP_STAT_CHUNK = saved_chunk
        if saved_env is None:
            os.environ.pop(GT.FP_THREADS_ENV, None)
        else:
            os.environ[GT.FP_THREADS_ENV] = saved_env
    return v


def fingerprint_identity_static() -> None:
    import contextlib
    import hashlib
    with tempfile.TemporaryDirectory() as td_:
        td = Path(td_)
        fx = fp_fixtures(td / "fx")
        pkg = td / "pkg"
        for rel, txt in (("refe/a.py", "print(1)\n"), ("refe/sub/b.sh", "echo\n"), ("refe/__pycache__/c.pyc", "x"), ("refe/notes.md", "n"), ("code/c.json", "{}"), ("code/d.patch", "p"),
                         ("splits/e.yaml", "x: 1\n"), ("splits/f.yml", "y: 2\n"), ("refe/.pytest_cache/g.txt", "g"), ("other/h.py", "h"), ("refe/big.txt", "z" * 300000)):
            (pkg / rel).parent.mkdir(parents=True, exist_ok=True)
            (pkg / rel).write_text(txt, encoding="utf-8", newline="\n")
        # the oracle is itself stable and sees what the fixtures were built to show
        ok(_old_fingerprint(fx["nested"]) == _old_fingerprint(fx["nested"]) and _old_fingerprint(fx["nested"])["n_files"] == 11)
        ok(_old_fingerprint(fx["skips"])["n_files"] == 2 and _old_fingerprint(fx["empty"])["n_files"] == 0 and _old_fingerprint(fx["empty"])["kind"] == "dir")
        ok(_old_tree_manifest(pkg).keys() == {"refe/a.py", "refe/sub/b.sh", "code/c.json", "code/d.patch", "splits/e.yaml", "splits/f.yml", "refe/big.txt"})
        # 1. the new code == the frozen old code, everywhere, at every thread count
        v = fp_violations(GT.fingerprint, GT.sha256_file, GT.data_manifest, GT.tree_manifest, fx, pkg)
        ok(v == [], v)
        ok(GT.fp_threads() >= 1)
        # 2. ... and a digest-altering change to the new code is RED: each mutation below makes the same oracle report a violation
        orig = {k: getattr(GT, k) for k in ("_stat_lines", "sha256_file", "_DIR_SKIP_NAMES", "data_manifest")}

        @contextlib.contextmanager
        def mutated(**kw):
            for k, f in kw.items():
                setattr(GT, k, f)
            try:
                yield
            finally:
                for k, f in orig.items():
                    setattr(GT, k, f)

        def cheap():
            return fp_violations(GT.fingerprint, GT.sha256_file, GT.data_manifest, GT.tree_manifest, fx, pkg, threads=("16",), thorough=False)

        def no_prefix(dirpath, rel_prefix, names):                      # relative path dropped: two files with one name look like one
            return orig["_stat_lines"](dirpath, "", names)

        def short_tail(dirpath, rel_prefix, names):                     # the last file of every stat task is lost
            return orig["_stat_lines"](dirpath, rel_prefix, names[:-1] if len(names) > 1 else names)

        def size_plus_one(dirpath, rel_prefix, names):                  # a size off by one on every file
            ls, t = orig["_stat_lines"](dirpath, rel_prefix, names)
            return [f"{ln.rsplit(chr(9), 1)[0]}\t{int(ln.rsplit(chr(9), 1)[1]) + 1}\n" for ln in ls], t + len(ls)

        def lose_last_byte(p, chunk: int = 1 << 22):                    # a digest that drops one byte of the last read
            h = hashlib.sha256()
            with open(p, "rb", buffering=0) as fh:
                buf = bytearray(chunk)
                mv = memoryview(buf)
                while True:
                    n = fh.readinto(mv)
                    if not n:
                        break
                    h.update(mv[:n - 1] if 1 < n < chunk else mv[:n])
            return h.hexdigest()

        def dm_no_gate_inputs(argv, data_flags, gate_inputs=None, path_map=None):
            return orig["data_manifest"](argv, data_flags, None, path_map)
        for name, kw in (("relative path dropped from the listing", dict(_stat_lines=no_prefix)),
                         ("last file of each stat task lost", dict(_stat_lines=short_tail)),
                         ("every size off by one", dict(_stat_lines=size_plus_one)),
                         ("sha256 loses a byte of the last read", dict(sha256_file=lose_last_byte)),
                         ("skip names no longer pruned", dict(_DIR_SKIP_NAMES=frozenset())),
                         ("gate inputs dropped from the data manifest", dict(data_manifest=dm_no_gate_inputs))):
            with mutated(**kw):
                red = cheap()
            ok(len(red) > 0, f"mutation not caught: {name}")
        if fx["symlinks"]:
            def lstat_lines(dirpath, rel_prefix, names):                 # symlinks NOT followed: a link's own size, not its target's
                lines, total = [], 0
                for fn in names:
                    size = os.lstat(os.path.join(dirpath, fn)).st_size
                    lines.append(f"{rel_prefix}{fn}\t{size}\n")
                    total += size
                return lines, total
            with mutated(_stat_lines=lstat_lines):
                ok(len(cheap()) > 0, "mutation not caught: symlinks no longer followed")
        # 3. the mutated state was restored: the clean run is clean again
        ok(cheap() == [])
        # 4. serial == threaded, on a directory that spans many stat tasks, and the thread-count switch is sane
        os.environ[GT.FP_THREADS_ENV] = "1"
        s1 = GT.fingerprint(fx["big"])
        os.environ[GT.FP_THREADS_ENV] = "8"
        GT.FP_STAT_CHUNK = 3
        try:
            s8 = GT.fingerprint(fx["big"])
        finally:
            GT.FP_STAT_CHUNK = 256
            os.environ.pop(GT.FP_THREADS_ENV, None)
        ok(s1 == s8 == _old_fingerprint(fx["big"]) and s1["n_files"] == 41)
        for junk in ("junk", "0", "-3", ""):
            os.environ[GT.FP_THREADS_ENV] = junk
            ok(GT.fp_threads() == GT.FP_THREADS_DEFAULT, junk)
        os.environ[GT.FP_THREADS_ENV] = "1"
        ok(GT.fp_threads() == 1)
        os.environ.pop(GT.FP_THREADS_ENV, None)
        # 5. the token path end to end: a token minted from the OLD manifests verifies against the NEW ones, and a file added afterwards still refuses
        key = bytes(range(32))
        argv = ["--targets", str(fx["nested"]), "--calib", str(td / "fx" / "flat" / "f6.bin")]
        _, path, tok = GT.mint(profile="refe_x", commit="a" * 40, tree_man=_old_tree_manifest(pkg), argv=argv, trainer="refe/train.py",
                               data=_old_data_manifest(argv, ("--targets", "--calib")), checks={"G": {"status": "PASS"}}, reasons=[], key=key, out_dir=td / "out")
        good, why, _ = GT.verify_token(path, argv, pkg, key=key, data_flags=("--targets", "--calib"))
        ok(good and not why, why)
        late = td / "fx" / "nested" / "a" / "new_file.txt"
        late.write_text("late", encoding="utf-8")
        ok(not GT.verify_token(path, argv, pkg, key=key, data_flags=("--targets", "--calib"))[0])
        late.unlink()
        ok(GT.verify_token(path, argv, pkg, key=key, data_flags=("--targets", "--calib"))[0])
