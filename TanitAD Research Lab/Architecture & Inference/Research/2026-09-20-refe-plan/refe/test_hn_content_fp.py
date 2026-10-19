#!/usr/bin/env python3
"""HNSCREEN-CACHE-CONTENT-FP: the P6-HN screen's `--ctx-cache` is bound by a per-file sha256 CONTENT manifest, not by a size listing.  (raw/2026-10-19-hnscreen-content-fp/RESULT.md)

Run:  PYTHONIOENCODING=utf-8 python refe/test_hn_content_fp.py   ->   ZZTEST_PASS <n>   (prints the mutation table).   CPU only; every fixture is a tiny real cache built by the REAL `ctx_cache.finalize`.
      python refe/test_hn_content_fp.py --arms   runs only the arms against THIS tree and prints `ZZHNCFP_RED <check>` per failing check (what the mutation matrix reads).

THE DEFECT (MEASURED, raw/2026-10-19-hnscreen-fp/out_probe_dir_fp_limit.log): `gate_token.fingerprint(dir)` is the LISTING (relpath + size): a byte flipped INSIDE a cache file at the same size passes it.
THE FIX: `ctx_cache.finalize` SEALS the cache (sha256 of every file -> content_manifest.json, once); the gate's `--ctx-cache` fingerprint re-hashes every file against it (`gate_token.content_fingerprint`).

WHAT IS PROVEN, and through WHICH CALLER (CLAUDE.md "built, tested, and unreachable from its caller" -- the arms below run the gate's own entry points, not only the helper):
  U  gate_token: sealed = valid `dirc:` key == sha256 of the manifest file == an independent hashlib oracle; a same-size flip, a renamed / added / deleted file, a missing, malformed, non-canonical, foreign or
     stale manifest, an empty directory are all INVALID; the listing fingerprint IS blind to the flip (the defect, reproduced here); thread count never moves the key; every non-content caller is byte-identical.
  B  the BUILDER: `ctx_cache.finalize` leaves a sealed cache (the test builds it through the real build_units + finalize); a partial cache is not sealed; `ctx_cache.py seal|check`; `hn_fingerprints.py`.
  G  the GATE: `launch_gate.py refusals` (cmd_refusals -> profile_refusals -> hnscreen_refusals) on the FILLED profile -- the clean control PASSES; each variant is REFUSED twice over: once with the registered pin
     (both the manifest check and the pin catch it) and once with the pin FOLLOWING the tampered bytes (only the manifest check can catch it; a missing manifest with the registered bytes is caught by nothing else).
     The MINT path (`launch_gate.finalize` -> token) and the SUPERVISOR path (`launch_gate.py verify`, in-process and as the subprocess the chain runs) bind the cache's content: a token minted over a clean
     cache MATCHES, the same token over a tampered cache is REFUSED.  `run_checks` records the content fingerprint in its evidence.
  M  every mutation below edits SOURCE (each snippet occurs EXACTLY once) and must turn the named checks RED in a fresh copy of the tree.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(HERE))
ARMS_MODE = "--arms" in sys.argv
ONLY = set(sys.argv[sys.argv.index("--only") + 1].split(",")) if "--only" in sys.argv else None      # a mutation's run: only the checks it must turn RED (the fixtures are the same)


def check(bad, name, fn):
    try:
        ok = bool(fn())
    except Exception as e:                                            # noqa: BLE001 -- an exception IS a failed check
        ok = False
        if os.environ.get("HNCFP_DEBUG"):
            print("  [check raised]", name, type(e).__name__, str(e)[:300])
    if not ok:
        bad.append(name)
    return ok


def arms() -> tuple:
    import gate_token as GT
    import launch_gate as L
    import gate_checks as C
    import ctx_cache as CC
    import hn_fingerprints as HF
    import test_ctx_cache as TCC
    import torch  # noqa: F401  -- the tiny real cache is built with torch

    bad: list = []
    n_checks = [0]

    def chk(name, fn):
        if ONLY is not None and name not in ONLY:
            return None
        n_checks[0] += 1
        return check(bad, name, fn)

    root = Path(tempfile.mkdtemp(prefix="hncfp_"))
    CONTENT = GT.CONTENT_MANIFEST

    # ============================================================================ fixtures: a REAL cache (build_units + finalize), a bank, the C1-based argv, the FILLED profile
    units = TCC.mk_units()
    model = TCC.Fake().eval()
    trunk_fp = CC.trunk_fingerprint(model.state_dict())
    cache = root / "cache"
    CC.build_units(model, units, TCC.load_batch, TCC.make_forward(model), str(cache), trunk_fp=trunk_fp, log=lambda *_: None)
    CC.finalize(str(cache), units, trunk_fp=trunk_fp, calib_sha256="calib-1")
    bank = root / "bank.hnb"
    bank.write_bytes(b"HNBANK1\n" + bytes(range(200)))
    ref = root / "pristine"
    shutil.copytree(cache, ref)                                         # an untouched twin of the sealed cache, to restore from

    def b_finalize_seals():  # the builder's own output, judged BEFORE anything is derived from it
        fp = GT.content_fingerprint(str(cache))
        man = json.loads((cache / CONTENT).read_text(encoding="utf-8"))
        return (cache / CONTENT).is_file() and fp["n_problems"] == 0 and fp["n_files"] == 1 + 3 * 3 and "manifest.json" in man["files"] and "units/log0.ctx.u16" in man["files"]
    chk("B_finalize_leaves_a_sealed_cache_covering_manifest_json_and_every_unit", b_finalize_seals)
    if not (cache / CONTENT).is_file():                                  # a cache the BUILDER did not seal: every arm below presupposes one
        shutil.rmtree(root, ignore_errors=True)
        return bad, n_checks[0]

    def sha_file(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()

    def oracle_key(d):
        """an INDEPENDENT oracle: the cache's content key from plain hashlib over every file, ignoring the stored manifest"""
        files = {}
        for r_, _ds, fs in os.walk(d):
            for fn in fs:
                full = os.path.join(r_, fn)
                rel = os.path.relpath(full, d).replace(os.sep, "/")
                if rel != CONTENT:
                    b = open(full, "rb").read()
                    files[rel] = (len(b), hashlib.sha256(b).hexdigest())
        return f"dirc:{len(files)}:{sum(v[0] for v in files.values())}:{hashlib.sha256(GT.manifest_bytes(files)).hexdigest()}"

    def variant(name, mutate, src=None):
        d = root / name
        shutil.copytree(src or cache, d)
        mutate(d)
        return d

    def payload(d):
        return sorted((d / "units").glob("*.ctx.u16"))[0]

    def flip(d):
        p = payload(d)
        b = bytearray(p.read_bytes())
        b[len(b) // 2] ^= 0x01                                           # ONE bit of ONE payload byte; the size is unchanged
        p.write_bytes(bytes(b))

    def rename(d):
        p = sorted((d / "units").glob("*.fan.f32"))[0]
        p.rename(p.with_name("renamed.fan.f32"))

    def add(d):
        (d / "units" / "zz_extra.bin").write_bytes(b"x")

    def delete(d):
        sorted((d / "units").glob("*.fan.f32"))[0].unlink()

    def nomanifest(d):
        (d / CONTENT).unlink()

    def stale(d):                                                         # a unit rebuilt AFTER the seal, at another size, the manifest not regenerated
        p = payload(d)
        p.write_bytes(p.read_bytes() + b"\x00" * 16)

    def noncanonical(d):                                                  # the same JSON, one extra space: not the bytes the seal wrote
        p = d / CONTENT
        p.write_bytes(p.read_bytes().replace(b'"algorithm": "sha256"', b'"algorithm":  "sha256"'))

    def malformed(d):
        (d / CONTENT).write_bytes(b"{not json")

    def foreign(d):                                                       # a perfectly valid manifest of ANOTHER directory
        other = root / "other_cache"
        if not other.exists():
            other.mkdir()
            (other / "manifest.json").write_text('{"tool": "other"}', encoding="utf-8")
            (other / "units").mkdir()
            (other / "units" / "x.bin").write_bytes(b"abc")
            GT.seal_content_manifest(other)
        shutil.copy(other / CONTENT, d / CONTENT)

    def resealed(d):                                                      # payload flipped AND the manifest regenerated: self-consistent, only the registered pin can refuse it
        flip(d)
        GT.seal_content_manifest(d)

    VARIANTS = {"flip": (flip, "BYTES CHANGED at the same size"), "rename": (rename, "extra, or a renamed file"), "add": (add, "NOT in the manifest"), "delete": (delete, "is MISSING"),
                "nomanifest": (nomanifest, f"no {CONTENT}"), "stale": (stale, "SIZE changed"), "noncanonical": (noncanonical, "not in the canonical form"), "malformed": (malformed, "is unreadable"),
                "foreign": (foreign, "is MISSING")}
    vdirs = {k: variant("v_" + k, f) for k, (f, _) in VARIANTS.items()}
    vdirs["resealed"] = variant("v_resealed", resealed)

    # ============================================================================ U: gate_token
    def u_sealed_valid():
        fp = GT.content_fingerprint(str(cache))
        key = GT.fp_key(fp)
        man = json.loads((cache / CONTENT).read_text(encoding="utf-8"))
        return (fp["kind"] == "dir-content" and fp["n_problems"] == 0 and key.startswith("dirc:") and fp["content_sha256"] == sha_file(cache / CONTENT) == oracle_key(cache).split(":")[3]
                and key == oracle_key(cache) and man["tool"] == "content_manifest/1" and man["n_files"] == fp["n_files"] and list(man["files"]) == sorted(man["files"]) and CONTENT not in man["files"]
                and "manifest.json" in man["files"] and all(v["sha256"] == sha_file(cache / k) and v["bytes"] == (cache / k).stat().st_size for k, v in man["files"].items())
                and (cache / CONTENT).read_bytes() == GT.manifest_bytes({k: (v["bytes"], v["sha256"]) for k, v in man["files"].items()}) and b"\r" not in (cache / CONTENT).read_bytes())
    chk("U_sealed_cache_is_valid_key_equals_manifest_sha256_equals_independent_oracle", u_sealed_valid)

    def u_listing_blind():                                                # the DEFECT, reproduced: the size listing cannot see the flip; the content fingerprint can
        return (GT.fp_key(GT.fingerprint(str(vdirs["flip"]))) == GT.fp_key(GT.fingerprint(str(cache))) and GT.fp_key(GT.content_fingerprint(str(vdirs["flip"]))).startswith("dir-content-INVALID:")
                and GT.fp_key(GT.content_fingerprint(str(cache))).startswith("dirc:"))
    chk("U_defect_reproduced_listing_fingerprint_blind_to_a_same_size_flip", u_listing_blind)

    for vname, (_f, frag) in VARIANTS.items():
        def u_invalid(vname=vname, frag=frag):
            fp = GT.content_fingerprint(str(vdirs[vname]))
            return fp["kind"] == "dir-content" and fp["n_problems"] >= 1 and any(frag in p for p in fp["problems"]) and GT.fp_key(fp).startswith("dir-content-INVALID:")
        chk(f"U_invalid_{vname}", u_invalid)

    def u_resealed_valid_but_different():
        fp = GT.content_fingerprint(str(vdirs["resealed"]))
        return fp["n_problems"] == 0 and GT.fp_key(fp).startswith("dirc:") and GT.fp_key(fp) != GT.fp_key(GT.content_fingerprint(str(cache))) and GT.fp_key(fp) == oracle_key(vdirs["resealed"])
    chk("U_resealed_flipped_cache_is_self_consistent_but_its_key_differs", u_resealed_valid_but_different)

    def u_empty_and_missing():
        e = root / "empty"
        e.mkdir()
        (e / "sub").mkdir()
        return (GT.fp_key(GT.content_fingerprint(str(e))).startswith("dir-content-INVALID:") and GT.content_fingerprint(str(root / "nope"))["kind"] == "missing"
                and GT.content_fingerprint(str(bank))["kind"] == "file" and GT.fp_key(GT.content_fingerprint(str(bank))) == GT.fp_key(GT.fingerprint(str(bank))))
    chk("U_empty_dir_invalid_missing_is_missing_a_file_is_its_sha256", u_empty_and_missing)

    def u_symlinked_dir():                                                # a symlinked directory is not walked, so its files would be bound by NOTHING: it is a problem, not a silent skip
        real = os.path.islink                                             # (portable: no symlink privilege needed, the walk's own predicate is made to say yes for `units`)
        try:
            os.path.islink = lambda p: os.path.basename(str(p)) == "units"
            fp = GT.content_fingerprint(str(cache))
        finally:
            os.path.islink = real
        return fp["n_problems"] >= 1 and any("symlinked directory" in x and "units" in x for x in fp["problems"]) and GT.fp_key(fp).startswith("dir-content-INVALID:") \
            and GT.fp_key(GT.content_fingerprint(str(cache))).startswith("dirc:")
    chk("U_a_symlinked_directory_is_a_problem_not_a_silent_skip", u_symlinked_dir)

    def u_seal_refuses():
        e = root / "empty2"
        e.mkdir()
        try:
            GT.seal_content_manifest(str(e))
            return False
        except GT.GateError:
            return not (e / CONTENT).exists()
    chk("U_seal_refuses_an_empty_directory", u_seal_refuses)

    def u_threads():
        keys = []
        old = os.environ.get(GT.HASH_THREADS_ENV)
        try:
            for t in ("1", "2", "8"):
                os.environ[GT.HASH_THREADS_ENV] = t
                keys.append(GT.fp_key(GT.content_fingerprint(str(cache))))
        finally:
            if old is None:
                os.environ.pop(GT.HASH_THREADS_ENV, None)
            else:
                os.environ[GT.HASH_THREADS_ENV] = old
        return len(set(keys)) == 1 and keys[0] == oracle_key(cache)
    chk("U_key_is_the_same_for_every_hash_thread_count", u_threads)

    def u_reseal_idempotent():
        d = variant("v_reseal", lambda d: None)
        before = (d / CONTENT).read_bytes()
        GT.seal_content_manifest(d)
        return (d / CONTENT).read_bytes() == before and not (d / (CONTENT + ".tmp")).exists()
    chk("U_resealing_an_unchanged_cache_rewrites_the_same_bytes", u_reseal_idempotent)

    def u_other_callers_identical():
        flags = ("--ctx-cache", "--hn-bank")
        av = ["--ctx-cache", str(cache), "--hn-bank", str(bank)]
        a, b = GT.data_manifest(av, flags), GT.data_manifest(av, flags, content_flags=())
        c = GT.data_manifest(av, flags, content_flags=("--ctx-cache",))
        return (a == b and a["--ctx-cache"]["kind"] == "dir" and c["--ctx-cache"]["kind"] == "dir-content" and c["--hn-bank"] == a["--hn-bank"] and GT.fp_key(a["--ctx-cache"]).startswith("dir:")
                and a["--ctx-cache"]["listing_sha256"] == GT.fingerprint(str(cache))["listing_sha256"])
    chk("U_data_manifest_without_content_flags_is_exactly_the_old_listing", u_other_callers_identical)

    def u_other_profiles():
        return (all(L.content_flags_of(v) == () for k, v in L.PROFILES.items() if k != "refe_hnscreen") and L.content_flags_of(L.PROFILES["refe_hnscreen"]) == ("--ctx-cache",) == L.HNSCREEN_CONTENT_FLAGS
                and L.content_flags_of({**L.PROFILES["refe_hnscreen"], "hn_rules": False}) == ())
    chk("U_only_the_hn_screen_profile_fingerprints_by_content_and_only_ctx_cache", u_other_profiles)

    # ============================================================================ B: the builder

    def b_partial_not_sealed():
        d = root / "partial"
        CC.build_units(model, units[:2], TCC.load_batch, TCC.make_forward(model), str(d), trunk_fp=trunk_fp, log=lambda *_: None)
        try:
            CC.finalize(str(d), units, trunk_fp=trunk_fp, calib_sha256="calib-1")                  # unit 3 is missing
            return False
        except CC.StaleCache:
            return not (d / CONTENT).exists() and not (d / "manifest.json").exists()
    chk("B_a_partial_cache_is_not_finalised_and_not_sealed", b_partial_not_sealed)

    def run_py(script, *args, cwd=None):
        env = dict(os.environ, PYTHONIOENCODING="utf-8", CUDA_VISIBLE_DEVICES="-1")
        return subprocess.run([sys.executable, str(HERE / script), *args], capture_output=True, text=True, timeout=600, env=env, cwd=cwd or str(HERE), encoding="utf-8", errors="replace")

    def b_cli():
        ok_ = run_py("ctx_cache.py", "check", "--dir", str(cache))
        bad_ = run_py("ctx_cache.py", "check", "--dir", str(vdirs["flip"]))
        d = variant("v_legacy", lambda d: (d / CONTENT).unlink())                                    # a cache finalised before sealing existed
        no_man = run_py("ctx_cache.py", "seal", "--dir", str(variant("v_noman", lambda d: (d / "manifest.json").unlink() or (d / CONTENT).unlink())))
        sl = run_py("ctx_cache.py", "seal", "--dir", str(d))
        return (ok_.returncode == 0 and ok_.stdout.startswith(oracle_key(cache)) and bad_.returncode == 1 and "BYTES CHANGED" in bad_.stderr and no_man.returncode == 2 and "REFUSED" in no_man.stderr
                and sl.returncode == 0 and "ZZCTX_SEALED" in sl.stdout and (d / CONTENT).read_bytes() == (cache / CONTENT).read_bytes())
    chk("B_ctx_cache_py_check_and_seal_cli", b_cli)

    def b_helper():
        cp = run_py("hn_fingerprints.py", "--hn-bank", str(bank), "--ctx-cache", str(cache))
        fx = json.loads(cp.stdout) if cp.returncode == 0 else {}
        fl = run_py("hn_fingerprints.py", "--hn-bank", str(bank), "--ctx-cache", str(vdirs["flip"]))
        nm = run_py("hn_fingerprints.py", "--hn-bank", str(bank), "--ctx-cache", str(vdirs["nomanifest"]))
        em = root / "emptyh"
        em.mkdir(exist_ok=True)
        ee = run_py("hn_fingerprints.py", "--hn-bank", str(bank), "--ctx-cache", str(em))
        return (cp.returncode == 0 and fx == {"--hn-bank": GT.fp_key(GT.fingerprint(str(bank))), "--ctx-cache": oracle_key(cache)} and "\r" not in cp.stdout
                and all(r.returncode == 2 and r.stdout == "" and "REFUSED" in r.stderr for r in (fl, nm, ee)) and "content check" in fl.stderr and "EMPTY directory" in ee.stderr)
    chk("B_hn_fingerprints_py_prints_the_dirc_key_and_refuses_an_unsealed_or_tampered_cache", b_helper)

    # ============================================================================ G: the gate's entry points
    import scorer_finetune as S
    c1 = list(json.loads((PKG / "eval" / "rls1b_arms.json").read_text(encoding="utf-8"))["arms"]["C1"]["argv"])

    P0 = L.PROFILES["refe_hnscreen"]
    dummy = root / "dummy_input.bin"
    dummy.write_bytes(b"an input the pod would hold; only its existence matters here")
    c1r = list(c1)
    for f_ in P0["data_flags"]:                                                                   # the other data inputs are RE-ROOTED to a file that exists here (path values may move; only the bank and the cache are pinned)
        if f_ in c1r and f_ not in L.HNSCREEN_FLAGS:
            c1r = C._set(c1r, f_, str(dummy))

    def argv_for(cdir, seed=1):
        return C._set(c1r, "--seed", str(seed)) + ["--hn-bank", str(bank), "--ctx-cache", str(cdir)]
    FX = HF.fingerprints_expected(str(bank), str(cache))                                           # the fill comes from the helper, never typed
    PFc = {**P0, "fingerprints_expected": dict(FX)}
    a_ok = argv_for(cache)

    def cli(cmd, prof, extra, *, stdin=None):
        buf = io.StringIO()
        old = L.PROFILES["refe_hnscreen"]
        L.PROFILES["refe_hnscreen"] = prof
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                rc = L.main([cmd, "--profile", "refe_hnscreen", *extra])
        finally:
            L.PROFILES["refe_hnscreen"] = old
        return rc, buf.getvalue()

    argf = root / "argv_tmp.json"

    def refusals_cli(argv, prof):
        GT.write_json(argf, argv)
        return cli("refusals", prof, ["--arm-name", "S1", "--args-file", str(argf)])

    def g_clean():
        rc, out = refusals_cli(a_ok, PFc)
        return rc == 0 and "ZZGATE-CLEANZZ" in out and L.profile_refusals({**PFc, "declared_arm": "S1"}, a_ok) == []
    chk("G_clean_control_the_sealed_cache_passes_the_refusals_cli", g_clean)

    for vname in list(VARIANTS) + ["resealed"]:
        frag = VARIANTS[vname][1] if vname in VARIANTS else "is not the registered one"

        def g_pin_original(vname=vname, frag=frag):                                                  # the registered pin, the tampered cache: refused (the manifest check and the pin both catch it)
            rc, out = refusals_cli(argv_for(vdirs[vname]), PFc)
            return rc == 5 and "ZZGATE-REFUSEDZZ" in out and "content fingerprint of --ctx-cache" in out and frag in out
        chk(f"G_refused_{vname}_pin_registered", g_pin_original)

        if vname == "resealed":
            continue

        def g_pin_follows(vname=vname, frag=frag):                                                   # the pin REFILLED from the tampered bytes: only the manifest check can refuse it
            d = vdirs[vname]
            if vname == "nomanifest":
                key = oracle_key(cache)                                                              # the registered bytes, unsealed: a content hash of the files alone matches the pin
            else:
                key = oracle_key(d)
            rc, out = refusals_cli(argv_for(d), {**P0, "fingerprints_expected": {**FX, "--ctx-cache": key}})
            return rc == 5 and "ZZGATE-REFUSEDZZ" in out and frag in out
        chk(f"G_refused_{vname}_pin_follows_the_tampered_bytes_only_the_manifest_check_can_refuse", g_pin_follows)

    def g_stale_listing_pin():
        rc, out = refusals_cli(a_ok, {**P0, "fingerprints_expected": {**FX, "--ctx-cache": GT.fp_key(GT.fingerprint(str(cache)))}})
        return rc == 5 and "is not a content fingerprint" in out
    chk("G_refused_a_stale_size_listing_pin_for_ctx_cache", g_stale_listing_pin)

    def g_wrong_dirc_pin():
        rc, out = refusals_cli(a_ok, {**P0, "fingerprints_expected": {**FX, "--ctx-cache": "dirc:10:1:" + "0" * 64}})
        return rc == 5 and "is not the registered one" in out
    chk("G_refused_a_wrong_dirc_pin_on_the_sealed_cache", g_wrong_dirc_pin)

    def g_unfilled_still_refused():
        rc, out = refusals_cli(a_ok, P0)
        return rc == 5 and "fingerprints_expected not filled" in out
    chk("G_the_shipped_profile_still_refuses_while_unfilled", g_unfilled_still_refused)

    # ---- the MINT path (launch_gate.finalize -> token) and the SUPERVISOR path (launch_gate.py verify): a token over a clean cache MATCHES, over a tampered one it does NOT
    pkg = root / "pkg"
    (pkg / "refe").mkdir(parents=True)
    (pkg / "refe" / "dummy.py").write_text("X = 1\n", encoding="utf-8")
    keyf = root / "gate.key"
    key = GT.load_key(keyf, create=True, forbid_under=[pkg])
    commit = "a" * 40
    vcache = root / "vcache"
    shutil.copytree(cache, vcache)
    a_v = argv_for(vcache)
    profv = {**PFc, "declared_arm": "S1"}
    outd = root / "lg"
    outd.mkdir()
    tree = GT.tree_manifest(pkg)
    binding = GT.binding_of(profv["name"], commit, GT.tree_digest(tree), a_v)
    for c_ in profv["checks"] + tuple(profv.get("na_ok", ())):                                    # PASS evidence for every check, bound to this launch: the MINT then runs the gate's own finalize()
        L.write_evidence(outd, C.evidence(c_, "PASS", [], {}), binding, {})
    verdict, tokp, tok = L.finalize(profv, a_v, pkg, outd, key, commit=commit)
    argv_json = root / "argv_v.json"
    GT.write_json(argv_json, a_v)

    def verify_cli():
        return cli("verify", PFc, ["--token", str(tokp), "--args-file", str(argv_json), "--pkg", str(pkg), "--key-file", str(keyf)])

    def verify_sub():                                                                              # the exact command shape of hnscreen_devbox.verify_pass_token (the supervisor's step)
        return subprocess.run([sys.executable, str(HERE / "launch_gate.py"), "verify", "--profile", "refe_hnscreen", "--token", str(tokp), "--args-file", str(argv_json), "--pkg", str(pkg),
                               "--key-file", str(keyf)], capture_output=True, text=True, timeout=600, encoding="utf-8", errors="replace", env=dict(os.environ, PYTHONIOENCODING="utf-8", CUDA_VISIBLE_DEVICES="-1"))

    def restore():
        shutil.rmtree(vcache)
        shutil.copytree(ref, vcache)

    chk("G_mint_finalize_PASSES_on_the_sealed_cache_and_binds_its_CONTENT_fingerprint",
        lambda: verdict == "PASS" and tok["binding"]["data"]["--ctx-cache"]["kind"] == "dir-content" and GT.fp_key(tok["binding"]["data"]["--ctx-cache"]) == oracle_key(vcache)
        and tok["binding"]["data"]["--hn-bank"]["kind"] == "file")

    def v_clean():
        rc, out = verify_cli()
        sp = verify_sub()
        return rc == 0 and "ZZGATE-MATCHZZ" in out and sp.returncode == 0 and "ZZGATE-MATCHZZ" in sp.stdout
    chk("G_verify_clean_control_MATCHES_in_process_and_as_the_chains_subprocess", v_clean)

    for vname in ("flip", "rename", "add", "delete", "nomanifest", "stale", "noncanonical", "malformed", "foreign"):
        def v_tampered(vname=vname):
            restore()
            VARIANTS[vname][0](vcache)                                                              # the cache is tampered AFTER the token was minted
            try:
                rc, out = verify_cli()
                sp = verify_sub()
            finally:
                restore()
            return rc == 5 and "ZZGATE-REFUSEDZZ" in out and "--ctx-cache" in out and "data differs" in out and sp.returncode == 5 and "ZZGATE-REFUSEDZZ" in sp.stdout
        chk(f"G_verify_REFUSES_{vname}_after_the_token_was_minted_in_process_and_as_subprocess", v_tampered)

    def v_names_the_file():
        restore()
        flip(vcache)
        try:
            rc, out = verify_cli()
        finally:
            restore()
        return rc == 5 and "BYTES CHANGED at the same size" in out and "units/log0.ctx.u16" in out
    chk("G_verify_refusal_names_the_changed_file", v_names_the_file)

    def v_after_restore():
        restore()
        rc, out = verify_cli()
        return rc == 0 and "ZZGATE-MATCHZZ" in out
    chk("G_verify_matches_again_once_the_cache_is_restored_byte_for_byte", v_after_restore)

    def g_finalize_fail():                                                                          # the MINT over a tampered cache: FAIL, with the content refusal in the token's reasons
        d2 = root / "fcache"
        shutil.copytree(vdirs["flip"], d2)
        a2 = argv_for(d2)
        out2 = root / "lg2"
        out2.mkdir()
        b2 = GT.binding_of(profv["name"], commit, GT.tree_digest(tree), a2)
        for c_ in profv["checks"] + tuple(profv.get("na_ok", ())):
            L.write_evidence(out2, C.evidence(c_, "PASS", [], {}), b2, {})
        v2, _p, t2 = L.finalize(profv, a2, pkg, out2, key, commit=commit)
        txt = " | ".join(t2["reasons"])
        return v2 == "FAIL" and "content fingerprint of --ctx-cache" in txt and "FAILS its content check" in txt and "BYTES CHANGED at the same size" in txt
    chk("G_mint_finalize_FAILS_on_a_same_size_flip", g_finalize_fail)

    def g_run_checks():                                                                              # run_checks (the gate's `run`) records the CONTENT fingerprint and fails every check on a tampered cache
        out3 = root / "lg3"
        st = L.run_checks(profv, argv_for(vdirs["flip"]), pkg, out3, commit=commit)
        ev = json.loads((out3 / "evidence_G-HYG.json").read_text(encoding="utf-8"))
        return (set(st.values()) == {"FAIL"} and ev["inputs_read"]["--ctx-cache"]["kind"] == "dir-content" and ev["inputs_read"]["--ctx-cache"]["n_problems"] >= 1
                and any("content fingerprint of --ctx-cache" in r for r in ev["reasons"]))
    chk("G_run_checks_fails_every_check_on_a_same_size_flip_and_records_the_content_fingerprint", g_run_checks)

    def g_cost_in_full_passes():                                                                    # COST, pinned: every gate entry point re-reads the WHOLE cache; how many times is a measured fact, not a guess
        counts = []
        real = GT._sha256_and_size
        n_files = len(json.loads((cache / CONTENT).read_text(encoding="utf-8"))["files"])

        def counting(path, *a, **k):
            if os.path.abspath(str(path)).startswith(os.path.abspath(str(vcache))):
                counts.append(1)
            return real(path, *a, **k)

        def passes(fn):
            counts.clear()
            fn()
            return len(counts) / n_files
        GT._sha256_and_size = counting
        try:
            restore()
            res = {"refusals_cli": passes(lambda: refusals_cli(argv_for(vcache), PFc)), "verify_cli": passes(lambda: verify_cli()),
                   "run_checks": passes(lambda: L.run_checks(profv, argv_for(vcache), pkg, root / "lg_cost", commit=commit, only=("G-HYG",))),
                   "finalize": passes(lambda: L.finalize(profv, a_v, pkg, outd, key, commit=commit))}
        finally:
            GT._sha256_and_size = real
        return res == {"refusals_cli": 1.0, "verify_cli": 1.0, "run_checks": 2.0, "finalize": 2.0}
    chk("G_cost_one_full_read_per_check_refusals_1_verify_1_run_2_finalize_2", g_cost_in_full_passes)

    shutil.rmtree(root, ignore_errors=True)
    return bad, n_checks[0]


# ================================================================================================== M: source mutations (each snippet must occur EXACTLY once; each must turn the named checks RED)
FLIPS = ["G_refused_flip_pin_registered", "G_refused_flip_pin_follows_the_tampered_bytes_only_the_manifest_check_can_refuse"]
MUTATIONS = [
    # --- the line that compares the content hash (the task's mutation): deleting it must make the same-size flip PASS somewhere
    ("gate_token.content_check: the per-file byte comparison is deleted",
     "gate_token.py", [("    problems += [f\"BYTES CHANGED at the same size ({live[r][0]}): {r} (sha256 manifest {stored[r][1][:12]}, now {live[r][1][:12]})\" for r in sorted(live)\n                 if live[r][1] != stored[r][1] or live[r][0] != stored[r][0]]\n",
                       "    pass  # MUTANT\n")],
     ["U_invalid_flip", "G_refused_flip_pin_follows_the_tampered_bytes_only_the_manifest_check_can_refuse", "G_refused_flip_pin_registered", "G_verify_names_the_changed_file".replace("verify_names", "verify_refusal_names")]),
    ("launch_gate.hnscreen_refusals: the comparison with the registered pin is deleted",
     "launch_gate.py", [("        if fk != want:\n            out.append(f\"{name}: the content fingerprint of {f} is not the registered one (RL-S1 C1 / C0's token fingerprint)\")\n",
                         "        if False:  # MUTANT\n            out.append(f\"{name}: the content fingerprint of {f} is not the registered one (RL-S1 C1 / C0's token fingerprint)\")\n")],
     ["G_refused_resealed_pin_registered", "G_refused_a_wrong_dirc_pin_on_the_sealed_cache"]),
    ("launch_gate.hnscreen_refusals: BOTH content comparisons are deleted (the same-size flip must now PASS the refusals)",
     "launch_gate.py", [("        if fk != want:\n            out.append(f\"{name}: the content fingerprint of {f} is not the registered one (RL-S1 C1 / C0's token fingerprint)\")\n",
                         "        if False:  # MUTANT\n            out.append(f\"{name}: the content fingerprint of {f} is not the registered one (RL-S1 C1 / C0's token fingerprint)\")\n"),
                        ("        if by_content and fp.get(\"n_problems\"):\n", "        if False:\n")],
     ["G_refused_flip_pin_registered", "G_refused_flip_pin_follows_the_tampered_bytes_only_the_manifest_check_can_refuse", "G_refused_resealed_pin_registered", "G_refused_add_pin_registered", "G_refused_delete_pin_registered"]),
    ("gate_token.verify_token: the data fingerprint comparison is deleted (the supervisor trusts the token)",
     "gate_token.py", [("    d_bad = [k for k in sorted(set(want) | set(data)) if k not in want or k not in data or fp_key(want[k]) != fp_key(data[k])]\n", "    d_bad = []  # MUTANT\n")],
     ["G_verify_REFUSES_flip_after_the_token_was_minted_in_process_and_as_subprocess", "G_verify_REFUSES_add_after_the_token_was_minted_in_process_and_as_subprocess",
      "G_verify_REFUSES_delete_after_the_token_was_minted_in_process_and_as_subprocess", "G_verify_REFUSES_nomanifest_after_the_token_was_minted_in_process_and_as_subprocess"]),
    ("launch_gate.cmd_verify: the content flags are not passed to verify_token (the profile -> caller wiring dropped)",
     "launch_gate.py", [("commit=a.commit, content_flags=content_flags_of(prof) if prof else ())", "commit=a.commit)")],
     ["G_verify_clean_control_MATCHES_in_process_and_as_the_chains_subprocess"]),                       # fail CLOSED: a token minted by content no longer matches a listing, so the clean control breaks; the flip is never silently passed
    ("launch_gate.finalize AND cmd_verify both drop the content flags (the token binds a size listing and the supervisor compares one: the same-size flip must now PASS the supervisor)",
     "launch_gate.py", [("commit=a.commit, content_flags=content_flags_of(prof) if prof else ())", "commit=a.commit)"),
                        ("    data = GT.data_manifest(argv, prof[\"data_flags\"], gate_inputs, timings=dtim, content_flags=content_flags_of(prof))\n    _log_manifest_timing(\"finalize\"",
                         "    data = GT.data_manifest(argv, prof[\"data_flags\"], gate_inputs, timings=dtim)\n    _log_manifest_timing(\"finalize\"")],
     ["G_verify_REFUSES_flip_after_the_token_was_minted_in_process_and_as_subprocess", "G_mint_finalize_PASSES_on_the_sealed_cache_and_binds_its_CONTENT_fingerprint"]),
    ("launch_gate.finalize: the content flags are not passed to data_manifest (the token binds a size listing)",
     "launch_gate.py", [("    data = GT.data_manifest(argv, prof[\"data_flags\"], gate_inputs, timings=dtim, content_flags=content_flags_of(prof))\n    _log_manifest_timing(\"finalize\"",
                         "    data = GT.data_manifest(argv, prof[\"data_flags\"], gate_inputs, timings=dtim)\n    _log_manifest_timing(\"finalize\"")],
     ["G_mint_finalize_PASSES_on_the_sealed_cache_and_binds_its_CONTENT_fingerprint", "G_mint_finalize_FAILS_on_a_same_size_flip"]),
    ("launch_gate.run_checks: the content flags are not passed to data_manifest",
     "launch_gate.py", [("    data = GT.data_manifest(argv, prof[\"data_flags\"], gate_inputs, timings=dtim, content_flags=content_flags_of(prof))\n    _log_manifest_timing(\"run\"",
                         "    data = GT.data_manifest(argv, prof[\"data_flags\"], gate_inputs, timings=dtim)\n    _log_manifest_timing(\"run\"")],
     ["G_run_checks_fails_every_check_on_a_same_size_flip_and_records_the_content_fingerprint"]),
    ("launch_gate.content_flags_of: the profile no longer fingerprints by content",
     "launch_gate.py", [("    return HNSCREEN_CONTENT_FLAGS if prof.get(\"hn_rules\") else ()\n", "    return ()\n")],
     ["U_only_the_hn_screen_profile_fingerprints_by_content_and_only_ctx_cache", "G_verify_REFUSES_flip_after_the_token_was_minted_in_process_and_as_subprocess"]),
    ("launch_gate.hnscreen_refusals: --ctx-cache is fingerprinted by size listing again (the defect)",
     "launch_gate.py", [("        by_content = f in HNSCREEN_CONTENT_FLAGS\n", "        by_content = False\n")],
     ["G_clean_control_the_sealed_cache_passes_the_refusals_cli"]),
    ("gate_token.content_check: a missing manifest is silent (no problem, nothing listed)",
     "gate_token.py", [("        return {\"problems\": [f\"no {CONTENT_MANIFEST} in {root}: the directory was not sealed at build (ctx_cache.finalize writes it; a cache built without it: `python refe/ctx_cache.py seal --dir <dir>`)\"],\n                \"live\": None, \"manifest_file_sha256\": None}\n",
                       "        return {\"problems\": [], \"live\": {r_: _sha256_and_size(root / r_) for r_ in _content_listing(root)[0]}, \"manifest_file_sha256\": None}  # MUTANT\n")],
     ["G_refused_nomanifest_pin_follows_the_tampered_bytes_only_the_manifest_check_can_refuse", "U_invalid_nomanifest", "G_verify_REFUSES_nomanifest_after_the_token_was_minted_in_process_and_as_subprocess"]),
    ("gate_token.content_check: an extra file is not reported",
     "gate_token.py", [("    problems += [f\"file NOT in the manifest (extra, or a renamed file): {r}\" for r in sorted(set(sizes) - set(stored))]\n", "    pass  # MUTANT\n")],
     ["G_refused_add_pin_follows_the_tampered_bytes_only_the_manifest_check_can_refuse", "U_invalid_add"]),
    ("gate_token.content_check: a missing file is not reported",
     "gate_token.py", [("    problems += [f\"file in the manifest is MISSING: {r}\" for r in sorted(set(stored) - set(sizes))]\n", "    pass  # MUTANT\n")],
     ["G_refused_delete_pin_follows_the_tampered_bytes_only_the_manifest_check_can_refuse", "U_invalid_delete"]),
    ("gate_token.content_check: a stale manifest (a resized file) is not reported",
     "gate_token.py", [("    problems += [f\"SIZE changed: {r} (manifest {stored[r][0]} bytes, now {sizes[r]})\" for r in sorted(set(stored) & set(sizes)) if stored[r][0] != sizes[r]]\n", "    pass  # MUTANT\n")],
     ["G_refused_stale_pin_follows_the_tampered_bytes_only_the_manifest_check_can_refuse", "U_invalid_stale"]),
    ("gate_token.content_check: the canonical-form check is deleted",
     "gate_token.py", [("    if stored is not None and raw != manifest_bytes(stored):\n", "    if False:\n")],
     ["U_invalid_noncanonical", "G_refused_noncanonical_pin_registered"]),
    ("gate_token._content_listing: a symlinked directory is silently skipped",
     "gate_token.py", [("        problems += [f\"a symlinked directory is not followed, so its files cannot be bound: {os.path.relpath(os.path.join(dirpath, d), rootp).replace(os.sep, '/')}\"\n                     for d in dirnames if os.path.islink(os.path.join(dirpath, d))]\n",
                        "        pass  # MUTANT\n")],
     ["U_a_symlinked_directory_is_a_problem_not_a_silent_skip"]),
    ("gate_token.fp_key: an INVALID content fingerprint is keyed like a valid one (problems ignored)",
     "gate_token.py", [("        if fp.get(\"n_problems\") or not fp.get(\"content_sha256\"):\n", "        if False:\n")],
     ["U_defect_reproduced_listing_fingerprint_blind_to_a_same_size_flip"]),
    ("ctx_cache.finalize: the cache is not sealed at build",
     "ctx_cache.py", [("    seal(out_dir)                                                  # LAST", "    pass                                                           # MUTANT LAST")],
     ["B_finalize_leaves_a_sealed_cache_covering_manifest_json_and_every_unit"]),
    ("hn_fingerprints: --ctx-cache is described by the size listing",
     "hn_fingerprints.py", [("    return GT.content_fingerprint(str(path)) if flag in HNSCREEN_CONTENT_FLAGS else GT.fingerprint(str(path))", "    return GT.fingerprint(str(path))")],
     ["B_hn_fingerprints_py_prints_the_dirc_key_and_refuses_an_unsealed_or_tampered_cache"]),
]


def apply_edits(src, edits, name):
    for old, new in edits:
        n = src.count(old)
        if n != 1:
            raise AssertionError(f"mutation {name!r}: snippet occurs {n} times (need exactly 1): {old[:90]!r}")
        src = src.replace(old, new)
    return src


def mutant_tree(dst: Path, fname: str, edits, name):
    """a fresh copy of the code the arms import (refe/*.py, code/, the C1 argv file) with ONE source edit applied"""
    ign = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(HERE, dst / "refe", ignore=ign)
    for sub in ("code", "splits"):
        if (PKG / sub).is_dir():
            shutil.copytree(PKG / sub, dst / sub, ignore=ign)
    (dst / "eval").mkdir()
    shutil.copy(PKG / "eval" / "rls1b_arms.json", dst / "eval" / "rls1b_arms.json")
    p = dst / "refe" / fname
    src = p.read_text(encoding="utf-8")
    p.write_text(apply_edits(src, edits, name), encoding="utf-8", newline="\n")


def run_arms(tree_refe: Path, only=None) -> list:
    env = dict(os.environ, PYTHONIOENCODING="utf-8", CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, str(tree_refe / "test_hn_content_fp.py"), "--arms"] + (["--only", ",".join(only)] if only else []), capture_output=True, text=True, timeout=1800, env=env,
                       cwd=str(tree_refe), encoding="utf-8", errors="replace")
    red = [ln.split(" ", 1)[1].strip() for ln in r.stdout.splitlines() if ln.startswith("ZZHNCFP_RED ")]
    if "ZZHNCFP_ARMS_DONE" not in r.stdout:
        red.append(f"arms process died rc {r.returncode}: {(r.stdout + r.stderr)[-300:]!r}")
    return red


def main() -> int:
    if ARMS_MODE:
        bad, n = arms()
        for b in bad:
            print("ZZHNCFP_RED", b)
        print(f"ZZHNCFP_ARMS_DONE checks {n} red {len(bad)}")
        return 0
    bad, n = arms()
    if bad:
        print("\n".join("PROBLEM the unmutated tree FAILS: " + b for b in bad))
        print(f"ZZTEST_FAIL {len(bad)} of {n} arm(s) failed on the real tree")
        return 1
    print(f"{n} arms green on the real tree.  Mutation matrix (each mutant is a fresh copy of the tree with ONE source edit; it runs the checks it must turn RED):")
    problems, k = [], 1
    for name, fname, edits, expect in MUTATIONS:
        tmp = Path(tempfile.mkdtemp(prefix="hncfp_mut_"))
        try:
            mutant_tree(tmp, fname, edits, name)
            red = run_arms(tmp / "refe", only=expect)
        except AssertionError as e:
            problems.append(str(e))
            red = []
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        miss = [c for c in expect if c not in red]
        if not red:
            problems.append(f"mutation {name!r} stayed GREEN: no arm can see it")
        elif miss:
            problems.append(f"mutation {name!r}: expected RED and stayed green: {miss} (RED: {red[:8]})")
        print(f"{name[:100]:100s} RED {len(red):3d}  e.g. {', '.join(red[:2])[:110]}", flush=True)
        k += 1
    if problems:
        print("\n".join("PROBLEM " + p for p in problems))
        print(f"ZZTEST_FAIL {len(problems)} problem(s)")
        return 1
    print(f"ZZTEST_PASS {n + len(MUTATIONS)}  ({n} arms on the real tree + {len(MUTATIONS)} RED mutations)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
