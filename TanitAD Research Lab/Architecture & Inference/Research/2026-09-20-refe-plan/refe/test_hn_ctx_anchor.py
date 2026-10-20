#!/usr/bin/env python3
"""HNSCREEN-FP-COST (MM ruling (b'), 2026-10-19): the HN screen's --ctx-cache is read in full 4 times per screen, not 20.  (raw/2026-10-19-hnscreen-fp-cost/RESULT.md)

Run:  PYTHONIOENCODING=utf-8 python refe/test_hn_ctx_anchor.py               ->  ZZTEST_PASS <n>   (arms + the mutation matrix, about 6 min)
      python refe/test_hn_ctx_anchor.py --no-mutants                        ->  ZZTEST_PASS <n>   (arms only: what G-SUITE runs; the matrix is run at landing and in the sibling suite)
      python refe/test_hn_ctx_anchor.py --arms [--only a,b]                 ->  `ZZHNANC_RED <check>` per failing check (what the matrix reads)
CPU only; every fixture is a tiny real cache built by the REAL `ctx_cache.finalize`; every arm runs the gate's OWN entry points (`launch_gate.main`: anchor / verify / refusals / end-verify, `cmd_run`, `finalize`,
`run_checks`; `verify` also as the subprocess the chain runs), not only the helper functions.

THE RULE (MM_RULING_FPCOST.md): the content check is kept IN FULL at the seal, the fill, the START of the chain (`launch_gate.py anchor`: one full fingerprint vs the registered pin, plus the signed
(relpath, size, mtime_ns) of EVERY file) and the END of the chain (`end-verify`: a second full fingerprint vs the anchor + a final stat compare).  An intermediate call in `--ctx-mode anchor-bound` reads NO byte: any
mismatch, missing or extra file is a refusal.  The residual (a same-size edit that restores the mtime_ns) is detected at the END: VOID, and the reader refuses a chain without an END_VERIFY_OK bound to the same anchor.
`content` (the default) is today's behaviour.

THE NAMED ARMS (the ruling's R1-R7):
  R1  a same-size, same-mtime_ns byte edit mid-chain: every intermediate call still MATCHES (the residual is real), the END verify FAILS -> VOID -> the reader gate refuses.
  R2  a size change: every intermediate entry point (verify in process and as the chain's subprocess, refusals, run_checks, finalize) refuses.
  R3  an mtime_ns change with identical bytes: refuses (conservative by design); restoring the mtime matches again; the END verify also voids it.
  R4  an extra file / a missing file / a renamed file: refuses.
  R5  the end verify skipped, deleted, VOID, bound to another anchor, edited, covering fewer seeds, tokens bound to two anchors, a listing-only token: the reader gate refuses.
  R6  an anchor from another cache directory, another chain, minted before the chain started, edited, signed with another key: refuses.  The mint refuses a cache that is not the registered pin / is unfilled / is tampered / changes while it is anchored.
  R7  `content` mode is today's behaviour (the identity arm): same data manifest, same refusals text, same verdicts; an anchor named under content is refused; the default is content; the full-read counts are the
      pinned 1 / 1 / 2 / 2 / 4; anchor-bound reads ZERO bytes at every intermediate entry point.
  W   wiring: `cmd_run`, `cmd_verify`, `cmd_refusals` hand THIS call's anchor to run_checks / finalize / verify_token / profile_refusals (built-and-tested-but-unreachable is the class CLAUDE.md names).
Every mutation below edits SOURCE (each snippet occurs EXACTLY once), deletes the single line that enforces a refusal, and must turn the named arms RED in a fresh copy of the tree.
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
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(HERE))
ARMS_MODE = "--arms" in sys.argv
NO_MUTANTS = "--no-mutants" in sys.argv
ONLY = set(sys.argv[sys.argv.index("--only") + 1].split(",")) if "--only" in sys.argv else None


def check(bad, name, fn):
    try:
        ok = bool(fn())
    except Exception as e:                                            # noqa: BLE001 -- an exception IS a failed check
        ok = False
        if os.environ.get("HNANC_DEBUG"):
            import traceback
            traceback.print_exc()
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

    root = Path(tempfile.mkdtemp(prefix="hnanc_"))
    CONTENT = GT.CONTENT_MANIFEST

    # ============================================================================ fixtures: a REAL sealed cache, a bank, the C1-based argv, the FILLED profile, a pkg, a key
    units = TCC.mk_units()
    model = TCC.Fake().eval()
    trunk_fp = CC.trunk_fingerprint(model.state_dict())
    cache = root / "cache"
    CC.build_units(model, units, TCC.load_batch, TCC.make_forward(model), str(cache), trunk_fp=trunk_fp, log=lambda *_: None)
    CC.finalize(str(cache), units, trunk_fp=trunk_fp, calib_sha256="calib-1")
    bank = root / "bank.hnb"
    bank.write_bytes(b"HNBANK1\n" + bytes(range(200)))
    pristine = root / "pristine"
    shutil.copytree(cache, pristine)
    dummy = root / "dummy_input.bin"
    dummy.write_bytes(b"an input the pod would hold; only its existence matters here")
    c1 = list(json.loads((PKG / "eval" / "rls1b_arms.json").read_text(encoding="utf-8"))["arms"]["C1"]["argv"])
    P0 = L.PROFILES["refe_hnscreen"]
    c1r = list(c1)
    for f_ in P0["data_flags"]:
        if f_ in c1r and f_ not in L.HNSCREEN_FLAGS:
            c1r = C._set(c1r, f_, str(dummy))
    FX = HF.fingerprints_expected(str(bank), str(pristine))
    PFc = {**P0, "fingerprints_expected": dict(FX)}
    profv = {**PFc, "declared_arm": "S1"}
    pkg = root / "pkg"
    (pkg / "refe").mkdir(parents=True)
    (pkg / "refe" / "dummy.py").write_text("X = 1\n", encoding="utf-8")
    keyf = root / "gate.key"
    key = GT.load_key(keyf, create=True, forbid_under=[pkg])
    commit = "a" * 40
    tree = GT.tree_manifest(pkg)
    checks_all = profv["checks"] + tuple(profv.get("na_ok", ()))

    def argv_for(cdir, seed=1):
        return C._set(c1r, "--seed", str(seed)) + ["--hn-bank", str(bank), "--ctx-cache", str(cdir)]

    def sha_file(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()

    def oracle_key(d):
        files = {}
        for r_, _ds, fs in os.walk(d):
            for fn in fs:
                full = os.path.join(r_, fn)
                rel = os.path.relpath(full, d).replace(os.sep, "/")
                if rel != CONTENT:
                    b = open(full, "rb").read()
                    files[rel] = (len(b), hashlib.sha256(b).hexdigest())
        return f"dirc:{len(files)}:{sum(v[0] for v in files.values())}:{hashlib.sha256(GT.manifest_bytes(files)).hexdigest()}"

    def cli(cmd, prof, extra):
        buf = io.StringIO()
        old = L.PROFILES["refe_hnscreen"]
        L.PROFILES["refe_hnscreen"] = prof
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                rc = L.main([cmd, "--profile", "refe_hnscreen", *extra])
        finally:
            L.PROFILES["refe_hnscreen"] = old
        return rc, buf.getvalue()

    seq = [0]

    class W:
        pass

    def work(prof_pin=PFc):
        """a working copy of the sealed cache, ANCHORED through the gate's own `anchor` command, with a PASS token minted by the real `finalize` in anchor-bound mode"""
        seq[0] += 1
        w = W()
        w.n = seq[0]
        w.dir = root / f"w{w.n}"
        shutil.copytree(pristine, w.dir)
        w.argv = argv_for(w.dir)
        w.argf = root / f"argv_w{w.n}.json"
        GT.write_json(w.argf, w.argv)
        w.chain = hashlib.sha256(f"chain{w.n}".encode()).hexdigest()[:32]
        w.start = time.time_ns()
        w.anchor_p = root / f"anchor_w{w.n}.json"
        rc, out = cli("anchor", prof_pin, ["--ctx-cache", str(w.dir), "--chain-id", w.chain, "--out", str(w.anchor_p), "--pkg", str(pkg), "--key-file", str(keyf)])
        assert rc == 0 and "ZZGATE-ANCHOR-OKZZ" in out, out
        w.doc, _r, w.sha = GT.load_anchor(w.anchor_p, key)
        w.anc = {"doc": w.doc, "sha256": w.sha}
        w.ctx = ["--ctx-mode", "anchor-bound", "--anchor", str(w.anchor_p), "--chain-id", w.chain, "--chain-start-ns", str(w.start)]
        w.lg = root / f"lg_w{w.n}"
        w.lg.mkdir()
        profa = {**profv, "ctx_anchor": w.anc}
        binding = GT.binding_of(profa["name"], commit, GT.tree_digest(tree), w.argv)
        for c_ in checks_all:
            L.write_evidence(w.lg, C.evidence(c_, "PASS", [], {}), binding, {})
        w.verdict, w.tokp, w.tok = L.finalize(profa, w.argv, pkg, w.lg, key, commit=commit)
        return w

    def verify_cli(w, ctx=None, token=None, argf=None):
        rec = root / f"vrec_{w.n}_{time.time_ns()}.json"
        rc, out = cli("verify", PFc, ["--token", str(token or w.tokp), "--args-file", str(argf or w.argf), "--pkg", str(pkg), "--key-file", str(keyf), "--json", str(rec)] + (w.ctx if ctx is None else ctx))
        return rc, out, (json.loads(rec.read_text(encoding="utf-8")) if rec.is_file() else None)

    def verify_sub(w, ctx=None):
        return subprocess.run([sys.executable, str(HERE / "launch_gate.py"), "verify", "--profile", "refe_hnscreen", "--token", str(w.tokp), "--args-file", str(w.argf), "--pkg", str(pkg), "--key-file", str(keyf)]
                              + (w.ctx if ctx is None else ctx), capture_output=True, text=True, timeout=600, encoding="utf-8", errors="replace",
                              env=dict(os.environ, PYTHONIOENCODING="utf-8", CUDA_VISIBLE_DEVICES="-1"))

    def refusals_cli(w, ctx=None):
        return cli("refusals", PFc, ["--arm-name", "S1", "--args-file", str(w.argf), "--key-file", str(keyf)] + (w.ctx if ctx is None else ctx))

    def end_cli(w, seeds="1", out=None, ctx_dir=None):
        o = out or (root / f"end_w{w.n}_{time.time_ns()}.json")
        rc, txt = cli("end-verify", PFc, ["--ctx-cache", str(ctx_dir or w.dir), "--anchor", str(w.anchor_p), "--chain-id", w.chain, "--chain-start-ns", str(w.start), "--seeds", seeds, "--out", str(o),
                                         "--pkg", str(pkg), "--key-file", str(keyf)])
        return rc, txt, o

    def runchecks(w, prof_ctx=True):
        o = root / f"rc_w{w.n}_{time.time_ns()}"
        pr = {**profv, "ctx_anchor": w.anc} if prof_ctx else profv
        st = L.run_checks(pr, w.argv, pkg, o, commit=commit, only=("G-HYG",))
        return st, o

    def finalize_again(w):
        o = root / f"fin_w{w.n}_{time.time_ns()}"
        o.mkdir()
        profa = {**profv, "ctx_anchor": w.anc}
        binding = GT.binding_of(profa["name"], commit, GT.tree_digest(tree), w.argv)
        for c_ in checks_all:
            L.write_evidence(o, C.evidence(c_, "PASS", [], {}), binding, {})
        return L.finalize(profa, w.argv, pkg, o, key, commit=commit)

    # ---- the edits of a cache (each isolates ONE cause)
    def payload(d):
        return sorted((Path(d) / "units").glob("*.ctx.u16"))[0]

    def edit_keep_mtime(p, fn):
        st = os.stat(p)
        b = bytearray(Path(p).read_bytes())
        out = fn(b)
        Path(p).write_bytes(bytes(out if out is not None else b))
        os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))
        assert os.stat(p).st_mtime_ns == st.st_mtime_ns, "premise: this filesystem restores mtime_ns exactly"
        return st

    def flip_same_size_same_mtime(d):
        def f(b):
            b[len(b) // 2] ^= 0x01
        return edit_keep_mtime(payload(d), f)

    def grow_same_mtime(d):
        def f(b):
            return bytes(b) + b"\x00"
        return edit_keep_mtime(payload(d), f)

    def touch_only(d, delta_ns=2_000_000_000):
        p = payload(d)
        st = os.stat(p)
        os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + delta_ns))
        return st

    def add_extra(d):
        (Path(d) / "units" / "zz_extra.bin").write_bytes(b"x")

    def delete_one(d):
        sorted((Path(d) / "units").glob("*.fan.f32"))[0].unlink()

    def rename_one(d):
        p = sorted((Path(d) / "units").glob("*.fan.f32"))[0]
        p.rename(p.with_name("renamed.fan.f32"))

    def refused_everywhere(w, frag):
        """every INTERMEDIATE entry point refuses: verify (in process + the chain's subprocess), refusals, run_checks, finalize"""
        rc, out, rec = verify_cli(w)
        sp = verify_sub(w)
        rcr, outr = refusals_cli(w)
        st, o = runchecks(w)
        ev = json.loads((o / "evidence_G-HYG.json").read_text(encoding="utf-8"))
        v2, _p, t2 = finalize_again(w)
        txt = " | ".join(t2["reasons"])
        return (rc == 5 and "ZZGATE-REFUSEDZZ" in out and "data differs" in out and frag in out and rec["verdict"] == "REFUSED" and sp.returncode == 5 and "ZZGATE-REFUSEDZZ" in sp.stdout
                and rcr == 5 and "ZZGATE-REFUSEDZZ" in outr and frag in outr and set(st.values()) == {"FAIL"} and ev["inputs_read"]["--ctx-cache"]["kind"] == "dir-anchor"
                and ev["inputs_read"]["--ctx-cache"]["n_problems"] >= 1 and v2 == "FAIL" and "anchor" in txt and frag in txt)

    # ---- a chain root layout for the reader gate
    lay = [0]

    def layout(ws, *, seeds=None, end=True, end_seeds=None, tokens=True):
        """root/passes/s<seed>/lg/PASS_*.json, root/ctx_chain/anchor.json, root/ctx_chain/end_verify.json (written by the REAL end-verify command).  ws = one work (all seeds) or {seed: work}"""
        lay[0] += 1
        r = root / f"chain_{lay[0]}"
        ws = ws if isinstance(ws, dict) else {s: ws for s in (seeds or [1])}
        w0 = next(iter(ws.values()))
        if tokens:
            for s, wv in ws.items():
                (r / "passes" / f"s{s}" / "lg").mkdir(parents=True)
                shutil.copy(wv.tokp, r / "passes" / f"s{s}" / "lg" / "PASS_t.json")
        (r / "ctx_chain").mkdir(parents=True, exist_ok=True)
        shutil.copy(w0.anchor_p, r / "ctx_chain" / "anchor.json")
        if end:
            rc, _t, o = end_cli(w0, seeds=",".join(str(s) for s in (end_seeds if end_seeds is not None else ws)))
            shutil.copy(o, r / "ctx_chain" / "end_verify.json")
        return r

    def refusals_of(r, seeds):
        return GT.ctx_chain_refusals(r, seeds, key)

    # ============================================================================ R0 the clean control: anchor-bound calls MATCH, the key IS the registered one, the modes are recorded
    def r0_control():
        w = work()
        d = w.tok["binding"]["data"]["--ctx-cache"]
        rc, out, rec = verify_cli(w)
        sp = verify_sub(w)
        rcr, outr = refusals_cli(w)
        rce, txte, oe = end_cli(w)
        end = json.loads(Path(oe).read_text(encoding="utf-8"))
        st, o = runchecks(w)
        return (w.verdict == "PASS" and d["kind"] == "dir-anchor" and d["mode"] == "anchor-bound" and d["anchor_sha256"] == w.sha and GT.fp_key(d) == oracle_key(w.dir) == FX["--ctx-cache"]
                and w.tok["binding"]["data"]["--hn-bank"]["kind"] == "file" and rc == 0 and "ZZGATE-MATCHZZ" in out and rec["verdict"] == "MATCH" and rec["info"]["ctx_cache"]["mode"] == "anchor-bound"
                and rec["info"]["ctx_cache"]["anchor_sha256"] == w.sha and sp.returncode == 0 and "ZZGATE-MATCHZZ" in sp.stdout and rcr == 0 and "ZZGATE-CLEANZZ" in outr
                and rce == 0 and "ZZGATE-END_VERIFY_OKZZ" in txte and end["status"] == "END_VERIFY_OK" and end["content_equal"] is True and end["stat_equal"] is True
                and set(st.values()) == {"PASS"})
    chk("R0_clean_control_anchor_bound_calls_MATCH_the_key_is_the_registered_pin_and_the_mode_is_recorded", r0_control)

    def r0_anchor_content():
        w = work()
        a = w.doc
        return (a["content"]["key"] == FX["--ctx-cache"] and a["pin"] == FX["--ctx-cache"] and a["stat"]["n_files"] == 1 + 3 * 3 + 1 and CONTENT in a["stat"]["files"] and "manifest.json" in a["stat"]["files"]
                and GT.signature_ok(a, key) and a["cache_dir"] == GT.norm_dir(w.dir) and a["minted_ns"] >= w.start and a["chain_id"] == w.chain and GT.sha256_bytes(Path(w.anchor_p).read_bytes()) == w.sha)
    chk("R0_the_anchor_binds_the_registered_pin_every_file_incl_the_content_manifest_a_signature_and_the_chain", r0_anchor_content)

    # ============================================================================ R1 the residual: same size, same mtime_ns
    def r1():
        w = work()
        st = flip_same_size_same_mtime(w.dir)                                        # premise asserted inside: the mtime_ns is exactly the original
        rc, out, _ = verify_cli(w)
        sp = verify_sub(w)
        rcr, _o = refusals_cli(w)
        stt, _ = runchecks(w)
        residual = rc == 0 and sp.returncode == 0 and rcr == 0 and set(stt.values()) == {"PASS"}       # the intermediate calls cannot see it (this is WHY the end verify exists)
        rce, txte, oe = end_cli(w)
        rec = json.loads(Path(oe).read_text(encoding="utf-8"))
        r = layout(w, end=False)
        shutil.copy(oe, r / "ctx_chain" / "end_verify.json")
        why = refusals_of(r, [1])
        voided = (rce == 5 and "ZZGATE-VOIDZZ" in txte and rec["status"] == "VOID" and rec["content_equal"] is False and rec["stat_equal"] is True and any("BYTES CHANGED" in p for p in rec["problems"]))
        return residual and voided and bool(why) and "VOID" in why[0]
    chk("R1_same_size_same_mtime_edit_mid_chain_passes_every_intermediate_call_the_END_verify_voids_and_the_reader_gate_refuses", r1)

    def r1_clean_control():
        w = work()
        r = layout(w)
        return refusals_of(r, [1]) == []
    chk("R1_control_an_untouched_chain_ends_END_VERIFY_OK_and_the_reader_gate_admits_it", r1_clean_control)

    def r1_restore():
        w = work()
        p = payload(w.dir)
        before = p.read_bytes()
        flip_same_size_same_mtime(w.dir)
        st = os.stat(p)
        p.write_bytes(before)
        os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))
        rce, txte, oe = end_cli(w)
        return rce == 0 and json.loads(Path(oe).read_text(encoding="utf-8"))["status"] == "END_VERIFY_OK"
    chk("R1_control_the_end_verify_compares_bytes_a_restored_file_ends_OK", r1_restore)

    # ============================================================================ R2 a size change
    def r2():
        w = work()
        grow_same_mtime(w.dir)                                                        # ONLY the size moved (mtime_ns restored)
        ok_refused = refused_everywhere(w, "SIZE changed since the anchor")
        rce, _t, oe = end_cli(w)
        return ok_refused and rce == 5 and json.loads(Path(oe).read_text(encoding="utf-8"))["status"] == "VOID"
    chk("R2_a_size_change_refuses_every_intermediate_entry_point_and_voids_at_the_end", r2)

    # ============================================================================ R3 an mtime change, identical bytes
    def r3():
        w = work()
        st = touch_only(w.dir)
        refused = refused_everywhere(w, "MTIME changed since the anchor")
        p = payload(w.dir)
        os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))                              # restore the mtime: the same bytes MATCH again
        rc, out, _ = verify_cli(w)
        return refused and rc == 0 and "ZZGATE-MATCHZZ" in out
    chk("R3_an_mtime_ns_change_with_identical_bytes_refuses_and_restoring_it_matches_again", r3)

    def r3e():
        w = work()
        touch_only(w.dir)                                                             # after the last intermediate call: nothing but the END can see it
        rce, txte, oe = end_cli(w)
        rec = json.loads(Path(oe).read_text(encoding="utf-8"))
        return rce == 5 and rec["status"] == "VOID" and rec["content_equal"] is True and rec["stat_equal"] is False and any("MTIME changed" in p for p in rec["problems"])
    chk("R3e_an_mtime_change_after_the_last_intermediate_call_voids_at_the_end_though_the_bytes_are_equal", r3e)

    # ============================================================================ R4 an extra file, a missing file, a renamed file
    def r4_extra():
        w = work()
        add_extra(w.dir)
        return refused_everywhere(w, "file NOT in the anchor")
    chk("R4_an_extra_file_refuses_every_intermediate_entry_point", r4_extra)

    def r4_missing():
        w = work()
        delete_one(w.dir)
        return refused_everywhere(w, "file in the anchor is MISSING")
    chk("R4_a_missing_file_refuses_every_intermediate_entry_point", r4_missing)

    def r4_renamed():
        w = work()
        rename_one(w.dir)
        return refused_everywhere(w, "file NOT in the anchor") and "MISSING" in verify_cli(w)[1]
    chk("R4_a_renamed_file_is_both_a_missing_and_an_extra_file_and_refuses", r4_renamed)

    def r4_symlink():                                                                 # a symlinked directory is not walked, so its files are bound by nothing: a problem, never a silent skip
        w = work()
        real = os.path.islink
        try:
            os.path.islink = lambda p: os.path.basename(str(p)) == "units"
            fp = GT.anchor_fingerprint(str(w.dir), w.anc)
        finally:
            os.path.islink = real
        return fp["n_problems"] >= 1 and any("symlinked directory" in x for x in fp["problems"]) and GT.fp_key(fp).startswith("dir-anchor-INVALID:") and GT.fp_key(GT.anchor_fingerprint(str(w.dir), w.anc)) == FX["--ctx-cache"]
    chk("R4_a_symlinked_directory_is_a_problem_not_a_silent_skip", r4_symlink)

    # ============================================================================ R5 the end verify skipped / deleted / VOID / foreign / edited -> the reader gate refuses
    def r5_skipped():
        w = work()
        r = layout(w, end=False)                                                     # the chain never ran its end verify
        why = refusals_of(r, [1])
        return len(why) == 1 and "no END_VERIFY_OK record" in why[0]
    chk("R5_the_end_verify_skipped_the_reader_gate_refuses", r5_skipped)

    def r5_deleted():
        w = work()
        r = layout(w)
        ok0 = refusals_of(r, [1]) == []
        (r / "ctx_chain" / "end_verify.json").unlink()
        why = refusals_of(r, [1])
        return ok0 and len(why) == 1 and "no END_VERIFY_OK record" in why[0]
    chk("R5_the_end_record_deleted_after_it_was_written_the_reader_gate_refuses", r5_deleted)

    def r5_void_flag_edited():
        w = work()
        flip_same_size_same_mtime(w.dir)
        r = layout(w)                                                                 # a REAL VOID record
        p = r / "ctx_chain" / "end_verify.json"
        rec = json.loads(p.read_text(encoding="utf-8"))
        void_ok = rec["status"] == "VOID" and "VOID" in refusals_of(r, [1])[0]
        rec["status"] = "END_VERIFY_OK"                                               # hand-edited to pass: the HMAC no longer verifies
        p.write_text(json.dumps(rec, indent=1, sort_keys=True), encoding="utf-8")
        why = refusals_of(r, [1])
        return void_ok and len(why) == 1 and "not a signed" in why[0]
    chk("R5_a_VOID_record_refuses_and_a_hand_edited_one_is_refused_by_its_signature", r5_void_flag_edited)

    def r5_other_anchor():
        w = work()
        r = layout(w)
        p = r / "ctx_chain" / "end_verify.json"
        rec = json.loads(p.read_text(encoding="utf-8"))
        rec.pop("hmac_sha256")
        rec["anchor_sha256"] = "0" * 64                                               # a perfectly valid, correctly SIGNED record -- of another anchor
        rec["hmac_sha256"] = GT.sign(rec, key)
        p.write_text(json.dumps(rec, indent=1, sort_keys=True), encoding="utf-8")
        why = refusals_of(r, [1])
        return len(why) >= 1 and "another anchor" in why[0]
    chk("R5_an_END_VERIFY_OK_signed_for_another_anchor_is_refused", r5_other_anchor)

    def r5_fewer_seeds():
        w = work()
        r = layout(w, seeds=[1, 2], end_seeds=[1])
        why = refusals_of(r, [1, 2])
        return len(why) == 1 and "covers seeds" in why[0] and refusals_of(r, [1]) == []
    chk("R5_an_END_record_that_does_not_cover_every_seed_read_is_refused", r5_fewer_seeds)

    def r5_two_anchors():
        w1, w2 = work(), work()
        r = layout({1: w1, 2: w2})
        why = refusals_of(r, [1, 2])
        return len(why) == 1 and "different anchors" in why[0]
    chk("R5_tokens_bound_to_two_different_anchors_are_refused", r5_two_anchors)

    def r5_no_token():
        w = work()
        r = layout(w, seeds=[1])
        why = refusals_of(r, [1, 2])
        return len(why) == 1 and "seed 2: no PASS token" in why[0]
    chk("R5_a_pass_without_a_token_is_refused", r5_no_token)

    def r5_listing_token():
        w = work()
        r = layout(w)
        tp = r / "passes" / "s1" / "lg" / "PASS_t.json"
        tok = json.loads(tp.read_text(encoding="utf-8"))
        tok.pop("hmac_sha256")
        tok["binding"]["data"]["--ctx-cache"] = {"kind": "dir", "n_files": 10, "bytes": 1, "listing_sha256": "0" * 64, "path": "x"}      # a size-listing token: the defect hnfp2 closed
        tok["hmac_sha256"] = GT.sign(tok, key)
        tp.write_text(json.dumps(tok, indent=1, sort_keys=True), encoding="utf-8")
        why = refusals_of(r, [1])
        return len(why) == 1 and "'dir'" in why[0]
    chk("R5_a_token_that_binds_only_a_size_listing_is_refused", r5_listing_token)

    def r5_content_tokens_need_no_end_record():                                      # R7's reader side: a content-mode chain (every call re-read the cache) is admitted without an anchor
        cd = root / "content_chain"
        w = work()
        profc = profv                                                                  # no anchor: content mode
        o = root / "lg_content"
        o.mkdir()
        binding = GT.binding_of(profc["name"], commit, GT.tree_digest(tree), w.argv)
        for c_ in checks_all:
            L.write_evidence(o, C.evidence(c_, "PASS", [], {}), binding, {})
        v, tokp, tok = L.finalize(profc, w.argv, pkg, o, key, commit=commit)
        (cd / "passes" / "s1" / "lg").mkdir(parents=True)
        shutil.copy(tokp, cd / "passes" / "s1" / "lg" / "PASS_t.json")
        return v == "PASS" and tok["binding"]["data"]["--ctx-cache"]["kind"] == "dir-content" and refusals_of(cd, [1]) == [] and refusals_of(cd, []) != []
    chk("R5_a_content_bound_chain_needs_no_end_record_and_an_empty_seed_list_is_refused", r5_content_tokens_need_no_end_record)

    def r5_unsigned_token():
        w = work()
        r = layout(w)
        tp = r / "passes" / "s1" / "lg" / "PASS_t.json"
        tok = json.loads(tp.read_text(encoding="utf-8"))
        tok["verdict"] = "PASS"
        tok["binding"]["data"]["--ctx-cache"]["n_problems"] = 0
        tok["created_utc"] = "1999-01-01T00:00:00Z"                                    # any edit without the key
        tp.write_text(json.dumps(tok, indent=1, sort_keys=True), encoding="utf-8")
        why = refusals_of(r, [1])
        return len(why) == 1 and "not a signed PASS" in why[0]
    chk("R5_an_edited_token_is_refused_by_its_signature", r5_unsigned_token)

    # ============================================================================ R6 the anchor's own binding
    def r6_other_dir():
        w1, w2 = work(), work()
        ctx = list(w1.ctx)                                                             # w1's anchor offered for w2's cache
        rc, out, _ = verify_cli(w2, ctx=ctx)
        sp = verify_sub(w2, ctx=ctx)
        rcr, outr = refusals_cli(w2, ctx=ctx)
        rce, txte, oe = end_cli(w1, ctx_dir=w2.dir)
        rcm, outm, _ = verify_cli(w2)
        return (rc == 5 and "another cache directory" in out and sp.returncode == 5 and rcr == 5 and "another cache directory" in outr and rce == 5 and not Path(oe).exists()
                and "ZZGATE-END_VERIFY-REFUSEDZZ" in txte and rcm == 0)
    chk("R6_an_anchor_minted_from_another_cache_directory_is_refused_by_every_entry_point", r6_other_dir)

    def r6_older_than_the_chain():
        w = work()
        late = ["--ctx-mode", "anchor-bound", "--anchor", str(w.anchor_p), "--chain-id", w.chain, "--chain-start-ns", str(w.doc["minted_ns"] + 1)]
        rc, out, _ = verify_cli(w, ctx=late)
        sp = verify_sub(w, ctx=late)
        rcr, outr = refusals_cli(w, ctx=late)
        rce, txte, oe = cli("end-verify", PFc, ["--ctx-cache", str(w.dir), "--anchor", str(w.anchor_p), "--chain-id", w.chain, "--chain-start-ns", str(w.doc["minted_ns"] + 1), "--seeds", "1",
                                                "--out", str(root / "late_end.json"), "--pkg", str(pkg), "--key-file", str(keyf)]) + (None,)
        on_time = ["--ctx-mode", "anchor-bound", "--anchor", str(w.anchor_p), "--chain-id", w.chain, "--chain-start-ns", str(w.doc["minted_ns"])]
        rcok, _o, _ = verify_cli(w, ctx=on_time)
        return (rc == 5 and "OLDER than the chain" in out and sp.returncode == 5 and rcr == 5 and "OLDER than the chain" in outr and rce == 5 and "OLDER than the chain" in txte
                and not (root / "late_end.json").exists() and rcok == 0)
    chk("R6_an_anchor_minted_before_the_chain_started_is_refused_by_every_entry_point", r6_older_than_the_chain)

    def r6_other_chain():
        w = work()
        other = ["--ctx-mode", "anchor-bound", "--anchor", str(w.anchor_p), "--chain-id", "cd" * 16, "--chain-start-ns", str(w.start)]
        rc, out, _ = verify_cli(w, ctx=other)
        rcr, outr = refusals_cli(w, ctx=other)
        return rc == 5 and "another chain" in out and rcr == 5 and "another chain" in outr
    chk("R6_an_anchor_of_another_chain_is_refused", r6_other_chain)

    def r6_edited_and_foreign_key():
        w = work()
        doc = json.loads(Path(w.anchor_p).read_text(encoding="utf-8"))
        edited = root / f"anchor_edited_{w.n}.json"
        doc["stat"]["files"]["units/log0.ctx.u16"][1] += 1                             # the anchor's mtime edited so a re-timestamped file would MATCH: the signature refuses
        edited.write_text(json.dumps(doc, indent=1, sort_keys=True), encoding="utf-8")
        ctx = ["--ctx-mode", "anchor-bound", "--anchor", str(edited), "--chain-id", w.chain, "--chain-start-ns", str(w.start)]
        rc, out, _ = verify_cli(w, ctx=ctx)
        key2 = GT.load_key(root / "other.key", create=True, forbid_under=[pkg])
        doc2 = json.loads(Path(w.anchor_p).read_text(encoding="utf-8"))
        doc2.pop("hmac_sha256")
        doc2["hmac_sha256"] = GT.sign(doc2, key2)
        foreign = root / f"anchor_foreign_{w.n}.json"
        foreign.write_text(json.dumps(doc2, indent=1, sort_keys=True), encoding="utf-8")
        ctx2 = ["--ctx-mode", "anchor-bound", "--anchor", str(foreign), "--chain-id", w.chain, "--chain-start-ns", str(w.start)]
        rc2, out2, _ = verify_cli(w, ctx=ctx2)
        rcr, outr = refusals_cli(w, ctx=ctx)
        return rc == 5 and "HMAC does not verify" in out and rc2 == 5 and "HMAC does not verify" in out2 and rcr == 5
    chk("R6_an_edited_anchor_and_an_anchor_signed_with_another_key_are_refused", r6_edited_and_foreign_key)

    def r6_mint_refusals():
        base_args = lambda d, o: ["--ctx-cache", str(d), "--chain-id", "ef" * 16, "--out", str(o), "--pkg", str(pkg), "--key-file", str(keyf)]    # noqa: E731
        d1 = root / "m_ok"
        shutil.copytree(pristine, d1)
        o1 = root / "m_ok.json"
        rc_ok, out_ok = cli("anchor", PFc, base_args(d1, o1))
        d2 = root / "m_flip"
        shutil.copytree(pristine, d2)
        flip_same_size_same_mtime(d2)
        o2 = root / "m_flip.json"
        rc_flip, out_flip = cli("anchor", PFc, base_args(d2, o2))
        o3 = root / "m_wrongpin.json"
        rc_pin, out_pin = cli("anchor", {**P0, "fingerprints_expected": {**FX, "--ctx-cache": "dirc:10:6012:" + "0" * 64}}, base_args(d1, o3))
        o4 = root / "m_unfilled.json"
        rc_un, out_un = cli("anchor", P0, base_args(d1, o4))
        o5 = root / "m_listingpin.json"
        rc_ls, out_ls = cli("anchor", {**P0, "fingerprints_expected": {**FX, "--ctx-cache": GT.fp_key(GT.fingerprint(str(pristine)))}}, base_args(d1, o5))
        o6 = root / "m_other.json"
        rc_o, out_o = cli("anchor", PFc, ["--ctx-cache", str(root / "does_not_exist"), "--chain-id", "ef" * 16, "--out", str(o6), "--pkg", str(pkg), "--key-file", str(keyf)])
        o7 = root / "m_badid.json"
        rc_id, out_id = cli("anchor", PFc, ["--ctx-cache", str(d1), "--chain-id", "not hex", "--out", str(o7), "--pkg", str(pkg), "--key-file", str(keyf)])
        return (rc_ok == 0 and o1.is_file() and rc_flip == 5 and "BYTES CHANGED" in out_flip and not o2.exists() and rc_pin == 5 and "not the registered pin" in out_pin and not o3.exists()
                and rc_un == 5 and "no registered content pin" in out_un and not o4.exists() and rc_ls == 5 and "no registered content pin" in out_ls and not o5.exists()
                and rc_o == 5 and not o6.exists() and rc_id == 5 and not o7.exists() and all("ZZGATE-ANCHOR-REFUSEDZZ" in x for x in (out_flip, out_pin, out_un, out_ls, out_o, out_id)))
    chk("R6_the_anchor_is_minted_only_for_the_registered_sealed_untampered_cache", r6_mint_refusals)

    def r6_pin_compare_only():                                                        # isolates the pin comparison: a perfectly valid, sealed cache that is simply NOT the registered one
        d = root / "m_resealed"
        shutil.copytree(pristine, d)
        flip_same_size_same_mtime(d)
        GT.seal_content_manifest(d)                                                    # self-consistent: only the registered pin can refuse it
        o = root / "m_resealed.json"
        rc, out = cli("anchor", PFc, ["--ctx-cache", str(d), "--chain-id", "ef" * 16, "--out", str(o), "--pkg", str(pkg), "--key-file", str(keyf)])
        return rc == 5 and "not the registered pin" in out and not o.exists()
    chk("R6_a_resealed_self_consistent_cache_is_refused_by_the_pin_alone", r6_pin_compare_only)

    def r6_edited_while_anchoring():
        d = root / "m_race"
        shutil.copytree(pristine, d)
        real = GT.content_fingerprint

        def racy(path, **k):
            fp = real(path, **k)
            p = payload(d)
            st = os.stat(p)
            os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))        # a file touched while the cache is being anchored
            return fp
        o = root / "m_race.json"
        GT.content_fingerprint = racy
        try:
            try:
                GT.mint_anchor(d, chain_id="ef" * 16, key=key, pin=FX["--ctx-cache"], out_path=o)
                return False
            except GT.GateError as e:
                return "changed while it was being anchored" in str(e) and not o.exists()
        finally:
            GT.content_fingerprint = real
    chk("R6_a_cache_edited_while_it_is_being_anchored_is_not_anchored", r6_edited_while_anchoring)

    # ============================================================================ R7 content mode is today's behaviour; the cost is pinned
    def r7_identity():
        w = work()
        av = ["--ctx-cache", str(w.dir), "--hn-bank", str(bank)]
        flags = ("--ctx-cache", "--hn-bank")
        old = GT.data_manifest(av, flags, content_flags=("--ctx-cache",))
        new = GT.data_manifest(av, flags, content_flags=("--ctx-cache",), ctx_anchor=None)
        by_hand = {"--ctx-cache": {"path": str(w.dir), **GT.content_fingerprint(str(w.dir))}, "--hn-bank": {"path": str(bank), **GT.fingerprint(str(bank))}}
        listing = GT.data_manifest(av, flags)
        return (json.dumps(old, sort_keys=True) == json.dumps(new, sort_keys=True) == json.dumps(by_hand, sort_keys=True) and old["--ctx-cache"]["kind"] == "dir-content"
                and listing["--ctx-cache"]["kind"] == "dir" and GT.fp_key(old["--ctx-cache"]) == oracle_key(w.dir))
    chk("R7_identity_content_mode_data_manifest_is_exactly_the_call_it_always_was", r7_identity)

    def r7_cli_default_is_content():
        w = work()
        # a content-mode token over the same bytes (no anchor in the profile)
        o = root / f"lg_c{w.n}"
        o.mkdir()
        binding = GT.binding_of(profv["name"], commit, GT.tree_digest(tree), w.argv)
        for c_ in checks_all:
            L.write_evidence(o, C.evidence(c_, "PASS", [], {}), binding, {})
        v, tokp, tok = L.finalize(profv, w.argv, pkg, o, key, commit=commit)
        kinds = (v, tok["binding"]["data"]["--ctx-cache"]["kind"], "ctx_cache" in str(tok["binding"]["data"]["--ctx-cache"].get("mode", "")))
        a1 = verify_cli(w, ctx=[], token=tokp)                                         # no ctx flag at all
        a2 = verify_cli(w, ctx=["--ctx-mode", "content"], token=tokp)
        same = a1[0] == a2[0] == 0 and a1[1] == a2[1] and "ctx_cache" not in a1[2]["info"] and "ctx_cache" not in a2[2]["info"]
        flip_same_size_same_mtime(w.dir)                                               # the DEFAULT (no flag) re-reads every byte: a same-size, same-mtime flip is REFUSED
        b1 = verify_cli(w, ctx=[], token=tokp)
        b2 = verify_sub(w, ctx=[])
        return kinds == ("PASS", "dir-content", False) and same and b1[0] == 5 and "BYTES CHANGED at the same size" in b1[1] and b2.returncode == 5
    chk("R7_default_mode_is_content_it_matches_the_clean_cache_and_refuses_the_same_size_same_mtime_flip", r7_cli_default_is_content)

    def r7_anchor_under_content_refused():
        w = work()
        a = verify_cli(w, ctx=["--anchor", str(w.anchor_p)])
        b = verify_cli(w, ctx=["--ctx-mode", "content", "--chain-id", w.chain])
        c = refusals_cli(w, ctx=["--anchor", str(w.anchor_p)])
        return a[0] == 5 and "only used in anchor-bound mode" in a[1] and b[0] == 5 and c[0] == 5 and "ZZGATE-REFUSEDZZ" in c[1]
    chk("R7_an_anchor_named_under_content_mode_is_refused_never_silently_ignored", r7_anchor_under_content_refused)

    def r7_anchor_incomplete_or_other_profile():
        w = work()
        a = verify_cli(w, ctx=["--ctx-mode", "anchor-bound", "--anchor", str(w.anchor_p)])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            rc = L.main(["verify", "--profile", "refe_rls1", "--token", str(w.tokp), "--args-file", str(w.argf), "--pkg", str(pkg), "--key-file", str(keyf)] + w.ctx)
        return a[0] == 5 and "needs --anchor, --chain-id and --chain-start-ns" in a[1] and rc == 5 and "defined only for a profile" in buf.getvalue()
    chk("R7_anchor_bound_needs_all_three_flags_and_exists_only_for_the_hn_screen_profile", r7_anchor_incomplete_or_other_profile)

    def r7_refusal_text_unchanged():
        d = root / "t_flip"
        shutil.copytree(pristine, d)
        flip_same_size_same_mtime(d)
        w = W()
        w.n, w.dir, w.argv = 0, d, argv_for(d)
        w.argf = root / "argv_t.json"
        GT.write_json(w.argf, w.argv)
        rc, out = cli("refusals", PFc, ["--arm-name", "S1", "--args-file", str(w.argf)])
        return rc == 5 and "the content fingerprint of --ctx-cache cannot be taken: 1 problem(s): BYTES CHANGED at the same size" in out and "anchor" not in out
    chk("R7_the_content_mode_refusal_text_is_exactly_the_one_it_always_was", r7_refusal_text_unchanged)

    def r7_cost():
        counts = []
        real = GT._sha256_and_size
        n_files = len(json.loads((pristine / CONTENT).read_text(encoding="utf-8"))["files"])

        def counting(path, *a, **k):
            counts.append(os.path.abspath(str(path)))
            return real(path, *a, **k)

        w = work()
        wc = work()
        prefix = os.path.abspath(str(w.dir)) + os.sep

        def passes(fn):
            counts.clear()
            fn()
            return len([c for c in counts if c.startswith(prefix)]) / n_files
        GT._sha256_and_size = counting
        try:
            anc = {"refusals_cli": passes(lambda: refusals_cli(w)), "verify_cli": passes(lambda: verify_cli(w)), "run_checks": passes(lambda: runchecks(w)),
                   "finalize": passes(lambda: finalize_again(w)), "verify_sub_not_counted_here": 0.0}
            runc = root / "cmd_run_a"
            ra = passes(lambda: cli("run", PFc, ["--arm-name", "S1", "--args-file", str(w.argf), "--pkg", str(pkg), "--out-dir", str(runc), "--only", "G-HYG", "--commit", commit, "--key-file", str(keyf)] + w.ctx))
            cprefix = os.path.abspath(str(wc.dir)) + os.sep
            counts.clear()
            cli("run", PFc, ["--arm-name", "S1", "--args-file", str(wc.argf), "--pkg", str(pkg), "--out-dir", str(root / "cmd_run_c"), "--only", "G-HYG", "--commit", commit, "--key-file", str(keyf)])
            rc_content = len([c for c in counts if c.startswith(cprefix)]) / n_files
            counts.clear()
            end_cli(w)
            end_reads = len([c for c in counts if c.startswith(prefix)]) / n_files
            counts.clear()
            cli("anchor", PFc, ["--ctx-cache", str(w.dir), "--chain-id", "ab" * 16, "--out", str(root / "again.json"), "--pkg", str(pkg), "--key-file", str(keyf)])
            anchor_reads = len([c for c in counts if c.startswith(prefix)]) / n_files
        finally:
            GT._sha256_and_size = real
        return (anc["refusals_cli"] == anc["verify_cli"] == anc["run_checks"] == anc["finalize"] == 0.0 and ra == 0.0 and rc_content == 4.0 and end_reads == 1.0 and anchor_reads == 1.0)
    chk("R7_cost_anchor_bound_intermediate_calls_read_ZERO_bytes_content_run_is_4_reads_anchor_1_end_verify_1", r7_cost)

    # ============================================================================ W wiring: the CLI commands hand THIS call's anchor to the functions that use it
    def w_cmd_run():
        w = work()
        seen = {}
        real_rc, real_fin = L.run_checks, L.finalize

        def spy_rc(prof, *a, **k):
            seen["run_checks"] = (prof.get("ctx_anchor") or {}).get("sha256")
            return {}

        def spy_fin(prof, *a, **k):
            seen["finalize"] = (prof.get("ctx_anchor") or {}).get("sha256")
            return "PASS", Path("x"), {"reasons": [], "binding": {"commit": "c" * 12, "tree_sha256": "t" * 12}}
        L.run_checks, L.finalize = spy_rc, spy_fin
        try:
            cli("run", PFc, ["--arm-name", "S1", "--args-file", str(w.argf), "--pkg", str(pkg), "--out-dir", str(root / "wr_a"), "--commit", commit, "--key-file", str(keyf)] + w.ctx)
            a = dict(seen)
            seen.clear()
            cli("run", PFc, ["--arm-name", "S1", "--args-file", str(w.argf), "--pkg", str(pkg), "--out-dir", str(root / "wr_c"), "--commit", commit, "--key-file", str(keyf)])
            c = dict(seen)
            seen.clear()
            rc, out = cli("run", PFc, ["--arm-name", "S1", "--args-file", str(w.argf), "--pkg", str(pkg), "--out-dir", str(root / "wr_x"), "--commit", commit, "--key-file", str(keyf),
                                       "--ctx-mode", "anchor-bound", "--anchor", str(w.anchor_p), "--chain-id", "00" * 16, "--chain-start-ns", str(w.start)])
            refused = (rc, "ZZGATE-REFUSEDZZ" in out, dict(seen))
        finally:
            L.run_checks, L.finalize = real_rc, real_fin
        return a == {"run_checks": w.sha, "finalize": w.sha} and c == {"run_checks": None, "finalize": None} and refused == (5, True, {})
    chk("W_cmd_run_hands_the_anchor_to_run_checks_and_finalize_and_content_mode_hands_none_and_a_bad_anchor_runs_nothing", w_cmd_run)

    def w_cmd_verify_and_refusals():
        w = work()
        seen = {}
        real_vt, real_pr = GT.verify_token, L.profile_refusals

        def spy_vt(*a, **k):
            seen["verify_token"] = (k.get("ctx_anchor") or {}).get("sha256")
            return True, [], {}

        def spy_pr(prof, argv):
            seen["profile_refusals"] = (prof.get("ctx_anchor") or {}).get("sha256")
            return []
        GT.verify_token, L.profile_refusals = spy_vt, spy_pr
        try:
            verify_cli(w)
            refusals_cli(w)
            a = dict(seen)
            seen.clear()
            verify_cli(w, ctx=[])
            refusals_cli(w, ctx=[])
            c = dict(seen)
        finally:
            GT.verify_token, L.profile_refusals = real_vt, real_pr
        return a == {"verify_token": w.sha, "profile_refusals": w.sha} and c == {"verify_token": None, "profile_refusals": None}
    chk("W_cmd_verify_and_cmd_refusals_hand_the_anchor_to_verify_token_and_profile_refusals_and_content_mode_hands_none", w_cmd_verify_and_refusals)

    def w_hnscreen_refusals_uses_the_anchor():
        w = work()
        calls = []
        real = GT.content_fingerprint

        def spy(p, **k):
            calls.append(p)
            return real(p, **k)
        GT.content_fingerprint = spy
        try:
            rs = L.profile_refusals({**profv, "ctx_anchor": w.anc}, w.argv)
            n_anchor = len([c for c in calls if str(w.dir) in str(c)])
            calls.clear()
            rs2 = L.profile_refusals(profv, w.argv)
            n_content = len([c for c in calls if str(w.dir) in str(c)])
        finally:
            GT.content_fingerprint = real
        return rs == [] and rs2 == [] and n_anchor == 0 and n_content == 1
    chk("W_hnscreen_refusals_stat_compares_under_an_anchor_and_re_reads_in_full_without_one", w_hnscreen_refusals_uses_the_anchor)

    shutil.rmtree(root, ignore_errors=True)
    return bad, n_checks[0]


# ================================================================================================== M: source mutations (each snippet must occur EXACTLY once; each deletes the line that enforces a refusal)
SIZE_LINE = "    out += [f\"SIZE changed since the anchor: {r} (anchor {want[r][0]} bytes, now {got[r][0]})\" for r in sorted(set(want) & set(got)) if want[r][0] != got[r][0]]\n"
MTIME_LINE = ("    out += [f\"MTIME changed since the anchor: {r} (anchor {want[r][1]} ns, now {got[r][1]} ns): the bytes are not re-read at an intermediate call, so a re-timestamped file is refused as a possible edit\"\n"
              "            for r in sorted(set(want) & set(got)) if want[r][1] != got[r][1]]\n")
EXTRA_LINE = "    out += [f\"file NOT in the anchor (extra, or a renamed file): {r}\" for r in sorted(set(got) - set(want))]\n"
MISSING_LINE = "    out = [f\"file in the anchor is MISSING: {r}\" for r in sorted(set(want) - set(got))]\n"
R4 = ["R4_an_extra_file_refuses_every_intermediate_entry_point", "R4_a_renamed_file_is_both_a_missing_and_an_extra_file_and_refuses"]
MUTATIONS = [
    ("anchor_listing_problems: the SIZE comparison is deleted", "gate_token.py", [(SIZE_LINE, "")], ["R2_a_size_change_refuses_every_intermediate_entry_point_and_voids_at_the_end"]),
    ("anchor_listing_problems: the MTIME comparison is deleted", "gate_token.py", [(MTIME_LINE, "")],
     ["R3_an_mtime_ns_change_with_identical_bytes_refuses_and_restoring_it_matches_again", "R3e_an_mtime_change_after_the_last_intermediate_call_voids_at_the_end_though_the_bytes_are_equal"]),
    ("anchor_listing_problems: an EXTRA file is not reported", "gate_token.py", [(EXTRA_LINE, "")], R4),
    ("anchor_listing_problems: a MISSING file is not reported", "gate_token.py", [(MISSING_LINE, "    out = []\n")], ["R4_a_missing_file_refuses_every_intermediate_entry_point", "R4_a_renamed_file_is_both_a_missing_and_an_extra_file_and_refuses"]),
    ("_stat_listing: a symlinked directory is silently skipped", "gate_token.py",
     [("        problems += [f\"a symlinked directory is not followed, so its files cannot be bound: {os.path.relpath(os.path.join(dirpath, d), rootp).replace(os.sep, '/')}\"\n"
       "                     for d in dirnames if os.path.islink(os.path.join(dirpath, d))]\n        for fn in sorted(filenames):\n            full = os.path.join(dirpath, fn)\n            rel = os.path.relpath(full, rootp).replace(os.sep, \"/\")\n            try:\n                st = os.stat(full)",
       "        for fn in sorted(filenames):\n            full = os.path.join(dirpath, fn)\n            rel = os.path.relpath(full, rootp).replace(os.sep, \"/\")\n            try:\n                st = os.stat(full)")],
     ["R4_a_symlinked_directory_is_a_problem_not_a_silent_skip"]),
    ("end_verify: the CONTENT comparison with the anchor is deleted (the end verify can never void on bytes)", "gate_token.py",
     [("    content_equal = k_end == doc[\"content\"][\"key\"]\n", "    content_equal = True\n")],
     ["R1_same_size_same_mtime_edit_mid_chain_passes_every_intermediate_call_the_END_verify_voids_and_the_reader_gate_refuses"]),
    ("end_verify: the final STAT comparison is deleted", "gate_token.py", [("    stat_equal = not stat_problems and not p1 and not p2\n", "    stat_equal = True\n")],
     ["R3e_an_mtime_change_after_the_last_intermediate_call_voids_at_the_end_though_the_bytes_are_equal"]),
    ("end_verify: the record is always END_VERIFY_OK", "gate_token.py", [("\"status\": END_OK if not problems else END_VOID", "\"status\": END_OK")],
     ["R1_same_size_same_mtime_edit_mid_chain_passes_every_intermediate_call_the_END_verify_voids_and_the_reader_gate_refuses", "R2_a_size_change_refuses_every_intermediate_entry_point_and_voids_at_the_end"]),
    ("anchor_binding_refusals: the cache-directory comparison is deleted", "gate_token.py",
     [("    if norm_dir(cache_dir) != doc.get(\"cache_dir\"):\n", "    if False:\n")], ["R6_an_anchor_minted_from_another_cache_directory_is_refused_by_every_entry_point"]),
    ("anchor_binding_refusals: the chain-id comparison is deleted", "gate_token.py", [("    if chain_id != doc.get(\"chain_id\"):\n", "    if False:\n")], ["R6_an_anchor_of_another_chain_is_refused"]),
    ("anchor_binding_refusals: the 'older than the chain' comparison is deleted", "gate_token.py",
     [("    if not isinstance(chain_start_ns, int) or doc[\"minted_ns\"] < chain_start_ns:\n", "    if not isinstance(chain_start_ns, int):\n")],
     ["R6_an_anchor_minted_before_the_chain_started_is_refused_by_every_entry_point"]),
    ("load_anchor: the signature check is deleted", "gate_token.py", [("    if not signature_ok(doc, key):\n        return None, [f\"the anchor's HMAC", "    if False:\n        return None, [f\"the anchor's HMAC")],
     ["R6_an_edited_anchor_and_an_anchor_signed_with_another_key_are_refused"]),
    ("mint_anchor: the comparison with the registered pin is deleted", "gate_token.py", [("    if k != pin:\n", "    if False:\n")],
     ["R6_a_resealed_self_consistent_cache_is_refused_by_the_pin_alone", "R6_the_anchor_is_minted_only_for_the_registered_sealed_untampered_cache"]),
    ("mint_anchor: the 'changed while being anchored' comparison is deleted", "gate_token.py", [("    if before != after:\n        raise GateError(\"cannot anchor: the cache changed", "    if False:\n        raise GateError(\"cannot anchor: the cache changed")],
     ["R6_a_cache_edited_while_it_is_being_anchored_is_not_anchored"]),
    ("mint_anchor: the content check of the cache at the start is ignored", "gate_token.py", [("    if fp.get(\"n_problems\"):\n        raise GateError(f\"cannot anchor: the cache FAILS", "    if False:\n        raise GateError(f\"cannot anchor: the cache FAILS")],
     ["R6_the_anchor_is_minted_only_for_the_registered_sealed_untampered_cache"]),
    ("ctx_chain_refusals: a missing END record is admitted", "gate_token.py",
     [("        return [f\"no END_VERIFY_OK record ({END_REL}): the chain's end verify did not run, or its record was deleted -- the screen's outputs are not read\"]\n", "        return []\n")],
     ["R5_the_end_verify_skipped_the_reader_gate_refuses", "R5_the_end_record_deleted_after_it_was_written_the_reader_gate_refuses"]),
    ("ctx_chain_refusals: a VOID record is admitted", "gate_token.py", [("    if rec.get(\"status\") != END_OK:\n", "    if False:\n")],
     ["R1_same_size_same_mtime_edit_mid_chain_passes_every_intermediate_call_the_END_verify_voids_and_the_reader_gate_refuses", "R5_a_VOID_record_refuses_and_a_hand_edited_one_is_refused_by_its_signature"]),
    ("ctx_chain_refusals: the END record's signature is not checked", "gate_token.py", [("or not signature_ok(rec, key):\n        return [f\"{END_REL} is not a signed", "or False:\n        return [f\"{END_REL} is not a signed")],
     ["R5_a_VOID_record_refuses_and_a_hand_edited_one_is_refused_by_its_signature"]),
    ("ctx_chain_refusals: the binding of the END record to THE SAME anchor is deleted", "gate_token.py",
     [("    if rec.get(\"anchor_sha256\") != asha or rec.get(\"chain_id\") != adoc[\"chain_id\"]:\n", "    if False:\n")], ["R5_an_END_VERIFY_OK_signed_for_another_anchor_is_refused"]),
    ("ctx_chain_refusals: the seed-coverage comparison is deleted", "gate_token.py", [("    if not set(seeds) <= {int(x) for x in (rec.get(\"seeds\") or [])}:\n", "    if False:\n")],
     ["R5_an_END_record_that_does_not_cover_every_seed_read_is_refused"]),
    ("ctx_chain_refusals: a pass without a token is skipped silently", "gate_token.py",
     [("            out.append(f\"seed {s}: no PASS token under {root / 'passes' / f's{s}' / 'lg'}: the cache binding of this pass cannot be established\")\n            continue\n", "            continue\n")],
     ["R5_a_pass_without_a_token_is_refused"]),
    ("ctx_chain_refusals: the single-anchor requirement is deleted", "gate_token.py", [("    if len(shas) != 1:\n", "    if False:\n")], ["R5_tokens_bound_to_two_different_anchors_are_refused"]),
    ("ctx_chain_refusals: a listing-only token is admitted as a content-bound one", "gate_token.py",
     [("        if kind == \"dir-content\" and not d.get(\"n_problems\") and d.get(\"content_sha256\"):\n", "        if kind in (\"dir-content\", \"dir\"):\n")], ["R5_a_token_that_binds_only_a_size_listing_is_refused"]),
    ("ctx_chain_refusals: the token signature is not checked", "gate_token.py", [("        if not signature_ok(tok, key) or tok.get(\"verdict\") != \"PASS\" or tok.get(\"kind\") != \"launch-gate-token\":\n", "        if tok.get(\"verdict\") != \"PASS\" or tok.get(\"kind\") != \"launch-gate-token\":\n")],
     ["R5_an_edited_token_is_refused_by_its_signature"]),
    ("launch_gate.cmd_run: the anchor of the call is not applied", "launch_gate.py", [("        prof = _with_ctx_anchor(a, prof, argv, key)                           # HNSCREEN-FP-COST: content mode (the default) returns the profile untouched\n", "        pass\n")],
     ["W_cmd_run_hands_the_anchor_to_run_checks_and_finalize_and_content_mode_hands_none_and_a_bad_anchor_runs_nothing",
      "R7_cost_anchor_bound_intermediate_calls_read_ZERO_bytes_content_run_is_4_reads_anchor_1_end_verify_1"]),
    ("launch_gate.cmd_verify: the anchor is not handed to verify_token", "launch_gate.py", [("content_flags=content_flags_of(prof) if prof else (), ctx_anchor=pa.get(\"ctx_anchor\"))", "content_flags=content_flags_of(prof) if prof else ())")],
     ["W_cmd_verify_and_cmd_refusals_hand_the_anchor_to_verify_token_and_profile_refusals_and_content_mode_hands_none", "R0_clean_control_anchor_bound_calls_MATCH_the_key_is_the_registered_pin_and_the_mode_is_recorded"]),
    ("launch_gate.finalize: the anchor is not handed to the data manifest", "launch_gate.py",
     [("content_flags=content_flags_of(prof), ctx_anchor=prof.get(\"ctx_anchor\"))\n    _log_manifest_timing(\"finalize\"", "content_flags=content_flags_of(prof))\n    _log_manifest_timing(\"finalize\"")],
     ["R7_cost_anchor_bound_intermediate_calls_read_ZERO_bytes_content_run_is_4_reads_anchor_1_end_verify_1", "R0_clean_control_anchor_bound_calls_MATCH_the_key_is_the_registered_pin_and_the_mode_is_recorded"]),
    ("launch_gate.run_checks: the anchor is not handed to the data manifest", "launch_gate.py",
     [("content_flags=content_flags_of(prof), ctx_anchor=prof.get(\"ctx_anchor\"))\n    _log_manifest_timing(\"run\"", "content_flags=content_flags_of(prof))\n    _log_manifest_timing(\"run\"")],
     ["R7_cost_anchor_bound_intermediate_calls_read_ZERO_bytes_content_run_is_4_reads_anchor_1_end_verify_1"]),
    ("launch_gate.hnscreen_refusals: the anchor is ignored (the cache is re-read in full)", "launch_gate.py",
     [("(GT.anchor_fingerprint(vals[0][0], anc) if anc is not None else GT.content_fingerprint(vals[0][0]))", "(GT.content_fingerprint(vals[0][0]))")],
     ["W_hnscreen_refusals_stat_compares_under_an_anchor_and_re_reads_in_full_without_one", "R7_cost_anchor_bound_intermediate_calls_read_ZERO_bytes_content_run_is_4_reads_anchor_1_end_verify_1"]),
    ("launch_gate: the --ctx-mode DEFAULT is anchor-bound", "launch_gate.py", [("choices=GT.CTX_MODES, default=GT.CTX_MODE_CONTENT,", "choices=GT.CTX_MODES, default=GT.CTX_MODE_ANCHOR,")],
     ["R7_default_mode_is_content_it_matches_the_clean_cache_and_refuses_the_same_size_same_mtime_flip"]),
    ("launch_gate._with_ctx_anchor: an anchor named under content mode is silently ignored", "launch_gate.py", [("        if named:\n            raise GT.GateError(\"--\" +", "        if False:\n            raise GT.GateError(\"--\" +")],
     ["R7_an_anchor_named_under_content_mode_is_refused_never_silently_ignored"]),
    ("launch_gate._with_ctx_anchor: anchor-bound is allowed for any profile", "launch_gate.py", [("    if not prof.get(\"hn_rules\"):\n        raise GT.GateError(f\"--ctx-mode anchor-bound", "    if False:\n        raise GT.GateError(f\"--ctx-mode anchor-bound")],
     ["R7_anchor_bound_needs_all_three_flags_and_exists_only_for_the_hn_screen_profile"]),
    ("launch_gate._with_ctx_anchor: the three flags are not required together", "launch_gate.py", [("    if len(named) != 3:\n", "    if False:\n")],
     ["R7_anchor_bound_needs_all_three_flags_and_exists_only_for_the_hn_screen_profile"]),
    ("gate_token.fp_key AND anchor_fingerprint: BOTH layers of the invalid-key guard deleted (an INVALID anchor-bound fingerprint carries the anchor's content key)", "gate_token.py",
     [("        if fp.get(\"n_problems\") or not fp.get(\"content_sha256\"):\n            return f\"dir-anchor-INVALID:", "        if False:\n            return f\"dir-anchor-INVALID:"),
      ("    if not problems:\n        c = doc[\"content\"]\n        fp.update(", "    if True:\n        c = doc[\"content\"]\n        fp.update(")],
     ["R2_a_size_change_refuses_every_intermediate_entry_point_and_voids_at_the_end"]),
]


def apply_edits(src, edits, name):
    for old, new in edits:
        n = src.count(old)
        if n != 1:
            raise AssertionError(f"mutation {name!r}: snippet occurs {n} times (need exactly 1): {old[:90]!r}")
        src = src.replace(old, new)
    return src


def mutant_tree(dst: Path, fname: str, edits, name):
    """a fresh copy of the code the arms import (refe/*.py, code/, splits/, the C1 argv file) with ONE source edit applied"""
    ign = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(HERE, dst / "refe", ignore=ign)
    for sub in ("code", "splits"):
        if (PKG / sub).is_dir():
            shutil.copytree(PKG / sub, dst / sub, ignore=ign)
    (dst / "eval").mkdir()
    shutil.copy(PKG / "eval" / "rls1b_arms.json", dst / "eval" / "rls1b_arms.json")
    p = dst / "refe" / fname
    p.write_text(apply_edits(p.read_text(encoding="utf-8"), edits, name), encoding="utf-8", newline="\n")


def run_arms(tree_refe: Path, only=None) -> list:
    env = dict(os.environ, PYTHONIOENCODING="utf-8", CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, str(tree_refe / "test_hn_ctx_anchor.py"), "--arms"] + (["--only", ",".join(only)] if only else []), capture_output=True, text=True, timeout=1800, env=env,
                       cwd=str(tree_refe), encoding="utf-8", errors="replace")
    red = [ln.split(" ", 1)[1].strip() for ln in r.stdout.splitlines() if ln.startswith("ZZHNANC_RED ")]
    if "ZZHNANC_ARMS_DONE" not in r.stdout:
        red.append(f"arms process died rc {r.returncode}: {(r.stdout + r.stderr)[-300:]!r}")
    return red


def main() -> int:
    if ARMS_MODE:
        bad, n = arms()
        for b in bad:
            print("ZZHNANC_RED", b)
        print(f"ZZHNANC_ARMS_DONE checks {n} red {len(bad)}")
        return 0
    bad, n = arms()
    if bad:
        print("\n".join("PROBLEM the unmutated tree FAILS: " + b for b in bad))
        print(f"ZZTEST_FAIL {len(bad)} of {n} arm(s) failed on the real tree")
        return 1
    if NO_MUTANTS:
        print(f"ZZTEST_PASS {n}  ({n} arms on the real tree; --no-mutants: the mutation matrix was not run here)")
        return 0
    print(f"{n} arms green on the real tree.  Mutation matrix (each mutant is a fresh copy of the tree with ONE source edit; it runs the checks it must turn RED):")
    problems = []
    for name, fname, edits, expect in MUTATIONS:
        tmp = Path(tempfile.mkdtemp(prefix="hnanc_mut_"))
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
    if problems:
        print("\n".join("PROBLEM " + p for p in problems))
        print(f"ZZTEST_FAIL {len(problems)} problem(s)")
        return 1
    print(f"ZZTEST_PASS {n + len(MUTATIONS)}  ({n} arms on the real tree + {len(MUTATIONS)} RED mutations)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
