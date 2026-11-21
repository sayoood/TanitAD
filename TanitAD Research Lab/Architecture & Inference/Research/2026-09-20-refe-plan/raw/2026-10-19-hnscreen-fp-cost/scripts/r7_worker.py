"""R7 differential worker.  usage: python r7_worker.py <side: base|mine> <refe dir> <pristine cache> <side root> <out.json>

Runs ONE implementation (the tip's refe/ = base, or mine) over a fixed list of cache states and records, for every call that touches the ctx-cache in CONTENT mode (the default):
  content_fingerprint / data_manifest / profile_refusals / `launch_gate verify` / `launch_gate refusals` (stdout, rc, the --json record) / the minted token's binding / the number of full-file reads each call made.
Everything is normalised (time, hmac, side-root path, seconds) so that base and mine can be compared LITERALLY by the driver.  `mine` additionally records the same calls with an explicit `--ctx-mode content`.
"""
import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

side, REFE, PRISTINE, ROOT, OUT = sys.argv[1:6]
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(0, REFE)
import gate_token as GT  # noqa: E402
import launch_gate as L  # noqa: E402
import gate_checks as C  # noqa: E402
import hn_fingerprints as HF  # noqa: E402

PKGROOT = os.path.dirname(REFE)
ROOT = Path(ROOT)
shutil.rmtree(ROOT, ignore_errors=True)
ROOT.mkdir(parents=True)
READS = []
_real = GT._sha256_and_size


def counting(path, *a, **k):
    READS.append(os.path.abspath(str(path)))
    return _real(path, *a, **k)


GT._sha256_and_size = counting

# dropped, each with its reason: wall-clock fields; hmac (it signs the timestamps); evidence_sha256 (each evidence file embeds its own write time, so two runs of the SAME code differ); gate_sha256 (the hash of the gate's own
# source files - it differs by construction between a changed and an unchanged tree)
DROP = re.compile(r"(^|_)(hmac|sig|signature|time|timing|ts|minted|created|date|utc|ns|secs|seconds|elapsed|host|user|evidence_sha256|gate_sha256)($|_)", re.I)


def norm_val(v):
    if isinstance(v, dict):
        return {k: norm_val(x) for k, x in sorted(v.items()) if not DROP.search(k) and k != "pkg_root"}
    if isinstance(v, (list, tuple)):
        return [norm_val(x) for x in v]
    if isinstance(v, str):
        return norm_txt(v)
    return v


def norm_txt(t):
    t = t.replace(str(ROOT).replace("\\", "/"), "<ROOT>").replace(str(ROOT), "<ROOT>")
    t = t.replace(side + "_", "SIDE_")
    t = re.sub(r"\d+(\.\d+)? ?s\b", "<N>s", t)
    t = re.sub(r"[0-9a-f]{40}", "<SHA40>", t) if False else t
    return t


bank = ROOT / "bank.hnb"
bank.write_bytes(b"HNBANK1\n" + bytes(range(200)))
dummy = ROOT / "dummy_input.bin"
dummy.write_bytes(b"an input the pod would hold; only its existence matters here")
c1 = list(json.loads(Path(PKGROOT, "eval", "rls1b_arms.json").read_text(encoding="utf-8"))["arms"]["C1"]["argv"])
P0 = L.PROFILES["refe_hnscreen"]
for f_ in P0["data_flags"]:
    if f_ in c1 and f_ not in L.HNSCREEN_FLAGS:
        c1 = C._set(c1, f_, str(dummy))
fill_cache = ROOT / "w_fill"
shutil.copytree(PRISTINE, fill_cache)
FX = HF.fingerprints_expected(str(bank), str(fill_cache))
PFc = {**P0, "fingerprints_expected": dict(FX)}
profv = {**PFc, "declared_arm": "S1"}
pkg = ROOT / "pkg"
(pkg / "refe").mkdir(parents=True)
(pkg / "refe" / "dummy.py").write_text("X = 1\n", encoding="utf-8")
keyf = ROOT / "gate.key"
if os.environ.get("R7_KEY"):
    shutil.copyfile(os.environ["R7_KEY"], keyf)                  # ONE key for both sides, so key_id (and every signature input) is comparable
key = GT.load_key(keyf, create=True, forbid_under=[pkg])
commit = "a" * 40
tree = GT.tree_manifest(pkg)
checks_all = profv["checks"] + tuple(profv.get("na_ok", ()))


def argv_for(cdir, seed=1):
    return C._set(c1, "--seed", str(seed)) + ["--hn-bank", str(bank), "--ctx-cache", str(cdir)]


def cli(cmd, prof, extra):
    buf = io.StringIO()
    old = L.PROFILES["refe_hnscreen"]
    L.PROFILES["refe_hnscreen"] = prof
    n0 = len(READS)
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            rc = L.main([cmd, "--profile", "refe_hnscreen", *extra])
    finally:
        L.PROFILES["refe_hnscreen"] = old
    return rc, buf.getvalue(), len(READS) - n0


def edit_keep_mtime(p, fn):
    st = os.stat(p)
    b = bytearray(open(p, "rb").read())
    nb = fn(b)
    open(p, "wb").write(bytes(nb if nb is not None else b))
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))


def payload(d):
    return sorted(Path(d, "units").glob("*.ctx.u16"))[0]


def s_clean(d):
    pass


def s_flip(d):
    edit_keep_mtime(payload(d), lambda b: b.__setitem__(len(b) // 2, b[len(b) // 2] ^ 1))


def s_grow(d):
    edit_keep_mtime(payload(d), lambda b: bytes(b) + b"\x00")


def s_touch(d):
    p = payload(d)
    st = os.stat(p)
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))


def s_extra(d):
    Path(d, "units", "extra.bin").write_bytes(b"stray")


def s_missing(d):
    os.remove(payload(d))


def s_nomanifest(d):
    os.remove(Path(d, GT.CONTENT_MANIFEST))


def s_edit_manifest_json(d):
    p = Path(d, "manifest.json")
    edit_keep_mtime(p, lambda b: b.__setitem__(len(b) // 3, (b[len(b) // 3] ^ 1) if b[len(b) // 3] not in (10, 13) else b[len(b) // 3]))


def s_resealed(d):
    """a VALID sealed cache that is not the registered one: one payload byte changed and the seal REWRITTEN (self-consistent) - the content check passes, the registered-pin compare must refuse (the other message branch)"""
    edit_keep_mtime(payload(d), lambda b: b.__setitem__(len(b) // 2, b[len(b) // 2] ^ 1))
    GT.seal_content_manifest(str(d))


SCEN = [("clean", s_clean), ("resealed_self_consistent", s_resealed), ("flip_same_size_same_mtime", s_flip), ("grow_size_change", s_grow), ("touch_mtime_only", s_touch), ("extra_file", s_extra), ("missing_file", s_missing),
        ("no_manifest", s_nomanifest), ("manifest_json_flip", s_edit_manifest_json)]

def mint_over(d, tag):
    """a PASS token minted by the real `finalize` over the CLEAN cache at `d` (the cache is tampered with only afterwards, in place - as the chain's own failure would)"""
    lg = ROOT / f"lg_{tag}"
    lg.mkdir()
    binding = GT.binding_of(profv["name"], commit, GT.tree_digest(tree), argv_for(d))
    for c_ in checks_all:
        L.write_evidence(lg, C.evidence(c_, "PASS", [], {}), binding, {})
    n0 = len(READS)
    verdict, tokp, tok = L.finalize(profv, argv_for(d), pkg, lg, key, commit=commit)
    return tokp, {"verdict": verdict, "token": norm_val(tok), "reads": len(READS) - n0}


rec = {"fx": FX}
out = {}
for name, fn in SCEN:
    d = ROOT / f"w_{name}"
    shutil.copytree(PRISTINE, d)
    tokp, mint_rec = mint_over(d, name)
    rec.setdefault("mint", mint_rec)
    fn(d)
    av = ["--ctx-cache", str(d), "--hn-bank", str(bank)]
    flags = ("--ctx-cache", "--hn-bank")
    e = {"mint_again": mint_rec}

    def attempt(key_, f):
        n = len(READS)
        try:
            v = f()
            e[key_] = {"ok": norm_val(v), "reads": len(READS) - n}
        except Exception as ex:                                              # noqa: BLE001 -- a refusal IS the observable
            e[key_] = {"raises": type(ex).__name__, "msg": norm_txt(str(ex)), "reads": len(READS) - n}
    attempt("content_fingerprint", lambda: GT.content_fingerprint(str(d)))
    attempt("fp_key", lambda: GT.fp_key(GT.content_fingerprint(str(d))))
    attempt("data_manifest", lambda: GT.data_manifest(av, flags, content_flags=("--ctx-cache",)))
    argf = ROOT / f"argv_{name}.json"
    GT.write_json(argf, argv_for(d))
    n = len(READS)
    e["profile_refusals"] = {"ok": norm_val(L.profile_refusals(profv, argv_for(d))), "reads": len(READS) - n}
    variants = [("", [])] + ([("explicit_content", ["--ctx-mode", "content"])] if side == "mine" else [])
    for vn, ctx in variants:
        jr = ROOT / f"vrec_{name}_{vn or 'plain'}.json"
        rc, txt, nr = cli("verify", PFc, ["--token", str(tokp), "--args-file", str(argf), "--pkg", str(pkg), "--key-file", str(keyf), "--json", str(jr)] + ctx)
        e["verify" + ("_" + vn if vn else "")] = {"rc": rc, "out": norm_txt(txt), "record": norm_val(json.loads(jr.read_text(encoding="utf-8"))) if jr.is_file() else None, "reads": nr}
        rc, txt, nr = cli("refusals", PFc, ["--arm-name", "S1", "--args-file", str(argf), "--key-file", str(keyf)] + ctx)
        e["refusals" + ("_" + vn if vn else "")] = {"rc": rc, "out": norm_txt(txt), "reads": nr}
    out[name] = e
rec["scenarios"] = out
json.dump(rec, open(OUT, "w", encoding="utf-8"), indent=1, sort_keys=True)
print("WORKER_DONE", side, len(out), "scenarios")
