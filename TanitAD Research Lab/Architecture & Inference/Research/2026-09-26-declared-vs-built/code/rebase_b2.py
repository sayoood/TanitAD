"""Batch 2, REBASED onto NEW-1 (the residual prior), plus the τ-file hook (Master Mind 2026-09-26).

    python rebase_b2.py <NEW-1 package root>

1. base   = ab436ee's blobs (refc_v3_train.py, declared_vs_built.py);
   ours   = patch_d3.py applied to that base (D3, exactly as gated on the dev box);
   theirs = NEW-1's full files (blobs asserted: 9cac3dab / a50ff690);
   merged = `git merge-file` on LF-normalised copies -- REFUSES on any conflict -- and every line
   either side added (removed) is asserted present (absent) in the merge.
2. The τ-file hook on the merged files (SPEC_REFCV7 §7: "its sha256 is recorded in config.json"):
   trainer `--nav-compliance-tau-file` + `_verify_navc_tau_file` (train() only) + the pin's
   refusals + the seam stamp; G-DVB entry `nav_compliance_tau_file` (203 -> 204); two DECLARED
   fields on RefCV3Config (G-HYG) on NEW-1's refc_v3.py; NEW-1's registry-count pin 203 -> 204.
3. Writes the full files into code/fix2/ with each file's tip EOL.
"""
import difflib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

N1 = Path(sys.argv[1]) / "code" / "fix"
PK = Path(__file__).resolve().parents[1]
OUT = PK / "code" / "fix2"
GIT = ["git", "--git-dir=C:/Users/Admin/tanitad-push/.git"]
THEIRS_BLOB = {"stack/scripts/refc_v3_train.py": "9cac3dabca2b2fbac29a1d0b625f95215f71d985",
               "stack/tanitad/train/declared_vs_built.py": "a50ff690a364966ee21215938b4d34dc0a0a79d8",
               "stack/tanitad/refs/refc_v3.py": "774d3a3b1d5f7fa4439362494aae9fdcbb1cda9a",
               "stack/tests/test_declared_vs_built.py": "5137fde792a8b5b6a4c51bafdbc876a6ee2ff239"}


def blob(p: Path) -> str:
    out = subprocess.run(["git", "hash-object", "--no-filters", str(p)], capture_output=True,
                         text=True).stdout.strip()
    assert len(out) == 40, (p, out)
    return out


def edit(s: str, old: str, new: str, tag: str) -> str:
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"[rebase_b2] {tag}: anchor found {n} times")
    return s.replace(old, new)


def lf(b: bytes) -> str:
    return b.decode("utf-8").replace("\r\n", "\n")


for rp, want in THEIRS_BLOB.items():
    got = blob(N1 / rp)
    if got != want:
        raise SystemExit(f"[rebase_b2] NEW-1 {rp} is blob {got}, expected {want} -- NEW-1 moved")

# ---------------------------------------------------------------- 1. the 3-way merge
tmp = Path(tempfile.mkdtemp(prefix="rebase_b2_", dir=str(PK / "code")))
try:
    for rp in ("stack/scripts/refc_v3_train.py", "stack/tanitad/train/declared_vs_built.py"):
        (tmp / "d3" / rp).parent.mkdir(parents=True, exist_ok=True)
        base = subprocess.run(GIT + ["cat-file", "-p", f"ab436ee:{rp}"], capture_output=True,
                              check=True).stdout
        (tmp / "d3" / rp).write_bytes(base)
    subprocess.run([sys.executable, str(PK / "code" / "patch_d3.py"), str(tmp / "d3")], check=True)
    merged = {}
    for rp in ("stack/scripts/refc_v3_train.py", "stack/tanitad/train/declared_vs_built.py"):
        base = lf(subprocess.run(GIT + ["cat-file", "-p", f"ab436ee:{rp}"], capture_output=True,
                                 check=True).stdout)
        ours = lf((tmp / "d3" / rp).read_bytes())
        theirs = lf((N1 / rp).read_bytes())
        for k, v in (("base", base), ("ours", ours), ("theirs", theirs)):
            (tmp / f"{Path(rp).stem}.{k}").write_text(v, encoding="utf-8", newline="\n")
        r = subprocess.run(["git", "merge-file", "-p", str(tmp / f"{Path(rp).stem}.ours"),
                            str(tmp / f"{Path(rp).stem}.base"), str(tmp / f"{Path(rp).stem}.theirs")],
                           capture_output=True)
        if r.returncode != 0:
            raise SystemExit(f"[rebase_b2] {rp}: merge-file reports {r.returncode} conflict(s)")
        m = r.stdout.decode("utf-8")
        B, M = base.splitlines(), m.splitlines()

        def added_removed(a, b):
            add, rem = [], []
            for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
                if tag in ("replace", "delete"):
                    rem += a[i1:i2]
                if tag in ("replace", "insert"):
                    add += b[j1:j2]
            return add, rem
        m_add, m_rem = added_removed(B, M)
        for side, S in (("ours", ours.splitlines()), ("theirs", theirs.splitlines())):
            add, rem = added_removed(B, S)
            lost = [x for x in add if x not in m_add]
            kept = [x for x in rem if x not in m_rem]
            assert not lost, f"{rp}: the merge LOST {len(lost)} line(s) {side} added: {lost[:3]}"
            assert not kept, f"{rp}: the merge KEPT {len(kept)} line(s) {side} removed: {kept[:3]}"
        assert len(M) == len(B) + (len(ours.splitlines()) - len(B)) + (len(theirs.splitlines()) - len(B))
        merged[rp] = m
        print(f"[rebase_b2] merged {rp}: {len(B)} -> {len(M)} lines, 0 conflicts")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# ---------------------------------------------------------------- 2. the τ-file hook
t = merged["stack/scripts/refc_v3_train.py"]
t = edit(t, '''def _logged_after(step: int, log_every: int, steps: int) -> bool:
''', '''#: SPEC_REFCV7 §7 (A2): `--nav-compliance-tau-rad` must EQUAL the banked τ at this tolerance --
#: the one `declared_vs_built.check_refcv7_required(tau_file=)` compares at.
NAVC_TAU_FILE_TOL = 1e-12


def _verify_navc_tau_file(args) -> dict | None:
    """-> ``{path, sha256, tau}`` for ``--nav-compliance-tau-file``, or ``None`` without it.

    ⛔ SPEC_REFCV7 §7 (A2, the PI's E1 ruling): the τ of `--graft-nav-compliance` is the BANKED
    derivation on the TRAIN split (…/2026-09-26-declared-vs-built/raw/nav_compliance_tau_train.json)
    and "its sha256 is recorded in config.json". REFUSES a missing or unreadable file, a τ that is
    not a positive finite number, and a `--nav-compliance-tau-rad` that differs from the file's τ
    by more than NAVC_TAU_FILE_TOL. Called by `train()` only: `_pin_trainer_cfg` never OPENS the
    file, because an eval rebuild re-runs the pin from the recorded argv on boxes where the file
    does not exist (`taniteval/tools/refcv3_arm.rebuild_config`) and must build the same model.
    """
    import hashlib
    p = getattr(args, "nav_compliance_tau_file", None)
    if not p:
        return None
    path = Path(p)
    if not path.is_file():
        raise SystemExit(
            f"[v3] ⛔ --nav-compliance-tau-file {p}: no such file. The tolerance must come from "
            f"the banked TRAIN-split derivation, or 'compliant' has no evidence behind it.")
    raw = path.read_bytes()
    try:
        tau = float(json.loads(raw.decode("utf-8"))["tau"])
    except (ValueError, KeyError, TypeError, UnicodeDecodeError) as exc:
        raise SystemExit(
            f"[v3] ⛔ --nav-compliance-tau-file {p}: not a tau file ({type(exc).__name__}: "
            f"{exc}); expected JSON with a numeric `tau` in rad.") from None
    if not (math.isfinite(tau) and tau > 0.0):
        raise SystemExit(f"[v3] ⛔ --nav-compliance-tau-file {p}: tau = {tau!r} is not a "
                         f"positive finite tolerance.")
    given = float(getattr(args, "nav_compliance_tau_rad", 0.0) or 0.0)
    if abs(given - tau) > NAVC_TAU_FILE_TOL:
        raise SystemExit(
            f"[v3] ⛔ --nav-compliance-tau-rad {given!r} differs from the banked tau {tau!r} in "
            f"{p} by {abs(given - tau):.3g} rad (> {NAVC_TAU_FILE_TOL:g}). Pass the file's value "
            f"exactly: the launch gate compares at the same tolerance.")
    return {"path": str(p), "sha256": hashlib.sha256(raw).hexdigest(), "tau": tau}


def _logged_after(step: int, log_every: int, steps: int) -> bool:
''', "T-A verifier")

t = edit(t, '''    _navc = bool(getattr(args, "graft_nav_compliance", False))
    _navc_tau = float(getattr(args, "nav_compliance_tau_rad", 0.0) or 0.0)
''', '''    _navc = bool(getattr(args, "graft_nav_compliance", False))
    _navc_tau = float(getattr(args, "nav_compliance_tau_rad", 0.0) or 0.0)
    # ⛔ SPEC_REFCV7 §7 (A2): the τ's BANKED file. The pin records its path and never OPENS it (an
    # eval rebuild re-runs this pin on boxes without the file); `train()` holds the float against
    # it (`_verify_navc_tau_file`) and stamps {path, sha256, tau} into config.json.
    _navc_file = getattr(args, "nav_compliance_tau_file", None)
    if _navc_file and not _navc:
        raise SystemExit(
            "[v3] ⛔ --nav-compliance-tau-file without --graft-nav-compliance: the banked "
            "tolerance of a term that is not built -- stamped and read by nothing.")
    if _navc_file and not _navc_tau > 0.0:
        raise SystemExit(
            "[v3] ⛔ --nav-compliance-tau-file needs --nav-compliance-tau-rad as well: the FILE "
            "verifies the float (train() refuses any difference > 1e-12 and stamps the file's "
            "sha256), and the FLOAT is what an eval rebuild on a box without the file builds "
            "from. Pass the file's `tau` exactly.")
    if _navc_file:
        cfg.nav_compliance_tau_file = str(_navc_file)
''', "T-B pin")

t = edit(t, '''    cfg = _pin_trainer_cfg(
        v3.refc_v3_smoke_config(args.arm == "hier") if args.smoke else
        v3.refc_v3_sized_config(args.size, hier=args.arm == "hier"), args)
''', '''    cfg = _pin_trainer_cfg(
        v3.refc_v3_smoke_config(args.arm == "hier") if args.smoke else
        v3.refc_v3_sized_config(args.size, hier=args.arm == "hier"), args)
    # ⛔ SPEC_REFCV7 §7 (A2): the banked τ file, held against the float BEFORE anything is built
    _navc_tau_stamp = _verify_navc_tau_file(args)
    if _navc_tau_stamp is not None:
        cfg.nav_compliance_tau_sha256 = _navc_tau_stamp["sha256"]
''', "T-C train")

t = edit(t, '''        "nav_compliance_tau_rad": float(getattr(core, "nav_compliance_tau_rad", 0.0)),
''', '''        "nav_compliance_tau_rad": float(getattr(core, "nav_compliance_tau_rad", 0.0)),
        # ⭐ SPEC_REFCV7 §7 (A2): the banked τ file behind it -- `None` without the flag
        "nav_compliance_tau_file": (
            {"path": cfg.nav_compliance_tau_file,
             "sha256": getattr(cfg, "nav_compliance_tau_sha256", None),
             "tau": float(getattr(core, "nav_compliance_tau_rad", 0.0))}
            if getattr(cfg, "nav_compliance_tau_file", None) else None),
''', "T-D seam stamp")

t = edit(t, '''                     help="the terminal-heading tolerance of --graft-nav-compliance, DERIVED "
                          "on the TRAIN split (no default that means anything; 0 = unset).")
''', '''                     help="the terminal-heading tolerance of --graft-nav-compliance, DERIVED "
                          "on the TRAIN split (no default that means anything; 0 = unset).")
    g6t.add_argument("--nav-compliance-tau-file", default=None,
                     help="SPEC_REFCV7 §7: the BANKED tau file (JSON with a numeric `tau`, e.g. "
                          ".../2026-09-26-declared-vs-built/raw/nav_compliance_tau_train.json). "
                          "Needs --nav-compliance-tau-rad too: train() REFUSES a float that "
                          "differs by > 1e-12, a missing file and an unreadable one, and stamps "
                          "{path, sha256, tau} into config.json (seams.nav_compliance_tau_file).")
''', "T-E argparse")

d = merged["stack/tanitad/train/declared_vs_built.py"]
d = edit(d, '''def _c_navc_tau(m, a):
''', '''def _c_navc_tau_file(m, a):
    """SPEC_REFCV7 §7: the BUILT tolerance against the BANKED file's τ -- a literal read from the
    file, never from the argv float the file exists to verify."""
    p = _a(a, "nav_compliance_tau_file", None)
    if not p:
        return []
    import json as _json
    try:
        with open(p, encoding="utf-8") as fh:
            want = float(_json.load(fh)["tau"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return [Mismatch(_flag("nav_compliance_tau_file"), p, None, "the banked tau file",
                         f"unreadable ({type(exc).__name__})")]
    return _near("nav_compliance_tau_file", want, getattr(_dec(m), "navc_tau_rad", None),
                 "core.decoder.navc_tau_rad", "the BUILT tolerance must be the banked one",
                 tol=1e-12)


def _c_navc_tau(m, a):
''', "D-F reader")
d = edit(d, '''_b("nav_compliance_tau_rad", _c_navc_tau)
''', '''_b("nav_compliance_tau_rad", _c_navc_tau)
_b("nav_compliance_tau_file", _c_navc_tau_file)
''', "D-F register")

r3 = lf((N1 / "stack/tanitad/refs/refc_v3.py").read_bytes())
r3 = edit(r3, '''    u0_absent_under_ddim: str | None = None
''', '''    u0_absent_under_ddim: str | None = None
    # ⭐ SPEC_REFCV7 §7 (A2): the BANKED τ file behind `core.nav_compliance_tau_rad` and its sha256,
    # stamped under `seams.nav_compliance_tau_file`. Provenance only -- the decoder reads the float.
    # DECLARED (G-HYG): the pin sets the path, `train()` the sha256 after reading the file.
    nav_compliance_tau_file: str | None = None
    nav_compliance_tau_sha256: str | None = None
''', "R3 fields")

tv = lf((N1 / "stack/tests/test_declared_vs_built.py").read_bytes())
tv = edit(tv, '''    assert len(dvb.REGISTRY) == 203
''', '''    assert len(dvb.REGISTRY) == 204          # + --nav-compliance-tau-file (batch 2, SPEC_REFCV7 §7)
''', "TV count")

# ---------------------------------------------------------------- 3. write, each with its tip EOL
out = {"stack/scripts/refc_v3_train.py": (t, "CRLF"),
       "stack/tanitad/train/declared_vs_built.py": (d, "LF"),
       "stack/tanitad/refs/refc_v3.py": (r3, "LF"),
       "stack/tests/test_declared_vs_built.py": (tv, "LF")}
for rp, (txt, e) in out.items():
    p = OUT / rp
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes((txt.replace("\n", "\r\n") if e == "CRLF" else txt).encode("utf-8"))
    print(f"[rebase_b2] wrote {rp} ({e}) blob {blob(p)}")
