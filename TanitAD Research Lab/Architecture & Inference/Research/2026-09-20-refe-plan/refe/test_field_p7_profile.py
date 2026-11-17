#!/usr/bin/env python3
"""P7-graft field-loss arms: launch-gate PROFILES `refe_field_p7` (F, --w-field 2.0) and `refe_field_p7_ctrl` (C, --w-field 0.0)  (PREREG_AUX_FIELD_LOSS_P7_DRAFT; gaps G1, G2, G5, G6).

  python test_field_p7_profile.py            everything, incl. the source-mutation harness
  python test_field_p7_profile.py --fast     no mutation harness
  python test_field_p7_profile.py --check N  one named check (what the harness runs on a mutated copy of this tree)

CPU only, no scipy, stdlib + the gate modules.  Prints ZZFIELDP7PROF_PASS <n>.  Asserts: both profiles exist and share the P5-C recipe; argv(F) - argv(C0) = {--w-field, --field-labels, --field-margin,
--field-interp, --out} and argv(C) - argv(C0) = {--field-labels, --field-margin, --field-interp, --out}; each arm's own template argv passes the gate's refusal checks; EVERY single-flag mutation is REFUSED by name
(wrong --w-field, missing --field-labels, a stray --traj-label-table, an aux-head flag, --seed 1, a wrong margin ...); every registered pin refuses when it is absent or wrong; the P5-C profiles are untouched and
still refuse the field flags.  The mutation harness neutralises ONE emitting line of a COPY of launch_gate.py at a time and requires the check to go RED.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gate_token as GT  # noqa: E402
import launch_gate as L  # noqa: E402

N = 0
NAMES = {"F": "refe_field_p7", "C": "refe_field_p7_ctrl"}
OLD_PROFILES_P5C = ("refe_p5c", "refe_p5c_c0", "refe_p5c_c0r")


def ok(cond, msg=""):
    global N
    assert cond, msg
    N += 1


def refusals(prof: dict, argv: list) -> str:
    return " || ".join(L.profile_refusals(prof, argv))


def swap(argv: list, flag: str, *vals) -> list:
    a = list(argv)
    i = a.index(flag)
    j = i + 1
    while j < len(a) and not a[j].startswith("--"):
        j += 1
    return a[:i] + [flag, *vals] + a[j:]


def drop(argv: list, flag: str) -> list:
    a = list(argv)
    i = a.index(flag)
    j = i + 1
    while j < len(a) and not a[j].startswith("--"):
        j += 1
    return a[:i] + a[j:]


def fl(a):
    return {k: tuple(v) for k, v in GT.flag_pairs(a)}


def chk_registry():
    for arm, nm in NAMES.items():
        ok(nm in L.PROFILES and L.PROFILES[nm]["name"] == nm and L.PROFILES[nm]["p7field_rules"] is True and L.PROFILES[nm]["p7field_arm"] == arm and L.PROFILES[nm]["field_arm"] == arm)
        pr = L.PROFILES[nm]
        ok(not pr.get("p5c_rules") and "p5c_arm" not in pr, "not a P5-C profile: p5c_refusals would refuse the field flags and test_p5c_profile pins p5c_rules to the three P5-C profiles")
        ok(pr["required_flags"]["--w-field"] == ("2.0" if arm == "F" else "0.0") and pr["field_w"] == (2.0 if arm == "F" else 0.0) and type(pr["field_w"]) is float)
        ok("--field-labels" in pr["required_flags"] and pr["required_flags"]["--field-margin"] == "0.25" and pr["required_flags"]["--field-interp"] == "5hz")
        ok("--field-labels" in pr["data_flags"] and "--init-from" in pr["data_flags"] and "--traj-label-table" not in pr["data_flags"])
        ok(pr["live_terms"] == L.P6_TERMS + ("field_L",) and pr["field_min_coverage"] == 0.999)
        ok(all(f in pr["forbidden_flags"] for f in L.AUX_HEAD_FLAGS) and "--traj-label-table" in pr["forbidden_flags"] and "--field-shuffle-seed" in pr["forbidden_flags"] and "--init-from" not in pr["forbidden_flags"])
        ok(pr["required_flags"]["--seed"] == "0" and pr["required_flags"]["--lr"] == "5e-5" and pr["required_flags"]["--steps"] == "2000" and pr["required_flags"]["--epochs"] == "0" and "--ego-yaw-table" in pr["required_flags"])
        ok(pr["init_from_sha256"] == L.P5C_BASE_SHA256 and pr["init_from_sha256"].startswith("2613fc17"), "the P7 base sha pin")
        ok(pr["expect_schedule"] == L.PROFILES["refe_p5c_c0"]["expect_schedule"], "the P5-C schedule and bank pins")
        names = [e["script"] for e in pr["suite"]]
        ok(all(e in pr["suite"] for e in L.PROFILES["refe_p5c_c0"]["suite"]), "suite = the P5-C suite ...")
        ok(all(e in pr["suite"] for e in L.FIELD_SUITE_EXTRA) and "test_field_loss.py" in names, "... plus FIELD_SUITE_EXTRA (test_field_loss.py)")
        ok(names[-1] == "test_field_p7_profile.py", "and this file")
        for e in pr["suite"]:
            ok((HERE / e["script"]).is_file(), e["script"])
        ok("G-RESERVE" not in pr["checks"])
    ok(L.FIELD_P7_W == 2.0 and type(L.FIELD_P7_W) is float)
    ok(L.FIELD_P7_STORE_SHA256 == "4a17db673b1548ba35724ab848a7dc551e4b0bc1e49c57ae93e6357454bf287a" and L.FIELD_P7_STORE_KEYS == 103139)
    ok(L.FIELD_P7_LOSS_SHA256 == "e29d91c349539ba366c73739fd2fe7fc6f1e7a6a015c4c0749df5f15df36c994")
    for nm in NAMES.values():
        pr = L.PROFILES[nm]
        ok((pr["field_store_sha256"], pr["field_store_keys"], pr["field_loss_sha256"]) == (L.FIELD_P7_STORE_SHA256, L.FIELD_P7_STORE_KEYS, L.FIELD_P7_LOSS_SHA256))
    ok(L.FIELD_P7_LOSS_SHA256 == hashlib.sha256((HERE / "field_loss.py").read_bytes().replace(b"\r\n", b"\n")).hexdigest(), "the registered field_loss.py hash is the shipped file's")


def chk_argv_delta():
    f0, ff, fc = fl(L.p5c_argv("C0")), fl(L.p7field_argv("F")), fl(L.p7field_argv("C"))
    dF = {k for k in set(f0) | set(ff) if f0.get(k) != ff.get(k)}
    dC = {k for k in set(f0) | set(fc) if f0.get(k) != fc.get(k)}
    ok(dF == {"--w-field", "--field-labels", "--field-margin", "--field-interp", "--out"}, dF)
    ok(dC == {"--field-labels", "--field-margin", "--field-interp", "--out"}, dC)
    ok(set(ff) - set(f0) == {"--field-labels", "--field-margin", "--field-interp"} and set(f0) - set(ff) == set())
    ok(ff["--w-field"] == ("2.0",) and fc["--w-field"] == ("0.0",) == f0["--w-field"])
    ok({k for k in set(ff) | set(fc) if ff.get(k) != fc.get(k)} == {"--w-field", "--out"}, "F and C differ by --w-field alone (and their own out dir)")
    ok(ff["--out"][0].endswith("/p7field_f") and fc["--out"][0].endswith("/p7field_c") and ff["--out"] != f0["--out"])
    ok(ff["--seed"] == ("0",) == fc["--seed"] and ff["--field-margin"] == ("0.25",) and ff["--field-interp"] == ("5hz",))
    ok(not any(k in ff for k in L.AUX_HEAD_FLAGS + ("--traj-label-table", "--field-shuffle-seed")))
    # the rendered pod argv: the store path is the full store beside P7's, the rest is the P5-C template's
    ok(ff["--field-labels"] == ("/workspace/data/refe_navtrain/aux_label_store_full",))
    ok(L.p7field_argv("F", paths={"field_labels": "/x/store"})[L.p7field_argv("F").index("--field-labels") + 1] == "/x/store")
    try:
        L.p7field_argv("C0")
        ok(False, "an unknown arm must raise")
    except ValueError:
        ok(True)


def chk_clean():
    for arm, nm in NAMES.items():
        ok(refusals(L.PROFILES[nm], L.p7field_argv(arm)) == "", (nm, refusals(L.PROFILES[nm], L.p7field_argv(arm))))
        ok("--resume" in L.p7field_argv(arm) and refusals(L.PROFILES[nm], drop(L.p7field_argv(arm), "--resume")) == "", "--resume is in the template and optional")


def chk_levers():
    """every single-flag mutation of either arm's argv is refused, by NAME."""
    for arm, nm in NAMES.items():
        pr, a = L.PROFILES[nm], L.p7field_argv(arm)
        wrong_w = ("0.0", "1.0", "2.5", "-2.0", "nan") if arm == "F" else ("2.0", "0.5", "1e-9")
        cases = [(f"--w-field {w}", swap(a, "--w-field", w), "--w-field") for w in wrong_w]
        cases += [
            ("no --w-field", drop(a, "--w-field"), "--w-field"),
            ("no --field-labels", drop(a, "--field-labels"), "--field-labels"),
            ("--field-labels without a path", swap(a, "--field-labels"), "--field-labels"),
            ("stray --traj-label-table", a + ["--traj-label-table", "/x/t.json.gz"], "--traj-label-table"),
            ("aux head --aux-drv", a + ["--aux-drv"], "--aux-drv"),
            ("aux head --w-aux-drv", a + ["--w-aux-drv", "0.1"], "--w-aux-drv"),
            ("aux head store", a + ["--aux-drv-labels", "/x"], "--aux-drv-labels"),
            ("--seed 1", swap(a, "--seed", "1"), "--seed"),
            ("no --seed", drop(a, "--seed"), "--seed"),
            ("margin 0.5", swap(a, "--field-margin", "0.5"), "--field-margin"),
            ("margin 0.0", swap(a, "--field-margin", "0.0"), "--field-margin"),
            ("no margin", drop(a, "--field-margin"), "--field-margin"),
            ("interp 10hz", swap(a, "--field-interp", "10hz"), "--field-interp"),
            ("no interp", drop(a, "--field-interp"), "--field-interp"),
            ("--field-shuffle-seed", a + ["--field-shuffle-seed", "1"], "--field-shuffle-seed"),
            ("lr 2e-4", swap(a, "--lr", "2e-4"), "--lr"),
            ("steps 1999", swap(a, "--steps", "1999"), "--steps"),
            ("epochs 25", swap(a, "--epochs", "25"), "--epochs"),
            ("no --init-from", drop(a, "--init-from"), "--init-from"),
            ("no yaw table", drop(a, "--ego-yaw-table"), "--ego-yaw-table"),
            ("another bank", swap(a, "--targets", "/workspace/data/refe_navtrain/train_grow_cmd_3c"), "bank"),
            ("batch 8", swap(a, "--batch", "8"), "--batch"),
            ("workers 8", swap(a, "--workers", "8"), "--workers"),
            ("log-every 5", swap(a, "--log-every", "5"), "--log-every"),
            ("score-w 0.2", swap(a, "--score-w", "0.2"), "--score-w"),
            ("weight decay 0", swap(a, "--weight-decay", "0.0"), "--weight-decay"),
            ("yaw wrap on", swap(a, "--yaw-wrap", "on"), "--yaw-wrap"),
            ("an unlisted lever", a + ["--ckpt-every-steps", "100"], "--ckpt-every-steps"),
            ("a lever twice", a + ["--field-margin", "0.25"], "--field-margin"),
            ("--grow", a + ["--grow"], "--grow"),
            # MM 2026-11-17: dropped levers that `required_flags` does NOT list: only p7field_refusals' "nothing dropped" loop can refuse these,
            # so the mutation that deletes that loop must go RED here (it survived when only required flags were dropped).
            ("no --tf32", drop(a, "--tf32"), "--tf32"),
            ("no --compile", drop(a, "--compile"), "--compile"),
            ("no --ckpt-every-min", drop(a, "--ckpt-every-min"), "--ckpt-every-min"),
            ("no --epoch-unit", drop(a, "--epoch-unit"), "--epoch-unit")]
        for what, mut, needle in cases:
            rs = refusals(pr, mut)
            ok(rs != "" and needle in rs, (arm, what, rs))
    # each arm refuses the other's argv, and the P5-C control's (no field flags)
    ok(refusals(L.PROFILES["refe_field_p7"], L.p7field_argv("C")) != "" and refusals(L.PROFILES["refe_field_p7_ctrl"], L.p7field_argv("F")) != "", "F and C refuse each other's argv")
    ok("--field-labels" in refusals(L.PROFILES["refe_field_p7_ctrl"], L.p5c_argv("C0")) and "--field-labels" in refusals(L.PROFILES["refe_field_p7"], L.p5c_argv("C0")), "P5-C's C0 argv is not a field arm")


def chk_pins():
    """each registered pin refuses when absent or wrong, by NAME; a wrong pin cannot be hidden by a right argv."""
    for arm, nm in NAMES.items():
        a = L.p7field_argv(arm)
        for key, bad, needle in (("init_from_sha256", None, "init_from_sha256"), ("init_from_sha256", "AB" * 32, "init_from_sha256"), ("init_from_sha256", "ab" * 31, "init_from_sha256"),
                                 ("field_w", None, "field_w"), ("field_w", 1.0 if arm == "F" else 2.0, "field_w"), ("field_w", 2, "field_w"),
                                 ("field_store_sha256", None, "field_store_sha256"), ("field_store_sha256", "zz", "field_store_sha256"),
                                 ("field_store_sha256", "cd" * 32, "loader pins"), ("field_store_keys", None, "field_store_keys"), ("field_store_keys", 92993, "loader pins"),
                                 ("field_loss_sha256", None, "field_loss_sha256"), ("field_loss_sha256", "ab" * 32, "field_loss.py hashes to")):
            p = dict(L.PROFILES[nm])
            if key == "field_w" and bad == 2 and arm == "C":
                bad = 0                                                          # an int 0 is not the float 0.0 the pin registers
            p[key] = bad
            rs = refusals(p, a)
            ok(rs != "" and needle in rs, (nm, key, bad, rs))
    p = dict(L.PROFILES["refe_field_p7"])
    p["expect_schedule"] = {**p["expect_schedule"], "steps": 1999}
    ok("registered schedule" in refusals(p, L.p7field_argv("F")))
    p = dict(L.PROFILES["refe_field_p7"])
    p["expect_schedule"] = {**p["expect_schedule"], "targets_sha256_by_bank": {"train_grow_cmd_3c_p7": "xx"}}
    ok("targets_sha256" in refusals(p, L.p7field_argv("F")))
    # field_loss.py is re-hashed from the shipped bytes: a tampered copy of the module beside a copy of the gate is refused
    td = Path(tempfile.mkdtemp(prefix="fp7_fl_"))
    try:
        for f in ("launch_gate.py", "gate_token.py", "gate_checks.py", "field_loss.py", "aux_drivable_head.py"):
            shutil.copyfile(HERE / f, td / f)
        (td / "field_loss.py").write_bytes((HERE / "field_loss.py").read_bytes() + b"\n# tampered\n")
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        code = "import sys; sys.path.insert(0, sys.argv[1]); import launch_gate as L; assert L.HERE == __import__('pathlib').Path(sys.argv[1]).resolve(); r = L.p7field_refusals(L.PROFILES['refe_field_p7'], L.p7field_argv('F')); print('REFUSED' if any('field_loss.py hashes to' in x for x in r) else 'ACCEPTED')"
        r = subprocess.run([sys.executable, "-c", code, str(td)], cwd=str(HERE), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        ok("REFUSED" in r.stdout, (r.stdout, r.stderr[-300:]))
    finally:
        shutil.rmtree(td, ignore_errors=True)


def chk_fresh_out():
    td = Path(tempfile.mkdtemp(prefix="fp7_out_"))
    try:
        for arm, nm in NAMES.items():
            a = swap(L.p7field_argv(arm), "--out", str(td / arm))
            ok(refusals(L.PROFILES[nm], a) == "", "no out dir yet")
            (td / arm).mkdir()
            ok(refusals(L.PROFILES[nm], a) == "", "an empty out dir is fresh")
            (td / arm / "ckpt_last.pt").write_bytes(b"x")
            rs = refusals(L.PROFILES[nm], a)
            ok("ckpt_last.pt" in rs and "FRESH" in rs, rs)
    finally:
        shutil.rmtree(td, ignore_errors=True)


def chk_untouched():
    """no older profile carries the new hook, the P5-C profiles still read clean on their own templates and refuse the field flags, and refe_field / refe_field_ctrl are still P2 clones."""
    for nm, pr in L.PROFILES.items():
        if nm in NAMES.values():
            continue
        ok(not pr.get("p7field_rules") and "p7field_arm" not in pr, nm)
    for arm, nm in (("C1", "refe_p5c"), ("C0", "refe_p5c_c0"), ("C0r", "refe_p5c_c0r")):
        p = L.p5c_profile(arm)
        ok(L.PROFILES[nm]["name"] == nm and L.PROFILES[nm]["p5c_arm"] == arm and L.PROFILES[nm]["required_flags"]["--w-field"] == "0.0")
        ok(refusals(p, L.p5c_argv(arm)) == "", (nm, refusals(p, L.p5c_argv(arm))))
        ok("--field-labels" in refusals(p, L.p7field_argv("C")), "a P5-C arm refuses the field flags (G2: the template forbids them there)")
    ok(L.PROFILES["refe_field"]["field_arm"] == "F" and "G-P2GATE" in L.PROFILES["refe_field"]["checks"] and L.PROFILES["refe_field_ctrl"]["field_arm"] == "C")


def chk_no_scipy():
    src = (HERE / "test_field_p7_profile.py").read_text(encoding="utf-8")
    ok(not re.search(r"^\s*(import|from)\s+scipy", src, re.M))
    for m in re.finditer(r"subprocess\.run\(", src):
        depth, j = 0, m.end() - 1
        for k in range(j, len(src)):
            depth += src[k] == "("
            depth -= src[k] == ")"
            if depth == 0:
                break
        call = src[j:k + 1]
        ok("text=True" not in call or "encoding=" in call, "subprocess.run with text=True and no encoding=")


def chk_live_wiring():
    """G5: the gate-RUN side of the P5-C machinery applies to both field arms.  check_live hands the build capture to p5c_live_rows for F and C (P7 grafts) and never for
    refe_p7; p5c_live_rows judges (f) against the field profile's OWN base pin and (h) refuses a label table in the live bank, naming the arm; run_checks builds the capture
    for G-LIVE alone (need_b); G-CLOCK requires the bank content pin.  Stubs only (no model): the real (f)/(h) probes are test_p5c_live's."""
    from types import SimpleNamespace as NS
    import gate_checks as C
    import p5c_gate as PG
    seen = []
    orig = C.p5c_live_rows
    C.p5c_live_rows = lambda cap, bc, prof: (seen.append((prof["name"], bc)) or (["SENTINEL (f)"], {}))
    try:
        fake = NS(rc=0, metrics=[{"traj_L1": 0.1}], grads=[], args=NS(scorer_mode="fixed", labeller="bank", w_field=0.0, aux_drv=False), model=None, first_batch=None)
        for nm in NAMES.values():
            bc = object()
            ev = C.check_live([], fake, {**L.PROFILES[nm], "live_terms": ()}, build_cap=bc)
            ok(any("SENTINEL" in r for r in ev["reasons"]) and seen[-1] == (nm, bc) and ev["status"] == "FAIL", (nm, ev["reasons"]))
        n = len(seen)
        ev = C.check_live([], fake, {**L.PROFILES["refe_p7"], "live_terms": ()}, build_cap=object())
        ok(len(seen) == n and not any("SENTINEL" in r for r in ev["reasons"]), "refe_p7 (not a graft) never reaches the P5-C probes")
    finally:
        C.p5c_live_rows = orig
    saved = (PG.base_facts, PG.judge_warm_identity)
    pins = []
    PG.base_facts = lambda p: {"file_sha256": "b" * 64}
    PG.judge_warm_identity = lambda step0, base, pin: (pins.append(pin) or [])
    try:
        for arm, nm in NAMES.items():
            prof = L.PROFILES[nm]
            cap = NS(args=NS(init_from="base.pt"), bank_objs=[NS(traj_label={"k": 1}, traj_label_sha256="t" * 64)], model=None)
            rs, _ = C.p5c_live_rows(cap, NS(step0=None), prof)
            ok(any(f"(h) arm {arm} must not carry a label table" in r for r in rs), rs)
            ok(pins and pins[-1] == prof.get("init_from_sha256"), "(f) is judged against the field profile's OWN registered base pin")
            cap0 = NS(args=NS(init_from="base.pt"), bank_objs=[NS(traj_label=None, traj_label_sha256=None)], model=None)
            rs0, det0 = C.p5c_live_rows(cap0, NS(step0=None), prof)
            ok(rs0 == [] and det0.get("isolation") == "no table (control arm)", rs0)
    finally:
        PG.base_facts, PG.judge_warm_identity = saved
    src = (HERE / "launch_gate.py").read_text(encoding="utf-8")
    ok('    need_b = need_b or bool("G-LIVE" in want and prof.get("p7field_rules"))' in src, "run_checks builds the capture for G-LIVE alone on a field arm (need_b)")
    gsrc = (HERE / "gate_checks.py").read_text(encoding="utf-8")
    ok('required=bool(profile.get("p7_rules") or profile.get("p5c_rules") or profile.get("p7field_rules")))' in gsrc, "G-CLOCK requires the bank content pin for a field arm")


CHECKS = {"registry": chk_registry, "argv_delta": chk_argv_delta, "clean": chk_clean, "levers": chk_levers, "pins": chk_pins, "fresh_out": chk_fresh_out, "untouched": chk_untouched,
          "no_scipy": chk_no_scipy, "live_wiring": chk_live_wiring}

# (file, old, new, the check that must go RED): ONE emitting line of a COPY of this tree is neutralised.
MUTATIONS = [
    ("launch_gate.py", "    if prof.get(\"p7field_rules\"):\n        out += p7field_refusals(prof, argv)\n", "", "levers"),                                        # the field rules are not hooked into profile_refusals
    ("launch_gate.py", "    if prof.get(\"field_w\") != FIELD_P7_ARM_W[arm] or type(prof.get(\"field_w\")) is not float:\n", "    if False:\n", "pins"),            # a wrong registered weight is accepted
    ("launch_gate.py", "    if not _p5c_hex64(prof.get(\"field_store_sha256\")):\n", "    if False:\n", "pins"),                                        # an unregistered store pin is accepted
    ("launch_gate.py", "        if got != prof[\"field_loss_sha256\"]:\n", "        if False:\n", "pins"),                                              # the shipped field_loss.py is not re-hashed
    ("launch_gate.py", "            if (AUXH.PINNED_STORE_SHA256, AUXH.PINNED_STORE_KEYS) != (prof[\"field_store_sha256\"], prof.get(\"field_store_keys\")):\n", "            if False:\n", "pins"),   # the loader's pin is not compared
    ("launch_gate.py", "    for f in sorted(set(got_) - set(want) - {\"--resume\"}):\n", "    for f in []:\n", "levers"),                              # an extra, unlisted lever is accepted
    ("launch_gate.py", "        elif f in want and f not in FIELD_P7_PATH_FLAGS and f != \"--resume\":\n", "        elif False:\n", "levers"),                     # a moved non-path lever is accepted
    ("launch_gate.py", "    for f in sorted(set(want) - set(got_) - {\"--resume\"}):\n", "    for f in []:\n", "levers"),                              # a dropped lever (--field-labels, margin) is not named by the template
    ("launch_gate.py", "    a[i + 1] = repr(float(FIELD_P7_ARM_W[arm]))\n", "    a[i + 1] = \"0.0\"\n", "argv_delta"),                                        # the template's F has w 0.0 (F would be a second control)
    ("launch_gate.py", "FIELD_P7_ARM_W = {\"F\": FIELD_P7_W, \"C\": 0.0}", "FIELD_P7_ARM_W = {\"F\": FIELD_P7_W, \"C\": FIELD_P7_W}", "argv_delta"),             # C carries the weight
    ("launch_gate.py", "\nFIELD_P7_W = 2.0 ", "\nFIELD_P7_W = 1.0 ", "registry"),                                        # the ratified weight drifts
    ("launch_gate.py", "(\"--field-shuffle-seed\",) if \"--field-shuffle-seed\" not in c0[\"forbidden_flags\"] else ()", "()", "registry"),              # the shuffled-label arm is no longer forbidden
    ("launch_gate.py", "\"--field-labels\": None, **FIELD_REGISTERED})", "\"--field-labels\": None})", "registry"),                                      # the margin / interp are no longer REQUIRED flags
    ("launch_gate.py", "list(c0[\"suite\"]) + FIELD_SUITE_EXTRA + [", "list(c0[\"suite\"]) + [", "registry"),                                          # test_field_loss.py leaves G-SUITE
    ("launch_gate.py", "\"live_terms\": P6_TERMS + (\"field_L\",)", "\"live_terms\": P6_TERMS", "registry"),                                           # field_L is not a declared live term
    ("gate_checks.py", "    if profile.get(\"p5c_rules\") or profile.get(\"p7field_rules\"):", "    if profile.get(\"p5c_rules\"):", "live_wiring"),            # G-LIVE never runs (f)/(h) on a field arm
    ("gate_checks.py", "{profile.get('p5c_arm') or profile.get('p7field_arm')} must not carry", "{profile.get('p5c_arm')} must not carry", "live_wiring"),   # (h) no longer names the field arm
    ("gate_checks.py", " or profile.get(\"p5c_rules\") or profile.get(\"p7field_rules\")))", " or profile.get(\"p5c_rules\")))", "live_wiring"),         # G-CLOCK skips the content pin
    ("launch_gate.py", "    need_b = need_b or bool(\"G-LIVE\" in want and prof.get(\"p7field_rules\"))", "    need_b = need_b", "live_wiring"),                  # no build capture for G-LIVE alone
    ("launch_gate.py", "        out.append(f\"{name}: --out {ow} already holds ckpt_last.pt: a field arm starts FRESH (fresh AdamW; a seeded checkpoint would resume an old optimiser)\")", "        pass", "fresh_out"),
]


def run_mutations() -> None:
    for fn, old, new, check in MUTATIONS:
        td = Path(tempfile.mkdtemp(prefix="fp7mut_"))
        try:
            for f in HERE.glob("*.py"):
                shutil.copyfile(f, td / f.name)
            for f in HERE.glob("p7_pod.sh"):
                shutil.copyfile(f, td / f.name)
            src = (td / fn).read_text(encoding="utf-8")
            ok(src.count(old) == 1, f"mutation target not found exactly once in {fn}: {old[:70]!r}")
            (td / fn).write_text(src.replace(old, new), encoding="utf-8", newline="\n")
            env = dict(os.environ, PYTHONIOENCODING="utf-8")
            r = subprocess.run([sys.executable, str(td / "test_field_p7_profile.py"), "--check", check], cwd=str(td), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
            ok(r.returncode != 0, f"MUTATION SURVIVED: {fn}: {old[:60]!r} -> check {check} still passed")
            print(f"  mutation RED as required: {fn} [{old.strip()[:56]}] -> {check}", flush=True)
        finally:
            shutil.rmtree(td, ignore_errors=True)


def main() -> int:
    if "--check" in sys.argv:
        name = sys.argv[sys.argv.index("--check") + 1]
        CHECKS[name]()
        print(f"CHECK_OK {name} {N}")
        return 0
    for name, fn in CHECKS.items():
        fn()
        print(f"  {name}: ok ({N})", flush=True)
    if "--fast" not in sys.argv:
        run_mutations()
    print(f"ZZFIELDP7PROF_PASS {N}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
