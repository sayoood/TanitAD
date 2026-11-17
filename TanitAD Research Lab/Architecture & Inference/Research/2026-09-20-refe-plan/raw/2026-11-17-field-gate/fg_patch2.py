"""MM 2026-11-17: close gap G5 for the P7 field arms -- the gate-RUN side of the P5-C machinery (G-LIVE (f) warm-start identity + (h) no table, the build
capture for G-LIVE, G-CLOCK's required bank-content pin) now applies to `p7field_rules` too.  Same byte edits to the scratch export and the land copies;
every P5-C-pinned line stays byte-identical (test_p5c_live pins the need_b line and mutates the p5c_live_rows call line).  Each edit asserts its old text
occurs exactly once."""
import os, subprocess, sys

PKR = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe"
EXP = "F:/tanitad_scratch/fieldgate/exp/" + PKR
LAND = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/field_p7/gate/land/" + PKR
M = "C:/Users/Admin/tanitad-push"
TIP = sys.argv[1]

LIVE_TEST = '''

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
'''

EDITS = {
    "gate_checks.py": [
        ('    if profile.get("p5c_rules"):\n        rs_, dt_ = p5c_live_rows(cap, build_cap, profile)',
         '    if profile.get("p5c_rules") or profile.get("p7field_rules"):        # P5-C arms AND the P7 field arms F/C (P7 grafts: the same (f) warm-start identity; (h) a non-C1 arm carries no table)\n        rs_, dt_ = p5c_live_rows(cap, build_cap, profile)'),
        ("{profile.get('p5c_arm')} must not carry a label table", "{profile.get('p5c_arm') or profile.get('p7field_arm')} must not carry a label table"),
        ('required=bool(profile.get("p7_rules") or profile.get("p5c_rules")))', 'required=bool(profile.get("p7_rules") or profile.get("p5c_rules") or profile.get("p7field_rules")))'),
    ],
    "launch_gate.py": [
        ('or bool("G-LIVE" in want and prof.get("p5c_rules"))        # P5-C: G-LIVE (f) reads the build capture\'s step-0 digest\n',
         'or bool("G-LIVE" in want and prof.get("p5c_rules"))        # P5-C: G-LIVE (f) reads the build capture\'s step-0 digest\n'
         '    need_b = need_b or bool("G-LIVE" in want and prof.get("p7field_rules"))      # the P7 field arms: G-LIVE (f) reads the same step-0 digest (G5)\n'),
        ("ONLY the argv-refusal side is wired: the gate-RUN side of the\n# P5-C machinery (gate_checks.p5c_live_rows warm-start identity, need_b, clock_bank_sha256 required=) keys on `p5c_rules` and is NOT enabled for these profiles (G5).",
         "G5's gate-RUN side IS wired (MM 2026-11-17): gate_checks.check_live routes\n# `p7field_rules` to p5c_live_rows ((f) warm-start identity, (h) no table), run_checks builds the capture for G-LIVE (need_b), G-CLOCK requires the bank content pin; chk_live_wiring pins all three."),
    ],
    "test_field_p7_profile.py": [
        ("(PREREG_AUX_FIELD_LOSS_P7_DRAFT; gaps G1, G2, G6).", "(PREREG_AUX_FIELD_LOSS_P7_DRAFT; gaps G1, G2, G5, G6)."),
        ("\n\nCHECKS = {", LIVE_TEST + "\n\nCHECKS = {"),
        ('          "no_scipy": chk_no_scipy}', '          "no_scipy": chk_no_scipy, "live_wiring": chk_live_wiring}'),
        ('    ("launch_gate.py", "        out.append(f\\"{name}: --out {ow} already holds ckpt_last.pt',
         '    ("gate_checks.py", "    if profile.get(\\"p5c_rules\\") or profile.get(\\"p7field_rules\\"):", "    if profile.get(\\"p5c_rules\\"):", "live_wiring"),            # G-LIVE never runs (f)/(h) on a field arm\n'
         '    ("gate_checks.py", "{profile.get(\'p5c_arm\') or profile.get(\'p7field_arm\')} must not carry", "{profile.get(\'p5c_arm\')} must not carry", "live_wiring"),   # (h) no longer names the field arm\n'
         '    ("gate_checks.py", " or profile.get(\\"p5c_rules\\") or profile.get(\\"p7field_rules\\")))", " or profile.get(\\"p5c_rules\\")))", "live_wiring"),         # G-CLOCK skips the content pin\n'
         '    ("launch_gate.py", "    need_b = need_b or bool(\\"G-LIVE\\" in want and prof.get(\\"p7field_rules\\"))", "    need_b = need_b", "live_wiring"),                  # no build capture for G-LIVE alone\n'
         '    ("launch_gate.py", "        out.append(f\\"{name}: --out {ow} already holds ckpt_last.pt'),
    ],
}

for fn, edits in EDITS.items():
    lp = os.path.join(LAND, fn)
    if not os.path.exists(lp):                                   # raw tip blob (no autocrlf) into the land set
        b = subprocess.run(["git", "-C", M, "cat-file", "blob", f"{TIP}:{PKR}/{fn}"], capture_output=True, check=True).stdout
        open(lp, "wb").write(b)
    for root in (EXP, LAND):
        p = os.path.join(root, fn)
        s = open(p, "rb").read().decode("utf-8")
        for old, new in edits:
            if s.count(new) == 1 and s.count(old) == (1 if old in new else 0):
                continue                                         # already applied
            assert s.count(old) == 1, (p, s.count(old), old[:80])
            s = s.replace(old, new)
        open(p, "wb").write(s.encode("utf-8"))
    a, b = (open(os.path.join(r, fn), "rb").read() for r in (EXP, LAND))
    print(fn, "export==land" if a == b else "EXPORT!=LAND")
print("ZZFGPATCH2_OK")
