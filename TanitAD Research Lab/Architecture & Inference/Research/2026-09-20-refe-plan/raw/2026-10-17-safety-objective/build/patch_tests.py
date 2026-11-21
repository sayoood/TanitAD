#!/usr/bin/env python3
"""patch_tests: the edits to the EXISTING tests that SAFEOBJ's trainer / gate / chain changes force.  usage: patch_tests.py <PK dir>  (applied AFTER patch_trainer/patch_gate/patch_chain)

Each edit is an exact anchor occurring exactly once (patchlib.apply_edits).  The aim is that every pinned test keeps catching its own mutations: the SAFEOBJ block of main.train_losses is the
ONE edit inside a pinned R0 segment, and it is reversed (test_rl_s1b.SAFEOBJ_REVERSALS / undo_safeobj) BEFORE R0c's, so the R0 sha256 pins still hold byte for byte.
"""
import os
import sys

import patchlib as PL
from patch_trainer import SEG_TRAIN_LOSSES_NEW

S1B_EDITS = [
    # the segment-level reversal (dedented, as S1B.segments sees main.train_losses)
    ('R0_REWARD_GIT_BLOB = "cd5c51445ad4ce38153d72ceab0aa01226aa7372"',
     '# SAFEOBJ (raw/2026-10-17-safety-objective): the constrained-EPO block (`if dual.active:` ...) is the ONE edit inside a pinned segment (main.train_losses).  The pins above are STILL R0\'s bytes:\n'
     '# P_pinned_* reverses SAFEOBJ\'s block FIRST (undo_safeobj), then R0c\'s edit (undo_r0c).  refe/test_rl_safeobj.py proves the block is the whole diff of the trainer (the reversed FILE is the 7c8071ea blob).\n'
     'SAFEOBJ_REVERSALS = {\n'
     '    "main.train_losses": ((' + repr(SEG_TRAIN_LOSSES_NEW) + ', ""),),\n'
     '}\n'
     '\n'
     '\n'
     'def undo_safeobj(name: str, text: str) -> str:\n'
     '    for new, old in SAFEOBJ_REVERSALS.get(name, ()):\n'
     '        assert text.count(new) == 1, f"SAFEOBJ edit of {name} not found exactly once"\n'
     '        text = text.replace(new, old)\n'
     '    return text\n'
     '\n'
     '\n'
     'R0_REWARD_GIT_BLOB = "cd5c51445ad4ce38153d72ceab0aa01226aa7372"'),
    ('        check(bad, f"P_pinned_{k}", lambda k=k, h=h: hashlib.sha256(undo_r0c(k, seg[k]).encode()).hexdigest() == h)',
     '        check(bad, f"P_pinned_{k}", lambda k=k, h=h: hashlib.sha256(undo_r0c(k, undo_safeobj(k, seg[k])).encode()).hexdigest() == h)'),
]

R0C_EDITS = [
    ('import test_rl_s1b as S1B  # noqa: E402  (the R0 segment pins, load / segments / git_blob / check)\n',
     'import test_rl_s1b as S1B  # noqa: E402  (the R0 segment pins, load / segments / git_blob / check)\n'
     'import test_rl_safeobj as SO  # noqa: E402  (SAFEOBJ, raw/2026-10-17-safety-objective: the trainer tip is the SAFEOBJ trainer; its reversal is the FIRST link of this chain, 7c8071ea -> d34d5cf6)\n'),
    ('CHANGED_FUNCTIONS = {"epo_refusals", "build_parser", "main", "main.train_losses"}              # every function whose text differs from the pre-patch trainer\n',
     'CHANGED_FUNCTIONS = {"epo_refusals", "build_parser", "main", "main.train_losses"} | SO.CHANGED_FUNCTIONS              # every function whose text differs from the pre-patch trainer (+ SAFEOBJ\'s)\n'),
    ('NEW_FUNCTIONS = {"epo_reference", "epo_reference_built"}\n', 'NEW_FUNCTIONS = {"epo_reference", "epo_reference_built"} | SO.NEW_FUNCTIONS\n'),
    ('    """the patched trainer -> the pre-patch trainer (every edit reversed; each must occur exactly once)"""\n    for new, old in R0C_FILE_REVERSALS:\n',
     '    """the patched trainer -> the pre-patch trainer (every edit reversed; each must occur exactly once).  SAFEOBJ\'s edits first: the tip is the SAFEOBJ trainer, whose reversal is the R0c tip (7c8071ea)"""\n'
     '    src = SO.reverse_edits(src)\n'
     '    for new, old in R0C_FILE_REVERSALS:\n'),
    ('lambda k=k: hashlib.sha256(S1B.undo_r0c(k, seg[k]).encode()).hexdigest() == S1B.R0_SEGMENTS[k])',
     'lambda k=k: hashlib.sha256(S1B.undo_r0c(k, S1B.undo_safeobj(k, seg[k])).encode()).hexdigest() == S1B.R0_SEGMENTS[k])'),
    ('terms_train=TERMS_D, ds=DS, epo_last={}, fwd=None)\n\n    def fresh(n_out):',
     'terms_train=TERMS_D, ds=DS, epo_last={}, fwd=None,\n'
     '                                  dual=M.EpoDual((), 0.0, 4.0, 10.0, 0.0), con_last={})          # SAFEOBJ: train_losses reads `dual` (OFF here = this patch\'s trainer exactly)\n'
     '\n    def fresh(n_out):'),
    ('            if r.get("event") == "data" and isinstance(r.get("epo"), dict):\n',
     '            if r.get("event") == "data":                                  # SAFEOBJ: the data event gained the `epo_con` block (the pre-R0c trainer has none)\n'
     '                r = {k: v for k, v in r.items() if k != "epo_con"}\n'
     '            if r.get("event") == "data" and isinstance(r.get("epo"), dict):\n'),
    ('        return set(nm) - {tuple(x.option_strings) for x in old_p._actions} == {("--epo-ref",)}\n',
     '        return set(nm) - {tuple(x.option_strings) for x in old_p._actions} == {("--epo-ref",)} | {(f,) for f in SO.NEW_FLAGS}          # SAFEOBJ: old_src is the PRE-R0c trainer, so the five --epo-con* flags join --epo-ref as the additions\n'),
]

EPO_EDITS = [
    ('terms_train=TERMS, ds=DS, epo_last={}, fwd=None)\n\n    def one_microbatch(n_out):',
     'terms_train=TERMS, ds=DS, epo_last={}, fwd=None,\n'
     '                                         dual=C.mod.EpoDual((), 0.0, 4.0, 10.0, 0.0), con_last={})          # SAFEOBJ: train_losses reads `dual` (OFF = the pre-SAFEOBJ assembly)\n'
     '\n    def one_microbatch(n_out):'),
    ('            if isinstance(n, ast.Assign) and getattr(n.value, "func", None) is not None and getattr(n.value.func, "id", "") == "epo_refusals":\n',
     '            v = n.value.left if isinstance(n, ast.Assign) and isinstance(n.value, ast.BinOp) else getattr(n, "value", None)          # SAFEOBJ: `bad_cli = epo_refusals(a, ap) + epo_con_refusals(a, ap)`\n'
     '            if isinstance(n, ast.Assign) and getattr(v, "func", None) is not None and getattr(v.func, "id", "") == "epo_refusals":\n'),
]

LG_R11 = '''    # ------------------------------------------------------------------ R11 SAFEOBJ (refe_rls1d, raw/2026-10-17-safety-objective): the arms R0d / R1d / R1o (the full matrix and its mutations: refe/test_rl_safeobj.py)
    Pd = L.PROFILES["refe_rls1d"]
    CON = ["--epo-con", "g5", "--epo-con-eps", "0", "--epo-con-lr", "4", "--epo-con-max", "10", "--epo-con-lam0", "0"]
    r1_ = C._set(r0, "--seed", "1")
    r0d, r1d, r1o = r0 + CON, r1_ + CON, r1_ + ["--epo-con", "off"]
    redd = lambda argv, frag, arm: red(argv, frag, arm, {**Pd, "declared_arm": arm})          # noqa: E731
    for argv_, arm_ in ((r0d, "R0d"), (r1d, "R1d"), (r1o, "R1o")):
        ok(refs(argv_, arm_, {**Pd, "declared_arm": arm_}) == [], arm_)
    for fl in ("--epo-con", "--epo-con-eps", "--epo-con-lr", "--epo-con-max", "--epo-con-lam0"):                 # a parser DEFAULT never carries a lever: each flag must be NAMED
        redd(without(r0d, fl), f"requires {fl} NAMED", "R0d")
        redd(without(r1d, fl), f"requires {fl} NAMED", "R1d")
    redd(r0, "requires --epo-con g5", "R0d")                                                                     # R0's own argv (the default `off`) is not R0d
    redd(C._set(r0d, "--seed", "1"), "registered with --seed 0", "R0d")
    redd(C._set(r1d, "--seed", "0"), "registered with --seed 1", "R1d")
    redd(C._set(r1o, "--seed", "0"), "registered with --seed 1", "R1o")
    redd(C._set(r1o, "--epo-con", "g5"), "requires --epo-con off", "R1o")                                        # the lever-off control must not carry the lever
    redd(without(r1o, "--epo-con"), "requires --epo-con NAMED", "R1o")
    redd(C._set(r0d, "--epo-con-lr", "2"), "requires --epo-con-lr 4.0", "R0d")
    for nm_, arm_, argv_ in (("refe_rls1", "R0", r0), ("refe_rls1b", "R0b", r0 + ["--epo-lam", "0.25"]), ("refe_rls1c", "R0c", r0 + ["--epo-ref", "armA"])):          # every older profile refuses the lever ...
        red(argv_ + CON, "requires --epo-con off", arm_, {**L.PROFILES[nm_], "declared_arm": arm_})
        ok(refs(argv_ + ["--epo-con", "off"], arm_, {**L.PROFILES[nm_], "declared_arm": arm_}) == [], nm_)         # ... and still accepts the switch NAMED off
    ok([e["script"] for e in Pd["suite"]][-1] == "test_rl_safeobj.py" and Pd["suite"][:-1] == L.PROFILES["refe_rls1c"]["suite"])                  # M8: the suite carries the new test
    ok(all(C.DVB_KINDS_SCORER.get(k) == "built" for k in ("epo_con", "epo_con_eps", "epo_con_lr", "epo_con_max", "epo_con_lam0")))               # M7: the five flags are classified
'''

LG_EDITS = [
    ('    ev = cons(r0, add={"G13_something_new": {"ok": True}})\n',
     '    ev = cons(r0, add={"G14_something_new": {"ok": True}})                                                   # (G13 is a KNOWN gate since SAFEOBJ: the unknown-gate fixture moves to G14)\n'),
    ('        if k.startswith(("G8_", "G12_")):                          # r0 carries no lane labels / no --aux-drv: G8 / G12 are not obliged, so such a flag in the fixture is simply not written\n',
     '        if k.startswith(("G8_", "G12_", "G13_")):                  # r0 carries no lane labels / no --aux-drv / no --epo-con: G8 / G12 / G13 are not obliged, so such a flag in the fixture is simply not written\n'),
    ('ev["details"]["observed_rows"] == 15 and ev["details"]["unknown_flags"] == [] and ev["details"]["stale_registry"] == [], ev["reasons"])      # 14 -> 15: R0c\'s epo.ref row\n',
     'ev["details"]["observed_rows"] == 23 and ev["details"]["unknown_flags"] == [] and ev["details"]["stale_registry"] == [], ev["reasons"])      # 14 -> 15: R0c\'s epo.ref row; 15 -> 23: SAFEOBJ\'s eight epo_con rows\n'),
    ('    ok(C.check_dvb_scorer(r0, pf_ok, P0)["status"] == "PASS")                                                # the arms left no residue\n',
     '    ok(C.check_dvb_scorer(r0, pf_ok, P0)["status"] == "PASS")                                                # the arms left no residue\n' + LG_R11),
]

CHAIN_EDITS = [
    ('    known = {"C0", "C1", "R0", "R1", "R2", "R0b", "R1b", "R0c", "*"}',
     '    known = {"C0", "C1", "R0", "R1", "R2", "R0b", "R1b", "R0c", "R0d", "R1d", "R1o", "*"}'),
    ('sw(\'REGISTERED_ARMS="C0 C1 R0 R1 R2 R0b R1b R0c"\', \'REGISTERED_ARMS="C0 C1 R0 R1 R2 R0c"\')',
     'sw(\'REGISTERED_ARMS="C0 C1 R0 R1 R2 R0b R1b R0c R0d R1d R1o"\', \'REGISTERED_ARMS="C0 C1 R0 R1 R2 R0c R0d R1d R1o"\')'),
    ('    regs = \'REGISTERED_ARMS="C0 C1 R0 R1 R2 R0b R1b R0c"\'\n',
     '    regs = \'REGISTERED_ARMS="C0 C1 R0 R1 R2 R0b R1b R0c R0d R1d R1o"\'\n'),
    ('sw(" refe/test_rl_s1b.py refe/test_rl_r0c.py \\\\", " \\\\")', 'sw(" refe/test_rl_s1b.py refe/test_rl_r0c.py refe/test_rl_safeobj.py \\\\", " \\\\")'),
    ('"R0c,", "R0d", "c",', '"R0c,", "R0dd", "c",'),          # R0d is a REGISTERED arm since SAFEOBJ (raw/2026-10-17-safety-objective); its misspellings are refused in refe/test_rls1_queue.py (B4) and refe/test_rl_safeobj.py (CH)
]

QUEUE_DOC = '''  B4. SAFEOBJ (raw/2026-10-17-safety-objective): the whole arm loop run for "R0d R1d R1o": the profile refe_rls1d, the argv the preflight and the trainer receive (--epo-con g5 ... / --epo-con off), each arm
     read against its OWN A, the seed-matched second control, R1o's --observe-only monitor AND replay, R1o refused until R0d and R1d both read S = PASS (the real rls1_checks.py safeobj-gate), a K3 kill of
     R0d / R1d that STOPS the chain, a killed R0d that refuses on a re-run, misspelled arms; and the mutations of each.
'''

QUEUE_EDITS = [
    ('  B3. RL-S1c: the whole arm loop run for "R0c":', QUEUE_DOC + '  B3. RL-S1c: the whole arm loop run for "R0c":'),
    ('      updates-check) echo "ZZRLS1_UPDATES_OK 1000 updates"; exit 0 ;;\n',
     '      safeobj-gate) exec "${FAKE_REAL_PY:-python3}" "${FAKE_REAL_RC:-rls1_checks.py}" "$@" ;;          # SAFEOBJ: the REAL gate (rls1_checks.py safeobj-gate) behind the fake python\n'
     '      updates-check) echo "ZZRLS1_UPDATES_OK 1000 updates"; exit 0 ;;\n'),
    ('    case "${FAKE_TRAINER:-finish}" in\n',
     '    tm_var="FAKE_TRAINER_$(basename "$ad")"      # a per-arm override (SAFEOBJ: R0d finishes while R1d is killed)\n'
     '    case "${!tm_var:-${FAKE_TRAINER:-finish}}" in\n'),
    ('"RLS1_MON_READY_S": "60", "HF_TOKEN": "hf_notarealtoken", "RLS1_ARMS": arms, **(env or {})}',
     '"RLS1_MON_READY_S": "60", "HF_TOKEN": "hf_notarealtoken", "RLS1_ARMS": arms,\n'
     '         "FAKE_REAL_PY": posix(sys.executable), "FAKE_REAL_RC": posix(HERE / "rls1_checks.py"), **(env or {})}'),
    ('def synth_scores(path, n, m=8, c=7, dup=False, space="logit"):', None),                # replaced below by (suite text + the anchor)
    ('        n += rls1c_all(text, tmp)\n        # 2026-10-15: rl_val_eval.load_scores',
     '        n += rls1c_all(text, tmp)\n        n += rls1d_all(text, tmp)\n        # 2026-10-15: rl_val_eval.load_scores'),
]


def _queue_edits(build_dir):
    suite = open(os.path.join(build_dir, "queue_suite.txt"), encoding="utf-8").read()
    out = []
    for old, new in QUEUE_EDITS:
        if new is None:
            new = suite + old
        out.append((old, new))
    return out


DUMP_EDITS = [
    ('        epo_dests = {"epo_ref"}\n',
     '        epo_dests = {"epo_ref"} | {"epo_con", "epo_con_eps", "epo_con_lr", "epo_con_max", "epo_con_lam0"}          # SAFEOBJ (raw/2026-10-17-safety-objective): five more default-OFF dests, ALL-or-none like the others; "off" is every pre-SAFEOBJ arm bit for bit\n'),
    ('and getattr(ns_, "epo_ref", "base") == "base", str(sorted(dests - set(BASE_DESTS))))',
     'and getattr(ns_, "epo_ref", "base") == "base" and getattr(ns_, "epo_con", "off") == "off", str(sorted(dests - set(BASE_DESTS))))'),
]

AUXWIRE_EDITS = [
    ('    if r.get("event") == "data" and isinstance(r.get("epo"), dict):          # R0c:',
     '    if r.get("event") == "data" and "epo_con" in r:          # SAFEOBJ (raw/2026-10-17-safety-objective): the data event gained the `epo_con` block (the pre-SAFEOBJ trainer has none); every other byte is unchanged\n'
     '        r = {k: v for k, v in r.items() if k != "epo_con"}\n'
     '    if r.get("event") == "data" and isinstance(r.get("epo"), dict):          # R0c:'),
]

TARGETS = [("test_rl_s1b.py", S1B_EDITS), ("test_rl_r0c.py", R0C_EDITS), ("test_rl_epo.py", EPO_EDITS), ("test_launch_gate.py", LG_EDITS),
           ("test_rls1_chain.py", CHAIN_EDITS), ("test_rls1_queue.py", None),
           ("test_dump_scores.py", DUMP_EDITS), ("test_aux_s1_wiring.py", AUXWIRE_EDITS)]


def main():
    pk = PL.pk_arg()
    only = set(sys.argv[2:])                                  # optional: patch just these files (they must be pristine); default = all
    for fn, edits in TARGETS:
        if only and fn not in only:
            continue
        if edits is None:
            edits = _queue_edits(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(pk, "refe", fn)
        src = PL.read(path)
        out, _ = PL.apply_edits(src, edits, fn)
        PL.write(path, out)
        print({"file": fn, "blob_before": PL.git_blob(src), "blob_after": PL.git_blob(out), "edits": len(edits)})


if __name__ == "__main__":
    main()
