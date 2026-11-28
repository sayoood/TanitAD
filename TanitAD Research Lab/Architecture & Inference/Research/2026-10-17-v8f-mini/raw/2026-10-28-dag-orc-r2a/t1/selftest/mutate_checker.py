"""Mutation proof of the NEW K-BUILD clauses: delete ONE clause of the lib's fix-mode checker (text mutation, in memory, the file on disk is untouched) and require that the matching deliberate-regression
scenario of dagorct1_selftest.py then ESCAPES (the record reads all-PASS = the selftest's NEG check would have gone RED).  A control (no mutation) must still be refused.
python -I mutate_checker.py > mutate_checker.out"""
import importlib.util
import sys
import types

HERE = "E:/Projects/TanitAD-artifacts/dagorc_t1"
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
import dagorct1_selftest as T  # noqa: E402

SRC = open(f"{HERE}/dagorct1_lib.py", encoding="utf-8").read()
tr, dt, tt = "1c88828094d40a2b5e3b5dce45b9fc110286c0b01db3a224faa205138e5c27bc", "c117e1cf5be08b9739f02e2730ce555be5a707a3183ded6481256e419f751752", "faad3be21e1bd4d673352c6fe793d271385839cfa2612c1b1a95e71754a34999"
FR, FL, R1, GL, GR = T.FIX_RES, T.FIX_LOG, T.R1_RES, T.GOLD_LAUNCH, T.GOLD_R1


def load(src, name):
    m = types.ModuleType(name)
    m.__file__ = f"{HERE}/dagorct1_lib.py"
    exec(compile(src, m.__file__, "exec"), m.__dict__)
    return m


# (id, what is deleted, exact text in the lib, replacement, scenario patches)
MUTANTS = [
    ("MC1_golden_clause_deleted", "the golden launch==r1 comparison", 'if gl is None or g1 is None or gl != g1:', 'if False:', {GL: lambda t: t + " "}),
    ("MC2_launch_bytes_not_compared", "the listed sha == launch blob comparison", 'elif lb is None or listed.get(f) != lb:', 'elif False:', {FR: lambda t: t.replace(tr, "0" * 64)}),
    ("MC3_test_file_not_required", "the compose test file's listed sha is not required", 'for f in sorted(set(differing) | {R1_COMPOSE_TEST} | set(listed) | set(conflicts)):', 'for f in sorted(set(differing) | set(listed) | set(conflicts)):',
     {FR: lambda t: "\n".join(x for x in t.split("\n") if tt not in x)}),
    ("MC4_summary_not_cross_checked", "the SUMMARY vs per-mutant-lines cross-check", 'elif len(mut) != int(mm.group(2)) or len(killed) != len(mut):', 'elif False:',
     {FR: lambda t: t.replace("M05_loop_never_composes: RED (killed)", "M05_loop_never_composes: SURVIVED")}),
    ("MC5_m13_not_required", "the M13 requirement", 'if not re.search(r"^M13_off_path_changed: RED \\(killed\\)", fix_results, re.M):', 'if False:',
     {FR: lambda t: "\n".join(x for x in t.split("\n") if not x.startswith("M13_off_path_changed")).replace("SUMMARY: 27/27", "SUMMARY: 26/26")}),      # M13 ABSENT, SUMMARY consistent: the per-mutant cross-check cannot see it
    ("MC6_controls_not_checked", "the unmutated-controls-GREEN clause", 'if not ctl or any(c != "0" for c in ctl):', 'if False:',
     {FR: lambda t: t.replace("-k 'flag_off_reproduces': rc=0", "-k 'flag_off_reproduces': rc=1")}),
    ("MC7_log_failure_not_checked", "the suite log's failure regex", 'if re.search(r"\\b\\d+ (failed|errors?)\\b", fix_log):', 'if False:',
     {FL: lambda t: t.replace("36 passed in 316.41s (0:05:16)", "36 passed, 1 error in 316.41s (0:05:16)")}),      # still ENDS in 'N passed': only the failure / error regex can see it
    ("MC8_log_must_not_end_in_passed_dropped", "the 'log ENDS in >= 26 passed' clause", 'if n_log is None or n_log < MIN_COMPOSE_TESTS:\n        why.append', 'if False:\n        why.append',
     {FL: lambda t: t.split("\n")[0]}),
    ("MC9_golden_writer_not_compared", "the golden-writer bytes == base blob comparison", 'if len(u) < 16 or not files[f]["sha256_base"].startswith(u):', 'if False:', {R1: lambda t: t.replace("30819a6fb947b2f4", "1111111111111111")}),
    ("MC10_conflicts_ignored", "an ambiguous (twice-listed, different sha) path", 'if f in conflicts:\n            problems[f"fix:launch_bytes:{f}"] = f"listed with different', 'if False:\n            problems[f"fix:launch_bytes:{f}"] = f"listed with different', None, "ambiguous"),
    ("MC11_flag_check_dropped", "the default-off flag check (shared tail)", 'if not default_off:\n            problems["flag"]', 'if False:\n            problems["flag"]',
     {T.TRAIN_LAUNCH: (lambda t: t.replace("default=False", "default=True"), 1)}, "flag:"),      # the hunk's default_off is False either way (the reader refuses on that), so the criterion is the NAMED reason
]
bad = []
# CONTROL: the unmutated lib refuses every scenario (the selftest's NEG arms); here only the control record: all PASS without patches
ctl = load(SRC, "lib_ctl")
T.L = ctl
real_R = None
spec = importlib.util.spec_from_file_location("dag_orc_read", f"{T.RD}/dag_orc_read.py")
R_ = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R_)


def run(mod, patches):
    T.L = mod
    with T.patched(patches or {}):
        kb = mod.kbuild("unused", T.MIRROR, T.FIX_LAUNCH, None, T.FIX_LAUNCH, T.FIXPKG)
    return kb, R_.k_build(kb)[0]


kb0, ok0 = run(ctl, None)
print(f"CONTROL (no mutation, no scenario): reader accepts = {ok0}, hunks {len(kb0['changed'])}, all PASS = {all(h['bit_identity_test'] == 'PASS' for h in kb0['changed'])}")
for mid, what, old, new, patches, *rest in MUTANTS:
    must = rest[0] if rest else None
    assert SRC.count(old) == 1, (mid, SRC.count(old))
    m = load(SRC.replace(old, new), "lib_" + mid)
    if patches is None:      # MC10: the ambiguous scenario
        patches = {FR: lambda t: t.replace(f"{tr}  stack/scripts/v8f_mini_train.py\n", f"{tr}  stack/scripts/v8f_mini_train.py\n{'0' * 64}  stack/scripts/v8f_mini_train.py\n")}
    kc, okc = run(ctl, patches)      # the scenario on the UNMUTATED lib: must be refused
    km, okm = run(m, patches)        # the scenario on the MUTATED lib: must ESCAPE (the selftest's NEG check would go RED)
    txt = lambda k: " ".join(h["bit_identity_test"] for h in k["changed"])
    if must:      # a clause whose deletion does not change the verdict (the generic mismatch still refuses) but DOES change the named reason
        killed = (okc is False) and (must in txt(kc)) and (must not in txt(km))
    else:
        killed = (okc is False) and (okm is True)
    print(f"{mid}: scenario on the unmutated lib -> reader accepts={okc} (must be False); with [{what}] deleted -> reader accepts={okm} (must be True = escapes{', or its reason ' + repr(must) + ' lost' if must else ''})  => {'KILLED by the selftest' if killed else 'NOT KILLED'}")
    if not killed:
        bad.append(mid)
T.L = ctl
print(f"SUMMARY: {len(MUTANTS) - len(bad)}/{len(MUTANTS)} checker mutants killed; survivors: {', '.join(bad) or 'none'}")
sys.exit(1 if bad else 0)
