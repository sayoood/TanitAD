"""Mutation run of the FIFTH pin (`pin5/read_t2_dominance_pin5.py`): each mutation breaks one wire or deletes one guard in a copy of the source held IN MEMORY (the file on disk is never touched) and the designated
test of `pin5/tests/test_pin5.py` must go RED (any exception). An anchor that does not occur exactly once is an error, so a rewording cannot turn a mutation into a no-op. The unmutated module is run first: every test
must be green.

Run from t2v8f/:  CUDA_VISIBLE_DEVICES=-1 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I pin5/run_mutations_pin5.py     (ends with `mutations RED: N of M` and `ZZ PIN5-MUT-DONE ZZ`; rc 0 only if all are RED)
Environment: PIN5_MUT_SCRATCH = the scratch dir (default C:/Users/Admin/pytest_tmp_pin5/mut_run; its parent must exist)."""
import inspect
import os
import pathlib
import shutil
import sys
import types

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, HERE + "/tests")
import pytest  # noqa: E402

import test_pin5 as TP  # noqa: E402  (binds `read_t2_dominance` to the pin-5 module and imports the frozen helper modules)

ORIG = TP.RD
SRC_PATH = ORIG.__file__
SRC = open(SRC_PATH, encoding="utf-8", newline="").read()
SCRATCH = pathlib.Path(os.environ.get("PIN5_MUT_SCRATCH", "C:/Users/Admin/pytest_tmp_pin5/mut_run"))
HOLDERS = (TP, TP.TQ, TP.T, TP.R)                                                  # the modules whose global `RD` names the module under test

T_FORMS = "test_arm_seeds_forms_default_and_named_refusals"
T_LAYOUT = "test_the_layout_follows_the_seeds_beyond_stage4_and_stays_two_seed_at_stage4"
T_DECIDE = "test_decide_seed0_only_never_names_dominant_t2_and_the_default_is_unchanged"
T_S4 = "test_main_arm_seeds_0_on_the_c0c_stage4_world_reads_seed0_only_and_never_loads_seed1"
T_S5 = "test_main_stage5_arm_seed0_only_is_read_on_a_seed0_stage_and_refused_by_name_on_a_two_seed_stage"
T_REF = "test_main_refuses_a_wrong_arm_seeds_before_anything_is_read"
T_ID = "test_default_read_is_the_fourth_pins_read_field_for_field_on_synthetic_worlds"
T_UNT = "test_the_fourth_pin_is_untouched_and_the_fifth_is_a_separate_file"

# (name, [(anchor, replacement)], [designated tests that must go RED])
MUT = [
    ("cap_removed_a_seed0_dominant_block_names_dominant_t2", [("out.update(outcome=SEED0_ONLY, n_seeds=len(arm_seeds)", "out.update(n_seeds=len(arm_seeds)")], [T_DECIDE, T_S4]),
    ("cap_only_lets_a_dominant_block_through", [("out.update(outcome=SEED0_ONLY,", 'out.update(outcome=(name if name == "DOMINANT-T2" else SEED0_ONLY),')], [T_DECIDE, T_S4]),
    ("cap_names_the_block_outcome", [("out.update(outcome=SEED0_ONLY,", "out.update(outcome=name,")], [T_DECIDE, T_S4]),
    ("counted_dominant_true", [("counted_dominant=False, statement=SEED0_STATEMENT, seed0_block", "counted_dominant=True, statement=SEED0_STATEMENT, seed0_block")], [T_DECIDE]),
    ("statement_reworded", [('SEED0_STATEMENT = "seed 0 only, not a counted DOMINANT"', 'SEED0_STATEMENT = "seed 0 only"')], [T_DECIDE, T_S4]),
    ("arrays_built_for_both_seeds", [("scenes, s, **kw) for s in arm_seeds}", "scenes, s, **kw) for s in SEEDS}")], [T_S4]),
    ("g3_plan_loads_the_seed1_cells_silently", [("range(n_l + n_g)) for s in arm_seeds)", "range(n_l + n_g)) for s in SEEDS)")], [T_S4]),
    ("g2e_checks_the_seed1_cells", [('for s in sp["seeds"]], excluded=excluded)', "for s in SEEDS], excluded=excluded)")], [T_S4, T_S5]),
    ("absence_checks_both_seeds", [("    for s in seeds:\n        if not glob.glob(", "    for s in SEEDS:\n        if not glob.glob(")], [T_S5]),
    ("stage4_layout_follows_the_read_seeds", [("return SEEDS if arm_stage <= 4 else tuple(arm_seeds)", "return tuple(arm_seeds)")], [T_LAYOUT, T_S4]),
    ("stage5_layout_ignores_the_read_seeds", [("return SEEDS if arm_stage <= 4 else tuple(arm_seeds)", "return SEEDS")], [T_LAYOUT, T_S5]),
    ("default_arm_seeds_is_zero", [('ap.add_argument("--arm-seeds", default="0,1"', 'ap.add_argument("--arm-seeds", default="0"')], [T_S4, T_ID]),
    ("default_read_adds_a_key", [("    absent = arm_absent(roots[arm_st], arm, arm_seeds)", "    res.update(n_seeds=len(arm_seeds))\n    absent = arm_absent(roots[arm_st], arm, arm_seeds)")], [T_ID]),
    ("seed_gap_computed_from_one_seed", [('return {"computable": False, "reason": SEED_GAP_NOT_COMPUTABLE}', "pass")], [T_DECIDE, T_S4]),
    ("forms_accept_seed1", [('ARM_SEEDS_FORMS = {"0,1": SEEDS, "0": (0,)}', 'ARM_SEEDS_FORMS = {"0,1": SEEDS, "0": (0,), "1": (1,)}')], [T_FORMS, T_REF]),
    ("forms_refusal_deleted", [("    if seeds is None:\n        why.append(", "    if False:\n        why.append(")], [T_FORMS, T_REF]),
    ("decide_default_accepts_one_block", [("    if sorted(blocks) != list(arm_seeds):\n        raise ValueError(", "    if False:\n        raise ValueError(")], [T_DECIDE]),
    ("md_ignores_the_seed0_form", [('seed0 = res.get("arm_seeds") == [0]', "seed0 = False")], [T_S4]),
    ("fourth_pin_path_wrong", [('HERE = os.path.dirname(PIN5_DIR) + "/code"', 'HERE = os.path.dirname(PIN5_DIR) + "/pin5"')], [T_UNT]),
    # the silent one: seed-1 cells loaded and seed 1's records fed to the row builder AS seed 0 (two edits that together read seed 1 under --arm-seeds 0)
    ("seed1_records_read_as_seed0", [("range(n_l + n_g)) for s in arm_seeds)", "range(n_l + n_g)) for s in SEEDS)"), ('arrays_for_seed(recs[f"{arm}_s{s}"]', 'arrays_for_seed(recs[f"{arm}_s{s | (len(arm_seeds) < 2)}"]')], [T_S4]),
    ("seeds_swapped_in_the_default_read", [('arrays_for_seed(recs[f"{arm}_s{s}"]', 'arrays_for_seed(recs[f"{arm}_s{s ^ 1}"]')], [T_S4]),   # the synthetic arrays ignore the records: the data-level arm is REAL_MUT below
]


# REAL-DATA arms: the mutated source reads the real stage 2/3/4 pulls (about 35 s each). (name, edits, extra argv, what must differ from the read of record)
REAL_MUT = [
    ("REAL seeds_swapped_in_the_default_read", [('arrays_for_seed(recs[f"{arm}_s{s}"]', 'arrays_for_seed(recs[f"{arm}_s{s ^ 1}"]')], [], "identity"),
    ("REAL seed1_records_read_as_seed0", [("range(n_l + n_g)) for s in arm_seeds)", "range(n_l + n_g)) for s in SEEDS)"), ('arrays_for_seed(recs[f"{arm}_s{s}"]', 'arrays_for_seed(recs[f"{arm}_s{s | (len(arm_seeds) < 2)}"]')], ["--arm-seeds", "0"], "seed0_rows"),
]
T2V8F = HERE.rsplit("/", 1)[0]
RECORD = f"{T2V8F}/stage4/score/DOMINANCE_T2.json"


def real_arm(name, edits, extra, what):
    """-> (is_red, text): the mutated source, written to the scratch with an absolute `HERE`, reads the real pulls with the read of record's arguments; RED = its output differs from the record where it must not."""
    import importlib.util
    import json
    import subprocess
    src = SRC
    for old, new in edits + [('HERE = os.path.dirname(PIN5_DIR) + "/code"', f'HERE = "{T2V8F}/code"')]:
        if SRC.count(old) != 1:
            return False, f"ANCHOR ERROR {old[:60]!r}"
        src = src.replace(old, new)
    d = SCRATCH / ("real_" + name.split()[-1])
    d.mkdir(parents=True, exist_ok=True)
    f = d / "read_t2_dominance_pin5_mut.py"
    with open(f, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(src)
    bash = os.environ.get("PIN5_BASH") or shutil.which("bash")  # never a bare "bash": CreateProcess searches System32 before PATH and finds WSL bash.exe (rc 1, empty stderr)
    if not bash or "system32" in bash.lower():
        return False, f"NO GIT BASH: {bash!r}"
    r = subprocess.run([bash, f"{T2V8F}/audit/read_record_args.sh", str(f).replace("\\", "/"), str(d / "out").replace("\\", "/")] + extra, capture_output=True, text=True, cwd=T2V8F, timeout=900)
    if r.returncode != 0:
        return False, f"the mutated read failed rc={r.returncode}: {r.stderr[-200:]} | stdout {r.stdout[-200:]}"
    spec = importlib.util.spec_from_file_location("cmp_real", f"{T2V8F}/code/compare_dominance_reads.py")
    cmp_ = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cmp_)
    new_path = str(d / "out" / "DOMINANCE_T2.json").replace("\\", "/")
    new, rec = json.load(open(new_path, encoding="utf-8")), json.load(open(RECORD, encoding="utf-8"))
    if what == "identity":
        c = cmp_.compare(RECORD, new_path)
        return (not c["identical"]), f"identical={c['identical']}; differences {c['differences'][:2]}"
    same = new["by_seed"]["s0"] == rec["by_seed"]["s0"]
    return (not same), f"seed-0 block equals the record's seed 0: {same}"


def load(src):
    mod = types.ModuleType("read_t2_dominance")
    mod.__file__ = SRC_PATH
    exec(compile(src, SRC_PATH, "exec"), mod.__dict__)
    ORIG.init()
    mod.STATE.update(ORIG.STATE)
    return mod


_N = [0]


def run_test(mod, name):
    """-> None when green, else 'ExcType: message'."""
    f = getattr(TP, name)
    for h in HOLDERS:
        h.RD = mod
    prm = inspect.signature(f).parameters
    kw, mp = {}, None
    _N[0] += 1
    d = SCRATCH / f"t{_N[0]:04d}"
    d.mkdir(parents=True)
    if "tmp_path" in prm:
        kw["tmp_path"] = d
    if "monkeypatch" in prm:
        mp = pytest.MonkeyPatch()
        kw["monkeypatch"] = mp
    try:
        f(**kw)
        return None
    except BaseException as e:                                              # noqa: BLE001
        return f"{type(e).__name__}: {str(e)[:150]}".replace("\n", " ")
    finally:
        if mp:
            mp.undo()
        for h in HOLDERS:
            h.RD = ORIG
        shutil.rmtree(d, ignore_errors=True)


def main():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    names = [n for n, f in inspect.getmembers(TP, inspect.isfunction) if n.startswith("test_") and f.__module__ == TP.__name__]
    bad = {n: r for n in names if (r := run_test(ORIG, n))}
    print(f"baseline: {len(names) - len(bad)} of {len(names)} tests green" + (f"; RED: {bad}" if bad else ""), flush=True)
    if bad:
        print("ZZ PIN5-MUT-DONE ZZ")
        return 2
    red = 0
    for name, edits, designated in MUT:
        src = SRC
        for old, new in edits:
            if SRC.count(old) != 1:
                print(f"{name}: ANCHOR ERROR (occurs {SRC.count(old)} times): {old[:80]!r}", flush=True)
                print("ZZ PIN5-MUT-DONE ZZ")
                return 3
            src = src.replace(old, new)
        mod = load(src)
        res = {t: run_test(mod, t) for t in designated}
        ok = all(v is not None for v in res.values())
        red += ok
        print(f"{name}: {'RED' if ok else 'GREEN  <-- not caught by ' + str([t for t, v in res.items() if v is None])}  " + "; ".join(f"{t[:40]} -> {(v or 'passed')[:100]}" for t, v in res.items()), flush=True)
    n_all = len(MUT)
    if os.environ.get("PIN5_MUT_REAL", "1") == "1":
        for name, edits, extra, what in REAL_MUT:
            n_all += 1
            is_red, text = real_arm(name, edits, extra, what)
            red += is_red
            print(f"{name}: {'RED' if is_red else 'GREEN  <-- not caught'}  {text}", flush=True)
    print(f"mutations RED: {red} of {n_all}", flush=True)
    print("ZZ PIN5-MUT-DONE ZZ", flush=True)
    return 0 if red == n_all else 1


if __name__ == "__main__":
    sys.exit(main())
