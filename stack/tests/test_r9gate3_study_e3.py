"""The refcv9 gate3 family takes a THIRD study, E3 (profile `refcv9-full`; MM 2026-11-11 E3-STUDY), beside e1 and e2.

  study e3 = root /home/nvidia/r8ns/r9e3, argv prefix r9e3, profile refcv9-full, dev-box stage F:/refcv9_e3/gate, ship stage .../r9gate3_ship_e3
             NO default arm, NO fixed explore arm, NO default record commit: the E2 GO arm V is not known when the tooling lands, so the operator names the arms
             (`V` for E3 alone, `V V_d1` to chain the D1 branch after it) and the explore arm is the first arm that is not a D1 branch.
  D1 arm   = `<V>_d1` (r9e3_<V>_d1.argv.json): gated under profile refcv9-full-d1 (`r9_profile_of_arm`), everything else under the study's profile.

Proved by EXECUTION: the resolution table; e1 / e2 are untouched (table, per-arm profile, explore-arm resolution); an unknown study still refuses; the two Thor scripts that run
off-Thor build the E3 / D1 gate command lines; a broken e3 entry / a broken D1 mapping goes RED; the chain / restart / pinned driver are wired structurally (they need ssh / WMI);
`r9e1_bind.sh --argv-prefix r9e3` resolves an E3 and a D1 argv (composed here by r9_e3_make_argv.py from the E2 K2M1N20 argv) and pins the MEASURED fact that neither
reuses E2's records (the harness's effective build argv carries the schedule flags).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import test_launch_gate_e1rig_e1rec2 as R2  # noqa: E402
import test_r9gate3_study as S  # noqa: E402

needs_bash = S.needs_bash
STUDY, GATE3, KEYS = S.STUDY, S.GATE3, S.KEYS
E1, E2 = S.E1, S.E2
E3 = dict(R9_STUDY="e3", R9_ROOT="/home/nvidia/r8ns/r9e3", R9_PREFIX="r9e3", R9_PROFILE="refcv9-full", R9_EXPLORE_ARM="", R9_DEFAULT_ARMS="",
          R9_DEFAULT_REC12="", R9_DEVSTAGE="F:/refcv9_e3/gate", R9_SHIP_STAGE="/f/c_overflow_2026-10-26/tmp/r9gate3_ship_e3",
          R9_STUDY_NOTE=" study=e3 profile=refcv9-full root=/home/nvidia/r8ns/r9e3")
V = "K2M1N20"          # an arm NAME for the tests only: the tooling is arm-agnostic
REC = "4e36c721d0e1"


def _fn(study: str | None, body: str, study_file: Path = STUDY) -> tuple[int, str]:
    """source the study file under R9_STUDY=study (None = unset), run `body`, return (rc, stdout)."""
    env = {} if study is None else {"R9_STUDY": study}
    rc, out, _ = S._run(["-c", "set -u; . '%s' || exit $?; %s" % (study_file.as_posix(), body)], env=env)
    return rc, out


# ================================================================================================ the study file ==
@needs_bash
def test_e3_resolution_table():
    rc, got, out = S._resolve(STUDY, {"R9_STUDY": "e3"})
    assert rc == 0 and got == E3, (got, out)


@needs_bash
@pytest.mark.parametrize("env,want", [(None, E1), ({"R9_STUDY": "e1"}, E1), ({"R9_STUDY": "e2"}, E2)])
def test_e1_and_e2_tables_are_unchanged_by_adding_e3(env, want):
    rc, got, out = S._resolve(STUDY, env)
    assert rc == 0 and got == want, (env, out)


@needs_bash
@pytest.mark.parametrize("val", ["e4", "E3", " e3", "e3 ", "e3 e2", "r9e3", "bogus", ""])
def test_an_unknown_study_still_REFUSES_and_e3_is_never_a_fallback(val):
    rc, got, out = S._resolve(STUDY, {"R9_STUDY": val})
    assert rc == 10 and "ZZR9STUDY-REFUSEDZZ" in out and "R9_ROOT" not in got, (val, rc, out)
    assert "(known: e1 e2 e3;" in out


@needs_bash
def test_the_prefix_lookup_knows_r9e3():
    rc, out = _fn("e1", "R9_STUDY_NOAUTOLOAD=1; for p in r9e1 r9e2 r9e3 r9e4; do echo \"p=$p->$(r9_study_of_prefix \"$p\" || echo NONE)\"; done")
    assert rc == 0 and out.split() == ["p=r9e1->e1", "p=r9e2->e2", "p=r9e3->e3", "p=r9e4->NONE"], out


@needs_bash
@pytest.mark.parametrize("study,arm,want", [("e1", "C0", "refcv9-e1rig"), ("e1", "C0_d1", "refcv9-e1rig"), ("e2", "K0", "refcv9-e2"), ("e2", "K0_d1", "refcv9-e2"),
                                            ("e3", V, "refcv9-full"), ("e3", V + "_d1", "refcv9-full-d1"), ("e3", "d1", "refcv9-full"), ("e3", "_d1", "refcv9-full"),
                                            ("e3", V + "_d10", "refcv9-full"), ("e3", V + "_D1", "refcv9-full")])
def test_the_profile_of_an_arm_is_the_studys_except_an_e3_D1_branch(study, arm, want):
    rc, out = _fn(study, 'r9_profile_of_arm "%s"' % arm)
    assert rc == 0 and out.strip() == want, (study, arm, out)


@needs_bash
@pytest.mark.parametrize("study,arms,want", [
    ("e3", V + "_d1 " + V + " W_d1 W", [V + "_d1=refcv9-full-d1", V + "=refcv9-full", "W_d1=refcv9-full-d1", "W=refcv9-full"]),
    ("e2", "K0 K2M1N20", ["K0=refcv9-e2", "K2M1N20=refcv9-e2"]),
    ("e1", "C0 C1", ["C0=refcv9-e1rig", "C1=refcv9-e1rig"])])
def test_the_profile_is_IDEMPOTENT_across_the_per_arm_loop_that_reassigns_R9_PROFILE(study, arms, want):
    """MM review 2026-11-11: the chain and the pinned driver assign R9_PROFILE="$(r9_profile_of_arm "$A")" INSIDE `for A in $ARMS`. A profile built from the
    mutable R9_PROFILE leaked a D1 arm's refcv9-full-d1 into every later arm and gave a second D1 arm refcv9-full-d1-d1. This replays that loop."""
    rc, out = _fn(study, 'for A in %s; do R9_PROFILE="$(r9_profile_of_arm "$A")"; echo "$A=$R9_PROFILE"; done' % arms)
    assert rc == 0 and out.split() == want, (study, arms, out)


@needs_bash
@pytest.mark.parametrize("study,arms,want,rc_want", [("e1", "X Y", "C0", 0), ("e2", "", "K0", 0), ("e3", V, V, 0), ("e3", V + " " + V + "_d1", V, 0), ("e3", V + "_d1 " + V, V, 0),
                                                     ("e3", V + "_d1", V + "_d1", 0), ("e3", "", "", 1)])
def test_the_explore_arm_is_the_first_arm_that_is_not_a_D1_branch_and_e1_e2_keep_theirs(study, arms, want, rc_want):
    rc, out = _fn(study, 'r9_resolve_explore_arm "%s"; r=$?; echo "r=$r arm=[$R9_EXPLORE_ARM]"' % arms)
    assert rc == 0 and out.strip() == "r=%d arm=[%s]" % (rc_want, want), (study, arms, out)


def _e3_table_holds(study_file: Path) -> bool:
    rc, got, _ = S._resolve(study_file, {"R9_STUDY": "e3"})
    ok = rc == 0 and got == E3
    for env, want in ((None, E1), ({"R9_STUDY": "e2"}, E2)):
        rc, got, _ = S._resolve(study_file, env)
        ok = ok and rc == 0 and got == want
    _, o = _fn("e3", 'echo "$(r9_profile_of_arm %s_d1)|$(r9_profile_of_arm %s)|$(r9_study_of_prefix r9e3)"' % (V, V), study_file)
    return ok and o.strip() == "refcv9-full-d1|refcv9-full|e3"


@needs_bash
def test_MUTATIONS_each_broken_clause_of_the_e3_entry_is_caught(tmp_path):
    assert _e3_table_holds(STUDY)                                                                              # control: the shipped file passes
    for name, old, new in (("e3 profile -> e2's", "R9_PROFILE=refcv9-full;", "R9_PROFILE=refcv9-e2;"),
                           ("e3 prefix -> r9e2", "R9_PREFIX=r9e3;", "R9_PREFIX=r9e2;"),
                           ("e3 root -> e2's", "R9_ROOT=/home/nvidia/r8ns/r9e3;", "R9_ROOT=/home/nvidia/r8ns/r9e2;"),
                           ("e3 devstage -> e2's", "R9_DEVSTAGE=F:/refcv9_e3/gate", "R9_DEVSTAGE=F:/refcv9_e2/gate"),
                           ("e3 ship stage -> e2's", "r9gate3_ship_e3", "r9gate3_ship_e2"),
                           ("e3 gets a fixed explore arm", 'R9_PROFILE=refcv9-full; R9_EXPLORE_ARM=""', "R9_PROFILE=refcv9-full; R9_EXPLORE_ARM=K0"),
                           ("e3 gets a default record commit", 'R9_DEFAULT_REC12=""; R9_DEVSTAGE=F:/refcv9_e3', "R9_DEFAULT_REC12=d297ba563d8f; R9_DEVSTAGE=F:/refcv9_e3"),
                           ("e3 dropped from the study list", 'R9_STUDIES="e1 e2 e3"', 'R9_STUDIES="e1 e2"'),
                           ("D1 arm -> the study's profile", 'echo "${_base}-d1"', 'echo "${_base}"'),
                           ("D1 matches every arm", "e3:?*_d1)", "e3:*)")):
        d = tmp_path / ("m%d" % abs(hash(name)))
        d.mkdir()
        assert not _e3_table_holds(S._mutant(d, old, new)), name


# ===================================================================== the two Thor scripts that run off-Thor ==
@needs_bash
def test_thor_gate_explore_for_e3_builds_the_E3_command_line_under_profile_refcv9_full(tmp_path):
    rc, out, got = S._gate(tmp_path, "r9gate3_thor_gate.sh", ["explore", V], {"R9_STUDY": "e3"}, "r9e3", "r9e3", V, REC)
    assert got is not None, out
    want = S._expected_explore("r9e3", "r9e3", "refcv9-full", V, REC) + S.EXPLORE_CHECKS + ["--out-dir", f"/home/nvidia/r8ns/r9e3/gate3_explore/{S.C12}_{V}"]
    assert got == want, (got, out)
    assert f"ZZR9E1G3-{V}-explore-DONEZZ" in out and "REFUSED" not in out


@needs_bash
def test_thor_gate_final_for_the_e3_D1_arm_runs_under_refcv9_full_d1_and_the_E3_arm_under_refcv9_full(tmp_path):
    for arm, prof in ((V + "_d1", "refcv9-full-d1"), (V, "refcv9-full")):
        rc, out, got = S._gate(tmp_path / arm, "r9gate3_thor_gate.sh", ["final", arm, "@SB@", "@EV@"], {"R9_STUDY": "e3"}, "r9e3", "r9e3", arm, REC)
        assert got is not None, out
        assert got[1] == "run" and got[2] == "--profile" and got[3] == prof, (arm, got[:4])
        assert got[9].endswith(f"/runs.d/r9e3_{arm}.argv.json") and all(f"/ovf_{arm}_{REC}/" in got[got.index(o) + 1] for o in ("--map-overfit-record", "--box-overfit-record")), got
        assert got[-1] == f"/home/nvidia/r8ns/r9e3/gate3/{arm}", got[-3:]


@needs_bash
def test_e3_has_NO_default_record_stamp_a_missing_REC_C12_refuses_before_the_gate(tmp_path):
    rc, out, got = S._gate(tmp_path, "r9gate3_thor_gate.sh", ["explore", V], {"R9_STUDY": "e3"}, "r9e3", "r9e3", V, None, rec_file=False)
    assert rc == 10 and got is None and f"ZZR9E1G3-{V}-explore-REFUSEDZZ" in out and "REC_C12 stamp" in out, (rc, out)


@needs_bash
@pytest.mark.parametrize("arm,profile", [(V, "refcv9-full"), (V + "_d1", "refcv9-full-d1")])
def test_rec_explore_judges_the_records_under_the_arms_profile(tmp_path, arm, profile):
    rc, out, got = S._gate(tmp_path, "r9gate3_rec_explore.sh", [arm], {"R9_STUDY": "e3"}, "r9e3", "r9e3", arm, REC)
    assert got is not None, out
    assert got[1:4] == ["run", "--profile", profile] and got[4] == "--checks", got
    assert got[got.index("--argv-file") + 1].endswith(f"/runs.d/r9e3_{arm}.argv.json"), got


@needs_bash
def test_MUTATION_a_broken_D1_mapping_makes_the_D1_gate_command_wrong(tmp_path):
    bad = STUDY.read_text(encoding="utf-8").replace('echo "${_base}-d1"', 'echo "${_base}"')
    rc, out, got = S._gate(tmp_path, "r9gate3_thor_gate.sh", ["final", V + "_d1", "@SB@", "@EV@"], {"R9_STUDY": "e3"}, "r9e3", "r9e3", V + "_d1", REC, study_text=bad)
    assert got is not None and got[3] == "refcv9-full" != "refcv9-full-d1"                                    # the mutant is visible to the test above


# ============================================================================== the hops that need ssh / WMI / a dev box ==
def _src(name: str) -> str:
    return (GATE3 / name).read_text(encoding="utf-8")


def test_the_chain_resolves_the_explore_arm_from_its_arms_and_reads_each_arms_profile():
    s = _src("r9gate3_chain.sh")
    assert 'r9_resolve_explore_arm "$ARMS" ||' in s and "ZZR9E1CHAIN-STOP-NO-ARMSZZ" in s
    assert s.index('r9_resolve_explore_arm "$ARMS"') < s.rindex("EXP=$R/gate3_explore/${C12}_${R9_EXPLORE_ARM}") < s.index('ev START "commit')
    assert s.count("EXP=$R/gate3_explore/") == 2                                                              # the first (pre-resolution) value is the e1 / e2 one; it is re-set
    assert 'for A in $ARMS; do\n  R9_PROFILE="$(r9_profile_of_arm "$A")"' in s
    assert s.index('r9_profile_of_arm "$A"') < s.index('r9gate3_zg_env.sh "$R9_PROFILE"')                    # the zero-gradient derivation reads THE ARM's profile
    assert "$SHA $R9_PROFILE >" in s                                                                          # the suite precheck stays on the study's profile (one suite per commit)


def test_restart_refuses_an_arm_less_study_and_resolves_the_explore_arm_before_using_it():
    s = _src("r9gate3_restart.sh")
    assert '[ -n "$ARMS" ] || die' in s and 'r9_resolve_explore_arm "$ARMS" || die' in s
    assert s.index('[ -n "$ARMS" ] || die') < s.index('r9_resolve_explore_arm "$ARMS"') < s.index("explore $R9_EXPLORE_ARM")


def test_the_pinned_driver_gates_each_arm_under_its_profile_and_maps_a_D1_branch_source():
    s = _src("r9gate3_devbox_pinned.sh")
    assert 'for ARM in $ARMS; do\n  R9_PROFILE="$(r9_profile_of_arm "$ARM")"' in s
    assert 'PMAP="$(cat "$STAGE/pathmap_${ARM}.txt"' in s and '${PMAP:+--path-map "$PMAP"}' in s
    assert s.index('r9_profile_of_arm "$ARM"') < s.index('--profile "$R9_PROFILE"')


@needs_bash
def test_the_pinned_driver_command_gets_a_path_map_only_when_the_arm_has_a_pathmap_file(tmp_path):
    """the `${PMAP:+--path-map "$PMAP"}` word expands to NOTHING when the file is absent (every e1 / e2 arm) and to two words when present."""
    st = tmp_path / "stage"
    st.mkdir()
    (st / "pathmap_X_d1.txt").write_bytes(b"/home/nvidia/r8ns/r9e3/runs/r9e3-X=F:/refcv9_e3/gate/src_X\r\n")
    prog = ('STAGE=%s; for ARM in X X_d1; do PMAP="$(cat "$STAGE/pathmap_${ARM}.txt" 2>/dev/null | tr -d \'\\r\' | head -1)"; '
            'set -- --argv-file a ${PMAP:+--path-map "$PMAP"} --suite-work w; echo "$ARM $#:$*"; done') % st.as_posix()
    rc, out, _ = S._run(["-c", prog])
    assert rc == 0 and out.split("\n")[:2] == ["X 4:--argv-file a --suite-work w", "X_d1 6:--argv-file a --path-map /home/nvidia/r8ns/r9e3/runs/r9e3-X=F:/refcv9_e3/gate/src_X --suite-work w"], out


def test_the_study_file_and_the_family_stay_LF_only_and_the_e3_values_have_one_home():
    for name in ("r9gate3_study.sh", *S.FAMILY):
        assert b"\r" not in (GATE3 / name).read_bytes(), name
    for name in S.FAMILY:
        for ln in S._code(name):
            for lit in ("r8ns/r9e3", "refcv9_e3", "refcv9-full"):
                assert lit not in ln, (name, lit, ln)                                                         # no operational e3 literal outside the study file


# ============================================================================================ r9e1_bind.sh --argv-prefix r9e3 ==
@pytest.fixture(scope="module")
def e3_code_tree(tmp_path_factory):
    """a code tree = the repo's stack (minus tests) + taniteval's .py files, with r9e3_<V>.argv.json and r9e3_<V>_d1.argv.json composed from the E2 K2M1N20 argv by r9_e3_make_argv.py."""
    x = tmp_path_factory.mktemp("e3code")
    shutil.copytree(ROOT, x / "stack", ignore=shutil.ignore_patterns("tests", "__pycache__"))
    shutil.copytree(ROOT.parent / "taniteval", x / "taniteval", ignore=lambda d, names: [n for n in names if (Path(d) / n).is_file() and not n.endswith(".py")])
    d = x / "stack" / "ops" / "runs.d"
    subprocess.run([sys.executable, str(x / "stack/scripts/r9_e3_make_argv.py"), "--e2", str(d / f"r9e2_{V}.argv.json"), "--out", f"/home/nvidia/r8ns/r9e3/runs/r9e3-{V}",
                    "--write-e3", str(d / f"r9e3_{V}.argv.json"), "--write-d1", str(d / f"r9e3_{V}_d1.argv.json")], check=True, encoding="utf-8", capture_output=True,
                   env=dict(os.environ, MSYS_NO_PATHCONV="1", PYTHONDONTWRITEBYTECODE="1"))
    return x


@needs_bash
@pytest.mark.parametrize("arm", [V, V + "_d1"])
def test_bind_resolves_an_E3_and_a_D1_argv_under_the_r9e3_prefix(tmp_path, e3_code_tree, arm):
    out, root = R2._bind(tmp_path, "--argv-prefix", "r9e3", "box", arm, code=e3_code_tree)
    assert "ZZR9E1BIND-DRYOKZZ" in out and "REFUSED" not in out and f"runs.d/r9e3_{arm}.argv.json sha256" in out, out
    assert (root / f"ovf_{arm}_{R2.HASH[:12]}" / "box.args").is_file()


@needs_bash
def test_bind_E3_cannot_reuse_E2s_box_record_nor_D1_E3s_the_harness_build_argv_carries_the_schedule(tmp_path, e3_code_tree):
    """MEASURED here (2026-11-11, harness --identity-only on this CPU): the E3 box invocation is NOT identical to E2's -- `effective_build_argv` differs (--steps / --warmup /
    --wsd-decay-steps / --save-at), so E3 needs its OWN box record; the D1 argv differs from E3's by --steps / --save-at / the branch flags and data_roots (--wsd-branch-from)."""
    out, _ = R2._bind(tmp_path / "a", "--argv-prefix", "r9e3", "box", V, "--reuse-from", V, "--reuse-argv-prefix", "r9e2", "--reuse-root", (tmp_path / "e2root").as_posix(), code=e3_code_tree)
    assert "ZZR9E1BIND-REFUSEDZZ" in out and "NOT identical" in out and "effective_build_argv" in out, out
    out, _ = R2._bind(tmp_path / "b", "--argv-prefix", "r9e3", "box", V + "_d1", "--reuse-from", V, "--reuse-root", (tmp_path / "e3root").as_posix(), code=e3_code_tree)
    assert "ZZR9E1BIND-REFUSEDZZ" in out and "NOT identical" in out and "data_roots" in out, out


@needs_bash
def test_bind_still_refuses_a_prefix_no_study_owns(tmp_path):
    out, _ = R2._bind(tmp_path, "--argv-prefix", "r9e4", "box", V)
    assert "ZZR9E1BIND-REFUSEDZZ" in out and "is not the argv prefix of any study" in out, out
