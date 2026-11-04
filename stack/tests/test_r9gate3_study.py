"""The refcv9 gate3 family (chain / thor_gate / rec_explore / restart / devbox_pinned / launch_pinned.ps1) and `r9e1_bind.sh` take ONE study selector, defined ONCE in
`stack/ops/r9e1_gate3/r9gate3_study.sh` (MM 207ebd E2-CHAIN, 2026-11-03: the E2 control arm K0 has to be bound, gated and trained by the SAME tooling as the E1 arms).

  study e1 (DEFAULT, selector UNSET) = what every script did before the file existed: root /home/nvidia/r8ns/r9e1, argv prefix r9e1, profile refcv9-e1rig, explore arm C0
  study e2                           = root /home/nvidia/r8ns/r9e2, prefix r9e2, profile refcv9-e2, explore arm K0
  anything else, INCLUDING THE EMPTY STRING = a loud REFUSAL (never a silent fall-back to e1)

What is proved here (each by EXECUTION, not by reading): the study file's resolution under unset / e1 / e2 / bogus / empty; the two Thor scripts that can run off-Thor (`thor_gate`,
`rec_explore`) build, for the default, EXACTLY the launch-gate command line they built at Y4 (literal list below) and, for e2, the K0 one; a mutated e2 mapping / a mutated
refusal clause goes RED; no operational e1 literal is left outside the study file; `r9e1_bind.sh --argv-prefix` resolves `r9e2_K0.argv.json`, refuses a prefix no study owns before touching
anything, reproduces the MEASURED K0 facts (map harness invocation IDENTICAL to C0's, box DIFFERENT => K0 needs its own box record) in a dry run. The restart / pinned-driver / ps1 hops are asserted
structurally (they need ssh / WMI): the study reaches every child.
"""
from __future__ import annotations

import os
import re
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

GATE3 = ROOT / "ops" / "r9e1_gate3"
STUDY = GATE3 / "r9gate3_study.sh"
BASH = R2.BASH
needs_bash = R2.needs_bash
SHA = "d297ba563d8fddc0ce9cfb8380408878a3778fb2"
C12 = SHA[:12]

E1 = dict(R9_STUDY="e1", R9_ROOT="/home/nvidia/r8ns/r9e1", R9_PREFIX="r9e1", R9_PROFILE="refcv9-e1rig", R9_EXPLORE_ARM="C0", R9_DEFAULT_ARMS="C0 C1",
          R9_DEFAULT_REC12="d297ba563d8f", R9_DEVSTAGE="F:/refcv9_e1/gate", R9_SHIP_STAGE="/f/c_overflow_2026-10-26/tmp/r9gate3_ship", R9_STUDY_NOTE="")
E2 = dict(R9_STUDY="e2", R9_ROOT="/home/nvidia/r8ns/r9e2", R9_PREFIX="r9e2", R9_PROFILE="refcv9-e2", R9_EXPLORE_ARM="K0", R9_DEFAULT_ARMS="K0",
          R9_DEFAULT_REC12="", R9_DEVSTAGE="F:/refcv9_e2/gate", R9_SHIP_STAGE="/f/c_overflow_2026-10-26/tmp/r9gate3_ship_e2",
          R9_STUDY_NOTE=" study=e2 profile=refcv9-e2 root=/home/nvidia/r8ns/r9e2")
KEYS = tuple(E1)


def _run(args, env=None, timeout=240, cwd=None):
    e = dict(os.environ, MSYS_NO_PATHCONV="1", PYTHONDONTWRITEBYTECODE="1")
    e.pop("R9_STUDY", None)
    e.update(env or {})
    r = subprocess.run([BASH, *args], capture_output=True, text=True, encoding="utf-8", env=e, timeout=timeout, cwd=cwd, stdin=subprocess.DEVNULL)
    return r.returncode, r.stdout.replace("\r", ""), r.stderr.replace("\r", "")


def _resolve(study_file: Path, env: dict | None):
    """source the study file under `env` (None = R9_STUDY unset) and print the variables -> (rc, {var: value}, stdout)."""
    prog = "set -u; . '%s' || exit $?; " % study_file.as_posix() + "; ".join('echo "%s=[$%s]"' % (k, k) for k in KEYS)
    rc, out, _ = _run(["-c", prog], env=env)
    got = {m.group(1): m.group(2) for m in re.finditer(r"^(R9_[A-Z0-9_]+)=\[(.*)\]$", out, re.M)}
    return rc, got, out


# ================================================================================================ the study file ==
@needs_bash
@pytest.mark.parametrize("env,want", [(None, E1), ({"R9_STUDY": "e1"}, E1), ({"R9_STUDY": "e2"}, E2)])
def test_study_resolution_table_unset_is_e1_and_e2_is_the_full_model(env, want):
    rc, got, out = _resolve(STUDY, env)
    assert rc == 0 and got == want, (env, out)


@needs_bash
@pytest.mark.parametrize("val", ["bogus", "", "E1", " e1", "e1 e2", "r9e1", "e3"])
def test_an_unknown_or_empty_study_REFUSES_and_never_falls_back_to_e1(val):
    rc, got, out = _resolve(STUDY, {"R9_STUDY": val})
    assert rc == 10 and "ZZR9STUDY-REFUSEDZZ" in out and "REFUSED: unknown R9_STUDY" in out and "R9_ROOT" not in got, (val, rc, out)


@needs_bash
def test_the_prefix_lookup_is_the_same_table_and_a_foreign_prefix_has_no_study():
    prog = "R9_STUDY_NOAUTOLOAD=1; . '%s'; for p in r9e1 r9e2 r9e3 ''; do echo \"p=$p->$(r9_study_of_prefix \"$p\" || echo NONE)\"; done" % STUDY.as_posix()
    rc, out, _ = _run(["-c", prog])
    assert rc == 0 and out.split() == ["p=r9e1->e1", "p=r9e2->e2", "p=r9e3->NONE", "p=->NONE"], out


def _mutant(tmp_path, old, new):
    src = STUDY.read_text(encoding="utf-8")
    assert src.count(old) == 1, old
    p = tmp_path / "r9gate3_study.sh"
    p.write_text(src.replace(old, new), encoding="utf-8")
    return p


def _the_table_still_holds(study_file) -> bool:
    ok = True
    for env, want in ((None, E1), ({"R9_STUDY": "e2"}, E2)):
        rc, got, _ = _resolve(study_file, env)
        ok = ok and rc == 0 and got == want
    for val in ("bogus", ""):
        rc, got, out = _resolve(study_file, {"R9_STUDY": val})
        ok = ok and rc == 10 and "ZZR9STUDY-REFUSEDZZ" in out and "R9_ROOT" not in got
    return ok


@needs_bash
def test_MUTATIONS_each_broken_clause_of_the_study_file_is_caught(tmp_path):
    assert _the_table_still_holds(STUDY)                                                                                  # control: the shipped file passes
    for name, old, new in (("e2 profile -> the rig's", "R9_PROFILE=refcv9-e2;", "R9_PROFILE=refcv9-e1rig;"),
                           ("e2 prefix -> r9e1", "R9_PREFIX=r9e2;", "R9_PREFIX=r9e1;"),
                           ("e2 explore arm -> C0", "R9_EXPLORE_ARM=K0;", "R9_EXPLORE_ARM=C0;"),
                           ("e2 root -> e1's", "R9_ROOT=/home/nvidia/r8ns/r9e2;", "R9_ROOT=/home/nvidia/r8ns/r9e1;"),
                           ("e1 default arms", 'R9_DEFAULT_ARMS="C0 C1"', 'R9_DEFAULT_ARMS="C0"'),
                           ("e1 default REC12", "R9_DEFAULT_REC12=d297ba563d8f;", "R9_DEFAULT_REC12=4e36c721d0e1;"),
                           ("an empty selector falls back to e1", '"${R9_STUDY-e1}"', '"${R9_STUDY:-e1}"'),
                           ("no refusal marker", 'echo "ZZR9STUDY-REFUSEDZZ"', ":")):
        d = tmp_path / ("m%d" % abs(hash(name)))
        d.mkdir()
        assert not _the_table_still_holds(_mutant(d, old, new)), name


# ===================================================================== the two Thor scripts that run off-Thor ==
_PY_SHIM = r"""#!/usr/bin/env bash
case "${1:-}" in
  *launch_gate.py) printf '%s\n' "$@" > "$REC_OUT"; echo "G-MAP-OVERFIT PASS"; echo "G-BOX-OVERFIT PASS"; exit 0;;
  -) exit 0;;
esac
exec "$REAL_PY" "$@"
"""


def _sandbox(tmp: Path, study_text: str | None = None) -> Path:
    """A Thor in a directory: every `/home/nvidia` of the scripts and of the study file is rewritten under `tmp/sb`; flock / nvidia-smi / the venv python are shims."""
    sb = tmp / "sb"
    (sb / "kit").mkdir(parents=True)
    (sb / "bin").mkdir()
    for f in ("r9gate3_thor_gate.sh", "r9gate3_rec_explore.sh", "r9gate3_study.sh"):
        t = (GATE3 / f).read_text(encoding="utf-8") if not (f == "r9gate3_study.sh" and study_text is not None) else study_text
        (sb / "kit" / f).write_text(t.replace("/home/nvidia", (sb / "home/nvidia").as_posix()), encoding="utf-8", newline="\n")
    for name, body in (("flock", "exit 0\n"), ("nvidia-smi", "exit 0\n")):
        (sb / "bin" / name).write_text("#!/usr/bin/env bash\n" + body, encoding="utf-8", newline="\n")
    py = sb / "home/nvidia/venvs/tanitad-train/bin/python"
    py.parent.mkdir(parents=True)
    py.write_text(_PY_SHIM, encoding="utf-8", newline="\n")
    for shim in (sb / "bin" / "flock", sb / "bin" / "nvidia-smi", py):         # EXECUTABLE: on Linux (Thor) a PATH lookup skips a non-executable shim and
        shim.chmod(0o755)                                                      # runs the REAL flock / nvidia-smi (the training GPU refused the gate), and `$PY` fails
    (sb / "home/nvidia/refcv7_post").mkdir(parents=True)
    return sb


def _arm_files(sb: Path, root: str, prefix: str, arm: str, rec12: str | None, rec_file: bool = True):
    r = sb / "home/nvidia/r8ns" / root
    (r / "launch_tree/stack/ops/runs.d").mkdir(parents=True, exist_ok=True)
    (r / "launch_tree.commit").write_text(SHA + "\n", encoding="utf-8")
    (r / "launch_tree/stack/ops/runs.d" / f"{prefix}_{arm}.argv.json").write_text("[]", encoding="utf-8")
    (r / "launch_tree/stack/ops/refcv8_eval_remap_a2_thor.json").write_text("{}", encoding="utf-8")
    (r / "gate3").mkdir(exist_ok=True)
    if rec_file and rec12:
        (r / "gate3/REC_C12").write_text(rec12 + "\n", encoding="utf-8")
    ov = r / f"ovf_{arm}_{rec12 or C12}"
    for rel in ("run_map/g_map_overfit.json", "map_closure.json", "run_box/gbo_binding.json", "box_closure.json"):
        (ov / rel).parent.mkdir(parents=True, exist_ok=True)
        (ov / rel).write_text("{}", encoding="utf-8")
    (r / "suite_bound.json").write_text("{}", encoding="utf-8")
    (r / "ev/evidence").mkdir(parents=True, exist_ok=True)
    (r / "ev/evidence/G-SUITE-PINNED.json").write_text("{}", encoding="utf-8")
    return r


def _gate(tmp: Path, script: str, args: list[str], env: dict | None, root: str, prefix: str, arm: str, rec12: str | None, study_text=None, rec_file=True):
    sb = _sandbox(tmp, study_text)
    r = _arm_files(sb, root, prefix, arm, rec12, rec_file)
    rec = sb / "recorded_args.txt"
    e = dict(env or {}, PATH=(sb / "bin").as_posix() + os.pathsep + os.environ.get("PATH", ""), REC_OUT=rec.as_posix(), REAL_PY=Path(sys.executable).as_posix())
    a = [(sb / "kit" / script).as_posix(), *[x.replace("@EV@", (r / "ev").as_posix()).replace("@SB@", (r / "suite_bound.json").as_posix()) for x in args]]
    rc, out, err = _run(a, env=e, cwd=sb.as_posix(), timeout=120)
    got = [ln.replace(sb.as_posix(), "") for ln in rec.read_text(encoding="utf-8").replace("\r", "").splitlines()] if rec.exists() else None
    return rc, out.replace(sb.as_posix(), ""), got


def _expected_explore(root, prefix, profile, arm, rec12):
    r = f"/home/nvidia/r8ns/{root}"
    ov = f"{r}/ovf_{arm}_{rec12}"
    return ["/home/nvidia/r8ns/%s/launch_tree/stack/scripts/launch_gate.py" % root, "run", "--profile", profile, "--tree", f"{r}/launch_tree", "--commit", SHA,
            "--argv-file", f"{r}/launch_tree/stack/ops/runs.d/{prefix}_{arm}.argv.json", "--eval-kit", "/home/nvidia",
            "--eval-remap-overrides", f"{r}/launch_tree/stack/ops/refcv8_eval_remap_a2_thor.json", "--omp", "8",
            "--map-overfit-record", f"{ov}/run_map/g_map_overfit.json", "--map-overfit-closure", f"{ov}/map_closure.json",
            "--box-overfit-record", f"{ov}/run_box/gbo_binding.json", "--box-overfit-closure", f"{ov}/box_closure.json"]


EXPLORE_CHECKS = ["--checks", "G-HYG,G-DVB,G-LIVE,G-CKPT,G-EVAL,G-CLOCK"]


@needs_bash
@pytest.mark.parametrize("env", [None, {"R9_STUDY": "e1"}])
def test_thor_gate_explore_with_the_selector_unset_builds_the_Y4_gate_command_line_for_C0(tmp_path, env):
    rc, out, got = _gate(tmp_path, "r9gate3_thor_gate.sh", ["explore", "C0"], env, "r9e1", "r9e1", "C0", None)       # no REC_C12 file: the e1 DEFAULT stamp is used
    assert got is not None, out
    want = _expected_explore("r9e1", "r9e1", "refcv9-e1rig", "C0", "d297ba563d8f") + EXPLORE_CHECKS + ["--out-dir", f"/home/nvidia/r8ns/r9e1/gate3_explore/{C12}_C0"]
    assert got == want, (got, out)
    assert "ZZR9E1G3-C0-explore-DONEZZ" in out and "REFUSED" not in out


@needs_bash
def test_thor_gate_final_for_e1_adds_only_the_stage_thor_tail(tmp_path):
    rc, out, got = _gate(tmp_path, "r9gate3_thor_gate.sh", ["final", "C1", "@SB@", "@EV@"], {"R9_STUDY": "e1"}, "r9e1", "r9e1", "C1", None)
    assert got is not None, out
    tail = got[-8:]
    assert tail[0] == "--stage" and tail[1] == "thor" and tail[2] == "--suite-verdict" and tail[4] == "--import-evidence" and tail[6] == "--out-dir", tail
    assert tail[7] == "/home/nvidia/r8ns/r9e1/gate3/C1" and got[3] == "refcv9-e1rig" and got[9].endswith("/runs.d/r9e1_C1.argv.json"), got


@needs_bash
def test_thor_gate_explore_for_e2_builds_the_K0_command_line_under_profile_refcv9_e2(tmp_path):
    rc, out, got = _gate(tmp_path, "r9gate3_thor_gate.sh", ["explore", "K0"], {"R9_STUDY": "e2"}, "r9e2", "r9e2", "K0", "23fbf8f7e2b1")
    assert got is not None, out
    want = _expected_explore("r9e2", "r9e2", "refcv9-e2", "K0", "23fbf8f7e2b1") + EXPLORE_CHECKS + ["--out-dir", f"/home/nvidia/r8ns/r9e2/gate3_explore/{C12}_K0"]
    assert got == want, (got, out)
    assert "ZZR9E1G3-K0-explore-DONEZZ" in out


@needs_bash
def test_e2_has_NO_default_record_stamp_a_missing_REC_C12_refuses_before_the_gate(tmp_path):
    rc, out, got = _gate(tmp_path, "r9gate3_thor_gate.sh", ["explore", "K0"], {"R9_STUDY": "e2"}, "r9e2", "r9e2", "K0", None, rec_file=False)
    assert rc == 10 and got is None and "ZZR9E1G3-K0-explore-REFUSEDZZ" in out and "REC_C12 stamp" in out, (rc, out)


@needs_bash
@pytest.mark.parametrize("val", ["bogus", ""])
def test_thor_gate_and_rec_explore_REFUSE_an_unknown_study_before_any_gate_call(tmp_path, val):
    rc, out, got = _gate(tmp_path / "a", "r9gate3_thor_gate.sh", ["explore", "C0"], {"R9_STUDY": val}, "r9e1", "r9e1", "C0", None)
    assert rc == 10 and got is None and "ZZR9STUDY-REFUSEDZZ" in out and "ZZR9E1G3-C0-explore-REFUSEDZZ" in out, (rc, out)
    rc, out, got = _gate(tmp_path / "b", "r9gate3_rec_explore.sh", ["C0"], {"R9_STUDY": val}, "r9e1", "r9e1", "C0", None)
    assert rc == 10 and got is None and "ZZR9E1G3-C0-rec-explore-REFUSEDZZ" in out, (rc, out)


@needs_bash
@pytest.mark.parametrize("env,root,prefix,profile,arm,rec12", [(None, "r9e1", "r9e1", "refcv9-e1rig", "C0", "d297ba563d8f"),
                                                                ({"R9_STUDY": "e2"}, "r9e2", "r9e2", "refcv9-e2", "K0", "23fbf8f7e2b1")])
def test_rec_explore_runs_the_records_job_under_the_studys_profile_argv_and_records(tmp_path, env, root, prefix, profile, arm, rec12):
    rc, out, got = _gate(tmp_path, "r9gate3_rec_explore.sh", [arm], env, root, prefix, arm, rec12 if env else None)
    assert got is not None, out
    r = f"/home/nvidia/r8ns/{root}"
    ov = f"{r}/ovf_{arm}_{rec12}"
    want = [f"{r}/launch_tree/stack/scripts/launch_gate.py", "run", "--profile", profile, "--checks", "G-MAP-OVERFIT,G-BOX-OVERFIT", "--tree", f"{r}/launch_tree", "--commit", SHA,
            "--argv-file", f"{r}/launch_tree/stack/ops/runs.d/{prefix}_{arm}.argv.json", "--out-dir", f"{r}/gate3_explore/rec_{C12}_{arm}",
            "--map-overfit-record", f"{ov}/run_map/g_map_overfit.json", "--map-overfit-closure", f"{ov}/map_closure.json",
            "--box-overfit-record", f"{ov}/run_box/gbo_binding.json", "--box-overfit-closure", f"{ov}/box_closure.json"]
    assert got == want, got


@needs_bash
def test_MUTATION_a_broken_e2_mapping_makes_the_K0_gate_command_wrong(tmp_path):
    src = STUDY.read_text(encoding="utf-8")
    bad = src.replace("R9_PROFILE=refcv9-e2;", "R9_PROFILE=refcv9-e1rig;")
    assert bad != src
    rc, out, got = _gate(tmp_path, "r9gate3_thor_gate.sh", ["explore", "K0"], {"R9_STUDY": "e2"}, "r9e2", "r9e2", "K0", "23fbf8f7e2b1", study_text=bad)
    want = _expected_explore("r9e2", "r9e2", "refcv9-e2", "K0", "23fbf8f7e2b1") + EXPLORE_CHECKS + ["--out-dir", f"/home/nvidia/r8ns/r9e2/gate3_explore/{C12}_K0"]
    assert got != want                                                                                                   # the mutant is visible to the test above


# ================================================ no operational e1 literal outside the study file; the hops carry the study ==
FAMILY = ("r9gate3_chain.sh", "r9gate3_thor_gate.sh", "r9gate3_rec_explore.sh", "r9gate3_restart.sh", "r9gate3_devbox_pinned.sh", "r9gate3_launch_pinned.ps1")
E1_LITERALS = (r"/home/nvidia/r8ns/r9e1", r"refcv9-e1rig", r"runs\.d/r9e1_", r"refcv9_e1", r"explore C0", r"_C0\b", r"ZZR9E1G3-C0", r"d297ba563d8f(?![0-9a-f])", r"ARM=r9e1_")


def _code(name: str):
    return [ln for ln in (GATE3 / name).read_text(encoding="utf-8").split("\n") if ln.strip() and not ln.lstrip().startswith("#")]


@pytest.mark.parametrize("name", FAMILY)
def test_no_operational_e1_literal_is_left_outside_the_study_file(name):
    for ln in _code(name):
        if name.endswith(".ps1") and ln.startswith("param("):                            # the ps1's defaults ARE the e1 study's (what it always did)
            continue
        for pat in E1_LITERALS:
            assert not re.search(pat, ln), (name, pat, ln)


@pytest.mark.parametrize("name", [f for f in FAMILY if f.endswith(".sh") and f != "r9gate3_restart.sh"])
def test_every_script_sources_the_study_from_its_own_directory_and_refuses_when_it_is_absent(name):
    src = (GATE3 / name).read_text(encoding="utf-8")
    assert '. "$HERE/r9gate3_study.sh" ||' in src and 'HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"' in src, name
    assert src.index('. "$HERE/r9gate3_study.sh"') < src.index("R9_ROOT") or name == "r9gate3_devbox_pinned.sh"


def test_restart_ships_the_study_file_FIRST_and_hands_the_selector_to_every_child():
    s = (GATE3 / "r9gate3_restart.sh").read_text(encoding="utf-8")
    assert re.search(r"scp -q \$HERE/r9gate3_study\.sh \$HERE/r9gate3_chain\.sh ", s), "the study file must precede the scripts that source it"
    assert s.count('$SE ') == 3 and 'SE="R9_STUDY=$R9_STUDY"' in s                                   # rec_explore, explore, chain (the three ssh'd children)
    assert '-Study "$R9_STUDY" -Stage "$DEVSTAGE_W"' in s                                            # the WMI hop: the environment cannot cross it
    assert "R9W_SHIP_STAGE=$R9_SHIP_STAGE" in s and 'ARMS="${2:-$R9_DEFAULT_ARMS}"; REC12="${3:-$R9_DEFAULT_REC12}"' in s
    assert "explore $R9_EXPLORE_ARM" in s and 'r9w_gsuite_precheck.py" || die' in s
    assert (s.index('. "$HERE/r9gate3_study.sh"') < s.index("R=$R9_ROOT") < s.index("r9w_ship.sh tree"))


def test_the_chain_reads_every_per_study_value_from_the_study():
    s = (GATE3 / "r9gate3_chain.sh").read_text(encoding="utf-8")
    for tok in ("R=$R9_ROOT", "${R9_EXPLORE_ARM}", "$R9_DEFAULT_ARMS", "$R9_STUDY_NOTE", "$SHA $R9_PROFILE >", "${R9_PREFIX}_${A}.argv.json", "ARM=${R9_PREFIX}_$A"):
        assert tok in s, tok
    assert "bash $KIT/r9gate3_thor_gate.sh final $A" in s                                            # the child inherits the EXPORTED R9_STUDY (r9_study_load exports it)


def test_the_pinned_driver_and_its_launcher_take_study_and_stage():
    d = (GATE3 / "r9gate3_devbox_pinned.sh").read_text(encoding="utf-8")
    p = (GATE3 / "r9gate3_launch_pinned.ps1").read_text(encoding="utf-8")
    assert "STAGE=$R9_DEVSTAGE" in d and "ROOT=$R9_ROOT" in d and '--profile "$R9_PROFILE"' in d and "${R9_PREFIX}_${ARM}.argv.json" in d
    assert '[string]$Study = "e1"' in p and '[string]$Stage = "F:\\refcv9_e1\\gate"' in p and "R9_STUDY=' + $Study" in p and "CurrentDirectory = $Stage" in p
    assert "-notmatch '^[A-Za-z0-9]+$'" in p                                                          # the study is embedded in a command line: a name, nothing else


# ============================================================================================ r9e1_bind.sh --argv-prefix ==
@needs_bash
def test_bind_resolves_the_K0_argv_under_the_r9e2_prefix_and_a_dry_run_needs_no_thor(tmp_path):
    out, root = R2._bind(tmp_path, "--argv-prefix", "r9e2", "box", "K0")
    assert "ZZR9E1BIND-DRYOKZZ" in out and "REFUSED" not in out and "runs.d/r9e2_K0.argv.json sha256" in out, out
    assert (root / f"ovf_K0_{R2.HASH[:12]}" / "box.args").is_file()
    out, _ = R2._bind(tmp_path / "x", "map", "K0", "--argv-prefix", "r9e2")                                     # also accepted AFTER the arm, like every other flag
    assert "ZZR9E1BIND-DRYOKZZ" in out, out


@needs_bash
@pytest.mark.parametrize("prefix", ["r9e3", "", "r9e1_", "R9E1", "r9e1;ls"])
def test_bind_refuses_a_prefix_no_study_owns_before_touching_anything(tmp_path, prefix):
    out, root = R2._bind(tmp_path, "--argv-prefix", prefix, "box", "K0")
    assert "ZZR9E1BIND-REFUSEDZZ" in out and "is not the argv prefix of any study" in out and "DRYOK" not in out, out
    assert not root.exists() or not list(root.iterdir())


@needs_bash
def test_bind_default_prefix_is_r9e1_and_an_arm_of_the_other_study_is_not_found(tmp_path):
    out, _ = R2._bind(tmp_path, "box", "C0")
    assert "runs.d/r9e1_C0.argv.json sha256" in out and "ZZR9E1BIND-DRYOKZZ" in out, out
    out, _ = R2._bind(tmp_path / "y", "--argv-prefix", "r9e2", "box", "C0")
    assert "ZZR9E1BIND-REFUSEDZZ" in out and "no rig argv" in out and "r9e2_C0.argv.json" in out, out
    out, _ = R2._bind(tmp_path / "z", "box", "K0")                                                              # K0 without its prefix: the e1 rig has no K0
    assert "ZZR9E1BIND-REFUSEDZZ" in out and "no rig argv" in out, out


@needs_bash
def test_bind_K0_reproduces_the_measured_facts_map_reuses_C0_box_needs_its_own_record(tmp_path):
    """MEASURED on Thor (LOOP_STATE 16:10, 23fbf8f): K0's map harness invocation is IDENTICAL to C0's, its box invocation is NOT. The dry run re-derives both with the
    harnesses' own --identity-only, on this CPU, with no record, no lock and no GPU."""
    out, _ = R2._bind(tmp_path, "map", "K0", "--argv-prefix", "r9e2", "--reuse-from", "C0", "--reuse-argv-prefix", "r9e1")
    assert "ZZR9E1BIND-DRYREUSE-OKZZ" in out and "harness invocation IDENTICAL to C0's" in out, out
    out, _ = R2._bind(tmp_path / "b", "box", "K0", "--argv-prefix", "r9e2", "--reuse-from", "C0", "--reuse-argv-prefix", "r9e1")
    assert "ZZR9E1BIND-REFUSEDZZ" in out and "NOT identical to C0's" in out and "run its OWN record" in out, out
    out, _ = R2._bind(tmp_path / "c", "map", "K0", "--argv-prefix", "r9e2", "--reuse-from", "C0", "--reuse-argv-prefix", "r9e9")
    assert "ZZR9E1BIND-REFUSEDZZ" in out and "--reuse-argv-prefix 'r9e9'" in out, out
    out, _ = R2._bind(tmp_path / "d", "map", "K0", "--argv-prefix", "r9e2", "--reuse-from", "K0")
    assert "names the arm itself" in out, out                                                                   # same name AND same prefix


def test_every_bind_snippet_that_names_a_profile_reads_the_studys_and_defaults_to_the_rig():
    s = R2.BIND.read_text(encoding="utf-8")
    assert "refcv9-e1rig\"]" not in s.replace('or "refcv9-e1rig"]', "")                                         # no bare PROFILES["refcv9-e1rig"] left
    assert s.count('__import__("os").environ.get("R9_PROFILE") or "refcv9-e1rig"') == 5
    code = next(c for c in R2._snippets() if "box-b3" in c)
    for prof, must in (("refcv9-e1rig", None), ("no-such-profile", "no-such-profile")):
        r = subprocess.run([sys.executable, "-I", "-c", code, str(R2.CODE_TREE), str(ROOT / "nothing.json"), str(ROOT / "ops/runs.d/r9e1_B3.argv.json"), R2.HASH],
                           capture_output=True, text=True, encoding="utf-8", env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", R9_PROFILE=prof), timeout=300)
        if must:
            assert r.returncode != 0 and must in r.stderr, r.stderr[-300:]                                      # the snippet READS R9_PROFILE
        else:
            assert r.returncode == 0 and "REFUSED:" in r.stdout, (r.stdout, r.stderr[-300:])


def test_the_bind_script_and_the_study_file_stay_LF_only():
    for p in (R2.BIND, STUDY, *[GATE3 / f for f in FAMILY]):
        assert b"\r" not in p.read_bytes(), p.name


@pytest.mark.skipif(os.name == "nt", reason="no executable bit on Windows: every file is runnable there, which is why the dev box never saw this")
def test_the_sandbox_shims_are_executable_so_a_linux_path_lookup_finds_them(tmp_path):
    """MEASURED on Thor 2026-11-04 (suite_c3140a46033f): the shims were written 0644, so PATH skipped them -- the REAL nvidia-smi read the training
    GPU ("the GPU is not empty under the lock") and `$PY` could not run: 8 ids of this file and test_r9gate3_lock_waits failed on Thor only."""
    sb = _sandbox(tmp_path)
    for name in ("flock", "nvidia-smi"):
        assert shutil.which(name, path=(sb / "bin").as_posix()) == (sb / "bin" / name).as_posix(), name
    assert os.access(sb / "home/nvidia/venvs/tanitad-train/bin/python", os.X_OK)
