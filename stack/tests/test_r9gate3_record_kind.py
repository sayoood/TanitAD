"""refcv9 E1 -- the gate3 kit picks the overfit record KIND the argv builds, for BOTH kinds (MM 2026-11-05).

MEASURED on the 384e79f restart: B3's own box record was bound by r9e1_bind.sh (kind ``box-b3``, ``ovf_B3_<c12>/run_box-b3/gbo_binding.json``, B3 OK
21:50:01Z), but ``r9gate3_thor_gate.sh`` / ``r9gate3_rec_explore.sh`` switched only the MAP kind on the argv (``--r9-map-banded``) and always read the
inherited ``run_box`` record, so the E1-REUSE pick refused B3 with "neither a run record nor a reuse binding ... for the box kind" and B3 left the chain:
built, tested, bound -- and unreachable from its caller.

Pinned here, on the COMMITTED argvs and the scripts' own block (no copy):
 (a) both scripts carry ONE byte-identical kind block, after refuse() / $PY / $ARGV and before the E1-REUSE pick reads a record;
 (b) every committed r9e1 argv gets the kind ``r9e1_bind.sh`` binds it with: B3 / B3p ``map`` + ``box-b3``, M1 / M2a / M2b ``map-banded`` + ``box``, the rest
     ``map`` + ``box`` (the aux-only B1 / B2 / B2c and the T1 pair included) -- classified by the gate's OWN ``E1_BOX_OWN_FLAGS``;
 (c) end to end with the E1-REUSE pick: B3 with its ``run_box-b3`` record PASSES the pick (it was refused before), a non-B3 arm still reads ``run_box``;
 (d) RED: the box-b3 line deleted puts B3 back on ``run_box`` (and the pick refuses it again); a classifier that cannot run REFUSES instead of falling back.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TREE = ROOT.parent
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import launch_gate as LG  # noqa: E402
import test_r9gate3_study as S  # noqa: E402

GATE3 = ROOT / "ops" / "r9e1_gate3"
THOR_GATE, REC_EXPLORE = GATE3 / "r9gate3_thor_gate.sh", GATE3 / "r9gate3_rec_explore.sh"
KIND_BEGIN, KIND_END = "# >>> E1 record kind", "# <<< E1 record kind"
PICK_BEGIN, PICK_END = "# >>> E1-REUSE record pick", "# <<< E1-REUSE record pick"
RUNS = ROOT / "ops" / "runs.d"

BASH = S.BASH                     # the shared resolver (test_launch_gate_e1rig_e1rec2)
needs_bash = S.needs_bash

#: the kinds r9e1_bind.sh binds each committed arm with (its header: "M1/M2a/M2b: box + map-banded; B3/B3p: map + box-b3"; aux-only and T1 arms: map + box)
EXPECTED = {"B3": ("map", "box-b3"), "B3p": ("map", "box-b3"),
            "M1": ("map-banded", "box"), "M2a": ("map-banded", "box"), "M2b": ("map-banded", "box"),
            "C0": ("map", "box"), "C1": ("map", "box"), "B1": ("map", "box"), "B2": ("map", "box"), "B2c": ("map", "box"),
            "T1": ("map", "box"), "T1s": ("map", "box")}
MREC_OF = {"map": "run_map/g_map_overfit.json", "map-banded": "run_map-banded/g_map_overfit.json"}
MCLO_OF = {"map": "map_closure.json", "map-banded": "map-banded_closure.json"}
BREC_OF = {"box": "run_box/gbo_binding.json", "box-b3": "run_box-b3/gbo_binding.json"}
BCLO_OF = {"box": "box_closure.json", "box-b3": "box-b3_closure.json"}


def _block(path: Path, begin: str, end: str) -> str:
    src = path.read_text(encoding="utf-8")
    assert "\r" not in src and src.count(begin) == 1 and src.count(end) == 1, path
    return src[src.index(begin):src.index(end) + len(end)]


def _run(tmp_path: Path, argv: Path, *, kind_block: str | None = None, pick: bool = False, records=(), py: str | None = None):
    """the scripts' OWN kind block (and, with pick, the OWN E1-REUSE pick after it) in bash; prints the four picked paths."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    ov = tmp_path / "ovf"
    for rel in records:
        p = ov / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}", encoding="utf-8")
    body = (kind_block or _block(THOR_GATE, KIND_BEGIN, KIND_END)) + "\n"
    if pick:
        body += _block(THOR_GATE, PICK_BEGIN, PICK_END) + "\n"
    script = tmp_path / "kind.sh"
    script.write_text("set -u\n"
                      f"OV={ov.as_posix()}\nT={TREE.as_posix()}\nARGV={argv.as_posix()}\nPY='{(py or sys.executable).replace(chr(92), '/')}'\n"
                      'refuse() { echo "REFUSED: $*"; exit 10; }\n' + body +
                      'echo "K-MREC=$MREC"\necho "K-MCLO=$MCLO"\necho "K-BREC=$BREC"\necho "K-BCLO=$BCLO"\n', encoding="utf-8")
    r = subprocess.run([BASH, script.as_posix()], capture_output=True, text=True, encoding="utf-8", timeout=300)
    out = r.stdout.replace("\r", "")
    got = {k: v for k, _, v in (ln.partition("=") for ln in out.splitlines() if ln.startswith("K-"))}
    return r.returncode, out + r.stderr, got, ov.as_posix()


def test_both_scripts_carry_the_SAME_kind_block_once_between_its_inputs_and_the_pick():
    assert _block(THOR_GATE, KIND_BEGIN, KIND_END) == _block(REC_EXPLORE, KIND_BEGIN, KIND_END)
    for path in (THOR_GATE, REC_EXPLORE):
        src = path.read_text(encoding="utf-8")
        k = src.index(KIND_BEGIN)
        for dep in ("refuse() {", "PY=/home/nvidia/venvs/tanitad-train/bin/python", "ARGV=$T/stack/ops/runs.d/${R9_PREFIX}_${ARM}.argv.json"):
            assert src.index(dep) < k, (path.name, dep)                                           # everything the block reads is set before it
        assert src.index(KIND_END) < src.index(PICK_BEGIN), path.name                            # the pick reads the KIND's defaults
        assert src.count("MREC=$OV/run_map/g_map_overfit.json") == 1 and src.count("BREC=$OV/run_box/gbo_binding.json") == 1, path.name   # one definition
        assert "E1_BOX_OWN_FLAGS" in _block(path, KIND_BEGIN, KIND_END)                          # the gate's own tuple, no second list


def test_the_expected_table_covers_every_committed_r9e1_argv_and_agrees_with_the_gate_tuple():
    arms = sorted(p.name[len("r9e1_"):-len(".argv.json")] for p in RUNS.glob("r9e1_*.argv.json"))
    assert arms == sorted(EXPECTED), arms                                                         # a new arm must be given its kind here
    for arm, (mk, bk) in EXPECTED.items():
        flags = {f for f, _ in LG.flag_pairs(LG.load_argv_file(RUNS / f"r9e1_{arm}.argv.json"))}
        assert (mk == "map-banded") == ("--r9-map-banded" in flags), arm
        assert (bk == "box-b3") == bool(flags & set(LG.E1_BOX_OWN_FLAGS)), arm


@needs_bash
@pytest.mark.parametrize("arm", sorted(EXPECTED))
def test_each_committed_argv_gets_the_kind_r9e1_bind_binds_it_with(tmp_path, arm):
    rc, out, got, ov = _run(tmp_path, RUNS / f"r9e1_{arm}.argv.json")
    mk, bk = EXPECTED[arm]
    assert rc == 0, out
    assert got == {"K-MREC": f"{ov}/{MREC_OF[mk]}", "K-MCLO": f"{ov}/{MCLO_OF[mk]}", "K-BREC": f"{ov}/{BREC_OF[bk]}", "K-BCLO": f"{ov}/{BCLO_OF[bk]}"}, out


@needs_bash
def test_B3_with_its_own_box_b3_record_PASSES_the_reuse_pick_and_a_base_arm_still_reads_run_box(tmp_path):
    rc, out, got, ov = _run(tmp_path / "b3", RUNS / "r9e1_B3.argv.json", pick=True, records=(MREC_OF["map"], BREC_OF["box-b3"]))
    assert rc == 0 and got["K-BREC"] == f"{ov}/{BREC_OF['box-b3']}" and got["K-MREC"] == f"{ov}/{MREC_OF['map']}", out
    rc, out, got, ov = _run(tmp_path / "c0", RUNS / "r9e1_C0.argv.json", pick=True, records=(MREC_OF["map"], BREC_OF["box"]))
    assert rc == 0 and got["K-BREC"] == f"{ov}/{BREC_OF['box']}", out
    rc, out, got, ov = _run(tmp_path / "m1", RUNS / "r9e1_M1.argv.json", pick=True, records=(MREC_OF["map-banded"], "run_box/reuse_binding.json"))
    assert rc == 0 and got["K-BREC"] == f"{ov}/run_box/reuse_binding.json" and got["K-MREC"] == f"{ov}/{MREC_OF['map-banded']}", out


@needs_bash
def test_RED_the_box_b3_line_deleted_reproduces_the_384e79f_refusal_of_B3(tmp_path):
    good = _block(THOR_GATE, KIND_BEGIN, KIND_END)
    line = '[ -n "$_bown" ] && { BREC=$OV/run_box-b3/gbo_binding.json; BCLO=$OV/box-b3_closure.json; }'
    assert good.count(line) == 1
    rc, out, got, ov = _run(tmp_path / "k", RUNS / "r9e1_B3.argv.json", kind_block=good.replace(line, ":"))
    assert rc == 0 and got["K-BREC"] == f"{ov}/{BREC_OF['box']}", out                            # the mutant reads the inherited record
    mutant = (good.replace(line, ":") + "\n" + _block(THOR_GATE, PICK_BEGIN, PICK_END))
    rc, out, _, _ = _run(tmp_path / "p", RUNS / "r9e1_B3.argv.json", kind_block=mutant, records=(MREC_OF["map"], BREC_OF["box-b3"]))
    assert rc == 10 and "neither a run record" in out and "for the box kind" in out, out        # ... and the pick refuses B3, as on 384e79f


@needs_bash
def test_RED_a_classifier_that_cannot_run_REFUSES_and_never_falls_back_to_run_box(tmp_path):
    rc, out, got, _ = _run(tmp_path / "py", RUNS / "r9e1_B3.argv.json", py=(tmp_path / "no_such_python").as_posix())
    assert rc == 10 and "could not classify the --r9-box-* flags" in out and not got, out
    bad = tmp_path / "bad.argv.json"
    bad.write_text("{not json", encoding="utf-8")
    rc, out, got, _ = _run(tmp_path / "argv", bad)
    assert rc == 10 and "could not classify" in out and not got, out


def test_the_committed_B3_argv_is_a_real_json_launch_argv():
    d = json.loads((RUNS / "r9e1_B3.argv.json").read_text(encoding="utf-8"))
    argv = d["argv"] if isinstance(d, dict) else d
    assert "--r9-box-quality" in argv or set(argv) & set(LG.E1_BOX_OWN_FLAGS)


@needs_bash
def test_END_TO_END_the_real_thor_gate_script_hands_B3_s_box_b3_record_to_the_gate(tmp_path):
    """r9gate3_thor_gate.sh explore B3 in test_r9gate3_study's Thor sandbox, with B3's COMMITTED argv and its records where r9e1_bind.sh writes them
    (run_map + run_box-b3, no run_box): the gate is reached and reads the box-b3 record -- at 384e79f this exact layout was REFUSED by the pick."""
    sb = S._sandbox(tmp_path)
    r = S._arm_files(sb, "r9e1", "r9e1", "B3", "8a95dc2fea3f")
    shutil.copy2(RUNS / "r9e1_B3.argv.json", r / "launch_tree/stack/ops/runs.d/r9e1_B3.argv.json")
    ov = r / "ovf_B3_8a95dc2fea3f"
    shutil.rmtree(ov / "run_box")
    (ov / "box_closure.json").unlink()
    for rel in (BREC_OF["box-b3"], BCLO_OF["box-b3"]):
        (ov / rel).parent.mkdir(parents=True, exist_ok=True)
        (ov / rel).write_text("{}", encoding="utf-8")
    rec = sb / "recorded_args.txt"
    e = dict(PATH=(sb / "bin").as_posix() + os.pathsep + os.environ.get("PATH", ""), REC_OUT=rec.as_posix(), REAL_PY=Path(sys.executable).as_posix())
    rc, out, err = S._run([(sb / "kit" / "r9gate3_thor_gate.sh").as_posix(), "explore", "B3"], env=e, cwd=sb.as_posix(), timeout=120)
    assert "REFUSED" not in out and rec.exists(), (out, err)
    got = rec.read_text(encoding="utf-8").replace("\r", "").splitlines()
    i = got.index("--box-overfit-record")
    assert got[i + 1].endswith("/ovf_B3_8a95dc2fea3f/run_box-b3/gbo_binding.json"), got
    assert got[got.index("--box-overfit-closure") + 1].endswith("/ovf_B3_8a95dc2fea3f/box-b3_closure.json"), got
    assert got[got.index("--map-overfit-record") + 1].endswith("/ovf_B3_8a95dc2fea3f/run_map/g_map_overfit.json"), got
