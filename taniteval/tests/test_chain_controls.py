"""chain_controls (INSTFIX, 2026-10-18): a chain stage may only GATE on a control that some stage PRODUCES.

THE DEFECT.  The free-point navtest chain's `checks` stage gated on `<work>/D3.json`; no stage wrote it
(`fp_smoke.py --swap-cams` wrote `<work>/smoke_swap/smoke_result.json`, was not in the chain, and a human `cp`
was the only link).  The chain finished `ZZCHAIN_DONE INVALID` -- the control reported *missing* only at the
FINAL read, after every expensive stage.  RESULT s5: "the `checks` stage must refuse at *startup*, not at the
read, when a gated control has no producer stage."

WHAT IS PINNED (every group has a control that must read a known value, and a mutation that must go RED):

  (a) KNOWN ANSWERS of `check_plan`: producer earlier -> clean; none -> NO-PRODUCER; later or same stage ->
      LATE-PRODUCER; globs; `external` (must exist when a work dir is given) and `arriving` (must not);
      duplicate stage / double producer; a producer whose script never names the file -> SOURCE-SILENT.
  (b) THE REAL CHAIN: the stage list is read from the BANKED `fp_chain.py` (AST), the gate list of `checks`
      from the BANKED `fp_checks.py` (scanner) -- nothing hand-copied.  The ORIGINAL plan is refused with exactly
      one problem, NO-PRODUCER for D3.json; the FIXED plan (a swap-smoke stage and a link stage) is clean.
  (b') THE IMPROVED COPY (`FlyWheels/.../2026-10-18-instfix/code/fp_chain_v2.py`): its declared plan passes its own
      startup gate through the real `main --check-only`; dropping the D3 stages makes `main` exit 2 with
      `ZZCHAIN_DONE REFUSED_AT_STARTUP` before any stage; a missing dev-box input is refused too.
  (c) STARTUP, NOT THE READ: a miniature chain run through the adoption pattern.  Without the producer it is
      refused before ANY stage ran; the contrast control (the same chain without the gate) runs every stage and
      ends INVALID -- the original behaviour.
  (d) MUTATION: removing the producer declaration (the stage, or just its `produces`) turns (b)/(c) RED;
      moving the producer after its consumer turns it RED; a producer script that does not name the file
      turns it RED; an undeclared read injected into the checks script turns the drift tripwire RED.
  (e) `verify_stage_outputs` / `verify_gates_present` (the run-time ledger) and the scanner, on synthetic
      sources with known answers, plus the CLI exit codes.

CPU only; stdlib only.  ASCII-only assertion text (cp1252 console).
"""
from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_TESTS = Path(__file__).resolve().parent
_TANI = _TESTS.parent
_REPO = _TANI.parent
_CC_PATH = _TANI / "tools" / "chain_controls.py"
_FP = (_REPO / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-09-20-refe-plan"
       / "raw" / "2026-10-17-free-point-navtest" / "code")

_spec = importlib.util.spec_from_file_location("chain_controls_under_test", _CC_PATH)
CC = importlib.util.module_from_spec(_spec)
sys.modules["chain_controls_under_test"] = CC
_spec.loader.exec_module(CC)

Stage, ChainPlan = CC.Stage, CC.ChainPlan


def _kinds(problems):
    return sorted(p.split(":", 1)[0] for p in problems)


# =========================================================================== #
# (a) known answers                                                            #
# =========================================================================== #
def _two(producer_idx: int, *, produces=("D3.json",), gates=("D3.json",)):
    """A two-stage plan; the producer sits at index `producer_idx` (0 = before the consumer, 1 = after)."""
    prod = Stage("prod", produces=produces)
    cons = Stage("cons", gates_on=gates)
    return ChainPlan(stages=(prod, cons) if producer_idx == 0 else (cons, prod))


def test_a1_a_producer_in_an_earlier_stage_is_clean():
    assert CC.check_plan(_two(0)) == []


def test_a2_no_producer_is_refused_naming_stage_and_control():
    plan = ChainPlan(stages=(Stage("prod", produces=("other.json",)), Stage("checks", gates_on=("D3.json",))))
    probs = CC.check_plan(plan)
    assert _kinds(probs) == ["NO-PRODUCER"], probs
    assert "'checks'" in probs[0] and "'D3.json'" in probs[0], probs[0]


def test_a3_a_producer_AFTER_its_consumer_is_no_producer():
    probs = CC.check_plan(_two(1))
    assert _kinds(probs) == ["LATE-PRODUCER"], probs
    assert "'prod'" in probs[0], probs[0]


def test_a3b_a_stage_cannot_be_its_own_producer():
    plan = ChainPlan(stages=(Stage("checks", gates_on=("D3.json",), produces=("D3.json",)),))
    assert _kinds(CC.check_plan(plan)) == ["LATE-PRODUCER"], CC.check_plan(plan)


def test_a4_globs_cover_names_and_a_glob_gate_needs_a_matching_declaration():
    ok = ChainPlan(stages=(Stage("picks", produces=("seam_*.npz",)),
                           Stage("checks", gates_on=("seam_BASE.npz", "seam_*.npz"))))
    assert CC.check_plan(ok) == []
    bad = ChainPlan(stages=(Stage("picks", produces=("seam_BASE.npz",)), Stage("checks", gates_on=("seam_*.npz",))))
    assert _kinds(CC.check_plan(bad)) == ["NO-PRODUCER"], CC.check_plan(bad)     # one named file does not cover a glob


def test_a5_external_must_exist_when_a_work_dir_is_given(tmp_path):
    plan = ChainPlan(stages=(Stage("checks", gates_on=("IC1.json",)),), external={"IC1.json": "fp_capture.py check-a9"})
    assert CC.check_plan(plan) == []                                              # no work dir: declaration only
    assert _kinds(CC.check_plan(plan, str(tmp_path))) == ["EXTERNAL-ABSENT"], CC.check_plan(plan, str(tmp_path))
    (tmp_path / "IC1.json").write_text("{}", encoding="utf-8")
    assert CC.check_plan(plan, str(tmp_path)) == []                               # the control: present -> clean


def test_a5b_an_external_that_names_no_producer_is_refused():
    plan = ChainPlan(stages=(Stage("checks", gates_on=("IC1.json",)),), external={"IC1.json": "  "})
    assert "EXTERNAL-UNNAMED" in _kinds(CC.check_plan(plan))


def test_a6_arriving_needs_a_named_producer_but_no_startup_file(tmp_path):
    plan = ChainPlan(stages=(Stage("picks", gates_on=("out/chunk_000.npz",)),), arriving={"out/chunk_000.npz": "fp_ship.py pull"})
    assert CC.check_plan(plan, str(tmp_path)) == []                               # the pull has not happened yet: fine
    probs = CC.verify_gates_present(plan.stages[0], str(tmp_path))                # ... but the STAGE refuses if it never came
    assert len(probs) == 1 and "chunk_000" in probs[0], probs
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "chunk_000.npz").write_bytes(b"x")
    assert CC.verify_gates_present(plan.stages[0], str(tmp_path)) == []


def test_a7_duplicate_stage_and_double_producer_are_refused():
    dup = ChainPlan(stages=(Stage("a", produces=("x",)), Stage("a", produces=("y",))))
    assert "DUPLICATE-STAGE" in _kinds(CC.check_plan(dup))
    dbl = ChainPlan(stages=(Stage("a", produces=("x",)), Stage("b", produces=("x",))))
    assert "DOUBLE-PRODUCER" in _kinds(CC.check_plan(dbl))
    assert "EMPTY-PLAN" in _kinds(CC.check_plan(ChainPlan(stages=())))


def test_a8_source_check_a_producer_script_that_never_names_the_file_is_refused(tmp_path):
    (tmp_path / "writes_it.py").write_text("import json\njson.dump({}, open(f'{w}/D3.json', 'w'))\n", encoding="utf-8")
    (tmp_path / "silent.py").write_text("print('does something else entirely')\n", encoding="utf-8")
    good = ChainPlan(stages=(Stage("smoke", produces=("D3.json",), script="writes_it.py"), Stage("checks", gates_on=("D3.json",))))
    bad = ChainPlan(stages=(Stage("smoke", produces=("D3.json",), script="silent.py"), Stage("checks", gates_on=("D3.json",))))
    gone = ChainPlan(stages=(Stage("smoke", produces=("D3.json",), script="nope.py"), Stage("checks", gates_on=("D3.json",))))
    assert CC.check_plan(good, script_root=str(tmp_path)) == []
    assert _kinds(CC.check_plan(bad, script_root=str(tmp_path))) == ["SOURCE-SILENT"], CC.check_plan(bad, script_root=str(tmp_path))
    assert _kinds(CC.check_plan(gone, script_root=str(tmp_path))) == ["SCRIPT-UNREADABLE"]      # fail closed
    assert CC.check_plan(bad, script_root=str(tmp_path), verify_source=False) == []             # the switch is explicit
    # a glob product: BOTH literal pieces must appear
    (tmp_path / "seam.py").write_text("np.savez(f'{w}/seam_{p}.txt')\n", encoding="utf-8")
    g = ChainPlan(stages=(Stage("picks", produces=("seam_*.npz",), script="seam.py"), Stage("c", gates_on=("seam_X.npz",))))
    assert _kinds(CC.check_plan(g, script_root=str(tmp_path))) == ["SOURCE-SILENT"]             # '.npz' is never named


def test_a9_require_producers_raises_with_every_problem_and_returns_the_ledger():
    plan = ChainPlan(stages=(Stage("a", gates_on=("m1.json", "m2.json")),))
    with pytest.raises(CC.ChainPlanError) as ei:
        CC.require_producers(plan)
    msg = str(ei.value)
    assert "m1.json" in msg and "m2.json" in msg and "startup" in msg, msg                     # all of them, not the first
    ok = ChainPlan(stages=(Stage("p", produces=("x.json",)), Stage("c", gates_on=("x.json", "e.json", "a.json"))),
                   external={"e.json": "somebody"}, arriving={"a.json": "the pull"})
    assert CC.require_producers(ok) == {"x.json": "p", "e.json": "external: somebody", "a.json": "arriving: the pull"}


# =========================================================================== #
# (e) the run-time ledger and the scanner, on synthetic sources                #
# =========================================================================== #
def test_e1_verify_stage_outputs_missing_empty_present_and_glob(tmp_path):
    st = Stage("s", produces=("a.json", "b.json", "seam_*.npz"))
    assert len(CC.verify_stage_outputs(st, str(tmp_path))) == 3                  # nothing written: all three reported
    (tmp_path / "a.json").write_text("", encoding="utf-8")                        # empty is NOT a product
    (tmp_path / "b.json").write_text("{}", encoding="utf-8")
    (tmp_path / "seam_X.npz").write_bytes(b"z")
    probs = CC.verify_stage_outputs(st, str(tmp_path))
    assert len(probs) == 1 and "EMPTY" in probs[0] and "a.json" in probs[0], probs
    (tmp_path / "a.json").write_text("{}", encoding="utf-8")
    assert CC.verify_stage_outputs(st, str(tmp_path)) == []                      # the control: all present -> clean


_SYN = '''
import json, os
import numpy as np

def stage_one(work):
    stp = f"{work}/score_logs/{label}.status.json"
    status = json.load(open(stp)) if os.path.exists(stp) else {}
    d3p = f"{work}/D3.json"
    d3 = json.load(open(d3p)) if os.path.exists(d3p) else {"missing": True}
    z = np.load(f"{work}/seam_{p}.npz")
    json.dump(status, open(f"{work}/out_a.json", "w"))
    np.savez(f"{work}/points.npz", x=1)
    open(os.path.join(work, "log.txt"), "a").write("x")
    return json.load(open(f"{work}/{f}.json"))

def stage_two(work):
    stp = f"{work}/something_else.json"          # same variable name as in stage_one, different path
    return open(stp).read()

def other(DATA):
    return open(f"{DATA}/not_in_work.json").read()
'''


def test_e2_scanner_known_answer_reads_writes_unresolved_and_scope():
    r = CC.scan_reads(_SYN)
    assert r["reads"] == {"score_logs/*.status.json", "D3.json", "seam_*.npz", "something_else.json"}, r
    assert r["writes"] == {"out_a.json", "points.npz", "log.txt"}, r
    assert r["unresolved"] == {"*.json"}, r                                       # f"{work}/{f}.json": reported, not dropped
    assert not any("not_in_work" in x for x in r["reads"]), r                     # a non-work root is not claimed


def test_e3_drift_tripwire_flags_an_undeclared_read_and_only_that():
    declared = Stage("checks", gates_on=("score_logs/*.status.json", "D3.json", "seam_*.npz", "something_else.json"),
                     produces=("out_a.json", "points.npz", "log.txt"))
    assert CC.check_declared_reads(declared, _SYN) == []                          # control: fully declared -> clean
    short = Stage("checks", gates_on=("D3.json", "seam_*.npz", "something_else.json"), produces=("out_a.json", "points.npz", "log.txt"))
    probs = CC.check_declared_reads(short, _SYN)
    assert len(probs) == 1 and "score_logs/*.status.json" in probs[0], probs
    optional = Stage("checks", gates_on=("D3.json", "seam_*.npz", "something_else.json"), optional_reads=("score_logs/*.status.json",),
                     produces=("out_a.json", "points.npz", "log.txt"))
    assert CC.check_declared_reads(optional, _SYN) == []


# =========================================================================== #
# (b) the REAL free-point chain                                                #
# =========================================================================== #
def _need_fp():
    for f in ("fp_chain.py", "fp_checks.py", "fp_read.py", "fp_smoke.py", "fp_picks.py", "fp_score.py", "fp_capture.py"):
        if not (_FP / f).is_file():
            pytest.skip(f"NO_TREE: the banked free-point package is not in this checkout ({_FP / f})")


def _banked_steps():
    """[(stage name, script basename)] read from the BANKED fp_chain.py's `steps = [...]` by AST."""
    tree = ast.parse((_FP / "fp_chain.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "steps":
            out = []
            for el in node.value.elts:
                name = el.elts[0].value
                first = el.elts[1].elts[0]                                   # f"{HERE}/fp_picks.py"
                script = [v.value for v in first.values if isinstance(v, ast.Constant)][-1].lstrip("/")
                out.append((name, script))
            return out
    raise AssertionError("no `steps = [...]` in the banked fp_chain.py")


#: the products each banked stage writes, READ OFF THE BANKED SCRIPTS' docstrings/code (fp_picks.py:4, fp_score.py:48/91,
#: fp_checks.py:5, fp_read.py:6).  These are the DECLARATIONS the original chain never had.
_PRODUCES = {
    "picks": ("picks_report.json", "points.npz", "seam_*.npz"),
    "score": ("score_logs/*.status.json", "score_status.json"),
    "subset": ("seam_*_sub*.npz",),
    "checks": ("IC3_IC4.json", "IC5.json", "IC6.json", "D1_D3.json", "gates_summary.json"),
    "read": ("result/freepoint_result.json", "result/RESULT_TABLES.md"),
}
#: the `read` stage's gates: it reads the identity controls in a loop (fp_read.py:161, UNRESOLVED by the scanner),
#: so these are declared by hand and the scanner's `unresolved` set is pinned below to prove it is the loop.
_READ_GATES = ("score_logs/*.status.json", "IC1.json", "IC3_IC4.json", "IC5.json", "IC6.json", "D1_D3.json")


def _real_plan(*, fixed: bool, d3_produces=("D3.json",), drop_d3_link=False, d3_after_checks=False):
    steps = _banked_steps()
    checks_reads = CC.scan_reads((_FP / "fp_checks.py").read_text(encoding="utf-8"))["reads"]
    stages = [Stage("ic5_prelude", produces=("ic5_forward.json",), gates_on=("out/chunk_000.npz",))]   # fp_chain.main: the meta copy
    d3 = [Stage("d3_smoke", produces=("smoke_swap/smoke_result.json",), script="fp_smoke.py"),
          Stage("d3_link", gates_on=("smoke_swap/smoke_result.json",), produces=d3_produces)]
    if drop_d3_link:
        d3 = d3[:1]
    for name, script in steps:
        if name == "checks" and fixed and not d3_after_checks:
            stages += d3
        stages.append(Stage(name, script=script, produces=_PRODUCES[name],
                            gates_on=tuple(sorted(checks_reads)) if name == "checks" else (_READ_GATES if name == "read" else ())))
        if name == "checks" and fixed and d3_after_checks:
            stages += d3
    return ChainPlan(stages=tuple(stages), external={"IC1.json": "fp_capture.py check-a9 (dev box, before the chain)"},
                     arriving={"out/chunk_000.npz": "fp_ship.py pull"})


@pytest.fixture()
def fp_work(tmp_path):
    (tmp_path / "IC1.json").write_text(json.dumps({"pass": True}), encoding="utf-8")        # the external control, present
    return tmp_path


def test_b1_the_stage_list_and_the_gate_list_are_read_off_the_banked_code():
    _need_fp()
    assert _banked_steps() == [("picks", "fp_picks.py"), ("score", "fp_score.py"), ("subset", "fp_score.py"),
                               ("checks", "fp_checks.py"), ("read", "fp_read.py")]
    r = CC.scan_reads((_FP / "fp_checks.py").read_text(encoding="utf-8"))
    assert r["reads"] == {"picks_report.json", "points.npz", "seam_*.npz", "ic5_forward.json", "IC1.json", "D3.json"}, r
    assert r["writes"] == set(_PRODUCES["checks"]), r
    rr = CC.scan_reads((_FP / "fp_read.py").read_text(encoding="utf-8"))
    assert rr["unresolved"] == {"*.json"}, rr                                    # the identity-control loop, named
    assert "score_logs/*.status.json" in rr["reads"], rr


def test_b2_the_ORIGINAL_chain_is_refused_at_startup_for_exactly_one_reason_D3(fp_work):
    _need_fp()
    probs = CC.check_plan(_real_plan(fixed=False), str(fp_work), script_root=str(_FP))
    assert len(probs) == 1, probs
    assert probs[0].startswith("NO-PRODUCER") and "'checks'" in probs[0] and "'D3.json'" in probs[0], probs[0]
    with pytest.raises(CC.ChainPlanError, match="D3.json"):
        CC.require_producers(_real_plan(fixed=False), str(fp_work), script_root=str(_FP))


def test_b3_the_FIXED_chain_passes_and_the_ledger_names_the_D3_producer(fp_work):
    _need_fp()
    ledger = CC.require_producers(_real_plan(fixed=True), str(fp_work), script_root=str(_FP))
    assert ledger["D3.json"] == "d3_link" and ledger["IC1.json"].startswith("external:"), ledger
    assert ledger["picks_report.json"] == "picks" and ledger["ic5_forward.json"] == "ic5_prelude", ledger
    assert ledger["IC3_IC4.json"] == "checks" and ledger["smoke_swap/smoke_result.json"] == "d3_smoke", ledger


def test_b4_the_declared_gates_cover_everything_the_banked_checks_script_reads():
    _need_fp()
    plan = _real_plan(fixed=True)
    st = next(s for s in plan.stages if s.name == "checks")
    assert CC.check_declared_reads(st, (_FP / "fp_checks.py").read_text(encoding="utf-8")) == []


# --------------------------------------------------------------------------- #
# (d) MUTATION: each way the producer declaration can be lost goes RED          #
# --------------------------------------------------------------------------- #
def test_d1_mutation_removing_the_producer_stage_turns_the_fixed_plan_RED(fp_work):
    _need_fp()
    assert CC.check_plan(_real_plan(fixed=True), str(fp_work), script_root=str(_FP)) == []        # control: GREEN
    probs = CC.check_plan(_real_plan(fixed=True, drop_d3_link=True), str(fp_work), script_root=str(_FP))
    assert [p for p in probs if p.startswith("NO-PRODUCER") and "'D3.json'" in p], probs          # RED


def test_d2_mutation_removing_only_the_produces_declaration_turns_it_RED(fp_work):
    _need_fp()
    probs = CC.check_plan(_real_plan(fixed=True, d3_produces=()), str(fp_work), script_root=str(_FP))
    assert [p for p in probs if p.startswith("NO-PRODUCER") and "'D3.json'" in p], probs


def test_d3_mutation_moving_the_producer_after_its_consumer_turns_it_RED(fp_work):
    _need_fp()
    probs = CC.check_plan(_real_plan(fixed=True, d3_after_checks=True), str(fp_work), script_root=str(_FP))
    assert [p for p in probs if p.startswith("LATE-PRODUCER") and "'D3.json'" in p], probs


def test_d4_mutation_a_producer_script_that_does_not_name_the_file_turns_it_RED(fp_work, tmp_path):
    """The label is declared but the code never writes the file -- the original bug with a label on it."""
    _need_fp()
    plan = _real_plan(fixed=True)
    stages = tuple(Stage(s.name, produces=s.produces, gates_on=s.gates_on, script="fp_capture.py") if s.name == "d3_smoke" else s
                   for s in plan.stages)                                        # fp_capture.py never mentions smoke_result.json
    probs = CC.check_plan(ChainPlan(stages=stages, external=plan.external, arriving=plan.arriving), str(fp_work), script_root=str(_FP))
    assert [p for p in probs if p.startswith("SOURCE-SILENT") and "smoke_result.json" in p], probs


def test_d5_mutation_an_undeclared_gate_injected_into_the_checks_script_turns_the_tripwire_RED():
    _need_fp()
    src = (_FP / "fp_checks.py").read_text(encoding="utf-8")
    st = next(s for s in _real_plan(fixed=True).stages if s.name == "checks")
    assert CC.check_declared_reads(st, src) == []                                  # control: GREEN
    needle = '    d3p = f"{work}/D3.json"\n'
    assert src.count(needle) == 1
    mutated = src.replace(needle, needle + '    d4 = json.load(open(f"{work}/D4.json")) if os.path.exists(f"{work}/D4.json") else {"missing": True}\n')
    probs = CC.check_declared_reads(st, mutated)
    assert len(probs) == 1 and "D4.json" in probs[0], probs                         # RED, naming the new control


def test_d6_mutation_the_external_control_vanishing_turns_the_startup_gate_RED(fp_work):
    _need_fp()
    (fp_work / "IC1.json").unlink()
    probs = CC.check_plan(_real_plan(fixed=True), str(fp_work), script_root=str(_FP))
    assert _kinds(probs) == ["EXTERNAL-ABSENT"], probs


# =========================================================================== #
# (c) STARTUP, NOT THE READ -- a miniature chain through the adoption pattern  #
# =========================================================================== #
def _mini_plan(with_producer: bool) -> ChainPlan:
    stages = [Stage("make_a", produces=("a.json",)), Stage("slow", gates_on=("a.json",), produces=("b.json",))]
    if with_producer:
        stages.append(Stage("make_d3", produces=("D3.json",)))
    stages += [Stage("checks", gates_on=("b.json", "D3.json"), produces=("summary.json",))]
    return ChainPlan(stages=tuple(stages))


def _runners(ran: list, work: Path):
    def mk(files):
        def go(name):
            ran.append(name)
            for f in files:
                (work / f).write_text("{}", encoding="utf-8")
        return go

    def checks(name):
        ran.append(name)
        d3 = work / "D3.json"
        (work / "summary.json").write_text(json.dumps({"status": "VALID" if d3.exists() else "INVALID"}), encoding="utf-8")
    return {"make_a": mk(["a.json"]), "slow": mk(["b.json"]), "make_d3": mk(["D3.json"]), "checks": checks}


def _adopted_chain(plan: ChainPlan, work: Path, ran: list) -> str:
    """The adoption pattern from chain_controls' docstring: gate BEFORE any stage, ledger AFTER each."""
    try:
        CC.require_producers(plan, str(work))
    except CC.ChainPlanError:
        return "REFUSED_AT_STARTUP"
    run = _runners(ran, work)
    for st in plan.stages:
        CC.raise_on(CC.verify_gates_present(st, str(work)), "gate missing")
        run[st.name](st.name)
        CC.raise_on(CC.verify_stage_outputs(st, str(work)), "stage wrote less than it declared")
    return json.loads((work / "summary.json").read_text(encoding="utf-8"))["status"]


def _unguarded_chain(plan: ChainPlan, work: Path, ran: list) -> str:
    """The ORIGINAL behaviour: no startup check, the missing control surfaces at the final stage."""
    run = _runners(ran, work)
    for st in plan.stages:
        run[st.name](st.name)
    return json.loads((work / "summary.json").read_text(encoding="utf-8"))["status"]


def test_c1_control_without_the_gate_every_stage_runs_and_the_chain_ends_INVALID(tmp_path):
    ran = []
    no_producer = ChainPlan(stages=tuple(s for s in _mini_plan(True).stages if s.name != "make_d3"))
    status = _unguarded_chain(no_producer, tmp_path, ran)
    assert status == "INVALID" and ran == ["make_a", "slow", "checks"], (status, ran)        # all the work, then 'missing'


def test_c2_with_the_gate_a_missing_producer_is_refused_before_ANY_stage_runs(tmp_path):
    ran = []
    assert _adopted_chain(_mini_plan(with_producer=False), tmp_path, ran) == "REFUSED_AT_STARTUP"
    assert ran == [] and not list(tmp_path.iterdir()), (ran, list(tmp_path.iterdir()))        # nothing ran, nothing written


def test_c3_with_the_producer_declared_the_same_chain_runs_to_VALID(tmp_path):
    ran = []
    assert _adopted_chain(_mini_plan(with_producer=True), tmp_path, ran) == "VALID"
    assert ran == ["make_a", "slow", "make_d3", "checks"], ran


def test_c4_mutation_removing_the_producer_declaration_from_the_working_chain_turns_it_RED(tmp_path):
    ok_dir, bad_dir = tmp_path / "ok", tmp_path / "bad"
    ok_dir.mkdir()
    bad_dir.mkdir()
    ran_ok, ran_bad = [], []
    ok = _adopted_chain(_mini_plan(with_producer=True), ok_dir, ran_ok)                       # control: GREEN
    bad = _adopted_chain(_mini_plan(with_producer=False), bad_dir, ran_bad)                   # the declaration removed: RED
    assert ok == "VALID" and bad == "REFUSED_AT_STARTUP" and ran_bad == [], (ok, bad, ran_bad)


def test_c5_a_stage_that_claims_a_product_and_wrote_nothing_is_caught_at_that_stage(tmp_path):
    plan = ChainPlan(stages=(Stage("liar", produces=("x.json",)), Stage("next", gates_on=("x.json",))))
    CC.require_producers(plan)                                                    # the DECLARATION is fine ...
    probs = CC.verify_stage_outputs(plan.stages[0], str(tmp_path))                # ... the ledger catches the stage
    assert len(probs) == 1 and "x.json" in probs[0], probs


# =========================================================================== #
# CLI                                                                          #
# =========================================================================== #
def _cli(plan: dict, tmp_path: Path, *extra):
    p = tmp_path / "plan.json"
    p.write_text(json.dumps(plan), encoding="utf-8")
    r = subprocess.run([sys.executable, str(_CC_PATH), "check", str(p), *extra], capture_output=True, text=True, timeout=120)
    return r.returncode, r.stdout, r.stderr


def test_f1_cli_exit_codes_refused_2_ok_0(tmp_path):
    orphan = {"stages": [{"name": "checks", "gates_on": ["D3.json"]}]}
    rc, out, err = _cli(orphan, tmp_path)
    assert rc == 2 and "CHAIN_PLAN_REFUSED" in err and "D3.json" in err, (rc, out, err)
    fixed = {"stages": [{"name": "d3", "produces": ["D3.json"]}, {"name": "checks", "gates_on": ["D3.json"]}]}
    rc, out, err = _cli(fixed, tmp_path)
    assert rc == 0 and "CHAIN_PLAN_OK" in out, (rc, out, err)
    ext = {"stages": [{"name": "checks", "gates_on": ["IC1.json"]}], "external": {"IC1.json": "somebody"}}
    assert _cli(ext, tmp_path)[0] == 0
    assert _cli(ext, tmp_path, "--work", str(tmp_path))[0] == 2                    # external, work given, file absent


def test_f2_the_module_is_ascii_only_so_a_cp1252_console_can_print_it():
    _CC_PATH.read_bytes().decode("ascii")


# =========================================================================== #
# (b, cont.) the IMPROVED COPY of the chain, in the new package                #
# =========================================================================== #
_V2 = (_REPO / "FlyWheels" / "TanitAD_EvalFlyWheel" / "incoming" / "2026-10-18-instfix" / "code" / "fp_chain_v2.py")


def _load_v2(monkeypatch):
    _need_fp()
    if not _V2.is_file():
        pytest.skip(f"NO_TREE: the INSTFIX package is not in this checkout ({_V2})")
    monkeypatch.setenv("TANITAD_REPO", str(_REPO))
    spec = importlib.util.spec_from_file_location("fp_chain_v2_under_test", _V2)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fp_chain_v2_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def _stage_externals(v2, work: Path):
    for ctl in v2.build_plan().external:                       # the dev-box files stage A leaves behind
        (work / ctl).parent.mkdir(parents=True, exist_ok=True)
        (work / ctl).write_text("{}", encoding="utf-8")


def test_b5_the_improved_chain_declares_a_plan_that_passes_its_own_startup_gate(monkeypatch, tmp_path):
    v2 = _load_v2(monkeypatch)
    try:
        _stage_externals(v2, tmp_path)
        plan = v2.build_plan()
        ledger = v2.CC.require_producers(plan, work=str(tmp_path), script_root=v2.FPCODE)
        assert ledger["D3.json"] == "d3_link" and ledger["IC1.json"] == "ic1", ledger
        assert [s.name for s in plan.stages][:2] == ["ic5_prelude", "picks"] and "read" == plan.stages[-1].name
        names = [s.name for s in plan.stages]
        assert names.index("d3_smoke") < names.index("d3_link") < names.index("checks"), names
        monkeypatch.setattr(sys, "argv", ["fp_chain_v2.py", "--work", str(tmp_path), "--check-only"])
        assert v2.main() == 0
        assert not (tmp_path / "chain.log").exists()            # --check-only runs nothing and writes nothing
    finally:
        sys.modules.pop("fp_chain_v2_under_test", None)


def test_b6_mutation_dropping_the_d3_stages_from_the_improved_chain_makes_main_refuse_at_startup(monkeypatch, tmp_path):
    v2 = _load_v2(monkeypatch)
    try:
        _stage_externals(v2, tmp_path)
        good = v2.build_plan()
        mutated = v2.CC.ChainPlan(stages=tuple(s for s in good.stages if s.name not in ("d3_smoke", "d3_link")),
                                  external=good.external, arriving=good.arriving)
        monkeypatch.setattr(v2, "build_plan", lambda w=None: mutated)
        monkeypatch.setattr(sys, "argv", ["fp_chain_v2.py", "--work", str(tmp_path), "--check-only"])
        assert v2.main() == 2
        log = (tmp_path / "chain.log").read_text(encoding="utf-8")
        assert "ZZCHAIN_DONE REFUSED_AT_STARTUP" in log and "D3.json" in log, log           # refused BEFORE any stage, naming the control
        assert not list(tmp_path.glob("chain_*.out"))                                        # and no stage ran
    finally:
        sys.modules.pop("fp_chain_v2_under_test", None)


def test_b7_the_improved_chain_refuses_when_an_external_dev_box_file_is_missing(monkeypatch, tmp_path):
    v2 = _load_v2(monkeypatch)
    try:
        monkeypatch.setattr(sys, "argv", ["fp_chain_v2.py", "--work", str(tmp_path), "--check-only"])
        assert v2.main() == 2                                                                 # no inputs_full.json / cap file in the work dir
        assert "EXTERNAL-ABSENT" in (tmp_path / "chain.log").read_text(encoding="utf-8")
    finally:
        sys.modules.pop("fp_chain_v2_under_test", None)
