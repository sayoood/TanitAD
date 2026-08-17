"""The S-T LAUNCH PATH — E1, E2, E5 fixed, and E4 PINNED as an open defect.

⛔ WHY A NEW FILE. `ST_LAUNCH_READINESS.md` (2026-08-17) found five defects in
the path around a transition the ladder suite calls green — 187 tests passed
while `v6_chain.py commands --step S-T` emitted a line that could not load the
S-W checkpoint. Its verdict is the brief for this file:

    "Every one of those defects lives in a seam no test reaches: the chain's
    emitted geometry is never diffed against a predecessor's config.json;
    --v2-subframe is never forwarded through a real encoder; the S-T gate is
    never adjudicated against the default (selector='none') plan; and the seam
    dump's import is never exercised under the launch's own PYTHONPATH.
    A green suite is not a launch rehearsal."

So every test here executes the seam rather than asserting about it.

⚠️ ONE OF THESE PINS A DEFECT THAT IS **NOT FIXED**. `test_E4_*` asserts that
the S-T gate is INCONCLUSIVE **by construction** on the arm the chain plans to
run. That is not a bug in the test — it is the defect, held still so it cannot
be silently "fixed" in a direction nobody chose. See §E4 below.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

_STACK = Path(__file__).resolve().parents[1]
_REPO = _STACK.parent
sys.path.insert(0, str(_STACK))
sys.path.insert(0, str(_STACK / "scripts"))
# ⛔ AND THE SIBLING, EXPLICITLY. `pytest -q` inside `stack/` does NOT see
# `taniteval` — which is E5's exact geometry, reproduced by the suite itself.
# Adding it here is the same act the launch line has to perform; the test that
# a launch WITHOUT it is refused runs in a SUBPROCESS with the narrow path, so
# this line cannot mask the defect it verifies.
sys.path.insert(0, str(_REPO / "taniteval"))

import v6_chain as C                                              # noqa: E402
import train_v6_staged as T                                       # noqa: E402


def _cfg(root, **kw) -> C.ChainConfig:
    c = C.ChainConfig(root=str(root).replace("\\", "/"), dry=True, tiny=True,
                      dry_steps=1)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def _seed(plan, c, extra=()):
    for s in plan:
        Path(s.out).mkdir(parents=True, exist_ok=True)
        args = C.parse_argv_geometry(
            list(C.TINY_GEOMETRY) + ["--n-candidates", str(c.n_candidates)]
            + list(extra)) | {"tac_goal_cond": bool(s.tac_goal_cond),
                              "selector": s.selector}
        (Path(s.out) / "config.json").write_text(json.dumps({"args": args}))


# ============================================================================
# THE DERIVATION — the geometry set is read off the trainer, not listed here
# ============================================================================
def test_the_arg_spec_read_from_SOURCE_matches_the_REAL_parser():
    """⛔ `v6_chain` parses `build_parser`'s `add_argument` calls out of the
    trainer's SOURCE so that `plan`/`commands`/`status` keep working on a pod
    whose torch a `uv pip install` has broken (MEASURED twice on pod4). That
    torch-free read is only trustworthy if it agrees with the parser that
    actually runs — so this pins it, dest by dest."""
    ast_spec = C._ast_arg_spec()
    real = {a.dest: a for a in T.build_parser()._actions
            if a.option_strings and a.dest != "help"}
    assert set(ast_spec) >= set(real), sorted(set(real) - set(ast_spec))
    for dest, act in real.items():
        s = ast_spec[dest]
        assert s.opt in act.option_strings, (dest, s.opt, act.option_strings)
        want_action = {"_StoreTrueAction": "store_true",
                       "_StoreFalseAction": "store_false"}.get(
                           type(act).__name__, "")
        assert s.action == want_action, (dest, s.action, want_action)
        if not want_action:            # a store_true reports nargs 0, not None
            assert s.nargs == ("" if act.nargs is None
                               else str(act.nargs)), dest


def test_the_geometry_set_is_DERIVED_and_covers_what_broke():
    """The set is every dest `build_stack_from_args` reads. It is not asserted
    field-by-field — a hand-list is the bug this replaces — but the fields whose
    ABSENCE produced `size mismatch for encoder.pos` must be in it, or the
    derivation has stopped working."""
    g = C.geometry_dests()
    for dest in ("enc_dim", "enc_depth", "enc_heads", "pred_dim", "pred_depth",
                 "pred_heads", "d_tac", "d_str", "frame_h", "frame_w", "patch",
                 "vit5_encoder", "pred_modern", "n_candidates", "window",
                 "horizons", "selector", "tac_goal_cond"):
        assert dest in g, dest
    assert len(g) > 40, len(g)
    # ⭐ and it grows with the architecture: every V6Config lever wired to a CLI
    # flag joins the check the moment it is wired, without editing v6_chain.
    assert {"agent_slots", "proposals", "mpc_refine"} <= g


# ============================================================================
# E1 — the emitted line CARRIES the ancestor's geometry, and cannot omit it
# ============================================================================
def test_E1_the_emitted_line_carries_the_ancestors_geometry(tmp_path):
    c = _cfg(tmp_path)
    plan = C.build_plan(c)
    _seed(plan, c)
    av = C.trainer_argv(C.step_by_key(plan, "S-T"), c, plan)
    for flag, want in (("--enc-dim", "32"), ("--enc-depth", "1"),
                       ("--pred-dim", "32"), ("--d-tac", "32"),
                       ("--frame-h", "32"), ("--frame-w", "32")):
        assert flag in av, flag
        assert av[av.index(flag) + 1] == want, flag


def test_E1_a_launch_line_with_NO_geometry_source_is_REFUSED(tmp_path):
    """⛔ THE DEFECT ITSELF. `commands --step S-T` emitted an argv with no model
    geometry at all, so the stack built at the trainer's defaults: MEASURED
    2026-08-17, 87 926 473 params / 407 keys / encoder.pos [1, 640, 384] against
    a 336 575 049-param, 575-key, [1, 640, 768] checkpoint. Emitting nothing is
    now impossible: with no ancestor record and no --geometry-from, the chain
    REFUSES rather than emit a line that cannot load."""
    c = _cfg(tmp_path)
    plan = C.build_plan(c)
    with pytest.raises(SystemExit, match="MODEL GEOMETRY and cannot read it"):
        C.trainer_argv(C.step_by_key(plan, "S-T"), c, plan)


def test_E1_the_guard_REFUSES_a_geometry_free_argv(tmp_path):
    """⛔ AND THE GUARD NOW CHECKS WHAT ITS NAME PROMISES. `assert_geometry_carry`
    read the predecessor's config.json — which holds EVERY geometry field — and
    compared exactly TWO (`selector`, `tac_goal_cond`). It had the data in hand
    and did not look."""
    c = _cfg(tmp_path)
    plan = C.build_plan(c)
    _seed(plan, c)
    st = C.step_by_key(plan, "S-T")
    naked = ["--stage", "S-T", "--out", st.out, "--steps", "10",
             "--n-candidates", str(c.n_candidates), "--tac-goal-cond"]
    with pytest.raises(SystemExit) as ei:
        C.assert_geometry_carry(st, plan, naked, c)
    msg = str(ei.value)
    assert "MODEL GEOMETRY does not match" in msg
    assert "--enc-dim" in msg and "TRAINER'S DEFAULTS" in msg
    # the healthy argv passes the SAME check, and says how many fields it saw
    ok = C.assert_geometry_carry(st, plan, C.trainer_argv(st, c, plan), c)
    assert ok["ok"] is True
    assert ok["full_diff"]["n_checked"] == len(C.geometry_dests())


def test_E1_the_guard_does_not_evaporate_when_the_ancestor_is_off_box(tmp_path):
    """⚠️ MEASURED while building this fix: reading only `<prev.out>/config.json`
    made the whole check return `ok: None` on the dev box — where commands are
    generated and where that path does not exist — so a geometry-free argv
    PASSED. The guard reads the same source the emitter carried from."""
    c = _cfg(tmp_path)
    plan = C.build_plan(c)
    st = C.step_by_key(plan, "S-T")
    banked = tmp_path / "banked.json"
    banked.write_text(json.dumps(
        {"args": C.parse_argv_geometry(list(C.TINY_GEOMETRY))
         | {"selector": "none", "tac_goal_cond": False}}))
    naked = ["--stage", "S-T", "--out", st.out, "--tac-goal-cond"]
    assert C.assert_geometry_carry(st, plan, naked)["ok"] is None   # no cfg
    c.geometry_from = str(banked)
    with pytest.raises(SystemExit, match="MODEL GEOMETRY does not match"):
        C.assert_geometry_carry(st, plan, naked, c)


def test_E1_a_DECLARED_geometry_change_is_reported_not_refused(tmp_path):
    """An operator who writes `--enc-dim 64` has made a visible decision and
    `load_stage_init` will adjudicate it. What is refused is the SILENT
    OMISSION — which is what E1 actually was."""
    c = _cfg(tmp_path)
    plan = C.build_plan(c)
    _seed(plan, c)
    st = C.step_by_key(plan, "S-T")
    av = C.trainer_argv(st, c, plan) + ["--adapter-hidden", "64"]
    r = C.assert_geometry_carry(st, plan, av, c)
    assert r["full_diff"]["declared_changes"]["adapter_hidden"]["this"] == 64


def test_E1_the_carry_never_forwards_RUN_CONTROL(tmp_path):
    """⛔ The ancestor's record also holds run control. Carrying `--force-rerun`
    forward would let a successor overwrite a DONE run, and carrying `--resume`
    or the ancestor's own `--out` is meaningless. The carry is exactly
    `geometry_dests()`."""
    c = _cfg(tmp_path)
    plan = C.build_plan(c)
    _seed(plan, c)
    for s in plan:
        p = Path(s.out) / "config.json"
        doc = json.loads(p.read_text())
        doc["args"] |= {"force_rerun": True, "resume": "off",
                        "gate_off_reason": "an ancestor's reason",
                        "out": "/somewhere/else"}
        p.write_text(json.dumps(doc))
    av = C.trainer_argv(C.step_by_key(plan, "S-S"), c, plan)
    assert "--force-rerun" not in av
    assert "--resume" not in av
    assert av[av.index("--out") + 1] == C.step_by_key(plan, "S-S").out


# ============================================================================
# E2 — --v2-subframe moves the DATA, not the MODEL, in THIS trainer
# ============================================================================
def test_E2_the_chain_emits_no_subframe(tmp_path):
    """⛔ MEASURED 2026-08-17 on the built production encoder: `--v2-subframe
    176x624` against a 256x640 encoder lets `--init-from` SUCCEED and parity
    PASS, then dies at the FIRST FORWARD with `encoder input is (176, 624) but
    the config declares (256, 640)` — after the corpus has mounted."""
    c = _cfg(tmp_path, dry=False, tiny=False,
             geometry_from=str(tmp_path / "g.json"))
    (tmp_path / "g.json").write_text(
        json.dumps({"args": C.parse_argv_geometry([])}))
    plan = C.build_plan(c)
    for key in ("S-T", "S-S", "S-J"):
        assert "--v2-subframe" not in C.trainer_argv(
            C.step_by_key(plan, key), c, plan), key


@pytest.mark.parametrize("sub,want", [
    (None, None), ("none", None), ("256x640", None), ("176x624", (176, 624)),
])
def test_E2_the_desync_is_detected_from_ARGS_ALONE(sub, want):
    """Args-only, so it fires in milliseconds at startup rather than at the
    first forward. A no-op sub-frame equal to the declared frame is consistent
    and must NOT be refused."""
    argv = ["--stage", "S-T", "--out", "/tmp/x", "--frame-h", "256",
            "--frame-w", "640"] + (["--v2-subframe", sub] if sub else [])
    a = T.build_parser().parse_args(argv)
    assert T.subframe_desync(a) == want


def test_E2_preflight_REFUSES_the_desync_and_says_which_way_out():
    a = T.build_parser().parse_args(
        ["--stage", "S-T", "--out", "/tmp/x", "--frame-h", "256",
         "--frame-w", "640", "--v2-subframe", "176x624", "--init-from", "x",
         "--v2-cache", "c"])
    problems = [p for p in T.preflight(a) if "v2-subframe" in p]
    assert len(problems) == 1
    p = problems[0]
    assert "moves the DATA and NOT the MODEL" in p
    assert "after the compute is paid for" in p
    assert "--frame-h 176 --frame-w 624" in p        # the other legal way out


# ============================================================================
# E5 — the seam dump: turned ON, importable, and LOUD when it is not
# ============================================================================
def test_E5_the_launch_line_carries_BOTH_roots_on_the_pythonpath(tmp_path):
    """⛔ `taniteval` is a SIBLING of `stack/`, not a member of it. MEASURED
    2026-08-17 on Thor AND the dev box: under `PYTHONPATH=<stack>`,
    `import taniteval` is a ModuleNotFoundError — and the trainer caught it
    broadly and printed "training continues", so `--dump-seam-plan` banked
    NOTHING at every save boundary, the first one ~1.8 h in."""
    c = _cfg(tmp_path, dry=False, tiny=False, workdir="/home/n/TanitAD/stack",
             geometry_from=str(tmp_path / "g.json"))
    (tmp_path / "g.json").write_text(
        json.dumps({"args": C.parse_argv_geometry([])}))
    assert c.pythonpath == "/home/n/TanitAD/stack:/home/n/TanitAD/taniteval"
    plan = C.build_plan(c)
    st = C.step_by_key(plan, "S-T")
    for text in (C.launch_line(st, c, plan), C.manifest_text(st, c, plan)):
        assert "/home/n/TanitAD/stack:/home/n/TanitAD/taniteval" in text


def test_E5_the_dump_is_ON_for_S_T_and_above_and_OFF_for_S_W(tmp_path):
    """⚠️ S-W BANKS NOTHING and must not be asked to: its emission head is at
    zero-init, so every control is exactly (0, 0) and `seam_dump_from_plan`
    correctly refuses the plan as DEGENERATE. A degenerate dump cannot answer
    the seam question and must not be produced to make the row non-empty."""
    c = _cfg(tmp_path, dry=False, tiny=False,
             geometry_from=str(tmp_path / "g.json"))
    (tmp_path / "g.json").write_text(
        json.dumps({"args": C.parse_argv_geometry([])}))
    plan = C.build_plan(c)
    assert "--dump-seam-plan" not in C.trainer_argv(
        C.step_by_key(plan, "S-W"), c, plan)
    for key in ("S-T", "S-S", "S-J"):
        s = C.step_by_key(plan, key)
        av = C.trainer_argv(s, c, plan)
        assert av[av.index("--dump-seam-plan") + 1] == C.seam_dir(s), key
        assert C.seam_dir(s) in C.launch_line(s, c, plan)     # and mkdir -p'd


def test_E5_a_seam_dump_that_cannot_import_REFUSES_AT_STARTUP():
    """⛔ THE ANALYSIS-TIME-IMPORT FAMILY. The in-loop catch is non-fatal by
    design (a diagnostic must never kill a 3-day run), which is exactly why the
    failure has to surface at parse time instead — F-16's instrument has
    produced zero real-arm numbers three times running."""
    argv = ["--stage", "S-T", "--out", "/tmp/x", "--init-from", "x",
            "--v2-cache", "c", "--dump-seam-plan", "/tmp/x/seam"]
    a = T.build_parser().parse_args(argv)
    # importable here (the suite runs with taniteval on the path) -> no problem
    assert T.seam_dump_import_error(a) == ""
    assert not [p for p in T.preflight(a) if "dump-seam-plan" in p]
    # and a subprocess WITHOUT taniteval on the path is refused in seconds
    # ⚠️ PYTHONUTF8 is not decoration: without it the refusal's own ⛔ cannot be
    # encoded on this box and the process dies with a UnicodeEncodeError (exit
    # 1) instead of the refusal (exit 2) — the message would be invisible.
    env = {"PYTHONPATH": str(_STACK), "PATH": "", "PYTHONUTF8": "1",
           "PYTHONIOENCODING": "utf-8",
           "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", "")}
    p = subprocess.run([sys.executable, str(_STACK / "scripts"
                                            / "train_v6_staged.py")] + argv,
                       capture_output=True, text=True, env=env, cwd=str(_STACK),
                       timeout=900)
    assert p.returncode == 2, p.stdout[-2000:]
    out = p.stdout + p.stderr
    assert "import taniteval.seam_dump` FAILS" in out
    assert "SIBLING of stack/" in out


def test_E5_the_seam_dump_flag_is_wired_in_the_TRAINER_not_only_the_chain():
    """⭐ `SEAM_INSTRUMENT.md` §8 called this blocked. It is not: the flag exists
    and the training loop calls `taniteval.seam_dump` at the `--save-every`
    boundary, re-using `L["out"]["plan"]` — ZERO extra GPU."""
    src = (_STACK / "scripts" / "train_v6_staged.py").read_text(
        encoding="utf-8")
    assert '"--dump-seam-plan"' in src
    assert "from taniteval.seam_dump import" in src
    assert "seam_dump_from_plan" in src and "save_seam_dump" in src
    from taniteval.seam_dump import save_seam_dump, seam_dump_from_plan  # noqa
    assert callable(seam_dump_from_plan) and callable(save_seam_dump)


# ============================================================================
# E4 — ⛔ PINNED AS AN OPEN DEFECT. NOT FIXED HERE. THE CHOICE IS THE PI'S.
# ============================================================================
# `STAGE_GATE_SPEC["S-T"]["required"]` contains `sel_gap`. `sel_gap` is emitted
# ONLY inside `if w.w_select:` (train_v6_staged.py, the selection-loss block),
# and `w_select > 0` REQUIRES `--selector != none` (the trainer refuses
# otherwise). The chain's DEFAULT S-T step is `selector="none", w_select=0.0`,
# because SEL-1 fired REFUSED on 2026-08-16 against a pre-registered threshold.
#
# ⇒ The gate spec and the default plan CONTRADICT each other, and the
#   contradiction resolves as "INCONCLUSIVE, always". It is the mirror of the
#   three vacuous gates found that week (K3 pinned at 0.5; the pre-S2
#   goal-provenance audit; `_grad_census`'s zero-parameter group): a criterion
#   whose verdict is decided BY CONSTRUCTION rather than by the model. Here it
#   cannot PASS instead of cannot FAIL — same class, same cause.
#
# ⚠️ It propagates: `STAGE_GATE_SPEC["S-S"]["required"]` contains
#   `sel_gap_revalidated`, with the identical dependency.
#
# TWO LEGITIMATE FIXES, BOTH THE PI'S CALL (ST_LAUNCH_READINESS.md §5.2):
#   (a) move `sel_gap` from `required` to `reported` WHEN `selector == "none"`,
#       and record that the S-T certificate then rests on TACTICAL_family alone;
#   (b) keep it required and make the refusal EXPLICIT AND EARLY —
#       `assert_may_launch` refuses an S-T step whose `selector == "none"` while
#       `sel_gap` is required, naming the contradiction, instead of letting it
#       surface as an unexplained INCONCLUSIVE 3.15 GPU-days later.
#
# This test does not choose. It holds the defect still, so that whichever way it
# is fixed, the fix is a DECISION someone made and not a quiet edit.
# ============================================================================
def test_E4_PIN_the_S_T_gate_cannot_read_PASS_on_the_arm_the_chain_plans(
        tmp_path):
    c = _cfg(tmp_path)
    st = C.step_by_key(C.build_plan(c), "S-T")
    assert (st.selector, st.w_select) == ("none", 0.0)      # SEL-1 REFUSED
    assert "sel_gap" in T.STAGE_GATE_SPEC["S-T"]["required"]

    # every OTHER required probe passing is still not a PASS
    probes = {p: {"pass": True} for p in T.STAGE_GATE_SPEC["S-T"]["required"]
              if p != "sel_gap"}
    g = T.stage_gate_dict("S-T", probes)
    assert g["verdict"] == "INCONCLUSIVE"
    assert "sel_gap" in (g["missing_required"] + g["inconclusive_required"])

    # ...and the trainer will not let the arm that COULD emit sel_gap be built
    # by simply turning the weight on: w_select > 0 requires a scorer.
    a = T.build_parser().parse_args(
        ["--stage", "S-T", "--out", str(tmp_path), "--init-from", "x",
         "--v2-cache", "c", "--w-select", "1.0"])
    assert a.selector == "none"
    assert [p for p in T.preflight(a) if "w-select" in p and "selector" in p]


def test_E4_PIN_it_propagates_to_S_S(tmp_path):
    assert "sel_gap_revalidated" in T.STAGE_GATE_SPEC["S-S"]["required"]
    g = T.stage_gate_dict("S-S", {
        p: {"pass": True} for p in T.STAGE_GATE_SPEC["S-S"]["required"]
        if p != "sel_gap_revalidated"})
    assert g["verdict"] == "INCONCLUSIVE"


def test_E4_PIN_the_readiness_finding_is_banked_with_both_options():
    """The pin is only as good as the record of what it is pinning."""
    doc = (_REPO / "TanitAD Research Hub" / "Architecture & Inference"
           / "Implementation" / "incoming" / "2026-08-17-st-launch-readiness"
           / "ST_LAUNCH_READINESS.md")
    assert doc.exists(), doc
    text = doc.read_text(encoding="utf-8")
    assert "it can never read PASS on the planned arm" in text
    assert "Both outcomes are legitimate and the choice is the PI's" in text
