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

⭐ E4 IS NOW RESOLVED (2026-08-17, PI decision: *"solve the s-t gate
contradiction, eventually we need a tactical selector"*). The `test_E4_*` block
below used to pin the OPEN defect; it now pins the resolution, the correction to
its own former statement of the mechanism, and — the requirement that makes any
of it worth having — that the gate can return **each** of its verdicts on a
constructed input. See §E4.
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
# E4 — ⭐ RESOLVED 2026-08-17. These three tests were PINS on an open defect;
#          they now pin the RESOLUTION, and they still name what was chosen.
# ============================================================================
# ⛔ WHAT THE PINS USED TO ASSERT, and why they were right to exist.
# `STAGE_GATE_SPEC["S-T"]["required"]` contains `sel_gap`; the chain's DEFAULT
# S-T step is `selector="none", w_select=0.0` because SEL-1 fired REFUSED on
# 2026-08-16; and on that arm `V6Stack` emits NO `sel_*` key at all, so there is
# no `sel_idx`, `sel_gap_tac` has no argument and `taniteval.selgap` has nothing
# to score. The verdict was INCONCLUSIVE BY CONSTRUCTION — the mirror of the
# three vacuous gates found that week (K3 pinned at 0.5; the pre-S2
# goal-provenance audit; `_grad_census`'s zero-parameter group), except this one
# could not PASS rather than could not FAIL.
#
# ⚠️ AND THE MECHANISM AS ORIGINALLY WRITTEN HERE WAS MISATTRIBUTED. The old
# comment said `sel_gap` "is emitted ONLY inside `if w.w_select:`". That line is
# real and it is NOT the emitter the gate reads: `run_stage_gate` populates
# probes from `--gate-probes`, `X3_isolation`, `spectrum` and `x4_spectra` and
# from NOTHING ELSE — no training-loop log key ever becomes a gate probe. The
# gate's `sel_gap` is the T1 instrument (`taniteval.selgap`), the log key is a
# T0 train-time monitor, and turning `--w-select` on would have made the LOG key
# appear while leaving the GATE exactly as INCONCLUSIVE. Root-cause class: a
# probe read at the WRONG SCOPE (the `df`-on-a-pod / Thor-`free` / cgroup
# family), here as two identically named quantities at two eval tiers. Pinned by
# `test_E4_the_gate_never_reads_the_TRAIN_TIME_log_key`.
#
# ⭐ THE RESOLUTION — (c) BOTH, in the only order that is honest.
#   (c1) The requirement is ARM-CONDITIONAL: `GATE_APPLICABILITY["S-T"]
#        ["sel_gap"] = "has_scorer"`, resolved from the BUILT STACK. On an arm
#        with a scorer `sel_gap` stays REQUIRED — a selector can never be
#        trained without being certified. On an arm without one it is NOT
#        APPLICABLE, recorded with its reason and with the pre-registered
#        measurement that would make it applicable. That is a STRENGTHENING:
#        before, a selector arm and a no-selector arm produced the same verdict,
#        so the gate was equally uninformative about both.
#   (c2) The selector is made REAL by REPAIRING ITS REOPENING PATH, not by
#        overriding the refusal. ⛔ MEASURED 2026-08-17 by execution: the path
#        was DEAD — `read_sw_admission` looked for a top-level `sigma_2s_m` and
#        `e_wc2_sigma_star.py` writes `references_and_ratios.sigma_perax_2s_m`,
#        so a PLANTED sigma of 0.30 m (recovered 0.3026, deep inside the FUNDED
#        band) still produced `verdict: null` and a refused launch. SEL-1 could
#        not be reopened by ANY measurement. Fixed in `resolve_sw_sigma`, and
#        the four-step recipe is now EMITTED by `v6_chain.py admission`.
#
# ⇒ Why not (a) "just enable a selector by default": it would require deleting a
#   FIRED pre-registration (`assert_selector_admissible`), spend 3.15 GPU-days
#   on an arm whose admission criterion is unmeasured, and train a scorer on a
#   goal input measured at sigma/ADE 9.9915 on the nearest available surface.
#   CLAUDE.md rule 5 settles conflicts with pre-registered experiments, not with
#   deference — including deference to a steer we agree with.
# ⇒ Why not (b) alone: it leaves tactical selection unmeasured with no route
#   back. (c2) is what supplies the route, and it is now executable.
# ============================================================================
def test_E4_the_S_T_gate_can_read_PASS_on_the_arm_the_chain_plans(tmp_path):
    """The defect, inverted: the planned arm's gate is now ADJUDICABLE."""
    c = _cfg(tmp_path)
    st = C.step_by_key(C.build_plan(c), "S-T")
    assert (st.selector, st.w_select) == ("none", 0.0)      # SEL-1 REFUSED
    # ⛔ the criterion is NOT deleted from the spec — that was never the fix.
    assert "sel_gap" in T.STAGE_GATE_SPEC["S-T"]["required"]

    no_scorer = {"has_scorer": False, "selector": "none"}
    probes = {p: {"pass": True} for p in T.STAGE_GATE_SPEC["S-T"]["required"]
              if p != "sel_gap"}
    g = T.stage_gate_dict("S-T", probes, arm=no_scorer)
    assert g["verdict"] == "PASS"                    # <- was INCONCLUSIVE
    assert g["required_effective"] == ["TACTICAL_family"]
    # ...and the skipped criterion is RECORDED, never silently dropped
    na = g["not_applicable_required"]
    assert [x["probe"] for x in na] == ["sel_gap"]
    assert "UNCOMPUTABLE" in na[0]["why_not_measured"]
    assert "e_wc2_sigma_star" in na[0]["what_would_make_it_applicable"]
    assert str(C.SW_LATENT_ADMISSION["funded_at_or_below_m"]) \
        in na[0]["what_would_make_it_applicable"]

    # ⛔ AND IT IS A STRENGTHENING: on an arm that HAS a scorer it still binds.
    with_scorer = {"has_scorer": True, "selector": "goal"}
    g2 = T.stage_gate_dict("S-T", probes, arm=with_scorer)
    assert g2["verdict"] == "INCONCLUSIVE"
    assert "sel_gap" in g2["missing_required"]
    assert g2["not_applicable_required"] == []

    # ⛔ and forgetting to describe the arm can only make it HARDER to pass.
    assert T.stage_gate_dict("S-T", probes)["verdict"] == "INCONCLUSIVE"

    # the trainer still refuses w_select>0 with no scorer — unchanged, and it
    # is NOT the mechanism this test is about (see the tier test below).
    a = T.build_parser().parse_args(
        ["--stage", "S-T", "--out", str(tmp_path), "--init-from", "x",
         "--v2-cache", "c", "--w-select", "1.0"])
    assert a.selector == "none"
    assert [p for p in T.preflight(a) if "w-select" in p and "selector" in p]


def test_E4_the_gate_never_reads_the_TRAIN_TIME_log_key(tmp_path):
    """⚠️ The correction to this file's own former claim, pinned.

    `train_v6_staged.py`'s `if w.w_select:` block emits a T0 log key named
    `sel_gap`. The GATE's `sel_gap` is the T1 instrument supplied through
    `--gate-probes`. Enabling `--w-select` would make the log key appear and
    leave the gate unchanged — so "turn the selector loss on" was never a fix.
    """
    src = (_REPO / "stack" / "scripts" / "train_v6_staged.py").read_text(
        encoding="utf-8")
    body = src.split("def run_stage_gate(")[1].split("\ndef ")[0]
    # the ONLY probe sources, read out of the function's own body
    assert "extra_probes" in body and "X3_isolation" in body
    assert 'probes["O6_spectrum"]' in body and 'probes["X4_spectrum_layers"]' in body
    # ...and nothing in it reads the training log
    for forbidden in ("train_log", 'log["sel_gap"]', "metrics.json"):
        assert forbidden not in body, forbidden
    assert T.STAGE_GATE_SPEC["S-T"]["owners"]["sel_gap"].startswith(
        "tanitad.models.tactical.sel_gap_tac")
    assert "T1 tier" in T.STAGE_GATE_SPEC["S-T"]["criteria"]["sel_gap"]
    assert "T1" in T.SEL_GAP_TIER_NOTE and "T0" in T.SEL_GAP_TIER_NOTE


def test_E4_it_is_resolved_at_S_S_too_not_moved_there(tmp_path):
    """A fix that frees S-T and leaves S-S deadlocked has moved the problem."""
    assert "sel_gap_revalidated" in T.STAGE_GATE_SPEC["S-S"]["required"]
    others = {p: {"pass": True} for p in T.STAGE_GATE_SPEC["S-S"]["required"]
              if p != "sel_gap_revalidated"}

    # no-selector lineage: not applicable, and S-S is adjudicable
    g = T.stage_gate_dict("S-S", others,
                          arm={"has_scorer": False, "selector": "none"})
    assert g["verdict"] == "PASS"
    assert [x["probe"] for x in g["not_applicable_required"]] \
        == ["sel_gap_revalidated"]

    # selector lineage: the chain CARRIES `--selector` into S-S for the
    # geometry, so the frozen scorer is in the stack and the revalidation
    # BINDS — which is exactly what STAGE_INVALIDATES["S-S"] demands.
    g2 = T.stage_gate_dict("S-S", others,
                           arm={"has_scorer": True, "selector": "goal"})
    assert g2["verdict"] == "INCONCLUSIVE"
    assert "sel_gap_revalidated" in g2["missing_required"]
    assert T.STAGE_INVALIDATES["S-S"] == ("S-T",)


def test_E4_a_gate_with_nothing_left_to_check_is_REFUSED(tmp_path):
    """⛔ The one way arm-conditionality could have become a loophole.

    If every required criterion were skipped, the gate would "pass" while
    measuring nothing — decoration, which is the defect this whole change
    exists to remove. It is forced to INCONCLUSIVE instead.
    """
    saved = dict(T.GATE_APPLICABILITY["S-T"])
    try:
        T.GATE_APPLICABILITY["S-T"] = {p: "has_scorer" for p in
                                       T.STAGE_GATE_SPEC["S-T"]["required"]}
        g = T.stage_gate_dict("S-T", {}, arm={"has_scorer": False})
        assert g["verdict"] == "INCONCLUSIVE"
        assert g["required_effective"] == []
        assert g["vacuous_gate"]["refused"] is True
    finally:
        T.GATE_APPLICABILITY["S-T"] = saved


def test_E4_the_resolution_is_banked_with_its_reasons():
    """The fix is only as good as the record of WHY it went this way."""
    doc = (_REPO / "TanitAD Research Hub" / "Architecture & Inference"
           / "Implementation" / "incoming" / "2026-08-17-e4-selector-resolution"
           / "E4_SELECTOR_RESOLUTION.md")
    assert doc.exists(), doc
    text = doc.read_text(encoding="utf-8")
    # the decision, the rejected alternative, and the executed proof
    assert "the reopening path was DEAD" in text
    assert "0.3026" in text                  # the recovered planted sigma
    assert "has_scorer" in text
    # the predecessor finding is still reachable, not overwritten
    old = (_REPO / "TanitAD Research Hub" / "Architecture & Inference"
           / "Implementation" / "incoming" / "2026-08-17-st-launch-readiness"
           / "ST_LAUNCH_READINESS.md")
    assert "it can never read PASS on the planned arm" in old.read_text(
        encoding="utf-8")


# ---------------------------------------------------------------------------
# ⛔ THE TRILEMMA — a gate that cannot return one of its verdicts is decoration
# ---------------------------------------------------------------------------
# "A gate that cannot report PASS and a gate that cannot report FAIL are the
# same defect" — and this programme found three of the latter in one week, plus
# (E4) one of the former, plus (below) an ADMISSION gate that could not report
# FUNDED. So the fix is not allowed to be trusted on inspection: each verdict is
# produced HERE, on a constructed input, for BOTH arms.
@pytest.mark.parametrize("has_scorer", [False, True])
@pytest.mark.parametrize("want", ["PASS", "FAIL", "INCONCLUSIVE"])
def test_E4_the_S_T_gate_can_return_EVERY_verdict_on_both_arms(want,
                                                               has_scorer):
    arm = {"has_scorer": has_scorer,
           "selector": "goal" if has_scorer else "none"}
    req = [p for p in T.STAGE_GATE_SPEC["S-T"]["required"]
           if has_scorer or p != "sel_gap"]
    assert req, "the arm must have SOMETHING to check"
    if want == "PASS":
        probes = {p: {"pass": True} for p in req}
    elif want == "FAIL":
        probes = {p: {"pass": True} for p in req} | {req[0]: {"pass": False}}
    else:                                   # INCONCLUSIVE: one probe not run
        probes = {p: {"pass": True} for p in req[1:]}
    g = T.stage_gate_dict("S-T", probes, arm=arm)
    assert g["verdict"] == want, (want, has_scorer, g["verdict"], g)
    # ⛔ a FAIL must survive: X5 gives it no override anywhere downstream
    if want == "FAIL":
        assert g["pass"] is False and g["failed_required"] == [req[0]]


def test_E4_a_FAIL_is_never_softened_into_not_applicable():
    """⛔ The nightmare version of arm-conditionality: a criterion that FAILED
    being re-read as 'this arm could not produce it'. Applicability is resolved
    from the ARM, never from the probe's own verdict, so a failing probe on an
    applicable arm stays a FAIL."""
    g = T.stage_gate_dict(
        "S-T", {"TACTICAL_family": {"pass": False}, "sel_gap": {"pass": False}},
        arm={"has_scorer": True, "selector": "goal"})
    assert g["verdict"] == "FAIL"
    assert g["not_applicable_required"] == []
    # and on the no-scorer arm the criterion that DID run still fails
    g2 = T.stage_gate_dict("S-T", {"TACTICAL_family": {"pass": False}},
                           arm={"has_scorer": False, "selector": "none"})
    assert g2["verdict"] == "FAIL"


def test_E4_a_SUPPLIED_verdict_is_never_discarded_by_the_predicate(tmp_path):
    """⛔ THE DEFECT THE INCUMBENT SUITE CAUGHT IN MY FIRST VERSION.

    `test_the_whole_ladder_hands_off_through_the_WRITTEN_files` plants a
    FAILING `sel_gap` through `--gate-probes` on a stack with NO scorer. The
    first version of `stage_gate_dict` excluded it as not-applicable and read
    **PASS on a rung that had FAILED** — a FAIL erased by the mechanism meant to
    stop vacuous verdicts. Applicability answers "can this arm produce it?"; it
    never licenses discarding a measurement someone supplied.
    """
    arm = {"has_scorer": False, "selector": "none"}
    probes = {"TACTICAL_family": {"pass": True},
              "sel_gap": {"pass": False, "status": "run", "value": 0.91}}
    g = T.stage_gate_dict("S-T", probes, arm=arm)
    assert g["verdict"] == "FAIL"
    assert g["failed_required"] == ["sel_gap"]
    assert g["not_applicable_required"] == []      # NOT skipped
    # ...and the contradiction is REPORTED, not silently resolved either way
    c = g["applicability_conflicts"]
    assert [x["probe"] for x in c] == ["sel_gap"]
    assert c[0]["supplied_pass"] is False and c[0]["predicate"] == "has_scorer"

    # a supplied PASS is honoured the same way (and still flagged)
    g2 = T.stage_gate_dict("S-T", probes | {"sel_gap": {"pass": True}}, arm=arm)
    assert g2["verdict"] == "PASS"
    assert [x["probe"] for x in g2["applicability_conflicts"]] == ["sel_gap"]

    # but a probe with pass:None (the `run_stage_gate` not-applicable stub) is
    # NOT a supplied verdict, so it is still skipped and does not block.
    g3 = T.stage_gate_dict(
        "S-T", {"TACTICAL_family": {"pass": True},
                "sel_gap": {"pass": None, "status": "not-applicable"}}, arm=arm)
    assert g3["verdict"] == "PASS"
    assert [x["probe"] for x in g3["not_applicable_required"]] == ["sel_gap"]
    assert g3["applicability_conflicts"] == []


# ---------------------------------------------------------------------------
# ⭐ (c2) — SEL-1's REOPENING PATH, executed end-to-end against the REAL
#          instrument. This is what makes "eventually we need a tactical
#          selector" reachable instead of a wish.
# ---------------------------------------------------------------------------
def test_E4_the_admission_gate_can_return_FUNDED_at_all(tmp_path):
    """⛔ MEASURED 2026-08-17: IT COULD NOT. `read_sw_admission` looked for a
    top-level `sigma_2s_m`; `e_wc2_sigma_star.py` writes the per-axis 2 s sigma
    at `references_and_ratios.sigma_perax_2s_m`. Name AND level both differed,
    so a planted sigma of 0.30 m (recovered 0.3026 — deep inside the FUNDED
    band) still read `verdict: null` and the launch was refused. The mirror of
    E4: an admission gate that cannot report FUNDED.

    ⚠️ THIS TEST RUNS THE REAL ESTIMATOR. Asserting against a hand-written
    artifact is exactly the fixture defect that let this survive.
    """
    sigma_star = __import__("e_wc2_sigma_star")
    sys.path.insert(0, str(_STACK / "tests"))
    from test_e_wc2_sigma_star import make_dump                  # noqa: E402

    root = tmp_path / "experiments"
    c = C.ChainConfig()
    c.root = str(root).replace("\\", "/")
    (root / c.sw_dir).mkdir(parents=True)

    for planted, expect in ((0.30, "FUNDED"), (1.10, "INCONCLUSIVE"),
                            (2.00, "REFUSED")):
        res = sigma_star.run(make_dump(sigma2=planted, sigma6=2 * planted,
                                       seed=3),
                             features=["pooled", "ctx"], n_boot=0)
        Path(C.admission_path(c)).write_text(
            json.dumps(res, indent=1, default=float), encoding="utf-8")
        adm = C.read_sw_admission(c)
        assert adm["present"] and adm["verdict"] == expect, (planted, adm)
        assert adm["read_at"] == "references_and_ratios.sigma_perax_2s_m"
        assert abs(adm["sigma_2s_m"] - planted) / planted < 0.10

        step = C.step_by_key(C.build_plan(C.ChainConfig(
            root=c.root, st_arms=("goal",))), "S-T:goal")
        if expect == "FUNDED":
            assert C.assert_selector_admissible(step, c)["ok"] is True
        else:
            with pytest.raises(C.ChainRefusal):
                C.assert_selector_admissible(step, c)


def test_E4_the_admission_reader_REFUSES_the_radial_unit(tmp_path):
    """⛔ The unit trap, refused by name. `sigma_radial_rms_m` is sqrt(2)x the
    per-axis sigma the 0.80/1.41 thresholds are defined on, so reading it would
    flip FUNDED -> INCONCLUSIVE on arithmetic alone."""
    hit = C.resolve_sw_sigma({"sigma_radial_rms_m": 0.50, "sigma_6s_m": 1.2})
    assert hit["ok"] is False
    assert "sigma_radial_rms_m" in hit["refused_alternatives"]
    assert "sqrt(2)" in hit["refused_alternatives"]["sigma_radial_rms_m"]
    assert "e_wc2_sigma_star.py" in hit["reason"]
    # ...and absence is never an admission
    assert C.resolve_sw_sigma({})["ok"] is False


def test_E4_the_reopening_recipe_is_EMITTED_not_described(tmp_path):
    """A route that lives only in a docstring is one nobody executes — the
    'please merge in a README' failure, applied to the next decision."""
    c = C.ChainConfig()
    c.root = str(tmp_path).replace("\\", "/")
    r = C.sw_admission_recipe(c)
    steps = {s["n"]: s for s in r["steps"]}
    assert len(steps) == 4
    # ⭐ UPDATED 2026-08-18: step 1 WAS `⛔ NOT BUILT` (correctly — a fabricated
    # command would have been worse). `scripts/v6_dump_sw_latents.py` built it,
    # so the pin now asserts the repaired state. The reason it could not be
    # reused from the REF-C producer is KEPT, because that is the finding.
    assert "refc_dump_latents" in steps[1]["why_not_reusable"]
    assert "v6_dump_sw_latents.py" in steps[1]["cmd"]
    # all four are real commands against real scripts
    for n in (1, 2, 3, 4):
        assert steps[n]["status"].startswith("✅")
        assert "cmd" in steps[n]
    assert "e_wc2_sigma_star.py" in steps[2]["cmd"]
    assert C.SW_LATENT_ADMISSION["artifact"] in steps[2]["cmd"]
    assert "--st-arms goal" in steps[4]["cmd"]
    assert "has_scorer" in r["what_binds_if_funded"]
    # and the refusal an operator actually hits points AT the recipe
    step = C.step_by_key(C.build_plan(C.ChainConfig(
        root=c.root, st_arms=("goal",))), "S-T:goal")
    with pytest.raises(C.ChainRefusal) as e:
        C.assert_selector_admissible(step, c)
    assert "admission" in str(e.value) and "NOT APPLICABLE" in str(e.value)
