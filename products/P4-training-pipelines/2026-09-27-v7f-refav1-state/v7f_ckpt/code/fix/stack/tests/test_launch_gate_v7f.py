"""The launch gate's `v7f` profile (train_v6_staged.py): every check's regression arm must go RED
through the gate, and every positive run must be GREEN or name a MEASURED defect of the tip.

⭐ THROUGH THE GATE: the job functions called here (`job_model_v6`, `job_smoke_v6`, `job_clock_v6`)
are exactly what `launch_gate.py check` runs in its child process; the arm is `ctx.arm`, as the
`--arm` flag / `TANITAD_GATE_ARM` sets it. CPU only, a TINY rehearsal (the profile's declared overlay +
a synthetic corpus), 4 smoke steps -- `…/2026-09-27-v7f-refav1-state/v7f_gate/DESIGN.md`.

Facts the positive runs pin (MEASURED by the gate agent, 2026-09-27, at c36b6ddd AND on the v7F merge
`…/2026-09-27-v7f-refav1-state/v7f_merge/`) -- each is a DEFECT the gate exists to catch; when one is
fixed its test must be updated IN THE SAME CHANGE:
  * G-HYG FAILS: EncoderConfig / PredictorConfig / ReadoutConfig / V6Config are not strict;
  * (F3 + F3b, FIXED 2026-09-27 and flipped below) G-CKPT: a resume replayed the sampler/RNG stream
    from --seed, and the cosine LR replay compounded on the loaded lr -- the rehearsals now PASS G-CKPT on
    the interrupted-run protocol (arms `v6_resume_drops_rng`, `v6_resume_lr_compounds`,
    `v6_ckpt_strips_rng`, `v6_resume_drops_opt` + a source mutation of the restore call).
FIXED in the merge (2026-09-27) and flipped here in the same change, each with an arm that must FAIL:
  * `--nav-cond` was declared but NOT BUILT (build_stack_from_args never mapped it) -- G-DVB now PASSES
    on the compliant S-T and S-W rehearsals; arm `v6_nav_not_mapped` + a source mutation;
  * F7: R2's operative nav channel trained nothing (no S-W loss read the nav-conditioned operative
    prediction) -- the S-W G-LIVE now PASSES with every operative nav leaf reached; arm
    `v6_nav_unbound_wm` + a source mutation of the three WM-loss call sites.
The rehearsal feeds nav through the trainer's own path (assert_cache_join -> NavEmitter -> batch) from a
synthetic nav label source (`launch_gate.v6_corpus_seam`); the S-W predecessor carries `--nav-cond`.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]                    # stack/
TREE = ROOT.parent                                            # the tree holding stack/ + taniteval/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import launch_gate as LG  # noqa: E402

torch = pytest.importorskip("torch")

#: a launch-LIKE S-T argv (Thor paths: the rehearsal must drop / replace them), the first compliant
#: configuration of the brief on the v7F merge: one goal-conditioned 6 s trajectory, no selector,
#: nav declared (R2), max speed input + cap (R1), all tactical labels (R3), tactical -> operative (R4,
#: mode = PI decision D1), strategic OFF (R6).
ST_ARGV = ["--stage", "S-T", "--out", "/home/nvidia/experiments/v7f-st", "--steps", "10000",
           "--batch", "8", "--lr", "0.0001",
           "--init-from", "/home/nvidia/experiments/v7f-sw/ckpt.pt",
           "--prev-gate", "/home/nvidia/experiments/v7f-sw/stage_gate.json",
           "--v2-cache", "/home/nvidia/data/physicalai-train-e438721ae894-w120-256x640cyl",
           "--require-parity", "--nav-labels", "/home/nvidia/data/v72/nav.jsonl.gz", "--nav-cond",
           "--newest-frame-only", "--in-channels", "3", "--enc-dim", "768", "--enc-depth", "12",
           "--enc-heads", "12", "--frame-h", "256", "--frame-w", "640", "--horizons", "1",
           "--n-candidates", "1", "--proposals", "query", "--selector", "none",
           "--strategic-off", "--max-speed-input-v6", "--plan-vmax-cap",
           "--speed-max-sidecar-v6", "/home/nvidia/data/refcv6_speed_max_v8_train.jsonl",
           "--w-tac-label-all", "1.0", "--goal-multilabel",
           "--s2-labels", "/home/nvidia/data/v72/labels/s2_labels_v7.2_train.jsonl.gz",
           "--tac-op-cond", "detached",
           "--save-every", "250", "--log-every", "2"]
SW_ARGV = list(ST_ARGV)
for _f in ("--init-from", "--prev-gate", "--n-candidates", "--proposals", "--selector",
           "--max-speed-input-v6", "--plan-vmax-cap", "--speed-max-sidecar-v6", "--w-tac-label-all",
           "--s2-labels", "--tac-op-cond"):
    SW_ARGV = LG.set_flag(SW_ARGV, _f, None)
SW_ARGV = LG.set_flag(SW_ARGV, "--stage", ["S-W"])


def _ctx(tmp: Path, argv: list[str], arm: str | None = None) -> LG.Ctx:
    return LG.Ctx(profile="v7f", tree=str(TREE), commit="7f" * 20, argv=list(argv),
                  out_dir=str(tmp), path_map=[], tree_sha256="0" * 64,
                  argv_sha256=LG.argv_sha256(argv),
                  options={"rehearsal": True, "cpu_only": True, "omp": 4, "smoke_steps": 4},
                  arm=arm)


def _reasons(ev: dict) -> str:
    return " || ".join(ev.get("reasons") or [])


@pytest.fixture(scope="module")
def st_positive(tmp_path_factory):
    """ONE positive S-T rehearsal (smoke first, as `run` orders the jobs), shared by the tests."""
    d = tmp_path_factory.mktemp("v7f_st_pos")
    ctx = _ctx(d, ST_ARGV)
    evs = LG.job_smoke_v6(ctx, ["G-LIVE", "G-CKPT"])
    evs.update(LG.job_model_v6(ctx, ["G-HYG", "G-DVB", "G-EVAL"]))
    return ctx, evs


# ------------------------------------------------------------------------------------------ #
# the profile and its argv rules (pure)                                                        #
# ------------------------------------------------------------------------------------------ #
def test_v7f_profile_is_registered_as_the_v6_family_and_refcv7_is_untouched():
    p = LG.PROFILES["v7f"]
    assert (p["family"], p["trainer"]) == ("v6", "stack/scripts/train_v6_staged.py")
    assert p["required_checks"] == LG.BASE_CHECKS
    assert p["stages"]["devbox"] == ("G-HYG", "G-DVB", "G-EVAL", "G-LIVE", "G-CKPT")
    assert p["stages"]["thor"] == LG.BASE_CHECKS and p["rehearsal_stages"] == ("devbox",)
    r7 = LG.PROFILES["refcv7"]
    assert "family" not in r7 and r7["trainer"] == "stack/scripts/refc_v3_train.py"
    assert LG.JOB_FUNCS["smoke"] is LG.job_smoke and LG.JOB_FUNCS_V6["smoke"] is LG.job_smoke_v6
    assert LG.JOB_FUNCS_V6["suite"] is LG.job_suite                 # G-SUITE is shared


@pytest.mark.parametrize("change,needle", [
    (("--stage", ["S-S"]), "--stage S-S is REFUSED"),
    (("--stage", ["S-J"]), "--stage S-J is REFUSED"),
    (("--nav-cond", None), "--nav-cond is REQUIRED ON"),
    (("--i-know-this-arm-predates-nav", []), "--i-know-this-arm-predates-nav is passed"),
    (("--i-know-this-is-the-control-arm", []), "--i-know-this-is-the-control-arm is passed"),
    (("--selector", ["goal"]), "--selector goal is REFUSED"),
    (("--selector", ["mlp"]), "--selector mlp is REFUSED"),
    (("--w-s2-goal", ["1.0"]), "--w-s2-goal"),
    (("--w-s1-multi", ["0.5"]), "--w-s1-multi"),
    # the v7F merge's levers
    (("--strategic-off", None), "--strategic-off is REQUIRED ON"),
    (("--tac-goal-cond", []), "--tac-goal-cond is passed"),
    (("--goal-factored", []), "--goal-factored is passed"),
    (("--tac-op-cond", ["off"]), "--tac-op-cond off is REFUSED"),
    (("--tac-op-cond", None), "--tac-op-cond is REQUIRED ON"),
    (("--max-speed-input-v6", None), "--max-speed-input-v6 is REQUIRED ON"),
    (("--plan-vmax-cap", None), "--plan-vmax-cap is REQUIRED ON"),
    (("--goal-multilabel", None), "--goal-multilabel is REQUIRED ON"),
    (("--w-tac-label-all", ["0"]), "--w-tac-label-all must be passed with a value > 0"),
])
def test_v7f_argv_rules_R1_to_R6_and_SEL1_each_refused_by_name(change, needle):
    p = LG.PROFILES["v7f"]
    refusals, pending = LG.profile_argv_rules(p, ST_ARGV)       # the compliant argv
    assert refusals == []
    assert len(pending) == 1 and pending[0].startswith("--tac-op-cond detached")   # PI decision D1
    bad = LG.set_flag(ST_ARGV, change[0], change[1])
    refusals, _pi = LG.profile_argv_rules(p, bad)
    assert any(needle in r for r in refusals), refusals


def test_v7f_the_R1_R3_R4_rows_bind_S_T_only():
    """The trainer REFUSES R1/R4 in S-W and zeroes R3 there: the profile must not require them."""
    p = LG.PROFILES["v7f"]
    assert LG.profile_argv_rules(p, SW_ARGV) == ([], [])


def _evidence(ctx, check, status="PASS", **over):
    ev = {"schema": LG.GATE_SCHEMA, "kind": "evidence", "check": check, "status": status,
          "reasons": [], "binding": ctx.binding(), "inputs_read": {}, "host": {"node": "t"},
          "arm": None, "started_utc": "2026-09-27T00:00:00Z",
          "finished_utc": "2026-09-27T00:00:01Z", "details": {}}
    ev.update(over)
    return ev


def test_finalize_never_counts_REHEARSAL_evidence_and_says_so_by_name(tmp_path):
    ctx = _ctx(tmp_path / "g", ST_ARGV)
    key = LG.load_key(tmp_path / "k.key")
    for c in LG.BASE_CHECKS:
        LG.write_evidence(ctx.out_dir, _evidence(ctx, c, rehearsal={"note": "tiny"}))
    verdict, _p, tok = LG.finalize(ctx, key)
    assert verdict == "INCOMPLETE"
    assert all(v["status"] == "MISSING" for v in tok["checks"].values())
    why = json.dumps(tok["checks"])
    assert "REHEARSAL evidence" in why
    assert set(tok["rehearsal"]["checks"]) == set(LG.BASE_CHECKS)   # what each instrument said
    # the dev box is not the launch host: its missing Thor paths are notes, not a FAIL
    assert any("is missing on this host" in n for n in tok["rehearsal"]["notes"])


def test_a_standalone_finalize_rederives_the_v7f_rules(tmp_path):
    bad = LG.set_flag(ST_ARGV, "--stage", ["S-S"])
    ctx = _ctx(tmp_path / "g", bad)
    ctx.options = {}                                           # a Thor finalize, no rehearsal
    key = LG.load_key(tmp_path / "k.key")
    for c in LG.BASE_CHECKS:
        LG.write_evidence(ctx.out_dir, _evidence(ctx, c))
    verdict, _p, tok = LG.finalize(ctx, key)
    assert verdict == "FAIL"
    assert any("--stage S-S is REFUSED" in r for r in tok["reasons"])


def _all_pass_finalize(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path / "g", ST_ARGV)
    ctx.options = {}
    key = LG.load_key(tmp_path / "k.key")
    monkeypatch.setattr(LG, "data_manifest_problems", lambda data: [])
    for c in LG.BASE_CHECKS:
        LG.write_evidence(ctx.out_dir, _evidence(ctx, c))
    return LG.finalize(ctx, key)


def test_the_v7f_has_NO_open_items_and_all_green_is_a_PI_DECISION_on_D1(tmp_path, monkeypatch):
    """R2-NAV-NOT-BUILT was the last open item; the merge closed it (nav mapped). What remains between
    an all-green evidence set and a PASS is the value of `--tac-op-cond`: PI decision D1."""
    assert LG.PROFILES["v7f"]["open_items"] == ()
    verdict, _p, tok = _all_pass_finalize(tmp_path, monkeypatch)
    assert verdict == "PI-DECISION"                 # never PASS while D1 is open
    assert not any("OPEN ITEM" in r for r in tok["reasons"])
    assert any("PI decision pending: --tac-op-cond" in r for r in tok["reasons"])


def test_an_OPEN_ITEM_still_blocks_a_v7f_PASS(tmp_path, monkeypatch):
    """The mechanism the closed item used stays live: re-open one and the token drops to INCOMPLETE."""
    item = {"id": "GATE-TEST-REOPENED", "what": "a re-opened launch-definition item",
            "owner": "test"}
    monkeypatch.setitem(LG.PROFILES["v7f"], "open_items", (item,))
    verdict, _p, tok = _all_pass_finalize(tmp_path, monkeypatch)
    assert verdict == "INCOMPLETE"                  # never PASS, never even PI-DECISION with an open item
    assert any("OPEN ITEM GATE-TEST-REOPENED" in r for r in tok["reasons"])


# ------------------------------------------------------------------------------------------ #
# the v6 G-DVB registry (pure)                                                                 #
# ------------------------------------------------------------------------------------------ #
def _trainer_and_parser(arm=None):
    ctx = _ctx(Path("."), ST_ARGV)
    T = LG.load_trainer_v6(ctx)
    return T, LG.v6_parser(T, arm=arm)


def test_dvb_v6_registry_covers_EVERY_trainer_flag():
    from tanitad.train import declared_vs_built_v6 as D6
    _T, ap = _trainer_and_parser()
    dests = sorted({a.dest for a in ap._actions if a.dest != "help"})
    # the tip's 231 dests (232 options: --(no-)require-parity share one) + the merge's 8
    assert len(dests) == 239
    assert D6.uncovered_v6(ap) == []
    census = D6.kinds_census(ap)
    assert sum(census.values()) == 239 and "UNCOVERED" not in census
    # the merge's own 7 run in the TRAINER; everything else is gate-time
    assert len(D6.GATE_TIME) == 232 and not set(D6.ALL_V6_DESTS) & D6.GATE_TIME


def test_dvb_v6_a_NEW_flag_without_an_entry_is_refused():
    from tanitad.train import declared_vs_built_v6 as D6
    _T, ap = _trainer_and_parser(arm="v6_uncovered_flag")
    assert D6.uncovered_v6(ap) == ["zz_gate_arm_unregistered_lever"]


def _tiny_stack(argv):
    from tanitad.train import declared_vs_built_v6 as D6
    T, ap = _trainer_and_parser()
    tiny, _dep = LG.v6_rehearsal_argv(LG.PROFILES["v7f"], argv)
    a = ap.parse_args(LG.set_flag(tiny, "--out", ["x"]))
    torch.manual_seed(0)
    return D6, T, ap, a, T.build_stack_from_args(a)


def _nav_mismatches(D6, st, a, ap, **kw):
    return [str(m) for m in D6.check_all(st, a, parser=ap, **kw) if m.lever == "--nav-cond"]


def test_dvb_v6_the_nav_reader_reads_the_BUILT_stack_both_ways():
    D6, _T, ap, a, st = _tiny_stack(ST_ARGV)
    # FIXED in the merge: the trainer maps --nav-cond, so the channel is declared AND built
    assert a.nav_cond is True and st.cfg.nav_cond is True and st.nav is not None
    assert _nav_mismatches(D6, st, a, ap) == []
    # the tip's (and the first merge's) unmapped build: the cfg field and the module both read BUILT False
    st.cfg.nav_cond, st.nav = False, None
    nav = _nav_mismatches(D6, st, a, ap)
    assert len(nav) == 2 and all("declared True but BUILT False" in m for m in nav)


def test_dvb_v6_the_TRAINER_side_checks_are_unchanged_by_the_gate_entries():
    """The gate-time entries never run inside the trainer: its build-time `check` and its R3-path
    `check_v6` see exactly the merge's seven levers -- so the merged trainer stays bit-identical. A
    defect only a gate-time entry reads (nav unmapped) is invisible to both and caught by `check_all`."""
    D6, T, ap, a, st = _tiny_stack(ST_ARGV)
    w = T._weights_from_args(a).for_stage("S-T")
    assert D6.check(st, a) == [] and D6.check_v6(st, a, weights_in_force=w, parser=ap) == []
    assert _nav_mismatches(D6, st, a, ap, weights_in_force=w) == []
    st.cfg.nav_cond, st.nav = False, None
    assert D6.check(st, a) == []                                 # nav is NOT read at build time
    assert D6.check_v6(st, a, weights_in_force=w, parser=ap) == []
    assert len(_nav_mismatches(D6, st, a, ap, weights_in_force=w)) == 2


# ------------------------------------------------------------------------------------------ #
# the positive S-T rehearsal: GREEN where the tip is right, a NAMED defect where it is not     #
# ------------------------------------------------------------------------------------------ #
def test_positive_G_LIVE_passes_with_R6_bit_identity_on_every_layer_str_tensor(st_positive):
    _ctx_, evs = st_positive
    ev = evs["G-LIVE"]
    assert ev["status"] == "PASS", _reasons(ev)
    d = ev["details"]
    assert [t["key"] for t in d["terms"]] == ["plan", "seam", "t1"]
    assert all(t["present_steps"] == d["expect_steps"] == 4 for t in d["terms"])
    assert d["r6"]["n_layer_str_tensors"] > 0 and d["r6"]["moved_in_memory"] == []
    assert d["r6"]["differs_in_saved_ckpt"] == []
    assert d["r6"]["grad_populated_tensors"] == 0            # REPORTED (requires_grad False in S-T)
    assert d["strategic_terms_seen"] == [] and d["loss_knobs_disagree"] == []
    assert all(e["groups"] for e in d["term_reach"].values())
    assert ev["rehearsal"] and ev["inputs_read"]["gate:rehearsal"]["kind"] == "rehearsal"


def test_positive_G_EVAL_passes_and_its_controls_are_live(st_positive):
    _ctx_, evs = st_positive
    ev = evs["G-EVAL"]
    assert ev["status"] == "PASS", _reasons(ev)
    d = ev["details"]
    assert d["state_dict"]["n_differ"] == 0 and d["forward"]["n_differ"] == 0
    assert d["determinism_control_outputs_differing"] == 0
    assert d["negative_control_outputs_moved"] > 0


def test_positive_G_CKPT_PASSES_the_resume_IS_the_uninterrupted_run(st_positive):
    """Was F3 (MEASURED 2026-09-27: the resumed S-T run drew the uninterrupted run's FIRST batches) and F3b
    (the cosine replay compounding on the loaded lr, hidden by an lr-excluding comparison). Flipped in the
    same change as the trainer fix: the interrupted-then-resumed S-T rehearsal draws the uninterrupted
    run's batches at the same steps and ends BIT-identical in model AND optimiser state."""
    _ctx_, evs = st_positive
    ev = evs["G-CKPT"]
    assert ev["status"] == "PASS", _reasons(ev)
    d = ev["details"]
    assert d["ckpt_rng_or_data_position_keys"] == ["data_pos", "rng_state"]
    assert len(d["draws_resumed"]) == 2                                       # resume_extra_steps
    assert d["draws_resumed"] == d["draws_uninterrupted_same_steps"]
    assert d["draws_resumed"] != d["draws_uninterrupted_first_steps"]         # NOT a replay
    for k in ("save_model_vs_memory", "save_opt_state_vs_memory", "resume_model_vs_saved",
              "resume_opt_state_vs_saved", "resumed_final_vs_uninterrupted_final",
              "resumed_final_opt_vs_uninterrupted_final"):
        assert d[k]["n"] > 0 and d[k]["n_differ"] == 0, k
    assert d["resume_lr"]["saved"] == d["resume_lr"]["after_replay"]
    assert d["resume_config_launch_mode"] == "resume" and d["resume_config_resume_rng"] == "restored"
    assert d["protocol"]["interrupt_after_checkpoint_at"] == 4 and d["protocol"]["resume_to"] == 6


def test_positive_S_W_G_CKPT_PASSES(tmp_path):
    ev = LG.job_smoke_v6(_ctx(tmp_path, SW_ARGV), ["G-CKPT"])["G-CKPT"]
    assert ev["status"] == "PASS", _reasons(ev)
    d = ev["details"]
    assert d["draws_resumed"] == d["draws_uninterrupted_same_steps"]
    assert d["draws_resumed"] != d["draws_uninterrupted_first_steps"]
    assert d["resumed_final_vs_uninterrupted_final"]["n_differ"] == 0
    assert d["resumed_final_opt_vs_uninterrupted_final"]["n_differ"] == 0


def test_positive_G_DVB_PASSES_now_that_the_merge_maps_nav(st_positive):
    """Was a FAIL naming `--nav-cond: declared True but BUILT False` (tip + first merge); the merge
    maps nav, so the compliant S-T rehearsal is GREEN -- and the behaviour probe proves the BUILT
    channel is consumed: a forward without the nav token now RAISES NavTokenMissing."""
    _ctx_, evs = st_positive
    ev = evs["G-DVB"]
    r = _reasons(ev)
    assert ev["status"] == "PASS", r
    assert "--nav-cond" not in r
    d = ev["details"]
    assert d["mismatches"] == [] and d["coverage_uncovered"] == []
    assert d["trainability_probe"]["trainable_groups_built"] == ["layer_tac", "planner"]
    pc = d["positive_control"]                                  # the registry SEES a built change
    assert pc["control_width"] == 2 and len(pc["control_seen"]) == 1
    bp = d["behaviour_probe"]
    assert bp["forward"] == "ran" and bp["plan_waypoints_shape"] == [1, 1, 60, 2]
    assert {"nav_token", "nav_args"} <= set(bp["synthetic_batch_keys"])
    assert bp["forward_without_nav_token"].startswith("NavTokenMissing")


def test_positive_the_S_T_rehearsal_predecessor_carries_nav(st_positive):
    """S-T may not INTRODUCE `nav.*` (STAGE_MAY_INTRODUCE), so the tiny S-W predecessor must be trained
    WITH `--nav-cond` and a nav label source -- the synthetic one the rehearsal seam serves."""
    _ctx_, evs = st_positive
    prec = evs["G-DVB"]["details"]["argv_local"]["rehearsal_predecessor"]
    sw = prec["sw_argv"]
    assert LG.flag_values(sw, "--stage") == ["S-W"] and "--nav-cond" in sw
    assert LG.flag_values(sw, "--nav-labels") == [LG.V7F_REHEARSAL_NAV_LABELS]


def test_positive_G_HYG_names_the_open_config_classes_and_nothing_else(st_positive):
    _ctx_, evs = st_positive
    ev = evs["G-HYG"]
    assert ev["status"] == "FAIL"
    assert "not_strict" in ev["details"], _reasons(ev)          # a refused build has no census
    assert sorted(ev["details"]["not_strict"]) == [
        "tanitad.config.EncoderConfig", "tanitad.config.PredictorConfig",
        "tanitad.config.ReadoutConfig", "tanitad.models.v6.V6Config"]
    assert ev["details"]["undeclared_attributes"] == []
    assert "UNDECLARED" not in _reasons(ev)


# ------------------------------------------------------------------------------------------ #
# every regression arm goes RED through the gate                                                #
# ------------------------------------------------------------------------------------------ #
def test_ARM_v6_strategic_leaf_trainable_FAILS_G_LIVE_on_R6_bit_identity(tmp_path):
    """The coordinator's arm on the S-T wiring WITHOUT --strategic-off (the tip's): the plan loss
    reaches `layer_str` through goal_head_tac(cond=e_g_str) (v6.py:5938), so the trainable leaf MOVES
    and R6's decisive clause (a) fails."""
    ctx = _ctx(tmp_path, LG.set_flag(ST_ARGV, "--strategic-off", None),
               arm="v6_strategic_leaf_trainable")
    ev = LG.job_smoke_v6(ctx, ["G-LIVE"])["G-LIVE"]
    r = _reasons(ev)
    assert ev["status"] == "FAIL" and "R6:" in r and "layer_str parameter(s) CHANGED" in r
    assert ev["details"]["arm"] == {"v6_strategic_leaf_trainable": "vocab_str.table.weight"}
    assert "vocab_str.table.weight" in ev["details"]["r6"]["moved_in_memory"]
    assert ev["details"]["r6"]["grad_populated_tensors"] >= 1  # reported, and now non-zero


def test_ARM_v6_strategic_leaf_trainable_under_strategic_off_FAILS_as_a_dead_trainable_leaf(tmp_path):
    """With --strategic-off (MEASURED on the merge) NO gradient reaches `layer_str`, so the arm's leaf
    cannot move -- and the gate still FAILS it: a declared-trainable leaf that receives nothing."""
    ctx = _ctx(tmp_path, ST_ARGV, arm="v6_strategic_leaf_trainable")
    ev = LG.job_smoke_v6(ctx, ["G-LIVE"])["G-LIVE"]
    r = _reasons(ev)
    assert ev["status"] == "FAIL" and "vocab_str.table" in r and "ZERO gradient" in r
    assert ev["details"]["r6"]["moved_in_memory"] == []         # R6 (a) holds: nothing moved


def test_ARM_v6_drop_term_FAILS_G_LIVE_naming_the_term(tmp_path):
    ctx = _ctx(tmp_path, ST_ARGV, arm="v6_drop_term")
    ev = LG.job_smoke_v6(ctx, ["G-LIVE"])["G-LIVE"]
    assert ev["status"] == "FAIL" and "declared term `t1`" in _reasons(ev)


def test_ARM_v6_resume_drops_opt_FAILS_G_CKPT_on_the_optimizer(tmp_path):
    ctx = _ctx(tmp_path, ST_ARGV, arm="v6_resume_drops_opt")
    ev = LG.job_smoke_v6(ctx, ["G-CKPT"])["G-CKPT"]
    assert ev["status"] == "FAIL" and "resume_opt_state_vs_saved" in _reasons(ev)


def test_ARM_v6_resume_drops_rng_REPLAYS_the_first_batches_and_FAILS_G_CKPT(tmp_path):
    """F3 re-introduced through the trainer's own seam: the resume restores no stream."""
    ev = LG.job_smoke_v6(_ctx(tmp_path, ST_ARGV, arm="v6_resume_drops_rng"), ["G-CKPT"])["G-CKPT"]
    r = _reasons(ev)
    assert ev["status"] == "FAIL" and "restarts at every launch" in r, r
    d = ev["details"]
    assert d["draws_resumed"] == d["draws_uninterrupted_first_steps"]         # the MEASURED defect
    assert "resumed_final_vs_uninterrupted_final" in r


def test_ARM_v6_resume_lr_compounds_FAILS_G_CKPT_on_the_lr(tmp_path):
    """F3b re-introduced: the draws stay exact, but the lr after the replay is lr_n x (lr_n / lr_0) --
    the whole remainder of the run on a scaled schedule."""
    ev = LG.job_smoke_v6(_ctx(tmp_path, SW_ARGV, arm="v6_resume_lr_compounds"), ["G-CKPT"])["G-CKPT"]
    r = _reasons(ev)
    assert ev["status"] == "FAIL" and "F3b" in r, r
    d = ev["details"]
    assert d["draws_resumed"] == d["draws_uninterrupted_same_steps"]          # F3 itself holds
    s, a = d["resume_lr"]["saved"][0], d["resume_lr"]["after_replay"][0]
    assert s > 0 and abs(a / s - s / 1e-4) < 1e-9                             # --lr 0.0001 in the argv


def test_ARM_v6_ckpt_strips_rng_a_PRE_F3_checkpoint_FAILS_G_CKPT_and_the_trainer_records_it(tmp_path):
    """A checkpoint in the pre-F3 layout {stack, opt, step, config}: the trainer resumes it LOUDLY
    (`resume_rng.mode = "absent-replayed"`), and G-CKPT FAILS on the missing keys, the replay AND the record."""
    ev = LG.job_smoke_v6(_ctx(tmp_path, SW_ARGV, arm="v6_ckpt_strips_rng"), ["G-CKPT"])["G-CKPT"]
    r = _reasons(ev)
    assert ev["status"] == "FAIL"
    assert "carries no RNG state" in r and "restarts at every launch" in r and "'absent-replayed'" in r, r
    assert ev["details"]["ckpt_rng_or_data_position_keys"] == []


#: the F3 restore call in the trainer (its first line) -- the source mutation below deletes the call
F3_RESTORE_CALL = "_rl = apply_resume_state(resume_state, device=device, rng=rng, gen=gen,"


def test_ARM_SOURCE_deleting_the_resume_restore_call_FAILS_G_CKPT(tmp_path, monkeypatch):
    """Mutation, not inspection: the F3 restore call replaced by a no-op in the trainer SOURCE (compiled
    under its own path) -- G-CKPT must FAIL on the replayed stream."""
    import types
    src_path = TREE / LG.PROFILES["v7f"]["trainer"]
    src = src_path.read_text(encoding="utf-8")
    assert src.count(F3_RESTORE_CALL) == 1                     # the fix is present, exactly once
    src = src.replace(F3_RESTORE_CALL, "_rl = (lambda *_a, **_k: {'t3_alpha_applied': None, "
                      "'t3_prog_applied': None, 'monitors': {}, 'applied': []})(resume_state, "
                      "device=device, rng=rng, gen=gen,")
    mod = types.ModuleType(LG._TRAINER_MOD_V6)
    mod.__file__ = str(src_path)
    monkeypatch.setitem(sys.modules, LG._TRAINER_MOD_V6, mod)  # the gate's loader returns it
    exec(compile(src, str(src_path), "exec"), mod.__dict__)
    ev = LG.job_smoke_v6(_ctx(tmp_path, SW_ARGV), ["G-CKPT"])["G-CKPT"]
    r = _reasons(ev)
    assert ev["status"] == "FAIL" and "restarts at every launch" in r, r
    assert ev["details"]["draws_resumed"] == ev["details"]["draws_uninterrupted_first_steps"]


def test_ARM_v6_undeclared_attribute_FAILS_G_HYG_by_name(tmp_path):
    ctx = _ctx(tmp_path, ST_ARGV, arm="v6_undeclared_attribute")
    ev = LG.job_model_v6(ctx, ["G-HYG"])["G-HYG"]
    assert ev["status"] == "FAIL"
    assert any("gate_arm_undeclared_lever" in ".".join(x) for x in
               ev["details"]["undeclared_attributes"])
    assert "UNDECLARED attribute" in _reasons(ev)


@pytest.mark.parametrize("arm,needles", [
    # the registry reads the BUILT width; the gate's own probe sees the forward break (MEASURED:
    # the goal embedding is expanded to the declared 1 while the fan holds 8 -> a size mismatch)
    ("v6_candidates_not_built", ("--n-candidates: declared 1 but BUILT 8",
                                 "synthetic_batch RAISED")),
    ("missing_dvb_v6_module", ("declared_vs_built_v6 is not importable",)),
    ("v6_uncovered_flag", ("--zz-gate-arm-unregistered-lever",)),
])
def test_ARM_G_DVB_each_FAILS_naming_its_defect(tmp_path, arm, needles):
    ctx = _ctx(tmp_path, ST_ARGV, arm=arm)
    ev = LG.job_model_v6(ctx, ["G-DVB"])["G-DVB"]
    assert ev["status"] == "FAIL"
    assert all(n in _reasons(ev) for n in needles), _reasons(ev)[:800]


def test_ARM_v6_nav_not_mapped_FAILS_G_DVB_on_S_W_naming_both_reads_and_the_behaviour(tmp_path):
    """The merge's nav fix REMOVED: the arm hands V6Stack a config with `nav_cond` False although argv
    declares `--nav-cond` (the tip's / first merge's build). G-DVB must name both reads (the cfg field
    and the missing NavConditioner) and its behaviour probe (a nav-less forward RUNS)."""
    ctx = _ctx(tmp_path, SW_ARGV, arm="v6_nav_not_mapped")
    ev = LG.job_model_v6(ctx, ["G-DVB"])["G-DVB"]
    r = _reasons(ev)
    assert ev["status"] == "FAIL"
    assert r.count("--nav-cond: declared True but BUILT False") == 2, r[:800]
    assert "forward WITHOUT a nav token RAN" in r
    assert ev["details"]["behaviour_probe"]["forward_without_nav_token"] == "ran"


#: the merge's R2 fix, verbatim: ONE line in build_stack_from_args (train_v6_staged.py)
NAV_MAPPING_LINE = '        nav_cond=bool(getattr(a, "nav_cond", False)),\n'


def test_ARM_SOURCE_deleting_the_merges_nav_mapping_line_FAILS_G_DVB(tmp_path, monkeypatch):
    """Mutation, not inspection: delete the fix's one line from the trainer SOURCE (compiled under the
    trainer's own path, so every path it derives is unchanged) and run the gate on it -- G-DVB must FAIL
    naming both nav reads."""
    import types
    src_path = TREE / LG.PROFILES["v7f"]["trainer"]
    src = src_path.read_text(encoding="utf-8")
    assert src.count(NAV_MAPPING_LINE) == 1                     # the fix is present, exactly once
    mod = types.ModuleType(LG._TRAINER_MOD_V6)
    mod.__file__ = str(src_path)
    monkeypatch.setitem(sys.modules, LG._TRAINER_MOD_V6, mod)  # the gate's loader returns it
    exec(compile(src.replace(NAV_MAPPING_LINE, ""), str(src_path), "exec"), mod.__dict__)
    ev = LG.job_model_v6(_ctx(tmp_path, SW_ARGV), ["G-DVB"])["G-DVB"]
    r = _reasons(ev)
    assert ev["status"] == "FAIL"
    assert r.count("--nav-cond: declared True but BUILT False") == 2, r[:800]
    assert "forward WITHOUT a nav token RAN" in r


def test_ARM_v6_nav_not_mapped_on_S_T_is_REFUSED_at_the_ladder_seam(tmp_path):
    """On S-T the same arm cannot even build: the (nav-carrying) predecessor has `nav.*` keys the
    unmapped stack lacks, and the trainer's own ladder check refuses it -- a FAIL, never a PASS."""
    ctx = _ctx(tmp_path, ST_ARGV, arm="v6_nav_not_mapped")
    ev = LG.job_model_v6(ctx, ["G-DVB"])["G-DVB"]
    r = _reasons(ev)
    assert ev["status"] == "FAIL"
    assert "is not a valid predecessor for stage 'S-T'" in r and "nav.embed.weight" in r


def test_ARM_v6_loader_drops_vocab_version_FAILS_G_EVAL(st_positive):
    ctx0, _evs = st_positive
    ctx = _ctx(Path(ctx0.out_dir), ST_ARGV, arm="v6_loader_drops_vocab_version")
    ev = LG.job_model_v6(ctx, ["G-EVAL"])["G-EVAL"]
    assert ev["status"] == "FAIL" and "eval loader FAILED to rebuild" in _reasons(ev)


def test_G_CLOCK_v6_is_ERROR_never_a_pass_until_it_is_built(tmp_path):
    ev = LG.job_clock_v6(_ctx(tmp_path, ST_ARGV), ["G-CLOCK"])["G-CLOCK"]
    assert ev["status"] == "ERROR" and "not implemented for the v6 family" in _reasons(ev)


def test_TIP_FINDING_uplink_ema_moves_the_strategic_EMA_adapter_and_R6_FAILS(tmp_path):
    """MEASURED 2026-09-27: under `--uplink ema`, `stack.ema_update()` (v6.py:5625) advances the
    STRATEGIC EMA adapter every S-T step even though `layer_str` is frozen; an EMA of a frozen source
    still drifts by float rounding (`mul_(d).add_(x, alpha=1-d)`, v6.py:4950), so the literal R6 rule
    FAILS and names the mechanism. A decision for the PI (R6-LITERALLY-OFF), not a gate bug."""
    ctx = _ctx(tmp_path, LG.set_flag(ST_ARGV, "--uplink", ["ema"]))
    ev = LG.job_smoke_v6(ctx, ["G-LIVE"])["G-LIVE"]
    r = _reasons(ev)
    assert ev["status"] == "FAIL" and "mechanism ['EMA']" in r
    assert ev["details"]["r6"]["moved_in_memory"] and all(
        n.startswith("ema_adapter_str.") for n in ev["details"]["r6"]["moved_in_memory"])
    assert ev["details"]["r6"]["grad_populated_tensors"] == 0    # no gradient: the EMA step did it


def test_TIP_FINDING_o14_carried_into_S_T_is_advertised_TRAINS_and_trains_nothing(tmp_path):
    """MEASURED 2026-09-27: `V6LossWeights.for_stage("S-T")` does not zero o11/o13/o14
    (train_v6_staged.py:373-379), so `--w-o14 1.0` in S-T is audited `TRAINS` by the trainer's own
    effective-weight table while the term has NO autograd graph (every module it touches is frozen).
    The gate's per-term reach catches the declared-but-inert term."""
    ctx = _ctx(tmp_path, LG.set_flag(ST_ARGV, "--w-o14", ["1.0"]))
    ev = LG.job_smoke_v6(ctx, ["G-LIVE"])["G-LIVE"]
    assert ev["status"] == "FAIL"
    assert "term `o14` is in force but its gradient reaches NO trainable parameter" in _reasons(ev)
    decl = {x["term"]: x["status"] for x in ev["details"]["declared_terms"]}
    assert decl["o14_fut"] == "TRAINS"                           # the trainer's own claim


#: R2's nav parameters in group `predictor_op` (train in S-W only; frozen from S-T on)
NAV_OPERATIVE_LEAVES = ["nav.arg_proj", "nav.embed", "nav.gate", "nav.layer_proj.operative"]
NAV_OPERATIVE_TENSORS = ["nav.arg_proj.bias", "nav.arg_proj.weight", "nav.embed.weight",
                         "nav.gate.operative", "nav.layer_proj.operative.bias",
                         "nav.layer_proj.operative.weight"]


def test_positive_S_W_G_LIVE_PASSES_and_the_operative_nav_leaves_TRAIN(tmp_path):
    """Was F7 (MEASURED 2026-09-27, flipped in the same change as the merge's fix): no S-W loss read the
    nav-conditioned operative prediction, so the six `predictor_op` nav tensors got NO gradient in S-W
    and S-T froze them. The merge's `_NavBoundPredictor` binds `stack.nav(..., "operative")` to the
    predictor the WM-loss helpers roll: the compliant S-W smoke is GREEN with NO acknowledgement, every
    operative nav leaf gets a gradient, and the trainer's own grad_reach.json agrees (nothing unreached)."""
    ctx = _ctx(tmp_path, SW_ARGV)
    assert "--allow-unreached" not in ctx.argv
    ev = LG.job_smoke_v6(ctx, ["G-LIVE"])["G-LIVE"]
    assert ev["status"] == "PASS", _reasons(ev)
    d = ev["details"]
    g = d["grads"]
    assert g["dead_leaves"] == [] and g["acknowledged_by_allow_unreached"] == []
    for leaf in NAV_OPERATIVE_LEAVES:                          # every operative nav leaf TRAINS
        live, n = g["leaf_live"][leaf]
        assert n > 0 and live == n, (leaf, live, n)
    ctl = d["control_trainer_grad_reach"]
    assert ctl["trainer_unreached"] == [] and ctl["gate_never_reached"] == []
    assert d["r6"]["n_layer_str_tensors"] > 0 and d["r6"]["moved_in_memory"] == []
    assert {t["key"] for t in d["terms"]} >= {"o5", "o6"}
    assert d["strategic_terms_seen"] == []
    # nav is BUILT and consumed by the forward (the G-DVB twin)
    dv = LG.job_model_v6(ctx, ["G-DVB"])["G-DVB"]
    assert dv["status"] == "PASS", _reasons(dv)
    assert dv["details"]["behaviour_probe"]["forward_without_nav_token"].startswith("NavTokenMissing")


def _assert_F7_named(ev: dict) -> None:
    r = _reasons(ev)
    assert ev["status"] == "FAIL" and len(ev["reasons"]) == 1, r
    assert "received ZERO gradient" in r and str(NAV_OPERATIVE_LEAVES) in r
    d = ev["details"]
    assert d["grads"]["dead_leaves"] == NAV_OPERATIVE_LEAVES
    for leaf in NAV_OPERATIVE_LEAVES:
        assert d["grads"]["leaf_live"][leaf][0] == 0, leaf
    ctl = d["control_trainer_grad_reach"]                       # the trainer's own census agrees
    assert ctl["trainer_unreached"] == ctl["gate_never_reached"] == NAV_OPERATIVE_TENSORS


def test_ARM_v6_nav_unbound_wm_reintroduces_F7_and_FAILS_G_LIVE_naming_the_operative_nav_leaves(tmp_path):
    """The gate's arm swaps `train_v6_staged._NavBoundPredictor` for the bare predictor (the pre-fix
    call sites): the operative nav leaves go dead again and G-LIVE names exactly them."""
    ev = LG.job_smoke_v6(_ctx(tmp_path, SW_ARGV, arm="v6_nav_unbound_wm"), ["G-LIVE"])["G-LIVE"]
    _assert_F7_named(ev)
    assert "v6_nav_unbound_wm" in (ev["details"].get("arm") or {})


#: the merge's F7 fix at its three WM-loss call sites (train_v6_staged.v6_loss_step)
F7_CALL_SITES = (("_pred_op, stack.step_readout_op", "stack.predictor_op, stack.step_readout_op", 1),
                 ("rollout_transitions(_pred_op,", "rollout_transitions(stack.predictor_op,", 2))


def test_ARM_SOURCE_unbinding_the_three_WM_loss_call_sites_FAILS_G_LIVE(tmp_path, monkeypatch):
    """Mutation, not inspection: the three call sites handed the BARE `stack.predictor_op` again in the
    trainer SOURCE (compiled under its own path) -- G-LIVE must FAIL naming the operative nav leaves."""
    import types
    src_path = TREE / LG.PROFILES["v7f"]["trainer"]
    src = src_path.read_text(encoding="utf-8")
    for old, new, n in F7_CALL_SITES:
        assert src.count(old) == n, old                        # the fix is present, exactly as expected
        src = src.replace(old, new)
    mod = types.ModuleType(LG._TRAINER_MOD_V6)
    mod.__file__ = str(src_path)
    monkeypatch.setitem(sys.modules, LG._TRAINER_MOD_V6, mod)  # the gate's loader returns it
    exec(compile(src, str(src_path), "exec"), mod.__dict__)
    _assert_F7_named(LG.job_smoke_v6(_ctx(tmp_path, SW_ARGV), ["G-LIVE"])["G-LIVE"])
