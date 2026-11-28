"""W8 (EvalFlyWheel, 2026-09-26): ``navsim_v2 --split navtest_single_stage`` — NAVSIM v2 EPDMS, ONE stage.

What is pinned, with LITERAL expectations (never an expression over the code under test):

* the split is wired as a ONE-STAGE profile (runner, logs dir, traffic policy, counts, protocol);
* the ONE-STAGE runner is the one LAUNCHED (the argv the scorer builds), with no synthetic override;
* MUTATION: a TWO-STAGE runner on this split goes RED — at the runner guard, AND at the CSV guard (a
  two-stage-shaped CSV fed to the single-stage count guard FAILS; the one-stage-shaped control PASSES);
* an unknown split still REFUSES; a subset / cache override on a two-stage split REFUSES;
* the dry run passes (offline: the dispatch; live: the real preflight, when the runtime is present).
"""
from __future__ import annotations

import dataclasses
import json
import subprocess
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TTS = Path("C:/Users/Admin/navsim-crun/devkit/navsim/planning/script/config/common/train_test_split")
PKG = REPO / "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-26-navtest-single-stage"
SMOKE_TOKENS = PKG / "raw" / "smoke_tokens_sub200.json"
SMOKE_CACHE = Path("C:/Users/Admin/navsim-crun/exp/metric_cache_navtest_v2_smoke200")

if not TTS.exists():                        # NO_TREE: not this dev box — skip, never fail silently
    pytest.skip(f"NO_TREE: the NavSim C: runtime split yamls are not on this machine ({TTS})", allow_module_level=True)

from taniteval.bench import cli                                       # noqa: E402
from taniteval.bench.navsim import profiles as P                     # noqa: E402
from taniteval.bench.navsim import scoring as SC                     # noqa: E402

NAVTEST = "navtest_single_stage"


# --------------------------------------------------------------------------- #
# 1. the profile                                                               #
# --------------------------------------------------------------------------- #
def test_navtest_is_wired_as_a_one_stage_profile():
    p = P.SPLITS[NAVTEST]
    assert p.protocol == "EPDMS_v2_navtest_single_stage"
    assert (p.stages, p.n_stage1, p.n_stage2, p.n_logs) == (1, 12146, 0, 136)
    assert p.tts == "navtest"
    assert p.runner == "pdm_score_one_stage"
    assert p.traffic_agents == "non_reactive"
    assert p.logs_dir == Path("D:/Archive/devbox-C/navsim/data/openscene/navsim_logs/test")
    assert p.syn_scenes is None and p.syn_sensors is None
    assert p.cache_verified_by == "CACHE_DONE.json"
    assert p.cache == Path("C:/Users/Admin/navsim-crun/exp/metric_cache_navtest_v2")


def test_two_stage_profiles_keep_the_two_stage_runner_and_the_c_logs():
    for name in ("warmup_two_stage", "navhard_two_stage"):
        p = P.SPLITS[name]
        assert (p.stages, p.runner, p.tts) == (2, "pdm_score", name)
        assert p.logs_dir == Path("C:/Users/Admin/navsim-crun/data/openscene/navsim_logs/test")


def test_the_protocol_declares_one_summary_row():
    shape = P.summary_row_shape("EPDMS_v2_navtest_single_stage")
    assert shape["rows"] == ("average_all_frames",) and shape["n"] == 1


def test_the_devkit_yaml_is_read_as_one_stage_with_the_literal_counts():
    y = P.read_split_yaml(NAVTEST)
    assert (len(y["stage_one"]), len(y["stage_two"]), len(y["log_names"]), len(y["mapping"])) == (12146, 0, 136, 0)
    t2l = P.token_to_log(P.SPLITS[NAVTEST])
    assert len(t2l) == 12146 and len(set(t2l.values())) == 136


def test_the_c_log_dir_would_silently_score_a_subset_and_the_profile_points_at_the_full_set():
    """⛔ The trap (brief 2026-09-26): dataloader.py:34-36 skips an absent log in silence."""
    logs = P.read_split_yaml(NAVTEST)["log_names"]
    d_dir = P.SPLITS[NAVTEST].logs_dir
    if not d_dir.exists():
        pytest.skip(f"NO_TREE: {d_dir} is not on this machine")
    on_c = sum((P.LOGS_C_NAVHARD / f"{ln}.pkl").exists() for ln in logs)
    on_d = sum((d_dir / f"{ln}.pkl").exists() for ln in logs)
    assert on_d == 136
    assert on_c < 136                      # MEASURED 2026-09-26: the C: dir holds navhard's 76 logs, not navtest's 136


# --------------------------------------------------------------------------- #
# 2. the ONE-STAGE runner is the one launched                                 #
# --------------------------------------------------------------------------- #
def _fake_devkit(csv_rows: list, tokens: list, captured: list):
    """A stand-in for the devkit subprocess: records the argv, writes the runner's log lines and a CSV."""
    def run(cmd, stdout=None, stderr=None, env=None, cwd=None, **kw):
        captured.append(list(cmd))
        out = Path(cmd[cmd.index("--out-dir") + 1]).parent / "fake_devkit_out.csv"
        header = ",token,valid," + ",".join(["no_at_fault_collisions", "drivable_area_compliance",
                                             "driving_direction_compliance", "traffic_light_compliance",
                                             "ego_progress", "time_to_collision_within_bound", "lane_keeping",
                                             "history_comfort", "two_frame_extended_comfort", "score"])
        lines = [header] + [f"{i},{r}" for i, r in enumerate(csv_rows)]
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        if stdout is not None:
            stdout.write(f"Number of successful scenarios: {len(tokens)}.\nNumber of failed scenarios: 0.\n"
                         f"Results are stored in: {out}.\n")
        return types.SimpleNamespace(returncode=0)
    return run


def _row(tok: str) -> str:
    return f"{tok},True,1.0,1.0,1.0,1.0,0.5,1.0,1.0,1.0,,0.7857142857142857"


def _score(tmp_path, monkeypatch, prof, rows, tokens):
    captured = []
    monkeypatch.setattr(SC.subprocess, "run", _fake_devkit(rows, tokens, captured))
    rep = SC.score_arm(arm="CV", prof=prof, raw_dir=tmp_path / "raw", exp_dir=tmp_path / "exp",
                       official_agent="constant_velocity_agent", tokens=tokens,
                       token_log={t: "LOG_A" for t in tokens}, log=lambda m: None)
    return rep, captured


TOKS = ["aaaaaaaaaaaaaaa1", "aaaaaaaaaaaaaaa2", "aaaaaaaaaaaaaaa3"]


def test_the_scorer_launches_the_one_stage_runner_with_the_pinned_overrides(tmp_path, monkeypatch):
    rows = [_row(t) for t in TOKS] + ["average_all_frames,True,1.0,1.0,1.0,1.0,0.5,1.0,1.0,1.0,,0.7857142857142857"]
    rep, cap = _score(tmp_path, monkeypatch, P.SPLITS[NAVTEST], rows, TOKS)
    cmd = cap[0]
    assert cmd[cmd.index("--script") + 1] == "pdm_score_one_stage"
    ov = cmd[cmd.index("--") + 1:]
    assert "train_test_split=navtest" in ov
    assert "traffic_agents=non_reactive" in ov
    assert "navsim_log_path=D:/Archive/devbox-C/navsim/data/openscene/navsim_logs/test" in ov
    assert "metric_cache_path=C:/Users/Admin/navsim-crun/exp/metric_cache_navtest_v2" in ov
    assert not [o for o in ov if o.startswith("synthetic_")]
    assert "train_test_split.scene_filter.log_names=['LOG_A']" in ov
    assert rep["status"] == "PASS", rep["failures"]
    assert rep["runner_script"] == "pdm_score_one_stage" and rep["n_tokens_subset"] == 3


def test_MUTATION_a_two_stage_runner_on_this_split_is_refused_before_launch():
    with pytest.raises(P.Refusal, match="must be scored with --script pdm_score_one_stage"):
        P.check_runner(P.SPLITS[NAVTEST], "pdm_score")


def test_MUTATION_rewiring_stage_one_to_the_two_stage_runner_goes_red(tmp_path, monkeypatch):
    """Reintroduce the defect for real: map ONE stage to the two-stage runner. The launched script
    changes, and the two-stage runner's CSV shape (three extended_pdm_score_* rows, no
    average_all_frames) FAILS the single-stage count guard."""
    monkeypatch.setattr(P, "RUNNER_FOR_STAGES", {1: "pdm_score", 2: "pdm_score"})
    two_stage_rows = [_row(t) for t in TOKS] + [f"{s},True,1,1,1,1,1,1,1,1,,1.0" for s in (
        "extended_pdm_score_stage_one", "extended_pdm_score_stage_two", "extended_pdm_score_combined")]
    rep, cap = _score(tmp_path, monkeypatch, P.SPLITS[NAVTEST], two_stage_rows, TOKS)
    assert cap[0][cap[0].index("--script") + 1] == "pdm_score"
    assert rep["status"] == "FAIL"
    joined = " | ".join(rep["failures"])
    assert "CSV summary rows [] != the EPDMS_v2_navtest_single_stage shape ['average_all_frames']" in joined
    assert "ANOTHER protocol" in joined


def test_the_one_stage_csv_control_passes_the_same_guard(tmp_path, monkeypatch):
    rows = [_row(t) for t in TOKS] + ["average_all_frames,True,1,1,1,1,0.5,1,1,1,,0.78"]
    rep, _ = _score(tmp_path, monkeypatch, P.SPLITS[NAVTEST], rows, TOKS)
    assert rep["status"] == "PASS" and rep["csv_valid_rows"] == 3 and rep["failures"] == []


def test_a_missing_token_fails_the_subset_guard(tmp_path, monkeypatch):
    rows = [_row(t) for t in TOKS[:2]] + ["average_all_frames,True,1,1,1,1,0.5,1,1,1,,0.78"]
    rep, _ = _score(tmp_path, monkeypatch, P.SPLITS[NAVTEST], rows, TOKS)
    assert rep["status"] == "FAIL"
    assert any("CSV token rows 2 / valid 2 != expected 3" in f for f in rep["failures"])


def test_a_multiline_preaggregation_dump_is_counted_in_records_not_lines(tmp_path, monkeypatch):
    """MEASURED 2026-09-26 (navtest sub200 smoke): the dump's ego_simulated_states cells span many
    lines; a line count read 24,600 for 200 records and hid AGGREGATION_FAILED behind a plain FAIL."""
    raw = tmp_path / "raw"
    raw.mkdir()

    def fake_run(cmd, stdout=None, **kw):
        cell = '"[[1.0 2.0]\n [3.0 4.0]\n [5.0 6.0]]"'                  # a numpy repr: 3 lines in ONE cell
        body = "".join(f"{t},0.5,{cell}\n" for t in TOKS)
        (raw / "CV_preaggregation.csv").write_text("token,score,ego_simulated_states\n" + body, encoding="utf-8")
        return types.SimpleNamespace(returncode=1)

    monkeypatch.setattr(SC.subprocess, "run", fake_run)
    rep = SC.score_arm(arm="CV", prof=P.SPLITS[NAVTEST], raw_dir=raw, exp_dir=tmp_path / "exp",
                       official_agent="constant_velocity_agent", tokens=TOKS,
                       token_log={t: "LOG_A" for t in TOKS}, log=lambda m: None)
    assert rep["preaggregation_rows"] == 3
    assert rep["status"] == "AGGREGATION_FAILED" and rep["scoring_complete"] is True


def test_the_two_stage_overrides_are_unchanged():
    ov = SC.overrides_for(P.SPLITS["warmup_two_stage"], exp_name="E", worker="sequential")
    assert ov == ["train_test_split=warmup_two_stage", "experiment_name=E",
                  "metric_cache_path=C:/Users/Admin/navsim/exp/metric_cache_warmup_two_stage",
                  "synthetic_sensor_path=C:/Users/Admin/navsim/data/openscene/warmup_two_stage/sensor_blobs",
                  "synthetic_scenes_path=C:/Users/Admin/navsim-crun/data/openscene/warmup_two_stage/synthetic_scene_pickles",
                  "worker=sequential"]
    with pytest.raises(P.Refusal, match="only defined on a single-stage split"):
        SC.overrides_for(P.SPLITS["warmup_two_stage"], exp_name="E", worker="sequential", tokens=["x"])


# --------------------------------------------------------------------------- #
# 3. refusals (controls)                                                      #
# --------------------------------------------------------------------------- #
def _bench_rec(root: Path) -> dict:
    return json.loads(next(root.rglob("bench_run.json")).read_text(encoding="utf-8"))


@pytest.mark.parametrize("split", ["navtest_two_stage_bogus", "navtest", "navtest_single"])
def test_an_unknown_split_still_refuses(tmp_path, split):
    rc = cli.main(["navsim_v2", "--ckpt", "none", "--split", split, "--results-root", str(tmp_path), "--no-report"])
    assert rc == 2
    rec = _bench_rec(tmp_path)
    assert rec["status"] == "REFUSED"
    assert f"split '{split}' not supported by navsim_v2 here" in rec["refusal"]


def test_a_token_subset_on_a_two_stage_split_refuses(tmp_path):
    tf = tmp_path / "t.json"
    tf.write_text(json.dumps(["x"]), encoding="utf-8")
    rc = cli.main(["navsim_v2", "--ckpt", "none", "--split", "warmup_two_stage", "--tokens-file", str(tf),
                   "--results-root", str(tmp_path), "--no-report"])
    assert rc == 2
    assert "single-stage options" in _bench_rec(tmp_path)["refusal"]


def test_a_cache_override_without_a_subset_refuses(tmp_path):
    rc = cli.main(["navsim_v2", "--ckpt", "none", "--split", NAVTEST, "--metric-cache", str(tmp_path / "c"),
                   "--results-root", str(tmp_path), "--no-report"])
    assert rc == 2
    assert "only accepted with --tokens-file" in _bench_rec(tmp_path)["refusal"]


def test_model_arms_are_refused_on_the_single_stage_split(tmp_path):
    ck = tmp_path / "m.pt"
    ck.write_bytes(b"0")
    rc = cli.main(["navsim_v2", "--ckpt", str(ck), "--split", NAVTEST, "--arms", "A1",
                   "--results-root", str(tmp_path), "--no-report"])
    assert rc == 2
    assert "not wired on a single-stage split" in _bench_rec(tmp_path)["refusal"]


def test_verify_cache_refuses_a_partial_cache_and_certifies_a_subset(tmp_path):
    cache = tmp_path / "metric_cache_x"
    (cache / "metadata").mkdir(parents=True)
    rows = ["file_name"] + [f"C:\\c\\LOG\\unknown\\{t}\\metric_cache.pkl" for t in ("tok1", "tok2")]
    (cache / "metadata" / "metric_cache_x_metadata_node_0.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (cache / "CACHE_DONE.json").write_text(json.dumps({"token_set": "SUBSET", "token_sets_equal_yaml": False,
                                                       "token_sets_equal_expected": True,
                                                       "scene_types_match_yaml": True}), encoding="utf-8")
    prof = dataclasses.replace(P.SPLITS[NAVTEST], cache=cache)
    with pytest.raises(P.Refusal, match="does not certify"):
        P.verify_cache(prof)                                  # the FULL split: a subset cache is not certified
    assert P.verify_cache(prof, expected={"tok1", "tok2"})["n_requested_tokens_cached"] == 2
    with pytest.raises(P.Refusal, match="1 of the 3 requested tokens are not cached"):
        P.verify_cache(prof, expected={"tok1", "tok2", "tok3"})


def test_export_agent_inputs_is_e2_plus_marked_additions():
    """The single-stage export lives in MARKED blocks: stripping them (the promotion test's own rule)
    must reproduce E2's origin BYTE-FOR-BYTE, and the two-stage path is therefore untouched."""
    from test_bench_suite_promotion import _blob, _strip_w1_additions
    bench = REPO / "taniteval" / "taniteval" / "bench"
    prov = json.loads((bench / "navsim/devkit_side/PROVENANCE.json").read_text(encoding="utf-8"))
    rec = prov["files"]["export_agent_inputs.py"]
    assert rec.get("w1_additions"), "the addition must be declared in PROVENANCE.json"
    raw = (bench / "navsim/devkit_side/export_agent_inputs.py").read_bytes()
    product = raw.decode("utf-8")
    origin = (REPO / rec["origin"]).read_bytes().decode("utf-8")
    assert _blob(raw) == rec["blob"]
    assert _strip_w1_additions(product) == origin
    assert product.count("# --- W1 ADDITION") == product.count("# --- end W1 ADDITION") == 3
    assert "def export_single_stage(" in product and "--single-stage" in product


def test_MUTATION_an_unmarked_edit_to_the_export_is_caught():
    """The pin has teeth: one unmarked character outside the blocks breaks the byte identity."""
    from test_bench_suite_promotion import _strip_w1_additions
    bench = REPO / "taniteval" / "taniteval" / "bench"
    prov = json.loads((bench / "navsim/devkit_side/PROVENANCE.json").read_text(encoding="utf-8"))
    product = (bench / "navsim/devkit_side/export_agent_inputs.py").read_text(encoding="utf-8")
    origin = (REPO / prov["files"]["export_agent_inputs.py"]["origin"]).read_text(encoding="utf-8")
    mutated = product.replace('SPLIT = os.environ.get("E2_SPLIT", "warmup_two_stage")',
                              'SPLIT = os.environ.get("E2_SPLIT", "navtest")', 1)
    assert mutated != product
    assert _strip_w1_additions(mutated) != origin


# --------------------------------------------------------------------------- #
# 4. dry runs                                                                  #
# --------------------------------------------------------------------------- #
def test_offline_dry_run_dispatches_to_the_one_stage_path(tmp_path, monkeypatch):
    seen = {}

    def fake_preflight(prof, **kw):
        seen["prof"], seen["kw"] = prof, kw
        return {"split": prof.name, "OFFLINE_TEST": "preflight replaced"}

    monkeypatch.setattr(P, "preflight", fake_preflight)
    rc = cli.main(["navsim_v2", "--ckpt", "none", "--split", NAVTEST, "--arms", "CV,STOP,HUMAN", "--dry-run",
                   "--results-root", str(tmp_path), "--no-report"])
    assert rc == 0
    run = next(tmp_path.rglob("bench_run.json")).parent
    rec = json.loads((run / "bench_run.json").read_text(encoding="utf-8"))
    plan = json.loads((run / "raw" / "plan.json").read_text(encoding="utf-8"))
    assert rec["status"] == "DRY_RUN_PASSED" and rec["protocol"] == "EPDMS_v2_navtest_single_stage"
    assert plan["runner"] == "pdm_score_one_stage" and plan["arms"] == ["CV", "STOP", "HUMAN"]
    assert "traffic_agents=non_reactive" in plan["overrides_example"]
    assert rec["stamps"]["reactive_background_traffic"] is False
    assert all(isinstance(v, str) for v in rec["stamps"]["loop"].values())      # W5's renderer formats them
    assert rec["split"]["n_scenes"] == 12146 and rec["split"]["n_logs"] == 136
    assert seen["prof"].name == NAVTEST and seen["kw"]["tokens"] is None


def _live_ok() -> str | None:
    if not P.PY.exists():
        return f"NO_TREE: the NavSim venv {P.PY} is not on this machine"
    if not P.SPLITS[NAVTEST].logs_dir.exists():
        return f"NO_TREE: {P.SPLITS[NAVTEST].logs_dir} is not on this machine"
    return None


def test_live_dry_run_on_the_smoke_subset(tmp_path):
    why = _live_ok()
    if why:
        pytest.skip(why)
    if not (SMOKE_CACHE / "CACHE_DONE.json").exists():
        pytest.skip(f"NO_TREE: the smoke cache {SMOKE_CACHE} is not built on this machine "
                    "(python -m taniteval.bench.navsim.cache_build --tokens-file … --cache …)")
    rc = cli.main(["navsim_v2", "--ckpt", "none", "--split", NAVTEST, "--tokens-file", str(SMOKE_TOKENS),
                   "--metric-cache", str(SMOKE_CACHE), "--dry-run", "--results-root", str(tmp_path), "--no-report"])
    run = next(tmp_path.rglob("bench_run.json")).parent
    rec = json.loads((run / "bench_run.json").read_text(encoding="utf-8"))
    assert rc == 0, rec.get("refusal")
    assert rec["status"] == "DRY_RUN_PASSED"
    assert "_scratch" in run.parts                           # a subset run is forced to scratch
    pre = json.loads((run / "raw" / "preflight.json").read_text(encoding="utf-8"))
    assert pre["subset"] == {"n_tokens": 200, "n_logs": 93} and pre["logs_present"] == 93


def test_live_dry_run_on_the_full_split(tmp_path):
    why = _live_ok()
    if why:
        pytest.skip(why)
    if not (P.SPLITS[NAVTEST].cache / "CACHE_DONE.json").exists():
        pytest.skip(f"NO_TREE: the 12,146-token v2 metric cache is not built on this machine "
                    f"({P.SPLITS[NAVTEST].cache}; build: cd taniteval && python -m taniteval.bench.navsim.cache_build "
                    f"--split {NAVTEST} --out <dir>)")
    rc = cli.main(["navsim_v2", "--ckpt", "none", "--split", NAVTEST, "--dry-run", "--results-root", str(tmp_path),
                   "--no-report"])
    rec = _bench_rec(tmp_path)
    assert rc == 0, rec.get("refusal")
    assert rec["status"] == "DRY_RUN_PASSED"


# --------------------------------------------------------------------------- #
# 5. OFFLINE twin of the smoke (the real single-stage code path; only the      #
#    devkit subprocesses are replaced by the smoke's BANKED outputs)           #
# --------------------------------------------------------------------------- #
FIX = PKG / "raw" / "smoke218_fixtures"
SMOKE218 = PKG / "raw" / "smoke_tokens_wholelog218.json"
#: LITERALS read off the banked devkit CSVs' `average_all_frames` rows (smoke218, 2026-09-26) —
#: never recomputed here.
SMOKE218_HEADLINES = {"CV": 0.26397269631654574, "STOP": 0.579805532200227, "HUMAN": 0.9512827651635418}


@pytest.fixture(scope="module")
def offline_single(tmp_path_factory):
    import shutil
    from taniteval.bench.navsim import export as EX
    need = [FIX / f"{a}{s}" for a in SMOKE218_HEADLINES for s in (".devkit.csv", "_hooks.json", "_final_scores_frame.csv")]
    need += [FIX / "navsim_agent_inputs.json", SMOKE218]
    miss = [str(p) for p in need if not p.exists()]
    assert not miss, f"MISSING banked smoke218 fixtures: {miss}"
    root = tmp_path_factory.mktemp("bench_single")
    calls = []

    def fake_preflight(prof, **kw):
        return {"split": prof.name, "OFFLINE_TEST": "preflight replaced", "cache": {"cache": str(prof.cache)}}

    def fake_export(prof, **kw):
        doc = json.loads((FIX / "navsim_agent_inputs.json").read_text(encoding="utf-8"))
        return doc, {"OFFLINE_TEST": "smoke218 banked export", "n_tokens": len(doc["tokens"])}

    def fake_score_arm(*, arm, prof, raw_dir, exp_dir, seam=None, official_agent=None, tokens=None, token_log=None, **kw):
        calls.append({"arm": arm, "seam": None if seam is None else str(seam), "official_agent": official_agent,
                      "n_tokens": None if tokens is None else len(tokens)})
        raw_dir = Path(raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(FIX / f"{arm}.devkit.csv", raw_dir / f"{arm}.devkit.csv")
        shutil.copyfile(FIX / f"{arm}_hooks.json", raw_dir / f"{arm}_hooks.json")
        shutil.copyfile(FIX / f"{arm}_final_scores_frame.csv", raw_dir / f"{arm}_final_scores_frame.csv")
        rep = {"arm": arm, "status": "PASS", "failures": [], "csv": str(raw_dir / f"{arm}.devkit.csv"), "wall_s": 0.0,
               "log_successful": 218, "log_failed": 0, "csv_valid_rows": 218, "csv_token_rows": 218,
               "expected_tokens": 218, "agent_calls": None, "runner_script": "pdm_score_one_stage",
               "traffic_agents": "non_reactive", "OFFLINE_TEST": "banked smoke218"}
        (raw_dir / f"{arm}.counts.json").write_text(json.dumps(rep), encoding="utf-8")
        return rep

    mp = pytest.MonkeyPatch()
    mp.setattr(P, "preflight", fake_preflight)
    mp.setattr(EX, "ensure_export", fake_export)
    mp.setattr(SC, "score_arm", fake_score_arm)
    try:
        rc = cli.main(["navsim_v2", "--ckpt", "none", "--split", NAVTEST, "--arms", "CV,STOP,HUMAN",
                       "--tokens-file", str(SMOKE218), "--metric-cache", str(tmp_path_factory.mktemp("c")),
                       "--no-report", "--results-root", str(root), "--run-id", "20260926T000000Z-navsim_v2-none-5a11e8"])
    finally:
        mp.undo()
    run = root / "_scratch" / "navsim_v2" / NAVTEST / "20260926T000000Z-navsim_v2-none-5a11e8"
    return rc, run, calls


def _s(run):
    return json.loads((run / "summary.json").read_text(encoding="utf-8"))


def test_offline_single_stage_run_completes_and_validates(offline_single):
    rc, run, calls = offline_single
    rec = json.loads((run / "bench_run.json").read_text(encoding="utf-8"))
    assert rc == 0, rec.get("exception") or rec.get("contract_errors") or rec.get("refusal")
    assert rec["status"] == "COMPLETE"
    from taniteval.bench import contract as C
    assert C.validate_bench_run(rec) == [] and C.validate_summary(_s(run)) == []
    assert [c["arm"] for c in calls] == ["CV", "STOP", "HUMAN"]
    assert calls[0]["official_agent"] == "constant_velocity_agent" and calls[2]["official_agent"] == "human_agent"
    assert calls[1]["official_agent"] is None and calls[1]["seam"].endswith("STOP.npz")
    assert all(c["n_tokens"] == 218 for c in calls)


def test_offline_headlines_are_the_banked_average_all_frames_literals(offline_single):
    s = _s(offline_single[1])
    for arm, want in SMOKE218_HEADLINES.items():
        h = s["arms"][arm]["headline"]
        assert h["value"] == want and h["row"] == "average_all_frames" and h["column"] == "score" and h["n"] == 218


def test_offline_controls_formula_average_and_counts_pass(offline_single):
    s = _s(offline_single[1])
    for arm in SMOKE218_HEADLINES:
        ctl = s["arms"][arm]["controls"]
        assert ctl["C4_formula"]["pass"] is True and ctl["C4_formula"]["n_rows"] == 218
        assert ctl["C_AVG"]["pass"] is True and ctl["C_AVG"]["n"] == 218


def test_offline_single_block_pairs_and_refused_interval(offline_single):
    s = _s(offline_single[1])
    assert s["protocol"] == "EPDMS_v2_navtest_single_stage" and s["claim_bearing"] is False
    assert s["provenance"]["harness"]["fix151"] == "post"
    for arm in SMOKE218_HEADLINES:
        a = s["arms"][arm]
        assert set(a["per_stage"]) == {"single_stage", "_note"}
        for f in ("STOP", "CV"):
            p = a["paired"][f]
            assert p["status"] == ("SELF" if f == arm else "OK")
            if f != arm:
                assert p["n_common"] == 218
        # 3 log clusters < the RG-14 floor of 8 -> the interval is REFUSED with its reason, never numeric
        assert a["interval"]["status"] == "UNAVAILABLE" and "n_clusters=3" in a["interval"]["reason"]
    assert s["arms"]["STOP"]["paired"]["CV"]["headline_delta"] == SMOKE218_HEADLINES["STOP"] - SMOKE218_HEADLINES["CV"]


def test_offline_four_families_reported_or_refused_per_family(offline_single):
    s = _s(offline_single[1])
    for arm in SMOKE218_HEADLINES:
        fam = s["arms"][arm]["families"]
        assert set(fam) == {"longitudinal", "lateral", "tactical", "strategic"}
        for k, b in fam.items():
            assert b["status"] in ("OK", "PARTIAL", "UNAVAILABLE")
            if b["status"] != "OK":
                assert b.get("reason"), (arm, k)


# --------------------------------------------------------------------------- #
# 6. --reuse-scored-arms: adopt a PASSED arm only under an exact identity      #
# --------------------------------------------------------------------------- #
def _src_run(root: Path, *, traffic="non_reactive", status="PASS", tokens=TOKS, manifest="m" * 64):
    from taniteval.bench.navsim import profiles as PP
    prof = PP.SPLITS[NAVTEST]
    run = root / "src_run"
    (run / "raw" / "CV").mkdir(parents=True)
    (run / "scores").mkdir()
    (run / "bench_run.json").write_text(json.dumps({"split": {"name": NAVTEST}, "protocol": prof.protocol,
                                                    "devkit": {"sha": "S" * 40, "patches": [{"name": "p", "raw_bytes_git_blob_now": "b"}]}}),
                                        encoding="utf-8")
    (run / "raw" / "preflight.json").write_text(json.dumps({"cache": {"cache_done": {"tokens_sha256": "t" * 64,
                                                                                     "manifest_sha256": manifest}}}),
                                                encoding="utf-8")
    counts = {"status": status, "rc": 0, "log_successful": len(TOKS), "log_failed": 0, "csv_valid_rows": len(TOKS),
              "runner_script": "pdm_score_one_stage", "traffic_agents": traffic,
              "cache": str(prof.cache).replace("\\", "/")}
    (run / "raw" / "CV" / "CV.counts.json").write_text(json.dumps(counts), encoding="utf-8")
    header = ",token,valid,no_at_fault_collisions,drivable_area_compliance,driving_direction_compliance," \
             "traffic_light_compliance,ego_progress,time_to_collision_within_bound,lane_keeping,history_comfort," \
             "two_frame_extended_comfort,score"
    body = [f"{i},{_row(t)}" for i, t in enumerate(tokens)] + [f"{len(tokens)},average_all_frames,True,1,1,1,1,1,1,1,1,,0.78"]
    (run / "scores" / "CV.csv").write_text("\n".join([header] + body) + "\n", encoding="utf-8")
    (run / "raw" / "CV" / "CV.devkit.csv").write_text("\n".join([header] + body) + "\n", encoding="utf-8")
    return run


class _Run:
    def __init__(self, path):
        self.path = Path(path)

    def p(self, rel):
        return self.path / rel


def _adopt(src, dst):
    from taniteval.bench.navsim import single_stage as SS
    pre = {"cache": {"cache_done": {"tokens_sha256": "t" * 64, "manifest_sha256": "m" * 64}}}
    return SS.adopt_scored_arm(arm="CV", src_run=src, run=_Run(dst), prof=P.SPLITS[NAVTEST], tokens_expected=set(TOKS),
                               devkit_sha="S" * 40, patches=[{"name": "p", "raw_bytes_git_blob_now": "b"}],
                               preflight=pre, export_sha256=None, new_seam=None)


def test_reuse_adopts_a_passed_arm_under_an_exact_identity(tmp_path):
    rep = _adopt(_src_run(tmp_path), tmp_path / "dst")
    assert rep["status"] == "PASS" and rep["reused_from"].endswith("src_run")
    assert (tmp_path / "dst" / "raw" / "CV" / "CV.devkit.csv").exists()
    assert json.loads((tmp_path / "dst" / "raw" / "CV" / "CV.reused.json").read_text(encoding="utf-8"))["n_tokens"] == 3


@pytest.mark.parametrize("mutation, message", [
    (dict(traffic="reactive"), "traffic_agents = 'reactive'"),
    (dict(status="RAM_GUARD_ABORT"), "status = 'RAM_GUARD_ABORT'"),
    (dict(tokens=TOKS[:2]), "token set differs: 2 vs 3"),
    (dict(manifest="x" * 64), "metric cache identity manifest_sha256"),
])
def test_MUTATION_reuse_refuses_any_identity_mismatch(tmp_path, mutation, message):
    from taniteval.bench.navsim import single_stage as SS
    with pytest.raises(SS.ArmReuseRefused, match=message.replace("(", r"\(").replace(")", r"\)")):
        _adopt(_src_run(tmp_path, **mutation), tmp_path / "dst")


def test_reuse_scored_arms_is_refused_on_a_two_stage_split(tmp_path):
    rc = cli.main(["navsim_v2", "--ckpt", "none", "--split", "warmup_two_stage", "--reuse-scored-arms", str(tmp_path),
                   "--results-root", str(tmp_path), "--no-report"])
    assert rc == 2 and "single-stage options" in _bench_rec(tmp_path)["refusal"]
