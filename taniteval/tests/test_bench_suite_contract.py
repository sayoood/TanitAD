"""The run-directory CONTRACT and the CLI's refusals (W1)."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from taniteval.bench import contract as C


def test_run_id_grammar_and_ckpt_tag():
    now = dt.datetime(2026, 9, 19, 19, 30, 12, tzinfo=dt.timezone.utc)
    rid = C.make_run_id("navsim_v2", C.ckpt_tag(None), now=now, rand="a1b2c3")
    assert rid == "20260919T193012Z-navsim_v2-none-a1b2c3"
    assert C.ckpt_tag("D:/x/ckpt_40284_FINAL.pt") == "ckpt_40284_final"
    assert C.ckpt_tag("x.pt", "refcv4b-b1-v72-40k") == "refcv4b_b1_v72_40k"
    from taniteval.bench.schema_check import load_schema, validate
    assert validate({"run_id": rid}, {"properties": load_schema("bench_run")["properties"]}) == []


def test_run_dir_refuses_to_reuse_a_run_id(tmp_path):
    r = C.RunDir("navsim_v2", "warmup_two_stage", "20260919T193012Z-navsim_v2-none-a1b2c3", root=tmp_path).create()
    assert (r.path / "scores").is_dir() and (r.path / "raw").is_dir()
    with pytest.raises(C.ContractError):
        C.RunDir("navsim_v2", "warmup_two_stage", "20260919T193012Z-navsim_v2-none-a1b2c3", root=tmp_path).create()


def test_large_files_move_off_repo(tmp_path, monkeypatch):
    r = C.RunDir("navsim_v2", "warmup_two_stage", "20260919T193012Z-navsim_v2-none-b1b2c3", root=tmp_path).create()
    monkeypatch.setattr(C, "OFFREPO_ROOT", tmp_path / "offrepo")
    r.offrepo = (tmp_path / "offrepo") / "navsim_v2/warmup_two_stage/x"
    big, small = r.p("raw/big.bin"), r.p("raw/small.txt")
    big.write_bytes(b"0" * 5000)
    small.write_text("hi", encoding="utf-8")
    files = r.finalize_files(size_limit=1000)
    assert files["raw/small.txt"]["location"] == "repo"
    assert files["raw/big.bin"]["location"] == "offrepo" and files["raw/big.bin"]["bytes"] == 5000
    assert not big.exists() and json.loads((r.p("raw/big.bin.offrepo.json")).read_text())["sha256"] == \
        files["raw/big.bin"]["sha256"]
    assert Path(files["raw/big.bin"]["offrepo_path"]).exists()


def test_validate_summary_cross_field_rules():
    from test_bench_suite_schema import _summary                      # the good example
    s = _summary()
    assert C.validate_summary(s) == []
    bad = json.loads(json.dumps(s))
    bad["floors"] = ["STOP", "CV", "GHOST"]
    assert any("GHOST" in e for e in C.validate_summary(bad))
    bad = json.loads(json.dumps(s))
    bad["arms"]["CV"]["paired"].pop("STOP")
    assert any("no entry for floor" in e for e in C.validate_summary(bad))
    bad = json.loads(json.dumps(s))
    bad["arms"]["CV"]["paired"]["CV"] = {"status": "OK", "headline_delta": 0.0, "n_common": 1}
    assert any("SELF" in e for e in C.validate_summary(bad))
    bad = json.loads(json.dumps(s))
    bad["arms"]["CV"]["kind"] = "model"
    assert any("floor must have kind" in e for e in C.validate_summary(bad))


def test_context_refuses_a_protocol_outside_the_closed_set(tmp_path):
    r = C.RunDir("navsim_v2", "warmup_two_stage", "20260919T193012Z-navsim_v2-none-c1b2c3", root=tmp_path).create()
    ctx = C.BenchContext(argparse_ns(), r, log=lambda m: None)
    with pytest.raises(C.ContractError):
        ctx.set_protocol("EPDMS_v2")                                   # not a tag in the closed set
    ctx.set_protocol("EPDMS_v2_warmup_two_stage")
    with pytest.raises(C.ContractError):
        ctx.add_arm("X", "baseline", [])                               # kind outside model|floor|reference


def argparse_ns():
    import argparse
    return argparse.Namespace(device="cpu", ckpt=None, split="warmup_two_stage", arms=[])


# --------------------------------------------------------------------------- #
# the NavSim floor rule (M-FLOOR) and the arm refusals                         #
# --------------------------------------------------------------------------- #
def test_stop_and_cv_are_added_by_rule():
    from taniteval.bench.navsim.benchmark import resolve_arms
    arms, notes = resolve_arms([], ckpt=None)
    assert [n for n, _ in arms] == ["CV", "STOP"]
    assert all(i["added_by_rule"] for _, i in arms) and len(notes) == 2
    arms, notes = resolve_arms(["STOP"], ckpt=None)
    assert [n for n, _ in arms] == ["CV", "STOP"] and dict(arms)["STOP"]["added_by_rule"] is False


def test_echo_is_added_only_with_a_checkpoint():
    from taniteval.bench.navsim.benchmark import resolve_arms
    assert [n for n, _ in resolve_arms(["A1"], ckpt="x.pt")[0]] == ["CV", "STOP", "ECHO", "A1"]
    assert "ECHO" not in [n for n, _ in resolve_arms([], ckpt=None)[0]]


def test_model_arm_without_a_checkpoint_is_refused():
    from taniteval.bench.navsim.benchmark import resolve_arms
    from taniteval.bench.navsim.profiles import Refusal
    with pytest.raises(Refusal, match="needs --ckpt"):
        resolve_arms(["A1"], ckpt=None)


def test_human_arm_is_refused_on_two_stage_splits():
    from taniteval.bench.navsim.benchmark import resolve_arms
    from taniteval.bench.navsim.profiles import Refusal
    with pytest.raises(Refusal, match="UNDEFINED on two-stage"):
        resolve_arms(["HUMAN"], ckpt=None)


def test_unknown_arm_is_refused():
    from taniteval.bench.navsim.benchmark import resolve_arms
    from taniteval.bench.navsim.profiles import Refusal
    with pytest.raises(Refusal, match="unknown arm"):
        resolve_arms(["NOPE"], ckpt=None)


# --------------------------------------------------------------------------- #
# plugin slots (W3 / W6) and the CLI                                           #
# --------------------------------------------------------------------------- #
def test_plugin_slots_are_the_declared_w3_w6_contract():
    from taniteval.bench import plugins
    assert plugins.PLUGIN_SLOTS == {"navsim_v1": "W3", "nuscenes_ol": "W6"}


@pytest.mark.parametrize("bench", ["navsim_v1", "nuscenes_ol"])
def test_landed_plugin_exposes_run_benchmark(bench):
    """W3 (navsim_v1) and W6 (nuscenes_ol) replaced W1's docstring-only stubs on 2026-09-19/20."""
    from taniteval.bench import plugins
    fn = plugins.load(bench)
    assert callable(fn), bench


def test_a_missing_plugin_module_refuses(monkeypatch):
    from taniteval.bench import plugins
    monkeypatch.setitem(plugins.PLUGIN_SLOTS, "ghost_bench", "Wx")
    with pytest.raises(plugins.PluginNotLanded, match="module missing"):
        plugins.load("ghost_bench")


def test_a_docstring_only_stub_refuses(monkeypatch):
    """The state W1 shipped the slots in: importable, but no run_benchmark -> PLUGIN_NOT_LANDED."""
    import types
    from taniteval.bench import plugins
    stub = types.ModuleType("stub")
    stub.__file__ = "stub.py"
    monkeypatch.setattr(plugins.importlib, "import_module", lambda n: stub)
    with pytest.raises(plugins.PluginNotLanded, match="no run_benchmark"):
        plugins.load("navsim_v1")


def test_refused_benchmark_leaves_no_run_dir(tmp_path, capsys):
    """A plugin that has not landed refuses BEFORE any run directory is created."""
    from taniteval.bench import cli, plugins
    monkey = pytest.MonkeyPatch()
    monkey.setattr(plugins, "load", lambda b: (_ for _ in ()).throw(plugins.PluginNotLanded("PLUGIN_NOT_LANDED: x")))
    try:
        rc = cli.main(["nuscenes_ol", "--ckpt", "none", "--split", "x", "--results-root", str(tmp_path)])
    finally:
        monkey.undo()
    assert rc == 2 and "PLUGIN_NOT_LANDED" in capsys.readouterr().out
    assert not list(tmp_path.rglob("bench_run.json"))


def test_unsupported_split_is_refused(tmp_path):
    from taniteval.bench import cli
    rc = cli.main(["navsim_v2", "--ckpt", "none", "--split", "navtest", "--results-root", str(tmp_path),
                   "--no-report"])
    assert rc == 2
    rec = json.loads(next(tmp_path.rglob("bench_run.json")).read_text(encoding="utf-8"))
    assert rec["status"] == "REFUSED" and "navtest" in rec["refusal"]


def test_missing_checkpoint_is_refused(tmp_path):
    from taniteval.bench import cli
    rc = cli.main(["navsim_v2", "--ckpt", str(tmp_path / "nope.pt"), "--split", "warmup_two_stage",
                   "--results-root", str(tmp_path), "--no-report"])
    assert rc == 2


def test_aggregation_failure_is_not_a_failed_scoring_run(tmp_path, monkeypatch):
    """⛔ MEASURED 2026-09-19 (navhard CV): all 5,912 scenarios scored in 68 min, then the aggregation
    raised and the runner wrote NO CSV. The suite must classify that as AGGREGATION_FAILED with the
    per-token evidence preserved — never as a failed scoring run."""
    from taniteval.bench.navsim import profiles as P, scoring
    prof = P.SPLITS["warmup_two_stage"]
    raw = tmp_path / "raw"
    raw.mkdir()

    def fake_run(cmd, **kw):
        # the runner logs its scoring summary, writes the pre-aggregation dump, then dies
        log = Path(cmd[cmd.index("--out-dir") + 1]) / "CV.score.log"
        (raw / "CV_preaggregation.csv").write_text("token,score\n" + "".join(f"t{i},0.5\n" for i in range(220)),
                                                   encoding="utf-8")
        class R:
            returncode = 1
        return R()
    monkeypatch.setattr(scoring.subprocess, "run", fake_run)
    rep = scoring.score_arm(arm="CV", prof=prof, raw_dir=raw, exp_dir=tmp_path / "exp",
                            official_agent="constant_velocity_agent", log=lambda m: None)
    assert rep["status"] == "AGGREGATION_FAILED" and rep["scoring_complete"] is True
    assert rep["preaggregation_rows"] == 220
    assert any("AGGREGATION FAILED" in f for f in rep["failures"])
    assert not any(f.startswith("no result CSV") for f in rep["failures"])


def test_standin_rows_are_counted_from_the_seam_artifact(tmp_path):
    """W-25: the refusal keys on the ARTIFACT (the seam's own `source` column), not on a producer's
    report — any arm whose stage 1 was answered by the devkit CV stand-in loses its official
    two-stage headline."""
    import numpy as np
    from taniteval.bench.navsim.benchmark import _seam_standin_rows
    p = tmp_path / "A1.npz"
    np.savez(p, token=np.asarray([f"t{i}" for i in range(20)]),
             fingerprint=np.asarray(["f"] * 20),
             source=np.asarray(["cv_standin"] * 16 + ["refcv4b"] * 4),
             poses=np.zeros((20, 8, 3), np.float32), knots=np.zeros((20, 8, 2), np.float32),
             sampling=np.asarray([8, 0.5]), arm=np.asarray("A1"))
    assert _seam_standin_rows(p) == 16
    clean = tmp_path / "STOP.npz"
    np.savez(clean, token=np.asarray(["t0"]), fingerprint=np.asarray(["f"]),
             source=np.asarray(["precomputed"]), poses=np.zeros((1, 8, 3), np.float32),
             knots=np.zeros((1, 8, 2), np.float32), sampling=np.asarray([8, 0.5]), arm=np.asarray("STOP"))
    assert _seam_standin_rows(clean) == 0
    assert _seam_standin_rows(tmp_path / "does_not_exist.npz") == 0


# --------------------------------------------------------------------------- #
# W6: ONE nuScenes convention per run, chosen explicitly (no default)          #
# --------------------------------------------------------------------------- #
def test_nuscenes_convention_flags_exist_and_have_no_default():
    """W6 MEASURED that re-scoring the SAME checkpoints under the two legacy harnesses FLIPS the
    UniAD/VAD ranking — so the convention is chosen, never defaulted."""
    from taniteval.bench import cli
    a = cli.build_parser().parse_args(["nuscenes_ol", "--ckpt", "none", "--split", "val"])
    assert a.protocol is None and a.nuscenes_root is None and a.construction is None
    a = cli.build_parser().parse_args(["nuscenes_ol", "--ckpt", "none", "--split", "val",
                                       "--protocol", "nuScenes_OL_L2_uniad"])
    assert a.protocol == "nuScenes_OL_L2_uniad"
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["nuscenes_ol", "--ckpt", "none", "--split", "val",
                                       "--protocol", "nuScenes_OL_L2"])          # not in the closed set


def test_nuscenes_run_without_a_convention_is_refused(tmp_path, monkeypatch, capsys):
    """The plugin's refusal must stay reachable THROUGH the CLI (exit 2), with no run left behind
    that could be read as a result."""
    from taniteval.bench import cli
    monkeypatch.delenv("TANITAD_NUSCENES_PROTOCOL", raising=False)
    rc = cli.main(["nuscenes_ol", "--ckpt", "none", "--split", "val", "--results-root", str(tmp_path),
                   "--no-report"])
    assert rc == 2
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in tmp_path.rglob("bench_run.json")]
    assert recs and all(r["status"] == "REFUSED" for r in recs)
    assert any("convention" in (r.get("refusal") or "") or "TANITAD_NUSCENES_PROTOCOL" in (r.get("refusal") or "")
               for r in recs), [r.get("refusal") for r in recs]


# --------------------------------------------------------------------------- #
# the report hook: assert on the ARTIFACT, not on the exit code                 #
# --------------------------------------------------------------------------- #
def test_report_hook_reads_the_page_not_the_exit_code(tmp_path, monkeypatch):
    """MEASURED 2026-09-20: W5's CLI exits 3 when the page renders but its gallery is unavailable."""
    from taniteval.bench import report_hook as RH
    run = tmp_path / "run"
    (run / "report").mkdir(parents=True)
    (run / "raw").mkdir()
    (run / "report" / "index.html").write_text("<html>page</html>", encoding="utf-8")
    (run / "report" / "verify.json").write_text(json.dumps({"errors": 0, "numbers": 90}), encoding="utf-8")
    monkeypatch.setattr(RH, "report_available", lambda: (True, "stub"))

    class R:
        returncode = 3
    monkeypatch.setattr(RH.subprocess, "run", lambda *a, **k: R())
    out = RH.render(run)
    assert out["status"] == "RENDERED" and out["rc"] == 3 and "warning" in out and out["verify"]["errors"] == 0
    (run / "report" / "index.html").unlink()
    out = RH.render(run)
    assert out["status"] == "REPORT_FAILED" and out["index_exists"] is False


def test_report_pending_when_w5_is_absent(monkeypatch, tmp_path):
    from taniteval.bench import report_hook as RH
    monkeypatch.setattr(RH, "report_available", lambda: (False, "W5 has not landed it"))
    out = RH.render(tmp_path)
    assert out["status"] == "REPORT_PENDING" and "how_to_render_later" in out


def test_cli_exposes_the_training_box_override_and_refuses_through_it(tmp_path, monkeypatch):
    """The shared gate reaches the CLI: a DeviceRefused is a REFUSAL (exit 2), and the flag exists on
    every benchmark (orchestrator arbitration 2026-09-20)."""
    from taniteval.bench import cli, contract as CC
    for b in ("navsim_v2", "internal_t1", "navsim_v1", "nuscenes_ol"):
        a = cli.build_parser().parse_args([b, "--ckpt", "none", "--split", "x", "--accept-training-box-load"])
        assert a.accept_training_box_load is True, b
    # a benchmark that refuses on the device gate must exit 2 and leave a REFUSED record
    import taniteval.bench.navsim as NS
    monkeypatch.setattr(NS, "run_benchmark",
                        lambda ctx: (_ for _ in ()).throw(CC.DeviceRefused("model arms WAIT for a GPU gap")))
    rc = cli.main(["navsim_v2", "--ckpt", "none", "--split", "warmup_two_stage", "--results-root", str(tmp_path),
                   "--no-report"])
    assert rc == 2
    rec = json.loads(next(tmp_path.rglob("bench_run.json")).read_text(encoding="utf-8"))
    assert rec["status"] == "REFUSED" and "WAIT for a GPU gap" in rec["refusal"]


def test_ckpt_none_never_consults_the_device_gate(monkeypatch):
    """Floors and reference agents are the devkit's own CPU scorer — not our-model inference — so a
    floors-only run must not be blocked by a training process."""
    from taniteval.bench import gpu_gap as G
    monkeypatch.setattr(G, "device_policy", lambda *a, **k: (_ for _ in ()).throw(AssertionError("gate called")))
    from taniteval.bench.navsim.benchmark import resolve_arms
    arms, _ = resolve_arms(["CV", "STOP"], ckpt=None)
    assert all(i["kind"] == "floor" for _, i in arms)      # nothing here runs a model


def test_summary_row_shape_belongs_to_the_protocol():
    """W3 (2026-09-20): NAVSIM v1.1 CSVs carry ONE `average` row — the three extended_pdm_score_*
    rows are the v2 two-stage shape. A checker that hard-codes 3 fails a correct v1 CSV."""
    from taniteval.bench.navsim import profiles as P
    v2 = P.summary_row_shape("EPDMS_v2_warmup_two_stage")
    assert v2["n"] == 3 and v2["headline_row"] == "extended_pdm_score_combined"
    v1 = P.summary_row_shape("PDMS_v1_navtest")
    assert v1["n"] == 1 and v1["rows"] == ("average",) and v1["headline_row"] == "average"
    one = P.summary_row_shape("EPDMS_v2_navtest_single_stage")
    assert one["n"] == 1 and one["rows"] == ("average_all_frames",)
    for shape in (v2, v1, one):
        assert shape["evidence"]
    with pytest.raises(P.Refusal, match="does not declare its summary-row shape"):
        P.summary_row_shape("nuScenes_OL_L2_stp3")


# --------------------------------------------------------------------------- #
# the results tree is APPEND-ONLY (W4, 2026-09-20)                             #
# --------------------------------------------------------------------------- #
def test_scratch_runs_land_in_a_subtree_consumers_skip(tmp_path):
    r = C.RunDir("navsim_v2", "warmup_two_stage", "20260920T193012Z-navsim_v2-none-a1b2c3",
                 root=tmp_path, scratch=True).create()
    assert C.SCRATCH_DIR in r.path.parts and r.path.parent.parent.parent.name == C.SCRATCH_DIR
    ok, why = C.is_consumable_run(r.path)
    assert ok is False and C.SCRATCH_DIR in why
    normal = C.RunDir("navsim_v2", "warmup_two_stage", "20260920T193012Z-navsim_v2-none-b1b2c3",
                      root=tmp_path).create()
    (normal.path / "bench_run.json").write_text("{}", encoding="utf-8")
    assert C.is_consumable_run(normal.path) == (True, "ok")


def test_a_withdrawn_run_is_tombstoned_never_deleted(tmp_path):
    """⛔ MEASURED by W4 2026-09-20: a leaderboard page cited a run directory that had been deleted."""
    r = C.RunDir("navsim_v2", "warmup_two_stage", "20260920T193012Z-navsim_v2-none-c1b2c3",
                 root=tmp_path).create()
    (r.path / "bench_run.json").write_text("{}", encoding="utf-8")
    p = C.write_tombstone(r.path, superseded_by="20260920T203012Z-navsim_v2-none-d1b2c3",
                          why="superseded by the frozen-code run", evidence_kept="identical CSVs")
    assert p.exists() and r.path.exists(), "the directory must survive"
    t = json.loads(p.read_text(encoding="utf-8"))
    assert t["TOMBSTONE"] is True and t["superseded_by"].endswith("d1b2c3") and t["why"]
    ok, why = C.is_consumable_run(r.path)
    assert ok is False and "withdrawn" in why


def test_cli_tombstone_subcommand_deletes_nothing(tmp_path, capsys):
    from taniteval.bench import cli
    d = tmp_path / "20260920T193012Z-navsim_v2-none-e1b2c3"
    d.mkdir()
    (d / "summary.json").write_text("{}", encoding="utf-8")
    rc = cli.main(["tombstone", str(d), "--superseded-by", "20260920T203012Z-navsim_v2-none-f1b2c3",
                   "--why", "superseded"])
    assert rc == 0 and (d / C.TOMBSTONE_FILE).exists() and (d / "summary.json").exists()
    assert "nothing was deleted" in capsys.readouterr().out


def test_every_results_run_is_either_consumable_or_tombstoned():
    """The live tree: every directory is a result, a tombstone, or under _scratch/ — nothing dangling."""
    root = C.RESULTS_ROOT
    if not root.exists():
        pytest.skip("NO_TREE: no results/bench on this machine")
    for bench in root.iterdir():
        if not bench.is_dir() or bench.name == C.SCRATCH_DIR:
            continue
        for split in bench.iterdir():
            for run in (d for d in split.iterdir() if d.is_dir()):
                ok, why = C.is_consumable_run(run)
                assert ok or (run / C.TOMBSTONE_FILE).exists() or "IN FLIGHT" in why, f"{run}: {why}"


# --- dry-run VOCABULARY (E1, 2026-09-20) ------------------------------------------------------
# A PASSING dry run used to print BENCH_STATUS=REFUSED -- a failure word on a success, which is how
# a clean preflight gets copied into a report as a failure. These pin the fix at three levels:
# the schema's enum, the CLI's normalising seam, and the end-to-end status line.

def test_dry_run_passed_is_a_pass_not_a_refusal_in_the_schema():
    from taniteval.bench.schema_check import load_schema
    enum = load_schema("bench_run")["properties"]["status"]["enum"]
    assert enum == ["COMPLETE", "PARTIAL", "FAILED", "REFUSED", "DRY_RUN_PASSED"]
    assert "DRY_RUN_PASSED" in enum and "REFUSED" in enum, "both words must exist and mean DIFFERENT things"


def test_navsim_dry_run_sets_the_pass_status_not_refused():
    """The producer says it directly, so reading benchmark.py alone cannot mislead."""
    src = (Path(__file__).resolve().parents[1] / "taniteval/bench/navsim/benchmark.py").read_text(encoding="utf-8")
    i = src.index('if getattr(a, "dry_run", False):')
    block = src[i:i + 700]
    assert 'ctx.rec["status"] = "DRY_RUN_PASSED"' in block
    assert 'ctx.rec["status"] = "REFUSED"' not in block, "a passing dry run must never be REFUSED"


def test_cli_normalises_a_dry_run_refusal_to_the_pass_status():
    """⛔ The MUTATION this guards (M-DRYRUN): drop the normalising branch in cli.py and a plugin that
    still sets REFUSED on a passing dry run reports a failure word. The seam is normalising ON PURPOSE
    -- W3's navsim_v1 plugin sets REFUSED and is fixed here WITHOUT editing a file W1 does not own."""
    src = (Path(__file__).resolve().parents[1] / "taniteval/bench/cli.py").read_text(encoding="utf-8")
    i = src.index('elif ctx.rec.get("status") == "REFUSED":')
    block = src[i:i + 900]
    assert 'ctx.rec["status"], code = "DRY_RUN_PASSED", EXIT_OK' in block
    assert 'if ctx.rec.get("dry_run"):' in block
    # and a REAL refusal must still be REFUSED and must NOT exit 0
    assert "code = EXIT_REFUSED" in block


def test_a_real_refusal_keeps_the_refusal_word():
    """The discriminating control: the fix must not turn every refusal into a pass."""
    src = (Path(__file__).resolve().parents[1] / "taniteval/bench/cli.py").read_text(encoding="utf-8")
    assert 'ctx.rec["status"], ctx.rec["refusal"] = "REFUSED", str(e)' in src
    assert src.count("EXIT_REFUSED") >= 5


# --- provenance.ckpt: a MODEL row must name its checkpoint (W4, 2026-09-20) --------------------
# ⛔ The leaderboard reads summary.json. Without this, a model row cannot say WHICH checkpoint made
# it. Structural on purpose: validate_summary is what write_summary raises on, so such a run FAILS
# its contract instead of publishing an anonymous row.

_MODEL = {"REFCV4B": {"kind": "model"}, "CV": {"kind": "floor"}}
_FLOORS = {"CV": {"kind": "floor"}, "STOP": {"kind": "floor"}}


def test_floor_only_run_needs_no_ckpt():
    """A devkit-floor run has no checkpoint and the page says so -- the check must stay SILENT."""
    assert C._ckpt_identified_for_model_arms({"provenance": {"ckpt": None}}, _FLOORS) == []


def test_model_arm_without_a_ckpt_is_refused():
    errs = C._ckpt_identified_for_model_arms({}, _MODEL)
    assert len(errs) == 1 and "/provenance/ckpt" in errs[0] and "REFCV4B" in errs[0]


def test_identity_fields_are_a_HARD_refusal():
    for missing in ("path", "sha256"):
        ck = {"path": "D:/x.pt", "sha256": "a" * 64, "registry_key": "k"}
        ck[missing] = None
        errs = C._ckpt_identified_for_model_arms({"provenance": {"ckpt": ck}}, _MODEL)
        assert any(f"/provenance/ckpt/{missing}" in e for e in errs), f"{missing} must be refused"


def test_registry_key_may_be_ABSENT_only_if_its_absence_is_STATED():
    """⚠️ The refinement, and the reason for it: a checkpoint identified by sha256 is NOT anonymous,
    so refusing over a missing cross-reference would reject a reproducible result. What is forbidden
    is SILENCE -- state the key, or state why it is missing."""
    base = {"path": "D:/x.pt", "sha256": "a" * 64}
    assert C._ckpt_identified_for_model_arms({"provenance": {"ckpt": base}}, _MODEL), "silence must be refused"
    with_key = dict(base, registry_key="refcv4b-b1-v72-40k")
    assert C._ckpt_identified_for_model_arms({"provenance": {"ckpt": with_key}}, _MODEL) == []
    with_status = dict(base, registry_key_status="NOT NAMED by the operator (--registry-key)")
    assert C._ckpt_identified_for_model_arms({"provenance": {"ckpt": with_status}}, _MODEL) == []


def test_every_model_bearing_summary_in_the_tree_names_its_checkpoint():
    """⛔ MUTATION M-ANONROW: null out provenance.ckpt on a summary with a model arm -> RED.
    Asserted on the ARTIFACTS: this is exactly how the internal_t1 run was caught publishing a model
    arm with a null checkpoint while the path sat one file down in raw/refcv3_arm.json."""
    import json
    root = C.RESULTS_ROOT
    if not root.exists():
        pytest.skip("NO_TREE: no results/bench on this machine")
    seen_model = 0
    for f in root.rglob("summary.json"):
        if C.SCRATCH_DIR in f.parts:
            continue
        s = json.loads(f.read_text(encoding="utf-8"))
        arms = s.get("arms") or {}
        errs = C._ckpt_identified_for_model_arms(s, arms)
        assert errs == [], f"{f}: {errs}"
        seen_model += any(str(a.get("kind", "")).lower() == "model" for a in arms.values())
    assert seen_model > 0, "control: the tree must contain at least one MODEL-bearing summary, or this proves nothing"


# --- a checkpoint cell is NEVER blank (orchestrator ruling, 2026-09-20) ------------------------

def test_ckpt_display_never_returns_a_blank():
    """⛔ A blank cell reads as 'no checkpoint' when the truth may be 'identified, not
    cross-referenced'. Those are different claims and only one of them is a problem."""
    cases = [
        ({"registry_key": "refcv4b-b1-v72-40k", "sha256": "a" * 64}, "refcv4b-b1-v72-40k"),
        ({"sha256": "54320ec2d72a0b6fd9f2413b"}, "sha256:54320ec2d72a"),
        ({"registry_key": "   ", "sha256": "b" * 64}, "sha256:bbbbbbbbbbbb"),   # whitespace is not a key
        ({"path": "C:/x.pt"}, "UNIDENTIFIED (path only, no sha256)"),
        ({"path": None, "sha256": None, "registry_key": None}, "no checkpoint (devkit floors only)"),
        (None, "no checkpoint (devkit floors only)"),
    ]
    for ck, want in cases:
        got = C.ckpt_display(ck)
        assert got == want, f"{ck} -> {got!r} != {want!r}"
        assert got.strip(), "NEVER blank"


def test_every_summary_in_the_tree_shows_something_identifying():
    """⛔ MUTATION M-BLANKKEY: blank a registry_key_display in any summary -> RED. Asserted on the
    ARTIFACTS, because the renderer reads the artifact, not the function."""
    import json
    root = C.RESULTS_ROOT
    if not root.exists():
        pytest.skip("NO_TREE: no results/bench on this machine")
    seen = 0
    for f in root.rglob("summary.json"):
        if C.SCRATCH_DIR in f.parts:
            continue
        ck = (json.loads(f.read_text(encoding="utf-8")).get("provenance") or {}).get("ckpt")
        if not isinstance(ck, dict):
            continue
        disp = ck.get("registry_key_display")
        assert isinstance(disp, str) and disp.strip(), f"{f}: blank/absent registry_key_display"
        if not str(ck.get("registry_key") or "").strip() and str(ck.get("sha256") or "").strip():
            assert disp.startswith("sha256:"), f"{f}: no key, so the sha256 must stand in its place"
        seen += 1
    assert seen > 0, "control: the tree must contain at least one ckpt block, or this proves nothing"
