"""OFFLINE twin of the W1 acceptance: the REAL navsim_v2 benchmark code path (arm resolution, seams,
artifact builder, criteria_check, summary builder, schema + contract validation, reference checks),
with only the two devkit subprocesses (export, official scoring) replaced by E1/E2's BANKED outputs.

It needs no NavSim runtime compute (seconds), so it runs anytime; the live acceptance
(``python -m taniteval.bench navsim_v2 --ckpt none --split warmup_two_stage --arms CV,STOP``) runs the
devkit for real. Expectations are LITERALS read off the banked files by E1/E2, never recomputed here.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
E1 = REPO / "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-warmup-reference-epdms/raw"
E2 = REPO / "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-refcv4b-bridge/raw"
TTS = Path("C:/Users/Admin/navsim-crun/devkit/navsim/planning/script/config/common/train_test_split")

if not TTS.exists():                                   # NO_TREE: not this dev box — skip, never fail silently
    pytest.skip(f"NO_TREE: the NavSim C: runtime split yamls are not on this machine ({TTS})", allow_module_level=True)

BANKED = {   # arm -> (csv, hooks, final_scores_frame)
    "CV": (E1 / "A1/devkit_2026.09.19.12.24.20.csv", E1 / "A1/A1_hooks.json", E1 / "A1/A1_final_scores_frame.csv"),
    "STOP": (E2 / "score_STOP_zero.csv", E2 / "score_STOP_zero_wrapper/STOP_zero_hooks.json",
             E2 / "score_STOP_zero_wrapper/STOP_zero_final_scores_frame.csv"),
}


def _missing():
    return [str(p) for v in BANKED.values() for p in v if not p.exists()] + \
           ([str(E2 / "navsim_agent_inputs.json")] if not (E2 / "navsim_agent_inputs.json").exists() else [])


@pytest.fixture(scope="module")
def offline_run(tmp_path_factory):
    miss = _missing()
    assert not miss, f"MISSING banked E1/E2 fixtures (the historical record moved?): {miss}"
    from taniteval.bench import cli
    from taniteval.bench.navsim import benchmark as NB
    from taniteval.bench.navsim import export as EX
    from taniteval.bench.navsim import profiles as P
    root = tmp_path_factory.mktemp("bench_results")
    calls = []

    def fake_preflight(prof, **kw):
        return {"split": prof.name, "OFFLINE_TEST": "preflight replaced; the live acceptance runs the real one"}

    def fake_export(prof, **kw):
        doc = json.loads((E2 / "navsim_agent_inputs.json").read_text(encoding="utf-8"))
        return doc, {"OFFLINE_TEST": "E2 banked export", "n_tokens": len(doc["tokens"])}

    def fake_score_arm(*, arm, prof, raw_dir, exp_dir, seam=None, official_agent=None, **kw):
        calls.append({"arm": arm, "seam": None if seam is None else str(seam), "official_agent": official_agent})
        csv_src, hooks, frame = BANKED[arm]
        raw_dir = Path(raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(csv_src, raw_dir / f"{arm}.devkit.csv")
        shutil.copyfile(hooks, raw_dir / f"{arm}_hooks.json")
        shutil.copyfile(frame, raw_dir / f"{arm}_final_scores_frame.csv")
        rep = {"arm": arm, "status": "PASS", "failures": [], "csv": str(raw_dir / f"{arm}.devkit.csv"), "wall_s": 0.0,
               "log_successful": 220, "log_failed": 0, "csv_valid_rows": 220, "agent_calls": None,
               "OFFLINE_TEST": f"banked {csv_src.name}"}
        (raw_dir / f"{arm}.counts.json").write_text(json.dumps(rep), encoding="utf-8")
        return rep

    mp = pytest.MonkeyPatch()
    mp.setattr(P, "preflight", fake_preflight)
    mp.setattr(EX, "ensure_export", fake_export)
    mp.setattr(NB.SC, "score_arm", fake_score_arm)
    try:
        a = cli.build_parser().parse_args(["navsim_v2", "--ckpt", "none", "--split", "warmup_two_stage",
                                           "--arms", "CV,STOP", "--no-report", "--results-root", str(root),
                                           "--run-id", "20260919T000000Z-navsim_v2-none-0ff11e"])
        rc = cli.run_benchmark_cmd(a, log=lambda m: None)
    finally:
        mp.undo()
    run = root / "navsim_v2" / "warmup_two_stage" / "20260919T000000Z-navsim_v2-none-0ff11e"
    return rc, run, calls


def _summary(run):
    return json.loads((run / "summary.json").read_text(encoding="utf-8"))


def test_offline_run_completes_and_validates(offline_run):
    rc, run, calls = offline_run
    rec = json.loads((run / "bench_run.json").read_text(encoding="utf-8"))
    assert rc == 0, rec.get("exception") or rec.get("contract_errors") or rec.get("refusal")
    assert rec["status"] == "COMPLETE"
    from taniteval.bench import contract as C
    assert C.validate_bench_run(rec) == []
    assert C.validate_summary(_summary(run)) == []
    for arm in ("CV", "STOP"):
        for rel in (f"scores/{arm}.csv", f"artifacts/{arm}.json", f"criteria/{arm}.txt"):
            assert (run / rel).stat().st_size > 0, rel


def test_floor_arms_route_as_banked(offline_run):
    _, _, calls = offline_run
    by = {c["arm"]: c for c in calls}
    assert by["CV"]["official_agent"] == "constant_velocity_agent" and by["CV"]["seam"] is None
    assert by["STOP"]["official_agent"] is None and by["STOP"]["seam"].endswith("STOP.npz")


def test_headlines_are_the_banked_literals(offline_run):
    s = _summary(offline_run[1])
    assert s["arms"]["CV"]["headline"]["value"] == 0.1853562745165113
    assert s["arms"]["STOP"]["headline"]["value"] == 0.3009023137456225
    assert s["arms"]["CV"]["headline"]["column"] == "score"
    assert round(s["arms"]["CV"]["headline"]["x100"], 4) == 18.5356          # HF warmup LB (INHERITED)


def test_reference_checks_fire(offline_run):
    s = _summary(offline_run[1])
    for arm in ("CV", "STOP"):
        rc = s["arms"][arm]["controls"]["reference_check"]
        assert rc["file_identical"] is True and rc["cells_identical"] is True and rc["headline_equals_banked"] is True
    assert s["arms"]["CV"]["controls"]["reference_check"]["external"]["reproduced"] is True


def test_e2_statistics_reproduced(offline_run):
    """E2's parse_scores numbers (raw/scores_summary.json), re-derived by the suite's promoted code."""
    s = _summary(offline_run[1])
    assert s["arms"]["CV"]["statistics"]["S2_EPDMS_u"]["value"] == 0.39713167136274696
    assert s["arms"]["STOP"]["statistics"]["S2_EPDMS_u"]["value"] == 0.5212469877807624
    p = s["arms"]["CV"]["paired"]["STOP"]
    assert p["status"] == "OK"
    assert p["S2_EPDMS_u_delta"] == pytest.approx(-0.12411531641801543, abs=1e-15)
    st2 = p["by_stage"]["stage_two"]
    assert (st2["wins"], st2["ties"], st2["losses"], st2["n"]) == (83, 38, 83, 204)
    assert p["headline_delta"] == pytest.approx(0.1853562745165113 - 0.3009023137456225, abs=1e-15)
    assert s["arms"]["STOP"]["paired"]["STOP"] == {"status": "SELF"}


def test_controls_c4_c5_pass_on_banked(offline_run):
    s = _summary(offline_run[1])
    for arm in ("CV", "STOP"):
        ctl = s["arms"][arm]["controls"]
        assert ctl["C4_formula"]["pass"] is True and ctl["C4_formula"]["n_rows"] == 220
        assert ctl["C5_aggregate"]["pass"] is True
        assert ctl["navsim_gate_mutations_all_red"] is True


def test_no_interval_on_warmup(offline_run):
    s = _summary(offline_run[1])
    for arm in ("CV", "STOP"):
        iv = s["arms"][arm]["interval"]
        assert iv["status"] == "UNAVAILABLE" and iv["n"] == 7 and "RG-14" in iv["reason"]


def test_cv_families_reproduce_e1_artifact(offline_run):
    """E1's CV artifact (raw/artifacts/A1_...json): speed MAE 0.8928, cross MAE 1.0658, heading 7.4799 (literals)."""
    s = _summary(offline_run[1])
    f = s["arms"]["CV"]["families"]
    assert f["longitudinal"]["status"] == "PARTIAL"                  # distance-keeping has no lead block
    assert f["longitudinal"]["metrics"]["speed_mae_mps"] == 0.8928
    assert f["lateral"]["metrics"]["cross_mae_m"] == 1.0658
    assert f["lateral"]["metrics"]["heading_mae_deg"] == 7.4799
    assert f["strategic"]["status"] == "UNAVAILABLE" and f["strategic"]["reason"]
    assert f["longitudinal"]["n"] == 16


def test_per_log_and_stage_blocks(offline_run):
    s = _summary(offline_run[1])
    cv = s["arms"]["CV"]
    assert cv["per_stage"]["stage_one"]["score"] == 0.46028921296917297
    assert cv["per_stage"]["stage_two"]["score"] == 0.3341286472565139
    logs = [k for k in cv["per_log"] if not k.startswith("_")]
    assert len(logs) == 7
    assert sum(v["stage_one"]["n"] + v["stage_two"]["n"] for k, v in cv["per_log"].items() if not k.startswith("_")) == 220


def test_no_criteria_violations_on_any_arm(offline_run):
    """A criteria ABSENT (silent omission) on any arm is a suite defect — MEASURED on the first STOP
    build: the harness's lateral family left heading/curvature/yaw as None (3 violations)."""
    s = _summary(offline_run[1])
    for arm in ("CV", "STOP"):
        cc = s["arms"][arm]["controls"]["criteria_check"]
        assert cc["rc"] == 0 and cc["json_written"] is True, cc
        assert cc["n_violations"] == 0, (arm, cc)


def test_stop_lateral_terms_are_refused_not_absent(offline_run):
    run = offline_run[1]
    s = _summary(run)
    lat = s["arms"]["STOP"]["families"]["lateral"]
    assert lat["status"] == "PARTIAL" and "heading_mae_deg" in lat["reason"]
    art = json.loads((run / "artifacts" / "STOP.json").read_text(encoding="utf-8"))
    for k in ("heading_mae_deg", "curvature_mae_1pm", "yaw_rate_mae_degps"):
        v = art["four_families"]["lateral"][k]
        assert isinstance(v, dict) and v["status"] == "UNAVAILABLE" and v["reason"] and v["n"] == 0
    assert art["_w1_lateral_refusals"] == ["heading_mae_deg", "curvature_mae_1pm", "yaw_rate_mae_degps"]


# --------------------------------------------------------------------------- #
# W-25: a stand-in stage 1 means NO official two-stage headline — ever          #
# --------------------------------------------------------------------------- #
def test_hybrid_arm_headline_is_refused_with_reason_and_n():
    """⛔ MEASURED (E2, 2026-09-19): warmup's 16 stage-1 scenes ship no camera frames, so a camera
    arm's stage 1 is answered by the devkit's CV STAND-IN — and those rows set both the stage-1
    factor and the stage-2 Gaussian kernel weights. The devkit still prints an
    `extended_pdm_score_combined`; it is HYBRID (CV ⊕ model) and is NOT that arm's EPDMS.
    Fixture: E2's own banked A1 (`refcv4b::A1_ego_cmd`) CSV."""
    from taniteval.bench.navsim import artifacts as ART
    from taniteval.bench.navsim import summarize as S
    csv = E2 / "score_A1_ego_cmd.csv"
    assert csv.exists(), f"MISSING E2 fixture {csv}"
    y = __import__("taniteval.bench.navsim.profiles", fromlist=["x"]).read_split_yaml("warmup_two_stage")
    S1, S2 = set(y["stage_one"]), set(y["stage_two"])
    hdr, raw = S.read_raw_rows(csv)
    hybrid_value = float(raw[S.HEADLINE_ROW][S.HEADLINE_COLUMN])
    # the devkit's own TEXT cell, parsed by Python's float() — bit-exact.
    # ⚠️ MEASURED: E2's banked scores_summary.json records 0.2184508123602675 for the same cell —
    # ONE ULP lower — because pandas' DEFAULT csv float parser is not round-trip exact. That is why
    # the suite never reads a HEADLINE through pandas.
    assert hybrid_value == 0.21845081236026753
    assert hybrid_value != 0.2184508123602675
    art = ART.build_arm_artifact(
        arm="A1", spec={"agent": "model", "ii": dict(cameras=True, lidar=False, ego_velocity=True,
                                                     ego_acceleration=True, ego_pose_history=False,
                                                     driving_command=True),
                        "sensor_set": "3-camera stitch", "setting": "zero-shot", "goal": "NavSim command",
                        "route_input": True},
        split="warmup_two_stage", protocol="EPDMS_v2_warmup_two_stage", raw_rows=raw, hooks=[],
        S1=S1, S2=S2, n_logs=7, interval={"status": "UNAVAILABLE", "reason": "7 logs", "n": 7}, n_standin=16)
    sc = art["benchmark"]["navsim"]["score"]
    assert sc["status"] == "UNAVAILABLE" and sc["n"] == 16
    assert "STAND-IN" in sc["reason"] or "stand-in" in sc["reason"]
    assert "value" not in sc                                       # ⛔ no number may be read out of it
    fams = S.families_refused(ART.NO_GT, 0)
    assert all(v["status"] == "UNAVAILABLE" and v["reason"] and v["n"] == 0 for v in fams.values())
    tok = S.load_scores(csv, {t: 1 for t in S1} | {t: 2 for t in S2})
    assert S.s2_group_uniform(tok, y["mapping"])["value"] == 0.46702137433213753   # the honest statistic (E2)


# --------------------------------------------------------------------------- #
# W5's FAILURE GALLERY inputs: plans/<arm>.npz + scenes.json                    #
# --------------------------------------------------------------------------- #
def test_gallery_inputs_have_w5s_shape(offline_run):
    """W5's gallery reads plans/<arm>.npz (token/poses/source/sampling) and scenes.json (token ->
    stage/log/v0/command/scene ids). The poses are the SCORER'S OWN (E1's pdm_score hook)."""
    import numpy as np
    run = offline_run[1]
    s = _summary(run)
    scenes = json.loads((run / "scenes.json").read_text(encoding="utf-8"))
    assert set(scenes) >= {"tokens", "synthetic_scene_pickles", "frame_bank", "maps_root", "source", "what"}
    assert len(scenes["tokens"]) == 220
    one = scenes["tokens"][next(iter(scenes["tokens"]))]
    assert set(one) == {"stage", "log", "v0", "command", "command_onehot", "scene_token", "map_name",
                        "pickle", "frame_type", "num_future_frames"}
    for arm in ("CV", "STOP"):
        p = run / "plans" / f"{arm}.npz"
        assert p.exists() and s["arms"][arm]["files"]["plans"] == f"plans/{arm}.npz"
        z = np.load(p, allow_pickle=False)
        assert set(z.files) >= {"token", "poses", "source", "sampling", "arm", "seam_file"}
        assert z["poses"].shape == (220, 8, 3) and z["poses"].dtype == np.float32
        assert list(z["sampling"]) == [8, 0.5] and len(z["token"]) == 220
        assert set(str(x) for x in z["source"]) <= {"devkit_agent", "precomputed", "refcv4b", "cv_standin"}
    # the STOP plan is all zeros (it is the all-zero floor) — the poses are the scorer's, not the seam's
    z = np.load(run / "plans" / "STOP.npz", allow_pickle=False)
    assert float(np.abs(z["poses"]).max()) == 0.0
    # and CV's are not
    zc = np.load(run / "plans" / "CV.npz", allow_pickle=False)
    assert float(np.abs(zc["poses"]).max()) > 1.0
    assert s["controls"]["gallery_inputs"]["scenes"]["status"] == "OK"


# --- navhard STAGE-1 reference check (E1 relay 2026-09-20) ------------------------------------
# E1's MEASURED navhard CV stage-1 (their aborted run completed all 450 stage-1 scenes), against
# the PUBLISHED navhard leaderboard read by W1 from the BANKED PDF (library key 2506.04218,
# p.8 Table 2, column "CV [8]"). Literals, not expressions over the code under test.

E1_CV_S1 = {"NC": 88.89, "DAC": 42.89, "DDC": 70.67, "TLC": 99.33,
            "EP": 77.53, "TTC": 87.33, "LK": 78.67, "HC": 97.11}
PUB_CV_S1 = {"NC": 88.8, "DAC": 42.8, "DDC": 70.6, "TLC": 99.3,
             "EP": 77.5, "TTC": 87.3, "LK": 78.6, "HC": 97.1}


def _navhard_cv_ref():
    import json
    d = json.loads((Path(__file__).resolve().parents[1]
                    / "taniteval/bench/navsim/references.json").read_text(encoding="utf-8"))
    return d["navhard_two_stage"]["CV"]["stage_one"]


def test_banked_navhard_stage1_reference_matches_the_literals():
    ref = _navhard_cv_ref()
    assert ref["measured_e1"] == E1_CV_S1
    assert ref["published_n2"] == PUB_CV_S1
    assert ref["n"] == 450
    assert ref["comparison_rule"] == "TRUNCATE to 1 dp, never round"
    assert "2506.04218" in ref["published_source"]        # cited by LIBRARY KEY, not by URL


def test_truncation_matches_8_of_8_and_rounding_only_4_of_8():
    """⭐ THE DISCRIMINATOR. The paper TRUNCATES. A reproduction rule written around ROUNDING marks a
    CORRECT reproduction as a MISS on half the metrics -- and the 4 it fails are exactly the cells
    whose 2nd decimal is >= 5 (NC .89, DAC .89, DDC .67, LK .67)."""
    from taniteval.bench.navsim import summarize as S
    out = S.stage1_reference_check(E1_CV_S1, _navhard_cv_ref(), n=450)
    assert out["n_compared"] == 8
    assert out["n_match_truncating"] == 8, "truncation must reproduce the published column exactly"
    assert out["n_match_rounding"] == 4, "and rounding must NOT -- that is what makes the rule load-bearing"
    failed_rounding = sorted(k for k, v in out["vs_published"].items() if not v["match_rounding"])
    assert failed_rounding == ["DAC", "DDC", "LK", "NC"]


def test_stage1_check_is_an_identity_against_e1_and_a_one_cell_drift_fails_it():
    """⛔ MUTATION M-S1REF: our numbers must EQUAL E1's. A single cell moved by 0.01 must fail --
    this is an identity check (same split, same scorer, same 450 scenes), not a tolerance."""
    from taniteval.bench.navsim import summarize as S
    ref = _navhard_cv_ref()
    ok = S.stage1_reference_check(E1_CV_S1, ref, n=450)
    assert ok["status"] == "MATCH" and ok["identical_to_e1"] is True
    drift = dict(E1_CV_S1, TTC=87.34)
    bad = S.stage1_reference_check(drift, ref, n=450)
    assert bad["status"] == "MISMATCH" and bad["identical_to_e1"] is False
    assert bad["vs_e1_measured"]["TTC"]["equal_2dp"] is False
    assert bad["vs_e1_measured"]["NC"]["equal_2dp"] is True        # only the moved cell fails


def test_trunc1_does_not_round_and_survives_binary_float():
    from taniteval.bench.navsim.summarize import _trunc1
    assert _trunc1(88.89) == 88.8 and _trunc1(78.67) == 78.6
    assert _trunc1(99.33) == 99.3 and _trunc1(97.11) == 97.1
    assert _trunc1(88.85) == 88.8, "must TRUNCATE .85 down, never round it up"
    assert _trunc1(2.675) == 2.6, "the classic binary-float case: round() would disagree"


# --- PER-ARM devkit-side provenance (orchestrator ruling, 2026-09-20) --------------------------
# bench_run.json's `suite_code_blobs` is a RUN-START snapshot, but every devkit-side file is loaded
# by a FRESH SUBPROCESS PER ARM. A file changed mid-run is therefore recorded truthfully for arm 1
# and FALSELY for arm 2 -- provenance that reads true and is not. These pin the drift detector.

NAVSIM_WIN = "taniteval/taniteval/bench/navsim/devkit_side/navsim_win.py"
PRE_REPROMOTION_BLOB = "64186cceb3b032bf5002b824f452695e4107cd28"   # before sustain 3 -> 60


def test_devkit_side_blobs_are_raw_byte_git_blobs_of_the_whole_tree():
    from taniteval.bench.navsim import scoring as SC
    blobs = SC._devkit_side_blobs()
    names = sorted(k.split("devkit_side/")[-1] for k in blobs)
    assert names == ["PROVENANCE.json", "export_agent_inputs.py", "navsim_win.py",
                     "reaggregate.py", "tanitad_seam_agent.py"]
    assert all(len(v) == 40 for v in blobs.values())
    # the RAW-byte basis is load-bearing: it must agree with the promotion record, which is raw bytes
    import json
    prov = json.loads((Path(__file__).resolve().parents[1]
                       / "taniteval/bench/navsim/devkit_side/PROVENANCE.json").read_text(encoding="utf-8"))
    assert blobs[NAVSIM_WIN] == prov["files"]["navsim_win.py"]["blob"]


def test_drift_detector_flags_todays_real_case_and_stays_quiet_otherwise():
    """⛔ MUTATION M-DRIFT: today's actual event is the positive case -- navsim_win.py went
    64186cce -> 935fa44c WHILE a navhard run was scoring. The three negative controls matter as
    much: a detector that cries wolf on every arm is a detector nobody reads."""
    from taniteval.bench.navsim import scoring as SC
    cur = SC._devkit_side_blobs()
    assert SC.devkit_side_drift(cur, cur) == []                                  # identical -> quiet
    moved = dict(cur, **{NAVSIM_WIN: PRE_REPROMOTION_BLOB})
    assert SC.devkit_side_drift(cur, moved) == [NAVSIM_WIN]                      # the real case
    no_json = {k: v for k, v in cur.items() if not k.endswith(".json")}
    assert SC.devkit_side_drift(cur, no_json) == [], "absent from the baseline is NOT drift"
    assert SC.devkit_side_drift(cur, {}) == [], "no snapshot -> no drift CLAIM"


def test_score_arm_accepts_a_baseline_and_records_the_per_arm_fields():
    """Reachability, not existence: the field must be in the record the caller writes."""
    import inspect
    from taniteval.bench.navsim import scoring as SC, benchmark as BM
    assert "baseline_blobs" in inspect.signature(SC.score_arm).parameters
    src = inspect.getsource(SC.score_arm)
    for f in ("devkit_side_blobs_at_launch", "devkit_side_drift", "devkit_side_blob_basis"):
        assert f'"{f}"' in src, f"{f} never reaches the arm record"
    # and the ONE call site must actually pass the run-start snapshot
    bsrc = inspect.getsource(BM)
    assert "baseline_blobs=(ctx.rec.get(\"git\") or {}).get(\"suite_code_blobs\") or {}" in bsrc


# --- BOTH hashes, labelled (E1 refinement, 2026-09-20) ----------------------------------------

def test_every_file_backed_patch_entry_carries_both_hashes_and_says_what_each_is_blind_to():
    from taniteval.bench.navsim import profiles as P
    entries = [e for e in P.devkit_patches() if e.get("raw_bytes_git_blob_now")]
    assert len(entries) >= 4
    for e in entries:
        assert len(e["raw_bytes_git_blob_now"]) == 40
        assert "hash_basis" in e, f"{e['name']}: a hash with no stated basis is the whole defect"
        assert "BLIND" in e["hash_basis"]["normalised_git_blob_now"]


def test_the_two_hashes_are_DIFFERENT_probes_not_a_duplicated_call():
    """⭐ THE DISCRIMINATOR, and it needs both signs to mean anything:
      * a 100 %-CRLF file must give TWO DIFFERENT ids (else the normalisation is not happening);
      * an LF-only file must give the SAME id (else they are not the same hash family at all).
    Without the second, a test could pass against a function that just returned garbage."""
    from taniteval.bench.navsim import profiles as P
    crlf = P.DEVKIT / "navsim/planning/simulation/observation/navsim_idm/navsim_idm_agent_manager.py"
    raw, norm = P.git_blob(crlf), P.git_blob_normalised(crlf)
    assert crlf.read_bytes().count(bytes((13, 10))) > 0, "precondition: this file is CRLF"
    assert raw == "b95bcc7f62246017ea9ab5dca92383c6b50d234b"     # E1's apply_*_patch.py record
    assert norm == "214ef5ee39944c3584cb16eaaffd53853fc9db85"    # what `git hash-object` prints
    assert raw != norm, "a CRLF file MUST hash differently under the two bases"
    lf = P.CR / "venv/Lib/site-packages/fcntl.py"
    if lf.exists() and lf.read_bytes().count(bytes((13, 10))) == 0:
        assert P.git_blob(lf) == P.git_blob_normalised(lf), "an LF file must hash IDENTICALLY"


def test_e1s_historically_cited_blob_is_the_NORMALISED_id():
    """⚠️ MEASURED 2026-09-20: E1 RESULT.md section 4 cites `596cb7d` for dataclasses.py. That is the
    NORMALISED id, not the raw-byte one (20673663…). Comparing a historical git-side number to a
    raw-byte field is how a correct file reads as a mismatch -- which is exactly why both are now
    carried, labelled."""
    from taniteval.bench.navsim import profiles as P
    e = next(x for x in P.devkit_patches() if "dataclasses.py" in x["name"])
    assert e["normalised_git_blob_now"].startswith("596cb7d")
    assert not e["raw_bytes_git_blob_now"].startswith("596cb7d")


# --- scenes.json must never carry a NULL (the navhard report-hook failure, 2026-09-20) ---------
# ⛔ A key PRESENT with a null value defeats every consumer's `.get(key, default)`: the default fires
# only on an ABSENT key. W5's `Path(run.scenes.get("frame_bank", ""))` therefore got None and raised
# TypeError: Path(None), taking down the WHOLE report of a COMPLETE run whose scores were fine.
# ⚠️ It hid for every warmup run because warmup HAS a DEFAULT_BANKS entry and navhard does not --
# a defect masked by a default, which is why "it worked on warmup" proved nothing.

def test_a_present_but_null_key_defeats_the_dict_default():
    """The mechanism itself, as a literal. If this ever reads '' the language changed, not our code."""
    assert {"frame_bank": None}.get("frame_bank", "") is None      # the trap
    assert {}.get("frame_bank", "") == ""                          # what the author expected


def test_write_scenes_drops_top_level_nulls_and_says_so():
    from taniteval.bench.navsim.plans import _drop_top_level_nulls
    d = {"a": 1, "frame_bank": None, "b": "x", "c": None}
    assert _drop_top_level_nulls(d) == ["c", "frame_bank"]
    assert d == {"a": 1, "b": "x"}
    assert d.get("frame_bank", "") == "", "after the drop the consumer's default finally fires"
    clean = {"a": 1}
    assert _drop_top_level_nulls(clean) == [] and clean == {"a": 1}, "a clean dict must be untouched"


def test_every_scenes_json_in_the_results_tree_is_null_free_at_the_top_level():
    """⛔ MUTATION M-NULLSCENE: re-add `"frame_bank": None` to any scenes.json and this goes RED.
    Asserted on the ARTIFACTS, not on the writer -- the run that broke was written by an earlier
    version of the writer, so testing only the writer would have passed while the tree stayed broken."""
    import json
    from taniteval.bench import contract as C
    root = C.RESULTS_ROOT
    if not root.exists():
        pytest.skip("NO_TREE: no results/bench on this machine")
    seen = 0
    for f in root.rglob("scenes.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        nulls = sorted(k for k, v in d.items() if v is None)
        assert nulls == [], f"{f}: top-level nulls {nulls} -- a consumer's .get(k, default) will get None"
        seen += 1
    assert seen > 0, "control: this test must actually have READ some scenes.json, or it proves nothing"


def test_report_hook_records_a_REASON_not_only_a_log_path():
    """A failure that says only 'REPORT_FAILED, see the log' is a status disconnected from its cause."""
    from taniteval.bench.report_hook import _last_error_line
    import tempfile
    NL = chr(10)          # written this way on purpose: a backslash escape does not survive the
                          # heredoc that generates this file (a documented trap in CLAUDE.md)
    p = Path(tempfile.mkstemp(suffix=".log")[1])
    log = NL.join(["Traceback (most recent call last):",
                   "  File 'x', line 1",
                   "TypeError: argument should be a str, not 'NoneType'"]) + NL
    p.write_text(log, encoding="utf-8")
    assert _last_error_line(p) == "TypeError: argument should be a str, not 'NoneType'"
    p.write_text("", encoding="utf-8")
    assert "EMPTY" in _last_error_line(p), "an empty log must say so, not return an empty reason"
    assert "UNAVAILABLE" in _last_error_line(Path("does_not_exist.log"))


# --- an external reference must be SELF-CONSISTENT (the error that got banked, 2026-09-20) -----
# ⛔ The navhard entry carried `delta: 0.0` beside `value_x100: 11.4` while our value is 11.4816 --
# a delta of 0.0816. It refuted itself on its own face and was banked anyway, because it merged TWO
# artifacts: the PAPER (Table 2, 1 dp, TRUNCATES) and the HF LEADERBOARD (4 dp, ROUNDS). They do not
# disagree; they are different renderings of one number. Quoting one convention against the other
# source manufactures a mismatch -- or, as here, a FALSE AGREEMENT.

def _refs():
    import json
    return json.loads((Path(__file__).resolve().parents[1]
                       / "taniteval/bench/navsim/references.json").read_text(encoding="utf-8"))


def test_navhard_external_separates_the_paper_from_the_leaderboard():
    ext = _refs()["navhard_two_stage"]["CV"]["external"]
    assert ext["paper"]["rule"] == "TRUNCATES" and ext["paper"]["dp"] == 1
    assert ext["leaderboard"]["rule"] == "ROUNDS" and ext["leaderboard"]["dp"] == 4
    assert "_retraction" in ext, "the correction must travel with the entry, not be silently applied"


def test_every_external_reference_is_ARITHMETICALLY_self_consistent():
    """⭐ THE CHECK THAT WOULD HAVE CAUGHT IT, and it needs no new data -- only the entry's own
    fields. A stated delta must equal |ours - published| under the entry's OWN stated rule and dp."""
    import math
    from decimal import Decimal, ROUND_DOWN
    refs = _refs()
    ours = refs["navhard_two_stage"]["CV"]["combined_score_x100"]
    ext = refs["navhard_two_stage"]["CV"]["external"]
    for which, rule in (("paper", "TRUNCATES"), ("leaderboard", "ROUNDS")):
        e = ext[which]
        assert e["rule"] == rule
        if rule == "TRUNCATES":
            got = float(Decimal(str(ours)).quantize(Decimal("1e-%d" % e["dp"]), rounding=ROUND_DOWN))
        else:
            got = round(ours, e["dp"])
        assert math.isclose(got, e["value_x100"], abs_tol=1e-9), (
            f"{which}: ours {ours} under {rule} at {e['dp']} dp -> {got}, but the entry says "
            f"{e['value_x100']} — the entry contradicts itself")
        if "delta" in e:
            assert math.isclose(abs(got - e["value_x100"]), e["delta"], abs_tol=1e-9), (
                f"{which}: stated delta {e['delta']} is not |ours - published|")


def test_navtest_v1_full_split_numbers_are_banked():
    v1 = _refs()["navtest_v1"]
    assert v1["n_tokens"] == 12146 and v1["n_logs"] == 136
    assert v1["CV"]["pdms_x100"] == 20.6517
    assert v1["STOP"]["pdms_x100"] == 61.8202
    assert v1["HUMAN"]["pdms_x100"] == 94.5514
    assert v1["CV"]["interval_x100"] == {"lo": 19.20, "hi": 22.22}
    assert "12,146 tokens" in v1["summary_row_shape_confirmed_at_full_scale"] or            "12,146" in v1["summary_row_shape_confirmed_at_full_scale"]


def test_one_underlying_value_explains_BOTH_v1_renderings():
    """⭐ The settling measurement, re-derived here rather than quoted: 20.65165... truncates to the
    paper's 20.6 AND rounds to the leaderboard's 20.6517. One number, two conventions, no conflict."""
    from decimal import Decimal, ROUND_DOWN
    underlying = 20.65165
    assert float(Decimal(str(underlying)).quantize(Decimal("0.1"), rounding=ROUND_DOWN)) == 20.6
    assert round(underlying, 4) == 20.6517
    assert round(underlying, 1) == 20.7, "and ROUNDING to 1 dp would NOT give the paper's 20.6"
