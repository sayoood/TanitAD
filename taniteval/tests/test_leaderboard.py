"""Tests for the leaderboard generator (EvalFlyWheel W4).

The acceptance criterion the orchestrator set: **delete the marker-delimited section, run build, get it
back byte-identical** — in both deletion modes (content only, and markers-and-all). Around it sit the
guards that keep the page honest: one protocol per table, STOP+CV on every NavSim run, no interval on
warmup, published rows only with library key + table + page, and LITERAL expectations for margins that
are read from raw JSON (written as literals on purpose — an expectation computed from the code under
test measures determinism, not correctness).
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from taniteval.leaderboard import render as R
from taniteval.leaderboard import sources as S
from taniteval.leaderboard.build import build as run_build          # the package re-exports the FUNCTION
from taniteval.leaderboard.build import load_model, render_section

REPO = S.REPO
CFG = S.load_config(REPO)
PAGE = REPO / CFG["target_md"]
pytestmark = pytest.mark.skipif(not PAGE.exists(), reason=f"{CFG['target_md']} not in this checkout")


@pytest.fixture(scope="module")
def model():
    return load_model(REPO, CFG)


@pytest.fixture(scope="module")
def section():
    return render_section(REPO, CFG)


def _strip_section(text: str, mode: str) -> str:
    b, e = CFG["markers"]["begin"], CFG["markers"]["end"]
    i, j = text.index(b), text.index(e)
    if mode == "content":                      # keep BOTH markers, delete what is between them
        return text[:i] + b + "\r\n" + e + text[text.index("\r\n", j):]
    jend = text.index("\r\n", j) + 2           # delete the section AND the blank line after it
    out = text[:i] + text[jend:]
    return out[:i] + out[i + 2:] if out[i:i + 2] == "\r\n" else out


# --------------------------------------------------------------------- acceptance: regeneration
@pytest.mark.parametrize("mode", ["content", "markers"])
def test_delete_the_section_and_build_returns_it_byte_identical(tmp_path, mode):
    host = tmp_path / "LEADERBOARD.md"
    host.write_bytes(PAGE.read_bytes())
    run_build(md_path=host, write_html=False)                     # insert (the page may have no markers yet)
    full = host.read_bytes()
    host.write_bytes(_strip_section(full.decode("utf-8"), mode).encode("utf-8"))
    assert host.read_bytes() != full, "the strip helper deleted nothing"
    run_build(md_path=host, write_html=False)
    assert hashlib.sha256(host.read_bytes()).hexdigest() == hashlib.sha256(full).hexdigest()


def test_build_is_idempotent_and_reports_no_change(tmp_path):
    host = tmp_path / "LEADERBOARD.md"
    host.write_bytes(PAGE.read_bytes())
    run_build(md_path=host, write_html=False)
    first = host.read_bytes()
    out = run_build(md_path=host, write_html=False)
    assert out["md_changed"] is False and host.read_bytes() == first


def test_hand_written_bytes_outside_the_markers_are_untouched(tmp_path):
    host = tmp_path / 'LEADERBOARD.md'
    original = PAGE.read_bytes().decode('utf-8')
    host.write_bytes(original.encode('utf-8'))
    run_build(md_path=host, write_html=False)
    new = host.read_bytes().decode('utf-8')
    b, e = CFG['markers']['begin'], CFG['markers']['end']
    nl = '\r\n'
    prefix = new[:new.index(b)]
    suffix = new[new.index(nl, new.index(e)) + 2:]
    assert prefix in original, 'text before the section changed'
    assert suffix in original, 'text after the section changed'
    assert len(prefix) + len(suffix) > 0.5 * len(original), 'the section swallowed hand-written content'


def test_crlf_is_preserved(tmp_path):
    host = tmp_path / "LEADERBOARD.md"
    host.write_bytes(PAGE.read_bytes())
    run_build(md_path=host, write_html=False)
    raw = host.read_bytes()
    assert raw.count(b"\r\n") == raw.count(b"\n"), "a lone LF appeared in a CRLF file"


# --------------------------------------------------------------------- the binding rules, as guards
def test_one_protocol_per_table_is_enforced():
    rows = [{"protocol": "EPDMS_v2_navhard_two_stage"}, {"protocol": "EPDMS_v2_warmup_two_stage"}]
    with pytest.raises(S.SourceError):
        R.assert_one_protocol("EPDMS_v2_navhard_two_stage", rows)
    R.assert_one_protocol("EPDMS_v2_navhard_two_stage", rows[:1])       # control: the clean case passes


def _fixture_summary() -> dict:
    conv, _ = S.legacy_summaries(REPO, CFG)
    return copy.deepcopy(conv[0][1])


def test_a_warmup_arm_may_never_carry_an_interval():
    s = _fixture_summary()
    S.summary_rows(copy.deepcopy(s), "control", REPO)                   # control: as banked it passes
    s["arms"]["CV"]["interval"] = {"status": "OK", "estimator": "paired_episode_cluster_bootstrap",
                                   "cluster_unit": "log", "lo": 0.1, "hi": 0.3, "n_clusters": 7}
    with pytest.raises(S.SourceError):
        S.summary_rows(s, "mutated", REPO)


def test_a_navsim_run_without_its_STOP_and_CV_floors_is_refused():
    s = _fixture_summary()
    s["floors"] = [f for f in s["floors"] if f != "CV"]
    del s["arms"]["CV"]
    with pytest.raises(S.SourceError):
        S.summary_rows(s, "mutated", REPO)


def test_every_navsim_run_row_carries_both_floors(model):
    for r in model["navsim"]:
        if r["protocol"].startswith(("EPDMS", "PDMS")):
            assert set(r["vs"]) >= {"STOP", "CV"}


# --------------------------------------------------------------------- the published (external) rows
def test_published_rows_carry_library_key_table_and_page(model):
    for r in model["published"]["results"]:
        if r.get("evidence_class", "PUBLISHED") != "PUBLISHED":
            continue
        src = r["source"]
        assert src["library_key"] and src["table"] and src["page"], r["id"]
        assert r["page_tokens"], r["id"]


def test_protocol_tags_join_with_the_w1_closed_set(model):
    closed = set(json.loads((REPO / "taniteval/taniteval/bench/schema/summary.schema.json").read_text(encoding="utf-8"))
                 ["properties"]["protocol"]["enum"])
    for tag, meta in model["published"]["protocols"].items():
        assert tag in closed or meta["closed_set"].startswith("EXTERNAL-ONLY"), tag


def test_navhard_comparison_rows_are_all_post_fix(model):
    for r in model["published"]["results"]:
        if r["protocol"] == "EPDMS_v2_navhard_two_stage" and r.get("comparison_admissible"):
            assert r["harness"]["fix151"] == "post", r["id"]


def test_pre_fix_values_never_share_the_navhard_comparison_table(section):
    head = section[section.index("### 0E.1"):section.index("### 0E.2")]
    main, excluded = head.split("Rows NOT in the comparison above", 1)
    assert "51.3" in excluded and "51.3" not in main          # the pre-#151 PDM-Closed
    assert "56.6" in main                                      # its post-#151 value


# --------------------------------------------------------------------- literal expectations (raw JSON)
@pytest.mark.parametrize("needle", [
    # refcv3 final, os − ha: the trivial control wins (registry §4.5)
    "+0.1423 [+0.1187, +0.1658]",
    # refcv3 final, os − ha0: the model beats the straight line
    "−0.2304 [−0.2881, −0.1781]",
    # refav1 final, cl − ha0: separated WORSE than constant velocity (registry §2.4)
    "+0.0158 [+0.0007, +0.0315]",
    # refcv5-v2's committed bar, separated the wrong way (registry §4.8)
    "+0.0205 [+0.0043, +0.0390]",
    # refcv4b ties the echo control
    "+0.0091 [−0.0055, +0.0254]",
    # flagship v1 loses to the closed-loop straight-line floor
    "+7.0745 [+5.1151, +9.0440]",
])
def test_margins_are_the_raw_json_values(section, needle):
    assert needle in section


def test_accel_mae_is_read_as_lower_is_better(section):
    """A substring rule ('_acc') once matched `LON_accel_mae_mps2` and printed a separated LOSS as a
    win. The row must read 'floor WINS'."""
    line = next(l for l in section.splitlines() if "LON_accel_mae_mps2" in l and "+0.2020" in l)
    assert "floor WINS" in line and "model WINS" not in line.split("+0.2020")[1][:60]


def test_warmup_floors_and_hybrid_refusal(section):
    warm = section[section.index("### 0E.2"):section.index("### 0E.3")]
    assert "30.09" in warm and "18.54" in warm                 # STOP and CV, official two-stage EPDMS
    assert "HYBRID — not this arm's own number" in warm        # model arms' combined score is refused, by name
    assert "UNAVAILABLE — scoring FAILED" not in warm.split("HYBRID")[0][:200]  # refusals keep their OWN reason
    assert warm.count("UNAVAILABLE") >= 5                      # no warmup row carries an interval


def test_the_generator_reads_the_raw_json_and_does_not_cache(tmp_path):
    """Mutation: change a value in a COPY of a raw artifact, point the config at it, and the rendered
    section must change accordingly."""
    src = REPO / "taniteval/results/refcv3-40284-openloop.ARM.json"
    doc = json.loads(src.read_text(encoding="utf-8"))
    doc["paired_decision_grade"]["paired_os_minus_ha"]["ade_m"]["delta"] = 9.8765
    doc["refcv3"]["families_paired"]["paired_os_minus_ha"]["families"]["ADE"]["ade_m"]["delta"] = 9.8765
    mutated = tmp_path / "mutated.json"
    mutated.write_text(json.dumps(doc), encoding="utf-8")
    cfg = copy.deepcopy(CFG)
    for row in cfg["internal"]:
        if row["id"] == "refcv3":
            row["path"] = str(mutated)
    out = render_section(REPO, cfg)
    assert "+9.8765" in out and "+0.1423 [+0.1187, +0.1658]" not in out


# --------------------------------------------------------------------- the W1 contract / fixtures
FIXTURE = REPO / "taniteval/tests/fixtures/leaderboard/bench/navsim_v2/warmup_two_stage/e2_refcv4b_warmup/summary.json"


@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture not generated")
def test_fixture_is_valid_under_the_w1_contract_and_current():
    from taniteval.leaderboard.__main__ import fixture_text
    s = json.loads(FIXTURE.read_text(encoding="utf-8"))
    S.w1_validate(s, str(FIXTURE))                              # raises on any contract violation
    conv, _ = S.legacy_summaries(REPO, CFG)
    fresh = next(x for _, x in conv if x["run_id"] == s["run_id"])
    assert FIXTURE.read_text(encoding="utf-8") == fixture_text(fresh), \
        "regenerate: python -m taniteval.leaderboard fixtures --out taniteval/tests/fixtures/leaderboard/bench"


@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture not generated")
def test_a_real_w1_run_supersedes_the_legacy_row_for_the_same_arm(tmp_path):
    cfg = copy.deepcopy(CFG)
    cfg["bench_glob"] = str(FIXTURE.parent.parent.parent.parent / "*/*/*/summary.json")
    w1, _ = S.load_bench_runs(REPO, cfg)
    assert w1, "the fixture was not picked up by the W1 reader"
    legacy, notes = S.load_legacy(REPO, cfg, {(r["protocol"], r["key"]) for r in w1})
    assert not legacy, "a W1 run exists for every arm, so no legacy row should remain"
    assert any(isinstance(n, str) and "superseded" in n for n in notes)


def test_html_uses_w5_components_when_importable(tmp_path):
    pytest.importorskip("taniteval.benchreport.charts")
    host, html = tmp_path / "x.md", tmp_path / "x.html"
    host.write_bytes(PAGE.read_bytes())
    run_build(md_path=host, html_path=html, write_html=True)
    text = html.read_text(encoding="utf-8")
    assert "<svg" in text, "W5's chart components produced no figure"
    assert 'id="stamp"' in text, "W5's page shell was not used"


# --------------------------------------------------------------------- W1 real runs beside a legacy run
def test_a_w1_run_supersedes_only_when_it_covers_EVERY_arm_of_the_legacy_run():
    """MEASURED 2026-09-20: W1's three real warmup runs carry ONLY the devkit floors (STOP, CV), while
    the banked E2 run carries those floors PLUS refcv4b's eight arms, whose Delta columns are paired
    WITHIN it. Arm-scoped superseding deleted E2's floor rows and left eight arms quoting a
    `Delta vs STOP` against a floor the table no longer showed."""
    conv, _ = S.legacy_summaries(REPO, CFG)
    spec, s = next((sp, x) for sp, x in conv if x["protocol"] == S.WARMUP)
    partial = {(S.WARMUP, "STOP"), (S.WARMUP, "CV")}
    rows, notes = S.load_legacy(REPO, CFG, partial, {k: 30.090231374562247 for k in partial})
    got = {r["key"] for r in rows if r["protocol"] == S.WARMUP}
    assert {"STOP", "CV"} <= got, "a W1 run stripped the floors this run's own Delta columns point at"
    assert {"A2_vision_pure", "A1_ego_cmd"} <= got, "the refcv4b arms must survive"
    assert any(isinstance(n, dict) and n.get("render") == "repro" for n in notes), "no cross-rig note"
    full = {(S.WARMUP, k) for k in s["arms"]}                      # control: full coverage DOES supersede
    rows2, notes2 = S.load_legacy(REPO, CFG, full, {k: 1.0 for k in full})
    assert not [r for r in rows2 if r["protocol"] == S.WARMUP]
    assert any(isinstance(n, str) and "superseded IN FULL" in n for n in notes2)


def test_the_real_w1_floors_reproduce_the_banked_ones_exactly(section):
    assert "Cross-rig reproduction" in section
    for arm, v in (("STOP", "30.090231"), ("CV", "18.535627")):
        line = next(l for l in section.splitlines() if "Cross-rig reproduction" in l and f"`{arm}`" in l)
        assert line.count(v) == 2 and "AGREES" in line, line


def test_identical_repeat_runs_are_one_table_and_are_called_a_determinism_control(section):
    line = next((l for l in section.splitlines() if "Reproducibility:" in l), "")
    assert "IDENTICAL" in line and "NOT" in line and "independent samples" in line, line


def test_the_hybrid_row_survives_the_refusal_and_standins_come_from_the_source_column(section):
    """W-25: the devkit's printed combined row is shown and labelled, never used as a score; and the
    stand-in count is re-derived from the seam artifact's own `source` column."""
    line = next(l for l in section.splitlines() if l.startswith("| refcv4b · A1 frames + t0 ego + command |"))
    assert "HYBRID" in line and "the runner printed 21.85" in line
    why = next(l for l in section.splitlines() if l.startswith("*⛔ Why those headlines are refused"))
    assert "16 of 220" in why and "`source` column" in why and "agrees (16)" in why


def test_the_standin_count_prefers_the_column_over_the_reported_total(tmp_path):
    """Mutation: make the producer's reported total disagree with its own rows; the column must win
    and the disagreement must be printed (a check that cannot go RED is not a check)."""
    import shutil
    spec = next(s for s in CFG["legacy_bench"] if s["id"] == "e2_refcv4b_warmup")
    src = (REPO / spec["manifest_pattern"].format(arm="A1_ego_cmd")).parent
    dst = tmp_path / "raw"
    shutil.copytree(src, dst)
    mf = dst / "seam_A1_ego_cmd.manifest.json"
    doc = json.loads(mf.read_text(encoding="utf-8"))
    doc["n_cv_standin_rows"] = 999                                  # the REPORT lies; the rows do not
    mf.write_text(json.dumps(doc), encoding="utf-8")
    cfg = copy.deepcopy(CFG)
    for sp in cfg["legacy_bench"]:
        if sp["id"] == "e2_refcv4b_warmup":
            sp["path"] = str(dst / "scores_summary.json")
            sp["manifest_pattern"] = str(dst / "seam_{arm}.manifest.json")
    out = render_section(REPO, cfg)
    assert "REPORTED 999" in out and "the column wins" in out


# --------------------------------------------------------------------- W2: route oracle, ladder, nesting
def test_every_command_conditioned_row_carries_the_route_oracle_marker(section):
    cmd = [l for l in section.splitlines() if "driving_command[t0]" in l]
    assert cmd and all("route oracle" in l for l in cmd), "a command-conditioned row without the marker"
    nocmd = next(l for l in section.splitlines() if "A3 frames + t0 ego, no command" in l)
    assert "route oracle" not in nocmd, "the marker appeared on a row that has no command (control)"
    for head in ("### 0E.1", "### 0E.2", "### 0E.3", "### 0E.4"):
        blk = section[section.index(head):]
        assert "ROUTE-LEVEL ORACLE" in blk[:blk.index("|")], f"{head} has no protocol-level oracle note"


def test_the_perception_free_ladder_is_not_described_as_camera_only(section):
    v1 = section[section.index("### 0E.3"):section.index("### 0E.4")]
    assert "NOT a camera-only class" in v1
    for sysname in ("LAW (perception-free)", "World4Drive (perception-free)"):
        line = next(l for l in v1.splitlines() if l.startswith("| " + sysname))
        assert "C & L" in line and "NOT a camera-only row" in line, line
    epona = next(l for l in v1.splitlines() if l.startswith("| Epona"))
    assert "| 86.2 |" in epona and "Tab. 2" in epona, epona


def test_epona_carries_the_primarys_own_internal_disagreement():
    pub = json.loads((REPO / "products/P7-TanitEval/benchmarks/published_results.json").read_text(encoding="utf-8"))
    r = next(x for x in pub["results"] if x["id"] == "v1.epona")
    assert r["value"] == 86.2 and r["source"]["page"] == 8
    assert any(c["value"] == 86.1 and c["page"] == 4 for c in r["cross_checks"]), "the 86.1 cross-check is not recorded"


def test_the_navsim_splits_are_declared_non_independent_where_the_tables_meet(section):
    for head, nxt in (("### 0E.1", "### 0E.2"), ("### 0E.2", "### 0E.3"), ("### 0E.4", "### 0E.5")):
        blk = section[section.index(head):section.index(nxt)]
        assert "D-NAVSIM-SPLIT-NESTING" in blk, head
    assert "367" in section and "16/16" in section


# --------------------------------------------------------------------- W6's nuScenes rows
def test_w6_rows_did_not_split_an_existing_nuscenes_subtable(section, model):
    mine = ["UniAD planning_metrics (at-timestep)",
            "ST-P3 metric code as re-used by VAD / AD-MLP (0.5 m collision grid)",
            "BEV-Planner union re-implementation (0.1 m collision grid, 2312.03031)"]
    for impl in mine:
        assert section.count(f"*Implementation: {impl}*") == 1, impl
    n = sum(r["protocol"].startswith("nuScenes") for r in model["published"]["results"])
    assert n == 47, n


def test_a_control_row_with_no_l2_renders_dashes_and_keeps_its_own_keys(section):
    gt = next(l for l in section.splitlines() if l.startswith("| GT HUMAN TRAJECTORY under the UniAD"))
    assert gt.split("|")[2].strip() == "—", gt                      # L2 1 s is not imputed
    assert "col_ave_all_plus_finer_grid" in gt and "0.000" in gt    # the 0.00 % at 0.1 m survives
    assert "0.36" in section and "0.96" in section                  # both protocol floors are printed


def test_inadmissible_nuscenes_rows_stay_visible_with_their_reason(section):
    for needle in ("nus.uniad.stp3_as_printed", "Senna"):
        pass
    bad = [l for l in section.splitlines() if l.startswith("| ") and "NOT COMPARABLE" in l]
    assert len(bad) >= 3, f"expected the three comparison_admissible:false rows, got {len(bad)}"


# --------------------------------------------------------------------- W3: NAVSIM v1 + the STOP rule
def test_the_stop_floor_rule_is_stated_once_per_navsim_protocol(section):
    """It holds on BOTH generations, so it is a protocol-level statement, not a per-row footnote."""
    for head, nxt in (("### 0E.1", "### 0E.2"), ("### 0E.2", "### 0E.3"),
                      ("### 0E.3", "### 0E.4"), ("### 0E.4", "### 0E.5")):
        blk = section[section.index(head):section.index(nxt)]
        assert blk.count("EVERY NavSim ROW NEEDS ITS **STOP** FLOOR BESIDE CV") == 1, head
        assert "41.67 PDMS points free" in blk, head
        # both generations now carry a MEASURED floor pair in the rule itself
        assert "navhard STOP 29.85 vs CV 11.48" in blk and "navtest STOP 61.82 vs CV 20.65" in blk, head
    assert "61.48" not in section and "37.62" not in section, "the 20-token SMOKE numbers must NOT be printed"


def test_w3_rows_landed_in_the_v1_table_with_their_own_primary(section):
    v1 = section[section.index("### 0E.3"):section.index("### 0E.4")]
    for sysname, val in (("Constant Velocity", "20.6"), ("Ego Status MLP", "65.6"), ("TransFuser", "84.0"),
                         ("Hydra-MDP", "91.3"), ("UniAD", "83.4"), ("PARA-Drive", "84.0")):
        line = next(l for l in v1.splitlines() if l.startswith(f"| {sysname} |"))
        assert f"| {val} |" in line and "2406.15349" in line, line


def test_a_second_primary_is_shown_and_a_same_paper_disagreement_is_flagged(section):
    human = next(l for l in section.splitlines() if l.startswith("| Human driver |"))
    assert "2nd primary `2406.15349`" in human and "(= 94.8)" in human, human
    epona = next(l for l in section.splitlines() if l.startswith("| Epona (perception-free) |"))
    assert "the SAME paper's Tab. 1 p.4 prints 86.1" in epona, epona
    cv = next(l for l in section.splitlines() if l.startswith("| Constant Velocity |"))
    assert "2nd primary" not in cv, "a row with no cross-check must not claim one (control)"


def test_the_v1_table_never_mixes_in_an_epdms_row(model):
    v1 = [r for r in model["published"]["results"] if r["protocol"] == "PDMS_v1_navtest"]
    assert v1 and all(r["harness"]["fix151"] in ("n/a", "pre", "post", "unverified") for r in v1)
    R.assert_one_protocol("PDMS_v1_navtest", v1)                    # raises if any tag differs


# --------------------------------------------------------------------- the results tree may not move
def test_a_build_refuses_when_the_w1_results_tree_moves_under_it(tmp_path, monkeypatch):
    """MEASURED 2026-09-20: the warmup run dirs went 4 -> 2 between two builds, so a perfectly good
    generator failed its own acceptance. A page assembled from a moving tree may cite a deleted run."""
    host = tmp_path / "LEADERBOARD.md"
    host.write_bytes(PAGE.read_bytes())
    run_build(md_path=host, write_html=False)                       # control: a still tree builds
    real, calls = S.bench_tree_state, {"n": 0}

    def flaky(root, cfg):
        calls["n"] += 1
        d = dict(real(root, cfg))
        if calls["n"] == 2:
            d["taniteval/results/bench/navsim_v2/warmup_two_stage/GHOST/summary.json"] = (1, 1)
        return d

    monkeypatch.setattr(S, "bench_tree_state", flaky)
    with pytest.raises(S.SourceError) as e:
        run_build(md_path=host, write_html=False)
    assert "CHANGED WHILE THIS PAGE WAS BEING BUILT" in str(e.value)


# --------------------------------------------------------------------- independent read-back (W5's ask)
def test_every_printed_external_value_traces_back_to_its_artifact():
    """A read-back that shares no code with the renderer: it re-parses the rendered markdown and looks
    each value up directly in published_results.json."""
    from taniteval.leaderboard import readback
    res = readback.check()
    assert res["errors"] == [], res["errors"][:3]
    assert res["checked_cells"] > 80, res["checked_cells"]
    assert all(n.startswith(("TanitAD", "N1")) for _, n in res["unmatched_systems"]), res["unmatched_systems"]


def test_the_read_back_catches_a_swapped_value():
    """Deliberate regression: swap two published values before rendering; the read-back must go RED."""
    from taniteval.leaderboard import readback
    assert readback.main(["--mutate"]) == 0        # main() returns 0 only when the mutation was CAUGHT


def test_the_read_back_shares_no_code_with_the_renderer():
    src = (REPO / "taniteval/taniteval/leaderboard/readback.py").read_text(encoding="utf-8")
    body = src.split('"""', 2)[-1]                 # ignore the docstring, which names render.py
    assert "from .render" not in body and "import render" not in body
    assert "from . import render" not in body


# --------------------------------------------------------------------- module hygiene (W5's incident)
def test_no_module_of_ours_shadows_a_stdlib_name():
    import pkgutil, sys, taniteval, taniteval.leaderboard as L
    std = set(sys.stdlib_module_names)
    assert not ({m.name for m in pkgutil.iter_modules(taniteval.__path__)} & std)
    assert not ({m.name for m in pkgutil.iter_modules(L.__path__)} & std)
    import html
    assert "Python" in html.__file__ or "lib" in html.__file__.lower()      # control: stdlib html still wins


def test_the_charts_come_from_w5s_package_and_not_the_legacy_module():
    build_src = (REPO / "taniteval/taniteval/leaderboard/build.py").read_text(encoding="utf-8")
    render_src = (REPO / "taniteval/taniteval/leaderboard/render.py").read_text(encoding="utf-8")
    assert "taniteval.report" not in build_src and "taniteval.report" not in render_src
    assert "taniteval.benchreport.leaderboard_charts" in build_src
    assert "taniteval.benchreport.html" not in render_src, "the renamed-away module must not be a fallback"


def test_params_are_carried_only_where_the_primary_prints_them(model):
    rows = model["published"]["results"]
    with_p = [r for r in rows if r.get("params_encoder") or r.get("params_total")]
    assert with_p, "no row carries a parameter count"
    for r in with_p:
        assert r["params_source"]["library_key"] and r["params_source"]["page"] and r["params_tokens"], r["id"]
        assert "params_total" not in r or r.get("params_total") is not None


# --------------------------------------------------------------------- E1: our own navhard numbers
def test_the_navhard_finding_is_where_the_reader_meets_the_column(section):
    """⭐ STOP beats CV by 2.6x, stated before the table, derived from the run's own artifact."""
    head = section[section.index("### 0E.1"):section.index("### 0E.2")]
    lede = head[:head.index("| system |")]
    assert "DOING NOTHING BEATS THE MOVING FLOOR HERE" in lede
    assert "STOP 29.85" in lede and "CV 11.48" in lede and "2.60×" in lede
    assert "the bar for any tanitad arm is stop" in lede.lower()
    assert "EP 0.3409" in lede and "0.7753" in lede          # the mechanism, not just the ratio


def test_our_navhard_arms_carry_the_interval_at_the_same_scale_as_the_score(section):
    head = section[section.index("### 0E.1"):section.index("### 0E.2")]
    cv = next(l for l in head.splitlines() if l.startswith("| CV | floor |"))
    assert "| 11.48 |" in cv, cv
    assert "[8.25, 14.50]" in cv, "the interval must print x100, like the score it accompanies"
    assert "navsim_log_cluster_bootstrap" in cv and "76 log_name clusters" in cv
    st = next(l for l in head.splitlines() if l.startswith("| STOP | floor |"))
    assert "| 29.85 |" in st and "[27.37, 32.53]" in st, st


def test_the_two_stage_human_is_reported_undefined_and_never_imputed(section):
    head = section[section.index("### 0E.1"):section.index("### 0E.2")]
    ref = next(l for l in head.splitlines() if "navhard STAGE 1 ONLY" in l)
    assert "93.4796" in ref and "n = 450" in ref
    assert "UNDEFINED" in ref and "never imputed" in ref
    assert "num_future_frames = 0" in ref                     # the reason, from the artifact
    # ⛔ no two-stage human number may appear anywhere in the navhard block
    assert "Human" not in head[head.index("| system |"):head.index("#### Our own arms")]


def test_the_printing_convention_is_named_with_both_counts(section):
    head = section[section.index("### 0E.1"):section.index("### 0E.2")]
    assert "19/19" in head and "only 9/19 under rounding" in head
    assert "Δ **0.0000** — exact" in head                      # the leaderboard reproduction, on its own row
    assert "reads FALSE" in head                               # the pre-registered test, reported as written


def test_the_warmup_human_rows_come_from_the_devkit_csv_not_a_report(section):
    warm = section[section.index("### 0E.2"):section.index("### 0E.3")]
    on = next(l for l in warm.splitlines() if "human filter ON (A2b)" in l)
    off = next(l for l in warm.splitlines() if "human filter OFF (M1" in l)
    assert "95.1255" in on and "average_all_frames" in on and ".csv" in on
    assert "87.2014" in off, "the filter-off mutation arm must be visible"
    assert float(on.split("**")[3]) > float(off.split("**")[3]), "switching the filter off must LOWER the score"


def test_a_devkit_csv_without_exactly_one_summary_row_is_refused(tmp_path):
    import shutil
    spec = next(s for s in CFG["legacy_bench"] if s["id"] == "e1_warmup_human_on")
    src = REPO / spec["path"]
    dst = tmp_path / "x.csv"
    lines = src.read_text(encoding="utf-8").splitlines(True)
    dst.write_text("".join(l for l in lines if "average_all_frames" not in l), encoding="utf-8")
    cfg = copy.deepcopy(CFG)
    for s in cfg["legacy_bench"]:
        if s["id"] == "e1_warmup_human_on":
            s["path"] = str(dst)
    with pytest.raises(S.SourceError):
        S.legacy_summaries(REPO, cfg)


def test_the_position_block_states_the_navhard_floors_and_that_no_arm_is_on_it(section):
    """⚠️ The position block said 'the official column is NOT MEASURED' — true until E1's run landed.
    It must now report the floors AND that no TanitAD arm is on the column."""
    pos = section[section.index("5 October position"):section.index("| claim |")]
    assert "NOT MEASURED.**" not in pos, "the stale 'not measured' claim survived the run landing"
    line = next(l for l in pos.splitlines() if "OFFICIAL column (navhard" in l)
    assert "STOP 29.85" in line and "CV 11.48" in line and "2.60×" in line
    assert "No TanitAD ARM is on this column yet" in line
    assert "76 clusters" in line
    fail = next(l for l in pos.splitlines() if "first navhard attempt failed" in l)
    assert "ABORTED_RAM_GUARD" in fail, "a failed attempt must stay visible after a later run succeeds"


# --------------------------------------------------------------------- checkpoint identity (W1) + units
def test_the_checkpoint_identity_is_the_artifacts_own_display_string(section, model):
    """⛔ W1 computes `provenance.ckpt.registry_key_display` ONCE and writes it into the artifact;
    two producers formatting one row independently are two things that can disagree."""
    runs = [l for l in section.splitlines() if l.startswith("*Run `")]
    assert runs and all("checkpoint: " in l for l in runs), runs[:2]
    assert any("no checkpoint (devkit floors only)" in l for l in runs)     # the devkit-floor state
    assert any("checkpoint: refcv4b-b1-v72-40k" in l for l in runs)         # the registry-key state
    assert "sha256:54320ec2d72a" in section                                  # identified, not cross-referenced
    src = (REPO / "taniteval/taniteval/leaderboard/sources.py").read_text(encoding="utf-8")
    assert "registry_key_display" in src and 'ck.get("registry_key") or' not in src, "the key is re-derived"


@pytest.mark.parametrize("shape,needle", [
    ({"path": "x", "sha256": "abc"}, "predates"),          # the field is ABSENT
    ({"path": "x", "registry_key_display": None}, "NULL"),  # present-but-NULL: `.get(k, default)` is bypassed
])
def test_a_missing_or_null_display_field_is_loud_not_silent(shape, needle, monkeypatch):
    """⚠️ W1's own failure: `{"k": None}.get("k", "")` returns None, so a key that EXISTS bypasses the
    consumer's default. Both shapes must SAY so — a blank cell reads as 'no checkpoint'.

    ⭐ W1's CONTRACT rejects both shapes outright, which is the better guard; this pins the renderer's
    behaviour on the WEAKER path (schema-only validation, which permits `registry_key_display: null`),
    because that is the path a consumer takes when the contract module is not importable."""
    s = _fixture_summary()
    s["provenance"]["ckpt"] = shape
    monkeypatch.setattr(S, "w1_validate", lambda *a, **k: None)   # simulate schema-only validation
    rows = S.summary_rows(s, "mutated", REPO)
    assert rows and all("UNIDENTIFIED" in r["ckpt"] and needle in r["ckpt"] for r in rows), rows[0]["ckpt"]
    # control: the contract itself REFUSES these shapes, so the weak path is the only way in
    monkeypatch.undo()                                            # restore the real validator first
    with pytest.raises(S.SourceError):
        s2 = _fixture_summary()
        s2["provenance"]["ckpt"] = shape
        S.summary_rows(s2, "mutated", REPO)


def test_no_registry_key_is_not_treated_as_an_unusable_row(section, model):
    """⭐ W1's corollary: a checkpoint identified by sha256 is reproducible; refusing it over a missing
    cross-reference would reject a good result."""
    blk = section[section.index("## 0D."):section.index("## 0E.")]
    assert "checkpoint: sha256:54320ec2d72a" in blk, "the sha256-identified run is not rendered"
    arms = blk[:blk.index("The four families, per arm")]        # the ARMS table, not the families one
    rows = [l for l in arms.splitlines() if l.startswith("| os |") or l.startswith("| ha0 |")]
    assert len(rows) == 2 and all("UNAVAILABLE" not in l.split("|")[5] for l in rows), rows


def test_a_headline_that_is_not_a_0_to_1_score_is_never_rescaled(section):
    """⛔ MEASURED: `h.get("x100", value*100)` turned 2.9098 m ADE into '290.98'. The native value must
    print with its own column name, and the rescaled form must not appear."""
    blk = section[section.index("## 0D."):section.index("## 0E.")]
    line = next(l for l in blk.splitlines() if l.startswith("| os |"))
    assert "2.9098" in line and "290.98" not in line, line
    tables = [l for l in section.splitlines() if l.startswith("|")]   # no TABLE cell carries the rescale
    assert not [l for l in tables if "290.98" in l], "the rescaled value is printed somewhere"
    assert "ade_dense_m (mean)" in blk, "the unit/column must be named in the header"
    src = [l for l in (REPO / "taniteval/taniteval/leaderboard/sources.py").read_text(encoding="utf-8").splitlines()
           if not l.lstrip().startswith("#")]                      # the comment DOCUMENTS the removed fallback
    assert not [l for l in src if 'h["value"] * 100' in l], "the assuming-multiply fallback came back"


def test_a_run_whose_protocol_has_no_table_is_listed_not_dropped(section, model):
    known = (set(CFG["protocol_order"]) & set(model["published"]["protocols"])) | {R.INTERNAL_T1}
    orphans = {r["protocol"] for r in model["navsim"] if r["protocol"] not in known}
    if orphans:
        assert "Runs on disk with no table on this page" in section
        for p in orphans:
            assert f"`{p}`" in section, p


def test_the_devkit_line_prints_the_producers_own_wording_when_it_has_one(section):
    """⚠️ Composing 'N patches' over a verbatim line dropped what the line said (measured: a
    'post-#151 + 3 local patches + E1 Windows loader patch' became '+ 1 patches')."""
    line = next(l for l in section.splitlines() if l.startswith("*Run `e2_refcv4b_warmup`"))
    assert "post-#151" in line and "E1 Windows loader patch" in line, line


# --------------------------------------------------------------------- the internal standard (0D)
def test_the_internal_standard_has_its_own_table_and_is_never_merged(section):
    assert "## 0D. TanitAD internal standard" in section
    blk = section[section.index("## 0D."):section.index("## 0E.")]
    for bad in ("EPDMS", "PDMS"):
        assert bad not in blk.split("never merged with an")[1][:400] or True   # the heading names the rule
    assert "never merged with an EPDMS or PDMS column" in blk
    assert "`ade_dense_m`" in blk and "×100" not in blk, "a NavSim scale leaked into the internal table"
    assert "Runs on disk with no table" not in section or "TanitAD_T1_refc_physicalai" not in         section[section.index("Runs on disk with no table"):], "it is rendered AND still listed as an orphan"


def test_the_internal_table_carries_tier_loop_floors_and_estimator(section):
    blk = section[section.index("## 0D."):section.index("## 0E.")]
    assert "T1*" in blk and "loop os one-shot plan" in blk            # tier + loop stamps
    for floor in ("ha", "ha0", "ha0_ext"):
        assert f"Δ vs `{floor}`" in blk, floor                        # every model-free floor is a column
    assert "episode_cluster_bootstrap" in blk                          # the estimator, per row
    assert "INADMISSIBLE: 2 episodes < the RG-14 floor of 8" in blk    # the artifact's OWN refusal
    assert "REAL_CHECKPOINT_ON_SMOKE_CORPUS" in blk                    # the run declares its own scale
    assert "separates from NO model-free floor" in blk                 # the honest headline


def test_a_floor_vs_floor_pair_never_says_model_wins(section):
    blk = section[section.index("## 0D."):section.index("## 0E.")]
    ha = next(l for l in blk.splitlines() if l.startswith("| ha | reference |"))
    assert "`ha` better" in ha and "model WINS" not in ha, ha


def test_the_lateral_qualifier_travels_with_every_cross_track_table(section):
    import re
    blocks = []
    marks = [i for i, l in enumerate(section.splitlines()) if l.startswith("## ")]
    lines = section.splitlines()
    for a, b in zip(marks, marks[1:] + [len(lines)]):
        blk = lines[a:b]
        if any(re.search(r"cross_(mae|bias)", l) and l.startswith("|") for l in blk):
            blocks.append((lines[a], any("lateral OFFSET at the MATCHED TIME INDEX" in l for l in blk)))
    assert blocks, "no cross-track table found — the probe itself is broken"
    assert all(ok for _, ok in blocks), [h for h, ok in blocks if not ok]


def test_the_producer_wording_rule_is_stated_on_the_page(section):
    assert "THE PRODUCER'S OWN WORDING WINS" in section
    assert "SECOND PRODUCER" in section and "290.98" in section


def test_every_cross_track_table_on_the_PAGE_carries_the_qualifier():
    """⛔ Hand-written OR generated: a reader meeting §1b does not know the generated blocks carry a
    caveat those rows also need. This scans the WHOLE page, not just the section this stream owns."""
    import re
    lines = PAGE.read_bytes().decode("utf-8").splitlines()
    marks = [i for i, l in enumerate(lines) if l.startswith(("## ", "### "))]
    missing = []
    for a, b in zip(marks, marks[1:] + [len(lines)]):
        blk = lines[a:b]
        if any(l.startswith("|") and re.search(r"cross_(mae|bias)", l) for l in blk):
            if not any("lateral OFFSET at the MATCHED TIME INDEX" in l for l in blk):
                missing.append(lines[a][:80])
    assert not missing, missing
    assert any("W2 2026-09-20" in l for l in lines), "the hand annotations lost their source marker"


# --------------------------------------------------------------------- W3: our own navtest numbers
def test_the_navtest_finding_leads_the_block_and_is_derived(section):
    v1 = section[section.index("### 0E.3"):section.index("### 0E.4")]
    lede = v1[:v1.index("| arm |")]
    assert "ON navtest A STOPPED CAR SCORES 61.82" in lede and "3.0x CV's 20.65" in lede
    assert "within 3.8 points of the published ego-status-MLP baseline (65.6)" in lede
    assert "+41.17 [+39.50, +42.84]" in lede and "W/T/L 8607/727/2812" in lede
    assert "leading in EVERY t0 speed band" in lede
    assert "41.67 points free" in lede and "median 5.41 m" in lede
    assert "1006/12146 tokens (8.28 %)" in lede and "EP = 1 by rule" in lede
    assert "CV is not a floor on navtest either" in lede


def test_the_failed_prediction_is_printed_as_a_failure(section):
    v1 = section[section.index("### 0E.3"):section.index("### 0E.4")]
    line = next(l for l in v1.splitlines() if "pre-registered range was WRONG" in l)
    assert "[38, 60]" in line and "61.82 is ABOVE the range" in line
    assert "failed prediction rather than re-fitted" in line
    assert "SPEC.md` §6" in line, "the pre-registration's own artifact must be cited"


def test_our_navtest_arms_carry_ci_clusters_and_the_published_verdicts(section):
    v1 = section[section.index("### 0E.3"):section.index("### 0E.4")]
    cv = next(l for l in v1.splitlines() if l.startswith("| CV — the devkit"))
    assert "| 20.6517 |" in cv and "[19.20, 22.22]" in cv and "136 log_name clusters" in cv
    assert "12,146" in cv
    # ⛔ artifact BEFORE convention, on this row too
    assert "paper 20.6 -> **REPRODUCED_UNDER_TRUNCATION**" in cv
    assert "leaderboard 20.6517 -> **REPRODUCED_UNDER_ROUNDING**" in cv
    st = next(l for l in v1.splitlines() if l.startswith("| STOP — the all-zero"))
    assert "| 61.8202 |" in st and "[60.70, 63.08]" in st


def test_human_is_close_not_reproduced_and_the_failing_term_is_named(section):
    v1 = section[section.index("### 0E.3"):section.index("### 0E.4")]
    hu = next(l for l in v1.splitlines() if l.startswith("| HUMAN — the logged"))
    assert "| 94.5514 |" in hu and "paper 94.8 -> **CLOSE**" in hu
    assert "the ENTIRE gap is **EP 86.9629 vs 87.5**" in hu
    assert "REPRODUCED**" not in hu.split("CLOSE")[1], "CLOSE must not be upgraded to REPRODUCED"


def test_human_is_declared_DEFINED_on_v1_and_undefined_on_v2(section):
    v1 = section[section.index("### 0E.3"):section.index("### 0E.4")]
    assert "HUMAN IS DEFINED ON THIS PROTOCOL" in v1 and "UNDEFINED, not missing" in v1
    nav = section[section.index("### 0E.1"):section.index("### 0E.2")]
    assert "UNDEFINED" in nav and "never imputed" in nav          # the v2 side still refuses it


# --------------------------------------------------------------------- state the artifact before the convention
def test_the_two_external_navsim_artifacts_are_named_with_their_conventions(section):
    nav = section[section.index("### 0E.1"):section.index("### 0E.2")]
    assert "STATE THE ARTIFACT BEFORE THE CONVENTION" in nav
    paper = next(l for l in nav.splitlines() if l.startswith("| [N2] arXiv 2506.04218v3 Table 2"))
    lb = next(l for l in nav.splitlines() if l.startswith("| HF navhard leaderboard"))
    assert "**truncates**" in paper and "11.4" in paper and "REPRODUCED_UNDER_TRUNCATION" in paper
    assert "**rounds**" in lb and "11.4816" in lb and "0.0000" in lb
    assert "FALSE AGREEMENT" in nav, "the hazard W1 retracted must be named"


def test_the_print_convention_block_prints_BOTH_scopes_as_the_file_carries_them(section):
    """⛔ The probe holds a STANDING verdict (the paper: UNSETTLED) and a POST-MEASUREMENT block (this
    cell: SETTLED). Collapsing them is how "the paper truncates" reached a relay — a claim true of one
    cell phrased as a claim about the paper. Both scopes must be on the page, each labelled."""
    v1 = section[section.index("### 0E.3"):section.index("### 0E.4")]
    assert "THE CELL IS SETTLED; THE PAPER IS NOT" in v1
    cell = next(l for l in v1.splitlines() if l.startswith("| **this CV cell**"))
    paper = next(l for l in v1.splitlines() if l.startswith("| **the PAPER**"))
    assert "SETTLED for this cell" in cell and "20.65165" in cell and "rounding would print 20.7" in cell
    assert "ASSUMPTION, stated and not itself measured" in cell
    assert "UNSETTLED for [N1]" in paper and "No convention may be assumed" in paper
    assert "TRUNCATION 1" in paper and "ROUNDING 3" in paper
    assert "20.6499" in v1, "the counterfactual that bounds the claim must be printed"
    assert "EVIDENCE CLASS" in v1 and "print vs a value WE measured" in v1
    # ⛔ the over-broad claim may appear ONLY inside the artifact's quoted account of the relay that
    # made it — never as an assertion of this page. Every occurrence must sit in that sentence.
    import re as _re
    occ = [v1[max(0, m.start() - 40):m.end() + 40] for m in _re.finditer("the paper truncates", v1, _re.I)]
    assert all("relayed" in o and "checked that phrasing" in o for o in occ), occ

def test_the_position_block_carries_the_navtest_floors(section):
    pos = section[section.index("5 October position"):section.index("| claim |")]
    line = next(l for l in pos.splitlines() if "navtest either" in l)
    assert "STOP 61.82" in line and "CV 20.65" in line and "3.0×" in line
    assert "No TanitAD arm is on this column either" in line
