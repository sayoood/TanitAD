"""Falsifiers for registry_lint.

The tool exists because of one measured incident (RETRACTION_LOG 07-25, C4): a
retracted headline stood in a MODEL_REGISTRY section header for four days, and a
second instance escaped a line-based grep by wrapping across a newline. So the
red cases here are that incident, plus the transcription-drift class the pointer
mechanism covers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import registry_lint as rl  # noqa: E402


RETRACTIONS = """# RETRACTION LOG

| date | retracted claim | class | cost |
|---|---|---|---|
| 07-25 | *"flagship-v1.6 = the best ADE in the program"* | C4 | four days in a header |
| 07-21 | *"REF-C-XL finishes 0.006 m behind flagship v1"* | C6 | split-mean artefact |

## Standing consequences
- C1 means only eval output is quotable.
"""


@pytest.fixture()
def env(tmp_path: Path) -> Path:
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "eval.json").write_text(
        json.dumps({"headline": {"ade_0_2s": {"mean": 0.4271, "lo": 0.3675}},
                    "arms": [{"name": "v1", "ade": 0.4375}]}),
        encoding="utf-8")
    (tmp_path / "RETRACTION_LOG.md").write_text(RETRACTIONS, encoding="utf-8")
    return tmp_path


def _doc(env: Path, body: str, name: str = "REG.md") -> Path:
    p = env / name
    p.write_text(body, encoding="utf-8")
    return p


def _lint(env: Path, doc: Path, **kw) -> rl.LintReport:
    return rl.lint(env, [doc], env / "RETRACTION_LOG.md", kw.pop("sidecar", None),
                   kw.pop("shingle", 4), kw.pop("context", 25),
                   kw.pop("gap", 1), kw.pop("rare_max", 0))


def _codes(rep: rl.LintReport) -> tuple[int, int]:
    return (sum(f.severity == "ERROR" for f in rep.findings),
            sum(f.severity == "WARN" for f in rep.findings))


# ------------------------------------------------------------------- CHECK 1


def test_pointer_agreeing_with_json_is_clean(env):
    doc = _doc(env, "# R\n\n<!-- src: results/eval.json#headline.ade_0_2s.mean -->\n"
                    "| v1 | 0.4271 |\n")
    assert _codes(_lint(env, doc)) == (0, 0)


def test_pointer_drift_fails(env):
    doc = _doc(env, "# R\n\n<!-- src: results/eval.json#headline.ade_0_2s.mean -->\n"
                    "| v1 | 0.4420 |\n")
    rep = _lint(env, doc)
    assert _codes(rep)[0] == 1
    assert "DRIFT" in rep.findings[0].message


def test_rounding_is_tolerated_in_both_directions(env):
    """The registry legitimately quotes more digits than the JSON stores (0.43746
    vs a stored 0.4375) and fewer (0.427 vs 0.4271). Both are honest."""
    for quoted in ("0.427", "0.4271", "0.42710"):
        doc = _doc(env, f"<!-- src: results/eval.json#headline.ade_0_2s.mean -->\n"
                        f"| v1 | {quoted} |\n", name=f"r{quoted}.md")
        assert _codes(_lint(env, doc)) == (0, 0), quoted
    (env / "results" / "coarse.json").write_text(
        json.dumps({"m": 0.4375}), encoding="utf-8")
    doc = _doc(env, "<!-- src: results/coarse.json#m -->\n| v1.6 | 0.43746 |\n",
               name="coarse.md")
    assert _codes(_lint(env, doc)) == (0, 0)


def test_near_disambiguates_a_row_with_several_numbers(env):
    doc = _doc(env, "<!-- src: results/eval.json#headline.ade_0_2s.mean "
                    "near=\"full-set\" -->\n"
                    "| v1 | 0.4522 +- 0.0312 (full-set 0.4271) |\n")
    assert _codes(_lint(env, doc)) == (0, 0)
    bad = _doc(env, "<!-- src: results/eval.json#headline.ade_0_2s.mean "
                    "near=\"full-set\" -->\n"
                    "| v1 | 0.4271 +- 0.0312 (full-set 0.9999) |\n", name="b.md")
    assert _codes(_lint(env, bad))[0] == 1, "near= must ignore the pre-marker number"


def test_dangling_pointer_is_a_finding(env):
    for spec in ("results/missing.json#headline.ade_0_2s.mean",
                 "results/eval.json#headline.nope.mean",
                 "results/eval.json#headline"):
        doc = _doc(env, f"<!-- src: {spec} -->\n| v1 | 0.4271 |\n",
                   name=f"d{abs(hash(spec))}.md")
        assert _codes(_lint(env, doc))[0] == 1, spec


def test_list_index_field_path(env):
    doc = _doc(env, "<!-- src: results/eval.json#arms[0].ade -->\n| v1 | 0.4375 |\n")
    assert _codes(_lint(env, doc)) == (0, 0)


def test_pointer_skips_a_table_header_to_the_first_numeric_row(env):
    doc = _doc(env, "<!-- src: results/eval.json#headline.ade_0_2s.mean -->\n"
                    "\n| arm | ADE |\n|---|---|\n| v1 | 0.4271 |\n")
    assert _codes(_lint(env, doc)) == (0, 0)


def test_sidecar_anchor_pointer(env):
    side = env / "ptr.jsonl"
    side.write_text(json.dumps({
        "anchor": "^\\| Flagship v1 ", "src": "results/eval.json",
        "field": "headline.ade_0_2s.mean"}) + "\n", encoding="utf-8")
    doc = _doc(env, "# R\n\n| Flagship v1 | 0.4271 |\n")
    assert _codes(_lint(env, doc, sidecar=side)) == (0, 0)
    bad = _doc(env, "# R\n\n| Flagship v1 | 0.4420 |\n", name="b2.md")
    assert _codes(_lint(env, bad, sidecar=side))[0] == 1


def test_ambiguous_sidecar_anchor_is_itself_a_finding(env):
    """A pointer that silently relocates is worse than one that is missing."""
    side = env / "ptr.jsonl"
    side.write_text(json.dumps({
        "anchor": "Flagship", "src": "results/eval.json",
        "field": "headline.ade_0_2s.mean"}) + "\n", encoding="utf-8")
    doc = _doc(env, "| Flagship v1 | 0.4271 |\n| Flagship v2 | 0.4271 |\n")
    rep = _lint(env, doc, sidecar=side)
    assert _codes(rep)[0] == 1
    assert "matched 2 line(s)" in rep.findings[0].message


def test_number_grouping_forms(env):
    assert [q.value for q in rl.numbers_in("29 999 steps")] == [29999]
    assert [q.value for q in rl.numbers_in("286,339,251 params")] == [286339251]
    vals = [q.value for q in rl.numbers_in("0.4522 0.0312")]
    assert vals == [0.4522, 0.0312], "grouping must not glue two decimals together"


def test_tolerance_scales_with_written_precision():
    assert rl.Quoted.parse("0.452", 0).tol == pytest.approx(5e-4)
    assert rl.Quoted.parse("0.4522", 0).tol == pytest.approx(5e-5)
    assert rl.Quoted.parse("29 999", 0).tol == pytest.approx(0.5)


# ------------------------------------------------------------------- CHECK 2


def test_retracted_claim_in_a_header_is_an_error(env):
    doc = _doc(env, "# R\n\n### 1.4b flagship-v1.6 -- best ADE in the program\n\nbody\n")
    rep = _lint(env, doc)
    assert _codes(rep)[0] == 1
    assert "SECTION HEADER" in rep.findings[0].message


def test_the_claim_sweep_is_multiline(env):
    """THE falsifier. A previous instance wrapped across a newline and walked
    straight through a line-based grep; no single line contains the phrase."""
    doc = _doc(env, "# R\n\n### 1.4b flagship-v1.6 -- the best\nADE in the program\n")
    rep = _lint(env, doc)
    errs = [f for f in rep.findings if f.severity == "ERROR"]
    assert errs, "a wrapped claim must still be found"
    assert "WRAPPED ACROSS A NEWLINE" in errs[0].message
    body = doc.read_text(encoding="utf-8")
    assert not any("best ADE in the program" in ln for ln in body.splitlines()), \
        "the fixture must not be catchable by a line-based grep"


def test_one_inserted_word_does_not_hide_a_claim(env):
    """The exact 07-21..25 asymmetry. RETRACTION_LOG's 07-21 entry quotes the
    claim as *"v1.6 ADE 0.4420 - best in the program"*; the header that then
    survived four days said "best **ADE** in the program". One inserted word is
    all the difference there ever was, so gap=0 reproduces the four-day MISS and
    gap=1 is what turns the tool into a detector."""
    log = env / "LOG_0721.md"
    log.write_text("| date | retracted claim | class | cost |\n|---|---|---|---|\n"
                   "| 07-21 | *\"v1.6 ADE 0.4420 - best in the program\"* | C1 | "
                   "trainer in-loop val |\n", encoding="utf-8")
    hdr = _doc(env, "### 1.4b flagship-v1.6 -- best ADE in the program\n",
               name="h.md")
    caught = rl.lint(env, [hdr], log, None, 4, 25, 1, 0)
    missed = rl.lint(env, [hdr], log, None, 4, 25, 0, 0)
    assert _codes(caught)[0] >= 1, "gap=1 must catch the real stale header"
    assert _codes(missed) == (0, 0), "gap=0 must reproduce the historical miss"


def test_body_prose_is_a_warning_not_an_error(env):
    doc = _doc(env, "# R\n\nSomebody wrote best ADE in the program here.\n")
    e, w = _codes(_lint(env, doc))
    assert (e, w) == (0, 1)


def test_documenting_a_retraction_is_not_a_finding(env):
    """The registry must be able to SAY what it retracted without tripping its
    own linter -- otherwise the tool punishes the correction."""
    doc = _doc(env, "# R\n\n### 1.4b -- TIED with v1\n\n> RETRACTED: this header "
                    "read \"best ADE in the program\" for four days.\n")
    assert _codes(_lint(env, doc)) == (0, 0)


def test_lint_ok_marker_exempts_a_line(env):
    doc = _doc(env, "# R\n\n<!-- lint-ok: quoting the retraction verbatim -->\n"
                    "### 1.4b -- best ADE in the program\n")
    assert _codes(_lint(env, doc)) == (0, 0)


def test_boilerplate_header_is_downgraded_by_the_rarity_guard(env):
    """MEASURED on the real registry: '### 4.4 REF-C CLOSED-LOOP ...' matches the
    retracted 'flagship v1 beats REF-C closed-loop' on house vocabulary alone."""
    log = env / "RETRACTION_LOG.md"
    log.write_text(RETRACTIONS + "\n| 07-23 | *\"flagship v1 beats REF-C "
                                 "closed-loop\"* | C5 | n=1 |\n", encoding="utf-8")
    doc = _doc(env, "### 4.4 REF-C CLOSED-LOOP -- AlpaSim suite\n\n"
                    + "REF-C closed-loop is a phrase. " * 30)
    strict = rl.lint(env, [doc], log, None, 4, 25, 1, 0)
    loose = rl.lint(env, [doc], log, None, 4, 25, 1, 25)
    assert _codes(strict)[0] >= 1, "rare_max=0 must report it"
    assert _codes(loose)[0] == 0, "the rarity guard must demote house vocabulary"


def test_all_stopword_phrases_never_match(env):
    log = env / "RETRACTION_LOG.md"
    log.write_text("| date | claim | c | x |\n|---|---|---|---|\n"
                   "| 07-25 | *\"it was in the of and the\"* | C1 | none |\n",
                   encoding="utf-8")
    doc = _doc(env, "### something it was in the of and the entirely unrelated\n")
    assert _codes(rl.lint(env, [doc], log, None, 4, 25, 1, 0)) == (0, 0)


def test_claims_are_read_only_from_the_claim_cell(env):
    """The cost column quotes the CORRECTION; mining it would flag every corrected
    number as a retracted claim."""
    claims = rl.load_retracted_claims(env / "RETRACTION_LOG.md")
    assert any("best ADE in the program" in c for c in claims)
    assert not any("only eval output is quotable" in c for c in claims)
    assert len(claims) == 2


# --------------------------------------------------------------------- driver


def test_self_test_is_green(tmp_path):
    text, rc = rl.self_test(tmp_path / "st")
    assert rc == 0, text
    assert "SELF-TEST: PASS" in text


def test_render_output_is_ascii(env):
    doc = _doc(env, "# R\n\n### 1.4b — best ADE in the program ⚠️\n")
    text, rc = rl.render(_lint(env, doc), strict=False)
    assert rc == 1
    text.encode("ascii")            # the cp1252 console lesson, as an assertion


def test_strict_makes_warnings_fatal(env):
    doc = _doc(env, "# R\n\nSomebody wrote best ADE in the program here.\n")
    rep = _lint(env, doc)
    assert rl.render(rep, strict=False)[1] == 0
    assert rl.render(rep, strict=True)[1] == 1


def test_missing_file_exits_2(tmp_path):
    assert rl.main(["--repo", str(tmp_path), "--file", "nope.md"]) == 2


def test_seeded_sidecar_pointers_resolve_against_the_live_repo():
    """The seeded pointers must actually AGREE with the committed eval JSON --
    a linter whose own seeds are stale teaches nobody anything."""
    repo = Path(__file__).resolve().parents[2]
    reg = repo / "Project Steering" / "MODEL_REGISTRY.md"
    side = repo / "tools" / "registry_pointers.jsonl"
    if not reg.is_file() or not side.is_file():
        pytest.skip("registry or sidecar not present in this worktree")
    rep = rl.lint(repo, [reg], repo / "Project Steering" / "RETRACTION_LOG.md",
                  side, 4, 25, 1, 25)
    assert rep.n_pointers >= 5, "the sidecar seeds must all bind"
    bad = [f for f in rep.findings if f.kind in ("pointer", "pointer-error")]
    assert not bad, "\n".join(f"{f.file}:{f.line} {f.message}" for f in bad)
