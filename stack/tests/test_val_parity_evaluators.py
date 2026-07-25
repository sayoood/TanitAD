"""VAL-side parity integrity for the EVALUATORS — the mirror of Wave-1 B.

Wave-1 B (``test_parity_manifest.py``) closed the TRAIN side: 15 trainers, 1 of
which had a check, and that check could not see truncation. Its §7 item 2 left
the val side open:

    "13 evaluators glob a val cache with no integrity check. A truncated *val*
     cache silently changes every published ADE the same way a truncated train
     cache changes every arm. This needs an owner."

This file is that owner's regression suite. It pins TWO premises about the
pre-fix code, permanently, so a future refactor that reintroduces either fails
here rather than in a published table:

  PREMISE PIN 1 — ``sorted(Path(cd).glob("*val*"))[-1]``, the val-dir resolver
    10 of these evaluators inherited from the trainers, is a LEXICOGRAPHIC max.
    ``physicalai-val-0c5f7dac3b11`` sorts BEFORE ``physicalai-val-f1b378f295ae``
    (``'0' < 'f'``), so it SELECTED the 78.5 %-leaked split whenever both were
    materialised under one epcache root.
  PREMISE PIN 2 — the episode list was ``glob("ep_*.pt")[:n]``, which returns
    fewer than ``n`` from a truncated cache and says nothing.

No GPU, no pod: every cache is a temp dir of empty ``ep_*.pt`` markers, which
also proves the guard fires BEFORE any tensor is unpickled.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tanitad.data import parity

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
CLEAN = parity.PARITY_VAL_KEY
LEAKY = next(iter(parity.LEAKY_SPLIT_KEYS))
CANONICAL = 40          # the TanitEval deployment -> 881 published windows
FULL = 600              # the epcache build on the training pods


def _cache(root: Path, key: str, n: int, start: int = 0) -> Path:
    d = root / key
    d.mkdir(parents=True, exist_ok=True)
    for i in range(start, start + n):
        (d / f"ep_{i:05d}.pt").touch()
    return d


# --------------------------------------------------------------------------- #
# 1. the premise pins                                                           #
# --------------------------------------------------------------------------- #
def test_the_legacy_resolver_would_have_SELECTED_the_leaky_split(tmp_path):
    """PREMISE PIN 1 — the finding that makes the val side urgent, not merely
    incomplete: the inherited resolver did not fail to check the split, it
    actively PREFERRED the leaked one."""
    _cache(tmp_path, CLEAN, CANONICAL)
    _cache(tmp_path, LEAKY, 79)
    assert sorted(tmp_path.glob("*val*"))[-1].name == LEAKY
    assert parity.resolve_val_dir(tmp_path, label="t").name == CLEAN


def test_the_old_slice_would_have_PASSED_a_truncated_val_cache(tmp_path):
    """PREMISE PIN 2 — ``[:40]`` of a 12-episode cache is 12 episodes, scored
    and published as if it were the 40-episode / 881-window statistic."""
    d = _cache(tmp_path, CLEAN, 12)
    assert len(sorted(d.glob("ep_*.pt"))[:CANONICAL]) == 12
    with pytest.raises(SystemExit):
        parity.assert_val_cache(d, label="--val-cache", requested=CANONICAL)


# --------------------------------------------------------------------------- #
# 2. the shared val guard                                                       #
# --------------------------------------------------------------------------- #
def test_registered_deployments_pass(tmp_path):
    for n in (CANONICAL, FULL):
        rec = parity.assert_val_cache(_cache(tmp_path / str(n), CLEAN, n),
                                      label="v", requested=min(n, CANONICAL))
        assert rec["parity"] is True and rec["episodes_present"] == n


@pytest.mark.parametrize("n", [1, 11, 12, 39, 41, 599, 601])
def test_unregistered_episode_counts_are_refused(tmp_path, n):
    """Any size that is not a REGISTERED deployment is either a truncation or a
    substitution. Both void the number; neither used to be visible."""
    with pytest.raises(SystemExit):
        parity.assert_val_cache(_cache(tmp_path / str(n), CLEAN, n), label="v")


def test_leaky_split_is_refused_by_the_val_guard(tmp_path):
    with pytest.raises(SystemExit) as ei:
        parity.assert_val_cache(_cache(tmp_path, LEAKY, 79), label="v")
    assert "LEAKED SPLIT" in str(ei.value) and CLEAN in str(ei.value)


def test_leaky_only_root_is_refused_by_the_resolver(tmp_path):
    _cache(tmp_path, LEAKY, 79)
    with pytest.raises(SystemExit):
        parity.resolve_val_dir(tmp_path, label="--cache-dirs")


def test_missing_val_dir_stays_an_AssertionError_not_a_refusal(tmp_path):
    """WAVE1_B_REPORT §5, re-applied: several evaluators ``except
    AssertionError`` around an optional val block. Turning "no val dir" into a
    SystemExit would kill a finished run at its metrics write."""
    with pytest.raises(AssertionError):
        parity.resolve_val_dir(tmp_path, label="--cache-dirs")


def test_non_parity_corpora_warn_but_never_block(tmp_path, capsys):
    d = _cache(tmp_path, "comma2k19-val-76b6e94a97a1", 64)
    rec = parity.assert_val_cache(d, label="--cache-dirs", requested=40)
    assert rec["parity"] is False and rec["checked"] is True
    assert "NON-PARITY" in capsys.readouterr().out


def test_empty_val_dir_does_not_raise_from_the_guard(tmp_path):
    d = tmp_path / CLEAN
    d.mkdir()
    rec = parity.assert_val_cache(d, label="v", requested=40)
    assert rec["checked"] is False


def test_note_leaky_audit_discloses_and_never_licenses_a_number(tmp_path,
                                                                capsys):
    rec = parity.note_leaky_audit(_cache(tmp_path, LEAKY, 79),
                                  label="--val", why="label audit")
    assert rec["leaky"] is True and rec["decision_grade"] is False
    assert "KNOWN-LEAKY" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# 3. the manifest's val entry                                                   #
# --------------------------------------------------------------------------- #
def test_val_deployments_are_registered_data_with_citations():
    """The admissible counts must be DATA with evidence, not a magic number in
    code — otherwise "40 is fine" becomes folklore the way "REF-B v2 died at
    22 600" did."""
    deps = parity.val_deployments()
    assert {d["n_episodes"] for d in deps} == {FULL, CANONICAL}
    for d in deps:
        assert d["evidence"] and d["where"]
        assert d["evidence_class"] == "MEASURED"
    ent = parity.manifest_entry(CLEAN)
    # ...and still NO invented uid hash (the Wave-1 B honesty invariant)
    assert ent["episode_uid_sha256"] is None
    bad = ent["deployments_seen_but_NOT_admissible"]
    assert any(x["n_episodes"] == 12 for x in bad), (
        "the 12-episode pod1 partial must stay documented as inadmissible")


def test_uid_digest_is_enforced_the_moment_one_is_recorded(tmp_path):
    """Count alone cannot see a SUBSTITUTED set of the right size. The moment
    ``make_parity_manifest.py --record --split val`` lands a digest, the guard
    must use it with no further code change."""
    good = _cache(tmp_path / "g", CLEAN, CANONICAL)
    swap = _cache(tmp_path / "s", CLEAN, CANONICAL, start=900)
    uids = parity.scan_cache_dir(good)
    ent = parity.build_entry(uids, corpus_key=CLEAN, split="val")
    ent["known_deployments"] = [{"n_episodes": CANONICAL, "role": "t",
                                 "where": "t", "evidence": "t",
                                 "evidence_class": "MEASURED"}]
    mp = tmp_path / "m.json"
    mp.write_text(json.dumps({"schema": parity.MANIFEST_SCHEMA,
                              "corpora": {CLEAN: ent}}), encoding="utf-8")
    assert "sha256" in parity.assert_val_cache(
        good, label="v", manifest_path=mp)["content_check"]
    with pytest.raises(SystemExit) as ei:
        parity.assert_val_cache(swap, label="v", manifest_path=mp)
    assert "MISMATCH" in str(ei.value)


# --------------------------------------------------------------------------- #
# 4. per-evaluator wiring — RED before the wiring, GREEN after                  #
# --------------------------------------------------------------------------- #
#: every evaluator that reads a val cache. Wave-1 B §1 listed 13 as out of
#: scope; ``eval_behavior`` and ``refc_v12_eval`` complete the surface.
EVALUATORS = [
    "evaluate_checkpoint.py", "compare_arms.py", "driving_diagnostic.py",
    "d1_probe_capacity.py", "d3_decompose.py", "run_spectral.py",
    "geom_sanity.py", "resolution_probe.py", "eval_grounded_rollout_4b.py",
    "eval_metric_rollout.py", "eval_behavior.py", "eval_flagship_v4.py",
    "eval_flagship_v15.py", "eval_flagship_v16.py",
]

#: the exact inherited resolver, in every spelling it appears in.
_LEGACY_RESOLVER = re.compile(
    r'sorted\(\s*(?:Path\()?[^\n]*\.glob\(\s*["\']\*val\*["\']\s*\)\s*\)?\s*'
    r'\)\s*\[\s*-1\s*\]')


@pytest.mark.parametrize("fn", EVALUATORS)
def test_evaluator_asserts_val_integrity_before_loading(fn):
    src = (SCRIPTS / fn).read_text(encoding="utf-8")
    assert "parity." in src, (
        f"{fn} reads a val cache with no call into the shared guard "
        f"(tanitad.data.parity)")
    assert ("assert_val_cache" in src or "resolve_val_dir" in src), (
        f"{fn} imports the guard but never asserts the val cache")


@pytest.mark.parametrize("fn", EVALUATORS)
def test_no_evaluator_still_uses_the_leaky_selecting_resolver(fn):
    """The pin that matters most: a code-level assertion that PREMISE PIN 1's
    pattern is gone from every evaluator. Comments describing it are stripped
    first, so documenting the bug does not fail the test."""
    src = (SCRIPTS / fn).read_text(encoding="utf-8")
    code = "\n".join(ln for ln in src.splitlines()
                     if not ln.lstrip().startswith("#"))
    hit = _LEGACY_RESOLVER.search(code)
    assert hit is None, (
        f"{fn} still resolves its val dir with {hit.group(0)!r} — that "
        f"selects {LEAKY} over {CLEAN}")


#: probes whose ``--val-cache`` DEFAULTED to the leaky split. Found by probing a
#: SECOND path and a SECOND argument name after the 13 evaluators were done —
#: "absence found at one location is not absence".
LEAKY_DEFAULT_PROBES = ["run_branchb_transfer.py", "run_idm_parity_validation.py",
                        "run_v1_encoder_char.py", "run_idm_proof.py",
                        "run_idm_ft.py"]


@pytest.mark.parametrize("fn", LEAKY_DEFAULT_PROBES)
def test_leaky_default_probes_now_require_an_explicit_opt_in(fn):
    src = (SCRIPTS / fn).read_text(encoding="utf-8")
    assert "--allow-leaky-val" in src, (
        f"{fn} points --val-cache at {LEAKY} with no explicit opt-in")
    assert "assert_val_cache" in src, (
        f"{fn} does not refuse the leaky split when the opt-in is absent")


#: label/VLM audits: reading the leaky split IS the sanctioned use. They must
#: disclose it, not refuse it.
LEAKY_AUDITS = ["route_label_audit.py", "vlm_route_labels.py",
                "vlm_kin_crossval.py"]


@pytest.mark.parametrize("fn", LEAKY_AUDITS)
def test_leaky_label_audits_disclose_instead_of_running_silently(fn):
    src = (SCRIPTS / fn).read_text(encoding="utf-8")
    assert "note_leaky_audit" in src, (
        f"{fn} reads the leaky split silently — it must self-label")


# --------------------------------------------------------------------------- #
# 5. one FUNCTIONAL end-to-end refusal through a real evaluator                 #
# --------------------------------------------------------------------------- #
def test_geom_sanity_loader_refuses_a_truncated_cache(tmp_path):
    """``geom_sanity.load_val_episodes`` is the one val loader that is a
    standalone function rather than inline in ``main()``, so it can be driven
    directly. Empty ``ep_*.pt`` markers prove the refusal happens BEFORE
    ``load_episode`` — an unguarded run would die inside ``torch.load``."""
    import sys
    sys.path.insert(0, str(SCRIPTS))
    import geom_sanity
    _cache(tmp_path, CLEAN, 12)
    with pytest.raises(SystemExit) as ei:
        geom_sanity.load_val_episodes(str(tmp_path), 40)
    assert "NOT a registered val deployment" in str(ei.value)


def test_geom_sanity_loader_refuses_a_leaky_only_root(tmp_path):
    import sys
    sys.path.insert(0, str(SCRIPTS))
    import geom_sanity
    _cache(tmp_path, LEAKY, 79)
    with pytest.raises(SystemExit) as ei:
        geom_sanity.load_val_episodes(str(tmp_path), 40)
    assert "LEAKED SPLIT" in str(ei.value)


def test_geom_sanity_loader_picks_the_clean_split_over_the_leaky_one(tmp_path):
    """The end-to-end version of PREMISE PIN 1: with BOTH splits present the
    evaluator now reads the clean 40, where it used to read the leaked 79."""
    import sys
    sys.path.insert(0, str(SCRIPTS))
    import geom_sanity
    _cache(tmp_path, CLEAN, CANONICAL)
    _cache(tmp_path, LEAKY, 79)
    with pytest.raises(RuntimeError) as ei:      # dies in torch.load, not the guard
        geom_sanity.load_val_episodes(str(tmp_path), 40)
    assert "mmap" in str(ei.value) or "torch.save" in str(ei.value), (
        "the guard passed the CLEAN split through, as intended")
