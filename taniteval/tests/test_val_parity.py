"""VAL-side parity integrity — the mirror image of Wave-1 B, on the side that
touches PUBLISHED ADEs.

Wave-1 B closed the train side: a 1 200-of-2 376 cache in a correctly-named
directory used to train silently. The val side is worse, because a wrong val
cache does not void a run — it produces a **plausible-looking wrong ADE**, and
nothing downstream can tell. Two concrete holes are pinned here permanently:

  * ``PREMISE PIN 1`` — ``sorted(Path(val).glob("ep_*.pt"))[:40]`` returns 12
    files from a 12-episode deployment and the harness scores them. The number
    is then published as if it were the canonical 40-episode / 881-window
    statistic (``test_the_old_glob_would_have_PASSED_a_truncated_val_cache``).
  * ``PREMISE PIN 2`` — the ``sorted(glob("*val*"))[-1]`` "newest dir wins"
    convention that 10 evaluators inherited from the trainers selects the
    **LEAKY** split: lexicographically ``physicalai-val-0c5f7dac3b11`` <
    ``physicalai-val-f1b378f295ae``
    (``test_the_legacy_resolver_would_have_SELECTED_the_leaky_split``).

Both pins assert the OLD behaviour on purpose, so a future refactor that
reintroduces either pattern fails here rather than in a published table.

No GPU, no pod, no network: every cache is a temp dir of empty ``ep_*.pt``
markers, which also proves the guard runs BEFORE any tensor is unpickled.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tanitad.data import parity
from taniteval import data as tdata

CLEAN = parity.PARITY_VAL_KEY          # physicalai-val-0c5f7dac3b11
LEAKY = next(iter(parity.LEAKY_SPLIT_KEYS))   # physicalai-val-f1b378f295ae
CANONICAL_EPISODES = 40                # -> the 881 stride-8 published windows
FULL_BUILD = 600


# --------------------------------------------------------------------------- #
# fixtures                                                                      #
# --------------------------------------------------------------------------- #
def _cache(root: Path, key: str, n: int, start: int = 0) -> Path:
    """A val split dir of ``n`` empty ``ep_%05d.pt`` markers."""
    d = root / key
    d.mkdir(parents=True, exist_ok=True)
    for i in range(start, start + n):
        (d / f"ep_{i:05d}.pt").touch()
    return d


@pytest.fixture
def clean40(tmp_path):
    return _cache(tmp_path, CLEAN, CANONICAL_EPISODES)


@pytest.fixture
def truncated12(tmp_path):
    """The 12-episode partial deployment that really exists on pod1 and really
    blocked a decision-grade run (P1_DECISION_GRADE_FINDINGS.md)."""
    return _cache(tmp_path, CLEAN, 12)


# --------------------------------------------------------------------------- #
# 1. the premise pins — what the code did BEFORE this workstream                #
# --------------------------------------------------------------------------- #
def test_the_old_glob_would_have_PASSED_a_truncated_val_cache(truncated12):
    """PREMISE PIN 1. The entire pre-fix val-side enforcement was
    ``sorted(glob("ep_*.pt"))[:n]`` — no count, no identity, no complaint.

    An evaluator asking for 40 episodes got 12 and scored them. The resulting
    ADE is a real number over a different benchmark, and it would have been
    published next to 881-window numbers with nothing to distinguish it."""
    files = sorted(truncated12.glob("ep_*.pt"))[:CANONICAL_EPISODES]
    assert len(files) == 12                       # asked 40, silently got 12
    assert CLEAN in str(truncated12)              # ...from a correctly-named dir


def test_the_legacy_resolver_would_have_SELECTED_the_leaky_split(tmp_path):
    """PREMISE PIN 2 — the finding that makes this workstream urgent.

    ``sorted(Path(root).glob("*val*"))[-1]`` is the val-dir resolver 10
    ``stack/scripts`` evaluators inherited. It is a LEXICOGRAPHIC max, and
    ``'0' < 'f'``, so whenever both splits are materialised under one epcache
    root it selects the 78.5 %-leaked split over the clean one."""
    _cache(tmp_path, CLEAN, CANONICAL_EPISODES)
    _cache(tmp_path, LEAKY, 79)
    legacy_choice = sorted(tmp_path.glob("*val*"))[-1]
    assert legacy_choice.name == LEAKY, (
        "the premise of this whole workstream: the legacy resolver picks the "
        "leaky split")
    # and the replacement picks the clean one
    assert parity.resolve_val_dir(tmp_path, label="t").name == CLEAN


# --------------------------------------------------------------------------- #
# 2. the guard — count / deployment / identity                                  #
# --------------------------------------------------------------------------- #
def test_truncated_val_cache_is_refused(truncated12):
    """⭐ the headline: 12 episodes in a correctly-named clean-val dir."""
    with pytest.raises(SystemExit) as ei:
        parity.assert_val_cache(truncated12, label="--val-cache")
    msg = str(ei.value)
    assert "12" in msg and "NOT a registered val deployment" in msg
    assert "40" in msg and "600" in msg          # expected-vs-actual, both sides


def test_canonical_40_episode_deployment_passes(clean40):
    rec = parity.assert_val_cache(clean40, label="--val-cache",
                                  requested=CANONICAL_EPISODES)
    assert rec["parity"] is True and rec["checked"] is True
    assert rec["episodes_present"] == CANONICAL_EPISODES
    assert rec["deployment"]["n_episodes"] == CANONICAL_EPISODES
    assert "881" in rec["deployment"]["role"]
    assert rec["decision_grade"] is True


def test_full_600_episode_build_passes(tmp_path):
    d = _cache(tmp_path, CLEAN, FULL_BUILD)
    rec = parity.assert_val_cache(d, label="--val-cache")
    assert rec["parity"] is True
    assert rec["episodes_present"] == FULL_BUILD


def test_requesting_more_episodes_than_present_is_refused(clean40):
    """Asking a 40-episode deployment for 100 must not silently score 40."""
    with pytest.raises(SystemExit) as ei:
        parity.assert_val_cache(clean40, label="--val-cache", requested=100)
    assert "SHORT BY 60" in str(ei.value)


def test_extra_or_foreign_episode_count_is_refused(tmp_path):
    """A cache of an unregistered size is refused even when it is BIGGER — an
    over-full val is a substituted val, not a bonus."""
    d = _cache(tmp_path, CLEAN, 41)
    with pytest.raises(SystemExit):
        parity.assert_val_cache(d, label="--val-cache")


def test_allow_partial_downgrades_to_a_warning_and_self_labels(truncated12,
                                                               capsys):
    """A deliberate partial-val probe stays possible — and stays labelled."""
    rec = parity.assert_val_cache(truncated12, label="probe",
                                  decision_grade=False)
    assert rec["decision_grade"] is False
    assert rec["episodes_present"] == 12
    out = capsys.readouterr().out
    assert "UNREGISTERED val deployment" in out and "NOT" in out


# --------------------------------------------------------------------------- #
# 3. content hash — enforced WHEN PRESENT, count-only otherwise (no invention)  #
# --------------------------------------------------------------------------- #
def test_val_manifest_is_count_only_and_no_hash_was_invented():
    """The honesty invariant, inherited from Wave-1 B and kept here.

    No committed artifact enumerates the val uid set. The Wave-1 B agent refused
    to derive one; this workstream refuses too. The manifest must still say so
    out loud, and the ADMISSIBLE COUNTS must carry citations rather than being
    folklore."""
    ent = parity.manifest_entry(CLEAN)
    assert ent["episode_uid_sha256"] is None
    assert "unrecorded" in ent["uid_source"]
    deps = parity.val_deployments()
    assert {d["n_episodes"] for d in deps} == {FULL_BUILD, CANONICAL_EPISODES}
    for d in deps:
        assert d["evidence_class"] == "MEASURED"
        assert d["evidence"] and d["where"]


def test_uid_digest_IS_enforced_once_the_manifest_carries_one(tmp_path):
    """The count check cannot see a SUBSTITUTED set of the right size. The uid
    digest can — and the guard must use it the moment one is recorded, with no
    further code change. Proven against a temp manifest holding a real digest
    for a 40-episode val, exactly what the pod ``--record`` command writes."""
    good = _cache(tmp_path / "good", CLEAN, CANONICAL_EPISODES)
    swapped = _cache(tmp_path / "swapped", CLEAN, CANONICAL_EPISODES, start=500)
    uids = parity.scan_cache_dir(good)
    dep = [{"n_episodes": CANONICAL_EPISODES, "role": "test", "where": "test",
            "evidence": "test", "evidence_class": "MEASURED"}]

    def _manifest(name, *, with_uid_list):
        ent = parity.build_entry(uids, corpus_key=CLEAN, split="val")
        if with_uid_list:                      # what `--record` actually writes
            ent["episode_uids"] = uids
        ent["known_deployments"] = dep
        p = tmp_path / name
        p.write_text(json.dumps({"schema": parity.MANIFEST_SCHEMA,
                                 "corpora": {CLEAN: ent}}), encoding="utf-8")
        return p

    # (a) the real upgrade path: `--record` writes BOTH the digest and the list
    mp = _manifest("recorded.json", with_uid_list=True)
    rec = parity.assert_val_cache(good, label="v", manifest_path=mp)
    assert "prefix" in rec["content_check"] or "sha256" in rec["content_check"]
    with pytest.raises(SystemExit) as ei:
        parity.assert_val_cache(swapped, label="v", manifest_path=mp)
    assert "NOT the canonical sorted prefix" in str(ei.value)

    # (b) a digest-only manifest must STILL content-check a full set, or a
    #     `build_entry`-shaped manifest would silently stay count-only.
    mp2 = _manifest("digest_only.json", with_uid_list=False)
    rec2 = parity.assert_val_cache(good, label="v", manifest_path=mp2)
    assert "sha256" in rec2["content_check"]
    with pytest.raises(SystemExit) as ei2:
        parity.assert_val_cache(swapped, label="v", manifest_path=mp2)
    assert "MISMATCH" in str(ei2.value)


def test_count_only_mode_says_it_is_count_only(clean40, capsys):
    """Until the digest lands, the log line must name the check as COUNT-ONLY
    and print the one command that upgrades it — a number is only as strong as
    the check it announces."""
    parity.assert_val_cache(clean40, label="--val-cache")
    out = capsys.readouterr().out
    assert "COUNT-ONLY" in out
    assert "make_parity_manifest.py --record" in out and "--split val" in out


# --------------------------------------------------------------------------- #
# 4. the leaky split — refused everywhere, by TWO independent paths             #
# --------------------------------------------------------------------------- #
def test_leaky_split_is_refused_by_the_shared_guard(tmp_path):
    d = _cache(tmp_path, LEAKY, 79)
    with pytest.raises(SystemExit) as ei:
        parity.assert_val_cache(d, label="--val-cache")
    msg = str(ei.value)
    assert "LEAKED SPLIT" in msg and CLEAN in msg


def test_leaky_split_is_refused_even_when_it_is_the_only_val_dir(tmp_path):
    _cache(tmp_path, LEAKY, 79)
    with pytest.raises(SystemExit):
        parity.resolve_val_dir(tmp_path, label="--cache-dir")


def test_resolver_prefers_clean_and_says_what_the_legacy_rule_would_have_picked(
        tmp_path, capsys):
    _cache(tmp_path, CLEAN, CANONICAL_EPISODES)
    _cache(tmp_path, LEAKY, 79)
    chosen = parity.resolve_val_dir(tmp_path, label="--cache-dir")
    assert chosen.name == CLEAN
    out = capsys.readouterr().out
    assert "LEAKY" in out and LEAKY in out


def test_taniteval_chokepoint_refuses_a_truncated_clean_val(truncated12):
    """``data.list_val_episodes`` is the chokepoint EVERY decision-grade
    taniteval entrypoint routes through (runner, closedloop, hierarchy,
    pathspeed, efficiency, refc_rerank, planning, planner_p2, bench,
    generalization, strategic_probes)."""
    with pytest.raises(SystemExit):
        tdata.list_val_episodes(str(truncated12), CANONICAL_EPISODES)


def test_taniteval_chokepoint_passes_the_canonical_deployment(clean40):
    files = tdata.list_val_episodes(str(clean40), CANONICAL_EPISODES)
    assert len(files) == CANONICAL_EPISODES
    rec = tdata.last_val_parity()
    assert rec["corpus_key"] == CLEAN
    assert rec["episodes_present"] == CANONICAL_EPISODES
    assert rec["episodes_listed"] == CANONICAL_EPISODES
    assert rec["decision_grade"] is True


def test_taniteval_allow_leaky_stays_possible_but_self_labels(tmp_path, capsys):
    """``label_overlay`` / route-label audits legitimately read the leaky split.
    The escape hatch survives — but the run announces itself."""
    d = _cache(tmp_path, LEAKY, 79)
    files = tdata.list_val_episodes(str(d), 5, allow_leaky=True)
    assert len(files) == 5
    assert tdata.last_val_parity()["decision_grade"] is False
    assert "LEAKY" in capsys.readouterr().out


def test_other_corpora_are_never_refused(tmp_path, capsys):
    """comma / cosmos / OOD generalization corpora are not parity corpora. They
    must warn (NON-PARITY) and run, never refuse — the guard triggers on the
    registered keys only."""
    for key, n in (("comma2k19-val-76b6e94a97a1", 64),
                   ("cosmos-val-e8f3cef4976b", 46)):
        d = _cache(tmp_path, key, n)
        files = tdata.list_val_episodes(str(d), 40)
        assert len(files) == 40
        assert tdata.last_val_parity()["parity"] is False
    assert "NON-PARITY" in capsys.readouterr().out


def test_absent_val_dir_is_not_a_parity_violation(tmp_path):
    """WAVE1_B_REPORT §5's lesson, re-applied on the val side: "you gave me no
    val dir" must stay a soft, non-SystemExit condition. Several callers
    ``except AssertionError`` around an OPTIONAL val block, and a hard exit
    there would kill a finished run at its metrics write."""
    files = tdata.list_val_episodes(str(tmp_path / "does-not-exist"), 40)
    assert files == []
    assert tdata.last_val_parity()["checked"] is False


# --------------------------------------------------------------------------- #
# 5. per-evaluator wiring — RED before the wiring, GREEN after                  #
# --------------------------------------------------------------------------- #
#: the offending shape in every real bypass site: build an EPISODE LIST from a
#: bare glob. Deliberately narrow — ``generalization.ood_corpus_status`` counts
#: ``ep_*.pt`` per dir for a data INVENTORY report and loads nothing, which is
#: not a bypass.
_BYPASS_RE = re.compile(r'(files|eps|episodes)\s*=\s*sorted\([^\n]*'
                        r'glob\(\s*f?["\']ep_\*\.pt["\']')


def _src(mod: str) -> str:
    return (Path(__file__).resolve().parents[1] / "taniteval" / f"{mod}.py"
            ).read_text(encoding="utf-8")


TANITEVAL_CHOKEPOINT_MODULES = [
    "runner", "closedloop", "hierarchy", "pathspeed", "efficiency",
    "refc_rerank", "planning", "planner_p2", "bench", "generalization",
    "strategic_probes",
]


@pytest.mark.parametrize("mod", TANITEVAL_CHOKEPOINT_MODULES)
def test_every_decision_grade_module_routes_through_the_chokepoint(mod):
    """Absence found at ONE location is not absence: this asserts the routing
    per MODULE, not once for the package. A module that starts globbing
    ``ep_*.pt`` directly bypasses every check above, which is exactly how the
    hole opened on the ``stack/`` side."""
    src = _src(mod)
    assert "list_val_episodes" in src, (
        f"{mod}.py no longer routes through data.list_val_episodes — the val "
        f"integrity guard does not cover it")
    bypass = _BYPASS_RE.search(src)
    assert bypass is None, (
        f"{mod}.py builds an episode list from a bare glob "
        f"({bypass.group(0)!r}), bypassing the val guard")


DIRECT_GLOB_MODULES = ["cam_overlay", "flagship_overlay", "corpus_overlay",
                       "direct_overlay", "plan_fan", "plan_fan_clips",
                       "label_overlay"]


@pytest.mark.parametrize("mod", DIRECT_GLOB_MODULES)
def test_viz_modules_also_route_through_the_chokepoint(mod):
    """The overlay/viz family produces no ADE, but it DOES pick a corpus — and
    ``label_overlay`` defaults to the LEAKY split. Routing them through the same
    chokepoint is what makes "the refusal covers the whole harness" true rather
    than true-for-the-modules-someone-remembered."""
    src = _src(mod)
    assert "list_val_episodes" in src, (
        f"{mod}.py selects a corpus without the val guard")
    bypass = _BYPASS_RE.search(src)
    assert bypass is None, (
        f"{mod}.py builds an episode list from a bare glob "
        f"({bypass.group(0)!r}), bypassing the val guard")
