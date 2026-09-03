"""D-V7-EVAL-EXCLUSION — the v7.2 EVAL split's pixels may not enter the v7
TRAINING corpus (BACKLOG R8, a v7f LAUNCH BLOCKER).

⛔ THE HAZARD. The B1 cache `physicalai-b1-w120-256x640cyl` holds 4,713 clips
and the v7.2 EVAL split is 147 label records of which **141 have their pixels in
that same cache**. `train_v6_staged.py --v2-cache` carried no exclusion list, so
a v7f run would train its world-model objectives on the evaluation split and
every later T1 number would be contaminated. Nothing crashes.

⛔⛔ AND IT IS A JOIN, NOT A FILTER. Two places map `ep_idx → clip_id` BY SORTED
CACHE ORDER and verify BY COUNT (the PSG/O10 join and the NAV join). Excluding
clips from the dataset but not from those lists — or the reverse — either
refuses at startup or SHIFTS EVERY LABEL BY N CLIPS while training happily. So
the tests here are not only "were the clips dropped": they are

  * the intersection is DERIVED from the label blob (change the blob, the number
    moves) and is never a literal — the `141` in the docs is an intersection;
  * the excluded clips are absent from the dataset;
  * BOTH joins still pass their count checks after the exclusion;
  * ⭐ and the same checks FAIL LOUDLY on a deliberately desynchronised list —
    the negative control that proves the join is still PROVEN and not merely
    made to agree with itself;
  * the refusals fire (exclusion switched off over a real overlap; the eval
    split unresolvable);
  * the override is printed and RECORDED;
  * ⭐ a cache with NO overlap comes back byte-identical to today's behaviour —
    the load-bearing default-preservation test, because this default is ON.

Nothing here needs the real 46 GB cache or a pod: `parity.v2_clip_ids` and both
joins read FILE NAMES ONLY, so a synthetic cache dir of empty files exercises
the exact code path (the `test_preflight_parity` fixture's rule). The tests that
DO need the gated v7.2 release skip elsewhere.
"""
from __future__ import annotations

import gzip
import json
import sys
import types
from pathlib import Path

import pytest

_STACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_STACK))
sys.path.insert(0, str(_STACK / "scripts"))

from tanitad.data import parity  # noqa: E402
from tanitad.train.intrain_eval import V72, V72_INDEX  # noqa: E402
import train_v6_staged as T  # noqa: E402

# --------------------------------------------------------------------------- #
# facts pinned by this file (MEASURED 2026-09-03, dev box, local release copy)
# --------------------------------------------------------------------------- #
B1_KEY = "physicalai-b1-w120-256x640cyl"
B1_DIGEST_4713 = "e8bfb98e06ebd62edf6566906f0f961b7e6efaf1869ee4aa6646511a9f1509da"
#: the canonical v7.2 EVAL blob. ⛔ THE MD5 IS THE IDENTITY — six copies of this
#: blob exist under three roots and their md5s differ (intrain_eval's own rule).
EVAL_MD5 = "aa12c948f062181c3297265b51526ec5"
EVAL_N = 147
#: ⚠️ NOT a hard-coded input anywhere in the trainer: it is what the derivation
#: below must PRODUCE. If this number ever changes, the release changed.
EVAL_WITH_PIXELS_IN_B1 = 141

_V72_ROOT = Path("C:/Users/Admin/tanitad-wt/_s2build/release/v72")
needs_release = pytest.mark.skipif(
    not (_V72_ROOT / "s2_labels_v7.2_eval.jsonl.gz").exists(),
    reason="local v7.2 release copy not on this box")


@pytest.fixture(autouse=True)
def _fresh_resolution():
    """The exclusion is resolved ONCE per (cache, flags) tuple — which is the
    point in production and a cross-test leak here."""
    T._EVAL_EXCL_CACHE.clear()
    yield
    T._EVAL_EXCL_CACHE.clear()


# --------------------------------------------------------------------------- #
# fixtures — a synthetic cache dir and a synthetic eval blob
# --------------------------------------------------------------------------- #
def _cache(root: Path, name: str, clip_ids) -> Path:
    """A v2 cache dir. ``v2_clip_ids`` and both joins read FILE NAMES ONLY, so
    the payloads need not exist — and deliberately do not, so this fixture
    cannot drift into testing the decoder."""
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    for c in clip_ids:
        (d / f"{c}{parity.V2_SUFFIX}").write_bytes(b"")
    return d


def _blob(root: Path, clip_ids, name="eval.jsonl.gz") -> Path:
    p = root / name
    with gzip.open(p, "wt", encoding="utf-8") as fh:
        for c in clip_ids:
            fh.write(json.dumps({"clip_id": c, "schema_version": "s2-geom-v7"})
                     + "\n")
    return p


def _args(cache_dirs, **kw):
    a = types.SimpleNamespace(
        v2_cache=[str(x) for x in cache_dirs], exclude_eval_clips="auto",
        allow_eval_clips_in_train=False, s2_labels=None, nav_labels=None,
        dry_run=False)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def _providers(cache_dirs):
    """Providers in EXACTLY `build_v2_providers`' order, each tagged with its
    clip id so a test can assert WHICH survived."""
    return [types.SimpleNamespace(clip=c, frames=None)
            for c in T.cache_clip_ids_in_provider_order(
                [str(x) for x in cache_dirs])]


EVALISH = [f"e0000000-0000-4000-8000-{i:012d}" for i in range(12)]
TRAINISH = [f"70000000-0000-4000-8000-{i:012d}" for i in range(30)]


# =========================================================================== #
# 1. the intersection is DERIVED from the blob, never hard-coded
# =========================================================================== #
def test_the_intersection_is_derived_from_the_blob_and_moves_with_it(tmp_path):
    """⛔ The `141` in every doc is an INTERSECTION, not an input. Change the
    blob and the number must move; if it did not, the trainer would be carrying
    a literal and would keep excluding 141 clips on a corpus that has 0."""
    cache = _cache(tmp_path, "cache", EVALISH[:5] + TRAINISH)
    b3 = _blob(tmp_path, EVALISH[:3], "b3.jsonl.gz")
    b9 = _blob(tmp_path, EVALISH[:9], "b9.jsonl.gz")

    r3 = T.eval_exclusion(_args([cache], exclude_eval_clips=str(b3)))
    T._EVAL_EXCL_CACHE.clear()
    r9 = T.eval_exclusion(_args([cache], exclude_eval_clips=str(b9)))

    assert r3["n_eval_labels"] == 3 and r9["n_eval_labels"] == 9
    # the cache holds only 5 of the eval-ish ids, so the intersection saturates
    assert r3["n_overlap"] == 3 and r3["n_removed"] == 3
    assert r9["n_overlap"] == 5 and r9["n_removed"] == 5
    assert r9["n_cache_clips"] == 35


def test_the_four_numbers_are_printed_on_one_line_and_recorded(tmp_path, capsys):
    """One line, four numbers: cache count, eval-label count, the intersection
    ACTUALLY REMOVED, and the resulting episode count — and the same four in
    `config.json`, because a number that lives only in a console line nobody
    re-reads is the "please merge this in a README" failure in a costume."""
    cache = _cache(tmp_path, "cache", EVALISH[:4] + TRAINISH)
    blob = _blob(tmp_path, EVALISH)
    a = _args([cache], exclude_eval_clips=str(blob))
    _kept, rec = T.apply_eval_exclusion(a, _providers([cache]))
    line = [x for x in capsys.readouterr().out.splitlines()
            if "eval-exclusion" in x]
    assert len(line) == 1, line
    for tok in ("cache 34 clips", "v7.2 EVAL 12 label records", "REMOVED 4",
                "30 training episodes remain"):
        assert tok in line[0], (tok, line[0])
    cfg = T.eval_exclusion_record(rec)
    assert (cfg["n_cache_clips"], cfg["n_eval_labels"], cfg["n_removed"],
            cfg["n_episodes_after"]) == (34, 12, 4, 30)
    json.dumps(cfg)                                   # must be config.json-safe


# =========================================================================== #
# 2. the excluded clips are absent from the dataset
# =========================================================================== #
def test_excluded_clips_are_absent_from_the_provider_list(tmp_path):
    cache = _cache(tmp_path, "cache", EVALISH + TRAINISH)
    blob = _blob(tmp_path, EVALISH)
    a = _args([cache], exclude_eval_clips=str(blob))
    kept, rec = T.apply_eval_exclusion(a, _providers([cache]))

    assert rec["mode"] == "excluded" and rec["n_removed"] == len(EVALISH)
    assert len(kept) == len(TRAINISH)
    assert {p.clip for p in kept} == set(TRAINISH)
    assert not ({p.clip for p in kept} & set(EVALISH))
    #: order preserved — the join is BY ORDER, so a reordering filter would be
    #: the off-by-N in a different costume
    assert [p.clip for p in kept] == sorted(TRAINISH)


def test_the_exclusion_survives_two_concatenated_cache_dirs(tmp_path):
    """``--v2-cache`` is nargs='+' and ``build_v2_providers`` CONCATENATES the
    dirs, so the provider order is per-dir-sorted, not globally sorted."""
    d1 = _cache(tmp_path, "d1", TRAINISH[:10] + EVALISH[:3])
    d2 = _cache(tmp_path, "d2", TRAINISH[10:20] + EVALISH[3:6])
    blob = _blob(tmp_path, EVALISH)
    a = _args([d1, d2], exclude_eval_clips=str(blob))
    order = T.cache_clip_ids_in_provider_order([d1, d2])
    assert order == sorted(TRAINISH[:10] + EVALISH[:3]) + \
        sorted(TRAINISH[10:20] + EVALISH[3:6])
    kept, rec = T.apply_eval_exclusion(a, _providers([d1, d2]))
    assert rec["n_removed"] == 6 and len(kept) == 20
    assert not ({p.clip for p in kept} & set(EVALISH))


# =========================================================================== #
# 3. BOTH joins still pass their count checks after the exclusion …
# =========================================================================== #
def test_both_joins_pass_their_count_checks_after_the_exclusion(tmp_path):
    cache = _cache(tmp_path, "cache", EVALISH + TRAINISH)
    blob = _blob(tmp_path, EVALISH)
    a = _args([cache], exclude_eval_clips=str(blob))
    kept, rec = T.apply_eval_exclusion(a, _providers([cache]))
    joined = T.join_clip_ids(a, cache)

    assert len(joined) == len(kept)
    assert joined == [p.clip for p in kept]           # SAME list, SAME order
    for who in ("PSG", "nav"):
        T.assert_cache_join(joined, len(kept), who=who,
                            excluded=rec["n_removed"])


# =========================================================================== #
# 4. … and FAIL LOUDLY on a desynchronised list — the negative control
# =========================================================================== #
@pytest.mark.parametrize("who,marker", [("PSG", "PSG:"), ("nav", "[nav]")])
@pytest.mark.parametrize("desync", ["short", "long"])
def test_a_desynchronised_join_list_still_fails_loudly(tmp_path, who, marker,
                                                       desync):
    """⭐ THE CONTROL THAT MAKES THE REST MEAN ANYTHING. A count check that can
    no longer fail is not a check — it is a comment. Here the exclusion is
    applied to the DATASET and a deliberately different list is handed to the
    join; the refusal must still arrive, in BOTH directions of mismatch."""
    cache = _cache(tmp_path, "cache", EVALISH + TRAINISH)
    blob = _blob(tmp_path, EVALISH)
    a = _args([cache], exclude_eval_clips=str(blob))
    kept, rec = T.apply_eval_exclusion(a, _providers([cache]))
    good = T.join_clip_ids(a, cache)
    bad = good[:-1] if desync == "short" else good + ["ffffffff-desync"]

    with pytest.raises(SystemExit) as ex:
        T.assert_cache_join(bad, len(kept), who=who,
                            excluded=rec["n_removed"])
    msg = str(ex.value)
    assert marker in msg and "off-by-N" in msg
    assert f"{len(bad)} clips" in msg and f"{len(kept)} episodes" in msg
    #: and it must NAME the exclusion, because with the flag in force that is
    #: the first thing a reader should suspect
    assert "--exclude-eval-clips" in msg


def test_the_unfiltered_cache_list_desynchronises_the_join_and_is_refused(
        tmp_path):
    """The specific failure this whole design exists to prevent: the dataset
    filtered, the join NOT. It must refuse, not silently shift every label."""
    cache = _cache(tmp_path, "cache", EVALISH + TRAINISH)
    blob = _blob(tmp_path, EVALISH)
    a = _args([cache], exclude_eval_clips=str(blob))
    kept, _ = T.apply_eval_exclusion(a, _providers([cache]))
    unfiltered = sorted(q.name[:-len(parity.V2_SUFFIX)]
                        for q in Path(cache).glob(parity.V2_EPISODE_GLOB))
    assert len(unfiltered) == len(kept) + len(EVALISH)
    with pytest.raises(SystemExit):
        T.assert_cache_join(unfiltered, len(kept), who="nav", excluded=12)


def test_the_pre_filter_count_proof_refuses_a_mismatched_provider_list(
        tmp_path):
    """The assertion that actually PROVES the join, and it runs BEFORE anything
    is dropped: cache clips vs providers returned. A wrong-length provider list
    must never reach the index-based filter."""
    cache = _cache(tmp_path, "cache", EVALISH + TRAINISH)
    blob = _blob(tmp_path, EVALISH)
    a = _args([cache], exclude_eval_clips=str(blob))
    with pytest.raises(SystemExit) as ex:
        T.apply_eval_exclusion(a, _providers([cache])[:-3])
    assert "ep_idx->clip_id join is BY SORTED CACHE ORDER" in str(ex.value)
    assert "before anything is dropped" in str(ex.value).lower()


# =========================================================================== #
# 5. the refusals
# =========================================================================== #
def test_excluding_over_an_overlap_is_never_itself_a_problem(tmp_path):
    """The happy path: the overlap exists, the default excluded it, preflight
    is silent. A guard that fired on the legitimate case would be removed
    within a week (parity.py §10c's design constraint)."""
    cache = _cache(tmp_path, "cache", EVALISH + TRAINISH)
    blob = _blob(tmp_path, EVALISH)
    a = _args([cache], exclude_eval_clips=str(blob))
    assert T.eval_exclusion(a)["n_removed"] == len(EVALISH)
    assert T._preflight_eval_exclusion(a) == []


def test_the_refusal_fires_on_a_real_overlap_with_the_exclusion_off(tmp_path,
                                                                    monkeypatch):
    """`--exclude-eval-clips none` over a cache that DOES overlap must refuse,
    naming the count and the override flag."""
    real = sorted(parity.v72_eval_clip_digests())      # digests, not ids
    # a cache whose ids ARE in the eval split: take them from the committed
    # digest set by construction — we cannot invert sha256, so instead point
    # the digest oracle at a synthetic set built from OUR ids.
    synth = tmp_path / "digests.json"
    digs = sorted(parity.clip_digest(c) for c in EVALISH)
    synth.write_text(json.dumps({
        "schema": parity.CLIP_DIGESTS_SCHEMA, "corpus_key": "synthetic",
        "is_full_corpus": False, "deployment": "synthetic-eval",
        "n_clips": len(digs), "clip_id_digests": digs,
        "digest_of_digests": parity.uid_digest(digs),
        "cross_check_source": "synthetic", "cross_check_episodes": len(digs),
    }), encoding="utf-8")
    monkeypatch.setattr(parity, "V72_EVAL_DIGESTS_PATH", synth)
    parity._CLIP_DIGEST_CACHE.pop(str(synth), None)
    assert len(real) == EVAL_N                          # the real one is intact

    cache = _cache(tmp_path, "cache", EVALISH + TRAINISH)
    a = _args([cache], exclude_eval_clips="none")
    rec = T.eval_exclusion(a)
    assert rec["n_overlap"] == len(EVALISH) and rec["n_removed"] == 0
    probs = T._preflight_eval_exclusion(a)
    assert len(probs) == 1
    assert f"OVERLAPS THE v7.2 EVAL SPLIT ON {len(EVALISH)} OF 42 CLIP(S)" \
        in probs[0]
    assert "--allow-eval-clips-in-train" in probs[0]
    assert "--exclude-eval-clips" in probs[0]


def test_the_run_refuses_when_the_eval_split_cannot_be_resolved_at_all(
        tmp_path, monkeypatch):
    """Neither the blob nor the digest set: the run cannot PROVE it is clean,
    so it refuses rather than assuming (parity §10's C112 lesson)."""
    monkeypatch.setattr(parity, "V72_EVAL_DIGESTS_PATH",
                        tmp_path / "does-not-exist.json")
    monkeypatch.delenv(T.V72_ROOT_ENV, raising=False)
    cache = _cache(tmp_path, "cache", TRAINISH)
    a = _args([cache])
    rec = T.eval_exclusion(a)
    assert rec["mode"] == "UNRESOLVED"
    probs = T._preflight_eval_exclusion(a)
    assert len(probs) == 1
    assert "could not be resolved" in probs[0]
    assert EVAL_MD5 in probs[0] and "--exclude-eval-clips" in probs[0]
    assert "--allow-eval-clips-in-train" in probs[0]


def test_an_exclusion_path_that_does_not_exist_refuses(tmp_path):
    cache = _cache(tmp_path, "cache", TRAINISH)
    a = _args([cache], exclude_eval_clips=str(tmp_path / "nope.jsonl.gz"))
    assert T.eval_exclusion(a)["mode"] == "UNRESOLVED"
    assert T._preflight_eval_exclusion(a)


def test_an_empty_exclusion_source_is_refused(tmp_path):
    """An empty exclusion is indistinguishable from no exclusion at all — which
    is the failure the flag exists to prevent."""
    p = _blob(tmp_path, [])
    with pytest.raises(SystemExit) as ex:
        T.eval_clip_ids_from(p)
    assert "ZERO clip ids" in str(ex.value)


# =========================================================================== #
# 6. the override is explicit, loud and RECORDED
# =========================================================================== #
def test_the_override_keeps_the_clips_and_is_printed_and_recorded(tmp_path,
                                                                  capsys):
    cache = _cache(tmp_path, "cache", EVALISH + TRAINISH)
    blob = _blob(tmp_path, EVALISH)
    a = _args([cache], exclude_eval_clips=str(blob),
              allow_eval_clips_in_train=True)
    kept, rec = T.apply_eval_exclusion(a, _providers([cache]))
    out = capsys.readouterr().out

    assert len(kept) == len(EVALISH) + len(TRAINISH)   # nothing removed
    assert rec["mode"] == "OVERRIDE" and rec["n_removed"] == 0
    assert rec["n_overlap"] == len(EVALISH)            # measured anyway
    assert "--allow-eval-clips-in-train IN FORCE" in out
    assert "is a measurement on training data" in out
    cfg = T.eval_exclusion_record(rec)
    assert cfg["allow_eval_clips_in_train"] is True
    assert cfg["mode"] == "OVERRIDE" and cfg["n_overlap"] == len(EVALISH)
    assert T._preflight_eval_exclusion(a) == []        # stamped ⇒ permitted


# =========================================================================== #
# 7. ⭐ DEFAULT PRESERVATION — a cache with NO overlap is unchanged
# =========================================================================== #
def test_a_cache_with_no_overlap_is_byte_identical_to_todays_behaviour(
        tmp_path, capsys):
    """⭐ THE LOAD-BEARING TEST, because this default is ON. Every arm whose
    cache does not meet the eval split (refav1, refcv3, every parity arm) must
    get back the SAME provider objects, in the SAME order, with no new
    assertion able to fire on it."""
    cache = _cache(tmp_path, "cache", TRAINISH)
    blob = _blob(tmp_path, EVALISH)                    # disjoint from the cache
    a = _args([cache], exclude_eval_clips=str(blob))
    before = _providers([cache])
    kept, rec = T.apply_eval_exclusion(a, before)

    assert rec["mode"] == "clean" and rec["n_overlap"] == 0
    assert rec["n_removed"] == 0
    assert kept == before                              # same objects, same order
    assert all(x is y for x, y in zip(kept, before))
    assert T.join_clip_ids(a, cache) == sorted(
        q.name[:-len(parity.V2_SUFFIX)]
        for q in Path(cache).glob(parity.V2_EPISODE_GLOB))
    assert T._preflight_eval_exclusion(a) == []
    #: the join check is the INCUMBENT one, unchanged, on the unchanged list
    T.assert_cache_join(T.join_clip_ids(a, cache), len(before), who="PSG")


def test_a_dry_run_and_a_cacheless_run_are_untouched(tmp_path):
    """A dry run mounts no corpus and trains nothing, so nothing here applies —
    the same scoping error that once truncated the whole dry ladder through a
    `KeyError('dry_ckpt')` three modules away."""
    cache = _cache(tmp_path, "cache", EVALISH)
    assert T._preflight_eval_exclusion(_args([cache], dry_run=True)) == []
    assert T._preflight_eval_exclusion(_args([])) == []
    assert T.eval_exclusion(_args([]))["mode"] == "not-applicable"


# =========================================================================== #
# 8. the flags, and one canonical resolution shared by all three consumers
# =========================================================================== #
def test_the_flags_exist_and_default_to_exclusion_on():
    a = T.build_parser().parse_args(["--stage", "S-W", "--out", "x"])
    assert a.exclude_eval_clips == "auto"
    assert a.allow_eval_clips_in_train is False


def test_all_three_consumers_read_ONE_resolution(tmp_path):
    """A second, independently computed exclusion would BE the off-by-N hazard
    wearing a fix's name. The memo is the mechanism, so assert identity."""
    cache = _cache(tmp_path, "cache", EVALISH + TRAINISH)
    blob = _blob(tmp_path, EVALISH)
    a = _args([cache], exclude_eval_clips=str(blob))
    r1 = T.eval_exclusion(a)
    T.join_clip_ids(a, cache)
    T.apply_eval_exclusion(a, _providers([cache]))
    assert T.eval_exclusion(a) is r1
    assert len(T._EVAL_EXCL_CACHE) == 1


def test_the_record_written_to_config_json_carries_no_clip_ids(tmp_path):
    """🔒 clip ids are gated-confidential: counts only, in the console AND in
    the run row."""
    cache = _cache(tmp_path, "cache", EVALISH + TRAINISH)
    blob = _blob(tmp_path, EVALISH)
    a = _args([cache], exclude_eval_clips=str(blob))
    _, rec = T.apply_eval_exclusion(a, _providers([cache]))
    blob_text = json.dumps(T.eval_exclusion_record(rec))
    assert "_excluded" not in blob_text
    for c in EVALISH + TRAINISH:
        assert c not in blob_text


def test_the_clip_index_route_is_accepted_as_a_source(tmp_path):
    """The clip index beside the blob names the same 147 clips; a host that has
    the index and not the blob must still be able to answer the question."""
    cache = _cache(tmp_path, "cache", EVALISH[:5] + TRAINISH)
    idx = tmp_path / "clip_index_eval.json"
    idx.write_text(json.dumps({"clips": {c: {"excluded": False}
                                         for c in EVALISH}}), encoding="utf-8")
    rec = T.eval_exclusion(_args([cache], exclude_eval_clips=str(idx)))
    assert rec["source_kind"] == "v7.2 clip index"
    assert rec["n_eval_clips"] == len(EVALISH) and rec["n_removed"] == 5


# =========================================================================== #
# 9. the committed digest set — the fallback oracle
# =========================================================================== #
def test_the_committed_digest_set_is_self_consistent_and_is_the_eval_split():
    """`load_clip_digests` re-hashes and refuses a truncated or hand-edited
    file — a short one would UNDER-exclude, which is a leak wearing a working
    guard as a disguise."""
    digs = parity.v72_eval_clip_digests()
    assert len(digs) == EVAL_N
    doc = parity.load_clip_digests(parity.V72_EVAL_DIGESTS_PATH)
    assert doc["schema"] == parity.CLIP_DIGESTS_SCHEMA
    assert doc["is_full_corpus"] is False              # 147 of 4,719 is a SUBSET
    assert doc["corpus_key"] == B1_KEY
    assert doc["cross_check_episodes"] == EVAL_N
    assert parity.clips_in_v72_eval(["not-a-clip"]) == []


def test_the_digest_set_is_used_only_when_the_blob_does_not_resolve(tmp_path,
                                                                    monkeypatch):
    monkeypatch.delenv(T.V72_ROOT_ENV, raising=False)
    cache = _cache(tmp_path, "cache", TRAINISH)
    rec = T.eval_exclusion(_args([cache]))             # auto, no blob anywhere
    assert rec["n_eval_labels"] == EVAL_N
    assert "FALLBACK" in rec["source_kind"]
    assert rec["mode"] == "clean"


def test_a_stale_digest_set_against_the_canonical_blob_is_refused(tmp_path,
                                                                  monkeypatch):
    """⛔ Two oracles that must agree and do not: refuse rather than pick one.
    Simulated by pointing the digest oracle at a DIFFERENT set while the source
    carries the canonical eval md5."""
    if not (_V72_ROOT / "s2_labels_v7.2_eval.jsonl.gz").exists():
        pytest.skip("local v7.2 release copy not on this box")
    digs = sorted(parity.clip_digest(c) for c in EVALISH)
    synth = tmp_path / "stale.json"
    synth.write_text(json.dumps({
        "schema": parity.CLIP_DIGESTS_SCHEMA, "corpus_key": "synthetic",
        "is_full_corpus": False, "deployment": "stale",
        "n_clips": len(digs), "clip_id_digests": digs,
        "digest_of_digests": parity.uid_digest(digs),
        "cross_check_source": "synthetic", "cross_check_episodes": len(digs),
    }), encoding="utf-8")
    monkeypatch.setattr(parity, "V72_EVAL_DIGESTS_PATH", synth)
    parity._CLIP_DIGEST_CACHE.pop(str(synth), None)
    cache = _cache(tmp_path, "cache", TRAINISH)
    a = _args([cache], exclude_eval_clips=str(
        _V72_ROOT / "s2_labels_v7.2_eval.jsonl.gz"))
    rec = T.eval_exclusion(a)
    assert rec["mode"] == "UNRESOLVED"
    assert "disagree on the membership" in rec["why"]
    assert T._preflight_eval_exclusion(a)


# =========================================================================== #
# 10. the REAL v7.2 release — the 141 is DERIVED, and the manifest agrees
# =========================================================================== #
def _release_ids(name):
    out = []
    with gzip.open(_V72_ROOT / name, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line)["clip_id"])
    return out


@needs_release
def test_the_real_eval_blob_is_the_canonical_pin_and_the_digest_set_agrees():
    p = _V72_ROOT / "s2_labels_v7.2_eval.jsonl.gz"
    ids, stamp = T.eval_clip_ids_from(p)
    assert stamp["md5"] == EVAL_MD5 == V72["eval"]["md5"]
    assert stamp["n_records"] == EVAL_N == V72["eval"]["n"]
    assert stamp["v72_side"] == "eval" and len(ids) == EVAL_N
    assert {parity.clip_digest(c) for c in ids} == set(
        parity.v72_eval_clip_digests())
    idx = _V72_ROOT / "clip_index_eval.json"
    if idx.exists():
        iids, istamp = T.eval_clip_ids_from(idx)
        assert istamp["md5"] == V72_INDEX["eval_index"]["md5"]
        assert istamp["v72_side"] == "eval_index" and iids == ids


@needs_release
def test_the_141_is_an_intersection_this_code_derives(tmp_path):
    """⭐ THE NUMBER IN THE BLOCKER, RE-DERIVED — not read from the manifest.

    B1 = union(v7.2 train 4,572, v7.2 eval 147) minus the 6 deployed-val40
    clips = 4,713, whose sorted-clip-id sha256 REPRODUCES the manifest's
    committed `clip_id_sha256_sorted`; the eval split's intersection with it is
    141. The synthetic cache is built from those ids (file names only), so this
    exercises the SAME code path a Thor launch takes.
    """
    ev = set(_release_ids("s2_labels_v7.2_eval.jsonl.gz"))
    tr = set(_release_ids("s2_labels_v7.2_train.jsonl.gz"))
    val40 = parity.deployed_val_clip_digests()
    b1 = sorted(c for c in (ev | tr) if parity.clip_digest(c) not in val40)
    assert len(b1) == 4713
    assert parity.uid_digest(b1) == B1_DIGEST_4713         # == the manifest's

    cache = _cache(tmp_path, "b1", b1)
    a = _args([cache], exclude_eval_clips=str(
        _V72_ROOT / "s2_labels_v7.2_eval.jsonl.gz"))
    kept, rec = T.apply_eval_exclusion(a, _providers([cache]))

    assert rec["n_cache_clips"] == 4713
    assert rec["n_eval_labels"] == EVAL_N
    assert rec["n_removed"] == EVAL_WITH_PIXELS_IN_B1 == 141
    assert rec["n_episodes_after"] == len(kept) == 4713 - 141 == 4572
    #: and the manifest's own recorded figure must be the SAME number, reached
    #: by a completely different route (a committed provenance field)
    reg = rec["registered"]
    assert reg["corpus_key"] == B1_KEY
    assert reg["matched_by"] == "clip-id digest (exact)"
    assert reg["registered_n_with_pixels"] == rec["n_removed"]
    assert reg["registered_eval_md5"] == EVAL_MD5
    #: both joins still prove themselves on the filtered corpus
    joined = T.join_clip_ids(a, cache)
    for who in ("PSG", "nav"):
        T.assert_cache_join(joined, len(kept), who=who,
                            excluded=rec["n_removed"])
    with pytest.raises(SystemExit):                    # the negative control
        T.assert_cache_join(joined + ["x"], len(kept), who="nav", excluded=141)


@needs_release
def test_auto_resolves_the_eval_blob_from_the_env_root(tmp_path, monkeypatch):
    """`auto` on a host that keeps the release somewhere unusual: one env var,
    and the PRIMARY (blob) source is used rather than the digest fallback."""
    monkeypatch.setenv(T.V72_ROOT_ENV, str(_V72_ROOT))
    cache = _cache(tmp_path, "cache", TRAINISH)
    rec = T.eval_exclusion(_args([cache]))
    assert rec["source_kind"] == "v7.2 label blob"
    assert rec["source_md5"] == EVAL_MD5
    assert rec["digest_set"]["agrees_with_blob"] is True
    assert rec["mode"] == "clean"
