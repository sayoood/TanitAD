"""V2 parity enforcement — the regression suite for ``parity.py`` §9.

THE HOLE THIS CLOSES (MEASURED, ``…/2026-07-28-wide-fov-build/WIDE_FOV_BUILD.md``
§6): ``train_flagship_v4`` has no v2 support at all, and ``train_flagship4b
--v2-cache`` read a v2 cache with **no parity check on that branch** — its guard
lives in ``_cache_split``, which only the ``--cache-dirs`` branch calls. The v5
wide cache (a v2 re-cache of the sacred split, because the raw epcache at
120°/256×640 is ~697 GB and fits on no host) was therefore trainable with **zero
parity enforcement**.

⚠️ **THE HEADLINE TEST IS :func:`test_guard_refuses_a_swapped_clip_at_identical_count`**
— drop one clip and add one foreign clip and the COUNT IS UNCHANGED. A guard
that cannot fail on that case is not a membership check, it is a count check
wearing one (class C13). Every refusal below asserts the SPECIFIC reason in the
message, not merely that something raised: a guard that fires for the wrong
cause is the same defect one layer down.

🔒 Every clip id in this file is synthetic. Real PhysicalAI-AV clip ids are
gated-confidential and appear in no test, no fixture and no artifact.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from tanitad.data import parity                                  # noqa: E402

SRC_KEY = "synthetic-train-aaaaaaaaaaaa"
SIB_KEY = "synthetic-w120x256cyl-bbbbbbbbbbbb"

#: 12 synthetic clips; indices 3 and 7 are the "recorded decode failures", the
#: stand-in for the parity corpus's 24 corrupt clips at 1798..1941.
CLIPS = [f"{i:02d}cl-{chr(ord('a') + i)}" for i in range(12)]
SKIP_INDICES = [3, 7]
GEOMETRY = {"height": 256, "width": 640, "f_ref": 305.5775,
            "projection": "cylindrical", "frame_tag": "256x640f305.5775cyl"}


# --------------------------------------------------------------------------- #
# fixtures                                                                      #
# --------------------------------------------------------------------------- #
def _write_manifest(path: Path, man: dict) -> Path:
    path.write_text(json.dumps(man, indent=1), encoding="utf-8")
    parity._MANIFEST_CACHE.pop(str(path), None)     # memoized by path
    return path


def _make_v2_cache(dirpath: Path, clip_ids) -> Path:
    """A v2 cache dir of EMPTY ``<clip_id>.v2ep.pt`` markers.

    Empty on purpose: the guard runs on FILENAMES and must fire before a single
    payload is opened. If it did not, ``torch.load`` would die first and the
    test would pass for the wrong reason."""
    dirpath.mkdir(parents=True, exist_ok=True)
    for c in clip_ids:
        (dirpath / f"{c}{parity.V2_SUFFIX}").touch()
    return dirpath


@pytest.fixture()
def manifest(tmp_path) -> Path:
    """A synthetic manifest whose source corpus carries a ``clip_membership``
    block shaped exactly like the committed one."""
    man = {
        "schema": parity.MANIFEST_SCHEMA,
        "corpora": {
            SRC_KEY: {
                "corpus_key": SRC_KEY, "split": "train",
                "episode_count": len(CLIPS) - len(SKIP_INDICES),
                "uid_kind": parity.EPCACHE_UID_KIND,
                "episode_uid_sha256": parity.uid_digest(
                    [f"ep_{i:05d}.pt" for i in range(len(CLIPS))
                     if i not in SKIP_INDICES]),
                "clip_membership": {
                    "n_clips": len(CLIPS),
                    "clip_id_sha256_sorted": parity.uid_digest(CLIPS),
                    "ordered_equals_sorted": CLIPS == sorted(CLIPS),
                    "decode_failures": len(SKIP_INDICES),
                },
                "skip_indices": SKIP_INDICES,
                "skip_count": len(SKIP_INDICES),
            },
        },
    }
    return _write_manifest(tmp_path / "parity_manifest.json", man)


@pytest.fixture()
def clip_list(tmp_path) -> Path:
    """🔒 the stand-in for the pod-side exported ordered clip list."""
    p = tmp_path / "clips.txt"
    p.write_text("\n".join(CLIPS) + "\n", encoding="utf-8")
    return p


def _register(tmp_path, manifest: Path, clip_ids, clip_list=None,
              key: str = SIB_KEY) -> tuple[Path, Path]:
    """Build a v2 cache holding ``clip_ids`` and register it under ``key``."""
    cache = _make_v2_cache(tmp_path / key, clip_ids)
    ent = parity.register_v2_geometry_sibling(
        cache, new_key=key, geometry=GEOMETRY, source_key=SRC_KEY,
        expect_clips=clip_list, manifest_path=manifest)
    man = json.loads(manifest.read_text(encoding="utf-8"))
    man["corpora"][key] = ent
    _write_manifest(manifest, man)
    return cache, manifest


# --------------------------------------------------------------------------- #
# 1. verify_v2_membership — the proof itself, both directions                   #
# --------------------------------------------------------------------------- #
def test_complete_build_verifies_digest_only(tmp_path, manifest):
    """No clip list on this host: a COMPLETE build still proves membership."""
    cache = _make_v2_cache(tmp_path / "wide", CLIPS)
    rec = parity.verify_v2_membership(cache, source_key=SRC_KEY,
                                      manifest_path=manifest)
    assert rec["membership_identical"] and rec["clips_built"] == len(CLIPS)
    assert rec["mode"].startswith("digest-only")


def test_digest_only_refuses_an_incomplete_build_and_names_the_limit(
        tmp_path, manifest):
    """⚠️ THE THING THE V2 PATH CANNOT DO. Without the clip list the check has
    no way to tell a legitimate decode failure from a lost clip, so it refuses —
    and must SAY that, or someone will read the refusal as a broken cache."""
    cache = _make_v2_cache(tmp_path / "wide",
                           [c for i, c in enumerate(CLIPS)
                            if i not in SKIP_INDICES])
    with pytest.raises(parity.ParityViolation) as e:
        parity.verify_v2_membership(cache, source_key=SRC_KEY,
                                    manifest_path=manifest)
    msg = str(e.value)
    assert "DIGEST-ONLY MODE cannot say WHICH clips differ" in msg
    assert "--expect-clips" in msg
    assert f"SHORT BY {len(SKIP_INDICES)}" in msg


def test_expect_clips_accepts_exactly_the_recorded_decode_failures(
        tmp_path, manifest, clip_list):
    """The PASS direction for a real build: the clips that fail to decode are
    the corpus's own recorded failures, so the shortfall is legitimate."""
    cache = _make_v2_cache(tmp_path / "wide",
                           [c for i, c in enumerate(CLIPS)
                            if i not in SKIP_INDICES])
    rec = parity.verify_v2_membership(cache, source_key=SRC_KEY,
                                      expect_clips=clip_list,
                                      manifest_path=manifest)
    assert rec["verified"] and rec["extra_count"] == 0
    assert rec["missing_count"] == len(SKIP_INDICES)
    assert rec["shortfall_identity_checked"] is True
    assert rec["shortfall_matches_recorded_skips"] is True


def test_refuses_a_shortfall_of_the_right_SIZE_but_the_wrong_clips(
        tmp_path, manifest, clip_list):
    """⭐ The strengthening over the sibling stream's ``verify_v2_parity.py``,
    which accepted any ``len(missing) == 24``. Here the SAME clips must fail
    again — a build that lost two OTHER clips is a different episode set."""
    wrong = [0, 5]
    assert len(wrong) == len(SKIP_INDICES) and set(wrong) != set(SKIP_INDICES)
    cache = _make_v2_cache(tmp_path / "wide",
                           [c for i, c in enumerate(CLIPS) if i not in wrong])
    with pytest.raises(parity.ParityViolation) as e:
        parity.verify_v2_membership(cache, source_key=SRC_KEY,
                                    expect_clips=clip_list,
                                    manifest_path=manifest)
    msg = str(e.value)
    assert "are the RECORDED failures and" in msg and "are NOT" in msg
    assert "Do not register and do not train" in msg


def test_refuses_a_foreign_clip(tmp_path, manifest, clip_list):
    cache = _make_v2_cache(tmp_path / "wide", CLIPS + ["ff-foreign"])
    with pytest.raises(parity.ParityViolation) as e:
        parity.verify_v2_membership(cache, source_key=SRC_KEY,
                                    expect_clips=clip_list,
                                    manifest_path=manifest)
    assert "FOREIGN clip(s)" in str(e.value)


def test_refuses_an_expect_clips_list_that_is_not_the_parity_split(
        tmp_path, manifest):
    """⚠️ The sibling stream's script compared the built digest against a digest
    carried in the SAME sidecar as the clip list, so a self-consistent WRONG
    pair verified. Here the list must first reproduce the committed manifest."""
    bogus = tmp_path / "bogus.txt"
    bogus.write_text("\n".join(CLIPS[:6] + ["zz-not-parity"]) + "\n",
                     encoding="utf-8")
    cache = _make_v2_cache(tmp_path / "wide", CLIPS[:6] + ["zz-not-parity"])
    with pytest.raises(parity.ParityViolation) as e:
        parity.verify_v2_membership(cache, source_key=SRC_KEY,
                                    expect_clips=bogus, manifest_path=manifest)
    assert "is not the parity split" in str(e.value)


def test_refuses_a_source_with_no_clip_membership_block(tmp_path):
    man = {"schema": parity.MANIFEST_SCHEMA,
           "corpora": {SRC_KEY: {"corpus_key": SRC_KEY, "split": "train",
                                 "episode_count": 3,
                                 "uid_kind": parity.EPCACHE_UID_KIND}}}
    mp = _write_manifest(tmp_path / "m.json", man)
    cache = _make_v2_cache(tmp_path / "wide", CLIPS)
    with pytest.raises(parity.ParityViolation) as e:
        parity.verify_v2_membership(cache, source_key=SRC_KEY, manifest_path=mp)
    assert "NO clip_membership block" in str(e.value)


# --------------------------------------------------------------------------- #
# 2. registration — the registration IS the proof                               #
# --------------------------------------------------------------------------- #
def test_register_refuses_a_key_absent_from_the_cache_path(tmp_path, manifest):
    """⚠️ WIDE_FOV_BUILD.md §6.3's sharp finding: on the v2 path a registration
    nothing resolves to is INERT. ``corpus_key_of`` matches on the path, so a
    key that appears nowhere in it would never be found at train time."""
    cache = _make_v2_cache(tmp_path / "pai_wide120_v2png_train", CLIPS)
    with pytest.raises(parity.ParityViolation) as e:
        parity.register_v2_geometry_sibling(
            cache, new_key=SIB_KEY, geometry=GEOMETRY, source_key=SRC_KEY,
            manifest_path=manifest)
    assert "INERT" in str(e.value) and "Rename the cache dir" in str(e.value)


def test_register_refuses_overwriting_an_existing_parity_key(tmp_path, manifest):
    cache = _make_v2_cache(tmp_path / parity.PARITY_TRAIN_KEY, CLIPS)
    with pytest.raises(parity.ParityViolation) as e:
        parity.register_v2_geometry_sibling(
            cache, new_key=parity.PARITY_TRAIN_KEY, geometry=GEOMETRY,
            source_key=SRC_KEY, manifest_path=manifest)
    assert "EXISTING registered corpus key" in str(e.value)


def test_register_refuses_a_cache_whose_membership_is_wrong(tmp_path, manifest,
                                                            clip_list):
    """No entry may exist for an unproven cache."""
    cache = _make_v2_cache(tmp_path / SIB_KEY, CLIPS[:-1] + ["ff-foreign"])
    with pytest.raises(parity.ParityViolation) as e:
        parity.register_v2_geometry_sibling(
            cache, new_key=SIB_KEY, geometry=GEOMETRY, source_key=SRC_KEY,
            expect_clips=clip_list, manifest_path=manifest)
    assert "extra      : 1" in str(e.value)


def test_registered_entry_records_kind_lineage_and_geometry(tmp_path, manifest,
                                                            clip_list):
    cache, manifest = _register(tmp_path, manifest,
                                [c for i, c in enumerate(CLIPS)
                                 if i not in SKIP_INDICES], clip_list)
    ent = json.loads(manifest.read_text(encoding="utf-8"))["corpora"][SIB_KEY]
    assert ent["uid_kind"] == parity.V2_UID_KIND
    assert ent["episode_count"] == len(CLIPS) - len(SKIP_INDICES)
    assert ent["provenance"]["derived_from"] == SRC_KEY
    assert ent["provenance"]["geometry"]["width"] == 640
    # 🔒 the entry must carry digests, never ids
    blob = json.dumps(ent)
    assert not any(c in blob for c in CLIPS)


# --------------------------------------------------------------------------- #
# 3. the trainer-facing guard — both directions, reason asserted                #
# --------------------------------------------------------------------------- #
def test_guard_accepts_the_registered_cache(tmp_path, manifest, clip_list):
    cache, manifest = _register(tmp_path, manifest,
                                [c for i, c in enumerate(CLIPS)
                                 if i not in SKIP_INDICES], clip_list)
    rec = parity.assert_v2_parity_cache(cache, label="v2-cache", require=True,
                                        manifest_path=manifest)
    assert rec["parity"] and rec["corpus_key"] == SIB_KEY
    assert rec["episodes_loaded"] == len(CLIPS) - len(SKIP_INDICES)
    assert "MATCHES" in rec["content_check"]


def test_guard_refuses_a_dropped_clip(tmp_path, manifest, clip_list):
    cache, manifest = _register(tmp_path, manifest, CLIPS, clip_list)
    victim = next(cache.glob(f"*{parity.V2_SUFFIX}"))
    victim.unlink()
    with pytest.raises(parity.ParityViolation) as e:
        parity.assert_v2_parity_cache(cache, label="v2-cache", require=True,
                                      manifest_path=manifest)
    msg = str(e.value)
    assert "TRUNCATED by 1" in msg and "MISMATCH" in msg


def test_guard_refuses_a_swapped_clip_at_identical_count(tmp_path, manifest,
                                                         clip_list):
    """⭐⭐ THE HEADLINE. Drop one clip, add one foreign clip: the count is
    IDENTICAL and only a membership check can see it. This is why the wide-FOV
    census recomputed the corpus key instead of counting clips, and why a
    count-based guard here would have been a guard that cannot fail."""
    cache, manifest = _register(tmp_path, manifest, CLIPS, clip_list)
    before = len(list(cache.glob(f"*{parity.V2_SUFFIX}")))
    next(cache.glob(f"*{parity.V2_SUFFIX}")).unlink()
    (cache / f"ff-foreign{parity.V2_SUFFIX}").touch()
    assert len(list(cache.glob(f"*{parity.V2_SUFFIX}"))) == before   # count OK!
    with pytest.raises(parity.ParityViolation) as e:
        parity.assert_v2_parity_cache(cache, label="v2-cache", require=True,
                                      manifest_path=manifest)
    msg = str(e.value)
    assert "count OK — MEMBERSHIP DIFFERS AT THE SAME COUNT" in msg
    assert "a COUNT alone cannot catch this class" in msg


def test_guard_refuses_a_reselected_split_of_the_same_size(tmp_path, manifest,
                                                           clip_list):
    """A wholly different selection of the same cardinality."""
    cache, manifest = _register(tmp_path, manifest, CLIPS, clip_list)
    for p in cache.glob(f"*{parity.V2_SUFFIX}"):
        p.unlink()
    for i in range(len(CLIPS)):
        (cache / f"re{i:02d}-sel{parity.V2_SUFFIX}").touch()
    with pytest.raises(parity.ParityViolation) as e:
        parity.assert_v2_parity_cache(cache, label="v2-cache", require=True,
                                      manifest_path=manifest)
    assert "MEMBERSHIP DIFFERS AT THE SAME COUNT" in str(e.value)


def test_guard_refuses_an_extra_clip(tmp_path, manifest, clip_list):
    cache, manifest = _register(tmp_path, manifest, CLIPS, clip_list)
    (cache / f"ff-foreign{parity.V2_SUFFIX}").touch()
    with pytest.raises(parity.ParityViolation) as e:
        parity.assert_v2_parity_cache(cache, label="v2-cache", require=True,
                                      manifest_path=manifest)
    assert "EXTRA 1" in str(e.value)


def test_guard_refuses_an_unregistered_cache_under_require(tmp_path, manifest):
    """⛔ THE ACTUAL v5 HOLE: the wide cache as built today references no
    registered key at all, so this is the state a v5 launch would start in."""
    cache = _make_v2_cache(tmp_path / "pai_wide120_v2png_train", CLIPS)
    with pytest.raises(parity.ParityViolation) as e:
        parity.assert_v2_parity_cache(cache, label="v2-cache", require=True,
                                      manifest_path=manifest)
    msg = str(e.value)
    assert "unregistered v2 cache" in msg
    assert "register_v2_sibling.py" in msg


def test_guard_passes_through_an_unregistered_cache_by_default(tmp_path,
                                                               manifest, capsys):
    """⚠️ NO DEFAULT MAY MOVE. ``physicalai-v2bal`` is a deliberate NON-parity
    9 000-clip corpus and one arm is training on it as this lands; without
    --require-parity it must still warn-and-proceed, exactly as before."""
    cache = _make_v2_cache(tmp_path / "physicalai-v2bal-4b7eeeac222d", CLIPS)
    rec = parity.assert_v2_parity_cache(cache, label="v2-cache",
                                        manifest_path=manifest)
    assert rec["parity"] is False and rec["checked"] is False
    assert "NON-PARITY v2 corpus" in capsys.readouterr().out


def test_guard_refuses_a_v2_cache_wearing_an_epcache_key(tmp_path, manifest):
    """A v2 directory named with a RAW corpus key. The uid spaces are not
    comparable, so passing would prove nothing — and the naive failure ("0
    episodes found") names the wrong cause."""
    cache = _make_v2_cache(tmp_path / SRC_KEY, CLIPS)
    with pytest.raises(parity.ParityViolation) as e:
        parity.assert_v2_parity_cache(cache, label="v2-cache", require=True,
                                      manifest_path=manifest)
    msg = str(e.value)
    assert "uid kind" in msg and parity.EPCACHE_UID_KIND in msg


def test_guard_refuses_the_same_clip_in_two_dirs(tmp_path, manifest, clip_list):
    """``--v2-cache`` is nargs='+' and the dirs are CONCATENATED, so a clip in
    both would contribute its windows twice and re-weight the corpus."""
    a = _make_v2_cache(tmp_path / "shard_a", CLIPS[:8])
    b = _make_v2_cache(tmp_path / "shard_b", CLIPS[6:])
    with pytest.raises(parity.ParityViolation) as e:
        parity.assert_v2_parity_cache([a, b], label="v2-cache",
                                      manifest_path=manifest)
    assert "appear in more than one --v2-cache dir" in str(e.value)


def test_guard_refuses_mixing_two_registered_corpora(tmp_path, manifest):
    a = _make_v2_cache(tmp_path / SIB_KEY, CLIPS[:6])
    b = _make_v2_cache(tmp_path / "synthetic-w100x256cyl-cccccccccccc",
                       CLIPS[6:])
    man = json.loads(manifest.read_text(encoding="utf-8"))
    for k, ids in ((SIB_KEY, CLIPS[:6]),
                   ("synthetic-w100x256cyl-cccccccccccc", CLIPS[6:])):
        man["corpora"][k] = {"corpus_key": k, "split": "train",
                             "episode_count": len(ids),
                             "uid_kind": parity.V2_UID_KIND,
                             "episode_uid_sha256": parity.uid_digest(ids)}
    _write_manifest(manifest, man)
    with pytest.raises(parity.ParityViolation) as e:
        parity.assert_v2_parity_cache([a, b], label="v2-cache",
                                      manifest_path=manifest)
    assert "DIFFERENT registered corpora" in str(e.value)


def test_guard_refuses_a_v2_entry_with_no_digest(tmp_path, manifest):
    """A hand-edited manifest must not create a corpus nothing can check."""
    cache = _make_v2_cache(tmp_path / SIB_KEY, CLIPS)
    man = json.loads(manifest.read_text(encoding="utf-8"))
    man["corpora"][SIB_KEY] = {"corpus_key": SIB_KEY, "split": "train",
                               "episode_count": len(CLIPS),
                               "uid_kind": parity.V2_UID_KIND,
                               "episode_uid_sha256": None}
    _write_manifest(manifest, man)
    with pytest.raises(parity.ParityViolation) as e:
        parity.assert_v2_parity_cache(cache, label="v2-cache",
                                      manifest_path=manifest)
    assert "NO clip-id digest" in str(e.value)


def test_every_v2_refusal_withholds_clip_ids(tmp_path, manifest, clip_list):
    """🔒 PhysicalAI-AV is gated-confidential. A refusal is written to a log
    that leaves the pod, so it may carry counts and digests and nothing else."""
    cache, manifest = _register(tmp_path, manifest, CLIPS, clip_list)
    next(cache.glob(f"*{parity.V2_SUFFIX}")).unlink()
    (cache / f"ff-foreign{parity.V2_SUFFIX}").touch()
    with pytest.raises(parity.ParityViolation) as e:
        parity.assert_v2_parity_cache(cache, label="v2-cache",
                                      manifest_path=manifest)
    msg = str(e.value)
    assert not any(c in msg for c in CLIPS), "a clip id leaked into a refusal"
    assert "ff-foreign" not in msg


# --------------------------------------------------------------------------- #
# 4. the raw path — unweakened, and its v2 absence made explicit                #
# --------------------------------------------------------------------------- #
def test_raw_guard_still_refuses_a_truncated_epcache(tmp_path):
    """⛔ NOT WEAKENED. The raw path's own headline case, re-asserted here so a
    future edit to §9 that softened §6 would fail in this file too."""
    ent = parity.manifest_entry(parity.PARITY_TRAIN_KEY)
    d = tmp_path / parity.PARITY_TRAIN_KEY
    d.mkdir()
    for u in list(ent["episode_uids"])[:-1]:
        (d / u).touch()
    with pytest.raises(parity.ParityViolation) as e:
        parity.assert_parity_corpus(d, label="t", require=True)
    assert "TRUNCATED by 1" in str(e.value)


def test_raw_guard_on_a_v2_dir_names_the_format_and_the_right_trainer(tmp_path):
    """The ``train_flagship_v4`` decision, made mechanical. v4 has NO v2 support
    and is not getting one speculatively — so when it is handed a v2 cache its
    refusal must NAME that, instead of the misleading "does not reference the
    canonical corpus" that sends the reader off to rename a directory."""
    cache = _make_v2_cache(tmp_path / "pai_wide120_v2png_train", CLIPS)
    with pytest.raises(parity.ParityViolation) as e:
        parity.assert_parity_corpus(cache, label="--train-cache", require=True)
    msg = str(e.value)
    assert "THIS IS A V2 COMPRESSED CACHE" in msg
    assert "--v2-cache" in msg and "train_flagship_v4" in msg


def test_raw_guard_names_v2_when_a_registered_dir_holds_v2_files(tmp_path):
    cache = _make_v2_cache(tmp_path / parity.PARITY_TRAIN_KEY, CLIPS)
    with pytest.raises(parity.ParityViolation) as e:
        parity.assert_parity_corpus(cache, label="--train-cache", require=True)
    assert "THIS IS A V2 COMPRESSED CACHE" in str(e.value)


# --------------------------------------------------------------------------- #
# 5. the committed manifest + the serialization it must match                   #
# --------------------------------------------------------------------------- #
def test_uid_digest_is_the_exporters_serialization():
    """``parity_split_export.py`` computes ``sha256('\\n'.join(sorted(ids)))``.
    The whole v2 check rests on ``uid_digest`` being byte-identical to it — if
    it were not, the committed digest could never match a live cache."""
    ids = ["c", "a", "b"]
    assert parity.uid_digest(ids) == hashlib.sha256(
        "a\nb\nc".encode()).hexdigest()


def test_committed_manifest_carries_clip_membership_for_both_splits():
    for key, n_clips in ((parity.PARITY_TRAIN_KEY, 2400),
                         (parity.PARITY_VAL_KEY, 600)):
        cm = parity.clip_membership_of(key)
        assert cm is not None, f"{key} has no clip_membership block"
        assert cm["n_clips"] == n_clips
        d = cm["clip_id_sha256_sorted"]
        assert isinstance(d, str) and len(d) == 64 and int(d, 16) >= 0


def test_clips_are_not_episodes():
    """⚠️ The correction that invalidates a check written against 2 400: the
    parity TRAIN split is 2 400 CLIPS of which 24 fail to decode, leaving 2 376
    EPISODES. Both numbers are in the manifest and they are not the same fact."""
    ent = parity.manifest_entry(parity.PARITY_TRAIN_KEY)
    cm = parity.clip_membership_of(parity.PARITY_TRAIN_KEY)
    assert ent["episode_count"] == 2376
    assert cm["n_clips"] == 2400
    assert cm["decode_failures"] == len(ent["skip_indices"]) == 24
    assert cm["n_clips"] - cm["decode_failures"] == ent["episode_count"]


def test_longest_match_cannot_change_legacy_resolution(tmp_path):
    """``corpus_key_of`` now breaks ties longest-key-first so a sibling under
    its parent's path resolves to the sibling. ⛔ That may not move any
    pre-existing answer — proven, not assumed: on every path where several of
    the three legacy keys match, longest-first and lexicographic agree."""
    legacy = sorted({parity.PARITY_TRAIN_KEY, parity.PARITY_VAL_KEY,
                     *parity.LEAKY_SPLIT_KEYS})
    for i, a in enumerate(legacy):
        for b in legacy[i:]:
            for path in (tmp_path / a / b, tmp_path / b / a):
                s = str(path.resolve()).replace("\\", "/")
                hits = [k for k in legacy if k in s]
                assert (sorted(hits)[0]
                        == sorted(hits, key=lambda k: (-len(k), k))[0])


#: The key V2_PARITY_ENFORCEMENT.md §7 tells a v5 launch to register. It
#: deliberately CONTAINS its parent's key so it reads as "e438721ae894,
#: re-cached at 120°/256x640 cylindrical" — which only resolves correctly
#: because of the longest-match tie-break.
V5_KEY = f"{parity.PARITY_TRAIN_KEY}-w120-256x640cyl"


def test_the_v5_runbook_key_resolves_to_the_sibling_not_the_parent(tmp_path,
                                                                   manifest):
    """⭐ Pins the exact key the runbook publishes. Under the old lexicographic
    rule this path resolves to the PARENT (a raw epcache corpus), and the guard
    would report a real registered cache as the wrong corpus."""
    man = json.loads(manifest.read_text(encoding="utf-8"))
    man["corpora"][V5_KEY] = {"corpus_key": V5_KEY, "split": "train",
                              "episode_count": 1,
                              "uid_kind": parity.V2_UID_KIND}
    _write_manifest(manifest, man)
    d = tmp_path / parity.PARITY_TRAIN_KEY / V5_KEY
    d.mkdir(parents=True)
    assert parity.corpus_key_of(d, manifest) == V5_KEY
    # ...and the OLD rule would have got it wrong — the reason the change
    # exists. Simulated over the SAME candidate set corpus_key_of uses (the
    # hardcoded keys UNION the manifest's), because over the manifest alone the
    # parent key is not even a candidate and the check would be vacuous.
    cand = ({parity.PARITY_TRAIN_KEY, parity.PARITY_VAL_KEY,
             *parity.LEAKY_SPLIT_KEYS}
            | set(json.loads(manifest.read_text(encoding="utf-8"))["corpora"]))
    s = str(d.resolve()).replace("\\", "/")
    matching = sorted(k for k in cand if k in s)
    assert parity.PARITY_TRAIN_KEY in matching and V5_KEY in matching
    assert matching[0] == parity.PARITY_TRAIN_KEY, \
        "the old lexicographic rule must resolve to the PARENT here, or this " \
        "test is not exercising the tie-break it exists for"


# --------------------------------------------------------------------------- #
# 6. the wiring — the guard must actually run, before the loader                #
# --------------------------------------------------------------------------- #
def _stub_v2_loader(monkeypatch, fn):
    """Put a STUB ``tanitad.data.v2_dataset`` in ``sys.modules``.

    ⚠️ Deliberately not ``importorskip("torchvision")``. The real module imports
    torchvision, which the dev box does not have — so an importorskip would make
    the two wiring tests SKIP exactly where they are most likely to be run, and a
    guard test that skips is a guard that cannot fail (class C13). The trainer's
    ``from tanitad.data.v2_dataset import build_v2_providers`` lives INSIDE the
    branch, so a stub module proves the ordering with no heavy dependency."""
    import types
    import tanitad.data as _td                                   # noqa: E402
    mod = types.ModuleType("tanitad.data.v2_dataset")
    mod.build_v2_providers = fn
    monkeypatch.setitem(sys.modules, "tanitad.data.v2_dataset", mod)
    monkeypatch.setattr(_td, "v2_dataset", mod, raising=False)
    return mod


def _parse_trainer_args(argv):
    """The REAL argparse Namespace ``train_flagship4b`` would build, without
    running ``train()``. Using the real parser is the point: a wiring test
    against a hand-built namespace proves nothing about the CLI."""
    import train_flagship4b as T4                                # noqa: E402
    holder: dict = {}
    real = T4.train
    T4.train = lambda a: holder.setdefault("a", a)
    try:
        T4.main(argv)
    finally:
        T4.train = real
    return holder["a"]


def test_require_parity_flag_defaults_off():
    """⛔ NO DEFAULT MAY MOVE: every command that exists today keeps behaving
    identically, and enforcement is opt-in."""
    assert _parse_trainer_args(["--out", "x", "--v2-cache", "/a"]) \
        .require_parity is False
    assert _parse_trainer_args(["--out", "x", "--cache-dirs", "/a"]) \
        .require_parity is False
    assert _parse_trainer_args(["--out", "x", "--v2-cache", "/a",
                                "--require-parity"]).require_parity is True


def test_trainer_v2_branch_refuses_before_the_loader_is_reached(tmp_path,
                                                               monkeypatch):
    """⭐ THE WIRING TEST. Before this change the v2 branch went straight to
    ``build_v2_providers`` with no check at all. The loader is booby-trapped
    here: if the guard did not run FIRST this fails loudly, so the test cannot
    pass by the guard merely existing somewhere unreachable."""
    import train_flagship4b as T4                                # noqa: E402

    reached = {"loader": False}

    def _boom(*a, **k):
        reached["loader"] = True
        raise AssertionError("build_v2_providers was REACHED despite an "
                             "unregistered cache under --require-parity")

    cache = _make_v2_cache(tmp_path / "pai_wide120_v2png_train", CLIPS)
    args = _parse_trainer_args(["--out", str(tmp_path / "out"), "--v2-cache",
                                str(cache), "--require-parity",
                                "--config", "smoke"])
    _stub_v2_loader(monkeypatch, _boom)
    monkeypatch.setattr(T4, "start_cache_guard", lambda *a, **k: None)
    with pytest.raises(parity.ParityViolation) as e:
        T4.train(args)
    assert reached["loader"] is False
    assert "unregistered v2 cache" in str(e.value)


def test_trainer_v2_branch_still_proceeds_on_a_non_parity_corpus(tmp_path,
                                                                monkeypatch):
    """The other direction of the same wiring: without --require-parity the
    guard warns and the loader IS reached — the ``physicalai-v2bal`` case, one
    arm of which is training as this lands."""
    import train_flagship4b as T4                                # noqa: E402

    seen: dict = {}

    def _stub(dirs, **k):
        seen["dirs"] = dirs
        raise RuntimeError("STOP-AFTER-GUARD")

    cache = _make_v2_cache(tmp_path / "physicalai-v2bal-4b7eeeac222d", CLIPS)
    args = _parse_trainer_args(["--out", str(tmp_path / "out2"), "--v2-cache",
                                str(cache), "--config", "smoke"])
    _stub_v2_loader(monkeypatch, _stub)
    monkeypatch.setattr(T4, "start_cache_guard", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="STOP-AFTER-GUARD"):
        T4.train(args)
    assert seen["dirs"] == [str(cache)]
