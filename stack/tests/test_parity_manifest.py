"""Parity-corpus INTEGRITY — the regression suite for ``tanitad/data/parity.py``.

The hole this closes (MEASURED, code audit 2026-07-25): the ONLY parity
enforcement in any Python trainer was a path SUBSTRING match
(``train_flagship_v4._assert_parity``: ``if PARITY_KEY not in tc: raise``). No
trainer asserted ``episode_count == 2376`` and none recomputed a content hash,
so a TRUNCATED corpus — the known ``/workspace`` MooseFS-quota failure mode this
program has already hit — sits in a correctly-named directory and trains
SILENTLY, voiding every cross-arm comparison off it, invisibly.

The headline test is :func:`test_truncated_cache_is_refused` and its per-trainer
mirrors below: a synthetic parity cache with episodes dropped MUST be refused,
loud and before any GPU work.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from tanitad.data import parity                                  # noqa: E402

KEY = parity.PARITY_TRAIN_KEY
VAL_KEY = parity.PARITY_VAL_KEY


# --------------------------------------------------------------------------- #
# fixtures — a synthetic epcache with the CANONICAL uid set                     #
# --------------------------------------------------------------------------- #
def _make_cache(dirpath: Path, uids, skips=()) -> Path:
    """A cache dir of empty ``ep_*.pt`` markers. The guard runs on FILENAMES and
    must fire BEFORE any tensor is unpickled — empty files therefore prove the
    ordering too: if the guard did not run first the loader would die inside
    ``torch.load``, not with a :class:`ParityViolation`."""
    dirpath.mkdir(parents=True, exist_ok=True)
    for u in uids:
        (dirpath / u).touch()
    for i in skips:
        (dirpath / f"skip_{i:05d}").write_text("synthetic")
    return dirpath


@pytest.fixture(scope="module")
def canon_uids():
    ent = parity.manifest_entry(KEY)
    assert ent is not None, "committed manifest has no parity train entry"
    return list(ent["episode_uids"])


@pytest.fixture(scope="module")
def good_root(tmp_path_factory, canon_uids):
    """A cache root holding the FULL, canonical parity train split."""
    root = tmp_path_factory.mktemp("epcache_good")
    _make_cache(root / KEY, canon_uids,
                skips=parity.manifest_entry(KEY)["skip_indices"])
    return root


@pytest.fixture(scope="module")
def truncated_root(tmp_path_factory, canon_uids):
    """The failure mode: the quota filled at episode 1200 and the build stopped.
    The directory KEEPS its canonical name — which is all the old check looked
    at."""
    root = tmp_path_factory.mktemp("epcache_truncated")
    _make_cache(root / KEY, canon_uids[:1200])
    return root


# --------------------------------------------------------------------------- #
# 1. the committed manifest is internally consistent                            #
# --------------------------------------------------------------------------- #
def test_manifest_is_self_consistent(canon_uids):
    ent = parity.manifest_entry(KEY)
    assert ent["episode_count"] == parity.PARITY_TRAIN_EPISODES == 2376
    assert len(canon_uids) == 2376 and len(set(canon_uids)) == 2376
    assert parity.uid_digest(canon_uids) == ent["episode_uid_sha256"]


def test_manifest_uid_set_reproduces_the_24_clip_skipset(canon_uids):
    """The 2 376 present indices + the 24 recorded skips must tile the 2 400
    ordered train sources exactly (``parity_skipset.sh``: ``len(train) == 2400``,
    24 skips). Endpoints 1798/1941 are the values written independently into
    ``scripts/rebuild_pai_rolling.py`` (``--skip-idx 1798,1835,…,1941``)."""
    ent = parity.manifest_entry(KEY)
    idx = {parity.episode_index(u) for u in canon_uids}
    skips = set(ent["skip_indices"])
    assert len(skips) == 24
    assert idx | skips == set(range(2400))
    assert not (idx & skips)
    assert (min(skips), max(skips)) == (1798, 1941)


def test_val_entry_is_count_only_and_says_so():
    """Honesty check: no committed artifact enumerates the val uid set, so the
    val entry carries the MEASURED count (600) and an explicit ``None`` digest —
    never an invented hash."""
    ent = parity.manifest_entry(VAL_KEY)
    assert ent["episode_count"] == parity.PARITY_VAL_EPISODES == 600
    assert ent["episode_uid_sha256"] is None
    assert "unrecorded" in ent["uid_source"]


def test_manifest_schema_and_digest_definition_are_pinned():
    man = parity.load_manifest()
    assert man["schema"] == parity.MANIFEST_SCHEMA == "tanitad.parity_manifest/1"
    # the canonical serialization — changing it invalidates every manifest
    assert parity.uid_digest(["b", "a"]) == parity.uid_digest(["a", "b"])
    import hashlib
    assert parity.uid_digest(["a", "b"]) == hashlib.sha256(
        b"a\nb").hexdigest()


# --------------------------------------------------------------------------- #
# 2. ⭐ THE REGRESSION — a truncated cache is REFUSED                            #
# --------------------------------------------------------------------------- #
def test_full_canonical_cache_is_accepted(good_root):
    rec = parity.assert_parity_corpus(good_root / KEY, label="train_cache",
                                      require=True)
    assert rec["parity"] is True
    assert rec["episodes_loaded"] == rec["episodes_expected"] == 2376
    assert rec["episode_uid_sha256"] == rec["episode_uid_sha256_expected"]
    assert rec["skip_markers_present"] == 24


def test_truncated_cache_is_refused(truncated_root):
    """⭐ The whole point. A correctly-NAMED directory holding 1 200 of 2 376
    episodes passed the old substring check silently."""
    with pytest.raises(parity.ParityViolation) as ei:
        parity.assert_parity_corpus(truncated_root / KEY, label="train_cache",
                                    require=True)
    msg = str(ei.value)
    assert "PARITY VIOLATION" in msg
    assert "1200 loaded, 2376 expected" in msg
    assert "TRUNCATED by 1176" in msg
    assert "ep_01200.pt" in msg            # names the first missing episode


def test_the_old_substring_check_would_have_PASSED_the_truncated_cache(
        truncated_root):
    """Pins the hole itself: the pre-2026-07-25 rule (``PARITY_KEY in path``)
    is satisfied by the truncated cache. If this ever fails, the substring rule
    changed and this suite's premise needs revisiting."""
    tc = str((truncated_root / KEY).resolve()).replace("\\", "/")
    assert KEY in tc                       # <- the ENTIRE old enforcement


def test_substituted_episode_set_of_the_right_size_is_refused(canon_uids):
    """Count alone is not enough: 2 376 episodes drawn from a DIFFERENT selection
    must also be refused (the content check)."""
    swapped = canon_uids[:-1] + ["ep_09999.pt"]
    with pytest.raises(parity.ParityViolation) as ei:
        parity.check_uids(swapped, corpus_key=KEY, label="train_cache")
    msg = str(ei.value)
    assert "count OK" in msg and "MISMATCH" in msg
    assert "ep_09999.pt" in msg            # names the intruder


def test_extra_episodes_are_refused(canon_uids):
    with pytest.raises(parity.ParityViolation, match="EXTRA 1"):
        parity.check_uids(canon_uids + ["ep_02400.pt"], corpus_key=KEY,
                          label="train_cache")


def test_empty_split_dir_is_refused(tmp_path):
    d = tmp_path / KEY
    d.mkdir()
    with pytest.raises(parity.ParityViolation, match="0 loaded"):
        parity.assert_parity_corpus(d, label="train_cache")


def test_refusal_names_the_rebuild_and_reverify_commands(truncated_root):
    """A refusal that does not tell a 3 a.m. operator what to do costs a night."""
    with pytest.raises(parity.ParityViolation) as ei:
        parity.assert_parity_corpus(truncated_root / KEY, label="train_cache")
    msg = str(ei.value)
    assert "rebuild_pai_rolling.py" in msg
    assert "compute_skipset.py" in msg
    assert "MooseFS" in msg                # the measured root cause, named


# --------------------------------------------------------------------------- #
# 3. subset mode, non-parity corpora, leaked splits, the firewall               #
# --------------------------------------------------------------------------- #
def test_subset_mode_accepts_the_canonical_prefix_but_not_foreign_ids(canon_uids):
    """``dino_precompute --train-n N`` / ``--episodes N`` build a sorted PREFIX
    of the parity set by construction, so REF-A's feature dirs are legitimately
    short. Subset mode still refuses foreign or renumbered ids."""
    rec = parity.check_uids(canon_uids[:400], corpus_key=KEY, label="feat",
                            mode="subset")
    assert rec["episodes_loaded"] == 400
    with pytest.raises(parity.ParityViolation, match="NOT the canonical sorted prefix"):
        parity.check_uids(canon_uids[:399] + ["ep_09999.pt"], corpus_key=KEY,
                          label="feat", mode="subset")


def test_non_parity_corpus_warns_but_does_not_block(tmp_path, capsys):
    """Toy / comma2k19 / the v2 9 000-clip corpus must keep working."""
    d = _make_cache(tmp_path / "physicalai-v2-4b7eeeac222d", ["ep_00000.pt"])
    rec = parity.assert_parity_corpus(d, label="train_cache")
    assert rec["parity"] is False
    assert "NON-PARITY" in capsys.readouterr().out


def test_non_parity_corpus_is_refused_when_the_caller_requires_parity(tmp_path):
    d = _make_cache(tmp_path / "some-other-corpus", ["ep_00000.pt"])
    with pytest.raises(parity.ParityViolation, match="PARITY VIOLATION"):
        parity.assert_parity_corpus(d, label="train_cache", require=True)


def test_known_leaky_val_split_is_always_refused(tmp_path):
    """``physicalai-val-f1b378f295ae``: 78.5 % of its populated episodes are IN
    the parity train set (MEASURED 2026-07-25, MODEL_REGISTRY §Branch-B)."""
    d = _make_cache(tmp_path / "physicalai-val-f1b378f295ae", ["ep_00000.pt"])
    with pytest.raises(parity.ParityViolation) as ei:
        parity.assert_parity_corpus(d, label="val_cache")
    assert "LEAKED SPLIT" in str(ei.value)
    assert VAL_KEY in str(ei.value)        # points at the clean replacement


def test_parity_firewall_blocks_a_side_model_from_the_parity_corpus(tmp_path):
    parity.assert_not_parity(tmp_path / "comma2k19-train-abc", label="dynenc")
    with pytest.raises(parity.ParityViolation, match="PARITY FIREWALL"):
        parity.assert_not_parity(tmp_path / KEY, label="dynenc")


# --------------------------------------------------------------------------- #
# 4. ⭐ PER-TRAINER WIRING — every trainer family refuses the truncated cache    #
#    (each of these FAILED before 2026-07-25: no trainer had a content check)    #
# --------------------------------------------------------------------------- #
def test_flagship_v4_assert_parity_refuses_truncation(truncated_root, good_root):
    import train_flagship_v4 as T
    assert T.PARITY_KEY == KEY and T.PARITY_SKIP_HASH == "f09e44db"
    with pytest.raises(parity.ParityViolation, match="TRUNCATED"):
        T._assert_parity(str(truncated_root / KEY), str(good_root / KEY))
    prov = T._assert_parity(str(good_root / KEY), str(good_root / KEY))
    assert prov["train_corpus_key"] == KEY
    assert prov["episodes_verified"] == 2376


def test_flagship4b_cache_split_refuses_truncation(truncated_root):
    """``_cache_split`` globs ``*train*`` under a root — the guard must fire
    before ``load_episode`` touches a byte."""
    import train_flagship4b as T4
    with pytest.raises(parity.ParityViolation, match="TRUNCATED"):
        T4._cache_split(truncated_root, "train", 0)


def test_refb_loader_refuses_truncation(truncated_root):
    """``refb_train.load_cached_episodes`` — also REF-C's loader (``refc_train``
    imports it)."""
    import refb_train
    with pytest.raises(parity.ParityViolation, match="TRUNCATED"):
        refb_train.load_cached_episodes(str(truncated_root), "*train*")


def test_refa_feature_loader_refuses_foreign_episodes(tmp_path, canon_uids):
    """REF-A's DINO feature dir is a legitimate PREFIX subset, so it runs in
    subset mode — which still refuses a renumbered / foreign feature cache."""
    import refa_train
    root = tmp_path / "feats"
    _make_cache(root / f"{KEY}-dinov2-b14", canon_uids[:99] + ["ep_09999.pt"])
    with pytest.raises(parity.ParityViolation, match="NOT the canonical sorted prefix"):
        refa_train.load_feature_episodes(str(root), "*train*")


def test_finetune_traj_loader_refuses_truncation(truncated_root):
    import finetune_traj
    with pytest.raises(parity.ParityViolation, match="TRUNCATED"):
        finetune_traj.load_parity_cache(str(truncated_root / KEY))


def test_v15_v16_label_caches_check_their_eid_lists(canon_uids):
    """v1.5 / v1.6 never glob the epcache — they consume an ``eids`` list baked
    into a label cache. Same manifest, same refusal."""
    import train_flagship_v15 as V15
    import train_flagship_v16 as V16
    for mod in (V15, V16):
        assert mod.assert_eids_parity(canon_uids, label="v15/v16")["parity"] is True
        with pytest.raises(parity.ParityViolation, match="TRUNCATED"):
            mod.assert_eids_parity(canon_uids[:1200], label="v15/v16")


def test_train_worldmodel_cached_path_refuses_truncation(truncated_root):
    """The base ``--data cached`` builder (``tanitad/train/train_worldmodel.py``)
    — the bake-off arm path, a THIRD copy of the same glob."""
    from tanitad.config import smoke_config
    from tanitad.train.train_worldmodel import _build_datasets
    with pytest.raises(parity.ParityViolation, match="TRUNCATED"):
        _build_datasets(smoke_config(), 0, "cached", str(truncated_root))


def test_idm_probe_family_refuses_truncation(truncated_root):
    """Every ``run_idm_*`` / ``run_v1_encoder_char`` / ``run_branchb_transfer`` /
    ``run_camcond_ablation`` probe routes through ``select_episodes``, whose
    ``if not p.exists(): continue`` silently absorbed a truncated cache."""
    import run_idm_proof
    with pytest.raises(parity.ParityViolation, match="TRUNCATED"):
        run_idm_proof.select_episodes({}, str(truncated_root / KEY), 1, 1)


def test_refc_v12_shard_cache_refuses_a_truncated_source():
    """REF-C v1.2 trains on a DISTILLED shard cache; the check it can still make
    in-process is that its recorded source was the full parity corpus."""
    import refc_v12_train as V12
    good = V12.check_shard_cache_parity(
        {"src": f"/workspace/pai_epcache/{KEY}", "episodes": 2376})
    assert good["parity"] is True and good["episodes_expected"] == 2376
    with pytest.raises(parity.ParityViolation, match="was truncated"):
        V12.check_shard_cache_parity(
            {"src": f"/workspace/pai_epcache/{KEY}", "episodes": 1200})


def test_missing_split_dir_stays_an_AssertionError_not_a_refusal(tmp_path):
    """REGRESSION: ``refa_train`` / ``refb_train`` ``except AssertionError``
    around their OPTIONAL val-metrics block. If ``resolve_split_dir`` raised
    ParityViolation (a SystemExit) for "no val dir", a finished 30 k run would
    die at the metrics write. "You gave me no val dir" is not a parity
    violation."""
    with pytest.raises(AssertionError, match="no cache dir matching"):
        parity.resolve_split_dir(tmp_path, "*val*")


def test_dynamics_encoder_keeps_its_parity_firewall(tmp_path):
    """``train_dynamics_encoder`` claims in its own docstring never to touch the
    parity corpus. That claim is now an assertion."""
    import train_dynamics_encoder as TDE
    TDE.assert_side_model_firewall(str(tmp_path / "comma2k19-train-abc"), None)
    with pytest.raises(parity.ParityViolation, match="PARITY FIREWALL"):
        TDE.assert_side_model_firewall(str(tmp_path / KEY), None)


# --------------------------------------------------------------------------- #
# 5. the manifest generator                                                     #
# --------------------------------------------------------------------------- #
def test_lake_filtering_skipset_is_index_reproducible_from_the_repo():
    """``filtering.CORRUPT_SKIPSET`` is (still) empty of clip ids — PhysicalAI-AV
    is tier ``firewalled`` and its UUIDs are recipe-only. What the repo CAN now
    reproduce is the skipset at INDEX level, from the committed manifest."""
    from tanitad.lake import filtering as FL
    assert FL.PARITY_SKIP_INDICES == tuple(
        parity.manifest_entry(KEY)["skip_indices"])
    assert len(FL.PARITY_SKIP_INDICES) == 24
    assert FL.PARITY_SKIP_KEY == parity.PARITY_SKIP_HASH == "f09e44db"
    assert FL.STRICT_PARITY_BUILD_KEY in KEY


def test_generator_refuses_to_overwrite_a_recorded_digest(tmp_path, canon_uids):
    """A truncated cache must never be able to quietly re-record itself into a
    passing manifest."""
    import make_parity_manifest as G
    man_p = tmp_path / "m.json"
    man_p.write_text(json.dumps(parity.load_manifest()), encoding="utf-8")
    man = json.loads(man_p.read_text(encoding="utf-8"))
    short = _make_cache(tmp_path / KEY, canon_uids[:10])
    with pytest.raises(SystemExit, match="ALREADY carries a uid digest"):
        G.record(man, short, "train", None, force=False)
    out = G.record(man, short, "train", None, force=True)
    assert out["corpora"][KEY]["episode_count"] == 10     # --force is explicit


def test_generator_cross_checks_reject_a_tampered_profile(tmp_path):
    """``--from-profile-csv`` must refuse a scan that disagrees with the other
    committed artifacts (skip endpoints / frame total / episode count)."""
    import make_parity_manifest as G
    csv_p = tmp_path / "bad.csv"
    rows = ["file,episode_id,T_out"] + [f"ep_{i:05d}.pt,{i},199" for i in range(10)]
    csv_p.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="cross-checks failed"):
        G.from_profile_csv(csv_p, {"corpora": {}})
