"""⛔⛔ A TRAINING RUN MUST REFUSE A CACHE THAT CONTAINS v7.2 EVAL CLIPS.

⭐⭐ WHY THIS FILE EXISTS. MEASURED 2026-09-23: `refc_v3_train.py`'s `--v2-cache` branch hands
the directory straight to `build_v2_providers`, which loads **every** `*.v2ep.pt` in it. The only
check before that is the PARITY membership guard, which does not refuse by default — so the
path had **no eval exclusion at all**. The refcv6 corpus cache on Thor
(`physicalai-b1-w120-416x1024cyl`, 4,713 clips) holds **141 v7.2 EVAL clips** beside its 4,572
train clips; a launch pointed at it would have trained on its own evaluation set, and every
eval-139 number that arm ever reported would have been void — silently, with a green run.

⭐ THE CLASSIFIER IS INDEPENDENT OF THE THING IT GUARDS. It reads the cache's FILENAMES and
checks them against the banked eval digest set (`parity.v72_eval_clip_digests`, sha256 of each
eval clip id, 147 entries) — it does not ask the loader, the dataset, or anything the training
path itself computes.

⛔ CPU only; opens no episode.
"""
from __future__ import annotations

import importlib.util
import os
import pathlib

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_TRAINER_PY = os.path.join(_HERE, "..", "scripts", "refc_v3_train.py")
#: the gated eval-139 cache on this box -- EVERY file in it is an eval clip
_EVAL139 = pathlib.Path("D:/Projects/TanitAD-artifacts/v2ep-eval139-416x1024cyl")
#: ⛔ LITERAL: the eval-139 cache holds 139 clips, all members of the v7.2 eval split
EXPECT_EVAL139 = 139


def _trainer():
    spec = importlib.util.spec_from_file_location("rt_evalx", _TRAINER_PY)
    T = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)
    return T


def _eval_ids(n: int) -> list[str]:
    """Real eval clip ids, recovered from the eval-139 cache's filenames."""
    if not _EVAL139.is_dir():
        pytest.skip("the eval-139 cache is not on this box")
    ids = sorted(f.name[: -len(".v2ep.pt")] for f in _EVAL139.glob("*.v2ep.pt"))
    return ids[:n]


def _fake_cache(root: pathlib.Path, names) -> pathlib.Path:
    root.mkdir(parents=True, exist_ok=True)
    for n in names:
        (root / f"{n}.v2ep.pt").write_bytes(b"x")
    return root


def test_the_classifier_finds_EVERY_eval_clip_in_the_eval_cache() -> None:
    """⭐ THE POWERED CONTROL. A classifier that finds nothing on a directory made entirely of
    eval clips cannot be trusted to find one in a train cache."""
    if not _EVAL139.is_dir():
        pytest.skip("the eval-139 cache is not on this box")
    assert len(_trainer()._eval_clips_in_v2_cache(str(_EVAL139))) == EXPECT_EVAL139


def test_a_train_cache_with_NO_eval_clips_is_clean(tmp_path) -> None:
    hits = _trainer()._eval_clips_in_v2_cache(
        _fake_cache(tmp_path / "train", ["not-a-clip-0000", "not-a-clip-0001"]))
    assert hits == []


def test_ONE_eval_clip_hidden_among_train_clips_is_found(tmp_path) -> None:
    """The realistic contamination: a single eval clip in an otherwise clean directory."""
    ev = _eval_ids(1)
    hits = _trainer()._eval_clips_in_v2_cache(
        _fake_cache(tmp_path / "mixed", ["not-a-clip-0000", "not-a-clip-0001"] + ev))
    assert hits == ev


def test_the_trainer_REFUSES_a_contaminated_cache(tmp_path) -> None:
    """⛔ THE GUARD ITSELF: calls the trainer's real refusal, not a copy of it.

    The first draft asserted the SOURCE ORDER and called the classifier -- and a mutation
    that disabled the refusal's `if` would have left both green. `refuse_eval_clips_in_train`
    is the function `train()` calls, so this cannot drift from what runs."""
    import argparse
    T = _trainer()
    cache = _fake_cache(tmp_path / "mixed", ["not-a-clip-0000"] + _eval_ids(2))
    with pytest.raises(SystemExit) as e:
        T.refuse_eval_clips_in_train(str(cache), argparse.Namespace())
    assert "EVAL" in str(e.value)


def test_a_clean_train_cache_is_NOT_refused(tmp_path) -> None:
    """⭐ The guard must not refuse everything -- a clean cache passes and returns no hits."""
    import argparse
    T = _trainer()
    cache = _fake_cache(tmp_path / "clean", ["not-a-clip-0000", "not-a-clip-0001"])
    assert T.refuse_eval_clips_in_train(str(cache), argparse.Namespace()) == []


def test_the_named_override_lets_a_contaminated_cache_through(tmp_path) -> None:
    import argparse
    T = _trainer()
    ev = _eval_ids(1)
    cache = _fake_cache(tmp_path / "mixed", ["not-a-clip-0000"] + ev)
    got = T.refuse_eval_clips_in_train(
        str(cache), argparse.Namespace(allow_eval_clips_in_train=True))
    assert got == ev


def test_the_trainer_CALLS_the_refusal_before_building_providers() -> None:
    """The call site: the refusal sits between the parity check and the provider build."""
    src = open(_TRAINER_PY, encoding="utf-8").read()
    i_par = src.index("v2_parity = parity.assert_v2_parity_cache(")
    i_grd = src.index("refuse_eval_clips_in_train(args.v2_cache, args)")
    i_bld = src.index("eps = build_v2_providers(args.v2_cache, lru_size=args.v2_lru)")
    assert i_par < i_grd < i_bld


def test_the_override_is_a_NAMED_flag_and_defaults_off() -> None:
    T = _trainer()
    P = T.build_parser()
    assert P.parse_args(["--arm", "hier", "--out", "z"]).allow_eval_clips_in_train is False
    assert P.parse_args(["--arm", "hier", "--out", "z",
                         "--allow-eval-clips-in-train"]).allow_eval_clips_in_train is True
