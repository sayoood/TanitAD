"""⛔ ``PREFLIGHT: OK`` MUST BE ABLE TO SAY ``BLOCKED`` — the parity leg.

THE DEFECT, MEASURED (2026-07-27, ``V5_EVALUABLE.md`` §9.2)
------------------------------------------------------------
``train_flagship_v4.py --print-launch --require-parity`` printed

    PREFLIGHT: OK

against a v2 cache directory whose ``parity.corpus_key_of`` resolves to
**``None``** — i.e. against exactly the unregistered cache ``--require-parity``
exists to refuse. Every check in ``preflight_asserts`` was ARGUMENT-level: it
verified that the *flag was present*, never that the *cache passes*. The guard
that can actually refuse (``parity.assert_v2_parity_cache``) runs inside
``train()``, which is *after* the orchestrator has launched.

That is the C13 shape this program has shipped several times — a guard that
cannot fail. It is worse than no guard, because ``PREFLIGHT: OK`` is what the
orchestrator reads before spending a GPU-week.

WHAT IS PINNED HERE — BOTH DIRECTIONS, AND THE THIRD STATE
-----------------------------------------------------------
* a REGISTERED cache **passes** (a guard that refuses everything is equally
  useless — it just fails in the other direction);
* an UNREGISTERED cache is **refused**, through ``main(--print-launch)``, with
  ``PREFLIGHT: BLOCKED`` and **exit 2** — the end-to-end shape the orchestrator
  actually reads;
* a cache whose CLIP SET does not match the manifest is refused (registered name,
  wrong membership);
* a cache that is **not on this host** is refused, because a ``PREFLIGHT: OK``
  that could not run its own check is indistinguishable from one that ran it and
  passed;
* nothing moves when ``--require-parity`` is absent — that flag stays opt-in.

🔒 Every clip id here is synthetic. The real corpus's clip ids are
gated-confidential and appear in no fixture.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from tanitad.data import parity                                   # noqa: E402

import train_flagship_v4 as T                                     # noqa: E402

# A synthetic registered key. It must NOT contain `physicalai-train-e438721ae894`
# or `physicalai-val-0c5f7dac3b11`, which `corpus_key_of` always knows about.
SYNTH_TRAIN_KEY = "synthcorpus-train-0000deadbeef-w120-256x640cyl"
SYNTH_VAL_KEY = "synthcorpus-val-0000cafebabe-w120-256x640cyl"


def _clips(n, tag):
    return [f"{tag}{i:08d}" for i in range(n)]


def _v2_dir(root: Path, name: str, clip_ids) -> Path:
    """A v2 cache directory. ``v2_clip_ids`` reads FILE NAMES only, so the
    payloads need not exist — and deliberately do not, so this fixture cannot
    drift into testing the decoder."""
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    for c in clip_ids:
        (d / f"{c}{parity.V2_SUFFIX}").write_bytes(b"")
    return d


def _manifest(path: Path, entries: dict) -> Path:
    """A manifest in the committed schema, carrying only synthetic corpora."""
    doc = {"schema": parity.MANIFEST_SCHEMA, "notes": "synthetic — tests only",
           "corpora": {}}
    for key, clips in entries.items():
        doc["corpora"][key] = {
            "corpus_key": key,
            "split": "val" if "-val-" in key else "train",
            "episode_count": len(clips),
            "uid_kind": parity.V2_UID_KIND,
            "uid_source": "synthetic",
            "episode_uid_sha256": parity.uid_digest(clips),
            "skip_indices": [], "skip_count": 0,
        }
    path.write_text(json.dumps(doc, indent=2))
    return path


def _argv(tr, va, *, require_parity=True):
    argv = ["--out", "o", "--v2-train-cache", str(tr), "--v2-val-cache", str(va),
            "--frame-h", "256", "--frame-w", "640", "--frame-hfov", "120",
            "--projection", "cylindrical", "--v2-subframe", "176x624",
            "--from-scratch"]
    if require_parity:
        argv.append("--require-parity")
    return argv


@pytest.fixture()
def registered(tmp_path, monkeypatch):
    """A registered train+val pair, plus the temp manifest that registers them.

    ``MANIFEST_PATH`` is monkeypatched (rather than threading a path through
    ``main``) because the end-to-end legs below drive ``main(argv)``, which is
    the surface the orchestrator actually reads."""
    tc, vc = _clips(6, "trn"), _clips(3, "val")
    tr = _v2_dir(tmp_path, SYNTH_TRAIN_KEY, tc)
    va = _v2_dir(tmp_path, SYNTH_VAL_KEY, vc)
    mp = _manifest(tmp_path / "synth_manifest.json",
                   {SYNTH_TRAIN_KEY: tc, SYNTH_VAL_KEY: vc})
    monkeypatch.setattr(parity, "MANIFEST_PATH", mp)
    parity._MANIFEST_CACHE.pop(str(mp), None)
    return tr, va, mp, tc, vc


# =========================================================================== #
# 1. ⭐ GREEN — a registered cache passes                                       #
# =========================================================================== #
def test_a_REGISTERED_v2_cache_PASSES_the_parity_preflight(registered):
    tr, va, _, _, _ = registered
    a = T.build_parser().parse_args(_argv(tr, va))
    assert T.preflight_parity_problems(a) == []
    assert not [p for p in T.preflight_asserts(a)
                if p.startswith("[PARITY-PREFLIGHT]")]


def test_main_print_launch_prints_OK_and_exits_0_on_a_registered_cache(
        registered, capsys):
    tr, va, _, _, _ = registered
    rc = T.main(_argv(tr, va) + ["--print-launch"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "PREFLIGHT: OK" in out
    # the guard did not merely stay silent — it RAN and said so
    assert "v2 VERIFIED" in out


# =========================================================================== #
# 2. ⛔ RED — the exact reported defect, end to end                             #
# =========================================================================== #
def test_main_print_launch_is_BLOCKED_and_exits_2_on_an_UNREGISTERED_cache(
        tmp_path, monkeypatch, capsys):
    """THE MEASURED DEFECT: this exact command printed ``PREFLIGHT: OK``.

    The directory is real, populated and readable — it simply carries a name no
    registered key resolves. That is the pre-rename state of v5's own train
    cache (``pai_wide120_v2png_train``), verbatim."""
    tc, vc = _clips(6, "trn"), _clips(3, "val")
    tr = _v2_dir(tmp_path, "pai_wide120_v2png_train", tc)     # NOT a key
    va = _v2_dir(tmp_path, SYNTH_VAL_KEY, vc)
    mp = _manifest(tmp_path / "m.json", {SYNTH_TRAIN_KEY: tc, SYNTH_VAL_KEY: vc})
    monkeypatch.setattr(parity, "MANIFEST_PATH", mp)
    parity._MANIFEST_CACHE.pop(str(mp), None)

    rc = T.main(_argv(tr, va) + ["--print-launch"])
    out = capsys.readouterr().out
    assert rc == 2, out
    assert "PREFLIGHT: BLOCKED" in out
    assert "PREFLIGHT: OK" not in out
    hit = [ln for ln in out.splitlines() if "[PARITY-PREFLIGHT]" in ln]
    assert hit, out
    # the refusal must carry the remedy, not just the complaint
    assert "register_v2_sibling.py" in out
    assert "DIRECTORY NAME" in out


def test_the_UNREGISTERED_cache_is_the_ONLY_thing_that_changed(tmp_path,
                                                              monkeypatch):
    """Rename the same directory to the registered key and the refusal goes away.

    ⚠️ Done with a real ``rename``, not a symlink: ``corpus_key_of`` calls
    ``Path.resolve()``, so a symlinked old name reads THROUGH to the new key and
    a negative control built on one is invalid (a sibling stream nearly recorded
    the rename guard as inert on exactly that mistake)."""
    tc, vc = _clips(6, "trn"), _clips(3, "val")
    bad = _v2_dir(tmp_path, "pai_wide120_v2png_train", tc)
    va = _v2_dir(tmp_path, SYNTH_VAL_KEY, vc)
    mp = _manifest(tmp_path / "m.json", {SYNTH_TRAIN_KEY: tc, SYNTH_VAL_KEY: vc})
    monkeypatch.setattr(parity, "MANIFEST_PATH", mp)
    parity._MANIFEST_CACHE.pop(str(mp), None)

    a_bad = T.build_parser().parse_args(_argv(bad, va))
    assert T.preflight_parity_problems(a_bad), "the unregistered name must refuse"

    good = bad.parent / SYNTH_TRAIN_KEY
    bad.rename(good)
    a_good = T.build_parser().parse_args(_argv(good, va))
    assert T.preflight_parity_problems(a_good) == []


# =========================================================================== #
# 3. ⛔ a registered NAME is not enough — the membership is checked             #
# =========================================================================== #
def test_a_registered_name_with_the_WRONG_CLIP_SET_is_refused(registered):
    """The name resolves, the count/digest do not. This is the check that makes
    the directory name a claim rather than a password."""
    tr, va, _, tc, _ = registered
    (tr / f"{tc[0]}{parity.V2_SUFFIX}").unlink()          # one clip short
    a = T.build_parser().parse_args(_argv(tr, va))
    probs = T.preflight_parity_problems(a)
    assert probs and "[PARITY-PREFLIGHT] --v2-train-cache" in probs[0]


def test_the_VAL_cache_is_checked_too_not_only_the_train_one(registered,
                                                             tmp_path):
    """The val split is where the mid-run held-out gate reads, so a registered
    train cache beside an unregistered val cache must still refuse."""
    tr, _, _, _, vc = registered
    rogue = _v2_dir(tmp_path, "some_unregistered_val_dir", vc)
    a = T.build_parser().parse_args(_argv(tr, rogue))
    probs = T.preflight_parity_problems(a)
    assert probs and all("--v2-val-cache" in p for p in probs), probs


# =========================================================================== #
# 4. ⛔ the third state: the check could not run at all                         #
# =========================================================================== #
def test_preflight_REFUSES_when_the_cache_is_not_on_THIS_host(tmp_path):
    """``--print-launch`` is routinely typed on the dev box against pod paths.

    An OK printed there is an OK for a check that never executed — which is the
    same defect with a different cause, so it is a BLOCK with a named reason."""
    a = T.build_parser().parse_args(
        _argv("/workspace/data/does-not-exist-here",
              "/workspace/data/nor-this-one"))
    probs = T.preflight_parity_problems(a)
    assert len(probs) == 2, probs
    assert all("COULD NOT BE CHECKED" in p for p in probs)
    assert any("--v2-train-cache" in p for p in probs)
    assert any("--v2-val-cache" in p for p in probs)


# =========================================================================== #
# 5. nothing moves without the flag — --require-parity stays opt-in             #
# =========================================================================== #
def test_the_parity_preflight_does_NOT_run_without_require_parity(tmp_path,
                                                                  monkeypatch):
    tc, vc = _clips(6, "trn"), _clips(3, "val")
    bad = _v2_dir(tmp_path, "pai_wide120_v2png_train", tc)
    va = _v2_dir(tmp_path, "another_unregistered_val", vc)
    mp = _manifest(tmp_path / "m.json", {SYNTH_TRAIN_KEY: tc})
    monkeypatch.setattr(parity, "MANIFEST_PATH", mp)
    parity._MANIFEST_CACHE.pop(str(mp), None)

    a = T.build_parser().parse_args(_argv(bad, va, require_parity=False))
    probs = T.preflight_asserts(a)
    assert not [p for p in probs if p.startswith("[PARITY-PREFLIGHT]")]
    # the pre-existing opt-in reminder still fires, unchanged
    assert any("TRAINS ANYWAY" in p for p in probs)


def test_a_run_with_no_v2_cache_at_all_is_untouched():
    """The raw-epcache path must not acquire a v2 check it cannot satisfy."""
    a = T.build_parser().parse_args(["--print-launch", "--require-parity"])
    assert T.preflight_parity_problems(a) == []
