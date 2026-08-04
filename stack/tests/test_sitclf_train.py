"""The promoted situation-classifier trainer — and, mostly, its REFUSALS.

WHY THESE TESTS LOOK LIKE THIS. Earlier today I shipped `--heldout-off-reason` as a
"required reason string, not a bare --force" and it had **three latent defects**: a
whitespace reason unlocked the guard (degrading it to exactly the boolean it was designed
not to be), the reason never survived into the run record, and it was never echoed despite
the help text saying so. Every defect sat downstream of the argument parser, and my check
had stopped at the parser.

⇒ **RULE bought there and applied here: for any guard, test the BYPASS PATH, not the happy
path.** Pass the degenerate value and assert it is refused; then assert the artifact
actually records what it claims to.

The guard under test is the PI's binding ruling — *labels may use ego; inference is VISION
ONLY* — and it is the one most likely to be argued away, because on the banked numbers the
INADMISSIBLE arms score BETTER (`head_ego` 0.0697 > `head_img_ego` 0.0525 > `head_img`
0.0376). A rule that is only a docstring loses that argument at 2 a.m.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
TRAINER = REPO / "scripts" / "sitclf_train.py"


def _substrate(tmp_path: Path, n_clips: int = 8, per_clip: int = 40, dim: int = 12) -> Path:
    """A tiny synthetic substrate with the required keys and >1 clip per fold."""
    rng = np.random.default_rng(0)
    n = n_clips * per_clip
    clip_id = np.repeat(np.arange(n_clips), per_clip).astype(np.int64)
    img = rng.standard_normal((n, dim)).astype(np.float32)
    y = (rng.random((n, 3)) < 0.25).astype(np.float32)
    valid = np.ones((n, 3), dtype=np.float32)
    p = tmp_path / "sub.npz"
    np.savez_compressed(p, img=img, y=y, valid=valid, clip_id=clip_id)
    return p


def _run(*argv: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(TRAINER), *argv],
        capture_output=True, text=True, cwd=str(REPO),
        env={**__import__("os").environ, "PYTHONPATH": str(REPO)},
    )


# --------------------------------------------------------------------------- #
# THE RULING — the bypass path, not the happy path                             #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("feat", ["ego", "img_ego", "fused"])
def test_ego_at_inference_is_REFUSED_with_a_nonzero_exit(tmp_path, feat):
    """Not warned about. Not silently ignored. REFUSED, with a non-zero exit.

    These three are exactly the arms that score better on the banked numbers, which is
    why the refusal has to be mechanical. `argparse` `choices` rejects them before the
    body runs; this asserts the *behaviour*, so a later refactor that widens `choices`
    without widening the check fails here.
    """
    sub = _substrate(tmp_path)
    r = _run("--substrate", str(sub), "--out", str(tmp_path / "o"), "--features", feat)
    assert r.returncode != 0, f"--features {feat} must NOT be trainable"
    assert not (tmp_path / "o" / "ckpt.pt").exists(), "a refused arm must leave no ckpt"


def test_only_img_is_admissible_and_the_refusals_are_declared_not_forgotten():
    """The refused sets are named in the source, so they read as decisions rather than
    as features nobody got round to. A future `img_ego` must trip this test first."""
    import scripts.sitclf_train as T  # noqa: PLC0415

    assert T.ADMISSIBLE_FEATURES == ("img",)
    assert set(T.REFUSED_FEATURES) == {"ego", "img_ego", "fused"}
    for reason in T.REFUSED_FEATURES.values():
        assert reason.strip(), "every refusal must carry its reason"


def test_config_records_the_ruling_in_MACHINE_READABLE_form(tmp_path):
    """A claim that is only in prose cannot be checked by a script.

    This mirrors `RefCModel.goal_provenance()`: the arm declares what reaches inference,
    so an auditor greps `config.json` instead of reading a docstring and trusting it.
    """
    sub = _substrate(tmp_path)
    out = tmp_path / "o"
    r = _run("--substrate", str(sub), "--out", str(out), "--win", "4",
             "--epochs", "1", "--width", "8", "--batch", "64")
    assert r.returncode == 0, r.stderr
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["ego_at_inference"] is False
    assert cfg["inference_inputs"] == ["image_features"]
    assert cfg["deployable"] is True
    assert "VISION ONLY" in cfg["ruling"]


def test_an_ego_block_in_the_substrate_does_not_become_an_input(tmp_path):
    """A substrate may legitimately carry `ego` — labels are DERIVED from ego state.

    The failure this prevents is the quiet one: reusing a label-side substrate and
    letting its ego block slide into the model input because it was simply *there*.
    The config records that the block existed and that it was not used.
    """
    rng = np.random.default_rng(1)
    n_clips, per_clip, dim = 8, 40, 12
    n = n_clips * per_clip
    p = tmp_path / "sub_ego.npz"
    np.savez_compressed(
        p,
        img=rng.standard_normal((n, dim)).astype(np.float32),
        ego=rng.standard_normal((n, 3)).astype(np.float32),      # present on purpose
        y=(rng.random((n, 3)) < 0.25).astype(np.float32),
        valid=np.ones((n, 3), dtype=np.float32),
        clip_id=np.repeat(np.arange(n_clips), per_clip).astype(np.int64),
    )
    out = tmp_path / "o"
    r = _run("--substrate", str(p), "--out", str(out), "--win", "4",
             "--epochs", "1", "--width", "8", "--batch", "64")
    assert r.returncode == 0, r.stderr
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["substrate_has_ego_block"] is True, "the block was there and must be recorded"
    assert cfg["ego_at_inference"] is False, "...and must NOT have reached inference"
    assert cfg["in_dim"] == dim, (
        f"in_dim {cfg['in_dim']} != image dim {dim}: the ego block was concatenated in"
    )


# --------------------------------------------------------------------------- #
# The other guards                                                             #
# --------------------------------------------------------------------------- #
def test_width_must_be_a_multiple_of_the_head_count(tmp_path):
    """`CausalSitHead` fixes 4 attention heads. Solving a capacity budget over all
    integers silently proposes widths the module refuses to build; failing in argparse
    beats failing inside `nn.TransformerEncoderLayer` after the substrate is loaded."""
    sub = _substrate(tmp_path)
    r = _run("--substrate", str(sub), "--out", str(tmp_path / "o"),
             "--width", "10", "--epochs", "1")
    assert r.returncode != 0
    assert "multiple of" in (r.stderr + r.stdout)


def test_a_missing_substrate_fails_loud_rather_than_training_on_nothing(tmp_path):
    r = _run("--substrate", str(tmp_path / "nope.npz"), "--out", str(tmp_path / "o"))
    assert r.returncode != 0
    assert "not found" in (r.stderr + r.stdout)


def test_a_substrate_missing_required_keys_names_what_is_missing(tmp_path):
    p = tmp_path / "bad.npz"
    np.savez_compressed(p, img=np.zeros((10, 4), dtype=np.float32))
    r = _run("--substrate", str(p), "--out", str(tmp_path / "o"))
    assert r.returncode != 0
    assert "missing required keys" in (r.stderr + r.stdout)


def test_folds_are_cluster_disjoint_not_row_disjoint(tmp_path):
    """Splitting on rows would put frames of one clip on both sides and leak. The unit
    of independence is the clip cluster — the same unit the interval estimator resamples."""
    sub = _substrate(tmp_path)
    out = tmp_path / "o"
    r = _run("--substrate", str(sub), "--out", str(out), "--win", "4",
             "--epochs", "1", "--width", "8", "--batch", "64")
    assert r.returncode == 0, r.stderr
    cfg = json.loads((out / "config.json").read_text())
    z = np.load(out / "heldout_scores.npz")
    sub_z = np.load(sub)
    held = set(np.unique(z["clip_id"]).tolist())
    assert held, "held-out fold is empty"
    assert len(held) == cfg["n_clusters_heldout"]
    all_clips = set(np.unique(sub_z["clip_id"]).tolist())
    assert held < all_clips, "the held-out fold must be a strict subset of the clips"


def test_the_trainer_does_NOT_print_its_own_headline_metric(tmp_path):
    """A trainer that scores itself is how "v1.6 is best-in-program" reached a report
    from a TRAINER LOG that ran ~10 % optimistic against `eval_*.py`. This one emits raw
    held-out scores and stops; AP and its interval belong to the eval harness."""
    sub = _substrate(tmp_path)
    out = tmp_path / "o"
    r = _run("--substrate", str(sub), "--out", str(out), "--win", "4",
             "--epochs", "1", "--width", "8", "--batch", "64")
    assert r.returncode == 0, r.stderr
    blob = (r.stdout + r.stderr).lower()
    for banned in ("ap-lift", "ap_lift", "average precision", "auc"):
        assert banned not in blob, f"trainer printed a headline metric: {banned!r}"
    assert (out / "heldout_scores.npz").exists(), "raw held-out scores must be emitted"
