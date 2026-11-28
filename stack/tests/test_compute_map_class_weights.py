"""refcv7 NEW-2 -- ``stack/scripts/compute_map_class_weights.py``.

Pinned with LITERAL targets:

* the median-frequency definition on hand-computed counts, and the 25 clip;
* the presence denominator (seen cells of the frames in which a class is present)
  and that NOT-SEEN cells are never counted, on a synthetic clip;
* end to end on a synthetic cache + GT: the JSON the trainer's loader accepts, the
  DRY RUN marker it refuses, the read-only manifest rule, the coverage floor, and
  that ``--limit`` is a dry-run-only lever.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import compute_map_class_weights as W                           # noqa: E402
from tanitad.data import semantic_map_gt as G                   # noqa: E402
from tanitad.data import semantic_map_gt_fine as F              # noqa: E402
from tanitad.data.v2_dataset import MANIFEST_NAME, MANIFEST_VERSION  # noqa: E402
from tanitad.models import map_head_hires as H                  # noqa: E402


def test_median_frequency_on_literal_counts():
    r = W.weights_from_counts([10, 20, 30, 40, 50, 60, 70, 80], [100] * 8,
                              definition="mf")
    assert r["median_freq"] == pytest.approx(0.45)
    assert r["weights"] == pytest.approx([4.5, 2.25, 1.5, 1.125, 0.9, 0.75,
                                          0.45 / 0.7, 0.5625])
    assert r["n_clipped"] == 0


def test_the_25_clip_and_an_absent_class():
    r = W.weights_from_counts([1, 1000, 1000, 1000, 1000, 1000, 1000, 1000],
                              [1000] * 8, definition="mf")
    assert r["weights_unclipped"][0] == pytest.approx(1000.0)
    assert r["weights"][0] == 25.0 and r["n_clipped"] == 1
    q = W.weights_from_counts([1, 1000, 1000, 1000, 1000, 1000, 1000, 1000],
                              [1000] * 8)                      # sqrt_mf (A8)
    assert q["weights_unclipped"][0] == pytest.approx(1000.0 ** 0.5)
    assert q["weights"][0] == 25.0 and q["n_clipped"] == 1
    with pytest.raises(SystemExit, match="never seen"):
        W.weights_from_counts([0, 1, 1, 1, 1, 1, 1, 1], [1] * 8)


def test_counts_ignore_not_seen_and_use_the_presence_denominator():
    c = np.full((2, 600, 320), 255, np.uint8)
    c[0, :100, :] = 1                      # frame 0: 32,000 drivable
    c[0, 100:110, :] = 2                   # + 3,200 lane
    c[1, :50, :] = 1                       # frame 1: 16,000 drivable only
    r = W._count_clip(c, 1, want_prior=True)
    assert r["n"].tolist() == [0, 48000, 3200, 0, 0, 0, 0, 0]
    assert r["seen"] == 51200
    # lane is present in frame 0 only, whose seen cells are 35,200
    assert r["S"].tolist() == [0, 51200, 35200, 0, 0, 0, 0, 0]
    assert r["present_frames"].tolist() == [0, 2, 1, 0, 0, 0, 0, 0]
    assert int(r["cell"][1, 0, 0]) == 2 and int(r["cell"][2, 105, 3]) == 1
    assert int(r["cell"][:, 300, 0].sum()) == 0          # never-seen cell: 0 counts


# --------------------------------------------------------------------------- #
# end to end on a synthetic cache                                              #
# --------------------------------------------------------------------------- #
def _gt(root: Path, clip_id: str, T: int = 4, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    fine = rng.integers(0, 8, (T, 600, 320)).astype(np.uint8)
    fine[:, :20] = 255
    t_cam = np.arange(3 * T + 3, dtype=np.float64) * 33_333.0 + 1_000.0
    t_query = np.linspace(t_cam[0], t_cam[0] + (T - 1) * 100_000.0, T)
    idx = np.searchsorted(t_cam, t_query)
    meta = {"schema": G.SCHEMA, "frame": "rig",
            "cartesian": {"x_max_m": 60.0, "y_half_m": 16.0, "cell_m": 0.5,
                          "shape": [120, 64]},
            "fine": dict(F.FINE_SPEC), "sub_samples_per_axis": 5,
            "channels": list(G.CHANNELS), "fraction_scale": 255,
            "source": {"clip_sha12": G.sha12(clip_id), "world_map_res_m": 0.1}}
    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        root / f"{G.sha12(clip_id)}{G.GT_SUFFIX}",
        meta_json=np.array(json.dumps(meta)), t_query_us=t_query,
        t_img_us=t_cam[idx].astype(np.int64), cam_frame_idx=idx.astype(np.int32),
        T_world_rig=np.tile(np.eye(4), (T, 1, 1)),
        cart_frac=F.block_fraction_u8(fine), fine_codes=fine)


def _cache(root: Path, clip_ids) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    files = []
    for c in clip_ids:
        (root / f"{c}.v2ep.pt").write_bytes(b"")
        files.append(f"{c}.v2ep.pt")
    torch.save({"version": MANIFEST_VERSION, "files": sorted(files),
                "clip_id": list(clip_ids), "n_stack": [3] * len(clip_ids)},
               root / MANIFEST_NAME)
    return root


#: the synthetic GT below is /2 (60 m x +-16 m); the script's default is the A7 extent
V2_EXTENT = ["--x-max-m", "60", "--y-half-m", "16"]


def test_end_to_end_train_json_is_loadable_and_a_dry_run_is_refused(tmp_path):
    ids = [f"synthetic-weights-{i}" for i in range(4)]
    cache = _cache(tmp_path / "cache", ids)
    for i, c in enumerate(ids):
        _gt(tmp_path / "gt", c, seed=i)
    out = tmp_path / "w_train.json"
    prior = tmp_path / "prior.npz"
    assert W.main(["--v2-cache", str(cache), "--gt-root", str(tmp_path / "gt"),
                   "--split", "train", "--out", str(out), *V2_EXTENT,
                   "--prior-out", str(prior)]) == 0
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["dry_run"] is False and rec["n_clips_used"] == 4
    assert rec["extent"] == {"x_max_m": 60.0, "y_half_m": 16.0}
    assert rec["pre_registered"] is True and rec["weight_floor"] == 0.0
    assert rec["definition_id"] == "sqrt_mf"
    assert rec["frames"] == 16 and len(rec["weights"]) == 8
    assert not any(c in out.read_text(encoding="utf-8") for c in ids)   # sha12 only
    w, st = H.load_class_weights(out)
    assert tuple(w.shape) == (8,) and st["inputs_sha256"] == rec["inputs_sha256"]
    with np.load(prior) as z:
        assert z["counts"].shape == (8, 600, 320)
        assert sorted(z["fit_clips"].tolist()) == sorted(G.sha12(c) for c in ids)
    dry = tmp_path / "w_dry.json"
    W.main(["--v2-cache", str(cache), "--gt-root", str(tmp_path / "gt"),
            "--split", "dry-run-eval", "--limit", "2", "--out", str(dry), *V2_EXTENT])
    with pytest.raises(ValueError, match="DRY RUN"):
        H.load_class_weights(dry)


def test_refusals(tmp_path):
    ids = [f"synthetic-weights-{i}" for i in range(4)]
    cache = _cache(tmp_path / "cache", ids)
    _gt(tmp_path / "gt", ids[0])                       # 1 of 4 clips has GT
    base = ["--v2-cache", str(cache), "--gt-root", str(tmp_path / "gt"),
            "--out", str(tmp_path / "w.json"), *V2_EXTENT]
    with pytest.raises(SystemExit, match="floor"):
        W.main(base + ["--split", "train"])
    with pytest.raises(SystemExit, match="dry-run"):
        W.main(base + ["--split", "train", "--limit", "1"])
    (cache / MANIFEST_NAME).unlink()
    with pytest.raises(SystemExit, match="never builds"):
        W.main(base + ["--split", "train"])
    assert not (cache / MANIFEST_NAME).exists()          # nothing written back


def test_the_named_definitions_on_literal_counts():
    n = [10, 20, 30, 40, 50, 60, 70, 80]
    S = [100, 100, 100, 100, 1000, 1000, 1000, 1000]
    p = W.weights_from_counts(n, S, definition="mf")           # Eigen & Fergus
    assert p["median_freq"] == pytest.approx(0.09)
    assert p["weights"] == pytest.approx([0.9, 0.45, 0.3, 0.225, 1.8, 1.5,
                                          0.09 / 0.07, 1.125])
    g = W.weights_from_counts(n, S, definition="mf_global")    # n_c / all
    assert g["weights"] == pytest.approx([4.5, 2.25, 1.5, 1.125, 0.9, 0.75,
                                          0.45 / 0.7, 0.5625])
    with pytest.raises(ValueError):
        W.weights_from_counts(n, S, definition="mean")


def test_the_A8_registered_weights_are_sqrt_mf_and_the_default():
    """SPEC_REFCV7 §13 (A8): refcv7 uses `sqrt_mf`; it is the default and the only
    definition a JSON marks pre_registered."""
    assert W.REGISTERED == "sqrt_mf"
    n = [10, 20, 30, 40, 50, 60, 70, 80]
    S = [100, 100, 100, 100, 1000, 1000, 1000, 1000]
    q = W.weights_from_counts(n, S)                            # the default
    assert q["weights"] == pytest.approx([0.9 ** 0.5, 0.45 ** 0.5, 0.3 ** 0.5,
                                          0.225 ** 0.5, 1.8 ** 0.5, 1.5 ** 0.5,
                                          (0.09 / 0.07) ** 0.5, 1.125 ** 0.5])


def test_the_floor_is_named_and_never_the_default():
    n = [10, 20, 30, 40, 50, 60, 70, 80]
    S = [100, 100, 100, 100, 1000, 1000, 1000, 1000]
    p = W.weights_from_counts(n, S, definition="mf")
    q = W.weights_from_counts(n, S, definition="sqrt_mf")
    assert q["weights"] == pytest.approx([v ** 0.5 for v in p["weights"]])
    f = W.weights_from_counts(n, S, definition="mf", floor=0.5)
    assert f["weights"] == pytest.approx([max(v, 0.5) for v in p["weights"]])
    assert W.weights_from_counts(n, S)["weights"] == q["weights"]   # default = A8


def test_the_weights_are_counted_at_the_declared_extent_and_bound_to_it(tmp_path):
    """The default extent is SPEC_REFCV7 §12's (100 m x +-30 m): a /2 GT file is
    REFUSED there, loudly, by the reader (never silently skipped or cropped), and a
    JSON counted at 60 x 16 is refused by the trainer's loader under the A7 extent."""
    ids = [f"synthetic-weights-{i}" for i in range(4)]
    cache = _cache(tmp_path / "cache", ids)
    for i, c in enumerate(ids):
        _gt(tmp_path / "gt", c, seed=i)
    with pytest.raises(F.FineSpecMismatch, match="Re-export"):
        W.main(["--v2-cache", str(cache), "--gt-root", str(tmp_path / "gt"),
                "--split", "train", "--out", str(tmp_path / "w7.json")])
    assert not (tmp_path / "w7.json").exists()
    out = tmp_path / "w2.json"
    W.main(["--v2-cache", str(cache), "--gt-root", str(tmp_path / "gt"),
            "--split", "train", "--out", str(out), *V2_EXTENT])
    H.load_class_weights(out, extent=F.EXTENT_V2)
    with pytest.raises(ValueError, match="Recompute"):
        H.load_class_weights(out, extent=F.EXTENT_REFCV7)
