"""Standalone tests for the WorldModel-Synthetic VIDEO-ONLY loader (backlog P0.1).

Zero real bytes, zero `av`, zero network: the video decode is injected and the
description JSONs are written to a tmp dir in the ACTUAL corpus schema. `tanitad`
must be importable (editable stack install, `pip install -e stack`).

    pytest "TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-07-10-worldmodel-synthetic-pose-probe/tests" -q
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tanitad_worldmodel_synth as wms  # noqa: E402
from tanitad.data._contract import assert_contract  # noqa: E402
from tanitad.data.comma2k19 import CORPUS_META as COMMA_META  # noqa: E402
from tanitad.instruments.checks import i7_task_identity  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixtures: a real-schema description + an injected decode                     #
# --------------------------------------------------------------------------- #
def _desc(weather="Rainy", tod="Night", nb=462):
    return {
        "framerate": 24.0, "nb_frames": nb,
        "t2w_windows": [{"start_frame": 0, "end_frame": nb,
                         "qwen2p5_7b_caption": "A pedestrian crosses in the rain."}],
        "metadata": {"weather": weather, "time_of_day": tod,
                     "surface_type": "Asphalt", "region": "Urban"},
    }


def _fake_decode(n_raw=24, size=256):
    def _f(mp4, s):
        g = torch.Generator().manual_seed(len(str(mp4)))
        return (torch.rand(n_raw, 3, s, s, generator=g) * 255).to(torch.uint8)
    return _f


def _make_clip_tree(root: Path, family="pedestrian", clip_id="ped_0001",
                    desc=None):
    cdir = root / family / clip_id
    (cdir / "video").mkdir(parents=True)
    (cdir / "description").mkdir(parents=True)
    (cdir / "video" / "front_wide.mp4").write_bytes(b"\x00")     # placeholder
    (cdir / "description" / "front_wide.json").write_text(
        json.dumps(desc or _desc()))
    return cdir


# --------------------------------------------------------------------------- #
# 1. description parsing on the real schema                                    #
# --------------------------------------------------------------------------- #
def test_parse_description_real_schema():
    rec = wms.parse_description(_desc(weather="Foggy", tod="Night", nb=300))
    assert rec["framerate"] == 24.0 and rec["nb_frames"] == 300
    assert rec["weather"] == "Foggy" and rec["time_of_day"] == "Night"
    assert rec["region"] == "Urban" and "pedestrian" in rec["caption"].lower()


def test_parse_description_missing_keys_safe():
    rec = wms.parse_description({})                 # nothing present
    assert rec["nb_frames"] == 0 and rec["weather"] == "unknown"
    assert rec["caption"] == ""


# --------------------------------------------------------------------------- #
# 2. discovery + metadata filtering                                           #
# --------------------------------------------------------------------------- #
def test_discover_and_filter(tmp_path):
    _make_clip_tree(tmp_path, "pedestrian", "ped_night",
                    _desc(weather="Rainy", tod="Night"))
    _make_clip_tree(tmp_path, "pedestrian", "ped_day",
                    _desc(weather="Clear", tod="Daytime"))
    _make_clip_tree(tmp_path, "emergency", "emg_night",
                    _desc(weather="Clear", tod="Night"))

    allc = wms.discover_clips(tmp_path)
    assert len(allc) == 3
    assert {c["family"] for c in allc} == {"pedestrian", "emergency"}

    peds = wms.discover_clips(tmp_path, family="pedestrian")
    assert len(peds) == 2

    night = wms.discover_clips(tmp_path, time_of_day="Night")
    assert {c["clip_id"] for c in night} == {"ped_night", "emg_night"}

    rainy = wms.discover_clips(tmp_path, weather="rainy")     # case-insensitive
    assert [c["clip_id"] for c in rainy] == ["ped_night"]


def test_discover_skips_incomplete_clip(tmp_path):
    # a clip with video but no description must be skipped
    cdir = tmp_path / "nudging" / "nud_0001"
    (cdir / "video").mkdir(parents=True)
    (cdir / "video" / "front_wide.mp4").write_bytes(b"\x00")
    assert wms.discover_clips(tmp_path) == []


# --------------------------------------------------------------------------- #
# 3. episode contract (G-D2) — frames real, actions/poses NaN sentinel         #
# --------------------------------------------------------------------------- #
def test_build_episode_contract(tmp_path):
    _make_clip_tree(tmp_path)
    clip = wms.discover_clips(tmp_path)[0]
    ep = wms.build_episode(clip, size=64, decode_fn=_fake_decode(n_raw=24, size=64))

    # positive: it IS a valid 9-channel contract episode
    assert_contract(ep, channels=9)
    assert ep.frames.dtype == torch.uint8
    assert ep.frames.shape[1] == 9 and ep.frames.shape[2:] == (64, 64)
    T = ep.frames.shape[0]
    assert ep.actions.shape == (T, 2) and ep.poses.shape == (T, 4)

    # honest: NO fabricated actions -> NaN sentinel everywhere
    assert torch.isnan(ep.actions).all()
    assert torch.isnan(ep.poses).all()
    wms.assert_video_only_contract(ep)          # combined guard passes


def test_stride_gives_12hz():
    # 24 fps -> stride round(24/12)=2 -> 12 raw kept from 24; 9-ch stack drops 2
    assert int(round(wms.SRC_FPS / wms.TARGET_HZ)) == 2


# --------------------------------------------------------------------------- #
# 4. I7 fingerprint EXCLUDES the corpus from the action-conditioned mix        #
# --------------------------------------------------------------------------- #
def test_i7_excludes_from_action_mix():
    identical, bad = i7_task_identity(COMMA_META, wms.CORPUS_META)
    assert not identical, "no-action corpus must NOT match comma2k19's fingerprint"
    assert "actions" in bad                      # the load-bearing mismatch
    assert wms.CORPUS_META["actions"] is None


def test_frames_geometry_matches_task():
    # the parts that DO match one task (so video-only pretraining shares an encoder)
    for k in ("channels", "image_size", "f_eff_px"):
        assert wms.CORPUS_META[k] == COMMA_META[k]


# --------------------------------------------------------------------------- #
# 5. manifest (scenario sourcing) + clip-level split                           #
# --------------------------------------------------------------------------- #
def test_build_manifest(tmp_path):
    _make_clip_tree(tmp_path, "pedestrian", "ped_night",
                    _desc(weather="Rainy", tod="Night"))
    _make_clip_tree(tmp_path, "weather_degradation", "wd_fog",
                    _desc(weather="Foggy", tod="Daytime"))
    man = wms.build_manifest(wms.discover_clips(tmp_path))
    assert len(man) == 2
    fams = {r["family"] for r in man}
    assert fams == {"pedestrian", "weather_degradation"}
    assert all("caption" in r and "weather" in r for r in man)


def test_split_clips_disjoint_and_clip_level():
    clips = [{"clip_id": f"c{i}", "family": "emergency"} for i in range(10)]
    tr, va = wms.split_clips(clips, val_frac=0.2, seed=0)
    ids_tr = {c["clip_id"] for c in tr}
    ids_va = {c["clip_id"] for c in va}
    assert ids_tr.isdisjoint(ids_va)
    assert len(ids_tr) + len(ids_va) == 10 and len(va) == 2
