"""PhysicalAI-WorldModel-Synthetic-Scenarios -> TanitAD contract (VIDEO-ONLY).

WHY THIS IS A *VIDEO-ONLY* LOADER (measured 2026-07-10, backlog P0.1)
--------------------------------------------------------------------
The 264k-clip / 8.3 TB OpenMDW-1.1 corpus was hoped to be a "near-zero cosmos
mirror" (share ``cosmos_drive.poses_to_signals`` + the 9-ch action-conditioned
contract).  A direct file-tree probe of the live HF repo (``probe_worldmodel_synth.py``)
settled the gating question against the ACTUAL layout, not the card prose:

    <family>/<clip_id>/
        video/{front_tele,front_wide,left_fisheye,rear_fisheye,
               rear_left,rear_right,right_fisheye}.mp4     # 4K@24, 462 frames
        description/{<same 7 cams>}.json                   # VLM caption + scene meta

Each ``description/*.json`` = ``{framerate, nb_frames, t2w_windows:[{start,end,
qwen2p5_7b_caption}], metadata:{weather,time_of_day,surface_type,region}}``.
**There is NO ego pose, NO CAN, NO action, NO calibration track anywhere in the
clip.**  So the cosmos-mirror assumption is FALSE: this corpus cannot supply
action-conditioned episodes without a trained inverse-dynamics head (H7 / IDM).

DESIGN CONSEQUENCE (P8 — no silent fabrication)
-----------------------------------------------
This loader therefore emits a contract-shaped episode whose FRAMES are real
(front_wide -> D-016 focal-canon -> D-015 9-ch stacks, identical geometry to
comma2k19 / Cosmos so the encoder sees one task) but whose ``actions`` and
``poses`` are an explicit **NaN sentinel** (``ACTION_SOURCE = "idm_pending"``).
Two honesty guards follow from that:
  1. ``CORPUS_META["actions"] is None`` -> the I7 task-identity fingerprint
     (D-017) MISMATCHES comma2k19 / Cosmos, so ``MixedWindowDataset`` /
     probe-fit admission mechanically EXCLUDE this corpus from the
     action-conditioned D-010 mix (same mechanism that correctly rejected the
     1-ch BEV adapter) until IDM labels exist.
  2. NaN actions make any action-conditioned trainer fail LOUD (NaN loss),
     never train on fabricated zeros.

USABLE TODAY, WITHOUT ACTIONS:
  * video-only / self-supervised visual pretraining (frames are real);
  * caption + scene-metadata sourcing for the SCENARIO_DATABASE joint duty
    (``build_manifest`` -> emergency / pedestrian / weather-degradation long-tail
    for SC-02/05/06) — this is the corpus's immediate value.
  * once a comma2k19/Cosmos-trained IDM head lands (H7 flywheel), the same
    ``discover_clips`` feeds pseudo-labeling -> real actions -> mix admission.

Front camera ``front_wide`` HFOV is taken as the nominal 120 deg (Hyperion class,
== Cosmos ``front_wide_120fov`` / PhysicalAI front-wide); the exact intrinsic is
UNPUBLISHED (no calib files) -> focal canon is nominal, pod-verify before any
trained claim.  Pure signal/contract code is unit-tested on synthetic fixtures.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from tanitad.data._contract import EpisodeWindowDataset, assert_contract
from tanitad.data.calib import focal_crop_resize, nominal_focal_from_hfov
from tanitad.data.comma2k19 import stack_frames
from tanitad.data.toy_driving import ToyEpisode

FRONT_CAM = "front_wide"
FAMILIES = ("emergency", "lanechange", "nudging", "pedestrian",
            "weather_degradation")
FRONT_WIDE_HFOV_DEG = 120.0    # nominal (no calib in corpus) -> pod-verify
SRC_FPS = 24.0                 # REAL rate (== per-clip description framerate;
                               # NOT a mux artifact, unlike Cosmos where 24 was)
TARGET_HZ = 12.0               # stride round(24/12)=2 -> exactly 12 Hz

ACTION_SOURCE = "idm_pending"  # actions are UNLABELED; NaN sentinel until IDM
DATA_TAG = "data:wms"          # every consuming experiment carries this tag
LICENSE = "OpenMDW-1.1"        # ungated; public-claimability = proposed D-022 (held)

# I7 fingerprint (D-017): actions/poses = None is DELIBERATE and load-bearing.
# It makes i7_task_identity mismatch comma2k19/Cosmos so this corpus is excluded
# from the action-conditioned mix until IDM pseudo-labels give it real actions.
CORPUS_META = {
    "channels": 9, "image_size": 256, "f_eff_px": 266.0, "hz": 12.0,
    "actions": None,           # <-- no ego actions (the whole point)
    "poses": None,
}


# --------------------------------------------------------------------------- #
# Description (caption + scene metadata) parsing                              #
# --------------------------------------------------------------------------- #
def parse_description(d: dict) -> dict:
    """One ``description/<cam>.json`` -> flat scene record (robust to missing keys).

    Caption is the first ``t2w_windows`` entry's ``*_caption`` (the Qwen2.5-7B
    text-to-world description); scene metadata is passed through.
    """
    meta = d.get("metadata", {}) or {}
    caption = ""
    wins = d.get("t2w_windows") or []
    if wins:
        w0 = wins[0]
        caption = next((v for k, v in w0.items()
                        if k.endswith("caption") and isinstance(v, str)), "")
    return {
        "framerate": float(d.get("framerate", SRC_FPS)),
        "nb_frames": int(d.get("nb_frames", 0)),
        "weather": meta.get("weather", "unknown"),
        "time_of_day": meta.get("time_of_day", "unknown"),
        "surface_type": meta.get("surface_type", "unknown"),
        "region": meta.get("region", "unknown"),
        "caption": caption,
    }


def _load_description(path: str | Path) -> dict:
    return parse_description(json.loads(Path(path).read_text()))


# --------------------------------------------------------------------------- #
# Clip discovery over the family/clip/{video,description} layout               #
# --------------------------------------------------------------------------- #
def discover_clips(root: str | Path, family: str | None = None,
                   weather: str | None = None, time_of_day: str | None = None,
                   desc_fn=_load_description) -> list[dict]:
    """Clips under ``root`` that have a front_wide video AND its description.

    Layout: ``root/<family>/<clip_id>/video/front_wide.mp4`` +
    ``root/<family>/<clip_id>/description/front_wide.json``. ``family`` /
    ``weather`` / ``time_of_day`` filter (metadata read from the description).
    """
    root = Path(root)
    fams = [family] if family else [f for f in FAMILIES if (root / f).is_dir()]
    if not fams:                                    # flat layout (root == family)
        fams = [""]
    out: list[dict] = []
    for fam in fams:
        fam_dir = root / fam if fam else root
        if not fam_dir.is_dir():
            continue
        for clip_dir in sorted(p for p in fam_dir.iterdir() if p.is_dir()):
            mp4 = clip_dir / "video" / f"{FRONT_CAM}.mp4"
            djs = clip_dir / "description" / f"{FRONT_CAM}.json"
            if not (mp4.exists() and djs.exists()):
                continue
            rec = desc_fn(djs)
            if weather and weather.lower() != str(rec["weather"]).lower():
                continue
            if time_of_day and time_of_day.lower() != str(rec["time_of_day"]).lower():
                continue
            out.append({"clip_id": clip_dir.name, "family": fam or "root",
                        "mp4": mp4, "desc": djs, **rec})
    return out


def _episode_id(family: str, clip_id: str) -> int:
    return int(hashlib.sha1(f"wms/{family}/{clip_id}".encode()).hexdigest()[:8], 16)


def _decode_mp4(mp4: Path, size: int) -> Tensor:
    """All frames -> uint8 [N,3,size,size], D-016 focal-canon (nominal 120 HFOV)."""
    import av                                                 # lazy: .[real]
    frames = []
    with av.open(str(mp4)) as c:
        stream = c.streams.video[0]
        stream.thread_type = "AUTO"
        for fr in c.decode(stream):
            rgb = torch.from_numpy(fr.to_ndarray(format="rgb24")).permute(2, 0, 1)
            frames.append(rgb)
    vid = torch.stack(frames)                                 # [N,3,H,W] u8
    f_px = nominal_focal_from_hfov(vid.shape[-1], FRONT_WIDE_HFOV_DEG)
    return focal_crop_resize(vid, f_px, size)


# --------------------------------------------------------------------------- #
# Build one VIDEO-ONLY episode (frames real, actions/poses = NaN sentinel)     #
# --------------------------------------------------------------------------- #
def build_episode(clip: dict, size: int = 256, n_stack: int = 3,
                  src_fps: float = SRC_FPS, decode_fn=_decode_mp4) -> ToyEpisode:
    """One WMS clip -> contract episode at ~TARGET_HZ with D-015 stacking.

    Frames are REAL (front_wide, focal-canon, 9-ch stacked). ``actions``/``poses``
    are an explicit **NaN sentinel** ([T,2]/[T,4]) — this corpus ships no ego
    pose (measured 2026-07-10). The shapes satisfy the contract; the NaN + the
    ``CORPUS_META["actions"] is None`` fingerprint keep it out of the
    action-conditioned mix (I7) and make any action trainer fail loud.
    """
    stride = max(1, int(round(src_fps / TARGET_HZ)))
    vid = decode_fn(clip["mp4"], size)                        # [M,3,S,S] u8
    vid = vid[::stride]
    stacked = stack_frames(vid, n_stack)                      # [T,9,S,S] u8
    T = stacked.shape[0]
    if T < 1:
        raise ValueError(f"clip {clip.get('clip_id')} too short after stride")
    nan_a = torch.full((T, 2), float("nan"), dtype=torch.float32)
    nan_p = torch.full((T, 4), float("nan"), dtype=torch.float32)
    ep = ToyEpisode(frames=stacked, actions=nan_a, poses=nan_p,
                    episode_id=_episode_id(clip.get("family", ""),
                                           clip["clip_id"]))
    return ep


def assert_video_only_contract(ep: ToyEpisode) -> None:
    """Frames satisfy the 9-ch contract AND actions/poses are the NaN sentinel.

    The positive half proves the episode is a valid contract object (shape,
    dtype, channel count); the NaN half proves we did NOT fabricate actions —
    the honest marker that this is a video-only episode (P8).
    """
    assert_contract(ep, channels=9)
    assert bool(torch.isnan(ep.actions).all()), "WMS actions must be NaN sentinel"
    assert bool(torch.isnan(ep.poses).all()), "WMS poses must be NaN sentinel"


# --------------------------------------------------------------------------- #
# Scenario-sourcing manifest (joint duty D-020 §5) — the corpus's value TODAY  #
# --------------------------------------------------------------------------- #
def build_manifest(clips: list[dict]) -> list[dict]:
    """Flat scene index for SCENARIO_DATABASE data-sourcing (no decode needed).

    One row per discovered clip: family, weather, time_of_day, surface, region,
    caption, frame count -> lets the Opponent/Bench agents filter the long-tail
    (e.g. family='pedestrian' + time_of_day='Night') to concrete clip lists.
    """
    return [{"family": c["family"], "clip_id": c["clip_id"],
             "weather": c["weather"], "time_of_day": c["time_of_day"],
             "surface_type": c["surface_type"], "region": c["region"],
             "nb_frames": c["nb_frames"], "caption": c["caption"]}
            for c in clips]


def split_clips(clips: list[dict], val_frac: float = 0.2,
                seed: int = 0) -> tuple[list[dict], list[dict]]:
    """CLIP-level split (I3 unit — each synthetic clip is independent)."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(clips), generator=g).tolist()
    n_val = max(1, int(len(clips) * val_frac))
    val_i = set(perm[:n_val])
    return ([c for i, c in enumerate(clips) if i not in val_i],
            [c for i, c in enumerate(clips) if i in val_i])


def WMSVideoDataset(clips: list[dict], window: int = 8, max_horizon: int = 16,
                    size: int = 256, n_stack: int = 3,
                    **build_kw) -> EpisodeWindowDataset:
    """Window dataset over WMS clips (frames real, actions NaN). For video-only
    pretraining — NOT for the action-conditioned mix (I7 excludes it)."""
    eps = []
    for c in clips:
        try:
            eps.append(build_episode(c, size=size, n_stack=n_stack, **build_kw))
        except Exception as e:                                # skip corrupt/partial
            print(f"[wms] skip {c.get('clip_id')}: {type(e).__name__}: {e}")
    return EpisodeWindowDataset(eps, window=window, max_horizon=max_horizon)


# --------------------------------------------------------------------------- #
# Pod-only real-clip sanity (not in CI — needs real bytes)                     #
# --------------------------------------------------------------------------- #
def verify_real_clip(clip: dict, size: int = 256) -> dict:
    """Decode ONE real clip; return sanity stats + A8 for the data card."""
    from tanitad.data._contract import frame_change_fraction, to_float_frames
    ep = build_episode(clip, size=size)
    assert_video_only_contract(ep)
    a8 = frame_change_fraction(to_float_frames(ep.frames)[:, -3:])
    return {"clip_id": clip["clip_id"], "family": clip["family"],
            "weather": clip["weather"], "time_of_day": clip["time_of_day"],
            "T": int(ep.frames.shape[0]),
            "a8_frame_change_fraction": a8,
            "action_source": ACTION_SOURCE}
