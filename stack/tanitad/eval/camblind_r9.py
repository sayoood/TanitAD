"""camblind_r9.py -- refcv9 C9: the camera-blind instrument (W4) that swaps EVERY camera-derived input, and the identity that PROVES it.

REFCV9_DESIGN 2.7 "Camera-blind instrument consequence", 5.2 row W4, 4.2 C9.  Replaces the *wrapper* of
``FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-27-refcv9-anticheat/code/camblind_roll.py`` (which swapped ``_window_u8``'s pixels only);
that file's derangement rule (Sattolo, seed 0, same-rank donor window) and its three controls are carried over VERBATIM here.

WHY.  In refcv9 the planner also reads the map channels, the critic's map read and the box tokens (C8).  They are all FUNCTIONS OF THE
TRUNK TOKENS, hence of the pixels, so a pixel swap covers them -- but only while no camera-derived tensor reaches the forward from
anywhere ELSE (a precomputed sidecar / cache).  This module makes that a CHECKED property instead of an assumption:

  1. ``CENSUS`` -- every dataset-item key, every model-forward input and every path-valued argv flag of the refcv9 / refcv8 argvs, each
     classified CAMERA / EGO / NAV / LABEL / CALIB / META with a ``file:needle`` source pin (``verify_census`` resolves the line).
  2. ``swap_item`` -- every CAMERA key from the donor window, every other key the window's OWN object.  FAIL-CLOSED: a key the census
     does not know is refused (an offline camera tensor added later cannot ride along unclassified), and a swap request that names an
     EGO / NAV / LABEL / CALIB / META key is refused.
  3. ``make_camblind_dataset_cls`` -- the roll-side wrapper (W4's ``CamBlindWindows`` with the census check in ``__getitem__``).
  4. The IDENTITY (``identity_sweep``): perturb ONE own key at a time (pixels, every offline camera tensor, every label) and measure the
     change of every planner input (trunk, planner BEV, map channels, box slots, box tokens, the critic's map read) and of the planner
     output.  Under the correct instrument that change is EXACTLY 0.0 for every pixel-derived key; every key that DOES move the planner
     must be a legitimate non-camera input (EGO / NAV / CALIB).  Under the MUTATION "swap pixels only" it is > 0 wherever an offline
     camera tensor is read.  Positive controls: perturbing the DONOR's pixels, or the window's own ego, moves the planner.

Pure parts (census, swap, derangement, controls) import numpy only; torch is imported lazily by the parts that need it.
No clip id is handled: episodes are INDICES into the roll's manifest.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CAMERA, EGO, NAV, LABEL, CALIB, META = "CAMERA", "EGO", "NAV", "LABEL", "CALIB", "META"
CLASSES = (CAMERA, EGO, NAV, LABEL, CALIB, META)
#: classes whose keys the instrument must LEAVE the window's own (a swap request that names one is refused)
KEEP_OWN = frozenset({EGO, NAV, LABEL, CALIB, META})
#: classes the model forward may legitimately read besides the camera
LEGIT_INPUT = frozenset({EGO, NAV, CALIB, META})
#: the keys of the roll's ``_window_u8`` dict that are camera tensors
WINDOW_CAMERA_KEYS = ("frames", "future_frames")


class UnclassifiedKey(KeyError):
    """A dataset-item key / model input / argv flag the census does not know: refused, because an unclassified key could be an offline camera tensor."""


class SwapRefused(ValueError):
    """A swap request that would replace a non-camera key (ego / nav / label / calibration) with the donor's."""


class CensusGap(AssertionError):
    """The census is inconsistent with the sources or with the instrument's configuration."""


@dataclass(frozen=True)
class Entry:
    name: str            # exact key name (item / model_input) or the argv flag
    kind: str            # "item" | "model_input" | "argv"
    cls: str             # CAMERA | EGO | NAV | LABEL | CALIB | META
    pixel_derived: bool  # is the tensor a FUNCTION OF PIXELS (online or offline)?
    reaches: bool        # can it reach the forward of the eval roll (directly or via an input derived from it)?
    src: str             # path under stack/
    needle: str          # a single-line substring that pins the definition
    after: str = ""      # locate ``needle`` only after the first occurrence of this anchor
    why: str = ""


_RB, _RC, _V3T, _V3, _BDF, _R8T = ("scripts/refb_train.py", "scripts/refc_train.py", "scripts/refc_v3_train.py", "tanitad/refs/refc_v3.py",
                                   "tanitad/train/box_depth_aux_flag.py", "tanitad/train/refcv8_train.py")
_FWD = "def forward(self, frames: Tensor, nav_cmd: Tensor | None = None,"


def _i(name, cls, reach, src, needle, why="", *, px=False, after=""):
    return Entry(name, "item", cls, px, reach, src, needle, after, why)


def _m(name, cls, reach, needle, why="", *, px=False):
    return Entry(name, "model_input", cls, px, reach, _V3, needle, _FWD, why)


def _a(name, cls, reach, why, *, px=False):
    return Entry(name, "argv", cls, px, reach, _V3T, 'add_argument("%s"' % name, "", why)


# --------------------------------------------------------------------------------------------------------------- the census
_ITEM = (
    # ---- the window (refb_train.py FailLoudWindowDataset._window_u8; the roll's Refcv6EvalWindows re-spells it with future_frames zeroed)
    _i("frames", CAMERA, True, _RB, '"frames": ep.frames[t:t + w],', "uint8 pixels [W, C, H, W] decoded from --v2-cache / --eval-cache: THE camera tensor; "
       "fed as the model's `frames`", px=True),
    _i("future_frames", CAMERA, False, _RB, '"future_frames": ep.frames[t + w:t + w + self.max_horizon],',
       "pixels of the future window; the roll's class returns torch.zeros(0) (r7_roll.py Refcv6EvalWindows._window_u8) and no forward takes it", px=True),
    _i("actions", EGO, True, _RB, '"actions": ep.actions[t:t + w],', "past actions; ego_state_from_batch reads actions[:, -1]"),
    _i("future_actions", LABEL, False, _RB, '"future_actions": ep.actions[t + w:t + w + self.max_horizon],', "future of the ego: a target"),
    _i("future_poses", LABEL, False, _RB, '"future_poses": ep.poses[t + w:t + w + self.max_horizon],', "future of the ego: a target"),
    _i("pose_last", EGO, True, _RB, '"pose_last": ep.poses[t + w - 1],', "pose at t0 (the last OBSERVED frame): v0 = pose_last[:, 3] and ego_state"),
    _i("episode_id", META, False, _RB, '"episode_id": ep.episode_id,', "join key (stable id); selects the rig geometry, carries no scene content"),
    _i("nav_cmd", NAV, True, _V3T, 'item["nav_cmd"] = torch.tensor(R8_V9_NAV_TO_NAV_CMD[_tok8], dtype=torch.long)',
       "the navigation command: an INPUT at training and at inference (PI); a token, not a function of pixels"),
    _i("nav_valid", NAV, False, _V3T, 'item["nav_valid"] = torch.tensor(True)', "validity of the nav token (a mask for the loss)"),
    _i("route_target", LABEL, False, _RB, 'item["route_target"] = torch.tensor(route_tgt, dtype=torch.long)', "aux CE target"),
    _i("route_valid", LABEL, False, _RC, 'item["route_valid"] = torch.tensor(bool(r["valid"]))', "mask of the aux target"),
    _i("lan", NAV, False, _RC, 'item["lan"] = torch.from_numpy(f)', "LAN arm route anchors (not enabled in any refcv8 / refcv9 argv: --arm flat|hier)"),
    _i("future_poses_ext", LABEL, False, _V3T, 'item["future_poses_ext"] = ep.poses[idx.clamp(max=T - 1)]', "60-step future: the trajectory target"),
    _i("future_valid_ext", LABEL, False, _V3T, 'item["future_valid_ext"] = idx <= (T - 1)', "mask of the target"),
    _i("pose_hist", EGO, True, _V3T, 'item["pose_hist"] = ep.poses[t:t + w]', "OBSERVED window ego track [t, t+w): the roll's ego_poses"),
    _i("goal_tac", LABEL, False, _V3T, 'item["goal_tac"] = g', "tactical goal target from the future track"),
    _i("goal_tac_valid", LABEL, False, _V3T, 'item["goal_tac_valid"] = gv', "mask"),
    _i("lat_v7", LABEL, False, _V3T, 'item["lat_v7"] = torch.tensor(lat_v7, dtype=torch.long)', "tactical class target"),
    _i("lon_v7", LABEL, False, _V3T, 'item["lon_v7"] = torch.tensor(lon_v7, dtype=torch.long)', "tactical class target"),
    _i("lat_allowed_v7", LABEL, False, _V3T, 'item["lat_allowed_v7"] = _j8["v9_lat_allowed"] & (not _x8)', "v9 allowed-set (loss mask)"),
    _i("lon_allowed_v7", LABEL, False, _V3T, 'item["lon_allowed_v7"] = _j8["v9_lon_allowed"] & (not _x8)', "v9 allowed-set (loss mask)"),
    _i("tac_goal_y", LABEL, False, _V3T, 'item["tac_goal_y"] = torch.tensor(_tg_y, dtype=torch.float32)', "tactical goal-token target (VLM CoT labels)"),
    _i("tac_goal_w", LABEL, False, _V3T, 'item["tac_goal_w"] = torch.tensor(_tg_w, dtype=torch.float32)', "its weights"),
    _i("nav_args", NAV, True, _V3T, 'item["nav_args"] = torch.zeros(3, dtype=torch.float32)', "(distance, time, valid) of the nav command"),
    _i("v_max_ms", EGO, True, _V3T, 'item["v_max_ms"] = torch.tensor(v_ms, dtype=torch.float32)',
       "the speed band: FUTURE-derived from the ego's own track (PI 2026-10-27, disclosed in every refcv9 number); not a function of pixels"),
    _i("v_max_valid", EGO, True, _V3T, 'item["v_max_valid"] = torch.tensor(ok, dtype=torch.float32)', "validity of the speed band"),
    # ---- the agent block: obstacle.offline 3-D tracks = GT labels (never `agent_gt` at inference: --agents head)
    _i("agent_ep", LABEL, False, _V3T, 'item = {"agent_ep": torch.tensor(eid, dtype=torch.long),', "GT agent block (obstacle.offline); a target"),
    _i("agent_box", LABEL, False, _V3T, '"agent_box": t["box"][0], "agent_yaw": t["yaw"][0],', "GT boxes"),
    _i("agent_yaw", LABEL, False, _V3T, '"agent_box": t["box"][0], "agent_yaw": t["yaw"][0],', "GT yaw"),
    _i("agent_cls", LABEL, False, _V3T, '"agent_cls": t["cls"][0], "agent_valid": t["valid"][0],', "GT class"),
    _i("agent_valid", LABEL, False, _V3T, '"agent_cls": t["cls"][0], "agent_valid": t["valid"][0],', "GT mask"),
    _i("agent_occ", LABEL, False, _V3T, '"agent_occ": t["occ"][0], "agent_rates": t["rates"][0],', "GT occlusion flag"),
    _i("agent_rates", LABEL, False, _V3T, '"agent_occ": t["occ"][0], "agent_rates": t["rates"][0],', "GT velocities"),
    _i("agent_rates_mask", LABEL, False, _V3T, '"agent_rates_mask": t["rates_mask"][0],', "GT mask"),
    _i("agent_label", LABEL, False, _V3T, '"agent_label": torch.tensor(bool(has)),', "GT presence flag"),
    _i("agent_n_raw", LABEL, False, _V3T, '"agent_n_raw": torch.tensor(int(n_raw), dtype=torch.long),', "GT count"),
    _i("agent_n_truncated", LABEL, False, _V3T, '"agent_n_truncated": torch.tensor(max(n_raw - pad, 0),', "GT count"),
    _i("agent_cz", LABEL, False, _V3T, 'item["agent_cz"] = _t3["cz"][0]', "GT box centre height (join3d)"),
    _i("agent_h", LABEL, False, _V3T, 'item["agent_h"] = _t3["h"][0]', "GT box height (join3d)"),
    _i("agent_zh_mask", LABEL, False, _V3T, 'item["agent_zh_mask"] = _t3["zh_mask"][0]', "GT mask (join3d)"),
    _i("bev_occ", LABEL, False, _V3T, 'item["bev_occ"] = torch.from_numpy(_occ)', "GT BEV occupancy rasterised from the GT boxes"),
    _i("bev_mask", LABEL, False, _V3T, 'item["bev_mask"] = torch.from_numpy(_msk)', "its mask"),
    _i("r7_agent_fut", LABEL, False, _V3T, 'return {"r7_agent_fut": box, "r7_agent_fut_valid": valid,', "oracle future agents (refcv7; --r7-agent-future off)"),
    _i("r7_agent_fut_valid", LABEL, False, _V3T, 'return {"r7_agent_fut": box, "r7_agent_fut_valid": valid,', "mask"),
    _i("r7_agent_fut_label", LABEL, False, _V3T, '"r7_agent_fut_label": lab}', "flag"),
    # ---- VIS-1: a per-pixel ray/cuboid z-buffer of the GT CUBOIDS (vis1.py header): a function of GT geometry + the rig, NOT of image content
    _i("agent_vis_full", LABEL, False, _V3T, 'return {"agent_vis_full": torch.from_numpy(nf), "agent_vis_px": torch.from_numpy(nv),',
       "VIS-1 visible fraction of a GT cuboid (precompute_vis1_sidecar.py)"),
    _i("agent_vis_px", LABEL, False, _V3T, 'return {"agent_vis_full": torch.from_numpy(nf), "agent_vis_px": torch.from_numpy(nv),', "VIS-1 visible pixel count of a GT cuboid"),
    _i("agent_vis_known", LABEL, False, _V3T, '"agent_vis_known": torch.from_numpy(kn),', "VIS-1 coverage flag"),
    _i("vis1_rekeyed", LABEL, False, _V3T, '"vis1_rekeyed": torch.tensor(int(rekeyed), dtype=torch.int64)}', "telemetry: 1 iff the window's VIS-1 rows were re-keyed (an ego-stripped frame); logged, never fed"),
    # ---- the SAM3 map targets: OFFLINE and PIXEL-DERIVED (SAM3 segments the camera frames) but a TARGET -- the only offline pixel-derived tensors in the item
    _i("map_frac", LABEL, False, _V3T, '"map_frac": torch.from_numpy(mf.cart[0])', "SAM3 BEV class fractions: a function of the frames, offline; a target", px=True),
    _i("map_seen", LABEL, False, _V3T, '"map_seen": torch.from_numpy(mf.seen[0]),', "SAM3 BEV seen mask", px=True),
    _i("map_label", LABEL, False, _V3T, '"map_label": torch.tensor(True),', "SAM3 label-present flag", px=True),
    _i("map_raw_frame", LABEL, False, _V3T, '"map_raw_frame": torch.tensor(int(mf.frame_idx[0]),', "SAM3 label frame index", px=True),
    _i("map_fine", LABEL, False, _V3T, '"map_fine": torch.from_numpy(fm.codes[0].copy()),', "SAM3 10 cm class codes: the banded map head's target", px=True),
    _i("map_fine_label", LABEL, False, _V3T, '"map_fine_label": torch.tensor(True)}', "SAM3 10 cm label-present flag", px=True),
    _i("map_ep", CALIB, True, _V3T, 'item["map_ep"] = torch.tensor(int(ep.episode_id), dtype=torch.long)',
       "selects the per-clip rig lift geometry (perception_grid / perception_valid); rig calibration, not image content"),
    # ---- C3B depth-aux targets (built from the GT boxes + the rig camera)
    *[_i(n, LABEL, False, _BDF, needle, "box-depth aux target (GT cuboids + rig)")
      for n, needle in (("bd_t8", 'out[f"bd_t{s}"]'), ("bd_fg8", 'out[f"bd_t{s}"]'), ("bd_w8", 'out[f"bd_t{s}"]'),
                        ("bd_t16", 'out[f"bd_t{s}"]'), ("bd_fg16", 'out[f"bd_t{s}"]'), ("bd_w16", 'out[f"bd_t{s}"]'),
                        ("bd_n_boxes", 'out["bd_n_boxes"] ='), ("bd_n_boxes_no_zh", 'out["bd_n_boxes_no_zh"] ='), ("bd_build_s", 'out["bd_build_s"] ='))],
    # ---- refcv8 WP-B v9 per-frame nav + route checkpoint
    _i("r8_nav_token", NAV, True, _R8T, '"r8_nav_token": torch.tensor(int(inp["nav_token"]), dtype=torch.long)', "v9 nav token (popped by the trainer and mapped to nav_cmd)"),
    _i("r8_nav_raw", NAV, True, _R8T, '"r8_nav_raw": torch.as_tensor(np.asarray(inp["nav_args"], np.float32)),', "v9 nav arguments"),
    _i("r8_nav_known", NAV, True, _R8T, '"r8_nav_known": torch.tensor(bool(inp["nav_args_valid"])),', "validity"),
    _i("r8_rc_raw", NAV, True, _R8T, '"r8_rc_raw": torch.as_tensor(np.asarray(inp["rc"], np.float32)),', "v9 route checkpoint"),
    _i("r8_rc_valid", NAV, True, _R8T, '"r8_rc_valid": torch.tensor(bool(inp["rc_valid"])),', "validity"),
    _i("r8_v9_lat_c", LABEL, False, _R8T, 'out["r8_v9_lat_c"] =', "v9 lateral constraint vector: a target"),
    _i("r8_v9_lon_c", LABEL, False, _R8T, 'out["r8_v9_lon_c"] =', "v9 longitudinal constraint vector: a target"),
    _i("r8_v9_speed", LABEL, False, _R8T, 'out["r8_v9_speed"] =', "v9 speed-goal vector: a target"),
)

_MODEL = (
    _m("frames", CAMERA, True, _FWD, "THE camera input", px=True),
    _m("nav_cmd", NAV, True, _FWD, "supplied by the roll (refcv3_arm.py: model(fr_b, nav_cmd=...))"),
    _m("v0", EGO, True, "v0: Tensor | None = None, steps: int = 0,", "pose_last[:, 3]"),
    _m("steps", META, True, "v0: Tensor | None = None, steps: int = 0,", "the sampler's step count: configuration"),
    _m("lan", NAV, False, "lan: Tensor | None = None,", "LAN arm only; the roll passes none"),
    _m("nav_known", NAV, False, "nav_known: Tensor | None = None,", "mask of the nav input; the roll passes none"),
    _m("ego_state", EGO, True, "ego_state: Tensor | None = None,", "ego_state_from_batch(pose_last, actions): v0, a, yaw rate, curvature, keep"),
    _m("withheld_speed", EGO, False, "withheld_speed: Tensor | None = None,", "diagnostic port; the roll passes none"),
    _m("gp_point", LABEL, False, "gp_point: Tensor | None = None,", "DIAGNOSTIC PORT fed from a label (refc_v3.py forward docstring); never supplied by the roll"),
    _m("gp_valid", LABEL, False, "gp_valid: Tensor | None = None,", "mask of the diagnostic port"),
    _m("agent_gt", LABEL, False, "agent_gt: dict | None = None,", "ORACLE agents (--agents oracle); refcv9 is --agents head; the roll passes none"),
    _m("nav_args", NAV, True, "nav_args: Tensor | None = None,", "nav distance / time"),
    _m("v_max_ms", EGO, True, "v_max_ms: Tensor | None = None,", "speed band (r7_roll.py Refcv6Proxy._inputs)"),
    _m("v_max_valid", EGO, True, "v_max_valid: Tensor | None = None,", "its validity"),
    _m("ego_poses", EGO, True, "ego_poses: Tensor | None = None,", "pose_hist (r7_roll.py Refcv6Proxy._inputs)"),
    _m("ego_n_past", META, True, "ego_n_past: int | None = None,", "length of ego_poses"),
    _m("perception_grid", CALIB, True, "perception_grid: Tensor | None = None,",
       "RIG CALIBRATION (refcv6_perception_branch.py FORWARD_EXCLUSIONS: 'THEY ARE NOT LABELS ... rig calibration'): the mount pose back-projected through the road plane"),
    _m("perception_valid", CALIB, True, "perception_valid: Tensor | None = None,", "mask of the lift geometry"),
    _m("ego_actions", EGO, False, "ego_actions: Tensor | None = None,", "the roll passes none"),
    _m("r8_nav", NAV, True, "r8_nav: Tensor | None = None,", "v9 per-frame nav arguments [B, 6]: fed by the roll's nav-feed overlay (battery r8_repoint.py R8_BATCH_KW); a route / nav token, not a function of pixels"),
    _m("r8_rc", NAV, True, "r8_rc: Tensor | None = None) -> dict:", "v9 route checkpoint: fed with r8_nav (same overlay); not a function of pixels"),
)

_ARGV = (
    _a("--v2-cache", CAMERA, True, "the pixel store (PNG-compressed frames): the SOURCE of `frames`, decoded online by LazyV2Episode; swapped through `_window_u8`", px=True),
    _a("--eval-cache", CAMERA, True, "the same store for the held-out split (the W4 roll's frames)", px=True),
    _a("--vis1-sidecar", LABEL, False, "VIS-1: ray/cuboid z-buffer of the GT cuboids (vis1.py header); a loss target only (`_vis1_batch_block`)"),
    _a("--agent-join", LABEL, False, "obstacle.offline 2-D join: GT agents"),
    _a("--join3d", LABEL, False, "obstacle.offline 3-D join: GT cuboids"),
    _a("--join-defect-masks", LABEL, False, "label-defect masks for the joins"),
    _a("--map-gt-root", LABEL, False, "SAM3 map targets: OFFLINE and PIXEL-DERIVED, consumed by the map LOSS only (item keys map_*)", px=True),
    _a("--agent-rig-extrinsics", CALIB, True, "per-clip camera extrinsics: rig calibration (the lift geometry), not image content"),
    _a("--agent-rig-camera", CALIB, True, "selects the rig-camera source for the lift"),
    _a("--clip-clock-sidecar", META, False, "per-clip label clock {grid_start_s, dt_s} from the 100 Hz egomotion log; a time axis for the label join"),
    _a("--pose-sync-sidecar", META, False, "per-row capture-instant offsets from the cameras' timestamps.parquet (X10): TIMESTAMPS, not pixel content; re-samples EGO tracks only "
       "('Adds NO inference input', its own help text)"),
    _a("--r8-v9-labels", NAV, True, "v9 per-frame nav token / args / route checkpoint (train)"),
    _a("--r8-v9-labels-eval", NAV, True, "the same for eval139"),
    _a("--v7-labels", NAV, True, "v7/v8 label release: carries the clip's nav token AND the tactical targets (the NAV half is an input)"),
    _a("--eval-labels", NAV, True, "the v8 label release for the eval split"),
    _a("--speed-max-sidecar-v6", EGO, True, "the speed band (train): ego-derived, future-derived"),
    _a("--speed-max-sidecar-v6-eval", EGO, True, "the speed band (eval)"),
    _a("--anchors", META, True, "anchor bank: a trained-vocabulary constant"),
    _a("--map-hires-class-weights", META, False, "loss weights"),
    _a("--nav-compliance-tau-file", META, False, "a constant for the nav-compliance loss"),
    _a("--init-from", META, True, "checkpoint (weights)"),
)

CENSUS: tuple = _ITEM + _MODEL + _ARGV


def _check_census(census=CENSUS) -> None:
    seen = set()
    for e in census:
        if e.cls not in CLASSES:
            raise CensusGap("bad class %r for %s" % (e.cls, e.name))
        if (e.kind, e.name) in seen:
            raise CensusGap("duplicate census entry %s %s" % (e.kind, e.name))
        seen.add((e.kind, e.name))
        if any(ch in e.name for ch in "*?["):
            raise CensusGap("wildcard census names are refused (a family pattern would classify a NEW key silently): %s" % e.name)
        if e.cls == CAMERA and not e.pixel_derived:
            raise CensusGap("a CAMERA entry must be pixel_derived: %s" % e.name)
        if e.cls in KEEP_OWN and e.cls != LABEL and e.pixel_derived:
            raise CensusGap("%s is pixel_derived but classed %s: only a CAMERA key (swapped) or a LABEL (never read) may be" % (e.name, e.cls))


_check_census()


def census_sha256(census=CENSUS) -> str:
    """sha256 of the census' semantic content (kind, name, class, pixel_derived, reaches -- not the source pins), stamped into the blind roll's bank."""
    return hashlib.sha256(json.dumps([[e.kind, e.name, e.cls, e.pixel_derived, e.reaches] for e in census], sort_keys=False).encode()).hexdigest()


def entries(kind: str, census=CENSUS) -> dict:
    return {e.name: e for e in census if e.kind == kind}


def classify(key: str, kind: str = "item", census=CENSUS) -> Entry:
    try:
        return entries(kind, census)[key]
    except KeyError:
        raise UnclassifiedKey("%s %r is not in the camera-blind census: classify it (CAMERA / EGO / NAV / LABEL / CALIB / META) in "
                              "tanitad/eval/camblind_r9.py::CENSUS before a blind roll can run" % (kind, key)) from None


def camera_keys(kind: str = "item", census=CENSUS) -> tuple:
    return tuple(e.name for e in census if e.kind == kind and e.cls == CAMERA)


def assert_all_classified(item, kind: str = "item", census=CENSUS) -> None:
    known = entries(kind, census)
    unknown = sorted(k for k in item if k not in known)
    if unknown:
        raise UnclassifiedKey("%s keys not in the camera-blind census: %s (each could be an offline camera tensor; classify them)" % (kind, unknown))


# ------------------------------------------------------------------------------------------------------------------ the swap
def swap_item(own: dict, donor: dict, *, census=CENSUS, swap_keys=None) -> dict:
    """The FED item: every CAMERA key from the DONOR window, every other key the window's OWN object (same python object, not a copy).

    ``swap_keys`` exists for the RED arms (a pixels-only swap, a census key dropped, an ego key swapped); the default is the census' CAMERA set.
    Refuses: an unclassified key in ``own`` (fail-closed); a ``swap_keys`` member that is not CAMERA (ego / nav / label / calibration stay the
    window's own); a CAMERA key present in ``own`` but absent from ``donor``."""
    assert_all_classified(own, "item", census)
    cam = camera_keys("item", census) if swap_keys is None else tuple(swap_keys)
    for k in cam:
        c = classify(k, "item", census).cls
        if c != CAMERA:
            raise SwapRefused("swap of %r refused: it is %s, and %s keys must stay the window's OWN (the blind arm removes the camera, nothing else)" % (k, c, c))
    out = dict(own)
    for k in cam:
        if k in own:
            if k not in donor:
                raise KeyError("the donor window has no camera key %r" % k)
            out[k] = donor[k]
    return out


# ------------------------------------------------------- the derangement (VERBATIM from anticheat_common.py, pinned by tests/test_camblind_r9.py)
def sattolo_derangement(episodes, seed: int = 0) -> dict:
    """Sattolo's algorithm on the SORTED episode list with random.Random(seed): one n-cycle, hence no fixed point.  pi[sorted[k]] = a[k]."""
    eps = sorted(int(e) for e in episodes)
    if len(eps) < 2 or len(set(eps)) != len(eps):
        raise ValueError("need >= 2 distinct episodes")
    a = list(eps)
    rnd = random.Random(int(seed))
    for i in range(len(a) - 1, 0, -1):
        j = rnd.randrange(i)
        a[i], a[j] = a[j], a[i]
    pi = {e: d for e, d in zip(eps, a)}
    assert all(pi[e] != e for e in eps), "derangement has a fixed point"
    assert sorted(pi.values()) == eps, "derangement is not a permutation (every episode must be a donor exactly once)"
    seen, cur = 0, eps[0]
    while True:
        cur = pi[cur]
        seen += 1
        if cur == eps[0]:
            break
    assert seen == len(eps), "derangement is not a single cycle"
    return pi


def donor_windows(index, pi: dict) -> dict:
    """index: iterable of (e, t).  {(e, t): (e_d, t_d)}: the donor episode's window of the same rank (min(rank, n_donor - 1))."""
    by_e: dict = {}
    for e, t in index:
        by_e.setdefault(int(e), []).append(int(t))
    for e in by_e:
        by_e[e].sort()
    out = {}
    for e, ts in by_e.items():
        d = pi[e]
        dts = by_e[d]
        for r, t in enumerate(ts):
            out[(e, t)] = (d, dts[min(r, len(dts) - 1)])
    return out


def digest_bytes(b) -> int:
    """blake2b with an 8-byte digest as a non-negative int < 2**64 (fits a uint64 array)."""
    return int.from_bytes(hashlib.blake2b(b, digest_size=8).digest(), "little")


# ------------------------------------------------------------------------------------------ controls (a)-(c), carried over verbatim
def check_changed_fraction(fed, ref) -> dict:
    """Control (b).  PASS iff EVERY window's fed-frame digest differs from the identity roll's (fraction exactly 1.0); a 0 digest = unrecorded = not changed."""
    fed, ref = np.asarray(fed, dtype=np.uint64), np.asarray(ref, dtype=np.uint64)
    if fed.shape != ref.shape or fed.size == 0:
        return {"PASS": False, "n": int(fed.size), "n_changed": 0, "fraction_changed": float("nan"), "reason": "shape mismatch or empty"}
    ok = (fed != ref) & (fed != 0) & (ref != 0)
    n, c = int(fed.size), int(ok.sum())
    return {"PASS": bool(c == n), "n": n, "n_changed": c, "fraction_changed": c / n}


def check_ego_equal(a, b) -> dict:
    """Control (c).  PASS iff every per-window digest of every NON-camera input is equal and non-zero (0 = unrecorded)."""
    a, b = np.asarray(a, dtype=np.uint64), np.asarray(b, dtype=np.uint64)
    if a.shape != b.shape or a.size == 0:
        return {"PASS": False, "n": int(a.size), "n_equal": 0, "reason": "shape mismatch or empty"}
    eq = (a == b) & (a != 0)
    return {"PASS": bool(eq.all()), "n": int(a.size), "n_equal": int(eq.sum())}


def check_bank_equal(base: dict, new: dict, keys=("traj_own", "z_tac", "sel_own")) -> dict:
    """Control (a).  Joined on (e_i, t).  PASS iff the key sets are equal and every array is bit-equal (max abs diff exactly 0.0 AND equal bytes)."""
    kb = {(int(e), int(t)): j for j, (e, t) in enumerate(zip(base["e_i"], base["t"]))}
    kn = {(int(e), int(t)): j for j, (e, t) in enumerate(zip(new["e_i"], new["t"]))}
    rec = {"n_base": len(kb), "n_new": len(kn), "keys_equal": bool(set(kb) == set(kn)), "arrays": {}}
    if not rec["keys_equal"] or not kb:
        rec["PASS"] = False
        return rec
    ks = sorted(kb)
    ib = np.array([kb[k] for k in ks])
    inn = np.array([kn[k] for k in ks])
    allok = True
    for k in keys:
        x, y = np.asarray(base[k])[ib], np.asarray(new[k])[inn]
        same_shape = x.shape == y.shape
        mx = float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64)))) if same_shape and x.size else float("nan")
        beq = bool(same_shape and x.tobytes() == y.tobytes())
        rec["arrays"][k] = {"shape": list(x.shape), "max_abs_diff": mx, "bytes_equal": beq}
        allok = allok and beq and mx == 0.0
    rec["PASS"] = bool(allok)
    return rec


def _tensor_bytes(v):
    import torch
    return np.ascontiguousarray(v.detach().cpu().numpy()).data if torch.is_tensor(v) else None


def other_inputs_digest(item: dict, nav_cmd, v0, ego_state, *, census=CENSUS) -> int:
    """Control (c): digest of EVERY dataset-item entry that is NOT a CAMERA key (by the census, not a hard-coded pair), plus nav_cmd / v0 / ego_state
    as passed to the model.  With the census' CAMERA set == {frames, future_frames} it is byte-identical to camblind_roll._other_inputs_digest."""
    import torch
    cam = set(camera_keys("item", census))
    h = hashlib.blake2b(digest_size=8)
    for k in sorted(item):
        if k in cam:
            continue
        v = item[k]
        h.update(k.encode())
        if torch.is_tensor(v):
            h.update(str(tuple(v.shape)).encode() + str(v.dtype).encode())
            h.update(np.ascontiguousarray(v.detach().cpu().numpy()).data)
        elif isinstance(v, np.ndarray):
            h.update(np.ascontiguousarray(v).data)
        else:
            h.update(repr(v).encode())
    for nm, v in (("nav_cmd", nav_cmd), ("v0", v0), ("ego_state", ego_state)):
        h.update(nm.encode())
        if v is None:
            h.update(b"None")
        elif torch.is_tensor(v):
            h.update(np.ascontiguousarray(v.detach().cpu().numpy()).data)
        else:
            h.update(repr(v).encode())
    return int.from_bytes(h.digest(), "little")


def camera_digests(item: dict, *, census=CENSUS) -> dict:
    """{camera key: blake2b-8 digest of its bytes} for every non-empty CAMERA key of the item (control (b), generalised to every camera key)."""
    import torch
    out = {}
    for k in camera_keys("item", census):
        if k not in item:
            continue
        v = item[k]
        x = v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
        out[k] = digest_bytes(np.ascontiguousarray(x).data) if x.size else 0
    return out


# ------------------------------------------------------------------------------------------------- the roll-side dataset class
def make_camblind_dataset_cls(base, *, camera: str, derange_seed: int = 0, own_every: int = 25, dig: dict | None = None, drange: dict | None = None,
                              census=CENSUS, donor_item_fn=None):
    """W4's ``CamBlindWindows`` with the census wired in.  ``base`` is the roll's eval dataset class (``r7_roll.make_dataset_cls``'s
    ``Refcv6EvalWindows``).  ``camera='identity'`` leaves the battery's own ``_window_u8`` UNCHANGED (control (a)); ``'derange'`` takes every CAMERA key
    of the window from the donor window (the SAME derangement and donor-window rule and seed as W4).

    * ``frames`` (and ``future_frames``, an EMPTY tensor in the roll's class) come from ``_window_u8`` exactly as W4 did;
    * any OTHER census CAMERA key (a future offline camera tensor) is taken from the donor row's item through ``donor_item_fn(ds, row)`` --
      REQUIRED then, else the class refuses to build (a blind arm that cannot swap a declared camera key is refused, not run);
    * ``__getitem__`` classifies every key of every item (``assert_all_classified``): an unknown key stops the roll at window 0."""
    import torch
    if camera not in ("identity", "derange"):
        raise ValueError("camera must be 'identity' or 'derange'")
    extra = tuple(k for k in camera_keys("item", census) if k not in WINDOW_CAMERA_KEYS)
    if camera == "derange" and extra and donor_item_fn is None:
        raise CensusGap("the census has CAMERA keys beyond the window pixels %s but no donor_item_fn was given: the blind arm would leave them as a leak" % (extra,))
    DIG = {} if dig is None else dig
    DRANGE = {} if drange is None else drange

    def _fdig(fr) -> int:
        x = fr.detach().cpu().numpy() if torch.is_tensor(fr) else np.asarray(fr)
        return digest_bytes(np.ascontiguousarray(x).data)

    class CamBlindWindows(base):
        _pi = None
        _dw = None
        _row = None

        def _ensure_map(self):
            cls = type(self)
            if cls._dw is None:
                eps = sorted({int(e) for e, _t in self.index})
                cls._pi = sattolo_derangement(eps, derange_seed)
                cls._dw = donor_windows([(int(e), int(t)) for e, t in self.index], cls._pi)
                cls._row = {(int(e), int(t)): k for k, (e, t) in enumerate(self.index)}
                DRANGE.update({"n_episodes": len(eps), "n_windows": len(self.index), "seed": derange_seed,
                               "fixed_points": int(sum(1 for e in eps if cls._pi[e] == e)),
                               "each_episode_a_donor_once": bool(sorted(cls._pi.values()) == eps),
                               "mapping_sha256": hashlib.sha256("".join(f"{e}:{d}," for e, d in sorted(cls._pi.items())).encode()).hexdigest(),
                               "camera_keys_swapped": list(camera_keys("item", census))})

        def _window_u8(self, i: int) -> dict:
            e_i, t = (int(x) for x in self.index[i])
            if camera == "identity":
                d = super()._window_u8(i)                                 # the battery's own code, unchanged
                DIG[(e_i, t)] = {"fed": _fdig(d["frames"]), "donor": (e_i, t), "own": 0}
                return d
            self._ensure_map()
            e_d, t_d = type(self)._dw[(e_i, t)]
            ep, epd, w = self.episodes[e_i], self.episodes[e_d], self.window
            fr = epd.frames[t_d:t_d + w]                                  # the DONOR's camera pixels
            own = 0
            if i % own_every == 0:                                        # control (b), self-contained sample
                own = _fdig(ep.frames[t:t + w])
            DIG[(e_i, t)] = {"fed": _fdig(fr), "donor": (e_d, t_d), "own": own}
            return {                                                      # r7_roll.make_dataset_cls._window_u8, verbatim, frames swapped
                "frames": fr,
                "actions": ep.actions[t:t + w],
                "future_frames": torch.zeros(0, dtype=torch.uint8),
                "future_actions": ep.actions[t + w:t + w + self.max_horizon],
                "future_poses": ep.poses[t + w:t + w + self.max_horizon],
                "pose_last": ep.poses[t + w - 1],
                "episode_id": ep.episode_id,
            }

        def __getitem__(self, i: int):
            item = super().__getitem__(i)
            assert_all_classified(item, "item", census)                   # FAIL-CLOSED: an unclassified key could be an offline camera tensor
            if camera == "derange" and extra:
                e_i, t = (int(x) for x in self.index[i])
                e_d, t_d = type(self)._dw[(e_i, t)]
                donor = donor_item_fn(self, type(self)._row[(e_d, t_d)])
                item = swap_item(item, donor, census=census, swap_keys=extra)
            return item

    return CamBlindWindows


# --------------------------------------------------------------------------------------------- the census against the sources
def locate(e: Entry, stack_root) -> tuple:
    """-> (line, count): the 1-based line of the first occurrence of the entry's needle (after its anchor) and the number of occurrences from there."""
    txt = (Path(stack_root) / e.src).read_bytes().decode("utf-8").replace("\r\n", "\n")
    start = 0
    if e.after:
        a = txt.find(e.after)
        if a < 0:
            return 0, 0
        start = a
    n = txt.count(e.needle, start)
    if n == 0:
        return 0, 0
    pos = txt.find(e.needle, start)
    return txt[:pos].count("\n") + 1, n


def verify_census(stack_root, census=CENSUS) -> list:
    """-> [(entry, 'path:line', count)] for every entry; a count of 0 is a stale pin."""
    out = []
    for e in census:
        ln, n = locate(e, stack_root)
        out.append((e, "%s:%d" % (e.src, ln) if n else "%s:?" % e.src, n))
    return out


def render_census_md(stack_root, census=CENSUS) -> str:
    rows = ["| kind | key | class | pixel-derived | reaches the roll's forward | source | why |", "|---|---|---|---|---|---|---|"]
    for e, loc, n in verify_census(stack_root, census):
        rows.append("| %s | `%s` | **%s** | %s | %s | `%s` | %s |" % (e.kind, e.name, e.cls, "yes" if e.pixel_derived else "no", "yes" if e.reaches else "no", loc, e.why.replace("|", "/")))
    return "\n".join(rows)


# ---------------------------------------------------------------------------------------------------- the planner taps + the identity
#: the planner OUTPUT: r7_roll.EXTRA_ROW_KEYS (what the roll banks per window) + the candidate fan
PLAN_KEYS = ("traj", "sel_idx", "anchor_traj", "anchor_logits", "lat_logits_tac", "lon_logits_tac", "tacv6_goal_logits", "tacv6_goal_conf",
             "tacv6_lat_logits", "tacv6_lon_logits")


def _flat(o) -> list:
    import torch
    if torch.is_tensor(o):
        return [o.detach().float().cpu().clone()]
    if isinstance(o, dict):
        return [t for k in sorted(o, key=str) for t in _flat(o[k])]
    if isinstance(o, (list, tuple)):
        return [t for v in o for t in _flat(v)]
    return []


class PlannerTaps:
    """Context manager recording every planner INPUT the camera can reach and the planner OUTPUT of a ``RefCV3Model`` forward.

    taps: ``trunk`` (the stride-8 / 16 feature maps), ``map_chan_1m`` and ``box_slots`` (the perception branch's outputs), ``bev_pool`` (the planner BEV),
    ``box_tokens`` (what the planner / tactical layers consumed through ``gated_mix``), ``critic_map`` (the critic's banded map read), ``plan`` (the planner
    OUTPUT: PLAN_KEYS), ``out_all`` (EVERY tensor of the model's output dict).  ``self.rec[tap]`` is a list of tensors; ``self.hooked`` names the taps that
    exist on this model."""

    def __init__(self, model):
        self.model, self.rec, self.hooked, self._undo = model, {}, set(), []

    def _add(self, tap, o):
        self.rec.setdefault(tap, []).extend(_flat(o))

    def __enter__(self):
        from tanitad.models import plan_wires_r9 as PW
        from tanitad.refs import refc as refc_mod
        m = self.model

        def hook(mod, tap, fn=None):
            if mod is None:
                return
            self.hooked.add(tap)
            h = mod.register_forward_hook(lambda _m, _i, o, tap=tap: self._add(tap, o if fn is None else fn(o)))   # returns None: never replaces the output
            self._undo.append(h.remove)
        hook(m, "trunk", lambda o: [o[k] for k in ("fmap_s8", "fmap_s16") if k in o])               # the trunk's stride-8 / stride-16 feature maps
        hook(getattr(m, "_perception", None), "map_chan_1m", lambda o: o.get("map_chan_1m", []))
        hook(getattr(m, "_perception", None), "box_slots", lambda o: o.get("box_slots", []))
        hook(getattr(getattr(m, "_perception", None), "bev_pool", None), "bev_pool")
        hook(m, "plan", lambda o: [o[k] for k in PLAN_KEYS if k in o])                               # the planner OUTPUT (the roll's banked rows + the fan)
        hook(m, "out_all")                                                                          # every tensor of the output dict (internal states too)
        orig_mix, orig_crit = refc_mod._pw_gated_mix, PW.critic_feature_banded

        def mix(attend, tok, pad, n_box, gate):
            self._add("box_tokens", tok[:, tok.shape[1] - n_box:] if n_box > 0 else [])
            return orig_mix(attend, tok, pad, n_box, gate)

        def crit(paths, mb, y_half_m=None):
            r = orig_crit(paths, mb, y_half_m)
            self._add("critic_map", r)
            return r
        refc_mod._pw_gated_mix, PW.critic_feature_banded = mix, crit
        self._undo.append(lambda: setattr(refc_mod, "_pw_gated_mix", orig_mix))
        self._undo.append(lambda: setattr(PW, "critic_feature_banded", orig_crit))
        self.hooked.update({"box_tokens", "critic_map"})
        return self

    def __exit__(self, *a):
        for u in reversed(self._undo):
            u()
        self._undo = []


def roll_feed(model, batch: dict) -> dict:
    """The kwargs of the eval ROLL's model call, built from a batched item exactly as ``refcv3_arm.py`` (the `model(fr_b, nav_cmd=..., v0=..., steps=...,
    ego_state=..., **_geom)` call) and ``r7_roll.Refcv6Proxy._inputs`` (ego_poses / ego_n_past / v_max_*) do.  NOTHING else of the item is passed: this
    function IS the statement 'labels never reach inference' that the identity then tests.
    NOT modelled (the tiny rig has no such seam; they are NAV by source, never pixel functions): the battery's per-frame nav overlay's ``r8_nav`` / ``r8_rc``."""
    import torch
    from tanitad.refs import refc_v3 as v3
    cfg = model.cfg
    fr = batch["frames"]
    kw = {"frames": fr.float().div(255.0) if fr.dtype == torch.uint8 else fr.float()}
    pose_last = batch["pose_last"].float()
    kw.update(nav_cmd=batch["nav_cmd"], v0=pose_last[:, 3], steps=int(cfg.core.decoder.diffusion_steps))
    if getattr(cfg, "ego_state_inject", False):
        kw["ego_state"] = v3.ego_state_from_batch({"pose_last": pose_last, "actions": batch["actions"].float()}, device="cpu")
    lb = getattr(model, "_lift_bank", None) or getattr(model, "_lift_bank_hires", None)
    if lb is not None:
        pg, pv = lb.for_episodes(batch["map_ep"], device="cpu")
        kw.update(perception_grid=pg, perception_valid=pv)
    if "pose_hist" in batch:
        kw.update(ego_poses=batch["pose_hist"].float(), ego_n_past=int(batch["pose_hist"].shape[1]))
    if "v_max_ms" in batch and "v_max_valid" in batch:
        kw.update(v_max_ms=batch["v_max_ms"].float(), v_max_valid=batch["v_max_valid"].float())
    return kw


def run_inference(model, batch: dict, *, ports=None, seed: int = 0):
    """One eval forward on ``roll_feed(batch)`` with the taps armed.  ``ports``: an optional context-manager factory ``ports(model, batch)`` that injects
    synthetic offline camera tensors (test fixtures) BEFORE the taps are armed.  -> {tap: [tensors]}."""
    import contextlib
    import torch
    model.eval()
    port = ports(model, batch) if ports is not None else contextlib.nullcontext()
    with port, PlannerTaps(model) as taps, torch.no_grad():
        torch.manual_seed(seed)
        out = model(**roll_feed(model, batch))
    taps.rec["_out_keys"] = sorted(out)
    return taps.rec, taps.hooked


def diff_taps(a: dict, b: dict) -> dict:
    """{tap: {max_abs, bit_equal, n}}: max |a - b| over the tap's tensors (inf on a count / shape mismatch)."""
    import torch
    res = {}
    for tap in sorted(k for k in a if not k.startswith("_")):
        x, y = a[tap], b.get(tap, [])
        if len(x) != len(y) or any(p.shape != q.shape for p, q in zip(x, y)):
            res[tap] = {"max_abs": float("inf"), "bit_equal": False, "n": len(x)}
            continue
        mx = max([float((p.double() - q.double()).abs().max()) if p.numel() else 0.0 for p, q in zip(x, y)] + [0.0])
        res[tap] = {"max_abs": mx, "bit_equal": all(torch.equal(p, q) for p, q in zip(x, y)), "n": len(x)}
    return res


def perturb(v, key: str = "", gen=None):
    """A deterministic, key-type-aware perturbation of one batched tensor (floats: + noise; uint8: redrawn; ints: +1 mod range; bool: flipped)."""
    import torch
    g = gen if gen is not None else torch.Generator().manual_seed(1234 + sum(map(ord, key)))
    if v.dtype == torch.uint8:
        return torch.randint(0, 256, v.shape, generator=g, dtype=torch.uint8)
    if v.dtype == torch.bool:
        return ~v
    if v.dtype.is_floating_point:
        w = v + 0.5 * torch.randn(v.shape, generator=g, dtype=torch.float32).to(v.dtype)
        return w.clamp(0.0, 1.0) if (key in WINDOW_CAMERA_KEYS and float(v.max()) <= 1.0 + 1e-6) else w
    return (v + 1) % max(int(v.max()) + 1, 2)         # stays inside the observed id range (nav_cmd [1, 2] -> [2, 0])


def couple_ego_track(pert: dict, own: dict, key: str) -> None:
    """The model REFUSES a pose window whose last speed differs from v0 (refc.py ``_residual_prior_inputs``, refcv7 NEW-1), so a perturbed ego key must
    keep its twin consistent: perturbing ``pose_last`` rewrites the last row of ``pose_hist`` to match, and a perturbed ``pose_hist`` keeps the OWN last row."""
    if "pose_hist" not in pert or "pose_last" not in pert:
        return
    if key == "pose_last":
        ph = pert["pose_hist"].clone()
        ph[:, -1] = pert["pose_last"].to(ph.dtype)
        pert["pose_hist"] = ph
    elif key == "pose_hist":
        ph = pert["pose_hist"].clone()
        ph[:, -1] = own["pose_hist"][:, -1]
        pert["pose_hist"] = ph


def identity_sweep(model, own: dict, donor: dict, *, instrument, keys, ports=None, census=CENSUS, kind: str = "item", couple=couple_ego_track) -> tuple:
    """For each key k in ``keys``: perturb the window's OWN ``k`` alone, run the roll feed on ``instrument(own', donor)`` and diff every tap against the
    unperturbed ``instrument(own, donor)``.  -> (reference taps, rows, hooked taps); a row = {key, cls, pixel_derived, taps{...}, max_all, max_plan}."""
    ref, hooked = run_inference(model, instrument(own, donor), ports=ports)
    rows = []
    for k in keys:
        pert = dict(own)
        pert[k] = perturb(own[k], k)
        if couple is not None:
            couple(pert, own, k)
        got, _ = run_inference(model, instrument(pert, donor), ports=ports)
        d = diff_taps(ref, got)
        e = classify(k, kind, census)
        rows.append({"key": k, "cls": e.cls, "pixel_derived": e.pixel_derived, "taps": d,
                     "max_all": max(v["max_abs"] for v in d.values()), "max_plan": d.get("plan", {}).get("max_abs", float("nan"))})
    return ref, rows, hooked


def donor_sweep(model, own: dict, donor: dict, *, instrument, keys, ports=None, census=CENSUS, kind: str = "item") -> list:
    """The POSITIVE control: perturb the DONOR's key ``k`` (own untouched) under ``instrument``.  For a camera key the model reads, the planner MUST move --
    the blind arm really sees the donor's camera, so a 0.0 above is not a dead probe."""
    ref, _ = run_inference(model, instrument(own, donor), ports=ports)
    rows = []
    for k in keys:
        pd = dict(donor)
        pd[k] = perturb(donor[k], k)
        got, _ = run_inference(model, instrument(own, pd), ports=ports)
        d = diff_taps(ref, got)
        e = classify(k, kind, census)
        rows.append({"key": k, "cls": e.cls, "taps": d, "max_all": max(v["max_abs"] for v in d.values()), "max_plan": d.get("plan", {}).get("max_abs", float("nan"))})
    return rows


def violations(rows) -> list:
    """The RED conditions of the identity.  (1) a PIXEL-DERIVED key (camera or SAM3 label) whose perturbation moves ANY planner tap; (2) a LABEL key that
    moves the planner OUTPUT (labels reach inference); (3) any key outside EGO / NAV / CALIB / META that moves the planner (e.g. an un-swapped CAMERA key)."""
    bad = []
    for r in rows:
        if r["pixel_derived"] and r["max_all"] > 0.0:
            bad.append("%s: pixel-derived (%s) yet moves the planner by %.6g" % (r["key"], r["cls"], r["max_all"]))
        elif r["cls"] == LABEL and r["max_plan"] > 0.0:
            bad.append("%s: a LABEL reaches the planner output (%.6g)" % (r["key"], r["max_plan"]))
        elif r["cls"] not in LEGIT_INPUT and r["max_all"] > 0.0:
            bad.append("%s: class %s moves the planner by %.6g" % (r["key"], r["cls"], r["max_all"]))
    return bad
