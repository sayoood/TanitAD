"""Session-bundle exporter: replay records -> a portable TanitResim bundle.

A *session bundle* is a directory that the TanitResim web app can serve with
zero external state:

    <bundle>/
        session.json                 # schema below, RELATIVE paths only
        frames/ep<i>_step<j>.jpg      # one shared camera frame per step

``session.json`` schema (JSON, all coordinates pre-computed so the front-end
never has to know the projection maths)::

    {
      "meta": {
        "session_name": str,
        "created": "YYYY-MM-DD HH:MM:SS",
        "waypoint_steps": [5, 10, 15, 20],
        "maneuver_classes": [str, ...] | null,
        "corpora": [str, ...],
        "arms": [ {name, color, ckpt, ade, fde, latency_p50}, ... ],
        "episodes": [ {idx, corpus_tag, n_steps, per_arm_ade:{name:ade},
                       worst_step, worst_ade, thumb}, ... ]
      },
      "episodes": [
        { "idx": int, "corpus_tag": str, "steps": [
            { "step": int, "frame": "ep<i>_step<j>.jpg",
              "gt_wp_img": [[u,v]...],   # pixel path (origin-prefixed), in the
                                         #   EXPORTED frame's pixel space
              "gt_wp_bev": [[x,y]...],   # ego metres (origin-prefixed): +x fwd,
                                         #   +y left — the BEV master panel maths
              "gt_action": {steer, accel},
              "ego": {speed, yaw_rate},
              "arms": { name: {
                  "wp_img": [[u,v]...] | null,
                  "wp_bev": [[x,y]...] | null,
                  "steer": float | null, "accel": float | null,
                  "ade": float | null,
                  "heads": { imag_rel?:{k:v}, sigma?, conf?, ood?,
                             maneuver_probs?:[...], maneuver_gt?, nav_cmd? }
              } } }, ... ] }, ...
      ]
    }

Image-plane waypoint projection reuses the replay app's pinhole
(:func:`tanitad.replay.rr_log.to_image_plane`) with the record's per-corpus
camera model (:func:`~tanitad.replay.rr_log.cam_for_corpus`) so the camera fans
line up with the rerun viz exactly; BEV coordinates are the engine's ego-frame
metric waypoints untouched.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from tanitad.replay.engine import WAYPOINT_STEPS, TimestepRecord
from tanitad.replay.rr_log import cam_for_corpus, to_image_plane

# TanitResim design-language palette (branded, distinct from the rerun
# ARM_COLORS in tanitad.replay.arms, which stays the canonical rerun palette).
# MAIN = gold/amber, REF-A = cyan, REF-B = magenta, GT = near-white (dashed).
RESIM_COLORS: dict[str, str] = {
    "gt": "#eef2f7",
    "main": "#f5b301",
    "refa": "#22d3ee",
    "refb": "#e35ce0",
}


def resim_color(name: str) -> str:
    """Branded arm color as a ``#rrggbb`` hex string; a stable (hash-salt-free)
    fallback for unknown arm names so a bundle is reproducible."""
    if name in RESIM_COLORS:
        return RESIM_COLORS[name]
    h = int(hashlib.md5(name.encode("utf-8")).hexdigest(), 16)
    return "#{:02x}{:02x}{:02x}".format(
        64 + h % 160, 64 + (h // 160) % 160, 64 + (h // 25600) % 160)


def static_dir() -> Path:
    """Absolute path to the bundled SPA static assets (index/app/style)."""
    return Path(__file__).resolve().parent / "static"


def _frame_rgb(frame: np.ndarray) -> np.ndarray:
    """``frame_u8`` output ([H,W,3] camera stack or [H,W] BEV) -> RGB uint8."""
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return np.ascontiguousarray(arr).astype(np.uint8)


def _write_frame(frame: np.ndarray, path: Path, max_w: int,
                 quality: int) -> tuple[int, int]:
    """Save one camera frame as JPEG (downscale if wider than ``max_w``).
    Returns the written ``(width, height)`` — the pixel space the image-plane
    waypoint projection must use so overlays line up with the JPEG."""
    from PIL import Image

    img = Image.fromarray(_frame_rgb(frame), mode="RGB")
    if img.width > max_w:
        new_h = max(1, round(img.height * max_w / img.width))
        img = img.resize((max_w, new_h), Image.BILINEAR)
    img.save(str(path), format="JPEG", quality=quality)
    return img.width, img.height


def _round_pts(pts: np.ndarray, nd: int) -> list[list[float]]:
    return [[round(float(a), nd), round(float(b), nd)] for a, b in pts]


def _ade(wp: np.ndarray, gt: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(wp) - gt, axis=-1).mean())


def export_bundle(records: Iterable[TimestepRecord], out_dir: str | Path,
                  session_name: str, *,
                  corpora: Sequence[str] | None = None,
                  arm_ckpts: dict[str, str] | None = None,
                  maneuver_classes: Sequence[str] | None = None,
                  jpeg_quality: int = 80, max_w: int = 640) -> dict:
    """Stream ``records`` into a portable session bundle under ``out_dir``.

    ``records`` is consumed once (a generator is fine); every record must
    carry a displayable ``frame`` (run the engine with ``emit_frames=True``).
    Returns the written ``session.json`` dict. Raises on an empty stream or a
    frameless record — a bundle with nothing to show is a wiring bug, not an
    empty success (repo fail-loud doctrine).
    """
    out = Path(out_dir)
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    arm_names: list[str] = []
    corpora_seen: list[str] = []
    ep_order: list[int] = []                       # ep_index in first-seen order
    ep_steps: dict[int, list[dict]] = {}
    ep_corpus: dict[int, str] = {}
    ep_counter: dict[int, int] = {}
    # accumulators for summaries
    arm_ade: dict[str, list[float]] = {}
    arm_fde: dict[str, list[float]] = {}
    arm_lat: dict[str, list[float]] = {}
    ep_arm_ade: dict[int, dict[str, list[float]]] = {}

    n_records = 0
    for rec in records:
        if rec.frame is None:
            raise ValueError(
                f"export needs displayable frames: record at step {rec.step} "
                f"has frame=None — run the engine with emit_frames=True")
        n_records += 1
        ep = rec.ep_index
        if ep not in ep_steps:
            ep_order.append(ep)
            ep_steps[ep] = []
            ep_corpus[ep] = rec.corpus
            ep_counter[ep] = 0
            ep_arm_ade[ep] = {}
        if rec.corpus not in corpora_seen:
            corpora_seen.append(rec.corpus)

        j = ep_counter[ep]
        ep_counter[ep] = j + 1
        frame_name = f"ep{ep}_step{j}.jpg"
        fw, fh = _write_frame(rec.frame, frames_dir / frame_name,
                              max_w, jpeg_quality)
        cam = cam_for_corpus(rec.corpus)   # per-corpus overlay geometry (D-016)

        gt_wp = np.asarray(rec.gt_waypoints, dtype=np.float64)
        gt_path = np.vstack([[0.0, 0.0], gt_wp])
        step_dict: dict = {
            "step": int(rec.step),
            "frame": frame_name,
            "gt_wp_img": _round_pts(to_image_plane(gt_path, fh, fw, cam=cam), 1),
            "gt_wp_bev": _round_pts(gt_path, 3),
            "gt_action": {"steer": round(float(rec.gt_action[0]), 4),
                          "accel": round(float(rec.gt_action[1]), 4)},
            "ego": {"speed": round(float(rec.speed), 4),
                    "yaw_rate": round(float(rec.yaw_rate), 4)},
            "arms": {},
        }
        worst_this_step = 0.0
        for name, o in rec.arms.items():
            if name not in arm_names:
                arm_names.append(name)
                arm_ade[name] = []
                arm_fde[name] = []
                arm_lat[name] = []
            arm_lat[name].append(float(o.latency_ms))
            ep_arm_ade[ep].setdefault(name, [])

            wp_img = wp_bev = None
            ade = None
            if o.waypoints is not None:
                wp = np.asarray(o.waypoints, dtype=np.float64)
                path = np.vstack([[0.0, 0.0], wp])
                wp_img = _round_pts(to_image_plane(path, fh, fw, cam=cam), 1)
                wp_bev = _round_pts(path, 3)
                ade = round(_ade(wp, gt_wp), 4)
                fde = float(np.linalg.norm(wp[-1] - gt_wp[-1]))
                arm_ade[name].append(ade)
                arm_fde[name].append(fde)
                ep_arm_ade[ep][name].append(ade)
                worst_this_step = max(worst_this_step, ade)

            heads: dict = {}
            if o.imag_rel:
                heads["imag_rel"] = {str(int(k)): round(float(v), 4)
                                     for k, v in o.imag_rel.items()}
            if o.sigma is not None:
                heads["sigma"] = round(float(o.sigma), 4)
            if o.conf is not None:
                heads["conf"] = round(float(o.conf), 4)
            if o.ood is not None:
                heads["ood"] = round(float(o.ood), 4)
            if o.maneuver_probs is not None:
                heads["maneuver_probs"] = [round(float(p), 4)
                                           for p in o.maneuver_probs]
            if o.maneuver_gt is not None:
                heads["maneuver_gt"] = int(o.maneuver_gt)
            if o.nav_cmd is not None:
                heads["nav_cmd"] = int(o.nav_cmd)

            step_dict["arms"][name] = {
                "wp_img": wp_img, "wp_bev": wp_bev,
                "steer": (round(float(o.action[0]), 4)
                          if o.action is not None else None),
                "accel": (round(float(o.action[1]), 4)
                          if o.action is not None else None),
                "ade": ade, "heads": heads,
            }
        step_dict["_worst"] = round(worst_this_step, 4)
        ep_steps[ep].append(step_dict)
        rec.frame = None                            # release; keep memory flat

    if n_records == 0:
        raise ValueError("export_bundle got zero records — nothing to export")

    # -- per-episode summaries + worst step ---------------------------------
    ep_meta: list[dict] = []
    episodes_out: list[dict] = []
    for ep in ep_order:
        steps = ep_steps[ep]
        worsts = [s.pop("_worst") for s in steps]
        worst_j = int(np.argmax(worsts)) if worsts else 0
        worst_ade = worsts[worst_j] if worsts else 0.0
        per_arm = {n: round(float(np.mean(v)), 4)
                   for n, v in ep_arm_ade[ep].items() if v}
        ep_meta.append({
            "idx": ep, "corpus_tag": ep_corpus[ep], "n_steps": len(steps),
            "per_arm_ade": per_arm, "worst_step": worst_j,
            "worst_ade": round(float(worst_ade), 4),
            "thumb": f"ep{ep}_step0.jpg",
        })
        episodes_out.append({"idx": ep, "corpus_tag": ep_corpus[ep],
                             "steps": steps})

    def _p50(xs: list[float]) -> float:
        return round(float(np.percentile(xs, 50)), 3) if xs else 0.0

    arms_meta = [{
        "name": n,
        "color": resim_color(n),
        "ckpt": (Path(arm_ckpts[n]).name
                 if arm_ckpts and n in arm_ckpts else None),
        "ade": round(float(np.mean(arm_ade[n])), 4) if arm_ade[n] else None,
        "fde": round(float(np.mean(arm_fde[n])), 4) if arm_fde[n] else None,
        "latency_p50": _p50(arm_lat[n]),
    } for n in arm_names]

    session = {
        "meta": {
            "session_name": session_name,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "waypoint_steps": list(WAYPOINT_STEPS),
            "maneuver_classes": (list(maneuver_classes)
                                 if maneuver_classes else None),
            "corpora": list(corpora) if corpora else corpora_seen,
            "arms": arms_meta,
            "episodes": ep_meta,
        },
        "episodes": episodes_out,
    }

    (out / "session.json").write_text(
        json.dumps(session, indent=1), encoding="utf-8")
    return session
