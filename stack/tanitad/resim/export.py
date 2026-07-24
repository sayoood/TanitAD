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
        "nav_commands": [str, ...],   # strategic route/goal labels (nav_cmd idx)
        "corpora": [str, ...],
        "uncalibrated_corpora": [str, ...],  # camera overlay off (BEV fallback)
        "arms": [ {name, color, ckpt, ade, fde, latency_p50}, ... ],
        "episodes": [ {idx, corpus_tag, n_steps, per_arm_ade:{name:ade},
                       worst_step, worst_ade, maneuver_counts:{cls:n}, thumb},
                      ... ]
      },
      "episodes": [
        { "idx": int, "corpus_tag": str, "steps": [
            { "step": int, "frame": "ep<i>_step<j>.jpg",
              "gt_wp_img": [[u,v]...] | null,  # pixel path (origin-prefixed),
                                         #   in the EXPORTED frame's pixel space;
                                         #   null on an uncalibrated corpus step
                                         #   (BEV-only fallback)
              "gt_wp_bev": [[x,y]...],   # ego metres (origin-prefixed): +x fwd,
                                         #   +y left — the BEV master panel maths
              "gt_action": {steer, accel},
              "maneuver": int | null,    # kinematic maneuver class at this
                                         #   window (index into meta.maneuver_
                                         #   classes); null if not computable
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

Per-window ``maneuver`` labels are the kinematic pseudo-labels of
``scripts/refb_labels.py`` (:func:`~refb_labels.classify_maneuver` via
:func:`~refb_labels.maneuver_labels`) — the SAME class each window's REF-B arm
targets, but computed once here from ground-truth ego poses so the display is
arm-independent (every bundle gets a maneuver strip, even a main-only run).
Pass ``ego_poses`` (``{ep_index: poses[T, 4]}``) to enable it; without it the
``maneuver`` field is ``null`` and the SPA simply hides the maneuver strip.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from tanitad.replay.engine import WAYPOINT_STEPS, TimestepRecord
from tanitad.replay.rr_log import cam_for_corpus, to_image_plane


def _labels_module():
    """Lazy import of ``scripts/refb_labels.py`` (kinematic maneuver labels).

    ``scripts`` is not a package, so mirror
    :func:`tanitad.replay.arms._script_module`: try a plain import first (the
    test suite / a pod put ``scripts`` on the path) and fall back to adding the
    sibling ``scripts`` dir. Reuses the pinned thresholds rather than
    re-deriving them, so the display class matches REF-B's target exactly.
    """
    try:
        return importlib.import_module("refb_labels")
    except ModuleNotFoundError:
        scripts = Path(__file__).resolve().parents[2] / "scripts"
        if not (scripts / "refb_labels.py").exists():
            raise
        if str(scripts) not in sys.path:
            sys.path.insert(0, str(scripts))
        return importlib.import_module("refb_labels")


def _episode_maneuvers(poses: Any, labels_mod) -> Any | None:
    """Per-timestep maneuver class ids for one episode's poses ``[T, 4]``.

    Returns the ``refb_labels.maneuver_labels`` array (length ``T - horizon``;
    entry ``t`` is the class comparing pose ``t`` with pose ``t + horizon``) or
    ``None`` when the poses are too short / malformed to label — a display aid
    degrades to "no strip", it never crashes the export.
    """
    try:
        import torch
        p = poses if isinstance(poses, torch.Tensor) else torch.as_tensor(
            np.asarray(poses), dtype=torch.float64)
        H = int(labels_mod.LABEL_HORIZON)
        if p.ndim != 2 or p.shape[1] != 4 or p.shape[0] <= H:
            return None
        return labels_mod.maneuver_labels(p, H)
    except Exception:
        return None

# TanitResim design-language palette (branded, distinct from the rerun
# ARM_COLORS in tanitad.replay.arms, which stays the canonical rerun palette).
# MAIN = gold/amber, REF-A = cyan, REF-B = magenta, GT = near-white (dashed).
RESIM_COLORS: dict[str, str] = {
    "gt": "#eef2f7",
    "main": "#f5b301",
    "refa": "#22d3ee",
    "refb": "#e35ce0",
}

# Strategic route/goal command names, indexed by ``ArmOutput.nav_cmd``. MUST
# stay in lockstep with ``tanitad.refs.refb.NAV_COMMANDS`` — mirrored (not
# imported) so the exporter carries no torch/refb dependency. Emitted into each
# bundle's ``meta.nav_commands`` so the SPA maps nav_cmd -> label from DATA
# rather than a hard-coded array (the old ["straight","left","right"] guess
# mislabelled indices 0 and 3). The decoded strategic route/goal in the viz
# standard's text HUD ("strategic: route {route}") reads this table.
NAV_COMMANDS: tuple[str, ...] = ("follow", "left", "right", "straight")


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
                  ego_poses: Mapping[int, Any] | None = None,
                  jpeg_quality: int = 80, max_w: int = 640,
                  arm_gates: dict | None = None,
                  gates_summary: dict | None = None,
                  uncalibrated_corpora: Iterable[str] | None = None) -> dict:
    """Stream ``records`` into a portable session bundle under ``out_dir``.

    ``records`` is consumed once (a generator is fine); every record must
    carry a displayable ``frame`` (run the engine with ``emit_frames=True``).
    Returns the written ``session.json`` dict. Raises on an empty stream or a
    frameless record — a bundle with nothing to show is a wiring bug, not an
    empty success (repo fail-loud doctrine).

    ``uncalibrated_corpora`` names corpora whose front-camera geometry is not
    recoverable (e.g. cosmos f-theta generations without a verified per-clip
    calib): their steps get ``gt_wp_img=null`` / ``wp_img=null`` so the SPA
    draws the raw frame with a "camera overlay disabled -- see BEV" note and
    the metric BEV inset carries the GT-vs-pred comparison instead. This is
    THE STANDARD's BEV-only fallback path (``taniteval.corpus_overlay``). The
    frame JPEG and all BEV/metric data are still written; only the image-plane
    projection is skipped. Omit it (default) to project every corpus.

    ``arm_gates`` (per-arm compact D1-D3 gate block) and ``gates_summary``
    (shared baselines + Phase-0 GO verdict) come from
    ``compare_arms.compact_gate_blocks`` — the SAME formal gate code the
    ``compare_arms.py`` / ``watch_gates.py`` harness uses, so the UI gate panel
    reconciles with a standalone gate run. Both optional (overlays-only bundle
    when omitted).

    ``ego_poses`` maps ``ep_index`` -> that episode's ground-truth poses
    ``[T, 4]`` (x, y, yaw, v). When given, each step gets a kinematic
    ``maneuver`` class id (``scripts/refb_labels``) indexed by the record's
    anchor ``t``; omit it to skip maneuver labelling.
    """
    out = Path(out_dir)
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    uncal = set(uncalibrated_corpora or ())        # corpora w/o camera overlay

    # Per-episode kinematic maneuver labels (computed once, indexed by anchor
    # t). Only when ego_poses is supplied — otherwise every step's maneuver is
    # null and the SPA hides the strip.
    labels_mod = _labels_module() if ego_poses else None
    ep_man_labels: dict[int, Any] = {}
    n_man_classes = 0
    if maneuver_classes:
        n_man_classes = len(list(maneuver_classes))

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
    ep_man_counts: dict[int, dict[int, int]] = {}   # ep -> {class id: count}

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
            ep_man_counts[ep] = {}
            if labels_mod is not None:
                ep_man_labels[ep] = _episode_maneuvers(
                    ego_poses.get(ep), labels_mod)
        if rec.corpus not in corpora_seen:
            corpora_seen.append(rec.corpus)

        j = ep_counter[ep]
        ep_counter[ep] = j + 1
        frame_name = f"ep{ep}_step{j}.jpg"
        fw, fh = _write_frame(rec.frame, frames_dir / frame_name,
                              max_w, jpeg_quality)
        cam = cam_for_corpus(rec.corpus)   # per-corpus overlay geometry (D-016)
        cam_ok = rec.corpus not in uncal   # False -> BEV-only fallback step

        # Kinematic maneuver at this window's anchor (t = record's anchor
        # frame). None when no ego_poses / anchor beyond the labelable span.
        maneuver: int | None = None
        lab = ep_man_labels.get(ep)
        if lab is not None and 0 <= int(rec.t) < int(lab.shape[0]):
            maneuver = int(lab[int(rec.t)])
            if 0 <= maneuver < n_man_classes or n_man_classes == 0:
                ep_man_counts[ep][maneuver] = \
                    ep_man_counts[ep].get(maneuver, 0) + 1

        gt_wp = np.asarray(rec.gt_waypoints, dtype=np.float64)
        gt_path = np.vstack([[0.0, 0.0], gt_wp])
        step_dict: dict = {
            "step": int(rec.step),
            "frame": frame_name,
            "gt_wp_img": (_round_pts(to_image_plane(gt_path, fh, fw, cam=cam), 1)
                          if cam_ok else None),
            "gt_wp_bev": _round_pts(gt_path, 3),
            "gt_action": {"steer": round(float(rec.gt_action[0]), 4),
                          "accel": round(float(rec.gt_action[1]), 4)},
            "maneuver": maneuver,
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
                wp_img = (_round_pts(to_image_plane(path, fh, fw, cam=cam), 1)
                          if cam_ok else None)
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

    man_names = list(maneuver_classes) if maneuver_classes else None

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
        # Maneuver histogram keyed by class name (or "m<id>") for the card
        # ribbon, in canonical class order so colors line up with the strip.
        man_counts = ep_man_counts.get(ep, {})
        maneuver_counts = {
            (man_names[c] if man_names and 0 <= c < len(man_names)
             else f"m{c}"): man_counts[c]
            for c in sorted(man_counts)}
        ep_meta.append({
            "idx": ep, "corpus_tag": ep_corpus[ep], "n_steps": len(steps),
            "per_arm_ade": per_arm, "worst_step": worst_j,
            "worst_ade": round(float(worst_ade), 4),
            "maneuver_counts": maneuver_counts,
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
        "gates": (arm_gates.get(n) if arm_gates else None),
    } for n in arm_names]

    session = {
        "meta": {
            "session_name": session_name,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "waypoint_steps": list(WAYPOINT_STEPS),
            "maneuver_classes": (list(maneuver_classes)
                                 if maneuver_classes else None),
            "nav_commands": list(NAV_COMMANDS),   # strategic route/goal labels
            "corpora": list(corpora) if corpora else corpora_seen,
            "uncalibrated_corpora": sorted(uncal),  # BEV-only fallback corpora
            "arms": arms_meta,
            "episodes": ep_meta,
            "gates": gates_summary,      # shared baselines + Phase-0 GO verdict
        },
        "episodes": episodes_out,
    }

    (out / "session.json").write_text(
        json.dumps(session, indent=1), encoding="utf-8")
    return session
