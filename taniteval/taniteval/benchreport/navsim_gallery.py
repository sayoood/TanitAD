#!/usr/bin/env python
"""The MAP-DEPENDENT half of the failure gallery — runs in the NAVSIM venv (Python 3.9), never in ours.

    <navsim venv python> navsim_gallery.py --job <job.json>

It reuses the devkit's OWN visualization module for everything map-shaped — ``add_map_to_bev_ax``
(nuPlan lanes, walkways, carparks, intersections, crosswalks, baseline paths),
``add_annotations_to_bev_ax`` (agent boxes) and ``add_trajectory_to_bev_ax`` (plans) — so no map
geometry is re-implemented here. Beside it, the stitched 3-camera frame the model actually saw (the
bank's ``[4,256,640,3]`` u8) with each plan projected through the bank's own cylindrical ray model
(``camproj.py``, imported from this file's directory).

Input job (written by ``taniteval.benchreport.gallery``): see ``JOB SCHEMA`` below. Output: one PNG per
scene plus ``manifest.json`` — and the caller checks the PNGs, not this script's exit code.

JOB SCHEMA
    {"out_dir", "maps_root", "devkit", "frame": {"height", "width", "f_ref"},
     "style": {"plan": "#2a78d6", ...}, "scenes": [{"token", "scene_token", "pickle", "map_name",
       "frames_npy", "title", "overlay": ["line", ...], "series": [{"name", "label", "poses": [[x,y,h]…],
       "color", "style", "width", "marker"}], "notes": ["…"]}]}
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import pickle
import sys
import traceback

HERE = pathlib.Path(__file__).resolve().parent
# ⛔ NEVER put HERE on sys.path: this package holds modules whose names collide with the STDLIB
# (a `page.py` today, an `html.py` until 2026-09-20 — which shadowed the stdlib `html` package and
# broke pyparsing's `import html.entities` inside matplotlib). camproj is loaded BY PATH instead.
def _load_beside(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Stub:
    def __init__(self, *a, **k):
        self.__dict__.update(k)


class _Tol(pickle.Unpickler):
    """The scene pickles embed PosixPath (written on Linux) and may name classes we do not read."""

    def find_class(self, module, name):
        if module == "pathlib" and name == "PosixPath":
            return pathlib.PurePosixPath
        try:
            return super().find_class(module, name)
        except Exception:                                       # noqa: BLE001
            return type(name, (_Stub,), {"__module__": module})


def load_pickle(path):
    with open(path, "rb") as f:
        return _Tol(f).load()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    a = ap.parse_args(argv)
    job = json.loads(pathlib.Path(a.job).read_text(encoding="utf-8"))
    out_dir = pathlib.Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUPLAN_MAPS_ROOT", job["maps_root"])
    os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
    sys.path.insert(0, job["devkit"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    from nuplan.common.actor_state.state_representation import StateSE2
    from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
    from navsim.common.dataclasses import Annotations, Trajectory
    from navsim.visualization.bev import (add_annotations_to_bev_ax, add_map_to_bev_ax,
                                          add_trajectory_to_bev_ax)
    from navsim.visualization.config import AGENT_CONFIG, BEV_PLOT_CONFIG

    camproj = _load_beside("camproj")

    style = job.get("style", {})
    frame_cfg = job["frame"]
    H, W, F = int(frame_cfg["height"]), int(frame_cfg["width"]), float(frame_cfg["f_ref"])
    # agent boxes stay NEUTRAL so the trajectory colours carry identity (devkit config, re-toned)
    agents = {k: dict(v) for k, v in AGENT_CONFIG.items()}
    for k, v in agents.items():
        v["fill_color"] = style.get("ego_fill", "#e8e8e4") if k == TrackedObjectType.EGO else \
            style.get("agent_fill", "#cfcfc9")
        v["line_color"] = style.get("ink", "#0b0b0b")
        v["fill_color_alpha"] = 1.0
        v["line_width"] = 1.2 if k == TrackedObjectType.EGO else 0.9
    maps = {}
    results = []
    for sc in job["scenes"]:
        rec = {"token": sc["token"], "png": None, "ok": False, "error": None}
        try:
            data = load_pickle(sc["pickle"])
            frame = data["frames"][-1]
            ego_pose = list(frame["ego_status"]["ego_pose"])
            ann = frame["annotations"]
            cam = frame["camera_dict"]
            f0 = cam.get("cam_f0", cam.get("CAM_F0"))
            M = camproj.canonical_basis(f0["sensor2lidar_rotation"])
            cam_t = np.asarray(f0["sensor2lidar_translation"], dtype=np.float64)
            name = sc["map_name"]
            if name not in maps:
                if len(maps) >= 2:
                    maps.clear()                                # bounded cache: maps are heavy
                maps[name] = get_maps_api(job["maps_root"], "nuplan-maps-v1.0", name)
            map_api = maps[name]

            fig = plt.figure(figsize=(13.2, 6.4), dpi=110)
            gs = fig.add_gridspec(2, 2, width_ratios=[1.02, 1.28], height_ratios=[1.0, 0.72],
                                  wspace=0.06, hspace=0.10)
            ax_bev = fig.add_subplot(gs[:, 0])
            ax_cam = fig.add_subplot(gs[0, 1])
            ax_txt = fig.add_subplot(gs[1, 1])

            # ---- BEV: the devkit's own map + boxes, then our plans
            xs = [0.0]
            ys = [0.0]
            for s in sc["series"]:
                p = np.asarray(s["poses"], dtype=np.float64)
                if p.size:
                    xs += list(p[:, 0])
                    ys += list(p[:, 1])
            fwd_max = max(24.0, max(xs) + 12.0)
            lat = max(16.0, max(abs(min(ys)), abs(max(ys))) + 10.0)
            radius = max(fwd_max, lat) + 20.0
            BEV_PLOT_CONFIG["figure_margin"] = (2 * radius, 2 * radius)
            add_map_to_bev_ax(ax_bev, map_api, StateSE2(*ego_pose))
            add_annotations_to_bev_ax(ax_bev, Annotations(
                boxes=np.asarray(ann["boxes"], dtype=np.float32),
                names=list(ann["names"]), velocity_3d=np.asarray(ann["velocity_3d"], dtype=np.float32),
                instance_tokens=list(ann["instance_tokens"]), track_tokens=list(ann["track_tokens"])))
            for k, v in agents.items():
                AGENT_CONFIG[k] = v                             # re-toned for THIS figure
            handles = []
            for s in sc["series"]:
                poses = np.asarray(s["poses"], dtype=np.float32)
                cfg = {"line_color": s["color"], "line_color_alpha": 1.0, "line_width": s.get("width", 2.0),
                       "line_style": s.get("style", "-"), "marker": s.get("marker", "o"),
                       "marker_size": s.get("marker_size", 4.5), "marker_edge_color": style.get("surface", "#ffffff"),
                       "zorder": 5}
                if poses.shape == (8, 3):
                    add_trajectory_to_bev_ax(ax_bev, Trajectory(poses), cfg)      # devkit's own drawer
                elif poses.size:
                    pp = np.concatenate([np.zeros((1, 2)), poses[:, :2]])
                    ax_bev.plot(pp[:, 1], pp[:, 0], color=s["color"], linewidth=cfg["line_width"],
                                linestyle=cfg["line_style"], marker=cfg["marker"], markersize=cfg["marker_size"],
                                zorder=5)
                else:                                            # a plan that does not move: mark the origin
                    ax_bev.plot([0], [0], marker="s", markersize=9, color=s["color"], zorder=6,
                                markeredgecolor=style.get("surface", "#ffffff"))
                handles.append(Line2D([0], [0], color=s["color"], lw=2.2, marker=s.get("marker", "o"),
                                      markersize=5, label=s["label"], linestyle=s.get("style", "-")))
            ax_bev.set_aspect("equal")
            ax_bev.set_xlim(lat, -lat)                           # left is +y, mirrored like the devkit
            ax_bev.set_ylim(-12.0, fwd_max)
            ax_bev.set_xticks([])
            ax_bev.set_yticks([])
            ax_bev.set_title("BEV · nuPlan map (lanes, drivable area, agent boxes) · " + sc["map_name"],
                             fontsize=9, color=style.get("ink2", "#52514e"), loc="left")
            ax_bev.legend(handles=handles, loc="lower right", fontsize=7.6, framealpha=0.9, borderpad=0.4)
            # scale bar: 10 m of the map's own metres (the axes are in metres, x mirrored)
            x0, y0 = lat - 2.0, -10.0
            ax_bev.plot([x0, x0 - 10.0], [y0, y0], color=style.get("ink", "#0b0b0b"), lw=2.0, zorder=7,
                        solid_capstyle="butt")
            ax_bev.text(x0 - 5.0, y0 + 0.8, "10 m", fontsize=7.6, ha="center", va="bottom",
                        color=style.get("ink", "#0b0b0b"), zorder=7)

            # ---- camera: the stitched 3-camera frame the model saw, with the plans projected
            frames = np.load(sc["frames_npy"], mmap_mode="r")
            img = np.asarray(frames[int(sc.get("frame_index", frames.shape[0] - 1))])
            ax_cam.imshow(img)
            ax_cam.set_xlim(0, W - 1)
            ax_cam.set_ylim(H - 1, 0)
            for s in sc["series"]:
                poses = np.asarray(s["poses"], dtype=np.float64)
                if not poses.size:
                    continue
                for seg in camproj.polyline_pixels(poses[:, :2], M, cam_t, H, W, F):
                    ax_cam.plot(seg[:, 0], seg[:, 1], color=s["color"], linewidth=s.get("width", 2.0),
                                linestyle=s.get("style", "-"), solid_capstyle="round", zorder=4)
            ax_cam.set_xticks([])
            ax_cam.set_yticks([])
            ax_cam.set_title("the model's input frame: 3-camera stitch -> 256x640 cylindrical (120 deg), "
                             "plans projected", fontsize=9, color=style.get("ink2", "#52514e"), loc="left")

            # ---- text overlay
            ax_txt.axis("off")
            ax_txt.text(0.0, 1.0, sc["title"], fontsize=10.5, fontweight="bold", va="top", ha="left",
                        color=style.get("ink", "#0b0b0b"), transform=ax_txt.transAxes)
            ax_txt.text(0.0, 0.90, "\n".join(sc["overlay"]), fontsize=9.2, va="top", ha="left",
                        color=style.get("ink", "#0b0b0b"), transform=ax_txt.transAxes, linespacing=1.5)
            if sc.get("notes"):
                ax_txt.text(0.0, 0.02, "\n".join(sc["notes"]), fontsize=8.2, va="bottom", ha="left",
                            color=style.get("ink2", "#52514e"), transform=ax_txt.transAxes, linespacing=1.4)
            png = out_dir / ("scene_%s.png" % sc["token"])
            fig.savefig(png, bbox_inches="tight", facecolor=style.get("surface", "#ffffff"))
            plt.close(fig)
            rec["png"] = str(png)
            rec["ok"] = png.exists() and png.stat().st_size > 5000
            rec["bytes"] = png.stat().st_size if png.exists() else 0
        except Exception as e:                                   # noqa: BLE001
            rec["error"] = "%s: %s" % (type(e).__name__, e)
            rec["traceback"] = traceback.format_exc()[-1200:]
            try:
                plt.close("all")
            except Exception:                                    # noqa: BLE001
                pass
        results.append(rec)
        print("[gallery] %s %s" % (rec["token"], "OK" if rec["ok"] else "FAILED: %s" % rec["error"]), flush=True)
    manifest = {"n": len(results), "n_ok": sum(1 for r in results if r["ok"]), "scenes": results,
                "devkit": job["devkit"], "maps_root": job["maps_root"], "python": sys.version.split()[0]}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return 0 if manifest["n_ok"] == manifest["n"] else 1


if __name__ == "__main__":
    sys.exit(main())
