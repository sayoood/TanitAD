"""Synthetic OpenScene-shaped fixtures: log pickles, a scene-filter yaml and ``navtrain_{current,history}_N.tgz`` archives (NO real data).

Layout copies what was MEASURED on the dev-box ``navtrain_current_25.tgz`` (member ``navtrain_current_25/<log>/<CAM>/<hash>.jpg``) and on
the OpenScene metadata (``cams[CAM]['data_path'] == '<log>/<CAM>/<hash>.jpg'``, frames at 2 Hz).  A log of ``n_frames`` frames has
devkit-valid tokens at positions ``3 .. n_frames - 11`` (window = 4 history + 10 future).
"""
from __future__ import annotations

import io
import os
import pickle
import tarfile

CAMS = ("CAM_L0", "CAM_F0", "CAM_R0")
ALL_CAMS = ("CAM_B0", "CAM_F0", "CAM_L0", "CAM_L1", "CAM_L2", "CAM_R0", "CAM_R1", "CAM_R2")


def tok(log, i):
    return f"{i:02d}{log[0]}"          # index FIRST: the lexicographic order of tokens interleaves the logs (so "sorted" is not "in log order")


def frame_name(log, i, cam):
    return f"{log}/{cam}/{tok(log, i)}_{cam[-2:].lower()}.jpg"


def make_frames(log, n_frames=20, empty_route=(), map_location="us-nv-las-vegas-strip"):
    return [{"token": tok(log, i), "log_name": log, "scene_token": f"sc_{log}", "map_location": map_location,
             "roadblock_ids": [] if i in empty_route else ["rb1"], "timestamp": 1_600_000_000_000_000 + i * 500_000,
             "cams": {c: {"data_path": frame_name(log, i, c), "cam_intrinsic": [[1.0]]} for c in ALL_CAMS}}
            for i in range(n_frames)]


def valid_positions(n_frames=20, nh=4, nf=10):
    return list(range(nh - 1, n_frames - nf))          # 3 .. n_frames - 11


def write_yaml(path, logs, tokens, crlf=False):
    lines = ["_target_: navsim.common.dataclasses.SceneFilter", "_convert_: 'all'", "num_history_frames: 4", "num_future_frames: 10",
             "frame_interval: 1", "has_route: true", "max_scenes: null", "log_names:"] + [f"  - '{x}'" for x in logs] + ["tokens:  "] \
        + [f"  - '{t}'" for t in tokens]
    open(path, "w", encoding="utf-8", newline="").write(("\r\n" if crlf else "\n").join(lines) + "\n")


def write_pickles(meta_dir, frames_by_log):
    os.makedirs(meta_dir, exist_ok=True)
    for log, frames in frames_by_log.items():
        with open(os.path.join(meta_dir, f"{log}.pkl"), "wb") as fh:
            pickle.dump(frames, fh)


def write_tgz(path, root, rels, content=b"jpgbytes"):
    """A gzipped tar whose file members are ``<root>/<rel>`` (directories included, as the real archives have them)."""
    seen_dirs = set()
    with tarfile.open(path, "w:gz") as tf:
        def add_dir(d):
            if d in seen_dirs:
                return
            seen_dirs.add(d)
            ti = tarfile.TarInfo(d)
            ti.type = tarfile.DIRTYPE
            tf.addfile(ti)
        add_dir(root)
        for rel in rels:
            parts = rel.split("/")
            add_dir(f"{root}/{parts[0]}")
            add_dir(f"{root}/{parts[0]}/{parts[1]}")
            data = content if isinstance(content, bytes) else content(rel)
            ti = tarfile.TarInfo(f"{root}/{rel}")
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))


def write_dir(root_dir, rels, content=b"jpgbytes"):
    for rel in rels:
        p = os.path.join(root_dir, *rel.split("/"))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "wb").write(content if isinstance(content, bytes) else content(rel))


def standard_world(tmp, logs=("b_log", "a_log", "d_log", "c_log", "e_log", "f_log"), n_frames=20, with_pickle=None,
                   current_frames=None, history_frames=None, cams=ALL_CAMS, drop_rels=(), extra_filter_tokens=()):
    """Everything on disk for the sample fixer.  Returns a dict of paths.  Defaults: every log has 20 frames; ``current`` archive holds
    frames 3..9, ``history`` frames 0..8 (union 0..9 = the history windows of tokens 3..9); logs with no pickle: ``e_log``."""
    tmp = str(tmp).replace("\\", "/")
    os.makedirs(tmp, exist_ok=True)
    with_pickle = [l for l in logs if l != "e_log"] if with_pickle is None else with_pickle
    cur = list(range(3, 10)) if current_frames is None else list(current_frames)
    his = list(range(0, 9)) if history_frames is None else list(history_frames)
    frames_by_log = {l: make_frames(l, n_frames) for l in logs}
    toks = [tok(l, i) for l in logs for i in valid_positions(n_frames)] + list(extra_filter_tokens)
    yaml_path = f"{tmp}/navtrain.yaml"
    write_yaml(yaml_path, list(logs), toks)
    meta = f"{tmp}/trainval"
    write_pickles(meta, {l: frames_by_log[l] for l in with_pickle})
    cur_rels = [frame_name(l, i, c) for l in logs for i in cur for c in cams if frame_name(l, i, c) not in drop_rels]
    his_rels = [frame_name(l, i, c) for l in logs for i in his for c in cams if frame_name(l, i, c) not in drop_rels]
    shards = f"{tmp}/shards"
    os.makedirs(shards, exist_ok=True)
    write_tgz(f"{shards}/navtrain_current_1.tgz", "navtrain_current_1", cur_rels)
    if his_rels:
        write_tgz(f"{shards}/navtrain_history_1.tgz", "navtrain_history_1", his_rels)
    return {"yaml": yaml_path, "meta": meta, "shards": shards, "frames_by_log": frames_by_log, "cur_rels": cur_rels,
            "his_rels": his_rels, "tmp": tmp}


# ------------------------------------------------------------------------------------------------------------------------------
# frame-bank fixtures: an inputs export in W3's record shape and a FAKE stitcher whose pixels encode the frame's identity
# ------------------------------------------------------------------------------------------------------------------------------
import hashlib  # noqa: E402

import numpy as np  # noqa: E402


def jpg_content(rel):
    """Unique bytes per file, so a frame's pixel value (below) identifies WHICH file was read."""
    return b"JPG:" + rel.encode()


def expected_value(rel_f0):
    """The pixel value the fake stitcher gives the frame whose CAM_F0 file is ``rel_f0`` -- derived HERE, independently of the builder."""
    return 10 + (int.from_bytes(hashlib.sha256(jpg_content(rel_f0)).digest()[:2], "big") % 200)


def make_export(frames_by_log, positions_by_log, split="navtrain"):
    toks = {}
    for log, frames in frames_by_log.items():
        for p in positions_by_log[log]:
            win = frames[p - 3:p + 11]
            toks[tok(log, p)] = {"stage": 1, "frame_type": "ORIGINAL", "log_name": log, "map_name": frames[p]["map_location"],
                                 "scene_token": frames[p]["scene_token"], "frame_tokens": [f["token"] for f in win[:4]],
                                 "timestamps_us": [f["timestamp"] for f in win[:4]],
                                 "cams": {c.lower(): [win[i]["cams"][c]["data_path"] for i in range(4)] for c in CAMS},
                                 "fingerprint": f"fp_{tok(log, p)}"}
    return {"split": split, "n_tokens": len(toks), "tokens": toks}


def make_sample_doc(tokens, status="FIXED"):
    toks = sorted(tokens)
    return {"status": status, "count": len(toks), "tokens": toks,
            "sha256_sorted_tokens": hashlib.sha256("\n".join(toks).encode()).hexdigest()}


class FakeFrame:
    height, width, f_ref, projection = 416, 1024, 488.92398517830253, "cylindrical"


class FakeE2:
    """Stands in for E2's ``build_scene`` module: reads the three files, returns a frame filled with ``expected_value(F0 rel)``."""

    def __init__(self, h=8, w=16, fail_if=None):
        self.FRAME = type("F", (), dict(height=h, width=w, f_ref=488.92398517830253, projection="cylindrical"))
        self.h, self.w, self.fail_if, self.calls = h, w, fail_if, 0

    def build_scene(self, frame_cams, root, cache):
        self.calls += 1
        cd = frame_cams[0]
        paths = [root / cd[c]["data_path"] for c in CAMS]
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} source jpg(s) absent, e.g. {missing[0]}")
        if self.fail_if and self.fail_if(paths):
            raise RuntimeError("synthetic stitch failure")
        data = paths[1].read_bytes()
        if data.startswith(b"BLACK"):
            a = np.zeros((1, self.h, self.w, 3), np.uint8)
        else:
            v = 10 + (int.from_bytes(hashlib.sha256(data).digest()[:2], "big") % 200)
            a = np.full((1, self.h, self.w, 3), v, np.uint8)
        return a, np.zeros((self.h, self.w), np.int8), "rigA", 0.9
