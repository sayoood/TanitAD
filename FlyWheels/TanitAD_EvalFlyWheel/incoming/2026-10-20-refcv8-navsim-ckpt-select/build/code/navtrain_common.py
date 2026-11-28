#!/usr/bin/env python3
"""navtrain_common.py -- shared helpers of the navtrain selection ports (numpy + stdlib; any python; no torch, no devkit).

* ``parse_scene_filter``  the OFFICIAL navtrain scene-filter yaml (v1.1 tree 3e8291b), parsed without a yaml dependency;
* ``load_log_pickle``     an OpenScene ``navsim_logs/<split>/<log>.pkl`` through a TOLERANT unpickler that never imports or executes
                          anything outside a small allow-list (the pickles are third-party data; unknown classes become inert stubs);
* ``scene_window``        the devkit's ``filter_scenes`` window of one token (``navsim/common/dataloader.py:14-66``, MEASURED against
                          W3's export of 334 navtest tokens in ``tests/test_fix_navtrain_sample.py``);
* ``iter_members``        the files of a shard source -- a ``.tgz`` (streamed, never extracted) or an extracted directory;
* ``readable_rels``       the ONE definition of the camera files a token needs (AMENDMENT 1a / KB3): 4 history frames x {CAM_L0, CAM_F0,
                          CAM_R0} = 12 jpgs.  Used by the sample fixer (to decide readability) AND the frame-bank builder (to demand
                          them), so the two can never disagree.

Layout facts (MEASURED 2026-10-20 on the dev-box ``navtrain_current_25.tgz`` by streaming its first members): every file member is
``navtrain_current_25/<log_name>/<CAM_x>/<hash>.jpg`` -- one root directory, then the SAME ``<log>/<CAM>/<file>`` path the metadata's
``cams[CAM]['data_path']`` carries (``<log>/CAM_F0/<hash>.jpg``).  The history archives are taken to be ``navtrain_history_N/<log>/...``
(the PI-stated layout); the code strips the first path component and does not care what it is called.
"""
from __future__ import annotations

import hashlib
import importlib
import os
import pathlib
import pickle
import re
import tarfile

CAMS = ("CAM_L0", "CAM_F0", "CAM_R0")
#: sha256 of the official navtrain scene filter, v1.1 tree 3e8291b (GROUNDWORK.md section 1, MEASURED on dev box and Thor).
NAVTRAIN_FILTER_SHA256 = "c37fea567a0cfdbc29076cca893d4f5dd32db59baec18ae214527206d6b64e6f"
TARGET_TOKENS = 2000          # D-R8-CKPT-RULE (1): "stopping at the first log that takes the total to >= 2,000 tokens"
MIN_TOKENS = 1000             # D-R8-CKPT-RULE (1): "If < 1,000 tokens are available the selection is UNDERPOWERED"
SHARD_NAME = re.compile(r"^navtrain_(current|history)_(\d+)(\.tgz|\.tar\.gz|\.tar)?$")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sha256_tokens(tokens) -> str:
    """sha256 of the SORTED token list joined by newlines -- exactly what ``r8ns_navtrain_select_LAUNCH.sh`` gate G1 re-derives."""
    return hashlib.sha256("\n".join(sorted(tokens)).encode()).hexdigest()


# --------------------------------------------------------------------------------------------------------------------------------
# the scene filter
# --------------------------------------------------------------------------------------------------------------------------------
def parse_scene_filter(path: str) -> dict:
    """``{header, log_names, tokens, file_sha256}`` of a NavSim scene-filter yaml (``- 'x'`` list items), without PyYAML."""
    raw = open(path, "rb").read()
    header, lists, cur = {}, {"log_names": [], "tokens": []}, None
    for line in raw.decode("utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if not line.startswith((" ", "-")) and ":" in s:
            k, v = s.split(":", 1)
            cur = k if k in lists else None
            if cur is None:
                header[k] = v.strip().strip("'\"")
            continue
        if cur and s.startswith("- "):
            lists[cur].append(s[2:].strip().strip("'\""))
    for key in ("num_history_frames", "num_future_frames", "frame_interval", "has_route"):
        if key not in header:
            raise ValueError(f"scene filter {path}: header key {key!r} absent")
    h = {"num_history_frames": int(header["num_history_frames"]), "num_future_frames": int(header["num_future_frames"]),
         "frame_interval": int(header["frame_interval"]), "has_route": header["has_route"].lower() == "true",
         "max_scenes": header.get("max_scenes")}
    return {"header": h, "log_names": lists["log_names"], "tokens": lists["tokens"], "file_sha256": hashlib.sha256(raw).hexdigest()}


# --------------------------------------------------------------------------------------------------------------------------------
# the log pickles (third-party data -> tolerant, allow-listed unpickler)
# --------------------------------------------------------------------------------------------------------------------------------
class _Stub:
    """An inert stand-in for a class the allow-list does not name: stores its constructor kwargs, executes nothing."""

    def __init__(self, *a, **k):
        self.__dict__.update(k)


_ALLOWED_BUILTINS = {"set", "frozenset", "complex", "slice", "range", "bytearray", "tuple", "list", "dict", "int", "float", "str",
                     "bytes", "bool"}
_ALLOWED = {("collections", "OrderedDict"), ("numpy", "ndarray"), ("numpy", "dtype"),
            ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
            ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"),
            ("numpy.core.numeric", "_frombuffer"), ("numpy._core.numeric", "_frombuffer")}
_PATHS = {(m, n): (pathlib.PurePosixPath if "Posix" in n else pathlib.PureWindowsPath)
          for m in ("pathlib", "pathlib._local") for n in ("PosixPath", "PurePosixPath", "WindowsPath", "PureWindowsPath")}


class TolerantUnpickler(pickle.Unpickler):
    """Allow-list unpickler.  A global outside the list becomes an inert ``_Stub`` subclass -- NOTHING is imported and no foreign callable
    ever runs (``os.system`` as a reduce target would become a stub that does nothing).  ``self.stubbed`` records the names seen."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.stubbed = set()

    def find_class(self, module, name):
        key = (module, name)
        if key in _PATHS:
            return _PATHS[key]
        if module == "builtins" and name in _ALLOWED_BUILTINS:
            return getattr(importlib.import_module("builtins"), name)
        if key in _ALLOWED:
            try:
                return getattr(importlib.import_module(module), name)
            except (ImportError, AttributeError):
                alt = module.replace("numpy.core.", "numpy._core.") if module.startswith("numpy.core.") else module.replace("numpy._core.", "numpy.core.")
                return getattr(importlib.import_module(alt), name)
        self.stubbed.add(f"{module}.{name}")
        return type(name, (_Stub,), {"__module__": module})


def load_log_pickle(path: str):
    """``(frames, stubbed_names)`` of one OpenScene log pickle: a list of per-frame dicts at 2 Hz."""
    with open(path, "rb") as fh:
        u = TolerantUnpickler(fh)
        frames = u.load()
    if not isinstance(frames, list) or not frames or not isinstance(frames[0], dict):
        raise ValueError(f"{path}: not a list of frame dicts")
    return frames, sorted(u.stubbed)


# --------------------------------------------------------------------------------------------------------------------------------
# the scene window and the camera files a token needs
# --------------------------------------------------------------------------------------------------------------------------------
def scene_window(frames: list, pos: int, nh: int = 4, nf: int = 10, has_route: bool = True):
    """The devkit's scene window of the token at log position ``pos``, or ``None`` if the devkit would drop it.

    ``filter_scenes`` (``navsim/common/dataloader.py:14-66``, frame_interval 1) takes every window ``frames[i : i + nh + nf]``; it drops a
    window shorter than ``nh + nf`` and (``has_route``) one whose frame ``nh - 1`` has no ``roadblock_ids``; the token is frame ``nh - 1``.
    So the token at position ``pos`` owns the window starting at ``pos - (nh - 1)``; its 4 history frames are the first ``nh`` of it."""
    start = pos - (nh - 1)
    if start < 0:
        return None
    win = frames[start:start + nh + nf]
    if len(win) < nh + nf:
        return None
    if has_route and len(win[nh - 1].get("roadblock_ids", ())) == 0:
        return None
    return win


def history_rels(win: list, nh: int = 4, cams=CAMS) -> dict:
    """``{cam: [rel path of history frame 0..nh-1]}`` -- the ``data_path`` strings of the metadata, forward-slashed."""
    return {c: [str(win[i]["cams"][c]["data_path"]).replace("\\", "/") for i in range(nh)] for c in cams}


def readable_rels(win: list, nh: int = 4, cams=CAMS) -> list:
    """The 12 (= nh x len(cams)) camera files a token needs.  AMENDMENT 1a / KB3: ALL of them, or the token is not readable."""
    d = history_rels(win, nh, cams)
    return [p for c in cams for p in d[c]]


# --------------------------------------------------------------------------------------------------------------------------------
# shard sources
# --------------------------------------------------------------------------------------------------------------------------------
def member_rel(name: str, strip_root: bool = True):
    """``<root>/<log>/<CAM>/<file>.jpg`` -> ``<log>/<CAM>/<file>.jpg`` (only the three cameras the bridge stitches); anything else -> None."""
    parts = [p for p in name.replace("\\", "/").split("/") if p and p != "."]
    if strip_root:
        parts = parts[1:]
    if len(parts) != 3 or parts[1] not in CAMS or not parts[2].lower().endswith(".jpg"):
        return None
    return "/".join(parts)


def iter_members(source: str):
    """Yield ``(rel, size)`` for every L0/F0/R0 jpg file of a shard source.  A ``.tgz`` is STREAMED (never extracted); a directory is walked."""
    if os.path.isdir(source):
        base = source.replace("\\", "/").rstrip("/")
        for dp, _dn, fns in os.walk(source):
            for fn in fns:
                full = os.path.join(dp, fn).replace("\\", "/")
                rel = member_rel(full[len(base) + 1:], strip_root=False)
                if rel:
                    yield rel, os.path.getsize(full)
        return
    with tarfile.open(source, "r|*") as tf:
        for m in tf:
            if not m.isfile():
                continue
            rel = member_rel(m.name, strip_root=True)
            if rel:
                yield rel, m.size


def discover_shards(root: str) -> list:
    """Sorted paths under ``root`` named ``navtrain_(current|history)_N`` (``.tgz`` files or extracted directories)."""
    out = []
    for fn in sorted(os.listdir(root)):
        m = SHARD_NAME.match(fn)
        if m:
            out.append(os.path.join(root, fn).replace("\\", "/"))
    return out
