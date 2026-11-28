"""scene_bank.py -- the scene-token store L7-S trains on (DRAFT for PILOT-S; no scorer has been trained on it).

WHY THIS FILE EXISTS (measured 2026-10-22 on D:/tanitad-archive/l7cap_thor_1022/train139_s{0,1}/dump/l7cap_scene.npz)
    * the capture wrote ``l7cap_scene.npz`` with ``np.savez_compressed``: every member is DEFLATE-compressed, so there is no random
      access and ``np.load(...)["r7_agent_tokens"]`` inflates the whole member: agent ``[4855, 300, 384]`` fp16 = 1.12 GB and bev
      ``[4855, 480, 96]`` fp16 = 0.45 GB PER BANK.  The PILOT's two banks held in RAM = 3.1 GB, on top of the ~1.4 GB the E / F runs
      already peak at: over the 4 GB per-process budget.
    * so the members are unpacked ONCE (streaming, 32 MiB at a time, CRC32 re-derived against the zip directory) into a content-addressed
      cache of raw ``.npy`` files, and each batch is gathered with 2 seek + readinto calls per window straight into a preallocated
      buffer.  Nothing is memory-mapped and nothing is held: the training process's working set is one batch of tokens
      (32 windows x (230,400 + 92,160) B fp16 = 10.3 MB, 20.6 MB after the fp32 cast) plus the OS file cache, which is not charged to it.
    * the cache is content-addressed (``<member>.<crc32>.<bytes>.npy``).  The two PILOT banks' scene members are BIT-IDENTICAL (equal zip
      CRC32 and size for r7_agent_tokens, r7_bev_tokens and r7_agent_pad: the scene tokens do not depend on the inference seed), so both
      banks resolve to ONE 1.57 GB file set.  Each bank is still opened, verified and aligned on its own.

ALIGNMENT IS FAIL-CLOSED (``align_rows``): the scene bank's (clip_index, ws) keys must be a bijection onto the fan bank's keys (same
size, no duplicates, no missing, no extras), the fan's ``t`` must be the same window set as ``ws`` (ws - t is one constant), and after the
join every gathered row's (clip_index, ws) is compared back against the fan row it was gathered for.  A bank whose rows are in a different
ORDER is re-joined by key (and reported); anything else is a SystemExit.  Scene tokens carry no identity of their own, so a bank whose
LABELS are wrong cannot be detected here; what can be pinned, and is, is that the join is by key and never by position.

Admissibility (DESIGN sec. 4) is unchanged and enforced where it always was: ``refc_scorer.assemble_inputs`` is the only path from these
arrays to the scorer, and it accepts ``bev_tokens / bev_pad / agent_tokens / agent_pad`` only.  The bank's ``r7_*`` names are renamed on the
way in (``rename`` below); the label bank (``l7cap_labels.npz``: agent futures, maps) is never opened by this module.
"""
from __future__ import annotations

import os
import zipfile
import zlib
from typing import Mapping

import numpy as np
import torch
from numpy.lib import format as npf

#: scorer key -> member of ``l7cap_scene.npz``
SCENE_MEMBERS: Mapping[str, str] = {"bev_tokens": "r7_bev_tokens", "agent_tokens": "r7_agent_tokens"}
PAD_MEMBERS: Mapping[str, str] = {"bev_pad": "r7_bev_pad", "agent_pad": "r7_agent_pad"}
CHUNK = 32 * 2 ** 20


# ---------------------------------------------------------------------------------------------------------------------
# unpack: deflate member -> raw .npy (streaming, CRC re-derived)
# ---------------------------------------------------------------------------------------------------------------------
def npy_header(f) -> tuple[tuple, np.dtype, int]:
    """(shape, dtype, data offset) of an open .npy stream; refuses Fortran order (rows would not be contiguous)."""
    ver = npf.read_magic(f)
    shape, fortran, dtype = (npf.read_array_header_1_0 if ver == (1, 0) else npf.read_array_header_2_0)(f)
    if fortran:
        raise SystemExit("[scene_bank] a Fortran-ordered scene member cannot be gathered row-wise")
    return tuple(shape), np.dtype(dtype), f.tell()


def cache_name(member: str, crc: int, size: int) -> str:
    return f"{member}.{crc:08x}.{size}.npy"


def crc32_file(path: str) -> int:
    crc = 0
    with open(path, "rb", buffering=0) as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                return crc & 0xFFFFFFFF
            crc = zlib.crc32(b, crc)


def unpack_member(npz_path: str, member: str, cache_dir: str, verify: bool = True) -> str:
    """Stream ``member`` (e.g. ``r7_agent_tokens``) out of ``npz_path`` into ``cache_dir`` and return the cache file's path.

    The cache file is byte-identical to the zip member (header + data), named by the zip directory's CRC32 and size.  The stream's CRC32 is
    recomputed while writing and compared with the directory's; an existing cache file of the right size is re-CRC'd when ``verify``.
    Writes go to ``<name>.part`` and are renamed only after the CRC matched, so a killed unpack never leaves a file that looks complete.
    """
    os.makedirs(cache_dir, exist_ok=True)
    with zipfile.ZipFile(npz_path) as z:
        info = z.getinfo(member + ".npy")
        dst = os.path.join(cache_dir, cache_name(member, info.CRC, info.file_size))
        if os.path.exists(dst):
            if os.path.getsize(dst) == info.file_size and (not verify or crc32_file(dst) == info.CRC):
                return dst
            os.remove(dst)                                   # wrong size or wrong CRC: it is debris, not a cache hit
        tmp = dst + ".part"
        crc = 0
        with z.open(info) as src, open(tmp, "wb") as out:
            while True:
                b = src.read(CHUNK)
                if not b:
                    break
                crc = zlib.crc32(b, crc)
                out.write(b)
        if (crc & 0xFFFFFFFF) != info.CRC or os.path.getsize(tmp) != info.file_size:
            os.remove(tmp)
            raise SystemExit(f"[scene_bank] CRC32 / size mismatch unpacking {member} from {npz_path}: refusing to cache it")
        os.replace(tmp, dst)
    return dst


# ---------------------------------------------------------------------------------------------------------------------
# the alignment join (fail closed)
# ---------------------------------------------------------------------------------------------------------------------
def align_rows(scene_ci, scene_ws, fan_ci, fan_ws, fan_t=None, what: str = "scene") -> tuple[np.ndarray, bool]:
    """Scene row for every FAN row.  ``fan_*`` are the fan bank's RAW arrays (all windows of the bank, in bank order).

    Raises SystemExit unless (clip_index, ws) is a bijection between the two banks.  Returns (rows, identical_order)."""
    sc = np.asarray(scene_ci).astype(np.int64)
    sw = np.asarray(scene_ws).astype(np.int64)
    fc = np.asarray(fan_ci).astype(np.int64)
    fw = np.asarray(fan_ws).astype(np.int64)
    if not (len(sc) == len(sw) and len(fc) == len(fw)):
        raise SystemExit(f"[{what}] clip_index / ws lengths disagree inside one bank")
    if len(sc) != len(fc):
        raise SystemExit(f"[{what}] {len(sc)} scene windows vs {len(fc)} fan windows: not the same capture")
    if fan_t is not None:
        off = np.unique(fw - np.asarray(fan_t).astype(np.int64))
        if len(off) != 1:
            raise SystemExit(f"[{what}] the fan's ws - t is not one constant ({off[:5].tolist()}): (clip_index, t) and (clip_index, ws) "
                             f"do not name the same windows")
    skey = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(sc, sw))}
    fkey = {(int(a), int(b)) for a, b in zip(fc, fw)}
    if len(skey) != len(sc):
        raise SystemExit(f"[{what}] duplicate (clip_index, ws) keys in the scene bank")
    if len(fkey) != len(fc):
        raise SystemExit(f"[{what}] duplicate (clip_index, ws) keys in the fan bank")
    missing = [k for k in fkey if k not in skey]
    if missing:
        raise SystemExit(f"[{what}] {len(missing)} fan windows have no scene row (e.g. {missing[0]})")
    extra = [k for k in skey if k not in fkey]
    if extra:
        raise SystemExit(f"[{what}] {len(extra)} scene windows are not in the fan bank (e.g. {extra[0]})")
    rows = np.asarray([skey[(int(a), int(b))] for a, b in zip(fc, fw)], np.int64)
    if not (np.array_equal(sc[rows], fc) and np.array_equal(sw[rows], fw)):              # re-read the join; never trust the dict
        raise SystemExit(f"[{what}] the key join does not reproduce the fan's (clip_index, ws)")
    return rows, bool(np.array_equal(rows, np.arange(len(rows))))


# ---------------------------------------------------------------------------------------------------------------------
# stores
# ---------------------------------------------------------------------------------------------------------------------
class SceneStore:
    """One bank's scene tokens on disk (raw ``.npy`` cache) with per-row ``gather``."""

    def __init__(self, npz_path: str, cache_dir: str, verify: bool = True):
        self.npz_path = npz_path
        with np.load(npz_path) as z:
            files = set(z.files)
            if not {"clip_index", "ws"} <= files:
                raise SystemExit(f"[scene_bank] {npz_path} has no clip_index / ws: cannot align it")
            self.clip_index = np.asarray(z["clip_index"])
            self.ws = np.asarray(z["ws"])
            self.pads = {k: np.asarray(z[m]).astype(bool) for k, m in PAD_MEMBERS.items() if m in files}
            absent = [m for m in SCENE_MEMBERS.values() if m not in files]
        if absent:
            raise SystemExit(f"[scene_bank] {npz_path} lacks {absent}: L7-S needs BOTH scene sources")
        self.cache_files: dict[str, str] = {}
        self._f: dict[str, object] = {}
        self._meta: dict[str, tuple] = {}
        n = len(self.clip_index)
        for key, member in SCENE_MEMBERS.items():
            path = unpack_member(npz_path, member, cache_dir, verify=verify)
            self.cache_files[key] = path
            f = open(path, "rb", buffering=0)
            shape, dtype, off = npy_header(f)
            if shape[0] != n:
                raise SystemExit(f"[scene_bank] {member}: {shape[0]} rows vs {n} (clip_index, ws) keys")
            self._f[key] = f
            self._meta[key] = (shape[1:], dtype, off, int(np.prod(shape[1:])) * dtype.itemsize)
        for k, p in self.pads.items():
            if p.shape[0] != n:
                raise SystemExit(f"[scene_bank] {k}: {p.shape[0]} rows vs {n} keys")

    @property
    def dims(self) -> dict:
        """{"bev": 96, "agent": 384} -- the widths ``L7Config.source_dims`` / ``SceneMemory`` are built with."""
        return {k[:-len("_tokens")]: int(self._meta[k][0][-1]) for k in SCENE_MEMBERS}

    def gather(self, rows) -> dict:
        rows = np.asarray(rows, np.int64)
        out = {}
        for key, (inner, dtype, off, rb) in self._meta.items():
            buf = np.empty((len(rows),) + tuple(inner), dtype)
            f = self._f[key]
            for i, r in enumerate(rows):
                f.seek(off + int(r) * rb)
                if f.readinto(buf[i]) != rb:
                    raise SystemExit(f"[scene_bank] short read of {key} row {int(r)} from {self.cache_files[key]}")
            out[key] = buf
        for k, p in self.pads.items():
            out[k] = p[rows]
        return out

    def close(self):
        for f in self._f.values():
            f.close()


class MemStore:
    """In-memory twin of SceneStore (the ``--synthetic`` smoke and the tests)."""

    def __init__(self, clip_index, ws, tokens: Mapping[str, np.ndarray], pads: Mapping[str, np.ndarray] | None = None):
        self.clip_index, self.ws = np.asarray(clip_index), np.asarray(ws)
        self.tokens = {k: np.asarray(v) for k, v in tokens.items()}
        self.pads = {k: np.asarray(v).astype(bool) for k, v in (pads or {}).items()}
        self.cache_files = {}
        self.npz_path = "<memory>"

    @property
    def dims(self) -> dict:
        return {k[:-len("_tokens")]: int(v.shape[-1]) for k, v in self.tokens.items()}

    def gather(self, rows) -> dict:
        rows = np.asarray(rows, np.int64)
        out = {k: v[rows] for k, v in self.tokens.items()}
        out.update({k: p[rows] for k, p in self.pads.items()})
        return out

    def close(self):
        pass


class SceneBank:
    """The scene side of a (possibly multi-bank) training set: one store per bank + each fan row's (bank, scene row)."""

    def __init__(self, stores: list, bank_of_row: np.ndarray, scene_row: np.ndarray, identical_order: list):
        dims = [s.dims for s in stores]
        if any(d != dims[0] for d in dims):
            raise SystemExit(f"[scene_bank] banks disagree on scene widths: {dims}")
        self.stores, self.bank_of_row, self.scene_row = stores, np.asarray(bank_of_row), np.asarray(scene_row)
        self.dims = dims[0]
        self.identical_order = identical_order

    def batch(self, ii) -> dict:
        """Scorer-side tensors for fan rows ``ii``: fp32 tokens (the fp16 on disk is exact in fp32), bool pads; order of ``ii``."""
        ii = np.asarray(ii)
        bk = self.bank_of_row[ii]
        parts: dict[str, np.ndarray] = {}
        for b in np.unique(bk):
            sel = np.where(bk == b)[0]
            g = self.stores[int(b)].gather(self.scene_row[ii[sel]])
            for k, v in g.items():
                if k not in parts:
                    parts[k] = np.empty((len(ii),) + v.shape[1:], v.dtype)
                parts[k][sel] = v
        out = {k: (torch.from_numpy(v).to(torch.float32) if k.endswith("_tokens") else torch.from_numpy(v))
               for k, v in parts.items()}
        return out

    def close(self):
        for st in self.stores:
            st.close()

    def describe(self) -> dict:
        return {"dims": self.dims, "identical_order": self.identical_order,
                "stores": [{"npz": s.npz_path.replace("\\", "/"),
                            "cache": {k: os.path.basename(v) for k, v in s.cache_files.items()}} for s in self.stores]}


def attach(fanz: dict, fan_npzs: list[str], cache_dir: str, verify: bool = True) -> SceneBank:
    """Build the SceneBank for ``train_l7.load_bank``'s concatenated ``fanz`` (which carries ``bank_of_row``, ``clip_index``, ``ws``).

    Each bank's scene file is the ``l7cap_scene.npz`` in the SAME dump directory as its ``fan_extras.npz`` -- so the scene bank has the
    fan bank's provenance (the train-vs-eval refusal in ``train_l7._bank_provenance`` already read that directory) and no second path can
    point at a different capture."""
    stores, rows_all = [], np.zeros(len(fanz["ws"]), np.int64)
    ident = []
    for i, fan_npz in enumerate(fan_npzs):
        scene_npz = os.path.join(os.path.dirname(os.path.abspath(fan_npz)), "l7cap_scene.npz")
        if not os.path.exists(scene_npz):
            raise SystemExit(f"[scene_bank] no l7cap_scene.npz next to {fan_npz}")
        st = SceneStore(scene_npz, cache_dir, verify=verify)
        with np.load(fan_npz) as z:
            _, same = align_rows(st.clip_index, st.ws, z["clip_index"], z["ws"], z["t"] if "t" in z.files else None, what=f"bank {i}")
        sel = np.where(fanz["bank_of_row"] == i)[0]
        rows, _ = align_rows_subset(st, fanz["clip_index"][sel], fanz["ws"][sel], what=f"bank {i}")
        rows_all[sel] = rows
        stores.append(st)
        ident.append(same)
    return SceneBank(stores, fanz["bank_of_row"], rows_all, ident)


def align_rows_subset(store, ci, ws, what: str = "scene") -> tuple[np.ndarray, bool]:
    """Scene row for each of the (possibly fewer) windows the TARGETS kept; every one must exist, and the re-read must match."""
    key = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(store.clip_index, store.ws))}
    try:
        rows = np.asarray([key[(int(a), int(b))] for a, b in zip(ci, ws)], np.int64)
    except KeyError as e:
        raise SystemExit(f"[{what}] a training window {e} has no scene row")
    if not (np.array_equal(np.asarray(store.clip_index)[rows].astype(np.int64), np.asarray(ci).astype(np.int64))
            and np.array_equal(np.asarray(store.ws)[rows].astype(np.int64), np.asarray(ws).astype(np.int64))):
        raise SystemExit(f"[{what}] the scene join does not reproduce the training windows' (clip_index, ws)")
    return rows, True


def synthetic_scene(fanz: dict, dims: Mapping[str, int] | None = None, n_tok: Mapping[str, int] | None = None, seed: int = 0) -> SceneBank:
    """A scene bank for ``train_l7.synthetic_bank``: every token carries its window's own (clip_index, ws) in its first two channels, then
    noise, so a join error is readable as a literal and a scorer has something to attend to."""
    dims = dict(dims or {"bev": 8, "agent": 12})
    n_tok = dict(n_tok or {"bev": 6, "agent": 5})
    rng = np.random.default_rng(seed)
    n = len(fanz["ws"])
    toks = {}
    for name in ("bev", "agent"):
        a = rng.normal(size=(n, n_tok[name], dims[name])).astype(np.float16)
        a[:, :, 0] = np.asarray(fanz["clip_index"])[:, None]
        a[:, :, 1] = np.asarray(fanz["ws"])[:, None]
        toks[name + "_tokens"] = a
    st = MemStore(fanz["clip_index"], fanz["ws"], toks, {"agent_pad": np.zeros((n, n_tok["agent"]), bool)})
    return SceneBank([st], np.zeros(n, np.int64), np.arange(n), [True])


def _cli():
    import argparse
    import json
    ap = argparse.ArgumentParser(description="unpack the scene members of a TRAIN capture into the raw-npy cache (never an eval bank)")
    ap.add_argument("--npz", nargs="+", required=True, help="l7cap_scene.npz path(s)")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()
    rep = {}
    for p in a.npz:
        if "eval" in p.replace("\\", "/").lower():
            raise SystemExit(f"[scene_bank] REFUSING to unpack {p}: eval captures are not read by the L7-S pilot")
        rep[p] = {m: os.path.basename(unpack_member(p, m, a.cache, verify=not a.no_verify)) for m in SCENE_MEMBERS.values()}
    print("ZZSCENEUNPACK " + json.dumps(rep))


if __name__ == "__main__":
    _cli()
