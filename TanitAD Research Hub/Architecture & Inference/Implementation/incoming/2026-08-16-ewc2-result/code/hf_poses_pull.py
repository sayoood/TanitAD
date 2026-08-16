"""Pull a POSES-ONLY val40 view from HF with HTTP RANGE reads — ~3 MB, not 4.70 GB.

WHY THIS SHAPE. `refc_dump_latents.py --backfill-endpoints` needs exactly two things
per episode: `poses` [T,4] and `episode_id`. A torch.save file is an UNCOMPRESSED zip,
so the poses tensor is a ~3 KB member inside a 117 MB archive. Reading the central
directory + that one member over HTTP Range costs ~80 KB/episode instead of 117 MB.

⛔ THE PROOF, and it is not "it looked right". Every episode's extracted poses bytes are
sha256'd and compared to the COMMITTED manifest
`…/2026-07-26-s3-decision-grade/artifacts/manifest_EVALPOD_val40.json`, which recorded
`sha256(poses.numpy().tobytes())` per episode off the canonical cache. A match is a
bit-identity proof that these are the canonical val40 poses; a mismatch refuses.

Writes `ep_%05d.pt` stubs in the shape `tanitad.data.mixing.load_episode` accepts, into
a directory whose name carries the parity key (the guard substring-matches the path).
`frames_u8` is a [T,9,1,1] zero placeholder — the backfill never reads frames, and the
window grid is a function of T alone. Same construction as the already-committed
`…/2026-08-04-distance-keeping-arms/code/build_local_val40_view.py`, which proved 40/40
against this same manifest on 2026-08-04.
"""
from __future__ import annotations

import hashlib
import io
import json
import pickle
import re
import sys
import time
import zipfile
from pathlib import Path

import httpx
import numpy as np
import torch
import truststore

truststore.inject_into_ssl()          # certifi fails behind this box's TLS proxy

REPO = "Sayood/tanitad-physicalai-w120-256x640cyl"
SUB = "epcache-256px-phase0/physicalai-val-0c5f7dac3b11"
KEYS = Path(r"G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/Keys.txt")
MANIFEST = Path(r"G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD") / (
    "TanitAD Research Hub/Architecture & Inference/Implementation/incoming/"
    "2026-07-26-s3-decision-grade/artifacts/manifest_EVALPOD_val40.json")

DT = {"FloatStorage": "<f4", "DoubleStorage": "<f8", "LongStorage": "<i8",
      "IntStorage": "<i4", "ByteStorage": "|u1", "BoolStorage": "|b1",
      "HalfStorage": "<f2", "ShortStorage": "<i2", "CharStorage": "|i1"}


# --------------------------------------------------------------------------- #
# a seekable read-only file over HTTP Range, with a chunk cache                 #
# --------------------------------------------------------------------------- #
class RangeFile(io.RawIOBase):
    CHUNK = 1 << 18                                    # 256 KiB

    def __init__(self, session, url, size, headers):
        self.s, self.url, self.size, self.h = session, url, size, headers
        self.pos, self.cache, self.bytes_fetched = 0, {}, 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1
                                            else self.size + off)
        return self.pos

    def tell(self):
        return self.pos

    def _chunk(self, i):
        if i not in self.cache:
            a = i * self.CHUNK
            b = min(a + self.CHUNK, self.size) - 1
            h = dict(self.h)
            h["Range"] = f"bytes={a}-{b}"
            r = self.s.get(self.url, headers=h, timeout=120,
                           follow_redirects=True)
            if r.status_code not in (206, 200):
                raise RuntimeError(f"range {a}-{b} -> HTTP {r.status_code}")
            if r.status_code == 200 and len(r.content) == self.size:
                raise RuntimeError("server IGNORED the Range header and sent the "
                                   "whole object — refusing (that is a 117 MB "
                                   "read masquerading as a 256 KB one)")
            self.cache[i] = r.content
            self.bytes_fetched += len(r.content)
        return self.cache[i]

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        out, got = bytearray(), 0
        while got < n:
            i, o = divmod(self.pos + got, self.CHUNK)
            c = self._chunk(i)
            take = min(n - got, len(c) - o)
            out += c[o:o + take]
            got += take
        b[:n] = out
        self.pos += n
        return n


# --------------------------------------------------------------------------- #
# torch-zip inspection WITHOUT materialising any tensor                         #
# --------------------------------------------------------------------------- #
class _Stub:
    def __init__(self, module, name):
        self.module, self.name = module, name

    def __call__(self, *a, **k):
        return {"__fn__": f"{self.module}.{self.name}", "args": a, "kwargs": k}


class _Inspect(pickle.Unpickler):
    def find_class(self, module, name):
        return _Stub(module, name)

    def persistent_load(self, pid):
        return {"__storage__": True, "key": str(pid[2]),
                "dtype": getattr(pid[1], "name", str(pid[1])),
                "numel": int(pid[4])}


def tensor_desc(v):
    """(storage key, dtype str, offset, shape) for a `_rebuild_tensor_v2` node."""
    if not (isinstance(v, dict) and "_rebuild_tensor" in str(v.get("__fn__", ""))):
        return None
    st, off, shape = v["args"][0], int(v["args"][1]), tuple(v["args"][2])
    return st["key"], st["dtype"], off, shape


def pull_episode(session, name, headers, want=("poses", "actions", "maneuvers")):
    url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{SUB}/{name}"
    r = session.head(url, headers=headers, follow_redirects=True, timeout=120)
    r.raise_for_status()
    size = int(r.headers["Content-Length"])
    final = str(r.url)
    f = RangeFile(session, final, size, headers)
    z = zipfile.ZipFile(f)
    names = z.namelist()
    pkl = [n for n in names if n.endswith("/data.pkl")]
    if not pkl:
        raise RuntimeError(f"{name}: no data.pkl — not a torch zip archive")
    root = pkl[0].rsplit("/", 1)[0]
    obj = _Inspect(io.BytesIO(z.read(pkl[0]))).load()
    if not isinstance(obj, dict):
        raise RuntimeError(f"{name}: top-level object is {type(obj)}, expected dict")
    out = {"episode_id": int(obj["episode_id"]), "_zip_size": size}
    for k in want:
        if k not in obj:
            continue
        d = tensor_desc(obj[k])
        if d is None:
            raise RuntimeError(f"{name}: {k} is not a rebuilt tensor node")
        key, dt, off, shape = d
        raw = z.read(f"{root}/data/{key}")
        dtype = DT[dt]
        n = int(np.prod(shape)) if shape else 0
        itemsize = np.dtype(dtype).itemsize
        arr = np.frombuffer(raw, dtype=dtype, count=n,
                            offset=off * itemsize).reshape(shape)
        out[k] = arr
    out["_bytes_fetched"] = f.bytes_fetched
    return out


def main(outdir: str, n: int = 40) -> int:
    tok = re.findall(r"hf_[A-Za-z0-9]+", KEYS.read_text(errors="ignore"))
    headers = {"Authorization": f"Bearer {tok[0]}"} if tok else {}
    man = {e["file"]: e for e in json.loads(MANIFEST.read_text())["episodes"]}
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    s = httpx.Client(timeout=120, follow_redirects=True)

    rows, tot, t0 = [], 0, time.time()
    for i in range(n):
        name = f"ep_{i:05d}.pt"
        d = pull_episode(s, name, headers)
        poses = np.ascontiguousarray(d["poses"], dtype=np.float32)
        sha = hashlib.sha256(poses.tobytes()).hexdigest()
        exp = man[name]
        ok = (sha == exp["poses_sha256"] and int(exp["episode_id"]) == d["episode_id"]
              and int(exp["T"]) == poses.shape[0])
        T = int(poses.shape[0])
        pay = {"frames_u8": torch.zeros((T, 9, 1, 1), dtype=torch.uint8),
               "actions": torch.from_numpy(np.ascontiguousarray(
                   d["actions"], dtype=np.float32)),
               "poses": torch.from_numpy(poses),
               "episode_id": int(d["episode_id"])}
        if "maneuvers" in d:
            pay["maneuvers"] = torch.from_numpy(
                np.ascontiguousarray(d["maneuvers"], dtype=np.int64))
        torch.save(pay, out / name)
        tot += d["_bytes_fetched"]
        rows.append({"file": name, "episode_id": int(d["episode_id"]), "T": T,
                     "poses_sha256": sha,
                     "manifest_poses_sha256": exp["poses_sha256"],
                     "sha_ok": sha == exp["poses_sha256"],
                     "eid_ok": int(exp["episode_id"]) == d["episode_id"],
                     "T_ok": int(exp["T"]) == T, "ok": bool(ok),
                     "bytes_fetched": d["_bytes_fetched"],
                     "remote_bytes": d["_zip_size"]})
        print(f"  {name} eid={d['episode_id']} T={T} sha_ok={rows[-1]['sha_ok']} "
              f"fetched={d['_bytes_fetched']}B", flush=True)

    starts = lambda t: np.arange(0, max(int(t) - 8 - 20, 0), 8)
    rep = {
        "_what": "poses-only val40 view pulled from HF by HTTP RANGE (frames never read)",
        "_source": f"hf:datasets/{REPO}/{SUB}",
        "_manifest": str(MANIFEST),
        "n_episodes": len(rows),
        "n_match_manifest": sum(r["ok"] for r in rows),
        "n_mismatch": sum(not r["ok"] for r in rows),
        "bytes_fetched_total": tot,
        "bytes_if_whole_files": sum(r["remote_bytes"] for r in rows),
        "saving_x": round(sum(r["remote_bytes"] for r in rows) / max(tot, 1), 1),
        "total_windows_from_T": int(sum(starts(r["T"]).size for r in rows)),
        "seconds": round(time.time() - t0, 1),
        "out_dir": str(out), "rows": rows}
    (out.parent / "val40_hf_poses_verify.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "rows"}, indent=1))
    return 0 if rep["n_mismatch"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 40))
