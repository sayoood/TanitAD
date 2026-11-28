"""Range-pull obstacle.offline parquets for the B1 TRAIN split, counting every
byte that actually crosses the wire.

WHY THE COUNTER: the published 2.3 GB figure is `505 KB x 4,572`, extrapolated
from ONE datum (eval obstacle parquets on disk).  That anchor measures EXTRACTED
PARQUET BYTES ON DISK, not HTTP range-request wire volume -- the per-chunk zip
tail (central directory) and the per-member local file header are unpriced, and
they are paid 1,387 and 4,572 times respectively.  So the cost is measured here
with a counter wrapped around the ONLY place bytes enter the process, and
reported as MEASURED with its method, not ESTIMATED.

Modes:
  --sample-chunks N   pull a random sample of N chunks (measurement); the pulled
                      parquets are banked, so the sample is not wasted work.
  (default)           pull every remaining TRAIN clip.

Every print is ASCII (this box is cp1252).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import re
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:  # pragma: no cover
    pass

HF_DATASET = "nvidia/PhysicalAI-Autonomous-Vehicles"
HF_OBS_BASE = (f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main/"
               "labels/obstacle.offline/")
OBS_NEED_COLS = ("timestamp_us", "track_id", "center_x", "center_y", "size_x",
                 "label_class")

# ---- the wire counter ------------------------------------------------------
_WIRE = {"bytes": 0, "requests": 0}
_WIRE_LOCK = threading.Lock()
#: per-chunk breakdown: chunk -> {"fixed": bytes before any member read,
#:                                "members": bytes attributable to members}
_PER_CHUNK: dict[int, dict] = {}


def _wire_add(n: int, chunk: int, phase: str) -> None:
    with _WIRE_LOCK:
        _WIRE["bytes"] += n
        _WIRE["requests"] += 1
        d = _PER_CHUNK.setdefault(chunk, {"fixed": 0, "members": 0,
                                          "req_fixed": 0, "req_members": 0})
        d[phase] += n
        d["req_" + phase] += 1


class _HTTPRangeFile(io.RawIOBase):
    """Seekable file over an HTTP resource honouring Range.

    Identical to ``build_lead_block_b1._HTTPRangeFile`` except every fetch is
    counted.  ``phase`` is flipped to 'members' once the zip's central directory
    has been parsed, so the FIXED per-chunk cost and the VARIABLE per-clip cost
    are separated instead of being blended into one ratio.
    """

    TAIL = 262144

    def __init__(self, url: str, auth: dict, chunk: int, retries: int = 4):
        import urllib.request as U

        self._U, self.auth, self.retries = U, auth, retries
        self.chunk = chunk
        self.phase = "fixed"
        self.url = self._resolve(url)
        self.size = self._length()
        self.pos = 0
        self._tail_off = max(0, self.size - self.TAIL)
        self._tail = self._fetch(self._tail_off, self.size - 1)

    def _resolve(self, url):
        U = self._U
        for i in range(self.retries):
            try:
                with U.urlopen(U.Request(url, headers={**self.auth,
                                                       "Range": "bytes=0-0"}),
                               timeout=120) as r:
                    _wire_add(1, self.chunk, self.phase)
                    return r.url
            except Exception:  # noqa: BLE001
                time.sleep(1.5 * (i + 1))
        return url

    def _length(self):
        U = self._U
        with U.urlopen(U.Request(self.url, headers={**self.auth,
                                                    "Range": "bytes=0-0"}),
                       timeout=120) as r:
            _wire_add(1, self.chunk, self.phase)
            cr = r.headers.get("Content-Range")
            return (int(cr.rsplit("/", 1)[1]) if cr and "/" in cr
                    else int(r.headers.get("Content-Length", 0)))

    def _fetch(self, a, b):
        U = self._U
        last = None
        for i in range(self.retries):
            try:
                with U.urlopen(U.Request(self.url,
                                         headers={**self.auth,
                                                  "Range": f"bytes={a}-{b}"}),
                               timeout=180) as r:
                    data = r.read()
                    _wire_add(len(data), self.chunk, self.phase)
                    return data
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(1.5 * (i + 1))
        raise last

    def readable(self):  # noqa: D102
        return True

    def seekable(self):  # noqa: D102
        return True

    def tell(self):  # noqa: D102
        return self.pos

    def seek(self, off, whence=0):  # noqa: D102
        self.pos = (off if whence == 0 else self.pos + off if whence == 1
                    else self.size + off)
        return self.pos

    def read(self, n=-1):  # noqa: D102
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size)
        if self.pos >= self._tail_off:
            s = self.pos - self._tail_off
            out = self._tail[s:s + (end - self.pos)]  # served from the tail: 0 wire
        else:
            out = self._fetch(self.pos, end - 1)
        self.pos += len(out)
        return out


def _hf_token(keys_path: str) -> str:
    for _ in range(6):
        try:
            with open(keys_path, encoding="utf-8", errors="ignore") as fh:
                m = re.search(r"hf_[A-Za-z0-9]+", fh.read())
            if not m:
                raise SystemExit("no hf_ token in %s" % keys_path)
            return m.group(0)
        except OSError:
            time.sleep(3)
    raise SystemExit("could not read %s" % keys_path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--chunk-map", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--keys", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sample-chunks", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260906)
    ap.add_argument("--report", required=True)
    a = ap.parse_args(argv)

    clips = json.load(open(a.clips, encoding="utf-8"))
    cm = pd.read_parquet(a.chunk_map)
    chunk_of = dict(zip(cm["clip_id"], cm["chunk"]))
    os.makedirs(a.out_dir, exist_ok=True)
    auth = {"Authorization": "Bearer " + _hf_token(a.keys)}

    by_chunk: dict[int, list[str]] = {}
    have = no_chunk = 0
    for c in clips:
        if os.path.exists(os.path.join(a.out_dir, c + ".parquet")):
            have += 1
            continue
        if c not in chunk_of:
            no_chunk += 1
            continue
        by_chunk.setdefault(int(chunk_of[c]), []).append(c)

    all_chunks = sorted(by_chunk)
    if a.sample_chunks and a.sample_chunks < len(all_chunks):
        rnd = random.Random(a.seed)
        pick = sorted(rnd.sample(all_chunks, a.sample_chunks))
    else:
        pick = all_chunks
    n_clips_todo = sum(len(by_chunk[c]) for c in pick)
    print("[pull] clips=%d already_have=%d no_chunk=%d | chunks_todo=%d "
          "(of %d) clips_todo=%d workers=%d"
          % (len(clips), have, no_chunk, len(pick), len(all_chunks),
             n_clips_todo, a.workers), flush=True)

    lock = threading.Lock()
    results, errors = {}, []
    t0 = time.time()

    def one(chunk):
        name = "obstacle.offline.chunk_%04d.zip" % chunk
        got = []
        try:
            f = _HTTPRangeFile(HF_OBS_BASE + name, auth, chunk)
            with zipfile.ZipFile(f) as z:
                names = z.namelist()
                f.phase = "members"          # the CD is parsed: fixed cost is done
                for cid in by_chunk[chunk]:
                    member = next((m for m in names
                                   if m.rsplit("/", 1)[-1].startswith(cid)), None)
                    if member is None:
                        with lock:
                            errors.append({"chunk": chunk, "clip": cid,
                                           "error": "member not in chunk"})
                        continue
                    data = z.read(member)
                    df = pd.read_parquet(io.BytesIO(data))
                    miss = [c for c in OBS_NEED_COLS if c not in df.columns]
                    if miss:
                        with lock:
                            errors.append({"chunk": chunk, "clip": cid,
                                           "error": "schema missing %s" % miss})
                        continue
                    p = os.path.join(a.out_dir, cid + ".parquet")
                    with open(p + ".part", "wb") as fh:
                        fh.write(data)
                    os.replace(p + ".part", p)
                    got.append({"clip_id": cid, "chunk": chunk,
                                "member_bytes_extracted": len(data),
                                "sha256": hashlib.sha256(data).hexdigest(),
                                "n_rows": int(len(df))})
        except Exception as e:  # noqa: BLE001
            with lock:
                errors.append({"chunk": chunk, "clip": None,
                               "error": "%s: %s" % (type(e).__name__, str(e)[:160])})
        return got

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(one, c) for c in pick]
        for i, fu in enumerate(as_completed(futs), 1):
            for r in fu.result():
                results[r["clip_id"]] = r
            if i % 25 == 0 or i == len(futs):
                el = time.time() - t0
                print("  [%d/%d chunks] clips=%d errs=%d wire=%.1f MB "
                      "reqs=%d %.0fs"
                      % (i, len(futs), len(results), len(errors),
                         _WIRE["bytes"] / 1e6, _WIRE["requests"], el), flush=True)

    # ---- the decomposition -------------------------------------------------
    fixed = sum(d["fixed"] for d in _PER_CHUNK.values())
    memb = sum(d["members"] for d in _PER_CHUNK.values())
    n_ch = len(_PER_CHUNK)
    n_cl = len(results)
    extracted = sum(r["member_bytes_extracted"] for r in results.values())
    rep = {
        "mode": "sample" if a.sample_chunks else "full",
        "seed": a.seed,
        "n_chunks_touched": n_ch,
        "n_clips_pulled": n_cl,
        "n_errors": len(errors),
        "errors": errors[:50],
        "seconds": round(time.time() - t0, 1),
        "wire_bytes_total": _WIRE["bytes"],
        "wire_requests_total": _WIRE["requests"],
        "wire_bytes_fixed_per_chunk_phase": fixed,
        "wire_bytes_member_phase": memb,
        "extracted_member_bytes": extracted,
        "mean_fixed_bytes_per_chunk": round(fixed / n_ch, 1) if n_ch else None,
        "mean_member_wire_bytes_per_clip": round(memb / n_cl, 1) if n_cl else None,
        "mean_extracted_bytes_per_clip": round(extracted / n_cl, 1) if n_cl else None,
        "mean_wire_bytes_per_clip_blended": round(_WIRE["bytes"] / n_cl, 1) if n_cl else None,
        "clips_per_chunk_in_sample": round(n_cl / n_ch, 4) if n_ch else None,
        "clips": results,
    }
    with open(a.report, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)
    print("PULL_DONE chunks=%d clips=%d errs=%d wire=%.3f MB extracted=%.3f MB "
          "%.0fs" % (n_ch, n_cl, len(errors), _WIRE["bytes"] / 1e6,
                     extracted / 1e6, time.time() - t0), flush=True)
    print("  fixed/chunk = %s B   member-wire/clip = %s B   extracted/clip = %s B"
          % (rep["mean_fixed_bytes_per_chunk"],
             rep["mean_member_wire_bytes_per_clip"],
             rep["mean_extracted_bytes_per_clip"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
