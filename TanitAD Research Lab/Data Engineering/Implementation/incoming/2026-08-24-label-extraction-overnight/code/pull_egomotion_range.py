"""Pull the Alpamayo corpus's egomotion by RANGE-READING the remote zips.

⭐ WHY THIS REPLACES THE WHOLE-ZIP PULLER. Each `egomotion.chunk_NNNN.zip` is
~38 MB and holds ~200 clips, but the Alpamayo selection needs only ~3.4 of
them. Downloading whole chunks moved 53 GB to extract ~4 GB, and MEASURED at
2 clips/min it would have taken ~26 hours — past the morning deadline.

HF serves these files with `Accept-Ranges` (verified: a `bytes=-65536` request
returns **206** and the EOCD parses), so `zipfile` can work against a seekable
HTTP object and fetch ONLY the central directory plus the members we want.
Expected traffic: ~12x less.

⚠️ The redirect is resolved ONCE and the CDN URL reused. Re-resolving per range
request would triple the request count and can land on a different CDN node
mid-file.
"""
import truststore; truststore.inject_into_ssl()

import io
import json
import os
import re
import sys
import threading
import time
import urllib.request as U
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

ROOT = "C:/Users/Admin/tanitad-data/physicalai"
EXTRA = f"{ROOT}/labels/egomotion_alpamayo"
KEYS = "G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/Keys.txt"
BASE = ("https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles/"
        "resolve/main/labels/egomotion/")
WORKERS = 12
TAIL = 262144          # 256 KB: EOCD + central directory for a ~200-member zip

os.makedirs(EXTRA, exist_ok=True)
TOK = re.search(r'hf_[A-Za-z0-9]+',
                open(KEYS, encoding='utf-8', errors='ignore').read()).group(0)
AUTH = {"Authorization": f"Bearer {TOK}"}


class HTTPRangeFile(io.RawIOBase):
    """Minimal seekable file over an HTTP resource that honours Range.

    Only what `zipfile` actually calls: seek/tell/read/readable/seekable. The
    final TAIL bytes are cached because zipfile reads the central directory
    repeatedly while enumerating members.
    """

    def __init__(self, url: str, retries: int = 4):
        self.retries = retries
        self.url = self._resolve(url)
        self.size = self._length()
        self.pos = 0
        self._tail_off = max(0, self.size - TAIL)
        self._tail = self._fetch(self._tail_off, self.size - 1)

    def _open(self, headers, timeout=120):
        last = None
        for i in range(self.retries):
            try:
                return U.urlopen(U.Request(self.url if hasattr(self, "url") else headers.pop("_url"),
                                           headers=headers), timeout=timeout)
            except Exception as e:                     # noqa: BLE001
                last = e
                time.sleep(1.5 * (i + 1))
        raise last

    def _resolve(self, url: str) -> str:
        for i in range(self.retries):
            try:
                with U.urlopen(U.Request(url, headers={**AUTH, "Range": "bytes=0-0"}),
                               timeout=120) as r:
                    return r.url
            except Exception:                          # noqa: BLE001
                time.sleep(1.5 * (i + 1))
        return url

    def _length(self) -> int:
        with U.urlopen(U.Request(self.url, headers={**AUTH, "Range": "bytes=0-0"}),
                       timeout=120) as r:
            cr = r.headers.get("Content-Range")
            if cr and "/" in cr:
                return int(cr.rsplit("/", 1)[1])
            return int(r.headers.get("Content-Length", 0))

    def _fetch(self, a: int, b: int) -> bytes:
        last = None
        for i in range(self.retries):
            try:
                with U.urlopen(U.Request(self.url,
                                         headers={**AUTH, "Range": f"bytes={a}-{b}"}),
                               timeout=180) as r:
                    return r.read()
            except Exception as e:                     # noqa: BLE001
                last = e
                time.sleep(1.5 * (i + 1))
        raise last

    # -- file protocol --------------------------------------------------------
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = (off if whence == 0 else
                    self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size)
        if self.pos >= self._tail_off:                 # served from the cache
            s = self.pos - self._tail_off
            out = self._tail[s:s + (end - self.pos)]
        else:
            out = self._fetch(self.pos, end - 1)
        self.pos += len(out)
        return out


man = json.load(open("C:/Users/Admin/tanitad-data/alpamayo/selection_manifest.json"))
want = {e['clip_id'] for e in man}
ci = pd.read_parquet(f"{ROOT}/clip_index.parquet")
sub = ci[ci.index.isin(want)]
by_chunk: dict[int, set[str]] = {}
for cid, row in sub.iterrows():
    by_chunk.setdefault(int(row.chunk), set()).add(cid)

have = {p.split('.')[0] for p in os.listdir(EXTRA)}
todo = sorted(c for c, ids in by_chunk.items() if ids - have)
print(f"wanted {len(want)} | have {len(have)} | chunks left {len(todo)}", flush=True)

lock = threading.Lock()
done = [0]
got = [0]
bytes_ = [0]
t0 = time.time()


def one(chunk: int):
    ids = by_chunk[chunk] - have
    if not ids:
        return chunk, 0, ""
    url = f"{BASE}egomotion.chunk_{chunk:04d}.zip"
    n = 0
    nb = 0
    try:
        f = HTTPRangeFile(url)
        with zipfile.ZipFile(f) as z:
            for m in z.namelist():
                cid = m.split('.')[0]
                if cid in ids:
                    out = f"{EXTRA}/{cid}.parquet"
                    if os.path.exists(out):
                        continue
                    data = z.read(m)
                    # Parse BEFORE banking: a short range read still unzips.
                    pd.read_parquet(io.BytesIO(data))
                    tmp = out + ".part"
                    with open(tmp, 'wb') as fh:
                        fh.write(data)
                    os.replace(tmp, out)
                    n += 1
                    nb += len(data)
        return chunk, n, ""
    except Exception as e:                             # noqa: BLE001
        return chunk, n, f"{type(e).__name__}: {str(e)[:110]}"
    finally:
        with lock:
            bytes_[0] += nb


errs = []
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    futs = {ex.submit(one, c): c for c in todo}
    for fu in as_completed(futs):
        chunk, n, err = fu.result()
        with lock:
            done[0] += 1
            got[0] += n
            if err:
                errs.append((chunk, err))
            if done[0] % 40 == 0 or done[0] == len(todo):
                el = time.time() - t0
                rate = done[0] / max(el, 1e-9)
                eta = (len(todo) - done[0]) / max(rate, 1e-9) / 60
                print(f"  [{done[0]:4d}/{len(todo)}] clips={got[0]:5d} "
                      f"{bytes_[0]/1e6:.0f} MB  {rate*60:.0f} chunks/min  "
                      f"ETA {eta:.0f} min  errs={len(errs)}", flush=True)

n_final = len([p for p in os.listdir(EXTRA) if p.endswith('.parquet')])
print(f"DONE {n_final}/{len(want)} clips, {bytes_[0]/1e6:.0f} MB fetched, "
      f"{time.time()-t0:.0f}s, {len(errs)} chunk errors", flush=True)
for c, e in errs[:15]:
    print(f"  ERR {c}: {e}", flush=True)
json.dump({"clips": n_final, "wanted": len(want), "errors": len(errs)},
          open("C:/Users/Admin/tanitad-wt/_s2build/dl/pull_status.json", "w"))
