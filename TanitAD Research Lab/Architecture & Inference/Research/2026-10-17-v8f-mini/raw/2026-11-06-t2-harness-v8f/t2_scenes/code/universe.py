"""Universe steps 1-3 of SCREEN_RULE.md. Reads only usdz zip central directories (member names/sizes/CRCs/offsets), never member content."""
import sys, io, json, zipfile, time, hashlib, os
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "E:/Projects/TanitAD-artifacts/dagland/motion_c/t2_scenes/code")
import hfc
T = "E:/Projects/TanitAD-artifacts/dagland/motion_c/t2_scenes/"
REV = "5fe7a149b40ee83a2d3e09f7fe5dc2f217cabe11"
NEED = ("volume.nurec", "rig_trajectories.json", "sequence_tracks.json")
TAIL = 262144

class TailFile(io.RawIOBase):
    def __init__(self, size, tail):
        self.size, self.tail, self.base, self.pos = size, tail, size - len(tail), 0
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos
    def read(self, n=-1):
        if n is None or n < 0: n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        if n and self.pos < self.base: raise IOError(f"read at {self.pos} before tail base {self.base}")
        b = self.tail[self.pos - self.base: self.pos - self.base + n]
        self.pos += len(b); return b
    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)

def directory(sc, size):
    url = f"https://huggingface.co/datasets/{hfc.REPO}/resolve/{REV}/sample_set/26.04_release/{sc}/{sc}.usdz"
    r = hfc.get(url, headers={"Range": f"bytes={max(0, size - TAIL)}-{size - 1}"})
    if r.status_code not in (200, 206): raise RuntimeError(f"HTTP {r.status_code}")
    body = r.content
    if len(body) != min(TAIL, size): raise RuntimeError(f"tail length {len(body)}")
    z = zipfile.ZipFile(TailFile(size, body))
    return {i.filename: [i.file_size, i.CRC, i.header_offset, i.compress_type] for i in z.infolist()}

def main():
    rel = json.load(open(T + "raw/RELEASE_LISTING.json"))
    tr = json.load(open(T + "raw/TRAIN_CLIPS_4512.json"))
    train = set(tr["train_clips_4512"])
    scenes = sorted(rel["scenes"])
    after_train = [s for s in scenes if s not in train]
    sizes = {s: [v["size"] for k, v in rel["scenes"][s].items() if k.endswith(".usdz")][0] for s in after_train}
    t0 = time.time(); out, err = {}, {}
    def one(s):
        for a in range(3):
            try: return s, directory(s, sizes[s]), None
            except Exception as e: last = repr(e)[:200]; time.sleep(2 * (a + 1))
        return s, None, last
    with ThreadPoolExecutor(8) as ex:
        for i, (s, d, e) in enumerate(ex.map(one, after_train)):
            if d is None: err[s] = e
            else: out[s] = d
            if i % 200 == 0: print(i, len(out), len(err), round(time.time() - t0), flush=True)
    renderable = [s for s in after_train if s in out and all(n in out[s] for n in NEED)]
    missing = {s: [n for n in NEED if n not in out[s]] for s in after_train if s in out and s not in set(renderable)}
    summ = {"n_release": len(scenes), "n_train_excluded": len(scenes) - len(after_train), "n_after_train": len(after_train),
            "n_directory_read_errors": len(err), "n_missing_members": len(missing), "n_renderable_universe": len(renderable),
            "train_excluded_ids": sorted(set(scenes) & train), "missing_members": missing, "directory_errors": err,
            "all_members_stored": all(v[3] == 0 for s in renderable for n, v in out[s].items() if n in NEED),
            "seconds": round(time.time() - t0)}
    json.dump(summ, open(T + "raw/UNIVERSE.json", "w"), indent=1)
    json.dump({s: {n: out[s][n] for n in NEED} for s in renderable}, open(T + "raw/USDZ_DIRECTORY_RENDERER_MEMBERS.json", "w"))
    print({k: v for k, v in summ.items() if k not in ("train_excluded_ids", "missing_members", "directory_errors")})

main()
