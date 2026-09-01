"""Pull + CONTENT-VERIFY the published w120 256x640-cylindrical v2 episode cache.

WHY THIS IS A PULL AND NOT A BUILD (2026-09-01, DataEng FlyWheel)
----------------------------------------------------------------
The cache REF-C v3 needs was already built and pushed. MEASURED by full
paginated enumeration of the whole ``Sayood`` HF namespace (18 datasets +
26 models): exactly two repos carry ``*.v2ep.pt`` --

  * ``Sayood/tanitad-physicalai-w120-256x640cyl``
      ``physicalai-train-e438721ae894-w120-256x640cyl/``  2,400 files  79.15 GB
      ``physicalai-val-0c5f7dac3b11-w120-256x640cyl/``      600 files  19.75 GB
  * ``Sayood/tanitad-transfer-2026-08``  20 files (a duplicate val subset)

and the 24 reference payloads already on the A40 pod at
``/workspace/TanitAD/data/eps/`` are **24/24 a subset of the train dir**, so the
published cache is the same artifact the pod samples came from. Rebuilding it
from the 4,719-clip v7 corpus would have spent ~3.4-4.6 h of pod time
regenerating bytes that already exist.

  Geometry (from the repo's own ``_geometry.json``, matching the pod samples
  field-for-field): 256x640, ``f_ref`` 305.5774907364391, ``projection``
  cylindrical, ``codec`` **png** (LOSSLESS -- load-bearing: ``v2_dataset.py``
  and ``slice_v2_cache.py`` REFUSE to sub-frame a lossy cache), ``n_stack`` 3.

⛔ VERIFY BY CONTENT, NEVER BY PRESENCE. A partial/aborted download leaves a
file at a plausible size that ``torch.load`` cannot read, or that reads and is
empty. ``--verify`` opens every payload, checks the invariants below, and only
a payload that PASSES counts as done -- which is also what makes the pull
resumable without re-fetching good bytes:

    jpeg_len.shape[0] > 0
    poses.shape[0]   == jpeg_len.shape[0]
    actions.shape[0] == jpeg_len.shape[0]
    image_h/image_w  == 256/640,  projection_mode == cylindrical
    codec == png,    n_stack == 3
    a real decode of one frame is non-zero      (the all-zero-bank trap)

Usage (pod):
    python pod_pull_b1_epcache.py --out /workspace/TanitAD/data/b1-epcache \
        --split train --workers 16
    python pod_pull_b1_epcache.py --out ... --verify-only

The HF token is read from a file (default ``/root/.hf_token``) and handed to
curl through a ``--config`` snippet on STDIN, so it never appears in argv and
never lands in any ``ps`` listing -- the same discipline as
``v2_compressed._hf_download``.
"""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = "Sayood/tanitad-physicalai-w120-256x640cyl"
DIRS = {
    "train": "physicalai-train-e438721ae894-w120-256x640cyl",
    "val": "physicalai-val-0c5f7dac3b11-w120-256x640cyl",
}
EXPECT = {"image_h": 256, "image_w": 640, "projection_mode": "cylindrical",
          "codec": "png", "n_stack": 3}


def _tok(path):
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return os.environ.get("HF_TOKEN", "").strip()


def _curl(url, tok, out=None, dump_hdr=None):
    """curl with the bearer token on STDIN (never argv)."""
    cmd = ["curl", "-fL", "--silent", "--show-error", "--max-time", "900",
           "--retry", "8", "--retry-delay", "3", "--connect-timeout", "30"]
    if out:
        cmd += ["-C", "-", "-o", out,
                "--speed-limit", "500000", "--speed-time", "30"]
    if dump_hdr:
        cmd += ["-D", dump_hdr]
    cfg = ""
    if tok:
        cmd += ["--config", "-"]
        cfg = 'header = "Authorization: Bearer ' + tok + '"\n'
    cmd.append(url)
    return subprocess.run(cmd, input=cfg.encode() if cfg else None,
                          capture_output=out is None)


def list_files(subdir, tok):
    """Paginated tree listing.

    ⛔ PAGINATION IS LOAD-BEARING. The tree API caps at 1,000 entries per page
    and a single-page read of this repo returns 993 of its 6,061 files -- a
    number that looks like a complete small repo. MEASURED 2026-09-01: that
    truncation is what made an earlier probe report "990 .pt files" for a repo
    that actually holds 3,000 .v2ep.pt. Follow the Link rel="next" header.
    """
    url = (f"https://huggingface.co/api/datasets/{REPO}"
           f"/tree/main?recursive=true&limit=1000")
    hdr = "/tmp/_hfhdr.txt"
    out, pages = [], 0
    while url and pages < 60:
        r = _curl(url, tok, dump_hdr=hdr)
        pages += 1
        try:
            d = json.loads(r.stdout.decode())
        except Exception as e:                                    # noqa: BLE001
            raise SystemExit(f"tree listing failed: {e}: "
                             f"{r.stdout[:200]!r}")
        if isinstance(d, dict):
            raise SystemExit(f"tree listing error: {str(d)[:300]}")
        out += [x for x in d if x.get("type") == "file"]
        m = None
        try:
            with open(hdr, errors="replace") as fh:
                m = re.search(r'[Ll]ink:\s*<([^>]+)>;\s*rel="next"', fh.read())
        except OSError:
            pass
        url = m.group(1) if m else None
    pref = subdir + "/"
    return sorted(((f["path"], f.get("size", 0) or 0) for f in out
                   if f["path"].startswith(pref)
                   and f["path"].endswith(".v2ep.pt")))


def verify(path):
    """Open the payload and check the invariants. Returns (ok, reason).

    ⚠️ Callers that fan this out across PROCESSES must pin the thread count
    first (``OMP_NUM_THREADS=1`` + ``torch.set_num_threads(1)``) -- torch spawns
    ~113 threads per process and a 16-way pool otherwise thrashes to a standstill
    that looks exactly like a hang. ``_verify_one`` does this.
    """
    import torch
    try:
        d = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:                                        # noqa: BLE001
        return False, f"load: {type(e).__name__}: {e}"
    try:
        n = int(d["jpeg_len"].shape[0])
        if n <= 0:
            return False, "jpeg_len empty"
        if int(d["poses"].shape[0]) != n:
            return False, f"poses {int(d['poses'].shape[0])} != jpeg_len {n}"
        if int(d["actions"].shape[0]) != n:
            return False, f"actions {int(d['actions'].shape[0])} != {n}"
        for k, want in EXPECT.items():
            got = d.get(k)
            if got != want:
                return False, f"{k}={got!r} != {want!r}"
        if int(d["jpeg_buf"].numel()) != int(d["jpeg_len"].sum()):
            return False, "jpeg_buf length != sum(jpeg_len)"
        # the all-zero-bank trap: a file can be full-size and decode to nothing
        import torchvision.io as tvio
        ln = int(d["jpeg_len"][0])
        img = tvio.decode_png(d["jpeg_buf"][:ln], mode=tvio.ImageReadMode.RGB)
        if tuple(img.shape) != (3, EXPECT["image_h"], EXPECT["image_w"]):
            return False, f"decoded shape {tuple(img.shape)}"
        if int(img.max()) == 0:
            return False, "decoded frame is ALL ZERO"
    except KeyError as e:
        return False, f"missing key {e}"
    except Exception as e:                                        # noqa: BLE001
        return False, f"verify: {type(e).__name__}: {e}"
    return True, f"ok n={n}"


def _verify_one(path):
    """Process-pool entry point: pin threads, then verify. See ``verify``."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:                                             # noqa: BLE001
        pass
    ok, why = verify(path)
    return path, ok, why


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="train", choices=sorted(DIRS))
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--token-file", default="/root/.hf_token")
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--limit", type=int, default=0,
                   help="stop after N files (timing probe)")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tok = _tok(a.token_file)
    subdir = DIRS[a.split]

    if a.verify_only:
        files = sorted(f for f in os.listdir(a.out) if f.endswith(".v2ep.pt"))
        paths = [os.path.join(a.out, f) for f in files]
        ok, bad, t0 = 0, [], time.time()
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for i, (pth, good, why) in enumerate(
                    ex.map(_verify_one, paths, chunksize=4)):
                if good:
                    ok += 1
                else:
                    bad.append((os.path.basename(pth), why))
                    print(f"  BAD {os.path.basename(pth)}: {why}", flush=True)
                if (i + 1) % 400 == 0:
                    print(f"  ..verified {i+1}/{len(paths)} ok={ok} "
                          f"bad={len(bad)} {time.time()-t0:.0f}s", flush=True)
        print(f"VERIFY split={a.split} files_on_disk={len(files)} "
              f"CONTENT_VERIFIED={ok} BAD={len(bad)} "
              f"({time.time()-t0:.0f}s)", flush=True)
        return 0 if not bad else 1

    want = list_files(subdir, tok)
    if a.limit:
        want = want[:a.limit]
    total_b = sum(s for _, s in want)
    print(f"[pull] repo={REPO} subdir={subdir} remote_files={len(want)} "
          f"remote_bytes={total_b/1024**3:.2f}GB", flush=True)

    todo = []
    skipped = 0
    for path, size in want:
        dest = os.path.join(a.out, os.path.basename(path))
        if os.path.exists(dest) and os.path.getsize(dest) == size:
            skipped += 1                       # size-gated; --verify-only is truth
            continue
        todo.append((path, size, dest))
    print(f"[pull] already present (size-matched)={skipped} to_fetch={len(todo)} "
          f"bytes={sum(s for _, s, _ in todo)/1024**3:.2f}GB", flush=True)

    t0 = time.time()
    got_b, got_n, fails = 0, 0, []

    def one(job):
        path, size, dest = job
        url = (f"https://huggingface.co/datasets/{REPO}/resolve/main/"
               + "/".join(path.split("/")))
        tmp = dest + ".tmp"
        r = _curl(url, tok, out=tmp)
        if r.returncode != 0 or not os.path.exists(tmp):
            return path, 0, f"curl rc={r.returncode}"
        if os.path.getsize(tmp) != size:
            got = os.path.getsize(tmp)
            os.unlink(tmp)
            return path, 0, f"size {got} != {size}"
        os.replace(tmp, dest)                  # atomic: no half file is ever seen
        return path, size, None

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(one, j) for j in todo]
        for i, fu in enumerate(as_completed(futs)):
            path, nb, err = fu.result()
            if err:
                fails.append((path, err))
                print(f"[pull] FAIL {os.path.basename(path)}: {err}", flush=True)
            else:
                got_b += nb
                got_n += 1
            if (i + 1) % 25 == 0 or (i + 1) == len(futs):
                el = time.time() - t0
                mbs = got_b / 1024**2 / max(el, 1e-6)
                left = sum(s for _, s, _ in todo) - got_b
                eta = left / max(got_b / max(el, 1e-6), 1.0)
                print(f"[pull] {i+1}/{len(futs)} ok={got_n} fail={len(fails)} "
                      f"{got_b/1024**3:.2f}GB {mbs:.1f}MB/s "
                      f"elapsed={el/60:.1f}min eta={eta/60:.1f}min", flush=True)

    el = time.time() - t0
    print(f"[pull] DONE ok={got_n} fail={len(fails)} "
          f"{got_b/1024**3:.2f}GB in {el/60:.1f}min "
          f"({got_b/1024**2/max(el,1e-6):.1f}MB/s)", flush=True)
    for path, err in fails[:40]:
        print(f"  FAILED {path}: {err}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
