"""Card PREREG_V8F_LEVER3 (4c019b01), build steps 1 and 2: repack clip files to the pool4x10 grid only.

  selftest                       the byte-compare must catch a planted 1-byte change, a dtype change and a missing key
  tenk   --shards SHARDS.json    pull each 10k shard tar from HF, verify its sha256 against the hub LFS sha256, repack its
                                 clips, check every repacked clip (control b1), delete the tar (logged). Resumable per shard.
  corpus --src <dir>             repack the corpus part (the union's clips that are not in the 10k manifest) from a local
                                 cache dir. Resumable per clip.

Repack = torch.load the clip, drop ONLY the key "pool8x20", torch.save. Control b1: the saved file, reloaded, has every
kept key with the same dtype, shape and bytes as the source (tensors compared as raw bytes), and no other key.
"""
import argparse, concurrent.futures as cf, datetime as dt, hashlib, io, json, os, re, shutil, sys, tarfile, time
import torch

DROP = "pool8x20"
GIB = 1024 ** 3
KEYS_TXT = "E:/Projects/TanitAD/Keys.txt"
REPO = "Sayood/tanitad-datascale-10k"


def now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(*a):
    print(now(), *a, flush=True)


def sha256_file(p, bs=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bs), b""):
            h.update(b)
    return h.hexdigest()


def tbytes(t):
    t = t.detach().contiguous().reshape(-1)
    return t.view(torch.uint8) if t.numel() else t


def same(a, b):
    """True iff a and b are equal key by key: tensors by dtype, shape and raw bytes; other values by ==."""
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
            return False
        if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape):
            return False
        return bool(torch.equal(tbytes(a), tbytes(b)))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def repack_bytes(raw: bytes, dst: str):
    """Repack one clip given its source bytes; returns the row. Raises on any control failure."""
    src = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if not isinstance(src, dict):
        raise RuntimeError(f"{dst}: source is not a dict")
    had = DROP in src
    kept = {k: v for k, v in src.items() if k != DROP}
    tmp = dst + ".tmp"
    torch.save(kept, tmp)
    back = torch.load(tmp, map_location="cpu", weights_only=True)
    if set(back) != set(kept) or DROP in back:
        raise RuntimeError(f"{dst}: b1 key set differs")
    for k in kept:
        if not same(back[k], src[k]):
            raise RuntimeError(f"{dst}: b1 key {k} differs from the source")
    os.replace(tmp, dst)
    with open(dst, "rb") as f:
        out = f.read()
    return {"bytes": len(out), "md5": hashlib.md5(out).hexdigest(), "src_bytes": len(raw),
            "src_md5": hashlib.md5(raw).hexdigest(), "dropped": DROP if had else None, "keys": sorted(kept)}


def selftest():
    g = torch.Generator().manual_seed(0)
    base = {"pool4x10": torch.randn(3, 4, 8, generator=g).to(torch.float8_e4m3fn), "pool8x20": torch.randn(3, 2, 8, generator=g),
            "v": torch.randn(3, generator=g), "pose_grid": torch.randn(3, 4, generator=g).double()}
    ok = same(base, {k: v.clone() for k, v in base.items()})
    m1 = {k: v.clone() for k, v in base.items()}
    b = m1["pool4x10"].view(torch.uint8).clone(); b[0, 0, 0] ^= 1; m1["pool4x10"] = b.view(torch.float8_e4m3fn)
    m2 = {k: v.clone() for k, v in base.items()}; m2["v"] = m2["v"].double()
    m3 = {k: v.clone() for k, v in base.items()}; m3.pop("v")
    res = {"equal_copy": ok, "planted_byte": same(base, m1), "dtype_change": same(base, m2), "missing_key": same(base, m3)}
    # an end-to-end repack of a synthetic clip
    buf = io.BytesIO(); torch.save(base, buf)
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"selftest_clip_{os.getpid()}.pt")
    row = repack_bytes(buf.getvalue(), d)
    back = torch.load(d, weights_only=True)
    res["repack_dropped"] = row["dropped"] == DROP and DROP not in back and same(back["pool4x10"], base["pool4x10"])
    os.remove(d)
    good = res["equal_copy"] and not res["planted_byte"] and not res["dtype_change"] and not res["missing_key"] and res["repack_dropped"]
    log("SELFTEST", json.dumps(res), "PASS" if good else "FAIL")
    return good


def hf_token():
    m = re.search(r"hf_[A-Za-z0-9]+", open(KEYS_TXT, encoding="utf-8", errors="ignore").read())
    if not m:
        raise SystemExit("no HF token in Keys.txt")
    return m.group(0)


def free_gib(p):
    return shutil.disk_usage(p).free / GIB


def download(shard, dl_dir, token):
    from huggingface_hub import hf_hub_download
    last = None
    for attempt in range(1, 4):
        try:
            t0 = time.time()
            p = hf_hub_download(REPO, shard["path"], repo_type="dataset", local_dir=dl_dir, token=token)
            sz = os.path.getsize(p)
            sh = sha256_file(p)
            if sz != shard["size"] or sh != shard["sha256"]:
                os.remove(p)
                raise RuntimeError(f"sha/size mismatch {shard['path']}: {sz} {sh}")
            return p, time.time() - t0, attempt
        except Exception as e:  # noqa: BLE001
            last = e
            log("DOWNLOAD-RETRY", shard["path"], attempt, repr(e)[:300])
            time.sleep(30 * attempt)
    raise RuntimeError(f"download failed {shard['path']}: {last!r}")


def cmd_tenk(a):
    if not selftest():
        raise SystemExit("selftest FAIL")
    S = json.load(open(a.shards, encoding="utf-8"))["shards"]
    clips_dir = os.path.join(a.out, "train", "clips")
    os.makedirs(clips_dir, exist_ok=True)
    os.makedirs(a.state, exist_ok=True)
    dl_dir = os.path.abspath(a.dl_dir)
    os.makedirs(dl_dir, exist_ok=True)
    token = hf_token()
    todo = [s for s in S if not os.path.exists(os.path.join(a.state, os.path.basename(s["path"]) + ".done.json"))]
    log("TENK start", len(S), "shards,", len(todo), "to do; C free", round(free_gib(a.out), 1), "GiB; dl free", round(free_gib(dl_dir), 1))
    seen = {}
    for s in S:   # clips of finished shards, for the duplicate check
        dj = os.path.join(a.state, os.path.basename(s["path"]) + ".rows.jsonl")
        if s not in todo and os.path.exists(dj):
            for ln in open(dj, encoding="utf-8"):
                r = json.loads(ln); seen[r["clip"]] = (s["path"], r["src_md5"])
    k = max(1, int(a.prefetch))
    ex = cf.ThreadPoolExecutor(max_workers=k)
    futs = {j: ex.submit(download, todo[j], dl_dir, token) for j in range(min(k, len(todo)))}
    for i, s in enumerate(todo):
        name = os.path.basename(s["path"])
        p, secs, att = futs.pop(i).result()
        log("GOT", name, s["size"], "B in", round(secs), "s", round(s["size"] / max(secs, 1) / 1e6, 1), "MB/s attempt", att)
        if free_gib(a.out) < a.min_free_gib:
            raise SystemExit(f"STOP: {a.out} free {free_gib(a.out):.1f} GiB < {a.min_free_gib}")
        if i + k < len(todo):
            futs[i + k] = ex.submit(download, todo[i + k], dl_dir, token)
        sdir = os.path.join(a.state, name + ".meta")
        os.makedirs(sdir, exist_ok=True)
        rows_p = os.path.join(a.state, name + ".rows.jsonl")
        n = 0
        t0 = time.time()
        man_clips = None
        with open(rows_p, "w", encoding="utf-8") as rf, tarfile.open(p, "r:") as t:
            for m in t:
                if not m.isfile():
                    continue
                if m.name.startswith("train/clips/") and m.name.endswith(".pt"):
                    clip = os.path.basename(m.name)[:-3]
                    raw = t.extractfile(m).read()
                    row = repack_bytes(raw, os.path.join(clips_dir, clip + ".pt"))
                    if clip in seen and seen[clip][1] != row["src_md5"]:
                        raise SystemExit(f"STOP: clip {clip} in {seen[clip][0]} and {name} with different bytes")
                    seen[clip] = (s["path"], row["src_md5"])
                    row = {"clip": clip, "part": "10k", "shard": s["path"], **row}
                    rf.write(json.dumps(row) + "\n")
                    n += 1
                    if n % 50 == 0:
                        rf.flush()
                        log(name, n, "clips", round((time.time() - t0) / n, 3), "s/clip")
                else:
                    dst = os.path.join(sdir, m.name.replace("/", "__"))
                    with open(dst, "wb") as f:
                        f.write(t.extractfile(m).read())
                    if m.name == "train/manifest.jsonl":
                        man_clips = [json.loads(x)["clip"] for x in open(dst, encoding="utf-8") if x.strip()]
        got = [json.loads(x)["clip"] for x in open(rows_p, encoding="utf-8")]
        if man_clips is None or sorted(set(man_clips)) != sorted(got):
            raise SystemExit(f"STOP: {name}: clip files {len(got)} vs shard manifest {None if man_clips is None else len(set(man_clips))}")
        # delete the verified tar by its asserted path (card section 3 step 1; rule: literal path, logged)
        assert os.path.abspath(p).startswith(dl_dir) and p.endswith(".cache.tar"), p
        sz = os.path.getsize(p)
        os.remove(p)
        with open(a.deletions, "a", encoding="utf-8") as df:
            df.write(f"{now()}\t{os.path.abspath(p)}\t{sz} B\tHF {REPO}:{s['path']} sha256 {s['sha256']} (re-creatable); repacked to {clips_dir}\n")
        with open(os.path.join(a.state, name + ".done.json"), "w", encoding="utf-8") as f:
            json.dump({"shard": s["path"], "sha256": s["sha256"], "n_clips": n, "seconds_repack": round(time.time() - t0, 1),
                       "seconds_download": round(secs, 1), "finished": now()}, f)
        log("DONE-SHARD", name, n, "clips; C free", round(free_gib(a.out), 1), "GiB")
    ex.shutdown()
    log("TENK ALL DONE", len(seen), "clips")
    with open(os.path.join(a.state, "TENK.done"), "w", encoding="utf-8") as f:
        f.write(f"{now()} {len(seen)}\n")


def cmd_corpus(a):
    if not selftest():
        raise SystemExit("selftest FAIL")
    union = [json.loads(x)["clip"] for x in open(a.union_manifest, encoding="utf-8") if x.strip()]
    tenk = {json.loads(x)["clip"] for x in open(a.tenk_manifest, encoding="utf-8") if x.strip()}
    want = [c for c in union if c not in tenk]
    log("CORPUS", len(union), "union clips,", len(tenk), "10k,", len(want), "corpus to repack")
    clips_dir = os.path.join(a.out, "train", "clips")
    os.makedirs(clips_dir, exist_ok=True)
    os.makedirs(a.state, exist_ok=True)
    rows_p = os.path.join(a.state, "corpus.rows.jsonl")
    done = set()
    if os.path.exists(rows_p):
        for ln in open(rows_p, encoding="utf-8"):
            r = json.loads(ln)
            if os.path.exists(os.path.join(clips_dir, r["clip"] + ".pt")):
                done.add(r["clip"])
    t0 = time.time(); n = 0
    with open(rows_p, "a", encoding="utf-8") as rf:
        for c in want:
            if c in done:
                continue
            if free_gib(a.out) < a.min_free_gib:
                raise SystemExit(f"STOP: {a.out} free {free_gib(a.out):.1f} GiB < {a.min_free_gib}")
            src = os.path.join(a.src, c + ".pt")
            with open(src, "rb") as f:
                raw = f.read()
            row = repack_bytes(raw, os.path.join(clips_dir, c + ".pt"))
            rf.write(json.dumps({"clip": c, "part": "corpus", "src": src, **row}) + "\n")
            n += 1
            if n % 100 == 0:
                rf.flush()
                log("corpus", len(done) + n, "/", len(want), round((time.time() - t0) / n, 3), "s/clip")
    log("CORPUS ALL DONE", len(done) + n)
    with open(os.path.join(a.state, "CORPUS.done"), "w", encoding="utf-8") as f:
        f.write(f"{now()} {len(done) + n}\n")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("selftest")
    t = sp.add_parser("tenk")
    t.add_argument("--shards", required=True); t.add_argument("--out", required=True); t.add_argument("--state", required=True)
    t.add_argument("--dl-dir", required=True); t.add_argument("--deletions", required=True)
    t.add_argument("--min-free-gib", type=float, default=25.0)
    t.add_argument("--prefetch", type=int, default=1, help="downloads in flight")
    c = sp.add_parser("corpus")
    c.add_argument("--src", required=True); c.add_argument("--union-manifest", required=True); c.add_argument("--tenk-manifest", required=True)
    c.add_argument("--out", required=True); c.add_argument("--state", required=True); c.add_argument("--min-free-gib", type=float, default=25.0)
    a = ap.parse_args()
    torch.set_num_threads(4)
    if a.cmd == "selftest":
        sys.exit(0 if selftest() else 1)
    (cmd_tenk if a.cmd == "tenk" else cmd_corpus)(a)


if __name__ == "__main__":
    main()
