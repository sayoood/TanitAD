"""Fetch source clips from the pinned HF v7 corpus into a builder staging root.

EVERY mp4 is sha256 + byte-length checked against THAT revision's own
``camera/camera_sha256.json`` -- the identical check Thor's
``/home/nvidia/sam3map/eval/corpus_feeder.py`` runs. A mismatch is a HARD
FAILURE: the file is moved aside and the process exits non-zero. It is never
downgraded to a warning, and a clip that mismatched is never built.

The repo id and revision are NOT guessed here; they are the constants the
production feeder itself uses.

Staging layout produced (what ``v2_compressed``/``physicalai`` require):

    <root>/r0/camera_front_wide/<clip>.mp4
    <root>/r0/camera_front_wide/<clip>.timestamps.parquet
    <root>/labels/egomotion/egomotion_all.zip    (members <clip>.egomotion.parquet)
    <root>/calibration/physicalai_front_wide_intrinsics.csv   (must be placed
        separately -- LOAD-BEARING: without the per-clip table the resampler
        reverts to a corpus-median cy, which is ~215 px wrong for rig B.)

The token is read IN PLACE from Keys.txt and never printed, copied, or put on a
command line.

Usage::

    python fetch_corpus_clips.py --ids ids.txt --root <staging root> \
        [--receipt receipt.json]

⭐ LOCAL-MIRROR MODE (added 2026-09-19, E17 follow-up; the default path above is
unchanged). The source mp4s are already on the dev box, byte-identical to HF
(`…/Data Engineering/Research/2026-09-19-c3-source-transfer-price/`), so::

    python fetch_corpus_clips.py --ids ids.txt --root <staging root> \
        --local-mirror C:/Users/Admin/tanitad-data/physicalai/camera/camera_front_wide_120fov \
        [--local-timestamps-tar <path>] [--local-egomotion-tar <path>] \
        [--link copy|hardlink] [--receipt receipt.json]

stages from disk and downloads NO mp4. ⛔ The content check is NOT relaxed: the
expected hashes still come from HF at the pinned revision -- the sha table AND the
Hub's LFS metadata, which must AGREE -- every mirror file is hashed before anything
is staged, one mismatch stages NOTHING, and every copy is re-hashed. The two tars,
when given locally, are checked against their HF LFS hashes the same way. Logic
and tests: `stack/scripts/corpus_mirror_stage.py`,
`stack/tests/test_corpus_mirror_stage.py`. ⚠️ `--link hardlink` works only on one
NTFS volume; D: is exFAT and refuses, which raises rather than silently copying.
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, shutil, sys, tarfile, time, zipfile
from pathlib import Path

import truststore                                                   # noqa: E402
truststore.inject_into_ssl()      # certifi fails behind this box's TLS proxy
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")
from huggingface_hub import hf_hub_download                         # noqa: E402

#: MEASURED from Thor's production feeder, not assumed:
#: ssh tanitad-thor-wifi 'sed -n 1,80p /home/nvidia/sam3map/eval/corpus_feeder.py'
REPO = "Sayood/tanitad-v7-training-corpus"
REV = "a0cf20dfb4eafa29b0ac1f3c05337f002bc33ca0"
KEYS = r"D:/Projects/TanitAD/Keys.txt"


def load_token():
    """Read the HF token IN PLACE. Never printed, never written anywhere."""
    if os.environ.get("HF_TOKEN"):
        return
    m = re.search(r"\bhf_[A-Za-z0-9]{20,}\b",
                  open(KEYS, encoding="utf-8", errors="replace").read())
    if not m:
        sys.exit(f"no hf_ token in {KEYS}")
    os.environ["HF_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"] = m.group(0)


def sha12(c):
    return hashlib.sha256(c.encode()).hexdigest()[:12]


def sha256f(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True, help="one clip_id per line")
    ap.add_argument("--root", required=True)
    ap.add_argument("--receipt", default="")
    ap.add_argument("--keep-tars", action="store_true")
    ap.add_argument("--local-mirror", default="",
                    help="dir holding <clip>.mp4; stage from it instead of downloading")
    ap.add_argument("--local-timestamps-tar", default="")
    ap.add_argument("--local-egomotion-tar", default="")
    ap.add_argument("--link", choices=("copy", "hardlink"), default="copy")
    a = ap.parse_args()
    load_token()
    if a.local_mirror:
        return main_local_mirror(a)
    root = Path(a.root); cam = root / "r0" / "camera_front_wide"; dl = root / "_dl"
    cam.mkdir(parents=True, exist_ok=True)
    (root / "labels" / "egomotion").mkdir(parents=True, exist_ok=True)
    dl.mkdir(parents=True, exist_ok=True)
    clips = [l.strip() for l in open(a.ids) if l.strip()]
    assert len(set(clips)) == len(clips), "duplicate clip ids"

    tab = json.load(open(hf_hub_download(REPO, "camera/camera_sha256.json",
                                         repo_type="dataset", revision=REV,
                                         local_dir=str(dl))))
    print(f"[fetch] sha table {len(tab)} clips @ rev {REV[:12]}", flush=True)
    unknown = [c for c in clips if c not in tab]
    if unknown:
        sys.exit(f"HARD FAIL: {len(unknown)} clip(s) absent from the revision's "
                 f"sha table: {[sha12(c) for c in unknown][:5]}")

    rows, t_all = [], time.time()
    for i, c in enumerate(clips, 1):
        dest = cam / f"{c}.mp4"
        if dest.exists() and dest.stat().st_size == int(tab[c]["bytes"]) \
                and sha256f(dest) == tab[c]["sha256"]:
            rows.append({"sha12": sha12(c), "bytes": dest.stat().st_size,
                         "sha256": tab[c]["sha256"], "dl_s": 0.0, "cached": True})
            continue
        t0 = time.time()
        p = Path(hf_hub_download(REPO, f"camera/{c}.mp4", repo_type="dataset",
                                 revision=REV, local_dir=str(dl)))
        dt = time.time() - t0
        h, nb = sha256f(p), p.stat().st_size
        if h != tab[c]["sha256"] or nb != int(tab[c]["bytes"]):
            bad = dl / f"MISMATCH_{sha12(c)}.mp4"; p.rename(bad)
            sys.exit(f"HARD FAIL sha/bytes mismatch {sha12(c)}: got {h[:16]}/{nb} "
                     f"want {tab[c]['sha256'][:16]}/{tab[c]['bytes']} -> {bad}")
        os.replace(p, dest)
        for m in (dl / ".cache" / "huggingface" / "download" / "camera").glob(
                f"{c}.mp4.*"):
            m.unlink()
        rows.append({"sha12": sha12(c), "bytes": nb, "sha256": h,
                     "dl_s": round(dt, 2), "cached": False})
        if i % 10 == 0 or i == len(clips):
            print(f"[fetch] {i}/{len(clips)} "
                  f"{sum(r['bytes'] for r in rows)/1e6:.0f} MB "
                  f"{(time.time()-t_all)/60:.1f} min", flush=True)

    need = set(clips)
    t0 = time.time()
    tsp = hf_hub_download(REPO, "timestamps/timestamps.tar", repo_type="dataset",
                          revision=REV, local_dir=str(dl))
    got = set()
    with tarfile.open(tsp) as tf:
        for m in tf:
            cid = os.path.basename(m.name).split(".")[0]
            if m.isfile() and cid in need:
                with open(cam / f"{cid}.timestamps.parquet", "wb") as out:
                    shutil.copyfileobj(tf.extractfile(m), out)
                got.add(cid)
    if got != need:
        sys.exit(f"HARD FAIL: timestamps missing for {len(need-got)} clip(s)")
    print(f"[fetch] timestamps {len(got)}/{len(clips)} in {time.time()-t0:.0f}s",
          flush=True)

    # physicalai.load_egomotion opens a ZIP and wants <clip>.egomotion.parquet;
    # the corpus ships a TAR of <clip>.parquet. Repack ONCE (STORED) so the
    # deployed loader is used UNMODIFIED.
    t0 = time.time()
    egp = hf_hub_download(REPO, "egomotion/egomotion_alpamayo.tar",
                          repo_type="dataset", revision=REV, local_dir=str(dl))
    zp = root / "labels" / "egomotion" / "egomotion_all.zip"
    got = set()
    with tarfile.open(egp) as tf, zipfile.ZipFile(str(zp) + ".tmp", "w",
                                                  zipfile.ZIP_STORED) as z:
        for m in tf:
            cid = os.path.basename(m.name).split(".")[0]
            if m.isfile() and cid in need and cid not in got:
                z.writestr(f"{cid}.egomotion.parquet", tf.extractfile(m).read())
                got.add(cid)
    if got != need:
        sys.exit(f"HARD FAIL: egomotion missing for {len(need-got)} clip(s)")
    os.replace(str(zp) + ".tmp", zp)
    print(f"[fetch] egomotion {len(got)}/{len(clips)} -> "
          f"{zp.stat().st_size/1e6:.1f} MB in {time.time()-t0:.0f}s", flush=True)
    if not a.keep_tars:
        for f in (tsp, egp):
            try:
                os.unlink(f)
            except OSError:
                pass

    nb = sum(r["bytes"] for r in rows)
    fresh = [r for r in rows if not r["cached"]]
    out = {"repo": REPO, "revision": REV, "n_clips": len(rows),
           "sha256_checked": len(rows), "sha256_mismatches": 0,
           "total_bytes": nb, "mb_per_clip": round(nb / len(rows) / 1e6, 2),
           "downloaded_now": len(fresh),
           "dl_seconds_total": round(sum(r["dl_s"] for r in fresh), 1),
           "dl_s_per_clip_mean": round(sum(r["dl_s"] for r in fresh)
                                       / max(len(fresh), 1), 2),
           "wall_seconds": round(time.time() - t_all, 1), "clips": rows}
    if a.receipt:
        json.dump(out, open(a.receipt, "w"), indent=1)
    print(f"[fetch] DONE {len(rows)} clips {nb/1e9:.2f} GB "
          f"({out['mb_per_clip']} MB/clip) in {out['wall_seconds']/60:.1f} min",
          flush=True)


def _hf_lfs_meta():
    """{rfilename: (sha256, size)} for every LFS file at the pinned REV -- the Hub's own
    metadata, independent of the uploader's `camera_sha256.json`. No content download."""
    from huggingface_hub import HfApi
    info = HfApi().repo_info(REPO, repo_type="dataset", revision=REV, files_metadata=True)
    out = {}
    for s in info.siblings:
        lfs = s.lfs
        sha = ((lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None))
               if lfs else None)
        if sha:
            out[s.rfilename] = (sha, s.size)
    return out


def main_local_mirror(a):
    """Stage from a local mirror. Same layout, same pinned revision, same hard failures.
    ⛔ The default path in main() is deliberately NOT refactored into this one: its code is
    left byte-identical so nothing about the HF behaviour can change with this mode."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "stack" / "scripts"))
    import corpus_mirror_stage as MS                                   # noqa: E402
    root = Path(a.root); cam = root / "r0" / "camera_front_wide"; dl = root / "_dl"
    cam.mkdir(parents=True, exist_ok=True)
    (root / "labels" / "egomotion").mkdir(parents=True, exist_ok=True)
    dl.mkdir(parents=True, exist_ok=True)
    clips = [l.strip() for l in open(a.ids) if l.strip()]
    assert len(set(clips)) == len(clips), "duplicate clip ids"
    t_all = time.time()

    tab = json.load(open(hf_hub_download(REPO, "camera/camera_sha256.json",
                                         repo_type="dataset", revision=REV,
                                         local_dir=str(dl))))
    lfs = _hf_lfs_meta()
    lfs_cam = {c: lfs["camera/%s.mp4" % c] for c in clips if "camera/%s.mp4" % c in lfs}
    try:
        expected = MS.reconcile_expected(tab, lfs_cam, clips)
        print(f"[mirror] HF sha table and LFS metadata AGREE on {len(expected)} clip(s) "
              f"@ rev {REV[:12]}", flush=True)
        rows = MS.stage(clips, a.local_mirror, cam, expected, mode=a.link)
    except MS.MirrorMismatch as e:
        sys.exit(str(e))
    print(f"[mirror] staged {len(rows)} clip(s) by {a.link}, every source hashed first "
          f"and every copy re-hashed, {(time.time()-t_all)/60:.1f} min", flush=True)

    downloaded, tar_src = [], {}

    def _tar(rel, local):
        if local:
            want = lfs.get(rel)
            if not want:
                sys.exit(f"HARD FAIL: {rel} has no LFS hash at rev {REV[:12]}")
            try:
                MS.verify_file(local, want[0], want[1], rel)
            except MS.MirrorMismatch as e:
                sys.exit(str(e))
            tar_src[rel] = "local (sha256 = HF LFS)"
            return local
        p = hf_hub_download(REPO, rel, repo_type="dataset", revision=REV, local_dir=str(dl))
        downloaded.append(p)
        tar_src[rel] = "downloaded"
        return p

    need = set(clips)
    tsp = _tar("timestamps/timestamps.tar", a.local_timestamps_tar)
    got = set()
    with tarfile.open(tsp) as tf:
        for m in tf:
            cid = os.path.basename(m.name).split(".")[0]
            if m.isfile() and cid in need:
                with open(cam / f"{cid}.timestamps.parquet", "wb") as out:
                    shutil.copyfileobj(tf.extractfile(m), out)
                got.add(cid)
    if got != need:
        sys.exit(f"HARD FAIL: timestamps missing for {len(need-got)} clip(s)")

    egp = _tar("egomotion/egomotion_alpamayo.tar", a.local_egomotion_tar)
    zp = root / "labels" / "egomotion" / "egomotion_all.zip"
    got = set()
    with tarfile.open(egp) as tf, zipfile.ZipFile(str(zp) + ".tmp", "w",
                                                  zipfile.ZIP_STORED) as z:
        for m in tf:
            cid = os.path.basename(m.name).split(".")[0]
            if m.isfile() and cid in need and cid not in got:
                z.writestr(f"{cid}.egomotion.parquet", tf.extractfile(m).read())
                got.add(cid)
    if got != need:
        sys.exit(f"HARD FAIL: egomotion missing for {len(need-got)} clip(s)")
    os.replace(str(zp) + ".tmp", zp)
    if not a.keep_tars:
        for f in downloaded:          # ⛔ ONLY what this run downloaded -- never local copies
            try:
                os.unlink(f)
            except OSError:
                pass

    nb = sum(r["bytes"] for r in rows)
    out = {"repo": REPO, "revision": REV, "mode": "local-mirror",
           "mirror": a.local_mirror, "link": a.link,
           "hash_sources": ("camera/camera_sha256.json AND the Hub's LFS metadata at the "
                            "pinned revision, required to agree"),
           "n_clips": len(rows), "sha256_checked": len(rows), "sha256_mismatches": 0,
           "total_bytes": nb, "mb_per_clip": round(nb / max(len(rows), 1) / 1e6, 2),
           "downloaded_now": 0, "tars": tar_src,
           "wall_seconds": round(time.time() - t_all, 1), "clips": rows}
    if a.receipt:
        json.dump(out, open(a.receipt, "w"), indent=1)
    print(f"[mirror] DONE {len(rows)} clips {nb/1e9:.2f} GB, 0 mp4 downloaded, "
          f"in {out['wall_seconds']/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
