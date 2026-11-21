"""ONE-TIME input staging for the B1 416x1024 rebuild on Thor.

WHY THIS FILE EXISTS: the banked builder
``2026-09-16-256x1024-cache/code/build_v2ep_wide.py`` expects

    <root>/r0/camera_front_wide/<clip>.mp4
    <root>/r0/camera_front_wide/<clip>.timestamps.parquet
    <root>/labels/egomotion/egomotion_all.zip

and Thor carries NONE of those three spellings (MEASURED 2026-09-23):

  * ``/home/nvidia/data/physicalai-b1/r0/camera_front_wide`` is a symlink to
    ``/home/nvidia/data/b1-bundle/camera`` -- a directory that DOES NOT EXIST.
    Zero mp4s are on the box; all 4,713 come from HF (61.55 GB, exact, summed
    from the corpus's own ``camera_sha256.json``).
  * the timestamps are one ``timestamps.tar`` (4,719 members), not loose files.
  * egomotion is 1,411 PER-CHUNK zips, not a single ``egomotion_all.zip``.

⭐ THE CHOICE MADE HERE, AND WHY: materialise the layout the banked builder
already expects, rather than fork the builder to read Thor's layout. The
builder is the object whose provenance we are relying on ("THIS IS A DRIVER,
NOT A SECOND RESAMPLER"); a second spelling of its input paths is exactly the
divergence its own docstring warns about. Merging is SAFE because
``physicalai.load_egomotion`` resolves a member by
``endswith(f"{clip_id}.egomotion.parquet")`` -- a flat merge preserves that
lookup byte-for-byte (verified: chunk_0007 holds exactly one such member).

Idempotent: every step checks CONTENT (openable archive / member count), never
presence, and re-does only what is missing.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, tarfile, time, zipfile

ROOT = "/home/nvidia/data/_b1stage416"
SRC = "/home/nvidia/data/physicalai-b1"
TSTAR = "/home/nvidia/data/b1-bundle/timestamps/timestamps.tar"
KEEP = "/home/nvidia/data/physicalai-b1-w120-256x640cyl"


def sha12(c):
    return hashlib.sha256(c.encode()).hexdigest()[:12]


def clip_list():
    """The 4,713 the KEEP corpus actually contains -- the authoritative set.

    NOT r0_selection.parquet (4,719): that is the pre-gate ingest list. The
    deployed 256x640 cache's own ``_geometry.json`` records kept=4713 after the
    parity gate dropped 6 clips inside the 40-episode val deployment. Rebuilding
    the 4,719 would be an episode RE-SELECTION.
    """
    ids = sorted(p[: -len(".v2ep.pt")] for p in os.listdir(KEEP)
                 if p.endswith(".v2ep.pt"))
    return ids


def stage_timestamps(cam_dir):
    """Extract all timestamps parquets (flat) from the tar. ~51 MB, kept."""
    have = sum(1 for p in os.listdir(cam_dir)
               if p.endswith(".timestamps.parquet"))
    if have >= 4719:
        print(f"ZZTS-SKIP-{have}ZZ", flush=True)
        return have
    with tarfile.open(TSTAR) as t:
        members = [m for m in t.getmembers()
                   if m.name.endswith(".timestamps.parquet")]
        for m in members:
            m.name = os.path.basename(m.name)          # flatten
            t.extract(m, cam_dir)
    have = sum(1 for p in os.listdir(cam_dir)
               if p.endswith(".timestamps.parquet"))
    print(f"ZZTS-{have}ZZ", flush=True)
    return have


def merge_ego(out_zip):
    """Merge the 1,411 per-chunk egomotion zips into one flat archive.

    CONTENT-checked: a pre-existing archive counts as done only if it OPENS and
    carries >= the expected member count. A truncated zip is removed, never
    resumed onto (the ``_ensure_ego`` lesson: curl -C - onto debris).
    """
    src = os.path.join(SRC, "labels", "egomotion")
    chunks = sorted(p for p in os.listdir(src) if p.endswith(".zip"))
    if os.path.exists(out_zip):
        try:
            with zipfile.ZipFile(out_zip) as z:
                n = len(z.namelist())
            if n >= 4713:
                print(f"ZZEGO-SKIP-{n}ZZ", flush=True)
                return n
        except Exception:                                        # noqa: BLE001
            pass
        os.unlink(out_zip)
    tmp = out_zip + ".tmp"
    seen = set()
    t0 = time.time()
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as out:
        for i, cz in enumerate(chunks):
            with zipfile.ZipFile(os.path.join(src, cz)) as z:
                for nm in z.namelist():
                    base = os.path.basename(nm)
                    if not base.endswith(".egomotion.parquet") or base in seen:
                        continue
                    seen.add(base)
                    out.writestr(base, z.read(nm))
            if (i + 1) % 400 == 0:
                print(f"ZZEGOP-{i+1}-{len(seen)}ZZ", flush=True)
    os.replace(tmp, out_zip)
    with zipfile.ZipFile(out_zip) as z:                  # re-open: CONTENT check
        n = len(z.namelist())
    print(f"ZZEGO-{n}-{time.time()-t0:.1f}sZZ", flush=True)
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=ROOT)
    a = p.parse_args()
    cam = os.path.join(a.root, "r0", "camera_front_wide")
    ego_dir = os.path.join(a.root, "labels", "egomotion")
    os.makedirs(cam, exist_ok=True)
    os.makedirs(ego_dir, exist_ok=True)

    ids = clip_list()
    with open(os.path.join(a.root, "clips_4713.txt"), "w") as fh:
        fh.write("\n".join(ids) + "\n")
    print(f"ZZCLIPS-{len(ids)}ZZ", flush=True)

    nts = stage_timestamps(cam)
    nego = merge_ego(os.path.join(ego_dir, "egomotion_all.zip"))

    # POSITIVE per-clip assertion: every clip we will build must have BOTH a
    # timestamps parquet and an egomotion member. A count alone would pass while
    # the missing ones are exactly ours.
    with zipfile.ZipFile(os.path.join(ego_dir, "egomotion_all.zip")) as z:
        egoset = {os.path.basename(n) for n in z.namelist()}
    miss_ts = [c for c in ids
               if not os.path.exists(os.path.join(cam, f"{c}.timestamps.parquet"))]
    miss_ego = [c for c in ids if f"{c}.egomotion.parquet" not in egoset]
    rep = {"n_clips": len(ids), "n_timestamps_extracted": nts,
           "n_egomotion_members": nego,
           "missing_timestamps_sha12": [sha12(c) for c in miss_ts][:50],
           "n_missing_timestamps": len(miss_ts),
           "missing_egomotion_sha12": [sha12(c) for c in miss_ego][:50],
           "n_missing_egomotion": len(miss_ego),
           "egomotion_all_bytes": os.path.getsize(
               os.path.join(ego_dir, "egomotion_all.zip")),
           "staged_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json.dump(rep, open(os.path.join(a.root, "_stage_report.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items()
                      if not k.startswith("missing_")}, indent=1), flush=True)
    ok = (len(miss_ts) == 0 and len(miss_ego) == 0)
    print(f"ZZSTAGE-{'OK' if ok else 'INCOMPLETE'}-{len(miss_ts)}-{len(miss_ego)}ZZ",
          flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
