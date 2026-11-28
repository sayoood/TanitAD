"""Build a v2ep episode cache at ANY (H, W) cylindrical frame over a fixed field.

THIS IS A DRIVER, NOT A SECOND RESAMPLER. Every pixel comes from
``v2_compressed.build_compressed`` -> ``physicalai`` / ``calib.cylindrical_rectify``
-- the same call chain that produced ``physicalai-b1-w120-256x640cyl``. The ONLY
thing this file changes is the ``CanonicalFrame`` handed to it. Two spellings of
a projection is how the sampler and the vocabulary silently disagree.

f_ref IS SOLVED FROM THE FIELD, NEVER COPIED. ``CanonicalFrame.from_hfov``
inverts ``HFOV = 2*(W/2)/f_ref`` (cylindrical is LINEAR in azimuth), so

    W =  640 -> f_ref = 320/1.0471976 = 305.5774907
    W = 1024 -> f_ref = 512/1.0471976 = 488.9239852

Both are asserted below against the payload's OWN stored frame, so a build that
kept the 640 focal at 1024 columns (which would deliver 74.93 deg, not 120) dies
before it writes anything.

ONE AXIS IS *NOT* PRESERVED, BY CONSTRUCTION. ``cylindrical_rays`` uses the SAME
``f_ref`` for both axes (``y_n = (v-(H-1)/2)/f_ref``), so holding H at 256 while
f_ref rises 1.6x SHRINKS the vertical field 45.456 deg -> 29.341 deg. The build
stamps both into the geometry record and prints the loss; it does not silently
absorb it. To hold the vertical field too, H must rise with W (408x1024 restores
45.296 deg).

Usage::

    python build_v2ep_wide.py --out <dir> --root <staging root> \
        --clips ids.txt --width 1024 --height 256 --workers 6 --role eval
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed

# Pin threads BEFORE torch/av are imported anywhere (pod_build_b1_epcache's
# MEASURED lesson: torch spawns ~113 threads/process and an unpinned pool makes
# no progress while looking exactly like a hang).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("V2_TORCH_THREADS", "1")
os.environ.setdefault("PAI_DECODE_THREADS", "1")

STACK = os.environ.get("TANITAD_STACK", r"C:/Users/Admin/tanitad-snap-20260915/stack")
sys.path.insert(0, STACK)
sys.path.insert(0, os.path.join(STACK, "scripts"))

# This dev box is cp1252. parity.guard_corpus_build prints U+26A0 on the
# sanctioned-audit branch, so the GATE ITSELF raises UnicodeEncodeError and the
# build dies at the guard instead of at the data. Make stdout UTF-8-safe here
# rather than editing the shared stack.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                                                # noqa: BLE001
        pass

HFOV_DEG = 120.0
N_STACK = 3
CODEC = "png"            # LOSSLESS and load-bearing: v2_dataset.py and
                         # slice_v2_cache.py REFUSE to sub-frame a lossy cache.
PROJECTION = "cylindrical"

_G = {}


def frame_for(height: int, width: int):
    """The canonical frame, with the field RE-DERIVED from the built object."""
    from tanitad.data.calib import CanonicalFrame
    fr = CanonicalFrame.from_hfov(HFOV_DEG, height, width, PROJECTION)
    want = (width / 2.0) / math.radians(HFOV_DEG / 2.0)
    assert abs(fr.f_ref - want) < 1e-6, f"f_ref {fr.f_ref} != {want}"
    assert abs(fr.hfov_deg - HFOV_DEG) < 1e-3, f"HFOV {fr.hfov_deg} != {HFOV_DEG}"
    return fr


def _ctx(root, height, width):
    if _G:
        return _G
    import v2_compressed as V
    import pod_pull_b1_epcache as P
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:                                                # noqa: BLE001
        pass
    # Reuse the DEPLOYED verifier; only its expected frame size is parameterized.
    P.EXPECT = {"image_h": height, "image_w": width,
                "projection_mode": PROJECTION, "codec": CODEC, "n_stack": N_STACK}
    _G.update(V=V, P=P, frame=frame_for(height, width),
              cam=os.path.join(root, "r0", "camera_front_wide"),
              ego=os.path.join(root, "labels", "egomotion", "egomotion_all.zip"))
    return _G


def sha12(c):
    return hashlib.sha256(c.encode()).hexdigest()[:12]


def _sha256f(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build_one(args):
    """Build ONE episode, CONTENT-verify it, return its geometry. Never raises."""
    cid, root, out_dir, height, width = args
    g = _ctx(root, height, width)
    os.makedirs(out_dir, exist_ok=True)      # the save is the LAST step of a
    out = os.path.join(out_dir, f"{cid}.v2ep.pt")   # ~40 s decode: make the cheap
    t0 = time.time()                                # precondition hold first.
    try:
        clip = {"clip_id": cid,
                "mp4": os.path.join(g["cam"], f"{cid}.mp4"),
                "timestamps": os.path.join(g["cam"], f"{cid}.timestamps.parquet"),
                "ego_zip": g["ego"]}
        # ⛔ size STAYS 256 -- it is the LEGACY SENTINEL, not the frame height.
        # calib.as_frame(frame, size, 266.0) REFUSES any size != 256 when a
        # CanonicalFrame is also passed ("two sources of truth is the bug this
        # object exists to remove"). Passing size=height happens to work at
        # H=256 and raises ValueError at every other height, so a 408-row build
        # would fail on all 139 clips. The real geometry is carried by `frame`;
        # `size` only lands in the payload's legacy `image_size` field, while
        # `image_h`/`image_w` carry the truth.
        g["V"].build_compressed(clip, out, size=256, n_stack=N_STACK,
                                frame=g["frame"], projection_mode=PROJECTION,
                                codec=CODEC)
    except Exception as e:                                           # noqa: BLE001
        if os.path.exists(out):
            try:
                os.unlink(out)               # never leave a half-built payload
            except OSError:
                pass
        return cid, 0, 0.0, 0, None, f"{type(e).__name__}: {str(e)[:200]}"
    ok, why = g["P"].verify(out)
    if not ok:
        try:
            os.unlink(out)                   # a payload failing content is NOT done
        except OSError:
            pass
        return cid, 0, time.time() - t0, 0, None, f"VERIFY: {why}"
    # ASSERT THE PAYLOAD'S OWN GEOMETRY, not the one we asked for.
    import torch
    d = torch.load(out, map_location="cpu", weights_only=False)
    fr = d["frame"]
    got_hfov = math.degrees(2.0 * (fr["width"] / 2.0) / fr["f_ref"])
    want_f = (width / 2.0) / math.radians(HFOV_DEG / 2.0)
    if (int(fr["width"]) != width or int(fr["height"]) != height
            or abs(float(fr["f_ref"]) - want_f) > 1e-6
            or abs(got_hfov - HFOV_DEG) > 1e-3 or fr["projection"] != PROJECTION):
        os.unlink(out)
        return cid, 0, time.time() - t0, 0, None, (
            f"GEOMETRY: payload frame {fr} -> HFOV {got_hfov:.6f}")
    n = int(d["jpeg_len"].shape[0])
    geo = {"f_ref": float(fr["f_ref"]), "hfov_deg": round(got_hfov, 9),
           "vfov_deg": round(math.degrees(2 * math.atan((fr["height"] / 2)
                                                        / fr["f_ref"])), 6),
           "height": int(fr["height"]), "width": int(fr["width"]),
           "projection": fr["projection"],
           "deg_per_col": round(got_hfov / width, 9)}
    return cid, os.path.getsize(out), time.time() - t0, n, geo, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--clips", required=True, help="one clip_id per line")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--role", default="eval", choices=["train", "eval"])
    p.add_argument("--sanctioned-audit", default="",
                   help="REASON. Builds the requested set whole and stamps "
                        "decision_grade False on everything produced from it. "
                        "Use ONLY when the membership is inherited from an "
                        "existing banked split and filtering would break the "
                        "row correspondence that is the point of the rebuild.")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ids = [ln.strip() for ln in open(a.clips) if ln.strip()]

    fr = frame_for(a.height, a.width)
    ref = frame_for(256, 640)
    print(f"[build] frame {a.height}x{a.width} f_ref={fr.f_ref:.7f} "
          f"HFOV={fr.hfov_deg:.9f} deg ({fr.hfov_deg/a.width:.6f} deg/col) "
          f"VFOV={math.degrees(2*math.atan((a.height/2)/fr.f_ref)):.4f} deg "
          f"[256x640 reference VFOV "
          f"{math.degrees(2*math.atan((256/2)/ref.f_ref)):.4f} deg]", flush=True)

    # THE PARITY INGEST GATE - RUN HERE, NOT ASSUMED (a driver that calls
    # build_compressed directly bypasses v2_compressed.build's own gate).
    from tanitad.data import parity
    parity.require_ingest_gate("build_v2ep_wide")
    kept, gate = parity.guard_corpus_build(
        ids, label=f"build_v2ep_wide -> {a.out}", role=a.role,
        mode="refuse", sanctioned_audit=(a.sanctioned_audit or None))
    print(f"[build] parity gate: role={a.role} kept={len(kept)} "
          f"in_parity_train={gate.get('in_parity_train')} "
          f"in_deployed_val={gate.get('in_deployed_val')} "
          f"decision_grade={gate.get('decision_grade')}", flush=True)
    # The fact the build cannot know it needs rides along uninvited: WHICH clips
    # carry the overlap, as sha12 (the repo-safe spelling), so the PI can act on
    # it instead of rediscovering it. Counts alone would not be actionable.
    try:
        leak = sorted(sha12(c) for c in parity.clips_in_parity_train(ids))
        vleak = sorted(sha12(c) for c in parity.clips_in_deployed_val(ids))
    except Exception as e:                                           # noqa: BLE001
        leak, vleak = [f"UNAVAILABLE: {type(e).__name__}"], []
    gate = dict(gate, overlap_sha12_parity_train=leak,
                overlap_sha12_deployed_val=vleak,
                clean_subset_n=len(ids) - len(leak))
    if a.sanctioned_audit:
        print(f"[build] SANCTIONED AUDIT: decision_grade is FALSE. "
              f"{len(leak)}/{len(ids)} clips are inside the parity TRAIN corpus; "
              f"a clean held-out subset would be {len(ids)-len(leak)} clips.",
              flush=True)

    # RESUME BY CONTENT, NOT BY PRESENCE: a killed build leaves payloads that
    # exist at a plausible size; only one recorded AND present counts as done.
    prev = {}
    man_p = os.path.join(a.out, "MANIFEST.json")
    if os.path.exists(man_p):
        prev = {r["clip_sha12"]: r for r in json.load(open(man_p))["clips"]}
    todo = [c for c in ids if sha12(c) not in prev
            or not os.path.exists(os.path.join(a.out, f"{c}.v2ep.pt"))]
    done = [c for c in ids if c not in todo]
    if a.limit:
        todo = todo[:a.limit]
    print(f"[build] clips={len(ids)} done={len(done)} todo={len(todo)} "
          f"workers={a.workers} codec={CODEC}", flush=True)

    rows = [prev[sha12(c)] for c in done if sha12(c) in prev]
    fails, t0, nb = [], time.time(), 0
    CHUNK = 40                    # short executor lifetimes: a hang costs ONE
    i = 0                         # chunk and the content resume makes it free.
    for c0 in range(0, len(todo), CHUNK):
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = [ex.submit(build_one, (c, a.root, a.out, a.height, a.width))
                    for c in todo[c0:c0 + CHUNK]]
            for fu in as_completed(futs):
                cid, b, el, n, geo, err = fu.result()
                i += 1
                if err:
                    fails.append((sha12(cid), err))
                    print(f"[build] FAIL {sha12(cid)}: {err}", flush=True)
                else:
                    nb += b
                    rows.append({"clip_sha12": sha12(cid), "n_frames": n,
                                 "bytes": b,
                                 "sha256": _sha256f(os.path.join(
                                     a.out, f"{cid}.v2ep.pt")),
                                 "build_s": round(el, 2), **geo})
                if i % 10 == 0 or i == len(todo):
                    w = time.time() - t0
                    print(f"[build] {i}/{len(todo)} ok={i-len(fails)} "
                          f"fail={len(fails)} {nb/1024**3:.2f}GB "
                          f"{i/max(w,1e-9)*3600:.0f}eps/h "
                          f"elapsed={w/60:.1f}min", flush=True)
    w = time.time() - t0
    built = [r for r in rows if r.get("build_s", 0) > 0]
    man = {"cache": os.path.basename(a.out.rstrip("/\\")),
           "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "builder": "TanitAD Research Lab/Data Engineering/Research/"
                      "2026-09-16-256x1024-cache/code/build_v2ep_wide.py",
           "resampler": "v2_compressed.build_compressed -> "
                        "tanitad.data.calib.cylindrical_rectify (UNMODIFIED)",
           "source_repo": "Sayood/tanitad-v7-training-corpus",
           "source_revision": "a0cf20dfb4eafa29b0ac1f3c05337f002bc33ca0",
           "frame": {"height": a.height, "width": a.width,
                     "f_ref": float(fr.f_ref), "projection": PROJECTION,
                     "hfov_deg": float(fr.hfov_deg),
                     "vfov_deg": math.degrees(2*math.atan((a.height/2)/fr.f_ref)),
                     "deg_per_col": float(fr.hfov_deg) / a.width,
                     "centre_col": (a.width - 1) / 2.0,
                     "tag": fr.tag()},
           "codec": CODEC, "n_stack": N_STACK, "projection_mode": PROJECTION,
           "parity_ingest_gate": gate,
           "n_clips": len(rows), "n_failed": len(fails),
           "total_bytes": sum(r["bytes"] for r in rows),
           "build_seconds_this_run": round(w, 1),
           "build_s_per_clip_mean": round(sum(r["build_s"] for r in built)
                                          / max(len(built), 1), 2),
           "workers": a.workers,
           "clips": sorted(rows, key=lambda r: r["clip_sha12"])}
    json.dump(man, open(man_p, "w"), indent=1)
    print(f"[build] DONE ok={len(rows)} fail={len(fails)} "
          f"{sum(r['bytes'] for r in rows)/1024**3:.2f}GB in {w/60:.1f}min",
          flush=True)
    if fails:
        json.dump(dict(fails), open(os.path.join(a.out, "_failures.json"), "w"),
                  indent=1)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
