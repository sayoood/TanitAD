"""Build the B1 (4,719-clip v7 corpus) v2 episode cache at 256x640 cylindrical.

Corpus decision (coordinator, 2026-09-01): B1, not the canonical parity 2,400.
MEASURED rationale: of the 4,719 B1 clips, **4,572 (96.9 %) carry a v7.2
tactical/strategic label**; the parity 2,400 carries one on only 190 (7.9 %),
so training the hierarchy heads on parity would supervise them on <8 % of
windows.

⭐ THIS IS A DRIVER, NOT A NEW ENCODER. Every pixel and every pose comes from
``v2_compressed.build_compressed`` -- the same function that produced the
published parity cache -- so the output cannot drift from the format the
trainer already consumes. What this file adds is (a) the v7 corpus's different
input layout, (b) process-level parallelism over the PNG encode, and (c)
resume-by-CONTENT.

Input layout (the v7 corpus ships a FLAT per-clip layout, not PhysicalAI's
per-chunk zips), staged on the pod as::

    /workspace/v7build/
      r0/camera_front_wide/<clip_id>.mp4                 (from camera/, 57.4 GB)
      r0/camera_front_wide/<clip_id>.timestamps.parquet  (from timestamps.tar)
      labels/egomotion/egomotion_all.zip                 (repacked, see below)
      calibration/physicalai_front_wide_intrinsics.csv   (from nvidia calib)

Two adapters were needed and both are content-checked rather than assumed:

* **egomotion.** ``physicalai.load_egomotion`` opens a ZIP and looks for the
  member ``<clip_id>.egomotion.parquet``; the v7 corpus ships a TAR of
  ``<clip_id>.parquet``. The tar is repacked ONCE into a single STORED zip with
  the member names the loader expects -- so the loader is used unmodified.
* **intrinsics.** ``_physicalai_root_of`` recovers the corpus root by walking up
  to a directory named ``r0``, which is why the mp4s live under
  ``r0/camera_front_wide/``. The root then resolves
  ``calibration/physicalai_front_wide_intrinsics.csv``, giving ``per_clip=True``
  intrinsics. ⛔ THIS IS LOAD-BEARING: without a per-clip table the fallback
  silently reverts the crop to geometric-centre, which is ~215 px wrong for
  rig B -- and rig B is the MAJORITY of this corpus (2,723 vs 1,996).

⛔ THE CORPUS IS **4,713**, NOT 4,719 -- AND THE GATE IS RUN HERE.
``v2_compressed.build`` runs ``parity.require_ingest_gate`` before fetching;
this driver calls ``build_compressed`` directly, so the gate had to be
re-asserted explicitly or it would simply be absent. MEASURED 2026-09-01 by the
instrument itself on the 4,719 v7 clips::

    201 in physicalai-train-e438721ae894
    6 of the 40 deployed-val episodes   (15 % of that episode set)
    -> DROPPED 6; "The corpus is 4713 clips; quote THAT number, never 4719."

Those 6 sit inside the 40-episode val deployment behind every published
open-loop number, so building all 4,719 would put val episodes into training.
The dropped ids are recorded in ``_build_manifest.json``. Independently
corroborated: the earlier Thor build of this corpus also produced **4,713**.

⚠️ A WIDER OVERLAP REMAINS AND IS NOT DROPPED (deliberately): **|B1 ∩ val600|
= 56** and **|B1 ∩ parity2400| = 201**. Only the val40 set is disqualifying for
a training corpus; the val600 overlap means a B1-trained arm's numbers on the
600-episode val set are contaminated for 47 of those episodes. State that before
comparing a B1 arm against any val600 figure.

Usage (pod)::

    python pod_build_b1_epcache.py --out /workspace/TanitAD/data/b1-epcache-4719 \
        --root /workspace/v7build --workers 48
    python pod_build_b1_epcache.py --out ... --contract-test <clip_id>
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

# ⛔ Pin threads BEFORE torch/av are imported anywhere. torch spawns ~113
# threads per process and av's decoder spawns its own; a 48-way pool of
# unpinned processes makes NO progress and looks exactly like a hang
# (MEASURED 2026-07-27: 7 arms at 0-6 % GPU for 50 min).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("V2_TORCH_THREADS", "1")
os.environ.setdefault("PAI_DECODE_THREADS", "1")

STACK = os.environ.get("TANITAD_STACK", "/workspace/TanitAD/stack")
sys.path.insert(0, STACK)
sys.path.insert(0, os.path.join(STACK, "scripts"))

HFOV_DEG = 120.0
FRAME_H, FRAME_W = 256, 640
N_STACK = 3
CODEC = "png"          # LOSSLESS and load-bearing: v2_dataset.py:325 and
                       # slice_v2_cache.py REFUSE to sub-frame a lossy cache.
PROJECTION = "cylindrical"

_G = {}


def _ctx(root):
    """Lazily build the per-process context (frame + paths)."""
    if _G:
        return _G
    from tanitad.data.calib import CanonicalFrame
    import v2_compressed as V
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:                                             # noqa: BLE001
        pass
    _G["V"] = V
    # f_ref falls out of the projection: (W/2)/radians(hfov/2) =
    # 320/1.0471975512 = 305.5774907364391 -- byte-identical to the published
    # parity cache's stored frame, which is the contract this build must match.
    _G["frame"] = CanonicalFrame.from_hfov(HFOV_DEG, FRAME_H, FRAME_W,
                                           PROJECTION)
    _G["cam"] = os.path.join(root, "r0", "camera_front_wide")
    _G["ego"] = os.path.join(root, "labels", "egomotion", "egomotion_all.zip")
    return _G


def build_one(args):
    """Build ONE episode, then CONTENT-verify it. Never raises."""
    cid, root, out_dir = args
    g = _ctx(root)
    # ⛔ The save is the LAST step of a ~20 s decode+encode. A missing out dir
    # therefore throws away the whole episode's compute at the final line and
    # reports a plain failure. MEASURED 2026-09-01: a probe pointed at a
    # non-existent dir burned 10 min and produced 0 files while looking like a
    # slow build. Same family as the analysis-time import that dies after the
    # rollout -- make the cheap precondition hold before paying the expensive part.
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{cid}.v2ep.pt")
    t0 = time.time()
    try:
        clip = {"clip_id": cid,
                "mp4": os.path.join(g["cam"], f"{cid}.mp4"),
                "timestamps": os.path.join(g["cam"],
                                           f"{cid}.timestamps.parquet"),
                "ego_zip": g["ego"]}
        g["V"].build_compressed(clip, out, size=FRAME_H, n_stack=N_STACK,
                                frame=g["frame"], projection_mode=PROJECTION,
                                codec=CODEC)
    except Exception as e:                                        # noqa: BLE001
        if os.path.exists(out):
            try:
                os.unlink(out)          # never leave a half-built payload
            except OSError:
                pass
        return cid, 0, 0.0, f"{type(e).__name__}: {str(e)[:160]}"
    from pod_pull_b1_epcache import verify
    ok, why = verify(out)
    if not ok:
        try:
            os.unlink(out)              # a payload that fails content is NOT done
        except OSError:
            pass
        return cid, 0, time.time() - t0, f"VERIFY: {why}"
    return cid, os.path.getsize(out), time.time() - t0, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--root", default="/workspace/v7build")
    p.add_argument("--workers", type=int, default=48)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--contract-test", default="",
                   help="build ONE clip id into --out and stop")
    p.add_argument("--no-parity-exclude", action="store_true",
                   help="record the overlap but do NOT drop val40 clips "
                        "(inadmissible for a training corpus; audit use only)")
    p.add_argument("--clips", default="",
                   help="file with one clip_id per line (default: all mp4s)")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cam = os.path.join(a.root, "r0", "camera_front_wide")

    if a.contract_test:
        cid, nb, el, err = build_one((a.contract_test, a.root, a.out))
        print(f"CONTRACT {cid}: bytes={nb} {el:.1f}s err={err}", flush=True)
        return 0 if err is None else 1

    if a.clips:
        ids = [ln.strip() for ln in open(a.clips) if ln.strip()]
    else:
        ids = sorted(f[:-4] for f in os.listdir(cam) if f.endswith(".mp4"))

    # ⛔ THE PARITY INGEST GATE — RUN HERE, NOT ASSUMED.
    # `v2_compressed.build` runs this before fetching; this driver calls
    # `build_compressed` directly, so the gate has to be re-asserted or it is
    # simply absent. MEASURED 2026-09-01 on B1: of the 4,719 v7 clips, **6 are
    # inside the 40-episode deployed val** (15 % of the episode set behind every
    # published open-loop number) and 201 are inside the parity train corpus.
    # The instrument's own words: "The corpus is 4713 clips; quote THAT number,
    # never 4719." Building all 4,719 would put val40 episodes into training.
    from tanitad.data import parity
    parity.require_ingest_gate("pod_build_b1_epcache")
    kept, gate = parity.guard_corpus_build(
        ids, label=f"pod_build_b1_epcache -> {a.out}", role="train",
        mode="keep" if a.no_parity_exclude else "exclude")
    if not a.no_parity_exclude:
        dropped = sorted(set(ids) - set(kept))
        ids = sorted(kept)
        print(f"[build] parity gate: kept {len(ids)}, dropped {len(dropped)} "
              f"(inside deployed val40): {dropped}", flush=True)

    # ⛔ RESUME BY CONTENT, NOT BY PRESENCE. A killed build leaves payloads that
    # exist at a plausible size; only one that passes `verify` counts as done.
    from pod_pull_b1_epcache import _verify_one
    have = [os.path.join(a.out, f) for f in os.listdir(a.out)
            if f.endswith(".v2ep.pt")]
    done = set()
    if have:
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for pth, ok, why in ex.map(_verify_one, have, chunksize=4):
                if ok:
                    done.add(os.path.basename(pth)[:-len(".v2ep.pt")])
                else:
                    try:
                        os.unlink(pth)
                    except OSError:
                        pass
        print(f"[build] resume scan: {len(have)} on disk -> "
              f"{len(done)} content-verified, {len(have)-len(done)} discarded "
              f"({time.time()-t0:.0f}s)", flush=True)

    todo = [c for c in ids if c not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"[build] corpus={len(ids)} done={len(done)} todo={len(todo)} "
          f"workers={a.workers} codec={CODEC} "
          f"frame={FRAME_H}x{FRAME_W}@{HFOV_DEG}deg {PROJECTION}", flush=True)

    man = {"corpus": "B1 (Sayood/tanitad-v7-training-corpus)",
           "parity_ingest_gate": gate,
           "clips_total": len(ids), "frame_h": FRAME_H, "frame_w": FRAME_W,
           "hfov_deg": HFOV_DEG, "projection_mode": PROJECTION,
           "codec": CODEC, "n_stack": N_STACK, "root": a.root}
    with open(os.path.join(a.out, "_build_manifest.json"), "w") as fh:
        json.dump(man, fh, indent=1)

    t0, nb, nok = time.time(), 0, 0
    fails = []
    with ProcessPoolExecutor(max_workers=a.workers,
                             max_tasks_per_child=8) as ex:
        futs = [ex.submit(build_one, (c, a.root, a.out)) for c in todo]
        for i, fu in enumerate(as_completed(futs)):
            cid, b, el, err = fu.result()
            if err:
                fails.append((cid, err))
                print(f"[build] FAIL {cid[:8]}: {err}", flush=True)
            else:
                nb += b
                nok += 1
            if (i + 1) % 50 == 0 or (i + 1) == len(futs):
                w = time.time() - t0
                eph = nok / max(w, 1e-9) * 3600
                left = (len(futs) - (i + 1)) / max(eph / 3600, 1e-9)
                print(f"[build] {i+1}/{len(futs)} ok={nok} fail={len(fails)} "
                      f"{nb/1024**3:.1f}GB {eph:.0f}eps/h "
                      f"elapsed={w/60:.1f}min eta={left/60:.1f}min", flush=True)
    w = time.time() - t0
    print(f"[build] DONE ok={nok} fail={len(fails)} {nb/1024**3:.2f}GB "
          f"in {w/60:.1f}min ({nok/max(w,1e-9)*3600:.0f}eps/h)", flush=True)
    if fails:
        with open(os.path.join(a.out, "_failures.json"), "w") as fh:
            json.dump(dict(fails), fh, indent=1)
        for cid, err in fails[:30]:
            print(f"  FAILED {cid}: {err}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
