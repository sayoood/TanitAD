"""GEO10K step 1: fetch the camera timestamps parquet of every 10k clip (and ego + timestamps of the 20 control clips).
Camera zip: one directory read + one small range read per member; no mp4 is read. Token read in place, never printed."""
import glob, io, json, os, sys, time, zipfile, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
os.environ["HF_HUB_DISABLE_XET"] = "1"
import pandas as pd

R = "nvidia/PhysicalAI-Autonomous-Vehicles"
CAMP = "camera/camera_front_wide_120fov/camera_front_wide_120fov.chunk_{c:04d}.zip"
EGOP = "labels/egomotion/egomotion.chunk_{c:04d}.zip"
OUT = "/root/geo10k/inputs"
NT = int(os.environ.get("G1_THREADS", "16"))


def fs():
    from huggingface_hub import HfFileSystem
    tok = open(glob.glob("/root/.hf_*")[0]).read().strip()
    return HfFileSystem(token=tok)


def do_chunk(chunk, clips, want_ego):
    last = None
    for _ in range(4):
        try:
            F = fs()
            got = []
            with F.open(f"datasets/{R}/" + CAMP.format(c=chunk), "rb") as f:
                z = zipfile.ZipFile(f)
                nm = set(z.namelist())
                for c in clips:
                    tsn = f"{c}.camera_front_wide_120fov.timestamps.parquet"
                    if tsn not in nm:
                        raise RuntimeError(f"no timestamps member for {c}")
                    open(f"{OUT}/ts/{c}.timestamps.parquet", "wb").write(z.read(tsn))
                    got.append(c)
            if want_ego:
                with F.open(f"datasets/{R}/" + EGOP.format(c=chunk), "rb") as f:
                    z = zipfile.ZipFile(f)
                    for c in clips:
                        m = [n for n in z.namelist() if n.endswith(f"{c}.egomotion.parquet")]
                        if len(m) != 1:
                            raise RuntimeError(f"{len(m)} ego members for {c}")
                        open(f"{OUT}/ctl_ego/{c}.egomotion.parquet", "wb").write(z.read(m[0]))
            return chunk, got, None
        except Exception as e:                                   # noqa: BLE001
            last = f"{type(e).__name__}: {str(e)[:150]}"
            time.sleep(3)
    return chunk, [], last


def main():
    os.makedirs(f"{OUT}/ts", exist_ok=True)
    os.makedirs(f"{OUT}/ctl_ego", exist_ok=True)
    chunk_of = {}
    for f in glob.glob("/root/datascale/keep10k/*/ledger_stage.jsonl"):
        for l in open(f):
            r = json.loads(l)
            if r.get("status") == "OK":
                chunk_of[r["clip"]] = int(r["chunk"])
    sel = [l.strip() for l in open("/root/datascale/selection/clips_10k.txt") if l.strip()]
    jobs = {}
    for c in sel:
        jobs.setdefault(chunk_of[c], []).append((c, False))
    # controls: 20 EXISTING-corpus clips (cache_pod train manifest), every 228th in manifest order
    ex = [json.loads(l)["clip"] for l in open("/root/data/cache_pod/train/manifest.jsonl")]
    ctl = [ex[i * 228] for i in range(20)]
    open("/root/geo10k/inputs/control_clips20.txt", "w").write("\n".join(ctl) + "\n")
    ci = pd.read_parquet("/root/datascale/meta/clip_index.parquet")
    for c in ctl:
        jobs.setdefault(int(ci.loc[c, "chunk"]), []).append((c, True))
    print(f"[g1] chunks {len(jobs)} clips {sum(len(v) for v in jobs.values())}", flush=True)
    t0, done, fails = time.time(), 0, []
    with ThreadPoolExecutor(NT) as ex_:
        futs = []
        for ch, cl in jobs.items():
            ctl_in = [c for c, w in cl if w]
            rest = [c for c, w in cl if not w]
            if rest:
                futs.append(ex_.submit(do_chunk, ch, rest, False))
            if ctl_in:
                futs.append(ex_.submit(do_chunk, ch, ctl_in, True))
        for fu in as_completed(futs):
            ch, got, err = fu.result()
            done += len(got)
            if err:
                fails.append((ch, err))
            if len(futs) and (done % 500 < 40):
                print(f"[g1] {done} clips {time.time()-t0:.0f}s fails={len(fails)}", flush=True)
    n = len(glob.glob(f"{OUT}/ts/*.parquet"))
    print(f"[g1] ts files {n} fails {fails[:3]}")
    print("ZZ GEO10K-G1 DONE ZZ" if n == 10020 and not fails else f"ZZ GEO10K FAIL ZZ g1 n={n} fails={len(fails)}")


main()
