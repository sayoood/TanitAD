"""Input feeder for the corpus SAM3 map production on Thor (PI 2026-09-15: "download what do you need, manage efficiently disk memory").
Keeps the front-wide mp4s of the next FEED_WINDOW clips that are neither OK nor given up in <root>/cam: downloaded from the private HF
corpus at a pinned revision, sha256-checked against that revision's camera/camera_sha256.json, placed by atomic rename (the driver
treats existence as completeness). The mp4 of every clip the ledger records OK or given up is removed from <root>/cam -- a transient
copy; the originals stay on HF and on the dev box. The given-up rule is sam3map_prod.given_up's (FAIL-EXPORT-CHECKS once, any FAIL
PROD_MAX_FAILS times, <out>/skip.txt). A sha mismatch is moved aside, never used; after two mismatches the clip goes to skip.txt.
Exits when <root>/DONE or <root>/STOP exists or nothing is left to feed. One feeder at a time (flock)."""
import collections, fcntl, hashlib, json, os, sys, time
from pathlib import Path
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")
from huggingface_hub import hf_hub_download

ROOT = Path(os.environ.get("CORPUS_ROOT", "/home/nvidia/sam3map/corpus"))
REPO, REV = "Sayood/tanitad-v7-training-corpus", "a0cf20dfb4eafa29b0ac1f3c05337f002bc33ca0"
WINDOW = int(os.environ.get("FEED_WINDOW", "120")); MAX_FAILS = int(os.environ.get("PROD_MAX_FAILS", "2"))
CAM, DL, OUT = ROOT / "cam", ROOT / "dl", ROOT / "out"


def log(msg):
    with open(ROOT / "feeder.log", "a") as fh:
        fh.write(time.strftime("%Y-%m-%dT%H:%M:%S ") + msg + "\n")


def sha12(clip):
    return hashlib.sha256(clip.encode()).hexdigest()[:12]


def ledger_state():
    ok, fails, checks = set(), collections.Counter(), set()
    p = OUT / "manifest.jsonl"
    if p.exists():
        for line in open(p):
            try:
                r = json.loads(line)
            except ValueError:                                                 # a row being appended
                continue
            s = str(r.get("status", ""))
            if s == "OK":
                ok.add(r["clip_sha12"])
            elif s.startswith("FAIL"):
                fails[r["clip_sha12"]] += 1
                if s == "FAIL-EXPORT-CHECKS":
                    checks.add(r["clip_sha12"])
    gone = set((OUT / "skip.txt").read_text().split()) if (OUT / "skip.txt").exists() else set()
    if MAX_FAILS:
        gone |= checks | {s for s, n in fails.items() if n >= MAX_FAILS}
    return ok | gone


def main():
    lock = open(ROOT / "feeder.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("another feeder holds the lock"); sys.exit(1)
    clips = [l.strip() for l in open(ROOT / "production_order.txt") if l.strip()]
    tab = json.load(open(ROOT / "hfmeta" / "camera" / "camera_sha256.json"))
    CAM.mkdir(exist_ok=True); DL.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
    st = {"pid": os.getpid(), "downloaded": 0, "bytes": 0, "evicted": 0, "sha_mismatch": 0, "errors": 0, "window": WINDOW}
    bad = collections.Counter(); err_streak = 0
    log(f"feeder start pid {os.getpid()} window {WINDOW} revision {REV[:12]}")
    while True:
        if (ROOT / "DONE").exists() or (ROOT / "STOP").exists():
            log("DONE/STOP present - exiting"); break
        finished = ledger_state()
        for f in CAM.glob("*.mp4"):
            if sha12(f.stem) in finished:
                f.unlink(); st["evicted"] += 1
        window = [c for c in clips if sha12(c) not in finished][:WINDOW]
        if not window:
            log("nothing left to feed - exiting"); break
        for c in [c for c in window if not (CAM / f"{c}.mp4").exists()]:
            if (ROOT / "STOP").exists():
                break
            try:
                a = time.time()
                p = Path(hf_hub_download(REPO, f"camera/{c}.mp4", repo_type="dataset", revision=REV, local_dir=str(DL)))
                h = hashlib.sha256(p.read_bytes()).hexdigest()
                for meta in (DL / ".cache" / "huggingface" / "download" / "camera").glob(f"{c}.mp4.*"):
                    meta.unlink()
                if h != tab[c]["sha256"] or p.stat().st_size != int(tab[c]["bytes"]):     # values: {sha256, bytes, duration_s, ...}
                    st["sha_mismatch"] += 1; bad[c] += 1
                    p.rename(DL / f"mismatch_{sha12(c)}_{int(time.time())}.mp4"); log(f"SHA-MISMATCH {sha12(c)} (#{bad[c]})")
                    if bad[c] >= 2:
                        with open(OUT / "skip.txt", "a") as fh:
                            fh.write(sha12(c) + "\n")
                        log(f"SKIP {sha12(c)}: two sha mismatches")
                    continue
                size = p.stat().st_size; os.replace(p, CAM / f"{c}.mp4")
                st["downloaded"] += 1; st["bytes"] += size; err_streak = 0
                log(f"got {sha12(c)} {size / 1e6:.1f} MB {time.time() - a:.1f}s")
            except Exception as e:
                st["errors"] += 1; err_streak += 1
                log(f"ERROR {sha12(c)} {type(e).__name__}: {str(e)[:200]} (streak {err_streak})")
                time.sleep(min(900, 30 * 2 ** min(err_streak, 5)))
                break
        st.update(t=time.strftime("%Y-%m-%dT%H:%M:%S"), in_cam=len(list(CAM.glob("*.mp4"))), finished=len(finished), total=len(clips))
        (ROOT / "feeder_stats.json.tmp").write_text(json.dumps(st)); os.replace(ROOT / "feeder_stats.json.tmp", ROOT / "feeder_stats.json")
        time.sleep(30)


if __name__ == "__main__":
    main()
