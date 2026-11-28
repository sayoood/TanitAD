"""Shard-staged driver for the B1 416x1024 rebuild on Thor.

Stage a shard of mp4s from HF -> run the BANKED builder over exactly that shard
-> evict the mp4s -> repeat. Peak transient disk is one shard of source video
(~1.6 GB at shard=120), so the build's disk profile is the OUTPUT plus a
constant, not output + 61.55 GB of source.

The staging pattern is the one already proven over 4,719 clips by
``/home/nvidia/sam3map/eval/corpus_feeder.py``: ``hf_hub_download`` at a PINNED
revision, sha256-checked against that revision's own ``camera_sha256.json``,
placed by atomic rename, and unlinked once consumed.

⛔ RESUME IS BY RECORD **AND** PRESENCE, NOT BY PRESENCE. A clip counts as done
only if it appears in an ARCHIVED shard manifest *and* its payload is on disk.
A payload with no manifest row is an interrupted build whose content checks may
never have run -- it is DELETED and rebuilt, costing at most one shard.

⛔ DISK IS RE-CHECKED BEFORE EVERY SHARD with a real ``statvfs`` on the build
filesystem, and the run REFUSES to start a shard that would take free space
below ``--min-free-gb``. Running out of disk mid-``torch.save`` is how a cache
acquires a truncated payload that still has a plausible size.

Usage::

    TANITAD_STACK=/home/nvidia/TanitAD/stack PYTHONPATH=/home/nvidia/b1build416 \
    python run_b1_416_build.py --pilot 10          # pilot, then STOP
    python run_b1_416_build.py --shard 120         # full run
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")

REPO = "Sayood/tanitad-v7-training-corpus"
REV = "a0cf20dfb4eafa29b0ac1f3c05337f002bc33ca0"
SHATAB = "/home/nvidia/sam3map/corpus/hfmeta/camera/camera_sha256.json"


def sha12(c):
    return hashlib.sha256(c.encode()).hexdigest()[:12]


def sha256f(p, buf=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(buf), b""):
            h.update(b)
    return h.hexdigest()


def free_gb(path):
    s = os.statvfs(path)
    return s.f_bavail * s.f_frsize / 1e9


def log(logp, msg):
    line = time.strftime("%Y-%m-%dT%H:%M:%SZ ", time.gmtime()) + msg
    print(line, flush=True)
    with open(logp, "a") as fh:
        fh.write(line + "\n")


def fetch_one(cid, cam, dl, tab):
    """Download + sha256-verify one mp4; atomic-rename into the cam dir."""
    from huggingface_hub import hf_hub_download
    dst = os.path.join(cam, f"{cid}.mp4")
    want = tab[cid]["sha256"]
    if os.path.exists(dst):
        if sha256f(dst) == want:
            return cid, os.path.getsize(dst), "cached"
        os.unlink(dst)
    p = hf_hub_download(REPO, f"camera/{cid}.mp4", repo_type="dataset",
                        revision=REV, local_dir=dl)
    got = sha256f(p)
    if got != want:
        os.unlink(p)
        return cid, 0, f"SHA_MISMATCH {got[:12]} != {want[:12]}"
    n = os.path.getsize(p)
    os.replace(p, dst)                       # atomic: presence == completeness
    return cid, n, "ok"


def stage_shard(shard, cam, dl, tab, threads):
    """Download + verify every mp4 of one shard. Returns (bytes, failures)."""
    nb, bad = 0, []
    with ThreadPoolExecutor(max_workers=threads) as ex:
        futs = [ex.submit(fetch_one, c, cam, dl, tab) for c in shard]
        for fu in as_completed(futs):
            cid, n, why = fu.result()
            if why.startswith("SHA") or n == 0:
                bad.append((sha12(cid), why))
            nb += n
    return nb, bad


def done_set(out_dir):
    """clip_sha12 recorded in an archived shard manifest AND present on disk."""
    sh = os.path.join(out_dir, "_shards")
    rec = {}
    if os.path.isdir(sh):
        for f in sorted(os.listdir(sh)):
            if not f.endswith(".json"):
                continue
            try:
                m = json.load(open(os.path.join(sh, f)))
            except Exception:                                    # noqa: BLE001
                continue
            for r in m.get("clips", []):
                rec[r["clip_sha12"]] = r
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/home/nvidia/data/_b1stage416")
    p.add_argument("--out", default="/home/nvidia/data/physicalai-b1-w120-416x1024cyl")
    p.add_argument("--builder", default="/home/nvidia/b1build416/build_v2ep_wide.py")
    p.add_argument("--height", type=int, default=416)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--dl-threads", type=int, default=8)
    p.add_argument("--shard", type=int, default=120)
    p.add_argument("--pilot", type=int, default=0, help="build N clips and STOP")
    p.add_argument("--min-free-gb", type=float, default=40.0)
    p.add_argument("--max-shards", type=int, default=0)
    p.add_argument("--max-attempts", type=int, default=2,
                   help="give up on a clip after this many failed builds")
    a = p.parse_args()

    os.makedirs(a.out, exist_ok=True)
    os.makedirs(os.path.join(a.out, "_shards"), exist_ok=True)
    # ⛔ ONE DRIVER AT A TIME. Two would both delete MANIFEST.json between each
    # other's builder runs and interleave shard numbers, losing records for
    # payloads that exist -- which the orphan rule would then delete.
    # ⭐ The lock fd is NOT inherited by the builder: subprocess defaults to
    # close_fds=True, the Python spelling of `200>&-`. Without that, the builder
    # would hold this lock for its whole life and no replacement driver could
    # ever start -- a permanent block that merely looks like a race.
    import fcntl
    _lk = open(os.path.join(a.out, ".driver.lock"), "w")
    try:
        fcntl.flock(_lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("ZZLOCKED-ANOTHER-DRIVER-RUNNINGZZ", flush=True)
        return 4
    _lk.write(f"{os.getpid()}\n")
    _lk.flush()
    cam = os.path.join(a.root, "r0", "camera_front_wide")
    dl = os.path.join(a.root, "dl")
    os.makedirs(dl, exist_ok=True)
    logp = os.path.join(a.out, "driver.log")

    ids = [l.strip() for l in open(os.path.join(a.root, "clips_4713.txt"))
           if l.strip()]
    tab = json.load(open(SHATAB))
    rec = done_set(a.out)

    # Reconcile record against disk (see module docstring).
    orphans = 0
    for c in ids:
        pth = os.path.join(a.out, f"{c}.v2ep.pt")
        has_f, has_r = os.path.exists(pth), sha12(c) in rec
        if has_f and not has_r:
            os.unlink(pth)
            orphans += 1
        elif has_r and not has_f:
            rec.pop(sha12(c), None)
    # ⛔ A DETERMINISTIC PER-CLIP FAILURE MUST NOT BE RETRIED FOREVER. A failed
    # clip stays at the head of todo and its mp4 was evicted, so without a
    # ledger the run re-downloads and re-fails the same clip every shard and
    # never advances -- an infinite loop whose log looks like steady progress.
    gp = os.path.join(a.out, "_attempts.json")
    att = json.load(open(gp)) if os.path.exists(gp) else {}
    gave_up = {s for s, n in att.items() if n >= a.max_attempts}
    todo = [c for c in ids if sha12(c) not in rec and sha12(c) not in gave_up]
    if gave_up:
        log(logp, f"ZZGAVEUP-{len(gave_up)}ZZ {sorted(gave_up)[:10]}")
    log(logp, f"ZZSTART-{len(ids)}-{len(rec)}-{len(todo)}-{orphans}ZZ "
              f"clips={len(ids)} done={len(rec)} todo={len(todo)} "
              f"orphans_removed={orphans} free={free_gb(a.out):.1f}GB")
    if not todo:
        log(logp, "ZZALLDONEZZ")
        return 0

    shard_n = a.pilot if a.pilot else a.shard
    # ⛔ MONOTONIC ACROSS INVOCATIONS, never a per-run counter. A counter that
    # restarts at 1 overwrites the FIRST shard's archived manifest on every
    # resume; its clips then have payloads with no record, the orphan rule
    # deletes them, and the run rebuilds a shard it already paid for -- once per
    # restart, silently, while the log reports progress.
    ex_sh = [f for f in os.listdir(os.path.join(a.out, "_shards"))
             if f.startswith("shard_") and f.endswith(".json")]
    nsh = max((int(f[6:10]) for f in ex_sh), default=0)
    nrun = 0
    n_at_start = len(rec)
    pending = None                       # (tuple(shard), Future) being prefetched
    pre_ex = ThreadPoolExecutor(max_workers=1)
    t_all = time.time()
    while todo:
        if a.max_shards and nrun >= a.max_shards:
            log(logp, f"ZZSTOP-MAXSHARDS-{nrun}ZZ")
            break
        fg = free_gb(a.out)
        if fg < a.min_free_gb:
            log(logp, f"ZZABORT-DISK-{fg:.1f}ZZ free {fg:.1f}GB < "
                      f"min {a.min_free_gb}GB -- refusing to start a shard")
            return 2
        shard = todo[:shard_n]
        nsh += 1
        nrun += 1
        # ---- stage (already running if the previous shard prefetched it) --
        t0 = time.time()
        if pending is not None and pending[0] == tuple(shard):
            nb, bad = pending[1].result()
            pending = None
        else:
            if pending is not None:          # prefetched the wrong set: drop it
                pending[1].result()
                pending = None
            nb, bad = stage_shard(shard, cam, dl, tab, a.dl_threads)
        dt = time.time() - t0
        rt = "PREFETCHED" if dt < 1.0 else f"{nb/1e6/max(dt,1e-9):.1f}MB/s"
        log(logp, f"ZZDL-{nsh}-{len(shard)}-{len(bad)}ZZ staged {len(shard)-len(bad)}"
                  f"/{len(shard)} {nb/1e9:.2f}GB in {dt/60:.1f}min "
                  f"({rt}) bad={len(bad)}")
        for s12, why in bad[:5]:
            log(logp, f"ZZDLBAD-{s12}ZZ {why}")
        staged = [c for c in shard
                  if os.path.exists(os.path.join(cam, f"{c}.mp4"))]
        if not staged:
            log(logp, f"ZZABORT-NOSTAGE-{nsh}ZZ")
            return 3
        # ---- build (banked builder, UNMODIFIED) --------------------------
        sc = os.path.join(a.root, "_shard_clips.txt")
        with open(sc, "w") as fh:
            fh.write("\n".join(staged) + "\n")
        man = os.path.join(a.out, "MANIFEST.json")
        if os.path.exists(man):
            os.unlink(man)          # per-shard manifest; archived after the run
        env = dict(os.environ, TANITAD_STACK="/home/nvidia/TanitAD/stack",
                   OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   PYTHONPATH=os.path.dirname(a.builder))
        cmd = [sys.executable, a.builder, "--out", a.out, "--root", a.root,
               "--clips", sc, "--width", str(a.width), "--height", str(a.height),
               "--workers", str(a.workers), "--role", "train"]
        # ⭐ PREFETCH THE NEXT SHARD WHILE THIS ONE BUILDS. Serially the download
        # is ~21 % of wall with every core idle (MEASURED: 0.6 min stage + 2.3 min
        # build per 36-clip shard). The builder is a SUBPROCESS, so these threads
        # do not contend for its GIL, and the extra transient disk is one more
        # shard of mp4 (~0.5 GB at shard=36). fetch_one is idempotent -- a failed
        # prefetch costs a re-download, never a wrong input.
        nxt = [c for c in todo[shard_n:] if c not in set(shard)][:shard_n]
        if nxt and not a.pilot:
            pending = (tuple(nxt),
                       pre_ex.submit(stage_shard, nxt, cam, dl, tab,
                                     a.dl_threads))
        t1 = time.time()
        bl = os.path.join(a.out, f"_shards/build_{nsh:04d}.log")
        with open(bl, "w") as fh:
            rc = subprocess.call(cmd, env=env, stdout=fh,
                                 stderr=subprocess.STDOUT)
        bdt = time.time() - t1
        # ⛔ the ARTIFACT is the evidence, not rc. A builder that exits 1 on a
        # per-clip failure still produced every other payload.
        rows = []
        if os.path.exists(man):
            m = json.load(open(man))
            rows = m.get("clips", [])
            shutil.copyfile(man, os.path.join(a.out,
                                              f"_shards/shard_{nsh:04d}.json"))
        else:
            log(logp, f"ZZNOMANIFEST-{nsh}-rc{rc}ZZ builder produced no "
                      f"MANIFEST.json -- see {bl}")
        got = sum(r["bytes"] for r in rows)
        log(logp, f"ZZBUILD-{nsh}-{len(rows)}-{len(staged)}ZZ rc={rc} "
                  f"built={len(rows)}/{len(staged)} {got/1e9:.2f}GB in "
                  f"{bdt/60:.1f}min ({bdt/max(len(staged),1):.1f}s/clip wall, "
                  f"{got/max(len(rows),1)/1e6:.1f}MB/ep)")
        # ---- evict -------------------------------------------------------
        ev = 0
        for c in shard:
            pth = os.path.join(cam, f"{c}.mp4")
            if os.path.exists(pth):
                os.unlink(pth)
                ev += 1
        # ⛔ NEVER rmtree the download dir here: a prefetch for the NEXT shard is
        # in flight inside it. fetch_one moves each finished file out with
        # os.replace, so only hub metadata accumulates; it is cleaned at the end.
        rec = done_set(a.out)
        for c in shard:                       # charge an attempt to each miss
            if sha12(c) not in rec:
                att[sha12(c)] = att.get(sha12(c), 0) + 1
        json.dump(att, open(gp, "w"), indent=1)
        gave_up = {s for s, n in att.items() if n >= a.max_attempts}
        todo = [c for c in ids
                if sha12(c) not in rec and sha12(c) not in gave_up]
        el = time.time() - t_all
        nd = len(rec)
        built_this_run = nd - n_at_start
        # rate over THIS RUN only -- clips already done when we started were not
        # paid for now, and charging them to this run's elapsed time is how an
        # ETA silently reports the resume's free progress as throughput.
        rate = built_this_run / max(el, 1e-9)
        eta = len(todo) / rate if rate > 0 else float("nan")
        log(logp, f"ZZPROG-{nd}-{len(ids)}ZZ evicted={ev} done={nd}/{len(ids)} "
                  f"free={free_gb(a.out):.1f}GB elapsed={el/3600:.2f}h "
                  f"rate={rate*3600:.0f}eps/h eta={eta/3600:.2f}h")
        if a.pilot:
            log(logp, f"ZZPILOT-STOP-{nd}ZZ")
            break
    if pending is not None:
        pending[1].result()
    pre_ex.shutdown(wait=True)
    shutil.rmtree(os.path.join(dl, "camera"), ignore_errors=True)
    shutil.rmtree(os.path.join(dl, ".cache"), ignore_errors=True)
    log(logp, f"ZZDRIVER-END-{len(ids)-len(todo)}-{len(ids)}ZZ "
              f"total={(time.time()-t_all)/3600:.2f}h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
