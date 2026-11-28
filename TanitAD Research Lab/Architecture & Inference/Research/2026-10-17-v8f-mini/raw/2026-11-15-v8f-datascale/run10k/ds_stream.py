#!/usr/bin/env python3
"""DATASCALE-10K streaming driver (pod). stage -> build -> encode -> bank, one WINDOW (= HF shard) at a time, 4 threads joined by bounded queues.
Resumable from per-window markers (/root/datascale/state10k) and ledgers. Reuses ds_stage.py / ds_build.py / cache_pod.py UNEDITED through the
ds_*_w.py wrappers. Order = first 10,000 clips of selection/order_all_eligible.parquet (frozen scope, seed 20261115).
  python ds_stream.py [--smoke] [--log PATH]
No address, port or pod id in this file; the HF token is read in place from /root/.hf_token and never printed.
"""
import argparse
import gzip
import hashlib
import json
import os
import queue
import shutil
import subprocess
import sys
import tarfile
import threading
import time

os.environ["HF_HUB_DISABLE_XET"] = "1"
B = "/root/datascale"
W = f"{B}/w"
ST = f"{B}/state10k"
KEEP = f"{B}/keep10k"
LED10K = f"{B}/ledgers10k"
PY = "/root/venv/bin/python"
REPO = "Sayood/tanitad-datascale-10k"
N_TOTAL = 10000
N_PILOT = 200
N_SMOKE = 20
SHARD = 250
MIN_FREE_GB = 60.0
ENC_TIMEOUT_S = 75 * 60
MANIFEST = f"{B}/corpus_manifest_10k.jsonl"
BANKED = f"{B}/banked10k.jsonl"
DELLOG = f"{B}/deletions10k.log"
OFFLINE = {"HF_HUB_CACHE": "/root/identity_pod/hf", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}

LOG = f"{B}/run10k.log"
abort = threading.Event()
abort_why = []
LK = threading.Lock()
T0 = time.time()
CNT = {"staged": 0, "built": 0, "encoded": 0, "banked": 0, "failed": 0}   # this-launch increments
BASE = {"staged": 0, "built": 0, "encoded": 0, "banked": 0, "failed": 0}   # from the banked ledger at start
ENC_T = []                                                                  # (time, n) encode events this launch


def say(m):
    with LK:
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(f"[ds10k {time.strftime('%F %T')}] {m}\n")


def die(why):
    if not abort.is_set():
        abort_why.append(why)
        abort.set()
    say(f"ABORT {why}")


def sha256_file(p, algo="sha256"):
    h = hashlib.new(algo)
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def jl(p):
    out = []
    if os.path.exists(p):
        for ln in open(p, encoding="utf-8"):
            ln = ln.strip()
            if ln:
                try:
                    out.append(json.loads(ln))
                except Exception:                                      # noqa: BLE001
                    pass
    return out


def last_by_clip(p):
    d = {}
    for r in jl(p):
        d[r["clip"]] = r
    return d


def free_gb():
    return shutil.disk_usage("/root").free / 1e9


def delete_tree(path, why):
    """Delete by literal absolute path under /root/datascale/, log (path, bytes, files)."""
    assert path.startswith(B + "/") and ".." not in path, path
    if not os.path.exists(path):
        return
    n = sz = 0
    if os.path.isdir(path):
        for r, _, fs in os.walk(path):
            for f in fs:
                try:
                    sz += os.path.getsize(os.path.join(r, f))
                    n += 1
                except OSError:
                    pass
        shutil.rmtree(path)
    else:
        sz = os.path.getsize(path)
        n = 1
        os.remove(path)
    with LK:
        with open(DELLOG, "a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%F %T')} {path} {sz} bytes {n} files ({why})\n")


def delete_files(paths, why, label):
    n = sz = 0
    for p in paths:
        assert p.startswith(B + "/") and ".." not in p, p
        if os.path.isfile(p):
            sz += os.path.getsize(p)
            n += 1
            os.remove(p)
    if n:
        with LK:
            with open(DELLOG, "a", encoding="utf-8") as fh:
                fh.write(f"{time.strftime('%F %T')} {label} {sz} bytes {n} files ({why})\n")


def run(cmd, out_path, env_extra=None, timeout=None, cwd=B):
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    with open(out_path, "a", encoding="utf-8") as fh:
        p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env, cwd=cwd, start_new_session=True)
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, 15)                                   # our own child's group, by PID
                time.sleep(5)
                os.killpg(p.pid, 9)
            except ProcessLookupError:
                pass
            rc = "TIMEOUT"
    return rc


def marker(name, phase):
    return f"{ST}/{name}.{phase}.done"


def mdone(name, phase):
    p = marker(name, phase)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def mwrite(name, phase, d):
    with open(marker(name, phase) + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(d, fh)
    os.replace(marker(name, phase) + ".tmp", marker(name, phase))


class Shard:
    def __init__(self, name, clips, pre=False):
        self.name, self.clips, self.pre = name, clips, pre
        self.wd = f"{W}/{name}"


def make_shards(smoke):
    order = f"{B}/selection/clips_10k.txt"
    if not os.path.exists(order):
        import pandas as pd
        o = pd.read_parquet(f"{B}/selection/order_all_eligible.parquet").sort_values("order")
        clips = list(o.index[:N_TOTAL])
        pilot = [ln.strip() for ln in open(f"{B}/selection/pilot_clips.txt", encoding="utf-8") if ln.strip()]
        assert clips[:N_PILOT] == pilot, "pilot is not the prefix of the order"
        assert len(set(clips)) == N_TOTAL
        open(order, "w", encoding="utf-8").write("\n".join(clips) + "\n")
    clips = [ln.strip() for ln in open(order, encoding="utf-8") if ln.strip()]
    assert len(clips) == N_TOTAL
    sh = [Shard("s000_pilot200", clips[:N_PILOT], pre=True), Shard("s001_smoke20", clips[N_PILOT:N_PILOT + N_SMOKE])]
    if smoke:
        return sh[1:]
    pos, i = N_PILOT + N_SMOKE, 2
    while pos < N_TOTAL:
        sh.append(Shard(f"s{i:03d}", clips[pos:pos + SHARD]))
        pos += SHARD
        i += 1
    return sh


# ---------------------------------------------------------------- phases
def phase_stage(s):
    if s.pre or mdone(s.name, "stage"):
        return
    while free_gb() < MIN_FREE_GB and not abort.is_set():
        say(f"PAUSE staging {s.name}: free {free_gb():.0f} GB < {MIN_FREE_GB:.0f}")
        time.sleep(60)
    os.makedirs(s.wd, exist_ok=True)
    open(f"{s.wd}/clips.txt", "w", encoding="utf-8").write("\n".join(s.clips) + "\n")
    ok = []
    for attempt in range(3):
        rc = run([PY, f"{B}/ds_stage_w.py", s.wd, "8"], f"{s.wd}/stage.out")
        led = last_by_clip(f"{s.wd}/ledger_stage.jsonl")
        ok = [c for c in s.clips if led.get(c, {}).get("status") == "OK"]
        if rc == 0 and len(ok) == len(s.clips):
            break
        say(f"{s.name} stage pass {attempt + 1} rc={rc} ok={len(ok)}/{len(s.clips)}")
        time.sleep(10)
    if not ok:
        die(f"{s.name}: stage produced 0 clips")
        return
    open(f"{s.wd}/okclips.txt", "w", encoding="utf-8").write("\n".join(ok) + "\n")
    fails = [{"clip": c, "phase": "stage", "err": last_by_clip(f"{s.wd}/ledger_stage.jsonl").get(c, {}).get("err", "no ledger row")} for c in s.clips if c not in set(ok)]
    for f in fails:
        with open(f"{B}/fails10k.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(f) + "\n")
    mwrite(s.name, "stage", {"ok": len(ok), "fail": len(fails)})
    with LK:
        CNT["staged"] += len(ok)
        CNT["failed"] += len(fails)


def phase_build(s):
    if s.pre or mdone(s.name, "build"):
        return
    n_ok = len(open(f"{s.wd}/okclips.txt", encoding="utf-8").read().split())
    rc = run([PY, f"{B}/ds_build_w.py", s.wd, "13"], f"{s.wd}/build.out", env_extra=OFFLINE)
    led = last_by_clip(f"{s.wd}/ledger_build.jsonl")
    if rc != 0 or "ZZ V2EP DONE ZZ" not in open(f"{s.wd}/build.log", encoding="utf-8").read():
        die(f"{s.name}: build rc={rc} / no DONE marker")
        return
    okc = set(open(f"{s.wd}/okclips.txt", encoding="utf-8").read().split())
    for c in okc - set(led):
        with open(f"{B}/fails10k.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"clip": c, "phase": "build", "err": "see ledger_build.jsonl.fail"}) + "\n")
    # v2ep verified by build_one's content check + ledger sha256: the mp4 + timestamps are no longer needed (re-creatable from NVIDIA)
    delete_tree(f"{s.wd}/stage/r0", "mp4 + timestamps; v2ep built and content-verified (ledger_build.jsonl)")
    mwrite(s.name, "build", {"ok": len(led), "fail": n_ok - len(led)})
    with LK:
        CNT["built"] += len(led)
        CNT["failed"] += n_ok - len(led)


def phase_encode(s):
    if s.pre or mdone(s.name, "encode"):
        return
    n_b = len(last_by_clip(f"{s.wd}/ledger_build.jsonl"))
    t0 = time.time()
    rc = 0
    if n_b:
        rc = run([PY, f"{B}/ds_encode_w.py", s.wd], f"{s.wd}/encode.out", env_extra=OFFLINE, timeout=ENC_TIMEOUT_S)
    man = last_by_clip(f"{s.wd}/cache/train/manifest.jsonl")
    led = last_by_clip(f"{s.wd}/ledger_build.jsonl")
    miss = [c for c in led if c not in man]
    for c in miss:
        with open(f"{B}/fails10k.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"clip": c, "phase": "encode", "err": f"not encoded, encode rc={rc}"}) + "\n")
    if n_b and not man:
        die(f"{s.name}: encode produced 0 clips rc={rc}")
        return
    mwrite(s.name, "encode", {"ok": len(man), "fail": len(miss), "rc": str(rc), "secs": round(time.time() - t0)})
    with LK:
        CNT["encoded"] += len(man)
        CNT["failed"] += len(miss)
        ENC_T.append((time.time(), len(man)))


def hf_api():
    from huggingface_hub import HfApi
    return HfApi(token=open("/root/.hf_token", encoding="utf-8").read().strip())


def upload_verified(api, local, repo_path):
    """Upload then require hub LFS sha256 == local sha256. Returns the sha256."""
    want = sha256_file(local)
    for attempt in range(4):
        try:
            api.upload_file(path_or_fileobj=local, path_in_repo=repo_path, repo_id=REPO, repo_type="dataset")
            info = api.get_paths_info(REPO, [repo_path], repo_type="dataset")
            got = info[0].lfs.sha256 if info and info[0].lfs else None
            if got == want:
                return want
            say(f"UPLOAD sha mismatch {repo_path} attempt {attempt + 1}: hub {str(got)[:12]} local {want[:12]}")
        except Exception as e:                                         # noqa: BLE001
            say(f"UPLOAD error {repo_path} attempt {attempt + 1}: {type(e).__name__}: {str(e)[:200]}")
        time.sleep(15 * (attempt + 1))
    raise RuntimeError(f"upload of {repo_path} never verified")


def phase_bank(s, api):
    if any(r["shard"] == s.name for r in jl(BANKED)):
        return
    t0 = time.time()
    if s.pre:
        cache_root = f"{B}/cache_pilot"
        stg = f"{B}/stage"
        led_stage = last_by_clip(f"{B}/ledger_stage.jsonl")
        led_build = last_by_clip(f"{B}/ledger_pilot.jsonl")
        v2ep_dir = None
    else:
        cache_root = f"{s.wd}/cache"
        stg = f"{s.wd}/stage"
        led_stage = last_by_clip(f"{s.wd}/ledger_stage.jsonl")
        led_build = last_by_clip(f"{s.wd}/ledger_build.jsonl")
        v2ep_dir = f"{s.wd}/v2ep/train"
    man = last_by_clip(f"{cache_root}/train/manifest.jsonl")
    clips = [c for c in s.clips if c in man]
    # 1. verify every cache file against its manifest row (size + md5) and take its sha256
    rows, bad = [], []
    for c in clips:
        p = f"{cache_root}/train/clips/{c}.pt"
        e = man[c]
        if not os.path.exists(p) or os.path.getsize(p) != e["bytes"] or sha256_file(p, "md5") != e["md5"]:
            bad.append(c)
            continue
        rows.append({"clip": c, "cache_sha256": sha256_file(p), "cache_bytes": e["bytes"], "content_md5": e["content_md5"], "n_frames": e["n_frames"]})
    for c in bad:
        with open(f"{B}/fails10k.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"clip": c, "phase": "verify", "err": "cache file failed size/md5 vs manifest"}) + "\n")
    good = [r["clip"] for r in rows]
    gset = set(good)
    # 2. obstacle join over the window's v2ep, before they are deleted (needs poses from the v2ep; pilot already has join_pilot.jsonl)
    kd = f"{KEEP}/{s.name}"
    os.makedirs(kd, exist_ok=True)
    join_gz = f"{kd}/join.jsonl.gz"
    if s.pre:
        if os.path.exists(f"{B}/join_pilot.jsonl") and not os.path.exists(join_gz):
            with open(f"{B}/join_pilot.jsonl", "rb") as fi, gzip.open(join_gz, "wb", 3) as fo:
                shutil.copyfileobj(fi, fo)
    elif good and not os.path.exists(join_gz):
        os.makedirs(f"{stg}/obsroot/labels/egomotion", exist_ok=True)
        lnk = f"{stg}/obsroot/labels/egomotion/egomotion.chunk_0000.zip"
        if not os.path.exists(lnk):
            os.symlink(f"{stg}/labels/egomotion/egomotion_all.zip", lnk)
        nv = len([f for f in os.listdir(v2ep_dir) if f.endswith(".v2ep.pt")])
        rcj = run([PY, "/root/tad/stack/scripts/build_obstacle_join.py", "--corpus", v2ep_dir, "--episodes", str(nv), "--out", f"{s.wd}/join.jsonl",
                   "--hf-cache", f"{s.wd}/hfc", "--labels-root", f"{stg}/obsroot", "--no-download"], f"{s.wd}/join.out", env_extra=OFFLINE, timeout=1800)
        if rcj == 0 and os.path.exists(f"{s.wd}/join.jsonl"):
            with open(f"{s.wd}/join.jsonl", "rb") as fi, gzip.open(join_gz, "wb", 3) as fo:
                shutil.copyfileobj(fi, fo)
            if os.path.exists(f"{s.wd}/join.jsonl.meta.json"):
                shutil.copy(f"{s.wd}/join.jsonl.meta.json", f"{kd}/join.jsonl.meta.json")
        else:
            say(f"WARN {s.name} obstacle join rc={rcj} (see {s.wd}/join.out)")
    # 3. the verified cache files exist -> the v2ep of those clips is no longer needed
    if v2ep_dir:
        delete_files([f"{v2ep_dir}/{c}.v2ep.pt" for c in good], "cache file verified (size+md5 vs manifest)", f"{s.name} v2ep")
    # 4. kept members (egomotion + obstacle parquet per clip) + ledgers -> local keep dir (moved, not copied) and a members tar
    for sub, pat in (("ego_members", "{c}.egomotion.parquet"), ("obs_members", "{c}.obstacle.offline.parquet")):
        os.makedirs(f"{kd}/{sub}", exist_ok=True)
        for c in good:
            src = f"{stg}/{sub}/{pat.format(c=c)}"
            if os.path.exists(src):
                shutil.copy(src, f"{kd}/{sub}/{pat.format(c=c)}")
    with open(f"{kd}/ledger_stage.jsonl", "w", encoding="utf-8") as fh:
        for c in good:
            if c in led_stage:
                fh.write(json.dumps(led_stage[c]) + "\n")
    with open(f"{kd}/ledger_build.jsonl", "w", encoding="utf-8") as fh:
        for c in good:
            if c in led_build:
                fh.write(json.dumps(led_build[c]) + "\n")
    mtar = f"{s.wd}/{s.name}.members.tar"
    os.makedirs(s.wd, exist_ok=True)
    with tarfile.open(mtar, "w") as tf:
        for r, _, fs in os.walk(kd):
            for f in sorted(fs):
                full = os.path.join(r, f)
                tf.add(full, arcname=os.path.relpath(full, kd))
    # 5. cache tar: train/clips/*.pt + manifest rows of the banked clips + provenance + meta
    ctar = f"{s.wd}/{s.name}.cache.tar"
    prov = last_by_clip(f"{cache_root}/train/pod_provenance.jsonl")
    with open(f"{s.wd}/manifest_rows.jsonl", "w", encoding="utf-8") as fh:
        for c in good:
            fh.write(json.dumps(man[c]) + "\n")
    with open(f"{s.wd}/prov_rows.jsonl", "w", encoding="utf-8") as fh:
        for c in good:
            if c in prov:
                fh.write(json.dumps(prov[c]) + "\n")
    with tarfile.open(ctar, "w") as tf:
        tf.add(f"{s.wd}/manifest_rows.jsonl", arcname="train/manifest.jsonl")
        tf.add(f"{s.wd}/prov_rows.jsonl", arcname="train/pod_provenance.jsonl")
        if os.path.exists(f"{cache_root}/train/meta.json"):
            tf.add(f"{cache_root}/train/meta.json", arcname="train/meta.json")
        for c in good:
            tf.add(f"{cache_root}/train/clips/{c}.pt", arcname=f"train/clips/{c}.pt")
    # 6. upload + hub LFS sha256 check (a shard is banked only when they are equal)
    sc = upload_verified(api, ctar, f"cache/{s.name}.cache.tar")
    sm = upload_verified(api, mtar, f"members/{s.name}.members.tar")
    # 7. record, then delete the local tars and clip files (banked and verified)
    with open(MANIFEST, "a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(dict(r, shard=s.name, shard_tar=f"cache/{s.name}.cache.tar", shard_sha256=sc, member_in_tar=f"train/clips/{r['clip']}.pt",
                                     members_tar=f"members/{s.name}.members.tar", members_sha256=sm)) + "\n")
    ld = f"{LED10K}/{s.name}"
    os.makedirs(ld, exist_ok=True)
    for f in ("ledger_stage.jsonl", "ledger_build.jsonl", "ledger_build.jsonl.fail", "build.log", "stage.out", "encode.out", "join.out", "okclips.txt", "clips.txt"):
        if os.path.exists(f"{s.wd}/{f}"):
            shutil.copy(f"{s.wd}/{f}", f"{ld}/{f}")
    rec = {"shard": s.name, "n_clips": len(good), "n_verify_bad": len(bad), "cache_tar_sha256": sc, "members_tar_sha256": sm,
           "cache_tar_bytes": os.path.getsize(ctar), "banked": time.strftime("%F %T"), "secs": round(time.time() - t0)}
    delete_files([ctar, mtar], "tar banked, hub LFS sha256 == local", f"{s.name} tars")
    if s.pre:
        # the pilot cache stays (kept pilot output per HANDOFF_DATASCALE.md; deletable by the MM now that it is banked); pilot staging temp is not needed any more
        for sub in ("r0", "obsroot", "labels", "obs_members", "ego_members"):
            delete_tree(f"{stg}/{sub}", "pilot staging temp; members banked + kept in keep10k")
    else:
        delete_tree(s.wd, "window banked: stage/v2ep/cache temp (members kept in keep10k, ledgers in ledgers10k)")
    with open(BANKED, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    with LK:
        CNT["banked"] += len(good)
    say(f"BANKED {s.name}: {len(good)} clips, cache tar {rec['cache_tar_bytes'] / 1e9:.2f} GB sha {sc[:12]}, {rec['secs']} s")


# ---------------------------------------------------------------- threads
def worker(fn, qin, qout, label):
    try:
        while not abort.is_set():
            try:
                s = qin.get(timeout=5) if qin is not None else None
            except queue.Empty:
                continue
            if s is None:
                break
            fn(s)
            if abort.is_set():
                break
            if qout is not None:
                while not abort.is_set():
                    try:
                        qout.put(s, timeout=5)
                        break
                    except queue.Full:
                        pass
    except Exception as e:                                             # noqa: BLE001
        import traceback
        say(f"EXC in {label}: {traceback.format_exc()[-800:]}")
        die(f"{label}: {type(e).__name__}: {str(e)[:200]}")
    finally:
        if qout is not None and not abort.is_set():
            qout.put(None)


def feeder(shards, q):
    for s in shards:
        while not abort.is_set():
            try:
                q.put(s, timeout=5)
                break
            except queue.Full:
                pass
    q.put(None)


def live_counts(shards):
    """base (banked) + live in-flight window counts."""
    c = dict(BASE)
    for s in shards:
        if any(r["shard"] == s.name for r in jl(BANKED)):
            continue
        if os.path.isdir(s.wd):
            c["staged"] += sum(1 for r in last_by_clip(f"{s.wd}/ledger_stage.jsonl").values() if r.get("status") == "OK")
            c["built"] += len(last_by_clip(f"{s.wd}/ledger_build.jsonl"))
            c["encoded"] += len(last_by_clip(f"{s.wd}/cache/train/manifest.jsonl"))
    return c


def progress(shards, stop):
    last = 0
    while not stop.is_set():
        if time.time() - last >= 600:
            last = time.time()
            c = live_counts(shards)
            banked = sum(r["n_clips"] for r in jl(BANKED))
            nb = len(jl(BANKED))
            rate = ""
            now = time.time()
            ev = [(t, n) for t, n in ENC_T if now - t < 3600]
            if ev:
                rate = f" enc_clips_per_h_last_h={sum(n for _, n in ev) / max(now - min(t for t, _ in ev) + 1, 1) * 3600:.0f}"
            say(f"PROGRESS staged={c['staged']} built={c['built']} encoded={c['encoded']} banked={banked} shards_banked={nb}/{len(shards)} "
                f"failed={sum(1 for _ in jl(f'{B}/fails10k.jsonl'))} free_disk_gb={free_gb():.0f} elapsed_h={(now - T0) / 3600:.2f}{rate}")
        stop.wait(10)


def main():
    global LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--log", default=f"{B}/run10k.log")
    a = ap.parse_args()
    LOG = a.log
    for d in (W, ST, KEEP, LED10K):
        os.makedirs(d, exist_ok=True)
    say(f"start pid={os.getpid()} smoke={a.smoke} scope_sha={hashlib.sha256(open(f'{B}/SCOPE_DATASCALE.md', 'rb').read()).hexdigest()[:16]} free={free_gb():.0f} GB")
    # real write probe (the traps entry: never trust df alone)
    pr = f"{B}/.ddprobe10k"
    rc = subprocess.run(["dd", "if=/dev/zero", f"of={pr}", "bs=1M", "count=512", "oflag=direct"], capture_output=True, encoding="utf-8").returncode
    delete_files([pr], "own write probe", "ddprobe")
    if rc != 0:
        die("dd write probe failed")
        return 1
    shards = make_shards(a.smoke)
    say(f"order sha256 {sha256_file(f'{B}/selection/clips_10k.txt')[:16]}; {len(shards)} shards, {sum(len(s.clips) for s in shards)} clips")
    api = hf_api()
    api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
    info = api.dataset_info(REPO)
    if not info.private:
        die(f"{REPO} is not private")
        return 1
    say(f"HF repo {REPO} private={info.private}")
    for r in jl(BANKED):
        BASE["banked"] += r["n_clips"]
    todo = [s for s in shards if not any(r["shard"] == s.name for r in jl(BANKED))]
    say(f"{len(todo)} shards to do, {len(shards) - len(todo)} already banked")
    q0, q1, q2, q3 = (queue.Queue(maxsize=1) for _ in range(4))
    stop = threading.Event()
    ts = [threading.Thread(target=feeder, args=(todo, q0), daemon=True),
          threading.Thread(target=worker, args=(phase_stage, q0, q1, "stage"), daemon=True),
          threading.Thread(target=worker, args=(phase_build, q1, q2, "build"), daemon=True),
          threading.Thread(target=worker, args=(phase_encode, q2, q3, "encode"), daemon=True),
          threading.Thread(target=worker, args=(lambda s: phase_bank(s, api), q3, None, "bank"), daemon=True),
          threading.Thread(target=progress, args=(shards, stop), daemon=True)]
    for t in ts:
        t.start()
    for t in ts[:5]:
        t.join()
    stop.set()
    if abort.is_set():
        say(f"ZZ DATASCALE-10K FAIL ZZ {'; '.join(abort_why)}")
        return 1
    nb = jl(BANKED)
    exp = sum(len(s.clips) for s in shards)
    got = sum(r["n_clips"] for r in nb if r["shard"] in {s.name for s in shards})
    nf = len(jl(f"{B}/fails10k.jsonl"))
    if not a.smoke:
        try:
            for f, rp in ((MANIFEST, "manifest/corpus_manifest_10k.jsonl"), (f"{B}/selection/clips_10k.txt", "manifest/clips_10k.txt"),
                          (f"{B}/SCOPE_DATASCALE.md", "manifest/SCOPE_DATASCALE.md"), (BANKED, "manifest/banked10k.jsonl"), (f"{B}/fails10k.jsonl", "manifest/fails10k.jsonl")):
                if os.path.exists(f):
                    api.upload_file(path_or_fileobj=f, path_in_repo=rp, repo_id=REPO, repo_type="dataset")
        except Exception as e:                                         # noqa: BLE001
            say(f"WARN manifest upload failed: {type(e).__name__}: {str(e)[:200]} (the files stay in {B})")
    if got + nf >= exp:
        say(f"ZZ DATASCALE-10K DONE ZZ banked={got} failed={nf} expected={exp} repo={REPO}")
        return 0
    say(f"ZZ DATASCALE-10K FAIL ZZ banked {got} + failed {nf} < expected {exp}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
