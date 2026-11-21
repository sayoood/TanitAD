"""Download the renderer members of the selected scenes (SCREEN_TABLE.json) by HTTP Range out of the STORED usdz. Resumable and detached.

    python -I download.py            runs/resumes; writes raw/DOWNLOAD_MANIFEST.json, raw/DOWNLOAD_STATUS.json and
                                     raw/T2_SCENES.done on success or raw/T2_SCENES.fail on any failure
    python -I download.py --plan-only   writes raw/DOWNLOAD_PLAN.json and stops

Placement: dev box F: first, then C:. Never leaves a drive below 10 GB free (hard floor; the plan keeps 11 GB). The token is read in
place by hfc.py and never printed or written. Verified = size and CRC32 equal the usdz central directory; the sha256 is computed and recorded
(the release publishes a sha256 for the whole usdz only, not for members).
"""
import sys, os, json, time, shutil, hashlib, zlib, struct, threading, traceback
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
T = "E:/Projects/TanitAD-artifacts/dagland/motion_c/t2_scenes/"
sys.path.insert(0, T + "code")
from concurrent.futures import ThreadPoolExecutor
import hfc, t2lib

GB = 1 << 30
HARD_FLOOR, PLAN_FLOOR = 10 * GB, 20 * GB
ROOTS = [("F", "F:/TanitAD-scratch/t2_scenes"), ("C", "C:/Users/Admin/TanitAD-scratch/t2_scenes")]
MEMBERS = ("volume.nurec", "rig_trajectories.json", "sequence_tracks.json")
SUBDIR = {"volume.nurec": "", "rig_trajectories.json": "", "sequence_tracks.json": "extracted/"}
WORKERS, CHUNK = 3, 8 << 20
PLAN, MAN, STATUS = T + "raw/DOWNLOAD_PLAN.json", T + "raw/DOWNLOAD_MANIFEST.json", T + "raw/DOWNLOAD_STATUS.json"
DONE, FAIL = T + "raw/T2_SCENES.done", T + "raw/T2_SCENES.fail"
lock = threading.Lock()
inflight = {}          # drive -> bytes reserved by running downloads


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def make_plan():
    tab = json.load(open(T + "raw/SCREEN_TABLE.json"))
    dirs = json.load(open(T + "raw/USDZ_DIRECTORY_RENDERER_MEMBERS.json"))
    sel = [r for r in tab["scenes"] if r["selected"]]
    sel.sort(key=lambda r: (0 if r["slot"].startswith("launch") else 1, r["order_index"]))
    free = {}
    for d, root in ROOTS:
        os.makedirs(root, exist_ok=True)
        free[d] = shutil.disk_usage(root).free
    used = {d: 0 for d, _ in ROOTS}
    scenes, total = [], 0
    for r in sel:
        sc = r["scene"]
        need = sum(dirs[sc][m][0] for m in MEMBERS)
        for d, root in ROOTS:
            if free[d] - used[d] - need >= PLAN_FLOOR:
                used[d] += need
                scenes.append({"scene": sc, "slot": r["slot"], "drive": d, "root": root, "bytes": need})
                break
        else:
            raise SystemExit(f"PLAN DOES NOT FIT: scene {sc} needs {need} B; free {free}, planned {used}")
        total += need
    plan = {"created_utc": utc(), "n_scenes": len(scenes), "planned_bytes": total, "free_bytes_at_plan": free, "planned_per_drive": used,
            "floor_bytes_hard": HARD_FLOOR, "floor_bytes_plan": PLAN_FLOOR, "scenes": scenes}
    json.dump(plan, open(PLAN, "w"), indent=1)
    return plan


def load_manifest():
    return json.load(open(MAN)) if os.path.exists(MAN) else {"files": {}}


def save_manifest(m):
    tmp = MAN + ".tmp"
    json.dump(m, open(tmp, "w"), indent=1)
    os.replace(tmp, MAN)


def hash_file(path):
    h, crc = hashlib.sha256(), 0
    with open(path, "rb") as f:
        while True:
            b = f.read(8 << 20)
            if not b:
                break
            h.update(b); crc = zlib.crc32(b, crc)
    return h.hexdigest(), crc & 0xFFFFFFFF


def data_offset(sc, name, ho):
    head = t2lib.fetch_range(sc, ho, ho + 30 + len(name) + 512 - 1)
    assert struct.unpack("<I", head[:4])[0] == 0x04034B50, "bad local header"
    fnl, exl = struct.unpack("<HH", head[26:30])
    return ho + 30 + fnl + exl


def fetch_file(sc, name, ent, dest, drive):
    size, crc, ho, ct = ent
    assert ct == 0
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    if have > size:
        os.replace(part, part + ".oversize"); have = 0
    with lock:
        free = shutil.disk_usage(os.path.dirname(dest)).free
        if free - inflight.get(drive, 0) - (size - have) < HARD_FLOOR:
            raise RuntimeError(f"DISK FLOOR: {drive}: free {free} would drop below 10 GB for {name} of {sc}")
        inflight[drive] = inflight.get(drive, 0) + (size - have)
    try:
        base = data_offset(sc, name, ho)
        attempts = 0
        while have < size:
            try:
                r = hfc.get(t2lib.usdz_url(sc), headers={"Range": f"bytes={base + have}-{base + size - 1}"}, stream=True)
                if r.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {r.status_code}")
                with open(part, "ab") as f:
                    for b in r.iter_content(CHUNK):
                        if b:
                            f.write(b); have += len(b)
                            with lock:
                                inflight[drive] -= len(b)
                r.close()
            except Exception:
                attempts += 1
                have = os.path.getsize(part) if os.path.exists(part) else 0
                if attempts > 8:
                    raise
                time.sleep(min(60, 3 * attempts))
        sha, c = hash_file(part)
        if os.path.getsize(part) != size or c != crc:
            os.replace(part, part + ".bad")
            raise RuntimeError(f"verify failed for {name} of {sc}: size {os.path.getsize(part + '.bad')} crc {c} want {size} {crc}")
        os.replace(part, dest)
        return sha, c
    finally:
        with lock:
            inflight[drive] = max(0, inflight.get(drive, 0))


def main():
    for mk in (DONE, FAIL):
        if os.path.exists(mk):
            os.replace(mk, mk + ".prev_" + time.strftime("%Y%m%dT%H%M%S"))
    plan = json.load(open(PLAN)) if os.path.exists(PLAN) else make_plan()
    if "--plan-only" in sys.argv:
        print(json.dumps({k: v for k, v in plan.items() if k != "scenes"}, indent=1)); return
    dirs = json.load(open(T + "raw/USDZ_DIRECTORY_RENDERER_MEMBERS.json"))
    man = load_manifest()
    jobs = []
    for s in plan["scenes"]:
        for m in MEMBERS:
            jobs.append((s["scene"], m, dirs[s["scene"]][m], f'{s["root"]}/{s["scene"]}/{SUBDIR[m]}{m}', s["drive"], s["slot"]))
    t0, done_bytes, failures = time.time(), 0, []

    def work(j):
        sc, m, ent, dest, drive, slot = j
        key = f"{sc}/{m}"
        rec = man["files"].get(key)
        if rec and rec.get("verified") and os.path.exists(dest) and os.path.getsize(dest) == ent[0]:
            return key, rec, True
        sha, c = fetch_file(sc, m, ent, dest, drive)
        rec = {"scene": sc, "slot": slot, "member": m, "bytes": ent[0], "crc32": f"{c:08x}", "crc32_expected": f"{ent[1]:08x}",
               "sha256": sha, "sha256_published": None, "location": dest, "verified": True, "verified_utc": utc(),
               "source": f"{hfc.REPO}@{t2lib.REV}:{t2lib.REL}/{sc}/{sc}.usdz member {m}"}
        return key, rec, False

    with ThreadPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(work, j) for j in jobs]
        for i, fu in enumerate(futs):
            try:
                key, rec, skipped = fu.result()
                with lock:
                    man["files"][key] = rec
                    done_bytes += rec["bytes"]
                    if not skipped or i % 20 == 0:
                        save_manifest(man)
            except Exception as e:
                failures.append({"job": f"{jobs[i][0]}/{jobs[i][1]}", "error": (type(e).__name__ + ": " + str(e))[:300]})
            if i % 10 == 0 or i == len(futs) - 1:
                json.dump({"utc": utc(), "files_done": len(man["files"]), "files_total": len(jobs), "bytes_done": done_bytes,
                           "bytes_planned": plan["planned_bytes"], "failures": len(failures), "elapsed_s": round(time.time() - t0)}, open(STATUS, "w"))
                print(f"{i + 1}/{len(jobs)} bytes {done_bytes} fail {len(failures)} t={round(time.time() - t0)}s", flush=True)
    save_manifest(man)
    man_out = {"n_files": len(man["files"]), "bytes": sum(r["bytes"] for r in man["files"].values()), "failures": failures}
    if failures or len(man["files"]) != len(jobs):
        open(FAIL, "w").write(json.dumps({"utc": utc(), **man_out}, indent=1)); sys.exit(1)
    open(DONE, "w").write(json.dumps({"utc": utc(), **man_out}) + "\n")
    print("DONE", man_out)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        open(FAIL, "w").write(utc() + "\n" + traceback.format_exc())
        raise
