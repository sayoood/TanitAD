#!/usr/bin/env python3
"""2026-10-19 pod reads under P6's overlap rule: (1) SH5 FULL frames -> D:/.../sh5_ctx_2026-10-19/frames_full + the SH5 plan re-run for FULL; (2) the nuPlan DBs of the 76 f10 logs
PRIVGEOM-SAFE could not read -> the flat local DB folder (D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits, the one `navtrain_scenarios.index_dbs()` reads).

    podstage.py frames|plan|dbs|finalize|all     (all = frames -> plan -> dbs -> finalize; each phase is resumable)

Everything pod-side runs through `podcopy2.py` (nice -n 19 ionice -c3, ONE stream, <= 120 MB batches, a P6_GUARD line in P6's chain log pauses it, never a GPU process, never `git`, never `pkill -f`).
The ssh prefix is read at run time from the file named in POD_CFG and is never printed or written; every logged string is redacted.
Markers (grep the log):  ZZFRAMES_LIST  ZZFRAMES_COPIED  ZZFRAMES_VERIFIED  ZZPLAN_FULL_OK  ZZDBS_LIST  ZZDBS_COPIED  ZZDBS_VERIFIED  ZZDBS_PARTIAL_DISK  ZZPODSTAGE_DONE  ZZPODSTAGE_FAIL
"""
import hashlib
import json
import os
import re
import shlex
import shutil
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
PS = os.path.dirname(HERE)                                              # C:/Users/Admin/par18/podstage
WORK = f"{PS}/work"
POD_CFG = os.environ.get("PODSTAGE_CFG") or os.path.join(PS, "pod_cfg.local")      # a local one-line pointer file (NOT staged, NOT in the repo) that names the file holding the ssh prefix
P6_LOG = "/workspace/data/refe_p6/chain.log"
POD_DBS = "/workspace/data/navtrain_dbs"
POD_PIX = "/workspace/data/navtrain_pixels"
NICE = "nice -n 19 ionice -c3"
DD = "D:/Projects/TanitAD-artifacts/sh5_ctx_2026-10-19"
FRAMES = f"{DD}/frames_full"
PLAN_FULL = f"{DD}/plan_full"
SHA_DIR = "D:/Projects/TanitAD-artifacts/pod_stage_2026-10-19"           # the sha256 lists live here (D:), counts only in MANIFEST.md
DB_DEST = "D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits"           # = navtrain_scenarios.DEFAULT_DB_ROOT = where PRIVGEOM's 41 local DBs live
SH5 = "C:/Users/Admin/par18/sh5"
NEEDED = f"{SH5}/work/plan/needed_frames.json"
PY = "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe"
MIN_FREE_GB = float(os.environ.get("PODSTAGE_MIN_FREE_GB", "20"))
RETRY_P6_S = 600


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def say(msg):
    print(f"[{now()}] {msg}", flush=True)


PODSSH = shlex.split(open(open(POD_CFG, encoding="utf-8").read().strip(), encoding="utf-8").read().strip())
os.environ["PODSSH"] = " ".join(shlex.quote(t) for t in PODSSH)
_REDACT = [re.escape(t) for t in PODSSH if len(t) >= 4]


def red(s):
    if isinstance(s, bytes):
        s = s.decode("utf-8", "replace")
    for pat in _REDACT:
        s = re.sub(pat, "<ssh>", s)
    return re.sub(r"\b\d{1,3}(\.\d{1,3}){3}\b", "<ip>", s)


def guard_prefix():
    return (f'G=$({NICE} tail -n 20 {shlex.quote(P6_LOG)} 2>/dev/null) || exit 94; case "$G" in *P6_GUARD*) exit 93;; esac; ')


def ssh(cmd, stdin=b"", timeout=900):
    r = subprocess.run(PODSSH + [cmd], input=stdin, capture_output=True, timeout=timeout)
    return r.returncode, r.stdout.decode("utf-8", "replace"), red(r.stderr)


def guarded_ssh(cmd, stdin=b""):
    """one guarded remote command; waits 10 min and retries on a P6_GUARD (93) / unreadable chain log (94)."""
    while True:
        rc, out, err = ssh(guard_prefix() + cmd, stdin)
        if rc in (93, 94):
            say(f"P6_GUARD / unreadable chain log (rc {rc}): waiting {RETRY_P6_S // 60} min")
            time.sleep(RETRY_P6_S)
            continue
        return rc, out, err


def run_copy(lst, dest, manifest, max_batches=0):
    """podcopy2 with the P6 wait (rc 3) and the ssh retry (rc 6); rc 4 (size/sha mismatch) and rc 5 (disk guard) are returned to the caller."""
    tries = 0
    while True:
        cmd = [sys.executable, "-B", f"{HERE}/podcopy2.py", "--list", lst, "--dest", dest, "--manifest", manifest, "--max-batch-mb", "120", "--min-free-gb", str(MIN_FREE_GB), "--p6-log", P6_LOG]
        if max_batches:
            cmd += ["--max-batches", str(max_batches)]
        rc = subprocess.call(cmd, env=os.environ.copy())
        if rc == 3:
            say(f"podcopy2 rc 3 (P6_GUARD): waiting {RETRY_P6_S // 60} min; resumable")
            time.sleep(RETRY_P6_S)
            continue
        if rc == 6:
            tries += 1
            if tries > 6:
                return 6
            say(f"podcopy2 rc 6 (ssh/tar failure) attempt {tries}: retry in 60 s; resumable")
            time.sleep(60)
            continue
        return rc


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------
def frames_list():
    d = json.load(open(NEEDED, encoding="utf-8"))
    rels = sorted({fn for v in d["tokens"].values() for fn in v["frames_filename_jpg"].values()})
    say(f"needed_frames.json: {d['n_tokens']} tokens, {len(rels)} distinct jpg names")
    stat = ('declare -A SH; while IFS= read -r rel; do [ -z "$rel" ] && continue; log=${rel%%/*}; '
            'if [ -z "${SH[$log]+x}" ]; then d=$(ls -d ' + POD_PIX + '/*/"$log" 2>/dev/null | head -1); SH[$log]=${d%/*}; fi; sh=${SH[$log]}; f="$sh/$rel"; '
            'if [ -f "$f" ]; then printf "%s\\t%s\\t%s\\n" "$(stat -c %s "$f")" "${sh##*/}" "$rel"; else printf "MISSING\\t\\t%s\\n" "$rel"; fi; done')
    rc, out, err = guarded_ssh(f"{NICE} bash -c {shlex.quote(stat)}", ("\n".join(rels) + "\n").encode())
    if rc:
        raise SystemExit(f"frame stat rc={rc} {err[:200]!r}")
    rows, miss = [], []
    for ln in out.splitlines():
        p = ln.split("\t")
        if p[0] == "MISSING":
            miss.append(p[2])
        elif len(p) == 3:
            rows.append((int(p[0]), f"{POD_PIX}/{p[1]}", p[2]))
    rows.sort(key=lambda r: (r[1], r[2]))
    open(f"{WORK}/copy_list_frames.tsv", "w", encoding="utf-8", newline="\n").write("".join(f"{b}\t{base}\t{rel}\n" for b, base, rel in rows))
    tot = sum(r[0] for r in rows)
    json.dump({"n_listed": len(rels), "n_found_on_pod": len(rows), "n_missing_on_pod": len(miss), "missing": miss[:50], "bytes_on_pod": tot, "n_shards": len({r[1] for r in rows}),
               "allocated_bytes_exfat_1MiB": sum(((r[0] + (1 << 20) - 1) // (1 << 20)) << 20 for r in rows)}, open(f"{WORK}/frames_stat.json", "w"), indent=1)
    say(f"ZZFRAMES_LIST listed {len(rels)} found {len(rows)} missing {len(miss)} bytes {tot} ({tot / 2**30:.3f} GiB) shards {len({r[1] for r in rows})}")
    return rows


def verify(lst, dest, manifest, kind, out_json, sha_out):
    """independent check of EVERY listed file: exists, size = listed = manifest, non-zero, magic bytes, sha256 of the bytes on disk = manifest local sha = pod sha; + a consumer-style open for DBs."""
    rows = [ln.rstrip("\n").split("\t") for ln in open(lst, encoding="utf-8") if ln.strip()]
    listed = {r[2]: int(r[0]) for r in rows}
    mf = {}
    for ln in open(manifest, encoding="utf-8"):
        p = ln.rstrip("\n").split("\t")
        if len(p) == 5 and p[4] == "OK":
            mf[p[0]] = p
    bad = {}

    def one(rel):
        p = f"{dest}/{rel}"
        if not os.path.isfile(p):
            return rel, "absent", None, None
        n = os.path.getsize(p)
        if n == 0:
            return rel, "zero-size", None, None
        if n != listed[rel] or rel not in mf or int(mf[rel][1]) != n:
            return rel, f"size {n} vs listed {listed[rel]} / manifest {mf.get(rel, ['-', '-'])[1]}", None, None
        with open(p, "rb") as f:
            head = f.read(16)
            f.seek(-2, 2)
            tail = f.read(2)
        if kind == "jpg" and head[:3] != b"\xff\xd8\xff":
            return rel, "not a JPEG (magic)", None, None
        if kind == "db" and head[:16] != b"SQLite format 3\x00":
            return rel, "not a SQLite file (magic)", None, None
        h = sha256_file(p)
        if not (h == mf[rel][2] == mf[rel][3]):
            return rel, f"sha disk {h[:12]} / local {mf[rel][2][:12]} / pod {mf[rel][3][:12]}", None, None
        if kind == "db":
            try:
                con = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
                ok = con.execute("select count(*) from sqlite_master where name in ('lidar_pc','image','camera')").fetchone()[0] == 3
                con.close()
            except Exception as e:                       # noqa: BLE001
                return rel, f"sqlite open failed: {e}", None, None
            if not ok:
                return rel, "sqlite opens but lidar_pc/image/camera tables are absent", None, None
        return rel, None, h, tail == b"\xff\xd9"

    n_ok, eoi, bytes_ok, shas = 0, 0, 0, []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for rel, why, h, e in ex.map(one, sorted(listed)):
            if why:
                bad[rel] = why
            else:
                n_ok += 1
                bytes_ok += listed[rel]
                eoi += bool(e)
                shas.append((rel, listed[rel], h))
    os.makedirs(SHA_DIR, exist_ok=True)
    open(sha_out, "w", encoding="utf-8", newline="\n").write("".join(f"{h}  {n}  {rel}\n" for rel, n, h in shas))
    res = {"kind": kind, "n_listed": len(listed), "n_verified": n_ok, "bytes_verified": bytes_ok, "n_failed": len(bad), "failed_first20": dict(list(bad.items())[:20]),
           "jpeg_ends_with_FFD9": eoi if kind == "jpg" else None, "sha_list": sha_out, "verified_at": now(),
           "checks": "exists; size = pod stat = manifest; non-zero; magic bytes; sha256 of the bytes on disk = sha256 of the stream = pod-side sha256sum" + ("; sqlite opens read-only with lidar_pc+image+camera" if kind == "db" else "")}
    json.dump(res, open(out_json, "w"), indent=1)
    return res


def phase_frames():
    os.makedirs(FRAMES, exist_ok=True)
    if not os.path.exists(f"{WORK}/copy_list_frames.tsv"):
        frames_list()
    rc = run_copy(f"{WORK}/copy_list_frames.tsv", FRAMES, f"{WORK}/manifest_frames.tsv")
    if rc not in (0, 5):
        say(f"ZZPODSTAGE_FAIL frames copy rc={rc} (4 = size/sha mismatch)")
        return rc
    lst_all = [ln for ln in open(f"{WORK}/copy_list_frames.tsv", encoding="utf-8") if ln.strip()]
    if rc == 5:
        say(f"ZZFRAMES_PARTIAL_DISK the disk guard ({MIN_FREE_GB} GiB free) stopped the frames copy; what is in is verified below; resume after space is freed or with PODSTAGE_MIN_FREE_GB lowered by the PM/PI")
        have = {ln.split("\t")[0] for ln in open(f"{WORK}/manifest_frames.tsv", encoding="utf-8")}
        lst_use = [ln for ln in lst_all if ln.rstrip("\n").split("\t")[2] in have]
        open(f"{WORK}/copy_list_frames_present.tsv", "w", encoding="utf-8", newline="\n").write("".join(lst_use))
        lst_path = f"{WORK}/copy_list_frames_present.tsv"
    else:
        say("ZZFRAMES_COPIED")
        lst_path = f"{WORK}/copy_list_frames.tsv"
    res = verify(lst_path, FRAMES, f"{WORK}/manifest_frames.tsv", "jpg", f"{WORK}/verify_frames.json", f"{SHA_DIR}/frames_full_sha256.txt")
    res["n_listed_total"] = len(lst_all)
    res["partial_disk_guard"] = rc == 5
    json.dump(res, open(f"{WORK}/verify_frames.json", "w"), indent=1)
    say(f"ZZFRAMES_VERIFIED {res['n_verified']}/{len(lst_all)} files, {res['bytes_verified'] / 1e9:.3f} GB, failed {res['n_failed']}")
    return (0 if rc == 0 else 5) if res["n_failed"] == 0 else 7


def phase_plan(out_dir=None):
    """READY.md Q1 exactly: CUDA_VISIBLE_DEVICES=-1 PY sh5_ctx_build.py plan --inputs DD/data/sh5_inputs.npz --out-dir DD/plan_full --images-root IMG"""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", PYTHONIOENCODING="utf-8")
    out_dir = out_dir or PLAN_FULL
    os.makedirs(out_dir, exist_ok=True)
    cmd = [PY, "-B", "-W", "ignore", "sh5_ctx_build.py", "plan", "--inputs", f"{DD}/data/sh5_inputs.npz", "--out-dir", out_dir, "--images-root", FRAMES]
    say("plan: " + " ".join(cmd[2:]))
    rc = subprocess.call(cmd, cwd=f"{SH5}/code", env=env)
    if rc:
        say(f"ZZPODSTAGE_FAIL plan rc={rc}")
        return rc
    cov = json.load(open(f"{out_dir}/coverage.json", encoding="utf-8"))
    say(f"ZZPLAN_{'FULL' if out_dir == PLAN_FULL else 'PARTIAL'}_OK tokens {cov['n_tokens_with_frames']}/{cov['n_tokens']} sets {cov['n_sets_kept']}/{cov['n_sets']} logs {cov['n_logs_with_any']} sets_kept_by_fold {cov['sets_kept_by_fold']}")
    return 0


def dbs_list():
    need = json.load(open(f"{WORK}/db_logs.json"))["need"]
    stat = ('while IFS= read -r rel; do [ -z "$rel" ] && continue; f=' + POD_DBS + '/"$rel"; '
            'if [ -f "$f" ]; then printf "%s\\t%s\\n" "$(stat -c %s "$f")" "$rel"; else printf "MISSING\\t%s\\n" "$rel"; fi; done')
    rc, out, err = guarded_ssh(f"{NICE} bash -c {shlex.quote(stat)}", ("\n".join(f"{lg}.db" for lg in need) + "\n").encode())
    if rc:
        raise SystemExit(f"DB stat rc={rc} {err[:200]!r}")
    rows, miss = [], []
    for ln in out.splitlines():
        p = ln.split("\t")
        if p[0] == "MISSING":
            miss.append(p[1])
        elif len(p) == 2:
            rows.append((int(p[0]), p[1]))
    rows.sort()                                                              # ascending size: if the disk guard stops the run, the MOST logs are in
    open(f"{WORK}/copy_list_dbs.tsv", "w", encoding="utf-8", newline="\n").write("".join(f"{b}\t{POD_DBS}\t{rel}\n" for b, rel in rows))
    tot = sum(b for b, _ in rows)
    json.dump({"n_wanted": len(need), "n_found": len(rows), "missing": miss, "total_bytes": tot, "total_GiB": tot / 2**30, "total_GB": tot / 1e9, "max_MiB": rows[-1][0] / 2**20, "min_MiB": rows[0][0] / 2**20},
              open(f"{WORK}/db_stat.json", "w"), indent=1)
    say(f"ZZDBS_LIST wanted {len(need)} found {len(rows)} missing {len(miss)} total {tot} B = {tot / 1e9:.3f} GB = {tot / 2**30:.3f} GiB")


def phase_dbs():
    if not os.path.exists(f"{WORK}/copy_list_dbs.tsv"):
        dbs_list()
    st = json.load(open(f"{WORK}/db_stat.json"))
    if st["total_GB"] > 20.0:
        say(f"ZZPODSTAGE_FAIL DB total {st['total_GB']:.2f} GB > 20 GB: NOT transferred, report the size")
        return 8
    rc = run_copy(f"{WORK}/copy_list_dbs.tsv", DB_DEST, f"{WORK}/manifest_dbs.tsv")
    if rc == 5:
        say("ZZDBS_PARTIAL_DISK the disk guard stopped the DB copy; what is in is verified below; resume with the same command after space is freed (or PODSTAGE_MIN_FREE_GB lowered by the PI)")
    elif rc:
        say(f"ZZPODSTAGE_FAIL dbs copy rc={rc} (4 = size/sha mismatch)")
        return rc
    else:
        say("ZZDBS_COPIED")
    # verify what the manifest holds (all of it when rc == 0)
    lst_all = [ln for ln in open(f"{WORK}/copy_list_dbs.tsv", encoding="utf-8") if ln.strip()]
    have = {ln.split("\t")[0] for ln in open(f"{WORK}/manifest_dbs.tsv", encoding="utf-8")} if os.path.exists(f"{WORK}/manifest_dbs.tsv") else set()
    sub = [ln for ln in lst_all if ln.rstrip("\n").split("\t")[2] in have] if rc == 5 else lst_all
    open(f"{WORK}/copy_list_dbs_present.tsv", "w", encoding="utf-8", newline="\n").write("".join(sub))
    res = verify(f"{WORK}/copy_list_dbs_present.tsv", DB_DEST, f"{WORK}/manifest_dbs.tsv", "db", f"{WORK}/verify_dbs.json", f"{SHA_DIR}/dbs_76_sha256.txt")
    res["n_wanted"] = len(lst_all)
    json.dump(res, open(f"{WORK}/verify_dbs.json", "w"), indent=1)
    say(f"ZZDBS_VERIFIED {res['n_verified']}/{len(lst_all)} DBs, {res['bytes_verified'] / 1e9:.3f} GB, failed {res['n_failed']}")
    return 0 if res["n_failed"] == 0 and rc == 0 else (5 if rc == 5 and res["n_failed"] == 0 else 7)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    os.makedirs(WORK, exist_ok=True)
    if cmd == "frames":
        return phase_frames()
    if cmd == "plan":
        return phase_plan(sys.argv[2] if len(sys.argv) > 2 else None)
    if cmd == "dbs":
        return phase_dbs()
    if cmd == "pilot":                                                       # the first two batches of the frames, synchronous (smoke of the whole copy+sha path)
        os.makedirs(FRAMES, exist_ok=True)
        if not os.path.exists(f"{WORK}/copy_list_frames.tsv"):
            frames_list()
        return run_copy(f"{WORK}/copy_list_frames.tsv", FRAMES, f"{WORK}/manifest_frames.tsv", max_batches=2)
    if cmd == "all":
        for ph, fn in (("frames", phase_frames), ("plan", phase_plan), ("dbs", phase_dbs)):
            say(f"=== phase {ph}")
            rc = fn()
            if rc and not (ph == "dbs" and rc == 5):
                say(f"ZZPODSTAGE_FAIL phase {ph} rc={rc}")
                return rc
            if rc == 5:
                say("phase dbs ended at the disk guard (rc 5): partial, see ZZDBS_PARTIAL_DISK")
        subprocess.call([sys.executable, "-B", f"{HERE}/build_package.py"])
        say("ZZPODSTAGE_DONE")
        return 0
    raise SystemExit("usage: podstage.py frames|plan|dbs|pilot|all")


if __name__ == "__main__":
    sys.exit(main())
