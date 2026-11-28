"""VM side of the signed-link route: download every file of /content/relay_job.json from its signed link -- NO token on
this VM -- and verify bytes + md5 against the relay MANIFEST (carried in the job). A mismatch is a FAILURE, never a
warning. Persists /content/relay_pull.json after every file; deletes the job file (live links) when done.

    & colab\\colab.ps1 upload -s <name> <scratchpad>\\relay_job.json /content/relay_job.json
    & colab\\colab.ps1 exec -s <name> --timeout 1800 -f colab\\signed_pull.py

Prints ZZPULL_OK / ZZPULL_FAIL with names, sizes, md5s, timings and the link HOST -- never a link. ASCII only
(colab-cli reads this file with the dev box's cp1252 codec).
"""
import hashlib
import json
import os
import time
import urllib.parse
import urllib.request

JOB, OUT = "/content/relay_job.json", "/content/relay_pull.json"
job = json.load(open(JOB))
dest = job.get("dest", "/content/relay")
res = {"repo": job.get("repo"), "prefix": job.get("prefix"), "dest": dest, "files": {},
       "t_start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def save():
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f, indent=1)
    os.replace(tmp, OUT)


ok = True
for rel, ent in job["files"].items():
    p = os.path.join(dest, rel)
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    h, n, t0 = hashlib.md5(), 0, time.time()
    try:
        with urllib.request.urlopen(ent["url"], timeout=60) as r, open(p + ".part", "wb") as f:
            for b in iter(lambda: r.read(1 << 22), b""):
                f.write(b)
                h.update(b)
                n += len(b)
        os.replace(p + ".part", p)
        good = n == ent["bytes"] and h.hexdigest() == ent["md5"]
        err = None
    except Exception as e:  # noqa: BLE001 -- an expired link or a network error is reported per file
        good, err = False, f"{type(e).__name__}: {str(e)[:120]}"
    dt = time.time() - t0
    res["files"][rel] = {"ok": good, "bytes": n, "want_bytes": ent["bytes"], "md5": h.hexdigest(),
                         "want_md5": ent["md5"], "seconds": round(dt, 1), "mb_s": round(n / 1e6 / max(dt, 1e-6), 1),
                         "host": urllib.parse.urlparse(ent["url"]).hostname, "error": err, "path": p}
    ok &= good
    save()
os.remove(JOB)
res["job_file_deleted"] = not os.path.exists(JOB)
res["t_end_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
save()
print(("ZZPULL_OK " if ok else "ZZPULL_FAIL ") + json.dumps(res))
