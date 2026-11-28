"""VM side of the private-HF relay: pull what the relay MANIFEST lists, verify every md5.

    & colab\\colab.ps1 upload -s <name> <local relay_job.json> /content/relay_job.json    # optional: repo/prefix/dest
    & colab\\colab.ps1 exec -s <name> --timeout 1800 -f colab\\hf_relay_pull.py

TOKEN: only a credential the PI provisions for the VM -- a Colab Secret named HF_TOKEN (Colab UI, key icon, on the
plan account), read here with google.colab.userdata. The orchestrator never moves the token from Keys.txt to a VM
(the permission system refused that as data exfiltration, 2026-09-27). Whether userdata resolves in a CLI-provisioned
session is UNVERIFIED until the first run; without a token this prints ZZPULL_NO_TOKEN and pulls nothing.
Reads /content/relay_job.json when present ({"repo", "prefix", "dest", "files"}; the defaults pull every file under
refe/snapshots). The MANIFEST is written LAST by hf_relay_push.py, so a file it lists is complete; a listed file
whose md5 differs is a FAILURE, never a warning. Prints ZZPULL_OK / ZZPULL_FAIL and writes /content/relay_pull.json
after every file. ASCII only (colab-cli reads this file with the dev box's cp1252 codec); no secret in this file.
"""
import hashlib
import json
import os
import time

job = {"repo": "Sayood/tanitad-colab-relay", "prefix": "refe/snapshots", "dest": "/content/relay", "files": None}
if os.path.exists("/content/relay_job.json"):
    job.update(json.load(open("/content/relay_job.json")))
OUT = "/content/relay_pull.json"
res = {"job": job, "files": {}, "t_start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def save():
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f, indent=1)
    os.replace(tmp, OUT)


tok = None
try:
    from google.colab import userdata
    tok = userdata.get("HF_TOKEN")
    res["token_source"] = "colab_secret"
except Exception as e:  # noqa: BLE001 -- absent secret / no frontend: reported, never guessed around
    res["token_source"] = f"none ({type(e).__name__})"
if not tok:
    save()
    print("ZZPULL_NO_TOKEN " + json.dumps(res))
    raise SystemExit(3)

from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

res["whoami"] = HfApi(token=tok).whoami()["name"]
man_path = hf_hub_download(job["repo"], f"{job['prefix']}/MANIFEST.json", repo_type="dataset", token=tok,
                           local_dir=job["dest"], force_download=True)
man = json.load(open(man_path))
want = job["files"] or sorted(man["files"])
ok = True
for rel in want:
    ent = man["files"].get(rel)
    if ent is None:
        res["files"][rel] = {"ok": False, "why": "not in MANIFEST"}
        ok = False
        save()
        continue
    t0 = time.time()
    p = hf_hub_download(job["repo"], f"{job['prefix']}/{rel}", repo_type="dataset", token=tok, local_dir=job["dest"])
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    good = h.hexdigest() == ent["md5"] and os.path.getsize(p) == ent["bytes"]
    res["files"][rel] = {"ok": good, "bytes": os.path.getsize(p), "md5": h.hexdigest(), "want_md5": ent["md5"],
                         "seconds": round(time.time() - t0, 1), "path": p}
    ok &= good
    save()
res["t_end_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
save()
print(("ZZPULL_OK " if ok else "ZZPULL_FAIL ") + json.dumps(res))
