"""Mint SHORT-LIVED signed download links for files in the private HF relay, for one Colab job.

    C:/Users/Admin/venvs/tanitad/Scripts/python.exe colab/hf_signed_links.py --prefix refe/snapshots \\
        --out <scratchpad>/relay_job.json [FILE ...]          # default: every file in the prefix's MANIFEST

PI, 2026-09-27: "Signed links" -- the HF token never leaves the dev box (moving it to a VM was refused as data
exfiltration, and a CLI-provisioned VM cannot read Colab Secrets). For each file this reads the relay MANIFEST's bytes
and md5 (with the token, HERE), then asks HF's resolve endpoint for the file WITHOUT following the redirect: the
redirect target is HF's own signed CDN link, which opens that ONE file until it expires. The job JSON carries only
those links + the expected bytes/md5, for signed_pull.py on the VM.
Prints names, sizes, the link HOST and its expiry -- never a link, never the token. The job file holds live links:
write it to the scratchpad (never the repo) and delete it after `colab upload`.
"""
import argparse
import calendar
import json
import re
import sys
import time
import urllib.parse
from pathlib import Path

import truststore

truststore.inject_into_ssl()
import requests  # noqa: E402
from huggingface_hub import hf_hub_download, hf_hub_url  # noqa: E402

REPO = "Sayood/tanitad-colab-relay"


def expiry_utc(url):
    q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    if "Expires" in q:
        return int(q["Expires"][0])
    if "X-Amz-Date" in q and "X-Amz-Expires" in q:
        return calendar.timegm(time.strptime(q["X-Amz-Date"][0], "%Y%m%dT%H%M%SZ")) + int(q["X-Amz-Expires"][0])
    return None


def signed(url, tok):
    """follow redirects that stay on huggingface.co (with auth); return the first OFF-hub Location, else None"""
    for _ in range(4):
        r = requests.get(url, headers={"Authorization": f"Bearer {tok}"}, allow_redirects=False, stream=True, timeout=60)
        r.close()
        if r.status_code not in (301, 302, 303, 307, 308):
            return None, r.status_code
        loc = urllib.parse.urljoin(url, r.headers["Location"])
        if urllib.parse.urlparse(loc).hostname.endswith("huggingface.co"):
            url = loc
            continue
        return loc, r.status_code
    return None, "too many hub redirects"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dest", default="/content/relay")
    ap.add_argument("files", nargs="*")
    a = ap.parse_args()
    if "D:/Projects/TanitAD" in a.out.replace("\\", "/"):
        print("ZZLINKS_REFUSED the job file holds live links; write it outside the repo")
        return 1
    toks = re.findall(r"hf_[A-Za-z0-9]{20,}", Path("D:/Projects/TanitAD/Keys.txt").read_text(encoding="utf-8", errors="replace"))
    if len(toks) != 1:
        print(f"ZZLINKS_FAIL expected exactly one hf_ token in Keys.txt, found {len(toks)}")
        return 1
    tok = toks[0]
    man = json.load(open(hf_hub_download(REPO, f"{a.prefix}/MANIFEST.json", repo_type="dataset", token=tok,
                                         force_download=True), encoding="utf-8"))
    want = a.files or sorted(man["files"])
    job = {"repo": REPO, "prefix": a.prefix, "dest": a.dest, "minted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "files": {}}
    for rel in want:
        ent = man["files"].get(rel)
        if ent is None:
            print(f"ZZLINKS_FAIL {rel} is not in the MANIFEST")
            return 1
        loc, status = signed(hf_hub_url(REPO, f"{a.prefix}/{rel}", repo_type="dataset"), tok)
        if not loc:
            print(f"ZZLINKS_FAIL {rel}: no signed redirect (status {status})")
            return 1
        exp = expiry_utc(loc)
        job["files"][rel] = {"url": loc, "bytes": ent["bytes"], "md5": ent["md5"], "expires_utc": exp}
        left = f"{(exp - time.time()) / 60:.0f} min" if exp else "expiry not in the link"
        print(f"signed {rel} {ent['bytes']} B md5 {ent['md5']} via {urllib.parse.urlparse(loc).hostname}, valid {left}")
    Path(a.out).write_text(json.dumps(job), encoding="utf-8")
    print(f"ZZLINKS_OK {len(job['files'])} file(s) -> job file (live links; upload, then delete)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
