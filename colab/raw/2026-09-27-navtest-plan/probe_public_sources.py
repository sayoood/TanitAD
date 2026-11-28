"""METADATA-ONLY probe of the public sources for every REFe-navtest input (HEAD, API listings, <1 MB range reads).

Nothing bulk is downloaded: the largest reads are the nuPlan test zip's tail (128 KiB) and its central directory
(~0.2 MB), and the HF / GitHub / PyPI JSON listings. Writes public_sources.json beside this file.

Run with an interpreter that has `truststore` (the dev box's TLS proxy breaks certifi -- memory note
`hf-access-on-dev-box`): C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe probe_public_sources.py
"""
from __future__ import annotations

import json
import os
import re
import struct
import sys
import time
import urllib.error
import urllib.request
import zipfile

import truststore

truststore.inject_into_ssl()

OUT = os.path.dirname(os.path.abspath(__file__))
UA = {"User-Agent": "tanitad-navtest-colab-plan/1.0 (metadata probe)"}
HF = "https://huggingface.co"
S3 = "https://motional-nuplan.s3.ap-northeast-1.amazonaws.com/public/nuplan-v1.1"
S3_MAPS_NAVSIM = "https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-maps-v1.1.zip"
LOCAL_TEST_ZIP = "D:/Projects/TanitAD/data/nuplan/nuplan-v1.1_test.zip"
FULL = "D:/Projects/TanitAD/data/refe_navtest/navtest_all_tokens.json"
SUB = ("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-v1-navtest/raw/"
       "A1_sub200_tokens.json")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_noredir = urllib.request.build_opener(NoRedirect)


def head(url, follow=False, timeout=60):
    req = urllib.request.Request(url, method="HEAD", headers=UA)
    t = time.time()
    try:
        r = (urllib.request.urlopen(req, timeout=timeout) if follow else _noredir.open(req, timeout=timeout))
        code, hdr = r.status, dict(r.headers)
    except urllib.error.HTTPError as e:
        code, hdr = e.code, dict(e.headers or {})
    keep = ("Content-Length", "Accept-Ranges", "ETag", "Last-Modified", "Content-Type", "Location",
            "X-Linked-Size", "X-Linked-ETag", "X-Repo-Commit", "x-amz-server-side-encryption")
    return {"url": url, "status": code, "ms": round(1000 * (time.time() - t)),
            "headers": {k: hdr.get(k) for k in keep if hdr.get(k) is not None}}


def get(url, rng=None, timeout=120):
    h = dict(UA)
    if rng:
        h["Range"] = f"bytes={rng[0]}-{rng[1]}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout) as r:
        return r.status, r.read()


def get_json(url):
    try:
        st, b = get(url)
        return st, json.loads(b.decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, None


def hf_tree(kind, repo, path=""):
    url = f"{HF}/api/{kind}/{repo}/tree/main" + (f"/{path}" if path else "")
    st, js = get_json(url)
    return st, js


def zip_entries(cd):
    out, off = [], 0
    while off + 46 <= len(cd) and cd[off:off + 4] == b"PK\x01\x02":
        (_, vm, vn, flg, mth, tm, dt, crc, csz, usz, nlen, elen, clen,
         dsk, iat, eat, lho) = struct.unpack("<IHHHHHHIIIHHHHHII", cd[off:off + 46])
        name = cd[off + 46:off + 46 + nlen].decode("utf-8", "replace")
        extra = cd[off + 46 + nlen:off + 46 + nlen + elen]
        if (usz == 0xFFFFFFFF or csz == 0xFFFFFFFF or lho == 0xFFFFFFFF) and extra:
            p = 0
            while p + 4 <= len(extra):
                hid, hsz = struct.unpack("<HH", extra[p:p + 4])
                body, q = extra[p + 4:p + 4 + hsz], 0
                if hid == 0x0001:
                    if usz == 0xFFFFFFFF:
                        usz = struct.unpack("<Q", body[q:q + 8])[0]; q += 8
                    if csz == 0xFFFFFFFF:
                        csz = struct.unpack("<Q", body[q:q + 8])[0]; q += 8
                    if lho == 0xFFFFFFFF:
                        lho = struct.unpack("<Q", body[q:q + 8])[0]; q += 8
                p += 4 + hsz
        out.append({"name": name, "method": mth, "csz": csz, "usz": usz, "lho": lho, "crc": crc})
        off += 46 + nlen + elen + clen
    return out


def remote_central_directory(url, total):
    _, tail = get(url, (max(0, total - 131072), total - 1))
    i = tail.rfind(b"PK\x05\x06")
    cdsz = struct.unpack("<I", tail[i + 12:i + 16])[0]
    cdoff = struct.unpack("<I", tail[i + 16:i + 20])[0]
    j = tail.rfind(b"PK\x06\x06")
    if j >= 0:
        cdsz = struct.unpack("<Q", tail[j + 40:j + 48])[0]
        cdoff = struct.unpack("<Q", tail[j + 48:j + 56])[0]
    _, cd = get(url, (cdoff, cdoff + cdsz - 1))
    return cd, {"tail_bytes_read": len(tail), "cd_bytes_read": len(cd), "cd_offset": cdoff}


def main() -> int:
    t0 = time.time()
    R = {"probed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "from": "dev box (truststore)"}
    full = json.load(open(FULL, encoding="utf-8"))
    sub = json.load(open(SUB, encoding="utf-8"))
    logs_full = sorted({full["token_log"][t] for t in full["tokens"]})
    logs_sub = sorted({sub["token_log"][t] for t in sub["tokens"]})

    # ---- A. nuPlan test DBs: the S3 zip, and ONLY its central directory ---------------------------------------
    url = f"{S3}/nuplan-v1.1_test.zip"
    h = head(url)
    R["nuplan_test_zip_head"] = h
    total = int(h["headers"].get("Content-Length", 0) or 0)
    cd, cdinfo = remote_central_directory(url, total)
    ents = [e for e in zip_entries(cd) if e["name"].endswith(".db")]
    by_log = {os.path.basename(e["name"])[:-3]: e for e in ents}
    loc = {}
    with zipfile.ZipFile(LOCAL_TEST_ZIP) as z:
        for i in z.infolist():
            if i.filename.endswith(".db"):
                loc[os.path.basename(i.filename)[:-3]] = (i.compress_size, i.file_size, i.CRC)
    agree = sum(1 for lg in logs_full if lg in by_log and lg in loc
                and (by_log[lg]["csz"], by_log[lg]["usz"], by_log[lg]["crc"]) == loc[lg])
    R["nuplan_test_zip_remote_cd"] = {
        **cdinfo, "n_db_entries": len(ents), "local_zip_bytes": os.path.getsize(LOCAL_TEST_ZIP),
        "remote_equals_local_size": total == os.path.getsize(LOCAL_TEST_ZIP),
        "navtest_logs_found": sum(lg in by_log for lg in logs_full), "navtest_logs": len(logs_full),
        "navtest_entries_identical_csz_usz_crc_to_local": agree,
        "full_compressed_bytes": sum(by_log[lg]["csz"] for lg in logs_full if lg in by_log),
        "full_uncompressed_bytes": sum(by_log[lg]["usz"] for lg in logs_full if lg in by_log),
        "sub200_compressed_bytes": sum(by_log[lg]["csz"] for lg in logs_sub if lg in by_log),
        "sub200_uncompressed_bytes": sum(by_log[lg]["usz"] for lg in logs_sub if lg in by_log),
        "methods": sorted({by_log[lg]["method"] for lg in logs_full if lg in by_log}),
    }
    # ---- B. maps (NAVSIM's own download_maps.sh URL, and the programme's S3 base) --------------------------------
    R["maps_zip_head_navsim_url"] = head(S3_MAPS_NAVSIM)
    R["maps_zip_head_base"] = head(f"{S3}/nuplan-maps-v1.1.zip")
    # ---- C. OpenScene on HF: metadata (navsim_logs) and the test camera shards -----------------------------------
    st, js = hf_tree("datasets", "OpenDriveLab/OpenScene", "openscene-v1.1")
    R["hf_openscene_v11_listing_status"] = st
    if js:
        R["hf_openscene_v11_top"] = [{"path": e["path"], "type": e["type"], "size": e.get("size"),
                                      "lfs_sha256": (e.get("lfs") or {}).get("oid")} for e in js]
    st, js = hf_tree("datasets", "OpenDriveLab/OpenScene", "openscene-v1.1/openscene_sensor_test_camera")
    R["hf_test_camera_listing_status"] = st
    if js:
        sh = [{"path": e["path"], "size": e.get("size"), "lfs_sha256": (e.get("lfs") or {}).get("oid")}
              for e in js if e["path"].endswith(".tgz")]
        R["hf_test_camera_shards"] = {"n": len(sh), "bytes": sum(x["size"] for x in sh), "shards": sh}
    for p in ("openscene-v1.1/openscene_metadata_test.tgz",
              "openscene-v1.1/openscene_sensor_test_camera/openscene_sensor_test_camera_0.tgz"):
        R.setdefault("hf_heads", {})[p] = head(f"{HF}/datasets/OpenDriveLab/OpenScene/resolve/main/{p}")
    # ---- D. the DINOv3 ViT-L/16 trunk (partial snapshots rebuild it) ----------------------------------------------
    st, js = hf_tree("models", "timm/vit_large_patch16_dinov3.lvd1689m")
    R["hf_timm_vitl16_listing_status"] = st
    if js:
        R["hf_timm_vitl16_files"] = [{"path": e["path"], "size": e.get("size"),
                                      "lfs_sha256": (e.get("lfs") or {}).get("oid")} for e in js]
    R["hf_timm_vitl16_head"] = head(f"{HF}/timm/vit_large_patch16_dinov3.lvd1689m/resolve/main/model.safetensors")
    # ---- E. code: GitHub repos + pinned commits (unauthenticated API) --------------------------------------------
    gh = {}
    for key, api in (("DriveZero_repo", "repos/XiaomiAutoL3/DriveZero"),
                     ("DriveZero_commit", "repos/XiaomiAutoL3/DriveZero/commits/24959547edac549d64c34b2aab62d5c52599d078"),
                     ("navsim_repo", "repos/autonomousvision/navsim"),
                     ("navsim_v11_commit", "repos/autonomousvision/navsim/commits/3e8291bfa89ff247231e0227778840cd0a036896"),
                     ("nuplan_devkit_repo", "repos/motional/nuplan-devkit"),
                     ("nuplan_devkit_v12_commit", "repos/motional/nuplan-devkit/commits/ce3c323af01c0d7ec5672f7832ef53f9c679aab0"),
                     ("TanitAD_repo", "repos/sayoood/TanitAD")):
        st, js = get_json(f"https://api.github.com/{api}")
        rec = {"status": st}
        if js and "private" in js:
            rec.update(private=js.get("private"), size_kb=js.get("size"), default_branch=js.get("default_branch"))
        if js and "sha" in js:
            rec.update(sha=js.get("sha"), date=(js.get("commit") or {}).get("committer", {}).get("date"))
        gh[key] = rec
    R["github"] = gh
    R["codeload_heads"] = {
        "navsim_v11_zip": head("https://codeload.github.com/autonomousvision/navsim/zip/"
                               "3e8291bfa89ff247231e0227778840cd0a036896", follow=True),
        "nuplan_devkit_v12_zip": head("https://codeload.github.com/motional/nuplan-devkit/zip/"
                                      "ce3c323af01c0d7ec5672f7832ef53f9c679aab0", follow=True),
        "drivezero_zip": head("https://codeload.github.com/XiaomiAutoL3/DriveZero/zip/"
                              "24959547edac549d64c34b2aab62d5c52599d078", follow=True),
    }
    R["seconds"] = round(time.time() - t0, 1)
    json.dump(R, open(os.path.join(OUT, "public_sources.json"), "w", encoding="utf-8"), indent=1)
    print(f"ZZPROBE_OK {R['seconds']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
