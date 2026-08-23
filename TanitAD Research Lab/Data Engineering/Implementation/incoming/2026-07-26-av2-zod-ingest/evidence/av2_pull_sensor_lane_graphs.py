"""Pull ALL Argoverse-2 *sensor* lane graphs (train+val+test = 1000 logs) anonymously.

PI-approved 2026-07-26: the ~147 MiB metadata-only pull. NO imagery, NO tars.
Access is unauthenticated (`s3://argoverse` is public; the licensor's own instruction is
`s5cmd --no-sign-request`). Nothing is circumvented.

Verification is COMPLETENESS, not presence:
  * remote `<Size>` from ListObjectsV2 vs local `os.path.getsize`   (byte count)
  * remote `<ETag>` (== MD5 for single-part objects) vs local md5    (integrity)
  * `json.loads` on every file                                      (parse)
  * lane_segments / successors present                              (semantic)
A file that exists but is short, corrupt, or unparseable counts as a MISS.

DEV-BOX TRAP: bare curl here dies with CRYPT_E_NO_REVOCATION_CHECK and reports HTTP=000,
indistinguishable from an outage. `--ssl-no-revoke` is mandatory.

Usage:  python av2_pull_sensor_lane_graphs.py <outdir> [max_workers]
"""
from __future__ import annotations

import collections
import concurrent.futures as cf
import hashlib
import json
import os
import re
import subprocess
import sys
import time

BUCKET_URL = "https://s3.amazonaws.com/argoverse"
CURL = ["curl", "-sS", "--ssl-no-revoke", "-m", "180", "--retry", "3", "--retry-delay", "2"]
SPLITS = ("train", "val", "test")

_CONTENTS_RE = re.compile(r"<Contents>(.*?)</Contents>", re.S)
_PREFIX_RE = re.compile(r"<Prefix>([^<]+)</Prefix>")
_TOKEN_RE = re.compile(r"<NextContinuationToken>([^<]+)</NextContinuationToken>")
_TRUNC_RE = re.compile(r"<IsTruncated>([^<]+)</IsTruncated>")


def _q(s: str) -> str:
    return s.replace("/", "%2F").replace("+", "%2B").replace("=", "%3D").replace("&", "%26")


def s3_list(prefix: str, delimiter: bool, max_keys: int = 1000) -> str:
    """One page of ListObjectsV2, anonymous."""
    url = f"{BUCKET_URL}?list-type=2&prefix={_q(prefix)}&max-keys={max_keys}"
    if delimiter:
        url += "&delimiter=%2F"
    return subprocess.run(CURL + [url], capture_output=True, text=True).stdout


def s3_list_all(prefix: str, delimiter: bool, max_keys: int = 1000):
    """Full pagination. Returns (common_prefixes, contents) where contents are dicts."""
    prefixes, contents, token, pages = [], [], None, 0
    while True:
        url = f"{BUCKET_URL}?list-type=2&prefix={_q(prefix)}&max-keys={max_keys}"
        if delimiter:
            url += "&delimiter=%2F"
        if token:
            url += f"&continuation-token={_q(token)}"
        body = subprocess.run(CURL + [url], capture_output=True, text=True).stdout
        pages += 1
        for blk in _CONTENTS_RE.findall(body):
            k = re.search(r"<Key>([^<]+)</Key>", blk)
            sz = re.search(r"<Size>(\d+)</Size>", blk)
            et = re.search(r"<ETag>(?:&quot;|\")?([0-9a-fA-F\-]+)(?:&quot;|\")?</ETag>", blk)
            if k:
                contents.append(dict(key=k.group(1),
                                     size=int(sz.group(1)) if sz else None,
                                     etag=et.group(1) if et else None))
        body_wo = _CONTENTS_RE.sub("", body)
        for p in _PREFIX_RE.findall(body_wo):
            if p != prefix:
                prefixes.append(p)
        trunc = _TRUNC_RE.search(body)
        tok = _TOKEN_RE.search(body)
        if trunc and trunc.group(1) == "true" and tok:
            token = tok.group(1)
        else:
            break
    return prefixes, contents, pages


def md5_of(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- stage 1: logs
def list_logs() -> dict:
    out = {}
    for sp in SPLITS:
        pre = f"datasets/av2/sensor/{sp}/"
        prefixes, _, pages = s3_list_all(pre, delimiter=True)
        logs = sorted({p.rstrip("/").split("/")[-1] for p in prefixes})
        out[sp] = logs
        print(f"[list] sensor/{sp}: {len(logs)} logs ({pages} pages)", file=sys.stderr)
    return out


# ------------------------------------------------- stage 2: locate each map key
def locate_map(args):
    split, log = args
    pre = f"datasets/av2/sensor/{split}/{log}/map/"
    _, contents, _ = s3_list_all(pre, delimiter=False, max_keys=100)
    arch = [c for c in contents if "log_map_archive_" in c["key"] and c["key"].endswith(".json")]
    rec = dict(split=split, log=log, n_map_objects=len(contents),
               has_ground_height=any("ground_height_surface" in c["key"] for c in contents),
               has_img_Sim2_city=any("img_Sim2_city" in c["key"] for c in contents))
    if not arch:
        rec.update(status="NO_MAP_ARCHIVE_LISTED", key=None, remote_size=None, etag=None)
        return rec
    a = arch[0]
    city = re.search(r"____([A-Z]+)_city", a["key"])
    rec.update(status="LISTED", key=a["key"], remote_size=a["size"], etag=a["etag"],
               city=city.group(1) if city else None, n_archives=len(arch))
    return rec


# ------------------------------------------------------- stage 3: fetch + verify
def fetch_verify(rec, outdir):
    if rec["status"] != "LISTED":
        return rec
    dest = os.path.join(outdir, rec["split"], rec["log"] + ".json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    need = True
    if os.path.exists(dest) and os.path.getsize(dest) == rec["remote_size"]:
        need = False
    http = None
    if need:
        for attempt in range(3):
            r = subprocess.run(CURL + [f"{BUCKET_URL}/{rec['key']}", "-o", dest,
                                       "-w", "%{http_code}"], capture_output=True, text=True)
            http = r.stdout.strip()
            if http == "200" and os.path.exists(dest) \
                    and os.path.getsize(dest) == rec["remote_size"]:
                break
            time.sleep(1.5)
    rec["http"] = http
    if not os.path.exists(dest):
        rec["status"] = "DOWNLOAD_FAILED"
        return rec
    local = os.path.getsize(dest)
    rec["local_size"] = local
    rec["size_match"] = (local == rec["remote_size"])
    try:
        rec["md5_match"] = (md5_of(dest) == (rec["etag"] or "").lower())
    except OSError:
        rec["md5_match"] = False
    try:
        with open(dest, "r", encoding="utf-8") as f:
            d = json.load(f)
        rec["parse_ok"] = True
    except Exception as e:                                   # noqa: BLE001
        rec["parse_ok"] = False
        rec["parse_error"] = f"{type(e).__name__}: {e}"[:200]
        rec["status"] = "PARSE_FAILED"
        return rec
    ls = d.get("lane_segments") or {}
    rec["top_level_keys"] = sorted(d.keys())
    rec["n_lane_segments"] = len(ls)
    rec["n_successor_edges"] = sum(len(s.get("successors") or []) for s in ls.values())
    rec["n_with_centerline"] = sum(1 for s in ls.values() if "centerline" in s)
    rec["n_is_intersection"] = sum(1 for s in ls.values() if s.get("is_intersection"))
    rec["n_left_neighbor"] = sum(1 for s in ls.values() if s.get("left_neighbor_id") is not None)
    rec["n_right_neighbor"] = sum(1 for s in ls.values() if s.get("right_neighbor_id") is not None)
    rec["n_pedestrian_crossings"] = len(d.get("pedestrian_crossings") or {})
    rec["n_drivable_areas"] = len(d.get("drivable_areas") or {})
    ids = {int(s["id"]) for s in ls.values()}
    dang = res = 0
    outdeg = collections.Counter()
    indeg = collections.Counter()
    for s in ls.values():
        su = [int(x) for x in (s.get("successors") or [])]
        outdeg[len(su)] += 1
        for t in su:
            indeg[t] += 1
            if t in ids:
                res += 1
            else:
                dang += 1
    rec["successor_refs_resolved"] = res
    rec["successor_refs_dangling"] = dang
    rec["n_branch_points"] = sum(c for k, c in outdeg.items() if k >= 2)
    rec["n_merge_points"] = sum(1 for v in indeg.values() if v >= 2)
    rec["outdeg_hist"] = {str(k): v for k, v in sorted(outdeg.items())}
    ok = (rec["size_match"] and rec["parse_ok"] and rec["n_lane_segments"] > 0
          and rec["n_successor_edges"] > 0)
    rec["status"] = "OK" if ok else "INCOMPLETE"
    return rec


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "av2_sensor_lane_graphs"
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()

    idx_path = os.path.join(outdir, "_log_index.json")
    if os.path.exists(idx_path):
        logs = json.load(open(idx_path))
        print("[list] reusing cached log index", file=sys.stderr)
    else:
        logs = list_logs()
        json.dump(logs, open(idx_path, "w"), indent=2)

    tasks = [(sp, lg) for sp in SPLITS for lg in logs[sp]]
    print(f"[locate] {len(tasks)} logs", file=sys.stderr)

    loc_path = os.path.join(outdir, "_map_keys.json")
    if os.path.exists(loc_path):
        recs = json.load(open(loc_path))
        print("[locate] reusing cached map-key index", file=sys.stderr)
    else:
        recs = []
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            for i, r in enumerate(ex.map(locate_map, tasks), 1):
                recs.append(r)
                if i % 100 == 0:
                    print(f"  located {i}/{len(tasks)}  ({time.time()-t0:.0f}s)", file=sys.stderr)
        json.dump(recs, open(loc_path, "w"), indent=2)

    print(f"[fetch] starting, {workers} workers", file=sys.stderr)
    done = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fetch_verify, r, outdir) for r in recs]
        for i, f in enumerate(cf.as_completed(futs), 1):
            done.append(f.result())
            if i % 100 == 0:
                print(f"  fetched {i}/{len(futs)}  ({time.time()-t0:.0f}s)", file=sys.stderr)

    done.sort(key=lambda r: (SPLITS.index(r["split"]), r["log"]))
    json.dump(done, open(os.path.join(outdir, "_manifest.json"), "w"), indent=2)

    st = collections.Counter(r["status"] for r in done)
    ok = [r for r in done if r["status"] == "OK"]
    per_split = collections.Counter((r["split"], r["status"]) for r in done)
    summary = dict(
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        source="s3://argoverse/datasets/av2/sensor/{train,val,test}/*/map/log_map_archive_*.json",
        access="anonymous unsigned HTTPS GET (no account, no token, no Terms click)",
        attempted=len(done),
        status_counts=dict(st),
        per_split={f"{a}/{b}": c for (a, b), c in sorted(per_split.items())},
        logs_per_split={k: len(v) for k, v in logs.items()},
        yield_ok=len(ok),
        yield_rate=round(len(ok) / max(1, len(done)), 6),
        n_size_match=sum(1 for r in done if r.get("size_match")),
        n_md5_match=sum(1 for r in done if r.get("md5_match")),
        n_parse_ok=sum(1 for r in done if r.get("parse_ok")),
        total_bytes=sum(r.get("local_size") or 0 for r in done),
        total_lane_segments=sum(r.get("n_lane_segments") or 0 for r in done),
        total_successor_edges=sum(r.get("n_successor_edges") or 0 for r in done),
        total_branch_points=sum(r.get("n_branch_points") or 0 for r in done),
        total_is_intersection=sum(r.get("n_is_intersection") or 0 for r in done),
        total_with_centerline=sum(r.get("n_with_centerline") or 0 for r in done),
        total_left_neighbor=sum(r.get("n_left_neighbor") or 0 for r in done),
        total_right_neighbor=sum(r.get("n_right_neighbor") or 0 for r in done),
        successor_refs_resolved=sum(r.get("successor_refs_resolved") or 0 for r in done),
        successor_refs_dangling=sum(r.get("successor_refs_dangling") or 0 for r in done),
        maps_with_a_branch=sum(1 for r in ok if (r.get("n_branch_points") or 0) > 0),
        maps_with_an_intersection=sum(1 for r in ok if (r.get("n_is_intersection") or 0) > 0),
        cities=dict(collections.Counter(r.get("city") for r in ok)),
        top_level_key_sets=dict(collections.Counter(
            ",".join(r.get("top_level_keys") or []) for r in ok)),
        failures=[{k: r.get(k) for k in ("split", "log", "status", "http", "remote_size",
                                         "local_size", "parse_error")}
                  for r in done if r["status"] != "OK"],
        elapsed_s=round(time.time() - t0, 1),
    )
    json.dump(summary, open(os.path.join(outdir, "_summary.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}, indent=2))
    print(f"\nFAILURES ({len(summary['failures'])}):", file=sys.stderr)
    for f in summary["failures"][:50]:
        print("  ", f, file=sys.stderr)


if __name__ == "__main__":
    main()
