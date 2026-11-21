"""S2 index (OFFLINE: reads only what S2 already fetched; makes no network request).

Inputs : C:/Users/Admin/nuplan_u/cd_cache/public_set_val_sensor.txt   (8,945 B, fetched in S2)
         raw/s2_archive_probe.json                                    (head + 2 windows per archive)
         logs/requests_*.jsonl                                        (every request made)
         the S1b fetch list + the 1,167 U names (read-only)
Output : raw/s2_index.json
Run    : python -I s2_index.py
"""
import collections, glob, hashlib, json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "raw")
TXT = "C:/Users/Admin/nuplan_u/cd_cache/public_set_val_sensor.txt"
FL_JSON = "C:/Users/Admin/par18/unseen_s1b/raw/fetch_list_s2_s4_filtered.json"
FL_TXT = "C:/Users/Admin/par18/unseen_s1b/raw/fetch_list_s2_s4_filtered.txt"
UNAMES = "C:/Users/Admin/par18/l2fresh/raw/scope_unseen_stratum_U_names.json"
TRAINVAL_DIR = "D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/trainval"   # read-only listing
CAMS = ["CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0"]
BASE = "https://motional-nuplan.s3.ap-northeast-1.amazonaws.com/public/nuplan-v1.1/sensor_blobs/val_set"

raw_txt = open(TXT, "rb").read()
groups, grp = {}, None
for l in raw_txt.decode("utf-8").splitlines():
    l = l.strip()
    m = re.match(r"File group:\s*(\d+)$", l)
    if m:
        grp = int(m.group(1)); groups[grp] = []
    elif l:
        groups[grp].append(l)
log2arch = collections.defaultdict(list)
for k, v in groups.items():
    for lg in v:
        log2arch[lg].append(k)

fl = json.load(open(FL_JSON, encoding="utf-8"))
lines = open(FL_TXT, encoding="utf-8").read().splitlines()
list_sha = hashlib.sha256("\n".join(lines).encode()).hexdigest()
assert len(lines) == len(set(lines)) == fl["n_jpgs"] and list_sha == fl["list_sha256"]
listed_logs = sorted({p.split("/")[0] for p in lines})
by_cam = collections.Counter(p.split("/")[1] for p in lines)
U = set(json.load(open(UNAMES, encoding="utf-8"))["U_trainval_minus_navtrain"])
tv = {f[:-3] for f in os.listdir(TRAINVAL_DIR) if f.endswith(".db")}
probe = json.load(open(os.path.join(RAW, "s2_archive_probe.json"), encoding="utf-8"))

# every request made, from the logs
per_log = {}
for fn in sorted(glob.glob(os.path.join(HERE, "..", "logs", "requests_*.jsonl"))):
    n = b = nh = fails = 0
    for l in open(fn, encoding="utf-8"):
        r = json.loads(l); n += 1
        if r["range"] == "HEAD":
            nh += 1
        if r["ok"] and r["range"] != "HEAD":
            b += r["bytes"]
        if not r["ok"]:
            fails += 1
    per_log[os.path.basename(fn)] = {"requests": n, "HEAD": nh, "range_body_bytes_ok": b, "failed_attempts": fails}
total_body = sum(v["range_body_bytes_ok"] for v in per_log.values())

in_txt = set(log2arch)
U_in = sorted(in_txt & U)
cov = {c: {"listed": by_cam[c], "found": 0, "missing": by_cam[c]} for c in CAMS}
res = {
    "what": "S2 index of the nuPlan val camera archives: which archive holds which log, and whether the 1,944 listed jpgs exist there",
    "evidence": "MEASURED (public_set_val_sensor.txt read from the bucket; corroborated by 3 range reads x 12 archives, 36 log sightings, 0 contradictions)",
    "FORMAT_FINDING": {
        "claim": "The 12 nuplan-v1.1_val_camera_{0..11}.zip objects are TAR archives (POSIX ustar), not zips: no EOCD, no central directory, no per-file CRC32, no compression.",
        "evidence": ["fetch_front_camera.py central_directory() raised 'no EOCD' on the last 128 KiB (code/s2_cd_attempt_FAILED.py, logs/requests_s2_cd_attempt.jsonl)",
                     "tail of archive 0 (128 KiB): 0 occurrences of the zip signatures PK05 06 / PK06 06 / PK01 02 / PK03 04; archive 0 head: 'nuplan-v1.1_val_camera_0/' + NULs, 'ustar' magic at byte 257 of every 512-byte header (typeflag '5' = dir, '0' = file; first jpg 172,075 B at offset 1536)",
                     "every one of the 12 archive sizes is a multiple of 512 (see archives below)"],
        "consequence": "S2's per-jpg (offset, compressed size, method, CRC32) cannot be read from an index: a tar has none, and tar members are stored uncompressed with no CRC. Locating a member needs a sequential header walk (header -> size -> next header). Not done: outside the approved scope.",
    },
    "fetch_list": {"n_jpgs": len(lines), "n_unique": len(set(lines)), "list_sha256_recomputed": list_sha,
                   "list_sha256_in_file": fl["list_sha256"], "n_logs_field": fl["n_logs"], "n_logs_with_jpgs": len(listed_logs),
                   "n_tokens": fl["n_tokens"], "per_cam": dict(by_cam), "unfilled": fl["unfilled"]},
    "public_set_val_sensor_txt": {"bytes": len(raw_txt), "sha256": hashlib.sha256(raw_txt).hexdigest(),
                                  "n_groups": len(groups), "n_logs": len(in_txt),
                                  "per_group": {str(k): len(v) for k, v in groups.items()},
                                  "logs_in_two_groups": [l for l, v in log2arch.items() if len(v) > 1]},
    "archives": {str(a["archive"]): {"url": f"{BASE}/nuplan-v1.1_val_camera_{a['archive']}.zip",
                                     "total_bytes": a["total_bytes"], "mod_512": a["total_mod_512"],
                                     "mod_10240": a["total_mod_10240"],
                                     "n_txt_logs": len(groups[a["archive"]]),
                                     "n_U_logs": len([l for l in groups[a["archive"]] if l in U]),
                                     "U_logs": [l for l in groups[a["archive"]] if l in U]} for a in probe["archives"]},
    "corroboration": {"windows_read": sum(len(a["reads"]) for a in probe["archives"]),
                      "log_sightings": sum(len(r["logs_seen"]) for a in probe["archives"] for r in a["reads"]),
                      "sightings_not_in_own_txt_group": probe["seen_not_in_own_group"],
                      "bytes_MEASURED": probe["bytes_read_MEASURED"],
                      "NOT_exhaustive": "a header sample cannot prove absence; that the txt is the complete archive listing is UNVERIFIED beyond (a) 0 contradictions in the sightings, (b) all 225 txt logs inside the 1,381 trainval DBs, (c) 214 of them outside U = exactly the trainval logs outside U, (d) density: 867 GB / 225 logs is about 3.9 GB per log for 8 cameras"},
    "U_coverage": {"n_U": len(U), "n_U_in_any_archive": len(U_in), "n_U_in_no_archive": len(U) - len(U_in),
                   "n_txt_logs_in_trainval_DBs": len(in_txt & tv), "n_txt_logs_not_in_U": len(in_txt - U),
                   "n_trainval_not_U": len(tv - U), "txt_covers_all_trainval_not_U": bool((tv - U) <= in_txt),
                   "U_logs_with_frames": {l: log2arch[l][0] for l in U_in}},
    "listed_logs": {"n": len(listed_logs), "n_in_any_archive": len(set(listed_logs) & in_txt),
                    "n_not_in_any_archive": len(set(listed_logs) - in_txt),
                    "log_to_archive": {l: (log2arch[l][0] if l in log2arch else None) for l in listed_logs}},
    "coverage_per_camera": cov,
    "coverage_total": {"listed": len(lines), "found": 0, "missing": len(lines)},
    "entries": [],
    "entries_note": "empty by construction: none of the 123 listed logs is in any archive, so no jpg of the list exists to be indexed",
    "bytes_read_S2_MEASURED": {"range_body_bytes_total": total_body, "per_log_file": per_log,
                               "note": "HEAD requests carry no body; contents: the 8,945 B txt, the 128 KiB archive tails read by the failed CD attempt, 12 x 4 KiB heads, 24 x 512 KiB windows"},
}
json.dump(res, open(os.path.join(RAW, "s2_index.json"), "w", encoding="utf-8"), indent=1)
print(json.dumps({k: res[k] for k in ("U_coverage", "coverage_total", "coverage_per_camera", "bytes_read_S2_MEASURED")}, indent=1)[:4000])
print("listed logs in any archive:", res["listed_logs"]["n_in_any_archive"], "of", res["listed_logs"]["n"])
