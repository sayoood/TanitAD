"""S2: read the central directories of the 12 nuPlan val camera zips and index the listed jpgs.

Run:  C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I s2_index.py
Network scope (PI 2026-10-19): the 12 val_camera zip central directories + public_set_val_sensor.txt,
same public bucket as pull_nuplan_splits.sh. Nothing else. Reuses the pinned fetch_front_camera.py
(entries(), central_directory()), with its rng/size_of replaced by the logged, guarded versions.
"""
from __future__ import annotations
import collections, concurrent.futures as cf, hashlib, importlib.util, json, os, re, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import netlog  # noqa: E402

BASE = "https://motional-nuplan.s3.ap-northeast-1.amazonaws.com/public/nuplan-v1.1"
VAL = BASE + "/sensor_blobs/val_set"
ROOT = "C:/Users/Admin/nuplan_u"
CDC = ROOT + "/cd_cache"
FL_JSON = "C:/Users/Admin/par18/unseen_s1b/raw/fetch_list_s2_s4_filtered.json"
FL_TXT = "C:/Users/Admin/par18/unseen_s1b/raw/fetch_list_s2_s4_filtered.txt"
UNAMES = "C:/Users/Admin/par18/l2fresh/raw/scope_unseen_stratum_U_names.json"
OUT = os.path.join(HERE, "..", "raw", "s2_cd_attempt_UNUSED_OUT.json")  # NEVER overwrite s2_index.json (the txt-based index)
LOGRE = re.compile(r"^\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}_veh-\d+_\d+_\d+$")
CAMS = ["CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0"]


def load_tool():
    os.environ["REFE_CD_CACHE"] = CDC  # never the tool's default (D:)
    spec = importlib.util.spec_from_file_location("fetch_front_camera", os.path.join(HERE, "fetch_front_camera.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.CD_CACHE == CDC, m.CD_CACHE
    m.rng = netlog.rng          # logged + guarded + <=4 parallel
    m.size_of = netlog.head_size
    return m


def split_name(name: str):
    parts = name.split("/")
    log = next((p for p in parts if LOGRE.match(p)), None)
    cam = next((p for p in parts if p.startswith("CAM_")), None)
    return log, cam, parts[-1]


def main():
    netlog.set_stage("s2")
    free0 = netlog.disk_guard("S2 start")
    os.makedirs(CDC, exist_ok=True)
    tool = load_tool()
    sha_tool = hashlib.sha256(open(os.path.join(HERE, "fetch_front_camera.py"), "rb").read()).hexdigest()

    # ---- the fetch list, checked here rather than assumed
    fl = json.load(open(FL_JSON, encoding="utf-8"))
    lines = open(FL_TXT, encoding="utf-8").read().splitlines()
    assert len(lines) == len(set(lines)) == fl["n_jpgs"] == 1944, (len(lines), fl["n_jpgs"])
    list_sha = hashlib.sha256("\n".join(lines).encode()).hexdigest()
    assert list_sha == fl["list_sha256"], (list_sha, fl["list_sha256"])
    want = {}
    for p in lines:
        log, cam, base = p.split("/")
        want[(log, cam, base)] = p
    list_logs = sorted({k[0] for k in want})
    listed_by_cam = collections.Counter(k[1] for k in want)
    U = json.load(open(UNAMES, encoding="utf-8"))["U_trainval_minus_navtrain"]
    Uset = set(U)
    assert len(Uset) == 1167
    assert set(fl["logs"]) <= Uset, "fetch-list logs not all in U"

    # ---- public_set_val_sensor.txt
    t0 = time.time()
    txt_url = VAL + "/public_set_val_sensor.txt"
    txt_size = netlog.head_size(txt_url)
    txt = netlog.rng(txt_url, 0, txt_size - 1)
    open(os.path.join(CDC, "public_set_val_sensor.txt"), "wb").write(txt)
    txt_lines = [l.strip() for l in txt.decode("utf-8", "replace").splitlines() if l.strip()]
    print(f"public_set_val_sensor.txt: {txt_size} B, {len(txt_lines)} non-empty lines; first: {txt_lines[:2]}")

    # ---- the 12 central directories, <=4 in parallel
    def fetch_cd(i):
        url = f"{VAL}/nuplan-v1.1_val_camera_{i}.zip"
        total = netlog.head_size(url)
        cd = tool.central_directory(url, total)
        return i, url, total, cd

    zips, found, zip_logs = {}, {}, {}
    cd_bytes_total = 0
    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        for i, url, total, cd in ex.map(fetch_cd, range(12)):
            es = tool.entries(cd)
            methods = collections.Counter(e["method"] for e in es if not e["name"].endswith("/"))
            logs_in = collections.defaultdict(lambda: collections.Counter())
            n_jpg = 0
            for e in es:
                if e["name"].endswith("/"):
                    continue
                log, cam, base = split_name(e["name"])
                if log is None or cam is None:
                    continue
                n_jpg += 1
                logs_in[log][cam] += 1
                k = (log, cam, base)
                if k in want:
                    if k in found:
                        found[k]["dup_in"] = found[k].get("dup_in", []) + [i]
                    else:
                        found[k] = {"zip": i, "name": e["name"], "offset": e["lho"], "csz": e["csz"],
                                    "usz": e["usz"], "method": e["method"], "crc32": e["crc"]}
            zips[i] = {"url": url, "total_bytes": total, "cd_bytes": len(cd), "n_entries": len(es),
                       "n_file_entries_with_log_and_cam": n_jpg, "methods": dict(methods),
                       "n_logs": len(logs_in), "n_listed_logs_present": len(set(logs_in) & set(list_logs)),
                       "n_U_logs_present": len(set(logs_in) & Uset)}
            zip_logs[i] = {lg: dict(c) for lg, c in logs_in.items()}
            cd_bytes_total += len(cd)
            print(f"zip {i}: {total/2**30:.2f} GB, CD {len(cd)/2**20:.1f} MiB, {len(es):,} entries, {len(logs_in)} logs, "
                  f"listed-logs present {zips[i]['n_listed_logs_present']}, U present {zips[i]['n_U_logs_present']}, methods {dict(methods)}")

    # ---- which zip holds each log (any camera), per-log camera completeness
    log2zips = collections.defaultdict(list)
    for i, d in zip_logs.items():
        for lg in d:
            log2zips[lg].append(i)
    U_in_any = sorted(lg for lg in Uset if lg in log2zips)
    U_multi = {lg: log2zips[lg] for lg in U_in_any if len(log2zips[lg]) > 1}
    listed_map = {}
    for lg in list_logs:
        zs = log2zips.get(lg, [])
        listed_map[lg] = {"zips": zs, "cams_in_zip": {str(i): zip_logs[i][lg] for i in zs}}
    not_in_any = [lg for lg in list_logs if lg not in log2zips]

    # ---- coverage per camera, and the per-jpg index
    cov = {c: {"listed": listed_by_cam[c], "found": 0, "missing": 0} for c in CAMS}
    missing, entries_out = [], []
    for k, path in sorted(want.items(), key=lambda kv: kv[1]):
        log, cam, base = k
        if k in found:
            cov[cam]["found"] += 1
            f = found[k]
            entries_out.append({"path": path, **f})
        else:
            cov[cam]["missing"] += 1
            missing.append(path)
    n_found = sum(c["found"] for c in cov.values())
    bytes_to_fetch = sum(e["csz"] for e in entries_out)
    methods_listed = collections.Counter(e["method"] for e in entries_out)
    per_zip_found = collections.Counter(e["zip"] for e in entries_out)

    st = netlog.stats()
    cd_dir_bytes = sum(os.path.getsize(os.path.join(CDC, f)) for f in os.listdir(CDC))
    res = {
        "what": "S2 index: nuPlan val camera zips (12) central directories -> where the 1,944 listed jpgs live",
        "evidence": "MEASURED (central directories read over HTTP range from the public nuPlan bucket)",
        "tool": {"file": "code/fetch_front_camera.py", "sha256": sha_tool,
                 "mirror_tip": "6fbd5baa0aa6c612e12a4b5d673c26ec856c4994 (origin/agent/arch-inf-20260803, fetched 2026-10-19)",
                 "path_at_tip": "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/code/fetch_front_camera.py",
                 "used": "entries(), central_directory() unchanged; rng()/size_of() replaced by code/netlog.py (logging, disk guard, <=4 parallel)"},
        "fetch_list": {"json": FL_JSON, "txt": FL_TXT, "n_jpgs": len(lines), "n_unique": len(set(lines)),
                       "list_sha256_recomputed": list_sha, "list_sha256_in_file": fl["list_sha256"],
                       "n_logs_field": fl["n_logs"], "n_logs_with_jpgs": len(list_logs), "n_tokens": fl["n_tokens"],
                       "unfilled": fl["unfilled"], "per_cam": dict(listed_by_cam)},
        "public_set_val_sensor_txt": {"url": txt_url, "bytes": txt_size, "n_nonempty_lines": len(txt_lines),
                                      "head": txt_lines[:5], "local": CDC + "/public_set_val_sensor.txt"},
        "zips": zips,
        "U_coverage": {"n_U": len(Uset), "n_U_in_any_zip": len(U_in_any),
                       "n_U_not_in_any_zip": len(Uset) - len(U_in_any),
                       "n_U_in_more_than_one_zip": len(U_multi), "U_multi": U_multi,
                       "n_logs_in_any_zip_total": len(log2zips), "n_zip_logs_not_in_U": len(set(log2zips) - Uset),
                       "U_not_in_any_zip": sorted(Uset - set(log2zips)),
                       "U_log_to_zips": {lg: log2zips[lg] for lg in U_in_any}},
        "listed_logs": {"n": len(list_logs), "n_not_in_any_zip": len(not_in_any), "not_in_any_zip": not_in_any,
                        "per_zip_jpgs_found": dict(sorted(per_zip_found.items())), "log_to_zip": listed_map},
        "coverage_per_camera": cov,
        "coverage_total": {"listed": len(lines), "found": n_found, "missing": len(missing)},
        "missing_paths": missing,
        "found_methods": {str(k): v for k, v in methods_listed.items()},
        "found_compressed_bytes_MEASURED": bytes_to_fetch,
        "duplicate_names_across_zips": [k for k, v in found.items() if "dup_in" in v][:20],
        "bytes_read_S2_MEASURED": {"payload_bytes_via_range": st["bytes"], "requests": st["requests"],
                                   "retries": st["retries"], "failures": st["failures"],
                                   "cd_cache_dir_bytes_on_disk": cd_dir_bytes, "cd_bytes_sum": cd_bytes_total,
                                   "note": "payload_bytes counts successful range bodies (tail 128 KiB + CD per zip + the txt); HEAD requests carry no body"},
        "c_free_bytes_before": free0, "c_free_bytes_after": netlog.disk_guard("S2 end"),
        "secs": round(time.time() - t0, 1),
        "entries": entries_out,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ("coverage_per_camera", "coverage_total", "bytes_read_S2_MEASURED", "found_methods")}, indent=1))
    print("U in any zip:", len(U_in_any), "of", len(Uset), "| listed logs not in any zip:", len(not_in_any))
    print("S2_DONE")


if __name__ == "__main__":
    main()
