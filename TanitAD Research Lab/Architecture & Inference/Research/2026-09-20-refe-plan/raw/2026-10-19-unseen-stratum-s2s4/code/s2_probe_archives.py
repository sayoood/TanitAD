"""S2b: the 'val camera zips' are TAR archives (ustar headers, no central directory). Verify the
public_set_val_sensor.txt log->archive map against the archives themselves with a few tiny range
reads: the first 4 KiB of each archive and two 512 KiB windows (at ~1/3 and ~2/3) per archive.
~13 MB in total. Every header found (512-aligned, 'ustar' at +257) names a log; each must be in the
txt's group for that archive and in no other group."""
import concurrent.futures as cf, json, os, re, sys, collections
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import netlog
netlog.set_stage("s2_probe")
VAL = "https://motional-nuplan.s3.ap-northeast-1.amazonaws.com/public/nuplan-v1.1/sensor_blobs/val_set"
TXT = "C:/Users/Admin/nuplan_u/cd_cache/public_set_val_sensor.txt"
LOGRE = re.compile(r"(\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}_veh-\d+_\d+_\d+)")
WIN = 512 * 1024

def parse_txt():
    g, grp = {}, None
    for l in open(TXT, encoding="utf-8").read().splitlines():
        l = l.strip()
        m = re.match(r"File group:\s*(\d+)", l)
        if m: grp = int(m.group(1)); g[grp] = []
        elif l: g[grp].append(l)
    return g

def headers(buf, base):
    out = []
    for off in range(0, len(buf) - 511, 512):
        b = buf[off:off + 512]
        if b[257:262] == b"ustar":
            name = b[:100].split(b"\0")[0].decode("utf-8", "replace")
            try: size = int(b[124:135].split(b"\0")[0].strip() or b"0", 8)
            except ValueError: size = None
            out.append({"offset": base + off, "name": name, "size": size, "type": b[156:157].decode()})
    return out

def probe(i, total):
    url = f"{VAL}/nuplan-v1.1_val_camera_{i}.zip"
    res = {"archive": i, "total_bytes": total, "total_mod_512": total % 512, "total_mod_10240": total % 10240, "reads": []}
    plan = [(0, 4095, "head")] + [(((total * k // 3) // 512) * 512, ((total * k // 3) // 512) * 512 + WIN - 1, f"mid{k}") for k in (1, 2)]
    for a, b, tag in plan:
        buf = netlog.rng(url, a, b)
        hs = headers(buf, a)
        logs = sorted({m.group(1) for h in hs for m in [LOGRE.search(h["name"])] if m})
        res["reads"].append({"tag": tag, "range": [a, b], "n_headers": len(hs), "logs_seen": logs,
                             "first_names": [h["name"] for h in hs[:3]], "sizes_sample": [h["size"] for h in hs[:6]]})
    return res

def main():
    g = parse_txt()
    log2grp = collections.defaultdict(list)
    for k, v in g.items():
        for l in v: log2grp[l].append(k)
    sizes = {}
    def one(i):
        url = f"{VAL}/nuplan-v1.1_val_camera_{i}.zip"
        return probe(i, netlog.head_size(url))
    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(one, range(12)))
    bad = []
    for r in results:
        for rd in r["reads"]:
            rd["in_own_txt_group"] = [l for l in rd["logs_seen"] if r["archive"] in log2grp.get(l, [])]
            rd["NOT_in_own_txt_group"] = [l for l in rd["logs_seen"] if r["archive"] not in log2grp.get(l, [])]
            bad += [(r["archive"], l) for l in rd["NOT_in_own_txt_group"]]
            print(r["archive"], rd["tag"], "hdrs", rd["n_headers"], "logs", len(rd["logs_seen"]), "in-own-group", len(rd["in_own_txt_group"]), "NOT", len(rd["NOT_in_own_txt_group"]))
    st = netlog.stats()
    out = {"txt_groups": {str(k): len(v) for k, v in g.items()}, "n_txt_logs": len(log2grp),
           "any_log_in_two_groups": [l for l, v in log2grp.items() if len(v) > 1],
           "archives": results, "seen_not_in_own_group": bad, "bytes_read_MEASURED": st["bytes"], "requests": st["requests"]}
    json.dump(out, open(os.path.join(HERE, "..", "raw", "s2_archive_probe.json"), "w", encoding="utf-8"), indent=1)
    print("seen-but-not-in-own-txt-group:", bad, "| bytes", st["bytes"], "| requests", st["requests"])

if __name__ == "__main__":
    main()
