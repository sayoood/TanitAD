"""Sample AV2 *sensor* per-log map archives (full log maps, not scenario-cropped).

The sensor split is the one that carries imagery (7 ring cameras + 2 stereo), so it is the
split that matters for a vision world model. Full log maps also let us test cycles
(roundabout proxy) and successor referential integrity without a crop artefact.
Metadata-scale only: N JSON files ~100-200 KB each.
"""
import json, re, subprocess, sys, collections, os

BUCKET = "https://s3.amazonaws.com/argoverse"
CURL = ["curl", "-sS", "--ssl-no-revoke", "-m", "120"]


def s3_list_raw(prefix, delimiter=True, max_keys=1000):
    url = f"{BUCKET}?list-type=2&prefix={prefix.replace('/', '%2F')}&max-keys={max_keys}"
    if delimiter:
        url += "&delimiter=%2F"
    return subprocess.run(CURL + [url], capture_output=True, text=True).stdout


def main(split="val", n=25):
    os.makedirs("av2_sensor_maps", exist_ok=True)
    out = s3_list_raw(f"datasets/av2/sensor/{split}/", max_keys=n + 5)
    logs = [p for p in re.findall(r"<Prefix>([^<]+)</Prefix>", out)
            if p.rstrip("/").split("/")[-1] != split][:n]
    print(f"sampled {len(logs)} sensor logs from sensor/{split}", file=sys.stderr)

    agg = dict(n_maps=0, n_lane_segments=0, n_is_intersection=0,
               n_edges_successor=0, n_left_neighbor=0, n_right_neighbor=0,
               n_pedestrian_crossings=0, n_drivable_areas=0,
               branch_hist=collections.Counter(), merge_hist=collections.Counter(),
               top_level_keys=collections.Counter(), field_names=collections.Counter(),
               dangling_successor_refs=0, resolved_successor_refs=0,
               maps_with_cycle=0, maps_with_a_branch=0, maps_with_an_intersection=0,
               cities=collections.Counter(), has_centerline_field=0,
               ground_height_raster_present=0, img_Sim2_city_present=0,
               total_bytes=0, files=[])

    for lp in logs:
        lid = lp.rstrip("/").split("/")[-1]
        mout = s3_list_raw(f"{lp}map/", delimiter=False, max_keys=50)
        keys = re.findall(r"<Key>([^<]+)</Key>", mout)
        arch = [k for k in keys if "log_map_archive_" in k]
        if not arch:
            print(f"  MISS no archive for {lid}", file=sys.stderr)
            continue
        agg["ground_height_raster_present"] += int(any("ground_height_surface" in k for k in keys))
        agg["img_Sim2_city_present"] += int(any("img_Sim2_city" in k for k in keys))
        key = arch[0]
        m = re.search(r"____([A-Z]+)_city", key)
        if m:
            agg["cities"][m.group(1)] += 1
        dest = f"av2_sensor_maps/{lid}.json"
        if not os.path.exists(dest):
            r = subprocess.run(CURL + [f"{BUCKET}/{key}", "-o", dest, "-w", "%{http_code}"],
                               capture_output=True, text=True)
            if r.stdout.strip() != "200":
                print(f"  MISS {lid} http={r.stdout.strip()}", file=sys.stderr)
                continue
        agg["total_bytes"] += os.path.getsize(dest)
        agg["files"].append(dict(log=lid, key=key, bytes=os.path.getsize(dest)))
        d = json.load(open(dest))
        agg["n_maps"] += 1
        for k in d:
            agg["top_level_keys"][k] += 1
        agg["n_pedestrian_crossings"] += len(d.get("pedestrian_crossings", {}) or {})
        agg["n_drivable_areas"] += len(d.get("drivable_areas", {}) or {})
        ls = d.get("lane_segments", {}) or {}
        succ, ids, indeg = {}, set(), collections.Counter()
        hb = hi = False
        for seg in ls.values():
            agg["n_lane_segments"] += 1
            for f in seg:
                agg["field_names"][f] += 1
            if "centerline" in seg:
                agg["has_centerline_field"] += 1
            i = int(seg["id"]); ids.add(i)
            su = [int(x) for x in (seg.get("successors") or [])]
            succ[i] = su
            agg["n_edges_successor"] += len(su)
            agg["branch_hist"][len(su)] += 1
            for s in su:
                indeg[s] += 1
            if len(su) > 1:
                hb = True
            if seg.get("is_intersection"):
                agg["n_is_intersection"] += 1; hi = True
            if seg.get("left_neighbor_id") is not None:
                agg["n_left_neighbor"] += 1
            if seg.get("right_neighbor_id") is not None:
                agg["n_right_neighbor"] += 1
        for v in indeg.values():
            agg["merge_hist"][v] += 1
        for src, sus in succ.items():
            for s in sus:
                if s in ids:
                    agg["resolved_successor_refs"] += 1
                else:
                    agg["dangling_successor_refs"] += 1
        agg["maps_with_a_branch"] += int(hb)
        agg["maps_with_an_intersection"] += int(hi)
        colour = {}
        sys.setrecursionlimit(50000)

        def dfs(u):
            colour[u] = 1
            for v in succ.get(u, []):
                if v not in succ:
                    continue
                if colour.get(v) == 1:
                    return True
                if colour.get(v, 0) == 0 and dfs(v):
                    return True
            colour[u] = 2
            return False
        agg["maps_with_cycle"] += int(any(colour.get(u, 0) == 0 and dfs(u) for u in list(succ)))

    for k in ("branch_hist", "merge_hist"):
        agg[k] = {str(a): b for a, b in sorted(agg[k].items())}
    for k in ("top_level_keys", "field_names", "cities"):
        agg[k] = dict(agg[k])
    json.dump(agg, open("av2_sensor_lanegraph_stats.json", "w"), indent=2)
    p = {k: v for k, v in agg.items() if k != "files"}
    print(json.dumps(p, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "val",
         int(sys.argv[2]) if len(sys.argv) > 2 else 25)
