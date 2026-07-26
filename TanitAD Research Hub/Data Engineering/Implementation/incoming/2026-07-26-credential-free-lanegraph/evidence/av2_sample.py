"""Sample AV2 map JSONs anonymously and measure lane-graph connectivity.

Metadata-scale only: N small JSON files (~60-130 KB each). No bulk download.
Raw bytes stay in the scratchpad; only DERIVED stats are staged into the repo
(AV2 data is CC-BY-NC-SA -> we ship pointers + derived features, never source bytes).
"""
import json, re, subprocess, sys, collections, statistics, os

BUCKET = "https://s3.amazonaws.com/argoverse"
CURL = ["curl", "-sS", "--ssl-no-revoke", "-m", "90"]


def s3_list(prefix, delimiter="/", max_keys=1000, token=None):
    url = f"{BUCKET}?list-type=2&prefix={prefix.replace('/', '%2F')}&max-keys={max_keys}"
    if delimiter:
        url += "&delimiter=%2F"
    if token:
        from urllib.parse import quote
        url += "&continuation-token=" + quote(token, safe="")
    out = subprocess.run(CURL + [url], capture_output=True, text=True).stdout
    prefixes = re.findall(r"<Prefix>([^<]+)</Prefix>", out)
    keys = re.findall(r"<Key>([^<]+)</Key>", out)
    sizes = [int(s) for s in re.findall(r"<Size>(\d+)</Size>", out)]
    tok = re.findall(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", out)
    return prefixes, keys, sizes, (tok[0] if tok else None), out


def s3_get(key, dest):
    r = subprocess.run(
        CURL + [f"{BUCKET}/{key}", "-o", dest, "-w", "%{http_code}"],
        capture_output=True, text=True)
    return r.stdout.strip()


def main(split="val", n=50):
    os.makedirs("av2_maps", exist_ok=True)
    pre, _, _, _, _ = s3_list(f"datasets/av2/motion-forecasting/{split}/", max_keys=n + 5)
    scen = [p for p in pre if p.rstrip("/").split("/")[-1] != split][:n]
    print(f"sampled {len(scen)} scenario dirs from motion-forecasting/{split}", file=sys.stderr)

    stats = dict(files=[], total_bytes=0)
    agg = dict(
        n_maps=0, n_lane_segments=0, n_with_successor=0, n_with_predecessor=0,
        n_is_intersection=0, n_left_neighbor=0, n_right_neighbor=0,
        n_edges_successor=0, n_edges_predecessor=0,
        n_pedestrian_crossings=0, n_drivable_areas=0,
        lane_types=collections.Counter(), mark_types=collections.Counter(),
        top_level_keys=collections.Counter(), field_names=collections.Counter(),
        branch_hist=collections.Counter(),   # out-degree of lane graph
        merge_hist=collections.Counter(),    # in-degree
        maps_with_a_branch=0, maps_with_an_intersection=0,
        dangling_successor_refs=0, resolved_successor_refs=0,
        cycles_detected_maps=0,
    )
    for sp in scen:
        sid = sp.rstrip("/").split("/")[-1]
        key = f"{sp}log_map_archive_{sid}.json"
        dest = f"av2_maps/{sid}.json"
        if not os.path.exists(dest):
            code = s3_get(key, dest)
            if code != "200":
                print(f"  MISS {sid} http={code}", file=sys.stderr)
                continue
        sz = os.path.getsize(dest)
        stats["files"].append(dict(scenario=sid, key=key, bytes=sz))
        stats["total_bytes"] += sz
        d = json.load(open(dest))
        agg["n_maps"] += 1
        for k in d:
            agg["top_level_keys"][k] += 1
        agg["n_pedestrian_crossings"] += len(d.get("pedestrian_crossings", {}) or {})
        agg["n_drivable_areas"] += len(d.get("drivable_areas", {}) or {})
        ls = d.get("lane_segments", {}) or {}
        ids = set()
        succ_map = {}
        indeg = collections.Counter()
        has_branch = False
        has_inter = False
        for lid, seg in ls.items():
            agg["n_lane_segments"] += 1
            for f in seg:
                agg["field_names"][f] += 1
            ids.add(int(seg["id"]))
            su = seg.get("successors") or []
            pr = seg.get("predecessors") or []
            succ_map[int(seg["id"])] = [int(x) for x in su]
            agg["n_edges_successor"] += len(su)
            agg["n_edges_predecessor"] += len(pr)
            if su:
                agg["n_with_successor"] += 1
            if pr:
                agg["n_with_predecessor"] += 1
            agg["branch_hist"][len(su)] += 1
            for s in su:
                indeg[int(s)] += 1
            if len(su) > 1:
                has_branch = True
            if seg.get("is_intersection"):
                agg["n_is_intersection"] += 1
                has_inter = True
            if seg.get("left_neighbor_id") is not None:
                agg["n_left_neighbor"] += 1
            if seg.get("right_neighbor_id") is not None:
                agg["n_right_neighbor"] += 1
            agg["lane_types"][seg.get("lane_type")] += 1
            agg["mark_types"][seg.get("left_lane_mark_type")] += 1
            agg["mark_types"][seg.get("right_lane_mark_type")] += 1
        for v in indeg.values():
            agg["merge_hist"][v] += 1
        # referential integrity of the successor relation
        for src, sus in succ_map.items():
            for s in sus:
                if s in ids:
                    agg["resolved_successor_refs"] += 1
                else:
                    agg["dangling_successor_refs"] += 1
        if has_branch:
            agg["maps_with_a_branch"] += 1
        if has_inter:
            agg["maps_with_an_intersection"] += 1
        # cycle detection on the successor graph (roundabout / loop proxy)
        colour = {}

        def dfs(u):
            colour[u] = 1
            for v in succ_map.get(u, []):
                if v not in succ_map:
                    continue
                if colour.get(v) == 1:
                    return True
                if colour.get(v, 0) == 0 and dfs(v):
                    return True
            colour[u] = 2
            return False

        if any(colour.get(u, 0) == 0 and dfs(u) for u in list(succ_map)):
            agg["cycles_detected_maps"] += 1

    agg["lane_types"] = dict(agg["lane_types"])
    agg["mark_types"] = dict(agg["mark_types"])
    agg["top_level_keys"] = dict(agg["top_level_keys"])
    agg["field_names"] = dict(agg["field_names"])
    agg["branch_hist"] = {str(k): v for k, v in sorted(agg["branch_hist"].items())}
    agg["merge_hist"] = {str(k): v for k, v in sorted(agg["merge_hist"].items())}
    out = dict(sample=stats, aggregate=agg)
    json.dump(out, open("av2_lanegraph_stats.json", "w"), indent=2)
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "val",
         int(sys.argv[2]) if len(sys.argv) > 2 else 50)
