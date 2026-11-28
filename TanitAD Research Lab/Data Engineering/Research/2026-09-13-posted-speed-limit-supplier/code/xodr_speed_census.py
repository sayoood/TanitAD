"""Census of <speed> records in a NuRec/AlpaSim map.xodr: where they sit, and whether lane speeds add anything over road-type speeds."""
import collections, hashlib, json, sys
import xml.etree.ElementTree as ET
src = sys.argv[1]; out = sys.argv[2]
raw = open(src, "rb").read()
r = ET.fromstring(raw)
roads = r.findall("road")
assert len(roads) > 0, "no <road> elements read"          # same-breath positive control
rows, agree, dis, nolane = collections.Counter(), 0, 0, 0
length_by = collections.defaultdict(float)
for rd in roads:
    types = [(t.get("type"), t.find("speed").get("max"), t.find("speed").get("unit")) for t in rd.findall("type") if t.find("speed") is not None]
    lane = set((s.get("max"), s.get("unit")) for s in rd.iter("speed") if s.get("sOffset") is not None)
    rt = set((m, u) for _, m, u in types)
    if not lane: nolane += 1
    elif lane <= rt: agree += 1
    else: dis += 1
    for ty, m, u in types:
        rows[(ty, m, u)] += 1; length_by[(ty, m, u)] += float(rd.get("length"))
res = {"source": src, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
       "header": r.find("header").attrib, "n_roads": len(roads),
       "n_speed_records": sum(1 for _ in r.iter("speed")),
       "road_type_speed_counts": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in rows.items()},
       "road_length_m_by_type_speed": {f"{k[0]}|{k[1]}|{k[2]}": round(v, 1) for k, v in length_by.items()},
       "lane_speed_subset_of_road_type_speed": agree, "lane_speed_disagrees": dis, "roads_without_lane_speed": nolane}
json.dump(res, open(out, "w"), indent=1); print(json.dumps(res, indent=1))
