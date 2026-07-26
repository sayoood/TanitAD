"""Byte-verify LANE-LEVEL routable connectivity in ASAM OpenDRIVE (.xodr) datasets.

P2 in this program's standing pre-registration demands the EDGE RELATION byte-verified
in a real downloaded file -- not "it has an HD map", not "OpenDRIVE supports lanes".
The trap that killed ZOD was exactly this: 2-D marking polylines with no successor field.

The routable lane graph in OpenDRIVE is built from TWO mechanisms, and BOTH are checked:

  1. <road><lanes><laneSection><left|center|right><lane id><link><successor id=/>
     -> lane-to-lane continuation across laneSections and across a direct road link.
  2. <junction><connection incomingRoad connectingRoad contactPoint>
         <laneLink from=".." to=".."/>
     -> EXPLICIT lane-to-lane turn connectivity through an intersection. This is the
        S1 branch-selection structure: one incoming lane, several outgoing lanes.

Node identity used here: (road_id, lane_section_index, lane_id).

Usage: python opendrive_lanegraph_probe.py <out.json> <file.xodr> [more.xodr ...]
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import sys
import xml.etree.ElementTree as ET


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def analyse(path):
    tree = ET.parse(path)
    root = tree.getroot()

    header = root.find("header")
    hdr = dict(header.attrib) if header is not None else {}
    geo = header.find("geoReference") if header is not None else None
    hdr["geoReference"] = (geo.text or "").strip() if geo is not None else None

    roads = root.findall("road")
    junctions = root.findall("junction")

    lane_type_counts = collections.Counter()
    lane_nodes = set()                       # (road, ls_idx, lane_id)
    driving_nodes = set()
    lane_link_succ = 0                       # <lane><link><successor>
    lane_link_pred = 0
    lanes_with_succ = set()
    lanes_with_pred = set()
    n_lane_elems = 0

    road_len = 0.0
    signals = collections.Counter()
    n_signal_elems = 0
    objects = 0
    speed_records = 0
    road_ids = set()

    # --- pass over roads -------------------------------------------------- #
    for r in roads:
        rid = r.get("id")
        road_ids.add(rid)
        try:
            road_len += float(r.get("length") or 0.0)
        except ValueError:
            pass
        for sig in r.iter("signal"):
            n_signal_elems += 1
            signals[(sig.get("type"), sig.get("subtype"))] += 1
        for _ in r.iter("object"):
            objects += 1
        for _ in r.iter("speed"):
            speed_records += 1

        lanes = r.find("lanes")
        if lanes is None:
            continue
        for ls_idx, ls in enumerate(lanes.findall("laneSection")):
            for side in ("left", "center", "right"):
                grp = ls.find(side)
                if grp is None:
                    continue
                for ln in grp.findall("lane"):
                    n_lane_elems += 1
                    lid = ln.get("id")
                    ltype = ln.get("type")
                    lane_type_counts[ltype] += 1
                    node = (rid, ls_idx, lid)
                    lane_nodes.add(node)
                    if ltype == "driving":
                        driving_nodes.add(node)
                    link = ln.find("link")
                    if link is not None:
                        s = link.findall("successor")
                        p = link.findall("predecessor")
                        if s:
                            lane_link_succ += len(s)
                            lanes_with_succ.add(node)
                        if p:
                            lane_link_pred += len(p)
                            lanes_with_pred.add(node)

    # --- pass over junctions: the explicit lane-to-lane turn graph --------- #
    n_connections = 0
    n_lane_links = 0
    junction_types = collections.Counter()
    # incoming (road, lane) -> set of (connectingRoad, lane)
    turn_options = collections.defaultdict(set)
    junction_of_conn = {}
    conn_per_junction = collections.Counter()

    for j in junctions:
        jid = j.get("id")
        junction_types[j.get("type") or "default"] += 1
        for c in j.findall("connection"):
            n_connections += 1
            conn_per_junction[jid] += 1
            inc = c.get("incomingRoad")
            con = c.get("connectingRoad")
            junction_of_conn[con] = jid
            for ll in c.findall("laneLink"):
                n_lane_links += 1
                frm, to = ll.get("from"), ll.get("to")
                turn_options[(inc, frm)].add((con, to))

    out_deg = collections.Counter(len(v) for v in turn_options.values())
    branch_points = sum(1 for v in turn_options.values() if len(v) >= 2)

    # ⚠️ MERGES ARE NOT MEASURED HERE, deliberately, and the number below is NOT a
    # merge count. A merge is two incoming lanes landing on the same *outgoing road*
    # lane; the key available at this level is (connectingRoad, to-lane), which is
    # unique BY CONSTRUCTION (each junction connection gets its own connecting road).
    # Resolving the true outgoing lane requires following connectingRoad's own
    # <link><successor> and its last laneSection. Reported as UNVERIFIED so that a
    # structural zero is never read as "this map has no merges".
    incoming_of = collections.defaultdict(set)
    for k, vs in turn_options.items():
        for v in vs:
            incoming_of[v].add(k)
    merge_points = sum(1 for v in incoming_of.values() if len(v) >= 2)

    # --- road-level link topology (elementType road vs junction) ---------- #
    road_link_kind = collections.Counter()
    for r in roads:
        lk = r.find("link")
        if lk is None:
            road_link_kind["<no link>"] += 1
            continue
        for tag in ("predecessor", "successor"):
            e = lk.find(tag)
            if e is not None:
                road_link_kind[f"{tag}:{e.get('elementType')}"] += 1

    return {
        "file": os.path.basename(path),
        "bytes": os.path.getsize(path),
        "md5": md5(path),
        "header": hdr,
        "n_roads": len(roads),
        "total_road_length_m": round(road_len, 1),
        "n_junctions": len(junctions),
        "junction_types": dict(junction_types),
        "n_lane_elements": n_lane_elems,
        "n_distinct_lane_nodes": len(lane_nodes),
        "n_driving_lane_nodes": len(driving_nodes),
        "lane_type_counts": dict(lane_type_counts.most_common()),
        "LANE_LEVEL_link_successor_elements": lane_link_succ,
        "LANE_LEVEL_link_predecessor_elements": lane_link_pred,
        "lanes_with_successor": len(lanes_with_succ),
        "lanes_with_predecessor": len(lanes_with_pred),
        "pct_lanes_with_successor": round(100.0 * len(lanes_with_succ) / max(1, len(lane_nodes)), 2),
        "JUNCTION_connections": n_connections,
        "JUNCTION_laneLink_edges": n_lane_links,
        "distinct_incoming_lanes_at_junctions": len(turn_options),
        "BRANCH_POINTS_outdeg_ge2": branch_points,
        "out_degree_histogram": dict(sorted(out_deg.items())),
        "MERGE_POINTS__UNVERIFIED_see_code_note": merge_points,
        "max_connections_in_one_junction": max(conn_per_junction.values()) if conn_per_junction else 0,
        "road_link_element_types": dict(road_link_kind),
        "n_signal_elements": n_signal_elems,
        "signal_type_subtype_top20": [
            {"type": k[0], "subtype": k[1], "n": v} for k, v in signals.most_common(20)
        ],
        "n_object_elements": objects,
        "n_lane_speed_records": speed_records,
    }


def main():
    out = sys.argv[1]
    results = [analyse(p) for p in sys.argv[2:]]
    agg = {
        "probe": "opendrive-lane-level-connectivity",
        "date": "2026-07-26",
        "criterion": "P2 -- explicit lane-to-lane edge relation, byte-verified in a downloaded file",
        "maps": results,
        "TOTALS": {
            "n_maps": len(results),
            "n_roads": sum(r["n_roads"] for r in results),
            "n_junctions": sum(r["n_junctions"] for r in results),
            "n_lane_nodes": sum(r["n_distinct_lane_nodes"] for r in results),
            "n_driving_lane_nodes": sum(r["n_driving_lane_nodes"] for r in results),
            "lane_successor_elements": sum(r["LANE_LEVEL_link_successor_elements"] for r in results),
            "junction_laneLink_edges": sum(r["JUNCTION_laneLink_edges"] for r in results),
            "branch_points": sum(r["BRANCH_POINTS_outdeg_ge2"] for r in results),
            "merge_points__UNVERIFIED": sum(r["MERGE_POINTS__UNVERIFIED_see_code_note"] for r in results),
            "signal_elements": sum(r["n_signal_elements"] for r in results),
            "road_length_km": round(sum(r["total_road_length_m"] for r in results) / 1000.0, 2),
        },
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)
    for r in results:
        print(f"\n=== {r['file']}  ({r['bytes']:,} B)")
        print(f"  roads={r['n_roads']:6d}  junctions={r['n_junctions']:4d}  "
              f"road_km={r['total_road_length_m']/1000:.2f}")
        print(f"  lane nodes={r['n_distinct_lane_nodes']:6d}  driving={r['n_driving_lane_nodes']:6d}")
        print(f"  LANE <link><successor> elems={r['LANE_LEVEL_link_successor_elements']:6d}  "
              f"lanes w/ succ={r['lanes_with_successor']} ({r['pct_lanes_with_successor']}%)")
        print(f"  JUNCTION connections={r['JUNCTION_connections']:5d}  "
              f"laneLink edges={r['JUNCTION_laneLink_edges']:5d}")
        print(f"  BRANCH points (out-deg>=2)={r['BRANCH_POINTS_outdeg_ge2']:5d}   "
              f"out-deg hist={r['out_degree_histogram']}")
        print(f"  MERGE points=UNVERIFIED   signals={r['n_signal_elements']}")
        print(f"  lane types: {dict(list(r['lane_type_counts'].items())[:8])}")
        print(f"  geoReference: {str(r['header'].get('geoReference'))[:120]}")
    print("\nTOTALS:", json.dumps(agg["TOTALS"], indent=2))


if __name__ == "__main__":
    main()
