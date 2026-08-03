#!/usr/bin/env python3
"""ADVERSARIAL independent re-derivation of the junction option counts.

Fresh ElementTree parse of map.xodr. Does NOT import xodr_map.py or
junction_probe.py. Pure topology: <junction><connection><laneLink>.
"""
import sys, json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

path = sys.argv[1]
queries = json.loads(sys.argv[2])   # [[junction_id, incoming_road, incoming_lane], ...]

root = ET.parse(path).getroot()
roads = {r.get("id"): r for r in root.findall("road")}
juncs = {j.get("id"): j for j in root.findall("junction")}

print("roads=%d junctions=%d" % (len(roads), len(juncs)))
nconn = sum(len(j.findall("connection")) for j in juncs.values())
nll = sum(len(c.findall("laneLink")) for j in juncs.values() for c in j.findall("connection"))
print("connections=%d laneLinks=%d" % (nconn, nll))

# how many connections carry NO laneLink at all?  a permissive filter would
# admit those for ANY lane
empty = [(jid, c.get("incomingRoad"), c.get("connectingRoad"))
         for jid, j in juncs.items() for c in j.findall("connection")
         if not c.findall("laneLink")]
print("connections with ZERO laneLink (would be admitted for any lane): %d" % len(empty))
if empty:
    print("   e.g.", empty[:6])

for jid, inc_r, inc_l in queries:
    j = juncs.get(str(jid))
    print("\n=== junction %s  incoming road %s lane %s ===" % (jid, inc_r, inc_l))
    if j is None:
        print("  JUNCTION NOT FOUND")
        continue
    conns = j.findall("connection")
    print("  total connections in this junction: %d" % len(conns))
    matching_road = []
    for c in conns:
        if c.get("incomingRoad") != str(inc_r):
            continue
        links = [(l.get("from"), l.get("to")) for l in c.findall("laneLink")]
        matching_road.append((c.get("connectingRoad"), c.get("contactPoint"), links))
    print("  connections with incomingRoad=%s : %d" % (inc_r, len(matching_road)))
    for cr, cp, links in matching_road:
        hit = any(a == str(inc_l) for a, _ in links)
        print("    connectingRoad=%-5s contactPoint=%-6s laneLinks=%-28s lane%s_match=%s%s"
              % (cr, cp, links, inc_l, hit, "  <-- NO LANELINK (permissive)" if not links else ""))
    strict = sorted({cr for cr, cp, links in matching_road
                     if any(a == str(inc_l) for a, _ in links)})
    permissive = sorted({cr for cr, cp, links in matching_road
                         if (not links) or any(a == str(inc_l) for a, _ in links)})
    anylane = sorted({cr for cr, cp, links in matching_road})
    print("  STRICT   options for lane %s : %d %s" % (inc_l, len(strict), strict))
    print("  PERMISSIVE (empty laneLink counts): %d %s" % (len(permissive), permissive))
    print("  ANY-LANE options for road %s   : %d %s" % (inc_r, len(anylane), anylane))
    # reverse check: is this road ALSO reachable as a connectingRoad here?
    rev = sorted({c.get("incomingRoad") for c in conns
                  if c.get("connectingRoad") == str(inc_r)})
    if rev:
        print("  NOTE road %s also appears as a connectingRoad fed by %s" % (inc_r, rev))
