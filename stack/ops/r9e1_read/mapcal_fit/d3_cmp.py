"""d3_cmp.py <BASE verdict> <D3 verdict> <ARM> -- MM 2026-11-07. SAME iff the --map-beside-d3 re-run changed nothing but what it is meant to:
verdict, failing clauses, clause 3, the selector and every old beside_pedgefit row equal; the only reads that left are <ARM>.beside.lane / .edge, and
only if they were NOT-COMPUTABLE; map_inputs gains only beside_d3; beside_pedgefit gains only the two D3 rows. Prints SAME|DIFFER <why> | the rows."""
import json
import sys

D3_KEYS = ("lane_tiou05_0_20", "drv_bnd_rec03_0_20")


def fc(d):
    return sorted(int(c.get("clause", 0)) for c in d.get("failing_clauses", []))


def rows(d):
    return {r["key"]: r for r in (d.get("beside_pedgefit") or {}).get("rows", [])}


def main():
    a = json.load(open(sys.argv[1], encoding="utf-8"))
    b = json.load(open(sys.argv[2], encoding="utf-8"))
    arm = sys.argv[3]
    mv = {arm + ".beside.lane", arm + ".beside.edge"}
    moved = [r for r in a["reads"] if r.get("id") in mv]
    why = []
    if a.get("verdict") != b.get("verdict") or fc(a) != fc(b):
        why.append("verdict %s %s -> %s %s" % (a.get("verdict"), fc(a), b.get("verdict"), fc(b)))
    if [r for r in a["reads"] if r.get("id") not in mv] != b["reads"]:
        why.append("reads")
    if any("NOT-COMPUTABLE" not in (r.get("state"), r.get("status")) for r in moved):
        why.append("a moved read was computable")
    if a.get("clause3") != b.get("clause3"):
        why.append("clause3")
    ma, mb = a.get("map_inputs") or {}, b.get("map_inputs") or {}
    if not set(ma) <= set(mb) or set(mb) - set(ma) - {"beside_d3"} or any(ma[k] != mb[k] for k in ma):
        why.append("map_inputs")
    pa, pb = (a.get("beside_pedgefit") or {}), (b.get("beside_pedgefit") or {})
    if {k: v for k, v in pa.items() if k != "rows"} != {k: v for k, v in pb.items() if k != "rows"}:
        why.append("beside_pedgefit header/selector")
    ra, rb = rows(a), rows(b)
    if any(rb.get(k) != v for k, v in ra.items()) or set(rb) - set(ra) - set(D3_KEYS):
        why.append("beside rows")
    out = []
    for k in D3_KEYS:
        r = rb.get(k, {})
        out.append("%s %s ref %s arm %s delta %s [%s, %s] sep %s %s" % (k, r.get("state"), r.get("ref_point"), r.get("arm_point"), r.get("delta"),
                                                                      r.get("ci_lo"), r.get("ci_hi"), r.get("separated"), r.get("direction") or r.get("why")))
    print(("DIFFER " + ",".join(why)) if why else "SAME", b.get("verdict"), fc(b), "moved", len(moved), "|", " | ".join(out))


if __name__ == "__main__":
    main()
