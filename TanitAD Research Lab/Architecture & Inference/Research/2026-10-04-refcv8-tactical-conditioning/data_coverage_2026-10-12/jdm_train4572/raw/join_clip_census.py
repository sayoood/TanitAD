"""Per-clip line census of the 3-D agent join (independent of the miner: prefix slice, no json parse of the body)."""
import lzma, json, sys, hashlib, collections
src, out = sys.argv[1], sys.argv[2]
P = '{"clip_id":"'
n = collections.Counter(); ne = collections.Counter(); bad = 0
with lzma.open(src, 'rt', encoding='utf-8') as fh:
    for line in fh:
        if not line.startswith(P) or line[12 + 36] != '"':
            bad += 1; continue
        cid = line[12:48]
        n[cid] += 1
        if '"agents":[]' not in line:
            ne[cid] += 1
rows = {hashlib.sha256(c.encode()).hexdigest()[:12]: {"lines": n[c], "lines_nonempty_agents": ne[c]} for c in n}
json.dump({"bad_prefix_lines": bad, "n_lines": sum(n.values()), "n_clips": len(n), "per_clip_sha12": rows}, open(out, 'w'))
print("CENSUS_DONE", sum(n.values()), len(n), bad)
