"""Independent corroboration of the load-bearing P3 number.

No numpy, no imported cut predicates, no shared code with
measure_train_agent_density.py -- a plain Python loop over the raw jsonl.
If this reads max 94 and 3,250 frames over 32, the table is not an artifact
of one implementation.

CONTROL: the same loop also recomputes n_frames / n_boxes / occ==0, which must
reproduce 433,040 / 12,122,129 / 0.4106 -- a loop that gets those wrong cannot
be trusted about the max either.
"""
import json, math
P = r"C:\Users\Admin\tanitad-data\joins\joins\train2400_agents.jsonl"
HALF = math.radians(60.0)
mx = 0; over32 = 0; over24 = 0; nb = 0; nf = 0; occ0 = 0; tot_in = 0
nearest_dropped_at_32 = None
for line in open(P, encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    nf += 1
    ag = json.loads(line).get("agents") or []
    nb += len(ag)
    rs = []
    for d in ag:
        cx = float(d["cx"]); cy = float(d["cy"])
        if int(d["occ"]) == 0:
            occ0 += 1
            if 0.0 <= cx <= 60.0 and abs(cy) <= 16.0:
                rs.append(math.hypot(cx, cy))
    n = len(rs)
    tot_in += n
    if n > mx:
        mx = n
    if n > 32:
        over32 += 1
        r = sorted(rs)[32]
        if nearest_dropped_at_32 is None or r < nearest_dropped_at_32:
            nearest_dropped_at_32 = r
    if n > 24:
        over24 += 1
print("n_frames        %d   (expect 433040   %s)" % (nf, nf == 433040))
print("n_boxes         %d   (expect 12122129 %s)" % (nb, nb == 12122129))
print("visible_frac    %.4f (expect 0.4106)" % (occ0 / nb))
print("infield_bevbox boxes %d (expect 1899481 %s)" % (tot_in, tot_in == 1899481))
print("MAX per frame   %d   (expect 94  %s)" % (mx, mx == 94))
print("frames > 32     %d   (expect 3250 %s)" % (over32, over32 == 3250))
print("frames > 24     %d   (expect 8614 %s)" % (over24, over24 == 8614))
print("nearest sacrificed target at N=32: %.4f m (expect 13.1)" % nearest_dropped_at_32)
