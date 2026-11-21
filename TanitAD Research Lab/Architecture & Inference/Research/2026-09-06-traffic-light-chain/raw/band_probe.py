"""+/-2.0 s band coverage of the v7.2 tactical GT. ASCII-only."""
import gzip, json, sys, collections

def load(p):
    with gzip.open(p, "rt", encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]

for path in sys.argv[1:]:
    recs = load(path)
    name = path.rsplit("/", 1)[-1]
    print("=" * 72)
    print("BLOB:", name, " n_records:", len(recs))

    # one record per clip?
    clips = collections.Counter(r["clip_id"] for r in recs)
    print("distinct clip_ids: %d ; max records per clip: %d"
          % (len(clips), max(clips.values())))

    t0 = [float(r.get("t0_s", float("nan"))) for r in recs]
    span = [float((r.get("horizon") or {}).get("recording_span_s", float("nan")))
            for r in recs]
    avail = [float((r.get("horizon") or {}).get("available_s", float("nan")))
             for r in recs]
    print("t0_s               :", collections.Counter(t0).most_common(5))
    sp = sorted(x for x in span if x == x)
    av = sorted(x for x in avail if x == x)
    if sp:
        print("recording_span_s   : n=%d min=%.1f p50=%.1f max=%.1f"
              % (len(sp), sp[0], sp[len(sp)//2], sp[-1]))
    if av:
        print("available_s        : n=%d min=%.1f p50=%.1f max=%.1f"
              % (len(av), av[0], av[len(av)//2], av[-1]))

    bands = collections.Counter(
        json.dumps((r.get("bands") or {}).get("tactical_s")) for r in recs)
    print("bands.tactical_s   :", bands.most_common(3))

    # ---- the +/-2.0 s window around the single anchor -------------------
    # The record is anchored at t0_s. A scored frame at clip time T carries
    # tactical GT only if it falls inside the anchored window.
    HALF = 2.0
    covered, total = 0.0, 0.0
    for r in recs:
        s = float((r.get("horizon") or {}).get("recording_span_s", 0.0))
        if s <= 0:
            continue
        t = float(r.get("t0_s", 0.0))
        lo, hi = max(0.0, t - HALF), min(s, t + HALF)
        covered += max(0.0, hi - lo)
        total += s
    print("+/-2.0 s ANCHOR WINDOW vs full recording:")
    print("   covered %.1f s of %.1f s total  = %.2f %% of recorded clip time"
          % (covered, total, 100.0 * covered / total if total else 0.0))
    print("   i.e. %.1f of every 100 s of these clips carries NO v7.2 tactical GT"
          % (100.0 - 100.0 * covered / total if total else 0.0))

    # ---- and against the 4 s tactical band [t0+2, t0+6] ----------------
    cov2, tot2 = 0.0, 0.0
    for r in recs:
        s = float((r.get("horizon") or {}).get("recording_span_s", 0.0))
        if s <= 0:
            continue
        t = float(r.get("t0_s", 0.0))
        tb = (r.get("bands") or {}).get("tactical_s") or [2.0, 6.0]
        lo, hi = max(0.0, t + tb[0]), min(s, t + tb[1])
        cov2 += max(0.0, hi - lo)
        tot2 += s
    print("TACTICAL BAND [t0+%.1f, t0+%.1f] vs full recording:" % (tb[0], tb[1]))
    print("   covered %.1f s of %.1f s total  = %.2f %%"
          % (cov2, tot2, 100.0 * cov2 / tot2 if tot2 else 0.0))

    # ---- traffic-light record coverage ---------------------------------
    tl = sum(1 for r in recs
             if any("TRAFFIC_LIGHT" in k
                    for k in ((r.get("g_tac") or {}).get("goals") or {})))
    print("records with a TRAFFIC_LIGHT_* goal: %d/%d" % (tl, len(recs)))
    tok = collections.Counter()
    for r in recs:
        for k in ((r.get("g_tac") or {}).get("goals") or {}):
            if "TRAFFIC_LIGHT" in k:
                tok[k] += 1
    for k in sorted(tok):
        print("   %-32s %d/%d" % (k, tok[k], len(recs)))
    print()
