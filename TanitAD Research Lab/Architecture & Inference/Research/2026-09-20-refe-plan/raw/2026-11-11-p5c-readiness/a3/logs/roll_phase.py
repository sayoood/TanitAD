import re, sys, datetime as dt
for d in sys.argv[1:]:
    lines = open(d + "/chain.log", encoding="utf-8", errors="replace").read().splitlines()
    fetch_t, tot_s, tot_rows, nb = {}, 0.0, 0, 0
    prev = None
    def ts(s):
        h, m, x = map(int, s.split(":")); return h * 3600 + m * 60 + x
    for ln in lines:
        m = re.match(r"(\d\d:\d\d:\d\d) batch (\d+): fetch rc=0", ln)
        if m: fetch_t[m.group(2)] = ts(m.group(1)); continue
        m = re.match(r"(\d\d:\d\d:\d\d) batch (\d+): roll rcs=\[.*?\] rows_ok=(\d+)", ln)
        if m and m.group(2) in fetch_t:
            dtv = ts(m.group(1)) - fetch_t[m.group(2)]
            if dtv < 0: dtv += 86400
            if dtv > 4 * 3600: continue
            tot_s += dtv; tot_rows += int(m.group(3)); nb += 1
    print(d, "batches", nb, "rows", tot_rows, "roll_phase_wall_h %.2f" % (tot_s / 3600), "wall_s_per_row %.3f" % (tot_s / max(1, tot_rows)))
