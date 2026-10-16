import numpy as np, sys
# (label, workers, n_logs, n_twins, wall_s_of_pool)  -- MEASURED 2026-10-16 on the pod beside P2-s1 training, nice 15, loadavg ~7
runs = [("k1 mini 51tw/17logs", 2, 17, 51, 87.0), ("k2 mini 14tw/14logs", 2, 14, 14, 63.0), ("k3 mini 9tw/9logs", 2, 9, 9, 41.0),
        ("cost 12tw/4logs", 1, 4, 12, 51.0)]
extra = float(sys.argv[1]) if len(sys.argv) > 1 else None
if extra:
    runs.append(("cost2 40tw/1log", 1, 1, 40, extra))
A = np.array([[r[2], r[3], r[1]] for r in runs], float)      # n_logs, n_twins, workers (init cost per worker)
y = np.array([r[4] * r[1] for r in runs], float)             # worker-seconds
sol, res, rk, sv = np.linalg.lstsq(A, y, rcond=None)
print("t_log=%.2f s  t_twin=%.2f s  t_init_per_worker=%.2f s   (worker-seconds; lstsq over %d runs)" % (*sol, len(runs)))
for r, pred in zip(runs, A @ sol):
    print("  %-22s measured %.0f worker-s, model %.0f" % (r[0], r[4] * r[1], pred))
# the real job: three builds, one per rank
logs = {1: 980, 2: 667, 3: 496}; tw = {1: 32929, 2: 8090, 3: 3390}
for W in (6, 8):
    tot = 0
    for k in (1, 2, 3):
        ws = logs[k] * sol[0] + tw[k] * sol[1] + W * sol[2]
        tot += ws / W
    print("workers=%d: ESTIMATED wall %.0f s = %.2f h (all three ranks, perfect scaling, no log imbalance)" % (W, tot, tot / 3600))
ws_all = sum(logs[k] * sol[0] + tw[k] * sol[1] for k in (1, 2, 3))
print("total worker-seconds %.0f = %.1f worker-hours" % (ws_all, ws_all / 3600))
