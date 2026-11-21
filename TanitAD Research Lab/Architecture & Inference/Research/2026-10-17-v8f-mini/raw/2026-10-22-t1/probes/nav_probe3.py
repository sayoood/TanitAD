import sys, numpy as np
from tanitad.data import v8f_mini_data as D
J = D.load_join(sys.argv[1] + "/join_eval141.npz")
dw = []; dist_dw = []
for i in range(len(J["clip"])):
    o, n = int(J["off"][i]), int(J["n"][i])
    nav, nd = J["nav"][o:o+n], J["nav_d"][o:o+n]
    tok = nav[:, :3].argmax(1)
    t = 0
    while t < n:
        if tok[t] > 0:
            s = t
            while t < n and tok[t] == tok[s]: t += 1
            run = np.arange(s, t)
            z = run[(nd[run] == 0)]
            if len(z) and z[-1] - z[0] + 1 == len(z) and t < n:   # run ends by a change, dwell at d==0 contiguous
                dw.append(len(z))
        else:
            t += 1
dw = np.array(dw)
print("n turn-runs ending in a change:", len(dw), " dwell at d==0 (cache steps of 0.2 s): median %.1f p10 %.1f p90 %.1f max %d" % (np.median(dw), np.percentile(dw,10), np.percentile(dw,90), dw.max()))
