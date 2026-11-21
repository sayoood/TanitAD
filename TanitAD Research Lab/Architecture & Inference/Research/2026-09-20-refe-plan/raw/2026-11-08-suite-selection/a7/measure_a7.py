"""AMENDMENT_7 step 1: A (arc length from first v<0.5 tick to end) and N (net displacement) on the box-centre 10 Hz path the scorer's HumanPath sees. CPU only."""
import os, sys, json, collections
import numpy as np
A6 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "a6")
sys.path.insert(0, A6); sys.path.insert(0, "E:/Projects/TanitAD-artifacts/refe_behaviour_20261107/code")
import select_suite_tokens as SS
c6 = json.load(open(f"{A6}/SUITE_CANDIDATES_A6.json"))["slots"]
want = [("TL-G", e) for e in c6["TL-G"]["ordered_one_per_log"] if e["rank"] <= 5] + [("TL-S", e) for e in c6["TL-S"]["ordered_one_per_log"] if e["rank"] <= 8]
by = collections.defaultdict(list)
for sl, e in want: by[e["log"]].append((sl, e))
out = {}
for log, lst in sorted(by.items()):
    ctx = SS.load_log(f"{SS.DB_DIR}/{log}.db")
    for sl, e in lst:
        r0 = ctx.tok2rank[e["token"]]; K = np.arange(SS.SPEC["k_lo"], SS.SPEC["k_hi"] + 1); idx = r0 + SS.SPEC["step"] * K
        yaw = ctx.eyaw[idx]; v = ctx.ev[idx]
        x = ctx.ex[idx] + np.cos(yaw) * SS.EGO_COG_OFF; y = ctx.ey[idx] + np.sin(yaw) * SS.EGO_COG_OFF
        seg = np.hypot(np.diff(x), np.diff(y)); cs = np.concatenate([[0], np.cumsum(seg)])
        low = np.where(v < 0.5)[0]; i0 = int(low[0]) if len(low) else None
        d = dict(slot=sl, rank=e["rank"], n_ticks=len(x), total_arc_to_last_pose=float(cs[-1]), first_v_lt_0p5_tick=i0)
        if i0 is not None:
            A = float(cs[-1] - cs[i0]); N = float(np.hypot(x[-1] - x[i0], y[-1] - y[i0]))
            d.update(arc_up_to_tick=float(cs[i0]), A=A, N=N, A_minus_N=A - N, confirmed_gt_0p5=bool(A - N > 0.5),
                     n_samples_after=int(len(x) - i0 - 1), n_steps_lt_0p2m_after=int((seg[i0:] < 0.2).sum()),
                     v_after_max=float(v[i0:].max()), v_after_min=float(v[i0:].min()))
            # moving-part check: any v>=0.5 after i0?
            d["n_moving_after_i0"] = int((v[i0:] >= 0.5).sum())
        out[e["token"][:16]] = d
        print(e["token"][:8], sl, {k: (round(x_, 3) if isinstance(x_, float) else x_) for k, x_ in d.items()}, flush=True)
json.dump(out, open("MEASURE_090d6a17.json", "w"), indent=1)
