"""Beside PREREG_V8F_PT3M, deciding nothing (Master Mind, 2026-11-09, written after seeing READ-END miss HOLD): (1) READ-DIFF: is the speed error common to two reads of the same clip, so that a difference cancels it?
(2) the world model's action response read through the PT3M reader: under the logged actions and under HOLD, how much does the read at knot 12 move, against how much the two action sequences differ? Tier T0, SCORE windows, CPU, no training. Paired EC-BOOT where an interval is printed. Writes DIAG_RESPONSE.json."""
import json, os, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import torch
import pt3m_lib as PM
from pt3m_lib import L3, P
import plantest3_real as R3

torch.set_num_threads(1)
RUNS = f"{HERE}/runs"
DT = P.DT


def slope(x, y):
    """y = s x + b by least squares, and R2 of that line."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    xc, yc = x - x.mean(), y - y.mean()
    s = float((xc * yc).sum() / (xc ** 2).sum())
    return dict(slope=s, r2=float(1 - ((yc - s * xc) ** 2).sum() / (yc ** 2).sum()), n=int(ok.sum()), sd_x=float(xc.std()), sd_y=float(yc.std()))


def roll_read(ctx, a, c=None, bs=128):
    out = []
    z0 = ctx.Z1[ctx.score_idx][:, 0]
    with torch.no_grad():
        for i in range(0, len(z0), bs):
            if c is None:
                r = ctx.st.rollout(1, z0[i:i + bs], a[i:i + bs])
            else:
                r = L3.shrunk_rollout(ctx.st, 1, z0[i:i + bs], a[i:i + bs], c)
            out.append(ctx.reader.read_raw(r[:, PM.KNOT_IDX, :].flatten(-2)))
    return np.concatenate(out)


def main():
    res = {}
    for t in ("M2b_s0", "M2b_s1"):
        ctx = R3.prepare3(t)
        ctx.reader = PM.load_reader(f"{RUNS}/reader_{t}.pt")
        sc = np.asarray(ctx.score_idx)
        _, Wt, pos = P.boot_setup(ctx.eid[sc])
        bm = lambda x: [float(v) for v in P.boot_mean(x[np.isfinite(x)], pos[np.isfinite(x)], Wt)]

        def bd(a, b):
            m = np.isfinite(a) & np.isfinite(b)
            d = P.boot_mean_diff(a[m], b[m], pos[m], Wt)
            return [float(v) for v in d] + [P.sep_label(*d)]

        mets = lambda p: P.window_metrics(p, ctx.A[sc], ctx.v0[sc])
        A, v0, w0 = ctx.A[sc], ctx.vk[sc, 0], ctx.A[sc, 0, 1]
        hold = mets(P.hold_plan(A))
        Y3, Y12 = ctx.reader.read_raw(PM.stack_knots(ctx.Z1[sc], 3)), ctx.reader.read_raw(PM.stack_knots(ctx.Z1[sc], 12))
        # (1) READ-DIFF: constant acceleration from the difference of two reads of the same clip (knots 3 and 12); the yaw-rate ramp stays READ-END's
        acc = (Y12[:, 0] - Y3[:, 0]) / (9 * DT)
        plan = PM.end_plan(v0, v0 + acc * 12 * DT, w0, Y12[:, 1])
        m = mets(plan)
        rec = dict(read_diff_acc_mae=bm(m["acc_mae"]), hold_acc_mae=bm(hold["acc_mae"]), read_diff_minus_hold=bd(m["acc_mae"], hold["acc_mae"]),
                   abs_read_error_speed_k12=float(np.nanmean(np.abs(Y12[:, 0] - ctx.vk[sc, 12]))), abs_error_of_read_difference=float(np.nanmean(np.abs((Y12[:, 0] - Y3[:, 0]) - (ctx.vk[sc, 12] - ctx.vk[sc, 3])))),
                   true_speed_change_abs_mean=float(np.nanmean(np.abs(ctx.vk[sc, 12] - ctx.vk[sc, 3]))))
        # (2) the action response of the world model through the reader
        a_log = ctx.wm.scale_actions(torch.as_tensor(A[:, :12]).float())
        a_hold = R3.R2.hold_c_of(ctx)[sc].float()
        hold_raw = P.hold_plan(A)
        vl = v0 + (A[:, :12, 0] * DT).sum(1)
        vh = v0 + (hold_raw[:, :12, 0] * DT).sum(1)
        for name, c in (("full", None), ("shrink", ctx.shrink_c)):
            Rl, Rh = roll_read(ctx, a_log, c), roll_read(ctx, a_hold, c)
            rec[f"response_speed_{name}"] = slope(vl - vh, Rl[:, 0] - Rh[:, 0])
            rec[f"response_yaw_{name}"] = slope(A[:, 11, 1] - hold_raw[:, 11, 1], Rl[:, 1] - Rh[:, 1])
            rec[f"rollout_vs_encoded_speed_abs_{name}"] = float(np.nanmean(np.abs(Rl[:, 0] - Y12[:, 0])))
        res[t] = rec
        print(t, json.dumps(rec, indent=0))
    json.dump(res, open(f"{HERE}/DIAG_RESPONSE.json", "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
