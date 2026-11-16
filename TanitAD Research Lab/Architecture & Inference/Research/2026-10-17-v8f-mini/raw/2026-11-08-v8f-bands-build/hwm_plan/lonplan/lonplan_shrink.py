"""LON-PLAN arm S (SHRINK), AMEND_HWMPLAN_3 section 4 (frozen, sha256 7bbdc9fe..., commit 4dc4876c). Post-hoc, zero compute, CPU only.
S = the matched-shrinkage control of A1: knots 1-10 are HOLD + alpha_c * (A0_plan - HOLD) per channel c (acc, yaw); knots 11-30 are A0's, bit for bit.
alpha_k,c = D_TRAIN,k,c(lam*_k) / D_TRAIN,k,c(0), read from the TRAIN selection record LAMBDA_SELECTION_R35_s{k}.json (runs[str(lam)]['D'] for acc, ['D_yaw'] for yaw). It is recomputed from that file on every call, never typed in;
the frozen values are asserted beside it (EXPECTED_ALPHA).
HOLD = hwm_plan.hold_plan(A[:, 0], n): the logged acc / yaw-rate at NOW, held. That is the reference lonplan_select.distance_D measures D against (lonplan_select.py distance_D: H.hold_plan(A[:, 0], ...)), plain HOLD and not the prior-box-clipped HOLD_c.
usage: python lonplan_shrink.py --plans A0/plans_hwm_oracle_s0.npz --windows DIR/windows.npz --selection sel_s0/LAMBDA_SELECTION_R35_s0.json --out S_s0/plans_hwm_oracle_s0.npz
The output keeps every key of the A0 file (cost1, iters1, idx, a1_raw, decode_resid are A0's and are NOT re-solved for S; only plan_raw is changed) and adds 'shrink_alpha' (2,). A stamp JSON is written beside it."""
import argparse, json, os, sys, time
sys.dont_write_bytecode = True
import lonplan_common as C
import numpy as np
import hwm_plan as H

N_SHRINK = 10                                  # knots 1-10 = the 0-2 s band
EXPECTED_ALPHA = {"R35_s0": (0.07843263912195275, 0.09896859284406319),     # (alpha_acc, alpha_yaw), frozen by the MM Log entry of the TRAIN selection
                  "R35_s1": (0.12367962338894242, 0.12045464645362154)}


def alpha_from_selection(sel_path):
    """(alpha_acc, alpha_yaw, info) recomputed from the selection record. Refuses a record without a selected lambda or with a zero D at lambda 0."""
    j = json.load(open(sel_path, encoding="utf-8"))
    lam = j["lam_star"]
    if lam is None or j.get("label") != "SELECTED":
        raise ValueError(f"selection record {sel_path} has no selected lambda (label {j.get('label')})")
    r0, rs = j["runs"][str(float(0.0))], j["runs"][str(float(lam))]
    if not (r0["D"] > 0 and r0["D_yaw"] > 0):
        raise ValueError("D at lambda 0 is not positive")
    a_acc, a_yaw = rs["D"] / r0["D"], rs["D_yaw"] / r0["D_yaw"]
    return a_acc, a_yaw, dict(chassis=j["chassis"], lam_star=float(lam), D0=[r0["D"], r0["D_yaw"]], D_star=[rs["D"], rs["D_yaw"]], prior_std_raw=j["prior_std_raw"])


def assert_expected(chassis, a_acc, a_yaw, tol=1e-12):
    exp = EXPECTED_ALPHA[chassis]
    assert abs(a_acc - exp[0]) < tol and abs(a_yaw - exp[1]) < tol, (chassis, (a_acc, a_yaw), exp)


def hold_of(A, n):
    return H.hold_plan(np.asarray(A)[:, 0], n)


def shrink_plan(plan_raw, A, alpha_acc, alpha_yaw, n_shrink=N_SHRINK):
    """Knots 1..n_shrink: HOLD + alpha_c (plan - HOLD) per channel; the rest is the input array's. alpha 1 keeps the plan's own bytes and alpha 0 returns HOLD's, so both ends are bit-exact."""
    plan = np.asarray(plan_raw, np.float64)
    assert plan.ndim == 3 and plan.shape[2] == 2 and plan.shape[0] == len(A), (plan.shape, len(A))
    hold = hold_of(A, plan.shape[1])
    out = plan.copy()
    for c, al in enumerate((alpha_acc, alpha_yaw)):
        if al == 1.0:
            continue                                                            # keeps A0's own values
        out[:, :n_shrink, c] = hold[:, :n_shrink, c] if al == 0.0 else hold[:, :n_shrink, c] + al * (plan[:, :n_shrink, c] - hold[:, :n_shrink, c])
    return out


def distance_D(plan_raw, A, std_raw, n=N_SHRINK):
    """(D_acc, D_yaw) exactly as lonplan_select.distance_D writes it, with the std taken from the selection record."""
    plan = np.asarray(plan_raw, np.float64)
    d = np.abs(plan[:, :n] - hold_of(A, plan.shape[1])[:, :n]).mean(axis=(0, 1)) / np.asarray(std_raw)
    return float(d[0]), float(d[1])


def build(plans_path, windows_path, sel_path, out_path, chassis_expect=None):
    if os.path.exists(out_path):
        raise FileExistsError(out_path)
    a_acc, a_yaw, info = alpha_from_selection(sel_path)
    if chassis_expect is None:
        chassis_expect = info["chassis"]
    if chassis_expect in EXPECTED_ALPHA:
        assert_expected(chassis_expect, a_acc, a_yaw)
    with np.load(plans_path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    with np.load(windows_path, allow_pickle=False) as w:
        A = w["A"].astype(np.float64)
    a0_plan = d["plan_raw"]
    s = shrink_plan(a0_plan, A, a_acc, a_yaw)
    assert np.array_equal(s[:, N_SHRINK:], a0_plan[:, N_SHRINK:])
    d["plan_raw"] = s
    d["shrink_alpha"] = np.array([a_acc, a_yaw])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez(out_path, **d)
    D_a0 = distance_D(a0_plan, A, info["prior_std_raw"])
    D_s = distance_D(s, A, info["prior_std_raw"])
    stamp = dict(kind="LONPLAN_ARM_S_SHRINK", amendment="AMEND_HWMPLAN_3 (frozen, sha256 7bbdc9fe..., commit 4dc4876c)", chassis=info["chassis"], lam_star=info["lam_star"], alpha_acc=a_acc, alpha_yaw=a_yaw,
                 alpha_rule="alpha_c = D_TRAIN,c(lam*) / D_TRAIN,c(0), from the selection record", expected_alpha_asserted=chassis_expect in EXPECTED_ALPHA,
                 hold_reference="hwm_plan.hold_plan(A[:, 0], n): logged a0 held (plain HOLD, as lonplan_select.distance_D measures D)",
                 knots_shrunk="1-10", knots_11_30="A0's, bit-equal (asserted)",
                 D_score_in_train_prior_std=dict(A0=D_a0, S=D_s, D_train_lam0=info["D0"], D_train_lam_star=info["D_star"], note="SCORE distances beside; S's value is alpha * D_A0"),
                 sha256=dict(a0_plans=C.sha256_file(plans_path), windows=C.sha256_file(windows_path), selection=C.sha256_file(sel_path), out=C.sha256_file(out_path), lonplan_shrink=C.sha256_file(os.path.abspath(__file__))),
                 n_windows=int(len(A)), written=time.strftime("%Y-%m-%dT%H:%M:%S"))
    C.dump_json(stamp, os.path.splitext(out_path)[0] + "_STAMP.json")
    return stamp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", required=True); ap.add_argument("--windows", required=True); ap.add_argument("--selection", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    st = build(a.plans, a.windows, a.selection, a.out)
    print("S written", a.out, "alpha", st["alpha_acc"], st["alpha_yaw"], "D_score A0/S", st["D_score_in_train_prior_std"]["A0"], st["D_score_in_train_prior_std"]["S"])


if __name__ == "__main__":
    main()
