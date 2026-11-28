"""Side-by-side of two egodrop_analyze JSONs (e.g. step 5,000 vs 9,500). Prints markdown."""
import json
import sys


def g(d, *ks, default=None):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def lv(r):
    if not isinstance(r, dict) or "mean" not in r:
        return "n/a"
    return f"{r['mean']:.4f} [{r['lo']:.4f}, {r['hi']:.4f}]"


def dl(r):
    if not isinstance(r, dict) or "delta" not in r:
        return "n/a"
    return f"{r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] {'SEP' if r['separated'] else 'ns'}"


def main():
    A = [json.load(open(p, encoding="utf-8")) for p in sys.argv[1:]]
    labs = [a.get("label") or str(a.get("model_step")) for a in A]
    print("| quantity | " + " | ".join(labs) + " |")
    print("|---|" + "---|" * len(A))
    rows = [
        ("|traj-bank[sel]| mean-slots, kept", lambda a: lv(g(a, "offset", "levels", "k", "tot_sel_norm_mean_slots"))),
        ("|traj-bank[sel]| mean-slots, withheld", lambda a: lv(g(a, "offset", "levels", "w", "tot_sel_norm_mean_slots"))),
        ("  paired w-k", lambda a: dl(g(a, "offset", "paired_w_minus_k", "tot_sel_norm_mean_slots"))),
        ("|traj-bank[sel]| @6 s, kept", lambda a: lv(g(a, "offset", "levels", "k", "tot_sel_norm_6s"))),
        ("|traj-bank[sel]| @6 s, withheld", lambda a: lv(g(a, "offset", "levels", "w", "tot_sel_norm_6s"))),
        ("need |GT-bank[a*]| kept", lambda a: lv(g(a, "offset", "levels", "k", "need_astar_masked_mean"))),
        ("need |GT-bank[a*]| withheld", lambda a: lv(g(a, "offset", "levels", "w", "need_astar_masked_mean"))),
        ("resid |GT-fan[a*]| kept", lambda a: lv(g(a, "offset", "levels", "k", "resid_astar_masked_mean"))),
        ("resid |GT-fan[a*]| withheld", lambda a: lv(g(a, "offset", "levels", "w", "resid_astar_masked_mean"))),
        ("burden coverage pooled k / w", lambda a: f"{g(a, 'offset', 'burden_coverage_pooled', 'k'):.4f} / {g(a, 'offset', 'burden_coverage_pooled', 'w'):.4f}"),
        ("along-6s vs (v0-10)*6 slope/r, withheld", lambda a: f"{g(a, 'offset', 'along_6s_vs_speed_burden', 'withheld', 'slope'):.3f} / {g(a, 'offset', 'along_6s_vs_speed_burden', 'withheld', 'r'):.3f}"),
        ("jerk mean|.| native8 os_k", lambda a: lv(g(a, "smoothness", "native8", "levels", "os_k", "jerk_mean_abs_mps3"))),
        ("jerk mean|.| native8 os_w", lambda a: lv(g(a, "smoothness", "native8", "levels", "os_w", "jerk_mean_abs_mps3"))),
        ("jerk mean|.| native8 gt", lambda a: lv(g(a, "smoothness", "native8", "levels", "gt", "jerk_mean_abs_mps3"))),
        ("  paired os_w-os_k", lambda a: dl(g(a, "smoothness", "native8", "paired", "os_w_minus_os_k", "jerk_mean_abs_mps3"))),
        ("  paired os_k-gt", lambda a: dl(g(a, "smoothness", "native8", "paired", "os_k_minus_gt", "jerk_mean_abs_mps3"))),
        ("krate mean|.| native8 os_k", lambda a: lv(g(a, "smoothness", "native8", "levels", "os_k", "krate_mean_abs_1pms"))),
        ("krate mean|.| native8 os_w", lambda a: lv(g(a, "smoothness", "native8", "levels", "os_w", "krate_mean_abs_1pms"))),
        ("krate mean|.| native8 gt", lambda a: lv(g(a, "smoothness", "native8", "levels", "gt", "krate_mean_abs_1pms"))),
        ("  paired os_w-os_k", lambda a: dl(g(a, "smoothness", "native8", "paired", "os_w_minus_os_k", "krate_mean_abs_1pms"))),
        ("lateral jerk native8 os_k / os_w / gt", lambda a: f"{g(a, 'smoothness', 'native8', 'levels', 'os_k', 'ljerk_mean_abs_mps3', 'mean')} / {g(a, 'smoothness', 'native8', 'levels', 'os_w', 'ljerk_mean_abs_mps3', 'mean')} / {g(a, 'smoothness', 'native8', 'levels', 'gt', 'ljerk_mean_abs_mps3', 'mean')}"),
        ("selection n_distinct k / w", lambda a: f"{g(a, 'selection', 'kept', 'n_distinct_selected')} / {g(a, 'selection', 'withheld', 'n_distinct_selected')}"),
        ("modal anchor (frac) k / w", lambda a: f"#{g(a, 'selection', 'kept', 'modal_anchor')} ({g(a, 'selection', 'kept', 'modal_frac')}) / #{g(a, 'selection', 'withheld', 'modal_anchor')} ({g(a, 'selection', 'withheld', 'modal_frac')})"),
        ("idx67 share k / w", lambda a: f"{g(a, 'selection', 'kept', 'straight_ahead_idx67_frac')} / {g(a, 'selection', 'withheld', 'straight_ahead_idx67_frac')}"),
        ("entropy ratio k / w", lambda a: f"{g(a, 'selection', 'kept', 'entropy_ratio')} / {g(a, 'selection', 'withheld', 'entropy_ratio')}"),
        ("agrees own oracle k / w", lambda a: f"{g(a, 'selection', 'kept', 'agrees_with_own_oracle_frac')} / {g(a, 'selection', 'withheld', 'agrees_with_own_oracle_frac')}"),
        ("same anchor across regimes", lambda a: f"{g(a, 'selection', 'cross_regime_same_anchor_frac')}"),
        ("ADE 2s os_k", lambda a: lv(g(a, "four_families_2s", "levels_ade", "os_k"))),
        ("ADE 2s os_w", lambda a: lv(g(a, "four_families_2s", "levels_ade", "os_w"))),
        ("ADE 2s ha / ha0", lambda a: f"{g(a, 'four_families_2s', 'levels_ade', 'ha', 'mean')} / {g(a, 'four_families_2s', 'levels_ade', 'ha0', 'mean')}"),
        ("  paired os_w-os_k ADE 2s", lambda a: dl(g(a, "four_families_2s", "families_paired", "paired_osw_minus_osk", "families", "ADE", "ade_m"))),
        ("  paired os_k-ha ADE 2s", lambda a: dl(g(a, "four_families_2s", "families_paired", "paired_osk_minus_ha", "families", "ADE", "ade_m"))),
        ("  paired os_k-ha0 ADE 2s", lambda a: dl(g(a, "four_families_2s", "families_paired", "paired_osk_minus_ha0", "families", "ADE", "ade_m"))),
        ("ADE 6s(true6) os_k", lambda a: lv(g(a, "four_families_6s_true6s", "levels_ade", "os_k"))),
        ("ADE 6s(true6) os_w", lambda a: lv(g(a, "four_families_6s_true6s", "levels_ade", "os_w"))),
        ("ADE 6s(true6) ha / ha0", lambda a: f"{g(a, 'four_families_6s_true6s', 'levels_ade', 'ha', 'mean')} / {g(a, 'four_families_6s_true6s', 'levels_ade', 'ha0', 'mean')}"),
        ("  paired os_w-os_k ADE 6s", lambda a: dl(g(a, "four_families_6s_true6s", "families_paired", "paired_osw_minus_osk", "families", "ADE", "ade_m"))),
        ("pred speed@2s MAE vs v0, withheld", lambda a: f"{g(a, 'option2_own_predicted_speed', 'pred_speed_2s_withheld_mae_vs_v0_mps'):.3f}"),
        ("OIV fixed10 / pred-speed / true-v0", lambda a: f"{g(a, 'option2_own_predicted_speed', 'oiv_fixed_10ms'):.4f} / {g(a, 'option2_own_predicted_speed', 'oiv_at_pred_speed_withheld'):.4f} / {g(a, 'option2_own_predicted_speed', 'oiv_at_true_v0_LEAK_BOUND'):.4f}"),
    ]
    for name, fn in rows:
        cells = []
        for a in A:
            try:
                cells.append(fn(a))
            except Exception as ex:                    # noqa: BLE001
                cells.append(f"err:{type(ex).__name__}")
        print(f"| {name} | " + " | ".join(str(c) for c in cells) + " |")


if __name__ == "__main__":
    main()
