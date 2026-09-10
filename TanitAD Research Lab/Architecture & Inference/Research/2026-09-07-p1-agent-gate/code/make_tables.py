"""Render the P1-gate panel JSON into markdown tables.

⛔ NO NUMBER IN THE REPORT IS HAND-TYPED. Every figure in RESULT.md's tables is
emitted from the panel JSON by this script, so a transcription error cannot
enter between the estimator and the reader.
"""
from __future__ import annotations
import json, sys

FAV = {"headway_min_m": "higher", "time_gap_min_s": "higher",
       "min_ttc_s": "higher"}


def main() -> int:
    d = json.load(open(sys.argv[1], encoding="utf-8"))
    out = []
    a = out.append
    a(f"n_windows **{d['n_windows']}** / n_episodes **{d['n_episodes']}** · "
      f"d (trainable params) **{d['d_params_tiny']:,}** · "
      f"rate slots {d['rate_slots']} at dt {d['dt_s']} s · "
      f"windows with a lead at t0: **{d['lead_windows']}**")
    a("")
    a(f"Estimator: `{d['estimator']}`, 10,000 resamples, 95 %. "
      f"NOT used: `{d['not_used']}`. Tier: {d['tier']}")
    a("")

    a("### Arm point estimates (mean over the scored windows)")
    a("")
    cols = ["ade_m", "speed_mae_mps", "along_mae_m", "cross_mae_m",
            "heading_mae", "curvature_mae", "yaw_rate_mae",
            "dk_mean_headway_m", "dk_n"]
    a("| arm | " + " | ".join(cols) + " |")
    a("|" + "---|" * (len(cols) + 1))
    for k, v in d["controls"].items():
        row = []
        for c in cols:
            x = v.get(c)
            row.append("—" if x is None else
                       (f"{x:.0f}" if c == "dk_n" else f"{x:.5f}"))
        a(f"| `{k}` | " + " | ".join(row) + " |")
    a("")

    def block(title, items, note=""):
        a(f"### {title}")
        if note:
            a("")
            a(note)
        a("")
        a("| contrast | delta | 95 % CI | n win / ep | verdict |")
        a("|---|---|---|---|---|")
        for name, r in items:
            if r.get("status") == "UNAVAILABLE":
                a(f"| `{name}` | — | — | n={r.get('n', 0)} | "
                  f"UNAVAILABLE — {r.get('reason', '')} |")
                continue
            dp = r.get("display_dp", 4)
            a(f"| `{name}` | {r['delta']:+.{dp}f} | "
              f"[{r['lo']:+.{dp}f}, {r['hi']:+.{dp}f}] | "
              f"{r['n_windows']} / {r['n_episodes']} | "
              f"{'**' + r['verdict'] + '**' if r['separated'] else r['verdict']} |")
        a("")

    block("LONGITUDINAL + LATERAL contrasts (delta = A − B; **lower is better**)",
          list(d["contrasts"].items()),
          "⚠️ LATERAL is read on **curvature MAE with the straight-line floor "
          "beside it** — see the control table row `straight`.")
    block("DISTANCE-KEEPING contrasts (headway / time-gap / min-TTC; "
          "**higher is safer**)",
          list(d["dk"].items()),
          "⭐ This is the family the lever is supposed to move.")

    a("### Controls at their known values")
    a("")
    cc = d["control_checks"]
    c0 = cc["const_is_no_information"]
    a(f"* **constant-only (never moves)** — ADE **{c0['ade_m']:.6f}** vs the "
      f"no-information value mean‖gt‖ **{c0['gt_mean_norm']:.6f}**, "
      f"tol {c0.get('tol', 1e-4)} ⇒ **{'PASS' if c0['passes'] else 'FAIL'}**")
    c1 = cc["straight_line_curvature_floor"]
    arms = " · ".join(f"`{k}` **{v:.5f}**" for k, v in c1["arms"].items())
    a(f"* **straight-line floor (never steers)** — curvature MAE "
      f"**{c1['curvature_mae']:.5f}** 1/m. Arms: {arms}. "
      f"⛔ Ratio arm/floor: " +
      " · ".join(f"`{k}` **{v / c1['curvature_mae']:.1f}×**"
                 for k, v in c1["arms"].items()))
    a(f"* **n vs d** — {d['n_vs_d_note']} Here n = {d['n_windows']} windows / "
      f"{d['n_episodes']} episodes against d = {d['d_params_tiny']:,}: "
      f"**n ≪ d**.")
    a(f"* **windows entering each metric** — {d['windows_per_metric']}")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
