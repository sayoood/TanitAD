"""CAPLADDER figure: fig/ladder_lead_gap.png, NMSE of lead_gap per rung with the bootstrap 95 % band, both M2b seeds; the reference 2.5th percentile (the second READABLE part) as a tick per rung.
Left: the ladder R0 -> R1 -> R2, then the two branches R2 -> R3 and R2 -> R4. Right (beside): R2m, the three readers, the three levels, the re-initialised networks. Written only when the controls are VALID."""
import json, os, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("CAPLADDER_OUT", HERE)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OI = {"base": "#222222", "s0": "#0072B2", "s1": "#D55E00", "rand": "#7f7f7f"}      # Okabe-Ito pair + neutral


def main():
    R = json.load(open(os.path.join(OUT, "RESULT_CAPLADDER.json"), encoding="utf-8"))
    if not R["controls"]["valid"]:
        print("[figs] controls INVALID: no figure written")
        return 0
    rows = R["rungs"]
    os.makedirs(os.path.join(OUT, "fig"), exist_ok=True)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.15, 1]})

    def pt(a, x, nm, col, off=0.0, mk="o", lab=None):
        v = rows[nm]
        a.errorbar([x + off], [v["nmse"]], yerr=[[v["nmse"] - v["lo"]], [v["hi"] - v["nmse"]]], fmt=mk, color=col, capsize=4, lw=1.8, ms=7, label=lab)
        a.plot([x + off - 0.12, x + off + 0.12], [v["ref_p2_5"]] * 2, color=col, lw=1.2, ls=":")
        return v["nmse"]

    xs, ys = [], []
    for x, nm in enumerate(["R0", "R1", "R2"]):
        if nm in rows:
            ys.append(pt(ax, x, nm, OI["base"], lab="R0 to R2 (no model)" if x == 0 else None))
            xs.append(x)
    ax.plot(xs, ys, color=OI["base"], lw=1.2, alpha=0.6)
    for sd, off in (("s0", -0.07), ("s1", 0.07)):
        for x, b in ((3, "R3"), (4, "R4")):
            nm = f"{b}_{sd}"
            if nm in rows:
                y = pt(ax, x, nm, OI[sd], off, "s", lab=f"M2b {sd}" if b == "R3" else None)
                if "R2" in rows:
                    ax.plot([2, x + off], [rows["R2"]["nmse"], y], color=OI[sd], lw=1.0, ls="--", alpha=0.6)
    ax.axhline(1.0, color="#999999", lw=1)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xticklabels(["R0\npatch 16x40", "R1\npool 8x20", "R2\npool 4x10\n(model input)", "R3\nreaders S|T|O", "R4\nlevels L1|L2|L3"], fontsize=8.5)
    ax.set_ylabel("NMSE of lead_gap (1 = fit-fold mean)")
    ax.set_title("Where the lead gap is lost (T0, eval-141, 2,489 frames, 56 episodes)", fontsize=10)
    fig.text(0.01, 0.01, "bars: bootstrap 95 % (episode cluster); dotted tick: 2.5th percentile of the 199-draw refit-under-permutation reference; grey line: NMSE 1 (the fit-fold mean). Squares: M2b seed 0 (blue) and seed 1 (orange).", fontsize=8, color="#444444")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    if "R0" not in rows:
        ax.annotate("R0 " + R["r0"]["label"], (0, 0.5), xycoords=("data", "axes fraction"), fontsize=8, color="#444444", ha="center")
    # beside panel
    labs = [("R2m", "R2m\nmean token", None)] + [(f"R3{k}", f"{k_}\nreader", None) for k, k_ in (("s", "read_s"), ("t", "read_t"), ("o", "read_o"))] + [(f"L{l}", f"L{l}\nlevel", None) for l in (1, 2, 3)]
    for i, (b, text, _) in enumerate(labs):
        if b == "R2m" and "R2m" in rows:
            pt(bx, i, "R2m", OI["base"])
        for sd, off in (("s0", -0.07), ("s1", 0.07)):
            nm = f"{b}_{sd}"
            if nm in rows:
                pt(bx, i, nm, OI[sd], off, "s")
    for j, (nm, text) in enumerate([("R3_r100", "R3 re-init\n(100)"), ("R3_r101", "R3 re-init\n(101)"), ("R4_r100", "R4 re-init\n(100)"), ("R4_r101", "R4 re-init\n(101)")]):
        if nm in rows:
            pt(bx, len(labs) + j, nm, OI["rand"], 0.0, "^")
    bx.axhline(1.0, color="#999999", lw=1)
    bx.set_xticks(range(len(labs) + 4))
    bx.set_xticklabels([t for _, t, _ in labs] + ["R3 re-init\n(100)", "R3 re-init\n(101)", "R4 re-init\n(100)", "R4 re-init\n(101)"], fontsize=7.5)
    bx.set_title("Beside: mean token, each reader, each level, untrained networks", fontsize=10)
    for a in (ax, bx):
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="y", color="#e6e6e6", lw=0.6)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(OUT, "fig", "ladder_lead_gap.png"), dpi=160)
    print("[figs] wrote fig/ladder_lead_gap.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
