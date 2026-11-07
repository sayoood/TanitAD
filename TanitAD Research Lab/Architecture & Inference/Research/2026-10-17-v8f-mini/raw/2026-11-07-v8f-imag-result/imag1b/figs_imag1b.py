"""IMAG1B figures from RESULT_IMAG1B.json (deciding dataset eval141 by default; pass `train` for the beside set). Written ONLY when that dataset ran and its controls 2 and 3 pass (an invalid
read draws nothing). No camera frame in any picture. Tier T0 in every title.
Writes fig/probe_transfer_<probe>_<ds>.png (predicted line, Copy-Last band and Oracle band per level), fig/timescale_heatmap_<ds>.png, fig/action_fan_<ds>.png, fig/decorrelation_<probe>_<ds>.png."""
import json, os, sys
sys.dont_write_bytecode = True
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ds = sys.argv[1] if len(sys.argv) > 1 else "eval141"
SUF = os.environ.get("IMAG1B_SUFFIX", "")                                  # test hooks, never set in a real run
FIG = os.environ.get("IMAG1B_FIGDIR", "fig")
def nanify(o):
    """NaN was written as null (strict json): restore it on load, except the 'nmse' fields, which are null on purpose for class targets."""
    if isinstance(o, dict):
        return {k: (v if k == "nmse" else nanify(v)) for k, v in o.items()}
    if isinstance(o, list):
        return [nanify(v) for v in o]
    return float("nan") if o is None else o

R = nanify(json.load(open(f"RESULT_IMAG1B{SUF}.json", encoding="utf-8")))["datasets"].get(ds)
if not R or R.get("status") != "RUN" or not (R["control2"]["pass"] and R["control3"]["pass"]):
    print(f"no figures for {ds}: dataset not run or its controls do not pass")
    sys.exit(0)
D = R
TAGS, LEV, KS = ("M2b_s0", "M2b_s1"), ("1", "2", "3"), ("1", "2", "3")
DT = {"1": 0.2, "2": 0.4, "3": 0.8}
COL = {"1": "#1b6ca8", "2": "#d1495b", "3": "#2e8b57"}
TG = ("speed", "accel", "yaw_rate", "lead_gap", "lead_closing", "agents30")
CLS = ("lead_present", "lon7")
ALL = ("speed", "accel", "lead_gap", "lead_closing", "ttc", "yaw_rate", "curvature", "lead_present", "lon7", "agents30")
k_ = lambda *a: "|".join(str(x) for x in a)
os.makedirs(FIG, exist_ok=True)


def nmse(P, tag, l, k, q, qt):
    return 1 - D["ia2"][k_(P, tag, l, k, q)]["rows"][qt]["point"]


for P in ("ridge", "mlp"):
    fig, axs = plt.subplots(2, 3, figsize=(13, 7))
    for ax, q in zip(axs.ravel(), TG):
        for l in LEV:
            h = [int(k) * DT[l] for k in KS]
            band = {}
            for qt in ("pred", "copy", "oracle"):
                band[qt] = np.array([[nmse(P, tag, l, k, q, qt) for k in KS] for tag in TAGS])
            ax.fill_between(h, band["copy"].min(0), band["copy"].max(0), color=COL[l], alpha=0.16, hatch="//", lw=0)       # Copy-Last band (range of the two seeds)
            ax.fill_between(h, band["oracle"].min(0), band["oracle"].max(0), color=COL[l], alpha=0.34, lw=0)                 # Oracle band
            ax.plot(h, band["pred"].mean(0), "-", color=COL[l], marker="o", ms=3, label=f"L{l} predicted")
        ax.axhline(1.0, color="0.5", lw=0.8)
        ax.set_title(q, fontsize=10)
        ax.set_xlabel("horizon (s)")
        ax.set_ylabel("NMSE (1 = the marginal mean)")
    axs[0, 0].legend(fontsize=6, ncol=3, title="line = predicted (mean of 2 seeds); hatched band = Copy-Last; solid band = Oracle (bands: range of 2 seeds)", title_fontsize=6)
    fig.suptitle(f"{ds}, {P} probe: probe transfer, NMSE vs horizon; T0 (world-model diagnostic)", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{FIG}/probe_transfer_{P}_{ds}.png", dpi=80)
    plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(10, 4.8))
for ax, P in zip(axs, ("ridge", "mlp")):
    M = np.zeros((len(ALL), 3))
    for i, q in enumerate(ALL):
        for j, l in enumerate(LEV):
            M[i, j] = np.mean([1 - D["ia1"][k_(P, tag, l, q)]["score"]["point"] for tag in TAGS])
    im = ax.imshow(M, cmap="viridis_r", aspect="auto", vmin=0.0, vmax=1.2)
    ax.set_xticks(range(3)); ax.set_xticklabels([f"L{l}" for l in LEV])
    ax.set_yticks(range(len(ALL))); ax.set_yticklabels([q + (" (1-score)" if q in CLS else "") for q in ALL], fontsize=8)
    for i in range(len(ALL)):
        for j in range(3):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="w" if M[i, j] > 0.6 else "k", fontsize=7)
    ax.set_title(f"{P}: IA-1 NMSE by target and level (mean of 2 seeds)", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)
fig.suptitle(f"{ds}: timescale heatmap; lower = better captured; classes show 1 - (balanced accuracy minus majority); T0", fontsize=9)
fig.tight_layout()
fig.savefig(f"{FIG}/timescale_heatmap_{ds}.png", dpi=80)
plt.close(fig)

fig, axs = plt.subplots(2, 3, figsize=(13, 7))
for r, (tq, cp, cm, dlt, lab) in enumerate((("speed", "lonp", "lonm", 1.0, "longitudinal contrast (m/s^2)"), ("yaw_rate", "latp", "latm", 0.25, "lateral contrast (rad/s)"))):
    for c, l in enumerate(LEV):
        ax = axs[r, c]
        for k in KS:
            h = int(k) * DT[l]
            ys = []
            for tag in TAGS:
                m = D["ia3"][k_("ridge", tag, l, k)]["means"][tq]
                ys.append([m[cm] - m["logged"], 0.0, m[cp] - m["logged"]])
            ys = np.array(ys)
            ax.plot([-dlt, 0, dlt], ys.mean(0), "o-", ms=3, label=f"imagined k={k} (h {h:.1f} s)")
            phys = [-dlt * (h if r == 0 else 1.0), 0, dlt * (h if r == 0 else 1.0)]
            ax.plot([-dlt, 0, dlt], phys, "--", color="0.4", lw=0.8)
        ax.axhline(0, color="0.7", lw=0.6)
        ax.set_title(f"L{l}: decoded {tq} change under the contrast (ridge probe, mean of 2 seeds)", fontsize=8)
        ax.set_xlabel(lab)
        ax.set_ylabel(f"decoded {tq} minus the logged-action prediction")
        if r == 0 and c == 0:
            ax.legend(fontsize=6)
fig.suptitle(f"{ds}: action fan; dashed grey = the physical answer (speed: da * h; yaw rate: d omega); T0", fontsize=9)
fig.tight_layout()
fig.savefig(f"{FIG}/action_fan_{ds}.png", dpi=80)
plt.close(fig)

dcs = D["decorrelation"]["cells"]
xl = ["+0.2 s", "+0.8 s", "+1.6 s", "other episode"]
for P in ("ridge", "mlp"):
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.8), sharey=False)
    for ax, fam in zip(axs, ("LON", "LAT", "TAC", "SCENE")):
        for l in LEV:
            ys = []
            for nmk in ("off1", "off4", "off8", "nf"):
                v = [dcs[k_(P, tag, l, k, fam)][nmk]["point"] for tag in TAGS for k in KS if k_(P, tag, l, k, fam) in dcs]
                ys.append(np.mean(v) if v else np.nan)
            ax.plot(range(4), ys, "o-", color=COL[l], ms=3, label=f"L{l}")
        ax.set_xticks(range(4)); ax.set_xticklabels(xl, fontsize=7)
        ax.axhline(0, color="0.6", lw=0.6)
        ax.set_title(fam, fontsize=9)
        ax.set_ylabel("pooled gap: score(Oracle) - score(latent)")
    axs[0].legend(fontsize=6)
    fig.suptitle(f"{ds}, {P}: decorrelation curve (beside, no bar), mean over seeds and k, targets with a readable Oracle; T0", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{FIG}/decorrelation_{P}_{ds}.png", dpi=80)
    plt.close(fig)
print("figures written for", ds)
