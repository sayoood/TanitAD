"""Figures for V8F_SCIENTIFIC_REPORT.md. CPU only. Reads ONLY banked raw JSON (copied under ./data, sha256 in data/SHA256SUMS.txt).

Usage:  CUDA_VISIBLE_DEVICES=-1 python make_figures.py [--data DIR] [--out DIR]

Every number plotted is read from a banked JSON; nothing is typed in except labels and the 2F band, which is
read from the JSONs too (F = |ADE6(M2b_s0) - ADE6(M2b_s1)|). fig_architecture.png is a schematic (no data).
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ap = argparse.ArgumentParser()
here = os.path.dirname(os.path.abspath(__file__))
ap.add_argument("--data", default=os.path.join(here, "data"))
ap.add_argument("--out", default=here)
a = ap.parse_args()
D, OUT = a.data, a.out

# ---- palette (dataviz reference instance, light surface) ---------------------------------------------------------
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA, VIOLET, RED, GRAY, YEL = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e34948", "#8a8984", "#eda100"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "font.family": "DejaVu Sans",
})


def J(name):
    with open(os.path.join(D, name), encoding="utf-8") as f:
        return json.load(f)


tac = J("raw_2026-10-20-tac_tac_read.json")
grp = J("raw_2026-10-21-tacgrp_tac_groups_read.json")
grp2 = J("raw_2026-10-21-tacgrp2_tac_groups2_read.json")
kind = J("raw_2026-10-25-kind_run_kind_read.json")
ver = J("raw_full_eval_verdict.json")
cmp_ = J("raw_full_eval_compare.json")
swing = J("raw_2026-10-25-ensemble_paired_swing.json")
wc, pgt = J("raw_2026-10-22-wc_run_wc_read_s0.json"), J("raw_2026-10-22-pgt_run_pgt_read_s0.json")
grid, hist = J("raw_2026-10-22-grid_run_grid_read_s0.json"), J("raw_2026-10-23-hist_run_hist_read_s0.json")
dsc = J("raw_2026-10-25-dscale_run_dscale_read.json")
floors = J("raw_full_eval_floors_report.json")

F_M2B, F_M1 = tac["F"]["M2b"], tac["F"]["M1"]
TWOF = 2 * F_M2B
CKS = ["M1_s0", "M1_s1", "M2b_s0", "M2b_s1"]
ARM_COL = {"M1": ORANGE, "M2b": BLUE}


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p, os.path.getsize(p) // 1024, "KB")


# ======================================================================================================================
# fig_oracle_headroom: own code vs GT-code oracle (T0) vs donor code; K-1 deltas vs P10
# ======================================================================================================================
fig, ax = plt.subplots(1, 2, figsize=(10.2, 4.1), gridspec_kw={"width_ratios": [1.35, 1]})
p10 = ver["families"]["_floors"]["P10_refit"]["traj"]["ade6_m"]
x = np.arange(len(CKS) + 1)
w = 0.26
a0 = ax[0]
a0.bar(0, p10["mean"], 0.55, color=GRAY, zorder=3)
a0.errorbar(0, p10["mean"], yerr=[[p10["mean"] - p10["lo"]], [p10["hi"] - p10["mean"]]], color=INK, lw=1, capsize=3, zorder=4)
a0.text(0, p10["hi"] + 0.15, f'{p10["mean"]:.2f}', ha="center", fontsize=8, color=INK2)
for i, ck in enumerate(CKS):
    arm = ck.split("_")[0]
    t = ver["families"][ck]["traj"]["ade6_m"]
    r = tac["rows"][ck]
    xi = i + 1
    a0.bar(xi - w, r["ade6_pred"], w, color=ARM_COL[arm], zorder=3, label="own code (open loop, own held outputs)" if i == 0 else None)
    a0.errorbar(xi - w, t["mean"], yerr=[[t["mean"] - t["lo"]], [t["hi"] - t["mean"]]], color=INK, lw=1, capsize=2, zorder=4)
    a0.bar(xi, r["ade6_orc"], w, color=AQUA, zorder=3, label="GT tactical code (T0 oracle)" if i == 0 else None)
    a0.bar(xi + w, r["ade6_shuf"], w, color=RED, zorder=3, label="another window's code (shuffle)" if i == 0 else None)
    for dx, v in ((-w, r["ade6_pred"]), (0, r["ade6_orc"]), (w, r["ade6_shuf"])):
        a0.text(xi + dx, v + (0.42 if dx == -w else 0.12), f"{v:.2f}", ha="center", fontsize=7, color=INK2)
a0.set_xticks(x)
a0.set_xticklabels(["P10-yaw\n-4512", "M1\nseed 0", "M1\nseed 1", "M2b\nseed 0", "M2b\nseed 1"])
a0.set_ylabel("ADE@6 s (m), eval-141 (2,399 windows)")
a0.set_title("O reads T's code: removing it costs 4-6 m, a perfect one gains ~1.1 m")
a0.legend(loc="upper right", fontsize=7.5, bbox_to_anchor=(1.0, 0.98))
a0.set_ylim(0, 10.2)

a1 = ax[1]
rows = []
for arm in ("M1", "M2b"):
    for s in (0, 1):
        d = ver["K1"]["per_arm"][arm]["K1_ade6_minus_P10"][s]
        rows.append((f"{arm} s{s}", arm, d["mean"], d["lo"], d["hi"]))
for i, (lab, arm, m, lo, hi) in enumerate(rows):
    y = len(rows) - 1 - i
    a1.plot([lo, hi], [y, y], color=ARM_COL[arm], lw=2.2, solid_capstyle="butt", zorder=3)
    a1.plot([m], [y], "o", color=ARM_COL[arm], ms=7, mec=SURF, mew=1.5, zorder=4)
    a1.text(hi + 0.02, y, f"{m:+.3f}", va="center", fontsize=8, color=INK2)
a1.set_yticks(range(len(rows)))
a1.set_yticklabels([r[0] for r in rows][::-1])
for val, col, lab in ((-2 * F_M1, ORANGE, f"-2F (M1) = -{2*F_M1:.3f}"), (-TWOF, BLUE, f"-2F (M2b) = -{TWOF:.3f}")):
    a1.axvline(val, color=col, lw=1, ls="--", zorder=2)
a1.axvline(0, color=INK2, lw=0.8)
a1.set_xlim(-0.75, 0.15)
a1.text(-2 * F_M1 - 0.01, 3.55, "-2F(M1)", color=ORANGE, fontsize=7.5, ha="right")
a1.text(-TWOF - 0.01, 3.55, "-2F(M2b)", color=BLUE, fontsize=7.5, ha="right")
a1.set_xlabel("ADE@6 s, arm minus P10-yaw-4512 (m), paired")
a1.set_title("H-MINI-K-1: M2b passes, M1 fails its own 2F")
a1.set_ylim(-0.6, 3.9)
fig.tight_layout()
save(fig, "fig_oracle_headroom.png")

# ======================================================================================================================
# fig_group_oracle: per-group oracle (M2b), signed
# ======================================================================================================================
order = ["goal", "latdec", "londec", "latcons", "lonspeed", "stoplead", "decisions", "constraints"]
lab = {"goal": "goal (22)", "latdec": "lateral decision (5)", "londec": "longitudinal decision (7)", "latcons": "lat. constraints (12)",
       "lonspeed": "speed constraints (8)", "stoplead": "stop/lead constraints (6)", "decisions": "ALL decisions (34)",
       "constraints": "ALL constraints (26)"}
fig, ax = plt.subplots(figsize=(8.4, 4.3))
H = 0.36
for j, (ck, col) in enumerate((("M2b_s0", BLUE), ("M2b_s1", "#86b6ef"))):
    for i, g in enumerate(order):
        d = grp["rows"][ck]["groups"][g]["paired"]
        y = len(order) - 1 - i + (0.19 if j == 0 else -0.19)
        ax.barh(y, d["delta"], H, color=col, zorder=3, label=f"M2b seed {j}" if i == 0 else None)
        ax.plot([d["lo"], d["hi"]], [y, y], color=INK, lw=1, zorder=4)
        ax.text(max(d["hi"], d["delta"]) + 0.03 if d["delta"] > 0 else d["lo"] - 0.03, y, f'{d["delta"]:+.2f}', va="center",
                ha="left" if d["delta"] > 0 else "right", fontsize=7.5, color=INK2)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([lab[g] for g in order][::-1])
ax.axvline(0, color=INK2, lw=0.8)
ax.axvspan(-TWOF, TWOF, color=GRAY, alpha=0.12, zorder=1)
ax.text(0, -0.78, "grey band = +-2F", ha="center", fontsize=7.5, color=INK2)
ax.set_xlim(-1.6, 0.75)
ax.set_xlabel("ADE@6 s, GT in this group minus T's own code (m)  [negative = O drives better with GT here]   T0 oracle, 95% paired CI")
ax.set_title("The headroom is the constraint SET (-1.08 / -0.92 m); two of its three groups HURT alone")
ax.legend(loc="upper right", fontsize=8)
ax.set_ylim(-0.95, len(order) - 0.4)
fig.tight_layout()
save(fig, "fig_group_oracle.png")

# ======================================================================================================================
# fig_dose_response: convex r(alpha)
# ======================================================================================================================
fig, ax = plt.subplots(figsize=(5.6, 4.2))
al = [0.0, 0.25, 0.5, 0.75, 1.0]
styles = {"M1_s0": (ORANGE, "-"), "M1_s1": (ORANGE, "--"), "M2b_s0": (BLUE, "-"), "M2b_s1": (BLUE, "--")}
for ck in CKS:
    dose = grp2["rows"][ck]["dose"]
    ys = [0.0] + [dose[f"{k:.2f}"]["r"] for k in (0.25, 0.5, 0.75, 1.0)]
    col, ls = styles[ck]
    ax.plot(al, ys, ls, color=col, lw=2, marker="o", ms=5, mec=SURF, mew=1.2, label=ck.replace("_s", " seed "))
ax.plot([0, 1], [0, 1], color=GRAY, lw=1, ls=":", label="linear response (r = alpha)")
ax.set_xlabel("alpha: T's constraint cells moved this fraction of the way to the GT")
ax.set_ylabel("share of the constraints-oracle gain realised, r(alpha)")
ax.set_title("Convex: a quarter of the way buys 8-16% of the gain")
ax.legend(loc="upper left", fontsize=8)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.05)
fig.tight_layout()
save(fig, "fig_dose_response.png")

# ======================================================================================================================
# fig_kind: per-kind gain and necessity
# ======================================================================================================================
kinds = ["speed", "angle", "tte", "timegap", "geom", "leadgap"]
klab = {"speed": "speed", "angle": "angle", "tte": "time-to-event", "timegap": "time gap", "geom": "geometry", "leadgap": "lead gap"}
fig, ax = plt.subplots(1, 2, figsize=(10.2, 4.0), sharey=True)
for pi, (key, ttl) in enumerate((("gain", "gain_k = ADE6(own) - ADE6(GT in kind k only)"), ("necessity", "necessity_k = ADE6(GT except k) - ADE6(full GT)"))):
    a_ = ax[pi]
    items = kinds + ["union"]
    for i, k in enumerate(items):
        if k == "union" and key != "gain":
            continue
        for j, (s, col) in enumerate((("s0", BLUE), ("s1", "#86b6ef"))):
            d = kind["rows"][s]["gain_union"] if k == "union" else kind["rows"][s][key][k]
            y = len(items) - 1 - i + (0.19 if j == 0 else -0.19)
            a_.barh(y, d["delta"], 0.34, color=col if k != "union" else (INK2 if j == 0 else "#a8a7a2"), zorder=3,
                    label=("seed 0" if j == 0 else "seed 1") if i == 0 else None)
            a_.plot([d["lo"], d["hi"]], [y, y], color=INK, lw=1, zorder=4)
    a_.axvline(0, color=INK2, lw=0.8)
    a_.axvspan(-TWOF, TWOF, color=GRAY, alpha=0.12, zorder=1)
    a_.set_title(ttl, fontsize=9)
    a_.set_xlabel("m of ADE@6 s (T0 oracle, M2b, 95% paired CI; grey band = +-2F)")
    a_.set_yticks(range(len(items)))
    a_.set_yticklabels([klab.get(k, "UNION (all 26 cells)") for k in items][::-1])
ax[0].legend(loc="upper right", fontsize=8)
A = kind["decision"]["A"]
ax[0].text(0.98, 0.52, f"additivity A = {A[0]:.2f} / {A[1]:.2f}  (bar 0.8)\nsum of single-kind gains {kind['rows']['s0']['sum_gain']:+.2f} m\nvs union {kind['rows']['s0']['gain_union']['delta']:+.2f} m (s0)",
           transform=ax[0].transAxes, ha="right", va="center", fontsize=7.5, color=INK2,
           bbox=dict(fc=SURF, ec=GRID, boxstyle="round,pad=0.3"))
fig.suptitle("PREREG_V8F_KIND: INTERACTIVE. O cannot use a partly right code.", x=0.01, ha="left", fontsize=10, fontweight="bold", y=1.03)
fig.tight_layout()
save(fig, "fig_kind.png")

# ======================================================================================================================
# fig_lever_arms: forest of every one-lever arm (seed 0) vs M2b_s0, plus WM-1
# ======================================================================================================================
levers = [
    ("WC: T's loss weight x8", wc["rows"]["s0"]["delta"]),
    ("PGT: scheduled sampling 0.5 -> 0.0", pgt["rows"]["s0"]["primary"]),
    ("GRID: 8x20 scene tokens", grid["rows"]["s0"]["primary"]),
    ("HIST: 10 history frames", hist["rows"]["s0"]["primary"]),
    ("DSCALE: half the clips (control)", dsc["rows"]["s0"]["beside"]["ade6"]["delta"]),
]
ws = ver["WM1"]["per_seed"]
levers += [("WM-1 aux: M2b - M1, seed 0", {"delta": ws[0]["mean"], "lo": ws[0]["lo"], "hi": ws[0]["hi"]}),
           ("WM-1 aux: M2b - M1, seed 1", {"delta": ws[1]["mean"], "lo": ws[1]["lo"], "hi": ws[1]["hi"]})]
seed_pair = tac["rows"]["M2b_s1"]["ade6_pred"] - tac["rows"]["M2b_s0"]["ade6_pred"]
fig, ax = plt.subplots(figsize=(8.6, 4.3))
n = len(levers) + 1
for i, (name, d) in enumerate(levers):
    y = n - 1 - i
    sep = (d["lo"] > 0) or (d["hi"] < 0)
    col = BLUE if sep else GRAY
    ax.plot([d["lo"], d["hi"]], [y, y], color=col, lw=2.2, solid_capstyle="butt", zorder=3)
    ax.plot([d["delta"]], [y], "o", color=col, ms=7, mec=SURF, mew=1.5, zorder=4)
    ax.text(0.47, y, f'{d["delta"]:+.3f} [{d["lo"]:+.3f}, {d["hi"]:+.3f}]' + ("  sep." if sep else "  ns"), va="center", fontsize=7.5, color=INK2)
ax.plot([seed_pair], [0], "D", color=ORANGE, ms=7, mec=SURF, mew=1.5, zorder=4)
ax.text(0.47, 0, f"{seed_pair:+.3f}  (point only; the same recipe, a different seed)", va="center", fontsize=7.5, color=INK2)
names = [l[0] for l in levers] + ["M2b seed 1 - seed 0 (NO lever moved)"]
ax.set_yticks(range(n))
ax.set_yticklabels(names[::-1])
ax.axvline(0, color=INK2, lw=0.8)
ax.axvspan(-TWOF, -0.0, color=AQUA, alpha=0.10, zorder=1)
ax.axvline(-TWOF, color=AQUA, lw=1.2, ls="--", zorder=2)
ax.text(-TWOF - 0.01, 3.5, "pre-registered bar:\n< -2F, separated", ha="right", va="center", fontsize=7.5, color="#0f7a55")
ax.set_xlim(-0.5, 1.15)
ax.set_xlabel("ADE@6 s, arm minus base (m); base = M2b seed 0 unless stated; paired episode-cluster CI")
ax.set_title("No lever clears 2F; the seed gap (diamond) exceeds every lever's point estimate")
fig.tight_layout()
save(fig, "fig_lever_arms.png")

# ======================================================================================================================
# fig_variances: three variances on one axis
# ======================================================================================================================
cis = {"WC": wc["rows"]["s0"]["delta"]["ci95"], "PGT": pgt["rows"]["s0"]["primary"]["ci95"], "GRID": grid["rows"]["s0"]["primary"]["ci95"],
       "HIST": hist["rows"]["s0"]["primary"]["ci95"], "DSCALE": dsc["rows"]["s0"]["beside"]["ade6"]["delta"]["ci95"]}
rep = max(v["repro_max_dev_m"] for v in tac["checks"].values())
bars = [
    ("EPISODES  (what the bootstrap answers)\nhalf-width of the paired 95% CI, 5 lever arms", list(cis.values()), BLUE),
    ("TRAINING SEED  F = |ADE6(s0) - ADE6(s1)|", [F_M2B, F_M1], ORANGE),
    ("TRAINING SEED  seed-paired swing of M2b - M1", [swing["ade6"]["swing"]], ORANGE),
    ("BAR  2F (M2b)", [TWOF], INK2),
    ("INFERENCE  re-running the same checkpoint\n(max per-window deviation, 4 checkpoints)", [rep], AQUA),
]
fig, ax = plt.subplots(figsize=(8.4, 3.7))
for i, (name, vals, col) in enumerate(bars):
    y = len(bars) - 1 - i
    for v in vals:
        ax.plot([v], [y], "o", color=col, ms=8, mec=SURF, mew=1.5, zorder=4)
    ax.plot([min(vals), max(vals)], [y, y], color=col, lw=2, zorder=3)
    txt = ", ".join(f"{v:.3f}" for v in vals)
    ax.text(max(vals) + 0.012, y, txt + ("   (deterministic forward pass)" if col == AQUA else ""), va="center", fontsize=7.5, color=INK2)
ax.set_yticks(range(len(bars)))
ax.set_yticklabels([b[0] for b in bars][::-1], fontsize=8)
ax.set_xlim(-0.01, 0.52)
ax.set_xlabel("ADE@6 s (m)")
ax.set_title("Three variances, one axis: the interval answers the smallest question")
fig.tight_layout()
save(fig, "fig_variances.png")

# ======================================================================================================================
# fig_strategic: learned S minus the nav-only kinematic baseline (paired, banked compare.json)
# ======================================================================================================================
fig, ax = plt.subplots(figsize=(8.2, 3.9))
rows = []
for arm, nm in (("M1", "M1"), ("M2b", "M2b"), ("M1-navshuf", "M1, nav shuffled")):
    for arc in (50, 80):
        for s in (0, 1):
            d = cmp_["arms"][arm]["strat"][f"vs_nav_only_baseline@{arc}"][s]
            rows.append((f"{nm} s{s} @ {arc} m", arm, d["mean"], d["lo"], d["hi"]))
col_of = {"M1": ORANGE, "M2b": BLUE, "M1-navshuf": RED}
for i, (lab_, arm, m, lo, hi) in enumerate(rows):
    y = len(rows) - 1 - i
    ax.plot([lo, hi], [y, y], color=col_of[arm], lw=2.2, solid_capstyle="butt", zorder=3)
    ax.plot([m], [y], "o", color=col_of[arm], ms=6, mec=SURF, mew=1.2, zorder=4)
    ax.text(hi + 0.25, y, f"{m:+.2f}", va="center", fontsize=7.5, color=INK2)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=7.5)
ax.axvline(0, color=INK2, lw=0.8)
ax.set_xlabel("checkpoint position error, learned S minus nav-only kinematic baseline (m); positive = S is WORSE; paired 95% CI")
ax.set_title("H-MINI-STRAT-1 clause 1 on banked numbers: the learned S does not beat the baseline")
fig.tight_layout()
save(fig, "fig_strategic.png")

# ======================================================================================================================
# fig_architecture: schematic (no data)
# ======================================================================================================================
fig, ax = plt.subplots(figsize=(10.4, 4.9))
ax.set_xlim(0, 104)
ax.set_ylim(0, 49)
ax.axis("off")
ax.grid(False)


def box(x, y, w_, h_, text, fc, ec=None, fs=8, tc=INK, bold=False):
    p = FancyBboxPatch((x, y), w_, h_, boxstyle="round,pad=0.25,rounding_size=1.2", fc=fc, ec=ec or INK2, lw=1)
    ax.add_patch(p)
    ax.text(x + w_ / 2, y + h_ / 2, text, ha="center", va="center", fontsize=fs, color=tc, fontweight="bold" if bold else "normal")


def arrow(x0, y0, x1, y1, col=INK2, ls="-", txt=None, tx=0, ty=0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10, color=col, lw=1.3, ls=ls))
    if txt:
        ax.text((x0 + x1) / 2 + tx, (y0 + y1) / 2 + ty, txt, fontsize=7.2, color=col, ha="center", va="center")


box(1, 27, 19, 20, "INPUTS (no labels)\n\nframes -> frozen DINOv3-L\n pooled 4x10 tokens x 4 frames\n\nego p10f: v, a, kappa\n (measured yaw rate)\n\nnav cmd + distance\nmax-speed one-hot\nmeasured odometry", "#eef3fb", fs=7.3)
box(25, 36, 17, 10, "S  strategic   ~1 Hz\nreads: scene, nav, ego,\nmax speed", "#dbe8fa", bold=False)
box(25, 20, 17, 12, "T  tactical   ~2 Hz\nreads: S's ckpt (held),\nscene, nav, ego,\nmax speed", "#dbe8fa")
box(25, 3, 17, 12, "O  operative  10 Hz\nreads: T's code (held),\nS's ckpt (held), scene,\nego, nav, ages", "#dbe8fa")
box(50, 36, 17, 10, "route checkpoints\nck in R^(3x3)\n(x, y, psi) @ 30/50/80 m", "#f6e6d9")
box(50, 20, 17, 12, "tactical code  c in R^60\n[goal 22 | lat 5 | lon 7 |\n cons 26]  typed interface", "#f6e6d9")
box(50, 3, 17, 12, "tau = up(P10(f8)) + B c_res\n60 pts @ 0.1 s (6 s)\nramped DCT basis B, 12/axis", "#f6e6d9")
box(75, 3, 13, 12, "frozen P10 prior\nkd_x + MLP(f8)\n(30 knots @ 0.2 s)", "#e8e7e3")
box(75, 20, 27, 12, "label side (OFFLINE only):\nv9 labels from ego log + tracks:\nRC-A ckpts, 22 goals, 5+7 classes, 26 cells", "#e3f4ec", fs=7.3)
box(75, 36, 27, 10, "H-JEPA aux (M2b): z1/z2/z3 at\n0.2 / 0.4 / 0.8 s on the shared adapter;\nSIGReg + IDM; weight 0.1 (training only)", "#ece8fa", fs=7.3)
arrow(20, 40, 25, 41)
arrow(20, 33, 25, 26)
arrow(20, 30, 25, 12)
arrow(42, 41, 50, 41)
arrow(42, 26, 50, 26)
arrow(42, 9, 50, 9)
arrow(58.5, 36, 58.5, 32.2, col=INK2)
ax.text(60.2, 34.1, "held", fontsize=7, color=INK2)
arrow(33.5, 36, 33.5, 32.2)
arrow(33.5, 20, 33.5, 15.2)
arrow(75, 9, 67.2, 9, col=INK2)
arrow(58.5, 20, 58.5, 15.2)
arrow(75, 26, 67, 26, col=AQUA, ls="--")
ax.text(88.5, 17.4, "teacher forcing (training only):\nwhole code, p_gt 1.0 -> 0.2", fontsize=7.2, color="#0f7a55", ha="center", va="center")
arrow(75, 29, 67, 40.5, col=AQUA, ls="--")
ax.text(1, 22.5, "Inference reads exactly 14 batch keys\n(key guard). No label, no situation-\nclassifier output, no steering angle.", fontsize=7.4, color="#a0302f", va="top")
ax.text(1, 0.4, "Schematic (no data). Tick rates are as designed; on the 0.2 s cache grid T's tick is 0-2 steps old and S's 0-4 steps older.",
        fontsize=7.2, color=INK2)
save(fig, "fig_architecture.png")
