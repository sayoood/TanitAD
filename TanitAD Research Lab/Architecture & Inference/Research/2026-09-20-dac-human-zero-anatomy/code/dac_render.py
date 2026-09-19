"""Render the recorded human's path over the SAM3 map for a stratified sample of windows.

One PNG per window, two panels: the drivable FRACTION the rule reads, and the map's own
argmax CLASS. The footprint is drawn every 5th tick and every corner sample is plotted —
green where the rule is satisfied, red where it fires. A human eye can then judge whether the
car left the drivable surface or the rule misread a painted road marking.

Stratified by the anatomy's per-window violation count: worst, median, near-threshold (a
single violating sample), plus CLEAN controls that must read dac = 1 under every variant.

⛔ File names and titles carry sha12 only. Read-only, CPU only.

    python dac_render.py --raw <raw dir> --out <media dir> [--per-stratum 4]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches          # noqa: E402
import matplotlib.pyplot as plt                # noqa: E402
import numpy as np                             # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dac_anatomy import (BANDS, CELL_M, DEFAULTS, THR, Y_HALF_M,        # noqa: E402
                         _import_harness)

PALETTE = ["#9e9e9e", "#2e7d32", "#ffd54f", "#42a5f5", "#ab47bc",
           "#e53935", "#fb8c00", "#6d4c41", "#eceff1"]


def panel(ax, arr, extent, *, cmap, vmin=None, vmax=None):
    return ax.imshow(arr.T, origin="lower", extent=extent, aspect="equal",
                     cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")


def render(rec, cart, seen, corners, viol, ch_names, out_png, cam_img=None):
    X, Y = corners[..., 0], corners[..., 1]
    ext = [0, BANDS[-1][1], -Y_HALF_M, Y_HALF_M]
    dr, cls = cart[ch_names.index("drivable")], np.argmax(cart, axis=0)
    cls = np.where(seen, cls, len(ch_names) - 1)
    if cam_img is None:
        fig, axes = plt.subplots(2, 1, figsize=(13, 9), constrained_layout=True)
    else:
        fig = plt.figure(figsize=(13, 12), constrained_layout=True)
        gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.25, 1.25])
        cam_ax = fig.add_subplot(gs[0])
        cam_ax.imshow(cam_img)
        cam_ax.set_axis_off()
        # ⚠️ NOT projected: nothing here is drawn from the trajectory. It is the scene the
        # map was built from, for context only — a wrong overlay would mislead worse than none.
        cam_ax.set_title("camera at t0 — the newest stacked frame, 416x1024 cylindrical "
                         "(context only; the path is NOT projected into it)", fontsize=10)
        axes = [fig.add_subplot(gs[1]), fig.add_subplot(gs[2])]
    # ⚠️ NaN must NOT render white here: unseen would then look exactly like fully drivable,
    # which is the one confusion this panel exists to prevent.
    grey = matplotlib.colormaps["Greys_r"].with_extremes(bad="#4fc3f7")
    panel(axes[0], np.where(seen, dr, np.nan), ext, cmap=grey, vmin=0, vmax=1)
    axes[0].set_title("drivable fraction the rule reads "
                      "(white = 1, black = 0, blue = unseen, carries no evidence)")
    cmap = matplotlib.colors.ListedColormap(PALETTE[:len(ch_names)])
    panel(axes[1], cls, ext, cmap=cmap, vmin=-0.5, vmax=len(ch_names) - 0.5)
    axes[1].set_title("the map's own class (argmax)")
    axes[1].legend(handles=[mpatches.Patch(color=PALETTE[i], label=n)
                            for i, n in enumerate(ch_names)],
                   loc="upper right", fontsize=7, ncol=2, framealpha=0.85)
    for ax in axes:
        for t in range(0, X.shape[0], 5):                      # footprint every 0.5 s
            poly = corners[t][[0, 1, 2, 3, 0]]
            ax.plot(poly[:, 0], poly[:, 1], lw=0.8, color="#1565c0", alpha=0.8)
        ok = ~viol
        ax.scatter(X[ok], Y[ok], s=4, color="#00e676", zorder=3, label="corner: rule satisfied")
        ax.scatter(X[viol], Y[viol], s=26, color="#ff1744", marker="x", zorder=4,
                   label="corner: rule fires")
        ax.set_xlabel("x forward (m)")
        ax.set_ylabel("y lateral (m)")
        for b, _ in BANDS[1:]:
            ax.axvline(b, color="#90a4ae", lw=0.5, ls=":")
    fig.suptitle(f"{rec['sha12']}  t0={rec['t0']}  dac={rec['dac']:.0f}  "
                 f"violating samples={rec['n_violating_samples']}  "
                 f"[{rec['stratum']}]  — {rec['note']}", fontsize=11)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    for k, v in DEFAULTS.items():
        ap.add_argument("--" + k, default=v)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-stratum", type=int, default=4)
    a = ap.parse_args()
    raw = Path(a.raw)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    S1, P, SMG = _import_harness()
    rows = [json.loads(l) for l in (raw / "dac_windows.jsonl").read_text(encoding="utf-8").splitlines()]
    vs = [json.loads(l) for l in (raw / "dac_violating_samples.jsonl").read_text(encoding="utf-8").splitlines()]
    by_w: dict = {}
    for s in vs:
        by_w.setdefault((s["sha12"], s["t0"]), []).append(s)

    zeroed = sorted((r for r in rows if r["dac"] == 0.0), key=lambda r: r["n_violating_samples"])
    k = a.per_stratum

    def distinct(seq, n):
        # ⚠️ overlapping windows of ONE clip fill a stratum otherwise: the four worst windows
        # were t0 40/48/51/52 of the same clip, which shows one scene four times.
        seen_c, out_ = set(), []
        for r in seq:
            if r["sha12"] in seen_c:
                continue
            seen_c.add(r["sha12"])
            out_.append(r)
            if len(out_) == n:
                break
        return out_

    mid = len(zeroed) // 2
    strata = {"worst": distinct(reversed(zeroed), k),
              "median": distinct(zeroed[mid:], k),
              "near-threshold (1 sample)": distinct(
                  (r for r in zeroed if r["n_violating_samples"] == 1), k),
              "clean control (dac = 1)": distinct(
                  sorted((r for r in rows if r["dac"] == 1.0),
                         key=lambda r: -r["frac_samples_seen"]), 2)}
    want = {}
    for name, rs in strata.items():
        for r in rs:
            cats = [s for s in by_w.get((r["sha12"], r["t0"]), [])]
            note = "no violating sample" if not cats else (
                "every violating cell is road surface" if all(c["road_surface"] >= THR for c in cats)
                else ("has an explicitly off-road cell"
                      if any(c["explicit_off"] >= THR for c in cats) else "mixed"))
            want[(r["sha12"], r["t0"])] = dict(r, stratum=name, note=note)

    corp = S1.Corpus(a.config, a.cache, a.labels, a.agents, a.maps, lru=2)
    ch_names = list(SMG.CHANNELS)
    made = []
    for wi in S1.trainer_windows(corp.ds, 1000):
        if corp.eligibility(wi) is not None:
            continue
        e_i, t = corp.ds.index[wi]
        key = (S1.sha12(str(corp.clip_ids[e_i])), int(t + corp.W - 1))
        if key not in want:
            continue
        it = corp.light_item(wi)
        g = corp.shim.map_store.get(str(corp.clip_ids[e_i]))
        mf = g.read(np.asarray([it["t0"] + corp.raw_off]))
        cart, seen = np.asarray(mf.cart[0], np.float32), np.asarray(mf.seen[0], bool)
        corners = P._ego_boxes(it["human"][None], P.PROXY)[0].numpy()
        ix = np.floor(corners[..., 0] / CELL_M).astype(np.int64)
        iy = np.floor((corners[..., 1] + Y_HALF_M) / CELL_M).astype(np.int64)
        h, w = cart.shape[1], cart.shape[2]
        inside = (ix >= 0) & (ix < h) & (iy >= 0) & (iy < w)
        ixc, iyc = np.clip(ix, 0, h - 1), np.clip(iy, 0, w - 1)
        viol = inside & seen[ixc, iyc] & (cart[ch_names.index("drivable")][ixc, iyc] < THR)
        rec = want[key]
        png = out / f"{rec['stratum'].split()[0]}_{key[0]}_t{key[1]}.png"
        # the NEWEST raw frame of the stacked row: row j concatenates raw frames j..j+n_stack-1
        # (`v2_dataset._decode_stacked`), so the last 3 channels are the frame at t0.
        row = corp.eps[e_i].frames[it["t0"]]
        cam_img = row[-3:].permute(1, 2, 0).numpy().astype(np.uint8)
        render(rec, cart, seen, corners, viol, ch_names, png, cam_img=cam_img)
        made.append({"png": png.name, **{q: rec[q] for q in
                                         ("sha12", "t0", "dac", "n_violating_samples",
                                          "n_violating_ticks", "stratum", "note")}})
        if len(made) == len(want):
            break
    (raw / "render_index.json").write_text(json.dumps(made, indent=1) + "\n",
                                           encoding="utf-8", newline="\n")
    for m in made:
        print("%-34s dac %.0f  n_viol %3d  %s" % (m["png"], m["dac"], m["n_violating_samples"],
                                                  m["note"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
