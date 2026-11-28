"""lv2_sheet.py: the layered sheets of the tool `latvis2` (card PREREG_V8F_PROOF_ADD38 sections 3 and 6). One picture per window, one row per level of abstraction (L0 to L5),
columns present / imagined (2.0 s) / true future (2.0 s), plus the two change columns of L2. A level that does not exist for a window is an empty tile with its reason. Matplotlib, Agg.

  python lv2_sheet.py --windows WINDOWS_LV2.json --out-dir DIR [--tok G_s0=DUMP_LV2_G_s0.npz --tok RR_s0=...] [--imag LS_s0=DUMP_LS_s0.npz ...] [--frames-dir private/frames]

Two forms of every sheet: DIR/private/sheets/sheet_<window>.png (all levels; may hold camera content when frames are given) and DIR/sheets_public/sheet_<window>.png (L2 to L5 only: colours of
a 4 x 10 grid of derived values and numbers of the lead head, no camera picture), plus one contact sheet of each form. No picture is evidence for a claim; the numbers of the reads are."""
import argparse
import json
import os
import sys
import textwrap

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

NROW_G, NCOL_G = 4, 10
OFFSETS_ORDER = (-2.0, -1.0, 0.0, 1.0)
ARM_COLOURS = ("#1b6ca8", "#d1495b", "#2a9d8f", "#8d6b3f", "#6a4c93", "#e09f3e")
LEVEL_NAMES = {0: "L0  what the camera saw", 1: "L1  pictures decoded from the states", "2t": "L2  token grid the model sees (true tokens)", 2: "L2  token grid read from the state", 3: "L3  where the imagination moved the state",
               4: "L4  lead vehicle, top-down", 5: "L5  what if the ego brakes or accelerates"}
TIER_LINE = "Tier T0 (world-model diagnostic on logged frames, never driving performance). STRATEGIC: PAUSED (PI item 25). A picture is an illustration, not evidence."


# ------------------------------------------------------------------------------------------------ small drawing helpers
def empty_tile(ax, title, reason):
    ax.set_facecolor("#eeeeee")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#bbbbbb")
    ax.text(0.5, 0.5, textwrap.fill(reason, 30), ha="center", va="center", fontsize=8, color="#555555", transform=ax.transAxes)
    ax.set_title(title, fontsize=8.5, loc="left")


def grid_axes(ax, title):
    ax.set_title(title, fontsize=8.5, loc="left")
    ax.set_xticks(np.arange(NCOL_G)); ax.set_yticks(np.arange(NROW_G))
    ax.set_xticklabels([]); ax.set_yticklabels(["sky", "far", "near", "hood"], fontsize=6.5)
    ax.tick_params(length=0)


def rgb_from(col, lo, hi):
    """col [40, 3] -> [4, 10, 3] in 0..1 with one scale (lo, hi per component) for all tiles of the row."""
    x = (np.asarray(col, np.float64) - lo) / np.maximum(hi - lo, 1e-9)
    return np.clip(x, 0, 1).reshape(NROW_G, NCOL_G, 3)


def rgb_diff(dcol, scale):
    return np.clip(0.5 + 0.5 * np.asarray(dcol, np.float64) / max(scale, 1e-9), 0, 1).reshape(NROW_G, NCOL_G, 3)


def draw_topdown(ax, title, marks, ymax=60.0, note=None):
    """marks: list of dicts {gap [m], label, colour, kind: 'label'|'read', filled, dx}. Ego at 0 m at the bottom, the lead at its gap."""
    ax.set_title(title, fontsize=8.5, loc="left")
    ax.set_xlim(-2.2, 2.2); ax.set_ylim(-3, ymax)
    ax.add_patch(Rectangle((-1.4, -3), 2.8, ymax + 3, facecolor="#dcdcdc", edgecolor="none", zorder=0))
    for x in (-0.7, 0.7):
        ax.plot([x, x], [-3, ymax], color="white", lw=1, ls=(0, (4, 6)), zorder=1)
    ax.add_patch(Rectangle((-0.4, -3.0), 0.8, 2.6, facecolor="#444444", zorder=3))
    ax.text(0, -2.7, "ego", color="white", fontsize=6, ha="center", va="center", zorder=4)
    ax.set_xticks([]); ax.set_yticks([0, 10, 20, 30, 40, 50]); ax.tick_params(labelsize=6.5)
    ax.set_ylabel("gap [m]", fontsize=7)
    n = 0
    for m in marks:
        g = m["gap"]
        if g is None or not np.isfinite(g):
            continue
        n += 1
        if m["kind"] == "label":
            ax.add_patch(Rectangle((-0.4 + m.get("dx", 0), g), 0.8, 1.8, facecolor="#222222", edgecolor="none", alpha=0.85, zorder=3))
            ax.text(0.75 + m.get("dx", 0), g + 0.9, m["label"], fontsize=6.5, va="center", zorder=4)
        else:
            ax.plot([m.get("dx", 0)], [g + 0.9], marker="D", ms=6, mfc=(m["colour"] if m.get("filled", True) else "white"), mec=m["colour"], mew=1.4, ls="none", zorder=5)
            ax.text(m.get("dx", 0) + 0.18, g + 0.9, m["label"], fontsize=6, color=m["colour"], va="center", zorder=5)
    if note:
        ax.text(0.5, 0.97, note, transform=ax.transAxes, fontsize=6.5, ha="center", va="top", color="#444444")
    if n == 0:
        ax.text(0.5, 0.5, "no value", transform=ax.transAxes, ha="center", fontsize=8, color="#777777")


# ------------------------------------------------------------------------------------------------ the data of a window
def imag_row(dump, eid, g0):
    hit = np.flatnonzero((dump["eid"] == eid) & (dump["g0"] == g0))
    return int(hit[0]) if len(hit) else None


def load_npz(path):
    z = np.load(path, allow_pickle=True)
    d = {k: z[k] for k in z.files}
    d["eid"] = d["eid"].astype(str)
    return d


def tok_info(tokd, widx):
    """-> (colours true [2,40,3], decoded [3,40,3] imagined/present/true, dist_pres [40], dist_imag [40]) or (None, reason)."""
    out = {"dist_pres": tokd["dist_pres"][widx], "dist_imag": tokd["dist_imag"][widx], "err": (tokd["err_pres"][widx], tokd["err_imag"][widx], tokd["err_true"][widx])}
    pw = tokd.get("panel_win")
    pos = None if pw is None else np.flatnonzero(pw == widx)
    if pos is None or len(pos) == 0:
        out["col"] = None
    else:
        out["col"] = (tokd["col_true"][pos[0]], tokd["col_dec"][pos[0]])
    return out


# ------------------------------------------------------------------------------------------------ one sheet
def render_sheet(win, toks, imags, frame_npz, out_path, private):
    """win: window record of WINDOWS_LV2.json; toks: {arm: token dump dict}; imags: {arm: imaghead dump dict}; frame_npz: path or None."""
    widx, eid, g0 = int(win["window"]), str(win["eid"]), int(win["g0"])
    levels = [0, 1, 2, 3, 4, 5] if private else [2, 3, 4, 5]
    tok_arms = list(toks)
    rows = []
    for L in levels:
        if L in (2, 3):
            if L == 2 and tok_arms:
                rows.append(("2t", tok_arms[0]))
            rows += [(L, a) for a in (tok_arms or [None])]
        else:
            rows.append((L, None))
    fig = plt.figure(figsize=(16, 2.55 * len(rows) + 1.2), dpi=100)
    gs = fig.add_gridspec(len(rows), 6, width_ratios=[0.9, 1, 1, 1, 1, 1], left=0.02, right=0.99, top=1 - 1.45 / (2.55 * len(rows) + 1.2), bottom=0.02, hspace=0.42, wspace=0.18)
    grp = win.get("group", "")
    title = f"window {widx}  |  episode {eid[:8]}  step {g0}  |  selected as: {grp.replace('_', ' ')}"
    fig.text(0.02, 1 - 0.22 / (2.55 * len(rows) + 1.2), title, fontsize=13, weight="bold", va="top")
    fig.text(0.02, 1 - 0.60 / (2.55 * len(rows) + 1.2), TIER_LINE, fontsize=8, color="#444444", va="top")
    l2s = l2_scale(toks, widx)
    for r, (L, arm) in enumerate(rows):
        lab = fig.add_subplot(gs[r, 0]); lab.axis("off")
        txt = LEVEL_NAMES[L] + (f"\n[{arm}]" if arm else "")
        extra = {0: "frame at t and at t + 2.0 s", 1: "decoded tiles of the earlier pixel reads", "2t": "colours = 3 main token directions; one colour scale for every L2 tile of the sheet", 2: "the same colours, read from the model state by a linear map fitted on other episodes",
                 3: "squared distance of the state to the true future state, per cell", 4: "black = label; diamonds = lead head's read", 5: "imagined gap at 1.0 s and 2.0 s under an offset of the ego acceleration"}[L]
        wt = textwrap.fill(LEVEL_NAMES[L], 20) + (f"\n[{arm}]" if arm and L != "2t" else "")
        lab.text(0, 0.97, wt, fontsize=9.5, weight="bold", va="top")
        lab.text(0, 0.45, textwrap.fill(extra, 28), fontsize=7.5, va="top", color="#444444")
        axs = [fig.add_subplot(gs[r, c]) for c in range(1, 6)]
        _row(L, arm, axs, widx, eid, g0, toks, imags, frame_npz, win, l2s)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


def l2_scale(toks, widx):
    """One colour scale (lo, hi per colour component, and the scale of the change tiles) for every L2 tile of a sheet, over the true tokens and every arm's reads."""
    cols, diffs = [], []
    for a in toks:
        ti = tok_info(toks[a], widx)
        if ti["col"] is None:
            continue
        ct, cd = ti["col"]
        cols += [ct.reshape(-1, 3), cd.reshape(-1, 3)]
        diffs += [(ct[1] - ct[0]).ravel(), (cd[0] - cd[1]).ravel()]
    if not cols:
        return None
    allc = np.concatenate(cols)
    return np.percentile(allc, 2, axis=0), np.percentile(allc, 98, axis=0), float(np.percentile(np.abs(np.concatenate(diffs)), 98))


def _row(L, arm, axs, widx, eid, g0, toks, imags, frame_npz, win, l2s=None):
    names = ["present (time t)", "imagined (t + 2.0 s)", "true future (t + 2.0 s)", "change: truth", "change: imagination"]
    if L == 0:
        if frame_npz and os.path.exists(frame_npz):
            z = np.load(frame_npz)
            fr = z["frames"]
            axs[0].imshow(fr[0]); axs[0].set_title("present: camera frame at t", fontsize=8.5, loc="left")
            axs[2].imshow(fr[1]); axs[2].set_title("true future: camera frame at t + 2.0 s", fontsize=8.5, loc="left")
            empty_tile(axs[1], "imagined (t + 2.0 s)", "the model does not imagine camera pictures")
            for i in (0, 2):
                axs[i].set_xticks([]); axs[i].set_yticks([])
            for i in (3, 4):
                axs[i].axis("off")
        else:
            for i, nm in enumerate(names[:3]):
                empty_tile(axs[i], nm, "no camera frame on this machine for this window" if i != 1 else "the model does not imagine camera pictures")
            axs[3].axis("off"); axs[4].axis("off")
    elif L == 1:
        for i, nm in enumerate(names[:3]):
            empty_tile(axs[i], nm, "decoded tile of the earlier pixel read is not cropped by this tool version")
        axs[3].axis("off"); axs[4].axis("off")
    elif L == "2t":
        ti = tok_info(toks[arm], widx)
        if ti["col"] is None or l2s is None:
            for i, nm in enumerate(names):
                empty_tile(axs[i], nm, "colours were not stored for this window (pod run without it in --panel-windows)")
            return
        ct, _ = ti["col"]
        lo, hi, sc = l2s
        axs[0].imshow(rgb_from(ct[0], lo, hi), aspect="auto", interpolation="nearest"); grid_axes(axs[0], "present (time t): true tokens")
        empty_tile(axs[1], "imagined (t + 2.0 s)", "the truth has no imagined grid; the rows below show what each model state holds")
        axs[2].imshow(rgb_from(ct[1], lo, hi), aspect="auto", interpolation="nearest"); grid_axes(axs[2], "true future (t + 2.0 s): true tokens")
        axs[3].imshow(rgb_diff(ct[1] - ct[0], sc), aspect="auto", interpolation="nearest"); grid_axes(axs[3], "change of the true tokens (grey = none)")
        axs[4].axis("off")
    elif L in (2, 3):
        if arm is None:
            for i, nm in enumerate(names):
                empty_tile(axs[i], nm, "no token dump given")
            return
        ti = tok_info(toks[arm], widx)
        if L == 2:
            if ti["col"] is None:
                for i, nm in enumerate(names):
                    empty_tile(axs[i], nm, "colours were not stored for this window (pod run without it in --panel-windows)")
                return
            ct, cd = ti["col"]
            allc = np.concatenate([ct.reshape(-1, 3), cd.reshape(-1, 3)])
            lo, hi = np.percentile(allc, 2, axis=0), np.percentile(allc, 98, axis=0)
            if l2s is not None:
                lo, hi = l2s[0], l2s[1]
            for i, g in enumerate((cd[1], cd[0], cd[2])):
                axs[i].imshow(rgb_from(g, lo, hi), aspect="auto", interpolation="nearest"); grid_axes(axs[i], names[i] + " (read from the state)")
            dt = ct[1] - ct[0]
            di = cd[0] - cd[1]
            sc = float(np.percentile(np.abs(np.concatenate([dt.ravel(), di.ravel()])), 98)) if l2s is None else l2s[2]
            axs[3].imshow(rgb_diff(dt, sc), aspect="auto", interpolation="nearest"); grid_axes(axs[3], "change of the true tokens (grey = none)")
            axs[4].imshow(rgb_diff(di, sc), aspect="auto", interpolation="nearest"); grid_axes(axs[4], "change the imagination makes")
        else:
            dp, di_ = ti["dist_pres"].reshape(NROW_G, NCOL_G), ti["dist_imag"].reshape(NROW_G, NCOL_G)
            vmax = float(np.nanmax([dp.max(), di_.max()]))
            for i, (g, nm) in enumerate(((dp, "present state to the true future"), (di_, "imagined state to the true future"))):
                im = axs[i].imshow(g, aspect="auto", interpolation="nearest", cmap="viridis_r", vmin=0, vmax=vmax); grid_axes(axs[i], nm)
            plt.colorbar(im, ax=axs[1], fraction=0.046, pad=0.02).ax.tick_params(labelsize=6)
            diff = dp - di_
            m = float(np.abs(diff).max()) or 1.0
            im2 = axs[2].imshow(diff, aspect="auto", interpolation="nearest", cmap="RdBu_r", vmin=-m, vmax=m); grid_axes(axs[2], "present minus imagined (red = moved toward the future)")
            plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.02).ax.tick_params(labelsize=6)
            e_pres, e_imag, e_true = ti["err"]
            axs[3].axis("off"); axs[4].axis("off")
            axs[3].text(0, 0.7, f"token error of the map (sum over cells)\npresent state {e_pres.sum():.1f}\nimagined state {e_imag.sum():.1f}\ntrue future state {e_true.sum():.1f}", fontsize=8, va="top", transform=axs[3].transAxes)
    elif L == 4:
        _l4(axs, widx, eid, g0, imags, win)
        axs[3].axis("off"); axs[4].axis("off")
    elif L == 5:
        _l5(axs, widx, eid, g0, imags)
        axs[2].axis("off"); axs[3].axis("off"); axs[4].axis("off")


def _no_lead(win):
    return "no lead vehicle labelled at t (the lead head is read only where a label exists)"


def _l4(axs, widx, eid, g0, imags, win):
    arms = list(imags)
    rows = {a: imag_row(imags[a], eid, g0) for a in arms}
    have = {a: r for a, r in rows.items() if r is not None}
    if not arms or not have:
        for i, nm in enumerate(("present (time t)", "imagined (t + 1.0 s, t + 2.0 s)", "true future (t + 2.0 s)")):
            empty_tile(axs[i], nm, _no_lead(win) if arms else "no lead-head dump given")
        return
    d0 = imags[next(iter(have))]
    r0 = have[next(iter(have))]
    lab_t, lab_1, lab_2 = float(d0["gap_t"][r0]), float(d0["gap_t1"][r0]), float(d0["gap_t2"][r0])
    marks = [[{"kind": "label", "gap": lab_t, "label": f"label {lab_t:.1f} m"}], [{"kind": "label", "gap": lab_2, "label": f"label at 2.0 s {lab_2:.1f} m", "dx": 0.0}],
             [{"kind": "label", "gap": lab_2, "label": f"label {lab_2:.1f} m"}]]
    for k, (a, r) in enumerate(have.items()):
        c, dx = ARM_COLOURS[k % len(ARM_COLOURS)], -0.9 + 0.5 * k
        d = imags[a]
        marks[0].append({"kind": "read", "gap": float(d["present"][r, 1]), "label": f"{a} {d['present'][r, 1]:.1f}", "colour": c, "dx": dx})
        marks[1].append({"kind": "read", "gap": float(d["roll_logged"][r, 0, 1]), "label": f"{a} 1.0 s {d['roll_logged'][r, 0, 1]:.1f}", "colour": c, "dx": dx, "filled": False})
        marks[1].append({"kind": "read", "gap": float(d["roll_logged"][r, 1, 1]), "label": f"{a} 2.0 s {d['roll_logged'][r, 1, 1]:.1f}", "colour": c, "dx": dx})
        marks[2].append({"kind": "read", "gap": float(d["true2"][r, 1]), "label": f"{a} {d['true2'][r, 1]:.1f}", "colour": c, "dx": dx})
    ymax = max(60.0, 1.15 * np.nanmax([m["gap"] for ms in marks for m in ms if np.isfinite(m["gap"])] + [1.0]))
    for i, (nm, mk) in enumerate(zip(("present (time t)", "imagined (hollow 1.0 s, filled 2.0 s)", "true future (t + 2.0 s)"), marks)):
        draw_topdown(axs[i], nm, mk, ymax=float(ymax))


def _l5(axs, widx, eid, g0, imags):
    arms = list(imags)
    have = {a: imag_row(imags[a], eid, g0) for a in arms}
    have = {a: r for a, r in have.items() if r is not None}
    if not have:
        for i, nm in enumerate(("imagined gap at 1.0 s", "imagined gap at 2.0 s")):
            empty_tile(axs[i], nm, _no_lead(None) if arms else "no lead-head dump given")
        return
    for t_i, (T, ax) in enumerate(((1.0, axs[0]), (2.0, axs[1]))):
        ax.set_title(f"imagined gap at {T:.1f} s\nunder an offset of the ego acceleration", fontsize=8.5, loc="left")
        ax.set_xlabel("offset of the ego acceleration [m/s2]", fontsize=7); ax.set_ylabel("gap [m]", fontsize=7); ax.tick_params(labelsize=6.5); ax.grid(alpha=0.3)
        for k, (a, r) in enumerate(have.items()):
            d = imags[a]
            offs = np.asarray(d["offsets"], float)
            order = np.argsort(offs)
            y = d["roll_off"][r, :, t_i, 1][order]
            c = ARM_COLOURS[k % len(ARM_COLOURS)]
            ax.plot(offs[order], y, "o-", color=c, label=f"{a}: imagined")
            i0 = int(np.flatnonzero(offs == 0.0)[0])
            kin = d["roll_off"][r, i0, t_i, 1] - 0.5 * offs[order] * T ** 2
            ax.plot(offs[order], kin, "--", color=c, alpha=0.8, label=f"{a}: kinematics line")
        ax.legend(fontsize=6, loc="best")


def contact_sheet(paths, out, ncol=6, width=420):
    from PIL import Image
    ims = [Image.open(p).convert("RGB") for p in paths]
    if not ims:
        return None
    th = [im.resize((width, int(im.height * width / im.width))) for im in ims]
    h = max(t.height for t in th)
    nrow = (len(th) + ncol - 1) // ncol
    sheet = Image.new("RGB", (ncol * width, nrow * h), "white")
    for i, t in enumerate(th):
        sheet.paste(t, ((i % ncol) * width, (i // ncol) * h))
    sheet.save(out)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tok", action="append", default=[], help="ARM=DUMP_LV2_<ARM>.npz of lv2_pod_tokens.py (L2, L3)")
    ap.add_argument("--imag", action="append", default=[], help="ARM=DUMP_<ARM>.npz of imag_dump.py (L4, L5)")
    ap.add_argument("--frames-dir", default=None, help="private/frames of lv2_frames.py (L0)")
    ap.add_argument("--only", default=None, help="comma list of window indices")
    ap.add_argument("--levels-public-only", action="store_true", help="write the public form only")
    ap.add_argument("--ncol", type=int, default=6)
    a = ap.parse_args(argv)
    W = json.load(open(a.windows, encoding="utf-8"))
    toks = {s.split("=", 1)[0]: load_npz(s.split("=", 1)[1]) for s in a.tok}
    imags = {s.split("=", 1)[0]: load_npz(s.split("=", 1)[1]) for s in a.imag}
    wins = W["windows"]
    if a.only:
        keep = {int(x) for x in a.only.split(",")}
        wins = [w for w in wins if int(w["window"]) in keep]
    forms = [("sheets_public", False)] + ([] if a.levels_public_only else [("private/sheets", True)])
    made = {}
    for sub, priv in forms:
        d = os.path.join(a.out_dir, sub)
        os.makedirs(d, exist_ok=True)
        if priv:
            with open(os.path.join(a.out_dir, "private", "PRIVATE_DO_NOT_LAND.txt"), "w", encoding="utf-8", newline="\n") as fh:
                fh.write("Sheets in this folder may hold camera frames or decoded pixel tiles. Private: dev box only, shown to the PI only, never landed or banked.\n")
        paths = []
        for w in wins:
            fp = os.path.join(a.frames_dir, f"frames_{int(w['window'])}.npz") if a.frames_dir else None
            paths.append(render_sheet(w, toks, imags, fp, os.path.join(d, f"sheet_{int(w['window'])}.png"), priv))
        made[sub] = paths
        contact_sheet(paths, os.path.join(d, "contact_sheet.png"), a.ncol)
    print(f"lv2_sheet: {len(wins)} windows, forms {list(made)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
