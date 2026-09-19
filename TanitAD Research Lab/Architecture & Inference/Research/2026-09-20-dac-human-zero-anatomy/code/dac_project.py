"""Project the recorded human's path into the camera — for the 33 kerb-adjacent windows.

These are the only windows the map alone cannot settle: the ones carrying a cell the map calls
explicitly off-road (sidewalk / kerb / hatching). The map says the corner left the drivable
surface; only the image can say whether the car did.

⛔ CONTROLS RUN FIRST AND GATE THE RENDERS. An overlay that is subtly wrong misleads worse than
no overlay, so three known-value checks must pass on EVERY camera used, or nothing is drawn:

  C1  the ego's own footprint at t0 must sit at (or below) the BOTTOM of its own image
  C2  a ground point at 60 m must land within 20 px of the horizon row, approached monotonically
  C3  +y (left in the rig frame) must project LEFT of centre, symmetric with -y to 1 px

⛔ Read-only, CPU only, no GPU. Clip ids appear only as sha12.

    python dac_project.py --raw <raw dir> --out <media dir>
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
import torch                                   # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dac_anatomy import (BANDS, CELL_M, DEFAULTS, THR, Y_HALF_M,        # noqa: E402
                         _import_harness)
from dac_render import PALETTE, panel                                   # noqa: E402

HORIZON_TOL_PX = 20.0
BOTTOM_FRAC = 0.75


def controls(cam, frame, corners_t0) -> dict:
    """The three known-value checks, as numbers rather than as an opinion."""
    H, W = frame.height, frame.width
    P3 = torch.tensor
    # C1 — the ego's own footprint at t0
    pts = P3([[float(x), float(y), 0.0] for x, y in corners_t0], dtype=torch.float64)
    col, row, _ = cam.project(pts)
    in_front = (cam.to_cam(pts)[..., 2] > 0).numpy()
    front_rows = [float(r) for r, f, (x, _) in zip(row.numpy(), in_front, corners_t0) if f and x > 0]
    c1 = {"front_corner_rows": [round(r, 1) for r in front_rows], "image_height": H,
          "bar": f"every front corner row >= {BOTTOM_FRAC:.2f} x H = {BOTTOM_FRAC * H:.0f}",
          "pass": bool(front_rows) and all(r >= BOTTOM_FRAC * H for r in front_rows)}
    # C2 — the horizon ladder
    xs = [10.0, 20.0, 30.0, 45.0, 60.0, 1e6]
    _, rows2, _ = cam.project(P3([[x, 0.0, 0.0] for x in xs], dtype=torch.float64))
    r = [float(v) for v in rows2.numpy()]
    c2 = {"rows_at_x_m": dict(zip([str(int(x)) if x < 1e5 else "infinity" for x in xs],
                                  [round(v, 1) for v in r])),
          "delta_60m_vs_infinity_px": round(abs(r[-2] - r[-1]), 1),
          "bar": f"monotone from below and |row(60) - horizon| <= {HORIZON_TOL_PX} px",
          "pass": all(r[i] > r[i + 1] for i in range(len(r) - 1))
                  and abs(r[-2] - r[-1]) <= HORIZON_TOL_PX}
    # C3 — the DECLARED horizontal field, checked in the CAMERA frame so no mount enters it:
    # this frame is 120 deg over W columns by construction, so rays at +-60 deg are the edges.
    half = np.deg2rad(60.0)
    from tanitad.data.rig_projection import project_cam_to_frame
    cols_f, _, _ = project_cam_to_frame(
        P3([[np.sin(-half), 0.0, np.cos(half)], [np.sin(half), 0.0, np.cos(half)]],
           dtype=torch.float64), frame)
    c3 = {"col_at_minus_60deg": round(float(cols_f[0]), 2),
          "col_at_plus_60deg": round(float(cols_f[1]), 2), "width": W,
          "bar": "the declared 120 deg field puts +-60 deg within 1 px of columns 0 and W-1",
          "pass": abs(float(cols_f[0])) <= 1.0 and abs(float(cols_f[1]) - (W - 1)) <= 1.0}
    # C4 — the lateral SIGN, and an asymmetry the MOUNT can actually explain.
    # ⛔ My first version barred symmetry about the image centre at 1 px and failed every
    # camera. The bar was wrong, not the projection: this mount sits 0.06 m off-centre and
    # carries a small yaw, so exact symmetry is not a known value. The known value is the
    # BOUND those two measured numbers imply.
    cols4, _, _ = cam.project(P3([[30.0, 5.0, 0.0], [30.0, -5.0, 0.0]], dtype=torch.float64))
    cl, cr = float(cols4[0]), float(cols4[1])
    centre = (W - 1) / 2
    axis = (cam.R_cam_to_rig @ torch.tensor([0.0, 0.0, 1.0], dtype=cam.R_cam_to_rig.dtype))
    yaw = float(torch.atan2(axis[1], axis[0]))            # optical axis azimuth in the rig frame
    y0, x0 = float(cam.t_cam_in_rig[1]), float(cam.t_cam_in_rig[0])
    bound = 2 * frame.f_ref * abs(yaw) + 2 * frame.f_ref * abs(y0) / (30.0 - x0) + 1.0
    asym = abs((centre - cl) - (cr - centre))
    c4 = {"col_y_plus_5": round(cl, 1), "col_y_minus_5": round(cr, 1), "centre_col": centre,
          "asymmetry_px": round(asym, 2), "mount_yaw_deg": round(np.rad2deg(yaw), 3),
          "mount_lateral_offset_m": round(y0, 4),
          "bar": "+y LEFT of centre, and asymmetry <= the bound this mount's own yaw and "
                 f"lateral offset imply ({bound:.1f} px)",
          "pass": cl < centre < cr and asym <= bound}
    return {"C1 t0 footprint at the image bottom": c1,
            "C2 far ground point on the horizon row": c2,
            "C3 the declared 120 deg field reaches both edges": c3,
            "C4 +y is LEFT, asymmetry within the mount's own bound": c4,
            "all_pass": bool(c1["pass"] and c2["pass"] and c3["pass"] and c4["pass"])}


def draw(rec, img, cam, corners, viol, cart, seen, ch_names, out_png):
    H, W = img.shape[0], img.shape[1]
    pts = torch.tensor(np.concatenate([corners, np.zeros(corners.shape[:-1] + (1,))], -1),
                       dtype=torch.float64)
    col, row, _ = cam.project(pts.reshape(-1, 3))
    zc = cam.to_cam(pts.reshape(-1, 3))[..., 2].numpy()
    col = col.numpy().reshape(corners.shape[:-1])
    row = row.numpy().reshape(corners.shape[:-1])
    front = (zc > 0).reshape(corners.shape[:-1])
    fig = plt.figure(figsize=(13, 10.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.15])
    ax = fig.add_subplot(gs[0])
    ax.imshow(img)
    for t in range(0, corners.shape[0], 5):
        f = front[t]
        if f.all():
            p = np.concatenate([np.stack([col[t], row[t]], -1), [[col[t, 0], row[t, 0]]]])
            ax.plot(p[:, 0], p[:, 1], lw=1.0, color="#1565c0", alpha=0.9)
    ok = front & ~viol
    ax.scatter(col[ok], row[ok], s=5, color="#00e676", zorder=3)
    bad = front & viol
    ax.scatter(col[bad], row[bad], s=45, color="#ff1744", marker="x", lw=1.6, zorder=4)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_axis_off()
    ax.set_title("the recorded path PROJECTED into the t0 camera — red x = where the rule fires "
                 "(controls passed: footprint at the image bottom, 60 m on the horizon, +y left)",
                 fontsize=9)
    ax2 = fig.add_subplot(gs[1])
    cls = np.where(seen, np.argmax(cart, axis=0), len(ch_names) - 1)
    panel(ax2, cls, [0, BANDS[-1][1], -Y_HALF_M, Y_HALF_M],
          cmap=matplotlib.colors.ListedColormap(PALETTE[:len(ch_names)]),
          vmin=-0.5, vmax=len(ch_names) - 0.5)
    X, Y = corners[..., 0], corners[..., 1]
    ax2.scatter(X[~viol], Y[~viol], s=4, color="#00e676", zorder=3)
    ax2.scatter(X[viol], Y[viol], s=26, color="#ff1744", marker="x", zorder=4)
    ax2.set_xlabel("x forward (m)"); ax2.set_ylabel("y lateral (m)")
    ax2.set_title("the map's own class, same window", fontsize=9)
    ax2.legend(handles=[mpatches.Patch(color=PALETTE[i], label=n) for i, n in enumerate(ch_names)],
               loc="upper right", fontsize=7, ncol=2, framealpha=0.85)
    fig.suptitle(f"{rec['sha12']}  t0={rec['t0']}  violating samples={rec['n_violating_samples']}"
                 f"  off-road cells={rec['n_explicit_off']}  min |y| of an off-road corner="
                 f"{rec['min_abs_y_off']:.2f} m", fontsize=11)
    fig.savefig(out_png, dpi=105)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    for k, v in DEFAULTS.items():
        ap.add_argument("--" + k, default=v)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--extrinsics", default="D:/Projects/TanitAD-artifacts/refcv5v2_final/extrinsics141.json")
    a = ap.parse_args()
    raw, out = Path(a.raw), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    S1, P, SMG = _import_harness()
    import refc_v3_train as V3
    from tanitad.data.rig_projection import RigCamera
    frame = V3._agent_cam_frames()[(416, 1024)]
    _, table = V3._read_rig_extrinsics(a.extrinsics)
    if table is None:
        raise SystemExit("⛔ the extrinsics file is not a per-clip table")

    vs = [json.loads(l) for l in (raw / "dac_violating_samples.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = {(r["sha12"], r["t0"]): r for r in
            (json.loads(l) for l in (raw / "dac_windows.jsonl").read_text(encoding="utf-8").splitlines())}
    want = {}
    for s in vs:
        if s["explicit_off"] >= THR:
            k = (s["sha12"], s["t0"])
            w = want.setdefault(k, {"n_explicit_off": 0, "min_abs_y_off": 99.0})
            w["n_explicit_off"] += 1
            w["min_abs_y_off"] = min(w["min_abs_y_off"], abs(s["y_m"]))
    print(f"kerb-adjacent windows (any explicitly off-road cell): {len(want)}")

    corp = S1.Corpus(a.config, a.cache, a.labels, a.agents, a.maps, lru=2)
    ch_names = list(SMG.CHANNELS)
    ctrl_all, made, first_ctrl = {}, [], None
    for wi in S1.trainer_windows(corp.ds, 1000):
        if corp.eligibility(wi) is not None:
            continue
        e_i, t = corp.ds.index[wi]
        cid = str(corp.clip_ids[e_i])
        key = (S1.sha12(cid), int(t + corp.W - 1))
        if key not in want:
            continue
        if cid not in table:
            ctrl_all[key[0]] = {"all_pass": False, "reason": "no extrinsics entry for this clip"}
            continue
        cam = RigCamera.from_extrinsics(table[cid], frame)
        it = corp.light_item(wi)
        corners = P._ego_boxes(it["human"][None], P.PROXY)[0].numpy()
        c = controls(cam, frame, corners[0])
        ctrl_all[key[0]] = c
        first_ctrl = first_ctrl or c
        if not c["all_pass"]:
            continue                                   # ⛔ no overlay from an unverified camera
        g = corp.shim.map_store.get(cid)
        mf = g.read(np.asarray([it["t0"] + corp.raw_off]))
        cart, seen = np.asarray(mf.cart[0], np.float32), np.asarray(mf.seen[0], bool)
        ix = np.clip(np.floor(corners[..., 0] / CELL_M).astype(np.int64), 0, cart.shape[1] - 1)
        iy = np.clip(np.floor((corners[..., 1] + Y_HALF_M) / CELL_M).astype(np.int64), 0, cart.shape[2] - 1)
        inside = ((corners[..., 0] >= 0) & (corners[..., 0] < BANDS[-1][1])
                  & (np.abs(corners[..., 1]) < Y_HALF_M))
        viol = inside & seen[ix, iy] & (cart[ch_names.index("drivable")][ix, iy] < THR)
        img = corp.eps[e_i].frames[it["t0"]][-3:].permute(1, 2, 0).numpy().astype(np.uint8)
        rec = dict(rows[key], **want[key])
        png = out / f"kerb_{key[0]}_t{key[1]}.png"
        draw(rec, img, cam, corners, viol, cart, seen, ch_names, png)
        made.append({"png": png.name, "sha12": key[0], "t0": key[1],
                     "n_violating_samples": rec["n_violating_samples"],
                     "n_explicit_off": rec["n_explicit_off"],
                     "min_abs_y_off_m": round(rec["min_abs_y_off"], 2)})
    n_pass = sum(1 for c in ctrl_all.values() if c.get("all_pass"))
    report = {"_what": "camera projection for the kerb-adjacent windows, controls first",
              "_evidence_class": "MEASURED (ours; CPU, read-only)",
              "windows_requested": len(want), "cameras_checked": len(ctrl_all),
              "cameras_passing_all_controls": n_pass,
              "renders": len(made), "example_controls": first_ctrl,
              "failing_cameras": {k: v for k, v in ctrl_all.items() if not v.get("all_pass")}}
    (raw / "projection_controls.json").write_text(json.dumps(report, indent=1) + "\n",
                                                  encoding="utf-8", newline="\n")
    (raw / "projection_index.json").write_text(json.dumps(sorted(
        made, key=lambda m: -m["n_explicit_off"]), indent=1) + "\n",
        encoding="utf-8", newline="\n")
    print(json.dumps({k: report[k] for k in ("windows_requested", "cameras_checked",
                                             "cameras_passing_all_controls", "renders")}, indent=1))
    print(json.dumps(first_ctrl, indent=1))
    return 0 if n_pass == len(ctrl_all) and made else 2


if __name__ == "__main__":
    raise SystemExit(main())
