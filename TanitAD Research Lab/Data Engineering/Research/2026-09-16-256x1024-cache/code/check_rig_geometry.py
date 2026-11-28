"""G5 + G6 — geometry end to end, and the token grid MEASURED on a real frame.

G5 runs through ``tanitad/data/rig_projection.py`` (the function the BEV head and
every projection consumer use), NOT through a re-derivation here.

  * straight ahead (x=0, z=1) -> centre column: 511.5 at W=1024, 319.5 at W=640.
  * a KNOWN azimuth: +30 deg right -> col 767.5 at W=1024 (f_ref*pi/6 = 256.0
    exactly, because f_ref = 512/(pi/3)).
  * MIRROR (R-2026-09-08-wpa-mirror): the mirrored azimuth MUST NOT land on the
    same column. The test is run at a NON-ZERO azimuth on purpose -- a mirrored
    image is invisible at azimuth 0, which is how that defect escaped once.

G6 reads the token grid off a real timm resnet101 features_only forward on a real
cache frame. Arithmetic is not evidence of what the encoder emits.
"""
from __future__ import annotations
import json, math, os, sys

STACK = os.environ.get("TANITAD_STACK", r"C:/Users/Admin/tanitad-snap-20260915/stack")
sys.path.insert(0, STACK)
sys.path.insert(0, os.path.join(STACK, "scripts"))
import torch                                                        # noqa: E402
from tanitad.data.calib import CanonicalFrame                       # noqa: E402
from tanitad.data import rig_projection as RP                       # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
#: remaining argv are "tag=path" pairs, e.g. 256x1024=frame.png 408x1024=f2.png
FRAMES = dict(p.split("=", 1) for p in sys.argv[2:] if "=" in p)
#: legacy positional form (two paths) kept working
if not FRAMES and len(sys.argv) > 2:
    FRAMES = {"256x1024": sys.argv[2]}
    if len(sys.argv) > 3:
        FRAMES["256x640"] = sys.argv[3]
GEOMS = [(int(t.split("x")[0]), int(t.split("x")[1])) for t in FRAMES]
if not GEOMS:
    GEOMS = [(256, 1024), (256, 640)]
WIDTHS = sorted({w for _, w in GEOMS} | {640, 1024})
R = {"G5": {}, "G6": {}}
FAIL = []


def rec(tag, ok, **kw):
    R["G5"][tag] = {"pass": bool(ok), **kw}
    if not ok:
        FAIL.append(tag)
    print(f"  [{'PASS' if ok else 'FAIL'}] {tag}: "
          + " ".join(f"{k}={v}" for k, v in kw.items()), flush=True)


def g5():
    print("G5 geometry through tanitad/data/rig_projection.py", flush=True)
    for H, W in sorted(set(GEOMS) | {(256, 1024), (256, 640)}):
        want_c = (W - 1) / 2.0
        fr = CanonicalFrame.from_hfov(120.0, H, W, "cylindrical")
        # --- straight ahead -> centre column -----------------------------------
        p = torch.tensor([[0.0, 0.0, 1.0]])
        col, row, valid = RP.project_cam_to_frame(p, fr)
        c = float(col[0])
        rec(f"H{H}W{W}_straight_ahead_centre_col", abs(c - want_c) < 1e-6 and bool(valid[0]),
            col=round(c, 9), want=want_c, valid=bool(valid[0]))
        # --- a KNOWN azimuth: +30 deg to the right -----------------------------
        phi = math.radians(30.0)
        pr = torch.tensor([[math.sin(phi), 0.0, math.cos(phi)]])
        colr, _, vr = RP.project_cam_to_frame(pr, fr)
        want_r = want_c + fr.f_ref * phi
        rec(f"H{H}W{W}_az+30deg_col", abs(float(colr[0]) - want_r) < 1e-4 and bool(vr[0]),
            col=round(float(colr[0]), 6), want=round(want_r, 6),
            f_ref_times_phi=round(fr.f_ref * phi, 6))
        # --- MIRROR: must NOT agree, and must be the mirror about the centre ---
        pm = torch.tensor([[-math.sin(phi), 0.0, math.cos(phi)]])
        colm, _, _ = RP.project_cam_to_frame(pm, fr)
        cm, cr = float(colm[0]), float(colr[0])
        same = abs(cm - cr) < 1e-6
        rec(f"H{H}W{W}_mirror_REJECTED", (not same) and abs(cm - (W - 1 - cr)) < 1e-4,
            mirrored_col=round(cm, 6), original_col=round(cr, 6),
            claim_same_col_is=("TRUE(BUG)" if same else "FALSE(correct)"),
            equals_W_minus_1_minus_col=round(W - 1 - cr, 6))
        # --- round trip: column -> ray -> column --------------------------------
        cols = torch.tensor([0.0, want_c, W - 1.0])
        d = RP.frame_to_cam_ray(cols, torch.full_like(cols, (H - 1) / 2.0), fr)
        back, _, _ = RP.project_cam_to_frame(d, fr)
        err = float((back - cols).abs().max())
        rec(f"H{H}W{W}_col_ray_roundtrip", err < 1e-3, max_abs_px_err=round(err, 9))
        # the VERTICAL axis -- the one 408x1024 exists to restore
        cr_ = (H - 1) / 2.0
        dz = RP.frame_to_cam_ray(torch.tensor([want_c]), torch.tensor([cr_]), fr)
        elev = math.degrees(math.atan2(float(dz[0, 1]), float(dz[0, 2])))
        vfov = math.degrees(2 * math.atan((H / 2.0) / fr.f_ref))
        rec(f"H{H}W{W}_centre_row_is_boresight", abs(elev) < 1e-6,
            centre_row=cr_, elevation_deg=round(elev, 9), vfov_deg=round(vfov, 4))
        # --- the field the edges actually span ---------------------------------
        d0 = RP.frame_to_cam_ray(torch.tensor([0.0]), torch.tensor([(H - 1) / 2.0]), fr)
        d1 = RP.frame_to_cam_ray(torch.tensor([W - 1.0]), torch.tensor([(H - 1) / 2.0]), fr)
        a0 = math.degrees(math.atan2(float(d0[0, 0]), float(d0[0, 2])))
        a1 = math.degrees(math.atan2(float(d1[0, 0]), float(d1[0, 2])))
        span = a1 - a0
        want_span = 120.0 * (W - 1) / W
        rec(f"H{H}W{W}_edge_to_edge_deg", abs(span - want_span) < 1e-3,
            span_deg=round(span, 6), want=round(want_span, 6),
            note="centre-to-centre of the outer columns = HFOV*(W-1)/W")


def g6():
    """Read the token grid off a REAL forward. Arithmetic is not evidence of
    what the encoder emits."""
    print("G6 token grid MEASURED with timm resnet101 features_only", flush=True)
    import timm
    import torchvision.io as tvio
    import numpy as np
    m = timm.create_model("resnet101", pretrained=False, features_only=True)
    m.eval()
    red = m.feature_info.reduction()
    for tag, path in sorted(FRAMES.items()):
        H, W = (int(x) for x in tag.split("x"))
        if not path or not os.path.exists(path):
            R["G6"][tag] = {"skipped": f"no real frame at {path!r}"}
            print(f"  [SKIP] {tag}: no real frame", flush=True)
            continue
        img = tvio.decode_png(torch.from_numpy(np.fromfile(path, dtype="uint8")),
                              mode=tvio.ImageReadMode.RGB)
        assert tuple(img.shape[1:]) == (H, W), (tuple(img.shape), tag)
        with torch.no_grad():
            fs = m(img.float().div(255.0).unsqueeze(0))
        got = {}
        for f, st in zip(fs, red):
            gh, gw = int(f.shape[-2]), int(f.shape[-1])
            got[f"stride{st}"] = {"grid": [gh, gw], "tokens": gh * gw,
                                  "deg_per_col": round(120.0 / gw, 6),
                                  "channels": int(f.shape[1])}
        R["G6"][tag] = {"frame_png": os.path.basename(path),
                        "input_shape": list(img.shape), "levels": got}
        for st in (16, 32):
            k = f"stride{st}"
            if k in got:
                print(f"  {tag} {k}: grid {got[k]['grid'][0]}x{got[k]['grid'][1]}"
                      f" = {got[k]['tokens']} tokens, {got[k]['deg_per_col']}"
                      f" deg/col", flush=True)
    # PRE-REGISTERED expectations, checked against the MEASURED shapes.
    # deg/col depends only on W; the row count is H/stride (ceil).
    exp = {("256x640", "stride16"): (16, 40, 3.0),
           ("256x640", "stride32"): (8, 20, 6.0),
           ("256x1024", "stride16"): (16, 64, 1.875),
           ("256x1024", "stride32"): (8, 32, 3.75),
           ("408x1024", "stride16"): (26, 64, 1.875),
           ("408x1024", "stride32"): (13, 32, 3.75)}
    for (tag, k), (h, w, d) in sorted(exp.items()):
        if tag not in FRAMES:
            continue
        lv = R["G6"].get(tag, {}).get("levels", {}).get(k)
        ok = bool(lv) and lv["grid"] == [h, w] and lv["tokens"] == h * w             and abs(lv["deg_per_col"] - d) < 1e-6
        R["G6"].setdefault("checks", {})[f"{tag}_{k}"] = {
            "pass": ok,
            "expected": {"grid": [h, w], "tokens": h * w, "deg_per_col": d},
            "measured": lv}
        if not ok:
            FAIL.append(f"G6_{tag}_{k}")
        print(f"  [{'PASS' if ok else 'FAIL'}] G6 {tag} {k}", flush=True)


if __name__ == "__main__":
    g5()
    g6()
    R["failures"] = FAIL
    p = os.path.join(OUT, "check_rig_geometry.json")
    json.dump(R, open(p, "w"), indent=1)
    print(f"\n{'ALL PASS' if not FAIL else 'FAILURES: ' + ','.join(FAIL)} -> {p}")
    sys.exit(1 if FAIL else 0)
