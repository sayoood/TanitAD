"""G0 / G1 / G2 / G3 -- does the 256x1024 cache hold the SAME projection as 256x640?

The exact correspondence is derived, not fitted (see PREREG.md §1):
  columns  k_1024 = 1.6*j_640 + 0.3  -> a 1/1.6 resize maps output col c to 640 col c
  rows     i_1024 = 127.5 + 1.6*(i'_640 - 127.5) -> output row r to 640 row 48 + r
so the comparison is ``resize(frame1024, (160,640))`` against ``frame640[48:208]``,
integer-aligned on both axes with NO interpolation of the reference.

G3 mutations run the SAME comparison against deliberately broken inputs. A check
that only ever sees the good case has not been shown to have teeth.
"""
from __future__ import annotations
import argparse, json, math, os, sys

STACK = os.environ.get("TANITAD_STACK", r"C:/Users/Admin/tanitad-snap-20260915/stack")
sys.path.insert(0, STACK)
sys.path.insert(0, os.path.join(STACK, "scripts"))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                                                # noqa: BLE001
        pass
import numpy as np                                                   # noqa: E402
import torch                                                         # noqa: E402
import torch.nn.functional as F                                      # noqa: E402
import torchvision.io as tvio                                        # noqa: E402

# PRE-REGISTERED TOLERANCES (PREREG.md §2) -- fixed before any measurement.
TOL = {"G0": {"mae": 1.0, "psnr": 40.0, "pose_absmax": 1e-6},
       "G2": {"median": 2.0, "mae": 4.0, "psnr": 28.0}}
SEARCH = 12               # +-px window for the G1 cross-correlation peak
RATIO = 1.6               # f_ref(1024)/f_ref(640), exact


def load_payload(p):
    d = torch.load(p, map_location="cpu", weights_only=False)
    return d


def frames_of(d, idx):
    """Decode the PNG frames at ``idx`` -> uint8 [n,3,H,W]."""
    off = torch.cat([torch.zeros(1, dtype=torch.int64), d["jpeg_len"].cumsum(0)])
    out = []
    for i in idx:
        out.append(tvio.decode_png(d["jpeg_buf"][off[i]:off[i + 1]],
                                   mode=tvio.ImageReadMode.RGB))
    return torch.stack(out)


def pick_frames(n, k=5):
    """Deterministic, spread, chosen BEFORE looking at content (PREREG §3)."""
    return [int(round(x)) for x in np.linspace(0, n - 1, k)]


def stats(a, b, mask):
    """a, b uint8 [3,H,W]; mask bool [H,W]. Returns the pre-registered metrics."""
    d = (a.float() - b.float()).abs()
    m = mask.unsqueeze(0).expand_as(d)
    v = d[m]
    if v.numel() == 0:
        return {"n_px": 0}
    mse = float((v ** 2).mean())
    return {"n_px": int(v.numel()),
            "mask_frac": round(float(mask.float().mean()), 6),
            "median": round(float(v.median()), 4),
            "mae": round(float(v.mean()), 4),
            "p99": round(float(v.quantile(0.99)), 4),
            "max": round(float(v.max()), 4),
            "psnr_db": round(20 * math.log10(255.0 / max(math.sqrt(mse), 1e-9)), 3)}


def xcorr_peak(a, b, w=SEARCH):
    """Integer-shift cross-correlation peak of two [H,W] float maps."""
    a = a - a.mean(); b = b - b.mean()
    H, W = a.shape
    ca = a[w:H - w, w:W - w]
    best, arg = -1e30, (None, None)
    for dy in range(-w, w + 1):
        for dx in range(-w, w + 1):
            cb = b[w + dy:H - w + dy, w + dx:W - w + dx]
            s = float((ca * cb).sum())
            if s > best:
                best, arg = s, (dy, dx)
    return arg


def compare(a_u8, b_u8, tag, tol):
    """One comparison -> metrics + pass/fail against the pre-registered tolerance."""
    mask = (a_u8.sum(0) > 0) & (b_u8.sum(0) > 0)
    st = stats(a_u8, b_u8, mask)
    g = a_u8.float().mean(0); h = b_u8.float().mean(0)
    dy, dx = xcorr_peak(g, h)
    st["xcorr_peak"] = [dy, dx]
    st["G1_geometry_pass"] = (dy == 0 and dx == 0)
    st["G2_photometric_pass"] = (st.get("median", 1e9) <= tol["median"]
                                 and st.get("mae", 1e9) <= tol["mae"]
                                 and st.get("psnr_db", -1e9) >= tol["psnr"])
    st["tag"] = tag
    return st


def down(x_u8):
    """1024 -> 640 at the EXACT pre-registered grid: (256,1024) -> (160,640)."""
    return F.interpolate(x_u8.float().unsqueeze(0), size=(160, 640),
                         mode="bilinear", align_corners=False,
                         antialias=True).squeeze(0).round().clamp(0, 255).to(
                             torch.uint8)


# --------------------------------------------------------------------------- #
def g0(args, R):
    """Builder parity: THIS box's 640 build vs Thor's DEPLOYED 640 payload."""
    print("G0 builder parity vs the deployed physicalai-b1-w120-256x640cyl",
          flush=True)
    rows, fail = [], []
    for f in sorted(os.listdir(args.g0_local)):
        if not f.endswith(".v2ep.pt"):
            continue
        ref_p = os.path.join(args.g0_thor, f)
        if not os.path.exists(ref_p):
            continue
        a, b = load_payload(os.path.join(args.g0_local, f)), load_payload(ref_p)
        import hashlib
        s12 = hashlib.sha256(a["clip_id"].encode()).hexdigest()[:12]
        na, nb = int(a["jpeg_len"].shape[0]), int(b["jpeg_len"].shape[0])
        pose_d = (float((a["poses"] - b["poses"]).abs().max())
                  if na == nb else float("nan"))
        act_d = (float((a["actions"] - b["actions"]).abs().max())
                 if na == nb else float("nan"))
        idx = pick_frames(min(na, nb), args.frames)
        fa, fb = frames_of(a, idx), frames_of(b, idx)
        per = [compare(fa[i], fb[i], f"{s12}@{idx[i]}", TOL["G2"]) for i in
               range(len(idx))]
        mae = max(p["mae"] for p in per); psnr = min(p["psnr_db"] for p in per)
        ok = (na == nb and pose_d <= TOL["G0"]["pose_absmax"]
              and act_d <= TOL["G0"]["pose_absmax"]
              and mae <= TOL["G0"]["mae"] and psnr >= TOL["G0"]["psnr"]
              and all(p["G1_geometry_pass"] for p in per))
        row = {"clip_sha12": s12, "n_frames_local": na, "n_frames_thor": nb,
               "poses_absmax": pose_d, "actions_absmax": act_d,
               "worst_mae": mae, "worst_psnr_db": psnr,
               "frames": idx, "pass": ok, "per_frame": per}
        rows.append(row)
        if not ok:
            fail.append(s12)
        print(f"  [{'PASS' if ok else 'FAIL'}] {s12}: n={na}/{nb} "
              f"poses_absmax={pose_d:g} actions_absmax={act_d:g} "
              f"worst_mae={mae:.4f} worst_psnr={psnr:.2f} dB", flush=True)
    R["G0"] = {"tolerance": TOL["G0"], "clips": rows, "failures": fail}
    return fail


def g1g2g3(args, R):
    """Downscale identity and its deliberate regressions."""
    print("\nG1/G2 downscale identity: resize(1024,(160,640)) vs 640[48:208]",
          flush=True)
    import hashlib
    ids = [l.strip() for l in open(args.ids) if l.strip()][:args.clips]
    rows, fail, mut = [], [], {}
    for cid in ids:
        s12 = hashlib.sha256(cid.encode()).hexdigest()[:12]
        pa = os.path.join(args.wide, f"{cid}.v2ep.pt")
        pb = os.path.join(args.ref, f"{cid}.v2ep.pt")
        if not (os.path.exists(pa) and os.path.exists(pb)):
            print(f"  [SKIP] {s12}: missing payload", flush=True); continue
        a, b = load_payload(pa), load_payload(pb)
        assert (int(a["image_w"]), int(a["image_h"])) == (1024, 256)
        assert (int(b["image_w"]), int(b["image_h"])) == (640, 256)
        assert abs(a["frame"]["f_ref"] / b["frame"]["f_ref"] - RATIO) < 1e-9
        n = min(int(a["jpeg_len"].shape[0]), int(b["jpeg_len"].shape[0]))
        idx = pick_frames(n, args.frames)
        fa, fb = frames_of(a, idx), frames_of(b, idx)
        per = []
        for i, fi in enumerate(idx):
            d = down(fa[i])                       # [3,160,640]
            ref = fb[i][:, 48:208, :]             # the exactly-corresponding band
            per.append(compare(d, ref, f"{s12}@{fi}", TOL["G2"]))
        ok = all(p["G1_geometry_pass"] and p["G2_photometric_pass"] for p in per)
        rows.append({"clip_sha12": s12, "n_frames": n, "frames": idx,
                     "pass": ok, "per_frame": per})
        if not ok:
            fail.append(s12)
        w = max(p["mae"] for p in per); q = min(p["psnr_db"] for p in per)
        md = max(p["median"] for p in per)
        print(f"  [{'PASS' if ok else 'FAIL'}] {s12}: frames={idx} "
              f"worst median={md:.3f} mae={w:.3f} psnr={q:.2f} dB "
              f"peaks={[p['xcorr_peak'] for p in per]}", flush=True)

        if cid == ids[0]:                       # mutations on the FIRST clip only
            print("\nG3 deliberate regressions (each MUST fail)", flush=True)
            i = 0
            d, ref = down(fa[i]), fb[i][:, 48:208, :]
            m2 = compare(d, fb[i][:, 0:160, :], "M2_wrong_row_crop", TOL["G2"])
            m3 = compare(torch.flip(d, dims=[2]), ref, "M3_mirrored", TOL["G2"])
            mut["M2_wrong_row_crop"] = m2
            mut["M3_mirrored_azimuth"] = m3
            for k, v in (("M2_wrong_row_crop", m2), ("M3_mirrored_azimuth", m3)):
                broke = not (v["G1_geometry_pass"] and v["G2_photometric_pass"])
                v["MUTATION_DETECTED"] = broke
                print(f"  [{'DETECTED' if broke else 'MISSED(BUG)'}] {k}: "
                      f"peak={v['xcorr_peak']} median={v['median']} "
                      f"mae={v['mae']} psnr={v['psnr_db']} dB", flush=True)
                if not broke:
                    fail.append(k)
    R["G1_G2"] = {"tolerance": TOL["G2"], "search_window_px": SEARCH,
                  "clips": rows, "failures": fail}
    R["G3_mutations"] = mut
    return fail


def g3_unscaled_fref(args, R):
    """M1: build ONE clip at W=1024 with the 640 focal -- the defect the brief
    warns about -- and show the SAME check rejects it."""
    print("\nG3/M1 unscaled f_ref (W=1024 built with f_ref=305.5774907)",
          flush=True)
    import hashlib
    import v2_compressed as V
    from tanitad.data.calib import CanonicalFrame
    cid = [l.strip() for l in open(args.ids) if l.strip()][0]
    s12 = hashlib.sha256(cid.encode()).hexdigest()[:12]
    bad_frame = CanonicalFrame(height=256, width=1024, f_ref=305.5774907364391,
                               projection="cylindrical")
    got_hfov = math.degrees(2 * (1024 / 2) / bad_frame.f_ref)
    out = os.path.join(args.tmp, f"{cid}.MUTANT-f305.v2ep.pt")
    os.makedirs(args.tmp, exist_ok=True)
    if not os.path.exists(out):
        V.build_compressed(
            {"clip_id": cid,
             "mp4": os.path.join(args.root, "r0", "camera_front_wide", f"{cid}.mp4"),
             "timestamps": os.path.join(args.root, "r0", "camera_front_wide",
                                        f"{cid}.timestamps.parquet"),
             "ego_zip": os.path.join(args.root, "labels", "egomotion",
                                     "egomotion_all.zip")},
            out, size=256, n_stack=3, frame=bad_frame,
            projection_mode="cylindrical", codec="png")
    a = load_payload(out)
    b = load_payload(os.path.join(args.ref, f"{cid}.v2ep.pt"))
    i0 = pick_frames(min(int(a["jpeg_len"].shape[0]),
                         int(b["jpeg_len"].shape[0])), args.frames)[0]
    d = down(frames_of(a, [i0])[0])
    v = compare(d, frames_of(b, [i0])[0][:, 48:208, :], "M1_unscaled_f_ref",
                TOL["G2"])
    broke = not (v["G1_geometry_pass"] and v["G2_photometric_pass"])
    v.update(MUTATION_DETECTED=broke, mutant_f_ref=bad_frame.f_ref,
             mutant_delivered_hfov_deg=round(got_hfov, 4),
             correct_f_ref=488.9239851782515, correct_hfov_deg=120.0,
             clip_sha12=s12, frame=i0)
    R.setdefault("G3_mutations", {})["M1_unscaled_f_ref"] = v
    print(f"  mutant delivers HFOV {got_hfov:.4f} deg, not 120", flush=True)
    print(f"  [{'DETECTED' if broke else 'MISSED(BUG)'}] M1: peak={v['xcorr_peak']} "
          f"median={v['median']} mae={v['mae']} psnr={v['psnr_db']} dB", flush=True)
    return [] if broke else ["M1_unscaled_f_ref"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wide", required=True, help="256x1024 cache dir")
    ap.add_argument("--ref", required=True, help="256x640 cache dir (same clips)")
    ap.add_argument("--ids", required=True)
    ap.add_argument("--root", required=True, help="staging root (for M1)")
    ap.add_argument("--tmp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clips", type=int, default=6)
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--g0-local", default="")
    ap.add_argument("--g0-thor", default="")
    a = ap.parse_args()
    torch.set_num_threads(4)
    R, F_ALL = {"prereg": "PREREG.md", "tolerances": TOL}, []
    if a.g0_local and a.g0_thor:
        F_ALL += g0(a, R)
    F_ALL += g1g2g3(a, R)
    F_ALL += g3_unscaled_fref(a, R)
    R["failures"] = F_ALL
    json.dump(R, open(a.out, "w"), indent=1)
    print(f"\n{'ALL PASS' if not F_ALL else 'FAILURES: ' + ','.join(map(str, F_ALL))}"
          f" -> {a.out}")
    sys.exit(1 if F_ALL else 0)
