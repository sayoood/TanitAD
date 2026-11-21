"""v6r: the native CAM_FW DRIVABLE decision replaced by SAM3's decision on the RECTIFIED virtual view of the same camera.

Why (MEASURED 2026-09-13): at night the native CAM_FW road mask puts 2.81 % of LiDAR tall-obstacle points on drivable at
8-15 m, v2's virtual 63.7-deg pinhole view of the SAME camera 1.43 % (a2_by_camera.py), and the v6 night clip vote FAILED
its bar (MAP_A2 0.1429). The pre-registered 2x zoom tile of the fisheye image changed NOTHING (night A2 0.1430, day
identical) -- magnification is refuted; what the virtual view also has is RECTIFICATION (straight lines, no fisheye
distortion). The virtual CAM_F0 of the v2 sequence shares CAM_FW's centre exactly (translation identical, 0.79 deg
rotation), so every native pixel maps to a virtual pixel through its ray alone, with no depth.

Rule, per token: for every native CAM_FW raster pixel whose ray lands inside the virtual 1920x1080 pinhole view, read the
banked v3raw CAM_F0 class there (same road prompts); native 1 (plain drivable) -> 0 where the rectified view says not
drivable, native 0 / 7 -> 1 where it says plain drivable. Paint (2, 3, 4, 6), edges (5) and every other camera unchanged;
points re-lifted exactly as the extractor did.

PRE-REGISTERED 2026-09-13 before its first run: v6r WORKS if (a) on the night clip the v6r clip vote MAP_A2 <= its own
SAM3_fused_1s + 0.05, (b) on the day clip the clip-vote MAP_A2 rises by no more than 0.01 over v6 (0.0225), and (c)
MAP_B >= 0.95 on both clips. Anything else is a FAIL, reported as such.
RESULT v6r (MEASURED): night clip vote A2 0.1429 -> 0.1331, B 0.9714 -- FAIL on (a) (bar 0.0624 + 0.05); 191,392 px to not
drivable, 28,301 to drivable over 96 frames. a2_fov_split.py then located the night bleed: native pixels INSIDE the v2
views bleed 20-40 % more than v2 did (FW 8-15 m 1.73 % vs 1.43 %, RL 0.93 % vs 0.66 %), and the PERIPHERY plus the tele
cameras add ~37 % of the night total (FW periphery 8-15 m 1.10 %, FT 8-15 m 0.92 %), while the day periphery is clean.

v6n (RECT_ALL=1 DROP_TELE=1), PRE-REGISTERED 2026-09-13 before its first run: the same rule for EVERY camera with a used v2
view of the same centre (FW<-F0, CL<-L0+L1, CR<-R0+R1, RL<-L2, RR<-R2), and the tele cameras' plain drivable pixels (FT, RT)
set to 0. SAME bars as v6r: night clip-vote A2 <= its fused +-1 s + 0.05, day A2 rise <= 0.01 over v6, MAP_B >= 0.95 on both.
Usage: sam3map_rectified_road.py <c8> <v6 raw dir> <v3raw dir> <dst raw dir>   (SAM3MAP_ROOT native root; V2_ROOT virtual)
"""
import json, os, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/home/nvidia/sam3map"); sys.path.insert(0, "/home/nvidia/sam3paint"); sys.path.insert(0, str(HERE))
import numpy as np
import sam3_paint as P
import camera_model as CM
import ground_surface as GS
from sam3map_consensus import relift

ROOT = Path(os.environ.get("SAM3MAP_ROOT", "/home/nvidia/sam3map/native7"))
V2R = Path(os.environ.get("V2_ROOT", "/home/nvidia/qwendrive/v2"))
DRIVE = (1, 2, 3, 4, 6)
RECT_ALL = os.environ.get("RECT_ALL") == "1"
DROP_TELE = os.environ.get("DROP_TELE") == "1"
USED_V2 = {"CAM_FW": ("CAM_F0",), "CAM_CL": ("CAM_L0", "CAM_L1"), "CAM_CR": ("CAM_R0", "CAM_R1"), "CAM_RL": ("CAM_L2",), "CAM_RR": ("CAM_R2",)}
VIEWS = os.environ.get("SAM3MAP_VIEWS", "CAM_FW,CAM_CL,CAM_CR,CAM_RL,CAM_RR,CAM_RT,CAM_FT").split(",")


def main():
    c8, src, v3, dst = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]); dst.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ii, jj = np.mgrid[0:540, 0:960]
    u_n = (jj * 2 + 1.0).ravel(); v_n = (ii * 2 + 1.0).ravel()
    tot = {"frames": 0, "to_not_drivable": 0, "to_drivable": 0, "fw_pixels_in_virtual_view": 0}
    for f in sorted(src.glob("[0-9][0-9][0-9].npz")):
        d = dict(np.load(f, allow_pickle=True)); tok = str(d["tok"])
        z3 = np.load(v3 / f.name, allow_pickle=True)
        assert str(z3["tok"]) == tok, (f.name, tok, str(z3["tok"]))
        fd = ROOT / f"seq_{c8}" / tok; vd = V2R / f"seq_{c8}" / tok
        cn = np.load(fd / "calib.npz"); fn = json.loads((fd / "frame.json").read_text())
        cv = np.load(vd / "calib.npz"); fv = json.loads((vd / "frame.json").read_text())
        n_to0 = n_to1 = 0
        for ncam, vnames in (USED_V2.items() if RECT_ALL else (("CAM_FW", ("CAM_F0",)),)):
            if f"cls_{ncam}" not in d:
                continue
            Cn = CM.Camera.from_calib(cn, fn["cam_order"].index(ncam))
            ray = Cn.rays_rig(u_n, v_n)
            good = np.isfinite(ray).all(axis=1)
            Pw = Cn.t + np.where(good[:, None], ray, 0.0) * 10.0
            c_v = np.full(len(u_n), 255, np.uint8); ok_any = np.zeros(len(u_n), bool)
            for vname in vnames:
                Cv = CM.Camera.from_calib(cv, fv["cam_order"].index(vname))
                assert np.allclose(Cn.t, Cv.t, atol=1e-6), "virtual and native centres differ: the depth-free mapping does not hold"
                uv, vv, ok = Cv.project_rig(Pw)
                ok &= good & ~ok_any                                             # first view wins where two virtual views overlap
                qi = np.clip((vv / 2).astype(np.int64), 0, 539); qj = np.clip((uv / 2).astype(np.int64), 0, 959)
                c_v[ok] = z3[f"cls_{vname}"][qi[ok], qj[ok]]; ok_any |= ok
            nat = d[f"cls_{ncam}"].ravel().copy()
            to0 = ok_any & (nat == 1) & ~np.isin(c_v, DRIVE)
            to1 = ok_any & np.isin(nat, (0, 7)) & (c_v == 1)
            new = nat.copy(); new[to0] = 0; new[to1] = 1
            d[f"cls_{ncam}"] = new.reshape(540, 960)
            n_to0 += int(to0.sum()); n_to1 += int(to1.sum()); tot["fw_pixels_in_virtual_view"] += int(ok_any.sum())
        if DROP_TELE:
            for tcam in ("CAM_FT", "CAM_RT"):
                if f"cls_{tcam}" in d:
                    t_ = d[f"cls_{tcam}"].copy(); n_to0 += int((t_ == 1).sum()); t_[t_ == 1] = 0; d[f"cls_{tcam}"] = t_
        to0 = np.zeros(1, bool); to1 = np.zeros(1, bool)
        tot["frames"] += 1; tot["to_not_drivable"] += n_to0; tot["to_drivable"] += n_to1
        grid, fb = P.ground_grid(np.load(fd / "lidar.npy").astype(np.float64)); sg = GS.smooth_grid(grid, fb)
        T = d["T_world_rig"]
        accp, accr = {k: [] for k in range(1, 8)}, {k: [] for k in range(1, 8)}
        for cam in [v for v in VIEWS if f"cls_{v}" in d]:
            C = CM.Camera.from_calib(cn, fn["cam_order"].index(cam))
            p_, r_ = relift(d[f"cls_{cam}"], C, sg, T)
            for k in p_:
                accp[k].append(p_[k]); accr[k].append(r_[k])
        for k in range(1, 8):
            d[f"pts_{k}"] = np.concatenate(accp[k]) if accp[k] else np.zeros((0, 2), np.float32)
            d[f"rng_{k}"] = np.concatenate(accr[k]) if accr[k] else np.zeros((0,), np.float16)
        st = json.loads(str(d["stats"])); st["_rectified_road"] = {"to_not_drivable": n_to0, "to_drivable": n_to1, "rect_all": RECT_ALL, "drop_tele": DROP_TELE}; d["stats"] = json.dumps(st)
        d["ver"] = "v6n" if (RECT_ALL or DROP_TELE) else "v6r"
        np.savez_compressed(dst / f.name, **d)
    (dst.parent / f"rectified_road_{dst.name}.json").write_text(json.dumps({**tot, "seconds": round(time.time() - t0, 1)}, indent=1), encoding="utf-8")
    print(f"totals {tot}  {time.time() - t0:.0f}s"); print("ZZRECTROAD-DONEZZ")


if __name__ == "__main__":
    main()
