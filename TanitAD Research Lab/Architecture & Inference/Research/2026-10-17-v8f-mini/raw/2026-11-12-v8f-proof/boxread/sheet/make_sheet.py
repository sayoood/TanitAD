"""make_sheet.py -- a PRIVATE sheet of evaluation frames with the detector's boxes (cards ADD127 to ADD129). The sheet holds camera pixels: it stays on the pod
and on the dev box (shown to the PI as a local file), it is never landed and never uploaded. This tool holds no pixels and may be landed.

For N_SAME evaluation clips on which the locator's pick is the label's pick and N_OTHER clips on which it is not (clips in list order, one frame per clip: the
middle one of the frames that qualify), the wide window of 32 degrees is rendered and drawn with: every vehicle detection (thin grey), the label's pick (green),
the locator's pick where it differs (orange), the locator's direction (cross). Under each tile: short clip id, frame, label gap, arm D's gap (first fit), box gap,
gated mean (card ADD129). CPU only. It reads labels (a bound view for the eye), so it is no arm and names nothing.
Usage: make_sheet.py <out.png> plus the arguments of read_boxfuse.py (controls ...), and --root <stage root of the wide clips> first."""
from __future__ import annotations

import math
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N_SAME, N_OTHER = 8, 4
TILE = 512
COLS = 4
BAND = 54


def main(argv):
    out_png, root = argv[0], argv[1]
    for p in (os.environ.get("BOXREAD_KIT_F") or "/workspace/v8proof/boxread/kit_f", os.environ.get("BOXREAD_KIT_DET") or "/workspace/v8proof/boxread/kit/det"):
        if p not in sys.path:
            sys.path.insert(0, p)
    import read_boxfuse as RF
    import box_detect as BD
    from PIL import Image, ImageDraw
    RB, BS = RF.RB, RF.BS
    ap = RB.build_parser()
    ap.add_argument("--add128-controls", default="")
    ap.add_argument("--add128-result", default="")
    a = ap.parse_args(argv[2:])
    J = BS.load_npz(a.eval_join)
    L = {k: BS.lay_dump(J, BS.load_npz(p)) for k, p in RB.dump_paths(a).items()}
    store = RB.DetStore(a.det_dir)
    tclips = BS.ids_of(BS.parse_kv(a.list_file)["train_list"])
    Gw = RB.build_map(RB.train_reads(a.box_dir, store, tclips, "wide"))
    Pz = BS.lay_pred(J, RB.load_locator(RB.locator_path(a)))
    n, off = np.asarray(J["n"], np.int64), np.asarray(J["off"], np.int64)
    t3 = BS.t3_rows(L)
    Lw = BD.Layout(root)
    picks = {"same": [], "other": []}
    for i, c in enumerate(J["clip"]):
        c = str(c)
        if len(picks["same"]) >= N_SAME and len(picks["other"]) >= N_OTHER:
            break
        d = store.get("wide", c)
        bp = os.path.join(a.box_dir, c + ".npz")
        if d is None or not os.path.isfile(bp):
            continue
        z = np.load(bp)
        ni = int(n[i])
        m = min(len(z["has_lead"]), ni)
        use = np.zeros(ni, bool)
        use[:m] = z["has_lead"].astype(bool)[:m] & np.isfinite(z["ctr_az"][:m]) & np.isfinite(z["ctr_el"][:m])
        laz, lel = np.full(ni, np.nan), np.full(ni, np.nan)
        laz[:m] = np.where(use[:m], np.asarray(z["ctr_az"], np.float64)[:m], np.nan)
        lel[:m] = np.where(use[:m], np.asarray(z["ctr_el"], np.float64)[:m], np.nan)
        sl = slice(int(off[i]), int(off[i] + ni))
        paz, pel = np.asarray(Pz["az"][sl], np.float64), np.asarray(Pz["el"][sl], np.float64)
        pk_lab = RB.pick_clip(d, laz, lel, ni)
        pk_loc = RB.pick_clip(d, paz, pel, ni)
        both = t3[sl] & (pk_lab >= 0) & (pk_loc >= 0)
        for kind, mask in (("same", both & (pk_lab == pk_loc)), ("other", both & (pk_lab != pk_loc))):
            want = N_SAME if kind == "same" else N_OTHER
            fr = np.flatnonzero(mask)
            if len(picks[kind]) < want and fr.size and not any(p["clip"] == c for p in picks["same"] + picks["other"]):
                f = int(fr[len(fr) // 2])
                w, h = RB.sizes_of(d, np.array([pk_loc[f]]))
                gbox = float(Gw.predict(w, h)[0])
                gd = float(L["D_0"]["gap"][sl][f])
                fg = math.sqrt(gbox * gd) if (gbox > 0 and gd > 0 and abs(math.log(gbox) - math.log(gd)) <= math.log(RF.GATE_FACTOR)) else gd
                picks[kind].append({"clip": c, "f": f, "d": d, "lab": int(pk_lab[f]), "loc": int(pk_loc[f]), "paz": float(paz[f]), "pel": float(pel[f]),
                                    "lab_gap": float(L["D_0"]["lab_gap"][sl][f]), "gd": gd, "gbox": gbox, "fg": fg, "kind": kind})
                break
    items = picks["same"] + picks["other"]
    rows = int(math.ceil(len(items) / COLS))
    sheet = Image.new("RGB", (COLS * TILE, rows * (TILE + BAND)), (20, 20, 20))
    for k, it in enumerate(items):
        d = it["d"]
        win = tuple(float(x) for x in d["win"])
        wins, ok, _info = BD.make_windows("wide", it["clip"], Lw, None, win, box_dir=a.box_dir, only_lead=True)
        img = Image.fromarray(np.transpose(wins[it["f"]], (1, 2, 0))) if ok[it["f"]] else Image.new("RGB", (int(win[1]), int(win[1])), (60, 0, 0))
        del wins
        dr = ImageDraw.Draw(img)
        boxes = np.asarray(d["det_box"], np.float64).reshape(-1, 4)
        for j in RB.vehicle_dets(d, int(d["n_frames"])):
            if int(d["det_frame"][j]) == it["f"]:
                dr.rectangle([float(x) for x in boxes[j]], outline=(170, 170, 170), width=1)
        if it["loc"] != it["lab"]:
            dr.rectangle([float(x) for x in boxes[it["loc"]]], outline=(255, 140, 0), width=3)
        dr.rectangle([float(x) for x in boxes[it["lab"]]], outline=(0, 220, 0), width=3)
        cc, rr, _ = RB.BG.dir_to_pix(it["paz"], it["pel"], d["win"])
        cc, rr = float(np.asarray(cc).reshape(-1)[0]), float(np.asarray(rr).reshape(-1)[0])
        dr.line([cc - 12, rr, cc + 12, rr], fill=(255, 255, 0), width=2)
        dr.line([cc, rr - 12, cc, rr + 12], fill=(255, 255, 0), width=2)
        img = img.resize((TILE, TILE))
        x0, y0 = (k % COLS) * TILE, (k // COLS) * (TILE + BAND)
        sheet.paste(img, (x0, y0))
        ds = ImageDraw.Draw(sheet)
        ds.text((x0 + 6, y0 + TILE + 4), "%s f%d  %s" % (it["clip"][:8], it["f"], "pick = label pick" if it["kind"] == "same" else "pick on another box (orange)"), fill=(235, 235, 235))
        ds.text((x0 + 6, y0 + TILE + 20), "label %.1f m | token head %.1f | box %.1f | joined %.1f" % (it["lab_gap"], it["gd"], it["gbox"], it["fg"]), fill=(235, 235, 235))
        ds.text((x0 + 6, y0 + TILE + 36), "green: label's pick  grey: vehicle boxes  cross: locator", fill=(150, 150, 150))
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    if os.path.exists(out_png):
        print("REFUSED: %s exists" % out_png)
        return 2
    sheet.save(out_png)
    open(os.path.join(os.path.dirname(os.path.abspath(out_png)), "PRIVATE_DO_NOT_LAND.txt"), "a", encoding="utf-8").write("camera pixels: private, never landed, never uploaded\n")
    print("ZZ SHEET DONE %d tiles (%d same, %d other) %d bytes" % (len(items), len(picks["same"]), len(picks["other"]), os.path.getsize(out_png)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
