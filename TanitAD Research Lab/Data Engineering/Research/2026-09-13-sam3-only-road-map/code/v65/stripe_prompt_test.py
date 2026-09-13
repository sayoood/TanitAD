"""Can SAM3 be prompted for the crosswalk STRIPES only (not the gaps between them)? Measured per prompt, on real crosswalk
views of our clips (auto-selected: the camera views with the most accepted crosswalk instances, day and night).

Reference, independent of SAM3's stripe prompts: the crosswalk AREA = union of "crosswalk" / "zebra crossing" /
"pedestrian crossing" instances (>= 0.5), and the BRIGHT PAINT = 127 px white top-hat > max(10, 2.5 x MAD of road) --
inside the area, bright paint is the stripes and the rest is the gaps.
Per prompt and threshold: instances, stripe precision (mask pixels inside the area that are bright paint), stripe recall
(bright area paint covered), gap coverage (non-bright area pixels covered -- the thing the PI wants at ~0), and the share
of the mask OUTSIDE the crosswalk area (stripes invented elsewhere). The area prompt "crosswalk" is the baseline; the
top-hat intersection (area AND bright paint) is the non-SAM alternative.
Usage: stripe_prompt_test.py <out dir>"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, "/home/nvidia/sam3vendor"); sys.path.insert(0, "/home/nvidia/sam3paint"); sys.path.insert(0, "/home/nvidia/sam3map")
import numpy as np
import cv2
from PIL import Image, ImageDraw
import sam3map_extract_v6 as E

SM = Path("/home/nvidia/sam3map"); out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
AREA_PROMPTS = ("crosswalk", "zebra crossing", "pedestrian crossing")
PROMPTS = ["crosswalk", "zebra crossing stripe", "crosswalk stripe", "zebra stripes", "pedestrian crossing stripe", "white stripe on road",
           "white painted stripe", "white road marking stripe", "white rectangle painted on road", "painted white bar on asphalt"]
THRS = (0.3, 0.4, 0.5)
XW = ("crosswalk", "zebra crossing", "pedestrian crossing")

# ---- pick views: most accepted crosswalk instances per (clip, token, camera); day, night and front-only clips
cands = []
for raw, root in ((SM / "4fbd97b6a4b7_v6raw", SM / "native7" / "seq_4fbd97b6a4b7"), (SM / "73495082f98b_v6raw", SM / "native7" / "seq_73495082f98b"),
                  (SM / "1f1f05ca011d_v5raw", SM / "front_native" / "seq_1f1f05ca011d"), (SM / "d672fc17a315_v5raw", SM / "front_native" / "seq_d672fc17a315"),
                  (SM / "b975bf8ebf95_v5raw", SM / "front_native" / "seq_b975bf8ebf95"), (SM / "f63e215a546a_v5raw", SM / "front_native" / "seq_f63e215a546a")):
    if not raw.exists():
        continue
    for f in sorted(raw.glob("[0-9][0-9][0-9].npz"))[::4]:
        z = np.load(f, allow_pickle=True); st = json.loads(str(z["stats"]))
        for cam, s in st.items():
            if cam.startswith("_") or not isinstance(s, dict):
                continue
            n = sum(s.get("accepted", {}).get(p, 0) for p in XW)
            if n:
                cands.append((n, raw.name, root, str(z["tok"]), cam))
cands.sort(key=lambda x: -x[0])
picked, seen_clip = [], {}
for n, rawname, root, tok, cam in cands:                       # at most 2 views per clip, distinct tokens
    k = rawname
    if seen_clip.get(k, 0) >= 2 or any(p[3] == tok for p in picked):
        continue
    picked.append((n, rawname, root, tok, cam)); seen_clip[k] = seen_clip.get(k, 0) + 1
    if len(picked) >= 9:
        break
print("views:", [(p[1], p[3], p[4], p[0]) for p in picked], flush=True)

t0 = time.time()
proc, _ = E.S.build(conf=0.25)
TOPHAT = cv2.getStructuringElement(cv2.MORPH_RECT, (127, 127))
res = {p: {t: {"views": 0, "inst": 0, "prec_num": 0, "prec_den": 0, "rec_num": 0, "rec_den": 0, "gap_num": 0, "gap_den": 0, "out_num": 0, "out_den": 0} for t in THRS} for p in PROMPTS}
res["area_AND_bright_paint (no SAM stripe prompt)"] = {0: {"views": 0, "inst": 0, "prec_num": 0, "prec_den": 0, "rec_num": 0, "rec_den": 0, "gap_num": 0, "gap_den": 0, "out_num": 0, "out_den": 0}}
tiles = []
for n, rawname, root, tok, cam in picked:
    img = Image.open(root / tok / "images" / f"{cam}.jpg").convert("RGB")
    state = proc.set_image(img)
    H, W = img.height, img.width
    area = np.zeros((H, W), bool)
    for ap in AREA_PROMPTS:
        for s, m in E.instances(proc, state, ap, 0.5):
            area |= m
    road = np.zeros((H, W), bool)
    for s, m in E.instances(proc, state, "road", 0.5):
        road |= m
    if area.sum() < 500:
        continue
    Y = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY).astype(np.float32)
    th = cv2.morphologyEx(Y, cv2.MORPH_TOPHAT, TOPHAT)
    base = th[road & ~area] if (road & ~area).sum() > 1000 else th
    paint = th > max(10.0, 2.5 * float(np.median(np.abs(base - np.median(base)))))
    area_d = cv2.dilate(area.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    ap_ = area & paint; ag_ = area & ~paint
    views_masks = {}
    for prompt in PROMPTS:
        inst = E.instances(proc, state, prompt, min(THRS))
        for t in THRS:
            U = np.zeros((H, W), bool); k = 0
            for s, m in inst:
                if s >= t:
                    U |= m; k += 1
            r = res[prompt][t]
            r["views"] += 1; r["inst"] += k
            r["prec_num"] += int((U & area_d & paint).sum()); r["prec_den"] += int((U & area_d).sum())
            r["rec_num"] += int((U & ap_).sum()); r["rec_den"] += int(ap_.sum())
            r["gap_num"] += int((U & ag_).sum()); r["gap_den"] += int(ag_.sum())
            r["out_num"] += int((U & ~area_d).sum()); r["out_den"] += int(U.sum())
            if t == 0.4:
                views_masks[prompt] = (U, k)
    U = area & paint; r = res["area_AND_bright_paint (no SAM stripe prompt)"][0]
    r["views"] += 1; r["prec_num"] += int((U & paint).sum()); r["prec_den"] += int(U.sum()); r["rec_num"] += int(U.sum()); r["rec_den"] += int(ap_.sum())
    r["gap_num"] += 0; r["gap_den"] += int(ag_.sum()); r["out_num"] += 0; r["out_den"] += int(U.sum())
    views_masks["area AND bright paint"] = (U, 0)
    # contact tile: crop around the crossing area
    ys, xs = np.nonzero(area_d)
    y0, y1 = max(0, ys.min() - 40), min(H, ys.max() + 40); x0, x1 = max(0, xs.min() - 60), min(W, xs.max() + 60)
    crop = np.asarray(img)[y0:y1, x0:x1]
    row = []
    for name in ["crosswalk", "zebra crossing stripe", "crosswalk stripe", "zebra stripes", "white stripe on road", "area AND bright paint"]:
        U, k = views_masks[name]
        a = crop.astype(np.float32).copy(); mm = U[y0:y1, x0:x1]
        a[mm] = 0.35 * a[mm] + 0.65 * np.array((0, 230, 220))
        tile = Image.fromarray(a.astype(np.uint8)); tile.thumbnail((420, 240))
        dd = ImageDraw.Draw(tile); dd.rectangle([0, 0, tile.width, 16], fill=(0, 0, 0)); dd.text((3, 2), f"{name} ({k})", fill=(255, 255, 255))
        row.append(tile)
    tiles.append((f"{rawname} {tok} {cam}", row))
    print(f"  {rawname} {tok} {cam}: area {int(area.sum())} px, paint in area {int(ap_.sum())} px  {time.time() - t0:.0f}s", flush=True)

summary = {}
for p, byt in res.items():
    summary[p] = {}
    for t, r in byt.items():
        summary[p][str(t)] = {"views": r["views"], "instances_per_view": round(r["inst"] / max(r["views"], 1), 1),
                              "stripe_precision": round(r["prec_num"] / max(r["prec_den"], 1), 3), "stripe_recall": round(r["rec_num"] / max(r["rec_den"], 1), 3),
                              "gap_coverage": round(r["gap_num"] / max(r["gap_den"], 1), 3), "share_outside_crossing": round(r["out_num"] / max(r["out_den"], 1), 3),
                              "mask_px": r["prec_den"] + r["out_num"]}
(out / "stripe_prompts.json").write_text(json.dumps({"views": [(p[1], p[3], p[4]) for p in picked], "summary": summary}, indent=1), encoding="utf-8")
if tiles:
    TW, TH_ = 420, 240
    sheet = Image.new("RGB", (6 * TW, len(tiles) * (TH_ + 22)), (10, 12, 11)); ds = ImageDraw.Draw(sheet)
    for q, (title, row) in enumerate(tiles):
        y = q * (TH_ + 22); ds.text((4, y + 3), title, fill=(230, 230, 230))
        for c, tile in enumerate(row):
            sheet.paste(tile, (c * TW, y + 20))
    sheet.save(out / "stripe_prompts_sheet.png")
print(json.dumps(summary, indent=1)); print("ZZSTRIPES-DONEZZ")
