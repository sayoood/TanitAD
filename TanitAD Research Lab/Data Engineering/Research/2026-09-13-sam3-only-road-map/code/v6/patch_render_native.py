from pathlib import Path
B = Path(__file__).resolve().parent
s = (B / "sam3map_render_v4.py").read_text(encoding="utf-8")


def sub(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:90])
    s = s.replace(old, new)


sub('COL = {1: (70, 120, 200), 2: (255, 214, 0), 3: (0, 220, 210), 4: (255, 60, 220), 5: (235, 40, 40), 6: (255, 140, 0)}',
    'COL = {1: (70, 120, 200), 2: (255, 214, 0), 3: (0, 220, 210), 4: (255, 60, 220), 5: (235, 40, 40), 6: (255, 140, 0), 7: (120, 150, 105)}')
sub('NAME = {1: "drivable road", 2: "lane / road line", 3: "crosswalk", 4: "arrow / text", 5: "non-drivable edge", 6: "hatched area"}',
    'NAME = {1: "drivable road", 2: "lane / road line", 3: "crosswalk", 4: "arrow / text", 5: "non-drivable edge", 6: "hatched area", 7: "sidewalk / verge"}')
sub('ALPHA = {1: 0.30, 2: 0.9, 3: 0.75, 4: 0.9, 5: 0.9, 6: 0.8}', 'ALPHA = {1: 0.30, 2: 0.9, 3: 0.75, 4: 0.9, 5: 0.9, 6: 0.8, 7: 0.35}')
sub('VIEWS = ["CAM_F0", "CAM_L0", "CAM_R0", "CAM_L1", "CAM_R1", "CAM_L2", "CAM_R2"]',
    'VIEWS = ["CAM_F0", "CAM_L0", "CAM_R0", "CAM_L1", "CAM_R1", "CAM_L2", "CAM_R2",\n'
    '         "CAM_FW", "CAM_CL", "CAM_CR", "CAM_RL", "CAM_RR", "CAM_RT", "CAM_FT"]                # virtual rig, then native cameras')
sub('    for k in (1, 2, 4, 6, 3, 5):', '    for k in (7, 1, 2, 4, 6, 3, 5):')
sub('    for k in (1, 2, 3, 4, 6, 5):', '    for k in (7, 1, 2, 3, 4, 6, 5):')
sub('    front_only = cams == ["CAM_F0"]', '    FRONT_CAM = "CAM_FW" if "CAM_FW" in cams else "CAM_F0"\n    front_only = cams == [FRONT_CAM]\n'
    '    SIDE = ("CAM_CL", "CAM_CR", "CAM_RL", "CAM_RR") if "CAM_CL" in cams else ("CAM_L0", "CAM_R0", "CAM_L2", "CAM_R2")\n'
    '    SIDE_NAME = {"CAM_L0": "front-left", "CAM_R0": "front-right", "CAM_L2": "rear-left", "CAM_R2": "rear-right",\n'
    '                 "CAM_CL": "cross-left", "CAM_CR": "cross-right", "CAM_RL": "rear-left", "CAM_RR": "rear-right"}')
sub('        f0img = np.asarray(Image.open(fd / "images" / "CAM_F0.jpg").convert("RGB"))\n        canvas.paste(overlay(f0img, byc["CAM_F0"]["ref"]).resize((1180, 664)), (10, 64))',
    '        f0img = np.asarray(Image.open(fd / "images" / f"{FRONT_CAM}.jpg").convert("RGB"))\n        canvas.paste(overlay(f0img, byc[FRONT_CAM]["ref"]).resize((1180, 664)), (10, 64))')
sub('            for q_, cam in enumerate(("CAM_L0", "CAM_R0", "CAM_L2", "CAM_R2")):', '            for q_, cam in enumerate(SIDE):')
sub('                d.text((16 + q_ * 297, 744), {"CAM_L0": "front-left", "CAM_R0": "front-right", "CAM_L2": "rear-left", "CAM_R2": "rear-right"}[cam],',
    '                d.text((16 + q_ * 297, 744), SIDE_NAME[cam],')
sub('        for k in (1, 2, 3, 6, 4, 5):', '        for k in (1, 2, 3, 6, 4, 5, 7):')
sub('        for q_, k in enumerate((1, 2, 3, 6, 4, 5)):\n            xx = 10 + q_ * 197', '        for q_, k in enumerate((1, 2, 3, 6, 4, 5, 7)):\n            xx = 10 + q_ * 168')
sub('        label = "v5 · native 120° camera · smooth ground" if ver == "v5" else f"{ver} extraction · v4 BEV"',
    '        label = {"v5": "v5 · native 120° camera · smooth ground", "v6": "v6 · all native cameras · smooth ground"}.get(ver, f"{ver} extraction · v4 BEV")')
(B / "sam3map_render_v4.py").write_text(s, encoding="utf-8", newline="\n")
print("ok")
