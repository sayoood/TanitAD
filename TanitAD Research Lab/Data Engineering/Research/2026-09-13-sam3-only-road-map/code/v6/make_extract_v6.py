from pathlib import Path
B = Path(__file__).resolve().parent
s = (B / "sam3map_extract_v5.py").read_text(encoding="utf-8")


def sub(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:90])
    s = s.replace(old, new)


sub('"""SAM3-ONLY road map, v5 =', '"""SAM3-ONLY road map, v6 = v5 on ALL NATIVE cameras (CAM_FW, CAM_CL, CAM_CR, CAM_RL, CAM_RR, CAM_RT, CAM_FT) plus\n'
    'class 7 SIDEWALK / VERGE (SAM3 sidewalk or grass that is not drivable -- the non-drivable ground a BEV head needs, Qwen-Drive\'s\n'
    '"walkway"). Forward-looking views for the hatched-prompt fallback: CAM_F0, CAM_FW, CAM_FT. Output <c8>_v6raw.\n\n'
    'SAM3-ONLY road map, v5 =')
sub('''    nondrive = np.zeros((H, W), bool); curb = np.zeros((H, W), bool)
    for p, s, m in inst["nondrive"]:
        nondrive |= m
        if p == "curb":
            curb |= m''', '''    nondrive = np.zeros((H, W), bool); curb = np.zeros((H, W), bool); walk = np.zeros((H, W), bool)
    for p, s, m in inst["nondrive"]:
        nondrive |= m
        if p == "curb":
            curb |= m
        if p in ("sidewalk", "grass"):
            walk |= m''')
sub('''    cls[drivable] = 1; cls[line] = 2; cls[sym] = 4; cls[hatch] = 6; cls[cross] = 3; cls[edge] = 5''',
    '''    cls[drivable] = 1; cls[line] = 2; cls[sym] = 4; cls[hatch] = 6; cls[cross] = 3; cls[walk & ~drivable] = 7; cls[edge] = 5''')
sub('''        lifted = {k: [] for k in range(1, 7)}; ranges = {k: [] for k in range(1, 7)}''',
    '''        lifted = {k: [] for k in range(1, 8)}; ranges = {k: [] for k in range(1, 8)}''')
sub('''            for k in range(1, 7):
                m = np.isin(cls, (1, 2, 3, 4, 6)) if k == 1 else (cls == k)''', '''            for k in range(1, 8):
                m = np.isin(cls, (1, 2, 3, 4, 6)) if k == 1 else (cls == k)''')
sub('''                            **{f"pts_{k}": (np.concatenate(lifted[k]) if lifted[k] else np.zeros((0, 2), np.float32)) for k in range(1, 7)},
                            **{f"rng_{k}": (np.concatenate(ranges[k]) if ranges[k] else np.zeros((0,), np.float16)) for k in range(1, 7)})''',
    '''                            **{f"pts_{k}": (np.concatenate(lifted[k]) if lifted[k] else np.zeros((0, 2), np.float32)) for k in range(1, 8)},
                            **{f"rng_{k}": (np.concatenate(ranges[k]) if ranges[k] else np.zeros((0,), np.float16)) for k in range(1, 8)})''')
sub('''        tot = {k: sum(len(a) for a in lifted[k]) for k in range(1, 7)}''', '''        tot = {k: sum(len(a) for a in lifted[k]) for k in range(1, 8)}''')
sub('            cls, st = classify(proc, img, (camm, sgrid), cam == "CAM_F0")', '            cls, st = classify(proc, img, (camm, sgrid), cam in ("CAM_F0", "CAM_FW", "CAM_FT"))')
sub('out = Path(f"/home/nvidia/sam3map/{c8}_v5raw")', 'out = Path(f"/home/nvidia/sam3map/{c8}_v6raw")')
sub('ver="v5",', 'ver="v6",')
# SAM3 evidence bitfield per camera (960x540): the per-prompt instance masks the class rules consumed, so later rules
# (clip consensus, stripe orientation, dashed-line gate on area paint) re-derive without re-running SAM3.
sub('''    return cls, st
''', '''    evid = np.zeros((H, W), np.uint8)      # 1 diag on-road, 2 crosswalk on-road (pre-rule), 4 line on-road, 8 dashed, 16 solid,
    for bit, masks in ((1, [m for p, s, m in ok["diag"]]), (2, [m for p, s, m in ok["crosswalk"]]),     # 32 symbol on-road (pre-rule),
                       (4, [m for p, s, m in ok["line"]]), (8, [m for p, s, m in gate if p == "dashed line"]),   # 64 curb, 128 sidewalk|grass
                       (16, [m for p, s, m in gate if p == "solid line"]), (32, [m for p, s, m in ok["symbol"]])):
        for m in masks:
            evid[m] |= bit
    evid[curb] |= 64; evid[walk] |= 128
    return cls, st, evid
''')
sub('            cls, st = classify(proc, img, (camm, sgrid), cam in ("CAM_F0", "CAM_FW", "CAM_FT"))',
    '            cls, st, evid = classify(proc, img, (camm, sgrid), cam in ("CAM_F0", "CAM_FW", "CAM_FT"))\n'
    '            small_e[cam] = cv2.resize(evid, (OUT_W, OUT_H), interpolation=cv2.INTER_NEAREST)')
sub('        small, stats_all = {}, {}', '        small, stats_all, small_e = {}, {}, {}')
sub('                            **{f"cls_{cam}": small[cam] for cam in VIEWS},',
    '                            **{f"cls_{cam}": small[cam] for cam in VIEWS}, **{f"evid_{cam}": small_e[cam] for cam in VIEWS},')
(B / "sam3map_extract_v6.py").write_text(s, encoding="utf-8", newline="\n")
print("ok")
