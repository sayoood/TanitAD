from pathlib import Path
B = Path(__file__).resolve().parent
s = (B / "sam3map_refine_v3.py").read_text(encoding="utf-8")


def sub(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:90])
    s = s.replace(old, new)


sub('"""v3 refinement of the SAM3-only map,', '"""v5 refinement (camera_model + smooth ground; same rules and the same bit-exact control as v3).\n\nv3 refinement of the SAM3-only map,')
sub("import sam3_paint as P\n", "import sam3_paint as P\nimport camera_model as CM\nimport ground_surface as GS\n")
sub('''def relift(cls_small, K, R, t, grid, fb, T):
    full = np.repeat(np.repeat(cls_small, 2, axis=0), 2, axis=1)
    pts, rng = {}, {}
    for k in range(1, 7):
        m = np.isin(full, DRIVE) if k == 1 else (full == k)
        if m.any():
            xy = P.lift(m, K, R, t, grid, fb, stride=2)
            if len(xy):
                pts[k] = (np.c_[xy, np.zeros(len(xy)), np.ones(len(xy))] @ T.T)[:, :2].astype(np.float32)
                rng[k] = np.hypot(xy[:, 0] - t[0], xy[:, 1] - t[1]).astype(np.float16)
    return pts, rng''', '''def relift(cls_small, cam, sgrid, T):
    full = np.repeat(np.repeat(cls_small, 2, axis=0), 2, axis=1)
    pts, rng = {}, {}
    for k in range(1, 7):
        m = np.isin(full, DRIVE) if k == 1 else (full == k)
        if m.any():
            xy, r = cam.lift(m, sgrid, stride=2)
            if len(xy):
                pts[k] = (np.c_[xy, np.zeros(len(xy)), np.ones(len(xy))] @ T.T)[:, :2].astype(np.float32)
                rng[k] = r.astype(np.float16)
    return pts, rng''')
sub('''        grid, fb = P.ground_grid(np.load(lp).astype(np.float64)) if lp.exists() else (np.full(2500, np.nan), 0.0)
        cams = {cam: (c["cam_intrinsic"][fr["cam_order"].index(cam)], c["sensor2lidar_rotation"][fr["cam_order"].index(cam)],
                      c["sensor2lidar_translation"][fr["cam_order"].index(cam)]) for cam in VIEWS}''',
    '''        grid, fb = P.ground_grid(np.load(lp).astype(np.float64)) if lp.exists() else (np.full(2500, np.nan), 0.0)
        sgrid = GS.smooth_grid(grid, fb)
        cams = {cam: CM.Camera.from_calib(c, fr["cam_order"].index(cam)) for cam in VIEWS}''')
sub('            p_, r_ = relift(d[f"cls_{cam}"], *cams[cam], grid, fb, T)', '            p_, r_ = relift(d[f"cls_{cam}"], cams[cam], sgrid, T)')
sub('            p_, r_ = relift(cl, *cams[cam], grid, fb, T)', '            p_, r_ = relift(cl, cams[cam], sgrid, T)')
assert "P.lift(" not in s
(B / "sam3map_refine_v5.py").write_text(s, encoding="utf-8", newline="\n")
print("ok")
