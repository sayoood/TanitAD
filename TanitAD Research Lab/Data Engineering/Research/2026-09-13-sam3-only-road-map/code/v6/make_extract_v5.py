from pathlib import Path
B = Path(__file__).resolve().parent
s = (B / "sam3map_extract_v31.py").read_text(encoding="utf-8")


def sub(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:90])
    s = s.replace(old, new)


sub('"""SAM3-ONLY road map, v3.1:',
    '"""SAM3-ONLY road map, v5 = v3.1 paint rules on corrected GEOMETRY: one camera model (pinhole virtual views or the\n'
    'NATIVE f-theta cameras, camera_model.py) and a SMOOTH ground surface (ground_surface.py) for every projection.\n'
    'Measured on the PI\'s arrow frame: per-2 m-cell ground heights put seams through painted arrows and halved the\n'
    'frame-to-frame agreement; the 63.7 deg virtual view dropped half of the 120 deg front camera. Output <c8>_v5raw.\n\n'
    'SAM3-ONLY road map, v3.1:')
sub("import stripe_geometry as SG\n", "import stripe_geometry as SG\nimport camera_model as CM\nimport ground_surface as GS\n")
sub('''def footprint(m, K, R, t, grid, fb):
    xy = P.lift(m, K, R, t, grid, fb, stride=2)''', '''def footprint(m, cam, sgrid):
    xy, _ = cam.lift(m, sgrid, stride=2)''')
sub('        verdict = SG.decide2(stripes, m | stripes, *lift_ctx)[0] if len(members) else "too_few"',
    '        verdict = SG.decide_points(lift_ctx[0].lift(stripes, lift_ctx[1])[0], lift_ctx[0].lift(m | stripes, lift_ctx[1])[0])[0] if len(members) else "too_few"')
sub('''        grid, fb = P.ground_grid(np.load(lp).astype(np.float64)) if lp.exists() else (np.full(2500, np.nan), 0.0)''',
    '''        grid, fb = P.ground_grid(np.load(lp).astype(np.float64)) if lp.exists() else (np.full(2500, np.nan), 0.0)
        sgrid = GS.smooth_grid(grid, fb)''')
sub('''            K, R, t = c["cam_intrinsic"][i], c["sensor2lidar_rotation"][i], c["sensor2lidar_translation"][i]
            img = Image.open(fd / "images" / f"{cam}.jpg").convert("RGB")
            cls, st = classify(proc, img, (K, R, t, grid, fb), cam == "CAM_F0")''',
    '''            img = Image.open(fd / "images" / f"{cam}.jpg").convert("RGB")
            camm = CM.Camera.from_calib(c, i, img.width, img.height)
            cls, st = classify(proc, img, (camm, sgrid), cam == "CAM_F0")''')
sub('''                    xy = P.lift(m, K, R, t, grid, fb, stride=2)
                    if len(xy):
                        lifted[k].append((np.c_[xy, np.zeros(len(xy)), np.ones(len(xy))] @ T.T)[:, :2].astype(np.float32))
                        ranges[k].append(np.hypot(xy[:, 0] - t[0], xy[:, 1] - t[1]).astype(np.float16))   # from the camera''',
    '''                    xy, rng = camm.lift(m, sgrid, stride=2)
                    if len(xy):
                        lifted[k].append((np.c_[xy, np.zeros(len(xy)), np.ones(len(xy))] @ T.T)[:, :2].astype(np.float32))
                        ranges[k].append(rng.astype(np.float16))                                          # from the camera''')
sub('out = Path(f"/home/nvidia/sam3map/{c8}_v31raw")', 'out = Path(f"/home/nvidia/sam3map/{c8}_v5raw")')
sub('ver="v3.1",', 'ver="v5",')
assert "P.lift(" not in s, "a pinhole-only lift survived"
(B / "sam3map_extract_v5.py").write_text(s, encoding="utf-8", newline="\n")
print("ok")
