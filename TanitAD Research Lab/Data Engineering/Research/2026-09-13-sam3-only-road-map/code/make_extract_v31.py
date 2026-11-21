from pathlib import Path
B = Path(__file__).resolve().parent
s = (B / "sam3map_extract_v3.py").read_text(encoding="utf-8")


def sub(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:80])
    s = s.replace(old, new)


sub('"""SAM3-ONLY road map, v3: instance-level paint classification (crosswalk vs hatched area vs line vs arrow/text).',
    '"""SAM3-ONLY road map, v3.1: v3 with the crosswalk / hatched-area split decided by stripe GEOMETRY ON THE GROUND.\n\n'
    'v3 used SAM3\'s "diagonal stripes on road" prompt, an IMAGE cue: zebras seen obliquely by the rear cameras looked\n'
    'diagonal (night clip: 78 of 84 crosswalk -> hatched decisions from L2/R2, 0 from F0). v3.1 (stripe_geometry.decide2,\n'
    'replayed on the prompt bank): local stripe directions in 1 m ground cells vs the region\'s long axis -- zebra >= 60 deg,\n'
    'hatched 20-55 deg or a two-direction chevron; ambiguous -> front camera: the v3 prompt rule; other views: crosswalk.\n'
    'Standalone diagonal-stripe instances become hatched only on the front camera. Output <c8>_v31raw.\n\n'
    'v3 docstring follows.')
sub("import sam3_paint as P\n", "import sam3_paint as P\nsys.path.insert(0, \"/home/nvidia/sam3map\")\nimport stripe_geometry as SG\n")
sub("def classify(proc, img, lift_ctx):", "def classify(proc, img, lift_ctx, front):")
sub('"hatched_standalone": 0, "symbol_dropped_dashed_solid": 0, "symbol_dropped_footprint": 0}',
    '"hatched_standalone": 0, "symbol_dropped_dashed_solid": 0, "symbol_dropped_footprint": 0,\n'
    '          "zebra_geom": 0, "hatched_geom": 0, "fallback_prompt_front": 0, "fallback_crosswalk": 0}')
sub("""        stripes = np.isin(lab, members)
        if stripes.sum() and (stripes & diag).sum() >= 0.3 * stripes.sum():
            hatch |= m | stripes; st["crosswalk_to_hatched"] += 1
        else:
            cross |= m | stripes; st["accepted"][p] = st["accepted"].get(p, 0) + 1""",
    """        stripes = np.isin(lab, members)
        verdict = SG.decide2(stripes, m | stripes, *lift_ctx)[0] if len(members) else "too_few"
        if verdict == "zebra":
            is_hatch = False; st["zebra_geom"] += 1
        elif verdict == "hatched":
            is_hatch = True; st["hatched_geom"] += 1
        elif front:
            is_hatch = bool(stripes.sum()) and (stripes & diag).sum() >= 0.3 * stripes.sum(); st["fallback_prompt_front"] += 1
        else:
            is_hatch = False; st["fallback_crosswalk"] += 1
        if is_hatch:
            hatch |= m | stripes; st["crosswalk_to_hatched"] += 1
        else:
            cross |= m | stripes; st["accepted"][p] = st["accepted"].get(p, 0) + 1""")
sub("""        if s >= 0.45 and not (m & (cross | hatch)).any():""", """        if front and s >= 0.45 and not (m & (cross | hatch)).any():""")
sub('out = Path(f"/home/nvidia/sam3map/{c8}_v3raw")', 'out = Path(f"/home/nvidia/sam3map/{c8}_v31raw")')
sub("            cls, st = classify(proc, img, (K, R, t, grid, fb))", "            cls, st = classify(proc, img, (K, R, t, grid, fb), cam == \"CAM_F0\")")
sub('ver="v3",', 'ver="v3.1",')
(B / "sam3map_extract_v31.py").write_text(s, encoding="utf-8", newline="\n")
print("ok")
