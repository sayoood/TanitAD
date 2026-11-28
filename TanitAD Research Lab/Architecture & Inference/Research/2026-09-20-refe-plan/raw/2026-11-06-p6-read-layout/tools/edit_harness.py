# one-off, deterministic exact-string replacements for the P6 layout fix (path roots only)
import sys
PK = "E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/eval/harness/w3e/code/"
RES = '''def _navsim_root():
    # P6 layout fix 2026-11-06 (path roots only): $TANITAD_NAVSIM_ROOT if set, else the FIRST EXISTING of E:/ F:/ D:/ Archive/devbox-C/navsim; REFUSES if none exists
    r = os.environ.get("TANITAD_NAVSIM_ROOT")
    cands = [r.replace("\\\\", "/").rstrip("/")] if r else [f"{d}:/Archive/devbox-C/navsim" for d in "EFD"]
    for c in cands:
        if os.path.isdir(c):
            return c
    raise SystemExit("REFUSING: navsim archive root not found (tried: " + ", ".join(cands) + "); set TANITAD_NAVSIM_ROOT. No silent default.")


_NAVSIM = _navsim_root()
'''
def rep(path, old, new):
    s = open(path, "rb").read().decode("utf-8")
    assert "\r\n" not in s
    assert s.count(old) == 1, (path, s.count(old), old[:70])
    open(path, "wb").write(s.replace(old, new).encode("utf-8"))

p = PK + "run_v1.py"
rep(p, '''V11_TREE = "E:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896"
EXP = "E:/Archive/devbox-C/navsim/exp/w3_navtest_v1"
DATA = "E:/Archive/devbox-C/navsim/data/openscene"
''', RES + '''V11_TREE = f"{_NAVSIM}/navsim-3e8291bfa89ff247231e0227778840cd0a036896"
EXP = f"{_NAVSIM}/exp/w3_navtest_v1"
DATA = f"{_NAVSIM}/data/openscene"
''')
rep(p, '''    "NUPLAN_MAPS_ROOT": "E:/Archive/devbox-C/navsim/data/maps",
    "NAVSIM_EXP_ROOT": EXP,''', '''    "NUPLAN_MAPS_ROOT": f"{_NAVSIM}/data/maps",
    "NAVSIM_EXP_ROOT": EXP,''')

p = PK + "navsim_v1_win.py"
rep(p, '''V11_TREE = "E:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896"
V11_SHA''', RES + '''V11_TREE = f"{_NAVSIM}/navsim-3e8291bfa89ff247231e0227778840cd0a036896"
V11_SHA''')
rep(p, '''NUPLAN_SOURCE = "E:/Archive/devbox-C/navsim/nuplan-devkit"''', '''NUPLAN_SOURCE = f"{_NAVSIM}/nuplan-devkit"''')
rep(p, '''    "NUPLAN_MAPS_ROOT": "E:/Archive/devbox-C/navsim/data/maps",
    "NAVSIM_EXP_ROOT": "E:/Archive/devbox-C/navsim/exp/w3_navtest_v1",
    "NAVSIM_DEVKIT_ROOT": V11_TREE,
    "OPENSCENE_DATA_ROOT": "E:/Archive/devbox-C/navsim/data/openscene",''', '''    "NUPLAN_MAPS_ROOT": f"{_NAVSIM}/data/maps",
    "NAVSIM_EXP_ROOT": f"{_NAVSIM}/exp/w3_navtest_v1",
    "NAVSIM_DEVKIT_ROOT": V11_TREE,
    "OPENSCENE_DATA_ROOT": f"{_NAVSIM}/data/openscene",''')

p = PK + "write_cache_metadata.py"
rep(p, '''EXP = "E:/Archive/devbox-C/navsim/exp/w3_navtest_v1"
''', RES + '''EXP = f"{_NAVSIM}/exp/w3_navtest_v1"
''')
print("edited ok")
