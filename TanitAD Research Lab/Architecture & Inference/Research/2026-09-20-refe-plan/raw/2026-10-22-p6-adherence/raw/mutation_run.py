"""MUTATION RUNS (P6-ADHERENCE): break the instrument at SOURCE level in the OVERLAY copy only (never the deliverable), run the full suite, require RED, restore.
Each mutation = (name, old text, new text). The deliverable under code/ is never touched; the overlay is re-synced from it afterwards."""
import subprocess, sys, shutil, os
from pathlib import Path
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
P = Path("C:/Users/Admin/par18/p6adh")
OV = P / "work/overlay" / PK / "eval"
SRC = P / "code" / PK / "eval" / "command_adherence.py"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
MUT = [
 ("M1_class_mapping_rotated", "LAT_TO_CMD = np.array([STRAIGHT, LEFT, RIGHT], dtype=np.int64)", "LAT_TO_CMD = np.array([STRAIGHT, RIGHT, LEFT], dtype=np.int64)"),
 ("M2_class_mapping_lane_keep_to_LEFT", "LAT_TO_CMD = np.array([STRAIGHT, LEFT, RIGHT], dtype=np.int64)", "LAT_TO_CMD = np.array([LEFT, STRAIGHT, RIGHT], dtype=np.int64)"),
 ("M3_labeller_reimplemented_with_smuggled_threshold",
  "        lat, _lon = RT.factor_from_kinematics(dyaw, dv, v0, v1)\n        out[s:s + chunk] = LAT_TO_CMD[lat.numpy()]",
  "        lat = (dyaw > 0.15).long() * 1 + (dyaw < -0.15).long() * 2\n        out[s:s + chunk] = LAT_TO_CMD[lat.numpy()]"),
 ("M4_bootstrap_stream_seed_shifted", "rng = np.random.default_rng(seed)\n        W = np.empty((n, G), dtype=np.float64)", "rng = np.random.default_rng(seed + 1)\n        W = np.empty((n, G), dtype=np.float64)"),
 ("M5_dt_0p1_instead_of_0p5", "N_POSES, DT_S = 8, 0.5 ", "N_POSES, DT_S = 8, 0.1 "),
 ("M6_UNKNOWN_swapped_with_STRAIGHT_in_command_decode", "cmd[i] = int(np.argmax(v))", "cmd[i] = int({0: 0, 1: 3, 2: 2, 3: 1}[int(np.argmax(v))])"),
]
src = SRC.read_text(encoding="utf-8")
only = sys.argv[1:]
for name, old, new in MUT:
    if only and name not in only: continue
    assert src.count(old) == 1, (name, src.count(old))
    (OV / "command_adherence.py").write_text(src.replace(old, new), encoding="utf-8", newline="\n")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run([PY, "-I", str(OV / "test_command_adherence.py")], cwd=str(OV.parent), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = r.stdout
    (P / "raw" / f"mutation_{name}.log").write_text(out + "\n--stderr--\n" + r.stderr[-2000:], encoding="utf-8")
    lines = [l for l in out.splitlines() if l.startswith("FAIL") or l.startswith("ZZADH ")]
    print(f"## {name}: rc={r.returncode}")
    for l in lines: print("   ", l[:230])
shutil.copyfile(SRC, OV / "command_adherence.py")
print("overlay restored from the deliverable")
