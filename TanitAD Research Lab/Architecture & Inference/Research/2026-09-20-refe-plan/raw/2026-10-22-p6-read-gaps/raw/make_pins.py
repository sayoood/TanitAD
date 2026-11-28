import hashlib, json, sys
from pathlib import Path
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
EVAL = Path(f"C:/Users/Admin/par18/p6gaps/code/{PK}/eval")
sys.path.insert(0, str(EVAL))
import harness_pins as HP
SP = Path("C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/a085b71c-4fb3-409b-ae6b-65f605ed575e/scratchpad")
root = EVAL / "harness"
files = {}
for rel in HP._py_files(root):
    rec = {"sha256_lf": HP.sha256_lf(root / rel), "bytes": (root / rel).stat().st_size}
    sp = SP / rel
    if rel == "w3e/code/run_v1_fix1.py":
        rec["origin"] = "repo_w3_original"
        rec["origin_path"] = "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-v1-navtest/code/run_v1_fix1.py (mirror tip 6723462; NT-FIX1, absent from the scratchpad)"
    elif sp.exists():
        raw = hashlib.sha256(sp.read_bytes()).hexdigest()
        rec["scratchpad_sha256"] = raw
        if raw == rec["sha256_lf"]:
            rec["origin"] = "scratchpad_byte_identical"
        else:
            rec["origin"] = "scratchpad_plus_one_line"
            rec["edit"] = "line 49 only: E1_WRAPPER was a hard-coded absolute path into the Temp scratchpad (.../scratchpad/w3e/e1/navsim_win.py); now relative to this file (../e1/navsim_win.py), whose bytes equal the scratchpad's"
    else:
        rec["origin"] = "NO_SCRATCHPAD_COUNTERPART"
    files[rel] = rec
out = {"what": "sha256 (LF-normalised) of every file of the banked scoring harness read by eval/stage2_sft.py (P6-READ-GAPS G3, 2026-10-22)",
       "scratchpad": str(SP),
       "provenance": "every scratchpad file equals the repo original (mirror tip 6723462) after the D:/ -> E:/ sed, LF-normalised, except navsim_v1_win.py line 49 (raw/harness_provenance.jsonl)",
       "files": files,
       "caller_variants": {
           "pre_NT_FIX1": "91ec3e6592a6292097fba0209aba97aaed18d31c267f212ed5508e533522c1de",
           "NT_FIX1": "94f7b024ab2a375a62cf7268e85b0d245ac81ff7b69f2f0c4a9295050ab962b7"},
       "caller_note": "eval/score_navtest_refe.py is what score_e.py loads. pre_NT_FIX1 = the caller that produced u400, P2 and P2-s1 (working trees D:/E: on 2026-10-22). NT_FIX1 = mirror tip 6723462 (commit 9e8d11f, 2026-10-19): loads run_v1_fix1.py, which byte-pins run_v1.py; the repointed copy needs RUN_V1_FIX1_ALLOW_DRIFT=1 (stage2_sft.py sets it)."}
(root / "HARNESS_PINS.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
for k, v in files.items(): print(v["origin"], k)
