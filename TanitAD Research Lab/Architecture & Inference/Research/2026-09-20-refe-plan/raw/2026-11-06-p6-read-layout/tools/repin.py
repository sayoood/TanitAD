"""re-pin after the P6 layout fix, with the repo's own tools: harness_pins.build() computes every sha256_lf / bytes; stage3_full's STAGE2_SFT_MD5 is the md5 of stage2_sft.py on disk (md5_file of stage3_full itself).
`harness_pins.py --build` drops the top-level keys what/scratchpad/provenance/caller_note, so the new file is the OLD dict with only the per-file hashes updated (and a layout_fix note per changed file)."""
import hashlib
import json
import sys
from pathlib import Path

PK = Path("E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan")
sys.path.insert(0, str(PK / "eval"))
import harness_pins as HP  # noqa: E402

pins_p = PK / "eval/harness/HARNESS_PINS.json"
raw = pins_p.read_bytes()
old = json.loads(raw.decode("utf-8"))
assert (json.dumps(old, indent=1) + "\n").replace("\n", "\r\n").encode("utf-8") == raw, "formatting is not reproducible: refusing"
built = HP.build(HP.HARNESS)
assert set(built["files"]) == set(old["files"]), "file set changed"
new = json.loads(json.dumps(old))
moved = {}
for rel, rec in built["files"].items():
    if rec["sha256_lf"] != old["files"][rel]["sha256_lf"]:
        moved[rel] = {"old": old["files"][rel]["sha256_lf"], "new": rec["sha256_lf"]}
        new["files"][rel]["sha256_lf"] = rec["sha256_lf"]
        new["files"][rel]["bytes"] = rec["bytes"]
        new["files"][rel]["layout_fix_2026_11_06"] = {"was_sha256_lf": old["files"][rel]["sha256_lf"], "was_bytes": old["files"][rel]["bytes"],
                                                      "change": "path roots only: the navsim archive root is resolved ($TANITAD_NAVSIM_ROOT, else first existing of E:/ F:/ D:/ Archive/devbox-C/navsim; refuses if none)"}
new["layout_fix_2026_11_06"] = ("P6 layout fix 2026-11-06 (navsim archive moved E:->F:, D: absent): the files named under files[*].layout_fix_2026_11_06 changed in their PATH ROOTS ONLY; "
                                "the pins above were recomputed by harness_pins.build(); origin/scratchpad_sha256 describe the pre-fix bytes")
out = (json.dumps(new, indent=1) + "\n").replace("\n", "\r\n").encode("utf-8")
pins_p.write_bytes(out)
r = HP.verify(HP.HARNESS)
print(json.dumps({"moved": moved, "verify_ok": r["ok"], "n_files": r["n_files"], "caller": r["caller"], "pins_file_sha256": r["pins_file_sha256"],
                  "old_pins_file_sha256": hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()}, indent=1))

# stage3_full.STAGE2_SFT_MD5 = md5 of stage2_sft.py as it is now
s2 = PK / "eval/stage2_sft.py"
new_md5 = hashlib.md5(s2.read_bytes()).hexdigest()
s3 = PK / "eval/stage3_full.py"
t = s3.read_bytes().decode("utf-8")
old_md5 = "3a30ab4c5a122664c534f378144cb4a5"
assert t.count(old_md5) == 1, t.count(old_md5)
t = t.replace(f'STAGE2_SFT_MD5 = "{old_md5}"', f'STAGE2_SFT_MD5 = "{new_md5}"', 1)
t = t.replace("# stage2_sft.py as of P6-READ-GAPS (2026-10-22:", f"# stage2_sft.py as of the P6 layout fix (2026-11-06: path roots only, EXPORT via the root resolver; was {old_md5}) <- P6-READ-GAPS (2026-10-22:", 1)
assert t.count(old_md5) == 1 and new_md5 in t
s3.write_bytes(t.encode("utf-8"))
print("STAGE2_SFT_MD5", old_md5, "->", new_md5)
