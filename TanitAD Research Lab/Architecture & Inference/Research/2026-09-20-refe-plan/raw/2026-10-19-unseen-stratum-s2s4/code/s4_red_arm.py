"""S4 RED arm: deliberately corrupted copies of ONE jpg must FAIL the verifier; the pristine control must PASS.

NO jpg was fetched in this package (S2 found 0 of 1,944 listed jpgs), so the stand-in is the first member of
a LOCAL container made by fetch_front_camera.py in an earlier session (read-only from D:, bytes copied to
the scratch dir on C:). Its size and CRC32 come from that container's own directory. This proves the
verifier's logic; it is NOT a verification of any fetched frame.

Everything goes to unseen_s2s4/red_scratch/ (never to frames/ and never to D:). Run with python -I.
"""
import importlib.util, json, os, subprocess, sys, zipfile, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
SCR = os.path.join(HERE, "..", "red_scratch")
STAND_IN_ZIP = "D:/Projects/TanitAD/data/nuplan-camera/scenarios4/2021.05.12.23.36.44_veh-35_01133_01535_CAM_B0.zip"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"

spec = importlib.util.spec_from_file_location("verify_s4", os.path.join(HERE, "verify_s4.py"))
vs = importlib.util.module_from_spec(spec); spec.loader.exec_module(vs)

with zipfile.ZipFile(STAND_IN_ZIP) as z:
    info = z.infolist()[0]
    data = z.read(info.filename)
    exp_size, exp_crc = info.file_size, info.CRC
assert zlib.crc32(data) == exp_crc and len(data) == exp_size
base = os.path.basename(info.filename)

def flip(b, i):
    a = bytearray(b); a[i] ^= 0x01; return bytes(a)

variants = {   # name -> (bytes, expect_fail)
    "ctrl_pristine": (data, False),
    "red_bitflip_mid": (flip(data, len(data) // 2), True),            # same size, CRC differs, usually still decodes
    "red_truncated_1KB": (data[:-1024], True),                         # size, EOI, decode
    "red_soi_zeroed": (b"\x00\x00" + data[2:], True),                  # SOI + decode
    "red_eoi_zeroed": (data[:-2] + b"\x00\x00", True),                 # EOI (+ CRC)
    "red_empty": (b"", True),
}
os.makedirs(SCR, exist_ok=True)
index = {"entries": []}
for name, (b, _) in variants.items():
    log, cam = f"RED_{name}", "CAM_B0"
    with zipfile.ZipFile(os.path.join(SCR, f"{log}_{cam}.zip"), "w", zipfile.ZIP_STORED) as zz:
        zz.writestr(base, b)          # the container's own CRC is of the CORRUPT bytes; the index CRC is the original
    index["entries"].append({"path": f"{log}/{cam}/{base}", "usz": exp_size, "crc32": exp_crc})
# one more: the bit-flip checked against an index with NO crc (tar-like) must FAIL CLOSED
log, cam = "RED_red_nocrc_bitflip", "CAM_B0"
with zipfile.ZipFile(os.path.join(SCR, f"{log}_{cam}.zip"), "w", zipfile.ZIP_STORED) as zz:
    zz.writestr(base, flip(data, len(data) // 2))
index["entries"].append({"path": f"{log}/{cam}/{base}", "usz": exp_size})
variants["red_nocrc_bitflip"] = (None, True)
json.dump(index, open(os.path.join(SCR, "red_index.json"), "w"), indent=1)

# 1) function level
fn = {}
for name, (b, expect_fail) in variants.items():
    if b is None:
        continue
    r = vs.verify_jpg(b, exp_size, exp_crc)
    fn[name] = {"ok": r["ok"], "size_ok": r["size_ok"], "crc_ok": r["crc_ok"], "soi_ok": r["soi_ok"],
                "eoi_ok": r["eoi_ok"], "decode_ok": r["decode_ok"], "expect_fail": expect_fail}
r = vs.verify_jpg(flip(data, len(data) // 2), exp_size, None)
fn["red_nocrc_bitflip"] = {"ok": r["ok"], "size_ok": r["size_ok"], "crc_ok": r["crc_ok"], "soi_ok": r["soi_ok"],
                           "eoi_ok": r["eoi_ok"], "decode_ok": r["decode_ok"], "expect_fail": True}
# 2) CLI level: the verifier process must exit non-zero on the scratch dir, and per-row results must match
out = os.path.join(SCR, "red_manifest.json")
p = subprocess.run([PY, "-I", os.path.join(HERE, "verify_s4.py"), "--index", os.path.join(SCR, "red_index.json"),
                    "--frames-dir", SCR, "--out", out, "--allow-subset"], capture_output=True, text=True)
man = json.load(open(out))
rows = {r["path"].split("/")[0][4:]: r["ok"] for r in man["rows"]}
cli_ok = all((not rows[n]) == variants[n][1] for n in rows) and set(rows) == set(variants)
# control through the CLI on its own index must exit 0
idx1 = {"entries": [index["entries"][0]]}
json.dump(idx1, open(os.path.join(SCR, "ctrl_index.json"), "w"))
pc = subprocess.run([PY, "-I", os.path.join(HERE, "verify_s4.py"), "--index", os.path.join(SCR, "ctrl_index.json"),
                     "--frames-dir", SCR, "--out", os.path.join(SCR, "ctrl_manifest.json")], capture_output=True, text=True)
passed = (all(v["ok"] != v["expect_fail"] for v in fn.values()) and cli_ok and p.returncode == 1 and pc.returncode == 0)
res = {"stand_in": {"zip": STAND_IN_ZIP, "member": info.filename, "bytes": exp_size, "crc32": exp_crc,
                    "note": "NOT a fetched jpg: S2 found 0 of the 1,944 listed jpgs"},
       "function_level": fn, "cli_rows_ok": rows, "cli_exit_corrupt_dir": p.returncode, "cli_exit_control": pc.returncode,
       "cli_stdout": p.stdout.strip(), "RED_ARM_PASSED": passed}
json.dump(res, open(os.path.join(HERE, "..", "raw", "s4_red_arm.json"), "w", encoding="utf-8"), indent=1)
for n, v in fn.items():
    print(f"{n:22s} ok={v['ok']!s:5} expect_fail={v['expect_fail']!s:5} size={v['size_ok']} crc={v['crc_ok']} soi={v['soi_ok']} eoi={v['eoi_ok']} decode={v['decode_ok']}")
print("CLI exit on corrupt dir:", p.returncode, "| on control:", pc.returncode, "| per-row match:", cli_ok)
print("RED_ARM_PASSED" if passed else "RED_ARM_FAILED")
sys.exit(0 if passed else 1)
