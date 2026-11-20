import sys, pathlib
p = pathlib.Path(sys.argv[1])
b = p.read_bytes()
old = b"SCORERFT_BASE_SHA256 = None                  # FILLED BY THE MM: sha256 of C1's final model_final.pt (/workspace/data/refe_runs/p5c_c1/), banked (HF + sha256) before launch"
new = (b"SCORERFT_BASE_SHA256 = \"f09949e2a5f2f9661ad47457e77ddfd10f49117d64e095487809c1d04f7a34a4\"   # FILLED BY THE MM 2026-11-20: sha256 of C1's final model_final.pt "
       b"(/workspace/data/refe_runs/p5c_c1/); MEASURED: the pod sha file == the verified HF pull (Sayood/tanitad-refe-p5c) == the s2_c1 read preflight")
assert b.count(old) == 1, b.count(old)
b = b.replace(old, new)
assert b.count(b"\r") == 0 or b.count(b"\r\n") == b.count(b"\n"), "mixed line endings"
p.write_bytes(b)
print("ZZBASEPIN_OK")
