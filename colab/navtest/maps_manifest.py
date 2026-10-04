"""Write the range-fetch manifest for the PUBLIC nuPlan maps zip (every member, paths kept), from the dev box's copy.

    python colab/navtest/maps_manifest.py <evidence dir>
The dev box's `nuplan-maps-v1.1.zip` has the remote object's exact size (970,997,691 B, NAVTEST_ON_COLAB_PLAN.md
section b); zip_member_fetch.py then checks each member's local header NAME and CRC-32 on the VM, so a remote layout
that differed from this copy would fail loudly, never inflate the wrong bytes.
"""
import json
import os
import sys
import zipfile

ZIP = "D:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.1.zip"
URL = "https://motional-nuplan.s3.ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-maps-v1.1.zip"
out = sys.argv[1]
z = zipfile.ZipFile(ZIP)
mem = {i.filename: {"name": i.filename, "dest_name": i.filename, "lho": i.header_offset, "csz": i.compress_size,
                    "usz": i.file_size, "crc": i.CRC, "method": i.compress_type}
       for i in z.infolist() if not i.filename.endswith("/")}
man = {"url": URL, "zip_bytes": os.path.getsize(ZIP), "members": mem, "n": len(mem),
       "csz_total": sum(m["csz"] for m in mem.values()), "usz_total": sum(m["usz"] for m in mem.values())}
os.makedirs(out, exist_ok=True)
json.dump(man, open(os.path.join(out, "maps_manifest.json"), "w", encoding="utf-8"), indent=1)
print(f"ZZMAPS_MANIFEST {len(mem)} members, {man['csz_total']:,} B compressed")
