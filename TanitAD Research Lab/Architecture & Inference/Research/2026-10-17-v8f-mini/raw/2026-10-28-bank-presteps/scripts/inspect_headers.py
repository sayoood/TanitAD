import json, sys
for s in ("bank_s0","bank_s1"):
    m=json.load(open(f"E:/Projects/TanitAD-artifacts/v8f-dagger/{s}/bank.json"))
    c=m["common"]
    print(s, "common keys:", [k for k in c.keys()])
    for k in c:
        if k in ("constants","cfg","eval","clips","train"): continue
        v=c[k]; print("  ",k, (str(v)[:200]))
    print("  eval n clips", len(c["eval"]["clips"]) if isinstance(c.get("eval"),dict) and "clips" in c["eval"] else None)
    print("  cfg", c["cfg"])
    print("  shards", [(x["file"], x["file_sha256"][:10], x["content_sha256"][:10], x.get("n_records")) for x in m["shards"]])
    print("  top keys", list(m.keys()))
    print("  reasons_by_tick", m["reasons_by_tick"], "coverage", m["coverage"])
print(open("E:/Projects/TanitAD-artifacts/v8f-dagger/bank_s0/PROVENANCE.json").read()[:2200])
