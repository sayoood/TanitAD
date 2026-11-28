import re, time, json
import truststore; truststore.inject_into_ssl()
import urllib.request
tok = re.search(r'hf_[A-Za-z0-9]+', open('D:/Projects/TanitAD/Keys.txt', encoding='utf-8', errors='ignore').read()).group(0)
url = "https://huggingface.co/datasets/Sayood/tanitad-refe-navtrain-banks/resolve/main/refe_train_grow.tar.zst"
res = []
for lo, n in ((0, 24 * 2**20), (400 * 2**20, 24 * 2**20)):
    req = urllib.request.Request(url, headers={"Authorization": "Bearer " + tok, "Range": f"bytes={lo}-{lo+n-1}"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as r:
        t1 = time.time(); got = 0
        while True:
            b = r.read(1 << 20)
            if not b: break
            got += len(b)
        t2 = time.time()
        res.append({"range_bytes": n, "got": got, "status": r.status, "ttfb_s": round(t1 - t0, 2), "total_s": round(t2 - t0, 2), "MBps": round(got / max(t2 - t1, 1e-6) / 1e6, 2), "MBps_incl_ttfb": round(got / (t2 - t0) / 1e6, 2)})
print(json.dumps(res, indent=1))
json.dump(res, open("dl_probe.json", "w"), indent=1)
