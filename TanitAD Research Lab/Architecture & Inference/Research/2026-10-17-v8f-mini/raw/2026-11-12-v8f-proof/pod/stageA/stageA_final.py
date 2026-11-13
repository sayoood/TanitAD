# pod-side final verification + STAGEA_REPORT.json. Exit 0 iff the cache is the 4,572-clip train cache (+141 eval) and every file is present.
import os, sys, json, hashlib, time
ROOT = "/workspace/v8proof/data/cache_pod"
HR = "/workspace/v8proof/hierrep"
LIST = "/workspace/v8proof/stageA/ops/train4572.txt"
EVAL = "/workspace/v8proof/stageA/ops/clips141.txt"
t0 = float(open(f"{HR}/stageA_start.epoch").read())
fails = []
rep = {}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def rows(split):
    d = {}
    for ln in open(f"{ROOT}/{split}/manifest.jsonl", encoding="utf-8"):
        ln = ln.strip()
        if ln:
            r = json.loads(ln)
            d[r["clip"]] = r
    return d


want = {"train": [l.strip() for l in open(LIST) if l.strip()], "eval": [l.strip() for l in open(EVAL) if l.strip()]}
rep["train_list_sha256"] = sha(LIST)
rep["eval_list_sha256"] = sha(EVAL)
tot = 0
for split, exp in (("train", 4572), ("eval", 141)):
    m = rows(split)
    missing = [c for c in want[split] if c not in m]
    extra = [c for c in m if c not in set(want[split])]
    nfile = bad = 0
    for c in want[split]:
        f = f"{ROOT}/{split}/clips/{c}.pt"
        if os.path.exists(f):
            nfile += 1
            s = os.path.getsize(f)
            tot += s
            if c in m and s != m[c]["bytes"]:
                bad += 1
    rep[split] = {"expected": exp, "manifest_rows": len(m), "files": nfile, "missing_rows": len(missing), "extra_rows": len(extra), "size_mismatch": bad}
    if not (len(m) == exp and nfile == exp and not missing and not extra and not bad):
        fails.append(f"{split}: {rep[split]}")
rep["bytes_total"] = tot
rep["clips"] = rep["train"]["files"]
rep["files"] = rep["train"]["files"] + rep["eval"]["files"]
try:
    meta = json.load(open(f"{ROOT}/train/meta.json"))
    rep["encoder_weights_sha256"] = meta.get("model_safetensors_sha256")
    rep["encoder_meta"] = {k: meta.get(k) for k in ("torch", "transformers", "cuda_device", "enc_dtype", "stored_dtype", "grids")}
    if rep["encoder_weights_sha256"] != "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179":
        fails.append("encoder weights sha differs from the pin")
except Exception as e:
    fails.append(f"meta.json: {e}")
rep["wall_seconds"] = round(time.time() - t0)
rep["finished"] = time.strftime("%F %T")
rep["ok"] = not fails
rep["fails"] = fails
json.dump(rep, open(f"{HR}/STAGEA_REPORT.json", "w"), indent=1)
print(json.dumps(rep, indent=1))
sys.exit(0 if not fails else 3)
