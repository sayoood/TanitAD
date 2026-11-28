"""P4 — pull exactly the DECLARED clips (p3_selection.json) from the canonical
train split on HF, into a cache dir named so the parity guard flags it.

⚠️ The dir is deliberately NOT the registered parity key: this is a 130-clip
SUBSET of `physicalai-train-e438721ae894`, so `parity=False` is the TRUE record
and must be emitted. Naming it the canonical key to silence the guard would be
the lie the guard exists to prevent.
EVAL clips are pulled FIRST so a truncated pull still yields the powered side.
"""
import json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import truststore; truststore.inject_into_ssl()
from huggingface_hub import hf_hub_download

KEYS = Path(r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\Keys.txt")
TOK = re.search(r"hf_[A-Za-z0-9]+", KEYS.read_text(encoding="utf-8", errors="ignore")).group(0)
REPO = "Sayood/tanitad-physicalai-w120-256x640cyl"
SRC = "physicalai-train-e438721ae894-w120-256x640cyl"
DST = "slotprobe-lead130-w120-256x640cyl"

sel = json.loads(Path(sys.argv[1]).read_text("utf-8"))
root = Path(sys.argv[2]); NW = int(sys.argv[3]) if len(sys.argv) > 3 else 8
dst = root / DST; dst.mkdir(parents=True, exist_ok=True)
staging = root / "_hf"; staging.mkdir(parents=True, exist_ok=True)
want = [(c, "eval") for c in sel["eval_clips"]] + [(c, "train") for c in sel["train_clips"]]

def pull(item):
    cid, side = item
    rel = f"{SRC}/{cid}.v2ep.pt"
    tgt = dst / f"{cid}.v2ep.pt"
    if tgt.exists() and tgt.stat().st_size > 1_000_000:
        return cid, side, tgt.stat().st_size, "CACHED"
    for k in range(4):
        try:
            p = hf_hub_download(REPO, rel, repo_type="dataset",
                                local_dir=str(staging), token=TOK)
            os.replace(p, tgt)
            return cid, side, tgt.stat().st_size, None
        except Exception as ex:
            if k == 3:
                return cid, side, 0, repr(ex)[:200]
            time.sleep(3 * (k + 1))
    return cid, side, 0, "unreachable"

# the split's own manifest (the join reads poses from it)
man_rel = f"{SRC}/_v2manifest.pt"
try:
    p = hf_hub_download(REPO, man_rel, repo_type="dataset", local_dir=str(staging), token=TOK)
    os.replace(p, dst / "_v2manifest.pt")
    print(f"[p4] manifest {os.path.getsize(dst/'_v2manifest.pt')} B", flush=True)
except Exception as ex:
    print(f"[p4] manifest MISSING: {ex!r}", flush=True)

t0 = time.time(); tot = 0; errs = []; done = 0
with ThreadPoolExecutor(max_workers=NW) as ex:
    futs = [ex.submit(pull, w) for w in want]
    for f in as_completed(futs):
        cid, side, n, err = f.result(); done += 1; tot += n
        if err and err != "CACHED":
            errs.append({"clip": cid, "err": err})
        if done % 10 == 0 or done == len(want):
            dt = time.time() - t0
            print(f"[p4] {done}/{len(want)}  {tot/1e9:.2f} GB  "
                  f"{tot/1e6/max(dt,1e-9):.2f} MB/s  {dt/60:.1f} min  errs={len(errs)}",
                  flush=True)
have = sorted(p.stem.replace(".v2ep", "") for p in dst.glob("*.v2ep.pt"))
rec = {"repo": REPO, "src_split": SRC, "cache_dir": str(dst),
       "n_requested": len(want), "n_present": len(have),
       "missing": sorted(set(c for c, _ in want) - set(have)),
       "total_bytes": tot, "wall_s": round(time.time()-t0, 1),
       "MB_per_s": round(tot/1e6/max(time.time()-t0, 1e-9), 2), "errors": errs,
       "parity_note": ("a 130-clip SUBSET of the canonical parity corpus; the dir "
                       "name is unregistered ON PURPOSE so the guard records "
                       "parity=False")}
(root / "p4_pull_eps.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in rec.items() if k != "errors"}, indent=1), flush=True)
