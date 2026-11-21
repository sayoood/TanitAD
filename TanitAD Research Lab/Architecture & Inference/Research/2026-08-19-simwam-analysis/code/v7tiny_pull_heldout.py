"""Pull a GENUINELY HELD-OUT episode set for v7-tiny's G2 measurement.

⛔ WHY THIS EXISTS. The v7-tiny arms were trained with NO --v2-val-cache, so all
130 clips of `slotprobe-lead130` are IN-SAMPLE. G2 asks "does the predictor beat
HOLD on next-latent prediction"; measured in-sample that question is flattered by
whatever the 19M model memorised of 8,000 sampled windows. The gate number has to
come from clips the encoder has never seen.

The 130 are a SUBSET of `physicalai-train-e438721ae894` (2,376 episodes), so
2,246 clips are available and untouched. This pulls a deterministic, evenly
strided sample of them -- strided rather than head-of-list so the held-out set
spans the split rather than sitting in one contiguous region of it.

⚠️ NON-PARITY by construction, exactly like the training subset: the dir name
says so, and that is the TRUE record.
"""
import json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import truststore; truststore.inject_into_ssl()
import urllib.request
from huggingface_hub import hf_hub_download

KEYS = Path(r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\Keys.txt")
TOK = re.search(r"hf_[A-Za-z0-9]+",
                KEYS.read_text(encoding="utf-8", errors="ignore")).group(0)
REPO = "Sayood/tanitad-physicalai-w120-256x640cyl"
SRC = "physicalai-train-e438721ae894-w120-256x640cyl"
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
TRAIN = ROOT / "slotprobe-lead130-w120-256x640cyl"
DST = ROOT / "v7tiny-heldout24-w120-256x640cyl"
N_WANT = int(sys.argv[2]) if len(sys.argv) > 2 else 24
DST.mkdir(parents=True, exist_ok=True)
staging = ROOT / "_hf"; staging.mkdir(parents=True, exist_ok=True)

trained = {p.name.split(".")[0] for p in TRAIN.glob("*.v2ep.pt")}
print(f"  in-sample (excluded): {len(trained)} clips", flush=True)

# tree API, paginated -- list_repo_files hangs on repos this size
names, cursor = [], None
for page in range(60):
    u = (f"https://huggingface.co/api/datasets/{REPO}/tree/main/{SRC}"
         f"?limit=1000" + (f"&cursor={cursor}" if cursor else ""))
    rq = urllib.request.Request(u, headers={"Authorization": f"Bearer {TOK}"})
    with urllib.request.urlopen(rq, timeout=90) as r:
        batch = json.loads(r.read().decode("utf-8"))
        link = r.headers.get("Link", "") or ""
    if not batch:
        break
    names += [e["path"].split("/")[-1] for e in batch if e.get("type") == "file"]
    m = re.search(r'cursor=([^&>;"]+)', link)
    if not m:
        break
    cursor = m.group(1)
print(f"  repo dir lists {len(names)} files", flush=True)

allc = sorted({n.split(".")[0] for n in names if n.endswith(".v2ep.pt")})
avail = [c for c in allc if c not in trained]
print(f"  {len(allc)} total clips · {len(avail)} never trained on", flush=True)
if len(avail) < N_WANT:
    print("  ⛔ not enough held-out clips"); sys.exit(1)
stride = max(1, len(avail) // N_WANT)
pick = [avail[i * stride] for i in range(N_WANT)]
print(f"  picking {len(pick)} at stride {stride}", flush=True)


def pull(cid):
    tgt = DST / f"{cid}.v2ep.pt"
    if tgt.exists() and tgt.stat().st_size > 1_000_000:
        return cid, tgt.stat().st_size, "CACHED"
    for k in range(4):
        try:
            p = hf_hub_download(REPO, f"{SRC}/{cid}.v2ep.pt", repo_type="dataset",
                                local_dir=str(staging), token=TOK)
            os.replace(p, tgt)
            return cid, tgt.stat().st_size, None
        except Exception as ex:
            if k == 3:
                return cid, 0, repr(ex)[:160]
            time.sleep(3 * (k + 1))
    return cid, 0, "unreachable"


t0, ok, bad = time.time(), [], []
with ThreadPoolExecutor(max_workers=6) as ex:
    for f in as_completed([ex.submit(pull, c) for c in pick]):
        cid, sz, err = f.result()
        (bad if err and err != "CACHED" else ok).append(cid)
        print(f"    [{len(ok)+len(bad):>2}/{len(pick)}] {cid[:10]} "
              f"{sz/1e6:>6.1f} MB {err or ''} ({time.time()-t0:.0f}s)", flush=True)

(DST / "_PROVENANCE.json").write_text(json.dumps({
    "_evidence_class": "MEASURED (ours; pulled from the canonical split on HF)",
    "purpose": "held-out episodes for v7-tiny G2 -- the trainer used NO val cache",
    "source_repo": REPO, "source_dir": SRC,
    "parity": False,
    "parity_note": "a 24-clip SUBSET of physicalai-train-e438721ae894; NON-PARITY "
                   "by construction, same as the 130-clip training subset",
    "disjoint_from": "slotprobe-lead130-w120-256x640cyl",
    "n_excluded_in_sample": len(trained), "n_available": len(avail),
    "stride": stride, "clips": sorted(ok), "failed": sorted(bad),
}, indent=1), encoding="utf-8")
print(f"\n  {len(ok)} pulled · {len(bad)} failed · {time.time()-t0:.0f}s -> {DST}")
