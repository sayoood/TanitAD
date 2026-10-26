"""l1c_select.py -- PREREG_L1C section 1 + AMENDMENT 1: the ordered candidate list of fresh PhysicalAI-AV clips.

Eligibility (AMENDMENT 1, A1.2; every test is on public metadata or the clip's own calibration -- nothing computed from refcv8):
  E1 clip_is_valid; E2 not in the programme's 4,719-clip universe (the local mp4 mirror; sorted-id digest asserted == a48251e89c7a8603...);
  E3 not in the programme digest lists (parity train / deployed val40 / v7.2 eval) by sha256(clip_id);
  E4 `splits/alpamayo_aligned` != alpasim (reserved for AlpaSim closed-loop evaluation);
  E5 the front-wide camera is 1920x1080 and its principal point cy lies in the corpus's observed rig support
     rig A cy in [530, 560) or rig B cy in [740, 770)  (corpus: A 1,996 / B 2,723, min 533.95, max 764.52, none in (560, 740)).
Order: sha256(clip_id) hex ascending.  Raw ids are written ONLY under F:/l1c/select/; the refcv9_l1c copy carries sha12 only.
    python l1c_select.py --n-eligible 420
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hfenv  # noqa: E402

SRC = "C:/Users/Admin/tanitad-data/physicalai"
META = Path("F:/l1c/meta")
ROOT = Path("F:/l1c/pai_root")
SEL = Path("F:/l1c/select")
OUT_SHA12 = HERE.parent / "select"
DIG = "C:/Users/Admin/refcv8_pf/gtmap_oracle/tree_fb47a58/stack/tanitad/data"
UNIVERSE_DIGEST16 = "a48251e89c7a8603"
REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"
RIG_A, RIG_B = (530.0, 560.0), (740.0, 770.0)


def sha(c: str) -> str:
    return hashlib.sha256(c.encode("utf-8")).hexdigest()


def corpus_ids() -> set:
    ids = sorted(os.path.basename(f)[:-4] for f in glob.glob(SRC + "/camera/camera_front_wide_120fov/*.mp4"))
    d = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    if len(ids) != 4719 or not d.startswith(UNIVERSE_DIGEST16):
        raise SystemExit(f"[select] the 4,719-clip universe did not reproduce: n={len(ids)} digest={d[:16]} want {UNIVERSE_DIGEST16}")
    return set(ids)


def digest_exclusions() -> set:
    out = set()
    for n in ("parity_train", "deployed_val40", "v72_eval"):
        d = json.load(open(f"{DIG}/{n}_clip_digests.json", encoding="utf-8"))
        out |= set(d["clip_id_digests"])
    return out


def intr_path(ch: int) -> Path:
    return ROOT / "calibration" / "camera_intrinsics" / f"camera_intrinsics.chunk_{ch:04d}.parquet"


def fetch_intr(ch: int) -> Path:
    p = intr_path(ch)
    if p.exists() and p.stat().st_size > 0:
        return p
    from huggingface_hub import hf_hub_download
    hf_hub_download(REPO, f"calibration/camera_intrinsics/camera_intrinsics.chunk_{ch:04d}.parquet", repo_type="dataset", local_dir=str(ROOT))
    return p


def front_wide_row(ch: int, cid: str):
    df = pd.read_parquet(fetch_intr(ch))
    r = df.loc[(cid, "camera_front_wide_120fov")]
    return {"cx": float(r["cx"]), "cy": float(r["cy"]), "width": int(r["width"]), "height": int(r["height"])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eligible", type=int, default=420, help="stop once this many rig-eligible candidates are listed")
    ap.add_argument("--max-walk", type=int, default=2000)
    a = ap.parse_args()
    hfenv.init()
    for d in (ROOT, SEL, OUT_SHA12):
        d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    idx = pd.read_parquet(f"{SRC}/clip_index.parquet")
    new = pd.read_parquet(META / "clip_index.parquet")
    if not (idx.index.equals(new.index) and (idx["chunk"] == new["chunk"]).all() and (idx["clip_is_valid"] == new["clip_is_valid"]).all()):
        raise SystemExit("[select] the local clip_index disagrees with the dataset's current clip_index.parquet")
    spl = pd.read_parquet(META / "splits" / "alpamayo_aligned.parquet")["split"]
    dspl = pd.read_parquet(META / "splits" / "default.parquet")["split"]
    corp, dig = corpus_ids(), digest_exclusions()
    n0 = len(idx)
    ok = idx[idx["clip_is_valid"]]
    n_valid = len(ok)
    ok = ok[~ok.index.isin(corp)]
    n_notcorp = len(ok)
    ok = ok[[sha(c) not in dig for c in ok.index]]
    n_nodig = len(ok)
    ok = ok[(spl.reindex(ok.index) != "alpasim").to_numpy()]
    n_noalpasim = len(ok)
    order = sorted(ok.index, key=sha)
    funnel = {"clip_index": n0, "E1_valid": n_valid, "E2_not_in_4719": n_notcorp, "E3_not_in_digest_lists": n_nodig,
              "E4_not_alpasim": n_noalpasim}
    print("[select] funnel", funnel, flush=True)
    # E5 walk in sha256 order; the intrinsics chunk parquets are tiny (~37 KB) -- prefetch the next block in threads
    cands, rej, i = [], [], 0
    BLOCK = 64
    while len(cands) < a.n_eligible and i < min(len(order), a.max_walk):
        blk = order[i:i + BLOCK]
        chunks = sorted({int(idx.loc[c, "chunk"]) for c in blk})
        with ThreadPoolExecutor(8) as ex:
            list(ex.map(fetch_intr, chunks))
        for c in blk:
            ch = int(idx.loc[c, "chunk"])
            try:
                r = front_wide_row(ch, c)
            except Exception as e:      # noqa: BLE001
                rej.append({"sha12": sha(c)[:12], "rank": i, "why": f"intrinsics unreadable: {type(e).__name__}"})
                i += 1
                continue
            rig = "A" if RIG_A[0] <= r["cy"] < RIG_A[1] else ("B" if RIG_B[0] <= r["cy"] < RIG_B[1] else None)
            if r["width"] != 1920 or r["height"] != 1080 or rig is None:
                rej.append({"sha12": sha(c)[:12], "rank": i, "why": f"E5 rig: cy={r['cy']:.1f} {r['width']}x{r['height']}"})
            else:
                cands.append({"clip_id": c, "sha12": sha(c)[:12], "sha256": sha(c), "rank": i, "chunk": ch, "rig": rig,
                              "cy": round(r["cy"], 3), "default_split": str(dspl.get(c)), "alpamayo_split": str(spl.get(c))})
            i += 1
            if len(cands) >= a.n_eligible:
                break
    with open(SEL / "candidates.jsonl", "w", encoding="utf-8") as f:
        for r in cands:
            f.write(json.dumps(r) + "\n")
    pub = [{k: v for k, v in r.items() if k not in ("clip_id", "sha256")} for r in cands]
    summ = {"tool": "l1c_select.py", "funnel": funnel, "walked": i, "n_listed": len(cands), "n_rejected_E5": len(rej),
            "rig_counts": {"A": sum(r["rig"] == "A" for r in cands), "B": sum(r["rig"] == "B" for r in cands)},
            "universe_digest16": UNIVERSE_DIGEST16, "ordering": "sha256(clip_id) hex ascending",
            "rejected": rej, "candidates_sha12": pub, "wall_s": round(time.time() - t0, 1),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT_SHA12 / "candidates_sha12.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(f"[select] listed {len(cands)} candidates (walked {i}, E5-rejected {len(rej)}); wall {summ['wall_s']} s  ZZL1CSELDONEZZ", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
