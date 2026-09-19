"""E20 input: every byte the refcv6 pod would read, measured where it lives today.

READ-ONLY. Local: file sizes, parquet FOOTERS and small JSON reads — no cache file is opened, so
the live A8 run reading D: is not contended. HF: metadata only (repo trees, ``usedStorage``);
nothing is downloaded, uploaded, created or changed. Token read in place from the git-ignored
``Keys.txt``; never printed, never on argv.

⛔ Clip ids never leave this process. The report carries sha12 = sha256(id)[:12], counts and bytes.

    python pod_data_inventory.py <out.json>
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

REPO = Path("D:/Projects/TanitAD")
ART = Path("D:/Projects/TanitAD-artifacts")
CAM = Path("C:/Users/Admin/tanitad-data/physicalai/camera/camera_front_wide_120fov")
V8 = Path("C:/Users/Admin/tanitad-wt/_s2build/release/v8")
CORPUS_REPO = "Sayood/tanitad-v7-training-corpus"


def s12(cid: str) -> str:
    return hashlib.sha256(cid.encode("utf-8")).hexdigest()[:12]


def du(p: Path) -> tuple[int, int]:
    if p.is_file():
        return p.stat().st_size, 1
    fs = [f for f in p.rglob("*") if f.is_file()]
    return sum(f.stat().st_size for f in fs), len(fs)


UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def _cid(r: dict) -> str:
    for k in ("clip_id", "clip", "id", "clip_uuid"):
        v = r.get(k)
        if isinstance(v, str) and UUID.fullmatch(v):
            return v
    for v in r.values():
        if isinstance(v, str) and UUID.fullmatch(v):
            return v
    raise KeyError("no clip-id field in a label record")


def label_ids(path: Path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [_cid(json.loads(line)) for line in f if line.strip()]


def clip_key(parts: list[str]) -> str | None:
    """sha12 of the clip a repo file belongs to, from a UUID or a 12-hex name component."""
    for comp in parts[2:]:
        m = UUID.search(comp)
        if m:
            return s12(m.group(0))
        m = re.fullmatch(r"([0-9a-f]{12})(?:[._-].*)?", comp)
        if m:
            return m.group(1)
    return None


def main(out_path: str) -> int:
    R: dict = {"read_only": True, "clip_ids_in_report": 0}

    # ── 1. the corpus mirror on C: (the build's source) ────────────────────────
    mp4 = {p.name[:36]: p.stat().st_size for p in CAM.glob("*.mp4")}
    frames = {c: pq.ParquetFile(CAM / f"{c}.timestamps.parquet").metadata.num_rows for c in mp4}
    R["mirror"] = {"clips": len(mp4), "mp4_bytes": sum(mp4.values()),
                   "timestamp_rows_total": sum(frames.values())}

    # ── 2. which clips the POD trains on: the v8 label split, checked against B1 ──
    v40 = json.loads((REPO / "stack/tanitad/data/deployed_val40_clip_digests.json").read_text(encoding="utf-8"))
    v40set = set(v40["clip_id_digests"])
    b1 = sorted(c for c in mp4 if hashlib.sha256(c.encode()).hexdigest() not in v40set)
    pm = json.loads((REPO / "stack/tanitad/data/parity_manifest.json").read_text(encoding="utf-8"))
    b1_pin = pm["corpora"]["physicalai-b1-w120-256x640cyl"]["clip_membership"]["clip_id_sha256_sorted"]
    b1_ok = hashlib.sha256("\n".join(b1).encode()).hexdigest() == b1_pin
    tr, ev = label_ids(V8 / "s2_labels_v8_train.jsonl.gz"), label_ids(V8 / "s2_labels_v8_eval.jsonl.gz")
    trs, evs, b1s = set(tr), set(ev), set(b1)
    R["splits"] = {
        "b1_clips": len(b1), "b1_membership_digest_reproduced": b1_ok,
        "v8_train_records": len(tr), "v8_train_distinct": len(trs),
        "v8_eval_records": len(ev), "v8_eval_distinct": len(evs),
        "train_and_eval_overlap": len(trs & evs),
        "train_inside_b1": len(trs & b1s), "eval_inside_b1": len(evs & b1s),
        "b1_minus_train_minus_eval": len(b1s - trs - evs),
        "train_with_mp4_in_mirror": len(trs & set(mp4)),
    }

    # ── 3. the only 416x1024 cache: per-episode bytes vs its source ───────────
    cdir = ART / "v2ep-eval139-416x1024cyl"
    man = json.loads((cdir / "MANIFEST.json").read_text(encoding="utf-8"))
    files12 = {s12(p.name[:36]): p.stat().st_size for p in cdir.glob("*.v2ep.pt")}
    by12 = {s12(c): c for c in mp4}
    rows = []
    for c in man["clips"]:
        cid = by12.get(c["clip_sha12"])
        # ⚠️ per-clip sizes in MB (6 dp = exact to the byte), never raw byte counts: an 8-digit
        # byte count IS an 8-hex token, and one coincided with a clip-id prefix — the staging
        # gate caught it on the first build of this report.
        rows.append({"sha12": c["clip_sha12"], "n_frames": c["n_frames"], "mb": round(c["bytes"] / 1e6, 6),
                     "mb_on_disk": round(files12[c["clip_sha12"]] / 1e6, 6) if c["clip_sha12"] in files12 else None,
                     "mp4_mb": round(mp4[cid] / 1e6, 6) if cid else None,
                     "timestamp_rows": frames.get(cid) if cid else None,
                     "in_v8_eval": (cid in evs) if cid else None,
                     "in_v8_train": (cid in trs) if cid else None})
    R["eval139_416"] = {"codec": man.get("codec"), "n_stack": man.get("n_stack"),
                        "frame": man.get("frame"), "n_clips": man.get("n_clips"),
                        "total_bytes_manifest": man.get("total_bytes"),
                        "dir_bytes": du(cdir)[0], "rows": rows}

    # ── 4. the train population's covariates (aggregates only) ─────────────────
    trm = [c for c in sorted(trs) if c in mp4]
    R["train_covariates"] = {
        "n": len(trm), "mp4_bytes": sum(mp4[c] for c in trm),
        "timestamp_rows": sum(frames[c] for c in trm),
        "mp4_bytes_mean": sum(mp4[c] for c in trm) / max(len(trm), 1),
        "timestamp_rows_mean": sum(frames[c] for c in trm) / max(len(trm), 1),
        "timestamp_rows_hist": dict(sorted(Counter(frames[c] // 10 * 10 for c in trm).items())),
        # per-clip covariates, sha12-keyed, so the size model can be re-applied without ids
        "per_clip": [[s12(c), round(mp4[c] / 1e6, 6), frames[c]] for c in trm],   # MB, see above
    }

    # ── 5. everything else the trainer reads (A8's argv, eval side, as it runs today) ──
    aux = {
        "clean124_halfA": ART / "v2ep-eval124clean-416x1024cyl-halfA",
        "clean124_halfB": ART / "v2ep-eval124clean-416x1024cyl-halfB",
        "sam3_maps_eval": ART / "sam3-maps-eval",
        "agent_join_3d_eval": ART / "b1-agent-join-3d-20260917",
        "extrinsics141": ART / "refcv5v2_final/extrinsics141.json",
        "v8_labels": V8,
    }
    R["aux_local"] = {k: dict(zip(("bytes", "files"), du(p))) for k, p in aux.items()}

    # ── 6. HF, metadata only ────────────────────────────────────────────────────
    import truststore
    truststore.inject_into_ssl()
    from huggingface_hub import HfApi
    tok = re.search(r"hf_[A-Za-z0-9]+", (REPO / "Keys.txt").read_text(encoding="utf-8")).group(0)
    api = HfApi(token=tok)
    info = api.dataset_info(CORPUS_REPO, expand=["usedStorage", "private", "sha"])
    folders, nfiles = defaultdict(int), defaultdict(int)
    maps12, unkeyed = defaultdict(dict), Counter()
    for f in api.list_repo_tree(CORPUS_REPO, repo_type="dataset", recursive=True, revision=info.sha):
        if getattr(f, "size", None) is None:
            continue
        parts = f.path.split("/")
        key = ("/".join(parts[:2]) if parts[0] in ("semantic_maps", "agents", "calibration",
                                                   "lidar_bev_gt") and len(parts) > 2 else parts[0])
        folders[key] += f.size
        nfiles[key] += 1
        if parts[0] == "semantic_maps" and len(parts) > 2 and parts[1] in ("gt", "gt_flagged"):
            k = clip_key(parts)
            if k:
                maps12[parts[1]][k] = maps12[parts[1]].get(k, 0) + f.size
            else:
                unkeyed[parts[1]] += 1      # a 0 coverage must not hide an unparsed name
    tr12 = {s12(c) for c in trs}
    R["hf_corpus"] = {
        "repo": CORPUS_REPO, "revision": info.sha, "private": info.private,
        "usedStorage": getattr(info, "used_storage", None) or getattr(info, "usedStorage", None),
        "folders": {k: {"bytes": folders[k], "files": nfiles[k]} for k in sorted(folders)},
        "maps": {t: {"clips": len(d), "bytes": sum(d.values()),
                     "train_clips_covered": len(set(d) & tr12)} for t, d in maps12.items()},
        "map_files_without_a_clip_key": dict(unkeyed),
    }
    # the account's two storage pools (the ordering trap needs both)
    pools = {"private": 0, "public": 0}
    big = []
    for kind, lister in (("model", api.list_models), ("dataset", api.list_datasets),
                         ("space", api.list_spaces)):
        for r in lister(author="Sayood"):
            i = api.repo_info(r.id, repo_type=kind, expand=["usedStorage", "private"])
            u = getattr(i, "used_storage", None) or getattr(i, "usedStorage", None) or 0
            pools["private" if i.private else "public"] += u
            big.append((u, r.id, kind, bool(i.private)))
    big.sort(reverse=True)
    R["hf_account"] = {"pools_bytes": pools, "largest": [
        {"repo": rid, "type": k, "private": p, "bytes": u} for u, rid, k, p in big[:6]]}

    text = json.dumps(R, indent=1)
    # ⛔ the id guard, before anything is written
    if re.search(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", text):
        raise SystemExit("⛔ a UUID reached the report; nothing written")
    Path(out_path).write_text(text + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in R.items() if k not in ("eval139_416", "train_covariates")}, indent=1)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
