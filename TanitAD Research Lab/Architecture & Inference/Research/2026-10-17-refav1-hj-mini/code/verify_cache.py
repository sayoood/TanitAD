"""Integrity checks for the RA-1a pooled-DINOv3 cache -- run AFTER an encode (see ../HANDOFF.md).

    python verify_cache.py --split train --n 600  [--out C:/Users/Admin/refav1_hj_cache] [--redo 5 --device cuda]

C1  manifest        every manifest line parses; every file exists, byte size and md5 match           (all files)
C2  content         torch.load(weights_only=True): key set, meta, shapes == manifest, n_frames ==
                    ceil(t_ep/2), every float tensor finite (kin3/p10 row 0 is the ONLY allowed NaN),
                    fp8 dequantises finite, content_md5 recomputed == manifest                        (--sample K, 0 = all)
C3  disjointness    manifest clips vs the eval-141 recomputed from FOUR independent sources           (all files)
C4  coverage        manifest clips == the deterministic pick (missing / extra counted)
C5  bit-exact redo  re-encode K clips through the SAME functions and compare content_md5 to the
                    manifest (the prereg's "bit-exact re-encode on 5 clips")                          (--redo K)
Prints one JSON summary and ZZVERIFY-OK / ZZVERIFY-FAIL; exit 0 / 1.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pull_encode_pooled as PE  # noqa: E402

FLOAT_KEYS = ("mean", "v", "a", "steer", "yaw_rate", "pose_grid", "kin3", "p10")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", choices=["train", "eval"], required=True)
    ap.add_argument("--n", type=int, default=0, help="train: the pick size this cache is meant to hold (0 = all 4,572)")
    ap.add_argument("--out", default=PE.D_OUT)
    ap.add_argument("--tag", default="", help="split directory suffix, as given to pull_encode_pooled (e.g. _m416)")
    ap.add_argument("--source", choices=PE.SOURCES, default="auto", help="C5 only: where to re-read the v2ep (same meaning as the encoder)")
    ap.add_argument("--src-dir", default=None)
    ap.add_argument("--src-geom", choices=["256", "416"], default="256")
    ap.add_argument("--grids", default="4x10", help="grids EVERY clip must carry (4x10 / 4x10,8x20)")
    ap.add_argument("--sample", type=int, default=50, help="C2 on K evenly spaced clips (0 = all)")
    ap.add_argument("--redo", type=int, default=0, help="C5: re-encode K clips and compare content_md5")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8, help="md5 threads")
    ap.add_argument("--allow-partial", action="store_true", help="C4: do not fail on missing clips (mid-run check)")
    ap.add_argument("--listing", default=PE.D_INDEX)
    ap.add_argument("--eval-eps", default=PE.D_EVAL_EPS)
    ap.add_argument("--eval-cache", default=PE.D_EVAL_CACHE)
    ap.add_argument("--eval-labels", default=PE.D_EVAL_LABELS)
    ap.add_argument("--train-labels", default=PE.D_TRAIN_LABELS)
    ap.add_argument("--encoder-module", default=PE.D_ENCODER)
    ap.add_argument("--thor", default=PE.THOR)
    a = ap.parse_args(argv)
    grids = [g for g in a.grids.split(",") if g]
    root = Path(a.out) / (a.split + a.tag)
    res: dict = {"split": a.split, "root": str(root), "checks": {}}
    fails: list[str] = []

    def check(name, ok, **info):
        res["checks"][name] = {"ok": bool(ok), **info}
        if not ok:
            fails.append(name)
        print(f"[verify] {name}: {'OK' if ok else 'FAIL'} {json.dumps(info)[:300]}", flush=True)

    # ---------------------------------------------------------------- C1
    ents = PE.load_manifest(root / "manifest.jsonl")
    n_lines = sum(1 for _ in open(root / "manifest.jsonl", encoding="utf-8")) if (root / "manifest.jsonl").exists() else 0
    t0 = time.time()

    def one(e):
        p = root / e["file"]
        if not p.exists():
            return e["clip"], "missing"
        if p.stat().st_size != e["bytes"]:
            return e["clip"], "size"
        if PE.file_hash(p) != e["md5"]:
            return e["clip"], "md5"
        return e["clip"], None

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        bad = [(c, w) for c, w in ex.map(one, ents.values()) if w]
    check("C1_manifest_files", len(ents) > 0 and not bad, n_entries=len(ents), n_manifest_lines=n_lines,
          n_duplicate_lines=n_lines - len(ents), n_bad=len(bad), first_bad=[(PE.sha12(c), w) for c, w in bad[:5]],
          secs=round(time.time() - t0, 1))
    on_disk = {p.stem for p in (root / "clips").glob("*.pt")}
    check("C1b_no_orphan_files", on_disk == set(ents), n_files=len(on_disk), n_files_without_manifest=len(on_disk - set(ents)),
          n_manifest_without_file=len(set(ents) - on_disk))

    # ---------------------------------------------------------------- C3 / C4
    sa = SimpleNamespace(eval_eps=a.eval_eps, eval_cache=a.eval_cache, eval_labels=a.eval_labels,
                         train_labels=a.train_labels, listing=a.listing, thor=a.thor)
    sets = PE.clip_sets(sa)
    ev = set(sets["eval"])
    ev_lab = {r["clip_id"] for r in PE.read_jsonl_gz(a.eval_labels)}
    cl = set(ents)
    if a.split == "train":
        check("C3_disjoint_from_eval", not (cl & ev) and not (cl & ev_lab) and cl <= set(sets["train_all"]),
              n_clips=len(cl), n_in_eval141=len(cl & ev), n_in_eval_labels=len(cl & ev_lab),
              n_outside_train_set=len(cl - set(sets["train_all"])))
        want = set(PE.pick_clips(sets, "train", a.n))
    else:
        check("C3_disjoint_from_train", not (cl & set(sets["train_all"])) and cl <= ev, n_clips=len(cl),
              n_in_train=len(cl & set(sets["train_all"])), n_outside_eval141=len(cl - ev))
        want = ev
    miss, extra = want - cl, cl - want
    check("C4_coverage", (not extra) and (a.allow_partial or not miss), n_expected=len(want), n_have=len(cl),
          n_missing=len(miss), n_extra=len(extra), partial_ok=a.allow_partial)

    # ---------------------------------------------------------------- C2
    ordered = sorted(ents, key=PE.sha12)
    if a.sample and len(ordered) > a.sample:
        step = len(ordered) / a.sample
        ordered = [ordered[int(i * step)] for i in range(a.sample)]
    problems = []
    for c in ordered:
        e = ents[c]
        try:
            d = torch.load(root / e["file"], map_location="cpu", weights_only=True)
            m = d["meta"]
            assert m["clip_id"] == c and m["pipeline"] == PE.PIPELINE and m["dtype"] == e["dtype"]
            assert m.get("source") == e.get("source"), "meta.source != manifest source"
            T = m["n_frames"]
            assert T == math.ceil(m["t_ep"] / 2) == e["n_frames"], "n_frames != ceil(t_ep/2)"
            for g in grids:
                t = d["pool" + g]
                gh, gw = PE.GRIDS[g]
                assert tuple(t.shape) == (T, gh * gw, 1024) and str(t.dtype).replace("torch.", "") == e["dtype"], f"{g} shape/dtype"
                assert bool(torch.isfinite(t.float()).all()), f"{g} not finite"
            assert tuple(d["mean"].shape) == (T, 1024) and d["mean"].dtype == torch.float16
            for k in FLOAT_KEYS:
                t = d[k].float()
                assert t.shape[0] == T, f"{k} length"
                body = t[1:] if k in ("kin3", "p10") else t
                assert bool(torch.isfinite(body).all()), f"{k} not finite"
            assert bool(torch.isnan(d["kin3"][0]).all()) and bool(torch.isnan(d["p10"][0]).all()), "row-0 NaN contract"
            got = PE.content_md5({k: v for k, v in d.items() if k != "meta"})
            assert got == e["content_md5"], "content_md5 differs from manifest"
            assert {k for k in d} >= {"meta", "mean", "v", "a", "steer", "yaw_rate", "pose_grid", "kin3", "p10"} | {"pool" + g for g in grids}
        except Exception as ex_:
            problems.append((PE.sha12(c), f"{type(ex_).__name__}: {str(ex_)[:100]}"))
    check("C2_content", not problems, n_checked=len(ordered), n_problems=len(problems), first=problems[:5])

    # ---------------------------------------------------------------- C5
    if a.redo:
        if a.threads > 0:
            torch.set_num_threads(a.threads)
        if a.device == "cuda" and not torch.cuda.is_available():
            raise SystemExit("ZZNO-CUDAZZ --device cuda requested and unavailable")
        enc = PE.load_encoder_module(a.encoder_module)
        model, mean, std, n_special = PE.build_model(a.device, a.dtype)
        qdtype = torch.float8_e4m3fn if PE.fp8_ok(Path(a.out)) else torch.float16
        cands = [c for c in sorted(ents, key=PE.sha12) if ents[c].get("enc_dtype") == a.dtype][: a.redo]
        work = Path(a.out) / "src_pull"
        work.mkdir(parents=True, exist_ok=True)
        fa = SimpleNamespace(split=a.split, source=a.source, src_dir=a.src_dir, src_geom=a.src_geom, eval_eps=a.eval_eps,
                             thor=a.thor)
        src = PE.resolve_source(fa)
        print(f"[verify] redo source: {src['kind']}", flush=True)
        rows = []
        for c in cands:
            e = ents[c]
            if e.get("source") != src["kind"]:
                rows.append({"clip": PE.sha12(c), "bit_exact": False, "built_on": e.get("device"), "redone_on": a.device,
                             "skipped": f"manifest source {e.get('source')} != redo source {src['kind']}"})
                continue
            o, spath, local = PE.prepare(fa, src, c, work)
            p = PE.encode_clip(o, enc, model, mean, std, n_special, a.device, e["grids"], qdtype)
            same = PE.content_md5(p) == e["content_md5"]
            rows.append({"clip": PE.sha12(c), "bit_exact": same, "grids": e["grids"], "built_on": e.get("device"),
                         "redone_on": a.device})
            if not local:
                Path(spath).unlink(missing_ok=True)
            print(f"[verify] redo {PE.sha12(c)} bit_exact={same}", flush=True)
        cross_dev = any(r["built_on"] != r["redone_on"] for r in rows)
        n_skip = sum(1 for r in rows if r.get("skipped"))
        check("C5_bit_exact_redo", bool(rows) and n_skip == 0 and all(r["bit_exact"] for r in rows), n_redone=len(rows) - n_skip,
              n_bit_exact=sum(r["bit_exact"] for r in rows), n_skipped_source_mismatch=n_skip,
              skipped=[r["skipped"] for r in rows if r.get("skipped")][:2], cross_device=cross_dev,
              note="cross-device redo is NOT a determinism test" if cross_dev else "same device")

    res["ok"] = not fails
    res["failed"] = fails
    print(json.dumps(res, indent=1))
    print("ZZVERIFY-OK" if not fails else f"ZZVERIFY-FAIL {fails}", flush=True)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
