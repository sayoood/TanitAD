"""HNSCREEN-FP-COST: MEASURE the full-content reads of the ctx-cache per screen, by instrumenting the reader (`gate_token._sha256_and_size` = one file fully hashed).

A READ is one `_sha256_and_size` call; a FULL PASS over the cache = reads / n_files.  Measured on the small synthetic sealed cache the chain test builds (real ctx_cache build + seal), with the REAL gate run as a subprocess (the shim in
refe/test_hnscreen_ctx_chain.py logs {sub, reads} per gate process) and the seal / the fill (hn_fingerprints.fingerprints_expected) counted in-process.

  content mode  = today's chain (bit-for-bit; the control)
  anchored mode = (b') : full content at the seal, the fill, the chain START (anchor) and the chain END (end verify); the intermediate calls compare (relpath, size, mtime_ns) only

usage (cwd = refe/):  python <this file> <out.json>
"""
import json
import os
import shutil
import sys
import time

REFE = "C:/Users/Admin/par18/hnfp3/work/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe"
sys.path.insert(0, REFE)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import test_hnscreen_ctx_chain as T  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "reads_per_screen.json"


def count_in_process(fn):
    """run fn() with gate_token._sha256_and_size counting every call; -> (result, list of paths read)"""
    import gate_token as GT
    seen = []
    real = GT._sha256_and_size

    def counting(path, *a, **k):
        seen.append(os.path.abspath(str(path)))
        return real(path, *a, **k)
    GT._sha256_and_size = counting
    try:
        res = fn()
    finally:
        GT._sha256_and_size = real
    return res, seen


def main():
    R = T.Real()
    out = {"tip": open("C:/Users/Admin/par18/hnfp3/TIP_SHA.txt").read().strip(), "n_files": R.n_files, "cache_bytes": R.cache_bytes, "evidence": "MEASURED (this script; synthetic sealed cache built by ctx_cache.finalize)"}
    try:
        import ctx_cache as CC
        import hn_fingerprints as HF
        # the SEAL: re-seal a copy of the pristine cache (finalize ends in exactly this call)
        c1 = os.path.join(R.w.base, "meas_seal")
        shutil.copytree(R.pristine, c1)
        _, seen = count_in_process(lambda: CC.seal(c1))
        out["seal"] = {"reads": len(seen), "full_passes": len(seen) / R.n_files, "files": len(set(seen))}
        # the FILL: hn_fingerprints.fingerprints_expected(bank, cache)
        bank = os.path.join(R.w.base, "meas_bank.bin")
        open(bank, "wb").write(b"BANK")
        _, seen = count_in_process(lambda: HF.fingerprints_expected(bank, c1))
        in_cache = [p for p in seen if os.path.abspath(c1) in p]
        out["fill"] = {"reads_of_the_cache": len(in_cache), "full_passes": len(in_cache) / R.n_files, "reads_of_the_bank": len(seen) - len(in_cache)}
        M = T.TC.load_module(T.read_src(), "meas")
        runs = {}
        for seeds in ((0,), (0, 1), (0, 1, 2)):
            for mode in ("content", "default"):
                tag = f"{mode}_{len(seeds)}seed"
                sc, rc, ch, rows, cache = T.scenario(R, M, "M_" + tag, seeds=seeds, mode=mode)
                gate_reads = sum(r["reads"] for r in rows)
                per_sub = {}
                for r in rows:
                    per_sub.setdefault(r["sub"], []).append(r["reads"])
                tot = gate_reads + out["seal"]["reads"] + out["fill"]["reads_of_the_cache"]
                runs[tag] = {"rc": rc, "gate_calls": [(r["sub"], r["reads"]) for r in rows], "gate_reads": gate_reads, "total_reads_with_seal_and_fill": tot, "total_full_passes": tot / R.n_files,
                             "per_seed_gate_reads": (gate_reads / len(seeds)) if mode == "content" else None, "stage_order": ch.stage_order_run}
        out["runs"] = runs
        c3, a3 = runs["content_3seed"], runs["default_3seed"]
        out["headline_3_seed_screen"] = {"today_reads": c3["total_reads_with_seal_and_fill"], "today_full_passes": c3["total_full_passes"], "anchored_reads": a3["total_reads_with_seal_and_fill"],
                                         "anchored_full_passes": a3["total_full_passes"], "ruling_and_hnfp2_say": 17, "note": "17 = the ruling's count (one verify per seed); the source and this measurement give 20: per seed run_checks 2 + finalize 2 + mint verify 1 + start verify 1 = 6, x3 = 18, + seal 1 + fill 1"}
    finally:
        R.close()
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    print(json.dumps(out["headline_3_seed_screen"], indent=1))
    for k, v in out["runs"].items():
        print(k, "rc", v["rc"], "gate", v["gate_calls"], "total", v["total_reads_with_seal_and_fill"], "=", v["total_full_passes"], "passes")
    print("seal", out["seal"], "fill", out["fill"])
    print("ZZMEASURE_DONE")


if __name__ == "__main__":
    main()
