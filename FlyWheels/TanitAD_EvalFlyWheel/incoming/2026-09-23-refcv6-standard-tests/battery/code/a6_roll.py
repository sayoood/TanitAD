"""SPEC A6 step 3: ONE refcv6 roll over the A6 TRAIN clips, in its own process (the orchestrating parent
never holds CUDA). Identical to the battery's roll (`run_battery.roll_one`: refcv3_arm.run_dump through
refcv6_roll, same flags, same arms, frame memo) except for three inputs -- the TRAIN sub-cache, the
TRAIN labels and the TRAIN max-speed sidecar -- which reach the loader through the RECORDED
`REFCV6_REMAP_OVERRIDES` file, and one window rule: the V3Dataset index restricted to t % 5 == 0.

usage (REFCV6_REMAP_OVERRIDES must point at a JSON {"--eval-cache": ..., "--eval-labels": ...,
"--speed-max-sidecar-v6-eval": ...} BEFORE this process starts):
  python a6_roll.py --ckpt <ckpt> --config <config.json> --seed 0 --episodes <train sub-cache>
                    --labels <train labels> --dump-dir <dir> --out-json <json>
"""
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8")
import refcv6_loader as L  # noqa: E402

L.bootstrap()
import torch  # noqa: E402
import refcv6_roll as RR  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-clips", type=int, default=0, help="SMOKE ONLY")
    ap.add_argument("--smoke-cpu-fp32-trunk", action="store_true", help="SMOKE ONLY (CPU has no native bf16)")
    a = ap.parse_args()
    if a.smoke_cpu_fp32_trunk:                          # roll_seed.py's smoke patch, verbatim
        _orig = L.build_model

        def _fp32(*aa, **kk):
            m, c, ar, r = _orig(*aa, **kk)
            for mod in m.modules():
                lv = getattr(mod, "memory_levers", None)
                if isinstance(lv, dict) and "bf16" in lv:
                    lv["bf16"] = False
                    lv["channels_last"] = False
            r["smoke_note"] = "trunk bf16/NHWC OFF (--smoke-cpu-fp32-trunk)"
            return m, c, ar, r
        L.build_model = _fp32
    ovr = L.REMAP_OVERRIDES_FILE
    if not ovr:
        raise SystemExit("[a6] REFCV6_REMAP_OVERRIDES is not set -- this would roll the EVAL cache; refusing")
    o = json.load(open(ovr, encoding="utf-8"))
    if os.path.normcase(os.path.abspath(o.get("--eval-cache", ""))) != os.path.normcase(os.path.abspath(a.episodes)):
        raise SystemExit("[a6] the override's --eval-cache is not --episodes -- refusing")
    st = RR.install_patches(config_path=a.config, windows=None, frame_memo=True)
    R = RR.ra3()
    orig = R.build_corpus

    def build_corpus(aa, cfg, prov):
        res = list(orig(aa, cfg, prov))
        ds = res[3]
        before = len(ds.index)
        keep = [tuple(x) for x in ds.index if int(x[1]) % 5 == 0]
        if a.max_clips:
            eps_keep = sorted({e for e, _ in keep})[: a.max_clips]
            keep = [x for x in keep if x[0] in set(eps_keep)][: a.max_clips * 2]
        ds.index = keep
        wr = {"rule": "SPEC A6: the trainer's V3Dataset index restricted to t % 5 == 0 (the eval S2 grid's stride)",
              "n_index_before": before, "n_index_after": len(ds.index),
              "smoke_max_clips": a.max_clips or None}
        res[5]["window_restriction"] = wr
        st["window_restriction"] = wr
        return tuple(res)

    R.build_corpus = build_corpus
    os.makedirs(a.dump_dir, exist_ok=True)
    argv = ["--ckpt", a.ckpt, "--config", a.config, "--episodes", a.episodes, "--labels", a.labels,
            "--nav-source", "v72", "--grid", "2s", "--action-units", "steer",
            "--window-stride", "1", "--with-oracle-sel", "--with-navflip",
            "--dump-dir", a.dump_dir, "--dump-only", "--out", os.path.join(a.dump_dir, "_unused.json"),
            "--device", a.device, "--infer-seed", str(a.seed), "--seed", "0", "--arm", "refcv6",
            "--lru", "8"]
    t0 = time.time()
    if a.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    R.main(argv)
    man = json.load(open(os.path.join(a.dump_dir, "manifest.json"), encoding="utf-8"))
    from tanitad.data.v2_dataset import load_or_build_manifest
    clip_ids = [str(c) for c in load_or_build_manifest(a.episodes, verbose=False)["clip_id"]]
    meta = RR.save_extras(st, os.path.join(a.dump_dir, "refcv6_extras.npz"), clip_ids)
    rec = {"tool": "a6_roll.py", "amendment": "A6", "seed": a.seed, "dump_dir": a.dump_dir,
           "episodes": a.episodes, "labels": a.labels, "remap_overrides": o,
           "wall_s": round(time.time() - t0, 1), "n_windows": man["grid"]["n_windows"],
           "n_episodes": man["grid"]["n_episodes"], "skipped": man["grid"]["n_windows_skipped"],
           "extras": meta, "device": a.device,
           "cuda_max_memory_allocated_gib": (round(torch.cuda.max_memory_allocated() / 2**30, 3)
                                             if torch.cuda.is_initialized() else None),
           "state_dict": st["model_record"]["state_dict"], "window_restriction": st.get("window_restriction"),
           "frame_memo": (None if st.get("memo") is None else
                          {"hits": st["memo"].hits, "misses": st["memo"].misses})}
    json.dump(rec, open(a.out_json, "w", encoding="utf-8"), indent=1, default=str)
    del st
    gc.collect()
    print(f"[a6_roll] seed {a.seed}: {rec['n_windows']} windows / {rec['n_episodes']} eps -> {a.out_json}")


if __name__ == "__main__":
    main()
