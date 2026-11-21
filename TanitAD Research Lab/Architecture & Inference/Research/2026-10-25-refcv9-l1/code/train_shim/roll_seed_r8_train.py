"""roll_seed_r8_train.py -- ONE inference-seed os-only roll of refcv8 over the A6 TRAIN clips (139, sha12-rule sample), in its own process.

NEW file (PKG/raw/retime_trainfit). The eval roll (`roll_seed_r8.py` -> `roll_seed_r7.main` -> `run_battery_r7.roll_one`) hard-wires
`--episodes KIT_EVAL --labels KIT_LABELS` and takes its window list from the banked refcv4b EVAL grid; none of that exists for train clips.
This script is `run_battery_r7.roll_one` (os-only branch, SPEC A3) with exactly four differences, all recorded in the roll json:
  1. the loader's REMAP OVERRIDES file is `refcv8_train_remap_devbox.json` (the eval remap + `--eval-cache / --eval-labels /
     --speed-max-sidecar-v6-eval / --r8-v9-labels-eval` pointed at the TRAIN counterparts), set on `r8_repoint.REMAP_OVERRIDES` before apply();
  2. `args.r8_v9_eval_md5` is set to the TRAIN release md5 (the loader checks the v9 npz against it; the argv carries the EVAL md5);
  3. the pose clock is the STANDALONE one (R8_NO_POSE_SYNC=1), the clock the eval trees step30000 and step30000ema _unc were rolled on;
  4. the window rule: the eval dataset's index restricted to t % 5 == 0 (ws = t + W - 1), the stride of the eval S2 grid (SPEC A6 precedent);
     the dump is later cut to S6 by the battery's own `build_s6_dump` (windows with a valid 60-step future).
    python roll_seed_r8_train.py --ckpt E:/refcv8_eval_kit/ckpt/ckpt_ema_30000.pt --config E:/refcv8_eval_kit/ckpt/config.json --seed 0 \
        --dump-dir <dir> --out-json <json> [--device cuda] [--max-clips N --max-windows-per-clip M --smoke-cpu-fp32-trunk]   # last three = SMOKE ONLY
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ["R8_NO_POSE_SYNC"] = "1"                      # the standalone clock (see 3.)
import r8_repoint as RP  # noqa: E402

TRAIN_REMAP = HERE / "refcv8_train_remap_devbox.json"
TRAIN_V9_MD5 = "83e30bd3791f0767707eade5f4671759"        # == the refcv8 argv's --r8-v9-md5 (the TRAIN release), Thor-verified 2026-10-19
STRIDE = 5
RP.REMAP_OVERRIDES = TRAIN_REMAP                          # BEFORE apply(): configure_env() sets REFCV7_REMAP_OVERRIDES from it
L = RP.apply()
_orig_bed = L.build_eval_dataset


def _bed(model, cfg, args, config, **kw):                 # 2.
    if os.path.basename(str(getattr(args, "r8_v9_labels_eval", ""))) == "v9_labels_train.npz":
        args.r8_v9_eval_md5 = TRAIN_V9_MD5
    return _orig_bed(model, cfg, args, config, **kw)


_bed._r8_patched = True
_bed._r8_orig = getattr(_orig_bed, "_r8_orig", _orig_bed)
L.build_eval_dataset = _bed
import torch  # noqa: E402
import r7_roll as RR  # noqa: E402  (battery code dir, on sys.path via r8_repoint.configure_env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-clips", type=int, default=0, help="SMOKE ONLY")
    ap.add_argument("--max-windows-per-clip", type=int, default=0, help="SMOKE ONLY")
    ap.add_argument("--smoke-cpu-fp32-trunk", action="store_true", help="SMOKE ONLY")
    a = ap.parse_args()
    ovr = json.load(open(os.environ["REFCV7_REMAP_OVERRIDES"], encoding="utf-8"))
    if os.path.normcase(os.path.abspath(os.environ["REFCV7_REMAP_OVERRIDES"])) != os.path.normcase(str(TRAIN_REMAP)):
        raise SystemExit("[roll_seed_r8_train] the loader is not using the TRAIN remap file -- refusing (it would roll the EVAL cache)")
    eps, lab = ovr["--eval-cache"], ovr["--eval-labels"]
    if "eval139" in eps or "_eval." in os.path.basename(lab):
        raise SystemExit("[roll_seed_r8_train] the remap names an EVAL cache/labels -- refusing")
    if a.smoke_cpu_fp32_trunk:
        _o = L.build_model

        def _fp32(*aa, **kk):
            m, c, ar, r = _o(*aa, **kk)
            for mod in m.modules():
                lv = getattr(mod, "memory_levers", None)
                if isinstance(lv, dict) and "bf16" in lv:
                    lv["bf16"] = False
                    lv["channels_last"] = False
            r["smoke_note"] = "trunk bf16/NHWC OFF (--smoke-cpu-fp32-trunk)"
            return m, c, ar, r
        L.build_model = _fp32
    st = RR.install_patches(config_path=a.config, windows=None, frame_memo=True, obedience=False, vmaxzero=False, with_navflip=False, filteroff=False)
    R = RR.ra3()
    orig = R.build_corpus

    def build_corpus(aa, cfg, prov):
        res = list(orig(aa, cfg, prov))
        ds = res[3]
        before = len(ds.index)
        keep = [tuple(x) for x in ds.index if int(x[1]) % STRIDE == 0]
        if a.max_clips:
            eps_keep = set(sorted({e for e, _ in keep})[: a.max_clips])
            keep = [x for x in keep if x[0] in eps_keep]
        if a.max_windows_per_clip:
            cnt, kk = {}, []
            for x in keep:
                cnt[x[0]] = cnt.get(x[0], 0) + 1
                if cnt[x[0]] <= a.max_windows_per_clip:
                    kk.append(x)
            keep = kk
        ds.index = keep
        wr = {"rule": f"the eval dataset's V3Dataset index restricted to t % {STRIDE} == 0 (ws = t + W - 1), TRAIN clips (SPEC A6 precedent)",
              "n_index_before": before, "n_index_after": len(ds.index), "n_clips_after": len({e for e, _ in ds.index}),
              "smoke_max_clips": a.max_clips or None, "smoke_max_windows_per_clip": a.max_windows_per_clip or None}
        res[5]["window_restriction"] = wr
        st["window_restriction"] = wr
        return tuple(res)

    R.build_corpus = build_corpus
    os.makedirs(a.dump_dir, exist_ok=True)
    argv = ["--ckpt", a.ckpt, "--config", a.config, "--episodes", eps, "--labels", lab, "--nav-source", "v72", "--grid", "2s", "--action-units", "steer",
            "--window-stride", "1", "--with-oracle-sel", "--dump-dir", a.dump_dir, "--dump-only", "--out", os.path.join(a.dump_dir, "_unused.json"),
            "--device", a.device, "--infer-seed", str(a.seed), "--seed", "0", "--arm", "refcv7", "--lru", "8", "--no-navshuf", "--no-navzero"]
    t0 = time.time()
    if a.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    R.main(argv)
    man = json.load(open(os.path.join(a.dump_dir, "manifest.json"), encoding="utf-8"))
    from tanitad.data.v2_dataset import load_or_build_manifest
    clip_ids = [str(c) for c in load_or_build_manifest(eps, verbose=False)["clip_id"]]
    meta = RR.save_extras(st, os.path.join(a.dump_dir, "refcv6_extras.npz"), clip_ids)
    mr = st["model_record"]
    rec = {"tool": "roll_seed_r8_train.py", "seed": a.seed, "arm_set": "os + model-free controls (SPEC A3)", "dump_dir": a.dump_dir, "episodes": eps, "labels": lab,
           "remap_overrides": ovr, "pose_clock": "STANDALONE (R8_NO_POSE_SYNC=1) = the clock of the eval trees step30000_unc and step30000ema_unc",
           "v9_eval_md5_set_to": TRAIN_V9_MD5, "wall_s": round(time.time() - t0, 1), "n_windows": man["grid"]["n_windows"], "n_episodes": man["grid"]["n_episodes"],
           "skipped": man["grid"]["n_windows_skipped"], "extras": meta, "device": a.device,
           "cuda_max_memory_allocated_gib": (round(torch.cuda.max_memory_allocated() / 2**30, 3) if torch.cuda.is_initialized() else None),
           "model_record": {k: mr.get(k) for k in ("state_dict", "param_breakdown", "anchor_file_vs_ckpt_buffers", "declared_vs_built")},
           "window_restriction": st.get("window_restriction"), "label_clock": (st.get("dataset_record") or {}).get("label_clock"),
           "dataset_record_keys": sorted((st.get("dataset_record") or {}).keys()),
           "frame_memo": (None if st.get("memo") is None else {"hits": st["memo"].hits, "misses": st["memo"].misses}),
           "smoke": bool(a.max_clips or a.max_windows_per_clip)}
    json.dump(rec, open(a.out_json, "w", encoding="utf-8"), indent=1, default=str)
    del st
    gc.collect()
    print(f"[roll_seed_r8_train] seed {a.seed}: {rec['n_windows']} windows / {rec['n_episodes']} eps -> {a.out_json}  ZZRTTRAINROLLOKZZ", flush=True)


if __name__ == "__main__":
    main()
