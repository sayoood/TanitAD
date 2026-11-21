#!/bin/sh
# READ-ONLY Thor probe: is everything STEP 1 needs already on the box, and is the
# live S-W run healthy? Zero GPU, zero writes, one `ssh -n`.
#
# ⚠️ `ssh -n` is MANDATORY — a nested ssh inside a pipe/heredoc EATS THE REST OF
#    THE SCRIPT'S STDIN and the tail silently never runs (CLAUDE.md).
# ⚠️ The remote emits ONE opaque `ZZ…ZZ`-framed JSON blob computed POD-SIDE, and
#    nothing here greps the raw stream for a token this command contains — that
#    is the PTY-echo trap, which has invented a false failure three times.
#
#   sh code/thor_readiness_probe.sh > raw/thor_readiness.json
set -e
ssh -n -o ConnectTimeout=15 -o BatchMode=yes tanitad-thor-wifi 'python3 - <<PYEOF
import json, os, subprocess, glob
R = "/home/nvidia"
cfg = json.load(open(R + "/experiments/v6F-SW-30k/config.json"))["args"]
tail = subprocess.run(["tail", "-n", "1",
                       R + "/experiments/v6F-SW-30k/train_log.jsonl"],
                      capture_output=True, text=True).stdout or "{}"
row = json.loads(tail)
ps = subprocess.run(["ps", "-o", "etime=,rss=,stat=", "-p", "25477"],
                    capture_output=True, text=True).stdout.split()
snaps = sorted(os.path.basename(p) for p in glob.glob(R + "/ckpt_snaps/*.pt"))
val = cfg.get("v2_val_cache") or [cfg.get("val_cache")]
out = {
  "live_run": {"pid": 25477, "alive": bool(ps), "etime": ps[0] if ps else None,
               "rss_kb": int(ps[1]) if len(ps) > 1 else None,
               "stat": ps[2] if len(ps) > 2 else None,
               "step": row.get("step"), "of": cfg.get("steps"),
               "step_s": row.get("step_s"), "loss": row.get("loss")},
  "run_args_step1_needs": {k: cfg.get(k) for k in
      ("stage", "window", "max_horizon", "v2_subframe", "frame_h", "frame_w",
       "in_channels", "v2_lru", "batch", "selector", "n_candidates",
       "readout_grid", "readout_dim")},
  "val_corpus_in_the_run_args": val,
  "val_corpus_exists": [bool(os.path.isdir(v)) for v in val if v],
  "config_json_beside_ckpt": os.path.isfile(R + "/experiments/v6F-SW-30k/config.json"),
  "ckpt_pt_present": os.path.isfile(R + "/experiments/v6F-SW-30k/ckpt.pt"),
  "ckpt_snaps": snaps,
  "ckpt_snaps_bytes": {os.path.basename(p): os.path.getsize(p)
                       for p in sorted(glob.glob(R + "/ckpt_snaps/*.pt"))},
  "dumper_shipped_to_thor": os.path.isfile(
      R + "/TanitAD/stack/scripts/v6_dump_sw_latents.py"),
}
print("ZZ" + json.dumps(out) + "ZZ")
PYEOF' | sed -n 's/^ZZ\(.*\)ZZ$/\1/p' | python -m json.tool
