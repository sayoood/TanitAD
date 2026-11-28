"""The LOCAL-VERIFY form of `pull_ckpt.py` for a checkpoint that is ALREADY on the dev box.

    python pull_record_local.py --step 50400 --full D:/refcv7_eval_kit/ckpt/ckpt_50400_full.pt \
        --model-only D:/refcv7_eval_kit/ckpt/ckpt_50400.pt \
        --metrics D:/refcv7_eval_kit/thor_reads/metrics_final_50400.jsonl \
        --out-json D:/refcv7_eval_kit/chain/pull_50400_OK.json

Written 2026-10-04 for the FINAL refcv7 checkpoint (step 50,400): the run finished
(`summary.json` done=true), so the rolling `ckpt.pt` on Thor IS the step-50,400 checkpoint and no
milestone file exists for that step. The Master Mind pulled it and extracted `{model, step}`. This tool
re-verifies, without copying anything again, every acceptance test `pull_ckpt.py` applies, and writes
the SAME record shape (`marker ZZPULLOK<N>ZZ`, `metrics_copy`, md5s) that `chain_milestone.sh` reads:

  1. Thor (read-only `ls` / `md5sum` only): `ckpt.pt` md5 BEFORE and AFTER the local checks, and its
     (size, mtime) unchanged across them; Thor's `metrics.jsonl` md5 == the local metrics copy's md5;
  2. local md5 of the FULL file == Thor's `ckpt.pt` md5 (the 3-way md5 of `pull_ckpt.py`);
  3. both local files carry `step == N`; the model-only file's `model` dict is `torch.equal` to the
     full file's `model` dict on EVERY tensor (same key set);
  4. the metrics copy carries exactly ONE eval row at step N and its last step is N.
Anything else -> ZZPULLFAIL<N>ZZ with the reason. Assert on the JSON, never on the exit code.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pull_ckpt as P  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--full", default=None,
                    help="the FULL rolling ckpt (final step); omit for a MILESTONE file, whose Thor "
                         "source is ckpt_<N>.pt and whose local copy IS --model-only")
    ap.add_argument("--model-only", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    N = int(a.step)
    final_rolling = a.full is not None
    remote = f"{P.RUN}/ckpt.pt" if final_rolling else f"{P.RUN}/ckpt_{N}.pt"
    if not final_rolling:
        a.full = a.model_only                 # a milestone file: the 3-way md5 is on the file itself
    rmetrics = f"{P.RUN}/metrics.jsonl"
    rec = {"tool": "pull_record_local.py", "step": N, "host": P.HOST, "run": P.RUN,
           "t_start": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "source_path": remote,
           "source_kind": ("FINAL rolling ckpt.pt (model+opt+step+data_pos), run DONE; model-only "
                           "{model, step} extracted locally (the battery's ckpt_<N>.pt)")
                          if final_rolling else
                          "milestone ckpt_<N>.pt (model+step), already on the dev box"}

    def done(marker, msg, code):
        rec.update(marker=marker, message=msg, t_end=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        json.dump(rec, open(a.out_json, "w", encoding="utf-8"), indent=1)
        print(f"{marker} {msg}", flush=True)
        return code

    s1 = P.stat_remote(remote)
    m_before = P.md5_remote(remote)
    mm_remote = P.md5_remote(rmetrics)
    rec.update(stat_1=s1, md5_remote_before=m_before, metrics_md5_remote=mm_remote)
    m_full = P.md5_local(a.full)
    m_model = P.md5_local(a.model_only)
    mm_local = P.md5_local(a.metrics)
    rec.update(md5_local_full=m_full, md5_local=m_model, metrics_md5_local=mm_local,
               size_local_full=os.path.getsize(a.full), size_local=os.path.getsize(a.model_only))
    import torch
    full = torch.load(a.full, map_location="cpu", weights_only=False, mmap=True)
    mo = torch.load(a.model_only, map_location="cpu", weights_only=False, mmap=True)
    rec["ckpt_step_key_full"] = int(full.get("step")) if full.get("step") is not None else None
    rec["ckpt_step_key"] = int(mo.get("step")) if mo.get("step") is not None else None
    rec["ckpt_keys_full"] = sorted(full.keys())
    rec["ckpt_keys"] = sorted(mo.keys())
    fk, mk = set(full["model"].keys()), set(mo["model"].keys())
    n_eq = sum(1 for k in fk & mk if torch.equal(full["model"][k], mo["model"][k]))
    rec["model_tensors"] = {"full": len(fk), "model_only": len(mk), "common": len(fk & mk),
                            "torch_equal": n_eq}
    del full, mo
    msta = P.metrics_state(a.metrics, N)
    rec["metrics_copy"] = a.metrics
    rec["metrics_state"] = msta
    s2 = P.stat_remote(remote)
    m_after = P.md5_remote(remote)
    rec.update(stat_2=s2, md5_remote_after=m_after, stat_after=s2)
    bad = []
    if not (len(m_before) == 32 and m_before == m_after == m_full):
        bad.append(f"3-way md5 before={m_before} after={m_after} local_full={m_full}")
    if s1 is None or s1 != s2 or s1[0] != rec["size_local_full"]:
        bad.append(f"Thor stat {s1} -> {s2} vs local size {rec['size_local_full']}")
    if not (len(mm_remote) == 32 and mm_remote == mm_local):
        bad.append(f"metrics md5 remote={mm_remote} local={mm_local}")
    if rec["ckpt_step_key"] != N or rec["ckpt_step_key_full"] != N:
        bad.append(f"step keys {rec['ckpt_step_key_full']} / {rec['ckpt_step_key']} != {N}")
    if not (fk == mk and n_eq == len(fk) and len(fk) > 0):
        bad.append(f"model tensors {rec['model_tensors']}")
    if not (msta["eval_row_at_step"] == 1 and msta["last_step"] >= N):
        bad.append(f"metrics state {msta}")
    if bad:
        return done(f"ZZPULLFAIL{N}ZZ", "; ".join(bad), 3)
    rec["local_path"] = a.model_only
    return done(f"ZZPULLOK{N}ZZ", f"md5 {m_model} (model-only; local {m_full} == Thor {os.path.basename(remote)}) step {N} "
                f"tensors {n_eq}/{len(fk)} equal metrics {os.path.basename(a.metrics)}", 0)


if __name__ == "__main__":
    sys.exit(main())
