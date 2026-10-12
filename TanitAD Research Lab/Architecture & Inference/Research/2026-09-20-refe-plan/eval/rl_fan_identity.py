#!/usr/bin/env python3
"""RL-S1 priority 1, the EVIDENCE: is `refe/rl_fan_dump.py`'s fan the fan the main run's step-10,075 sets carry?  (dev box; GPU; ~1-2 h)

Three questions, three different tools.  They are kept apart because they have different answers:

  A. CODE PATH + WEIGHTS -- same hardware.  `rl_fan_dump.py` (FULL `model_final.pt`) against the OLD dumper `onpolicy_dump.py` (PARTIAL `ckpt_last.pt` of
     step 10,075), run back to back on this GPU over the same keys.  `ckpt_last@10075` and `model_final.pt` hold the same 276 trainable tensors
     bit for bit (`raw/rl_pool/ckpt_last_vs_model_final.json`), so any difference here is the dumper, not the weights.  EXACT equality is required.
  B. HARDWARE -- cross-device.  This dev box (RTX 4060, Ada) against the banked sets the A40 (Ampere) wrote.  bf16 autocast + TF32 kernels are NOT
     guaranteed to agree across architectures, so this one is MEASURED, not assumed; if it is not exact the bit-identity gate for the production run is
     `rl_fan_dump.py --compare-sets` ON THE POD, before the 51,000-scene dump (exit 5 on any difference).
  C. INVARIANCE.  The same keys under batch size 1 / 2 / 4, shifted batch composition and prefetch threads must give the same bytes; otherwise the
     shard / batch layout of the production run would change the fan.

    python eval/rl_fan_identity.py --work <dir> --py <python with torch+cuda+cv2> [--keys N] [--skip-old]

`<work>` must hold (see HANDOFF_RL_S1.md, 2026-10-12 RL-S1-S1b section, for how the dev-box inputs were built):
  dev/bank/targets_rank0.jsonl   the bank rows of the proxy tokens (10,000)       bank_full/calib_table.json     the bank's rig table
  sets_10075_proxy.jsonl          the step-10,075 sets of those tokens, from onpolicy_sets_logs_full.tar.zst
Writes <work>/identity/ and prints one JSON; the banked copy is raw/rl_fan/identity_devbox.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PKG, "refe"))
import rl_fan_dump as RF  # noqa: E402

FINAL = "D:/Projects/TanitAD-artifacts/refe-final-2026-10-01"
PIXELS = "D:/Projects/TanitAD/data/refe_proxy/pixels"
MODEL_FINAL_SHA = "4246a320be71ad0e315f8f298b7dadab3dbbba2c24f4fc30706e0129f5057e66"      # refe-final-2026-10-01/MANIFEST.txt
STEP = 10075


def sha(p):
    return RF.sha256_bytes(p)


def run(cmd, log, env=None):
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONIOENCODING="utf-8", **(env or {})))
    return rc, round(time.time() - t0, 1)


def new_records(path):
    out = {}
    for ln in open(path, encoding="utf-8"):
        r = json.loads(ln)
        out[RF.key_of(r)] = (r["traj"], r["yaw"])
    return out


def old_props(qdir):
    """The OLD dumper's queue chunks -> {key: (traj [M][T][2], yaw [M][T])} in the set format."""
    import glob
    out = {}
    for fp in sorted(glob.glob(os.path.join(qdir, "props_r*_*.jsonl*"))):
        for ln in open(fp, encoding="utf-8"):
            r = json.loads(ln)
            k = (r["log_name"], r["token"], int(r["step"]), int(r["rank"]))
            out[k] = ([[[p[0], p[1]] for p in prop] for prop in r["props"]], [[p[2] for p in prop] for prop in r["props"]])
    return out


def diff(a, b, label):
    """Exact comparison of two {key: (traj, yaw)} maps over their common keys."""
    common = sorted(set(a) & set(b))
    unequal, worst = [], 0.0
    n_vals = n_bad = 0
    for k in common:
        (ta, ya), (tb, yb) = a[k], b[k]
        d = 0.0
        bad = 0
        for pa, pb in zip(ta, tb):
            for qa, qb in zip(pa, pb):
                for u, v in zip(qa, qb):
                    n_vals += 1
                    if u != v:
                        bad += 1
                        d = max(d, abs(u - v))
        for pa, pb in zip(ya, yb):
            for u, v in zip(pa, pb):
                n_vals += 1
                if u != v:
                    bad += 1
                    d = max(d, abs(u - v))
        n_bad += bad
        if bad:
            unequal.append((k, d))
            worst = max(worst, d)
    ds = sorted(d for _, d in unequal)
    return {"compare": label, "keys_in_a": len(a), "keys_in_b": len(b), "common": len(common), "bit_identical": len(common) - len(unequal), "not_identical": len(unequal),
            "values_compared": n_vals, "values_different": n_bad, "max_abs_delta": worst,
            "median_max_delta_over_unequal": (ds[len(ds) // 2] if ds else 0.0), "first_unequal": [[list(k), d] for k, d in unequal[:3]]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--py", required=True, help="python with torch+cuda+cv2 (the dev box: venvs/driverl-eval)")
    ap.add_argument("--keys", type=int, default=0, help="use only the first N keys (0 = every key that has both images and a step-10,075 set)")
    ap.add_argument("--skip-old", action="store_true")
    ap.add_argument("--old-keys", type=int, default=0, help="run the OLD dumper on only the first N keys (0 = all): the code-path comparison needs few, the GPU is shared")
    ap.add_argument("--inv-n", type=int, default=24, help="keys in the invariance subset")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    W = a.work.replace("\\", "/")
    idd = os.path.join(W, "identity")
    os.makedirs(idd, exist_ok=True)
    sets = [json.loads(l) for l in open(os.path.join(W, "sets_10075_proxy.jsonl"), encoding="utf-8")]
    pod = {}
    for r in sets:
        if r["rank"] == 0:
            pod[RF.key_of(r)] = (r["traj"], r["yaw"])
    keys = sorted({(k[0], k[1]) for k in pod})
    if a.keys:
        keys = keys[:a.keys]
    kp = os.path.join(idd, "keys.tsv")
    with open(kp, "w", encoding="utf-8", newline="\n") as f:
        f.write("token\tlog\n" + "".join(f"{t}\t{l}\n" for l, t in keys))
    # the bank restricted to those keys: the OLD dumper walks every row it is given
    bank = os.path.join(idd, "bank")
    os.makedirs(bank, exist_ok=True)
    kset = set(keys)
    n_rows = 0
    with open(os.path.join(bank, "targets_rank0.jsonl"), "w", encoding="utf-8", newline="\n") as g:
        for ln in open(os.path.join(W, "dev", "bank", "targets_rank0.jsonl"), encoding="utf-8"):
            r = json.loads(ln)
            if (r["log_name"], r["token"]) in kset:
                g.write(ln if ln.endswith("\n") else ln + "\n")
                n_rows += 1
    assert n_rows == len(keys), (n_rows, len(keys))
    res = {"tool": "rl_fan_identity/1", "date": time.strftime("%Y-%m-%d %H:%M:%S"), "keys": len(keys), "step": STEP,
           "inputs": {"model_final_sha256": MODEL_FINAL_SHA, "model_final_sha256_on_disk": sha(f"{FINAL}/model_final.pt"), "ckpt_last_sha256": sha(f"{FINAL}/ckpt_last.pt"),
                      "pod_sets_file_sha256": sha(os.path.join(W, "sets_10075_proxy.jsonl")), "keys_tsv_sha256": sha(kp)}}
    assert res["inputs"]["model_final_sha256_on_disk"] == MODEL_FINAL_SHA, "model_final.pt on disk is not the registry's file"
    dumper = os.path.join(PKG, "refe", "rl_fan_dump.py")
    common = ["--ckpt", f"{FINAL}/model_final.pt", "--expect-ckpt-sha256", MODEL_FINAL_SHA, "--targets", bank, "--images", PIXELS,
              "--calib", os.path.join(W, "bank_full", "calib_table.json")]

    def new_run(tag, keys_path, extra=()):
        od = os.path.join(idd, f"new_{tag}")
        rc, sec = run([a.py, dumper, *common, "--keys", keys_path, "--out", od, "--tag", tag, *extra], os.path.join(idd, f"new_{tag}.log"))
        man = json.load(open(os.path.join(od, f"rl_fan_dump_{tag}_s0of1.manifest.json"))) if rc == 0 else {}
        return rc, sec, od, man

    # ---------------------------------------------------------------- A/B: the full key set, the old dumper's own settings (batch 4, serial decode)
    rc, sec, od, man = new_run("full", kp)
    res["new_full"] = {"rc": rc, "wall_sec": sec, "s_per_set_total": man.get("s_per_set_total"), "s_per_set_forward": man.get("s_per_set_forward"),
                       "peak_gpu_gb": man.get("peak_gpu_gb"), "gpu": man.get("gpu"), "torch": man.get("torch"), "startup_sec": man.get("startup_sec")}
    assert rc == 0, "the new dumper failed; see new_full.log"
    new = new_records(os.path.join(od, "onpolicy_rlfan_full_s0of1.jsonl"))
    res["B_new_devbox_vs_pod_sets_A40"] = diff(new, pod, "rl_fan_dump (RTX 4060) vs the banked step-10075 sets (A40)")
    cs = RF.compare_sets(os.path.join(od, "onpolicy_rlfan_full_s0of1.jsonl"), [os.path.join(W, "sets_10075_proxy.jsonl")], STEP)
    res["B_compare_sets_gate"] = cs
    if not a.skip_old:
        q = os.path.join(idd, "queue_old")
        os.makedirs(q, exist_ok=True)
        bank_old = bank
        if a.old_keys:                                                   # the old dumper walks every row of the bank it is given
            bank_old = os.path.join(idd, "bank_old")
            os.makedirs(bank_old, exist_ok=True)
            first = set(keys[:a.old_keys])
            with open(os.path.join(bank_old, "targets_rank0.jsonl"), "w", encoding="utf-8", newline="\n") as g:
                for ln in open(os.path.join(bank, "targets_rank0.jsonl"), encoding="utf-8"):
                    r = json.loads(ln)
                    if (r["log_name"], r["token"]) in first:
                        g.write(ln if ln.endswith("\n") else ln + "\n")
        rc, sec = run([a.py, os.path.join(PKG, "refe", "onpolicy_dump.py"), "--run-dir", FINAL, "--targets", bank_old, "--images", PIXELS, "--calib", os.path.join(W, "bank_full", "calib_table.json"),
                       "--queue", q, "--ranks", "0", "--max-pending", "1000000", "--chunk", "32", "--batch", "4"], os.path.join(idd, "old_dump.log"))
        old = old_props(q)
        res["old_dumper"] = {"rc": rc, "wall_sec": sec, "sets": len(old)}
        res["A_new_vs_old_dumper_same_gpu"] = diff(new, old, "rl_fan_dump (model_final.pt) vs onpolicy_dump (ckpt_last.pt@10075), same GPU")
        res["B2_old_dumper_devbox_vs_pod_sets_A40"] = diff(old, pod, "onpolicy_dump (RTX 4060) vs the banked step-10075 sets (A40)")
    # ---------------------------------------------------------------- C: invariance on a small subset
    sub = keys[:a.inv_n]
    sp = os.path.join(idd, "keys_sub.tsv")
    sp2 = os.path.join(idd, "keys_sub_shift.tsv")
    for path, ks in ((sp, sub), (sp2, sub[1:])):
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write("token\tlog\n" + "".join(f"{t}\t{l}\n" for l, t in ks))
    inv = {}
    for tag, kpth, extra in (("b1", sp, ("--batch", "1")), ("b2", sp, ("--batch", "2")), ("b4", sp, ("--batch", "4")), ("b4shift", sp2, ("--batch", "4")), ("b4w2", sp, ("--batch", "4", "--workers", "2"))):
        rc, sec, od2, m2 = new_run(tag, kpth, extra)
        if rc != 0:
            inv[tag] = {"rc": rc}
            continue
        inv[tag] = new_records(os.path.join(od2, f"onpolicy_rlfan_{tag}_s0of1.jsonl"))
    ref = inv.get("b4")
    res["C_invariance"] = {t: diff(v, ref, f"{t} vs b4") for t, v in inv.items() if t != "b4" and isinstance(v, dict) and "rc" not in v and ref is not None}
    # the subset at batch 4 must also equal the same keys inside the full run (batch composition differs: 24 keys vs 1,365)
    if ref is not None:
        res["C_invariance"]["b4_subset_vs_full_run"] = diff(ref, new, "subset(batch 4) vs the same keys inside the full run")
    out = a.out or os.path.join(idd, "identity_devbox.json")
    json.dump(res, open(out, "w", encoding="utf-8", newline="\n"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "inputs"}, indent=1)[:6000])
    ok = (res.get("A_new_vs_old_dumper_same_gpu", {}).get("not_identical", 0) == 0) and all(v["not_identical"] == 0 for v in res["C_invariance"].values())
    print("ZZFANID_A_AND_C_EXACT" if ok else "ZZFANID_A_OR_C_NOT_EXACT")
    return 0 if ok else 5


if __name__ == "__main__":
    sys.exit(main())
