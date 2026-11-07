"""Check a REAL L (LoRA arm) checkpoint against the format the LC fold expects, and print its merged q/v digest. Nothing is fetched, written or modified.

    python -I check_l_ckpt.py <ckpt_last.pt> [--expect-sha256 HEX] [--expect-merged-sha256 HEX] [--snapshot DIR] [--no-fold] [--json OUT.json]

The first L checkpoint banks to the PRIVATE HF dataset `Sayood/tanitad-lorapod-ckpts` at step 1,000 (`bank_ckpts.py`); the Master Mind fetches it (HF_HUB_DISABLE_XET=1, token read in place) and runs this on it.
What it verifies (each line of the printout is PASS / FAIL / WARN with the reason):
  * the file's sha256 (and `--expect-sha256`, e.g. the line of `/root/logs/ckpt_bank_ledger.jsonl`);
  * the blob keys the L trainer's `save()` writes (`v8f_mini_train.py`): `model` (the heads, B's layout: NO `lora_*`/`backbone.*` key), `cfg` (kin_mode yaw: `V8FMiniT2Policy` refuses a steer checkpoint), `lora`, `encoder_levers`, `step`;
  * `encoder_levers`: encoder lora, r, alpha, targets, 24 blocks, 3,145,728 trainable, the cache builder's weights sha256 -- and equal to `lora.meta`;
  * the `lora` entry, STRICTLY (t2v8f_lora_merge.parse_lora): exactly 96 fp32 finite tensors named `...layer.<i>.attention.<q|v>_proj.lora_<A|B>`, A [r, 1024], B [1024, r];
  * B is NOT all zero when step > 0 (an all-zero B is the identity: the encoder never moved, so "LC" would be the frozen arm under another name);
  * unless `--no-fold`: the local DINOv3-L safetensors is the builder's file, the LoRA folds into it on THIS machine's CPU, and the merged q/v digest (one value + the first tensors) is printed -- and equals
    `--expect-merged-sha256` when given (run it on box 2 against the digest in MANIFEST5 to see the cross-CPU question answered before a launch).
rc 0 = every check PASS (WARN allowed), 1 = a check FAILED, 2 = the file could not be read as a blob.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, HERE)
_SIBLING = os.path.join(os.path.dirname(HERE), "code")             # dev layout only (code_lc/ beside code/); on box 2 t2a_v8f_encoder.py sits in pkg/code beside this file
if not os.path.isfile(f"{HERE}/t2a_v8f_encoder.py") and os.path.isdir(_SIBLING) and _SIBLING not in sys.path:
    sys.path.append(_SIBLING)
import t2v8f_lora_merge as LM  # noqa: E402


def run(path, expect_sha=None, expect_merged=None, snapshot=None, fold=True) -> dict:
    rep = {"file": path, "checks": [], "verdict": None}
    rows = rep["checks"]

    def add(name, status, detail=""):
        rows.append({"check": name, "status": status, "detail": detail})
    if not os.path.isfile(path):
        add("file", "FAIL", "not a file")
        rep["verdict"] = "REFUSED"
        return rep
    rep["bytes"], rep["sha256"] = os.path.getsize(path), LM.file_sha256(path)
    add("file_sha256", "PASS" if expect_sha in (None, rep["sha256"]) else "FAIL", rep["sha256"] if expect_sha in (None, rep["sha256"]) else f"{rep['sha256']} != expected {expect_sha}")
    try:
        ck = LM.load_blob(path)
    except Exception as e:                                                                              # noqa: BLE001
        add("torch.load", "FAIL", repr(e)[:300])
        rep["verdict"] = "REFUSED"
        return rep
    rep["keys"] = sorted(ck)
    need = ("model", "cfg", "lora", "encoder_levers", "step")
    miss = [k for k in need if k not in ck]
    add("blob_keys", "PASS" if not miss else "FAIL", f"{sorted(ck)}" if not miss else f"missing {miss}: this is not an L trainer blob (a B/M2b checkpoint has no `lora`)")
    if miss:
        rep["verdict"] = "FAIL"
        return rep
    rep["step"], rep["seed"], rep["arm"] = ck.get("step"), ck.get("seed"), ck.get("arm")
    heads = ck["model"]
    stray = [k for k in heads if "lora_" in k or k.startswith("backbone.")]
    add("heads_layout", "PASS" if not stray else "FAIL", f"{len(heads)} head tensors, none LoRA/backbone" if not stray else f"`model` holds encoder keys {stray[:3]}: V8FMiniT2Policy loads it STRICT")
    kin = (ck["cfg"] or {}).get("kin_mode")
    add("cfg_kin_mode", "PASS" if kin == "yaw" else "FAIL", f"kin_mode={kin!r} (V8FMiniT2Policy refuses anything but 'yaw')")
    lv = ck["encoder_levers"] or {}
    want = {"encoder": "lora", "lora_r": 32, "lora_alpha": 32.0, "lora_targets": ["q", "v"], "lora_blocks": LM.N_BLOCKS, "lora_trainable": 3_145_728, "weights_sha256": LM.BASE_WEIGHTS_SHA256}
    off = {k: (lv.get(k), v) for k, v in want.items() if lv.get(k) != v}
    add("encoder_levers", "PASS" if not off else "FAIL", "lora r32 alpha32 q,v 24 blocks 3,145,728 trainable, builder weights sha" if not off else f"(got, want): {off}")
    try:
        mt = ck["lora"]["meta"]
        parsed = LM.parse_lora(ck["lora"], r=int(mt["r"]), alpha=float(mt["alpha"]), targets=tuple(mt["targets"]), blocks=int(mt["blocks"]))
        add("lora_entry", "PASS", f"{2 * len(parsed)} tensors, meta {mt}")
    except (ValueError, KeyError, TypeError) as e:
        add("lora_entry", "FAIL", str(e)[:400])
        rep["verdict"] = "FAIL"
        return rep
    meta_ne = {k: (mt.get(k), lv.get("lora_" + k)) for k in ("r", "alpha", "targets", "blocks") if mt.get(k) != lv.get("lora_" + k)}
    add("meta_vs_levers", "PASS" if not meta_ne else "FAIL", "equal" if not meta_ne else f"(meta, levers): {meta_ne}")
    bmax = float(max(float(B.abs().max()) for _, B in parsed.values()))
    nz = int(sum(1 for _, B in parsed.values() if float(B.abs().sum()) == 0.0))
    rep["b_abs_max"], rep["n_b_zero_modules"] = bmax, nz
    step = int(ck["step"])
    if step > 0 and nz == len(parsed):
        add("b_moved", "FAIL", f"step {step} but every B is exactly zero: the encoder never moved (this would be the frozen arm under another name)")
    elif step > 0 and nz:
        add("b_moved", "WARN", f"{nz} of {len(parsed)} modules still have B == 0 at step {step}; max|B| {bmax:.3e}")
    else:
        add("b_moved", "PASS", f"step {step}, max|B| {bmax:.3e}, zero modules {nz}")
    rep["lora_tensors_sha256"] = LM.lora_tensors_sha256(parsed)
    if fold:
        try:
            import t2a_v8f_encoder as E
            snap = snapshot or E.find_local_snapshot()
            got = LM.snapshot_safetensors_sha256(snap)
            add("base_weights", "PASS" if got == LM.BASE_WEIGHTS_SHA256 else "FAIL", f"{snap} sha256 {got}")
            if got == LM.BASE_WEIGHTS_SHA256:
                import copy
                m = copy.deepcopy(LM.load_base_model(snap))
                LM.fold_lora(m, parsed, r=int(mt["r"]), alpha=float(mt["alpha"]))
                d = LM.model_digest(m)
                rep["merged_qv_sha256"], rep["merged_qv_tensor_sha256_first"] = d["merged_qv_sha256"], dict(sorted(d["tensor_sha256"].items())[:4])
                add("merged_digest", "PASS" if expect_merged in (None, d["merged_qv_sha256"]) else "FAIL",
                    d["merged_qv_sha256"] if expect_merged in (None, d["merged_qv_sha256"]) else f"{d['merged_qv_sha256']} != expected {expect_merged}")
        except Exception as e:                                                                          # noqa: BLE001
            add("fold", "FAIL", repr(e)[:400])
    rep["verdict"] = "FAIL" if any(r["status"] == "FAIL" for r in rows) else "PASS"
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--expect-sha256", default=None)
    ap.add_argument("--expect-merged-sha256", default=None)
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--no-fold", action="store_true", help="format checks only (no DINOv3-L needed)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    rep = run(a.ckpt, a.expect_sha256, a.expect_merged_sha256, a.snapshot, not a.no_fold)
    for r in rep["checks"]:
        print(f"{r['status']:5s} {r['check']:16s} {r['detail']}")
    print(f"VERDICT {rep['verdict']}  merged_qv_sha256={rep.get('merged_qv_sha256')}")
    if a.json:
        with open(a.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(rep, f, indent=1, default=str)
    return {"PASS": 0, "FAIL": 1, "REFUSED": 2}[rep["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
