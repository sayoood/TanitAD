#!/usr/bin/env python3
"""S0-L step 3 (README_S0L.md section 2): the L_s0-MERGED eval-141 cache, built by the B-prime cache's OWN pipeline at folded weights.

Route: /root/identity_pod/cache_pod.py (the B' cache builder, imported UNEDITED) -> pull_encode_pooled.encode_clip -> dinov3_fp8_encode_ship.encode_episode
(bf16 weights, batches of 8, pool 4x10 / 8x20 in fp32, fp8 e4m3fn). The ONLY change: pull_encode_pooled.build_model returns the builder's own model with
the state_dict of tanitad.models.v8f_lora_encoder.DinoV3LoRAEncoder.merge() loaded (strict), i.e. W + (alpha/r) B A (fp32 sum rounded to bf16) on the
q and v projections of the 24 blocks, the LoRA tensors being L_s0's checkpoint['lora'].

Modes:
  --mode zero    control (README step 3): every lora_B set to 0 before the fold. The merged weights must equal the frozen ones bit for bit (c1) and the
                 encoded clips must equal /root/data/cache_pod/eval bit for bit (c3). Run with --only on a few clips.
  --mode merged  the S0-L cache: all 141 eval clips (/root/ops/clips141.txt) into --out.
Checks (written to <out>/lmerge_checks.json, printed):
  c1 tensors whose merged value differs from the frozen backbone: merged -> exactly the q,v weights of every block (48 if q_proj/v_proj are separate,
     24 if fused); zero -> 0.
  c2 fold identity (merged only): on 8 cache frames of the first clip, the UNMERGED LoRA encoder (the trained module, adapter branch in fp32) vs the
     merged builder model, pooled 4x10 fp32 before fp8 -> rel-L2 and max-abs (bf16 rounding of the fold is the only expected difference), and the
     LoRA effect size for scale: merged vs frozen on the same frames.
  c3 (zero only) each encoded clip vs cache_pod/eval: pool4x10 / pool8x20 bitwise equal, rel-L2 (README tau_route 1e-3).
meta.json of <out>/eval gets 'lora_merge' (ckpt path + sha256, LoRA meta, merged-weights sha256, mode) and pod_stage names S0-L.
Tier: none (an input cache; no metric). No address, port or pod id lives in this file.
"""
import argparse, hashlib, importlib.util, json, sys, time
from pathlib import Path

import torch

CACHE_POD = "/root/identity_pod/cache_pod.py"
CKPT = "/root/runs/lora/L_s0/ckpt_last.pt"
CLIPS = "/root/ops/clips141.txt"
REF_EVAL = Path("/root/data/cache_pod/eval/clips")
sys.path.insert(0, "/root/tad/stack")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def sd_sha256(sd):
    h = hashlib.sha256()
    for k in sorted(sd):
        t = sd[k].detach().contiguous().cpu()
        h.update(k.encode()); h.update(str(t.dtype).encode()); h.update(str(tuple(t.shape)).encode())
        h.update(t.view(torch.uint8).numpy().tobytes() if t.dtype == torch.bfloat16 else t.numpy().tobytes())
    return h.hexdigest()


def log(m):
    print(f"[lmerge {time.strftime('%F %T')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("zero", "merged", "lora-zero", "lora"), required=True,
                    help="zero/merged = folded weights (W + B A rounded to bf16); lora-zero/lora = the TRAINED encoder module, adapters unmerged (fp32 branch, "
                         "added in bf16), exactly the forward the heads were trained on. MM 2026-11-15: c2 of the merged run measured the fold at rel-L2 0.050 "
                         "against the trained path, so the cache of record is 'lora' (decided before any S0-L read; the folded cache is kept beside).")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    out = Path(a.out)
    if (out / "eval" / "manifest.jsonl").exists():
        sys.exit(f"REFUSED: {out}/eval/manifest.jsonl exists (never overwrite; resume is not offered here)")
    clips = a.only if a.only else [ln.strip() for ln in open(CLIPS, encoding="utf-8") if ln.strip()]
    assert len(clips) == (len(a.only) if a.only else 141), len(clips)
    from tanitad.models import v8f_lora_encoder as LE

    ck_sha = sha256_file(CKPT)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    lsd = ck["lora"]
    del ck
    n_b = 0
    lora_route = a.mode.startswith("lora")
    if a.mode in ("zero", "lora-zero"):
        lsd = {"meta": lsd["meta"], "tensors": {k: (torch.zeros_like(v) if "lora_B" in k else v) for k, v in lsd["tensors"].items()}}
        n_b = sum("lora_B" in k for k in lsd["tensors"])
    m = lsd["meta"]
    log(f"ckpt {ck_sha[:12]} lora meta r {m['r']} alpha {m['alpha']} targets {m['targets']} blocks {m['blocks']} tensors {len(lsd['tensors'])} mode {a.mode} (B zeroed: {n_b})")
    enc = LE.DinoV3LoRAEncoder.from_pretrained(r=int(m["r"]), alpha=float(m["alpha"]), targets=tuple(m["targets"]), dropout=0.0, seed=0, grid="4x10",
                                               quantize=True, device="cuda", dtype=torch.bfloat16)
    enc.load_lora_state_dict(lsd)
    enc.eval()
    merged = enc.merge()
    msd = {k: v.detach() for k, v in merged.state_dict().items()}
    mw_sha = sd_sha256(msd)

    spec = importlib.util.spec_from_file_location("cache_pod", CACHE_POD)
    cp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cp)
    cp.BOX2_PULLED = Path("/nonexistent_no_box2_compare")          # L tokens are never compared with box 2's frozen tokens
    checks = {"mode": a.mode, "ckpt": CKPT, "ckpt_sha256": ck_sha, "lora_meta": m, "merged_weights_sha256": mw_sha}
    orig_load = cp.load_builder

    def load_builder():
        pep, ship = orig_load()
        orig_build = pep.build_model

        def build_model(device, dtype):
            model, mean, std, n_special = orig_build(device, dtype)
            if lora_route:
                # the trained module itself: its frozen base weights must be the builder's (bit for bit), then it replaces the builder's model
                bsd = model.state_dict()
                esd = {k.replace(".base.", "."): v for k, v in enc.backbone.state_dict().items() if "lora_" not in k}
                same = set(bsd) == set(esd) and all(torch.equal(bsd[k], esd[k].to(bsd[k].device)) for k in bsd)
                checks["c1"] = {"route": "lora (unmerged)", "base_weights_equal_builder": same, "n_lora_tensors": len(lsd["tensors"])}
                log(f"c1 lora route: trained module's base weights == builder's: {same}")
                assert same, "the LoRA module's frozen weights differ from the builder's model"
                del model
                enc.backbone.eval()
                if a.mode == "lora":
                    led = cp.ledger_last()[clips[0]]
                    o, _ = cp.prepare((clips[0], led["kind"], led))
                    fr = LE.decode_cache_frames(o, list(range(8))).to(device)
                    import copy as _copy
                    e32 = _copy.deepcopy(enc).float()
                    with torch.no_grad():
                        q_l32 = e32.pooled(fr, quantize=False).double()
                        q_m32 = e32.merged_encoder().pooled(fr, quantize=False).double()
                        checks["c2_ref"] = {"first8_pooled_fp8": enc.pooled(fr, quantize=True).float().cpu()}
                    checks["c4_fp32_fold"] = {"rel_l2_merged_vs_lora_fp32": float((q_m32 - q_l32).norm() / q_l32.norm()), "max_abs": float((q_m32 - q_l32).abs().max())}
                    log(f"c4 fp32 fold identity {checks['c4_fp32_fold']}")
                    del e32
                return enc.backbone, mean, std, n_special
            base = {k: v.detach().clone() for k, v in model.state_dict().items()}
            assert set(base) == set(msd), "the builder's model and the merged backbone have different state_dict keys"
            model.load_state_dict(msd, strict=True)
            now = model.state_dict()
            changed = sorted(k for k in base if not torch.equal(base[k], now[k]))
            same_as_merged = all(torch.equal(now[k], msd[k].to(now[k].device)) for k in now)
            checks["c1"] = {"n_changed": len(changed), "changed_first": changed[:4], "loaded_equals_merged": same_as_merged,
                            "all_changed_are_q_or_v": all(any(t in k for t in ("q_proj", "v_proj", "qkv", "query", "value")) for k in changed)}
            log(f"c1 changed tensors {len(changed)} (first {changed[:2]}); loaded == merged {same_as_merged}")
            if a.mode == "zero":
                assert len(changed) == 0, "B = 0 fold changed weights"
            else:
                assert len(changed) in (24, 48) and checks["c1"]["all_changed_are_q_or_v"], f"unexpected changed set {changed[:6]}"
                # c2 fold identity on 8 cache frames of the first clip
                led = cp.ledger_last()[clips[0]]
                o, _ = cp.prepare((clips[0], led["kind"], led))
                fr = LE.decode_cache_frames(o, list(range(8))).to(device)
                frozen = LE.frozen_plain_encoder(device=device, dtype=torch.bfloat16, grid="4x10", quantize=False)
                with torch.no_grad():
                    p_lora = enc.pooled(fr, quantize=False).double()
                    p_merg = enc.merged_encoder().pooled(fr, quantize=False).double()
                    p_froz = frozen.pooled(fr, quantize=False).double()
                rel = lambda x, y: float((x - y).norm() / y.norm())
                checks["c2"] = {"clip": clips[0], "frames": 8, "rel_l2_merged_vs_lora": rel(p_merg, p_lora), "max_abs_merged_vs_lora": float((p_merg - p_lora).abs().max()),
                                "rel_l2_lora_vs_frozen": rel(p_lora, p_froz), "rel_l2_merged_vs_frozen": rel(p_merg, p_froz)}
                log(f"c2 {checks['c2']}")
                del frozen
            return model, mean, std, n_special

        pep.build_model = build_model
        return pep, ship

    cp.load_builder = load_builder
    sys.argv = ["cache_pod.py", "--out", str(out), "--only", *clips, "--threads", str(a.threads)]
    t0 = time.time()
    rc = cp.main()
    checks["cache_pod_rc"] = rc
    checks["wall_s"] = round(time.time() - t0, 1)
    if a.mode == "lora":
        ref = checks.pop("c2_ref")["first8_pooled_fp8"]
        x = torch.load(out / "eval" / "clips" / f"{clips[0]}.pt", map_location="cpu", weights_only=False)["pool4x10"][:8].float()
        checks["c2_route"] = {"clip": clips[0], "cache_first8_vs_module_pooled_bitwise": bool(torch.equal(x, ref)),
                              "rel_l2": float((x.double() - ref.double()).norm() / ref.double().norm())}
        log(f"c2 route: cache rows == the trained module's pooled fp8 on the first 8 frames: {checks['c2_route']}")
    if a.mode in ("zero", "lora-zero"):
        rows = []
        for c in clips:
            x = torch.load(out / "eval" / "clips" / f"{c}.pt", map_location="cpu", weights_only=False)
            y = torch.load(REF_EVAL / f"{c}.pt", map_location="cpu", weights_only=False)
            r = {"clip": c}
            for g in ("pool4x10", "pool8x20"):
                r[g] = {"bitwise_equal": bool(torch.equal(x[g].view(torch.uint8), y[g].view(torch.uint8))),
                        "rel_l2": float((x[g].double() - y[g].double()).norm() / y[g].double().norm())}
            rows.append(r)
        checks["c3"] = rows
        checks["C3_PASS"] = all(r[g]["bitwise_equal"] for r in rows for g in ("pool4x10", "pool8x20"))
        log(f"c3 bitwise equal to cache_pod/eval on all {len(rows)} clips: {checks['C3_PASS']}")
    mp = out / "eval" / "meta.json"
    meta = json.loads(mp.read_text(encoding="utf-8"))
    meta["pod_stage"] = f"S0-L step 3: L_s0-merged eval-141 re-encode (mode {a.mode})"
    meta["lora_merge"] = {k: checks[k] for k in ("mode", "ckpt", "ckpt_sha256", "lora_meta", "merged_weights_sha256")}
    mp.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    (out / "lmerge_checks.json").write_text(json.dumps(checks, indent=1, default=str), encoding="utf-8")
    n = len(cp.read_jsonl_last(out / "eval" / "manifest.jsonl"))
    log(f"ZZ LMERGE {a.mode.upper()} END ZZ rc {rc} eval clips {n}/{len(clips)} wall {checks['wall_s']} s")
    return 0 if (rc == 0 and n == len(clips) and (a.mode in ("merged", "lora") or checks.get("C3_PASS"))) else 1


if __name__ == "__main__":
    sys.exit(main())
