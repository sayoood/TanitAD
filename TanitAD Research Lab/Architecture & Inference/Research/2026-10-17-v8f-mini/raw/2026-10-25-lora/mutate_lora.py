"""PREREG_V8F_LORA: the mutation harness. Each MUTANT is ONE source edit in a scratch COPY of the tree; the tests built to catch it must go RED, and the unmutated copy must be GREEN.

    python mutate_lora.py --tree <repo root holding stack/ taniteval/ "TanitAD Research Lab"/> --work <scratch dir> [--only name1,name2] [--skip-green]

The four mutants the prereg names (A1-A4) plus the ones the instruments' own history asked for:
  A1 detach the encoder output                       A2 drop --lora-r from the built config (the flag stops reaching the build)
  A3 change one preprocessing constant                A4 set B's init to non-zero
  B1 remove the global-RNG save / restore around the encoder build (the heads' init stops being B's)
  B2 the encoder is allowed into train mode (HF DINOv3's RoPE then rescales its coordinates at random)
  B3 block the straight-through gradient of the fp8 quantiser
  B4 off-by-one in the frame indices of a window (the cache and the frames disagree by one cache step)
  B5 `--encoder` defaults to lora (the cache arm stops being today's trainer)
  B6 do not freeze the backbone (the trainable count and the frozen-grad control must catch it)
  B7 A is initialised from the GLOBAL RNG
  B8 the merge adds the update with the wrong sign
  B9 the unique-frame key forgets the clip (windows of different clips share frames)
  B10 the heads' loss is evaluated per micro-batch (plain accumulation) instead of once on the full batch
  B11/B12 the smoke script / the eval-route script resolve the repo root one directory too high (the off-by-one that hid in both)
Output: a table; exit 0 iff every mutant is RED and the unmutated tree is GREEN.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ENC = "stack/tanitad/models/v8f_lora_encoder.py"
TRN = "stack/scripts/v8f_mini_train.py"
DAT = "stack/tanitad/data/v8f_mini_data.py"
T = "stack/tests/test_v8f_lora.py"
PKC = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/code/"
SMK = PKC + "v8f_lora_smoke.py"
EVR = PKC + "v8f_lora_eval_cache.py"

#: (name, file, old, new, pytest -k expression naming the tests that must go red)
MUTANTS = [
    ("A1_detach_encoder_output", ENC, "        return fp8_ste(p) if (self.quantize if quantize is None else quantize) else p",
     "        return (fp8_ste(p) if (self.quantize if quantize is None else quantize) else p).detach()",
     "step_one_has_nonzero or detached_encoder or reachability or gradient_cache"),
    ("A2_lora_r_not_reaching_the_build", TRN, "    return L.DinoV3LoRAEncoder.from_pretrained(r=a.lora_r, alpha=a.lora_alpha,",
     "    return L.DinoV3LoRAEncoder.from_pretrained(r=32, alpha=a.lora_alpha,", "real_loader or reachability"),
    ("A2b_lora_r_dropped_in_the_test_stand_in", T, "    return E.DinoV3LoRAEncoder(bb, m, s, r=a.lora_r, alpha=a.lora_alpha, targets=TR.parse_targets(a.lora_targets), seed=a.seed, grid=a.grid[len(\"pool\"):]).to(dev)",
     "    return E.DinoV3LoRAEncoder(bb, m, s, r=8, alpha=a.lora_alpha, targets=TR.parse_targets(a.lora_targets), seed=a.seed, grid=a.grid[len(\"pool\"):]).to(dev)",
     "reachability or launch_gate"),
    ("A3_preprocessing_constant_255_to_256", ENC, "PIXEL_SCALE = 255.0", "PIXEL_SCALE = 256.0",
     "cache_builders_encode_episode or pool_is_the_builders or b_zero_forward or constants"),
    ("A3b_preprocessing_std_constant", ENC, "IMAGE_STD = (0.229, 0.224, 0.225)", "IMAGE_STD = (0.229, 0.224, 0.226)",
     "cache_builders_encode_episode or pool_is_the_builders"),
    ("A4_B_init_nonzero", ENC, "        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.r, dtype=torch.float32, device=base.weight.device))",
     "        self.lora_B = nn.Parameter(torch.full((self.out_features, self.r), 0.01, dtype=torch.float32, device=base.weight.device))",
     "b_zero_forward or step_one_has_nonzero or building_the_encoder or launch_gate"),
    ("B1_no_rng_restore", TRN, "    finally:\n        torch.set_rng_state(st)", "    finally:\n        pass", "encoder_build_never_moves or real_loader"),
    ("B2_encoder_may_enter_train_mode", ENC, "        super().train(mode)\n        self.backbone.eval()\n", "        super().train(mode)\n",
     "train_mode_randomises_rope or checkpointing_changes_nothing"),
    ("B3_ste_blocks_the_gradient", ENC, "    def backward(ctx, g):\n        return g\n", "    def backward(ctx, g):\n        return g * 0.0\n",
     "fp8_ste or step_one_has_nonzero or reachability or gradient_cache"),
    ("B4_off_by_one_frame_index", DAT, "        idx = np.clip(np.arange(t - self.hmax + 1, t + 1), 0, it[\"T\"] - 1)",
     "        idx = np.clip(np.arange(t - self.hmax, t), 0, it[\"T\"] - 1)", "frame_windows_carry or end_to_end_identity"),
    ("B5_encoder_defaults_to_lora", TRN, "ap.add_argument(\"--encoder\", choices=(\"cache\", \"lora\"), default=\"cache\",",
     "ap.add_argument(\"--encoder\", choices=(\"cache\", \"lora\"), default=\"lora\",", "default_that_builds_nothing or cache_is_the_pre_lora or cache_encoder_refuses"),
    ("B6_backbone_not_frozen", ENC, "    for p in model.parameters():\n        p.requires_grad_(False)\n    n = 0\n", "    n = 0\n",
     "trainable_count or step_one_has_nonzero or launch_gate"),
    ("B7_A_init_from_global_rng", ENC, "        a.uniform_(-bound, bound, generator=gen)\n        self.lora_A = nn.Parameter(a.to(base.weight.device))",
     "        a.uniform_(-bound, bound)\n        self.lora_A = nn.Parameter(a.to(base.weight.device))", "never_touches_the_global_rng"),
    ("B8_merge_sign_flipped", ENC, "m.base.weight.data = (m.base.weight.data.float() + m.delta_weight().to(m.base.weight.device)).to(m.base.weight.dtype)",
     "m.base.weight.data = (m.base.weight.data.float() - m.delta_weight().to(m.base.weight.device)).to(m.base.weight.dtype)", "merge_equals_unmerged or eval_route"),
    ("B9_unique_frames_forgets_the_clip", DAT, "            k = (c, int(idx[b, f]))", "            k = (0, int(idx[b, f]))", "unique_frames or collate_frames or end_to_end_identity"),
    ("B10_plain_microbatch_accumulation", ENC,
     "    out = run_loss(leaf[frame_map.to(leaf.device)])\n    out[\"terms\"][\"total\"].backward()\n    g = leaf.grad",
     "    _fm = frame_map.to(leaf.device)\n    _h = max(1, _fm.shape[0] // 2)\n    out = None\n    for _s in range(0, _fm.shape[0], _h):\n        _o = run_loss(leaf[_fm[_s:_s + _h]])\n"
     "        (_o[\"terms\"][\"total\"] * (_fm[_s:_s + _h].shape[0] / _fm.shape[0])).backward()\n        out = _o if out is None else out\n    g = leaf.grad",
     "accumulation_is_exact or gradient_cache_equals"),
    ("B11_smoke_repo_root_off_by_one", SMK, "REPO = HERE.parents[5]", "REPO = HERE.parents[6]", "smoke_script"),
    ("B12_eval_route_repo_root_off_by_one", EVR, "REPO = HERE.parents[5]", "REPO = HERE.parents[6]", "eval_route"),
]


def copy_tree(src: Path, dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    ign = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")
    for sub in ("stack", "taniteval", "TanitAD Research Lab", "tools"):
        if (src / sub).exists():
            shutil.copytree(src / sub, dst / sub, ignore=ign)


def run_pytest(tree: Path, kexpr: str | None, work: Path, tag: str, timeout: int) -> tuple:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONPATH=os.pathsep.join([str(tree / "stack"), str(tree / "taniteval")]), PYTHONDONTWRITEBYTECODE="1")
    cmd = [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "--basetemp", str(work / f"bt_{tag}"), str(tree / T), "-W", "ignore"]
    if kexpr:
        cmd += ["-k", kexpr]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(tree), env=env, capture_output=True, text=True, encoding="utf-8", timeout=timeout)
    out = r.stdout + r.stderr
    failed = [l.split(" - ")[0].replace("FAILED ", "").strip() for l in out.splitlines() if l.startswith("FAILED")]
    summ = [l for l in out.splitlines() if (" passed" in l or " failed" in l or " error" in l) and "==" not in l][-1:] or [""]
    return r.returncode, failed, summ[0].strip(), time.time() - t0, out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--skip-green", action="store_true")
    ap.add_argument("--timeout", type=int, default=1500)
    a = ap.parse_args(argv)
    tree, work = Path(a.tree), Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    pristine, scratch = work / "pristine", work / "scratch"
    copy_tree(tree, pristine)
    rows, ok = [], True
    if not a.skip_green:
        copy_tree(pristine, scratch)
        rc, failed, summ, secs, out = run_pytest(scratch, "not real_loader and not real_dinov3", work, "green", a.timeout)
        rows.append(("UNMUTATED", "GREEN" if rc == 0 else "RED", summ, failed, secs))
        print(f"UNMUTATED: {'GREEN' if rc == 0 else 'RED  <-- the baseline must be green'} ({summ}, {secs:.0f} s)", flush=True)
        ok &= rc == 0
        if rc:
            print(out[-1500:])
    only = set(x for x in a.only.split(",") if x)
    for name, rel, old, new, kexpr in MUTANTS:
        if only and name not in only:
            continue
        copy_tree(pristine, scratch)
        p = scratch / rel
        s = p.read_text(encoding="utf-8")
        if s.count(old) != 1:
            print(f"{name}: MUTATION DID NOT APPLY ({s.count(old)} matches for the old text in {rel})", flush=True)
            rows.append((name, "NOT-APPLIED", "", [], 0.0))
            ok = False
            continue
        p.write_text(s.replace(old, new), encoding="utf-8", newline="")
        rc, failed, summ, secs, out = run_pytest(scratch, kexpr, work, name, a.timeout)
        red = rc != 0
        rows.append((name, "RED" if red else "GREEN (SURVIVED)", summ, failed, secs))
        print(f"{name}: {'RED' if red else 'GREEN  <-- SURVIVED'}  {summ}  first red: {failed[:2]}  ({secs:.0f} s)", flush=True)
        ok &= red
    json.dump([{"mutant": r[0], "result": r[1], "pytest": r[2], "failed_tests": r[3], "seconds": r[4]} for r in rows], open(work / "mutate_lora.json", "w"), indent=1)
    print("VERDICT", "ALL MUTANTS RED, BASELINE GREEN" if ok else "SOME MUTANT SURVIVED OR THE BASELINE IS RED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
