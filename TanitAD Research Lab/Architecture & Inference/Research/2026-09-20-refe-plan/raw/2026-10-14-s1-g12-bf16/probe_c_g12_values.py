"""The G12 dicts the REAL main() --preflight writes on this machine's GPU (CUDA + bf16, H2 argv, stub 256-wide trunk), for: the fixed probe, the OLD probe block, the old block on CPU (the dev-box path),
and the feed path cut.  -> probe_c_g12_values.json.   Run: PYTHONIOENCODING=utf-8 python raw/2026-10-14-s1-g12-bf16/probe_c_g12_values.py"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REFE = HERE.parent.parent / "refe"
sys.path.insert(0, str(REFE))
import test_aux_g12_bf16 as T  # noqa: E402

src = T.SRC_PATH.read_text(encoding="utf-8")
h, nb, tl = T.split_block(src)
old = h + T.OLD_BLOCK + tl
KEYS = ("ok", "feed", "feed_zero_init", "feed_step0_equal_base", "feed_probe_reads_difference", "feed_probe_known_field_reads_difference", "feed_probe_max_abs_diff",
        "feed_probe_elems_differ", "feed_probe_elems", "feed_probe_scalar_std_over_proposals", "feed_proj_grad_abs")
out = {}
for name, s, patch, cuda in (("fixed_probe_cuda_bf16", src, None, True), ("OLD_probe_cuda_bf16", old, None, True), ("OLD_probe_cpu_fp32_(dev_box_path)", old, None, False),
                             ("fixed_probe_cuda_bf16_FEED_CUT", src, T.patch_cut, True), ("fixed_probe_cpu_fp32_(dev_box_path)", src, None, False)):
    r = T.pre_cuda(s, "p_" + name.split("_")[0], patch, cuda=cuda)
    g = T.W.gate(r, T.G12)
    out[name] = {k: g.get(k) for k in KEYS} if g else None
    out[name]["consumer"] = T.consume(r)["status"]
# ---- H1 (feed off): real CrossBlock scorer stub at 7,680 tokens; the identity probe on CUDA bf16
H1K = ("ok", "feed", "reference_replay_identical", "scorer_params_bit_identical_2steps", "identity_probe_reads_one_ulp", "adam_head_params", "adam_head_params_with_state")
warn = src.replace("torch.use_deterministic_algorithms(True, warn_only=False)", "torch.use_deterministic_algorithms(True, warn_only=True)")
leak = src.replace("lg_, sd_ = H.predict(sctx_d)\n", "lg_, sd_ = H.predict(sctx_d + S.score_head.bias.sum())\n")
for name, s_, cuda in (("H1_fixed_strict_cuda_bf16", src, True), ("H1_fixed_strict_cuda_bf16_run2", src, True), ("H1_OLD_warn_only_cuda_bf16_run1", warn, True), ("H1_OLD_warn_only_cuda_bf16_run2", warn, True),
                       ("H1_OLD_warn_only_cuda_bf16_run3", warn, True), ("H1_OLD_warn_only_cpu_fp32_(dev_box_path)", warn, False), ("H1_head_loss_reaches_scorer_cuda_bf16", leak, True)):
    r = T.pre_h1(s_, "h1p", cuda=cuda)
    g = T.W.gate(r, T.G12)
    out[name] = {k: g.get(k) for k in H1K} if g else None
    out[name]["consumer"] = T.consume(r, "H1-s0")["status"]
Path(HERE / "probe_c_g12_values.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out, indent=1))
