#!/usr/bin/env python3
"""make_synthetic_llr.py -- a SYNTHETIC LLR-shaped checkpoint for the instruments' tests (NOT an LLR checkpoint: C0's weights + FRESH lvl_head / lead_head / e_aux, cfg level_head on, lead_head on, w_lead 3.362, aux_cond token),
built with the LLR read tree's model so the cfg and key set are the real build's.  Writes F:/TanitAD-scratch/motion_c/llr_read/pt/synth/M2b_s{s}_recft_LLR/ckpt_last.pt + argv.json + SHA256SUMS.txt (a synthetic rdb for the pre-check
tests).  usage: python -I make_synthetic_llr.py [--seed 0]"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
TREE = "F:/TanitAD-scratch/motion_c/llr_read/tree"
RDB = "E:/Projects/TanitAD-artifacts/v8f-recover"
DST = "F:/TanitAD-scratch/motion_c/llr_read/pt/synth"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dst", default=DST)
    a = ap.parse_args()
    import torch
    spec = importlib.util.spec_from_file_location("v8f_t1_eval_ri", f"{TREE}/stack/scripts/v8f_t1_eval_ri.py")
    T = importlib.util.module_from_spec(spec)
    sys.modules["v8f_t1_eval_ri"] = T
    spec.loader.exec_module(T)
    ck = torch.load(f"{RDB}/M2b_s{a.seed}_recft_C0/ckpt_last.pt", map_location="cpu", weights_only=False)
    cfg = dict(ck["cfg"])
    cfg.update(level_head="on", w_level=0.2027, lead_head="on", w_lead=3.362, aux_cond="token")
    torch.manual_seed(1234 + a.seed)
    m = T.M.V8FMini(T.M.V8FMiniConfig(**cfg), T.M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * T.M.K30)))
    miss, unexp = m.load_state_dict(ck["model"], strict=False)
    assert not unexp, unexp
    ck2 = dict(ck)
    ck2["cfg"], ck2["model"], ck2["step"] = cfg, m.state_dict(), 3000
    d = f"{a.dst}/M2b_s{a.seed}_recft_LLR"
    os.makedirs(d, exist_ok=True)
    torch.save(ck2, f"{d}/ckpt_last.pt")
    h = hashlib.sha256(open(f"{d}/ckpt_last.pt", "rb").read()).hexdigest()
    open(f"{d}/SHA256SUMS.txt", "w", newline="\n").write(f"{h}  ckpt_last.pt\n")
    print("wrote", d, "fresh keys:", sorted({k.split('.')[0] for k in miss}), "sha", h[:12])


if __name__ == "__main__":
    main()
