"""make_planted_reader.py OUT.pt -- a tokprobe checkpoint with a CONSTANT output (presence logit 2.0 -> 0.88, gap 25 m, closing 0) for the wrapper smoke (card ADD46). Pod CPU, torch."""
import sys

import torch

W, N_TOK, HID, FD, HIST = 16, 40, 128, 64, 5
sd = {"tok.0.weight": torch.zeros(HID, 1024), "tok.0.bias": torch.zeros(HID), "tok.2.weight": torch.zeros(W, HID), "tok.2.bias": torch.zeros(W),
      "head.frame.weight": torch.zeros(FD, N_TOK * W), "head.frame.bias": torch.zeros(FD), "head.mlp.0.weight": torch.zeros(128, HIST * FD), "head.mlp.0.bias": torch.zeros(128),
      "head.mlp.2.weight": torch.zeros(3, 128), "head.mlp.2.bias": torch.tensor([2.0, 0.5, 0.0])}
torch.save({"kind": "tokprobe", "arm": "PLANTED", "W": W, "seed": 0, "grid": "pool4x10", "n_tok": N_TOK, "d_feat": 1024, "hidden": HID, "hist": HIST, "input_norm": "ln", "step": 0,
            "argv_sha": "", "state_dict": sd, "n_tok_seen": N_TOK}, sys.argv[1])
print("wrote", sys.argv[1])
