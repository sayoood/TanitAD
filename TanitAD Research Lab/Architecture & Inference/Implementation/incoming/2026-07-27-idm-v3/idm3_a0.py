"""Persist the DEPLOYED head's (A0 = idm_head_v1.pt) predictions on the identical
4,195 v3 val windows, so every v3 contrast can be paired against the artifact
that is actually in production."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/root/idm2")
sys.path.insert(0, "/root/taniteval")
sys.path.insert(0, "/root/v4eval/stack")
sys.path.insert(0, "/root/v4eval/stack/scripts")

import idm2_lib as L        # noqa: E402
import idm_head as ih       # noqa: E402

KBUILD = 8


@torch.no_grad()
def main():
    _, va_tags = L.split_tags()
    va = L.build_set(va_tags, k=KBUILD, stride=2, want_seq=False)
    d = torch.load("/root/idmval/idm_head_v1.pt", weights_only=False)
    h = ih.IDMHead(**d["config"]["head_kwargs"]).cuda()
    h.load_state_dict(d["state_dict"])
    h.eval()
    Z = va["Z"][:, KBUILD - 4:KBUILD + 5].cuda().float()
    S, T = [], []
    for i in range(0, Z.shape[0], 1024):
        o = h(Z[i:i + 1024])
        S.append(o["scalars"].cpu())
        T.append(o["traj"].cpu())
    np.save("/workspace/idm3/out/a0_preds.npy",
            {"S": torch.cat(S).numpy().astype(np.float64),
             "Traj": torch.cat(T).numpy().astype(np.float64)}, allow_pickle=True)
    print("WROTE a0_preds.npy", torch.cat(S).shape)


if __name__ == "__main__":
    main()
