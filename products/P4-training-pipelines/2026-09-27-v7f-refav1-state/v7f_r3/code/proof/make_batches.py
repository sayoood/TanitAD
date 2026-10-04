"""Build the fixed batches the flag-OFF proof feeds BOTH trees (run once, in the modified tree).

``plain``   : the incumbent synthetic batch (frames/actions/targets), no label keys at all
``labels``  : the same + the REAL v7.2 join's keys for 9 literal clips WITH the R3 targets enabled
              (S2 strategic keys + tac_lat/lon/valid + every TAC_LABEL_BATCH_KEY) -- an OFF trainer
              must ignore every one of them.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

TREE = os.environ["PROOF_TREE"]
sys.path.insert(0, os.path.join(TREE, "stack"))
sys.path.insert(0, os.path.join(TREE, "stack", "scripts"))
sys.path.insert(0, os.path.join(TREE, "stack", "tests"))

import torch  # noqa: E402

import train_v6_staged as T  # noqa: E402
import test_tactical_label_reach_v6 as F  # noqa: E402  (the literal 9-clip fixture)


def main() -> None:
    s = F._stack()
    b = T.synthetic_train_batch(s, batch=len(F.CIDS), k=12, seed=1)
    b["gt_wp"] = torch.randn(len(F.CIDS), 10, 2, generator=torch.Generator().manual_seed(1))
    keys, _ = F._join(Path(tempfile.mkdtemp()), s, [F._rec(i) for i in range(9)],
                      negatives="all")
    assert set(T.TAC_LABEL_BATCH_KEYS) <= set(keys)
    torch.save({"plain": dict(b), "labels": dict(b) | keys}, sys.argv[1])
    print(f"wrote {sys.argv[1]}: plain {len(b)} keys, labels {len(b) + len(keys)} keys")


if __name__ == "__main__":
    main()
