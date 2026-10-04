"""Flag-OFF bit-identity proof, CROSS-PROCESS: run once per source tree, compare the JSONs.

usage:  PROOF_TREE=<tree root holding stack/> python bit_identity_off.py <batches.pt> <out.json>

Per (config, stage, batch): the loss value and every term as float.hex, the log key set, a digest
of the global RNG state after the step, a digest of EVERY parameter gradient after backward, and a
digest of the freshly-built stack's state_dict (proves both trees built the same weights).
Default ``V6LossWeights()`` -- i.e. ``--w-tac-label-all`` OFF -- in every stage.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

TREE = os.environ["PROOF_TREE"]
sys.path.insert(0, os.path.join(TREE, "stack"))
sys.path.insert(0, os.path.join(TREE, "stack", "scripts"))

import torch  # noqa: E402

torch.set_num_threads(1)

import train_v6_staged as T  # noqa: E402
import tanitad  # noqa: E402
from tanitad.config import EncoderConfig, PredictorConfig, ReadoutConfig  # noqa: E402
from tanitad.models.v6 import STAGES, V6Config, V6Stack  # noqa: E402


def dig(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def cfg(**kw) -> V6Config:
    base = dict(
        encoder=EncoderConfig(in_channels=3, image_size=32, image_width=32, patch_size=16,
                              d_model=32, depth=1, n_heads=2),
        readout=ReadoutConfig(grid=4, d_readout=8),
        predictor=PredictorConfig(d_model=32, depth=1, n_heads=2, window=4, horizons=(1,),
                                  action_dim=3),
        d_tac=32, d_str=16, d_goal_embed=16, adapter_hidden=32, f_hidden_tac=32,
        f_hidden_str=32, d_plan_feat=16, emission_hidden=16, n_candidates=3, aux_hidden=16,
        sigreg_slices=8)
    base.update(kw)
    return V6Config(**base)


def main() -> None:
    batches = torch.load(sys.argv[1], weights_only=False)
    res = {"trainer_file": T.__file__, "tanitad_file": tanitad.__file__,
           "has_r3_symbol": hasattr(T, "tac_label_all_loss"), "runs": {}}
    for cname, ckw in (("default", {}), ("multilabel", {"goal_multilabel": True})):
        for bname, b in batches.items():
            for stage in STAGES:
                torch.manual_seed(0)
                s = V6Stack(cfg(**ckw))
                sd = hashlib.sha256("".join(
                    f"{k}:{dig(v)}" for k, v in sorted(s.state_dict().items())).encode()
                ).hexdigest()
                s.zero_grad(set_to_none=True)
                torch.manual_seed(3)
                L = T.v6_loss_step(s, dict(b), stage=stage, weights=T.V6LossWeights(),
                                   o1_k=10, o5_k=12,
                                   generator=torch.Generator().manual_seed(11))
                rng = dig(torch.random.get_rng_state())
                L["loss"].backward()
                grads = {n: dig(p.grad) for n, p in s.named_parameters() if p.grad is not None}
                res["runs"][f"{cname}|{bname}|{stage}"] = {
                    "loss_hex": float(L["loss"].detach()).hex(),
                    "terms": {t: float(L[t].detach()).hex() for t in L["log"]["terms"]},
                    "log_keys": sorted(L["log"]),
                    "rng_after_step": rng,
                    "n_param_grads": len(grads),
                    "grads_digest": hashlib.sha256(
                        json.dumps(grads, sort_keys=True).encode()).hexdigest(),
                    "state_dict_digest": sd}
    with open(sys.argv[2], "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    print(f"wrote {sys.argv[2]}: {len(res['runs'])} runs, trainer={T.__file__}")


if __name__ == "__main__":
    main()
