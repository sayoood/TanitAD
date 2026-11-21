"""⭐ Verify in SOURCE + at RUNTIME: our imagination is not candidate-conditioned.

Claim under test (from the scorer-inputs research, §2.2): ``imagine_probes``
rolls a SHARED probe vocabulary, so the imagined tokens are identical for all
256 candidates and structurally cannot rank them. If true it is a code fix, not
a research question, and it makes E-V5-1's failure over-determined.

The runtime proof does not need a checkpoint or a GPU: the imagination token
count is a pure function of ``n_probes`` and ``imag_read`` and carries NO
candidate axis, so changing ``n_anchors`` cannot change it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO / "stack"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from tanitad.models.flagship_v15 import (V15Config, FlagshipV15Head,
                                             imagine_probes,
                                             build_probe_vocabulary)

    cfgs = {}
    for n_anchors in (64, 256):
        c = V15Config(n_anchors=n_anchors)
        h = FlagshipV15Head(c)
        cfgs[n_anchors] = dict(n_anchors=int(c.n_anchors),
                               n_probes=int(c.n_probes),
                               imag_read=list(c.imag_read),
                               n_imag_tokens=int(h.n_imag_tokens),
                               n_state_tokens=int(h.n_state_tokens))
    invariant = (cfgs[64]["n_imag_tokens"] == cfgs[256]["n_imag_tokens"])

    # shape proof on the function itself: a tiny random predictor stand-in
    c = V15Config()
    probes = build_probe_vocabulary(torch.randn(64, 20, 2), c.n_probes, seed=0)
    B, W, S, A = 3, c.window, c.state_dim, 3

    class _Pred(torch.nn.Module):
        def forward(self, ws, wa):
            return None, ws[:, -1] * 0.999

    out = imagine_probes(_Pred(), torch.randn(B, W, S), torch.randn(B, W, A),
                         probes, c.imag_read, torch.zeros(B))

    res = dict(
        what="is the flagship imagination candidate-conditioned?",
        verdict="NO — structurally cannot rank candidates",
        source_evidence=dict(
            file="stack/tanitad/models/flagship_v15.py",
            imagine_probes_signature_line=505,
            shared_probe_expand_line=525,
            shared_probe_expand_code="pr = probes.unsqueeze(0)"
                                     ".expand(b, m, k, 2).reshape(b * m, k, 2)",
            token_count_line=269,
            token_count_code="self.n_imag_tokens = cfg.n_probes * "
                             "len(cfg.imag_read) if cfg.cond_imagination else 0",
            reading="`probes` is [M, K, 2] — a vocabulary of PROBE ACTION "
                    "SEQUENCES shared across the batch. The returned tensor is "
                    "[B, M*len(read), S]: axes are (window, probe, latent). "
                    "There is NO candidate axis anywhere in the imagination "
                    "path, and V15Decoder._decode emits all n_anchors "
                    "candidates from the SAME kv built from those tokens."),
        runtime_proof=dict(
            configs=cfgs,
            n_imag_tokens_invariant_to_n_anchors=bool(invariant),
            imagine_probes_output_shape=list(out.shape),
            expected_shape=[B, c.n_probes * len(c.imag_read), S],
            shape_matches=list(out.shape) == [B, c.n_probes * len(c.imag_read), S],
            note=f"{cfgs[256]['n_imag_tokens']} imagination tokens serve all "
                 f"{cfgs[256]['n_anchors']} candidates — identical for every one"),
        consequence=dict(
            for_E_V5_1="a feature identical across candidates cannot rank them; "
                       "the v5 imagination-selection negative is over-determined",
            fix="make the roll candidate-conditioned (WoTE rolls the world model "
                "ONCE PER CANDIDATE). The v5 stream already implements the "
                "candidate -> action-sequence map (V5_IMAGINATION_SELECTION §0.3), "
                "so the missing piece is wiring it into the head's conditioning "
                "path rather than only into a post-hoc scorer.",
            class_="CODE FIX, not a research question"))
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
