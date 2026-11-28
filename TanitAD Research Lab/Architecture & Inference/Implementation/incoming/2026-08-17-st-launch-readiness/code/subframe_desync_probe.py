"""Does `--v2-subframe 176x624` actually desync the v6 encoder? EXECUTE it.

⚠️ I nearly reported this from reading, and the reading would have been WRONG
about WHERE it breaks. `resolve_v2_frames`'s own docstring says "The frame is
applied to ``cfg`` too, so the ENCODER is sized for what it will be fed" — true
for `train_flagship_v4`, whose ``cfg`` IS the model config. In
`train_v6_staged` the call is `resolve_eval_frames(a, cfg_eval)` and ``cfg_eval``
is `eval_flagship_v4._eval_cfg()` — a FLAGSHIP-V4 config used for the plan and
the eval seam. The v6 stack was already built, from ``a.frame_h``/``a.frame_w``,
120 lines earlier. So the sub-frame moves the DATA and not the MODEL.

Consequence, measured below: `--init-from` succeeds (the checkpoint and the
stack are both 256x640) and the run dies later, on the first forward, after the
corpus has mounted. That is the expensive failure shape this programme keeps
paying for — a refusal that arrives after the compute.

Evidence class: MEASURED (ours).
"""
from __future__ import annotations

import pyarrow  # noqa: F401  ⚠️ BEFORE torch on the dev box
import json
import shlex
import sys
from pathlib import Path

STACK = Path(r"G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/stack")
sys.path.insert(0, str(STACK))
sys.path.insert(0, str(STACK / "scripts"))

import torch  # noqa: E402

from train_v6_staged import build_parser, build_stack_from_args  # noqa: E402

GEOM = shlex.split(
    "--stage S-W --out /tmp/x --in-channels 9 --frame-h 256 --frame-w 640 "
    "--patch 16 --enc-dim 768 --enc-depth 12 --enc-heads 12 "
    "--readout-grid 4 --readout-dim 128 --pred-modern --pred-dim 1024 "
    "--pred-depth 12 --pred-heads 16 --window 6 --horizons 1 2 4 "
    "--d-tac 768 --d-str 512 --d-goal-embed 128 --adapter-hidden 512 "
    "--n-candidates 8 --param-budget 350000000 --f-hidden-tac 1024 "
    "--f-hidden-str 1024 --f-blocks 6 --vit5-encoder --n-registers 4 "
    "--plan-steps 60 --dt 0.1 --a-max 4.0 --kappa-max 0.2 "
    "--uplink stopgrad --ema-decay 0.996 --sigreg-slices 512")


def main() -> int:
    out = {"_evidence_class": "MEASURED (ours; real forward on the built stack)"}
    ap = build_parser()
    a = ap.parse_args(GEOM)
    torch.manual_seed(0)
    stack = build_stack_from_args(a)
    enc = stack.encoder
    enc.eval()
    out["encoder_built_for"] = {"frame_h": a.frame_h, "frame_w": a.frame_w}
    out["encoder_pos_shape"] = list(
        dict(enc.named_parameters()).get(
            "enc.pos", dict(enc.named_parameters()).get("pos")).shape) \
        if any(n.endswith("pos") for n, _ in enc.named_parameters()) else None
    for name, (h, w) in {"live_256x640": (256, 640),
                         "subframe_176x624": (176, 624)}.items():
        x = torch.zeros(1, 9, h, w)
        try:
            with torch.no_grad():
                y = enc(x)
            out[name] = {"ok": True,
                         "out_shape": [list(t.shape) for t in y]
                         if isinstance(y, (tuple, list)) else list(y.shape)}
        except BaseException as exc:
            out[name] = {"ok": False, "error_type": type(exc).__name__,
                         "error": str(exc)[:500]}
    print(json.dumps(out, indent=1))
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
