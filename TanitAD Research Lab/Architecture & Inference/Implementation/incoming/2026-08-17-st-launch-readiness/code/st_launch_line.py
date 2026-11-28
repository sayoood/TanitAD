"""COMPOSE the S-T launch line for Thor — and VALIDATE it, not just print it.

⛔ WHY IT IS NOT `v6_chain.py commands --step S-T`. MEASURED 2026-08-17:
`v6_chain.trainer_argv` emits NO model-geometry flag at all on a non-dry step,
so the S-T stack is built at the trainer's DEFAULTS (enc 384x8, pred 768x6,
d_tac 512, d_str 256, no ViT-5, no modern predictor) while the live S-W
checkpoint is 768x12 / 1024x12 / 768 / 512 / ViT-5 / modern. `--init-from` then
dies on SHAPES — `encoder.pos [1,640,768] vs [1,640,384]` — a failure
`STAGE_MAY_INTRODUCE` never sees, because `load_state_dict(strict=False)`
tolerates missing/unexpected KEYS and still RAISES on shape. `ChainConfig` has
an `extra_common` escape hatch and NO CLI flag reaches it.

This file composes the line from the LIVE run's own `config.json["args"]`
(read off Thor) rather than from prose, then EXECUTES the validation:

  1. the composed argv parses under HEAD's `build_parser`;
  2. the stack it builds is the production 336.58 M / 575-key S-T stack;
  3. `load_stage_init` on the REAL S-W artifact returns missing=[] unexpected=[]
     and introduces exactly the two zero-init `cond_tac_dyn.*` keys.

A command that has not been through (3) is a command someone typed.

Evidence class: MEASURED (ours).
"""
from __future__ import annotations

import pyarrow  # noqa: F401  ⚠️ BEFORE torch on the dev box
import argparse
import json
import shlex
import sys
from pathlib import Path

STACK = Path(r"G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/stack")
sys.path.insert(0, str(STACK))
sys.path.insert(0, str(STACK / "scripts"))

import torch  # noqa: E402

from train_v6_staged import build_parser, build_stack_from_args, load_stage_init  # noqa: E402

# ---- THOR FACTS, read off the box 2026-08-17 (not defaults, not prose) -----
THOR = {
    "python": "/home/nvidia/venvs/tanitad-train/bin/python",
    "workdir": "/home/nvidia/TanitAD/stack",
    "root": "/home/nvidia/experiments",
    "sw_dir": "/home/nvidia/experiments/v6F-SW-30k",
    "train_cache": "/home/nvidia/data/physicalai-train-e438721ae894-w120-256x640cyl",
    "val_cache": "/home/nvidia/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl",
}
OUT = "/home/nvidia/experiments/v6F-ST-10k"

#: ⛔ EVERY geometry + vocabulary flag the live S-W run carries, verbatim. The
#: S-T stack must be built from the SAME ones or `--init-from` dies on shapes.
#: `--v2-subframe` is DELIBERATELY ABSENT: the live run has `v2_subframe: null`,
#: and in `train_v6_staged` the sub-frame moves the DATA and not the MODEL
#: (see `code/subframe_desync_probe.py` — MEASURED ValueError at the first
#: forward, after the corpus has mounted).
GEOMETRY = shlex.split("""
--in-channels 9 --frame-h 256 --frame-w 640 --patch 16
--enc-dim 768 --enc-depth 12 --enc-heads 12 --grad-checkpoint
--readout-grid 4 --readout-dim 128 --pred-modern
--pred-dim 1024 --pred-depth 12 --pred-heads 16 --window 6 --horizons 1 2 4
--d-tac 768 --d-str 512 --d-goal-embed 128 --adapter-hidden 512
--n-candidates 8 --param-budget 350000000
--f-hidden-tac 1024 --f-hidden-str 1024 --f-blocks 6
--vit5-encoder --n-registers 4 --plan-steps 60 --dt 0.1
--a-max 4.0 --kappa-max 0.2 --uplink stopgrad --ema-decay 0.996
--sigreg-slices 512 --sigreg-free-dims 0
""")

#: S-T's own stage flags. `--tac-goal-cond` is F-1's g_str->P_T port: S-T is
#: where it is BUILT, `STAGE_MAY_INTRODUCE['S-T']` admits the zero-init keys,
#: and every stage above must carry the flag or the keys become UNEXPECTED.
STAGE = shlex.split("""
--stage S-T --steps 10000 --lr 0.0001 --batch 8 --max-horizon 60
--tac-goal-cond --plan-wta-eps 0.05 --w-t1 1.0
--o5-k 60 --o1-k 10
""")

#: Data + runtime. ⚠️ `--v2-lru 64` is the LIVE run's value; `v6_chain`'s Thor
#: default is 6 and that is a host-RAM knob, not a geometry one — but the live
#: run has been stable at 64 for 44 h, so it is the measured-safe setting.
RUNTIME = shlex.split(f"""
--v2-cache {THOR['train_cache']} --v2-val-cache {THOR['val_cache']}
--v2-lru 64 --frame-hfov 120.0 --projection cylindrical --require-parity
--eps-per-batch 4 --wd 0.05 --clip 1.0 --log-every 50 --save-every 250
--device cuda --seed 0
""")

#: ⭐ F-16: the ONLY thing that banks the 60-step plan. Without it the S-T gate's
#: `X2_seam` row is `not-run` again and the instrument produces its third
#: consecutive zero real numbers. ZERO extra GPU — the plan is already computed
#: for the step's loss.
SEAM = ["--dump-seam-plan", f"{OUT}/seam"]


def argv() -> list[str]:
    return (STAGE + ["--out", OUT,
                     "--init-from", f"{THOR['sw_dir']}/ckpt.pt",
                     "--prev-gate", f"{THOR['sw_dir']}/stage_gate.json"]
            + GEOMETRY + RUNTIME + SEAM)


def launch_line() -> str:
    body = " ".join(shlex.quote(x) for x in argv())
    return (f"mkdir -p {OUT}/seam && cd {THOR['workdir']} && "
            f"PYTHONPATH={THOR['workdir']} OMP_NUM_THREADS=6 "
            f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
            f"setsid nohup {THOR['python']} -u scripts/train_v6_staged.py "
            f"{body} > {OUT}/train.out 2>&1 < /dev/null &")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-ckpt", default=None)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-txt", required=True)
    a = ap.parse_args()
    rep: dict = {"_evidence_class": "MEASURED (ours; the composed argv is "
                                    "parsed, built and load-tested here)",
                 "launch_line": launch_line(), "argv": argv()}

    p = build_parser()
    p.add_argument("--i-know-this-is-the-control-arm", action="store_true",
                   dest="control_arm_ack")
    ns = p.parse_args(argv())
    rep["parsed"] = {"stage": ns.stage, "out": ns.out, "steps": ns.steps,
                     "selector": ns.selector, "tac_goal_cond": ns.tac_goal_cond,
                     "v2_subframe": ns.v2_subframe,
                     "dump_seam_plan": ns.dump_seam_plan,
                     "init_from": ns.init_from}
    torch.manual_seed(0)
    st = build_stack_from_args(ns)
    rep["st_params"] = sum(q.numel() for q in st.parameters())
    rep["st_keys"] = len(st.state_dict())
    if a.real_ckpt and Path(a.real_ckpt).exists():
        rep["load_against_real_sw"] = load_stage_init(
            st, a.real_ckpt, stage="S-T")
    Path(a.out_json).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    Path(a.out_txt).write_text(launch_line() + "\n", encoding="utf-8")
    print(json.dumps(rep, indent=1)[:5000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
