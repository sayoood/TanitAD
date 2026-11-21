"""EXECUTE the S-W -> S-T transition at the LIVE production geometry.

⛔ WHY THIS EXISTS AND WHY THE SUITE IS NOT ENOUGH. `tests/
test_v6_stage_init_introduction.py` proves the `STAGE_MAY_INTRODUCE` mechanism
— but it builds `V6Config()` at its DEFAULTS (enc 384x8, pred 768x6, d_tac 512,
d_str 256, no ViT-5 encoder, no modern predictor, plan_steps 24). The live
`v6F-SW-30k` run is enc 768x12 / pred 1024x12 / d_tac 768 / d_str 512 /
--vit5-encoder / --pred-modern / --n-registers 4 / --plan-steps 60. A passing
test at the default geometry says nothing about the transition on the artifact
that will actually exist at 30k. This runs it on BOTH:

  A. a synthetic S-W checkpoint built from the LIVE argv (read verbatim off
     PID 25477's /proc cmdline), and
  B. the REAL fp16 snapshot `v6F_sw_step010000.fp16.pt` pulled from Thor and
     md5-verified against the source.

⚠️ (B) is a WEIGHTS-ONLY snapshot (`ops/ckpt_fp16_snapshot.py` drops `opt` by
design). It proves the KEY SET and the SHAPES of the real trunk — which is
exactly what `load_stage_init` adjudicates — and it proves the fp16 unwrap
path. It does NOT prove anything about optimiser state, and `--init-from`
starts a NEW run at step 0 so there is no optimiser state to inherit anyway.
What it cannot prove is a `--resume` (RESUME_CONTRACT["has_optimiser"] refuses
this artifact for that, correctly).

Evidence class: MEASURED (ours). Every number this prints is produced by
running the real `load_stage_init` from `scripts/train_v6_staged.py` at HEAD.
"""
from __future__ import annotations

import pyarrow  # noqa: F401  ⚠️ BEFORE torch on the dev box (0xC0000005, A/B proven)
import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path

STACK = Path(__file__).resolve().parents[5] / "stack"
if not (STACK / "scripts" / "train_v6_staged.py").exists():          # pragma: no cover
    STACK = Path(r"G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/stack")
sys.path.insert(0, str(STACK))
sys.path.insert(0, str(STACK / "scripts"))

import torch  # noqa: E402

from train_v6_staged import (  # noqa: E402
    STAGE_MAY_INTRODUCE, build_parser, build_stack_from_args, load_stage_init)

# ---------------------------------------------------------------------------
# THE LIVE ARGV — read verbatim from Thor PID 25477 (`ps -o cmd`), 2026-08-17.
# Everything after the interpreter and the script path.
# ---------------------------------------------------------------------------
LIVE_ARGV = shlex.split("""
--stage S-W --out /home/nvidia/experiments/v6F-SW-30k --resume auto
--in-channels 9 --frame-h 256 --frame-w 640 --patch 16
--enc-dim 768 --enc-depth 12 --enc-heads 12 --grad-checkpoint
--readout-grid 4 --readout-dim 128 --pred-modern
--pred-dim 1024 --pred-depth 12 --pred-heads 16 --window 6 --horizons 1 2 4
--d-tac 768 --d-str 512 --d-goal-embed 128 --adapter-hidden 512
--n-candidates 8 --param-budget 350000000
--f-hidden-tac 1024 --f-hidden-str 1024 --f-blocks 6
--vit5-encoder --n-registers 4 --plan-steps 60 --dt 0.1
--a-max 4.0 --kappa-max 0.2 --uplink stopgrad --ema-decay 0.996
--o1-k 10 --w-o1-ctrl 1.0 --w-o1-fact 1.0 --w-o1-scene 0.3
--dkappa 0.02 --daccel 2.0 --rand-dkappa-max 0.05 --rand-daccel-max 3.0
--w-o2 1.0 --o2-tau-s 2.0 --w-o3 1.0 --o3-mode action
--o3-blocks 2 --o3-block-h 2 --o3-block-w 2 --o3-band-rows 0
--o4-alpha 1.0 --o4-floor 0.25 --w-o5 1.0 --o5-k 60 --o5-mode uniform
--w-o6 0.1 --sigreg-slices 512 --sigreg-free-dims 0 --spectrum-every 200
--w-t1 1.0 --w-s1 1.0
--v2-cache /home/nvidia/data/physicalai-train-e438721ae894-w120-256x640cyl
--v2-val-cache /home/nvidia/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl
--v2-lru 64 --frame-hfov 120.0 --projection cylindrical --require-parity
--eps-per-batch 4 --max-horizon 60 --steps 30000 --batch 8 --lr 0.0001
--wd 0.05 --clip 1.0 --log-every 50 --save-every 250 --device cuda --seed 0
--dry-steps 2 --dry-batch 2 --dry-k 12
""")

#: The geometry flags the live run carries. Any S-T launch that omits ONE of
#: these builds a different model and `--init-from` dies on SHAPES, which
#: `STAGE_MAY_INTRODUCE` never sees (`load_state_dict(strict=False)` still
#: raises on shape).
GEOMETRY_FLAGS = (
    "--in-channels", "--frame-h", "--frame-w", "--patch", "--enc-dim",
    "--enc-depth", "--enc-heads", "--readout-grid", "--readout-dim",
    "--pred-modern", "--pred-dim", "--pred-depth", "--pred-heads", "--window",
    "--horizons", "--d-tac", "--d-str", "--d-goal-embed", "--adapter-hidden",
    "--n-candidates", "--param-budget", "--f-hidden-tac", "--f-hidden-str",
    "--f-blocks", "--vit5-encoder", "--n-registers", "--plan-steps", "--dt",
    "--a-max", "--kappa-max", "--uplink", "--ema-decay", "--sigreg-slices",
    "--sigreg-free-dims")


def _parse(argv):
    ap = build_parser()
    ap.add_argument("--i-know-this-is-the-control-arm", action="store_true",
                    dest="control_arm_ack")
    return ap.parse_args(argv)


def _st_argv(*, geometry: bool, subframe: bool, tac_goal_cond: bool,
             selector: str = "none", out: str = "/tmp/st"):
    """Build an S-T argv. ``geometry=False`` reproduces exactly what
    `v6_chain.trainer_argv` emits (no geometry flags at all)."""
    av = ["--stage", "S-T", "--out", out, "--steps", "10000", "--batch", "8",
          "--lr", "0.0001", "--n-candidates", "8", "--max-horizon", "60",
          "--plan-wta-eps", "0.05", "--w-t1", "1.0"]
    if tac_goal_cond:
        av += ["--tac-goal-cond"]
    if selector != "none":
        av += ["--selector", selector]
    if geometry:
        keep, i = [], 0
        while i < len(LIVE_ARGV):
            tok = LIVE_ARGV[i]
            if tok in GEOMETRY_FLAGS:
                keep.append(tok)
                j = i + 1
                while j < len(LIVE_ARGV) and not LIVE_ARGV[j].startswith("--"):
                    keep.append(LIVE_ARGV[j])
                    j += 1
                i = j
            else:
                i += 1
        av += keep
    else:
        av += ["--frame-h", "256", "--frame-w", "640"]      # the chain emits these
    if subframe:
        av += ["--v2-subframe", "176x624"]
    return av


def _try(label: str, fn):
    try:
        return {"label": label, "ok": True, "result": fn()}
    except BaseException as exc:                     # SystemExit included
        return {"label": label, "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc)[:900]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-ckpt", default=None,
                    help="the fp16 snapshot pulled from Thor")
    ap.add_argument("--work", required=True, help="scratch dir for the synth ckpt")
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    rep: dict = {"_evidence_class": "MEASURED (ours; real load_stage_init at HEAD)",
                 "allowance_S_T": list(STAGE_MAY_INTRODUCE["S-T"])}

    # -- 0. does HEAD's parser still accept the LIVE argv? -------------------
    # ⛔ THE RESUME-SAFETY CHECK. If a file-ship of HEAD's trainer to Thor makes
    # the live command unparseable, the supervisor's next relaunch dies.
    rep["live_argv_parses_at_HEAD"] = _try("parse LIVE_ARGV", lambda: {
        "stage": _parse(LIVE_ARGV).stage})

    # -- 1. build the S-W stack at the LIVE geometry ------------------------
    torch.manual_seed(0)
    sw_args = _parse(LIVE_ARGV)
    sw = build_stack_from_args(sw_args)
    n_sw = sum(p.numel() for p in sw.parameters())
    rep["sw_params"] = n_sw
    rep["sw_keys"] = len(sw.state_dict())
    synth = work / "sw_synth.pt"
    torch.save({"stack": sw.state_dict(), "step": 30000,
                "config": {"stage": "S-W"}}, synth)
    del sw

    # -- 2. the FOUR S-T candidate launches ---------------------------------
    cases = {
        # what v6_chain.py emits TODAY, verbatim in geometry terms
        "A_chain_emitted": dict(geometry=False, subframe=True,
                                tac_goal_cond=True),
        # the obvious operator fix: paste the live geometry in
        "B_geometry_pasted_subframe_kept": dict(geometry=True, subframe=True,
                                                tac_goal_cond=True),
        # geometry + drop the subframe
        "C_geometry_no_subframe": dict(geometry=True, subframe=False,
                                       tac_goal_cond=True),
        # ...and without the F-1 port, for contrast
        "D_geometry_no_subframe_no_port": dict(geometry=True, subframe=False,
                                               tac_goal_cond=False),
    }
    rep["cases"] = {}
    for name, kw in cases.items():
        def run(kw=kw, name=name):
            torch.manual_seed(0)
            st = build_stack_from_args(_parse(_st_argv(**kw)))
            out = {"st_params": sum(p.numel() for p in st.parameters()),
                   "st_keys": len(st.state_dict())}
            out["load"] = load_stage_init(st, synth, stage="S-T")
            return out
        rep["cases"][name] = _try(name, run)

    # -- 3. THE REAL ARTIFACT ------------------------------------------------
    if a.real_ckpt and Path(a.real_ckpt).exists():
        p = Path(a.real_ckpt)
        h = hashlib.md5()
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        rep["real_ckpt"] = {"path": str(p), "bytes": p.stat().st_size,
                            "md5_local": h.hexdigest()}
        def run_real():
            torch.manual_seed(0)
            st = build_stack_from_args(
                _parse(_st_argv(geometry=True, subframe=False,
                                tac_goal_cond=True)))
            return load_stage_init(st, str(p), stage="S-T")
        rep["real_transition"] = _try("REAL fp16 snapshot -> S-T", run_real)

        def run_real_sel():
            torch.manual_seed(0)
            st = build_stack_from_args(
                _parse(_st_argv(geometry=True, subframe=False,
                                tac_goal_cond=True, selector="goal")))
            return load_stage_init(st, str(p), stage="S-T")
        rep["real_transition_selector_goal"] = _try(
            "REAL fp16 snapshot -> S-T(--selector goal)", run_real_sel)
    else:
        rep["real_ckpt"] = {"_read": "not present — synthetic arm only"}

    Path(a.out_json).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1)[:8000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
