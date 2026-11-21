"""kindx_eval.py -- PREREG_V8F_KINDX driver: ONE eval of X = M2b-mixo_s0 on the dev-box CPU, through the tip's own `v8f_mini_eval.py run` (imported, unmodified).

  python -I kindx_eval.py <name>      name in own | only-<kind> | union | all | none | loo-<kind>

The queue lines are KIND's (`v8fkind_queue_10250717.cmd`, :kindown / :kindorc) with the paths swapped and the checkpoint swapped for X:
  own      run --ckpt X --tag T --tac pred --project --families          (the banked standard eval E0's flags; `--tac pred` is the default)
  <spec>   run --ckpt X --tag T --tac oracle --tac-groups <spec> --tac-alpha 1
           spec: <kind> | constraints (union) | all (full GT) | all-<kind> (loo, beside only)
  none     the IDENTITY ORACLE: the same oracle path with an EMPTY group selection (GT in no kind). `v8f_mini_eval` refuses alpha 0 and has no empty spec, so the ONE patch here is
           `v8f_mini.code_group_index('none') -> []`; with grp_idx == [] the eval's own code does `sel = zeros(CODE_DIM); sel[[]] = True; code = where(sel, code, code0)` = code0 exactly.
The model code is the tip eeb2043e EXPORT under OUT/code (asserted); never the stale editable `tanitad` on G:. CPU only, torch threads capped."""
from __future__ import annotations

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "6")

SC = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad"
OUT = f"{SC}/kindx"
CODE = f"{OUT}/code"
SF = f"{SC}/sfresh"                                   # the eval-141 inputs (cache, v9 release, lead, speed sidecar, P10 prior, native tracks): md5-verified against box 2 by SFRESH
CKPT = "C:/Users/Admin/box2_ship/logs/v8f_mixo/runs/M2b_s0_mixo/ckpt_last.pt"
CKPT_SHA = "c6a628762790b4fc3f7634867fc15b8167b035692ff518203c9fc0b3c126250d"
TAGBASE = "M2b-mixo_s0-kindx"
KINDS = ("speed", "angle", "tte", "timegap", "geom", "leadgap")

for _p in (f"{CODE}/taniteval", f"{CODE}/stack"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("KINDX_THREADS", "6")))
import importlib.util  # noqa: E402

from tanitad.models import v8f_mini as M  # noqa: E402

if not os.path.abspath(M.__file__).replace("\\", "/").startswith(CODE):
    raise ImportError(f"tanitad imported from {M.__file__}, not from the tip export {CODE}")

_orig = M.code_group_index


def _cgi(spec):
    return [] if spec == "none" else _orig(spec)


M.code_group_index = _cgi

spec = importlib.util.spec_from_file_location("v8f_mini_eval_kindx", f"{CODE}/stack/scripts/v8f_mini_eval.py")
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)


def argv_for(name: str) -> list:
    tag = f"{TAGBASE}-{name}"
    common = ["run", "--ckpt", CKPT, "--tag", tag,
              "--v9-eval", f"{SF}/inputs/v9/v9_labels_eval139.npz", "--lead-eval", f"{SF}/inputs/v9/lead_eval139.npz",
              "--speed-eval", f"{SF}/inputs/v9/refcv6_speed_max_v8_eval.jsonl", "--kin-mode", "yaw", "--p10", f"{SF}/inputs/p10/p10_prior.pt",
              "--out-dir", f"{OUT}/eval", "--eval-cache", f"{SF}/cache", "--tracks", f"{SF}/inputs/native_tracks.pt"]
    if name == "own":
        return common + ["--tac", "pred", "--project", "--families"]
    if name in KINDS:
        groups = name
    elif name == "union":
        groups = "constraints"
    elif name in ("all", "none"):
        groups = name
    elif name.startswith("loo-") and name[4:] in KINDS:
        groups = "all-" + name[4:]
    else:
        raise SystemExit(f"unknown eval name {name!r}")
    # the tag keeps KIND's `only-` prefix for single kinds
    if name in KINDS:
        common[common.index("--tag") + 1] = f"{TAGBASE}-only-{name}"
    return common + ["--tac", "oracle", "--tac-groups", groups, "--tac-alpha", "1"]


if __name__ == "__main__":
    import hashlib
    name = sys.argv[1]
    argv = argv_for(name)
    os.makedirs(f"{OUT}/eval", exist_ok=True)
    os.makedirs(f"{OUT}/logs", exist_ok=True)
    h = hashlib.sha256()
    with open(CKPT, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    if h.hexdigest() != CKPT_SHA:
        raise SystemExit(f"REFUSED: checkpoint sha256 {h.hexdigest()} != {CKPT_SHA}")
    with open(f"{OUT}/logs/argv_{argv[argv.index('--tag') + 1]}.txt", "w") as f:
        f.write("\n".join([sys.executable, f"{CODE}/stack/scripts/v8f_mini_eval.py"] + argv) + "\n")
    rc = E.main(argv)
    print("EVAL-RC", rc, "tanitad from", M.__file__)
    sys.exit(int(rc or 0))
