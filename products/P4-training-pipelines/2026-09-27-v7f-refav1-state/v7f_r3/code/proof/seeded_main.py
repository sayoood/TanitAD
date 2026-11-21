"""Run ``train_v6_staged.py`` main() with the GLOBAL RNGs seeded first.

⚠️ WHY: ``dry_run()`` does not seed the global torch RNG before ``build_stack_from_args`` (``train()``
does, at ``torch.manual_seed(a.seed)``), so two processes running the SAME tree and the SAME argv
build different random weights -- MEASURED: two runs of the untouched c36b6ddd tree differ
(``raw/cli_off_dryrun_compare.txt``). A cross-tree comparison is only meaningful with the seed fixed
OUTSIDE the trainer, which is what this wrapper does (it changes nothing inside either tree).

usage: python seeded_main.py <trainer.py> <argv...>
"""
import random
import runpy
import sys

import torch

torch.manual_seed(0)
random.seed(0)
trainer = sys.argv[1]
sys.argv = [trainer] + sys.argv[2:]
try:
    runpy.run_path(trainer, run_name="__main__")
except SystemExit as e:
    code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    sys.exit(code)
