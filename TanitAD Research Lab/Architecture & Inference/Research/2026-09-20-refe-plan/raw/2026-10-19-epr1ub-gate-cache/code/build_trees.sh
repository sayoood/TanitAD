#!/bin/bash
# usage: build_trees.sh  -> D:/eu1/base_ub (deploy layout of the tip), D:/eu1/base_c2 (sel-rank code dir at the tip)
cd C:/Users/Admin/tanitad-push
R="TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw"
exp_flat() { # <commit> <repo dir> <dest> [names...]
  local c="$1" d="$2" dest="$3"; shift 3
  mkdir -p "$dest"
  if [ $# -eq 0 ]; then set -- $(git ls-tree --name-only "$c:$d" | tr '\n' ' '); fi
  for n in "$@"; do
    if git cat-file -e "$c:$d/$n" 2>/dev/null && [ "$(git cat-file -t "$c:$d/$n")" = blob ]; then git cat-file -p "$c:$d/$n" > "$dest/$n"; fi
  done
}
# EP-R1ub deploy layout: landed parents byte-exact from the landed commit, staged files from the tip
exp_flat 1afc54f6fa5ee865f2966e3c0a7b3141f24ef0c1 "$R/2026-10-18-ep-r1/code" D:/eu1/base_ub epr1.py epr1_gate.py zero_tail.py test_epr1.py epr1_dev.py template_pins.json template_pins_make.py
exp_flat FETCH_HEAD "$R/2026-10-19-ep-r1ub/code" D:/eu1/base_ub epr1ub.py epr1ub_gate.py test_epr1ub.py test_epr1ub_gate.py launch_epr1ub.ps1
exp_flat FETCH_HEAD "$R/2026-10-18-sel-rank/code" D:/eu1/base_c2
ls D:/eu1/base_ub; ls D:/eu1/base_c2 | wc -l
