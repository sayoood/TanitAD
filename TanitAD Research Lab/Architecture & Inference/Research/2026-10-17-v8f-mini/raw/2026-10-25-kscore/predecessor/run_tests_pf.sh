#!/bin/bash
# KSCORE chain + tests after MM's pre-flight PROCEED change, on a rig cut from tip 25ca43b7 (+ FLAT's chain from the scratch tree; FLAT has not landed)
K=/c/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/kscore_pred
SC=/c/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
R=$K/rig_pf; mkdir -p "$R" && cd "$R" || exit 1
git -C C:/Users/Admin/tanitad-push -c safe.directory='*' -c core.autocrlf=false archive --format=tar 25ca43b750363f5d5bd9f0afb31d938714c0c660 -- stack/pyproject.toml stack/scripts stack/tanitad stack/tests/conftest.py taniteval/taniteval "Project Steering/PREREG_V8F_KSCORE.md" "$PK/code" "$PK/raw/2026-10-25-khyp" | tar -xf - || exit 2
cp "$K/chain_v8f_kscore.sh" stack/scripts/ && cp "$K/test_v8f_kscore.py" stack/tests/ && mkdir -p "$PK/code/box2_ops" && cp "$SC/flat/$PK/code/box2_ops/chain_v8f_flat.sh" "$PK/code/box2_ops/" || exit 3
CUDA_VISIBLE_DEVICES=-1 PYTHONUTF8=1 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I -m pytest stack/tests/test_v8f_kscore.py --basetemp="$K/pt_tmp_pf" -p no:cacheprovider -q -rs > "$K/tests_pf.out" 2>&1
echo "pytest rc=$?"; tail -3 "$K/tests_pf.out"
