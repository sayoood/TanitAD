#!/usr/bin/env bash
# The six repo tests that pin the queue's label set, run on a scratch OVERLAY of the tip (6cd741b6: stack/ + the v8f-mini code dir) with
#   (1) the PATCHED queue + the EDITED tests        -> must PASS
#   (2) the PATCHED queue + the UNEDITED tests      -> must go RED on the label sets (the negative control: the test edit is needed, and the tests do see the new label)
# Nothing in the repo is touched; the overlay lives under F:/TanitAD-scratch/dagorc_rebase_tmp/overlay.  Markers: ZZLABEL-DONE | ZZLABEL-STOP.
HERE="$(cd "$(dirname "$0")" && pwd)"
S=F:/TanitAD-scratch/dagorc_rebase_tmp; O=$S/overlay
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
T=6cd741b63b3b629282d48f3b88d86f99264f9c9f
PYV=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
export CUDA_VISIBLE_DEVICES=-1
n=$($PYV C:/Users/Admin/orcread/pytest_runs.py 2>/dev/null | tail -1)
[ "${n:-9}" -lt 2 ] || { echo "ZZLABEL-STOP pytest_runs shows $n running (must be < 2)"; exit 1; }
rm -rf "$O" && mkdir -p "$O"
git -C C:/Users/Admin/tanitad-push -c safe.directory='*' -c core.autocrlf=false archive --format=tar $T stack taniteval "$PK/code" "$PK/raw/eval_grid_windows.json" | tar -xf - -C "$O" || { echo "ZZLABEL-STOP archive"; exit 1; }
cp "$HERE/stage/$PK/code/v8f_mini_box2_queue.cmd" "$O/$PK/code/v8f_mini_box2_queue.cmd"
TESTS="stack/tests/test_v8f_flat.py stack/tests/test_v8f_oacc.py stack/tests/test_v8f_kind.py stack/tests/test_v8f_mixo.py stack/tests/test_v8f_lora_ar2.py stack/tests/test_v8f_orc_chain.py"
mkdir -p $S/pt_base
echo "== (2) NEGATIVE CONTROL first: the patched queue with the UNEDITED tests"
(cd "$O/stack" && $PYV -m pytest -q -p no:cacheprovider --basetemp=$S/pt_base/neg ${TESTS//stack\//} 2>&1 | tail -25) > "$HERE/logs/label_tests_negative.out"
tail -8 "$HERE/logs/label_tests_negative.out"
cp "$HERE"/stage/stack/tests/test_v8f_*.py "$O/stack/tests/"
echo "== (1) the EDITED tests"
(cd "$O/stack" && $PYV -m pytest -q -p no:cacheprovider --basetemp=$S/pt_base/pos ${TESTS//stack\//} 2>&1 | tail -25) > "$HERE/logs/label_tests_edited.out"
tail -8 "$HERE/logs/label_tests_edited.out"
echo "ZZLABEL-DONE"
