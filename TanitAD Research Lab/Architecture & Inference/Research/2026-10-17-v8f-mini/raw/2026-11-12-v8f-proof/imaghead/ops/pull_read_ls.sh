#!/usr/bin/env bash
# pull_read_ls.sh : dev box. Pulls the imagined-lead dumps of LS_s0 and LS_s1 from the pod (binary-clean, sha256 compared) and runs the read of card PREREG_V8F_PROOF_ADD36.
# usage: bash pull_read_ls.sh <sha256 dump s0> <sha256 json s0> <sha256 dump s1> <sha256 json s1>
set -u
H=/f/TanitAD-scratch/v8proof/imaghead; C=/c/Users/Admin/v8pod/pcat.sh; PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
R=/workspace/v8proof/runs/imag_z/dump; D=$H/dump; mkdir -p "$D" "$H/read"
want=("$1" "$2" "$3" "$4"); i=0
for f in DUMP_LS_s0.npz DUMP_LS_s0.npz.json DUMP_LS_s1.npz DUMP_LS_s1.npz.json; do
  timeout 300 bash $C $R/$f "$D/$f" || { echo "ZZ PULL FAILED $f"; exit 1; }
  got=$(sha256sum "$D/$f" | cut -c1-64)
  [ "$got" = "${want[$i]}" ] || { echo "ZZ SHA MISMATCH $f: $got"; exit 1; }
  echo "pulled $f $(stat -c %s "$D/$f") bytes, sha256 equal"; i=$((i+1))
done
export CUDA_VISIBLE_DEVICES=-1
cd "$H" || exit 1
$PY imag_read.py read --dump "$D/DUMP_LS_s0.npz" "$D/DUMP_LS_s1.npz" --arm LS --seeds 0,1 --out-dir "$H/read/LS" > "$H/read/LS.log" 2>&1
echo "read rc $? (0 = controls pass, 2 = a control failed and no name is given)"
tail -n 12 "$H/read/LS.log" | cut -c1-400
