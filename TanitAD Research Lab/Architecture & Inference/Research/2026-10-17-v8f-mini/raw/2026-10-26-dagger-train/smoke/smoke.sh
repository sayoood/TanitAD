#!/usr/bin/env bash
# CPU smoke of the DAgger lever through the trainer's REAL command line (no pytest, no monkeypatch): synthetic world (T = 60, the trainer's own), a trainer-written checkpoint, a bank made by
# v8f_dagger_bank.py gen/merge from it, then the in-process launch gate (--selfcheck-only), a few training steps with the lever on, DAG-0, and the refusals.
# usage: smoke.sh <tree root> <work dir> [python]        (the pattern of C:/Users/Admin/orct1: a few-step run on synthetic data, every artifact kept)
set -u
TREE=$1; W=$2; PY=${3:-C:/Users/Admin/venvs/tanitad/Scripts/python.exe}
export CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
TRAIN="$TREE/stack/scripts/v8f_mini_train.py"; BANKER="$TREE/stack/scripts/v8f_dagger_bank.py"
HERE="$(cd "$(dirname "$0")" && pwd)"
COMMON="--synthetic --batch 8 --d 32 --n-q 2 --log-every 1 --warmup 2 --ckpt-every 100 --threads 1 --kin-mode yaw --seed 5 --arm M2b"
rm -rf "$W"; mkdir -p "$W"; cd "$W"
echo "== 1. a trainer-written checkpoint (2 steps, the trainer's own synthetic world)"
$PY "$TRAIN" --out init $COMMON --steps 2 > init.log 2>&1; echo "rc=$?"; tail -2 init.log
$PY "$HERE/tame_ckpt.py" init/ckpt_last.pt init_tame.pt
echo '{"argv": ["--arm", "M2b", "--seed", "5", "--out", "x", "--kin-mode", "yaw", "--batch", "8", "--d", "32", "--n-q", "2"]}' > argv.json
echo "== 2. the bank (gen + merge, one shard)"
W0=init/_synthetic
$PY "$BANKER" gen --ckpt init_tame.pt --m2b-argv argv.json --cache-dir $W0 --tracks $W0/native_tracks.pt --join-train $W0/join_train.npz --eval-grid $W0/eval_grid_windows.json --out-dir bank --batch 8 --replay-frac 0.1 > bank_gen.log 2>&1; echo "rc=$?"; tail -3 bank_gen.log
BANK_SHA=$(sha256sum bank/bank.json | cut -d' ' -f1); echo "bank.json sha256 $BANK_SHA"
echo "== 3. the in-process launch gate on the bank (--selfcheck-only): G-DVB + K-B2/K-B3/COVERAGE + dagger_probe"
$PY "$TRAIN" --out gate $COMMON --init-ckpt init_tame.pt --dagger-p 0.1 --dagger-bank bank --dagger-bank-sha256 $BANK_SHA --dagger-seed 0 --steps 20 --selfcheck-only > gate.log 2>&1; echo "rc=$?"; tail -c 400 gate.log; echo
echo "== 4. DAG: 20 steps with the lever on (p = 0.5 so that a 20-step run counts something)"
$PY "$TRAIN" --out dag $COMMON --init-ckpt init_tame.pt --dagger-p 0.5 --dagger-bank bank --dagger-bank-sha256 $BANK_SHA --dagger-seed 0 --steps 20 > dag.log 2>&1; echo "rc=$?"; tail -2 dag.log
echo "== 5. DAG-0: the same fine-tune naming only --dagger-p 0"
$PY "$TRAIN" --out dag0 $COMMON --init-ckpt init_tame.pt --dagger-p 0 --steps 20 > dag0.log 2>&1; echo "rc=$?"; tail -2 dag0.log
echo "== 6. refusals (each must be rc != 0 and write no selfcheck.json)"
$PY "$TRAIN" --out ref_ckpt $COMMON --init-ckpt init/ckpt_last.pt --dagger-p 0.1 --dagger-bank bank --dagger-bank-sha256 $BANK_SHA --steps 2 > ref_kb2.log 2>&1; echo "K-B2 (another init checkpoint) rc=$?"; grep -o "ZZDVB-REFUSEDZZ.*" ref_kb2.log | cut -c1-260
$PY "$TRAIN" --out ref_seed $COMMON --dagger-seed 3 --steps 2 > ref_seed.log 2>&1; echo "seed at p=0 rc=$?"; grep -o "ZZDVB-REFUSEDZZ.*" ref_seed.log | cut -c1-260
$PY "$TRAIN" --out ref_p0bank $COMMON --init-ckpt init_tame.pt --dagger-p 0 --dagger-bank bank --steps 2 > ref_p0bank.log 2>&1; echo "bank at p=0 rc=$?"; grep -o "ZZDVB-REFUSEDZZ.*" ref_p0bank.log | cut -c1-260
$PY "$TRAIN" --out ref_ego $COMMON --init-ckpt init_tame.pt --dagger-p 0.1 --dagger-bank bank --dagger-bank-sha256 $BANK_SHA --ego-perturb-p 0.3 --steps 2 > ref_ego.log 2>&1; echo "ego+dagger rc=$?"; grep -o "ZZDVB-REFUSEDZZ.*" ref_ego.log | cut -c1-260
$PY "$TRAIN" --out ref_nosha $COMMON --init-ckpt init_tame.pt --dagger-p 0.1 --dagger-bank bank --steps 2 > ref_nosha.log 2>&1; echo "no --dagger-bank-sha256 at p>0 rc=$?"; grep -o "ZZDVB-REFUSEDZZ.*" ref_nosha.log | cut -c1-260
$PY "$TRAIN" --out ref_p0sha $COMMON --init-ckpt init_tame.pt --dagger-p 0 --dagger-bank-sha256 $BANK_SHA --steps 2 > ref_p0sha.log 2>&1; echo "--dagger-bank-sha256 at p=0 rc=$?"; grep -o "ZZDVB-REFUSEDZZ.*" ref_p0sha.log | cut -c1-260
$PY "$TRAIN" --out ref_wrongsha $COMMON --init-ckpt init_tame.pt --dagger-p 0.1 --dagger-bank bank --dagger-bank-sha256 0000000000000000000000000000000000000000000000000000000000000000 --steps 2 > ref_wrongsha.log 2>&1; echo "wrong --dagger-bank-sha256 rc=$?"; grep -o "ZZDVB-REFUSEDZZ.*" ref_wrongsha.log | cut -c1-260
for d in ref_ckpt ref_seed ref_p0bank ref_ego ref_nosha ref_p0sha ref_wrongsha; do ls $d/selfcheck.json 2>/dev/null && echo "UNEXPECTED selfcheck.json in $d"; done
echo "== 7. the eval-clip regression: a bank with one eval clip injected"
$PY - <<PYEOF
import json, shutil, sys, os
sys.path[:0] = ["$TREE/stack"]
from tanitad.data import v8f_dagger as G
shutil.rmtree("bank_evil", ignore_errors=True); os.makedirs("bank_evil")
man = json.load(open("bank/bank.json"))
for s in man["shards"]:
    arrays, header = G.load_shard("bank/" + s["file"])
    k0 = sorted(header["clips"], key=int)[0]
    print("renaming", header["clips"][k0], "->", "syn-eval-000")
    header["clips"][k0] = "syn-eval-000"
    G.save_shard("bank_evil/" + s["file"], arrays, header)
    s["file_sha256"] = G.sha256_file("bank_evil/" + s["file"]); s["content_sha256"] = G.content_sha256(arrays)
open("bank_evil/bank.json", "w", newline="\n").write(json.dumps(man, sort_keys=True, indent=1, default=G._jsonable) + "\n")
PYEOF
EVIL_SHA=$(sha256sum bank_evil/bank.json | cut -d' ' -f1)
$PY "$TRAIN" --out ref_eval $COMMON --init-ckpt init_tame.pt --dagger-p 0.1 --dagger-bank bank_evil --dagger-bank-sha256 $EVIL_SHA --steps 2 > ref_eval.log 2>&1; echo "eval clip rc=$?"; grep -o "ZZDVB-REFUSEDZZ.*" ref_eval.log | cut -c1-300
echo "== 8. the records"
$PY - <<PYEOF
import json
for d in ("dag", "dag0"):
    s = json.load(open(d + "/summary.json"))
    print(d, {k: s[k] for k in ("dagger_n", "dagger_eligible", "dagger_rows", "dagger_nav_kept")}, "levers.dagger_p", s["levers"]["dagger_p"], "levers.dagger_bank_sha256", s["levers"]["dagger_bank_sha256"], "declared", s["levers"]["dagger_bank_sha256_declared"])
sc = json.load(open("gate/selfcheck.json"))
pr = sc["dagger_probe"]
print("gate dagger_probe", {k: pr[k] for k in ("pass", "n_rows", "n_eligible", "n_selected_at_p1", "loss_diff", "grad_diff_all", "control_repeat_grad_diff", "control_p0_same_object", "control_p0_loss_diff", "control_p0_grad_diff", "control_prior_grad_diff", "control_wm_grad_diff", "selected_rows_bit_identical_exempt", "rc_max_abs_err")})
print("gate controls", json.dumps(sc["info"]["dagger"]["kb3"]), json.dumps(sc["info"]["dagger"]["coverage"]))
print("bank.json sha256 == levers.dagger_bank_sha256 == levers.dagger_bank_sha256_declared:", "$BANK_SHA" == sc["levers"]["dagger_bank_sha256"] == sc["levers"]["dagger_bank_sha256_declared"])
PYEOF
echo SMOKE-DONE
