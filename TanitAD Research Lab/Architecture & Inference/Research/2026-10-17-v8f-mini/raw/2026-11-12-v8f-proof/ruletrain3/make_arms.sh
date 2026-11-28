#!/bin/bash
# builds PINS.txt and ARMS_RULEOWNL1.txt from the files on disk (sha256 read from patched/ and from ../ruletrain2/PINS.txt)
cd /f/TanitAD-scratch/v8proof/ruletrain3 || exit 1
T=/workspace/v8proof/tree_ruleown2
sh() { sha256sum "$1" | cut -d' ' -f1; }
TR=$(sh patched/scripts/v8f_mini_train.py); RO=$(sh patched/tanitad/data/rule_own.py); TE=$(sh patched/tests/test_v8f_rule_own.py); TL=$(sh patched/tests/test_v8f_rule_own_l1.py)
get() { grep -E "^same +$1 " ../ruletrain2/PINS.txt | awk '{print $3}'; }
{
echo "# PINS: sha256 of the files of work/tree_patch (paths relative to the repo root; patched/ holds the changed and new ones without the leading stack/)."
echo "# base = the tree of card ADD34 (ruletrain2/work/tree_patch: trainer a5806960..., rule_own.py e4ad14ac... verified before the work). changed = base sha256 -> patched sha256"
echo "changed  stack/scripts/v8f_mini_train.py  a5806960d09625895970a7591dc2901151f36968e011619ed01d0423fbbd370b -> $TR"
echo "changed  stack/tanitad/data/rule_own.py  e4ad14ac2c43bc2308e325f9888821c73e56cafbda102ca8ed4e0cf827130480 -> $RO"
echo "changed  stack/tests/test_v8f_rule_own.py  a3364480ae61f5d9563a4316360082a94a7090fec2248dad2be59bca14ad09be -> $TE"
echo "new      stack/tests/test_v8f_rule_own_l1.py  $TL"
echo "# unchanged files the chain pins (same sha256 in base and patched)"
grep -E "^same " ../ruletrain2/PINS.txt | grep -v "stack/tanitad/data/rule_own.py"
} > PINS.txt
EXP_BASE="levers.wm_state=grid levers.wm_z1_width=640 levers.wm_cell_dim=16 levers.wm_l1_pred=flat levers.wm_l1_pred_width=256 levers.wm_l1_act=acc_yaw levers.wm_l1_futtok_k=0 levers.w_wm_futtok=0 levers.lead_sup=off"
{
echo "# RULEOWNL1 (note PREREG_V8F_PROOF_ADD34_N1): fine-tunes from the banked arm-G checkpoints (--init-ckpt, strict load, fresh optimiser), batch 64, peak lr 1e-4, warm-up 100, cosine to a tenth."
echo "# GOL = --w-rule-own 1.0 --rule-own-form l1 (the named arm), GOL4 = --w-rule-own 4.0 --rule-own-form l1 (dose arm), 2,000 steps, the argv of GOW / GOW4 of card ADD34 plus --rule-own-form l1. Controls and floor: GFT / GFTb of card ADD32, GOW / GOW4 of card ADD34 (same bases, same steps; this tree with the flag absent is bit for bit the tree they were trained on)."
echo "# The ARM field's seed column is the fine-tune --seed (GFT convention: base seed). Labels come from the label join (--join-train): the gate refuses a launch whose probe batch has no label lead (count 0)."
echo "# Expectation strings: gate_extra compares whole floats as ints, so --w-rule-own 1.0 reads levers.w_rule_own=1; levers.rule_own_form=l1 is present for the linear form only (the default form has no such key). info.rule_own is read from the first real-data selfcheck, not predeclared."
for f in stack/tanitad/models/v8f_mini.py stack/tanitad/models/v8f_beta.py stack/tanitad/models/hjepa.py stack/scripts/v8f_mini_train.py stack/scripts/v8f_t1_eval.py stack/tanitad/data/v8f_mini_data.py stack/tanitad/data/rule_relabel.py stack/tanitad/data/dist_rule.py stack/tanitad/data/rule_own.py stack/tests/test_v8f_rule_own.py stack/tests/test_v8f_rule_own_l1.py; do
  case $f in
    stack/scripts/v8f_mini_train.py) s=$TR;; stack/tanitad/data/rule_own.py) s=$RO;; stack/tests/test_v8f_rule_own.py) s=$TE;; stack/tests/test_v8f_rule_own_l1.py) s=$TL;;
    stack/tanitad/data/rule_relabel.py|stack/tanitad/data/dist_rule.py) s=$(get $f);;
    *) s=$(get $f);;
  esac
  echo "PIN|$T|$f|$s"
done
for sd in 0 1; do
  CK="/workspace/v8proof/runs/spatial/G_s$sd/ckpt_last.pt"
  A="--wm-state grid --wm-cell-dim 16 --init-ckpt $CK"
  echo "ARM|GOL_s$sd|$T|HIER|$sd|540:600|/workspace/v8proof/data/cache_pod|$A --steps 2000 --lr 1e-4 --warmup 100 --w-rule-own 1.0 --rule-own-form l1|$EXP_BASE levers.w_rule_own=1 levers.rule_own_form=l1 info.n_clips=4512"
done
for sd in 0 1; do
  CK="/workspace/v8proof/runs/spatial/G_s$sd/ckpt_last.pt"
  A="--wm-state grid --wm-cell-dim 16 --init-ckpt $CK"
  echo "ARM|GOL4_s$sd|$T|HIER|$sd|540:600|/workspace/v8proof/data/cache_pod|$A --steps 2000 --lr 1e-4 --warmup 100 --w-rule-own 4.0 --rule-own-form l1|$EXP_BASE levers.w_rule_own=4 levers.rule_own_form=l1 info.n_clips=4512"
done
} > ARMS_RULEOWNL1.txt
