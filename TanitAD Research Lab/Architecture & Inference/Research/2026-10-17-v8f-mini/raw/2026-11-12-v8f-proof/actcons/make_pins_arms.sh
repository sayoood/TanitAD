#!/bin/bash
# builds PINS.txt and ARMS_ACTCONS.txt from the files on disk (sha256 read from work/tree_base and work/tree_patch)
cd /f/TanitAD-scratch/v8proof/actcons || exit 1
T=/workspace/v8proof/tree_actcons
h() { sha256sum "$1" | cut -d' ' -f1; }
CH="stack/scripts/v8f_mini_train.py"
NEW="stack/tanitad/models/act_cons.py stack/tests/test_v8f_act_cons.py"
SAME="stack/tanitad/models/v8f_mini.py stack/tanitad/models/hjepa.py stack/tanitad/models/v8f_beta.py stack/scripts/v8f_t1_eval.py stack/tanitad/data/v8f_mini_data.py stack/tanitad/data/rule_relabel.py stack/tanitad/data/dist_rule.py stack/tanitad/data/rule_own.py stack/scripts/lead_dump.py stack/tests/test_v8f_rule_own.py stack/tests/test_v8f_rule_own_l1.py stack/tests/test_v8f_rule_relabel.py stack/tests/test_v8f_lead_sup.py stack/tests/test_v8f_lead_cond.py stack/tests/test_v8f_target_levers.py stack/tests/test_v8f_cell_predictor.py stack/tests/test_v8f_motion_levers.py"
{
echo "# PINS: sha256 of the files of work/tree_patch (paths relative to the repo root; patched/ holds the changed and new ones without the leading stack/)."
echo "# base = the tree of LEADCOND (leadcond/work/tree_patch: trainer 85035ca4..., model 2e662208..., verified on the copy before the work). changed = base sha256 -> patched sha256"
for f in $CH; do echo "changed  $f  $(h work/tree_base/$f) -> $(h work/tree_patch/$f)"; done
for f in $NEW; do echo "new      $f  $(h work/tree_patch/$f)"; done
echo "# unchanged files the chain pins (same sha256 in base and patched)"
for f in $SAME; do a=$(h work/tree_base/$f); b=$(h work/tree_patch/$f); [ "$a" = "$b" ] || { echo "MISMATCH $f" >&2; exit 2; }; echo "same     $f  $a"; done
} > PINS.txt
LSX="levers.wm_state=grid levers.wm_z1_width=640 levers.wm_cell_dim=16 levers.wm_l1_pred=flat levers.wm_l1_pred_width=256 levers.wm_l1_act=acc_yaw levers.wm_l1_futtok_k=0 levers.w_wm_futtok=0 levers.lead_sup=on levers.w_lead_sup=1 levers.lead_hist=5 levers.lead_sup_detach=off"
{
echo "# ACTCONS (lever --w-act-cons): fine-tunes from the banked LS checkpoints (leadsup_a/LS_s0, LS_s1: --init-ckpt, fresh optimiser), batch 64, peak lr 1e-4, warm-up 100, cosine to a tenth, 6,000 steps, all on the LS flags --wm-state grid --wm-cell-dim 16 --lead-sup on --w-lead-sup 1.0 (the argv of the LFT arms of ARMS_LEADCOND.txt)."
echo "# LA = LFT + --w-act-cons 1.0 (the named arm), LA01 = LFT + --w-act-cons 0.1 (dose arm, names nothing). --act-cons-max stays at its default 1.5 (not on the argv; the expectation string levers.act_cons_max=1.5 reads it back from the BUILT record). The ARM field's seed column is the fine-tune --seed (base seed, as LFT)."
echo "# Expectation strings: gate_extra compares whole floats as ints, so --w-act-cons 1.0 reads levers.w_act_cons=1; 0.1 reads levers.w_act_cons=0.1. The two levers.* keys are present for the lever arms only (the default has no such key). If the weight or the max changes, edit BOTH the argv token and the expectation token."
echo "# Departures from the LFT line besides the flag: none. The term is TRAINING ONLY: the checkpoint cfg, the state-dict keys and every inference output are the LFT arm's kind; the selfcheck adds act_cons_probe (term, n_valid, gradient norms, exact zeros) and train_log.jsonl adds act_cons and act_cons_n."
for f in stack/tanitad/models/v8f_mini.py stack/tanitad/models/act_cons.py stack/tanitad/models/v8f_beta.py stack/tanitad/models/hjepa.py stack/scripts/v8f_mini_train.py stack/scripts/v8f_t1_eval.py stack/tanitad/data/v8f_mini_data.py stack/tanitad/data/rule_relabel.py stack/tanitad/data/dist_rule.py stack/tanitad/data/rule_own.py stack/tests/test_v8f_rule_own.py stack/tests/test_v8f_rule_own_l1.py stack/tests/test_v8f_lead_cond.py stack/tests/test_v8f_act_cons.py; do
  echo "PIN|$T|$f|$(h work/tree_patch/$f)"
done
arm() { # name seed extra_argv extra_expect
  CK="/workspace/v8proof/runs/leadsup_a/LS_s$2/ckpt_last.pt"
  echo "ARM|$1_s$2|$T|HIER|$2|540:600|/workspace/v8proof/data/cache_pod|--wm-state grid --wm-cell-dim 16 --lead-sup on --w-lead-sup 1.0 --init-ckpt $CK --steps 6000 --lr 1e-4 --warmup 100$3|$LSX$4 info.n_clips=4512"
}
for sd in 0 1; do arm LA $sd " --w-act-cons 1.0" " levers.w_act_cons=1 levers.act_cons_max=1.5"; done
for sd in 0 1; do arm LA01 $sd " --w-act-cons 0.1" " levers.w_act_cons=0.1 levers.act_cons_max=1.5"; done
} > ARMS_ACTCONS.txt
echo built
