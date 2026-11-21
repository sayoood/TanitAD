# POD_STEPS_A1: complete HB-s0's h48x and run HB-SHUF (companion to PREREG_VDAC_P6_A5_A1_DRAFT.md)

`PK` = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`. `<LANDED>` = the commit that carries the six changed files. No IP, port, pod id or token appears here. Steps B and D run on the pod, A and F on the dev box.

## A. Dev box: land, re-ship (the whole tree, as POD_RUN_PLAN section 2)
Changed files (all under `PK/refe/`): `scorer_finetune.py`, `vdac_a5_layer.py`, `gate_checks.py`, `vdac_pod.sh`, `vdac_pause_p7.sh`, `test_vdac_hb_h48x.py` (new). Land through the lander with `base=` blobs of the tip (these are not registers). `git ls-remote origin <branch>` equals the local tip BEFORE building.
Then POD_RUN_PLAN Step A unchanged: `git archive <LANDED>` of `refe code splits` + the six `eval/` files, `launch_gate.py ship-manifest ... --out SHIP_MANIFEST.json` (`ZZSHIP_OK`), tar with LF files. **Regenerate the ship list** (one file was added): `python refe/p7_ship_list.py derive --pkg "$P" --out <list> --profile refe_vdac_p6`, ship it with the tar.
Before shipping, on the dev box with `CUDA_VISIBLE_DEVICES=-1`: `test_vdac_hb_h48x.py` (`ZZTEST_PASS 49`), `test_vdac_hb.py` (133), `test_vdac_hbmem.py` (83), `test_vdac_gate.py`, `test_launch_gate.py --static --no-banked`; and the ~69 launch_gate-reading tests (a pin edit moves profile fingerprints). Do NOT run tests on the pod while P7 trains.

## B. Pod (no GPU): keep the HB-s0 data, place the new tree
HB-s0's outputs live in the DATA root, not in the package dir: do not touch `/workspace/data/refe_vdac/hb-s0`.
```
D=/workspace/data/refe_vdac/hb-s0
sha256sum $D/aux_drv.pt $D/scorer_states_last.pt $D/model_sft_A.pt $D/model_sft_B.pt   # record: these go into the amendment (section 5a)
cp -p $D/sft.log $D/sft.log.crash_2026-11-13      # the crash log, kept beside it (small)
mv /workspace/refe-plan-vdac /workspace/refe-plan-vdac.hbs0_tree
mkdir -p /workspace/refe-plan-vdac && tar -xf <tar> -C /workspace/refe-plan-vdac
python /workspace/refe-plan-vdac/refe/p7_ship_list.py check --pkg /workspace/refe-plan-vdac --list <new ship list>      # ZZP7SHIP_OK
```
Put the frozen A1 next to the card: `/workspace/data/refe_vdac/PREREG_VDAC_P6_A5_A1.md`, `sha256sum` recorded.
`model_sft_A.pt` / `model_sft_B.pt` are the pod's ONLY copies: delete nothing.

## C. Pod: one P7 pause, stage `A5HBFIX` (HB-s0 forward-only, then HB-SHUF)
```
ps -eo pid,args | grep -E '[p]7_pod\.sh|[t]rain\.py'        # the two PIDs (no pgrep -f)
setsid nohup bash /workspace/refe-plan-vdac/refe/vdac_pause_p7.sh <P7_POD_SH_PID> <TRAIN_PY_PID> A5HBFIX >> /workspace/data/refe_vdac/pause.out 2>&1 < /dev/null &
```
What the stage does: waits for P7's next checkpoint, stops P7, confirms the GPU empty, then (1) `ARM=HB-s0 MODE=h48x timeout 3600 bash vdac_pod.sh` = the arm's own argv + `--h48x-only $D`, log `$D/h48x_only.log`, ends `ZZSFT_H48X_ONLY_DONE`; if it does not, HB-SHUF is NOT run; (2) the chain for HB-SHUF (`vdac_chain.sh HB-SHUF`, timeout 14,400 s: precheck, dd, preflight, mutants, gate + token, train 2,000, run check, post-gate, export, egress); P7 resumes via the EXIT trap (unless `HOLD_P7`). Watch with a script, not a model turn; terminal lines in `pause.log`.
Optional smaller first step (recommended if the pod time is short): `A5HBFIX` cannot stop after step 1; to run ONLY the forward (about minutes), run it by hand inside your own P7 pause: `ARM=HB-s0 MODE=h48x bash /workspace/refe-plan-vdac/refe/vdac_pod.sh` with the GPU empty.

## D. Pod: after the pause, HB-s0's checks by hand (the chain cannot resume mid-arm)
```
cat $D/h48x_provenance.json | python -c "import json,sys; p=json.load(sys.stdin); print(p['update_read_back'], p['states_file_cross_check'], p['frames'], p['excluded_frames_audit'], p['frames_equal_audit'])"
#   want: {'aux_drv.pt': 10000, 'scorer_states_last.pt': 10000}  {'aux_head_equal': True, 'hb_lora_equal': True}  N  N  True
python /workspace/refe-plan-vdac/refe/vdac_postgate.py $D --updates 10000      # ZZVDAC_POSTGATE_OK
```
`loaded.*.sha256` in the provenance must equal the step-B sha256s. Required files for the reader: `$D/scores/head_pred_update10000.npz` (from the training run), `$D/head_pred_h48x.npz` (new).

## E. Egress (the chain's push, same repo and prefix as `vdac_chain.sh:373-374`; token file read in place, never printed)
```
python /workspace/refe-plan-vdac/refe/hf_egress.py push --repo Sayood/tanitad-refe-scorer1p6 --prefix vdac/hb-s0 --max-push-bytes 6e9 --tag arm_vdac_hb-s0_a1 \
  --chain-log /workspace/data/refe_vdac/chain_hb-s0.log --egress-log /workspace/data/refe_vdac/egress.jsonl --token-file "$TF" \
  --files $D/aux_drv.pt $D/head_pred_h48x.npz $D/h48x_provenance.json $D/h48x_only.log $D/h48x_only.jsonl $D/scores/head_pred_update10000.npz $D/sft.log $D/sft.log.crash_2026-11-13 $D/sft.jsonl $D/postgate.log
```
(`$TF` = the HF token file the chain resolves, `vdac_pause_p7.sh` `TF=`; run `postgate` output to `$D/postgate.log` first: `... > $D/postgate.log`.) Verify the `ZZEGRESS_OK` line and the LFS sha256, then pull with `HF_HUB_DISABLE_XET=1` and the tanitad venv. HB-SHUF's egress is the chain's own.

## F. Dev box: the read (unchanged from POD_RUN_PLAN Step F)
Freeze record first, manifest `manifest_a5_hb.json` written BEFORE the pulled files are opened (`hb_s0_a` = `scores/head_pred_update10000.npz`, `hb_s0_b` = `head_pred_h48x.npz`, `hb_shuf_*` from HB-SHUF), then `s1_read_v2.py --manifest ... --out ...` with `S1_READ_V1` set as there. Nothing of HB-s0 is read before the manifest exists.

## Decisions for the MM before running
1. Base the fix on the tip `1f65b081` (it already differs from `06c9bd79` in `scorer_finetune.py`, `aux_hb.py`, `gate_checks.py`, `vdac_pod.sh`, `vdac_a5_layer.py`: the HB memory edits, preflight-only): if the pod's tree must stay byte-identical to `06c9bd79` plus ONLY this fix, say so and the patch is re-based (the new entries are disjoint from the memory edits).
2. A new pause stage `A5HBFIX` was added (so one P7 pause serves both); if you prefer the unchanged `A5HB` chain for HB-SHUF and a separate manual forward, drop the `vdac_pause_p7.sh` edit.
