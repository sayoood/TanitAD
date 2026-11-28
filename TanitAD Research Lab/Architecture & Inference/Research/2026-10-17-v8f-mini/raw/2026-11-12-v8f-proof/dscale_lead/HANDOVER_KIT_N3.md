# HANDOVER_KIT_N3: the kit of card ADD45 after note N3 (the evaluation view), for the tree owner session

Read with HANDOVER_KIT.md; only what changed is listed. Note: PREREG_V8F_PROOF_ADD45_N3.md, sha256 9953872e293635b83f4528802ea53664e56d978603f0535c1eff798040f610ed.

## 1. Files to copy over the kit on the pod (sha256)

- `pod_dscale_lead_run.sh`  90f4cdc236840c56171b28fb587033c29b63796954144dc899fc2c672fe71a79
- `dscale_evalview.py`  5d50a2d07411cee20f11ff663be4737d1d3e83f0dcd69919d68cc699d4e64806
- `TREE_PINS.txt`  b43e3673a75c35e7f5e43a34b3af571349f63010cdd322c102848e1a9cdf3447
- `dscale_xenc.py`  61fe14ff8f5943d67ca9ee544c31b506df5989ee5819e977069cd162ae30ae54
- `xenc/xenc_sample_v8proof.npz`  d3d3997086c915c954f3431896b02f266e54df84d7cfea60e514a3d50833eebf

Every other kit file keeps the hash of HANDOVER_KIT.md. The pins file now holds 10 pins; the kit refuses a pins file with fewer.

## 2. Second run

- Same variables as the first run, with a NEW output folder (OUT=/root/dscale_lead/out2). EVAL_CACHE stays the pod's own evaluation cache: the kit builds the view under OUT/eval_view itself and reads the evaluation set from it.
- Stages precheck, then cpucheck (CPU only). The GPU stages wait for the slot the PI set.
- New in the log: a line evaluation view rc=0 at every stage; records OUT/eval_view/EVALVIEW.<stage>.json.

## 3. Report d3x (CPU, gates nothing, once, any time before the read)

    PY KIT/dscale_xenc.py compare --eval-cache <the pod's evaluation cache> --sample KIT/xenc/xenc_sample_v8proof.npz --out OUT/precheck/XENC.json

It reads 20 evaluation clips (about 21 MB each) one after the other; torch and numpy on CPU.

## 4. Back to the coordinator fork

OUT/precheck/ (with XENC.json), OUT/eval_view/EVALVIEW.*.json and evalview.*.out, dscale_lead.log, markers, and the STOP file if there is one. Not the view's manifest links, not checkpoints.
