"""Cost of the PREREG_V8F_LORA Amendment 1 fallback: 're-encode the train cache from the on-the-fly route, then retrain M2b at 2 seeds'.
Pure arithmetic (ESTIMATED) on constants that D3_SCOPING.md section 2 tabulates as MEASURED, each with its artifact. Nothing is run or read here beyond those constants.
"""
from __future__ import annotations

# --- constants (MEASURED, per D3_SCOPING.md section 2; artifact in the comment)
N_TRAIN_CACHE = 4572            # box 2 manifest_train.jsonl (probe_provenance.json): 4,512 trained + 60 RA-1 holdout (excluded by position 540:600)
N_TRAINED = 4512
N_EVAL = 141
ENC_S = 5.01                    # s/clip, box 2 SAFE MODE DINOv3-L, 423 clips (box2_ship/logs/encode_A2s.log; 4.98 on 64)
ENC_S_EVAL = 4.20               # s/clip mean over the 141 eval clips (manifest_eval.jsonl enc_s)
BUILD_DEV_S = 8.23              # s/clip dev box, 6 workers (600 clips in 82.3 min, raw/s4c/build_train600.log)
BUILD_BOX2_S = 11.7             # s/clip box-2 build alone (16.66 all-in minus encode/verify; chain_ra1_full.out: 3,972 clips 10-18 12:14 -> 10-19 06:37)
ALLIN_BOX2_S = 16.66            # s/clip box 2 build+encode+verify+delete (the RA-1 FULL recipe)
V2EP_MB = 36.99                 # MB/clip native 256x640 PNG v2ep (raw/s4c/check_report_train600.json)
LAN_MBPS = 14.64                # dev -> box 2 (transfer_check.json)
MP4_SHIP_S = 3517               # 51,480,904,147 B dev -> box 2 (ship_trainall.log)
STEPS, SPS = 19164, 1.217       # M2b (RESULT_V8F_GRID.md)
SEEDS = 2

h = lambda s: s / 3600.0
enc_train_h = N_TRAIN_CACHE * ENC_S / 3600
enc_trained_h = N_TRAINED * ENC_S / 3600
enc_eval_h = N_EVAL * ENC_S_EVAL / 3600
train_h_seed = STEPS / SPS / 3600
train_h = SEEDS * train_h_seed
print(f"re-encode, box-2 GPU : train cache {N_TRAIN_CACHE} clips x {ENC_S} s = {enc_train_h:.2f} h ({N_TRAINED} trained clips only: {enc_trained_h:.2f} h); eval-141 x {ENC_S_EVAL} s = {enc_eval_h:.2f} h")
print(f"retrain M2b          : {STEPS} steps / {SPS} steps/s = {train_h_seed:.2f} h per seed, x{SEEDS} seeds = {train_h:.2f} h box-2 GPU")
print()
print("Frames must exist: box 2 deleted the 3,972 v2ep (and their mp4) after each verified chunk (ra1_full_box2.py:212-214); only train600-256 (600 clips) is kept there.")
print("  route A (dev box builds the v2ep, box 2 only encodes = D3 'V3'):")
dev_build_h = N_TRAIN_CACHE * BUILD_DEV_S / 3600
ship_h = N_TRAIN_CACHE * V2EP_MB / LAN_MBPS / 3600
print(f"    dev CPU build {N_TRAIN_CACHE} x {BUILD_DEV_S} s = {dev_build_h:.1f} h; LAN ship {N_TRAIN_CACHE * V2EP_MB / 1024:.0f} GB at {LAN_MBPS} MB/s = {ship_h:.1f} h (overlapped in 400-clip chunks; box 2 D: held 85.9 GB free on 10-19, so chunks, delete after verify)")
boxA = enc_train_h + enc_eval_h + train_h
wallA = max(dev_build_h, ship_h + enc_train_h) + enc_eval_h + train_h
print(f"    box-2 GPU hours = {enc_train_h:.2f} encode + {enc_eval_h:.2f} eval + {train_h:.2f} retrain = {boxA:.1f} h ; wall ~ {wallA:.0f} h (pipelined build/ship/encode, then the two retrains; training itself is not overlappable with a box that is encoding)")
print("  route B (box 2 builds and encodes = the RA-1 FULL recipe, mp4s re-shipped first):")
allin_h = N_TRAIN_CACHE * ALLIN_BOX2_S / 3600
print(f"    ship mp4: {MP4_SHIP_S / 3600:.2f} h ; box-2 all-in {N_TRAIN_CACHE} x {ALLIN_BOX2_S} s = {allin_h:.1f} h (D3_SCOPING quotes 20.9 h for 4,512) ; + eval + retrain = {allin_h + enc_eval_h + train_h:.1f} box-2 h ; wall ~ {MP4_SHIP_S / 3600 + allin_h + enc_eval_h + train_h:.0f} h")
print()
print("  cheapest discriminating alternative (NOT a fallback; the thing to run first): the identity control on box 2 itself, GPU bf16, the builder's own encode_episode / the arm's B=0 encoder,")
print("    native256 frames, against cache_full. Cost ~ 2 clips x 101 frames x ~0.05 s of GPU + the 2 v2ep (already on box 2 for the 600) = minutes, not hours.")
print("    MEASURED precedent: box 2's own verify_cache C5 re-encoded 5 clips on the same device and got 5/5 bit-exact content_md5 (box2_ship/logs/encode_A2s.log), so a same-device,")
print("    same-batching identity is expected to read ~0, not 3 %.")
