import json, hashlib, sys
SP = sys.argv[1]
R = r"D:\tanitad\runs\v8f_mini_full"
# B = the argv the QUEUE PRINTS for `train M2b 0` (raw/queue_dryrun_lines.txt), tokenised verbatim
B = ["--arm","M2b","--seed","0","--grid","pool4x10","--out",R+r"\M2b_s0",
 "--cache-dir",r"D:\tanitad\cache_full","--v9-train",r"D:\tanitad\data\v9\v9_labels_train.npz","--v9-train-md5","83e30bd3791f0767707eade5f4671759",
 "--lead-train",r"D:\tanitad\data\v9\lead_train4572.npz","--speed-train",r"D:\tanitad\data\v9\refcv6_speed_max_v8_train.jsonl",
 "--join-train",R+r"\join_train.npz","--p10",R+r"\p10\p10_prior.pt","--kin-mode","yaw","--tracks",r"D:\tanitad\cache_full\native_tracks.pt",
 "--exclude","540:600","--batch","64","--steps-mult","4","--device","cuda"]
# L = B + the LoRA flags (PREREG_V8F_LORA 'The arm'); --out/--frames-root/--accum are VENUE choices no committed file fixes -> placeholders
LORA = ["--encoder","lora","--lora-r","32","--lora-alpha","32","--lora-targets","q,v","--enc-ckpt","--accum","64",
        "--frames-root",r"<VENUE>\lora_frames\train"]
L = list(B); L[L.index("--out")+1] = R+r"\M2b-lora_s0"; L += LORA
json.dump(B, open(SP+"/raw/argv_B_M2b_s0.json","w"), indent=1)
json.dump(L, open(SP+"/raw/argv_L_s0_canonical.json","w"), indent=1)
def sh(a): return hashlib.sha256(" ".join(a).encode()).hexdigest()[:16]
print("B", sh(B), len(B)); print("L", sh(L), len(L)); print(" ".join(L))
