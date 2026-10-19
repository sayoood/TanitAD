# (key, table, group, label, cols, values) -- values are the printed strings, in the printed column order.
# cols use: NC DAC DDC TL EP TTC LK HC EC C PDMS EPDMS
V1 = ["NC", "DAC", "EP", "TTC", "C", "PDMS"]
V1b = ["NC", "DAC", "TTC", "EP", "PDMS"]
V2 = ["NC", "DAC", "DDC", "TL", "EP", "TTC", "LK", "HC", "EC", "EPDMS"]
S1 = ["NC", "DAC", "DDC", "TL", "EP", "TTC", "LK", "HC", "EC"]
ROWS = [
    # ---- Hydra-MDP (2406.06978): distilling the overall PDM score vs the sub-scores; inference weighting
    ("2406.06978", "Tab.1", "hydra-mdp", "Hydra-MDP-V8192 (sub-score heads)", V1, ["97.9", "91.7", "77.6", "92.9", "100", "83.0"]),
    ("2406.06978", "Tab.1", "hydra-mdp", "Hydra-MDP-V8192-PDM (distil the overall PDM score only)", V1, ["97.5", "88.9", "74.8", "92.5", "100", "80.2"]),
    ("2406.06978", "Tab.1", "hydra-mdp", "Hydra-MDP-V8192-W (weighted confidence at inference)", V1, ["98.1", "96.1", "77.8", "93.9", "100", "85.7"]),
    ("2406.06978", "Tab.1", "hydra-mdp", "Hydra-MDP-V8192-W-EP (+ EP head)", V1, ["98.3", "96.0", "78.7", "94.6", "100", "86.5"]),
    # ---- Hydra-MDP++ (2503.12820)
    ("2503.12820", "Tab.3", "hydra-mdp++", "ResNet34, no weighted confidence, no TSE, no perception aux", V1, ["97.5", "92.0", "80.4", "91.8", "100", "85.0"]),
    ("2503.12820", "Tab.3", "hydra-mdp++", "ResNet34, + weighted confidence (W)", V1, ["97.4", "96.0", "81.0", "92.8", "100", "86.5"]),
    ("2503.12820", "Tab.3", "hydra-mdp++", "ResNet34, W + temporal SE", V1, ["97.6", "96.0", "80.4", "93.1", "100", "86.6"]),
    ("2503.12820", "Tab.3", "hydra-mdp++", "ResNet34, W + TS + perception aux tasks (P)", V1, ["97.6", "95.6", "80.1", "93.3", "100", "86.1"]),
    ("2503.12820", "Tab.1", "hydra-mdp++", "Hydra-MDP++ ResNet34", V1, ["97.6", "96.0", "80.4", "93.1", "100", "86.6"]),
    ("2503.12820", "Tab.1", "hydra-mdp++", "Hydra-MDP++ V2-99", V1, ["98.6", "98.6", "85.7", "95.1", "100", "91.0"]),
    # ---- DriveSuprim (2506.06659)
    ("2506.06659", "Tab.2", "drivesuprim", "Hydra-MDP ResNet34", V1, ["98.3", "96.0", "78.7", "94.6", "100", "86.5"]),
    ("2506.06659", "Tab.2", "drivesuprim", "DriveSuprim ResNet34", V1, ["97.8", "97.3", "86.7", "93.6", "100", "89.9"]),
    ("2506.06659", "Tab.2", "drivesuprim", "Hydra-MDP ViT-L", V1, ["98.4", "97.7", "85.0", "94.5", "100", "89.9"]),
    ("2506.06659", "Tab.2", "drivesuprim", "DriveSuprim ViT-L", V1, ["98.6", "98.6", "91.3", "95.5", "100", "93.5"]),
    ("2506.06659", "Tab.5", "drivesuprim", "v2 ablation: single-stage baseline", V2, ["97.2", "97.5", "99.4", "99.6", "83.1", "96.5", "94.4", "98.2", "70.9", "81.4"]),
    ("2506.06659", "Tab.5", "drivesuprim", "v2 ablation: + coarse-to-fine multi-stage", V2, ["97.5", "96.4", "99.1", "99.6", "87.0", "96.4", "95.3", "98.2", "75.0", "82.4"]),
    ("2506.06659", "Tab.5", "drivesuprim", "v2 ablation: + rotation augmentation", V2, ["96.9", "96.9", "99.4", "99.6", "87.9", "96.0", "95.5", "98.3", "76.5", "82.7"]),
    ("2506.06659", "Tab.5", "drivesuprim", "v2 ablation: + self-distillation (full)", V2, ["97.5", "96.5", "99.4", "99.6", "88.4", "96.6", "95.5", "98.3", "77.0", "83.1"]),
    ("2506.06659", "Tab.1", "drivesuprim", "oracle: best PDMS among the top-1 predicted", V1, ["99.0", "98.7", "86.5", "96.2", "100", "91.9"]),
    ("2506.06659", "Tab.1", "drivesuprim", "oracle: best PDMS among the top-256 predicted", V1, ["100", "100", "97.1", "99.9", "100", "98.7"]),
    # ---- WoTE (2504.01941)
    ("2504.01941", "Tab.3", "wote", "trajectory prediction only (TransFuser baseline)", V1, ["96.4", "91.5", "76.2", "90.3", "99.9", "81.0"]),
    ("2504.01941", "Tab.3", "wote", "+ trajectory evaluation, no future BEV states", V1, ["97.1", "92.9", "78.2", "91.9", "100.0", "83.2"]),
    ("2504.01941", "Tab.3", "wote", "+ BEV world-model future states (full)", V1, ["98.0", "94.7", "79.9", "93.4", "100.0", "85.6"]),
    ("2504.01941", "Tab.4", "wote", "imitation reward only", V1, ["97.9", "92.5", "77.7", "93.4", "99.9", "83.5"]),
    ("2504.01941", "Tab.4", "wote", "simulation reward only", V1, ["96.5", "94.7", "79.2", "90.1", "99.4", "83.6"]),
    ("2504.01941", "Tab.7", "wote", "train 256 / test 256 anchors, anchors unrefined", V1, ["96.6", "94.6", "79.0", "91.1", "100.0", "84.0"]),
    ("2504.01941", "Tab.7", "wote", "train 256 / test 1024 anchors (unseen), unrefined", V1, ["97.2", "95.6", "79.3", "92.4", "100.0", "85.3"]),
    ("2504.01941", "Tab.7", "wote", "train 256 / test 256, refined trajectories", V1, ["98.0", "94.7", "79.9", "93.4", "100.0", "85.6"]),
    # ---- iPad (2505.15111)
    ("2505.15111", "Tab.4", "ipad", "BEVFormer, no proposal refinement", V1b, ["97.6", "93.0", "92.9", "68.9", "78.5"]),
    ("2505.15111", "Tab.4", "ipad", "BEVFormer + proposal refinement", V1b, ["96.9", "93.2", "90.8", "71.5", "79.4"]),
    ("2505.15111", "Tab.4", "ipad", "ProFormer (proposal-aware spatial cross-attention)", V1b, ["98.1", "96.5", "94.3", "84.2", "89.8"]),
    ("2505.15111", "Tab.4", "ipad", "+ proposal-centric mapping task", V1b, ["98.3", "97.9", "94.4", "85.9", "90.5"]),
    ("2505.15111", "Tab.4", "ipad", "+ proposal-centric prediction task (full)", V1b, ["98.6", "98.3", "94.9", "88.0", "91.7"]),
    # ---- ZTRS (2510.24108)
    ("2510.24108", "Tab.8", "ztrs", "V2-99 reward = safety only", V2, ["98.6", "99.3", "99.9", "99.7", "63.3", "96.3", "96.2", "79.8", "24.5", "72.0"]),
    ("2510.24108", "Tab.8", "ztrs", "V2-99 reward = safety + progress", V2, ["98.4", "99.2", "99.8", "99.7", "83.8", "97.8", "95.9", "96.0", "49.8", "82.4"]),
    ("2510.24108", "Tab.8", "ztrs", "V2-99 reward = safety + progress + comfort (full)", V2, ["97.8", "99.4", "99.8", "99.8", "84.1", "97.0", "96.2", "98.2", "77.2", "85.3"]),
    # ---- Versatile samples (2609.01799): columns = plain | longitudinal-only | lateral-only | both (Tab.5); plain | random | designed (Tab.4)
    ("2609.01799", "Tab.5", "versatile", "DiffusionDrive EPDMS: plain | lon-only | lat-only | both", ["plain", "lon", "lat", "both"], ["89.71", "89.96", "89.89", "90.05"]),
    ("2609.01799", "Tab.5", "versatile", "DiffusionDrive NC: plain | lon-only | lat-only | both", ["plain", "lon", "lat", "both"], ["98.39", "98.54", "98.47", "98.56"]),
    ("2609.01799", "Tab.5", "versatile", "DiffusionDrive DAC: plain | lon-only | lat-only | both", ["plain", "lon", "lat", "both"], ["97.52", "97.62", "97.72", "97.70"]),
    ("2609.01799", "Tab.5", "versatile", "MeanFuser EPDMS: plain | lon-only | lat-only | both", ["plain", "lon", "lat", "both"], ["90.12", "90.24", "90.36", "90.41"]),
    ("2609.01799", "Tab.5", "versatile", "MeanFuser NC: plain | lon-only | lat-only | both", ["plain", "lon", "lat", "both"], ["98.22", "98.28", "98.26", "98.27"]),
    ("2609.01799", "Tab.5", "versatile", "MeanFuser DAC: plain | lon-only | lat-only | both", ["plain", "lon", "lat", "both"], ["97.95", "98.01", "98.26", "98.27"]),
    ("2609.01799", "Tab.4", "versatile", "DiffusionDrive EPDMS: plain | random samples | designed", ["plain", "random", "design"], ["89.71", "89.72", "90.05"]),
    ("2609.01799", "Tab.4", "versatile", "DiffusionDrive NC: plain | random | designed", ["plain", "random", "design"], ["98.39", "98.37", "98.56"]),
    ("2609.01799", "Tab.4", "versatile", "DiffusionDrive DAC: plain | random | designed", ["plain", "random", "design"], ["97.52", "97.54", "97.70"]),
    ("2609.01799", "Tab.4", "versatile", "MeanFuser DAC: plain | random | designed", ["plain", "random", "design"], ["97.95", "97.98", "98.27"]),
    # ---- DA-WAM (2608.19085)
    ("2608.19085", "Tab.3", "da-wam", "no future prediction", ["PDMS", "NC", "DAC", "EP", "TTC", "C"], ["93.31", "98.45", "98.27", "91.36", "95.48", "99.99"]),
    ("2608.19085", "Tab.3", "da-wam", "shared global future (control)", ["PDMS", "NC", "DAC", "EP", "TTC", "C"], ["92.81", "99.02", "98.46", "88.68", "96.54", "99.99"]),
    ("2608.19085", "Tab.3", "da-wam", "action-conditioned future, no hard negatives", ["PDMS", "NC", "DAC", "EP", "TTC", "C"], ["93.46", "98.88", "98.58", "90.47", "96.33", "99.69"]),
    ("2608.19085", "Tab.3", "da-wam", "action-conditioned future + hard negatives", ["PDMS", "NC", "DAC", "EP", "TTC", "C"], ["93.68", "99.11", "98.88", "89.97", "96.81", "99.77"]),
    # ---- SafeDrive (2602.18887)
    ("2602.18887", "Tab.5", "safedrive", "no world model, scene-level safety", ["NC", "DAC", "TTC", "PDMS"], ["99.0", "97.7", "96.3", "90.1"]),
    ("2602.18887", "Tab.5", "safedrive", "BEV world + scene-level safety", ["NC", "DAC", "TTC", "PDMS"], ["99.2", "98.3", "96.5", "90.9"]),
    ("2602.18887", "Tab.5", "safedrive", "sparse world + scene-level safety", ["NC", "DAC", "TTC", "PDMS"], ["99.2", "98.2", "96.7", "90.9"]),
    ("2602.18887", "Tab.5", "safedrive", "sparse world + scene-level + fine-grained safety", ["NC", "DAC", "TTC", "PDMS"], ["99.5", "99.0", "97.2", "91.6"]),
    ("2602.18887", "Tab.6", "safedrive", "neither PwNC nor TwDAC", ["NC", "DAC", "TTC", "PDMS"], ["99.2", "98.5", "96.0", "90.9"]),
    ("2602.18887", "Tab.6", "safedrive", "PwNC only (per-agent, per-step collision factor)", ["NC", "DAC", "TTC", "PDMS"], ["99.5", "98.7", "97.3", "91.5"]),
    ("2602.18887", "Tab.6", "safedrive", "TwDAC only (per-step DAC + 9-point map lookup)", ["NC", "DAC", "TTC", "PDMS"], ["99.2", "99.0", "96.7", "91.4"]),
    ("2602.18887", "Tab.6", "safedrive", "both", ["NC", "DAC", "TTC", "PDMS"], ["99.5", "99.0", "97.2", "91.6"]),
    # ---- FlowR2A (2606.24231) Table 7: aggregation weights as a free EP<->TTC knob
    ("2606.24231", "Tab.7", "flowr2a", "(wEP,wTTC,wHC)=(1,1,1) default", V1, ["98.8", "98.0", "90.1", "96.0", "100", "92.8"]),
    ("2606.24231", "Tab.7", "flowr2a", "(1,2,1)", V1, ["99.0", "98.0", "89.0", "96.8", "100", "92.8"]),
    ("2606.24231", "Tab.7", "flowr2a", "(1,3,1)", V1, ["99.1", "98.1", "88.0", "97.1", "100", "92.6"]),
    # ---- TOAD (2606.07170) Table 7, navhard stage 1, iPad base
    ("2606.07170", "Tab.7", "toad", "iPad", S1, ["96.2", "94.9", "96.8", "99.6", "82.0", "95.8", "94.2", "97.8", "42.2"]),
    ("2606.07170", "Tab.7", "toad", "iPad + re-scoring with DrivoR scorer", S1, ["98.7", "97.8", "99.9", "99.6", "75.2", "98.7", "94.4", "88.9", "44.0"]),
    ("2606.07170", "Tab.7", "toad", "iPad + re-scoring with GTRS scorer", S1, ["95.3", "87.3", "93.8", "99.6", "69.9", "95.6", "90.2", "89.6", "34.2"]),
    ("2606.07170", "Tab.7", "toad", "iPad + re-scoring with SparseDriveV2 scorer", S1, ["96.7", "85.1", "98.7", "100.0", "79.4", "96.4", "95.6", "92.0", "36.9"]),
    ("2606.07170", "Tab.7", "toad", "iPad + search with GTRS scorer", S1, ["93.3", "77.8", "89.8", "99.8", "59.9", "94.0", "79.1", "94.2", "15.6"]),
    ("2606.07170", "Tab.7", "toad", "iPad + search with SparseDriveV2 scorer", S1, ["96.9", "86.9", "96.7", "100.0", "74.7", "96.9", "86.0", "97.6", "32.0"]),
    ("2606.07170", "Tab.7", "toad", "iPad + TOAD (search with DrivoR scorer)", S1, ["97.7", "97.1", "99.6", "99.8", "75.6", "98.4", "93.6", "97.6", "43.1"]),
    # ---- GTRS (2506.06664) Table 2: ensemble members vs ensemble, navhard stage 1
    ("2506.06664", "Tab.2", "gtrs", "GTRS-Dense EVA-ViT-L stage 1", S1, ["97.6", "95.8", "99.7", "99.8", "77.2", "97.8", "95.3", "97.3", "46.7"]),
    ("2506.06664", "Tab.2", "gtrs", "GTRS-Aug EVA-ViT-L stage 1", S1, ["98.7", "98.0", "99.1", "99.8", "75.9", "98.7", "94.7", "97.6", "49.8"]),
    ("2506.06664", "Tab.2", "gtrs", "GTRS-E-Lite (2-model ensemble) stage 1", S1, ["98.2", "98.9", "99.6", "99.6", "75.8", "98.4", "96.9", "97.3", "53.3"]),
    ("2506.06664", "Tab.2", "gtrs", "GTRS-E (6-model ensemble) stage 1", S1, ["98.9", "99.3", "99.8", "99.8", "75.2", "98.4", "96.0", "97.6", "51.6"]),
    # ---- SparseDriveV2 (2603.29163)
    ("2603.29163", "Tab.2", "sparsedrivev2", "SparseDriveV2 ResNet-34 (NC DAC TTC C EP PDMS)", ["NC", "DAC", "TTC", "C", "EP", "PDMS"], ["98.5", "98.4", "95.0", "99.9", "88.6", "92.0"]),
    ("2603.29163", "Tab.1", "sparsedrivev2", "Hydra-MDP, 1024 anchors (EPDMS)", ["EPDMS"], ["85.02"]),
    ("2603.29163", "Tab.1", "sparsedrivev2", "Hydra-MDP, 16384 anchors (EPDMS)", ["EPDMS"], ["87.35"]),
    # ---- MM-Future (2609.20377) Table 3 block C
    ("2609.20377", "Tab.3", "mm-future", "scoring on history only", ["NC", "DAC", "TTC", "EP", "PDMS"], ["98.8", "98.3", "95.5", "90.2", "92.9"]),
    ("2609.20377", "Tab.3", "mm-future", "scoring on history + paired predicted future", ["NC", "DAC", "TTC", "EP", "PDMS"], ["98.9", "98.5", "96.0", "90.3", "93.3"]),
    # ---- CLOVER (2605.15120): fixed-pool scorer diagnostic
    ("2605.15120", "Tab.20", "clover", "DINOv2-Small scorer, train / test top-1 PDMS", ["train", "test"], ["95.0", "92.4"]),
    ("2605.15120", "Tab.20", "clover", "DINOv3-Base scorer, train / test", ["train", "test"], ["95.1", "92.7"]),
    ("2605.15120", "Tab.20", "clover", "Wan2.2-5B features, train / test", ["train", "test"], ["97.7", "92.8"]),
    # ---- SparseDriveV2 (2603.29163) Table 6 (EPDMS only): vocabulary size | path-scene attention x re-conditioning
    ("2603.29163", "Tab.6", "sparsedrivev2", "MHA, no re-conditioning (Np=512,Nv=128 row)", ["Np", "Nv", "EPDMS_vocab", "EPDMS_attn"], ["512", "128", "88.7", "87.7"]),
    ("2603.29163", "Tab.6", "sparsedrivev2", "MHA + re-conditioning (Np=512,Nv=256 row)", ["Np", "Nv", "EPDMS_vocab", "EPDMS_attn"], ["512", "256", "89.2", "89.9"]),
    ("2603.29163", "Tab.6", "sparsedrivev2", "deformable aggregation, no re-conditioning (Np=1024,Nv=128 row)", ["Np", "Nv", "EPDMS_vocab", "EPDMS_attn"], ["1024", "128", "89.5", "89.9"]),
    ("2603.29163", "Tab.6", "sparsedrivev2", "deformable aggregation + re-conditioning (Np=1024,Nv=256 row)", ["Np", "Nv", "EPDMS_vocab", "EPDMS_attn"], ["1024", "256", "90.1", "90.1"]),
    # ---- DrivoR (2601.05083) Table 6 (PDMS only, navval): score count / disentanglement
    ("2601.05083", "Tab.6", "drivor", "(c) separate decoders, no disentanglement, 6 scores", ["score_num", "PDMS"], ["6", "86.8"]),
    ("2601.05083", "Tab.6", "drivor", "(d) separate decoders, disentangled, 6 scores", ["score_num", "PDMS"], ["6", "90.0"]),
    ("2601.05083", "Tab.6", "drivor", "separate decoders, disentangled, 1 final score", ["score_num", "PDMS"], ["1", "88.2"]),
    # ---- FlowR2A (2606.24231) Table 4(a): generator conditioned on score / sub-rewards / per-timestep arrays (single proposal)
    ("2606.24231", "Tab.4a", "flowr2a", "EP: score-only | +sub-rewards | +per-timestep", ["so", "sr", "pt"], ["83.3", "84.3", "84.0"]),
    ("2606.24231", "Tab.4a", "flowr2a", "TTC: score-only | +sub-rewards | +per-timestep", ["so", "sr", "pt"], ["86.9", "88.8", "94.9"]),
    # ---- CLOVER (2605.15120) Table 5: selected vs oracle@64 (PDMS in 0-1)
    ("2605.15120", "Tab.5", "clover", "Selected PDMS: baseline | stage 1 | stage 2", ["base", "s1", "s2"], ["0.9369", "0.9413", "0.9448"]),
    ("2605.15120", "Tab.5", "clover", "Oracle@64 PDMS: baseline | stage 1 | stage 2", ["base", "s1", "s2"], ["0.9933", "0.9976", "0.9939"]),
    ("2605.15120", "Tab.5", "clover", "Gap oracle - selected: baseline | stage 1 | stage 2", ["base", "s1", "s2"], ["0.0564", "0.0563", "0.0495"]),
    # ---- SafeDrive (2602.18887) Table 4: ablation of main components (NC DAC TTC EP PDMS)
    ("2602.18887", "Tab.4", "safedrive", "DiffusionDrive head only", ["NC", "DAC", "TTC", "EP", "PDMS"], ["98.8", "96.3", "95.8", "82.3", "88.6"]),
    ("2602.18887", "Tab.4", "safedrive", "ProposalNet (trajectory-guided deformable attention)", ["NC", "DAC", "TTC", "EP", "PDMS"], ["98.9", "97.1", "95.1", "83.2", "89.1"]),
    ("2602.18887", "Tab.4", "safedrive", "ProposalNet + scene-level safety evaluation", ["NC", "DAC", "TTC", "EP", "PDMS"], ["99.0", "97.7", "96.3", "83.5", "90.1"]),
    # ---- Hydra-MDP (2406.06978) Table 1: sim-in-the-loop post-processing with the predicted perception (PP)
    ("2406.06978", "Tab.1", "hydra-mdp", "Vadv2-V4096 (imitation scorer, own distance loss)", V1, ["97.1", "88.8", "74.9", "91.4", "100", "79.7"]),
    ("2406.06978", "Tab.1", "hydra-mdp", "Vadv2-V4096-PP (rule-based post-processing on Transfuser perception)", V1, ["97.0", "89.1", "75.0", "91.2", "100", "79.9"]),
    # ---- GTRS (2506.06664) Table 1: scorer trained on a super-dense static vocabulary, run on the generator's own proposals Vdp (EPDMS1 real / EPDMS2 synthetic / EPDMS)
    ("2506.06664", "Tab.1", "gtrs-vocab", "generator proposals chosen at random (no scorer)", ["EPDMS1", "EPDMS2", "EPDMS"], ["60.9", "40.8", "25.6"]),
    ("2506.06664", "Tab.1", "gtrs-vocab", "train VXL (16,384), infer on Vdp only", ["EPDMS1", "EPDMS2", "EPDMS"], ["76.6", "48.6", "36.7"]),
    ("2506.06664", "Tab.1", "gtrs-vocab", "train VXL, infer on VXL", ["EPDMS1", "EPDMS2", "EPDMS"], ["78.1", "50.2", "39.7"]),
    ("2506.06664", "Tab.1", "gtrs-vocab", "train VXL, infer on Vdp u VL", ["EPDMS1", "EPDMS2", "EPDMS"], ["71.8", "57.3", "42.0"]),
    ("2506.06664", "Tab.1", "gtrs-vocab", "train VXL with vocabulary dropout, infer on Vdp u VL", ["EPDMS1", "EPDMS2", "EPDMS"], ["73.1", "59.0", "43.4"]),
]
