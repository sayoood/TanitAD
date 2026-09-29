#!/usr/bin/env python3
"""rd_cost_model.py -- Stream R-D (HiCAP), 2026-09-29.  Pure arithmetic, no dependencies, runs in < 1 s.

Every number here is ESTIMATED: constants are assumptions (backbone sizes are RECALLED, not verified; Thor effective
TFLOP/s is a scenario, not a measurement -- no Thor latency exists for any HiCAP backbone candidate). It exists so that
the tables in RD_cameras_tokens_audio.md sec.3.5 / 3.7 / 4.4 can be re-derived and re-run with different constants.
Thor LPDDR5X bandwidth 273 GB/s is quoted from Production & Optimization/FLAGSHIP_V1_INFERENCE_OPTIMIZATION.md:295 (PUB+E).
Chunk sizes are MEASURED in Data Engineering/.../2026-07-26-physicalai-feature-probe/PHYSICALAI_FEATURE_PROBE.md:70-76.
"""
Pv, Lv, dv = 0.4e9, 27, 1152          # tower: SO400M-class (27 layers, width 1152: recalled, UNVERIFIED)
PL, LL, dL = 1.7e9, 28, 2048          # LLM: ~1.7B-class (28 layers, width 2048: recalled, UNVERIFIED)
N0 = 64                                # ego/text/prompt tokens
BW = 273e9                             # Thor LPDDR5X (repo doc, PUB+E)
def tower(npatch): return 2*Pv*npatch + 4*Lv*npatch**2*dv
def llm(N): return 2*PL*N + 4*LL*N**2*dL
def cfg(tokens_per_cam):               # list of tokens per camera (post 2x2 merge); patches = 4x tokens
    Ft = sum(tower(4*t) for t in tokens_per_cam)
    N = N0 + sum(tokens_per_cam)
    return Ft, llm(N), N
def ms(Ft, Fl, E):                     # stage-wise roofline: tower (weights 0.8GB) then LLM (weights 3.4GB)
    return 1e3*(max(Ft/(E*1e12), 0.8e9/BW) + max(Fl/(E*1e12), 3.4e9/BW))
rows = [("C1  FW@160", [160]),
        ("C2  FW@160 + 1x@160", [160,160]),
        ("C2p FW@160 + 1x@64", [160,64]),
        ("C3  FW@160 + 2x@160", [160,160,160]),
        ("C3p FW@160 + 2x@64", [160,64,64]),
        ("C4  4x@160 (AR1 default cams, 1 frame)", [160]*4),
        ("C4p FW@160 + 3x@64", [160,64,64,64]),
        ("C7  7x@160", [160]*7),
        ("C7p FW@160 + 6x@64", [160]+[64]*6),
        ("AR1-style 4 cams x 4 frames @160 (16 images)", [160]*16)]
# --- calibration to NVIDIA-published Thor rows as tabulated by the sibling stream R-A (RA_vlm_backbones.md sec.5.3; INHERITED, not
#     re-verified here; TRT-Edge-LLM 0.10.0, NVFP4 LLM + FP16 ViT, batch 1, +-25 % band): total ms for 1 cam x 1 frame @160 tok and
#     3 cams x 1 frame; the marginal camera cost b160 = (ms3 - ms1)/2 is linear (R-A caution 3: batching cameras helps the ViT only ~18 %).
#     A peripheral camera at 64 tokens / 256 patches is assumed to cost 0.42 x b160 (ViT ~ patches, LLM slope ~ tokens): ASSUMPTION.
RA = {"2B-class (Qwen3-VL-2B ~ Cosmos3-Edge reasoner)": (16.6, 35.5), "4B-class (Qwen3.5-4B / Qwen-Drive VLM)": (30.8, 60.5)}
def ra_ms(tokens_per_cam, a, m3):
    b = (m3 - a) / 2.0
    return a + sum(b if t >= 160 else 0.42 * b for t in tokens_per_cam[1:])
print(f"{'config':48s} {'tokens':>6s} {'tower TF':>8s} {'LLM TF':>7s} {'tot TF':>7s} {'ms@25':>7s} {'ms@100':>7s} {'2B TRT':>7s} {'4B TRT':>7s}")
for name, t in rows:
    Ft, Fl, N = cfg(t)
    r2 = ra_ms(t, *RA["2B-class (Qwen3-VL-2B ~ Cosmos3-Edge reasoner)"]) if len(t) <= 7 else float('nan')
    r4 = ra_ms(t, *RA["4B-class (Qwen3.5-4B / Qwen-Drive VLM)"]) if len(t) <= 7 else float('nan')
    print(f"{name:48s} {N:6d} {Ft/1e12:8.2f} {Fl/1e12:7.2f} {(Ft+Fl)/1e12:7.2f} {ms(Ft,Fl,25):7.1f} {ms(Ft,Fl,100):7.1f} {r2:7.1f} {r4:7.1f}")
print('per-image tower 640 patches TF', tower(640)/1e12, ' 256 patches', tower(256)/1e12)
print('LLM weight-stream floor ms', 3.4e9/BW*1e3, ' tower weight floor ms', 0.8e9/BW*1e3)
# crossover tokens where LLM compute = weight streaming at E
for E in (25,50,100):
    N = 3.4e9/BW*E*1e12/(2*PL); print('LLM compute=stream crossover tokens at E=%d TF:'%E, round(N))
# training cache cost
frames = 472627; sub = 5
per_frame_tower_all = 7*tower(640)
llm_1 = llm(N0+160); llm_single = llm(N0+320); llm_all = llm(N0+7*160)
tot = per_frame_tower_all + llm_1 + 6*llm_single + llm_all
print('cache-all-8-options per frame TFLOP', tot/1e12, ' frames/5 =', frames//sub, ' total EFLOP', tot*(frames//sub)/1e18, ' A40 GPU-h at 40 TF eff', tot*(frames//sub)/40e12/3600)
# audio
def beats(sec, P=90e6, L=12, d=768):
    n = 49.6*sec; return 2*P*n + 4*L*n**2*d
print('BEATs-base 2s window GFLOP', beats(2)/1e9, ' per 1s', beats(1)/1e9, ' ms@50TF', beats(2)/50e12*1e3, ' weight floor ms', 90e6*2/BW*1e3)
wh = 2*88e6*1500 + 4*12*1500**2*768; print('Whisper-small enc 30s GFLOP', wh/1e9, ' ms@50', wh/50e12*1e3)
wh2 = 2*88e6*100 + 4*12*100**2*768; print('Whisper-small enc truncated 2s GFLOP', wh2/1e9)
aut = 2*0.65e9*25; print('AuT-class 0.65B(UNVERIFIED) 2s (25 tok) GFLOP', aut/1e9, ' weight floor ms', 0.65e9*2/BW*1e3)
print('audio LLM tokens 2s @12.5Hz = 25 tok -> GFLOP', 25*2*PL/1e9, '; 8 pooled ->', 8*2*PL/1e9)
# extra-camera data cost
chunks = 197; mb = dict(FT=1627, CL=2261, CR=2521, RL=2130, RR=2392, RT=2001)
print('per camera transient GB (197 chunks):', {k: round(v*chunks/1e3) for k,v in mb.items()}, ' sum GB', round(sum(mb.values())*chunks/1e3))
tot_gb = sum(mb.values())*chunks/1e3
for r in (2.1, 17, 42, 118): print(f'  at {r} MB/s: {tot_gb*1e3/r/3600:.1f} h')
print('per-clip MB per camera ~ chunk/97:', {k: round(v/97,1) for k,v in mb.items()})
print('build time: 2376 clips / 26 clips/min =', 2376/26, 'min per camera (front_wide measured rate, 5 workers)')
print('JPEG cache 256x256 2.9 MB/ep x 2376 =', 2.9*2376/1e3, 'GB per camera; x2.5 for 256x640 =', 2.9*2.5*2376/1e3)
