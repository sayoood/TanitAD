"""Stream A (RG note): ESTIMATED Jetson-Thor cost of candidate FROZEN audio towers for one 2 s window at a 2 Hz refresh.

Arithmetic only, no hardware.  Parameter counts marked MEASURED come from the checkpoints' safetensors HEADERS
(fetched with HTTP Range by `rg_audio_tower_params.py`, result in `rg_audio_tower_params_result.json`); the others are
ESTIMATED from the architecture.  Conventions (both INHERITED, not re-derived here):
  * weight-stream floor at 0.7 x 273 GB/s = 191 GB/s, FP16 weights   (sec_eff.tex tab:bracket caption, as used in RF note s1.3)
  * effective dense FP16 compute 25.7 TFLOP/s, i.e. the figure implied by sec_elastic.tex's "18 GFLOP ~ 0.7 ms on Thor"
T_window = max(weight floor, compute); duty = 2 Hz * T_window.  Encoder FLOPs ~ 2 * P_layers * n_tokens + attention 4*L*n^2*d
+ the convolutional front end where it is not negligible.  Run: python3 rg_audio_tower_cost.py  -> *_result.json next to it.
"""
import json, os

BW = 0.7 * 273e9          # B/s
FLOPS = 18e9 / 0.7e-3     # 25.7 TFLOP/s effective
HZ = 2.0

# name: params, evidence, input rate, tokens entering the transformer per 2 s, layers, width, extra (conv) GFLOP, tokens OUT per 2 s
TOWERS = {
    "whisper-large-v3 encoder (2 s block, not the 30 s pad)": dict(P=636_968_960, ev="MEASURED header", sr=16000, n=100, L=32, d=1280, conv=1.2, out=50,
        note="Whisper itself pads every input to 30 s (1500 positions) -> x15 the FLOPs; Qwen2.5-Omni runs it in 2 s blocks (report p.5)"),
    "whisper-large-v3 encoder, 30 s padded (native)": dict(P=636_968_960, ev="MEASURED header", sr=16000, n=1500, L=32, d=1280, conv=18.0, out=750, note="for contrast"),
    "Qwen2.5-Omni-3B audio tower (Whisper-v3 init + Linear 1280->2048)": dict(P=637_676_544, ev="MEASURED header", sr=16000, n=100, L=32, d=1280, conv=1.2, out=50,
        note="Qwen RESEARCH licence (non-commercial)"),
    "Qwen3-Omni AuT encoder (thinker.audio_tower)": dict(P=647_927_168, ev="MEASURED header", sr=16000, n=25, L=32, d=1280, conv=8.3, out=25,
        note="3 x Conv2d(3x3, s2, 480 ch) front end ~8.3 GFLOP per 2 s (computed from config)"),
    "Gemma-4 E2B/E4B audio tower": dict(P=304_825_088, ev="MEASURED header", sr=16000, n=50, L=12, d=1024, conv=0.5, out=50, note="40 ms/token"),
    "Gemma-3n E4B audio tower (USM-based)": dict(P=681_348_640, ev="MEASURED header (unsloth mirror; google repo gated)", sr=16000, n=50, L=12, d=1536, conv=0.5, out=12.5,
        note="conformer runs at 40 ms frames, output decimated x4 -> 160 ms/token"),
    "BEATs-iter3 base": dict(P=90_000_000, ev="ESTIMATED (12 x 768, ffn 3072, patch 16x16; weights not on HF)", sr=16000, n=96, L=12, d=768, conv=0.1, out=96,
        note="128 fbank x 200 frames -> 12 x 8 patches"),
    "CLAP HTSAT (laion/clap-htsat-unfused audio branch)": dict(P=28_000_000, ev="ESTIMATED (Swin-T-like: depths 2-2-6-2, width 96)", sr=48000, n=None, L=12, d=768, conv=0.0, out=1,
        note="fixed 10 s @ 48 kHz input (repeat-pad), ~9 GFLOP per call ESTIMATED; one pooled 512-d embedding"),
}


def cost(t):
    if t["n"] is None: fl = 9e9
    else: fl = 2 * t["P"] * t["n"] + 4 * t["L"] * t["n"] ** 2 * t["d"] + t["conv"] * 1e9
    w = 2 * t["P"] / BW
    c = fl / FLOPS
    T = max(w, c)
    return dict(params=t["P"], params_evidence=t["ev"], input_sr_hz=t["sr"], tokens_in_per_2s=t["n"], tokens_out_per_2s=t["out"],
                gflop_per_window=round(fl / 1e9, 1), weight_floor_ms=round(w * 1e3, 2), compute_ms=round(c * 1e3, 2),
                T_window_ms_ESTIMATED=round(T * 1e3, 2), duty_at_2Hz=round(HZ * T, 4), note=t["note"])


if __name__ == "__main__":
    out = {"conventions": dict(weight_stream_Bps=BW, eff_flops=FLOPS, refresh_hz=HZ, evidence="ESTIMATED (arithmetic); params as marked"),
           "towers": {k: cost(v) for k, v in TOWERS.items()}}
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rg_audio_tower_cost_result.json")
    json.dump(out, open(p, "w"), indent=1)
    for k, v in out["towers"].items(): print(f"{k:70s} {v['T_window_ms_ESTIMATED']:7.2f} ms  duty {v['duty_at_2Hz']:.4f}  {v['gflop_per_window']} GFLOP")
