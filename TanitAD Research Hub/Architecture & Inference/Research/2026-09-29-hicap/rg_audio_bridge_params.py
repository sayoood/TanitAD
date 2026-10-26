"""Stream A: MEASURED trainable-parameter counts of stack/tanitad/hicap/audio.py (AudioBridge, EventHeads, BridgeAlignLoss) and of the
existing model.GatedAudio when HiCAPConfig.d_audio = d_v, at d_v = 2048 / 4096, for an AuT/Whisper-width tower (1280) and BEATs (768),
mono and a 4-mic array.  Run from the repo root: PYTHONPATH=stack python3 "<this file>"  -> rg_audio_bridge_params_result.json."""
import json, os
from tanitad.hicap import audio as A
from tanitad.hicap.model import GatedAudio, HiCAPConfig

cfg, res = A.AudioConfig(), {}
for d_v in (2048, 4096):
    for name, d_enc in (("AuT/Whisper-large d_enc=1280", 1280), ("BEATs d_enc=768", 768)):
        for ch in (1, 4):
            br = A.AudioBridge(A.BridgeConfig(d_enc=d_enc, d_v=d_v, d_spatial=cfg.d_spatial(ch)))
            h, al, ga = A.EventHeads(d_v), A.BridgeAlignLoss(), GatedAudio(HiCAPConfig(d_v=d_v, d_audio=d_v))
            sub = {n: sum(p.numel() for p in m.parameters()) for n, m in br.named_children()}
            sub["latents+type_emb"] = br.latents.numel() + br.type_emb.numel()
            res[f"d_v={d_v} | {name} | mics={ch}"] = dict(
                bridge=A.n_params(br), event_heads=A.n_params(h), align_logit_scale=A.n_params(al),
                bridge_plus_heads=A.n_params(br) + A.n_params(h),
                model_GatedAudio_at_d_audio_eq_d_v=sum(p.numel() for p in ga.parameters()), bridge_breakdown=sub)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rg_audio_bridge_params_result.json")
json.dump(dict(evidence="MEASURED (code count, default BridgeConfig width 512, 2 layers, 8 latents, MLP hidden 1024)", rows=res), open(out, "w"), indent=1)
for k, v in res.items(): print(f"{k:45s} bridge {v['bridge']:>10,}  heads {v['event_heads']:>9,}  total {v['bridge_plus_heads']:>10,}  GatedAudio {v['model_GatedAudio_at_d_audio_eq_d_v']:>9,}")
