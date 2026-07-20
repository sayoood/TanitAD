"""TanitEval — run the state_dict profiler over the whole registry -> results/profile.json."""
import json
import os
from taniteval.registry import MODELS
from taniteval import profiling

os.makedirs("/root/taniteval/results", exist_ok=True)
out = {}
print(f"{'model':24}{'trained':>10}{'+frozen':>10}{'=infer':>10}{'ckpt':>9}")
for m in MODELS:
    p = profiling.analyze_state_dict(m["ckpt"], encoder=m["encoder"],
                                     encoder_frozen=m["encoder_frozen"])
    out[m["key"]] = {
        "meta": {k: m.get(k) for k in ("name", "arch", "encoder", "encoder_frozen",
                                       "speed_input", "action_dim", "step", "hf",
                                       "anti_collapse", "note")},
        "profile": p, "gate": m.get("gate", {}),
    }
    print(f"{m['name']:24}{p['trained_params_m']:9.1f}M{p['frozen_encoder_params_m']:9.1f}M"
          f"{p['total_inference_params_m']:9.1f}M{p['ckpt_mb']:8.0f}M")
    top = list(p["by_component_m"].items())[:5]
    print(f"    components: " + ", ".join(f"{k} {v:.1f}M" for k, v in top))
json.dump(out, open("/root/taniteval/results/profile.json", "w"), indent=2, default=str)
print("\nwrote /root/taniteval/results/profile.json")
