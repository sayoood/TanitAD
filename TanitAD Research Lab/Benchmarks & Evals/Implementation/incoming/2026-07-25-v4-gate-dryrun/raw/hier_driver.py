"""Run taniteval.hierarchy for (a) flagship v1 (the gate_emitters dry-run fixture)
and (b) the v4.2b checkpoint, WITHOUT editing any pod file: the v4 entry is
appended to registry.MODELS at runtime.

Emits the JSON that gate_emitters.py `nonav-route` reads."""
import sys, json, traceback
sys.path.insert(0, "/root/taniteval")
from taniteval import data, loaders
from taniteval.registry import MODELS
import taniteval.hierarchy as H

# v4's TRUNK is architecturally the v1 fourbrain WorldModel (ck["model"] +
# ck["grounding"]); the v4 head is separate. hierarchy probes the TRUNK's
# strategic route seam, which is exactly what nonav_route_beats_majority names.
V4_ENTRY = dict(key="flagship-v4.2b-dryrun", name="flagship v4.2b (dry-run)",
                family="TanitAD", arch="flagship-worldmodel",
                ckpt="/workspace/v4gate/ckpt/ckpt.pt",
                config="flagship4b", encoder="trained ViT-12 (9ch, 256px)",
                encoder_frozen=False, speed_input=True, action_dim=3,
                note="TEMP runtime entry for the 2026-07-25 v4 gate dry-run")
MODELS.append(V4_ENTRY)

files = data.list_val_episodes("/root/valdata/physicalai-val-0c5f7dac3b11", 40)

for key, out in (("flagship-30k", "/workspace/v4gate/results/hierarchy_flagship-30k.json"),
                 ("flagship-v4.2b-dryrun",
                  "/workspace/v4gate/results/hierarchy_flagship-v4.2b-dryrun.json")):
    print(f"\n########## hierarchy: {key} ##########", flush=True)
    try:
        e = [m for m in MODELS if m["key"] == key][0]
        L = loaders.load(e, "cuda")
        eps = (data.load_frames(files) if L["feed"] == "frames"
               else data.load_features(files, L["feed"], "cuda"))
        res = H.run(L["model"], L["step_readout"], eps, "cuda",
                    speed_input=bool(e.get("speed_input")), max_eps=40, stride=8,
                    yaw_input=bool(e.get("yaw_input")),
                    dyn_input=bool(e.get("dyn_input")))
        with open(out, "w") as fh:
            json.dump(res, fh, indent=2, default=str)
        sn = res.get("seam_nav_to_strategic", {})
        print(f"[{key}] OK -> {out}")
        print(f"   route_acc_follow={sn.get('route_acc_follow')} "
              f"majority_straight_rate={sn.get('majority_straight_rate')} "
              f"vision_route_beats_majority={sn.get('vision_route_beats_majority')}",
              flush=True)
    except Exception:
        print(f"[{key}] FAILED:\n{traceback.format_exc()}", flush=True)
print("\n########## HIERARCHY DONE ##########", flush=True)
