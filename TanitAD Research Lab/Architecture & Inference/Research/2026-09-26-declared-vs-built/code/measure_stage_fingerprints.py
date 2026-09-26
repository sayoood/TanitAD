"""FIX-5 instrument: the per-stage pretrained-weight fingerprints pinned in `timm_trunk.py`.

For each measured backbone, sha256[:16] of the fp32 bytes of the FIRST conv of every stage
(layer1..layer4) of the timm `features_only` model built with `pretrained=True`, compared with the
same tensor read straight from the cached checkpoint file, plus |w|.sum() of that conv for the
pretrained and a random-init build (why a |w|.sum() band is NOT a safe per-stage witness).
Offline (HF_HUB_OFFLINE=1): reads the local HF cache only.
"""
import hashlib, json, re, sys
from pathlib import Path
import torch, timm
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

out = {"what": "FIX-5 per-stage fingerprints", "evidence_class": "MEASURED (dev box, CPU, offline HF cache)",
       "timm": timm.__version__, "torch": torch.__version__, "backbones": {}}
for name in ("resnet34.a1_in1k", "resnet101.a1_in1k"):
    net = timm.create_model(name, pretrained=True, features_only=True, out_indices=(3, 4))
    torch.manual_seed(0)
    rnd = timm.create_model(name, pretrained=False, features_only=True, out_indices=(3, 4))
    path = hf_hub_download(f"timm/{name}", "model.safetensors")
    sd = load_file(path)
    groups = {}
    for n, m in net.named_modules():
        g = re.match(r"^(layer\d+|stages[._]\d+)\.", n)
        if isinstance(m, torch.nn.Conv2d) and g and g.group(1) not in groups:
            groups[g.group(1)] = n
    rmods = dict(rnd.named_modules())
    rec = {}
    for stage, n in sorted(groups.items()):
        w = dict(net.named_modules())[n].weight.detach().float().contiguous().cpu()
        f = sd[n + ".weight"].float().contiguous()
        wr = rmods[n].weight.detach().float()
        rec[n] = {"sha16_built": hashlib.sha256(w.numpy().tobytes()).hexdigest()[:16],
                  "sha16_file": hashlib.sha256(f.numpy().tobytes()).hexdigest()[:16],
                  "bit_equal_to_file": bool(torch.equal(w, f)),
                  "abs_sum_pretrained": float(w.abs().sum()),
                  "abs_sum_random_init_seed0": float(wr.abs().sum()),
                  "ratio_random_over_pretrained": float(wr.abs().sum() / w.abs().sum())}
    out["backbones"][name] = {"checkpoint": str(path), "stages": rec}
Path(sys.argv[1]).write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out, indent=1))
