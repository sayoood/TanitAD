"""fetch_weights.py -- card ADD127: the ONE download the PI approved. Fetches the COCO weights of torchvision's fasterrcnn_resnet50_fpn_v2 into TORCH_HOME (the card's own folder),
with torch's own hash check of the file name's eight characters, and writes WEIGHTS_RECORD.json beside this script's output folder. CPU only; builds no model; installs nothing."""
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import torch  # noqa: E402
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights  # noqa: E402


def main():
    out = sys.argv[1]
    home = os.environ.get("TORCH_HOME")
    if not home or not home.startswith("/workspace/v8proof/boxread/"):
        print("ZZ FETCH STOP TORCH_HOME is not the card's own folder")
        return 3
    w = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    url = w.url
    name = url.rsplit("/", 1)[1]
    t0 = time.time()
    torch.hub.load_state_dict_from_url(url, model_dir=os.path.join(home, "hub", "checkpoints"), map_location="cpu", progress=False, check_hash=True)
    path = os.path.join(home, "hub", "checkpoints", name)
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    tag = name.rsplit("-", 1)[1].split(".")[0]
    rec = {"file": name, "bytes": os.path.getsize(path), "sha256": h.hexdigest(), "name_tag": tag, "sha256_begins_with_name_tag": h.hexdigest().startswith(tag),
           "host": url.split("/")[2], "seconds": round(time.time() - t0, 1), "categories_n": len(w.meta["categories"]),
           "vehicle_ids": {c: w.meta["categories"].index(c) for c in ("car", "motorcycle", "bus", "truck")}, "torchvision": __import__("torchvision").__version__}
    with open(os.path.join(out, "WEIGHTS_RECORD.json"), "x", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    print("ZZ FETCH DONE", rec["file"], rec["bytes"], rec["sha256"][:16], rec["sha256_begins_with_name_tag"], rec["vehicle_ids"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
