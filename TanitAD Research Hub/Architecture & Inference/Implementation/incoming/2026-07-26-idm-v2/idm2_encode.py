"""IDM-v2 substrate: encode every eval-pod episode with the FROZEN flagship-v1
encoder, once, to per-episode latents.  Reuses run_idm_proof.load_encoder /
encode_frames verbatim (no re-implementation) so the latents are bit-identical
to the ones idm_head_v1 was trained on (proved by idmval_zcmp cosine 1.0000).

Out: /root/idm2/lat/<tag>.pt = {z fp16 [T,2048], poses [T,4], actions [T,2],
                                episode_id, domain}
"""
from __future__ import annotations
import hashlib, json, sys, time
from pathlib import Path
import torch

sys.path.insert(0, "/root/v4eval/stack")
sys.path.insert(0, "/root/v4eval/stack/scripts")
import run_idm_proof as rip  # noqa: E402

CKPT = "/root/models/flagship-30k/ckpt.pt"
OUT = Path("/root/idm2/lat")
CACHES = [
    ("pai", "/root/valdata/physicalai-val-0c5f7dac3b11"),
    ("cm",  "/root/valdata/comma2k19-val-76b6e94a97a1"),
]


def md5_of(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    dev = "cuda"
    m5 = md5_of(CKPT)
    print("encoder md5", m5, flush=True)
    assert m5 == "b5f07d9e3dd2ca643949bc86832e6585", "NOT the flagship-v1 encoder"
    enc, ro, meta = rip.load_encoder(CKPT, dev)
    print("meta", meta, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    man = []
    t0 = time.time()
    for dom, cache in CACHES:
        eps = sorted(Path(cache).glob("*.pt"))
        for i, p in enumerate(eps):
            tag = f"{dom}_{i:05d}"
            lf = OUT / f"{tag}.pt"
            d = torch.load(p, weights_only=False)
            rec = {"tag": tag, "domain": dom, "src": str(p),
                   "episode_id": int(d.get("episode_id", -1)),
                   "T": int(d["poses"].shape[0])}
            man.append(rec)
            if lf.exists():
                continue
            z = rip.encode_frames(enc, ro, d["frames_u8"], dev, batch=32)
            torch.save({"z": z, "poses": d["poses"].float(),
                        "actions": d["actions"].float(),
                        "episode_id": rec["episode_id"], "domain": dom,
                        "src": str(p)}, lf)
            if i % 10 == 0:
                print(f"[{time.time()-t0:.0f}s] {tag} z{tuple(z.shape)}", flush=True)
    Path("/root/idm2/manifest.json").write_text(json.dumps(
        {"encoder_ckpt": CKPT, "encoder_md5": m5, "meta": meta,
         "episodes": man}, indent=1))
    print("ENCODE_DONE", len(man), f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
