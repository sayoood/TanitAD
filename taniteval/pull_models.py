"""Pull the 4 gated TanitAD models from HF into /root/models/ on the eval pod.
Token is read from STDIN (piped in-place from a training pod's Keys.txt) — never
written to a file, a command arg, or printed. Not persisted after the run."""
import sys
try:
    import truststore; truststore.inject_into_ssl()
except Exception:
    pass
from huggingface_hub import snapshot_download

token = sys.stdin.read().strip()
assert token.startswith("hf_"), "no token on stdin"
repos = [
    "Sayood/tanitad-flagship-4b-phase0",
    "Sayood/tanitad-refb-speed",
    "Sayood/tanitad-refa-dinov2-4b",
    "Sayood/tanitad-refa-ijepa-4b",
]
for r in repos:
    name = r.split("/")[1]
    print(f"[pull] {r} ...", flush=True)
    p = snapshot_download(r, token=token, local_dir=f"/root/models/{name}",
                          allow_patterns=["*.pt", "*.json", "*.md"])
    print(f"[pull] {name} done", flush=True)
print("PULL_DONE", flush=True)
