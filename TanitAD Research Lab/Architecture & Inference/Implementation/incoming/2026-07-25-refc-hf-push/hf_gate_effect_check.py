"""Prove the gate actually bites: an ANONYMOUS (no-token) fetch of ckpt.pt must be refused,
while the public card metadata is readable. No token is used or needed by this script."""
import requests

for repo in ("Sayood/tanitad-refc-xl", "Sayood/tanitad-refc-base"):
    print(f"=== {repo} (anonymous, no Authorization header) ===")
    # public metadata should be visible
    r = requests.get(f"https://huggingface.co/api/models/{repo}", timeout=30)
    print(f"  api/models        -> {r.status_code}")
    if r.status_code == 200:
        d = r.json()
        print(f"     private={d.get('private')} gated={d.get('gated')} "
              f"files={[s.get('rfilename') for s in d.get('siblings', [])]}")
    # weights must NOT be downloadable
    u = f"https://huggingface.co/{repo}/resolve/main/ckpt.pt"
    r2 = requests.get(u, timeout=30, allow_redirects=False, stream=True)
    print(f"  resolve/ckpt.pt   -> {r2.status_code} "
          f"(expect 401/403 = gated; 200/302 would mean WORLD-DOWNLOADABLE)")
    print(f"     x-error-code: {r2.headers.get('x-error-code')}")
    print(f"     x-error-message: {r2.headers.get('x-error-message')}")
    r2.close()
print("GATE_EFFECT_CHECK_DONE")
