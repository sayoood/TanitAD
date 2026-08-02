#!/bin/bash
# read-only probe of pod2's two v5 caches
set -u
T=/workspace/data/pai_wide120_v2png_train
V=/workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl
echo "=== host ==="; hostname; date -u
echo "=== gpu ==="; nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo "=== train dir ==="
ls "$T" | grep -c 'v2ep.pt$'
ls -la "$T"/_geometry.json* "$T"/_v2manifest.pt 2>&1
echo "=== val dir ==="
ls "$V" | grep -c 'v2ep.pt$'
ls -la "$V"/_geometry.json* "$V"/_v2manifest.pt 2>&1
echo "=== geometry json values ==="
python3 - <<'PY'
import json
for tag, p in (("train", "/workspace/data/pai_wide120_v2png_train/_geometry.json"),
               ("val", "/workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl/_geometry.json")):
    d = json.load(open(p))
    g = d.get("geometry_check", {})
    print(tag, "top-level keys:", sorted(d.keys()))
    print(tag, "observed_frac      =", g.get("observed_frac"))
    print(tag, "observed_frac_supersed =", g.get("observed_frac_superseded"))
    print(tag, "frame =", json.dumps(d.get("frame")))
    print(tag, "frame_tag =", d.get("frame_tag"))
    print(tag, "rig_observability present:", "rig_observability" in d)
    print(tag, "subframe_observability present:", "subframe_observability" in d)
    print(tag, "corrections:", len(d.get("corrections") or []))
PY
echo "=== stack HEAD on pod2 ==="
cd /workspace/TanitAD 2>/dev/null && git log --oneline -1 && git status --short | head -20
echo "=== parity manifest on pod2 ==="
python3 - <<'PY'
import json
p = "/workspace/TanitAD/stack/tanitad/data/parity_manifest.json"
try:
    d = json.load(open(p))
except Exception as e:
    print("ERR", e); raise SystemExit
c = d.get("corpora", d)
print("keys:", list(c.keys()) if isinstance(c, dict) else type(c))
PY
echo "=== paritysplit lists ==="
ls -la /workspace/wfov/paritysplit/ 2>&1 | head -20
ls -la /workspace/rigfix/ 2>&1 | head -20
echo "=== disk (dd, NOT df) ==="
dd if=/dev/zero of=/workspace/.ddtest bs=1M count=500 conv=fsync 2>&1 | tail -1; rm -f /workspace/.ddtest
