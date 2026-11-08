"""Writes / checks CODE_PIN.json: sha256 of every file in this folder except CODE_PIN.json, HANDOFF.md, MANIFEST.md (those cite the pin).
python make_pin.py          write      |      python make_pin.py --check   verify (exit 1 on any difference)"""
import hashlib, json, sys
from pathlib import Path
import numpy, scipy
HERE = Path(__file__).resolve().parent
SKIP = {"CODE_PIN.json", "HANDOFF.md", "MANIFEST.md"}
SCORECARD = "fac6c7f150eb026de0708c6f6557df52481216f76b35f3c8c8b8045bcb8ec0c5"
def files():
    return {str(p.relative_to(HERE)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(HERE.rglob("*")) if p.is_file() and p.name not in SKIP and "__pycache__" not in p.parts}
if __name__ == "__main__":
    cur = files()
    pin = dict(amendment_3_sha256="886e44654566c82bf2f3f6146217a8ee54b01a1a4af4e7d98b99d0b6578d1035", amendment_5_sha256=open(HERE.parent / "AMENDMENT_5_QC.sha256").read().split()[0], scorecard_sha256=SCORECARD, scorecard_path="E:/Projects/TanitAD-artifacts/refe_behaviour_20261107/BEHAVIOUR_SCORECARD.md",
               schema_version="refe-rollout-v1", python="3.11 (C:/Users/Admin/venvs/driverl-eval)", numpy=numpy.__version__, scipy=scipy.__version__, files=cur)
    f = HERE / "CODE_PIN.json"
    if "--check" in sys.argv:
        old = json.loads(f.read_text())["files"]
        bad = sorted(k for k in set(old) | set(cur) if old.get(k) != cur.get(k))
        print("PIN OK" if not bad else "PIN MISMATCH: " + ", ".join(bad)); sys.exit(1 if bad else 0)
    f.write_text(json.dumps(pin, indent=1, sort_keys=True), encoding="utf-8"); print("wrote", f, len(cur), "files")
