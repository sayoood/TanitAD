"""BEFORE / AFTER through the gate's own entry point: does `launch_gate.py refusals` (cmd_refusals) PASS a --ctx-cache whose payload had ONE BIT flipped at the same size?
usage: repro_same_size_flip.py <refe dir>      (run once on the base tree, once on the fixed tree; prints one JSON line)
The profile is filled the way each tree fills it: base = the size-listing key (hn_fingerprints.py on base), fixed = hn_fingerprints.py's content key.  Everything else is identical."""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(HERE))
os.chdir(HERE)
import gate_checks as C   # noqa: E402
import gate_token as GT   # noqa: E402
import hn_fingerprints as HF  # noqa: E402
import launch_gate as L   # noqa: E402
import ctx_cache as CC    # noqa: E402
import test_ctx_cache as TCC  # noqa: E402

root = Path(tempfile.mkdtemp(prefix="repro_", dir=os.environ.get("TEMP")))
units = TCC.mk_units()
model = TCC.Fake().eval()
fp = CC.trunk_fingerprint(model.state_dict())
cache = root / "cache"
CC.build_units(model, units, TCC.load_batch, TCC.make_forward(model), str(cache), trunk_fp=fp, log=lambda *_: None)
CC.finalize(str(cache), units, trunk_fp=fp, calib_sha256="calib-1")
bank = root / "bank.hnb"
bank.write_bytes(b"HNBANK1\n" + bytes(range(200)))
fx = HF.fingerprints_expected(str(bank), str(cache))                      # the tree's OWN fill procedure, from the pristine cache
c1 = list(json.loads((HERE.parent / "eval" / "rls1b_arms.json").read_text(encoding="utf-8"))["arms"]["C1"]["argv"])
dummy = root / "dummy.bin"
dummy.write_bytes(b"x")
P = L.PROFILES["refe_hnscreen"]
for f_ in P["data_flags"]:
    if f_ in c1 and f_ not in L.HNSCREEN_FLAGS:
        c1 = C._set(c1, f_, str(dummy))


def argv_for(c):
    return C._set(c1, "--seed", "1") + ["--hn-bank", str(bank), "--ctx-cache", str(c)]


flipped = root / "flipped"
shutil.copytree(cache, flipped)
p = sorted((flipped / "units").glob("*.ctx.u16"))[0]
b = bytearray(p.read_bytes())
b[len(b) // 2] ^= 1
p.write_bytes(bytes(b))
assert p.stat().st_size == sorted((cache / "units").glob("*.ctx.u16"))[0].stat().st_size


def refusals(c):
    af = root / "argv.json"
    GT.write_json(af, argv_for(c))
    old = L.PROFILES["refe_hnscreen"]
    L.PROFILES["refe_hnscreen"] = {**P, "fingerprints_expected": dict(fx)}
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            rc = L.main(["refusals", "--profile", "refe_hnscreen", "--arm-name", "S1", "--args-file", str(af)])
    finally:
        L.PROFILES["refe_hnscreen"] = old
    return rc, [ln for ln in buf.getvalue().splitlines() if ln.startswith("REFUSED") or "ZZGATE" in ln]


rc_ok, out_ok = refusals(cache)
rc_fl, out_fl = refusals(flipped)
print(json.dumps({"tree": str(HERE.parent.parent.parent.name) + "/" + HERE.parent.parent.name, "fill": {k: v[:40] for k, v in fx.items()},
                  "pristine_cache": {"rc": rc_ok, "out": out_ok}, "same_size_flipped_cache": {"rc": rc_fl, "out": [o[:230] for o in out_fl]},
                  "verdict": ("DEFECT: the gate PASSES a same-size byte flip" if rc_fl == 0 else "the gate REFUSES the same-size byte flip") if rc_ok == 0 else "pristine control was refused: invalid run"}, indent=1))
shutil.rmtree(root, ignore_errors=True)
