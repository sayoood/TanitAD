"""R7, the LITERAL form: in content mode (the default) the gate behaves bit-for-bit as the tip's does.  Builds one real sealed cache, runs scripts/r7_worker.py once against the TIP's refe/ (base) and once against mine, and
compares every recorded observable (fingerprints, data manifest, profile refusals, verify rc/stdout/record, refusals rc/stdout, the minted token's binding, the NUMBER OF FULL-FILE READS of each call), modulo time / hmac / side root.
Also compares mine-with-an-explicit `--ctx-mode content` against base-without-the-flag.   usage: python r7_identity_vs_tip.py <out.json>     (cwd anywhere; CPU only; one python at a time)
"""
import json
import os
import shutil
import subprocess
import sys

RP = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
BASE = f"C:/Users/Admin/par18/hnfp3/base/{RP}/refe"
MINE = os.environ.get("R7_MINE") or f"C:/Users/Admin/par18/hnfp3/work/{RP}/refe"        # R7_MINE: a MUTATED copy of mine (r7_control.py) - the differential must then report a difference
TMP = os.environ.get("R7_TMP") or "C:/Users/Admin/par18/hnfp3/tmp/r7"
OUT = sys.argv[1] if len(sys.argv) > 1 else "r7_identity_vs_tip.json"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
shutil.rmtree(TMP, ignore_errors=True)
os.makedirs(TMP)

sys.path.insert(0, MINE)
import ctx_cache as CC  # noqa: E402
import test_ctx_cache as TCC  # noqa: E402
import torch  # noqa: E402,F401

units = TCC.mk_units()
model = TCC.Fake().eval()
fp = CC.trunk_fingerprint(model.state_dict())
pristine = f"{TMP}/pristine"
CC.build_units(model, units, TCC.load_batch, TCC.make_forward(model), pristine, trunk_fp=fp, log=lambda *_: None)
CC.finalize(pristine, units, trunk_fp=fp, calib_sha256="calib-1")
print("pristine cache built", flush=True)

import gate_token as _GT0                                       # noqa: E402
_GT0.load_key(f"{TMP}/shared.key", create=True)
res = {}
for side, refe in (("base", BASE), ("mine", MINE)):
    o = f"{TMP}/{side}.json"
    r = subprocess.run([sys.executable, "C:/Users/Admin/par18/hnfp3/scripts/r7_worker.py", side, refe, pristine, f"{TMP}/root", o], capture_output=True, text=True, timeout=1500, encoding="utf-8", errors="replace",
                       env=dict(os.environ, R7_KEY=f"{TMP}/shared.key", PYTHONIOENCODING="utf-8", CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1"))
    print(side, "rc", r.returncode, (r.stdout or "").strip()[-200:], (r.stderr or "")[-600:], flush=True)
    if r.returncode != 0:
        sys.exit(3)
    res[side] = json.load(open(o, encoding="utf-8"))

b, m = res["base"], res["mine"]
rows, ndiff = [], 0


def cmp(label, x, y):
    global ndiff
    same = json.dumps(x, sort_keys=True) == json.dumps(y, sort_keys=True)
    rows.append({"observable": label, "identical": same})
    if not same:
        ndiff += 1
        print("DIFF", label, "\n  base:", json.dumps(x, sort_keys=True)[:700], "\n  mine:", json.dumps(y, sort_keys=True)[:700])


cmp("fingerprints_expected (the fill)", b["fx"], m["fx"])
cmp("mint: verdict + token binding + reads", b["mint"], m["mint"])
for sc in b["scenarios"]:
    for k, v in b["scenarios"][sc].items():
        cmp(f"{sc}/{k}", v, m["scenarios"][sc][k])
    for k, v in m["scenarios"][sc].items():
        if k.endswith("_explicit_content"):                                  # mine with --ctx-mode content == base without the flag
            cmp(f"{sc}/{k} vs base plain", b["scenarios"][sc][k.replace("_explicit_content", "")], v)
extra = sorted(set(m["scenarios"]["clean"]) - set(b["scenarios"]["clean"]))
summary = {"n_observables": len(rows), "n_identical": sum(r["identical"] for r in rows), "n_different": ndiff, "scenarios": list(b["scenarios"]), "mine_only_keys": extra,
           "reads_clean": {k: b["scenarios"]["clean"][k].get("reads") for k in b["scenarios"]["clean"]}, "evidence": "MEASURED (this script, tip vs mine, same sealed cache)"}
json.dump({"summary": summary, "rows": rows}, open(OUT, "w", encoding="utf-8"), indent=1)
print(json.dumps(summary, indent=1))
print("ZZR7_IDENTICAL" if ndiff == 0 else f"ZZR7_DIFFERENT {ndiff}")
sys.exit(0 if ndiff == 0 else 1)
