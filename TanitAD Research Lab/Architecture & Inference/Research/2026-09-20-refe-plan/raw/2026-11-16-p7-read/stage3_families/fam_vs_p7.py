"""Stage 3's families_vs for P7 (base P6, also P2), recomputed after families6 ran through families6_lowmem.py.
Exactly stage3_full.families_vs: PE.fam_adverse(base, arm), REPORTED NOT GATING. Inputs md5-checked against stage3_full's pins."""
import hashlib, json, sys
TIP = "F:/TanitAD_p7read_tip/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
sys.path.insert(0, TIP + "/eval")
import proxy_eval as PE  # noqa: E402
assert PE.__file__.replace("\\", "/").startswith(TIP), PE.__file__
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
ARM = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p7_next/fam6lowmem/run/p7_full_families.json"
BASE = "F:/TanitAD_p6read_work/stage3_p6_full/p6_full_families.json"
ALSO = TIP + "/raw/2026-10-05-sft-stage2/p2_full/p2_full_families.json"
out = {"what": "stage3 families_vs for P7 (R2: vs P6; R2b: vs P2), recomputed 2026-11-17 after the families6 MemoryError",
       "note": "REPORTED, NOT GATING (amendment section 5); unpaired interval non-overlap, a conservative reading",
       "inputs": {"arm": [ARM, md5(ARM)], "base_p6": [BASE, md5(BASE)], "also_p2": [ALSO, md5(ALSO)]},
       "base_p6_md5_pin_prefix": "b9dcbf62"}
assert md5(BASE).startswith("b9dcbf62"), md5(BASE)
fa, fb, fc = (json.load(open(p, encoding="utf-8")) for p in (ARM, BASE, ALSO))
out["R2_families_vs_base_p6"] = PE.fam_adverse(fb, fa)
out["R2b_families_vs_also_p2"] = PE.fam_adverse(fc, fa)
json.dump(out, open("E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p7_next/fam6lowmem/run/p7_families_vs.json", "w",
                    encoding="utf-8"), indent=1, default=str)
for k in ("R2_families_vs_base_p6", "R2b_families_vs_also_p2"):
    print(k, json.dumps(out[k], default=str)[:900])
