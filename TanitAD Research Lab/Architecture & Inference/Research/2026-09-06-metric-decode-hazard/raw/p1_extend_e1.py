"""P1d - do the FOUR T1 artifacts the 82-row sweep did NOT name also have
register rows quoting their metres?  (postrain30k, postrain30k_freeze,
rdw8p30k nonparity40, v7 smoke)

Every number is searched in BOTH bare and comma-formatted form, and every
search carries a same-breath control that must read non-zero.
"""
import json
import os
import re

CLONE = r"C:\Users\Admin\refcv4b_repo"
REG = r"C:\Users\Admin\tanitad-mdhazard\_ref\GOALS_AND_CLAIMS.md"
OUT = r"C:\Users\Admin\tanitad-mdhazard\_work\p1_extend_e1.json"

PATHS = {
 "postrain30k": r"TanitAD Research Lab\Architecture & Inference\Implementation\incoming\2026-08-27-t1-parity-first\raw\t1_postrain30k.json",
 "postrain30k_freeze": r"TanitAD Research Lab\Architecture & Inference\Implementation\incoming\2026-08-27-t1-parity-first\raw\t1_postrain30k_freeze.json",
 "rdw8p30k_nonparity40": r"TanitAD Research Lab\Architecture & Inference\Research\2026-08-24-action-conditioning-and-heldout\raw\t1_nonparity40_rdw8p30k.json",
 "v7_smoke": r"TanitAD Research Lab\Architecture & Inference\Research\2026-08-24-action-conditioning-and-heldout\raw\t1_v7_smoke.json",
}

reg = open(REG, encoding="utf-8", errors="replace").read()
ROWKEY = re.compile(r"^\|\s*[^|]*?\b([A-Z][A-Z0-9-]{3,}[A-Z0-9])\b", re.M)


def forms(x):
    f = float(x)
    out = set()
    for nd in (2, 3, 4):
        out.add(f"{f:.{nd}f}")
        out.add(f"{f:,.{nd}f}")
    return sorted(out)


def rows_for(needle):
    ids = set()
    for m in re.finditer(re.escape(needle), reg):
        start = reg.rfind("\n|", 0, m.start())
        seg = reg[start:start + 400]
        km = ROWKEY.search(seg)
        if km:
            ids.add(km.group(1))
    return sorted(ids)


res = {}
for arm, rel in PATHS.items():
    p = os.path.join(CLONE, rel)
    if not os.path.isfile(p):
        res[arm] = {"status": "NOT_ON_THIS_BOX"}
        continue
    d = json.load(open(p, encoding="utf-8"))
    out = {"ckpt": d.get("ckpt"), "decoder": d.get("decoder"),
           "n_windows": d.get("n_windows"), "n_episodes": d.get("n_episodes"),
           "metrics": {}, "register": {}}
    for armk in ("cl", "ha", "ol"):
        mt = (((d.get("arms") or {}).get(armk) or {}).get("intervals")
              or {}).get("metrics") or {}
        for k in ("ade_dense_m", "fde_last_m", "LON_speed_mae_mps",
                  "LAT_cross_mae_m", "LAT_heading_mae_deg"):
            v = mt.get(k)
            if isinstance(v, dict):
                v = v.get("mean", v.get("point"))
            if v is None:
                continue
            out["metrics"][f"{armk}.{k}"] = v
            hits = {}
            for s in forms(v):
                n = len(re.findall(re.escape(s), reg))
                if n:
                    hits[s] = {"count": n, "rows": rows_for(s)}
            if hits:
                out["register"][f"{armk}.{k}"] = hits
    res[arm] = out

# same-breath controls
res["_controls"] = {
    "must_be_nonzero_H-ESTIM-SEED-1": len(re.findall("H-ESTIM-SEED-1", reg)),
    "must_be_nonzero_14.069": len(re.findall(re.escape("14.069"), reg)),
    "must_be_zero_nonsense": len(re.findall("ZZ_MUST_BE_ZERO_ZZ", reg)),
    "register_bytes": len(reg),
}

with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=1)

for arm, r in res.items():
    if arm.startswith("_"):
        continue
    print(f"== {arm}  ckpt={r.get('ckpt')}")
    print(f"   decoder={r.get('decoder')}  n_win={r.get('n_windows')} "
          f"n_ep={r.get('n_episodes')}")
    print(f"   metrics: {json.dumps(r.get('metrics', {}))[:300]}")
    if r.get("register"):
        for k, v in r["register"].items():
            print(f"   REGISTER HIT {k}: {json.dumps(v)}")
    else:
        print("   REGISTER HIT: none of its metric numbers appear")
print("\ncontrols:", json.dumps(res["_controls"]))
print("[wrote]", OUT)
