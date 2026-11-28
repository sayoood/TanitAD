"""P1b - WHICH published numbers came out of the untrained readout?

Reads the banked T1 artifacts, pulls the decoder provenance + the headline
metric-space numbers, and then greps GOALS_AND_CLAIMS.md for each number in
BOTH its bare and comma-formatted forms (the 2026-09-04 lesson).
"""
import json
import os
import re
import hashlib

REPO_LOCAL = r"C:\Users\Admin\refcv4b_repo"
INC = os.path.join(REPO_LOCAL, "TanitAD Research Lab",
                   "Architecture & Inference", "Implementation", "incoming")
REG = r"C:\Users\Admin\tanitad-mdhazard\_ref\GOALS_AND_CLAIMS.md"
OUT = r"C:\Users\Admin\tanitad-mdhazard\_work\p1_consumer_trace.json"

T1 = {
    "emao14_30k": ("2026-08-30-t1-first-v7-read", "t1_emao14_30k.json"),
    "emao14_30k_tauramp": ("2026-08-30-t1-first-v7-read",
                           "t1_emao14_30k_tauramp.json"),
    "o14fut30k": ("2026-08-30-t1-first-v7-read", "t1_o14fut30k.json"),
    "postrain30k": ("2026-08-27-t1-parity-first", "t1_postrain30k.json"),
    "postrain30k_freeze": ("2026-08-27-t1-parity-first",
                           "t1_postrain30k_freeze.json"),
    "rdw8p30k_nonparity40": (None, None),
}

RES = {}


def dig(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


rows = {}
for arm, (pkg, fn) in T1.items():
    if pkg is None:
        continue
    p = os.path.join(INC, pkg, "raw", fn)
    if not os.path.isfile(p):
        rows[arm] = {"status": "NOT_ON_THIS_BOX"}
        continue
    raw = open(p, "rb").read()
    d = json.loads(raw.decode("utf-8"))
    r = {"md5": hashlib.md5(raw).hexdigest(),
         "repo_path": f"TanitAD Research Lab/Architecture & Inference/"
                      f"Implementation/incoming/{pkg}/raw/{fn}",
         "decoder": d.get("decoder"),
         "decoder_kind": dig(d, "rollout_provenance", "decoder", "kind"),
         "decoder_source": dig(d, "rollout_provenance", "decoder", "source"),
         "ckpt": d.get("ckpt"),
         "n_windows": d.get("n_windows"),
         "n_episodes": d.get("n_episodes"),
         "dt_s": d.get("dt_s"), "horizon_steps": d.get("horizon_steps"),
         "paired_decision_grade_empty": d.get("paired_decision_grade") == {},
         "paired_legacy_empty": d.get("paired_legacy") == {},
         }
    for armk in ("cl", "ha"):
        a = dig(d, "arms", armk, default={})
        mt = dig(a, "intervals", "metrics", default={})
        r[armk] = {k: dig(mt, k, "point") if isinstance(dig(mt, k), dict)
                   else dig(mt, k)
                   for k in ("ade_dense_m", "fde_last_m", "LON_speed_mae_mps",
                             "LAT_cross_mae_m", "LAT_heading_mae_deg")}
        lon = dig(a, "four_families", "longitudinal", default={})
        r[armk]["ego_progress"] = {
            k: dig(lon, "ego_progress", k)
            for k in ("under_progress", "speed_bias_mps", "pred_speed_mps",
                      "gt_speed_mps")}
        hv = dig(lon, "anti_echo", "holdv0_baseline", default={})
        r[armk]["holdv0_verdict"] = hv.get("verdict") or hv.get("summary")
        r[armk]["copy_detector"] = dig(lon, "anti_echo", "copy_detector",
                                       "verdict")
    rows[arm] = r

RES["t1_artifacts"] = rows

# ---- which of these numbers appear in the register? ---------------------- #
reg = open(REG, encoding="utf-8", errors="replace").read()


def both_forms(x):
    """bare and comma-formatted spellings of a number (the 2026-09-04 lesson)"""
    s = f"{x}"
    out = {s}
    try:
        f = float(x)
        for nd in (3, 4):
            out.add(f"{f:.{nd}f}")
            out.add(f"{f:,.{nd}f}")
        out.add(f"{f:,.4f}".rstrip("0").rstrip("."))
    except Exception:
        pass
    return sorted(out)


NEEDLES = {
    "ade 14.069 (emao14 cl)": "14.069",
    "ade 13.879 (emao14 ha)": "13.879",
    "fde 26.297": "26.297",
    "heading 94.63": "94.63",
    "LON_speed 10.703": "10.703",
    "speed_bias -10.381": "10.381",
    "under_progress 0.9925": "0.9925",
    "holdv0 delta 10.2192": "10.2192",
    "tauramp ade 13.8645": "13.8645",
    "cross-track 1.0722": "1.0722",
    "cross-track 1.1704": "1.1704",
    "o14fut ade 14.293": "14.293",
}
hits = {}
for label, needle in NEEDLES.items():
    n = len(re.findall(re.escape(needle), reg))
    hits[label] = {"needle": needle, "count_in_register": n}
# same-breath NON-ZERO control: a token that must be present
hits["CONTROL_H-ESTIM-SEED-1"] = {
    "needle": "H-ESTIM-SEED-1",
    "count_in_register": len(re.findall("H-ESTIM-SEED-1", reg))}
hits["CONTROL_absent_nonsense"] = {
    "needle": "ZZ_THIS_MUST_BE_ZERO_ZZ",
    "count_in_register": len(re.findall("ZZ_THIS_MUST_BE_ZERO_ZZ", reg))}
RES["register_number_hits"] = hits

# which register rows mention a v7-tiny T1 arm AND a metre-valued number
ROWKEY = re.compile(r"^\|\s*[^|]*?\b([A-Z][A-Z0-9-]{3,}[A-Z0-9])\b", re.M)
rows_with = {}
for label, needle in NEEDLES.items():
    ids = set()
    for m in re.finditer(re.escape(needle), reg):
        # walk back to the start of the table row
        start = reg.rfind("\n|", 0, m.start())
        seg = reg[start:start + 400]
        km = ROWKEY.search(seg)
        if km:
            ids.add(km.group(1))
    rows_with[label] = sorted(ids)
RES["register_rows_quoting_each_number"] = rows_with

with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(RES, fh, indent=1)

print(json.dumps({"t1_artifacts": {k: {kk: vv for kk, vv in v.items()
                                       if kk in ("decoder", "decoder_kind",
                                                 "md5", "ckpt", "n_windows",
                                                 "paired_decision_grade_empty")}
                                   for k, v in rows.items()}}, indent=1))
print()
for k, v in hits.items():
    print(f"{k:38s} needle={v['needle']:24s} count={v['count_in_register']}")
print()
for k, v in rows_with.items():
    print(f"{k:38s} rows={v}")
print("[wrote]", OUT)
