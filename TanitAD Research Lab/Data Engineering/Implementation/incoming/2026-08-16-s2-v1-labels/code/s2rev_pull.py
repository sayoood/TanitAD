"""S2 visual review — deterministic clip selection + asset pull (mp4 / v2ep).

Selection is stratified by decision value (VISUAL_REVIEW brief):
  S1 all 19 LANE_TARGET; S2 6 TURN spot-checks (u-turn, L/R, with+without
  dist arg); S3 5 STOP_AT; S4 ROUTE_TO disposition (2 TL + 2 TR + 1 STOP
  remap + the abstain); S5 the 4 excluded val records; S6 4 FOLLOW controls.
Assets: aug120 -> bridged_w120train_2400/videos/<cid>.mp4 (pose-aligned
10 Hz, v2_to_pilot); val -> physicalai-val .v2ep.pt (raw JPEG buffer).
"""
import json
import os
import re
import sys

import truststore

truststore.inject_into_ssl()
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

KEYS = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\Keys.txt"
TOKEN = re.search(r"hf_[A-Za-z0-9]+",
                  open(KEYS, encoding="utf-8", errors="replace").read()).group(0)

REPO = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"
PKG = os.path.join(REPO, "TanitAD Research Hub", "Data Engineering",
                   "Implementation", "incoming", "2026-08-16-s2-v1-labels")
SP = (r"C:\Users\Admin\AppData\Local\Temp\claude"
      r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
      r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad")
AST = os.path.join(SP, "s2rev_assets")
os.makedirs(os.path.join(AST, "mp4"), exist_ok=True)
os.makedirs(os.path.join(AST, "v2ep"), exist_ok=True)


def load_jsonl(p):
    out = {}
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        out[r["clip_id"]] = r
    return out


labels = load_jsonl(os.path.join(PKG, "labels", "s2_labels_aug120.jsonl"))
excluded = json.load(open(os.path.join(PKG, "labels",
                                       "s2_excluded_w120val.json"),
                          encoding="utf-8"))


def gtok(c):
    return labels[c]["g_str"]["token"]


def remap(c):
    return bool(labels[c]["g_str"]["corroboration"]
                .get("remapped_from_route_to"))


def dist_arg(c):
    g = labels[c]["g_str"]
    return g["args"][0] if g["arg_mask"][0] else None


lt = sorted(c for c in labels if gtok(c) == "LANE_TARGET")
tl = sorted(c for c in labels if gtok(c) == "TURN_LEFT" and not remap(c))
tr = sorted(c for c in labels if gtok(c) == "TURN_RIGHT" and not remap(c))
stop = sorted(c for c in labels if gtok(c) == "STOP_AT" and not remap(c))
fol = sorted(c for c in labels if gtok(c) == "FOLLOW_MAIN_ROAD")
abst = sorted(c for c in labels if gtok(c) == "NONE_ABSTAIN")
rm_tl = sorted(c for c in labels if gtok(c) == "TURN_LEFT" and remap(c))
rm_tr = sorted(c for c in labels if gtok(c) == "TURN_RIGHT" and remap(c))
rm_st = sorted(c for c in labels if gtok(c) == "STOP_AT" and remap(c))

uturn = [c for c in tl if "u_turn" in str(labels[c]["g_str"]["sources"])]
tl_d = [c for c in tl if dist_arg(c) is not None and c not in uturn]
tl_n = [c for c in tl if dist_arg(c) is None and c not in uturn]
tr_d = [c for c in tr if dist_arg(c) is not None]
tr_n = [c for c in tr if dist_arg(c) is None]

# S2: u-turn + biggest-dist TL + a no-dist TL + biggest-dist TR + a no-dist TR
# + median-dist TL  (deterministic: sorted by (dist, cid))
tl_d.sort(key=lambda c: (dist_arg(c), c))
tr_d.sort(key=lambda c: (dist_arg(c), c))
s2 = (uturn[:1] + [tl_d[-1], tl_d[len(tl_d) // 2]] + tl_n[:1]
      + [tr_d[-1]] + tr_n[:1])

# S3: STOP_AT spread over stop distance: min, 25%, median, 75%, max
stop_d = sorted((c for c in stop), key=lambda c: (dist_arg(c) or 0.0, c))
qs = [0, len(stop_d) // 4, len(stop_d) // 2, 3 * len(stop_d) // 4,
      len(stop_d) - 1]
s3 = [stop_d[i] for i in sorted(set(qs))]

# S4: ROUTE_TO disposition — first 2 TL remaps, first 2 TR remaps, the STOP
# remap, the abstain
s4 = rm_tl[:2] + rm_tr[:2] + rm_st[:1] + abst[:1]

# S6: FOLLOW controls — 1 route-backed slowest, 1 route-backed fastest,
# the merge, 1 pi_default
ea = load_jsonl(os.path.join(PKG, "labels", "engine_a_aug120.jsonl"))


def v0(c):
    return (ea[c]["engine_a"].get("speed_profile") or {}).get("v_t0_ms") or 0.0


fol_route = sorted((c for c in fol if "route_v3=follow"
                    in str(labels[c]["g_str"]["sources"])),
                   key=lambda c: (v0(c), c))
fol_merge = [c for c in fol if "merge" in str(labels[c]["g_str"]["sources"])]
fol_def = sorted(c for c in fol if "pi_default"
                 in str(labels[c]["g_str"]["sources"]))
s6 = [fol_route[0], fol_route[-1]] + fol_merge[:1] + fol_def[:1]

sel = {
    "S1_LANE_TARGET": lt,
    "S2_TURN": s2,
    "S3_STOP_AT": s3,
    "S4_ROUTE_TO_DISPOSITION": s4,
    "S5_EXCLUDED_VAL": [e["clip_id"] for e in excluded],
    "S6_FOLLOW_CONTROLS": s6,
}
flat_aug = [c for k, v in sel.items() if k != "S5_EXCLUDED_VAL" for c in v]
assert len(flat_aug) == len(set(flat_aug)), "aug selection overlap"
json.dump(sel, open(os.path.join(AST, "selection.json"), "w"), indent=1)
print({k: len(v) for k, v in sel.items()},
      "total", len(flat_aug) + len(sel["S5_EXCLUDED_VAL"]), flush=True)

# ---- pull ------------------------------------------------------------------ #
from huggingface_hub import hf_hub_download  # noqa: E402
import concurrent.futures as cf              # noqa: E402
import shutil                                # noqa: E402

jobs = []
for c in flat_aug:
    jobs.append(("Sayood/tanitad-ph0-aug120",
                 f"bridged_w120train_2400/videos/{c}.mp4",
                 os.path.join(AST, "mp4", f"{c}.mp4")))
for c in sel["S5_EXCLUDED_VAL"]:
    jobs.append(("Sayood/tanitad-physicalai-w120-256x640cyl",
                 f"physicalai-val-0c5f7dac3b11-w120-256x640cyl/{c}.v2ep.pt",
                 os.path.join(AST, "v2ep", f"{c}.v2ep.pt")))


def pull(job):
    repo, rf, dst = job
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return "cached"
    p = hf_hub_download(repo, rf, repo_type="dataset", token=TOKEN)
    shutil.copyfile(p, dst)
    return "pulled"


n = 0
with cf.ThreadPoolExecutor(max_workers=6) as ex:
    for r in ex.map(pull, jobs):
        n += 1
        print(f"{n}/{len(jobs)} {r}", flush=True)
print(f"PULL_DONE {n}/{len(jobs)}", flush=True)
sys.exit(0)
