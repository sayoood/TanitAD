import json, subprocess
REPO = "C:/Users/Admin/tanitad-push"; TIP = "c98e6114"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/"
NAMES = ("min_ttc_s", "headway_min_m", "time_gap_min_s")


def tip(p):
    return json.loads(subprocess.check_output(["git", "-C", REPO, "show", f"{TIP}:{PK}{p}"]).decode("utf-8", "replace"))


def comp(a):
    return a.get("component") or a.get("metric")


def show(name, lst, extra=""):
    names = [comp(a) if isinstance(a, dict) else a for a in lst]
    non = [n for n in names if str(n).split(".")[-1] not in NAMES]
    m = [n for n in names if str(n).split(".")[-1] in NAMES]
    print(f"{name}: total {len(names)}, margin {m}, non-margin {len(non)} -> guard {'still FAIL' if non else 'would PASS (this list alone)'} {extra}")


def pairs(p):
    j = tip(p) if not p.startswith("E:") else json.load(open(p, encoding="utf-8"))
    for pr, v in j["pairs"].items():
        for fr, fv in v["frames"].items():
            lst = fv.get("separated_adverse") or []
            if any(comp(a).split(".")[-1] in NAMES for a in lst):
                show(f"{p.split('raw/')[-1]} {pr} {fr}", lst)


for p in ("raw/2026-11-20-p5c-read/families/families_summary.json", "raw/2026-11-20-p5c-next/srule_fam/families/families_summary.json",
          "raw/2026-11-13-p5c-read/dryrun/families_summary.json"):
    pairs(p)
for p in ("raw/2026-11-20-p5c-read/read.json",):
    j = tip(p)
    show("p5c read.json g2", j["g2"]["adverse"], f"| G2 status {j['g2']['status']} | verdict {j.get('verdict') if not isinstance(j.get('verdict'), dict) else j['verdict'].get('verdict')}")
    print("  verdict keys:", list(j.keys())[:30])
    print("  verdict:", json.dumps(j.get("verdict"))[:700])
for p in ("raw/2026-11-20-field-read/dry_runs/dry_C1_vs_C0.json", "raw/2026-11-20-field-read/dry_runs/dry_selfread_C1_vs_C1.json",
          "raw/2026-11-20-field-read/rawdac/dry_C1_vs_C0_rawdac.json", "raw/2026-11-20-field-read/rawdac/dry_selfread_C1_vs_C1_rawdac.json"):
    j = tip(p)
    g = j["families"]["guard_longitudinal_lateral"]
    show(p.split("raw/")[-1], g["adverse"], f"| status {g['status']} | verdict {j['verdict']['verdict']}")
