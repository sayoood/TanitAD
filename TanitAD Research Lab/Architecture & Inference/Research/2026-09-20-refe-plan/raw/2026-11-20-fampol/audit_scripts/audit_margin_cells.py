import json, subprocess, glob
REPO = "C:/Users/Admin/tanitad-push"
TIP = "c98e6114"
NAMES = ("min_ttc_s", "headway_min_m", "time_gap_min_s")
ls = subprocess.check_output(["git", "-C", REPO, "ls-tree", "-r", "--name-only", TIP]).decode("utf-8", "replace").splitlines()
paths = [p for p in ls if p.endswith(".json") and ("fam_ci" in p.lower() or "famci_" in p.lower())]
srcs = [("tip", p) for p in paths] + [("E:", p) for p in sorted(glob.glob("E:/Projects/TanitAD-artifacts/p5c_read/out/families/famci_*.json"))]


def load(kind, p):
    if kind == "tip":
        return json.loads(subprocess.check_output(["git", "-C", REPO, "show", f"{TIP}:{p}"]).decode("utf-8", "replace"))
    return json.load(open(p, encoding="utf-8"))


def cells(j):
    # fam_ci schema: frames -> frame -> families -> longitudinal -> comp -> entry ; older: families -> ...
    frs = j.get("frames")
    if frs:
        for fr, v in frs.items():
            for k, e in ((v.get("families") or {}).get("longitudinal") or {}).items():
                yield fr, k, e
    else:
        for k, e in ((j.get("families") or {}).get("longitudinal") or {}).items():
            yield "-", k, e


for kind, p in srcs:
    try:
        j = load(kind, p)
    except Exception as ex:
        print("ERR", p, ex)
        continue
    rows = []
    for fr, k, e in cells(j):
        if k.split(".")[-1] in NAMES and isinstance(e, dict) and e.get("status") == "OK" and isinstance(e.get("diff_B_minus_A"), dict):
            d = e["diff_B_minus_A"]
            inh, cor = d["lo"] > 0, d["hi"] < 0
            if inh or cor:
                rows.append((fr, k, round(d["point"], 4), round(d["lo"], 4), round(d["hi"], 4), "inherited-ADVERSE" if inh else "-", "corrected-ADVERSE" if cor else "-"))
    short = p.split("2026-09-20-refe-plan/")[-1] if kind == "tip" else p
    print("==", short, f"[{j.get('label_a')} -> {j.get('label_b')}]", "separated margin cells:", len(rows))
    for r in rows:
        print("   ", *r)
