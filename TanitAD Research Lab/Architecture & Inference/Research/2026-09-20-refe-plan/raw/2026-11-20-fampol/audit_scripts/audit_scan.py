import json, subprocess, sys, re
REPO = "C:/Users/Admin/tanitad-push"
TIP = "c98e6114"
NAMES = ("min_ttc_s", "headway_min_m", "time_gap_min_s")
files = [l.strip() for l in open("C:/Users/Admin/swp/fampol/both.txt", encoding="utf-8") if l.strip()]
extra = sys.argv[1:]


def load_tip(p):
    return json.loads(subprocess.check_output(["git", "-C", REPO, "show", f"{TIP}:{p}"]).decode("utf-8", "replace"))


def hit(x):
    s = json.dumps(x)
    return any(n in s for n in NAMES)


def walk(o, path, out):
    if isinstance(o, dict):
        for k, v in o.items():
            kp = f"{path}/{k}"
            if re.search(r"adverse|guard", str(k), re.I) and isinstance(v, (list, dict)):
                items = v if isinstance(v, list) else [v]
                for it in items:
                    if hit(it):
                        out.append((kp, it))
            walk(v, kp, out)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            walk(v, f"{path}[{i}]", out)


for p in files + extra:
    try:
        j = load_tip(p) if p in files else json.load(open(p, encoding="utf-8"))
    except Exception as e:
        print("ERR", p, e)
        continue
    out = []
    walk(j, "", out)
    print("==", p.split("2026-09-20-refe-plan/")[-1], "hits:", len(out))
    seen = set()
    for kp, it in out[:12]:
        sig = json.dumps(it)[:300]
        if sig in seen:
            continue
        seen.add(sig)
        print("  ", kp[:90], "::", sig)
