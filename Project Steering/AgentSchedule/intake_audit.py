import os, re, datetime
HUB = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\TanitAD Research Hub"
TODAY = datetime.date(2026, 8, 3)
HEAD = re.compile(r"^\s{0,3}#{1,4}\s*\**\s*(ORCHESTRATOR\s+)?VERDICT\b.*$", re.I)
NEXT_HEAD = re.compile(r"^\s{0,3}#{1,4}\s")
# the untouched template menu = NOT a decision
MENU = re.compile(r"integrate\s*/\s*integrate-with-changes\s*/\s*defer\s*/\s*reject", re.I)
DECIDED = re.compile(r"\*\*\s*(integrate(-with-changes)?|defer|reject(-with-reason)?|superseded|withdrawn|kill)\b", re.I)

res = {"DECIDED": [], "MENU_LEFT": [], "ABSENT": []}
for dirpath, _, filenames in os.walk(HUB):
    if "INTAKE.md" not in filenames:
        continue
    p = os.path.join(dirpath, "INTAKE.md")
    lines = open(p, encoding="utf-8", errors="replace").read().splitlines()
    slug = os.path.basename(dirpath)
    disc = os.path.relpath(dirpath, HUB).split(os.sep)[0]
    dm = re.match(r"(\d{4}-\d{2}-\d{2})", slug)
    age = (TODAY - datetime.date.fromisoformat(dm.group(1))).days if dm else -1
    chunk = []
    for i, ln in enumerate(lines):
        if HEAD.match(ln):
            for ln2 in lines[i+1:]:
                if NEXT_HEAD.match(ln2):
                    break
                chunk.append(ln2)
            break
    blob = "\n".join(chunk)
    state = "ABSENT"
    if blob.strip():
        if DECIDED.search(blob) and not MENU.search(blob):
            state = "DECIDED"
        elif MENU.search(blob) or "<!--" in blob:
            state = "MENU_LEFT"
        elif [c for c in chunk if c.strip() and not re.search(r"(filled by|writes here|do not pre-fill|Date / by|^\s*-\s*$)", c, re.I)]:
            state = "DECIDED"
        else:
            state = "MENU_LEFT"
    res[state].append((age, disc, slug))

for k in ("ABSENT", "MENU_LEFT", "DECIDED"):
    res[k].sort(reverse=True)
tot = sum(len(v) for v in res.values())
unadj = res["ABSENT"] + res["MENU_LEFT"]
unadj.sort(reverse=True)
print(f"TOTAL INTAKE: {tot}")
print(f"  DECIDED (a real verdict written): {len(res['DECIDED'])}")
print(f"  MENU LEFT INTACT (template never filled): {len(res['MENU_LEFT'])}")
print(f"  VERDICT SECTION ABSENT/EMPTY: {len(res['ABSENT'])}")
print(f"  ==> UN-ADJUDICATED TOTAL: {len(unadj)}  oldest {unadj[0][0]} days")
print()
print("UN-ADJUDICATED (oldest first):")
for age, disc, slug in unadj:
    print(f"  {age:>3}d  {disc:<26} {slug}")
print()
print("DECIDED:")
for age, disc, slug in res["DECIDED"]:
    print(f"  {age:>3}d  {disc:<26} {slug}")
