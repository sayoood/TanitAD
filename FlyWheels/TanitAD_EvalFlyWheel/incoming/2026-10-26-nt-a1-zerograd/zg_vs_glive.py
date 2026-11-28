import json, re, sys
zg = json.load(open(sys.argv[1])); cfg = json.load(open(sys.argv[2]))
argv = cfg.get("argv") or []
src = open(sys.argv[3], encoding="utf-8").read()
def has(f): return any(a == f or a.startswith(f + "=") for a in argv)
# admitted groups (launch_gate tip): refcv8 rc group + the NT K4 map-only groups, each with its flag
m = re.search(r"_NT_MAP_ONLY_GROUPS = \((.*?)\)\n", src, re.S)
groups = [(g, "--map-min-coverage") for g in re.findall(r'"([^"]+)"', m.group(1))]
groups.append(("core.decoder.r8_rc_to_cond", "--r8-no-rc"))
print("argv has --r8-no-rc:", has("--r8-no-rc"), " --map-min-coverage:", has("--map-min-coverage"),
      "value:", argv[argv.index("--map-min-coverage") + 1] if "--map-min-coverage" in argv else None)
rows = []
for o in zg["offenders"]:
    n = o["names"][0]
    hit = [g for g, f in groups if has(f) and (n == g or n.startswith(g + "."))]
    rows.append((n, o["case"], o["numel_lo"], hit[0] if hit else None))
for r in rows: print(r)
cov = sum(1 for r in rows if r[3]); print("covered by a pre-launch admitted group:", cov, "of", len(rows),
      "| numel covered", sum(r[2] for r in rows if r[3]), "of", sum(r[2] for r in rows))
print("NOT covered:", [r[0] for r in rows if not r[3]])
json.dump({"offenders": rows, "n": len(rows), "covered": cov}, open(sys.argv[4], "w"), indent=1)
