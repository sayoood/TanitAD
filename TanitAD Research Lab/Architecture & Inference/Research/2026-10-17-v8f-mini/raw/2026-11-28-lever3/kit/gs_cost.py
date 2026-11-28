import tarfile, re, json
t = tarfile.open("F:/TanitAD-scratch/rollhist/lever3/rec/records/mm_records_1124.tar.gz")
res = {}
want = {f"rollhist/junit_rh/shard{i}.xml" for i in range(8)} | {"rollhist/rh_gsuite.sh"}
for m in t:
    if m.name in want:
        f = t.extractfile(m); head = f.read(2000).decode("utf-8", "replace")
        if m.name.endswith(".sh"):
            res["rh_gsuite.sh"] = head[:1900] + f.read(1500).decode("utf-8", "replace")
        else:
            mt = re.search(r'<testsuite [^>]*', head)
            res[m.name] = mt.group(0)[:300] if mt else head[:200]
        if len(res) == len(want): break
json.dump(res, open("F:/TanitAD-scratch/rollhist/lever3/kit/dry/gsuite_cost_raw.json", "w"), indent=1)
print("done", len(res))
