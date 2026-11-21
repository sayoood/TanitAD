"""List the pinned NuRec release (recursive tree API, metadata only; no scene content is read)."""
import sys, json, re, collections
sys.path.insert(0, "E:/Projects/TanitAD-artifacts/dagland/motion_c/t2_scenes/code")
import hfc, requests
REV = "5fe7a149b40ee83a2d3e09f7fe5dc2f217cabe11"
REL = "sample_set/26.04_release"
url = f"https://huggingface.co/api/datasets/{hfc.REPO}/tree/{REV}/{REL.replace('/', '%2F')}?recursive=true&expand=false&limit=1000"
rows, pages = [], 0
while url:
    r = requests.get(url, headers=hfc.auth(), timeout=120)
    r.raise_for_status()
    rows += r.json(); pages += 1
    m = re.search(r'<([^>]+)>;\s*rel="next"', r.headers.get("Link", ""))
    url = m.group(1) if m else None
files = [x for x in rows if x["type"] == "file"]
by = collections.defaultdict(dict)
for x in files:
    rel = x["path"][len(REL) + 1:]
    sc, _, name = rel.partition("/")
    by[sc][name] = {"size": x["size"], "oid": x.get("oid"), "lfs_sha256": (x.get("lfs") or {}).get("oid")}
out = {"repo": hfc.REPO, "revision": REV, "release_dir": REL, "pages": pages, "n_entries": len(rows), "n_files": len(files), "n_scenes": len(by), "scenes": by}
json.dump(out, open("E:/Projects/TanitAD-artifacts/dagland/motion_c/t2_scenes/raw/RELEASE_LISTING.json", "w"), indent=0, sort_keys=True)
names = collections.Counter(n for s in by.values() for n in s)
print("pages", pages, "entries", len(rows), "files", len(files), "scenes", len(by))
print(names.most_common(8))
tot = sum(v["size"] for s in by.values() for v in s.values()); print("total bytes", tot)
