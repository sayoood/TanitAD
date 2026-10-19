"""Wrap the verifier's (empty) manifest with the status that explains why it is empty. python -I."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(HERE, "..", "raw", "s4_manifest.json")
m = json.load(open(p, encoding="utf-8"))
idx = json.load(open(os.path.join(HERE, "..", "raw", "s2_index.json"), encoding="utf-8"))
m = {"status": "NOT_APPLICABLE_NO_FRAMES_EXIST_IN_THE_APPROVED_ARCHIVES",
     "why": "S2 found 0 of the 1,944 listed jpgs (0 of 122 listed logs are in any of the 12 nuPlan val camera archives). Nothing was fetched, so nothing is verified. The verifier exits 3 (NOTHING_TO_VERIFY), never 0, on an empty index.",
     "listed_jpgs": idx["coverage_total"]["listed"], "found_in_archives": idx["coverage_total"]["found"],
     "fetched": 0, "verified_ok": 0, "failed": 0, "missing": 0, "MB_fetched_MEASURED": 0.0,
     "frames_dir": "C:/Users/Admin/nuplan_u/frames (empty, 0 files)", **m}
json.dump(m, open(p, "w", encoding="utf-8"), indent=1)
print(m["status"], m["fetched"], m["verified_ok"], m["failed"])
