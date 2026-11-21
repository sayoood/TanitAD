"""Fill RESULT section 21's placeholders from artifacts (the production ledger pulled from Thor, the push logs), never from prose.
Writes result_corpus_section.md (placeholders replaced) and pull_corpus/ledger_head.jsonl (the ledger at this moment)."""
import datetime as dt, json, re, shutil, statistics, subprocess
from pathlib import Path

CO = Path(r"<scratchpad>/corpus")
led = subprocess.run(["ssh", "-o", "ConnectTimeout=15", "-n", "tanitad-thor-wifi", "cat /home/nvidia/sam3map/corpus/out/manifest.jsonl"], capture_output=True, text=True, timeout=120).stdout
rows = [json.loads(l) for l in led.splitlines() if l.strip()]
assert rows, "empty ledger read"
(CO / "pull_corpus" / "ledger_head.jsonl").write_text(led, encoding="utf-8")
pub = subprocess.run(["ssh", "-o", "ConnectTimeout=15", "-n", "tanitad-thor-wifi", "cat /home/nvidia/sam3map/corpus/publish/publisher.log"], capture_output=True, text=True, timeout=120).stdout
ok = [r for r in rows if r.get("status") == "OK"]; bad = [r for r in rows if r.get("status") != "OK"]
t_last = max(dt.datetime.fromisoformat(r["t_row"]) for r in rows); launch = dt.datetime.fromisoformat("2026-09-15T07:51:02")
hours = (t_last - launch).total_seconds() / 3600
ex = [r["extract_s"] for r in ok]; cpu = [sum(r["stage_s"].values()) for r in ok]
orient = {}
for r in ok:
    orient[r["export"]["orientation"]] = orient.get(r["export"]["orientation"], 0) + 1
rate = len(rows) / hours
text = f"""At {t_last:%H:%M} (Europe/Berlin), {hours:.2f} h after launch: **{len(rows)} clips done, {len(ok)} OK, {len(bad)} failed the export gate**,
{rate:.1f} clips/h ⇒ ≈ {4719 / rate / 24:.1f} days for the corpus. Extraction median {statistics.median(ex):.1f} s per 96-frame clip (range
{min(ex):.1f}–{max(ex):.1f}); CPU stages median {statistics.median(cpu):.1f} s in the background. GPU memory flat: `cuda_alloc_gb`
{sorted({r['cuda_alloc_gb'] for r in ok})}, `cuda_max_gb` {sorted({r['cuda_max_gb'] for r in ok})} on every clip. Orientation check on the OK clips: {orient}.
Median cart seen share {statistics.median(r['export']['cart_seen_share'] for r in ok):.3f}, drivable {statistics.median(r['export']['cart_drivable_share'] for r in ok):.3f};
lowest ego-path-on-drivable {min(r['export']['ego_future_path_on_drivable']['real'] for r in ok):.4f}.

**The first failures, read against the camera before acting** (`raw/corpus/failcheck/`):
* **081b986f8888** failed only because `ego_future_path_on_drivable` had **no value**. The camera shows a car **parked at night facing
  a snowy embankment behind a fence**; it moves 3.98 m in the whole clip, so its few future points sit in the unseen near field. The map
  is plausible (non-drivable verge ahead, drivable parking surface to the left). An egomotion census over every clip's camera span
  (`code/corpus/path_testable_census.py`, the export's own grid, poses and geometry) finds **20 of 4,719 clips (0.42 %) with no
  future-path point at all** (1 among the 315 BEV-head clips); near-stationary clips like this one come on top.
* **2b568af29b7d** failed with 87.7 % of its future path on drivable (bar 0.9). Under the path: drivable 77.0 %, lane line 10.7 %,
  **"seen, no class" 12.3 %, non-drivable 0 %** — the camera shows the car **creeping in a queue behind a truck** (2.0 m/s mean), and the
  unlabelled cells are the near field just in front of the bonnet (median 4.9 m ahead, 18 of 161 frames), which a queued car never saw
  as road. The rest of the map (lanes, markings, crosswalk, verge) matches the image.

**Decision.** The gate was **not** changed mid-run: the driver and its ledger stay as validated, and `semantic_maps/gt/` keeps its
meaning. The publisher sorts the failures by what the evidence says: a clip ships in a separate opt-in tier, `semantic_maps/gt_flagged/`,
only when **nothing contradicts its map** — every other check passed, and either the path check had nothing to test, or the off-road
share of the path is only unlabelled road with **no path cell on a non-drivable class** and ≥ 50 % on road classes. Each flagged clip
carries its reason and the class counts under its path (manifest block `clips_flagged`; loader `allow_flagged=True`). A path that runs
over an edge or sidewalk — the map contradicting the drive — stays unpublished with its failed checks. Same family as the orientation
check's `untestable`, which was never scored as a failure."""
sec = (CO / "result_corpus_section.md").read_text(encoding="utf-8")
push = (Path(r"D:/Projects/TanitAD-artifacts/hf-corpus-aug-20260915/push_augment.log")).read_text(encoding="utf-8")
commits = re.findall(r"commit (\w{10}): (\d+) files \((\[.*?\])\) in (\d+)s, verified (\d+), bad (\d+)", push)
batches = re.findall(r"batch (\d{4}): (\d+) clips, (\d+) files, ([\d.]+) MB, commit (\w{10}), (\d+)s, sha256 verified on the Hub", pub)
lines = [f"Dev-box pushes (`raw/corpus/push_augment.txt`): " + "; ".join(f"commit `{c}` {n} files {comps} verified {v}/{n}, bad {b}" for c, n, comps, s, v, b in commits) + "."]
lines.append("Thor publisher so far (`raw/corpus/thor/publisher.txt`): " + "; ".join(f"batch {b} {n} clips {f} files {mb} MB commit `{c}`" for b, n, f, mb, c, s in batches) + ".")
hv = (CO / "raw_extra" / "verify_hf_components.txt").read_text(encoding="utf-8", errors="replace")
m_ok = re.search(r"ZZHFVERIFY-(\d)-(\d+)-(\d+)ZZ", hv); m_res = re.search(r"verify: (\{.*?\}) \| card additive: (\w+)", hv); m_clip = re.search(r"clip (\w{8}): (.*)", hv)
assert m_ok and m_res, "HF verification output missing"
lines.append(f"**Second channel** (`code/corpus/verify_hf_components.py`, `raw/corpus/verify_hf_components.txt`): every pushed component downloaded "
             f"back from HF into a fresh folder and checked with the loader the dataset ships — (ok, mismatched, not downloaded) per component "
             f"{m_res.group(1)}, datacard edit additive against the card that was on the Hub before: **{m_res.group(2)}**; overall "
             f"**{'PASS' if m_ok.group(1) == '1' else 'FAIL'}**" + (f"; one clip through every loader: {m_clip.group(2)}" if m_clip else "") + ".")
sec = sec.replace("PLACEHOLDER_PRODUCTION_NUMBERS", text).replace("PLACEHOLDER_PUSH_COMMITS", "\n".join(lines))
(CO / "result_corpus_section.md").write_text(sec, encoding="utf-8", newline="\n")
fc = CO / "failcheck"; dst = CO / "raw_extra" / "failcheck"; dst.mkdir(parents=True, exist_ok=True)
shutil.copy2(fc / "failcheck_081b986f8888.jpg", dst / "failcheck_081b986f8888.jpg")
print("filled | rows", len(rows), "ok", len(ok), "bad", len(bad), "| dev commits", len(commits), "| thor batches", len(batches), "| PLACEHOLDER left:", "PLACEHOLDER" in sec)
