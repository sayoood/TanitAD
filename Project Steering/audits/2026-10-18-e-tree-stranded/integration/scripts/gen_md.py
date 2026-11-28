import json, collections, sys, re
W = "C:/Users/Admin/integ/work/"
OUT = "C:/Users/Admin/integ/out/Project Steering/audits/2026-10-18-e-tree-stranded/integration/"
T = json.load(open(OUT + "triage.json"))
S = T["summary"]
FT = S["tip_C_checked_against"][:8]
AB = S["tip_AB_analysed_against"][:8]


def short(v):
    m = re.search(r"Research/(.+?)/code/(fix[^/]*)/", v)
    if m:
        return f"{m.group(1)[:34]} / {m.group(2)}"
    m = re.search(r"(2026-\d\d-\d\d-[a-z0-9-]+)/(.*?)/code/fix/", v)
    if m:
        return f"{m.group(1)[:26]} / {m.group(2)}"
    m = re.search(r"incoming/(2026-[^/]+)/code/(fix)", v)
    if m:
        return f"{m.group(1)} / {m.group(2)}"
    return v[-60:]


L = []
w = L.append
w("# Integration triage: which stranded-work INTEGRATION items are real defects (2026-10-18)\n")
w(f"Tip analysed for A and B: `{TIP_AB if False else AB}` (`363a24ffd753985efd91c857d9453d8d88990cbe`); section C re-checked at the END of the run against `{FT}` "
  f"(`{S['tip_C_checked_against']}`). No file under `stack/`, `taniteval/`, `tools/`, `colab/` or `data/` changed between the two tips (MEASURED: `git diff --name-only`), so the A and B verdicts stand at the later tip. "
  "Mirror `C:/Users/Admin/tanitad-push`, branch `agent/arch-inf-20260803`. E: files were read only. Evidence class of every count here: **MEASURED** (scripts in `scripts/`, artifacts in `raw/`).\n")
w("## Headline\n")
a = S["A"]
w(f"* **MISSING = {a.get('MISSING',0)} of the audit's 29 package-local code variants (A), and all {a.get('MISSING',0)} are ONE defect: the v7F R1-R6 change set never reached `stack/`.** "
  f"The other 24: APPLIED {a.get('APPLIED',0)}, SUPERSEDED {a.get('SUPERSEDED',0)}, OBSOLETE (variant held by decision) {a.get('OBSOLETE',0)}.")
w("* **None of the MISSING items touches a LIVE chain.** P6 runs `refe/` (self-contained: no `from tanitad`/`from taniteval` import anywhere under `2026-09-20-refe-plan/refe`, MEASURED by `git grep` at the tip; it only references `launch_gate`, and the one launch_gate variant in the 29 is SUPERSEDED). "
  "The refcv7/refcv8 stack trainers (`refc_v3_train.py`, `refc.py`, `refc_v3.py`, `refcv8_train.py`, `refcv8_conditioning.py`, `launch_gate.py`): every variant targeting them is APPLIED, SUPERSEDED or HELD, none MISSING. "
  "The MISSING files are `train_v6_staged.py`, `models/v6.py`, `declared_vs_built_v6.py`: no `stack/ops/runs.d` argv exists for v6/v7F, `launch_gate.py` has **0** occurrences of `v7f` (no gate profile, so under the PI's launch-gate rule a v7F run cannot start), `refc_v3_train.py` mentions `train_v6_staged` only in a comment, and the registers' last v7F row is at line 15812 of 16817 (a 2026-09-20/21 row). **Dormant, not live.**")
w("* **Beyond the 29 (extension, MEASURED from every `package path -> repo path` line in the 74 LANDING_READY manifests): the manifests declare 39 mappings the tip does not hold** (21 target files differ, 18 are NEW files absent), in five families. "
  "13 are the held refcv7 R6 / pre-HQS rows and 10 are the v7F merge (both already counted in A). **The other 16 are new findings:** refav1 R1/R3/R5/R6 (11), refav1 R1-VMAX (3), the v7F launch-gate profile (2). "
  "Two of those touch LIVE files: `refa_v1_plan.py` and `taniteval/tools/refav1_arm.py` (both edited on the tip 2026-10-18, HJ-UNITS-2) lack the R1-VMAX cap (flag-off, bit-identical when off per the package; no live run passes it). A clean 3-way patch is supplied for these two.")
w("* **Candidate edits (B, 47): MISSING 23 / SUPERSEDED 18 / REGISTER (report only) 5 / machine-local 1.** The one that matters for running code: **the leaderboard builder fix is on E: and not on the tip** "
  "(`taniteval/taniteval/leaderboard/{build,render,sources,__main__}.py`, `leaderboard_sources.json`, `tests/test_leaderboard.py`): the tip's builder has no TOMBSTONE handling, no hand-edit guard and no section fingerprint, while the tip carries 10+ `TOMBSTONE.json` withdrawn runs and `Benchmarks & Eval/LEADERBOARD.md` has 0 `SECTION-SHA256` fingerprints. "
  "A rebuild from the tip's code would re-publish withdrawn runs and could silently undo a retraction (E: RESULT section 25). ⚠️ **Coupling:** the chunked landing is bringing `test_leaderboard_hand_edit_guard.py` / `test_leaderboard_yawmask_input.py` / `test_leaderboard_navtest_single_stage_ours.py` (they were pending at chunk 2); they pin the E: behaviour and will be RED against the tip's builder unless the leaderboard patch lands with them. I did not run them (UNVERIFIED).")
w("* **Section C: the audit's 1,073 'absent from the tip' manifest entries are mostly NOT absent.** After cleaning the audit parser's artefacts (md5/comment text glued onto the path): "
  f"{S['C_after_cleaning'].get('PARSE-ARTIFACT:FILE-ON-TIP',0)} + {S['C_after_cleaning'].get('PARSE-ARTIFACT:DIR-ON-TIP',0)} were never missing; {S['C_after_cleaning'].get('LANDED-SINCE-AUDIT',0)} were landed by chunks 1-3 of the running landing; "
  f"{S['C_after_cleaning'].get('MOVED',0)} are content-moved; **{S['C_after_cleaning'].get('PENDING-LANDING',0)} still pending (chunk 4)**; {S['C_after_cleaning'].get('OPTIONAL-P3',0)} optional 5-20 MB; {S['C_after_cleaning'].get('HOLD-SCRUB',0)} held for scrub; "
  f"{S['C_after_cleaning'].get('ABSENT-EVERYWHERE',0)} absent everywhere (2 reel .mp4 = BULK media, 2 prose lines).\n")

w("## A. The 29 S4 package-local code variants\n")
w("Method: for each variant, (1) blob id vs the tip target and vs EVERY historical blob of the target (`git log --raw -m`; none of the 29 is byte-equal to the tip or to any historical version); "
  "(2) the closest historical blob by line-set distance is taken as the variant's base, `git diff --histogram -U0` base -> variant is cut into hunks, and each hunk's added block is searched in the tip text (APPLIED), its removed block likewise "
  "(MISSING = tip still has the old block and lacks the new; DEL-MISSING = the variant DELETES lines the tip still has, i.e. the variant is the older stage); (3) the package's own docs and the E: mtime vs the superseding commit decide SUPERSEDED. "
  "Hunk tables: `raw/A_hunk_analysis.json`. 'hunks' below = counts by status. READ THE STATUS AGAINST THE DIRECTION: in the `fix_P7` and box-head `fix` rows a MISSING / DEL-MISSING hunk means the TIP holds the "
  "later (L1/L2/HQS) form of a line that the OLDER variant lacks or holds in its earlier form; only the v7F rows are variants that are NEWER than the tip file.\n")
w("| # | variant (package / dir) | target | verdict | hunks | evidence |")
w("|---:|---|---|---|---|---|")
for r in T["A"]:
    h = ", ".join(f"{k} {v}" for k, v in r["hunks"].items()) or "(new file)"
    w(f"| {r['i']} | {short(r['variant'])} | `{r['target'].split('stack/')[-1] if 'stack/' in r['target'] else r['target']}` | **{r['verdict']}** | {h} | {r['evidence']} |")
w("")
w("### The 5 MISSING items (v7F) - what they fix, what runs them, patch\n")
for r in T["A"]:
    if r["verdict"] == "MISSING":
        w(f"* #{r['i']} `{r['target']}`: {r.get('what_it_fixes','')}")
w("\n* **Superseded BY an unlanded set:** `v7f_merge` (package `products/P4-training-pipelines/2026-09-27-v7f-refav1-state/v7f_merge/`) is the COMPLETE v7F set (R1+R3+R4+R2 nav fixes+R6 `--strategic-off`) and its RESULT says it supersedes every `code/fix/` file of `v7f_r1r4/` and `v7f_r3/`. "
  "It is not on the tip's `stack/` either: tip `train_v6_staged.py` last changed 3886f7b (2026-09-06), `v6.py` 982476c (2026-09-06). The tip blob equals the merge's declared base, so the merge applies cleanly.")
w("* **What the merge fixes (INHERITED from the package RESULT, not re-verified by me):** a real `--nav-cond` v7F run could not START at the tip (three defects: `synthetic_batch` carries no nav keys, `nav_token` overwritten by `None` at step 1, the NavConditioner's 12 parameters belong to no `_GROUP_PREFIXES` group); "
  "no tactical label family reaches any v7F loss (traffic-light colours included, `D-TLIGHT-1` for v7F); the tactical behaviours do not reach the operative plan (R4); no hard speed cap (R1). The package reports 178 tests passed on the merge and 2,655 passed / 8 failed / 24 errors on the 107-file v6 suite with the failure set identical to its baseline (INHERITED).")
w("* **Live use:** none (see headline). No `runs.d` argv, no gate profile.")
w("* **Proposed patch (NOT applied):** `patches/v7f_merge_R1R6.patch` = tip -> `v7f_merge` for `train_v6_staged.py`, `models/v6.py`, `models/plan_speed_cap.py` (new), `train/declared_vs_built_v6.py` (new) and six test files; "
  "`git apply --check` passes on the tip's exact bytes and the result is byte-identical to the package files (`patches/v7f_merge_R1R6.json`); all `.py` files compile. **Tests were not run** (needs the full tree + torch). "
  f"Tests that name the targets (MEASURED, `raw/tests_naming_the_targets.json`): {len(T['tests_naming']['stack/scripts/train_v6_staged.py'])} files name `train_v6_staged`, {len(T['tests_naming']['stack/tanitad/models/v6.py'])} import `tanitad.models.v6`, 0 name `declared_vs_built_v6`. "
  "⚠️ `models/v6.py` is imported by data modules shared with other programmes (`bev_raster.py`, `tactical_goals.py`, `g_tac_geom.py`, `anchor_goal.py`, `refav1_loader.py`...): run the full suite before landing; the package's 'bit-identical with every new flag off' claim is INHERITED.")

w("\n## A+. Extension: every declared `package path -> repo path` mapping not applied (beyond the 29)\n")
w(f"Source: all 74 `LANDING_READY*.txt` in the E: tree, {len(T['X_unapplied_families'])} families below. Mapping statuses (MEASURED, `raw/X_manifest_mappings.json`): {S['X_mapping_statuses']}. "
  "'tip == declared base' means the tip still holds the exact bytes the variant was built from, i.e. the variant was NEVER applied.\n")
w("| family | repo targets not applied | tip == declared base | live? | disposition |")
w("|---|---|---|---|---|")
disp = {
    "v7F trainer (v7f_merge)": ("dormant (see headline)", "= section A MISSING; `patches/v7f_merge_R1R6.patch`"),
    "v7F launch-gate profile (v7f_gate)": ("dormant; `launch_gate.py` itself is live (edited 3766dcd, 2026-10-17)", "MISSING; `patches/v7f_gate_profile.patch` 3-way merges CLEAN onto the tip's launch_gate.py (base = declared). Needs v7f_merge first (its checks call `declared_vs_built_v6.check_all`)."),
    "refav1 R1/R3/R5/R6 (refav1_r1r6)": ("refa_v1_train.py last changed 2026-09-03: dormant trainer; `refa_v1.py` is live (HJ-UNITS-2, 10-18)", "MISSING (feature set, flags default OFF, 'bit-identical when off' INHERITED). `patches/refav1_R1R6_plus_R1VMAX.patch` is PARTIAL: `stack/tanitad/refs/refa_v1.py` has 1 merge conflict against the tip (HJ-UNITS-2) and is OMITTED; do not apply the patch without resolving it."),
    "refav1 R1-VMAX": ("`refa_v1_plan.py` and `refav1_arm.py` are LIVE files; the cap is flag-off and no live run passes it", "MISSING; 3-way merge CLEAN onto the tip (included in the refav1 patch). PI requirement R1 of 2026-09-27."),
    "refcv7 box-head R6 (held)": ("not adopted", "OBSOLETE: held by decision (argv `why`, BUILD.md, LANDING_READY_R6)."),
}
for f, rows in T["X_unapplied_families"].items():
    nb = sum(1 for r in rows if r["tip_equals_declared_base"])
    d = disp.get(f, ("", ""))
    w(f"| {f} | {len(rows)} ({sum(1 for r in rows if r['status']=='NEW-NOT-ON-TIP')} new files) | {nb} | {d[0]} | {d[1]} |")
w("\n(Per-file list: `triage.json` -> `X_unapplied_families`.)\n")

w("## B. The 47 S2b candidate edits\n")
w("Method: E blob vs every historical blob of the path on the tip (none matches), E-only lines (E lines absent from the tip file), the closest-history-base hunk analysis (`raw/B_candidate_edits.json`), and a single-pass `git grep -F -f` of every E-only line of 28+ chars against the WHOLE tip "
  "(`raw/B_E_only_lines_found_on_tip.json`: does the content live under another path?). Registers: report only, lead-ID check (`raw/REGISTER_lead_ids.json`); E-only lines of the two registers flagged for public IP/port are NOT exported. "
  "mtime is used only as a secondary hint.\n")
w("| # | path | verdict | E-only / tip-only | evidence |")
w("|---:|---|---|---|---|")
for r in T["B"]:
    p = r["path"].replace("TanitAD Research Lab/Architecture & Inference/Research/", "AI/R/").replace("FlyWheels/TanitAD_EvalFlyWheel/incoming/", "EF/")
    w(f"| {r['i']} | `{p[-78:]}` | **{r['verdict']}** | {r['e_only_nonblank_lines']} / {r['tip_only_lines']} | {r['evidence']} |")
w("\nPatch for the leaderboard code (the only MISSING group that is running code): `patches/leaderboard_hand_edit_guard_and_withdrawn_runs.patch` (tip -> E: for 6 files; apply-check passes; byte-identical). "
  f"Tests naming it: {len(T['tests_naming']['taniteval/taniteval/leaderboard/*'])} (`taniteval/tests/test_leaderboard.py` + the fixtures). A run of `test_leaderboard.py` on a partial export failed identically before the patch on missing input files (UNVERIFIED either way; needs the full tree).\n")

w("## C. LANDING_READY entries absent from the tip (audit section 7), after the chunked landing\n")
w(f"Re-checked at `{FT}` (chunks 1-3 of 4 landed; chunk 4 pending at the time of writing). The audit's `not_on_tip_paths` (1,073 entries over 40 manifests) were re-parsed: its parser glued md5 / comment text onto many paths. Classes per entry (MEASURED, `raw/C_landing_ready_absent.json`):\n")
w("| class | n |")
w("|---|---:|")
for k, v in sorted(S["C_after_cleaning"].items(), key=lambda kv: -kv[1]):
    w(f"| {k} | {v} |")
w("\nPer manifest (only manifests with a non-artifact, non-landed remainder; the two big ones first):\n")
w("| manifest | at audit | landed since | parse artifact (never missing) | remaining (class: n) |")
w("|---|---:|---:|---:|---|")
rows = []
for p in T["C_per_manifest"]:
    c = p["classes"]
    rem = {k: v for k, v in c.items() if not k.startswith(("PARSE", "LANDED"))}
    pa = sum(v for k, v in c.items() if k.startswith("PARSE"))
    rows.append((-(p["absent_at_audit"]), p["manifest"], p["absent_at_audit"], c.get("LANDED-SINCE-AUDIT", 0), pa, rem))
for _, m, n, la, pa, rem in sorted(rows):
    if rem or n > 100:
        w(f"| `{m.split('incoming/')[-1] if 'incoming/' in m else m[-80:]}` | {n} | {la} | {pa} | {', '.join(f'{k}: {v}' for k, v in rem.items()) or '-'} |")
w("\n* NAVSIM50K_LEGAL (173 at audit) and NAVSIM_THOR (317) are the md5-format manifests and were NOT parse artifacts: they were absent for real and are being landed by the chunks (see 'landed since').")
w("* 'MOVED' = the file's CONTENT is on the tip (or in history) under another path (audit classes S4m/S4h); 'OPTIONAL-P3' = 5-20 MB raw data in `paths_P3_optional.json`; 'HOLD-SCRUB' = redacted copy listed in `paths_scrubbed.json`.")
w("* ABSENT-EVERYWHERE (4): the reel + preview `.mp4` (155.9 MB and 13.0 MB; BULK media, not on E:) and two prose lines.\n")

w("## Patches (all against the tip, NONE applied; `patches/*.json` carries the per-file tip blob ids)\n")
w("| patch | files | `git apply --check` on tip bytes | byte-identical to source | notes |")
w("|---|---:|---|---|---|")
notes = {"v7f_merge_R1R6": "complete v7F set; 4 existing targets + 6 new",
         "refav1_R1R6_plus_R1VMAX": "PARTIAL: `refa_v1.py` omitted (1 conflict). R1-VMAX files merged 3-way from the declared bases",
         "v7f_gate_profile": "adds the `v7f` launch-gate profile + test; depends on v7f_merge",
         "leaderboard_hand_edit_guard_and_withdrawn_runs": "tip -> E: copy of 6 files"}
for p in T["patches"]:
    w(f"| `patches/{p['name']}.patch` | {len(p['files'])} | {p['apply_check']} | {p['byte_identical_after_apply']} | {notes[p['name']]} |")
w("\n## Caveats, stated plainly\n")
w("* No test was run on any patch (the suite needs the full tree and torch); the claims 'bit-identical with flags off' and the package test counts are **INHERITED**.")
w("* A 'hunk' classification is textual. SUPERSEDED rests on (a) the variant lacking lines the tip has, (b) E: mtimes preceding the superseding landing commit, (c) the package's own statements. Where only (b) supported it I say so; the A rows with 'CHANGED' hunks were each read by hand.")
w("* The audit's `S2b` verdicts were taken as given for set membership; B re-derives the content verdict. mtimes on a live working tree are a hint only.")
w("* The brief's 'P6 runs refe/ at bcc04b5' is INHERITED; I verified only that `refe/` at the tip is import-independent of the 29 targets.")
w("* The register E-only lines were NOT exported (two are flagged for public IP/port); all lead IDs found are already on the tip.")
open(OUT + "TRIAGE.md", "w", encoding="utf8").write("\n".join(L) + "\n")
print(len("\n".join(L)), "chars")
