"""Compose TRIAGE.md / triage.json from the raw analysis files. Verdicts are the reviewer's, derived from the hunk data."""
import json, os, shutil, collections, sys, re, subprocess

W = "C:/Users/Admin/integ/work/"
OUT = "C:/Users/Admin/integ/out/"
PKG = "Project Steering/audits/2026-10-18-e-tree-stranded/integration/"
D = OUT + PKG
FINAL_TIP = sys.argv[1]
TIP_AB = "363a24ffd753985efd91c857d9453d8d88990cbe"
os.makedirs(D + "patches", exist_ok=True)
os.makedirs(D + "raw", exist_ok=True)
os.makedirs(D + "scripts", exist_ok=True)

A = json.load(open(W + "A.json"))
A2 = json.load(open(W + "A2.json"))
B = json.load(open(W + "B.json"))
B3 = json.load(open(W + "B3.json"))
X = json.load(open(W + "X.json"))
C = json.load(open(W + "C.json"))
REG = json.load(open(W + "REG.json"))
TN = json.load(open(W + "tests_naming.json"))

# ---------------------------------------------------------------- A
BOX_HELD = ("OBSOLETE", "variant HELD BY DECISION: +R6 (A10.1) denoising-query groups behind --slot-dn-groups (default 0). The tip's "
            "refcv7-r101-s0.argv.json `why` says \"+R6 ... is NOT adopted\"; BUILD.md (box-head) says R6 was \"deprioritised by A14.1\"; "
            "LANDING_READY_R6.txt says \"Use ONLY if a SPEC amendment makes +R6 the launch configuration\". No live chain passes the flag.")
PRE_HQS = ("SUPERSEDED", "pre-HQS 'MAIN' box-head variant: it LACKS the A14 HQS lines the tip carries (--slot-query-select, its G-DVB entry, "
           "REGISTRY pin 221); the tip took the ANCHORED/HQS landing (commit 28d8365 'refcv7 box head lands (A9-A14.1)'). Variant mtime 2026-09-27 12:50 < 15:26.")
P7 = ("SUPERSEDED", "fix_P7 is the P7 stage (E mtime 2026-10-04 22:35-22:39) that preceded the combined WP-B landings L1 badb582 (23:51) and L2 a5c2a9b (2026-10-05); "
      "every hunk of diff(L1 -> variant) either DELETES lines the tip still has (v9-cons, enc8, emit_live, pose-sync) or is an older form of a line the tip carries in newer form.")
V7F = ("MISSING", "v7F R1/R3/R4 stream copy; SUPERSEDED BY the unlanded v7f_merge change set (itself NOT in stack/). The tip's train_v6_staged.py/v6.py are byte-unchanged "
       "since 2026-09-06 (3886f7b / 982476c): every hunk is absent from the tip.")
AV = {
    0: ("APPLIED", "7/7 hunks of diff(c39798d -> variant) are on the tip (X10 --pose-sync-sidecar wiring); landed by the WP-B landings L1/L2 (badb582, a5c2a9b)."),
    1: ("SUPERSEDED", "only change is the REGISTRY pin 256 -> 257 (+X10 comment); the tip's pin has moved on (a5c2a9b: 264+) and carries the X10 comment lines."),
    2: PRE_HQS, 4: PRE_HQS, 5: PRE_HQS, 6: PRE_HQS,
    3: ("APPLIED", "9/9 hunks of diff(cef9709 -> variant) are on the tip (A14.1 box-head perception branch), landed by 28d8365."),
    7: BOX_HELD, 9: BOX_HELD, 10: BOX_HELD, 11: BOX_HELD, 12: BOX_HELD,
    8: ("OBSOLETE", "6 hunks APPLIED (the A14.1 branch, on the tip); the other 6 hunks are the R6-only DN-embedder lines - HELD BY DECISION, see the fix_r6 rows."),
    13: P7, 14: P7, 15: P7, 16: P7, 17: P7, 18: P7, 21: P7, 22: P7, 23: P7,
    19: ("SUPERSEDED", "the X3 past-only ceiling lines are on the tip at refc_v3.py:1930-1943 in a later form (condition widened to 'also a speed-only arm (V0)', enc8 branch added)."),
    20: ("APPLIED", "1/1 hunk of diff(c39798d -> variant) is on the tip."),
    24: V7F, 25: V7F, 26: V7F, 27: V7F, 28: V7F,
}
FIX_TEXT = {
    24: "R1 (--max-speed-input-v6 / --plan-vmax-cap) + R4 (--tac-op-cond: the tactical BEHAVIOURS a_lat x a_lon reach the operative plan; MEASURED swap moves plan by exactly 0.0) - package v7f_r1r4/RESULT.md",
    25: "same stream: v6.py model side of R1/R4",
    26: "same stream: NEW declared-vs-built registry for the v6 trainer (new file; incompatible reader signature with the R3 file of the same name)",
    27: "R3 (PI binding): all tactical labels reach the tactical loss - at the tip NO tactical label family reaches ANY v7F loss, traffic-light colours included (D-TLIGHT-1 for v7F) - package v7f_r3/RESULT.md",
    28: "same stream: NEW declared-vs-built registry (v7f_merge holds the UNION of the two)",
}
A_rows = []
for i, (a, a2) in enumerate(zip(A, A2)):
    v, ev = AV[i]
    hs = a2.get("hunks") or []
    cnt = collections.Counter(("ADD-MISSING" if (h["status"] == "CHANGED" and h["n_rem"] == 0) else h["status"]) for h in hs)
    row = {"i": i, "variant": a["variant"], "target": a["target"], "verdict": v, "evidence": ev,
           "variant_blob": a["variant_blob"], "tip_blob": a["tip_blob"], "tip_blob_at": TIP_AB[:8],
           "closest_history_base": {"commit": a2.get("base_commit", "")[:8], "date": a2.get("base_date", "")[:16], "subj": a2.get("base_subj", "")[:110],
                                    "line_diff_to_variant": a2.get("base_to_variant_linediff")},
           "line_diff_tip_to_variant": a2.get("tip_to_variant_linediff"), "hunks": dict(cnt), "variant_blob_in_any_history": a["variant_blob_in_any_history"]}
    if i in FIX_TEXT:
        row["what_it_fixes"] = FIX_TEXT[i]
    A_rows.append(row)
A_counts = collections.Counter(r["verdict"] for r in A_rows)

# ---------------------------------------------------------------- B
BV = {}
def S(i, v, ev): BV[i] = (v, ev)
S(0, "N/A", "MACHINE-LOCAL permission file (3 lines); never landed.")
S(1, "SUPERSEDED", "E is a strict line-subset of the tip (tip only adds 24 lines).")
for i in (2, 3, 4, 5):
    S(i, "SUPERSEDED", "E is an older raw-output draft (E mtime precedes the tip's landing commit); the tip holds the later value.")
S(6, "MISSING", "doc: AMENDMENT A6 (pre-registered 2026-09-26 ~14:40, 'written BEFORE any A6 number existed') - tip's last change is 14:17; 29 of 29 long E-only lines absent from the whole tip. Pre-registration chronology risk: land from E with its timestamp.")
S(7, "MISSING", "doc: 'E1 RESOLVED 2026-09-26' paragraph (+5) absent from the tip.")
S(8, "MISSING", "manifest additions (+43) describing the leaderboard fix; entries partly pending in the chunked landing.")
S(9, "MISSING", "doc: RESULT section 25 'the leaderboard READ runs it never SHOWED, would have re-published four withdrawn runs, and could silently undo a retraction' (+98) - the write-up of the leaderboard code fix below.")
S(10, "MISSING", "manifest additions (+18).")
S(11, "MISSING", "raw JSON: 2 value lines differ (max_abs_traj_diff_m, ckpt path); E newer by 7 days. UNVERIFIED which run is canonical.")
S(12, "REGISTER", "report only. 2 E-only lines (row E10); lead ID is on the tip.")
S(13, "REGISTER", "report only. 23 E-only lines; all 15 lead-IDs found are on the tip (rows exist; E carries in-row wording).")
S(14, "REGISTER", "report only. 10 E-only lines (flagged public_ip/port: not exported); lead IDs on the tip.")
S(15, "REGISTER", "report only. E is a strict subset of the tip (0 E-only lines).")
S(16, "REGISTER", "report only. 3 E-only lines (flagged: not exported); lead ID on the tip.")
S(17, "SUPERSEDED", "E is the 2026-10-04 draft; the tip rewrote the file through d014ff2 (2026-10-12); 763 tip-only lines.")
S(18, "MISSING", "constitution file, 9 lines: Colab Pro capacity policy (PI 2026-09-27). Tip unchanged since 2026-08-23. Master-Mind-owned - land only by the Master Mind.")
S(19, "MISSING", "doc: status banner 'PAUSED - NO DATA YET' (+9).")
S(20, "MISSING", "doc: status banner 'DONE - verdict landed 2026-09-20' (+8).")
S(21, "MISSING", "raw: a re-run of the P-runner mutation proof written 1h40 after the landing (n_passed 32/25/32/32/30 recorded).")
S(22, "SUPERSEDED", "E (2026-10-15 06:29) predates the tip's rewrite 421ffc3 (10-15 18:12).")
S(23, "SUPERSEDED", "1-line difference; tip d257647 is later.")
for i in (24, 25, 26):
    S(i, "SUPERSEDED", "refe/ P6-lineage code: E (2026-10-15 06:xx) predates the tip's later commits (c2d4042 10-18 09:11 / c968f99 / e2d4422); tip-only lines 363/10/7. LIVE file family, but the tip is the newer.")
for i in (27, 28, 30):
    S(i, "MISSING", "doc: appended execution log / design table written after the tip's landing 49a0655 / 2081f5d (E-only 23 / 137 / 29 lines; <=1 found elsewhere on tip).")
S(29, "MISSING", "package-local probe pgrad.py: TF32 OFF (the registered linearity control FAILED with TF32 on: 6.2e-4 vs 1e-4) + the --batch-start/--batch-count PARTS; tip's instrument lacks both. No live chain.")
S(31, "MISSING", "manifest (E 125 entries, +120 lines vs tip); 66 of its listed paths are parse artifacts of the audit (section C).")
S(32, "MISSING", "raw: 1 line (wall_s 50.4).")
S(33, "SUPERSEDED", "registers: E has NO entry key the tip lacks (E 624 keys, tip 639); the 31 changed entries differ only in cited_by/tags/note wording. Report only.")
S(34, "SUPERSEDED", "same as LIBRARY.md - generated from library.json by --reindex; never hand-edit.")
S(35, "MISSING", "doc: 'APPLIED 2026-09-27 on the PI's instruction (register mcp server)' note (+5). Tip unchanged since 2026-08-16.")
S(36, "SUPERSEDED", "E (2026-09-21 00:10) predates the tip's 24d5ce2 (09-26).")
for i in (37, 38, 44, 46):
    S(i, "SUPERSEDED", "E (2026-09-27 02:29) predates the tip's later commits (cef9709 05:06 / 1b8170e 12:42); the tip is the later map_hires version.")
S(39, "MISSING", "LEADERBOARD CODE: adds --discard-hand-edits. Patch: patches/leaderboard_hand_edit_guard_and_withdrawn_runs.patch.")
S(40, "MISSING", "LEADERBOARD CODE: TOMBSTONE (withdrawn-run) handling, hand-edit/fingerprint guard (section_state, dropped_lines), discard_hand_edits.")
S(41, "MISSING", "LEADERBOARD CODE: registers the refcv3-40284 yawmask ARM json as a source (file itself is in the pending landing set).")
S(42, "MISSING", "LEADERBOARD CODE: section fingerprint (FP_PREFIX, section_fingerprint), withdrawn-run table, nuScenes runs shown.")
S(43, "MISSING", "LEADERBOARD CODE: load_bench_runs reads the TOMBSTONE marker; legacy-supersession map excludes withdrawn runs.")
S(45, "MISSING", "LEADERBOARD TEST: run_build(..., discard_hand_edits=True) in 6 call sites.")
assert len(BV) == 47, len(BV)
B_rows = []
for i, r in enumerate(B):
    v, ev = BV[i]
    hc = collections.Counter(("ADD-MISSING" if (h["status"] == "CHANGED" and h["n_rem"] == 0) else h["status"]) for h in r.get("hunks", []))
    b3 = B3.get(str(i))
    B_rows.append({"i": i, "path": r["path"], "audit_verdict": r["audit_verdict"][:60], "mtime_hint": r["mtime_hint"], "verdict": v, "evidence": ev,
                   "e_blob": r["e_blob"], "tip_blob": r.get("tip_blob"), "tip_last_commit": r.get("tip_last_commit"), "tip_last_date": r.get("tip_last_date"),
                   "e_only_nonblank_lines": r.get("e_only_nonblank"), "tip_only_lines": r.get("tip_only_lines"), "hunks": dict(hc),
                   "e_only_lines_found_elsewhere_on_tip": (b3 or {}).get("n_found_on_tip_elsewhere")})
B_counts = collections.Counter(r["verdict"] for r in B_rows)

# ---------------------------------------------------------------- X (extension)
fam = collections.defaultdict(list)
for r in X:
    if r["status"] in ("DIFFERS-NOT-IN-HISTORY", "NEW-NOT-ON-TIP"):
        pk = r.get("pkg_resolved") or r["pkg"]
        if "refcv7-box-head" in pk:
            f = "refcv7 box-head R6 (held)"
        elif "v7f_merge" in pk:
            f = "v7F trainer (v7f_merge)"
        elif "v7f_gate" in pk:
            f = "v7F launch-gate profile (v7f_gate)"
        elif "refav1_r1r6" in pk:
            f = "refav1 R1/R3/R5/R6 (refav1_r1r6)"
        else:
            f = "refav1 R1-VMAX"
        fam[f].append({"repo": r["repo"], "status": r["status"], "tip_equals_declared_base": r.get("tip_equals_declared_base"), "package_file": pk})
X_counts = collections.Counter(r["status"] for r in X)
Cs = collections.Counter(r["reason"].split(" (")[0] for r in C["rows"])

# ---------------------------------------------------------------- patches
patches = []
for n in sorted(os.listdir(W + "patches_out")):
    if n.endswith(".patch") or n.endswith(".json"):
        shutil.copy(W + "patches_out/" + n, D + "patches/" + n)
for n in ("v7f_merge_R1R6", "refav1_R1R6_plus_R1VMAX", "v7f_gate_profile", "leaderboard_hand_edit_guard_and_withdrawn_runs"):
    j = json.load(open(W + "patches_out/" + n + ".json"))
    patches.append(j)

# ---------------------------------------------------------------- raw + scripts
for n, f in [("A_variants_vs_tip.json", "A.json"), ("A_hunk_analysis.json", "A2.json"), ("B_candidate_edits.json", "B.json"),
             ("B_E_only_lines_found_on_tip.json", "B3.json"), ("X_manifest_mappings.json", "X.json"), ("REGISTER_lead_ids.json", "REG.json"),
             ("tests_naming_the_targets.json", "tests_naming.json")]:
    shutil.copy(W + f, D + "raw/" + n)
shutil.copy(W + "C.json", D + "raw/C_landing_ready_absent.json")
for s in ("partA.py", "partA2.py", "partB.py", "partB3.py", "partX.py", "partC.py", "mkpatch.py", "mkpatch3.py", "compose.py"):
    shutil.copy(W + s, D + "scripts/" + s)

summary = {
    "tip_AB_analysed_against": TIP_AB, "tip_C_checked_against": FINAL_TIP,
    "A": dict(A_counts), "B": dict(B_counts), "X_mapping_statuses": dict(X_counts),
    "C_after_cleaning": dict(Cs), "C_absent_entries_in_audit": len(C["rows"]),
}
json.dump({"summary": summary, "A": A_rows, "B": B_rows, "X_unapplied_families": fam, "C_per_manifest": C["per_manifest"], "patches": patches,
           "tests_naming": TN, "register_lead_ids": REG}, open(D + "triage.json", "w"), indent=1)
json.dump(summary, open(W + "summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
