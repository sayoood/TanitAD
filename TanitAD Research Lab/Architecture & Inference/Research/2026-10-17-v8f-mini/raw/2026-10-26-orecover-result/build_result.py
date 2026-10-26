#!/usr/bin/env python3
"""Builds RESULT_V8F_ORECOVER.md from the JSONs. Every number in the document goes through V()/D(), which records
(source key, key path, format, rendered text) in numbers_manifest.json; verify_numbers.py re-reads every one from the JSONs with its own code.
Prose numerals that are not data (dates, section numbers, identifiers) are listed in verify_numbers.py's whitelist."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ART = "E:/Projects/TanitAD-artifacts"
SRC = {
    "READ": ART + "/v8f-orecover-t1/read/ORC_READ.json",
    "FAM": HERE + "/ORC_FAMILIES.json",
    "PF0": HERE + "/PAIRFAM_T0_s0.json",
    "PF1": HERE + "/PAIRFAM_T0_s1.json",
}
for a in ("ORC", "ORC0"):
    for s in (0, 1):
        SRC[f"T1J_{a}_s{s}"] = f"{ART}/v8f-orecover-t1/t1v8f_{a}_s{s}_m10_causal_heads.json"
        SRC[f"T0J_{a}_s{s}"] = f"{ART}/v8f-orecover/M2b_s{s}_{a.lower()}/eval_M2b-{a.lower()}_s{s}.json"
OUT_MD = HERE + "/RESULT_V8F_ORECOVER.md"
OUT_MAN = HERE + "/numbers_manifest.json"
_cache = {}


def J(k):
    if k not in _cache:
        with open(SRC[k], encoding="utf-8") as f:
            _cache[k] = json.load(f)
    return _cache[k]


def get(src, path):
    o = J(src)
    for p in str(path).split("|"):          # '|' separates path parts so keys may hold dots
        o = o[int(p)] if isinstance(o, list) else o[p]
    return o


MAN = []


def fmt(kind, x):
    if kind == "str":
        return str(x)
    if kind == "yn":
        return "yes" if x else "no"
    if kind == "i":
        return f"{int(x)}"
    if kind == "ic":
        return f"{int(x):,}"
    if kind.startswith("s") and kind[1:].isdigit():
        return f"{x:+.{int(kind[1:])}f}"
    if kind.startswith("f") and kind[1:].isdigit():
        return f"{x:.{int(kind[1:])}f}"
    if kind == "e":
        return f"{x:.1e}"
    if kind == "g6":
        return f"{x:.6g}"
    if kind == "sha8":
        return str(x)[:8]
    if kind == "sha16":
        return str(x)[:16]
    raise ValueError(kind)


def V(src, path, kind="f4"):
    x = get(src, path)
    t = fmt(kind, x)
    MAN.append({"op": "V", "src": src, "path": str(path), "fmt": kind, "text": t})
    return t


def D(op, a, b, kind="f2"):
    """Derived: a, b are (src, path). op in div (|a|/|b|), sub (a-b), abs."""
    xa, xb = get(*a), get(*b)
    x = {"div": abs(xa) / abs(xb), "sub": xa - xb, "abs": abs(xa), "mean": (xa + xb) / 2}[op]
    t = fmt(kind, x)
    MAN.append({"op": "D:" + op, "a": list(a), "b": list(b), "fmt": kind, "text": t})
    return t


def LEN(src, path):
    t = str(len(get(src, path)))
    MAN.append({"op": "LEN", "src": src, "path": str(path), "text": t})
    return t


def CNT(src, path, name):
    t = str(sum(1 for k in get(src, path) if k == name or k.startswith(name + ":")))
    MAN.append({"op": "CNT", "src": src, "path": str(path), "name": name, "text": t})
    return t


def tag(sep, delta, direction):
    """direction: 'lower' / 'higher' = which way is better for the ORC - ORC-0 difference; None = no label."""
    if not sep:
        return "not separated"
    if direction is None:
        return "separated (no direction label)"
    if direction == "lowerS":
        return "separated, S nearer than CTRV" if delta < 0 else "separated, S farther than CTRV"
    better = (delta < 0) if direction == "lower" else (delta > 0)
    return "separated, ORC better" if better else "separated, ORC WORSE"


def cell(src, base, direction=None, kd="delta", dp=4, with_n=False):
    """'+d [lo, hi] tag' for the node at base (keys kd, lo, hi, separated)."""
    n = get(src, base)
    d, lo, hi, sep = n.get(kd), n.get("lo"), n.get("hi"), n.get("separated")
    if not all(isinstance(v, (int, float)) for v in (d, lo, hi)):
        return "n/c (" + str(n.get("status", "no interval")) + ")"
    t = f"{V(src, base + '|' + kd, 's%d' % dp)} [{V(src, base + '|lo', 's%d' % dp)}, {V(src, base + '|hi', 's%d' % dp)}], {tag(sep, d, direction)}"
    MAN.append({"op": "TAG", "src": src, "base": base, "direction": direction, "kd": kd, "text": tag(sep, d, direction)})
    if with_n:
        t += f" (n {V(src, base + '|n_windows', 'ic')} / {V(src, base + '|n_episodes', 'i')})"
    return t


L = []


def w(s=""):
    L.append(s)


R = lambda p, k="f4": V("READ", p, k)                                    # noqa: E731
FAMP = lambda t, s, key: f"families_paired_ORC_minus_ORC0|{key}"        # noqa: E731
PFS = lambda s: f"PF{s}"                                                # noqa: E731

# =====================================================================================================================================================
w("# PREREG_V8F_ORECOVER: result, both seeds (ORC vs ORC-0; trained on box 2, T1-kin read on the dev-box CPU; read 2026-10-26)")
w("")
w(f"**Outcome, as committed: {V('READ', 'OUTCOME|OUTCOME', 'str')}.** The pre-registered reader (`orc_read.py`, sha256 `{V('READ', 'reader_sha256', 'sha16')}…`) matched exactly one row of the outcome table, "
  f"`{V('READ', 'OUTCOME|matching_rows_in_table_order|0', 'str')}`, on its condition *\"{V('READ', 'OUTCOME|condition', 'str')}\"*. Two guards fired: R-ADE is separated-worse at seed 0 ("
  f"{V('READ', 'rows|s0|R-ADE|delta', 's4')} m), and R-T0 is above +F/2 at both seeds (seed 0 {V('READ', 'rows|s0|R-T0|point', 's4')} m, seed 1 {V('READ', 'rows|s1|R-T0|point', 's4')} m, against {V('READ', 'bars|F_over_2_t0_point_m', 'f4')} m).")
w("")
w("* **Committed next, verbatim from the PREREG's HARMS row:** *\"" + V("READ", "OUTCOME|committed_next", "str") + "\"*.")
w(f"* **Evidence class:** {V('READ', 'evidence_class', 'str')}. Every number below is copied from `ORC_READ.json` (the reader; sha256 `a50dbba9…`), from `ORC_FAMILIES.json` / `PAIRFAM_T0_s0.json` / `PAIRFAM_T0_s1.json` "
  "(section 3; produced by this draft's driver and by the unchanged programme CLI), or from the run records they point at. The key path of every number is in `numbers_manifest.json`; `verify_numbers.py` re-reads all of them.")
w(f"* **Tier, as the reader states it:** {V('READ', 'tier', 'str')}. The families in section 3 carry their own tier label per table. Nothing here is a driving claim, and \"closed loop\" does not occur in this document except in that negation.")
w(f"* **Estimator of every interval:** {V('READ', 'estimator', 'str')}")
w(f"* **Pre-registration:** {V('READ', 'prereg', 'str')}. Claim row `{V('READ', 'claim_row', 'str')}`. Code under test and gate commit: `{V('READ', 'controls|GATE-TOKEN:ORC_s0|detail|commit', 'str')}`.")
w("* **Status of this file:** a draft for the Master Mind. Nothing was staged into the repository, committed or pushed by it.")
w("")
w("---")
w("")
w("## 1. The outcome as committed")
w("")
w("The PREREG's outcome table (section 4, with Amendment 2's wording of NO EFFECT), quoted:")
w("")
w("| outcome | condition | committed next |")
w("|---|---|---|")
w("| ROBUST | R-TLAT and R-TLON separated-better and >= 2F, at both seeds; R-ADE not separated-worse; R-T0 within +F/2 | adopt ORC as the base. Re-read T1 dominance over P10 and the FLAT control on it |")
w("| PARTIAL | at least one F1 row separated-better and >= 2F at both seeds, the other not, with the guards held | adopt for the head that moved; the next arm targets the other head's inputs |")
w("| NO EFFECT | no F1 row separated-better and >= 2F at both seeds, guards held | on-policy states (DAgger form), under its own PREREG |")
w("| **HARMS** | **R-ADE separated-worse at >= 1 seed, or R-T0 above +F/2** | **ORC is not adopted. Reported with the rows; next is NO EFFECT's arm at p = 0.1** |")
w("| REFUSED | any control fails | fix and re-run |")
w("")
w(f"The reader evaluates every row's predicate literally and lists every match in table order: `{V('READ', 'OUTCOME|matching_rows_in_table_order|0', 'str')}` only. The guards-held flag reads "
  f"`{V('READ', 'OUTCOME|flags|guards_held', 'yn')}`. Because HARMS tests the guards, the F1 rows below cannot change the outcome in either direction.")
w("")
w("### 1.1 The four rows, per seed (ORC - ORC-0)")
w("")
w(f"**Estimator:** paired episode-cluster bootstrap, B = {V('READ', 'n_boot', 'ic')}, {V('READ', 'n_episodes', 'i')} episodes, {V('READ', 'n_windows', 'ic')} windows. "
  f"F1 rows: T1-kin, ticks >= 1, *matched* cells (cells scored in both runs; the reader's adopted reading, Amendment 2). ADE rows: per-window mean. "
  f"Bars: 2F = {V('READ', 'bars|2F|lat', 'f4')} (lat) / {V('READ', 'bars|2F|lon', 'f4')} (lon), with F = {V('READ', 'bars|seed_floor|lat', 'f4')} / {V('READ', 'bars|seed_floor|lon', 'f4')} (the T0 fresh-age values the PREREG names); "
  f"the R-ADE guard is *not separated-worse*; the R-T0 guard is point <= {V('READ', 'bars|F_over_2_t0_point_m', 'f4')} m (F/2, with F = {V('READ', 'bars|F_ade6_m', 'f4')} m).")
w("")
w("| row | seed | tier | ORC (level) | ORC-0 (level) | ORC - ORC-0 point [95 % interval] | separated | bar | bar met |")
w("|---|---|---|---|---|---|---|---|---|")
for s in (0, 1):
    for key, lab, head in (("R-TLAT", "R-TLAT: lat macro-F1, ticks >= 1", "lat"), ("R-TLON", "R-TLON: lon macro-F1, ticks >= 1", "lon")):
        b = f"rows|s{s}|{key}"
        w(f"| {lab} | s{s} | T1-kin | {R(b + '|f1_a')} | {R(b + '|f1_b')} | **{R(b + '|delta', 's4')} [{R(b + '|lo', 's4')}, {R(b + '|hi', 's4')}]** | {R(b + '|separated', 'yn')} "
          f"| separated-better and >= 2F ({R(b + '|two_F')}) | {R(b + '|meets_2F', 'yn')} |")
    b = f"rows|s{s}|R-ADE"
    w(f"| R-ADE: T1-kin plain ADE6, m | s{s} | T1-kin | {R(b + '|mean_a')} | {R(b + '|mean_b')} | **{R(b + '|delta', 's4')} [{R(b + '|lo', 's4')}, {R(b + '|hi', 's4')}]** | {R(b + '|separated', 'yn')} "
      f"| guard: not separated-worse | separated-worse: **{R(b + '|worse', 'yn')}** |")
    b = f"rows|s{s}|R-T0"
    w(f"| R-T0: T0 own-code ADE6, m | s{s} | T0 | {R(b + '|mean_a')} | {R(b + '|mean_b')} | **{R(b + '|delta', 's4')} [{R(b + '|lo', 's4')}, {R(b + '|hi', 's4')}]** (unrounded point {R(b + '|point', 's6')}) | {R(b + '|separated', 'yn')} "
      f"| guard: point <= {R(b + '|bar_point_le')} | above the bar: **{R('OUTCOME|flags|R-T0_above_F_over_2_per_seed|s%d' % s, 'yn')}** |")
w("")
w(f"Cell counts behind the F1 rows (matched pairs): lat s0 {R('rows|s0|R-TLAT|n_cells_matched_pairs', 'ic')}, s1 {R('rows|s1|R-TLAT|n_cells_matched_pairs', 'ic')}; lon s0 {R('rows|s0|R-TLON|n_cells_matched_pairs', 'ic')}, s1 {R('rows|s1|R-TLON|n_cells_matched_pairs', 'ic')}. "
  f"All four rows are over {R('rows|s0|R-TLAT|n_episodes', 'i')} episodes.")
w("")
w("### 1.2 Which guard fired")
w("")
w(f"* **R-ADE (T1-kin ADE6), separated-worse:** seed 0 **{R('OUTCOME|flags|R-ADE_separated_worse_per_seed|s0', 'yn')}** ({R('rows|s0|R-ADE|delta', 's4')} m, interval excludes 0), seed 1 **{R('OUTCOME|flags|R-ADE_separated_worse_per_seed|s1', 'yn')}** "
  f"({R('rows|s1|R-ADE|delta', 's4')} [{R('rows|s1|R-ADE|lo', 's4')}, {R('rows|s1|R-ADE|hi', 's4')}] m, not separated). HARMS fires at any seed, so seed 0 alone fires it.")
w(f"* **R-T0 (T0 ADE6) above +F/2:** seed 0 **{R('OUTCOME|flags|R-T0_above_F_over_2_per_seed|s0', 'yn')}** ({R('rows|s0|R-T0|point', 's4')} m), seed 1 **{R('OUTCOME|flags|R-T0_above_F_over_2_per_seed|s1', 'yn')}** ({R('rows|s1|R-T0|point', 's4')} m); the bar is {R('bars|F_over_2_t0_point_m')} m. "
  f"Both intervals exclude 0 (s0 [{R('rows|s0|R-T0|lo', 's4')}, {R('rows|s0|R-T0|hi', 's4')}], s1 [{R('rows|s1|R-T0|lo', 's4')}, {R('rows|s1|R-T0|hi', 's4')}]). This second clause fires at both seeds independently of the first.")
w(f"* **Size against F:** the T0 points are {D('div', ('READ', 'rows|s0|R-T0|point'), ('READ', 'bars|F_ade6_m'), 'f2')} F (s0) and {D('div', ('READ', 'rows|s1|R-T0|point'), ('READ', 'bars|F_ade6_m'), 'f2')} F (s1), "
  f"with F = {R('bars|F_ade6_m')} m the only run-to-run information on this rig (two M2b seeds). Two M2b training runs differ by F on this metric, so this rig alone cannot tell a difference of that size from training noise (H-ESTIM-SEED-1); see section 4.")
w("")
w("### 1.3 The F1 rows, which the HARMS reading does not use (reported as the PREREG requires)")
w("")
w(f"* **R-TLAT (lat macro-F1):** the points are {R('rows|s0|R-TLAT|delta', 's4')} (s0) and {R('rows|s1|R-TLAT|delta', 's4')} (s1), both above 2F = {R('bars|2F|lat')}; neither interval is separated "
  f"(s0 lower bound {R('rows|s0|R-TLAT|lo', 's4')}, s1 lower bound {R('rows|s1|R-TLAT|lo', 's4')}). So the bar \"separated-better and >= 2F\" is met at neither seed (separated: s0 {R('rows|s0|R-TLAT|separated', 'yn')}, s1 {R('rows|s1|R-TLAT|separated', 'yn')}).")
w(f"* **R-TLON (lon macro-F1):** s0 {R('rows|s0|R-TLON|delta', 's4')} [{R('rows|s0|R-TLON|lo', 's4')}, {R('rows|s0|R-TLON|hi', 's4')}], not separated; s1 {R('rows|s1|R-TLON|delta', 's4')} [{R('rows|s1|R-TLON|lo', 's4')}, {R('rows|s1|R-TLON|hi', 's4')}], "
  f"separated-better and above 2F = {R('bars|2F|lon')} (meets: {R('rows|s1|R-TLON|meets_2F', 'yn')}). The reader's flag `separated_better_and_ge_2F_per_seed.lon` reads s0 {R('OUTCOME|flags|separated_better_and_ge_2F_per_seed|lon|s0', 'yn')}, s1 {R('OUTCOME|flags|separated_better_and_ge_2F_per_seed|lon|s1', 'yn')}; for lat both read no. "
  "The two seeds disagree in sign on the lon row.")
w(f"* **Own-cells version beside (cells scored in each run on its own, not the matched ones):** lat s0 {R('rows|s0|R-TLAT_own_cells_beside|delta', 's4')} [{R('rows|s0|R-TLAT_own_cells_beside|lo', 's4')}, {R('rows|s0|R-TLAT_own_cells_beside|hi', 's4')}] (separated: {R('rows|s0|R-TLAT_own_cells_beside|separated', 'yn')}), "
  f"lat s1 {R('rows|s1|R-TLAT_own_cells_beside|delta', 's4')} [{R('rows|s1|R-TLAT_own_cells_beside|lo', 's4')}, {R('rows|s1|R-TLAT_own_cells_beside|hi', 's4')}] (separated: {R('rows|s1|R-TLAT_own_cells_beside|separated', 'yn')}, meets 2F: {R('rows|s1|R-TLAT_own_cells_beside|meets_2F', 'yn')}); "
  f"lon s0 {R('rows|s0|R-TLON_own_cells_beside|delta', 's4')} [{R('rows|s0|R-TLON_own_cells_beside|lo', 's4')}, {R('rows|s0|R-TLON_own_cells_beside|hi', 's4')}], lon s1 {R('rows|s1|R-TLON_own_cells_beside|delta', 's4')} [{R('rows|s1|R-TLON_own_cells_beside|lo', 's4')}, {R('rows|s1|R-TLON_own_cells_beside|hi', 's4')}]. "
  "Disclosure: on the own-cells definition the lat row at s1 has its interval just above 0. The decisive definition is the matched one (the PREREG adopts the reader's choice in Amendment 2), and at s0 the lat row is not separated under either definition, so no definition meets the lat bar at both seeds.")
w(f"* **The reader's other flags:** `head_moved_at_both_seeds` lat {R('OUTCOME|flags|head_moved_at_both_seeds|lat', 'yn')}, lon {R('OUTCOME|flags|head_moved_at_both_seeds|lon', 'yn')}; `sub_bar_separated_better_at_both_seeds` lat {R('OUTCOME|flags|sub_bar_separated_better_at_both_seeds|lat', 'yn')}, lon {R('OUTCOME|flags|sub_bar_separated_better_at_both_seeds|lon', 'yn')}.")
w("")
w("---")
w("")
w("## 2. The controls: all PASS")
w("")
ctl = J("READ")["controls_PASS"]
ctl_n = len(ctl)
w(f"The reader ran {LEN('READ', 'controls_PASS')} controls and every one passed; a failed control would have produced REFUSED. The count is len(`controls_PASS`) and the all-true test is re-run in `verify_numbers.py`. "
  "By family (the second column counts how many of the controls are in the family; the third says what it checks; `s0`/`s1`/`ORC`/`ORC0` instances are counted separately):")
w("")
w("| control family | instances | what it checks |")
w("|---|---|---|")
FAMS = [
    ("PREREG-TEXT", "every cell of the outcome table is verbatim in the PREREG file"),
    ("M2B-CKPT", "the M2b init checkpoint's sha256 equals the one banked in the PREREG (`601cb435…` / `6104fb15…`)"),
    ("ARGV", "each run's argv names the registered fine-tune"),
    ("SUMMARY", "the run's summary record (argv sha, 3,000 steps)"),
    ("INIT-CKPT", "the trainer's recorded `--init-ckpt` sha256 equals the M2b checkpoint's"),
    ("PERTURB-REACHED", "the perturbation reached training: ORC's perturbed share of eligible samples is 0.3 +/- 0.02; ORC-0's count is exactly 0"),
    ("GATE-TOKEN", "a launch-gate PASS token binding commit, argv sha256 and data sha256 (the HMAC is not verified by the reader: key on the launch host; the tokens record `hmac: absent`)"),
    ("T0-CKPT", "box 2's standard T0 eval names this run's checkpoint"),
    ("T0F-CKPT", "the fresh-age T0 eval names this run's checkpoint (Amendment 4)"),
    ("T0-AGES", "the standard eval says ages `cycle`, the fresh eval ages `fresh`, so neither stands in for the other"),
    ("PAIR", "ORC and ORC-0 share init, steps, data and seed; argv differs only in the perturbation flags and `--out`"),
    ("K1", "T1-HEADS K1 (instrument identity on the run)"),
    ("K2", "T1-HEADS K2, causal run"),
    ("K3", "T1-HEADS K3, causal run"),
    ("K5", "T1-HEADS K5 (mirror control)"),
    ("K6", "T1-HEADS K6"),
    ("K7", "the executed track rebuilt from the dumped plans equals the dumped track (Amendment 2's new K7)"),
    ("GRID", "the causal run is on the 2,399-window grid"),
    ("T1-CONFIG", "the causal run's configuration is the registered one (causal bucket, m = 10, periodic, logged scene, dead-reckoned nav, heads dumped)"),
    ("K2-logged", "K2 on the `--limit logged` run"),
    ("K3-logged", "K3 on the `--limit logged` run"),
    ("GRID-logged", "the logged run is on the grid"),
    ("T1-CONFIG-logged", "the logged run's configuration"),
    ("K4", "tick 0 of the logged run equals the arm's own fresh-age T0 eval to 1e-3 (Amendment 4)"),
    ("DUMP-WINDOWS", "dump windows equal tick windows, window for window"),
    ("GRID-COUNTS", "2,399 windows / 141 episodes"),
    ("R2-ADE6", "the reader's ADE6 delta equals the run's own analysis delta"),
    ("SAME-WINDOWS", "ORC and ORC-0 are on identical windows (ticks, dumps, T0 eid)"),
    ("K1-ORC-ROWS", "the reader's own K1 on its output rows"),
]
tot = 0
for nm, what in FAMS:
    cnt = sum(1 for k in ctl if k == nm or k.startswith(nm + ":"))
    tot += cnt
    w(f"| `{nm}` | {CNT('READ', 'controls_PASS', nm)} | {what} |")
assert tot == ctl_n, (tot, ctl_n)
w("")
w(f"Control values worth quoting (all from `controls.*.detail`):")
w("")
w("| run | perturbed / eligible samples | share | steps | K4: max abs position diff vs own T0 (3 horizons), m | K4: lat / lon F1 diff | K7: max track diff, m |")
w("|---|---|---|---|---|---|---|")
for r in ("ORC_s0", "ORC_s1", "ORC0_s0", "ORC0_s1"):
    pr = f"controls|PERTURB-REACHED:{r}|detail"
    if r.startswith("ORC_"):
        pcol = f"{R(pr + '|ego_perturb_n', 'ic')} / {R(pr + '|ego_perturb_eligible', 'ic')} | {R(pr + '|ratio', 'f4')}"
    else:
        pcol = f"{R(pr + '|ego_perturb_n', 'ic')} (exactly 0) | n/a"
    k4 = f"controls|K4:{r}|detail|abs_diff_vs_own_T0"
    w(f"| {r} | {pcol} | {R('controls|SUMMARY:' + r + '|detail|steps', 'ic')} | {R(k4 + '|pos|0', 'e')}, {R(k4 + '|pos|1', 'e')}, {R(k4 + '|pos|2', 'e')} | {R(k4 + '|lat', 'f4')} / {R(k4 + '|lon', 'f4')} | {R('controls|K7:' + r + '|detail|x1|max_cl_m', 'f4')} |")
w("")
w(f"* **Deliberate regression:** not spent as a GPU arm, as committed: the PREREG records it as proven at build time (deleting the perturbation line turns the lever's tests RED). Not re-verified here (INHERITED).")
w(f"* **Gate token HMAC:** the four tokens record `hmac: {R('controls|GATE-TOKEN:ORC_s0|detail|hmac', 'str')}`; the reader checks the token's binding to argv, commit and data, and does not verify an HMAC. So \"launch-gate PASS\" here means the binding was checked, not that the signature was.")
w(f"* **Provenance pins:** M2b init checkpoints s0 `{R('m2b_ckpt_sha256|s0', 'sha16')}…`, s1 `{R('m2b_ckpt_sha256|s1', 'sha16')}…`; manifest sha256 `{R('manifest_sha256', 'sha16')}…`; the reader read {R('wall_s', 'f1')} s.")
w("")
w("---")
w("")

# =====================================================================================================================================================
w("## 3. The four metric families (binding), per family, never pooled")
w("")
w("The reader's rows are two F1 rows and two ADE6 rows, and HARMS is decided by the two ADE6 rows alone. The binding rule (CLAUDE.md, \"every eval reports four metric families\") asks for LONGITUDINAL, LATERAL, TACTICAL and STRATEGIC beside the ADE. "
  "The reader computes only part of that (the lat / lon head F1 at T1-kin, per-run S-POS against CTRV, the tick-0 and goal F1s). The rest was computed for this draft, ORC - ORC-0 per seed on the same windows, with the programme's own instruments:")
w("")
w("* `stack/scripts/v8f_paired_families.py` (unchanged CLI, commit `07dcb513`): TACTICAL (plan-derived decisions, goal point, lat / lon head F1, goal head F1) and STRATEGIC (S's checkpoint position and heading error at 30 / 50 / 80 m), on the **T0 eval npz** of each arm. It reads only an eval `run` dump, so it cannot read the T1 tick dumps (reason stated per row).")
w("* `taniteval.v8f_beta_eval.family_windows` + `paired_row` (the paired per-window family instrument that `v8f_kscore.py` and the OACC reader use) for LONGITUDINAL speed / accel / along-track, LATERAL heading / curvature / yaw-rate / cross-track and the plan-derived decision agreement; "
  "and `v8f_dk_read.read_dk` (the distance-keeping reader, with its own identity control) for headway / time gap / TTC / LEAD-VIOL. These run on both tiers: T0 on box 2's standard eval `pred`, T1-kin on the `cl` (executed) track of the causal dump. Driver: `orc_families.py`.")
w(f"* **Same estimator everywhere:** paired episode-cluster bootstrap, B = {V('FAM', 'n_boot', 'ic')}, seed 0, over {V('FAM', 'n_episodes', 'i')} episodes ({V('FAM', 'n_windows', 'ic')} windows; rows restricted to windows with a valid value say their n).")
w("")
w("**Controls on this section (all must hold):**")
w("")
rep = J("FAM")["controls"]["reproduces_reader_ade6"]
w(f"* **Reproduction of the reader.** The driver's ADE6 ORC - ORC-0 equals the reader's, to the estimator's own four decimals, interval and separation included: T0 s0 {V('FAM', 'controls|reproduces_reader_ade6|s0|T0|here|delta', 's4')} vs {V('READ', 'rows|s0|R-T0|delta', 's4')}, "
  f"T0 s1 {V('FAM', 'controls|reproduces_reader_ade6|s1|T0|here|delta', 's4')} vs {V('READ', 'rows|s1|R-T0|delta', 's4')}, T1-kin s0 {V('FAM', 'controls|reproduces_reader_ade6|s0|T1|here|delta', 's4')} vs {V('READ', 'rows|s0|R-ADE|delta', 's4')}, "
  f"T1-kin s1 {V('FAM', 'controls|reproduces_reader_ade6|s1|T1|here|delta', 's4')} vs {V('READ', 'rows|s1|R-ADE|delta', 's4')}. PASS flags: s0 {V('FAM', 'controls|reproduces_reader_ade6|s0|PASS', 'yn')}, s1 {V('FAM', 'controls|reproduces_reader_ade6|s1|PASS', 'yn')}. "
  "So the families below are on the reader's windows and the reader's tracks.")
w("* **Identity.** Every paired family row run on ORC against ORC reads exactly +0 [0, 0], and `read_dk`'s own identity control (headway, time gap, TTC, LEAD-VIOL) reads +0 [0, 0] (flags in `ORC_FAMILIES.json`, `identity_all_exactly_zero` and `distance_keeping.identity_control`; asserted again in `verify_numbers.py`).")
w(f"* **Grid.** At T1-kin the dump's windows, episode ids, `2 * ws` frames and `v0` equal the T0 grid's for both arms and seeds; the dump GT differs from the grid GT by at most {V('FAM', 'controls|T1_grid_s0|ORC|g_max_abs_diff_vs_grid_gt_m', 'e')} m (float32 round trip). The lead block's sha256 is `{V('FAM', 'inputs|lead_block_sha256', 'sha16')}…`, the registered one.")
w(f"* **Cross-instrument.** At T0 the plan-derived lateral / longitudinal agreement from `v8f_paired_families.py` equals the `family_windows` rows below (asserted in `verify_numbers.py`).")
w("")

TIERS = (("T0", "T0 (teacher-forced own-code; box 2's standard eval-141, ages cycle)"), ("T1", "T1-kin (self-action OPEN loop; plain causal bucket; nav dead-reckoned; m = 10)"))
w("**Tier labels used in the tables.** `T0` = the PREREG's name for box 2's standard eval (R-T0's source). The eval JSON stamps itself "
  f"`raw.tier = \"{V('T0J_ORC_s0', 'raw|tier', 'str')}\"` and the paired-families tool stamps its output \"T1-class open loop (own held outputs; no recorded future action)\"; "
  "these are three labels for one open-loop scoring of the logged future, and I have not reconciled them (the same open item as in `RESULT_V8F_OACC.md`). I use the PREREG's names. "
  "`T1-kin` = the causal-bucket dev-box rollout whose executed track `cl` the reader scores; the lead follows its LOGGED future and does not react to the ego.")
w("")
w("Column key: each cell is `ORC - ORC-0` point [95 % interval], then whether the interval excludes 0 and which way is better. Directions are labelled only where the metric has one (error: lower is better; agreement, F1: higher; LEAD-VIOL share: lower). "
  "Levels (each arm's mean on the paired windows) are in `ORC_FAMILIES.json` (`mean_a_on_pairs`, `mean_b_on_pairs`).")
w("")

FK = lambda t, s, key: f"{t}|s{s}|families_paired_ORC_minus_ORC0|{key}"    # noqa: E731


def famrow(label, key, direction):
    cs = []
    for t in ("T0", "T1"):
        for s in (0, 1):
            cs.append(cell("FAM", FK(t, s, key), direction, kd="mean", with_n=True))
    return f"| {label} | " + " | ".join(cs) + " |"


HDR = "| cell (ORC - ORC-0) | T0 s0 | T0 s1 | T1-kin s0 | T1-kin s1 |\n|---|---|---|---|---|"

# ------------------------------------------------------------------------------------------------------------------------ LONGITUDINAL
w("### 3.1 LONGITUDINAL")
w("")
w("**Anti-echo caveat, per tier.** At T0 the eval JSONs of all four arms carry `anti_echo.status`: " + ", ".join(
    f"{a}_s{s} {V('T0J_%s_s%d' % (a, s), 'raw|families|longitudinal|anti_echo|status', 'str')}" for a in ("ORC", "ORC0") for s in (0, 1)) +
  " (no ego speed at t0 in the window dict), and `distance_keeping.status` is " + V('T0J_ORC_s0', 'raw|families|longitudinal|distance_keeping|status', 'str') +
  " in the eval JSON itself; the distance-keeping rows below come from the lead-block join (`read_dk`), not from the eval JSON. "
  "At T1-kin the anti-echo block is computed and reads, for each of the four runs: ")
w("")
for a in ("ORC", "ORC0"):
    for s in (0, 1):
        p = f"arms|cl|four_families|longitudinal|anti_echo"
        w(f"* `{a}_s{s}`: status {V('T1J_%s_s%d' % (a, s), p + '|status', 'str')}, `longitudinal_claim_admissible` **{V('T1J_%s_s%d' % (a, s), p + '|longitudinal_claim_admissible', 'yn')}**, flagged {V('T1J_%s_s%d' % (a, s), p + '|flagged', 'yn')}; {V('T1J_%s_s%d' % (a, s), p + '|summary', 'str')}.")
w("")
w("⇒ **Every LONGITUDINAL number in this document, for ORC and for ORC-0 alike, is a paired contrast of two arms that the programme's own block does not admit as a longitudinal capability claim.** The contrast ORC - ORC-0 is still a valid description of what the perturbation changed on these windows; it is not a statement that either arm sets the right speed.")
w("")
w(HDR)
w(famrow("**speed MAE, m/s**", "lon_speed_mae", "lower"))
w(famrow("**accel MAE, m/s^2**", "lon_accel_mae", "lower"))
w(famrow("along-track MAE, m", "lon_along_mae", "lower"))
w("")
w("**Distance keeping, K = 30 (6 s), where a lead exists** (`read_dk`; the lead block's windows: ", )
L[-1] += (f"LEAD {V('FAM', 'T0|s0|distance_keeping|join|window_states|LEAD', 'ic')}, NO_LEAD {V('FAM', 'T0|s0|distance_keeping|join|window_states|NO_LEAD', 'ic')}, "
          f"NOT_STRAIGHT {V('FAM', 'T0|s0|distance_keeping|join|window_states|NOT_STRAIGHT', 'ic')}, NO_LABEL {V('FAM', 'T0|s0|distance_keeping|join|window_states|NO_LABEL', 'ic')}; "
          "only LEAD windows are scored, the others are counted and never scored as free flow; the lead follows its logged future). Differences are ORC - ORC-0; n shown as windows used / episodes.")
w("")
w("| cell (ORC - ORC-0) | T0 s0 | T0 s1 | T1-kin s0 | T1-kin s1 |")
w("|---|---|---|---|---|")


def dkrow(label, sub, direction, kd="delta"):
    cs = []
    for t in ("T0", "T1"):
        for s in (0, 1):
            base = f"{t}|s{s}|distance_keeping|pair|{sub}"
            n = get("FAM", base)
            nkey = "n_used" if "n_used" in n else "n_windows"
            cs.append(cell("FAM", base, direction, kd=kd)+ f" (n {V('FAM', base + '|' + nkey, 'ic')} / {V('FAM', base + '|n_episodes', 'i')})")
    return f"| {label} | " + " | ".join(cs) + " |"


w(dkrow("min headway, m", "dk|headway_min_m", None))
w(dkrow("min time gap, s", "dk|time_gap_min_s", None))
w(dkrow("min TTC, s (censored at a 30 s cap)", "dk|min_ttc_s", None))
w(dkrow("\\|headway error\\| vs the human, m", "abs_error_vs_human|headway_min_m", "lower"))
w(dkrow("\\|time-gap error\\| vs the human, s", "abs_error_vs_human|time_gap_min_s", "lower"))
w(dkrow("\\|min-TTC error\\| vs the human, s", "abs_error_vs_human|min_ttc_s", "lower"))
w(dkrow("LEAD-VIOL, at risk (violating share)", "lead_viol|at_risk", "lower"))
w(dkrow("LEAD-VIOL, all lead windows", "lead_viol|all_lead_windows", "lower"))
w(dkrow("lead-progress ratio", "lead_viol|progress", None))
w("")
w("LEAD-VIOL counts (violations ORC / ORC-0 among the at-risk windows):")
w("")
w("| tier, seed | at risk | violations ORC | violations ORC-0 |")
w("|---|---|---|---|")
for t, tl in (("T0", "T0"), ("T1", "T1-kin")):
    for s in (0, 1):
        b = f"{t}|s{s}|distance_keeping|pair|lead_viol|at_risk"
        w(f"| {tl} s{s} | {V('FAM', b + '|n_at_risk', 'ic')} | {V('FAM', b + '|n_viol_a', 'ic')} | {V('FAM', b + '|n_viol_b', 'ic')} |")
w("")
w(f"Min-TTC means are over censored data: the closing-window counts per arm are in `ORC_FAMILIES.json` (`distance_keeping.arms.*.metrics.min_ttc_s.n_closing`). "
  f"`headway`, `time gap` and `TTC` carry no direction label from the instrument (more distance kept is not automatically better), so they are tagged \"no direction label\" when separated.")
w("")

# ------------------------------------------------------------------------------------------------------------------------ LATERAL
w("### 3.2 LATERAL")
w("")
w(HDR)
w(famrow("**heading MAE, deg**", "lat_heading_mae_deg", "lower"))
w(famrow("**curvature MAE, 1/m**", "lat_curvature_mae_1pm", "lower"))
w(famrow("**yaw-rate MAE, deg/s**", "lat_yaw_rate_mae_degps", "lower"))
w(famrow("cross-track MAE, m", "lat_cross_mae", "lower"))
w("")
w("Heading, curvature and yaw-rate are means of per-window components over the windows where both paths move (so their n is below the full grid); the n of every cell is printed. "
  "Cross-track is an offset at a matched time index, not a distance to the GT path, and is uninterpretable alone when the along-track error is large (the instrument's own caveat).")
w("")

# ------------------------------------------------------------------------------------------------------------------------ TACTICAL
w("### 3.3 TACTICAL")
w("")
w("**Plan-derived decisions** (the manoeuvre read off each arm's executed plan against the one read off the human path; both tiers; `family_windows`):")
w("")
w(HDR)
w(famrow("lateral decision agreement", "tac_lat_acc", "higher"))
w(famrow("longitudinal decision agreement", "tac_lon_acc", "higher"))
w("")
w("**Head-based decisions and goal setting.** The decisive R-TLAT / R-TLON are in section 1.1 (T1-kin, ticks >= 1). The T0 versions below are the head argmax at the window, against the v9 labels, from `v8f_paired_families.py` (the head F1 is re-computed on every bootstrap resample). "
  "The paired-families tool cannot read the T1 tick dumps (it requires an eval `run` dump with `pred`, `lat_logits`, `lon_logits`, `goal_prob`, `ckpt`, `ck_pos_err`, `ade6` per window), so the T1-kin head rows are the reader's R-TLAT / R-TLON and the per-run values beside.")
w("")
w("| cell (ORC - ORC-0) | T0 s0 | T0 s1 | T1-kin s0 | T1-kin s1 |")
w("|---|---|---|---|---|")


def pfrow(label, path, direction, t1=None):
    c0 = cell("PF0", path, direction, with_n=True)
    c1 = cell("PF1", path, direction, with_n=True)
    t1v = t1() if callable(t1) else t1                       # evaluated AFTER c0 / c1 so the manifest order is the document order
    return f"| {label} | {c0} | {c1} | {t1v[0] if t1v else 'n/c (see text)'} | {t1v[1] if t1v else 'n/c (see text)'} |"


def r1(key, s):
    return cell("READ", f"rows|s{s}|{key}", "higher", with_n=False) + f" (matched cells {V('READ', 'rows|s%d|%s|n_cells_matched_pairs' % (s, key), 'ic')})"


w(pfrow("lat head macro-F1 (T0: window argmax; T1-kin: ticks >= 1 matched cells = R-TLAT)", "tactical|heads|lateral|macro_f1", "higher", lambda: (r1("R-TLAT", 0), r1("R-TLAT", 1))))
w(pfrow("lon head macro-F1 (T0: window argmax; T1-kin: ticks >= 1 matched cells = R-TLON)", "tactical|heads|longitudinal|macro_f1", "higher", lambda: (r1("R-TLON", 0), r1("R-TLON", 1))))
w(pfrow("lat head accuracy", "tactical|heads|lateral|accuracy", "higher"))
w(pfrow("lon head accuracy", "tactical|heads|longitudinal|accuracy", "higher"))
w(pfrow("goal head macro-F1 (22 tokens)", "tactical|heads|goal|macro_f1", "higher"))
w(pfrow("5-way collapsed manoeuvre accuracy (plan-derived)", "tactical|plan_derived|maneuver_5way_collapsed|accuracy", "higher"))
w(pfrow("lateral decision accuracy (plan-derived, tool's own)", "tactical|plan_derived|lateral_decision|accuracy", "higher"))
w(pfrow("longitudinal decision accuracy (plan-derived, tool's own)", "tactical|plan_derived|longitudinal_decision|accuracy", "higher"))
w(pfrow("goal-point error (FDE6), m", "tactical|goal_setting|goal_point_error_m", "lower"))
w(pfrow("goal bearing MAE, deg", "tactical|goal_setting|goal_bearing_mae_deg", "lower"))
w("")
w("**Per run, beside, not paired** (the reader's `beside.per_seed`; point values with no interval, so a difference between two of them is a description and not a finding). "
  "Tick 0 is the logged-state rollout step; ticks >= 1 are the model's own executed states. The PREREG's reason for tick 0: it shows whether ORC lost logged-state skill.")
w("")
w("| seed | head | tick-0 F1 ORC | tick-0 F1 ORC-0 | ORC - ORC-0 (tick 0) | goal macro-F1, ticks >= 1: ORC | ORC-0 | goal macro-F1, tick 0: ORC | ORC-0 |")
w("|---|---|---|---|---|---|---|---|---|")
for s in (0, 1):
    for hd in ("lat", "lon"):
        b = f"beside|per_seed|s{s}"
        a_, b_ = f"{b}|tick0_f1|ORC|{hd}", f"{b}|tick0_f1|ORC0|{hd}"
        gm = lambda: (f"{R(b + '|goal_macro_f1|ORC|f1_ge1')} | {R(b + '|goal_macro_f1|ORC0|f1_ge1')} | {R(b + '|goal_macro_f1|ORC|f1_tick0')} | {R(b + '|goal_macro_f1|ORC0|f1_tick0')}") if hd == "lat" else "(same head-independent value as the lat row) | | | "
        w(f"| s{s} | {hd} | {R(a_)} | {R(b_)} | {D('sub', ('READ', a_), ('READ', b_), 's4')} | {gm()} |")
w("")
w(f"(Tick-0 F1 is over {R('beside|per_seed|s0|tick0_f1|ORC|lat_n', 'ic')} lat cells and {R('beside|per_seed|s0|tick0_f1|ORC|lon_n', 'ic')} lon cells, the same for every run. The goal macro-F1 is one value per run, so it is printed on the lat row only.)")
w("")

# ------------------------------------------------------------------------------------------------------------------------ STRATEGIC
w("### 3.4 STRATEGIC")
w("")
w(f"**Status: {R('strategic_family', 'str')}.** It is reported, not paused, and it decides nothing.")
w("")
w("**T0, paired** (S's checkpoint error against the v9 route checkpoints at 30 / 50 / 80 m of arc; `v8f_paired_families.py`, ORC - ORC-0; lower is better; windows with a valid checkpoint label):")
w("")
w("| cell (ORC - ORC-0) | T0 s0 | T0 s1 | T1-kin s0 | T1-kin s1 |")
w("|---|---|---|---|---|")
for arc in ("30", "50", "80"):
    w(pfrow(f"checkpoint position error @{arc} m, m", f"strategic|position_error_m|{arc}", "lower", ("n/c (paired instrument reads eval dumps only)", "n/c (same reason)")))
for arc in ("30", "50", "80"):
    w(pfrow(f"checkpoint heading error @{arc} m, deg", f"strategic|heading_error_deg|{arc}", "lower", ("n/c (same reason)", "n/c (same reason)")))
w("")
w("**T1-kin, per run, unpaired** (the reader's S-POS against a constant-turn-rate-and-velocity (CTRV) rollout from the same pose, ticks >= 1, within each run; delta = S - CTRV position error, so negative means S is nearer the route checkpoint; each is a within-run contrast with its own interval, not an ORC vs ORC-0 contrast):")
w("")
w("| seed | arc | ORC: S - CTRV | ORC-0: S - CTRV |")
w("|---|---|---|---|")
for s in (0, 1):
    for arc in ("30", "50", "80"):
        b = f"beside|per_seed|s{s}|per_run_t1heads"
        c_a = cell("READ", f"{b}|ORC|S-POS vs CTRV (ticks>=1, within this run)|S-POS@{arc}", "lowerS")
        c_b = cell("READ", f"{b}|ORC0|S-POS vs CTRV (ticks>=1, within this run)|S-POS@{arc}", "lowerS")
        w(f"| s{s} | {arc} m | {c_a} | {c_b} |")
w("")
w("**Not computed:** a PAIRED ORC - ORC-0 contrast on STRATEGIC at T1-kin. Reason: the only paired STRATEGIC instrument used here (`v8f_paired_families.py`) needs an eval `run` dump, and the T1 tick dump is a different object; building the paired S-POS from the ticks is a work item (the per-run unpaired values above are what exists). n for that cell: 0 (not attempted), UNVERIFIED either way.")
w("")
w(f"Crawl fraction (ticks >= 1, beside): ORC s0 {R('beside|per_seed|s0|per_run_t1heads|ORC|crawl_fraction_ticks_ge1|lat')}, ORC-0 s0 {R('beside|per_seed|s0|per_run_t1heads|ORC0|crawl_fraction_ticks_ge1|lat')}, "
  f"ORC s1 {R('beside|per_seed|s1|per_run_t1heads|ORC|crawl_fraction_ticks_ge1|lat')}, ORC-0 s1 {R('beside|per_seed|s1|per_run_t1heads|ORC0|crawl_fraction_ticks_ge1|lat')} (the reader stores one value for both heads; unpaired points, no interval).")
w("")
w("### 3.5 What the four families say, descriptively (no interpretation yet)")
w("")
w("* **LONGITUDINAL.** At T0 the three kinematic rows (speed, accel, along-track MAE) are separated-worse for ORC at both seeds. At T1-kin accel MAE is separated-worse at seed 0 only, and speed and along-track are not separated at either seed.")
w("* **LATERAL.** At T1-kin heading, curvature and yaw-rate MAE are separated-better for ORC at both seeds. At T0 curvature and yaw-rate are separated-better at seed 0 only; cross-track is not better at either tier.")
w("* **TACTICAL.** Plan-derived lateral agreement is separated-better at T1-kin at both seeds; at T0 it is separated-better at s0 and separated-worse at s1. The T1-kin head F1 rows are those of section 1.1 (lat n.s. at both seeds, lon n.s. at s0 and separated-better at s1). "
  "At T0 the lat and lon head macro-F1 rows are not separated at either seed, and the goal head macro-F1 is separated-worse at s0 and not separated at s1.")
w("* **DISTANCE KEEPING (a LONGITUDINAL cell group).** The LEAD-VIOL rate (at risk and over all lead windows) is not separated at either tier or seed. At T1-kin ORC's minimum headway is separated-lower than ORC-0's at both seeds and its minimum time gap at seed 0; at T0 the minimum time gap is separated-lower at both seeds and the minimum headway is not separated. The instrument gives these cells no direction label.")
w("* **STRATEGIC.** At T0 no checkpoint position or heading cell (30 / 50 / 80 m) is separated at either seed. No paired T1-kin contrast was computed (stated above); the per-run unpaired S-POS rows are what exists at that tier.")
w("")
w("### 3.6 Per-family gaps: what was not computed, and why")
w("")
w("* **LONGITUDINAL:** the target-speed accuracy bands and the ego-progress ratio are in each arm's eval JSON (`raw.families.longitudinal`) as unpaired levels and were not paired here. The anti-echo condition is undischarged at T0 and flagged at T1-kin (section 3.1). Nothing else missing: speed, accel, along-track and distance keeping are paired at both tiers.")
w("* **LATERAL:** none missing at either tier.")
w("* **TACTICAL:** at T1-kin the paired head accuracy, the goal-head F1 and the 5-way collapsed accuracy were not computed, because `v8f_paired_families.py` reads eval `run` dumps and not the T1 tick dumps; the reader's R-TLAT / R-TLON and the unpaired per-run goal F1 stand in. Anchor selection and selected-vs-executed decisions are UNAVAILABLE by design (the tool's own `unavailable_by_design` block: a trajectory dump carries one path per window and no tactical selection).")
w("* **STRATEGIC:** the paired T1-kin contrast was not computed (section 3.4). The T0 paired cells exist. The eval JSON's own strategic block reads `status: see H-MINI-STRAT-1`, `paused: false`, i.e. it points elsewhere and carries no number.")
w("")
w("These bullets restate the table cells in words. `verify_numbers.py` re-derives each directional sentence from the JSONs (the tags in the table are computed, not typed).")
w("")
w("---")
w("")

# =====================================================================================================================================================
w("## 4. Tier, estimator, and the three variance questions")
w("")
w("| item | value |")
w("|---|---|")
w(f"| tier of R-TLAT / R-TLON / R-ADE | T1-kin: self-action OPEN loop, plain causal bucket, nav dead-reckoned, scene logged, replan every 10 steps; the model conditions on its own executed states and the lead follows its logged future. Not a driving claim |")
w(f"| tier of R-T0 | T0: teacher-forced own-code eval-141 (box 2's standard eval, ages cycle) |")
w(f"| estimator | {R('estimator', 'str')} |")
w(f"| windows / episodes / B | {R('n_windows', 'ic')} / {R('n_episodes', 'i')} / {R('n_boot', 'ic')} |")
w("")
w("**Which of the three variances each interval answers.**")
w("")
w("1. **Another draw of EPISODES** (the models held fixed): answered by every interval here. This is all the bootstrap answers.")
w(f"2. **Another TRAINING run:** answered only by the two seeds (the reader's own wording: \"the training-seed question is answered only by the two seeds (H-ESTIM-SEED-1); a single ORC-0 per seed is the control for '3,000 more steps', not a replicate of the lever's noise\"). "
  f"There is no replicate of ORC at a given seed. Consequences: (a) the R-T0 guard is broken at both seeds with separated intervals, so the sign agrees across the two training runs, but two runs are two runs; "
  f"(b) the R-ADE guard is broken at s0 only, and its s0 size ({R('rows|s0|R-ADE|delta', 's4')} m) is below F = {R('bars|F_ade6_m')} m; (c) the lon F1 row changes sign across seeds ({R('rows|s0|R-TLON|delta', 's4')}, {R('rows|s1|R-TLON|delta', 's4')}). "
  "A separated interval from one training seed is necessary, not sufficient, for a lever effect.")
w("3. **Another INFERENCE run:** the reader's own text is SILENT on whether the T1-kin planner is deterministic (`ORC_READ.json` and `NOTES.md` contain no such statement; UNVERIFIED as a stated property). "
  f"What the reader's controls show instead: K4 reproduces each arm's own T0 eval from an independent T1 rollout's tick 0 to at most {R('controls|K4:ORC_s1|detail|abs_diff_vs_own_T0|pos|1', 'e')} m with 2,399 of 2,399 identical argmax decisions, and K7 rebuilds the executed track from the dumped plans to {R('controls|K7:ORC_s0|detail|x1|max_cl_m', 'f4')} m. "
  "Both are consistent with a deterministic planner (there is no sampling step in this rollout's configuration as the reader records it) but they do not test a second inference run of the same checkpoint. I therefore do not claim the inference question is closed; it is probably not binding here, and it is not measured.")
w("")
w("---")
w("")

# =====================================================================================================================================================
w("## 5. Interpretation (POST-DATA; separate from the outcome; HYPOTHESIS class unless a cell is cited)")
w("")
w("The outcome above was fixed by the committed table before and independent of anything in this section. What follows was written after all the numbers were in, so it can only generate hypotheses.")
w("")
w(f"1. **Lat F1 rose by about {D('mean', ('READ', 'rows|s0|R-TLAT|delta'), ('READ', 'rows|s1|R-TLAT|delta'), 's3')} at both seeds, but not separated.** The matched-cells points are {R('rows|s0|R-TLAT|delta', 's4')} (s0) and {R('rows|s1|R-TLAT|delta', 's4')} (s1): same sign, same size, both above 2F = {R('bars|2F|lat')}, "
  f"and neither interval excludes 0 (s1's lower bound is {R('rows|s1|R-TLAT|lo', 's4')}). Two same-sign, same-size points are what a real small effect would leave, and noise of this size can also produce them; the data do not say which. "
  "Per the committed outcome table it is not an effect, and it does not rescue ORC from HARMS.")
w(f"2. **The T0 head rows do not move where the T1-kin ones do.** At T0 (logged states) the lat head macro-F1 difference is {V('PF0', 'tactical|heads|lateral|macro_f1|delta', 's4')} (s0) and {V('PF1', 'tactical|heads|lateral|macro_f1|delta', 's4')} (s1), neither separated, against {R('rows|s0|R-TLAT|delta', 's4')} / {R('rows|s1|R-TLAT|delta', 's4')} at T1-kin ticks >= 1 (the model's own executed states). HYPOTHESIS: the lever, which perturbs the ego inputs, acts where the model reads its own motion and not on logged states. This is a description of two different protocols (window argmax at T0, matched ticks at T1-kin) and not a paired contrast between them.")
w(f"3. **T0 ADE worsened at both seeds, and the T0 longitudinal kinematics worsened with it.** R-T0 is {R('rows|s0|R-T0|point', 's4')} / {R('rows|s1|R-T0|point', 's4')} m (separated both); speed, accel and along-track MAE are separated-worse at T0 at both seeds (section 3.1). "
  "HYPOTHESIS: T0 scores the model on logged ego states, and a fine-tune on states pulled off the logged trajectory spends some accuracy at the logged states. "
  f"The tick-0 F1 beside agrees in direction: ORC's tick-0 lat F1 is below ORC-0's at both seeds ({D('sub', ('READ', 'beside|per_seed|s0|tick0_f1|ORC|lat'), ('READ', 'beside|per_seed|s0|tick0_f1|ORC0|lat'), 's4')}, {D('sub', ('READ', 'beside|per_seed|s1|tick0_f1|ORC|lat'), ('READ', 'beside|per_seed|s1|tick0_f1|ORC0|lat'), 's4')}) "
  f"and so is the lon F1 ({D('sub', ('READ', 'beside|per_seed|s0|tick0_f1|ORC|lon'), ('READ', 'beside|per_seed|s0|tick0_f1|ORC0|lon'), 's4')}, {D('sub', ('READ', 'beside|per_seed|s1|tick0_f1|ORC|lon'), ('READ', 'beside|per_seed|s1|tick0_f1|ORC0|lon'), 's4')}); "
  "those are point values with no interval, so they are a description and not a finding.")
w("4. **The families split by axis.** At T1-kin the executed LATERAL kinematics improve under the perturbation (heading, curvature, yaw-rate MAE separated-better at both seeds) and so does plan-derived lateral agreement, while the LONGITUDINAL rows do not improve (accel MAE separated-worse at s0) and the T0 longitudinal rows are separated-worse at both seeds. "
  "HYPOTHESIS: the yaw-rate component of the perturbation is learnable and helps the executed lateral track, and the speed / acceleration component costs longitudinal accuracy at the logged states. "
  "⛔ The PREREG states \"one lever with two components: a separated effect is attributed to the perturbation as a whole, never to its lateral or longitudinal half; splitting them is a follow-up arm under its own read.\" "
  "This pattern is the reason such an arm could be worth a PREREG; it is not an attribution, and it is one training seed per arm.")
w(f"5. **The lon F1 row is not a stable signal.** It is negative and not separated at s0 and positive, separated and above 2F at s1; the T0 fresh-age floor F = {R('bars|seed_floor|lon')} that the bar uses is far smaller than that sign flip, which is evidence that the floor is the wrong scale for the T1-kin F1 rows (a T0 floor is not a T1-kin training-run replicate). I do not re-score anything against a different floor; the committed bar stands.")
w("6. **The anti-echo flags** (section 3.1) mean none of the longitudinal contrasts supports a claim that ORC or ORC-0 sets the right speed; they describe a change between two arms.")
w("")
w("---")
w("")
w("## 6. What this does and does not establish")
w("")
w(f"* **Established (MEASURED):** under the committed table, ORC (3,000 steps of fine-tune with 30 % of samples perturbed) is HARMS relative to ORC-0, because R-ADE is separated-worse at s0 and R-T0 exceeds the bar at both seeds. All {LEN('READ', 'controls_PASS')} controls pass.")
w("* **Not established:** that the perturbation helps the T head (no F1 row meets its bar at both seeds); that it hurts the T1-kin planner at both seeds (R-ADE is not separated at s1); that the ADE cost is the lever and not seed noise (one seed per arm; the cost is of the size of F); any attribution to the lateral or the longitudinal half.")
w("* **Committed next (verbatim, not re-decided here):** *\"" + V("READ", "OUTCOME|committed_next", "str") + "\"*. Reading, UNVERIFIED: \"NO EFFECT's arm\" is the on-policy-states (DAgger form) arm named in the NO EFFECT row, and the p in it is a perturbed / on-policy fraction; "
  "no PREREG file for that arm exists in `Project Steering/` at commit `07dcb513` (checked by file name: there is `PREREG_V8F_ORECOVER.md` and no successor or DAgger file), so it needs one before any run.")
w("* **Rule Zero:** HARMS is a waypoint, and the committed next is the lever. Unblocked and 0 GPU: writing that arm's PREREG. Needs the Master Mind or the PI: launching it on box 2 (compute). Not committed, for the Master Mind to weigh: section 5 item 4 points at the follow-up the ORC PREREG itself names (\"splitting them is a follow-up arm under its own read\"), a lateral-only perturbation arm.")
w("")
w("## 7. Where every number comes from")
w("")
w("| artifact | path | sha256 |")
w("|---|---|---|")
w(f"| reader output (primary) | `E:/Projects/TanitAD-artifacts/v8f-orecover-t1/read/ORC_READ.json` | `{R('reader_sha256', 'str') if False else 'a50dbba9fee93828b17b8b1b71335b2db73e249fd7ab9e78c9b4ea4e3577afd6'}` (checked by `verify_numbers.py`) |")
w("| families, ORC - ORC-0, both tiers | `C:/Users/Admin/results_orc/ORC_FAMILIES.json` (driver `orc_families.py`) | in `bank_list.txt` |")
w("| TACTICAL heads + STRATEGIC, T0 | `C:/Users/Admin/results_orc/PAIRFAM_T0_s0.json`, `PAIRFAM_T0_s1.json` (CLI `v8f_paired_families.py` at `07dcb513`) | in `bank_list.txt` |")
w("| T0 evals | `E:/Projects/TanitAD-artifacts/v8f-orecover/M2b_s{0,1}_{orc,orc0}/eval_M2b-<arm>_s<S>.{json,npz}` | `ids` in `ORC_FAMILIES.json` (T0) |")
w("| T1-kin runs | `E:/Projects/TanitAD-artifacts/v8f-orecover-t1/t1v8f_<ARM>_s<S>_m10_causal_heads.json` and `dump_<ARM>_s<S>_m10_causal_heads/` | `runs.json` carries the ticks sha256 |")
w("| number manifest | `numbers_manifest.json`: every number, its source file and key path | |")
w("")
open(OUT_MD, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
SPLIT = len(MAN)

# =====================================================================================================================================================
# register_cell.md: the new lead text of the status cell of H-V8F-ORECOVER-1 (numbers go through V() too; verify_numbers.py checks them as a second document)
def rr(key, s, dp="s4"):
    b = f"rows|s{s}|{key}"
    return f"{V('READ', b + '|delta', dp)} [{V('READ', b + '|lo', dp)}, {V('READ', b + '|hi', dp)}]"


def ff(t, s, key):
    b = f"{t}|s{s}|families_paired_ORC_minus_ORC0|{key}"
    return f"{V('FAM', b + '|mean', 's4')}"


REGTXT = (
    f"**{V('READ', 'OUTCOME|OUTCOME', 'str')} (MEASURED 2026-10-26, both seeds).** ORC (M2b + 3,000 steps with 30 % of samples ego-perturbed) against ORC-0 (the same fine-tune with the perturbation off): the committed reader matched {V('READ', 'OUTCOME|matching_rows_in_table_order|0', 'str')} only. "
    f"R-ADE (T1-kin plain ADE6, m) ORC - ORC-0: s0 {rr('R-ADE', 0)} separated-worse; s1 {rr('R-ADE', 1)} not separated. "
    f"R-T0 (T0 ADE6, m): s0 {rr('R-T0', 0)}, s1 {rr('R-T0', 1)}, both separated and above the {V('READ', 'bars|F_over_2_t0_point_m', 'f4')} m bar (F = {V('READ', 'bars|F_ade6_m', 'f4')} m). "
    f"F1 rows (T1-kin, ticks >= 1, matched cells): lat s0 {rr('R-TLAT', 0)}, s1 {rr('R-TLAT', 1)}, neither separated; lon s0 {rr('R-TLON', 0)} not separated, s1 {rr('R-TLON', 1)} separated and above 2F ({V('READ', 'bars|2F|lon', 'f4')}); "
    f"no F1 row clears 'separated-better and >= 2F' at both seeds. All {LEN('READ', 'controls_PASS')} controls PASS. "
    f"Four families, ORC - ORC-0, never pooled (`RESULT_V8F_ORECOVER.md` section 3): LONGITUDINAL: T0 speed, accel and along-track MAE separated-worse at both seeds (accel {ff('T0', 0, 'lon_accel_mae')} / {ff('T0', 1, 'lon_accel_mae')} m/s^2), T1-kin accel MAE separated-worse at s0 only, and the anti-echo block flags all four T1-kin runs (longitudinal_claim_admissible false), so these are contrasts and not capability claims. "
    f"LATERAL: T1-kin heading, curvature and yaw-rate MAE separated-better at both seeds (yaw-rate {ff('T1', 0, 'lat_yaw_rate_mae_degps')} / {ff('T1', 1, 'lat_yaw_rate_mae_degps')} deg/s); T0 curvature and yaw-rate separated-better at s0 only. "
    f"TACTICAL: plan-derived lateral agreement separated-better at T1-kin at both seeds ({ff('T1', 0, 'tac_lat_acc')} / {ff('T1', 1, 'tac_lat_acc')}); the T0 lat and lon head F1 rows not separated at either seed. "
    "STRATEGIC: T0 checkpoint position and heading error at 30 / 50 / 80 m not separated at either seed; no paired T1-kin STRATEGIC contrast was computed. "
    "Tier: R-TLAT, R-TLON and R-ADE are T1-kin (self-action OPEN loop, plain causal bucket); R-T0 is T0. "
    f"Paired episode-cluster bootstrap, B = {V('READ', 'n_boot', 'ic')}, {V('READ', 'n_episodes', 'i')} episodes; one training seed per arm, so the intervals answer the episode question only and the seed question only through the two seeds (H-ESTIM-SEED-1). "
    f"Committed next, verbatim: *\"{V('READ', 'OUTCOME|committed_next', 'str')}\"*. "
    "Interpretation (POST-DATA, in the RESULT's section 5, HYPOTHESIS class): the lateral rows improve and the longitudinal rows worsen, which is the pattern a split arm could test; the PREREG forbids attributing the effect to either half. "
    "`RESULT_V8F_ORECOVER.md`, `raw/2026-10-26-orecover-result/ORC_READ.json`.")
open(HERE + "/register_cell.md", "w", encoding="utf-8", newline="\n").write(REGTXT + "\n")
json.dump({"sources": SRC, "split": SPLIT, "entries": MAN}, open(OUT_MAN, "w", encoding="utf-8", newline="\n"), indent=1)
print("wrote", OUT_MD, len(L), "lines;", len(MAN), "manifest entries; split", SPLIT)
