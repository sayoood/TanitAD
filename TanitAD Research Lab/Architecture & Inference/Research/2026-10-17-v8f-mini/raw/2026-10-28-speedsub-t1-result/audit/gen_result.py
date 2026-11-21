"""gen_result.py -- builds RESULT_V8F_SPEEDSUB_T1.md, REGISTER_ROW.md and NUMBERS_AUDIT.md for the SPEEDSUB-T1 read.

Every input is read with `git show <tip>:<path>` from the mirror (never a working tree) and checked against a pinned sha256. Every number in the RESULT is formatted once, here, and logged to
NUMBERS_AUDIT.md with its source key path; derived numbers carry their arithmetic. The decision of PREREG_V8F_SPEEDSUB_T1 sections 5 and 6 (with Amendment 1) is recomputed here from the
per-run cells of SPEEDSUB_T1_READ.json, in code written from the PREREG text and not imported from the reader, and the script stops on any mismatch with the reader's own flags, with the
cells' separation flags, or with any control flag. Nothing is written until every check has passed.
run: python -I -X utf8 gen_result.py     (one process, no GPU, no repo working tree)
"""
import hashlib
import json
import re
import subprocess
import sys

sys.dont_write_bytecode = True

OUT = "E:/Projects/TanitAD-artifacts/dagland/speedsub_t1_result/"
TIP = "266b470e4b803f7f8975d1e247172bfee778915a"
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
V8 = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
RDD = V8 + "raw/2026-10-28-speedsub-t1-read/"
BLD = V8 + "raw/2026-10-28-speedsub-t1-build/"
PKG = "raw/2026-10-28-speedsub-t1-result/"
MINUS = "\u2212"
NDASH = "\u2013"
MULT = "\u00d7"

shab = lambda b: hashlib.sha256(b).hexdigest()


def blob(path):
    r = subprocess.run(GIT + ["show", f"{TIP}:{path}"], capture_output=True)
    assert r.returncode == 0, path
    return r.stdout


def git_time(rev):
    r = subprocess.run(GIT + ["log", "-1", "--format=%cI", rev], capture_output=True)
    assert r.returncode == 0, rev
    return r.stdout.decode().strip()


AUD = []          # (section, rendered, source, note)
SEC = [""]


def aud(rendered, src, note=""):
    AUD.append((SEC[0], rendered, src, note))
    return rendered


def sg(x, nd=4):
    s = f"{abs(x):.{nd}f}"
    return (MINUS if (x < 0 and float(s) != 0.0) else "+") + s


def un(x, nd=4):
    return f"{x:.{nd}f}"


def pct(x, nd=1):
    t = f"{abs(100 * x):.{nd}f}"
    return (MINUS if (x < 0 and float(t) != 0.0) else "") + t + " %"


def th(n):
    return f"{n:,}"


def sci(x, nd=3):
    return f"{x:.{nd}e}"


def hhmmss_s(t):
    h, m, s = (int(v) for v in t.split(":"))
    return 3600 * h + 60 * m + s


def mm_ss(sec):
    return f"{sec // 60} min {sec % 60:02d} s"


# ---------------------------------------------------------------------------------------------------------------- sources and pins (all `git show` at the tip, sha256 pinned)
PIN = {
    RDD + "SPEEDSUB_T1_READ.json": "b3b72325e5b8943a90c8b96ecd7c484e1b1be53ab814e4e0aa525c9b530adae0",
    RDD + "SPEEDSUB_T1_BESIDE.json": "cce47a696766d150a6f6c6d44677bb043ea17cbea895e7571fc009b627de6c2c",
    RDD + "pins.json": "59ea9a855630a7ce9bf29188be487777788fd1155bfc1cf972e21711e70dd479",
    RDD + "runs_manifest.json": "9f03cb3b21e46383f1996aabae12f865c601d3ac18e6eafd532439f60c30fb94",
    RDD + "chain.log": "a8a8f8b11554e3d4efd1fdee521861478e4d93c16b2a8d177ba074af9380831f",
    RDD + "chain_speedsub_t1.out": "7a7fe043ca09f5d8a18d0dcbfc1d3b96a9a72268f3639f11368b914afce6e0ac",
    RDD + "preflight_CTRL_s0.json": "116c9e5509b55a12ea855f419ef1ab25e8bdd41b17291df85691f308d6a7dd7d",
    RDD + "smoke_check_CTRL_s0.json": "d2b936e1ebe03b60bad2afa48d973940df7355517744e94c5f09a64cfb5a36aa",
    BLD + "read/speedsub_t1_read.py": "9d3c27926ec31768684dd52dc3bae586aa52984eefe8eb2763fb69234af0b98f",
    BLD + "evidence/fit_c.json": "9aaa5edde167b767d1bbf586f2e2909f1f21b40fc17e53047123101d60fd499a",
    BLD + "NOTES.md": "e90ebbaf381edf8f7c5112f017cab1ed9381b01bc627dd8312bc51475fe6cb5b",
    BLD + "run_speedsub_t1.sh": None,
    "Project Steering/PREREG_V8F_SPEEDSUB_T1.md": "69187967e82bf64dac85bf9ae81810b6f8aac7161cd2558d719d45cdf61fd861",
    V8 + "raw/2026-10-28-speedsub-t1-prereg/rep_s0_diag/REP_S0_SPEED_BIAS.md": "211601de6592fee86b1e4f4ab883d16acc12bdba47028b7bf008ac7eb49a1d64",
    V8 + "RESULT_V8F_DAG_ORC.md": "5ce374218288bf2a02edfd7965239e84c21dc8159ba23ac5e7e3aa8206059a6c",
    V8 + "RESULT_V8F_SPEEDSUB.md": "4bdc87a27a13bb34ea161cf6ea05e43edca2b4960fc7c2fe9e645a09c8965f7c",
}
RAW = {}
for p, h in PIN.items():
    RAW[p] = blob(p)
    if h is not None:
        assert shab(RAW[p]) == h, (p, shab(RAW[p]))
    else:
        PIN[p] = shab(RAW[p])
for p, b in RAW.items():
    if p.endswith((".md", ".py", ".sh")):
        assert b"\r" not in b, ("CR in a pinned source", p)
READ = json.loads(RAW[RDD + "SPEEDSUB_T1_READ.json"].decode("utf-8"))
BES = json.loads(RAW[RDD + "SPEEDSUB_T1_BESIDE.json"].decode("utf-8"))
PINS = json.loads(RAW[RDD + "pins.json"].decode("utf-8"))
MAN = json.loads(RAW[RDD + "runs_manifest.json"].decode("utf-8"))
PRE = RAW["Project Steering/PREREG_V8F_SPEEDSUB_T1.md"].decode("utf-8")
PRE_SHA = shab(PRE.encode("utf-8"))
FITC = json.loads(RAW[BLD + "evidence/fit_c.json"].decode("utf-8"))
CHAIN = RAW[RDD + "chain.log"].decode("utf-8").replace("\r\n", "\n")
SMOKE = json.loads(RAW[RDD + "smoke_check_CTRL_s0.json"].decode("utf-8"))
PREF = json.loads(RAW[RDD + "preflight_CTRL_s0.json"].decode("utf-8"))
REPD = RAW[V8 + "raw/2026-10-28-speedsub-t1-prereg/rep_s0_diag/REP_S0_SPEED_BIAS.md"].decode("utf-8")
ORC = RAW[V8 + "RESULT_V8F_DAG_ORC.md"].decode("utf-8")
RUNSH = RAW[BLD + "run_speedsub_t1.sh"].decode("utf-8")
READER_SHA = PIN[BLD + "read/speedsub_t1_read.py"]
READ_SHA = PIN[RDD + "SPEEDSUB_T1_READ.json"]
BES_SHA = PIN[RDD + "SPEEDSUB_T1_BESIDE.json"]

RUNS = ("CTRL_s0", "CTRL_s1", "REP_s0", "REP_s1")
CTRL = RUNS[:2]
RUNH = {r: r.replace("_", " ") for r in RUNS}
TAB = READ["tables"]
assert set(TAB) == {"egoc-identity", "egoc-P10", "identity-P10", "egoshufc-identity", "identity-identity"}
LABELS_ALL = list(TAB["egoc-identity"]["CTRL_s0"])
assert len(LABELS_ALL) == 21 and LABELS_ALL[0] == "ADE6" and LABELS_ALL[-2:] == ["VIOL-ANY", "VIOL-ANY-OL"]
LABELS19 = LABELS_ALL[:19]                       # Read A's 19 rows (the G-WORSE / G-NEW domain); VIOL-ANY rows are G-VIOL's
for c in TAB:
    for r in RUNS:
        assert list(TAB[c][r]) == LABELS_ALL, (c, r)
DIRECTION = {l: "lower" for l in LABELS_ALL}
DIRECTION.update({"TAC-LAT": "higher", "TAC-LON": "higher"})
FF = READ["four_families"]
FAMILY = {"ADE6": "PRIMARY", "M1 crawl share": "crawl"}
for fam in ("LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC", "RULES"):
    for l in FF[fam]["rows"]:
        FAMILY[l] = fam
assert sorted(FAMILY) == sorted(LABELS_ALL), set(LABELS_ALL) ^ set(FAMILY)
for l in LABELS_ALL:                              # the family blocks and the flat tables are the same cells
    fam = FAMILY[l]
    if fam in ("PRIMARY", "crawl"):
        continue
    for contrast in ("egoc-identity", "egoc-P10"):
        for r in RUNS:
            a, b = FF[fam]["rows"][l][contrast][r], TAB[contrast][r][l]
            if isinstance(a, str):
                assert a == "NOT COMPUTED" and b.get("status") == "NOT COMPUTED", (fam, l, contrast, r)
            else:
                assert all(a[k] == b[k] for k in ("delta", "lo", "hi", "separated")), (fam, l, contrast, r)

# ---------------------------------------------------------------------------------------------------------------- cells: computed / separated / better / worse (coded from the PREREG text)
F_M = 0.1207
BAR = F_M / 2.0
assert READ["bar"] == 0.06035 and abs(BAR - 0.06035) < 1e-12 and "F/2 = 0.06035" in PRE.replace("**", "").replace("`", "")
GUARD_RUNS_MIN = 3


def computed(e):
    return isinstance(e, dict) and e.get("status") != "NOT COMPUTED" and e.get("delta") is not None


def worsening(l, e):
    d = float(e["delta"])
    return d if DIRECTION[l] == "lower" else -d


def sepw(l, e):
    return bool(computed(e) and e["separated"] and worsening(l, e) > 0)


def sepb(l, e):
    return bool(computed(e) and e["separated"] and worsening(l, e) < 0)


def ci_excludes0(e):
    return bool(e["lo"] > 0 or e["hi"] < 0)


def is_edge(e):
    return bool(computed(e) and (e["lo"] == 0.0 or e["hi"] == 0.0))


EDGE = []        # (contrast, run, label) where a rounded bound is exactly 0.0000 (the flag is read from the estimator, not from the rounded bound)
MISMATCH = []    # the subset where the flag disagrees with the rounded interval
for c in ("egoc-identity", "egoshufc-identity", "egoc-P10", "identity-P10"):
    for r in RUNS:
        for l in LABELS_ALL:
            e = TAB[c][r][l]
            if not computed(e):
                continue
            if bool(e["separated"]) != ci_excludes0(e):
                assert is_edge(e), ("separation flag disagrees with the CI and no bound rounds to 0", c, r, l, e)
                MISMATCH.append((c, r, l))
            if is_edge(e):
                EDGE.append((c, r, l))
for bsc in BES["egol_minus_identity_l"].values():
    for l, e in bsc.items():
        if computed(e):
            assert bool(e["separated"]) == ci_excludes0(e) or is_edge(e), ("beside", l, e)

# ---------------------------------------------------------------------------------------------------------------- the decision, recomputed (PREREG sections 5 and 6, Amendment 1)
EI, EP, IP, SI = TAB["egoc-identity"], TAB["egoc-P10"], TAB["identity-P10"], TAB["egoshufc-identity"]
CT = READ["controls"]
N_FLAGS = len(CT)
assert N_FLAGS == 158 and all(v is True for v in CT.values()) and READ["controls_PASS"] is True and READ["flags"]["controls_ok"] is True
# K-ZERO is exact; K-REG is the deliberate regression
for r in RUNS:
    for l, e in TAB["identity-identity"][r].items():
        if computed(e):
            assert e["delta"] == 0.0 and e["lo"] == 0.0 and e["hi"] == 0.0, (r, l)
assert all(sepw("ADE6", SI[r]["ADE6"]) for r in RUNS)

d_ade = {r: EI[r]["ADE6"] for r in RUNS}
HIT = {r: bool(d_ade[r]["hi"] < 0 and d_ade[r]["delta"] <= -BAR) for r in RUNS}
HIT_CI_ONLY = {r: bool(d_ade[r]["hi"] < 0) for r in RUNS}
HIT_BAR_ONLY = {r: bool(d_ade[r]["delta"] <= -BAR) for r in RUNS}
N_HIT = sum(HIT.values())
G_ADE_RUNS = [r for r in RUNS if sepw("ADE6", EI[r]["ADE6"])]
GW = {l: [r for r in RUNS if sepw(l, EI[r][l])] for l in LABELS19}
GW = {l: v for l, v in GW.items() if v}
G_WORSE = any(len(v) >= GUARD_RUNS_MIN for v in GW.values())
GN_BOTH = [l for l in LABELS19 if all(sepw(l, EP[r][l]) for r in CTRL)]
GN_ID = {l: [sepw(l, IP[r][l]) for r in CTRL] for l in GN_BOTH}
GN_CODED = [l for l in GN_BOTH if not all(GN_ID[l])]            # the reader's rule: identity-P10 not separated-worse at >= 1 CTRL seed
GN_AMEND = [l for l in GN_BOTH if not any(GN_ID[l])]           # Amendment 1 item 3: identity-P10 not separated-worse at each CTRL seed
assert set(GN_AMEND) <= set(GN_CODED)
G_NEW_CODED, G_NEW_AMEND = bool(GN_CODED), bool(GN_AMEND)
GV_RUNS = [r for r in RUNS if sepw("VIOL-ANY", EI[r]["VIOL-ANY"])]
G_VIOL = len(GV_RUNS) >= GUARD_RUNS_MIN
GUARDS = {"G-ADE": bool(G_ADE_RUNS), "G-WORSE": G_WORSE, "G-NEW": G_NEW_CODED, "G-VIOL": G_VIOL}


def decide(controls_ok, runs_present, hits, guards):
    if not controls_ok or not runs_present:
        return "REFUSED", None
    if any(guards.values()):
        return "HARMS", None
    n = sum(hits.values())
    if n == len(RUNS):
        return "PASS", None
    if all(hits[r] for r in CTRL):
        return "PARTIAL", None
    return "NULL", ("NULL-FLAT" if n == 0 else "NULL-ONE")


OUTCOME, LABEL = decide(True, True, HIT, GUARDS)
OUT_NO_GVIOL = decide(True, True, HIT, {k: v for k, v in GUARDS.items() if k != "G-VIOL"})
OUT_AMEND = decide(True, True, HIT, {**GUARDS, "G-NEW": G_NEW_AMEND})
# --- the reader's own record must agree with every recomputed piece
assert OUTCOME == READ["OUTCOME"] == "NULL" and LABEL == READ["label"] == "NULL-ONE", (OUTCOME, LABEL, READ["OUTCOME"], READ["label"])
assert OUT_NO_GVIOL == OUT_AMEND == (OUTCOME, LABEL)
assert HIT == READ["hits"] == READ["flags"]["hits"] and N_HIT == READ["flags"]["n_hit"] == 1 and READ["flags"]["hits_at_both_ctrl"] is False
assert READ["bar"] == 0.06035 == round(BAR, 5), (READ["bar"], BAR)
assert READ["flags"]["guards_fired"] == [] and not any(GUARDS.values())
for r in RUNS:
    for k in ("delta", "lo", "hi", "separated"):
        assert READ["d_ADE6_egoc_minus_identity"][r][k] == d_ade[r][k], (r, k)
RG = READ["guards"]
assert RG["G-ADE"]["fires"] == bool(G_ADE_RUNS) and RG["G-ADE"]["runs"] == G_ADE_RUNS
assert RG["G-WORSE"]["fires"] == G_WORSE and RG["G-WORSE"]["rows_and_runs"] == GW
assert RG["G-NEW"]["fires"] == G_NEW_CODED and RG["G-NEW"]["rows"] == GN_CODED and RG["G-NEW"]["fires_if_identity_clear_at_both"] == GN_AMEND
assert sorted(RG["G-NEW"]["detail"]) == sorted(GN_BOTH) and all(RG["G-NEW"]["detail"][l]["identity_P10_sep_worse"] == GN_ID[l] for l in GN_BOTH)
assert RG["G-VIOL"]["fires"] == G_VIOL and RG["G-VIOL"]["runs"] == GV_RUNS and RG["G-VIOL"]["active"] is True
assert (OUTCOME, LABEL) == ("NULL", "NULL-ONE") and READ["committed_next"] == "next: section 8 (a), the history-aware causal read-out"
assert READ["claim_row"] == "H-V8F-SPEEDSUB-T1-1"
# the two halves of the hit rule name the same runs (so neither the CI rule nor the bar alone decides the word)
assert HIT_CI_ONLY == HIT_BAR_ONLY == HIT
# the prereg text carries the table rows this code implements
PRE_FLAT = PRE.replace("**", "").replace("`", "")
for need in ("PASS | no guard; a hit at all four runs", "PARTIAL | no guard; hits at both CTRL seeds, not at all four runs", "NULL | anything else | next: section 8 (a), the history-aware read-out",
             "separated-better (CI upper end < 0) and d_X ≤ −0.06035", "G-VIOL: VIOL-ANY is not separated-worse at ≥ 3 of the 4 runs.", "G-ADE: egoc − identity ADE6 is not separated-worse at any run.",
             "no Read-A row (ADE6 included) has egoc − identity separated-worse at ≥ 3 of the 4 runs"):
    assert need in PRE_FLAT, need

# --- the floor rule (section 6.2), from the means in the cells (cross-check) and from the reader's points (the rule applied)
FR = READ["floor_rule_H_V8F_SPEEDSUB_T1_2"]
FLOOR_ROWS = ("ADE6", "yaw-rate MAE", "curvature MAE")
FLOOR = {}
for s, key in (("0", "s0 (decisive, beside)"), ("1", "s1 (reported)")):
    rec = FR[key]
    gid_r = {r: float(rec["g_id"][r]) for r in FLOOR_ROWS}
    gc_r = {r: float(rec["g_c"][r]) for r in FLOOR_ROWS}
    gid_m = {r: IP[f"REP_s{s}"][r]["mean_arm"] - IP[f"CTRL_s{s}"][r]["mean_arm"] for r in FLOOR_ROWS}
    gc_m = {r: EP[f"REP_s{s}"][r]["mean_arm"] - EP[f"CTRL_s{s}"][r]["mean_arm"] for r in FLOOR_ROWS}
    per_r = {r: bool(gid_r[r] > 0.0 and gc_r[r] <= 0.5 * gid_r[r]) for r in FLOOR_ROWS}
    per_m = {r: bool(gid_m[r] > 0.0 and gc_m[r] <= 0.5 * gid_m[r]) for r in FLOOR_ROWS}
    assert per_r == per_m == rec["per_row"], (s, per_r, per_m, rec["per_row"])
    verdict = "COLLAPSE" if all(per_r.values()) else "NOT COLLAPSE"
    assert verdict == rec["verdict"] == "NOT COLLAPSE" and rec["consequence"] == "the floors stay as DAG-ORC's"
    for r in FLOOR_ROWS:       # the reader's paired point and the difference of the cells' means agree to the window-mask (ADE6: 4 dp)
        tol = 5e-5 if r == "ADE6" else 0.015 * max(abs(gid_r[r]), 1e-9) + 1e-4
        assert abs(gid_r[r] - gid_m[r]) <= tol and abs(gc_r[r] - gc_m[r]) <= tol, (s, r, gid_r[r], gid_m[r], gc_r[r], gc_m[r])
    FLOOR[s] = {"gid": gid_r, "gc": gc_r, "gid_m": gid_m, "gc_m": gc_m, "per": per_r, "verdict": verdict}
# the identity that ties g_c to g_id and the two substitution effects (ADE6, s0): g_c = g_id + d(REP) - d(CTRL)
for s in ("0", "1"):
    lhs = FLOOR[s]["gc"]["ADE6"]
    rhs = FLOOR[s]["gid"]["ADE6"] + EI[f"REP_s{s}"]["ADE6"]["delta"] - EI[f"CTRL_s{s}"]["ADE6"]["delta"]
    assert abs(lhs - rhs) <= 2e-4, (s, lhs, rhs)

# --- beside lane
BL = BES["egol_minus_identity_l"]
assert set(BL) == {"CTRL_s0", "CTRL_s1"} and BES["role"] == "beside; decides nothing; never a capability row"
assert all(not BL[r]["ADE6"]["separated"] for r in BL)

# ---------------------------------------------------------------------------------------------------------------- the chain log
RO = {}      # tag -> (start, done, seconds)
for m in re.finditer(r"^(\d\d:\d\d:\d\d) START (\S+)$", CHAIN, re.M):
    RO[m[2]] = [m[1], None, None]
for m in re.finditer(r"^(\d\d:\d\d:\d\d) DONE\s+(\S+) in (\d+) s$", CHAIN, re.M):
    RO[m[2]][1], RO[m[2]][2] = m[1], int(m[3])
assert len(RO) == 18 and all(v[1] and v[2] is not None for v in RO.values())
DECISIVE = [f"SST1_{r}_{m}" for r in RUNS for m in ("identity", "egoc", "egoshufc")]
SMOKE_T = ["SST1_SMOKE_CTRL_s0_identity", "SST1_SMOKE_CTRL_s0_egoc"]
BESIDE_T = [f"SST1_{r}_{m}" for r in CTRL for m in ("identity_l", "egol")]
assert set(RO) == set(DECISIVE) | set(SMOKE_T) | set(BESIDE_T)
for t in RO:
    assert hhmmss_s(RO[t][1]) - hhmmss_s(RO[t][0]) in range(RO[t][2] - 2, RO[t][2] + 3), (t, RO[t])        # the 'in N s' field equals the stamp difference to 2 s
STG = {m[2]: m[1] for m in re.finditer(r"^(\d\d:\d\d:\d\d) ZZSST1-STAGE-DONE (\w+)$", CHAIN, re.M)}
assert list(STG) == ["fit", "pin", "static", "smoke", "ctrl0", "preflight", "ctrl1", "rep0", "rep1", "read", "beside"], list(STG)
T_START = re.match(r"(\d\d:\d\d:\d\d) run_speedsub_t1 start", CHAIN)[1]
assert "ZZSST1-CHAIN-DONE" in CHAIN and "fit: reusing F:/TanitAD-scratch/speedsub_t1/fit (fit_c.json FITTED)" in CHAIN
assert "ZZSPEEDSUB-T1-SMOKE-OK" in CHAIN and "ZZSPEEDSUB-T1-PREFLIGHT-OK" in CHAIN
T_END = STG["beside"]
T_MAN = re.search(r"^(\d\d:\d\d:\d\d) ZZSST1-STAGE-DONE rep1$", CHAIN, re.M)[1]
assert "manifest written" in CHAIN.split("ZZSST1-STAGE-DONE rep1")[1].split("\n")[1]
assert "OMP_NUM_THREADS=\"${OMP_NUM_THREADS:-8}\" MKL_NUM_THREADS=\"${MKL_NUM_THREADS:-8}\"" in RUNSH and "CUDA_VISIBLE_DEVICES=-1" in RUNSH
WALL_READER = READ["provenance"]["wall_s"]
assert WALL_READER == 454.4 and READ["provenance"]["reader_sha256"] == READER_SHA and READ["provenance"]["prereg_sha256"] == PRE_SHA and READ["provenance"]["pins_sha256"] == PIN[RDD + "pins.json"]
assert READ["provenance"]["manifest_sha256"] == PIN[RDD + "runs_manifest.json"]
# pins: checkpoints and tools
CK = {r: PINS["ckpts"][r]["sha256"] for r in RUNS}
assert [CK[r][:8] for r in RUNS] == ["67265ec1", "90b01474", "b03fd6ef", "77853e56"]
assert PINS["readout_c"]["sha256"].startswith("799f65ed") and PINS["readout_banked"]["sha256"].startswith("a205609a") and PINS["tool_sha256"].startswith("61e140ed")
assert PINS["readout_c"]["fit_c_json_sha256"] == PIN[BLD + "evidence/fit_c.json"] and PINS["perm_sha256"].startswith("64785f8a") and PINS["reader_sha256"] == READER_SHA
assert FITC["controls_pass"] is True and FITC["controls"]["C-FIT"]["pass"] is True
T_PROMO, T_AMEND, T_READCOMMIT = git_time("842ec2ac"), git_time("4cec09e7"), git_time(TIP)
for t_ in (T_PROMO, T_AMEND, T_READCOMMIT):
    assert t_.startswith("2026-10-28T") and t_.endswith("+01:00"), t_
hm = lambda iso: iso[11:19]
assert hhmmss_s(hm(T_AMEND)) < hhmmss_s(RO[SMOKE_T[0]][0]) and hhmmss_s(hm(T_PROMO)) < hhmmss_s(hm(T_AMEND))        # Amendment 1 landed before the first rollout

# ---------------------------------------------------------------------------------------------------------------- parsed inputs from the repo documents
REPN = {}
for cell_, lab in (("lon_v_target_ms", "lon_v_target_ms"), ("v_a_ms", "v_a_ms"), ("v_end_ms", "v_end_ms"), ("v_lo_ms", "v_lo_ms"), ("v_hi_ms", "v_hi_ms")):
    m = re.search(r"^\| " + lab + r" \| ([\d.]+) \| \*\*([\u2212-][\d.]+) \[([\u2212-][\d.]+), ([\u2212-][\d.]+)\]\*\* \|", REPD, re.M)
    assert m, lab
    REPN[lab] = (float(m[1]), float(m[2].replace(MINUS, "-")), float(m[3].replace(MINUS, "-")), float(m[4].replace(MINUS, "-")))
REP_MIN, REP_MAX = min(abs(v[1]) for v in REPN.values()), max(abs(v[1]) for v in REPN.values())
assert (un(REP_MIN, 4), un(REP_MAX, 4)) == ("0.7315", "1.1700")
m = re.search(r"^\| speed bias \(m/s\) \| ([\u2212-][\d.]+) \| \*\*([\u2212-][\d.]+)\*\*", REPD, re.M)
SPEEDBIAS = (float(m[1].replace(MINUS, "-")), float(m[2].replace(MINUS, "-")))
m = re.search(r"^\| along-track final bias \(m\) \| ([\u2212-][\d.]+) \| \*\*([\u2212-][\d.]+)\*\*", REPD, re.M)
FINALBIAS = (float(m[1].replace(MINUS, "-")), float(m[2].replace(MINUS, "-")))
assert SPEEDBIAS == (-0.1762, -0.544) and FINALBIAS == (-0.7531, -3.0806)
ORC_A = ORC.split("### (a) A")[1].split("### (b)")[0]
ORC_B = ORC.split("### (b) The share")[1].split("### (c)")[0]
CRAWL = ("yaw-rate MAE", "curvature MAE", "L2 curvature MAE (GT>=5.2)", "M1 crawl share")
A_EFF, A_SHARE = {}, {}
for l in CRAWL:
    m = re.search(r"^\| " + re.escape(l) + r" \| ([\u2212+][\d.]+) \[([\u2212+][\d.]+), ([\u2212+][\d.]+)\] B \| ([\u2212+][\d.]+) \[([\u2212+][\d.]+), ([\u2212+][\d.]+)\] B \|$", ORC_A, re.M)
    assert m, l
    A_EFF[l] = {"s0": [float(m[i].replace(MINUS, "-")) for i in (1, 2, 3)], "s1": [float(m[i].replace(MINUS, "-")) for i in (4, 5, 6)]}
    for s in ("s0", "s1"):
        m2 = re.search(r"^\| " + s + r" \| " + re.escape(l) + r" \| [\u2212+][\d.]+ \| [\u2212+][\d.]+ \| ([\d.]+) % \| ([\d.]+) % \|$", ORC_B, re.M)
        assert m2, (s, l)
        A_SHARE[(s, l)] = float(m2[1])
assert A_EFF["yaw-rate MAE"]["s0"][0] == -2.7748 and A_EFF["M1 crawl share"]["s1"][0] == -0.0166 and A_SHARE[("s0", "yaw-rate MAE")] == 79.9 and A_SHARE[("s1", "M1 crawl share")] == 27.6
SPEEDSUB_QUOTE = "−0.1290 [−0.2107, −0.0548] (s0) and −0.0686 [−0.1151, −0.0232] (s1)"
assert SPEEDSUB_QUOTE in PRE.replace("`", "").replace("**", "")
assert "RESULT_V8F_DAG_ORC.md" in PRE and "(a) The history-aware causal read-out" in PRE.replace("**", "") and "ESTIMATED under\n  1 CPU-h" in PRE.replace("**", "")
SPEEDSUB_RESULT = RAW[V8 + "RESULT_V8F_SPEEDSUB.md"].decode("utf-8")
assert SPEEDSUB_RESULT.split("\n")[0].startswith("# RESULT_V8F_SPEEDSUB: ") and "Outcome **PASS**" in SPEEDSUB_RESULT.split("\n")[0]
assert "−0.1290 [−0.2107, −0.0548]" in SPEEDSUB_RESULT and "−0.0686 [−0.1151, −0.0232]" in SPEEDSUB_RESULT
MN_A, MX_A = min(A_SHARE.values()), max(A_SHARE.values())
assert (MN_A, MX_A) == (27.6, 79.9)


# ---------------------------------------------------------------------------------------------------------------- cell renderers (format once, log once)
def mark(l, e):
    return "B" if sepb(l, e) else ("W" if sepw(l, e) else NDASH)


def civ(e):
    return f"{sg(e['delta'])} [{sg(e['lo'])}, {sg(e['hi'])}]"


def cell(c, r, l, bold=False):
    e = TAB[c][r][l]
    if not computed(e):
        return "NOT COMPUTED"
    s = civ(e) + " " + mark(l, e)
    aud(civ(e), f"SPEEDSUB_T1_READ.json tables.{c}.{r}.{l}.{{delta,lo,hi}}", f"flag separated = {e['separated']}")
    if (c, r, l) in EDGE:
        s += " \u2020"
    return f"**{s}**" if bold else s


def ncell(c, r, l):
    e = TAB[c][r][l]
    return e["n"] if computed(e) else "-"


# ---------------------------------------------------------------------------------------------------------------- document
L = []


def P(*xs):
    L.extend(xs)


def hdr(t, sec):
    SEC[0] = sec
    P("", t, "")


def yn(b):
    return "yes" if b else "no"


def rows_table(contrast, labels, heading="row"):
    P(f"| {heading} | family | " + " | ".join(RUNH[r] for r in RUNS) + " |", "|---|---|" + "---|" * 4)
    for l in labels:
        P(f"| {l} | {FAMILY[l]} | " + " | ".join(cell(contrast, r, l) for r in RUNS) + " |")


SEC[0] = "header"
P("# RESULT_V8F_SPEEDSUB_T1: does a causal ego read-out in O's five speed cells, at every replan tick, move DAG-OACC at T1-kin? Outcome **NULL** (label NULL-ONE: one hit of four runs, at the slow-T draw REP s0)")
P("")
P("Claim row `H-V8F-SPEEDSUB-T1-1`; the floor rule is its beside hypothesis `H-V8F-SPEEDSUB-T1-2`. Pre-registration `Project Steering/PREREG_V8F_SPEEDSUB_T1.md`, sha256 `" + PRE_SHA[:8] + "…` at tip `266b470e`:"
  f" promoted at `842ec2ac` ({aud(hm(T_PROMO), 'git log -1 --format=%cI 842ec2ac', T_PROMO)}), before any T1-kin substitution rollout and before its read-out was fitted; **Amendment 1** (`A1-2026-10-28`: guard G-VIOL and four settled readings),"
  f" landed at `4cec09e7` ({aud(hm(T_AMEND), 'git log -1 --format=%cI 4cec09e7', T_AMEND)}), before the smoke ({aud(RO[SMOKE_T[0]][0], 'chain.log: START SST1_SMOKE_CTRL_s0_identity')}) and before any rollout of this experiment."
  f" The instrument was built against that text (`raw/2026-10-28-speedsub-t1-build/`) and the read-out `mlp_ego_c` fitted once (`readout_c.pt`, sha256 `799f65ed…`). Chain run 2026-10-28 {aud(T_START, 'chain.log first line')} to {aud(T_END, 'chain.log ZZSST1-STAGE-DONE beside')}"
  f" ({mm_ss(hhmmss_s(T_END) - hhmmss_s(T_START))}) on the dev box (CPU, `CUDA_VISIBLE_DEVICES=-1`, OMP / MKL 8): smoke, twelve decisive rollouts, the reader, and four beside rollouts. The pre-registered reader `speedsub_t1_read.py` (sha256 `{READER_SHA[:8]}…`)"
  f" ran after the last decisive rollout (stage `read` done {aud(STG['read'], 'chain.log ZZSST1-STAGE-DONE read')}; the reader's own wall {aud(un(WALL_READER, 1) + ' s', 'SPEEDSUB_T1_READ.json provenance.wall_s')}), printed `ZZSPEEDSUB-T1-READ-DONE NULL NULL-ONE`,"
  f" and **all {aud(str(N_FLAGS), 'SPEEDSUB_T1_READ.json len(controls)', 'every value true')} control flags PASS**. Output `raw/2026-10-28-speedsub-t1-read/SPEEDSUB_T1_READ.json` (sha256 `{READ_SHA[:8]}…`), committed at `266b470e` ({aud(hm(T_READCOMMIT), 'git log -1 --format=%cI 266b470e', T_READCOMMIT)}),"
  " beside it `SPEEDSUB_T1_BESIDE.json` (sha256 `" + BES_SHA[:8] + "…`)."
  " **The outcome word is not copied:** section 1 recomputes the hits, the four guards, the PREREG section 6 table and the section 6.2 floor rule from the per-run cells, in code written from the PREREG text, and `gen_result.py` stops on any mismatch with the reader."
  " Every number below is MEASURED (ours; artifact path in `raw/2026-10-28-speedsub-t1-result/audit/NUMBERS_AUDIT.md`) unless marked INHERITED (another document, not re-measured here), DERIVED (arithmetic shown in the audit), ESTIMATED or PUBLISHED.")
aud("2026-10-28", "run date: chain.log stamps (HH:MM:SS), git commit dates")
aud("A1-2026-10-28", "PREREG_V8F_SPEEDSUB_T1.md Amendment 1 id")
P("")
P("**Summary** (each value is in the section named; evidence class and tier as stated there).")
P("")
P("* **Outcome NULL, label NULL-ONE (section 1).** A hit needs `egoc − identity` ADE6 separated-better and at or below −" + un(BAR, 5) + " m; PASS needs it at all four runs; the read has " + f"{N_HIT} hit of 4 runs (REP s0, {civ(d_ade['REP_s0'])} m)." + " No control fails and no guard fires.")
P("* **The healthy draws do not move (section 5).** CTRL s0 " + civ(d_ade["CTRL_s0"]) + ", CTRL s1 " + civ(d_ade["CTRL_s1"]) + ", REP s1 " + civ(d_ade["REP_s1"]) + " m: all intervals include 0; M1 crawl share moves at no run.")
P("* **The hit is a repair of a slow-T draw (section 4).** At REP s0 " + str(sum(sepb(l, EI["REP_s0"][l]) for l in LABELS_ALL)) + " rows separate better and `VIOL-ANY` separates worse (" + sg(EI["REP_s0"]["VIOL-ANY"]["delta"]) + "); floor rule NOT COLLAPSE (the ADE6 gap to CTRL s0 " + un(FLOOR["0"]["gid"]["ADE6"]) + " → " + un(FLOOR["0"]["gc"]["ADE6"]) + " m).")
P("* **The logged-lane read-out is flat too (section 6, beside).** " + sg(BL["CTRL_s0"]["ADE6"]["delta"]) + " and " + sg(BL["CTRL_s1"]["ADE6"]["delta"]) + " m at CTRL s0 and s1, not separated.")
P("* **Next (section 7).** Committed: PREREG section 8 (a). Master Mind's ranking (ESTIMATED / INFERENCE): (a) stays registered and is not run now; the largest measured crawl lever is DAG-ORC's perturbation, and its registered route (LAUNCH-INTENT, box 2) waits on the PI's go.")
P("")
TIER = READ["tier"]
EST = READ["estimator"]
assert "B=2000, seed 0" in EST and "another draw of EPISODES" in EST and "Never 'closed loop', never T2." in TIER and TIER.startswith("T1-kin (self-action OPEN loop;")
P("* **Tier.** READ.json `tier`: `" + TIER + "` So every decisive row is **T1-kin, a self-action open loop** (the executed plan feeds the next tick; the lead follows its logged future and the cameras are logged), never closed loop and never T2."
  " The `gt`-style ceiling of SPEEDSUB (T0) has no counterpart here: no row of this read is T0. The one T0 item in this document is the INHERITED speed-cell diagnostic of REP s0 (section 4).")
P("* **Estimator.** READ.json `estimator`: `" + EST + "` It answers *\"would another draw of EPISODES say this?\"* (141 episodes / 2,399 windows per run, paired against the same run's `identity` rollout, or against the P10 floor). The four runs"
  " (CTRL s0 / s1 = DAG-OACC at trainer seeds 0 / 1; REP s0 / s1 = DAG-OACC's argv at trainer seeds 10 / 11) are four training runs of one recipe and carry *\"would another TRAINING RUN say this?\"* at n = 4 (`H-ESTIM-SEED-1`)."
  " The read-out is fitted **once**: *\"would another READ-OUT FIT say this?\"* is not answered. The planner is deterministic, so there is no inference-seed term.")
aud("141 episodes / 2,399 windows; n = 4", "PREREG section 3; SPEEDSUB_T1_READ.json tables.*.*.ADE6.n = 2399/141; four runs", "")
aud("B=2000, seed 0; B = 2,000", "SPEEDSUB_T1_READ.json estimator (taniteval.ci.paired_episode_cluster_bootstrap, B=2000, seed 0); PREREG section 3")
aud("10 / 11", "PREREG section 1: REP runs are DAG-OACC's argv at trainer seed 10 / 11 (pins.json banked_argv REP_s0, REP_s1)")
P("* **Base and arms.** X = DAG-OACC, four checkpoints, sha256 prefixes " + ", ".join(f"{RUNH[r]} `{CK[r][:8]}…`" for r in RUNS) + " (pins.json `ckpts`; K-SHA true at all 12 rollouts). No model is trained. Per run: `identity` (X's own cells through the hook; the paired base),"
  " **`egoc`** (the arm: `mlp_ego_c(ego block at the tick)`, standardised by X's `cons_mu` / `cons_sd`, into code indices [46, 56, 57, 58, 59] at every replan tick, `--limit causal`) and `egoshufc` (the same cells permuted across the 2,399 windows by"
  " `numpy.random.default_rng(20261029).permutation(2399)`, sha256 `64785f8a…`; the deliberate-regression arm). Beside, CTRL only: `identity_l` and `egol` in the `--limit logged` lane with the banked `mlp_ego` (`a205609a…`).")
aud("[46, 56, 57, 58, 59]", "PREREG section 2/3; pins.json: CODE_IDX in the reader (speedsub_t1_read.py CODE_IDX)")
aud("20261029; 2399", "PREREG section 3; READ.json controls K-SHUF.* (registered_sha256 64785f8a…, n 2399)")
aud("20 rows", "PREREG_V8F_SPEEDSUB_T1.md section 0 (the lane row): causal bucket = speed_max_bin(max measured ego speed over rows [j - 20, j])")
P("* **The primary lane is not the open channel.** `--limit causal` feeds the bucket of the **past** 20 rows of measured ego speed (`causal_bucket`). SPEEDSUB's PASS was read with the **logged** `limit` lane, the bucket of the ego's own realised **future** max speed"
  " (`D-SPEEDBAND-EGO-ECHO-1`, a declared open channel). The logged lane appears only beside the table (section 6) and decides nothing.")
P("* **Coverage.** The four families are in section 3, per family and per run, never pooled. Named gaps, per family and with the reason: S-POS across arms (STRATEGIC), the anti-echo block (LONGITUDINAL) and goal-head F1 at T1-kin (TACTICAL). The outcome above uses none of them.")

# ---------------------------------------------------------------------------------------------------------------- 1
hdr("## 1. The outcome, as the pre-registered table decides it", "1")
SEC[0] = "1"
P("**Hit rule (PREREG section 6):** a hit at run X is `egoc − identity` ADE6 **separated-better (CI upper end < 0) and a point at or below −" + aud(un(BAR, 5), "F / 2 = 0.1207 / 2; READ.json bar", "F = 0.1207 m (PREREG)") + " m**. PASS needs a hit at all four runs.")
P("")
P("| ADE6, `egoc − identity` (m; point [95 % CI]) | " + " | ".join(RUNH[r] for r in RUNS) + " |", "|---|" + "---|" * 4)
P("| point [CI] | " + " | ".join("**" + civ(d_ade[r]) + "**" if HIT[r] else civ(d_ade[r]) for r in RUNS) + " |")
for r in RUNS:
    aud(civ(d_ade[r]), f"SPEEDSUB_T1_READ.json d_ADE6_egoc_minus_identity.{r}.{{delta,lo,hi}}; tables.egoc-identity.{r}.ADE6", "equal in both places (asserted)")
P("| separated-better (CI upper end < 0) | " + " | ".join(yn(d_ade[r]["hi"] < 0) for r in RUNS) + " |")
P(f"| point ≤ −{un(BAR, 5)} | " + " | ".join(yn(d_ade[r]["delta"] <= -BAR) for r in RUNS) + " |")
P("| **hit** | " + " | ".join(("**yes**" if HIT[r] else "no") for r in RUNS) + " |")
P("| point as a multiple of the bar (DERIVED) | " + " | ".join(aud(f"{sg(d_ade[r]['delta'] / BAR, 2)} {MULT}", f"DERIVED: delta / {un(BAR, 5)}", f"{d_ade[r]['delta']} / {BAR}") for r in RUNS) + " |")
P("| mean ADE6, `identity` → `egoc` (m) | " + " | ".join(aud(f"{un(EI[r]['ADE6']['mean_ref'])} → {un(EI[r]['ADE6']['mean_arm'])}", f"SPEEDSUB_T1_READ.json tables.egoc-identity.{r}.ADE6.{{mean_ref,mean_arm}}") for r in RUNS) + " |")
P("| hits | " + aud(f"{N_HIT} of 4", "recomputed here and READ.json flags.n_hit", "REP s0 only") + " | | | |")
P("")
P("**The table's walk (REFUSED first, then HARMS, then PASS, PARTIAL, NULL):**")
P("")
P("| step (PREREG section 6, with Amendment 1) | reading | fires |", "|---|---|---|")
P(f"| REFUSED: a control fails, or a run is missing | {aud(str(N_FLAGS) + ' of ' + str(N_FLAGS), 'READ.json controls: all true')} control flags true; 4 of 4 runs present | no |")
P(f"| HARMS: a guard fires (G-ADE, G-WORSE, G-NEW or G-VIOL) | none of the four fires (guard table below) | no |")
P(f"| PASS: no guard; a hit at all four runs | {N_HIT} of 4 | no |")
P(f"| PARTIAL: no guard; hits at both CTRL seeds | CTRL s0 {yn(HIT['CTRL_s0'])}, CTRL s1 {yn(HIT['CTRL_s1'])} | no |")
P(f"| **NULL**: anything else | hits at {N_HIT} of 4 runs, not at both CTRL seeds: label **{LABEL}** (the label describes, the table decides) | **yes** |")
P("")
P("**Committed next, verbatim from READ.json:** *\"" + READ["committed_next"] + "\"* (section 7 gives it, its cost, and the Master Mind's ranking beside it).")
P("")
P("### The guards, recomputed from the cells")
P("")
gwmax = max(len(v) for v in GW.values())
assert GW == {"accel MAE": ["CTRL_s1"], "yaw-rate MAE": ["CTRL_s1"], "L1 lateral-accel MAE": ["CTRL_s1"]} and gwmax == 1
assert GN_BOTH == ["curvature MAE", "L2 curvature MAE (GT>=5.2)", "M1 crawl share"] and all(GN_ID[l] == [True, True] for l in GN_BOTH) and GN_CODED == GN_AMEND == []
assert GV_RUNS == ["REP_s0"]
P("| guard | rule (PREREG) | reading, recomputed | fires |", "|---|---|---|---|")
P(f"| G-ADE | `egoc − identity` ADE6 not separated-worse at any run | separated-worse at {aud(str(len(G_ADE_RUNS)), 'recomputed: sepw(ADE6, egoc-identity) over 4 runs')} of 4 runs | no |")
P(f"| G-WORSE | no Read-A row (19 rows, ADE6 included) has `egoc − identity` separated-worse at ≥ 3 of the 4 runs | the most any row reaches is {aud(str(gwmax), 'recomputed over the 19 rows')} of 4 runs; three rows are separated-worse at one run each, all at CTRL s1: "
  f"accel MAE {cell('egoc-identity', 'CTRL_s1', 'accel MAE')}, yaw-rate MAE {cell('egoc-identity', 'CTRL_s1', 'yaw-rate MAE')}, L1 lateral-accel MAE {cell('egoc-identity', 'CTRL_s1', 'L1 lateral-accel MAE')} | no |")
P(f"| G-NEW | no row with `identity − P10` not separated-worse becomes separated-worse for `egoc − P10` at both CTRL seeds (Amendment 1 item 3: read per seed, must hold at both) | rows with `egoc − P10` separated-worse at both CTRL seeds: {', '.join(GN_BOTH)}; "
  "each is **also separated-worse under `identity − P10` at both seeds** (the crawl gap to P10 is there before the substitution), so none is new. The reader's coded rule (identity not separated-worse at ≥ 1 CTRL seed) and the Amendment 1 reading (not separated-worse at each) both give the empty set | no |")
P(f"| G-VIOL (Amendment 1) | `VIOL-ANY` not separated-worse at ≥ 3 of the 4 runs | separated-worse at {aud(str(len(GV_RUNS)), 'recomputed: sepw(VIOL-ANY, egoc-identity) over 4 runs')} of 4 runs (REP s0: {cell('egoc-identity', 'REP_s0', 'VIOL-ANY')}) | no |")
P("")
P("**Sensitivity (DERIVED, same code).** Without G-VIOL the word is the same (" + OUT_NO_GVIOL[0] + ", " + OUT_NO_GVIOL[1] + "); under the Amendment 1 reading of G-NEW the word is the same. The two halves of the hit rule name the same single run: separated-better alone gives REP s0 only,"
  " and the bar alone (no interval) gives REP s0 only (no other run's point reaches −" + un(BAR, 5) + "). So neither the interval rule nor the bar decides the word.")
P("")
P("### How to read this NULL (reported as written; none of it changes the word)")
P("")
lo_beyond = [r for r in RUNS if d_ade[r]["lo"] <= -BAR]
assert lo_beyond == ["CTRL_s0", "REP_s0", "REP_s1"]
P(f"* **NULL is \"the bar was not cleared at all four runs\", not \"the effect is zero\".** At CTRL s0 and REP s1 the interval's lower end ({sg(d_ade['CTRL_s0']['lo'])}, {sg(d_ade['REP_s1']['lo'])}) lies beyond −{un(BAR, 5)}: the data at those runs do not exclude an ADE6 cut as large as the bar."
  f" At CTRL s1 the lower end ({sg(d_ade['CTRL_s1']['lo'])}) lies inside the bar, so a bar-sized cut is excluded there at the 95 % level (the episode question). All three points are small (the multiples of the bar are in the table above) and their intervals include 0.")
P("* **The one hit is the slow-T draw.** REP s0's T predicts all five speed cells 0.73 to 1.17 m/s slow at T0 (INHERITED, section 4). PREREG section 10 limit 4 reads this in advance: a hit there \"partly measures repair of that draw\", which is why both CTRL seeds were required as well. Neither CTRL seed hit.")
P("* **The regression arm reads its known value.** `egoshufc − identity` ADE6 is separated-worse at all four runs (K-REG, section 2): the instrument sees a harm of about 4 m when O is fed shuffled cells, so the small `egoc` movements at the healthy draws are not a blind instrument.")

P("")
P("### Controls (each must read its known value before any row is read; any failure is REFUSED)")
P("")
kgroups = {}
for k in CT:
    g = k.split(":")[0]
    kgroups.setdefault(g, []).append(k)
assert sum(len(v) for v in kgroups.values()) == N_FLAGS
for g_, v_ in kgroups.items():
    aud(str(len(v_)), f"SPEEDSUB_T1_READ.json controls: number of flags in group {g_}", "K1+K2+K3+K5+K6+K7+GRID = 7 x 12 = 84" if g_ == "K1" else "")
aud("84", "DERIVED: flags in K1, K2, K3, K5, K6, K7, GRID = 7 groups x 12 rollouts", "7 x 12")
CD = READ["controls_detail"]
mx = lambda f: max(f(r) for r in RUNS)
k_rec_max = max(CD[f"K-REC:{r}"]["max_abs_used"] for r in RUNS)
k_rec_o_max = max(CD[f"K-REC:{r}"]["max_abs_o_read"] for r in RUNS)
k_id_max = max(CD[f"K-ID:{r}"]["max_abs_ade6_diff_m"] for r in RUNS)
k_id_cl = max(CD[f"K-ID:{r}"]["max_abs_cl_diff_m"] for r in RUNS)
k_nonspeed = max(v for k, v in CD["K-NONSPEED"].items())
k7 = [CD[f"K7:{r}:{m}"]["x1"] for r in RUNS for m in ("identity", "egoc", "egoshufc")]
k7_pose = max(max(x["max_s_pose_vs_rebuilt_m_f32_tol"], x["max_t_pose_vs_rebuilt_m_f32_tol"]) for x in k7)
assert k_id_max == 0.0 and k_id_cl == 0.0 and k_nonspeed == 0.0 and all(x["max_cl_m"] == 0.0 and x["max_pose_tick_f64"] == 0.0 for x in k7)
assert all(CD[f"K-REACH:{r}"]["egoc_moved_fraction"] == 1.0 and CD[f"K-REACH:{r}"]["egoshufc_differs_from_egoc_fraction"] == 1.0 and CD[f"K-REACH-PLAN:{r}"]["egoc_cl_differs_from_identity_fraction"] == 1.0 for r in RUNS)
assert all(CD[f"K-SHUF:{r}"]["n_fixed"] == 0 and CD[f"K-SHUF:{r}"]["recorded_sha256"] == CD[f"K-SHUF:{r}"]["registered_sha256"] for r in RUNS)
assert all(CD[f"K-FLOOR:{r}:{m}"]["max_abs_diff"] == 0.0 for r in RUNS for m in ("identity", "egoc", "egoshufc"))
assert all(CD[f"K-ALIGN:{r}:{m}"]["rows"] == 7197 for r in RUNS for m in ("identity", "egoc", "egoshufc"))
assert k_rec_max < 1e-5 and k_rec_o_max < 1e-5 and k7_pose < 1e-4
P("| control | must read | reading (MEASURED, all 4 runs) | catches |", "|---|---|---|---|")
P(f"| K-ID ({len(kgroups['K-ID'])}) | `identity` per-window ADE6 equals the banked own rollout within 1e-4 m | max absolute difference in ADE6 {aud(un(k_id_max, 1), 'READ.json controls_detail K-ID:*.max_abs_ade6_diff_m')} m, and over every `cl` coordinate {un(k_id_cl, 1)} m (bit-equal) | hook drift, a thread-count or tool-tree change |")
P(f"| K-ZERO ({len(kgroups['K-ZERO'])}) | the reader's `identity − identity` reads exactly 0.0 on every computed row | delta = lo = hi = 0.0 on every computed row of the 4 runs (asserted here from `tables.identity-identity`) | a reader that manufactures differences |")
P(f"| K-NONSPEED (1; 12 values) | at every (window, tick) the 55 non-speed code dimensions O reads equal T's own | max absolute difference {un(k_nonspeed, 1)} at all 12 rollouts | a wider substitution than declared |")
P(f"| K-REC ({len(kgroups['K-REC'])}) | the cells O reads in `egoc` equal `mlp_ego_c` on the recorded ego block, standardised, ≤ 1e-5 | max {aud(sci(k_rec_max, 4), 'READ.json controls_detail K-REC:*.max_abs_used (max over runs)')} (O-read side {sci(k_rec_o_max, 4)}); 7,197 rows per run | a substitution that is not the read-out |")
P(f"| K-REACH ({len(kgroups['K-REACH'])}) and K-REACH-PLAN ({len(kgroups['K-REACH-PLAN'])}) | `egoc` moves the cells away from T's own on ≥ 99 % of (window, tick); `egoshufc` differs from `egoc` on ≥ 95 %; the `egoc` executed plan differs from `identity` | moved fraction 1.0, 1.0, 1.0 at all 4 runs | a substitution that never reaches O |")
P(f"| K-SHUF ({len(kgroups['K-SHUF'])}) | the permutation sha is the registered one; fixed points ≤ max(3, 1 %) | recorded = recomputed = registered `64785f8a…`; 0 fixed points of 2,399 | an unregistered shuffle |")
P(f"| K-FLOOR ({len(kgroups['K-FLOOR'])}) and K-WIN ({len(kgroups['K-WIN'])}) | the P10 floor arm, windows and GT equal the banked rollout bit for bit | max absolute difference 0.0 at all 12 rollouts; 2,399 windows / 141 episodes | a patch reaching the floor |")
P(f"| K-SHA ({len(kgroups['K-SHA'])}) and K-ALIGN ({len(kgroups['K-ALIGN'])}) | checkpoints, `mlp_ego_c`, tool, wrapper, instrument equal their pins; the tick rows align with the tool's | true at all 12 rollouts; 7,197 tick rows each (2,399 windows × 3 ticks) | a swapped file, a misaligned record |")
P(f"| K1, K2, K3, K5, K6, K7, GRID ({len(kgroups['K1']) + len(kgroups['K2']) + len(kgroups['K3']) + len(kgroups['K5']) + len(kgroups['K6']) + len(kgroups['K7']) + len(kgroups['GRID'])}) | the T1 heads pipeline's own per-rollout controls (inherited from the DAG-ORC reader's layer; their definitions are in that reader and are not restated here). K7: the rebuilt executed track equals the dumped `cl` | all true at all 12 rollouts; K7 max absolute difference in `cl` 0.0, pose deviation ≤ {aud(sci(k7_pose, 2), 'READ.json controls_detail K7:*:*.x1.max_{s,t}_pose_vs_rebuilt_m_f32_tol (max)')} m (float32 class) | pipeline drift |")
P(f"| **K-REG** (1) | `egoshufc − identity` ADE6 separated-worse at all four runs | " + "; ".join(f"{RUNH[r]} {civ(SI[r]['ADE6'])}" for r in RUNS) + " | an instrument that cannot see a harm |")
for r in RUNS:
    aud(civ(SI[r]["ADE6"]), f"SPEEDSUB_T1_READ.json controls_detail K-REG.{r}; tables.egoshufc-identity.{r}.ADE6")
aud("1e-4; 1e-5; 99 %; 95 %; 55; 3; 1 %", "PREREG section 4 tolerances (K-ID 1e-4 m, K-REC 1e-5, K-REACH 99 % / 95 %, K-NONSPEED 55 dims, K-SHUF max(3, 1 %))")
aud("7,197", "READ.json controls_detail K-ALIGN:*.rows = 7197 (2,399 windows x 3 ticks)")
aud("12 rollouts", "3 modes x 4 runs")
P("")
cfit = FITC["controls"]["C-FIT"]
ccausal = FITC["controls"]["C-CAUSAL"]
ccover = FITC["controls"]["C-COVER"]
cpar = FITC["controls"]["C-PARITY"]
assert ccausal["pass"] and ccover["pass"] and cpar["pass"] and FITC["controls"]["C-DISJ"]["pass"] and FITC["controls"]["C-DISJ"]["intersection"] == [] and all(v == 0 for v in ccausal["n_mismatch"].values())
assert ccover["n_windows"] == ccover["n_limit_ok_causal"] == 306566 and cpar["n_clips"] == 4512 and FITC["controls"]["C-DISJ"]["n_eval_clips"] == 141
P("**Controls before any rollout (the fit; INHERITED from the build, MEASURED there in `raw/2026-10-28-speedsub-t1-build/evidence/fit_c.json`, sha256 `" + PIN[BLD + "evidence/fit_c.json"][:8] + "…`):** "
  f"C-FIT PASS at {aud(sci(cfit['holdout_pred_max_abs_diff_ms'], 3), 'fit_c.json controls.C-FIT.holdout_pred_max_abs_diff_ms')} m/s against the 1e-5 tolerance (**a thin margin, disclosed**); "
  f"C-CAUSAL 0 mismatches against the tool's own `build_item --limit causal` at tick 0 on {th(ccausal['n_compared']['a_t=0'])} windows and at ticks 1 and 2 on {ccausal['n_compared']['a_t=1']} each; "
  f"C-COVER {th(ccover['n_limit_ok_causal'])} of {th(ccover['n_windows'])} TRAIN windows carry a finite past (causal-bucket agreement with the logged bucket {pct(ccover['agreement_with_logged'])}); "
  f"C-DISJ eval ∩ fit clips = ∅ ({FITC['controls']['C-DISJ']['n_eval_clips']} eval clips, {th(FITC['controls']['C-DISJ']['n_fit_clips'])} fit clips); C-PARITY {th(cpar['n_clips'])} clips / {th(ccover['n_windows'])} windows; C-INSTR PASS."
  f" The smoke (`--max-windows 200`, CTRL s0, `identity` and `egoc`, {SMOKE['controls']['K-ALIGN:identity']['detail']['rows']} tick rows) passed K-NONSPEED, K-REC ({sci(SMOKE['controls']['K-REC']['detail']['max_abs_used'], 3)}), K-REACH, K-SHA and K-ALIGN at {RO[SMOKE_T[0]][0]}–{STG['smoke']}; the CTRL s0 preflight passed before CTRL s1 was rolled ({STG['preflight']}).")
aud("1e-5", "PREREG section 2 / Amendment 1 item 5: C-FIT tolerance")
for k_, v_ in (("C-CAUSAL a_t=0", th(ccausal['n_compared']['a_t=0'])), ("C-CAUSAL a_t=1/2", str(ccausal['n_compared']['a_t=1'])), ("C-COVER", th(ccover['n_limit_ok_causal'])), ("C-COVER agreement", pct(ccover['agreement_with_logged'])),
               ("C-DISJ", str(FITC['controls']['C-DISJ']['n_eval_clips'])), ("C-DISJ fit clips", th(FITC['controls']['C-DISJ']['n_fit_clips'])), ("C-PARITY", th(cpar['n_clips'])), ("smoke rows", str(SMOKE['controls']['K-ALIGN:identity']['detail']['rows'])),
               ("smoke K-REC", sci(SMOKE['controls']['K-REC']['detail']['max_abs_used'], 3))):
    aud(v_, f"fit_c.json controls.{k_}; smoke_check_CTRL_s0.json", "")
aud(RO[SMOKE_T[0]][0] + "–" + STG["smoke"], "chain.log: smoke start (first smoke rollout) and ZZSST1-STAGE-DONE smoke")
aud(STG["preflight"], "chain.log ZZSST1-STAGE-DONE preflight")
aud("200", "PREREG Amendment 1 item 4: --max-windows 200")

# ---------------------------------------------------------------------------------------------------------------- 2
hdr("## 2. The rows: `egoc − identity` per run, on every row (T1-kin; point [95 % CI]; B / W = separated-better / separated-worse, " + NDASH + " = not separated)", "2")
SEC[0] = "2"
P("Each cell is the paired episode-cluster bootstrap of the arm minus the same run's `identity` rollout on the same windows (B = 2,000, seed 0). Direction: an error row is better when lower; TAC-LAT and TAC-LON (agreement) are better when higher."
  " `n` is windows / episodes (DK rows are the LEAD windows: n differs by row and run; lateral rows exclude tangent-less steps). Rows are grouped by the family they belong to; the family tables and the gaps are in section 3.")
P("")
rows_table("egoc-identity", LABELS_ALL)
P("")
P("Units are those of the DAG-ORC reader's rows (`RESULT_V8F_DAG_ORC.md` section 2); shares and agreement rows are fractions. `n` per cell (windows / episodes) for this table:")
P("")
P("| row | " + " | ".join(RUNH[r] for r in RUNS) + " |", "|---|" + "---|" * 4)
for l in LABELS_ALL:
    P(f"| {l} | " + " | ".join(ncell("egoc-identity", r, l) for r in RUNS) + " |")
    for r in RUNS:
        if computed(TAB["egoc-identity"][r][l]):
            aud(ncell("egoc-identity", r, l), f"SPEEDSUB_T1_READ.json tables.egoc-identity.{r}.{l}.n")
if EDGE:
    P("")
    P("\u2020 A rounded bound of the cell is exactly 0.0000. The flag (B / W / " + NDASH + ") is the estimator's own reading of the unrounded bound, not of the rounded one: " + "; ".join(f"{c} {RUNH[r]} {l}" for c, r, l in EDGE)
      + (". Cells where the flag disagrees with the rounded interval: " + (", ".join(f"{c} {RUNH[r]} {l}" for c, r, l in MISMATCH) if MISMATCH else "none") + "."))
nB = {r: sum(sepb(l, EI[r][l]) for l in LABELS_ALL) for r in RUNS}
nW = {r: sum(sepw(l, EI[r][l]) for l in LABELS_ALL) for r in RUNS}
nC = {r: sum(computed(EI[r][l]) for l in LABELS_ALL) for r in RUNS}
assert nC == {r: 18 for r in RUNS} and nB == {"CTRL_s0": 0, "CTRL_s1": 0, "REP_s0": 8, "REP_s1": 0} and nW == {"CTRL_s0": 0, "CTRL_s1": 3, "REP_s0": 2, "REP_s1": 0}
P("")
P("**Census of the table above (DERIVED; counted by `gen_result.py` over the " + aud(str(nC["CTRL_s0"]), "recomputed: rows with a computed `egoc - identity` cell (21 rows less the 3 NOT COMPUTED S-POS rows)") + " computed rows per run):**")
P("")
P("| run | rows separated-better | rows separated-worse | rows not separated |", "|---|---|---|---|")
for r in RUNS:
    P(f"| {RUNH[r]} | {aud(str(nB[r]), f'recomputed census, egoc-identity {r}')} | {aud(str(nW[r]), f'recomputed census, egoc-identity {r}')} | {aud(str(nC[r] - nB[r] - nW[r]), f'recomputed census, egoc-identity {r}')} |")
P("")
P("### The deliberate-regression arm, `egoshufc − identity` (T1-kin; every row; K-REG reads the ADE6 row)")
P("")
rows_table("egoshufc-identity", LABELS_ALL)
nBs = {r: sum(sepb(l, SI[r][l]) for l in LABELS_ALL) for r in RUNS}
nWs = {r: sum(sepw(l, SI[r][l]) for l in LABELS_ALL) for r in RUNS}
P("")
SB_LIST = [(r, l) for r in RUNS for l in LABELS_ALL if sepb(l, SI[r][l])]
SB_BY_ROW = {}
for r_, l_ in SB_LIST:
    SB_BY_ROW.setdefault(l_, []).append(RUNH[r_])
assert SB_BY_ROW == {"DK-VIOL": ["REP s0", "REP s1"], "VIOL-ANY": ["CTRL s0", "CTRL s1", "REP s1"], "VIOL-ANY-OL": ["CTRL s0", "CTRL s1", "REP s1"]}
for r in RUNS:
    for l in ("speed MAE", "along-track MAE", "accel MAE", "heading MAE", "yaw-rate MAE", "curvature MAE", "cross-track MAE", "L1 lateral-accel MAE"):
        assert sepw(l, SI[r][l]), (r, l)
P("The shuffled cells hurt: ADE6 is separated-worse at all four runs (K-REG), and so are the speed, along-track, accel, heading, yaw-rate, curvature, cross-track and L1 rows at every run (rows separated-worse per run, DERIVED: "
  + ", ".join(f"{RUNH[r]} {aud(str(nWs[r]), f'recomputed census, egoshufc-identity {r}')}" for r in RUNS) + "; separated-better: " + ", ".join(f"{RUNH[r]} {aud(str(nBs[r]), f'recomputed census, egoshufc-identity {r}')}" for r in RUNS) + ")."
  " The only cells that read **separated-better** under the shuffle are rule-violation shares: " + "; ".join(f"{l} at {', '.join(v)}" for l, v in SB_BY_ROW.items())
  + ". A lower violation share under a shuffle that costs about 4 m of ADE6 is not an improvement of the plan; the cells are listed so that no separated-better cell is omitted.")

# ---------------------------------------------------------------------------------------------------------------- 3
hdr("## 3. The four families (beside the decision; ADE is one row of four; per family, never pooled)", "3")
SEC[0] = "3"
P("Per family and per run, two things: the census of the section 2 cells for that family, and the arm's standing against the **P10 floor** (`egoc − P10`, T1-kin; same estimator and windows). STRATEGIC is reported, not paused, for v8F.")
FAMS = [("LONGITUDINAL", ["speed MAE", "along-track MAE", "accel MAE", "DK-VIOL", "DK-HW", "DK-TG"]), ("LATERAL", ["heading MAE", "yaw-rate MAE", "curvature MAE", "cross-track MAE", "L1 lateral-accel MAE", "L2 curvature MAE (GT>=5.2)"]),
        ("TACTICAL", ["TAC-LAT", "TAC-LON"])]
for fam, labs in FAMS:
    assert [l for l in LABELS_ALL if FAMILY[l] == fam] == labs
    SEC[0] = "3 " + fam
    P("")
    P(f"### {fam}")
    P("")
    P("| census of `egoc − identity` | " + " | ".join(RUNH[r] for r in RUNS) + " |", "|---|" + "---|" * 4)
    P("| rows separated-better / separated-worse / not separated | " + " | ".join(
        "{} / {} / {}".format(aud(str(sum(sepb(l, EI[r][l]) for l in labs)), f"recomputed census {fam} {r}"), aud(str(sum(sepw(l, EI[r][l]) for l in labs)), f"recomputed census {fam} {r}"),
                              aud(str(sum(computed(EI[r][l]) and not EI[r][l]['separated'] for l in labs)), f"recomputed census {fam} {r}")) for r in RUNS) + " |")
    P("")
    P(f"`egoc − P10` ({fam}):")
    P("")
    P("| row | " + " | ".join(RUNH[r] for r in RUNS) + " |", "|---|" + "---|" * 4)
    for l in labs:
        P(f"| {l} | " + " | ".join(cell("egoc-P10", r, l) for r in RUNS) + " |")
SEC[0] = "3 STRATEGIC"
P("")
P("### STRATEGIC")
P("")
sp = FF["STRATEGIC"]["rows"]
assert all(sp[l]["egoc-identity"][r] == "NOT COMPUTED" for l in sp for r in RUNS)
assert FF["STRATEGIC"]["note"].startswith("S-POS is within run (S - CTRV, ticks >= 1); paired S-POS across arms is NOT COMPUTED")
P("S-POS@30 / 50 / 80 are the strategic head's position error **within a run** (S − CTRV, ticks ≥ 1; READ.json `four_families.STRATEGIC.note`, `n` = *\"ticks>=1, unpaired vs CTRV\"*, interval machinery INHERITED from the DAG reader; direction `lower` = better, as the reader codes it)."
  " The reader's two columns below are the same quantity in two rollouts. **The paired `egoc − identity` S-POS row is NOT COMPUTED** (PREREG section 7): the reader scores S against CTRV inside each rollout and does not pair the two rollouts' S errors"
  " window by window, so a difference between the two columns is not a tested contrast. STRATEGIC therefore carries no across-arm statement in this read; it is reported, with this reason, and not passed.")
P("")
P("| row (S − CTRV, within run) | mode | " + " | ".join(RUNH[r] for r in RUNS) + " |", "|---|---|" + "---|" * 4)
for l in ("S-POS@30", "S-POS@50", "S-POS@80"):
    P(f"| {l} | `identity` | " + " | ".join(cell("identity-P10", r, l) for r in RUNS) + " |")
    P(f"| {l} | `egoc` | " + " | ".join(cell("egoc-P10", r, l) for r in RUNS) + " |")
    P(f"| {l} | `egoc − identity` | " + " | ".join("NOT COMPUTED" for r in RUNS) + " |")
P("")
P("### Gaps, per family: what is not computed, with the reason")
P("")
P("| family | row | status | reason |", "|---|---|---|---|")
P("| LONGITUDINAL | anti-echo block | **NOT BUILT** | the DAG-ORC reader carries none (READ.json note; PREREG section 7 allows *\"NOT BUILT with the reason\"*). A work item. |")
P("| LONGITUDINAL | DK-VIOL / DK-HW / DK-TG | computed | on the LEAD windows only (n per cell in section 2); DK is computed in this read, unlike SPEEDSUB's T1-class path |")
P("| TACTICAL | goal-head F1 at T1-kin | **NOT COMPUTABLE here** | the paired tool reads eval dumps, not tick dumps (READ.json note); n not available. TAC-LAT / TAC-LON are the 3-way agreement of the executed track with the human's. A work item. |")
P("| STRATEGIC | S-POS paired across arms | **NOT COMPUTED** | the reader does not pair S errors across rollouts (above) |")
P("| all | stop-window and event strata | not computed | not registered for this read |")
P("")
P("### The rule-violation rows (post-Amendment 1; RULES, beside the families)")
P("")
P("`VIOL-ANY` is the per-window any-rule violation of the executed `cl` path (rules SPEED / STOP / TURN / LEAD, the T0 eval's own `violations`, against the GT constraints of `eval_context`); `VIOL-ANY-OL` is the tick-0 plan, comparable with SPEEDSUB's T0 finding, and decides nothing.")
P("")
P("| row | contrast | " + " | ".join(RUNH[r] for r in RUNS) + " |", "|---|---|" + "---|" * 4)
for l in ("VIOL-ANY", "VIOL-ANY-OL"):
    P(f"| {l} | `egoc − identity` | " + " | ".join(cell("egoc-identity", r, l) for r in RUNS) + " |")
P(f"| VIOL-ANY | `egoc − P10` | " + " | ".join(cell("egoc-P10", r, "VIOL-ANY") for r in RUNS) + " |")
P(f"| VIOL-ANY | `identity − P10` | " + " | ".join(cell("identity-P10", r, "VIOL-ANY") for r in RUNS) + " |")
P("")
P("Mean violation share of the executed path (`identity` → `egoc`): " + "; ".join(
    f"{RUNH[r]} {aud(un(EI[r]['VIOL-ANY']['mean_ref']) + ' → ' + un(EI[r]['VIOL-ANY']['mean_arm']), f'tables.egoc-identity.{r}.VIOL-ANY.{{mean_ref,mean_arm}}')}" for r in RUNS) + ". The P10 floor sits at 0.7999 on this measure (Amendment 1's known-value check), so every DAG-OACC run is far below it with or without the substitution.")
assert all(max(EI[r]["VIOL-ANY"]["mean_ref"], EI[r]["VIOL-ANY"]["mean_arm"]) < 0.7999 for r in RUNS)
aud("0.7999", "PREREG Amendment 1 item 1 (MEASURED on the banked runs: P10 floor 0.7999 / 0.7999; build probes/probe_viol_ctrl_s0.json)")

# ---------------------------------------------------------------------------------------------------------------- 4
hdr("## 4. REP s0, the only hit: the repair of a slow-T draw", "4")
SEC[0] = "4"
sepb_rep = [l for l in LABELS_ALL if sepb(l, EI["REP_s0"][l])]
sepw_rep = [l for l in LABELS_ALL if sepw(l, EI["REP_s0"][l])]
assert sepb_rep == ["ADE6", "speed MAE", "along-track MAE", "accel MAE", "heading MAE", "yaw-rate MAE", "curvature MAE", "L2 curvature MAE (GT>=5.2)"] and sepw_rep == ["VIOL-ANY", "VIOL-ANY-OL"]
P("At REP s0 `egoc` cuts ADE6 by **" + civ(d_ade["REP_s0"]) + " m**, " + aud(un(abs(d_ade["REP_s0"]["delta"]) / BAR, 2), "DERIVED: |delta| / bar", f"{abs(d_ade['REP_s0']['delta'])} / {BAR}") + " times the bar. Of the 18 computed rows (T1-kin, `egoc − identity`), " + str(len(sepb_rep)) + " are separated-better and 2 separated-worse (section 2):")
P("")
P("| row | REP s0, `egoc − identity` |", "|---|---|")
for l in sepb_rep + sepw_rep:
    P(f"| {l} | {cell('egoc-identity', 'REP_s0', l)} |")
P("")
P("The separated-better rows are the three longitudinal ones (speed, along-track, accel), ADE6, and four lateral rows (heading, yaw-rate, curvature, L2). **The cost is rule violation:** the share of windows on which the executed path breaks a SPEED / STOP / TURN / LEAD rule rises from " + un(EI["REP_s0"]["VIOL-ANY"]["mean_ref"]) + " to " + un(EI["REP_s0"]["VIOL-ANY"]["mean_arm"]) + " (`VIOL-ANY` " + sg(EI["REP_s0"]["VIOL-ANY"]["delta"]) + ")."
  " G-VIOL needs 3 of the 4 runs and fires at 1, so it is a disclosed cost beside the NULL, not a guard firing. M1 crawl share does not separate (" + cell("egoc-identity", "REP_s0", "M1 crawl share") + "): the repair is of the speed cells' draw, not of the crawl as such.")
P("")
P("### Why REP s0 (INHERITED, T0 diagnostic; `raw/2026-10-28-speedsub-t1-prereg/rep_s0_diag/REP_S0_SPEED_BIAS.md`, sha256 `" + PIN[V8 + 'raw/2026-10-28-speedsub-t1-prereg/rep_s0_diag/REP_S0_SPEED_BIAS.md'][:8] + "…`)")
P("")
P("REP s0's T predicts every speed cell slow against CTRL s0 on the same 2,399 windows at T0, separated (REP s0 − CTRL s0, m/s, mean [95 % CI]): "
  + "; ".join(f"`{k}` {sg(v[1])} [{sg(v[2])}, {sg(v[3])}]" for k, v in REPN.items()) + ". So the five cells read " + aud(un(REP_MIN, 2), "REP_S0_SPEED_BIAS.md table: min |REP s0 - CTRL s0|, v_a_ms 0.7315") + " to " + aud(un(REP_MAX, 2), "REP_S0_SPEED_BIAS.md table: max |REP s0 - CTRL s0|, v_lo_ms 1.1700")
  + " m/s slow. The plan inherits it: speed bias " + f"{sg(SPEEDBIAS[1], 3)} against {sg(SPEEDBIAS[0], 3)} m/s, final along-track bias {sg(FINALBIAS[1], 2)} against {sg(FINALBIAS[0], 2)} m (T0)."
  " REP s1 and A sit with CTRL. The diagnosis is the Master Mind's, written before this read; this read is its T1-kin test. The substitution helps at the draw whose cells are most wrong and at no other run, which is what the diagnosis predicts; it is **one** slow-T draw (n = 1), and the floor rule below shows the cells carry part of REP s0's T1-kin gap, not all of it.")
aud("0.73", "REP_S0_SPEED_BIAS.md: 'about 1 m/s slow', 0.73-1.17 m/s (min of the five cells)")
aud("1.17", "REP_S0_SPEED_BIAS.md: 0.73-1.17 m/s (max of the five cells)")
for k_, v_ in REPN.items():
    aud(f"{sg(v_[1])} [{sg(v_[2])}, {sg(v_[3])}]", f"REP_S0_SPEED_BIAS.md table row {k_}, column REP s0 - CTRL s0")
aud(sg(SPEEDBIAS[1], 3) + " / " + sg(SPEEDBIAS[0], 3), "REP_S0_SPEED_BIAS.md second table: speed bias REP s0 / CTRL s0")
aud(sg(FINALBIAS[1], 2) + " / " + sg(FINALBIAS[0], 2), "REP_S0_SPEED_BIAS.md second table: along-track final bias REP s0 / CTRL s0")
P("")
P("### The floor rule `H-V8F-SPEEDSUB-T1-2` (beside; it decides which floors later arms use)")
P("")
P("`g_id(r)` = REP s0^identity − CTRL s0^identity and `g_c(r)` = REP s0^egoc − CTRL s0^egoc, both paired on the same windows (READ.json `floor_rule_H_V8F_SPEEDSUB_T1_2`). **COLLAPSE** needs `g_id(r) > 0` and `g_c(r) ≤ 0.5 · g_id(r)` on **all three** rows; otherwise **NOT COLLAPSE** and the floors stay DAG-ORC's.")
P("")
P("| s0 (decisive) row | g_id | g_c | g_c / g_id | closed by `egoc` (DERIVED) | threshold 0.5 · g_id | collapses |", "|---|---|---|---|---|---|---|")
for r in FLOOR_ROWS:
    gid, gc = FLOOR["0"]["gid"][r], FLOOR["0"]["gc"][r]
    P(f"| {r} | {aud(sg(gid), f'READ.json floor_rule.s0.g_id.{r}')} | {aud(sg(gc), f'READ.json floor_rule.s0.g_c.{r}')} | {aud(un(gc / gid, 3), 'DERIVED: g_c / g_id', f'{gc} / {gid}')} | {aud(pct(1 - gc / gid), 'DERIVED: 1 - g_c / g_id')} | {aud(sg(0.5 * gid), 'DERIVED: 0.5 * g_id')} | {yn(FLOOR['0']['per'][r])} |")
P("")
P(f"**s0 verdict: {FLOOR['0']['verdict']}** (READ.json: *\"{FR['s0 (decisive, beside)']['consequence']}\"*). The REP s0 − CTRL s0 gap on ADE6 falls from {un(FLOOR['0']['gid']['ADE6'])} to {un(FLOOR['0']['gc']['ADE6'])} m (a drop of {pct(1 - FLOOR['0']['gc']['ADE6'] / FLOOR['0']['gid']['ADE6'])}), "
  "yaw-rate MAE and curvature MAE fall by less, and no row reaches half. The arithmetic ties the pieces together (DERIVED, asserted): g_c = g_id + d(REP s0) − d(CTRL s0) = "
  f"{un(FLOOR['0']['gid']['ADE6'])} + ({sg(d_ade['REP_s0']['delta'])}) {MINUS} ({sg(d_ade['CTRL_s0']['delta'])}) = {un(FLOOR['0']['gid']['ADE6'] + d_ade['REP_s0']['delta'] - d_ade['CTRL_s0']['delta'])} (reader: {un(FLOOR['0']['gc']['ADE6'])}), so the part of the gap the substitution removes is REP s0's own repair net of the small CTRL s0 movement.")
aud(sg(d_ade['REP_s0']['delta']) + "; " + sg(d_ade['CTRL_s0']['delta']), "DERIVED identity: d(REP s0) and d(CTRL s0) from READ.json d_ADE6_egoc_minus_identity")
aud(un(FLOOR['0']['gid']['ADE6'] + d_ade['REP_s0']['delta'] - d_ade['CTRL_s0']['delta']), "DERIVED: g_id + d(REP s0) - d(CTRL s0)")
P("")
P("| s1 (reported) row | g_id | g_c | collapses | note |", "|---|---|---|---|---|")
for r in FLOOR_ROWS:
    gid, gc = FLOOR["1"]["gid"][r], FLOOR["1"]["gc"][r]
    note = "g_id is not above 0: there is no gap to collapse" if gid <= 0 else "g_id above 0 and g_c below half"
    P(f"| {r} | {aud(sg(gid), f'READ.json floor_rule.s1.g_id.{r}')} | {aud(sg(gc), f'READ.json floor_rule.s1.g_c.{r}')} | {yn(FLOOR['1']['per'][r])} | {note} |")
P("")
P("**s1 verdict: " + FLOOR["1"]["verdict"] + ".** REP s1 − CTRL s1 is about 0 on identity (the gap rows are +0.0116 / +0.8903 / −0.0008), so the rule cannot collapse what is not there; curvature MAE has `g_id` below 0 and fails the first clause.")
P("")
P("The means in the cells reproduce the ADE6 points to 4 dp; for yaw-rate and curvature the difference of means over the two runs' valid windows agrees with the reader's paired point within 1.5 % (the windows differ by one tangent-less step at s0). The verdict per row is the same either way (asserted).")
aud("1.5 %", "generator tolerance for the mean-difference cross-check on yaw-rate / curvature rows (windows differ by one at s0)")

# ---------------------------------------------------------------------------------------------------------------- 5
hdr("## 5. The healthy draws: CTRL s0, CTRL s1 and REP s1 show no separated ADE6 change", "5")
SEC[0] = "5"
HEALTHY = ("CTRL_s0", "CTRL_s1", "REP_s1")
assert all(not d_ade[r]["separated"] and d_ade[r]["hi"] > 0 for r in HEALTHY)
assert not any(EI[r]["M1 crawl share"]["separated"] for r in RUNS)
P("| run | ADE6 `egoc − identity` | multiple of the bar | rows separated-better / separated-worse | note |", "|---|---|---|---|---|")
notes = {"CTRL_s0": "no row of 18 separates", "CTRL_s1": "accel MAE, yaw-rate MAE and L1 lateral-accel MAE separated-worse (one run each: G-WORSE needs 3 runs per row)", "REP_s1": "no row of 18 separates"}
for r in HEALTHY:
    P(f"| {RUNH[r]} | {civ(d_ade[r])} | {sg(d_ade[r]['delta'] / BAR, 2)} {MULT} | {nB[r]} / {nW[r]} | {notes[r]} |")
P("")
P("* **CTRL s0 and REP s1 move nowhere:** 0 of 18 computed rows separate in either direction (" + f"{nB['CTRL_s0']} + {nW['CTRL_s0']} and {nB['REP_s1']} + {nW['REP_s1']}" + "). Their ADE6 points are " + sg(d_ade['CTRL_s0']['delta']) + " and " + sg(d_ade['REP_s1']['delta']) + " m, with intervals that include 0.")
P("* **CTRL s1 is the one healthy draw that moves, and it moves the wrong way on three rows:** " + "; ".join(f"{l} {cell('egoc-identity', 'CTRL_s1', l)}" for l in ("accel MAE", "yaw-rate MAE", "L1 lateral-accel MAE")) + ". ADE6 is " + sg(d_ade['CTRL_s1']['delta']) + " m, not separated, 0.13 of the bar."
  " Three separated-worse rows at one run do not fire G-WORSE (3 runs per row), but they are the first place a harm would show: the logged-lane read-out raises the same three rows at CTRL s1 (section 6), which is INFERENCE that CTRL s1's O reacts to substituted speed cells on those rows, not a measured mechanism.")
aud("0.13", "DERIVED: |CTRL s1 ADE6 delta| / bar = 0.0076 / 0.06035 (section 1 table)")
P("* **M1 crawl share moves at no run.** `egoc − identity`: " + ", ".join(f"{RUNH[r]} {cell('egoc-identity', r, 'M1 crawl share')}" for r in RUNS) + ". The crawl is not a speed-cell effect that this read-out reaches.")
P("")
P("### How much of the crawl gap to P10 `egoc` closes (DERIVED, in DAG-ORC's form: (identity−P10 − egoc−P10) / identity−P10; negative = the gap widens)")
P("")
P("The P10 floor is the crawl reference. DAG-ORC's perturbation arm A closed 27.6 % to 79.9 % of the DAG-OACC crawl gap (INHERITED, `RESULT_V8F_DAG_ORC.md` section 4 (b)); the same arithmetic for `egoc`:")
P("")
P("| row | CTRL s0 `egoc` | CTRL s0 DAG-ORC A | CTRL s1 `egoc` | CTRL s1 DAG-ORC A | REP s0 `egoc` | REP s1 `egoc` |", "|---|---|---|---|---|---|---|")
SHARE = {}
for l in CRAWL:
    row = []
    for r in RUNS:
        idp, ecp = IP[r][l]["delta"], EP[r][l]["delta"]
        assert idp > 0
        SHARE[(r, l)] = (idp - ecp) / idp
    parts = []
    for r in RUNS:
        idp, ecp = IP[r][l]["delta"], EP[r][l]["delta"]
        parts.append(aud(pct(SHARE[(r, l)]), f"DERIVED: (identity-P10 - egoc-P10) / identity-P10, {r} {l}", f"({idp} - {ecp}) / {idp}"))
    a0 = aud(pct(A_SHARE[("s0", l)] / 100), f"RESULT_V8F_DAG_ORC.md section 4 (b), s0 {l}, share closed (INHERITED)")
    a1 = aud(pct(A_SHARE[("s1", l)] / 100), f"RESULT_V8F_DAG_ORC.md section 4 (b), s1 {l}, share closed (INHERITED)")
    P(f"| {l} | {parts[0]} | {a0} | {parts[1]} | {a1} | {parts[2]} | {parts[3]} |")
mx_ctrl = max(SHARE[(r, l)] for r in CTRL for l in CRAWL)
mn_a = min(A_SHARE[(s, l)] for s in ("s0", "s1") for l in CRAWL)
assert mx_ctrl < 0.13 and mn_a / 100 > mx_ctrl
P("")
P(f"At the healthy CTRL seeds `egoc` closes at most {aud(pct(mx_ctrl), 'DERIVED: max over CTRL seeds and the four crawl rows')} of the gap on any crawl row (and widens it on {sum(SHARE[(r, l)] < 0 for r in CTRL for l in CRAWL)} of 8 cells), against {aud(pct(mn_a / 100), 'RESULT_V8F_DAG_ORC.md section 4 (b): smallest share closed by A')} at the least for DAG-ORC's perturbation."
  f" At REP s0, the slow-T draw, it closes {pct(min(SHARE[('REP_s0', l)] for l in CRAWL[:3]))} to {pct(max(SHARE[('REP_s0', l)] for l in CRAWL[:3]))} on yaw-rate, curvature and L2 and {pct(SHARE[('REP_s0', 'M1 crawl share')])} on M1."
  " The two tables are not one experiment (different arms, different n by one to three windows per cell); the comparison is of sizes, not a contrast.")
assert 0.19 <= min(SHARE[("REP_s0", l)] for l in CRAWL[:3]) and max(SHARE[("REP_s0", l)] for l in CRAWL[:3]) <= 0.27 and SHARE[("REP_s0", "M1 crawl share")] < 0.06
for _v in (min(SHARE[("REP_s0", l)] for l in CRAWL[:3]), max(SHARE[("REP_s0", l)] for l in CRAWL[:3]), SHARE[("REP_s0", "M1 crawl share")]):
    aud(pct(_v), "DERIVED: (identity-P10 - egoc-P10) / identity-P10 at REP s0 (min / max over yaw-rate, curvature, L2; M1)")
aud("8", "count of CTRL-seed x crawl-row cells (2 x 4)")

# ---------------------------------------------------------------------------------------------------------------- 6
hdr("## 6. Beside, deciding nothing: the logged-lane bridge", "6")
SEC[0] = "6"
P("Lane: `" + BES["lane"] + "`. Role: " + BES["role"] + ". CTRL only: `egol − identity_l`, the banked `mlp_ego` (sha256 `a205609a…`, fitted on the logged bucket) in the lane where T itself reads the logged bucket. Tier: T1-kin. Same estimator.")
P("")
bl_rows = ["ADE6", "speed MAE", "along-track MAE", "accel MAE", "heading MAE", "yaw-rate MAE", "curvature MAE", "L1 lateral-accel MAE", "L2 curvature MAE (GT>=5.2)", "M1 crawl share", "TAC-LAT", "TAC-LON"]


def bcell(r, l):
    e = BL[r][l]
    s = civ(e) + " " + mark(l, e)
    aud(civ(e), f"SPEEDSUB_T1_BESIDE.json egol_minus_identity_l.{r}.{l}.{{delta,lo,hi}}", f"flag separated = {e['separated']}")
    return s


P("| row | CTRL s0 `egol − identity_l` | CTRL s1 `egol − identity_l` | CTRL s0 `egoc − identity` (primary lane) | CTRL s1 `egoc − identity` (primary lane) |", "|---|---|---|---|---|")
for l in bl_rows:
    P(f"| {l} | {bcell('CTRL_s0', l)} | {bcell('CTRL_s1', l)} | {cell('egoc-identity', 'CTRL_s0', l)} | {cell('egoc-identity', 'CTRL_s1', l)} |")
bl_sep = {r: [l for l in LABELS19 if computed(BL[r][l]) and BL[r][l]["separated"]] for r in BL}
assert bl_sep == {"CTRL_s0": ["M1 crawl share"], "CTRL_s1": ["accel MAE", "yaw-rate MAE", "L1 lateral-accel MAE", "L2 curvature MAE (GT>=5.2)", "M1 crawl share"]}
assert all(sepw(l, BL[r][l]) for r in BL for l in bl_sep[r])
P("")
P(f"**The read-out with the most future information available moves ADE6 by {sg(BL['CTRL_s0']['ADE6']['delta'])} and {sg(BL['CTRL_s1']['ADE6']['delta'])} m at CTRL s0 and s1: not separated** (intervals {sg(BL['CTRL_s0']['ADE6']['lo'])} to {sg(BL['CTRL_s0']['ADE6']['hi'])} and {sg(BL['CTRL_s1']['ADE6']['lo'])} to {sg(BL['CTRL_s1']['ADE6']['hi'])}). "
  "Its separated rows are all separated-worse: M1 crawl share at CTRL s0, and accel MAE, yaw-rate MAE, L1 lateral-accel MAE, L2 curvature MAE and M1 at CTRL s1 (the same three rows that `egoc` raises at CTRL s1, plus L2 and M1).")
P("")
P("What the bridge says and does not say:")
P("")
P("* **It is the read-out SPEEDSUB passed with.** SPEEDSUB (M2b-mixo, T1-class open loop, the same `mlp_ego`, the logged lane) cut ADE6 by −0.1290 m [−0.2107, −0.0548] at s0 and −0.0686 m [−0.1151, −0.0232] at s1 (INHERITED, `RESULT_V8F_SPEEDSUB.md`, PREREG section 0). On DAG-OACC at T1-kin the same read-out in the same lane does not move ADE6.")
P("* **The change has four ingredients at once:** the base (M2b-mixo against DAG-OACC, whose O has seen its own rollouts), the tier (T1-class against T1-kin), the rollout machinery, and the checkpoint seeds. No arm here isolates one. The bridge does not say why SPEEDSUB's PASS did not carry; it says that on the adopted base the open-channel read-out is not a lever either.")
P("* **The open channel (`D-SPEEDBAND-EGO-ECHO-1`).** The logged bucket is the bucket of the ego's own realised future max speed (R² 0.9702 from bin plus `v0`; INHERITED, `Project Steering/GOALS_AND_CLAIMS.md`), admissible only as a declared input and never a training signal. **INFERENCE:** because the best-informed read-out is also flat at CTRL, the causal read-out's flatness is not explained by the future information it lacks.")
aud("0.9702", "Project Steering/GOALS_AND_CLAIMS.md D-SPEEDBAND-EGO-ECHO-1 ruling (INHERITED; also quoted in RESULT_V8F_SPEEDSUB.md)")
for s_, q_ in (("-0.1290 [-0.2107, -0.0548]", "s0"), ("-0.0686 [-0.1151, -0.0232]", "s1")):
    aud(s_.replace("-", MINUS), f"PREREG_V8F_SPEEDSUB_T1.md section 0 table row SPEEDSUB ({q_}); RESULT_V8F_SPEEDSUB.md", "INHERITED")
aud("0.1290; 0.2107; 0.0548; 0.0686; 0.1151; 0.0232", "PREREG_V8F_SPEEDSUB_T1.md section 0 (SPEEDSUB ADE6 cuts at s0 / s1), as quoted")
P("")
P("TRAIN hold-out RMSE of the two read-outs (mean over the five cells, SD units, lower is better; fold 0, 846 clips, 57,479 hold-out rows; decides nothing): the logged-bucket read-out "
  + aud(un(FITC["beside"]["logged"]["holdout_rmse_sd_mean"], 4), "fit_c.json beside.logged.holdout_rmse_sd_mean") + ", the causal `mlp_ego_c` " + aud(un(FITC["beside"]["causal"]["holdout_rmse_sd_mean"], 4), "fit_c.json beside.causal.holdout_rmse_sd_mean")
  + "; the logged bucket is the better predictor of T's speed cells on TRAIN, as it should be for an oracle input.")
assert FITC["beside"]["logged"]["n_holdout_clips"] == 846 and FITC["beside"]["logged"]["n_holdout_rows"] == 57479
aud("846; 57,479", "fit_c.json beside.logged.n_holdout_clips / n_holdout_rows")

# ---------------------------------------------------------------------------------------------------------------- 7
hdr("## 7. Next (Rule Zero): a NULL leaves a committed next, a ranking, and named blockers", "7")
SEC[0] = "7"
tot12 = sum(RO[t][2] for t in DECISIVE)
mean12 = tot12 / 12
lo12, hi12 = min(RO[t][2] for t in DECISIVE), max(RO[t][2] for t in DECISIVE)
P("**Committed next, verbatim from READ.json:** *\"" + READ["committed_next"] + "\"*. PREREG section 8 (a): the tick-frame design discards the four-frame history that `frame_ok` only masks; the same MLP is fitted on the four history frames of `p10f` plus the causal bucket and nav"
  " (an extract over `train_arrays`' window index; ESTIMATED under 1 CPU-h), **under its own amendment with the same table**. Its stated reason: on the pure-dynamics fit the per-tick information is what loses to `T_own`.")
aud("1 CPU-h", "PREREG_V8F_SPEEDSUB_T1.md section 8 (a): ESTIMATED under 1 CPU-h")
P("")
P("**Cost of (a), so the PI can overrule the ranking below (ESTIMATED from measured rollout times):** the extract and fit under 1 CPU-h (PREREG), then twelve rollouts at the measured " + f"{lo12} to {hi12} s each" + f" ({mm_ss(tot12)} for the twelve decisive rollouts of this read, MEASURED in `chain.log`) plus the reader ({un(WALL_READER, 1)} s): at most about {aud(un((3600 + tot12 + WALL_READER) / 3600, 1), 'ESTIMATED: (3600 s + sum of the twelve rollouts + reader wall) / 3600', f'(3600 + {tot12} + {WALL_READER}) / 3600')} h of dev-box CPU in all. No GPU and no spend; the PREREG names no PI decision for it. It is blocked on nothing external; it needs only its amendment text.")
aud(f"{lo12} to {hi12} s", "chain.log DONE lines of the twelve decisive rollouts: min and max seconds")
aud(mm_ss(tot12), "DERIVED: sum of the twelve decisive rollouts' seconds in chain.log", f"{tot12} s")
P("")
P("### Master Mind's ranking (beside the table)")
P("")
P("**Label: ESTIMATED / INFERENCE ranking, not a measured outcome.** The table's committed next above is unchanged; this ranks what to do first.")
P("")
P("1. **(a) stays registered and is not run now, because arithmetic predicts NULL.** The read-out with the most future information available, the logged oracle bucket (TRAIN hold-out "
  + un(FITC["beside"]["logged"]["holdout_rmse_sd_mean"], 4) + " SD units, against " + un(FITC["beside"]["causal"]["holdout_rmse_sd_mean"], 4) + " for `mlp_ego_c`), did not move healthy CTRL at T1-kin (ADE6 "
  + sg(BL["CTRL_s0"]["ADE6"]["delta"]) + " and " + sg(BL["CTRL_s1"]["ADE6"]["delta"]) + " m, not separated; section 6). A history-aware causal read-out carries less future information than that oracle. **So the speed-cell channel is not what limits healthy DAG-OACC draws.**"
  " Its measured use is repairing slow-T training draws (REP s0, section 4).")
P("2. **The largest measured crawl lever is DAG-ORC's perturbation** (`RESULT_V8F_DAG_ORC.md` section 4 (a)): arm A is separated-better on every crawl row at both seeds, "
  + "yaw-rate MAE " + f"{sg(A_EFF['yaw-rate MAE']['s0'][0])} and {sg(A_EFF['yaw-rate MAE']['s1'][0])}, curvature MAE {sg(A_EFF['curvature MAE']['s0'][0])} and {sg(A_EFF['curvature MAE']['s1'][0])}, L2 {sg(A_EFF['L2 curvature MAE (GT>=5.2)']['s0'][0])} and {sg(A_EFF['L2 curvature MAE (GT>=5.2)']['s1'][0])}, M1 {sg(A_EFF['M1 crawl share']['s0'][0])} and {sg(A_EFF['M1 crawl share']['s1'][0])}"
  + " (s0 and s1), closing 27.6 % to 79.9 % of the crawl gap (section 5), while `egoc` at the same CTRL seeds moves M1 by " + sg(EI["CTRL_s0"]["M1 crawl share"]["delta"]) + " and " + sg(EI["CTRL_s1"]["M1 crawl share"]["delta"]) + ", not separated."
  " **The registered crawl route is LAUNCH-INTENT on box 2, blocked on the PI's go.** INHERITED: `RESULT_V8F_DAG_ORC.md` section 6 registers NEXT-CRAWL → LAUNCH-INTENT as the sequel; the blocker (the PI's go, and box 2 for the training runs) is the Master Mind's statement for this ranking and is UNVERIFIED here.")
for l in CRAWL:
    for s in ("s0", "s1"):
        aud(sg(A_EFF[l][s][0]), f"RESULT_V8F_DAG_ORC.md section 4 (a): A - CTRL {l} {s} (INHERITED; parsed from the table)")
aud("27.6 % to 79.9 %", "RESULT_V8F_DAG_ORC.md section 4 (b): smallest and largest share closed by A (INHERITED)")
P("3. **(a) runs if a later arm shows healthy-draw speed cells binding.** This document's reading of \"binding\", for the Master Mind to confirm or restate: an arm whose healthy draws (not a slow-T draw) move ADE6 or a crawl row under a speed-cell substitution, separated and at a size above the rig's replicate floor. Until then it is a registered, costed, unrun item.")
P("")
P("**Not in the ranking, for the Master Mind to place (unblocked, 0 GPU):** PREREG section 8 (b), the per-cell attribution at T1-kin (each of the five cells substituted alone at CTRL s0 / s1: 10 rollouts, ESTIMATED about 1 h, "
  + f"about {mm_ss(10 * round(mean12))} at the measured mean of {un(mean12, 1)} s per rollout" + "). The PREREG routes (b) from HARMS, not from NULL; it is the instrument that would name the cell behind the three CTRL s1 rows in section 5.")
aud(mm_ss(10 * round(mean12)), "DERIVED: 10 rollouts x the measured mean seconds of the twelve decisive rollouts", f"10 x {round(mean12)} s")
aud(un(mean12, 1), "DERIVED: mean of the twelve decisive rollouts' seconds in chain.log", f"{tot12} / 12")
aud("10 rollouts; about 1 h", "PREREG_V8F_SPEEDSUB_T1.md section 8 (b) (ESTIMATED)")
P("")
P("### What is blocked, and on what (Rule Zero: every remaining lever)")
P("")
P("| lever | status | blocker, and what unblocks it |", "|---|---|---|")
P("| (a) the history-aware causal read-out | registered, costed, **not run** by the ranking | nothing external; its own amendment. Deprioritised by arithmetic (item 1), not blocked. |")
P("| (b) per-cell attribution at T1-kin | unblocked, not routed from NULL | the Master Mind's call to place it. |")
P("| LAUNCH-INTENT (the registered crawl route, NEXT-CRAWL) | registered | **the PI's go** and box 2 for the training runs (the Master Mind's statement, UNVERIFIED here; the route itself is INHERITED from `RESULT_V8F_DAG_ORC.md` section 6). |")
P("| the dominance read (Read A on the substituted CTRL, PREREG section 6.1) | **not reachable** | it is read on PASS only; the outcome is NULL. |")
P("| adoption of `mlp_ego_c` as a `forward_infer` speed sub-head | **not adopted** | the table adopts on PASS only. |")

# ---------------------------------------------------------------------------------------------------------------- 8
hdr("## 8. What this read cannot claim, disclosures, and the run record", "8")
SEC[0] = "8"
P("### What this read cannot claim")
P("")
P("* **It is not closed loop** (T1-kin is a self-action open loop) and it is not a capability claim for the deployed car: the read-out is not built as a sub-head, and no component here is deployed.")
P("* **One read-out fit.** *\"Would another fit say this?\"* is not answered (PREREG section 10 limit 1).")
P("* **Ticks 1 and 2 read the model's own executed motion**, outside the fit's distribution by construction (limit 2). The ticks-1-and-2 inputs may be why the substitution does less at T1-kin than the per-tick TRAIN accuracy suggests; this read does not separate that from the other explanations.")
P("* **`egoshufc` is a global shuffle** (limit 3), a control with a committed reading, not an estimate of the read-out's information.")
P("* **REP s0 is a slow-T draw** (limit 4): its hit partly measures repair of that draw. **The four runs are two seeds × two replicates of one recipe, not four recipes** (limit 5).")
P("* **NULL is not zero** (section 1): at CTRL s0 and REP s1 the intervals do not exclude a bar-sized cut; the episode-cluster bootstrap cannot say more.")
P("* **Not computed or not built:** S-POS across arms, the anti-echo block, goal-head F1 at T1-kin (section 3). They are work items.")
P("* **No replicate of the read-out and no inference-seed arm.** The planner is deterministic; the rig's run-to-run floor for a lever effect is the REP pair, not an interval.")
P("")
P("### Disclosures")
P("")
P("1. **Amendment 1 precedes every rollout.** It landed at `4cec09e7` (" + hm(T_AMEND) + "), the first rollout (the smoke) started at " + RO[SMOKE_T[0]][0] + ", and the chain ran from the tree at that commit (`landed_4cec09e7` in `pins.json`). It added G-VIOL and settled four readings (the tick frame is the window's NOW row; G-NEW per seed; the smoke's control set; C-FIT's float32 tolerance)."
  " **No guard fires (G-NEW under either reading),** so the word NULL does not depend on the amendment (section 1 sensitivity).")
P("2. **G-NEW's two readings.** The reader's coded rule fires at least as often as the Amendment 1 reading (set inclusion on this read's data, asserted); the reader's docstring and the build notes (`NOTES.md` section 7 item 3) word the coded rule differently from each other. Both rules give the empty set here, so the difference does not matter for this result.")
P("3. **C-FIT's margin is thin:** " + sci(cfit["holdout_pred_max_abs_diff_ms"], 3) + " m/s against the 1e-5 tolerance (94.1 % of it; a float32 summation-order effect, reproducible on the rerun, disclosed in Amendment 1 item 5). The control passed as written.")
aud("94.1 %", "DERIVED: 9.412e-6 / 1e-5", f"{cfit['holdout_pred_max_abs_diff_ms'] / 1e-5:.4f}")
assert f"{cfit['holdout_pred_max_abs_diff_ms'] / 1e-5 * 100:.1f}" == "94.1"
P("4. **The fit was reused, not re-run, in the chain.** The chain's fit stage found `fit_c.json` FITTED and reused it (`chain.log`: *\"fit: reusing F:/TanitAD-scratch/speedsub_t1/fit (fit_c.json FITTED)\"*). The fit was made once at build time, before Amendment 1 and before any rollout; `readout_c.pt` sha256 `799f65ed…` and `fit_c.json` sha256 `" + PIN[BLD + "evidence/fit_c.json"][:8] + "…` were pinned in `pins.json` before the first rollout. There is no selection among read-outs, and C-DISJ (eval ∩ fit clips = ∅) holds, so nothing was tuned on eval-141.")
P("5. **Threads and cost.** OMP / MKL 8 (the default of `run_speedsub_t1.sh`, `${OMP_NUM_THREADS:-8}`; the detached launcher `run_speedsub_t1.cmd` sets no override, and the effective value is not echoed in the log), `CUDA_VISIBLE_DEVICES=-1`, one rollout at a time. Measured rollout times are in the table below: " + f"{lo12} to {hi12} s per decisive rollout, against the PREREG's ESTIMATE of about 305 s (MEASURED for DAG-OACC in FINDING_3, INHERITED). The cause of the difference is not examined; K-ID reads a max absolute difference of 0.0 against the banked rollouts, so the speed difference changes no number.")
aud("305 s", "PREREG_V8F_SPEEDSUB_T1.md section 9: about 305 s per rollout (INHERITED estimate)")
P("6. **Prior.** The PREREG's ESTIMATED prior was that PASS is not the expected outcome (the causal fit loses the oracle part of `limit`). The outcome agrees with the prior; that is not evidence for the prior beyond this one read.")
P("7. **What the re-derivation is and is not.** `gen_result.py` recomputes the decision from the reader's per-run cells (4 dp, with the estimator's separation flags) and checks the JSON against itself and against the reader's flags: it tests the decision logic, the guards, the floor rule and the consistency of `SPEEDSUB_T1_READ.json`. It does **not** recompute the cells from the per-window dumps or re-run the bootstrap (the dumps are a single copy on F:); a defect inside the estimator or the heavy layer would not show here. The reader's heavy layer was exercised on the banked DAG-OACC dumps before the read (38 of 38 rows equal, max absolute difference 0.0; INHERITED, `NOTES.md` section 4).")
P("8. **Single copies.** The per-episode dumps, tick arrays and run trees are on `F:/TanitAD-scratch/speedsub_t1/out/` only (single copy); the reader's JSON, the manifest, the pins and the logs are in the repo. If F: is lost, the tables here stay checkable against READ.json but cannot be recomputed from the dumps.")
P("")
aud("38 of 38", "raw/2026-10-28-speedsub-t1-build/NOTES.md section 4 (heavy-layer dry run on the banked DAG-OACC dumps; INHERITED)")
P("### Run record (`chain.log`; every stage ran once, in order; no stop, no refusal, no re-run)")
P("")
P("| rollout | start | done | wall (s) |", "|---|---|---|---|")
for grp, tags in (("smoke", SMOKE_T), ("decisive", DECISIVE), ("beside (logged lane)", BESIDE_T)):
    for t in tags:
        P(f"| {t} ({grp}) | {RO[t][0]} | {RO[t][1]} | {RO[t][2]} |")
        aud(RO[t][0] + " / " + RO[t][1] + " / " + str(RO[t][2]), f"chain.log START / DONE lines for {t}")
P("")
P("Stage ends (`ZZSST1-STAGE-DONE`): " + ", ".join(f"{k} {v}" for k, v in STG.items()) + ". The manifest was written at " + T_MAN + " and the reader ran from there to " + STG["read"] + f" (stage wall {mm_ss(hhmmss_s(STG['read']) - hhmmss_s(T_MAN))}; the reader's own wall {un(WALL_READER, 1)} s).")
for k, v in STG.items():
    aud(v, f"chain.log ZZSST1-STAGE-DONE {k}")
aud(T_MAN, "chain.log ZZSST1-STAGE-DONE rep1 (manifest written next line)")
aud(mm_ss(hhmmss_s(STG['read']) - hhmmss_s(T_MAN)), "DERIVED from chain.log stamps")
aud(mm_ss(hhmmss_s(T_END) - hhmmss_s(T_START)), "DERIVED: last stage stamp minus first stamp", f"{hhmmss_s(T_END) - hhmmss_s(T_START)} s")

# ---------------------------------------------------------------------------------------------------------------- 9
hdr("## 9. Provenance (artifact path, sha256)", "9")
SEC[0] = "9"
P("Repo paths are under `" + V8 + "` unless they start with `Project Steering/`. **Status** says where the bytes are: *in the repo at the tip* (checked against the live tip by the hold builder), *staged by this hold*, or *not in the repo*.")
P("")
P("| role | repo path | sha256 | status |", "|---|---|---|---|")


def rel(p):
    return p[len(V8):] if p.startswith(V8) else p


for role, p in (("the read (decision of record)", RDD + "SPEEDSUB_T1_READ.json"), ("the beside logged-lane read", RDD + "SPEEDSUB_T1_BESIDE.json"), ("pins (checkpoints, read-out, tool, wrapper, instrument, reader)", RDD + "pins.json"),
                ("runs manifest", RDD + "runs_manifest.json"), ("chain log", RDD + "chain.log"), ("chain console output", RDD + "chain_speedsub_t1.out"), ("CTRL s0 preflight record", RDD + "preflight_CTRL_s0.json"),
                ("smoke record", RDD + "smoke_check_CTRL_s0.json"), ("the pre-registered reader", BLD + "read/speedsub_t1_read.py"), ("the fit and its controls", BLD + "evidence/fit_c.json"), ("build notes", BLD + "NOTES.md"),
                ("chain script", BLD + "run_speedsub_t1.sh"), ("PREREG with Amendment 1", "Project Steering/PREREG_V8F_SPEEDSUB_T1.md"),
                ("REP s0 speed-cell diagnostic (INHERITED)", V8 + "raw/2026-10-28-speedsub-t1-prereg/rep_s0_diag/REP_S0_SPEED_BIAS.md"), ("DAG-ORC RESULT (INHERITED numbers, section 4)", V8 + "RESULT_V8F_DAG_ORC.md"),
                ("SPEEDSUB RESULT (INHERITED numbers)", V8 + "RESULT_V8F_SPEEDSUB.md")):
    P(f"| {role} | `{rel(p)}` | `{PIN[p]}` | in the repo at the tip |")
STAGED = ("gen_result.py", "audit_cover.py", "NUMBERS_AUDIT.md", "REGISTER_ROW.md", "mutate_gen_check.py", "gen_mutation_check.out")
P("| this document's generator, coverage check, audit, register-row text, mutation check and its output, builder | `" + PKG + "audit/` and `" + PKG + "landing/` (hashes in `" + PKG + "PACKAGE_SHA256SUMS.txt`) | see that file | staged by this hold |")
P("")
P("**Not in the repo (single copy on F: scratch):** `F:/TanitAD-scratch/speedsub_t1/out/` (the 18 rollouts' dumps, tick arrays and records, " + "`runs_manifest.json` lists every path), `F:/TanitAD-scratch/speedsub_t1/fit/readout_c.pt` (sha256 `" + PINS["readout_c"]["sha256"] + "`), the four DAG-OACC / DAG-ORC checkpoints (sha256 in `pins.json`: "
  + ", ".join(f"{RUNH[r]} `{CK[r]}`" for r in RUNS) + "), and the banked `mlp_ego` (`" + PINS["readout_banked"]["sha256"] + "`).")
P("")
P("The in-repo instrument tests are INHERITED from the build (`NOTES.md`): the wrapper 33 tests and 29 of 29 mutants, the pin step 52 tests and 25 of 25 mutants, the reader 73 tests and 110 of 110 mutants. They were not re-run for this document.")
aud("33; 29; 52; 25; 73; 110", "raw/2026-10-28-speedsub-t1-build/NOTES.md sections 2, 3, 4 (INHERITED)")

TXT = "\n".join(L) + "\n"
TXT = TXT.replace("\n\n\n", "\n\n")
FORBID = ("void", "superseded", "retracted", "formerly", "no longer", "correction", "withdrawn")
for w in FORBID:
    assert w not in TXT.lower(), w
assert "\r" not in TXT and not re.search(r"hf_[A-Za-z0-9]{8,}", TXT)
assert TXT.count("Outcome **NULL**") == 1 and "closed loop" in TXT

# ---------------------------------------------------------------------------------------------------------------- register row
SEC[0] = "register"
d = d_ade
rowtxt = (
    "**NULL (label NULL-ONE; MEASURED 2026-10-28, T1-kin self-action open loop, Amendment 1 in force; all " + aud(str(N_FLAGS), "READ.json len(controls)") + " reader control flags PASS, no guard fires; the outcome, the four guards and the floor rule re-derived from the per-run cells by `gen_result.py`).** "
    "ADE6 `egoc − identity` (m, 95 % CI): " + "; ".join(f"{RUNH[r]} {civ(d[r])}" + (" (the only hit)" if HIT[r] else "") for r in RUNS) + f"; a hit needs separated-better and a point at or below −{un(BAR, 5)} m, and PASS needs a hit at all four runs. "
    "REP s0 is the slow-T draw (its T speed cells read " + un(REP_MIN, 2) + "–" + un(REP_MAX, 2) + " m/s slow at T0): there `egoc` also moves speed, along-track, accel, heading, yaw-rate, curvature and L2 separated-better and VIOL-ANY separated-worse (" + civ(EI["REP_s0"]["VIOL-ANY"]) + "); "
    "G-VIOL needs 3 of 4 runs and fires at 1. The healthy draws (CTRL s0, CTRL s1, REP s1) show no separated ADE6 change; CTRL s0 and REP s1 move no row of 18 separated; CTRL s1 has accel, yaw-rate and L1 separated-worse at that one run, so G-WORSE (3 of 4 runs) does not fire; "
    "M1 crawl share moves at no run. Deliberate-regression arm `egoshufc` ADE6 " + " / ".join(sg(SI[r]["ADE6"]["delta"]) for r in RUNS) + " m, separated-worse at all four runs (K-REG). "
    f"Floor rule (H-V8F-SPEEDSUB-T1-2): NOT COLLAPSE at s0 (the REP s0 − CTRL s0 gap on ADE6 {un(FLOOR['0']['gid']['ADE6'])} → {un(FLOOR['0']['gc']['ADE6'])} m, yaw-rate MAE {un(FLOOR['0']['gid']['yaw-rate MAE'])} → {un(FLOOR['0']['gc']['yaw-rate MAE'])}, curvature MAE {un(FLOOR['0']['gid']['curvature MAE'])} → {un(FLOOR['0']['gc']['curvature MAE'])}, each above half) and at s1. "
    "Four families per run (RESULT section 3, never pooled): LONGITUDINAL (incl. distance keeping), LATERAL and TACTICAL reported; STRATEGIC reported, S-POS across arms NOT COMPUTED (the reader does not pair S errors across rollouts); anti-echo NOT BUILT; goal-head F1 at T1-kin NOT COMPUTABLE. "
    "Beside, decides nothing: the logged-lane read-out (a declared open channel, `D-SPEEDBAND-EGO-ECHO-1`) moves CTRL's ADE6 by " + sg(BL["CTRL_s0"]["ADE6"]["delta"]) + " [" + sg(BL["CTRL_s0"]["ADE6"]["lo"]) + ", " + sg(BL["CTRL_s0"]["ADE6"]["hi"]) + "] / " + sg(BL["CTRL_s1"]["ADE6"]["delta"]) + " [" + sg(BL["CTRL_s1"]["ADE6"]["lo"]) + ", " + sg(BL["CTRL_s1"]["ADE6"]["hi"]) + "] m, not separated. "
    "Paired episode-cluster bootstrap, B = 2,000, 141 episodes, per run (the episode question only); four training runs carry the training-run question; one read-out fit; deterministic planner. "
    "Committed next: section 8 (a), the history-aware causal read-out. Master Mind's ranking (ESTIMATED / INFERENCE, not a measured outcome): (a) stays registered and is not run now, because the logged oracle bucket, the read-out with the most future information, did not move healthy CTRL; "
    "DAG-ORC's perturbation is the largest measured crawl lever, and its registered route (LAUNCH-INTENT, box 2) waits on the PI's go. "
    "`RESULT_V8F_SPEEDSUB_T1.md`, `raw/2026-10-28-speedsub-t1-read/SPEEDSUB_T1_READ.json` (sha256 " + READ_SHA[:8] + "…).")
for w in FORBID:
    assert w not in rowtxt.lower(), w
assert "|" not in rowtxt and "\n" not in rowtxt and not rowtxt.rstrip().endswith("Was:") and "Was:" not in rowtxt
aud("register row status prefix", "same cells as the RESULT sections 1, 3, 4, 5, 6, 7; SPEEDSUB_T1_READ.json tables, floor_rule, guards; SPEEDSUB_T1_BESIDE.json")
rr = [
    "# H-V8F-SPEEDSUB-T1-1: replacement text for the register row's status prefix (staged, not landed)",
    "",
    "Generated by `gen_result.py` from `SPEEDSUB_T1_READ.json` (sha256 `" + READ_SHA[:8] + "…`) and `SPEEDSUB_T1_BESIDE.json`. The hold builder `mk_hold_speedsub_t1_result.py` reads the paragraph between the two markers and replaces the row's short status prefix with it (from `**NULL (label NULL-ONE;`",
    "up to and including `RESULT to follow. `). Everything from `Was: **PRE-REGISTERED, NOT RUN (2026-10-28).**` on stays byte-identical; the claim cell, the evidence cell (`MEASURED (NULL)`) and the source cell stay. The file's line count does not change (the row is one line).",
    "",
    "Row: the line of `Project Steering/GOALS_AND_CLAIMS.md` that starts `| **H-V8F-SPEEDSUB-T1-1** (2026-10-28) |`.",
    "",
    "## Status cell (new prefix)",
    "",
    "<!-- STATUS-BEGIN -->",
    rowtxt,
    "<!-- STATUS-END -->",
    "",
]
RRT = "\n".join(rr)
assert "\r" not in RRT

# ---------------------------------------------------------------------------------------------------------------- audit
AUDIT_PREFACE = ["# NUMBERS_AUDIT: every number in RESULT_V8F_SPEEDSUB_T1.md and REGISTER_ROW.md, with the source it was read from", "",
                 "Generated by `gen_result.py` in the same run that wrote the RESULT, so each rendered value below is the value that was printed. Paths are JSON key paths into `SPEEDSUB_T1_READ.json`"
                 f" (sha256 `{READ_SHA}`), `SPEEDSUB_T1_BESIDE.json` (sha256 `{BES_SHA}`), `fit_c.json`, `pins.json`, `runs_manifest.json`, or lines of `chain.log`, all read with `git show` at tip `{TIP[:8]}`. `{{a,b}}` means each of those keys. Derived numbers show the arithmetic.", "",
                 "Cross-checks the script asserts before it writes (a failure stops the build):", "",
                 "* The PREREG section 6 table with Amendment 1, coded in the script from the PREREG text, applied to the per-run cells of `tables.egoc-identity` / `egoc-P10` / `identity-P10` gives NULL, label NULL-ONE, equal to the reader's `OUTCOME`, `label`, `flags`, `hits`, each guard (`fires`, rows, runs) and `committed_next`.",
                 "* Every cell's separation flag agrees with its rounded interval (`lo > 0` or `hi < 0`), except cells whose rounded bound is exactly 0.0000 (listed in RESULT section 2 with a dagger); `identity - identity` reads exactly 0.0 on every computed row (K-ZERO).",
                 f"* All {N_FLAGS} control flags are true; K-ID, K-NONSPEED and K-FLOOR read exactly 0.0; K-REC and the K7 pose deviation are below tolerance; K-REG is separated-worse at all four runs; the two families' cells equal the flat tables' cells.",
                 "* The floor rule is recomputed from the reader's paired points and, as a cross-check, from the cells' means (identical verdicts per row); the identity g_c = g_id + d(REP) - d(CTRL) holds to 2e-4 m at both seeds.",
                 "* All inputs hash to their pins (listed in RESULT section 9); the chain log has 18 START / DONE pairs whose `in N s` equals the stamp difference within 2 s; Amendment 1 landed before the first rollout.",
                 "* Coverage (`audit_cover.py`, run at the end of the build and asserted to report 0 uncovered): every numeric token of the RESULT and of the register row text has a matching token in the table below."
                 " Not counted as data: section references, list and heading numerals, hex digests, dates inside repo paths.",
                 "", "| section | rendered | source | arithmetic / note |", "|---|---|---|---|"]

# prose numbers that carry no table cell are registered (with their source) in the block below; it is filled from the audit_cover listing and every entry names where the number comes from
PROSE = []
for tok, src in PROSE:
    aud(tok, "prose count or constant: " + src)

# ensure each distinct section-9 provenance hash prefix etc. do not need numbers; write outputs only after every check above has passed
A_ = list(AUDIT_PREFACE)
for sec_, ren, src, note in AUD:
    A_.append("| " + " | ".join(x.replace("|", "\\|").replace("\n", " ") for x in (sec_, ren, src, note)) + " |")
open(OUT + "RESULT_V8F_SPEEDSUB_T1.md", "w", encoding="utf-8", newline="\n").write(TXT)
open(OUT + "REGISTER_ROW.md", "w", encoding="utf-8", newline="\n").write(RRT)
open(OUT + "NUMBERS_AUDIT.md", "w", encoding="utf-8", newline="\n").write("\n".join(A_) + "\n")
print("wrote RESULT", len(TXT), "chars,", len(AUD), "audit entries; outcome", OUTCOME, LABEL, "hits", HIT, "edge cells", EDGE)
import importlib.util
_spec = importlib.util.spec_from_file_location("audit_cover", OUT + "audit_cover.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_un = _mod.main()
assert all(len(v) == 0 for v in _un.values()), _un
