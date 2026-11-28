"""Build the E2-TAIL bank hold: e2/e2raw/e2tail/{stageA, stageB/{read,logs}, paired_diag, scripts, README_E2TAIL.md, SHA256SUMS} + GOALS H-R9-E2-TAIL.
usage: python -I build_e2tail_hold.py <tip sha> <stageA pull root> <stageB read dir> <s6c_resume.log> <ROLL_STATUS.txt> <paired_diag dir or -> <hold dir> [--no-goals]
  stageA pull root = F:/refcv9_e3/e2tail_bank/pull (holds runs/E2TAIL_K2DC/stageA/..., runs/r9e2-*/model_ema_tail5k6k.pt.provenance.json, inputs/*.sh)
The verdict text is QUOTED from E2_VERDICT.md (the read's own words). M-map / M-lat are read from the banked files by the prereg's committed bars."""
import hashlib
import json
import os
import shutil
import subprocess
import sys

TIP, PA, RD, S6LOG, ROLLST, PAIRED, H = sys.argv[1:8]
NOGOALS = "--no-goals" in sys.argv[8:]
GD = "C:/Users/Admin/tanitad-push/.git"
GIT = shutil.which("git")
E2T = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-27-perception-redesign/e2/e2raw/e2tail"
GOALS = "Project Steering/GOALS_AND_CLAIMS.md"
SCR = "C:/Users/Admin/AppData/Local/Temp/claude/4ac3a056b343bb35/4a3d05c7-0ead-4eba-a77c-659814aeb9bf/scratchpad/k2dc"
READ_FILES = ["E2_VERDICT.json", "E2_VERDICT.md", "EVAL_K0.json", "EVAL_K2.json", "INPUTS.json", "camblind_s0.json", "camblind_s1.json",
              "distkeep_s0.json", "distkeep_s1.json", "families_s0.json", "families_s1.json",
              "s6c/S6C_K0_s0.json", "s6c/S6C_K0_s1.json", "s6c/S6C_K2_s0.json", "s6c/S6C_K2_s1.json"]
SA = PA + "/runs/E2TAIL_K2DC/stageA"
STAGEA_FILES = ["STATUS.txt", "EVAL_K2tail_fitthr.json", "EVAL_K2tail_default.json", "EVAL_K0tail_fitthr.json", "EVAL_K0tail_default.json",
                "thr_K2tail_psw.json", "thr_K2tail_pswpedgeF.json", "thr_K0tail_psw.json", "thr_K0tail_pswpedgeF.json", "fit_remap_e2.json"]
DUMPS = ["fitdump_K2", "fitdump_K0", "devdump_K2", "devdump_K0"]
MD5 = {"K2": "893c790e1a722b1d40d76fbfbbaa9de4", "K0": "f0ae6f32ebfe0fa2187f060d06aef2d5"}
# committed bars (PREREG_E2TAIL_2026-11-16.md section 4)
M_MAP_BAR, P1_BAR, P2_BAR, EMA_S0, RAW_S0 = 0.4712, 0.1080, 0.1123, 0.1040, 0.1163


def git(*a):
    return subprocess.run([GIT, "--git-dir=" + GD, *a], capture_output=True, check=True).stdout


assert not os.path.exists(H), H
os.makedirs(H + "/code")
expect = []


def put_new(rel, data):
    assert len(rel) <= 200, (len(rel), rel)
    r = subprocess.run([GIT, "--git-dir=" + GD, "rev-parse", "-q", "--verify", TIP + ":" + rel], capture_output=True)
    assert r.returncode != 0, "exists at tip: " + rel
    dst = H + "/code/" + rel
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    open(dst, "wb").write(data)
    expect.append((rel, "NEW"))


def pt(f):
    d = json.load(open(f, encoding="utf-8"))["nearfield"]["ci"]["nf_swedge_prec_lv_5_20"]
    return d["point"], d["lo"], d["hi"]


f4 = lambda v: "%.4f" % v  # noqa: E731

# --- stage A (map) ---
missing = [f for f in STAGEA_FILES if not os.path.isfile(SA + "/" + f)] + [d for d in DUMPS if not os.path.isfile(SA + "/" + d + "/dump.json")]
missing += [f for f in READ_FILES if not os.path.isfile(RD + "/" + f)]
assert not missing, missing
for f in STAGEA_FILES:
    put_new(E2T + "/stageA/" + f, open(SA + "/" + f, "rb").read())
for d in DUMPS:
    dj = json.load(open(SA + "/" + d + "/dump.json", encoding="utf-8"))
    assert dj.get("ckpt_md5") == MD5[d[-2:]] and int(dj.get("step")) == 6000, (d, dj.get("ckpt_md5"), dj.get("step"))
    put_new(E2T + "/stageA/dump_headers/" + d + ".dump.json", open(SA + "/" + d + "/dump.json", "rb").read())
for arm in ("K2DC", "K0"):
    put_new(E2T + "/stageA/tail_weights/" + arm + ".model_ema_tail5k6k.pt.provenance.json",
            open(PA + "/runs/r9e2-" + arm + "/model_ema_tail5k6k.pt.provenance.json", "rb").read())
for f in ("e2tail_stageA.sh", "e2tail_stageB_launch.sh"):
    put_new(E2T + "/scripts/" + f, open(PA + "/inputs/" + f, "rb").read())
for f in ("e2tail_paired_a.sh", "e2tail_s6c_pull_score.sh", "wait_e2tailB_then_s6c.sh", "build_e2tail_hold.py"):
    put_new(E2T + "/scripts/" + f, open(SCR + "/" + f, "rb").read().replace(b"\r\n", b"\n"))
k2, k2lo, k2hi = pt(SA + "/EVAL_K2tail_fitthr.json")
k0, k0lo, k0hi = pt(SA + "/EVAL_K0tail_fitthr.json")
k2d, k0d = pt(SA + "/EVAL_K2tail_default.json"), pt(SA + "/EVAL_K0tail_default.json")
a_pass = k2 > k0
m_map = "CONFIRMED" if k2 >= M_MAP_BAR else "FAILED"

# --- stage B (planner) ---
for f in READ_FILES:
    put_new(E2T + "/stageB/read/" + f, open(RD + "/" + f, "rb").read())
put_new(E2T + "/stageB/logs/ROLL_STATUS.txt", open(ROLLST, "rb").read())
put_new(E2T + "/stageB/logs/s6c_resume.log", open(S6LOG, "rb").read())
lat = {}
for s in ("s0", "s1"):
    r = json.load(open(RD + "/families_" + s + ".json", encoding="utf-8"))["rows"]["LATERAL"]["LAT_cross_mae_m"]
    lat[s] = r
x2 = lat["s0"]["a_mean"]
m_lat = "P1 (averaging)" if x2 <= P1_BAR else ("P2 (init pull)" if x2 >= P2_BAR else "MIXED")
md = open(RD + "/E2_VERDICT.md", encoding="utf-8").read().replace("\r\n", "\n")
first = md.split("\n", 1)[0].strip()
verdict = "GO" if "verdict: GO" in first else ("NO-GO" if "NO-GO" in first else "UNKNOWN: " + first)
tab = [ln for ln in md.split("\n") if ln.startswith("|")]
named = [ln for ln in md.split("\n") if ln.startswith("NO-GO names") or ln.startswith("GO ")][:6]
rows = {}
for ln in tab:
    c = [x.strip() for x in ln.strip("|").split("|")]
    if len(c) >= 2 and c[0][:3] in ("(a)", "(b)", "(c)", "(d)", "(e)"):
        rows[c[0][:3]] = c[1].replace("*", "")
fails = [k for k, v in rows.items() if "FAIL" in v.upper()]
if verdict == "GO":
    outcome = "committed outcome 1 (GO): report to the PI, who decides whether E3 fires; E3 does not fire by itself."
elif "(d)" in fails and m_lat.startswith("P2"):
    outcome = ("committed outcome 2 ((d) fails, M-lat P2): next arm = K2DC share target 0.02 (~20.5 Thor-h, needs the PI); the cheaper "
               "alternative beside it = a planner-only L2-SP anchor toward init (new code, own prereg).")
elif "(d)" in fails and m_lat.startswith("P1"):
    outcome = "committed outcome 3 ((d) fails, M-lat P1): next arm = K2DC share target 0.02 (~20.5 Thor-h, needs the PI)."
elif "(d)" in fails:
    outcome = ("(d) fails with M-lat MIXED: not a separate pre-registered branch; outcomes 2 and 3 both name the same next arm, "
               "K2DC share target 0.02 (~20.5 Thor-h, needs the PI).")
else:
    outcome = "NO-GO on " + ", ".join(fails) + ": not a pre-registered branch -> diagnose before proposing anything."
extra = [k for k in fails if k != "(d)"]
if extra and verdict != "GO":
    outcome += (" Also failing: " + ", ".join(extra) + " (no branch of section 5 names it; reported as written, see the clause rows).")
if "(d)" in fails:
    outcome += (" E2-WISE (PREREG_E2WISE_2026-11-17.md, landed ad9f1cbe before these lateral rows existed) is triggered: "
                "NO-GO with (d) failing.")

paired_line = ""
if PAIRED != "-":
    for f in sorted(os.listdir(PAIRED)):
        put_new(E2T + "/paired_diag/" + f, open(PAIRED + "/" + f, "rb").read())
    fr = json.load(open(PAIRED + "/FLOOR.json", encoding="utf-8"))["rows"]
    bits = []
    for k in ("nf_swedge_prec_lv_5_20", "nf_swedge_prec_lv_0_5", "nf_swedge_prec_lv_5_10", "nf_swedge_prec_lv_10_20"):
        if k in fr:
            q = fr[k]
            lab = {"nf_swedge_prec_lv_5_20": "primary 5-20 m", "nf_swedge_prec_lv_0_5": "0-5 m", "nf_swedge_prec_lv_5_10": "5-10 m",
                   "nf_swedge_prec_lv_10_20": "10-20 m"}[k]
            bits.append("%s %+.4f [%+.4f, %+.4f] %s" % (lab, q["delta"],
                        q["ci_lo"], q["ci_hi"], "SEPARATED" if q["separated"] else "not separated"))
    for k in [k for k in fr if "ap2m_all" in k][:1]:
        q = fr[k]
        bits.append("box %s %+.4f [%+.4f, %+.4f] %s" % (k, q["delta"], q["ci_lo"], q["ci_hi"], "SEPARATED" if q["separated"] else "not separated"))
    n_ep = fr.get("nf_swedge_prec_lv_5_20", {}).get("n_episodes")
    paired_line = ("* Paired diagnostic of clause (a) (`paired_diag/FLOOR.md`, decides nothing; K2DC tail - K0 tail at own FIT thresholds, paired "
                   "episode-cluster bootstrap B 2000, " + str(n_ep) + " episodes, T0): " + "; ".join(bits) + ".\n")

readme = ("# E2-TAIL: the E2 rule on the TAIL-EMA weights of K2DC and K0 -- verdict " + verdict + "\n\n"
          "Pre-registered before any tail number: `../PREREG_E2TAIL_2026-11-16.md` (landed c39d9fd3). Tail weights: `stack/scripts/ema_tail_weights.py`\n"
          "(landed e8a4631a): W_tail = (EMA_6000 - 0.81872 EMA_5000) / 0.18128 = the exact EMA-weighted mean of the raw weights at steps 5001-6000,\n"
          "with no step-0 weights (as any EMA at E3's 73,280 steps). Zero training. Thor read tree `read_tree_9697fa4df004_k2dc` (stage A) and\n"
          "`read_tree_9697fa4df004_k2dc_raw1c48ee8f` (stage B, roll 3816129936e8 with `R9E2_CKPT_NAME=model_ema_tail5k6k.pt`). DEV = eval139 even\n"
          "half, 70 episodes. One training run per arm (`H-ESTIM-SEED-1`): a separated interval answers another draw of EPISODES only.\n\n"
          "## Stage A -- clause (a), map primary `nf_swedge_prec_lv_5_20` at each arm's OWN FIT thresholds (T0, MEASURED)\n\n"
          "| arm | EMA (E2) | RAW (E2-RAW) | TAIL (this read) | default cut (decides nothing) |\n|---|---|---|---|---|\n"
          "| K2DC | 0.3940 | 0.4912 | **" + f4(k2) + "** [" + f4(k2lo) + ", " + f4(k2hi) + "] | " + f4(k2d[0]) + " |\n"
          "| K0 | 0.4776 | 0.4729 | **" + f4(k0) + "** [" + f4(k0lo) + ", " + f4(k0hi) + "] | " + f4(k0d[0]) + " |\n\n"
          "* **Clause (a) on tail weights: " + ("PASS" if a_pass else "FAIL") + "** (point rule as pre-registered): " + "%+.4f" % (k2 - k0) + ". Per-arm CIs\n"
          "  (`r9_e1_eval.py` bootstrap B 2000 seed 0) overlap; the paired interval is below.\n"
          "* **M-map: " + m_map + "** (bar: K2DC tail >= raw - 0.02 = " + f4(M_MAP_BAR) + "; read " + f4(k2) + ").\n"
          "* At the default cut K0 leads by a wide margin, as in E2-RAW (K0's fitted pedge thresholds are near zero): the heads sit at different\n"
          "  operating points, which is why the design scores (a) at fitted thresholds.\n" + paired_line +
          "* Tail guard: every dump.json records ckpt_md5 = the tail file's md5 (K2DC 893c790e, K0 f0ae6f32) and step 6000 (`stageA/dump_headers/`).\n\n"
          "## Stage B -- the read's verdict (quoted, unedited from `stageB/read/E2_VERDICT.md`)\n\n" + first + "\n\n" + "\n".join(tab) + "\n\n"
          + "\n".join(named) + "\n\n"
          "## M-lat (decides the DIAGNOSIS, not the GO)\n\n"
          "* K2DC seed-0 LAT cross-track mean on tail weights: **" + f4(x2) + " m** (EMA " + f4(EMA_S0) + ", raw " + f4(RAW_S0) + "; bars <= "
          + f4(P1_BAR) + " P1, >= " + f4(P2_BAR) + " P2) -> **" + m_lat + "**.\n"
          "* Paired K2DC - K0 LAT_cross_mae_m (T1, 70 ep): s0 " + "%+.4f [%+.4f, %+.4f]" % (lat["s0"]["delta"], lat["s0"]["lo"], lat["s0"]["hi"])
          + (" separated" if lat["s0"]["separated"] else " not separated") + " (K0 " + f4(lat["s0"]["b_mean"]) + "); s1 "
          + "%+.4f [%+.4f, %+.4f]" % (lat["s1"]["delta"], lat["s1"]["lo"], lat["s1"]["hi"]) + (" separated" if lat["s1"]["separated"] else " not separated")
          + " (K2DC " + f4(lat["s1"]["a_mean"]) + ", K0 " + f4(lat["s1"]["b_mean"]) + ").\n\n"
          "## Next (Rule Zero)\n\n* " + outcome + "\n* The original E2 and E2-RAW verdicts stand as written.\n\n"
          "## Provenance\n\n* Stage A files: pulled from Thor `runs/E2TAIL_K2DC/stageA/` 2026-11-17, sha256-equal (18 files). Rasters and perc_det.pkl not\n"
          "  banked (re-creatable from the tail files + read tree). Stage B read files: byte-exact copies of the S6c driver's pull (sha256-verified\n"
          "  against Thor `runs/E2TAIL_K2DC/stageB/`).\n")
put_new(E2T + "/README_E2TAIL.md", readme.encode("utf-8"))
sums = [hashlib.sha256(open(H + "/code/" + rel, "rb").read()).hexdigest() + "  " + rel[len(E2T) + 1:] for rel, _ in expect]
put_new(E2T + "/SHA256SUMS", ("\n".join(sorted(sums, key=lambda s: s[66:])) + "\n").encode("utf-8"))

if not NOGOALS:
    base_blob = git("rev-parse", TIP + ":" + GOALS).decode("ascii").strip()
    raw = git("cat-file", "blob", base_blob)
    crlf = b"\r\n" in raw[:20000]
    txt = raw.decode("utf-8").replace("\r\n", "\n")
    lines = txt.split("\n")
    idx = [i for i, l in enumerate(lines) if l.startswith("| **H-R9-E2-TAIL -- ")]
    assert len(idx) == 1, idx
    old = "| **PRE-REGISTERED"
    assert lines[idx[0]].count(old) == 1
    cl = ", ".join("%s %s" % (k, rows.get(k, "?")) for k in ("(a)", "(b)", "(c)", "(d)", "(e)"))
    new = ("| **E2-TAIL VERDICT (MEASURED 2026-11-17, read by the pre-registered pipeline; " + E2T + "/README_E2TAIL.md): " + verdict + " -- " + cl
           + ". Stage A (T0, DEV, own FIT thresholds): K2DC tail " + f4(k2) + " vs K0 tail " + f4(k0) + " (M-map " + m_map + "). M-lat: K2DC s0 "
           + "cross-track " + f4(x2) + " m (EMA " + f4(EMA_S0) + ", raw " + f4(RAW_S0) + ") -> " + m_lat + ". " + outcome + " One training run per arm.**"
           + " Earlier: **PRE-REGISTERED")
    lines[idx[0]] = lines[idx[0]].replace(old, new)
    out = "\n".join(lines)
    out = out.replace("\n", "\r\n") if crlf else out
    p = H + "/code/" + GOALS
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(out.encode("utf-8"))
    expect.insert(0, (GOALS, base_blob))
open(H + "/expect.tsv", "wb").write(("".join(a + "\t" + b + "\n" for a, b in expect)).encode("utf-8"))
print("HOLD OK", len(expect), "files; verdict", verdict, rows, "| (a)", "PASS" if a_pass else "FAIL", "M-map", m_map, "M-lat", m_lat, "x2 %.4f" % x2)
print("OUTCOME", outcome)
