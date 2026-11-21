"""Build an E2-WISE bank hold (MM 4a3d05, 2026-11-17): e2/e2raw/e2wise/<mode>/{stageA, stageB/{read,logs}, scripts, README_E2WISE_<MODE>.md, SHA256SUMS}
+ the GOALS row H-R9-E2-WISE (mode plan: a NEW row after H-R9-E2-TAIL; mode enc: the WISE-ENC verdict prepended to that row).
usage: python -I build_e2wise_hold.py <plan|enc> <tip sha> <stageA pull root> <stageB read dir> <s6c_resume.log> <ROLL_STATUS.txt> <hold dir> [--no-goals]
  stageA pull root holds runs/E2WISE_<MODE>/stageA/... and runs/r9e2-{K2DC,K0}/model_wise_<mode>_a050.pt.provenance.json and inputs/*.sh
Clause statuses come from E2_VERDICT.json; the verdict text is QUOTED from E2_VERDICT.md. Outcomes follow PREREG_E2WISE_2026-11-17.md s4 and
ADDENDUM 1 (both landed before any E2-WISE number). E2-TAIL reference values are read from the banked e2tail files at <tip>."""
import hashlib
import json
import os
import shutil
import subprocess
import sys

MODE, TIP, PA, RD, S6LOG, ROLLST, H = sys.argv[1:8]
assert MODE in ("plan", "enc"), MODE
NOGOALS = "--no-goals" in sys.argv[8:]
GD = "C:/Users/Admin/tanitad-push/.git"
GIT = shutil.which("git")
E2R = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-27-perception-redesign/e2/e2raw"
E2W = E2R + "/e2wise/" + MODE
E2T = E2R + "/e2tail"
GOALS = "Project Steering/GOALS_AND_CLAIMS.md"
SCR = "C:/Users/Admin/AppData/Local/Temp/claude/4ac3a056b343bb35/4a3d05c7-0ead-4eba-a77c-659814aeb9bf/scratchpad/k2dc"
TAG = {"plan": "wise", "enc": "wenc"}[MODE]
CK = "model_wise_" + MODE + "_a050.pt"
UP = MODE.upper()
READ_FILES = ["E2_VERDICT.json", "E2_VERDICT.md", "EVAL_K0.json", "EVAL_K2.json", "INPUTS.json", "camblind_s0.json", "camblind_s1.json",
              "distkeep_s0.json", "distkeep_s1.json", "families_s0.json", "families_s1.json",
              "s6c/S6C_K0_s0.json", "s6c/S6C_K0_s1.json", "s6c/S6C_K2_s0.json", "s6c/S6C_K2_s1.json"]
SA = PA + "/runs/E2WISE_" + UP + "/stageA"
STAGEA_FILES = ["STATUS.txt"] + ["%s_%s%s_%s.json" % (k, a, TAG, m) for a in ("K2", "K0") for k, ms in
                                 (("EVAL", ("fitthr", "default")), ("thr", ("psw", "pswpedgeF"))) for m in ms]
OPTIONAL_A = ["fit_remap_e2.json"]
DUMPS = ["fitdump_K2", "fitdump_K0", "devdump_K2", "devdump_K0"]
SCRIPTS_IN = {"plan": ["e2wise_stageA.sh", "e2wise_stageB_launch.sh"], "enc": ["e2wenc_stageA.sh", "e2wenc_stageB_launch.sh"]}[MODE]
SCRIPTS_SCR = {"plan": ["e2wise_s6c_pull_score.sh", "wait_e2wiseA.sh", "wait_e2wiseB_then_s6c.sh"],
               "enc": ["e2wenc_s6c_pull_score.sh", "wait_e2wencA.sh", "wait_e2wencB_then_s6c.sh"]}[MODE] + ["param_census.py", "build_e2wise_hold.py"]
TAIL_B = (0.30762299437913837, 0.3076645380376285)   # E2-TAIL clause (b) k2, k0 (banked e2tail/stageB/read/E2_VERDICT.json)


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


def tipjson(rel):
    return json.loads(git("show", TIP + ":" + rel).decode("utf-8"))


def mappt(d):
    q = d["nearfield"]["ci"]["nf_swedge_prec_lv_5_20"]
    return q["point"], q["lo"], q["hi"]


f4 = lambda v: "%.4f" % v  # noqa: E731
sg = lambda v: "%+.4f" % v  # noqa: E731

# --- stage A ---
missing = [f for f in STAGEA_FILES if not os.path.isfile(SA + "/" + f)] + [d for d in DUMPS if not os.path.isfile(SA + "/" + d + "/dump.json")]
missing += [f for f in READ_FILES if not os.path.isfile(RD + "/" + f)]
missing += [f for f in SCRIPTS_IN if not os.path.isfile(PA + "/inputs/" + f)] + [f for f in SCRIPTS_SCR if not os.path.isfile(SCR + "/" + f)]
missing += [a for a in ("K2DC", "K0") if not os.path.isfile(PA + "/runs/r9e2-" + a + "/" + CK + ".provenance.json")]
assert not missing, missing
for f in STAGEA_FILES + [f for f in OPTIONAL_A if os.path.isfile(SA + "/" + f)]:
    put_new(E2W + "/stageA/" + f, open(SA + "/" + f, "rb").read())
md5 = {}
for arm in ("K2DC", "K0"):
    pv = json.load(open(PA + "/runs/r9e2-" + arm + "/" + CK + ".provenance.json", encoding="utf-8"))
    md5[arm] = pv["md5_out"]
    if MODE == "plan":
        assert pv.get("perception_untouched_all_bit_equal") is True, (arm, "perception not bit-equal per the tool")
    put_new(E2W + "/stageA/weights/" + arm + "." + CK + ".provenance.json", open(PA + "/runs/r9e2-" + arm + "/" + CK + ".provenance.json", "rb").read())
for d in DUMPS:
    dj = json.load(open(SA + "/" + d + "/dump.json", encoding="utf-8"))
    arm = "K2DC" if d.endswith("K2") else "K0"
    assert int(dj.get("step")) == 6000, (d, dj.get("step"))
    if md5[arm]:
        assert dj.get("ckpt_md5") == md5[arm], (d, dj.get("ckpt_md5"), md5[arm])
    else:
        md5[arm] = dj.get("ckpt_md5")
    put_new(E2W + "/stageA/dump_headers/" + d + ".dump.json", open(SA + "/" + d + "/dump.json", "rb").read())
for f in SCRIPTS_IN:
    put_new(E2W + "/scripts/" + f, open(PA + "/inputs/" + f, "rb").read())
for f in SCRIPTS_SCR:
    put_new(E2W + "/scripts/" + f, open(SCR + "/" + f, "rb").read().replace(b"\r\n", b"\n"))
k2, k2lo, k2hi = mappt(json.load(open(SA + "/EVAL_K2" + TAG + "_fitthr.json", encoding="utf-8")))
k0, k0lo, k0hi = mappt(json.load(open(SA + "/EVAL_K0" + TAG + "_fitthr.json", encoding="utf-8")))
t2 = mappt(tipjson(E2T + "/stageA/EVAL_K2tail_fitthr.json"))[0]
t0 = mappt(tipjson(E2T + "/stageA/EVAL_K0tail_fitthr.json"))[0]
same_map = abs(k2 - t2) <= 1e-9 and abs(k0 - t0) <= 1e-9

# --- stage B ---
for f in READ_FILES:
    put_new(E2W + "/stageB/read/" + f, open(RD + "/" + f, "rb").read())
put_new(E2W + "/stageB/logs/ROLL_STATUS.txt", open(ROLLST, "rb").read())
put_new(E2W + "/stageB/logs/s6c_resume.log", open(S6LOG, "rb").read())
vj = json.load(open(RD + "/E2_VERDICT.json", encoding="utf-8"))
cl = {c["id"]: c for c in vj["clauses"]}
st = {k: cl[k]["status"] for k in ("a", "b", "c", "d", "e")}
verdict = vj["verdict"]
bk2, bk0 = cl["b"]["per_seed"]["0"]["k2"], cl["b"]["per_seed"]["0"]["k0"]
b_equal = abs(bk2 - TAIL_B[0]) <= 1e-6 and abs(bk0 - TAIL_B[1]) <= 1e-6
d_detail = {s: cl["d"]["per_seed"][s].get("detail", "") for s in ("0", "1")}
md = open(RD + "/E2_VERDICT.md", encoding="utf-8").read().replace("\r\n", "\n")
first = md.split("\n", 1)[0].strip()
tab = [ln for ln in md.split("\n") if ln.startswith("|")]
named = [ln for ln in md.split("\n") if ln.startswith("NO-GO names") or ln.startswith("GO ")][:6]

# (d) rows: every row separated worse at tail OR here, both seeds, tail vs this read (paired K2DC - K0, T1, 70 ep)
fam = {s: json.load(open(RD + "/families_" + s + ".json", encoding="utf-8"))["rows"] for s in ("s0", "s1")}
tfam = {s: tipjson(E2T + "/stageB/read/families_" + s + ".json")["rows"] for s in ("s0", "s1")}


def worse(q):
    if not isinstance(q, dict) or not q.get("separated"):
        return False
    return (q["delta"] > 0) if q.get("better") == "lower" else (q["delta"] < 0)


rowset = []
for F in ("LONGITUDINAL", "LATERAL", "TACTICAL"):
    for m in fam["s0"][F]:
        if any(worse(src[s][F].get(m)) for src in (fam, tfam) for s in ("s0", "s1")) and (F, m) not in rowset:
            rowset.append((F, m))


def cell(q):
    if not isinstance(q, dict) or "delta" not in q:
        return "n/a"
    return "%+.4f [%+.4f, %+.4f]%s" % (q["delta"], q["lo"], q["hi"], " **WORSE**" if worse(q) else (" better" if q.get("separated") else ""))


dtab = ("| family / row | tail s0 | " + MODE + " s0 | tail s1 | " + MODE + " s1 |\n|---|---|---|---|---|\n"
        + "".join("| %s / `%s` | %s | %s | %s | %s |\n" % (F, m, cell(tfam["s0"][F].get(m)), cell(fam["s0"][F].get(m)),
                                                           cell(tfam["s1"][F].get(m)), cell(fam["s1"][F].get(m))) for F, m in rowset))
lat = fam["s0"]["LATERAL"]["LAT_cross_mae_m"]
tlat = tfam["s0"]["LATERAL"]["LAT_cross_mae_m"]
both_up = lat["a_mean"] > tlat["a_mean"] and lat["b_mean"] > tlat["b_mean"]
gap, tgap = lat["a_mean"] - lat["b_mean"], tlat["a_mean"] - tlat["b_mean"]

if MODE == "plan":
    if not b_equal or not same_map:
        outcome = ("ADDENDUM 1 CHECK FAILS: (b) or the stage-A map point differs from E2-TAIL, so perception was not untouched. "
                   "This is a tool or read defect, not a result; diagnose before reading (d).")
    elif st["d"] == "PASS" and st["a"] == "PASS" and st["c"] == "PASS" and st["e"] == "PASS":
        outcome = ("committed outcome 1 (WISE-PLAN passes (d); (a), (c), (e) pass; (b) equals E2-TAIL): planner drift carries the lateral "
                   "deficit. Next: the planner-only L2-SP anchor (own G-DVB entry + own prereg; launch is the PI's call), reported beside the "
                   "planner-interpolated K2DC as a zero-training candidate config.")
    elif st["d"] == "PASS":
        outcome = ("committed outcome 3 (WISE-PLAN passes (d) but breaks " + ", ".join("(%s)" % k for k in ("c", "e") if st[k] != "PASS")
                   + "): report the trade; do not propose the interpolated config; diagnose that clause first.")
    else:
        outcome = ("committed outcome 2 (WISE-PLAN fails (d)): planner drift alone does not carry the lateral deficit. Next: WISE-ENC "
                   "(S_enc = core.encoder, alpha 0.5), started on Thor 2026-11-17 11:29:32Z from the families rows; this verdict file confirms "
                   "the (d) FAIL it was started on.")
    if both_up:
        outcome += (" Outcome-4 condition met (both arms' s0 cross-track above their tail values): alpha 0.5 may be too far; not a reason "
                    "to tune alpha on DEV.")
else:
    if st["d"] == "PASS" and st["a"] == "PASS":
        outcome = ("prereg s4 outcome 2, WISE-ENC branch A (WISE-ENC passes (d) with (a) passing): trunk drift carries the lateral deficit. "
                   "The 0.02 share target is the lever; the PI approved it 2026-11-17 (\"1 i approve it\") CONDITIONAL on exactly this read "
                   "(WISE-PLAN failed (d), WISE-ENC passes (d)) -> the K2DC02 launch sequence starts (records queue 22, gate, ~20.5 Thor-h).")
    elif st["d"] == "PASS":
        outcome = ("WISE-ENC passes (d) but (a) FAILS: not the pre-registered trunk-drift branch (it needs (a) still passing). Report to the "
                   "PI; the conditional K2DC02 approval does not fire on this read.")
    else:
        outcome = ("prereg s4 outcome 2, WISE-ENC branch B (WISE-ENC fails (d) as well): neither half alone carries the lateral deficit. "
                   "Reported; the 0.02 arm stays the pre-registered next arm, but the PI's conditional approval does NOT fire on this read -> PI.")
    if both_up:
        outcome += (" Outcome-4 condition met (both arms' s0 cross-track above their tail values): alpha 0.5 may be too far; not a reason "
                    "to tune alpha on DEV.")
extra = [k for k in ("a", "b", "c", "e") if st[k] != "PASS"]
if extra:
    outcome += " Other clauses not PASS: " + ", ".join("(%s) %s" % (k, st[k]) for k in extra) + " (reported as written)."

readme = ("# E2-WISE " + UP + ": the E2 rule on " + ("PLANNER" if MODE == "plan" else "TRUNK (core.encoder)") + "-interpolated weights, "
          "K2DC vs K0 -- verdict " + verdict + "\n\n"
          "Pre-registered before any E2-WISE number: `../../PREREG_E2WISE_2026-11-17.md` (landed ad9f1cbe) + `../../PREREG_E2WISE_2026-11-17_ADDENDUM1.md`\n"
          "(landed 5f647742). Weights: `stack/scripts/weight_interp.py` (landed 20ab9f54): theta = theta_init + 0.5 (theta_tail - theta_init) on S_"
          + MODE + ", every other tensor copied from `model_ema_tail5k6k.pt`; init `/home/nvidia/r8ns/ckpt/ckpt_ema_48852.pt`. Both arms get the same\n"
          "S and alpha. Zero training. DEV = eval139 even half, 70 episodes. One training run per arm (`H-ESTIM-SEED-1`): a separated interval\n"
          "answers another draw of EPISODES only. Interpolated file md5: K2DC " + str(md5["K2DC"]) + ", K0 " + str(md5["K0"]) + " (every dump.json\n"
          "records it, `stageA/dump_headers/`).\n\n"
          "## Stage A -- clause (a), map primary `nf_swedge_prec_lv_5_20` at each arm's OWN FIT thresholds (T0, MEASURED)\n\n"
          "| arm | E2-TAIL | this read |\n|---|---|---|\n"
          "| K2DC | " + f4(t2) + " | **" + f4(k2) + "** [" + f4(k2lo) + ", " + f4(k2hi) + "] |\n"
          "| K0 | " + f4(t0) + " | **" + f4(k0) + "** [" + f4(k0lo) + ", " + f4(k0hi) + "] |\n\n"
          + ("* Perception check (ADDENDUM 1): map points equal E2-TAIL to 1e-9: " + ("YES" if same_map else "NO") + "; clause (b) equals E2-TAIL to 1e-6: "
             + ("YES" if b_equal else "NO") + " (K2DC %.6f vs %.6f, K0 %.6f vs %.6f). WISE-PLAN copies `core.encoder`, `_perception` and "
             "`core.agent_head` bit-exactly, so it cannot GO while (b) reproduces the tail tie; (d) decides the lever.\n" % (bk2, TAIL_B[0], bk0, TAIL_B[1])
             if MODE == "plan" else
             "* WISE-ENC changes `core.encoder`, so stage A is a real re-read: (a) " + st["a"] + ", (b) " + st["b"] + " (K2DC %.6f, K0 %.6f; E2-TAIL "
             "%.6f / %.6f).\n" % (bk2, bk0, TAIL_B[0], TAIL_B[1])) + "\n"
          "## Stage B -- the read's verdict (quoted, unedited from `stageB/read/E2_VERDICT.md`)\n\n" + first + "\n\n" + "\n".join(tab) + "\n\n"
          + "\n".join(named) + "\n\n"
          "* (d) seed 0: " + d_detail["0"] + "\n* (d) seed 1: " + d_detail["1"] + "\n\n"
          "## (d), row by row: E2-TAIL vs this read (paired K2DC - K0, episode-cluster bootstrap, T1, 70 episodes)\n\n"
          "Every row separated worse at either seed in either read. WORSE = separated in the bad direction.\n\n" + dtab + "\n"
          "## Secondary readout: K2DC seed-0 `LAT_cross_mae_m` (decides nothing alone)\n\n"
          "* K2DC " + f4(lat["a_mean"]) + " m (tail " + f4(tlat["a_mean"]) + "), K0 " + f4(lat["b_mean"]) + " m (tail " + f4(tlat["b_mean"]) + "); gap "
          + sg(gap) + " (tail " + sg(tgap) + "). Both arms above their tail values: " + ("YES" if both_up else "NO") + ".\n"
          "* STRATEGIC: the read prints PAUSED (its PI item-25 text). The PI ended that pause 2026-11-16; in these arms the strategic modules are\n"
          "  equal to the init (S_plan excludes them), so there is no trained strategic row to compare.\n\n"
          "## Next (Rule Zero)\n\n* " + outcome + "\n* E2, E2-RAW and E2-TAIL stand as written.\n\n"
          "## Provenance\n\n* Stage A files pulled from Thor `/home/nvidia/r8ns/r9e2/runs/E2WISE_" + UP + "/stageA/`, sha256-equal. Rasters, perc_det.pkl and\n"
          "  the interpolated .pt files are not banked (re-creatable: weight_interp.py on Thor's runs/r9e2-*/model_ema_tail5k6k.pt, md5 above). Stage B read files are\n"
          "  byte-exact copies of the dev-box S6c driver's pull (sha256-verified against Thor `runs/E2WISE_" + UP + "/stageB/`).\n")
put_new(E2W + "/README_E2WISE_" + UP + ".md", readme.encode("utf-8"))
sums = [hashlib.sha256(open(H + "/code/" + rel, "rb").read()).hexdigest() + "  " + rel[len(E2W) + 1:] for rel, _ in expect]
put_new(E2W + "/SHA256SUMS", ("\n".join(sorted(sums, key=lambda s: s[66:])) + "\n").encode("utf-8"))

if not NOGOALS:
    base_blob = git("rev-parse", TIP + ":" + GOALS).decode("ascii").strip()
    raw = git("cat-file", "blob", base_blob)
    crlf = b"\r\n" in raw[:20000]
    lines = raw.decode("utf-8").replace("\r\n", "\n").split("\n")
    cls = ", ".join("(%s) %s" % (k, st[k]) for k in ("a", "b", "c", "d", "e"))
    vtxt = ("**WISE-" + UP + " VERDICT (MEASURED 2026-11-17, read by the pre-registered pipeline; " + E2W + "/README_E2WISE_" + UP + ".md): "
            + verdict + " -- " + cls + ". Stage A (T0, DEV, own FIT thresholds): K2DC " + f4(k2) + " vs K0 " + f4(k0) + ". Secondary: s0 cross-track "
            "K2DC " + f4(lat["a_mean"]) + " / K0 " + f4(lat["b_mean"]) + " m (tail " + f4(tlat["a_mean"]) + " / " + f4(tlat["b_mean"]) + "), paired "
            + sg(lat["delta"]) + " [" + sg(lat["lo"]) + ", " + sg(lat["hi"]) + "]" + (" separated" if lat["separated"] else " not separated")
            + ". " + outcome + " One training run per arm; zero training.**")
    if MODE == "plan":
        assert not [l for l in lines if l.startswith("| **H-R9-E2-WISE -- ")]
        idx = [i for i, l in enumerate(lines) if l.startswith("| **H-R9-E2-TAIL -- ")]
        assert len(idx) == 1, idx
        row = ("| **H-R9-E2-WISE -- E2-TAIL's lateral deficit (K2DC - K0 separated worse on LATERAL at seeds 0 and 1) is carried by PLANNER drift "
               "(WISE-PLAN: the planner subset pulled halfway back to the init passes (d)) or by TRUNK drift (WISE-ENC: core.encoder pulled halfway "
               "passes (d)); the answer picks the training lever (planner L2-SP anchor vs the 0.02 share target). Zero training, alpha 0.5 fixed, "
               "both arms identical processing (pre-registered `" + E2R + "/PREREG_E2WISE_2026-11-17.md` + ADDENDUM1, written before any "
               "E2-WISE number)** | " + vtxt + " |")
        lines.insert(idx[0] + 1, row)
    else:
        idx = [i for i, l in enumerate(lines) if l.startswith("| **H-R9-E2-WISE -- ")]
        assert len(idx) == 1, idx
        old = "**WISE-PLAN VERDICT"
        assert lines[idx[0]].count(old) == 1
        lines[idx[0]] = lines[idx[0]].replace(old, vtxt + " Earlier: " + old)
    out = "\n".join(lines)
    out = out.replace("\n", "\r\n") if crlf else out
    p = H + "/code/" + GOALS
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(out.encode("utf-8"))
    expect.insert(0, (GOALS, base_blob))
open(H + "/expect.tsv", "wb").write(("".join(a + "\t" + b + "\n" for a, b in expect)).encode("utf-8"))
print("HOLD OK", len(expect), "files; verdict", verdict, st, "| map", f4(k2), f4(k0), "same_map", same_map, "b_equal", b_equal,
      "| s0 cross", f4(lat["a_mean"]), f4(lat["b_mean"]), "both_up", both_up)
print("ROWS", len(rowset))
print("OUTCOME", outcome)
