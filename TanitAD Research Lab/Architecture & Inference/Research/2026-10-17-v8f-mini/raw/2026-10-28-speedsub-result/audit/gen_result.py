"""gen_result.py -- builds RESULT_V8F_SPEEDSUB.md, REGISTER_ROW.md and NUMBERS_AUDIT.md for the SPEEDSUB read.

Every number in the RESULT is read here from SUBREAD.json / REDERIVE.json / the family JSONs / the per-mode eval JSONs (or parsed from the banked logs), formatted once,
and logged to NUMBERS_AUDIT.md with its source path. Derived numbers carry their arithmetic. The decision of PREREG_V8F_SPEEDSUB section 5 is recomputed here from the
reader's own rows (coded below from the PREREG's text) and the script stops if it disagrees with SUBREAD.json, with REDERIVE.json (the independent re-derivation from
the per-window dumps), or with any control flag.
run: python -I -X utf8 gen_result.py     (one process, no GPU)
"""
import datetime
import hashlib
import json
import re
import subprocess
import sys

sys.dont_write_bytecode = True

OUT = "E:/Projects/TanitAD-artifacts/dagland/speedsub_result/"
W = "F:/TanitAD-scratch/speedsub/"
SR = "F:/TanitAD-scratch/speedread/"
WAIT = "F:/TanitAD-scratch/speedsub_after_t1.log"
TIP = "ea5e3d3bd72250b28af48ae06b6b046c4f446628"
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
V8 = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
PKG = "raw/2026-10-28-speedsub-result/"
MINUS = "\u2212"
NDASH = "\u2013"

rdb = lambda p: open(p, "rb").read()
shaf = lambda p: hashlib.sha256(rdb(p)).hexdigest()
shab = lambda b: hashlib.sha256(b).hexdigest()
jl = lambda p: json.loads(rdb(p).decode("utf-8"))


def blob(path):
    r = subprocess.run(GIT + ["show", f"{TIP}:{path}"], capture_output=True)
    assert r.returncode == 0, path
    return r.stdout


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
    return f"{100 * x:.{nd}f} %"


def th(n):
    return f"{n:,}"


def sci(x):
    return f"{x:.2e}"


# ---------------------------------------------------------------------------------------------------------------- sources and pins
SUBP = W + "subread/SUBREAD.json"
SUB_SHA = shaf(SUBP)
assert SUB_SHA == "1fe7fcb366e8123721d82891408c6d3c15948745173deb87f1ddeb5a4c53826d", SUB_SHA
SUB = jl(SUBP)
RDP = OUT + "REDERIVE.json"
RD_SHA = shaf(RDP)
RD = jl(RDP)
PRE = blob("Project Steering/PREREG_V8F_SPEEDSUB.md")
PRE_SHA = shab(PRE)
assert PRE_SHA.startswith("1951a240"), PRE_SHA
PRET = PRE.decode("utf-8")
SRREAD = blob(V8 + "raw/2026-10-28-speedread/read/READ.json")
assert shab(SRREAD).startswith("46543c39") and SRREAD == rdb(SR + "read/READ.json")
SRR = json.loads(SRREAD.decode("utf-8"))
assert shab(blob("stack/scripts/v8f_mini_eval_speedread.py")) == "a95f4fb14da3fd8f6f89e55dfcb23d5d26a81b09dd24285aba394f36546634ca"
assert shaf("F:/TanitAD-scratch/speedsub_tmp/tree/stack/scripts/v8f_mini_eval_speedread.py") == "a95f4fb14da3fd8f6f89e55dfcb23d5d26a81b09dd24285aba394f36546634ca"
assert shaf("F:/TanitAD-scratch/speedsub_tmp/tree/stack/scripts/families_longlat_sub.py").startswith("f981ea7a")
assert shab(blob(V8 + "raw/2026-10-28-speedsub-draft/families/families_longlat_sub.py")).startswith("f981ea7a")
assert shaf(SR + "fit/readout.pt") == "a205609ab47325a0ad9c0141be8ae3a0fec653d59c6db17ec1cf0d460bf2316d"
FIT = jl(SR + "fit/fit.json")
assert FIT["readout_sha256"].startswith("a205609a") and shaf(SR + "fit/fit.json").startswith("c2e51ac2")
# FINDING_1 (the prerequisites agent's attribution refits; read here, not re-run)
F1P = "E:/Projects/TanitAD-artifacts/dagland/speedsub_t1_prep/FINDING_1_ATTRIBUTION.md"
ATTP = "F:/TanitAD-scratch/speedsub_t1_prep/refit_all.json"
assert shaf(F1P).startswith("e89af4a0") and shaf(ATTP).startswith("0f830665")
F1T = rdb(F1P).decode("utf-8")
ATT = jl(ATTP)
ATT_DES = {"full": 22, "b_nonav": 17, "c_nolimit": 17, "a_p10f_fok": 12}
ATT_MEAN = {k: sum(ATT["designs"][k]["mlp"]["holdout_rmse_sd"]) / 5 for k in ATT_DES}
for k, d in ATT_DES.items():
    assert ATT["designs"][k]["d"] == d
assert [f"{ATT_MEAN[k]:.5f}" for k in ("full", "b_nonav", "c_nolimit", "a_p10f_fok")] == ["0.24751", "0.26313", "0.28032", "0.30849"]
for t in ("0.24751", "0.26313", "0.28032", "0.30849", "D-SPEEDBAND-EGO-ECHO-1", "--limit causal", "846 clips"):
    assert t in F1T, t
assert abs(ATT_MEAN["full"] - sum(FIT["mlp_ego"]["holdout_rmse_sd"]) / 5) < 1e-7           # the control refit reproduces fit.json
assert ATT_MEAN["full"] < ATT_MEAN["b_nonav"] < ATT_MEAN["c_nolimit"] < ATT_MEAN["a_p10f_fok"]
_gc = blob("Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8").split("\n")
assert "D-SPEEDBAND-EGO-ECHO-1" in _gc[11517] and "0.9702" in _gc[11517] and "DECLARATION, not refusal" in _gc[11517] and "never a training signal" in _gc[11517]
_t1 = blob("stack/scripts/v8f_t1_eval.py").decode("utf-8")
assert "speed bucket from the human's realised future max speed (D-SPEEDBAND-EGO-ECHO-1)" in _t1 and "def causal_bucket" in _t1 and "Reads NOTHING after row j" in _t1
SEEDS = ("0", "1")
MODES = ("own", "identity", "gt", "ego", "egoshuf")
EJ = {(s, m): jl(f"{W}eval_s{s}/eval_M2b-mixo_s{s}-speedsub-{m}.json") for s in SEEDS for m in MODES}
LL = {s: jl(f"{W}families/longlat_s{s}.json") for s in SEEDS}
PF = {(a, s): jl(f"{W}families/pairfam_{a}_s{s}.json") for a in ("ego", "gt") for s in SEEDS}
XPIN = {"0": ("c6a628762790b4fc3f7634867fc15b8167b035692ff518203c9fc0b3c126250d", "C:/Users/Admin/box2_ship/logs/v8f_mixo/runs/M2b_s0_mixo/ckpt_last.pt"),
        "1": ("68a0f3356c37b25ce15b5f860f8dc919e23efc8b6bf200a2997f9ad6b521e915", "C:/Users/Admin/box2_ship/logs/v8f_kindx_s1/runs/M2b_s1_mixo/ckpt_last.pt")}
for s in SEEDS:
    for m in MODES:
        assert EJ[(s, m)]["speedread"]["x_sha256"] == XPIN[s][0] and EJ[(s, m)]["ckpt"] == XPIN[s][1], (s, m)
        assert EJ[(s, m)]["speedread"]["readout_sha256"] == SUB["readout_sha256"]

# ---------------------------------------------------------------------------------------------------------------- the decision, recomputed (PREREG section 5)
F_M, F_PASS = SUB["F_m"], SUB["F_pass_m"]
assert F_M == 0.1207 and abs(F_PASS - F_M / 2.0) < 1e-12 and f"{F_PASS:.5f}" == "0.06035"
sep_b = lambda e: e["hi"] < 0
sep_w = lambda e: e["lo"] > 0
ROW = {s: SUB["rows"][s] for s in SEEDS}
CT = SUB["controls"]
PER_SEED_FLAGS = ["S1_identity_eq_own", "S2_nonspeed_untouched", "S3_tick_mirror", "S4_eid_aligned", "S5_own_eq_speedread_own", "S6_gt_eq_speedread_gt",
                  "S7_ego_path_eq_identity_pred", "S7b_identity_pred_eq_speedread_banked", "S10_substitution_reaches_O", "S8_readout_sha", "S9_shuffle_registered_and_aligned"]
for s in SEEDS:
    assert sorted(CT[s]) == sorted(PER_SEED_FLAGS) and all(v is True for v in CT[s].values()), s
assert CT["S11_train_parity"] is True and CT["S12_disjoint"] is True and CT["S13_selection_rule_picks_mlp_ego"] is True
assert sorted(CT["S6b_gt_delta_eq_speedread_read_json"]) == ["0", "1"] and all(CT["S6b_gt_delta_eq_speedread_read_json"].values())
NFLAG = 2 * len(PER_SEED_FLAGS) + 3 + 2
assert NFLAG == 27
assert RD["controls_all_pass_recomputed"] is True and RD["mismatches_vs_SUBREAD_json"] == [] and RD["equals_SUBREAD_json_outcome"] is True


def decide(rows, ok):
    if not ok or sorted(rows) != ["0", "1"]:
        return "REFUSED"
    if not all(sep_w(rows[s]["egoshuf"]) for s in SEEDS):
        return "REFUSED"
    if any(sep_w(rows[s]["ego"]) for s in SEEDS):
        return "HARMS"
    if all(sep_b(rows[s]["ego"]) and rows[s]["ego"]["delta"] <= -F_PASS for s in SEEDS):
        return "PASS"
    return "NULL"


MY_OUTCOME = decide(ROW, True)
assert MY_OUTCOME == SUB["outcome"] == "PASS" == RD["decision_rounded_delta"] == RD["decision_unrounded_point"] and SUB["detail"] is None
for s in SEEDS:
    r = ROW[s]
    assert r["regression_arm_separated_worse"] is True and sep_w(r["egoshuf"])
    assert abs(r["capture"] - r["ego"]["delta"] / r["gt"]["delta"]) < 1e-12
    assert sep_b(r["ego"]) and r["ego"]["delta"] <= -F_PASS and not sep_w(r["ego"]) and sep_b(r["gt"])
    for k in ("ego", "egoshuf", "gt"):                                    # the independent re-derivation equals the reader's rows
        for q in ("delta", "lo", "hi"):
            assert round(RD["seeds"][s]["rows"][k][q], 4) == round(r[k][q], 4), (s, k, q)
    mm = {m: EJ[(s, m)]["raw"]["traj"]["ade6_m"]["mean"] for m in MODES}
    assert abs(round(mm["ego"] - mm["own"], 4) - r["ego"]["delta"]) <= 1.0001e-4 and abs(round(mm["gt"] - mm["own"], 4) - r["gt"]["delta"]) <= 1.0001e-4
    assert abs(round(mm["egoshuf"] - mm["own"], 4) - r["egoshuf"]["delta"]) <= 1.0001e-4 and mm["identity"] == mm["own"]
# the table's other rows, to name which conditions this run meets
COND = {"HARMS": any(sep_w(ROW[s]["ego"]) for s in SEEDS), "PASS": True, "REFUSED": False}
assert not COND["HARMS"]
# the tip's SPEEDREAD READ.json has the same gt rows (S6b by quotation)
for s in SEEDS:
    for q in ("delta", "lo", "hi"):
        assert abs(SRR["rows"][s]["gt"][q] - ROW[s]["gt"][q]) <= 5e-4

# ---------------------------------------------------------------------------------------------------------------- run.out
RUN = [l for l in rdb(W + "run.out").decode("utf-8").splitlines() if l.strip()]
TS = {}


def tstamp(line):
    m = re.match(r"(\d\d)-(\d\d) (\d\d:\d\d:\d\d) (.*)", line)
    return datetime.datetime(2026, int(m[1]), int(m[2]), *[int(x) for x in m[3].split(":")]), m[3], m[4]


T0S = tstamp(RUN[0])
assert "run start (threads 4)" in RUN[0] and RUN[-1].endswith("ZZSPEEDSUB-RUN-DONE PASS") and not any("STOP" in l for l in RUN)
RT = {}
for l in RUN:
    dt, hms, rest = tstamp(l)
    RT[rest] = (dt, hms)
dur = lambda a, b: int((RT[b][0] - RT[a][0]).total_seconds())
d_s0 = dur("evals_s0: start", "evals_s0: done ([ok] M2b-mixo_s0-speedsub-egoshuf)")
d_s1 = dur("evals_s1: start", "evals_s1: done ([ok] M2b-mixo_s1-speedsub-egoshuf)")
d_sr = dur("subread: start", "subread: done (ZZSPEEDSUB-DONE PASS)")
d_all = int((tstamp(RUN[-1])[0] - T0S[0]).total_seconds())
d_fam = int((tstamp(RUN[-1])[0] - RT["subread: done (ZZSPEEDSUB-DONE PASS)"][0]).total_seconds())
mm_ss = lambda sec: (f"{sec // 60} min {sec % 60:02d} s" if sec >= 60 else f"{sec} s")
WL = [l for l in rdb(WAIT).decode("utf-8").splitlines() if l.strip()]
assert any("T1 DONE and 7138 MB available: launching run_speedsub.sh" in l for l in WL) and WL[-1].endswith("ZZSPEEDSUB-RUN-DONE PASS")
PF_LOG = [l for l in RUN if l.endswith("rc=0") and "pairfam" in l]
assert len(PF_LOG) == 4
PRE_LOG = [l for l in RUN if "preflight_s0: done" in l][0]
PRE_KEY = "preflight_s0: done (" + PRE_LOG.split("done (")[1]
assert "S7b dev 0 (<=1e-5)  S7 dev 0 (<=1e-5)  moved 1.0000 (>=0.99)" in PRE_LOG

# ---------------------------------------------------------------------------------------------------------------- the family rows
LLK = {"ade6_m": "ADE6 (m)", "ade2_m": "ADE2 (m)", "speed_mae_mps": "speed MAE (m/s)", "along_mae_m": "along-track MAE (m)", "accel_mae_mps2": "accel MAE (m/s\u00b2)",
       "heading_mae_deg": "heading MAE (deg)", "yaw_rate_mae_degps": "yaw-rate MAE (deg/s)", "curvature_mae_1pm": "curvature MAE (1/m)", "cross_mae_m": "cross-track MAE (m)",
       "L1_lat_accel_mae_mps2": "L1 lateral-accel MAE (m/s\u00b2)", "L2_curvature_mae_fast_1pm": "L2 curvature MAE, GT \u2265 5.2 m/s (1/m)",
       "L3_yaw_rate_mae_fast_degps": "L3 yaw-rate MAE, GT \u2265 5.2 m/s (deg/s)"}
PFK = [("tactical.plan_derived.lateral_decision.accuracy", "TAC-LAT (lateral-decision accuracy)"), ("tactical.plan_derived.longitudinal_decision.accuracy", "TAC-LON (longitudinal-decision accuracy)"),
       ("tactical.plan_derived.maneuver_5way_collapsed.accuracy", "5-way manoeuvre accuracy"), ("tactical.goal_setting.goal_point_error_m", "goal-point error (m)"),
       ("tactical.goal_setting.goal_long_bias_m", "goal longitudinal bias (m)"), ("tactical.goal_setting.goal_lat_bias_m", "goal lateral bias (m)"),
       ("tactical.goal_setting.goal_bearing_mae_deg", "goal bearing MAE (deg)"), ("tactical.goal_setting.goal_range_ratio", "goal range ratio"),
       ("tactical.heads.lateral.macro_f1", "lateral head macro-F1"), ("tactical.heads.longitudinal.macro_f1", "longitudinal head macro-F1"), ("tactical.heads.goal.macro_f1", "goal head macro-F1"),
       ("strategic.position_error_m.30", "S-POS@30 (m)"), ("strategic.position_error_m.50", "S-POS@50 (m)"), ("strategic.position_error_m.80", "S-POS@80 (m)"),
       ("strategic.heading_error_deg.30", "S heading@30 (deg)"), ("strategic.heading_error_deg.50", "S heading@50 (deg)"), ("strategic.heading_error_deg.80", "S heading@80 (deg)")]


def llcell(arm, s, k):
    e = LL[s]["contrasts"][f"{arm}-own"][k]
    return e, "lower"


def pfcell(arm, s, path):
    d = PF[(arm, s)]
    for p in path.split("."):
        d = d[p]
    assert d["status"] == "OK", (arm, s, path)
    return d, d["better"]


def tagof(e, better):
    if not e["separated"]:
        return NDASH
    if better is None:
        return "S+" if e["delta"] > 0 else "S" + MINUS
    good = (e["delta"] < 0) if better == "lower" else (e["delta"] > 0)
    return "B" if good else "W"


def cellstr(e, better, src):
    t = tagof(e, better)
    s = f"{sg(e['delta'])} [{sg(e['lo'])}, {sg(e['hi'])}] {t}"
    s = f"**{s}**" if t != NDASH else s
    return aud(s, src + ".{delta,lo,hi,separated,better}")


for s in SEEDS:      # the ADE6 row of the family tool equals the reader's row (LL-2) and the pairfam / longlat controls hold
    assert LL[s]["controls_ok"] is True and PF[("ego", s)]["controls_ok"] is True and PF[("gt", s)]["controls_ok"] is True
    for k in ("LL-1_identity", "LL-2_ade6_equals_READ_json", "LL-3_reproduces_dump_family_values"):
        assert LL[s]["controls"][k]["ok"] is True
    for a in ("ego", "gt"):
        e = LL[s]["contrasts"][f"{a}-own"]["ade6_m"]
        assert (e["delta"], e["lo"], e["hi"]) == (ROW[s][a]["delta"], ROW[s][a]["lo"], ROW[s][a]["hi"])
    assert LL[s]["unavailable"]["same_in_all_dumps_of_this_seed"] is True
    assert LL[s]["unavailable"]["distance_keeping"]["status"] == "UNAVAILABLE" and LL[s]["unavailable"]["distance_keeping"]["n"] == 0
    assert LL[s]["unavailable"]["anti_echo"]["status"] == "UNAVAILABLE" and LL[s]["unavailable"]["anti_echo"]["n"] == 0 and LL[s]["unavailable"]["anti_echo"]["n_windows_available"] == 2399
    assert "no lead-agent track supplied" in LL[s]["unavailable"]["distance_keeping"]["reason"] and "no ego speed at t0 in the window dict" in LL[s]["unavailable"]["anti_echo"]["reason"]
    assert LL[s]["n_windows"] == 2399 and PF[("ego", s)]["n_windows"] == 2399 and PF[("ego", s)]["n_episodes"] == 141 and PF[("ego", s)]["n_boot"] == 2000
    assert PF[("ego", s)]["strategic_family"].startswith("ACTIVE for v8F-mini")

# ---------------------------------------------------------------------------------------------------------------- document
L = []


def P(*xs):
    L.extend(xs)


def hdr(t):
    SEC[0] = t
    P("", t, "")


def yn(b):
    return "yes" if b else "no"


SEC[0] = "header"
sub_tier, sub_tier_gt, sub_tier_reg, sub_est = SUB["tier"], SUB["tier_gt"], SUB["tier_regression"], SUB["estimator"]
assert "B=2000, seed 0" in sub_est and "another draw of EPISODES" in sub_est
P("# RESULT_V8F_SPEEDSUB: does the banked ego-only read-out, put into the five speed cells O reads, cut O's ADE6? Outcome **PASS** (a screening result at T1-class open loop)")
P("")
P("Claim row `H-V8F-SPEEDSUB-1`. Pre-registration `Project Steering/PREREG_V8F_SPEEDSUB.md` (sha256 `" + PRE_SHA[:8] + "…` at tip `ea5e3d3b`; no amendment; promoted by the Master Mind before any SPEEDSUB number)."
  " Instrument `stack/scripts/v8f_mini_eval_speedread.py`, sha256 `a95f4fb1…` (the landed blob and the run tree's copy hash to it, compared here)."
  f" Run 2026-10-28 {RT['run start (threads 4)'][1]} to {tstamp(RUN[-1])[1]} ({mm_ss(d_all)}; dev-box CPU, 4 threads, `CUDA_VISIBLE_DEVICES=-1`, `logs/run.out`);"
  f" the pre-registered reader `subread` ran {RT['subread: start'][1]}–{RT['subread: done (ZZSPEEDSUB-DONE PASS)'][1]} and printed `ZZSPEEDSUB-DONE PASS`;"
  f" all **{NFLAG} control flags PASS** in `SUBREAD.json` (sha256 `{SUB_SHA[:8]}…`). The outcome word was not copied: section 1 recomputes it from the reader's rows,"
  " and `read/REDERIVE.json` recomputes the rows themselves from the per-window dumps. Every number below is MEASURED in this run unless marked INHERITED, ESTIMATED or DERIVED (arithmetic shown in `audit/NUMBERS_AUDIT.md`).")
aud("2026-10-28", "run date; logs/run.out stamps (MM-DD HH:MM:SS), year from the dump mtimes")
aud(f"{RT['run start (threads 4)'][1]} / {tstamp(RUN[-1])[1]}", "logs/run.out first line 'run start (threads 4)' and the last line (ZZSPEEDSUB-RUN-DONE PASS)")
aud(mm_ss(d_all), "DERIVED: last run.out stamp minus first ('run start' stamp 07:59:08)", f"{tstamp(RUN[-1])[1]} - {T0S[1]} = {d_all} s")
aud(f"{RT['subread: start'][1]}–{RT['subread: done (ZZSPEEDSUB-DONE PASS)'][1]}", "logs/run.out 'subread: start' and 'subread: done'")
aud(str(NFLAG), "SUBREAD.json controls: 11 per-seed flags x 2 seeds + S11 + S12 + S13 + S6b x 2", "22 + 3 + 2 = 27")
aud("a95f4fb1", "sha256 of stack/scripts/v8f_mini_eval_speedread.py at tip ea5e3d3b and of the run tree's copy (asserted)")
aud("ea5e3d3b", "the live tip of refs/heads/agent/arch-inf-20260803 (mirror C:/Users/Admin/tanitad-push), full sha ea5e3d3bd72250b28af48ae06b6b046c4f446628")
aud(PRE_SHA[:8], "sha256 of Project Steering/PREREG_V8F_SPEEDSUB.md at tip ea5e3d3b (git show)")
aud(SUB_SHA[:8], "sha256 of F:/TanitAD-scratch/speedsub/subread/SUBREAD.json")
aud("B = 2,000", "SUBREAD.json estimator (B=2000, seed 0); pairfam_ego_s{0,1}.json n_boot")
P("")
P(f"* **Tier.** SUBREAD.json `tier`: `{sub_tier}`. `tier_gt`: `{sub_tier_gt}`. `tier_regression`: `{sub_tier_reg}`. So this is **T1-class open loop**, never closed loop: no recorded"
  " future action, and the `gt` arm is the T0 ceiling and a control, never a result. The unmodified harness stamps each dump `raw.tier` = `T0-open-loop-own-held-outputs (trunk-probe grid)`;"
  " the tier of this read is the instrument's stamp above, the tier of SPEEDCAL's and SPEEDREAD's ADE6 rows. **ADE6 at this tier is not a capability claim** (PREREG section 6): no ADE6 here is a driving-performance number.")
P(f"* **Estimator.** SUBREAD.json `estimator`: `{sub_est}`. 141 episodes / 2,399 windows per seed, paired against `own`. It answers *\"would another draw of EPISODES say this?\"* and nothing else."
  " It does not answer *\"would another TRAINING RUN say this?\"* (`H-ESTIM-SEED-1`): **there is no replicate arm in this experiment**, so the two seeds (O's two training runs) carry that question at n = 2."
  " It does not answer *\"would another READ-OUT FIT say this?\"*: one fit (`readout.pt` sha256 `a205609a…`) serves both seeds. The planner is deterministic, so there is no inference-seed term.")
aud("141 episodes / 2,399 windows; n = 2", "PREREG section 3; pairfam_ego_s{0,1}.json n_windows, n_episodes; SUBREAD.json estimator")
aud("a205609a", "SUBREAD.json readout_sha256 = fit.json readout_sha256 = sha256 of F:/TanitAD-scratch/speedread/fit/readout.pt (asserted)")
P(f"* **Base and arms.** X_s = `M2b-mixo_s<s>` (the SPEEDCAL / KINDX / SPEEDREAD base), **not** DAG-OACC or DAG-ORC. s0 sha256 `{XPIN['0'][0][:8]}…`, s1 `{XPIN['1'][0][:8]}…`, each recorded in every dump"
  " (`speedread.x_sha256`). No model is trained. Modes per seed: `own` (X_s's cells), `identity` (own through the hook; control S1), `gt` (the GT cells where the window has a v9 row, else own; T0 ceiling),"
  " `ego` (**the arm**: `mlp_ego(ego block at the tick)`), `egoshuf` (the same cells shuffled across the 2,399 windows by `numpy.random.default_rng(20261029).permutation(2399)`; the deliberate-regression arm).")
aud(f"{XPIN['0'][0][:8]} / {XPIN['1'][0][:8]}", "eval_s{0,1}/*.json speedread.x_sha256 (all ten dumps), equal to PREREG section 1")
aud("20261029; 2399", "PREREG section 3 and eval_s{0,1}/eval_*-egoshuf.json speedread.hook.shuffle.{seed,n}")
for s in SEEDS:
    sh = EJ[(s, "egoshuf")]["speedread"]["hook"]["shuffle"]
    assert sh["seed"] == 20261029 and sh["n"] == 2399 and sh["n_seen"] == 2399 and sh["n_fixed"] == 0 and sh["perm_sha256"].startswith("64785f8a")
P("* **Declared open channel (`D-SPEEDBAND-EGO-ECHO-1`).** The PASS was read with the **logged `limit` lane**. The logged speed-limit bucket in the ego block is the bucket of the ego's own realised future max speed: the T1 tool (`stack/scripts/v8f_t1_eval.py`) lists it as an OPEN channel under `--limit logged`,"
  " and bin plus `v0` recovers it at R\u00b2 0.9702 (`Project Steering/GOALS_AND_CLAIMS.md:11518` at tip `ea5e3d3b`, read here; INHERITED claim). By PI ruling it is admissible as a DECLARED input, never a training signal, and T reads it too (PREREG section 2)."
  " `nav` is the other declared input in the block (a route input, admissible at training and inference by PI ruling). Section 4.5 says how much of the read-out's skill rests on each.")
aud("0.9702", "Project Steering/GOALS_AND_CLAIMS.md line 11518 at tip ea5e3d3b (D-SPEEDBAND-EGO-ECHO-1: bin + v0 recovers R^2 0.9702); INHERITED claim, line read by gen_result.py")
aud("11518", "line number of D-SPEEDBAND-EGO-ECHO-1 in Project Steering/GOALS_AND_CLAIMS.md at tip ea5e3d3b (asserted)")
P("* **Coverage.** The four families are in section 3, per family and per seed, never pooled. Gaps are named there, per family, with their reason and n: distance keeping and the anti-echo block (LONGITUDINAL), anchor selection and selected-vs-executed manoeuvre (TACTICAL)"
  " are UNAVAILABLE in the harness path (not computed, not passed), and the `egoshuf` control has no family run. The outcome above uses none of them.")

# ---------------------------------------------------------------------------------------------------------------- 1
hdr("## 1. The outcome, as the pre-registered table decides it")
SEC[0] = "1"
E, G, X = {s: ROW[s]["ego"] for s in SEEDS}, {s: ROW[s]["gt"] for s in SEEDS}, {s: ROW[s]["egoshuf"] for s in SEEDS}
mean = {(s, m): EJ[(s, m)]["raw"]["traj"]["ade6_m"]["mean"] for s in SEEDS for m in MODES}
margin = {s: round(-E[s]["delta"] - F_PASS, 5) for s in SEEDS}
assert margin["0"] > 0 and margin["1"] > 0 and f"{margin['0']:.4f}" == "0.0687" and f"{margin['1']:.4f}" == "0.0083"
share = {s: F_PASS / -G[s]["delta"] for s in SEEDS}


def rb(e, src):
    return aud(f"{sg(e['delta'])} [{sg(e['lo'])}, {sg(e['hi'])}]", src + ".{delta,lo,hi}")


def mstr(s, a, b):
    return aud(f"{un(mean[(s, a)])} → {un(mean[(s, b)])}", f"eval_s{s}/eval_M2b-mixo_s{s}-speedsub-{{{a},{b}}}.json raw.traj.ade6_m.mean")


P("| | s0 | s1 |", "|---|---|---|")
P("| **ARM** ADE6, `ego` − `own` (m) | **" + rb(E["0"], 'SUBREAD.json rows.0.ego') + "** | **" + rb(E["1"], 'SUBREAD.json rows.1.ego') + "** |")
P(f"| ADE6 mean, `own` → `ego` (m) | {mstr('0', 'own', 'ego')} | {mstr('1', 'own', 'ego')} |")
P(f"| separated-better (CI upper end < 0) | yes: upper end {aud(sg(E['0']['hi']), 'rows.0.ego.hi')} | yes: upper end {aud(sg(E['1']['hi']), 'rows.1.ego.hi')} |")
P(f"| bar: point ≤ −F/2 = −{un(F_PASS, 5)} m | yes: {aud(un(margin['0']), 'DERIVED', f"{un(-E['0']['delta'])} - {F_PASS} = {margin['0']:.5f}")} m beyond the bar"
  f" | yes: {aud(un(margin['1']), 'DERIVED', f"{un(-E['1']['delta'])} - {F_PASS} = {margin['1']:.5f}")} m beyond the bar |")
aud(un(F_PASS, 5), "SUBREAD.json F_pass_m", "F/2 = 0.1207 / 2 = 0.06035, computed in the instrument as F_M / 2.0")
P(f"| HARMS test: separated-worse (CI lower end > 0) | no: lower end {aud(sg(E['0']['lo']), 'rows.0.ego.lo')} | no: lower end {aud(sg(E['1']['lo']), 'rows.1.ego.lo')} |")
P("| **REGRESSION ARM** `egoshuf` − `own` (must be separated-worse) | " + rb(X["0"], 'rows.0.egoshuf') + ", separated-worse | " + rb(X["1"], 'rows.1.egoshuf') + ", separated-worse |")
P(f"| ADE6 mean, `own` → `egoshuf` (m) | {mstr('0', 'own', 'egoshuf')} | {mstr('1', 'own', 'egoshuf')} |")
P("| **CEILING** `gt` − `own` (T0, control only) | " + rb(G["0"], 'rows.0.gt') + f"; mean {mstr('0', 'own', 'gt')} | " + rb(G["1"], 'rows.1.gt') + f"; mean {mstr('1', 'own', 'gt')} |")
P(f"| CAPTURE = `ego` / (`gt` − `own`) | {aud(un(ROW['0']['capture']), 'rows.0.capture', '-0.1290 / -0.7164')} | {aud(un(ROW['1']['capture']), 'rows.1.capture', '-0.0686 / -0.6676')} |")
P(f"| F/2 as a share of the `gt` ceiling (derived) | {aud(pct(share['0']), 'DERIVED', '0.06035 / 0.7164')} | {aud(pct(share['1']), 'DERIVED', '0.06035 / 0.6676')} |")
P("")
P("Keys: `SUBREAD.json` `rows.<s>.ego | egoshuf | gt | capture | regression_arm_separated_worse`. The means are the dumps' `raw.traj.ade6_m.mean` (4 decimals), so a delta can differ from the difference of two printed means by 0.0001"
  f" (`egoshuf` − `own` at s0: {un(mean[('0', 'egoshuf')] - mean[('0', 'own')])} from the means, {sg(X['0']['delta'])} from the paired cells). \"Separated-better\" is the CI's upper end below 0; \"separated-worse\" is its lower end above 0 (PREREG section 5)."
  " F = 0.1207 m is the cross-seed spread of M2b (INHERITED, `RESULT_V8F_GRID.md`; the instrument's `F_M`, read back in SUBREAD.json `F_m`).")
aud("0.0001", "arithmetic note on 4-decimal means (as RESULT_V8F_SPEEDREAD.md section 1)")
aud(un(mean[("0", "egoshuf")] - mean[("0", "own")]), "DERIVED", f"{mean[('0', 'egoshuf')]} - {mean[('0', 'own')]}")
aud("0.1207", "SUBREAD.json F_m; PREREG section 1 (INHERITED from RESULT_V8F_GRID.md)")
P("")
P("The pre-registered conditions (PREREG section 5, REFUSED first, then HARMS, then PASS), restated against the rows. The reader's word in SUBREAD.json is the decision of record.")
P("")
P("| outcome | condition | this run |", "|---|---|---|")
P(f"| REFUSED | a control failed, a seed is missing, or `egoshuf` is not separated-worse at both seeds | no: all {NFLAG} flags true, both seeds present, `egoshuf` lower ends {aud(sg(X['0']['lo']), 'rows.0.egoshuf.lo')} / {aud(sg(X['1']['lo']), 'rows.1.egoshuf.lo')} |")
P("| HARMS | `ego` separated-worse at either seed | no: lower ends are below 0 at both seeds |")
P(f"| **PASS** | at **both** seeds: separated-better **and** `ego_s ≤ −{un(F_PASS, 5)}` m | **yes**: upper ends {sg(E['0']['hi'])} / {sg(E['1']['hi'])} and points {sg(E['0']['delta'])} / {sg(E['1']['delta'])} |")
P("| NULL | anything else | not reached |")
P("")
nxt = SUB["next"]
assert nxt.startswith("T1-kin zero-training substitution of the banked ego read-out on the DAG-OACC / DAG-ORC base, both seeds, under its own PREREG")
P(f"**PASS.** The script that built this document coded the table above from PREREG section 5 and applied it to the reader's rows: PASS, equal to SUBREAD.json `outcome` (`detail` null). `read/REDERIVE.json` (`read/rederive_sub.py`) recomputed the three paired rows"
  f" per seed (`ego`, `egoshuf`, `gt`, each against `own`) from the per-window ADE6 of the dumps with `taniteval.ci.paired_episode_cluster_bootstrap` called directly (B = 2,000, seed 0), applied the same table, and got PASS with identical delta, lo and hi to 4 dp on all {aud('18', 'REDERIVE.json: 3 rows x 3 values x 2 seeds', '3 x 3 x 2 = 18 values compared with SUBREAD.json to 4 dp')} values; the decision is the same whether the point is taken rounded or unrounded"
  f" (unrounded points {aud(sg(RD['seeds']['0']['rows']['ego']['unrounded_point']), 'REDERIVE.json seeds.0.rows.ego.unrounded_point')} / {aud(sg(RD['seeds']['1']['rows']['ego']['unrounded_point']), 'REDERIVE.json seeds.1.rows.ego.unrounded_point')} m).")
P("")
P(f"**Committed next, quoted from SUBREAD.json `next`:** *\"{nxt}\"*. Section 6 says what that requires and what is in progress.")
P("")
P("### How to read this PASS (reported as written; none of it changes the word)")
P("")
sl = {s: -E[s]["delta"] / F_PASS for s in SEEDS}
slq = {s: f"{sl[s]:.2f}" for s in SEEDS}
spread = E["0"]["delta"] - E["1"]["delta"]
hw = {s: (E[s]["hi"] - E[s]["lo"]) / 2 for s in SEEDS}
P("1. **It is a screening result, by the PREREG's own terms.** The arm was commissioned after SPEEDREAD's beside table (which holds eval-141 RMSE) was read; the selection rule was restated on TRAIN fold 0 only before any SPEEDSUB number"
  f" (control S13, section 2: mean fold-0 hold-out RMSE `mlp_ego` {aud(un(RD['S13_mean_fold0_holdout_rmse_sd']['mlp_ego']), 'REDERIVE.json S13_mean_fold0_holdout_rmse_sd.mlp_ego; fit.json mlp_ego.holdout_rmse_sd mean over 5 cells')} against `lin` {aud(un(RD['S13_mean_fold0_holdout_rmse_sd']['lin']), 'REDERIVE.json S13_mean_fold0_holdout_rmse_sd.lin')},"
  f" `lin_ego` {aud(un(RD['S13_mean_fold0_holdout_rmse_sd']['lin_ego']), 'REDERIVE.json S13_mean_fold0_holdout_rmse_sd.lin_ego')}, `mlp` {aud(un(RD['S13_mean_fold0_holdout_rmse_sd']['mlp']), 'REDERIVE.json S13_mean_fold0_holdout_rmse_sd.mlp')}). It is still the third read-out examined on these 141 episodes (`lin` and `mlp` were run in SPEEDREAD; `lin_ego` was not),"
  " so PREREG section 0 and section 11 apply: *a PASS is a statement about two O training runs and one read-out fit at T1-class open loop; it is not a lever claim for any trained arm and not a driving result.*")
P(f"2. **s1 clears the bar by {aud(un(margin['1']), 'DERIVED', 'see section 1')} m only** (point {sg(E['1']['delta'])} against −{un(F_PASS, 5)}; {aud(slq['1'], 'DERIVED', '0.0686 / 0.06035')} times the bar). At both seeds the CI's upper end ({sg(E['0']['hi'])} / {sg(E['1']['hi'])}) lies **above** −{un(F_PASS, 5)}, so the intervals do not exclude an effect smaller than the bar."
  " The registered table tests the point estimate against the bar and the interval's sign against 0, not the interval against the bar; it is reported as registered. s0 is "
  f"{aud(slq['0'], 'DERIVED', '0.1290 / 0.06035')} times the bar.")
P(f"3. **The PREREG's expectation about which condition binds did not hold.** Section 1 of the PREREG expected the separation condition to bind before the point bar, because SPEEDREAD's CI half-widths were 0.075 and 0.055. Here the CI half-widths are {aud(un(hw['0'], 3), 'DERIVED', '(-0.0548 - -0.2107) / 2 = 0.07795')} (s0) and {aud(un(hw['1'], 3), 'DERIVED', '(-0.0232 - -0.1151) / 2 = 0.04595')} (s1);"
  f" at s1 the point bar binds ({un(margin['1'])} m of margin) before separation (upper end {sg(E['1']['hi'])}, {aud(un(-E['1']['hi']), 'DERIVED', '0 - -0.0232')} m of margin). That sentence was an ESTIMATED prior, not a criterion.")
aud("0.075 and 0.055", "PREREG section 1 (quoting SPEEDREAD's `read` CI half-widths 0.075 s0 / 0.055 s1; INHERITED)")
P(f"4. **The two seeds' effects differ by {aud(un(abs(spread)), 'DERIVED', '-0.1290 - -0.0686 = -0.0604')} m** ({sg(E['0']['delta'])} against {sg(E['1']['delta'])}), about half of F. That is the only look this experiment has at training-run variance (n = 2, no replicate arm); it is a description, not an estimate of it."
  f" CAPTURE of the T0 ceiling is {un(ROW['0']['capture'])} at s0 and {un(ROW['1']['capture'])} at s1: `ego` recovers {pct(ROW['0']['capture'], 0)} and {pct(ROW['1']['capture'], 0)} of what the GT cells would give, and O still has most of the 0.7 m left on the table.")
aud(pct(ROW['0']['capture'], 0) + " and " + pct(ROW['1']['capture'], 0), "DERIVED from rows.<s>.capture", "0.1801 -> 18 %, 0.1028 -> 10 %")
aud("0.7 m", "rounded from the gt ceilings -0.7164 / -0.6676 (PREREG section 0 wording)")
P("5. **What `ego` reads is wider than 'ego motion'** (PREREG section 2): the 22-dim ego block at the tick frame holds the 8 P10 kinematic features, the 5 `nav` channels (a route input; optimistic on PhysicalAI, where the only route supplier is the ego's own future path),"
  " the 4-wide speed-limit bucket and its validity (the logged `limit` lane, a declared open channel, see the header; in v8F's own `INFERENCE_INPUT_KEYS`) and the 4 history-validity flags. No ablation was fitted in this experiment (no re-fit is allowed in it), so **which input carries the ADE6 cut is not measured here**."
  " The read-out's own skill was attributed afterwards by the prerequisites agent (section 4.5): it survives without `nav` and does not survive without `limit`.")
aud("22-dim; 8; 5; 4; 4", "PREREG section 2 table (INHERITED, MEASURED from source 2026-10-28 by the PREREG author)", "8 + 5 + 4 + 1 + 4 = 22 (limit 4 + limit_ok 1)")

# ---------------------------------------------------------------------------------------------------------------- 2
hdr("## 2. Controls: each must read its known value before any row is read")
SEC[0] = "2"
C = {s: RD["seeds"][s]["controls_recomputed"] for s in SEEDS}
HK = {(s, m): EJ[(s, m)]["speedread"]["hook"] for s in SEEDS for m in ("identity", "gt", "ego", "egoshuf")}
for s in SEEDS:
    for m in ("identity", "gt", "ego", "egoshuf"):
        assert HK[(s, m)]["n_batches"] == 38 and HK[(s, m)]["nonspeed_max_abs_dev"] == 0.0 and HK[(s, m)]["tick_mirror_ok"] is True
    assert HK[(s, "gt")]["oracle_code_max_abs_dev"] == 0.0
    assert C[s]["S5_max_abs_dev_own_vs_speedread_own"] <= 1e-5 and C[s]["S6_max_abs_dev_gt_vs_speedread_gt"] <= 1e-4
P(f"The reader's flags are in `SUBREAD.json` `controls`; the right-hand column of each row is the value `read/REDERIVE.json` recomputed from the arrays (or the dump's own field), so the flag and its reading are both on the page. All {NFLAG} flags are true"
  " (11 per seed, then S11, S12, S13 once, and S6b per seed), and `egoshuf` reads separated-worse at both seeds (section 1), which is the regression arm's committed reading.")
P("")
P("| id | must read (PREREG section 4) | s0 | s1 | measured detail (s0 / s1) |", "|---|---|---|---|---|")


def ctl(i, must, key, detail, per_seed=True):
    if per_seed:
        P(f"| {i} | {must} | {yn(CT['0'][key])} | {yn(CT['1'][key])} | {detail} |")


S5v = (aud(sci(C['0']['S5_max_abs_dev_own_vs_speedread_own']), 'REDERIVE.json seeds.0.controls_recomputed.S5_max_abs_dev_own_vs_speedread_own'),
       aud(sci(C['1']['S5_max_abs_dev_own_vs_speedread_own']), 'REDERIVE.json seeds.1.controls_recomputed.S5_max_abs_dev_own_vs_speedread_own'))
S6v = (aud(sci(C['0']['S6_max_abs_dev_gt_vs_speedread_gt']), 'REDERIVE.json seeds.0.controls_recomputed.S6_max_abs_dev_gt_vs_speedread_gt'),
       aud(sci(C['1']['S6_max_abs_dev_gt_vs_speedread_gt']), 'REDERIVE.json seeds.1.controls_recomputed.S6_max_abs_dev_gt_vs_speedread_gt'))
ctl("S1", "`identity` − `own` ADE6 max \\|Δ\\| = 0.0 and plans equal", "S1_identity_eq_own", f"ADE6 means `own` / `identity`: {aud(un(mean[('0', 'own')]), 'eval_s0 own raw.traj.ade6_m.mean')} / {aud(un(mean[('0', 'identity')]), 'eval_s0 identity raw.traj.ade6_m.mean')} ; {un(mean[('1', 'own')])} / {un(mean[('1', 'identity')])}; max \\|Δ\\| and plan equality recomputed: true")
ctl("S2", "non-speed cells O reads = X_s's own, max \\|Δ\\| = 0.0, in `identity`, `gt`, `ego`, `egoshuf`", "S2_nonspeed_untouched", "`speedread.hook.nonspeed_max_abs_dev` = 0.0 in all four hooked dumps at both seeds")
ctl("S3", "the hook's tick slice = the model's own `_scene` pad mask (first batch)", "S3_tick_mirror", f"`speedread.hook.tick_mirror_ok` true in all four hooked dumps at both seeds; {aud('38', 'eval_s{0,1}/*.json speedread.hook.n_batches')} batches each")
ctl("S4", "episode ids equal across all five modes", "S4_eid_aligned", "recomputed from the dumps' `eid`: true")
ctl("S5", "`own` ADE6 = SPEEDREAD's banked `own`, per window max \\|Δ\\| ≤ 1e-5 m", "S5_own_eq_speedread_own", f"max \\|Δ\\| {S5v[0]} / {S5v[1]} m (bar 1e-5; the run used 4 threads against SPEEDREAD's 8, which the bar allows)")
ctl("S6", "`gt` ADE6 = SPEEDREAD's banked `gt`, per window ≤ 1e-4 m", "S6_gt_eq_speedread_gt", f"max \\|Δ\\| {S6v[0]} / {S6v[1]} m")
P(f"| S6b | `gt` − `own` (delta, lo, hi) reproduces SPEEDREAD `READ.json` `rows.<s>.gt` within 5e-4 | {yn(CT['S6b_gt_delta_eq_speedread_read_json']['0'])} | {yn(CT['S6b_gt_delta_eq_speedread_read_json']['1'])} | {sg(G['0']['delta'])} [{sg(G['0']['lo'])}, {sg(G['0']['hi'])}] / {sg(G['1']['delta'])} [{sg(G['1']['lo'])}, {sg(G['1']['hi'])}], equal to the tip's SPEEDREAD READ.json |")
ctl("S7", "what O reads in `ego` = this run's identity `pred_mlp_ego`, standardised, ≤ 1e-5", "S7_ego_path_eq_identity_pred", f"max \\|Δ\\| {aud(un(C['0']['S7_max_abs_dev_ego_used_vs_identity_pred_std'], 1), 'REDERIVE.json S7_max_abs_dev_ego_used_vs_identity_pred_std')} / {un(C['1']['S7_max_abs_dev_ego_used_vs_identity_pred_std'], 1)} (std units); the preflight after s0 read the same (S7 dev 0)")
ctl("S7b", "this run's identity `pred_mlp_ego` = SPEEDREAD's banked one, ≤ 1e-5 m/s", "S7b_identity_pred_eq_speedread_banked", f"max \\|Δ\\| {un(C['0']['S7b_max_abs_dev_identity_pred_vs_speedread_banked'], 1)} / {un(C['1']['S7b_max_abs_dev_identity_pred_vs_speedread_banked'], 1)} m/s: the same read-out, checkpoint and windows SPEEDREAD scored")
ctl("S8", "`readout_sha256` in every hooked dump = `fit.json` = the pin `a205609a…`", "S8_readout_sha", "all ten dumps carry the pin; `readout.pt`, `fit.json` and the pin agree (recomputed)")
sh0 = EJ[("0", "egoshuf")]["speedread"]["hook"]["shuffle"]
ctl("S9", "`egoshuf`: permutation sha equals the registered one, alignment max \\|Δ\\| ≤ 1e-6, `n_seen` = 2,399, fixed points ≤ max(3, 1 %)", "S9_shuffle_registered_and_aligned",
    f"perm sha256 `{sh0['perm_sha256'][:8]}…` recomputed from the seed; alignment {aud(un(C['0']['S9_alignment_max_abs_dev_egoshuf_used_vs_ego_used_permuted'], 1), 'REDERIVE.json S9_alignment_max_abs_dev_egoshuf_used_vs_ego_used_permuted')}; `n_seen` {th(sh0['n_seen'])}; fixed points {aud(str(sh0['n_fixed']), 'eval_s0 egoshuf speedread.hook.shuffle.n_fixed')}")
ctl("S10", "`ego` moves ≥ 99 % of windows' cells away from `own`'s; `egoshuf` differs from `ego` on ≥ 95 %", "S10_substitution_reaches_O",
    f"fraction moved {aud(un(C['0']['S10_frac_ego_cells_moved_vs_own']), 'REDERIVE.json S10_frac_ego_cells_moved_vs_own')} / {un(C['1']['S10_frac_ego_cells_moved_vs_own'])}; `egoshuf` differs from `ego` on {aud(un(C['0']['S10_frac_egoshuf_differs_from_ego']), 'REDERIVE.json S10_frac_egoshuf_differs_from_ego')} / {un(C['1']['S10_frac_egoshuf_differs_from_ego'])}")
P(f"| S11 | train parity (4,512 / 306,566 / `f6a1dee0…`) | {yn(CT['S11_train_parity'])} (one flag for both seeds) | | `fit.json` `train_parity_ok` true |")
P(f"| S12 | eval-141 ∩ fit clips = ∅ | {yn(CT['S12_disjoint'])} (one flag) | | `fit.json` `eval_train_disjoint` true |")
P(f"| S13 | the section 0 selection rule applied to `fit.json` picks `mlp_ego` | {yn(CT['S13_selection_rule_picks_mlp_ego'])} (one flag) | | argmin of the mean fold-0 RMSE over the four read-outs = `{RD['S13_argmin']}` (values in section 1, point 1) |")
aud("4,512 / 306,566", "PREREG section 4 S11 (train parity; fit.json extract: 4,512 clips, 306,566 windows)")
aud("1e-5; 1e-4; 5e-4; 1e-6", "PREREG section 4 tolerances (S5/S7/S7b/S10-adjacent 1e-5, S6 1e-4, S6b 5e-4, S9 1e-6)")
aud("99 %; 95 %; 1 %; 3", "PREREG section 4 S9 and S10 thresholds")
aud("6", "S6b row id (PREREG section 4 S6b), id text")
P("")
P("Beside the S-series, the family tools' own controls (section 3) all hold at both seeds: `longlat` LL-1 (identity reads exactly 0.0 on every row), LL-2 (its ADE6 row equals `SUBREAD.json`, for `ego`, `egoshuf` and `gt`) and LL-3 (per-arm values equal the dump's own, to 1e-4),"
  f" and `pairfam` (identity on {aud(str(PF[('ego', '0')]['controls'][0]['n_cells']), 'pairfam_ego_s0.json controls[0].n_cells')} cells, reproduction of {aud(str(PF[('ego', '0')]['controls'][1]['n_checked']), 'pairfam_ego_s0.json controls[1].n_checked')} point values per arm, ADE6 recompute max deviation 0.0).")
for s in SEEDS:
    ll3 = LL[s]["controls"]["LL-3_reproduces_dump_family_values"]
    assert ll3["max_abs_dev"] < 1e-4 and LL[s]["controls"]["LL-1_identity"]["n_rows"] == 12 and PF[("ego", s)]["controls"][0]["n_cells"] == 45 and PF[("ego", s)]["controls"][1]["n_checked"] == 119
    assert PF[("ego", s)]["controls"][0]["n_not_zero"] == 0 and PF[("ego", s)]["alignment"]["ade6_recompute_max_dev"] == {"A": 0.0, "B": 0.0}
ll3m = max(LL[s]["controls"]["LL-3_reproduces_dump_family_values"]["max_abs_dev"] for s in SEEDS)
P(f"LL-3's largest deviation is {aud(sci(ll3m), 'longlat_s{0,1}.json controls.LL-3_reproduces_dump_family_values.max_abs_dev (max of the two seeds)')} (bar 1e-4), the 4-decimal rounding of the dumps' stored values. The stop rule worked as designed: the preflight after s0 ({RT[PRE_KEY][1]}) read"
  " S7b dev 0, S7 dev 0, moved 1.0000 before seed 1 started.")
aud(RT[PRE_KEY][1], "logs/run.out 'preflight_s0: done' stamp")
aud("1.0000", "logs/run.out preflight_s0: moved 1.0000 (>=0.99)")

# ---------------------------------------------------------------------------------------------------------------- 3
hdr("## 3. The four families (beside the decision; ADE is one row of four; per family, never pooled)")
SEC[0] = "3"
P("The reader's table decides the outcome from ADE6 only. The binding eval standard asks for four families beside it, each with its estimator, so they were computed on the same dumps by the family tools named in the PREREG (section 9): `v8f_paired_families.py` (unmodified, from the run tree) for TACTICAL and STRATEGIC,"
  " and `families_longlat_sub.py` (a parametrised copy of SPEEDREAD's `families_longlat.py`, sha256 `f981ea7a…`) for LONGITUDINAL and LATERAL. They changed nothing in section 1.")
P("")
P("* **Tier.** T1-class open loop (own held outputs; no recorded future action; the tools stamp the same words). The `gt` − `own` table is the T0 ceiling and a control, never a result.")
P("* **Estimator.** Paired episode-cluster bootstrap, B = 2,000, seed 0, 141 episodes: another draw of EPISODES only. One training run per seed and no replicate arm, so a separated cell is a pointer (`H-ESTIM-SEED-1`).")
P("* **STRATEGIC is reported, not paused, for v8F** (`strategic_family`: ACTIVE for v8F-mini).")
P("* **Markers.** B = separated-better, W = separated-worse, " + NDASH + " = not separated, S+ / S" + MINUS + " = separated on a row with no better direction (a bias or a ratio). Bold = separated. Error rows are lower-better; accuracy and F1 rows are higher-better.")


def fam_table(arm):
    P("| family | row | s0 | s1 | n windows s0 / s1 |", "|---|---|---|---|---|")
    fam_done = set()

    def row(fam, label, kind, key):
        f = fam if fam not in fam_done else ""
        fam_done.add(fam)
        cs, ns = [], []
        for s in SEEDS:
            if kind == "ll":
                e, b = llcell(arm, s, key)
                src = f"longlat_s{s}.json contrasts.{arm}-own.{key}"
                ns.append(e["n_windows"])
            else:
                e, b = pfcell(arm, s, key)
                src = f"pairfam_{arm}_s{s}.json {key}"
                ns.append(e["n_windows"])
            cs.append(cellstr(e, b, src))
        nn = f"{th(ns[0])} / {th(ns[1])}"
        aud(nn, "n_windows of the two cells above")
        P(f"| {f} | {label} | {cs[0]} | {cs[1]} | {nn} |")

    for k in ("ade6_m", "ade2_m"):
        row("PRIMARY" if k == "ade6_m" else "", LLK[k], "ll", k)
    for k in ("speed_mae_mps", "along_mae_m", "accel_mae_mps2"):
        row("LONGITUDINAL", LLK[k], "ll", k)
    P(f"|  | DK-VIOL / DK-HW / DK-TG | UNAVAILABLE | UNAVAILABLE | n = 0 / 0; the dump's reason: \"no lead-agent track supplied\" |")
    P(f"|  | anti-echo (hold-v0 baseline) | UNAVAILABLE | UNAVAILABLE | n = 0 / 0; 2,399 windows available; the dump's reason: \"no ego speed at t0 in the window dict\" |")
    for k in ("heading_mae_deg", "yaw_rate_mae_degps", "curvature_mae_1pm", "cross_mae_m", "L1_lat_accel_mae_mps2", "L2_curvature_mae_fast_1pm", "L3_yaw_rate_mae_fast_degps"):
        row("LATERAL", LLK[k], "ll", k)
    for key, label in PFK:
        fam = "TACTICAL" if key.startswith("tactical") else "STRATEGIC"
        row(fam, label, "pf", key)


for s_ in SEEDS:
    assert PF[("ego", s_)]["strategic"]["arcs_m"] == [30, 50, 80]
    assert "GT >= 5.2 m/s" in LL[s_]["contrasts"]["ego-own"]["L2_curvature_mae_fast_1pm"]["label"]
aud("30; 50; 80", "pairfam_ego_s{0,1}.json strategic.arcs_m = [30, 50, 80]")
aud("5.2", "longlat_s{0,1}.json contrasts.*.L2_curvature_mae_fast_1pm.label and L3: 'GT >= 5.2 m/s' (family-tool threshold, PREREG section 9)")
aud("60", "PREREG section 9: the DK adapter 'of about 60 lines' (ESTIMATED by the PREREG author)")
hdr("### `ego` − `own` (the arm; point [95 % CI])")
SEC[0] = "3 ego"
fam_table("ego")
hdr("### `gt` − `own` (T0 ceiling, a control only; point [95 % CI])")
SEC[0] = "3 gt"
fam_table("gt")
SEC[0] = "3"
aud("0 / 0; 2,399", "longlat_s{0,1}.json unavailable.distance_keeping.n, unavailable.anti_echo.{n,n_windows_available}")
aud("0.0", "heads and S rows read exactly +0.0000 [+0.0000, +0.0000] (section 3 tables)")

# separated-cell census (registered 12 longlat rows + the pairfam cells with an interval)
CENS = {}
for s in SEEDS:
    for arm in ("ego", "gt"):
        ncell = len(LLK) + PF[(arm, s)]["summary"]["tactical"]["n_ok"] + PF[(arm, s)]["summary"]["strategic"]["n_ok"]
        assert ncell == 49
        sep = []
        for k in LLK:
            e = LL[s]["contrasts"][f"{arm}-own"][k]
            if e["separated"]:
                sep.append((k, e["delta"], e["lo"], e["hi"], tagof(e, "lower")))
        for it in PF[(arm, s)]["summary"]["tactical"]["separated"] + PF[(arm, s)]["summary"]["strategic"]["separated"]:
            tg_ = {"A better": "B", "A worse": "W", "A higher": "S+", "A lower": "S" + MINUS}[it["reading"]]
            sep.append((it["cell"], it["delta"], it["lo"], it["hi"], tg_))
        CENS[(arm, s)] = (ncell, sep)
assert len(CENS[("ego", "0")][1]) == 11 and len(CENS[("ego", "1")][1]) == 9

hdr("### What the rows say (beside; none of this moves the outcome)")
SEC[0] = "3 reading"
P("Separated cells of `ego` − `own`, by seed (a 49-cell read: the 12 longlat rows plus the 37 family-tool cells that carry an interval; at a nominal 5 % about 2 to 3 would separate with no effect, PREREG section 9):")
P("")
for s in SEEDS:
    n, sep = CENS[("ego", s)]
    items = "; ".join(f"`{k}` {sg(d)} [{sg(lo)}, {sg(hi)}] {t}" for k, d, lo, hi, t in sep)
    for k, d, lo, hi, t in sep:
        aud(f"{sg(d)} [{sg(lo)}, {sg(hi)}] {t}", f"{'longlat' if k in LLK else 'pairfam_ego'}_s{s}.json (cell {k}) delta, lo, hi")
    P(f"* **s{s}:** {aud(str(len(sep)), f'count of separated cells, ego-own, seed {s}', f'longlat rows separated + pairfam summary separated list')} of {aud(str(n), 'longlat 12 rows + pairfam tactical.n_ok 31 + strategic.n_ok 6', '12 + 31 + 6 = 49')} cells separated: {items}.")
aud("2 to 3", "PREREG section 9 (about 49 cells read per contrast; at 5 % about 2 to 3 would separate with no effect)")
aud("12; 37", "12 longlat rows; 31 + 6 = 37 pairfam cells with an interval", "12 + 37 = 49")
P("")
# derived ratios of the longitudinal rows to the ceiling
ratio = {}
for s in SEEDS:
    for k in ("speed_mae_mps", "along_mae_m", "accel_mae_mps2"):
        ratio[(s, k)] = LL[s]["contrasts"]["ego-own"][k]["delta"] / LL[s]["contrasts"]["gt-own"][k]["delta"]
for s in SEEDS:
    for k in ("speed_mae_mps", "along_mae_m", "accel_mae_mps2"):
        assert LL[s]["contrasts"]["ego-own"][k]["separated"] and LL[s]["contrasts"]["ego-own"][k]["delta"] < 0
    for k in ("heading_mae_deg", "yaw_rate_mae_degps", "curvature_mae_1pm", "cross_mae_m", "L1_lat_accel_mae_mps2", "L2_curvature_mae_fast_1pm", "L3_yaw_rate_mae_fast_degps"):
        e = LL[s]["contrasts"]["ego-own"][k]
        assert not (e["separated"] and e["delta"] > 0), (s, k)           # no lateral row separated-worse
    assert LL[s]["contrasts"]["ego-own"]["L3_yaw_rate_mae_fast_degps"]["separated"] and LL[s]["contrasts"]["ego-own"]["L3_yaw_rate_mae_fast_degps"]["delta"] < 0
    assert not LL[s]["contrasts"]["ego-own"]["cross_mae_m"]["separated"] and not LL[s]["contrasts"]["ego-own"]["ade2_m"]["separated"] and LL[s]["contrasts"]["ego-own"]["ade2_m"]["delta"] > 0
    for k in ("tactical.plan_derived.lateral_decision.accuracy", "tactical.plan_derived.longitudinal_decision.accuracy", "tactical.plan_derived.maneuver_5way_collapsed.accuracy",
              "tactical.heads.lateral.macro_f1", "tactical.heads.longitudinal.macro_f1", "tactical.heads.goal.macro_f1", "strategic.position_error_m.30", "strategic.heading_error_deg.80"):
        e, _ = pfcell("ego", s, k)
        assert not e["separated"]
    for k in ("tactical.heads.lateral.macro_f1", "tactical.heads.longitudinal.macro_f1", "tactical.heads.goal.macro_f1", "strategic.position_error_m.30", "strategic.position_error_m.50", "strategic.position_error_m.80",
              "strategic.heading_error_deg.30", "strategic.heading_error_deg.50", "strategic.heading_error_deg.80"):
        e, _ = pfcell("ego", s, k)
        assert (e["delta"], e["lo"], e["hi"]) == (0.0, 0.0, 0.0)
    e, _ = pfcell("ego", s, "tactical.goal_setting.goal_point_error_m")
    assert e["separated"] and e["delta"] < 0
    e, _ = pfcell("ego", s, "tactical.goal_setting.goal_long_bias_m")
    assert e["separated"] and e["delta"] > 0
g0, g1 = [pfcell("ego", s, "tactical.goal_setting.goal_long_bias_m")[0] for s in SEEDS]
P(f"1. **LONGITUDINAL: the three registered rows move, at both seeds.** Speed MAE, along-track MAE and accel MAE are separated-better at s0 and at s1 (SPEEDREAD's `lin` read moved none of them, INHERITED, `RESULT_V8F_SPEEDREAD.md` section 4). The `ego` change is "
  f"{aud(pct(ratio[('0', 'speed_mae_mps')]), 'DERIVED', 'longlat_s0 ego-own / gt-own, speed_mae_mps')} (s0) and {aud(pct(ratio[('1', 'speed_mae_mps')]), 'DERIVED', 'longlat_s1 ego-own / gt-own, speed_mae_mps')} (s1) of the `gt` ceiling on speed MAE, "
  f"{aud(pct(ratio[('0', 'along_mae_m')]), 'DERIVED', 'longlat_s0 along_mae_m')} / {aud(pct(ratio[('1', 'along_mae_m')]), 'DERIVED', 'longlat_s1 along_mae_m')} on along-track MAE (both in line with the ADE6 CAPTURE of section 1), but {aud(pct(ratio[('0', 'accel_mae_mps2')]), 'DERIVED', 'longlat_s0 accel_mae_mps2')} / {aud(pct(ratio[('1', 'accel_mae_mps2')]), 'DERIVED', 'longlat_s1 accel_mae_mps2')} on accel MAE: `ego` takes nearly the whole of the ceiling's accel gain, which is itself small (the ceiling moves accel MAE by only 0.05 and 0.03 m/s²)."
  " **Distance keeping (DK-VIOL / DK-HW / DK-TG) and the anti-echo block are not computed**: status UNAVAILABLE, n = 0, in every dump of both seeds (the scoring call was given no lead-agent track and no t0 speed). That is a work item, not a pass;"
  " until the adapter exists (about 60 lines, PREREG section 9) half of the LONGITUDINAL family is unmeasured for this arm.")
P(f"2. **LATERAL: pointers only, none worse.** s0 separates heading MAE, L1 lateral-accel MAE and L3 yaw-rate MAE (GT ≥ 5.2 m/s), all separated-better; s1 separates yaw-rate MAE, L2 curvature MAE and L3 yaw-rate MAE, all separated-better. L3 is separated-better at both seeds"
  f" ({sg(LL['0']['contrasts']['ego-own']['L3_yaw_rate_mae_fast_degps']['delta'])} / {sg(LL['1']['contrasts']['ego-own']['L3_yaw_rate_mae_fast_degps']['delta'])} deg/s). No lateral row is separated-worse at either seed; cross-track MAE is not separated (point {sg(LL['0']['contrasts']['ego-own']['cross_mae_m']['delta'])} / {sg(LL['1']['contrasts']['ego-own']['cross_mae_m']['delta'])} m).")
aud(f"{sg(LL['0']['contrasts']['ego-own']['L3_yaw_rate_mae_fast_degps']['delta'])} / {sg(LL['1']['contrasts']['ego-own']['L3_yaw_rate_mae_fast_degps']['delta'])}", "longlat_s{0,1}.json contrasts.ego-own.L3_yaw_rate_mae_fast_degps.delta")
aud(f"{sg(LL['0']['contrasts']['ego-own']['cross_mae_m']['delta'])} / {sg(LL['1']['contrasts']['ego-own']['cross_mae_m']['delta'])}", "longlat_s{0,1}.json contrasts.ego-own.cross_mae_m.delta")
a2 = [LL[s]["contrasts"]["ego-own"]["ade2_m"] for s in SEEDS]
P(f"3. **TACTICAL: the decisions do not move; the goal point does.** TAC-LAT, TAC-LON and the 5-way manoeuvre accuracy are not separated at either seed. The goal-point error is separated-better at both seeds"
  f" ({sg(pfcell('ego', '0', 'tactical.goal_setting.goal_point_error_m')[0]['delta'])} / {sg(pfcell('ego', '1', 'tactical.goal_setting.goal_point_error_m')[0]['delta'])} m) and the goal longitudinal bias moves by {sg(g0['delta'])} / {sg(g1['delta'])} m"
  f" (S+: a bias has no better direction; the base's own bias is {sg(g0['b'])} m at s0 and {sg(g1['b'])} m at s1, and `ego` reduces its magnitude). The three head macro-F1 rows read exactly +0.0000 [+0.0000, +0.0000] at both seeds: the substitution acts on the code O reads, after T has run, so T's heads cannot move."
  " The goal-point and bias cells are read from O's plan, so they move with ADE6 and are not an independent confirmation of it. The per-token goal F1 rows are in `families/pairfam_ego_s0.md`; this family's gaps are in the table below.")
aud(f"{sg(pfcell('ego', '0', 'tactical.goal_setting.goal_point_error_m')[0]['delta'])} / {sg(pfcell('ego', '1', 'tactical.goal_setting.goal_point_error_m')[0]['delta'])}", "pairfam_ego_s{0,1}.json tactical.goal_setting.goal_point_error_m.delta")
aud(f"{sg(g0['delta'])} / {sg(g1['delta'])}", "pairfam_ego_s{0,1}.json tactical.goal_setting.goal_long_bias_m.delta")
aud(f"{sg(g0['b'])}", "pairfam_ego_s0.json tactical.goal_setting.goal_long_bias_m.b"); aud(f"{sg(g1['b'])}", "pairfam_ego_s1.json tactical.goal_setting.goal_long_bias_m.b")
P(f"4. **STRATEGIC: exactly zero, by construction.** All six S rows (S-POS@30/50/80, S heading@30/50/80) read +0.0000 [+0.0000, +0.0000] at both seeds, for `ego` and for `gt`. The substitution acts on the code O reads after S and T have run, so this is a property of the substitution site and is **not evidence about strategic quality**.")
P(f"5. **The 2 s horizon does not improve.** ADE2 is not separated at either seed and its point estimate is positive (worse): {sg(a2[0]['delta'])} [{sg(a2[0]['lo'])}, {sg(a2[0]['hi'])}] / {sg(a2[1]['delta'])} [{sg(a2[1]['lo'])}, {sg(a2[1]['hi'])}] m. "
  "The gain is a long-horizon gain; the `gt` ceiling moves ADE2 by less than 0.02 m (table above).")
aud("0.05 and 0.03", "longlat_s{0,1}.json contrasts.gt-own.accel_mae_mps2.delta (-0.0536, -0.0303), rounded")
aud("0.02", "bound on the gt ADE2 rows in the table above (-0.0109, -0.0119: SPEEDREAD's gt rows; this run's gt-own ade2_m rows are in the gt table)")
for s in SEEDS:
    assert abs(LL[s]["contrasts"]["gt-own"]["ade2_m"]["delta"]) < 0.02
P("6. **Read the separated cells as pointers.** They come from a 49-cell read at one nominal level, with no replicate arm. Several rows share one mechanism (speed MAE, along-track MAE, accel MAE, ADE6 and the goal-point error all follow the plan's longitudinal profile), so they are not independent confirmations of each other.")
P("7. **Not run:** the `egoshuf` control has no family tables (the PREREG asks for its ADE6 row only, which section 1 reports); this is stated so its absence is not read as a pass.")

hdr("### Gaps, per family: what is not computed, with its reason and n")
SEC[0] = "3 gaps"
UNA = {}
for s in SEEDS:
    for a in ("ego", "gt"):
        d = PF[(a, s)]
        ab = d["tactical"]["unavailable_by_design"]
        assert ab["anchor_selection"]["status"] == "UNAVAILABLE" and ab["anchor_selection"]["n"] == 2399 and ab["selected_vs_executed"]["status"] == "UNAVAILABLE" and ab["selected_vs_executed"]["n"] == 2399
        assert "ONE path per window" in ab["anchor_selection"]["reason"] and "a trajectory dump does not carry" in ab["selected_vs_executed"]["reason"]
        pt = d["tactical"]["heads"]["goal"]["per_token"]
        un_tok = {k: (v["n_labelled"], v["n_positive"]) for k, v in pt.items() if v.get("status") == "UNAVAILABLE"}
        assert un_tok == {"SPEED_BAND": (0, 0), "TAKE_EXIT_L": (695, 0), "TRAFFIC_LIGHT_REACT": (695, 0)}, un_tok
        assert d["summary"]["tactical"]["n_unavailable"] == 5 and d["summary"]["tactical"]["n_point_only"] == 8 and d["summary"]["strategic"]["n_unavailable"] == 0
P("The same gaps hold in all four family JSONs (`ego` and `gt`, both seeds).")
P("")
P("| family | row not computed | status | n | reason |", "|---|---|---|---|---|")
P("| LONGITUDINAL | DK-VIOL / DK-HW / DK-TG | UNAVAILABLE | 0 | the scoring call was given no lead-agent track (\"no lead-agent track supplied\"). The lead block exists, but the reading tool's CLI is hard-wired to other arms: an adapter of about 60 lines (PREREG section 9, INHERITED); a work item, not built |")
P("| LONGITUDINAL | anti-echo (hold-v0 baseline) | UNAVAILABLE | 0 of 2,399 windows | no ego speed at t0 in the window dict (\"no ego speed at t0 in the window dict\") |")
P("| LONGITUDINAL | target-speed accuracy with an interval | point values only (section 4.4) | 2,399 windows | the family tools have no paired row for it |")
P("| TACTICAL | anchor selection | UNAVAILABLE | 2,399 | each dump's arm commits to ONE path per window: there is no candidate fan and no selection to score |")
P("| TACTICAL | selected vs executed manoeuvre | UNAVAILABLE | 2,399 | both label streams are executed manoeuvres; scoring a declared decision needs a tactical head's selection, which a trajectory dump does not carry |")
P("| TACTICAL | goal per-token F1: `SPEED_BAND`, `TAKE_EXIT_L`, `TRAFFIC_LIGHT_REACT` | UNAVAILABLE | 0 / 695 / 695 labelled windows, 0 positives each | no positive label for the token on this grid, so its F1 is undefined |")
P("| TACTICAL | goal per-token F1, 8 further cells | point values only, no interval | see `families/pairfam_ego_s0.md` | F1 undefined on at least one resample (no positive label in the drawn episodes) |")
P("| TACTICAL | confusion over the manoeuvre classes | not in the tool's output (it reports accuracy and kappa) | 2,399 | a work item: the dumps carry the predictions, the tool does not print the matrix |")
P("| LATERAL, STRATEGIC | none | computed | 2,096 to 2,399 (LATERAL); 2,294 to 2,322 (STRATEGIC) | STRATEGIC reads exactly 0.0000 by construction |")
P("| all four | the `egoshuf` control's family tables | not run | | the PREREG asks for its ADE6 row only |")
P("| all four | the T1-kin tier | not run for this arm | | it is the next arm (section 6) |")
aud("0 / 695 / 695", "pairfam_*_s*.json tactical.heads.goal.per_token.{SPEED_BAND,TAKE_EXIT_L,TRAFFIC_LIGHT_REACT}.n_labelled")
aud("8", "pairfam_*_s*.json summary.tactical.n_point_only")
aud("2,096 to 2,399; 2,294 to 2,322", "min and max n_windows over the LATERAL longlat cells and the STRATEGIC pairfam cells in the tables above")
assert min(LL[s]["contrasts"]["ego-own"][k]["n_windows"] for s in SEEDS for k in LLK) == 2096 and min(pfcell("ego", s, "strategic.position_error_m.80")[0]["n_windows"] for s in SEEDS) == 2294
assert max(pfcell("ego", s, "strategic.position_error_m.30")[0]["n_windows"] for s in SEEDS) == 2322

# ---------------------------------------------------------------------------------------------------------------- 4
hdr("## 4. Beside (deciding nothing)")
SEC[0] = "4"
hdr("### 4.1 The read-out's own accuracy, cell by cell (eval RMSE to the GT, m/s)")
SEC[0] = "4.1"
CELLS = SUB["cells"]["0"]["cells"]
assert CELLS == ["lon_v_target_ms", "v_a_ms", "v_end_ms", "v_lo_ms", "v_hi_ms"] and SUB["cells"]["0"]["n_present"] == SUB["cells"]["1"]["n_present"] == [2334, 2351, 2317, 2351, 2351]
assert SUB["cells"]["0"]["mlp_ego"]["rmse"] == SUB["cells"]["1"]["mlp_ego"]["rmse"]
P("Per-cell RMSE on the eval-141 windows where the GT cell is present (`cells.<s>.n_present`), `T_own` = X_s's own speed cells as O reads them, `mlp_ego` = the arm's cells (identical at the two seeds: the read-out does not depend on the seed). Recomputed from the arrays in `REDERIVE.json` to 1e-4; the values are SUBREAD.json `cells`.")
P("")
P("| cell | n present | `T_own` s0 | `T_own` s1 | `mlp_ego` | cut vs `T_own` s0 | cut vs `T_own` s1 |", "|---|---|---|---|---|---|---|")
allbelow = True
for j, c in enumerate(CELLS):
    t0, t1, me = SUB["cells"]["0"]["T_own"]["rmse"][j], SUB["cells"]["1"]["T_own"]["rmse"][j], SUB["cells"]["0"]["mlp_ego"]["rmse"][j]
    allbelow &= (me < t0 and me < t1)
    P(f"| `{c}` | {aud(th(SUB['cells']['0']['n_present'][j]), 'SUBREAD.json cells.0.n_present')} | {aud(un(t0), f'SUBREAD.json cells.0.T_own.rmse[{j}]')} | {aud(un(t1), f'SUBREAD.json cells.1.T_own.rmse[{j}]')} | {aud(un(me), f'SUBREAD.json cells.0.mlp_ego.rmse[{j}]')} | {aud(pct(1 - me / t0), 'DERIVED', f'1 - {me:.4f} / {t0:.4f}')} | {aud(pct(1 - me / t1), 'DERIVED', f'1 - {me:.4f} / {t1:.4f}')} |")
assert allbelow
P("")
P("`mlp_ego` is below `T_own` in all five cells at both seeds, as the PREREG's section 0 said it would be (that table is re-derived here from the arrays, not copied). **That did not hold for SPEEDREAD's ridge, which cut `v_a_ms` by 40–59 % and moved ADE6 by about 0** (INHERITED, PREREG section 0), so a lower cell RMSE was not a sufficient condition; here it coincides with an ADE6 cut at both seeds."
  " Which of the five cells carries the cut is not known from this experiment (that is the question of the unrun PREREG_V8F_SPEEDCELL, section 6).")
aud("40–59 %", "PREREG section 0 (INHERITED from SPEEDREAD READ.json: ridge cut v_a_ms RMSE 59 % at s0 and 40 % at s1)")
aud("1e-4", "REDERIVE.json mismatches_vs_SUBREAD_json tolerance on cell RMSE (rederive_sub.py)")

hdr("### 4.2 Post-hoc, not registered: the plan's rule-violation rates (point values and paired intervals)")
SEC[0] = "4.2"
VK = [("viol_any", "any rule"), ("viol_SPEED", "SPEED"), ("viol_STOP", "STOP"), ("viol_TURN", "TURN"), ("viol_LEAD", "LEAD")]
for s in SEEDS:
    cons = EJ[(s, "ego")]["raw"]["consistency_a"]
    assert abs(cons["any"]["rate_all"] - RD["seeds"][s]["rows"]["viol_post_hoc"]["viol_any"]["rate_ego"]) < 1e-12
    for kk, nm in (("SPEED", "viol_SPEED"), ("STOP", "viol_STOP"), ("TURN", "viol_TURN"), ("LEAD", "viol_LEAD")):
        assert abs(cons[kk]["rate_all"] - RD["seeds"][s]["rows"]["viol_post_hoc"][nm]["rate_ego"]) < 1e-12
        assert abs(EJ[(s, "own")]["raw"]["consistency_a"][kk]["rate_all"] - RD["seeds"][s]["rows"]["viol_post_hoc"][nm]["rate_own"]) < 1e-12
        assert abs(EJ[(s, "gt")]["raw"]["consistency_a"][kk]["rate_all"] - RD["seeds"][s]["rows"]["viol_post_hoc"][nm]["rate_gt"]) < 1e-12
P("This block is **not in the PREREG**. The harness scores every plan against constraints built from the GT (`consistency_a` in each dump, label CONS(a) in the stage logs: speed band, stop pass, turn, lead time-gap; `taniteval.v8f_consistency.consistency`, called in `v8f_mini_eval.py` `evaluate`),"
  " and the per-window violation flags are in the dumps. They were read after section 1 with the same paired estimator (B = 2,000, seed 0, 141 episodes), rate = share of the 2,399 windows that break the rule, **lower is better**. They decide nothing and are one more pointer family (about 5 rows per contrast, post-hoc),"
  " but they bear on how far an ADE6 gain can be trusted, so they are reported. The rates equal the dumps' own `raw.consistency_a.*.rate_all` (asserted).")
P("")
P("| rule | rate `own` s0 / s1 | rate `ego` s0 / s1 | rate `gt` s0 / s1 | `ego` − `own` s0 | `ego` − `own` s1 |", "|---|---|---|---|---|---|")
for vk, nm in VK:
    V0, V1 = RD["seeds"]["0"]["rows"]["viol_post_hoc"][vk], RD["seeds"]["1"]["rows"]["viol_post_hoc"][vk]

    def vc(V, s):
        e = V["ego_minus_own"]
        t = tagof({"separated": e["separated"], "delta": e["delta"]}, "lower")
        st = f"{sg(e['delta'])} [{sg(e['lo'])}, {sg(e['hi'])}] {t}"
        st = f"**{st}**" if t != NDASH else st
        return aud(st, f"REDERIVE.json seeds.{s}.rows.viol_post_hoc.{vk}.ego_minus_own.{{delta,lo,hi,separated}}")
    rates = lambda k: aud(f"{un(V0['rate_' + k])} / {un(V1['rate_' + k])}", f"REDERIVE.json seeds.{{0,1}}.rows.viol_post_hoc.{vk}.rate_{k}")
    P(f"| {nm} | {rates('own')} | {rates('ego')} | {rates('gt')} | {vc(V0, '0')} | {vc(V1, '1')} |")
vany = [RD["seeds"][s]["rows"]["viol_post_hoc"]["viol_any"]["ego_minus_own"] for s in SEEDS]
vlead = [RD["seeds"][s]["rows"]["viol_post_hoc"]["viol_LEAD"]["ego_minus_own"] for s in SEEDS]
gtany = [RD["seeds"][s]["rows"]["viol_post_hoc"]["viol_any"]["gt_minus_own"]["delta"] for s in SEEDS]
assert all(e["separated"] and e["delta"] > 0 for e in vany) and all(e["separated"] and e["delta"] > 0 for e in vlead)
assert all(RD["seeds"][s]["rows"]["viol_post_hoc"]["viol_any"]["gt_minus_own"]["separated"] and RD["seeds"][s]["rows"]["viol_post_hoc"]["viol_any"]["gt_minus_own"]["delta"] < 0 for s in SEEDS)
P("")
P(f"**Reading.** The rate of plans that break at least one rule rises under `ego` at both seeds and is separated-worse at both ({sg(vany[0]['delta'])} [{sg(vany[0]['lo'])}, {sg(vany[0]['hi'])}] / {sg(vany[1]['delta'])} [{sg(vany[1]['lo'])}, {sg(vany[1]['hi'])}]), carried by the LEAD rule at both seeds"
  f" ({sg(vlead[0]['delta'])} / {sg(vlead[1]['delta'])}) and by SPEED at s1; the `gt` cells move the same rate the other way, down by {aud(un(-gtany[0]), 'REDERIVE.json viol_any.gt_minus_own.delta, s0', '0 - -0.0817')} / {aud(un(-gtany[1]), 'REDERIVE.json viol_any.gt_minus_own.delta, s1', '0 - -0.0688')} (T0 ceiling, a control). So the `ego` cells cut the mean trajectory error while making the plan break a GT-derived rule more often. This block does not alter the PASS (ADE6 is the registered row)."
  " It is the strongest reason for the next arm to carry a rule-violation guard beside ADE6 (section 6).")
aud(f"{sg(vany[0]['delta'])} [{sg(vany[0]['lo'])}, {sg(vany[0]['hi'])}] / {sg(vany[1]['delta'])} [{sg(vany[1]['lo'])}, {sg(vany[1]['hi'])}]", "REDERIVE.json viol_any.ego_minus_own (both seeds)")
aud(f"{sg(vlead[0]['delta'])} / {sg(vlead[1]['delta'])}", "REDERIVE.json viol_LEAD.ego_minus_own.delta (both seeds)")

hdr("### 4.3 Post-hoc, not registered: ADE6 by event stratum (point values, no interval)")
SEC[0] = "4.3"
STR = ("turn", "stop", "lead_brake")
for s in SEEDS:
    for m in ("own", "ego", "gt"):
        for k in STR:
            assert EJ[(s, m)]["raw"]["strata_ade6"][k]["n"] == EJ[("0", "own")]["raw"]["strata_ade6"][k]["n"]
P("The harness reports the mean ADE6 inside three event strata (`raw.strata_ade6`: turn, stop, lead-brake windows; the PREREG section 7 'cheapest first, no run needed' item). The dumps keep the stratum **mean** only, not the window list, so there is **no interval** here:"
  " these are point values on small, possibly overlapping sets (the stop stratum is 81 windows), to be read as a direction only. Reported because on the stop windows `ego` and `gt` pull in opposite directions.")
P("")
P("| stratum | n windows | `own` s0 / s1 | `ego` s0 / s1 | `gt` s0 / s1 | `ego` − `own` s0 / s1 | `gt` − `own` s0 / s1 |", "|---|---|---|---|---|---|---|")
for k in STR:
    n = EJ[("0", "own")]["raw"]["strata_ade6"][k]["n"]
    v = {(s, m): EJ[(s, m)]["raw"]["strata_ade6"][k]["ade6_m"] for s in SEEDS for m in ("own", "ego", "gt")}
    d_e = {s: v[(s, "ego")] - v[(s, "own")] for s in SEEDS}
    d_g = {s: v[(s, "gt")] - v[(s, "own")] for s in SEEDS}
    P(f"| {k} | {aud(th(n), 'eval_s0/eval_*-own.json raw.strata_ade6.' + k + '.n')} | {aud(un(v[('0', 'own')]) + ' / ' + un(v[('1', 'own')]), f'eval_s{{0,1}}/*-own.json raw.strata_ade6.{k}.ade6_m')} | {aud(un(v[('0', 'ego')]) + ' / ' + un(v[('1', 'ego')]), f'eval_s{{0,1}}/*-ego.json raw.strata_ade6.{k}.ade6_m')}"
      f" | {aud(un(v[('0', 'gt')]) + ' / ' + un(v[('1', 'gt')]), f'eval_s{{0,1}}/*-gt.json raw.strata_ade6.{k}.ade6_m')} | {aud(sg(d_e['0']) + ' / ' + sg(d_e['1']), 'DERIVED', f'ego - own {k}')} | {aud(sg(d_g['0']) + ' / ' + sg(d_g['1']), 'DERIVED', f'gt - own {k}')} |")
    if k == "stop":
        stop_e = d_e
        stop_g = d_g
    if k == "lead_brake":
        lb_e1 = d_e["1"]
assert all(stop_e[s] > 0 for s in SEEDS) and all(stop_g[s] < -0.5 for s in SEEDS)
P("")
P(f"**Reading.** On the {th(EJ[('0', 'own')]['raw']['strata_ade6']['stop']['n'])} stop windows `ego` is *worse* than `own` at both seeds ({sg(stop_e['0'])} / {sg(stop_e['1'])} m, point values) while `gt` is better by {un(-stop_g['0'])} / {un(-stop_g['1'])} m; turn windows improve at both seeds, and lead-brake windows improve at s0 and read {sg(lb_e1)} m at s1."
  " If this survives a paired interval, the `ego` cells help on the turn windows and hurt on the stop windows, where the GT cells help. Getting those intervals needs the strata's window lists, which the harness builds inside `evaluate` and does not store: a small adapter, named in section 6, not built here.")
aud(f"{un(-stop_g['0'])} / {un(-stop_g['1'])}", "DERIVED", "-(gt - own) stop windows")

hdr("### 4.4 Point values from the dumps: target-speed accuracy (no interval)")
SEC[0] = "4.4"
P("The binding LONGITUDINAL family names target-speed accuracy. The dumps carry it as the fraction of horizon **steps** (not windows) whose speed error is inside a band (`raw.families.longitudinal.target_speed_acc`; the bands are proposed reporting tolerances, not a gate). Point values only; no paired interval was computed.")
P("")
P("| band | `own` s0 / s1 | `ego` s0 / s1 | `gt` s0 / s1 |", "|---|---|---|---|")
for b in ("within_0.5_mps", "within_1.0_mps", "within_2.0_mps"):
    vv = {(s, m): EJ[(s, m)]["raw"]["families"]["longitudinal"]["target_speed_acc"][b] for s in SEEDS for m in ("own", "ego", "gt")}
    P(f"| {b.replace('within_', 'within ').replace('_mps', ' m/s')} | " + " | ".join(aud(un(vv[('0', m)]) + " / " + un(vv[('1', m)]), f"eval_s{{0,1}}/*-{m}.json raw.families.longitudinal.target_speed_acc.{b}") for m in ("own", "ego", "gt")) + " |")
    assert vv[("0", "ego")] >= vv[("0", "own")] and vv[("1", "ego")] >= vv[("1", "own")]
aud("0.5; 1.0; 2.0", "band names within_0.5_mps, within_1.0_mps, within_2.0_mps")

hdr("### 4.5 Attribution of the read-out's skill (FINDING_1; beside, changes no outcome)")
SEC[0] = "4.5"
FM = {k: f"{ATT_MEAN[k]:.5f}" for k in ATT_DES}
cost = {k: ATT_MEAN[k] - ATT_MEAN["full"] for k in ATT_DES}
gap = ATT_MEAN["a_p10f_fok"] - ATT_MEAN["full"]
close_lim = (ATT_MEAN["a_p10f_fok"] - ATT_MEAN["b_nonav"]) / gap
close_nav = (ATT_MEAN["a_p10f_fok"] - ATT_MEAN["c_nolimit"]) / gap
SRC_ATT = "F:/TanitAD-scratch/speedsub_t1_prep/refit_all.json designs.{d}.mlp.holdout_rmse_sd, mean over the 5 cells (E:/Projects/TanitAD-artifacts/dagland/speedsub_t1_prep/FINDING_1_ATTRIBUTION.md section 1 table)"
P("The PREREG section 6 'required first' attribution was run by the prerequisites agent and is cited here, not re-run: `E:/Projects/TanitAD-artifacts/dagland/speedsub_t1_prep/FINDING_1_ATTRIBUTION.md` (sha256 `" + shaf(F1P)[:8] + "…`); artifact `" + ATTP + "` (sha256 `" + shaf(ATTP)[:8] + "…`)."
  " MEASURED on TRAIN fold 0 (the hold-out of `fit.json`, 846 clips), not on eval-141 and not on ADE6. The `mlp_ego` family was refitted on subsets of the 22-dim block; the first row is a control refit that reproduces the banked `mlp_ego` (the script asserts the mean equals `fit.json`'s). Mean hold-out RMSE over the five cells, train-SD units, lower is better:")
aud(shaf(F1P)[:8], "sha256 of E:/Projects/TanitAD-artifacts/dagland/speedsub_t1_prep/FINDING_1_ATTRIBUTION.md")
aud(shaf(ATTP)[:8], "sha256 of F:/TanitAD-scratch/speedsub_t1_prep/refit_all.json")
aud("846", "FINDING_1_ATTRIBUTION.md section 4: fold 0 = 846 clips (INHERITED, read)")
P("")
P("| design | dim | mean fold-0 RMSE (SD units) | cost against the full block |", "|---|---|---|---|")
for k, lab in (("full", "full block = `mlp_ego` as banked (control refit)"), ("b_nonav", "no `nav`: `p10f + limit + limit_ok + frame_ok`"), ("c_nolimit", "no `limit`: `p10f + nav + frame_ok`"), ("a_p10f_fok", "neither: `p10f + frame_ok`")):
    cst = "none" if k == "full" else aud("+" + un(cost[k]), "DERIVED", f"{FM[k]} - {FM['full']}")
    P(f"| {lab} | {aud(str(ATT_DES[k]), 'refit_all.json designs.' + k + '.d')} | {aud(FM[k], SRC_ATT.replace('{d}', k))} | {cst} |")
bl = {k: sum(FIT[k]["holdout_rmse_sd"]) / 5 for k in ("lin", "lin_ego", "mlp")}
assert ATT_MEAN["b_nonav"] < bl["lin"] < ATT_MEAN["c_nolimit"] < bl["lin_ego"] < bl["mlp"]
P("")
P(f"**Reading.** The read-out's skill survives without `nav` and does not survive without `limit`. Dropping `nav` costs {un(cost['b_nonav'])} SD and the no-`nav` read-out ({FM['b_nonav']}) is still below the best other banked read-out on this fold (`lin`, {un(bl['lin'])}); dropping `limit` costs {un(cost['c_nolimit'])} SD and leaves the read-out ({FM['c_nolimit']}) above `lin`."
  f" Starting from `p10f + frame_ok` alone, adding `limit` closes {pct(close_lim, 0)} of the gap to the full block and adding `nav` closes {pct(close_nav, 0)} (derived from the four values; FINDING_1's three-seed figures are 73 % and 46 %)."
  " The MLP's own seed floor on these means is 0.0003 to 0.0013 over three seeds (FINDING_1 section 1, INHERITED), far below the gaps, so the ordering is not a seed effect; it is one hold-out fold, not an eval.")
aud(pct(close_lim, 0), "DERIVED", f"({FM['a_p10f_fok']} - {FM['b_nonav']}) / ({FM['a_p10f_fok']} - {FM['full']}) = {close_lim:.4f}")
aud(pct(close_nav, 0), "DERIVED", f"({FM['a_p10f_fok']} - {FM['c_nolimit']}) / ({FM['a_p10f_fok']} - {FM['full']}) = {close_nav:.4f}")
aud(un(bl['lin']), "fit.json lin.holdout_rmse_sd mean over the 5 cells (same value as section 1 point 1)")
aud(un(cost['b_nonav']), "DERIVED", f"{FM['b_nonav']} - {FM['full']}")
aud(un(cost['c_nolimit']), "DERIVED", f"{FM['c_nolimit']} - {FM['full']}")
aud("73 % and 46 %", "FINDING_1_ATTRIBUTION.md section 1 (three-seed means; INHERITED)")
aud("0.0003 to 0.0013", "FINDING_1_ATTRIBUTION.md section 1 table, column '3-seed range of the mean' (0.00027 to 0.00133; INHERITED)")
P("")
P("**What this does and does not say about the PASS.** It answers the route-supply worry of section 1 at the read-out level: `nav` is not the main carrier of the read-out's skill. The main carrier is `limit`, the logged bucket of the ego's own realised future max speed, the channel declared in the header. It does **not** say which input carries the ADE6 cut:"
  " no reduced read-out was substituted (they have narrower `xm / xs` than the 22 columns `subevals` hands a read-out; a column-map field is about 5 lines of instrument change, not built, FINDING_1 section 5). If the ADE6 cut follows the read-out's skill it rests mostly on a declared open channel; that is a hypothesis, not a measurement.")

# ---------------------------------------------------------------------------------------------------------------- 5
hdr("## 5. What this read cannot claim")
SEC[0] = "5"
P("* **Not a driving result, not closed loop.** T1-class open loop: X_s's own held S / T outputs, five cells replaced; ADE6 at this tier is not a capability claim (PREREG section 6). The T1-kin read (self-action open loop) is the first tier at which one can be made, and it has not been run for this arm.")
P("* **Not a lever claim for any trained arm.** Two O training runs, one read-out fit, zero training. The episode-cluster bootstrap is blind to training variance (`H-ESTIM-SEED-1`): a separated interval from a one-seed arm is necessary, not sufficient, and this experiment has **no replicate arm**. The two seeds agree in sign and both clear the bar, which is the replication this design has; it does not measure the run-to-run spread.")
P("* **Not a statement about DAG-OACC / DAG-ORC.** The base is M2b-mixo. DAG-trained O has seen its own rollouts, so its response to the five cells may differ (PREREG section 6, infeasible-as-written item 1).")
P("* **Not an attribution of the ADE6 cut.** The 22-dim ego block carries `nav` (a route input, optimistic on PhysicalAI) and the logged `limit` lane beside the P10 kinematics. The read-out's skill was attributed (section 4.5: it rests mostly on `limit`), but no reduced read-out was substituted, so which input carries the ADE6 cut is unmeasured.")
P("* **Read with an open channel.** The PASS was read with the logged `limit` lane, a channel declared under `D-SPEEDBAND-EGO-ECHO-1` (the bucket of the ego's own realised future max speed; admissible by PI ruling as a declared input, never a training signal; T reads it too). It is not a causal-bucket result: the T1-kin lane uses `--limit causal` (section 6).")
P("* **Not a complete longitudinal read.** Distance keeping and the anti-echo block are UNAVAILABLE (n = 0), so the LONGITUDINAL family is half measured (section 3). The `egoshuf` control has no family run. TACTICAL cannot move by construction on the three head rows, and STRATEGIC cannot move at all.")
P("* **A selection disclosure stands.** `mlp_ego` was commissioned after the eval-141 beside table was read; the TRAIN-only rule (S13) picks the same read-out, and this is the third read-out examined on these windows.")
P("* **Disclosed design limits that still apply** (PREREG section 11): tick frame only, no ego history; `nav` and the logged `limit` bucket ride in the ego block; the fit used the eval's age cycle, not the trainer's random ages; the `egoshuf` permutation is global, so it tests harm against far windows, not near neighbours; `lin_ego` was fitted and is untested here.")

# ---------------------------------------------------------------------------------------------------------------- 6
hdr("## 6. Next (Rule Zero): the PASS leaves a committed next; the attribution is measured and the follow-up is being pre-registered")
SEC[0] = "6"
m6 = re.search(r"(?s)(## 6\. .*?)(?=\n## 7\.)", PRET)
assert m6 is not None
S6 = m6.group(1).strip("\n")
assert "T1-kin" in S6 and "attribution" in S6.lower() and "pair-compatibility" in S6
P("**Committed next, verbatim from SUBREAD.json `next`:** *\"" + nxt + "\"*.")
P("")
P("**PREREG_V8F_SPEEDSUB section 6, verbatim** (the Master Mind's design; the file at tip `ea5e3d3b`, sha256 `" + PRE_SHA[:8] + "…`):")
P("")
for ln in S6.split("\n"):
    P("> " + ln if ln.strip() else ">")
for tok in re.findall(r"(?<![\w.-])(?:\d[\d,]*(?:\.\d+)?)", S6):
    aud(tok, "PREREG_V8F_SPEEDSUB.md section 6 (verbatim quote)")
P("")
P("**Status of that section's prerequisites at the time of this document (done by the prerequisites agent, not by this document):**")
P("")
P("| prerequisite (PREREG section 6) | status |", "|---|---|")
P("| the attribution the 22-dim block hides: re-fit the read-out as `p10f + frame_ok` only, and as `p10f + limit + frame_ok` without `nav`, on `train/train_arrays.npz` | **done: FINDING_1 (MEASURED; section 4.5).** The skill survives without `nav`, not without `limit` |")
P("| pair-compatibility of a DAG base with the banked read-out (the same `--kin-mode yaw` and the same `cons_mu / cons_sd` cell standardisation; checkable, as SPEEDAVG's `pair_violations`) | reported finished by the prerequisites agent; **its result is not cited in this RESULT** |")
P("| the base checkpoints' paths, shas and the rollout command (UNVERIFIED in the PREREG) | reported finished by the prerequisites agent; **its result is not cited in this RESULT** |")
P("")
P("")
P("**The T1-kin lane uses `--limit causal`, so the follow-up refits the read-out on the causal bucket.** The T1-kin rollouts of the DAG line feed `limit` as a causal bucket, `speed_max_bin` of the max measured ego speed over a trailing window of rows (20 rows, FINDING_1 section 3), computed by `causal_bucket` in `stack/scripts/v8f_t1_eval.py`, which reads nothing after the current row (the function and its docstring are read at tip; the lane assignment is INHERITED from FINDING_1 and the coordinator). "
  "It is not the logged sidecar the banked read-out was fitted on. The banked weights on `limit` were learned against the logged bucket, the future-speed oracle, so under `--limit causal` the read-out's strongest input changes meaning; FINDING_1 estimates the skill would sit between the neither design and the full block (ESTIMATED, not measured). "
  "The follow-up is therefore a new TRAIN fit of the read-out on the causal bucket, recomputed per window from `native_tracks.pt` (an extract-like step over the 306,566 train windows, not done; whether the tracks cover every train clip is not checked, FINDING_1 section 3). "
  "The driving model stays untrained, so the substitution stays training-free for O; what PREREG section 6 (quoted above) calls the banked read-out becomes a refitted one. **That follow-up is pre-registered as `PREREG_V8F_SPEEDSUB_T1` (landed `842ec2ac`, before its read-out was fitted and before any T1-kin substitution rollout).**")
aud("20", "FINDING_1_ATTRIBUTION.md section 3: causal bucket over rows [j-20, j] (INHERITED)")
P("")
P("What is not blocked and can proceed in the meantime, zero GPU, in the order of the largest expected information per hour (these are suggestions of this document for the Master Mind; none is a registered criterion, and the committed next above is unchanged):")
P("")
P("1. **Carry a rule-violation guard into the T1-kin PREREG.** The plan's rule-violation rate rose under `ego` at both seeds (section 4.2), separated-worse, while ADE6 fell. A T1-kin PASS that cut ADE6 and raised this rate would be a different result from one that did not; pre-register the guard and its direction before the arm runs.")
P("2. **Paired intervals on the stop windows.** Section 4.3's point values put the `ego` loss on the stop windows and the `gt` gain there too. The strata's window lists are not in the dumps; an adapter that stores them (or recomputes `event_strata` from the same inputs) turns a pointer into a paired row. Cheap, CPU, minutes of compute after the adapter.")
P("3. **The distance-keeping adapter** (about 60 lines, PREREG section 9). Until it exists, DK-VIOL / DK-HW / DK-TG are not measured at this tier (the T1-class harness path). The T1-kin readers of the DAG line carry DK rows, so the T1-kin arm's DK half may come from those; whether they accept this arm is not checked here.")
P("4. **PREREG_V8F_SPEEDCELL is not triggered by a PASS** (PREREG section 7 names it for NULL or HARMS). It is still the cheapest way to learn which of the five cells carries the cut and whether `ego` helps or hurts per cell (ESTIMATED 1.0 to 1.5 CPU-h at 4 threads, PREREG section 7). Whether to run it before T1-kin is the Master Mind's call.")
aud("1.0 to 1.5", "PREREG section 7 (ESTIMATED CPU-h for 20 cheap evals)")
P("")
P("**What blocks the committed next:** its own PREREG (own bar, both seeds, a replicate; PREREG section 6), which that section says needs the attribution (done) and the pair-compatibility check first; it is pre-registered as `PREREG_V8F_SPEEDSUB_T1` (landed `842ec2ac`), and the causal-bucket refit above is part of it; its instrument is being built. No compute, spend or PI decision is named as a blocker by the PREREG for this zero-training step. The rollout cost is INHERITED (about 260 s per rollout, SPEEDREAD section 6 citing FLATDAG).")
aud("260", "PREREG section 6 (INHERITED: about 260 s per rollout)")

# ---------------------------------------------------------------------------------------------------------------- 7
hdr("## 7. Run record and provenance (artifact path, sha256)")
SEC[0] = "7"
P("**Run record** (`logs/run.out`, `logs/speedsub_after_t1.log`). No stop, no refusal, no re-run: every stage ran once, in order.")
P("")
P("| stage | start | end | wall |", "|---|---|---|---|")
P(f"| waiter | {WL[0].split(' waiting')[0][6:]} | {RT['run start (threads 4)'][1]} | the waiter held `run_speedsub.sh` until the DAG-ORC T1 chain finished and 7,138 MB were free (`logs/speedsub_after_t1.log`) |")
P(f"| `evals_s0` (5 modes) | {RT['evals_s0: start'][1]} | {RT['evals_s0: done ([ok] M2b-mixo_s0-speedsub-egoshuf)'][1]} | {mm_ss(d_s0)} |")
P(f"| `preflight_s0` | {RT['preflight_s0: start'][1]} | {RT[PRE_KEY][1]} | S7b, S7 and the moved fraction, before seed 1 |")
P(f"| `evals_s1` (5 modes) | {RT['evals_s1: start'][1]} | {RT['evals_s1: done ([ok] M2b-mixo_s1-speedsub-egoshuf)'][1]} | {mm_ss(d_s1)} |")
P(f"| `subread` | {RT['subread: start'][1]} | {RT['subread: done (ZZSPEEDSUB-DONE PASS)'][1]} | {mm_ss(d_sr)} |")
P(f"| families (`pairfam` x 4, `longlat` x 4) | {RT['subread: done (ZZSPEEDSUB-DONE PASS)'][1]} | {tstamp(RUN[-1])[1]} | {mm_ss(d_fam)} |")
aud(WL[0].split(' waiting')[0][6:], "logs/speedsub_after_t1.log first stamp")
aud("7,138", "logs/speedsub_after_t1.log: 'T1 DONE and 7138 MB available: launching run_speedsub.sh'")
for st in ("evals_s0: start", "evals_s0: done ([ok] M2b-mixo_s0-speedsub-egoshuf)", "preflight_s0: start", "evals_s1: start", "evals_s1: done ([ok] M2b-mixo_s1-speedsub-egoshuf)", "subread: start"):
    aud(RT[st][1], "logs/run.out stage stamp")
aud(mm_ss(d_s0), "DERIVED from run.out stamps", f"{d_s0} s"); aud(mm_ss(d_s1), "DERIVED from run.out stamps", f"{d_s1} s")
aud(mm_ss(d_sr), "DERIVED from run.out stamps", f"{d_sr} s"); aud(mm_ss(d_fam), "DERIVED from run.out stamps", f"{d_fam} s")
P("")
P("No SPEEDSUB stage stopped. The waiter log also carries a stop of the DAG-ORC T1 chain at 06:05:21 (a box-2 run-record control; that is not this run) and the waiter's re-arm at 06:11:48.")
aud("06:05:21; 06:11:48", "logs/speedsub_after_t1.log lines 'T1 chain STOPPED' and 'waiting for' (re-arm)")
assert any(l.startswith("10-28 06:05:21 T1 chain STOPPED") for l in WL) and any(l.startswith("10-28 06:11:48 waiting for") for l in WL)
P("")
P("The run script is `raw/2026-10-28-speedsub-draft/run/run_speedsub.sh` (sha256 `8255f69f…`, already in the repo and byte-identical to the package copy); the executed copy of the waiter's invocation is not separately hashed in the logs (UNVERIFIED which byte-copy the waiter called, the stage outputs match the script's stages). "
  "The run tree is `F:/TanitAD-scratch/speedsub_tmp/tree` (an overlay of SPEEDREAD's `f3971165` tree with the two staged scripts; `run/make_tree.sh`).")
aud("8255f69f", "sha256 of run/run_speedsub.sh (tip raw/2026-10-28-speedsub-draft/run/ and E: package copy; asserted by the hold builder)")
aud("f3971165", "run/make_tree.sh: SRC tree_f3971165 (git archive of commit f3971165)")
P("")
P("Repo paths are under `" + V8 + "` and are staged by the hold (`landing/mk_hold_speedsub_result.py`). Source paths are where the bytes were read from.")
P("")
P("| role | source path | repo path | sha256 |", "|---|---|---|---|")
prov = []


def pr(role, src, repo, h=None):
    h = h or shaf(src)
    prov.append((role, src, repo, h))
    P(f"| {role} | `{src}` | `{repo}` | `{h}` |")


pr("reader output (decision of record)", SUBP, PKG + "read/SUBREAD.json")
pr("reader outcome", W + "subread/OUTCOME.txt", PKG + "read/OUTCOME.txt")
pr("independent re-derivation (this package)", RDP, PKG + "read/REDERIVE.json")
pr("re-derivation script", OUT + "rederive_sub.py", PKG + "read/rederive_sub.py")
pr("re-derivation console output", OUT + "rederive_sub.out", PKG + "read/rederive_sub.out")
for s in SEEDS:
    for m in MODES:
        pr(f"eval dump s{s} {m}", f"{W}eval_s{s}/eval_M2b-mixo_s{s}-speedsub-{m}.json", PKG + f"eval_s{s}/eval_M2b-mixo_s{s}-speedsub-{m}.json")
for s in SEEDS:
    for ext in ("json", "out"):
        pr(f"longlat s{s} ({ext})", f"{W}families/longlat_s{s}.{ext}", PKG + f"families/longlat_s{s}.{ext}")
    for a in ("ego", "gt"):
        for ext in ("json", "md", "out"):
            pr(f"pairfam {a} s{s} ({ext})", f"{W}families/pairfam_{a}_s{s}.{ext}", PKG + f"families/pairfam_{a}_s{s}.{ext}")
for f_, r_ in (("run.out", "logs/run.out"), ("evals_s0.log", "logs/evals_s0.log"), ("evals_s1.log", "logs/evals_s1.log"), ("preflight_s0.log", "logs/preflight_s0.log"), ("subread.log", "logs/subread.log")):
    pr(f"log {f_}", W + f_, PKG + r_)
pr("waiter log", WAIT, PKG + "logs/speedsub_after_t1.log")
pr("run wrapper generated by the run script", W + "run_with_taniteval.py", PKG + "run/run_with_taniteval.py")
pr("PREREG at the tip (already in the repo)", "git show ea5e3d3b:Project Steering/PREREG_V8F_SPEEDSUB.md", "(repo root) Project Steering/PREREG_V8F_SPEEDSUB.md", PRE_SHA)
pr("instrument (already in the repo)", "F:/TanitAD-scratch/speedsub_tmp/tree/stack/scripts/v8f_mini_eval_speedread.py", "(repo root) stack/scripts/v8f_mini_eval_speedread.py", shaf("F:/TanitAD-scratch/speedsub_tmp/tree/stack/scripts/v8f_mini_eval_speedread.py"))
pr("family tool, LONGITUDINAL / LATERAL (already in the repo)", "F:/TanitAD-scratch/speedsub_tmp/tree/stack/scripts/families_longlat_sub.py", "raw/2026-10-28-speedsub-draft/families/families_longlat_sub.py", shaf("F:/TanitAD-scratch/speedsub_tmp/tree/stack/scripts/families_longlat_sub.py"))
pr("SPEEDREAD READ.json (already in the repo; S5 to S7b inputs)", SR + "read/READ.json", "raw/2026-10-28-speedread/read/READ.json")
pr("SPEEDREAD fit.json (already in the repo)", SR + "fit/fit.json", "raw/2026-10-28-speedread/fit/fit.json")
P(f"| FINDING_1 attribution (the prerequisites agent's file; not staged by this hold) | `{F1P}` | none | `{shaf(F1P)}` |")
P(f"| attribution refits (not staged by this hold) | `{ATTP}` | none | `{shaf(ATTP)}` |")
P(f"| read-out `readout.pt` (not staged; single copy on F:) | `{SR}fit/readout.pt` | none | `{shaf(SR + 'fit/readout.pt')}` |")
P(f"| checkpoint X_0 `M2b-mixo_s0` (not staged) | `{XPIN['0'][1]}` | none | `{XPIN['0'][0]}` |")
P(f"| checkpoint X_1 `M2b-mixo_s1` (not staged) | `{XPIN['1'][1]}` | none | `{XPIN['1'][0]}` |")
P("")
import glob, os
_ev = [os.path.getsize(f_) for f_ in glob.glob(W + "eval_s*/eval_*.npz")]
_ce = [os.path.getsize(f_) for f_ in glob.glob(W + "eval_s*/speedread_cells_*.npz")]
assert len(_ev) == 10 and len(_ce) == 8
SZ = f"{min(_ev) / 1e6:.1f} to {max(_ev) / 1e6:.1f} MB and {min(_ce) / 1e6:.1f} to {max(_ce) / 1e6:.1f} MB each"
aud(SZ, "DERIVED: os.path.getsize of the ten eval_*.npz and eight speedread_cells_*.npz under F:/TanitAD-scratch/speedsub/eval_s{0,1}/ (decimal MB)")
P(f"**Not in the repo, single copy on F: (dev-box scratch).** The per-mode `eval_*.npz` and `speedread_cells_*.npz` dumps under `F:/TanitAD-scratch/speedsub/eval_s{{0,1}}/` ({SZ}; `REDERIVE.json` was computed from them and the family JSONs record the `ego` / `own` / `gt` npz sha256 of their inputs), SPEEDREAD's banked `eval_*.npz`, "
  "`speedread_cells_*-identity.npz` and `fit/readout.pt` (controls S5, S6, S7b read them), the run tree, and the checkpoints. If F: is lost, controls S5, S6 and S7b and the family tools cannot be reproduced; the JSONs, logs and this audit remain.")
P("")
P("The in-repo instrument tests are INHERITED from the draft package (`raw/2026-10-28-speedsub-draft/`, PREREG section 10): 66 tests pass (39 new, 27 SPEEDREAD), 31 of the 39 new tests fail on the unmodified instrument, 28 of 28 mutants killed. They were not re-run for this document.")
aud("66; 39; 27; 31; 28", "PREREG section 10 (INHERITED; logs/pytest_staged_green.txt, logs/pytest_RED_on_original.txt, logs/mutation_results.json in the draft package)")

TXT = "\n".join(L) + "\n"
TXT = TXT.replace("\n\n\n", "\n\n")
FORBID = ("void", "superseded", "retracted", "formerly", "no longer", "correction", "withdrawn")
for w in FORBID:
    assert w not in TXT.lower(), w
assert "\r" not in TXT and not re.search(r"hf_[A-Za-z0-9]{20,}", TXT)
open(OUT + "RESULT_V8F_SPEEDSUB.md", "w", encoding="utf-8", newline="\n").write(TXT)

# ---------------------------------------------------------------------------------------------------------------- register row
SEC[0] = "register"
cap0, cap1 = ROW["0"]["capture"], ROW["1"]["capture"]
rowtxt = (
    f"**PASS (MEASURED 2026-10-28, a screening result, no amendment; all {NFLAG} reader control flags PASS and the rows recomputed from the per-window dumps).** "
    f"Replacing only the five speed cells O reads by the banked ego-only read-out `mlp_ego` (no re-fit) cut X_s's own-code ADE6 on eval-141 by {sg(E['0']['delta'])} [{sg(E['0']['lo'])}, {sg(E['0']['hi'])}] m at s0 and {sg(E['1']['delta'])} [{sg(E['1']['lo'])}, {sg(E['1']['hi'])}] m at s1, "
    f"separated-better at both seeds and at or beyond the {MINUS}{un(F_PASS, 5)} m bar (F/2) at both; s1 clears the bar by {un(margin['1'])} m only, and at both seeds the interval's upper end lies above {MINUS}{un(F_PASS, 5)}. "
    f"The deliberate-regression arm `egoshuf` reads {sg(X['0']['delta'])} [{sg(X['0']['lo'])}, {sg(X['0']['hi'])}] / {sg(X['1']['delta'])} [{sg(X['1']['lo'])}, {sg(X['1']['hi'])}] m, separated-worse at both seeds. "
    f"CAPTURE of the `gt` ceiling (T0, a control; {sg(G['0']['delta'])} / {sg(G['1']['delta'])} m) is {un(cap0)} / {un(cap1)}. "
    "Tier: T1-class open loop (X_s's own held S / T outputs; only the five speed cells are replaced), never closed loop; ADE6 here is not a capability claim. "
    "Estimator: paired episode-cluster bootstrap, B = 2,000, 141 episodes, per seed; it answers the episode question only; no replicate arm (`H-ESTIM-SEED-1`); two O training runs and one read-out fit; base M2b-mixo, not DAG-OACC. "
    "Selection disclosure stands: the arm was chosen after the eval-141 beside table was read (S13 restates the rule on TRAIN fold 0), the third read-out examined on these windows. "
    f"Four families beside (RESULT section 3, per family, pointers only): LONGITUDINAL speed, along-track and accel MAE separated-better at both seeds, distance keeping and anti-echo NOT COMPUTED (UNAVAILABLE, n = 0; work item); LATERAL mixed pointers, none separated-worse; "
    f"TACTICAL decisions not separated, goal-point error separated-better at both seeds, heads exactly 0 by construction, anchor selection and selected-vs-executed UNAVAILABLE (n = 2,399; work item); STRATEGIC exactly 0.0000 by construction. "
    f"Beside, post-hoc and not registered: the plan rule-violation rate rises under `ego`, separated-worse at both seeds ({sg(vany[0]['delta'])} / {sg(vany[1]['delta'])}), and the {th(EJ[('0', 'own')]['raw']['strata_ade6']['stop']['n'])} stop windows read worse under `ego` than `own` at both seeds (point values, no interval). "
    f"Declared open channel: the PASS was read with the logged `limit` lane (the bucket of the ego's own realised future max speed, `D-SPEEDBAND-EGO-ECHO-1`; a declared input, never a training signal; T reads it too). Attribution of the read-out's skill (FINDING_1, TRAIN fold 0, mean hold-out RMSE in SD units, lower is better): full {FM['full']}, no nav {FM['b_nonav']}, no limit {FM['c_nolimit']}, p10f + frame_ok only {FM['a_p10f_fok']}: the skill survives without nav, not without limit; no reduced read-out was substituted, so the ADE6 attribution is unmeasured. "
    "Committed next (PREREG section 6): the T1-kin zero-training substitution on the DAG-OACC / DAG-ORC base under its own PREREG; the T1-kin lane uses `--limit causal`, so the follow-up refits the read-out on the causal bucket, pre-registered as `PREREG_V8F_SPEEDSUB_T1` (landed `842ec2ac`, row `H-V8F-SPEEDSUB-T1-1`). "
    "`RESULT_V8F_SPEEDSUB.md`, `raw/2026-10-28-speedsub-result/read/SUBREAD.json`. Was:")
for w in FORBID:
    assert w not in rowtxt.lower(), w
assert "|" not in rowtxt and "\n" not in rowtxt and rowtxt.endswith("Was:")
aud("register row status prefix", "same cells as the RESULT sections 1, 3, 4.2, 4.3, 4.5; SUBREAD.json rows.<s>.{ego,egoshuf,gt,capture}; REDERIVE.json viol_any; eval_s0 strata_ade6.stop.n")
aud(f"{NFLAG}", "SUBREAD.json controls (27 flags)")
EVID_OLD, EVID_NEW = "PRE-REGISTERED", "MEASURED (screening)"
rr = [
    "# H-V8F-SPEEDSUB-1: replacement text for the register row (staged, not landed)",
    "",
    "Generated by `gen_result.py` from `SUBREAD.json` (sha256 `" + SUB_SHA[:8] + "…`). The hold builder `mk_hold_speedsub_result.py` reads the paragraph between the two markers, inserts it at the start of the row's status cell, keeps the previous status text after `Was: `, and",
    "changes the evidence cell. The claim cell and the source cell stay. The file's line count does not change (the row is one line).",
    "",
    "Row: the line of `Project Steering/GOALS_AND_CLAIMS.md` that starts `| **H-V8F-SPEEDSUB-1** (2026-10-28) |`. As `H-V8F-DAG-ORC-1` was promoted: new result first, `Was:` then the earlier text unchanged.",
    "",
    "## Status cell (new prefix)",
    "",
    "<!-- STATUS-BEGIN -->",
    rowtxt,
    "<!-- STATUS-END -->",
    "",
    "## Evidence cell",
    "",
    f"`{EVID_OLD}` becomes `{EVID_NEW}`. The ADE6 outcome is MEASURED (PASS, a screening result at T1-class open loop); nothing in the row is a lever claim for a trained arm.",
    "",
    "<!-- EVID-OLD: " + EVID_OLD + " -->",
    "<!-- EVID-NEW: " + EVID_NEW + " -->",
    "",
]
RRT = "\n".join(rr)
assert "\r" not in RRT
open(OUT + "REGISTER_ROW.md", "w", encoding="utf-8", newline="\n").write(RRT)

# ---------------------------------------------------------------------------------------------------------------- audit
A_ = ["# NUMBERS_AUDIT: every number in RESULT_V8F_SPEEDSUB.md and REGISTER_ROW.md, with the source it was read from", "",
      "Generated by `gen_result.py` in the same run that wrote the RESULT, so each rendered value below is the value that was printed. Paths are JSON key paths into `SUBREAD.json`"
      f" (sha256 `{SUB_SHA}`), `REDERIVE.json` (sha256 `{RD_SHA}`), the family JSONs `longlat_s<s>.json` / `pairfam_<arm>_s<s>.json` and the per-mode eval JSONs `eval_s<s>/eval_M2b-mixo_s<s>-speedsub-<mode>.json`, unless a file is named."
      " `{a,b}` means each of those keys. Derived numbers show the arithmetic.", "",
      "Cross-checks the script asserts before it writes (a failure stops the build):", "",
      "* The PREREG section 5 table, coded in the script, applied to `SUBREAD.json` `rows` gives PASS, equal to `SUBREAD.json` `outcome` and to `REDERIVE.json` (rounded and unrounded points).",
      "* `REDERIVE.json` (from the per-window dumps, not through the instrument's reader) equals `SUBREAD.json` on delta, lo, hi to 4 dp for `ego`, `egoshuf` and `gt` at both seeds, and on the per-cell RMSE table; it recomputes S1 to S13 within the PREREG tolerances.",
      f"* All {NFLAG} control flags in `SUBREAD.json` are true; `capture` = `ego` / `gt` delta; `egoshuf` is separated-worse at both seeds; `F_pass_m` = 0.1207 / 2.",
      "* Means in the dumps minus each other equal the paired deltas within 0.0001; the `gt` rows equal the tip's SPEEDREAD `READ.json` `rows.<s>.gt` within 5e-4.",
      "* The family tools' controls (LL-1..3, pairfam identity / reproduction) are true at both seeds; the longlat ADE6 rows equal the reader's rows; the 49-cell census (12 + 31 + 6) holds; the qualitative statements in section 3 (which rows are separated, which are exactly zero) are asserted against the cells.",
      "* The rule-violation rates in section 4.2 equal each dump's `raw.consistency_a.*.rate_all`.",
      "* The instrument at the tip, the run tree's copy, `families_longlat_sub.py`, `readout.pt`, the PREREG and SPEEDREAD's `READ.json` hash to their pins.",
      "* Coverage (`audit_cover.py`, run at the end of the build and asserted to report 0 uncovered): every numeric token of the RESULT and of the register row text has a matching token in the table below."
      " Not counted as data: section references, list and heading numerals, hex digests, dates inside repo paths.",
      "", "| section | rendered | source | arithmetic / note |", "|---|---|---|---|"]
for sec_, ren, src, note in AUD:
    A_.append("| " + " | ".join(x.replace("|", "\\|").replace("\n", " ") for x in (sec_, ren, src, note)) + " |")
open(OUT + "NUMBERS_AUDIT.md", "w", encoding="utf-8", newline="\n").write("\n".join(A_) + "\n")
print("wrote RESULT", len(TXT), "chars,", len(AUD), "audit entries, provenance rows", len(prov))
import importlib.util
_spec = importlib.util.spec_from_file_location("audit_cover", OUT + "audit_cover.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_un = _mod.main()
assert all(len(v) == 0 for v in _un.values()), _un
