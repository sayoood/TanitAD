# -*- coding: utf-8 -*-
"""Re-verify every quotable number in Paper/TANITAD_PAPER.md sections 16-17
against its banked artifact.  Two-sided: each row asserts (a) the value is in
the artifact at the stated path, and (b) the literal string appears in the
paper's new sections.  A row that cannot be read is INCONCLUSIVE, never PASS.
"""
import json, os, sys, time, glob, io
sys.stdout.reconfigure(encoding="ascii", errors="replace")

ART = r"C:\Users\Admin\AppData\Local\Temp\claude\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD\f407bc82-7969-457c-a947-6be2014fee89\scratchpad\art"
PAPER = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\Paper\TANITAD_PAPER.md"

def rd(p, tries=200, s=0.4):
    last = None
    for _ in range(tries):
        try:
            b = open(p, "rb").read()
            if b: return b
            last = "EMPTY"
        except Exception as e: last = repr(e)
        time.sleep(s)
    return None

def art(pat):
    hits = glob.glob(os.path.join(ART, "*" + pat))
    if not hits: raise SystemExit("missing local artifact copy: " + pat)
    return json.load(io.open(hits[0], encoding="utf-8"))

def arttxt(pat):
    hits = glob.glob(os.path.join(ART, "*" + pat))
    if not hits: raise SystemExit("missing local artifact copy: " + pat)
    return io.open(hits[0], encoding="utf-8", errors="replace").read()

b = rd(PAPER)
if b is None: raise SystemExit("INCONCLUSIVE: paper unreadable")
P = b.decode("utf-8")
i16 = P.find("## 16. Four results")
if i16 < 0: raise SystemExit("INCONCLUSIVE: section 16 not found")
NEW = P[i16:]

T1   = art("refcv4b_t1.json")
FULL = art("paired_v4b_vs_v3_FULL.json")
ECHO = art("ECHO_INDEX_CORRECTION.json")
NAVP = art("paired_navpred.json")
TREA = art("navpred_treated_subsets.json")
PROB = art("probe_results.json")
MOT  = art("motion_probe.json")
BANK = art("bank_meta.json")
CENS = art("p1_fullcensus.json")
PANEL= art("panel_report.json")
DOSE = arttxt("wkappa_dose.txt")
FEAS = arttxt("feas_audit.txt")
SEED = arttxt("seed_floor.txt")
FRON = arttxt("frontier.txt")
DDV2 = arttxt("diffusiondrivev2_model_rl.py")
LIB  = art("library.json")

pdg = T1["paired_decision_grade"]
ff  = T1["arms"]["os"]["four_families"]
strat = T1["refcv3"]["strategic"]

rows = []  # (id, literal-as-printed, artifact-derived value, artifact ref)
def R(i, lit, val, ref):
    rows.append((i, lit, val, ref))

def f(x, n): return ("%." + str(n) + "f") % x

# ---- 16.1
R("16.1-a", "\u22120.1444", f(FULL["paired_B_minus_A"]["delta"],4).replace("-","\u2212"), "paired_v4b_vs_v3_FULL.json:paired_B_minus_A.delta")
R("16.1-b", "[\u22120.1647, \u22120.1227]", "[%s, %s]"%(f(FULL["paired_B_minus_A"]["lo"],4).replace("-","\u2212"), f(FULL["paired_B_minus_A"]["hi"],4).replace("-","\u2212")), "paired_B_minus_A.lo/hi")
R("16.1-c", "0.2975", f(FULL["A_mean"],4), "paired_v4b_vs_v3_FULL.json:A_mean")
R("16.1-d", "0.4419", f(FULL["B_mean"],4), "paired_v4b_vs_v3_FULL.json:B_mean")
R("16.1-e", "4,823", "{:,}".format(FULL["n_windows"]), "n_windows")
R("16.1-f", "141", str(FULL["n_episodes"]), "n_episodes")
R("16.1-g", "0.0000000000", "%.10f"%FULL["control_A_vs_A"]["delta"], "control_A_vs_A.delta")
R("16.1-h", "\u22120.3748", f(pdg["paired_os_minus_ha0"]["ade_m"]["delta"],4).replace("-","\u2212"), "paired_os_minus_ha0")
R("16.1-i", "\u22120.0021", f(pdg["paired_os_minus_ha"]["ade_m"]["delta"],4).replace("-","\u2212"), "paired_os_minus_ha")
R("16.1-j", "+0.0101", "+"+f(pdg["paired_os_minus_ha0ext"]["ade_m"]["delta"],4), "paired_os_minus_ha0ext")
R("16.1-k", "+0.1054", "+"+f(pdg["paired_os_navzero_minus_ha0ext"]["ade_m"]["delta"],4), "paired_os_navzero_minus_ha0ext")
R("16.1-l", "[+0.0874, +0.1241]", "[+%s, +%s]"%(f(pdg["paired_os_navzero_minus_ha0ext"]["ade_m"]["lo"],4), f(pdg["paired_os_navzero_minus_ha0ext"]["ade_m"]["hi"],4)), "same, lo/hi")
R("16.1-m", "\u22120.0953", f(pdg["paired_os_minus_navzero"]["ade_m"]["delta"],4).replace("-","\u2212"), "paired_os_minus_navzero")
R("16.1-n", "+0.9179", "+"+f(pdg["paired_oraclesel_minus_os"]["ade_m"]["delta"],4), "paired_oraclesel_minus_os")
for arm, lit in [("os","0.2975"),("ha","0.2996"),("ha0","0.6723"),("ha0_ext","0.2874"),("os_navshuf","0.3013"),("os_navzero","0.3928"),("oracle_sel","1.2154")]:
    R("16.1-ade-"+arm, lit, f(T1["arms"][arm]["intervals"]["metrics"]["ade_dense_m"]["mean"],4), "arms.%s.intervals.metrics.ade_dense_m.mean"%arm)
R("16.1-o", "0.2909", f(ff["longitudinal"]["speed_mae_mps"],4), "os.four_families.longitudinal.speed_mae_mps")
R("16.1-p", "0.2555", f(ff["longitudinal"]["along_mae_m"],4), "longitudinal.along_mae_m")
R("16.1-q", "0.8317", f(ff["longitudinal"]["target_speed_acc"]["within_0.5_mps"],4), "target_speed_acc")
import re as _re
R("16.1-r", "19,292", "{:,}".format(int(_re.search(r"the (\d+) horizon", ff["longitudinal"]["target_speed_acc_note"]).group(1))), "target_speed_acc_note n_steps")
R("16.1-s", "1.0049", f(ff["longitudinal"]["ego_progress"]["progress_ratio_mean"],4), "ego_progress.progress_ratio_mean")
R("16.1-t", "28.4748", f(ff["longitudinal"]["distance_keeping"]["mean_headway_min_m"],4), "distance_keeping.mean_headway_min_m")
R("16.1-u", "4.1185", f(ff["longitudinal"]["distance_keeping"]["mean_time_gap_min_s"],4), "mean_time_gap_min_s")
R("16.1-v", "24.8331", f(ff["longitudinal"]["distance_keeping"]["mean_min_ttc_s"],4), "mean_min_ttc_s")
R("16.1-w", "1,225", "{:,}".format(ff["longitudinal"]["distance_keeping"]["n"]), "distance_keeping.n")
R("16.1-x", "472", str(ff["longitudinal"]["distance_keeping"]["n_closing"]), "n_closing")
R("16.1-y", "0.0979", f(ff["lateral"]["cross_mae_m"],4), "lateral.cross_mae_m")
R("16.1-z", "0.008097", f(ff["lateral"]["curvature_mae_1pm"],6), "lateral.curvature_mae_1pm")
R("16.1-aa","0.006802", f(T1["arms"]["ha0"]["four_families"]["lateral"]["curvature_mae_1pm"],6), "ha0.lateral.curvature_mae_1pm")
R("16.1-ab","0.003712", f(T1["arms"]["ha0_ext"]["four_families"]["lateral"]["curvature_mae_1pm"],6), "ha0_ext.lateral.curvature_mae_1pm")
R("16.1-ac","0.8289", f(ff["tactical"]["lateral_decision"]["kappa"],4), "tactical.lateral_decision.kappa")

# ---- 16.2
R("16.2-a", "0.4852", f(strat["conditionings"]["nav_true"]["kappa"],4), "refcv3.strategic.conditionings.nav_true.kappa")
R("16.2-b", "[0.4057, 0.5671]", "[%s, %s]"%(f(strat["conditionings"]["nav_true"]["ci"]["kappa"]["lo"],4), f(strat["conditionings"]["nav_true"]["ci"]["kappa"]["hi"],4)), "ci.kappa")
R("16.2-c", "0.7786", f(strat["conditionings"]["nav_true"]["accuracy"],4), "nav_true.accuracy")
R("16.2-d", "3,622", "{:,}".format(strat["n_route_labeled"]), "n_route_labeled")
R("16.2-e", "0.6742", f(strat["conditionings"]["nav_true"]["majority_class_rate"],4), "majority_class_rate")
R("16.2-f", "0.6405", f(ECHO["CORRECTIONS"]["nav_echo_index"]["corrected"],4), "ECHO_INDEX_CORRECTION.CORRECTIONS.nav_echo_index.corrected")
R("16.2-g", "0.3370", f(ECHO["CORRECTIONS"]["route_follows_SHUFFLED_NAV_under_shuffle"]["corrected"],4), "CORRECTIONS.route_follows_SHUFFLED_NAV")
R("16.2-h", "1,311", "{:,}".format(ECHO["THE_CLEAN_ANTI_ECHO_READ"]["n"]), "THE_CLEAN_ANTI_ECHO_READ.n")
R("16.2-i", "0.7124", f(ECHO["THE_CLEAN_ANTI_ECHO_READ"]["route_follows_LABEL"],4), "route_follows_LABEL")
R("16.2-j", "0.1739", f(ECHO["THE_CLEAN_ANTI_ECHO_READ"]["route_follows_SHUFFLED_NAV"],4), "route_follows_SHUFFLED_NAV")
R("16.2-k", "1.0806", f(ECHO["THIRD_DEFECT_mutual_exclusivity"]["rates_sum_to"],4), "rates_sum_to")
R("16.2-l", "0.3012", f(NAVP["ade"]["os_navpred"]["mean"],4), "paired_navpred.ade.os_navpred.mean")
R("16.2-m", "[0.2721, 0.3314]", "[%s, %s]"%(f(NAVP["ade"]["os_navpred"]["ci95"][0],4), f(NAVP["ade"]["os_navpred"]["ci95"][1],4)), "ade.os_navpred.ci95")
R("16.2-n", "\u22120.0914", f(NAVP["paired"]["os_navpred_minus_os_navzero"]["delta_m"],4).replace("-","\u2212"), "paired.os_navpred_minus_os_navzero")
R("16.2-o", "0.001031", f(NAVP["controls"]["C3_os_reproduction"]["abs_diff"],6), "controls.C3.abs_diff")
R("16.2-p", "1,908", "{:,}".format(NAVP["controls"]["C6_arm_is_not_degenerate"]["n_navpred_differs_from_navcmd"]), "controls.C6")
R("16.2-q", "3.0065", f(NAVP["controls"]["C5_coverage_identity"]["CONTROL_max_abs_path_diff_on_DIFFERENT_token_m"],4), "controls.C5 control")
tr = TREA["subsets"]["commanded_only"]
R("16.2-r", "1,743", "{:,}".format(tr["n_windows"]), "navpred_treated_subsets.commanded_only.n_windows")
R("16.2-s", "+0.0028", "+"+f(tr["os_navpred_minus_os"]["delta_m"],4), "commanded_only.os_navpred_minus_os")
R("16.2-t", "+0.0031", "+"+f(tr["os_navshuf_minus_os"]["delta_m"],4), "commanded_only.os_navshuf_minus_os")
R("16.2-u", "+0.0648", "+"+f(tr["os_navzero_minus_os"]["delta_m"],4), "commanded_only.os_navzero_minus_os")

# ---- 16.3 (text artifacts: assert the literal is present in the artifact text)
for i, lit, src in [("16.3-a","0.030982",DOSE),("16.3-b","0.040083",DOSE),("16.3-c","0.055369",DOSE),
                    ("16.3-d","0.038019",DOSE),("16.3-e","0.080478",DOSE),("16.3-f","0.8934",DOSE),
                    ("16.3-g","15.2704",DOSE),("16.3-h","0.9251",DOSE),
                    ("16.3-i","0.1701",FEAS),("16.3-j","0.1852",FEAS),("16.3-k","0.2963",FEAS),
                    ("16.3-l","0.0607",SEED.replace("0.06070","0.0607")),("16.3-m","0.00200",FRON)]:
    R(i, lit, lit if lit in src else "NOT-IN-ARTIFACT", "text artifact literal")

# ---- 16.4
g30 = PROB["lead_gap_m_cap30"]; c30 = PROB["lead_closing_mps_cap30"]
R("16.4-a", "+0.000000", "+%.6f"%g30["constant"]["r2"], "probe_results.lead_gap_m_cap30.constant.r2")
R("16.4-b", "\u22120.0513", f(g30["pix"]["r2"],4).replace("-","\u2212"), "gap.pix.r2")
R("16.4-c", "+0.2225", "+"+f(g30["dino"]["r2"],4), "gap.dino.r2")
R("16.4-d", "+0.3632", "+"+f(g30["field"]["r2"],4), "gap.field.r2")
R("16.4-e", "[+0.2069, +0.5088]", "[+%s, +%s]"%(f(g30["field"]["ci"][0],4), f(g30["field"]["ci"][1],4)), "gap.field.ci")
R("16.4-f", "+0.4145", "+"+f(g30["_paired"]["field_minus_pix"]["delta"],4), "gap._paired.field_minus_pix")
R("16.4-g", "[+0.2018, +0.6120]", "[+%s, +%s]"%(f(g30["_paired"]["field_minus_pix"]["ci"][0],4), f(g30["_paired"]["field_minus_pix"]["ci"][1],4)), "same ci")
R("16.4-h", "+0.3995", "+"+f(g30["field"]["r2_within_clip"],4), "gap.field.r2_within_clip")
R("16.4-i", "\u22120.0013", f(g30["shuffle_within_clip"]["r2_within_clip"],4).replace("-","\u2212"), "gap.shuffle_within_clip.r2_within_clip")
R("16.4-j", "+0.0307", "+"+f(g30["shuffle_within_clip"]["r2"],4), "gap.shuffle_within_clip.r2")
R("16.4-k", "+0.3326", "+"+f(g30["field"]["r2"]-g30["shuffle_within_clip"]["r2"],4), "field.r2 minus shuffle.r2")
R("16.4-l", "1,586", "{:,}".format(g30["field"]["n_score"]), "gap n_score")
R("16.4-m", "+0.0061", "+"+f(c30["field"]["r2"],4), "closing.field.r2")
R("16.4-n", "[\u22120.0406, +0.0513]", "[%s, +%s]"%(f(c30["field"]["ci"][0],4).replace("-","\u2212"), f(c30["field"]["ci"][1],4)), "closing.field.ci")
R("16.4-o", "1,491", "{:,}".format(c30["field"]["n_score"]), "closing n_score")
R("16.4-p", "+0.5883", "+"+f(g30["dino_rff"]["r2"],4), "gap.dino_rff.r2")
R("16.4-q", "+0.4411", "+"+f(g30["field_rff"]["r2"],4), "gap.field_rff.r2")
R("16.4-r", "\u22120.1472", f(g30["_paired"]["field_rff_minus_dino_rff"]["delta"],4).replace("-","\u2212"), "field_rff_minus_dino_rff")
R("16.4-s", "+0.1407", "+"+f(g30["_paired"]["field_minus_dino"]["delta"],4), "field_minus_dino")
R("16.4-t", "+0.0052", "+"+f(MOT["lead_closing_mps_cap30"]["field_diff"]["r2"],4), "motion.closing.field_diff.r2")
R("16.4-u", "\u22120.0009", f(MOT["lead_closing_mps_cap30"]["_paired"]["field_diff_minus_field_t"]["delta"],4).replace("-","\u2212"), "motion paired field_diff-field_t")
R("16.4-v", "+0.0004", "+"+f(MOT["lead_gap_m_cap30"]["field_diff"]["r2"],4), "motion.gap.field_diff.r2")
R("16.4-w", "+0.3647", "+"+f(MOT["lead_gap_m_cap30"]["field_t"]["r2"],4), "motion.gap.field_t.r2")
R("16.4-x", "\u22120.3644", f(MOT["lead_gap_m_cap30"]["_paired"]["field_diff_minus_field_t"]["delta"],4).replace("-","\u2212"), "motion.gap paired")
R("16.4-y", "407", str(BANK["freeze"]["n_tensors"]), "bank_meta.freeze.n_tensors")
R("16.4-z", "18,477", "{:,}".format(BANK["n_rows"]), "bank_meta.n_rows")
R("16.4-aa","3,615", "{:,}".format(BANK["n_lead_rows"]), "bank_meta.n_lead_rows")
R("16.4-ab","21,109", "{:,}".format(BANK["ckpt_step"]), "bank_meta.ckpt_step")

# ---- 16.5
off = CENS["off"]; onm = CENS["on_mutation"]
R("16.5-a", "138", str(off["n_in_optimizer"]), "p1_fullcensus.off.n_in_optimizer")
R("16.5-b", "42", str(off["n_unreached"]), "off.n_unreached")
R("16.5-c", "5,305,667", "{:,}".format(off["unreached_numel"]), "off.unreached_numel")
R("16.5-d", "10,169,731", "{:,}".format(CENS["trainable_numel"]), "trainable_numel")
R("16.5-e", "52.2", "%.1f"%(100.0*off["unreached_numel"]/CENS["trainable_numel"]), "derived share")
R("16.5-f", "1,577,216", "{:,}".format(onm["unreached_numel"]), "on_mutation.unreached_numel")
R("16.5-g", "15.5", "%.1f"%(100.0*onm["unreached_numel"]/CENS["trainable_numel"]), "derived share (mutation)")

# ---- 16.6
for i, tok, want in [("16.6-a","cliprange",0),("16.6-b","clip_range",0),("16.6-c","kl_coef",0),
                     ("16.6-d","kl_div",0),("16.6-e","ref_model",0),("16.6-f","old_log_probs",0)]:
    R(i, "%s absent"%tok, ("%s absent"%tok) if DDV2.count(tok)==want else "PRESENT(%d)"%DDV2.count(tok), "source token census")
for i, tok in [("16.6-ctl1","advantages"),("16.6-ctl2","per_token_logps"),("16.6-ctl3","PDMScorer"),("16.6-ctl4","std_dev_t_add")]:
    R(i, "%s present (control)"%tok, ("%s present (control)"%tok) if DDV2.count(tok)>0 else "ABSENT-CONTROL-FAILED", "source control")
R("16.6-g", "exp(logp \u2212 logp.detach())", "present" if "torch.exp(per_token_logps - per_token_logps.detach()) * advantages" in DDV2 else "ABSENT", "loss line")
R("16.6-h", "std_dev_t_add = 0.0 both branches", "yes" if DDV2.count("std_dev_t_add = torch.tensor(0.0)")==2 else "no(%d)"%DDV2.count("std_dev_t_add = torch.tensor(0.0)"), "sampler branches")
R("16.6-i", "2512.07745", "2512.07745" if "2512.07745" in LIB["entries"] else "MISSING", "library.json key")
R("16.6-j", "076ce47e0b0a323c9bb0072432e7f1219ccdcf645b8ff6db172f1bb9efb85023", LIB["entries"]["2512.07745"]["sha256"], "library sha256")

# ---- 17.1
def cells(o, path, acc):
    if isinstance(o, dict):
        if "separated" in o and "delta" in o:
            acc.append(bool(o["separated"])); return
        for k, v in o.items(): cells(v, path+"/"+k, acc)
acc = []
cells(PANEL["arms"]["A0b_replicate"]["paired_vs_A0"], "", acc)
R("17.1-a", "6 separated", "%d separated"%sum(acc), "panel_report.A0b_replicate.paired_vs_A0")
R("17.1-b", "42 cells", "%d cells"%len(acc), "same")
R("17.1-c", "14.3", "%.1f"%(100.0*sum(acc)/len(acc)), "derived rate")
R("17.1-d", "0.0607", "0.0607" if "0.06070" in SEED else "NOT-IN-ARTIFACT", "seed_floor.txt ADE dev-box")
R("17.1-e", "0.1035", "0.1035" if "0.10350" in FRON else "NOT-IN-ARTIFACT", "frontier.txt binding ADE floor")

# ---------------------------------------------------------------- report
ok = fail = miss = 0
print("%-14s %-34s %-34s %s" % ("id", "paper literal", "artifact value", "verdict"))
print("-"*118)
for i, lit, val, ref in rows:
    litnorm = lit.replace("\u2212", "-")
    valnorm = str(val).replace("\u2212", "-")
    match = (litnorm.strip("+") == valnorm.strip("+")) or (litnorm in valnorm) or (valnorm in litnorm)
    exempt = ("absent" in lit) or ("present (control)" in lit) or ("both branches" in lit) or ("logp.detach" in lit)
    inpaper = True if exempt else ((lit in NEW) or (lit.replace(u"−","-") in NEW))
    if exempt and litnorm==valnorm: match=True
    if not match:
        v = "MISMATCH"; fail += 1
    elif not inpaper:
        v = "NOT-IN-PAPER"; miss += 1
    else:
        v = "PASS"; ok += 1
    if True:
        print("%-14s %-34s %-34s %s   <- %s" % (i, lit[:34], str(val)[:34], v, ref))
print("-"*118)
print("PASS %d   MISMATCH %d   NOT-IN-PAPER %d   TOTAL %d" % (ok, fail, miss, len(rows)))
