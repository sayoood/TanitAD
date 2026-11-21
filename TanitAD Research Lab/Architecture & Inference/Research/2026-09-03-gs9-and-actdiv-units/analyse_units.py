"""D-ACTDIV-UNITS-INVARIANCE - decide SPEC section 4's B0-B5 from the three raw JSONs."""
import json, math

BANK = (r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\TanitAD Research Lab\Architecture & "
        r"Inference\Research\2026-09-03-anchored-actdiv-refav1\raw\actdiv_anchored_refav1_step1000.json")
SCR = (r"C:\Users\Admin\AppData\Local\Temp\claude\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
       r"\8e7cfa33-c625-47cf-88aa-711db80ac113\scratchpad\out")
J = {k: json.load(open(SCR + "\\actdiv_units_" + k + ".json", encoding="utf-8"))
     for k in ("kappa", "steer")}
B = json.load(open(BANK, encoding="utf-8"))
ARMS = ("incumbent_fp32", "clean_epoch_ema_bf16")
L_ENC = 2.9


def sig(x, n=6):
    return float("%.*g" % (n, x)) if isinstance(x, float) and math.isfinite(x) else x


def ax(u, arm, sp="tac"):
    return J[u]["arms"][arm]["read"]["spaces"][sp]["grid"]["per_axis"]


def ff(u, arm):
    return J[u]["arms"][arm]["read"]["full_field_exact"]


print("=" * 100)
print("units provenance:", json.dumps({k: v.get("action_units") for k, v in J.items()}, indent=1))
print("banked", B["generated_utc"], "| kappa", J["kappa"]["generated_utc"],
      "| steer", J["steer"]["generated_utc"])

print("\n" + "=" * 100)
print("B0  REPRODUCTION - the kappa pass vs the banked run, at 6 significant figures")
bad = 0
for arm in ARMS:
    rn, ro = J["kappa"]["arms"][arm]["read"], B["arms"][arm]["read"]
    checks = []
    for sp in ("tac", "pooled", "full_subsample"):
        gn, go = rn["spaces"][sp]["grid"], ro["spaces"][sp]["grid"]
        for key in ("F_sep", "F_over_null_p95", "scene_spread", "rel_units",
                    "rel_mag_at_material_level", "max_abs_displacement"):
            checks.append((sp + "." + key, gn[key], go[key]))
        for a in ("kappa", "accel"):
            checks.append((sp + "." + a + ".F_sep", gn["per_axis"][a]["F_sep"],
                           go["per_axis"][a]["F_sep"]))
            checks.append((sp + "." + a + ".F_over_p95", gn["per_axis"][a]["F_over_null_p95"],
                           go["per_axis"][a]["F_over_null_p95"]))
            checks.append((sp + "." + a + ".null_p95", gn["per_axis"][a]["null"]["p95"],
                           go["per_axis"][a]["null"]["p95"]))
    fn, fo = rn["full_field_exact"], ro["full_field_exact"]
    checks.append(("full.F_sep", fn["F_sep"], fo["F_sep"]))
    for a in ("kappa", "accel"):
        for lv in fn["axes"][a]["norm_by_level"]:
            checks.append(("full." + a + ".norm[" + lv + "]",
                           fn["axes"][a]["norm_by_level"][lv], fo["axes"][a]["norm_by_level"][lv]))
            checks.append(("full." + a + ".cos[" + lv + "]",
                           fn["axes"][a]["sign_cos_by_level"][lv],
                           fo["axes"][a]["sign_cos_by_level"][lv]))
    checks.append(("ctrl.C0", rn["controls"]["C0_identity_max_abs_diff"],
                   ro["controls"]["C0_identity_max_abs_diff"]))
    checks.append(("ctrl.zero_model", rn["controls"]["zero_model_max_abs_displacement"],
                   ro["controls"]["zero_model_max_abs_displacement"]))
    for i, s in enumerate(rn["own_first_action_sigma"]):
        checks.append(("sigma[%d]" % i, s, ro["own_first_action_sigma"][i]))
    mism = [(n, a, b) for n, a, b in checks if sig(a) != sig(b)]
    print("  %-22s %d statistics compared, %d mismatch | verdict %s (banked %s)"
          % (arm, len(checks), len(mism), rn["verdict"]["verdict"], ro["verdict"]["verdict"]))
    for n, a, b in mism[:12]:
        print("      MISMATCH %s: new %r banked %r" % (n, a, b))
    bad += len(mism)
print("  => B0 " + ("HOLDS - the flag is a true pass-through and the read is deterministic"
                    if bad == 0 else "FAILS - Task B is VOID"))

print("\n" + "=" * 100)
print("B1  VERDICT INVARIANCE - three spaces, both arms, both conventions")
flip = 0
for arm in ARMS:
    for u in ("kappa", "steer"):
        sa = J[u]["arms"][arm]["read"]["space_agreement"]
        print("  %-22s %-6s %s" % (arm, u, sa))
        flip += sum(1 for v in sa.values() if v != "LAT-INSENSITIVE-REFUTED")
print("  => B1 " + ("HOLDS" if flip == 0 else "FAILS") + " (%d non-REFUTED)" % flip)

print("\n" + "=" * 100)
print("B2  SEPARATION-RATIO - kappa axis F_sep/null_p95 (tac); band |log(c/r)| <= log(1.25)")
b2 = []
for arm in ARMS:
    r = ax("kappa", arm)["kappa"]["F_over_null_p95"]
    c = ax("steer", arm)["kappa"]["F_over_null_p95"]
    ok = abs(math.log(c / r)) <= math.log(1.25)
    b2.append(ok)
    print("  %-22s kappa: raw %12.3f  steer %12.3f  x%.3f  within-band=%s"
          % (arm, r, c, c / r, ok))
    ra = ax("kappa", arm)["accel"]["F_over_null_p95"]
    ca = ax("steer", arm)["accel"]["F_over_null_p95"]
    print("  %-22s accel: raw %12.3f  steer %12.3f  x%.4f  (untouched axis)"
          % ("", ra, ca, ca / ra))
print("  => B2 " + ("INVARIANT" if all(b2) else "MOVES"))

print("\n" + "=" * 100)
print("B3  LINEARITY vs THE SWEPT LEVELS - full-field ||m|| ratios to level 0.02")
pred = dict((lv, math.atan(L_ENC * lv) / math.atan(L_ENC * 0.02)) for lv in (0.02, 0.05, 0.1))
print("     arctan prediction (linear in the FED value): "
      + " ".join("%s:%.4f" % (lv, pred[lv]) for lv in (0.02, 0.05, 0.1)))
for arm in ARMS:
    for u in ("kappa", "steer"):
        nb = ff(u, arm)["axes"]["kappa"]["norm_by_level"]
        base = nb["0.02"]
        rho = ff(u, arm)["axes"]["kappa"]["spearman_rho_abslevel_vs_norm"]
        print("  %-22s %-6s norms %s | ratios %s | rho=%s"
              % (arm, u, " ".join("%s:%.6e" % (k, v) for k, v in nb.items()),
                 " ".join("%s:%.4f" % (k, v / base) for k, v in nb.items()), rho))
    nbs = ff("steer", arm)["axes"]["kappa"]["norm_by_level"]
    nbk = ff("kappa", arm)["axes"]["kappa"]["norm_by_level"]
    print("      -> top-level ratio  steer %.4f  vs arctan-linear %.4f (dev %+.4f)  vs raw %.4f"
          % (nbs["0.1"] / nbs["0.02"], pred[0.1],
             nbs["0.1"] / nbs["0.02"] - pred[0.1], nbk["0.1"] / nbk["0.02"]))
    print("      -> per-level gain steer/raw: "
          + " ".join("%s:x%.4f" % (k, nbs[k] / nbk[k]) for k in ("0.02", "0.05", "0.1")))

print("\n" + "=" * 100)
print("B4  ANTISYMMETRY - cos(m(+L), m(-L)) per level (bar <= -0.99 on kappa)")
for arm in ARMS:
    for u in ("kappa", "steer"):
        for a in ("kappa", "accel"):
            cs = ff(u, arm)["axes"][a]["sign_cos_by_level"]
            print("  %-22s %-6s %-6s %s"
                  % (arm, u, a, " ".join("%s:%+.4f" % (k, v) for k, v in cs.items())))

print("\n" + "=" * 100)
print("B5  ||m(kappa=0.1)|| / ||m(a=1.5)||, and whether 'matched sigma' survives")
for arm in ARMS:
    v = {}
    for u in ("kappa", "steer"):
        f = ff(u, arm)["axes"]
        v[u] = f["kappa"]["norm_by_level"]["0.1"] / f["accel"]["norm_by_level"]["1.5"]
        for sp in ("tac", "pooled"):
            g = J[u]["arms"][arm]["read"]["spaces"][sp]["grid"]["per_axis"]
            k = (g["kappa"]["mean_norms"]["kappa+0.1"] + g["kappa"]["mean_norms"]["kappa-0.1"]) / 2
            a = (g["accel"]["mean_norms"]["accel+1.5"] + g["accel"]["mean_norms"]["accel-1.5"]) / 2
            v[u + "." + sp] = k / a
    print("  %-22s full-field raw %.6f  steer %.6f  moves x%.3f"
          % (arm, v["kappa"], v["steer"], v["steer"] / v["kappa"]))
    print("  %-22s tac raw %.6f steer %.6f x%.3f | pooled raw %.6f steer %.6f x%.3f"
          % ("", v["kappa.tac"], v["steer.tac"], v["steer.tac"] / v["kappa.tac"],
             v["kappa.pooled"], v["steer.pooled"], v["steer.pooled"] / v["kappa.pooled"]))
    sg = J["kappa"]["arms"][arm]["read"]["own_first_action_sigma"]
    s_a, s_k = sg[0], sg[1]
    s_kap = math.tan(s_k) / L_ENC
    print("  %-22s sigma(ch0 accel)=%.6f m/s2 | sigma(ch1 AS FED = steer)=%.6f rad = %.7f rad/m"
          % ("", s_a, s_k, s_kap))
    print("  %-22s a=1.5 -> %.3f sigma_a | RAW level 0.1 fed %.6f rad = %.3f sigma_steer"
          % ("", 1.5 / s_a, 0.1, 0.1 / s_k))
    print("  %-22s CONVERTED level 0.1 fed %.6f rad = %.3f sigma_steer ; 0.1 rad/m = %.3f sigma_kappa"
          % ("", math.atan(L_ENC * 0.1), math.atan(L_ENC * 0.1) / s_k, 0.1 / s_kap))
    print("  %-22s the level that IS 2 sigma in curvature: %.7f rad/m -> fed %.6f rad = %.4f sigma_steer"
          % ("", 2 * s_kap, math.atan(L_ENC * 2 * s_kap), math.atan(L_ENC * 2 * s_kap) / s_k))

print("\n" + "=" * 100)
print("CONTROLS / material bar / n, both conventions")
for arm in ARMS:
    for u in ("kappa", "steer"):
        r = J[u]["arms"][arm]["read"]
        g = r["spaces"]["tac"]["grid"]
        A = J[u]["arms"][arm]
        print("  %-22s %-6s n=%d rel_units=%s rel_mag@2sig=%.5f max|d|=%.4f scene=%.6f "
              "C0=%s zero=%s speed_channel=%s scale=%s/%s md5=%s"
              % (arm, u, r["n_windows"], g["rel_units"], g["rel_mag_at_material_level"],
                 g["max_abs_displacement"], g["scene_spread"],
                 r["controls"]["C0_identity_max_abs_diff"],
                 r["controls"]["zero_model_max_abs_displacement"], A["speed_channel"],
                 A["speed_scale_mps"], A["speed_scale_effective"], A["ckpt_md5"]))
        for a in ("kappa", "accel"):
            nl = r["spaces"]["tac"]["grid"]["per_axis"][a]["null"]
            print("        null %-6s median %.4f p95 %.4f max %.4f (n_perm %d)"
                  % (a, nl["median"], nl["p95"], nl["max"], nl["n_perm"]))
