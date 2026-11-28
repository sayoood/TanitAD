"""Check the headline numbers of the v44 sections against the raw JSON/MD they cite.

Reads only. Run: python -I check_numbers_v44.py   (CPU, no GPU, no network). Exit 0 if every check holds.
Sources: E: originals (theory/*), the tip of agent/arch-inf-20260803 through the mirror (git show), and the C: pull of the identity gate.
"""
import hashlib
import json
import re
import subprocess
import sys

TH = "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/"
MIRROR = "C:/Users/Admin/tanitad-push"
TIP = sys.argv[1] if len(sys.argv) > 1 else "0423bb9de2d0461bb1decceedee61979908cf3f8"  # the full tip sha (v44 build); the first agent's default was the mirror's remote-tracking ref
RAW = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/"
import tempfile
ID_GATE = tempfile.gettempdir().replace("\\", "/") + "/v44_G_ID_C0CRE.json"  # extracted below from the tip (id_cell/G_ID_C0CRE.json landed with the C0C-RE read, 6b1127f4)

FAIL = []
N = [0]
ROWS = []
SRC = ["card prefix hashes"]


def tip(path):
    r = subprocess.run(["git", "-c", "safe.directory=*", "show", TIP + ":" + RAW + path], cwd=MIRROR, capture_output=True)
    assert r.returncode == 0, (path, r.stderr[:200])
    return r.stdout


def J(path):
    return json.loads(open(TH + path, "rb").read())


def chk(label, actual, expected, tol=5e-5):
    N[0] += 1
    ok = abs(actual - expected) <= tol if isinstance(expected, float) else actual == expected
    ROWS.append((label, actual, expected, ok, SRC[0]))
    if not ok:
        FAIL.append((label, actual, expected))
    print(("PASS " if ok else "FAIL ") + label, actual if not isinstance(actual, float) else round(actual, 6))


def card_sha(path, nbytes, prefix8):
    b = open(TH + path, "rb").read()[:nbytes]
    h = hashlib.sha256(b).hexdigest()
    chk("card " + path + " first " + str(nbytes) + " bytes", h[:8], prefix8)
    return h


# ---------------------------------------------------------------- frozen cards (prefix sha256)
card_sha("PREREG_V8F_WMACT.md", 5478, "d21b14cc")
card_sha("AMENDMENT_WMACT_1.md", 2249, "503ba1b8")
card_sha("PREREG_V8F_LATSEM.md", 4882, "0b0c2f73")
card_sha("AMENDMENT_LATSEM_1.md", 1375, "ddb07ff8")
card_sha("AMENDMENT_LATSEM_2.md", 1642, "83446cc6")
card_sha("PREREG_V8F_MRATE.md", 4213, "50667a79")
card_sha("PREREG_V8F_LATSEM2.md", 3769, "1325f21c")
b = tip("2026-11-07-v8f-town-result/PREREG_V8F_TOWN.md")[:4676]
chk("card TOWN first 4676 bytes", hashlib.sha256(b).hexdigest()[:8], "707e29f3")
b = tip("2026-11-07-t2-renc-gate/PREREG_T2_RENC.md")[:5544]
chk("card RENC first 5544 bytes", hashlib.sha256(b).hexdigest()[:8], "425d3496")

# ---------------------------------------------------------------- WMACT (T0, eval-141)
SRC[0] = 'theory/wmact/RESULT_WMACT.json [M, T0, episode-cluster bootstrap B 2,000 seed 0]'
W = J("wmact/RESULT_WMACT.json")
chk("WMACT windows used", W["windows_used"], 1976)
chk("WMACT episodes", W["episodes_used"], 141)
for s, v in (("s0", 0), ("s1", 1)):
    R = W["rows"]["M2b_" + s]
    exp = {
        "s0": {"L1/k1": (0.0010, 0.0054, 0.0029), "L2/k1": (0.0295, 0.0424, 0.0317), "L3/k1": (0.1316, 0.1850, 0.1041)},
        "s1": {"L1/k1": (-0.0010, 0.0049, 0.0030), "L2/k1": (0.0264, 0.0462, 0.0254), "L3/k1": (0.1351, 0.2000, 0.1074)},
    }[s]
    for lk, (sk, au, am) in exp.items():
        chk("WMACT %s %s SKILL" % (s, lk), R[lk]["SKILL"]["point"], sk)
        chk("WMACT %s %s ACTION-USE" % (s, lk), R[lk]["ACTION-USE"]["point"], au)
        chk("WMACT %s %s ACTION-USE-MEAN" % (s, lk), R[lk]["ACTION-USE-MEAN"]["point"], am)
        assert R[lk]["ACTION-USE"]["separated"] and R[lk]["ACTION-USE-MEAN"]["separated"]
    chk("WMACT %s L1/k1 SKILL separated" % s, R["L1/k1"]["SKILL"]["separated"], False)
    chk("WMACT %s L2/k3 SKILL" % s, R["L2/k3"]["SKILL"]["point"], (0.1084, 0.1185)[v])
    chk("WMACT %s L3/k3 SKILL" % s, R["L3/k3"]["SKILL"]["point"], (0.3283, 0.3175)[v])
    chk("WMACT %s L3/k3 ACTION-USE" % s, R["L3/k3"]["ACTION-USE"]["point"], (0.3423, 0.3238)[v])
    chk("WMACT %s L1/k5 SKILL" % s, R["L1/k5"]["SKILL"]["point"], (-0.0689, -0.0577)[v])
    chk("WMACT %s L1/k10 SKILL" % s, R["L1/k10"]["SKILL"]["point"], (-0.1500, -0.1724)[v])
    chk("WMACT %s L1/k10 ACTION-USE" % s, R["L1/k10"]["ACTION-USE"]["point"], (0.0897, 0.0900)[v])
    # zeroed level-1 action embedding (control 2), eval
    Z = W["controls"]["2_zeroed_act_embed"]["M2b_" + s]
    chk("WMACT %s zeroed-embedding L1 k1 SKILL" % s, Z["k1"]["SKILL"]["point"], (-0.0130, -0.0185)[v])
    chk("WMACT %s zeroed-embedding L1 k5 SKILL" % s, Z["k5"]["SKILL"]["point"], (-0.2128, -0.2139)[v])
    chk("WMACT %s zeroed-embedding L1 k10 SKILL" % s, Z["k10"]["SKILL"]["point"], (-0.4737, -0.4921)[v])
    print("   zeroed-embedding %s: k1 %.4f k5 %.4f k10 %.4f ; real: k1 %.4f k5 %.4f k10 %.4f" % (
        s, Z["k1"]["SKILL"]["point"], Z["k5"]["SKILL"]["point"], Z["k10"]["SKILL"]["point"],
        R["L1/k1"]["SKILL"]["point"], R["L1/k5"]["SKILL"]["point"], R["L1/k10"]["SKILL"]["point"]))
chk("WMACT control 3 s0 L1 k1 start-donor SKILL", W["rows"]["M2b_s0"]["L1/k1"]["CTRL3_SKILL_startdonor"]["point"], -39.2162, 5e-4)
chk("WMACT control 3 s1 L1 k1 start-donor SKILL", W["rows"]["M2b_s1"]["L1/k1"]["CTRL3_SKILL_startdonor"]["point"], -40.5909, 5e-4)
chk("WMACT control 4 reproduction s0", W["controls"]["4_reproduction"]["M2b_s0"]["max_abs_diff"], 0.0)
chk("WMACT control 4 reproduction s1", W["controls"]["4_reproduction"]["M2b_s1"]["max_abs_diff"], 0.0)
for s_ in ("s0", "s1"):
    for k_ in ("L1/k1", "L1/k5", "L1/k10", "L2/k1", "L2/k3", "L3/k1", "L3/k3"):
        assert W["rows"]["M2b_" + s_][k_]["CTRL3_SKILL_startdonor"]["separated"] and W["rows"]["M2b_" + s_][k_]["CTRL3_SKILL_startdonor"]["point"] < 0
print("   control 3 below 0 and separated at all 7 rows, both seeds: yes")

# ---------------------------------------------------------------- TRAIN read
SRC[0] = 'theory/wmact/RESULT_WMACT_TRAIN.json [M, T0]'
T = J("wmact/RESULT_WMACT_TRAIN.json")
chk("TRAIN windows", T["windows_used"], 1833) if "windows_used" in T else None
for s, v in (("s0", 0), ("s1", 1)):
    R = T["rows"]["M2b_" + s]
    chk("TRAIN %s L1/k1 SKILL" % s, R["L1/k1"]["SKILL"]["point"], (-0.0010, 0.0063)[v])
    chk("TRAIN %s L1/k1 ACTION-USE" % s, R["L1/k1"]["ACTION-USE"]["point"], (0.0159, 0.0155)[v])
    chk("TRAIN %s L2/k1 SKILL" % s, R["L2/k1"]["SKILL"]["point"], (0.0471, 0.0578)[v])
    chk("TRAIN %s L3/k1 SKILL" % s, R["L3/k1"]["SKILL"]["point"], (0.1801, 0.1897)[v])
    chk("TRAIN %s L3/k3 SKILL" % s, R["L3/k3"]["SKILL"]["point"], (0.4217, 0.4308)[v])
    chk("TRAIN %s L2/k3 SKILL" % s, R["L2/k3"]["SKILL"]["point"], (0.1818, 0.2020)[v])

# ---------------------------------------------------------------- SHRINK
SRC[0] = 'theory/wmact/RESULT_WMACT_SHRINK.json [M, T0]'
S = J("wmact/RESULT_WMACT_SHRINK.json")
chk("SHRINK controls_pass", S["controls_pass"], True)
for s, v in (("s0", 0), ("s1", 1)):
    for k, c, sk in (("k1", (0.496, 0.526), (0.0313, 0.0310)), ("k5", (0.501, 0.509), (0.0902, 0.0925)), ("k10", (0.461, 0.457), (0.1310, 0.1249))):
        r = S["rows"]["M2b_%s/L1/%s" % (s, k)]
        chk("SHRINK %s %s c_train" % (s, k), r["c_train"], c[v], 5e-4)
        chk("SHRINK %s %s SKILL_shrunk" % (s, k), r["SKILL_shrunk"]["point"], sk[v])
        chk("SHRINK %s %s SKILL_shrunk separated" % (s, k), r["SKILL_shrunk"]["separated"], True)
    r = S["rows"]["M2b_%s/L1/k1" % s]
    chk("SHRINK %s k1 ceiling cos2" % s, r["oracle_cos2_eval"], (0.0313, 0.0311)[v])
    chk("SHRINK %s k1 best c on eval" % s, r["oracle_c_eval"], (0.504, 0.496)[v], 5e-4)

# ---------------------------------------------------------------- LATSEM (T0, 139 episodes)
SRC[0] = 'theory/latsem/RESULT_LATSEM.json [M, T0]'
L = J("latsem/RESULT_LATSEM.json")
chk("LATSEM frames", L["n_bank_frames"], 9460)
chk("LATSEM episodes", L["n_episodes"], 139)
chk("LATSEM controls_pass", L["controls_pass"], True)
for s, v in (("s0", 0), ("s1", 1)):
    for lvl, vv in (("L1", (0.651, 0.656)), ("L2", (0.516, 0.563)), ("L3", (0.420, 0.520))):
        chk("LATSEM R1 %s %s speed R2" % (s, lvl), L["R1"]["M2b_%s/%s/v" % (s, lvl)]["R2"]["point"], vv[v], 5e-4)
    for tgt in ("a", "w", "dv2", "dpsi2"):
        sep = [L["R1"]["M2b_%s/%s/%s" % (s, lvl, tgt)]["R2"] for lvl in ("L1", "L2", "L3")]
        print("   R1 %s %s separated-above-0 flags:" % (s, tgt), [(round(x["point"], 3), x["separated"] and x["point"] > 0) for x in sep])
    chk("LATSEM DD %s" % s, L["R1"]["M2b_%s/DD" % s]["point"], (0.497, 0.270)[v], 5e-4)
a2 = L["A2"]
def dec(s, lvl, k):
    r = a2["M2b_%s/%s/%s/shrunk" % (s, lvl, k)] if "M2b_%s/%s/%s/shrunk" % (s, lvl, k) in a2 else a2["M2b_%s/%s/%s/raw" % (s, lvl, k)]
    return r
for s, v in (("s0", 0), ("s1", 1)):
    for lvl, k, ex in (("L3", "k3", (0.233, 0.223)), ("L3", "k1", (0.147, 0.145)), ("L2", "k3", (0.168, 0.157)), ("L2", "k1", (0.103, 0.090)), ("L1", "k1", (0.045, 0.004)), ("L1", "k5", (0.052, 0.011)), ("L1", "k10", (0.042, 0.012))):
        chk("LATSEM A2 %s %s %s deciding" % (s, lvl, k), dec(s, lvl, k)["point"], ex[v], 5e-4)
    chk("LATSEM A2 %s L3 k3 oracle" % s, a2["M2b_%s/L3/k3/oracle" % s]["point"], (0.038, 0.062)[v], 5e-4)
    chk("LATSEM A2 %s L3 k3 separated" % s, dec(s, "L3", "k3")["separated"], True)
    chk("LATSEM A2 %s L2 k3 separated" % s, dec(s, "L2", "k3")["separated"], True)
    for lvl, ex in (("L1", (2.723, 3.536)), ("L2", (1.349, 1.674)), ("L3", (3.634, 2.895))):
        chk("LATSEM R3 %s %s CF_LON" % (s, lvl), L["R3"]["M2b_%s/%s/CF_LON" % (s, lvl)]["point"], ex[v], 5e-4)
    for lvl, ex in (("L1", (-0.028, -0.019)), ("L2", (-0.004, 0.008)), ("L3", (0.012, 0.062))):
        chk("LATSEM R3 %s %s CF_LAT" % (s, lvl), L["R3"]["M2b_%s/%s/CF_LAT" % (s, lvl)]["point"], ex[v], 5e-4)
print("   kinematic refs:", [round(L["R3"]["M2b_s0/%s/kinematic_ref_LON" % l], 3) if not isinstance(L["R3"]["M2b_s0/%s/kinematic_ref_LON" % l], dict) else L["R3"]["M2b_s0/%s/kinematic_ref_LON" % l] for l in ("L1", "L2", "L3")])
print("   outcomes:", {k: L["outcomes"][k] for k in ("R3/L1/LON", "R3/L1/LAT", "R3/L2/LAT", "R3/L3/LAT", "A2/L1/k1") if k in L["outcomes"]})

# ---------------------------------------------------------------- TOWN (T1-kin own / T0 twin)
SRC[0] = 'tip raw/2026-11-07-v8f-town-result/RESULT_TOWN.json [M, T1-kin own / T0 twin, episode-cluster bootstrap B 2,000 seed 0]'
TW = json.loads(tip("2026-11-07-v8f-town-result/RESULT_TOWN.json"))
chk("TOWN outcome", TW["outcome"], "MIXED")
chk("TOWN windows", TW["n"]["windows"], 2399)
chk("TOWN replans", TW["n"]["replans"], 7197)
for sd, v in (("s0", 0), ("s1", 1)):
    X = TW["seeds"][sd]
    chk("TOWN %s slope" % sd, X["slope"]["point"], (-0.0009, -0.0078)[v])
    chk("TOWN %s slope lo" % sd, X["slope"]["lo"], (-0.0599, -0.0595)[v])
    chk("TOWN %s slope hi" % sd, X["slope"]["hi"], (0.0607, 0.0415)[v])
    chk("TOWN %s slope rows" % sd, X["slope"]["n_rows"], (2729, 2745)[v])
    chk("TOWN %s lon F1 delta" % sd, X["lon_f1"]["delta"], (-0.0831, -0.0702)[v])
    chk("TOWN %s lon F1 own" % sd, X["lon_f1"]["own"], (0.4031, 0.4023)[v])
    chk("TOWN %s lon F1 logged" % sd, X["lon_f1"]["logged"], (0.4862, 0.4725)[v])
    chk("TOWN %s lat F1 delta" % sd, X["lat_f1"]["delta"], (-0.0828, -0.0938)[v])
    chk("TOWN %s constraint RMSE delta" % sd, X["cons"]["delta"], (0.9628, 0.794)[v], 2e-3)
    chk("TOWN %s own slower share delta" % sd, X["beside"]["share_argmax_slower_than_gt"]["delta"], (0.0437, 0.0342)[v])
    chk("TOWN %s REG slope" % sd, X["regression_arm"]["slope"]["point"], (0.0041, -0.0081)[v])
    chk("TOWN %s REG slope not separated" % sd, X["regression_arm"]["slope"]["not_separated"], True)
    chk("TOWN %s C0 dv mean" % sd, X["deficit"]["dv_mean"], (0.0875, 0.059)[v], 2e-3)
print("   TOWN controls:", json.dumps(TW["controls"])[:400])

# ---------------------------------------------------------------- RENC (token level, T2 inputs)
SRC[0] = 'tip raw/2026-11-07-t2-renc-gate/{M_RENC,G_ID0,TRAIN_RESULT_A,TRAIN_RESULT_B}.json [M, token level, scene-cluster bootstrap B 2,000 seed 0]'
M = json.loads(tip("2026-11-07-t2-renc-gate/M_RENC.json"))
chk("RENC verdict", M["verdict"], "M-RENC-PASS")
chk("RENC M-RENC median", M["M_RENC"]["median"], 0.9427, 5e-5)
chk("RENC M-RENC before", M["M_RENC"]["median_before"], 0.8401, 5e-5)
chk("RENC rise", M["M_RENC"]["rise"], 0.1026, 5e-5)
chk("RENC ci lo", M["M_RENC"]["ci_median_scene_cluster_B2000_seed0"][0], 0.9390, 5e-5)
chk("RENC ci hi", M["M_RENC"]["ci_median_scene_cluster_B2000_seed0"][1], 0.9466, 5e-5)
chk("RENC p10", M["M_RENC"]["p10"], 0.9104, 5e-5)
chk("RENC p90", M["M_RENC"]["p90"], 0.9603, 5e-5)
chk("RENC scenes improved", M["M_RENC"]["scenes_improved"], 48)
chk("RENC pairs", M["M_RENC"]["n_pairs"], 960)
chk("RENC G-KEEP median", M["G_KEEP"]["median"], 0.9967, 5e-5)
chk("RENC G-KEEP ci lo", M["G_KEEP"]["ci_median_scene_cluster_B2000_seed0"][0], 0.9965, 5e-5)
chk("RENC G-KEEP ci hi", M["G_KEEP"]["ci_median_scene_cluster_B2000_seed0"][1], 0.9968, 5e-5)
chk("RENC G-KEEP p10", M["G_KEEP"]["p10"], 0.9954, 5e-5)
chk("RENC G-ID0 adapter", M["G_ID0"]["adapter_B0_bit_equal"], 100)
chk("RENC G-ID0 merged", M["G_ID0"]["merged_B0_bit_equal"], 100)
chk("RENC G-ID0 perturbed", M["G_ID0"]["perturbed_B_bit_equal"], 0)
chk("RENC same-fold flagged", M["regression_arms"]["M_RENC_same_fold"]["flagged_scenes"], 48)
chk("RENC same-fold naive", M["regression_arms"]["M_RENC_same_fold"]["naive_median_after"], 0.9697, 5e-5)
chk("RENC shifted pairing", M["regression_arms"]["G_KEEP_pairing_shifted_by_one"]["median"], 0.9254, 5e-5)
chk("RENC gate_valid", M["gate_valid"], True)
chk("RENC leaks", len(M["leaks"]), 0)
per = M["per_scene"]
chk("RENC min per-scene after", min(v["after"] for v in per.values()), 0.8924, 5e-5)
chk("RENC min per-scene keep", min(v["keep"] for v in per.values()), 0.9939, 5e-5)
chk("RENC min per-scene rise", min(v["after"] - v["before"] for v in per.values()), 0.0312, 5e-5)
chk("RENC scenes with adapted cosine below 0.90", sum(1 for v in per.values() if v["after"] < 0.90), 1)
chk("RENC scenes with keep below 0.99", sum(1 for v in per.values() if v["keep"] < 0.99), 0)
for f, secs, sps, babs in (("A", 4088.2, 1.3536, 0.1141), ("B", 4121.0, 1.3650, 0.1030)):
    t = json.loads(tip("2026-11-07-t2-renc-gate/TRAIN_RESULT_%s.json" % f))
    chk("RENC train %s seconds" % f, t["seconds"], secs, 0.05)
    chk("RENC train %s s/step" % f, t["s_per_step_mean"], sps, 5e-4)
    chk("RENC train %s max |B|" % f, t["b_abs_max"], babs, 5e-4)
    chk("RENC train %s trainable" % f, t["n_trainable"], 3145728)
    chk("RENC train %s pairs" % f, t["n_pairs"], 2400)
    chk("RENC train %s scenes" % f, t["n_train_scenes"], 24)
    chk("RENC train %s epochs x steps" % f, t["epochs"] * t["steps_per_epoch"], 3000)
    print("   %s loss epoch1 %.5f epoch10 %.5f" % (f, t["loss_per_epoch"][0]["loss"], t["loss_per_epoch"][-1]["loss"]))

open(ID_GATE, "wb").write(tip("2026-11-07-t2-c0cre-result/id_cell/G_ID_C0CRE.json"))  # v44 build: the identity file is on the tip
# ---------------------------------------------------------------- C0C-RE identity gate (file on the tip since 6b1127f4)
SRC[0] = '2026-11-07-t2-c0cre-result/id_cell/G_ID_C0CRE.json [M, tip]'
G = json.loads(open(ID_GATE, "rb").read())
chk("G-ID-C0CRE verdict", G["verdict"], "G-ID-C0CRE-PASS")
chk("G-ID-C0CRE scenes", G["n_scenes"], 2)
chk("G-ID-C0CRE cells", G["n_cells"], 8)
chk("G-ID-C0CRE records equal", G["n_records_equal"], 16)
chk("G-ID-C0CRE ticks stamped", G["n_ticks_stamped"], 48)
chk("G-ID-C0CRE problems", len(G["problems"]), 0)
print("G-ID-C0CRE file sha256", hashlib.sha256(open(ID_GATE, "rb").read()).hexdigest())


# ================================================================ v44 section 7.48 and the edits (added by the second agent)
def tipfile(path):
    r = subprocess.run(["git", "-c", "safe.directory=*", "show", TIP + ":" + path], cwd=MIRROR, capture_output=True)
    assert r.returncode == 0, (path, r.stderr[:200])
    return r.stdout


def commit_exists(c):
    return subprocess.run(["git", "-c", "safe.directory=*", "cat-file", "-e", c + "^{commit}"], cwd=MIRROR, capture_output=True).returncode == 0


SRC[0] = "commits on the tip (git cat-file through the mirror)"
for c in ("822a0175", "1c0a348c", "c85eed44", "e8789391", "448a67b5", "b9f1a043", "afb3920a", "df55198a"):
    chk("commit %s exists" % c, commit_exists(c), True)

SRC[0] = "card files on the tip, prefix sha256 [M]"
b = tipfile(RAW + "2026-11-07-v8f-tswap-prereg/PREREG_V8F_TSWAP.md")[:3642]
chk("card TSWAP first 3642 bytes", hashlib.sha256(b).hexdigest()[:8], "fc775ca1")
b = tipfile(RAW + "2026-11-07-t2-dagger-prereg/PREREG_T2_DAGGER.md")[:5415]
chk("card T2-DAGGER first 5415 bytes", hashlib.sha256(b).hexdigest()[:8], "9234478c")
b = tipfile(RAW + "2026-11-07-v8f-mrate-prereg/PREREG_V8F_MRATE.md")[:4213]
chk("card MRATE (tip) first 4213 bytes", hashlib.sha256(b).hexdigest()[:8], "50667a79")
b = tipfile(RAW + "2026-11-07-v8f-latsem2-prereg/PREREG_V8F_LATSEM2.md")[:3769]
chk("card LATSEM2 (tip) first 3769 bytes", hashlib.sha256(b).hexdigest()[:8], "1325f21c")
b = tipfile(RAW + "2026-11-07-v8f-town-result/IMPL_CHOICES.md")
chk("TOWN IMPL_CHOICES sha256 prefix", hashlib.sha256(b).hexdigest()[:8], "2c6fc876")

SRC[0] = "tip raw/2026-11-07-t2-renc-gate/M_RENC.json, TRAIN_RESULT_*.json [M]"
chk("RENC tool sha256 prefix", M["tool_sha256"][:8], "dfe13e8d")
fa = [k for k in per if k.startswith("fa035232")][0]
chk("RENC fa035232 fold", per[fa]["fold"], "B")
chk("RENC fa035232 before", per[fa]["before"], 0.7784, 5e-5)
chk("RENC fa035232 after", per[fa]["after"], 0.8924, 5e-5)
chk("RENC second-lowest scene after (all others at or above)", sorted(v["after"] for v in per.values())[1], 0.9036, 5e-5)
chk("RENC scenes at or above 0.90", sum(1 for v in per.values() if v["after"] >= 0.90), 47)
chk("RENC scenes per fold A", sum(1 for v in per.values() if v["fold"] == "A"), 24)
chk("RENC scenes per fold B", sum(1 for v in per.values() if v["fold"] == "B"), 24)
chk("RENC pairs per scene", M["M_RENC"]["n_pairs"] // M["M_RENC"]["n_scenes"], 20)
for f, a, z in (("A", 0.00830, 0.00364), ("B", 0.00888, 0.00390)):
    t = json.loads(tip("2026-11-07-t2-renc-gate/TRAIN_RESULT_%s.json" % f))
    chk("RENC train %s loss epoch 1" % f, t["loss_per_epoch"][0]["loss"], a, 5e-6)
    chk("RENC train %s loss epoch 10" % f, t["loss_per_epoch"][-1]["loss"], z, 5e-6)
    chk("RENC train %s fold label" % f, t["fold"], f)
chk("RENC G_ID0.json pass", json.loads(tip("2026-11-07-t2-renc-gate/G_ID0.json"))["pass"], True)

SRC[0] = "card text checks (E: t2_prereg/PREREG_T2_RENC.md with log) and RESULT_V8F_T1_HEADS.md on the tip [M]"
RENC_E = TH.replace("theory/", "t2_prereg/") + "PREREG_T2_RENC.md"
rc = open(RENC_E, "rb").read().decode("utf-8")
for needle in ("F = 0.12074", "RE-DOMINANT", "RE-MOVES", "RE-NULL or RE-HARMS", "RE-FAILS-MECH", "fa035232` reads 0.8924", "3,000 steps per fold",
               "2,990 steps", "drive running on box 2 since 12:44:32", "self-test 26 / 26, 9 / 9 mutations red", "Tests: 17 passed", "1.6 to 1.9 h"):
    chk("RENC card text has: " + needle, needle in rc, True)
chk("RENC card frozen prefix still 5544", hashlib.sha256(open(RENC_E, "rb").read()[:5544]).hexdigest()[:8], "425d3496")
h = tipfile("TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/RESULT_V8F_T1_HEADS.md").decode("utf-8")
chk("section 7.15 T-lon F1 0.461 to 0.336 (s0)", "0.461 \u2192 0.336" in h, True)
chk("section 7.15 T-lon F1 0.487 to 0.357 (s1)", "0.487 \u2192 0.357" in h, True)

SRC[0] = "C:/Users/Admin/t2v8f_pulls/stage5_C0C-RE_ID/G_ID_C0CRE.json [M, dev box, not on the tip]"
chk("G-ID-C0CRE scene ids", ",".join(G["scenes"]), "9e17ce05,e848c843")
chk("G-ID-C0CRE manifest5 prefix", G["manifest5_sha256"][:8], "3230269a")
chk("G-ID-C0CRE file sha256 prefix", hashlib.sha256(open(ID_GATE, "rb").read()).hexdigest()[:8], "16cefcec")

SRC[0] = "tip raw/2026-11-07-v8f-town-result/RESULT_TOWN.json [M]"
for sd, v in (("s0", 0), ("s1", 1)):
    X = TW["seeds"][sd]
    chk("TOWN %s slope episodes" % sd, X["slope"]["n_episodes"], (139, 138)[v])
    chk("TOWN %s lon F1 n" % sd, X["lon_f1"]["n"], 6175)
    chk("TOWN %s lat F1 n" % sd, X["lat_f1"]["n"], 6168)
    chk("TOWN %s lon F1 lo" % sd, X["lon_f1"]["lo"], (-0.1155, -0.1055)[v])
    chk("TOWN %s lon F1 hi" % sd, X["lon_f1"]["hi"], (-0.0396, -0.0287)[v])
    chk("TOWN %s lat F1 lo" % sd, X["lat_f1"]["lo"], (-0.1254, -0.1398)[v])
    chk("TOWN %s lat F1 hi" % sd, X["lat_f1"]["hi"], (-0.0462, -0.0577)[v])
    chk("TOWN %s lat F1 own" % sd, X["lat_f1"]["own"], (0.6993, 0.6914)[v])
    chk("TOWN %s lat F1 logged" % sd, X["lat_f1"]["logged"], (0.7820, 0.7852)[v])
    chk("TOWN %s cons lo" % sd, X["cons"]["lo"], (0.7613, 0.620)[v], 2e-3)
    chk("TOWN %s cons hi" % sd, X["cons"]["hi"], (1.1608, 0.993)[v], 2e-3)
    chk("TOWN %s cons own RMSE" % sd, X["cons"]["own_rmse"], (3.688, 3.505)[v], 2e-3)
    chk("TOWN %s cons logged RMSE" % sd, X["cons"]["logged_rmse"], (2.725, 2.711)[v], 2e-3)
    sl = X["beside"]["share_argmax_slower_than_gt"]
    chk("TOWN %s share slower own" % sd, sl["own"], (0.2321, 0.2410)[v], 2e-3)
    chk("TOWN %s share slower logged" % sd, sl["logged"], (0.1883, 0.2070)[v], 2e-3)
    chk("TOWN %s share slower lo" % sd, sl["lo"], (0.0244, 0.0178)[v])
    chk("TOWN %s share slower hi" % sd, sl["hi"], (0.0661, 0.0520)[v])
    t12 = X["beside"]["ticks_1_2_lon_f1"]
    chk("TOWN %s ticks1-2 lon F1 delta" % sd, t12["delta"], (-0.1295, -0.1214)[v])
    chk("TOWN %s ticks1-2 lon F1 lo" % sd, t12["lo"], (-0.1788, -0.1791)[v])
    chk("TOWN %s ticks1-2 lon F1 hi" % sd, t12["hi"], (-0.0657, -0.0505)[v])
    tg = X["beside"]["slope_three_group"]
    chk("TOWN %s three-group slope" % sd, tg["point"], (0.0455, 0.0374)[v])
    chk("TOWN %s three-group slope lo" % sd, tg["lo"], (0.0210, 0.0147)[v])
    chk("TOWN %s three-group slope hi" % sd, tg["hi"], (0.0716, 0.0616)[v])
    rg = X["regression_arm"]
    chk("TOWN %s REG slope lo" % sd, rg["slope"]["lo"], (-0.0017, -0.0189)[v])
    chk("TOWN %s REG slope hi" % sd, rg["slope"]["hi"], (0.0099, 0.0010)[v])
    chk("TOWN %s REG half-width" % sd, rg["slope"]["half_width"], (0.0058, 0.0100)[v], 5e-4)
    chk("TOWN %s C0 half-width" % sd, X["slope"]["half_width"], (0.0603, 0.0505)[v], 5e-4)
    chk("TOWN %s REG dv mean" % sd, rg["dv_mean"], (1.217, 0.879)[v], 2e-3)
    chk("TOWN %s REG lon F1 on logged" % sd, rg["lon_f1"]["logged"], (0.2599, 0.244)[v], 2e-3)
    chk("TOWN %s C0 slope-row dv signs slower" % sd, TW["dv_signs_C0"][sd]["n_own_slower_ge_0.5"] if sd in TW.get("dv_signs_C0", {}) else X["dv_signs_C0"]["n_own_slower_ge_0.5"], (1595, 1515)[v])
    chk("TOWN %s C0 slope-row dv signs faster" % sd, TW["dv_signs_C0"][sd]["n_own_faster_ge_0.5"] if sd in TW.get("dv_signs_C0", {}) else X["dv_signs_C0"]["n_own_faster_ge_0.5"], (1134, 1230)[v])
chk("TOWN ratio of half-widths s0 (about 10x)", round(TW["seeds"]["s0"]["slope"]["half_width"] / TW["seeds"]["s0"]["regression_arm"]["slope"]["half_width"]), 10)
chk("TOWN ratio of half-widths s1 (about 5x)", round(TW["seeds"]["s1"]["slope"]["half_width"] / TW["seeds"]["s1"]["regression_arm"]["slope"]["half_width"]), 5)
chk("TOWN ticks per window", TW["n"]["ticks_per_window"], 3)
chk("TOWN regression arm zeroed T token on 450 calls (in controls)", json.dumps(TW["controls"]).count("450") >= 2, True)

# ================================================================ v44 build agent: results that landed after the parts were written (each quoted number is found in its source file AND in the built report)
import os
OUTMD = sys.argv[2] if len(sys.argv) > 2 else "E:/Projects/TanitAD-artifacts/dagland/report_v44/out/V8F_SCIENTIFIC_REPORT.md"
REPORT = open(OUTMD, encoding="utf-8").read() if os.path.exists(OUTMD) else ""


def both(label, src, pat, rep, cls="M"):
    """pat is found in the tip source file `src`; rep (the report's spelling, with the unicode minus) is found in the built report"""
    SRC[0] = "%s [%s, tip]" % (src, cls)
    t = tip(src).decode("utf-8-sig").replace("\r", "")
    chk(label + " (source)", pat in t, True)
    chk(label + " (report)", rep in REPORT, True)
    if " [" in rep:  # all-occurrences rule: every place the report prints this interval prints it with this point (a number changed in one place only is caught)
        br = rep[rep.index(" ["):]
        chk(label + " (report, every occurrence of the interval carries this point)", REPORT.count(br) == REPORT.count(rep), True)


C = "2026-11-07-t2-c0cre-result/RESULT_T2_RENC.md"
both("C0C-RE outcome", C, "# RESULT_T2_RENC: RE-HARMS", "reads **RE-HARMS**")
both("C0C-RE scenes and starts", C, "Scenes read on: 48; starts: 93", "48 of 62 scenes and 93 of 121 starts")
for lab, pat, rep in [
    ("C0C-RE ADE6 s0", "| PRIMARY | ADE6 | -0.1956 | -0.4391 | +0.0895 | unseparated |", "−0.1956 [−0.4391, +0.0895]"),
    ("C0C-RE ADE6 s1", "| PRIMARY | ADE6 | -0.1002 | -0.3435 | +0.1502 | unseparated |", "−0.1002 [−0.3435, +0.1502]"),
    ("C0C-RE M1 s0", "| crawl | M1 crawl share | -0.0523 | -0.0852 | -0.0249 | separated-better |", "−0.0523 [−0.0852, −0.0249]"),
    ("C0C-RE M1 s1", "| crawl | M1 crawl share | -0.0142 | -0.0450 | +0.0112 | unseparated |", "−0.0142 [−0.0450, +0.0112]"),
    ("C0C-RE speed MAE s0", "| LONGITUDINAL | speed MAE | -0.1234 | -0.2157 | -0.0268 | separated-better |", "−0.1234 [−0.2157, −0.0268]"),
    ("C0C-RE speed MAE s1", "| LONGITUDINAL | speed MAE | -0.1282 | -0.2440 | -0.0142 | separated-better |", "−0.1282 [−0.2440, −0.0142]"),
    ("C0C-RE along-track s0", "| LONGITUDINAL | along-track MAE | -0.0965 | -0.3476 | +0.2074 | unseparated |", "−0.0965 [−0.3476, +0.2074]"),
    ("C0C-RE along-track s1", "| LONGITUDINAL | along-track MAE | -0.0204 | -0.2695 | +0.2473 | unseparated |", "−0.0204 [−0.2695, +0.2473]"),
    ("C0C-RE accel s0", "| LONGITUDINAL | accel MAE | -0.0156 | -0.1567 | +0.1346 | unseparated |", "−0.0156 [−0.1567, +0.1346]"),
    ("C0C-RE accel s1", "| LONGITUDINAL | accel MAE | -0.0788 | -0.1755 | +0.0051 | unseparated |", "−0.0788 [−0.1755, +0.0051]"),
    ("C0C-RE DK-HW s0", "| LONGITUDINAL | DK-HW | +0.5474 | +0.0413 | +1.2069 | separated-worse |", "+0.5474 [+0.0413, +1.2069]"),
    ("C0C-RE DK-HW s1", "| LONGITUDINAL | DK-HW | +0.6043 | -0.0051 | +1.3520 | unseparated |", "+0.6043 [−0.0051, +1.3520]"),
    ("C0C-RE heading MAE s1 (pair)", "| LATERAL | heading MAE | +2.8549 | -1.8623 | +8.6949 | unseparated |", "+2.8549 [−1.8623, +8.6949]"),
    ("C0C-RE family counts s0", "Family counts (separated-better / separated-worse / unseparated): PRIMARY 0/0/1; LONGITUDINAL 1/1/4; LATERAL 0/0/6; TACTICAL 0/0/2; crawl 1/0/0", "LONGITUDINAL 1/1/4 and 1/0/5"),
    ("C0C-RE family counts s1", "Family counts (separated-better / separated-worse / unseparated): PRIMARY 0/0/1; LONGITUDINAL 1/0/5; LATERAL 0/0/6; TACTICAL 0/0/2; crawl 0/0/1", "crawl 1/0/0 and 0/0/1"),
    ("C0C-RE seed gap ADE6 C0C-RE", "seed gap C0C-RE 0.05391687892963104", "0.0539 m for C0C-RE"),
    ("C0C-RE seed gap ADE6 C0C", "C0C 0.14936760308671115", "0.1494 m for C0C"),
    ("C0C-RE NOT-DOMINANT", "**NOT-DOMINANT**", "NOT-DOMINANT at both seeds"),
    ("C0C-RE heading MAE vs P10 (new row at s1)", "new in C0C-RE | none | heading MAE", "heading MAE"),
]:
    both(lab, C, pat, rep)
both("C0C-RE exclusions ATTRIBUTED, 14 excluded", "2026-11-07-t2-c0cre-result/exclusions5.out", '"verdict": "ATTRIBUTED", "n_excluded": 14, "problems": []', "ATTRIBUTED, 14 excluded")
both("C0C-RE next lever and DAgger card", C, "DAgger at T2", "DAgger at T2")
T = "2026-11-07-v8f-tswap-result/RESULT_TSWAP.md"
both("TSWAP outcome", T, "**Outcome by the card: T-NOT.**", "T-NOT")
for lab, pat, rep in [
    ("TSWAP M1 s0 TSWAP-OWN", "| M1 crawl share | +0.0031 [-0.0003, +0.0073] | n.s. | +0.0569 [+0.0380, +0.0790] | worse | +0.0533 [+0.0356, +0.0744] | worse |", "+0.0031 [−0.0003, +0.0073]"),
    ("TSWAP M1 s1 TSWAP-OWN", "| M1 crawl share | -0.0037 [-0.0086, +0.0007] | n.s. | +0.0559 [+0.0348, +0.0801] | worse | +0.0596 [+0.0368, +0.0860] | worse |", "−0.0037 [−0.0086, +0.0007]"),
    ("TSWAP M1 s0 TSWAP-P10", "+0.0569 [+0.0380, +0.0790]", "+0.0569 [+0.0380, +0.0790]"),
    ("TSWAP M1 s1 TSWAP-P10", "+0.0559 [+0.0348, +0.0801]", "+0.0559 [+0.0348, +0.0801]"),
    ("TSWAP M1 s0 OWN-P10", "+0.0533 [+0.0356, +0.0744]", "+0.0533 [+0.0356, +0.0744]"),
    ("TSWAP M1 s1 OWN-P10", "+0.0596 [+0.0368, +0.0860]", "+0.0596 [+0.0368, +0.0860]"),
    ("TSWAP T share s0", "T share: -0.059 [-0.150, +0.006] (numerator -0.0031, denominator +0.0533, 2,105 windows", "−0.059 [−0.150, +0.006]"),
    ("TSWAP T share s1", "T share: +0.061 [-0.014, +0.133] (numerator +0.0037, denominator +0.0596, 2,106 windows", "+0.061 [−0.014, +0.133]"),
    ("TSWAP ADE6 s0", "| PRIMARY | ADE6 | -0.0466 [-0.0742, -0.0203] | better |", "−0.0466 [−0.0742, −0.0203]"),
    ("TSWAP ADE6 s1", "| PRIMARY | ADE6 | -0.0563 [-0.0849, -0.0298] | better |", "−0.0563 [−0.0849, −0.0298]"),
    ("TSWAP speed MAE s0", "| LONGITUDINAL | speed MAE | -0.0511 [-0.0806, -0.0218] | better |", "−0.0511 [−0.0806, −0.0218]"),
    ("TSWAP speed MAE s1", "| LONGITUDINAL | speed MAE | -0.0699 [-0.0956, -0.0459] | better |", "−0.0699 [−0.0956, −0.0459]"),
    ("TSWAP L1 lateral-accel MAE s0 worse", "| LATERAL | L1 lateral-accel MAE | +0.1501 [+0.0084, +0.3084] | worse |", "+0.1501 [+0.0084, +0.3084]"),
    ("TSWAP cross-track s1 worse", "| LATERAL | cross-track MAE | +0.0200 [+0.0068, +0.0368] | worse |", "+0.0200 [+0.0068, +0.0368]"),
    ("TSWAP TAC-LAT s1 worse", "| TACTICAL | TAC-LAT | -0.0167 [-0.0267, -0.0067] | worse |", "−0.0167 [−0.0267, −0.0067]"),
    ("TSWAP controls", "* s0: G-ID PASS, G-TICK0 PASS, G-LIVE PASS, G-MUT PASS", "Controls 1 to 4 all PASS"),
]:
    both(lab, T, pat, rep)
L2B = "2026-11-07-v8f-latsem2b-result/RESULT_LATSEM2B.md"
for lab, pat, rep in [
    ("LATSEM2B L1 agents30 TRAIN s0", "+0.357 [+0.191, +0.478]†", "+0.357 [+0.191, +0.478]†"), ("LATSEM2B L1 agents30 TRAIN s1", "+0.329 [+0.122, +0.485]†", "+0.329 [+0.122, +0.485]†"),
    ("LATSEM2B L1 agents30 eval s0", "+0.308 [+0.036, +0.429]†", "+0.308 [+0.036, +0.429]†"), ("LATSEM2B L1 agents30 eval s1", "+0.231 [-0.162, +0.391]", "+0.231 [−0.162, +0.391]"),
    ("LATSEM2B L1 lead gap TRAIN s1", "+0.215 [+0.010, +0.397]†", "+0.215 [+0.010, +0.397]†"), ("LATSEM2B L2 agents30 s0", "+0.227 [-0.017, +0.400]", "+0.227 [−0.017, +0.400]"),
    ("LATSEM2B L2 agents30 s1", "+0.362 [+0.164, +0.494]†", "+0.362 [+0.164, +0.494]†"), ("LATSEM2B L3 agents30 s0", "+0.156 [-0.015, +0.274]", "+0.156 [−0.015, +0.274]"),
    ("LATSEM2B L3 agents30 s1", "+0.246 [+0.066, +0.373]†", "+0.246 [+0.066, +0.373]†"),
    ("LATSEM2B C3 S1 classification", "| S1_classification | TRAIN | 360 | 10 | 2.78 % | 7.50 % (27) |", "2.78 % / 2.50 %"),
    ("LATSEM2B C3 S2 interval alone", "| S2 | TRAIN | 2,520 | 90 | 3.57 % | 57.98 % (1461) |", "57.98 % / 57.74 %"),
    ("LATSEM2B C3 S3-PARTIAL", "| S3_PARTIAL | TRAIN | 120 | 1 | 0.83 % |", "0.83 % / 0.83 %"),
    ("LATSEM2B deciding set", "9,857 bank frames, 141 episodes", "9,857 frames, 141 episodes, 1,833 windows, 461 S3 windows"),
    ("LATSEM2B S3-PARTIAL min", "| TRAIN | L1 | s0 | +0.72 [+0.12, +1.37] m |", "+0.72 to +3.30 m"), ("LATSEM2B S3-PARTIAL max", "| TRAIN | L3 | s0 | +3.30 [+2.33, +4.26] m |", "+0.72 to +3.30 m"),
    ("LATSEM2B raw S3 min", "+1.85 [+1.25, +2.47] m", "+1.85 to +6.66 m"), ("LATSEM2B raw S3 max", "+6.66 [+5.71, +7.57] m", "+1.85 to +6.66 m"),
    ("LATSEM2B next lever", "L1: the reliance arm on that level.", "the reliance arm on level 1"),
]:
    both(lab, L2B, pat, rep)
L2 = "2026-11-07-v8f-latsem2-run1/RESULT_LATSEM2.md"
for lab, pat, rep in [
    ("LATSEM2 run 1 INVALID", "**READ INVALID: control(s) 3, 4 failed.**", "Run 1 of LATSEM2 is INVALID"),
    ("LATSEM2 C3 draws", "classification (lead_present, lon7, turn): 18 of 360; continuous (the other 6): 0 of 720", "18 of 360 draws (5.0 %)"), ("LATSEM2 C4 draws", "25 are separated below 0 and 50 above 0", "25 of 120 draws fell below 0 and 50 above"),
    ("LATSEM2 gap-speed correlation", "correlate at r = +0.52", "r = +0.52"),
]:
    both(lab, L2, pat, rep)
both("imagination plan G7", "2026-11-07-v8f-imagination-plan/GAPS.md", "+0.72 to +3.30 m", "S3-PARTIAL +0.72 to +3.30 m")
both("imagination plan G3", "2026-11-07-v8f-imagination-plan/GAPS.md", "+0.012 / +0.062 of 0.5 rad/s", "+0.012 / +0.062 of 0.5 rad/s")
both("imagination plan E6 running", "2026-11-07-v8f-imagination-plan/PLAN_IMAGINATION.md", "full-suite pass of the MRATE trainer change, which is running", "waits for a full-suite pass")
for nm, f, nb, p8 in [("IMAG1", "2026-11-07-v8f-imagination-plan/PREREG_V8F_IMAG1.md", 9212, "14846eef"), ("VOE1", "2026-11-07-v8f-imagination-plan/PREREG_V8F_VOE1.md", 5822, "555c5a38"), ("TSWAP", "2026-11-07-v8f-tswap-result/PREREG_V8F_TSWAP.md", 3642, "fc775ca1"),
                      ("LATSEM2B", "2026-11-07-v8f-latsem2b-result/PREREG_V8F_LATSEM2B.md", 4611, "bf6b3717"), ("LATSEM2B Amendment 1", "2026-11-07-v8f-latsem2b-result/AMENDMENT_LATSEM2B_1.md", 2778, "120072ff"),
                      ("LATSEM2 (tip copy)", "2026-11-07-v8f-latsem2-run1/PREREG_V8F_LATSEM2.md", 3769, "1325f21c"), ("MRATE (tip copy)", "2026-11-07-v8f-mrate-prereg/PREREG_V8F_MRATE.md", 4213, "50667a79")]:
    SRC[0] = f + " [M, tip]"
    chk("card %s first %d bytes" % (nm, nb), hashlib.sha256(tip(f)[:nb]).hexdigest()[:8], p8)
SRC[0] = "git history of the tip [M]"
for c in ("6b1127f4", "fafbb4aa", "d2771e01", "59bc4e26", "912e82e1", "0423bb9d"):
    chk("landing commit %s is an ancestor of the tip" % c, subprocess.run(["git", "-c", "safe.directory=*", "merge-base", "--is-ancestor", c, TIP], cwd=MIRROR).returncode == 0, True)
SRC[0] = "TanitAD Research Lab/Library/LIBRARY.md header [M, tip and its parent]"
chk("library index after (793 entries, 5755.9 MB)", "**793 entries · 5755.9 MB" in subprocess.run(["git", "-c", "safe.directory=*", "show", TIP + ":TanitAD Research Lab/Library/LIBRARY.md"], cwd=MIRROR, capture_output=True).stdout.decode("utf-8")[:2000], True)
chk("library index before (743 entries, 5320.5 MB)", "**743 entries · 5320.5 MB" in subprocess.run(["git", "-c", "safe.directory=*", "show", "0423bb9d^:TanitAD Research Lab/Library/LIBRARY.md"], cwd=MIRROR, capture_output=True).stdout.decode("utf-8")[:2000], True)

# ---------------------------------------------------------------- NUMBERS_V44.md is generated from the rows above
with open("NUMBERS_V44.md", "w", encoding="utf-8", newline="\n") as fh:
    fh.write("# NUMBERS_V44: headline numbers of sections 7.47 and 7.48 and of the edits, each re-read from its source by `check_numbers_v44.py`\n\n")
    fh.write("Generated by `python -I check_numbers_v44.py` (rewritten on every run). Each row: the check, the value read from the source, the value the text quotes, PASS or FAIL, and the source. Evidence class is in brackets in the source column: [M] MEASURED. The rows cover the numbers that decide a claim; the derived [E], inherited [I] and hypothesis [H] numbers are in the second table.\n\n")
    fh.write("| # | check | read from source | quoted in the text | result | source and class |\n|---|---|---|---|---|---|\n")
    for i, (lab, act, exp, ok, src) in enumerate(ROWS, 1):
        a = ("%.6g" % act) if isinstance(act, float) else str(act)
        e = ("%.6g" % exp) if isinstance(exp, float) else str(exp)
        fh.write("| %d | %s | %s | %s | %s | %s |\n" % (i, lab.replace("|", "/"), a.replace("|", "/"), e.replace("|", "/"), "PASS" if ok else "FAIL", src.replace("|", "/")))


DERIVED = [
    ("20 pairs per scene", "960 pairs / 48 scenes (checked above)", "M_RENC.json", "E"),
    ("47 of 48 scenes at or above 0.90", "count of per_scene after >= 0.90 (checked above)", "M_RENC.json per_scene", "E"),
    ("10x / 5x narrower REG interval", "C0 slope half-width / REG slope half-width, rounded (checked above)", "RESULT_TOWN.json", "E"),
    ("0.0603, 0.0058, 0.0505, 0.0100", "slope half-widths (checked above)", "RESULT_TOWN.json", "M"),
    ("about +-0.06 rank units per m/s", "the C0 slope half-widths 0.0603 / 0.0505 rounded", "RESULT_TOWN.md text", "E"),
    ("1.6 to 1.9 h for the C0C-RE drive", "estimate in the card log of 07:57", "PREREG_T2_RENC.md log", "E"),
    ("17 tests passed; self-test 26 of 26; 9 of 9 mutations red", "quoted from the card log, not re-run (text presence checked above)", "PREREG_T2_RENC.md log", "I"),
    ("C0C - P10 M1 crawl share +0.4073 / +0.4041", "quoted from v43 section 7.45.2", "v43 report", "I"),
    ("F = 0.12074 m", "the T1-kin keep tolerance in the RENC card section 5 (text presence checked above)", "PREREG_T2_RENC.md", "M"),
    ("0.461 to 0.336 / 0.487 to 0.357 (section 7.15)", "text presence checked above; a different arm, not compared", "RESULT_V8F_T1_HEADS.md", "M"),
    ("3 replans per window, 7,197 replans", "2,399 x 3 (checked above)", "RESULT_TOWN.json", "M"),
    ("ABS1 and S81 numbers for 7.47 (SKILL, ACTION-USE, R2, c, speed decode, 1,976 and 1,833 windows)", "each equals a row checked in the WMACT, TRAIN, SHRINK and LATSEM blocks above; the bracketed intervals are copied from section 7.47", "check blocks above", "M"),
    ("MRATE W0/W1 design, K = 3, 0.4 / 1.2 / 2.8 s of history", "card text, no result", "PREREG_V8F_MRATE.md", "H"),
    ("TSWAP outcomes T-CARRIES / T-PARTIAL / T-NOT", "card text; the result (T-NOT) and its rows are checked above (v44 build)", "PREREG_V8F_TSWAP.md, RESULT_TSWAP.md", "M"),
    ("C0C-RE outcomes and the 13:04 readings", "card text; the result (RE-HARMS) and its rows are checked above (v44 build)", "PREREG_T2_RENC.md section 6 and log, RESULT_T2_RENC.md", "M"),
    ("C0C-RE: the label rests on one s1 row (heading MAE)", "reading of RESULT_T2_RENC.md: that row is separated-worse against P10 for C0C-RE and not for C0C, and its pair row is not separated (both rows checked above)", "RESULT_T2_RENC.md", "E"),
    ("T share bounds, 0.0539 and 0.1494 seed gaps", "the printed seed gaps are 0.05391687892963104 and 0.14936760308671115 rounded to 4 decimals (checked above)", "RESULT_T2_RENC.md", "E"),
    ("LATSEM2B eval-141 rows, post hoc", "the card says so; the rows are quoted only as the same-sign condition of SCENE-AWARE", "PREREG_V8F_LATSEM2B.md", "M"),
    ("Imagination plan: gaps G1 to G9, questions Q1 to Q6, modules IA-0 to IA-6, experiments E1 to E9", "plan text of the three files; no result; only the numbers G3 and G7 and the E6 status are checked above", "PLAN_IMAGINATION.md, GAPS.md", "H"),
    ("IMAG1 launched 15:20:39; IMPL_CHOICES_IMAG1 2ecd212f; VOE1 INVALID by its controls 1 and 2 and VOE1B written", "dev-box files not on the tip (HANDOFF_IMAG1.md, HANDOFF_VOE1.md); read at the build, quoted in 7.47.9 as [I, UNVERIFIED by the build]", "E:/.../theory/imag1, theory/voe1", "I"),
]
with open("NUMBERS_V44.md", "a", encoding="utf-8", newline="\n") as fh:
    fh.write("\n## Derived, inherited and hypothesis numbers\n\n| number | derivation or note | source | class |\n|---|---|---|---|\n")
    for a, b2, c2, d2 in DERIVED:
        fh.write("| %s | %s | %s | [%s] |\n" % (a, b2, c2, d2))

print("\nchecks run: %d, failed: %d" % (N[0], len(FAIL)))
for f in FAIL:
    print("FAILED", f)
sys.exit(1 if FAIL else 0)
