"""test_rk_gate_record.py -- REVISION 2 (Amendment 1). Dev-box test of rk_gate_record.py on planted files (stdlib only, no pod).
All revision-1 cases stay (K1-K22, planted with the cache). New: K23 the cache file absent, K24 the cache sha mismatch (built levers read another sha), K25 the OTHER seed's cache named in the argv,
K26 the CPU precompute's selfcheck missing, K27 its census differs from the bound selfcheck's, K28 its levers carry another sha, K29 the argv has no cache flag, K30 the built levers carry no cache sha,
K31 the built levers carry another cache path, K32 the cache file changed after the mint (verify must refuse), K33 the cache file removed after the mint (verify must refuse), K34 the G-SUITE at 0ca6f49d (the old commit), K35 the record edited to drop its cache binding (verify must refuse).

A well-formed RK_s0 record must MINT and VERIFY (and so must the boundary share 0.50); each mutation must REFUSE, and for the stated REASON (a mutation that refuses for another reason is a test bug):
  K1 argv without --dagger-label           K2 --dagger-label speed_v1          K3 built levers say speed_v1          K4 n_label_kin 0            K5 n_label_kin key missing
  K6 census share 0.49 (49 of 100)         K7 kin_status_counts missing        K8 G-SUITE at df250a00                K9 G-SUITE reads a regression
  K10 tag RA_s0                            K11 the RK tree marker missing (the RA marker only)                       K12 bank sha256 in the argv is not bank.json's
  K13 the commit placeholder left unfilled K14 --dagger-mode add in the argv   K15 grad_diff_all 0                    K16 control_p0_grad_diff != 0
  K17 rl_label_rows_changed 0              K18 --dagger-bank is bank_k          K19 the pair check did not pass       K20 a bound file changes after the mint (verify must refuse)
  K21 the record's census share edited after the mint (verify must refuse)       K22 the bound selfcheck's counts edited so the share drops under 0.5 (verify must refuse)
Usage: python -I test_rk_gate_record.py <rk_gate_record.py> <scratch dir>"""
import hashlib
import json
import os
import shutil
import subprocess
import sys

GR0, ROOT = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
GR = GR0
COMMIT = "1a8b11a1abd50c09636da8b7425e97573cdf1d04"
DF25 = "df250a0032e56aae89304421918ade5c583d53a0"
OLD0CA = "0ca6f49dd076c035c1cd42da5e4e89a6f6961b9a"
os.makedirs(ROOT, exist_ok=True)
src = open(GR0, encoding="utf-8").read()
assert f'COMMIT = "{COMMIT}"' in src, "the gate record under test does not carry the KINLAB commit"


def w(p, o):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(o if isinstance(o, str) else json.dumps(o))


def counts_for(kin, other):
    return {"kin_emitted": kin, "refused_not_merged": other // 2, "refused_path_exhausted": other - other // 2}


def plant(R, tag="RK_s0", mut=None, kin=63, other=37):
    if os.path.exists(R):
        shutil.rmtree(R)
    seed = tag.split("_s")[1]
    w(f"{R}/bank/s{seed}/bank.json", {"bank": "planted"})
    bsha = hashlib.sha256(open(f"{R}/bank/s{seed}/bank.json", "rb").read()).hexdigest()
    w(f"{R}/bank/s{seed}/BANK_CHECK.json", {"pass": True, "readout": "causal_g1x"})
    bank = f"{R}/bank_k/s{seed}" if mut == "K18" else f"{R}/bank/s{seed}"
    argv = ["--dagger-p", "0.1", "--dagger-bank", bank, "--dagger-bank-sha256", bsha if mut != "K12" else "0" * 64,
            "--dagger-bank-readout", "causal_g1x", "--dagger-bank-kappa", "0.2", "--out", f"{R}/runs/{tag}/train"]
    if mut != "K1":
        argv += ["--dagger-label", "speed_v1" if mut == "K2" else "kin_v1"]
    if mut == "K14":
        argv += ["--dagger-mode", "add"]
    oseed = "1" if seed == "0" else "0"
    if mut != "K29":
        argv += ["--dagger-label-cache", f"{R}/kin_cache/s{oseed if mut == 'K25' else seed}_kin_v1.npz"]
    w(f"{R}/runs/{tag}/argv.json", {"argv": argv})
    if mut == "K6":
        kin, other = 49, 51
    probe = {"pass": True, "n_selected": 12, "grad_diff_all": 0.0 if mut == "K15" else 3.5, "control_p0_grad_diff": 0.25 if mut == "K16" else 0.0,
             "rl_label_rows_changed": 0 if mut == "K17" else 20, "kin_status_counts": counts_for(kin, other)}
    if mut != "K5":
        probe["n_label_kin"] = 0 if mut == "K4" else 9
    if mut == "K7":
        del probe["kin_status_counts"]
    cpath = f"{R}/kin_cache/s{seed}_kin_v1.npz"
    if mut != "K23":
        w(f"{R}/kin_cache/s{seed}_kin_v1.npz", "planted cache bytes seed " + seed)
    csha = hashlib.sha256(("planted cache bytes seed " + seed).encode()).hexdigest()
    levers = {"dagger": {"dagger_bank_readout": "causal_g1x", "dagger_label": "speed_v1" if mut == "K3" else "kin_v1", "dagger_label_cache": "/elsewhere/c.npz" if mut == "K31" else cpath,
                         "dagger_label_cache_sha256": ("0" * 64 if mut == "K24" else None if mut == "K30" else csha)}}
    w(f"{R}/runs/{tag}/selfcheck/selfcheck.json", {"declared_vs_built": "PASS", "acc_probe": {"pass": True}, "dagger_probe": probe, "levers": levers})
    cprobe = {"kin_status_counts": counts_for(kin + 1, other - 1) if mut == "K27" else counts_for(kin, other)}
    clev = {"dagger": {"dagger_label": "kin_v1", "dagger_label_cache": cpath, "dagger_label_cache_sha256": "1" * 64 if mut == "K28" else csha}}
    if mut != "K26":
        w(f"{R}/kin_cache/s{seed}_selfcheck_cpu.json", {"levers": clev, "dagger_probe": cprobe})
    w(f"{R}/gsuite_result_rk.json", {"new_commit": DF25 if mut == "K8" else OLD0CA if mut == "K34" else COMMIT, "n_regressions": 1 if mut == "K9" else 0, "new_test_failures": [], "missing_in_new": []})
    w(f"{R}/pilot/PILOT_RH.json", {"pass": True, "readout": "causal_g1x"})
    w(f"{R}/done_ra/tree.done", "x")
    if mut != "K11":
        w(f"{R}/done_rk/tree.done", "x")
    w(f"{R}/runs/{tag}/pair_{tag}.log", "no pass here\n" if mut == "K19" else "ZZ ROLLHIST-PAIR PASS\n")


def run(R, *a, gr=None):
    env = dict(os.environ, RH_ROOT=R)
    p = subprocess.run([sys.executable, "-I", gr or GR, *a], env=env, capture_output=True, text=True, encoding="utf-8")
    return p.returncode, p.stdout + p.stderr


ok = True
R = os.path.join(ROOT, "R")


def report(name, good, extra=""):
    global ok
    ok &= bool(good)
    print(name, "PASS" if good else "FAIL", extra)


# well-formed, and the boundary share 0.50
plant(R)
rc, out = run(R, "mint", "--tag", "RK_s0", "--readout", "causal_g1x")
rc2, out2 = run(R, "verify", "--tag", "RK_s0")
rec = json.load(open(f"{R}/runs/RK_s0/gate/RH_GATE_RECORD.json")) if rc == 0 else {}
report("WELL-FORMED mint+verify", rc == 0 and "MINTED" in out and rc2 == 0 and "VERIFY OK" in out2 and abs(rec["census"]["relabelled_share_of_ok"] - 0.63) < 1e-12 and rec["census"]["n_ok"] == 100,
       (out.strip().splitlines()[-2:], out2.strip().splitlines()[-1:]))
plant(R, kin=50, other=50)
rc, out = run(R, "mint", "--tag", "RK_s0", "--readout", "causal_g1x")
rc2, out2 = run(R, "verify", "--tag", "RK_s0")
report("BOUNDARY share 0.50 mints (bar is >= 0.5)", rc == 0 and rc2 == 0)
plant(R, tag="RK_s1")
rc, out = run(R, "mint", "--tag", "RK_s1", "--readout", "causal_g1x")
rc2, out2 = run(R, "verify", "--tag", "RK_s1")
report("WELL-FORMED seed 1 mint+verify", rc == 0 and rc2 == 0)

# the mutations: (name, tag, expected reason substring)
MUT = [("K1", "RK_s0", "--dagger-label is None"), ("K2", "RK_s0", "--dagger-label is 'speed_v1', not kin_v1"), ("K3", "RK_s0", "built levers carry dagger_label ['speed_v1']"),
       ("K4", "RK_s0", "n_label_kin [0]"), ("K5", "RK_s0", "n_label_kin []"), ("K6", "RK_s0", "CENSUS BAR"), ("K7", "RK_s0", "no usable kin_status_counts"),
       ("K8", "RK_s0", "G-SUITE new commit is 'df250a0032e56aae89304421918ade5c583d53a0'"), ("K9", "RK_s0", "G-SUITE reads regressions"),
       ("K10", "RA_s0", "serves RK_s0 / RK_s1 only"), ("K11", "RK_s0", "missing tree_done"), ("K12", "RK_s0", "not the sha256 of bank.json"),
       ("K14", "RK_s0", "names --dagger-mode"), ("K15", "RK_s0", "grad_diff_all [0.0]"), ("K16", "RK_s0", "control_p0_grad_diff reads [0.25]"),
       ("K17", "RK_s0", "rl_label_rows_changed [0]"), ("K18", "RK_s0", "not the RH bank"), ("K19", "RK_s0", "did not print ZZ ROLLHIST-PAIR PASS"),
       ("K23", "RK_s0", "missing label_cache"), ("K24", "RK_s0", "the built levers carry dagger_label_cache_sha256"), ("K25", "RK_s0", "this seed's bank's cache"),
       ("K26", "RK_s0", "missing cache_selfcheck"), ("K27", "RK_s0", "the CPU precompute's kin_status_counts"), ("K28", "RK_s0", "it is not the run that wrote this cache"),
       ("K29", "RK_s0", "--dagger-label-cache is None"), ("K30", "RK_s0", "the built levers carry dagger_label_cache_sha256 ['None']"), ("K31", "RK_s0", "the built levers carry dagger_label_cache ['/elsewhere/c.npz']"),
       ("K34", "RK_s0", "G-SUITE new commit is '0ca6f49dd076c035c1cd42da5e4e89a6f6961b9a'")]
for m, tag, why in MUT:
    plant(R, tag=("RK_s0" if tag == "RA_s0" else tag), mut=m)
    if tag == "RA_s0":                                               # K10: the files of an RK arm, the RA tag
        shutil.copytree(f"{R}/runs/RK_s0", f"{R}/runs/RA_s0")
    rc, out = run(R, "mint", "--tag", tag, "--readout", "causal_g1x")
    red = rc != 0 and "REFUSED" in out and why in out and not os.path.exists(f"{R}/runs/{tag}/gate/RH_GATE_RECORD.json")
    report(m, red, [l for l in out.splitlines() if l.startswith("PROBLEM")][:2] if not red else f"RED ({why})")

# K13: the commit placeholder unfilled (a copy of the gate record whose COMMIT is the placeholder)
unf = os.path.join(ROOT, "rk_gate_record_unfilled.py")
open(unf, "w", encoding="utf-8", newline="\n").write(src.replace(f'COMMIT = "{COMMIT}"', 'COMMIT = "__RK_COMMIT__"'))
plant(R)
rc, out = run(R, "mint", "--tag", "RK_s0", "--readout", "causal_g1x", gr=unf)
report("K13", rc != 0 and "REFUSED" in out and "filled 40-hex commit" in out and not os.path.exists(f"{R}/runs/RK_s0/gate/RH_GATE_RECORD.json"), "RED (placeholder refuses)")
run(R, "mint", "--tag", "RK_s0", "--readout", "causal_g1x")                                    # a real record, then verified by the UNFILLED copy: the commit differs
rc, out = run(R, "verify", "--tag", "RK_s0", gr=unf)
report("K13b verify under the placeholder", rc != 0 and "REFUSED" in out and "not a pass for this commit" in out)

# K20: a bound file changes after the mint
for what, path, newtxt in (("gsuite", f"{R}/gsuite_result_rk.json", json.dumps({"new_commit": COMMIT, "n_regressions": 0, "new_test_failures": [], "missing_in_new": [], "touched": 1})),
                           ("argv", f"{R}/runs/RK_s0/argv.json", None), ("bank.json", f"{R}/bank/s0/bank.json", json.dumps({"bank": "edited"})),
                           ("tree marker", f"{R}/done_rk/tree.done", "y")):
    plant(R)
    run(R, "mint", "--tag", "RK_s0", "--readout", "causal_g1x")
    if newtxt is None:
        a = json.load(open(path))
        a["argv"] += ["--w-extra", "1"]
        newtxt = json.dumps(a)
    w(path, newtxt)
    rc, out = run(R, "verify", "--tag", "RK_s0")
    report(f"K20 {what} changed after the mint", rc != 0 and "REFUSED" in out and "changed or missing since the mint" in out)
os.remove(f"{R}/done_rk/tree.done")
rc, out = run(R, "verify", "--tag", "RK_s0")
report("K20b tree marker removed after the mint", rc != 0 and "REFUSED" in out)

# K32 / K33: the label cache file changes / disappears after the mint (verify must refuse); K32b the CPU precompute's selfcheck edited
plant(R)
run(R, "mint", "--tag", "RK_s0", "--readout", "causal_g1x")
w(f"{R}/kin_cache/s0_kin_v1.npz", "a different cache")
rc, out = run(R, "verify", "--tag", "RK_s0")
report("K32 cache file changed after the mint", rc != 0 and "REFUSED" in out and "label_cache" in out)
plant(R)
run(R, "mint", "--tag", "RK_s0", "--readout", "causal_g1x")
os.remove(f"{R}/kin_cache/s0_kin_v1.npz")
rc, out = run(R, "verify", "--tag", "RK_s0")
report("K33 cache file removed after the mint", rc != 0 and "REFUSED" in out and "label_cache" in out)
plant(R)
run(R, "mint", "--tag", "RK_s0", "--readout", "causal_g1x")
w(f"{R}/kin_cache/s0_selfcheck_cpu.json", {"edited": 1})
rc, out = run(R, "verify", "--tag", "RK_s0")
report("K32b CPU precompute selfcheck changed after the mint", rc != 0 and "REFUSED" in out and "cache_selfcheck" in out)
# K35: the record is edited to drop its cache binding (every remaining bound file unchanged): verify must refuse
plant(R)
run(R, "mint", "--tag", "RK_s0", "--readout", "causal_g1x")
rp2 = f"{R}/runs/RK_s0/gate/RH_GATE_RECORD.json"
rec2 = json.load(open(rp2))
del rec2["binds"]["label_cache"]
json.dump(rec2, open(rp2, "w"))
rc, out = run(R, "verify", "--tag", "RK_s0")
report("K35 record without its cache binding", rc != 0 and "REFUSED" in out and "does not bind this seed's cache" in out)
# K36-K41: the pre-lock precheck of the cache pair (the chain's step `caches` and the DRY walk): well-formed OK; each absence / change REFUSES for its reason; it writes nothing
plant(R)
before = sorted(os.listdir(f"{R}/runs/RK_s0"))
rc, out = run(R, "precheck", "--tag", "RK_s0")
report("K36 precheck well-formed", rc == 0 and "ZZ RK-CACHE-PRECHECK OK RK_s0" in out and "census 0.6300" in out and sorted(os.listdir(f"{R}/runs/RK_s0")) == before and not os.path.exists(f"{R}/runs/RK_s0/gate"), out.strip().splitlines()[-2:])
plant(R, tag="RK_s1")
rc, out = run(R, "precheck", "--tag", "RK_s1")
report("K36b precheck seed 1", rc == 0)
for m, why in (("K23", "missing label_cache"), ("K26", "missing cache_selfcheck"), ("K24", None), ("K28", "the CPU precompute's selfcheck carries dagger_label_cache_sha256"), ("K6", "CENSUS BAR")):
    plant(R, mut=m)
    if m == "K24":                                                                   # the cache edited AFTER the precompute: the CPU selfcheck still reads the old sha
        w(f"{R}/kin_cache/s0_kin_v1.npz", "edited after the precompute")
        why = "the CPU precompute's selfcheck carries dagger_label_cache_sha256"
    rc, out = run(R, "precheck", "--tag", "RK_s0")
    report(f"K37 precheck {m}", rc != 0 and "REFUSED" in out and why in out, [l for l in out.splitlines() if l.startswith("PROBLEM")][:1])
for what, key, val, why in (("K39 precheck: the CPU selfcheck names another cache path", "dagger_label_cache", "/elsewhere/c.npz", "dagger_label_cache ['/elsewhere/c.npz']"),
                            ("K40 precheck: the CPU selfcheck built label speed_v1", "dagger_label", "speed_v1", "dagger_label ['speed_v1']")):
    plant(R)
    spp = f"{R}/kin_cache/s0_selfcheck_cpu.json"
    scj = json.load(open(spp))
    scj["levers"]["dagger"][key] = val
    json.dump(scj, open(spp, "w"))
    rc, out = run(R, "precheck", "--tag", "RK_s0")
    report(what, rc != 0 and "REFUSED" in out and why in out, [l for l in out.splitlines() if l.startswith("PROBLEM")][:1])
rc, out = run(R, "precheck", "--tag", "RA_s0")
report("K38 precheck refuses another tag", rc != 0 and "REFUSED" in out)
# K25b: seed 1 names seed 0's cache
plant(R, tag="RK_s1")
a1 = json.load(open(f"{R}/runs/RK_s1/argv.json"))
a1["argv"][-1] = f"{R}/kin_cache/s0_kin_v1.npz"
w(f"{R}/runs/RK_s1/argv.json", a1)
rc, out = run(R, "mint", "--tag", "RK_s1", "--readout", "causal_g1x")
report("K25b seed 1 argv names seed 0's cache", rc != 0 and "REFUSED" in out and "this seed's bank's cache" in out)

# K21: the record's own census edited after the mint (every bound file unchanged)
plant(R)
run(R, "mint", "--tag", "RK_s0", "--readout", "causal_g1x")
rp = f"{R}/runs/RK_s0/gate/RH_GATE_RECORD.json"
rec = json.load(open(rp))
rec["census"]["relabelled_share_of_ok"] = 0.99
json.dump(rec, open(rp, "w"))
rc, out = run(R, "verify", "--tag", "RK_s0")
report("K21 record census edited", rc != 0 and "census" in out)
rec["census"]["relabelled_share_of_ok"] = 0.63
rec["label"] = "speed_v1"
json.dump(rec, open(rp, "w"))
rc, out = run(R, "verify", "--tag", "RK_s0")
report("K21b record label edited", rc != 0 and "REFUSED" in out)

# K22: the bound selfcheck's counts edited (its sha then differs: refused as a changed bound file, and the share is re-derived)
plant(R)
run(R, "mint", "--tag", "RK_s0", "--readout", "causal_g1x")
sp = f"{R}/runs/RK_s0/selfcheck/selfcheck.json"
sc = json.load(open(sp))
sc["dagger_probe"]["kin_status_counts"] = counts_for(10, 90)
json.dump(sc, open(sp, "w"))
rc, out = run(R, "verify", "--tag", "RK_s0")
report("K22 selfcheck counts edited after the mint", rc != 0 and "REFUSED" in out and "census" in out and "selfcheck" in out)
shutil.rmtree(R)
print("ZZ RK-GATE-RECORD TEST", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
