"""test_ki_gate_record.py -- Dev-box test of ki_gate_record.py on planted files (stdlib only, no pod). K-INPUT (card PREREG_V8F_KINPUT, commit 4097a89e). There is no label cache in KI: no cache cases.

A well-formed KI_s0 / KI_s1 record must MINT and VERIFY; each mutation must REFUSE, and for the stated REASON (a mutation that refuses for another reason is a test bug):
  G1 argv without --dagger-base            G2 --dagger-base own                  G3 built levers say dagger_base own      G4 probe base_source own            G5 probe rows_changed.base 7
  G6 rows_changed missing                  G7 n_selected_at_p1 0                 G8 G-SUITE at 1a8b11a1 (RK's commit)     G9 G-SUITE reads a regression       G10 tag RK_s0
  G11 the KI tree marker missing           G12 bank sha256 in argv is not bank.json's                                    G13 --dagger-mode swap in the argv   G14 grad_diff_all 0
  G15 control_p0_grad_diff != 0            G16 --dagger-label kin_v1 in the argv G17 --dagger-bank is bank_k              G18 the pair check did not pass      G19 built levers dagger_label speed_v1
  G20 --dagger-label-cache in the argv     G21 dagger_probe pass false           G22 --dagger-kin-tb in the argv          G23 declared_vs_built FAIL          G24 acc_probe fail
  G25 pilot readout differs                G26 bank check fails                  G27 n_selected_at_p1 key missing         G28 base_source key missing         G29 two rows_changed blocks
  G30 the built levers carry no dagger_base
The commit placeholder (G31), verify after a changed bound file / removed marker / edited record / edited bound selfcheck (V1-V7), and the pre-lock `precheck` (P1-P7: well-formed writes nothing; each refusal for its reason) follow.
Usage: python -I test_ki_gate_record.py <ki_gate_record.py> <scratch dir>"""
import hashlib
import json
import os
import shutil
import subprocess
import sys

GR0, ROOT = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
GR = GR0
COMMIT = "4097a89e4b4a6c9bf309cbae9dc6a8dd1e574fcb"
OLD1A = "1a8b11a1abd50c09636da8b7425e97573cdf1d04"
os.makedirs(ROOT, exist_ok=True)
src = open(GR0, encoding="utf-8").read()
assert f'COMMIT = "{COMMIT}"' in src, "the gate record under test does not carry the K-INPUT commit"


def w(p, o):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(o if isinstance(o, str) else json.dumps(o))


def plant(R, tag="KI_s0", mut=None):
    if os.path.exists(R):
        shutil.rmtree(R)
    seed = tag.split("_s")[1]
    w(f"{R}/bank/s{seed}/bank.json", {"bank": "planted"})
    bsha = hashlib.sha256(open(f"{R}/bank/s{seed}/bank.json", "rb").read()).hexdigest()
    w(f"{R}/bank/s{seed}/BANK_CHECK.json", {"pass": mut != "G26", "readout": "causal_g1x"})
    bank = f"{R}/bank_k/s{seed}" if mut == "G17" else f"{R}/bank/s{seed}"
    argv = ["--dagger-p", "0.1", "--dagger-bank", bank, "--dagger-bank-sha256", bsha if mut != "G12" else "0" * 64,
            "--dagger-bank-readout", "causal_g1x", "--dagger-bank-kappa", "0.2", "--out", f"{R}/runs/{tag}/train"]
    if mut == "G13":
        argv += ["--dagger-mode", "swap"]
    if mut == "G16":
        argv += ["--dagger-label", "kin_v1"]
    if mut == "G20":
        argv += ["--dagger-label-cache", f"{R}/kin_cache/s{seed}_kin_v1.npz"]
    if mut == "G22":
        argv += ["--dagger-kin-tb", "0.5"]
    if mut != "G1":
        argv += ["--dagger-base", "own" if mut == "G2" else "logged"]
    w(f"{R}/runs/{tag}/argv.json", {"argv": argv})
    probe = {"pass": mut != "G21", "n_rows": 64, "n_selected_at_p1": 0 if mut == "G7" else 12, "base_source": "own" if mut == "G4" else "logged",
             "rows_changed": {"p10f": 12, "base": 7 if mut == "G5" else 0, "kin3": 12, "p10_t": 12, "pose": 12, "pose_t": 12, "pose_s": 12, "nav": 3},
             "grad_diff_all": 0.0 if mut == "G14" else 3.5, "control_p0_grad_diff": 0.25 if mut == "G15" else 0.0}
    if mut == "G6":
        del probe["rows_changed"]
    if mut == "G27":
        del probe["n_selected_at_p1"]
    if mut == "G28":
        del probe["base_source"]
    if mut == "G29":
        probe["live_swapped"] = {"rows_changed": {"base": 0}}
    lev = {"dagger_bank_readout": "causal_g1x", "dagger_label": "speed_v1" if mut == "G19" else "human", "dagger_base": "own" if mut == "G3" else "logged"}
    if mut == "G30":
        del lev["dagger_base"]
    w(f"{R}/runs/{tag}/selfcheck/selfcheck.json", {"declared_vs_built": "FAIL" if mut == "G23" else "PASS", "acc_probe": {"pass": mut != "G24"}, "dagger_probe": probe, "levers": {"dagger": lev}})
    w(f"{R}/gsuite_result_ki.json", {"new_commit": OLD1A if mut == "G8" else COMMIT, "n_regressions": 1 if mut == "G9" else 0, "new_test_failures": [], "missing_in_new": []})
    w(f"{R}/pilot/PILOT_RH.json", {"pass": True, "readout": "causal_g1" if mut == "G25" else "causal_g1x"})
    w(f"{R}/done_rk/tree.done", "x")
    if mut != "G11":
        w(f"{R}/done_ki/tree.done", "x")
    w(f"{R}/runs/{tag}/pair_{tag}.log", "no pass here\n" if mut == "G18" else "ZZ ROLLHIST-PAIR PASS\n")


def run(R, *a, gr=None, inp=None):
    env = dict(os.environ, RH_ROOT=R)
    p = subprocess.run([sys.executable, "-I", gr or GR, *a], env=env, capture_output=True, text=True, encoding="utf-8", input=inp)
    return p.returncode, p.stdout + p.stderr


ok = True
R = os.path.join(ROOT, "R")


def report(name, good, extra=""):
    global ok
    ok &= bool(good)
    print(name, "PASS" if good else "FAIL", extra)


# well-formed, both seeds
for tag in ("KI_s0", "KI_s1"):
    plant(R, tag=tag)
    rc, out = run(R, "mint", "--tag", tag, "--readout", "causal_g1x")
    rc2, out2 = run(R, "verify", "--tag", tag)
    rec = json.load(open(f"{R}/runs/{tag}/gate/RH_GATE_RECORD.json")) if rc == 0 else {}
    report(f"WELL-FORMED {tag} mint+verify", rc == 0 and "MINTED" in out and rc2 == 0 and "VERIFY OK" in out2 and rec["probe"] == {"base_source": ["logged"], "rows_changed_base": [0], "n_selected_at_p1": [12]}
           and rec["base"] == "logged" and "label_cache" not in rec["binds"], (out.strip().splitlines()[-2:], out2.strip().splitlines()[-1:]))

MUT = [("G1", "KI_s0", "--dagger-base is None"), ("G2", "KI_s0", "--dagger-base is 'own', not logged"), ("G3", "KI_s0", "built levers carry dagger_base ['own']"),
       ("G4", "KI_s0", "base_source ['own']"), ("G5", "KI_s0", "rows_changed.base [7]"), ("G6", "KI_s0", "rows_changed.base []"), ("G7", "KI_s0", "n_selected_at_p1 [0]"),
       ("G8", "KI_s0", "G-SUITE new commit is '1a8b11a1abd50c09636da8b7425e97573cdf1d04'"), ("G9", "KI_s0", "G-SUITE reads regressions"), ("G10", "RK_s0", "serves KI_s0 / KI_s1 only"),
       ("G11", "KI_s0", "missing tree_done"), ("G12", "KI_s0", "not the sha256 of bank.json"), ("G13", "KI_s0", "names --dagger-mode"), ("G14", "KI_s0", "grad_diff_all [0.0]"),
       ("G15", "KI_s0", "control_p0_grad_diff reads [0.25]"), ("G16", "KI_s0", "names --dagger-label "), ("G17", "KI_s0", "not the RH bank"), ("G18", "KI_s0", "did not print ZZ ROLLHIST-PAIR PASS"),
       ("G19", "KI_s0", "dagger_label ['speed_v1']"), ("G20", "KI_s0", "names --dagger-label-cache"), ("G21", "KI_s0", "dagger_probe did not pass"), ("G22", "KI_s0", "names --dagger-kin-tb"),
       ("G23", "KI_s0", "declared_vs_built = 'FAIL'"), ("G24", "KI_s0", "acc_probe did not pass"), ("G25", "KI_s0", "PILOT_RH.json pass=True readout='causal_g1'"),
       ("G26", "KI_s0", "full-bank check json is not a pass"), ("G27", "KI_s0", "n_selected_at_p1 []"), ("G28", "KI_s0", "base_source []"), ("G29", "KI_s0", "rows_changed.base [0, 0]"),
       ("G30", "KI_s0", "built levers carry dagger_base []")]
for m, tag, why in MUT:
    plant(R, tag=("KI_s0" if tag == "RK_s0" else tag), mut=m)
    if tag == "RK_s0":                                               # G10: the files of a KI arm, the RK tag
        shutil.copytree(f"{R}/runs/KI_s0", f"{R}/runs/RK_s0")
    rc, out = run(R, "mint", "--tag", tag, "--readout", "causal_g1x")
    red = rc != 0 and "REFUSED" in out and why in out and not os.path.exists(f"{R}/runs/{tag}/gate/RH_GATE_RECORD.json")
    report(m, red, [l for l in out.splitlines() if l.startswith("PROBLEM")][:2] if not red else f"RED ({why})")

# G31: the commit placeholder unfilled (a copy of the gate record whose COMMIT is the placeholder)
unf = os.path.join(ROOT, "ki_gate_record_unfilled.py")
open(unf, "w", encoding="utf-8", newline="\n").write(src.replace(f'COMMIT = "{COMMIT}"', 'COMMIT = "__KI_COMMIT__"'))
plant(R)
rc, out = run(R, "mint", "--tag", "KI_s0", "--readout", "causal_g1x", gr=unf)
report("G31", rc != 0 and "REFUSED" in out and "filled 40-hex commit" in out and not os.path.exists(f"{R}/runs/KI_s0/gate/RH_GATE_RECORD.json"), "RED (placeholder refuses)")
run(R, "mint", "--tag", "KI_s0", "--readout", "causal_g1x")                                    # a real record, then verified by the UNFILLED copy: the commit differs
rc, out = run(R, "verify", "--tag", "KI_s0", gr=unf)
report("G31b verify under the placeholder", rc != 0 and "REFUSED" in out and "not a pass for this commit" in out)

# V1-V4: a bound file changes after the mint
for what, path, newtxt in (("gsuite", f"{R}/gsuite_result_ki.json", json.dumps({"new_commit": COMMIT, "n_regressions": 0, "new_test_failures": [], "missing_in_new": [], "touched": 1})),
                           ("argv", f"{R}/runs/KI_s0/argv.json", None), ("bank.json", f"{R}/bank/s0/bank.json", json.dumps({"bank": "edited"})),
                           ("tree marker", f"{R}/done_ki/tree.done", "y")):
    plant(R)
    run(R, "mint", "--tag", "KI_s0", "--readout", "causal_g1x")
    if newtxt is None:
        a = json.load(open(path))
        a["argv"] += ["--w-extra", "1"]
        newtxt = json.dumps(a)
    w(path, newtxt)
    rc, out = run(R, "verify", "--tag", "KI_s0")
    report(f"V1 {what} changed after the mint", rc != 0 and "REFUSED" in out and "changed or missing since the mint" in out)
os.remove(f"{R}/done_ki/tree.done")
rc, out = run(R, "verify", "--tag", "KI_s0")
report("V2 tree marker removed after the mint", rc != 0 and "REFUSED" in out)

# V3: the record's own probe / base edited (every bound file unchanged)
plant(R)
run(R, "mint", "--tag", "KI_s0", "--readout", "causal_g1x")
rp = f"{R}/runs/KI_s0/gate/RH_GATE_RECORD.json"
rec = json.load(open(rp))
rec["probe"]["rows_changed_base"] = [5]
json.dump(rec, open(rp, "w"))
rc, out = run(R, "verify", "--tag", "KI_s0")
report("V3 record probe edited", rc != 0 and "probe" in out and "REFUSED" in out)
rec["probe"]["rows_changed_base"] = [0]
rec["base"] = "own"
json.dump(rec, open(rp, "w"))
rc, out = run(R, "verify", "--tag", "KI_s0")
report("V3b record base edited", rc != 0 and "REFUSED" in out and "not a pass for this commit, base and tag" in out)

# V4: the bound selfcheck's rows_changed.base edited (its sha then differs: refused as a changed bound file, and the probe is re-derived)
plant(R)
run(R, "mint", "--tag", "KI_s0", "--readout", "causal_g1x")
sp = f"{R}/runs/KI_s0/selfcheck/selfcheck.json"
sc = json.load(open(sp))
sc["dagger_probe"]["rows_changed"]["base"] = 4
json.dump(sc, open(sp, "w"))
rc, out = run(R, "verify", "--tag", "KI_s0")
report("V4 selfcheck rows_changed.base edited after the mint", rc != 0 and "REFUSED" in out and "selfcheck" in out and "probe" in out)

# V5: the record dropped a binding (every remaining bound file unchanged): the record is still verified against what it binds, so it must still be a pass only if the selfcheck binding is present
plant(R)
run(R, "mint", "--tag", "KI_s0", "--readout", "causal_g1x")
rec = json.load(open(rp))
del rec["binds"]["selfcheck"]
json.dump(rec, open(rp, "w"))
rc, out = run(R, "verify", "--tag", "KI_s0")
report("V5 record without its selfcheck binding", rc != 0 and "REFUSED" in out and "no bound selfcheck" in out)

# P1-P7: the pre-lock precheck (the chain's step `inputs_gate` and the DRY walk): well-formed OK and writes nothing; each refusal for its reason
plant(R)
os.remove(f"{R}/runs/KI_s0/selfcheck/selfcheck.json")                         # the pre-lock state: no selfcheck, no pair log, no tree marker yet
os.remove(f"{R}/runs/KI_s0/pair_KI_s0.log")
os.remove(f"{R}/done_ki/tree.done")
before = sorted(os.listdir(f"{R}/runs/KI_s0"))
rc, out = run(R, "precheck", "--tag", "KI_s0", "--readout", "causal_g1x")
report("P1 precheck well-formed (no selfcheck needed), writes nothing", rc == 0 and "ZZ KI-INPUTS-PRECHECK OK KI_s0" in out and sorted(os.listdir(f"{R}/runs/KI_s0")) == before and not os.path.exists(f"{R}/runs/KI_s0/gate"), out.strip().splitlines()[-2:])
plant(R, tag="KI_s1")
rc, out = run(R, "precheck", "--tag", "KI_s1", "--readout", "causal_g1x")
report("P1b precheck seed 1", rc == 0)
for m, why in (("G8", "G-SUITE new commit is '1a8b11a1"), ("G9", "G-SUITE reads regressions"), ("G12", "not the sha256 of bank.json"), ("G2", "--dagger-base is 'own'"), ("G1", "--dagger-base is None"),
               ("G16", "names --dagger-label "), ("G17", "not the RH bank"), ("G25", "PILOT_RH.json"), ("G26", "full-bank check json")):
    plant(R, mut=m)
    rc, out = run(R, "precheck", "--tag", "KI_s0", "--readout", "causal_g1x")
    report(f"P2 precheck {m}", rc != 0 and "REFUSED" in out and why in out, [l for l in out.splitlines() if l.startswith("PROBLEM")][:1])
plant(R)
os.remove(f"{R}/gsuite_result_ki.json")
rc, out = run(R, "precheck", "--tag", "KI_s0", "--readout", "causal_g1x")
report("P3 precheck: G-SUITE result missing", rc != 0 and "missing gsuite" in out)
rc, out = run(R, "precheck", "--tag", "RK_s0", "--readout", "causal_g1x")
report("P4 precheck refuses another tag", rc != 0 and "REFUSED" in out)
# P5-P7: the argv on stdin (`--argv -`, the pre-lock walk before runs/KI_s0/argv.json exists): OK when well-formed, refused for a base / dose / extra-flag problem or garbage
plant(R)
good = open(f"{R}/runs/KI_s0/argv.json", encoding="utf-8").read()
shutil.rmtree(f"{R}/runs/KI_s0")                                                           # no argv file on disk at all
before = sorted(os.listdir(R))
rc, out = run(R, "precheck", "--tag", "KI_s0", "--readout", "causal_g1x", "--argv", "-", inp=good)
report("P5 precheck with the argv on stdin (no argv file), writes nothing", rc == 0 and "ZZ KI-INPUTS-PRECHECK OK KI_s0" in out and sorted(os.listdir(R)) == before and not os.path.exists(f"{R}/runs/KI_s0"), out.strip().splitlines()[-1:])
g = json.loads(good)
for name, mod, why in (("P6a base own", lambda a: a[:-1] + ["own"], "--dagger-base is 'own'"), ("P6b extra label flag", lambda a: a + ["--dagger-label", "kin_v1"], "names --dagger-label "),
                       ("P6c no base flag", lambda a: a[:-2], "--dagger-base is None"), ("P6d extra cache flag", lambda a: a + ["--dagger-label-cache", "/x.npz"], "names --dagger-label-cache")):
    rc, out = run(R, "precheck", "--tag", "KI_s0", "--readout", "causal_g1x", "--argv", "-", inp=json.dumps({"argv": mod(g["argv"])}))
    report(f"P6 precheck stdin {name}", rc != 0 and "REFUSED" in out and why in out, [l for l in out.splitlines() if l.startswith("PROBLEM")][:1])
rc, out = run(R, "precheck", "--tag", "KI_s0", "--readout", "causal_g1x", "--argv", "-", inp="not json")
report("P7 precheck stdin garbage is refused, not a crash", rc != 0 and "REFUSED" in out and "the argv is unreadable" in out and "Traceback" not in out)
shutil.rmtree(R)
print("ZZ KI-GATE-RECORD TEST", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
