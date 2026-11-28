"""Build qland/hold_v8f_hacc_lat_result: RESULT_V8F_HACC_LAT (NOT FIXED + the descriptive diagnosis + the instrument noise floor), its runs,
fits, controls and code as run (banked under raw/, not integrated), PREREG_V8F_T1_LATCOND (before any arm is read on its rows), the register
rows, and an erratum on RESULT_V8F_T1KIN_CAUSALBUCKET.  python mk_hold92.py <tip>"""
import glob
import os
import re
import subprocess
import sys

TIP = sys.argv[1]
SC = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/"
T = SC + "t1kin/"
HL = "C:/Users/Admin/hlat/"
H = "C:/Users/Admin/qland/hold_v8f_hacc_lat_result"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RD = f"{PK}/raw/2026-10-26-hacc-lat"
TOKEN = re.compile(rb"hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}")


def git(*a):
    r = subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", *a], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(kind, rel, data, text=True):
    assert not TOKEN.search(data), f"token-shaped string in {rel}"
    if kind == "stage":
        assert git("rev-parse", f"{TIP}:{rel}") is None, f"NEW path exists at the tip: {rel}"
    p = os.path.join(H, kind, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data.replace(b"\r\n", b"\n") if text else data)


def rd(p):
    return open(p, "rb").read()


for n in ("hold_v8f_hacc_lat_result", "landed_hold_v8f_hacc_lat_result"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
assert "ZZHLAT-DONE 12" in open(HL + "logs/chain.out", encoding="utf-8").read()
put("stage", f"{PK}/RESULT_V8F_HACC_LAT.md", rd(T + "RESULT_V8F_HACC_LAT.md"))
put("stage", "Project Steering/PREREG_V8F_T1_LATCOND.md", rd(T + "PREREG_V8F_T1_LATCOND.md"))
for f in ("t1kl_read.py", "lat_tick_probe.py", "lat_tick_probe.log", "lat_tick_probe_s0.json", "lat_tick_probe_s1.json",
          "curv_noise_floor.py", "curv_noise_floor.log", "curv_noise_floor.json"):
    put("stage", f"{RD}/{f}", rd(T + f))
for f in glob.glob(T + "readl/*"):
    put("stage", f"{RD}/read/{os.path.basename(f)}", rd(f))
for f in sorted(glob.glob(HL + "run/t1v8f_*.json")):
    put("stage", f"{RD}/run/{os.path.basename(f)}", rd(f))
for f in sorted(glob.glob(HL + "logs/*")):
    if os.path.isfile(f):
        put("stage", f"{RD}/run/logs/{os.path.basename(f)}", rd(f))
for sub, pat in (("fit", "fit/*.json"), ("ckpt", "ckpt/BAKE_*.json"), ("controls", "controls/*.json")):
    for f in sorted(glob.glob(HL + pat)):
        put("stage", f"{RD}/run/{sub}/{os.path.basename(f)}", rd(f))
for f in ("RUN_SUMMARY.json", "HANDOFF.md", "PINS.sha256", "PINS_H.sha256", "hlat_chain.sh", "ct_smoke.sh", "ct_compare_ticks.py", "hlat_summary.py"):
    put("stage", f"{RD}/run/{f}", rd(HL + f))
put("stage", f"{RD}/run/hlat_chain.cmd", rd(HL + "hlat_chain.cmd"), text=False)
for f in sorted(glob.glob(HL + "work/*.py")):
    put("stage", f"{RD}/run/work/{os.path.basename(f)}", rd(f))
for rel in ("stack/scripts/v8f_hlat.py", "stack/scripts/v8f_t1_eval.py", "stack/tanitad/models/v8f_mini.py", "stack/tests/test_v8f_hlat.py",
            "stack/tests/test_v8f_hlat_gate.py"):
    put("stage", f"{RD}/code/{rel}", rd(HL + "code/" + rel))
for f in sorted(glob.glob(T + "consolidated_l/*")):
    b = os.path.basename(f)
    if "CA1" in b or "CB1" in b:
        continue                                                     # the controls' dumps equal the T1KIN HACC dumps (C-A1 / C-B1: 0.0)
    put("stage", f"{RD}/run/dumps/{b}", rd(f), text=not f.endswith(".npz"))
put("stage", f"{RD}/mk_hold92.py", rd(SC + "grid_a2/mk_hold92.py"))

r = f"{PK}/RESULT_V8F_T1KIN_CAUSALBUCKET.md"
rt = git("show", f"{TIP}:{r}").decode("utf-8")
a = "## Controls (all read before the outcome)\n"
assert rt.count(a) == 1 and "Erratum (2026-10-26)" not in rt
rt = rt.replace(a, ("⚠ **Erratum (2026-10-26).** The PREREG's header says it was written \"~01:40 Berlin\". That label is wrong: its commit `d2ed136d` is"
                    " 01:12:47 +0100, and the first datum (the C-L1 smoke) 01:17:26. The commit order is the evidence that it preceded the data; the"
                    " header was a mis-estimated clock. `PREREG_V8F_T1KIN_STRICT`'s \"~01:55\" is wrong the same way (commit `963906e8` at 01:45:51).\n\n") + a)
put("mods-rw", r, rt.encode("utf-8"))

g = "Project Steering/GOALS_AND_CLAIMS.md"
lines = git("show", f"{TIP}:{g}").decode("utf-8").split("\n")
ix = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-HACC-LAT-1** (2026-10-26) |")]
assert len(ix) == 1
lines[ix[0]] = lines[ix[0]].rstrip().rstrip("|").rstrip() + (
    " **RUN (2026-10-26): NOT FIXED** (verbatim; MEASURED, T1-kin m = 10, paired episode-cluster bootstrap, B = 2,000; `RESULT_V8F_HACC_LAT.md`)."
    " Arm A (λ fallback 0): ADE6 −0.22 / +0.05 m against P10, far worse than HACC. Arm B (v\\* 2 / 8 m/s): −0.876 / −0.761 m, KEEP at s0 only;"
    " curvature still separated-worse at both seeds. Controls all PASS. DESCRIPTIVE: from the same state HACC's plan curvature is no worse than"
    " P10's on train ticks, and B at s1 (the cross axis = P10's below 8 m/s) leaves the eval excess unchanged. INSTRUMENT: 5–10 mm of position"
    " noise on the GT scores 0.021–0.038 1/m curvature MAE and 2.4–4.7 deg/s yaw-rate MAE, the size of the deficit, almost all of it below"
    " 5 m/s. ⇒ The canonical curvature and yaw-rate rows are mm-noise-dominated near standstill (class of `RETR-2026-09-26-YAWMASK`). The committed"
    " box-2 lateral training lever is HELD, named, until `PREREG_V8F_T1_LATCOND` is read. |")
lines.insert(ix[0] + 1, (
    "| **H-V8F-T1-LATCOND-1** (2026-10-26) | On well-conditioned lateral rows (L1: lateral-acceleration MAE; L2 / L3: curvature and yaw-rate MAE on"
    " steps with GT speed ≥ 5.2 m/s, the threshold read from the instrument's noise floor alone), HACC with the causal speed bucket DOMINATES P10"
    " at T1-kin m = 10: ADE6 ≤ −2F, separated, and no row separated-worse, at BOTH seeds | **PRE-REGISTERED, NOT RUN (2026-10-26).**"
    " `Project Steering/PREREG_V8F_T1_LATCOND.md`, landed with `RESULT_V8F_HACC_LAT` before any arm is read on its rows. Re-reads banked dumps;"
    " the canonical rows stay reported beside. Outcomes: CONDITIONED DOMINANCE (⇒ the box-2 lateral lever is dropped and the video re-rendered on"
    " T1 tracks) / CONDITIONED LATERAL DEFICIT (⇒ the box-2 lever runs, aimed at the failing row) / OTHER. Zero training. |"))
put("mods-rw", g, "\n".join(lines).encode("utf-8"))
open(os.path.join(H, "msg.txt"), "wb").write(
    b"HACC-LAT read: NOT FIXED (both zero-training arms); the deficit is not in O's plan and the curvature/yaw rows are mm-noise-dominated "
    b"below 5 m/s; PREREG_V8F_T1_LATCOND\n\n"
    b"PREREG_V8F_HACC_LAT (96604184) read verbatim: arm A (lambda fallback 0) far worse than HACC; arm B (speed gate) keeps the margin but\n"
    b"curvature stays worse. Train-tick probe: HACC's plan curvature is no worse than P10's from the same state. Instrument probe: 5-10 mm of\n"
    b"GT position noise scores the deficit's size. PREREG_V8F_T1_LATCOND (well-conditioned lateral rows) lands before any arm is read on them.\n"
    b"Code as run banked under raw/ (not integrated). Erratum on two PREREG header times.\n\n"
    b"Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
print("hold ok")
