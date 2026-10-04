"""The adopted route fix on the FULL navtest, end to end and unattended:
  1 run_adopted_rest.py   (waits for the shared GPU lock; the default goal path on the 341 remaining changed tokens + 8 controls)
  2 compose_full_seam.py  (the 12,146-token seam; refuses unless controls are bit-identical and the changed set matches)
  3 score_navtest_refe    (W3's harness on the whole navtest scene filter; retried while the harness RAM guard fires)
  4 parse_navtest6        (floors + the PAIRED delta against the unfixed final model, same tokens)
  5 families6             (the four families)
Each step logs to adopted_pipeline.log with its exit code; the run ends with ZZADOPTED_PIPELINE_DONE / _FAIL <step>.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SP = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/a085b71c-4fb3-409b-ae6b-65f605ed575e/scratchpad"
D = "E:/Projects/TanitAD/data/refe_navtest"
PY_T = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
PY_D = "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe"
EV6E = f"{SP}/ev6e"
LOG = os.path.join(HERE, "adopted_pipeline.log")
ENV = dict(os.environ, PYTHONIOENCODING="utf-8", REFE_DRIVE="E:")


def log(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%F %T')} {m}\n")


def step(name, cmd, cwd, out):
    log(f"start {name}")
    with open(out, "w", encoding="utf-8") as f:
        rc = subprocess.call(cmd, cwd=cwd, env=ENV, stdout=f, stderr=subprocess.STDOUT)
    log(f"end {name} rc={rc}")
    return rc


def free_mb():
    r = subprocess.run(["powershell", "-NoProfile", "-Command",
                        "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1024)"], capture_output=True, text=True)
    try:
        return int(r.stdout.strip())
    except ValueError:
        return 0


def main():
    if step("rest_seam", [PY_T, "run_adopted_rest.py"], HERE, os.path.join(HERE, "adopted_rest.out")) != 0:
        log("ZZADOPTED_PIPELINE_FAIL rest_seam"); return 1
    if step("compose", [PY_T, "compose_full_seam.py"], HERE, os.path.join(HERE, "compose.out")) != 0:
        log("ZZADOPTED_PIPELINE_FAIL compose"); return 1
    seam = f"{D}/seams/refe_navtest_final_routefix.npz"
    for att in range(1, 7):
        while free_mb() < 4000:
            time.sleep(60)
        step(f"score attempt {att}", [PY_D, "score_e.py", "--label", "refe_navtest_final_routefix", "--seam", seam,
                                      "--out", f"{D}/score"], SP, os.path.join(HERE, "score_routefix.log"))
        last = open(os.path.join(HERE, "score_routefix.log"), encoding="utf-8", errors="replace").read().splitlines()
        st = next((json.loads(l) for l in reversed(last) if l.startswith("{") and '"status"' in l), {})
        log(f"score status {st.get('status')} {json.dumps(st.get('summary_x100_4dp'))}")
        if st.get("status") == "PASS":
            break
        time.sleep(120)
    else:
        log("ZZADOPTED_PIPELINE_FAIL score"); return 1
    pdir = f"{D}/points/navtest_final_routefix"
    os.makedirs(pdir, exist_ok=True)
    csv_p = f"{D}/score/refe_navtest_final_routefix/refe_navtest_final_routefix.csv"
    step("parse", [PY_T, "parse_navtest6.py", "--arm-csv", csv_p, "--tokens", f"{D}/navtest_all_tokens.json",
                   "--label", "REFe-navtest_final_routefix", "--out", f"{pdir}/3_parse.json",
                   "--extra", f"R6_final_off={D}/score/refe_navtest_final/refe_navtest_final.csv"], EV6E, f"{pdir}/3_parse.log")
    step("families", [PY_T, "families6.py", "--seam", seam, "--inputs",
                      "E:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz", "--stage", "1",
                      "--label", "REFe-navtest_final_routefix", "--out", f"{pdir}/4_families.json", "--n-boot", "2000"],
         EV6E, f"{pdir}/4_families.log")
    log("ZZADOPTED_PIPELINE_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
