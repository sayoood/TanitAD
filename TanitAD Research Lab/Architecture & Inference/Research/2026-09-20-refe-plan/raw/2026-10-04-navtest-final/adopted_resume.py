"""Resume adopted_pipeline.py from the SCORE step. The rest seam ran on the dev-box GPU at 12:55-13:05 (the lock freed);
compose first failed on KeyError 'rest' because run_a9_chunked.run_chunk wrote its chunk token list over the registered
tokens_adopted_rest.json (restored from git, blob 461cd84c; run_chunk fixed to '.chunk.json'), then passed: ZZCOMPOSE_OK,
controls 24/24 + 8/8 bit-identical. This runs score -> parse -> families exactly as adopted_pipeline.main does."""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import adopted_pipeline as A  # noqa: E402


def main():
    A.log("resume from score (compose OK after the token-file restore)")
    seam = f"{A.D}/seams/refe_navtest_final_routefix.npz"
    for att in range(1, 7):
        while A.free_mb() < 4000:
            time.sleep(60)
        A.step(f"score attempt {att}", [A.PY_D, "score_e.py", "--label", "refe_navtest_final_routefix", "--seam", seam,
                                        "--out", f"{A.D}/score"], A.SP, os.path.join(HERE, "score_routefix.log"))
        last = open(os.path.join(HERE, "score_routefix.log"), encoding="utf-8", errors="replace").read().splitlines()
        st = next((json.loads(l) for l in reversed(last) if l.startswith("{") and '"status"' in l), {})
        A.log(f"score status {st.get('status')} {json.dumps(st.get('summary_x100_4dp'))}")
        if st.get("status") == "PASS":
            break
        time.sleep(120)
    else:
        A.log("ZZADOPTED_PIPELINE_FAIL score"); return 1
    pdir = f"{A.D}/points/navtest_final_routefix"
    os.makedirs(pdir, exist_ok=True)
    csv_p = f"{A.D}/score/refe_navtest_final_routefix/refe_navtest_final_routefix.csv"
    A.step("parse", [A.PY_T, "parse_navtest6.py", "--arm-csv", csv_p, "--tokens", f"{A.D}/navtest_all_tokens.json",
                     "--label", "REFe-navtest_final_routefix", "--out", f"{pdir}/3_parse.json",
                     "--extra", f"R6_final_off={A.D}/score/refe_navtest_final/refe_navtest_final.csv"], A.EV6E, f"{pdir}/3_parse.log")
    A.step("families", [A.PY_T, "families6.py", "--seam", seam, "--inputs",
                        "E:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz", "--stage", "1",
                        "--label", "REFe-navtest_final_routefix", "--out", f"{pdir}/4_families.json", "--n-boot", "2000"],
           A.EV6E, f"{pdir}/4_families.log")
    A.log("ZZADOPTED_PIPELINE_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
