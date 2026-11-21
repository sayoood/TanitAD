"""AMEND_HWMPLAN_6 driver: AMEND_5's amend5_run.run_stage on the amend6 folders (amend6_env first). Usage: amend6_run.py plan --stage train|score --chassis s0|s1"""
import amend6_env  # noqa: F401
import argparse
import amend5_run as RUN

ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--stage", required=True); ap.add_argument("--chassis", required=True)
a = ap.parse_args()
d = RUN.run_stage(a.stage, a.chassis, ("A0",))
print("PLANNED", a.stage, a.chassis, [f"{x[0]}_{x[1]}" for x in d], flush=True)
