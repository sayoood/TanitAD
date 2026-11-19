"""run_mutations_actcons.py -- runs the mutation list of actcons on the pod (CPU, one process at a time) through common/pod_mut_driver.py and writes MUTATIONS_ACTCONS.json.
Steps (each a plain command from the dev box, Git bash):
  1. python make_spec_actcons.py                      # checks every old text occurs once, writes spec_actcons.json
  2. bash ../common/podsetup.sh actcons <pod tree>    # copies the lever tree twice on the pod and pushes the spec  (distrule: the src folder is pushed as in HANDOFF.md)
  3. bash ../common/podrun.sh actcons [--only ID,ID]  # control, every mutation, in a fresh child process each; one pod command at most 900 s
  4. bash ../common/podrun.sh actcons --control-only  # control after
  5. python ../common/finalize.py actcons pod_out_actcons.json spec_actcons.json MUTATIONS_ACTCONS.json
This file only runs steps 1 and 5 locally; the pod steps are the commands above."""
import subprocess
import sys

HERE = __file__.rsplit("/", 1)[0] if "/" in __file__ else "."
subprocess.check_call([sys.executable, "make_spec_actcons.py"], cwd=HERE)
subprocess.check_call([sys.executable, "../common/finalize.py", "actcons", "pod_out_actcons.json", "spec_actcons.json", "MUTATIONS_ACTCONS.json"], cwd=HERE)
