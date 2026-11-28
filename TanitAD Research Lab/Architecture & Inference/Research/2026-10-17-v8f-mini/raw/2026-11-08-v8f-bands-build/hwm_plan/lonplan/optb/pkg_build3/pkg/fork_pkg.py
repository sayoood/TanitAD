"""Fork the ACTRSP chain package files to OPTB (names only; behaviour edits are made afterwards by hand). Run once from optb/pkg."""
import os
SRC = "F:/TanitAD-scratch/actrsp/launch/pkg"
BS = chr(92)
FILES = [("actrsp_chain.sh", "optb_chain.sh"), ("actrsp_refresh.py", "optb_refresh.py"), ("actrsp_ship_tree.py", "optb_ship_tree.py"),
         ("stages/actrsp_common.sh", "stages/optb_common.sh"), ("stages/actrsp_run.sh", "stages/optb_run.sh"), ("stages/actrsp_ship.sh", "stages/optb_ship.sh"),
         ("jobs/actrsp_ship_job.cmd", "jobs/optb_ship_job.cmd"), ("jobs/actrsp_stage_job.cmd", "jobs/optb_stage_job.cmd"),
         ("frozen/export_list_actrsp.txt", "frozen/export_list_optb.txt"),
         ("tests/dry_helpers.py", "tests/dry_helpers.py"), ("tests/fake_box.py", "tests/fake_box.py"), ("tests/plant_trainer.py", "tests/plant_trainer.py"),
         ("tests/sftp_stub.sh", "tests/sftp_stub.sh"), ("tests/ssh_stub.sh", "tests/ssh_stub.sh"), ("tests/test_actrsp_dryrun.sh", "tests/test_optb_dryrun.sh"),
         ("README_ACTRSP.md", "README_OPTB.md")]
for a, b in FILES:
    raw = open(f"{SRC}/{a}", "rb").read()
    crlf = raw.count(b"\r\n") > raw.count(b"\n") // 2
    s = raw.decode("utf-8").replace("\r\n", "\n")
    for x, y in (("PILOT_RSP_s0", "PILOT_FTRSP_s0"), ("RSP_s0|RSP_s2|NULL_s4|RSP_s1|RSP_s3|NULL_s5", "FTRSP_s0|FTNULL_s0|FTRSP_s2|FTNULL_s2|FTNULL_s4|FTRSP_s1|FTNULL_s1|FTRSP_s3|FTNULL_s3|FTNULL_s5"),
                 ("ACTRSP", "OPTB"), ("actrsp", "optb")):
        s = s.replace(x, y)
    os.makedirs(os.path.dirname(b) or ".", exist_ok=True)
    open(b, "wb").write(s.replace("\n", "\r\n").encode("utf-8") if crlf else s.encode("utf-8"))
    print(b, "CRLF" if crlf else "LF")
