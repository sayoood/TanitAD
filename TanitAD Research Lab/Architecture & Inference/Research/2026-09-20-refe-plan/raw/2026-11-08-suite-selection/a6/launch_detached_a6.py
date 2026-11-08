import subprocess
F = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
subprocess.Popen(["cmd", "/c", r"E:\Projects\TanitAD-artifacts\tsf_bundle2_20261107\a6\chain_a6.cmd"],
                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=F)
