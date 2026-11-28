import subprocess
F = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
subprocess.Popen(["cmd", "/c", r"E:\Projects\TanitAD-artifacts\scorer_gym_20261107\lane_truth\code\chain_lane.cmd"],
                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=F)
