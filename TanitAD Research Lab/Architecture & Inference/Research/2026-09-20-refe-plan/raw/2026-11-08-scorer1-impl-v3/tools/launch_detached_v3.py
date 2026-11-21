import subprocess
F = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
subprocess.Popen([r"C:\Users\Admin\venvs\tanitad\Scripts\python.exe",
                  r"E:\Projects\TanitAD-artifacts\scorer1_impl_20261108\v3\tools\run_suite_v3.py",
                  r"F:\scorer1_v3\t", r"F:\scorer1_v3\res", "3", "2400"],
                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=F)
