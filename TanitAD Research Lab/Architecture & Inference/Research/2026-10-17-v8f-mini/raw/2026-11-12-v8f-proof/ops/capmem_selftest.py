"""capmem_selftest.py <mode> : jobs for the self-test of capmem.py. small = 50 MB, big = 700 MB, hold = 50 MB held for 6 s, argv = print argv and path."""
import os
import sys
import time

import numpy as np

mode = sys.argv[1]
if mode == "small":
    a = np.ones(50 * 1024 * 1024 // 8)
    print("small ok", int(a.sum()))
elif mode == "big":
    a = np.ones(700 * 1024 * 1024 // 8)
    print("big ok", int(a.sum()))
elif mode == "hold":
    a = np.ones(50 * 1024 * 1024 // 8)
    print("hold start %.1f" % time.time(), flush=True)
    time.sleep(6)
    print("hold end %.1f" % time.time(), flush=True)
elif mode == "argv":
    print("argv", sys.argv, "| path0", os.path.basename(sys.path[0]), "| name", __name__)
