"""config.json copy for the TRAIN-DIAG pass: identical to the run's, except the eval-side v9 md5 names the TRAIN v9 file the remap points at."""
import json, sys
src, dst = sys.argv[1], sys.argv[2]
c = json.load(open(src))
argv = c["argv"]
i = argv.index("--r8-v9-eval-md5")
old = argv[i + 1]
j = argv.index("--r8-v9-md5")
argv[i + 1] = argv[j + 1]                    # the TRAIN v9 file's md5 (83e30bd3...), already in the run's own argv
json.dump(c, open(dst, "w"))
print("r8-v9-eval-md5", old, "->", argv[i + 1])
