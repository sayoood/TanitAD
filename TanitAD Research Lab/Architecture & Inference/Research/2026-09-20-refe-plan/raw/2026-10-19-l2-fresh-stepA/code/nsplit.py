import json
import sys

n = int(json.load(open(sys.argv[1], encoding="utf-8"))["n"])
h = ((n // 2) // 10) * 10                      # export shards are 10 tokens; worker 1 takes [0,h), worker 2 [h,n)
print(n, h)
