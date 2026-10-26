import json
from pathlib import Path
p = Path("C:/Users/Admin/refcv8_pf/refcv9_r9pf/code/next_r9pf.json"); x = json.loads(p.read_text(encoding="utf-8"))
pre = ("**VOID first: the one failed control is K-EQ, a numerical-tolerance failure, not a logic one.** Decisions agree 515/515 at both seeds; "
       "max |d| is 3.05e-5 m (seed 0) and 4.58e-5 m (seed 1); 28 and 24 of 515 windows exceed 2 ulp of their own coordinates. "
       "Cheapest discriminator, not run: the same ON roll on CPU (~15-30 CPU-min, no GPU). "
       "If CPU in-forward == offline within 2 ulp -> the residual is CUDA-vs-CPU head arithmetic, and a Master-Mind ruling on the tolerance "
       "(amendment 2) would turn the bars below into the registered read. If not -> a real composition mismatch to find. "
       "**Blocked on:** that roll + the MM ruling. The bars are BESIDE until then. The provisional word on them, had K-EQ held, is FAIL-GUARD.\n\n")
x["text"] = pre + x["text"].split("**VOID first", 1)[-1] if not x["text"].startswith("**VOID first") else x["text"]
p.write_text(json.dumps(x, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"); print("ok")
