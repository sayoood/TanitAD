"""P1 step 1 - ENUMERATE the Alpamayo record schema. No assumptions."""
import json
import pandas as pd

REC = "C:/Users/Admin/tanitad-data/alpamayo/records.parquet"
df = pd.read_parquet(REC)
print("ROWS:", len(df))
print("COLUMNS:", list(df.columns))
print()
for c in df.columns:
    nn = df[c].notna().sum()
    print("  col=%-28s non_null=%-8d dtype=%s" % (c, nn, df[c].dtype))
print()

# task column?
for cand in ("task", "task_name", "subset", "split"):
    if cand in df.columns:
        print("TASK COUNTS (%s):" % cand)
        print(df[cand].value_counts().to_string())
        print()

# show one row per task, truncated
if "task" in df.columns:
    for t in sorted(df["task"].dropna().unique()):
        sub = df[df["task"] == t]
        print("=" * 70)
        print("TASK %s  rows=%d" % (t, len(sub)))
        r = sub.iloc[0]
        for c in df.columns:
            v = r[c]
            s = str(v)
            if len(s) > 400:
                s = s[:400] + "...<TRUNC len=%d>" % len(str(v))
            print("   %-22s = %s" % (c, s))
        print()
