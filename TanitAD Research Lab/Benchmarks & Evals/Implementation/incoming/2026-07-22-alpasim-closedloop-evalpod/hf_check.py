import sys
tok = sys.stdin.readline().strip().lstrip("﻿")
from huggingface_hub import HfApi
api = HfApi(token=tok)
print("=== Sayood models ===")
try:
    for m in api.list_models(author="Sayood"):
        print(" ", m.id)
except Exception as e:
    print("list_models failed:", repr(e)[:150])
print("=== targeted repo existence ===")
for rid in ["Sayood/flagship-v15", "Sayood/flagship-v16", "Sayood/flagship-v1.5",
            "Sayood/flagship-v1.6", "Sayood/flagship-4b-phase0"]:
    try:
        api.model_info(rid)
        print("  EXISTS", rid)
    except Exception as e:
        print("  no", rid, "->", type(e).__name__)
print("HF_CHECK_DONE")
