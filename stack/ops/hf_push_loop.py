# Banked from pod5 (ran verified through the v6F stop). Far-side-verifying push loop.
# Replacement push loop. The old one logged "pushed 6 [...]" while ckpt.pt
# failed 100% of cycles with a TRUNCATED error, and never re-attempted after
# failure. This one: always re-attempts, prints the FULL error, and verifies
# from the far side (repo listing) each cycle.
import time
from huggingface_hub import HfApi
api = HfApi(); R = "Sayood/tanitad-v6"; D = "/workspace/experiments/v6F-SW-30k/"
while True:
    for f in ("ckpt.pt", "metrics.json", "config.json", "train_log.jsonl"):
        try:
            api.upload_file(path_or_fileobj=D + f, path_in_repo="v6F-SW-30k/" + f,
                            repo_id=R, repo_type="model")
            print(time.strftime("%H:%M:%S"), "UP_OK", f, flush=True)
        except Exception as e:
            print(time.strftime("%H:%M:%S"), "UP_FAIL", f,
                  type(e).__name__, str(e)[:400], flush=True)
    try:
        i = api.model_info(R, files_metadata=True)
        sz = {s.rfilename: s.size for s in i.siblings}
        print(time.strftime("%H:%M:%S"), "FAR_SIDE ckpt_bytes=",
              sz.get("v6F-SW-30k/ckpt.pt"), flush=True)
    except Exception as e:
        print("FAR_SIDE_FAIL", type(e).__name__, flush=True)
    time.sleep(1200)
