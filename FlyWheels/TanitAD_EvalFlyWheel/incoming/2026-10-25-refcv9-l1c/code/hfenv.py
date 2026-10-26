"""Load the HF token from Keys.txt IN PLACE into the process env (never printed, never an argv). TLS via truststore."""
import os, re
def init():
    try:
        import truststore; truststore.inject_into_ssl()
    except Exception as e:
        print("[hfenv] truststore unavailable:", e)
    p = "G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/Keys.txt"
    txt = open(p, encoding="utf-8", errors="ignore").read()
    m = re.search(r"hf_[A-Za-z0-9]+", txt)
    if not m: raise SystemExit("[hfenv] no hf_ token in Keys.txt")
    os.environ["HF_TOKEN"] = m.group(0)
    os.environ["HUGGING_FACE_HUB_TOKEN"] = m.group(0)
    os.environ.setdefault("HF_HOME", "F:/l1c/hf_home")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    return True
