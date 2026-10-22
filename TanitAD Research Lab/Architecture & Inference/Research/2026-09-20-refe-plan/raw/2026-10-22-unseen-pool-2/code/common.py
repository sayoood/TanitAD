"""Shared helpers for the UNSEEN-POOL-2 probe (CPU only, read-only on every input)."""
import re, json, hashlib, collections

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

def session_of(log):            # = draw_reserve.session_of
    return log.rsplit("_", 2)[0]

def vehicle_of(session):        # = draw_reserve.vehicle_of
    return session.split("_")[1]

def vehicle_day_of(session):    # = draw_reserve.unit_of(session, "vehicle-day")
    return "%s_%s" % (session.split("_")[0][:10], vehicle_of(session))

def parse_yaml_lists(path):
    """Minimal parser for NAVSIM scene_filter yaml: returns {'log_names': [...], 'tokens': [...] or None, 'scalars': {...}}."""
    out = {"log_names": [], "tokens": None, "scalars": {}}
    cur = None
    for line in open(path, encoding="utf-8"):
        s = line.rstrip("\n")
        m = re.match(r"^([A-Za-z_]+):\s*(.*)$", s)
        if m and not s.startswith(" ") and not s.startswith("-"):
            k, v = m.group(1), m.group(2).strip()
            if v == "" and k in ("log_names", "tokens"):
                cur = k
                if k == "tokens":
                    out["tokens"] = []
                continue
            cur = None
            out["scalars"][k] = v
            continue
        m = re.match(r"^\s*-\s*'?\"?([^'\"]+?)'?\"?\s*$", s)
        if m and cur:
            (out["log_names"] if cur == "log_names" else out["tokens"]).append(m.group(1))
    return out

def counts(logs):
    logs = sorted(set(logs))
    sess = sorted({session_of(l) for l in logs})
    vd = sorted({vehicle_day_of(s) for s in sess})
    veh = sorted({vehicle_of(s) for s in sess})
    return {"logs": len(logs), "sessions": len(sess), "vehicle_days": len(vd), "vehicles": len(veh)}
