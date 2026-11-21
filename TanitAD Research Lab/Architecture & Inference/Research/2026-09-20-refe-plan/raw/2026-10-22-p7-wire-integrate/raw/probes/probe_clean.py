import sys, json, tempfile
from pathlib import Path
REFE = Path(sys.argv[1]); sys.path.insert(0, str(REFE))
import session_reserve as SR, test_session_reserve as TS
res = SR.load_reserve(REFE / "fixtures" / "p7_reserve_sessions.json")
print(sorted(res)[:16])
for lg in TS.ALL_LOGS: print(lg, SR.session_of(lg) in res)
