"""Insert / update a single row in GOALS_AND_CLAIMS.md by claim ID.

⛔ Cites the claim ID, never a line offset -- the register grew 152 KB in one
session and moved 28 row offsets. Anchors on the ID text itself.
⛔ Writes via a temp file + replace so a mount drop cannot leave a half file.
"""
import io
import os
import sys

G = ("G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/"
     "Project Steering/GOALS_AND_CLAIMS.md")


def main():
    row_file, anchor_id, new_id = sys.argv[1], sys.argv[2], sys.argv[3]
    with io.open(row_file, "r", encoding="utf-8") as f:
        row = f.read().strip("\n")
    with io.open(G, "r", encoding="utf-8") as f:
        txt = f.read()

    # already present? -> replace that row in place
    if new_id in txt:
        lines = txt.split("\n")
        for i, ln in enumerate(lines):
            if ln.startswith("|") and new_id in ln:
                lines[i] = row
                txt2 = "\n".join(lines)
                print("REPLACED existing row for %s" % new_id)
                break
        else:
            print("ID %s present but not as a table row -- refusing" % new_id)
            return 2
    else:
        lines = txt.split("\n")
        for i, ln in enumerate(lines):
            if ln.startswith("|") and anchor_id in ln:
                lines.insert(i + 1, "")
                lines.insert(i + 2, row)
                txt2 = "\n".join(lines)
                print("INSERTED %s after anchor %s (line %d)"
                      % (new_id, anchor_id, i + 1))
                break
        else:
            print("ANCHOR %s NOT FOUND -- refusing to guess a location" % anchor_id)
            return 2

    tmp = G + ".tmp_percprobe"
    with io.open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(txt2)
    os.replace(tmp, G)
    # positive verification with a control that must read non-zero
    with io.open(G, "r", encoding="utf-8") as f:
        back = f.read()
    n = back.count(new_id)
    ctl = back.count(anchor_id)
    print("VERIFY: %s occurs %d time(s); control %s occurs %d time(s)"
          % (new_id, n, anchor_id, ctl))
    if n == 0 or ctl == 0:
        print("INCONCLUSIVE OR LOST")
        return 3
    print("OK bytes=%d" % len(back.encode("utf-8")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
