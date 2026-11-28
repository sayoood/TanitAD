"""mk_jobs.py -- card ADD152: writes the box-2 job files of the queue (CRLF, backslash paths).  python -I mk_jobs.py"""
import os

BS = chr(92)
CHAIN = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "rhacc10k", "chain"))


def win(p):
    return p.replace("/", BS)


def write(name, args):
    lines = ["@echo off", win("D:/tanitad/venv/Scripts/python.exe") + " " + win("D:/tanitad/rhacc10k/kit/box2_headclose_queue.py") + args]
    with open(os.path.join(CHAIN, name), "wb") as f:
        f.write(("\r\n".join(lines) + "\r\n").encode("ascii"))


write("headclose_qall.cmd", "")
write("headclose_ceiling.cmd", " --ceiling")
print("ok")
