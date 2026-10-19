#!/usr/bin/env python3
"""Shared helper: read a banked PDF page into visual lines (word y-clustering) so a table row can be checked
as an ORDERED SUBSEQUENCE of numbers on one line.  Content check, not presence check."""
import re, sys, glob, os
import fitz

def find_pdf(lib_root, key):
    c = glob.glob(os.path.join(lib_root, key + "_*.pdf"))
    if not c:
        raise FileNotFoundError(key)
    return c[0]

_cache = {}
def page_lines(pdf, pno, ytol=3.0):
    k = (pdf, pno)
    if k in _cache:
        return _cache[k]
    doc = fitz.open(pdf)
    page = doc[pno - 1]
    words = page.get_text("words")  # x0,y0,x1,y1,word,block,line,wordno
    # cluster by vertical centre
    ws = sorted(((w[1] + w[3]) / 2, w[0], w[4]) for w in words)
    lines = []
    for yc, x, t in ws:
        if lines and abs(lines[-1]["y"] - yc) <= ytol:
            lines[-1]["w"].append((x, t))
            lines[-1]["y"] = (lines[-1]["y"] * (len(lines[-1]["w"]) - 1) + yc) / len(lines[-1]["w"])
        else:
            lines.append({"y": yc, "w": [(x, t)]})
    out = []
    for L in lines:
        L["w"].sort()
        out.append([t for _, t in L["w"]])
    _cache[k] = out
    return out

NUM = re.compile(r"^[<>~≈]?[-+−]?\d+(?:[.,]\d+)?%?$")
def tokens(line):
    toks = []
    for t in line:
        t2 = t.replace("−", "-").replace("±", " ").strip()
        for p in t2.split():
            toks.append(p)
    return toks

def ordered_subseq(nums, toks):
    i = 0
    for t in toks:
        if i < len(nums) and (t == nums[i] or t.rstrip("%") == nums[i]):
            i += 1
    return i == len(nums)

def row_found(pdf, pno, nums, label=None):
    """True if some visual line on the page contains nums as an ordered subsequence (and the label words, if given)."""
    for line in page_lines(pdf, pno):
        toks = tokens(line)
        if ordered_subseq(nums, toks):
            if label is None or all(w.lower() in " ".join(toks).lower() for w in label.split("|")):
                return True
    return False

def pages_with(pdf, nums):
    doc = fitz.open(pdf)
    return [p for p in range(1, len(doc) + 1) if row_found(pdf, p, nums)]
