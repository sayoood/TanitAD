import tanitad
f = tanitad.__file__.replace("\\", "/")
assert "theory/mrate/tree/stack" in f, f
print("tanitad from", f)
