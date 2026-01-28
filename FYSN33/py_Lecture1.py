# =======================================
# Lecture 0: Python Refresher


# =======================================




# --------------------------- traps ------------------------------------

# Mutable defaults
def append_bad(element, seq = []):
    seq.append(element)
    return seq

test = append_bad(1)
tt = append_bad(2)
print(test)
print(tt) # wtf



# Correct way to avoid mutable defaults

def append_good(element, seq = None):
    if seq is None:
        seq = []
    seq.append(element)
    return seq


test2 = append_good(1)
print(test2)
test3 = append_good(2)
print(test3)

L = [1,3,4]
test4 = append_good(2,L)
print(test4)


