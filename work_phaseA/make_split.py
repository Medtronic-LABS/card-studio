"""A1 + A2: merged labels copy, seeded page-level 70/15/15 split, 5-fold page CV. Writes split.json."""
import csv, json, random, collections, shutil
from common import *

# A1 -- labels_new.csv wins on conflict; it does not exist, so merged == labels.csv
new = WORK / "labels_new.csv"
assert not new.exists(), "labels_new.csv now exists: merge logic needed"
shutil.copy(WORK / "labels.csv", HERE / "merged_labels.csv")
rows = read_labels(HERE / "merged_labels.csv")
assert len({(r["page"], r["row"], r["col"]) for r in rows}) == len(rows)

pages = sorted({r["page"] for r in rows})
by_page = collections.defaultdict(collections.Counter)
for r in rows:
    by_page[r["page"]][r["row"]] += 1
tot = collections.Counter(r["row"] for r in rows)
FIELDS = [f for f, n in tot.items() if n >= 13]   # fields we try to cover in test (n>=13 total)
NCELL = len(rows)

def counts(ps):
    c = collections.Counter()
    for p in ps:
        c.update(by_page[p])
    return c

def score(test, val):
    ct, cv_ = counts(test), counts(val)
    cov_t = sum(ct[f] >= 3 for f in FIELDS)
    cov_v = sum(cv_[f] >= 2 for f in FIELDS)
    dev = abs(sum(ct.values()) / NCELL - .15) + abs(sum(cv_.values()) / NCELL - .15)
    return (cov_t, cov_v, -dev)

rng = random.Random(20261103)
best, bestkey = None, None
n_test = n_val = round(0.15 * len(pages))
for _ in range(30000):
    s = pages[:]; rng.shuffle(s)
    test, val = s[:n_test], s[n_test:n_test + n_val]
    k = score(test, val)
    if bestkey is None or k > bestkey:
        best, bestkey = (sorted(test), sorted(val), sorted(s[n_test + n_val:])), k
test, val, train = best

# 5-fold page CV, same style of search for balanced field coverage per fold
bestf, bestfk = None, None
for _ in range(30000):
    s = pages[:]; rng.shuffle(s)
    folds = [sorted(s[i::5]) for i in range(5)]
    mins = [min(counts(f)[x] for x in FIELDS) for f in folds]
    cells = [sum(counts(f).values()) for f in folds]
    k = (sum(mins), -(max(cells) - min(cells)))
    if bestfk is None or k > bestfk:
        bestf, bestfk = folds, k

for a, b in [(train, val), (train, test), (val, test)]:
    assert not set(a) & set(b)
assert sorted(train + val + test) == pages
assert sorted(sum(bestf, [])) == pages and all(not set(bestf[i]) & set(bestf[j]) for i in range(5) for j in range(i))
json.dump({"seed": 20261103, "unit": "page (writer unknown; page is the split unit)", "train": train, "val": val, "test": test,
           "cv_folds": bestf}, open(HERE / "split.json", "w"), indent=1)

def show(name, ps):
    c = counts(ps); print(f"{name:6} pages={len(ps):2d} cells={sum(c.values()):3d} ", {f: c[f] for f in sorted(tot, key=lambda x: -tot[x])})
print("totals", dict(tot.most_common()))
show("train", train); show("val", val); show("test", test)
for i, f in enumerate(bestf): show(f"fold{i}", f)
print("test fields with >=3 cells:", [f for f in FIELDS if counts(test)[f] >= 3], "of", len(FIELDS))
print("test fields (n>=13) with <3:", {f: counts(test)[f] for f in FIELDS if counts(test)[f] < 3})
