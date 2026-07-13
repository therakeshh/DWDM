import math
from collections import Counter
from pprint import pprint

# Attributes defined once
attributes = ["Age", "Competition", "Type", "Profit"]

# Dataset rows
rows = [
    ["old","Yes","SW","Down"],
    ["old","No","SW","Down"],
    ["old","No","HW","Down"],
    ["mild","Yes","SW","Down"],
    ["mild","Yes","HW","Down"],
    ["mild","No","HW","Up"],
    ["mild","No","SW","Up"],
    ["new","Yes","SW","Up"],
    ["new","No","HW","Up"],
    ["new","No","SW","Up"],

]

# Convert rows to dicts
data = [dict(zip(attributes, row)) for row in rows]

# Entropy
def entropy(data):
    labels = [row["Profit"] for row in data]
    total = len(labels)
    counts = Counter(labels)
    return sum((-c/total)*math.log2(c/total) for c in counts.values())

# Info Gain
def info_gain(data, attr):
    total_entropy = entropy(data)
    values = set(row[attr] for row in data)
    return total_entropy - sum((len(sub)/len(data))*entropy(sub)
        for v in values for sub in [[r for r in data if r[attr]==v]])

# ID3
def id3(data, attrs):
    labels = [row["Profit"] for row in data]
    if labels.count(labels[0]) == len(labels): return labels[0]
    if not attrs: return Counter(labels).most_common(1)[0][0]
    gains = {a: info_gain(data,a) for a in attrs if a!="Profit"}
    best = max(gains,key=gains.get)
    tree = {best:{}}
    for v in set(row[best] for row in data):
        subset = [r for r in data if r[best]==v]
        tree[best][v] = id3(subset,[a for a in attrs if a!=best])
    return tree

# Build tree
decision_tree = id3(data, attributes)
print(decision_tree)
pprint(decision_tree)
