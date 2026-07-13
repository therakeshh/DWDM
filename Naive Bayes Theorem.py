from collections import Counter

# Define attributes once
attributes = ["Confident", "Studied", "Sick", "Result"]

# Dataset rows
rows = [
    ["Yes","No","No","Fail"],
    ["Yes","No","No","Pass"],
    ["No","Yes","Yes","Fail"],
    ["No","Yes","Yes","Pass"],
    ["Yes","Yes","Yes","Pass"],
]

# Convert rows into list of dictionaries
data = [dict(zip(attributes, row)) for row in rows]

# Separate by class
classes = [row["Result"] for row in data]
class_counts = Counter(classes)
total = len(data)

# Prior probabilities
priors = {c: class_counts[c]/total for c in class_counts}

# Conditional probability function
def cond_prob(attr, value, target_class):
    subset = [row for row in data if row["Result"] == target_class]
    count = sum(1 for row in subset if row[attr] == value)
    return count / len(subset) if subset else 0

# Naive Bayes prediction
def predict(instance):
    probs = {}
    for c in class_counts:
        prob = priors[c]
        for attr, value in instance.items():
            prob *= cond_prob(attr, value, c)
        probs[c] = prob
    return max(probs, key=probs.get), probs

# Example: Confident=Yes, Studied=No, Sick=No
test_instance = {"Confident":"Yes","Studied":"No","Sick":"No"}
result, details = predict(test_instance)

print("Prediction:", result)
print("Probabilities:", details)