#1.2 Standard Normalization
def standard_scaler(data):

    mean_val = sum(data) / len(data)
    std_dev = (sum((x - mean_val) ** 2 for x in data) / len(data)) ** 0.5

    standardized = [(x - mean_val) / std_dev for x in data]
    return standardized


# Example usage
data = [10, 20, 30, 40, 50]
standardized_data = standard_scaler(data)

print("Original Data:", data)
print("Standardized Data:", standardized_data)