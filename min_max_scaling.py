def min_max_scaling(data, feature_range=(0, 1)):

    min_val = min(data)
    max_val = max(data)
    scale_min, scale_max = feature_range

    normalized = [
        ((x - min_val) / (max_val - min_val)) * (scale_max - scale_min) + scale_min
        for x in data
    ]
    return normalized

# Example usage
data = [10, 20, 30, 40, 50]
normalized_data = min_max_scaling(data, feature_range=(0, 1))

print("Original Data:", data)
print("Normalized Data:", normalized_data)

