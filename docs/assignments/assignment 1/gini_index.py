import pandas as pd
import os

# Define the dataset
data = {
    'A1': ['T', 'T', 'T', 'F', 'F', 'F', 'F', 'T', 'F'],
    'A2': ['T', 'T', 'T', 'F', 'F', 'F', 'F', 'T', 'F'],
    'Target Class': ['+', '+', '-', '+', '-', '-', '-', '+', '-']
}

df = pd.DataFrame(data)


def gini(attribute, target_class):
    unique_values = attribute.unique()
    gini_sum = 0
    total_count = len(target_class)

    for value in unique_values:
        subset = target_class[attribute == value]
        count = len(subset)
        positive_prob = (subset == '+').sum() / count
        negative_prob = (subset == '-').sum() / count
        gini_value = 1 - (positive_prob**2 + negative_prob**2)
        gini_sum += (count / total_count) * gini_value

    return gini_sum


# Calculate Gini index for each attribute
gini_A1 = gini(df['A1'], df['Target Class'])
gini_A2 = gini(df['A2'], df['Target Class'])

# Determine the best split
best_attribute = 'A1' if gini_A1 < gini_A2 else 'A2'

# Save the output as a CSV table
output_data = {
    'Attribute': ['A1', 'A2', 'Best Split'],
    'Gini Index': [gini_A1, gini_A2, best_attribute]
}
output_df = pd.DataFrame(output_data)

# Create the output directory if it doesn't exist
output_dir = './docs/assignments/assignment 1/output'
os.makedirs(output_dir, exist_ok=True)

# Save the DataFrame to a CSV file
output_df.to_csv(os.path.join(output_dir, 'gini_index.csv'), index=False)

print(f"Gini(A1) = {gini_A1:.3f}")
print(f"Gini(A2) = {gini_A2:.3f}")
print(f"The best split is {best_attribute}.")
