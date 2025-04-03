import pandas as pd
from sklearn.utils import shuffle

# Example: Loading a dataset
# Replace this with your actual dataset loading code
df = pd.read_csv('phishing_site_urls.csv')

# Print the first few rows of the dataframe to check if it's loaded correctly
print(df.head())

# Verify column names
print(df.columns)

# Check for the presence of 'URL' and 'Label' columns
if 'URL' not in df.columns or 'Label' not in df.columns:
    raise ValueError("The dataset must contain 'URL' and 'Label' columns")

# Shuffle the dataset
df = shuffle(df, random_state=42)

# Separate the dataset into two classes
class_0 = df[df['Label'] == 'good']
class_1 = df[df['Label'] == 'bad']

# Print the number of samples in each class
print(f"Number of samples in class 0: {len(class_0)}")
print(f"Number of samples in class 1: {len(class_1)}")

# Calculate the number of features to select from each class based on the minimum available samples
min_samples = min(len(class_0), len(class_1))
num_features_per_class = min(7000 // 2, min_samples)

# Print the number of features to be sampled from each class
print(f"Sampling {num_features_per_class} features from each class.")

# Randomly sample the required number of features from each class
if num_features_per_class > 0:
    selected_class_0 = class_0.sample(n=num_features_per_class, random_state=42)
    selected_class_1 = class_1.sample(n=num_features_per_class, random_state=42)

    # Combine the selected samples from both classes
    selected_df = pd.concat([selected_class_0, selected_class_1])

    # Shuffle the final dataset to mix the classes
    selected_df = shuffle(selected_df, random_state=42)

    # Reset index
    selected_df.reset_index(drop=True, inplace=True)

    # Save or use the selected dataset
    selected_df.to_csv('phishing_dataset.csv', index=False)

    print(selected_df.head())
else:
    print("Not enough samples in one or both classes to select the desired number of features.")
