import pandas as pd

# Load the training dataset
train_df = pd.read_csv('train.csv')

# Calculate and print the percentage of missing values in each column
missing_percentage = (train_df.isnull().sum() / len(train_df)) * 100
print("\nPercentage of missing values in each column:")
print(missing_percentage.sort_values(ascending=False))

# Optional: Visualize missing values (requires seaborn)
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.heatmap(train_df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values Heatmap")
plt.show()