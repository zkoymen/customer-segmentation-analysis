import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
from sklearn.cluster import KMeans

print("All libraries imported successfully.")

print("Downloading dataset from Kaggle Hub...")
# The download function returns the path to the downloaded dataset folder
path = kagglehub.dataset_download("vjchoudhary7/customer-segmentation-tutorial-in-python")
data_file = f"{path}/Mall_Customers.csv"
print(f"Dataset downloaded to: {path}")

df = pd.read_csv(data_file)

# First data-şnspection
print("\nFirst 5 rows of the dataset:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nStatistical summary of the dataset:")
print(df.describe())




# Starting Exploratory Data Analysis (EDA)
# First lets divide customers into specific classes


# Plotting distributions of Age, Annual Income, and Spending Score
plt.style.use('seaborn-v0_8-whitegrid')

plt.figure(figsize=(15, 5))

# Age distribution
plt.subplot(1, 3, 1)
sns.histplot(df['Age'], kde=True, bins=20, color='skyblue')
plt.title('Distribution of Age')

# Annual Income distribution
plt.subplot(1, 3, 2)
sns.histplot(df['Annual Income (k$)'], kde=True, bins=20, color='salmon')
plt.title('Distribution of Annual Income (k$)')

# Spending Score distribution
plt.subplot(1, 3, 3)
sns.histplot(df['Spending Score (1-100)'], kde=True, bins=20, color='lightgreen')
plt.title('Distribution of Spending Score (1-100)')

plt.suptitle('Customer Data Distributions', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# Scatter plot of Annual Income vs. Spending Score
plt.figure(figsize=(10, 7))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Gender')
plt.title('Annual Income vs. Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
