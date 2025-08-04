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


# Selecting features for clustering
# We will use 'Annual Income' and 'Spending Score' for segmentation
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
print("Features selected: Annual Income and Spending Score.")



# WCSS = Within-Cluster Sum of Squares
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow Method graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.xticks(range(1, 11))
plt.show()

# It appears that optimal number  of cluster is about 5 or so. We will select 5


# ----------------- TRAINING SESSION -------------------

# Training the K-Means model with the optimal number of clusters

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X)

# Adding the cluster information to the original DataFrame
df['Customer Segment'] = y_kmeans

print("K-Means model trained and segments predicted.")
print("\nDataset with Customer Segments:")
print(df.head(10))


# -----------------VISUALITSATION -------------------------


plt.figure(figsize=(12, 8))
sns.scatterplot(x=X[y_kmeans == 0, 0], y=X[y_kmeans == 0, 1], s=100, label='Standard', palette='viridis')
sns.scatterplot(x=X[y_kmeans == 1, 0], y=X[y_kmeans == 1, 1], s=100, label='Careless')
sns.scatterplot(x=X[y_kmeans == 2, 0], y=X[y_kmeans == 2, 1], s=100, label='Target')
sns.scatterplot(x=X[y_kmeans == 3, 0], y=X[y_kmeans == 3, 1], s=100, label='Sensible')
sns.scatterplot(x=X[y_kmeans == 4, 0], y=X[y_kmeans == 4, 1], s=100, label='Careful')

# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', label='Centroids', marker='*')

plt.title('Customer Segments', fontsize=16)
plt.xlabel('Annual Income (k$)', fontsize=12)
plt.ylabel('Spending Score (1-100)', fontsize=12)
plt.legend()
plt.show()





