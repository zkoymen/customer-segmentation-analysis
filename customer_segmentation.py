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
