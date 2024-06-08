import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np
from sklearn.preprocessing import StandardScaler

# Generate sample data
X,y= make_blobs(n_samples=150, centers=3, n_features=4, random_state=42)
data = {'ID': range(1, 101),'spend': np.random.randint(1000, 5000, 100),'purchase': np.random.randint(1, 50, 100)}
df = pd.DataFrame(data)
print(df.head())

features = df[['spend', 'purchase']]
features_scaled =  StandardScaler().fit_transform(features)

# Choose the optimal number of clusters and perform k-Means clustering
k = int(input("Enter the optimal number of clusters : "))

# Apply K-means clustering
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(features_scaled)
df['Cluster'] = clusters

# Scatter plot of the clusters
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df['spend'], df['purchase'], c=df['Cluster'], cmap='viridis')
plt.xlabel('Annual Spend')
plt.ylabel('Purchase Frequency')
