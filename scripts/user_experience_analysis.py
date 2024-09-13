import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

def aggregate_customer_data(data_cleaned):
    # Group by MSISDN/Number (customer ID)
    grouped = data_cleaned.groupby('MSISDN/Number')
    
    aggregated = grouped.agg({
        'TCP DL Retrans. Vol (Bytes)': 'mean',
        'TCP UL Retrans. Vol (Bytes)': 'mean',
        'Avg RTT DL (ms)': 'mean',
        'Avg RTT UL (ms)': 'mean',
        'Handset Type': lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,
        'Avg Bearer TP DL (kbps)': 'mean',
        'Avg Bearer TP UL (kbps)': 'mean'
    })
    
    # Rename columns for clarity
    aggregated.columns = [
        'Avg TCP Retrans DL', 'Avg TCP Retrans UL', 'Avg RTT DL', 'Avg RTT UL',
        'Handset Type', 'Avg Throughput DL', 'Avg Throughput UL'
    ]
    
    # Handle missing values and outliers
    for col in aggregated.columns:
        if col != 'Handset Type':
            aggregated[col] = aggregated[col].fillna(aggregated[col].mean())
            
    # Calculate average TCP retransmission
    aggregated['Avg TCP Retrans'] = (aggregated['Avg TCP Retrans DL'] + aggregated['Avg TCP Retrans UL']) / 2
    
    # Calculate average RTT
    aggregated['Avg RTT'] = (aggregated['Avg RTT DL'] + aggregated['Avg RTT UL']) / 2
    
    # Calculate average throughput
    aggregated['Avg Throughput'] = (aggregated['Avg Throughput DL'] + aggregated['Avg Throughput UL']) / 2
    
    return aggregated

def get_top_bottom_frequent(data_cleaned, column, n=10):
    top_n = data_cleaned[column].nlargest(n).tolist()
    bottom_n = data_cleaned[column].nsmallest(n).tolist()
    most_frequent = data_cleaned[column].value_counts().nlargest(n).index.tolist()
    
    return top_n, bottom_n, most_frequent

def plot_distribution(data_cleaned, x, y, title):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=x, y=y, data=data_cleaned)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()

def perform_kmeans_clustering(data_cleaned, k=3):
    # Select features for clustering
    features = ['Avg TCP Retrans', 'Avg RTT', 'Avg Throughput']
    X = data_cleaned[features]
    
    # Normalize the features
    X_normalized = (X - X.mean()) / X.std()
    
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    data_cleaned['Cluster'] = kmeans.fit_predict(X_normalized)
    
    return data_cleaned, kmeans.cluster_centers_

def describe_clusters(data_cleaned, cluster_centers, features):
    descriptions = []
    for i, center in enumerate(cluster_centers):
        cluster_data = data_cleaned[data_cleaned['Cluster'] == i]
        desc = f"Cluster {i} ({len(cluster_data)} customers):\n"
        for j, feature in enumerate(features):
            desc += f"  - {feature}: {center[j]:.2f} (Avg: {cluster_data[feature].mean():.2f})\n"
        descriptions.append(desc)
    return descriptions