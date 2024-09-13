import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

def aggregate_metrics_per_customer(data_cleaned: pd.DataFrame) -> pd.DataFrame:
    aggregated = data_cleaned.groupby('MSISDN/Number').agg({
        'Bearer Id': 'count',  # sessions frequency
        'Dur. (ms)': 'sum',    # total duration
        'Total DL (Bytes)': 'sum',
        'Total UL (Bytes)': 'sum'
    }).reset_index()
    
    aggregated['Total Traffic'] = aggregated['Total DL (Bytes)'] + aggregated['Total UL (Bytes)']
    aggregated.columns = ['MSISDN', 'Sessions', 'Duration', 'DL Traffic', 'UL Traffic', 'Total Traffic']
    
    top_10 = {
        'Sessions': aggregated.nlargest(10, 'Sessions'),
        'Duration': aggregated.nlargest(10, 'Duration'),
        'Total Traffic': aggregated.nlargest(10, 'Total Traffic')
    }
    
    return aggregated, top_10

def normalize_and_cluster(data_cleaned: pd.DataFrame, k: int = 3) -> Tuple[pd.DataFrame, KMeans]:
    scaler = StandardScaler()
    normalized_data_cleaned = pd.DataFrame(scaler.fit_transform(data_cleaned[['Sessions', 'Duration', 'Total Traffic']]),
                                 columns=['Sessions', 'Duration', 'Total Traffic'])
    
    kmeans = KMeans(n_clusters=k, random_state=42)
    normalized_data_cleaned['Cluster'] = kmeans.fit_predict(normalized_data_cleaned)
    
    return normalized_data_cleaned, kmeans

def compute_cluster_stats(data_cleaned: pd.DataFrame, normalized_data_cleaned: pd.DataFrame) -> pd.DataFrame:
    data_cleaned['Cluster'] = normalized_data_cleaned['Cluster']
    return data_cleaned.groupby('Cluster').agg({
        'Sessions': ['min', 'max', 'mean', 'sum'],
        'Duration': ['min', 'max', 'mean', 'sum'],
        'Total Traffic': ['min', 'max', 'mean', 'sum']
    })

def aggregate_traffic_per_app(data_cleaned: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    apps = ['Social Media', 'Youtube', 'Netflix', 'Google', 'Email', 'Gaming', 'Other']
    top_10_per_app = {}
    
    for app in apps:
        data_cleaned[f'{app} Total'] = data_cleaned[f'{app} DL (Bytes)'] + data_cleaned[f'{app} UL (Bytes)']
        top_10 = data_cleaned.groupby('MSISDN/Number')[f'{app} Total'].sum().nlargest(10).reset_index()
        top_10.columns = ['MSISDN', f'{app} Total Traffic']
        top_10_per_app[app] = top_10
    
    return top_10_per_app

def plot_top_apps(data_cleaned: pd.DataFrame) -> plt.Figure:
    apps = ['Social Media', 'Youtube', 'Netflix', 'Google', 'Email', 'Gaming', 'Other']
    total_traffic = [data_cleaned[f'{app} DL (Bytes)'].sum() + data_cleaned[f'{app} UL (Bytes)'].sum() for app in apps]
    top_3_apps = sorted(zip(apps, total_traffic), key=lambda x: x[1], reverse=True)[:3]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar([app[0] for app in top_3_apps], [app[1] for app in top_3_apps])
    ax.set_title('Top 3 Most Used Applications')
    ax.set_xlabel('Application')
    ax.set_ylabel('Total Traffic (Bytes)')
    
    return fig

def elbow_method(data_cleaned: pd.DataFrame, max_k: int = 10) -> plt.Figure:
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data_cleaned[['Sessions', 'Duration', 'Total Traffic']])
    
    inertias = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(normalized_data)
        inertias.append(kmeans.inertia_)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, max_k + 1), inertias, marker='o')
    ax.set_title('Elbow Method for Optimal k')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Inertia')
    

    return fig