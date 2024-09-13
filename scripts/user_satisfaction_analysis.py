import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from typing import Tuple, List

def calculate_euclidean_distance(point: np.ndarray, centroid: np.ndarray) -> float:
    return np.sqrt(np.sum((point - centroid)**2))

    
def assign_scores(engagement_data: pd.DataFrame, experience_data: pd.DataFrame, 
                  engagement_kmeans: KMeans, experience_kmeans: KMeans,
                  msisdn_column: pd.Series) -> pd.DataFrame:

    # Normalize data
    scaler = StandardScaler()
    engagement_normalized = scaler.fit_transform(engagement_data[['Sessions', 'Duration', 'Total Traffic']])
    experience_normalized = scaler.fit_transform(experience_data[['Avg TCP Retrans', 'Avg RTT', 'Avg Throughput']])
    
    # Find least engaged and worst experience centroids
    least_engaged_centroid = engagement_kmeans.cluster_centers_[np.argmin(np.sum(engagement_kmeans.cluster_centers_, axis=1))]
    worst_experience_centroid = experience_kmeans.cluster_centers_[np.argmin(np.sum(experience_kmeans.cluster_centers_, axis=1))]
    
    # Calculate scores
    engagement_scores = np.array([calculate_euclidean_distance(point, least_engaged_centroid) 
                                  for point in engagement_normalized])
    experience_scores = np.array([calculate_euclidean_distance(point, worst_experience_centroid) 
                                  for point in experience_normalized])
    
    # Create DataFrame with scores
    scores_data = pd.DataFrame({
        'MSISDN': msisdn_column.unique(),
        'Engagement Score': engagement_scores,
        'Experience Score': experience_scores
    })
    
    return scores_data

def calculate_satisfaction_scores(scores_data: pd.DataFrame) -> pd.DataFrame:
    scores_data['Satisfaction Score'] = (scores_data['Engagement Score'] + scores_data['Experience Score']) / 2
    top_10_satisfied = scores_data.nlargest(10, 'Satisfaction Score')
    return scores_data, top_10_satisfied

def build_regression_model(scores_data: pd.DataFrame) -> Tuple[LinearRegression, float, float]:
    X = scores_data[['Engagement Score', 'Experience Score']]
    y = scores_data['Satisfaction Score']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, mse, r2

def cluster_satisfaction(scores_data: pd.DataFrame) -> Tuple[pd.DataFrame, List[float], List[float]]:
    X = scores_data[['Engagement Score', 'Experience Score']]
    kmeans = KMeans(n_clusters=2, random_state=42)
    scores_data['Cluster'] = kmeans.fit_predict(X)
    
    avg_satisfaction = scores_data.groupby('Cluster')['Satisfaction Score'].mean().tolist()
    avg_experience = scores_data.groupby('Cluster')['Experience Score'].mean().tolist()
    
    return scores_data, avg_satisfaction, avg_experience

def plot_clusters(scores_data: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(scores_data['Engagement Score'], scores_data['Experience Score'], 
                         c=scores_data['Cluster'], cmap='viridis')
    ax.set_xlabel('Engagement Score')
    ax.set_ylabel('Experience Score')
    ax.set_title('User Clusters based on Engagement and Experience')
    plt.colorbar(scatter)
    return fig