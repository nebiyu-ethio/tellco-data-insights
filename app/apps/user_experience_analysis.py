import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import streamlit as st
import psycopg2

def init_connection():
    return psycopg2.connect(**st.secrets["postgres"])

conn = init_connection()

def app():
    # Query the data
    query = 'SELECT * FROM xdr_data'
    data = pd.read_sql(query)

    query = 'SELECT * FROM xdr_data_cleaned'
    data_cleaned = pd.read_sql(query)

    # Display the data
    st.title("User Experience Analysis")
    st.subheader("Raw Data")
    st.write(data.head())
    
    st.subheader("Cleaned Data")
    st.write(data_cleaned.head())

    # Task 3 - Experience Analytics
    st.header("Task 3: Experience Analytics")

    # Task 3.1: Aggregate metrics per customer
    aggregated_data = data_cleaned.groupby('IMSI').agg({
        'TCP DL Retrans. Vol (Bytes)': 'mean',
        'TCP UL Retrans. Vol (Bytes)': 'mean',
        'Avg RTT DL (ms)': 'mean',
        'Avg RTT UL (ms)': 'mean',
        'Avg Bearer TP DL (kbps)': 'mean',
        'Avg Bearer TP UL (kbps)': 'mean',
        'Handset Type': lambda x: x.mode().iloc[0] if not x.mode().empty else None
    }).reset_index()

    # Calculate total TCP retransmission, RTT, and throughput
    aggregated_data['Total_TCP'] = (aggregated_data['TCP DL Retrans. Vol (Bytes)'] +
                                      aggregated_data['TCP UL Retrans. Vol (Bytes)'])
    aggregated_data['Total_RTT'] = (aggregated_data['Avg RTT DL (ms)'] +
                                      aggregated_data['Avg RTT UL (ms)'])
    aggregated_data['Total_Throughput'] = (aggregated_data['Avg Bearer TP DL (kbps)'] +
                                             aggregated_data['Avg Bearer TP UL (kbps)'])

    aggregated_data = aggregated_data.drop(columns=[
        'TCP DL Retrans. Vol (Bytes)',
        'TCP UL Retrans. Vol (Bytes)',
        'Avg RTT DL (ms)',
        'Avg RTT UL (ms)',
        'Avg Bearer TP DL (kbps)',
        'Avg Bearer TP UL (kbps)'
    ])

    st.subheader("Aggregated Data")
    st.write(aggregated_data.head())

    # Task 3.2: Compute and list top, bottom, and most frequent
    def compute_top_bottom_frequent(data, column, top_n=10):
        top_values = data[column].nlargest(top_n).reset_index(drop=True)
        bottom_values = data[column].nsmallest(top_n).reset_index(drop=True)
        most_frequent_values = data[column].mode()
        
        if len(most_frequent_values) > top_n:
            most_frequent_values = most_frequent_values.head(top_n)
        
        return {
            'top': top_values,
            'bottom': bottom_values,
            'most_frequent': most_frequent_values
        }

    # Compute statistics for each metric
    tcp_stats = compute_top_bottom_frequent(aggregated_data, 'Total_TCP')
    rtt_stats = compute_top_bottom_frequent(aggregated_data, 'Total_RTT')
    throughput_stats = compute_top_bottom_frequent(aggregated_data, 'Total_Throughput')

    # Display results
    st.subheader("Statistics for Total TCP")
    st.write("Top 10:", tcp_stats['top'])
    st.write("Bottom 10:", tcp_stats['bottom'])
    st.write("Most Frequent:", tcp_stats['most_frequent'])

    st.subheader("Statistics for Total RTT")
    st.write("Top 10:", rtt_stats['top'])
    st.write("Bottom 10:", rtt_stats['bottom'])
    st.write("Most Frequent:", rtt_stats['most_frequent'])

    st.subheader("Statistics for Total Throughput")
    st.write("Top 10:", throughput_stats['top'])
    st.write("Bottom 10:", throughput_stats['bottom'])
    st.write("Most Frequent:", throughput_stats['most_frequent'])

    # Task 3.3: Distribution of the average throughput per handset type
    throughput_distribution = aggregated_data.groupby('Handset Type')['Total_Throughput'].mean().sort_values(ascending=False).head(10)
    st.subheader("Top 10 Average Throughput per Handset Type")
    st.write(throughput_distribution)

    # Average TCP retransmission view per handset type
    tcp_distribution = aggregated_data.groupby('Handset Type')['Total_TCP'].mean().sort_values(ascending=False).head(10)
    st.subheader("Top 10 Average TCP Retransmission per Handset Type")
    st.write(tcp_distribution)

    # Task 3.4: Perform k-means clustering
    features = aggregated_data[['Total_TCP', 'Total_RTT', 'Total_Throughput']]
    kmeans = KMeans(n_clusters=3, random_state=0)
    kmeans.fit(features)
    aggregated_data['cluster'] = kmeans.labels_

    st.subheader("Cluster Descriptions")
    cluster_description = aggregated_data.groupby('cluster').mean()
    st.write(cluster_description)

# Note: Ensure to call this function in your main app script.