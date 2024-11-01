import psycopg2
import numpy as np
import json
from sklearn.cluster import KMeans
from dotenv import load_dotenv
import os
from urllib.parse import urlparse

load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL')
BATCH_SIZE = 1000

def get_database_connection():
    return psycopg2.connect(DATABASE_URL)

def fetch_clusters():
    connection = get_database_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT profile_id, cluster_id FROM clusters")
    clusters_data = cursor.fetchall()
    cursor.close()
    connection.close()
    return clusters_data


def clear_cluster_table():
    connection = get_database_connection()
    cursor = connection.cursor()
    cursor.execute("DELETE FROM clusters")
    connection.commit()
    cursor.close()
    connection.close()

def save_clusters(df):
    connection = get_database_connection()
    cursor = connection.cursor()
    
    cursor.execute("DELETE FROM clusters")  # Clear existing clusters data
    
    for index, row in df.iterrows():
        bio_vector = row['bio_vector'].tolist()
        group_id = row['group']
        
        cursor.execute(
            """
            INSERT INTO clusters (profile_id, ethnicity, adoption_year, bio_vector, group_id)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (int(row['id']), row['ethnicity'], row['adoption_year'], json.dumps(bio_vector), int(group_id))
        )
    
    connection.commit()
    cursor.close()
    connection.close()
    
 
        

def fetch_clusters_and_profiles():
    connection = get_database_connection()
    cursor = connection.cursor()
    cursor.execute("""
        SELECT c.profile_id, c.cluster_id, c.bio_vector, c.feature_vector, p.bio, p.data
        FROM clusters c
        JOIN profiles p ON c.profile_id = p.id
    """)
    clusters_data = cursor.fetchall()
    cursor.close()
    connection.close()
    return clusters_data
