import psycopg2
import pandas as pd
from ML.utils import preprocess_bios
from ML.clustering import clear_cluster_table, save_clusters
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
from urllib.parse import urlparse

load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL')

BATCH_SIZE = 1000

def get_database_connection():
    return psycopg2.connect(DATABASE_URL)
    

def fetch_data_in_batches(batch_size):
    connection = get_database_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT id, data, bio FROM profiles")
    all_data = []
    while True:
        batch = cursor.fetchmany(batch_size)
        if not batch:
            break
        for row in batch:
            id = row[0]
            data = row[1]
            bio = row[2]
            ethnicity = data['raw']['ethnicity']
            adoption_year = data['raw']['adoptionYear']
            all_data.append((id, bio, ethnicity, adoption_year))
    cursor.close()
    connection.close()
    return all_data

def load_and_process_data(batch_size):
    all_data = fetch_data_in_batches(batch_size)
    df = pd.DataFrame(all_data, columns=['id', 'bio', 'ethnicity', 'adoption_year'])
    
    cleaned_bios = preprocess_bios(df['bio'])
    df['cleaned_bio'] = cleaned_bios
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    bio_vectors = model.encode(cleaned_bios.tolist())
    
    df['bio_vector'] = list(bio_vectors)

    # Group by ethnicity and adoption year
    df['group'] = df.groupby(['ethnicity', 'adoption_year']).ngroup() + 1  # Start groups from 1
    df['group_size'] = df.groupby('group')['group'].transform('count')  # Get the size of each group

    # Assign group 0 to those without matches
    df['group'] = df.apply(lambda x: 0 if x['group_size'] == 1 else x['group'], axis=1)

    df.drop(columns=['group_size'], inplace=True)  # Clean up temporary column
    
    return df

def load_and_process_data_with_new_profile(batch_size, new_profile):
    all_data = fetch_data_in_batches(batch_size)
    df = pd.DataFrame(all_data, columns=['id', 'bio', 'ethnicity', 'adoption_year'])
    
    new_profile_df = pd.DataFrame([new_profile], columns=['id', 'bio', 'ethnicity', 'adoption_year'])
    df = pd.concat([df, new_profile_df], ignore_index=True)

    cleaned_bios = preprocess_bios(df['bio'])
    df['cleaned_bio'] = cleaned_bios
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    bio_vectors = model.encode(cleaned_bios.tolist())
    
    df['bio_vector'] = list(bio_vectors)

    # Group by ethnicity and adoption year
    df['group'] = df.groupby(['ethnicity', 'adoption_year']).ngroup() + 1  # Start groups from 1
    df['group_size'] = df.groupby('group')['group'].transform('count')  # Get the size of each group

    # Assign group 0 to those without matches
    df['group'] = df.apply(lambda x: 0 if x['group_size'] == 1 else x['group'], axis=1)

    df.drop(columns=['group_size'], inplace=True)  # Clean up temporary column
    
    return df




if __name__ == "__main__":
    # Load and process data
    print(DATABASE_URL)