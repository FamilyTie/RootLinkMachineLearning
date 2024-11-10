from flask_cors import CORS
from flask import Flask, request, jsonify
import pandas as pd
from ML.utils import preprocess_bios
from ML.clustering import  save_clusters
from ML.clustering import get_database_connection
import numpy as np
import json
import psycopg2
from batch_process_profiles import load_and_process_data_with_new_profile
from server_utils import process_existing_profile, process_new_profile
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
from urllib.parse import urlparse

load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL')
if DATABASE_URL is None:
    raise Exception("DATABASE_URL environment variable not set")

BATCH_SIZE = 1000
print(DATABASE_URL)
def get_database_connection():
    try:
        conn = psycopg2.connect(DATABASE_URL)
        print("Database connection successful")
        return conn
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        raise e
    
app = Flask(__name__)
CORS(app)


global_vectorizer = SentenceTransformer('all-MiniLM-L6-v2')




@app.route('/model/process_user', methods=['POST'])
def process_user():
    try:
        connection = get_database_connection()
        cursor = connection.cursor()
        new_profile = request.json
        if not new_profile:
            return jsonify({"error": "Invalid input data"}), 400
        
        new_profile_bio = new_profile.get('bio')
        new_profile_ethnicity = new_profile.get('ethnicity')
        new_profile_adoption_year = new_profile.get('adoption_year')
        new_profile_id = new_profile.get('id')

        # Check if the profile already exists
        cursor.execute("SELECT group_id FROM clusters WHERE profile_id = %s", (new_profile_id,))
        existing_profile = cursor.fetchone()

        cleaned_bio = preprocess_bios([new_profile_bio])[0]
        # Convert the vector to a list of Python-native float types
        new_bio_vector = [float(x) for x in global_vectorizer.encode([cleaned_bio])[0]]

        # Processing existing profile
        if existing_profile:
            return process_existing_profile(existing_profile, cursor, connection, new_profile_id, new_bio_vector)
        
        # Processing new profile
        else:
            try:
                new_profile_entry = {
                    'id': new_profile_id,
                    'bio': new_profile_bio,
                    'ethnicity': new_profile_ethnicity,
                    'adoption_year': new_profile_adoption_year
                }
                return process_new_profile(new_profile_entry, load_and_process_data_with_new_profile, save_clusters)
            except Exception as e:
                print(f"Error processing new profile: {str(e)}")  # Debugging statement
                return jsonify({"error": f"Error processing new profile: {str(e)}"}), 500

    except Exception as e:
        print(f"General error: {str(e)}")  # Debugging statement
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)


