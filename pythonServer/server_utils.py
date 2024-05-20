# imports
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from flask import  jsonify

def process_existing_profile(existing_profile, cursor, connection, new_profile_id, new_bio_vector):
        group_id = existing_profile[0]
        print(f"Group ID for existing profile: {group_id}")  # Debugging statement
        if group_id == 0:
            
            cursor.close()
            connection.close()
            return jsonify({"message": "No matches found yet"}), 200
        else:
            try:
                cursor.execute("SELECT profile_id, bio_vector FROM clusters WHERE group_id = %s AND profile_id != %s", 
                            (group_id, new_profile_id))
                existing_profiles = cursor.fetchall()
                print(f"Existing profiles in group: {existing_profiles}")  # Debugging statement
                if existing_profiles:
                    existing_vectors = np.array([json.loads(profile[1]) if isinstance(profile[1], str) else profile[1] for profile in existing_profiles])
                    profile_ids = [profile[0] for profile in existing_profiles]
                    similarities = cosine_similarity([new_bio_vector], existing_vectors)[0]
                    sorted_indices = np.argsort(similarities)[::-1][:5]
                    most_similar_profiles = [{"profile_id": profile_ids[i], "similarity": similarities[i]} for i in sorted_indices]
                else:
                    most_similar_profiles = []
            except Exception as e:
                print(f"Error processing existing profiles: {str(e)}")  # Debugging statement
                return jsonify({"error": f"Error processing existing profiles: {str(e)}"}), 500

            cursor.close()
            connection.close()

            return jsonify({
                "message": "Profile processed successfully",
                "group_id": int(group_id),  # Convert to native Python type
                "most_similar_profiles": most_similar_profiles
            })
            
            
    
    
def process_new_profile(new_profile_entry, load_and_process_data_with_new_profile, save_clusters):
   
    # Load and process data including the new profile
    df = load_and_process_data_with_new_profile(1000, new_profile_entry)
    save_clusters(df)

    # Find the new profile's group
    new_profile_group = df[df['id'] == new_profile_entry['id']]['group'].values[0]

    most_similar_profiles = []

    # If group is not 0, that means theres matches
    if new_profile_group != 0:
        group_profiles = df[df['group'] == new_profile_group]
        existing_vectors = np.array(group_profiles[group_profiles['id'] != new_profile_entry['id']]['bio_vector'].tolist())
        profile_ids = group_profiles[group_profiles['id'] != new_profile_entry['id']]['id'].tolist()
        new_bio_vector = np.array(group_profiles[group_profiles['id'] == new_profile_entry['id']]['bio_vector'].tolist())
        similarities = cosine_similarity(new_bio_vector, existing_vectors)[0]
        sorted_indices = np.argsort(similarities)[::-1][:5]
        most_similar_profiles = [{"profile_id": profile_ids[i], "similarity": similarities[i]} for i in sorted_indices]

    return jsonify({
        "message": "Profile processed successfully",
        "group_id": int(new_profile_group),  # Convert to native Python type
        "most_similar_profiles": most_similar_profiles if new_profile_group != 0 else "No matches found yet"
    })