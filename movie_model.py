import pandas as pd
import numpy as np
import ast
import pickle
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# Load Datasets
# =========================================================

movies = pd.read_csv('movies.csv')
credits = pd.read_csv('credits.csv')

# =========================================================
# Merge Datasets
# =========================================================

movies = movies.merge(credits, on='title')

# =========================================================
# Select Required Columns
# =========================================================

movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# =========================================================
# Remove Missing Values
# =========================================================

movies.dropna(inplace=True)

# =========================================================
# Convert String Lists to Python Lists
# =========================================================

def convert(text):
    L = []

    for i in ast.literal_eval(text):
        L.append(i['name'])

    return L


movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

# =========================================================
# Fetch Top 3 Cast Members
# =========================================================

def convert_cast(text):
    L = []
    counter = 0

    for i in ast.literal_eval(text):

        if counter != 3:
            L.append(i['name'])
            counter += 1

        else:
            break

    return L


movies['cast'] = movies['cast'].apply(convert_cast)

# =========================================================
# Fetch Director Name
# =========================================================

def fetch_director(text):
    L = []

    for i in ast.literal_eval(text):

        if i['job'] == 'Director':
            L.append(i['name'])

    return L


movies['crew'] = movies['crew'].apply(fetch_director)

# =========================================================
# Convert Overview into List
# =========================================================

movies['overview'] = movies['overview'].apply(lambda x: x.split())

# =========================================================
# Remove Spaces Between Words
# =========================================================

movies['genres'] = movies['genres'].apply(
    lambda x: [i.replace(" ", "") for i in x]
)

movies['keywords'] = movies['keywords'].apply(
    lambda x: [i.replace(" ", "") for i in x]
)

movies['cast'] = movies['cast'].apply(
    lambda x: [i.replace(" ", "") for i in x]
)

movies['crew'] = movies['crew'].apply(
    lambda x: [i.replace(" ", "") for i in x]
)

# =========================================================
# Create Tags Column
# =========================================================

movies['tags'] = (
    movies['overview']
    + movies['genres']
    + movies['keywords']
    + movies['cast']
    + movies['crew']
)

# =========================================================
# Create New DataFrame
# =========================================================

new_df = movies[['movie_id', 'title', 'tags']]

# =========================================================
# Convert Tags List into String
# =========================================================

new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

# Convert to lowercase
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

# =========================================================
# Text Vectorization
# =========================================================

cv = CountVectorizer(max_features=5000, stop_words='english')

vectors = cv.fit_transform(new_df['tags']).toarray()

# =========================================================
# Calculate Cosine Similarity
# =========================================================

similarity = cosine_similarity(vectors)

# =========================================================
# Save PKL Files in Project Folder
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

movie_path = os.path.join(BASE_DIR, 'movie_list.pkl')
similarity_path = os.path.join(BASE_DIR, 'similarity.pkl')

pickle.dump(new_df.to_dict(), open(movie_path, 'wb'))
pickle.dump(similarity, open(similarity_path, 'wb'))

# =========================================================
# Success Message
# =========================================================

print("✅ movie_list.pkl created successfully!")
print("✅ similarity.pkl created successfully!")
print("🎬 Movie Recommendation Model Ready!")