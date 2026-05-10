import streamlit as st
import pickle
import pandas as pd
import os

# =========================================================
# Load PKL Files
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

movie_list_path = os.path.join(BASE_DIR, 'movie_list.pkl')
similarity_path = os.path.join(BASE_DIR, 'similarity.pkl')

movies_dict = pickle.load(open(movie_list_path, 'rb'))
movies = pd.DataFrame(movies_dict)

similarity = pickle.load(open(similarity_path, 'rb'))

# =========================================================
# Recommendation Function
# =========================================================

def recommend(movie):

    movie_index = movies[movies['title'] == movie].index[0]

    distances = similarity[movie_index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []

    for i in movie_list:
        recommended_movies.append(movies.iloc[i[0]].title)

    return recommended_movies

# =========================================================
# Streamlit Page Configuration
# =========================================================

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="centered"
)

# =========================================================
# Custom Styling
# =========================================================

st.markdown(
    """
    <style>
    .main {
        background-color: #0E1117;
    }

    h1 {
        color: #FF4B4B;
        text-align: center;
    }

    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 18px;
    }

    .recommendation {
        background-color: #262730;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        color: white;
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# App Title
# =========================================================

st.title("🎬 Movie Recommendation System")

st.write(
    "Get movie recommendations based on your favorite movie using Machine Learning."
)

# =========================================================
# Movie Selection Dropdown
# =========================================================

selected_movie = st.selectbox(
    "Select a movie:",
    movies['title'].values
)

# =========================================================
# Recommendation Button
# =========================================================

if st.button("Recommend Movies"):

    recommendations = recommend(selected_movie)

    st.subheader("Recommended Movies")

    for movie in recommendations:

        st.markdown(
            f"""
            <div class="recommendation">
                👉 {movie}
            </div>
            """,
            unsafe_allow_html=True
        )

# =========================================================
# Footer
# =========================================================

st.markdown("---")

st.markdown(
    "<center>Built using Python, Machine Learning & Streamlit</center>",
    unsafe_allow_html=True
)