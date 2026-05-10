# 🎬 Movie Recommendation System

A Machine Learning-based Movie Recommendation System built using Python, Scikit-learn, and Streamlit that recommends similar movies based on content similarity. The system analyzes movie genres, cast, keywords, and overview using vectorization and cosine similarity techniques to generate personalized recommendations.

---

## 🚀 Features

- 🎥 Recommends similar movies instantly
- 🧠 Uses Machine Learning for recommendation
- 📊 Content-based filtering approach
- 🎨 Interactive Streamlit web interface
- ⚡ Fast and responsive recommendation system
- 🌐 Deployable on Streamlit Community Cloud

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Cosine Similarity
- CountVectorizer
- Pickle

---

## 📂 Project Structure

movie_recommender/

├── app.py

├── movie_model.py

├── movie_list.pkl

├── similarity.pkl

├── movies.csv

├── credits.csv

├── requirements.txt

├── README.md

└── .gitignore

---

## 📊 Dataset Used

This project uses the TMDB 5000 Movie Dataset from Kaggle.

Dataset Link:
https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

Files used:
- tmdb_5000_movies.csv
- tmdb_5000_credits.csv

---

## ⚙️ How the System Works

1. Movie datasets are loaded and merged.
2. Important features such as genres, cast, keywords, and overview are combined into a single tags column.
3. Text vectorization converts movie data into numerical vectors.
4. Cosine similarity calculates similarity scores between movies.
5. When a user selects a movie, the system recommends the most similar movies.

---

## ▶️ How to Run the Project

### 1️⃣ Clone Repository

```bash
git clone https://github.com/kumarvl13/movie-recommendation-system.git
