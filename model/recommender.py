import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ================= LOAD DATA =================
movies = pd.read_csv("data/movies_metadata.csv", low_memory=False)
ratings = pd.read_csv("data/ratings.csv")
credits = pd.read_csv("data/credits.csv")

movies = movies[['id', 'title', 'overview', 'vote_average', 'vote_count']]
movies['id'] = pd.to_numeric(movies['id'], errors='coerce')
movies.dropna(subset=['id', 'title'], inplace=True)

movies['overview'] = movies['overview'].fillna("")

credits['id'] = pd.to_numeric(credits['id'], errors='coerce')

# ================= POPULARITY SCORE =================
C = movies['vote_average'].mean()
m = movies['vote_count'].quantile(0.90)

def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v / (v + m) * R) + (m / (m + v) * C)

movies['popularity_score'] = movies.apply(weighted_rating, axis=1)

# ================= TF-IDF (NO FULL MATRIX) =================
tfidf = TfidfVectorizer(
    stop_words='english',
    max_features=5000
)

tfidf_matrix = tfidf.fit_transform(movies['overview'])

# ================= HELPERS =================
def get_cast_and_director(movie_id):
    row = credits[credits['id'] == movie_id]
    if row.empty:
        return [], "Unknown"

    row = row.iloc[0]
    cast = [c['name'] for c in ast.literal_eval(row['cast'])[:5]]
    crew = ast.literal_eval(row['crew'])
    director = next((c['name'] for c in crew if c['job'] == 'Director'), "Unknown")

    return cast, director

# ================= POPULAR MOVIES =================
def get_popular_movies(n=50):
    popular = movies.sort_values(
        "popularity_score", ascending=False
    ).head(n)

    return popular[['title', 'vote_average', 'vote_count']].round(1)

# ================= MOVIE DETAILS =================
def get_movie_details(title):
    movie = movies[movies['title'] == title]
    if movie.empty:
        return None

    movie = movie.iloc[0]
    cast, director = get_cast_and_director(movie['id'])

    return {
        "title": movie['title'],
        "rating": round(movie['vote_average'], 1),
        "votes": int(movie['vote_count']),
        "director": director,
        "cast": cast,
        "overview": movie['overview']
    }

# ================= TRUE HYBRID RECOMMENDER =================
def hybrid_recommend(title, n=10):
    if title not in movies['title'].values:
        return pd.DataFrame()

    # Index of searched movie
    idx = movies[movies['title'] == title].index[0]

    # 🔥 KEY FIX: compute similarity ONLY for ONE movie
    cosine_scores = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    # Top similar movies
    sim_indices = cosine_scores.argsort()[::-1][1:50]

    recs = movies.iloc[sim_indices].copy()

    # Hybrid score = content + popularity
    recs['hybrid_score'] = (
        cosine_scores[sim_indices] * 0.6 +
        recs['popularity_score'] * 0.4
    )

    return recs.sort_values(
        'hybrid_score', ascending=False
    ).head(n)[['title', 'vote_average', 'vote_count']].round(1)
