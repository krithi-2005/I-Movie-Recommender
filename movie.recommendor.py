import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    return movies, ratings

def content_based_recommend(movies, movie_title, top_n=5):
    movies = movies.copy()
    movies['genres'] = movies['genres'].fillna('')
    tfidf = TfidfVectorizer(token_pattern='[a-zA-Z0-9]+')
    tfidf_matrix = tfidf.fit_transform(movies['genres'])

    try:
        idx = movies[movies['title'].str.contains(movie_title, case=False, na=False)].index[0]
    except IndexError:
        print("Movie not found.")
        return

    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_indices = cosine_sim.argsort()[-top_n-1:-1][::-1]
    recommendations = movies.iloc[similar_indices][['title', 'genres']]
    print("\nTop Content-Based Recommendations:")
    print(recommendations.to_string(index=False))

def collaborative_recommend(ratings, user_id, top_n=5):
    user_movie_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    if user_id not in user_movie_matrix.index:
        print("User ID not found.")
        return
    similarity = cosine_similarity(user_movie_matrix)
    user_index = user_id - 1
    sim_scores = list(enumerate(similarity[user_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:]
    similar_users = [i[0] for i in sim_scores[:5]]
    movie_scores = user_movie_matrix.iloc[similar_users].mean(axis=0)
    user_seen = user_movie_matrix.iloc[user_index]
    movie_scores = movie_scores[user_seen == 0]
    recommendations = movie_scores.sort_values(ascending=False).head(top_n)
    print("\nTop Collaborative Recommendations (Movie IDs):")
    print(recommendations)

# Run the system
movies, ratings = load_data()
content_based_recommend(movies, "Toy Story")
collaborative_recommend(ratings,
user_id=1
