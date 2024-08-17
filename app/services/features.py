from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize_columns(anime_data):
  # vectorize 'Genres' column
  genres_vectorizer = TfidfVectorizer()
  genres_tfidf = genres_vectorizer.fit_transform(anime_data["Genres"])

  # vectorize 'synopsis' column
  synopsis_vectorizer = TfidfVectorizer(max_features=5000)
  synopsis_tfidf = synopsis_vectorizer.fit_transform(anime_data["synopsis"])

  return genres_tfidf, synopsis_tfidf
