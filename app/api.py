from fastapi import FastAPI, Query, HTTPException
from app.services.recommendation import load_model, get_latent_space, get_recommendations
from app.services.fuzzy_search import fuzzy_match_anime
from app.services.features import vectorize_columns
from app.services.read_csv_file import read_csv
import pandas as pd
from scipy.sparse import hstack
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from app.models.autoencoder import model_train_and_save


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to be more specific if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class AnimeRequest(BaseModel):
    anime_name: str


df = read_csv("./data/anime_filtered.csv")
# df["synopsis"].fillna("", inplace=True)
df["synopsis"] = df["synopsis"].fillna("")

genres_tfidf, synopsis_tfidf = vectorize_columns(df)

anime_features = hstack([genres_tfidf, synopsis_tfidf])
input_dim = anime_features.shape[1]
# model_train_and_save(anime_features, df)

model = load_model("app/anime_model.pth", input_dim)
latent_space = get_latent_space(model, anime_features)


@app.get("/")
def read_root():
    return {"message": "Welcome to the Anime Recommender API"}

@app.get("/autocomplete/")
def autocomplete(query: str = Query(..., min_length=1)):
  suggestions = fuzzy_match_anime(df, query)
  return suggestions

@app.post("/recommendations/")
def recommend_anime(request : AnimeRequest):
  try:
    recommendations = get_recommendations(df, latent_space, request.anime_name)
    return recommendations
  except IndexError:
    raise HTTPException(status_code=404, detail="Anime not found")
