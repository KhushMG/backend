from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from app.services.recommendation import (
    load_model,
    get_latent_space,
    get_recommendations,
)
from app.services.fuzzy_search import fuzzy_match_anime
from app.services.features import vectorize_columns
from app.services.read_csv_file import read_csv
from pydantic import BaseModel
from scipy.sparse import hstack

# Global variables for lazy loading and caching
model = None
latent_space = None
anime_features = None
df = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, latent_space, anime_features, df

    # Initialize resources
    df = read_csv("data/anime_filtered.csv")
    df["synopsis"] = df["synopsis"].fillna("")
    genres_tfidf, synopsis_tfidf = vectorize_columns(df)
    anime_features = hstack([genres_tfidf, synopsis_tfidf])

    input_dim = anime_features.shape[1]
    model = load_model("app/anime_model.pth", input_dim)
    latent_space = get_latent_space(model, anime_features)

    # Run the application
    yield

    # Clean up resources if necessary
    model = None
    latent_space = None
    anime_features = None
    df = None


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://anime-rec-psi.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnimeRequest(BaseModel):
    anime_name: str


@app.get("/")
def read_root():
    return {"message": "Welcome to the Anime Recommender API"}


@app.get("/autocomplete/")
def autocomplete(query: str = Query(..., min_length=1)):
    suggestions = fuzzy_match_anime(df, query)
    return JSONResponse(content=suggestions)


@app.post("/recommendations/")
def recommend_anime(request: AnimeRequest):
    try:
        recommendations = get_recommendations(df, latent_space, request.anime_name)
        return JSONResponse(
            content=recommendations
        )
    except IndexError:
        raise HTTPException(status_code=404, detail="Anime not found")
