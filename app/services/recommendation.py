import torch
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
from app.models.autoencoder import AnimeAutoencoder


def load_model(model_path, input_dim):
    model = AnimeAutoencoder(input_dim)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model

def get_latent_space(model, anime_features):
    with torch.no_grad():
        latent_space, _ = model(torch.tensor(anime_features.toarray(), dtype=torch.float32))
    return latent_space

def get_recommendations(df, latent_space, anime_name):
    anime_idx = df[df['Name'].str.contains(anime_name, case=False, na=False) | 
                   df['English name'].str.contains(anime_name, case=False, na=False)].index[0]

    distance_matrix = euclidean_distances(latent_space.numpy())
    similar_anime_indices = distance_matrix[anime_idx].argsort()[1:21]

    rec_anime = df.loc[similar_anime_indices, ["Name", "English name", "Score"]]
    similarity_scores = distance_matrix[anime_idx, similar_anime_indices]

    rec_anime["Similarity"] = (1 / (1 + similarity_scores)) * 100
    rec_anime["Similarity"] = rec_anime["Similarity"].apply(lambda x: f"{x:.2f}%")
    
    rec_anime_sorted = rec_anime.sort_values(by=["Similarity", "Score"], ascending=[True, False])

    return rec_anime_sorted.to_dict(orient="records")
