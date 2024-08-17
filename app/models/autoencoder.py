import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from app.services.read_csv_file import read_csv
from app.services.features import vectorize_columns


# autoencoder can better understand abstract relationships and similarities or something like that
class AnimeAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=128):

        super(AnimeAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
        )

    def forward(self, x):
        latent_space = self.encoder(x)
        reconstructed = self.decoder(latent_space)
        return latent_space, reconstructed


def model_train_and_save(anime_features, df):
    input_dim = anime_features.shape[1]
    model = AnimeAutoencoder(input_dim)

    # init optimizer and loss function
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    # convert features to a PyTorch tensor
    X_train, X_test = train_test_split(anime_features, test_size=0.2, random_state=42)

    # training set
    X_train_tensor = torch.tensor(X_train.toarray(), dtype=torch.float32)

    # test set
    X_test_tensor = torch.tensor(X_test.toarray(), dtype=torch.float32)

    # Training Loop
    num_epochs = 15

    for epoch in range(num_epochs):
        model.train()

        # forward pass: get model predictions for training data
        latent_space_train, reconstructed_train = model(X_train_tensor)

        # compute loss between model's prediction and the true data
        train_loss = criterion(reconstructed_train, X_train_tensor)

        # backward pass: compute gradients
        optimizer.zero_grad()  # zero out gradients from previous steps
        train_loss.backward()  # backpropagate the loss

        # update model weights
        optimizer.step()  # perform one step of optimization (gradient descent)

        # Evaluation Phase

        model.eval()
        with torch.no_grad():
            latent_space_test, reconstructed_test = model(X_test_tensor)
            test_loss = criterion(reconstructed_test, X_test_tensor)

        # print training loss and test loss for current epoch
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss.item()}, Test Loss: {test_loss.item()}"
        )

    torch.save(model.state_dict(), "anime_model.pth")
    return model

if __name__ == "__main__":
    # Example data loading and preprocessing
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from scipy.sparse import hstack

    # Load your dataset
    df = read_csv("data/anime_filtered.csv")

    # Example preprocessing: Vectorize genres and synopsis
    vectorizer = TfidfVectorizer()
    genres_tfidf, synopsis_tfidf = vectorize_columns(df)

    # Combine features into a single matrix
    anime_features = hstack([genres_tfidf, synopsis_tfidf])

    # Train the model
    trained_model = model_train_and_save(anime_features, df)
