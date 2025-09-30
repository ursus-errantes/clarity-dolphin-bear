import json
import logging
from pathlib import Path
import hydra
from omegaconf import DictConfig
import pandas as pd
import torch

import torch.nn as nn
import torch.nn.functional as F

from clarity.utils.file_io import read_jsonl

logger = logging.getLogger(__name__)

class mlp_scalar_features(nn.Module):
    def __init__(self, num_scalar_features, hidden_sizes=[64, 32], k=3.0):
        """
        Args:
            num_scalar_features: number of scalar input features
            hidden_sizes: list of hidden layer sizes
            k: steepness factor for logistic curve (k > 1 outputs closer to 0 or 1)
        """
        super().__init__()
        
        layers = []
        in_features = num_scalar_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.ReLU())
            in_features = h
        
        self.mlp = nn.Sequential(*layers)
        self.fc_out = nn.Linear(in_features, 1)
        self.k = k

    def forward(self, x):
        """
        x: tensor of shape (batch_size, num_scalar_features)
        """
        h = self.mlp(x)
        logit = self.fc_out(h)
        out = torch.sigmoid(self.k * logit) # skew toward edges of [0, 1]
        return out
    

def load_features(cfg, split: str, system: str, feature: str | None) -> pd.DataFrame:
    """Load dataset and add prediction scores.

    Args:

        cfg (DictConfig): Configuration object.
        split (str): Dataset split to load ('train' or 'valid')
        system (str): System name ('stoi' or 'whisper' or 'features')
        feature (str): Specific feature name ('VAR (dB)' or other scalar feature)
    Returns:

        pd.DataFrame: DataFrame containing dataset records with added scores.
    """
    dataset_filename = (
        Path(cfg.data.cadenza_data_root)
        / cfg.data.dataset
        / "metadata"
        / f"{split}_metadata.json"
    )
    with dataset_filename.open("r", encoding="utf-8") as fp:
        records = json.load(fp)

    # Load STOI or Whisper scores and add them to the records
    system_path = f"{cfg.data.dataset}.{split}.{system}.jsonl"
    system_score = read_jsonl(str(system_path))
    system_score_index = {
        record["signal"]: record[system + (f" {feature}" if feature else "")] for record in system_score
    }
    for record in records:
        record[f"{system}"] = system_score_index[record["signal"]]

    return pd.DataFrame(records)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def run_train_model(cfg: DictConfig) -> None:
    """Train a model to predict the intelligibility score from scalar features."""

    logger.info(f"Training model on {cfg.split} set...")

    # Define model
    batch_size = 1
    num_scalar_features = 3
    model = mlp_scalar_features(num_scalar_features)

    # gather STOI score, whisper score, and VAR dB as input features from jsonl files
    stoi_df = load_features(cfg, "train", "stoi", None)
    whisper_df = load_features(cfg, "train", "whisper", None)
    features_df = load_features(cfg, "train", "features", "VAR (dB)")
    # merge dataframes on 'signal' column
    merged_df = stoi_df.merge(whisper_df[['signal', 'whisper']], on='signal')
    merged_df = merged_df.merge(features_df[['signal', 'features']], on='signal')
    print(merged_df.head())

    # create input tensor
    input_features = merged_df[['stoi', 'whisper', 'features']].values
    input_tensor = torch.tensor(input_features, dtype=torch.float32)
    # create labels tensor
    labels = merged_df['correctness'].values
    labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)  # shape: (num_samples, 1)
    # create dataloader
    dataset = torch.utils.data.TensorDataset(input_tensor, labels_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # define training loop
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10
    criterion = nn.MSELoss() 

    for epoch in range(num_epochs):
        for features, labels in dataloader:
            outputs = model(features)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
    
    # Save model
    model_path = f"{cfg.data.dataset}.{cfg.split}.mlp_scalar_features.pth"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")


@hydra.main(config_path="configs", config_name="config", version_base=None)
def run_inference(cfg: DictConfig) -> None:
    """Run inference using the trained model on the validation set."""
    model_path = f"{cfg.data.dataset}.{cfg.split}.mlp_scalar_features.pth"
    num_scalar_features = 3
    model = mlp_scalar_features(num_scalar_features)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # gather STOI score, whisper score, and VAR dB as input features from jsonl files
    stoi_df = load_features(cfg, "valid", "stoi", None)
    whisper_df = load_features(cfg, "valid", "whisper", None)
    features_df = load_features(cfg, "valid", "features", "VAR (dB)")
    # merge dataframes on 'signal' column
    merged_df = stoi_df.merge(whisper_df[['signal', 'whisper']], on='signal')
    merged_df = merged_df.merge(features_df[['signal', 'features']], on='signal')
    print(merged_df.head())
    # create input tensor
    input_features = merged_df[['stoi', 'whisper', 'features']].values
    validation_tensor = torch.tensor(input_features, dtype=torch.float32)

    # Run inference
    with torch.no_grad():
        outputs = model(validation_tensor)
        print(outputs)


if __name__ == "__main__":
    run_train_model()
    run_inference()
