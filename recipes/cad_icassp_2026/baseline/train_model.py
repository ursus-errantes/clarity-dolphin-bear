import json
import logging
from pathlib import Path
import hydra
from omegaconf import DictConfig
import pandas as pd
import torch
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F

from clarity.utils.file_io import read_jsonl

logger = logging.getLogger(__name__)

class mlp_scalar_features(nn.Module):
    def __init__(self, num_scalar_features, hidden_sizes=[64, 64, 32], k=3.0):
        """
        Args:
            num_scalar_features: number of scalar input features
            hidden_sizes: list of hidden layer sizes
            k: steepness factor for sigmoid (k > 1 outputs closer to 0 or 1)
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
    batch_size = 32
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

    # define parameters for training
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 150
    criterion = nn.MSELoss()
    patience = 10
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    train_losses = []
    val_losses = []

    # Split train/val (simple split)
    val_split = 0.1
    num_samples = len(dataset)
    num_val = int(num_samples * val_split)
    num_train = num_samples - num_val
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [num_train, num_val])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    logger.info(f"Training model with parameters:\n batch size {batch_size}, optimizer Adam, loss MSE, num_epochs {num_epochs}, early stopping patience {patience}")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
        train_loss /= num_train
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_outputs = model(val_x)
                val_loss += criterion(val_outputs, val_y).item() * val_x.size(0)
        val_loss /= num_val
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break

    # Plot losses
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.savefig(f"{cfg.data.dataset}.train.mlp_scalar_features.loss_curve.png")
    plt.close()
    
    # Save model
    model_path = f"{cfg.data.dataset}.train.mlp_scalar_features.pth"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")

    # Log a model summary and final loss
    logger.info(model)
    logger.info(f"Final training loss: {train_losses[-1]:.4f}, final validation loss: {val_losses[-1]:.4f}")


@hydra.main(config_path="configs", config_name="config", version_base=None)
def run_inference(cfg: DictConfig) -> None:
    """Run inference using the trained model on the validation set."""
    model_path = f"{cfg.data.dataset}.train.mlp_scalar_features.pth"
    logger.info(f"Running inference using model from {model_path}...")

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
        # save outputs to csv
        merged_df['predicted_correctness'] = outputs.numpy()
        output_csv_path = f"{cfg.data.dataset}.valid.mlp_scalar_features.inference.csv"
        merged_df.to_csv(output_csv_path, index=False)
        logger.info(f"Inference results saved to {output_csv_path}")


if __name__ == "__main__":
    run_train_model()
    run_inference()
