import json
import logging
import os
from pathlib import Path
import hydra
from omegaconf import DictConfig
import pandas as pd
import torch
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from clarity.utils.file_io import read_jsonl

logger = logging.getLogger(__name__)


class temporal_attention_pool(nn.Module):
    """Learnable attention pooling over time dimension. Works for both 1d and 2d feature inputs."""
    def __init__(self, input_dim):
        super().__init__()
        # Linear attention layer maps each time step feature vector to a scalar score
        self.attention = nn.Linear(input_dim, 1)

    def forward(self, x, mask=None):
        """
        x: tensor of shape (batch_size, input_dim, time_steps)
        mask: tensor of shape (batch_size, time_steps) indicating valid elements
        mask is currently not used but will be needed for batching variable-length inputs
        """
        # Compute attention scores
        attn_scores = self.attention(x.permute(0, 2, 1))  # shape: (batch_size, time_steps, input_dim) -> (batch_size, time_steps, 1)
        attn_weights = F.softmax(attn_scores, dim=1)  # shape: (batch_size, time_steps, 1)

        # Weighted sum of inputs
        pooled = torch.sum(attn_weights * x.permute(0, 2, 1), dim=1)  # shape: (batch_size, input_dim)
        return pooled
    

class multimodal_conv_mlp(nn.Module):
    """Combines MLP for scalar features with CNN and attention pooling for 1d and 2d features."""
    def __init__(self, c1_in, c2_in, scalar_dim, k=3.0, p_dropout=0.3):
        """
        Args:
            c1_in: number of input channels for 1D CNN
            c2_in: number of input channels for 2D CNN
            scalar_dim: dimensionality of scalar features
            k: steepness factor for sigmoid (k > 1 outputs closer to 0 or 1)
            p_dropout: dropout probability for final MLP
        """
        super().__init__()

        # testing with 2d mfccs, not using 1d features for now
        # # 1D CNN encoder for temporal features
        # self.encoder_1d = nn.Sequential(
        #     nn.Conv1d(c1_in, 64, kernel_size=5, padding=2),
        #     nn.ReLU(),
        #     nn.Conv1d(64, 64, kernel_size=3, padding=1),
        #     nn.ReLU(),
        # )
        # self.attn_pool_1d = temporal_attention_pool(64)

        # 2D CNN encoder for spectro-temporal features
        self.encoder_2d = nn.Sequential(
            nn.Conv2d(c2_in, 64, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
        )
        # projection after frequency pooling
        self.projection_2d = nn.Conv1d(64, 64, kernel_size=1) # learn linear weights for each channel per time step
        self.attn_pool_2d = temporal_attention_pool(64)

        # simple MLP for scalar features
        self.mlp_scalar = nn.Sequential(
            nn.Linear(scalar_dim, 32),
            nn.ReLU(),
        )

        # final MLP concatenating the summaries of the three feature types
        self.final_mlp = nn.Sequential(
            nn.Linear(64 + 32, 64),
            # nn.BatchNorm1d(64), # using batch size 1 for testing, so batchnorm not appropriate
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(64, 1),
        )
        # sigmoid steepness factor
        self.k = k

    def forward(self, x1d, x2d, x_scalar, mask=None):
        """
        x1d: tensor of shape (batch_size, c1_in, time_steps)
        x2d: tensor of shape (batch_size, c2_in, freq_bins, time_steps)
        x_scalar: tensor of shape (batch_size, scalar_dim)
        mask: tensor of shape (batch_size, time_steps) indicating valid elements (needed for batching variable-length inputs)
        """
        # testing with 2d mfccs, not using 1d features for now
        # Process 1D features
        # x1d = self.encoder_1d(x1d)  # shape: (batch_size, 64, time_steps)
        # x1d = self.attn_pool_1d(x1d, mask)  # shape: (batch_size, 64)

        # Process 2D features
        x2d = self.encoder_2d(x2d)  # shape: (batch_size, 64, freq_bins, time_steps)
        x2d = torch.mean(x2d, dim=2)  # average over frequency dimension -> (batch_size, 64, time_steps)
        x2d = self.projection_2d(x2d)  # shape: (batch_size, 64, time_steps)
        x2d = self.attn_pool_2d(x2d, mask)  # shape: (batch_size, 64)

        # Process scalar features
        x_scalar = self.mlp_scalar(x_scalar)  # shape: (batch_size, 32)

        # Concatenate all summaries
        # combined = torch.cat([x1d, x2d, x_scalar], dim=-1)  # shape: (batch_size, 64 + 64 + 32)
        combined = torch.cat([x2d, x_scalar], dim=-1)  # shape: (batch_size, 64 + 32)

        # Final MLP
        output = self.final_mlp(combined)  # shape: (batch_size, 1)
        output = torch.sigmoid(self.k * output)  # skew toward edges of [0, 1]
        return output


class mlp_scalar_features(nn.Module):
    def __init__(
        self, num_scalar_features, hidden_sizes=[64, 64, 32], k=3.0, p_dropout=0.1
    ):
        """
        Args:
            num_scalar_features: number of scalar input features
            hidden_sizes: list of hidden layer sizes
            k: steepness factor for sigmoid (k > 1 outputs closer to 0 or 1)
            p_dropout: dropout probability for each hidden layer
        """
        super().__init__()

        layers = []
        in_features = num_scalar_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p_dropout))
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
        out = torch.sigmoid(self.k * logit)  # skew toward edges of [0, 1]
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
        record["signal"]: record[system + (f" {feature}" if feature else "")]
        for record in system_score
    }
    for record in records:
        record[f"{system}"] = system_score_index[record["signal"]]

    return pd.DataFrame(records)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def run_train_model(cfg: DictConfig) -> None:
    """Train a model to predict the intelligibility score from scalar features."""

    logger.info(f"Training model on {cfg.split} set...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Define model
    batch_size = 1
    num_scalar_features = 3
    # num_1d_channels = 1
    num_1d_channels = 0  # testing with 2d mfccs, not using 1d features for now
    num_2d_channels = 2  # MFCCs in stereo
    k = 1.0
    p_dropout = 0.3
    # model = mlp_scalar_features(num_scalar_features, k=k, p_dropout=p_dropout)
    model = multimodal_conv_mlp(
        num_1d_channels, num_2d_channels, num_scalar_features, k=k, p_dropout=p_dropout
    )
    model = model.to(device) # move model to GPU if available

    # load mfcc data
    mfcc_dir = "/mnt/d/cadenza_extracted_features/mfccs_raw_batch_1/mfcc/"
    x2d_dfs = []
    for file in os.listdir(mfcc_dir):
        if file.endswith(".json"):
            mfcc_path = os.path.join(mfcc_dir, file)
            mfcc_df = pd.read_json(mfcc_path)
            x2d_dfs.append(mfcc_df) # single column with signal name, then two rows of mfcc data
    x2d_df = pd.concat(x2d_dfs, axis=1) # -> [2 rows (channels) x num signals]

    # prepare scalar inputs
    # gather STOI score, whisper score, and VAR dB as input features from jsonl files
    stoi_df = load_features(cfg, "train", "stoi", None)
    whisper_df = load_features(cfg, "train", "whisper", None)
    features_df = load_features(cfg, "train", "features", "VAR (dB)")
    # use z-score normalization for VAR (dB)
    mean_var = features_df["features"].mean()
    std_var = features_df["features"].std()
    features_df["features"] = (features_df["features"] - mean_var) / std_var
    print("min VAR (dB): ", features_df["features"].min())
    print("max VAR (dB): ", features_df["features"].max())
    # merge dataframes on 'signal' column
    merged_df = stoi_df.merge(whisper_df[["signal", "whisper"]], on="signal")
    merged_df = merged_df.merge(features_df[["signal", "features"]], on="signal")
    # remove signals that don't have mfcc data
    merged_df = merged_df[merged_df["signal"].isin(x2d_df.columns)]
    assert len(merged_df) == x2d_df.shape[1], "Mismatch in number of samples between scalar and 2d features"
    print("scalar features:\n",merged_df.head())

    # prepare tensor of mfccs, padding to largest time dimension
    mfccs = []
    for col in x2d_df.columns:
        # Get the two channels of variable-length arrays
        arrays = [torch.tensor(arr, dtype=torch.float32) for arr in x2d_df[col]]
        # Each `arr` is (13, t_i)
        stacked = torch.stack(arrays, dim=0)  # shape: (2, 13, t_i)
        mfccs.append(stacked)
    # Convert each to (t_i, 2, 13) for pad_sequence
    seqs = [x.permute(2, 0, 1) for x in mfccs]
    # Pad to (batch, max_t, 2, 13)
    padded = pad_sequence(seqs, batch_first=True)
    # Move back to (batch, 2, 13, max_t)
    padded = padded.permute(0, 2, 3, 1)
    assert padded.shape[0] == len(merged_df), "Mismatch in number of samples between scalar and 2d features"
    assert padded.shape[1] == num_2d_channels, "Mismatch in number of 2d channels"
    assert padded.shape[2] == 13, "Expected 13 MFCC coefficients"
    x2d = padded
    print(f"x2d shape: {x2d.shape}")

    # create input tensor of scalar features
    input_scalar_features = merged_df[["stoi", "whisper", "features"]].values
    scalar_tensor = torch.tensor(input_scalar_features, dtype=torch.float32)
    # create labels tensor
    labels = merged_df["correctness"].values
    labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(
        1
    )  # shape: (num_samples, 1)

    # create dummy data for 1d feature over max time dimension
    x1d = torch.randn(len(merged_df), num_1d_channels, x2d.shape[-1])
    print(f"x1d shape: {x1d.shape}")

    # create dataset combining multimodal features
    dataset = torch.utils.data.TensorDataset(x1d, x2d, scalar_tensor, labels_tensor)

    # define parameters for training
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 100
    criterion = nn.MSELoss()
    patience = 10
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_state = None
    train_losses = []
    val_losses = []

    # Split train/val (simple split)
    val_split = 0.1
    num_samples = len(dataset)
    num_val = int(num_samples * val_split)
    num_train = num_samples - num_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [num_train, num_val]
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    logger.info(
        f"Training model with parameters:\n batch size {batch_size}, sigmoid steepness {k}, dropout probability {p_dropout}, optimizer Adam, loss MSE, num_epochs {num_epochs}, early stopping patience {patience}"
    )

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_x1d, batch_x2d, batch_scalar, batch_y in train_loader:
            # move data to GPU if available
            batch_x1d = batch_x1d.to(device)
            batch_x2d = batch_x2d.to(device)
            batch_scalar = batch_scalar.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x1d, batch_x2d, batch_scalar)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_scalar.size(0) # average loss * batch size = total loss
        train_loss /= num_train
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_x1d, val_x2d, val_scalar, val_y in val_loader:
                # move data to GPU if available
                val_x1d = val_x1d.to(device)
                val_x2d = val_x2d.to(device)
                val_scalar = val_scalar.to(device)
                val_y = val_y.to(device)

                val_outputs = model(val_x1d, val_x2d, val_scalar)
                val_loss += criterion(val_outputs, val_y).item() * val_scalar.size(0) # average loss * batch size = total loss
        val_loss /= num_val
        val_losses.append(val_loss)

        logger.info(
            f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}"
        )

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
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.savefig(f"{cfg.data.dataset}.train.mlp_scalar_features.loss_curve.png")
    plt.close()

    # Save model
    model_path = f"{cfg.data.dataset}.train.mlp_scalar_features.pth"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")

    # Log a model summary and final loss
    logger.info(model)
    logger.info(
        f"Final training loss: {train_losses[-1]:.4f}, final validation loss: {val_losses[-1]:.4f}"
    )


@hydra.main(config_path="configs", config_name="config", version_base=None)
def run_inference(cfg: DictConfig) -> None:
    """Run inference using the trained model on both the train and validation sets.
    Inference on train set can be compared to correctness labels to get a sense of model fit.
    Inference on validation set must be submitted to leaderboard to evaluate performance.
    """
    model_path = f"{cfg.data.dataset}.train.mlp_scalar_features.pth"
    logger.info(f"Running inference using model from {model_path}...")

    num_scalar_features = 3
    model = mlp_scalar_features(num_scalar_features)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # gather STOI score, whisper score, and VAR dB as input features from jsonl files
    stoi_df = load_features(cfg, "train", "stoi", None)
    whisper_df = load_features(cfg, "train", "whisper", None)
    features_df = load_features(cfg, "train", "features", "VAR (dB)")
    # use z-score normalization for VAR (dB)
    mean_var = features_df["features"].mean()
    std_var = features_df["features"].std()
    features_df["features"] = (features_df["features"] - mean_var) / std_var
    print("min VAR (dB): ", features_df["features"].min())
    print("max VAR (dB): ", features_df["features"].max())
    # merge dataframes on 'signal' column
    merged_df_train = stoi_df.merge(whisper_df[["signal", "whisper"]], on="signal")
    merged_df_train = merged_df_train.merge(
        features_df[["signal", "features"]], on="signal"
    )
    print(merged_df_train.head())
    # create input tensor
    input_features = merged_df_train[["stoi", "whisper", "features"]].values
    input_tensor = torch.tensor(input_features, dtype=torch.float32)

    # Run inference
    with torch.no_grad():
        outputs = model(input_tensor)
        print(outputs)
        # save outputs to csv
        merged_df_train["predicted_correctness"] = outputs.numpy()
        output_csv_path = f"{cfg.data.dataset}.train.mlp_scalar_features.inference.csv"
        merged_df_train.to_csv(output_csv_path, index=False)
        logger.info(f"Inference results saved to {output_csv_path}")

    # repeat for validation set
    stoi_df = load_features(cfg, "valid", "stoi", None)
    whisper_df = load_features(cfg, "valid", "whisper", None)
    features_df = load_features(cfg, "valid", "features", "VAR (dB)")
    # merge dataframes on 'signal' column
    merged_df_valid = stoi_df.merge(whisper_df[["signal", "whisper"]], on="signal")
    merged_df_valid = merged_df_valid.merge(
        features_df[["signal", "features"]], on="signal"
    )
    print(merged_df_valid.head())
    # create input tensor
    input_features = merged_df_valid[["stoi", "whisper", "features"]].values
    input_tensor = torch.tensor(input_features, dtype=torch.float32)

    # Run inference
    with torch.no_grad():
        outputs = model(input_tensor)
        print(outputs)
        # save outputs to csv
        merged_df_valid["predicted_correctness"] = outputs.numpy()
        output_csv_path = f"{cfg.data.dataset}.valid.mlp_scalar_features.inference.csv"
        merged_df_valid.to_csv(output_csv_path, index=False)
        logger.info(f"Inference results saved to {output_csv_path}")


if __name__ == "__main__":
    run_train_model()
    run_inference()
