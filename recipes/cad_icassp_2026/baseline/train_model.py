import json
import logging
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
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
        """
        # x: (batch, input_dim, time_steps) -> permute to (batch, time_steps, input_dim)
        seq = x.permute(0, 2, 1)
        # Compute attention scores -> (batch, time_steps, 1) -> squeeze -> (batch, time_steps)
        attn_scores = self.attention(seq).squeeze(-1)

        if mask is not None:
            # mask: (batch, time_steps) boolean (True=valid). If it's float/int, convert.
            if mask.dtype != torch.bool:
                mask = mask.bool()
            # Set scores for invalid positions to a large negative value before softmax
            attn_scores = attn_scores.masked_fill(~mask, float("-1e9"))

        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)  # (batch, time_steps, 1)

        # Weighted sum of inputs -> (batch, input_dim)
        pooled = torch.sum(attn_weights * seq, dim=1)
        return pooled


class multimodal_conv_mlp(nn.Module):
    """Combines MLP for scalar features with CNN and attention pooling for 1d and 2d features."""
    def __init__(self, c1_in, c2_in, scalar_dim, k=1.0, p_dropout=0.1, post_dropout=0.05):
        """
        Args:
            c1_in: number of input channels for 1D CNN
            c2_in: number of input channels for 2D CNN
            scalar_dim: dimensionality of scalar features
            k: steepness factor for sigmoid (k > 1 outputs closer to 0 or 1)
            p_dropout: dropout probability for final MLP
            post_dropout: dropout probability for post-attention dropout layers
        """
        super().__init__()

        # 1D CNN encoder for temporal features
        # left channel
        self.encoder_1d_left = nn.Sequential(
            nn.Conv1d(c1_in, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.GroupNorm(1, 32),
            nn.ReLU(),
        )
        self.attn_pool_1d_left = temporal_attention_pool(32)
        self.postdrop_1d_left = nn.Dropout(post_dropout)
        # right channel
        self.encoder_1d_right = nn.Sequential(
            nn.Conv1d(c1_in, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.GroupNorm(1, 32),
            nn.ReLU(),
        )
        self.attn_pool_1d_right = temporal_attention_pool(32)
        self.postdrop_1d_right = nn.Dropout(post_dropout)

        # 2D CNN encoder for spectro-temporal features
        self.encoder_2d = nn.Sequential(
            nn.Conv2d(c2_in, 64, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
        )
        # projection after frequency pooling
        self.projection_2d = nn.Conv1d(64, 64, kernel_size=1) # learn linear weights for each channel per time step
        self.norm_2d = nn.GroupNorm(1, 64)
        self.attn_pool_2d = temporal_attention_pool(64)
        self.postdrop_2d = nn.Dropout(post_dropout)

        # simple MLP for scalar features
        self.mlp_scalar = nn.Sequential(
            nn.Linear(scalar_dim, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Dropout(post_dropout),
        )

        # final MLP concatenating the summaries of the three feature types
        self.final_mlp = nn.Sequential(
            nn.Linear(32 + 32 + 64 + 16, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        # sigmoid steepness factor
        self.k = k


    def forward(self, x1d_left, x1d_right, x2d, spec_left, spec_right, x_scalar, mask1d=None, mask2d=None):
        """
        x1d_left: tensor of shape (batch_size, c1_in, time_steps)
        x1d_right: tensor of shape (batch_size, c1_in, time_steps)
        x2d: tensor of shape (batch_size, c2_in, freq_bins, time_steps)
        spec_left/spec_right: optional spectral features (batch, 2, time)
        x_scalar: tensor of shape (batch_size, scalar_dim)
        mask: tensor of shape (batch_size, time_steps) indicating valid elements (needed for batching variable-length inputs)
        """
        # Process 1D features
        x1d_left = self.encoder_1d_left(x1d_left)  # (batch_size, c1_in, time_steps) -> (batch_size, 32, time_steps)
        x1d_left = self.attn_pool_1d_left(x1d_left, mask1d)  # shape: (batch_size, 32)
        x1d_left = self.postdrop_1d_left(x1d_left) # dropout before concatenation between branches
        x1d_right = self.encoder_1d_right(x1d_right)  # (batch_size, c1_in, time_steps) -> (batch_size, 32, time_steps)
        x1d_right = self.attn_pool_1d_right(x1d_right, mask1d)  # shape: (batch_size, 32)
        x1d_right = self.postdrop_1d_right(x1d_right) # dropout before concatenation between branches

        # Process 2D features
        x2d = self.encoder_2d(x2d)  # shape: (batch_size, 64, freq_bins, time_steps)
        x2d = torch.mean(x2d, dim=2)  # average over frequency dimension -> (batch_size, 64, time_steps)
        x2d = self.projection_2d(x2d)  # shape: (batch_size, 64, time_steps)
        x2d = self.norm_2d(x2d)
        x2d = self.attn_pool_2d(x2d, mask2d)  # shape: (batch_size, 64)
        x2d = self.postdrop_2d(x2d) # dropout before concatenation between branches

        # Process scalar features
        x_scalar = self.mlp_scalar(x_scalar)  # shape: (batch_size, 16)

        # Concatenate all summaries
        combined = torch.cat([x1d_left, x1d_right, x2d, x_scalar], dim=-1)  # shape: (batch_size, 32 + 32 + 64 + 16)

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


class simple_meanpool_scalar_mlp(nn.Module):
    """Simple model that uses 1D whisper embeddings (left/right), 2D MFCCs, and scalar features.
    It performs mean pooling over frequency and masked mean pooling over time for MFCCs,
    masked mean pooling for 1D embeddings, projects pooled vectors and concatenates
    everything before an MLP.
    """
    def __init__(self, c1_in, scalar_dim, proj_dim=32, c2_in=2, proj_dim2=32, k=1.0, p_dropout=0.1, post_dropout: float = 0.05):
        super().__init__()

        # linear projection after mean pooling to reduce dimensionality
        self.proj_left = nn.Linear(c1_in, proj_dim)
        self.proj_right = nn.Linear(c1_in, proj_dim)
        # projection for 2D MFCC pooled features
        self.proj_2d = nn.Linear(c2_in, proj_dim2)
        # small spectral feature branches: process centroid+rolloff per-channel -> channels=4 each
        self.spec_conv_left = nn.Conv1d(2, 4, kernel_size=3, padding=1)
        self.spec_gn_left = nn.GroupNorm(1, 4)
        self.spec_conv_right = nn.Conv1d(2, 4, kernel_size=3, padding=1)
        self.spec_gn_right = nn.GroupNorm(1, 4)

        # MLP for scalar features
        self.mlp_scalar = nn.Sequential(nn.Linear(scalar_dim, 32), nn.ReLU())

        # final classifier
        self.final_mlp = nn.Sequential(
            nn.Linear(proj_dim * 2 + proj_dim2 + 32 + 8, 64),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(32, 1),
        )
        self.k = k
        # post-branch dropout applied after each branch and before concatenation
        self.postdrop_left = nn.Dropout(post_dropout)
        self.postdrop_right = nn.Dropout(post_dropout)
        self.postdrop_2d = nn.Dropout(post_dropout)
        self.postdrop_scalar = nn.Dropout(post_dropout)
        self.postdrop_spec = nn.Dropout(post_dropout)

    @staticmethod
    def _masked_mean_pool(x: torch.Tensor, mask: torch.Tensor | None):
        """Mean-pool over time with optional mask.
        x shape: (batch, features, time)
        mask shape: (batch, time)
        returns (batch, features)
        """
        if mask is None:
            return x.mean(dim=-1)
        if mask.dtype != torch.bool:
            mask = mask.bool()
        mask_f = mask.unsqueeze(1).to(x.dtype)  # (batch,1,time)
        summed = (x * mask_f).sum(dim=-1)
        lengths = mask_f.sum(dim=-1).clamp(min=1.0)
        return summed / lengths

    def forward(self, x1d_left, x1d_right, x2d, spec_left, spec_right, x_scalar, mask1d=None, mask2d=None):
        """
        x1d_left/right: (batch, c1_in, time)
        x2d: (batch, c2_in, freq, time)
        spec: (batch, 2, time)  # spectral centroid and rolloff per time
        x_scalar: (batch, scalar_dim)
        mask1d: (batch, time)
        mask2d: (batch, time)
        """
        # pooled embeddings via masked mean pooling over time (batch, c1_in)
        pooled_left = self._masked_mean_pool(x1d_left, mask1d)
        pooled_right = self._masked_mean_pool(x1d_right, mask1d)

        # project to smaller dimension
        left_feat = torch.relu(self.proj_left(pooled_left))
        # apply small dropout after left/right projections
        left_feat = self.postdrop_left(left_feat)
        right_feat = torch.relu(self.proj_right(pooled_right))
        right_feat = self.postdrop_right(right_feat)

        # scalar embedding
        scalar_feat = self.mlp_scalar(x_scalar)
        scalar_feat = self.postdrop_scalar(scalar_feat)

        # Process spectral features per-channel (centroid + rolloff): spec_left/right shape (batch, 2, t)
        if spec_left is None or spec_right is None:
            spec_feat = torch.zeros((x1d_left.size(0), 16), device=x1d_left.device)
        else:
            # left branch
            spec_l = self.spec_conv_left(spec_left)
            spec_l = self.spec_gn_left(spec_l)
            spec_l = torch.relu(spec_l)
            spec_feat_l = self._masked_mean_pool(spec_l, mask2d)
            # right branch
            spec_r = self.spec_conv_right(spec_right)
            spec_r = self.spec_gn_right(spec_r)
            spec_r = torch.relu(spec_r)
            spec_feat_r = self._masked_mean_pool(spec_r, mask2d)
            spec_feat = torch.cat([spec_feat_l, spec_feat_r], dim=-1)
            spec_feat = self.postdrop_spec(spec_feat)

        # Process 2D MFCCs: average over frequency, then temporal attention over time
        if x2d is None:
            # fallback zero vector if no x2d provided
            pooled_2d = torch.zeros((x1d_left.size(0), self.proj_2d.in_features), device=x1d_left.device)
        else:
            # x2d: (batch, c2, freq, time) -> mean over freq -> (batch, c2, time)
            x2d_freq = x2d.mean(dim=2)
            pooled_2d = self._masked_mean_pool(x2d_freq, mask2d)  # (batch, c2)
        # project 2d pooled features and apply small dropout
        proj_2d = torch.relu(self.proj_2d(pooled_2d))
        proj_2d = self.postdrop_2d(proj_2d)
        combined = torch.cat([left_feat, right_feat, proj_2d, scalar_feat, spec_feat], dim=-1)
        out = self.final_mlp(combined)
        out = torch.sigmoid(self.k * out)
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


class EmbeddingDataset(torch.utils.data.Dataset):
    """Dataset that lazily loads per-signal whisper embeddings and MFCCs.

    Each item is a tuple: (left_tensor, right_tensor, x2d_tensor, scalar_tensor, label_tensor)
    - left/right: torch.Tensor shape (time, d_model)
    - x2d: torch.Tensor shape (2, freq_bins, time)
    - scalar: torch.Tensor shape (num_scalar_features,)
    - label: torch.Tensor shape (1,)
    """

    def __init__(
        self,
        merged_df: pd.DataFrame,
        whisper_emb_dir: Path,
        x2d_df: pd.DataFrame,
        system: str,
        augment: bool = False,
        swap_prob: float = 0.5,
        noise_std: float | None = None,
        emb_noise_std: float | None = None,
        scalar_jitter: float = 0.01,
    ):
        self.signals = merged_df["signal"].tolist()
        self.scalar_arr = merged_df[["stoi", "whisper", "features"]].values.astype(np.float32)

        # Some splits (e.g. validation) may not have correctness labels.
        # Support unlabeled datasets by creating a dummy label array of zeros
        # so downstream collate/fwd code can uniformly expect a label tensor.
        if "correctness" in merged_df.columns:
            self.labels = merged_df["correctness"].values.astype(np.float32)
        else:
            logging.getLogger(__name__).warning(
                "Merged DataFrame has no 'correctness' column; creating dummy zero labels."
            )
            self.labels = np.zeros((len(self.signals),), dtype=np.float32)

        self.whisper_emb_dir = Path(whisper_emb_dir)
        self.x2d_df = x2d_df
        self.system = system

        # augmentation parameters
        # noise_std: if None, compute per-sample as 0.01 * std(mfcc) in _augment
        self.augment = bool(augment)
        self.swap_prob = float(swap_prob)
        self.noise_std = None if noise_std is None else float(noise_std)
        # embedding noise std: if None, compute per-sample as 0.01 * std(embedding)
        self.emb_noise_std = None if emb_noise_std is None else float(emb_noise_std)
        # relative jitter to apply to scalar features; default 1%
        self.scalar_jitter = float(scalar_jitter)
        # optional caches for memmap and per-sample stds to avoid repeated IO/computation
        self._memmap_cache = {}
        self._emb_std = None
        self._mfcc_std = None

        # If augmentation is enabled, we may precompute per-sample stds to avoid
        # calling .std() on tensors repeatedly in __getitem__/_augment.
        if self.augment:
            # Precompute embedding stds (one-time pass over files)
            emb_stds = []
            for sig in self.signals:
                left_path = self.whisper_emb_dir / f"{sig}.{self.system}.left.npy"
                try:
                    arr = np.load(str(left_path), mmap_mode="r")
                    emb_stds.append(float(arr.std()))
                except Exception:
                    emb_stds.append(0.0)
            self._emb_std = np.array(emb_stds, dtype=np.float32)
            # Precompute MFCC stds only if x2d_df was provided
            if getattr(self, "x2d_df", None) is not None and not self.x2d_df.empty:
                mfcc_stds = []
                for sig in self.signals:
                    try:
                        col = self.x2d_df[sig]
                        # col may be a list-like of two arrays
                        arr = np.hstack([np.asarray(c).ravel() for c in col])
                        mfcc_stds.append(float(np.std(arr)))
                    except Exception:
                        mfcc_stds.append(0.0)
                self._mfcc_std = np.array(mfcc_stds, dtype=np.float32)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        sig = self.signals[idx]
        left_path = self.whisper_emb_dir / f"{sig}.{self.system}.left.npy"
        right_path = self.whisper_emb_dir / f"{sig}.{self.system}.right.npy"
        if not left_path.exists() or not right_path.exists():
            raise FileNotFoundError(f"Missing embedding for signal {sig}")

        # load with mmap to reduce peak memory; avoid unnecessary copies
        # Use a tiny cache to avoid reopening files repeatedly across epochs
        def _load_memmap(p: Path):
            key = str(p)
            mm = self._memmap_cache.get(key)
            if mm is None:
                mm = np.load(key, mmap_mode="r")
                self._memmap_cache[key] = mm
            return mm

        left_arr = _load_memmap(left_path)
        right_arr = _load_memmap(right_path)

        # ensure shape is (time, d_model)
        if left_arr.ndim == 2 and left_arr.shape[0] == left_arr.shape[1]:
            # ambiguous, keep as is
            pass
        if left_arr.ndim == 2 and left_arr.shape[0] == 512 and left_arr.shape[1] != 512:
            left_arr = left_arr.T
        if right_arr.ndim == 2 and right_arr.shape[0] == 512 and right_arr.shape[1] != 512:
            right_arr = right_arr.T

        # minimize copies: convert memmap to float32 if necessary without forcing a copy
        if left_arr.dtype != np.float32:
            left_arr = left_arr.astype(np.float32, copy=False)
        if right_arr.dtype != np.float32:
            right_arr = right_arr.astype(np.float32, copy=False)
        left_t = torch.from_numpy(left_arr)
        right_t = torch.from_numpy(right_arr)

        # MFCCs are stored in x2d_df as two arrays per column
        # MFCCs and optional spectral features (centroid, rolloff) are stored
        # in `x2d_df` per column. We try to extract MFCC arrays (2D) and any
        # 1D spectral arrays (treated as centroid/rolloff) if present.
        # prepare x2d and spectral centroid/rolloff per-channel
        spec_left = None
        spec_right = None
        if getattr(self, "x2d_df", None) is None or self.x2d_df.empty:
            x2d = torch.zeros((2, 1, 1), dtype=torch.float32)
            spec_left = torch.zeros((2, 1), dtype=torch.float32)
            spec_right = torch.zeros((2, 1), dtype=torch.float32)
        else:
            mfcc_col = self.x2d_df[sig]
            arrays = []
            spec_2ch_list = []
            for item in mfcc_col:
                a = np.asarray(item)
                # Spectral features (centroid/rolloff) may be stored as (2, t) or (1, t).
                # MFCCs are (freq, time) (e.g. 13 x T). Classify by first-dim size.
                if a.ndim == 2 and a.shape[0] in (1, 2):
                    # (1, t) -> duplicate to stereo; (2, t) -> already stereo
                    if a.shape[0] == 1:
                        arr2 = np.vstack([a, a])
                        spec_2ch_list.append(arr2)
                    else:
                        spec_2ch_list.append(a)
                elif a.ndim == 2:
                    # treat as MFCC-like (freq, time)
                    arrays.append(torch.tensor(a, dtype=torch.float32))
                elif a.ndim == 1:
                    # fallback: 1D arrays (t,) - assume mono spectral; duplicate to both channels
                    arr2 = np.stack([a, a], axis=0)
                    spec_2ch_list.append(arr2)

            if arrays:
                x2d = torch.stack(arrays, dim=0)
            else:
                x2d = torch.zeros((2, 1, 1), dtype=torch.float32)

            # Build per-channel spec_left/spec_right where each is (2, t): [centroid, rolloff]
            if len(spec_2ch_list) >= 2:
                f0 = spec_2ch_list[0]
                f1 = spec_2ch_list[1]
                L = max(f0.shape[1], f1.shape[1])
                def pad_2ch(a, L):
                    if a.shape[1] == L:
                        return a
                    out = np.zeros((2, L), dtype=a.dtype)
                    out[:, : a.shape[1]] = a
                    return out

                f0 = pad_2ch(f0, L)
                f1 = pad_2ch(f1, L)
                # spec_left: stack centroid_left, rolloff_left -> shape (2, L)
                spec_left = torch.tensor(np.stack([f0[0, :], f1[0, :]], axis=0), dtype=torch.float32)
                # spec_right: stack centroid_right, rolloff_right -> shape (2, L)
                spec_right = torch.tensor(np.stack([f0[1, :], f1[1, :]], axis=0), dtype=torch.float32)
            elif len(spec_2ch_list) == 1:
                f0 = spec_2ch_list[0]
                L = f0.shape[1]
                # duplicate second feature as zeros
                f1 = np.zeros((2, L), dtype=f0.dtype)
                spec_left = torch.tensor(np.stack([f0[0, :], f1[0, :]], axis=0), dtype=torch.float32)
                spec_right = torch.tensor(np.stack([f0[1, :], f1[1, :]], axis=0), dtype=torch.float32)
            else:
                spec_left = torch.zeros((2, 1), dtype=torch.float32)
                spec_right = torch.zeros((2, 1), dtype=torch.float32)

        scalar = torch.from_numpy(self.scalar_arr[idx]).float()
        label = torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(0)

        # Apply augmentation if enabled (only intended for training)
        if self.augment:
            # choose per-sample defaults using precomputed stds when available
            noise_std = self.noise_std
            if noise_std is None:
                if self._mfcc_std is not None:
                    noise_std = 0.01 * float(self._mfcc_std[idx])
            emb_noise_std = self.emb_noise_std
            if emb_noise_std is None:
                if self._emb_std is not None:
                    emb_noise_std = 0.01 * float(self._emb_std[idx])

            left_t, right_t, x2d, spec_left, spec_right, scalar = self._augment(
                left_t, right_t, x2d, spec_left, spec_right, scalar, noise_std=noise_std, emb_noise_std=emb_noise_std
            )

        return left_t, right_t, x2d, spec_left, spec_right, scalar, label

    def _augment(
        self,
        left_t: torch.Tensor,
        right_t: torch.Tensor,
        x2d: torch.Tensor,
        spec_left: torch.Tensor,
        spec_right: torch.Tensor,
        scalar: torch.Tensor,
        noise_std: float | None = None,
        emb_noise_std: float | None = None,
    ):
        """Apply data augmentation:
        - With probability `swap_prob` swap left/right embeddings and swap the two MFCC channels together.
        - Add Gaussian noise to MFCCs with std=`noise_std` (per-sample default 1% of MFCC std).
        - Add Gaussian noise to whisper embeddings (left/right) with std=`emb_noise_std` (per-sample default 1% of embedding std).
        - Apply relative jitter to scalar features with standard deviation `scalar_jitter` (fractional).

        left_t/right_t shapes: (time, d_model)
        x2d shape: (2, freq, time)
        scalar shape: (num_scalar_features,)
        Returns augmented (left_t, right_t, x2d, scalar)
        """
        # Decide whether to swap
        if np.random.rand() < self.swap_prob:
            # swap embeddings
            left_t, right_t = right_t, left_t
            # swap MFCC channels along channel dim 0
            x2d = x2d[[1, 0], ...]
            # swap spectral left/right arrays as well
            spec_left, spec_right = spec_right, spec_left

        # Add Gaussian noise to MFCCs (in-place creation)
        # noise_std and emb_noise_std are provided by caller where possible
        if noise_std is None:
            noise_std = self.noise_std

        if noise_std is not None and noise_std > 0.0:
            noise = torch.randn_like(x2d) * float(noise_std)
            x2d = x2d + noise
            # add small noise to spectral features (match temporal dim)
            try:
                if spec_left is not None:
                    spec_noise_l = torch.randn_like(spec_left) * float(noise_std)
                    spec_left = spec_left + spec_noise_l
                if spec_right is not None:
                    spec_noise_r = torch.randn_like(spec_right) * float(noise_std)
                    spec_right = spec_right + spec_noise_r
            except Exception:
                pass

        # Add Gaussian noise to whisper embeddings
        if emb_noise_std is None:
            emb_noise_std = self.emb_noise_std

        if emb_noise_std is not None and emb_noise_std > 0.0:
            left_t = left_t + torch.randn_like(left_t) * float(emb_noise_std)
            right_t = right_t + torch.randn_like(right_t) * float(emb_noise_std)

        # Apply relative jitter to scalar features (multiplicative noise ~ N(1, scalar_jitter))
        if self.scalar_jitter is not None and float(self.scalar_jitter) > 0.0:
            scalar = scalar * (1.0 + torch.randn_like(scalar) * float(self.scalar_jitter))

        return left_t, right_t, x2d, spec_left, spec_right, scalar


def collate_batch(samples):
    """Collate function that pads variable-length temporal dimensions per-batch.

    Expects samples list of (left_t, right_t, x2d, scalar, label)
    Returns tensors shaped for model input:
      x1d_left: (batch, d_model, max_t)
      x1d_right: (batch, d_model, max_t)
      x2d: (batch, c2, freq_bins, max_t2)
      scalar: (batch, num_scalar_features)
      labels: (batch, 1)
    """
    lefts = [s[0] for s in samples]
    rights = [s[1] for s in samples]
    x2ds = [s[2] for s in samples]
    specs_left = [s[3] for s in samples]
    specs_right = [s[4] for s in samples]
    scalars = torch.stack([s[5] for s in samples], dim=0)
    labels = torch.stack([s[6] for s in samples], dim=0)

    # pad left/right (list of (t, d)) -> pad_sequence -> (batch, max_t, d)
    # also compute boolean masks indicating valid (non-padded) time steps
    lengths_left = [l.shape[0] for l in lefts]
    padded_left = pad_sequence(lefts, batch_first=True)
    padded_right = pad_sequence(rights, batch_first=True)
    # convert to (batch, d, max_t)
    padded_left = padded_left.permute(0, 2, 1)
    padded_right = padded_right.permute(0, 2, 1)

    # build mask for 1D sequences: shape (batch, max_t)
    max_t = padded_left.shape[-1]
    lengths_tensor = torch.tensor(lengths_left, dtype=torch.long)
    mask1d = (torch.arange(max_t).unsqueeze(0) < lengths_tensor.unsqueeze(1))

    # pad x2d: inputs are (2, freq, t) -> permute to (t, 2, freq) for pad_sequence
    seqs_x2d = [x.permute(2, 0, 1) for x in x2ds]
    lengths_x2d = [s.shape[0] for s in seqs_x2d]
    padded_x2d = pad_sequence(seqs_x2d, batch_first=True)  # (batch, max_t2, 2, freq)
    padded_x2d = padded_x2d.permute(0, 2, 3, 1)  # (batch, 2, freq, max_t2)

    # build mask for 2D sequences: shape (batch, max_t2)
    max_t2 = padded_x2d.shape[-1]
    lengths_x2d_tensor = torch.tensor(lengths_x2d, dtype=torch.long)
    mask2d = (torch.arange(max_t2).unsqueeze(0) < lengths_x2d_tensor.unsqueeze(1))

    # pad spectral features (each spec_left/spec_right is (2, t)) -> permute to (t, 2) for pad_sequence
    seqs_spec_left = [s.permute(1, 0) for s in specs_left]
    seqs_spec_right = [s.permute(1, 0) for s in specs_right]
    lengths_spec = [s.shape[0] for s in seqs_spec_left]
    padded_spec_left = pad_sequence(seqs_spec_left, batch_first=True)  # (batch, max_t_spec, 2)
    padded_spec_left = padded_spec_left.permute(0, 2, 1)  # (batch, 2, max_t_spec)
    padded_spec_right = pad_sequence(seqs_spec_right, batch_first=True)  # (batch, max_t_spec, 2)
    padded_spec_right = padded_spec_right.permute(0, 2, 1)  # (batch, 2, max_t_spec)
    max_t_spec = padded_spec_left.shape[-1]
    lengths_spec_tensor = torch.tensor(lengths_spec, dtype=torch.long)
    mask_spec = (torch.arange(max_t_spec).unsqueeze(0) < lengths_spec_tensor.unsqueeze(1))

    return padded_left, padded_right, padded_x2d, padded_spec_left, padded_spec_right, scalars, labels, mask1d, mask2d


@hydra.main(config_path="configs", config_name="config", version_base=None)
def run_train_model(cfg: DictConfig) -> None:
    """Train a model to predict the intelligibility score from scalar features."""

    logger.info(f"Training model on {cfg.split} set...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Define model
    batch_size = 16
    num_scalar_features = 3
    num_1d_channels = 512  # whisper encoder embedding size
    num_2d_channels = 2  # MFCCs in stereo
    beta = 0.1  # for SmoothL1Loss
    k = 1.0 # sigmoid steepness
    p_dropout = 0.1
    post_dropout = 0.05
    # Choose model type from config (defaults to simple mean-pool model)
    model_type = getattr(cfg.baseline, "model_type", "simple")
    if model_type == "simple":
        proj_dim = getattr(cfg.baseline, "proj_dim", 64)
        model = simple_meanpool_scalar_mlp(
            num_1d_channels, num_scalar_features, proj_dim=proj_dim, k=k, p_dropout=p_dropout, post_dropout=post_dropout
        )
    else:
        model = multimodal_conv_mlp(
            num_1d_channels, num_2d_channels, num_scalar_features, k=k, p_dropout=p_dropout, post_dropout=post_dropout
        )
    model = model.to(device) # move model to GPU if available
    # DataLoader parallel settings
    num_workers = 2
    pin_memory = True if device.type == "cuda" else False

    # Note: loading all whisper per-frame embeddings at once can use a lot of memory
    # We will defer loading per-signal embeddings until after we build and filter
    # the metadata DataFrame so we only load embeddings needed for training.
    whisper_emb_dir = Path(cfg.data.cadenza_data_root) / cfg.data.dataset / "embeddings"

    # load MFCC data
    mfcc_dir = "/mnt/d/cadenza_extracted_features/full_batch_cmvn_mfccs/mfcc/"
    spectral_centroid_dir = "/mnt/d/cadenza_extracted_features/spectral-centroid-train/centroid/"
    spectral_rolloff_dir = "/mnt/d/cadenza_extracted_features/spectral-rolloff-train/spectral-rolloff/"
    x2d_dfs = []
    for file in os.listdir(mfcc_dir):
        if file.endswith(".json"):
            print(f"Processing file: {file}")
            mfcc_path = os.path.join(mfcc_dir, file)
            mfcc_df = pd.read_json(mfcc_path)
            x2d_dfs.append(mfcc_df) # single column with signal name, then two rows of mfcc data
    x2d_df = pd.concat(x2d_dfs, axis=1) # -> [2 rows (channels) x num signals]
    # Optionally load spectral centroid and rolloff files and append them as additional rows
    if os.path.isdir(spectral_centroid_dir):
        centroid_dfs = []
        for file in os.listdir(spectral_centroid_dir):
            if file.endswith('.json'):
                centroid_path = os.path.join(spectral_centroid_dir, file)
                try:
                    cdf = pd.read_json(centroid_path)
                    centroid_dfs.append(cdf)
                except Exception:
                    logger.warning(f"Failed to read centroid file {centroid_path}")
        if centroid_dfs:
            centroid_df = pd.concat(centroid_dfs, axis=1)
            # ensure columns align with MFCC columns; reindex to avoid misalignment
            centroid_df = centroid_df.reindex(columns=x2d_df.columns)
            missing = list(set(x2d_df.columns) - set(centroid_df.columns))
            if missing:
                logger.warning(
                    "Spectral centroid data missing for %d signals; these will be filled with NaN: %s",
                    len(missing), str(missing[:5]),
                )
            x2d_df = pd.concat([x2d_df, centroid_df], axis=0)
    if os.path.isdir(spectral_rolloff_dir):
        rolloff_dfs = []
        for file in os.listdir(spectral_rolloff_dir):
            if file.endswith('.json'):
                rolloff_path = os.path.join(spectral_rolloff_dir, file)
                try:
                    rdf = pd.read_json(rolloff_path)
                    rolloff_dfs.append(rdf)
                except Exception:
                    logger.warning(f"Failed to read rolloff file {rolloff_path}")
        if rolloff_dfs:
            rolloff_df = pd.concat(rolloff_dfs, axis=1)
            # reindex to MFCC columns to guarantee alignment
            rolloff_df = rolloff_df.reindex(columns=x2d_df.columns)
            missing = list(set(x2d_df.columns) - set(rolloff_df.columns))
            if missing:
                logger.warning(
                    "Spectral rolloff data missing for %d signals; these will be filled with NaN: %s",
                    len(missing), str(missing[:5]),
                )
            x2d_df = pd.concat([x2d_df, rolloff_df], axis=0)
    assert len(x2d_df.columns) == 8802, f"Expected 8802 signals in MFCC data but got {len(x2d_df.columns)}"

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
    # If using multimodal model, remove signals that don't have mfcc data and
    # perform the usual sanity checks. For the simple model, skip filtering.
    merged_df = merged_df[merged_df["signal"].isin(x2d_df.columns)]
    assert len(merged_df) == x2d_df.shape[1], "Mismatch in number of samples between scalar and 2d features"
    # Sanity check: expected train set size
    assert len(merged_df) == 8802, f"Expected 8802 training samples but got {len(merged_df)}"
    print("scalar features:\n",merged_df.head())

    # Create streaming datasets: one with augmentation enabled for training
    # and one without augmentation for validation. We will split indices so
    # augmentation only affects the training subset.
    # Default augmentation hyperparameters
    aug_swap_prob = 0.25
    # If None, noise std is computed per-sample inside the dataset as 0.01 * mfcc_std
    aug_noise_std = None
    # label jitter applied to training targets (fractional/additive). Default 1%.
    label_jitter = 0.005
    emb_noise_std = None
    scalar_jitter = 0.005
    dataset_aug = EmbeddingDataset(
        merged_df,
        whisper_emb_dir,
        x2d_df,
        system=cfg.baseline.system,
        augment=True,
        swap_prob=aug_swap_prob,
        noise_std=aug_noise_std,
        emb_noise_std=emb_noise_std,
        scalar_jitter=scalar_jitter,
    )
    dataset_noaug = EmbeddingDataset(
        merged_df,
        whisper_emb_dir,
        x2d_df,
        system=cfg.baseline.system,
        augment=False,
    )

    # define parameters for training
    model.train()
    lr = 5e-5
    wd = 1e-5
    # LayerNorm and biases should not have weight decay
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if param.ndim == 1 or "bias" in name or "norm" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)
    optimizer = torch.optim.Adam(
        [
            {"params": decay, "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
    )
    # Support resuming training from a checkpoint provided in the config
    start_epoch = 0
    resume_path = getattr(cfg, "resume_checkpoint", None)
    if resume_path:
        resume_path = Path(str(resume_path))
        if resume_path.exists():
            try:
                ckpt = torch.load(str(resume_path), map_location=device)
                # support both raw state_dict and dict containing model/optimizer
                if isinstance(ckpt, dict) and "model_state" in ckpt:
                    model_state = ckpt["model_state"]
                else:
                    model_state = ckpt
                try:
                    model.load_state_dict(model_state)
                except Exception as e:
                    logger.warning(f"Failed to load model_state from resume checkpoint: {e}")

                if isinstance(ckpt, dict) and "optimizer_state" in ckpt:
                    try:
                        optimizer.load_state_dict(ckpt.get("optimizer_state"))
                    except Exception as e:
                        logger.warning(f"Failed to load optimizer_state from resume checkpoint: {e}")

                # restore epoch and best metric values if present (use safe defaults)
                start_epoch = int(ckpt.get("epoch", 0)) if isinstance(ckpt, dict) else 0
                best_val_loss = ckpt.get("best_val_loss", float("inf")) if isinstance(ckpt, dict) else float("inf")
                best_val_pearson = ckpt.get("best_val_pearson", float("-inf")) if isinstance(ckpt, dict) else float("-inf")
                logger.info(f"Resuming training from checkpoint {resume_path}, starting at epoch {start_epoch}")
            except Exception as e:
                logger.warning(f"Failed to load resume checkpoint {resume_path}: {e}")
        else:
            logger.warning(f"Configured resume_checkpoint {resume_path} does not exist; starting from scratch")
    num_epochs = 50
    criterion = nn.SmoothL1Loss(beta=beta)
    patience = 10
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_state = None
    best_val_pearson = float("-inf")
    train_losses = []
    val_losses = []
    train_mses = []
    val_mses = []

    # Split train/val (support dynamic splitting)
    val_split = getattr(cfg.data, "val_split", 0.2)
    num_samples = len(dataset_noaug)
    num_val = int(num_samples * val_split)
    num_train = num_samples - num_val

    # dynamic_split controls whether we resample the train/val indices each epoch.
    # Options: 'none' (default) -> deterministic one-time split; 'epoch' -> resample each epoch.
    dynamic_split = getattr(cfg.data, "dynamic_split", "none")
    seed = getattr(cfg, "seed", None)

    # prepare index array and RNG; RNG allows reproducible but changing splits when desired
    indices = np.arange(num_samples)
    rng = np.random.default_rng(seed)

    if dynamic_split == "none":
        # deterministic one-time split
        rng.shuffle(indices)
        train_idx = indices[:num_train].tolist()
        val_idx = indices[num_train:].tolist()
        train_dataset = torch.utils.data.Subset(dataset_aug, train_idx)
        val_dataset = torch.utils.data.Subset(dataset_noaug, val_idx)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_batch,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_batch,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        # For dynamic_split == 'epoch', we will resample indices each epoch and recreate loaders.
        # Create placeholders for train_loader/val_loader; they'll be (re)created inside epoch loop.
        train_loader = None
        val_loader = None

    logger.info(
        f"Training model with parameters:\n batch size {batch_size}, sigmoid steepness {k}, dropout probability {p_dropout}, post_dropout probability {post_dropout},"
        f"optimizer Adam, lr {lr}, weight decay {wd}, loss {criterion}, beta {beta}, num_epochs {num_epochs}, early stopping patience {patience}"
    )
    logger.info(f"Using augmentation: swap_prob {aug_swap_prob}, noise_std {aug_noise_std}, emb_noise_std {emb_noise_std}, scalar_jitter {scalar_jitter}."
                f"Using split strategy: {dynamic_split}, using label_jitter {label_jitter} on training targets.")
    logger.info(f"Model type: {model_type}, pin_memory: {pin_memory}, num_workers: {num_workers}")

    for epoch in range(start_epoch, num_epochs):
        # Optionally resample train/validation split per-epoch
        if dynamic_split == "epoch":
            rng.shuffle(indices)
            train_idx = indices[:num_train].tolist()
            val_idx = indices[num_train:].tolist()
            train_dataset = torch.utils.data.Subset(dataset_aug, train_idx)
            val_dataset = torch.utils.data.Subset(dataset_noaug, val_idx)
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_batch,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_batch,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )

        # Training phase
        model.train()
        train_loss = 0.0
        train_mse_total = 0.0
        for batch_x1d_left, batch_x1d_right, batch_x2d, batch_spec_left, batch_spec_right, batch_scalar, batch_y, batch_mask1d, batch_mask2d in train_loader:
            # move data to GPU if available
            batch_x1d_left = batch_x1d_left.to(device)
            batch_x1d_right = batch_x1d_right.to(device)
            batch_x2d = batch_x2d.to(device)
            batch_spec_left = batch_spec_left.to(device)
            batch_spec_right = batch_spec_right.to(device)
            batch_scalar = batch_scalar.to(device)
            batch_y = batch_y.to(device)
            # apply label jitter (only during training) to smooth targets
            if label_jitter is not None and float(label_jitter) > 0.0:
                batch_y = (batch_y + torch.randn_like(batch_y) * float(label_jitter)).clamp(0.0, 1.0)
            # masks are boolean tensors of shape (batch, time)
            batch_mask1d = batch_mask1d.to(device)
            batch_mask2d = batch_mask2d.to(device)

            optimizer.zero_grad()
            if model_type == "simple":
                outputs = model(
                    batch_x1d_left,
                    batch_x1d_right,
                    batch_x2d,
                    batch_spec_left,
                    batch_spec_right,
                    batch_scalar,
                    mask1d=batch_mask1d,
                    mask2d=batch_mask2d,
                )
            else:
                outputs = model(
                    batch_x1d_left,
                    batch_x1d_right,
                    batch_x2d,
                    batch_spec_left,
                    batch_spec_right,
                    batch_scalar,
                    mask1d=batch_mask1d,
                    mask2d=batch_mask2d,
                )
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_scalar.size(0) # average loss * batch size = total loss
            # accumulate sum-squared error for MSE calculation
            batch_mse_sum = F.mse_loss(outputs, batch_y, reduction="sum").item()
            train_mse_total += batch_mse_sum
        train_loss /= num_train
        train_losses.append(train_loss)
        # finalize train MSE (mean over all samples)
        train_mse = train_mse_total / num_train
        train_mses.append(train_mse)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_mse_total = 0.0
        # collect preds/targets to compute Pearson correlation on validation set
        val_preds = []
        val_trues = []
        with torch.no_grad():
            for val_x1d_left, val_x1d_right, val_x2d, val_spec_left, val_spec_right, val_scalar, val_y, val_mask1d, val_mask2d in val_loader:
                # move data to GPU if available
                val_x1d_left = val_x1d_left.to(device)
                val_x1d_right = val_x1d_right.to(device)
                val_x2d = val_x2d.to(device)
                val_spec_left = val_spec_left.to(device)
                val_spec_right = val_spec_right.to(device)
                val_scalar = val_scalar.to(device)
                val_y = val_y.to(device)
                val_mask1d = val_mask1d.to(device)
                val_mask2d = val_mask2d.to(device)

                if model_type == "simple":
                    val_outputs = model(
                        val_x1d_left,
                        val_x1d_right,
                        val_x2d,
                        val_spec_left,
                        val_spec_right,
                        val_scalar,
                        mask1d=val_mask1d,
                        mask2d=val_mask2d,
                    )
                else:
                    val_outputs = model(
                        val_x1d_left,
                        val_x1d_right,
                        val_x2d,
                        val_spec_left,
                        val_spec_right,
                        val_scalar,
                        mask1d=val_mask1d,
                        mask2d=val_mask2d,
                    )
                val_loss += criterion(val_outputs, val_y).item() * val_scalar.size(0) # average loss * batch size = total loss
                # accumulate sum-squared error for validation MSE
                val_mse_total += F.mse_loss(val_outputs, val_y, reduction="sum").item()
                # accumulate preds and trues for Pearson
                vals = val_outputs.detach().cpu().squeeze(-1).tolist()
                trues = val_y.detach().cpu().squeeze(-1).tolist()
                # ensure lists
                if isinstance(vals, float):
                    vals = [vals]
                if isinstance(trues, float):
                    trues = [trues]
                val_preds.extend(vals)
                val_trues.extend(trues)
        val_loss /= num_val
        val_losses.append(val_loss)
        # finalize val MSE
        val_mse = val_mse_total / num_val
        val_mses.append(val_mse)
        # compute Pearson correlation for validation predictions
        try:
            if len(val_preds) > 1:
                val_pearson = float(np.corrcoef(np.array(val_trues), np.array(val_preds))[0, 1])
            else:
                val_pearson = 0.0
            if not np.isfinite(val_pearson):
                val_pearson = 0.0
        except Exception:
            val_pearson = 0.0

        logger.info(
            f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, "
            f"Train MSE: {train_mse:.6f}, Val MSE: {val_mse:.6f}, Val Pearson: {val_pearson:.4f}",
        )

        # Early stopping check: require BOTH val loss decrease AND Pearson correlation increase
        if (val_loss < best_val_loss) and (val_pearson > best_val_pearson):
            best_val_loss = val_loss
            best_val_pearson = val_pearson
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                # Save a checkpoint including optimizer state so training can be resumed
                ckpt_path = f"{cfg.data.dataset}.train.{model.__class__.__name__}.earlystop.epoch{epoch+1}.pth"
                try:
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "best_val_loss": best_val_loss,
                            "best_val_pearson": best_val_pearson,
                        },
                        ckpt_path,
                    )
                    logger.info(f"Saved early-stop checkpoint to {ckpt_path}")
                except Exception as e:
                    logger.warning(f"Failed to save early-stop checkpoint: {e}")
                break

        # Periodic checkpointing every 5 epochs
        if (epoch + 1) % 5 == 0:
            ckpt_path = f"{cfg.data.dataset}.train.{model.__class__.__name__}.epoch{epoch+1}.pth"
            try:
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "best_val_loss": best_val_loss,
                        "best_val_pearson": best_val_pearson,
                    },
                    ckpt_path,
                )
                logger.info(f"Saved checkpoint to {ckpt_path}")
            except Exception as e:
                logger.warning(f"Failed to save checkpoint at epoch {epoch+1}: {e}")

    # Plot losses
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.savefig(f"{cfg.data.dataset}.train.{model.__class__.__name__}.loss_curve.png")
    plt.close()

    # Save model
    model_path = f"{cfg.data.dataset}.train.{model.__class__.__name__}.pth"
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    logger.info(f"Using device: {device}")

    num_scalar_features = 3
    # Use the same 1D channel size used for training so Conv1d layers are valid
    num_1d_channels = 512
    num_2d_channels = 2  # MFCCs in stereo
    k = 1.0
    p_dropout = 0.1
    # Select model type consistent with training config
    model_type = getattr(cfg.baseline, "model_type", "simple")
    if model_type == "simple":
        proj_dim = getattr(cfg.baseline, "proj_dim", 64)
        # keep inference dropout the same small post-dropout used during training
        post_dropout = 0.05
        model = simple_meanpool_scalar_mlp(
            num_1d_channels, num_scalar_features, proj_dim=proj_dim, k=k, p_dropout=p_dropout, post_dropout=post_dropout
        )
    else:
        model = multimodal_conv_mlp(num_1d_channels, num_2d_channels, num_scalar_features, k=k, p_dropout=p_dropout)

    # Find all checkpoint files produced during training and run inference for each.
    # Checkpoints follow the pattern: {dataset}.train.{ModelClass}*.pth
    ckpt_pattern = f"{cfg.data.dataset}.train.{model.__class__.__name__}*.pth"
    ckpt_dir = Path(".")
    ckpt_paths = sorted(ckpt_dir.glob(ckpt_pattern))

    # If no checkpoints found, fall back to the final model path if present
    if not ckpt_paths:
        final_path = Path(f"{cfg.data.dataset}.train.{model.__class__.__name__}.pth")
        if final_path.exists():
            ckpt_paths = [final_path]
        else:
            logger.error(f"No checkpoint files found matching {ckpt_pattern} and final model not found at {final_path}")
            return

    logger.info(f"Found {len(ckpt_paths)} checkpoint(s) for inference")

    # choose split for inference; allow cfg.split to specify otherwise default to 'valid'
    split = getattr(cfg, "split", "valid")
    logger.info(f"Preparing inference data for split={split}")

    # mfcc directories differ between train/valid on the host; mirror train logic
    if split == "train":
        mfcc_dir = "/mnt/d/cadenza_extracted_features/full_batch_cmvn_mfccs/mfcc/"
        spectral_centroid_dir = "/mnt/d/cadenza_extracted_features/spectral-centroid-train/centroid/"
        spectral_rolloff_dir = "/mnt/d/cadenza_extracted_features/spectral-rolloff-train/spectral-rolloff/"
    else:
        mfcc_dir = "/mnt/d/cadenza_extracted_features/mfccs_cmvn_valid/mfcc/"
        spectral_centroid_dir = "/mnt/d/cadenza_extracted_features/spectral-centroid-valid/centroid/"
        spectral_rolloff_dir = "/mnt/d/cadenza_extracted_features/spectral-rolloff-valid/spectral-rolloff/"

    # load MFCCs (each file is a json with a single-column of signals)
    x2d_dfs = []
    for file in os.listdir(mfcc_dir):
        if file.endswith(".json"):
            mfcc_path = os.path.join(mfcc_dir, file)
            mfcc_df = pd.read_json(mfcc_path)
            x2d_dfs.append(mfcc_df)
    x2d_df = pd.concat(x2d_dfs, axis=1)
    # Append spectral centroid/rolloff for inference if available
    if os.path.isdir(spectral_centroid_dir):
        centroid_dfs = []
        for file in os.listdir(spectral_centroid_dir):
            if file.endswith('.json'):
                centroid_path = os.path.join(spectral_centroid_dir, file)
                try:
                    cdf = pd.read_json(centroid_path)
                    centroid_dfs.append(cdf)
                except Exception:
                    logger.warning(f"Failed to read centroid file {centroid_path}")
        if centroid_dfs:
            centroid_df = pd.concat(centroid_dfs, axis=1)
            centroid_df = centroid_df.reindex(columns=x2d_df.columns)
            missing = list(set(x2d_df.columns) - set(centroid_df.columns))
            if missing:
                logger.warning(
                    "Spectral centroid data missing for %d signals; these will be filled with NaN: %s",
                    len(missing), str(missing[:5]),
                )
            x2d_df = pd.concat([x2d_df, centroid_df], axis=0)
    if os.path.isdir(spectral_rolloff_dir):
        rolloff_dfs = []
        for file in os.listdir(spectral_rolloff_dir):
            if file.endswith('.json'):
                rolloff_path = os.path.join(spectral_rolloff_dir, file)
                try:
                    rdf = pd.read_json(rolloff_path)
                    rolloff_dfs.append(rdf)
                except Exception:
                    logger.warning(f"Failed to read rolloff file {rolloff_path}")
        if rolloff_dfs:
            rolloff_df = pd.concat(rolloff_dfs, axis=1)
            rolloff_df = rolloff_df.reindex(columns=x2d_df.columns)
            missing = list(set(x2d_df.columns) - set(rolloff_df.columns))
            if missing:
                logger.warning(
                    "Spectral rolloff data missing for %d signals; these will be filled with NaN: %s",
                    len(missing), str(missing[:5]),
                )
            x2d_df = pd.concat([x2d_df, rolloff_df], axis=0)

    # load scalar features (STOI, whisper score, VAR dB)
    stoi_df = load_features(cfg, split, "stoi", None)
    whisper_df = load_features(cfg, split, "whisper", None)
    features_df = load_features(cfg, split, "features", "VAR (dB)")
    # normalize VAR (dB) with z-score using the split statistics
    mean_var = features_df["features"].mean()
    std_var = features_df["features"].std()
    features_df["features"] = (features_df["features"] - mean_var) / std_var

    merged_df = stoi_df.merge(whisper_df[["signal", "whisper"]], on="signal")
    merged_df = merged_df.merge(features_df[["signal", "features"]], on="signal")
    # keep only signals that have MFCCs available
    merged_df = merged_df[merged_df["signal"].isin(x2d_df.columns)]
    # Quick sanity check for expected sizes
    expected = 8802 if split == "train" else 1175
    if len(merged_df) != x2d_df.shape[1]:
        logger.warning(
            f"Mismatch in number of samples between scalar and 2d features: merged_df={len(merged_df)} x2d_cols={x2d_df.shape[1]}"
        )
    assert len(merged_df) == expected, f"Expected {expected} samples for split='{split}' but got {len(merged_df)}"

    # whisper embeddings dir (same as training)
    whisper_emb_dir = Path(cfg.data.cadenza_data_root) / cfg.data.dataset / "embeddings"

    # Create streaming dataset and dataloader used for inference
    dataset = EmbeddingDataset(merged_df, whisper_emb_dir, x2d_df, system=cfg.baseline.system)
    inf_batch_size = getattr(cfg, "inference_batch_size", 32)
    # inference dataloader: use the same worker/pin settings
    inf_num_workers = min(8, max(1, (os.cpu_count() or 4) - 2))
    inf_pin_memory = True if device.type == "cuda" else False
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=inf_batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=inf_num_workers,
        pin_memory=inf_pin_memory,
    )

    # Run inference for each checkpoint and save predictions per-checkpoint
    for ckpt_path in ckpt_paths:
        logger.info(f"Running inference using checkpoint {ckpt_path}...")

        # load checkpoint (support both raw state_dict and dict with 'model_state')
        try:
            ckpt = torch.load(str(ckpt_path), map_location=device)
        except Exception as e:
            logger.warning(f"Failed to load checkpoint {ckpt_path}: {e}")
            continue

        if isinstance(ckpt, dict) and "model_state" in ckpt:
            state_dict = ckpt["model_state"]
        else:
            state_dict = ckpt

        try:
            model.load_state_dict(state_dict)
        except Exception as e:
            logger.warning(f"Failed to load state_dict from {ckpt_path}: {e}")
            continue

        model.to(device)
        model.eval()

        outputs = []
        with torch.no_grad():
            for x1d_left, x1d_right, x2d_batch, spec_left_batch, spec_right_batch, scalar_batch, y_dummy, mask1d_left, mask1d_right, mask2d in dataloader:
                x1d_left = x1d_left.to(device)
                x1d_right = x1d_right.to(device)
                x2d_batch = x2d_batch.to(device)
                spec_left_batch = spec_left_batch.to(device)
                spec_right_batch = spec_right_batch.to(device)
                scalar_batch = scalar_batch.to(device)
                mask1d_left = mask1d_left.to(device)
                mask1d_right = mask1d_right.to(device)
                mask2d = mask2d.to(device)

                out = model(
                    x1d_left,
                    x1d_right,
                    x2d_batch,
                    spec_left_batch,
                    spec_right_batch,
                    scalar_batch,
                    mask1d_left=mask1d_left,
                    mask1d_right=mask1d_right,
                    mask2d=mask2d,
                )

                values = out.detach().cpu().squeeze(-1).tolist()
                if isinstance(values, float):
                    values = [values]
                outputs.extend(values)

        # attach predictions to a copy of merged_df (keep original order)
        df_out = merged_df.reset_index(drop=True).copy()
        if len(outputs) < len(df_out):
            logger.warning(f"Checkpoint {ckpt_path}: got {len(outputs)} outputs but expected {len(df_out)}; truncating or padding with NaN")
        # assign predictions (truncate if more outputs)
        preds = outputs[: len(df_out)] + [float('nan')] * max(0, len(df_out) - len(outputs))
        df_out["predicted_correctness"] = preds

        out_name = f"{cfg.data.dataset}.{split}.{model.__class__.__name__}.{ckpt_path.stem}.inference.csv"
        df_out.to_csv(out_name, index=False)
        logger.info(f"Saved inference results for checkpoint {ckpt_path} to {out_name}")


if __name__ == "__main__":
    run_train_model()
    # run_inference()
