"""Shared utilities for STOI baseline prediction experiments."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from scipy.optimize import curve_fit
from scipy.signal import correlate
from torch.nn import Module

from clarity.utils.file_io import read_jsonl, read_signal, write_signal
from clarity.utils.signal_processing import resample
from clarity.utils.source_separation_support import separate_sources

logger = logging.getLogger(__name__)


class LogisticModel:
    """Class to represent a logistic mapping.

    Fits a logistic mapping from input values x to output values y.
    """

    params: np.ndarray | None = None  # The model params

    def _logistic_mapping(self, x, x_0, k):
        """Logistic function

        Args:
            x - the input value
            x_0 - logistic parameter: the x value of the logistic's midpoint
            k - logistic parameter: the growth rate of the curve

        Returns:
            The output of the logistic function.
        """
        return 100.0 / (1 + np.exp(-k * (x - x_0)))

    def fit(self, x, y):
        """Fit a mapping from x values to y values."""
        initial_guess = [0.5, 1.0]  # Initial guess for parameter values
        self.params, *_pcov = curve_fit(self._logistic_mapping, x, y, initial_guess)

    def predict(self, x):
        """Predict y values given x.

        Raises:
            TypeError: If the predict() method is called before fit().
        """
        if self.params is None:
            raise TypeError(
                "params is None. Logistic fit() must be called before predict()."
            )

        return self._logistic_mapping(x, self.params[0], self.params[1])


def input_align(
    reference: np.ndarray, processed: np.ndarray, fsamp: float = 10000
) -> tuple[np.ndarray, np.ndarray]:
    """Align the processed signal to the reference signal.
    Code based on the `evaluator.haspi.eb` but for variable sampling rate
    """
    # Match the length of the processed output to the reference for the purposes
    # of computing the cross-covariance
    reference_n = len(reference)
    processed_n = len(processed)
    min_sample_length = min(reference_n, processed_n)

    # Determine the delay of the output relative to the reference
    reference_processed_correlation = correlate(
        reference[:min_sample_length] - np.mean(reference[:min_sample_length]),
        processed[:min_sample_length] - np.mean(processed[:min_sample_length]),
        "full",
    )  # Matlab code uses xcov thus the subtraction of mean
    index = np.argmax(np.abs(reference_processed_correlation))
    delay = min_sample_length - index - 1

    # Back up 2 msec to allow for dispersion
    delay = np.rint(delay - 2 * fsamp / 1000.0).astype(int)  # Back up 2 ms

    # Align the output with the reference allowing for the dispersion
    if delay > 0:
        # Output delayed relative to the reference
        processed = np.concatenate((processed[delay:processed_n], np.zeros(delay)))
    else:
        # Output advanced relative to the reference
        processed = np.concatenate((np.zeros(-delay), processed[: processed_n + delay]))

    # Find the start and end of the noiseless reference sequence
    reference_abs = np.abs(reference)
    reference_max = np.max(reference_abs)
    reference_threshold = 0.001 * reference_max  # Zero detection threshold

    above_threshold = np.where(reference_abs > reference_threshold)[0]
    reference_n_above_threshold = above_threshold[0]
    reference_n_below_threshold = above_threshold[-1]

    # Prune the sequences to remove the leading and trailing zeros
    reference_n_below_threshold = min(reference_n_below_threshold, processed_n)

    return (
        reference[reference_n_above_threshold : reference_n_below_threshold + 1],
        processed[reference_n_above_threshold : reference_n_below_threshold + 1],
    )


def estimate_vocals(
    signal: np.ndarray, sample_rate: int, model: Module, device: str = "cpu"
) -> np.ndarray:
    """
    Estimate vocals from the input signal using the pre-trained model.

    Args:
        signal (torch.Tensor | np.ndarray): Input audio signal.
        sample_rate (int): Sample rate of the input signal.
        model (torch.nn.Module): Pre-trained source separation model.
        device (str): Device to run the model on ('cpu' or 'cuda').

    Returns:
        np.ndarray: Estimated vocals.
    """
    if not isinstance(signal, torch.Tensor):
        signal = torch.tensor(signal, dtype=torch.float32)

    if signal.ndim == 1:
        # HDemucs works with 2 channels, so we need to stack the signal
        # to create a stereo signal
        signal = torch.stack([signal, signal], dim=0)

    signal = signal.to(device)

    ref = signal.mean(0)
    signal = (signal - ref.mean()) / ref.std()
    ref = ref.cpu().detach().numpy()

    sources = separate_sources(
        model,
        signal[None],
        sample_rate=sample_rate,
        device=device,
    )[0]
    sources = sources * ref.std() + ref.mean()

    sources_list = model.sources
    sources = list(sources)

    audios = dict(zip(sources_list, sources))
    estimated_vocals = audios["vocals"]
    return estimated_vocals


def estimate_vocals_and_accomp(
    signal: np.ndarray, sample_rate: int, model: Module, device: str = "cpu"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate vocals and accompaniment from the input signal using the pre-trained model.

    Args:
        signal (torch.Tensor | np.ndarray): Input audio signal.
        sample_rate (int): Sample rate of the input signal.
        model (torch.nn.Module): Pre-trained source separation model.
        device (str): Device to run the model on ('cpu' or 'cuda').

    Returns:
        Tuple[np.ndarray, np.ndarray]: Estimated vocals and accompaniment.
    """
    if not isinstance(signal, torch.Tensor):
        signal = torch.tensor(signal, dtype=torch.float32)

    if signal.ndim == 1:
        # HDemucs works with 2 channels, so we need to stack the signal
        # to create a stereo signal
        signal = torch.stack([signal, signal], dim=0)

    signal = signal.to(device)

    ref = signal.mean(0)
    signal = (signal - ref.mean()) / ref.std()
    ref = ref.cpu().detach().numpy()

    sources = separate_sources(
        model,
        signal[None],
        sample_rate=sample_rate,
        device=device,
    )[0]
    sources = sources * ref.std() + ref.mean()

    sources_list = model.sources
    sources = list(sources)

    audios = dict(zip(sources_list, sources))
    estimated_vocals = audios["vocals"]
    print(estimated_vocals.shape)
    logger.info(f"estimated vocals max, mean, std: {np.max(estimated_vocals)}, {np.mean(estimated_vocals)}, {np.std(estimated_vocals)}")
    estimated_accomp = audios["drums"] + audios["bass"] + audios["other"]
    logger.info(f"estimated accompaniment max, mean, std: {np.max(estimated_accomp)}, {np.mean(estimated_accomp)}, {np.std(estimated_accomp)}")
    return estimated_vocals, estimated_accomp


def load_mixture(
    dataroot: Path, record: dict, cfg: DictConfig
) -> tuple[np.ndarray, float]:
    """Load the mixture signal for a given record.

    Args:

        dataroot (Path): Root path to the dataset.
        record (dict): Record containing signal metadata.
        cfg (DictConfig): Configuration object.

    Returns:

        tuple[np.ndarray, int]: Mixture signal and its sample rate.
    """
    signal_name = record["signal"]

    if cfg.baseline.reference == "processed":
        logger.info("Loading processed signal as reference for source separation")
        mix_signal_path = (
            dataroot / cfg.split / "signals" / f"{signal_name}.flac"
        )
    elif cfg.baseline.reference == "unprocessed":
        logger.info("Loading unprocessed signal as reference for source separation")
        mix_signal_path = (
            dataroot
            / cfg.split
            / "unprocessed"
            / f"{signal_name}_unproc.flac"
        )
    else:
        raise ValueError(f"Unknown reference type: {cfg.baseline.reference}")

    mixture = read_signal(
        mix_signal_path,
        sample_rate=cfg.data.sample_rate,
    )
    return mixture, cfg.data.sample_rate


def load_vocals(
    dataroot: Path, record: dict, cfg: DictConfig, separation_model, device="cpu"
) -> np.ndarray:
    """Load or compute estimated vocals for a given record.

    Args:
        dataroot (Path): Root path to the dataset.
        record (dict): Record containing signal metadata.
        cfg (DictConfig): Configuration object.
        separation_model: Pre-trained source separation model.
        device (str): Device to run the model on ('cpu' or 'cuda').

    Returns:
        np.ndarray: Estimated vocals signal.
    """

    signal_name = record["signal"]

    vocals_path = Path("est_vocals") / cfg.split / f"{signal_name}_est_vocals.wav"
    if not vocals_path.exists():
        signal, signal_sr = load_mixture(dataroot, record, cfg)
        if signal_sr != cfg.data.sample_rate:
            logger.info(f"resampling mixture signal to {cfg.data.sample_rate} Hz")
            signal = resample(signal, signal_sr, cfg.data.sample_rate)

        # Estimate vocals to create a processed signal
        estimated_vocals = estimate_vocals(
            signal.T,
            cfg.baseline.separator.sample_rate,
            separation_model,
            device=device,
        ).T

        if cfg.baseline.separator.keep_vocals:
            vocals_path.parent.mkdir(parents=True, exist_ok=True)
            write_signal(vocals_path, estimated_vocals, cfg.data.sample_rate)
    else:
        estimated_vocals = read_signal(vocals_path, cfg.data.sample_rate)

    return estimated_vocals


def load_vocals_and_accomp(
    dataroot: Path, record: dict, cfg: DictConfig, separation_model, device="cpu"
) -> tuple[np.ndarray, np.ndarray]:
    """Load or compute estimated vocals and accompaniment for a given record.

    Args:
        dataroot (Path): Root path to the dataset.
        record (dict): Record containing signal metadata.
        cfg (DictConfig): Configuration object.
        separation_model: Pre-trained source separation model.
        device (str): Device to run the model on ('cpu' or 'cuda').

    Returns:
        tuple[np.ndarray, np.ndarray]: Estimated vocals and accompaniment signals.
    """

    signal_name = record["signal"]

    vocals_path = Path("est_vocals") / cfg.split / f"{signal_name}_est_vocals.wav"
    accomp_path = Path("est_accomp") / cfg.split / f"{signal_name}_est_accomp.wav"
    if not vocals_path.exists() or not accomp_path.exists():
        signal, signal_sr = load_mixture(dataroot, record, cfg)
        if signal_sr != cfg.data.sample_rate:
            logger.info(f"resampling mixture signal to {cfg.data.sample_rate} Hz")
            signal = resample(signal, signal_sr, cfg.data.sample_rate)

        # Estimate vocals and accompaniment to create a processed signal
        estimated_vocals, estimated_accomp = estimate_vocals_and_accomp(
            signal.T,
            cfg.baseline.separator.sample_rate,
            separation_model,
            device=device,
        )
        estimated_vocals = estimated_vocals.T
        estimated_accomp = estimated_accomp.T

        if cfg.baseline.separator.keep_vocals:
            vocals_path.parent.mkdir(parents=True, exist_ok=True)
            write_signal(vocals_path, estimated_vocals, cfg.data.sample_rate)
        if cfg.baseline.separator.keep_accomp:
            accomp_path.parent.mkdir(parents=True, exist_ok=True)
            write_signal(accomp_path, estimated_accomp, cfg.data.sample_rate)
    else:
        estimated_vocals = read_signal(vocals_path, cfg.data.sample_rate)
        estimated_accomp = read_signal(accomp_path, cfg.data.sample_rate)

    return estimated_vocals, estimated_accomp


def load_dataset_with_score(cfg, split: str) -> pd.DataFrame:
    """Load dataset and add prediction scores.

    Args:

        cfg (DictConfig): Configuration object.
        split (str): Dataset split to load ('train' or 'valid')

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
    system_path = f"{cfg.data.dataset}.{split}.{cfg.baseline.system}.jsonl"
    system_score = read_jsonl(str(system_path))
    system_score_index = {
        record["signal"]: record[cfg.baseline.system] for record in system_score
    }
    for record in records:
        record[f"{cfg.baseline.system}"] = system_score_index[record["signal"]]

    return pd.DataFrame(records)
