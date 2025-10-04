"""Compute the STOI scores."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
from tqdm import tqdm

from clarity.utils.file_io import read_jsonl, write_jsonl
from clarity.utils.flac_encoder import read_flac_signal
from clarity.utils.signal_processing import resample
from recipes.cad_icassp_2026.baseline.shared_predict_utils import (
    load_vocals_and_accomp,
)

logger = logging.getLogger(__name__)


def compute_VAR_db_for_signal(
    cfg: DictConfig, record: dict, data_root: str, estimated_vocals: np.ndarray, estimated_accomp: np.ndarray
) -> float:
    """Compute the VAR (dB) for a given signal.

    Args:
        cfg (DictConfig): configuration object
        record (dict): the metadata dict for the signal
        data_root (str): root path to the dataset
        estimated_vocals (np.ndarray): estimated vocals signal
        estimated_accomp (np.ndarray): estimated accompaniment signal

    Returns:
        float: vocal-to-accompaniment ratio in dB
    """
    signal_name = record["signal"]

    # normalize
    estimated_vocals = estimated_vocals / (np.max(np.abs(estimated_vocals)) + 1e-10)
    estimated_accomp = estimated_accomp / (np.max(np.abs(estimated_accomp)) + 1e-10)

    # Compute vocal-to-accompaniment ratio in dB
    energy_vocals = torch.sum(torch.tensor(estimated_vocals, dtype=torch.float32) ** 2)
    energy_accomp = torch.sum(torch.tensor(estimated_accomp, dtype=torch.float32) ** 2)
    var_db = 10 * torch.log10(energy_vocals / energy_accomp)
    return var_db.item()


# pylint: disable = no-value-for-parameter
@hydra.main(config_path="configs", config_name="config", version_base=None)
def run_compute_features(cfg: DictConfig) -> None:
    """Compute various audio features."""
    assert cfg.baseline.name == "features"

    logger.info(f"Running {cfg.baseline.system} baseline on {cfg.split} set...")

    # Load the set of signal for which we need to compute features
    dataroot = Path(cfg.data.cadenza_data_root) / cfg.data.dataset

    dataset_filename = dataroot / "metadata" / f"{cfg.split}_metadata.json"

    with dataset_filename.open("r", encoding="utf-8") as fp:
        records = json.load(fp)

    total_records = len(records)
    # Load existing results file if present
    batch_str = (
        f".{cfg.baseline.batch}_{cfg.baseline.n_batches}"
        if cfg.baseline.n_batches > 1
        else ""
    )

    results_file = Path(
        f"{cfg.data.dataset}.{cfg.split}.{cfg.baseline.system}{batch_str}.jsonl"
    )
    results = read_jsonl(str(results_file)) if results_file.exists() else []
    results_index = {result["signal"]: result for result in results}

    # Find signals for which we don't have scores
    records = [
        record for record in records if record["signal"] not in results_index.keys()
    ]
    records = records[cfg.baseline.batch - 1 :: cfg.baseline.n_batches]

    # Prepare audio source separation model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    separation_model = HDEMUCS_HIGH_MUSDB_PLUS.get_model()
    separation_model.to(device)

    # Iterate over the signals that need scoring
    logger.info(f"Computing features for {len(records)} out of {total_records} signals")
    if cfg.baseline.separator.keep_vocals:
        logger.info("Saving estimated vocals. If exist, they will not be recomputed.")

    for record in tqdm(records):
        signal_name = record["signal"]
        # Load or estimate both vocals and accompaniment from processed signal
        estimated_vocals, estimated_accomp = load_vocals_and_accomp(
            dataroot, record, cfg, separation_model, device=device
        )
        var_db = compute_VAR_db_for_signal(cfg, record, dataroot, estimated_vocals, estimated_accomp)

        # Results are appended to the results file to allow interruption
        result = {"signal": signal_name, f"{cfg.baseline.system} VAR (dB)": var_db}
        write_jsonl(str(results_file), [result])


if __name__ == "__main__":
    run_compute_features()
