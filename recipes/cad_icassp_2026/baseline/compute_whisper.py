"""Compute the Whisper correctness scores."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import numpy as np
import torch
import whisper
from omegaconf import DictConfig
from torch.nn import Module
from tqdm import tqdm

from clarity.utils.file_io import read_jsonl, write_jsonl, write_signal
from recipes.cad_icassp_2026.baseline.shared_predict_utils import (
    load_mixture,
)
from recipes.cad_icassp_2026.baseline.transcription_scorer import SentenceScorer

logger = logging.getLogger(__name__)


def compute_asr_for_signal(
    cfg: DictConfig, record: dict, signal: np.ndarray, asr_model: Module
) -> float:
    """Compute the correctness score for a given signal.

    Args:

        cfg (DictConfig): configuration object.
        record (dict): the metadata dict for the signal.
        signal (np.ndarray): the signal to compute the score for.
        asr_model (Module): the ASR model to use for transcription.

    Returns:

        float: correctness score
    """
    reference = record["prompt"]
    logger.info(f"Computing ASR for signal: {record['signal']}")

    score_left = compute_correctness(
        signal[:, 0],
        cfg.data.sample_rate,
        reference,
        asr_model,
        cfg.baseline.contractions_file,
    )
    score_right = compute_correctness(
        signal[:, 1],
        cfg.data.sample_rate,
        reference,
        asr_model,
        cfg.baseline.contractions_file,
    )

    return np.max([score_left, score_right])


def compute_correctness(
    signal: np.ndarray,
    sample_rate: int,
    reference: str,
    asr_model: Module,
    contraction_file: str,
) -> float:
    """Compute the correctness score for a given signal.

    Args:
        signal (np.ndarray): the signal to compute the score for
        sample_rate (int): the sample rate of the signal
        reference (str): the reference transcription
        asr_model (Module): the ASR model to use for transcription
        contraction_file (str): path to the contraction file for the scorer

    Returns:
        float: correctness score.
    """
    scorer = SentenceScorer(contraction_file)

    # create a temporary file to store the signal as flac
    # for Whisper to open it
    path_temp = Path("temp.flac")
    write_signal(
        filename=path_temp, signal=signal, sample_rate=sample_rate, floating_point=False
    )

    # Run Whisper ASR
    hypothesis = asr_model.transcribe(
        str(path_temp),
        fp16=False,
        language="en",
        temperature=0.0,
        suppress_tokens=[-1],
        condition_on_previous_text=False,
        no_speech_threshold=0.0,
        logprob_threshold=-999.0,
        compression_ratio_threshold=100.0,
    )["text"]
    logger.info(f"Reference transcription: {reference}")
    logger.info(f"ASR transcription: {hypothesis}")

    # Score the transcription
    results = scorer.score([reference], [hypothesis])
    total_words = results.substitutions + results.deletions + results.hits

    # Delete temporal file
    Path(path_temp).unlink()

    return results.hits / total_words


def get_whisper_encoder_embeddings(
    asr_model: Module, signal: np.ndarray, sample_rate: int, device: str | None = None
) -> np.ndarray:
    """Return Whisper encoder outputs for a single-channel waveform as a numpy array.

    Args:
        asr_model: object returned by whisper.load_model(...)
        signal: 1-D numpy array (audio samples)
        sample_rate: sample rate of the signal
        device: optional device string ("cuda"/"cpu"). If None, auto-selects.

    Returns:
        numpy.ndarray with shape (seq_len, d_model)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve wrapper -> underlying nn.Module
    model = getattr(asr_model, "model", asr_model)
    model = model.to(device)
    model.eval()

    # Normalize/convert input
    if signal.dtype != np.float32:
        if np.issubdtype(signal.dtype, np.integer):
            signal = signal.astype(np.float32) / np.iinfo(signal.dtype).max
        else:
            signal = signal.astype(np.float32)

    # Resample to 16 kHz if necessary
    target_sr = 16000
    if sample_rate != target_sr:
        try:
            import torchaudio

            sig_t = torch.from_numpy(signal)
            sig_t = torchaudio.functional.resample(sig_t, orig_freq=sample_rate, new_freq=target_sr)
            audio = sig_t.float()
        except Exception:
            try:
                import librosa

                resampled = librosa.resample(signal, orig_sr=sample_rate, target_sr=target_sr)
                audio = torch.from_numpy(resampled).float()
            except Exception:
                # last resort: use scipy
                from scipy.signal import resample_poly

                gcd = np.gcd(sample_rate, target_sr)
                up = target_sr // gcd
                down = sample_rate // gcd
                resampled = resample_poly(signal, up, down)
                audio = torch.from_numpy(resampled).float()
    else:
        audio = torch.from_numpy(signal).float()

    audio = audio.to(device)

    # Make log-mel spectrogram (Whisper helper expects 1-D audio tensor)
    mel = whisper.log_mel_spectrogram(audio)
    mel = mel.unsqueeze(0).to(device)  # (1, n_mel, T)

    # original time frames (number of mel frames)
    orig_t = mel.shape[-1]

    # base.en model expects 3000 time frames (~30s)
    expected_len = 3000

    print("expected_len (frames):", expected_len)
    print("mel shape (n_mel, T):", mel.shape)

    with torch.no_grad():
        t = orig_t
        # If shorter than expected, pad; if longer, trim to expected_len
        if t < expected_len:
            logger.info("Padding mel from %d to %d frames", t, expected_len)
            mel_proc = torch.nn.functional.pad(mel, (0, expected_len - t))
        else:
            if t > expected_len:
                logger.info("Trimming mel from %d to %d frames", t, expected_len)
                mel_proc = mel[:, :, :expected_len]

        print("mel_proc shape (1, n_mel, T_proc):", mel_proc.shape)

        enc = model.encoder(mel_proc)

        if isinstance(enc, tuple):
            enc = enc[0]

        enc = enc.squeeze(0)  # (expected_len, d_model)

        # Slice to original length so returned embeddings match input length
        enc = enc[:orig_t].cpu().numpy()

    return enc


def run_asr_from_mixture(
    dataroot: Path, records: list, results_file: Path, cfg: DictConfig
) -> None:
    """Load the mixture signal for a given record.

    Args:

        dataroot (Path): the root path to the dataset.
        records (list): list of records to process.
        results_file (Path): path to the results file.
        cfg (DictConfig): configuration object.
    """
    # Prepare dnn models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    asr_model = whisper.load_model(cfg.baseline.whisper_version, device=device)

    # Iterate through the signals that need scoring
    for record in tqdm(records):
        signal_name = record["signal"]

        # Load mixture
        signal_to_whisper, _ = load_mixture(dataroot, record, cfg)

        # Compute ASR
        correct = compute_asr_for_signal(cfg, record, signal_to_whisper, asr_model)

        # Optionally compute and save encoder embeddings per channel
        if getattr(cfg.baseline, "save_embeddings", False):
            try:
                emb_dir = dataroot / "embeddings"
                emb_dir.mkdir(parents=True, exist_ok=True)

                # signal_to_whisper expected shape (n_samples, channels)
                left = signal_to_whisper[:, 0]
                right = signal_to_whisper[:, 1]

                emb_left = get_whisper_encoder_embeddings(asr_model, left, cfg.data.sample_rate, device=device)
                emb_right = get_whisper_encoder_embeddings(asr_model, right, cfg.data.sample_rate, device=device)
                print(f"emb_left shape: {emb_left.shape}, emb_right shape: {emb_right.shape}")

                np.save(emb_dir / f"{signal_name}.{cfg.baseline.system}.left.npy", emb_left)
                np.save(emb_dir / f"{signal_name}.{cfg.baseline.system}.right.npy", emb_right)
            except Exception:
                logger.exception("Failed to compute/save embeddings for %s", signal_name)

        # Results are appended to the results file to allow interruption
        result = {"signal": signal_name, f"{cfg.baseline.system}": correct}
        write_jsonl(str(results_file), [result])


# pylint: disable = no-value-for-parameter
@hydra.main(config_path="configs", config_name="config", version_base=None)
def run_compute_whisper(cfg: DictConfig) -> None:
    """Run the Whisper to compute correctness hits/total words."""
    assert cfg.baseline.name == "whisper"

    logger.info(f"Running {cfg.baseline.system} baseline on {cfg.split} set...")

    # Load the set of signal for which we need to compute scores
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

    # Iterate over the signals that need scoring
    logger.info(f"Computing scores for {len(records)} out of {total_records} signals")

    run_asr_from_mixture(dataroot, records, results_file, cfg)


if __name__ == "__main__":
    run_compute_whisper()
