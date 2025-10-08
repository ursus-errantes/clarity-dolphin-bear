import numpy as np
import scipy

def normalize_audio(
    audio: np.ndarray,
    sr: int,
    target_level_db: float = -20.0,
    lowcut: float = 50.0,
    highcut: float = 15000.0,
    apply_compression: bool = True,
    compression_ratio: float = 4.0,
    eps: float = 1e-9,
) -> np.ndarray:

    # band limit
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = scipy.signal.butter(4, [low, high], btype="band")
    audio = scipy.signal.lfilter(b, a, audio)

    # rms norm.
    rms = np.sqrt(np.mean(audio**2) + eps)
    current_level_db = 20 * np.log10(rms + eps)
    gain_db = target_level_db - current_level_db
    gain = 10 ** (gain_db / 20)
    audio = audio * gain

    if apply_compression:
        threshold = 10 ** (target_level_db / 20)
        above_threshold = np.abs(audio) > threshold
        audio[above_threshold] = np.sign(audio[above_threshold]) * (
            threshold
            + (np.abs(audio[above_threshold]) - threshold) / compression_ratio
        )
    peak = np.max(np.abs(audio)) + eps
    if peak > 1.0:
        audio = audio / peak

    return audio