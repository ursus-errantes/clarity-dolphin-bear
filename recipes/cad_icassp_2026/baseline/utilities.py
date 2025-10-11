import numpy as np
import scipy
import matplotlib.pyplot as plt
import torch

from train_model import mlp_scalar_features

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


def investigate_model_weights():
    """Investigate the weights of the trained MLP model."""
    # Load model weights
    model_path = "./exp/cadenza_data.train.mlp_scalar_features.pth"
    model_weights = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Extract weights and biases from the first layer
    first_layer_weight = model_weights['mlp.0.weight'].numpy()  # shape (64, num_scalar_features)
    first_layer_bias = model_weights['mlp.0.bias'].numpy()      # shape (64,)

    # Plot magnitude of weights
    plt.figure(figsize=(10, 6))
    plt.imshow(np.abs(first_layer_weight), aspect='auto', cmap='viridis')
    plt.colorbar(label='Weight Magnitude')
    plt.xlabel('Input Features')
    plt.ylabel('Neurons in First Layer')
    plt.title('Magnitude of Weights of the First Layer of MLP')
    plt.show()
    
    # Print the biases of the first layer
    print("Biases of the first layer:")
    print(first_layer_bias)

    # Print mean weight magnitude across the features
    mean_weights = np.mean(np.abs(first_layer_weight), axis=0)
    print("Mean weights across the features:")
    print(mean_weights)

    # Compare gradient importance for each feature
    x = torch.tensor([[0.5, 0.5, 0.5]], requires_grad=True)  # Example input with 3 features
    model = mlp_scalar_features(num_scalar_features=3)
    model.load_state_dict(model_weights)
    model.eval()
    y = model(x)
    y.backward()

    grad_importance = (x.grad * x).detach().cpu().numpy().flatten()
    plt.bar(["Feature 1", "Feature 2", "Feature 3"], np.abs(grad_importance))
    plt.title("Feature relevance via grad × input")
    plt.show()
