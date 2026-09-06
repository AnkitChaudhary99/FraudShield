import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
import warnings

# Suppress librosa warnings about PySoundFile if standard backend fails
warnings.filterwarnings('ignore', category=UserWarning)

class AntiSpoofingCNN(nn.Module):
    """
    A simple CNN architecture for detecting AI-generated/spoofed audio.
    Expects input of shape (batch, channels, height, width).
    Specifically for mel-spectrograms: (1, 1, n_mels, time_steps).
    """
    def __init__(self):
        super(AntiSpoofingCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 1) # Binary classification: 0 (genuine) to 1 (spoofed)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.adaptive_pool(x)
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

def extract_features(audio_path, sr=16000, n_mels=128):
    """
    Extracts Mel-spectrogram features from an audio file.
    """
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        # Pad or truncate to ensure consistent length for simplicity (e.g., 5 seconds)
        max_length = sr * 5
        if len(y) > max_length:
            y = y[:max_length]
        else:
            y = np.pad(y, (0, max_length - len(y)), mode='constant')

        melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        log_melspec = librosa.power_to_db(melspec, ref=np.max)
        
        # Add channel and batch dimensions: (1, 1, n_mels, time_steps)
        tensor_features = torch.tensor(log_melspec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return tensor_features
    except Exception as e:
        print(f"Error processing audio in anti_spoofing: {e}")
        return None

def analyze_audio_spoofing(audio_path):
    """
    Analyzes the audio and returns a spoofing score.
    Score closer to 1.0 indicates high likelihood of being a deepfake/spoof.
    """
    # Initialize model
    model = AntiSpoofingCNN()
    
    # In a real scenario, we would load trained weights here:
    # model.load_state_dict(torch.load("path_to_weights.pth"))
    # For now, we will use the randomly initialized model to provide a functional pipeline.
    model.eval()
    
    features = extract_features(audio_path)
    if features is None:
        return 0.5 # Neutral fallback
    
    with torch.no_grad():
        score = model(features)
        
    return score.item()

if __name__ == "__main__":
    # Test with dummy data if run independently
    print("Anti-Spoofing Module Loaded.")
