import torch
import librosa
import numpy as np
import soundfile as sf

target_file = 'drum.wav'
source_file = 'voice.wav'
output_file = 'nmf_v2_output.wav'

def kl_divergence(V, W, H):
    WH = W @ H
    WH_V = V / (WH.clamp(min=1e-10))
    return (V * torch.log(WH_V) - V + WH).sum()


def nmf_pytorch(V, W, H, n_iter=200):
    V = torch.tensor(V, dtype=torch.float)
    W = torch.tensor(W, dtype=torch.float, requires_grad=True)
    H = torch.tensor(H, dtype=torch.float, requires_grad=True)
    
    optimizer = torch.optim.Adam([H], lr=0.01)
    
    for i in range(n_iter):
        optimizer.zero_grad()
        loss = kl_divergence(V, W, H)
        loss.backward()
        optimizer.step()
        H.data.clamp_(min=0)
        print(f"Iteration {i+1}: Loss: {loss.item()}")

    return H.detach().numpy()

def audio_mosaic_pytorch(target_file, source_file, n_iter=200):
    # Load target and source audio files
    target_audio, sr_target = librosa.load(target_file, sr=None, mono=True)
    source_audio, sr_source = librosa.load(source_file, sr=None, mono=True)

    # Compute the short-time Fourier transform (STFT) of target and source recordings
    target_stft = librosa.stft(target_audio)
    source_stft = librosa.stft(source_audio)

    # Set the target and source spectrograms as V and W, respectively
    V = np.abs(target_stft)
    W = np.abs(source_stft)

    # Initialize the activation matrix H randomly
    n, m = W.shape
    _, p = V.shape
    H = np.random.rand(m, p)

    # Perform NMF with PyTorch, updating only the activation matrix H, and print progress after every iteration
    H_updated = nmf_pytorch(V, W, H, n_iter)

    # Multiply the learned activation matrix H with the complex-valued source spectrogram
    audio_mosaic_stft = source_stft.dot(H_updated)

    # Apply the inverse STFT to obtain the audio mosaic
    audio_mosaic = librosa.istft(audio_mosaic_stft)

    return audio_mosaic, sr_target


audio_mosaic, sample_rate = audio_mosaic_pytorch(target_file, source_file, n_iter=200)

# save file
print("saving output")
sf.write(output_file, audio_mosaic, sample_rate)