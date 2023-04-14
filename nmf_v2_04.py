import torch
import torch.nn as nn
import librosa
import numpy as np
import soundfile as sf



target_file = 'drum.wav'
source_file = 'guitar.wav'
output_file = 'nmf_v2_output-2.wav'

rep_r = 20
iterations = 100
learning_rate = 0.05
polyphony = 4

def kl_divergence(V, W, H):
    WH = W @ H
    WH_V = V / (WH.clamp(min=1e-10))
    return (V * torch.log(WH_V) - V + WH).sum()


def repetition_restriction(H, r, l, L):
    iterfac = 1 - float(l + 1) / L
    # Step 1: Avoid repeated activations
    maxpool = nn.MaxPool1d(kernel_size=2 * r + 1, stride=1, padding=r)
    MuH = maxpool(H.unsqueeze(0)).squeeze(0)
    mask = H < MuH
    H_new = H.clone()
    H_new[mask] = H[mask] * iterfac
    return H_new

def polyphony_restriction(R, p):
    _, top_indices = torch.topk(R, p, dim=0)
    mask = torch.zeros_like(R).scatter_(0, top_indices, 1).bool()
    P = torch.zeros_like(R)
    P[mask] = R[mask]
    return P



def nmf_pytorch(V, W, H, n_iter, r, p):
    V = torch.tensor(V, dtype=torch.float)
    W = torch.tensor(W, dtype=torch.float, requires_grad=True)
    H = torch.tensor(H, dtype=torch.float, requires_grad=True)
    
    optimizer = torch.optim.Adam([H], lr=learning_rate)
    
    for i in range(n_iter):
        optimizer.zero_grad()
        R = repetition_restriction(H, r, i, n_iter)
        H = polyphony_restriction(R, p)

        loss = kl_divergence(V, W, R)
        loss.backward()
        optimizer.step()
        H.data.clamp_(min=0)
        print(f"Iteration {i+1}: Loss: {loss.item()}")

    return H.detach().numpy()

def audio_mosaic_pytorch(target_file, source_file, n_iter, r, p):
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
    H_updated = nmf_pytorch(V, W, H, n_iter, r, p)

    # Multiply the learned activation matrix H with the complex-valued source spectrogram
    audio_mosaic_stft = source_stft.dot(H_updated)

    # Apply the inverse STFT to obtain the audio mosaic
    audio_mosaic = librosa.istft(audio_mosaic_stft)

    return audio_mosaic, sr_target


audio_mosaic, sample_rate = audio_mosaic_pytorch(target_file, source_file, iterations, rep_r, polyphony)

# save file
print("saving output")
sf.write(output_file, audio_mosaic, sample_rate)