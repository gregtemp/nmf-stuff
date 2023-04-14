import numpy as np
import nimfa
import librosa
from sklearn.neighbors import NearestNeighbors
import soundfile as sf

# Load the source and target audio files
source_audio_file = 'voice.wav'
target_audio_file = 'drum.wav'
output_file = 'output.wav'

# Read the audio files
y_source, sr_source = librosa.load(source_audio_file)
y_target, sr_target = librosa.load(target_audio_file)

# Compute the magnitude spectrograms
D_source = np.abs(librosa.stft(y_source, n_fft=2048, hop_length=512))
D_target = np.abs(librosa.stft(y_target, n_fft=2048, hop_length=512))

# Apply NMF to the source and target spectrograms
rank = 50
nmf_source = nimfa.Nmf(D_source, rank=rank, method='lsnmf')
nmf_target = nimfa.Nmf(D_target, rank=rank, method='lsnmf')

fit_source = nmf_source()
fit_target = nmf_target()

W_source = np.asarray(fit_source.basis())
H_source = np.asarray(fit_source.coef())

W_target = np.asarray(fit_target.basis())
H_target = np.asarray(fit_target.coef())

# Find the closest matching slices in W_source for each slice in W_target
knn_W = NearestNeighbors(n_neighbors=1)
knn_W.fit(W_source.T)
_, indices_W = knn_W.kneighbors(W_target.T)

n_extra_slices = 3

W_target_matched = np.zeros_like(W_target)
for col_idx in range(W_target.shape[1]):
    source_col_idx = indices_W.flatten()[col_idx]
    start_col_idx = max(col_idx - n_extra_slices, 0)
    end_col_idx = min(col_idx + n_extra_slices + 1, W_target.shape[1])
    W_target_matched[:, start_col_idx:end_col_idx] += W_source[:, source_col_idx].reshape(-1, 1)


# Find the closest matching slices in H_source for each slice in H_target
knn_H = NearestNeighbors(n_neighbors=1)
knn_H.fit(H_source.T)
_, indices_H = knn_H.kneighbors(H_target.T)


H_target_matched = np.zeros_like(H_target)
for row_idx in range(H_target.shape[0]):
    source_row_idx = indices_H.flatten()[row_idx]
    start_row_idx = max(row_idx - n_extra_slices, 0)
    end_row_idx = min(row_idx + n_extra_slices + 1, H_target.shape[0])
    if source_row_idx < H_source.shape[0]:
        H_target_matched[start_row_idx:end_row_idx, :] += H_source[source_row_idx, :].reshape(1, -1)


# Combine the matched basis vectors and activation coefficients
S_new = np.dot(W_target + W_target_matched, H_target + H_target_matched)

# Perform the inverse Short-Time Fourier Transform (iSTFT)
y_new = librosa.istft(S_new, hop_length=512)

# save file
sf.write(output_file, y_new, sr_target)
