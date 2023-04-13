import numpy as np
import nimfa
import librosa
import pywt
from sklearn.neighbors import NearestNeighbors
import soundfile as sf

# Load the source and target audio files
source_audio_file = 'guitar.wav'
target_audio_file = 'drum.wav'
output_file = 'output.wav'

# Read the audio files
y_source, sr_source = librosa.load(source_audio_file)
y_target, sr_target = librosa.load(target_audio_file)

# Compute the Continuous Wavelet Transform (CWT)
wavelet = 'morl'
scales = np.arange(1, 101)
D_source = np.abs(pywt.cwt(y_source, scales, wavelet)[0])
D_target = np.abs(pywt.cwt(y_target, scales, wavelet)[0])

# Apply NMF to the source and target wavelet transforms
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
W_target_matched = W_source[:, indices_W.flatten()]

# Find the closest matching slices in H_source for each slice in H_target
knn_H = NearestNeighbors(n_neighbors=1)
knn_H.fit(H_source.T)
_, indices_H = knn_H.kneighbors(H_target.T)
H_target_matched = H_source[:, indices_H.flatten()]

# Combine the matched basis vectors and activation coefficients
S_new = np.dot(W_target + W_target_matched, H_target + H_target_matched)

# Perform the inverse Continuous Wavelet Transform (iCWT)
y_new = pywt.icwt(S_new, np.zeros_like(S_new), wavelet, 1.0)

# save file
sf.write(output_file, y_new, sr_target)
