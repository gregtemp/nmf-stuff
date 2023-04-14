import numpy as np
import nimfa
import librosa
from sklearn.neighbors import NearestNeighbors
import soundfile as sf
from custom_knn import custom_knn_search

# Load the source and target audio files
source_audio_file = 'voice.wav'
target_audio_file = 'drum.wav'
output_file = 'output.wav'

sr = 48000  # sample rate
rank = 50
fft_size = 2048
hop_size = 512
diff_knn_amount = 4



# Read the audio files
print("Loading audio files...")
y_source, sr_source = librosa.load(source_audio_file, sr=sr)
y_target, sr_target = librosa.load(target_audio_file, sr=sr)

print("Computing STFT...")
D_source = np.abs(librosa.stft(y_source, n_fft=fft_size, hop_length=hop_size))
D_target = np.abs(librosa.stft(y_target, n_fft=fft_size, hop_length=hop_size))

print("Applying NMF...")
nmf_source = nimfa.Nmf(D_source, rank=rank, method='lsnmf')
nmf_target = nimfa.Nmf(D_target, rank=rank, method='lsnmf')

fit_source = nmf_source()
fit_target = nmf_target()

W_source = np.asarray(fit_source.basis())
H_source = np.asarray(fit_source.coef())

W_target = np.asarray(fit_target.basis())
H_target = np.asarray(fit_target.coef())

print("Finding nearest neighbors...")
indices_W = custom_knn_search(W_source, W_target, reuse_range=diff_knn_amount, max_tries=5)
W_target_matched = W_source[:, indices_W.flatten()]

indices_H = custom_knn_search(H_source, H_target, reuse_range=diff_knn_amount, max_tries=5)
H_target_matched = H_source[:, indices_H.flatten()]

print("Combining matched basis vectors and activation coefficients...")
S_new = np.dot(W_target + W_target_matched, H_target + H_target_matched)

print("Performing inverse STFT...")
y_new = librosa.istft(S_new, hop_length=hop_size)

print("Saving output audio file...")
sf.write(output_file, y_new, sr_target)

print("Done.")
