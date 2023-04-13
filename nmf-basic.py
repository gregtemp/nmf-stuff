import numpy as np
import nimfa
import librosa
from sklearn.neighbors import NearestNeighbors
import soundfile as sf # for some reason librosa.output.write_wav doesnt work so lets use this


rank = 50
reuse_range=5

# Load the source and target audio files
source_audio_file = 'guitar.wav'
target_audio_file = 'drum.wav'
output_file = 'output.wav'

# Read the audio files
y_source, sr_source = librosa.load(source_audio_file)
y_target, sr_target = librosa.load(target_audio_file)

# Compute the magnitude spectrograms
D_source = np.abs(librosa.stft(y_source, n_fft=2048, hop_length=512))
D_target = np.abs(librosa.stft(y_target, n_fft=2048, hop_length=512))

# Apply NMF to the source and target spectrograms

nmf_source = nimfa.Nmf(D_source, rank=rank, method='lsnmf')
nmf_target = nimfa.Nmf(D_target, rank=rank, method='lsnmf')

fit_source = nmf_source()
fit_target = nmf_target()

W_source = np.asarray(fit_source.basis())
H_source = np.asarray(fit_source.coef())

W_target = np.asarray(fit_target.basis())
H_target = np.asarray(fit_target.coef())

def custom_knn_search(source_matrix, target_matrix, reuse_range, max_tries=20):
    knn = NearestNeighbors(n_neighbors=1)
    knn.fit(source_matrix.T)

    used_indices = set()
    matched_indices = []

    for target_vec in target_matrix.T:
        best_index = None
        best_distance = float('inf')
        tries = 0

        while tries < max_tries:
            _, index = knn.kneighbors(target_vec.reshape(1, -1))
            index = index.flatten()[0]

            if index not in used_indices and all(abs(index - used_idx) > reuse_range for used_idx in used_indices):
                matched_indices.append(index)
                used_indices.add(index)
                break

            # Update the best match found so far
            distance = np.linalg.norm(target_vec - source_matrix[:, index])
            if distance < best_distance:
                best_distance = distance
                best_index = index
            print("tries and index", tries, index)
            tries += 1

        # If no suitable match is found within max_tries, use the best match found so far
        if tries == max_tries:
            matched_indices.append(best_index)
            used_indices.add(best_index)

    return np.array(matched_indices)



# Find the closest matching slices in W_source for each slice in W_target
indices_W = custom_knn_search(W_source, W_target, reuse_range)
W_target_matched = W_source[:, indices_W.flatten()]

# Find the closest matching slices in H_source for each slice in H_target
indices_H = custom_knn_search(H_source, H_target, reuse_range)
H_target_matched = H_source[:, indices_H.flatten()]

# Combine the matched basis vectors and activation coefficients
S_new = np.dot(W_target + W_target_matched, H_target + H_target_matched)

# Perform the inverse Short-Time Fourier Transform (iSTFT)
y_new = librosa.istft(S_new, hop_length=512)

# save file
sf.write(output_file, y_new, sr_target)