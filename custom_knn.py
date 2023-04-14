import numpy as np
import nimfa
import librosa
from sklearn.neighbors import NearestNeighbors


def custom_knn_search(source_matrix, target_matrix, reuse_range, max_tries=1000):
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

            tries += 1

        # If no suitable match is found within max_tries, use the best match found so far
        if tries == max_tries:
            matched_indices.append(best_index)
            used_indices.add(best_index)

    return np.array(matched_indices)
