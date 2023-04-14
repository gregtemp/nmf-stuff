# limit amount of threads so the computer doesn't catch fire
import os
os.environ['OMP_NUM_THREADS'] = '2'  # Set the number of threads to 2 (adjust as needed)

import numpy as np
import nimfa
import librosa
import soundfile as sf

target_file = 'drum.wav'
source_file = 'voice.wav'
output_file = 'nmf_v2_output.wav'

def print_progress(fit, iteration):
    print(f"Iteration {iteration}: Objective value: {fit.objective}")

def audio_mosaic(target_file, source_file, n_iter=20):
    # Load target and source audio files
    print("loading files")
    target_audio, sr_target = librosa.load(target_file, sr=None, mono=True)
    source_audio, sr_source = librosa.load(source_file, sr=None, mono=True)

    # Compute the short-time Fourier transform (STFT) of target and source recordings
    print("compute stft")
    target_stft = librosa.stft(target_audio)
    source_stft = librosa.stft(source_audio)

    # Set the target and source spectrograms as V and W, respectively
    V = np.abs(target_stft)
    W = np.abs(source_stft)

    # Initialize the activation matrix H randomly
    n, m = W.shape
    _, p = V.shape
    H = np.random.rand(m, p)

    # Perform NMF, updating only the activation matrix H
    print("start nmf")
    nmf = nimfa.Nmf(V, W=W, H=H, max_iter=n_iter, update='divergence', objective='div', callback=print_progress)
    nmf_fit = nmf()

    # Get the updated activation matrix H
    H_updated = nmf_fit.fit.H

    # Multiply the learned activation matrix H with the complex-valued source spectrogram
    print("multiply source by H")
    audio_mosaic_stft = source_stft.dot(H_updated)

    # Apply the inverse STFT to obtain the audio mosaic
    audio_mosaic = librosa.istft(audio_mosaic_stft)

    return audio_mosaic, sr_target




audio_mosaic, sample_rate = audio_mosaic(target_file, source_file)

# save file
print("saving output")
sf.write(output_file, audio_mosaic, sample_rate)