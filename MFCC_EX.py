#!/usr/bin/env python
# coding: utf-8

# In[2]:


import librosa
import numpy as np
import soundfile as sf
import os

def enhance_audio(input_path, output_path):
    # Load the original noisy audio
    noisy_audio, sr = librosa.load(input_path, sr=None)

    # Calculate the magnitude spectrum of the noisy audio
    noisy_spectrum = np.abs(librosa.stft(noisy_audio))

    # Estimate the noise profile
    noise_profile = np.mean(noisy_spectrum, axis=1)

    # Expand the noise profile to match the shape of the noisy spectrum
    expanded_noise_profile = np.tile(noise_profile, (noisy_spectrum.shape[1], 1)).T

    # Apply spectral subtraction
    alpha = 2.0  # Scaling factor for noise subtraction
    clean_spectrum = np.maximum(noisy_spectrum - alpha * expanded_noise_profile, 0)

    # Estimate the power spectrum of the noisy audio (for Wiener filter)
    noisy_power_spectrum = np.abs(noisy_spectrum) ** 2

    # Estimate the power spectrum of the clean audio (for Wiener filter)
    clean_power_spectrum = np.abs(clean_spectrum) ** 2

    # Estimate the Wiener filter coefficients
    wiener_filter = clean_power_spectrum / (clean_power_spectrum + noise_profile[:, np.newaxis]**2)

    # Apply the Wiener filter to the noisy audio
    enhanced_spectrum = wiener_filter * noisy_power_spectrum

    # Synthesize the enhanced speech signal
    enhanced_audio = librosa.istft(np.sqrt(enhanced_spectrum) * np.exp(1.0j * np.angle(noisy_spectrum)))

    # Save the enhanced audio to a file using soundfile
    sf.write(output_path, enhanced_audio, sr)

# Directory containing the noisy audio files
input_directory = 'C:/Users/ABHISHEK/program_1/depression_dir'

# Directory to save the enhanced audio files
output_directory = 'C:/Users/ABHISHEK/program_1/depression_dir_enhanced'

# List all audio files in the input directory
audio_files = [f for f in os.listdir(input_directory) if f.endswith('.wav')]

for audio_file in audio_files:
    input_path = os.path.join(input_directory, audio_file)
    output_path = os.path.join(output_directory, os.path.splitext(audio_file)[0] + '_enhanced.wav')
    enhance_audio(input_path, output_path)


# In[6]:


import os
import librosa
import numpy as np
from scipy.fftpack import dct
import soundfile as sf

# Directory containing the audio files
input_directory = 'C:/Users/ABHISHEK/program_1/depression_dir_enhanced'
output_directory = 'C:/Users/ABHISHEK/program_1/mfcc_coff'

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# List all audio files in the input directory
audio_files = [f for f in os.listdir(input_directory) if f.endswith('.wav')]

# Parameters
num_ceps = 12  # Number of cepstral coefficients to keep
NFFT = 512
num_filters = 26
frame_size = 0.025  # Frame size in seconds (typically 25 ms)
frame_stride = 0.010  # Frame stride in seconds (typically 10 ms)

for audio_file in audio_files:
    # Load the audio
    audio, sr = librosa.load(os.path.join(input_directory, audio_file), sr=None)

    # Pre-emphasis
    def pre_emphasis(signal, alpha=0.97):
        return np.append(signal[0], signal[1:] - alpha * signal[:-1])

    preemphasized_signal = pre_emphasis(audio)

    # Framing
    frame_length, frame_step = int(sr * frame_size), int(sr * frame_stride)
    signal_length = len(preemphasized_signal)
    num_frames = int(np.ceil(float(signal_length - frame_length) / frame_step))

    frames = np.zeros((num_frames, frame_length))
    for i in range(num_frames):
        frames[i] = preemphasized_signal[i * frame_step:i * frame_step + frame_length]

    # Windowing (Hamming window)
    window = np.hamming(frame_length)
    frames *= window

    # FFT
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))

    # Mel Filter Bank
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sr / 2) / 700))
    mel_points = np.linspace(low_freq_mel, high_freq_mel, num_filters + 2)

    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bin_points = np.floor((NFFT + 1) * hz_points / sr).astype(int)

    filter_bank = np.zeros((num_filters, int(NFFT / 2) + 1))
    for m in range(1, num_filters + 1):
        filter_bank[m - 1, bin_points[m - 1]:bin_points[m]] = (hz_points[m] - hz_points[m - 1]) / (
                bin_points[m] - bin_points[m - 1])
        filter_bank[m - 1, bin_points[m]:bin_points[m + 1]] = (hz_points[m + 1] - hz_points[m]) / (
                bin_points[m + 1] - bin_points[m])

    mel_spectrum = np.dot(mag_frames, filter_bank.T)

    # Log and Robust Power Compression
    log_mel_spectrum = np.log(mel_spectrum + 1e-10)
    power_compressed_mel = np.power(log_mel_spectrum, 2.0)

    # DCT
    cepstral_coefficents = np.zeros((num_frames, num_ceps))
    for i in range(num_frames):
        cepstral_coefficents[i] = dct(power_compressed_mel[i])[:num_ceps]

    # Save MFCC features to a file
    output_file = os.path.join(output_directory, os.path.splitext(audio_file)[0] + '_mfcc.npy')
    np.save(output_file, cepstral_coefficents)


# In[8]:


import os
import librosa
import soundfile as sf
import numpy as np

# Input directory containing audio files
input_directory = 'C:/Users/ABHISHEK/program_1/depression_dir_enhanced'

# Output directory for saving extracted features
output_directory = 'C:/Users/ABHISHEK/OneDrive/Documents/Image processing/fundamental_freq'

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Process each audio file in the input directory
audio_files = [f for f in os.listdir(input_directory) if f.endswith('.wav')]

for audio_file in audio_files:
    # Load the audio signal
    audio, sr = librosa.load(os.path.join(input_directory, audio_file), sr=None)

    # Estimate F0 using librosa
    f0, voiced = librosa.piptrack(y=audio, sr=sr)

    # Extract the F0 values from the pitch tracker
    f0_values = []

    for i in range(f0.shape[1]):
        f0_frame = f0[:, i][voiced[:, i] > 0]
        if len(f0_frame) > 0:
            f0_values.extend(f0_frame)

    # Calculate statistics on the extracted F0 values (e.g., mean)
    if f0_values:
        f0_mean = np.mean(f0_values)
    else:
        f0_mean = 0.0  # Handle cases with no voiced frames

    # Create a new filename for the output
    output_filename = os.path.splitext(audio_file)[0] + '_mean_f0.csv'
    output_path = os.path.join(output_directory, output_filename)

    # Save the mean F0 value to a CSV file
    np.savetxt(output_path, [f0_mean], delimiter=',', fmt='%.2f')

print(f"Processed {audio_file} and saved mean F0 value to {output_path}")

print("Processing complete.")


# In[17]:


import numpy as np
import os

# Directory containing the .npy files
npy_directory = 'C:/Users/ABHISHEK/program_1/mfcc_coff'
output_directory = 'C:/Users/ABHISHEK/program_1/mfcc_csv'

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# List all .npy files in the directory
npy_files = [f for f in os.listdir(npy_directory) if f.endswith('.npy')]

for npy_file in npy_files:
    # Load MFCC coefficients from the .npy file
    mfcc = np.load(os.path.join(npy_directory, npy_file))

    # Create a new filename for the output CSV
    output_csv_file = os.path.splitext(npy_file)[0] + '.csv'
    output_csv_path = os.path.join(output_directory, output_csv_file)

    # Save the MFCC coefficients to a CSV file
    np.savetxt(output_csv_path, mfcc, delimiter=',', fmt='%.6f')

print(f"Processed {npy_file} and saved as {output_csv_file}")

print("Processing complete.")


# In[24]:


import os
import numpy as np

# Directory containing CSV files with MFCC coefficients
mfcc_directory = 'C:/Users/ABHISHEK/program_1/mfcc_csv'

# Directory containing CSV files with fundamental frequency values
fundamental_freq_directory = 'C:/Users/ABHISHEK/program_1/fundamental_freq'

# Output CSV file
output_csv_file = 'C:/Users/ABHISHEK/program_1/features.csv'

# Initialize arrays to store mean and variance values of MFCC coefficients
mfcc_mean = []
mfcc_variance = []

# Initialize arrays to store mean fundamental frequency values
fundamental_freq_mean = []

# Process each CSV file in the MFCC directory
for mfcc_file in os.listdir(mfcc_directory):
    if mfcc_file.endswith('.csv'):
        # Extract the base name of the MFCC file
        base_name = os.path.splitext(mfcc_file)[0]

        # Check if there is a matching fundamental frequency file
        fundamental_freq_file = os.path.join(fundamental_freq_directory, base_name + '_features.csv')

        if os.path.exists(fundamental_freq_file):
            # Load MFCC coefficients from the MFCC file
            mfcc_data = np.loadtxt(os.path.join(mfcc_directory, mfcc_file), delimiter=',')

            # Calculate mean and variance of MFCC coefficients
            mfcc_mean.append(np.mean(mfcc_data, axis=0))
            mfcc_variance.append(np.var(mfcc_data, axis=0))

            # Load mean fundamental frequency from the matching file
            fundamental_freq_data = np.loadtxt(fundamental_freq_file, delimiter=',')
            mean_value = np.mean(fundamental_freq_data)

            # Store the mean fundamental frequency value
            fundamental_freq_mean.append(mean_value)

# Combine all values into a single array
combined_features = np.column_stack((mfcc_mean, mfcc_variance, fundamental_freq_mean))

# Save the combined features to a CSV file
header = "MFCC_Mean, MFCC_Variance, Fundamental_Freq_Mean"
np.savetxt(output_csv_file, combined_features, delimiter=',', header=header, comments='')

print("Combined features saved to", output_csv_file)


# In[ ]:




