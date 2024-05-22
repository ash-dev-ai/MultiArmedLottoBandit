import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write as wav_write
import os

# Function to generate waves
def generate_wave(wave_type, amplitude, frequency, time):
    if wave_type == 'sine':
        return amplitude * np.sin(2 * np.pi * frequency * time)
    elif wave_type == 'square':
        return amplitude * np.sign(np.sin(2 * np.pi * frequency * time))
    elif wave_type == 'sawtooth':
        return amplitude * 2 * (time * frequency - np.floor(time * frequency + 0.5))
    else:
        return None

# Function to save wave as audio file
def save_audio(wave, sample_rate, file_path):
    wav_write(file_path, sample_rate, wave)

# Read the lottery data
df = pd.read_csv('Lottery_Powerball_Winning_Numbers__Beginning_2010(4).csv')
df['Winning Numbers List'] = df['Winning Numbers'].apply(lambda x: [int(num) for num in x.split()])

# Initialize DataFrame to store amplitude, frequencies, and wave data
wave_data_df = pd.DataFrame(columns=['Winning Numbers', 'Amplitude', 'Fixed Frequency', 'Average Frequency', 'Normalized Sum Frequency'])

# Create directories to save the files
output_dirs = {
    'plots': 'wave_plots',
    'audio': 'wave_audio',
    'numpy': 'wave_numpy'
}

for dir_path in output_dirs.values():
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Constants
fixed_frequency = 1  # Fixed frequency for all waves
normalization_factor = 10  # Frequency normalization factor
time = np.linspace(0, 1, 1000)  # Time array for plotting

# Loop through all the winning combinations
for idx, row in df.iterrows():
    winning_numbers = row['Winning Numbers List']
    amplitude = max(winning_numbers) - min(winning_numbers)
    average_frequency = np.mean(winning_numbers)
    sum_frequency = np.sum(winning_numbers)
    normalized_sum_frequency = sum_frequency / normalization_factor

    # Generate and save the waves
    for wave_type in ['sine', 'square', 'sawtooth']:
        for freq_name, frequency in zip(['Fixed', 'Average', 'Normalized Sum'], [fixed_frequency, average_frequency, normalized_sum_frequency]):
            wave = generate_wave(wave_type, amplitude, frequency, time)

            # Save as plot
            plt.figure()
            plt.plot(time, wave)
            plt.title(f"{wave_type.capitalize()} Wave (Frequency: {frequency:.2f})")
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plot_file_path = os.path.join(output_dirs['plots'], f"{idx}_{wave_type}_{freq_name}.png")
            plt.savefig(plot_file_path)
            plt.close()

            # Save as audio
            audio_file_path = os.path.join(output_dirs['audio'], f"{idx}_{wave_type}_{freq_name}.wav")
            save_audio(wave.astype(np.int16), 44100, audio_file_path)

            # Save as NumPy array
            numpy_file_path = os.path.join(output_dirs['numpy'], f"{idx}_{wave_type}_{freq_name}.npy")
            np.save(numpy_file_path, wave)

    # Add to DataFrame
    wave_data_df.loc[idx] = [winning_numbers, amplitude, fixed_frequency, average_frequency, normalized_sum_frequency]

# Save the DataFrame as a CSV file
wave_data_df.to_csv('wave_data_full.csv', index=False)
