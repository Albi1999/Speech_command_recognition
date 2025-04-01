import os
import librosa
import numpy as np
import librosa.display
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.partitioning import which_set

class SpeechPreprocessor:
    def __init__(self, dataset_path, output_path="Data/processed_dataset",
                 sample_rate=16000, n_mels=40, frame_size=0.025, frame_step=0.010):
        """
        Initializes the SpeechPreprocessor class.

        Args:
            dataset_path (str): Path to the dataset folder.
            output_path (str): Path to the output folder.
            sample_rate (int): Sampling rate of the audio files.
            n_mels (int): Number of Mel bands to generate.
            frame_size (float): Size of the frame window in seconds.
            frame_step (float): Size of the frame step in seconds.
        """
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.frame_size = frame_size
        self.frame_step = frame_step

        # Ensure output directories exist
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(f"{self.output_path}/train", exist_ok=True)
        os.makedirs(f"{self.output_path}/val", exist_ok=True)
        os.makedirs(f"{self.output_path}/test", exist_ok=True)

    def _get_spectrogram(self, filepath):
        """
        Converts a .wav file into a log Mel spectrogram.
        
        Args:
            filepath (str): Path to the .wav file.
            
        Returns:
            np.ndarray: Log Mel spectrogram of the audio file.
        """
        y, sr = librosa.load(filepath, sr=self.sample_rate)

        spectrogram = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=self.n_mels,
            hop_length=int(self.frame_step * sr),
            n_fft=int(self.frame_size * sr)
        )
        spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

        # Pad or truncate to fixed width
        target_width = 101
        if spectrogram.shape[1] < target_width:
            pad_width = target_width - spectrogram.shape[1]
            spectrogram = np.pad(spectrogram, ((0, 0), (0, pad_width)), mode="constant")
        elif spectrogram.shape[1] > target_width:
            spectrogram = spectrogram[:, :target_width]

        return spectrogram


    def process_audio_files(self):
        """
        Processes all audio files and applies dataset partitioning.
        The processed spectrograms are saved in the output directory.
        The dataset partitioning is done using the official method from the dataset.
        The processed dataset will have the following structure:
        processed_dataset/
        ├── train/ # Training set
        │   ├── class1/
        │   │   ├── file1.npy
        │   │   ├── file2.npy
        │   │   └── ...
        │   ├── class2/
        │   └── ...
        ├── val/ # Validation set
        │   ├── class1/
        │   ├── class2/
        │   └── ...
        └── test/ # Testing set
            ├── class1/
            ├── class2/
            └── ...
        """
        print("Processing audio files...")

        for label in tqdm(os.listdir(self.dataset_path)):
            label_path = os.path.join(self.dataset_path, label)
            # Skip if the path is not a directory or the label is _background_noise_
            if not os.path.isdir(label_path) or label == "_background_noise_":
                continue

            for filename in os.listdir(label_path):
                if not filename.endswith(".wav"):
                    continue

                filepath = os.path.join(label_path, filename)

                # Determine dataset partition
                dataset_type = which_set(filename, 10, 10)

                # Check if the spectrogram is already saved
                output_dir = os.path.join(self.output_path, dataset_type, label)
                os.makedirs(output_dir, exist_ok=True)

                output_file = os.path.join(output_dir, filename.replace(".wav", ".npy"))
                
                if os.path.exists(output_file):
                    continue  # Skip if the file already exists

                # Convert to spectrogram
                spectrogram = self._get_spectrogram(filepath)

                # Normalize
                spectrogram = (spectrogram - np.mean(spectrogram)) / np.std(spectrogram)

                # Save spectrogram
                np.save(output_file, spectrogram)

        print("Data processing complete!")

    def visualize_random_sample(self):
        sample_class = random.choice(os.listdir(f"{self.output_path}/train"))
        sample_file = random.choice(os.listdir(f"{self.output_path}/train/{sample_class}"))

        spectrogram = np.load(f"{self.output_path}/train/{sample_class}/{sample_file}")

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(spectrogram, sr=self.sample_rate, 
                                 hop_length=int(self.frame_step * self.sample_rate),
                                 x_axis="time", y_axis="mel")
        plt.colorbar(format="%+2.0f dB")
        plt.title(f"Spectrogram of {sample_file}")
        plt.show()


def preprocess():
    # Path to the dataset
    raw_data_dir = "Data/speech_commands_v0.02"  # Raw dataset path

    # Create the preprocessor instance
    processor = SpeechPreprocessor(raw_data_dir)

    # Process all audio files
    processor.process_audio_files()

    # Visualize a sample spectrogram
    processor.visualize_random_sample()

if __name__ == '__main__':
    preprocess()
