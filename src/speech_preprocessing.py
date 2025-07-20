import os
import librosa
import numpy as np
import librosa.display
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from .partitioning import which_set
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class SpeechPreprocessor:
    def __init__(self, dataset_path, output_path="Data/processed_dataset",
                 sample_rate=16000, n_mels=40, frame_size=0.025, frame_step=0.010,
                 noise_prob=0.3):
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
        self.noise_prob = noise_prob
        self.noise_samples = self._load_noise_samples()

        # Ensure output directories exist
        os.makedirs(self.output_path, exist_ok=True)
        for split in ["train", "val", "test"]:
            os.makedirs(f"{self.output_path}/{split}", exist_ok=True)
        print(f"Initialized SpeechPreprocessor with dataset path: {self.dataset_path} and output path: {self.output_path}")

    def _load_noise_samples(self):
        """
        Loads background noise samples from the dataset.
        Returns:
            list: List of noise samples.
        """
        noise_dir = os.path.join(self.dataset_path, "_background_noise_")
        noises = []
        for fname in os.listdir(noise_dir):
            if fname.endswith(".wav"):
                noise_path = os.path.join(noise_dir, fname)
                y, _ = librosa.load(noise_path, sr=self.sample_rate)
                noises.append(y)
        return noises

    def _time_shift(self, audio, shift_max=0.1):
        """
        Applies a random time shift to the audio signal.
        Args:
            audio (np.ndarray): Audio signal to be shifted.
            shift_max (float): Maximum shift in seconds.
        Returns:
            np.ndarray: Time-shifted audio signal.
        """
        shift = int(random.uniform(-shift_max, shift_max) * self.sample_rate)
        if shift > 0:
            audio = np.pad(audio, (shift, 0), mode='constant')[:len(audio)]
        elif shift < 0:
            audio = np.pad(audio, (0, -shift), mode='constant')[-shift:]
        return audio
    
    def _spec_augment(self, spectrogram, freq_mask_param=3, time_mask_param=8):
        """
        Applies SpecAugment to the spectrogram.
        Args:
            spectrogram (np.ndarray): Input spectrogram.
            freq_mask_param (int): Maximum frequency mask size.
            time_mask_param (int): Maximum time mask size.
        Returns:
            np.ndarray: Augmented spectrogram.
        """
        spec = np.copy(spectrogram)
        num_mel_channels = spec.shape[0]
        num_time_steps = spec.shape[1]

        # Frequency masking
        for _ in range(1):
            f = random.randint(0, freq_mask_param)
            f0 = random.randint(0, num_mel_channels - f)
            spec[f0:f0 + f, :] = 0

        # Time masking
        for _ in range(1):
            t = random.randint(0, time_mask_param)
            t0 = random.randint(0, num_time_steps - t)
            spec[:, t0:t0 + t] = 0

        return spec

    def _add_noise(self, audio):
        """
        Adds random background noise to the audio signal.
        Args:
            audio (np.ndarray): Audio signal to which noise will be added.
        Returns:
            np.ndarray: Audio signal with added noise.
        """
        # If no noise samples are loaded or noise probability is not met, return original audio
        if not self.noise_samples or random.random() > self.noise_prob:
            return audio

        noise = random.choice(self.noise_samples)
        if len(noise) > len(audio):
            start_idx = random.randint(0, len(noise) - len(audio))
            noise = noise[start_idx:start_idx + len(audio)]
        else:
            noise = np.pad(noise, (0, len(audio) - len(noise)))

        # Random noise level between 0.1x and 0.4x of audio signal
        noise_level = random.uniform(0.1, 0.4)
        return audio + noise_level * noise

    def _get_spectrogram(self, filepath, add_noise=False, apply_time_shift=False, apply_spec_augment=False,
                        time_shift_prob=0.5, spec_augment_prob=0.5):
        """
        Converts a .wav file into a log Mel spectrogram.
        
        Args:
            filepath (str): Path to the .wav file.
            add_noise (bool): Whether to add background noise to the audio.
            
        Returns:
            np.ndarray: Log Mel spectrogram of the audio file.
        """
        y, sr = librosa.load(filepath, sr=self.sample_rate)

        if apply_time_shift and random.random() < time_shift_prob:
            y = self._time_shift(y)

        if add_noise:
            y = self._add_noise(y)

        spectrogram = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=self.n_mels,
            hop_length=int(self.frame_step * sr),
            n_fft=int(self.frame_size * sr)
        )
        spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

        if apply_spec_augment and random.random() < spec_augment_prob:
            spectrogram = self._spec_augment(spectrogram)

        # Pad or truncate to fixed width
        target_width = 101
        if spectrogram.shape[1] < target_width:
            spectrogram = np.pad(spectrogram, ((0, 0), (0, target_width - spectrogram.shape[1])), mode="constant")
        else:
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
        Data augmentation is applied only to the training set with noise addition, time shifting and spec augment.
        """
        print("Processing audio files...")

        augmented_files_log = open("augmented_files_log.txt", "w")
        noise_count = 0
        time_shift_count = 0
        spec_augment_count = 0

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

                # Add noise only for training set
                is_train = (dataset_type == "train")

                # Check if the spectrogram is already saved
                output_dir = os.path.join(self.output_path, dataset_type, label)
                os.makedirs(output_dir, exist_ok=True)

                output_file = os.path.join(output_dir, filename.replace(".wav", ".npy"))
                
                if os.path.exists(output_file):
                    continue  # Skip if the file already exists

                apply_noise = is_train and (random.random() < self.noise_prob)
                apply_time_shift = is_train and (random.random() < 0.3)
                apply_spec_augment = is_train and (random.random() < 0.3)

                # Convert to spectrogram
                spectrogram = self._get_spectrogram(
                    filepath, apply_noise, is_train, is_train,
                    time_shift_prob=0.3, spec_augment_prob=0.3
                )

                # Normalize
                spectrogram = (spectrogram - np.mean(spectrogram)) / np.std(spectrogram)

                # Save spectrogram
                np.save(output_file, spectrogram)

                # Log the augmented files
                if apply_noise:
                    augmented_files_log.write(f"{output_file}\n")
                    noise_count += 1
                if apply_time_shift:
                    time_shift_count += 1
                if apply_spec_augment:
                    spec_augment_count += 1

        augmented_files_log.close()
        print("Data processing complete!")

        # Stats summary
        total_processed = sum(
            len([f for f in files if f.endswith(".npy")])
            for split in ["train", "val", "test"]
            for root, _, files in os.walk(os.path.join(self.output_path, split))
        )

        train_total = sum(
            len([f for f in files if f.endswith(".npy")])
            for root, _, files in os.walk(os.path.join(self.output_path, "train"))
        )

        if total_processed > 0:
            print(f"\nSummary of Preprocessing:")
            print(f"- Total samples processed: {total_processed}")

        if train_total > 0:
            print(f"- Total training samples: {train_total}")
            print(f"- Training samples with noise: {noise_count} ({(noise_count / train_total) * 100:.2f}%)")
            print(f"- Training samples with time shift: {time_shift_count} ({(time_shift_count / train_total) * 100:.2f}%)")
            print(f"- Training samples with SpecAugment: {spec_augment_count} ({(spec_augment_count / train_total) * 100:.2f}%)\n")



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

    # Visualize a sample spectrogram as a sanity check
    processor.visualize_random_sample()

if __name__ == '__main__':
    preprocess()
