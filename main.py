from src import SpeechPreprocessor

def main():

    # Path to the dataset
    dataset_path = "Data/speech_commands_v0.02"

    # Create the preprocessor instance
    processor = SpeechPreprocessor(dataset_path)

    # Process all audio files
    processor.process_audio_files()

    # Visualize a sample spectrogram
    processor.visualize_random_sample()


if __name__ == '__main__':
    main()
