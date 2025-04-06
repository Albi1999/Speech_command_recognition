from src import SpeechPreprocessor, ModelEvaluator


def preprocess():
    # Path to the dataset
    raw_data_dir = "Data/speech_commands_v0.02"  # Raw dataset path

    # Create the preprocessor instance
    processor = SpeechPreprocessor(raw_data_dir)

    # Process all audio files
    processor.process_audio_files()

    # Visualize a random sample spectrogram
    #processor.visualize_random_sample()


def evaluate_model():
    # Path to the models
    models_path = "models"

    # Create the evaluator instance
    evaluator = ModelEvaluator(
        models_path=models_path,
        label_map_path="label_to_index.pkl",
        processed_dataset_path="Data/processed_dataset",
        batch_size=32,
        output_dir="output"
    )

    # Evaluate all models
    evaluator.evaluate_all()

    # Export results to CSV
    df = evaluator.export_results_csv()
    print(df)

if __name__ == '__main__':
    preprocess()
    evaluate_model()