import os
import hashlib
import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer

# Define input and output directories
INPUT_FILEPATH = "SDV_Library/Annual_Electricity_Data.csv"
OUTPUT_DIR = "SDV_Library"

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data_from_file(filepath):
    """Load data from a CSV file."""
    if os.path.exists(filepath):
        data = pd.read_csv(filepath)
        print(f"✅ Data loaded from {filepath}")
        return data
    else:
        raise FileNotFoundError(f"❌ Error: File '{filepath}' not found.")

def generate_metadata(data):
    """Generate metadata from the given DataFrame."""
    metadata = Metadata.detect_from_dataframe(data)
    return metadata

def get_data_hash(data):
    """Generate a hash based on the structure and content of the dataset."""
    hash_obj = hashlib.md5(pd.util.hash_pandas_object(data, index=True).values)
    return hash_obj.hexdigest()

def get_model_filename(data_hash):
    """Generate a unique filename for the model based on data hash."""
    return os.path.join(OUTPUT_DIR, f"synthesizer_model_{data_hash}.pkl")

def get_output_filename(data_hash):
    """Generate a unique filename for the synthetic data output."""
    return os.path.join(OUTPUT_DIR, f"synthetic_data_{data_hash}.csv")

def train_synthesizer(metadata, data):
    """Train the GaussianCopulaSynthesizer model."""
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(data)
    return synthesizer

def generate_synthetic_data(synthesizer, num_rows=5):
    """Generate synthetic data using the trained synthesizer."""
    synthetic_data = synthesizer.sample(num_rows=num_rows)
    return synthetic_data

def save_model(synthesizer, filepath):
    """Save the trained synthesizer model to a file."""
    synthesizer.save(filepath)
    print(f"✅ Model saved to {filepath}")

def load_model(filepath):
    """Load a synthesizer model from a file."""
    synthesizer = GaussianCopulaSynthesizer.load(filepath)
    print(f"✅ Model loaded from {filepath}")
    return synthesizer

def save_synthetic_data(data, filepath):
    """Save synthetic data to a CSV file."""
    data.to_csv(filepath, index=False)
    print(f"✅ Synthetic data saved to {filepath}")

def main():
    try:
        # Step 1: Load Sample Data
        data = load_data_from_file(INPUT_FILEPATH)

        # Step 2: Generate Metadata
        metadata = generate_metadata(data)

        # Step 3: Generate a Hash of the Data
        data_hash = get_data_hash(data)

        # Step 4: Determine Filenames for Model and Output
        model_filepath = get_model_filename(data_hash)
        output_filepath = get_output_filename(data_hash)

        # Step 5: Check if Model Exists and Load or Train
        if os.path.exists(model_filepath):
            print("✅ Using existing model.")
            synthesizer = load_model(model_filepath)
        else:
            print("🔄 Training a new model.")
            synthesizer = train_synthesizer(metadata, data)
            save_model(synthesizer, model_filepath)

        # Step 6: Generate Synthetic Data
        synthetic_data = generate_synthetic_data(synthesizer)

        # Step 7: Save Synthetic Data to a CSV File
        save_synthetic_data(synthetic_data, output_filepath)

    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()