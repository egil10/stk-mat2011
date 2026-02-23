"""
create_samples.py
================
Generates small sample CSV files from the processed Parquet data.
These samples help users understand the data structure without downloading 
the full dataset.
"""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "code" / "data" / "processed"
SAMPLES_DIR = PROJECT_ROOT / "code" / "data" / "samples"

def create_samples():
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Define representative files to sample
    targets = [
        ("histdata_sample.csv", "eurusd_last_202601.parquet"),
        ("truefx_sample.csv", "eurusd_truefx_bid_202601.parquet"),
        ("dukascopy_sample.csv", "eurusd_dukascopy_bid_202601.parquet"),
    ]
    
    for sample_name, source_name in targets:
        source_path = PROCESSED_DIR / source_name
        if source_path.exists():
            print(f"Creating sample from {source_name}...")
            df = pd.read_parquet(source_path)
            sample_df = df.head(1000)
            sample_df.to_csv(SAMPLES_DIR / sample_name, index=False)
            print(f"  Saved to {SAMPLES_DIR / sample_name}")
        else:
            print(f"  Skipping {source_name} (file not found)")

if __name__ == "__main__":
    create_samples()
