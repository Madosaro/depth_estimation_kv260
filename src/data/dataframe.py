# data/dataframe.py

import os
import glob

import pandas as pd


def get_dataframe(dataset_path: str, split: str) -> pd.DataFrame:
    path = os.path.join(dataset_path, split)
    rgb_files = sorted(
        glob.glob(os.path.join(path, "**", "rgb_*.png"), recursive=True)
    )
    dep_files = sorted(
        glob.glob(os.path.join(path, "**", "depth_*.png"), recursive=True)
    )
    assert len(rgb_files) == len(dep_files), (
        f"Mismatch: {len(rgb_files)} rgb vs {len(dep_files)} depth"
    )
    print(f"Files found : {len(rgb_files)} pairs rgb/depth")

    return pd.DataFrame({"rgb": rgb_files, "depth": dep_files})


def split_dataframe(
    df: pd.DataFrame,
    n,
    split_proportion: float = 0.8,
    seed: int = 42,
):
    
    total_available = len(df)
    if n is None or n <= 0 or n > total_available:
        n = total_available

    shuffled_df = df.sample(n=n, random_state=seed).reset_index(drop=True)

    split_idx = int(n * split_proportion)

    train_df = shuffled_df.iloc[:split_idx].reset_index(drop=True)
    val_df = shuffled_df.iloc[split_idx:].reset_index(drop=True)
    
    return train_df, val_df