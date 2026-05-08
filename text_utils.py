# text_utils.py
# Text processing utilities: save/load text descriptions to/from CSV.
import os
import csv
from typing import List
import pandas as pd


def save_texts_to_csv(texts: List[str], file_path: str):
    """
    Save text descriptions to a CSV file.

    Args:
        texts: List of text descriptions.
        file_path: Path to the output CSV file.
    """
    if not isinstance(texts, list):
        raise ValueError("texts must be a list")
    if not isinstance(file_path, str):
        raise ValueError("file_path must be a string")

    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    try:
        df = pd.DataFrame({
            'window_id': range(len(texts)),
            'text_description': texts
        })
        df.to_csv(file_path, index=False, encoding='utf-8', quoting=csv.QUOTE_MINIMAL)
        print(f"Saved {len(texts)} window text descriptions to: {file_path}")
    except Exception as e:
        print(f"Failed to save CSV: {e}")
        raise


def load_texts_from_csv(file_path: str) -> List[str]:
    """
    Load text descriptions from a CSV file.

    Args:
        file_path: Path to the CSV file.

    Returns:
        List of text descriptions.
    """
    if not isinstance(file_path, str):
        raise ValueError("file_path must be a string")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return []

    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        if 'text_description' not in df.columns:
            print(f"CSV missing required column 'text_description': {file_path}")
            return []
        texts = df['text_description'].tolist()
        print(f"Loaded {len(texts)} window text descriptions from: {file_path}")
        return texts
    except pd.errors.EmptyDataError:
        print(f"Empty CSV file: {file_path}")
        return []
    except pd.errors.ParserError as e:
        print(f"CSV parsing error: {e}")
        return []
    except Exception as e:
        print(f"Failed to load CSV: {e}")
        return []