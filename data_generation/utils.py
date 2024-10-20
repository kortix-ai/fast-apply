import os
import pandas as pd
import tiktoken
import pyarrow.parquet as pq
import math

def calculate_cost(input_tokens: int, output_tokens: int) -> float:
    """
    Calculates the cost based on input and output tokens.
    Args:
        input_tokens (int): Number of input tokens.
        output_tokens (int): Number of output tokens.
    Returns:
        float: Calculated cost in USD.
    """
    input_cost = (input_tokens / 1_000_000) * 1.25  # $1.25 per 1M tokens for input
    output_cost = (output_tokens / 1_000_000) * 5    # $5 per 1M tokens for output
    return input_cost + output_cost

def load_data(file_path: str, n: int = None) -> pd.DataFrame:
    """
    Loads data from a Parquet or CSV file into a DataFrame.
    Args:
        file_path (str): Path to the input data file.
        n (int, optional): Number of rows to load.
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    if file_path.endswith('.parquet'):
        df = pq.read_table(file_path).to_pandas()
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a .parquet or .csv file.")
    return df.head(n) if n is not None else df

def count_tokens(text: str, model: str) -> int:
    """
    Counts the number of tokens in the given text using the specified model's tokenizer.
    Args:
        text (str): The text to tokenize.
        model (str): The name of the model to use for tokenization.
    Returns:
        int: The number of tokens in the text.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print(f"Warning: model {model} not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def load_parquet(file_path: str) -> pd.DataFrame:
    """
    Load a Parquet file and return a DataFrame.
    Args:
        file_path (str): Path to the Parquet file.
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    df = pq.read_table(file_path).to_pandas()
    return df

def save_parquet(df: pd.DataFrame, file_path: str):
    """
    Save a DataFrame to a Parquet file.
    Args:
        df (pd.DataFrame): DataFrame to save.
        file_path (str): Path to save the Parquet file.
    """
    df.to_parquet(file_path, index=False)
    print(f"Updated Parquet file saved to {file_path}")

def save_json(df: pd.DataFrame, json_file: str):
    """
    Save a DataFrame to a JSON file.
    Args:
        df (pd.DataFrame): DataFrame to save.
        json_file (str): Path to save the JSON file.
    """
    df.to_json(json_file, orient='records', indent=2)
    print(f"JSON file saved to {json_file}")

def display_parquet_info(df: pd.DataFrame):
    """
    Display information about a Parquet DataFrame.
    Args:
        df (pd.DataFrame): DataFrame to display information about.
    """
    print("Parquet File Information:")
    print(f"Number of records: {len(df)}")
    print("\nSchema:")
    print(df.dtypes)
    print("\nFirst few rows:")
    print(df.head())
