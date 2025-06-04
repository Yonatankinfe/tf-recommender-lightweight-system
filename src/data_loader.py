import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """Loads item data from a CSV file.

    Args:
        file_path: Path to the CSV file.

    Returns:
        A pandas DataFrame containing the item data.
    """
    try:
        df = pd.read_csv(file_path)
        # Combine relevant text fields for processing
        df['content'] = df['title'].fillna('') + ' ' + df['description'].fillna('') + ' ' + df['tags'].fillna('')
        print(f"Loaded {len(df)} items from {file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}")
        return pd.DataFrame() # Return empty DataFrame on error
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

if __name__ == '__main__':
    # Example usage:
    data_path = '../data/items.csv' # Adjust path relative to this script if run directly
    items_df = load_data(data_path)
    if not items_df.empty:
        print("\nFirst 5 rows of loaded data:")
        print(items_df.head())
        print("\nContent column sample:")
        print(items_df['content'].head())

