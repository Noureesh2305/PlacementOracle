import pandas as pd

def load_data(filepath):
    """
    Loads the placement dataset from CSV file
    """
    df = pd.read_csv(filepath)
    return df
