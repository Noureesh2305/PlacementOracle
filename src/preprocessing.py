import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    # Drop salary column
    df = df.drop(columns=['salary'])

    # Encode target variable
    df['status'] = df['status'].map({'Placed': 1, 'Not Placed': 0})

    # Encode binary columns
    binary_cols = ['gender', 'ssc_b', 'hsc_b', 'workex', 'specialisation']
    for col in binary_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    # One-hot encode remaining categorical columns
    df = pd.get_dummies(df, drop_first=True)

    return df
