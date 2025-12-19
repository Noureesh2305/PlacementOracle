import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def explore_data(df):
    print("\n🔹 First 5 rows:")
    print(df.head())

    print("\n🔹 Dataset Info:")
    print(df.info())

    print("\n🔹 Missing Values:")
    print(df.isnull().sum())

    print("\n🔹 Placement Count:")
    print(df['status'].value_counts())

    # Class distribution plot
    sns.countplot(x='status', data=df)
    plt.title("Placement Status Distribution")
    plt.show()
