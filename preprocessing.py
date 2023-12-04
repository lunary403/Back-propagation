import pandas as pd




def read_file(file):
    df = pd.read_excel(file)
    df["Class"].replace({"BOMBAY": 0, "CALI": 1, "SIRA": 2}, inplace=True)
    return df

def fill_null_values(df, column):
    new_values = df[column].fillna(df[column].mean())
    return new_values

def train_test_split(df):
    X = df.iloc[:, 0:5]
    y = df.iloc[:, 5:6]

    X_normalized = (X - X.min()) / (X.max() - X.min())
    X_train = pd.concat([X_normalized.iloc[0:30], X_normalized.iloc[50:80], X_normalized.iloc[100:130]], axis=0)
    y_train = pd.concat([y.iloc[0:30], y.iloc[50:80], y.iloc[100:130]], axis=0)
    X_test = pd.concat([X_normalized.iloc[30:50], X_normalized.iloc[80:100], X_normalized.iloc[130:150]], axis=0)
    y_test = pd.concat([y.iloc[30:50], y.iloc[80:100], y.iloc[130:150]], axis=0)

    return X_train, y_train, X_test, y_test


