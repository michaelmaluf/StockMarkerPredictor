import yfinance as yf
import numpy as np
import pandas as pd
from collections import deque
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def shuffle_in_unison(a, b):
    # Shuffle two arrays consistently
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)

def load_data(ticker, n_steps=50, scale=True, shuffle=True, lookup_step=1, split_by_date=True,
              test_size=0.2, feature_columns=['adjclose', 'volume', 'open', 'high', 'low']):
    # Check if ticker is a string or DataFrame
    if isinstance(ticker, str):
        # Obtain data from yfinance API
        stock_info = yf.Ticker(ticker)
        df = stock_info.history(period="max")
    elif isinstance(ticker, pd.DataFrame):
        # Use given DataFrame
        df = ticker
    else:
        raise TypeError("ticker can be either a str or a `pd.DataFrame` instances")

    # Rename columns to match the original function
    df = df.rename(columns={'Close': 'adjclose', 'Volume': 'volume', 'Open': 'open', 'High': 'high', 'Low': 'low'})

    # Store elements to return from this function
    result = {}
    # Include original dataframe in result
    result['df'] = df.copy()
    # Ensure feature_columns exist in the dataframe
    for col in feature_columns:
        assert col in df.columns, f"'{col}' does not exist in the dataframe."
    # Add date as a column
    if "date" not in df.columns:
        df["date"] = df.index
    if scale:
        column_scaler = {}
        # Scale data (prices) between 0 and 1
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler
        # Add MinMaxScaler instances to the result
        result["column_scaler"] = column_scaler
    df['future'] = df['adjclose'].shift(-lookup_step)
    # Store last `lookup_step` columns before dropping NaNs
    last_sequence = np.array(df[feature_columns].tail(lookup_step))
    # Drop NaNs
    df.dropna(inplace=True)
    sequence_data = []
    sequences = deque(maxlen=n_steps)
    for entry, target in zip(df[feature_columns + ["date"]].values, df['future'].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])
    # Get last sequence by appending the last `n_step` sequence with `lookup_step` sequence
    last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
    last_sequence = np.array(last_sequence).astype(np.float32)
    # Add to result
    result['last_sequence'] = last_sequence
    # Create X and y arrays
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    if split_by_date:
        # Split dataset by date for train & test sets
        train_samples = int((1 - test_size) * len(X))
        result["X_train"] = X[:train_samples]
        result["y_train"] = y[:train_samples]
        result["X_test"] = X[train_samples:]
        result["y_test"] = y[train_samples:]
        if shuffle:
            # Shuffle the datasets for training (if shuffle parameter is set)
            shuffle_in_unison(result["X_train"], result["y_train"])
            shuffle_in_unison(result["X_test"], result["y_test"])
    else:
        # Randomly split the dataset
        result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y,
                                                                                                    test_size=test_size,
                                                                                                    shuffle=shuffle)
    # Get the list of test set dates
    dates = result["X_test"][:, -1, -1]
    # Retrieve test features from the original dataframe
    result["test_df"] = result["df"].loc[dates]
    # Remove duplicated dates in the testing dataframe
    result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(keep='first')]
    # Remove dates from the training/testing sets & convert to float32
    result["X_train"] = result["X_train"][:, :, :len(feature_columns)].astype(np.float32)
    result["X_test"] = result["X_test"][:, :, :len(feature_columns)].astype(np.float32)
    return result
