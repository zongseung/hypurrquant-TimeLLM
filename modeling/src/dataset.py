import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

class TimeSeriesDataset(Dataset):
    """
    Dataset for time-series forecasting.
    Returns:
      x_seq: (num_features, in_len)
      y_seq: (1, out_len)
      y0: scalar previous value
    """
    def __init__(self, df, feature_cols, in_len, out_len):
        raw_X = df[feature_cols].values  # (T, N_features)
        raw_y = df["Close"].values.reshape(-1, 1)  # (T, 1)

        # fit scalers on full series
        self.scaler_X = MinMaxScaler().fit(raw_X)
        self.scaler_y = MinMaxScaler().fit(raw_y)

        # transform data
        Xs = self.scaler_X.transform(raw_X)  # (T, N_features)
        ys = self.scaler_y.transform(raw_y).flatten()  # (T,)

        # convert to tensors
        self.X = torch.tensor(Xs, dtype=torch.float32)  # (T, N)
        self.y = torch.tensor(ys, dtype=torch.float32).unsqueeze(-1)  # (T, 1)

        self.in_len = in_len
        self.out_len = out_len

    def __len__(self) -> int:
        return len(self.X) - self.in_len - self.out_len + 1

    def __getitem__(self, idx: int):
        # get input sequence and transpose to (N_features, in_len)
        x_seq = self.X[idx : idx + self.in_len].T  # (in_len, N) -> (N, in_len)
        # get output sequence and transpose to (1, out_len)
        y_seq = self.y[idx + self.in_len : idx + self.in_len + self.out_len].squeeze()  # (out_len,)
        # y0: last value of input sequence
        y0 = self.y[idx + self.in_len - 1].squeeze()  # scalar
        return x_seq, y_seq, y0

    def get_scalers(self):
        return self.scaler_X, self.scaler_y