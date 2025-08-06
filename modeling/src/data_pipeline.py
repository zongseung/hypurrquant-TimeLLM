# data_pipeline.py
import os
import numpy as np
from loader import load_candles_via_db
from dataset import TimeSeriesDataset
from torch.utils.data import DataLoader, Subset

def get_dataloaders(
    feature_cols: list[str],
    in_len: int,
    out_len: int,
    batch_size: int,
    num_workers: int = 10
):
    # 1) DB에서 시계열 불러오기
    df = load_candles_via_db()

    # 2) Dataset 생성
    dataset = TimeSeriesDataset(df, feature_cols, in_len, out_len)
    
    # 3) Train, Validation, Test 분할
    total_size = len(dataset)
    train_size = int(total_size * 0.7)
    val_size = int(total_size * 0.15)
    
    indices = list(range(total_size))
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # 4) DataLoader 래핑
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, dataset.get_scalers()
