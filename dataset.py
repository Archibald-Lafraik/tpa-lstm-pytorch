import math
import pandas as pd

import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl


class ElectricityDatasetRaw(Dataset):
    def __init__(self, mode, split_ratios, window_size):
        self.w_size = window_size
        self.label_col_name = "price actual"
        self.raw_df = pd.read_csv('data/clean_and_merge_data.csv', index_col=0)

        self.train_frac = split_ratios['train']
        self.val_frac = split_ratios['val']
        self.test_frac = split_ratios['predict']

        self.train_lim = math.floor(self.train_frac * self.raw_df.shape[0]) 
        self.val_lim = math.floor(self.val_frac * self.raw_df.shape[0]) + self.train_lim

        if mode == "train":
            self.df = self.raw_df.iloc[:self.train_lim]
        if mode == "val":
            self.df = self.raw_df.iloc[self.train_lim:self.val_lim]
        if mode == "predict":
            self.df = self.raw_df.iloc[self.val_lim:]

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.X = torch.tensor(self.df.values, dtype=torch.float32).to(self.device)
        self.y = torch.tensor(self.df.loc[:, self.label_col_name].values, dtype=torch.float32) \
                .unsqueeze(1).to(self.device)
    
    def __getitem__(self, idx):
        return self.X[idx:idx + self.w_size, :], self.y[idx + self.w_size]

    def __len__(self):
        return len(self.df) - self.w_size


class ElectricityDataset(Dataset):
    def __init__(self, mode, split_ratios, window_size):
        self.w_size = window_size
        self.raw_dataset = np.load('data/dataset_final.npy')

        self.train_frac = split_ratios['train']
        self.val_frac = split_ratios['val']
        self.test_frac = split_ratios['predict']

        self.train_lim = math.floor(self.train_frac * self.raw_dataset.shape[0]) 
        self.val_lim = math.floor(self.val_frac * self.raw_dataset.shape[0]) + self.train_lim

        if mode == "train":
            self.dataset = self.raw_dataset[:self.train_lim]
        if mode == "val":
            self.dataset = self.raw_dataset[self.train_lim:self.val_lim]
        if mode == "predict":
            self.dataset = self.raw_dataset[self.val_lim:]

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.X = torch.tensor(self.dataset, dtype=torch.float32).to(self.device)
        self.y = torch.tensor(self.dataset[:, -1], dtype=torch.float32) \
                .unsqueeze(1).to(self.device)
    
    def __getitem__(self, idx):
        return self.X[idx:idx + self.w_size, :], self.y[idx + self.w_size]

    def __len__(self):
        return len(self.dataset) - self.w_size


class ElectricityDataModule(pl.LightningDataModule):
    def __init__(self, dataset_splits, batch_size=64, window_size=24):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_splits = dataset_splits
        self.window_size = window_size


    def setup(self, stage):
        if stage == "fit":
            self.data_train = ElectricityDataset(
                mode="train",
                split_ratios=self.dataset_splits,
                window_size=self.window_size
            )
            self.data_val = ElectricityDataset(
                mode="val",
                split_ratios=self.dataset_splits,
                window_size=self.window_size
            )
        elif stage == "predict":
            self.data_pred = ElectricityDataset(
                mode="predict",
                split_ratios=self.dataset_splits,
                window_size=self.window_size
            )

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.data_pred, batch_size=self.batch_size, shuffle=False)