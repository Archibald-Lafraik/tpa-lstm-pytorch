# -*-coding:utf-8-*-

'''
Pytorch Implementation of TPA-LSTM
Paper link: https://arxiv.org/pdf/1809.04206v2.pdf
Author: Jing Wang (jingw2@foxmail.com)
Date: 04/10/2020
'''

import torch
from torch import nn
import torch.nn.functional as F
import argparse
from progressbar import *
from torch.optim import Adam
import util
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
from datetime import date


def train(
        X,
        y,
        params
):
    '''
    Args:
    - X (array like): shape (num_samples, num_features, num_periods)
    - y (array like): shape (num_samples, num_periods)
    - epoches (int): number of epoches to run
    - step_per_epoch (int): steps per epoch to run
    - seq_len (int): output horizon
    - likelihood (str): what type of likelihood to use, default is gaussian
    - num_skus_to_show (int): how many skus to show in test phase
    - num_results_to_sample (int): how many samples in test phase as prediction
    '''
    num_ts, num_periods, num_features = X.shape
    model = TPALSTM(1, params.seq_len,
                    params.hidden_size, params.num_obs_to_train, params.n_layers)
    optimizer = Adam(model.parameters(), lr=params.lr)
    random.seed(2)
    # select sku with most top n quantities
    Xtr, ytr, Xte, yte = util.train_test_split(X, y)
    losses = []
    cnt = 0

    yscaler = None
    if params.standard_scaler:
        yscaler = util.StandardScaler()
    elif params.log_scaler:
        yscaler = util.LogScaler()
    elif params.mean_scaler:
        yscaler = util.MeanScaler()
    elif params.max_scaler:
        yscaler = util.MaxScaler()
    if yscaler is not None:
        ytr = yscaler.fit_transform(ytr)

    # training
    seq_len = params.seq_len
    obs_len = params.num_obs_to_train
    progress = ProgressBar()
    for epoch in progress(range(params.num_epoches)):
        # print("Epoch {} starts...".format(epoch))
        for step in range(params.step_per_epoch):
            Xtrain, ytrain, Xf, yf = util.batch_generator(Xtr, ytr, obs_len, seq_len, params.batch_size)
            Xtrain = torch.from_numpy(Xtrain).float()
            ytrain = torch.from_numpy(ytrain).float()
            Xf = torch.from_numpy(Xf).float()
            yf = torch.from_numpy(yf).float()
            ypred = model(ytrain)
            # loss = util.RSE(ypred, yf)
            loss = F.mse_loss(ypred, yf)

            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

    # test
    mape_list = []
    # select skus with most top K
    X_test = Xte[:, -seq_len - obs_len:-seq_len, :].reshape((num_ts, -1, num_features))
    Xf_test = Xte[:, -seq_len:, :].reshape((num_ts, -1, num_features))
    y_test = yte[:, -seq_len - obs_len:-seq_len].reshape((num_ts, -1))
    yf_test = yte[:, -seq_len:].reshape((num_ts, -1))
    yscaler = None
    if params.standard_scaler:
        yscaler = util.StandardScaler()
    elif params.log_scaler:
        yscaler = util.LogScaler()
    elif params.mean_scaler:
        yscaler = util.MeanScaler()
    elif params.max_scaler:
        yscaler = util.MaxScaler()
    if yscaler is not None:
        ytr = yscaler.fit_transform(ytr)
    if yscaler is not None:
        y_test = yscaler.fit_transform(y_test)
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()
    Xf_test = torch.from_numpy(Xf_test).float()
    ypred = model(y_test)
    ypred = ypred.data.numpy()
    if yscaler is not None:
        ypred = yscaler.inverse_transform(ypred)
    ypred = ypred.ravel()

    loss = np.sqrt(np.sum(np.square(yf_test - ypred)))
    print("losses: ", loss)

    if params.show_plot:
        plt.figure(1, figsize=(20, 5))
        plt.plot([k + seq_len + obs_len - seq_len \
                  for k in range(seq_len)], ypred, "r-")
        plt.title('Prediction uncertainty')
        yplot = yte[-1, -seq_len - obs_len:]
        plt.plot(range(len(yplot)), yplot, "k-")
        plt.legend(["prediction", "true", "P10-P90 quantile"], loc="upper left")
        ymin, ymax = plt.ylim()
        plt.vlines(seq_len + obs_len - seq_len, ymin, ymax, color="blue", linestyles="dashed", linewidth=2)
        plt.ylim(ymin, ymax)
        plt.xlabel("Periods")
        plt.ylabel("Y")
        plt.show()
    return losses, mape_list

def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size
        
    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])
        
        if single_step:
            labels.append(target[i + target_size])
        else:
            labels.append(target[i : i + target_size])

    return np.array(data), np.array(labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epoches", "-e", type=int, default=1000)
    parser.add_argument("--step_per_epoch", "-spe", type=int, default=2)
    parser.add_argument("-lr", type=float, default=1e-3)
    parser.add_argument("--n_layers", "-nl", type=int, default=3)
    parser.add_argument("--hidden_size", "-hs", type=int, default=24)
    parser.add_argument("--seq_len", "-sl", type=int, default=7)
    parser.add_argument("--num_obs_to_train", "-not", type=int, default=1)
    parser.add_argument("--num_results_to_sample", "-nrs", type=int, default=10)
    parser.add_argument("--show_plot", "-sp", action="store_true")
    parser.add_argument("--run_test", "-rt", action="store_true")
    parser.add_argument("--standard_scaler", "-ss", action="store_true")
    parser.add_argument("--log_scaler", "-ls", action="store_true")
    parser.add_argument("--mean_scaler", "-ms", action="store_true")
    parser.add_argument("--max_scaler", "-max", action="store_true")
    parser.add_argument("--batch_size", "-b", type=int, default=64)
    # parser.add_argument("--sample_size", type=int, default=100)

    args = parser.parse_args()

#     if args.run_test:
#         data_path = util.get_data_path()
#         data = pd.read_csv(os.path.join(data_path, "LD_MT200_hour.csv"), parse_dates=["date"])
#         data["year"] = data["date"].apply(lambda x: x.year)
#         data["day_of_week"] = data["date"].apply(lambda x: x.dayofweek)
#         data = data.loc[(data["date"] >= date(2014, 1, 1)) & (data["date"] <= date(2014, 3, 1))]

#         features = ["hour", "day_of_week"]
#         # hours = pd.get_dummies(data["hour"])
#         # dows = pd.get_dummies(data["day_of_week"])
#         hours = data["hour"]
#         dows = data["day_of_week"]
#         X = np.c_[np.asarray(hours), np.asarray(dows)]
#         num_features = X.shape[1]
#         num_periods = len(data)
#         X = np.asarray(X).reshape((-1, num_periods, num_features))
#         y = np.asarray(data["MT_200"]).reshape((-1, num_periods))
        # X = np.tile(X, (10, 1, 1))
        # y = np.tile(y, (10, 1))
        
    data_df = pd.read_csv('data_kaggle/clean_and_merge_data.csv', index_col=0)
    train_end_idx = 27048
    cv_end_idx = 31056
    test_end_idx = 35064

    X = data_df[data_df.columns.drop('price actual')].values
    y = data_df['price actual'].values
    y = y.reshape(-1, 1)
    print(X.shape, y.shape)

    dataset_norm = np.concatenate((X, y), axis=1)


    past_history = 24
    future_target = 0

    X_train, y_train = multivariate_data(dataset_norm, dataset_norm[:, -1],
                                         0, train_end_idx, past_history, 
                                         future_target, step=1, single_step=False)
    print(dataset_norm.shape, X_train.shape, y_train.shape)



    losses, mape_list = train(X_train, y_train, args)
    if args.show_plot:
        plt.plot(range(len(losses)), losses, "k-")
        plt.xlabel("Period")
        plt.ylabel("Loss")
        plt.show()