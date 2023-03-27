# tpa-lstm-pytorch

Implementation of the TPA-LSTM model using pytorch lightning, from the paper (https://arxiv.org/pdf/1809.04206.pdf). This implementation is built for multivariate time series forecasting, but can be adapted for time-series classification.

The repo implements: 
* TPA-LSTM, found in the file ``tpa_lstm.py``
* LSTNet, found in the file ``lstnet.py``
* Vanilla LSTM with 2 linear layers, found in ``other_models.py``

The goal of this repo is to compare the performance of the 3 models above on an electricity price forecasting task. The data used contains energy production and demand, as well as weather reports in spain.

# About

This repo was made for a project at ENS Paris-Saclay by Archibald Fraikin and Charlene de Guitaut, to validate Dr. Laurent Oudre's 'Time-series for machine learning' class.


# Sources

Implementations of TPA-LSTM and LSTNet are based off the following sources:
* https://github.com/shunyaoshih/TPA-LSTM
* https://github.com/jingw2/demand_forecast

Data pre-processing inspired by:
* https://www.kaggle.com/code/dimitriosroussis/electricity-price-forecasting-with-dnns-eda/notebook
