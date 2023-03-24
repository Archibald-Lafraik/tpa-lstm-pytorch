import torch
from torch import nn, optim
import lightning.pytorch as pl

from util import RMSE

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class LSTM(pl.LightningModule):
    def __init__(
            self,
            input_size,
            lstm_hid_size,
            linear_hid_size,
            output_horizon=1,
            n_layers=1,
            lr=1e-3
        ) -> None:
        super().__init__()
        self.lr = lr
        self.n_layers = n_layers
        self.lstm_hid_size = lstm_hid_size
        self.output_horizon = output_horizon

        self.lstm = nn.LSTM(input_size, lstm_hid_size, n_layers, \
                            bias=True, batch_first=True) 

        self.linear = nn.Sequential(
            nn.Linear(lstm_hid_size, linear_hid_size),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(linear_hid_size, output_horizon) 
        )
        
        self.criterion = nn.MSELoss()

    def forward(self, x):
        batch_size, obs_len, f_dim = x.size()

        ht = torch.zeros(self.n_layers, batch_size, self.lstm_hid_size).to(device)
        ct = ht.clone()
        for t in range(obs_len):
            xt = x[:, t, :].view(batch_size, 1, -1)
            out, (ht, ct) = self.lstm(xt, (ht, ct))
            htt = ht.permute(1, 0, 2)
            htt = htt[:, -1, :]

        htt = htt.reshape(batch_size, -1)
        out = self.linear(htt).unsqueeze(-1)

        return out
    
    def training_step(self, batch, batch_idx):
        inputs, label = batch 

        outputs = self.forward(inputs)
        loss = self.criterion(outputs, label)
        score = RMSE(outputs, label)

        self.log("train/loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("train/score", score, prog_bar=True, on_epoch=True, on_step=False)

        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, label = batch 

        outputs = self.forward(inputs)
        loss = self.criterion(outputs, label)
        score = RMSE(outputs, label)

        self.log("val/loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val/score", score, prog_bar=True, on_epoch=True, on_step=False)

    def predict_step(self, batch, batch_idx):
        inputs, label = batch 
        pred = self.forward(inputs)

        return pred
    
    def configure_optimizers(self):
        optimiser = optim.Adam(self.parameters(), lr=self.lr, amsgrad=True)
        return optimiser