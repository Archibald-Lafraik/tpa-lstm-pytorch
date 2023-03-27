import torch
import torch.nn as nn
from torch import nn, optim
import torch.nn.functional as F
import lightning.pytorch as pl

from util import RSE, CORR

class LSTNet(pl.LightningModule):
    def __init__(
            self, 
            num_features=30,
            window_size=24,
            conv1_out_channels=32, 
            conv1_kernel_height=7,
            recc1_out_channels=64, 
            skip=24, 
            skip_reccs_out_channels=4, 
            output_out_features=1,
            hw_window_size=7,
            output_fun="sigmoid"
        ):
        super(LSTNet, self).__init__()
        self.P = window_size
        self.num_features = num_features 
        self.hidR = recc1_out_channels
        self.hidC = conv1_out_channels
        self.hidS = skip_reccs_out_channels
        self.Ck = conv1_kernel_height
        self.skip = skip
        self.pt = (self.P - self.Ck) // self.skip
        self.hw = hw_window_size
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size = (self.Ck, self.num_features))
        self.GRU1 = nn.GRU(self.hidC, self.hidR)
        self.dropout = nn.Dropout(p = 0.2)
        if (self.skip > 0):
            self.GRUskip = nn.GRU(self.hidC, self.hidS)
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.num_features)
        else:
            self.linear1 = nn.Linear(self.hidR, self.num_features)
        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1)
        self.output = None
        if (output_fun == 'sigmoid'):
            self.output = F.sigmoid
        if (output_fun == 'tanh'):
            self.output = F.tanh
        
        self.criterion = nn.MSELoss()
 
    def forward(self, x):
        batch_size = x.size(0)
        
        #CNN
        c = x.view(-1, 1, self.P, self.num_features)
        c = F.relu(self.conv1(c))
        c = self.dropout(c)
        c = torch.squeeze(c, 3)
        
        # RNN 
        r = c.permute(2, 0, 1).contiguous()
        _, r = self.GRU1(r)
        r = self.dropout(torch.squeeze(r,0))

        
        #skip-rnn
        if (self.skip > 0):
            s = c[:,:, int(-self.pt * self.skip):].contiguous()
            s = s.view(batch_size, self.hidC, self.pt, self.skip)
            s = s.permute(2,0,3,1).contiguous()
            s = s.view(self.pt, batch_size * self.skip, self.hidC)
            _, s = self.GRUskip(s)
            s = s.view(batch_size, self.skip * self.hidS)
            s = self.dropout(s)
            r = torch.cat((r,s),1)
        
        res = self.linear1(r)
        
        #highway
        if (self.hw > 0):
            z = x[:, -self.hw:, :]
            z = z.permute(0,2,1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1,self.num_features)
            res = res + z
            
        if (self.output):
            res = self.output(res)
        return res
    

    def training_step(self, batch, batch_idx):
        inputs, label = batch
        label = label.squeeze()[:, None] 

        outputs = self.forward(inputs)
        loss = self.criterion(outputs, label)
        corr = CORR(outputs, label)
        rse = RSE(outputs, label)

        self.log("train/loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("train/corr", corr, prog_bar=True, on_epoch=True, on_step=False)
        self.log("train/rse", rse, prog_bar=True, on_epoch=True, on_step=False)

        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, label = batch
        label = label.squeeze()[:, None] 

        outputs = self.forward(inputs)
        loss = self.criterion(outputs, label)
        corr = CORR(outputs, label)
        rse = RSE(outputs, label)

        self.log("val/loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val/corr", corr, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val/rse", rse, prog_bar=True, on_epoch=True, on_step=False)

    def predict_step(self, batch, batch_idx):
        inputs, label = batch 
        label = label.squeeze()[:, None]
        pred = self.forward(inputs)

        return pred
    
    def configure_optimizers(self):
        optimiser = optim.Adam(
            self.parameters(),
            lr=1e-3,
            amsgrad=False,
        )
        return optimiser