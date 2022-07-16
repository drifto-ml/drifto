import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

class LinearRegression(pl.LightningModule):
    def __init__(self, 
        input_dim,
        output_dim,
        n_embeds,
        embed_dim,
        lr=1e-4):

        super().__init__()
        self.weights = nn.Linear(input_dim, output_dim)
        self.E = torch.nn.Embedding(n_embeds, embed_dim)
        self.lr = lr

    def forward(self, x):
        x_dense, x_offs = x
        x = torch.concat((x_dense, self.E(x_offs).squeeze(1)),dim=1)
        return self.weights(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        loss = F.mse_loss(self(x), y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        pass

    def predict_step(self, batch, batch_idx):
        x = batch
        return self(x)

class LogisticRegression(LinearRegression):
    def __init__(self, 
        input_dim,
        output_dim,
        n_embeds,
        embed_dim,
        lr=1e-4):
        super().__init__(input_dim, output_dim, n_embeds, embed_dim, lr=lr)

    def forward(self, x):
        return F.softmax(super().forward(x))

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        loss = F.cross_entropy(self(x), y)
        self.log('train_loss', loss)
        return loss

class NaiveBayes(pl.LightningModule):
    def __init__(self,
        input_dim : list[int],
        n_cats : list[int],
        n_classes: int):
        super().__init__()
        self.p = torch.zeros()
        self.p_lbl = torch.zeros()
        self.p_num = None
        self.n_classes = n_classes
        L = sum(n_cats)
        self.offsets = torch.tensor(n_cats) - 1
        self.N = torch.nn.Parameter(torch.zeros(L))
        self.C = torch.nn.Parameter(torch.zeros(n_classes))
        self.P = torch.nn.Parameter(torch.zeros(L, n_classes))

    def forward(self, x, include_prob=False):
        # TODO: Consider converting this into a log sum for numerical reasons
        Pi = torch.prod(self.P[x]/self.N[x], dim=1) # TODO dim
        y_hat = torch.argmax(self.C * Pi)
        if include_prob:
            return y_hat
        else:
            Z = torch.sum(self.C * Pi)
            return y_hat, self.C * Pi / Z

    def configure_optimizers(self):
        optimizer = torch.optim.Optimizer
        optimizer.step = lambda : None
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        loss = F.cross_entropy(self(x), y)
        
        # Perform update
        self.N[x] += 1
        self.P[torch.stack(x,y)] += 1 # TODO
        self.C += torch.sum(F.one_hot(y, num_classes=self.n_classes), dim=0)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        pass


