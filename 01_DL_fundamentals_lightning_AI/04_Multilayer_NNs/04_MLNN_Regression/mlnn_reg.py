import os, sys
import torch
import numpy as np
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt

class PyTorchMLP(torch.nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.all_layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(num_features, 50),
            torch.nn.ReLU(),
            # 2nd hidden layer
            torch.nn.Linear(50, 25),
            torch.nn.ReLU(),
            # output layer
            torch.nn.Linear(25, 1), )

    def forward(self, x):
        logits = self.all_layers(x).flatten()
        return logits

class MyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.targets = y

    def __getitem__(self, index):
        x = self.features[index]
        y = self.targets[index]
        return x, y

    def __len__(self):
        return self.targets.shape[0]

if __name__ =="__main__":

   X_train = torch.tensor([258.0, 270.0, 294.0, 320.0, 342.0, 368.0, 396.0, 446.0, 480.0, 586.0]).view(-1, 1)
   y_train = torch.tensor([236.4, 234.4, 252.8, 298.6, 314.2, 342.2, 360.8, 368.0, 391.2, 390.8])

   #plt.scatter(X_train, y_train)
   #plt.xlabel("Feature variable")
   #plt.ylabel("Target variable")
   #plt.show()

   x_mean, x_std = X_train.mean(), X_train.std()
   y_mean, y_std = y_train.mean(), y_train.std()
   X_train_norm = (X_train - x_mean) / x_std
   y_train_norm = (y_train - y_mean) / y_std

   train_ds = MyDataset(X_train_norm, y_train_norm)

   train_loader = DataLoader(
       dataset=train_ds,
       batch_size=20,
       shuffle=True )

   torch.manual_seed(1)
   model = PyTorchMLP(num_features=1)
   optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

   num_epochs = 30

   loss_list = []
   train_acc_list, val_acc_list = [], []
   
   for epoch in range(num_epochs):
     model = model.train()

     for batch_idx, (features, targets) in enumerate(train_loader):
         logits = model(features)
         loss = F.mse_loss(logits, targets)
 
         optimizer.zero_grad()
         loss.backward()
         optimizer.step()

         if not batch_idx % 250:
             print("Epoch: %.3d/%.3d | Batch idx: %.3d/%.3d | Train Loss = %4.3f" %(epoch+1, num_epochs, batch_idx, len(train_loader), loss))

         loss_list.append(loss.item())

   model.eval()

   X_range = torch.arange(150, 800, 0.1).view(-1, 1)
   X_range_norm = ( X_range - x_mean )/ x_std

   with torch.no_grad():
        y_mlp_norm = model(X_range_norm)

   # Un-normalise for plotting #
   y_mlp = y_mlp_norm*y_std + y_mean

   # plot results
   plt.scatter(X_train, y_train, label="Training points")

   plt.plot(X_range, y_mlp, color="C1", label="MLP fit", linestyle="-")

   plt.xlabel("Feature variable")
   plt.ylabel("Target variable")
   plt.legend()
   # plt.savefig("mlp.pdf")
   plt.show()
