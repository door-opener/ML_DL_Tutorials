import os, sys
import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap

class MyDataset(Dataset):
    def __init__(self, X, y):
        self.features = torch.tensor(X, dtype=torch.float32)
        self.labels = torch.tensor(y, dtype=torch.int64)

    def __getitem__(self, index):
        x = self.features[index]
        y = self.labels[index]        
        return x, y

    def __len__(self):
        return self.labels.shape[0]

class PyTorchMLP(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.all_layers = torch.nn.Sequential(

            # 1st hidden layer
            torch.nn.Linear(num_features, 25),
            torch.nn.ReLU(),

            # 2nd hidden layer
            torch.nn.Linear(25, 15),
            torch.nn.ReLU(),

            # output layer
            torch.nn.Linear(15, num_classes),
        )

    def forward(self, x):
        logits = self.all_layers(x)
        return logits

def splitData( X, y ):
   # Split into training and test arrays, each arrays have different features, 85% of the data is used for training. #
   # Stratify -> Ensures the distribution of 0's and 1's in the training and test sets are the same #
   # random_state -> ensures that the splitting is the same for each function call. #
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1, stratify=y)

   # Split the training set further, 10% of the data is used for validation #
   X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1, stratify=y_train)

   print("Training size:", X_train.shape)
   print("Validation size :", X_val.shape)
   print("Test size: ", X_test.shape)

   return X_train, X_test, X_val, y_train, y_test, y_val

def getAcc(model, dataloader):
    model = model.eval()
    
    correct = 0.0
    total_examples = 0
    
    for idx, (features, labels) in enumerate(dataloader):
        with torch.inference_mode(): # basically the same as torch.no_grad
            logits = model(features)
        
        predictions = torch.argmax(logits, dim=1)
        dummy = labels.squeeze(1)
        compare = dummy == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)
    return correct / total_examples

def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('D', '^', 'x', 's', 'v')
    colors = ('C0', 'C1', 'C2', 'C3', 'C4')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    tensor = torch.tensor(np.array([xx1.ravel(), xx2.ravel()]).T).float()
    logits = classifier.forward(tensor)
    Z = np.argmax(logits.detach().numpy(), axis=1)

    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        dummy = y.squeeze(1)
        plt.scatter(x=X[dummy == cl, 0], y=X[dummy == cl, 1],
                    alpha=0.8, color=cmap(idx),
                    #edgecolor='black',
                    marker=markers[idx], 
                    label=cl)
    plt.show()

def main():

   df = pd.read_csv("xor.csv") 

   X = df[["x1", "x2"]].values
   y = df[["class label"]].values

   # Splitting data ... #
   X_train, X_test, X_val, y_train, y_test, y_val = splitData( X, y )

   # Preparing data loaders ... #
   train_ds = MyDataset(X_train, y_train)
   val_ds = MyDataset(X_val, y_val)
   test_ds = MyDataset(X_test, y_test)

   train_loader = DataLoader(dataset=train_ds, batch_size=32, shuffle=True)
   val_loader = DataLoader(dataset=val_ds, batch_size=32, shuffle=True)
   test_loader = DataLoader(dataset=test_ds, batch_size=32, shuffle=True)

   torch.manual_seed(1)
   model = PyTorchMLP(num_features=2, num_classes=2)
   optimizer = torch.optim.SGD(model.parameters(), lr=0.05) # Stochastic grad. desc.
   
   num_epochs = 10

   for i in range( num_epochs ):
       model = model.train()
       for idx, (features, labels) in enumerate(train_loader):
           logits = model(features)
           dummy = labels.squeeze(1)
           loss = F.cross_entropy(logits, dummy)

           optimizer.zero_grad()
           loss.backward()
           optimizer.step()

           print("Epoch #%.2d/%.2d | Batch #%.2d/%.2d | Train/Val Loss = %6.4f" %(i+1, num_epochs, idx+1, len(train_loader), loss))

       train_acc = getAcc( model, train_loader )
       val_acc = getAcc( model, val_loader )
       print("\nTraining set accuracy = %3.2f %% | Validation set accuracy = %3.2f %%\n" %(train_acc*100.0, val_acc*100.0))

   train_acc = getAcc( model, train_loader )
   val_acc = getAcc( model, val_loader )
   test_acc = getAcc(model, test_loader)

   print("Training Acc   = %3.2f %%" %(train_acc*100.0) )
   print("Validation Acc = %3.2f %%" %(val_acc*100.0) )
   print("Test Acc       = %3.2f %%\n" %(test_acc*100.0) )

   #plot_decision_regions(X_train, y_train, classifier=model)

if __name__ == "__main__":
   main()
