import os, sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class MyDataset(Dataset):

      def __init__(self, X, y):
          #self.features = torch.tensor(X, dtype=torch.float32)
          #self.labels = torch.tensor(y, dtype=torch.float32)
          self.features = X
          self.labels = y

      def __getitem__(self, index):
          x = self.features[index]
          y = self.labels[index]
          return x,y

      def __len__(self):
          return self.labels.shape[0]

class LogRegression(torch.nn.Module):

      def __init__(self, num_features):
          super().__init__()
          self.linear = torch.nn.Linear( num_features, 1 )

      def forward(self, x):
          logits = self.linear(x)
          probas = torch.sigmoid(logits)
          return probas

def readData( in_file ):
    dat = np.genfromtxt(in_file, delimiter="\t")
    x_values = [ [ x[0],x[1] ] for x in dat ]
    y_values = [ x[-1] for x in dat ]
    return torch.tensor(x_values, dtype=torch.float32), torch.tensor(y_values, dtype=torch.float32)

def Normalise( x_data ):
    return (x_data - x_data.mean(axis=0)) / x_data.std(axis=0)

def dumpNorm( x_val, y_val ):
    x_norm = Normalise(x_val)
    for i in range(len( x_norm )):
        print("%4.3f\t %4.3f\t %d"%(x_norm[i][0], x_norm[i][1], y_val[i]))

def checkAccuracy( model, dataloader ):
    
    model = model.eval()
    corr, total_ex = 0.0, 0

    for idx, (features, class_labels) in enumerate(dataloader):

        with torch.no_grad():
             probas = model(features)
        pred = torch.where( probas > 0.5, 1, 0 )

        lab = class_labels.view(pred.shape).to(pred.dtype)

        cmp = lab == pred
        corr += torch.sum(cmp)
        total_ex += len(cmp)

    return corr/total_ex

def checkBoundary( model, x ):
    w1 = model.linear.weight[0][0].detach()
    w2 = model.linear.weight[0][1].detach()
    b = model.linear.bias[0].detach()
    return -w1/w2, -b/w2

def main():

    # Read data #
    try:
       in_file = sys.argv[1]
    except:
       print("\nENTER A FILE\n")
       sys.exit()

    x_val, y_val = readData( in_file )
    #dumpNorm( x_val, y_val )
    x_val = Normalise(x_val)

    # Initial Testing #

    #x = torch.tensor( [ 1.1, 2.1] )
    #torch.manual_seed(1)
    #model = LogRegression( num_features=2 )

    # This construction is used to save memory #   
    # a.k.a  'with torch.inference_mode()' #

    #with torch.no_grad():
    #   proba = model(x)

    # NB. size of minibatches for gradient descent and learning rate are hyperparameters which can be tuned to optimise results #

    minibatch = 10
    learning_rate = 0.05

    train_ds = MyDataset(x_val, y_val)
    train_loader = DataLoader(
               dataset=train_ds,
               batch_size=minibatch,
               shuffle=True,
               )

    torch.manual_seed(1)
    model = LogRegression( num_features=2 )
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    num_epochs = 20

    # Log file can be used to record Loss as a function of Epoch number #

    log = False

    if log:
       f=open( ("log_%s_%s.out"%(minibatch, learning_rate)), "w" )
       f.write("Epoch#\t Loss\n")

    print("\nMinibatch size = %d" %(minibatch))
    print("Learning rate  = %s" %("{:2.1E}".format(learning_rate)))
    print("\nStarting Training...\n")

    for i in range( num_epochs ):
        model = model.train()

        for idx, (features, class_labels) in enumerate(train_loader):

            step = (i)*(x_val.shape[0]/minibatch)+idx+1
            probas = model( features )
            loss = F.binary_cross_entropy( probas, class_labels.view(probas.shape) )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #print("Step = %d" %( (i)*(x_val.shape[0]/minibatch)+idx+1 ))
            print("%s: %.3d/%.3d | %s: %.3d/%.3d | %s: %4.3f" %("Epoch", (i+1), num_epochs, "Batch", idx, len(train_loader), "Loss", (loss)) ) 

        if log:
           f.write("%d\t %6.4f\n" %(i+1, loss))

    acc = checkAccuracy( model, train_loader )

    print("\nModel accuracy = %3.1f%%\n" %(acc*100.0) )

    grad, intcpt = checkBoundary(model, x_val)

    print("Decision boundary -> y = %3.2fx + %3.2f\n" %(grad, intcpt) )

if __name__ == "__main__":
   main()
