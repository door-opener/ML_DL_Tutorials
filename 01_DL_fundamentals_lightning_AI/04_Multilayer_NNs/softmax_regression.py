import os, sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class MyDataset(Dataset):

      def __init__(self, X, y):
          self.features = X
          self.labels = y

      def __getitem__(self, index):
          x = self.features[index]
          y = self.labels[index]
          return x,y

      def __len__(self):
          return self.labels.shape[0]

class SMRegression(torch.nn.Module):

      def __init__(self, num_features):
          super().__init__()
          self.linear = torch.nn.Linear( num_features, 3 )

      def forward(self, x):
          logits = self.linear(x)
          probas = torch.nn.functional.softmax(logits)
          return probas

def readData( in_file ):
    dat = np.genfromtxt(in_file, delimiter=",")
    x_values = [ [ x[0],x[1],x[2],x[3] ] for x in dat ]
    y_values = [ x[-1] for x in dat ]
    return torch.tensor(x_values, dtype=torch.float32), torch.tensor(y_values, dtype=torch.float32)

def Normalise( x_data, default=True):
    if default:
       return (x_data - x_data.mean(axis=0)) / x_data.std(axis=0)
    else:
       return (x_data - torch.min(x_data)) / (torch.max(x_data) - torch.min(x_data))

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

def learnStep( train_loader, params, verbose=False ):

    minibatch, learning_rate, num_epochs = params[0], params[1], params[2]

    model = LogRegression( num_features=4 )
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Log file can be used to record Loss as a function of Epoch number #

    log = False
    if log:
       f=open( ("log_%s_%s.out"%(minibatch, learning_rate)), "w" )
       f.write("Epoch#\t Loss\n")

    if verbose:
       print("\nMinibatch size   = %d" %(minibatch))
       print("Number of epochs = %d" %(minibatch))
       print("Learning rate  = %s" %("{:2.1E}".format(learning_rate)))

    for i in range( num_epochs ):
        model = model.train()

        for idx, (features, class_labels) in enumerate(train_loader):

            probas = model( features )
            loss = F.binary_cross_entropy( probas, class_labels.view(probas.shape) )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if verbose:
               if not idx%20:
                  print("%s: %.3d/%.3d | %s: %.3d/%.3d | %s: %4.3f" %("Epoch", (i+1), num_epochs, "Batch", idx, len(train_loader), "Loss", (loss)) ) 

        if log:
           f.write("%d\t %6.4f\n" %(i+1, loss))

    acc = checkAccuracy( model, train_loader )
    #grad, intcpt = checkBoundary(model, x_features)
    #print("Decision boundary -> y = %3.2fx + %3.2f\n" %(grad, intcpt) )
    return acc*100.0, model

def main():

    # Read data #
    try:
       in_file = sys.argv[1]
    except:
       print("\nENTER A FILE\n")
       sys.exit()

    x_val, y_val = readData( in_file )
    x_val = Normalise(x_val, default=True)

    # Split the dataset into training and validation sets #
    # 80% of data for training, 20% for validation #

    minibatch = 10

    train_size = int(x_val.shape[0]*0.8)
    val_size = x_val.shape[0] - train_size

    dataset = MyDataset( x_val, y_val )

    torch.manual_seed(1)

    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(dataset=train_set, batch_size=minibatch, shuffle=True )
    val_loader = DataLoader(dataset=val_set, batch_size=minibatch, shuffle=False )

    ########################################################

    learning_rates = np.linspace( 0.05, 0.2, 4 )
    num_epochs = np.linspace( 5, 20, 4, dtype=np.int8 )
    lr_opt, epoch_opt = 0.0, 0.0
    model_opt = None

    print("\nStarting Training...\n")

    for i in range(len( num_epochs )):
        for j in range(len( learning_rates)):
            step = i*len(num_epochs) + j+1
            acc, model = learnStep( train_loader, [ minibatch, learning_rates[j], num_epochs[i] ] )
            print("Step #%.2d | LR = %3.2f | # Epochs = %.2d | Accuracy = %5.4f" %(step, learning_rates[j], num_epochs[i], acc) )

            if acc > 98.0:
               lr_opt, epoch_opt = learning_rates[j], num_epochs[i]
               model_opt = model
               break 

        if lr_opt and epoch_opt: 
           break

    print("\n----------------------------------------------------------")
    print("\nPrediction accuracy over 98% on training set achieved.\n")
    print("----------------------------------------------------------")
    print("\nLearning rate = %3.2f | Num. epochs = %d\n" %(lr_opt, epoch_opt))
    print("----------------------------------------------------------")

    val_acc = checkAccuracy( model_opt, val_loader )

    print("\nPrediction accuracy on validation set = %3.2f%%\n" %(val_acc*100.0) )

if __name__ == "__main__":
   main()
