import os, sys
import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F

from collections import Counter
from helper_plotting import plot_training_loss

from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import torchvision

class PyTorchMLP(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.all_layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(num_features, 50),
            torch.nn.ReLU(),
            # 2nd hidden layer
            torch.nn.Linear(50, 25),
            torch.nn.ReLU(),
            # output layer
            torch.nn.Linear(25, num_classes),
        )

    def forward(self, x):
        # Converting 28*28 pixel images into 784 length vector #
        x = torch.flatten(x, start_dim=1)
        logits = self.all_layers(x)
        return logits

def chkDistribution( train_loader, val_loader, test_loader ):
    train_counter = Counter()
    for images, labels in train_loader:
        train_counter.update(labels.tolist())
    
    print("Training label distribution:")
    print(sorted(train_counter.items()))
    val_counter = Counter()
    for images, labels in val_loader:
        val_counter.update(labels.tolist())
    
    print("\nValidation label distribution:")
    print(sorted(val_counter.items()))
    test_counter = Counter()
    for images, labels in test_loader:
        test_counter.update(labels.tolist())

    print("\nTest label distribution:")
    print(sorted(test_counter.items()))

    majority_class = test_counter.most_common(1)[0]
    print("\nMajority class:", majority_class[0])

    baseline_acc = majority_class[1] / sum(test_counter.values())
    print("Accuracy when always predicting the majority class:")
    print("Baseline Accuracy = %3.2f %%\n"%(baseline_acc*100.0))

def createLoaders( training_set, validation_set, test_set ):
    train_loader = DataLoader(dataset=training_set,batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset=validation_set,batch_size=64, shuffle=False)
    test_loader = DataLoader(dataset=test_set,batch_size=64, shuffle=False)
    return train_loader, val_loader, test_loader

def visualiseData( train_loader ):
    for images, labels in train_loader:  
        break
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training images")
    plt.imshow(np.transpose(torchvision.utils.make_grid(images[:64],padding=1,pad_value=1.0,normalize=True),(1, 2, 0)))
    plt.show()

def getAcc(model, dataloader):
    model = model.eval()
    
    correct = 0.0
    total_examples = 0
    
    for idx, (features, labels) in enumerate(dataloader):
        with torch.inference_mode(): # basically the same as torch.no_grad
            logits = model(features)
        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)
    return correct / total_examples

def main():

   # Import data #
   train_dataset = datasets.MNIST(
   root="./mnist", train=True, transform=transforms.ToTensor(), download=True)
   test_dataset = datasets.MNIST(
   root="./mnist", train=False, transform=transforms.ToTensor())

   print("\n# Training examples = %d | # Test examples = %d\n"%(len(train_dataset), len(test_dataset)))

   torch.manual_seed(1)
   train_dataset, val_dataset = random_split(train_dataset, lengths=[55000,5000])
   train_loader, val_loader, test_loader = createLoaders( train_dataset, val_dataset, test_dataset )
   #chkDistribution( train_loader, val_loader, test_loader )
   #visualiseData( train_loader )

   # 28*28 image -> 784 length vector. Needs to classifiy each image to a digit 0 - 9. #
   torch.manual_seed(1)
   model = PyTorchMLP(num_features=784, num_classes=10)
   optimizer = torch.optim.SGD(model.parameters(), lr=0.05) # Stochastic grad. desc.
   
   num_epochs = 10

   loss_list = []
   train_acc_list, val_acc_list = [], []

   f=open("accuracy.dat", "w")
   f.write("Epoch#,Training Accuracy,Validation Accuracy\n")
   for i in range( num_epochs ):
       model = model.train()
       for idx, (features, labels) in enumerate(train_loader):
           logits = model(features)
           loss = F.cross_entropy(logits, labels)

           optimizer.zero_grad()
           loss.backward()
           optimizer.step()

           if not idx%250:
              print("Epoch #%.2d/%.2d | Batch #%.3d/%.3d | Train Loss = %6.4f" %(i+1, num_epochs, idx, len(train_loader), loss))

           loss_list.append(loss.item())

       train_acc = getAcc( model, train_loader )
       val_acc = getAcc( model, val_loader )
       print("\nTraining set accuracy = %3.2f %% | Validation set accuracy = %3.2f %%\n" %(train_acc*100.0, val_acc*100.0))
       f.write("%d,%6.4f,%6.4f\n"%(i+1,train_acc, val_acc))
       train_acc_list.append( train_acc )
       val_acc_list.append( val_acc )

   f.close()

   train_acc = getAcc( model, train_loader )
   val_acc = getAcc( model, val_loader )
   test_acc = getAcc(model, test_loader)

   print("Training Acc   = %3.2f %%" %(train_acc*100.0) )
   print("Validation Acc = %3.2f %%" %(val_acc*100.0) )
   print("Test Acc       = %3.2f %%\n" %(test_acc*100.0) )

   f=open("training_loss.dat", "w")
   f.write("#Iterations,Loss\n")
   [ f.write("%d,%6.4f\n"%(i+1,x)) for i,x in enumerate(loss_list) ]
   f.close()

   averaging_iterations=100
   average = np.convolve(loss_list,np.ones(averaging_iterations)/averaging_iterations,mode="valid")

   f=open("averaged_loss.dat", "w")
   [ f.write("%d,%6.4f\n"%(i+1,x)) for i,x in enumerate(average) ]
   f.close()
   
   #plot_training_loss( minibatch_loss_list=loss_list, num_epochs=num_epochs, iter_per_epoch=(len(loss_list)/num_epochs), results_dir="./" )

if __name__ == "__main__":
   main()
