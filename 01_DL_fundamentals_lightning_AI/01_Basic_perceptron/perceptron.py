import os,sys
import numpy as np
import random as rnd

# Initialising Weights and Biases with random numbers improves the training! #

class Perceptron:

      def __init__(self, num_features):
          rnd.seed(123)
          self.num_features = num_features
          self.weights = np.zeros( num_features, dtype=np.float64)
          self.weights = [ rnd.uniform(-0.5, 0.5) for x in self.weights ]
          self.bias = rnd.uniform(-0.5,0.5)

      def forward(self, x):
          w_sum = self.bias
          for i in range(len(self.weights)):
              w_sum += x[i]*self.weights[i]
          if w_sum > 0.0:
             pred = 1
          else:
             pred = 0
          return pred

      def update(self, x, true_y):
          pred = self.forward(x)
          err = true_y - pred
          self.bias += err
          for i in range(len(self.weights)):
              self.weights[i] += err*x[i]
          return err

      def decision_boundary(self, x):
          grad = (-self.weights[0]*x[0][0])/self.weights[1]
          intercept = -self.bias/self.weights[1]
          return grad, intercept

def readData( in_file ):
    dat = np.genfromtxt(in_file, delimiter="\t")
    x_values = [ [ x[0],x[1] ] for x in dat ]
    y_values = [ x[-1] for x in dat ]
    return x_values, y_values

def checkAcc(model, x, y):
    corr = 0.0
    for i in range(len(x)):
        pred = model.forward(x[i])
        if y[i] == pred:
           corr += 1.0
    return corr/len(y)

def train(model, all_x, all_y, epochs, bnd=False):
    for i in range(epochs):
        err_cnt = 0

        for x,y in zip(all_x, all_y):
            err = model.update(x, y)
            err_cnt += abs(err)

        print("Epoch #" + str(i+1) + " Errors " + str(err_cnt))
        acc = checkAcc( model, all_x, all_y)
        print("Model accuracy = " + str(acc*100))
        print("\n")
        if acc == 1.0:
           print("Model 100% accurate, ending training.")
           break
        #print( model.weights )

    if bnd:
       m, c = model.decision_boundary(all_x)
       print("Decision boundary: y = " + str(m) + "x + " + str(c) )
   
def main():

    # Read data 
    try:
       in_file = sys.argv[1]
    except:
       print("\nENTER A FILE\n")
       sys.exit()

    x_val, y_val = readData( in_file )
    ppn = Perceptron(num_features=2)
    train(ppn, x_val, y_val, 10)

    accuracy = checkAcc( ppn, x_val, y_val )

    print( ("Model accuracy = %2.1f %%"%(accuracy*100.0)) )

    sys.exit()

    # Testing #

    x = [1.1, 1.2]

    # Initial weights and biases
    print(ppn.weights, ppn.bias)

    # Test prediction
    print(ppn.forward(x))

    # Weight update 
    print(ppn.update(x, true_y=1.0))

    # Check new weights
    print(ppn.weights, ppn.bias)

if __name__ == "__main__":
   main()
