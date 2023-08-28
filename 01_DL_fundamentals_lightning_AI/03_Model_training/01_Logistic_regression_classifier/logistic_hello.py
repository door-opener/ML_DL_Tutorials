import os, sys
import torch
import torch.nn.functional as F
from torch.autograd import grad

def main():

    # Model params
    w_1 = torch.tensor( [0.23], requires_grad=True )
    b = torch.tensor( [0.1], requires_grad=True )

    # Inputs, target
    x_1 = torch.tensor( [1.23] )
    y = torch.tensor( [1.] )

    # Compute the weighted sum
    u = x_1*w_1
    z = u + b

    # Sigmoid activation fn.
    #a = torch.sigmoid(z)

    # Loss function i.e. binary cross entropy
    #l = F.binary_cross_entropy(a, y)

    # Using 'with_logits' avoids having to apply sigmoid fn.
    l = F.binary_cross_entropy_with_logits(z, y)
  
    # Gradient of loss w.r.t w_1
    grad_L_w1 = grad(l, w_1, retain_graph=True)

    # Gradient of loss w.r.t bias
    grad_L_b = grad(l, b, retain_graph=True)

    # .backward() calculates the gradient without calling grad
    l.backward()

    print(grad_L_w1, w_1.grad)
    print(grad_L_b, b.grad)

if __name__ == "__main__":
   main()
