import sys, os
import torch
import torch.nn.functional as F

# Example function which can be used to override PyTorch default weight initialisation #

"""
   def weights_init m
       if isinstance m torch nn Linear
          torch nn init * m weight
          torch nn init * m bias
   model apply weights_init

To use Kaiming initialisation [https://arxiv.org/abs/1502.01852v1].

   nn init kaiming_normal_ m weight data nonlinearity='relu'
   nn init constant_ m bias data 0

"""

def cross_entropy( net_input, y ):
    act = torch.softmax( net_input, dim=1 )
    oh = F.one_hot(y)
    train_loss = - torch.sum( torch.log( act ) * (oh), dim=1 )
    return torch.mean( train_loss )

def main():

    # Softmax activation #
    z = torch.tensor( [ [ 3.1, -2.3, 5.8 ], [1.1, 1.9, -8.9]] )
    sm = F.softmax(z, dim=1)

    # One-hot encoding, True class labels #
    y = torch.tensor( [ 0, 2, 2, 1 ] )
    oh = F.one_hot(y)

    # Net input -> softmax -> one-hot -> class labels #
    net_input = torch.tensor( [ [ 1.5, 0.1, -0.4 ], [ 0.5, 0.7, 2.1 ], [-2.1, 1.1, 0.8], [1.1, 2.5, -1.2] ])

    # NB. dim=1 ensures that normalised probabilities along the row axis sum to 1, dim =0 ensures that column axis probabilities would sum to 1 #
    act = torch.softmax( net_input, dim=1 )

    print( cross_entropy( net_input, y )  )
    print( F.cross_entropy( net_input, y ) )

if __name__ == "__main__":
   main()
