import os, sys, timeit
import torch
import numpy as np
import random as rnd

def vanilla(x,w,b):
    out = b
    for a,b in zip(x,w):
        out += a*b
    return out

def PyTorch(x,w,b):
    return x.dot(w) + b 

def speedTest1():

    b = 0.0
    x = [ rnd.random() for x in range(1000) ]
    w = [ rnd.random() for x in range(1000) ]

    v_st = timeit.default_timer()
    dump = vanilla(x,w,b)
    v_end = timeit.default_timer()
    print("Vanilla Python: Elapsed = %s µs" %("{:6.4E}".format(v_end-v_st)) )

    b = torch.tensor( b )
    x = torch.tensor( x )
    w = torch.tensor( w )

    t_st = timeit.default_timer()
    dump = PyTorch(x,w,b)
    t_end = timeit.default_timer()
    print("Torch: Elapsed = %s µs" %("{:6.4E}".format(t_end-t_st)) )
    print("Speed up = %3.2f" %( (v_end-v_st)/(t_end-t_st) ) )

def vanillaM(X,w,b):
    outs = []
    for i in range(len(X)):
        out = b
        for j in range(len(X[i])):
            out += X[i][j]*w[j]
        outs.append( out )
    return outs

def PyTorchM(X,w,b):
    return X.matmul(w) + b

def speedTest2():

    b = 0.0
    X = [ [1.2, 2.2], [4.4, 5.5] ]
    w = [3.3, 4.3]

    v_st = timeit.default_timer()
    dump = vanillaM(X,w,b)
    v_end = timeit.default_timer()

    print("Vanilla Python: Elapsed = %s µs" %("{:6.4E}".format(v_end-v_st)) )

    b = torch.tensor( [0.0] )    
    X = torch.tensor( [ [1.2, 2.2], [4.4, 5.5] ] )
    w = torch.tensor( [3.3, 4.3] )

    t_st = timeit.default_timer()
    dump = PyTorchM(X,w,b)
    t_end = timeit.default_timer()
    print("Torch: Elapsed = %s µs" %("{:6.4E}".format(t_end-t_st)) )
    print("Speed up = %3.2f" %( (v_end-v_st)/(t_end-t_st) ) )

def vanillaMat(X,W,b):
    W = W.T
    out = np.zeros( (len(X), len(W[0])), dtype=np.float64 )
    for i in range(len( X )):
        for j in range(len( W[0] )):
            for k in range(len( W )):
                out[i][j] += X[i][k] * W[k][j]
    return out

def TorchMat(X,W,b):
    return torch.matmul(X, W.T)

def speedTest3():

    b = 0.0
    X = [ [ rnd.random() for x in range(100) ] for y in range(50) ]
    W = np.asarray([ [ rnd.random() for x in range(100) ] for y in range(50) ], dtype=np.float64 )
    
    v_st = timeit.default_timer()
    out = vanillaMat(X,W,b)
    v_end = timeit.default_timer()

    print("Vanilla Python: Elapsed = %s µs" %("{:6.4E}".format(v_end-v_st)) )

    X = torch.tensor( X )
    W = torch.tensor([ [ rnd.random() for x in range(100) ] for y in range(50) ] )

    t_st = timeit.default_timer()
    dump = TorchMat(X,W,b)
    t_end = timeit.default_timer()

    print("Torch: Elapsed = %s µs" %("{:6.4E}".format(t_end-t_st)) )
    print("Speed up = %3.2f" %( (v_end-v_st)/(t_end-t_st) ) )

def main():

    # Dot product Torch #

    #speedTest()

    # Matrix Multiplication #

    speedTest2()

    # Matrix multiplication with weights matrix. #

    #speedTest3()

    # Broadcasting with Torch #

    a = torch.tensor([1.1, 2.1, 3.1, 4.1])
    b = torch.tensor([5.4, 5.5, 5.6, 5.7])

    print( a + b )

    A = torch.tensor([[1.1, 2.1, 3.1, 4.1],
                  [1.2, 2.2, 3.2, 4.2]])
    b = torch.tensor([5.4, 5.5, 5.6, 5.7])

    print( A + b )

if "__name__" == main():
  main()
