import torch

class Perceptron:
    def __init__(self):
        self.weights = torch.tensor([2.86, 1.98])
        self.bias = torch.tensor(-3.0)

    def forward1(self, x):
        weighted_sum_z = torch.dot(x,self.weights) + self.bias
        return torch.where( weighted_sum_z > 0.0, 1.0, 0.0 )

    def forward2(self, x):
        return torch.where( torch.matmul(x, self.weights) + self.bias > 0.0, 1.0, 0.0 )
        
def main():

    X_data = torch.tensor([ [-1.0, -2.0], [-3.0, 4.5], [5.0, 6.0] ])

    # Use torch.where #
    ppn1 = Perceptron()
    out1 = []
    for i in range(len(X_data)):
        out1.append( [ ppn1.forward1( X_data[i] ) ] )
    out1 = torch.tensor( out1 )

    # Use torch.matmul & torch.where #
    ppn2 = Perceptron()
    out2 = ppn2.forward2( X_data )

    print(out1, out2)

if __name__ == "__main__":
   main()
