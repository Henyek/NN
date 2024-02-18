import numpy as np
import pickle as pkl



def RElU(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Network:
    def __init__(self):
        self.W1 = np.random.rand(16,784)
        self.B1 = np.random.rand(16)
        self.W2 = np.random.rand(16,16)
        self.B2 = np.random.rand(16)
        self.W3 = np.random.rand(10,16)
        self.B3 = np.random.rand(10)

    def prnt(self):
        print(self.W1, self.B1, self.W2, self.B2, self.W3, self.B3)
        
    def Fprop(self, inputs):
        A1, A2, A3 = [], [], []
        for w, b in zip(self.W1, self.B1):
            A1 = RElU((inputs * w) + b)
        for w, b in zip(self.W2, self.B2):
            A2 = RElU((A1 * w) + b)
        for w, b in zip(self.W3, self.B3):
            A3 = sigmoid((A2 * w) + b)
            
        return A3
        
    def save_network(self):
        with open('network_data.pickle', 'wb') as file:
            pkl.dump(self, file)
        
        
    def load_network(self, file):
        old = pkl.load(file)
        self.W1 = old.W1
        self.B1 = old.B1
        self.W2 = old.W2
        self.B2 = old.B2
        self.W3 = old.W3
        self.B3 = old.B3