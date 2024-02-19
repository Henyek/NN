import numpy as np
import pickle as pkl
import pygame



def RElU(x):
    return np.maximum(0.1*x, x)

def RElU_deriv(x):
    return x > 0

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Network:
    def __init__(self):
        self.W1 = np.random.rand(16,784)*2-1
        self.Z1, self.Z2, self.Z3 = [], [], []
        self.A1, self.A2, self.A3 = [], [], []
        self.B1 = np.random.rand(16,1)*2-1
        self.W2 = np.random.rand(16,16)*2-1
        self.B2 = np.random.rand(16,1)*2-1
        self.W3 = np.random.rand(10,16)*2-1
        self.B3 = np.random.rand(10,1)*2-1

    def prnt(self):
        print(self.W1, self.B1, self.W2, self.B2, self.W3, self.B3)
        
    def Fprop(self, inputs):
        self.Z1 = np.add(self.W1.dot(inputs), self.B1)
        self.A1 = RElU(self.Z1)
        self.Z2 = np.add(self.W2.dot(self.A1), self.B2)
        self.A2 = RElU(self.Z2)
        self.Z3 = np.add(self.W3.dot(self.A2), self.B3)
        self.A3 = sigmoid(self.Z3)
        return self.A3
    
    def Bprop(self, labels, inputs, shape):
        expected = np.zeros((labels.size, 10))
        expected[np.arange(labels.size), labels] = 1
        dZ3 = np.subtract(self.A3, expected.T)
        dW3 = (1 / shape[0]) * dZ3.dot(self.A2.T)
        dB3 = (1 / shape[0]) * np.sum(dZ3)
        dZ2 = self.W3.T.dot(dZ3) * RElU_deriv(self.Z2)
        dW2 = (1 / shape[0]) * dZ2.dot(self.A1.T)
        dB2 = (1 / shape[0]) * np.sum(dZ2)
        dZ1 = self.W2.T.dot(dZ2) * RElU_deriv(self.Z1)
        dW1 = (1 / shape[0]) * dZ1.dot(inputs.T)
        dB1 = (1 / shape[0]) * np.sum(dZ1)
        return(dW3, dB3, dW2, dB2, dW1, dB1)
    
    def update_params(self, dW3, dB3, dW2, dB2, dW1, dB1, alpha):
        self.W1 = self.W1 - alpha * dW1
        self.B1 = self.B1 - alpha * dB1
        self.W2 = self.W2 - alpha * dW2
        self.B2 = self.B2 - alpha * dB2
        self.W3 = self.W3 - alpha * dW3
        self.B3 = self.B3 - alpha * dB3
        
    def get_perdict(self):
        return np.argmax(self.A3, 0)
    
    def get_accu(self, predictions, labels):
        # print(predictions, labels)
        return np.sum(predictions == labels) / labels.size
        
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
    
    # def draw_network(self, SCREEN, pos):
    #     for x in 
        

def draw_image(SCREEN, img, pos):
    temp1, temp2 = 0, 0
    for x in range(len(img)):
        color = img[x]
        if(x % 28 == 0):
            temp1 = 0
            temp2 += 1
        else:
            temp1 += 1
        rect = pygame.Rect(pos[0]+(temp1*5),pos[1]+(temp2*5), 5, 5)
        pygame.draw.rect(SCREEN, (color, color, color), rect)
