import numpy as np
import pickle as pkl
import pygame
import cupy as cp



def RElU(x):
    return np.maximum(0.1*x, x)


def RElU_CUDA(x):
    return cp.maximum(0.1*x, x)

def RElU_deriv(x):
    return x > 0

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_CUDA(x):
    return 1 / (1 + cp.exp(-x))

def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))

class Network:
    def __init__(self, geometry):
        self.W1 = np.random.rand(geometry[1], geometry[0])*2-1
        self.Z1, self.Z2, self.Z3 = [], [], []
        self.A1, self.A2, self.A3 = [], [], []
        self.B1 = np.random.rand(geometry[1],1)*2-1
        self.W2 = np.random.rand(geometry[2],geometry[1])*2-1
        self.B2 = np.random.rand(geometry[2],1)*2-1
        self.W3 = np.random.rand(geometry[3],geometry[2])*2-1
        self.B3 = np.random.rand(geometry[3],1)*2-1
        
    def init_CUDA(self, geometry):
        self.W1 = cp.random.rand(geometry[1], geometry[0])*2-1
        self.Z1, self.Z2, self.Z3 = [], [], []
        self.A1, self.A2, self.A3 = [], [], []
        self.B1 = cp.random.rand(geometry[1],1)*2-1
        self.W2 = cp.random.rand(geometry[2],geometry[1])*2-1
        self.B2 = cp.random.rand(geometry[2],1)*2-1
        self.W3 = cp.random.rand(geometry[3],geometry[2])*2-1
        self.B3 = cp.random.rand(geometry[3],1)*2-1

    def prnt(self):
        print(self.W1, self.B1, self.W2, self.B2, self.W3, self.B3)
        
    def Fprop(self, inputs):
        self.Z1 = np.add(self.W1.dot(inputs), self.B1)
        self.A1 = sigmoid(self.Z1)
        self.Z2 = np.add(self.W2.dot(self.A1), self.B2)
        self.A2 = RElU(self.Z2)
        self.Z3 = np.add(self.W3.dot(self.A2), self.B3)
        self.A3 = sigmoid(self.Z3)
        return self.A3
    
    def Fprop_CUDA(self, inputs):
        self.Z1 = cp.add(cp.dot(self.W1, inputs), self.B1)
        self.A1 = sigmoid(self.Z1)
        self.Z2 = cp.add(cp.dot(self.W2, self.A1), self.B2)
        self.A2 = RElU(self.Z2)
        self.Z3 = cp.add(cp.dot(self.W3, self.A2), self.B3)
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
        dZ1 = self.W2.T.dot(dZ2) * (sigmoid(self.Z1) * (1 - sigmoid(self.Z1)))
        dW1 = (1 / shape[0]) * dZ1.dot(inputs.T)
        dB1 = (1 / shape[0]) * np.sum(dZ1)
        return(dW3, dB3, dW2, dB2, dW1, dB1)
    
    def Bprop_CUDA(self, labels, inputs, shape):
        expected = cp.zeros((labels.size, 10))
        expected[cp.arange(labels.size), labels] = 1
        dZ3 = cp.subtract(self.A3, expected.T)
        dW3 = (1 / shape[0]) * cp.dot(dZ3, self.A2.T)
        dB3 = (1 / shape[0]) * cp.sum(dZ3)
        dZ2 = cp.dot(self.W3.T, dZ3) * RElU_deriv(self.Z2)
        dW2 = (1 / shape[0]) * cp.dot(dZ2, self.A1.T)
        dB2 = (1 / shape[0]) * cp.sum(dZ2)
        dZ1 = cp.dot(self.W2.T, dZ2) * sigmoid_deriv(self.Z1)
        dW1 = (1 / shape[0]) * cp.dot(dZ1, inputs.T)
        dB1 = (1 / shape[0]) * cp.sum(dZ1)
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
    
    def get_perdict_CUDA(self):
        return cp.argmax(self.A3, 0)
    
    def get_accu(self, predictions, labels):
        return np.sum(predictions == labels) / labels.size    
    
    def get_accu_CUDA(self, predictions, labels):
        return cp.sum(predictions == labels) / labels.size
        
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
    rects = []
    for x in range(len(img)):
        color = img[x]*255
        if(x % 28 == 0):
            temp1 = 0
            temp2 += 1
        else:
            temp1 += 1
        rect = pygame.Rect(pos[0]+(temp1*10),pos[1]+(temp2*10), 10, 10)
        rects.append(rect)
        pygame.draw.rect(SCREEN, (color, color, color), rect)
    return rects

def editor(rects, img):
    for rect_index in range(len(rects)):
        if rects[rect_index].collidepoint(pygame.mouse.get_pos()):
            if img[rect_index] < 1:
                img[rect_index] = 1

            if img[rect_index-1] <= 0.9:
                img[rect_index-1] += 0.05

            if img[rect_index+1] <= 0.9:
                img[rect_index+1] += 0.05

            if img[rect_index-28] <= 0.9:
                img[rect_index-28] += 0.05

            if img[rect_index+28] <= 0.9:
                img[rect_index+28] += 0.05