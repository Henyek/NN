from functions import *
import pandas as pd

n = Network()


dataset = pd.read_csv(r'mnist_train.csv', nrows=600)

data = np.array(dataset)

m, n =  data.shape

labels = data.T[0]
inputs = data.T[1:n]

print(labels)
print(inputs)
