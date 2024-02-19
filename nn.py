from functions import *
import pandas as pd

NN = Network()

pygame.init()

SCREEN = pygame.display.set_mode((640, 640))

dataset = pd.read_csv(r'mnist_train.csv', nrows=100)

data = np.array(dataset)

m, n =  data.shape

labels = data.T[0]
inputs = data.T[1:n]/255

inputs_color = data.T[1:n]

iterations = 10000
i = 0
for x in range(iterations):
    i += 1
    NN.Fprop(inputs)

    dW3, dB3, dW2, dB2, dW1, dB1 = NN.Bprop(labels, inputs, (m, n))
    NN.update_params(dW3, dB3, dW2, dB2, dW1, dB1, 0.1)    
    if i % 50 == 0:
        print("Iteration: ", i)
        predictions = NN.get_perdict()
        print(NN.get_accu(predictions, labels))



NN.save_network()

# NN.load_network(open('network_data.pickle', 'rb'))
# nmb = 7

# NN.Fprop(np.reshape(inputs.T[nmb], (784, 1)))
# predictions = NN.get_perdict()
# print(predictions)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            quit()
    draw_image(SCREEN, inputs_color.T[12], (25, 25))
    pygame.display.flip()