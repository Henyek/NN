from functions import *
import pandas as pd
from matplotlib import pyplot
import time
# import cupy as cp

geometry = [10, 256, 1028, 3072]

NN = Network(geometry)
# NN.init_CUDA(geometry)

pygame.init()

SCREEN = pygame.display.set_mode((640, 640))

# dataset = pd.read_csv(r'mnist_train.csv', nrows=60000)

# batch_label, labels, data, filenames

data1 = unpickle(f"data_batch_1")
data2 = unpickle(f"data_batch_2")
data3 = unpickle(f"data_batch_3")
data4 = unpickle(f"data_batch_4")
data5 = unpickle(f"data_batch_5")

labels = np.concatenate([data1[b'data'], data2[b'data'], data3[b'data'], data4[b'data'], data5[b'data']])/255
inputs_temp = np.concatenate([data1[b'labels'], data2[b'labels'], data3[b'labels'], data4[b'labels'], data5[b'labels']])

# print(labels)

inputs = np.zeros((inputs_temp.size, 10))
inputs[np.arange(inputs_temp.size), inputs_temp] = 1

inputs = inputs.T


m, n = inputs.shape

# print(inputs.shape)


# data = np.array(dataset)

# m, n =  data.shape

# labels = data.T[0]
# inputs = data.T[1:n]/255

# print(inputs.shape)

# inputs_color = data.T[1:n]

iterations = 25
i = 0
fitness = 0

# timer = time.time()

# NN.load_network(open('network_data_16_16.pickle', 'rb'))

# for x in range(iterations):
#     i += 1
#     NN.Fprop_CUDA(inputs)

#     dW3, dB3, dW2, dB2, dW1, dB1 = NN.Bprop_CUDA(labels, inputs, (m, n))
#     NN.update_params(dW3, dB3, dW2, dB2, dW1, dB1, 0.3)    
#     if i % 20 == 0:
#         print("Iteration: ", i)
#         predictions = NN.get_perdict_CUDA()
#         fitness = NN.get_accu_CUDA(predictions, labels)
#         print(fitness)

CurrentImage = []


SCREEN.fill((255,255,255))
pygame.display.flip()
while True:
    
    SCREEN.fill((255,255,255))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            quit()
        
    while i < iterations:
        i += 1
        NN.Fprop(inputs)

        dW3, dB3, dW2, dB2, dW1, dB1 = NN.Bprop(labels, inputs, (m, n))
        NN.update_params(dW3, dB3, dW2, dB2, dW1, dB1, 0.2)    
        if i % 1 == 0:
            print("Iteration: ", i)
            # predictions = NN.get_perdict()
            # fitness = NN.get_accu(predictions, labels)
            # print(fitness)

            CurrentImage = NN.Fprop(np.array(np.reshape([0,1,0,0,0,0,0,0,0,0], (10, 1))))
            rects = draw_image(SCREEN, CurrentImage, (100, 100))
            pygame.display.flip()

# print(time.time() - timer)

# NN.save_network()

# NN.load_network(open('network_data.pickle', 'rb'))
# nmb = 7

# NN.Fprop(np.reshape(inputs.T[nmb], (784, 1)))
# predictions = NN.get_perdict()
# print(predictions)

# NN.Fprop(np.reshape(inputs.T[1], (784,1)))

# pyplot.ion()

# for x in range(16):
#     New_im = sum(np.reshape(np.subtract(NN.W2[x], NN.B1[x]), (16, 1)) * NN.W1)
#     pyplot.imshow(np.reshape(NN.W1[x], (28,28)), interpolation='nearest')
#     pyplot.show()
#     pyplot.pause(1)  

# CurrentImage = [0.0] * 784

# draw_state = False

# reset_button = pygame.Rect(0, 0, 200, 60)
# reset_button.centerx = SCREEN.get_rect().centerx
# reset_button.centery = SCREEN.get_rect().height - 70 


# while True:
#     SCREEN.fill((255,255,255))
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             quit()
#         if event.type == pygame.MOUSEBUTTONDOWN:
#             if reset_button.collidepoint(pygame.mouse.get_pos()):
#                 CurrentImage = [0.0] * 784
#             draw_state = True
#         if event.type == pygame.MOUSEBUTTONUP:
#             draw_state = False
#     rects = draw_image(SCREEN, CurrentImage, (100, 100))
#     NN.Fprop(np.reshape(np.array(CurrentImage), (784, 1)))
#     print(NN.get_perdict())

#     if draw_state:  
#         editor(rects, CurrentImage)
#     pygame.draw.rect(SCREEN, (255,0,0), reset_button)
#     pygame.display.flip()