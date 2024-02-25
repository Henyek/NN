from functions import *
import pandas as pd

NN = Network()

pygame.init()

SCREEN = pygame.display.set_mode((640, 640))

# dataset = pd.read_csv(r'mnist_train.csv', nrows=100)

# data = np.array(dataset)

# m, n =  data.shape

# labels = data.T[0]
# inputs = data.T[1:n]/255

# inputs_color = data.T[1:n]

# iterations = 10000
# i = 0
# for x in range(iterations):
#     i += 1
#     NN.Fprop(inputs)

#     dW3, dB3, dW2, dB2, dW1, dB1 = NN.Bprop(labels, inputs, (m, n))
#     NN.update_params(dW3, dB3, dW2, dB2, dW1, dB1, 0.1)    
#     if i % 50 == 0:
#         print("Iteration: ", i)
#         predictions = NN.get_perdict()
#         print(NN.get_accu(predictions, labels))



# NN.save_network()

NN.load_network(open('network_data.pickle', 'rb'))
# nmb = 7

# NN.Fprop(np.reshape(inputs.T[nmb], (784, 1)))
# predictions = NN.get_perdict()
# print(predictions)

CurrentImage = [0.0] * 784

draw_state = False

reset_button = pygame.Rect(0, 0, 200, 60)
reset_button.centerx = SCREEN.get_rect().centerx
reset_button.centery = SCREEN.get_rect().height - 70


while True:
    SCREEN.fill((255,255,255))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            quit()
        if event.type == pygame.MOUSEBUTTONDOWN:
            if reset_button.collidepoint(pygame.mouse.get_pos()):
                CurrentImage = [0.0] * 784
            draw_state = True
        if event.type == pygame.MOUSEBUTTONUP:
            draw_state = False


    rects = draw_image(SCREEN, CurrentImage, (100, 100))
    NN.Fprop(np.reshape(CurrentImage, (784, 1)))
    print(NN.get_perdict())

    if draw_state:  
        editor(rects, CurrentImage)
    pygame.draw.rect(SCREEN, (255,0,0), reset_button)
    pygame.display.flip()