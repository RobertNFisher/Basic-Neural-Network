import Neural_Network
import numpy as np

precept = Neural_Network.precept_neural_network()
multi_layer_network = Neural_Network.multi_layer_network()
epochs = 30
data = np.zeros((16,4))

index = 0
# create a list of training data that consists of all possible combination of dark and light squares
for zero in range(-1,2,2):
    for one in range(-1,2,2):
        for two in range(-1,2,2):
            for three in range(-1,2,2):
                data[index] = [zero,one,two,three]
                index += 1

# create a list where [0] is the inputs and [1] are the expected values
# since the first 5 data are the only ones that classify as dark, we tag it as such
train_data = []
for index in range(0, data.__len__()):
    # if there are 0 or 1 light squares, the results = dark
    if(index <= 5):
        train_data.append([data[index], -1, [0,1]])
    # if there are more than 1 light squares, the results = light
    else:
        train_data.append([data[index], 1, [1,0]])

index = 0
# run data through sampling
for epoch in range(0, epochs):
    accuracy = 0
    multi_layer_acuracy = 0.0
    for train in range(0, train_data.__len__()):
        precept.train(train_data[train])
        multi_layer_network.train(train_data[train])


    for test in range(0, train_data.__len__()):
        accuracy += precept.test(train_data[test])
        multi_layer_acuracy += multi_layer_network.test(train_data[test])

    accuracy /= train_data.__len__()
    multi_layer_acuracy /= train_data.__len__()
    print("Single-layer Accuracy for current epoch{}: {}".format(epoch + 1, accuracy))
    print("Multi-layer Accuracy for current epoch{}: {}".format(epoch+1, multi_layer_acuracy))
