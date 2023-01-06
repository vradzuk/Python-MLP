from nn import nn, utils


inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs = [[0], [1], [1], [0]]

network = nn.Network([2, 2, 1], 0.1)

utils.train(network, inputs, outputs)
utils.test(network, 2)