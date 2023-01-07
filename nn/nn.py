import random
import math


class Network:
    
    def __init__(self, sizes, lr=0.1):
        
        self.lr = lr
        
        self.num_layers = len(sizes)
        self.sizes = sizes
        
        self.layers = []
        for i, size in enumerate(sizes):
            self.layers.append(Layer(self, i, size))
        
    def clear_deriv(self):
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.d = 0
        
    def prop(self, input):
        for i, neuron in enumerate(self.layers[0].neurons):
            neuron.value = input[i]
        return [neuron.val() for neuron in self.layers[-1].neurons]
    
    def backprop(self, output):
        self.clear_deriv()
        for layer in reversed(self.layers[1:]):
            for neuron in layer.neurons:
                if layer.is_last():
                    neuron.d = 2 * (neuron.value - output[neuron.index])
                neuron.d *= sigd(neuron.value)
                neuron.bias -= neuron.d * self.lr
                for w in range(len(neuron.weights)):
                    prev_neuron = layer.prev().neurons[w]
                    neuron.weights[w] -= neuron.d * prev_neuron.value * self.lr
                    prev_neuron.d += neuron.d * neuron.weights[w]
        return sum((neuron.value - output[neuron.index])**2 for neuron in self.layers[-1].neurons)


class Layer:
    
    def __init__(self, network, index, size):
        
        self.network = network
        self.index = index
        
        self.size = size
        self.neurons = [Neuron(self, i) for i in range(size)]
    
    def is_last(self):
        return self.index == self.network.num_layers - 1
    
    def prev(self):
        if self.index:
            return self.network.layers[self.index-1]
        
    def next(self):
        if self.index < self.network.num_layers - 1:
            return self.network.layers[self.index+1]


class Neuron:
    
    def __init__(self, layer, index):
        
        self.layer = layer
        self.index = index
        
        self.value = None
        
        self.bias = 0
        self.weights = [random.random()*2 - 1
                        for _ in range(layer.prev().size)] if layer.index else []
        
        self.d = None

    def val(self):
        if not self.layer.index:
            return self.value
        self.value = sig(sum((neuron.value if self.layer.index==1 else neuron.val()) * self.weights[i]
                             for i, neuron in enumerate(self.layer.prev().neurons))
                         + self.bias)
        return self.value

    
def sig(x):
    return 1/(1+math.exp(-x))

def sigd(x):
    return x*(1-x)