import random
import numpy


def sigmoid_function(number):
    return 1/(1+2.71828**-number)


def convert(array):
    size = array.size
    x = 0
    while x < size:
        array[x][0] = sigmoid_function(array[x][0])
        x += 1
    return array


class NeuralNetwork():
    def __init__(self, inputs, outputs):
        self.hidden = round(inputs * 0.6666, 0)
        self.inputs = inputs
        self.outputs = outputs
        self.input_hidden_weights = numpy.rand((inputs.size, 1))
        self.hidden_output_weights = numpy.rand((self.hidden, 1))

    def FeedForward(self):
        self.hidden_layer = convert(numpy.dot(self.inputs, self.input_hidden_weights))
        self.output_layer = convert(numpy.dot(self.hidden_layer, self.hidden_output_weights))





