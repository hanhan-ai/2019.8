"""
命名规则：
        类和文件用帕斯卡命名法
        函数用驼峰命名法
        变量用下划线命名法

"""

import numpy as np


class Network:
    def __init__(
            self,
            size_inputlayer=0,
            size_hiddenlayer_1=0,
            size_hiddenlayer_2=0,
            size_outputlayer=0,

                 ):
        self.layer_input = size_inputlayer
        self.layer_hidden_1 = size_hiddenlayer_1
        self.layer_hidden_2 = size_hiddenlayer_2
        self.layer_output = size_outputlayer

        self.weight_input = 2*np.random.rand(size_inputlayer, size_hiddenlayer_1) - 1
        self.weight_hidden_1 = 2*np.random.rand(size_hiddenlayer_1, size_hiddenlayer_2) - 1
        self.weight_hidden_2 = 2*np.random.rand(size_hiddenlayer_2, size_outputlayer) - 1

        self.bias_input = np.random.rand(size_hiddenlayer_1)
        self.bias_hidden_1 = np.random.rand(size_hiddenlayer_2)
        self.bias_hidden_2 = np.random.rand(size_outputlayer)

        self.mutate_freq = 0.5
        self.final_mutate_freq = 0.01
        self.mutate_freq_decay_rate = 0.7
        self.mutate_freq_decay_step = 1000

        self.generation = 1

    def decrease_mutate_freq(self):
        if self.mutate_freq > self.final_mutate_freq:
            self.mutate_freq = self.mutate_freq*self.mutate_freq_decay_rate ^ \
                               (self.mutate_freq_decay_step/self.generation)


    def activationFunc(self, name, x):
        if name == 'relu':
            return np.maximum(0, x)
        elif name == 'sigmoid':
            y = 1/(1 + np.exp(-x))
            return y
        elif name == 'tanh':
            y = (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
            return y

    def run(self, input_vector):

        out_input = np.dot(input_vector, self.weight_input) + self.bias_input
        activated_input = self.activationFunc('relu', out_input)

        out_hidden_1 = np.dot(activated_input, self.weight_hidden_1)
        activated_hidden_1 = self.activationFunc('relu', out_hidden_1)

        out_hidden_2 = np.dot(activated_hidden_1, self.weight_hidden_2)
        activated_hidden_2 = self.activationFunc('tanh', out_hidden_2)

        out_result = activated_hidden_2

        return out_result


def initNetwork(i, h1, h2, o):
    net = Network(i, h1, h2, o)
    return net


def cross(network_1, network_2):

    crossed_network = Network(
        network_1.layer_input,
        network_1.layer_hidden_1,
        network_1.layer_hidden_2,
        network_1.layer_output
                              )

    crossed_network.generation = network_1.generation + 1

    crossed_network.weight_input = (network_1.weight_input + network_2.weight_input)/2
    crossed_network.weight_hidden_1 = (network_1.weight_hidden_1 + network_2.weight_hidden_1)/2
    crossed_network.weight_hidden_2 = (network_1.weight_hidden_2 + network_2.weight_hidden_2)/2

    crossed_network.bias_input = (network_1.bias_input + network_2.bias_input)/2
    crossed_network.bias_hidden_1 = (network_1.bias_hidden_1 + network_2.bias_hidden_1)/2
    crossed_network.bias_hidden_2 = (network_1.bias_hidden_2 + network_2.bias_hidden_2)/2

    return crossed_network


def mutate(origin_network):

    for i in range(origin_network.layer_input*origin_network.layer_hidden_1):
        if np.random.rand() < origin_network.mutate_freq:
            origin_network.weight_input[np.random.randint(origin_network.layer_input),
                                        np.random.randint(origin_network.layer_hidden_1)] \
                                        *= 2*np.random.rand()

    for i in range(origin_network.layer_hidden_1*origin_network.layer_hidden_2):
        if np.random.rand() < origin_network.mutate_freq:
            origin_network.weight_hidden_1[np.random.randint(origin_network.layer_hidden_1),
                                        np.random.randint(origin_network.layer_hidden_2)] \
                                        *= 2*np.random.rand()

    for i in range(origin_network.layer_hidden_2*origin_network.layer_output):
        if np.random.rand() < origin_network.mutate_freq:
            origin_network.weight_hidden_2[np.random.randint(origin_network.layer_hidden_2),
                                        np.random.randint(origin_network.layer_output)] \
                                        *= 2*np.random.rand()
    return origin_network



# test
network_1 = initNetwork(2, 4, 5, 6)
network_2 = initNetwork(2, 4, 5, 6)

crossed = cross(network_1, network_2)
print('weight input: \n', crossed.weight_input)
print('weight h1: \n', crossed.weight_hidden_1)
print('weight h2: \n', crossed.weight_hidden_2)
print('layer_input :\n', crossed.layer_input)
crossed = mutate(crossed)
print('mutated weight input: \n', crossed.weight_input)
print('mutated weight h1: \n', crossed.weight_hidden_1)
print('mutated weight h2: \n', crossed.weight_hidden_2)






