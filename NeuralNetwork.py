# Neural Network class

import numpy as np

class neuron_layer():
    def __init__(self, neuronCount, inputsCount):
        self.synapticWeights = 2* np.random.random((inputsCount,neuronCount))-1

class NeuralNetwork():
    def __init__(self, layer1, layer2):
        self.layer1 = layer1
        self.layer2 = layer2

    def sigmoid(self,x):
        return 1/ (1+np.exp(-x))

    def sigmoid_derivative(self,x):
        return x*(1-x)

    def train(self, trainingInputs, trainingOutputs, count):
        for i  in range(count):
            output1,output2 = self.think(trainingInputs)

            layer2err = trainingOutputs-output2
            #print(len(output2))
            layer2delta = layer2err*self.sigmoid_derivative(output2)

            #print(layer2delta)

            layer1err = layer2delta.dot(self.layer2.synapticWeights.T)
            layer1delta = layer1err * self.sigmoid_derivative(output1)

            layer1adjust = trainingInputs.T.dot(layer1delta)
            layer2adjust = output1.T.dot(layer2delta)

            self.layer1.synapticWeights += layer1adjust
            self.layer2.synapticWeights += layer2adjust


    def think(self,inputs):
        layer1out = self.sigmoid(np.dot(inputs,self.layer1.synapticWeights))
        layer2out = self.sigmoid(np.dot(layer1out,self.layer2.synapticWeights))
        return layer1out, layer2out

    def printSynapticWeights(self):
        print("Layer 1)",self.layer1.synapticWeights)
        print("Layer 2)",self.layer2.synapticWeights)
