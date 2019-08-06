# Neural Network class

import numpy as np

def loadFromFile(fileName):
    pass

class neuron_layer():
    def __init__(self, neuronCount, inputsCount):
        self.synapticWeights = 2* np.random.random((inputsCount,neuronCount))-1

class NeuralNetwork():
    def __init__(self, neuronLayers):
        self.layers = []
        for i in range(len(neuronLayers)-1):
            self.layers.append(neuron_layer(neuronLayers[i+1],neuronLayers[i]))

    def sigmoid(self,x):
        return 1/ (1+np.exp(-x))

    def sigmoid_derivative(self,x):
        return x*(1-x)

    def train(self, trainingInputs, trainingOutputs, count):
        for i in range(count):
            self.train_iteration(trainingInputs, trainingOutputs)

    def write(self,fileName):
        file = open(fileName,"w")
        for layer in self.layers:
            file.write("{")
            for i in layer.synapticWeights:
                file.write("[")
                for j in i:
                    file.write(str(j)+",")
                file.write("]")
            file.write("}")


    def trainByAccuracy(self, trainingInputs, trainingOutputs, threshold):
        count = 0
        while self.test(trainingInputs, trainingOutputs) > threshold:
            self.train_iteration(trainingInputs, trainingOutputs)
            count += 1
        return count


    def train_iteration(self, trainingInputs, trainingOutputs):
        output = self.think(trainingInputs)

        lastlayererr = trainingOutputs-output[-1]
        #print(len(output2))
        lastLayerDelta = lastlayererr*self.sigmoid_derivative(output[-1])

        #print(layer2delta)
        layerDeltas = []
        for i in range(len(self.layers)):
            layerDeltas.append(0)
        layerDeltas[-1] = lastLayerDelta
        #print(layerDeltas[0])
        l = len(self.layers)
        for i in range(2,l+1):
            #print(layerDeltas[-(i)])
            layererr = layerDeltas[l-i+1].dot(self.layers[l-i+1].synapticWeights.T)
            layerdelta = layererr * self.sigmoid_derivative(output[l-i])
            layerDeltas[-i] = layerdelta

        #print(layerDeltas)

        for i in range(len(self.layers)):
            if i > 0:
                self.layers[i].synapticWeights += output[i-1].T.dot(layerDeltas[i])
            else:
        #        print(layerDeltas[i+1])
        #        print(trainingInputs.T)
                self.layers[i].synapticWeights += trainingInputs.T.dot(layerDeltas[i])

    def test(self,inputs, outputs):
        output = self.get(inputs)
        #averageExpected = np.average(outputs)
        arr = outputs-output
        arr[arr<0] = -arr[arr<0]
        accuracy = np.average(arr)
        if accuracy < 0:
            accuracy = -accuracy
        #print(accuracy,"%")
        return accuracy


    def think(self,inputs):
        outs = []
        ran = False
        for layer in self.layers:
         #print(layer.synapticWeights)
            if ran:
                new = self.sigmoid(np.dot(prev,layer.synapticWeights))
            else:
                ran = True
                new = self.sigmoid(np.dot(inputs,layer.synapticWeights))
            prev = new
            outs.append(new)
        return outs

    def get(self,inputs):
        return self.think(inputs)[len(self.layers)-1]

    def printSynapticWeights(self):
        for layer in self.layers:
            print(self.layers.index(layer)+1, layer.synapticWeights)
