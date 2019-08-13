#smiley/unhappy network

from NeuralNetwork import *
import numpy as np
import json

def run():
    with open("dataset.txt") as json_file:
        data = json.load(json_file)['datas']

    trainingData = []
    trainingOutputs = []
    for d in data:
        #print(data)
        trainingOutputs.append([d['val']])
        n = []
        for row in d['data']:
            for v in row:
                n.append(v)
        trainingData.append(n)

    trainingData = np.array(trainingData)
    trainingOutputs = np.array(trainingOutputs)

    #print(trainingOutputs)

    myNetwork = NeuralNetwork([100,100,50,30,1])
    myNetwork.loadFromJson("faceNet.txt")
    n = myNetwork.trainByAccuracy(trainingData,trainingOutputs,0.04)
    print("Trained for ",n)
    net = {'layers' : []}
    for layer in myNetwork.layers:
        net['layers'].append(layer.synapticWeights.tolist())
    data = {'network' : net}
    with open('faceNet.txt', 'w') as outfile:
        json.dump(data, outfile)

    input()
