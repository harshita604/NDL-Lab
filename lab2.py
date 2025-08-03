import numpy as np
class Neuron:
    def __init__(self, num_inputs):
        self.weights= np.random.rand(num_inputs)
    
    def activate(self, inputs):
        weighted_output= np.dot(inputs, self.weights)
        return weighted_output
    
    def learn_hebbian(self,inputs, l_r):
        activation= self.activate(inputs)
        self.weights+= l_r*activation*inputs
    
if __name__=="__main__":
    num_inputs=3
    neuron= Neuron(num_inputs)
    inputs= np.array([0.5,0.3,0.2])
    l_r=0.1
    num_iter=1000
    for i in range(num_iter):
        neuron.learn_hebbian(inputs, l_r)
    print(neuron.weights)