import numpy as np
class Neuron:
    def __init__(self, num_inputs):
        self.weights= np.random.rand(num_inputs)
        self.bias= np.random.rand()
    
    def activate(self, inputs):
        weighted_output= np.dot(inputs, self.weights)+ self.bias
        activation= self.sigmoid(weighted_output)
        return activation
    
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    def train(self, inputs, target_output, l_r):
        actual_output= self.activate(inputs)
        error= target_output - actual_output
        self.weights+= error*l_r*inputs
        self.bias+= error*l_r

if __name__=="__main__":
    num_inputs=3
    neuron= Neuron(num_inputs)
    X_train= np.array([[0,0,1], [1,1,1], [1,0,1], [0,1,1]])
    y_train= np.array([0,1,1,0])
    x_test= np.array([[0,0,1], [1,1,1], [1,0,1], [0,1,1]])

    l_r=0.1
    num_iter=1000
    for i in range(num_iter):
        index= np.random.randint(len(X_train))
        inputs= X_train[index]
        output= y_train[index]
        neuron.train(inputs, output, l_r)
    
    for x in x_test:
        output= neuron.activate(x)
        print("Input: ", x, "Output: ", output)


