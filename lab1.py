import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0,x)

def leaky_relu(x, alpha=0.5):
    return np.where(x>0,x, alpha*x)

def binary_step(x):
    return np.where(x<0,0,1)

def linear(x):
    return 3*x

def softmax(x):
    e_x= np.exp(x-np.max(x))
    return e_x/np.sum(e_x, axis=0)

x= np.array([-5,-4,-3,-2,-1,0,1,2,3,4])
y= leaky_relu(x)
plt.plot(x,y)
plt.xlabel("x")
plt.ylabel("Output using activation function")
plt.title("Activation Function")
plt.show()
