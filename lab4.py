import numpy as np
from scipy.linalg import lstsq
import matplotlib.pyplot as plt

def rbf(x,c,s):
    return np.exp(-np.linalg.norm(x-c, axis=1)**2/(2*s**2)) #imp

class RBFNetwork:
    def __init__(self, centers, sigma, regularization=0):
        self.centers= centers
        self.sigma= sigma
        self.regularization= regularization
        self.weights= None
    
    def construct_phi(self,X):
        phi= np.zeros((X.shape[0], len(self.centers))) #imp
        for i,c in enumerate(self.centers):
            phi[:,i]= rbf(X,c,self.sigma)
        return phi
    
    def train(self,X,y):
        phi= self.construct_phi(X)
        if self.regularization>0:
            reg_matrix= self.regularization* np.eye(phi.shape[1])
            self.weights= np.linalg.inv(phi.T@phi + reg_matrix) @ phi.T @ y
        else:
            self.weights, _, _, _= lstsq(phi,y)
    
    def predict(self,X):
        phi= self.construct_phi(X)
        return phi @ self.weights

X= np.array([[0,0],[0,1],[1,0],[1,1]])
y= np.array([0,1,1,0])
centers= X.copy()
sigma=1.0
rbf_net= RBFNetwork(centers, sigma, regularization=0)
rbf_net.train(X,y)
y_pred= rbf_net.predict(X)
print(f"The predictions without regularization: {y_pred}")

regularization= 0.1
rbf_net_reg= RBFNetwork(centers, sigma, regularization)
rbf_net_reg.train(X,y)
y_pred_reg= rbf_net_reg.predict(X)
print(f"The predictions with regularization: {y_pred_reg}")

plt.figure(figsize=(10,5))
plt.xlabel("Data points")
plt.ylabel("Output")
plt.plot(range(len(y)), y, 'o-',label='True predictions')
plt.plot(range(len(y_pred)), y_pred, 's--',label='Predictions with Regularization')
plt.plot(range(len(y_pred_reg)), y_pred_reg, 'x--',label='Predictions without regularization')
plt.legend()
plt.title("Effect of regularization on XOR")
plt.grid()
plt.show()