import numpy as np
import matplotlib.pyplot as plt

def generate_data(samples=1000):
    mean=[0,0]
    cov=[[3,2],[2,2]]
    data= np.random.multivariate_normal(mean, cov, samples)
    return data

class LinearHebbian:
    def __init__(self, input_dim,l_r=0.01):
        self.weights= np.random.rand(input_dim)
        self.l_r= l_r
    
    def learn_hebbian(self,data, epochs=1):
        for epoch in range(epochs):
            for x in data:
                x=x.reshape(-1)
                y=np.dot(self.weights, x)
                self.weights += self.l_r* y* x
    
    def get_weights(self):
        return self.weights

data= generate_data(samples=1000)
input_dim= data.shape[1]
epochs=10
l_r=0.01
neuron= LinearHebbian(input_dim=input_dim, l_r=l_r)
neuron.learn_hebbian(data=data, epochs=epochs)
weights_hebbian= neuron.get_weights()
normalized_weights_hebbian= weights_hebbian/np.linalg.norm(weights_hebbian)

from sklearn.decomposition import PCA
pca= PCA(n_components=1)
pca.fit(data)
pca_component= pca.components_[0]
normalized_pca_component= pca_component/np.linalg.norm(pca_component)
print(normalized_weights_hebbian)
print(normalized_pca_component)
plt.scatter(data[:,0], data[:,1], alpha=0.3, label='input Data')
plt.quiver(0,0, normalized_weights_hebbian[0], normalized_weights_hebbian[1], scale=3, color='r', label='Hebbian Direction')
plt.quiver(0,0, normalized_pca_component[0], normalized_pca_component[1], scale=3, color='g', label='PCA Direction') 
plt.legend()
plt.xlabel('X1') 
plt.ylabel('X2')
plt.title("Hebbian learning vs PCA")
plt.show()