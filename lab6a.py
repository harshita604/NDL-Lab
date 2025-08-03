import numpy as np
import matplotlib.pyplot as plt

class SelfOrganizingMap:
    def __init__(self, grid_size, input_dims, l_r=0.1, epochs=100, radius=None):
        self.grid_size= grid_size
        self.input_dims= input_dims
        self.radius= radius if radius else max(grid_size)/2
        self.l_r= l_r
        self.epochs= epochs
        self.weights= np.random.rand(grid_size[0], grid_size[1], input_dims)
    
    def find_bmu(self, sample):
        distance= np.linalg.norm(self.weights-sample, axis=2)
        return np.unravel_index(np.argmin(distance), distance.shape)
    
    def update_weights(self, bmu, sample, iteration):
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                dist_to_bmu= np.linalg.norm(np.array([i,j])- np.array(bmu))
                if dist_to_bmu<= self.radius:
                    influence= np.exp(-dist_to_bmu**2/(2*(self.radius**2)))
                    self.weights[i,j] += self.l_r*influence*(sample-self.weights[i,j])
    
    def train(self,data):
        for epoch in range(self.epochs):
            for sample in data:
                bmu= self.find_bmu(sample)
                self.update_weights(bmu, sample, epoch)
            self.l_r *=0.995
            self.radius*= 0.995
    
    def visualise(self):
        plt.imshow(self.weights.reshape(self.grid_size[0], self.grid_size[1], self.input_dims))
        plt.title("Self Organzing Map")
        plt.show()

data= np.random.rand(100,3)
input_dims=3
grid_size=(10,10)
epochs=50
som= SelfOrganizingMap(grid_size=grid_size, input_dims=input_dims, epochs=epochs)
som.train(data)
som.visualise()